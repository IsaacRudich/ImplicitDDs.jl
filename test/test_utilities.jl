using JuMP
using HiGHS
using MathOptInterface
const MOI = MathOptInterface

# Use Main.ImplicitDDs when included directly, or ImplicitDDs when used as package
const IDD = isdefined(Main, :ImplicitDDs) ? Main.ImplicitDDs : (@eval using ImplicitDDs; ImplicitDDs)

"""
    compare_solvers(model::JuMP.Model; verbose=false, optimizer_attrs=Dict())

Compare ImplicitDDs Optimizer against HiGHS on a given model.

# Arguments
- `model::JuMP.Model`: JuMP model to solve (will be modified by setting optimizer)
- `verbose::Bool`: Print solver output and results (default: false)
- `optimizer_attrs::Dict`: Attributes to set on ImplicitDDs optimizer (e.g., `Dict("relaxed_w" => 5000)`)

# Returns
NamedTuple with fields:
- `passed::Bool`: Whether the solvers agreed
- `status::Symbol`: `:match`, `:objective_mismatch`, `:both_infeasible`, or `:feasibility_mismatch`
- `highs_status`: HiGHS termination status
- `highs_objective`: HiGHS objective value (NaN if not optimal)
- `implicitdds_status`: ImplicitDDs termination status
- `implicitdds_objective`: ImplicitDDs objective value (NaN if not optimal)
"""
function compare_solvers(model::JuMP.Model; verbose=false, optimizer_attrs=Dict())
    # Solve with HiGHS
    set_optimizer(model, HiGHS.Optimizer)
    !verbose && set_silent(model)
    optimize!(model)
    highs_status = termination_status(model)
    highs_obj = highs_status == MOI.OPTIMAL ? objective_value(model) : NaN

    if verbose
        println("HiGHS Status: ", highs_status)
        println("HiGHS Objective: ", highs_obj)
    end

    # Solve with ImplicitDDs
    set_optimizer(model, IDD.Optimizer)
    !verbose && set_silent(model)
    for (attr, val) in optimizer_attrs
        set_optimizer_attribute(model, attr, val)
    end
    optimize!(model)

    dd_status = termination_status(model)
    dd_obj = dd_status == MOI.OPTIMAL ? objective_value(model) : NaN
    dd_feasible = dd_status in (MOI.OPTIMAL, MOI.TIME_LIMIT) && primal_status(model) == MOI.FEASIBLE_POINT

    # Save ImplicitDDs solution for potential verification
    dd_solution = dd_feasible ? Dict(v => value(v) for v in all_variables(model)) : nothing

    if verbose
        println("ImplicitDDs Status: ", dd_status)
        println("ImplicitDDs Objective: ", dd_obj)
    end

    # Compare results
    if dd_feasible && highs_status == MOI.OPTIMAL
        obj_match = isapprox(highs_obj, dd_obj, rtol=1e-3)
        if obj_match
            passed = true
            status = :match
        else
            # Objective mismatch - determine which solver is wrong
            sense = objective_sense(model)
            highs_is_better = (sense == MOI.MIN_SENSE && highs_obj < dd_obj) ||
                              (sense == MOI.MAX_SENSE && highs_obj > dd_obj)

            if highs_is_better
                # ImplicitDDs found a suboptimal but valid solution
                passed = false
                status = :implicitdds_suboptimal
            else
                # ImplicitDDs claims better - verify solution is feasible
                set_optimizer(model, HiGHS.Optimizer)
                !verbose && set_silent(model)
                for (v, val) in dd_solution
                    fix(v, val, force=true)
                end
                optimize!(model)
                verification_status = termination_status(model)

                # Unfix variables for future use
                for v in all_variables(model)
                    unfix(v)
                end

                if verification_status == MOI.OPTIMAL
                    # ImplicitDDs solution is valid and better 
                    passed = true
                    status = :highs_suboptimal
                else
                    # ImplicitDDs returned an infeasible solution
                    passed = false
                    status = :implicitdds_invalid
                end
            end
        end
    elseif !dd_feasible && highs_status != MOI.OPTIMAL
        passed = true
        status = :both_infeasible
    elseif dd_feasible && highs_status != MOI.OPTIMAL
        # ImplicitDDs found a solution but HiGHS says infeasible - verify DD solution
        set_optimizer(model, HiGHS.Optimizer)
        !verbose && set_silent(model)
        for (v, val) in dd_solution
            fix(v, val, force=true)
        end
        optimize!(model)
        verification_status = termination_status(model)

        # Unfix variables for future use
        for v in all_variables(model)
            unfix(v)
        end

        if verification_status == MOI.OPTIMAL
            # ImplicitDDs solution is valid, HiGHS was wrong about infeasibility
            passed = true
            status = :highs_false_infeasible
        else
            # ImplicitDDs returned an infeasible solution
            passed = false
            status = :implicitdds_invalid
        end
    else
        # HiGHS found optimal but ImplicitDDs says infeasible
        passed = false
        status = :implicitdds_infeasible_highs_feasible
    end

    return (
        passed = passed,
        status = status,
        highs_status = highs_status,
        highs_objective = highs_obj,
        implicitdds_status = dd_status,
        implicitdds_objective = dd_obj,
    )
end


"""
    test_random_mip(; verbose=false, optimizer_attrs=Dict(), problem_kwargs...)

Generate a random MIP and validate ImplicitDDs against HiGHS.

# Arguments
- `verbose::Bool`: Print solver output (default: false)
- `optimizer_attrs::Dict`: Attributes for ImplicitDDs optimizer
- `problem_kwargs...`: Passed to generate_random_mip (e.g., n_int=10, n_cont=5, m=8, seed=42)

# Returns
Same NamedTuple as compare_solvers
"""
function test_random_mip(; verbose=false, optimizer_attrs=Dict(), problem_kwargs...)
    model = IDD.generate_random_mip(; problem_kwargs...)
    return compare_solvers(model; verbose=verbose, optimizer_attrs=optimizer_attrs)
end


"""
    run_validation_suite(n_tests=10; verbose=false, optimizer_attrs=Dict(), problem_kwargs...)

Run multiple random MIP tests and report summary.

# Arguments
- `n_tests::Int`: Number of random tests to run (default: 10)
- `verbose::Bool`: Print per-test output (default: false)
- `optimizer_attrs::Dict`: Attributes for ImplicitDDs optimizer
- `problem_kwargs...`: Base parameters for generate_random_mip (seed will be varied)

# Returns
NamedTuple with:
- `passed::Int`: Number of tests passed
- `failed::Int`: Number of tests failed
- `failed_seeds::Vector{Int}`: Seeds of failed tests
"""
function run_validation_suite(n_tests::Int=10; verbose=false, optimizer_attrs=Dict(), problem_kwargs...)
    failed_seeds = Int[]
    passed = 0
    failed = 0

    for i in 1:n_tests
        # Progress indicator every 100 tests
        if i % 100 == 0
            println("Progress: $i/$n_tests tests completed...")
        end

        # Use test index as seed for reproducibility
        result = test_random_mip(; verbose=verbose, optimizer_attrs=optimizer_attrs, seed=i, problem_kwargs...)

        if result.passed
            passed += 1
        else
            failed += 1
            push!(failed_seeds, i)
            println("Test $i FAILED: $(result.status)")
        end
    end

    println("\nValidation Suite: $passed/$n_tests passed")

    return (passed=passed, failed=failed, failed_seeds=failed_seeds)
end


"""
    run_serial_validation(n_tests::Int=5000; verbose=false)

Run serial validation suite (small width, no parallelism).
Uses: width=4, num_LPs=1, 10 int vars, 5 cont vars, 5 constraints, int bounds [-1,2]
"""
function run_serial_validation(n_tests::Int=5000; verbose=false)
    run_validation_suite(n_tests;
        verbose=verbose,
        optimizer_attrs=Dict(
            "relaxed_w" => 4,
            "restricted_w" => 4,
            "num_LPs_to_run" => 1,
            "parallel_processing" => false
        ),
        n_int=10, n_cont=5, m=5,
        density=1.0, int_lb_range=(-1,-1), int_ub_range=(2,2), cont_domain_size=4.0
    )
end


"""
    run_parallel_validation(n_tests::Int=5000; verbose=false)

Run parallel validation suite (larger width, parallelism enabled).
Uses: width=10, num_LPs=3, 12 int vars, 5 cont vars, 5 constraints, int bounds [-2,2]
"""
function run_parallel_validation(n_tests::Int=5000; verbose=false)
    run_validation_suite(n_tests;
        verbose=verbose,
        optimizer_attrs=Dict(
            "relaxed_w" => 10,
            "restricted_w" => 10,
            "num_LPs_to_run" => 3,
            "parallel_processing" => true
        ),
        n_int=12, n_cont=5, m=5,
        density=1.0, int_lb_range=(-2,-2), int_ub_range=(2,2), cont_domain_size=4.0
    )
end

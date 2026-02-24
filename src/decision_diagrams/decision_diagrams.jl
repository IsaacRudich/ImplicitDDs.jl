include("bnb_and_pnb/bnb_and_pnb.jl")
include("pruning/pruning.jl")
include("relaxed_dds/relaxed_dds.jl")
include("restricted_dds/restricted_dds.jl")
include("bnb_and_pnb/SolveResult.jl")
include("bnb_and_pnb/bnb_subroutines.jl")
include("bnb_and_pnb/branch_and_bound.jl")


function test_dds(;
    # Solver parameters
    relaxed_w::Int = 10000, restricted_w::Int = 10000, num_LPs_to_run = 100,
    solution_print::Bool = false,  bounds_print::Bool = true, suppress_all_prints::Bool = false, debug_mode::Bool = false,
    numerical_precision::DataType = Float32,
    log_file_path::Union{String, Nothing} = nothing, wait_to_write_solutions::Bool = true,
    timer_outputs::Bool = false, parallel_processing::Bool = false,

    # Problem generation parameters
    num_int_vars::Int = 10, num_cont_vars::Int = 20, n_constraints::Int = 10, test_seed::Union{Int, Nothing} = 1,
    density::Float64 = 0.5,
    int_domain_size::Int = 10, cont_domain_size::Float64 = 10.0,
    int_lb_range::Union{Tuple{Int, Int}, Nothing} = nothing,
    int_ub_range::Union{Tuple{Int, Int}, Nothing} = nothing,
    cont_lb_range::Union{Tuple{Float64, Float64}, Nothing} = nothing,
    cont_ub_range::Union{Tuple{Float64, Float64}, Nothing} = nothing,
    obj_density::Float64 = 1.0,
    obj_coeff_range::Tuple{Float64, Float64} = (-100.0, 100.0),
    obj_coeff_dist::Symbol = :uniform,
    rhs_range::Tuple{Float64, Float64} = (-100.0, 100.0),
    coeff_distribution::Symbol = :uniform,
    coeff_range::Tuple{Float64, Float64} = (-10.0, 10.0),
    integer_coefficients::Bool = false,
    mip_precision::Int = 2,
    coefficient_scale::Float64 = 1.0,
    block_structure::Bool = false,
    num_blocks::Int = 1,
)
    
    #TODO let user choose LP solver

    model = generate_random_mip(
        n_int = num_int_vars,
        n_cont = num_cont_vars,
        m = n_constraints,
        density = density,
        int_domain_size = int_domain_size,
        cont_domain_size = cont_domain_size,
        int_lb_range = int_lb_range,
        int_ub_range = int_ub_range,
        cont_lb_range = cont_lb_range,
        cont_ub_range = cont_ub_range,
        obj_density = obj_density,
        obj_coeff_range = obj_coeff_range,
        obj_coeff_dist = obj_coeff_dist,
        rhs_range = rhs_range,
        coeff_distribution = coeff_distribution,
        coeff_range = coeff_range,
        integer_coefficients = integer_coefficients,
        precision = mip_precision,
        coefficient_scale = coefficient_scale,
        block_structure = block_structure,
        num_blocks = num_blocks,
        seed = test_seed
    )

    optimize!(model)
    opt_val = objective_value(model)
    status = termination_status(model)
    sense = objective_sense(model)
    println("Objective Sense: ", sense)
    println("HiGHS Objective Value: ", opt_val, "\n")
    println("HiGHS Termination Status: ", status, "\n")
    println("HiGHS Iteration Count: ",MOI.get(model, MOI.SimplexIterations()))


    result = solve_mip(
        model;
        relaxed_w = relaxed_w,
        restricted_w = restricted_w,
        num_LPs_to_run = num_LPs_to_run,
        solution_print = solution_print,
        bounds_print = bounds_print,
        suppress_all_prints = suppress_all_prints,
        debug_mode = debug_mode,
        numerical_precision = numerical_precision,
        log_file_path = log_file_path,
        wait_to_write_solutions = wait_to_write_solutions,
        timer_outputs = timer_outputs,
        parallel_processing = parallel_processing
    )

    feasibility_disagreement  = false
    if result.is_feasible && status == MOI.OPTIMAL
        if ImplicitDDs.eps_equals(opt_val, result.objective_value, eps= 1e-3)
            is_disagreement = false
        else
            is_disagreement = true
        end
    elseif (result.is_feasible && status != MOI.OPTIMAL) || (!result.is_feasible && status == MOI.OPTIMAL)
        feasibility_disagreement  = true
        is_disagreement = false
    elseif !result.is_feasible && status != MOI.OPTIMAL
        # Both solvers agree the problem is infeasible - no disagreement
        is_disagreement = false
    end

    is_suboptimal = true
    if is_disagreement
        if opt_val < result.objective_value
            # println("Solver found suboptimal solution.")
        else
            for i in 1:num_int_vars
                fix(all_variables(model)[i], result.bks_int[i], force=true)
            end
            for i in 1:num_cont_vars
                fix(all_variables(model)[num_int_vars+i], result.bks_cont[i], force=true)
            end

            optimize!(model)
            verification_status = termination_status(model)

            if verification_status == MOI.OPTIMAL
                is_suboptimal = false
            end
        end
    end

    return is_disagreement, is_suboptimal, feasibility_disagreement
end
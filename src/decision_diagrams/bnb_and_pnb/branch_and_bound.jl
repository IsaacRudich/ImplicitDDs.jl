"""
    solve_mip(model::JuMP.Model; kwargs...)

Solve a mixed-integer programming problem using decision diagram-based branch-and-bound.

This is the main solver entry point. At each branch-and-bound node it uses restricted decision diagrams to find feasible
solutions, and relaxed decision diagrams to compute lower bounds.
to prove optimality.

# Arguments

## Required
- `model::JuMP.Model`: A JuMP model containing your MIP formulation

## Decision Diagram Parameters

- `relaxed_w::Int = 1000`: Maximum width for relaxed decision diagrams
  - **Larger values**: Tighter lower bounds, slower per-iteration, potentially fewer B&B nodes
  - **Smaller values**: Faster per-iteration, weaker bounds, potentially more B&B nodes
  - **How to choose**: Make it as large as you can without noticably slowing down the solver. 10000 is a reasonable starting poitn for large problems. 

- `restricted_w::Int = 1000`: Maximum width for restricted decision diagrams
  - **Larger values**: Better chance of finding good solutions early, but slower
  - **Smaller values**: Faster construction, but may find worse initial solutions
  - **How to choose**: Usually same as `relaxed_w` or smaller. Larger may be helpful if finding any feasible solution is hard.

- `num_LPs_to_run::Int = 100`: Maximum number of LP subproblems to solve per restricted DD
  - Each LP optimizes continuous variables for a terminal node in the restricted DD
  - **Trade-off**: More LPs = better solutions but MUCH slower restricted DD phase
  - **How to choose**: Ongoing research...

## Performance Parameters

- `parallel_processing::Bool = true`: Use multi-threading for branch-and-bound

- `numerical_precision::DataType = Float32`: Precision for all floats used in the solver.
  - **Float32**: Use for most problems, if precision isn't critical consider lower
  - **Float64**: Slower, more memory (~8 bytes/float), but higher numerical accuracy
  - **Important**: Must match the precision used in `precompile_solver()` to avoid JIT compilation overhead

## Output Control

- `solution_print::Bool = false`: Print full solution vectors when found
  - Shows complete variable assignments for debugging
  - Can be very verbose for large problems

- `bounds_print::Bool = true`: Print lower/upper bounds and optimality gap as they improve
  - Shows solver progress through the B&B tree
  - Recommended to keep enabled unless running batch experiments

- `suppress_all_prints::Bool = false`: Disable all console output
  - Overrides `solution_print` and `bounds_print`
  - Use for batch runs or when integrating into other systems
  - Does not override debug_mode or timer_outputs

- `debug_mode::Bool = false`: Print detailed internal solver state
  - Shows DD construction details, node processing, etc.
  - Very verbose - only use when debugging solver issues

## Logging Parameters

- `log_file_path::Union{String, Nothing} = nothing`: Path to JSON log file
  - If provided, writes solutions and bounds to file in JSON format
  - Use for experiment tracking and post-processing
  - Example: `"output/my_problem.json"`

- `wait_to_write_solutions::Bool = false`: Batch log writes for performance
  - `true`: Accumulate logs in memory, write once at end (faster)
  - `false`: Write each log entry immediately (safer if solver crashes)

- `timer_outputs::Bool = false`: Generate detailed timing breakdown
  - Shows time spent in each solver phase (preprocessing, DD construction, LPs, etc.)
  - Useful for performance profiling

## Other Parameters

- `time_limit::Union{<:Real, Nothing} = nothing`: Time limit in minutes for solve phase only, excluding model conversion (nothing if no time limit)
  - **Purpose**: Gracefully terminate solver after specified time, returning best solution found
  - **Behavior**: Solver checks time budget at decision diagram construction and B&B loop boundaries
  - **Return**: Returns best feasible solution found before timeout (may be suboptimal)

- `custom_variable_order::Union{Vector{VariableRef}, Nothing} = nothing`: Custom ordering for integer variables
  - **Purpose**: Override METIS-based variable ordering with a problem-specific ordering
  - **Format**: Vector of `VariableRef`s containing all integer variables from the model in desired processing order
  - **Example**: For TSP-TW, order arc variables by time window deadlines for natural temporal ordering
  - **Validation**: Must contain exactly the same integer variables as the model (checked at runtime)
  - **Default**: `nothing` uses adaptive METIS-based ordering

# Returns

Returns a tuple `(is_feasible, bks_int, bks_cont, bkv)`:

- `is_feasible::Bool`: Whether a feasible solution was found
- `bks_int`: Best integer variable values (in original model order)
  - If feasible: `BitVector` for binary problems, `Vector{Int8/Int16/Int32/Int64}` otherwise
  - If infeasible: Empty `Int[]`
- `bks_cont::Vector{numerical_precision}`: Best continuous variable values (in original model order)
  - If infeasible: Empty `Float64[]`
- `bkv`: Best objective value found (in original objective sense)
  - If infeasible: `0`

# Requirements

1. **Bounded variables**: All variables must have finite bounds. Solver will attempt OBBT (Optimization-Based Bound Tightening) on unbounded variables.

2. **Linear constraints**: Only linear constraints are supported

3. **Supported variable types**:
   - Integer variables (including binary)
   - Continuous variables
   - Mixed-integer linear constraints

4. **Objective**: Linear minimization or maximization

5. **Optimizer required**: Model must have an LP-capable optimizer attached (e.g., HiGHS, Gurobi, CPLEX) for solving continuous variable subproblems and bound tightening

# Algorithm Overview

1. **Preprocessing**:
   - Convert model to internal format with variables reordered internally
   - Run OBBT on unbounded variables
   - Compute initial infimum gaps and run FBBT (Feasibility-Based Bound Tightening)

2. **Initial Restricted DD**: Find initial feasible solution

3. **Initial Relaxed DD**: Compute lower bound and generate initial B&B nodes

4. **Branch-and-Bound Loop**:
   - For each B&B node: Run FBBT, build restricted DD (find solutions), build relaxed DD (get bounds, create child nodes)
   - Prune nodes with bounds worse than best known solution
   - Continue until all nodes processed or optimality proven

5. **Continuous Variable Refinement**: Resolve LP with integer variables fixed to get exact continuous values

# Performance Tips

1. **Precompile first**: Call `precompile_solver()` before your first solve to avoid JIT overhead, see precompile_solver() documentation for more information

2. **Match precision**: Use same `numerical_precision` in `precompile_solver()` and `solve_mip()`

3. **Parallel for large problems**: Enable `parallel_processing` and run `julia -t N` with N threads

4. **Tune widths**: Start with 1000, increase if bounds aren't tight enough, decrease if construction is too slow

5. **Monitor gaps**: Watch bound logs to see if solver is making progress

6. **Binary variables**: Solver automatically uses BitVectors for memory efficiency on binary problems
"""
function solve_mip(
    model::JuMP.Model;
    relaxed_w::Int = 1000,
    restricted_w::Int = 1000,
    num_LPs_to_run::Int = 100,
    parallel_processing::Bool = true,
    solution_print::Bool = false,
    bounds_print::Bool = true,
    suppress_all_prints::Bool = false,
    debug_mode::Bool = false,
    numerical_precision::DataType = Float32,
    log_file_path::Union{String, Nothing} = nothing,
    wait_to_write_solutions::Bool = false,
    timer_outputs::Bool = false,
    time_limit::Union{<:Real, Nothing} = nothing,
    custom_variable_order::Union{Vector{VariableRef}, Nothing} = nothing,
    _force_parallel_precompile::Bool = false
)

    #for logging only
    true_start = time()
    if wait_to_write_solutions
        runtime_log_file_path = nothing
    else
        runtime_log_file_path = log_file_path
    end

    #for threading
    if _force_parallel_precompile
        num_workers = 2
    elseif parallel_processing
        num_workers = min(Threads.nthreads(), Sys.CPU_THREADS)
    else
        num_workers = 1
    end


    model, lbs_int, ubs_int, lbs_cont, ubs_cont, num_int_vars, num_cont_vars, num_constraints, int_obj_coeffs, cont_obj_coeffs, obj_const, coefficient_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, coefficient_matrix_cont_cols, cont_var_to_pos_rows, cont_var_to_neg_rows, cont_var_to_zero_rows, coefficient_matrix_rhs_vector, int_var_refs, cont_var_refs, sense_multiplier, inverse_order, conversion_time = setup_from_JuMP_model(model, numerical_precision, suppress_all_prints, debug_mode, custom_variable_order)

    time_limit_start = time()
    time_limit_seconds = isnothing(time_limit) ? nothing : Float64(time_limit * 60)


    start_time = time()
    if !suppress_all_prints
        print("Preallocating Memory ... ")
    end
    
    # Ensure widths are at least as large as maximum domain size
    max_domain_size = maximum(ubs_int .- lbs_int .+ 1)
    relaxed_w = max(relaxed_w, max_domain_size)
    restricted_w = max(restricted_w, max_domain_size)
    w = max(relaxed_w, restricted_w)  # Use max for preallocation

    if debug_mode
        println("\n","Adjusted widths: relaxed_w=", relaxed_w, ", restricted_w=", restricted_w, " (max domain size: ", max_domain_size, ")")
    end
    
    # Pick a type for the intgers
    min_int_bound = minimum(lbs_int)
    max_int_bound = maximum(ubs_int)

    if min_int_bound >= 0 && max_int_bound <= 1
        all_vars_binary = true
        int_data_type = Bool
    else
        all_vars_binary = false
        if min_int_bound >= typemin(Int8) && max_int_bound <= typemax(Int8)
            int_data_type = Int8
        elseif min_int_bound >= typemin(Int16) && max_int_bound <= typemax(Int16)
            int_data_type = Int16
        elseif min_int_bound >= typemin(Int32) && max_int_bound <= typemax(Int32)
            int_data_type = Int32
        else
            int_data_type = Int64
        end
    end

    # TYPE VARIES BASED ON all_vars_binary
    if all_vars_binary
        lbs_int = BitVector(lbs_int)
        ubs_int = BitVector(ubs_int)
    else
        lbs_int = Vector{int_data_type}(lbs_int)
        ubs_int = Vector{int_data_type}(ubs_int)
    end

    #for parallel processing
    dd_workspaces = Vector{DDWorkspace{int_data_type, numerical_precision}}()
    for i in 1:num_workers
        push!(dd_workspaces, preallocate_dd_memory(num_int_vars, num_cont_vars, num_constraints, lbs_int, ubs_int, int_data_type, all_vars_binary, restricted_w, relaxed_w, w, max_domain_size, numerical_precision, coefficient_matrix_int_cols))
    end
    dd_ws = dd_workspaces[1]

    # Compute and store total workspace memory usage
    dd_ws.timing_stats.workspace_memory_bytes = compute_workspace_memory(dd_workspaces)
    #for logging only
    logs = Vector{Dict{String, Any}}()
    if all_vars_binary
        editable_int_solution = BitVector(undef, num_int_vars)
    else
        editable_int_solution = Vector{int_data_type}(undef, num_int_vars)
    end
    
    if !suppress_all_prints
        println("Finished")
        # println("Preallocating Memory took: ", round(preallocating_memory_time, digits=3), " seconds", "\n")
    end
    end_time = time()
    preallocating_memory_time = end_time - start_time




    start_time = time()
    if !suppress_all_prints
        print("Running OBBT on any unbounded variables ... ")
    end
    #run obbt
    has_bounded_variables = run_obbt!(ubs_int, lbs_int, ubs_cont, lbs_cont, model, int_var_refs, cont_var_refs)
    if !has_bounded_variables
        if !suppress_all_prints
                println("\n","This solver requires bounded variables, and the solver was unable to bound all the variables.")
        end
        OBBT_time = time() - start_time
        finalize_and_log_timing!(
            timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
            0.0, 0.0, conversion_time, preallocating_memory_time,
            OBBT_time, 0.0, 0.0, 0.0, true_start, logs, log_file_path
        )
        return return_unbounded_error(int_data_type, numerical_precision, time() - true_start)
    end

    if !suppress_all_prints
        println("Finished")
        # println("OBBT took: ", round(OBBT_time, digits=3), " seconds", "\n")
    end
    end_time = time()
    OBBT_time = end_time - start_time




    solve_start = time()
    start_time = time()
    #Initial Calculations
    if !suppress_all_prints
        print("Preprocessing ... ")
    end

    # Precompute inverse coefficients (once at startup - never changes)
    @inbounds for var_idx in 1:num_int_vars
        @inbounds @simd for row in 1:num_constraints
            coeff_row = coefficient_matrix_int_cols[row, var_idx]
            dd_ws.inv_coeff[row, var_idx] = (coeff_row != 0) ? (1.0 / coeff_row) : 0.0
        end
    end

    # Precompute objective contributions for full global domain (once at startup)
    @inbounds for var_idx in 1:num_int_vars
        lb = lbs_int[var_idx]
        ub = ubs_int[var_idx]
        coeff = int_obj_coeffs[var_idx]
        @inbounds @simd for local_idx in 1:(ub - lb + 1)
            val = lb + local_idx - 1
            dd_ws.coeff_times_val[local_idx, var_idx] = coeff * val
        end
    end

    # Precompute inverse objective coefficients (once at startup - for cap calculations)
    @inbounds @simd for var_idx in 1:num_int_vars
        coeff = int_obj_coeffs[var_idx]
        dd_ws.inv_obj_coeffs[var_idx] = (coeff != 0) ? (1.0 / coeff) : 0.0
    end

    # Copy precomputed values to all workspaces
    @inbounds for i in 2:length(dd_workspaces)
        dd_workspaces[i].inv_coeff .= dd_ws.inv_coeff
        dd_workspaces[i].coeff_times_val .= dd_ws.coeff_times_val
        dd_workspaces[i].inv_obj_coeffs .= dd_ws.inv_obj_coeffs
    end

    # initial infimum gap values for each constraint row in a MIP model, normalized for <= constraints.
    infimum_gaps, cont_inf_gap_ctrbtns = compute_initial_infimum_gaps(
        coefficient_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows, lbs_int, ubs_int,
        coefficient_matrix_cont_cols, cont_var_to_pos_rows, cont_var_to_neg_rows, lbs_cont, ubs_cont,
        coefficient_matrix_rhs_vector
    )

    if debug_mode
        println("\n", "Initial Infiumum Gaps: ", infimum_gaps)
        println("Continuous Var Gap Contributions: ", cont_inf_gap_ctrbtns)
    end

    # Save original bounds before any FBBT for consistent coeff_times_val/gap_adjustments indexing
    for workspace in dd_workspaces
        workspace.original_lbs_int .= lbs_int
        workspace.original_ubs_int .= ubs_int
    end

    fbbt_yields_feasible_bounds, tmp1, tmp2 = fbbt_bound_tightening!(
        ubs_int, lbs_int, ubs_cont, lbs_cont,
        coefficient_matrix_int_cols, coefficient_matrix_cont_cols,
        int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, cont_var_to_pos_rows, cont_var_to_neg_rows,
        infimum_gaps, cont_inf_gap_ctrbtns, num_int_vars, num_cont_vars, dd_ws.gap_adjustments,
        max_iterations = 100
    )

    if !fbbt_yields_feasible_bounds
        if !suppress_all_prints
            println("\n", "Problem Infeasible: because FBBT yields infeasible variable bounds")
        end
        preprocessing_time = time() - start_time
        finalize_and_log_timing!(
            timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
            0.0, 0.0, conversion_time, preallocating_memory_time,
            OBBT_time, preprocessing_time, 0.0, 0.0, true_start, logs, log_file_path
        )
        return return_infeasible(int_data_type, numerical_precision, 0, time() - true_start, true, false)
    end

    #setup rough bounding
    compute_rough_bounding_vector!(dd_ws.rough_bounds_int, int_obj_coeffs, num_int_vars, lbs_int, ubs_int)
    rough_bounds_cont_val = compute_total_rough_bound(cont_obj_coeffs, num_cont_vars, lbs_cont, ubs_cont)

    #create LP models for subproblems (one per worker for thread safety)
    lp_params = Vector{Tuple{JuMP.Model, Vector{JuMP.VariableRef}, Vector{JuMP.ConstraintRef}}}()
    for i in 1:num_workers
        lp_sub_model, lp_vars, lp_constraint_refs = create_LP_subproblem_model(coefficient_matrix_cont_cols, cont_obj_coeffs, lbs_cont, ubs_cont, infimum_gaps, cont_inf_gap_ctrbtns, num_cont_vars, num_constraints)
        push!(lp_params, (lp_sub_model, lp_vars, lp_constraint_refs))
    end
    
    # Use first LP model for initial checks
    lp_sub_model, lp_vars, lp_constraint_refs = lp_params[1]

    if num_cont_vars > 0
        optimize!(lp_sub_model)
        # println("LP Status: ", termination_status(lp_sub_model))
        if termination_status(lp_sub_model) == MOI.OPTIMAL
            lp_obj_val = objective_value(lp_sub_model)
            if debug_mode
                println("\n","Initial LP Subproblem Optimal value: ", lp_obj_val)
            end
            if lp_obj_val > rough_bounds_cont_val
                rough_bounds_cont_val = numerical_precision(lp_obj_val)
            end
            # set_attribute(lp_sub_model, "presolve", "off")
        else
            if !suppress_all_prints
                println("\n", "Problem Infeasible: because LP Subproblem Infeasible")
            end
            preprocessing_time = time() - start_time
            finalize_and_log_timing!(
                timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                0.0, 0.0, conversion_time, preallocating_memory_time,
                OBBT_time, preprocessing_time, 0.0, 0.0, true_start, logs, log_file_path
            )
            return return_infeasible(int_data_type, numerical_precision, 0, time() - true_start, true, false)
        end
    end

    if !suppress_all_prints
        println("Finished")
        # println("Preprocessing took: ", round(preprocessing_time, digits=3), " seconds", "\n")
    end

    end_time = time()
    preprocessing_time = end_time - start_time





    start_time = time()
    if !suppress_all_prints
        print("\n","Running initial restricted DD ... ")
    end

    bkv = typemax(numerical_precision)

    initial_dd_width = (num_workers > 1) ? max(max_domain_size, min(10000,restricted_w)) : restricted_w
    initial_dd_width = min(initial_dd_width, restricted_w)

    if initial_dd_width >= 100000
        bkv, is_exact, is_feasible_restricted = setup_and_run_restricted_dd!(
            dd_ws.rdd, bkv, dd_ws.bks_int, dd_ws.bks_cont,
            obj_const,
            int_var_to_pos_rows, int_var_to_neg_rows,
            ubs_int, lbs_int, dd_ws.original_lbs_int,
            num_int_vars, num_cont_vars, num_constraints,
            dd_ws.rough_bounds_int, rough_bounds_cont_val, infimum_gaps, cont_inf_gap_ctrbtns,
            dd_ws.coeff_times_val, dd_ws.gap_adjustments, dd_ws.inv_coeff, dd_ws.inv_obj_coeffs,
            lp_sub_model, lp_vars, lp_constraint_refs,
            dd_ws.bin_counts, dd_ws.restricted_bins_matrix, dd_ws.cumulative_bins, 10000, num_LPs_to_run,
            dd_ws.timing_stats,
            calculate_child_time_budget(time_limit_seconds, time_limit_start)
        )

        if is_feasible_restricted
            if is_exact
                # Fix integer variables and resolve LP for continuous variables
                bkv = resolve_continuous_variables!(model, dd_ws.bks_int, dd_ws.bks_cont, int_var_refs, cont_var_refs, num_int_vars, num_cont_vars, sense_multiplier, bkv, suppress_all_prints)

                log_solution(
                    dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                    solution_print, suppress_all_prints,
                    true,
                    log_file_path, solve_start, true_start,
                    logs, wait_to_write_solutions
                )

                initial_restricted_dd_time = time() - start_time
                finalize_and_log_timing!(
                    timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                    0.0, 0.0, conversion_time, preallocating_memory_time,
                    OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
                )
                return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, bkv, sense_multiplier, 1, time() - true_start, true, false)
            else
                log_solution(
                    dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                    solution_print, suppress_all_prints,
                    false,
                    runtime_log_file_path, solve_start, true_start,
                    logs, wait_to_write_solutions
                )
            end
        elseif is_exact
            if bkv == typemax(numerical_precision)
                # No solution found yet, truly infeasible
                if !suppress_all_prints
                    println("Problem Infeasible: by brute force")
                end
                initial_restricted_dd_time = time() - start_time
                finalize_and_log_timing!(
                    timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                    0.0, 0.0, conversion_time, preallocating_memory_time,
                    OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
                )
                return return_infeasible(int_data_type, numerical_precision, 1, time() - true_start, true, false)
            else
                # Solution was found in a previous pass and DD is exact - it's optimal
                bkv = resolve_continuous_variables!(model, dd_ws.bks_int, dd_ws.bks_cont, int_var_refs, cont_var_refs, num_int_vars, num_cont_vars, sense_multiplier, bkv, suppress_all_prints)

                log_solution(
                    dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                    solution_print, suppress_all_prints,
                    true,
                    log_file_path, solve_start, true_start,
                    logs, wait_to_write_solutions
                )

                initial_restricted_dd_time = time() - start_time
                finalize_and_log_timing!(
                    timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                    0.0, 0.0, conversion_time, preallocating_memory_time,
                    OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
                )
                return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, bkv, sense_multiplier, 1, time() - true_start, true, false)
            end
        end
    end

    bkv, is_exact, is_feasible_restricted = setup_and_run_restricted_dd!(
        dd_ws.rdd, bkv, dd_ws.bks_int, dd_ws.bks_cont,
        obj_const,
        int_var_to_pos_rows, int_var_to_neg_rows,
        ubs_int, lbs_int, dd_ws.original_lbs_int,
        num_int_vars, num_cont_vars, num_constraints,
        dd_ws.rough_bounds_int, rough_bounds_cont_val, infimum_gaps, cont_inf_gap_ctrbtns,
        dd_ws.coeff_times_val, dd_ws.gap_adjustments, dd_ws.inv_coeff, dd_ws.inv_obj_coeffs,
        lp_sub_model, lp_vars, lp_constraint_refs,
        dd_ws.bin_counts, dd_ws.restricted_bins_matrix, dd_ws.cumulative_bins, initial_dd_width, num_LPs_to_run,
        dd_ws.timing_stats,
        calculate_child_time_budget(time_limit_seconds, time_limit_start)
    )

    if !suppress_all_prints
        println("Finished")
        # println("Initial restricted DD took: ", round(initial_restricted_dd_time, digits=3), " seconds")
    end

    if is_feasible_restricted
        if is_exact
             # Fix integer variables and resolve LP for continuous variables
            bkv = resolve_continuous_variables!(model, dd_ws.bks_int, dd_ws.bks_cont, int_var_refs, cont_var_refs, num_int_vars, num_cont_vars, sense_multiplier, bkv, suppress_all_prints)

            log_solution(
                dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                solution_print, suppress_all_prints,
                true,
                log_file_path, solve_start, true_start,
                logs, wait_to_write_solutions
            )

            initial_restricted_dd_time = time() - start_time
            finalize_and_log_timing!(
                timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                0.0, 0.0, conversion_time, preallocating_memory_time,
                OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
            )
            return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, bkv, sense_multiplier, 1, time() - true_start, true, false)
        else
            log_solution(
                dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                solution_print, suppress_all_prints,
                false,
                runtime_log_file_path, solve_start, true_start,
                logs, wait_to_write_solutions
            )
        end
    elseif is_exact
        if bkv == typemax(numerical_precision)
            # No solution found yet, truly infeasible
            if !suppress_all_prints
                println("Problem Infeasible: by brute force")
            end
            initial_restricted_dd_time = time() - start_time
            finalize_and_log_timing!(
                timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                0.0, 0.0, conversion_time, preallocating_memory_time,
                OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
            )
            return return_infeasible(int_data_type, numerical_precision, 1, time() - true_start, true, false)
        else
            # Solution was found in a previous pass and DD is exact - it's optimal
            bkv = resolve_continuous_variables!(model, dd_ws.bks_int, dd_ws.bks_cont, int_var_refs, cont_var_refs, num_int_vars, num_cont_vars, sense_multiplier, bkv, suppress_all_prints)

            log_solution(
                dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                solution_print, suppress_all_prints,
                true,
                log_file_path, solve_start, true_start,
                logs, wait_to_write_solutions
            )

            initial_restricted_dd_time = time() - start_time
            finalize_and_log_timing!(
                timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                0.0, 0.0, conversion_time, preallocating_memory_time,
                OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
            )
            return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, bkv, sense_multiplier, 1, time() - true_start, true, false)
        end
    end
    if !suppress_all_prints
        println()
    end
    end_time = time()
    initial_restricted_dd_time = end_time - start_time

    # Check if restricted DD timed out
    if time_budget_exceeded(time_limit_seconds, time_limit_start)
        finalize_and_log_timing!(
            timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
            0.0, 0.0, conversion_time, preallocating_memory_time,
            OBBT_time, preprocessing_time, initial_restricted_dd_time, 0.0, true_start, logs, log_file_path
        )
        if is_feasible_restricted
            return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, typemin(numerical_precision), sense_multiplier, 1, time() - true_start, false, true)
        else
            return return_infeasible(int_data_type, numerical_precision, 1, time() - true_start, false, true)
        end
    end

    start_time = time()
    if !suppress_all_prints
        print("Running initial relaxed DD ... ")
    end
   
    # println("Creating Initial Relaxed DD")

    initial_dd_width = (num_workers > 1) ? max(max_domain_size, min(100,relaxed_w), num_workers) : relaxed_w
    initial_dd_width = min(initial_dd_width, relaxed_w)

    best_bound, bkv, is_feasible, bks_was_updated, dd_ws.extra_layer = setup_run_process_relaxed_dd!(
        int_obj_coeffs, obj_const, coefficient_matrix_int_cols, coefficient_matrix_rhs_vector, dd_ws.inv_coeff, dd_ws.coeff_times_val,
        int_var_to_pos_rows, int_var_to_neg_rows,
        ubs_int, lbs_int, dd_ws.original_lbs_int, bkv, dd_ws.bks_int, dd_ws.bks_cont,
        num_int_vars, num_cont_vars, num_constraints,
        dd_ws.rough_bounds_int, rough_bounds_cont_val,
        infimum_gaps, dd_ws.infimum_gap_matrices, dd_ws.node_matrix, dd_ws.extra_layer,
        dd_ws.ltr_matrix, dd_ws.ltt_matrix,
        dd_ws.lb_matrix, dd_ws.ub_matrix, dd_ws.low_indexes, dd_ws.high_indexes, dd_ws.rel_path, dd_ws.feasibility_accumulator,
        dd_ws.wrk_vec, cont_inf_gap_ctrbtns,
        dd_ws.arc_count_per_node, dd_ws.node_bin_counts, dd_ws.node_cumulative, dd_ws.global_lower, dd_ws.global_upper, dd_ws.bins_matrix,
        dd_ws.bnb_queue,  dd_ws.added_queue_nodes,
        lp_sub_model, lp_vars, lp_constraint_refs,
        initial_dd_width, dd_ws.timing_stats;
        debug_mode = debug_mode,
        time_remaining = calculate_child_time_budget(time_limit_seconds, time_limit_start)
    )

    if !suppress_all_prints
        println("Finished")
        # println("Initial relaxed DD took: ", round(initial_relaxed_dd_time, digits=3), " seconds")
    end

    if bks_was_updated
        log_solution(
            dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
            solution_print, suppress_all_prints,
            false, 
            runtime_log_file_path, solve_start, true_start,
            logs, wait_to_write_solutions
        )
    end
    if !is_feasible && !time_budget_exceeded(time_limit_seconds, time_limit_start)
        if is_feasible_restricted
            # Fix integer variables and resolve LP for continuous variables
            bkv = resolve_continuous_variables!(model, dd_ws.bks_int, dd_ws.bks_cont, int_var_refs, cont_var_refs, num_int_vars, num_cont_vars, sense_multiplier, bkv, suppress_all_prints)

            log_solution(
                dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
                solution_print, suppress_all_prints,
                true,
                log_file_path, solve_start, true_start,
                logs, wait_to_write_solutions
            )

            initial_relaxed_dd_time = time() - start_time
            finalize_and_log_timing!(
                timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                0.0, 0.0, conversion_time, preallocating_memory_time,
                OBBT_time, preprocessing_time, initial_restricted_dd_time, initial_relaxed_dd_time, true_start, logs, log_file_path
            )
            return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, bkv, sense_multiplier, 1, time() - true_start, true, false)
        else
            if !suppress_all_prints
                println("Problem Infeasible: by no feasible relaxation")
            end
            initial_relaxed_dd_time = time() - start_time
            finalize_and_log_timing!(
                timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
                0.0, 0.0, conversion_time, preallocating_memory_time,
                OBBT_time, preprocessing_time, initial_restricted_dd_time, initial_relaxed_dd_time, true_start, logs, log_file_path
            )
            return return_infeasible(int_data_type, numerical_precision, 1, time() - true_start, true, false)
        end
    else
        log_bounds(
            best_bound, bkv, sense_multiplier, length(dd_ws.bnb_queue),
            bounds_print, suppress_all_prints, runtime_log_file_path,
            solve_start, true_start,
            logs, wait_to_write_solutions
        )
    end
    end_time = time()
    initial_relaxed_dd_time = end_time - start_time

    # Check if relaxed DD timed out
    if time_budget_exceeded(time_limit_seconds, time_limit_start)
        finalize_and_log_timing!(
            timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
            0.0, 0.0, conversion_time, preallocating_memory_time,
            OBBT_time, preprocessing_time, initial_restricted_dd_time, initial_relaxed_dd_time, true_start, logs, log_file_path
        )
        if is_feasible_restricted || bks_was_updated
            return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, best_bound, sense_multiplier, 1, time() - true_start, false, true)
        else
            return return_infeasible(int_data_type, numerical_precision, 1, time() - true_start, false, true)
        end
    end

    # Capture overhead from initial DD phases (printing, logging, setup, etc.)
    dd_ws.timing_stats.other_work_time += (initial_restricted_dd_time - dd_ws.timing_stats.restricted_dd_total)
    dd_ws.timing_stats.other_work_time += (initial_relaxed_dd_time - dd_ws.timing_stats.relaxed_dd_total)

    #########################################
    ######### BRANCH AND BOUND ##############
    #########################################
    start_time = time()
    double_counted_time = 0.0
    node_count = 1  # Root node already processed by initial restricted/relaxed DDs

    if !suppress_all_prints
        println("\n","Brance-and-Bound Running")
    end

    timed_out = false

    if num_workers == 1
        while !isempty(dd_ws.bnb_queue)
            # Check time budget before processing next node
            if time_budget_exceeded(time_limit_seconds, time_limit_start)
                timed_out = true
                break
            end

            next_node = heap_pop!(dd_ws.bnb_queue)

            cur_bound = next_node.implied_bound
            if cur_bound > bkv
                break
            end

            if debug_mode
                println("\n\n","Next Node: ", next_node)
                println("Queue Length: ", length(dd_ws.bnb_queue))
            end

            if best_bound < cur_bound
                best_bound = cur_bound
                log_bounds(best_bound, bkv, sense_multiplier, length(dd_ws.bnb_queue), bounds_print, suppress_all_prints, runtime_log_file_path,  solve_start, true_start,  logs, wait_to_write_solutions)
            end

            temp_bkv, overhead_time, dd_time_during = process_queue_node!(next_node, dd_ws, int_obj_coeffs, coefficient_matrix_int_cols, coefficient_matrix_cont_cols, coefficient_matrix_rhs_vector, int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, cont_var_to_pos_rows, cont_var_to_neg_rows, lbs_int, ubs_int, lbs_cont, ubs_cont, cont_obj_coeffs, num_int_vars, num_cont_vars, num_constraints, infimum_gaps, cont_inf_gap_ctrbtns, bkv, lp_sub_model, lp_vars, lp_constraint_refs, restricted_w, relaxed_w, num_LPs_to_run, debug_mode, inverse_order, sense_multiplier, solve_start, true_start, calculate_child_time_budget(time_limit_seconds, time_limit_start))
            node_count += 1
            double_counted_time += dd_time_during

            if temp_bkv < bkv
                bkv = temp_bkv
                prune_suboptimal_nodes!(dd_ws.bnb_queue, bkv)
                merge_and_log_solution!(dd_ws.logs, logs, dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars, solution_print, suppress_all_prints, runtime_log_file_path, solve_start, true_start, wait_to_write_solutions)
            end

        end #end bnb serial
    else
        timing_start = time()
        # Parallel B&B loop with work-stealing pattern
        global_queue = dd_ws.bnb_queue

        # Create channels for communication
        # node_to_process, current_bkv, time_remaining
        work_channel = Channel{Tuple{QueueNode{int_data_type, numerical_precision}, numerical_precision, Union{Float64, Nothing}}}(num_workers * 2)
        # new_bkv, new_nodes, bks_int, bks_cont, logs, node_bound
        if all_vars_binary
            results_channel = Channel{Tuple{numerical_precision, Vector{QueueNode{int_data_type, numerical_precision}}, BitVector, Vector{numerical_precision}, Vector{Dict{String, Any}}, numerical_precision}}(num_workers * 2)
        else
            results_channel = Channel{Tuple{numerical_precision, Vector{QueueNode{int_data_type, numerical_precision}}, Vector{int_data_type}, Vector{numerical_precision}, Vector{Dict{String, Any}}, numerical_precision}}(num_workers * 2)
        end

        # Launch worker tasks
        worker_tasks = Vector{Task}(undef, num_workers)
        for worker_id in 2:num_workers
            # Unpack LP params for this worker and capture in local scope
            local_lp_model, local_lp_vars, local_lp_constraints = lp_params[worker_id]
            local_workspace = dd_workspaces[worker_id]
            worker_tasks[worker_id] = Threads.@spawn begin
                try
                    worker_loop(
                        work_channel, results_channel,
                        local_workspace,
                        int_obj_coeffs, coefficient_matrix_int_cols, coefficient_matrix_cont_cols, coefficient_matrix_rhs_vector,
                        int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows,
                        cont_var_to_pos_rows, cont_var_to_neg_rows,
                        lbs_int, ubs_int, lbs_cont, ubs_cont, cont_obj_coeffs,
                        num_int_vars, num_cont_vars, num_constraints,
                        infimum_gaps, cont_inf_gap_ctrbtns,
                        local_lp_model, local_lp_vars, local_lp_constraints,
                        restricted_w, relaxed_w, num_LPs_to_run, debug_mode,
                        inverse_order, sense_multiplier, solve_start, true_start
                    )
                catch e
                    println("╔═══════════════════════════════════════════════════════════")
                    println("║ WORKER CRASH - Thread $(Threads.threadid())")
                    println("╠═══════════════════════════════════════════════════════════")
                    println("║ Exception: ", e)
                    println("║ Stacktrace:")
                    for (exc, bt) in Base.catch_stack()
                        showerror(stdout, exc, bt)
                        println()
                    end
                    println("╚═══════════════════════════════════════════════════════════")
                    flush(stdout)
                    rethrow()
                end
            end
        end

        # Track in-flight work for termination detection
        in_flight = Threads.Atomic{Int}(0)
        max_channel_size = num_workers * 2

        # Track bounds of in-flight nodes for accurate bound logging
        in_flight_bounds = Vector{numerical_precision}()

        timing_end = time()
        double_counted_time += (timing_end - timing_start)
        # Master loop

        while true
            # Check time budget before processing
            if time_budget_exceeded(time_limit_seconds, time_limit_start)
                timed_out = true

                # Drain work channel to prevent workers from picking up new work
                while isready(work_channel)
                    take!(work_channel)
                end

                # Drain results channel to unblock in-flight workers
                while isready(results_channel)
                    take!(results_channel)
                    Threads.atomic_sub!(in_flight, 1)
                end
            else
                # 1. Process any completed results first
                while isready(results_channel)
                    timing_start = time()

                    temp_bkv, new_nodes, worker_bks_int, worker_bks_cont, new_logs, node_bound = take!(results_channel)
                    node_count += 1
                    Threads.atomic_sub!(in_flight, 1)

                    # Remove this worker's bound from tracking
                    idx = findfirst(==(node_bound), in_flight_bounds)
                    deleteat!(in_flight_bounds, idx)

                    # Update best known value and solution
                    if temp_bkv < bkv
                        bkv = temp_bkv
                        dd_ws.bks_int .= worker_bks_int
                        dd_ws.bks_cont .= worker_bks_cont
                        prune_suboptimal_nodes!(global_queue, bkv)
                        merge_and_log_solution!(new_logs, logs, dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars, solution_print, suppress_all_prints, runtime_log_file_path, solve_start, true_start, wait_to_write_solutions)
                    end

                    # Add new nodes to global queue (filter out pruned nodes first)
                    filter!(node -> node.implied_bound <= bkv, new_nodes)
                    if !isempty(new_nodes)
                        heap_insert!(global_queue, new_nodes)
                    end

                    # Calculate and log best bound if it improved
                    if !isempty(in_flight_bounds) || !isempty(global_queue)
                        current_best_bound = typemax(numerical_precision)
                        if !isempty(in_flight_bounds)
                            current_best_bound = min(current_best_bound, minimum(in_flight_bounds))
                        end
                        if !isempty(global_queue)
                            current_best_bound = min(current_best_bound, global_queue[1].implied_bound)
                        end

                        if current_best_bound > best_bound
                            best_bound = current_best_bound
                            log_bounds(best_bound, bkv, sense_multiplier, length(global_queue), bounds_print, suppress_all_prints, runtime_log_file_path, solve_start, true_start, logs, wait_to_write_solutions)
                        end
                    end

                    timing_end = time()
                    double_counted_time += (timing_end - timing_start)
                end

                # 2. Dispatch work from queue to workers
                while !isempty(global_queue) && in_flight[] < max_channel_size
                    timing_start = time()

                    next_node = heap_pop!(global_queue)

                    # Skip nodes that are already pruned (all remaining nodes)
                    if next_node.implied_bound > bkv
                        empty!(global_queue)
                        break
                    end

                    if debug_mode
                        println("\n\n","Dispatching node: ", next_node)
                        println("Queue Length: ", length(global_queue))
                    end

                    put!(work_channel, (next_node, bkv, calculate_child_time_budget(time_limit_seconds, time_limit_start)))
                    push!(in_flight_bounds, next_node.implied_bound)
                    Threads.atomic_add!(in_flight, 1)

                    timing_end = time()
                    double_counted_time += (timing_end - timing_start)
                end
            end


            # 3. Wait for in-flight work or check termination
            if (in_flight[] == 0 && isempty(global_queue)) || timed_out
                timing_start = time()

                # No work in queue and no workers processing - we're done
                for _ in 1:num_workers
                    poison_pill = QueueNode{int_data_type, numerical_precision}(
                        typemax(numerical_precision),  # ltr
                        typemin(int_data_type),       # implied_lb
                        typemin(int_data_type),       # implied_ub
                        int_data_type[],              # empty path (shutdown signal)
                        typemax(numerical_precision),  # implied_bound
                        typemax(numerical_precision)   # cont_bound_contr
                    )
                    put!(work_channel, (poison_pill, typemax(numerical_precision), nothing))
                end

                timing_end = time()
                double_counted_time += (timing_end - timing_start)

                break
            else
                # Queue has work but channel is full, yield to let workers drain
                yield()
            end
        end #end master loop 

        # Wait for all workers to finish
        for worker_id in 2:num_workers
            wait(worker_tasks[worker_id])
        end
        timing_start = time()

        close(work_channel)
        close(results_channel)

        timing_end = time()
        double_counted_time += (timing_end - timing_start)
    end #end bnb parallel

    if !suppress_all_prints
        println("\n","Finished Branch-and-Bound")
        # println("Branch-and-Bound took: ", round(bnb_time, digits=3), " seconds")
    end

    end_time = time()
    bnb_time = end_time - start_time

    finalize_and_log_timing!(
        timer_outputs, wait_to_write_solutions, num_workers, dd_ws, dd_workspaces,
        bnb_time, double_counted_time, conversion_time, preallocating_memory_time,
        OBBT_time, preprocessing_time, initial_restricted_dd_time, initial_relaxed_dd_time,
        true_start, logs, log_file_path
    )

    end_time = time()
    true_time = end_time - true_start
    solve_time = end_time - solve_start

    if !suppress_all_prints
        println("Solver took: ", round(solve_time, digits=3), " seconds")
        println("Loading and preallocation took: ", round(true_time-solve_time, digits=3), " seconds")
        println("Total: ", round(true_time, digits=3), " seconds")
    end

    if bkv != typemax(numerical_precision)
        # Fix integer variables and resolve LP for continuous variables
        bkv = resolve_continuous_variables!(model, dd_ws.bks_int, dd_ws.bks_cont, int_var_refs, cont_var_refs, num_int_vars, num_cont_vars, sense_multiplier, bkv, suppress_all_prints)

        log_solution(
            dd_ws.bks_int, dd_ws.bks_cont, editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars,
            solution_print, suppress_all_prints,
            !timed_out,
            log_file_path, solve_start, true_start,
            logs, false
        )

        return return_solution(dd_ws.bks_int, dd_ws.bks_cont, inverse_order, bkv, timed_out ? best_bound : bkv, sense_multiplier, node_count, time() - true_start, !timed_out, timed_out)
    else
        return return_infeasible(int_data_type, numerical_precision, node_count, time() - true_start, !timed_out, timed_out)
    end
end




"""
    worker_loop(work_channel::Channel, results_channel::Channel, dd_ws::DDWorkspace{Z,T}, ...) where {Z<:Integer, T<:Real}

Worker loop for parallel branch-and-bound processing using work-stealing pattern.

# Arguments
- `work_channel::Channel`: Shared channel to pull work from (work-stealing)
- `results_channel::Channel`: Channel to send results back to master
- `dd_ws::DDWorkspace{Z,T}`: Worker's dedicated workspace
- (remaining arguments same as process_queue_node!)

# Algorithm
1. Pull work from shared channel (blocks until work available)
2. Check for poison pill shutdown signal (empty path)
3. Clear worker's queue to prepare for new nodes
4. Process the queue node using process_queue_node!
5. Extract newly added nodes from worker's queue
6. Send results back to master
7. Loop back to step 1
"""
function worker_loop(
    work_channel::Channel,
    results_channel::Channel,
    dd_ws::DDWorkspace{Z,T},
    #problem description
    int_obj_coeffs::Vector{T}, coefficient_matrix_int_cols::Matrix{T}, coefficient_matrix_cont_cols::Matrix{T}, coefficient_matrix_rhs_vector::Vector{T},
    int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, int_var_to_zero_rows::Dict{Int, Vector{Int}},
    cont_var_to_pos_rows::Dict{Int, Vector{Int}}, cont_var_to_neg_rows::Dict{Int, Vector{Int}},
    lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector}, lbs_cont::Vector{T}, ubs_cont::Vector{T}, cont_obj_coeffs::Vector{T},
    num_int_vars::Int, num_cont_vars::Int, num_constraints::Int,
    #global infimum gaps
    infimum_gaps::Vector{T}, cont_inf_gap_ctrbtns::Vector{T},
    #LP model
    lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef},
    #settings
    restricted_w::Int, relaxed_w::Int, num_LPs_to_run::Int, debug_mode::Bool,
    #logging
    inverse_order::Vector{Int}, sense_multiplier::Int, solve_start::Float64, true_start::Float64
) where {Z<:Integer, T<:Real}

    while true
        # Track idle time waiting for work
        idle_start = time()

        node, current_bkv, time_remaining = take!(work_channel)
        work_received_time = time()
        dd_ws.timing_stats.worker_idle_time += (work_received_time - idle_start)

        # Track total iteration time
        iteration_start = time()

        # Check for poison pill shutdown signal (empty path)
        if isempty(node.path)
            break
        end

        # Clear worker's queue before processing
        empty!(dd_ws.bnb_queue)

        # Recalculate time remaining to account for overhead before processing
        time_remaining = calculate_child_time_budget(time_remaining, work_received_time)

        # Process the node
        temp_bkv, overhead_time, dd_time_during = process_queue_node!(
            node, dd_ws,
            int_obj_coeffs, coefficient_matrix_int_cols, coefficient_matrix_cont_cols, coefficient_matrix_rhs_vector,
            int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows,
            cont_var_to_pos_rows, cont_var_to_neg_rows,
            lbs_int, ubs_int, lbs_cont, ubs_cont, cont_obj_coeffs,
            num_int_vars, num_cont_vars, num_constraints,
            infimum_gaps, cont_inf_gap_ctrbtns,
            current_bkv,
            lp_sub_model, lp_vars, lp_constraint_refs,
            restricted_w, relaxed_w, num_LPs_to_run, debug_mode,
            inverse_order, sense_multiplier, solve_start, true_start,
            time_remaining
        )

        # Extract newly added nodes from worker's queue
        new_nodes = copy(dd_ws.bnb_queue)

        new_logs = copy(dd_ws.logs)
        empty!(dd_ws.logs)

        # Send results back to master
        put!(results_channel, (temp_bkv, new_nodes, copy(dd_ws.bks_int), copy(dd_ws.bks_cont), new_logs, node.implied_bound))

        # Track other work time (everything except DD construction)
        iteration_end = time()
        dd_ws.timing_stats.other_work_time += (iteration_end - iteration_start) - dd_time_during
    end
end


"""
    process_queue_node!(next_node::QueueNode{Z,T}, dd_ws::DDWorkspace{Z,T}, ...) where {Z<:Integer, T<:Real}

Process a single queue node in the branch-and-bound loop by running restricted and relaxed decision diagrams.

# Arguments
- `next_node::QueueNode{Z,T}`: Queue node to process with partial variable assignments
- `dd_ws::DDWorkspace{Z,T}`: Preallocated workspace containing all DD data structures
- `int_obj_coeffs::Vector{T}`: Objective coefficients for integer variables
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `coefficient_matrix_cont_cols::Matrix{T}`: Constraint coefficient matrix for continuous variables
- `coefficient_matrix_rhs_vector::Vector{T}`: Constraint RHS vector
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Integer variable to positive coefficient rows mapping
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Integer variable to negative coefficient rows mapping
- `int_var_to_zero_rows::Dict{Int, Vector{Int}}`: Integer variable to zero coefficient rows mapping
- `cont_var_to_pos_rows::Dict{Int, Vector{Int}}`: Continuous variable to positive coefficient rows mapping
- `cont_var_to_neg_rows::Dict{Int, Vector{Int}}`: Continuous variable to negative coefficient rows mapping
- `lbs_int::Union{Vector{Z}, BitVector}`: Global lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Global upper bounds for integer variables
- `lbs_cont::Vector{T}`: Global lower bounds for continuous variables
- `ubs_cont::Vector{T}`: Global upper bounds for continuous variables
- `cont_obj_coeffs::Vector{T}`: Objective coefficients for continuous variables
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `infimum_gaps::Vector{T}`: Global infimum gaps for each constraint
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps
- `bkv::T`: Current best known objective value
- `lp_sub_model::JuMP.Model`: LP model for continuous variable subproblems
- `lp_vars::Vector{JuMP.VariableRef}`: LP variable references
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: LP constraint references
- `restricted_w::Int`: Maximum width for restricted decision diagrams
- `relaxed_w::Int`: Maximum width for relaxed decision diagrams
- `num_LPs_to_run::Int`: Maximum number of LP subproblems to solve
- `debug_mode::Bool`: Whether to print detailed debug information
- `inverse_order::Vector{Int}`: Inverse permutation mapping METIS order back to original order
- `sense_multiplier::Int`: Objective sense multiplier (+1 for Min, -1 for Max)
- `solve_start::Float64`: Timestamp when solve phase started
- `true_start::Float64`: Timestamp when overall execution started
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds (nothing if no time limit)

# Returns
- `Tuple{T, Float64, Float64}`: A tuple containing:
  1. `T`: Updated best known objective value (potentially improved from input bkv)
  2. `Float64`: Overhead time (time spent outside DD construction)
  3. `Float64`: DD time (time spent in DD construction)

# Algorithm
1. Copy global bounds to local workspace and fix node path variables
2. Compute node-specific infimum gaps from queue node partial assignment
3. Run FBBT bound tightening on remaining variables
4. Return early if FBBT detects infeasibility
5. Compute rough bounds for pruning
6. Run restricted DD to find feasible solutions (updates dd_ws.bks_int/bks_cont if improved)
7. Return early if restricted DD is exact
8. Run relaxed DD to generate new branching nodes (may update dd_ws.bks_int/bks_cont)
9. Return updated objective value, overhead time, and DD time
"""
function process_queue_node!(
    next_node::QueueNode{Z,T},
    dd_ws::DDWorkspace{Z,T},
    #problem description
    int_obj_coeffs::Vector{T}, coefficient_matrix_int_cols::Matrix{T}, coefficient_matrix_cont_cols::Matrix{T}, coefficient_matrix_rhs_vector::Vector{T},
    int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, int_var_to_zero_rows::Dict{Int, Vector{Int}},
    cont_var_to_pos_rows::Dict{Int, Vector{Int}}, cont_var_to_neg_rows::Dict{Int, Vector{Int}},
    lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector}, lbs_cont::Vector{T}, ubs_cont::Vector{T}, cont_obj_coeffs::Vector{T},
    num_int_vars::Int, num_cont_vars::Int, num_constraints::Int,
    #global infimum gaps
    infimum_gaps::Vector{T}, cont_inf_gap_ctrbtns::Vector{T},
    #best stuff
    bkv::T,
    #LP model
    lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef},
    #settings
    restricted_w::Int, relaxed_w::Int, num_LPs_to_run::Int, debug_mode::Bool,
    #logging
    inverse_order::Vector{Int}, sense_multiplier::Int, solve_start::Float64, true_start::Float64,
    time_remaining::Union{Float64, Nothing} = nothing
) where {Z<:Integer, T<:Real}
    function_start_time = time()
    dd_time_before = dd_ws.timing_stats.restricted_dd_total + dd_ws.timing_stats.relaxed_dd_total

    node_path = next_node.path
    node_path_length = length(node_path)


    # *** FBBT Integration ***
    # 1. Copy global bounds to local working copies
    dd_ws.local_lbs_int .= lbs_int
    dd_ws.local_ubs_int .= ubs_int
    dd_ws.local_lbs_cont .= lbs_cont
    dd_ws.local_ubs_cont .= ubs_cont
    dd_ws.local_cont_inf_contr .= cont_inf_gap_ctrbtns

    # 1.5. Fix bounds for variables that are already assigned in the queue node path
    @inbounds for i in 1:node_path_length
        fixed_value = next_node.path[i]
        dd_ws.local_lbs_int[i] = fixed_value
        dd_ws.local_ubs_int[i] = fixed_value
    end

    # 2. Compute node-specific infimum gaps (existing function)
    compute_infimum_gaps_for_qnode!(
        dd_ws.local_infimum_gaps, infimum_gaps, next_node,
        coefficient_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows,
        lbs_int, ubs_int, node_path_length
    )

    # 3. Run FBBT with node-specific gaps and bounds
    fbbt_yields_feasible_bounds, fbbt_bounds_improved, fbbt_iterations = fbbt_bound_tightening!(
        dd_ws.local_ubs_int, dd_ws.local_lbs_int, dd_ws.local_ubs_cont, dd_ws.local_lbs_cont,
        coefficient_matrix_int_cols, coefficient_matrix_cont_cols,
        int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows,
        cont_var_to_pos_rows, cont_var_to_neg_rows,
        dd_ws.local_infimum_gaps, dd_ws.local_cont_inf_contr, num_int_vars, num_cont_vars, dd_ws.gap_adjustments,
        max_iterations = 100
    )

    # 4. Skip node if FBBT detected infeasibility
    if ! fbbt_yields_feasible_bounds
        if debug_mode
            println("FBBT detected infeasibility - skipping node")
        end

        dd_time_during = (dd_ws.timing_stats.restricted_dd_total + dd_ws.timing_stats.relaxed_dd_total) - dd_time_before
        overhead_time = (time() - function_start_time) - dd_time_during
        return bkv, overhead_time, dd_time_during
    end

    #do rough bounding
    compute_rough_bounding_vector!(dd_ws.rough_bounds_int, int_obj_coeffs, num_int_vars, dd_ws.local_lbs_int, dd_ws.local_ubs_int)
    rough_bounds_cont_val = compute_total_rough_bound(cont_obj_coeffs, num_cont_vars, dd_ws.local_lbs_cont, dd_ws.local_ubs_cont)
    local_cont_bound = max(next_node.cont_bound_contr, rough_bounds_cont_val)

    temp_bkv, is_exact = setup_and_run_restricted_dd!(
        dd_ws.rdd, next_node, bkv, dd_ws.bks_int, dd_ws.bks_cont,
        int_var_to_pos_rows, int_var_to_neg_rows, dd_ws.local_ubs_int, dd_ws.local_lbs_int, dd_ws.original_lbs_int,
        num_int_vars, num_cont_vars, num_constraints,
        dd_ws.rough_bounds_int, local_cont_bound, dd_ws.local_infimum_gaps, dd_ws.local_cont_inf_contr,
        dd_ws.coeff_times_val, dd_ws.gap_adjustments, dd_ws.inv_coeff, dd_ws.inv_obj_coeffs,
        lp_sub_model, lp_vars, lp_constraint_refs,
        dd_ws.bin_counts, dd_ws.restricted_bins_matrix, dd_ws.cumulative_bins, restricted_w, num_LPs_to_run,
        dd_ws.timing_stats,
        calculate_child_time_budget(time_remaining, function_start_time)
    )

    if debug_mode
        println("Restricted Node Exact: ", is_exact)
    end

    if temp_bkv < bkv
        bkv = temp_bkv
        log_solution(dd_ws.bks_int, dd_ws.bks_cont, dd_ws.editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars, false, true, false, nothing, solve_start, true_start, dd_ws.logs, true)
    end
    if is_exact
        dd_time_during = (dd_ws.timing_stats.restricted_dd_total + dd_ws.timing_stats.relaxed_dd_total) - dd_time_before
        overhead_time = (time() - function_start_time) - dd_time_during
        return bkv, overhead_time, dd_time_during
    end

    # Check if time budget exceeded after restricted DD
    if time_budget_exceeded(time_remaining, function_start_time)
        dd_time_during = (dd_ws.timing_stats.restricted_dd_total + dd_ws.timing_stats.relaxed_dd_total) - dd_time_before
        overhead_time = (time() - function_start_time) - dd_time_during
        return bkv, overhead_time, dd_time_during
    end

    temp_bkv, bks_was_updated, dd_ws.extra_layer = setup_run_process_relaxed_dd!(
        int_obj_coeffs, coefficient_matrix_int_cols, coefficient_matrix_rhs_vector, dd_ws.inv_coeff, dd_ws.coeff_times_val,
        int_var_to_pos_rows, int_var_to_neg_rows,
        dd_ws.local_ubs_int, dd_ws.local_lbs_int, dd_ws.original_lbs_int, bkv,  dd_ws.bks_int, dd_ws.bks_cont, next_node,
        num_int_vars, num_cont_vars, num_constraints,
        dd_ws.rough_bounds_int, local_cont_bound,
        dd_ws.local_infimum_gaps, dd_ws.infimum_gap_matrices, dd_ws.node_matrix, dd_ws.extra_layer,
        dd_ws.ltr_matrix, dd_ws.ltt_matrix,
        dd_ws.lb_matrix, dd_ws.ub_matrix, dd_ws.low_indexes, dd_ws.high_indexes, dd_ws.rel_path, dd_ws.feasibility_accumulator,
        dd_ws.wrk_vec, dd_ws.local_cont_inf_contr,
        dd_ws.arc_count_per_node, dd_ws.node_bin_counts, dd_ws.node_cumulative, dd_ws.global_lower, dd_ws.global_upper, dd_ws.bins_matrix,
        dd_ws.bnb_queue, dd_ws.added_queue_nodes,
        lp_sub_model, lp_vars, lp_constraint_refs,
        relaxed_w, dd_ws.timing_stats;
        debug_mode = debug_mode,
        time_remaining = calculate_child_time_budget(time_remaining, function_start_time)
    )

    if temp_bkv < bkv
        bkv = temp_bkv
        log_solution(dd_ws.bks_int, dd_ws.bks_cont, dd_ws.editable_int_solution, inverse_order, bkv, sense_multiplier, num_int_vars, false, true, false, nothing, solve_start, true_start, dd_ws.logs, true)
    end

    dd_time_during = (dd_ws.timing_stats.restricted_dd_total + dd_ws.timing_stats.relaxed_dd_total) - dd_time_before
    overhead_time = (time() - function_start_time) - dd_time_during
    return bkv, overhead_time, dd_time_during
end


"""
    setup_from_JuMP_model(model::JuMP.Model, precision::DataType, suppress_all_prints::Bool, debug_mode::Bool, custom_variable_order::Union{Vector{VariableRef}, Nothing} = nothing)

Converts a JuMP model into ImplicitDDs solver data structures with adaptive variable ordering.

# Arguments
- `model::JuMP.Model`: JuMP model containing MIP formulation
- `precision::DataType`: Numerical precision for continuous variables (e.g., Float32, Float64)
- `suppress_all_prints::Bool`: Whether to suppress progress print statements
- `debug_mode::Bool`: Whether to print detailed debug information
- `custom_variable_order::Union{Vector{VariableRef}, Nothing}`: Optional custom ordering for integer variables.
  If provided, must contain exactly the same VariableRefs as the model's integer variables. If `nothing`,
  uses METIS-based adaptive ordering.

# Returns
- `Tuple`: A tuple containing:
  1. `JuMP.Model`: Original JuMP model
  2. `Vector{Int}`: Lower bounds for integer variables
  3. `Vector{Int}`: Upper bounds for integer variables
  4. `Vector{T}`: Lower bounds for continuous variables
  5. `Vector{T}`: Upper bounds for continuous variables
  6. `Int`: Number of integer variables
  7. `Int`: Number of continuous variables
  8. `Int`: Number of constraints
  9. `Vector{T}`: Objective coefficients for integer variables
  10. `Vector{T}`: Objective coefficients for continuous variables
  11. `T`: Objective constant term
  12. `Matrix{T}`: Constraint coefficient matrix for integer variables
  13. `Dict{Int, Vector{Int}}`: Integer variable to positive coefficient rows mapping
  14. `Dict{Int, Vector{Int}}`: Integer variable to negative coefficient rows mapping
  15. `Dict{Int, Vector{Int}}`: Integer variable to zero coefficient rows mapping
  16. `Matrix{T}`: Constraint coefficient matrix for continuous variables
  17. `Dict{Int, Vector{Int}}`: Continuous variable to positive coefficient rows mapping
  18. `Dict{Int, Vector{Int}}`: Continuous variable to negative coefficient rows mapping
  19. `Dict{Int, Vector{Int}}`: Continuous variable to zero coefficient rows mapping
  20. `Vector{T}`: Constraint RHS vector
  21. `Vector{JuMP.VariableRef}`: Ordered integer variable references
  22. `Vector{JuMP.VariableRef}`: Continuous variable references
  23. `Int`: Objective sense multiplier (+1 for Min, -1 for Max)
  24. `Vector{Int}`: Inverse permutation mapping ordered variables back to original order

# Algorithm
1. Extract integer and continuous variable references from model
2. Determine coefficient type from constraint formulation
3. Apply variable ordering: use custom order if provided, otherwise adaptive ordering
4. Extract variable bounds (with binary variable bound correction)
5. Build objective coefficient vectors with sense normalization
6. Construct constraint coefficient matrices with row mappings
7. Return all solver-ready data structures
"""
function setup_from_JuMP_model(model::JuMP.Model, precision::DataType, suppress_all_prints::Bool, debug_mode::Bool, custom_variable_order::Union{Vector{VariableRef}, Nothing} = nothing)
    if !suppress_all_prints
        print("Converting Model ... ")
    end
    start_time = time()

    moi_model = backend(model)

    #variable sorting
    int_vars, cont_vars = get_var_refs(model)
    num_int_vars = length(int_vars)
    num_cont_vars = length(cont_vars)

    # Get all constraint types to determine coefficient type
    constraint_types = MOI.get(moi_model, MOI.ListOfConstraintTypesPresent())
    # Find the coefficient type from ScalarAffineFunction constraints
    coefficient_type = Float64  # default fallback
    for (func_type, set_type) in constraint_types
        if func_type <: MOI.ScalarAffineFunction
            # Extract the type parameter from ScalarAffineFunction{T}
            coefficient_type = func_type.parameters[1]
            break
        end
    end

    # Convert to GenericModel using the detected coefficient type
    gm = MathOptInterface.Utilities.Model{coefficient_type}()
    MOI.copy_to(gm, moi_model)

    moi_int_vars = index.(int_vars)

    # Determine variable ordering: use custom order if provided, otherwise METIS
    if custom_variable_order !== nothing
        # Validate length
        if length(custom_variable_order) != num_int_vars
            error("custom_variable_order has length $(length(custom_variable_order)) but model has $num_int_vars integer variables")
        end

        # Validate custom order contains exactly the same variables as model
        custom_set = Set(custom_variable_order)
        model_set = Set(int_vars)
        if custom_set != model_set
            missing_vars = setdiff(model_set, custom_set)
            extra_vars = setdiff(custom_set, model_set)
            error_msg = "custom_variable_order does not match model's integer variables."
            if !isempty(missing_vars)
                error_msg *= " Missing: $(length(missing_vars)) variables."
            end
            if !isempty(extra_vars)
                error_msg *= " Extra: $(length(extra_vars)) variables."
            end
            error(error_msg)
        end

        # Compute sort order: map each custom position to original position
        # int_var_sort_order[i] = j means: put original variable j at position i
        var_to_original_pos = Dict(var => i for (i, var) in enumerate(int_vars))
        int_var_sort_order = [var_to_original_pos[var] for var in custom_variable_order]
    else
        int_var_sort_order = adaptive_metis_variable_ordering(
            gm,
            moi_int_vars,
            num_int_vars,
            coefficient_type
        )
    end

    if debug_mode
        println("\n","Variable Order:", int_var_sort_order)
    end

    int_vars[:] = int_vars[int_var_sort_order]

    #prebound the variables
    ubs_int, lbs_int, ubs_cont, lbs_cont = extract_variable_bounds(int_vars, cont_vars, precision)
    
    moi_int_vars_ordered = index.(int_vars)
    moi_cont_vars_ordered = index.(cont_vars)
    
    gm = MathOptInterface.Utilities.Model{coefficient_type}()
    MOI.copy_to(gm, moi_model)
    int_obj_coeffs, cont_obj_coeffs, obj_const, sense_multiplier = get_objective_vector(gm, moi_int_vars_ordered, moi_cont_vars_ordered, coefficient_type, precision)

    coefficient_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, coefficient_matrix_cont_cols, cont_var_to_pos_rows, cont_var_to_neg_rows, cont_var_to_zero_rows, coefficient_matrix_rhs_vector = get_coefficient_matrix(gm, moi_int_vars_ordered, moi_cont_vars_ordered, coefficient_type, precision)
    num_constraints = size(coefficient_matrix_int_cols, 1)

    # This maps from METIS position back to original position
    inverse_order = invperm(int_var_sort_order)

    if debug_mode
        println("Int Obj Coeffs: ")
        for coeff in int_obj_coeffs
            print("  ", coeff)
        end
        println()
        println("Cont Obj Coeffs: ")
        for coeff in cont_obj_coeffs
            print("  ", coeff)
        end
        println()
        println("Obj Const: ")
        println("  ", obj_const)
        println("RHS: ", coefficient_matrix_rhs_vector)
        println("Int Bounds: ")
        for i in 1:lastindex(lbs_int)
            print("[",lbs_int[i],",", ubs_int[i],"] ")
        end
        println()
        println("Continuous Bounds: ")
        for i in 1:lastindex(lbs_cont)
            print("[",lbs_cont[i],",", ubs_cont[i],"] ")
        end
        println()
    end

    end_time = time()
    conversion_time = end_time - start_time

    if !suppress_all_prints
        println("Finished")
        # println("Model conversion took: ", round(conversion_time, digits=3), " seconds", "\n")
        println("\n","Integer Variables: ", num_int_vars)
        println("Continuous Variables: ", num_cont_vars)
        println("Constraints: ", num_constraints, "\n")
    end

    return model, lbs_int, ubs_int, lbs_cont, ubs_cont, num_int_vars, num_cont_vars, num_constraints, int_obj_coeffs, cont_obj_coeffs, obj_const, coefficient_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, coefficient_matrix_cont_cols, cont_var_to_pos_rows, cont_var_to_neg_rows, cont_var_to_zero_rows, coefficient_matrix_rhs_vector, int_vars, cont_vars, sense_multiplier, inverse_order, conversion_time
end
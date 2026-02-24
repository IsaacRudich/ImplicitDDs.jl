# =============================================================================
# MOI.optimize!: build JuMP model, call solve_mip, store results
# =============================================================================

"""
    MOI.optimize!(optimizer::Optimizer)

Build a temporary JuMP model from stored MOI data, call `solve_mip`, and
store the results for later retrieval.
"""
function MOI.optimize!(optimizer::Optimizer)
    if optimizer.src_model === nothing
        optimizer.termination_status = MOI.OTHER_ERROR
        optimizer.primal_status = MOI.NO_SOLUTION
        return nothing
    end

    start_time = time()
    src = optimizer.src_model

    # =========================================================================
    # Build a temporary JuMP model from the MOI data
    # =========================================================================
    inner_model = JuMP.Model(HiGHS.Optimizer)
    if optimizer.silent
        JuMP.set_silent(inner_model)
    end

    # Copy MOI data into the JuMP model's backend
    model_map = MOI.copy_to(JuMP.backend(inner_model), src)

    # =========================================================================
    # Translate custom_variable_order from MOI indices to JuMP VariableRefs
    # =========================================================================
    custom_order = nothing
    if optimizer.custom_variable_order !== nothing
        custom_order = Vector{JuMP.VariableRef}()
        for moi_vi in optimizer.custom_variable_order
            # Map from original source index → src_model index → inner model index
            src_vi = optimizer.src_index_map[moi_vi]
            inner_vi = model_map[src_vi]
            push!(custom_order, JuMP.VariableRef(inner_model, inner_vi))
        end
    end

    # =========================================================================
    # Convert time limit: seconds → minutes
    # =========================================================================
    time_limit_minutes = if optimizer.time_limit !== nothing
        optimizer.time_limit / 60.0
    else
        nothing
    end

    # =========================================================================
    # Call solve_mip
    # =========================================================================
    result = try
        solve_mip(
            inner_model;
            relaxed_w = optimizer.relaxed_w,
            restricted_w = optimizer.restricted_w,
            num_LPs_to_run = optimizer.num_LPs_to_run,
            parallel_processing = optimizer.parallel_processing,
            numerical_precision = optimizer.numerical_precision,
            debug_mode = optimizer.debug_mode,
            log_file_path = optimizer.log_file_path,
            bounds_print = optimizer.bounds_print,
            solution_print = optimizer.solution_print,
            suppress_all_prints = optimizer.silent,
            wait_to_write_solutions = optimizer.wait_to_write_solutions,
            timer_outputs = optimizer.timer_outputs,
            time_limit = time_limit_minutes,
            custom_variable_order = custom_order,
        )
    catch e
        optimizer.termination_status = MOI.OTHER_ERROR
        optimizer.primal_status = MOI.NO_SOLUTION
        optimizer.solve_time = time() - start_time
        rethrow()
    end

    optimizer.solve_time = result.solve_time
    optimizer.node_count = result.node_count

    # =========================================================================
    # Store results
    # =========================================================================
    if result.is_feasible
        # Build primal_solution vector: [int_values..., cont_values...]
        n_int = length(result.bks_int)
        n_cont = length(result.bks_cont)
        optimizer.primal_solution = Vector{Float64}(undef, n_int + n_cont)
        for i in 1:n_int
            optimizer.primal_solution[i] = Float64(result.bks_int[i])
        end
        for i in 1:n_cont
            optimizer.primal_solution[n_int + i] = Float64(result.bks_cont[i])
        end

        optimizer.objective_value = Float64(result.objective_value)
        optimizer.objective_bound = Float64(result.objective_bound)
        optimizer.primal_status = MOI.FEASIBLE_POINT
        optimizer.result_count = 1

        # Compute relative gap using same formula as internal solver:
        # gap / (|bound| + gap), which avoids division by zero
        gap = abs(result.objective_value - result.objective_bound)
        denominator = abs(result.objective_bound) + gap
        optimizer.relative_gap = denominator > 0 ? gap / denominator : 0.0

        if result.timed_out
            optimizer.termination_status = MOI.TIME_LIMIT
        else
            optimizer.termination_status = MOI.OPTIMAL
        end
    else
        optimizer.primal_solution = nothing
        optimizer.objective_value = NaN
        optimizer.objective_bound = NaN
        optimizer.relative_gap = NaN
        optimizer.primal_status = MOI.NO_SOLUTION
        optimizer.result_count = 0

        if result.unbounded_error
            optimizer.termination_status = MOI.OTHER_ERROR
        elseif result.timed_out
            optimizer.termination_status = MOI.TIME_LIMIT
        elseif result.is_optimal
            optimizer.termination_status = MOI.INFEASIBLE
        else
            # Search incomplete (e.g., early termination without proof)
            optimizer.termination_status = MOI.OTHER_ERROR
        end
    end

    return nothing
end

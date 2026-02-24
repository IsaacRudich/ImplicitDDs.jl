# =============================================================================
# Result query methods
# =============================================================================

# Termination status
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return optimizer.termination_status
end

# Primal status
function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    return optimizer.primal_status
end

# Dual status (not supported - decision diagram solver doesn't produce duals)
function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    return MOI.NO_SOLUTION
end

# Result count
function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    return optimizer.result_count
end

# Objective value
function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.objective_value
end

# Objective bound (best known lower/upper bound depending on sense)
function MOI.get(optimizer::Optimizer, ::MOI.ObjectiveBound)
    return optimizer.objective_bound
end

# Relative gap: |objective_value - objective_bound| / |objective_value|
function MOI.get(optimizer::Optimizer, ::MOI.RelativeGap)
    return optimizer.relative_gap
end

# Solve time in seconds
function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.solve_time
end

# Node count (B&B nodes processed)
function MOI.get(optimizer::Optimizer, ::MOI.NodeCount)
    return optimizer.node_count
end

# Variable primal value
function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    if optimizer.primal_solution === nothing
        error("No solution available")
    end
    # Map from MOI variable index to position in solution vector
    pos = optimizer.variable_map[vi]
    return optimizer.primal_solution[pos]
end

# Vectorized variable primal
function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vis::Vector{MOI.VariableIndex})
    return [MOI.get(optimizer, attr, vi) for vi in vis]
end

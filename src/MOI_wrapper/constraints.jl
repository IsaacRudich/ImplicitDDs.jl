# =============================================================================
# Supported constraint types
# =============================================================================

# Linear constraints: a'x <= b, a'x >= b, a'x == b
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{<:Union{MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}},
)
    return true
end

# Variable bounds: lb <= x <= ub, x <= ub, x >= lb, x == val
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:Union{MOI.Interval{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64}, MOI.EqualTo{Float64}}},
)
    return true
end

# Integrality: x ∈ Z, x ∈ {0, 1}
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:Union{MOI.Integer, MOI.ZeroOne}},
)
    return true
end

# Objective: linear functions
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

# Also support single-variable objectives (e.g. @objective(model, Min, x))
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.VariableIndex},
)
    return true
end

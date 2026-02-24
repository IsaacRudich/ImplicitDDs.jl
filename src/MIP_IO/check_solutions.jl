"""
COMMENTS UP TO DATE
"""

"""
    compute_objective_value(solution::Vector{Int}, obj_coeffs::Vector{T}) where T<:Real

Computes the objective value of a candidate solution given the coefficient vector.

# Arguments
- `solution::Vector{Int}`: Candidate solution vector
- `obj_coeffs::Vector{T}`: Objective coefficient vector

# Returns
- `T`: The computed objective value
"""
function compute_objective_value(
    solution    ::Vector{Int},
    obj_coeffs  ::Vector{T}
) where T<:Real
    return mapreduce(*, +, obj_coeffs, solution)
end

"""
    compute_objective_value(int_solution::Vector{Int}, cont_solution::Vector{T}, int_obj_coeffs::Vector{T}, cont_obj_coeffs::Vector{T}) where T<:Real

Computes the objective value of a candidate solution with separate integer and continuous components.

# Arguments
- `int_solution::Vector{Int}`: Integer solution vector
- `cont_solution::Vector{T}`: Continuous solution vector
- `int_obj_coeffs::Vector{T}`: Integer variable objective coefficients
- `cont_obj_coeffs::Vector{T}`: Continuous variable objective coefficients

# Returns
- `T`: The computed objective value
"""
function compute_objective_value(
    int_solution    ::Vector{Int},
    cont_solution   ::Vector{T},
    int_obj_coeffs  ::Vector{T},
    cont_obj_coeffs ::Vector{T}
) where T<:Real
    int_contrib = mapreduce(*, +, int_obj_coeffs, int_solution)
    cont_contrib = mapreduce(*, +, cont_obj_coeffs, cont_solution)
    return int_contrib + cont_contrib
end


"""
    colwise_feasible!(coefficient_matrix_int_cols::AbstractMatrix{T}, int_solution::AbstractVector{<:Integer}, rhs::AbstractVector{T}, work::AbstractVector{T}, num_int_vars::Int, num_constraints::Int; atol::T = T(0)) where {T<:Real}

Returns `true` if the solution vector is primal-feasible for the ≤ system `A * x ≤ rhs`, otherwise `false`.

# Arguments
- `coefficient_matrix_int_cols::AbstractMatrix{T}`: Coefficient matrix for integer variables
- `int_solution::AbstractVector{<:Integer}`: Integer solution vector (length = num_int_vars)
- `rhs::AbstractVector{T}`: Right-hand side vector (length = num_constraints)
- `work::AbstractVector{T}`: Pre-allocated workspace vector (length = num_constraints), gets overwritten with LHS values
- `num_int_vars::Int`: Number of integer variables
- `num_constraints::Int`: Number of constraints
- `atol::T`: Absolute tolerance for feasibility test (default T(0))

# Returns
- `Bool`: True if solution satisfies all constraints, false otherwise
"""
function colwise_feasible!(coefficient_matrix_int_cols::AbstractMatrix{T}, int_solution::AbstractVector{<:Integer}, rhs::AbstractVector{T}, work::AbstractVector{T}, num_int_vars::Int, num_constraints::Int; atol::T = T(0)) where {T<:Real}
    # --- 1. Accumulate the LHS one column at a time -------------------------
    @inbounds fill!(work, zero(T))                  # zero the scratch space

    # Integer variable contribution
    @inbounds for j = 1:num_int_vars
        val = int_solution[j]
        val == 0 && continue                        # cheap skip for zeros
        col = view(coefficient_matrix_int_cols, :, j)  # alias, no copy
        @simd for i = 1:num_constraints
            work[i] += col[i] * val
        end
    end

    # --- 2. Compare LHS ≤ RHS ------------------------------------------------
    @inbounds for i = 1:num_constraints
        if work[i] > rhs[i] + atol
            return false                            # first violation → stop early
        end
    end
    
    return true
end



"""
    colwise_feasible!(coefficient_matrix_int_cols::AbstractMatrix{T}, int_solution::AbstractVector{<:Integer}, coefficient_matrix_cont_cols::AbstractMatrix{T}, cont_solution::AbstractVector{T}, rhs::AbstractVector{T}, work::AbstractVector{T}, num_int_vars::Int, num_cont_vars::Int, num_constraints::Int; atol::T = T(0)) where {T<:Real}

Returns `true` if the solution vector is primal-feasible for the ≤ system `[A_int A_cont] * [x_int; x_cont] ≤ rhs`, otherwise `false`.

# Arguments
- `coefficient_matrix_int_cols::AbstractMatrix{T}`: Coefficient matrix for integer variables
- `int_solution::AbstractVector{<:Integer}`: Integer solution vector (length = num_int_vars)
- `coefficient_matrix_cont_cols::AbstractMatrix{T}`: Coefficient matrix for continuous variables
- `cont_solution::AbstractVector{T}`: Continuous solution vector (length = num_cont_vars)
- `rhs::AbstractVector{T}`: Right-hand side vector (length = num_constraints)
- `work::AbstractVector{T}`: Pre-allocated workspace vector (length = num_constraints), gets overwritten with LHS values
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `atol::T`: Absolute tolerance for feasibility test (default T(0))

# Returns
- `Bool`: True if solution satisfies all constraints, false otherwise
"""
function colwise_feasible!(coefficient_matrix_int_cols::AbstractMatrix{T}, int_solution::AbstractVector{<:Integer}, coefficient_matrix_cont_cols::AbstractMatrix{T}, cont_solution::AbstractVector{T}, rhs::AbstractVector{T}, work::AbstractVector{T}, num_int_vars::Int, num_cont_vars::Int, num_constraints::Int; atol::T = T(0)) where {T<:Real}
    # --- 1. Accumulate the LHS one column at a time -------------------------
    @inbounds fill!(work, zero(T))                  # zero the scratch space

    # Integer variable contribution
    @inbounds for j = 1:num_int_vars
        val = int_solution[j]
        val == 0 && continue                        # cheap skip for zeros
        col = view(coefficient_matrix_int_cols, :, j)  # alias, no copy
        @simd for i = 1:num_constraints
            work[i] += col[i] * val
        end
    end

    # Continuous variable contribution  
    @inbounds for j = 1:num_cont_vars
        val = cont_solution[j]
        val == 0 && continue                        # cheap skip for zeros
        col = view(coefficient_matrix_cont_cols, :, j)  # alias, no copy
        @simd for i = 1:num_constraints
            work[i] += col[i] * val
        end
    end

    # --- 2. Compare LHS ≤ RHS ------------------------------------------------
    @inbounds for i = 1:num_constraints
        if work[i] > rhs[i] + atol
            return false                            # first violation → stop early
        end
    end
    
    return true
end

"""
    check_solution_feasibility(solution::Vector{U}, coefficient_matrix_int_cols::Matrix{U}, coefficient_matrix_cont_cols::Matrix{U}, coefficient_matrix_rhs_vector::Vector{U}, num_int_vars::Int, num_cont_vars::Int) where {U<:Real}

Checks if a solution is feasible by verifying all constraints are satisfied.

# Arguments
- `solution::Vector{U}`: Complete solution vector [integer_vars..., continuous_vars...]
- `coefficient_matrix_int_cols::Matrix{U}`: Coefficient matrix for integer variables
- `coefficient_matrix_cont_cols::Matrix{U}`: Coefficient matrix for continuous variables
- `coefficient_matrix_rhs_vector::Vector{U}`: Right-hand side values for constraints
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables

# Returns
- `Tuple{Bool, Vector{Tuple{Int, Float64, Float64}}}`: A tuple containing:
  1. `Bool`: True if solution satisfies all constraints, false otherwise
  2. `Vector{Tuple{Int, Float64, Float64}}`: List of (constraint_index, lhs_value, rhs_value) for violated constraints
"""
function check_solution_feasibility(
    solution                        ::Vector{U},
    coefficient_matrix_int_cols     ::Matrix{U},
    coefficient_matrix_cont_cols    ::Matrix{U},
    coefficient_matrix_rhs_vector   ::Vector{U},
    num_int_vars                    ::Int,
    num_cont_vars                   ::Int
) where {U<:Real}
    
    # Extract integer and continuous parts of solution
    int_solution = solution[1:num_int_vars]
    cont_solution = solution[num_int_vars+1:num_int_vars+num_cont_vars]
    
    num_constraints = length(coefficient_matrix_rhs_vector)
    constraint_violations = Tuple{Int, Float64, Float64}[]
    is_feasible = true
    
    println("___________________________________")
    println("Integer vars: ", int_solution)
    println("Continuous vars: ", cont_solution)
    
    # Check each constraint
    for constraint_idx in 1:num_constraints
        # Calculate left-hand side: sum of (coefficient * variable_value)
        lhs_value = 0.0
        
        # Integer variable contribution
        for var_idx in 1:num_int_vars
            lhs_value += coefficient_matrix_int_cols[constraint_idx, var_idx] * int_solution[var_idx]
        end
        
        # Continuous variable contribution
        for var_idx in 1:num_cont_vars
            lhs_value += coefficient_matrix_cont_cols[constraint_idx, var_idx] * cont_solution[var_idx]
        end
        
        rhs_value = coefficient_matrix_rhs_vector[constraint_idx]
        
        # Check if constraint is satisfied (assuming ≤ constraints)
        if lhs_value > rhs_value + 1e-4  # Small tolerance for numerical precision
            is_feasible = false
            push!(constraint_violations, (constraint_idx, lhs_value, rhs_value))
            println("  VIOLATED - Constraint $constraint_idx: $lhs_value > $rhs_value (violation: $(lhs_value - rhs_value))")
        else
            # println("  OK - Constraint $constraint_idx: $lhs_value ≤ $rhs_value (slack: $(rhs_value - lhs_value))")
        end
    end
    
    println()
    if is_feasible
        println("✓ Solution is FEASIBLE - all constraints satisfied")
    else
        println("✗ Solution is INFEASIBLE - $(length(constraint_violations)) constraint(s) violated")
    end
    println("___________________________________")
    
    return is_feasible, constraint_violations
end




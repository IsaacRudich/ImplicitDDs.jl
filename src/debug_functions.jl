"""
COMMENTS UP TO DATE
"""

"""
    debug_infimum_gap_vector(coefficient_matrix, path, var_to_pos_rows, var_to_neg_rows, lbs, ubs)

Debug function that generates the infimum gap vector for a given path to be used in verification.

This function computes the infimum gap vector by:
1. Starting with the RHS values from the constraint matrix
2. For each variable in the path, subtracting its contribution based on its assigned value
3. The result represents how much "gap" remains for each constraint after the path assignments

# Arguments
- `coefficient_matrix::Matrix{T}`: Dense matrix where each row corresponds to a constraint; columns 1 through end-1 contain variable coefficients, and column end contains the RHS values
- `path::Vector{Int}`: Variable assignments for the path (indexed by variable position)
- `var_to_pos_rows::Dict{Int, Vector{Int}}`: Maps variable indices to constraint row indices where their coefficient is positive
- `var_to_neg_rows::Dict{Int, Vector{Int}}`: Maps variable indices to constraint row indices where their coefficient is negative
- `lbs::Vector{Int}`: Lower bounds for each variable
- `ubs::Vector{Int}`: Upper bounds for each variable

# Returns
- `infimum_gap_vector::Vector{T}`: Vector where each element represents the infimum gap for the corresponding constraint row after applying the path assignments

# Example
```julia
# For a path [2, 1, 3] representing variable assignments
gap_vector = debug_infimum_gap_vector(coefficient_matrix, [2, 1, 3], var_to_pos_rows, var_to_neg_rows, lbs, ubs)
println("Infimum gap vector: ", gap_vector)
```
"""
function debug_infimum_gap_vector(
    coefficient_matrix::Matrix{T},
    path::Vector{Int},
    var_to_pos_rows::Dict{Int, Vector{Int}},
    var_to_neg_rows::Dict{Int, Vector{Int}},
    lbs::Vector{Int},
    ubs::Vector{Int}
) where {T<:Real}
    
    # First compute the initial infimum gaps (matches compute_initial_infimum_gaps)
    infimum_gaps = copy(view(coefficient_matrix, :, size(coefficient_matrix, 2)))
    
    # Subtract the bound-based minimal contributions for ALL variables (not just the path)
    num_vars = size(coefficient_matrix, 2) - 1  # -1 for RHS column
    @inbounds for var_idx in 1:num_vars
        var_coefficients = view(coefficient_matrix, :, var_idx)
        pos_rows = var_to_pos_rows[var_idx]
        neg_rows = var_to_neg_rows[var_idx]
        
        lb = lbs[var_idx]
        ub = ubs[var_idx]
        
        # For positive coefficients, subtract lb * coeff (minimal contribution)
        for row_idx in pos_rows
            infimum_gaps[row_idx] -= lb * var_coefficients[row_idx]
        end
        
        # For negative coefficients, subtract ub * coeff (minimal contribution)
        for row_idx in neg_rows
            infimum_gaps[row_idx] -= ub * var_coefficients[row_idx]
        end
    end
    
    # Now compute the gap for this specific path
    # This matches the restricted DD computation: wrk_vec[j] - (val*col[j])
    wrk_vec = copy(infimum_gaps)
    
    # Add back the bound contributions for the path variables
    @inbounds for (var_idx, var_value) in enumerate(path)
        var_coefficients = view(coefficient_matrix, :, var_idx)
        pos_rows = var_to_pos_rows[var_idx]
        neg_rows = var_to_neg_rows[var_idx]
        
        lb = lbs[var_idx]
        ub = ubs[var_idx]
        
        # Add back the bound contribution
        for row_idx in pos_rows
            wrk_vec[row_idx] += lb * var_coefficients[row_idx]
        end
        for row_idx in neg_rows
            wrk_vec[row_idx] += ub * var_coefficients[row_idx]
        end
        
        # Then subtract the actual value contribution
        @inbounds for row_idx in 1:size(coefficient_matrix, 1)
            wrk_vec[row_idx] -= var_value * var_coefficients[row_idx]
        end
    end
    
    return wrk_vec
end
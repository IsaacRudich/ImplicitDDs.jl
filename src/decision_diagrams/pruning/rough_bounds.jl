"""
COMMENTS UP TO DATE
"""


"""
    compute_rough_bounding_vector!(rough_bounds::Vector{U}, obj_coeffs::Vector{U}, num_vars::Int, lbs::Union{Vector{V}, BitVector}, ubs::Union{Vector{V}, BitVector}) where {U<:Real, V<:Real}

Computes rough relaxed bounds for each variable's objective contribution to enable pruning in decision diagram construction.

Creates overly optimistic estimates of remaining objective contribution for pruning decisions.
For each variable i, computes the best possible contribution from variables i through num_vars,
ignoring constraint interactions. Uses preallocated memory for efficiency.

# Arguments
- `rough_bounds::Vector{U}`: Preallocated output vector (modified in-place)
- `obj_coeffs::Vector{U}`: Objective coefficients for all variables
- `num_vars::Int`: Total number of variables
- `lbs::Union{Vector{V}, BitVector}`: Lower bounds for variables
- `ubs::Union{Vector{V}, BitVector}`: Upper bounds for variables

# Algorithm
Processes variables in reverse order (num_vars down to 1):
- **Positive coefficients**: Use lower bounds (minimize contribution in minimization)
- **Negative coefficients**: Use upper bounds (minimize contribution in minimization)
- **Zero coefficients**: No contribution
- **Cumulative sum**: `rough_bounds[i]` includes contributions from variables i through num_vars

# Usage in Pruning
Used as: `current_path_cost + rough_bounds[i+1] ≥ best_known_value` to prune dominated paths.
The rough bound excludes variable i to avoid double-counting when evaluating variable i assignments.

# Usage in Branch-and-Bound
After FBBT tightens bounds at each B&B node, recompute local rough bounds with tightened variable bounds:
```julia
compute_rough_bounding_vector!(rough_bounds_int, int_obj_coeffs, num_int_vars, local_lbs_int, local_ubs_int)
```
"""
function compute_rough_bounding_vector!(
    rough_bounds::Vector{U},
    obj_coeffs::Vector{U},
    num_vars::Int,
    lbs::Union{Vector{V}, BitVector},
    ubs::Union{Vector{V}, BitVector}
) where {U<:Real, V<:Real}
    running_sum = U(0)

    @inbounds for i in num_vars:-1:1
        coeff = obj_coeffs[i]

        if coeff > 0
            running_sum += (coeff * lbs[i])
        elseif coeff < 0
            running_sum += (coeff * ubs[i])
        end

        rough_bounds[i] = running_sum
    end
end

"""
    compute_total_rough_bound(obj_coeffs::Vector{U}, num_vars::Int, lbs::Union{Vector{V}, BitVector}, ubs::Union{Vector{V}, BitVector}) where {U<:Real, V<:Real}

Computes the total rough bound contribution from all variables without allocating a vector.

Efficiently computes only the cumulative sum (equivalent to `rough_bounds[1]` from `compute_rough_bounding_vector`)
for use when only the total bound is needed, not per-variable bounds.

# Arguments
- `obj_coeffs::Vector{U}`: Objective coefficients for all variables
- `num_vars::Int`: Total number of variables
- `lbs::Union{Vector{V}, BitVector}`: Lower bounds for variables
- `ubs::Union{Vector{V}, BitVector}`: Upper bounds for variables

# Returns
- `U`: Total rough bound contribution from all variables

# Algorithm
Sums contributions from all variables:
- **Positive coefficients**: Use lower bounds (minimize contribution in minimization)
- **Negative coefficients**: Use upper bounds (minimize contribution in minimization)
- **Zero coefficients**: No contribution

# Usage
Primarily for continuous variables where only a single scalar bound is needed:
```julia
rough_bounds_cont_val = compute_total_rough_bound(cont_obj_coeffs, num_cont_vars, lbs_cont, ubs_cont)
```
"""
function compute_total_rough_bound(
    obj_coeffs::Vector{U},
    num_vars::Int,
    lbs::Union{Vector{V}, BitVector},
    ubs::Union{Vector{V}, BitVector}
) where {U<:Real, V<:Real}
    total = U(0)

    if num_vars == 0
        return total
    end

    @inbounds for i in 1:num_vars
        coeff = obj_coeffs[i]

        if coeff > 0
            total += (coeff * lbs[i])
        elseif coeff < 0
            total += (coeff * ubs[i])
        end
    end

    return total
end

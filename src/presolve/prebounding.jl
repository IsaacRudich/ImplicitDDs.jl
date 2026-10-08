"""
COMMENTS UP TO DATE
"""


"""
    run_obbt!(upper_bounds_int::Union{Vector{Z}, BitVector}, lower_bounds_int::Union{Vector{Z}, BitVector}, upper_bounds_cont::Vector{T}, lower_bounds_cont::Vector{T}, model::JuMP.Model, integer_vars::Vector{JuMP.VariableRef}, continuous_vars::Vector{JuMP.VariableRef}; tolerance::U=1e-6) where {Z<:Integer, T<:Real, U<:AbstractFloat}

Optimization-Based Bound Tightening (OBBT): Relaxes integrality constraints and individually optimizes each unbounded variable to compute tight variable-specific bounds.

# Arguments
- `upper_bounds_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables (modified in-place, only unbounded variables are optimized)
- `lower_bounds_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (modified in-place, only unbounded variables are optimized)
- `upper_bounds_cont::Vector{T}`: Upper bounds for continuous variables (modified in-place, only unbounded variables are optimized)
- `lower_bounds_cont::Vector{T}`: Lower bounds for continuous variables (modified in-place, only unbounded variables are optimized)
- `model::JuMP.Model`: A JuMP model with MIP formulation (integrality will be temporarily relaxed)
- `integer_vars::Vector{JuMP.VariableRef}`: Integer variables to potentially bound
- `continuous_vars::Vector{JuMP.VariableRef}`: Continuous variables to potentially bound
- `tolerance::U`: Tolerance for floating point precision when rounding bounds (default: 1e-6)

# Returns
- `Tuple{Bool, Bool}`: A tuple containing:
  1. `Bool`: `true` if all variables are bounded, `false` otherwise
  2. `Bool`: `false` if the LP relaxation is infeasible, `true` otherwise

# Algorithm
1. Check if any variables are unbounded (at typemax/typemin limits)
2. If all variables are already bounded, return true immediately (no optimization needed)
3. If unbounded variables exist, relax integrality constraints and solve LP relaxation
4. If initial LP relaxation is infeasible, restore model state and return false
5. If LP relaxation is feasible, for each unbounded variable:
   - Optimize variable to find bound (Max for upper bounds, Min for lower bounds)
   - If optimization fails (infeasible/unbounded), restore model state and return false immediately
   - If optimization succeeds, extract and store the bound
   - Apply appropriate rounding for integer variables
6. If all optimizations succeed, restore original objective and integrality constraints
7. Return true indicating successful bound tightening
"""
function run_obbt!(upper_bounds_int::Union{Vector{Z}, BitVector}, lower_bounds_int::Union{Vector{Z}, BitVector}, upper_bounds_cont::Vector{T}, lower_bounds_cont::Vector{T}, model::JuMP.Model, integer_vars::Vector{JuMP.VariableRef}, continuous_vars::Vector{JuMP.VariableRef}; tolerance::U=1e-6) where {Z<:Integer, T<:Real, U<:AbstractFloat}
    # Check if any variables are unbounded before running LP relaxation
    has_unbounded_vars = any(ub == typemax(Int) for ub in upper_bounds_int) ||
                        any(lb == typemin(Int) for lb in lower_bounds_int) ||
                        any(ub == typemax(T) for ub in upper_bounds_cont) ||
                        any(lb == typemin(T) for lb in lower_bounds_cont)

    if !has_unbounded_vars
        return true, true
    end

    undo_relaxation = relax_integrality(model)
    optimize!(model)

    if is_solved_and_feasible(model)
         #get the objective function and sense to replace it later
        obj_func = objective_function(model)
        sense = objective_sense(model)
    
        @inbounds for (i,var) in enumerate(integer_vars)
            if upper_bounds_int[i] == typemax(Int)
                @objective(model, Max, var)
                optimize!(model)
                if termination_status(model) == OPTIMAL
                    raw_val = objective_value(model)
                    # Handle floating point precision errors: if very close to integer, round to it
                    if abs(raw_val - round(raw_val)) < tolerance
                        obj_val = Int(round(raw_val))
                    else
                        obj_val = Int(floor(raw_val))
                    end
                    upper_bounds_int[i] = obj_val
                else
                    # Unbounded, infeasible, or other failure - cannot bound this variable
                    @objective(model, sense,obj_func)
                    undo_relaxation()
                    return false, true
                end
            end

            if lower_bounds_int[i] == typemin(Int)
                @objective(model, Min, var)
                optimize!(model)
                if termination_status(model) == OPTIMAL
                    raw_val = objective_value(model)
                    # Handle floating point precision errors: if very close to integer, round to it
                    if abs(raw_val - round(raw_val)) < tolerance
                        obj_val = Int(round(raw_val))
                    else
                        obj_val = Int(ceil(raw_val))
                    end
                    lower_bounds_int[i] = obj_val
                else
                    # Unbounded, infeasible, or other failure - cannot bound this variable
                    @objective(model, sense,obj_func)
                    undo_relaxation()
                    return false, true
                end
            end
        end

        @inbounds for (i,var) in enumerate(continuous_vars)
            if upper_bounds_cont[i] == typemax(T)
                @objective(model, Max, var)
                optimize!(model)
                if termination_status(model) == OPTIMAL
                    obj_val = objective_value(model)
                    upper_bounds_cont[i] = obj_val
                else
                    # Unbounded, infeasible, or other failure - cannot bound this variable
                    @objective(model, sense,obj_func)
                    undo_relaxation()
                    return false, true
                end
            end

            if lower_bounds_cont[i] == typemin(T)
                @objective(model, Min, var)
                optimize!(model)
                if termination_status(model) == OPTIMAL
                    obj_val = objective_value(model)
                    lower_bounds_cont[i] = obj_val
                else
                    # Unbounded, infeasible, or other failure - cannot bound this variable
                    @objective(model, sense,obj_func)
                    undo_relaxation()
                    return false, true
                end
            end
        end

        @objective(model, sense,obj_func)
        undo_relaxation()
        return true, true
    else
        # Only an INFEASIBLE status proves the relaxation infeasible
        relaxation_feasible = termination_status(model) != INFEASIBLE
        undo_relaxation()
        return false, relaxation_feasible
    end
end


"""
    get_var_refs(model::JuMP.Model)

Separates all variables in the given JuMP model into integer/binary variables and continuous variables. Iterates through all variables in the model and categorizes them based on their type using JuMP's `is_integer()` and `is_binary()` functions.

# Arguments
- `model::JuMP.Model`: A JuMP model containing mixed-integer or linear programming formulation.

# Returns
- `Tuple{Vector{JuMP.VariableRef}, Vector{JuMP.VariableRef}}`: A tuple containing:
  - A vector of integer and binary variable references.
  - A vector of continuous variable references.
"""
function get_var_refs(model::JuMP.Model)
    # Collect integer (and binary) variable references
    integer_vars = Vector{JuMP.VariableRef}()
    continuous_vars = Vector{JuMP.VariableRef}()
    for var in all_variables(model)
        if is_integer(var) || is_binary(var)
            push!(integer_vars, var)
        else
            push!(continuous_vars, var)
        end
    end
    return integer_vars, continuous_vars
end

"""
    integer_upper_bound(bound::Real; tolerance::Real = 1e-6)

Converts a model upper bound on an integer variable to an Int.

Rounds down to the largest integer within the bound, snapping to the nearest integer when within `tolerance` of it to absorb floating point noise in the model.
Bounds outside the range of Int (e.g. 1e30 used as "infinity") are returned as the `typemax(Int)` sentinel so OBBT will try to find a finite bound.

# Arguments
- `bound::Real`: Upper bound from the model
- `tolerance::Real`: Distance to the nearest integer within which the bound is snapped to it (default: 1e-6)

# Returns
- `Int`: Integer upper bound, or `typemax(Int)` if the bound is not representable
"""
function integer_upper_bound(bound::Real; tolerance::Real = 1e-6)
    if !(typemin(Int) < bound < typemax(Int))
        return typemax(Int)
    end
    nearest = round(bound)
    return abs(bound - nearest) < tolerance ? Int(nearest) : floor(Int, bound)
end

"""
    integer_lower_bound(bound::Real; tolerance::Real = 1e-6)

Converts a model lower bound on an integer variable to an Int.

Rounds up to the smallest integer within the bound, snapping to the nearest integer when within `tolerance` of it to absorb floating point noise in the model.
Bounds outside the range of Int (e.g. -1e30 used as "-infinity") are returned as the `typemin(Int)` sentinel so OBBT will try to find a finite bound.

# Arguments
- `bound::Real`: Lower bound from the model
- `tolerance::Real`: Distance to the nearest integer within which the bound is snapped to it (default: 1e-6)

# Returns
- `Int`: Integer lower bound, or `typemin(Int)` if the bound is not representable
"""
function integer_lower_bound(bound::Real; tolerance::Real = 1e-6)
    if !(typemin(Int) < bound < typemax(Int))
        return typemin(Int)
    end
    nearest = round(bound)
    return abs(bound - nearest) < tolerance ? Int(nearest) : ceil(Int, bound)
end


"""
    extract_variable_bounds(integer_vars::Vector{JuMP.VariableRef}, continuous_vars::Vector{JuMP.VariableRef}, numerical_precision::DataType)

Extracts bounds from JuMP variables, using reasonable defaults when bounds are not specified.

# Arguments
- `integer_vars::Vector{JuMP.VariableRef}`: Integer and binary variable references
- `continuous_vars::Vector{JuMP.VariableRef}`: Continuous variable references
- `numerical_precision::DataType`: Precision for continuous variable bounds

# Returns
- `Tuple{Vector{Int}, Vector{Int}, Vector{T}, Vector{T}}`: A tuple containing:
  1. `Vector{Int}`: Upper bounds for integer variables
  2. `Vector{Int}`: Lower bounds for integer variables
  3. `Vector{T}`: Upper bounds for continuous variables
  4. `Vector{T}`: Lower bounds for continuous variables
"""
function extract_variable_bounds(integer_vars::Vector{JuMP.VariableRef}, continuous_vars::Vector{JuMP.VariableRef}, numerical_precision::DataType)
    # Initialize bound vectors
    upper_bounds_int = Vector{Int}(undef, length(integer_vars))
    lower_bounds_int = Vector{Int}(undef, length(integer_vars))
    upper_bounds_cont = Vector{numerical_precision}(undef, length(continuous_vars))
    lower_bounds_cont = Vector{numerical_precision}(undef, length(continuous_vars))

    # Extract integer variable bounds
    @inbounds for (i, var) in enumerate(integer_vars)
        # JuMP reports no bounds for fixed variables, so read the fixed value directly
        if is_fixed(var)
            upper_bounds_int[i] = integer_upper_bound(fix_value(var))
        elseif has_upper_bound(var)
            upper_bounds_int[i] = integer_upper_bound(upper_bound(var))
        else
            upper_bounds_int[i] = typemax(Int)
        end
        if is_binary(var) && upper_bounds_int[i] > 1
            upper_bounds_int[i] = 1
        end

        if is_fixed(var)
            lower_bounds_int[i] = integer_lower_bound(fix_value(var))
        elseif has_lower_bound(var)
            lower_bounds_int[i] = integer_lower_bound(lower_bound(var))
        else
            lower_bounds_int[i] = typemin(Int)
        end
        if is_binary(var) && lower_bounds_int[i] < 0
            lower_bounds_int[i] = 0
        end
    end

    # Extract continuous variable bounds
    @inbounds for (i, var) in enumerate(continuous_vars)
        if is_fixed(var)
            upper_bounds_cont[i] = numerical_precision(fix_value(var))
        elseif has_upper_bound(var)
            upper_bounds_cont[i] = numerical_precision(upper_bound(var))
        else
            upper_bounds_cont[i] = typemax(numerical_precision)
        end

        if is_fixed(var)
            lower_bounds_cont[i] = numerical_precision(fix_value(var))
        elseif has_lower_bound(var)
            lower_bounds_cont[i] = numerical_precision(lower_bound(var))
        else
            lower_bounds_cont[i] = typemin(numerical_precision)
        end
    end

    return upper_bounds_int, lower_bounds_int, upper_bounds_cont, lower_bounds_cont
end
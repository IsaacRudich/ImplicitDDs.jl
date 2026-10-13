"""
COMMENTS UP TO DATE
"""

"""
    create_LP_subproblem_model(coefficient_matrix_cont_cols::Matrix{T}, cont_obj_coeffs::Vector{T}, lbs_cont::Vector{T}, ubs_cont::Vector{T}, infimum_gaps::Vector{T}, cont_inf_gap_ctrbtns::Vector{T}, num_cont_vars::Int, num_constraints::Int) where {T<:Real}

Creates an LP model for optimizing continuous variables with constraint RHS values based on infimum gaps.

# Arguments
- `coefficient_matrix_cont_cols::Matrix{T}`: Coefficient matrix for continuous variables by constraints
- `cont_obj_coeffs::Vector{T}`: Objective coefficients for continuous variables
- `lbs_cont::Vector{T}`: Lower bounds for continuous variables
- `ubs_cont::Vector{T}`: Upper bounds for continuous variables
- `infimum_gaps::Vector{T}`: Initial infimum gap values representing constraint slack
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints

# Returns
- `Tuple{JuMP.Model, Vector{JuMP.VariableRef}, Vector{JuMP.ConstraintRef}}`: A tuple containing:
  1. `JuMP.Model`: Configured LP model for continuous optimization
  2. `Vector{JuMP.VariableRef}`: References to continuous variable objects
  3. `Vector{JuMP.ConstraintRef}`: References to constraint objects for RHS updates
"""
function create_LP_subproblem_model(
    coefficient_matrix_cont_cols::Matrix{T},
    cont_obj_coeffs::Vector{T},
    lbs_cont::Vector{T},
    ubs_cont::Vector{T},
    infimum_gaps::Vector{T},
    cont_inf_gap_ctrbtns::Vector{T},
    num_cont_vars::Int,
    num_constraints::Int
) where {T<:Real}

    # Handle case with no continuous variables
    if num_cont_vars == 0
        # Return empty model with proper types
        empty_model = Model(HiGHS.Optimizer)
        set_silent(empty_model)
        empty_vars = Vector{VariableRef}()
        empty_constraints = Vector{ConstraintRef}()
        return empty_model, empty_vars, empty_constraints
    end

    # Create the model
    LP_model = Model(HiGHS.Optimizer)
    set_silent(LP_model)

    set_attribute(LP_model, "presolve", "on")
    set_attribute(LP_model, "solver", "choose")
    # set_attribute(LP_model, "solver", "simplex")
    set_attribute(LP_model, "log_to_console", false)
    set_attribute(LP_model, "output_flag", false)

    # Add continuous variables with bounds
    @variable(LP_model, lbs_cont[i] <= x[i=1:num_cont_vars] <= ubs_cont[i])

    # Store constraint references
    constraint_refs = Vector{ConstraintRef}(undef, num_constraints)

    # Add constraints using adjusted RHS values
    for row_idx in 1:num_constraints
        # Build constraint: sum(coeff * x) <= adjusted_rhs
        constraint_expr = sum(coefficient_matrix_cont_cols[row_idx, col_idx] * x[col_idx] for col_idx in 1:num_cont_vars)
        
        # This represents the gap remaining after integer variables are fixed at their current values
        adjusted_rhs = infimum_gaps[row_idx] - cont_inf_gap_ctrbtns[row_idx]
        
        constraint_refs[row_idx] = @constraint(LP_model, constraint_expr <= adjusted_rhs)
    end

    # Set objective: minimize sum(cont_obj_coeffs * x)
    @objective(LP_model, Min, sum(cont_obj_coeffs[i] * x[i] for i in 1:num_cont_vars))

    return LP_model, x, constraint_refs
end

"""
    compute_tighter_continuous_bound!(
        path::Union{Vector{Z}, BitVector}, coefficient_matrix_int_cols::Matrix{T},
        coefficient_matrix_rhs_vector::Vector{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}},
        int_var_to_neg_rows::Dict{Int, Vector{Int}}, lbs_int::Union{Vector{Z}, BitVector},
        ubs_int::Union{Vector{Z}, BitVector}, num_int_vars::Int, num_cont_vars::Int,
        num_constraints::Int, first_paired_row::Int, lp_sub_model::JuMP.Model,
        lp_constraint_refs::Vector{JuMP.ConstraintRef}, rough_bounds_cont_val::T,
        adjusted_gaps::Vector{T}, timing_stats::TimingStats
    ) where {Z<:Integer, T<:Real}

Computes a tighter bound on the contribution of the continuous variables given a set of fixed integer variables.

# Arguments
- `path::Union{Vector{Z}, BitVector}`: Fixed integer variable values for the first variables
- `coefficient_matrix_int_cols::Matrix{T}`: Coefficient matrix for integer variables by constraints
- `coefficient_matrix_rhs_vector::Vector{T}`: Right-hand side values for constraints
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `num_int_vars::Int`: Total number of integer variables
- `num_constraints::Int`: Number of constraints
- `first_paired_row::Int`: Index of the first paired row (each == constraint is stored as a pair of rows)
- `lp_sub_model::JuMP.Model`: Existing LP model for continuous variable optimization
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to LP constraint objects for RHS updates
- `rough_bounds_cont_val::T`: Fallback bound value if LP is infeasible
- `adjusted_gaps::Vector{T}`: Preallocated working vector for gap computations
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations

# Returns
- `T`: Bound on continuous variable contribution to the objective function, by accounting for the partial integer assignments
"""
function compute_tighter_continuous_bound!(
    path                         ::Union{Vector{Z}, BitVector},
    coefficient_matrix_int_cols  ::Matrix{T},
    coefficient_matrix_rhs_vector::Vector{T},
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    lbs_int                      ::Union{Vector{Z}, BitVector},
    ubs_int                      ::Union{Vector{Z}, BitVector},
    num_int_vars                 ::Int,
    num_cont_vars                ::Int,
    num_constraints              ::Int,
    first_paired_row             ::Int,
    lp_sub_model                 ::JuMP.Model,
    lp_constraint_refs           ::Vector{JuMP.ConstraintRef},
    rough_bounds_cont_val        ::T,
    adjusted_gaps                ::Vector{T},
    timing_stats::TimingStats
) where {Z<:Integer, T<:Real}
    
    # Handle case with no continuous variables
    if num_cont_vars == 0
        return rough_bounds_cont_val
    end
    
    #compute adjusted infimum gaps for this partial assignment using preallocated vector
    adjusted_gaps .= coefficient_matrix_rhs_vector
    
    #number of fixed integer variables
    num_fixed = length(path)
    
    @inbounds begin
        #subtract contributions from fixed integer variables using their actual values
        for i in 1:num_fixed
            var_coefficients = @view coefficient_matrix_int_cols[:, i]
            pos_coef_inds = int_var_to_pos_rows[i]
            neg_coef_inds = int_var_to_neg_rows[i]
            
            fixed_value = path[i]
            
            @simd for row_ind in pos_coef_inds
                adjusted_gaps[row_ind] -= fixed_value * var_coefficients[row_ind]
            end
            @simd for row_ind in neg_coef_inds
                adjusted_gaps[row_ind] -= fixed_value * var_coefficients[row_ind]
            end
        end
        
        #subtract bound-based contributions from remaining unfixed integer variables
        for i in (num_fixed + 1):num_int_vars
            var_coefficients = @view coefficient_matrix_int_cols[:, i]
            pos_coef_inds = int_var_to_pos_rows[i]
            neg_coef_inds = int_var_to_neg_rows[i]
            
            lb = lbs_int[i]
            ub = ubs_int[i]
            
            #use bound that maximizes slack (minimizes constraint LHS)
            @simd for row_ind in pos_coef_inds
                adjusted_gaps[row_ind] -= lb * var_coefficients[row_ind]
            end
            @simd for row_ind in neg_coef_inds
                adjusted_gaps[row_ind] -= ub * var_coefficients[row_ind]
            end
        end
    end
    
    #update constraint RHS values in the existing LP model
    set_LP_rhs!(adjusted_gaps, first_paired_row, num_constraints, lp_constraint_refs)

    #solve the LP
    @time_operation timing_stats post_relaxed_dd_true_LP begin
        optimize!(lp_sub_model)
    end
    timing_stats.simplex_iterations += MOI.get(lp_sub_model, MOI.SimplexIterations())
    
    if termination_status(lp_sub_model) == MOI.OPTIMAL
        return T(objective_value(lp_sub_model))
    else
        #if infeasible, return a conservative bound
        return rough_bounds_cont_val
    end
end


"""
    resolve_continuous_variables!(model::JuMP.Model, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T}, int_var_refs::Vector{JuMP.VariableRef}, cont_var_refs::Vector{JuMP.VariableRef}, num_int_vars::Int, num_cont_vars::Int, sense_multiplier::Int, bkv::T, suppress_all_prints::Bool) where {Z<:Integer, T<:Real}

Fixes integer variables to solver solution values and resolves the LP for optimal continuous variable values.

# Arguments
- `model::JuMP.Model`: Original JuMP model with all variables and constraints
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector
- `bks_cont::Vector{T}`: Best known continuous solution vector (modified in-place with optimal continuous values)
- `int_var_refs::Vector{JuMP.VariableRef}`: References to integer variables in the model
- `cont_var_refs::Vector{JuMP.VariableRef}`: References to continuous variables in the model
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `sense_multiplier::Int`: Multiplier for objective sense conversion (+1 for minimization, -1 for maximization)
- `bkv::T`: Current best known objective value
- `suppress_all_prints::Bool`: Whether to suppress print statements

# Returns
- `T`: Updated optimal objective value after LP resolution
"""
function resolve_continuous_variables!(
    model::JuMP.Model,
    bks_int::Union{Vector{Z}, BitVector},
    bks_cont::Vector{T},
    int_var_refs::Vector{JuMP.VariableRef},
    cont_var_refs::Vector{JuMP.VariableRef},
    num_int_vars::Int,
    num_cont_vars::Int,
    sense_multiplier::Int,
    bkv::T,
    suppress_all_prints::Bool
) where {Z<:Integer, T<:Real}
    if num_cont_vars > 0
        if !suppress_all_prints
            print("\n","Re-solving continuous variables with fixed integers to remove rounding error ... ")
        end

        # Fix integer variables to solver solution values
        # fix(...; force = true) deletes variable bounds and unfix does not restore them,
        # so save them first to leave the caller's model unchanged.
        # Variables already fixed in the model must take that value, so they are left untouched.
        original_lbs = Union{Nothing, Float64}[has_lower_bound(var) ? lower_bound(var) : nothing for var in int_var_refs]
        original_ubs = Union{Nothing, Float64}[has_upper_bound(var) ? upper_bound(var) : nothing for var in int_var_refs]
        was_fixed = is_fixed.(int_var_refs)
        @inbounds for i in 1:num_int_vars
            if !was_fixed[i]
                fix(int_var_refs[i], bks_int[i]; force = true)
            end
        end

        # Solve the LP with fixed integers
        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            # Update continuous variables in solution
            @inbounds for i in 1:num_cont_vars
                bks_cont[i] = value(cont_var_refs[i])
            end

            # Update objective value
            bkv = T(objective_value(model) * sense_multiplier)  # Convert back to minimization form

            if !suppress_all_prints
                println("Finished")
            end
        else
            if !suppress_all_prints
                println("\n","WARNING: LP re-solve failed with status: ", termination_status(model))
            end
        end

        # Unfix integer variables and restore their original bounds
        @inbounds for i in 1:num_int_vars
            if !was_fixed[i]
                var = int_var_refs[i]
                unfix(var)
                !isnothing(original_lbs[i]) && set_lower_bound(var, original_lbs[i])
                !isnothing(original_ubs[i]) && set_upper_bound(var, original_ubs[i])
            end
        end
    end

    return bkv
end


"""
    compute_path_rhs!(rhs_buffer::Vector{U}, path::Union{Vector{Z}, BitVector}, coefficient_matrix_int_cols::Matrix{U}, coefficient_matrix_rhs_vector::Vector{U}, num_int_vars::Int, num_constraints::Int) where {Z<:Integer, U<:Real}

Computes the exact right-hand side of the continuous subproblem for a complete integer path, `b - A_int * path`.

The gaps stored in the DDs are clamped to be non-negative, so the right-hand side derived from them can be looser
than this one when the integer part slightly violates a constraint.

# Arguments
- `rhs_buffer::Vector{U}`: Preallocated working vector of length num_constraints, overwritten with the right-hand side
- `path::Union{Vector{Z}, BitVector}`: Complete integer path (length num_int_vars)
- `coefficient_matrix_int_cols::Matrix{U}`: Coefficient matrix for integer variables
- `coefficient_matrix_rhs_vector::Vector{U}`: Right-hand side values for constraints
- `num_int_vars::Int`: Number of integer variables
- `num_constraints::Int`: Number of constraints
"""
function compute_path_rhs!(
    rhs_buffer                   ::Vector{U},
    path                         ::Union{Vector{Z}, BitVector},
    coefficient_matrix_int_cols  ::Matrix{U},
    coefficient_matrix_rhs_vector::Vector{U},
    num_int_vars                 ::Int,
    num_constraints              ::Int
) where {Z<:Integer, U<:Real}
    # Accumulate b - A_int * path one column at a time
    rhs_buffer .= coefficient_matrix_rhs_vector
    @inbounds for j in 1:num_int_vars
        val = path[j]
        val == 0 && continue
        col = view(coefficient_matrix_int_cols, :, j)
        @simd for i in 1:num_constraints
            rhs_buffer[i] -= col[i] * val
        end
    end
end


"""
    set_LP_rhs!(lp_rhs::Vector{U}, first_paired_row::Int, num_constraints::Int, lp_constraint_refs::Vector{JuMP.ConstraintRef}) where {U<:Real}

Sets the right-hand sides of the LP subproblem, first moving the two rows of any crossed == constraint to their midpoint.

# Arguments
- `lp_rhs::Vector{U}`: Right-hand side for each constraint (crossed pairs are overwritten)
- `first_paired_row::Int`: Index of the first paired row (each == constraint is stored as a pair of rows)
- `num_constraints::Int`: Number of constraints
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to LP constraint objects
"""
function set_LP_rhs!(
    lp_rhs              ::Vector{U},
    first_paired_row    ::Int,
    num_constraints     ::Int,
    lp_constraint_refs  ::Vector{JuMP.ConstraintRef}
) where {U<:Real}
    # The right-hand sides of an == pair can only sum below zero through rounding error
    @inbounds for row_idx in first_paired_row:2:num_constraints
        if lp_rhs[row_idx] + lp_rhs[row_idx + 1] < 0
            lp_rhs[row_idx] = (lp_rhs[row_idx] - lp_rhs[row_idx + 1]) / 2
            lp_rhs[row_idx + 1] = -lp_rhs[row_idx]
        end
    end

    @inbounds for row_idx in 1:num_constraints
        set_normalized_rhs(lp_constraint_refs[row_idx], lp_rhs[row_idx])
    end
end


"""
    run_path_LP!(rhs::Vector{U}, num_constraints::Int, lp_sub_model::JuMP.Model, lp_constraint_refs::Vector{JuMP.ConstraintRef}, timing_stats::TimingStats) where {U<:Real}

Solves the continuous subproblem with the given right-hand side, typically the exact one from `compute_path_rhs!`.
Callers time the call with their own LP timer, since the restricted and relaxed DDs track LP time separately.

# Arguments
- `rhs::Vector{U}`: Right-hand side for each constraint
- `num_constraints::Int`: Number of constraints
- `lp_sub_model::JuMP.Model`: LP model for continuous variables
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to LP constraint objects
- `timing_stats::TimingStats`: Timing statistics tracker (simplex iterations are added to it)

# Returns
- `Tuple{Bool, U}`: A tuple containing:
  1. `Bool`: True if the LP was solved to optimality, false otherwise
  2. `U`: Objective contribution from continuous variables (zero if not optimal)
"""
function run_path_LP!(
    rhs                 ::Vector{U},
    num_constraints     ::Int,
    lp_sub_model        ::JuMP.Model,
    lp_constraint_refs  ::Vector{JuMP.ConstraintRef},
    timing_stats        ::TimingStats
) where {U<:Real}
    @inbounds for row_idx in 1:num_constraints
        set_normalized_rhs(lp_constraint_refs[row_idx], rhs[row_idx])
    end

    optimize!(lp_sub_model)
    timing_stats.simplex_iterations += MOI.get(lp_sub_model, MOI.SimplexIterations())

    if termination_status(lp_sub_model) == MOI.OPTIMAL
        return true, U(objective_value(lp_sub_model))
    end
    return false, zero(U)
end
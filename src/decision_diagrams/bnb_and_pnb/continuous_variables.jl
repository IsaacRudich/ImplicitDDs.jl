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
        num_constraints::Int, lp_sub_model::JuMP.Model,
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
    @inbounds for row_idx in 1:num_constraints
        new_rhs = adjusted_gaps[row_idx]
        set_normalized_rhs(lp_constraint_refs[row_idx], new_rhs)
    end
    
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
        @inbounds for i in 1:num_int_vars
            fix(int_var_refs[i], bks_int[i]; force = true)
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

        # Unfix integer variables for future use
        @inbounds for i in 1:num_int_vars
            unfix(int_var_refs[i])
        end
    end

    return bkv
end
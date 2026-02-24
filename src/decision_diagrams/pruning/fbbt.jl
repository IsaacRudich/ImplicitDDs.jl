"""
COMMENTS UP TO DATE
"""

"""
    fbbt_bound_tightening!(ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector}, ubs_cont::Vector{T}, lbs_cont::Vector{T}, coefficient_matrix_int_cols::Matrix{T}, coefficient_matrix_cont_cols::Matrix{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, int_var_to_zero_rows::Dict{Int, Vector{Int}}, cont_var_to_pos_rows::Dict{Int, Vector{Int}}, cont_var_to_neg_rows::Dict{Int, Vector{Int}}, infimum_gaps::Vector{T}, cont_inf_gap_ctrbtns::Vector{T}, num_int_vars::Int, num_cont_vars::Int, gap_adjustments::Array{T,3}; max_iterations::Int = 10, tolerance::U = 1e-3) where {Z<:Integer, T<:Real, U<:AbstractFloat}

Performs Feasibility-Based Bound Tightening (FBBT) to strengthen variable bounds through constraint-based propagation for mixed-integer problems.

# Algorithm
For each constraint `A_{iS}x_S + a_{ik}x_k ≤ b_i`:
1. Use precomputed infimum gaps: `gap_i = b_i - inf{A_{iS}x_S}` from solver
2. Derive bounds:
   - If `a_{ik} > 0`: `x_k ≤ gap_i/a_{ik} + current_lower_bound` → apply floor for integers
   - If `a_{ik} < 0`: `x_k ≥ gap_i/a_{ik} + current_upper_bound` → apply ceiling for integers
3. Update infimum gaps in real-time as bounds change for immediate propagation
4. Iterate until convergence or maximum iterations reached

# Arguments
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables (modified in-place)
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (modified in-place)
- `ubs_cont::Vector{T}`: Upper bounds for continuous variables (modified in-place)
- `lbs_cont::Vector{T}`: Lower bounds for continuous variables (modified in-place)
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables (constraints * variables)
- `coefficient_matrix_cont_cols::Matrix{T}`: Constraint coefficient matrix for continuous variables (constraints * variables)
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `int_var_to_zero_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with zero coefficients (for gap_adjustments population)
- `cont_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from continuous variables to constraint rows with positive coefficients
- `cont_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from continuous variables to constraint rows with negative coefficients
- `infimum_gaps::Vector{T}`: Precomputed infimum gaps from solver (RHS - constraint infimums) (modified in-place to reflect tightened bounds)
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps (modified in-place alongside infimum_gaps for mathematical consistency in LP RHS calculations)
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `gap_adjustments::Array{T,3}`: Precomputed gap adjustment lookup table [num_constraints, max_domain_size, num_int_vars] (populated at end of FBBT with final tightened bounds)
- `max_iterations::Int`: Maximum number of propagation iterations (default: 10)
- `tolerance::U`: Minimum bound improvement to continue iterations for continuous variables (default: 1e-3)

# Returns
- `Tuple{Bool, Bool, Int}`: A tuple containing:
  1. `Bool`: True if the problem is feasible, false if infeasible
  2. `Bool`: True if any bounds were tightened, false otherwise
  3. `Int`: Number of iterations performed before convergence

# Mathematical Foundation
The algorithm exploits precomputed constraint states for efficient propagation:
- Uses solver's infimum gaps: `gap_i = RHS_i - inf{∑ⱼ aᵢⱼxⱼ}` computed once
- Real-time gap updates: `new_gap = old_gap + (old_bound - new_bound) * coeff`
- Bound derivation: `x_k ≤ gap_i/a_{ik} + bound_complement` for constraint isolation
- Integer variables: floor/ceiling operations preserve discrete feasibility
- Continuous variables: tolerance-based iteration control prevents micro-improvements
- **Critical**: Both `infimum_gaps` and `cont_inf_gap_ctrbtns` must be updated synchronously when continuous bounds change to maintain mathematical consistency for subsequent LP RHS calculations
"""
function fbbt_bound_tightening!(
    ubs_int                      ::Union{Vector{Z}, BitVector},
    lbs_int                      ::Union{Vector{Z}, BitVector},
    ubs_cont                     ::Vector{T},
    lbs_cont                     ::Vector{T},
    coefficient_matrix_int_cols  ::Matrix{T},
    coefficient_matrix_cont_cols ::Matrix{T},
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    int_var_to_zero_rows         ::Dict{Int, Vector{Int}},
    cont_var_to_pos_rows         ::Dict{Int, Vector{Int}},
    cont_var_to_neg_rows         ::Dict{Int, Vector{Int}},
    infimum_gaps                 ::Vector{T},
    cont_inf_gap_ctrbtns         ::Vector{T},
    num_int_vars                 ::Int,
    num_cont_vars                ::Int,
    gap_adjustments              ::Array{T,3};
    max_iterations               ::Int = 10,
    tolerance                    ::U = 1e-3
) where {Z<:Integer, T<:Real, U<:AbstractFloat}

    # Determine the integer type for floor/ceil operations
    IntType = ubs_int isa BitVector ? Bool : Z

    bounds_improved = false
    iteration = 0

    for iter in 1:max_iterations
        iteration = iter
        iter_bounds_improved = false

        # Step a: Use infimum gaps directly (already in perfect format)
        # infimum_gaps[i] = RHS[i] - inf{∑ⱼ aᵢⱼxⱼ} = "remaining budget" in constraint i

        # Process integer variables
        @inbounds for var_idx in 1:num_int_vars
            # Step b: Adjust gaps by removing this variable's contribution
            current_bound_lower = lbs_int[var_idx]  # Use lower bound for positive coeffs
            current_bound_upper = ubs_int[var_idx]  # Use upper bound for negative coeffs

            # Get constraint rows where this variable appears
            pos_rows = int_var_to_pos_rows[var_idx]
            neg_rows = int_var_to_neg_rows[var_idx]

            # Step c: Derive bounds from positive coefficient constraints (upper bounds)
            for row_idx in pos_rows
                # Add back this variable's contribution to get gap excluding this variable
                # Compute potential upper bounds: x_k ≤ adjusted_gap / coeff
                potential_upper_bound = (infimum_gaps[row_idx] / coefficient_matrix_int_cols[row_idx, var_idx]) + current_bound_lower

                if potential_upper_bound < current_bound_upper
                    current_bound_upper = potential_upper_bound
                end
            end

            if ubs_int[var_idx] != current_bound_upper
                # Check if new bound would be infeasible before trying to convert
                new_upper_bound_real = floor(current_bound_upper)
                if new_upper_bound_real < lbs_int[var_idx]
                    # Infeasible: upper bound is less than lower bound
                    return false, bounds_improved, iteration
                end

                # Update infimum gaps before overwriting old bound
                new_upper_bound = floor(IntType, current_bound_upper)
                for row_idx in neg_rows  # negative coefficients use upper bounds
                    infimum_gaps[row_idx] += coefficient_matrix_int_cols[row_idx, var_idx] * (ubs_int[var_idx] - new_upper_bound)
                end
                ubs_int[var_idx] = new_upper_bound
                current_bound_upper = new_upper_bound  # Update for lower bound calculations
                iter_bounds_improved = true
            end

            # Step c: Derive bounds from negative coefficient constraints (lower bounds)
            for row_idx in neg_rows
                # Compute potential lower bounds: x_k ≥ adjusted_gap / coeff
                potential_lower_bound = (infimum_gaps[row_idx] / coefficient_matrix_int_cols[row_idx, var_idx]) + current_bound_upper

                if potential_lower_bound > current_bound_lower
                    current_bound_lower = potential_lower_bound
                end
            end

            if lbs_int[var_idx] != current_bound_lower
                # Check if new bound would be infeasible before trying to convert
                new_lower_bound_real = ceil(current_bound_lower)
                if new_lower_bound_real > ubs_int[var_idx]
                    # Infeasible: lower bound exceeds upper bound
                    return false, bounds_improved, iteration
                end

                # Update infimum gaps before overwriting old bound
                new_lower_bound = ceil(IntType, current_bound_lower)
                for row_idx in pos_rows  # positive coefficients use lower bounds
                    infimum_gaps[row_idx] += coefficient_matrix_int_cols[row_idx, var_idx] * (lbs_int[var_idx] - new_lower_bound)
                end
                lbs_int[var_idx] = new_lower_bound
                iter_bounds_improved = true
            end
        end

        # Process continuous variables
        @inbounds for var_idx in 1:num_cont_vars
            # Step b: Adjust gaps by removing this variable's contribution
            current_bound_lower = lbs_cont[var_idx]  # Use lower bound for positive coeffs
            current_bound_upper = ubs_cont[var_idx]  # Use upper bound for negative coeffs

            # Get constraint rows where this variable appears
            pos_rows = cont_var_to_pos_rows[var_idx]
            neg_rows = cont_var_to_neg_rows[var_idx]

            # Step c: Derive bounds from positive coefficient constraints (upper bounds)
            for row_idx in pos_rows
                # Add back this variable's contribution to get gap excluding this variable
                # Compute potential upper bounds: x_k ≤ adjusted_gap / coeff
                potential_upper_bound = (infimum_gaps[row_idx] / coefficient_matrix_cont_cols[row_idx, var_idx]) + current_bound_lower

                if potential_upper_bound < current_bound_upper
                    current_bound_upper = potential_upper_bound
                end
            end

            if ubs_cont[var_idx] != current_bound_upper
                # Update infimum gaps before overwriting old bound
                for row_idx in neg_rows  # negative coefficients use upper bounds
                    bound_change = coefficient_matrix_cont_cols[row_idx, var_idx] * (ubs_cont[var_idx] - current_bound_upper)
                    infimum_gaps[row_idx] += bound_change
                    cont_inf_gap_ctrbtns[row_idx] += bound_change
                end
                ubs_cont[var_idx] = current_bound_upper
                # Only set iter_bounds_improved if improvement exceeds tolerance
                if (ubs_cont[var_idx] - current_bound_upper) > tolerance
                    iter_bounds_improved = true
                end
            end

            # Step c: Derive bounds from negative coefficient constraints (lower bounds)
            for row_idx in neg_rows
                # Compute potential lower bounds: x_k ≥ adjusted_gap / coeff
                potential_lower_bound = (infimum_gaps[row_idx] / coefficient_matrix_cont_cols[row_idx, var_idx]) + current_bound_upper

                if potential_lower_bound > current_bound_lower
                    current_bound_lower = potential_lower_bound
                end
            end

            if lbs_cont[var_idx] != current_bound_lower
                # Update infimum gaps before overwriting old bound
                for row_idx in pos_rows  # positive coefficients use lower bounds
                    bound_change = coefficient_matrix_cont_cols[row_idx, var_idx] * (lbs_cont[var_idx] - current_bound_lower)
                    infimum_gaps[row_idx] += bound_change
                    cont_inf_gap_ctrbtns[row_idx] += bound_change
                end
                lbs_cont[var_idx] = current_bound_lower
                # Only set iter_bounds_improved if improvement exceeds tolerance
                if (current_bound_lower - lbs_cont[var_idx]) > tolerance
                    iter_bounds_improved = true
                end
            end

            # Check for infeasibility
            if lbs_cont[var_idx] > ubs_cont[var_idx]
                return false, bounds_improved, iteration
            end
        end

        # Check convergence
        if !iter_bounds_improved
            break
        else
            bounds_improved = true
        end
    end

    # Populate gap_adjustments with final tightened bounds
    @inbounds for var_idx in 1:num_int_vars
        lb = lbs_int[var_idx]
        ub = ubs_int[var_idx]

        # Positive coefficients: use lb
        @inbounds for row_idx in int_var_to_pos_rows[var_idx]
            coeff_row = coefficient_matrix_int_cols[row_idx, var_idx]
            @inbounds @simd for local_idx in 1:(ub - lb + 1)
                val = lb + local_idx - 1
                gap_adjustments[row_idx, local_idx, var_idx] = coeff_row * (lb - val)
            end
        end

        # Negative coefficients: use ub
        @inbounds for row_idx in int_var_to_neg_rows[var_idx]
            coeff_row = coefficient_matrix_int_cols[row_idx, var_idx]
            @inbounds @simd for local_idx in 1:(ub - lb + 1)
                val = lb + local_idx - 1
                gap_adjustments[row_idx, local_idx, var_idx] = coeff_row * (ub - val)
            end
        end

        # Zero coefficients: explicitly set to zero for unified loop processing
        @inbounds for row_idx in int_var_to_zero_rows[var_idx]
            @inbounds @simd for local_idx in 1:(ub - lb + 1)
                gap_adjustments[row_idx, local_idx, var_idx] = 0
            end
        end
    end

    return true, bounds_improved, iteration
end

"""
    compute_infimum_gaps_for_qnode!(local_gaps::Vector{T}, base_gaps::Vector{T}, qnode::QueueNode{Z,U}, coefficient_matrix_int_cols::Matrix{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, original_lbs_int::Union{Vector{Z}, BitVector}, original_ubs_int::Union{Vector{Z}, BitVector}, L::Int) where {Z<:Integer, T<:Real, U<:Real}

Computes the infimum gaps specific to a queue node's partial integer variable assignments, updating a vector directly.

This is a vector-based version of `compute_infimum_gaps_for_qnode!` designed for FBBT integration.
It adjusts baseline infimum gaps to account for fixed integer variable values in the queue node's path,
writing results directly to the provided gap vector.

# Arguments
- `local_gaps::Vector{T}`: Output vector to be filled with updated infimum gap values. Modified in-place.
- `base_gaps::Vector{T}`: Baseline infimum gaps computed with all variables at their bound-based minimal contributions
- `qnode::QueueNode{Z,U}`: Queue node containing partial integer variable assignments in its path
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `original_ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `L::Int`: Number of fixed integer variables in the queue node's path

# Algorithm
1. Copies baseline gaps into the local gap vector
2. For each fixed integer variable in the queue node path:
   - Adds back its bound-based contribution (lower bound for positive coefficients, upper bound for negative coefficients)
   - Subtracts the actual assigned value contribution
3. Continuous variable contributions remain unchanged from the baseline gaps
"""
@inline function compute_infimum_gaps_for_qnode!(
    local_gaps               ::Vector{T},                         # output (mutated)
    base_gaps                ::Vector{T},                          # original infimum_gaps
    qnode                    ::QueueNode{Z,U},                     # the node whose gaps we want
    coefficient_matrix_int_cols::Matrix{T},
    int_var_to_pos_rows      ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows      ::Dict{Int, Vector{Int}},
    original_lbs_int         ::Union{Vector{Z}, BitVector},
    original_ubs_int         ::Union{Vector{Z}, BitVector},
    L                        ::Int
) where {Z<:Integer, T<:Real, U<:Real}
    # 1) copy in the base
    @inbounds @simd for i in eachindex(base_gaps)
        local_gaps[i] = base_gaps[i]
    end

    # 2) for every fixed variable, add its bound‐contribution back into gaps
    # 3) for every fixed variable, subtract the assigned value
    @inbounds for j in 1:(L)
        original_lb, original_ub = original_lbs_int[j], original_ubs_int[j]
        val = qnode.path[j]
        @inbounds @simd for r in int_var_to_pos_rows[j]
            local_gaps[r] += (original_lb - val) * coefficient_matrix_int_cols[r, j]
        end
        @inbounds @simd for r in int_var_to_neg_rows[j]
            local_gaps[r] += (original_ub - val) * coefficient_matrix_int_cols[r, j]
        end
    end
end



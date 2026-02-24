"""
COMMENTS UP TO DATE
"""

"""
    compute_implied_column_bounds!(
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        var_index::Int, infimum_gap_matrices::Matrix{T},
        buffer_offset::Int, int_var_to_pos_rows::Dict{Int, Vector{Int}},
        int_var_to_neg_rows::Dict{Int, Vector{Int}},
        lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector},
        inv_coeff::Matrix{T}, nodes::NodeLayer{Z}
    ) where {Z<:Integer, T<:Real}

Computes implied variable domain bounds for each parent node using constraint propagation with optimized operations.

For each active node in the previous layer, uses the node's infimum gap values to determine
the feasible domain for the next variable. Applies constraint-based bound tightening with
integer rounding to ensure feasibility.

# Arguments
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix; column `var_index-1` updated in-place
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix; column `var_index-1` updated in-place
- `var_index::Int`: Index of the variable for which bounds are being computed
- `infimum_gap_matrices::Matrix{T}`: Double-buffered matrix containing constraint gap values
- `buffer_offset::Int`: Current buffer offset for accessing gap matrix columns
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Variable to positive coefficient rows mapping
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Variable to negative coefficient rows mapping
- `lbs_int::Union{Vector{Z}, BitVector}`: Global lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Global upper bounds for integer variables
- `inv_coeff::Matrix{T}`: Precomputed inverse coefficients [num_constraints, num_int_vars] for division→multiplication optimization
- `nodes::NodeLayer{Z}`: Previous layer nodes; inactive nodes get infeasible bounds

# Algorithm
For positive coefficients: `new_ub = floor((gap * inv_coeff) + lb)` (multiplication instead of division)
For negative coefficients: `new_lb = ceil((gap * inv_coeff) + ub)` (multiplication instead of division)
Uses branchless min/max operations and column views for cache locality.
Inactive nodes receive infeasible bounds (`lb > ub`).
"""
@inline function compute_implied_column_bounds!(
    lb_matrix               ::Union{Matrix{Z}, BitMatrix},
    ub_matrix               ::Union{Matrix{Z}, BitMatrix},
    var_index               ::Int,
    infimum_gap_matrices    ::Matrix{T},
    buffer_offset           ::Int,
    int_var_to_pos_rows     ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows     ::Dict{Int, Vector{Int}},
    lbs_int                 ::Union{Vector{Z}, BitVector},
    ubs_int                 ::Union{Vector{Z}, BitVector},
    inv_coeff               ::Matrix{T},
    nodes                   ::NodeLayer{Z};
    rounding_tol            ::T = T(1e-6),
) where {Z<:Integer, T<:Real}
    # Determine the integer type for floor/ceil operations
    IntType = lbs_int isa BitVector ? Bool : Z

    # Create non-allocating view for inv_coeff column for efficiency
    inv_col = @view inv_coeff[:, var_index]
    #and the bound column
    lb_col = @view lb_matrix[:, var_index-1]
    ub_col = @view ub_matrix[:, var_index-1]

    # Grab the overall lower/upper bounds for this variable
    lb = lbs_int[var_index]
    ub = ubs_int[var_index]

    pos_rows = int_var_to_pos_rows[var_index]
    neg_rows = int_var_to_neg_rows[var_index]

    #get the active/inactive states
    nodes_activation_states = nodes.active

    # For each node (which corresponds to a column in the infimum matrices), compute
    # how the constraints restrict the next variable's feasible region.
    @inbounds for i in 1:nodes.size
        if nodes_activation_states[i]
            # Create view for this column to improve cache locality
            inf_col = @view infimum_gap_matrices[:, i+buffer_offset]

            # Start from the global [lb, ub] and tighten as needed
            temp_lb = lb
            temp_ub = ub

            ###############
            # Constraints with a positive coefficient
            ###############
            # --- ≤ constraints ---
            @inbounds for row_index in pos_rows
                test_limit = (inf_col[row_index] * inv_col[row_index]) + lb
                temp_ub = min(temp_ub, test_limit)
            end
            ###############
            # Constraints with a negative coefficient
            ###############
            # --- ≤ constraints ---
            @inbounds for row_index in neg_rows
                test_limit = (inf_col[row_index] * inv_col[row_index]) + ub
                temp_lb = max(temp_lb, test_limit)
            end
            # Store the results for this column (tolerance handles inv_coeff precision loss)
            temp_ub = floor(IntType, temp_ub + rounding_tol)
            temp_lb = ceil(IntType, temp_lb - rounding_tol)


            lb_col[i] = temp_lb
            ub_col[i] = temp_ub
        else
            lb_col[i] = 1
            ub_col[i] = 0
        end
    end
end





"""
    update_layer_ltr!(
        ltr_matrix::Matrix{U}, node_matrix::Vector{NodeLayer{Z}}, var_index::Int,
        coeff_times_val::Matrix{U}, original_lbs_int::Union{Vector{Z}, BitVector},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix}
    ) where {Z<:Integer, U<:Real}

Updates the LTR matrix for the current layer by finding optimal feasible parents for each node.

For each node in the current layer, scans its parent arc range `[first_arc, last_arc]` to find
the parent with minimum LTR value whose bounds allow the node's variable value. Updates the
node's LTR as the best parent cost plus the node's objective contribution using precomputed
coefficient products.

# Arguments
- `ltr_matrix::Matrix{U}`: Length-to-root matrix; updated in-place for current layer
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers containing merged node information
- `var_index::Int`: Index of the current layer being processed
- `coeff_times_val::Matrix{U}`: Precomputed coefficient products [max_domain_size, num_int_vars] for arc objectives
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original global lower bounds for integer variables (before FBBT tightening) used for consistent array indexing into coeff_times_val
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for feasible parent-child transitions
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for feasible parent-child transitions

# Algorithm
Uses threaded processing to find `ltr[child] = min over feasible_parents (parent_ltr + coeff * child_value)`
where feasible parents satisfy `lb ≤ child_value ≤ ub`. Uses precomputed coefficient products to avoid
runtime multiplication.
"""
@inline function update_layer_ltr!(
    ltr_matrix          ::Matrix{U},
    node_matrix         ::Vector{NodeLayer{Z}},
    var_index           ::Int,
    coeff_times_val     ::Matrix{U},
    original_lbs_int    ::Union{Vector{Z}, BitVector},
    lb_matrix           ::Union{Matrix{Z}, BitMatrix},
    ub_matrix           ::Union{Matrix{Z}, BitMatrix}
) where {Z<:Integer, U<:Real}
    # Use @view or view(...) to avoid slicing allocations
    ltr_col         = @view ltr_matrix[:, var_index]
    ltr_parent_col  = @view ltr_matrix[:, var_index - 1]
    interval_lbs = view(lb_matrix, :, var_index-1)
    interval_ubs = view(ub_matrix, :, var_index-1)

    # Precomputed coefficient products for this variable
    coeff_col = @view coeff_times_val[:, var_index]
    original_lb = original_lbs_int[var_index]
    lb_offset = 1 - original_lb

    nodes_curr_layer = node_matrix[var_index]
    first_arcs = nodes_curr_layer.first_arcs
    last_arcs  = nodes_curr_layer.last_arcs
    values     = nodes_curr_layer.values

    # @inbounds Threads.@threads for i in 1:nodes_curr_layer.size
    @inbounds for i in 1:nodes_curr_layer.size
        first_arc = first_arcs[i]
        last_arc = last_arcs[i]
        nval = values[i]

        # Initialize best_val from the first parent index
        best_val = ltr_parent_col[first_arc]

        # Scan all parent indices to find the minimum feasible parent
        @inbounds @simd for parent_idx in (first_arc+1) : last_arc
            lb = interval_lbs[parent_idx]
            ub = interval_ubs[parent_idx]

            if nval >= lb && nval <= ub
                parent_val = ltr_parent_col[parent_idx]
                best_val = min(best_val, parent_val)
            end
        end

        # Incorporate the cost term using precomputed coefficient products
        val_idx = nval + lb_offset
        contribution = coeff_col[val_idx]
        ltr_col[i] = best_val + contribution
    end
end


"""
    compute_next_infimum_gap_matrix!(
        infimum_gap_matrices::Matrix{U}, buffer_offset::Int, w::Int,
        node_layers::Vector{NodeLayer{Z}}, var_index::Int,
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        coefficient_matrix_int_cols::Matrix{U},
        lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}
    ) where {Z<:Integer, U<:Real}

Updates the infimum gap matrix for the next layer using double-buffering and max-based relaxed updates.

This function propagates constraint state information from parent nodes to child nodes in relaxed DDs.
Unlike exact propagation, it uses max operations on the infimum gaps to maintain valid constraint relaxations.

# Arguments
- `infimum_gap_matrices::Matrix{U}`: Double-buffered matrix storing constraint gaps (modified in-place)
- `buffer_offset::Int`: Current buffer position (0 or w) for double-buffering
- `w::Int`: Maximum layer width determining buffer size
- `node_layers::Vector{NodeLayer{Z}}`: Decision diagram layers containing merged node information
- `var_index::Int`: Index of current variable layer being processed
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for feasible parent-child transitions
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for feasible parent-child transitions
- `coefficient_matrix_int_cols::Matrix{U}`: Constraint coefficient matrix for integer variables
- `lbs_int::Union{Vector{Z}, BitVector}`: Global lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Global upper bounds for integer variables
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Variable to positive coefficient rows mapping
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Variable to negative coefficient rows mapping

# Returns
- `Int`: New buffer offset for continued double-buffering

# Algorithm
1. **Buffer Selection**: Alternates between matrix halves for double-buffering
2. **Max Propagation**: For each child node, takes max over feasible parent gaps
3. **Bound Correction**: Adjusts gaps based on variable assignments vs. original bounds
"""
@inline function compute_next_infimum_gap_matrix!(
    infimum_gap_matrices        ::Matrix{U},
    buffer_offset               ::Int,
    w                           ::Int,
    node_layers                 ::Vector{NodeLayer{Z}},
    var_index                   ::Int,
    lb_matrix                   ::Union{Matrix{Z}, BitMatrix},
    ub_matrix                   ::Union{Matrix{Z}, BitMatrix},
    coefficient_matrix_int_cols ::Matrix{U},
    lbs_int                     ::Union{Vector{Z}, BitVector},
    ubs_int                     ::Union{Vector{Z}, BitVector},
    int_var_to_pos_rows         ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows         ::Dict{Int, Vector{Int}},
) where {Z<:Integer, U<:Real}
    # Decide which half of infimum_gap_matrices we'll write into
    new_buffer_offset = ifelse(buffer_offset == 0, w, 0)

    # Cache some sizes to avoid repeated size(...) calls
    row_count = size(infimum_gap_matrices, 1)

    # "intervals" is a view of (lb,ub) for each "parent_idx" in the previous layer
    interval_lbs = view(lb_matrix, :, var_index-1)
    interval_ubs = view(ub_matrix, :, var_index-1)

    # The current layer’s SoA
    current_nodes = node_layers[var_index]
    first_arcs    = current_nodes.first_arcs
    last_arcs     = current_nodes.last_arcs
    values        = current_nodes.values
    layer_size = current_nodes.size

    #get the active/inactive state of the nodes
    nodes_activation_state = current_nodes.active


    #---------------------------------------------------------------------------
    # 1) Zero out the "next_cols" region we will write to, with an explicit fill
    #---------------------------------------------------------------------------
    next_cols_start = new_buffer_offset + 1
    next_cols_end   = new_buffer_offset + layer_size

    fill!(view(infimum_gap_matrices, :, next_cols_start:next_cols_end), 0)

    # Coefficients for this variable (column)
    coeff_col = @view coefficient_matrix_int_cols[:, var_index]
    #---------------------------------------------------------------------------
    # 2) For each node, look at all its parents. If (lb <= val <= ub), then
    #    perform the per-row max update: 
    #
    #    next_cols[row,node_idx] = max(next_cols[row,node_idx], half_cols[row,parent_idx])
    #
    #    We map:
    #      next_cols[:, node_idx]  -> infimum_gap_matrices[:, new_buffer_offset + node_idx]
    #      half_cols[:, parent_idx]-> infimum_gap_matrices[:, buffer_offset + parent_idx]
    #---------------------------------------------------------------------------
    @inbounds for node_idx in 1:layer_size
        if nodes_activation_state[node_idx]
            val   = values[node_idx]
            f_arc = first_arcs[node_idx]
            l_arc = last_arcs[node_idx]
            cur_col_idx = new_buffer_offset + node_idx
            @inbounds for parent_idx in f_arc:l_arc
                lb = interval_lbs[parent_idx]
                ub = interval_ubs[parent_idx]
                if lb <= val && val <= ub
                    parent_col_idx = buffer_offset + parent_idx

                    @inbounds for row in 1:row_count
                        parent_val = infimum_gap_matrices[row, parent_col_idx]
                        if parent_val > infimum_gap_matrices[row, cur_col_idx]
                            infimum_gap_matrices[row, cur_col_idx] = parent_val
                        end
                    end
                end
            end
        end
    end

    #---------------------------------------------------------------------------
    # 3) Subtract out the per-state contribution, and add the initial bound contribution back in
    #---------------------------------------------------------------------------
    lb = lbs_int[var_index]
    ub = ubs_int[var_index]
    pos_rows = int_var_to_pos_rows[var_index]
    neg_rows = int_var_to_neg_rows[var_index]

    @inbounds @simd for node_idx in 1:layer_size
        if nodes_activation_state[node_idx]
            val = values[node_idx]
            lb_diff = lb - val
            ub_diff = ub - val
            col_idx = node_idx + new_buffer_offset
            gap_col = @view infimum_gap_matrices[:, col_idx]

            @inbounds @simd for row_index in pos_rows
                gap_col[row_index] += coeff_col[row_index] * lb_diff
            end
            @inbounds @simd for row_index in neg_rows
                gap_col[row_index] += coeff_col[row_index] * ub_diff
            end
        end
    end

    #---------------------------------------------------------------------------
    # 4) Clamp gaps to non-negative: compute_implied_column_bounds! validated feasibility,
    #    so any negative gaps are floating point artifacts (e.g., gap ≈ coeff yields -1e-7)
    #---------------------------------------------------------------------------
    @inbounds for node_idx in 1:layer_size
        if nodes_activation_state[node_idx]
            col_idx = node_idx + new_buffer_offset
            @inbounds for row in 1:row_count
                if infimum_gap_matrices[row, col_idx] < 0
                    infimum_gap_matrices[row, col_idx] = U(0)
                end
            end
        end
    end

    return new_buffer_offset
end


"""
    rough_bound_relaxed_layer!(node_matrix::Vector{NodeLayer{Z}}, cur_idx::Int, rough_bounds_int::Vector{T}, rough_bounds_cont_val::T, ltr_matrix::Matrix{T}, bkv::T) where {Z<:Integer, T<:Real}

Prunes dominated nodes in a relaxed DD layer using rough bound estimates.

Uses conservative rough bounds to identify nodes that cannot improve the best known value.
Deactivates such nodes via the BitVector rather than removing them, preserving layer structure.

# Arguments
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers containing nodes to prune
- `cur_idx::Int`: Current layer index being processed
- `rough_bounds_int::Vector{T}`: Rough bounds for remaining integer variables
- `rough_bounds_cont_val::T`: Rough bound estimate for continuous variables
- `ltr_matrix::Matrix{T}`: Length-to-root values for current paths
- `bkv::T`: Best known value for pruning comparison

# Algorithm
Computes `total_bound = ltr[node] + rough_bound_remaining` and deactivates nodes
where `total_bound ≥ bkv`. Uses `rough_bounds_int[cur_idx+1]` to avoid double-counting.
"""
@inline function rough_bound_relaxed_layer!(node_matrix::Vector{NodeLayer{Z}}, cur_idx::Int, rough_bounds_int::Vector{T}, rough_bounds_cont_val::T, ltr_matrix::Matrix{T}, bkv::T)where{Z<:Integer, T<:Real}
    rough_bound    = rough_bounds_int[cur_idx+1] + rough_bounds_cont_val
    current_layer  = node_matrix[cur_idx]
    cur_layer_size = current_layer.size
    ltr_col        = @view ltr_matrix[:, cur_idx]

    # Pre-compute threshold and hoist active BitVector access
    threshold = bkv - rough_bound
    active = current_layer.active

    @inbounds @simd for node_idx in 1:cur_layer_size
        if ltr_col[node_idx] >= threshold
            active[node_idx] = false
        end
    end
end
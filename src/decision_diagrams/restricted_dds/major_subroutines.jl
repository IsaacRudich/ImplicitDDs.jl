"""
COMMENTS UP TO DATE
"""

"""
    compute_implied_column_bounds!(lb_col::Union{Vector{Z}, BitVector}, ub_col::Union{Vector{Z}, BitVector}, var_index::Int, infimum_gap_matrix::Matrix{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector}, inv_coeff::Matrix{T}, size::Integer) where {Z<:Integer, T<:Real}

Computes the variable bounds for each node implied by their infimum gaps (slack in the constraints).

# Arguments
- `lb_col::Union{Vector{Z}, BitVector}`: Preallocated vector for performance efficiency - stores computed lower bounds for each node
- `ub_col::Union{Vector{Z}, BitVector}`: Preallocated vector for performance efficiency - stores computed upper bounds for each node
- `var_index::Int`: Index of the variable for which bounds are being computed
- `infimum_gap_matrix::Matrix{T}`: Matrix tracking constraint gap values for each node
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from variables to constraint rows with negative coefficients
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `inv_coeff::Matrix{T}`: Precomputed inverse coefficients [num_constraints, num_int_vars] for division→multiplication optimization
- `size::Integer`: Number of nodes in the current layer

# Algorithm
1. For each node, starts with global variable bounds [lb, ub]
2. Tightens bounds using constraint propagation:
   - For positive coefficients: uses ≤ constraints to tighten upper bounds
   - For negative coefficients: uses ≤ constraints to tighten lower bounds
3. Applies integer rounding (floor/ceil) to ensure integer feasibility

# Returns
- `Nothing`: Bounds are computed in-place and stored in lb_col and ub_col vectors
"""
@inline function compute_implied_column_bounds!(
    lb_col                  ::Union{Vector{Z}, BitVector},
    ub_col                  ::Union{Vector{Z}, BitVector},
    var_index               ::Int,
    infimum_gap_matrix      ::Matrix{T},
    int_var_to_pos_rows     ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows     ::Dict{Int, Vector{Int}},
    lbs_int                 ::Union{Vector{Z}, BitVector},
    ubs_int                 ::Union{Vector{Z}, BitVector},
    inv_coeff               ::Matrix{T},
    size                    ::Integer;
    rounding_tol            ::T = T(1e-6),
) where {Z<:Integer, T<:Real}
    # Determine the integer type for floor/ceil operations
    IntType = lbs_int isa BitVector ? Bool : Z

    # Create non-allocating view for inv_coeff column for efficiency
    inv_col = @view inv_coeff[:, var_index]

    # Grab the overall lower/upper bounds for this variable
    lb = lbs_int[var_index]
    ub = ubs_int[var_index]



    pos_rows = int_var_to_pos_rows[var_index]
    neg_rows = int_var_to_neg_rows[var_index]


    # For each node (which corresponds to a column in the infimum matrices), compute
    # how the constraints restrict the next variable's feasible region.
    # @inbounds Threads.@threads for i in 1:size
    @inbounds for i in 1:size
        # Create view for this column to improve cache locality
        infimum_col = @view infimum_gap_matrix[:, i]

        # Start from the global [lb, ub] and tighten as needed
        temp_lb = lb
        temp_ub = ub

        # Two-loop branchless approach for optimal performance
        # Process positive coefficient constraints
        @inbounds for row_index in pos_rows
            test_limit = (infimum_col[row_index] * inv_col[row_index]) + lb
            temp_ub = min(temp_ub, test_limit)
        end

        # Process negative coefficient constraints
        @inbounds for row_index in neg_rows
            test_limit = (infimum_col[row_index] * inv_col[row_index]) + ub
            temp_lb = max(temp_lb, test_limit)
        end
        # Store the results for this column (tolerance handles inv_coeff precision loss)
        temp_ub = floor(IntType, temp_ub + rounding_tol)
        temp_lb = ceil(IntType, temp_lb - rounding_tol)

        lb_col[i] = temp_lb
        ub_col[i] = temp_ub
    end
end


"""
    build_restricted_node_layer!(layer_a::RestrictedNodeLayer, layer_b::RestrictedNodeLayer, gap_a::Matrix{T}, gap_b::Matrix{T}, inv_coeff::T, var_index::Int, threshold_bin::Int, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_constraints::Int, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, bins_matrix::Matrix{<:Integer}, w::Int) where{Z<:Integer, T<:Real}

Constructs the next decision diagram layer by expanding arcs based on precomputed bin membership.

# Arguments
- `layer_a::RestrictedNodeLayer`: Parent layer containing nodes to expand from
- `layer_b::RestrictedNodeLayer`: Preallocated layer for performance efficiency - populated with child nodes
- `gap_a::Matrix{T}`: Gap buffer to read parent gap values from
- `gap_b::Matrix{T}`: Gap buffer to write child gap values to
- `inv_coeff::T`: Inverse of objective coefficient (1/coeff) - used to determine iteration direction for value reconstruction
- `var_index::Int`: Index of the current variable being processed
- `threshold_bin::Int`: Bin threshold - only arcs with bin ≤ threshold_bin are expanded
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (post-FBBT, for domain bounds and gap_adjustments indexing)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original lower bounds before any FBBT (for coeff_times_val indexing)
- `num_constraints::Int`: Number of constraints (for unified gap update loop)
- `coeff_times_val::Matrix{T}`: Precomputed arc objective contributions [max_domain_size, num_int_vars] (indexed with original bounds, populated once at startup)
- `gap_adjustments::Array{T,3}`: Precomputed gap adjustment values [num_constraints, max_domain_size, num_int_vars] (indexed with current bounds, repopulated by FBBT)
- `bins_matrix::Matrix{<:Integer}`: Precomputed arc-to-bin mapping [max_domain_size, w] with compact storage and -1 sentinels
- `w::Int`: Maximum width constraint for the new layer

# Returns
- `Nothing`: Layer_b and gap_b are populated in-place with nodes whose arc bin ≤ threshold_bin, up to width limit w
"""
@inline function build_restricted_node_layer!(
    layer_a::RestrictedNodeLayer,
    layer_b::RestrictedNodeLayer,
    gap_a::Matrix{T},
    gap_b::Matrix{T},
    inv_coeff::T,
    var_index::Int,
    threshold_bin::Int,
    uncapped::Bool,
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    num_constraints::Int,
    coeff_times_val::Matrix{T},
    gap_adjustments::Array{T,3},
    bins_matrix::Matrix{<:Integer},
    w::Int
) where{Z<:Integer, T<:Real}
   
    if threshold_bin == w
        # Handle normal case where all bins pass (rough bounds already applied)
        enumerate_all_arcs!(layer_a, layer_b, gap_a, gap_b, var_index, lbs_int, original_lbs_int, num_constraints, coeff_times_val, gap_adjustments, bins_matrix, inv_coeff)
    elseif threshold_bin == -1 && uncapped
        # Handle degenerate case where all arcs have identical objective and all fit
        enumerate_all_arcs_ignore_objective!(layer_a, layer_b, gap_a, gap_b, var_index, lbs_int, original_lbs_int, num_constraints, coeff_times_val, gap_adjustments)
    elseif threshold_bin == 0 && !uncapped
        # Handle leftmost bin too large - enumerate bin 1 arcs only
        enumerate_arcs_in_bin_one!(layer_a, layer_b, gap_a, gap_b, inv_coeff, var_index, lbs_int, original_lbs_int, num_constraints, coeff_times_val, gap_adjustments, bins_matrix, w)
    elseif threshold_bin == -1
        # Handle degenerate case where not all arcs fit
        enumerate_arcs_ignore_objective!(layer_a, layer_b, gap_a, gap_b, var_index, lbs_int, original_lbs_int, num_constraints, coeff_times_val, gap_adjustments, w)
    else
        # Normal bin-based filtering (1 <= threshold_bin < w)
        enumerate_arcs_using_hybrid_threshold!(layer_a, layer_b, gap_a, gap_b, inv_coeff, threshold_bin, var_index, lbs_int, original_lbs_int, num_constraints, coeff_times_val, gap_adjustments, bins_matrix, w)
    end

    # Clamp gaps to non-negative: compute_implied_column_bounds! already validated feasibility,
    # so any negative gaps are floating point artifacts (e.g., 1.79 + (-1.79) = -1e-7 instead of 0)
    clamp_gaps_to_nonnegative!(gap_b, num_constraints, layer_b.size)
end


"""
    clamp_gaps_to_nonnegative!(gap_matrix::Matrix{T}, num_constraints::Int, layer_size::Integer) where {T<:Real}

Clamps all gap values to non-negative after layer construction.

Since compute_implied_column_bounds! already validated that enumerated arcs are feasible,
any negative gaps are purely floating point artifacts (e.g., gap ≈ coeff and adjustment = -coeff
yields -1e-7 instead of exactly 0). This cleanup pass ensures gaps remain non-negative as
required by the mathematical model.
"""
@inline function clamp_gaps_to_nonnegative!(gap_matrix::Matrix{T}, num_constraints::Int, layer_size::Integer) where {T<:Real}
    @inbounds for node_idx in 1:layer_size
        @inbounds for row_idx in 1:num_constraints
            if gap_matrix[row_idx, node_idx] < 0
                gap_matrix[row_idx, node_idx] = T(0)
            end
        end
    end
end


"""
    enumerate_all_arcs!(layer_a::RestrictedNodeLayer, layer_b::RestrictedNodeLayer, gap_a::Matrix{T}, gap_b::Matrix{T}, var_index::Int, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_constraints::Int, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, bins_matrix::Matrix{<:Integer}, inv_coeff::T) where{Z<:Integer, T<:Real}

Helper function to enumerate all feasible arcs when total count is guaranteed ≤ w.

Uses bins_matrix sentinels to respect rough bound pruning applied during histogram phase.

Used for cases where all arcs fit: threshold_bin == w (all bins pass) or degenerate case with uncapped == true.
"""
@inline function enumerate_all_arcs!(
    layer_a::RestrictedNodeLayer,
    layer_b::RestrictedNodeLayer,
    gap_a::Matrix{T},
    gap_b::Matrix{T},
    var_index::Int,
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    num_constraints::Int,
    coeff_times_val::Matrix{T},
    gap_adjustments::Array{T,3},
    bins_matrix::Matrix{<:Integer},
    inv_coeff::T
) where{Z<:Integer, T<:Real}
    # Setup views and bounds
    coeff_times_val_col = @view coeff_times_val[:, var_index+1]
    gap_adjustments_col = @view gap_adjustments[:, :, var_index+1]
    lb = lbs_int[var_index+1]
    original_lb = original_lbs_int[var_index+1]
    lb_offset = 1 - lb
    original_lb_offset = 1 - original_lb

    ltrs   = layer_a.ltrs
    lb_col = layer_a.implied_lbs
    ub_col = layer_a.implied_ubs
    gap_matrix = gap_a

    empty_layer!(layer_b)
    new_values = layer_b.values

    # Define iteration direction based on coefficient sign (matches histogram logic)
    make_range = if inv_coeff > 0
        (start, count) -> (start:(start + count - 1))
    elseif inv_coeff < 0
        (start, count) -> (start:-1:(start - count + 1))
    else
        (start, count) -> (start:(start + count - 1))
    end
    start_selector = inv_coeff >= 0 ? lb_col : ub_col

    # Enumerate all feasible arcs (respecting rough bound pruning via sentinels)
    @inbounds for i in 1:layer_a.size
        lb_i = lb_col[i]
        ub_i = ub_col[i]

        # Skip infeasible nodes
        if ub_i < lb_i
            continue
        end

        # Find sentinel position to determine enumeration range
        arc_idx = 1
        @inbounds while bins_matrix[arc_idx, i] != -1
            arc_idx += 1
        end
        arcs_to_enumerate = arc_idx - 1

        base_ltr = ltrs[i]
        parent_gaps = @view gap_matrix[:, i]
        start_val = start_selector[i]

        # Enumerate exact range (no sentinel check in loop!)
        @inbounds for val in make_range(start_val, arcs_to_enumerate)
            # Compute indices (optimized: precomputed offsets)
            gap_idx = val + lb_offset
            coeff_idx = val + original_lb_offset

            # Compute arc objective
            arc_ltr = base_ltr + coeff_times_val_col[coeff_idx]

            # Add node
            add_node!(layer_b, i, arc_ltr)
            new_node_idx = layer_b.size
            new_values[new_node_idx] = val

            # Update gaps
            @inbounds @simd for row_idx in 1:num_constraints
                gap_b[row_idx, new_node_idx] = parent_gaps[row_idx] + gap_adjustments_col[row_idx, gap_idx]
            end

        end
    end
end


"""
    enumerate_arcs_ignore_objective!(layer_a::RestrictedNodeLayer, layer_b::RestrictedNodeLayer, gap_a::Matrix{T}, gap_b::Matrix{T}, var_index::Int, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_constraints::Int, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, w::Int) where{Z<:Integer, T<:Real}

Helper function to enumerate arcs left-to-right until width constraint w is reached.

Used for degenerate case where objective can be ignored: threshold_bin == -1 && !uncapped.
"""
@inline function enumerate_arcs_ignore_objective!(
    layer_a::RestrictedNodeLayer,
    layer_b::RestrictedNodeLayer,
    gap_a::Matrix{T},
    gap_b::Matrix{T},
    var_index::Int,
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    num_constraints::Int,
    coeff_times_val::Matrix{T},
    gap_adjustments::Array{T,3},
    w::Int
) where{Z<:Integer, T<:Real}
    # Setup views and bounds
    coeff_times_val_col = @view coeff_times_val[:, var_index+1]
    gap_adjustments_col = @view gap_adjustments[:, :, var_index+1]
    lb = lbs_int[var_index+1]
    original_lb = original_lbs_int[var_index+1]
    lb_offset = 1 - lb
    original_lb_offset = 1 - original_lb

    ltrs   = layer_a.ltrs
    lb_col = layer_a.implied_lbs
    ub_col = layer_a.implied_ubs
    gap_matrix = gap_a

    empty_layer!(layer_b)
    new_values = layer_b.values

    # Enumerate arcs left-to-right until width constraint
    @inbounds for i in 1:layer_a.size
        lb_i = lb_col[i]
        ub_i = ub_col[i]

        # Skip infeasible nodes
        if ub_i < lb_i
            continue
        end

        base_ltr = ltrs[i]
        parent_gaps = @view gap_matrix[:, i]

        # Limit arc enumeration to remaining capacity
        arcs_from_parent = ub_i - lb_i + 1
        remaining_capacity = w - layer_b.size
        val_max = arcs_from_parent <= remaining_capacity ? ub_i : lb_i + remaining_capacity - 1

        @inbounds for val in lb_i:val_max
            # Compute indices (optimized: precomputed offsets)
            gap_idx = val + lb_offset
            coeff_idx = val + original_lb_offset

            # Compute arc objective
            arc_ltr = base_ltr + coeff_times_val_col[coeff_idx]

            # Add node
            add_node!(layer_b, i, arc_ltr)
            new_node_idx = layer_b.size
            new_values[new_node_idx] = val

            # Update gaps
            @inbounds @simd for row_idx in 1:num_constraints
                gap_b[row_idx, new_node_idx] = parent_gaps[row_idx] + gap_adjustments_col[row_idx, gap_idx]
            end

        end

        # Break if width limit reached
        if layer_b.size == w
            break
        end
    end
end


"""
    enumerate_all_arcs_ignore_objective!(layer_a::RestrictedNodeLayer, layer_b::RestrictedNodeLayer, gap_a::Matrix{T}, gap_b::Matrix{T}, var_index::Int, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_constraints::Int, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}) where{Z<:Integer, T<:Real}

Helper function to enumerate ALL arcs for degenerate case where all arcs have identical objective values.

Used when threshold_bin == -1 && uncapped == true (all arcs have same objective AND all fit within width w).
No width checking needed since uncapped guarantees all arcs fit. No bins_matrix needed since objective values are identical.
"""
@inline function enumerate_all_arcs_ignore_objective!(
    layer_a::RestrictedNodeLayer,
    layer_b::RestrictedNodeLayer,
    gap_a::Matrix{T},
    gap_b::Matrix{T},
    var_index::Int,
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    num_constraints::Int,
    coeff_times_val::Matrix{T},
    gap_adjustments::Array{T,3}
) where{Z<:Integer, T<:Real}
    # Setup views and bounds
    coeff_times_val_col = @view coeff_times_val[:, var_index+1]
    gap_adjustments_col = @view gap_adjustments[:, :, var_index+1]
    lb = lbs_int[var_index+1]
    original_lb = original_lbs_int[var_index+1]
    lb_offset = 1 - lb
    original_lb_offset = 1 - original_lb

    ltrs   = layer_a.ltrs
    lb_col = layer_a.implied_lbs
    ub_col = layer_a.implied_ubs
    gap_matrix = gap_a

    empty_layer!(layer_b)
    new_values = layer_b.values

    # Enumerate ALL arcs (uncapped=true guarantees they all fit)
    @inbounds for i in 1:layer_a.size
        lb_i = lb_col[i]
        ub_i = ub_col[i]

        # Skip infeasible nodes
        if ub_i < lb_i
            continue
        end

        base_ltr = ltrs[i]
        parent_gaps = @view gap_matrix[:, i]

        # Enumerate all arcs from this parent (no width check needed)
        @inbounds for val in lb_i:ub_i
            # Compute indices (optimized: precomputed offsets)
            gap_idx = val + lb_offset
            coeff_idx = val + original_lb_offset

            # Compute arc objective
            arc_ltr = base_ltr + coeff_times_val_col[coeff_idx]

            # Add node
            add_node!(layer_b, i, arc_ltr)
            new_node_idx = layer_b.size
            new_values[new_node_idx] = val

            # Update gaps
            @inbounds @simd for row_idx in 1:num_constraints
                gap_b[row_idx, new_node_idx] = parent_gaps[row_idx] + gap_adjustments_col[row_idx, gap_idx]
            end

        end
    end
end


"""
    enumerate_arcs_in_bin_one!(layer_a::RestrictedNodeLayer, layer_b::RestrictedNodeLayer, gap_a::Matrix{T}, gap_b::Matrix{T}, inv_coeff::T, var_index::Int, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_constraints::Int, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, bins_matrix::Matrix{<:Integer}, w::Int) where{Z<:Integer, T<:Real}

Helper function to enumerate arcs in bin 1 only, until width constraint w is reached.

Used when leftmost bin is too large: threshold_bin == 0 && !uncapped. Scans bins_matrix to find bin 1 arcs and enumerates them in iteration order matching histogram construction.
"""
@inline function enumerate_arcs_in_bin_one!(
    layer_a::RestrictedNodeLayer,
    layer_b::RestrictedNodeLayer,
    gap_a::Matrix{T},
    gap_b::Matrix{T},
    inv_coeff::T,
    var_index::Int,
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    num_constraints::Int,
    coeff_times_val::Matrix{T},
    gap_adjustments::Array{T,3},
    bins_matrix::Matrix{<:Integer},
    w::Int
) where{Z<:Integer, T<:Real}
    # Setup views and bounds
    coeff_times_val_col = @view coeff_times_val[:, var_index+1]
    gap_adjustments_col = @view gap_adjustments[:, :, var_index+1]
    lb = lbs_int[var_index+1]
    original_lb = original_lbs_int[var_index+1]
    lb_offset = 1 - lb
    original_lb_offset = 1 - original_lb

    ltrs   = layer_a.ltrs
    lb_col = layer_a.implied_lbs
    ub_col = layer_a.implied_ubs
    gap_matrix = gap_a

    empty_layer!(layer_b)
    new_values = layer_b.values

    # Define iteration direction based on coefficient sign (matches histogram logic)
    make_range = if inv_coeff > 0
        (start, count) -> (start:(start + count - 1))
    elseif inv_coeff < 0
        (start, count) -> (start:-1:(start - count + 1))
    else
        (start, count) -> (start:(start + count - 1))
    end
    start_selector = inv_coeff >= 0 ? lb_col : ub_col

    # Enumerate arcs in bin 1 until width constraint
    @inbounds for i in 1:layer_a.size
        lb_i = lb_col[i]
        ub_i = ub_col[i]

        # Skip infeasible nodes
        if ub_i < lb_i
            continue
        end

        # Find where bin 1 ends by scanning bins_matrix
        arc_idx = 1
        @inbounds while bins_matrix[arc_idx, i] == 1
            arc_idx += 1
        end
        num_bin_1_arcs = arc_idx - 1

        # Limit to remaining capacity
        remaining_capacity = w - layer_b.size
        arcs_to_enumerate = min(num_bin_1_arcs, remaining_capacity)

        # Skip parent if no arcs to enumerate
        if arcs_to_enumerate == 0
            continue
        end

        base_ltr = ltrs[i]
        parent_gaps = @view gap_matrix[:, i]
        start_val = start_selector[i]

        # Enumerate exact range (no if statement in loop!)
        @inbounds for val in make_range(start_val, arcs_to_enumerate)
            # Compute indices (optimized: precomputed offsets)
            gap_idx = val + lb_offset
            coeff_idx = val + original_lb_offset

            # Compute arc objective
            arc_ltr = base_ltr + coeff_times_val_col[coeff_idx]

            # Add node
            add_node!(layer_b, i, arc_ltr)
            new_node_idx = layer_b.size
            new_values[new_node_idx] = val

            # Update gaps
            @inbounds @simd for row_idx in 1:num_constraints
                gap_b[row_idx, new_node_idx] = parent_gaps[row_idx] + gap_adjustments_col[row_idx, gap_idx]
            end

        end

        # Break if width limit reached
        if layer_b.size == w
            break
        end
    end
end


"""
    enumerate_arcs_using_hybrid_threshold!(layer_a::RestrictedNodeLayer, layer_b::RestrictedNodeLayer, gap_a::Matrix{T}, gap_b::Matrix{T}, inv_coeff::T, threshold_bin::Int, var_index::Int, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_constraints::Int, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, bins_matrix::Matrix{<:Integer}, w::Int) where{Z<:Integer, T<:Real}

Helper function to enumerate arcs using hybrid threshold (baseline + budget-based extras).

Used for normal bin-based filtering: 1 ≤ threshold_bin < w. Enumerates ALL arcs in bins ≤ threshold_bin, then fills remaining capacity with arcs from bin threshold_bin + 1.
"""
@inline function enumerate_arcs_using_hybrid_threshold!(
    layer_a::RestrictedNodeLayer,
    layer_b::RestrictedNodeLayer,
    gap_a::Matrix{T},
    gap_b::Matrix{T},
    inv_coeff::T,
    threshold_bin::Int,
    var_index::Int,
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    num_constraints::Int,
    coeff_times_val::Matrix{T},
    gap_adjustments::Array{T,3},
    bins_matrix::Matrix{<:Integer},
    w::Int
) where{Z<:Integer, T<:Real}
    # Setup views and bounds
    coeff_times_val_col = @view coeff_times_val[:, var_index+1]
    gap_adjustments_col = @view gap_adjustments[:, :, var_index+1]
    lb = lbs_int[var_index+1]
    original_lb = original_lbs_int[var_index+1]
    lb_offset = 1 - lb
    original_lb_offset = 1 - original_lb

    ltrs   = layer_a.ltrs
    lb_col = layer_a.implied_lbs
    ub_col = layer_a.implied_ubs
    gap_matrix = gap_a

    empty_layer!(layer_b)
    new_values = layer_b.values

    # Define iteration direction based on coefficient sign (matches histogram logic)
    make_range = if inv_coeff > 0
        (start, count) -> (start:(start + count - 1))
    elseif inv_coeff < 0
        (start, count) -> (start:-1:(start - count + 1))
    else
        (start, count) -> (start:(start + count - 1))
    end
    start_selector = inv_coeff >= 0 ? lb_col : ub_col

    # Enumerate arcs using hybrid threshold (baseline + budget-based extras)
    @inbounds for i in 1:layer_a.size
        lb_i = lb_col[i]
        ub_i = ub_col[i]

        # Skip infeasible nodes
        if ub_i < lb_i
            continue
        end

        # Find arcs in bins <= threshold_bin (baseline)
        arc_idx = 1
        @inbounds while bins_matrix[arc_idx, i] <= threshold_bin && bins_matrix[arc_idx, i] != -1
            arc_idx += 1
        end
        num_baseline_arcs = arc_idx - 1

        # Find arcs in bin threshold_bin + 1 (extras)
        @inbounds while bins_matrix[arc_idx, i] == threshold_bin + 1
            arc_idx += 1
        end
        num_extra_arcs = arc_idx - num_baseline_arcs - 1

        # Calculate total arcs to enumerate based on budget
        remaining_budget = w - layer_b.size
        baseline_to_enumerate = min(num_baseline_arcs, remaining_budget)

        arcs_to_enumerate = if baseline_to_enumerate == num_baseline_arcs
            # Baseline fits, add extras from remaining budget
            remaining_after_baseline = remaining_budget - num_baseline_arcs
            extras_to_enumerate = min(num_extra_arcs, remaining_after_baseline)
            num_baseline_arcs + extras_to_enumerate
        else
            # Can't fit all baseline, just enumerate what fits
            baseline_to_enumerate
        end

        # Skip parent if no arcs to enumerate
        if arcs_to_enumerate == 0
            continue
        end

        base_ltr = ltrs[i]
        parent_gaps = @view gap_matrix[:, i]
        start_val = start_selector[i]

        # Enumerate exact range (no if statement in loop!)
        @inbounds for val in make_range(start_val, arcs_to_enumerate)
            # Compute indices (optimized: precomputed offsets)
            gap_idx = val + lb_offset
            coeff_idx = val + original_lb_offset

            # Compute arc objective
            arc_ltr = base_ltr + coeff_times_val_col[coeff_idx]

            # Add node
            add_node!(layer_b, i, arc_ltr)
            new_node_idx = layer_b.size
            new_values[new_node_idx] = val

            # Update gaps
            @inbounds @simd for row_idx in 1:num_constraints
                gap_b[row_idx, new_node_idx] = parent_gaps[row_idx] + gap_adjustments_col[row_idx, gap_idx]
            end

        end

        # Break if width limit reached
        if layer_b.size == w
            break
        end
    end
end
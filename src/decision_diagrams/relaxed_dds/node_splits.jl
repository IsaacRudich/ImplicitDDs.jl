"""
COMMENTS UP TO DATE
"""

"""
    invert_implied_column_bounds!(
        node_matrix::Vector{NodeLayer{Z}}, new_domain::UnitRange,
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        var_index::Int, lowest_idx::Vector{U}, highest_idx::Vector{U}
    ) where {Z<:Integer, U<:Integer}

Assigns interval indices to each domain element based on implied column bounds using a two-pass algorithm.

Creates nodes for the current variable layer by determining which parent intervals (arcs) are associated with each domain value. The algorithm performs two passes over the parent intervals to find the lowest and highest interval indices that cover each domain element, then creates nodes with the appropriate arc ranges.

# Arguments
- `node_matrix::Vector{NodeLayer{Z}}`: Vector of decision diagram layers; results for current variable are added to `node_matrix[var_index]`
- `new_domain::UnitRange`: Domain range for the current variable
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for variable domains at each parent interval
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for variable domains at each parent interval
- `var_index::Int`: Index of current layer being processed (1-based)
- `lowest_idx::Vector{U}`: Pre-allocated workspace vector to store lowest interval index for each domain element
- `highest_idx::Vector{U}`: Pre-allocated workspace vector to store highest interval index for each domain element

# Algorithm
1. **Left-to-right pass**: For each parent interval, assign its index as the lowest index to all domain elements in its range that aren't yet assigned
2. **Right-to-left pass**: For each parent interval, assign its index as the highest index to all domain elements in its range that aren't yet assigned
3. **Node creation**: Create nodes for each domain element using its lowest and highest interval indices as arc range bounds
"""
@inline function invert_implied_column_bounds!(
    node_matrix ::Vector{NodeLayer{Z}},
    new_domain  ::UnitRange,
    lb_matrix   ::Union{Matrix{Z}, BitMatrix},
    ub_matrix   ::Union{Matrix{Z}, BitMatrix},
    var_index   ::Int,
    lowest_idx  ::Vector{U},
    highest_idx ::Vector{U}
)where{Z<:Integer, U<:Integer}

    # Extract the intervals implied by the previous variable
    interval_lbs = view(lb_matrix, :, var_index-1)
    interval_ubs = view(ub_matrix, :, var_index-1)
    m = node_matrix[var_index-1].size

    # We'll store the lowest and highest interval indices for each domain value
    domain_size = length(new_domain)
    lowest_idx .= 0
    highest_idx .= 0

    # We'll track how many domain elements got their lowest index assigned,
    # so we can break early.
    assigned_count = 0
    #local vars for loop
    start_val = 0
    end_val = 0
    start_idx = 0
    end_idx = 0
    lb = 0
    ub = 0

    # Hoist domain bounds to avoid repeated function calls
    domain_first = first(new_domain)
    domain_last = last(new_domain)
    # Precompute index offset for domain value -> array index conversion
    offset = 1 - domain_first

    # -------------------- PASS 1: Assign lowest indexes --------------------
    @inbounds for i in 1:m
        lb = interval_lbs[i]
        ub = interval_ubs[i]
        # Intersect [lb, ub] with the domain range
        # new_domain is from new_domain[1] to new_domain[end].
        start_val = max(lb, domain_first)
        end_val   = min(ub, domain_last)

        # If there's no overlap, skip.
        if start_val <= end_val
            # Convert those domain values to array indices.
            # domain_idx(k) = k - domain_first + 1 = k + offset
            start_idx = start_val + offset
            end_idx   = end_val + offset

            # For each domain element in this intersection:
            @inbounds @simd for dom_pos in start_idx:end_idx
                if lowest_idx[dom_pos] == 0
                    lowest_idx[dom_pos] = i
                    assigned_count += 1
                end
            end
            if assigned_count == domain_size
                break
            end
        end
    end

    # We'll track how many domain elements got their highest index assigned
    # so we can break early in the second pass as well.
    assigned_count = 0

    # -------------------- PASS 2: Assign highest indexes --------------------
    @inbounds for i in m:-1:1
        lb = interval_lbs[i]
        ub = interval_ubs[i]

        start_val = max(lb, domain_first)
        end_val   = min(ub, domain_last)
        if start_val <= end_val
            start_idx = start_val + offset
            end_idx   = end_val + offset

            @inbounds @simd for dom_pos in start_idx:end_idx
                if highest_idx[dom_pos] == 0
                    highest_idx[dom_pos] = i
                    assigned_count += 1
                end
            end
            if assigned_count == domain_size
                break
            end
        end
    end

    # -------------------- Build the NodeLayer --------------------
    node_layer = node_matrix[var_index]

    @inbounds for (k, val) in enumerate(new_domain)
        if lowest_idx[k] != 0
            add_node!(node_layer, lowest_idx[k], highest_idx[k], val)
        end
    end
end


"""
    split_nodes!(
        node_matrix::Vector{NodeLayer{Z}}, extra_layer::NodeLayer{Z},
        ltr_matrix::Matrix{U}, var_index::Int,
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        int_obj_coeffs::Vector{U}, w::Int, domain::UnitRange, timing_stats::TimingStats,
        arc_count_per_node::Vector{<:Integer}, node_bin_counts::Matrix{<:Integer},
        node_cumulative::Matrix{<:Integer}, global_lower::Vector{<:Integer},
        global_upper::Vector{<:Integer}, bins_matrix::Matrix{<:Integer},
        coeff_times_val::Matrix{U}, original_lbs_int::Union{Vector{Z}, BitVector}
    ) where {Z<:Integer, U<:Real} -> (NodeLayer{Z}, Bool)

Uses dual-threshold hybrid algorithm to build the next layer with size at or near target width w. Computes conservative and aggressive thresholds, then uses budget-based switching to achieve optimal width utilization.

# Arguments
- `node_matrix::Vector{NodeLayer{Z}}`: The decision diagram layers; the layer at `var_index` will be replaced
- `extra_layer::NodeLayer{Z}`: Pre-allocated empty layer used as workspace for double-buffering
- `ltr_matrix::Matrix{U}`: Length-To-Root values for parents
- `var_index::Int`: Index of current layer being processed (1-based)
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for variable domains at each parent
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for variable domains at each parent
- `int_obj_coeffs::Vector{U}`: Objective coefficients for integer variables
- `w::Int`: Maximum allowed layer width (width constraint)
- `domain::UnitRange`: Domain range for the current variable
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `arc_count_per_node::Vector{<:Integer}`: Preallocated workspace for feasible arc counts per node, size [K] (allocated as [max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_bin_counts::Matrix{<:Integer}`: Preallocated workspace for bin count histograms per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_cumulative::Matrix{<:Integer}`: Preallocated workspace for cumulative bin counts per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `global_lower::Vector{<:Integer}`: Preallocated workspace for global lower bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `global_upper::Vector{<:Integer}`: Preallocated workspace for global upper bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `bins_matrix::Matrix{<:Integer}`: Preallocated workspace for individual arc-to-bin mappings, size [w, K] (allocated as [relaxed_w, max_domain_size], stores bin index for each in-arc)
- `coeff_times_val::Matrix{U}`: Precomputed coefficient products [max_domain_size, num_int_vars] for arc objectives
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original global lower bounds for integer variables (before FBBT tightening) used for consistent array indexing into coeff_times_val

# Returns
- `(NodeLayer{Z}, Bool)`: A tuple containing:
  1. `NodeLayer{Z}`: The displaced layer for reuse in double-buffering pattern. In normal operation returns the original layer that was replaced. In edge cases where no expansion occurs, returns the unused workspace layer.
  2. `Bool`: `true` if the constructed layer is exact (all nodes have `first_arc == last_arc`), `false` if inexact (contains merged blocks). Exactness determined by which build routine was called:
     - `build_layer_all_arcs!` (when `t_safe_idx == w`): always exact
     - `build_layer_with_hybrid_thresholds!` (when `t_safe_idx < w`): always inexact
     - `build_layer_arbitrarily!` (when `t_safe_idx == -1`): depends on budget

# Algorithm Details
1. **Cap Range Calculation**: Determines minimum and maximum possible Length-To-Root values based on parent layer and variable domain
2. **Initial Assessment**: Counts nodes that would result from using maximum cap (most expansion)
3. **Bin-Based Search**: If initial count exceeds width limit, uses histogram binning to find heuristic cap value for selective expansion
4. **Layer Construction**: Builds new layer where arcs are either expanded to individual nodes (LTR ≤ cap) or kept as merged blocks (LTR > cap)
5. **Exactness Tracking**: Returns whether the layer contains only individual nodes (exact) or includes merged blocks (inexact) for efficient last exact layer identification
"""
@inline function split_nodes!(
    node_matrix         ::Vector{NodeLayer{Z}}, # all layers
    extra_layer         ::NodeLayer{Z},
    ltr_matrix          ::Matrix{U},
    var_index           ::Int,  # current layer index
    lb_matrix           ::Union{Matrix{Z}, BitMatrix},
    ub_matrix           ::Union{Matrix{Z}, BitMatrix},
    int_obj_coeffs      ::Vector{U},
    w                   ::Int,
    domain              ::UnitRange,
    timing_stats        ::TimingStats,
    arc_count_per_node  ::Vector{<:Integer},
    node_bin_counts     ::Matrix{<:Integer},
    node_cumulative     ::Matrix{<:Integer},
    global_lower        ::Vector{<:Integer},
    global_upper        ::Vector{<:Integer},
    bins_matrix         ::Matrix{<:Integer},
    coeff_times_val     ::Matrix{U},
    original_lbs_int    ::Union{Vector{Z}, BitVector}
) where {Z<:Integer, U<:Real}
    
    # Get views and cost coefficient
    parent_size = node_matrix[var_index-1].size
    parent_ltr_col = @view ltr_matrix[1:parent_size, var_index - 1]
    interval_lbs   = @view lb_matrix[1:parent_size, var_index - 1]
    interval_ubs   = @view ub_matrix[1:parent_size, var_index - 1]
    c = int_obj_coeffs[var_index]

    # Our original layer to be split.
    old_layer = node_matrix[var_index]

    # Edge case: if layer already at width limit, no room for splitting
    if old_layer.size == w
        # Must check if layer is exact - can't assume!
        is_exact = true
        @inbounds for i in 1:w
            if old_layer.first_arcs[i] != old_layer.last_arcs[i]
                is_exact = false
                break
            end
        end

        return extra_layer, is_exact
    end

    @time_operation timing_stats relaxed_dd_bin_counter begin
        # Compute dual thresholds and build bins_matrix for hybrid algorithm
        t_safe_idx, t_aggressive_idx = compute_bounding_thresholds(
            old_layer, parent_ltr_col, interval_lbs, interval_ubs, c, w, domain,
            arc_count_per_node, node_bin_counts, node_cumulative, global_lower, global_upper,
            bins_matrix, coeff_times_val, original_lbs_int, var_index
        )
    end

    # Track exactness based on which build routine is called
    is_exact = false  # Default to inexact

    @time_operation timing_stats relaxed_dd_layer_construction begin
        # Build layer using hybrid threshold strategy with budget-based switching
        if t_safe_idx != -1
            if t_safe_idx == w
                # All arcs fit - individualize everything (fastest path)
                build_layer_all_arcs!(old_layer, extra_layer, bins_matrix)
                is_exact = true  # This routine ALWAYS produces exact layers
            else
                build_layer_with_hybrid_thresholds!(
                    old_layer, extra_layer, w,
                    bins_matrix, t_safe_idx, t_aggressive_idx
                )
                # is_exact remains false - this routine ALWAYS produces inexact layers
            end
        else
            is_exact = build_layer_arbitrarily!(old_layer, extra_layer, interval_lbs, interval_ubs, w)
        end
        node_matrix[var_index] = extra_layer
    end

    # if w - extra_layer.size > 5 && t_safe_idx != t_aggressive_idx
    #     error("Hybrid threshold guarantee violated: " *
    #           "layer_size = $(extra_layer.size), w = $w, gap = $(w - extra_layer.size) " *
    #           "(expected gap ≤ 1 when t_safe_idx=$t_safe_idx != t_aggressive_idx=$t_aggressive_idx)")
    # end
    return old_layer, is_exact
end


"""
    build_layer_with_hybrid_thresholds!(
        node_layer::NodeLayer{Z}, new_layer::NodeLayer{Z},
        w::Int, bins_matrix::Matrix{<:Integer},
        t_safe_bin_idx::Int, t_aggressive_bin_idx::Int
    ) where {Z<:Integer}

Builds the next layer using dual thresholds with budget-based switching to achieve at least w-K nodes.
Thre is a budget for extra splits that gets used by each node from left to right.
It may be overcharged by 1 for each node resulting in w-K nodes instead of w.

Optimized to reuse histogram data from compute_bounding_thresholds, avoiding redundant arc_ltr computations.

First simulates with t_safe (using histogram bins) to establish baseline node count, then uses a three-zone decision strategy:
- Safe zone (bin < t_safe_bin_idx): Always create individual nodes (part of baseline)
- Aggressive zone (bin >= t_aggressive_bin_idx): Always merge into blocks
- Middle zone (t_safe_bin_idx ≤ bin < t_aggressive_bin_idx): Individualize only if budget allows (switches to blocking when budget exhausted)

# Arguments
- `node_layer::NodeLayer{Z}`: Current layer containing nodes to process
- `new_layer::NodeLayer{Z}`: Target layer to build (will be cleared first)
- `w::Int`: Target width (maximum nodes allowed in new layer)
- `bins_matrix::Matrix{<:Integer}`: Pre-computed bin assignments for each arc [local_arc_idx, node_idx], with -1 for infeasible arcs
- `t_safe_bin_idx::Int`: Bin index corresponding to t_safe threshold
- `t_aggressive_bin_idx::Int`: Bin index corresponding to t_aggressive threshold

# Returns
Nothing (modifies `new_layer` in place)

# Algorithm
1. Simulates with t_safe using histogram bins to get baseline_count (avoids recomputing arc_ltr)
2. Computes budget = w - baseline_count (available budget for extra individualizations)
3. Handles edge case: if t_safe_bin_idx == t_aggressive_bin_idx, sets budget = 0 (no middle zone)
4. For each arc, looks up pre-computed bin from bins_matrix and decides zone using bin indices (skips -1 sentinels for infeasible arcs)
5. For middle zone arcs, computes cost = (block_open ? 2 : 1) and individualizes only if cost ≤ budget
6. Guarantees all t_safe arcs are individualized and final size ≤ w
"""
@inline function build_layer_with_hybrid_thresholds!(
    node_layer              ::NodeLayer{Z},
    new_layer               ::NodeLayer{Z},
    w                       ::Int,
    bins_matrix             ::Matrix{<:Integer},
    t_safe_bin_idx          ::Int,
    t_aggressive_bin_idx    ::Int
) where {Z<:Integer}

    empty_layer!(new_layer)

    # Optimization: if t_safe_bin_idx == t_aggressive_bin_idx, no middle zone exists
    # Budget will automatically be 0 since baseline_count = w, but set explicitly for clarity
    if t_safe_bin_idx == t_aggressive_bin_idx
        budget = 0
    else
        # Simulate with t_safe using histogram bins (avoids recomputing arc_ltr)
        baseline_count = simulate_layer_size_at_bin(
            t_safe_bin_idx, bins_matrix, node_layer
        )
        budget = w - baseline_count  # Available budget for extra individualizations
    end

    @inbounds for i in 1:node_layer.size
        bins_col = @view bins_matrix[:, i]
        f_arc = node_layer.first_arcs[i]
        l_arc = node_layer.last_arcs[i]
        val   = node_layer.values[i]

        block_start = Int32(-1)

        deferred_cost = false

        @inbounds for (local_idx, arc_idx) in enumerate(f_arc:l_arc)
            # Look up bin for this arc (uses -1 sentinel for infeasible arcs)
            bin = bins_col[local_idx]

            # Skip infeasible arcs
            if bin == -1
                continue
            end

            # Three-zone decision logic using bin indices
            if bin < t_safe_bin_idx
                # Safe zone: always individualize
                if block_start != -1
                    add_node!(new_layer, block_start, arc_idx-1, val)
                    block_start = -1
                end

                # Refund deferred cost: if we charged for closing a block to create a middle arc,
                # but then a safe arc closes naturally, we overcharged by 1
                if deferred_cost
                    budget += 1
                    deferred_cost = false
                end

                add_node!(new_layer, arc_idx, arc_idx, val)

            elseif bin >= t_aggressive_bin_idx
                # Aggressive zone: always block
                if block_start == -1
                    block_start = arc_idx
                end
                # Note: We don't charge deferred_cost here anymore.
                # After removing deferred cost from block closures, deferred_cost is only set
                # by middle zone fresh-starts as a run tracker, which shouldn't be charged.
                # Reset it to ensure clean state for next middle zone.
                deferred_cost = false
            else
                # Middle zone (t_safe_bin_idx <= bin < t_aggressive_bin_idx): individualize if budget allows

                # Simple budget model: B's + C-sequences - 1
                # - Each B (individual middle arc) costs 1
                # - Each C-sequence closure (closing block to create B) costs 1
                # Equivalently: close+indiv costs 2, fresh-start/in-run costs 1

                if block_start != -1
                    # Close block + individualize middle arc
                    # Cost: 1 for B + 1 for closing block = 2
                    if budget >= 2
                        add_node!(new_layer, block_start, arc_idx-1, val)
                        add_node!(new_layer, arc_idx, arc_idx, val)
                        budget -= 2
                        block_start = -1
                        deferred_cost = true
                    end
                else
                    # No open block - fresh-start or in-run
                    # Cost: 1 for B
                    if budget >= 1
                        add_node!(new_layer, arc_idx, arc_idx, val)
                        budget -= 1
                        deferred_cost = true
                    else
                        # Start merging
                        if block_start == -1
                            block_start = arc_idx
                        end
                        deferred_cost = false
                    end
                end
            end
        end

        # Close any remaining block for this value
        if block_start != -1
            add_node!(new_layer, block_start, l_arc, val)

            # Refund deferred cost: final block closure is like a safe arc - it was going to happen anyway
            # If we charged for closing a block to create a middle arc, but then the final block closes,
            # we overcharged by 1 (the block would have closed anyway at the end)
            if deferred_cost
                budget += 1
                deferred_cost = false
            end
        end
    end
end


"""
    build_layer_arbitrarily!(
        node_layer::NodeLayer{Z}, new_layer::NodeLayer{Z},
        interval_lbs::SubArray{Z}, interval_ubs::SubArray{Z},
        w::Int
    ) where {Z<:Integer} -> Bool

Builds the next layer with simple budget-based splitting when objective-based thresholds cannot be computed.

Used as a fallback when all arcs have identical objective values (degenerate case where `max_cap == min_cap` in threshold computation).
Splits every arc from left to right until excess width budget runs out.

# Arguments
- `node_layer::NodeLayer{Z}`: Current layer to be split
- `new_layer::NodeLayer{Z}`: Target layer to populate with split nodes
- `interval_lbs::SubArray{Z}`: Lower bounds for variable domains at each parent interval
- `interval_ubs::SubArray{Z}`: Upper bounds for variable domains at each parent interval
- `w::Int`: Maximum allowed layer width (width constraint)

# Returns
- `Bool`: `true` if layer is exact (all nodes have `first_arc == last_arc`), `false` if inexact (contains merged blocks)

# Algorithm
1. **Budget calculation**: Compute available splitting budget as `w - node_layer.size` (reserves space for unsplit nodes)
2. **Sequential splitting**: Process each parent node's arcs in order:
   - While budget > 0: Create individual nodes (split arcs)
   - When budget exhausted: Create merged node for remaining arcs of current parent
3. **Label separation**: Maintains one node per domain value per parent to preserve DD structure

# Notes
- Skips infeasible arcs (where parent value violates domain bounds)
- Guarantees output layer size ≤ w by construction
- Does not require objective value information (threshold-free splitting)
- Returns exactness to enable efficient last exact layer tracking
"""
@inline function build_layer_arbitrarily!(
    node_layer              ::NodeLayer{Z},
    new_layer               ::NodeLayer{Z},
    interval_lbs            ::SubArray{Z},
    interval_ubs            ::SubArray{Z},
    w                       ::Int,
) where {Z<:Integer}

    empty_layer!(new_layer)
    # Reserve one slot per compact node for worst case (each might need one merged block)
    # This ensures total nodes never exceeds w
    budget = w - node_layer.size
    is_exact = true  # Track if layer remains exact (no merged blocks)

    @inbounds for i in 1:node_layer.size
        f_arc = node_layer.first_arcs[i]
        l_arc = node_layer.last_arcs[i]
        val   = node_layer.values[i]

        @inbounds for arc_idx in f_arc:l_arc
            lb = interval_lbs[arc_idx]
            ub = interval_ubs[arc_idx]

            # Skip infeasible arcs
            if val < lb || val > ub
                continue
            end

            if budget > 0
                add_node!(new_layer, arc_idx, arc_idx, val)
                budget -= 1
            else
                # Budget exhausted - create merged block for remaining arcs
                # Only mark as inexact if we're actually merging multiple arcs
                if arc_idx < l_arc
                    is_exact = false
                end
                add_node!(new_layer, arc_idx, l_arc, val)
                break
            end
        end
    end

    return is_exact
end


"""
    build_layer_all_arcs!(
        node_layer::NodeLayer{Z}, new_layer::NodeLayer{Z},
        bins_matrix::Matrix{<:Integer}
    ) where {Z<:Integer}

Builds the next layer by individualizing all feasible arcs without any width constraints.

Used when total feasible arcs ≤ w (detected by t_safe_bin_idx == w), allowing complete
expansion without threshold-based filtering or budget tracking.

# Arguments
- `node_layer::NodeLayer{Z}`: Current layer to be expanded
- `new_layer::NodeLayer{Z}`: Target layer to populate with individualized nodes
- `bins_matrix::Matrix{<:Integer}`: Pre-computed bin assignments with -1 for infeasible arcs

# Algorithm
For each node in the current layer, creates an individual child node for every feasible arc.
Skips infeasible arcs marked with -1 sentinel in bins_matrix.

# Performance
Avoids all overhead from threshold-based splitting:
- Single bin lookup per arc (vs 2 bounds checks)
- Simple -1 comparison (vs 2 comparisons for bounds)
- No threshold comparisons
- No budget tracking or block management
- Direct individualization of all feasible arcs

# Notes
- Guarantees output layer size = total feasible arcs ≤ w
- Most efficient construction method when applicable
- Only called when t_safe_bin_idx == w in compute_bounding_thresholds
"""
@inline function build_layer_all_arcs!(
    node_layer  ::NodeLayer{Z},
    new_layer   ::NodeLayer{Z},
    bins_matrix ::Matrix{<:Integer}
) where {Z<:Integer}

    empty_layer!(new_layer)

    @inbounds for i in 1:node_layer.size
        bins_col = @view bins_matrix[:, i]
        f_arc = node_layer.first_arcs[i]
        l_arc = node_layer.last_arcs[i]
        val   = node_layer.values[i]

        @inbounds for (local_idx, arc_idx) in enumerate(f_arc:l_arc)
            bin = bins_col[local_idx]

            # Only individualize feasible arcs (skip -1 sentinels)
            if bin != -1
                add_node!(new_layer, arc_idx, arc_idx, val)
            end
        end
    end
end


"""
    compute_bounding_thresholds(
        old_layer::NodeLayer{Z}, parent_ltr_col::SubArray{U},
        interval_lbs::SubArray{Z}, interval_ubs::SubArray{Z},
        c::U, w::Int, domain::UnitRange,
        arc_count_per_node::Vector{<:Integer},
        node_bin_counts::Matrix{<:Integer}, node_cumulative::Matrix{<:Integer},
        global_lower::Vector{<:Integer}, global_upper::Vector{<:Integer},
        bins_matrix::Matrix{<:Integer},
        coeff_times_val::Matrix{U}, original_lbs_int::Union{Vector{Z}, BitVector}, var_index::Int
    ) where {Z<:Integer, U<:Real}

Generates two bin indices for dual-threshold node splitting algorithms: a conservative threshold that guarantees layer size ≤ w (tier-1), and an aggressive threshold that guarantees layer size >= w (tier-2).
Uses pre-binned histogram representation to efficiently compute global lower and upper bounds for all candidate thresholds, then extracts the bounding indices and builds individual arc-to-bin mappings.

# Arguments
- `old_layer::NodeLayer{Z}`: Current layer to be split
- `parent_ltr_col::SubArray{U}`: Length-To-Root values from parent layer
- `interval_lbs::SubArray{Z}`: Lower bounds for variable domains at each parent
- `interval_ubs::SubArray{Z}`: Upper bounds for variable domains at each parent
- `c::U`: Objective coefficient for current variable
- `w::Int`: Maximum allowed layer width (width constraint)
- `domain::UnitRange`: Domain range for the current variable
- `arc_count_per_node::Vector{<:Integer}`: Pre-allocated workspace for feasible arc counts per node, size [K] (max value: w, type: Int8/16/32/64)
- `node_bin_counts::Matrix{<:Integer}`: Pre-allocated workspace for bin count histograms per node, size [w, K] (max value: w, type: Int8/16/32/64)
- `node_cumulative::Matrix{<:Integer}`: Pre-allocated workspace for cumulative bin counts per node, size [w, K] (max value: w, type: Int8/16/32/64)
- `global_lower::Vector{<:Integer}`: Pre-allocated workspace for global lower bound estimates per threshold, size [w] (max value: K×w, type: Int8/16/32/64)
- `global_upper::Vector{<:Integer}`: Pre-allocated workspace for global upper bound estimates per threshold, size [w] (max value: K×w, type: Int8/16/32/64)
- `bins_matrix::Matrix{<:Integer}`: Pre-allocated workspace for individual arc-to-bin mappings, size [w, K] (stores bin index 1 to w-1 for feasible arcs, -1 for infeasible arcs)
- `coeff_times_val::Matrix{U}`: Precomputed coefficient products [max_domain_size, num_int_vars] for arc objectives
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original global lower bounds for integer variables (before FBBT tightening) used for consistent array indexing into coeff_times_val
- `var_index::Int`: Index of current variable layer being processed

# Returns
- Tuple of `(t_safe_idx, t_aggressive_idx)` where:
  1. `t_safe_idx::Int`: Bin index for conservative threshold guaranteeing layer size ≤ w
  2. `t_aggressive_idx::Int`: Bin index for aggressive threshold targeting layer size ≈ w

# Algorithm
1. **Phase 1 - Pre-binning**: Compute all arc objectives, assign to w-1 bins, store -1 sentinel for infeasible arcs, and build histograms in node_bin_counts for cache-efficient processing
2. **Phase 2 - Bounds computation**: Calculate global_lower and global_upper bounds for all thresholds using three-case analysis (all-below, all-above, mixed)
3. **Phase 3 - Threshold extraction**: Scan bounds arrays to find highest bin indices satisfying conservative (upper bound) and aggressive (lower bound) constraints
"""
@inline function compute_bounding_thresholds(
    old_layer           ::NodeLayer{Z},
    parent_ltr_col      ::SubArray{U},
    interval_lbs        ::SubArray{Z},
    interval_ubs        ::SubArray{Z},
    c                   ::U,
    w                   ::Int,
    domain              ::UnitRange,
    arc_count_per_node  ::Vector{<:Integer},
    node_bin_counts     ::Matrix{<:Integer},
    node_cumulative     ::Matrix{<:Integer},
    global_lower        ::Vector{<:Integer},
    global_upper        ::Vector{<:Integer},
    bins_matrix         ::Matrix{<:Integer},
    coeff_times_val     ::Matrix{U},
    original_lbs_int    ::Union{Vector{Z}, BitVector},
    var_index           ::Int
) where {Z<:Integer, U<:Real}

    # Determine arc objective range
    min_cap = minimum(parent_ltr_col)
    max_cap = maximum(parent_ltr_col)
    
    if c < 0
        min_cap += (maximum(domain) * c)
        max_cap += (minimum(domain) * c)
    elseif c > 0
        max_cap += (maximum(domain) * c)
        min_cap += (minimum(domain) * c)
    end
    min_cap = floor(Int, min_cap)
    max_cap = ceil(Int, max_cap)

    if max_cap == min_cap
        return (-1, -1)
    end

    # Phase 1: Build pre-binned structure
    # Layout: node_bin_counts[w, K] where each column is one node (column-major friendly)
    K = old_layer.size

    # Zero out the workspace arrays (only the regions we'll use: K columns)
    @views arc_count_per_node[1:K] .= 0
    @views node_bin_counts[:, 1:K] .= 0

    # Total range width for mapping arc_ltr values to bins
    range_width = max_cap - min_cap
    inv_range_width = 1 / range_width
    w_minus_1 = w - 1

    # Precomputed coefficient products for this variable
    lb_offset = 1 - original_lbs_int[var_index]
    coeff_col = @view coeff_times_val[:, var_index]

    @inbounds @fastmath for node_idx in 1:K
        bins_col = @view bins_matrix[:, node_idx]
        counts_col = @view node_bin_counts[:, node_idx]
        val = old_layer.values[node_idx]
        f_arc = old_layer.first_arcs[node_idx]
        l_arc = old_layer.last_arcs[node_idx]
        val_idx = val + lb_offset
        contribution = coeff_col[val_idx]
        feasible_count = 0

        @inbounds @fastmath for (local_idx, parent_idx) in enumerate(f_arc:l_arc)
            lb = interval_lbs[parent_idx]
            ub = interval_ubs[parent_idx]

            if val >= lb && val <= ub  # feasible arc
                # Compute the LTR for the arc:
                arc_obj = parent_ltr_col[parent_idx] + contribution

                # Normalize within the [min_cap, max_cap] range:
                frac = (arc_obj - min_cap) * inv_range_width
                # Map to bin: bin k contains arcs where bin_edges[k] < arc_obj <= bin_edges[k+1]
                bin_j = clamp(ceil(Int, frac * w_minus_1), 1, w_minus_1)

                bins_col[local_idx] = bin_j
                counts_col[bin_j] += 1
                feasible_count += 1
            else
                # Store sentinel for infeasible arcs
                bins_col[local_idx] = -1
            end
        end

        arc_count_per_node[node_idx] = feasible_count
    end

    # Build cumulative histogram: node_cumulative[ω, K]
    # Each column is one node's cumulative bin counts (uses pre-allocated workspace)
    @inbounds for node_idx in 1:K
        cumulative_col = @view node_cumulative[:, node_idx]
        bin_counts_col = @view node_bin_counts[:, node_idx]

        cumulative_col[1] = bin_counts_col[1]
        @inbounds for j in 2:w
            cumulative_col[j] = cumulative_col[j-1] + bin_counts_col[j]
        end
    end

    # Phase 2: Compute lower and upper bounds for all thresholds (uses pre-allocated workspaces)
    @inbounds for j in 1:w
        lower_sum = 0
        upper_sum = 0

        # SIMD-optimized inner loop: restructured to avoid branches that inhibit vectorization
        @inbounds @simd for node_idx in 1:K
            total_arcs = arc_count_per_node[node_idx]

            # n_low = number of arcs in bins [1, j-1] (below threshold)
            # n_high = number of arcs in bins [j, w] (at or above threshold)
            n_low = (j > 1) ? node_cumulative[j-1, node_idx] : 0
            n_high = total_arcs - n_low

            # Three-case analysis for threshold bounding
            # Case 1: n_high == 0 (all arcs below threshold)
            # Case 2: n_low == 0 (all arcs at/above threshold)
            # Case 3: mixed (both low and high arcs present)
            if total_arcs > 0
                if n_high == 0  # all arcs below threshold
                    lower_contrib = n_low
                    upper_contrib = n_low
                elseif n_low == 0  # all arcs at/above threshold
                    lower_contrib = 1
                    upper_contrib = 1
                else  # mixed case
                    lower_contrib = n_low + 1
                    upper_contrib = n_low + min(n_low + 1, n_high)
                end
                lower_sum += lower_contrib
                upper_sum += upper_contrib
            end
        end

        global_lower[j] = lower_sum
        global_upper[j] = upper_sum
    end

    # Phase 3: find bounding thresholds
    # Find initial search window by scanning bounds
    low = 1
    high = w

    # Find highest index where we might still fit (global_lower <= w)
    @inbounds for j in w:-1:1
        if global_lower[j] <= w
            high = j
            break
        end
    end

    # Find highest index where we definitely fit (global_upper <= w)
    @inbounds for j in w:-1:1
        if global_upper[j] <= w
            low = j
            break
        end
    end

    # Return bin indices only (threshold values no longer needed)
    return (low, high)
end


"""
    simulate_layer_size_at_bin(
        bin_index::Int, bins_matrix::Matrix{<:Integer}, node_layer::NodeLayer{Z}
    ) -> Int where {Z<:Integer}

Computes exact layer size for a given bin threshold using precomputed arc-to-bin mappings.

Used by `build_layer_with_hybrid_thresholds` to compute baseline layer size at t_safe threshold.
Scans arcs in parent order to count individual nodes (from low arcs) and merged blocks
(from contiguous high arc segments). Column-major memory access pattern for cache efficiency.

# Arguments
- `bin_index::Int`: Threshold bin index; arcs in bins < bin_index expand individually
- `bins_matrix::Matrix{<:Integer}`: Precomputed bin assignments for each arc (column per node), with -1 for infeasible arcs
- `node_layer::NodeLayer{Z}`: Current layer containing arc ranges for each node

# Algorithm
For each node, scans all arcs sequentially (including infeasible arcs marked with -1):
- Infeasible arc (bin == -1): Skip
- Low arc (bin < bin_index): Creates individual node
- High arc (bin >= bin_index): Merges into contiguous block with adjacent high arcs
- Each contiguous run of high arcs contributes 1 merged node
"""
@inline function simulate_layer_size_at_bin(
    bin_index   ::Int,
    bins_matrix ::Matrix{<:Integer},
    node_layer  ::NodeLayer{Z}
) ::Int where {Z<:Integer}
    layer_size = 0

    @inbounds for node_idx in 1:node_layer.size
        bins_col = @view bins_matrix[:, node_idx]
        f_arc = node_layer.first_arcs[node_idx]
        l_arc = node_layer.last_arcs[node_idx]
        total_arcs = l_arc - f_arc + 1

        # Scan arcs in order to count exact segments
        in_high_segment = false

        @inbounds for arc_idx in 1:total_arcs
            bin = bins_col[arc_idx]

            # Skip infeasible arcs
            if bin == -1
                continue
            end

            if bin < bin_index
                # Low arc: creates individual node
                layer_size += 1

                in_high_segment = false
            else
                # High arc: part of merged block
                if !in_high_segment
                    # Start of new high segment
                    layer_size += 1

                    in_high_segment = true
                end
                # Else: continuation of current high segment, no new node
            end
        end
    end

    return layer_size
end
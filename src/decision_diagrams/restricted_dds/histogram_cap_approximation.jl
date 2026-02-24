"""
COMMENTS UP TO DATE
"""

"""
    find_cap_histogram_approximation!(layer_a::RestrictedNodeLayer, inv_coeff::V, w::Int, bkv::V, rough_bounds_int::Vector{V}, rough_bounds_cont_sum::V, var_index::Int, original_lbs_int::Union{Vector{Z}, BitVector}, coeff_times_val::Matrix{V}, bin_counts::Vector{<:Integer}, bins_matrix::Matrix{<:Integer}, cumulative_bins::Vector{<:Integer}) where {Z<:Integer, V<:Real}

Finds the threshold bin that limits the next layer to at most w nodes using bin-based precomputation with rough bound integration.

This function implements an O(Kw) bin-based approach that precomputes bin assignments for all arcs in a single pass,
then uses cumulative counts to find the threshold bin.

# Arguments
- `layer_a::RestrictedNodeLayer`: Current layer containing nodes with LTR values and implied bounds
- `inv_coeff::V`: Inverse objective coefficient (1/coeff) for multiplication instead of division
- `w::Int`: Maximum width constraint for the decision diagram layer
- `bkv::V`: Best known objective value for pruning suboptimal paths
- `rough_bounds_int::Vector{V}`: Rough bounds for remaining integer variables (overly optimistic estimates)
- `rough_bounds_cont_sum::V`: Sum of rough bounds for continuous variables
- `var_index::Int`: Index of current variable (used for rough bounds indexing)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original lower bounds before any FBBT (for coeff_times_val indexing)
- `coeff_times_val::Matrix{V}`: Precomputed arc objective contributions [max_domain_size, num_int_vars]
- `bin_counts::Vector{<:Integer}`: Preallocated workspace vector (length w) for bin counts
- `bins_matrix::Matrix{<:Integer}`: Preallocated workspace [max_domain_size, w] for arc-to-bin mapping with compact storage
- `cumulative_bins::Vector{<:Integer}`: Preallocated workspace (length w) for cumulative sum computation

# Returns
- `Tuple{Int, Bool}`: A tuple containing:
  1. `Int`: Threshold bin index
  2. `Bool`: Whether all arcs were kept without filtering (uncapped)

# Return Cases
- `(-1, true/false)`: Degenerate case (min_cap == max_cap) - objective can be ignored, enumerate arcs directly
- `(0, true)`: No feasible arcs exist (all nodes infeasible or rough bound eliminates all)
- `(0, false)`: Leftmost bin has too many arcs - cannot fit even first bin within width w
- `(1..bin_size, false)`: Normal threshold with filtering - arcs with bin ≤ threshold_bin are expanded
- `(bin_size, true)`: All arcs fit within width w - no filtering needed

# Algorithm
1. **Range Calculation**: Determines [min_cap, max_cap] based on coefficient sign using precomputed coeff_times_val
2. **Rough Bound Filtering**: Uses rough bounds to tighten range and early-exit during binning
3. **Bin Setup**: Creates w evenly-spaced bins across [min_cap, max_cap]
4. **Arc Binning** (O(Kw)): For each node and each feasible arc:
   - Computes arc objective using precomputed lookup
   - Applies rough bound pruning with early exit (monotonicity guarantees remaining arcs also fail)
   - Assigns arc to bin and stores compactly at indices 1, 2, 3, ... with -1 sentinel marking end
   - Values reconstructed during build by iterating in same direction
5. **Threshold Finding** (O(w)): Uses cache-friendly cumsum! and binary search to find threshold bin where cumulative count ≤ w

# Complexity
- O(K*w + w) = O(K*w) where K = max_domain_size
- Single pass through all arcs eliminates repeated counting
"""
@inline function find_cap_histogram_approximation!(
    layer_a     ::RestrictedNodeLayer,
    inv_coeff   ::V,
    w           ::Int,
    bkv         ::V,
    rough_bounds_int   ::Vector{V},
    rough_bounds_cont_sum ::V,
    var_index   ::Int,
    original_lbs_int ::Union{Vector{Z}, BitVector},
    coeff_times_val ::Matrix{V},
    bin_counts  ::Vector{<:Integer},
    bins_matrix ::Matrix{<:Integer},            # Preallocated arc-to-bin mapping [max_domain_size, layer_size]
    cumulative_bins ::Vector{<:Integer}               # Preallocated workspace for cumulative sum (length w)
) where {Z<:Integer, V<:Real}

    # Extract the columns
    ltrs   = layer_a.ltrs
    lb_col = layer_a.implied_lbs
    ub_col = layer_a.implied_ubs
    layer_size = layer_a.size

    # Extract precomputed column and original lower bound for this variable (for coeff_times_val indexing)
    coeff_times_val_col = @view coeff_times_val[:, var_index]
    original_lb = original_lbs_int[var_index]
    coeff_idx_offset = 1 - original_lb  # Precompute index offset for hot loop

    ############################################
    # 1) Compute min_cap, max_cap based on sign of inv_coeff using precomputed lookups
    ############################################
    # Find first feasible node (shared check for both positive and negative inv_coeff)
    first_feasible_idx = findfirst(i -> ub_col[i] >= lb_col[i], 1:layer_size)

    if first_feasible_idx === nothing
        # All nodes infeasible - exact but no arcs to expand
        return 0, true  # Threshold bin 0 = no arcs pass
    end

    if inv_coeff < 0
        min_cap = ltrs[first_feasible_idx] + coeff_times_val_col[ub_col[first_feasible_idx] - original_lb + 1]
        max_cap = ltrs[first_feasible_idx] + coeff_times_val_col[lb_col[first_feasible_idx] - original_lb + 1]

        @inbounds for node_idx in first_feasible_idx:layer_size
            lb = lb_col[node_idx]
            ub = ub_col[node_idx]

            # Skip infeasible nodes
            if ub < lb
                continue
            end

            ltr = ltrs[node_idx]

            # test with x = ub (gives minimum for negative coeff)
            test_cap = ltr + coeff_times_val_col[ub - original_lb + 1]
            if test_cap < min_cap
                min_cap = test_cap
            end
            # test with x = lb (gives maximum for negative coeff)
            test_cap = ltr + coeff_times_val_col[lb - original_lb + 1]
            if test_cap > max_cap
                max_cap = test_cap
            end
        end
    elseif inv_coeff > 0
        min_cap = ltrs[first_feasible_idx] + coeff_times_val_col[lb_col[first_feasible_idx] - original_lb + 1]
        max_cap = ltrs[first_feasible_idx] + coeff_times_val_col[ub_col[first_feasible_idx] - original_lb + 1]

        @inbounds for node_idx in first_feasible_idx:layer_size
            lb = lb_col[node_idx]
            ub = ub_col[node_idx]

            # Skip infeasible nodes
            if ub < lb
                continue
            end

            ltr = ltrs[node_idx]

            # test with x = lb (gives minimum for positive coeff)
            test_cap = ltr + coeff_times_val_col[lb - original_lb + 1]
            if test_cap < min_cap
                min_cap = test_cap
            end
            # test with x = ub (gives maximum for positive coeff)
            test_cap = ltr + coeff_times_val_col[ub - original_lb + 1]
            if test_cap > max_cap
                max_cap = test_cap
            end
        end
    else
        # inv_coeff == 0 (coeff == 0 case)
        min_cap = minimum(ltrs)
        max_cap = maximum(ltrs)
    end

    ############################################
    # 2) Apply rough bound filtering to tighten search range
    ############################################
    if var_index + 1 <= length(rough_bounds_int)
        remaining_bound = rough_bounds_int[var_index+1] + rough_bounds_cont_sum
    else
        remaining_bound = rough_bounds_cont_sum
    end
    bkv_threshold = bkv - remaining_bound  # Precompute threshold for hot loop
    effective_max_cap = min(max_cap, bkv_threshold)

    # If even the minimum possible cap exceeds BKV, all arcs are suboptimal
    if min_cap > effective_max_cap
        return 0, true  # Threshold bin 0 = no arcs pass rough bound filter
    end

    # Use tightened range for more focused search
    max_cap = effective_max_cap

    # Handle degenerate case where all arcs have identical objective values
    if min_cap == max_cap
        # Count total arcs to determine if we need filtering
        total_arcs = 0
        @inbounds for i in 1:layer_size
            total_arcs += max(ub_col[i] - lb_col[i] + 1, 0)
        end
        # Return sentinel (-1, all_fit) - caller will enumerate arcs directly
        return -1, total_arcs <= w
    end

    ############################################
    # 3) Bin-based precomputation (O(K×w) complexity)
    ############################################
    # Use full width for bin granularity
    bin_size = w  # w bins

    inv_range_width = 1 / (max_cap - min_cap)

    # Reset bin_counts
    fill!(bin_counts, 0)

    # Count total arcs for early exit check
    total_arc_count = 0

    ############################################
    # 3a) Precompute bin assignment with compact storage
    ############################################
    # Define iteration direction based on coefficient sign (no branching in hot loop)
    # Monotonicity allows early exit on rough bound failure
    make_iterator = if inv_coeff > 0
        (lb, ub) -> (lb:ub)  # Forward iteration
    elseif inv_coeff < 0
        (lb, ub) -> (ub:-1:lb)  # Backward iteration
    else
        (lb, ub) -> (lb:ub)  # coeff == 0: forward (arbitrary)
    end

    @inbounds for node_idx in 1:layer_size
        lb_i = lb_col[node_idx]
        ub_i = ub_col[node_idx]

        # Skip infeasible nodes
        if ub_i < lb_i
            continue
        end

        base_ltr = ltrs[node_idx]

        # Create view for this column (cache-friendly, reduces indexing overhead)
        bins_col = @view bins_matrix[:, node_idx]

        # Compact storage: feasible arcs stored at indices 1, 2, 3, ...
        # Values reconstructed during build by iterating in same direction
        arc_idx = 1

        @inbounds @fastmath for x in make_iterator(lb_i, ub_i)
            # Compute arc objective using precomputed lookup (with precomputed offset)
            arc_ltr = base_ltr + coeff_times_val_col[x + coeff_idx_offset]

            # Early exit on rough bound failure (using precomputed threshold)
            if arc_ltr >= bkv_threshold
                bins_col[arc_idx] = -1  # Sentinel marks end
                break
            end

            # Assign arc to bin and store compactly (fastmath safe - bin mapping is stored)
            normalized_pos = (arc_ltr - min_cap) * inv_range_width
            bin_idx = clamp(ceil(Int, normalized_pos * bin_size), 1, bin_size)

            bins_col[arc_idx] = bin_idx
            bin_counts[bin_idx] += 1
            total_arc_count += 1
            arc_idx += 1
        end

        # Sentinel to mark end of valid arcs for scanning during build phase
        bins_col[arc_idx] = -1
    end

    # Early exit: if all arcs fit, no filtering needed
    if total_arc_count <= w
        # All arcs kept - return sentinel threshold that includes everything
        return bin_size, true  # All arcs kept, no filtering
    end

    ############################################
    # 3b) Find threshold bin using cumulative counts (cache-friendly)
    ############################################
    # Compute cumulative sums (vectorized, SIMD-optimized, cache-friendly)
    cumsum!(cumulative_bins, bin_counts)

    # Binary search to find last bin where cumulative count <= w
    threshold_bin = searchsortedlast(cumulative_bins, w)

    # Return threshold bin index and filtering status
    # Build function will use bins_matrix and threshold_bin directly
    return threshold_bin, false  # Filtering occurred
end
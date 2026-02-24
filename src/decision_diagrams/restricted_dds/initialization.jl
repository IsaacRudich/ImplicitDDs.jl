"""
COMMENTS UP TO DATE
"""

"""
    generate_initial_node_layer!(node_layer::RestrictedNodeLayer, gap_buffer_a::Matrix{U}, obj_const::T, ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, infimum_gaps::Vector{U}, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}) where {Z<:Integer, T<:Real, U<:Real}

Generates the initial root layer of a restricted decision diagram by creating a node for each feasible value of the first integer variable.

# Arguments
- `node_layer::RestrictedNodeLayer`: Preallocated layer for performance efficiency - populated with root nodes
- `gap_buffer_a::Matrix{U}`: Gap buffer to write initial gaps to (always uses buffer_a for first layer)
- `obj_const::T`: Constant term in the objective function
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables (post-FBBT)
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (post-FBBT)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original lower bounds before any FBBT (for coeff_times_val indexing)
- `infimum_gaps::Vector{U}`: Precomputed infimum gap values for constraint tracking
- `coeff_times_val::Matrix{T}`: Precomputed arc objective contributions [max_domain_size, num_int_vars] (indexed with original bounds, populated once at startup)
- `gap_adjustments::Array{T,3}`: Precomputed gap adjustment values [num_constraints, max_domain_size, num_int_vars] (indexed with current bounds, repopulated by FBBT)

# Algorithm
1. Creates one node for each value in the first variable's domain [lb, ub]
2. Computes and writes infimum gaps to gap_buffer_a using precomputed gap_adjustments (indexed with current bounds)
3. Calculates Length-To-Root (LTR) values for objective tracking using precomputed coeff_times_val (indexed with original bounds)
4. Sets variable value for each node

# Returns
- `Nothing`: Layer and gap_buffer_a are populated in-place with root nodes
"""
function generate_initial_node_layer!(
    #to edit in place
    node_layer          ::RestrictedNodeLayer,
    gap_buffer_a        ::Matrix{U},

    #problem description
    obj_const               ::T,
    ubs_int                 ::Union{Vector{Z}, BitVector},
    lbs_int                 ::Union{Vector{Z}, BitVector},
    original_lbs_int        ::Union{Vector{Z}, BitVector},

    #precomputed values
    infimum_gaps        ::Vector{U},
    coeff_times_val     ::Matrix{T},
    gap_adjustments     ::Array{T,3},
)where{Z<:Integer, T<:Real, U<:Real}
    values_vec = node_layer.values
    domain = lbs_int[1]:ubs_int[1]

    # Extract column views for efficient access
    coeff_times_val_col = @view coeff_times_val[:, 1]
    gap_adjustments_col = @view gap_adjustments[:, :, 1]

    original_lb = original_lbs_int[1]
    @inbounds for (i, val) in enumerate(domain)
        # Convert value to array index using original bounds for coeff_times_val
        coeff_idx = val - original_lb + 1

        # Write infimum gaps to gap buffer using precomputed adjustments
        @inbounds @simd for j in 1:lastindex(infimum_gaps)
            gap_buffer_a[j, i] = infimum_gaps[j] + gap_adjustments_col[j, i]
        end

        # Compute ltr using precomputed coefficient-times-value
        ltr = obj_const + coeff_times_val_col[coeff_idx]

        # Set the value for this node
        values_vec[i] = val

        add_node!(node_layer, -1, ltr)
    end
end


"""
    generate_node_layer_from_queue!(qnode::QueueNode, node_layer::RestrictedNodeLayer, gap_buffer_a::Matrix{U}, base_gaps::Vector{U}) where {U<:Real}

Initializes a restricted decision diagram layer from a branch-and-bound queue node (partial solution) by copying the queue node's state and precomputed constraint gaps.

# Arguments
- `qnode::QueueNode`: Branch-and-bound queue node containing partial solution path and LTR value
- `node_layer::RestrictedNodeLayer`: Preallocated layer for performance efficiency - populated with single reconstructed node
- `gap_buffer_a::Matrix{U}`: Gap buffer to write initial gaps to (always uses buffer_a for queue node initialization)
- `base_gaps::Vector{U}`: Precomputed infimum gaps that correspond to the queue node's constraint state (already adjusted for fixed variables and local bounds)

# Algorithm
1. Creates single node with queue node's LTR value and implied bounds
2. Writes precomputed infimum gaps to gap_buffer_a
3. Sets variable value for this node (last element of queue node path)

# Returns
- `Nothing`: Layer and gap_buffer_a are populated in-place with reconstructed node state

# Notes
The base_gaps parameter should contain gaps that have already been computed for the local bounds and fixed variables (e.g., from FBBT), eliminating the need for gap recomputation in this function.
"""
@inline function generate_node_layer_from_queue!(
    qnode      ::QueueNode,
    node_layer ::RestrictedNodeLayer,
    gap_buffer_a     ::Matrix{U},
    base_gaps        ::Vector{U}
) where {U<:Real}
    # clear the node layer
    empty_layer!(node_layer)


    #most parameters are copied
    add_node!(node_layer, -1, qnode.ltr, qnode.implied_lb, qnode.implied_ub)
    cur_idx = node_layer.size

    # Gaps: Write to gap buffer
    gap_buffer_col = @view gap_buffer_a[:, cur_idx]
    @inbounds @simd for i in eachindex(base_gaps)
        gap_buffer_col[i] = base_gaps[i]
    end

    # Value: Set the variable value for this layer (last element of path)
    node_layer.values[cur_idx] = qnode.path[end]

end
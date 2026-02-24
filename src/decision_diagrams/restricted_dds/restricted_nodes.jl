"""
COMMENTS UP TO DATE
"""

mutable struct RestrictedNodeLayer{T<:Integer, U<:Real, V<:Integer}
    arcs        ::Vector{V}
    values      ::Union{Vector{T}, BitVector}
    ltrs        ::Vector{U}
    implied_lbs ::Union{Vector{T}, BitVector}
    implied_ubs ::Union{Vector{T}, BitVector}
    size        ::V
end

"""
    RestrictedNodeLayer(::Type{T}, size::Int, numerical_precision::DataType) where {T<:Integer}

Constructs a preallocated restricted decision diagram layer for performance efficiency.

# Type Parameter
- `T<:Integer`: Type of variable values (includes Int64, Bool, etc.)

# Arguments
- `::Type{T}`: Type parameter for variable values
- `size::Int`: Maximum number of nodes the layer can hold
- `numerical_precision::DataType`: Data type for floating-point calculations (e.g., Float64, Float32)

# Returns
- `RestrictedNodeLayer{T, numerical_precision, index_type}`: Empty layer with preallocated vectors (index_type selected based on size)
"""
function RestrictedNodeLayer(::Type{T}, size::Int, numerical_precision::DataType) where {T<:Integer}
    # Compute index type based on size (same logic as type_w)
    index_type = if size <= typemax(Int8)
        Int8
    elseif size <= typemax(Int16)
        Int16
    elseif size <= typemax(Int32)
        Int32
    else
        Int64
    end

    arcs = Vector{index_type}()
    ltrs = Vector{numerical_precision}()
    sizehint!(arcs, size)
    sizehint!(ltrs, size)

    if T == Bool
        values = BitVector(undef, size)
        implied_lbs = falses(size)
        implied_ubs = falses(size)
    else
        values = Vector{T}(undef, size)
        implied_lbs = zeros(T, size)
        implied_ubs = zeros(T, size)
    end

    return RestrictedNodeLayer{T, numerical_precision, index_type}(arcs, values, ltrs, implied_lbs, implied_ubs, index_type(0))
end


"""
Double-buffered gap matrices for restricted DD construction.

Gaps are separated from layers because:
1. They're large (4 MB each for typical problems)
2. They're only needed during construction (not for path reconstruction)
3. Double buffering 2 matrices is much cheaper than storing n matrices

# Type Parameter
- `U<:Real`: Numerical precision type (Float64, Float32, etc.)

# Fields
- `gap_matrix_a::Matrix{U}`: First gap matrix (num_constraints x max_width)
- `gap_matrix_b::Matrix{U}`: Second gap matrix (num_constraints x max_width)
"""
struct RestrictedGapBuffers{U<:Real}
    gap_matrix_a ::Matrix{U}
    gap_matrix_b ::Matrix{U}
end


"""
    RestrictedGapBuffers(::Type{U}, num_constraints::Int, max_width::Int) where {U<:Real}

Constructs double-buffered gap matrices for restricted DD layer construction.

# Type Parameter
- `U<:Real`: Numerical precision type (Float64, Float32, etc.)

# Arguments
- `::Type{U}`: Type parameter for gap matrix elements
- `num_constraints::Int`: Number of constraints in the problem (matrix rows)
- `max_width::Int`: Maximum layer width (matrix columns)

# Returns
- `RestrictedGapBuffers{U}`: Pair of preallocated gap matrices for alternating construction
"""
function RestrictedGapBuffers(::Type{U}, num_constraints::Int, max_width::Int) where {U<:Real}
    return RestrictedGapBuffers{U}(
        Matrix{U}(undef, num_constraints, max_width),
        Matrix{U}(undef, num_constraints, max_width)
    )
end


"""
Restricted decision diagram structure containing all layers and a 2 sets of infimum gaps for double buffering.

Stores all n layers to enable O(1) path reconstruction via parent pointers.
Gap matrices are separated and double-buffered to save memory (only need 2, not n).

# Type Parameters
- `T<:Integer`: Type of variable values (includes Int64, Bool, etc.)
- `U<:Real`: Numerical precision type (Float64, Float32, etc.)
- `V<:Integer`: Type for node indices and counts (Int8/16/32/64 based on width)

# Fields
- `layers::Vector{RestrictedNodeLayer{T,U,V}}`: All n layers for path reconstruction
- `gap_buffers::RestrictedGapBuffers{U}`: Double-buffered gap matrices (only 2 needed)
"""
struct RestrictedDD{T<:Integer, U<:Real, V<:Integer}
    layers      ::Vector{RestrictedNodeLayer{T,U,V}}
    gap_buffers ::RestrictedGapBuffers{U}
end

"""
    RestrictedDD(::Type{T}, size::Int, num_constraints::Int, num_vars::Int, numerical_precision::DataType) where {T<:Integer}

Constructs a complete restricted decision diagram with all layers and gap buffers.

# Type Parameter
- `T<:Integer`: Type of variable values (includes Int64, Bool, etc.)

# Arguments
- `::Type{T}`: Type parameter for variable values
- `size::Int`: Maximum number of nodes each layer can hold
- `num_constraints::Int`: Number of constraints in the problem
- `num_vars::Int`: Number of variables in the problem
- `numerical_precision::DataType`: Data type for floating-point calculations

# Returns
- `RestrictedDD{T, numerical_precision}`: Complete DD with n+1 preallocated layers and gap buffers

# Notes
Allocates n+1 layers (one per variable plus extra for construction swapping).
Only allocates 2 gap matrices regardless of n (double buffering).
"""
function RestrictedDD(::Type{T}, size::Int, num_constraints::Int, num_vars::Int, numerical_precision::DataType) where {T<:Integer}
    # Compute index type based on size (same logic as type_w)
    index_type = if size <= typemax(Int8)
        Int8
    elseif size <= typemax(Int16)
        Int16
    elseif size <= typemax(Int32)
        Int32
    else
        Int64
    end

    # Allocate n+1 layers (one per variable + extra for building)
    layers = Vector{RestrictedNodeLayer{T, numerical_precision, index_type}}(undef, num_vars + 1)
    @inbounds for i in 1:(num_vars + 1)
        layers[i] = RestrictedNodeLayer(T, size, numerical_precision)
    end

    # Create gap buffers (only 2 needed)
    gap_buffers = RestrictedGapBuffers(numerical_precision, num_constraints, size)

    return RestrictedDD{T, numerical_precision, index_type}(layers, gap_buffers)
end

"""
    reconstruct_path!(path_buffer::Union{Vector{Z}, BitVector}, layers::Vector{RestrictedNodeLayer{Z,U,V}}, terminal_node::Int, qnode::Union{QueueNode, Nothing}, start_layer::Int, end_layer::Int) where {Z<:Integer, U<:Real, V<:Integer}

Reconstruct complete path by backtracking through layer history using parent pointers, with optional queue node prefix.

# Arguments
- `path_buffer::Union{Vector{Z}, BitVector}`: Output buffer to store reconstructed path (length n)
- `layers::Vector{RestrictedNodeLayer{Z,U}}`: All layers from DD construction
- `terminal_node::Int`: Node index in the terminal layer
- `qnode::Union{QueueNode, Nothing}`: Queue node containing prefix path (nothing if started from root)
- `start_layer::Int`: First layer to reconstruct from (1 for root, length(qnode.path) for queue node)
- `end_layer::Int`: Last layer to reconstruct from (usually num_int_vars)

# Algorithm
1. Backtrack from end_layer to start_layer using parent pointers:
   - Read variable value from current layer: `path[i] = layer.values[terminal_node]`
   - Follow parent pointer: `terminal_node = layer.arcs[terminal_node]`
2. If start_layer > 1, prepend the queue node's prefix (variables 1 through start_layer-1)

# Notes
Handles both root-based construction (qnode = nothing, start_layer = 1) and queue-node-based construction (qnode provided, start_layer > 1).
```
"""
@inline function reconstruct_path!(
    path_buffer  ::Union{Vector{Z}, BitVector},
    layers       ::Vector{RestrictedNodeLayer{Z,U,V}},
    terminal_node::Int,
    qnode        ::Union{QueueNode, Nothing},
    start_layer  ::Int,
    end_layer    ::Int
) where {Z<:Integer, U<:Real, V<:Integer}
    # Reconstruct the suffix (variables start_layer through end_layer)
    @inbounds for layer_idx in end_layer:-1:start_layer
        layer = layers[layer_idx]
        path_buffer[layer_idx] = layer.values[terminal_node]
        terminal_node = layer.arcs[terminal_node]  # Follow parent pointer
    end

    # Prepend queue node prefix if needed
    if start_layer > 1
        path_buffer[1:start_layer-1] .= qnode.path[1:start_layer-1]
    end
end


"""
    add_node!(layer::RestrictedNodeLayer{V,U,W}, arc::T, ltr::U) where{V<:Integer, U<:Real, W<:Integer, T<:Integer}

Adds a new node to the restricted layer with arc reference and length-to-root value.

# Arguments
- `layer::RestrictedNodeLayer{V,U,W}`: Layer to add the node to
- `arc::T`: Arc reference from parent node (or -1 for root nodes)
- `ltr::U`: Length-to-root objective value for this node

# Returns
- `Int32`: The size of the layer after adding the node
"""
@inline function add_node!(layer::RestrictedNodeLayer{V,U,W}, arc::T, ltr::U) where{V<:Integer, U<:Real, W<:Integer, T<:Integer}
    push!(layer.arcs, arc)
    push!(layer.ltrs, ltr)

    layer.size += 1

    return layer.size
end

"""
    add_node!(layer::RestrictedNodeLayer{V,U,W}, arc::T, ltr::U, implied_lb::V, implied_ub::V) where{V<:Integer, U<:Real, W<:Integer, T<:Integer}

Adds a new node to the restricted layer with arc reference, length-to-root value, and implied bounds.

# Arguments
- `layer::RestrictedNodeLayer{V,U,W}`: Layer to add the node to
- `arc::T`: Arc reference from parent node (or -1 for root nodes)
- `ltr::U`: Length-to-root objective value for this node
- `implied_lb::V`: Implied lower bound for the next variable at this node
- `implied_ub::V`: Implied upper bound for the next variable at this node

# Returns
- `Int32`: The size of the layer after adding the node
"""
@inline function add_node!(layer::RestrictedNodeLayer{V,U,W}, arc::T, ltr::U, implied_lb::V, implied_ub::V) where{V<:Integer, U<:Real, W<:Integer, T<:Integer}
    push!(layer.arcs, arc)
    push!(layer.ltrs, ltr)

    layer.size += 1

    # Use indexed assignment instead of push! for pre-allocated arrays
    layer.implied_lbs[layer.size] = implied_lb
    layer.implied_ubs[layer.size] = implied_ub

    return layer.size
end

"""
    empty_layer!(layer::RestrictedNodeLayer{T,U,V}) where {T<:Integer, U<:Real, V<:Integer}

Clears all nodes from the restricted layer while preserving preallocated memory.

# Arguments
- `layer::RestrictedNodeLayer{T,U,V}`: Layer to clear

# Returns
- `Nothing`: Layer is cleared in-place
"""
@inline function empty_layer!(layer::RestrictedNodeLayer{T,U,V}) where {T<:Integer, U<:Real, V<:Integer}
    empty!(layer.arcs)
    empty!(layer.ltrs)
    layer.size = 0
end

Base.show(io::IO, node_layer::RestrictedNodeLayer{T,U,V}) where {T<:Integer, U<:Real, V<:Integer} = pretty_print_node_layer(node_layer)

"""
    pretty_print_node_layer(nl::RestrictedNodeLayer{T,U,V}) where {T<:Integer, U<:Real, V<:Integer}

Prints a human-readable representation of all nodes in the restricted layer.

# Arguments
- `nl::RestrictedNodeLayer{T,U,V}`: Layer to display

# Returns
- `Nothing`: Prints node information to console
"""
function pretty_print_node_layer(nl::RestrictedNodeLayer{T,U,V}) where {T<:Integer, U<:Real, V<:Integer}
    for i in 1:nl.size
        print(" (idx:",i, ", ltr:",nl.ltrs[i],", bounds:[", nl.implied_lbs[i], ",", nl.implied_ubs[i], "]) ")
    end
end
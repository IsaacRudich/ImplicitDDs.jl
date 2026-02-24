"""
COMMENTS UP TO DATE
"""

"""
    NodeLayer{T}

Parameterized mutable struct representing a layer of merged nodes in a relaxed decision diagram.

Relaxed DDs use interval-based merging where each node stores its parent
arcs as a range of indexes `[first_arc, last_arc]` to enable efficient construction.

# Type Parameter
- `T<:Integer`: Type of variable values stored in the layer (includes Int64, Int32, Bool, etc.)

# Fields
- `first_arcs::Vector{Int32}`: Starting index of parent arc range for each node
- `last_arcs::Vector{Int32}`: Ending index of parent arc range for each node
- `values::Union{Vector{T}, BitVector}`: Variable value assigned to each node (BitVector when T=Bool, Vector{T} otherwise)
- `active::BitVector`: Activation status for rough bound pruning (true = active, false = pruned)
- `size::Int32`: Current number of nodes in the layer

# Mathematical Interpretation
Each node i represents all arcs from parent `first_arcs[i]` to parent `last_arcs[i]`
(inclusive) that assign value `values[i]` to the current variable. This interval-based
representation enables:
- **Node merging**: Multiple parent paths converge to single child nodes
- **Relaxation**: Merged paths may not correspond to actual feasible solutions
- **Lower bounds**: Provides valid bounds through constraint relaxation
"""
mutable struct NodeLayer{T<:Integer}
    first_arcs ::Vector{Int32}
    last_arcs  ::Vector{Int32}
    values     ::Union{Vector{T}, BitVector}
    active     ::BitVector
    size       ::Int32
end

"""
    NodeLayer{T}(size::Int) where {T<:Integer}

Constructs a preallocated empty relaxed decision diagram layer for performance efficiency.

# Type Parameter
- `T<:Integer`: Type of variable values (includes Int64, Bool, etc.)

# Arguments
- `size::Int`: Maximum number of nodes the layer can hold (used for preallocation)

# Returns
- `NodeLayer{T}`: Empty layer with preallocated vectors for efficient node addition
"""
function NodeLayer{T}(size::Int) where {T<:Integer}
    first_arcs = Vector{Int32}()
    last_arcs = Vector{Int32}()
    if T == Bool
        values = BitVector()
    else
        values = Vector{T}()
    end
    sizehint!(first_arcs, size)
    sizehint!(last_arcs, size)
    sizehint!(values, size)
    return NodeLayer{T}(first_arcs, last_arcs, values, falses(size), 0)
end

"""
    NodeMatrix(::Type{T}, num_layers::Int, layer_width::Int) where {T<:Integer}

Constructs a complete relaxed decision diagram matrix with preallocated layers.

# Type Parameter
- `T<:Integer`: Type of variable values (includes Int64, Bool, etc.)

# Arguments
- `::Type{T}`: Type parameter for the node values
- `num_layers::Int`: Number of variable layers in the decision diagram
- `layer_width::Int`: Maximum width (number of nodes) for each layer

# Returns
- `Vector{NodeLayer{T}}`: Vector of preallocated empty NodeLayer objects, one per variable

# Notes
Each layer is initialized to hold up to `layer_width` nodes. This structure supports
the full DD construction pipeline from initialization through post-processing.
"""
function NodeMatrix(::Type{T}, num_layers::Int, layer_width::Int) where {T<:Integer}
    nm = Vector{NodeLayer{T}}()
    sizehint!(nm, num_layers)
    @inbounds for i in 1:num_layers
        push!(nm, NodeLayer{T}(layer_width))
    end
    return nm
end

"""
    add_node!(layer::NodeLayer{V}, first_arc::T, last_arc::U, value::V) where {V<:Integer, T<:Integer, U<:Integer}

Adds a new merged node to the relaxed decision diagram layer.

# Arguments
- `layer::NodeLayer{V}`: Target layer to add the node to
- `first_arc::T`: Starting index of the parent arc range (inclusive)
- `last_arc::U`: Ending index of the parent arc range (inclusive)
- `value::V`: Variable value assigned to this node (type matches layer's type parameter)

# Returns
- `Int32`: The size of the layer after adding the node

# Algorithm
1. Appends the arc range `[first_arc, last_arc]` and variable value to the layer
2. Increments the layer size counter
3. Activates the new node (sets active[new_size] = true)
4. Returns the updated layer size

# Mathematical Interpretation
The new node represents all arcs from parent nodes `first_arc` through `last_arc`
that assign `value` to the current variable.
"""
@inline function add_node!(layer::NodeLayer{V}, first_arc::T, last_arc::U, value::V) where {V<:Integer, T<:Integer, U<:Integer}
    push!(layer.first_arcs, first_arc)
    push!(layer.last_arcs, last_arc)
    push!(layer.values, value)
    layer.size += 1
    layer.active[layer.size] = true
    return layer.size
end

"""
    empty_layer!(layer::NodeLayer{T}) where {T<:Integer}

Clears all nodes from the relaxed decision diagram layer while preserving preallocated memory.

# Arguments
- `layer::NodeLayer{T}`: Layer to be cleared

# Returns
- `Nothing`: Layer is cleared in-place

# Notes
This function preserves the preallocated capacity of internal vectors for performance
efficiency during layer reuse in DD construction algorithms. The cleared layer can
be immediately reused without additional memory allocation.
"""
@inline function empty_layer!(layer::NodeLayer{T}) where {T<:Integer}
    empty!(layer.first_arcs)
    empty!(layer.last_arcs)
    empty!(layer.values)
    layer.active .= false
    layer.size = 0
end



Base.show(io::IO, node_layer::NodeLayer{T}) where {T<:Integer} = pretty_print_node_layer(node_layer)

"""
    pretty_print_node_layer(nl::NodeLayer{T}) where {T<:Integer}

Prints a human-readable representation of all nodes in the relaxed decision diagram layer.

# Arguments
- `nl::NodeLayer{T}`: Layer to display

# Returns
- `Nothing`: Prints node information to console

# Output Format
For each node i, prints: (idx:i, nval:value, arcs:[first_arc,last_arc])
where:
- `idx`: Node index in the layer
- `nval`: Variable value assigned to this node
- `arcs`: Arc range [first_arc, last_arc] representing merged parent intervals

# Notes
This function provides a compact visualization of the layer's node structure,
showing how parent arc ranges are merged into individual child nodes. Used
primarily for debugging and understanding relaxed DD structure.
"""
function pretty_print_node_layer(nl::NodeLayer{T}) where {T<:Integer}
    for i in 1:nl.size
        print(" (idx:",i, ", nval:",nl.values[i], ", arcs:[", nl.first_arcs[i], ",", nl.last_arcs[i], "]) ")
    end
end
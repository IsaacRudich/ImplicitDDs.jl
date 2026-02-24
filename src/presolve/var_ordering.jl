"""
COMMENTS UP TO DATE
"""


"""
    VariableBlock

Represents a block of variables with ordering information for decision diagrams.

# Fields
- `variables::Vector{Int}`: Variable indices in this block (positions in original ordering)
- `conductance::Float64`: Conductance value of the split that created this block
- `external_degree::Int`: Number of cross-block edges from this block
- `is_boundary::Vector{Bool}`: True if variable has neighbors outside this block
"""
struct VariableBlock
    variables::Vector{Int}
    conductance::Float64
    external_degree::Int
    is_boundary::Vector{Bool}
end


"""
    adaptive_metis_variable_ordering(model::MOI.Utilities.GenericModel,
                                    int_var_order::Vector{MOI.VariableIndex},
                                    num_int_vars::Int,
                                    ::Type{T};
                                    beta::Float64=0.20,
                                    phi_target::Float64=0.12,
                                    epsilon::Float64=0.05) where {T<:AbstractFloat}

Computes adaptive METIS-based variable ordering using recursive bisection with conductance-based acceptance.

Uses decision diagram-aware heuristics: boundary variables (those with cross-block connections)
are prioritized early in the ordering to minimize frontier growth during DD construction.

# Arguments
- `model::MOI.Utilities.GenericModel`: MOI model containing constraints
- `int_var_order::Vector{MOI.VariableIndex}`: Current integer variable ordering
- `num_int_vars::Int`: Number of integer variables
- `::Type{T}`: Floating point precision type for constraint processing
- `beta::Float64`: Minimum balance ratio (default: 0.20)
- `phi_target::Float64`: Target conductance threshold (default: 0.12)
- `epsilon::Float64`: Required improvement factor (default: 0.05)

# Returns
- `Vector{Int}`: New variable ordering optimized for decision diagram construction
"""
function adaptive_metis_variable_ordering(
    model::MOI.Utilities.GenericModel,
    int_var_order::Vector{MOI.VariableIndex},
    num_int_vars::Int,
    ::Type{T};
    beta::Float64=0.20,
    phi_target::Float64=0.12,
    epsilon::Float64=0.05
) where {T<:AbstractFloat}

    if num_int_vars <= 4
        return collect(1:num_int_vars)
    end


    # Build adjacency graph
    adjacency = build_variable_adjacency_from_moi(model, int_var_order, num_int_vars, T)


    # Find connected components
    g = SimpleGraph(num_int_vars)
    rows, cols, _ = findnz(adjacency)
    @inbounds for i in 1:lastindex(rows)
        if rows[i] < cols[i]
            add_edge!(g, rows[i], cols[i])
        end
    end

    # Handle disconnected case
    if ne(g) == 0
        return collect(1:num_int_vars)
    end

    # Get connected components as initial blocks
    components = connected_components(g)

    # Initialize processing stack with root blocks
    stack = Vector{Tuple{Vector{Int}, Float64}}()  # (block_variables, parent_conductance)
    for component in components
        push!(stack, (component, Inf))
    end

    # Process blocks recursively
    final_blocks = Vector{VariableBlock}()

    while !isempty(stack)
        block_vars, parent_conductance = pop!(stack)

        # Attempt to split the block
        split_result = attempt_block_split(adjacency, block_vars, beta, phi_target, epsilon, parent_conductance)

        if split_result === nothing
            # Split rejected - add as final block
            block_set = Set(block_vars)
            is_boundary = compute_boundary_status(adjacency, block_vars, block_set, num_int_vars)
            external_deg = count_external_degree(adjacency, block_vars, block_set, num_int_vars)
            push!(final_blocks, VariableBlock(block_vars, parent_conductance, external_deg, is_boundary))
        else
            # Split accepted - add children to stack
            left_block, right_block, conductance = split_result
            push!(stack, (left_block, conductance))
            push!(stack, (right_block, conductance))
        end
    end

    # Order blocks using block quotient graph for optimal connectivity
    ordered_blocks = order_blocks_by_connectivity(final_blocks, adjacency)

    # Build final variable ordering
    final_ordering = Vector{Int}()
    for (block_position, block) in enumerate(ordered_blocks)
        block_order = order_variables_within_block(adjacency, block, block_position, ordered_blocks)
        append!(final_ordering, block_order)
    end

    return final_ordering
end


"""
    build_variable_adjacency_from_moi(model::MOI.Utilities.GenericModel,
                                     int_var_order::Vector{MOI.VariableIndex},
                                     num_int_vars::Int,
                                     ::Type{T}) where {T<:AbstractFloat}

Builds a variable-variable adjacency graph directly from MOI model constraints without constructing the full coefficient matrix.

Variables are connected if they appear together in the same constraint. Only considers integer variables.

# Arguments
- `model::MOI.Utilities.GenericModel`: MOI model containing constraints
- `int_var_order::Vector{MOI.VariableIndex}`: Integer variable ordering from MOI
- `num_int_vars::Int`: Number of integer variables
- `::Type{T}`: Floating point precision type for constraint processing

# Returns
- `SparseMatrixCSC{Bool}`: Symmetric adjacency matrix (num_int_vars * num_int_vars)
"""
function build_variable_adjacency_from_moi(
    model::MOI.Utilities.GenericModel,
    int_var_order::Vector{MOI.VariableIndex},
    num_int_vars::Int,
    ::Type{T}
) where {T<:AbstractFloat}

    # Create mapping from MOI variable index to position in int_var_order
    int_var_to_pos = Dict{MOI.VariableIndex, Int}()
    for (i, var) in enumerate(int_var_order)
        int_var_to_pos[var] = i
    end

    # Initialize adjacency matrix
    adjacency = spzeros(Bool, num_int_vars, num_int_vars)

    # Get all constraint types
    leq_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}())
    eq_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}())
    geq_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}())

    # Process each constraint type
    all_constraints = [leq_cons; eq_cons; geq_cons]

    int_vars_in_constraint = Vector{Int}()
    sizehint!(int_vars_in_constraint, num_int_vars)

    @inbounds for con in all_constraints
        # Get constraint function
        f = MOI.get(model, MOI.ConstraintFunction(), con)

        # Find all integer variables in this constraint
        empty!(int_vars_in_constraint)
        @inbounds for term in f.terms
            if haskey(int_var_to_pos, term.variable)
                push!(int_vars_in_constraint, int_var_to_pos[term.variable])
            end
        end

        # Connect all pairs of integer variables in this constraint
        for i in eachindex(int_vars_in_constraint)
            @inbounds for j in (i+1):lastindex(int_vars_in_constraint)
                var1 = int_vars_in_constraint[i]
                var2 = int_vars_in_constraint[j]
                adjacency[var1, var2] = true
                adjacency[var2, var1] = true  # Ensure symmetry
            end
        end
    end

    return adjacency
end


"""
    attempt_block_split(adjacency::SparseMatrixCSC{Bool}, block_vars::Vector{Int}, beta::Float64, phi_target::Float64, epsilon::Float64, parent_conductance::Float64)

Attempts to split a block using METIS and validates the split using conductance criteria.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `block_vars::Vector{Int}`: Variables in the block to split
- `beta::Float64`: Minimum balance ratio requirement
- `phi_target::Float64`: Target conductance threshold
- `epsilon::Float64`: Required improvement factor
- `parent_conductance::Float64`: Conductance of parent block

# Returns
- `Union{Nothing, Tuple{Vector{Int}, Vector{Int}, Float64}}`: Returns nothing if split is rejected, otherwise tuple containing:
  1. `Vector{Int}`: Left block variables
  2. `Vector{Int}`: Right block variables
  3. `Float64`: Conductance of the split
"""
function attempt_block_split(
    adjacency::SparseMatrixCSC{Bool},
    block_vars::Vector{Int},
    beta::Float64,
    phi_target::Float64,
    epsilon::Float64,
    parent_conductance::Float64
)
    # Extract subgraph for this block
    n_block = length(block_vars)

    # Build local adjacency matrix
    local_adj = spzeros(Bool, n_block, n_block)
    for i in 1:n_block
        @inbounds @simd for j in (i+1):n_block
            var1, var2 = block_vars[i], block_vars[j]
            if adjacency[var1, var2]
                local_adj[i, j] = true
                local_adj[j, i] = true
            end
        end
    end

    # Create graph and attempt METIS bisection
    local_graph = SimpleGraph(n_block)
    rows, cols, _ = findnz(local_adj)
    for i in 1:lastindex(rows)
        if rows[i] < cols[i]
            add_edge!(local_graph, rows[i], cols[i])
        end
    end

    if ne(local_graph) == 0
        return nothing  # No edges to split on
    end

    try
        # Use METIS to bisect
        partition_labels, _ = Metis.partition(local_graph, 2)

        # Convert back to original variable indices
        left_indices = Vector{Int}()
        right_indices = Vector{Int}()
        @inbounds for (local_idx, partition_id) in enumerate(partition_labels)
            if partition_id == 1
                push!(left_indices, block_vars[local_idx])
            else
                push!(right_indices, block_vars[local_idx])
            end
        end

        # Compute conductance once
        conductance = compute_conductance(adjacency, left_indices, right_indices)

        # Validate split criteria using precomputed conductance
        if !is_split_acceptable_with_conductance(left_indices, right_indices, conductance, beta, phi_target, epsilon, parent_conductance)
            return nothing
        end

        return (left_indices, right_indices, conductance)

    catch e
        return nothing  # METIS failed
    end
end


"""
    compute_conductance(adjacency::SparseMatrixCSC{Bool}, left::Vector{Int}, right::Vector{Int})

Computes the conductance of a cut between two sets of variables.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `left::Vector{Int}`: Variables in left partition
- `right::Vector{Int}`: Variables in right partition

# Returns
- `Float64`: Conductance value (cut_edges / min_volume)
"""
function compute_conductance(adjacency::SparseMatrixCSC{Bool}, left::Vector{Int}, right::Vector{Int})
    # Count cut edges
    cut_edges = 0
    for v1 in left
        @inbounds for v2 in right
            if adjacency[v1, v2]
                cut_edges += 1
            end
        end
    end

    if cut_edges == 0
        return 0.0
    end

    # Compute volumes (degree sums within the entire split L ∪ R)
    all_split_vars = [left; right]
    vol_left = compute_volume_in_split(adjacency, left, all_split_vars)
    vol_right = compute_volume_in_split(adjacency, right, all_split_vars)

    if min(vol_left, vol_right) == 0
        return Inf
    end

    return cut_edges / min(vol_left, vol_right)
end


"""
    compute_volume_in_split(adjacency::SparseMatrixCSC{Bool}, partition::Vector{Int}, all_split_vars::Vector{Int})

Computes the volume (sum of degrees within the split) for a partition in a 2-way split.

Volume counts all edges from partition variables to ANY variable in the split (both partitions),
not just internal edges within the partition itself.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `partition::Vector{Int}`: Variables in this partition (L or R)
- `all_split_vars::Vector{Int}`: All variables in the split (L U R)

# Returns
- `Int`: Sum of degrees within the split for this partition
"""
function compute_volume_in_split(adjacency::SparseMatrixCSC{Bool}, partition::Vector{Int}, all_split_vars::Vector{Int})
    volume = 0
    for var1 in partition
        @inbounds for var2 in all_split_vars
            if var1 != var2 && adjacency[var1, var2]
                volume += 1
            end
        end
    end
    return volume
end


"""
    is_split_acceptable_with_conductance(left::Vector{Int}, right::Vector{Int}, conductance::Float64, beta::Float64, phi_target::Float64, epsilon::Float64, parent_conductance::Float64)

Validates a proposed split using balance and precomputed conductance criteria.

# Arguments
- `left::Vector{Int}`: Variables in left partition
- `right::Vector{Int}`: Variables in right partition
- `conductance::Float64`: Precomputed conductance value for the split
- `beta::Float64`: Minimum balance ratio requirement
- `phi_target::Float64`: Target conductance threshold
- `epsilon::Float64`: Required improvement factor
- `parent_conductance::Float64`: Conductance of parent block

# Returns
- `Bool`: True if split meets all acceptance criteria
"""
function is_split_acceptable_with_conductance(
    left::Vector{Int},
    right::Vector{Int},
    conductance::Float64,
    beta::Float64,
    phi_target::Float64,
    epsilon::Float64,
    parent_conductance::Float64
)
    left_size = length(left)
    right_size = length(right)
    total_size = left_size + right_size
    min_size = min(left_size, right_size)

    # Balance check
    if min_size / total_size < beta
        return false
    end

    # Conductance threshold
    if isinf(parent_conductance)
        # Root split
        return conductance <= phi_target
    else
        # Non-root split - require improvement
        return conductance <= min(phi_target, (1.0 - epsilon) * parent_conductance)
    end
end


"""
    compute_boundary_status(adjacency::SparseMatrixCSC{Bool}, block_vars::Vector{Int}, block_set::Set{Int}, num_int_vars::Int)

Determines which variables in a block have neighbors outside the block.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `block_vars::Vector{Int}`: Variables in the block
- `block_set::Set{Int}`: Set of variables in the block (for efficient lookup)
- `num_int_vars::Int`: Total number of integer variables

# Returns
- `Vector{Bool}`: Boolean vector indicating which variables are boundary variables
"""
function compute_boundary_status(adjacency::SparseMatrixCSC{Bool}, block_vars::Vector{Int}, block_set::Set{Int}, num_int_vars::Int)
    is_boundary = falses(length(block_vars))

    for (i, var) in enumerate(block_vars)
        @inbounds for other_var in 1:num_int_vars
            if !(other_var in block_set) && adjacency[var, other_var]
                is_boundary[i] = true
                break
            end
        end
    end

    return is_boundary
end


"""
    count_external_degree(adjacency::SparseMatrixCSC{Bool}, block_vars::Vector{Int}, block_set::Set{Int}, num_int_vars::Int)

Counts the total number of edges from this block to other blocks.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `block_vars::Vector{Int}`: Variables in the block
- `block_set::Set{Int}`: Set of variables in the block (for efficient lookup)
- `num_int_vars::Int`: Total number of integer variables

# Returns
- `Int`: Total number of external edges from the block
"""
function count_external_degree(adjacency::SparseMatrixCSC{Bool}, block_vars::Vector{Int}, block_set::Set{Int}, num_int_vars::Int)
    external_edges = 0

    for var in block_vars
        @inbounds for other_var in 1:num_int_vars
            if !(other_var in block_set) && adjacency[var, other_var]
                external_edges += 1
            end
        end
    end

    return external_edges
end


"""
    order_blocks_by_connectivity(blocks::Vector{VariableBlock}, adjacency::SparseMatrixCSC{Bool})

Orders blocks based on their connectivity patterns using a block quotient graph approach.

Creates a quotient graph where each block is a node and edge weights represent
the number of connections between blocks, then finds an ordering that maximizes
contiguity of strongly connected blocks for efficient decision diagram construction.

# Arguments
- `blocks::Vector{VariableBlock}`: Final blocks from partitioning
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix

# Returns
- `Vector{VariableBlock}`: Blocks ordered for maximal contiguity and DD efficiency
"""
function order_blocks_by_connectivity(blocks::Vector{VariableBlock}, adjacency::SparseMatrixCSC{Bool})
    num_blocks = length(blocks)

    if num_blocks <= 1
        return blocks
    end 

    # Build block quotient graph
    block_connectivity = build_block_quotient_graph(blocks, adjacency)

    # Find ordering that minimizes edge cuts
    block_ordering = find_optimal_block_ordering(block_connectivity, blocks)

    return blocks[block_ordering]
end


"""
    build_block_quotient_graph(blocks::Vector{VariableBlock}, adjacency::SparseMatrixCSC{Bool})

Constructs a quotient graph where nodes represent blocks and edge weights represent connectivity.

# Arguments
- `blocks::Vector{VariableBlock}`: Blocks to analyze
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix

# Returns
- `Matrix{Int}`: Symmetric matrix where entry (i,j) is the number of edges between block i and block j
"""
function build_block_quotient_graph(blocks::Vector{VariableBlock}, adjacency::SparseMatrixCSC{Bool})
    num_blocks = length(blocks)
    block_connectivity = zeros(Int, num_blocks, num_blocks)

    # Create block membership lookup
    var_to_block = Dict{Int, Int}()
    for (block_idx, block) in enumerate(blocks)
        @inbounds for var in block.variables
            var_to_block[var] = block_idx
        end
    end

    # Count inter-block edges by iterating over actual edges in adjacency matrix
    rows, cols, _ = findnz(adjacency)
    @inbounds for i in 1:lastindex(rows)
        var1, var2 = rows[i], cols[i]
        if var1 < var2  # Only count each edge once (exploit symmetry)
            block1 = var_to_block[var1]
            block2 = var_to_block[var2]
            if block1 != block2
                # Cross-block edge found - increment both directions
                block_connectivity[block1, block2] += 1
                block_connectivity[block2, block1] += 1
            end
        end
    end

    return block_connectivity
end


"""
    find_optimal_block_ordering(connectivity::Matrix{Int}, blocks::Vector{VariableBlock})

Finds a block ordering that maximizes contiguity of strongly connected blocks.

Uses a greedy approach with two-ended placement optimized for decision diagram efficiency:
start with the highest external degree block, then repeatedly select the block with
highest connectivity to already-selected blocks and place it at whichever end (front
or back) has stronger connectivity to its immediate neighbor. This creates tighter
clusters while keeping strongly connected variables close together for efficient
constraint propagation during DD construction.

# Arguments
- `connectivity::Matrix{Int}`: Block-to-block connectivity matrix
- `blocks::Vector{VariableBlock}`: Blocks to order

# Returns
- `Vector{Int}`: Ordering of block indices
"""
function find_optimal_block_ordering(connectivity::Matrix{Int}, blocks::Vector{VariableBlock})
    num_blocks = length(blocks)

    if num_blocks <= 2
        # For 1-2 blocks, order by external degree
        return sortperm(blocks, by=b -> b.external_degree, rev=true)
    end

    # Greedy connectivity-based ordering with two-ended placement
    selected = Set{Int}()
    ordering = Vector{Int}()

    # Start with highest external degree block for DD efficiency
    # (Alternative: use argmin for crossing minimization, but contiguity is better for DDs)
    external_degrees = [b.external_degree for b in blocks]
    start_block = argmax(external_degrees)
    push!(ordering, start_block)
    push!(selected, start_block)

    # Initialize connectivity sums with connections to start_block
    candidate_connectivities = zeros(Int, num_blocks)
    @inbounds for candidate in 1:num_blocks
        if candidate != start_block
            candidate_connectivities[candidate] = connectivity[candidate, start_block]
        end
    end

    # Repeatedly add the block with highest connectivity to already-selected blocks
    while length(selected) < num_blocks
        best_block = 0
        best_connectivity = -1

        # Find best unselected candidate
        @inbounds for candidate in 1:num_blocks
            if !(candidate in selected)
                total_connectivity = candidate_connectivities[candidate]

                # Tie-break by external degree (prioritize high-connectivity blocks)
                if total_connectivity > best_connectivity || (total_connectivity == best_connectivity && blocks[candidate].external_degree > blocks[best_block].external_degree)
                    best_block = candidate
                    best_connectivity = total_connectivity
                end
            end
        end

        # Update connectivity sums incrementally for next iteration
        @inbounds for candidate in 1:num_blocks
            if !(candidate in selected) && candidate != best_block
                candidate_connectivities[candidate] += connectivity[candidate, best_block]
            end
        end

        # Two-ended placement: try front and back, choose side with higher connectivity to neighbor
        if length(ordering) == 1
            # Only one block so far, just append
            push!(ordering, best_block)
        else
            front_neighbor = ordering[1]
            back_neighbor = ordering[end]
            front_connectivity = connectivity[best_block, front_neighbor]
            back_connectivity = connectivity[best_block, back_neighbor]

            if front_connectivity > back_connectivity
                # Place at front
                pushfirst!(ordering, best_block)
            else
                # Place at back (default for ties)
                push!(ordering, best_block)
            end
        end

        push!(selected, best_block)
    end

    return ordering
end


"""
    order_variables_within_block(adjacency::SparseMatrixCSC{Bool}, block::VariableBlock, block_position::Int, all_blocks::Vector{VariableBlock})

Orders variables within a block using directional boundary strategy for decision diagram efficiency.

Categorizes variables into three groups based on their external connections:
- Left boundary: Connected to earlier blocks in the ordering
- Interior: Only connected within the current block
- Right boundary: Connected to later blocks in the ordering

This creates a "flow" through the block that aligns with the overall block ordering.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `block::VariableBlock`: Block containing variables and boundary information
- `block_position::Int`: Position of this block in the overall ordering (1-indexed)
- `all_blocks::Vector{VariableBlock}`: All blocks in their final ordering

# Returns
- `Vector{Int}`: Ordered list of variables (left boundary, interior, right boundary)
"""
function order_variables_within_block(adjacency::SparseMatrixCSC{Bool}, block::VariableBlock, block_position::Int, all_blocks::Vector{VariableBlock})
    # Build sets for efficient lookups
    earlier_blocks_vars = Set{Int}()
    later_blocks_vars = Set{Int}()

    # Collect variables from earlier blocks (positions 1 to block_position-1)
    @inbounds for i in 1:(block_position-1)
        for var in all_blocks[i].variables
            push!(earlier_blocks_vars, var)
        end
    end

    # Collect variables from later blocks (positions block_position+1 to end)
    @inbounds for i in (block_position+1):length(all_blocks)
        for var in all_blocks[i].variables
            push!(later_blocks_vars, var)
        end
    end

    # Categorize variables based on their external connections
    left_boundary = Vector{Int}()
    right_boundary = Vector{Int}()
    interior = Vector{Int}()

    @inbounds for var in block.variables
        has_left_connections = false
        has_right_connections = false

        # Check connections to earlier blocks
        for other_var in earlier_blocks_vars
            if adjacency[var, other_var]
                has_left_connections = true
                break
            end
        end

        # Check connections to later blocks
        for other_var in later_blocks_vars
            if adjacency[var, other_var]
                has_right_connections = true
                break
            end
        end

        # Categorize based on connection pattern
        if has_left_connections && has_right_connections
            # Connected to both sides - use stronger connection to decide
            left_strength = count_connections_to_set(adjacency, var, earlier_blocks_vars)
            right_strength = count_connections_to_set(adjacency, var, later_blocks_vars)

            if left_strength > right_strength
                push!(left_boundary, var)
            elseif right_strength > left_strength
                push!(right_boundary, var)
            else
                # True tie - default to left boundary (process early for safety)
                push!(left_boundary, var)
            end
        elseif has_left_connections
            push!(left_boundary, var)
        elseif has_right_connections
            push!(right_boundary, var)
        else
            push!(interior, var)
        end
    end

    # Sort left boundary by connections to earlier blocks (descending)
    left_degrees = [(var, count_connections_to_set(adjacency, var, earlier_blocks_vars)) for var in left_boundary]
    sort!(left_degrees, by=x -> x[2], rev=true)
    ordered_left = [var for (var, _) in left_degrees]

    # Sort interior by internal degree (descending)
    interior_degrees = [(var, count_internal_connections(adjacency, var, block.variables)) for var in interior]
    sort!(interior_degrees, by=x -> x[2], rev=true)
    ordered_interior = [var for (var, _) in interior_degrees]

    # Sort right boundary by connections to later blocks (ascending - most connected last)
    right_degrees = [(var, count_connections_to_set(adjacency, var, later_blocks_vars)) for var in right_boundary]
    sort!(right_degrees, by=x -> x[2], rev=false)
    ordered_right = [var for (var, _) in right_degrees]

    # Directional flow: left boundary → interior → right boundary
    return [ordered_left; ordered_interior; ordered_right]
end


"""
    count_connections_to_set(adjacency::SparseMatrixCSC{Bool}, var::Int, target_set::Set{Int})

Counts connections from a variable to variables in a specific target set.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `var::Int`: Variable to count connections for
- `target_set::Set{Int}`: Set of variables to count connections to

# Returns
- `Int`: Number of connections to variables in the target set
"""
function count_connections_to_set(adjacency::SparseMatrixCSC{Bool}, var::Int, target_set::Set{Int})
    count = 0

    @inbounds for other_var in target_set
        if adjacency[var, other_var]
            count += 1
        end
    end

    return count
end


"""
    count_internal_connections(adjacency::SparseMatrixCSC{Bool}, var::Int, block_vars::Vector{Int})

Counts connections from a variable to other variables within its block.

# Arguments
- `adjacency::SparseMatrixCSC{Bool}`: Variable adjacency matrix
- `var::Int`: Variable to count connections for
- `block_vars::Vector{Int}`: Variables in the block

# Returns
- `Int`: Number of connections to other variables within the block
"""
function count_internal_connections(adjacency::SparseMatrixCSC{Bool}, var::Int, block_vars::Vector{Int})
    count = 0

    @inbounds for other_var in block_vars
        if other_var != var && adjacency[var, other_var]
            count += 1
        end
    end

    return count
end
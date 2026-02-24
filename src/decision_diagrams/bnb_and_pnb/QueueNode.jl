"""
COMMENTS UP TO DATE
"""

"""
    QueueNode{T<:Integer, U<:Real}

Represents a node in the branch-and-bound queue containing partial integer assignments and associated bound information.

# Fields
- `ltr::U`: Length-to-root bound value from decision diagram construction
- `implied_lb::T`: Lower bound for the next integer variable to be assigned
- `implied_ub::T`: Upper bound for the next integer variable to be assigned
- `path::Union{Vector{T}, BitVector}`: Partial integer variable assignments (BitVector when T=Bool, Vector{T} otherwise)
- `implied_bound::U`: Total bound estimate used for queue priority ordering
- `cont_bound_contr::U`: Continuous variable contribution to the bound estimate
"""
struct QueueNode{T<:Integer, U<:Real}
    ltr             ::U
    implied_lb      ::T
    implied_ub      ::T
    path            ::Union{Vector{T}, BitVector}
    implied_bound   ::U
    cont_bound_contr::U
end

function pretty_print_queue_node(io::IO, node::QueueNode{T,U}) where {T<:Integer, U<:Real}
    print(io, "\n","QueueNode(ltr=", node.ltr,
          ", lb=", node.implied_lb,
          ", ub=", node.implied_ub,
          ", path=", node.path,
          ", implied_bound=", node.implied_bound,
          ", continuous contribution to bound=", node.cont_bound_contr,
    ")")
end

Base.show(io::IO, node::QueueNode{T,U}) where {T<:Integer, U<:Real} = pretty_print_queue_node(io, node)

"""
    heap_insert!(heap::Vector{QueueNode{Z,T}}, node::QueueNode{Z,T}) where {Z<:Integer, T<:Real}

Inserts a single QueueNode into a min-heap, maintaining the heap property based on implied_bound values.

# Arguments
- `heap::Vector{QueueNode{Z,T}}`: Min-heap of queue nodes ordered by implied_bound
- `node::QueueNode{Z,T}`: Node to insert into the heap

# Returns
- Nothing, modifies the heap in-place by adding the new node in the correct position
"""
@inline function heap_insert!(heap::Vector{QueueNode{Z,T}}, node::QueueNode{Z,T}) where {Z<:Integer, T<:Real}
    # Append the new node (so the array is long enough)
    push!(heap, node)
    
    # We'll "lift" the new node up without swapping it repeatedly.
    # Instead, we compare with the parent, move the parent down as needed,
    # and only place our new node once at the final position.
    i = length(heap)

    @inbounds while i > 1
        parent = i >> 1  # or i >>> 1
        parentNode = heap[parent]
        if node.implied_bound < parentNode.implied_bound
            # Move the parent down
            heap[i] = parentNode
            i = parent
        else
            # The correct spot for node is found
            break
        end
    end

    # Insert the lifted node exactly where it belongs
    heap[i] = node
end

"""
    heap_pop!(heap::Vector{QueueNode{Z,T}}) where {Z<:Integer, T<:Real}

Removes and returns the minimum QueueNode from a min-heap, maintaining the heap property.

# Arguments
- `heap::Vector{QueueNode{Z,T}}`: Min-heap of queue nodes ordered by implied_bound

# Returns
- `QueueNode{Z,T}`: The node with the smallest implied_bound value that was removed from the heap
"""
@inline function heap_pop!(heap::Vector{QueueNode{Z,T}}) where {Z<:Integer, T<:Real}
    # The root (index 1) is the minimum element
    min_node = heap[1]

    # Remove the last node from the array
    last_node = pop!(heap)  # shrinks the heap by one

    # If the heap is now empty after popping, just return the min
    if isempty(heap)
        return min_node
    end

    # 1. Place the last node at the root.
    # 2. "Sink" it down by repeatedly moving the smaller child up until the correct position is found.
    @inbounds begin
        # We'll place last_node only when we find its correct spot.
        # For now, let's just hold onto it and move smaller children up.
        i = 1
        len = length(heap)

        while true
            left  = i << 1   # left child index = 2i
            if left > len
                # No children, so we've reached the correct position
                break
            end
            right = left + 1 # right child index = 2i + 1

            # Determine which child has a smaller implied_bound
            c = if right <= len && heap[right].implied_bound < heap[left].implied_bound
                    right
                else
                    left
                end

            # If the chosen child is smaller than last_node, move that child up
            if heap[c].implied_bound < last_node.implied_bound
                heap[i] = heap[c]
                i = c
            else
                # last_node belongs here
                break
            end
        end

        # Put last_node in its final position
        heap[i] = last_node
    end

    return min_node
end



"""
    sink_nodes!(heap::Vector{QueueNode{Z,T}}, i::Int, n::Int) where {Z<:Integer, T<:Real}

Restores heap property by sinking a node down from position i until the heap constraint is satisfied.

# Arguments
- `heap::Vector{QueueNode{Z,T}}`: Min-heap array being modified
- `i::Int`: Starting position index to sink from
- `n::Int`: Effective size of the heap (elements beyond this index are ignored)

# Returns
- Nothing, modifies the heap in-place by moving the node at position i to its correct heap position

# Algorithm
Repeatedly compares the node with its children and moves the smaller child up until the node finds
its proper position in the heap hierarchy based on implied_bound values.
"""
@inline function sink_nodes!(heap::Vector{QueueNode{Z,T}}, i::Int, n::Int) where {Z<:Integer, T<:Real}
    node = heap[i]
    nodeBound = node.implied_bound

    @inbounds while true
        child = i << 1  # left child
        if child > n
            # No children
            break
        end

        # Check if there's a right child that is smaller.
        local leftBound, rightBound
        leftBound = heap[child].implied_bound
        if child + 1 <= n
            rightBound = heap[child + 1].implied_bound
            if rightBound < leftBound
                child += 1
                leftBound = rightBound  # now leftBound is actually the smaller child's bound
            end
        end

        # If the smaller child is still less than nodeBound, move it up.
        if leftBound < nodeBound
            heap[i] = heap[child]
            i = child
        else
            break
        end
    end

    # Finally place the sunk node
    heap[i] = node
end


"""
    add_unsorted!(heap::Vector{QueueNode{Z,T}}, nodes::Vector{QueueNode{Z,T}}) where {Z<:Integer, T<:Real}

Efficiently adds multiple unsorted QueueNodes to an existing heap by bulk appending and re-heapifying.

# Arguments
- `heap::Vector{QueueNode{Z,T}}`: Existing min-heap of queue nodes
- `nodes::Vector{QueueNode{Z,T}}`: Vector of nodes to add (does not need to be sorted)

# Returns
- `Vector{QueueNode{Z,T}}`: The modified heap containing all original and new nodes in heap order
"""
@inline function add_unsorted!(heap::Vector{QueueNode{Z,T}}, nodes::Vector{QueueNode{Z,T}}) where {Z<:Integer, T<:Real}
    # Inlining the function is a hint for the compiler to embed code at call sites.
    # Appending new nodes: the heap and new items are just combined in one vector.
    @inbounds append!(heap, nodes)

    # After appending, we restore the heap property from the last parent down to the root.
    total = length(heap)
    
    @inbounds for i in div(total, 2):-1:1
        sink_nodes!(heap, i, total)
    end

    return heap
end



"""
    heap_insert!(heap::Vector{QueueNode{Z,T}}, new_nodes::Vector{QueueNode{Z,T}}) where {Z<:Integer, T<:Real}

Intelligently inserts multiple QueueNodes into a heap using either bulk insertion or individual insertion based on size heuristics.

# Arguments
- `heap::Vector{QueueNode{Z,T}}`: Existing min-heap of queue nodes
- `new_nodes::Vector{QueueNode{Z,T}}`: Vector of nodes to insert

# Returns
- `Vector{QueueNode{Z,T}}`: The modified heap containing all original and new nodes in heap order

# Algorithm
Uses a threshold-based approach: if the number of new nodes is at least 10% of the existing heap size,
bulk insertion via `add_unsorted!` is used for efficiency. Otherwise, nodes are inserted individually
to maintain heap structure throughout the process.
"""
function heap_insert!(
    heap::Vector{QueueNode{Z,T}},
    new_nodes::Vector{QueueNode{Z,T}}
) where {Z<:Integer, T<:Real}

    old_size = length(heap)
    threshold = div(old_size, 10)
    new_size = length(new_nodes)

    # Decision rule example:
    # If the new list is at least as large as 1/10 of the existing heap,
    # we bulk-append and call add_unsorted!.
    # Otherwise, insert nodes one by one using push!.
    if new_size >= threshold
        # Using add_unsorted! for bulk insertion (threshold-based)
        add_unsorted!(heap, new_nodes)
    else
        #allocate memory for the new nodes
        sizehint!(heap, old_size+new_size)
        # Inserting each node individually
        @inbounds for node in new_nodes
            heap_insert!(heap, node)
        end
    end

    return heap
end


"""
    prune_suboptimal_nodes!(heap::Vector{QueueNode{Z,T}}, bkv::T) where {Z<:Integer, T<:Real}

Removes all nodes with `implied_bound >= bkv` by scanning from right to left to find the last unprunable node.

Scans the heap array from the end backwards to find the rightmost node with `implied_bound < bkv`.
All nodes beyond that point are removed, and the remaining heap is re-heapified to restore the min-heap property.

# Arguments
- `heap::Vector{QueueNode{Z,T}}`: Min-heap of branch-and-bound queue nodes
- `bkv::T`: Best known objective value (nodes with implied_bound >= bkv are prunable)

# Returns
- `Int`: Number of nodes pruned from the queue

# Algorithm
1. Scan array from right to left (end to beginning)
2. Find the last index where `implied_bound < bkv` (last unprunable node)
3. Truncate array to remove all nodes after that index
4. Re-heapify the remaining elements to restore min-heap property

# Performance
More efficient than removing nodes one-by-one when many nodes at the end of the array are prunable.
Re-heapification is O(n) where n is the remaining heap size.
"""
function prune_suboptimal_nodes!(heap::Vector{QueueNode{Z,T}}, bkv::T) where {Z<:Integer, T<:Real}
    if isempty(heap)
        return 0
    end

    n = length(heap)
    last_good_idx = 0

    # Scan from right to left to find the last unprunable node
    @inbounds for i in n:-1:1
        if heap[i].implied_bound < bkv
            last_good_idx = i
            break
        end
    end

    # If no unprunable nodes found, empty the entire heap
    if last_good_idx == 0
        pruned_count = n
        empty!(heap)
        return pruned_count
    end

    # Calculate how many nodes we're removing
    pruned_count = n - last_good_idx

    # Truncate the heap (no re-heapification needed since we're only removing from the end)
    if pruned_count > 0
        resize!(heap, last_good_idx)
    end

    return pruned_count
end




















# function test_heap!()
#     # Create an empty heap.
#     heap = Vector{QueueNode{Float64}}()
    
#     # The values for implied_bound to insert in order.
#     values = [10, 7, 11, 5, 4, 13]
    
#     for v in values
#         # Create a dummy QueueNode.
#         # Only the implied_bound is significant for the heap property,
#         # the other fields use dummy values.
#         node = QueueNode(
#             0.0,         # ltr (dummy value)
#             0,           # implied_lb (dummy value)
#             0,           # implied_ub (dummy value)
#             Int[],       # path (empty vector)
#             float(v),     # implied_bound, using the test value
#             0.0         #cont contribution (dummy value)
#         )
        
#         # Push the node into the heap using our custom push! function.
#         heap_insert!(heap, node)
        
#         # After each insertion, print the heap's implied_bound values.
#         println("Heap after inserting $(v): ", [n.implied_bound for n in heap])
#     end


#     # At this point, your heap should look like:
#     # [4.0, 5.0, 11.0, 10.0, 7.0, 13.0]
#     println("\nHeap built: ", [n.implied_bound for n in heap])
    
#     println("\n=== Popping nodes ===")
#     # Expected pop sequence (min elements in order):
#     # 1st pop: 4.0  -> heap becomes [5.0, 7.0, 11.0, 10.0, 13.0]
#     # 2nd pop: 5.0  -> heap becomes [7.0, 10.0, 11.0, 13.0]
#     # 3rd pop: 7.0  -> heap becomes [10.0, 13.0, 11.0]
#     # 4th pop: 10.0 -> heap becomes [11.0, 13.0]
#     # 5th pop: 11.0 -> heap becomes [13.0]
#     # 6th pop: 13.0 -> heap becomes []
    
#     iteration = 1
#     while !isempty(heap)
#         min_node = heap_pop!(heap)
#         println("\nAfter pop iteration $(iteration), popped: ", min_node.implied_bound)
#         println("Heap now: ", [n.implied_bound for n in heap])
#         iteration += 1
#     end
# end


# function test_add_unsorted!()
#     # 1. Create an empty heap
#     heap = Vector{QueueNode{Float64}}()

#     # 2. Initial values to insert
#     initial_values = [10, 7, 11, 5, 4, 13]
#     println("=== Inserting initial values ===")
#     for v in initial_values
#         # Create a dummy node focusing on implied_bound
#         node = QueueNode(
#             0.0,
#             0,
#             0,
#             Int[],
#             float(v),
#             0.0
#         )
        
#         # Insert and print
#         push!(heap, node)
#         println("Heap after inserting $v: ", [n.implied_bound for n in heap])
#     end

#     println("\nFinal heap after initial insertions: ", [n.implied_bound for n in heap])
    
#     # 3. Now add an unsorted list to the existing heap
#     new_values = [9, 2, 25, 1]  # Example unsorted data
#     new_nodes = Vector{QueueNode{Float64}}()
#     for v in new_values
#         push!(new_nodes, QueueNode(0.0, 0, 0, Int[], float(v),0.0))
#     end

#     println("\nAdding new unsorted values: $new_values")
#     add_unsorted!(heap, new_nodes)
    
#     # 4. Print the final heap
#     println("Final heap after add_unsorted!: ", [n.implied_bound for n in heap])



#      # 1. Create an empty heap
#      heap = Vector{QueueNode{Float64}}()

#      # 2. Initial values to insert
#      initial_values = [10, 7, 11, 5, 4, 13]
#      for v in initial_values
#          # Create a dummy node focusing on implied_bound
#          node = QueueNode(
#              0.0,
#              0,
#              0,
#              Int[],
#              float(v),
#              0.0
#          )
         
#          # Insert and print
#          heap_insert!(heap, node)
#      end
#      new_values = [9, 2, 25, 1]
#      for v in new_values
#         # Create a dummy node focusing on implied_bound
#         node = QueueNode(
#             0.0,
#             0,
#             0,
#             Int[],
#             float(v),
#             0.0
#         )
        
#         # Insert and print
#         heap_insert!(heap, node)
#     end
#      println("Correct final heap: ", [n.implied_bound for n in heap])
# end
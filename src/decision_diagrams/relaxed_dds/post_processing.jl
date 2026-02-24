"""
COMMENTS UP TO DATE
"""

"""
    post_process_relaxed_dd!(
        queue_nodes::Vector{QueueNode{Z,V}}, bkv::V, bks_int::Union{Vector{Z}, BitVector},
        bks_cont::Vector{V}, fc_new_nodes::Vector{QueueNode{Z,V}},
        rel_path::Union{Vector{Z}, BitVector},
        feasibility_accumulator::Vector{V}, node_matrix::Vector{NodeLayer{Z}},
        num_int_vars::Int, num_cont_vars::Int, num_constraints::Int,
        ltr_matrix::Matrix{V}, ltt_matrix::Matrix{V},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        int_obj_coeffs::Vector{V}, first_layer::Int,
        preceeding_path::Union{Vector{Z}, BitVector}, coefficient_matrix_int_cols::Matrix{V},
        coefficient_matrix_rhs_vector::Vector{V}, wrk_vec::Vector{V},
        infimum_gap_matrices::Matrix{V}, buffer_offset::Int, last_exact_layer_idx::Int,
        cont_inf_gap_ctrbtns::Vector{V}, rough_bounds_cont_val::V,
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector},
        lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef},
        lp_constraint_refs::Vector{JuMP.ConstraintRef},
        timing_stats::TimingStats; debug_mode::Bool = false, time_remaining::Union{Float64, Nothing} = nothing
    ) where {Z<:Integer, V<:Real}

Processes a relaxed decision diagram to extract bounds, find optimal solutions, and generate branching nodes

# Arguments
- `queue_nodes::Vector{QueueNode{Z,V}}`: Branch-and-bound queue for new subproblems
- `bkv::V`: Best known value for bound comparisons
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update
- `bks_cont::Vector{V}`: Best known continuous solution vector to update
- `fc_new_nodes::Vector{QueueNode{Z,V}}`: Workspace for frontier cutset nodes
- `rel_path::Union{Vector{Z}, BitVector}`: Workspace for integer path extraction
- `feasibility_accumulator::Vector{V}`: Workspace for feasibility checking
- `node_matrix::Vector{NodeLayer{Z}}`: Relaxed decision diagram layers
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `ltr_matrix::Matrix{V}`: Length-to-root matrix with costs from root
- `ltt_matrix::Matrix{V}`: Length-to-terminal matrix with cost-to-go estimates
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for feasible transitions
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for feasible transitions
- `int_obj_coeffs::Vector{V}`: Objective coefficients for integer variables
- `first_layer::Int`: Starting layer index for processing
- `preceeding_path::Union{Vector{Z}, BitVector}`: Path values preceding the first layer
- `coefficient_matrix_int_cols::Matrix{V}`: Coefficient matrix for integer variables
- `coefficient_matrix_rhs_vector::Vector{V}`: Right-hand side constraint values
- `wrk_vec::Vector{V}`: Preallocated working vector of length num_constraints for gap computations
- `infimum_gap_matrices::Matrix{V}`: Precomputed constraint state matrices
- `buffer_offset::Int`: Column offset for terminal nodes in gap matrices
- `last_exact_layer_idx::Int`: Index of the last exact layer (last layer where all nodes have `first_arc == last_arc`). Used for efficient frontier cutset identification.
- `cont_inf_gap_ctrbtns::Vector{V}`: Continuous variable contributions to gaps
- `rough_bounds_cont_val::V`: Rough bound estimate for continuous variables
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to positive constraint rows
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to negative constraint rows
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `lp_sub_model::JuMP.Model`: LP model for continuous variable optimization
- `lp_vars::Vector{JuMP.VariableRef}`: LP variable references
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: LP constraint references
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `debug_mode::Bool`: Enable debug output (default: false)
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{V, V, Bool, Bool}`: A tuple containing:
  1. `V`: Lower bound from relaxed decision diagram processing
  2. `V`: Updated best known value after processing
  3. `Bool`: False if branch was pruned, true otherwise
  4. `Bool`: True if the best known solution was updated, false otherwise
"""
@inline function post_process_relaxed_dd!(
    queue_nodes                     ::Vector{QueueNode{Z,V}},
    bkv                             ::V,
    bks_int                         ::Union{Vector{Z}, BitVector},
    bks_cont                        ::Vector{V},

    #preallocated work space
    fc_new_nodes                    ::Vector{QueueNode{Z,V}},
    rel_path                        ::Union{Vector{Z}, BitVector},
    feasibility_accumulator         ::Vector{V},

    node_matrix                     ::Vector{NodeLayer{Z}},

    #problem description
    num_int_vars                    ::Int,
    num_cont_vars                   ::Int,
    num_constraints                 ::Int,

    #state information for integer variables
    ltr_matrix                      ::Matrix{V},
    ltt_matrix                      ::Matrix{V},
    lb_matrix                       ::Union{Matrix{Z}, BitMatrix},
    ub_matrix                       ::Union{Matrix{Z}, BitMatrix},
    int_obj_coeffs                  ::Vector{V},
    first_layer                     ::Int,
    preceeding_path                 ::Union{Vector{Z}, BitVector},
    coefficient_matrix_int_cols     ::Matrix{V},

    #added for continuous variable handling
    coefficient_matrix_rhs_vector   ::Vector{V},
    wrk_vec                         ::Vector{V},
    infimum_gap_matrices            ::Matrix{V},
    buffer_offset                   ::Int,
    last_exact_layer_idx            ::Int,
    cont_inf_gap_ctrbtns            ::Vector{V},
    rough_bounds_cont_val           ::V,
    int_var_to_pos_rows             ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows             ::Dict{Int, Vector{Int}},
    lbs_int                         ::Union{Vector{Z}, BitVector},
    ubs_int                         ::Union{Vector{Z}, BitVector},
    lp_sub_model                    ::JuMP.Model,
    lp_vars                         ::Vector{JuMP.VariableRef},
    lp_constraint_refs              ::Vector{JuMP.ConstraintRef},
    timing_stats::TimingStats;
    debug_mode                      ::Bool = false,
    time_remaining                  ::Union{Float64, Nothing} = nothing
) where {Z<:Integer, V<:Real}
    fn_start = time()

    #if there are no terminal nodes, then the branch has been trimmed
    if last(node_matrix).size == 0
        return typemax(V), bkv, false, false
    end

    #get the lowest ltr of the diagram
    layer = node_matrix[num_int_vars]
    cur_idx = argmin(ltr_matrix[(1:layer.size), num_int_vars])
    lowest_ltr = ltr_matrix[cur_idx, num_int_vars]

    if num_cont_vars > 0
        #collapse the state matrix to a vector
        compute_terminal_infimum_gaps!(wrk_vec, infimum_gap_matrices, node_matrix, num_int_vars, num_constraints, buffer_offset)

        @time_operation timing_stats post_relaxed_dd_potential_top_level_LP begin
            is_feasible, cont_contribution = run_relaxed_LP!(
                bkv,
                wrk_vec,
                lowest_ltr,
                num_constraints,
                cont_inf_gap_ctrbtns,
                rough_bounds_cont_val,
                lp_sub_model,
                lp_constraint_refs,
                timing_stats
            )
        end

        #if there is no LP solution, then the branch has been trimmed
        if !is_feasible
            return typemax(V), bkv, false, false
        end
    else
        cont_contribution = zero(V)
        #get the relaxed path through the diagram
        backtrack_path!(rel_path, node_matrix, ltr_matrix, lb_matrix, ub_matrix, num_int_vars, cur_idx, first_layer, preceeding_path)
        #check if rel_path is feasible, return if yes
        is_feasible = colwise_feasible!(coefficient_matrix_int_cols, rel_path, coefficient_matrix_rhs_vector, feasibility_accumulator, num_int_vars, num_constraints)

        # println("Rel Path: ", rel_path, " Feasible: ", is_feasible)
        if is_feasible && lowest_ltr < bkv
            bks_int .= rel_path
            return lowest_ltr, lowest_ltr, true, true
        end
    end

    # Check time budget before update_dd_ltt!
    if time_budget_exceeded(time_remaining, fn_start)
        return typemin(V), bkv, true, false
    end

    #update the length to terminal matrix
    @time_operation timing_stats post_relaxed_dd_update_ltt begin
        update_dd_ltt!(ltt_matrix, node_matrix, int_obj_coeffs, lb_matrix, ub_matrix, num_int_vars, first_layer)
    end

    # Check time budget after update_dd_ltt!
    if time_budget_exceeded(time_remaining, fn_start)
        return typemin(V), bkv, true, false
    end

    # println("LTT MATRIX")
    # println(ltt_matrix)
    #all printed values must agree
    # path_ltr = compute_objective_value(rel_path, obj_coeffs)
    # println("Path Sanity Bound: ", path_ltr)
    # println("LTR Sanity Bound: ", lowest_ltr)
    # first_layer_nodes = node_matrix[first_layer]
    # c_k = obj_coeffs[first_layer]
    # min_ltt = ltt_matrix[1,first_layer] += c_k * first_layer_nodes.values[1]
    # @inbounds @simd for i in 2:first_layer_nodes.size
    #     test_val = ltt_matrix[i, first_layer] + (c_k * first_layer_nodes.values[i])
    #     println(test_val)
    #     if test_val < min_ltr
    #         min_ltr = test_val
    #     end
    # end
    # lowest_ltt = min_ltt
    # println("LTT Sanity Bound: ", lowest_ltt)

    #find the exact cutset
    @time_operation timing_stats post_relaxed_dd_exact_cutset begin
        bkv, bks_was_updated, timed_out = process_last_exact_layer!(
            node_matrix, last_exact_layer_idx, bkv, bks_int, bks_cont, lb_matrix, ub_matrix, ltr_matrix, ltt_matrix,
            num_int_vars, num_cont_vars, num_constraints,
            fc_new_nodes, first_layer, preceeding_path,
            cont_contribution, infimum_gap_matrices, buffer_offset, cont_inf_gap_ctrbtns,
            lp_sub_model, lp_vars, lp_constraint_refs,
            coefficient_matrix_int_cols, coefficient_matrix_rhs_vector,
            int_var_to_pos_rows, int_var_to_neg_rows,
            lbs_int, ubs_int, wrk_vec, timing_stats;
            time_remaining = calculate_child_time_budget(time_remaining, fn_start)
        )
    end

    # Check if process_last_exact_layer! timed out or if we've exceeded our own time budget
    if timed_out || time_budget_exceeded(time_remaining, fn_start)
        return typemin(V), bkv, true, false
    end
    if debug_mode
        println("New to Queue")
        println(fc_new_nodes)
    end

    heap_insert!(queue_nodes, fc_new_nodes)

    local_bound = cont_contribution + lowest_ltr
    return local_bound, bkv, true, bks_was_updated
end


"""
    backtrack_path!(
        rel_path::Union{Vector{Z}, BitVector}, node_matrix::Vector{NodeLayer{Z}},
        ltr_matrix::Matrix{U}, lb_matrix::Union{Matrix{Z}, BitMatrix},
        ub_matrix::Union{Matrix{Z}, BitMatrix}, num_layers::Int, cur_idx::Int,
        first_layer::Int, preceeding_path::Union{Vector{Z}, BitVector}
    ) where {Z<:Integer, U<:Real}

Constructs the shortest relaxed path by backtracking through the decision diagram from a terminal node.

# Arguments
- `rel_path::Union{Vector{Z}, BitVector}`: Preallocated path vector to store the extracted solution
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers containing node information
- `ltr_matrix::Matrix{U}`: Length-to-root matrix with optimal costs for each terminal node
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Domain lower bound matrix
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Domain upper bound matrix
- `num_layers::Int`: Total number of layers in the decision diagram
- `cur_idx::Int`: Starting node index in the terminal layer
- `first_layer::Int`: Index of the first layer to process
- `preceeding_path::Union{Vector{Z}, BitVector}`: Path values preceding the first layer

# Returns
Nothing, the `rel_path` vector is updated in-place with the optimal path values.
"""
function backtrack_path!(
    rel_path        ::Union{Vector{Z}, BitVector},
    node_matrix     ::Vector{NodeLayer{Z}},
    ltr_matrix      ::Matrix{U},
    lb_matrix       ::Union{Matrix{Z}, BitMatrix},
    ub_matrix       ::Union{Matrix{Z}, BitMatrix},
    num_layers      ::Int,
    cur_idx         ::Int,
    first_layer     ::Int,
    preceeding_path ::Union{Vector{Z}, BitVector}
) where {Z<:Integer, U<:Real}
    # clear any existing contents
    rel_path .= 0

    # walk backwards from layer num_vars up to layer 2
    @inbounds begin 
        for layer_idx in num_layers:-1:(first_layer+1)
            layer = node_matrix[layer_idx]
            # record the current node's value
            cur_val = layer.values[cur_idx]
            rel_path[layer_idx] = cur_val  # store value

            first_arc = layer.first_arcs[cur_idx]
            last_arc  = layer.last_arcs[cur_idx]

            # find the feasible parent with minimum LTR
            prev_layer_idx = layer_idx - 1
            lb_col = @view lb_matrix[:, prev_layer_idx]
            ub_col = @view ub_matrix[:, prev_layer_idx]
            ltr_col = @view ltr_matrix[:, prev_layer_idx]

            # initialize with the first parent's LTR
            best_ltr = ltr_col[first_arc]
            @inbounds for par_idx in first_arc:last_arc
                if lb_col[par_idx] <= cur_val <= ub_col[par_idx]
                    par_ltr = ltr_col[par_idx]
                    if par_ltr <= best_ltr
                        best_ltr = par_ltr
                        cur_idx  = par_idx
                    end
                end
            end
        end

        # finally, add the root‐layer value
        rel_path[first_layer] = node_matrix[first_layer].values[cur_idx]
        #add any preceeding path to the root
        @simd for pre_path_idx in 1:first_layer-1
            rel_path[pre_path_idx] = preceeding_path[pre_path_idx]
        end
    end
end


"""
    backtrack_exact_path!(
        path::Union{Vector{Z}, BitVector}, depth::Int, node_idx::Int,
        dd::Vector{NodeLayer{Z}}, first_layer::Int,
        preceeding_path::Union{Vector{Z}, BitVector}
    ) where {Z<:Integer}

Constructs an exact path by backtracking through a relaxed decision diagram from a terminal node.

# Arguments
- `path::Union{Vector{Z}, BitVector}`: Preallocated path vector to store the extracted solution
- `depth::Int`: Starting layer depth for backtracking
- `node_idx::Int`: Starting node index in the terminal layer
- `dd::Vector{NodeLayer{Z}}`: Relaxed decision diagram layers
- `first_layer::Int`: Index of the first layer to process
- `preceeding_path::Union{Vector{Z}, BitVector}`: Path values preceding the first layer

# Returns
Nothing, the `path` vector is updated in-place with the exact path values.
"""
@inline function backtrack_exact_path!(
    path            ::Union{Vector{Z}, BitVector},
    depth           ::Int,
    node_idx        ::Int,
    dd              ::Vector{NodeLayer{Z}},
    first_layer     ::Int,
    preceeding_path ::Union{Vector{Z}, BitVector},
) where {Z<:Integer}
    @inbounds for k = depth:-1:first_layer
        layer = dd[k]
        path[k] = layer.values[node_idx]        # record value
        node_idx = layer.first_arcs[node_idx]       # exactly one parent by assumption
    end

    #add any preceeding path to the root
    @inbounds @simd for pre_path_idx in 1:first_layer-1
        path[pre_path_idx] = preceeding_path[pre_path_idx]
    end
end


"""
    update_dd_ltt!(
        ltt_matrix::Matrix{U}, node_matrix::Vector{NodeLayer{Z}}, obj_coeffs::Vector{U},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        num_layers::Int, first_layer::Int
    ) where {Z<:Integer, U<:Real}

Computes length-to-terminal values for nodes in the relaxed dd by propagating costs upward through the decision diagram.

# Arguments
- `ltt_matrix::Matrix{U}`: Length-to-terminal matrix to be updated in-place
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers containing node information
- `obj_coeffs::Vector{U}`: Objective coefficients for integer variables
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for feasible transitions
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for feasible transitions
- `num_layers::Int`: Total number of layers in the decision diagram
- `first_layer::Int`: Index of the first layer to process

# Returns
Nothing, the `ltt_matrix` is updated in-place with length-to-terminal values.
"""
@inline function update_dd_ltt!(
    ltt_matrix  ::Matrix{U},
    node_matrix ::Vector{NodeLayer{Z}},
    obj_coeffs  ::Vector{U},
    lb_matrix   ::Union{Matrix{Z}, BitMatrix},
    ub_matrix   ::Union{Matrix{Z}, BitMatrix},
    num_layers  ::Int,
    first_layer ::Int,
) where {Z<:Integer, U<:Real}

    @inbounds begin
        # start with bottom layer (length‑to‑terminal = 0)
        child_layer = node_matrix[num_layers]
        ltt_children = @view ltt_matrix[(1:child_layer.size),num_layers]

        fill!(ltt_children, zero(U))     # ensure bottom layer is 0

        # propagate upward: k = (n‑1) … 1
        @inbounds for k in (num_layers-1):-1:first_layer
            parent_layer = node_matrix[k]
            ltt_parents = @view ltt_matrix[(1:parent_layer.size),k]

            update_layer_ltt!(ltt_parents, ltt_children, parent_layer, child_layer, k, obj_coeffs, lb_matrix, ub_matrix)

            # next iteration: these become the “children”
            child_layer = parent_layer
            ltt_children = ltt_parents
        end 

        # ----- add c1*x1 to every node in the first layer -----
        # c_k = obj_coeffs[first_layer]
        # @inbounds @simd for i in 1:child_layer.size
        #     ltt_children[i] += c_k * child_layer.values[i]
        # end
    end
end


"""
    update_layer_ltt!(
        ltt_parents::SubArray{U}, ltt_children::SubArray{U}, parent_layer::NodeLayer{Z},
        child_layer::NodeLayer{Z}, k::Int, obj_coeffs::Vector{U},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix}
    ) where {Z<:Integer, U<:Real}

Updates length-to-terminal values for a single layer by finding optimal child transitions.

# Arguments
- `ltt_parents::SubArray{U}`: Length-to-terminal values for parent layer nodes
- `ltt_children::SubArray{U}`: Length-to-terminal values for child layer nodes
- `parent_layer::NodeLayer{Z}`: Parent layer containing node information
- `child_layer::NodeLayer{Z}`: Child layer containing node information
- `k::Int`: Current layer index being processed
- `obj_coeffs::Vector{U}`: Objective coefficients for integer variables
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for feasible transitions
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for feasible transitions

# Returns
Nothing, the `ltt_parents` array is updated in-place with optimal length-to-terminal values.
"""
@inline function update_layer_ltt!(
    ltt_parents  ::SubArray{U},
    ltt_children ::SubArray{U},
    parent_layer ::NodeLayer{Z},
    child_layer  ::NodeLayer{Z},
    k            ::Int,
    obj_coeffs   ::Vector{U},
    lb_matrix    ::Union{Matrix{Z}, BitMatrix},
    ub_matrix    ::Union{Matrix{Z}, BitMatrix},
) where {Z<:Integer, U<:Real}
    @inbounds begin
        lb_col        = @view lb_matrix[:,k]
        ub_col        = @view ub_matrix[:,k]
        active_col    = child_layer.active

        c_next        = obj_coeffs[k + 1]

        # initialise parents to +∞ once, then take minima
        @inbounds @simd for i in 1:parent_layer.size
            ltt_parents[i] = typemax(U)
        end


        # --- main propagation loop -----------------------------------------
        @inbounds for child_idx in 1:child_layer.size
            # Skip inactive nodes (pruned by rough bounds)
            if !active_col[child_idx]
                continue
            end

            val      = child_layer.values[child_idx]
            cost     = (c_next * val) + ltt_children[child_idx]

            first_arc  = child_layer.first_arcs[child_idx]
            last_arc   = child_layer.last_arcs[child_idx]

            @inbounds @simd for par_idx in first_arc:last_arc
                # Branchless min optimization: fewer branches, better SIMD vectorization
                if val >= lb_col[par_idx] && val <= ub_col[par_idx]
                    ltt_parents[par_idx] = min(ltt_parents[par_idx], cost)
                end
            end
        end
    end
end


"""
    compute_terminal_infimum_gaps!(relaxed_infimum_gaps::Vector{V}, infimum_gap_matrices::Matrix{V}, node_matrix::Vector{NodeLayer{Z}}, num_int_vars::Int, num_constraints::Int, buffer_offset::Int) where {Z<:Integer, V<:Real}

Computes a vector of relaxed infimum gaps that is valid for every terminal node in the relaxed dd
by taking the maximum infimum_gap across all terminal nodes for each constraint.

# Arguments
- `relaxed_infimum_gaps::Vector{V}`: Preallocated output vector for relaxed gap values
- `infimum_gap_matrices::Matrix{V}`: Matrix of infimum gaps where columns represent nodes
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram node layers
- `num_int_vars::Int`: Number of integer variables (terminal layer index)
- `num_constraints::Int`: Number of constraints (matrix rows)
- `buffer_offset::Int`: Column offset for terminal layer nodes in gap matrices

# Returns
- `Vector{V}`: The relaxed infimum gaps vector representing best-case constraint states
"""
function compute_terminal_infimum_gaps!(
    relaxed_infimum_gaps    ::Vector{V},
    infimum_gap_matrices    ::Matrix{V},
    node_matrix             ::Vector{NodeLayer{Z}},
    num_int_vars            ::Int,
    num_constraints         ::Int,
    buffer_offset           ::Int,
) where {Z<:Integer, V<:Real}
    first_column = buffer_offset + 1
    relaxed_infimum_gaps .= @view infimum_gap_matrices[:,first_column]

    terminal_layer = node_matrix[num_int_vars]
    terminal_layer_size = terminal_layer.size

    terminal_cols_range = (first_column + 1):(buffer_offset + terminal_layer_size)
    @inbounds for col in terminal_cols_range
        @inbounds @simd for row in 1:num_constraints
            relaxed_infimum_gaps[row] = max(relaxed_infimum_gaps[row], infimum_gap_matrices[row,col])
        end
    end
    
    return relaxed_infimum_gaps
end


"""
    run_relaxed_LP!(bkv::U, gap_vector::AbstractVector{U}, ltr::U, num_constraints::Int, cont_inf_gap_ctrbtns::Vector{U}, rough_bounds_cont_val::U, lp_sub_model::JuMP.Model, lp_constraint_refs::Vector{JuMP.ConstraintRef}, timing_stats::TimingStats) where {U<:Real}

Computes the continuous variable contribution to the relaxed bound by solving an LP subproblem.

# Arguments
- `bkv::U`: Best known value for early termination check
- `gap_vector::AbstractVector{U}`: Infimum gaps representing relaxed constraint states
- `ltr::U`: shortest length-to-root value from terminal nodes
- `num_constraints::Int`: Number of constraints in the problem
- `cont_inf_gap_ctrbtns::Vector{U}`: Continuous variable contributions to infimum gaps
- `rough_bounds_cont_val::U`: Rough bound estimate for continuous variables
- `lp_sub_model::JuMP.Model`: LP model for continuous variables
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to LP constraint objects
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations

# Returns
- `Tuple{Bool, U}`: A tuple containing:
  1. `Bool`: True if LP subproblem was solved optimally, false otherwise
  2. `U`: Objective contribution from continuous variables (zero if infeasible)
"""
function run_relaxed_LP!(
    bkv                          ::U,
    gap_vector                   ::AbstractVector{U},
    ltr                          ::U,
    num_constraints              ::Int,
    cont_inf_gap_ctrbtns         ::Vector{U},
    rough_bounds_cont_val        ::U,
    lp_sub_model                 ::JuMP.Model,
    lp_constraint_refs           ::Vector{JuMP.ConstraintRef},
    timing_stats                 ::TimingStats
) where{U<:Real}

    is_feasible = false
    cont_contribution = zero(U)

    # Skip if this node can't improve bkv
    if ltr + rough_bounds_cont_val >= bkv
        return is_feasible, cont_contribution
    end

    #update the rhs of the constraints
    @inbounds for row_idx in 1:num_constraints
        # println("Real RHS:", coefficient_matrix_rhs_vector[row_idx], " Gap Matrix:", gap_matrix[row_idx, node_idx] , " Continuous Contribution Removal:", cont_inf_gap_ctrbtns[row_idx])
        new_rhs = gap_vector[row_idx] - cont_inf_gap_ctrbtns[row_idx]
        set_normalized_rhs(lp_constraint_refs[row_idx], new_rhs)
    end

    #check the solution against the best known solution
    @time_operation timing_stats post_relaxed_dd_true_LP begin
        optimize!(lp_sub_model)
    end
    timing_stats.simplex_iterations += MOI.get(lp_sub_model, MOI.SimplexIterations())

    if termination_status(lp_sub_model) == MOI.OPTIMAL
        is_feasible = true

        cont_contribution = U(objective_value(lp_sub_model))
    end

    return is_feasible, cont_contribution
end


"""
    handle_exact_terminal_nodes!(
        ltr_matrix::Matrix{U},
        relaxed_dd::Vector{NodeLayer{Z}}, cur_layr_idx::Int, first_layer::Int,
        preceeding_path::Union{Vector{Z}, BitVector}, bkv::U,
        bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{U},
        infimum_gap_matrices::Matrix{U}, buffer_offset::Int,
        cont_inf_gap_ctrbtns::Vector{U}, rough_bounds_cont_val::U,
        lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef},
        lp_constraint_refs::Vector{JuMP.ConstraintRef},
        num_cont_vars::Int, num_constraints::Int,
        timing_stats::TimingStats;
        time_remaining::Union{Float64, Nothing} = nothing
    ) where {Z<:Integer, U<:Real}

Processes all exact terminal nodes in a layer by solving continuous subproblems.
For integer programs, finds the node with minimum LTR. For mixed-integer programs,
solves an LP for each terminal node to determine the best complete solution.

# Arguments
- `ltr_matrix::Matrix{U}`: Length-to-root matrix containing costs from root to nodes
- `relaxed_dd::Vector{NodeLayer{Z}}`: Relaxed decision diagram layers
- `cur_layr_idx::Int`: Terminal layer index to process
- `first_layer::Int`: Index of the first layer to process
- `preceeding_path::Union{Vector{Z}, BitVector}`: Path values preceding the first layer
- `bkv::U`: Best known value to compare against
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update
- `bks_cont::Vector{U}`: Best known continuous solution vector to update
- `infimum_gap_matrices::Matrix{U}`: Precomputed infimum gap matrix with terminal node states
- `buffer_offset::Int`: Column offset for terminal layer nodes in gap matrices
- `cont_inf_gap_ctrbtns::Vector{U}`: Continuous variable contributions to infimum gaps
- `rough_bounds_cont_val::U`: Rough bound estimate for continuous variables
- `lp_sub_model::JuMP.Model`: LP model for continuous variables
- `lp_vars::Vector{JuMP.VariableRef}`: References to LP variable objects
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to LP constraint objects
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds (nothing if no time limit)

# Returns
- `Tuple{U, Bool, Bool}`: A tuple containing:
  1. `U`: The updated best known value after processing terminal nodes
  2. `Bool`: True if the solution was updated, false otherwise
  3. `Bool`: True if processing timed out before completion, false otherwise
"""
function handle_exact_terminal_nodes!(
    ltr_matrix                  ::Matrix{U},
    relaxed_dd                  ::Vector{NodeLayer{Z}},
    cur_layr_idx                ::Int,
    first_layer                 ::Int,
    preceeding_path             ::Union{Vector{Z}, BitVector},
    bkv                         ::U,
    bks_int                     ::Union{Vector{Z}, BitVector},
    bks_cont                    ::Vector{U},
    infimum_gap_matrices        ::Matrix{U},
    buffer_offset               ::Int,
    cont_inf_gap_ctrbtns        ::Vector{U},
    rough_bounds_cont_val       ::U,
    lp_sub_model                ::JuMP.Model,
    lp_vars                     ::Vector{JuMP.VariableRef},
    lp_constraint_refs          ::Vector{JuMP.ConstraintRef},
    num_cont_vars               ::Int,
    num_constraints             ::Int,
    timing_stats::TimingStats;
    time_remaining::Union{Float64, Nothing} = nothing
) where {Z<:Integer, U<:Real}
    @time_operation timing_stats post_relaxed_dd_handle_exact_terminals begin
        fn_start = time()
        updated_solution = false
        timed_out = false

        current_layer = relaxed_dd[cur_layr_idx]
        layer_size = current_layer.size
        ltr_vals = @view ltr_matrix[:, cur_layr_idx]

        if num_cont_vars == 0
            # Integer program: find node with minimum LTR
            best_ltr, best_idx = findmin(@view ltr_vals[1:layer_size])

            if best_ltr < bkv
                backtrack_exact_path!(bks_int, cur_layr_idx, best_idx, relaxed_dd, first_layer, preceeding_path)
                bkv = best_ltr
                updated_solution = true
            end

        else
            active_col = current_layer.active
            # Mixed-integer program: solve LP for each terminal node
            @inbounds for node_idx in 1:layer_size
                # Check time budget
                if time_budget_exceeded(time_remaining, fn_start)
                    timed_out = true
                    break
                end

                # Early skip for nodes pruned by rough bounds
                if !active_col[node_idx]
                    continue
                end

                node_ltr = ltr_vals[node_idx]

                # Only proceed if this terminal node could improve the bound
                if node_ltr + rough_bounds_cont_val < bkv
                    inf_gap_col_idx = buffer_offset + node_idx
                    inf_gap_col = @view infimum_gap_matrices[:, inf_gap_col_idx]

                    # Solve the continuous subproblem for this specific integer assignment
                    @time_operation timing_stats post_relaxed_dd_potential_low_level_LP begin
                        is_feasible, cont_contribution = run_relaxed_LP!(
                            bkv,
                            inf_gap_col,
                            node_ltr,
                            num_constraints,
                            cont_inf_gap_ctrbtns,
                            rough_bounds_cont_val,
                            lp_sub_model,
                            lp_constraint_refs,
                            timing_stats
                        )
                    end

                    if is_feasible
                        total_obj_value = node_ltr + cont_contribution

                        if total_obj_value < bkv
                            # Update best known solution with complete mixed-integer solution
                            bkv = total_obj_value

                            backtrack_exact_path!(bks_int, cur_layr_idx, node_idx, relaxed_dd, first_layer, preceeding_path)

                            @inbounds for var_idx in 1:num_cont_vars
                                bks_cont[var_idx] = value(lp_vars[var_idx])
                            end

                            updated_solution = true
                        end

                    end
                end
            end
        end
    end

    return bkv, updated_solution, timed_out
end


"""
    process_last_exact_layer!(
        relaxed_dd::Vector{NodeLayer{Z}}, last_exact_layer_idx::Int, bkv::U,
        bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{U},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        ltr_matrix::Matrix{U}, ltt_matrix::Matrix{U}, num_int_vars::Int,
        num_cont_vars::Int, num_constraints::Int, new_nodes::Vector{QueueNode{Z,U}},
        first_layer::Int, preceeding_path::Union{Vector{Z}, BitVector}, cont_contribution::U,
        infimum_gap_matrices::Matrix{U}, buffer_offset::Int,
        cont_inf_gap_ctrbtns::Vector{U}, lp_sub_model::JuMP.Model,
        lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef},
        coefficient_matrix_int_cols::Matrix{U}, coefficient_matrix_rhs_vector::Vector{U},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector},
        wrk_vec::Vector{U}, timing_stats::TimingStats;
        time_remaining::Union{Float64, Nothing} = nothing
    ) where {Z<:Integer, U<:Real}

Processes nodes in the last exact layer for branch-and-bound queue generation.
If the last exact layer is the terminal layer, calls `handle_exact_terminal_nodes!` to
extract complete solutions. Otherwise, creates queue nodes for all nodes in the last exact
layer, as each represents a unique exact path from the root.

# Arguments
- `relaxed_dd::Vector{NodeLayer{Z}}`: Relaxed decision diagram layers
- `last_exact_layer_idx::Int`: Index of the last exact layer (last layer where all nodes have single parents)
- `bkv::U`: Best known value for bound comparisons
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update
- `bks_cont::Vector{U}`: Best known continuous solution vector to update
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for feasible transitions
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for feasible transitions
- `ltr_matrix::Matrix{U}`: Length-to-root matrix with costs from root to nodes
- `ltt_matrix::Matrix{U}`: Length-to-terminal matrix with cost-to-go estimates
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `new_nodes::Vector{QueueNode{Z,U}}`: Output vector for new branch-and-bound queue nodes
- `first_layer::Int`: Starting layer index for path reconstruction with preceding_path
- `preceeding_path::Union{Vector{Z}, BitVector}`: Path values preceding the first layer
- `cont_contribution::U`: Global continuous variable contribution bound
- `infimum_gap_matrices::Matrix{U}`: Precomputed constraint state matrices
- `buffer_offset::Int`: Column offset for terminal nodes in gap matrices
- `cont_inf_gap_ctrbtns::Vector{U}`: Continuous variable contributions to gaps
- `lp_sub_model::JuMP.Model`: LP model for terminal node processing
- `lp_vars::Vector{JuMP.VariableRef}`: LP variable references for continuous variables
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: LP constraint references
- `coefficient_matrix_int_cols::Matrix{U}`: Coefficient matrix for integer variables by constraints
- `coefficient_matrix_rhs_vector::Vector{U}`: Right-hand side values for constraints
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `wrk_vec::Vector{U}`: Preallocated working vector for gap computations
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds (nothing if no time limit)

# Returns
- `Tuple{U, Bool, Bool}`: A tuple containing:
  1. `U`: Updated best known value after processing
  2. `Bool`: True if the best known solution was updated, false otherwise
  3. `Bool`: True if processing timed out before completion, false otherwise
"""
function process_last_exact_layer!(
    relaxed_dd              ::Vector{NodeLayer{Z}},
    last_exact_layer_idx    ::Int,
    bkv                     ::U,
    bks_int                 ::Union{Vector{Z}, BitVector},
    bks_cont                ::Vector{U},
    lb_matrix               ::Union{Matrix{Z}, BitMatrix},
    ub_matrix               ::Union{Matrix{Z}, BitMatrix},
    ltr_matrix              ::Matrix{U},
    ltt_matrix              ::Matrix{U},
    num_int_vars            ::Int,
    num_cont_vars           ::Int,
    num_constraints         ::Int,
    new_nodes               ::Vector{QueueNode{Z,U}},
    first_layer             ::Int,
    preceeding_path         ::Union{Vector{Z}, BitVector},
    cont_contribution       ::U,
    infimum_gap_matrices    ::Matrix{U},
    buffer_offset           ::Int,
    cont_inf_gap_ctrbtns    ::Vector{U},
    lp_sub_model            ::JuMP.Model,
    lp_vars                 ::Vector{JuMP.VariableRef},
    lp_constraint_refs      ::Vector{JuMP.ConstraintRef},
    coefficient_matrix_int_cols::Matrix{U},
    coefficient_matrix_rhs_vector::Vector{U},
    int_var_to_pos_rows     ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows     ::Dict{Int, Vector{Int}},
    lbs_int                 ::Union{Vector{Z}, BitVector},
    ubs_int                 ::Union{Vector{Z}, BitVector},
    wrk_vec                 ::Vector{U},
    timing_stats::TimingStats;
    time_remaining::Union{Float64, Nothing} = nothing
) where {Z<:Integer, U<:Real}
    fn_start = time()
    empty!(new_nodes)
    updated_solution = false
    timed_out = false

    # Edge case: last exact layer is the terminal layer (entire DD is exact)
    if last_exact_layer_idx == num_int_vars
        # Process all exact terminal nodes to extract complete solutions
        bkv, updated_solution, timed_out = handle_exact_terminal_nodes!(
            ltr_matrix,
            relaxed_dd,
            last_exact_layer_idx,
            first_layer,
            preceeding_path,
            bkv,
            bks_int,
            bks_cont,
            infimum_gap_matrices,
            buffer_offset,
            cont_inf_gap_ctrbtns,
            cont_contribution,
            lp_sub_model,
            lp_vars,
            lp_constraint_refs,
            num_cont_vars,
            num_constraints,
            timing_stats;
            time_remaining = calculate_child_time_budget(time_remaining, fn_start)
        )

        return bkv, updated_solution, timed_out
    end

    # Normal case: process all nodes in last exact layer as queue nodes
    current_layer = relaxed_dd[last_exact_layer_idx]
    ltr_vals = @view ltr_matrix[:, last_exact_layer_idx]
    ltt_vals = @view ltt_matrix[:, last_exact_layer_idx]
    lb_col = @view lb_matrix[:, last_exact_layer_idx]
    ub_col = @view ub_matrix[:, last_exact_layer_idx]

    active_col = current_layer.active

    @inbounds for p_idx in 1:current_layer.size
        # Check time budget
        if time_budget_exceeded(time_remaining, fn_start)
            timed_out = true
            break
        end

        # Early skip for nodes pruned by rough bounds
        if !active_col[p_idx]
            continue
        end

        # Calculate implied bound for this node
        cur_ltr = ltr_vals[p_idx]
        implied_bound = cur_ltr + ltt_vals[p_idx] + cont_contribution

        if implied_bound < bkv
            # Extract exact path from root to this node
            scratch_path = Vector{Z}(undef, last_exact_layer_idx)
            backtrack_exact_path!(scratch_path, last_exact_layer_idx, p_idx, relaxed_dd, first_layer, preceeding_path)

            # Compute tighter continuous bound using FBBT-based constraint propagation
            @time_operation timing_stats post_relaxed_dd_compute_tighter_bound begin
                cont_contribution_version_two = compute_tighter_continuous_bound!(
                    scratch_path, coefficient_matrix_int_cols, coefficient_matrix_rhs_vector,
                    int_var_to_pos_rows, int_var_to_neg_rows, lbs_int, ubs_int, num_int_vars, num_cont_vars, num_constraints,
                    lp_sub_model, lp_constraint_refs, cont_contribution, wrk_vec, timing_stats
                )
            end

            # Use tighter bound if available
            if cont_contribution_version_two > cont_contribution
                implied_bound -= cont_contribution
                implied_bound += cont_contribution_version_two
                local_cont_contribution = cont_contribution_version_two
            else
                local_cont_contribution = cont_contribution
            end

            # Create queue node if still promising after continuous tightening
            if implied_bound < bkv
                new_queue_node = QueueNode(
                    cur_ltr,
                    lb_col[p_idx], ub_col[p_idx],
                    scratch_path,
                    implied_bound,
                    local_cont_contribution
                )

                push!(new_nodes, new_queue_node)
            end
        end
    end

    return bkv, updated_solution, timed_out
end
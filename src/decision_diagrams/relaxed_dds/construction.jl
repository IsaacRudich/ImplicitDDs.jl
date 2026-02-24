"""
COMMENTS UP TO DATE
"""

"""
    compute_relaxed_dd!(
        int_obj_coeffs::Vector{T}, coefficient_matrix_int_cols::Matrix{T}, inv_coeff::Matrix{T},
        coeff_times_val::Matrix{T},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector},
        original_lbs_int::Union{Vector{Z}, BitVector},
        bkv::T, num_int_vars::Int, rough_bounds_int::Vector{T}, rough_bounds_cont_val::T,
        domain::UnitRange, layer_idx::Int,
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        low_indexes::Vector{<:Integer}, high_indexes::Vector{<:Integer},
        infimum_gap_matrices::Matrix{T}, node_matrix::Vector{NodeLayer{Z}},
        extra_layer::NodeLayer{Z}, ltr_matrix::Matrix{T},
        arc_count_per_node::Vector{<:Integer},
        node_bin_counts::Matrix{<:Integer}, node_cumulative::Matrix{<:Integer},
        global_lower::Vector{<:Integer}, global_upper::Vector{<:Integer},
        bins_matrix::Matrix{<:Integer},
        w::Int, timing_stats::TimingStats, time_remaining::Union{Float64, Nothing} = nothing
    ) where {Z<:Integer, T<:Real}

Constructs a relaxed decision diagram by node separation iteratively top-down.

# Arguments
- `int_obj_coeffs::Vector{T}`: Objective coefficients for integer variables
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `inv_coeff::Matrix{T}`: Precomputed inverse coefficients [num_constraints, num_int_vars] for division→multiplication optimization
- `coeff_times_val::Matrix{T}`: Precomputed coefficient products [max_domain_size, num_int_vars] for arc objectives
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to positive constraint rows
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to negative constraint rows
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (may be FBBT-tightened)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original global lower bounds for integer variables (before FBBT tightening) used for consistent array indexing into coeff_times_val
- `bkv::T`: Best known value for pruning infeasible paths
- `num_int_vars::Int`: Number of integer variables
- `rough_bounds_int::Vector{T}`: Rough bounds for integer variables
- `rough_bounds_cont_val::T`: Rough bound estimate for continuous variables
- `domain::UnitRange`: Domain range for the starting variable
- `layer_idx::Int`: Starting layer index for diagram construction
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Preallocated matrix for lower bound computations
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Preallocated matrix for upper bound computations
- `low_indexes::Vector{<:Integer}`: Preallocated vector for indexing operations
- `high_indexes::Vector{<:Integer}`: Preallocated vector for indexing operations
- `infimum_gap_matrices::Matrix{T}`: Matrix storing constraint state information across nodes
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers to be constructed
- `extra_layer::NodeLayer{Z}`: Temporary layer for node merging operations
- `ltr_matrix::Matrix{T}`: Length-to-root matrix tracking path costs
- `arc_count_per_node::Vector{<:Integer}`: Preallocated workspace for feasible arc counts per node, size [K] (allocated as [max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_bin_counts::Matrix{<:Integer}`: Preallocated workspace for bin count histograms per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_cumulative::Matrix{<:Integer}`: Preallocated workspace for cumulative bin counts per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `global_lower::Vector{<:Integer}`: Preallocated workspace for global lower bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `global_upper::Vector{<:Integer}`: Preallocated workspace for global upper bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `bins_matrix::Matrix{<:Integer}`: Preallocated workspace for individual arc-to-bin mappings, size [w, K] (allocated as [relaxed_w, max_domain_size], stores bin index for each in-arc, type: Int8/16/32/64 based on w)
- `w::Int`: Maximum width (node limit) for each layer
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{NodeLayer{Z}, Int, Int}`: A tuple containing:
  1. `NodeLayer{Z}`: The final extra layer after construction
  2. `Int`: Buffer offset for the terminal layer in infimum gap matrices
  3. `Int`: Index of the last exact layer (last layer where all nodes have `first_arc == last_arc`). Minimum value is `layer_idx` (the starting layer). Used for efficient frontier cutset identification in post-processing.

# Algorithm
Iteratively constructs relaxed DD layers by:
1. Computing implied variable domain bounds for each node
2. Inverting bounds to generate new nodes for each domain value: generate a node for each element in the domain and find where its parents are
3. Splitting the nodes as long as the diagram width is less than w
4. Tracking the last contiguous exact layer for efficient post-processing
5. Updating length-to-root values and rough bounds for pruning
6. Updating constraint state matrices for the next layer
"""
@inline function compute_relaxed_dd!(
    #problem description
    int_obj_coeffs               ::Vector{T},
    coefficient_matrix_int_cols  ::Matrix{T},
    inv_coeff                    ::Matrix{T},
    coeff_times_val              ::Matrix{T},
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    ubs_int                      ::Union{Vector{Z}, BitVector},
    lbs_int                      ::Union{Vector{Z}, BitVector},
    original_lbs_int             ::Union{Vector{Z}, BitVector},
    bkv                          ::T,
    num_int_vars                 ::Int,

    #precomputed values
    rough_bounds_int     ::Vector{T},
    rough_bounds_cont_val::T,
    domain               ::UnitRange,
    layer_idx            ::Int,

    #preallocated memory
    lb_matrix           ::Union{Matrix{Z}, BitMatrix},
    ub_matrix           ::Union{Matrix{Z}, BitMatrix},
    low_indexes         ::Vector{<:Integer},
    high_indexes        ::Vector{<:Integer},
    infimum_gap_matrices::Matrix{T},
    node_matrix         ::Vector{NodeLayer{Z}},
    extra_layer         ::NodeLayer{Z},
    ltr_matrix          ::Matrix{T},
    arc_count_per_node  ::Vector{<:Integer},
    node_bin_counts     ::Matrix{<:Integer},
    node_cumulative     ::Matrix{<:Integer},
    global_lower        ::Vector{<:Integer},
    global_upper        ::Vector{<:Integer},
    bins_matrix         ::Matrix{<:Integer},

    #settings
    w                   ::Int,
    timing_stats::TimingStats,
    time_remaining      ::Union{Float64, Nothing} = nothing
)where{Z<:Integer, T<:Real}
    fn_start = time()

    buffer_offset = 0
    last_exact_layer_idx = layer_idx  # Track last layer where all nodes are exact (start from initial layer)

    #state information for debugging
    # println("Initial Layer")
    # println(node_matrix[layer_idx])
    # println("LTRs ", ltr_matrix[:, layer_idx])
    # println("Infimum Gaps: ")
    #     cs = buffer_offset + 1
    #     cf =  buffer_offset + node_matrix[layer_idx].size
    #     for i in cs:cf
    #         print(infimum_gap_matrices[:, i], " ")
    #     end
    # println("\n")
    for var_index in (layer_idx):num_int_vars-1
        # Check time budget at layer boundary
        if time_budget_exceeded(time_remaining, fn_start)
            return extra_layer, buffer_offset, -1
        end

        #get the new domain
        next_index = var_index+1
        new_domain = lbs_int[next_index]:ubs_int[next_index]
        
        # println("Layer ", var_index, "  Domain: ", new_domain)

        @time_operation timing_stats relaxed_dd_implied_column_bounds begin
            #compute the bounds on x_{i+1} implied by each node on layer i
            compute_implied_column_bounds!(
                lb_matrix, ub_matrix, next_index, infimum_gap_matrices, buffer_offset,
                int_var_to_pos_rows, int_var_to_neg_rows, lbs_int, ubs_int, inv_coeff, node_matrix[var_index]
            )
        end
    
        @time_operation timing_stats relaxed_dd_invert_bounds begin
            #generate a node for each element in the domain and figure out where its parent interval
            invert_implied_column_bounds!(node_matrix, new_domain, lb_matrix, ub_matrix, next_index, low_indexes, high_indexes)
        end

        if node_matrix[next_index].size == 0
            break
        end
    
        
        @time_operation timing_stats relaxed_dd_split_nodes begin
            extra_layer, is_exact = split_nodes!(node_matrix, extra_layer, ltr_matrix, next_index, lb_matrix, ub_matrix, int_obj_coeffs, w, domain, timing_stats, arc_count_per_node, node_bin_counts, node_cumulative, global_lower, global_upper, bins_matrix, coeff_times_val, original_lbs_int)
        end

        # Track last exact layer - only update if this extends a contiguous exact sequence
        if is_exact && last_exact_layer_idx == next_index - 1
            last_exact_layer_idx = next_index
        end

        if node_matrix[next_index].size == 0
            break
        end

        @time_operation timing_stats relaxed_dd_update_ltr begin
            update_layer_ltr!(ltr_matrix, node_matrix, next_index, coeff_times_val, original_lbs_int, lb_matrix, ub_matrix)
        end

        @time_operation timing_stats relaxed_dd_rough_bounding begin
            if next_index < num_int_vars
                rough_bound_relaxed_layer!(node_matrix, next_index, rough_bounds_int, rough_bounds_cont_val, ltr_matrix, bkv)
            end
        end
        
        @time_operation timing_stats relaxed_dd_infimum_matrix_cubic begin
            new_buffer_offset = compute_next_infimum_gap_matrix!(infimum_gap_matrices, buffer_offset, w, node_matrix, next_index, lb_matrix, ub_matrix, coefficient_matrix_int_cols, lbs_int, ubs_int, int_var_to_pos_rows, int_var_to_neg_rows)
        end

        #chnage the domain for the next variable
        domain = new_domain
        buffer_offset = new_buffer_offset
    end

    return extra_layer, buffer_offset, last_exact_layer_idx
end


"""
    setup_initial_relaxed_dd!(
        int_obj_coeffs::Vector{T}, obj_const::T,
        coefficient_matrix_int_cols::Matrix{T},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector},
        infimum_gaps::Vector{T}, infimum_gap_matrices::Matrix{T},
        node_matrix::Vector{NodeLayer{Z}}, ltr_matrix::Matrix{T}
    ) where {Z<:Integer, T<:Real}

Initializes the first layer of a relaxed decision diagram and computes initial constraint state information (infimum gaps)

# Arguments
- `int_obj_coeffs::Vector{T}`: Objective coefficients for integer variables
- `obj_const::T`: Objective function constant term
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to positive constraint rows
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to negative constraint rows
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `infimum_gaps::Vector{T}`: Initial infimum gaps for constraint states
- `infimum_gap_matrices::Matrix{T}`: Matrix to store constraint state information across nodes
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers to be initialized
- `ltr_matrix::Matrix{T}`: Length-to-root matrix for tracking path costs

# Returns
- `UnitRange`: Domain range for the first integer variable

# Algorithm
1. Sets up the domain for the first variable based on its bounds
2. Generates the initial node layer with one node per domain value
3. Computes the initial infimum gap matrix representing constraint states for first layer nodes
"""
@inline function setup_initial_relaxed_dd!(
    int_obj_coeffs               ::Vector{T},
    obj_const                    ::T,
    coefficient_matrix_int_cols  ::Matrix{T},
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    ubs_int                      ::Union{Vector{Z}, BitVector},
    lbs_int                      ::Union{Vector{Z}, BitVector},
    infimum_gaps                 ::Vector{T},
    infimum_gap_matrices         ::Matrix{T},
    node_matrix                  ::Vector{NodeLayer{Z}},
    ltr_matrix                   ::Matrix{T}
) where {Z<:Integer, T<:Real}
    # Setup domain for the first variable
    domain = lbs_int[1]:ubs_int[1]

    # Initialize the first node layer
    generate_initial_node_layer!(int_obj_coeffs, obj_const, domain, node_matrix, ltr_matrix)

    # Compute the infimum gap matrix for the first layer
    compute_initial_infimum_gap_matrix!(
        infimum_gap_matrices,
        infimum_gaps,
        lbs_int,
        ubs_int,
        coefficient_matrix_int_cols,
        int_var_to_pos_rows,
        int_var_to_neg_rows,
        node_matrix
    )

    return domain
end

"""
    setup_relaxed_dd_from_queue_node!(
        node_matrix::Vector{NodeLayer{Z}}, ltr_matrix::Matrix{T},
        infimum_gap_matrices::Matrix{T}, qnode::QueueNode{Z,U},
        infimum_gaps::Vector{T}, lbs_int::Union{Vector{Z}, BitVector},
        ubs_int::Union{Vector{Z}, BitVector}
    ) where {Z<:Integer, T<:Real, U<:Real}

Initializes a relaxed decision diagram from a branch-and-bound queue node with partial integer assignments.

# Arguments
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers to be initialized
- `ltr_matrix::Matrix{T}`: Length-to-root matrix for tracking path costs
- `infimum_gap_matrices::Matrix{T}`: Matrix to store constraint state information across nodes
- `qnode::QueueNode{Z,U}`: Queue node containing partial integer variable assignments
- `infimum_gaps::Vector{T}`: Infimum gaps to initialize the gap matrix (may include FBBT updates)
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables

# Returns
- `Tuple{Int, UnitRange}`: A tuple containing:
  1. `Int`: Layer index where construction should resume
  2. `UnitRange`: Domain range for the next integer variable

# Algorithm
1. Determines the starting layer index from the queue node's partial path length
2. Generates the starting node layer from the queue node's state
3. Copies the provided infimum gaps directly into the gap matrix for DD construction
"""
@inline function setup_relaxed_dd_from_queue_node!(
    node_matrix              ::Vector{NodeLayer{Z}},
    ltr_matrix               ::Matrix{T},
    infimum_gap_matrices     ::Matrix{T},
    qnode                    ::QueueNode{Z,U},
    infimum_gaps             ::Vector{T},
    lbs_int                  ::Union{Vector{Z}, BitVector},
    ubs_int                  ::Union{Vector{Z}, BitVector},
) where {Z<:Integer, T<:Real, U<:Real}
    layer_idx = length(qnode.path)

    generate_node_layer_from_queue!(
        qnode,
        node_matrix,
        ltr_matrix,
        layer_idx
    )

    # 2) copy in the infimum gaps
    @inbounds @simd for i in eachindex(infimum_gaps)
        infimum_gap_matrices[i, 1] = infimum_gaps[i]
    end

    return layer_idx, lbs_int[layer_idx]:ubs_int[layer_idx]
end



"""
    setup_run_process_relaxed_dd!(
        int_obj_coeffs::Vector{T}, obj_const::T,
        coefficient_matrix_int_cols::Matrix{T}, coefficient_matrix_rhs_vector::Vector{T}, inv_coeff::Matrix{T},
        coeff_times_val::Matrix{T},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector},
        original_lbs_int::Union{Vector{Z}, BitVector},
        bkv::T, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T},
        num_int_vars::Int, num_cont_vars::Int, num_constraints::Int,
        rough_bounds_int::Vector{T}, rough_bounds_cont_val::T,
        base_infimum_gaps::Vector{T}, infimum_gap_matrices::Matrix{T},
        node_matrix::Vector{NodeLayer{Z}}, extra_layer::NodeLayer{Z},
        ltr_matrix::Matrix{T}, ltt_matrix::Matrix{T},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        low_indexes::Vector{<:Integer}, high_indexes::Vector{<:Integer},
        rel_path::Union{Vector{Z}, BitVector}, feasibility_accumulator::Vector{T},
        wrk_vec::Vector{T}, cont_inf_gap_ctrbtns::Vector{T},
        arc_count_per_node::Vector{<:Integer},
        node_bin_counts::Matrix{<:Integer}, node_cumulative::Matrix{<:Integer},
        global_lower::Vector{<:Integer}, global_upper::Vector{<:Integer},
        bins_matrix::Matrix{<:Integer},
        queue_nodes::Vector{QueueNode{Z,T}}, fc_new_nodes::Vector{QueueNode{Z,T}},
        lp_sub_model::JuMP.Model,
        lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef},
        w::Int, timing_stats::TimingStats; debug_mode::Bool = false, time_remaining::Union{Float64, Nothing} = nothing
    ) where {Z<:Integer, T<:Real}

Executes the complete relaxed decision diagram solution bounding process for mixed-integer programming problems.

This function provides the main interface for relaxed DD-based bound computation, integrating constraint handling
for both integer and continuous variables. It orchestrates the full solution workflow from problem setup through
bound extraction and (potential) solution generation.

# Returns
- `Tuple{T, T, Bool, Bool, NodeLayer}`: A tuple containing:
  1. `T`: Computed relaxed bound providing a lower bound on the optimal objective value
  2. `T`: Updated best known objective value after processing
  3. `Bool`: Feasibility status indicating whether a feasible solution exists
  4. `Bool`: Solution update flag indicating whether the best known solution was improved
  5. `NodeLayer`: Updated extra layer for proper double-buffering in subsequent calls

# Algorithm
Integrates three core phases to solve mixed-integer programs via relaxed decision diagrams:
1. **Initialization**: Sets up the constraint state information for the root layer
2. **Construction**: Builds the complete relaxed DD
3. **Solution Extraction**: Processes terminal nodes to extract bounds and try to generate improved solutions

# Arguments
## Problem Description
- `int_obj_coeffs::Vector{T}`: Objective coefficients for integer variables
- `obj_const::T`: Objective function constant term
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `coefficient_matrix_rhs_vector::Vector{T}`: Right-hand side constraint values
- `inv_coeff::Matrix{T}`: Precomputed inverse coefficients [num_constraints, num_int_vars] for division→multiplication optimization
- `coeff_times_val::Matrix{T}`: Precomputed coefficient products [max_domain_size, num_int_vars] for arc objectives
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to positive constraint rows
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to negative constraint rows
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (may be FBBT-tightened)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original global lower bounds for integer variables (before FBBT tightening) used for consistent array indexing into coeff_times_val
- `bkv::T`: Best known objective value
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector
- `bks_cont::Vector{T}`: Best known continuous solution vector

## General Information
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `rough_bounds_int::Vector{T}`: Rough bounds for integer variables
- `rough_bounds_cont_val::T`: Rough bound estimate for continuous variables

## Precomputed & Workspace
- `base_infimum_gaps::Vector{T}`: Base infimum gaps used as input for constraint state initialization
- `infimum_gap_matrices::Matrix{T}`: Matrix storing constraint state information across nodes
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers
- `extra_layer::NodeLayer{Z}`: Temporary layer for node operations
- `ltr_matrix::Matrix{T}`: Length-to-root matrix for path costs
- `ltt_matrix::Matrix{T}`: Length-to-terminal matrix for terminal costs
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for computations
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for computations
- `low_indexes::Vector{<:Integer}`: Preallocated vector for indexing operations
- `high_indexes::Vector{<:Integer}`: Preallocated vector for indexing operations
- `rel_path::Union{Vector{Z}, BitVector}`: Preallocated path vector
- `feasibility_accumulator::Vector{T}`: Working vector for feasibility computations
- `wrk_vec::Vector{T}`: Working vector for computations (gets overwritten)
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps
- `arc_count_per_node::Vector{<:Integer}`: Preallocated workspace for feasible arc counts per node, size [K] (allocated as [max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_bin_counts::Matrix{<:Integer}`: Preallocated workspace for bin count histograms per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_cumulative::Matrix{<:Integer}`: Preallocated workspace for cumulative bin counts per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `global_lower::Vector{<:Integer}`: Preallocated workspace for global lower bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `global_upper::Vector{<:Integer}`: Preallocated workspace for global upper bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `bins_matrix::Matrix{<:Integer}`: Preallocated workspace for individual arc-to-bin mappings, size [w, K] (allocated as [relaxed_w, max_domain_size], stores bin index for each in-arc, type: Int8/16/32/64 based on w)

## Frontier-Cut Data
- `queue_nodes::Vector{QueueNode{Z,T}}`: Branch-and-bound queue for generated nodes
- `fc_new_nodes::Vector{QueueNode{Z,T}}`: Temporary storage for new frontier nodes

## Continuous Variable Handling
- `lp_sub_model::JuMP.Model`: LP model for continuous variable optimization
- `lp_vars::Vector{JuMP.VariableRef}`: LP variable references
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: LP constraint references

## Settings
- `w::Int`: Maximum width for decision diagram layers
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `debug_mode::Bool`: Enable debug output (default: false)
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)
"""
function setup_run_process_relaxed_dd!(
    # problem description
    int_obj_coeffs               ::Vector{T},
    obj_const                    ::T,
    coefficient_matrix_int_cols  ::Matrix{T},
    coefficient_matrix_rhs_vector::Vector{T},
    inv_coeff                    ::Matrix{T},
    coeff_times_val              ::Matrix{T},
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    ubs_int                      ::Union{Vector{Z}, BitVector},
    lbs_int                      ::Union{Vector{Z}, BitVector},
    original_lbs_int             ::Union{Vector{Z}, BitVector},
    bkv                          ::T,
    bks_int                      ::Union{Vector{Z}, BitVector},
    bks_cont                     ::Vector{T},

    # general information
    num_int_vars                 ::Int,
    num_cont_vars                ::Int,
    num_constraints              ::Int,
    rough_bounds_int             ::Vector{T},
    rough_bounds_cont_val        ::T,

    # precomputed & workspace
    base_infimum_gaps            ::Vector{T},
    infimum_gap_matrices         ::Matrix{T},
    node_matrix                  ::Vector{NodeLayer{Z}},
    extra_layer                  ::NodeLayer{Z},
    ltr_matrix                   ::Matrix{T},
    ltt_matrix                   ::Matrix{T},
    lb_matrix                    ::Union{Matrix{Z}, BitMatrix},
    ub_matrix                    ::Union{Matrix{Z}, BitMatrix},
    low_indexes                  ::Vector{<:Integer},
    high_indexes                 ::Vector{<:Integer},
    rel_path                     ::Union{Vector{Z}, BitVector},
    feasibility_accumulator      ::Vector{T},
    wrk_vec                      ::Vector{T},
    cont_inf_gap_ctrbtns         ::Vector{T},
    arc_count_per_node           ::Vector{<:Integer},
    node_bin_counts              ::Matrix{<:Integer},
    node_cumulative              ::Matrix{<:Integer},
    global_lower                 ::Vector{<:Integer},
    global_upper                 ::Vector{<:Integer},
    bins_matrix                  ::Matrix{<:Integer},

    # frontier‐cut data
    queue_nodes                  ::Vector{QueueNode{Z,T}},
    fc_new_nodes                 ::Vector{QueueNode{Z,T}},

    #continuous variable handling
    lp_sub_model                 ::JuMP.Model,
    lp_vars                      ::Vector{JuMP.VariableRef},
    lp_constraint_refs           ::Vector{JuMP.ConstraintRef},

    # settings
    w::Int,
    timing_stats::TimingStats;
    debug_mode::Bool = false,
    time_remaining::Union{Float64, Nothing} = nothing
) where {Z<:Integer, T<:Real}
    fn_start = time()
    @time_operation timing_stats relaxed_dd begin
        # 1) initial setup: domain + first‐layer + initial infimum gaps
        domain = setup_initial_relaxed_dd!(
            int_obj_coeffs,
            obj_const,
            coefficient_matrix_int_cols,
            int_var_to_pos_rows,
            int_var_to_neg_rows,
            ubs_int,
            lbs_int,
            base_infimum_gaps,
            infimum_gap_matrices,
            node_matrix,
            ltr_matrix
        )
        @time_operation timing_stats create_relaxed_dd begin
            # 2) build the full relaxed DD
            extra_layer, buffer_offset, last_exact_layer_idx = compute_relaxed_dd!(
                int_obj_coeffs,
                coefficient_matrix_int_cols,
                inv_coeff,
                coeff_times_val,
                int_var_to_pos_rows,
                int_var_to_neg_rows,
                ubs_int,
                lbs_int,
                original_lbs_int,
                bkv,
                num_int_vars,
                rough_bounds_int,
                rough_bounds_cont_val,
                domain,
                1,                    # start layer
                lb_matrix,
                ub_matrix,
                low_indexes,
                high_indexes,
                infimum_gap_matrices,
                node_matrix,
                extra_layer,
                ltr_matrix,
                arc_count_per_node,
                node_bin_counts,
                node_cumulative,
                global_lower,
                global_upper,
                bins_matrix,
                w,
                timing_stats,
                calculate_child_time_budget(time_remaining, fn_start)
            )
        end

        # Check if compute_relaxed_dd! timed out (sentinel value -1)
        if last_exact_layer_idx == -1
            return typemin(T), bkv, false, false, extra_layer
        end

        @time_operation timing_stats post_relaxed_dd begin
            # 3) post‐process to extract the best‐so‐far LTR
            relaxed_bound, bkv, is_feasible, bks_was_updated = post_process_relaxed_dd!(
                queue_nodes,
                bkv,
                bks_int,
                bks_cont,

                fc_new_nodes,
                rel_path,
                feasibility_accumulator,

                node_matrix,

                num_int_vars,
                num_cont_vars,
                num_constraints,

                ltr_matrix,
                ltt_matrix,
                lb_matrix,
                ub_matrix,
                int_obj_coeffs,
                1,
                Vector{Z}(),
                coefficient_matrix_int_cols,

                coefficient_matrix_rhs_vector,
                wrk_vec,
                infimum_gap_matrices,
                buffer_offset,
                last_exact_layer_idx,
                cont_inf_gap_ctrbtns,
                rough_bounds_cont_val,
                int_var_to_pos_rows,
                int_var_to_neg_rows,
                lbs_int,
                ubs_int,
                lp_sub_model,
                lp_vars,
                lp_constraint_refs,
                timing_stats,
                debug_mode = debug_mode,
                time_remaining = calculate_child_time_budget(time_remaining, fn_start)
            )
        end
    end
    return relaxed_bound, bkv, is_feasible, bks_was_updated, extra_layer
end



"""
    setup_run_process_relaxed_dd!(
        int_obj_coeffs::Vector{T}, coefficient_matrix_int_cols::Matrix{T},
        coefficient_matrix_rhs_vector::Vector{T}, inv_coeff::Matrix{T},
        coeff_times_val::Matrix{T},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector},
        original_lbs_int::Union{Vector{Z}, BitVector},
        bkv::T, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T}, qnode::QueueNode,
        num_int_vars::Int, num_cont_vars::Int, num_constraints::Int,
        rough_bounds_int::Vector{T}, rough_bounds_cont_val::T,
        base_infimum_gaps::Vector{T}, infimum_gap_matrices::Matrix{T},
        node_matrix::Vector{NodeLayer{Z}}, extra_layer::NodeLayer{Z},
        ltr_matrix::Matrix{T}, ltt_matrix::Matrix{T},
        lb_matrix::Union{Matrix{Z}, BitMatrix}, ub_matrix::Union{Matrix{Z}, BitMatrix},
        low_indexes::Vector{<:Integer}, high_indexes::Vector{<:Integer},
        rel_path::Union{Vector{Z}, BitVector}, feasibility_accumulator::Vector{T},
        wrk_vec::Vector{T}, cont_inf_gap_ctrbtns::Vector{T},
        arc_count_per_node::Vector{<:Integer},
        node_bin_counts::Matrix{<:Integer}, node_cumulative::Matrix{<:Integer},
        global_lower::Vector{<:Integer}, global_upper::Vector{<:Integer},
        bins_matrix::Matrix{<:Integer},
        queue_nodes::Vector{QueueNode{Z,T}}, fc_new_nodes::Vector{QueueNode{Z,T}},
        lp_sub_model, lp_vars, lp_constraint_refs,
        w, timing_stats::TimingStats; debug_mode::Bool = false, time_remaining::Union{Float64, Nothing} = nothing
    ) where {Z<:Integer, T<:Real}

Executes the complete relaxed decision diagram solution bounding process for mixed-integer programming problems.

This function provides the main interface for relaxed DD-based bound computation, integrating constraint handling
for both integer and continuous variables. It orchestrates the full solution workflow from problem setup through
bound extraction and (potential) solution generation.

This variant initializes from a partial solution path represented by a queue node, constructs the relaxed DD 
from that starting point, and extracts bounds and solution improvements.

# Arguments
## Problem Description
- `int_obj_coeffs::Vector{T}`: Objective coefficients for integer variables
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `coefficient_matrix_rhs_vector::Vector{T}`: Right-hand side constraint values
- `inv_coeff::Matrix{T}`: Precomputed inverse coefficients [num_constraints, num_int_vars] for division→multiplication optimization
- `coeff_times_val::Matrix{T}`: Precomputed coefficient products [max_domain_size, num_int_vars] for arc objectives
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to positive constraint rows
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to negative constraint rows
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (may be FBBT-tightened)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original global lower bounds for integer variables (before FBBT tightening) used for consistent array indexing into coeff_times_val
- `bkv::T`: Best known objective value
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector
- `bks_cont::Vector{T}`: Best known continuous solution vector
- `qnode::QueueNode`: Branch-and-bound queue node containing partial assignments

## Solver Settings
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `rough_bounds_int::Vector{T}`: Rough bounds for integer variables
- `rough_bounds_cont_val::T`: Rough bound estimate for continuous variables

## Precomputed & Workspace
- `base_infimum_gaps::Vector{T}`: Base infimum gaps used as input for constraint state initialization
- `infimum_gap_matrices::Matrix{T}`: Matrix storing constraint state information across nodes
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers
- `extra_layer::NodeLayer{Z}`: Temporary layer for node operations
- `ltr_matrix::Matrix{T}`: Length-to-root matrix for path costs
- `ltt_matrix::Matrix{T}`: Length-to-terminal matrix for terminal costs
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for computations
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for computations
- `low_indexes::Vector{<:Integer}`: Preallocated vector for indexing operations
- `high_indexes::Vector{<:Integer}`: Preallocated vector for indexing operations
- `rel_path::Union{Vector{Z}, BitVector}`: Preallocated path vector
- `feasibility_accumulator::Vector{T}`: Working vector for feasibility computations
- `wrk_vec::Vector{T}`: Working vector for computations (gets overwritten)
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps
- `arc_count_per_node::Vector{<:Integer}`: Preallocated workspace for feasible arc counts per node, size [K] (allocated as [max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_bin_counts::Matrix{<:Integer}`: Preallocated workspace for bin count histograms per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `node_cumulative::Matrix{<:Integer}`: Preallocated workspace for cumulative bin counts per node, size [w, K] (allocated as [relaxed_w, max_domain_size], max value: w, type: Int8/16/32/64 based on w)
- `global_lower::Vector{<:Integer}`: Preallocated workspace for global lower bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `global_upper::Vector{<:Integer}`: Preallocated workspace for global upper bound estimates per threshold, size [w] (allocated as [relaxed_w], max value: K×w, type: Int8/16/32/64 based on K×w)
- `bins_matrix::Matrix{<:Integer}`: Preallocated workspace for individual arc-to-bin mappings, size [w, K] (allocated as [relaxed_w, max_domain_size], stores bin index for each in-arc, type: Int8/16/32/64 based on w)

## Frontier-Cut Data
- `queue_nodes::Vector{QueueNode{Z,T}}`: Branch-and-bound queue for generated nodes
- `fc_new_nodes::Vector{QueueNode{Z,T}}`: Temporary storage for new frontier nodes

## LP Subproblem Handling
- `lp_sub_model`: LP model for continuous variable optimization
- `lp_vars`: LP variable references
- `lp_constraint_refs`: LP constraint references

## Settings
- `w`: Maximum width for decision diagram layers
- `timing_stats::TimingStats`: Timing statistics tracker for relaxed DD operations
- `debug_mode::Bool`: Enable debug output (default: false)
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{T, Bool, NodeLayer}`: A tuple containing:
  1. `T`: Updated best known objective value after processing
  2. `Bool`: Solution update flag indicating whether the best known solution was improved
  3. `NodeLayer`: Updated extra layer for proper double-buffering in subsequent calls

# Algorithm
Processes relaxed DD construction from a queue node through three phases:
1. **Queue Node Setup**: Initializes DD layers from the queue node's partial path and constraint state
2. **DD Construction**: Builds the complete relaxed DD from the starting layer
3. **Solution Processing**: Extracts bounds and attempts solution improvements via frontier-cut analysis
"""
function setup_run_process_relaxed_dd!(
    # problem description
    int_obj_coeffs::Vector{T},
    coefficient_matrix_int_cols::Matrix{T},
    coefficient_matrix_rhs_vector::Vector{T},
    inv_coeff::Matrix{T},
    coeff_times_val::Matrix{T},
    int_var_to_pos_rows::Dict{Int, Vector{Int}},
    int_var_to_neg_rows::Dict{Int, Vector{Int}},
    ubs_int::Union{Vector{Z}, BitVector},
    lbs_int::Union{Vector{Z}, BitVector},
    original_lbs_int::Union{Vector{Z}, BitVector},
    bkv::T,
    bks_int::Union{Vector{Z}, BitVector},
    bks_cont::Vector{T},
    qnode::QueueNode,

    # solver settings
    num_int_vars::Int,
    num_cont_vars::Int,
    num_constraints::Int,
    rough_bounds_int::Vector{T},
    rough_bounds_cont_val::T,

    # precomputed & workspace
    base_infimum_gaps::Vector{T},
    infimum_gap_matrices::Matrix{T},
    node_matrix::Vector{NodeLayer{Z}},
    extra_layer::NodeLayer{Z},
    ltr_matrix::Matrix{T},
    ltt_matrix::Matrix{T},
    lb_matrix::Union{Matrix{Z}, BitMatrix},
    ub_matrix::Union{Matrix{Z}, BitMatrix},
    low_indexes::Vector{<:Integer},
    high_indexes::Vector{<:Integer},
    rel_path::Union{Vector{Z}, BitVector},
    feasibility_accumulator::Vector{T},
    wrk_vec::Vector{T},
    cont_inf_gap_ctrbtns::Vector{T},
    arc_count_per_node::Vector{<:Integer},
    node_bin_counts::Matrix{<:Integer},
    node_cumulative::Matrix{<:Integer},
    global_lower::Vector{<:Integer},
    global_upper::Vector{<:Integer},
    bins_matrix::Matrix{<:Integer},

    # frontier‐cut data
    queue_nodes::Vector{QueueNode{Z,T}},
    fc_new_nodes::Vector{QueueNode{Z,T}},

    # LP subproblem handling
    lp_sub_model,
    lp_vars,
    lp_constraint_refs,

    # settings
    w,
    timing_stats::TimingStats;
    debug_mode::Bool = false,
    time_remaining::Union{Float64, Nothing} = nothing
) where {Z<:Integer, T<:Real}
    fn_start = time()
    @time_operation timing_stats relaxed_dd begin
        # 1) initial setup: domain + first‐layer + initial infimum gaps
        layer_idx, domain = setup_relaxed_dd_from_queue_node!(
            node_matrix, 
            ltr_matrix,
            infimum_gap_matrices,
            qnode,
            base_infimum_gaps,
            lbs_int,
            ubs_int
        )

        @time_operation timing_stats create_relaxed_dd begin
            # 2) build the full relaxed DD
            extra_layer, buffer_offset, last_exact_layer_idx = compute_relaxed_dd!(
                int_obj_coeffs,
                coefficient_matrix_int_cols,
                inv_coeff,
                coeff_times_val,
                int_var_to_pos_rows,
                int_var_to_neg_rows,
                ubs_int,
                lbs_int,
                original_lbs_int,
                bkv,
                num_int_vars,
                rough_bounds_int,
                rough_bounds_cont_val,
                domain,
                layer_idx,# start layer
                lb_matrix,
                ub_matrix,
                low_indexes,
                high_indexes,
                infimum_gap_matrices,
                node_matrix,
                extra_layer,
                ltr_matrix,
                arc_count_per_node,
                node_bin_counts,
                node_cumulative,
                global_lower,
                global_upper,
                bins_matrix,
                w,
                timing_stats,
                calculate_child_time_budget(time_remaining, fn_start)
            )
        end

        # Check if compute_relaxed_dd! timed out (sentinel value -1)
        if last_exact_layer_idx == -1
            return bkv, false, extra_layer
        end

        @time_operation timing_stats post_relaxed_dd begin
            # 3) post‐process to extract the best‐so‐far LTR
            relaxed_bound, bkv, is_feasible, bks_was_updated = post_process_relaxed_dd!(
                queue_nodes,
                bkv,
                bks_int,
                bks_cont,

                fc_new_nodes,
                rel_path,
                feasibility_accumulator,

                node_matrix,

                num_int_vars,
                num_cont_vars,
                num_constraints,

                ltr_matrix,
                ltt_matrix,
                lb_matrix,
                ub_matrix,
                int_obj_coeffs,
                layer_idx,
                qnode.path,
                coefficient_matrix_int_cols,

                coefficient_matrix_rhs_vector,
                wrk_vec,
                infimum_gap_matrices,
                buffer_offset,
                last_exact_layer_idx,
                cont_inf_gap_ctrbtns,
                rough_bounds_cont_val,
                int_var_to_pos_rows,
                int_var_to_neg_rows,
                lbs_int,
                ubs_int,
                lp_sub_model,
                lp_vars,
                lp_constraint_refs,
                timing_stats,
                debug_mode = debug_mode,
                time_remaining = calculate_child_time_budget(time_remaining, fn_start)
            )
        end
    end

    return bkv, bks_was_updated, extra_layer
end
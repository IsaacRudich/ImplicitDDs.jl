"""
    DDWorkspace{Z<:Integer, T<:Real}

Preallocated workspace for decision diagram construction and branch-and-bound operations.
Contains all temporary buffers and data structures needed for a single solver thread.

Type parameters match the problem's integer type (Z) and numerical precision (T).

# Fields

## Relaxed DD Matrices and Vectors
- `lb_matrix::Union{Matrix{Z}, BitMatrix}`: Lower bound matrix for column bounds (width: relaxed_w)
- `ub_matrix::Union{Matrix{Z}, BitMatrix}`: Upper bound matrix for column bounds (width: relaxed_w)
- `infimum_gap_matrices::Matrix{T}`: Double-buffered infimum gap matrix (width: 2*relaxed_w)
- `ltr_matrix::Matrix{T}`: Length-to-root matrix (width: relaxed_w)
- `ltt_matrix::Matrix{T}`: Length-to-terminal matrix (width: relaxed_w)
- `node_matrix::Vector{NodeLayer{Z}}`: Node matrix for relaxed DDs (width: relaxed_w)
- `extra_layer::NodeLayer{Z}`: Extra layer for relaxed DDs (width: relaxed_w)

## Restricted DD Structures
- `rdd::RestrictedDD`: Restricted decision diagram with all layers and gap buffers (width: restricted_w)

## Working Vectors
- `low_indexes::Vector{<:Integer}`: Working vector for low indexes (relaxed DD only, width: relaxed_w, type: Int8/16/32/64 based on w)
- `high_indexes::Vector{<:Integer}`: Working vector for high indexes (relaxed DD only, width: relaxed_w, type: Int8/16/32/64 based on w)
- `bin_counts::Vector{Int}`: Working vector for bin counts (both DDs, width: w)
- `wrk_vec::Vector{T}`: Working vector for computations (both DDs)
- `rel_path::Union{Vector{Z}, BitVector}`: Relaxed path vector (relaxed DD only)
- `feasibility_accumulator::Vector{T}`: Feasibility accumulator vector (relaxed DD only)
- `arc_count_per_node::Vector{<:Integer}`: Workspace for feasible arc counts per node, size: [max_domain_size] (max value: w, type: Int8/16/32/64 based on w)
- `node_bin_counts::Matrix{<:Integer}`: Workspace for bin count histograms per node, size: [relaxed_w, max_domain_size] (max value: w, type: Int8/16/32/64 based on w)
- `node_cumulative::Matrix{<:Integer}`: Workspace for cumulative bin counts per node, size: [relaxed_w, max_domain_size] (max value: w, type: Int8/16/32/64 based on w)
- `global_lower::Vector{<:Integer}`: Workspace for global lower bound estimates per threshold, size: [relaxed_w] (max value: K×w, type: Int8/16/32/64 based on K×w)
- `global_upper::Vector{<:Integer}`: Workspace for global upper bound estimates per threshold, size: [relaxed_w] (max value: K×w, type: Int8/16/32/64 based on K×w)
- `bins_matrix::Matrix{<:Integer}`: Workspace for individual arc-to-bin mappings, size: [max_domain_size, relaxed_w] (stores bin index for each in-arc of each parent node, type: Int8/16/32/64 based on w)

## Solution Tracking
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector (both DDs)
- `bks_cont::Vector{T}`: Best known continuous solution vector (both DDs)

## Local Bounds (B&B Node-Specific)
- `local_lbs_int::Union{Vector{Z}, BitVector}`: Local lower bounds for integer variables (BnB loop only)
- `local_ubs_int::Union{Vector{Z}, BitVector}`: Local upper bounds for integer variables (BnB loop only)
- `local_lbs_cont::Vector{T}`: Local lower bounds for continuous variables (BnB loop only)
- `local_ubs_cont::Vector{T}`: Local upper bounds for continuous variables (BnB loop only)
- `local_infimum_gaps::Vector{T}`: Local infimum gaps vector (BnB loop only)
- `local_cont_inf_contr::Vector{T}`: Local continuous infimum contributions vector (BnB loop only)
- `rough_bounds_int::Vector{T}`: Local rough bounds vector for integer variables (BnB loop only)

## Branch-and-Bound Structures
- `bnb_queue::Vector{QueueNode{Z, T}}`: Branch-and-bound priority queue (both DDs)
- `added_queue_nodes::Vector{QueueNode{Z, T}}`: Added queue nodes vector (relaxed DD only)

## Logging
- `logs::Vector{Dict{String, Any}}`: Vector of log entries for solution/bounds tracking with true timestamps
- `editable_int_solution::Union{Vector{Z}, BitVector}`: Working buffer for reordering integer solutions to original variable order

## Timing Statistics
- `timing_stats::TimingStats`: Timing statistics for DD operations (both restricted and relaxed, parallel-safe per-worker tracking)
"""
mutable struct DDWorkspace{Z<:Integer, T<:Real}
    # Relaxed DD matrices and vectors
    lb_matrix::Union{Matrix{Z}, BitMatrix}
    ub_matrix::Union{Matrix{Z}, BitMatrix}
    infimum_gap_matrices::Matrix{T}
    ltr_matrix::Matrix{T}
    ltt_matrix::Matrix{T}
    node_matrix::Vector{NodeLayer{Z}}
    extra_layer::NodeLayer{Z}

    # Restricted DD structures
    rdd::RestrictedDD

    # Working vectors (relaxed DD)
    low_indexes::Vector{<:Integer}
    high_indexes::Vector{<:Integer}
    wrk_vec::Vector{T}
    rel_path::Union{Vector{Z}, BitVector}
    feasibility_accumulator::Vector{T}
    arc_count_per_node::Vector{<:Integer}
    node_bin_counts::Matrix{<:Integer}
    node_cumulative::Matrix{<:Integer}
    global_lower::Vector{<:Integer}
    global_upper::Vector{<:Integer}
    bins_matrix::Matrix{<:Integer}

    # Working vectors (restricted DD)
    bin_counts::Vector{<:Integer}
    cumulative_bins::Vector{<:Integer}
    restricted_bins_matrix::Matrix{<:Integer}

    # Precomputed coefficient products (hot loop optimization)
    coeff_times_val::Matrix{T}        # [max_domain_size, num_int_vars] - arc objectives: coeff * val (populated once at startup)
    gap_adjustments::Array{T,3}       # [num_constraints, max_domain_size, num_int_vars] - gap updates: coeff * (bound - val) (populated per-DD)
    inv_coeff::Matrix{T}              # [num_constraints, num_int_vars] - inverse constraint coefficients for division→multiplication (populated once at startup)
    inv_obj_coeffs::Vector{T}         # [num_int_vars] - inverse objective coefficients for cap calculation optimization (populated once at startup)

    # Solution tracking
    bks_int::Union{Vector{Z}, BitVector}
    bks_cont::Vector{T}

    # Local bounds (B&B node-specific)
    local_lbs_int::Union{Vector{Z}, BitVector}
    local_ubs_int::Union{Vector{Z}, BitVector}
    local_lbs_cont::Vector{T}
    local_ubs_cont::Vector{T}

    # Original bounds (pre-FBBT) for consistent array indexing
    original_lbs_int::Union{Vector{Z}, BitVector}  # [num_int_vars] - bounds before any FBBT
    original_ubs_int::Union{Vector{Z}, BitVector}  # [num_int_vars] - bounds before any FBBT
    local_infimum_gaps::Vector{T}
    local_cont_inf_contr::Vector{T}
    rough_bounds_int::Vector{T}

    # Branch-and-bound structures
    bnb_queue::Vector{QueueNode{Z, T}}
    added_queue_nodes::Vector{QueueNode{Z, T}}

    #logging
    logs::Vector{Dict{String, Any}}
    editable_int_solution::Union{Vector{Z}, BitVector}

    #timing statistics
    timing_stats::TimingStats
end


"""
    preallocate_dd_memory(num_int_vars::Int, num_cont_vars::Int, num_constraints::Int, lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector}, int_data_type::Type{Z}, all_vars_binary::Bool, restricted_w::Int, relaxed_w::Int, w::Int, numerical_precision::Type{T}, coefficient_matrix_int_cols::Matrix{T}) where {Z<:Integer, T<:Real}

Preallocates all memory structures required for decision diagram construction, branch-and-bound, and logging.

# Arguments
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables
- `int_data_type::Type{Z}`: Integer data type (Bool, Int8, Int16, Int32, or Int64)
- `all_vars_binary::Bool`: Whether all integer variables are binary
- `restricted_w::Int`: Maximum width for restricted decision diagrams
- `relaxed_w::Int`: Maximum width for relaxed decision diagrams
- `w::Int`: Maximum width across both DD types (max(restricted_w, relaxed_w))
- `numerical_precision::Type{T}`: Numerical precision type (Float32 or Float64)
- `coefficient_matrix_int_cols::Matrix{T}`: Coefficient matrix for integer variables

# Returns
- `DDWorkspace{Z, T}`: Workspace container with all preallocated memory structures (see DDWorkspace documentation for field details)
"""
function preallocate_dd_memory(
    num_int_vars::Int,
    num_cont_vars::Int,
    num_constraints::Int,
    lbs_int::Union{Vector{Z}, BitVector},
    ubs_int::Union{Vector{Z}, BitVector},
    int_data_type::Type{Z},
    all_vars_binary::Bool,
    restricted_w::Int,
    relaxed_w::Int,
    w::Int,
    max_domain_size::Int,
    numerical_precision::Type{T},
    coefficient_matrix_int_cols::Matrix{T}
) where {Z<:Integer, T<:Real}
    # Select integer type based on maximum values stored
    # Type 1: Arrays bounded by w (arc_count_per_node, node_bin_counts, node_cumulative)
    type_w = if relaxed_w <= typemax(Int8)
        Int8
    elseif relaxed_w <= typemax(Int16)
        Int16
    elseif relaxed_w <= typemax(Int32)
        Int32
    else
        Int64
    end

    # Type 2: Arrays bounded by K×w (global_lower, global_upper)
    max_kw = relaxed_w * max_domain_size
    type_kw = if max_kw <= typemax(Int8)
        Int8
    elseif max_kw <= typemax(Int16)
        Int16
    elseif max_kw <= typemax(Int32)
        Int32
    else
        Int64
    end

    # allocate the vectors for the impled column bounds
    # TYPE VARIES BASED ON all_vars_binary
    lb_matrix, ub_matrix = preallocate_column_bound_matrices(relaxed_w, num_int_vars, lbs_int, ubs_int, all_vars_binary)


    #preallocate working vectors
    low_indexes = Vector{type_w}(undef, relaxed_w)
    high_indexes = Vector{type_w}(undef, relaxed_w)
    wrk_vec = preallocate_vector(num_constraints, numerical_precision)


    # TYPE VARIES BASED ON all_vars_binary
    if all_vars_binary
        rel_path = BitVector(undef, num_int_vars)
    else
        rel_path = Vector{int_data_type}(undef, num_int_vars)
    end
    feasibility_accumulator = Vector{numerical_precision}(undef, num_constraints)
    # TYPE VARIES BASED ON all_vars_binary
    if all_vars_binary
        bks_int = BitVector(undef, num_int_vars)
    else
        bks_int = Vector{int_data_type}(undef, num_int_vars)
    end
    if num_cont_vars > 0
        bks_cont = Vector{numerical_precision}(undef, num_cont_vars)
    else
        bks_cont = Vector{numerical_precision}()
    end

    #preallocate gap matrix vars
    infimum_gap_matrices = preallocate_infimum_gap_matrices(coefficient_matrix_int_cols, numerical_precision, relaxed_w)

    # Preallocate local bounds working copies (overwritten each queue node)
    # TYPE VARIES BASED ON all_vars_binary
    if all_vars_binary
        local_lbs_int = BitVector(undef, num_int_vars)
        local_ubs_int = BitVector(undef, num_int_vars)
    else
        local_lbs_int = Vector{int_data_type}(undef, num_int_vars)
        local_ubs_int = Vector{int_data_type}(undef, num_int_vars)
    end

    # Allocate original bounds (pre-FBBT) for consistent array indexing
    if all_vars_binary
        original_lbs_int = BitVector(undef, num_int_vars)
        original_ubs_int = BitVector(undef, num_int_vars)
    else
        original_lbs_int = Vector{int_data_type}(undef, num_int_vars)
        original_ubs_int = Vector{int_data_type}(undef, num_int_vars)
    end
    local_lbs_cont = Vector{numerical_precision}(undef, num_cont_vars)
    local_ubs_cont = Vector{numerical_precision}(undef, num_cont_vars)
    local_infimum_gaps = Vector{numerical_precision}(undef, num_constraints)
    local_cont_inf_contr = Vector{numerical_precision}(undef, num_constraints)


    #preallocate node related matrices
    ltr_matrix = preallocate_zero_matrix(relaxed_w, num_int_vars, numerical_precision)
    ltt_matrix = preallocate_zero_matrix(relaxed_w, num_int_vars, numerical_precision)
    # TYPE VARIES BASED ON all_vars_binary
    node_matrix = NodeMatrix(int_data_type, num_int_vars, relaxed_w)
    extra_layer = NodeLayer{int_data_type}(relaxed_w)
    rdd = RestrictedDD(int_data_type, restricted_w, num_constraints, num_int_vars, numerical_precision)


    #preallocate queue
    # TYPE VARIES BASED ON all_vars_binary
    bnb_queue = Vector{QueueNode{int_data_type, numerical_precision}}()
    added_queue_nodes = Vector{QueueNode{int_data_type, numerical_precision}}()

    rough_bounds_int = Vector{numerical_precision}(undef, num_int_vars)

    # Allocate histogram workspace arrays for dual-threshold selection

    # Relaxed DD histogram workspace
    arc_count_per_node = Vector{type_w}(undef, max_domain_size)
    node_bin_counts = Matrix{type_w}(undef, relaxed_w, max_domain_size)
    node_cumulative = Matrix{type_w}(undef, relaxed_w, max_domain_size)
    global_lower = Vector{type_kw}(undef, relaxed_w)
    global_upper = Vector{type_kw}(undef, relaxed_w)
    bins_matrix = Matrix{type_w}(undef, relaxed_w, max_domain_size)

    # Restricted DD histogram workspace
    bin_counts = Vector{type_kw}(undef, w)
    cumulative_bins = Vector{type_kw}(undef, w)
    restricted_bins_matrix = Matrix{type_w}(undef, max_domain_size + 1, w)  # +1 for sentinel

    # Allocate precomputed coefficient products (populated during preprocessing and per-DD)
    coeff_times_val = Matrix{numerical_precision}(undef, max_domain_size, num_int_vars)
    gap_adjustments = Array{numerical_precision,3}(undef, num_constraints, max_domain_size, num_int_vars)
    inv_coeff = Matrix{numerical_precision}(undef, num_constraints, num_int_vars)
    inv_obj_coeffs = Vector{numerical_precision}(undef, num_int_vars)

    logs = Vector{Dict{String, Any}}()
    if all_vars_binary
        editable_int_solution = BitVector(undef, num_int_vars)
    else
        editable_int_solution = Vector{int_data_type}(undef, num_int_vars)
    end

    timing_stats = TimingStats()

    container = DDWorkspace(
        lb_matrix, ub_matrix, infimum_gap_matrices, ltr_matrix, ltt_matrix,
        node_matrix, extra_layer, rdd,
        low_indexes, high_indexes, wrk_vec, rel_path, feasibility_accumulator,
        arc_count_per_node, node_bin_counts, node_cumulative, global_lower, global_upper, bins_matrix,
        bin_counts, cumulative_bins, restricted_bins_matrix,
        coeff_times_val, gap_adjustments, inv_coeff, inv_obj_coeffs,
        bks_int, bks_cont,
        local_lbs_int, local_ubs_int, local_lbs_cont, local_ubs_cont,
        original_lbs_int, original_ubs_int,
        local_infimum_gaps, local_cont_inf_contr, rough_bounds_int,
        bnb_queue, added_queue_nodes,
        logs, editable_int_solution,
        timing_stats
    )

    return container
end


"""
    compute_workspace_memory(dd_workspaces::Vector{DDWorkspace{Z, T}}) where {Z<:Integer, T<:Real}

Computes the total memory used by all DD workspaces in bytes.

# Arguments
- `dd_workspaces::Vector{DDWorkspace{Z, T}}`: Vector of all DD workspaces

# Returns
- `Int`: Total memory used by all workspaces in bytes
"""
function compute_workspace_memory(dd_workspaces::Vector{DDWorkspace{Z, T}}) where {Z<:Integer, T<:Real}
    total_bytes = 0
    for ws in dd_workspaces
        total_bytes += Base.summarysize(ws)
    end
    return total_bytes
end


"""
    write_logs_to_file(logs::Vector{Dict{String, Any}}, log_file_path::Union{String, Nothing})

Writes all collected log entries to a JSON file.

# Arguments
- `logs::Vector{Dict{String, Any}}`: Vector of log dictionaries to write
- `log_file_path::Union{String, Nothing}`: Path to the JSON log file, or nothing to skip writing
"""
function write_logs_to_file(
    logs::Vector{Dict{String, Any}},
    log_file_path::Union{String, Nothing}
)
    if isnothing(log_file_path)
        return
    end

    open(log_file_path, "a") do io
        for log_entry in logs
            println(io, JSON.json(sanitize_for_json(log_entry)))
        end
    end
end


"""
    log_solution(bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T}, editable_solution::Union{Vector{Z}, BitVector}, inverse_order::Vector{Int}, bkv::T, sense_multiplier::Int, num_int_vars::Int, solution_print::Bool, suppress_all_prints::Bool, is_optimal::Bool, log_file_path::Union{String, Nothing}, solve_start::Float64, true_start::Float64, logs::Vector{Dict{String, Any}}, save_log_data::Bool; timestamp::Union{Float64, Nothing}=nothing) where {Z<:Integer, T<:Real}

Logs solution information to console, file, or saves to logs vector.

# Arguments
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution in METIS order
- `bks_cont::Vector{T}`: Best known continuous solution
- `editable_solution::Union{Vector{Z}, BitVector}`: Working buffer for reordered integer solution
- `inverse_order::Vector{Int}`: Inverse permutation to convert from METIS order to original order
- `bkv::T`: Best known objective value
- `sense_multiplier::Int`: Multiplier for objective sense (+1 for minimization, -1 for maximization)
- `num_int_vars::Int`: Number of integer variables
- `solution_print::Bool`: Whether to print detailed solution values to console
- `suppress_all_prints::Bool`: Whether to suppress all console output
- `is_optimal::Bool`: Whether this solution is proven optimal
- `log_file_path::Union{String, Nothing}`: Path to JSON log file, or nothing for no file logging
- `solve_start::Float64`: Timestamp when solve phase started
- `true_start::Float64`: Timestamp when overall execution started
- `logs::Vector{Dict{String, Any}}`: Vector to save log data to
- `save_log_data::Bool`: Whether to save the log data to logs vector
- `timestamp::Union{Float64, Nothing}`: Optional timestamp to use instead of current time (for merging logs with accurate timestamps)
"""
function log_solution(
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{T},
    editable_solution   ::Union{Vector{Z}, BitVector},
    inverse_order       ::Vector{Int},
    bkv                 ::T,
    sense_multiplier    ::Int,
    num_int_vars        ::Int,
    solution_print      ::Bool,
    suppress_all_prints ::Bool,
    is_optimal          ::Bool,
    log_file_path       ::Union{String, Nothing},
    solve_start         ::Float64,
    true_start          ::Float64,
    logs                ::Vector{Dict{String, Any}},
    save_log_data       ::Bool;
    timestamp           ::Union{Float64, Nothing} = nothing
) where {Z<:Integer, T<:Real}
    #do nothing if nothing to do
    if suppress_all_prints && !solution_print && isnothing(log_file_path) && !save_log_data
        return
    end

    editable_solution[:] = bks_int[inverse_order]
    now = isnothing(timestamp) ? time() : timestamp

    if is_optimal
        leadin = "Optimal "
    else
        leadin = "Feasible "
    end

    if !suppress_all_prints
        println("\n",leadin,"Objective Value: ", sense_multiplier * bkv)

        if solution_print
            #print integers
            println(leadin, "Solution:")

            print("Integer Variables:")
            for (idx,val) in enumerate(editable_solution)
                print("x_$idx:$val  ")
                if idx % 15 == 0
                    println()
                end
            end

            println()

            print("Continuous Variables:")
            for (idx,val) in enumerate(bks_cont)
                shift_idx = idx+num_int_vars
                print("x_$shift_idx:$val  ")
                if idx % 10 == 0
                    println()
                end
            end

            println("\n")
        end
    end

    #create log data if needed
    if !isnothing(log_file_path) || save_log_data
        solution_type = is_optimal ? "optimal" : "feasible"

        # Convert BitVector to regular Int vector for JSON serialization
        int_solution = editable_solution isa BitVector ? Vector{Int}(editable_solution) : editable_solution

        log_data = Dict(
            "objective_value" => sense_multiplier * bkv,
            "solution_type" => solution_type,
            "integer_variable_values" => int_solution,
            "continuous_variable_values" => bks_cont,
            "time_from_solve_start" => now - solve_start,
            "time_from_true_start" => now - true_start
        )

        #write to file if requested
        if !isnothing(log_file_path)
            open(log_file_path, "a") do io
                println(io, JSON.json(sanitize_for_json(log_data)))
            end
        end

        #save to logs vector if requested
        if save_log_data
            push!(logs, log_data)
        end
    end
end


"""
    log_bounds(best_bound::T, bkv::T, sense_multiplier::Int, queue_size::Int, bounds_print::Bool, suppress_all_prints::Bool, log_file_path::Union{String, Nothing}, solve_start::Float64, true_start::Float64, logs::Vector{Dict{String, Any}}, save_log_data::Bool) where {T<:Real}

Logs bound information to console, file, or saves to logs vector.

# Arguments
- `best_bound::T`: Current best lower bound
- `bkv::T`: Best known objective value (upper bound)
- `sense_multiplier::Int`: Multiplier for objective sense (+1 for minimization, -1 for maximization)
- `queue_size::Int`: Current size of the branch-and-bound queue
- `bounds_print::Bool`: Whether to print bounds to console
- `suppress_all_prints::Bool`: Whether to suppress all console output
- `log_file_path::Union{String, Nothing}`: Path to JSON log file, or nothing for no file logging
- `solve_start::Float64`: Timestamp when solve phase started
- `true_start::Float64`: Timestamp when overall execution started
- `logs::Vector{Dict{String, Any}}`: Vector to save log data to
- `save_log_data::Bool`: Whether to save the log data to logs vector
"""
function log_bounds(
    best_bound          ::T,
    bkv                 ::T,
    sense_multiplier    ::Int,
    queue_size          ::Int,
    bounds_print        ::Bool,
    suppress_all_prints ::Bool,
    log_file_path       ::Union{String, Nothing},
    solve_start         ::Float64,
    true_start          ::Float64,
    logs                ::Vector{Dict{String, Any}},
    save_log_data       ::Bool
) where {T<:Real}
    #do nothing if nothing to do
    if suppress_all_prints && !bounds_print && isnothing(log_file_path) && !save_log_data
        return
    end

    now = time()
    if sense_multiplier < 0
        upper_bound = sense_multiplier * best_bound
        lower_bound = sense_multiplier * bkv
    else
        lower_bound = sense_multiplier * best_bound
        upper_bound = sense_multiplier * bkv
    end
    gap = upper_bound - lower_bound
    denominator = abs(lower_bound) + gap
    optimality_gap = (gap / denominator) * 100
    # Handle NaN from Inf/Inf (timeout returns -Inf bound) or 0/0 (optimal value = 0)
    if !isfinite(optimality_gap)
        optimality_gap = gap > 0 ? 100.0 : 0.0
    end

    if !suppress_all_prints && bounds_print
        println("Bounds: ", lower_bound, " to ", upper_bound, "  Optimality Gap: ", round(optimality_gap, digits = 2),"%  Queue Size: ", queue_size)
    end

    #create log data if needed
    if !isnothing(log_file_path) || save_log_data
        log_data = Dict(
            "lower_bound" => lower_bound,
            "upper_bound" => upper_bound,
            "optimality_gap" => optimality_gap,
            "queue_size" => queue_size,
            "time_from_solve_start" => now - solve_start,
            "time_from_true_start" => now - true_start
        )

        #write to file if requested
        if !isnothing(log_file_path)
            open(log_file_path, "a") do io
                println(io, JSON.json(sanitize_for_json(log_data)))
            end
        end

        #save to logs vector if requested
        if save_log_data
            push!(logs, log_data)
        end
    end
end


"""
    merge_and_log_solution!(dd_ws_logs::Vector{Dict{String, Any}}, main_logs::Vector{Dict{String, Any}}, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T}, editable_solution::Union{Vector{Z}, BitVector}, inverse_order::Vector{Int}, bkv::T, sense_multiplier::Int, num_int_vars::Int, solution_print::Bool, suppress_all_prints::Bool, log_file_path::Union{String, Nothing}, solve_start::Float64, true_start::Float64, wait_to_write_solutions::Bool) where {Z<:Integer, T<:Real}

Merges workspace logs into main logs and optionally writes to file with console output.

# Arguments
- `dd_ws_logs::Vector{Dict{String, Any}}`: Workspace logs vector with accurate timestamps from subroutines
- `main_logs::Vector{Dict{String, Any}}`: Main logs vector for aggregation
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution in METIS order
- `bks_cont::Vector{T}`: Best known continuous solution
- `editable_solution::Union{Vector{Z}, BitVector}`: Working buffer for reordered integer solution
- `inverse_order::Vector{Int}`: Inverse permutation to convert from METIS order to original order
- `bkv::T`: Best known objective value
- `sense_multiplier::Int`: Multiplier for objective sense (+1 for minimization, -1 for maximization)
- `num_int_vars::Int`: Number of integer variables
- `solution_print::Bool`: Whether to print detailed solution values to console
- `suppress_all_prints::Bool`: Whether to suppress all console output
- `log_file_path::Union{String, Nothing}`: Path to JSON log file, or nothing for no file logging
- `solve_start::Float64`: Timestamp when solve phase started
- `true_start::Float64`: Timestamp when overall execution started
- `wait_to_write_solutions::Bool`: Whether to batch writes (true) or write immediately (false)
"""
function merge_and_log_solution!(
    dd_ws_logs          ::Vector{Dict{String, Any}},
    main_logs           ::Vector{Dict{String, Any}},
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{T},
    editable_solution   ::Union{Vector{Z}, BitVector},
    inverse_order       ::Vector{Int},
    bkv                 ::T,
    sense_multiplier    ::Int,
    num_int_vars        ::Int,
    solution_print      ::Bool,
    suppress_all_prints ::Bool,
    log_file_path       ::Union{String, Nothing},
    solve_start         ::Float64,
    true_start          ::Float64,
    wait_to_write_solutions::Bool
) where {Z<:Integer, T<:Real}

    # Merge workspace logs into main logs
   for latest in dd_ws_logs
        # Get the latest log entry for timestamp and console output
        latest_timestamp = latest["time_from_true_start"] + true_start

        # Console output using log_solution with the accurate timestamp from subroutine
        log_solution(
            bks_int, bks_cont, editable_solution, inverse_order, bkv, sense_multiplier,
            num_int_vars, solution_print, suppress_all_prints, false,
            log_file_path,  # Write to file in immediate mode, nothing in batch mode
            solve_start, true_start,
            main_logs, wait_to_write_solutions,  # Save to main_logs in batch mode
            timestamp = latest_timestamp  # Use accurate timestamp from subroutine
        )
    end

    empty!(dd_ws_logs)
end


"""
    finalize_and_log_timing!(timer_outputs::Bool, wait_to_write_solutions::Bool, num_workers::Int, dd_ws::DDWorkspace{Z,T}, dd_workspaces::Vector{DDWorkspace{Z,T}}, bnb_time::Float64, double_counted_time::Float64, conversion_time::Float64, preallocating_memory_time::Float64, OBBT_time::Float64, preprocessing_time::Float64, initial_restricted_dd_time::Float64, initial_relaxed_dd_time::Float64, true_start::Float64, logs::Vector{Dict{String, Any}}, log_file_path::Union{String, Nothing}) where {Z<:Integer, T<:Real}

Finalize timing statistics, merge worker contributions, and write logs to file.

This function is called before returning from solve_mip to ensure timing and solution logs are properly finalized regardless of which return path is taken.

# Arguments
- `timer_outputs::Bool`: Whether to compute and log timing statistics
- `wait_to_write_solutions::Bool`: Whether to batch-write accumulated solution logs
- `num_workers::Int`: Number of worker threads used
- `dd_ws::DDWorkspace{Z,T}`: Primary worker's DDWorkspace
- `dd_workspaces::Vector{DDWorkspace{Z,T}}`: Vector of all worker DDWorkspace instances
- `bnb_time::Float64`: Time spent in branch-and-bound loop
- `double_counted_time::Float64`: Time already accounted for in timing_stats that needs adjustment
- `conversion_time::Float64`: Time spent converting model to internal format
- `preallocating_memory_time::Float64`: Time spent preallocating memory structures
- `OBBT_time::Float64`: Time spent in optimization-based bound tightening
- `preprocessing_time::Float64`: Time spent in preprocessing phase
- `initial_restricted_dd_time::Float64`: Time spent building initial restricted DD
- `initial_relaxed_dd_time::Float64`: Time spent building initial relaxed DD
- `true_start::Float64`: Absolute start time of solve
- `logs::Vector{Dict{String, Any}}`: Vector of accumulated solution and bound logs
- `log_file_path::Union{String, Nothing}`: Path to log file, or nothing if logging disabled
"""
function finalize_and_log_timing!(
    timer_outputs                  ::Bool,
    wait_to_write_solutions        ::Bool,
    num_workers                    ::Int,
    dd_ws                          ::DDWorkspace{Z,T},
    dd_workspaces                  ::Vector{DDWorkspace{Z,T}},
    bnb_time                       ::Float64,
    double_counted_time            ::Float64,
    conversion_time                ::Float64,
    preallocating_memory_time      ::Float64,
    OBBT_time                      ::Float64,
    preprocessing_time             ::Float64,
    initial_restricted_dd_time     ::Float64,
    initial_relaxed_dd_time        ::Float64,
    true_start                     ::Float64,
    logs                           ::Vector{Dict{String, Any}},
    log_file_path                  ::Union{String, Nothing}
) where {Z<:Integer, T<:Real}
    if timer_outputs
        if num_workers == 1
            dd_ws.timing_stats.other_work_time += bnb_time - double_counted_time
        else
            dd_ws.timing_stats.other_work_time += double_counted_time
            dd_ws.timing_stats.worker_idle_time += bnb_time - double_counted_time
        end
        worker_contributions = nothing
        worker_idle_times = nothing
        if num_workers > 1
            worker_contributions = Vector{Float64}()
            worker_idle_times = Vector{Float64}()
            workers_idle = conversion_time + preallocating_memory_time + OBBT_time + preprocessing_time + initial_restricted_dd_time + initial_relaxed_dd_time
            timing_stats = dd_ws.timing_stats
            @inbounds for idx in 2:num_workers
                ts2 = dd_workspaces[idx].timing_stats
                ts2.worker_idle_time += workers_idle
                merge_timing_stats!(timing_stats, ts2, worker_contributions, worker_idle_times)
            end

        end
        clock_time = time() - true_start
        log_timing_stats(
            dd_ws.timing_stats, worker_contributions = worker_contributions, worker_idle_times = worker_idle_times,
            model_conversion_time = conversion_time, memory_preallocation_time = preallocating_memory_time,
            obbt_time = OBBT_time, preprocessing_time = preprocessing_time, clock_time = clock_time,
            timing_log_file = log_file_path
        )
    end

    if wait_to_write_solutions
        write_logs_to_file(logs, log_file_path)
    end
end
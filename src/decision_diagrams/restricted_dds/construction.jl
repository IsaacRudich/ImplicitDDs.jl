"""
COMMENTS UP TO DATE
"""

"""
    create_restricted_dd!(rdd::RestrictedDD, bkv::T, first_layer_idx::Int, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_int_vars::Int, num_constraints::Int, rough_bounds_int::Vector{T}, rough_bounds_cont_val::T, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, inv_coeff::Matrix{T}, inv_obj_coeffs::Vector{T}, bin_counts::Vector{<:Integer}, bins_matrix::Matrix{<:Integer}, cumulative_bins::Vector{<:Integer}, w::Int, timing_stats::TimingStats, time_remaining::Union{Float64, Nothing} = nothing) where {Z<:Integer, T<:Real}

Constructs a width-limited restricted decision diagram for integer variables using top-down layer-by-layer construction with all layers stored for path reconstruction.

# Arguments
- `rdd::RestrictedDD`: Restricted DD structure with all n layers and double-buffered gap matrices
- `bkv::T`: Best known objective value for arc pruning decisions
- `first_layer_idx::Int`: Starting layer index for DD construction (1 for root, higher for queue nodes)
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables (post-FBBT)
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (post-FBBT)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original lower bounds before any FBBT (for coeff_times_val indexing)
- `num_int_vars::Int`: Total number of integer variables
- `num_constraints::Int`: Number of constraints in the problem
- `rough_bounds_int::Vector{T}`: Rough bound estimates for remaining integer variables (used for pruning)
- `rough_bounds_cont_val::T`: Rough bound contribution from continuous variables
- `coeff_times_val::Matrix{T}`: Precomputed arc objective contributions [max_domain_size, num_int_vars] (indexed with original bounds, populated once at startup)
- `gap_adjustments::Array{T,3}`: Precomputed gap adjustment values [num_constraints, max_domain_size, num_int_vars] (indexed with current bounds, repopulated by FBBT)
- `inv_coeff::Matrix{T}`: Precomputed inverse constraint coefficients [num_constraints, num_int_vars] for division→multiplication (hot loop optimization)
- `inv_obj_coeffs::Vector{T}`: Precomputed inverse objective coefficients [num_int_vars] for cap calculation optimization
- `bin_counts::Vector{<:Integer}`: Pre-allocated workspace vector for histogram bin counting [w]
- `bins_matrix::Matrix{<:Integer}`: Pre-allocated workspace matrix for arc-to-bin mapping [max_domain_size, w]
- `cumulative_bins::Vector{<:Integer}`: Pre-allocated workspace vector for cumulative sum computation [w]
- `w::Int`: Width limit for the restricted decision diagram
- `timing_stats::TimingStats`: Timing statistics collector for performance profiling
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{Bool, Bool, Matrix{T}}`: A tuple containing:
  1. `Bool`: True if DD construction was exact (no width truncation occurred), false otherwise
  2. `Bool`: True if at least one feasible path exists in the DD, false otherwise
  3. `Matrix{T}`: Gap buffer matrix containing the terminal layer gaps

# Notes
Terminal layer is always at rdd.layers[num_int_vars] if construction succeeds (is_feasible = true).
Gap buffers: first_layer_idx always starts in gap_buffer_a, then layers alternate.
"""
function create_restricted_dd!(
    rdd                 ::RestrictedDD,
    bkv                 ::T,
    first_layer_idx     ::Int,
    #problem description
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    ubs_int                      ::Union{Vector{Z}, BitVector},
    lbs_int                      ::Union{Vector{Z}, BitVector},
    original_lbs_int             ::Union{Vector{Z}, BitVector},
    num_int_vars                 ::Int,
    num_constraints              ::Int,

    #precomputed values
    rough_bounds_int        ::Vector{T},
    rough_bounds_cont_val   ::T,
    coeff_times_val         ::Matrix{T},
    gap_adjustments         ::Array{T,3},
    inv_coeff               ::Matrix{T},
    inv_obj_coeffs          ::Vector{T},

    #preallocated memory
    bin_counts          ::Vector{<:Integer},
    bins_matrix         ::Matrix{<:Integer},
    cumulative_bins     ::Vector{<:Integer},

    #settings
    w                   ::Int,
    timing_stats        ::TimingStats,
    time_remaining      ::Union{Float64, Nothing} = nothing,
)where{Z<:Integer, T<:Real}
    fn_start = time()

    is_exact = true
    current_layer = rdd.layers[first_layer_idx]

    # Always start with gap_buffer_a, then alternate
    gap_a = rdd.gap_buffers.gap_matrix_a
    gap_b = rdd.gap_buffers.gap_matrix_b

    for var_index in first_layer_idx:num_int_vars-1
        # Check time budget at layer boundary
        if time_budget_exceeded(time_remaining, fn_start)
            return false, false, gap_a
        end

        next_layer = rdd.layers[var_index + 1]

        lb_col = current_layer.implied_lbs
        ub_col = current_layer.implied_ubs

        inv_coeff_val = inv_obj_coeffs[var_index+1]

        @time_operation timing_stats restricted_dd_implied_column_bounds begin
            # Compute the implied bounds from gap_a
            compute_implied_column_bounds!(lb_col, ub_col, var_index+1, gap_a, int_var_to_pos_rows, int_var_to_neg_rows, lbs_int, ubs_int, inv_coeff, current_layer.size)
        end

        @time_operation timing_stats restricted_dd_histogram_approximation begin
            threshold_bin, uncapped = find_cap_histogram_approximation!(current_layer, inv_coeff_val, w, bkv, rough_bounds_int, rough_bounds_cont_val, var_index+1, original_lbs_int, coeff_times_val, bin_counts, bins_matrix, cumulative_bins)
        end

        if is_exact & !uncapped
            is_exact = false
        end

        # Early termination if no feasible arcs exist
        if threshold_bin == 0 && uncapped
            return is_exact, false, gap_a
        end

        @time_operation timing_stats restricted_dd_build_layer begin
            # Read from gap_a, write to gap_b
            build_restricted_node_layer!(current_layer, next_layer, gap_a, gap_b, inv_coeff_val, var_index, threshold_bin, uncapped, lbs_int, original_lbs_int, num_constraints, coeff_times_val, gap_adjustments, bins_matrix, w)
        end

        current_layer = next_layer

        if current_layer.size == 0
            return is_exact, false, gap_b
        end

        # Swap buffers for next iteration
        gap_a, gap_b = gap_b, gap_a
    end

    # Terminal gaps are in gap_a (we swapped after writing to gap_b)
    return is_exact, true, gap_a
end


"""
    setup_and_run_restricted_dd!(rdd::RestrictedDD, bkv::T, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T}, obj_const::T, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_int_vars::Int, num_cont_vars::Int, num_constraints::Int, rough_bounds_int::Vector{T}, rough_bounds_cont_val::T, infimum_gaps::Vector{T}, cont_inf_gap_ctrbtns::Vector{T}, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, inv_coeff::Matrix{T}, inv_obj_coeffs::Vector{T}, lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef}, bin_counts::Vector{<:Integer}, bins_matrix::Matrix{<:Integer}, cumulative_bins::Vector{<:Integer}, w::Int, num_LPs_to_run::Int, timing_stats::TimingStats, time_remaining::Union{Float64, Nothing} = nothing) where {Z<:Integer, T<:Real}

Initializes and constructs a complete restricted decision diagram from the root to find initial feasible MIP solutions.

# Arguments
- `rdd::RestrictedDD`: Restricted DD structure with all n layers and double-buffered gap matrices
- `bkv::T`: Best known objective value to compare against for solution updates
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update if improved solutions are found
- `bks_cont::Vector{T}`: Best known continuous solution vector to update if improved solutions are found
- `obj_const::T`: Constant term in the objective function
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables (post-FBBT)
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (post-FBBT)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original lower bounds before any FBBT (for coeff_times_val indexing)
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints in the problem
- `rough_bounds_int::Vector{T}`: Rough bound estimates for remaining integer variables (used for pruning)
- `rough_bounds_cont_val::T`: Rough bound contribution from continuous variables
- `infimum_gaps::Vector{T}`: Base infimum gap values before any variable assignments
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps
- `coeff_times_val::Matrix{T}`: Precomputed arc objective contributions [max_domain_size, num_int_vars] (indexed with original bounds, populated once at startup)
- `gap_adjustments::Array{T,3}`: Precomputed gap adjustment values [num_constraints, max_domain_size, num_int_vars] (indexed with current bounds, repopulated by FBBT)
- `inv_coeff::Matrix{T}`: Precomputed inverse constraint coefficients [num_constraints, num_int_vars] for division→multiplication (hot loop optimization)
- `inv_obj_coeffs::Vector{T}`: Precomputed inverse objective coefficients [num_int_vars] for cap calculation optimization
- `lp_sub_model::JuMP.Model`: LP model for continuous variable optimization
- `lp_vars::Vector{JuMP.VariableRef}`: References to continuous variable objects
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to constraint objects for RHS updates
- `bin_counts::Vector{<:Integer}`: Pre-allocated workspace vector for histogram bin counting [w]
- `bins_matrix::Matrix{<:Integer}`: Pre-allocated workspace matrix for arc-to-bin mapping [max_domain_size, w]
- `cumulative_bins::Vector{<:Integer}`: Pre-allocated workspace vector for cumulative sum computation [w]
- `w::Int`: Width limit for the restricted decision diagram
- `num_LPs_to_run::Int`: Maximum number of LP subproblems to solve at terminal nodes
- `timing_stats::TimingStats`: Timing statistics collector for performance profiling
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{T, Bool, Bool}`: A tuple containing:
  1. `T`: Best objective value found after DD construction and continuous optimization
  2. `Bool`: True if DD construction was exact (no width truncation occurred), false otherwise
  3. `Bool`: True if at least one feasible solution was found, false otherwise
"""
function setup_and_run_restricted_dd!(
    rdd                 ::RestrictedDD,
    bkv                 ::T,
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{T},
    #problem description
    obj_const           ::T,
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    ubs_int             ::Union{Vector{Z}, BitVector},
    lbs_int             ::Union{Vector{Z}, BitVector},
    original_lbs_int    ::Union{Vector{Z}, BitVector},
    num_int_vars        ::Int,
    num_cont_vars       ::Int,
    num_constraints     ::Int,

    #precomputed values
    rough_bounds_int        ::Vector{T},
    rough_bounds_cont_val   ::T,
    infimum_gaps            ::Vector{T},
    cont_inf_gap_ctrbtns    ::Vector{T},
    coeff_times_val         ::Matrix{T},
    gap_adjustments         ::Array{T,3},
    inv_coeff               ::Matrix{T},
    inv_obj_coeffs          ::Vector{T},

    lp_sub_model        ::JuMP.Model,
    lp_vars             ::Vector{JuMP.VariableRef},
    lp_constraint_refs  ::Vector{JuMP.ConstraintRef},

    #preallocated
    bin_counts          ::Vector{<:Integer},
    bins_matrix         ::Matrix{<:Integer},
    cumulative_bins     ::Vector{<:Integer},

    #settings
    w                   ::Int,
    num_LPs_to_run      ::Int,
    timing_stats        ::TimingStats,
    time_remaining      ::Union{Float64, Nothing} = nothing,
)where{Z<:Integer, T<:Real}
    fn_start = time()
    @time_operation timing_stats restricted_dd begin
        gap_buffer_a = rdd.gap_buffers.gap_matrix_a
        first_layer = rdd.layers[1]

        generate_initial_node_layer!(first_layer, gap_buffer_a, obj_const, ubs_int, lbs_int, original_lbs_int, infimum_gaps, coeff_times_val, gap_adjustments)

        @time_operation timing_stats create_restricted_dd begin
            is_exact, is_feasible, terminal_gap_buffer = create_restricted_dd!(
                rdd, bkv, 1,
                int_var_to_pos_rows, int_var_to_neg_rows, ubs_int, lbs_int, original_lbs_int, num_int_vars, num_constraints,
                rough_bounds_int, rough_bounds_cont_val,
                coeff_times_val, gap_adjustments, inv_coeff, inv_obj_coeffs,
                bin_counts, bins_matrix, cumulative_bins, w, timing_stats,
                calculate_child_time_budget(time_remaining, fn_start)
            )
        end

        @time_operation timing_stats post_restricted_dd begin
            if is_feasible
                bkv, is_feasible, is_exact = post_process_restricted_dd!(bkv, bks_int, bks_cont, is_exact, rdd, terminal_gap_buffer, nothing, 1, num_int_vars, num_cont_vars, num_constraints, cont_inf_gap_ctrbtns, rough_bounds_cont_val, lp_sub_model, lp_vars, lp_constraint_refs, num_LPs_to_run, timing_stats, calculate_child_time_budget(time_remaining, fn_start))
            end
        end
    end

    return bkv, is_exact, is_feasible
end


"""
    setup_and_run_restricted_dd!(rdd::RestrictedDD, qnode::QueueNode, bkv::T, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, ubs_int::Union{Vector{Z}, BitVector}, lbs_int::Union{Vector{Z}, BitVector}, original_lbs_int::Union{Vector{Z}, BitVector}, num_int_vars::Int, num_cont_vars::Int, num_constraints::Int, rough_bounds_int::Vector{T}, rough_bounds_cont_val::T, infimum_gaps::Vector{T}, cont_inf_gap_ctrbtns::Vector{T}, coeff_times_val::Matrix{T}, gap_adjustments::Array{T,3}, inv_coeff::Matrix{T}, inv_obj_coeffs::Vector{T}, lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef}, bin_counts::Vector{<:Integer}, bins_matrix::Matrix{<:Integer}, cumulative_bins::Vector{<:Integer}, w::Int, num_LPs_to_run::Int, timing_stats::TimingStats, time_remaining::Union{Float64, Nothing} = nothing) where {Z<:Integer, T<:Real}

Constructs a restricted decision diagram from a branch-and-bound queue node with partially fixed integer variables.

# Arguments
- `rdd::RestrictedDD`: Restricted DD structure with all n layers and double-buffered gap matrices
- `qnode::QueueNode`: Queue node containing partially fixed integer variable assignments in its path
- `bkv::T`: Best known objective value to compare against for solution updates
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update if improved solutions are found
- `bks_cont::Vector{T}`: Best known continuous solution vector to update if improved solutions are found
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `ubs_int::Union{Vector{Z}, BitVector}`: Upper bounds for integer variables (post-FBBT)
- `lbs_int::Union{Vector{Z}, BitVector}`: Lower bounds for integer variables (post-FBBT)
- `original_lbs_int::Union{Vector{Z}, BitVector}`: Original lower bounds before any FBBT (for coeff_times_val indexing)
- `num_int_vars::Int`: Number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints in the problem
- `rough_bounds_int::Vector{T}`: Rough bound estimates for remaining integer variables (used for pruning)
- `rough_bounds_cont_val::T`: Rough bound contribution from continuous variables
- `infimum_gaps::Vector{T}`: Base infimum gap values for constraint state reconstruction
- `cont_inf_gap_ctrbtns::Vector{T}`: Continuous variable contributions to infimum gaps
- `coeff_times_val::Matrix{T}`: Precomputed arc objective contributions [max_domain_size, num_int_vars] (indexed with original bounds, populated once at startup)
- `gap_adjustments::Array{T,3}`: Precomputed gap adjustment values [num_constraints, max_domain_size, num_int_vars] (indexed with current bounds, repopulated by FBBT)
- `inv_coeff::Matrix{T}`: Precomputed inverse constraint coefficients [num_constraints, num_int_vars] for division→multiplication (hot loop optimization)
- `inv_obj_coeffs::Vector{T}`: Precomputed inverse objective coefficients [num_int_vars] for cap calculation optimization
- `lp_sub_model::JuMP.Model`: LP model for continuous variable optimization
- `lp_vars::Vector{JuMP.VariableRef}`: References to continuous variable objects
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to constraint objects for RHS updates
- `bin_counts::Vector{<:Integer}`: Pre-allocated workspace vector for histogram bin counting [w]
- `bins_matrix::Matrix{<:Integer}`: Pre-allocated workspace matrix for arc-to-bin mapping [max_domain_size, w]
- `cumulative_bins::Vector{<:Integer}`: Pre-allocated workspace vector for cumulative sum computation [w]
- `w::Int`: Width limit for the restricted decision diagram
- `num_LPs_to_run::Int`: Maximum number of LP subproblems to solve at terminal nodes
- `timing_stats::TimingStats`: Timing statistics collector for performance profiling
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{T, Bool}`: A tuple containing:
  1. `T`: Best objective value found after DD construction and continuous optimization
  2. `Bool`: True if DD construction was exact (no width truncation occurred), false otherwise
"""
function setup_and_run_restricted_dd!(
    rdd                 ::RestrictedDD,
    qnode               ::QueueNode,
    bkv                 ::T,
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{T},

    #problem description
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    ubs_int             ::Union{Vector{Z}, BitVector},
    lbs_int             ::Union{Vector{Z}, BitVector},
    original_lbs_int    ::Union{Vector{Z}, BitVector},
    num_int_vars        ::Int,
    num_cont_vars       ::Int,
    num_constraints     ::Int,

    #precomputed values
    rough_bounds_int        ::Vector{T},
    rough_bounds_cont_val   ::T,
    infimum_gaps            ::Vector{T},
    cont_inf_gap_ctrbtns    ::Vector{T},
    coeff_times_val         ::Matrix{T},
    gap_adjustments         ::Array{T,3},
    inv_coeff               ::Matrix{T},
    inv_obj_coeffs          ::Vector{T},

    lp_sub_model        ::JuMP.Model,
    lp_vars             ::Vector{JuMP.VariableRef},
    lp_constraint_refs  ::Vector{JuMP.ConstraintRef},

    #preallocated
    bin_counts          ::Vector{<:Integer},
    bins_matrix         ::Matrix{<:Integer},
    cumulative_bins     ::Vector{<:Integer},

    #settings
    w                   ::Int,
    num_LPs_to_run      ::Int,
    timing_stats        ::TimingStats,
    time_remaining      ::Union{Float64, Nothing} = nothing,
)where{Z<:Integer, T<:Real}
    fn_start = time()
    @time_operation timing_stats restricted_dd begin
        first_layer_idx = length(qnode.path)
        gap_buffer_a = rdd.gap_buffers.gap_matrix_a
        first_layer = rdd.layers[first_layer_idx]

        generate_node_layer_from_queue!(
            qnode, first_layer, gap_buffer_a, infimum_gaps
        )

        @time_operation timing_stats create_restricted_dd begin
            is_exact, is_feasible, terminal_gap_buffer = create_restricted_dd!(
                rdd, bkv, first_layer_idx,
                int_var_to_pos_rows, int_var_to_neg_rows, ubs_int, lbs_int, original_lbs_int, num_int_vars, num_constraints,
                rough_bounds_int, rough_bounds_cont_val,
                coeff_times_val, gap_adjustments, inv_coeff, inv_obj_coeffs,
                bin_counts, bins_matrix, cumulative_bins, w, timing_stats,
                calculate_child_time_budget(time_remaining, fn_start)
            )
        end

        @time_operation timing_stats post_restricted_dd begin
            if is_feasible
                bkv, is_feasible, is_exact = post_process_restricted_dd!(bkv, bks_int, bks_cont, is_exact, rdd, terminal_gap_buffer, qnode, first_layer_idx, num_int_vars, num_cont_vars, num_constraints, cont_inf_gap_ctrbtns, rough_bounds_cont_val, lp_sub_model, lp_vars, lp_constraint_refs, num_LPs_to_run, timing_stats, calculate_child_time_budget(time_remaining, fn_start))
            end
        end
    end

    return bkv, is_exact
end

"""
    post_process_restricted_dd!(bkv::V, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{V}, is_exact::Bool, rdd::RestrictedDD, terminal_gap_buffer::Matrix{V}, qnode::Union{QueueNode, Nothing}, first_layer_idx::Int, num_int_vars::Int, num_cont_vars::Int, num_constraints::Int, cont_inf_gap_ctrbtns::Vector{V}, rough_bounds_cont_val::V, lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef}, num_LPs_to_run::Int, timing_stats::TimingStats, time_remaining::Union{Float64, Nothing} = nothing) where {Z<:Integer, V<:Real}

Post-processes restricted decision diagram results by solving continuous LPs and reconstructing solution paths.

# Arguments
- `bkv::V`: Best known objective value to compare against for solution updates
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update if improved solutions are found
- `bks_cont::Vector{V}`: Best known continuous solution vector to update if improved solutions are found
- `is_exact::Bool`: Whether the restricted DD integer construction was exact
- `rdd::RestrictedDD`: Complete restricted DD with all layers
- `terminal_gap_buffer::Matrix{V}`: Gap buffer matrix containing the terminal layer gaps
- `qnode::Union{QueueNode, Nothing}`: Queue node if started from partial solution
- `first_layer_idx::Int`: First layer that was built
- `num_int_vars::Int`: Total number of integer variables
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints in the problem
- `cont_inf_gap_ctrbtns::Vector{V}`: Continuous variable contributions to infimum gaps
- `rough_bounds_cont_val::V`: Rough bound contribution from continuous variables
- `lp_sub_model::JuMP.Model`: LP model for continuous variable optimization
- `lp_vars::Vector{JuMP.VariableRef}`: References to continuous variable objects
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to constraint objects for RHS updates
- `num_LPs_to_run::Int`: Maximum number of LP subproblems to solve
- `timing_stats::TimingStats`: Timing statistics collector for performance profiling
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{V, Bool, Bool}`: A tuple containing:
  1. `V`: Updated best known objective value
  2. `Bool`: True if at least one feasible solution was found
  3. `Bool`: True if restricted DD is fully exact
"""
function post_process_restricted_dd!(
    bkv                 ::V,
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{V},
    is_exact            ::Bool,
    rdd                 ::RestrictedDD,
    terminal_gap_buffer ::Matrix{V},
    qnode               ::Union{QueueNode, Nothing},
    first_layer_idx     ::Int,
    num_int_vars        ::Int,
    num_cont_vars       ::Int,
    num_constraints     ::Int,
    cont_inf_gap_ctrbtns::Vector{V},
    rough_bounds_cont_val::V,
    lp_sub_model        ::JuMP.Model,
    lp_vars             ::Vector{JuMP.VariableRef},
    lp_constraint_refs  ::Vector{JuMP.ConstraintRef},
    num_LPs_to_run      ::Int,
    timing_stats        ::TimingStats,
    time_remaining      ::Union{Float64, Nothing} = nothing,
) where{Z<:Integer, V<:Real}
    fn_start = time()

    if num_cont_vars > 0
        # Finish the partial solutions with LPs
        bkv, is_feasible, lps_exact = run_restricted_LPs!(bkv, bks_int, bks_cont, rdd, terminal_gap_buffer, qnode, first_layer_idx, num_int_vars, num_cont_vars, num_constraints, cont_inf_gap_ctrbtns, rough_bounds_cont_val, lp_sub_model, lp_vars, lp_constraint_refs, num_LPs_to_run, timing_stats, calculate_child_time_budget(time_remaining, fn_start))

        overall_exact = is_exact && lps_exact
        return V(bkv), is_feasible, overall_exact

    else
        # No continuous variables - just reconstruct the best integer solution path
        if time_budget_exceeded(time_remaining, fn_start)
            return bkv, false, false
        end

        terminal_layer = rdd.layers[num_int_vars]
        idx = argmin(terminal_layer.ltrs)

        # Reconstruct complete path (including queue node prefix if applicable)
        reconstruct_path!(bks_int, rdd.layers, idx, qnode, first_layer_idx, num_int_vars)

        return terminal_layer.ltrs[idx], true, is_exact
    end
end

"""
    run_restricted_LPs!(bkv::U, bks_int::Union{Vector{Z}, BitVector}, bks_cont::Vector{U}, rdd::RestrictedDD, terminal_gap_buffer::Matrix{U}, qnode::Union{QueueNode, Nothing}, first_layer_idx::Int, num_int_vars::Int, num_cont_vars::Int, num_constraints::Int, cont_inf_gap_ctrbtns::Vector{U}, rough_bounds_cont_val::U, lp_sub_model::JuMP.Model, lp_vars::Vector{JuMP.VariableRef}, lp_constraint_refs::Vector{JuMP.ConstraintRef}, num_LPs_to_run::Int, timing_stats::TimingStats, time_remaining::Union{Float64, Nothing} = nothing) where {Z<:Integer, U<:Real}

Solves LP subproblems for terminal nodes in a restricted decision diagram to find feasible solutions to a MIP.

# Arguments
- `bkv::U`: Best known objective value to compare against for solution updates
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution vector to update if improved solutions are found
- `bks_cont::Vector{U}`: Best known continuous solution vector to update if improved solutions are found
- `rdd::RestrictedDD`: Complete restricted DD with all layers
- `terminal_gap_buffer::Matrix{U}`: Gap buffer matrix containing the terminal layer gaps
- `qnode::Union{QueueNode, Nothing}`: Queue node if started from partial solution (nothing if from root)
- `first_layer_idx::Int`: First layer that was built (1 for root, length(qnode.path) for queue node)
- `num_int_vars::Int`: Number of integer variables (terminal layer index)
- `num_cont_vars::Int`: Number of continuous variables
- `num_constraints::Int`: Number of constraints in the problem
- `cont_inf_gap_ctrbtns::Vector{U}`: Continuous variable contributions to infimum gaps
- `rough_bounds_cont_val::U`: Rough bound contribution from continuous variables (used for early termination)
- `lp_sub_model::JuMP.Model`: LP model for continuous variable optimization
- `lp_vars::Vector{JuMP.VariableRef}`: References to continuous variable objects
- `lp_constraint_refs::Vector{JuMP.ConstraintRef}`: References to constraint objects for RHS updates
- `num_LPs_to_run::Int`: Maximum number of LP subproblems to solve
- `timing_stats::TimingStats`: Timing statistics collector for performance profiling
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds for this function (nothing if no time limit)

# Returns
- `Tuple{U, Bool, Bool}`: A tuple containing:
  1. `U`: Updated best known objective value
  2. `Bool`: True if at least one feasible solution was found
  3. `Bool`: True if all promising nodes were processed
"""
function run_restricted_LPs!(
    bkv                 ::U,
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{U},
    rdd                 ::RestrictedDD,
    terminal_gap_buffer ::Matrix{U},
    qnode               ::Union{QueueNode, Nothing},
    first_layer_idx     ::Int,
    num_int_vars        ::Int,
    num_cont_vars       ::Int,
    num_constraints     ::Int,
    cont_inf_gap_ctrbtns::Vector{U},
    rough_bounds_cont_val::U,
    lp_sub_model        ::JuMP.Model,
    lp_vars             ::Vector{JuMP.VariableRef},
    lp_constraint_refs  ::Vector{JuMP.ConstraintRef},
    num_LPs_to_run      ::Int,
    timing_stats        ::TimingStats,
    time_remaining      ::Union{Float64, Nothing} = nothing,
) where{Z<:Integer, U<:Real}
    fn_start = time()

    terminal_layer = rdd.layers[num_int_vars]
    gap_matrix = terminal_gap_buffer
    ltrs = terminal_layer.ltrs
    is_feasible = false

    is_exact = true
    nodes_processed = 0

    node_indices = collect(1:terminal_layer.size)
    sort!(node_indices, by = i -> ltrs[i])

    @inbounds for node_idx in node_indices
        # Check time budget
        if time_budget_exceeded(time_remaining, fn_start)
            is_exact = false
            break
        end

        # Skip if this node can't improve bkv
        if ltrs[node_idx] + rough_bounds_cont_val >= bkv
            continue
        else
            if nodes_processed == num_LPs_to_run
                is_exact = false
                break
            end
            nodes_processed += 1
        end

        # Update the RHS of the constraints
        @inbounds for row_idx in 1:num_constraints
            new_rhs = gap_matrix[row_idx, node_idx] - cont_inf_gap_ctrbtns[row_idx]
            set_normalized_rhs(lp_constraint_refs[row_idx], new_rhs)
        end

        # Solve the LP subproblem
        @time_operation timing_stats restricted_dd_lp_solver_call begin
            optimize!(lp_sub_model)
        end
        timing_stats.simplex_iterations += MOI.get(lp_sub_model, MOI.SimplexIterations())

        if termination_status(lp_sub_model) == MOI.OPTIMAL
            is_feasible = true
            new_obj_value = objective_value(lp_sub_model) + ltrs[node_idx]

            if new_obj_value < bkv
                bkv = new_obj_value

                # Reconstruct complete integer solution path (including queue node prefix if applicable)
                reconstruct_path!(bks_int, rdd.layers, node_idx, qnode, first_layer_idx, num_int_vars)

                # Extract continuous solution
                @inbounds for var_idx in 1:num_cont_vars
                    bks_cont[var_idx] = value(lp_vars[var_idx])
                end
            end
        end
    end

    return bkv, is_feasible, is_exact
end
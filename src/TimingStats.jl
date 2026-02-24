mutable struct TimingStats
    workspace_memory_bytes::Int #total memory used by DD workspaces (computed once after allocation)
    worker_idle_time::Float64 #time spent waiting for work (parallel only)
    other_work_time::Float64 #time spent in B&B overhead (FBBT, rough bounds, logging, etc.)
    simplex_iterations::Int

    restricted_dd_count::Int #the main call
    restricted_dd_total::Float64
        create_restricted_dd_count::Int #the call to actually iterate through the layers
        create_restricted_dd_total::Float64
            restricted_dd_implied_column_bounds_count::Int
            restricted_dd_implied_column_bounds_total::Float64
            restricted_dd_histogram_approximation_count::Int
            restricted_dd_histogram_approximation_total::Float64
            restricted_dd_build_layer_count::Int
            restricted_dd_build_layer_total::Float64
        post_restricted_dd_count::Int #mostly the call to the LPs
        post_restricted_dd_total::Float64
            restricted_dd_lp_solver_call_count::Int #individual call to an LP
            restricted_dd_lp_solver_call_total::Float64

    relaxed_dd_count::Int #the main call
    relaxed_dd_total::Float64
        create_relaxed_dd_count::Int #the call to actually iterate through the layers
        create_relaxed_dd_total::Float64
            relaxed_dd_implied_column_bounds_count::Int
            relaxed_dd_implied_column_bounds_total::Float64
            relaxed_dd_invert_bounds_count::Int
            relaxed_dd_invert_bounds_total::Float64
            relaxed_dd_split_nodes_count::Int
            relaxed_dd_split_nodes_total::Float64
                relaxed_dd_bin_counter_count::Int
                relaxed_dd_bin_counter_total::Float64
                relaxed_dd_layer_construction_count::Int
                relaxed_dd_layer_construction_total::Float64
            relaxed_dd_update_ltr_count::Int
            relaxed_dd_update_ltr_total::Float64
            relaxed_dd_rough_bounding_count::Int
            relaxed_dd_rough_bounding_total::Float64
            relaxed_dd_infimum_matrix_cubic_count::Int
            relaxed_dd_infimum_matrix_cubic_total::Float64
        post_relaxed_dd_count::Int #frontier cutset and LP calls
        post_relaxed_dd_total::Float64
            post_relaxed_dd_true_LP_count::Int
            post_relaxed_dd_true_LP_total::Float64
            post_relaxed_dd_potential_top_level_LP_count::Int
            post_relaxed_dd_potential_top_level_LP_total::Float64
            post_relaxed_dd_update_ltt_count::Int
            post_relaxed_dd_update_ltt_total::Float64
            post_relaxed_dd_exact_cutset_count::Int
            post_relaxed_dd_exact_cutset_total::Float64
                post_relaxed_dd_handle_exact_terminals_count::Int
                post_relaxed_dd_handle_exact_terminals_total::Float64
                    post_relaxed_dd_potential_low_level_LP_count::Int
                    post_relaxed_dd_potential_low_level_LP_total::Float64
                post_relaxed_dd_compute_tighter_bound_count::Int
                post_relaxed_dd_compute_tighter_bound_total::Float64

    TimingStats() = new(
        0,    # workspace_memory_bytes
        0.0,  # worker_idle_time
        0.0,  # other_work_time
        0, #simplex iterations
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,

        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
        0, 0.0,
    )
end

macro time_operation(stats, operation, expr)
    quote
        local start_time = time()
        local result = $(esc(expr))
        $(esc(stats)).$(Symbol(operation, :_total)) += time() - start_time
        $(esc(stats)).$(Symbol(operation, :_count)) += 1
        result
    end
end

"""
    merge_timing_stats!(ts1::TimingStats, ts2::TimingStats, worker_contributions::Vector{Float64}, worker_idle_times::Vector{Float64})

Merge timing statistics from `ts2` into `ts1` by adding all corresponding fields together.
Modifies `ts1` in place and tracks worker contributions and idle times in the provided vectors.

# Arguments
- `ts1::TimingStats`: The destination TimingStats object that will be modified
- `ts2::TimingStats`: The source TimingStats object to merge from
- `worker_contributions::Vector{Float64}`: Vector to accumulate worker DD time contributions (empty for first merge)
- `worker_idle_times::Vector{Float64}`: Vector to accumulate worker idle times (empty for first merge)

# Algorithm
- First merge (empty vectors): Captures ts1's work/idle and ts2's work/idle, pushes both to the vectors
- Subsequent merges: Pushes only ts2's work/idle to the existing vectors
- Work is measured as the sum of restricted_dd_total + relaxed_dd_total + other_work_time for each worker
- Idle time is measured as worker_idle_time for each worker
"""
function merge_timing_stats!(ts1::TimingStats, ts2::TimingStats, worker_contributions::Vector{Float64}, worker_idle_times::Vector{Float64})
    # Calculate work amounts and idle times BEFORE merging
    ts2_work = ts2.restricted_dd_total + ts2.relaxed_dd_total + ts2.other_work_time
    ts2_idle = ts2.worker_idle_time

    if isempty(worker_contributions)
        # First merge: capture ts1's work and idle before they get modified
        ts1_work = ts1.restricted_dd_total + ts1.relaxed_dd_total + ts1.other_work_time
        ts1_idle = ts1.worker_idle_time
        push!(worker_contributions, ts1_work)
        push!(worker_contributions, ts2_work)
        push!(worker_idle_times, ts1_idle)
        push!(worker_idle_times, ts2_idle)
    else
        # Subsequent merge: just add the new worker
        push!(worker_contributions, ts2_work)
        push!(worker_idle_times, ts2_idle)
    end

    # Merge the stats (after capturing original values)
    for field in fieldnames(TimingStats)
        setfield!(ts1, field, getfield(ts1, field) + getfield(ts2, field))
    end

    # Sort both vectors in the same order (smallest to largest contribution)
    sort_perm = sortperm(worker_contributions)
    worker_contributions .= worker_contributions[sort_perm]
    worker_idle_times .= worker_idle_times[sort_perm]
end

function log_timing_stats(ts::TimingStats;
                          worker_contributions::Union{Vector{Float64}, Nothing} = nothing,
                          worker_idle_times::Union{Vector{Float64}, Nothing} = nothing,
                          clock_time::Float64 = 0.0,
                          model_conversion_time::Float64 = 0.0,
                          memory_preallocation_time::Float64 = 0.0,
                          obbt_time::Float64 = 0.0,
                          preprocessing_time::Float64 = 0.0,
                          timing_log_file::Union{String, Nothing} = nothing)
    line_chars = 100

    # Initialize timing data dictionary for JSON output
    timing_data = Dict{String, Any}()

    println("\n" * "="^line_chars)
    println("Decision Diagram Timing Statistics")
    println("="^line_chars)

    # Helper function to format time (converts to ms, s, or m as appropriate)
    function format_time(seconds::Float64)
        if seconds < 0.001
            return @sprintf("%.2f μs", seconds * 1e6)
        elseif seconds < 1.0
            return @sprintf("%.2f ms", seconds * 1000)
        elseif seconds < 60.0
            return @sprintf("%.2f s ", seconds)
        else
            return @sprintf("%.2f m ", seconds / 60)
        end
    end

    # Helper function to format memory (converts to KB, MB, or GB as appropriate)
    function format_memory(bytes::Int)
        if bytes < 1024
            return @sprintf("%d B", bytes)
        elseif bytes < 1024^2
            return @sprintf("%.2f KB", bytes / 1024)
        elseif bytes < 1024^3
            return @sprintf("%.2f MB", bytes / 1024^2)
        else
            return @sprintf("%.2f GB", bytes / 1024^3)
        end
    end

    # Helper to format percentage
    function format_pct(part::Float64, total::Float64)
        if total == 0.0
            return "  -%  "
        else
            return @sprintf("%5.2f%%", (part / total) * 100)
        end
    end

    # Print header
    col_spacing = "%-40s %12s %12s %12s %12s"
    println(Printf.format(Printf.Format(col_spacing), "Operation", "Calls", "Total Time", "Avg Time", "% Total"))
    println("-"^line_chars)

    # High-level summary
    setup_time = model_conversion_time + memory_preallocation_time + obbt_time + preprocessing_time
    solve_time = ts.restricted_dd_total + ts.relaxed_dd_total + ts.other_work_time
    grand_total_time = setup_time + solve_time

    # Store high-level timing data under "overview" key
    timing_data["overview"] = Dict(
        "grand_total_time" => grand_total_time,
        "setup_time" => setup_time,
        "model_conversion_time" => model_conversion_time,
        "memory_preallocation_time" => memory_preallocation_time,
        "workspace_memory_bytes" => ts.workspace_memory_bytes,
        "obbt_time" => obbt_time,
        "preprocessing_time" => preprocessing_time,
        "solve_time" => solve_time,
        "restricted_dd_total" => ts.restricted_dd_total,
        "restricted_dd_lp_count" => ts.restricted_dd_lp_solver_call_count,
        "restricted_dd_lp_total" => ts.restricted_dd_lp_solver_call_total,
        "relaxed_dd_total" => ts.relaxed_dd_total,
        "relaxed_dd_lp_count" => ts.post_relaxed_dd_true_LP_count,
        "relaxed_dd_lp_total" => ts.post_relaxed_dd_true_LP_total,
        "other_work_time" => ts.other_work_time,
        "simplex_iterations" => ts.simplex_iterations
    )

    # Store detailed restricted DD statistics
    timing_data["restricted_dd"] = Dict(
        "count" => ts.restricted_dd_count,
        "total_time" => ts.restricted_dd_total,
        "create" => Dict(
            "count" => ts.create_restricted_dd_count,
            "total_time" => ts.create_restricted_dd_total,
            "implied_column_bounds" => Dict(
                "count" => ts.restricted_dd_implied_column_bounds_count,
                "total_time" => ts.restricted_dd_implied_column_bounds_total
            ),
            "histogram_approximation" => Dict(
                "count" => ts.restricted_dd_histogram_approximation_count,
                "total_time" => ts.restricted_dd_histogram_approximation_total
            ),
            "build_layer" => Dict(
                "count" => ts.restricted_dd_build_layer_count,
                "total_time" => ts.restricted_dd_build_layer_total
            )
        ),
        "post_process" => Dict(
            "count" => ts.post_restricted_dd_count,
            "total_time" => ts.post_restricted_dd_total,
            "lp_solver_calls" => Dict(
                "count" => ts.restricted_dd_lp_solver_call_count,
                "total_time" => ts.restricted_dd_lp_solver_call_total
            )
        )
    )

    # Store detailed relaxed DD statistics
    timing_data["relaxed_dd"] = Dict(
        "count" => ts.relaxed_dd_count,
        "total_time" => ts.relaxed_dd_total,
        "create" => Dict(
            "count" => ts.create_relaxed_dd_count,
            "total_time" => ts.create_relaxed_dd_total,
            "implied_column_bounds" => Dict(
                "count" => ts.relaxed_dd_implied_column_bounds_count,
                "total_time" => ts.relaxed_dd_implied_column_bounds_total
            ),
            "invert_bounds" => Dict(
                "count" => ts.relaxed_dd_invert_bounds_count,
                "total_time" => ts.relaxed_dd_invert_bounds_total
            ),
            "split_nodes" => Dict(
                "count" => ts.relaxed_dd_split_nodes_count,
                "total_time" => ts.relaxed_dd_split_nodes_total,
                "bin_counter" => Dict(
                    "count" => ts.relaxed_dd_bin_counter_count,
                    "total_time" => ts.relaxed_dd_bin_counter_total
                ),
                "layer_construction" => Dict(
                    "count" => ts.relaxed_dd_layer_construction_count,
                    "total_time" => ts.relaxed_dd_layer_construction_total
                )
            ),
            "update_ltr" => Dict(
                "count" => ts.relaxed_dd_update_ltr_count,
                "total_time" => ts.relaxed_dd_update_ltr_total
            ),
            "rough_bounding" => Dict(
                "count" => ts.relaxed_dd_rough_bounding_count,
                "total_time" => ts.relaxed_dd_rough_bounding_total
            ),
            "infimum_matrix_cubic" => Dict(
                "count" => ts.relaxed_dd_infimum_matrix_cubic_count,
                "total_time" => ts.relaxed_dd_infimum_matrix_cubic_total
            )
        ),
        "post_process" => Dict(
            "count" => ts.post_relaxed_dd_count,
            "total_time" => ts.post_relaxed_dd_total,
            "true_LP" => Dict(
                "count" => ts.post_relaxed_dd_true_LP_count,
                "total_time" => ts.post_relaxed_dd_true_LP_total
            ),
            "potential_top_level_LP" => Dict(
                "count" => ts.post_relaxed_dd_potential_top_level_LP_count,
                "total_time" => ts.post_relaxed_dd_potential_top_level_LP_total
            ),
            "update_ltt" => Dict(
                "count" => ts.post_relaxed_dd_update_ltt_count,
                "total_time" => ts.post_relaxed_dd_update_ltt_total
            ),
            "exact_cutset" => Dict(
                "count" => ts.post_relaxed_dd_exact_cutset_count,
                "total_time" => ts.post_relaxed_dd_exact_cutset_total,
                "handle_exact_terminals" => Dict(
                    "count" => ts.post_relaxed_dd_handle_exact_terminals_count,
                    "total_time" => ts.post_relaxed_dd_handle_exact_terminals_total,
                    "potential_low_level_LP" => Dict(
                        "count" => ts.post_relaxed_dd_potential_low_level_LP_count,
                        "total_time" => ts.post_relaxed_dd_potential_low_level_LP_total
                    )
                ),
                "compute_tighter_continuous_bound" => Dict(
                    "count" => ts.post_relaxed_dd_compute_tighter_bound_count,
                    "total_time" => ts.post_relaxed_dd_compute_tighter_bound_total ,
                )
            )
        )
    )

    if grand_total_time > 0.0
        println(Printf.format(Printf.Format(col_spacing),
            "Total Runtime",
            "-",
            format_time(grand_total_time),
            "-",
            "100.0%"))

        println("  │")

        # Setup times
        if setup_time > 0.0
            pct = format_pct(setup_time, grand_total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Setup Time",
                "-",
                format_time(setup_time),
                "-",
                pct))

            if model_conversion_time > 0.0
                pct = format_pct(model_conversion_time, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Model Conversion",
                    "-",
                    format_time(model_conversion_time),
                    "-",
                    pct))
            end

            if memory_preallocation_time > 0.0
                pct = format_pct(memory_preallocation_time, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Memory Preallocation",
                    "-",
                    format_time(memory_preallocation_time),
                    "-",
                    pct))
            end

            if obbt_time > 0.0
                pct = format_pct(obbt_time, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ OBBT",
                    "-",
                    format_time(obbt_time),
                    "-",
                    pct))
            end

            if preprocessing_time > 0.0
                pct = format_pct(preprocessing_time, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  └─ Preprocessing",
                    "-",
                    format_time(preprocessing_time),
                    "-",
                    pct))
            end

            println("  │")
        end

        # Restricted DD summary
        if ts.restricted_dd_total > 0.0
            pct = format_pct(ts.restricted_dd_total, grand_total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Restricted DD",
                ts.restricted_dd_count,
                format_time(ts.restricted_dd_total),
                format_time(ts.restricted_dd_total / ts.restricted_dd_count),
                pct))

            # Only show breakdown if there are LP calls
            if ts.restricted_dd_lp_solver_call_total > 0.0
                # Individual LP Calls
                pct = format_pct(ts.restricted_dd_lp_solver_call_total, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Individual LP Calls",
                    ts.restricted_dd_lp_solver_call_count,
                    format_time(ts.restricted_dd_lp_solver_call_total),
                    format_time(ts.restricted_dd_lp_solver_call_total / ts.restricted_dd_lp_solver_call_count),
                    pct))

                # All Other Work
                other_work = ts.restricted_dd_total - ts.restricted_dd_lp_solver_call_total
                pct = format_pct(other_work, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  └─ All Other Work",
                    "-",
                    format_time(other_work),
                    "-",
                    pct))
            end
            println("  │")
        end

        # Relaxed DD summary
        if ts.relaxed_dd_total > 0.0
            pct = format_pct(ts.relaxed_dd_total, grand_total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Relaxed DD",
                ts.relaxed_dd_count,
                format_time(ts.relaxed_dd_total),
                format_time(ts.relaxed_dd_total / ts.relaxed_dd_count),
                pct))

            # Only show breakdown if there are LP calls
            if ts.post_relaxed_dd_true_LP_total > 0.0
                # Individual LP Calls (using true_LP)
                pct = format_pct(ts.post_relaxed_dd_true_LP_total, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Individual LP Calls",
                    ts.post_relaxed_dd_true_LP_count,
                    format_time(ts.post_relaxed_dd_true_LP_total),
                    format_time(ts.post_relaxed_dd_true_LP_total / ts.post_relaxed_dd_true_LP_count),
                    pct))

                # All Other Work
                other_work = ts.relaxed_dd_total - ts.post_relaxed_dd_true_LP_total
                pct = format_pct(other_work, grand_total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  └─ All Other Work",
                    "-",
                    format_time(other_work),
                    "-",
                    pct))
            end
            println("  │")
        end

        # All Other Work at top level (in case there's any time unaccounted for)
        # Should only show truly unaccounted time (not setup time, which is already displayed)
        top_level_other = grand_total_time - setup_time - ts.restricted_dd_total - ts.relaxed_dd_total
        if top_level_other > 0.0
            pct = format_pct(top_level_other, grand_total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  └─ All Other Work",
                "-",
                format_time(top_level_other),
                "-",
                pct))
        else
            println("  └─")
        end

        println("\n")
    end

    # Simplex Iterations
    if ts.simplex_iterations > 0
        println("Simplex Iterations: ", ts.simplex_iterations)
        println("\n")
    end

    # Memory Usage
    if ts.workspace_memory_bytes > 0
        println("Workspace Memory: ", format_memory(ts.workspace_memory_bytes))
        println("\n")
    end

    # Level 1: Restricted DD (top level)
    if ts.restricted_dd_count > 0
        total_time = ts.restricted_dd_total
        component_time = 0.0
        avg_time = total_time / ts.restricted_dd_count
        println(Printf.format(Printf.Format(col_spacing),
            "Restricted DD",
            ts.restricted_dd_count,
            format_time(total_time),
            format_time(avg_time),
            "100.0%"))

        println("  │")

        # Level 2: Create Restricted DD
        if ts.create_restricted_dd_count > 0
            sub_total_time = ts.create_restricted_dd_total
            component_time += sub_total_time
            avg_time = sub_total_time / ts.create_restricted_dd_count
            pct = format_pct(sub_total_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Create Restricted DD",
                ts.create_restricted_dd_count,
                format_time(sub_total_time),
                format_time(avg_time),
                pct))

            sub_component_time = 0.0
            # Level 3: Implied Column Bounds
            if ts.restricted_dd_implied_column_bounds_count > 0
                sub_sub_total_time = ts.restricted_dd_implied_column_bounds_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.restricted_dd_implied_column_bounds_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Implied Column Bounds",
                    ts.restricted_dd_implied_column_bounds_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            # Level 3: Histogram Approximation
            if ts.restricted_dd_histogram_approximation_count > 0
                sub_sub_total_time = ts.restricted_dd_histogram_approximation_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.restricted_dd_histogram_approximation_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Histogram Approximation",
                    ts.restricted_dd_histogram_approximation_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            # Level 3: Build Layer
            if ts.restricted_dd_build_layer_count > 0
                sub_sub_total_time = ts.restricted_dd_build_layer_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.restricted_dd_build_layer_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Build Layer",
                    ts.restricted_dd_build_layer_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            other_time = sub_total_time - sub_component_time
            pct = format_pct(other_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  │  └─ All Other Work",
                "-",
                format_time(other_time),
                "-",
                pct)
            )
            println("  │")
        end

        # Level 2: Post Restricted DD (LP solving)
        if ts.post_restricted_dd_count > 0
            sub_total_time = ts.post_restricted_dd_total
            component_time += sub_total_time
            avg_time = sub_total_time / ts.post_restricted_dd_count
            pct = format_pct(sub_total_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Post Process Restricted DD",
                ts.post_restricted_dd_count,
                format_time(sub_total_time),
                format_time(avg_time),
                pct))

            sub_component_time = 0.0
            # Level 3: Individual LP Solver Calls
            if ts.restricted_dd_lp_solver_call_count > 0
                sub_sub_total_time = ts.restricted_dd_lp_solver_call_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.restricted_dd_lp_solver_call_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Individual LP Calls",
                    ts.restricted_dd_lp_solver_call_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            other_time = sub_total_time - sub_component_time
            pct = format_pct(other_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  │  └─ All Other Work",
                "-",
                format_time(other_time),
                "-",
                pct)
            )
            println("  │")
        end

        other_time = total_time - component_time
        pct = format_pct(other_time, total_time)
        println(Printf.format(Printf.Format(col_spacing),
            "  └─ All Other Work",
            "-",
            format_time(other_time),
            "-",
            pct)
        )

        println("\n")
    end

    

     # Level 1: Relaxed DD (top level)
    if ts.relaxed_dd_count > 0
        total_time = ts.relaxed_dd_total
        component_time = 0.0
        avg_time = total_time / ts.relaxed_dd_count
        println(Printf.format(Printf.Format(col_spacing),
            "Relaxed DD",
            ts.relaxed_dd_count,
            format_time(total_time),
            format_time(avg_time),
            "100.0%"))

        println("  │")

        # Level 2: Create relaxed DD
        if ts.create_relaxed_dd_count > 0
            sub_total_time = ts.create_relaxed_dd_total
            component_time += sub_total_time
            avg_time = sub_total_time / ts.create_relaxed_dd_count
            pct = format_pct(sub_total_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Create Relaxed DD",
                ts.create_relaxed_dd_count,
                format_time(sub_total_time),
                format_time(avg_time),
                pct))

            println("  │  │")

            sub_component_time = 0.0

            # Level 3: Implied Column Bounds
            if ts.relaxed_dd_implied_column_bounds_count > 0
                sub_sub_total_time = ts.relaxed_dd_implied_column_bounds_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.relaxed_dd_implied_column_bounds_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Implied Column Bounds",
                    ts.relaxed_dd_implied_column_bounds_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            # Level 3: Invert Bounds
            if ts.relaxed_dd_invert_bounds_count > 0
                sub_sub_total_time = ts.relaxed_dd_invert_bounds_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.relaxed_dd_invert_bounds_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Invert Bounds",
                    ts.relaxed_dd_invert_bounds_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            println("  │  │")

            # Level 3: Split Nodes
            if ts.relaxed_dd_split_nodes_count > 0
                sub_sub_total_time = ts.relaxed_dd_split_nodes_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.relaxed_dd_split_nodes_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Split Nodes",
                    ts.relaxed_dd_split_nodes_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))

                sub_sub_component_time = 0.0
                # Level 4: Bin Counter
                if ts.relaxed_dd_bin_counter_count > 0
                    sub_sub_sub_total_time = ts.relaxed_dd_bin_counter_total
                    sub_sub_component_time += sub_sub_sub_total_time
                    avg_time = sub_sub_sub_total_time / ts.relaxed_dd_bin_counter_count
                    pct = format_pct(sub_sub_sub_total_time, total_time)
                    println(Printf.format(Printf.Format(col_spacing),
                        "  │  │  ├─ Bin Counter",
                        ts.relaxed_dd_bin_counter_count,
                        format_time(sub_sub_sub_total_time),
                        format_time(avg_time),
                        pct))
                end

                # Level 4: Layer Construction
                if ts.relaxed_dd_layer_construction_count > 0
                    sub_sub_sub_total_time = ts.relaxed_dd_layer_construction_total
                    sub_sub_component_time += sub_sub_sub_total_time
                    avg_time = sub_sub_sub_total_time / ts.relaxed_dd_layer_construction_count
                    pct = format_pct(sub_sub_sub_total_time, total_time)
                    println(Printf.format(Printf.Format(col_spacing),
                        "  │  │  ├─ Layer Construction",
                        ts.relaxed_dd_layer_construction_count,
                        format_time(sub_sub_sub_total_time),
                        format_time(avg_time),
                        pct))
                end

                other_time_l4 = sub_sub_total_time - sub_sub_component_time
                pct = format_pct(other_time_l4, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  │  └─ All Other Work",
                    "-",
                    format_time(other_time_l4),
                    "-",
                    pct))
            end

            println("  │  │")

            # Level 3: Update LTR
            if ts.relaxed_dd_update_ltr_count > 0
                sub_sub_total_time = ts.relaxed_dd_update_ltr_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.relaxed_dd_update_ltr_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Update LTR",
                    ts.relaxed_dd_update_ltr_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            # Level 3: Rough Bounding
            if ts.relaxed_dd_rough_bounding_count > 0
                sub_sub_total_time = ts.relaxed_dd_rough_bounding_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.relaxed_dd_rough_bounding_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Rough Bounding",
                    ts.relaxed_dd_rough_bounding_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            # Level 3: Infimum Matrix
            if ts.relaxed_dd_infimum_matrix_cubic_count > 0
                sub_sub_total_time = ts.relaxed_dd_infimum_matrix_cubic_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.relaxed_dd_infimum_matrix_cubic_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Infimum Matrix",
                    ts.relaxed_dd_infimum_matrix_cubic_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            other_time = sub_total_time - sub_component_time
            pct = format_pct(other_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  │  └─ All Other Work",
                "-",
                format_time(other_time),
                "-",
                pct)
            )
            println("  │")
        end


        # Level 2: post process relaxed DD
        if ts.post_relaxed_dd_count > 0
            sub_total_time = ts.post_relaxed_dd_total
            component_time += sub_total_time
            avg_time = sub_total_time / ts.post_relaxed_dd_count
            pct = format_pct(sub_total_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  ├─ Post Process Relaxed DD",
                ts.post_relaxed_dd_count,
                format_time(sub_total_time),
                format_time(avg_time),
                pct))

            println("  │  │")

            sub_component_time = 0.0

            # Level 3: Potential Top Level LP (skipping true_LP_count)
            if ts.post_relaxed_dd_potential_top_level_LP_count > 0
                sub_sub_total_time = ts.post_relaxed_dd_potential_top_level_LP_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.post_relaxed_dd_potential_top_level_LP_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Potential Top Level LP",
                    ts.post_relaxed_dd_potential_top_level_LP_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            # Level 3: Update LTT
            if ts.post_relaxed_dd_update_ltt_count > 0
                sub_sub_total_time = ts.post_relaxed_dd_update_ltt_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.post_relaxed_dd_update_ltt_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Update LTT",
                    ts.post_relaxed_dd_update_ltt_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))
            end

            println("  │  │")

            # Level 3: Exact Cutset
            if ts.post_relaxed_dd_exact_cutset_count > 0
                sub_sub_total_time = ts.post_relaxed_dd_exact_cutset_total
                sub_component_time += sub_sub_total_time
                avg_time = sub_sub_total_time / ts.post_relaxed_dd_exact_cutset_count
                pct = format_pct(sub_sub_total_time, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  ├─ Exact Cutset",
                    ts.post_relaxed_dd_exact_cutset_count,
                    format_time(sub_sub_total_time),
                    format_time(avg_time),
                    pct))

                sub_sub_component_time = 0.0
                # Level 4: Handle Exact Terminals
                if ts.post_relaxed_dd_handle_exact_terminals_count > 0
                    sub_sub_sub_total_time = ts.post_relaxed_dd_handle_exact_terminals_total
                    sub_sub_component_time += sub_sub_sub_total_time
                    avg_time = sub_sub_sub_total_time / ts.post_relaxed_dd_handle_exact_terminals_count
                    pct = format_pct(sub_sub_sub_total_time, total_time)
                    println(Printf.format(Printf.Format(col_spacing),
                        "  │  │  ├─ Handle Exact Terminals",
                        ts.post_relaxed_dd_handle_exact_terminals_count,
                        format_time(sub_sub_sub_total_time),
                        format_time(avg_time),
                        pct))

                    sub_sub_sub_component_time = 0.0
                    # Level 5: Potential Low Level LP
                    if ts.post_relaxed_dd_potential_low_level_LP_count > 0
                        sub_sub_sub_sub_total_time = ts.post_relaxed_dd_potential_low_level_LP_total
                        sub_sub_sub_component_time += sub_sub_sub_sub_total_time
                        avg_time = sub_sub_sub_sub_total_time / ts.post_relaxed_dd_potential_low_level_LP_count
                        pct = format_pct(sub_sub_sub_sub_total_time, total_time)
                        println(Printf.format(Printf.Format(col_spacing),
                            "  │  │  │  ├─ Potential Low Level LP",
                            ts.post_relaxed_dd_potential_low_level_LP_count,
                            format_time(sub_sub_sub_sub_total_time),
                            format_time(avg_time),
                            pct))
                    end

                    other_time_l5 = sub_sub_sub_total_time - sub_sub_sub_component_time
                    pct = format_pct(other_time_l5, total_time)
                    println(Printf.format(Printf.Format(col_spacing),
                        "  │  │  │  └─ All Other Work",
                        "-",
                        format_time(other_time_l5),
                        "-",
                        pct))
                end

                # Level 4: Handle compute tighter bounds
                if ts.post_relaxed_dd_compute_tighter_bound_count > 0
                    println("  │  │  │")
                    sub_sub_sub_total_time = ts.post_relaxed_dd_compute_tighter_bound_total
                    sub_sub_component_time += sub_sub_sub_total_time
                    avg_time = sub_sub_sub_total_time / ts.post_relaxed_dd_compute_tighter_bound_count
                    pct = format_pct(sub_sub_sub_total_time, total_time)
                    println(Printf.format(Printf.Format(col_spacing),
                        "  │  │  ├─ Tighten Continuous Bounds",
                        ts.post_relaxed_dd_compute_tighter_bound_count,
                        format_time(sub_sub_sub_total_time),
                        format_time(avg_time),
                        pct))
                end

                other_time_l4 = sub_sub_total_time - sub_sub_component_time
                pct = format_pct(other_time_l4, total_time)
                println(Printf.format(Printf.Format(col_spacing),
                    "  │  │  └─ All Other Work",
                    "-",
                    format_time(other_time_l4),
                    "-",
                    pct))
            end

            println("  │  │")

            other_time = sub_total_time - sub_component_time
            pct = format_pct(other_time, total_time)
            println(Printf.format(Printf.Format(col_spacing),
                "  │  └─ All Other Work",
                "-",
                format_time(other_time),
                "-",
                pct)
            )
            println("  │")
        end

        other_time = total_time - component_time
        pct = format_pct(other_time, total_time)
        println(Printf.format(Printf.Format(col_spacing),
            "  └─ All Other Work",
            "-",
            format_time(other_time),
            "-",
            pct)
        )

        println("\n")
    end

    # Worker contributions (if parallel processing was used)
    if worker_contributions !== nothing && !isempty(worker_contributions)
        println("Worker Contributions (sorted smallest to largest):")
        println("-"^line_chars)

        # Add setup time to worker 1 (master thread)
        worker_contributions[1] += setup_time

        # Calculate total time for percentages
        total_contribution = sum(worker_contributions)

        # Fixed column width for alignment
        col_width = 15
        label_width = 18  # Width for right-aligned labels

        # Build time row
        time_row = Printf.format(Printf.Format("%$(label_width)s"), "Work Time: ")
        for (idx, contrib) in enumerate(worker_contributions)
            time_str = format_time(contrib)
            time_row *= Printf.format(Printf.Format("%$(col_width)s"), time_str)
            if idx < length(worker_contributions)
                time_row *= " | "
            end
        end
        println(time_row)

        # Build percentage row
        pct_row = Printf.format(Printf.Format("%$(label_width)s"), "% Work Time: ")
        for (idx, contrib) in enumerate(worker_contributions)
            pct_str = string(format_pct(contrib, total_contribution)," ")
            pct_row *= Printf.format(Printf.Format("%$(col_width)s"), pct_str)
            if idx < length(worker_contributions)
                pct_row *= " | "
            end
        end
        println(pct_row)
    end

    # Worker idle times (if parallel processing was used)
    if worker_idle_times !== nothing && !isempty(worker_idle_times)
        println()

        # Fixed column width for alignment
        col_width = 15
        label_width = 18  # Width for right-aligned labels (to fit "Total Clock Time:")

        # Build total clock time row (same for all workers)
        clock_row = Printf.format(Printf.Format("%$(label_width)s"), "Total Clock Time: ")
        clock_row *= format_time(clock_time)
        println(clock_row)
        println()

        # Build idle times row
        idle_row = Printf.format(Printf.Format("%$(label_width)s"), "Idle Time: ")
        for (idx, idle) in enumerate(worker_idle_times)
            time_str = format_time(idle)
            idle_row *= Printf.format(Printf.Format("%$(col_width)s"), time_str)
            if idx < length(worker_idle_times)
                idle_row *= " | "
            end
        end
        println(idle_row)

        # Build percentage of clock time row
        clock_pct_row = Printf.format(Printf.Format("%$(label_width)s"), "% Clock Time: ")
        for (idx, idle) in enumerate(worker_idle_times)
            pct_str = if clock_time > 0.0
                string(format_pct(idle, clock_time), " ")
            else
                "  -%   "
            end
            clock_pct_row *= Printf.format(Printf.Format("%$(col_width)s"), pct_str)
            if idx < length(worker_idle_times)
                clock_pct_row *= " | "
            end
        end
        println(clock_pct_row)
    end

    println("="^line_chars * "\n")

    # Store parallel processing data if available
    if worker_contributions !== nothing && !isempty(worker_contributions)
        timing_data["parallel"] = Dict(
            "worker_contributions" => collect(worker_contributions),
            "worker_idle_times" => worker_idle_times !== nothing ? collect(worker_idle_times) : nothing,
            "clock_time" => clock_time
        )
    else
        timing_data["parallel"] = nothing
    end

    # Write timing data to JSON file if requested
    # Note: JSON.jl automatically converts Inf/NaN to null
    if !isnothing(timing_log_file)
        open(timing_log_file, "a") do io  # Append mode
            println(io, JSON.json(sanitize_for_json(timing_data)))
        end
    end
end

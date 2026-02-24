using PrecompileTools

# =============================================================================
# Automatic precompilation at package install time
# =============================================================================

@setup_workload begin
    @compile_workload begin
        # Precompile all combinations: (Bool, Int8, Int16) × (Float32, Float64)
        # Uses _force_parallel_precompile to compile parallel code path (which covers all subroutines)
        for precision in (Float32, Float64)
            # Bool path (binary variables)
            model = generate_random_mip(
                n_int=2, n_cont=1, m=1,
                density=0.5, cont_domain_size=1.0,
                int_lb_range=(0, 0), int_ub_range=(1, 1),
                seed=2
            )
            solve_mip(model;
                relaxed_w=2, restricted_w=2,
                suppress_all_prints=true,
                numerical_precision=precision,
                _force_parallel_precompile=true,
            )

            # Int8 path
            model = generate_random_mip(
                n_int=2, n_cont=1, m=1,
                density=0.2, cont_domain_size=1.0,
                int_lb_range=(0, 0), int_ub_range=(2, 2),
                seed=2
            )
            solve_mip(model;
                relaxed_w=3, restricted_w=3,
                suppress_all_prints=true,
                numerical_precision=precision,
                _force_parallel_precompile=true,
            )

            # Int16 path
            model = generate_random_mip(
                n_int=3, n_cont=60, m=1,
                density=1.0, cont_domain_size=1000.0,
                int_lb_range=(0, 0), int_ub_range=(400, 400),
                seed=3
            )
            solve_mip(model;
                relaxed_w=129, restricted_w=129,
                suppress_all_prints=true,
                numerical_precision=precision,
                _force_parallel_precompile=true
            )
        end
    end
end


# =============================================================================
# Manual precompilation function (existing)
# =============================================================================
"""
    precompile_solver(; precision::DataType = Float32, biggest_int_domain::Int = 2)

Manually precompile the solver for a specific integer type and precision.

# When You Need This

Automatic precompilation at package load time already compiles Bool, Int8, and Int16 paths
for both Float32 and Float64. You only need this function if you want to use a precision
type other than Float32 or Float64.

# Arguments

- `precision::DataType = Float32`: Numerical precision for continuous variables and objective values

- `biggest_int_domain::Int = 2`: Size of the largest integer variable domain you expect
  - **biggest_int_domain == 2**: Bool/BitVector path (binary variables {0, 1})
  - **biggest_int_domain in [3, 127]**: Int8 path
  - **biggest_int_domain in [128, 32767]**: Int16 path
  - **biggest_int_domain ≥ 32768**: Warning (Int32/Int64 not supported by this function)

Compiles the parallel code path if multiple threads are available, otherwise the serial path.
"""
function precompile_solver(;precision::DataType = Float32, biggest_int_domain::Int = 2)
    if biggest_int_domain == 2
        model = generate_random_mip(
            n_int = 2, n_cont = 1, m = 1,
            density = .5, cont_domain_size = 1.0,
            int_lb_range = (0, 0), int_ub_range = (1, 1),
            seed = 2
        )
        solve_mip(model,
            relaxed_w = 2, restricted_w = 2,
            parallel_processing = true, solution_print = false,
            bounds_print = false, suppress_all_prints = true, numerical_precision = precision,
        )
    elseif biggest_int_domain < 128
        model = generate_random_mip(
            n_int = 2, n_cont = 1, m = 1,
            density = .2, cont_domain_size = 1.0,
            int_lb_range = (0, 0), int_ub_range = (2, 2),
            seed = 2
        )

        solve_mip(model,
            relaxed_w = 3, restricted_w = 3,
            parallel_processing = true, solution_print = false,
            bounds_print = false, suppress_all_prints = true,  numerical_precision = precision,
        )
    elseif biggest_int_domain < 32768
        # Compile Int16
        model = generate_random_mip(
            n_int = 3, n_cont = 60, m = 1,
            density = 1.0, cont_domain_size = 1000.0,
            int_lb_range = (0, 0), int_ub_range = (400, 400),
            seed = 3
        )
        solve_mip(model,
            relaxed_w = 129, restricted_w = 129,
            parallel_processing = true, solution_print = false,
            bounds_print = false, suppress_all_prints = true,  numerical_precision = precision,
        )
    else
        println("WARNING: Integer domains larger than 32767 (Int16) require the user to precompile the package.")
    end
end
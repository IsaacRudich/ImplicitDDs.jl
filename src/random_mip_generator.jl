"""
Random MIP Generator for ImplicitDDs Testing

This module provides comprehensive random MIP generation with fine-grained control
over problem characteristics including size, sparsity, structure, and numerical properties.
"""


"""
    generate_random_mip(;
        n_int::Int,
        n_cont::Int,
        m::Int,
        density::Float64,
        kwargs...
    )

Generate a random mixed-integer programming problem with extensive control over problem characteristics.

# Required Arguments
- `n_int::Int`: Number of integer variables
- `n_cont::Int`: Number of continuous variables
- `m::Int`: Number of constraints
- `density::Float64`: Constraint matrix density (0.0 to 1.0)

# Variable Domains and Bounds
- `int_domain_size::Int = 10`: Domain size for integer variables (max - min)
- `cont_domain_size::Float64 = 10.0`: Domain size for continuous variables
- `int_lb_range::Tuple{Int, Int} = (0, int_domain_size)`: Range for integer lower bounds
- `int_ub_range::Tuple{Int, Int} = (0, int_domain_size)`: Range for integer upper bounds
- `cont_lb_range::Tuple{Float64, Float64} = (0.0, cont_domain_size)`: Range for continuous lower bounds
- `cont_ub_range::Tuple{Float64, Float64} = (0.0, cont_domain_size)`: Range for continuous upper bounds

# Objective Function
- `obj_density::Float64 = 1.0`: Objective sparsity (fraction of vars with non-zero cost)
- `obj_coeff_range::Tuple{Float64, Float64} = (-100.0, 100.0)`: Range for objective coefficients
- `obj_coeff_dist::Symbol = :uniform`: Distribution (:uniform, :normal, :exponential)

# Constraint Properties
- `rhs_range::Tuple{Float64, Float64} = (-100.0, 100.0)`: Range for RHS values

# Coefficient Distribution
- `coeff_distribution::Symbol = :uniform`: How to generate coefficients
  - `:uniform`: Uniform random in coeff_range
  - `:binary`: Coefficients are 0 or 1
  - `:small_int`: Small integers (1, 2, 3, 4, 5)
  - `:normal`: Normally distributed (truncated to coeff_range)
- `coeff_range::Tuple{Float64, Float64} = (-10.0, 10.0)`: Range for non-zero coefficients

# Numerical Properties
- `integer_coefficients::Bool = false`: Force all coefficients to be integers
- `precision::Int = 2`: Decimal places for floating-point coefficients
- `coefficient_scale::Float64 = 1.0`: Overall scale factor

# Problem Structure
- `block_structure::Bool = false`: Create block diagonal structure
- `num_blocks::Int = 1`: Number of independent blocks (if block_structure=true)

# Other
- `seed::Union{Int, Nothing} = nothing`: Random seed for reproducibility

# Returns
- `JuMP.Model`: A JuMP model with the randomly generated MIP

# Examples
```julia
# Simple binary knapsack-like problem
model = generate_random_mip(
    n_int = 20,
    n_cont = 0,
    m = 5,
    density = 0.3,
    int_domain_size = 1,  # binary
    coeff_distribution = :small_int
)

# Large sparse problem with continuous variables
model = generate_random_mip(
    n_int = 100,
    n_cont = 50,
    m = 80,
    density = 0.1,
    int_domain_size = 10,
    cont_domain_size = 100.0
)

# Block structured problem
model = generate_random_mip(
    n_int = 60,
    n_cont = 30,
    m = 40,
    density = 0.2,
    block_structure = true,
    num_blocks = 3
)
```
"""
function generate_random_mip(;
    # Required parameters
    n_int::Int,
    n_cont::Int,
    m::Int,
    density::Float64,

    # Variable domains/bounds
    int_domain_size::Int = 10,
    cont_domain_size::Float64 = 10.0,
    int_lb_range::Union{Tuple{Int, Int}, Nothing} = nothing,
    int_ub_range::Union{Tuple{Int, Int}, Nothing} = nothing,
    cont_lb_range::Union{Tuple{Float64, Float64}, Nothing} = nothing,
    cont_ub_range::Union{Tuple{Float64, Float64}, Nothing} = nothing,

    # Objective function
    obj_density::Float64 = 1.0,
    obj_coeff_range::Tuple{Float64, Float64} = (-100.0, 100.0),
    obj_coeff_dist::Symbol = :uniform,

    # Constraint properties
    rhs_range::Tuple{Float64, Float64} = (-100.0, 100.0),

    # Coefficient distribution
    coeff_distribution::Symbol = :uniform,
    coeff_range::Tuple{Float64, Float64} = (-10.0, 10.0),

    # Numerical properties
    integer_coefficients::Bool = false,
    precision::Int = 2,
    coefficient_scale::Float64 = 1.0,

    # Problem structure
    block_structure::Bool = false,
    num_blocks::Int = 1,

    # Reproducibility
    seed::Union{Int, Nothing} = nothing
)
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Input validation
    @assert 0.0 <= density <= 1.0 "density must be in [0, 1]"
    @assert 0.0 <= obj_density <= 1.0 "obj_density must be in [0, 1]"
    @assert n_int >= 0 && n_cont >= 0 "n_int and n_cont must be non-negative"
    @assert m >= 0 "m must be non-negative"
    @assert coefficient_scale > 0 "coefficient_scale must be positive"

    # Set default ranges if not provided
    int_lb_range_actual = isnothing(int_lb_range) ? (0, int_domain_size-1) : int_lb_range
    int_ub_range_actual = isnothing(int_ub_range) ? (0, int_domain_size-1) : int_ub_range
    cont_lb_range_actual = isnothing(cont_lb_range) ? (0.0, cont_domain_size) : cont_lb_range
    cont_ub_range_actual = isnothing(cont_ub_range) ? (0.0, cont_domain_size) : cont_ub_range

    n_vars = n_int + n_cont

    # Generate variable bounds
    int_lbs = rand(int_lb_range_actual[1]:int_lb_range_actual[2], n_int)
    int_ubs = rand(int_ub_range_actual[1]:int_ub_range_actual[2], n_int)

    # Ensure lb <= ub for integer variables
    for i in 1:n_int
        if int_lbs[i] > int_ubs[i]
            int_lbs[i], int_ubs[i] = int_ubs[i], int_lbs[i]
        end
    end

    cont_lbs = cont_lb_range_actual[1] .+ rand(n_cont) .* (cont_lb_range_actual[2] - cont_lb_range_actual[1])
    cont_ubs = cont_ub_range_actual[1] .+ rand(n_cont) .* (cont_ub_range_actual[2] - cont_ub_range_actual[1])

    # Ensure lb <= ub for continuous variables
    for i in 1:n_cont
        if cont_lbs[i] > cont_ubs[i]
            cont_lbs[i], cont_ubs[i] = cont_ubs[i], cont_lbs[i]
        end
    end

    # Generate constraint matrix
    if block_structure && num_blocks > 1
        A = generate_block_structure(n_int, n_cont, m, density, num_blocks, coeff_distribution, coeff_range)
    else
        A = generate_sparse_matrix(n_vars, m, density, coeff_distribution, coeff_range)
    end

    # Apply numerical properties
    if integer_coefficients
        A = round.(A)
    elseif precision >= 0
        A = round.(A, digits=precision)
    end
    A = A .* coefficient_scale

    # Generate RHS
    b = rhs_range[1] .+ rand(m) .* (rhs_range[2] - rhs_range[1])
    if integer_coefficients
        b = round.(b)
    elseif precision >= 0
        b = round.(b, digits=precision)
    end
    b = b .* coefficient_scale

    # Generate objective coefficients
    c = generate_objective(n_vars, obj_density, obj_coeff_range, obj_coeff_dist)
    if integer_coefficients
        c = round.(c)
    elseif precision >= 0
        c = round.(c, digits=precision)
    end
    c = c .* coefficient_scale

    # Build JuMP model
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Add variables
    @variable(model, int_lbs[i] <= x_int[i=1:n_int] <= int_ubs[i], Int)
    if n_cont > 0
        @variable(model, cont_lbs[i] <= x_cont[i=1:n_cont] <= cont_ubs[i])
    end

    # Combine variable references
    x_all = n_cont > 0 ? vcat(x_int, x_cont) : x_int

    # Add constraints (all <=)
    for i in 1:m
        row_indices = findnz(A[i, :])[1]
        if !isempty(row_indices)
            @constraint(model, sum(A[i, j] * x_all[j] for j in row_indices) <= b[i])
        end
    end

    # Set objective (always minimize)
    @objective(model, Min, sum(c[i] * x_all[i] for i in 1:n_vars))

    return model
end


"""
    generate_sparse_matrix(n_cols::Int, n_rows::Int, density::Float64, distribution::Symbol, coeff_range::Tuple{Float64, Float64})

Generate a sparse matrix with specified density and coefficient distribution.
"""
function generate_sparse_matrix(n_cols::Int, n_rows::Int, density::Float64, distribution::Symbol, coeff_range::Tuple{Float64, Float64})
    n_nonzeros = round(Int, n_rows * n_cols * density)

    # Generate random nonzero positions
    rows = Int[]
    cols = Int[]

    while length(rows) < n_nonzeros
        r = rand(1:n_rows)
        c = rand(1:n_cols)
        if !((r, c) in zip(rows, cols))  # avoid duplicates
            push!(rows, r)
            push!(cols, c)
        end
    end

    # Generate values according to distribution
    vals = generate_coefficients(n_nonzeros, distribution, coeff_range)

    return sparse(rows, cols, vals, n_rows, n_cols)
end


"""
    generate_coefficients(n::Int, distribution::Symbol, coeff_range::Tuple{Float64, Float64})

Generate coefficients according to specified distribution.
"""
function generate_coefficients(n::Int, distribution::Symbol, coeff_range::Tuple{Float64, Float64})
    if distribution == :uniform
        return coeff_range[1] .+ rand(n) .* (coeff_range[2] - coeff_range[1])
    elseif distribution == :binary
        return float.(rand([0, 1], n))
    elseif distribution == :small_int
        return float.(rand(1:5, n))
    elseif distribution == :normal
        μ = (coeff_range[1] + coeff_range[2]) / 2
        σ = (coeff_range[2] - coeff_range[1]) / 6  # 3σ rule
        vals = μ .+ σ .* randn(n)
        return clamp.(vals, coeff_range[1], coeff_range[2])
    else
        error("Unknown distribution: $distribution")
    end
end


"""
    generate_objective(n::Int, obj_density::Float64, obj_coeff_range::Tuple{Float64, Float64}, obj_coeff_dist::Symbol)

Generate objective function coefficients.
"""
function generate_objective(n::Int, obj_density::Float64, obj_coeff_range::Tuple{Float64, Float64}, obj_coeff_dist::Symbol)
    c = zeros(n)
    n_nonzero = round(Int, n * obj_density)
    nonzero_indices = randperm(n)[1:n_nonzero]

    if obj_coeff_dist == :uniform
        c[nonzero_indices] = obj_coeff_range[1] .+ rand(n_nonzero) .* (obj_coeff_range[2] - obj_coeff_range[1])
    elseif obj_coeff_dist == :normal
        μ = (obj_coeff_range[1] + obj_coeff_range[2]) / 2
        σ = (obj_coeff_range[2] - obj_coeff_range[1]) / 6
        c[nonzero_indices] = μ .+ σ .* randn(n_nonzero)
        c[nonzero_indices] = clamp.(c[nonzero_indices], obj_coeff_range[1], obj_coeff_range[2])
    elseif obj_coeff_dist == :exponential
        # Exponentially distributed positive coefficients
        c[nonzero_indices] = rand(Exponential(1.0), n_nonzero)
        # Scale to range
        c[nonzero_indices] = c[nonzero_indices] ./ maximum(c[nonzero_indices]) .* (obj_coeff_range[2] - obj_coeff_range[1]) .+ obj_coeff_range[1]
    else
        error("Unknown objective distribution: $obj_coeff_dist")
    end

    return c
end


"""
    generate_block_structure(n_int::Int, n_cont::Int, m::Int, density::Float64, num_blocks::Int, distribution::Symbol, coeff_range::Tuple{Float64, Float64})

Generate a constraint matrix with block diagonal structure.
"""
function generate_block_structure(n_int::Int, n_cont::Int, m::Int, density::Float64, num_blocks::Int, distribution::Symbol, coeff_range::Tuple{Float64, Float64})
    n_vars = n_int + n_cont

    # Allocate variables and constraints to blocks
    vars_per_block = div(n_vars, num_blocks)
    cons_per_block = div(m, num_blocks)

    # Start with zeros
    A = spzeros(m, n_vars)

    # Fill in blocks
    for b in 1:num_blocks
        var_start = (b - 1) * vars_per_block + 1
        var_end = b == num_blocks ? n_vars : b * vars_per_block

        con_start = (b - 1) * cons_per_block + 1
        con_end = b == num_blocks ? m : b * cons_per_block

        n_vars_block = var_end - var_start + 1
        n_cons_block = con_end - con_start + 1

        if n_cons_block > 0 && n_vars_block > 0
            block = generate_sparse_matrix(n_vars_block, n_cons_block, density, distribution, coeff_range)
            A[con_start:con_end, var_start:var_end] = block
        end
    end

    return A
end

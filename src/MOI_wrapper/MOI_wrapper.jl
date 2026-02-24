"""
    ImplicitDDs MOI Wrapper

This module provides a MathOptInterface wrapper for the ImplicitDDs solver,
enabling use with JuMP's standard solver interface:

```julia
using JuMP, ImplicitDDs
model = Model(ImplicitDDs.Optimizer)
set_optimizer_attribute(model, "relaxed_w", 10000)
@variable(model, x[1:3], Bin)
@constraint(model, sum(x) <= 2)
@objective(model, Max, x[1] + 2x[2] + 3x[3])
optimize!(model)
```
"""

import MathOptInterface as MOI

"""
    Optimizer <: MOI.AbstractOptimizer

ImplicitDDs solver optimizer for MathOptInterface.

# Supported Parameters (via `set_optimizer_attribute`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relaxed_w` | Int | 10000 | Width limit for relaxed DDs |
| `restricted_w` | Int | 10000 | Width limit for restricted DDs |
| `num_LPs_to_run` | Int | 10 | Max LP subproblems per restricted DD |
| `parallel_processing` | Bool | true | Enable multi-threading |
| `numerical_precision` | DataType | Float32 | Float precision |
| `debug_mode` | Bool | false | Verbose debug output |
| `log_file_path` | String/Nothing | nothing | JSON log file path |
| `bounds_print` | Bool | true | Print bound improvements |
| `solution_print` | Bool | false | Print full solutions |
| `wait_to_write_solutions` | Bool | false | Batch log writes |
| `timer_outputs` | Bool | false | Detailed timing breakdown |
| `custom_variable_order` | Vector{MOI.VariableIndex}/Nothing | nothing | Custom variable ordering |
| `MOI.Silent()` | Bool | false | Suppress all output |
| `MOI.TimeLimitSec()` | Float64/Nothing | nothing | Time limit in seconds |

# Example

```julia
using JuMP, ImplicitDDs

model = Model(ImplicitDDs.Optimizer)
set_optimizer_attribute(model, "relaxed_w", 5000)
set_optimizer_attribute(model, MOI.Silent(), true)

@variable(model, x[1:5], Bin)
@constraint(model, sum(x) <= 3)
@objective(model, Max, sum(i * x[i] for i in 1:5))
optimize!(model)

println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
println("Solution: ", value.(x))
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    # =========================================================================
    # Parameters (user-settable, maps to solve_mip arguments)
    # =========================================================================
    relaxed_w::Int
    restricted_w::Int
    num_LPs_to_run::Int
    parallel_processing::Bool
    numerical_precision::DataType
    debug_mode::Bool
    log_file_path::Union{String, Nothing}
    bounds_print::Bool
    solution_print::Bool
    wait_to_write_solutions::Bool
    timer_outputs::Bool
    custom_variable_order::Union{Vector{MOI.VariableIndex}, Nothing}

    # Standard MOI attributes
    time_limit::Union{Float64, Nothing}  # seconds (converted to minutes for solve_mip)
    silent::Bool                          # maps to suppress_all_prints

    # =========================================================================
    # Results (populated after optimize!, user-queryable)
    # =========================================================================
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    objective_value::Float64
    objective_bound::Float64
    solve_time::Float64
    result_count::Int
    relative_gap::Float64
    node_count::Int

    # Solution data
    primal_solution::Union{Vector{Float64}, Nothing}
    variable_map::Union{Dict{MOI.VariableIndex, Int}, Nothing}

    # =========================================================================
    # Internal (transient model data, set by copy_to, used by optimize!)
    # =========================================================================
    src_model::Union{MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}, Nothing}
    src_index_map::Union{MOI.Utilities.IndexMap, Nothing}
end

"""
    Optimizer()

Create a new ImplicitDDs optimizer with default parameters.
"""
function Optimizer()
    return Optimizer(
        # Parameters
        10000,          # relaxed_w
        10000,          # restricted_w
        10,             # num_LPs_to_run
        true,           # parallel_processing
        Float32,        # numerical_precision
        false,          # debug_mode
        nothing,        # log_file_path
        true,           # bounds_print
        false,          # solution_print
        false,          # wait_to_write_solutions
        false,          # timer_outputs
        nothing,        # custom_variable_order
        nothing,        # time_limit
        false,          # silent

        # Results (initial state)
        MOI.OPTIMIZE_NOT_CALLED,  # termination_status
        MOI.NO_SOLUTION,          # primal_status
        NaN,                      # objective_value
        NaN,                      # objective_bound
        0.0,                      # solve_time
        0,                        # result_count
        NaN,                      # relative_gap
        0,                        # node_count
        nothing,                  # primal_solution
        nothing,                  # variable_map
        nothing,                  # src_model
        nothing                   # src_index_map
    )
end

# Include supporting files
include("attributes.jl")
include("constraints.jl")
include("copy_to.jl")
include("optimize.jl")
include("results.jl")

# =============================================================================
# Basic MOI interface
# =============================================================================

function MOI.is_empty(optimizer::Optimizer)
    return optimizer.src_model === nothing &&
           optimizer.variable_map === nothing &&
           optimizer.termination_status == MOI.OPTIMIZE_NOT_CALLED
end

function MOI.empty!(optimizer::Optimizer)
    # Clear results, keep parameters
    optimizer.termination_status = MOI.OPTIMIZE_NOT_CALLED
    optimizer.primal_status = MOI.NO_SOLUTION
    optimizer.objective_value = NaN
    optimizer.objective_bound = NaN
    optimizer.solve_time = 0.0
    optimizer.result_count = 0
    optimizer.relative_gap = NaN
    optimizer.node_count = 0
    optimizer.primal_solution = nothing
    optimizer.variable_map = nothing
    optimizer.src_model = nothing
    optimizer.src_index_map = nothing
    return nothing
end

function MOI.get(::Optimizer, ::MOI.SolverName)
    return "ImplicitDDs"
end

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    # Should match version in Project.toml
    return "0.1.0"
end

function Base.show(io::IO, optimizer::Optimizer)
    print(io, "ImplicitDDs.Optimizer(")
    print(io, "relaxed_w=$(optimizer.relaxed_w), ")
    print(io, "restricted_w=$(optimizer.restricted_w), ")
    print(io, "status=$(optimizer.termination_status)")
    print(io, ")")
end

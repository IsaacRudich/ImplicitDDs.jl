# ImplicitDDs

[![Build Status](https://github.com/IsaacRudich/ImplicitDDs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/IsaacRudich/ImplicitDDs.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Mixed-Integer Programming solver using Decision Diagrams for branch-and-bound.

## When to Use ImplicitDDs

ImplicitDDs is designed for specific problem characteristics:

**Best suited for:**
- Binary integer programs (0-1 variables)
- Problems with weak linear relaxations where traditional LP-based solvers struggle
- Highly combinatorial problems where objective value depends primarily on integer variables
- Problems with small integer domains (precompiled for domains up to Int16, i.e., [-32768, 32767])

**Requirements:**
- All variables must have finite bounds

**Not recommended for:**
- Problems with large integer domains (Int32/Int64)
- Problems where the linear relaxation is tight
- Problems dominated by continuous variable contributions

## Installation

```julia
using Pkg
Pkg.add("ImplicitDDs")
```

## Basic Usage

```julia
using JuMP, ImplicitDDs

model = Model(ImplicitDDs.Optimizer)

@variable(model, x[1:5], Bin)
@constraint(model, sum(x) <= 3)
@objective(model, Max, sum(i * x[i] for i in 1:5))

optimize!(model)

println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
println("Solution: ", value.(x))
```

## Optimizer Attributes

Configure the solver using `set_optimizer_attribute`:

```julia
model = Model(ImplicitDDs.Optimizer)
set_optimizer_attribute(model, "relaxed_w", 5000)
set_optimizer_attribute(model, "restricted_w", 5000)
set_optimizer_attribute(model, MOI.Silent(), true)
```

### Decision Diagram Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relaxed_w` | Int | 10000 | Width limit for relaxed decision diagrams. Larger values give tighter bounds but slower iterations. |
| `restricted_w` | Int | 10000 | Width limit for restricted decision diagrams. Larger values improve solution quality but slow construction. |
| `num_LPs_to_run` | Int | 10 | Maximum LP subproblems to solve per restricted DD. More LPs can find better solutions but significantly slow the restricted DD phase. |

**Tuning tips:**
- Start with default widths (10000) and adjust based on performance
- If bounds aren't tight enough, increase `relaxed_w`
- If finding good solutions is difficult, increase `restricted_w`
- For problems with few continuous variables, reduce `num_LPs_to_run`

### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_processing` | Bool | true | Enable multi-threaded branch-and-bound. Run Julia with `-t N` for N threads. |
| `numerical_precision` | DataType | Float32 | Float precision for calculations. Use Float64 if numerical issues arise. |

### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MOI.Silent()` | Bool | false | Suppress all console output. |
| `bounds_print` | Bool | true | Print lower/upper bounds as they improve. |
| `solution_print` | Bool | false | Print full solution vectors when found. |
| `debug_mode` | Bool | false | Verbose internal solver state (very verbose). |
| `timer_outputs` | Bool | false | Print detailed timing breakdown at end. |

### Logging Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_file_path` | String/Nothing | nothing | Path for JSON log file. |
| `wait_to_write_solutions` | Bool | false | If true, batch log writes for performance. If false, write immediately (safer if solver crashes). |

### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MOI.TimeLimitSec()` | Float64/Nothing | nothing | Time limit in seconds. |
| `custom_variable_order` | Vector{MOI.VariableIndex}/Nothing | nothing | Custom ordering for integer variables. Default uses METIS-based ordering. |

## Example with Configuration

```julia
using JuMP, ImplicitDDs

model = Model(ImplicitDDs.Optimizer)

# Configure solver
set_optimizer_attribute(model, "relaxed_w", 50000)
set_optimizer_attribute(model, "restricted_w", 50000)
set_optimizer_attribute(model, "parallel_processing", true)
set_optimizer_attribute(model, MOI.Silent(), true)
set_optimizer_attribute(model, MOI.TimeLimitSec(), 300.0)  # 5 minute limit

# Define problem
@variable(model, x[1:10], Bin)
@constraint(model, sum(x) <= 5)
@objective(model, Max, sum(i * x[i] for i in 1:10))

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found!")
    println("Objective: ", objective_value(model))
elseif termination_status(model) == MOI.TIME_LIMIT
    println("Time limit reached")
    if has_values(model)
        println("Best solution found: ", objective_value(model))
    end
end
```

## Querying Results

Standard JuMP/MOI result queries are supported:

```julia
termination_status(model)  # MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.INFEASIBLE, etc.
objective_value(model)     # Best objective found
objective_bound(model)     # Best bound (lower for min, upper for max)
relative_gap(model)        # Relative optimality gap
solve_time(model)          # Solve time in seconds
value(x)                   # Variable values
```

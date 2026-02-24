# ImplicitDDs Solver Documentation

This document provides comprehensive guidance for understanding the ImplicitDDs solver architecture, algorithms, and usage.

## Project Overview

ImplicitDDs is a Julia package implementing a Branch-and-Bound solver using Decision Diagrams (DDs) for Mixed-Integer Programming (MIP) problems. The solver combines **restricted decision diagrams** (providing feasible solutions and upper bounds) with **relaxed decision diagrams** (providing lower bounds through node splitting) to find exact optimal solutions.

## Getting Started

### Basic Usage

```julia
using JuMP
using ImplicitDDs

# Create or load a JuMP model
model = ... # your MIP model

# Solve with default parameters
result = ImplicitDDs.solve_mip(
    model;
    relaxed_w = 1000,      # Width limit for relaxed DDs
    restricted_w = 1000,    # Width limit for restricted DDs
    parallel_processing = true,
    time_limit = nothing    # Optional time limit in seconds
)

# Access results
if result.is_feasible
    println("Objective: ", result.objective_value)
    println("Integer solution: ", result.bks_int)
    println("Continuous solution: ", result.bks_cont)
end
```

### MOI Wrapper Usage

```julia
using JuMP
using ImplicitDDs

# Use ImplicitDDs as a JuMP solver
model = Model(ImplicitDDs.Optimizer)
set_optimizer_attribute(model, "relaxed_w", 10000)
set_optimizer_attribute(model, "restricted_w", 10000)

@variable(model, x[1:5], Bin)
@constraint(model, sum(x) <= 3)
@objective(model, Max, sum(i * x[i] for i in 1:5))

optimize!(model)
println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
```

### To Compile the Package

```julia
# In Julia REPL
include("src/ImplicitDDs.jl")  # Compiles and runs example
```

### Main Function Parameters

- `model::JuMP.Model`: JuMP optimization model to solve
- `relaxed_w::Int`: Maximum width for relaxed decision diagrams (default: 1000)
- `restricted_w::Int`: Maximum width for restricted decision diagrams (default: 1000)
- `num_LPs_to_run::Int`: Maximum number of continuous LP subproblems to solve (default: 100)
- `parallel_processing::Bool`: Enable multi-threaded parallel processing (default: true)
- `solution_print::Bool`: Print solutions when found (default: false)
- `bounds_print::Bool`: Print bound improvements (default: true)
- `suppress_all_prints::Bool`: Suppress all console output (default: false)
- `numerical_precision::DataType`: Float precision for calculations (default: Float32)
- `log_file_path::Union{String, Nothing}`: Path for JSON logging (default: nothing)
- `time_limit::Union{<:Real, Nothing}`: Time limit in seconds (default: nothing)

### Return Values

`solve_mip` returns a `SolveResult{Z,T}` struct with the following fields:

- `is_feasible::Bool`: Whether a feasible solution was found
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution (in original model order)
- `bks_cont::Vector{T}`: Best known continuous solution (in original model order)
- `objective_value::T`: Best known objective value (in original objective sense)
- `objective_bound::T`: Best known lower bound (for minimization) or upper bound (for maximization)
- `node_count::Int`: Number of branch-and-bound nodes processed
- `solve_time::Float64`: Total solve time in seconds
- `is_optimal::Bool`: Whether optimality was proven (gap closed, or infeasibility proven)
- `timed_out::Bool`: Whether the solver terminated due to time limit
- `unbounded_error::Bool`: Whether the solver failed due to unbounded variables (OBBT failure)

## Core Architecture

### Directory Structure

```bash
src/
├── decision_diagrams/
│   ├── bnb_and_pnb/          # Branch-and-bound solver (main entry point)
│   │   ├── branch_and_bound.jl  # solve_mip() function, parallel processing
│   │   ├── bnb_subroutines.jl   # DDWorkspace struct, memory preallocation
│   │   ├── SolveResult.jl       # Result struct returned by solve_mip
│   │   ├── QueueNode.jl          # B&B queue node data structure
│   │   └── continuous_variables.jl  # LP optimization for continuous variables
│   ├── restricted_dds/        # Restricted DDs (feasible solutions, upper bounds)
│   │   ├── construction.jl       # Main construction functions
│   │   ├── restricted_nodes.jl   # Data structures
│   │   ├── major_subroutines.jl  # Layer building, cap selection
│   │   └── histogram_cap_approximation.jl  # Bin-based threshold finding
│   ├── relaxed_dds/          # Relaxed DDs (lower bounds via node splitting)
│   │   ├── construction.jl    # Main construction pipeline
│   │   ├── node_splits.jl     # Hybrid threshold algorithm for width control
│   │   └── ...
│   └── pruning/
│       └── fbbt.jl           # Feasibility-Based Bound Tightening
├── MOI_wrapper/              # MathOptInterface wrapper for JuMP integration
│   ├── MOI_wrapper.jl        # Optimizer struct and basic MOI interface
│   ├── attributes.jl         # Parameter get/set methods
│   ├── constraints.jl        # Supported constraint types
│   ├── copy_to.jl            # Model copying from MOI
│   ├── optimize.jl           # MOI.optimize! implementation
│   └── results.jl            # Result query methods
├── MIP_IO/                   # Problem input/output
│   └── process_MIP.jl        # MPS file reading, model conversion
└── presolve/
    ├── presolve.jl           # Problem preprocessing
    ├── var_ordering.jl       # METIS-based variable ordering
    └── prebounding.jl        # Initial bound tightening
```

### Algorithm Flow

1. **Setup & Preprocessing**
   - Load MIP problem from JuMP model, normalize for minimization, and apply METIS variable ordering,
   - Preallocate DD workspace structures
   - Compute initial infimum gaps
   - Run global FBBT for bound tightening

2. **Initial Restricted DD**
   - Build restricted DD from root with width limit
   - Generate initial feasible integer variable assignments (partial solutions)
   - Fix integers from promising partial solutions, solve LPs for continuous variables to generate feasible solutions (upper bounds)
   - May terminate early if DD is exact (proves optimality or infeasibility)

3. **Initial Relaxed DD**
   - Build relaxed DD from root with width limit
   - Extract lower bound from terminal nodes
   - Populate B&B queue with last inexact layer
   - May improve upper bound if better integer solution found
   - May terminate early if infeasible or if lower bound equals upper bound

4. **Branch-and-Bound Loop**
   - Process nodes from priority queue (ordered by lower bound)
   - For each node:
     - Apply local FBBT with node-specific bounds
     - Build restricted DD → update upper bound if better solution found
     - Build relaxed DD → extract lower bound and add last exact layer to BnB queue
   - Prune nodes where `node lower_bound > global upper_bound`
   - Terminate when queue empty or optimality proven

5. **Parallel Execution** (if enabled)
   - Master thread manages priority queue and dispatches nodes
   - Worker threads process nodes independently
   - Results synchronized through channels
   - Termination when queue empty and all workers idle

## Key Algorithms

### Restricted Decision Diagrams

**Purpose**: Generate feasible solutions and provide upper bounds.

**Algorithm**: Uses objective-threshold-based filtering to maintain width limit (w). For each node:

1. Compute implied domain bounds on next variable using constraint propagation
2. Create a histogram of the out-arcs: bin the arcs based on objecive value
3. Identify the highest valued bin edge such that the number of arcs below tha threshold is less than w.
4. Expand all arcs ≤ threshold plus additional arcs from next bin to reach width w
5. Terminal nodes represent complete feasible integer solutions if there are no continuous variables
6. With continuous variables, LPs solved at terminal nodes to find complete feasible solutions

**Key Functions**:

- `setup_and_run_restricted_dd!`: Entry point
- `create_restricted_dd!`: Main loop over the layers (variables)
- `compute_implied_column_bounds!`: Constraint propagation
- `find_cap_histogram_approximation!`: Bin-based threshold selection
- `build_restricted_node_layer!`: Layer construction

### Relaxed Decision Diagrams

**Purpose**: Provide valid lower bounds through node splitting (relaxation).

**Mathematical Properties**:

- Node splitting creates additional paths without eliminating optimal ones
- Guarantees `relaxed_bound ≤ optimal_value`
- Max-based infimum gap (slack residuals) updates ensure constraint relaxation validity
- Min-based LTR updates find optimal paths through split nodes

**Algorithm**: Layer-by-layer construction with width control (w):

1. Compute implied domain bounds on next variable using constraint propagation
2. Generate compact layer with one node per domain value, tracking parents as index intervals (instead of pointers)
3. Split nodes using hybrid threshold algorithm, similar to what the restricted DDs do, to expand toward width w in a single pass
4. Update LTR values and apply rough bound pruning
5. Propagate constraints (infimum gaps) to next layer
6. Track last exact layer for efficient post-processing

**Width Control**: Hybrid threshold with budget-based individualization

- Compact layer starts with K nodes (K = domain size)
- Three zones: safe (always split), middle (budget-based), aggressive (keep as blocks)
- Guarantees: `w - K ≤ layer_size ≤ w` (except when layer is exact with fewer nodes needed)

**Post-processing**:

1. Find terminal node with minimum LTR (best lower bound candidate)
2. If continuous variables exist: compute terminal infimum gaps (constraint residuals) and solve one LP to get continuous contribution to the relaxation
3. If no continuous variables: backtrack to extract best integer path and check feasibility
4. If path is feasible and improves upper bound, update best solution and return early
5. Compute LTT (length-to-terminal) values backward through all layers for bound-based pruning
6. Process last exact layer to generate nodes for B&B queue
7. Return lower bound (minimum LTR + continuous contribution) and updated best solution

**Key Functions**:

- `setup_run_process_relaxed_dd!`: Full pipeline entry point
- `compute_relaxed_dd!`: Main construction loop over variables
- `compute_implied_column_bounds!`: Constraint propagation for domain bounds
- `invert_implied_column_bounds!`: Generate compact layer from parent intervals
- `split_nodes!`: Layer building entry point (selects and calls appropriate build strategy)
- `compute_bounding_thresholds`: Dual-threshold selection for width control
- `build_layer_with_hybrid_thresholds`: Budget-based individualization
- `post_process_relaxed_dd!`: Bound extraction and feasibility checking
- `process_last_exact_layer!`: Frontier cutset generation

### Feasibility-Based Bound Tightening (FBBT)

**Purpose**: Strengthen variable bounds through constraint propagation.

**Mathematical Foundation**:
For constraint `∑ a_ij x_j ≤ b_i`:

1. Use precomputed infimum gap: `gap_i = b_i - inf{∑_j a_ij x_j}`
2. Derive bounds by isolating variable `x_k`:
   - If `a_ik > 0`: `x_k ≤ gap_i/a_ik + lb_k` → apply floor for integers
   - If `a_ik < 0`: `x_k ≥ gap_i/a_ik + ub_k` → apply ceiling for integers
3. Update infimum gaps in real-time as bounds tighten for immediate propagation
4. Iterate until convergence or maximum iterations reached

**Key Properties**:

- Exploits constraint structure directly through algebraic manipulation
- Bound improvements propagate through constraint network
- Integrates naturally with infimum gap calculations

**Implementation**: `src/decision_diagrams/pruning/fbbt.jl`

- Applied globally during preprocessing
- Applied locally at each B&B node before building DDs

## Key Concepts

### Length-To-Root (LTR)

**LTR** = cumulative objective value from root to current node (not "Left-To-Right").

Used for:

- Prioritizing which arcs to expand in width-limited layers
- Determining thresholds for layer construction
- Pruning via rough bounds: nodes with `ltr + rough_bound ≥ best_upper_bound` are pruned

### Infimum Gaps

**Infimum gaps** (called "constraint residuals" in the paper) track constraint slack throughout DD construction:

**Definition**: For constraint `i`:

```julia
infimum_gap[i] = RHS[i] - ∑_j (coeff[i,j] * bound[j])
```

where bounds are chosen to minimize the sum (lower bounds for positive coefficients, upper bounds for negative).

**Usage**:

- **Restricted DDs**: `compute_implied_column_bounds!` uses gaps to determine feasible variable domains at each node
- **Relaxed DDs**: Terminal nodes use gaps for continuous LP subproblems
- **Dynamic updates**: Gaps propagate through DD layers as variables are fixed
- **Feasibility checking**: Negative gaps indicate constraint violations

### Type Parameterization

All data structures are parameterized for memory efficiency:

**Integer Types** (`Z<:Integer`):

- Automatically selects smallest type based on variable bounds
- All binary (0-1): `Bool` with `BitVector` storage (1 bit per element)
- Bounds fit in Int8/Int16/Int32: uses corresponding type
- Otherwise: `Int64`
- Reduces memory footprint for problems with small domains
- `BitVector` for binary variables uses 1 bit per element (compact storage)

**Real Types** (`T<:Real`):

- Typically `Float32` or `Float64`
- Configurable via `numerical_precision` parameter

## Data Structures

### DDWorkspace{Z,T}

Container for all preallocated memory structures (prevents repeated allocation during B&B).

**Location**: `src/decision_diagrams/bnb_and_pnb/bnb_subroutines.jl`

**Key Fields**:

```julia
mutable struct DDWorkspace{Z<:Integer, T<:Real}
    # Relaxed DD
    lb_matrix::Union{Matrix{Z}, BitMatrix}
    ub_matrix::Union{Matrix{Z}, BitMatrix}
    infimum_gap_matrices::Matrix{T}
    ltr_matrix::Matrix{T}
    ltt_matrix::Matrix{T}
    node_matrix::Vector{NodeLayer{Z}}

    # Restricted DD
    rdd::RestrictedDD

    # Precomputed values (hot loop optimization)
    coeff_times_val::Matrix{T}      # Arc objectives: coeff * val
    gap_adjustments::Array{T,3}     # Gap updates: coeff * (bound - val)
    inv_coeff::Matrix{T}            # Inverse coefficients for division→multiplication
    inv_obj_coeffs::Vector{T}       # Inverse objective coefficients

    # Solution tracking
    bks_int::Union{Vector{Z}, BitVector}
    bks_cont::Vector{T}

    # Local bounds (B&B node-specific, modified by FBBT)
    local_lbs_int::Union{Vector{Z}, BitVector}
    local_ubs_int::Union{Vector{Z}, BitVector}
    local_lbs_cont::Vector{T}
    local_ubs_cont::Vector{T}
    local_infimum_gaps::Vector{T}
    local_cont_inf_contr::Vector{T}

    # Original bounds (pre-FBBT, for consistent array indexing)
    original_lbs_int::Union{Vector{Z}, BitVector}
    original_ubs_int::Union{Vector{Z}, BitVector}

    # Branch-and-bound
    bnb_queue::Vector{QueueNode{Z, T}}

    # Logging
    logs::Vector{Dict{String, Any}}
    timing_stats::TimingStats
    ...
end
```

### QueueNode{Z,T}

Branch-and-bound queue node representing a subproblem with partially fixed variables.

**Location**: `src/decision_diagrams/bnb_and_pnb/QueueNode.jl`

**Key Fields**:

```julia
struct QueueNode{Z<:Integer, T<:Real}
    ltr::T                    # Length-to-root (objective value of fixed variables)
    implied_lb::Z             # Implied lower bound for next unfixed variable
    implied_ub::Z             # Implied upper bound for next unfixed variable
    path::Union{Vector{Z}, BitVector}  # Variable assignments (length = number of fixed vars)
    implied_bound::T          # Lower bound for this subproblem
    cont_bound_contr::T       # Continuous variable contribution to bound
end
```

**Usage**: Nodes are stored in a priority queue ordered by `implied_bound` (best-first search).

### RestrictedNodeLayer{T,U,V}

Individual layer in restricted decision diagram.

**Location**: `src/decision_diagrams/restricted_dds/restricted_nodes.jl`

**Key Fields**:

```julia
mutable struct RestrictedNodeLayer{T<:Integer, U<:Real, V<:Integer}
    arcs::Vector{V}                  # Parent node indices (-1 for root)
    values::Union{Vector{T}, BitVector}  # Variable values for this layer
    ltrs::Vector{U}                  # Length-to-root objective values
    implied_lbs::Union{Vector{T}, BitVector}  # Implied lower bounds from constraint propagation
    implied_ubs::Union{Vector{T}, BitVector}  # Implied upper bounds from constraint propagation
    size::V                          # Current number of nodes in layer
end
```

**Architecture**: Struct-of-arrays design where each index `i` represents node `i`. For example, node 3 has parent `arcs[3]`, value `values[3]`, objective `ltrs[3]`, and bounds `[implied_lbs[3], implied_ubs[3]]`. This layout provides better cache locality and SIMD optimization compared to array-of-structs.

**Note**: Gap matrices are stored separately in `RestrictedDD.gap_buffers` for double-buffering efficiency.

### NodeLayer{Z}

Individual layer in relaxed decision diagram (interval-based representation).

**Location**: `src/decision_diagrams/relaxed_dds/nodes.jl`

**Key Fields**:

```julia
mutable struct NodeLayer{Z<:Integer}
    first_arcs::Vector{Int32}  # First parent index in interval
    last_arcs::Vector{Int32}   # Last parent index in interval
    values::Union{Vector{Z}, BitVector}  # Variable values
    active::BitVector          # Pruning flags (rough bound filtering)
    size::Int32                # Current number of nodes
end
```

**Architecture**: Struct-of-arrays design where each index `i` represents node `i`. For example, node 5 has value `values[5]`, parent interval `[first_arcs[5], last_arcs[5]]`, and pruning flag `active[5]`. This layout provides better cache locality and SIMD optimization.

**Key Characteristic**: Uses interval-based arc ranges `[first_arc, last_arc]` to represent groups of parent nodes efficiently.

## Parallel Processing

### Architecture

**Master-Worker Pattern** with work-stealing for B&B parallelization.

**Implementation**: `src/decision_diagrams/bnb_and_pnb/branch_and_bound.jl`

**Master Thread** (Thread 1):

- Owns priority queue exclusively (no locking needed)
- Pops highest-priority nodes and dispatches via `work_channel`
- Receives results via `results_channel`
- Updates global best solution and lower bound
- Manages termination

**Worker Threads** (Threads 2...N):

- Own isolated `DDWorkspace{Z,T}` and LP model (thread-safe)
- Receive `(node, current_bkv, time_remaining)` from master
- Call `process_queue_node!()` independently
- Return `(temp_bkv, new_nodes, bks_int_copy, bks_cont_copy, logs_copy, node_bound)`
- Clear local state between tasks

### Communication Protocol

**Channels**:

- `work_channel`: Master → Workers, capacity `num_workers * 2`
- `results_channel`: Workers → Master, capacity `num_workers * 2`

**Poison Pill Pattern**:

- Master sends `QueueNode` with empty path to signal shutdown
- Workers detect empty path and exit loop

**Termination**:

- `in_flight` atomic counter tracks active workers
- Master terminates when `isempty(global_queue) && in_flight[] == 0` or timeout
- On timeout: drains both channels, sends poison pills, waits for workers

### Critical Design Points

**yield() for Fairness**: Master loop calls `yield()` when waiting to prevent CPU starvation of worker threads.

**Solution Vector Copying**: Workers must send `copy(dd_ws.bks_int)` and `copy(dd_ws.bks_cont)` since vectors are mutable.

**Log Clearing**: Workers must `empty!(dd_ws.logs)` after sending to prevent accumulation.

**LP Thread-Safety**: Each worker owns its LP model exclusively. This may be a critical issue for LP solvers that are not thread safe.

**JIT Compilation**: Automatically precompiles hot paths before launching workers to avoid JIT deadlocks.

## Code Style Guidelines

### Docstring Format (Public Functions)

```julia
"""
    function_name(arg1::Type1, arg2::Type2) where {T<:SomeType}

Brief one-line description ending with a period.

# Arguments
- `arg1::Type1`: Description of the argument
- `arg2::Type2`: Description of the argument

# Returns
- `ReturnType`: Description containing:
  1. `SubType1`: Description of first return component
  2. `SubType2`: Description of second return component
"""
```

**Requirements**:

- Include full function signature with types
- End description with period
- Use `::FullType` format in Arguments section
- Use numbered lists for tuple/complex return types

### Inline Comments (Internal Functions)

- Use `#` for single-line comments describing purpose (not implementation)
- Group parameters with section headers: `#problem description`, `#precomputed values`, `#settings`
- Rely on descriptive variable names to reduce comment needs

### Variable Naming Conventions

**Common Abbreviations**:

- `ltr`: Length-to-root (cumulative objective from root)
- `ltt`: Length-to-terminal (distance to terminal nodes)
- `rdd`: Restricted decision diagram
- `bkv`: Best known value (upper bound)
- `bks`: Best known solution
- `w`: Width limit for decision diagram layers
- `dd_ws`: Decision diagram workspace

## Performance Considerations

### Memory Optimization

- Type parameterization: Automatic selection of smallest integer type
- Preallocated workspace: All DD structures allocated once at startup
- Double-buffering: Gap matrices and node layers alternate to avoid unneeded allocation
- Column views: Cache-friendly access patterns in hot loops

### Computational Optimization

- Bin-based histogram: O(Kw) threshold finding via single-pass arc binning and cumulative counting
- Branchless operations: Uses `min`/`max` for better pipeline utilization
- Precomputed values: `coeff_times_val`, `inv_coeff` eliminate repeated arithmetic in hot loops

### Tuning Parameters

**Width Parameters** (`relaxed_w`, `restricted_w`):

- Larger values: Tighter bounds, more memory, longer construction time
- Typical range: 1,000 to 1,000,000
- Start with 10,000-100,000 for most problems

**Numerical Precision**:

- `Float32`: Faster, less memory, sufficient for most problems
- `Float64`: Use if numerical issues arise, no known examples yet

## Dependencies

- **JuMP**: Mathematical optimization modeling
- **MathOptInterface**: Optimization interface (aliased as MOI)
- **HiGHS**: LP solver for continuous variable optimization
- **Metis**: Graph partitioning for variable ordering
- **Graphs**: Graph data structures for adjacency analysis
- **JSON**: Logging and output serialization
- **SparseArrays**: Sparse matrix operations
- **LinearAlgebra**: Linear algebra operations
- **Printf**: Formatted output
- **Revise**: Development hot-reloading (dev only)
- **Random**: Random number generation (dev only)

**Note**: Performance timing uses a custom `TimingStats` struct (src/TimingStats.jl), not an external package.

"""
    SolveResult{Z<:Integer, T<:Real}

Result struct returned by `solve_mip` containing solution data and solver statistics.

# Fields
- `is_feasible::Bool`: Whether a feasible solution was found
- `bks_int::Union{Vector{Z}, BitVector}`: Best known integer solution (in original model order)
- `bks_cont::Vector{T}`: Best known continuous solution (in original model order)
- `objective_value::T`: Best known objective value (in original objective sense)
- `objective_bound::T`: Best known lower bound (for minimization) or upper bound (for maximization)
- `node_count::Int`: Number of branch-and-bound nodes processed
- `solve_time::Float64`: Total solve time in seconds (excluding model conversion)
- `is_optimal::Bool`: Whether optimality was proven (gap closed, or infeasibility proven)
- `timed_out::Bool`: Whether the solver terminated due to time limit
- `unbounded_error::Bool`: Whether the solver failed due to unbounded variables (OBBT failure)
"""
struct SolveResult{Z<:Integer, T<:Real}
    is_feasible::Bool
    bks_int::Union{Vector{Z}, BitVector}
    bks_cont::Vector{T}
    objective_value::T
    objective_bound::T
    node_count::Int
    solve_time::Float64
    is_optimal::Bool
    timed_out::Bool
    unbounded_error::Bool
end

"""
    return_infeasible(::Type{Z}, ::Type{T}, node_count::Int, solve_time::Float64, is_optimal::Bool, timed_out::Bool) where {Z<:Integer, T<:Real}

Construct a SolveResult indicating an infeasible or unsolved problem.

# Arguments
- `Z::Type{<:Integer}`: Integer type for the result (determined by variable bounds)
- `T::Type{<:Real}`: Real type for the result (numerical precision setting)
- `node_count::Int`: Number of B&B nodes processed before termination
- `solve_time::Float64`: Total solve time in seconds
- `is_optimal::Bool`: Whether infeasibility was proven (true) or search was incomplete (false)
- `timed_out::Bool`: Whether the solver terminated due to time limit

# Returns
- `SolveResult{Z, T}`: Result with `is_feasible=false, unbounded_error=false`

# State Interpretation
- `is_optimal=true, timed_out=false`: Problem proven infeasible
- `is_optimal=false, timed_out=true`: Timed out without finding a solution
"""
function return_infeasible(::Type{Z}, ::Type{T}, node_count::Int, solve_time::Float64, is_optimal::Bool, timed_out::Bool) where {Z<:Integer, T<:Real}
    return SolveResult{Z, T}(
        false,           # is_feasible
        Z[],             # bks_int
        T[],             # bks_cont
        T(0),            # objective_value
        T(0),            # objective_bound
        node_count,      # node_count
        solve_time,      # solve_time
        is_optimal,      # is_optimal
        timed_out,       # timed_out
        false            # unbounded_error
    )
end


"""
    return_unbounded_error(::Type{Z}, ::Type{T}, solve_time::Float64) where {Z<:Integer, T<:Real}

Construct a SolveResult indicating a solver error due to unbounded variables.

This is returned when OBBT fails to establish finite bounds on all variables,
which is a requirement for the decision diagram solver.

# Arguments
- `Z::Type{<:Integer}`: Integer type for the result
- `T::Type{<:Real}`: Real type for the result
- `solve_time::Float64`: Total solve time in seconds

# Returns
- `SolveResult{Z, T}`: Result with `is_feasible=false, is_optimal=false, timed_out=false`

# MOI Mapping
This should map to `MOI.OTHER_ERROR` termination status, not `MOI.INFEASIBLE`.
"""
function return_unbounded_error(::Type{Z}, ::Type{T}, solve_time::Float64) where {Z<:Integer, T<:Real}
    return SolveResult{Z, T}(
        false,           # is_feasible
        Z[],             # bks_int
        T[],             # bks_cont
        T(0),            # objective_value
        T(0),            # objective_bound
        0,               # node_count
        solve_time,      # solve_time
        false,           # is_optimal
        false,           # timed_out
        true             # unbounded_error
    )
end


"""
    return_solution(bks_int, bks_cont, inverse_order, bkv, best_bound, sense_multiplier, node_count, solve_time, is_optimal, timed_out)

Construct a SolveResult with solution data, reordering variables to original order.

# Arguments
- `bks_int::Union{Vector{Z}, BitVector}`: Best integer solution in METIS (internal) order
- `bks_cont::Vector{T}`: Best continuous solution (not reordered)
- `inverse_order::Vector{Int}`: Permutation mapping METIS order → original variable order
- `bkv::T`: Best objective value in solver's internal minimization sense
- `best_bound::T`: Best lower bound in solver's internal minimization sense
- `sense_multiplier::Int`: +1 for minimization, -1 for maximization (to convert back)
- `node_count::Int`: Number of B&B nodes processed
- `solve_time::Float64`: Total solve time in seconds
- `is_optimal::Bool`: Whether optimality was proven (lower bound = upper bound)
- `timed_out::Bool`: Whether the solver terminated due to time limit

# Returns
- `SolveResult{Z, T}`: Result with solution in original variable order and objective sense
"""
function return_solution(
    bks_int             ::Union{Vector{Z}, BitVector},
    bks_cont            ::Vector{T},
    inverse_order       ::Vector{Int},
    bkv                 ::T,
    best_bound          ::T,
    sense_multiplier    ::Int,
    node_count          ::Int,
    solve_time          ::Float64,
    is_optimal          ::Bool,
    timed_out           ::Bool
) where {Z<:Integer, T<:Real}
    # Determine integer type (BitVector uses Bool, Vector{Z} uses Z)
    IntType = bks_int isa BitVector ? Bool : Z

    # Reorder integer variables from METIS order to original user order
    reordered_int = bks_int[inverse_order]

    # Convert objective value and bound back to original sense
    # (solver normalizes to minimization; sense_multiplier reverses this for max problems)
    obj_val = sense_multiplier * bkv
    obj_bound = sense_multiplier * best_bound

    return SolveResult{IntType, T}(
        true,            # is_feasible
        reordered_int,   # bks_int (now in original order)
        bks_cont,        # bks_cont
        obj_val,         # objective_value (in original sense)
        obj_bound,       # objective_bound (in original sense)
        node_count,      # node_count
        solve_time,      # solve_time
        is_optimal,      # is_optimal
        timed_out,       # timed_out
        false            # unbounded_error
    )
end

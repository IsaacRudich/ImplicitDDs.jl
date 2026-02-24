"""
COMMENTS UP TO DATE
"""

"""
    eps_equals(a::T, b::U; eps::V = .0000000001) where {T,U,V <: Real}

Returns true if a and b are within eps of each other.

# Arguments
- `a::T`: First number to compare
- `b::U`: Second number to compare
- `eps::V`: The allowable gap between a and b (default: 1e-10)

# Returns
- `Bool`: True if |a - b| < eps, false otherwise
"""
function eps_equals(a::T, b::U; eps::V = .0000000001)where{T,U,V <: Real}
    if abs(a-b) < eps
        return true
    else
        return false
    end
end


"""
    create_preallocated_set(type::DataType, capacity::Int)

Creates a set with preallocated capacity for improved performance.

# Arguments
- `type::DataType`: The element type of the set
- `capacity::Int`: The expected number of elements to preallocate space for

# Returns
- `Set{type}`: A new set with preallocated capacity
"""
function create_preallocated_set(type::DataType, capacity::Int)
    s = Set{type}()
    sizehint!(s, capacity)
    return s
end


"""
    pretty_print(m::Matrix{T}; show_zeros::Bool = true) where {T<:Real}

Prints a matrix in a more human readable format than Julia's default display.

# Arguments
- `m::Matrix{T}`: The matrix to print
- `show_zeros::Bool`: Whether to display zero elements (default: true)

# Returns
- `Nothing`: Prints output to console
"""
function pretty_print(m::Matrix{T}; show_zeros::Bool = true) where T<:Real
    for i in 1:size(m, 1)
        for j in 1:size(m, 2)
            if !show_zeros && m[i,j]!= 0
                print(round(m[i,j],digits=2), "  ")
            end
        end
        println()
    end
end


"""
    preallocate_int32_vector(w::Int)

Preallocates an uninitialized vector of Int32 elements for performance optimization.

# Arguments
- `w::Int`: The size of the vector to allocate

# Returns
- `Vector{Int32}`: An uninitialized vector of length w
"""
@inline function preallocate_int32_vector(w::Int)
    return Vector{Int32}(undef, w)
end

"""
    preallocate_int64_vector(w::Int)

Preallocates an uninitialized vector of Int64 elements for performance optimization.

# Arguments
- `w::Int`: The size of the vector to allocate

# Returns
- `Vector{Int64}`: An uninitialized vector of length w
"""
@inline function preallocate_int64_vector(w::Int)
    return Vector{Int64}(undef, w)
end


"""
    preallocate_vector(w::Int, numerical_precision::DataType)

Preallocates an uninitialized vector of specified numeric type for performance optimization.

# Arguments
- `w::Int`: The size of the vector to allocate
- `numerical_precision::DataType`: The numeric type for vector elements (e.g., Float32, Float64)

# Returns
- `Vector{numerical_precision}`: An uninitialized vector of length w with specified element type
"""
@inline function preallocate_vector(w::Int, numerical_precision::DataType)
    return Vector{numerical_precision}(undef, w)
end

"""
    preallocate_zero_matrix(num_rows::Int, num_cols::Int, numerical_precision::DataType)

Preallocates a zero-initialized matrix of specified numeric type for algorithms that require zero-initialization.

# Arguments
- `num_rows::Int`: The number of rows in the matrix
- `num_cols::Int`: The number of columns in the matrix
- `numerical_precision::DataType`: The numeric type for matrix elements (e.g., Float32, Float64)

# Returns
- `Matrix{numerical_precision}`: A zero-initialized matrix of size (num_rows * num_cols) with specified element type
"""
@inline function preallocate_zero_matrix(num_rows::Int, num_cols::Int, numerical_precision::DataType)
    return zeros(numerical_precision, num_rows, num_cols)
end


"""
    time_budget_exceeded(time_remaining::Union{Float64, Nothing}, fn_start::Float64)

Checks if the time budget for a function has been exceeded.

# Arguments
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds (nothing if no time limit)
- `fn_start::Float64`: Timestamp when function started execution (from time())

# Returns
- `Bool`: True if time budget exceeded, false otherwise (always false if time_remaining is nothing)
"""
@inline function time_budget_exceeded(time_remaining::Union{Float64, Nothing}, fn_start::Float64)
    return !isnothing(time_remaining) && (time() - fn_start >= time_remaining)
end


"""
    calculate_child_time_budget(time_remaining::Union{Float64, Nothing}, fn_start::Float64)

Calculates the remaining time budget to pass to a child function.

# Arguments
- `time_remaining::Union{Float64, Nothing}`: Time budget in seconds (nothing if no time limit)
- `fn_start::Float64`: Timestamp when parent function started execution (from time())

# Returns
- `Union{Float64, Nothing}`: Remaining time budget for child function (nothing if no time limit)
"""
@inline function calculate_child_time_budget(time_remaining::Union{Float64, Nothing}, fn_start::Float64)
    return isnothing(time_remaining) ? nothing : time_remaining - (time() - fn_start)
end


"""
    sanitize_for_json(x)

Recursively sanitizes data for JSON serialization by replacing Inf/-Inf with typemax/typemin.
JSON spec does not allow Inf/NaN values, so this function converts them to valid numbers.

# Arguments
- `x`: Any value to sanitize (Dict, Vector, Number, or other)

# Returns
- Sanitized version of x with Inf replaced by typemax and -Inf replaced by typemin
"""
function sanitize_for_json(x::AbstractFloat)
    if isinf(x)
        return x > 0 ? floatmax(typeof(x)) : -floatmax(typeof(x))
    elseif isnan(x)
        return nothing
    else
        return x
    end
end

function sanitize_for_json(x::Dict)
    return Dict(k => sanitize_for_json(v) for (k, v) in x)
end

function sanitize_for_json(x::AbstractVector)
    return [sanitize_for_json(v) for v in x]
end

function sanitize_for_json(x)
    return x
end

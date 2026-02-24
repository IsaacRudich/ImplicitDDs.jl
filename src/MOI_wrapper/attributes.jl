# =============================================================================
# RawOptimizerAttribute support (string-based solver parameters)
# =============================================================================

const SUPPORTED_RAW_ATTRIBUTES = Set([
    "relaxed_w",
    "restricted_w",
    "num_LPs_to_run",
    "parallel_processing",
    "numerical_precision",
    "debug_mode",
    "log_file_path",
    "bounds_print",
    "solution_print",
    "wait_to_write_solutions",
    "timer_outputs",
    "custom_variable_order",
    "time_limit",
    "silent",
])

function MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute)
    return attr.name in SUPPORTED_RAW_ATTRIBUTES
end

function MOI.get(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute)
    name = attr.name
    if name == "relaxed_w"
        return optimizer.relaxed_w
    elseif name == "restricted_w"
        return optimizer.restricted_w
    elseif name == "num_LPs_to_run"
        return optimizer.num_LPs_to_run
    elseif name == "parallel_processing"
        return optimizer.parallel_processing
    elseif name == "numerical_precision"
        return optimizer.numerical_precision
    elseif name == "debug_mode"
        return optimizer.debug_mode
    elseif name == "log_file_path"
        return optimizer.log_file_path
    elseif name == "bounds_print"
        return optimizer.bounds_print
    elseif name == "solution_print"
        return optimizer.solution_print
    elseif name == "wait_to_write_solutions"
        return optimizer.wait_to_write_solutions
    elseif name == "timer_outputs"
        return optimizer.timer_outputs
    elseif name == "custom_variable_order"
        return optimizer.custom_variable_order
    elseif name == "time_limit"
        return optimizer.time_limit
    elseif name == "silent"
        return optimizer.silent
    else
        throw(MOI.UnsupportedAttribute(attr))
    end
end

function MOI.set(optimizer::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    name = attr.name
    if name == "relaxed_w"
        value isa Real && isinteger(value) && value >= 1 || throw(ArgumentError("relaxed_w must be a positive integer"))
        optimizer.relaxed_w = Int(value)
    elseif name == "restricted_w"
        value isa Real && isinteger(value) && value >= 1 || throw(ArgumentError("restricted_w must be a positive integer"))
        optimizer.restricted_w = Int(value)
    elseif name == "num_LPs_to_run"
        value isa Real && isinteger(value) && value >= 0 || throw(ArgumentError("num_LPs_to_run must be a non-negative integer"))
        optimizer.num_LPs_to_run = Int(value)
    elseif name == "parallel_processing"
        value isa Bool || throw(ArgumentError("parallel_processing must be a Bool"))
        optimizer.parallel_processing = value
    elseif name == "numerical_precision"
        value isa DataType && value <: AbstractFloat || throw(ArgumentError("numerical_precision must be a floating-point type"))
        optimizer.numerical_precision = value
    elseif name == "debug_mode"
        value isa Bool || throw(ArgumentError("debug_mode must be a Bool"))
        optimizer.debug_mode = value
    elseif name == "log_file_path"
        value === nothing || value isa String || throw(ArgumentError("log_file_path must be a String or nothing"))
        optimizer.log_file_path = value
    elseif name == "bounds_print"
        value isa Bool || throw(ArgumentError("bounds_print must be a Bool"))
        optimizer.bounds_print = value
    elseif name == "solution_print"
        value isa Bool || throw(ArgumentError("solution_print must be a Bool"))
        optimizer.solution_print = value
    elseif name == "wait_to_write_solutions"
        value isa Bool || throw(ArgumentError("wait_to_write_solutions must be a Bool"))
        optimizer.wait_to_write_solutions = value
    elseif name == "timer_outputs"
        value isa Bool || throw(ArgumentError("timer_outputs must be a Bool"))
        optimizer.timer_outputs = value
    elseif name == "custom_variable_order"
        value === nothing || value isa Vector{MOI.VariableIndex} || throw(ArgumentError("custom_variable_order must be a Vector{MOI.VariableIndex} or nothing"))
        optimizer.custom_variable_order = value
    elseif name == "time_limit"
        value === nothing || (value isa Real && value >= 0) || throw(ArgumentError("time_limit must be a non-negative number or nothing"))
        optimizer.time_limit = value === nothing ? nothing : Float64(value)
    elseif name == "silent"
        value isa Bool || throw(ArgumentError("silent must be a Bool"))
        optimizer.silent = value
    else
        throw(MOI.UnsupportedAttribute(attr))
    end
    return nothing
end

# =============================================================================
# Standard MOI attributes
# =============================================================================

# Silent mode (maps to suppress_all_prints in solve_mip)
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return nothing
end

# Time limit in seconds (converted to minutes for solve_mip)
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec) = optimizer.time_limit

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, value::Union{Real, Nothing})
    if value !== nothing && value < 0
        throw(ArgumentError("TimeLimitSec must be non-negative"))
    end
    optimizer.time_limit = value === nothing ? nothing : Float64(value)
    return nothing
end

"""
COMMENTS UP TO DATE
"""

include("process_MIP.jl")
include("check_solutions.jl")

"""
    read_MIP_from_MPS(file_path::String)

Reads a Mixed-Integer Programming (MIP) model from an MPS file.

# Arguments
- `file_path::String`: Path to the MPS file

# Returns
- `MathOptInterface.Utilities.GenericModel`: The loaded MIP model
"""
function read_MIP_from_MPS(file_path::String)
    mps_model = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_MPS)

    MOI.read_from_file(mps_model, file_path)

    return mps_model
end


"""
    read_into_HiGHS_from_MPS(file_path::String)

Reads a Mixed-Integer Programming (MIP) model from an MPS file and stores it as a JuMP model.

# Arguments
- `file_path::String`: Path to the MPS file

# Returns
- `JuMP.Model`: A JuMP model initialized with HiGHS optimizer and loaded with the MIP from the MPS file
"""
function read_into_HiGHS_from_MPS(file_path::String)
    # Create a JuMP model with HiGHS as the optimizer (supports both integer and continuous variables)
    jump_model = Model(HiGHS.Optimizer)
    
    # Read the MPS file into the JuMP model's MOI backend.
    MOI.read_from_file(backend(jump_model), file_path)

    # Turn off logging
    set_silent(jump_model)

    return jump_model
end
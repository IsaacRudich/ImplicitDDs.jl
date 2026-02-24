module ImplicitDDs
    #io 
    using JuMP
    using MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using JSON

    #variable ordering
    using SparseArrays
    using LinearAlgebra
    using Graphs
    using Metis

    #solver
    using HiGHS

    # random mip generation
    using Random

    #for development only
    # using Revise
 
    #compilation instructions
    include("utilities.jl")
    include("TimingStats.jl")
    include("MIP_IO/MIP_IO.jl")
    include("presolve/presolve.jl")
    include("decision_diagrams/decision_diagrams.jl")
    include("MOI_wrapper/MOI_wrapper.jl")
    include("random_mip_generator.jl")
    include("precompile.jl")

    include("debug_functions.jl")

    export Optimizer
end
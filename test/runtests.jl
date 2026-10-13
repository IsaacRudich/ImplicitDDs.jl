using Test

include("test_utilities.jl")

@testset "ImplicitDDs.jl" begin
    @testset "Solver Correctness" begin
        result = run_serial_validation(15)
        @test result.failed == 0
    end
    @testset "Equality and Two-Sided Constraints" begin
        result = run_mixed_validation(15)
        @test result.failed == 0
    end
end

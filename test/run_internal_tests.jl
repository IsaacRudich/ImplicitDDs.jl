# Internal regression tests (not run by Pkg.test)
using Test
using JuMP
using HiGHS
using ImplicitDDs

# Solve with ImplicitDDs directly and report whether the problem was proven infeasible
function proven_infeasible(model::JuMP.Model)
    result = ImplicitDDs.solve_mip(model; suppress_all_prints = true, parallel_processing = false)
    return !result.is_feasible && result.is_optimal
end

@testset "Internal regression tests" begin
    @testset "Inconsistent bounds are infeasible" begin
        # Continuous variable declared with lower > upper
        m = Model(HiGHS.Optimizer)
        @variable(m, 0 <= x <= 3, Int)
        @variable(m, -10 <= z <= -11)
        @objective(m, Min, x + z)
        @test proven_infeasible(m)

        # Integer variable declared with lower > upper
        m = Model(HiGHS.Optimizer)
        @variable(m, -10 <= x <= -11, Int)
        @variable(m, 0 <= y <= 3, Int)
        @objective(m, Min, x + y)
        @test proven_infeasible(m)

        # Two-sided constraint with lower > upper, containing a continuous variable
        m = Model(HiGHS.Optimizer)
        @variable(m, 0 <= x <= 3, Int)
        @variable(m, -20 <= z <= 20)
        @constraint(m, -10 <= x + z <= -11)
        @objective(m, Min, x + z)
        @test proven_infeasible(m)

        # Two-sided constraint empty by less than constraint_feasibility_tol is solved, not infeasible
        m = Model(HiGHS.Optimizer)
        @variable(m, 0 <= x <= 3, Int)
        @variable(m, -20 <= z <= 20)
        @constraint(m, -10 <= x + z <= -10 - 1e-9)
        @objective(m, Min, x + z)
        result = ImplicitDDs.solve_mip(m; suppress_all_prints = true, parallel_processing = false)
        @test result.is_feasible
        @test isapprox(result.objective_value, -10; atol = 1e-6)
    end
end

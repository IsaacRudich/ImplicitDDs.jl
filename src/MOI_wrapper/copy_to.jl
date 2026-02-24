# =============================================================================
# MOI.copy_to: receive model data from CachingOptimizer
# =============================================================================

"""
    MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)

Copy the model from `src` into the optimizer. This stores a lightweight MOI
model copy and builds the variable index mapping. The actual JuMP model for
solve_mip is built transiently inside `optimize!`.
"""
function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)

    # Copy source into an internal MOI model
    inner = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    index_map = MOI.copy_to(inner, src)

    dest.src_model = inner
    dest.src_index_map = index_map

    # Build variable_map: source variable index → position in solution vector.
    # solve_mip returns solutions as [int_vars..., cont_vars...] in the order
    # they appear in all_variables(model), separated by integrality.
    # We replicate that ordering here using the copied model's data.
    src_vars = MOI.get(src, MOI.ListOfVariableIndices())

    # Determine which variables are integer/binary vs continuous
    integer_indices = Set{MOI.VariableIndex}()
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Integer}())
        vi = MOI.get(src, MOI.ConstraintFunction(), ci)
        push!(integer_indices, vi)
    end
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.ZeroOne}())
        vi = MOI.get(src, MOI.ConstraintFunction(), ci)
        push!(integer_indices, vi)
    end

    # Build position mapping: integers first, then continuous
    int_pos = 0
    cont_pos = 0
    n_int = count(vi -> vi in integer_indices, src_vars)

    dest.variable_map = Dict{MOI.VariableIndex, Int}()
    for vi in src_vars
        if vi in integer_indices
            int_pos += 1
            dest.variable_map[vi] = int_pos
        else
            cont_pos += 1
            dest.variable_map[vi] = n_int + cont_pos
        end
    end

    return index_map
end

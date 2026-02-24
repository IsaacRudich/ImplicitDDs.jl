"""
COMMENTS UP TO DATE
"""


"""
    get_coefficient_matrix(model::MathOptInterface.Utilities.GenericModel, int_var_order::Vector{MOI.VariableIndex}, cont_var_order::Vector{MOI.VariableIndex}, ::Type{T}, ::Type{U}) where {T<:AbstractFloat, U<:AbstractFloat}

Builds a dense coefficient matrix from a MOI model's linear constraints, normalizing everything as ≤ constraints.

# Arguments
- `model::MathOptInterface.Utilities.GenericModel`: A MOI model containing linear constraints
- `int_var_order::Vector{MOI.VariableIndex}`: Ordered vector of MOI.VariableIndex that defines the integer variable column order
- `cont_var_order::Vector{MOI.VariableIndex}`: Ordered vector of MOI.VariableIndex that defines the continuous variable column order
- `T::Type{<:AbstractFloat}`: Float type used in constraint coefficients
- `U::Type{<:AbstractFloat}`: Float type for the output matrices

# Returns
- `Tuple{Matrix{U}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Matrix{U}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Vector{U}}`: A tuple containing:
  1. `Matrix{U}`: Dense matrix for integer variable coefficients
  2. `Dict{Int,Vector{Int}}`: Mapping integer variable indices to positive coefficient row indices
  3. `Dict{Int,Vector{Int}}`: Mapping integer variable indices to negative coefficient row indices
  4. `Dict{Int,Vector{Int}}`: Mapping integer variable indices to zero coefficient row indices
  5. `Matrix{U}`: Dense matrix for continuous variable coefficients
  6. `Dict{Int,Vector{Int}}`: Mapping continuous variable indices to positive coefficient row indices
  7. `Dict{Int,Vector{Int}}`: Mapping continuous variable indices to negative coefficient row indices
  8. `Dict{Int,Vector{Int}}`: Mapping continuous variable indices to zero coefficient row indices
  9. `Vector{U}`: RHS vector
"""
function get_coefficient_matrix(model::MathOptInterface.Utilities.GenericModel, int_var_order::Vector{MOI.VariableIndex}, cont_var_order::Vector{MOI.VariableIndex}, ::Type{T}, ::Type{U}) where {T<:AbstractFloat, U<:AbstractFloat}
    leq_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}())
    eq_cons  = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}())
    geq_cons  = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}())
   
    #initalize the inequality matrix size
    n_rows = length(leq_cons) + length(geq_cons) + (2*length(eq_cons))
    n_int_cols = length(int_var_order)
    n_cont_cols = length(cont_var_order)
    # Create a dense matrix: columns 1..n_vars for variables coefficients,
    # Column n_vars+1 is for the RHS
    constraint_matrix_int_cols = zeros(U, n_rows, n_int_cols)
    constraint_matrix_cont_cols = zeros(U, n_rows, n_cont_cols)
    constraint_matrix_rhs_vector = zeros(U, n_rows)

    # Build a dictionary mapping each variable index to the rows it appears in
    int_var_to_pos_rows = Dict{Int, Vector{Int}}()
    int_var_to_neg_rows = Dict{Int, Vector{Int}}()
    int_var_to_zero_rows = Dict{Int, Vector{Int}}()
    
    @inbounds for i in 1:length(int_var_order)
        int_var_to_pos_rows[i] = Vector{Int}()
        int_var_to_neg_rows[i] = Vector{Int}()
        int_var_to_zero_rows[i] = Vector{Int}()
    end

    cont_var_to_pos_rows = Dict{Int, Vector{Int}}()
    cont_var_to_neg_rows = Dict{Int, Vector{Int}}()
    cont_var_to_zero_rows = Dict{Int, Vector{Int}}()
    
    @inbounds for i in 1:length(cont_var_order)
        cont_var_to_pos_rows[i] = Vector{Int}()
        cont_var_to_neg_rows[i] = Vector{Int}()
        cont_var_to_zero_rows[i] = Vector{Int}()
    end

    # Build a dictionary mapping each variable to its column index.
    int_var_to_col = Dict{MOI.VariableIndex, Int}()
    @inbounds for (i, var) in enumerate(int_var_order)
        int_var_to_col[var] = i
    end
    cont_var_to_col = Dict{MOI.VariableIndex, Int}()
    @inbounds for (i, var) in enumerate(cont_var_order)
        cont_var_to_col[var] = i
    end
    
    # Process each <= constraint
    if length(leq_cons) > 0
        @inbounds for (i, con) in enumerate(leq_cons)
            # Fill in the coefficients for each variable present in the constraint.
            f = MOI.get(model, MOI.ConstraintFunction(), con)
            for term in f.terms
                if haskey(int_var_to_col, term.variable)
                    var_index =  int_var_to_col[term.variable]
                    constraint_matrix_int_cols[i,var_index] = term.coefficient
                    if term.coefficient < 0
                        push!(int_var_to_neg_rows[var_index], i)
                    elseif term.coefficient > 0
                        push!(int_var_to_pos_rows[var_index], i)
                    end
                elseif haskey(cont_var_to_col, term.variable)
                    var_index =  cont_var_to_col[term.variable]
                    constraint_matrix_cont_cols[i,var_index] = term.coefficient
                    if term.coefficient < 0
                        push!(cont_var_to_neg_rows[var_index], i)
                    elseif term.coefficient > 0
                        push!(cont_var_to_pos_rows[var_index], i)
                    end
                end
            end
            
            # Store the RHS in column n_vars+1
            s = MOI.get(model, MOI.ConstraintSet(), con)
            constraint_matrix_rhs_vector[i] = s.upper - f.constant
        end
    end
    offset = length(leq_cons)
    # Process each >= constraint
    if length(geq_cons) > 0
        for (i, con) in enumerate(geq_cons)
            # Fill in the coefficients for each variable present in the constraint.
            f = MOI.get(model, MOI.ConstraintFunction(), con)
            for term in f.terms
                if haskey(int_var_to_col, term.variable)
                    var_index =  int_var_to_col[term.variable]
                    constraint_matrix_int_cols[i+offset, var_index] = -1 * term.coefficient
                    if term.coefficient > 0 #flipped because the actual term is flipped since its normalized to <=
                        push!(int_var_to_neg_rows[var_index], i+offset)
                    elseif term.coefficient < 0
                        push!(int_var_to_pos_rows[var_index], i+offset)
                    end
                elseif haskey(cont_var_to_col, term.variable)
                    var_index =  cont_var_to_col[term.variable]
                    constraint_matrix_cont_cols[i+offset, var_index] = -1 * term.coefficient
                    if term.coefficient > 0 #flipped because the actual term is flipped since its normalized to <=
                        push!(cont_var_to_neg_rows[var_index], i+offset)
                    elseif term.coefficient < 0
                        push!(cont_var_to_pos_rows[var_index], i+offset)
                    end
                end
            end
            
            # Store the RHS in column n_vars+1
            s = MOI.get(model, MOI.ConstraintSet(), con)
            constraint_matrix_rhs_vector[i+offset] = -1 * (s.lower - f.constant)
        end
    end

    #process each == constraint
    if length(eq_cons) > 0
        offset += length(geq_cons)
        for (i, con) in enumerate(eq_cons)
            #compute actual index since eq_cons uses 2 rows for each constraint
            actual_index = ((i-1)*2)+1
            # Fill in the coefficients for each variable present in the constraint.
            f = MOI.get(model, MOI.ConstraintFunction(), con)
            for term in f.terms
                if haskey(int_var_to_col, term.variable)
                    var_index =  int_var_to_col[term.variable]
                    constraint_matrix_int_cols[actual_index+offset,var_index] = term.coefficient
                    constraint_matrix_int_cols[actual_index+1+offset,var_index] = -1*term.coefficient
                    if term.coefficient < 0
                        push!(int_var_to_neg_rows[var_index], actual_index+offset)
                        push!(int_var_to_pos_rows[var_index], actual_index+1+offset)
                    elseif term.coefficient > 0
                        push!(int_var_to_pos_rows[var_index], actual_index+offset)
                        push!(int_var_to_neg_rows[var_index], actual_index+1+offset)
                    end
                elseif haskey(cont_var_to_col, term.variable)
                    var_index =  cont_var_to_col[term.variable]
                    constraint_matrix_cont_cols[actual_index+offset,var_index] = term.coefficient
                    constraint_matrix_cont_cols[actual_index+1+offset,var_index] = -1*term.coefficient
                    if term.coefficient < 0
                        push!(cont_var_to_neg_rows[var_index], actual_index+offset)
                        push!(cont_var_to_pos_rows[var_index], actual_index+1+offset)
                    elseif term.coefficient > 0
                        push!(cont_var_to_pos_rows[var_index], actual_index+offset)
                        push!(cont_var_to_neg_rows[var_index], actual_index+1+offset)
                    end
                end
            end
            
            # Store the RHS in column n_vars+1
            s = MOI.get(model, MOI.ConstraintSet(), con)
            constraint_matrix_rhs_vector[actual_index+offset] = s.value - f.constant
            constraint_matrix_rhs_vector[actual_index+1+offset] = -1*(s.value - f.constant)
        end
    end

    # Update var_to_zero_rows for each variable index: assign all row indexes
    # that are not already recorded in var_to_pos_rows or var_to_neg_rows.
    @inbounds for i in 1:length(int_var_order)
        used_rows = union(int_var_to_pos_rows[i], int_var_to_neg_rows[i])
        int_var_to_zero_rows[i] = setdiff(1:n_rows, used_rows)
    end
    @inbounds for i in 1:length(cont_var_order)
        used_rows = union(cont_var_to_pos_rows[i], cont_var_to_neg_rows[i])
        cont_var_to_zero_rows[i] = setdiff(1:n_rows, used_rows)
    end
    

    return constraint_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, constraint_matrix_cont_cols, cont_var_to_pos_rows, cont_var_to_neg_rows, cont_var_to_zero_rows, constraint_matrix_rhs_vector
end




"""
    get_objective_vector(model::MathOptInterface.Utilities.GenericModel, int_var_order::Vector{MOI.VariableIndex}, cont_var_order::Vector{MOI.VariableIndex}, ::Type{T}, ::Type{U}) where {T<:AbstractFloat, U<:AbstractFloat}

Gets the objective function as a vector, normalized for minimization.

# Arguments
- `model::MathOptInterface.Utilities.GenericModel`: An MOI model containing the optimization problem
- `int_var_order::Vector{MOI.VariableIndex}`: Ordered vector of MOI.VariableIndex that defines the integer variable column order
- `cont_var_order::Vector{MOI.VariableIndex}`: Ordered vector of MOI.VariableIndex that defines the continuous variable column order
- `T::Type{<:AbstractFloat}`: Float type used in MOI objective function
- `U::Type{<:AbstractFloat}`: Float type for the output vectors

# Returns
- `Tuple{Vector{U}, Vector{U}, U, Int}`: A tuple containing:
  1. `Vector{U}`: Objective coefficients for the integer variables
  2. `Vector{U}`: Objective coefficients for the continuous variables
  3. `U`: The objective constant term
  4. `Int`: Sense multiplier (1 for minimization, -1 for maximization)
"""
function get_objective_vector(model::MathOptInterface.Utilities.GenericModel, int_var_order::Vector{MOI.VariableIndex}, cont_var_order::Vector{MOI.VariableIndex}, ::Type{T}, ::Type{U}) where {T<:AbstractFloat, U<:AbstractFloat}
    int_obj_vec = zeros(U, length(int_var_order))
    cont_obj_vec = zeros(U, length(cont_var_order))

    # Build an internal dictionary for variable -> column lookup
    int_var_to_col = Dict{MOI.VariableIndex, Int}()
    cont_var_to_col = Dict{MOI.VariableIndex, Int}()

    for (i, var) in enumerate(int_var_order)
        int_var_to_col[var] = i
    end

    for (i, var) in enumerate(cont_var_order)
        cont_var_to_col[var] = i
    end

    # Extract the objective function
    obj_fun = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
    
    # Fill in the variable coefficients
    for term in obj_fun.terms
        if haskey(int_var_to_col, term.variable)
            int_obj_vec[int_var_to_col[term.variable]] = U(term.coefficient)
        elseif haskey(cont_var_to_col, term.variable)
            cont_obj_vec[cont_var_to_col[term.variable]] = U(term.coefficient)
        end
    end

    # The last entry stores the constant term
    obj_const = U(obj_fun.constant)
    
    original_sense = MOI.get(model, MOI.ObjectiveSense())
    sense_multiplier = 1
    if original_sense == MOI.MAX_SENSE
        int_obj_vec .*= -1
        cont_obj_vec .*= -1
        obj_const *= -1
        sense_multiplier = -1
    end

    return int_obj_vec, cont_obj_vec, obj_const, sense_multiplier
end
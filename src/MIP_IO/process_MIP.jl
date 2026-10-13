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
- `Tuple{Matrix{U}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Matrix{U}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Dict{Int,Vector{Int}}, Vector{U}, Int}`: A tuple containing:
  1. `Matrix{U}`: Dense matrix for integer variable coefficients
  2. `Dict{Int,Vector{Int}}`: Mapping integer variable indices to positive coefficient row indices
  3. `Dict{Int,Vector{Int}}`: Mapping integer variable indices to negative coefficient row indices
  4. `Dict{Int,Vector{Int}}`: Mapping integer variable indices to zero coefficient row indices
  5. `Matrix{U}`: Dense matrix for continuous variable coefficients
  6. `Dict{Int,Vector{Int}}`: Mapping continuous variable indices to positive coefficient row indices
  7. `Dict{Int,Vector{Int}}`: Mapping continuous variable indices to negative coefficient row indices
  8. `Dict{Int,Vector{Int}}`: Mapping continuous variable indices to zero coefficient row indices
  9. `Vector{U}`: RHS vector
  10. `Int`: Index of the first paired row (each == constraint and each two-sided constraint with both sides finite is stored as a pair of rows)
"""
function get_coefficient_matrix(model::MathOptInterface.Utilities.GenericModel, int_var_order::Vector{MOI.VariableIndex}, cont_var_order::Vector{MOI.VariableIndex}, ::Type{T}, ::Type{U}) where {T<:AbstractFloat, U<:AbstractFloat}
    leq_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}())
    eq_cons  = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}())
    geq_cons  = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}())
    interval_cons = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T}, MOI.Interval{T}}())

    # Two-sided constraints become a pair of rows when both sides are finite, and a single row when one side is
    paired_interval_cons = filter(con -> (s = MOI.get(model, MOI.ConstraintSet(), con); isfinite(s.lower) && isfinite(s.upper)), interval_cons)
    one_sided_interval_cons = filter(con -> (s = MOI.get(model, MOI.ConstraintSet(), con); isfinite(s.lower) != isfinite(s.upper)), interval_cons)

    #initalize the inequality matrix size
    n_rows = length(leq_cons) + length(geq_cons) + length(one_sided_interval_cons) + (2*length(eq_cons)) + (2*length(paired_interval_cons))
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
            write_constraint_row!(i, f, 1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)

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
            write_constraint_row!(i+offset, f, -1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)

            # Store the RHS in column n_vars+1
            s = MOI.get(model, MOI.ConstraintSet(), con)
            constraint_matrix_rhs_vector[i+offset] = -1 * (s.lower - f.constant)
        end
    end
    offset += length(geq_cons)

    # Process each two-sided constraint with one infinite side, as a single row for its finite side
    for (i, con) in enumerate(one_sided_interval_cons)
        f = MOI.get(model, MOI.ConstraintFunction(), con)
        s = MOI.get(model, MOI.ConstraintSet(), con)
        if isfinite(s.upper)
            write_constraint_row!(i+offset, f, 1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)
            constraint_matrix_rhs_vector[i+offset] = s.upper - f.constant
        else
            write_constraint_row!(i+offset, f, -1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)
            constraint_matrix_rhs_vector[i+offset] = -1 * (s.lower - f.constant)
        end
    end
    offset += length(one_sided_interval_cons)

    #process each == constraint
    if length(eq_cons) > 0
        for (i, con) in enumerate(eq_cons)
            #compute actual index since eq_cons uses 2 rows for each constraint
            actual_index = ((i-1)*2)+1
            # Fill in the coefficients for each variable present in the constraint.
            f = MOI.get(model, MOI.ConstraintFunction(), con)
            write_constraint_row!(actual_index+offset, f, 1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)
            write_constraint_row!(actual_index+1+offset, f, -1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)

            # Store the RHS in column n_vars+1
            s = MOI.get(model, MOI.ConstraintSet(), con)
            constraint_matrix_rhs_vector[actual_index+offset] = s.value - f.constant
            constraint_matrix_rhs_vector[actual_index+1+offset] = -1*(s.value - f.constant)
        end
    end
    offset += 2*length(eq_cons)

    # Process each two-sided constraint with both sides finite, as a pair of rows
    for (i, con) in enumerate(paired_interval_cons)
        actual_index = ((i-1)*2)+1
        f = MOI.get(model, MOI.ConstraintFunction(), con)
        write_constraint_row!(actual_index+offset, f, 1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)
        write_constraint_row!(actual_index+1+offset, f, -1, int_var_to_col, cont_var_to_col, constraint_matrix_int_cols, constraint_matrix_cont_cols, int_var_to_pos_rows, int_var_to_neg_rows, cont_var_to_pos_rows, cont_var_to_neg_rows)

        s = MOI.get(model, MOI.ConstraintSet(), con)
        constraint_matrix_rhs_vector[actual_index+offset] = s.upper - f.constant
        constraint_matrix_rhs_vector[actual_index+1+offset] = -1*(s.lower - f.constant)
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
    

    # == constraints and two-sided constraints with both sides finite are stored last, as pairs of rows
    first_paired_row = length(leq_cons) + length(geq_cons) + length(one_sided_interval_cons) + 1

    return constraint_matrix_int_cols, int_var_to_pos_rows, int_var_to_neg_rows, int_var_to_zero_rows, constraint_matrix_cont_cols, cont_var_to_pos_rows, cont_var_to_neg_rows, cont_var_to_zero_rows, constraint_matrix_rhs_vector, first_paired_row
end


"""
    write_constraint_row!(row_idx::Int, f::MOI.ScalarAffineFunction, sign::Int, int_var_to_col::Dict{MOI.VariableIndex, Int}, cont_var_to_col::Dict{MOI.VariableIndex, Int}, constraint_matrix_int_cols::Matrix{U}, constraint_matrix_cont_cols::Matrix{U}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, cont_var_to_pos_rows::Dict{Int, Vector{Int}}, cont_var_to_neg_rows::Dict{Int, Vector{Int}}) where {U<:AbstractFloat}

Writes the coefficients of `f` times `sign` into row `row_idx`, and records the row in the positive and negative coefficient row mappings.

# Arguments
- `row_idx::Int`: Row to write
- `f::MOI.ScalarAffineFunction`: Constraint function
- `sign::Int`: 1 to write f <= rhs, -1 to write a >= constraint as -f <= -rhs
- `int_var_to_col::Dict{MOI.VariableIndex, Int}`: Column of each integer variable
- `cont_var_to_col::Dict{MOI.VariableIndex, Int}`: Column of each continuous variable
- `constraint_matrix_int_cols::Matrix{U}`: Coefficient matrix for integer variables
- `constraint_matrix_cont_cols::Matrix{U}`: Coefficient matrix for continuous variables
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping integer variable indices to positive coefficient row indices
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping integer variable indices to negative coefficient row indices
- `cont_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping continuous variable indices to positive coefficient row indices
- `cont_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping continuous variable indices to negative coefficient row indices
"""
function write_constraint_row!(
    row_idx                     ::Int,
    f                           ::MOI.ScalarAffineFunction,
    sign                        ::Int,
    int_var_to_col              ::Dict{MOI.VariableIndex, Int},
    cont_var_to_col             ::Dict{MOI.VariableIndex, Int},
    constraint_matrix_int_cols  ::Matrix{U},
    constraint_matrix_cont_cols ::Matrix{U},
    int_var_to_pos_rows         ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows         ::Dict{Int, Vector{Int}},
    cont_var_to_pos_rows        ::Dict{Int, Vector{Int}},
    cont_var_to_neg_rows        ::Dict{Int, Vector{Int}}
) where {U<:AbstractFloat}
    for term in f.terms
        coefficient = sign * term.coefficient
        if haskey(int_var_to_col, term.variable)
            var_index = int_var_to_col[term.variable]
            constraint_matrix_int_cols[row_idx, var_index] = coefficient
            if coefficient < 0
                push!(int_var_to_neg_rows[var_index], row_idx)
            elseif coefficient > 0
                push!(int_var_to_pos_rows[var_index], row_idx)
            end
        elseif haskey(cont_var_to_col, term.variable)
            var_index = cont_var_to_col[term.variable]
            constraint_matrix_cont_cols[row_idx, var_index] = coefficient
            if coefficient < 0
                push!(cont_var_to_neg_rows[var_index], row_idx)
            elseif coefficient > 0
                push!(cont_var_to_pos_rows[var_index], row_idx)
            end
        end
    end
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
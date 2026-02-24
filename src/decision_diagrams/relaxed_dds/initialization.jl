"""
COMMENTS UP TO DATE
"""

"""
    compute_initial_infimum_gaps(coefficient_matrix_int_cols::Matrix{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, lower_bounds_int::Union{Vector{Z}, BitVector}, upper_bounds_int::Union{Vector{Z}, BitVector}, coefficient_matrix_cont_cols::Matrix{T}, cont_var_to_pos_rows::Dict{Int, Vector{Int}}, cont_var_to_neg_rows::Dict{Int, Vector{Int}}, lower_bounds_cont::Vector{T}, upper_bounds_cont::Vector{T}, coefficient_matrix_rhs_vector::Vector{T}) where {Z<:Integer, T<:Real}

Computes initial infimum gap values for each constraint row in a MIP model, normalized for <= constraints.

# Arguments
- `coefficient_matrix_int_cols::Matrix{T}`: Dense matrix where each row corresponds to a constraint and columns contain integer variable coefficients.
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: A mapping from each integer variable's index to the vector of row indices where the variable appears with a positive coefficient.
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: A mapping from each integer variable's index to the vector of row indices where the variable appears with a negative coefficient.
- `lower_bounds_int::Union{Vector{Z}, BitVector}`: Vector of lower bounds for the integer variables.
- `upper_bounds_int::Union{Vector{Z}, BitVector}`: Vector of upper bounds for the integer variables.
- `coefficient_matrix_cont_cols::Matrix{T}`: Dense matrix where each row corresponds to a constraint and columns contain continuous variable coefficients.
- `cont_var_to_pos_rows::Dict{Int, Vector{Int}}`: A mapping from each continuous variable's index to the vector of row indices where the variable appears with a positive coefficient.
- `cont_var_to_neg_rows::Dict{Int, Vector{Int}}`: A mapping from each continuous variable's index to the vector of row indices where the variable appears with a negative coefficient.
- `lower_bounds_cont::Vector{T}`: Vector of lower bounds for the continuous variables.
- `upper_bounds_cont::Vector{T}`: Vector of upper bounds for the continuous variables.
- `coefficient_matrix_rhs_vector::Vector{T}`: Vector containing the RHS values for each constraint.

# Returns
- `Tuple{Vector{T}, Vector{T}}`: A tuple containing:
  1. `Vector{T}`: Infimum gaps for each constraint row - computed by taking the RHS and subtracting the sum of the lower bound contributions for positive coefficients and the upper bound contributions for negative coefficients
  2. `Vector{T}`: Continuous variable contributions to the infimum gap for each constraint row
"""
function compute_initial_infimum_gaps(
    coefficient_matrix_int_cols  ::Matrix{T},
    int_var_to_pos_rows          ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows          ::Dict{Int, Vector{Int}},
    lower_bounds_int             ::Union{Vector{Z}, BitVector},
    upper_bounds_int             ::Union{Vector{Z}, BitVector},
    coefficient_matrix_cont_cols ::Matrix{T},
    cont_var_to_pos_rows         ::Dict{Int, Vector{Int}},
    cont_var_to_neg_rows         ::Dict{Int, Vector{Int}},
    lower_bounds_cont            ::Vector{T},
    upper_bounds_cont            ::Vector{T},
    coefficient_matrix_rhs_vector::Vector{T}
) where {Z<:Integer, T<:Real}
    #initialize infimums
    infimum_gaps = copy(coefficient_matrix_rhs_vector)
    cont_inf_gap_ctrbtns = zeros(T, length(coefficient_matrix_rhs_vector))

    @inbounds begin
        # iterate over the variables and calculate their impact on the infimum for the <= constraints
        for (i,var_coefficients) in enumerate(eachcol(coefficient_matrix_int_cols))
            #retrieve the positive and negative coefficient row indexes
            pos_coef_inds = int_var_to_pos_rows[i]
            neg_coef_inds = int_var_to_neg_rows[i]

            #retrieve the lower and upper bound of the variable
            lb = lower_bounds_int[i]
            ub = upper_bounds_int[i]

            #calculate the implications of the positive coefficients
            for row_ind in pos_coef_inds
                c = var_coefficients[row_ind]
                infimum_gaps[row_ind] -= lb*c
            end

            #calculate the implications of the negative coefficients
            for row_ind in neg_coef_inds
                c = var_coefficients[row_ind]
                infimum_gaps[row_ind] -= ub*c
            end
        end
        for (i,var_coefficients) in enumerate(eachcol(coefficient_matrix_cont_cols))
            #retrieve the positive and negative coefficient row indexes
            pos_coef_inds = cont_var_to_pos_rows[i]
            neg_coef_inds = cont_var_to_neg_rows[i]

            #retrieve the lower and upper bound of the variable
            lb = lower_bounds_cont[i]
            ub = upper_bounds_cont[i]

            #calculate the implications of the positive coefficients
            for row_ind in pos_coef_inds
                c = var_coefficients[row_ind]
                infimum_gaps[row_ind] -= lb*c
                cont_inf_gap_ctrbtns[row_ind] -= lb*c
            end

            #calculate the implications of the negative coefficients
            for row_ind in neg_coef_inds
                c = var_coefficients[row_ind]
                infimum_gaps[row_ind] -= ub*c
                cont_inf_gap_ctrbtns[row_ind] -= ub*c
            end
        end
    end

    return infimum_gaps, cont_inf_gap_ctrbtns
end





"""
    preallocate_infimum_gap_matrices(coefficient_matrix::Matrix{T}, numerical_precision::DataType, w::Int) where {T<:Real}

Preallocates a zero-initialized matrix for storing infimum gap values, organized for double buffering.

# Arguments
- `coefficient_matrix::Matrix{T}`: Matrix used to determine the number of constraints (rows) - only the row count is used
- `numerical_precision::DataType`: Numeric type (e.g., `Float64`) that specifies the precision for the preallocated matrix
- `w::Int`: Base width of a single buffer - the final matrix will have `w*2` columns for double buffering

# Returns
- `Matrix{<:Real}`: Zero-initialized matrix of dimensions `(size(coefficient_matrix, 1), w*2)` with rows equal to the number of constraints and `w*2` columns providing two buffers of width `w` for efficient alternating updates
"""
@inline function preallocate_infimum_gap_matrices(coefficient_matrix::Matrix{T}, numerical_precision::DataType, w::Int) where {T<:Real}
    return zeros(numerical_precision, size(coefficient_matrix, 1), w*2)
end








"""
    compute_initial_infimum_gap_matrix!(
        infimum_gap_matrices::Matrix{T}, infimum_gaps::Vector{T},
        lbs_int::Union{Vector{Z}, BitVector}, ubs_int::Union{Vector{Z}, BitVector},
        coefficient_matrix_int_cols::Matrix{T},
        int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}},
        node_matrix::Vector{NodeLayer{Z}}
    ) where {Z<:Integer, T<:Real}

Computes the updated infimum gap column for the value of each node in the first layer.

This function uses a baseline array of `infimum_gaps`, which includes the minimal contributions of *all* variables,
assuming each variable is set to the bound that minimizes each row. For the variable under consideration:

1. It adds the bound-based contribution (lower bound if the coefficient is positive, upper bound if negative)
   from `infimum_gaps` to form a partial sum that excludes this variable's effect.
2. For each node in `node_matrix[1]` (i.e., each possible integer value of the current variable), it subtracts
   `node.value * coefficient[row]` to the partial sum, storing the resulting row-wise values as a column in
   `infimum_gap_matrices`.

Hence, each column `i` of `infimum_gap_matrices` corresponds to infimum gap after choosing the `i`th node's integer value for this variable,
while all other variables remain at their bound-based minimal contribution in every row.

# Arguments
- `infimum_gap_matrices::Matrix{T}`: Each column of the left half will be filled with updated infimum gap values for one specific node state. Modified in-place.
- `infimum_gaps::Vector{T}`: A baseline vector where each entry represents the RHS minus the sum of bound-based contributions from *all* variables, including the variable we're focusing on. Must be computed beforehand.
- `lbs_int::Union{Vector{Z}, BitVector}`: The lower bounds for each variable. Only `lbs_int[1]` is used here.
- `ubs_int::Union{Vector{Z}, BitVector}`: The upper bounds for each variable. Only `ubs_int[1]` is used here.
- `coefficient_matrix_int_cols::Matrix{T}`: A matrix storing coefficients for all constraints (rows) and variables (columns). Only the first column is used here.
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Maps variable indices to the constraint row indices where their coefficient is positive. Only `int_var_to_pos_rows[1]` is used here.
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Maps variable indices to the constraint row indices where their coefficient is negative. Only `int_var_to_neg_rows[1]` is used here.
- `node_matrix::Vector{NodeLayer{Z}}`: An array of vectors storing discrete nodes for each variable. Only `node_matrix[1]` (the first variable's layer) is processed.
"""
@inline function compute_initial_infimum_gap_matrix!(
    infimum_gap_matrices::Matrix{T},
    infimum_gaps        ::Vector{T},
    lbs_int             ::Union{Vector{Z}, BitVector},
    ubs_int             ::Union{Vector{Z}, BitVector},
    coefficient_matrix_int_cols  ::Matrix{T},
    int_var_to_pos_rows ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows ::Dict{Int, Vector{Int}},
    node_matrix         ::Vector{NodeLayer{Z}}
) where {Z<:Integer, T<:Real}
    # Preallocate the offset vector
    alpha = copy(infimum_gaps)
    lb = lbs_int[1]
    ub = ubs_int[1]

    # Create views for the column corresponding to var_index
    col = view(coefficient_matrix_int_cols, :, 1)

    # compute entries for alpha
    @inbounds @simd for row_index in int_var_to_pos_rows[1]
        alpha[row_index] += lb * col[row_index]
    end
    @inbounds @simd for row_index in int_var_to_neg_rows[1]
        alpha[row_index] += ub * col[row_index]
    end

    #compute the initial infimum matrix
    first_layer = node_matrix[1]
    @inbounds for i in 1:first_layer.size
        @views infimum_gap_matrices[:, i] .= alpha .- (first_layer.values[i] * col)
    end
end


"""
    compute_infimum_gaps_for_qnode!(infimum_gap_matrices::Matrix{T}, base_gaps::Vector{T}, qnode::QueueNode{Z,U}, coefficient_matrix_int_cols::Matrix{T}, int_var_to_pos_rows::Dict{Int, Vector{Int}}, int_var_to_neg_rows::Dict{Int, Vector{Int}}, lbs_int::Vector{Z}, ubs_int::Vector{Z}, L::Int) where {Z<:Integer, T<:Real, U<:Real}

Computes the infimum gaps specific to a queue node's partial integer variable assignments.

This function adjusts the baseline infimum gaps to account for the fixed integer variable values in the queue node's path.
The baseline gaps assume all variables are set to their bound-based minimal contributions. For each fixed integer variable,
the function removes its bound-based contribution and adds its actual assigned value contribution.

# Arguments
- `infimum_gap_matrices::Matrix{T}`: Output matrix where the first column will be filled with updated infimum gap values. Modified in-place.
- `base_gaps::Vector{T}`: Baseline infimum gaps computed with all variables at their bound-based minimal contributions
- `qnode::QueueNode{Z,U}`: Queue node containing partial integer variable assignments in its path
- `coefficient_matrix_int_cols::Matrix{T}`: Constraint coefficient matrix for integer variables
- `int_var_to_pos_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with positive coefficients
- `int_var_to_neg_rows::Dict{Int, Vector{Int}}`: Mapping from integer variables to constraint rows with negative coefficients
- `lbs_int::Vector{Z}`: Lower bounds for integer variables
- `ubs_int::Vector{Z}`: Upper bounds for integer variables
- `L::Int`: Number of fixed integer variables in the queue node's path

# Algorithm
1. Copies baseline gaps into the first column of the infimum gap matrix
2. For each fixed integer variable in the queue node path:
   - Adds back its bound-based contribution (lower bound for positive coefficients, upper bound for negative coefficients)
   - Subtracts the actual assigned value contribution
3. Continuous variable contributions remain unchanged from the baseline gaps
"""
@inline function compute_infimum_gaps_for_qnode!(
    infimum_gap_matrices     ::Matrix{T},                         # output (mutated)
    base_gaps                ::Vector{T},                          # original infimum_gaps
    qnode                    ::QueueNode{Z,U},                     # the node whose gaps we want
    coefficient_matrix_int_cols::Matrix{T},
    int_var_to_pos_rows      ::Dict{Int, Vector{Int}},
    int_var_to_neg_rows      ::Dict{Int, Vector{Int}},
    lbs_int                  ::Vector{Z},
    ubs_int                  ::Vector{Z},
    L                        ::Int
) where {Z<:Integer, T<:Real, U<:Real}
    # 1) copy in the base
    @inbounds @simd for i in eachindex(base_gaps)
        infimum_gap_matrices[i, 1] = base_gaps[i]
    end

    # 2) for every fixed variable, add its bound‐contribution back into gaps
    # 3) for every fixed variable, subtract the assigned value
    @inbounds for j in 1:(L)
        lb, ub = lbs_int[j], ubs_int[j]
        val = qnode.path[j]
        @inbounds @simd for r in int_var_to_pos_rows[j]
            infimum_gap_matrices[r, 1] += (lb - val) * coefficient_matrix_int_cols[r, j]
        end
        @inbounds @simd for r in int_var_to_neg_rows[j]
            infimum_gap_matrices[r, 1] += (ub - val) * coefficient_matrix_int_cols[r, j]
        end
    end
end





"""
    preallocate_column_bound_matrices(w::Int, num_vars::Int, lbs::Union{Vector{Z}, BitVector}, ubs::Union{Vector{Z}, BitVector}, all_vars_binary::Bool = false) where {Z<:Integer}

Preallocates matrices for storing variable bound information during decision diagram construction.

# Arguments
- `w::Int`: Maximum width (number of nodes) for each decision diagram layer
- `num_vars::Int`: Number of integer variables in the problem
- `lbs::Union{Vector{Z}, BitVector}`: Lower bounds for each integer variable
- `ubs::Union{Vector{Z}, BitVector}`: Upper bounds for each integer variable
- `all_vars_binary::Bool`: Flag indicating whether all variables are binary (default: false)

# Returns
- `Tuple{Union{Matrix{Z}, BitMatrix}, Union{Matrix{Z}, BitMatrix}}`: A tuple containing:
  1. Lower bound matrix of dimensions `(w, num_vars)` - `BitMatrix` if all variables are binary, `Matrix{Z}` otherwise
  2. Upper bound matrix of dimensions `(w, num_vars)` - `BitMatrix` if all variables are binary, `Matrix{Z}` otherwise
  Each column is filled with the corresponding variable's bound value repeated across all rows.

# Algorithm
Creates two matrices where each row represents a potential node in a layer and each column represents a variable.
When `all_vars_binary` is true, uses memory-efficient `BitMatrix` representation. Otherwise uses standard `Matrix{Z}`.
Both matrices are initialized with the variable bounds repeated across all potential nodes for efficient access during
decision diagram construction algorithms.
"""
@inline function preallocate_column_bound_matrices(w::Int, num_vars::Int, lbs::Union{Vector{Z}, BitVector}, ubs::Union{Vector{Z}, BitVector}, all_vars_binary::Bool = false) where {Z<:Integer}
    if all_vars_binary
        lb_matrix = BitMatrix(undef, w, num_vars)
        ub_matrix = BitMatrix(undef, w, num_vars)
    else
        lb_matrix = Matrix{Z}(undef, w, num_vars)
        ub_matrix = Matrix{Z}(undef, w, num_vars)
    end

    @inbounds @simd for i in 1:num_vars
        lb_matrix[:, i] .= fill(lbs[i], w)
        ub_matrix[:, i] .= fill(ubs[i], w)
    end

    return lb_matrix, ub_matrix
end



"""
    generate_initial_node_layer!(int_obj_coeffs::Vector{T}, obj_const::T, domain::UnitRange, node_matrix::Vector{NodeLayer{Z}}, ltr_matrix::Matrix{T}) where{Z<:Integer, T<:Real}

Generates the initial layer of a decision diagram with one node per domain value for the first integer variable.

# Arguments
- `int_obj_coeffs::Vector{T}`: Objective coefficients for integer variables - only the first coefficient is used
- `obj_const::T`: Objective function constant term
- `domain::UnitRange`: Domain range for the first integer variable (e.g., `lbs_int[1]:ubs_int[1]`)
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers - the first layer will be populated with nodes
- `ltr_matrix::Matrix{T}`: Length-to-root matrix for tracking path costs - the first column will be filled with initial costs

# Algorithm
1. Creates one node in the first layer for each value in the domain range
2. Sets each node's value to the corresponding domain value with placeholder parent indices (-1, -1)
3. Computes the length-to-root cost for each node as `value * coefficient + constant`

# Notes
This function initializes the root layer of the decision diagram where each node represents choosing a specific
value for the first integer variable. The LTR values represent the objective contribution from this single variable assignment.
"""
@inline function generate_initial_node_layer!(int_obj_coeffs::Vector{T}, obj_const::T, domain::UnitRange, node_matrix::Vector{NodeLayer{Z}}, ltr_matrix::Matrix{T}) where{Z<:Integer, T<:Real}
    coeff = int_obj_coeffs[1]
    
    first_layer = node_matrix[1]
    @inbounds for (i, val) in enumerate(domain)
        add_node!(first_layer, -1, -1, val)
        ltr_matrix[i, 1] = (val*coeff)+obj_const
    end
end

"""
    generate_node_layer_from_queue!(qnode::QueueNode{Z,U}, node_matrix::Vector{NodeLayer{Z}}, ltr_matrix::Matrix{U}, layer_idx::Int) where {Z<:Integer, U<:Real}

Generates a decision diagram layer from a branch-and-bound queue node's pre-computed state.

# Arguments
- `qnode::QueueNode{Z,U}`: Queue node containing partial integer variable assignments and pre-computed length-to-root value
- `node_matrix::Vector{NodeLayer{Z}}`: Decision diagram layers - layers from `layer_idx` onward will be cleared and the target layer populated
- `ltr_matrix::Matrix{U}`: Length-to-root matrix - the first row at `layer_idx` will be set with the queue node's LTR value
- `layer_idx::Int`: Target layer index where the new node should be created

# Algorithm
1. Clears all decision diagram layers from `layer_idx` to the end to remove any previous state
2. Creates a single node in the target layer with the last integer value from the queue node's path
3. Sets the length-to-root value directly from the queue node's pre-computed LTR (includes all variable contributions)

# Notes
This function sets up DD construction to resume from a branch-and-bound queue node's partial state.
The queue node's LTR value already includes both integer and continuous variable contributions,
so no additional computations are needed.
"""
@inline function generate_node_layer_from_queue!(
    qnode      ::QueueNode{Z,U},
    node_matrix::Vector{NodeLayer{Z}},
    ltr_matrix ::Matrix{U},
    layer_idx  ::Int
) where {Z<:Integer, U<:Real}
    # 1) clear every NodeLayer from this layer onward
    #    (we assume each NodeLayer has fields `values`, `first_arcs`, `last_arcs`)
    @inbounds @simd for i in layer_idx:lastindex(node_matrix)
        empty_layer!(node_matrix[i])
    end

    # 2) add the next exact nodes into the target layer
    add_node!(
        node_matrix[layer_idx],
        -1,
        -1,
        last(qnode.path)
    )
    ltr_matrix[1, layer_idx] = qnode.ltr
end
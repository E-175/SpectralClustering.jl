# Fallback method
function compute_laplacian(W::AbstractMatrix, method::AbstractLaplacian)
    error("Laplacian method $(typeof(method)) is not implemented yet.")
end

# ---------------------------------------------------------
# TODO: Christoph (Unnormalized)
# ---------------------------------------------------------
function compute_laplacian(W::AbstractMatrix, ::UnnormalizedLaplacian)
    # TODO: Compute Degree matrix D
    # TODO: return L = D - W
    error("Laplacian method $(typeof(method)) is not implemented yet.")
end

"""
    compute_laplacian(W::AbstractMatrix, ::RandomWalkLaplacian)

Compute the random-walk normalized graph Laplacian from an affinity matrix `W`.

The random-walk normalized Laplacian is defined as

    L_rw = I - D⁻¹W

where `W` is the affinity matrix and `D` is the degree matrix.

This implementation is based on the normalized cuts formulation by Shi and Malik.
Their paper formulates the problem as the generalized eigenvalue problem

    (D - W)y = λDy

with

    L = D - W

Multiplying both sides by `D⁻¹` gives

    D⁻¹(D - W)y = λy

and therefore

    (I - D⁻¹W)y = λy

# Arguments
- `W`: Square symmetric affinity matrix. Entry `W[i, j]` stores the similarity between samples `i` and `j`.
- `::RandomWalkLaplacian`: Selects the random-walk normalized Laplacian.

# Returns
The random-walk normalized Laplacian matrix `L_rw`.

# Throws
- `ArgumentError` if `W` is not square.
- `ArgumentError` if `W` is not symmetric.
- `ArgumentError` if `W` contains negative affinity values.
- `ArgumentError` if at least one node has degree zero.
"""
function compute_laplacian(W::AbstractMatrix, ::RandomWalkLaplacian)

    # Get the number of rows and columns of the affinity matrix.
    # A valid affinity matrix must compare every sample with every other sample.
    n, m = size(W)

    # W must be square because the graph has one row and one column per sample.
    n == m || throw(ArgumentError("Affinity matrix W must be square."))

    # W must be symmetric because we model the similarities as an undirected graph.
    # This means that the similarity from i to j must equal the similarity from j to i.
    W ≈ W' || throw(ArgumentError("Affinity matrix W must be symmetric."))

    # Affinity values describe similarities.
    # Negative similarities are not valid for this graph Laplacian.
    any(<(0), W) && throw(ArgumentError("Affinity matrix W must not contain negative values."))

    # Compute the degree of each node.
    # The degree is the sum of all affinity values in one row.
    degrees = vec(sum(W, dims=2))

    # D⁻¹ is only defined if every degree is greater than zero.
    # A zero degree would mean that a node has no connection to the graph.
    all(degrees .> 0) || throw(ArgumentError("RandomWalkLaplacian is not defined for zero-degree nodes."))

    # Initialize the result matrix.
    # This matrix will contain L_rw = I - D⁻¹W.
    Lrw = zeros(Float64, n, n)

    # Build the matrix row by row.
    # For D⁻¹W, every row i of W is divided by the degree of node i.
    for i in 1:n
        # Add the identity matrix part I.
        Lrw[i, i] = 1.0

        # Subtract the normalized affinity values.
        # This gives L_rw[i, j] = I[i, j] - W[i, j] / degree[i].
        for j in 1:n
            Lrw[i, j] -= W[i, j] / degrees[i]
        end
    end

    return Lrw

end

function compute_laplacian(W::AbstractMatrix, ::SymmetricLaplacian)
    # TODO: Compute Degree matrix D
    # TODO: return L_sym = I - D^{-1/2} W D^{-1/2}
    n, m = size(W)

    n == m || throw(ArgumentError("Affinity matrix W must be square."))
    W ≈ W' || throw(ArgumentError("Affinity matrix W must be symmetric."))
    any(<(0), W) && throw(ArgumentError("Affinity matrix W must not contain negative values."))

    degrees = vec(sum(W, dims=2))
    all(degrees .> 0) || throw(ArgumentError("SymmetricLaplacian is not defined for zero-degree nodes."))


    DInverseSquareRoot = Diagonal(1 ./ sqrt.(degrees))
    L_sym = I - DInverseSquareRoot * W * DInverseSquareRoot
    return L_sym
end

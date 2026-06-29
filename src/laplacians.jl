# Fallback method
function compute_laplacian(W::AbstractMatrix, method::AbstractLaplacian)
    error("Laplacian method $(typeof(method)) is not implemented yet.")
end

"""
    function compute_laplacian(W::AbstractMatrix, ::UnnormalizedLaplacian)

Computes the unnormalized Laplacian of an input matrix `W`. 
The unnormalized Laplacian is defined as:
    L = D - W
with 
    D being the degree Matrix with the degree of a node being the sum of all the (affinity) values in its row, and
    W being the input Matrix (in our case the affinity Matrix) 

# Arguments
- `W`: Input matrix. In our case an affinity Matrix in which entry W[i,j] describes how similar points i and j are
    W has to be square and symmetric and must not contain any negative values. It can however contain zero degree nodes.
- `::UnnormalizedLaplacian`: Selects the unnormalized Laplacian.
    
# Returns
The unnormalized Laplacian `L`.

The specific type of the Laplacian depends on the type of the input matrix W. It will, however, always be a subtype of AbstractMatrix.
    In our implementation the function will always be called with an affinity matrix of type Matrix{Float64}.
    In such a case the output will also be of that type.

# Throws
- `ArgumentError` if `W` is not square.
- `ArgumentError` if `W` is not symmetric.
- `ArgumentError` if `W` contains negative values.
- Does accept a matrix with zero degree nodes

"""
function compute_laplacian(W::AbstractMatrix, ::UnnormalizedLaplacian)

    #Ensure that requirements for arguments are fulfilled
    #Ensure that W is square
    n,m = size(W)
    n == m || throw(ArgumentError("Affinity matrix W must be square."))
    #Ensure that W is symmetric
    W ≈ W' || throw(ArgumentError("Affinity matrix W must be symmetric."))
    #Ensure that W does not contain negative values
    any(<(0), W) && throw(ArgumentError("Affinity matrix W must not contain negative values."))


    #Calculate the degree of each node. The degree of a node is the sum of all the values in its row.
    degrees = vec(sum(W, dims=2))

    #Create the degree matrix
    D = Diagonal(degrees)

    #Calculate the unnormalized Laplacian as L = D - W
    L = D - W


    return L
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


"""
    function compute_laplacian(W::AbstractMatrix, ::SymmetricLaplacian)

    Computes the symmetric normalized Laplacian of an input matrix `W`. 

    The symmetric normalized Laplacian is defined as:

        L_sym = I - D^{-1/2} W D^{-1/2}

    with 

        D being the degree Matrix with the degree of a node being the sum of all the (affinity) values in its row and
            D^{-1/2} being the square root of its inverse, and

        W being the input Matrix (in our case the affinity Matrix) 

    
# Arguments
- `W`: Input matrix. In our case an affinity Matrix in which entry W[i,j] describes how similar points i and j are
    W has to be square and symmetric and must not contain any negative values. Furthermore it must not contain zero degree nodes.
- `::SymmetricLaplacian`: Selects the symmetric normalized Laplacian.
    


# Returns
The symmetric normalized Laplacian `L`.

The specific type of the Laplacian depends on the type of the input matrix W. It will, however, always be a subtype of AbstractMatrix.
    In our implementation the function will always be called with an affinity matrix of type Matrix{Float64}.
    In such a case the output will also be of that type.
    


# Throws
- `ArgumentError` if `W` is not square.
- `ArgumentError` if `W` is not symmetric.
- `ArgumentError` if `W` contains negative values.
- `ArgumentError` if `W` contains at least one zero degree node.

"""
function compute_laplacian(W::AbstractMatrix, ::SymmetricLaplacian)
    #Ensure that requirements for arguments are fulfilled
    #Ensure that W is square
    n, m = size(W)
    n == m || throw(ArgumentError("Affinity matrix W must be square."))
    #Ensure that W is symmetric
    W ≈ W' || throw(ArgumentError("Affinity matrix W must be symmetric."))
    #Ensure that W does not contain negative values
    any(<(0), W) && throw(ArgumentError("Affinity matrix W must not contain negative values."))
    #Calculate the degree of each node. The degree of a node is the sum of all the values in its row.
    degrees = vec(sum(W, dims=2))
    #Ensure that there are no zero degree nodes
    all(degrees .> 0) || throw(ArgumentError("SymmetricLaplacian is not defined for zero-degree nodes."))

    #Calculate D^{-1/2}, with D being the degree Matrix
    DInverseSquareRoot = Diagonal(1 ./ sqrt.(degrees))
    #Calculate the symmetric normalized Laplacian L_sym as L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = I - DInverseSquareRoot * W * DInverseSquareRoot
    return L_sym
end

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
    W has to be square and symmetric and must not contain and negative values. It can however contain zero degree nodes.
- `::UnnormalizedLaplacian`: Selects the unnormalized Laplacian.
    
# Returns
The unnormalized Laplacian `L`.

The specific type of the Laplacian depends on the type of the input matrix W. It will, however, always be a subtype of Matrix.
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
# ---------------------------------------------------------
# TODO: Carolin (Normalized Cuts)
# ---------------------------------------------------------
function compute_laplacian(W::AbstractMatrix, ::RandomWalkLaplacian)
    # TODO: Compute Degree matrix D
    # TODO: return L_rw = I - D^{-1}W
end

function compute_laplacian(W::AbstractMatrix, ::SymmetricLaplacian)
    # TODO: Compute Degree matrix D
    # TODO: return L_sym = I - D^{-1/2} W D^{-1/2}
end

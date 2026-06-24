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
    n,m = size(W)
    n == m || throw(ArgumentError("Affinity matrix W must be square."))
    W ≈ W' || throw(ArgumentError("Affinity matrix W must be symmetric."))
    any(<(0), W) && throw(ArgumentError("Affinity matrix W must not contain negative values."))

    degrees = vec(sum(W, dims=2))

    D = diagm(degrees)

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

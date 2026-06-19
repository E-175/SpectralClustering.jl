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
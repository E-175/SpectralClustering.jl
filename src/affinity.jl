using LinearAlgebra

"""
    compute_affinity_matrix(X; sigma=1.0)

Computes the RBF similarity matrix for an N x D data matrix `X`.
"""
function compute_affinity_matrix(X; sigma=1.0)
    n = size(X, 1)
    A = zeros(n, n)
    
    for i in 1:n
        for j in i:n
            dist_sq = sum((X[i, :] .- X[j, :]).^2)
            sim = exp(-dist_sq / (2 * sigma^2))
            A[i, j] = sim
            A[j, i] = sim # The affinity matrix is symmetric
        end
    end
    # Ensure self-affinity is zero if preferred, or left as 1.0 depending on your reference paper
    return A
end

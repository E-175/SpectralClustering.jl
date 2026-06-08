"""
    compute_affinity_matrix(X; sigma=1.0, self_affinity=0.0)

Compute the RBF affinity matrix for a data matrix `X`.

`X` is expected to have shape `n_features × n_samples`, meaning each column is one sample.

The affinity between two samples `i` and `j` is computed as

    exp(-||xᵢ - xⱼ||² / (2σ²))

# Keyword arguments
- `sigma`: Width parameter of the RBF kernel. Must be positive.
- `self_affinity`: Value used on the diagonal of the affinity matrix.

# Returns
A symmetric `n_samples × n_samples` affinity matrix.
"""
function compute_affinity_matrix(
    X::AbstractMatrix;
    sigma::Real=1.0,
    self_affinity::Real=0.0,
)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))

    n = size(X, 2)
    A = zeros(n, n)
    
    for i in 1:n
        A[i, i] = self_affinity
        
        for j in (i+1):n
            dist_sq = sum(abs2, X[:, i] .- X[:, j])
            sim = exp(-dist_sq / (2 * sigma^2))
            A[i, j] = sim
            A[j, i] = sim # The affinity matrix is symmetric
        end
    end
    
    return A
end

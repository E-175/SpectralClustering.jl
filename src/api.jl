"""
    spectral_cluster(X, k; affinity, laplacian, discretizer, rng=default_rng())

Perform spectral clustering on data matrix `X`.
`X` must have shape `n_features × n_samples`, with one sample per column.

If `k` is an integer, the algorithm computes a `k`-dimensional spectral embedding
and returns one cluster label per sample.

If `k` is `nothing`, `discretizer` must be `SelfTuningDiscretization()`. In that
case the discretizer determines the final number of clusters automatically, and
the number of eigenvectors computed upstream is capped by `discretizer.k_max`.

If an `rng` is provided, it is forwarded to the discretization step. This is
primarily relevant for discretizers with randomized behavior, such as
`KMeansDiscretization`.

# Arguments
- `X`: Data matrix of shape `n_features × n_samples`.
- `k`: Number of clusters, or `nothing` for self-tuning discretization.

# Keyword arguments
- `affinity`: Affinity configuration passed to `compute_affinity`.
- `laplacian`: Laplacian configuration passed to `compute_laplacian`.
- `discretizer`: Discretization configuration passed to `discretize`.
- `rng`: Random number generator forwarded to the discretization step.

# Returns
A `Vector{Int}` containing one cluster label per sample.

# Throws
- `ArgumentError` if `k` is an integer outside `1:size(X, 2)`.
- `ArgumentError` if `k` is `nothing` and `discretizer` is not `SelfTuningDiscretization()`.
- `ArgumentError` if `k` is `nothing` and fewer than two samples are available.
- `ArgumentError` propagated from `compute_affinity`, `compute_laplacian`,
  `compute_eigenvectors`, or `discretize` when their input requirements are not met.
"""
function spectral_cluster(X::AbstractMatrix, k::Union{Integer, Nothing}; 
                          affinity::AbstractAffinity = RBFKernel(),
                          laplacian::AbstractLaplacian = RandomWalkLaplacian(),
                          discretizer::AbstractDiscretization = KMeansDiscretization(false),
                          rng::AbstractRNG = default_rng())

    n_samples = size(X, 2)

    if isnothing(k)
        discretizer isa SelfTuningDiscretization || throw(ArgumentError("k can only be nothing when using SelfTuningDiscretization()."))
        n_samples >= 2 || throw(ArgumentError("k can only be nothing when at least two samples are available."))

        embedding_k = min(n_samples, discretizer.k_max)
    else
        1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of samples."))

        embedding_k = k
    end
    
    # 1. Build Similarity Graph
    W = compute_affinity(X, affinity)
    
    # 2. Build Graph Laplacian
    L = compute_laplacian(W, laplacian)
    
    # 3. Solve Eigendecomposition
    V = compute_eigenvectors(L, embedding_k)
    
    # 4. Discretize into labels
    labels = discretize(rng, V, discretizer, k=k)
    
    return labels
end
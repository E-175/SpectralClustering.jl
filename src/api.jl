"""
    spectral_cluster(X, k; affinity, laplacian, discretizer)

Perform spectral clustering on data matrix `X` into `k` clusters.
`X` should be `n_features × n_samples`.
"""
function spectral_cluster(X::AbstractMatrix, k::Int; 
                          affinity::AbstractAffinity = RBFKernel(),
                          laplacian::AbstractLaplacian = RandomWalkLaplacian(),
                          discretizer::AbstractDiscretization = KMeansDiscretization(false))
    
    # 1. Build Similarity Graph
    W = compute_affinity(X, affinity)
    
    # 2. Build Graph Laplacian
    L = compute_laplacian(W, laplacian)
    
    # 3. Solve Eigendecomposition
    V = compute_eigenvectors(L, k) 
    
    # 4. Discretize into labels
    labels = discretize(V, discretizer, k=k)
    
    return labels
end
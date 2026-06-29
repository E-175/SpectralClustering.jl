# Fallback method
function discretize(V::AbstractMatrix, method::AbstractDiscretization; k::Union{Int, Nothing}=nothing)
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

"""
    discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)

Discretize a spectral embedding into cluster labels using K-Means.

The input matrix `V` is the eigenvector embedding produced by spectral clustering.
Each column of `V` represents one sample, and each row represents one selected
eigenvector.

K-Means is applied to the columns of `V`, so samples with similar coordinates in
the spectral embedding are assigned to the same cluster.

If `method.normalize_samples` is `true`, each column of `V` is normalized to unit
length before K-Means is applied. This is used for variants such as the
Ng-Jordan-Weiss normalized spectral clustering method.

# Arguments
- `V`: Spectral embedding matrix with one column per sample (features × samples).
- `method`: K-Means discretization configuration.
- `k`: Number of clusters.

# Returns
A vector of cluster labels with one label per sample.

# Throws
- `ArgumentError` if `k` is not provided.
- `ArgumentError` if `k` is smaller than 1 or larger than the number of samples.
- `ArgumentError` if row normalization is requested and at least one row has norm zero.
"""
function discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)
    # The K-Means expects a fixed number of clusters.
    isnothing(k) && throw(ArgumentError("K-Means requires a specific number of clusters 'k'."))

    # The columns of V correspond to samples.
    # The rows of V correspond to selected eigenvectors.
    n_eigenvectors, n_samples = size(V)

    # k must be meaningful for the number of available samples.
    1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of samples."))

    # The embedding must contain at least one eigenvector.
    n_eigenvectors >= 1 || throw(ArgumentError("The spectral embedding must contain at least one eigenvector."))

    # Work on a floating-point copy so the input matrix is not modified.
    embedding = Matrix{Float64}(V)
    
    if method.normalize_samples
        # Compute the Euclidean norm of each column.
        # This measures the length of each embedded sample vector.
        col_norms = sqrt.(sum(abs2, embedding, dims=1))

        # A column with norm zero cannot be normalized.
        all(col_norms .> 0) || throw(ArgumentError("Cannot normalize columns with zero norm."))

        # Normalize every column to unit length.
        embedding ./= col_norms
    end
    
    # Clustering.kmeans expects data in the format features × samples.
    result = kmeans(embedding, k)

    # Return one cluster label per sample.
    return assignments(result)
end

# ---------------------------------------------------------
# TODO: Jens (Self-Tuning)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::SelfTuningDiscretization; k::Union{Int, Nothing}=nothing)
    # TODO: Implement discretization for self tuning
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

# ---------------------------------------------------------
# SVD Discretization (Yu & Shi 2003)
# ---------------------------------------------------------
"""
    discretize(V::AbstractMatrix, method::SVDDiscretization; k::Union{Int, Nothing}=nothing)

Discretize a spectral embedding into cluster labels using the optimal multiclass
discretization method by Yu and Shi (2003).

The method computes a discrete solution closest to the continuous optima in an
iterative fashion using singular value decomposition (SVD) and non-maximum suppression.

# Arguments
- `V`: Spectral embedding matrix with one column per sample (features × samples).
- `method`: SVD discretization configuration.
- `k`: Number of clusters.

# Returns
A vector of cluster labels with one label per sample.

# Throws
- `ArgumentError` if `k` is not provided.
- `ArgumentError` if `k` does not equal the number of eigenvectors (rows) in `V`.
- `ArgumentError` if `k` is smaller than 1 or larger than the number of samples.
"""
function discretize(V::AbstractMatrix, method::SVDDiscretization; k::Union{Int, Nothing}=nothing)
    isnothing(k) && throw(ArgumentError("SVD Discretization requires a specific number of clusters 'k'."))
    
    n_eigenvectors, n_samples = size(V)
    k == n_eigenvectors || throw(ArgumentError("SVD Discretization expects exactly `k` eigenvectors."))
    1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of samples."))

    # Z* is N x K. The input V is K x N.
    Z_star = Matrix{Float64}(V')
    
    # Normalization (Step 3)
    X_tilde_star = zeros(n_samples, k)
    row_norms = sqrt.(sum(abs2, Z_star, dims=2))
    for i in 1:n_samples
        if row_norms[i] > 0
            X_tilde_star[i, :] = Z_star[i, :] ./ row_norms[i]
        end
    end

    # Initialization of R* (Step 4)
    R_star = zeros(k, k)
    
    # Pick a fixed starting row (e.g. 1) for reproducibility instead of random
    initial_idx = 1
    R_star[:, 1] = X_tilde_star[initial_idx, :]
    
    c = zeros(n_samples)
    for j in 2:k
        c .+= abs.(X_tilde_star * R_star[:, j-1])
        idx = argmin(c)
        R_star[:, j] = X_tilde_star[idx, :]
    end

    # Convergence variables
    phi_star = 0.0
    X_star = zeros(Int, n_samples, k)
    labels = zeros(Int, n_samples)
    
    # Iterative Refinement (Steps 6-8)
    while true
        # Rotate continuous optima
        X_tilde = X_tilde_star * R_star
        
        # Non-maximum suppression (Step 6)
        fill!(X_star, 0)
        for i in 1:n_samples
            l = argmax(X_tilde[i, :])
            labels[i] = l
            X_star[i, l] = 1
        end
        
        # SVD of X*^T \\tilde{X}* (Step 7)
        svd_result = svd(X_star' * X_tilde_star)
        
        # Check convergence
        phi = sum(svd_result.S)
        if abs(phi - phi_star) < eps(Float64)
            break
        end
        
        phi_star = phi
        
        # Update R* = \\tilde{U} U^T
        # Note: svd in Julia returns U, S, V where M = U * Diagonal(S) * V'
        # Yu & Shi formula M = U \\Omega \\tilde{U}^T.
        # Thus, their U is our U, their \\tilde{U} is our V.
        # R* = \\tilde{U} U^T = V * U'
        R_star = svd_result.V * svd_result.U'
    end
    
    return labels
end

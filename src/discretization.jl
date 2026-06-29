using LinearAlgebra: I
using Optim: optimize, BFGS, minimizer
using ForwardDiff: gradient!

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

    if !isnothing(method.seed)
        Random.seed!(method.seed)
    end

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
    

    # Manual implementation of KMeans
    if method.use_manual_implementation
        timeout = 1000
        currentCentroids = V[:,randperm(size(V,2))[1:k]]
        nearestCentroids = zeros(Int,size(V,2))
        for _ in 1:timeout
            # Currently uses euclidian distance / L2 Norm
            distancesMatrix = [norm(V[:,x] - currentCentroids[:,y]) for x = 1:size(V,2), y = 1:k]
            newNearestCentroids = vec([i[2] for i in argmin(distancesMatrix,dims=2)])
            if newNearestCentroids == nearestCentroids
                break
            else
                nearestCentroids = newNearestCentroids
            end
            # Move Centroid of Cluster
            for currentClusterIndex in 1:k
                currentCluster = V[:,nearestCentroids .== currentClusterIndex]
                if isempty(currentCluster)
                    currentCentroids[:,currentClusterIndex] = V[:,rand(1:size(V,2))]
                else
                    currentCentroids[:,currentClusterIndex] = vec(mean(currentCluster,dims=2))
                end
            end
        end
        return nearestCentroids
    end

    # Clustering.kmeans expects data in the format features × samples.
    result = kmeans(embedding, k)

    # Return one cluster label per sample.
    return assignments(result)
end

# ---------------------------------------------------------
# TODO: Jens (Self-Tuning)
# ---------------------------------------------------------

# ---------------------------------------------------------
# AI GENERATED
# ---------------------------------------------------------
"""
    discretize(V, method::SelfTuningDiscretization; k=nothing)

Discretize the continuous eigenvectors into cluster labels using the Self-Tuning approach.

Instead of using K-means, this method finds an optimal rotation matrix `R` to align the 
eigenvectors with the canonical coordinate system. It assigns cluster labels based on the 
maximum absolute value in each row of the rotated matrix `Z = V * R`.

If `k` is not provided, the algorithm automatically determines the optimal number of clusters 
by evaluating the alignment cost for different numbers of clusters (from 2 up to the number of columns in `V`)
and selecting the one that minimizes the cost.

`V` is expected to have shape `n_samples × max_clusters`, containing the top eigenvectors 
of the normalized affinity matrix.

# Keyword arguments
- `k`: Optional integer specifying the exact number of clusters. If `nothing` (default), the algorithm self-tunes to find the optimal number of clusters.

# Returns
A `Vector{Int}` of length `n_samples` containing the assigned cluster labels.
"""
function discretize(V::AbstractMatrix, method::SelfTuningDiscretization; k::Union{Int, Nothing}=nothing)
    n_samples, max_clusters = size(V)
    
    # Case 1: The user provided a specific 'k' (No self-tuning needed for number of clusters)
    if !isnothing(k)
        if k > max_clusters
            throw(ArgumentError("k cannot be larger than the number of eigenvectors provided in V"))
        end
        V_subset = V[:, 1:k]
        Z, _ = optimize_rotation(V_subset)
        return get_cluster_assignments(Z)
    end
    
    # Case 2: Self-Tuning (Find the optimal 'k' automatically)
    best_cost = Inf
    best_k = 2
    best_Z = V[:, 1:2]
    
    # The paper suggests checking all possible cluster numbers up to max_clusters
    for current_k in 2:max_clusters
        V_subset = V[:, 1:current_k]
        
        # Optimize rotation for this specific number of clusters
        Z, cost = optimize_rotation(V_subset)
        
        # If this number of clusters provides a better (lower) cost, save it
        if cost < best_cost
            best_cost = cost
            best_k = current_k
            best_Z = Z
        end
    end
    
    # Once we have found the best k and its rotated Z, assign the final labels
    return get_cluster_assignments(best_Z)
end

# ---------------------------------------------------------
# Helper 1: The Cost Function (Equation 3 from the paper)
# ---------------------------------------------------------
"""
Computes the alignment cost J = sum(Z_ij^2 / M_i^2)
where M_i is the maximum absolute value in row i of Z.
"""
function calculate_alignment_cost(Z::AbstractMatrix)
    n_samples, c_clusters = size(Z)
    cost = zero(eltype(Z)) # Use zero(eltype) to support AutoDiff Dual numbers
    
    for i in 1:n_samples
        # Find the maximum squared value in the row (M_i^2)
        # We use eps() to prevent division by zero in edge cases
        max_val_sq = maximum(abs2, Z[i, :]) + eps(Float64)
        
        for j in 1:c_clusters
            cost += (Z[i, j]^2) / max_val_sq
        end
    end
    
    return cost
end

# ---------------------------------------------------------
# Helper 1.5: Givens Rotation Builder
# ---------------------------------------------------------
"""
Builds a c x c orthogonal rotation matrix from a vector of angles (thetas)
using Givens rotations.
"""
function make_rotation_matrix(thetas::AbstractVector{T}, c::Int) where T
    R = Matrix{T}(I, c, c)
    k = 1
    for i in 1:(c-1)
        for j in (i+1):c
            theta = thetas[k]
            c_theta = cos(theta)
            s_theta = sin(theta)
            
            # Apply Givens rotation in the (i, j) plane
            G = Matrix{T}(I, c, c)
            G[i, i] = c_theta
            G[j, j] = c_theta
            G[i, j] = s_theta
            G[j, i] = -s_theta
            
            R = R * G
            k += 1
        end
    end
    return R
end

# ---------------------------------------------------------
# Helper 2: The Gradient Descent (Appendix A from the paper)
# ---------------------------------------------------------
"""
Finds the optimal rotation matrix R that aligns the columns of V_subset.
Returns the rotated matrix Z and its alignment cost.
"""
function optimize_rotation(V_subset::AbstractMatrix)
    n_samples, c_clusters = size(V_subset)
    
    # Base case: if there's only 1 cluster, no rotation is needed
    if c_clusters == 1
        return V_subset, calculate_alignment_cost(V_subset)
    end
    
    # We need C(C-1)/2 angles for a full rotation matrix in C dimensions
    num_thetas = div(c_clusters * (c_clusters - 1), 2)
    initial_thetas = zeros(num_thetas)
    
    # The objective function to minimize
    function objective(thetas)
        R = make_rotation_matrix(thetas, c_clusters)
        Z = V_subset * R
        return calculate_alignment_cost(Z)
    end
    
    # Use Optim.jl to find the optimal angles via Automatic Differentiation
    g!(G, thetas) = gradient!(G, objective, thetas)
    result = optimize(objective, g!, initial_thetas, BFGS())
    
    best_thetas = minimizer(result)
    R_opt = make_rotation_matrix(best_thetas, c_clusters)
    
    Z = V_subset * R_opt 
    cost = calculate_alignment_cost(Z)
    
    return Z, cost
end

# ---------------------------------------------------------
# Helper 3: Final Non-Maximum Suppression
# ---------------------------------------------------------
"""
Takes the optimally rotated matrix Z and assigns cluster labels.
A point is assigned to cluster c if the maximum value in its row is at column c.
"""
function get_cluster_assignments(Z::AbstractMatrix)
    n_samples = size(Z, 1)
    labels = zeros(Int, n_samples)
    
    for i in 1:n_samples
        # Find the index of the maximum squared value in the row
        _, max_idx = findmax(abs2.(Z[i, :]))
        labels[i] = max_idx
    end
    
    return labels
end

# ---------------------------------------------------------
# SVD Discretization (Yu & Shi 2003) - AI Generated
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

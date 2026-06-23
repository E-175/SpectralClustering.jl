using LinearAlgebra: I
using Optim: optimize, BFGS, minimizer

# Fallback method
function discretize(V::AbstractMatrix, method::AbstractDiscretization; k::Union{Int, Nothing}=nothing)
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

# ---------------------------------------------------------
# TODO: Christoph & Carolin (Standard & Normalized)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)
    if isnothing(k)
        error("K-Means requires a specific number of clusters 'k'.")
    end
    
    if method.normalize_rows
        # TODO for Carolin: Normalize rows of V to unit length
    end
    
    # TODO: Apply K-Means clustering algorithm on rows of V
    # return labels
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
# Helper 1: The Cost Function (Equation 3 from the paper) | AI GENERATED
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
# Helper 1.5: Givens Rotation Builder | AI GENERATED
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
# Helper 2: The Gradient Descent (Appendix A from the paper) | AI GENERATED
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
    result = optimize(objective, initial_thetas, BFGS(), autodiff=:forward)
    
    best_thetas = minimizer(result)
    R_opt = make_rotation_matrix(best_thetas, c_clusters)
    
    Z = V_subset * R_opt 
    cost = calculate_alignment_cost(Z)
    
    return Z, cost
end

# ---------------------------------------------------------
# Helper 3: Final Non-Maximum Suppression | AI GENERATED
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
# TODO: Janus (Multi-Class)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::SVDDiscretization; k::Union{Int, Nothing}=nothing)
    if isnothing(k)
        error("SVD Discretization requires a specific number of clusters 'k'.")
    end
    # TODO: Implement.
    # return labels
end
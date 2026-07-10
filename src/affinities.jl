# Fallback method
function compute_affinity(X::AbstractMatrix, method::AbstractAffinity; self_affinity::Real=0)
    error("Affinity method $(typeof(method)) is not implemented yet.")
end


"""
    compute_affinity(X, method::RBFKernel; self_affinity=0.0)

Compute the RBF affinity matrix. The affinity between two samples `i` and `j` is 
computed as:

    exp(-||xᵢ - xⱼ||² / (2σ²))

where `σ` is defined in the `RBFKernel` struct.

# Arguments
- `X`: Data matrix with shape `n_features × n_samples`. Each column is one sample.
- `method`: RBF kernel configuration containing the width parameter `sigma`.
- `self_affinity`: Value used on the diagonal of the affinity matrix.

# Returns
A symmetric `n_samples × n_samples` affinity matrix.
"""
function compute_affinity(X::AbstractMatrix, method::RBFKernel; self_affinity::Real=0)
    Base.require_one_based_indexing(X)
    
    sigma = method.sigma

    sigma > 0 || throw(ArgumentError("sigma must be positive"))

    # find type T that can hold values from all of these inputs without loss
    T = float(promote_type(eltype(X), typeof(sigma), typeof(self_affinity)))
    sigma = convert(T, sigma)
    self_affinity = convert(T, self_affinity)
    scale = convert(T, 2) * sigma^2

    n = size(X, 2)
    A = zeros(T, n, n)
    
    for i in 1:n
        A[i, i] = self_affinity
        
        for j in (i+1):n
            dist_sq = zero(T)

            @inbounds for r in axes(X, 1)
                difference = X[r, i] - X[r, j]
                dist_sq += abs2(difference)
            end

            sim = exp(-dist_sq / scale)
            A[i, j] = sim
            A[j, i] = sim # The affinity matrix is symmetric
        end
    end
    
    return A
end

"""
    compute_affinity(X, method::LocalScaling; self_affinity=0.0)

Compute the local Scaling affinity matrix. The affinity between two samples `i` and `j` is 
computed as:

    exp(-d²(xᵢ,xⱼ) / σᵢσⱼ)

where `σᵢ` is determined by measuring the distance from point sᵢ to its K-th nearest neighbor.

`X` is expected to have shape `n_features × n_samples`, meaning each column is one sample.

# Keyword arguments
- `self_affinity`: Value used on the diagonal of the affinity matrix. Has to be 0.

# Returns
A symmetric `n_samples × n_samples` affinity matrix.
"""
function compute_affinity(X::AbstractMatrix, method::LocalScaling; self_affinity::Real=0)
    Base.require_one_based_indexing(X)

    T = float(eltype(X))
    
    self_affinity = zero(T) # self_affinity has to be 0
    k_neighbor = method.k
    n = size(X, 2)
    
    # Step 1: Precompute all pairwise squared distances
    dist_sq = zeros(T, n, n)

    for i in 1:n
        for j in (i+1):n
            distance = zero(eltype(dist_sq))

            @inbounds for r in axes(X, 1)
                difference = X[r, i] - X[r, j]
                distance += abs2(difference)
            end

            dist_sq[i, j] = distance
            dist_sq[j, i] = distance
        end
    end
    
    # Step 2: Compute local scale (sigma) for each point
    sigmas = zeros(T, n)
    @views for i in 1:n
        # Copy the squared distances from point i to all other points.
        column_distances = copy(view(dist_sq, :, i))
        
        # Index is k+1 because the 1st element is always the distance to itself (0.0)
        neighbor_idx = min(k_neighbor + 1, n)

        # Find the k-th nearest neighbor using squared distances.
        kth_dist_sq = partialsort!(column_distances, neighbor_idx)

        # Store the actual distance as sigma.
        sigmas[i] = sqrt(kth_dist_sq)
        
        # Safety check: avoid division by zero if duplicate points exist
        if iszero(sigmas[i])
            sigmas[i] = eps(T)
        end
    end
    
    # Step 3: Construct the final affinity matrix W
    W = zeros(T, n, n)
    for i in 1:n
        W[i,i]  = self_affinity
        for j in (i+1):n
            # Using the exact formula from the paper: exp(-d^2 / (sigma_i * sigma_j))
            sim = exp(-dist_sq[i, j] / (sigmas[i] * sigmas[j]))
            W[i, j] = sim
            W[j,i] = sim
        end
    end
    
    return W
end

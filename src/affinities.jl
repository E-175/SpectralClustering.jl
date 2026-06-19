# Fallback method
function compute_affinity(X::AbstractMatrix, method::AbstractAffinity; self_affinity::Real=0.0)
    error("Affinity method $(typeof(method)) is not implemented yet.")
end


"""
    compute_affinity(X, method::RBFKernel; self_affinity=0.0)

Compute the RBF affinity matrix. The affinity between two samples `i` and `j` is 
computed as:

    exp(-||xᵢ - xⱼ||² / (2σ²))

where `σ` is defined in the `RBFKernel` struct.

`X` is expected to have shape `n_features × n_samples`, meaning each column is one sample.

# Keyword arguments
- `self_affinity`: Value used on the diagonal of the affinity matrix.

# Returns
A symmetric `n_samples × n_samples` affinity matrix.
"""
function compute_affinity(X::AbstractMatrix, method::RBFKernel; self_affinity::Real=0.0)
    sigma = method.sigma

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

# ---------------------------------------------------------
# TODO: Jens (Self-Tuning)
# ---------------------------------------------------------
function compute_affinity(X::AbstractMatrix, method::LocalScaling; self_affinity::Real=0.0)
    k_neighbor = method.k
    # TODO: Implement self-tuning affinity matrix
    
    # return W
end

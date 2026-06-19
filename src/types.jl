# ABSTRACT TYPES
abstract type AbstractAffinity end
abstract type AbstractLaplacian end
abstract type AbstractDiscretization end

# AFFINITIES
"""
    RBFKernel(; sigma=1.0)
Gaussian kernel for affinity. `sigma` must be positive.
"""
Base.@kwdef struct RBFKernel <: AbstractAffinity 
    sigma::Float64 = 1.0
end

"""
    LocalScaling(k::Int)
Local scaling affinity where scale depends on the `k`-th nearest neighbor.
"""
struct LocalScaling <: AbstractAffinity 
    k::Int 
end


# LAPLACIANS
""" Unnormalized Graph Laplacian: L = D - W """
struct UnnormalizedLaplacian <: AbstractLaplacian end

""" Random Walk Normalized Laplacian: L_rw = I - D^{-1}W """
struct RandomWalkLaplacian <: AbstractLaplacian end

""" Symmetric Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2} """
struct SymmetricLaplacian <: AbstractLaplacian end

# DISCRETIZATION
""" 
    KMeansDiscretization(normalize_rows::Bool)
If `normalize_rows` is true, eigenvectors are row-normalized before K-Means.
"""
struct KMeansDiscretization <: AbstractDiscretization 
    normalize_rows::Bool
end

""" Zelnik-Manor & Perona method (Automatic k, cost minimization) """
struct SelfTuningDiscretization <: AbstractDiscretization end

""" Yu & Shi method (Optimal multiclass using SVD) """
struct SVDDiscretization <: AbstractDiscretization end
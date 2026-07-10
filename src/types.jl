# ABSTRACT TYPES
"""
    AbstractAffinity

Abstract supertype for affinity configurations used by `compute_affinity`.
"""
abstract type AbstractAffinity end

"""
    AbstractLaplacian

Abstract supertype for Laplacian configurations used by `compute_laplacian`.
"""
abstract type AbstractLaplacian end

"""
    AbstractDiscretization

Abstract supertype for discretization configurations used by `discretize`.
"""
abstract type AbstractDiscretization end

# AFFINITIES
"""
    RBFKernel(; sigma=1.0)

Gaussian affinity kernel with bandwidth `sigma`.
`compute_affinity` requires `sigma > 0`.
"""
Base.@kwdef struct RBFKernel{T<:Real} <: AbstractAffinity
    sigma::T = 1.0
end

"""
    LocalScaling(k::Integer)

Local-scaling affinity where each sample scale is derived from its `k`-th nearest
neighbor distance.
"""
struct LocalScaling{T<:Integer} <: AbstractAffinity 
    k::T
end


# LAPLACIANS
"""
    UnnormalizedLaplacian()

Configuration for the unnormalized graph Laplacian `L = D - W`.
"""
struct UnnormalizedLaplacian <: AbstractLaplacian end

"""
    RandomWalkLaplacian()

Configuration for the random-walk normalized Laplacian `L_rw = I - D^{-1}W`.
"""
struct RandomWalkLaplacian <: AbstractLaplacian end

"""
    SymmetricLaplacian()

Configuration for the symmetric normalized Laplacian
`L_sym = I - D^{-1/2} W D^{-1/2}`.
"""
struct SymmetricLaplacian <: AbstractLaplacian end

# DISCRETIZATION
""" 
    KMeansDiscretization(normalize_samples::Bool)
    KMeansDiscretization(normalize_samples::Bool, use_manual_implementation::Bool)

Discretize a spectral embedding with K-Means.
If `normalize_samples` is `true`, each sample vector is normalized before clustering.
`use_manual_implementation` selects the package implementation (`false`) or the
manual implementation (`true`).
"""
struct KMeansDiscretization <: AbstractDiscretization 
    normalize_samples::Bool
    use_manual_implementation::Bool
end

function KMeansDiscretization(normalize_samples::Bool)
    return KMeansDiscretization(normalize_samples,false)
end

"""
    SelfTuningDiscretization(; k_max=10)

Self-tuning discretization based on the Zelnik-Manor and Perona method.
When used with `spectral_cluster(...; k=nothing)`, `k_max` bounds the largest
number of clusters considered during automatic model selection.
"""
Base.@kwdef struct SelfTuningDiscretization <: AbstractDiscretization 
    k_max::Int = 10
end

"""
    SVDDiscretization()

Discretization method based on the Yu and Shi multiclass SVD formulation.
"""
struct SVDDiscretization <: AbstractDiscretization end
module SpectralClustering


using LinearAlgebra: eigen, Diagonal, I, svd, norm, Symmetric, issymmetric
using Clustering: kmeans, assignments
using Random: MersenneTwister, AbstractRNG, randn, default_rng, randperm
using Statistics: mean
using Optim: optimize, BFGS, minimizer
using ForwardDiff: gradient!

# Export Types
export AbstractAffinity, AbstractLaplacian, AbstractDiscretization
export RBFKernel, LocalScaling
export UnnormalizedLaplacian, RandomWalkLaplacian, SymmetricLaplacian
export KMeansDiscretization, SelfTuningDiscretization, SVDDiscretization

# Export API and Data Generators
export make_circles, make_moons, make_blobs
export compute_affinity, spectral_cluster, compute_laplacian, compute_eigenvectors, discretize

# Include Sub-files
include("types.jl")
include("affinities.jl")
include("laplacians.jl")
include("eigensolvers.jl")
include("discretization.jl")
include("datasets.jl")
include("api.jl")

end
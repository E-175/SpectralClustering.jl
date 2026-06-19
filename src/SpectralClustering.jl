module SpectralClustering

using LinearAlgebra

# Export Types
export AbstractAffinity, AbstractLaplacian, AbstractDiscretization
export RBFKernel, LocalScaling
export UnnormalizedLaplacian, RandomWalkLaplacian, SymmetricLaplacian
export KMeansDiscretization, SelfTuningDiscretization, SVDDiscretization

# Export API and Data Generators
export make_circles, make_moons, make_blobs
export compute_affinity

# Include Sub-files
include("types.jl")
include("affinities.jl")
include("laplacians.jl")
include("eigensolvers.jl")
include("discretization.jl")
include("datasets.jl")
include("api.jl")

end

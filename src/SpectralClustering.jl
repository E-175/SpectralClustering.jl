module SpectralClustering

include("DataGeneration.jl")
include("affinity.jl")

export make_circles, make_moons, make_blobs
export compute_affinity_matrix

end

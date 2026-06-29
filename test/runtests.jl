using SpectralClustering
using Test

@testset verbose=true "SpectralClustering.jl" begin
    include("test_datasets.jl")
    include("test_affinity.jl")
    include("test_discretization.jl")
    include("test_laplacian.jl")
end

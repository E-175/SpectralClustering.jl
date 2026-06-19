using SpectralClustering
using Test

@testset verbose=true "SpectralClustering.jl" begin
    include("test_datasets.jl")
    include("test_affinity.jl")
end

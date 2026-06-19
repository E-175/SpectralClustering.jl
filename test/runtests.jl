using SpectralClustering
using Test

@testset "SpectralClustering.jl" begin
    include("DataGenerationTests.jl")
    include("AffinityTests.jl")
end

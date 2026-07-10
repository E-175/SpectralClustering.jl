using SpectralClustering
using Test

@testset verbose=true "SpectralClustering.jl" begin
    include("test_datasets.jl")
    include("test_affinity.jl")
    include("test_api.jl")
    include("test_discretization.jl")
    include("test_eigensolvers.jl")
    include("test_laplacians.jl")
    include("test_self_tuning.jl")
    include("test_aqua.jl")
    include("test_indexing.jl")

end

using SpectralClustering
using Test

@testset verbose=true "SpectralClustering.jl" begin
    include("test_datasets.jl")
    include("test_affinity.jl")
<<<<<<< HEAD
    include("test_self_tuning.jl")
=======
    include("test_discretization.jl")
>>>>>>> beab6e2 (feat: SVD Discretization implemented)
end

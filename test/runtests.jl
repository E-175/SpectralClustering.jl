using SpectralClustering
using Test

@testset verbose=true "SpectralClustering.jl" begin
    include("test_datasets.jl")
    include("test_affinity.jl")
<<<<<<< HEAD
<<<<<<< HEAD
    include("test_self_tuning.jl")
=======
    include("test_discretization.jl")
>>>>>>> beab6e2 (feat: SVD Discretization implemented)
=======
    include("test_laplacian.jl")
>>>>>>> 2e4bfa5 (Add some tests)
end

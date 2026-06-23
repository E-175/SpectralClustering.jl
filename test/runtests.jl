using SpectralClustering
using Test

@testset verbose=true "SpectralClustering.jl" begin
    include("test_datasets.jl")
    include("test_affinity.jl")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    include("test_self_tuning.jl")
=======
=======
>>>>>>> 81a6ea7 (tests)
    include("test_discretization.jl")
>>>>>>> beab6e2 (feat: SVD Discretization implemented)
=======
    include("test_laplacian.jl")
<<<<<<< HEAD
>>>>>>> 2e4bfa5 (Add some tests)
=======
=======
    include("test_self_tuning.jl")
>>>>>>> f84002b (tests)
>>>>>>> 81a6ea7 (tests)
end

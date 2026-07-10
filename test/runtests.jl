using SpectralClustering
using Test

const RUN_AQUA_TESTS = get(ENV, "SPECTRAL_CLUSTERING_RUN_AQUA", "false") == "true"

@testset verbose=true "SpectralClustering.jl" begin
    @testset "Datasets" begin
        include("test_datasets.jl")
    end

    @testset "Affinities" begin
        include("test_affinity.jl")
    end

    @testset "Discretization" begin
        include("test_discretization.jl")
    end

    @testset "Eigensolvers" begin
        include("test_eigensolvers.jl")
    end

    @testset "Laplacians" begin
        include("test_laplacians.jl")
    end

    @testset "Self-Tuning" begin
        include("test_self_tuning.jl")
    end

    @testset "Public API" begin
        include("test_api.jl")
    end

    if RUN_AQUA_TESTS
        @testset "Aqua" begin
            include("test_aqua.jl")
        end
    end

    @testset "Indexing" begin
        include("test_indexing.jl")
    end

end

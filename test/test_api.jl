using Test
using SpectralClustering

@testset "Public API" begin
    @testset "spectral_cluster clusters a simple two-cluster dataset" begin
        X = [0.0 0.0 5.0 5.0;
             0.0 1.0 5.0 6.0]

        labels = spectral_cluster(
            X,
            2;
            affinity=RBFKernel(1.0),
            laplacian=SymmetricLaplacian(),
            discretizer=SVDDiscretization(),
        )

        @test length(labels) == 4
        @test length(unique(labels)) == 2
        @test labels[1] == labels[2]
        @test labels[3] == labels[4]
        @test labels[1] != labels[3]
    end
end
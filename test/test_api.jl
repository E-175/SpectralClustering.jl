using Test
using SpectralClustering
using Random

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

    @testset "spectral_cluster supports all end-to-end variants" begin
        rng = MersenneTwister(1234)
        X, _ = make_moons(rng, 100; noise=0.05)

        variants = [
            (RBFKernel(1.0), UnnormalizedLaplacian(), KMeansDiscretization(false, false, 7)),
            (RBFKernel(1.0), RandomWalkLaplacian(), KMeansDiscretization(true, false, 7)),
            (RBFKernel(1.0), SymmetricLaplacian(), SVDDiscretization()),
            (LocalScaling(7), UnnormalizedLaplacian(), KMeansDiscretization(false, true, 7)),
            (LocalScaling(7), RandomWalkLaplacian(), SelfTuningDiscretization()),
            (LocalScaling(7), SymmetricLaplacian(), SVDDiscretization()),
        ]

        for (affinity, laplacian, discretizer) in variants
            labels = spectral_cluster(
                X,
                2;
                affinity=affinity,
                laplacian=laplacian,
                discretizer=discretizer,
            )

            @test length(labels) == size(X, 2)
            @test sort(unique(labels)) == [1, 2]
        end
    end
end
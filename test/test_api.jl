using Test
using SpectralClustering
using Random

function pairwise_cluster_agreement(labels::AbstractVector, truth::AbstractVector)
    length(labels) == length(truth) || throw(ArgumentError("labels and truth must have the same length"))

    matches = 0
    total = 0
    for i in 1:(length(labels) - 1)
        for j in (i + 1):length(labels)
            matches += (labels[i] == labels[j]) == (truth[i] == truth[j])
            total += 1
        end
    end

    return matches / total
end

function assert_perfect_recovery(labels, truth)
    @test sort(unique(labels)) == [1, 2]
    @test pairwise_cluster_agreement(labels, truth) == 1.0
end

const LINEAR_DATA = (
    X=[
        -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 5.0 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9;
         0.0  0.1 -0.1  0.05 -0.05 0.08 -0.08 0.02 -0.02 0.0 5.0 5.1 4.9 5.05 4.95 5.08 4.92 5.02 4.98 5.0;
    ],
    y=vcat(fill(1, 10), fill(2, 10)),
)

const MOONS_DATA = let
    X, y = make_moons(MersenneTwister(1234), 40; noise=0.05)
    (X=X, y=y)
end

const API_VARIANTS = [
    (
        name="rbf + symmetric + svd on linear data",
        affinity=RBFKernel(1.0),
        laplacian=SymmetricLaplacian(),
        discretizer=SVDDiscretization(),
        X=LINEAR_DATA.X,
        y=LINEAR_DATA.y,
        k=2,
        rng=MersenneTwister(101),
    ),
    (
        name="rbf + random-walk + normalized kmeans on moons",
        affinity=RBFKernel(0.1),
        laplacian=RandomWalkLaplacian(),
        discretizer=KMeansDiscretization(true, false),
        X=MOONS_DATA.X,
        y=MOONS_DATA.y,
        k=2,
        rng=MersenneTwister(102),
    ),
    (
        name="local scaling + unnormalized + manual kmeans on linear data",
        affinity=LocalScaling(7),
        laplacian=UnnormalizedLaplacian(),
        discretizer=KMeansDiscretization(false, true),
        X=LINEAR_DATA.X,
        y=LINEAR_DATA.y,
        k=2,
        rng=MersenneTwister(103),
    ),
    (
        name="local scaling + random-walk + self tuning on linear data",
        affinity=LocalScaling(7),
        laplacian=RandomWalkLaplacian(),
        discretizer=SelfTuningDiscretization(4),
        X=LINEAR_DATA.X,
        y=LINEAR_DATA.y,
        k=nothing,
        rng=MersenneTwister(104),
    ),
]

@testset "Public API" begin
    @testset "Basic clustering behavior" begin
        @testset "spectral_cluster clusters a simple two-cluster dataset with defaults" begin
            X = [0.0 0.0 5.0 5.0;
                 0.0 1.0 5.0 6.0]

            labels = spectral_cluster(
                X,
                2;
                rng=MersenneTwister(1234),
            )

            @test length(labels) == 4
            @test labels[1] == labels[2]
            @test labels[3] == labels[4]
            @test labels[1] != labels[3]
        end

        @testset "spectral_cluster accepts nothing for self-tuning discretization" begin
            X = [0.0 0.0 5.0 5.0;
                 0.0 1.0 5.0 6.0]
            y = [1, 1, 2, 2]

            labels = spectral_cluster(
                X,
                nothing;
                affinity=RBFKernel(1.0),
                laplacian=SymmetricLaplacian(),
                discretizer=SelfTuningDiscretization(),
                rng=MersenneTwister(1234),
            )

            assert_perfect_recovery(labels, y)
        end
    end

    @testset "Public option smoke tests" begin
        for variant in API_VARIANTS
            @testset "$(variant.name)" begin
                labels = spectral_cluster(
                    variant.X,
                    variant.k;
                    affinity=variant.affinity,
                    laplacian=variant.laplacian,
                    discretizer=variant.discretizer,
                    rng=variant.rng,
                )

                @test length(labels) == size(variant.X, 2)
                assert_perfect_recovery(labels, variant.y)
            end
        end
    end

    @testset "Argument validation" begin
        @testset "spectral_cluster rejects invalid k" begin
            X = [-1.0 -0.9 1.0 0.9;
                  0.0  0.1 0.0 -0.1]

            @test_throws ArgumentError spectral_cluster(
                X,
                0;
                affinity=RBFKernel(0.5),
                laplacian=RandomWalkLaplacian(),
                discretizer=KMeansDiscretization(false, false),
            )

            @test_throws ArgumentError spectral_cluster(
                X,
                5;
                affinity=RBFKernel(0.5),
                laplacian=RandomWalkLaplacian(),
                discretizer=KMeansDiscretization(false, false),
            )

            @test_throws ArgumentError spectral_cluster(
                X,
                nothing;
                affinity=RBFKernel(0.5),
                laplacian=RandomWalkLaplacian(),
                discretizer=KMeansDiscretization(false, false),
            )
        end
    end
end
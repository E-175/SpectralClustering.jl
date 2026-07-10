using Test
using OffsetArrays
using SpectralClustering

#created by AI
@testset "One-based indexing requirements" begin

    @testset "compute_affinity rejects non-1-based arrays" begin
        # The implementation uses 1-based indexing internally.
        # OffsetArray shifts the array indices away from Julia's default 1-based axes.
        # Therefore, compute_affinity should reject it explicitly.

        X = OffsetArray(
            [0.0 1.0;
             0.0 1.0],
            0:1,
            0:1,
        )

        @test_throws ArgumentError compute_affinity(X, RBFKernel(1.0))
    end

    @testset "compute_laplacian rejects non-1-based arrays" begin
        # Laplacian implementations use direct indexing like W[i, j].
        # Therefore, non-1-based affinity matrices should be rejected clearly.

        W = OffsetArray(
            [0.0 1.0;
             1.0 0.0],
            0:1,
            0:1,
        )

        @test_throws ArgumentError compute_laplacian(W, RandomWalkLaplacian())
    end

    @testset "compute_eigenvectors rejects non-1-based arrays" begin
        # The eigensolver assumes standard 1-based matrix indexing.
        # This test ensures that custom-index arrays fail early with a clear error.

        L = OffsetArray(
            [0.0 0.0;
             0.0 1.0],
            0:1,
            0:1,
        )

        @test_throws ArgumentError compute_eigenvectors(L, 1)
    end

    @testset "discretize rejects non-1-based arrays" begin
        # Discretization expects the spectral embedding in a normal Matrix-like layout.
        # Non-1-based arrays should be rejected before K-Means is called.

        V = OffsetArray(
            [-0.8 -0.7 0.8 0.9;
              0.0  0.1 0.0 -0.1],
            0:1,
            0:3,
        )

        @test_throws ArgumentError discretize(
            V,
            KMeansDiscretization(false, false);
            k=2,
        )
    end
end
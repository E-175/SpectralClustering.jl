using SpectralClustering
using Test
using LinearAlgebra
using Random

@testset "Data Generation" begin
    Random.seed!(42)  # For reproducibility

    @testset "make_circles" begin
        n = 100
        X, y = make_circles(n, factor=0.5, noise=0.0)
        
        @test size(X) == (2, n)
        @test length(y) == n
        @test sum(y .== 1) == 50
        @test sum(y .== 2) == 50
        
        # Test radii without noise
        @test all(isapprox.(norm.(eachcol(X[:, y .== 1])), 1.0; atol=1e-5))
        @test all(isapprox.(norm.(eachcol(X[:, y .== 2])), 0.5; atol=1e-5))

        # Test error for invalid factor
        @test_throws ArgumentError make_circles(n, factor=1.5)
        @test_throws ArgumentError make_circles(n, factor=-0.5)

        # Test noise addition
        X_noisy, _ = make_circles(n, factor=0.5, noise=0.1)
        @test size(X_noisy) == (2, n)
        @test X_noisy != X

        @testset "Mixed float types in make_circles" begin
            n = 100
            # Float32 noise, Float64 factor
            noise_f32 = 0.1f0
            factor_f64 = 0.8

            X, y = make_circles(n, noise=noise_f32, factor=factor_f64)

            @test size(X) == (2, n)
            @test length(y) == n

            # Check that the resulting array type is Float64
            @test eltype(X) == Float64
    end
    end

    @testset "make_moons" begin
        n = 100
        X, y = make_moons(n, noise=0.0)
        
        @test size(X) == (2, n)
        @test length(y) == n
        @test sum(y .== 1) == 50
        @test sum(y .== 2) == 50
        
        # Test basic shape properties without noise
        # Outer moon y >= 0
        @test all(X[2, y .== 1] .>= 0.0)
        # Inner moon y <= 0.5
        @test all(X[2, y .== 2] .<= 0.5)

        # Test noise addition
        X_noisy, _ = make_moons(n, noise=0.1)
        @test size(X_noisy) == (2, n)
        @test X_noisy != X
    end

    @testset "make_blobs" begin
        n = 100
        centers = 4
        X, y = make_blobs(n, centers=centers)
        
        @test size(X) == (2, n)
        @test length(y) == n
        @test sort(unique(y)) == 1:centers
        # Since 100 is divisible by 4, each should have 25
        @test all(count(==(i), y) == 25 for i in 1:centers)
        
        # Odd number
        X, y = make_blobs(101, centers=3)
        @test length(y) == 101
    end
end

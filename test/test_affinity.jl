using Test
using SpectralClustering
using Random

@testset "Affinity matrix" begin
    X = [0.0 1.0 3.0;
         0.0 0.0 0.0]

    A = compute_affinity(X, RBFKernel(1.0))

    @test size(A) == (3, 3)
    @test A ≈ A'

    @test A[1, 1] == 0.0
    @test A[2, 2] == 0.0
    @test A[3, 3] == 0.0

    @test all(A .>= 0.0)
    @test all(A .<= 1.0)

    expected_12 = exp(-1.0 / 2.0)
    expected_13 = exp(-9.0 / 2.0)
    expected_23 = exp(-4.0 / 2.0)

    @test A[1, 2] ≈ expected_12
    @test A[2, 1] ≈ expected_12

    @test A[1, 3] ≈ expected_13
    @test A[3, 1] ≈ expected_13

    @test A[2, 3] ≈ expected_23
    @test A[3, 2] ≈ expected_23
end

@testset "Affinity matrix self-affinity" begin
    X = [0.0 1.0;
         0.0 0.0]

    A_default = compute_affinity(X, RBFKernel(1.0))
    A = compute_affinity(X, RBFKernel(1.0); self_affinity=1.0)

    @test A[1, 1] == 1.0
    @test A[2, 2] == 1.0
    @test A[1, 2] == A_default[1, 2]
    @test A[2, 1] == A_default[2, 1]
end

@testset "Affinity matrix sigma validation" begin
    X = [0.0 1.0;
         0.0 0.0]

    @test_throws ArgumentError compute_affinity(X, RBFKernel(0.0))

    @test_throws ArgumentError compute_affinity(X, RBFKernel(-1.0))
end

@testset "Affinity preserves input float types" begin
    X = Float32[0 1 3;
                0 0 0]

    A_default = compute_affinity(X, RBFKernel())
    A_rbf = compute_affinity(X, RBFKernel(1.0f0))
    A_local = compute_affinity(X, LocalScaling(1))

    @test eltype(A_default) == Float32
    @test eltype(A_rbf) == Float32
    @test eltype(A_local) == Float32
end
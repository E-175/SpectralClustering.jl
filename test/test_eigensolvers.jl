using Test
using SpectralClustering
using LinearAlgebra


@testset "Eigensolvers" begin

    @testset "returns bottom k eigenvectors in features by samples format" begin
        # This test uses a diagonal Laplacian where the eigenvectors are easy
        # to understand.
        #
        # The eigenvalues are 0, 2, and 5.
        # For k = 2, the function should select the eigenvectors belonging
        # to the two smallest eigenvalues.

        L = [0.0 0.0 0.0;
             0.0 2.0 0.0;
             0.0 0.0 5.0]

        V = compute_eigenvectors(L, 2)

        # The package uses features × samples format.
        # Therefore, k eigenvectors become k rows and n samples become n columns.
        @test size(V) == (2, 3)

        # The two returned eigenvectors should be orthonormal.
        # Since they are stored as rows, V * V' should be the 2 × 2 identity.
        @test V * V' ≈ Matrix(I, 2, 2)

        # For this diagonal example, the selected eigenvectors are the first
        # two standard basis vectors, up to possible sign changes.
        @test abs.(V) ≈ [1.0 0.0 0.0;
                         0.0 1.0 0.0]
    end

    
    @testset "works with non-symmetric random-walk Laplacian" begin
        # The random-walk normalized Laplacian is generally not symmetric.
        #
        # This test checks that compute_eigenvectors can still handle such
        # matrices through the general eigensolver path.

        Lrw = [ 1.0  -0.5  -0.5;
               -1.0   1.0   0.0;
               -1.0   0.0   1.0]

        V = compute_eigenvectors(Lrw, 2)

        @test size(V) == (2, 3)
        @test eltype(V) <: Real
        @test all(isfinite, V)
    end


    @testset "rejects non-square Laplacian matrix" begin
        # A Laplacian must be square because it describes relationships
        # between the same set of samples.

        L_non_square = [1.0 0.0 0.0;
                        0.0 1.0 0.0]

        @test_throws ArgumentError compute_eigenvectors(L_non_square, 1)
    end


    @testset "rejects invalid k" begin
        # k must be at least 1 and at most the number of samples.

        L = [0.0 0.0;
             0.0 1.0]

        @test_throws ArgumentError compute_eigenvectors(L, 0)
        @test_throws ArgumentError compute_eigenvectors(L, 3)
    end

        @testset "does not modify input matrix" begin
        # compute_eigenvectors should read from L but not modify it in-place.

        L = [0.0 0.0 0.0;
             0.0 1.0 0.0;
             0.0 0.0 2.0]

        L_original = copy(L)

        compute_eigenvectors(L, 2)

        @test L == L_original
    end


    @testset "rejects selected eigenvectors with non-negligible imaginary parts" begin
        L = [0.0 -1.0;
             1.0  0.0]

        @test_throws ArgumentError compute_eigenvectors(L, 1)
    end
end
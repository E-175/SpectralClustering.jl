using Test
using SpectralClustering

@testset "Eigensolvers" begin
    @testset "rejects selected eigenvectors with non-negligible imaginary parts" begin
        L = [0.0 -1.0;
             1.0  0.0]

        @test_throws ArgumentError compute_eigenvectors(L, 1)
    end
end
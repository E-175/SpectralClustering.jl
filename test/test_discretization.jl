using SpectralClustering
using Test

@testset "Discretization" begin
    @testset "SVDDiscretization" begin
        method = SVDDiscretization()
        
        # Test input validation
        V_dummy = rand(2, 5)
        
        # 1. Missing k
        @test_throws ArgumentError discretize(V_dummy, method)
        
        # 2. k doesn't match eigenvectors
        @test_throws ArgumentError discretize(V_dummy, method; k=3)
        
        # 3. k out of bounds
        @test_throws ArgumentError discretize(rand(6, 5), method; k=6)
        
        # Test logic on a perfect block diagonal embedding
        # 4 samples, 2 clusters. First two belong to C1, last two belong to C2.
        V_ideal = [1.0 1.0 0.0 0.0; 
                   0.0 0.0 1.0 1.0]
        
        labels = discretize(V_ideal, method; k=2)
        
        @test length(labels) == 4
        
        # The first two should have the same label
        @test labels[1] == labels[2]
        
        # The last two should have the same label
        @test labels[3] == labels[4]
        
        # But they must be different clusters
        @test labels[1] != labels[3]
    end
end

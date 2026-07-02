using SpectralClustering
using Test
using Random

struct DummyDiscretization <: AbstractDiscretization end

@testset "Discretization" begin
    @testset "KMeansDiscretization constructor" begin
        method = KMeansDiscretization(true)

        @test method.normalize_samples === true
        @test method.use_manual_implementation === false
        @test isnothing(method.seed)
    end

    @testset "KMeansDiscretization validation" begin
        method = KMeansDiscretization(false)
        V = [1.0 1.0 0.0 0.0;
             0.0 0.0 1.0 1.0]

        @test_throws ArgumentError discretize(V, method)
        @test_throws ArgumentError discretize(V, method; k=0)
        @test_throws ArgumentError discretize(V, method; k=5)
        @test_throws ArgumentError discretize(zeros(0, 4), method; k=1)
    end

    @testset "KMeansDiscretization normalization validation" begin
        method = KMeansDiscretization(true, false, 7)
        V = [1.0 0.0 1.0;
             0.0 0.0 1.0]

        @test_throws ArgumentError discretize(V, method; k=2)
    end

    @testset "KMeansDiscretization clustering" begin
        V = [3.0 3.1 -3.0 -3.1;
             0.0 0.1  0.0 -0.1]

        method = KMeansDiscretization(false, false, 11)
        labels = discretize(V, method; k=2)

        @test length(labels) == 4
        @test labels[1] == labels[2]
        @test labels[3] == labels[4]
        @test labels[1] != labels[3]
        @test V == [3.0 3.1 -3.0 -3.1;
                    0.0 0.1  0.0 -0.1]
    end

    @testset "KMeansDiscretization manual implementation" begin
        V = [2.0 2.2 -2.0 -2.2;
             0.0 0.1  0.0 -0.1]

        method = KMeansDiscretization(false, true, 19)
        labels = discretize(V, method; k=2)

        @test length(labels) == 4
        @test labels[1] == labels[2]
        @test labels[3] == labels[4]
        @test labels[1] != labels[3]
    end

    @testset "Discretization fallback" begin
        V = [1.0 0.0;
             0.0 1.0]

        @test_throws ErrorException discretize(V, DummyDiscretization())
    end

    @testset "SVDDiscretization" begin
        method = SVDDiscretization()
        
        # Test input validation
        V_dummy = rand(MersenneTwister(21), 2, 5)
        
        # 1. Missing k
        @test_throws ArgumentError discretize(V_dummy, method)
        
        # 2. k doesn't match eigenvectors
        @test_throws ArgumentError discretize(V_dummy, method; k=3)
        
        # 3. k out of bounds
        @test_throws ArgumentError discretize(rand(MersenneTwister(22), 6, 5), method; k=6)
        
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

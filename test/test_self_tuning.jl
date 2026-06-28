using Test
using SpectralClustering
import SpectralClustering: discretize, SelfTuningDiscretization, LocalScaling, compute_affinity

# =========================================================
# Tests for Self-Tuning Discretization
# =========================================================

@testset "Self-Tuning Discretization Validation" begin
    V = [1.0 0.0; 
         0.0 1.0]
    method = SelfTuningDiscretization()
    
    # Should throw an error because k=3 is greater than the 2 columns in V
    @test_throws ArgumentError discretize(V, method; k=3)
end

@testset "Self-Tuning Discretization Fixed k" begin
    # Perfect eigenvectors for 2 clusters
    V = [1.0 0.0;
         1.0 0.0;
         0.0 1.0;
         0.0 1.0]
    
    method = SelfTuningDiscretization()
    labels = discretize(V, method; k=2)
    
    @test length(labels) == 4
    # Points 1 and 2 should be in the same cluster
    @test labels[1] == labels[2]
    # Points 3 and 4 should be in the same cluster
    @test labels[3] == labels[4]
    # The two clusters should be different
    @test labels[1] != labels[3]
end

@testset "Self-Tuning Discretization with Rotation Recovery" begin
    # Perfect eigenvectors
    V_perfect = [1.0 0.0;
                 1.0 0.0;
                 0.0 1.0;
                 0.0 1.0]
                 
    # Mix the eigenvectors by applying a 45-degree rotation (pi/4)
    theta = pi / 4
    R_mix = [cos(theta) -sin(theta); 
             sin(theta) cos(theta)]
    V_mixed = V_perfect * R_mix
    
    method = SelfTuningDiscretization()
    
    # The optimization step should "un-rotate" the matrix and recover the clusters
    labels = discretize(V_mixed, method; k=2)
    
    @test length(labels) == 4
    @test labels[1] == labels[2]
    @test labels[3] == labels[4]
    @test labels[1] != labels[3]
end

@testset "Self-Tuning Discretization Automatic k (Self-Tuning)" begin
    # Perfect eigenvectors for exactly 3 distinct clusters
    V_perfect = [1.0 0.0 0.0;
                 1.0 0.0 0.0;
                 0.0 1.0 0.0;
                 0.0 1.0 0.0;
                 0.0 0.0 1.0;
                 0.0 0.0 1.0]
                 
    # Real eigensolvers return a mixed orthogonal basis of the eigenspace.
    # We simulate this reality by applying a 3x3 orthogonal mixing matrix Q.
    Q = [ 1/sqrt(3)   1/sqrt(2)   1/sqrt(6);
          1/sqrt(3)  -1/sqrt(2)   1/sqrt(6);
          1/sqrt(3)   0.0        -2/sqrt(6)]
         
    V_mixed = V_perfect * Q
         
    method = SelfTuningDiscretization()
    
    # Do not provide k, forcing the algorithm to find the optimal k
    labels = discretize(V_mixed, method)
    
    @test length(labels) == 6
    # The algorithm should now correctly identify that there are exactly 3 clusters
    @test length(unique(labels)) == 3
    
    # Check that the assignments are grouped correctly
    @test labels[1] == labels[2]
    @test labels[3] == labels[4]
    @test labels[5] == labels[6]
end

# =========================================================
# Tests for LocalScaling Affinity Matrix
# =========================================================

@testset "LocalScaling Affinity Matrix Basic Properties" begin
    # 3 points in 2D
    X = [0.0 1.0 3.0;
         0.0 0.0 0.0]

    method = LocalScaling(1) # Assuming struct takes k=1
    
    A = compute_affinity(X, method)

    @test size(A) == (3, 3)
    @test A ≈ A'

    # Diagonal must be 0.0 as hardcoded in the function
    @test A[1, 1] == 0.0
    @test A[2, 2] == 0.0
    @test A[3, 3] == 0.0

    @test all(A .>= 0.0)
    @test all(A .<= 1.0)
end

@testset "LocalScaling Affinity Matrix Explicit Values" begin
    # 1D points: 0.0, 2.0, 5.0 to easily trace the math manually
    X = [0.0 2.0 5.0]
    
    # k=1 means 1st nearest neighbor (excluding self)
    # Distances for pt 1 (0.0): to pt 2 is 2.0, to pt 3 is 5.0. -> sigma_1 = 2.0
    # Distances for pt 2 (2.0): to pt 1 is 2.0, to pt 3 is 3.0. -> sigma_2 = 2.0
    # Distances for pt 3 (5.0): to pt 1 is 5.0, to pt 2 is 3.0. -> sigma_3 = 3.0
    method = LocalScaling(1)
    
    A = compute_affinity(X, method)
    
    # Formula: exp(-d^2 / (sigma_i * sigma_j))
    expected_12 = exp(- (2.0^2) / (2.0 * 2.0)) # exp(-4/4) = exp(-1)
    expected_23 = exp(- (3.0^2) / (2.0 * 3.0)) # exp(-9/6) = exp(-1.5)
    expected_13 = exp(- (5.0^2) / (2.0 * 3.0)) # exp(-25/6)
    
    @test A[1, 2] ≈ expected_12
    @test A[2, 1] ≈ expected_12
    
    @test A[2, 3] ≈ expected_23
    @test A[3, 2] ≈ expected_23
    
    @test A[1, 3] ≈ expected_13
    @test A[3, 1] ≈ expected_13
end

@testset "LocalScaling Affinity Matrix Duplicate Points (Zero Sigma Safety)" begin
    # Two identical points and one distinct point
    X = [1.0 1.0 5.0]
    
    method = LocalScaling(1)
    A = compute_affinity(X, method)
    
    # Sigmas for points 1 and 2 will be eps(Float64) because distance to k=1 neighbor is 0
    # The affinity between 1 and 2 should be exp(-0 / eps^2) == 1.0
    # Ensuring it doesn't evaluate to NaN due to 0/0 division
    @test !isnan(A[1, 2])
    @test A[1, 2] ≈ 1.0
end

@testset "LocalScaling Affinity self_affinity override" begin
    X = [0.0 1.0;
         0.0 0.0]

    method = LocalScaling(1)
    # Even if the user requests self_affinity=1.0, your function forces it back to 0.0
    A = compute_affinity(X, method; self_affinity=1.0)

    @test A[1, 1] == 0.0
    @test A[2, 2] == 0.0
end

@testset "LocalScaling Affinity Matrix works with generated data" begin
    # Assuming make_moons is available in your test environment
    X, y = make_moons(100; noise=0.05)

    # The paper suggests k=7 as a good default for local scaling
    method = LocalScaling(7)
    A = compute_affinity(X, method)

    @test size(A) == (100, 100)
    @test A ≈ A'
    @test all(A .>= 0.0)
    @test all(A .<= 1.0)
    
    # Ensure diagonal was correctly enforced to 0 on the generated data
    @test all([A[i, i] == 0.0 for i in 1:100])
end
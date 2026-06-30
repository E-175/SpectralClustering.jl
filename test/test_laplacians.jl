using Test
using SpectralClustering


@testset "Symmetric Laplacian Part 1" begin
    A = [0.0 1.0 1.0;
         1.0 0.0 1.0;
         1.0 1.0 0.0]

    B = [1.0 -0.5 -0.5;
         -0.5 1.0 -0.5;
         -0.5 -0.5 1.0]

    @test compute_laplacian(A,SymmetricLaplacian()) ≈ B
    

    C = [1.0 0.0 0.0;
         0.0 1.0 0.0;
         0.0 0.0 1.0]


     D = [0.0 0.0 0.0;
         0.0 0.0 0.0;
         0.0 0.0 0.0]


    @test compute_laplacian(C,SymmetricLaplacian()) ≈ D

    #Non Square
    E = [1.0 0.0 0.0;
         0.0 1.0 0.0]

    @test_throws ArgumentError compute_laplacian(E,SymmetricLaplacian())

     #Zero Degree
    F = [1.0 0.0 1.0;
         0.0 0.0 0.0;
         1.0 0.0 1.0]

   @test_throws ArgumentError compute_laplacian(F,SymmetricLaplacian())

     #Negative Value
    G = [-1.0 0.0 0.0;
         0.0 0.0 0.0;
         0.0 0.0 1.0]

   @test_throws ArgumentError compute_laplacian(G,SymmetricLaplacian())

     #Not symmetric
    H = [1.0 0.0 1.0;
         0.0 1.0 0.0;
         0.0 0.0 1.0]


   @test_throws ArgumentError compute_laplacian(H,SymmetricLaplacian())



    J = [0.0 16.0 35.0 67.0 13.0 17.0 49.0;
         16.0 0.0 31.0 36.0 66.0 40.0 57.0;
         35.0 31.0 0.0 33.0 79.0 57.0 35.0;
         67.0 36.0 33.0 0.0 64.0 46.0 34.0;
         13.0 66.0 79.0 64.0 0.0 53.0 48.0;
         17.0 40.0 57.0 46.0 53.0 0.0 36.0;
         49.0 57.0 35.0 34.0 48.0 36.0 0.0]

     
     K = [1.0 -0.07268073591480126 -0.15175850835474114 -0.28527431679536136 -0.05153579477758927 -0.0767566717046357 -0.21692673761838976;
     -0.07268073591480126 1.0 -0.1202852176047607 -0.13716898705776376 -0.2341397341059733 -0.16161912985516966 -0.22581740901226904;
     -0.1517585083547411 -0.12028521760476069 1.0 -0.12001983962979582 -0.2675124187089514 -0.21983320925853073 -0.13235375898521928;
     -0.28527431679536136 -0.13716898705776376 -0.12001983962979582 1.0 -0.21281375619376333 -0.1742124315708461 -0.12625541662100884;
     -0.051535794777589274 -0.2341397341059733 -0.2675124187089515 -0.21281375619376333 1.0 -0.1868852108571645 -0.16595490543218622;
     -0.0767566717046357 -0.16161912985516969 -0.21983320925853073 -0.17421243157084612 -0.18688521085716453 1.0 -0.14175975213276507;
     -0.21692673761838976 -0.22581740901226902 -0.1323537589852193 -0.12625541662100884 -0.16595490543218622 -0.14175975213276507 1.0]
     
     @test compute_laplacian(J,SymmetricLaplacian()) ≈ K



     L = [0.0 2.0 1.0 3.0 2.0;
          2.0 0.0 2.0 1.0 4.0;
          1.0 2.0 0.0 2.0 3.0;
          3.0 1.0 2.0 0.0 2.0;
          2.0 4.0 3.0 2.0 0.0]


     M = [1.0 -0.2357022603955158 -0.12499999999999999 -0.37499999999999994 -0.21320071635561041;
     -0.2357022603955158 1.0 -0.2357022603955158 -0.1178511301977579 -0.40201512610368484;
     -0.12499999999999999 -0.2357022603955158 1.0 -0.24999999999999997 -0.31980107453341566;
     -0.37499999999999994 -0.1178511301977579 -0.24999999999999997 1.0 -0.21320071635561041;
     -0.21320071635561041 -0.40201512610368484 -0.3198010745334156 -0.21320071635561041 1.0]      

     @test compute_laplacian(L,SymmetricLaplacian()) ≈ M

end


@testset "Symmetric Laplacian Part 2" begin

A,B = make_circles()
C = compute_affinity(A,RBFKernel())
D = compute_laplacian(C,SymmetricLaplacian())

#Test Symmetry
@test D ≈ D'
#Diagonal of Laplacian should be all 1 as self-affinity is 0
@test diag(D) ≈ ones(100)
#Eigenvectors should compute
@test compute_eigenvectors(D,2) isa AbstractMatrix

E,F = make_moons()
G = compute_affinity(E,RBFKernel())
H = compute_laplacian(G,SymmetricLaplacian())

#Test Symmetry
@test H ≈ H'
#Diagonal of Laplacian should be all 1 as self-affinity is 0
@test diag(H) ≈ ones(100)
#Eigenvectors should compute
@test compute_eigenvectors(H,2) isa AbstractMatrix

J,K = make_blobs()
L = compute_affinity(J,RBFKernel())
M = compute_laplacian(L,SymmetricLaplacian())

#Test Symmetry
@test M ≈ M'
#Diagonal of Laplacian should be all 1 as self-affinity is 0
@test diag(M) ≈ ones(100)
#Eigenvectors should compute    	
@test compute_eigenvectors(M,3) isa AbstractMatrix
   @test_throws ArgumentError compute_laplacian(H,SymmetricLaplacian())

end



@testset "Unnormalized Laplacian Part 1" begin
    A = [0.0 1.0 1.0;
         1.0 0.0 1.0;
         1.0 1.0 0.0]

    B = [2.0 -1.0 -1.0;
         -1.0 2.0 -1.0;
         -1.0 -1.0 2.0]

    @test compute_laplacian(A,UnnormalizedLaplacian()) ≈ B
    

    C = [1.0 0.0 0.0;
         0.0 1.0 0.0;
         0.0 0.0 1.0]


     D = [0.0 0.0 0.0;
         0.0 0.0 0.0;
         0.0 0.0 0.0]


         #Non Square
    @test compute_laplacian(C,UnnormalizedLaplacian()) ≈ D

    E = [1.0 0.0 0.0;
         0.0 1.0 0.0]

    @test_throws ArgumentError compute_laplacian(E,UnnormalizedLaplacian())


    #Zero Degree
    F = [1.0 0.0 0.0;
         0.0 0.0 0.0;
         0.0 0.0 1.0]


   @test compute_laplacian(F,UnnormalizedLaplacian()) ≈ D

     #Negative Value
    G = [-1.0 0.0 0.0;
         0.0 0.0 0.0;
         0.0 0.0 1.0]

   @test_throws ArgumentError compute_laplacian(G,UnnormalizedLaplacian())


   #Not symmetric
    H = [1.0 0.0 1.0;
         0.0 1.0 0.0;
         0.0 0.0 1.0]

   @test_throws ArgumentError compute_laplacian(H,UnnormalizedLaplacian())



   

    J = [0.0 16.0 35.0 67.0 13.0 17.0 49.0;
         16.0 0.0 31.0 36.0 66.0 40.0 57.0;
         35.0 31.0 0.0 33.0 79.0 57.0 35.0;
         67.0 36.0 33.0 0.0 64.0 46.0 34.0;
         13.0 66.0 79.0 64.0 0.0 53.0 48.0;
         17.0 40.0 57.0 46.0 53.0 0.0 36.0;
         49.0 57.0 35.0 34.0 48.0 36.0 0.0]

    K = [197.0 -16.0 -35.0 -67.0 -13.0 -17.0 -49.0;
         -16.0 246.0 -31.0 -36.0 -66.0 -40.0 -57.0;
         -35.0 -31.0 270.0 -33.0 -79.0 -57.0 -35.0;
         -67.0 -36.0 -33.0 280.0 -64.0 -46.0 -34.0;
         -13.0 -66.0 -79.0 -64.0 323.0 -53.0 -48.0;
         -17.0 -40.0 -57.0 -46.0 -53.0 249.0 -36.0;
         -49.0 -57.0 -35.0 -34.0 -48.0 -36.0 259.0]
     @test compute_laplacian(J,UnnormalizedLaplacian()) ≈ K



     L = [0.0 2.0 1.0 3.0 2.0;
          2.0 0.0 2.0 1.0 4.0;
          1.0 2.0 0.0 2.0 3.0;
          3.0 1.0 2.0 0.0 2.0;
          2.0 4.0 3.0 2.0 0.0]


     M = [8.0 -2.0 -1.0 -3.0 -2.0;
          -2.0 9.0 -2.0 -1.0 -4.0;
          -1.0 -2.0 8.0 -2.0 -3.0;
          -3.0 -1.0 -2.0 8.0 -2.0;
          -2.0 -4.0 -3.0 -2.0 11.0]
    

     @test compute_laplacian(L,UnnormalizedLaplacian()) ≈ M
end


@testset "Unnormalized Laplacian Part 2" begin

A,B = make_circles()
C = compute_affinity(A,RBFKernel())
D = compute_laplacian(C,UnnormalizedLaplacian())

#Test Symmetry
@test D ≈ D'
#Sum in every row should be 0 as self affinity is 0
degreesD = vec(sum(D, dims=2))
@test isapprox(degreesD, zeros(100), atol=0.001)
#Eigenvectors should compute
@test compute_eigenvectors(D,2) isa AbstractMatrix

E,F = make_moons()
G = compute_affinity(E,RBFKernel())
H = compute_laplacian(G,UnnormalizedLaplacian())

#Test Symmetry
@test H ≈ H'
#Sum in every row should be 0 as self affinity is 0
degreesH = vec(sum(H, dims=2))
@test isapprox(degreesH, zeros(100), atol=0.001)
#Eigenvectors should compute
@test compute_eigenvectors(H,2) isa AbstractMatrix

J,K = make_blobs()
L = compute_affinity(J,RBFKernel())
M = compute_laplacian(L,UnnormalizedLaplacian())

#Test Symmetry
@test M ≈ M'
#Sum in every row should be 0 as self affinity is 0
degreesM = vec(sum(M, dims=2))
@test isapprox(degreesM, zeros(100), atol=0.001)
#Eigenvectors should compute
@test compute_eigenvectors(M,3) isa AbstractMatrix

end


@testset "RandomWalkLaplacian" begin

    @testset "computes expected random-walk Laplacian" begin
        # This test checks the core formula of the RandomWalkLaplacian:
        #     L_rw = I - D⁻¹W

        W = [0.0 1.0 1.0;
             1.0 0.0 0.0;
             1.0 0.0 0.0]

        # Compute the random-walk normalized Laplacian from W.
        Lrw = compute_laplacian(W, RandomWalkLaplacian())

        # The degrees are the row sums of W:
        #
        # degree(1) = 1 + 1 = 2
        # degree(2) = 1
        # degree(3) = 1
        #
        # Therefore:
        #
        # D⁻¹W =
        # [0.0 0.5 0.5;
        #  1.0 0.0 0.0;
        #  1.0 0.0 0.0]
        #
        # and:
        #
        # L_rw = I - D⁻¹W
        expected = [ 1.0  -0.5  -0.5;
                    -1.0   1.0   0.0;
                    -1.0   0.0   1.0]

        # The output must have the same shape as the input affinity matrix.
        @test size(Lrw) == size(W)

        # The computed matrix must match the hand-calculated expected matrix.
        @test Lrw ≈ expected
    end

    @testset "row sums are approximately zero" begin
        # This test checks an important property of the random-walk Laplacian.
        #
        # Since L_rw = I - D⁻¹W and every row of D⁻¹W sums to 1,
        # every row of L_rw should sum to 0.

        W = [0.0 2.0 1.0;
             2.0 0.0 3.0;
             1.0 3.0 0.0]

        Lrw = compute_laplacian(W, RandomWalkLaplacian())

        # For a valid random-walk Laplacian, every row should sum to approximately 0.
    	# Floating-point arithmetic can create tiny numerical differences
		# instead of exact 0, so we use an explicit absolute tolerance.
    	@test isapprox(vec(sum(Lrw, dims=2)), zeros(size(W, 1)); atol=1e-12)
    end

    @testset "handles nonzero diagonal affinities" begin
        # This test checks that the implementation also works when W has
        # nonzero diagonal entries.
        #
        # A nonzero diagonal means that a sample has self-affinity.
        # The implementation should not assume that the diagonal is always zero.

        W = [1.0 2.0;
             2.0 1.0]

        Lrw = compute_laplacian(W, RandomWalkLaplacian())

        # Row sums are both 3.
        #
        # D⁻¹W =
        # [1/3  2/3;
        #  2/3  1/3]
        #
        # L_rw = I - D⁻¹W =
        # [2/3  -2/3;
        # -2/3   2/3]
        expected = [ 2/3  -2/3;
                    -2/3   2/3]

        @test Lrw ≈ expected
    end

    @testset "rejects non-square affinity matrix" begin
        # An affinity matrix must be square because it stores pairwise
        # similarities between the same set of samples.
        #
        # A 2 × 3 matrix cannot represent a valid graph Laplacian.

        W_non_square = [0.0 1.0 2.0;
                        1.0 0.0 3.0]

        @test_throws ArgumentError compute_laplacian(W_non_square, RandomWalkLaplacian())
    end

    @testset "rejects non-symmetric affinity matrix" begin
        # Spectral clustering assumes an undirected similarity graph here.
        #
        # That means W[i, j] must equal W[j, i].
        # This matrix violates that condition because W[1, 2] != W[2, 1].

        W_non_symmetric = [0.0 1.0;
                           0.0 0.0]

        @test_throws ArgumentError compute_laplacian(W_non_symmetric, RandomWalkLaplacian())
    end

    @testset "rejects negative affinity values" begin
        # Affinity values represent similarities.
        #
        # Negative similarities are not valid for this graph Laplacian.
        # Therefore, the function should throw an ArgumentError.

        W_negative = [ 0.0 -1.0;
                      -1.0  0.0]

        @test_throws ArgumentError compute_laplacian(W_negative, RandomWalkLaplacian())
    end

    @testset "rejects zero-degree nodes" begin
        # The random-walk Laplacian uses D⁻¹.
        #
        # If a node has degree 0, D⁻¹ would require division by zero.
        # Therefore, isolated nodes with degree 0 must be rejected.

        W_zero_degree = [0.0 0.0;
                         0.0 0.0]

        @test_throws ArgumentError compute_laplacian(W_zero_degree, RandomWalkLaplacian())
    end
end

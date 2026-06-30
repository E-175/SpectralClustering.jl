using Test
using SpectralClustering

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
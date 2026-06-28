using Test
using SpectralClustering
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
degreesD ≈ zeros(100) 
#Eigenvectors should compute
@test compute_eigenvectors(D,2) isa AbstractMatrix

E,F = make_moons()
G = compute_affinity(E,RBFKernel())
H = compute_laplacian(G,UnnormalizedLaplacian())

#Test Symmetry
@test H ≈ H'
#Sum in every row should be 0 as self affinity is 0
degreesH = vec(sum(H, dims=2))
degreesH ≈ zeros(100)
#Eigenvectors should compute
@test compute_eigenvectors(H,2) isa AbstractMatrix

J,K = make_blobs()
L = compute_affinity(J,RBFKernel())
M = compute_laplacian(L,UnnormalizedLaplacian())

#Test Symmetry
@test M ≈ M'
#Sum in every row should be 0 as self affinity is 0
degreesM = vec(sum(M, dims=2))
degreesM ≈ zeros(100)
#Eigenvectors should compute
@test compute_eigenvectors(M,3) isa AbstractMatrix

end

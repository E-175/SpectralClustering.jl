using Test
using SpectralClustering

@testset "Symmetric Laplacian" begin
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

end



@testset "Unnormalized Laplacian" begin
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
         0.0 0.0 0.0;
         0.0 0.0 1.0]

   @test_throws ArgumentError compute_laplacian(H,UnnormalizedLaplacian())

end

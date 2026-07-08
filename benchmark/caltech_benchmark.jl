using SpectralClustering
using Clustering # For the evaluation metrics
using MAT

# 1. Open the downloaded file from the repository
mat_path = joinpath(@__DIR__, "Caltech101_7.mat")
file = matopen(mat_path)

# 2. Extract the ground truth labels (named 'gnd' in this repo)
true_labels = Int.(vec(read(file, "gnd")))

# 3. Extract the features ('X')
X_cell = read(file, "X")

# Because it's multi-view, X_cell contains multiple feature matrices.
# Let's grab the very first view (e.g., HOG or Gabor features)
X_raw = X_cell[1] 

# The repo states X_raw is shaped (sample_number, feature_dimension).
# Your package expects (features, samples), so we MUST transpose it!
X = copy(X_raw')

close(file)

println("Loaded features shape: ", size(X))
println("Loaded labels length: ", length(true_labels))

# Now feed 'X' directly into your algorithm!
A = compute_affinity(X, LocalScaling(7))

# 2. Define the different algorithm combinations you want to test
configs = [
    # Increased sigma from 1.0 to 100.0 to prevent underflow in 48D space
    (name="Standard (RBF + KMeans)", aff=RBFKernel(100.0), disc=KMeansDiscretization(true)),
    (name="Local Scaling + KMeans", aff=LocalScaling(7), disc=KMeansDiscretization(true)),
    (name="Local Scaling + Self-Tuning", aff=LocalScaling(7), disc=SelfTuningDiscretization())
]

# 3. Evaluate each configuration
for conf in configs
    # Step A: Compute Affinity
    A_matrix = compute_affinity(X, conf.aff)
    
    # Step B: Compute Laplacian (Using your exported SymmetricLaplacian)
    L = compute_laplacian(A_matrix, SymmetricLaplacian())
    
    # Step C: Extract Eigenvectors (k=7 since there are 7 Caltech classes)
    V = compute_eigenvectors(L, 7)
    
    # Step D: Discretize
    predicted_labels = discretize(V, conf.disc; k=7)
    
    # Step E: Evaluate against Ground Truth
    # randindex returns a tuple; the 4th element is the Adjusted Rand Index
    ari = randindex(true_labels, predicted_labels)[4] 
    nmi = mutualinfo(true_labels, predicted_labels)
    
    println("=== $(conf.name) ===")
    println("ARI: ", round(ari, digits=3))
    println("NMI: ", round(nmi, digits=3))
    println("-------------------")
end
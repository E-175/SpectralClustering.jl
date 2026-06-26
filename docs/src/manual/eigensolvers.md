# Eigensolvers

After constructing the graph Laplacian, the core of Spectral Clustering is to compute its eigenvalues and eigenvectors. The first $k$ eigenvectors corresponding to the smallest eigenvalues contain the structural information needed to partition the graph into $k$ clusters.

`SpectralClustering.jl` provides the `compute_eigenvectors` function to seamlessly extract these vectors, returning them in a features × samples format (so the rows correspond to the eigenvectors).

```julia
using SpectralClustering

# Compute the first k eigenvectors
V = compute_eigenvectors(L, k)
```

## Visualizing the Eigenvectors

Why do we compute the eigenvectors in the first place? 

The original dataset might be tangled and non-linearly separable (like two concentric circles or interlocking moons). The eigenvectors map our data into a new $k$-dimensional space (often called the **spectral embedding**). In this new space, the complex geometric structures are "unrolled" and the clusters become simply linearly separable points that standard K-Means can easily cluster.

Let's visualize this transformation!

```@example eigensolvers
using SpectralClustering
using Plots
using LinearAlgebra
using Random

# 1. Generate interlocking moons (non-linearly separable)
rng = Xoshiro(42)
X, y = make_moons(rng, 300; noise=0.05)

# 2. Compute Affinity and Laplacian
W = compute_affinity(X, RBFKernel(sigma=0.1))
L = compute_laplacian(W, RandomWalkLaplacian())

# 3. Compute eigenvalues and eigenvectors
# We'll use LinearAlgebra.eigen here so we can also look at the eigenvalues!
# (In your own code, you can use `V = compute_eigenvectors(L, 3)` to just get the vectors).
F = eigen(L)
idx = sortperm(real.(F.values))
λ = real.(F.values[idx])
V = real.(F.vectors[:, idx])

# 4. Plot the Eigenvalues
# The "Eigengap" heuristic tells us the ideal number of clusters.
# A large jump between eigenvalue k and k+1 suggests k clusters.
p1 = bar(1:8, λ[1:8], 
    title="Bottom 8 Eigenvalues",
    xlabel="Index", 
    ylabel="Eigenvalue",
    legend=false,
    color=:coral
)

# 5. Plot the data in the Spectral Embedding (2nd vs 3rd eigenvector)
p2 = scatter(V[:, 2], V[:, 3], 
    group=y, 
    markerstrokewidth=0,
    markersize=4,
    title="Spectral Embedding",
    xlabel="2nd Eigenvector",
    ylabel="3rd Eigenvector",
    legend=:topright
)

plot(p1, p2, layout=(1, 2), size=(800, 400), framestyle=:box, margin=5Plots.mm)
```

#### 1. The Eigengap (Left Plot)
Notice that the bars for the first two eigenvalues ($k=1$ and $k=2$) are so small they are barely visible, essentially being zero. This mathematically confirms that there are **two distinct structural components** (clusters) in our data. The massive vertical jump (the "Eigengap") right after the 2nd eigenvalue acts as a perfect mathematical signal telling us we should configure our algorithm to search for $k=2$ clusters.

#### 2. The Spectral Embedding (Right Plot)
As you can see, when we map the data into the new space created by the 2nd and 3rd eigenvectors, the original tangled moons are pulled apart into two tight, isolated clusters that are now completely linearly separable!

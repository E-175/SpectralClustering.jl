# Getting Started

This guide will show you how to start using `SpectralClustering.jl` to perform clustering on non-linearly separable datasets.

## Basic Usage

Spectral clustering is particularly useful when the clusters in your data are not linearly separable (like circles or moons). 
Let's see how `SpectralClustering.jl` performs on the "moons" dataset.

```@example getting_started
using SpectralClustering
using Plots
using Random

# 1. Generate a non-linearly separable dataset
rng = Xoshiro(42)
X, y_true = make_moons(rng, 400, noise=0.05)

# 2. Perform Spectral Clustering
# We expect 2 clusters (k=2)
k = 2
y_pred = spectral_cluster(X, k)

# 3. Plot the results
p1 = scatter(X[1,:], X[2,:], group=y_true, title="Ground Truth", legend=false, markersize=3)
p2 = scatter(X[1,:], X[2,:], group=y_pred, title="Spectral Clustering", legend=false, markersize=3)

plot(p1, p2, layout=(1,2), size=(800, 400), margin=5Plots.mm)
```

*(Note: Because clustering is unsupervised, it assigns arbitrary integer labels to the groups it finds, meaning the predicted colors may appear swapped compared to the ground truth.)*

### Customizing the Algorithm

You can customize the different steps of the spectral clustering algorithm (affinity matrix construction, graph laplacian type, and discretization method) by passing keyword arguments to the `spectral_cluster` function.

```@example getting_started
# Using a different Laplacian and Kernel bandwidth
y_pred_custom = spectral_cluster(
    X, 2;
    affinity = RBFKernel(sigma=0.1),
    laplacian = RandomWalkLaplacian(),
    discretizer = KMeansDiscretization(true) # true for normalize_rows
)

scatter(X[1,:], X[2,:], group=y_pred_custom, title="Custom Spectral Clustering", legend=false, markersize=3)
```
## Self-Tuning Spectral Clustering

One of the biggest challenges in standard spectral clustering is manually choosing the global scale parameter (`sigma`) for the affinity matrix and guessing the correct number of clusters (`k`). 

`SpectralClustering.jl` implements **Self-Tuning Spectral Clustering** (based on Zelnik-Manor & Perona, 2004), which solves both of these issues automatically:

1. **Local Scaling (`LocalScaling`)**: Instead of a global `sigma`, it computes a local scale for each data point based on its distance to its `k`-th nearest neighbor. This allows the algorithm to perfectly handle data with multiple scales and varying densities.
2. **Automatic Cluster Selection (`SelfTuningDiscretization`)**: It analyzes the structure of the eigenvectors to find an optimal rotation matrix. This completely eliminates the need for the K-Means step and automatically determines the optimal number of clusters.

Here is how you can use the self-tuning features:

```@example getting_started
# We pass `nothing` for `k` so the algorithm finds the optimal number of clusters automatically!
y_pred_selftuning = spectral_cluster(
    X, 
    nothing; 
    affinity = LocalScaling(7), # 7 is a robust default for the k-th nearest neighbor
    discretizer = SelfTuningDiscretization()
)

scatter(X[1,:], X[2,:], group=y_pred_selftuning, title="Self-Tuning Clustering", legend=false, markersize=3)

# SpectralClustering.jl - Architecture & Team Assignments

This document outlines the architecture of the package, the file structure, and the specific feature assignments for the team. 

Core idea is to use **Multiple Dispatch** for the different variants of Spectral Clustering. We share a single pipeline (`api.jl`) and implement specific paper variants by defining custom `structs` (in `types.jl`) and methods that dispatch on them.

---

## Repository Structure

```text
SpectralClustering.jl/
├── Project.toml
├── src/
│   ├── SpectralClustering.jl  # Main module, includes files below
│   ├── types.jl               # All abstract types and structs
│   ├── affinity.jl            # Similarity graph construction (W)
│   ├── laplacians.jl          # Graph laplacian construction (L)
│   ├── eigensolvers.jl        # Arpack/KrylovKit wrappers (V)
│   ├── discretization.jl      # Turning eigenvectors into discrete labels
│   ├── datasets.jl            # Data generation (make_moons, etc.)
│   └── api.jl                 # Shared pipeline function: spectral_cluster()
└── test/
    └── runtests.jl
```

---

## Team Assignments


### 1. Unnormalized Spectral Clustering
**Assignee:** Christoph
*Builds the baseline algorithm.*
* **`types.jl`**: Define `UnnormalizedLaplacian`.
* **`affinities.jl`**: Implement the standard Gaussian (RBF) kernel with a global $\sigma$. (Already Done)
* **`laplacians.jl`**: Implement $L = D - W$.
* **`discretization.jl`**: Implement standard K-Means clustering on the eigenvectors.

### 2. Normalized Spectral Clustering (Shi & Malik / NJW)
**Assignee:** Carolin
*Builds the standard normalized cuts variants.*
* **`types.jl`**: Define `RandomWalkLaplacian`, `SymmetricLaplacian`.
* **`affinities.jl`**: (Reuses RBF Kernel).
* **`laplacians.jl`**: Implement Random Walk ($L_{rw} = I - D^{-1}W$) and Symmetric ($L_{sym} = I - D^{-1/2} W D^{-1/2}$) Laplacians.
* **`discretization.jl`**: Implement row-normalization for eigenvectors (needed for NJW) before passing to K-Means.

### 3. Self-Tuning Spectral Clustering (Zelnik-Manor & Perona)
**Assignee:** Jens
*Builds automatic scale estimation and cluster counting.*
* **`types.jl`**: Define `LocalScaling` affinity and `SelfTuningDiscretization`.
* **`affinities.jl`**: Implement Local Scaling (calculating a local $\sigma_i$ based on the $K$-th neighbor).
* **`laplacians.jl`**: (Reuses Symmetric Laplacian).
* **`discretization.jl`**: Implement the eigenvector rotation/alignment cost-minimization method to find $k$ and assign labels without K-Means.

### 4. Multi-Class Spectral Clustering (Yu & Shi)
**Assignee:** Janus
*Builds the optimal discrete solution using SVD.*
* **`types.jl`**: Define `YuShiDiscretization`.
* **`affinities.jl`**: (Reuses RBF Kernel).
* **`laplacians.jl`**: (Reuses Random Walk Laplacian).
* **`discretization.jl`**: Implement the iterative Singular Value Decomposition (SVD) and non-maximum suppression technique to replace standard K-Means.

---

## 📝 Coding Guidelines
* **Types/Structs:** Use `PascalCase` (e.g., `SymmetricLaplacian`).
* **Functions/Variables:** Use `snake_case` (e.g., `compute_affinity`, `noise_level`).
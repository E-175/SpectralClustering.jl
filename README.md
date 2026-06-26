# SpectralClustering.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://E-175.github.io/SpectralClustering/dev/)
[![Build Status](https://github.com/E-175/SpectralClustering/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/E-175/SpectralClustering/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/E-175/SpectralClustering/branch/main/graph/badge.svg)](https://codecov.io/gh/E-175/SpectralClustering)

A Julia package for performing non-convex and non-linearly separable spectral clustering. This package is built with a highly modular pipeline, allowing you to seamlessly swap out different Affinities, Graph Laplacians, and Discretization algorithms.

## Installation

To install and use this package, run the following in the Julia REPL (press `]` to enter package mode):

```julia
pkg> add https://github.com/E-175/SpectralClustering.jl
```

## Quick Start

`SpectralClustering.jl` comes with synthetic dataset generators (`make_moons`, `make_circles`, `make_blobs`) so you can test algorithms right out of the box!

```julia
using SpectralClustering
using Plots

# 1. Generate 400 samples of non-linearly separable data
X, y_true = make_moons(400, noise=0.05)

# 2. Perform Spectral Clustering into 2 clusters
y_pred = spectral_cluster(X, 2)

# 3. Visualize the clustered results!
scatter(X[1,:], X[2,:], group=y_pred, legend=false, title="Spectral Clustering")
```

For advanced usage (like customizing the `RBFKernel` bandwidth or using the `LocalScaling` affinity), check out our [Documentation](https://E-175.github.io/SpectralClustering/dev/)!

## Demos

The `demo/` directory contains scripts showcasing the package's capabilities:
- `data_generation_demo.jl`: Demonstrates how to generate and visualize different synthetic datasets.

## Course Material

- [Task Description PDF](course_material/SpecClust.pdf)
- [Project Topic Details](course_material/ProjectTopic.png)

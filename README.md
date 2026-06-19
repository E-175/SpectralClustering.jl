# SpectralClustering

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://E-175.github.io/SpectralClustering/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://E-175.github.io/SpectralClustering/dev/)
[![Build Status](https://github.com/E-175/SpectralClustering/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/E-175/SpectralClustering/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/E-175/SpectralClustering/branch/main/graph/badge.svg)](https://codecov.io/gh/E-175/SpectralClustering)


# SpectralClustering (Group A)
A Julia package for performing non-convex and non-linearly separable spectral clustering.

## Getting Started

### Prerequisites

- **Julia:** v1.11 or higher.

### Installation

To install and use this package locally for development, run the following in the Julia REPL:

```julia
using Pkg
Pkg.develop(url="https://github.com/E-175/SpectralClustering")
```

### Usage

Here is a quick example of how to load the package and generate a non-linearly separable dataset (e.g., interleaving half-moons):

```julia
using SpectralClustering

# Generate 500 samples with some noise
X, y = make_moons(500, noise=0.05)
```

## Demos

The `demo/` directory contains scripts to showcase the package's capabilities:
- `data_generation_demo.jl`: Demonstrates how to generate non-linearly separable datasets (e.g., concentric circles, moons).

## Course Material

- [Task Description PDF](course_material/SpecClust.pdf)
- [Project Topic Details](course_material/ProjectTopic.png)
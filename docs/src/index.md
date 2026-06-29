# SpectralClustering.jl

A Julia package for performing non-convex and non-linearly separable spectral clustering.

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://E-175.github.io/SpectralClustering/dev/)
[![Build Status](https://github.com/E-175/SpectralClustering/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/E-175/SpectralClustering/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/E-175/SpectralClustering/branch/main/graph/badge.svg)](https://codecov.io/gh/E-175/SpectralClustering)

Welcome to the documentation for `SpectralClustering.jl`! 
This package provides a modular pipeline for performing spectral clustering, including configurable affinity kernels, graph Laplacians, and discretization methods.

To get started, see the [Getting Started](@ref) guide.

## References

This package implements various spectral clustering algorithms based on the following papers:

- **[1]** Ulrike von Luxburg. "[A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189.pdf)." *Statistics and Computing*, 2007.
- **[2]** Jianbo Shi and Jitendra Malik. "[Normalized cuts and image segmentation](https://doi.org/10.1109/34.868688)." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2000.
- **[3]** Lihi Zelnik-Manor and Pietro Perona. "[Self-Tuning Spectral Clustering](https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf)." *Advances in Neural Information Processing Systems*, 2004.
- **[4]** Stella X. Yu and Jianbo Shi. "[Multiclass spectral clustering](https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf)." *IEEE International Conference on Computer Vision*, 2003.

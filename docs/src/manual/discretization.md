# Discretization

Discretization is the final step in spectral clustering. After computing a
spectral embedding, the algorithm converts the continuous eigenvector
coordinates into discrete cluster labels.

In `SpectralClustering.jl`, the embedding matrix is expected to have shape
`n_features × n_samples`, with one sample per column. The package provides
three discretization methods.

## KMeans

`KMeansDiscretization` applies K-Means to the columns of the spectral
embedding. In the standard spectral clustering pipeline, the embedding usually
contains one selected eigenvector per requested cluster.

This is the default discretizer used by `spectral_cluster`. It is a good
general-purpose choice when the spectral embedding already separates the samples
well.

```@example discretization
using SpectralClustering

discretizer = KMeansDiscretization(true)
```

Set `normalize_samples=true` to normalize each sample vector before clustering.
This is commonly used with normalized spectral clustering variants. The full
constructor also allows switching to the manual implementation with
`KMeansDiscretization(normalize_samples, use_manual_implementation)`.

## SelfTuning

`SelfTuningDiscretization` uses a rotation-based discretization method instead
of K-Means. If `k` is provided, it uses the first `k` eigenvectors of the
embedding. If `k` is omitted, it evaluates several candidate cluster counts and
selects the one with the lowest alignment cost.

When used through `spectral_cluster(X, nothing; discretizer=...)`, automatic
model selection is enabled and `k_max` bounds the largest number of clusters
considered.

```@example discretization
self_tuning = SelfTuningDiscretization(k_max=8)
```

This method is useful when the number of clusters is not known in advance.

## SVD

`SVDDiscretization` implements the Yu-Shi multiclass discretization method.
Like `SelfTuningDiscretization`, it works directly on the spectral embedding,
but it requires an explicit `k` and expects the embedding to contain exactly
`k` eigenvectors.

```@example discretization
svd_discretizer = SVDDiscretization()
```

This method is most appropriate when the number of clusters is known and you
want a deterministic discretization method based on singular value
decomposition.

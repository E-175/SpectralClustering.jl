# Discretization

The discretization is the final step in spectral clustering. In this step the datapoints are separated into the respective Clusters. `SpectralClustering.jl` implements three different methods for discretization.

## KMeans

KMeans receives k eigenvectors (the k vectors with the smalles corresponging eigenvalues) as well as the number of clusters k that the dataset should be partitioned into. KMeans also works with a different number of eigenvectors though using a different number of eigenvectors is detrimental for the accuracy of the algorithm. KMeans can also be used on the dataset directly. This is highly effective for datasets in which the datapoints are grouped together in convex groups that are isolated from another (like make_blobs), but struggles in cases where the clusters are not convex (like make_circles).

## SelfTuning

SelfTuning receives a matrix containing eigenvectors and may or may not also receive the number of (desired) Clusters k. If k is not provided SelfTuning automatically determines the optimal number for k based on a cost function.

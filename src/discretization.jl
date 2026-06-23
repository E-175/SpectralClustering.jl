# Fallback method
function discretize(V::AbstractMatrix, method::AbstractDiscretization; k::Union{Int, Nothing}=nothing)
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

"""
    discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)

Discretize a spectral embedding into cluster labels using K-Means.

The input matrix `V` is the eigenvector embedding produced by spectral clustering.
Each row of `V` represents one sample, and each column represents one selected
eigenvector.

K-Means is applied to the rows of `V`, so samples with similar coordinates in
the spectral embedding are assigned to the same cluster.

If `method.normalize_rows` is `true`, each row of `V` is normalized to unit
length before K-Means is applied. This is used for variants such as the
Ng-Jordan-Weiss normalized spectral clustering method.

# Arguments
- `V`: Spectral embedding matrix with one row per sample.
- `method`: K-Means discretization configuration.
- `k`: Number of clusters.

# Returns
A vector of cluster labels with one label per sample.

# Throws
- `ArgumentError` if `k` is not provided.
- `ArgumentError` if `k` is smaller than 1 or larger than the number of samples.
- `ArgumentError` if row normalization is requested and at least one row has norm zero.
"""
function discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)
    # The K-Means expects a fixed number of clusters.
    isnothing(k) && throw(ArgumentError("K-Means requires a specific number of clusters 'k'."))

    # The rows of V correspond to samples.
    # The columns of V correspond to selected eigenvectors.
    n_samples, n_eigenvectors = size(V)

    # k must be meaningful for the number of available samples.
    1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of samples."))

    # The embedding must contain at least one eigenvector.
    n_eigenvectors >= 1 || throw(ArgumentError("The spectral embedding must contain at least one eigenvector."))

    # Work on a floating-point copy so the input matrix is not modified.
    embedding = Matrix{Float64}(V)
    
    if method.normalize_rows
        # Compute the Euclidean norm of each row.
        # This measures the length of each embedded sample vector.
        row_norms = sqrt.(sum(abs2, embedding, dims=2))

        # A row with norm zero cannot be normalized.
        all(row_norms .> 0) || throw(ArgumentError("Cannot normalize rows with zero norm."))

        # Normalize every row to unit length.
        embedding ./= row_norms
    end
    
    # Clustering.kmeans expects data in the format features × samples.
    # Since embedding is samples × features (eigenvectors), we transpose it.
    result = kmeans(Matrix(embedding)', k)

    # Return one cluster label per sample.
    return assignments(result)
end

# ---------------------------------------------------------
# TODO: Jens (Self-Tuning)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::SelfTuningDiscretization; k::Union{Int, Nothing}=nothing)
    # TODO: Implement discretization for self tuning
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

# ---------------------------------------------------------
# TODO: Janus (Multi-Class)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::SVDDiscretization; k::Union{Int, Nothing}=nothing)
    if isnothing(k)
        error("SVD Discretization requires a specific number of clusters 'k'.")
    end
    # TODO: Implement.
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

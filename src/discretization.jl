# Fallback method
function discretize(V::AbstractMatrix, method::AbstractDiscretization; k::Union{Int, Nothing}=nothing)
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

function discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)
    # The K-Means expects a fixed number of clusters.
    isnothing(k) && throw(ArgumentError("K-Means requires a specific number of clusters 'k'."))

    if !isnothing(method.seed)
        Random.seed!(method.seed)
    end

    # The columns of V correspond to samples.
    # The rows of V correspond to selected eigenvectors.
    n_eigenvectors, n_samples = size(V)

    # k must be meaningful for the number of available samples.
    1 <= k <= n_samples || throw(ArgumentError("k must be between 1 and the number of samples."))

    # The embedding must contain at least one eigenvector.
    n_eigenvectors >= 1 || throw(ArgumentError("The spectral embedding must contain at least one eigenvector."))

    # Work on a floating-point copy so the input matrix is not modified.
    embedding = Matrix{Float64}(V)
    
    if method.normalize_samples
        # Compute the Euclidean norm of each column.
        # This measures the length of each embedded sample vector.
        col_norms = sqrt.(sum(abs2, embedding, dims=1))

        # A column with norm zero cannot be normalized.
        all(col_norms .> 0) || throw(ArgumentError("Cannot normalize columns with zero norm."))

        # Normalize every column to unit length.
        embedding ./= col_norms
    end
    

    # Manual implementation of KMeans
    if method.use_manual_implementation
        timeout = 1000
        currentCentroids = V[:,randperm(size(V,2))[1:k]]
        nearestCentroids = zeros(Int,size(V,2))
        for _ in 1:timeout
            # Currently uses euclidian distance / L2 Norm
            distancesMatrix = [norm(V[:,x] - currentCentroids[:,y]) for x = 1:size(V,2), y = 1:k]
            newNearestCentroids = vec([i[2] for i in argmin(distancesMatrix,dims=2)])
            if newNearestCentroids == nearestCentroids
                break
            else
                nearestCentroids = newNearestCentroids
            end
            # Move Centroid of Cluster
            for currentClusterIndex in 1:k
                currentCluster = V[:,nearestCentroids .== currentClusterIndex]
                if isempty(currentCluster)
                    currentCentroids[:,currentClusterIndex] = V[:,rand(1:size(V,2))]
                else
                    currentCentroids[:,currentClusterIndex] = vec(mean(currentCluster,dims=2))
                end
            end
        end
        return nearestCentroids
    end

    # Clustering.kmeans expects data in the format features × samples.
    result = kmeans(embedding, k)

    # Return one cluster label per sample.
    return assignments(result)
end

# ---------------------------------------------------------
# TODO: Jens (Self-Tuning)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::SelfTuningDiscretization; k::Union{Int, Nothing}=nothing)
    # TODO: Implement discretization for self tuning
end

# ---------------------------------------------------------
# TODO: Janus (Multi-Class)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::SVDDiscretization; k::Union{Int, Nothing}=nothing)
    if isnothing(k)
        error("SVD Discretization requires a specific number of clusters 'k'.")
    end
    # TODO: Implement.
    # return labels
end
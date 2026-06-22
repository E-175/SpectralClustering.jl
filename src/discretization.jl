# Fallback method
function discretize(V::AbstractMatrix, method::AbstractDiscretization; k::Union{Int, Nothing}=nothing)
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

function discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)
    if isnothing(k)
        error("K-Means requires a specific number of clusters 'k'.")
    end
    
    
    # TODO: Apply K-Means clustering algorithm on rows of V


    #Either this or Julias standard function, still unclear 


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
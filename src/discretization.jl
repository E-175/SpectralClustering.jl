# Fallback method
function discretize(V::AbstractMatrix, method::AbstractDiscretization; k::Union{Int, Nothing}=nothing)
    error("Discretization method $(typeof(method)) is not implemented yet.")
end

# ---------------------------------------------------------
# TODO: Christoph & Carolin (Standard & Normalized)
# ---------------------------------------------------------
function discretize(V::AbstractMatrix, method::KMeansDiscretization; k::Union{Int, Nothing}=nothing)
    if isnothing(k)
        error("K-Means requires a specific number of clusters 'k'.")
    end
    
    if method.normalize_rows
        # TODO for Carolin: Normalize rows of V to unit length
    end
    
    # TODO: Apply K-Means clustering algorithm on rows of V
    # return labels
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
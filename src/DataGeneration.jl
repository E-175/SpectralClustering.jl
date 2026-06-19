using Random

"""
    make_circles(n_samples::Int=100; noise::Float64=0.0, factor::Float64=0.8)

Generate a large circle containing a smaller circle in 2D.

# Arguments
- `n_samples`: Total number of points to generate (split equally between the two circles).
- `noise`: Standard deviation of Gaussian noise added to the data.
- `factor`: Scale factor between inner and outer circle (radius of inner circle / radius of outer circle).

# Returns
- `X`: A `2 × n_samples` Matrix of Float64.
- `y`: A `n_samples` Vector of Int (cluster labels `1` and `2`).
"""
function make_circles(n_samples::Int=100; noise::AbstractFloat=0.0, factor::AbstractFloat=0.8)
    if factor >= 1.0 || factor < 0.0
        throw(ArgumentError("factor must be in [0, 1)"))
    end

    # Split samples between the two circles
    n_samples_out = n_samples ÷ 2
    n_samples_in = n_samples - n_samples_out

    # Generate angles
    linspace_out = range(0, stop=2*pi, length=n_samples_out)
    linspace_in = range(0, stop=2*pi, length=n_samples_in)

    # Outer circle (radius 1)
    outer_circ_x = cos.(linspace_out)
    outer_circ_y = sin.(linspace_out)

    # Inner circle (radius `factor`)
    inner_circ_x = cos.(linspace_in) .* factor
    inner_circ_y = sin.(linspace_in) .* factor

    # Combine X and Y coordinates into a (2 × n_samples) matrix
    X = vcat(
        vcat(outer_circ_x, inner_circ_x)', # Row 1: X-coordinates
        vcat(outer_circ_y, inner_circ_y)'  # Row 2: Y-coordinates
    )

    # Create target vector with class labels 1 (outer) and 2 (inner)
    y = vcat(fill(1, n_samples_out), fill(2, n_samples_in))

    # Add Gaussian noise to the coordinates
    if noise > 0.0
        X .+= noise .* randn(size(X))
    end

    return X, y
end

"""
    make_moons(n_samples::Int=100; noise::Float64=0.0)

Generate two interleaving half circles in 2D.

# Arguments
- `n_samples`: Total number of points.
- `noise`: Standard deviation of Gaussian noise.

# Returns
- `X`: A `2 × n_samples` Matrix of Float64.
- `y`: A `n_samples` Vector of Int (cluster labels `1` and `2`).
"""
function make_moons(n_samples::Int=100; noise::AbstractFloat=0.0)

    # Split samples between the two moons
    n_samples_upper = n_samples ÷ 2
    n_samples_lower = n_samples - n_samples_upper

    # Generate angles
    linspace_upper = range(0, stop=pi, length=n_samples_upper)
    linspace_lower = range(0, stop=pi, length=n_samples_lower)

    # Upper moon (radius 1)
    upper_moon_x = cos.(linspace_upper)
    upper_moon_y = sin.(linspace_upper)

    # Lower moon (radius 1, shifted down and right)
    lower_moon_x = 1.0 .- cos.(linspace_lower)
    lower_moon_y = 0.5 .- sin.(linspace_lower)

    # Combine X and Y coordinates into a (2 × n_samples) matrix
    X = vcat(
        vcat(upper_moon_x, lower_moon_x)', # Row 1: X-coordinates
        vcat(upper_moon_y, lower_moon_y)'  # Row 2: Y-coordinates
    )

    # Create target vector with class labels 1 (upper) and 2 (lower)
    y = vcat(fill(1, n_samples_upper), fill(2, n_samples_lower))

    # Add Gaussian noise to the coordinates
    if noise > 0.0
        X .+= noise .* randn(size(X))
    end

    return X, y
end

"""
    make_blobs(n_samples::Int=100; centers::Int=3, cluster_std::Float64=1.0)

Generate isotropic Gaussian blobs for clustering.

# Arguments
- `n_samples`: Total number of points.
- `centers`: Number of clusters.
- `cluster_std`: Standard deviation of the clusters.

# Returns
- `X`: A `2 × n_samples` Matrix of Float64.
- `y`: A `n_samples` Vector of Int (cluster labels `1` to `centers`).
"""
function make_blobs(n_samples::Int=100; centers::Int=3, cluster_std::Real=1.0, center_box::Tuple{Real, Real}=(-10.0, 10.0))
    n_per_cluster = [n_samples ÷ centers for i in 1:centers]
    # Distribute the remainder
    for i in 1:(n_samples % centers)
        n_per_cluster[i] += 1
    end

    # Generate random center positions in a box [center_box[1], center_box[2]]
    center_pos = (center_box[2] - center_box[1]) .* rand(2, centers) .+ center_box[1]

    T = typeof(float(cluster_std))
    X = zeros(T, 2, n_samples)
    y = zeros(Int, n_samples)

    start_idx = 1
    for c in 1:centers
        n = n_per_cluster[c]
        end_idx = start_idx + n - 1
        
        # Generate 'n' points normally distributed around the cluster's center
        X[:, start_idx:end_idx] .= center_pos[:, c] .+ cluster_std .* randn(2, n)
        
        # Assign cluster label 'c' to these points
        y[start_idx:end_idx] .= c
        
        start_idx = end_idx + 1
    end

    return X, y
end

### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7a5a3800-5db6-11f1-1b67-9d5fc62b6f44
begin
    import Pkg
    # This points Pluto to the parent directory of your notebook
    Pkg.develop(path=joinpath(@__DIR__, ".."))
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
end

# ╔═╡ 8dccb3ee-ad5c-4b25-a65c-feae111e2803
begin
    using SpectralClustering
    using Plots
    using PlutoUI
end

# ╔═╡ f1a79c3b-5a1e-436d-9b8c-1e8c9f5d3b2a
md"""
# Data Generation Demo
This notebook demonstrates the data generation functions available in `SpectralClustering.jl`. We use `Plots.jl` to visualize the generated synthetic datasets and `PlutoUI` to provide interactive sliders for the parameters.
"""

# ╔═╡ d2e5c6a1-9b3f-48d2-a7e1-8f5c3b4a2d9e
md"""
## Circles Dataset
The `make_circles` function generates a large circle containing a smaller circle in 2D.

**Number of samples:** $(@bind n_samples_circ Slider(10:10:2000, default=500, show_value=true))
**Noise:** $(@bind noise_circ Slider(0.0:0.01:1.0, default=0.05, show_value=true))
**Factor:** $(@bind factor_circ Slider(0.0:0.01:0.99, default=0.5, show_value=true))
"""

# ╔═╡ a9b8c7d6-e5f4-4123-b8a7-c6d5e4f3a2b1
begin
    X_circ, y_circ = make_circles(n_samples_circ, noise=noise_circ, factor=factor_circ)
    scatter(X_circ[1, :], X_circ[2, :], group=y_circ, title="Generated circles data", aspect_ratio=:equal, legend=false, markersize=3, markerstrokewidth=0)
end

# ╔═╡ c5b4a392-8170-4e3f-a9b8-c7d6e5f4a3b2
md"""
## Moons Dataset
The `make_moons` function generates two interleaving half circles in 2D.

**Number of samples:** $(@bind n_samples_moons Slider(10:10:2000, default=500, show_value=true))
**Noise:** $(@bind noise_moons Slider(0.0:0.01:1.0, default=0.05, show_value=true))
"""

# ╔═╡ e1f2a3b4-c5d6-47e8-9f0a-1b2c3d4e5f6a
begin
    X_moons, y_moons = make_moons(n_samples_moons, noise=noise_moons)
    scatter(X_moons[1, :], X_moons[2, :], group=y_moons, title="Generated half moons data", aspect_ratio=:equal, legend=false, markersize=3, markerstrokewidth=0)
end

# ╔═╡ b8a7c6d5-e4f3-4a2b-8c1d-9e0f1a2b3c4d
md"""
## Blobs Dataset
The `make_blobs` function generates isotropic Gaussian blobs for clustering.

**Number of samples:** $(@bind n_samples_blobs Slider(10:10:2000, default=500, show_value=true))
**Centers:** $(@bind centers_blobs Slider(1:10, default=4, show_value=true))
**Cluster Std:** $(@bind cluster_std_blobs Slider(0.1:0.1:5.0, default=0.8, show_value=true))
"""

# ╔═╡ d5e4f3a2-b1c0-4d9e-8f7a-6b5c4d3e2f1a
begin
    X_blobs, y_blobs = make_blobs(n_samples_blobs, centers=centers_blobs, cluster_std=cluster_std_blobs)
    scatter(X_blobs[1, :], X_blobs[2, :], group=y_blobs, title="Generate blobs data", aspect_ratio=:equal, legend=false, markersize=3, markerstrokewidth=0)
end

# ╔═╡ Cell order:
# ╟─f1a79c3b-5a1e-436d-9b8c-1e8c9f5d3b2a
# ╠═7a5a3800-5db6-11f1-1b67-9d5fc62b6f44
# ╠═8dccb3ee-ad5c-4b25-a65c-feae111e2803
# ╟─d2e5c6a1-9b3f-48d2-a7e1-8f5c3b4a2d9e
# ╠═a9b8c7d6-e5f4-4123-b8a7-c6d5e4f3a2b1
# ╟─c5b4a392-8170-4e3f-a9b8-c7d6e5f4a3b2
# ╠═e1f2a3b4-c5d6-47e8-9f0a-1b2c3d4e5f6a
# ╟─b8a7c6d5-e4f3-4a2b-8c1d-9e0f1a2b3c4d
# ╠═d5e4f3a2-b1c0-4d9e-8f7a-6b5c4d3e2f1a

using SpectralClustering
using Documenter

# Set headless mode for Plots.jl to avoid opening windows during doc build
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(SpectralClustering, :DocTestSetup, :(using SpectralClustering); recursive=true)

makedocs(;
    modules=[SpectralClustering],
    authors="Jens Eckert <j.eckert.1@campus.tu-berlin.de>, Christoph Seidelmann <c.seidelmann@campus.tu-berlin.de>, Carolin Witschonke <c.witschonke@campus.tu-berlin.de>, Janus Trotzer <trotzer@campus.tu-berlin.de>",
    sitename="SpectralClustering.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://E-175.github.io/SpectralClustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Getting Started" => "getting_started.md",
        "Manual" => [
            "Datasets" => "manual/datasets.md",
            "Affinities" => "manual/affinities.md",
            "Laplacians" => "manual/laplacians.md",
            "Eigensolvers" => "manual/eigensolvers.md",
            "Discretization" => "manual/discretization.md",
        ],
        "API Reference" => "api.md",
        "Getting Started" => "getting_started.md",
    ],
)

deploydocs(;
    repo="github.com/E-175/SpectralClustering.jl",
    devbranch="main",
)

using SpectralClustering
using Documenter

DocMeta.setdocmeta!(SpectralClustering, :DocTestSetup, :(using SpectralClustering); recursive=true)

makedocs(;
    modules=[SpectralClustering],
    authors="Jens Eckert <j.eckert.1@campus.tu-berlin.de>",
    sitename="SpectralClustering.jl",
    format=Documenter.HTML(;
        canonical="https://E-175.github.io/SpectralClustering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/E-175/SpectralClustering.jl",
    devbranch="main",
)

# Installation

`SpectralClustering.jl` requires **Julia v1.11 or higher**.

## Using the Package Manager

To install the package directly from the GitHub repository, open the Julia REPL and press `]` to enter Pkg mode. Then run:

```julia
pkg> add https://github.com/E-175/SpectralClustering.jl
```

## Local Development

If you want to contribute to the package, view the source code, or run the included demos locally, you can clone the repository and instantiate the environment:

```bash
git clone https://github.com/E-175/SpectralClustering.jl
cd SpectralClustering.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

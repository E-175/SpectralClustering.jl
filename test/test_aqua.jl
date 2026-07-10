using Aqua
using SpectralClustering

@testset "Aqua.jl" begin
  Aqua.test_all(SpectralClustering; stale_deps=(ignore=[:Aqua, :MAT],))
end
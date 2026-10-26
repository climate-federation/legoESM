# SurfaceFluxes.jl v1.5.1 stability functions on a zeta grid.
# Writes: psi_m/psi_h for (a) parameter sets equal to legoESM's coefficients and
# (b) the ClimaParams defaults; plus the default parameter values themselves.
using SurfaceFluxes, ClimaParams, DelimitedFiles
const UF = SurfaceFluxes.UniversalFunctions
out = ARGS[1]
zg = vec(readdlm(ARGS[2]))
M = UF.MomentumTransport(); H = UF.HeatTransport()
sets = Dict(
  "businger_lego" => UF.BusingerParams{Float64}(Pr_0=1.0, a_m=5.0, a_h=5.0, b_m=16.0, b_h=16.0),
  "gryanik_lego"  => UF.GryanikParams{Float64}(Pr_0=0.98, a_m=5.0, a_h=5.0, b_m=0.3, b_h=0.4, b_m_unstable=16.0, b_h_unstable=16.0),
  "grachev_lego"  => UF.GrachevParams{Float64}(Pr_0=1.0, a_m=5.0, a_h=5.0, b_m=5.0/6.5, b_h=5.0, c_h=3.0, b_m_unstable=16.0, b_h_unstable=16.0),
  "businger_default" => UF.BusingerParams(Float64),
  "gryanik_default"  => UF.GryanikParams(Float64),
  "grachev_default"  => UF.GrachevParams(Float64),
)
open(out * "_params.txt", "w") do io
  for (k, p) in sort(collect(sets), by=first)
    println(io, k, " ", p)
  end
end
for (k, p) in sets
  A = zeros(length(zg), 3)
  for (i, z) in enumerate(zg)
    A[i, 1] = z; A[i, 2] = UF.psi(p, z, M); A[i, 3] = UF.psi(p, z, H)
  end
  writedlm(out * "_" * k * ".txt", A)
end
# the source Julia actually loaded, checked by make_fixture.py
println("SF_SRC ", pkgdir(SurfaceFluxes))
println("SF_V151_OK ", pkgversion(SurfaceFluxes), " ClimaParams ", pkgversion(ClimaParams))

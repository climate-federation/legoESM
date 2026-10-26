# SurfaceFluxes.jl v0.12.0 (last tag before Beljaars was removed) Beljaars-Holtslag psi.
using SurfaceFluxes, ClimaParams, DelimitedFiles
const UF = SurfaceFluxes.UniversalFunctions
out = ARGS[1]
zg = vec(readdlm(ARGS[2]))
M = UF.MomentumTransport(); H = UF.HeatTransport()
plego = UF.BeljaarsParams{Float64}(Pr_0=1.0, a_m=1.0, a_h=1.0, b_m=0.667, b_h=0.667, c_m=5.0, c_h=5.0, d_m=0.35, d_h=0.35, ζ_a=0.0, γ=0.0)
pdef = UF.BeljaarsParams(Float64)
open(out * "_params.txt", "w") do io
  println(io, "beljaars_lego ", plego); println(io, "beljaars_default ", pdef)
end
for (k, p) in ("beljaars_lego" => plego, "beljaars_default" => pdef)
  uf = UF.Beljaars{Float64, typeof(p)}(1.0, p)
  A = zeros(length(zg), 3)
  for (i, z) in enumerate(zg)
    A[i, 1] = z; A[i, 2] = UF.psi(uf, z, M); A[i, 3] = UF.psi(uf, z, H)
  end
  writedlm(out * "_" * k * ".txt", A)
end
# the source Julia actually loaded, checked by make_fixture.py
println("SF_SRC ", pkgdir(SurfaceFluxes))
println("SF_V0120_OK ", pkgversion(SurfaceFluxes), " ClimaParams ", pkgversion(ClimaParams))

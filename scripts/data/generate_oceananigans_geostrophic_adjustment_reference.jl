# Geostrophic-adjustment reference for the Oceananigans free-surface COUPLING node.
#
# A DETERMINISTIC test of the free-surface predictor-corrector coupling + Coriolis
# + time-stepper (the §5-residual's suspected home, per the Phase-1 tendency match
# which proved the momentum tendency itself faithful). An unbalanced free-surface
# bump (η Gaussian, u=v=0) radiates inertia-gravity waves and adjusts toward
# geostrophic balance. Both codes start IDENTICAL; the η/u/v evolution is compared.
#
# Same grid as the bickley/tendency references (unit sphere, rotation=1, g=1,
# WENOVectorInvariant(9), ImplicitFreeSurface) so the bridge is the validated one.
#
# Run:
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_geostrophic_adjustment_reference.jl <out_dir> [Nh] [nsteps] [dt]

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using NCDatasets
using Printf

out_dir = length(ARGS) >= 1 ? ARGS[1] : "."
Nh      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
nsteps  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40
Δt      = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.01
mkpath(out_dir)

Nφ = Int(Nh / 2)
grid = LatitudeLongitudeGrid(size = (Nh, Nφ, 1), radius = 1,
                             longitude = (-180, 180), latitude = (-80, 80),
                             z = (0, 1), halo = (7, 7, 7))

# Unbalanced free-surface bump (Gaussian in lon/lat), zero velocity.
η₀ = 0.1
λ0, φ0, σ = 0.0, 0.0, 30.0   # centre (deg) and width (deg)
ηᵢ(λ, φ, z) = η₀ * exp(-((λ - λ0)^2 + (φ - φ0)^2) / (2σ^2))

momentum_advection = WENOVectorInvariant(vorticity_order = 9)
free_surface = ImplicitFreeSurface(gravitational_acceleration = 1)
model = HydrostaticFreeSurfaceModel(grid; momentum_advection,
                                    tracer_advection = WENO(),
                                    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1),
                                    free_surface, buoyancy = nothing)
set!(model, η = ηᵢ)

dump_times = unique(Int.(round.(range(0, nsteps, length = 5))))
ηs = Dict{Int, Array{Float64,2}}(); us = Dict{Int,Array{Float64,2}}(); vs = Dict{Int,Array{Float64,2}}()
function snap!(it)
    ηs[it] = Array{Float64}(interior(model.free_surface.displacement)[:, :, 1])
    us[it] = Array{Float64}(interior(model.velocities.u)[:, :, 1])
    vs[it] = Array{Float64}(interior(model.velocities.v)[:, :, 1])
end
snap!(0)
for it in 1:nsteps
    time_step!(model, Δt)
    if it in dump_times; snap!(it); end
end
@printf("geo-adjust: Nh=%d nsteps=%d dt=%.3f  max|eta0|=%.4f max|u_final|=%.4e\n",
        Nh, nsteps, Δt, maximum(abs, ηs[0]), maximum(abs, us[nsteps]))

nc = joinpath(out_dir, "geostrophic_adjustment.nc")
NCDataset(nc, "c") do ds
    its = sort(collect(keys(ηs)))
    defDim(ds, "t", length(its))
    nxc, nyc = size(ηs[0]); defDim(ds, "xc", nxc); defDim(ds, "yc", nyc)
    nxu, nyu = size(us[0]); defDim(ds, "xu", nxu); defDim(ds, "yu", nyu)
    nxv, nyv = size(vs[0]); defDim(ds, "xv", nxv); defDim(ds, "yv", nyv)
    defVar(ds, "iters", its, ("t",))
    ev = defVar(ds, "eta", Float64, ("t", "xc", "yc"))
    uv = defVar(ds, "u",   Float64, ("t", "xu", "yu"))
    vv = defVar(ds, "v",   Float64, ("t", "xv", "yv"))
    for (i, it) in enumerate(its)
        ev[i, :, :] = ηs[it]; uv[i, :, :] = us[it]; vv[i, :, :] = vs[it]
    end
end
@info "DONE: wrote $nc"

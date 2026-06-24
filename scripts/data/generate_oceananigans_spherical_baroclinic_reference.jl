# Spherical baroclinic-adjustment reference — the SPHERE analog of baroclinic_adjustment.
#
# Identical physics to generate_oceananigans_baroclinic_adjustment_reference.jl (a
# BuoyancyTracer front b = N²z + Δb·ramp(y), WENOVectorInvariant momentum) but on a
# real LatitudeLongitudeGrid + HydrostaticSphericalCoriolis instead of the Cartesian
# RectilinearGrid + BetaPlane. Isolates ONLY the spherical cos(lat) METRIC in the
# baroclinic-eddy regime — the test the Cartesian precursor could not do (does the
# full-velocity vertical-advection fix hold on legoESM's lat-lon C-grid?).
#
# Region: a ~1000 km × 1000 km patch centred at φ₀=-45° (same scale as the Cartesian
# case): Δφ=9°, Δλ=Δφ/cos(45°)≈12.73°, periodic in λ, bounded (walls) in φ.
#
# Run:
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_spherical_baroclinic_reference.jl <out_dir> [stop_days] [dump_days]

using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using Random
Random.seed!(8675309)

ref_root  = length(ARGS) >= 1 ? ARGS[1] : "."
stop_days = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 30.0
dump_days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 6.0
out_dir = joinpath(ref_root, "spherical_baroclinic")
mkpath(out_dir)

φ₀ = -45.0
Δφ = 9.0                       # ~1000 km in latitude
Δλ = Δφ / cosd(φ₀)             # ~1000 km in longitude at φ₀ (≈12.73°)
Nx, Ny, Nz = 48, 48, 8
Lz = 1kilometers
grid = LatitudeLongitudeGrid(size = (Nx, Ny, Nz),
                             halo = (6, 6, 4),
                             longitude = (-Δλ/2, Δλ/2),
                             latitude = (φ₀ - Δφ/2, φ₀ + Δφ/2),
                             z = (-Lz, 0),
                             topology = (Periodic, Bounded, Bounded))

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = BuoyancyTracer(), tracers = :b,
                                    free_surface = ImplicitFreeSurface(),
                                    momentum_advection = WENOVectorInvariant(vorticity_order = 9),
                                    tracer_advection = WENO(order = 7))

# y measured from the front centre φ₀ (metres); same b-front as the Cartesian case.
Rₑ = grid.radius
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)
N² = 1e-5; M² = 1e-7; Δy = 100kilometers
Δb = Δy * M²; ϵb = 1e-2 * Δb
bᵢ(λ, φ, z) = N² * z + Δb * ramp(Rₑ * deg2rad(φ - φ₀), Δy) + ϵb * randn()
set!(model, b = bᵢ)

simulation = Simulation(model, Δt = 10minutes, stop_time = stop_days * days)
wizard = TimeStepWizard(cfl = 0.2, max_change = 1.1, max_Δt = 20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))
progress(sim) = @printf("t=%s iter=%d dt=%s max|u|=%.4e\n", prettytime(time(sim)),
    iteration(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u))
simulation.callbacks[:p] = Callback(progress, IterationInterval(200))

Nzs = Nz
bs = Dict{Int,Array{Float64,2}}(); us = Dict{Int,Array{Float64,2}}(); vs = Dict{Int,Array{Float64,2}}()
times = Float64[]
function snap!()
    push!(times, time(simulation)); i = length(times)
    bs[i] = Array{Float64}(interior(model.tracers.b)[:, :, Nzs])
    us[i] = Array{Float64}(interior(model.velocities.u)[:, :, Nzs])
    vs[i] = Array{Float64}(interior(model.velocities.v)[:, :, Nzs])
end
simulation.callbacks[:snap] = Callback(_ -> snap!(), TimeInterval(dump_days * days))
snap!()
run!(simulation)

@printf("spherical_baroclinic: stop=%.0fd, %d snaps, max|u_final|=%.4e\n",
        stop_days, length(times), maximum(abs, us[length(times)]))
nc = joinpath(out_dir, "spherical_baroclinic.nc")
NCDataset(nc, "c") do ds
    nt = length(times); defDim(ds, "t", nt)
    nxb, nyb = size(bs[1]); defDim(ds, "xb", nxb); defDim(ds, "yb", nyb)
    nxu, nyu = size(us[1]); defDim(ds, "xu", nxu); defDim(ds, "yu", nyu)
    nxv, nyv = size(vs[1]); defDim(ds, "xv", nxv); defDim(ds, "yv", nyv)
    defVar(ds, "times_s", times, ("t",))
    bv = defVar(ds, "b", Float64, ("t","xb","yb"))
    uv = defVar(ds, "u", Float64, ("t","xu","yu"))
    vv = defVar(ds, "v", Float64, ("t","xv","yv"))
    for i in 1:nt; bv[i,:,:] = bs[i]; uv[i,:,:] = us[i]; vv[i,:,:] = vs[i]; end
end
@info "DONE: wrote $nc"

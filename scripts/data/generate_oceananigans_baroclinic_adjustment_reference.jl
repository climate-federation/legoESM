# Baroclinic-adjustment reference (the §5 precursor) for the Oceananigans recipe.
#
# Oceananigans' canonical baroclinic-instability example: a buoyancy front on a
# beta-plane goes baroclinically unstable and forms eddies. SAME physics as §5
# (BuoyancyTracer + WENO vector-invariant + stratified front) but CLEAN: no
# restoring forcing, no free-slip-wall stress, standard 48x48x8 resolution. If
# legoESM matches THIS, the §5 residual is specific to §5's forcing/resolution/walls.
#
# Dumps b/u/v time series (Center/Face) for the legoESM<->Oceananigans comparison.
#
# Run:
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_baroclinic_adjustment_reference.jl <out_dir> [stop_days] [dump_days]

using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using Random
Random.seed!(8675309)

out_dir   = length(ARGS) >= 1 ? ARGS[1] : "."
stop_days = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 30.0
dump_days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 6.0
mkpath(out_dir)

Lx = Ly = 1000kilometers
Lz = 1kilometers
grid = RectilinearGrid(size = (48, 48, 8),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = BetaPlane(latitude = -45),
                                    buoyancy = BuoyancyTracer(), tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO())

ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)
N² = 1e-5; M² = 1e-7; Δy = 100kilometers
Δb = Δy * M²; ϵb = 1e-2 * Δb
bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
set!(model, b = bᵢ)

# Geostrophically-balanced spin from rest is set by the buoyancy field; the
# wizard adapts dt (baroclinic-instability decks NaN on a fixed dt as eddies grow).
simulation = Simulation(model, Δt = 10minutes, stop_time = stop_days * days)
wizard = TimeStepWizard(cfl = 0.2, max_change = 1.1, max_Δt = 20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))
progress(sim) = @printf("t=%s iter=%d dt=%s max|u|=%.4e\n", prettytime(time(sim)),
    iteration(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u))
simulation.callbacks[:p] = Callback(progress, IterationInterval(200))

# Dump surface (k=Nz) b, u, v for the comparison (the eddy field is surface-intensified).
Nz = 8
bs = Dict{Int,Array{Float64,2}}(); us = Dict{Int,Array{Float64,2}}(); vs = Dict{Int,Array{Float64,2}}()
times = Float64[]
function snap!()
    push!(times, time(simulation))
    i = length(times)
    bs[i] = Array{Float64}(interior(model.tracers.b)[:, :, Nz])
    us[i] = Array{Float64}(interior(model.velocities.u)[:, :, Nz])
    vs[i] = Array{Float64}(interior(model.velocities.v)[:, :, Nz])
end
simulation.callbacks[:snap] = Callback(_ -> snap!(), TimeInterval(dump_days * days))
snap!()  # t=0
run!(simulation)

@printf("baroclinic_adjustment: stop=%.0fd, %d snapshots, max|u_final|=%.4e\n",
        stop_days, length(times), maximum(abs, us[length(times)]))
nc = joinpath(out_dir, "baroclinic_adjustment.nc")
NCDataset(nc, "c") do ds
    nt = length(times)
    defDim(ds, "t", nt)
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

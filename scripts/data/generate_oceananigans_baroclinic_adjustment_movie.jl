# Movie variant of the baroclinic_adjustment oracle: parameterized resolution +
# daily surface (u,v,b) snapshots, for a side-by-side legoESM-vs-Oceananigans
# vorticity animation. FAITHFUL §5-precursor setup (WENOVectorInvariant order 9,
# BetaPlane -45, N²=1e-5 front), only the grid size + dump cadence differ.
#
# Run:
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_baroclinic_adjustment_movie.jl <out_dir> [stop_days] [dump_days] [N]
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf

ref_root  = length(ARGS) >= 1 ? ARGS[1] : "."
stop_days = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 40.0
dump_days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
N         = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 128
out_dir = joinpath(ref_root, "baroclinic_adjustment_movie"); mkpath(out_dir)

Lx = Ly = 1000kilometers; Lz = 1kilometers; Nz = 8
# Use the GPU when CUDA is functional in this env (else CPU).
arch = try
    using CUDA
    CUDA.functional() ? GPU() : CPU()
catch
    CPU()
end
@info "architecture = $arch"
grid = RectilinearGrid(arch, size = (N, N, Nz), halo = (6, 6, 4),
                       x = (0, Lx), y = (-Ly/2, Ly/2), z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))
model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = BetaPlane(latitude = -45),
                                    buoyancy = BuoyancyTracer(), tracers = :b,
                                    momentum_advection = WENOVectorInvariant(vorticity_order = 9),
                                    tracer_advection = WENO(order = 7))
ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)
N² = 1e-5; M² = 1e-7; Δy = 100kilometers
Δb = Δy * M²; ϵb = 1e-2 * Δb
bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + ϵb * randn()
set!(model, b = bᵢ)

simulation = Simulation(model, Δt = 10minutes, stop_time = stop_days * days)
wizard = TimeStepWizard(cfl = 0.2, max_change = 1.1, max_Δt = 20minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(20))
progress(sim) = @printf("t=%s iter=%d dt=%s max|u|=%.4e\n", prettytime(time(sim)),
    iteration(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u))
simulation.callbacks[:p] = Callback(progress, IterationInterval(400))

us = Dict{Int,Array{Float64,2}}(); vs = Dict{Int,Array{Float64,2}}(); bs = Dict{Int,Array{Float64,2}}()
times = Float64[]
function snap!()
    push!(times, time(simulation)); i = length(times)
    us[i] = Array{Float64}(interior(model.velocities.u)[:, :, Nz])
    vs[i] = Array{Float64}(interior(model.velocities.v)[:, :, Nz])
    bs[i] = Array{Float64}(interior(model.tracers.b)[:, :, Nz])
end
simulation.callbacks[:snap] = Callback(_ -> snap!(), TimeInterval(dump_days * days))
snap!(); run!(simulation)

@printf("baroclinic_adjustment_movie: N=%d stop=%.0fd %d frames max|u_final|=%.4e\n",
        N, stop_days, length(times), maximum(abs, us[length(times)]))
nc = joinpath(out_dir, "baroclinic_adjustment_movie.nc")
NCDataset(nc, "c") do ds
    nt = length(times)
    defDim(ds, "x", N); defDim(ds, "xv", N); defDim(ds, "y", N); defDim(ds, "yv", N+1); defDim(ds, "t", nt)
    defVar(ds, "times_s", times, ("t",))
    ds.attrib["Lx_m"] = Float64(Lx)
    uv = defVar(ds, "u", Float64, ("x","y","t")); vv = defVar(ds, "v", Float64, ("x","yv","t"))
    bv = defVar(ds, "b", Float64, ("x","y","t"))
    for i in 1:nt; uv[:,:,i] = us[i]; vv[:,:,i] = vs[i]; bv[:,:,i] = bs[i]; end
end
@info "DONE: wrote $nc"

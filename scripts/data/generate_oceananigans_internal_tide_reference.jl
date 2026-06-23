# Internal-tide reference (Oceananigans examples/internal_tide.jl), for the legoESM
# fidelity comparison. A barotropic M2 tide oscillates a stratified (constant N²)
# hydrostatic flow over a Gaussian ridge → radiates internal tides (baroclinic waves).
# The FIRST fidelity case with TOPOGRAPHY + (tidal) FORCING + internal-wave generation.
#
# 2D x-z (Periodic-x, Flat-y, Bounded-z), f-plane(-45), WENO momentum+tracer, NO closure.
# Tidal body force on u: A₂·sin(ω₂ t). Dumps w(x,z) and b'(x,z)=b−N²z at each output +
# the bottom depth H_bathy(x), for the legoESM comparator (pattern-match the radiated w).
#
# Run:
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_internal_tide_reference.jl <out_dir> [stop_days] [dump_hours]

using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf

ref_root   = length(ARGS) >= 1 ? ARGS[1] : "."
stop_days  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4.0
dump_hours = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 3.0
Nz_arg     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 64
out_dir = joinpath(ref_root, "internal_tide"); mkpath(out_dir)

Nx, Nz = 256, Nz_arg
H, L = 2kilometers, 1000kilometers
underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = (-H, 0),
                                  topology = (Periodic, Flat, Bounded))
h₀ = 250meters; width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
bottom(x) = -H + hill(x)
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

coriolis = FPlane(latitude = -45)
T₂ = 12.421hours; ω₂ = 2π / T₂
ϵ = 0.1; U₂ = ϵ * ω₂ * width
A₂ = U₂ * (ω₂^2 - coriolis.f^2) / ω₂
@inline tidal_forcing(x, z, t, p) = p.A₂ * sin(p.ω₂ * t)
u_forcing = Forcing(tidal_forcing, parameters = (; A₂, ω₂))

model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                    buoyancy = BuoyancyTracer(), tracers = :b,
                                    momentum_advection = WENO(), tracer_advection = WENO(),
                                    forcing = (; u = u_forcing))
Nᵢ² = 1e-4
bᵢ(x, z) = Nᵢ² * z
set!(model, u = U₂, b = bᵢ)

simulation = Simulation(model; Δt = 5minutes, stop_time = stop_days * days)
progress(sim) = @printf("t=%s max|w|=%.3e max|u|=%.3e\n", prettytime(time(sim)),
    maximum(abs, model.velocities.w), maximum(abs, model.velocities.u))
simulation.callbacks[:p] = Callback(progress, IterationInterval(500))

xc = xnodes(grid, Center()); zc = znodes(grid, Center())
Hb = Float64[-(-H + hill(x)) for x in xc]            # positive water depth at each x
ws = Dict{Int,Array{Float64,2}}(); bp = Dict{Int,Array{Float64,2}}(); times = Float64[]
function snap!()
    push!(times, time(simulation)); i = length(times)
    w = interior(model.velocities.w)[:, 1, :]         # (Nx, Nz+1) at z-faces
    ws[i] = Array{Float64}(0.5 .* (w[:, 1:end-1] .+ w[:, 2:end]))  # → centres (Nx,Nz)
    b = interior(model.tracers.b)[:, 1, :]
    bp[i] = Array{Float64}(b .- reshape(Nᵢ² .* zc, 1, :))         # b' = b − N²z
end
simulation.callbacks[:snap] = Callback(_ -> snap!(), TimeInterval(dump_hours * hours))
snap!(); run!(simulation)

@printf("internal_tide: stop=%.1fd, %d snaps, T₂=%.3fh, U₂=%.4e, max|w_final|=%.3e\n",
        stop_days, length(times), T₂/3600, U₂, maximum(abs, ws[length(times)]))
nc = joinpath(out_dir, "internal_tide.nc")
NCDataset(nc, "c") do ds
    nt = length(times); defDim(ds, "t", nt); defDim(ds, "x", Nx); defDim(ds, "z", Nz)
    ds.attrib["T2_s"] = T₂; ds.attrib["U2"] = U₂; ds.attrib["N2"] = Nᵢ²; ds.attrib["H"] = H
    defVar(ds, "times_s", times, ("t",))
    defVar(ds, "x", Array{Float64}(xc), ("x",)); defVar(ds, "z", Array{Float64}(zc), ("z",))
    defVar(ds, "H_bathy", Hb, ("x",))
    wv = defVar(ds, "w", Float64, ("t","x","z")); bv = defVar(ds, "bprime", Float64, ("t","x","z"))
    for i in 1:nt; wv[i,:,:] = ws[i]; bv[i,:,:] = bp[i]; end
end
@info "DONE: wrote $nc"

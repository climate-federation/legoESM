# Single-step w / continuity reference for the internal_tide partial-cell staircase
# (issue #576). Builds the internal_tide geometry (RectilinearGrid x-z, Flat y,
# PartialCellBottom bump, FPlane, FLUX-FORM WENO() momentum = the faithful scheme),
# sets a PRESCRIBED state (u=U2, v=V0*exp(-x^2/2W^2), b=N2 z), runs update_state!
# (which diagnoses w from continuity), and dumps w(x,z) + the horizontal flux
# divergence — the per-node oracle to diff against legoESM's diagnosed w over the
# partial-cell steps. NO time stepping.
#
# Run:
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_internal_tide_w_reference.jl <out_dir> [Nz]

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!, time_step!
using NCDatasets
using Printf

out_dir = length(ARGS) >= 1 ? ARGS[1] : "."
Nz_arg  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
mkpath(out_dir)

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

model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                    momentum_advection = WENO(), tracer_advection = WENO(),
                                    buoyancy = BuoyancyTracer(), tracers = :b)
Nᵢ² = 1e-4; V0 = 0.1
uᵢ(x, z) = U₂
vᵢ(x, z) = V0 * exp(-x^2 / 2width^2)
bᵢ(x, z) = Nᵢ² * z
set!(model, u = uᵢ, v = vᵢ, b = bᵢ)
update_state!(model)

xc = xnodes(grid, Center()); zc = znodes(grid, Center())
zf = znodes(grid, Face())
Hb = Float64[-(-H + hill(x)) for x in xc]

w = interior(model.velocities.w)[:, 1, :]    # (Nx, Nz+1) at prescribed state
u = interior(model.velocities.u)[:, 1, :]
v = interior(model.velocities.v)[:, 1, :]
b0 = Array(interior(model.tracers.b)[:, 1, :])
# The FIRST time_step! is an Euler step, so Δb/dt = the INSTANTANEOUS RHS tracer
# tendency (= -advection of b, NO closure). This is the clean per-node oracle to
# compare against legoESM's directly-computed (un-AB2-weighted) advection flux
# divergence (`scripts/tmp/_it_inst_tend.py`).
time_step!(model, 150.0)
b = interior(model.tracers.b)[:, 1, :]
Gb = (b .- b0) ./ 150.0

@printf("it-w ref: Nx=%d Nz=%d U2=%.4e V0=%.3f  max|w|=%.4e max|Gb|=%.4e\n",
        Nx, Nz, U₂, V0, maximum(abs, w), maximum(abs, Gb))

nc = joinpath(out_dir, "internal_tide_w.nc")
NCDataset(nc, "c") do ds
    defDim(ds, "x", Nx); defDim(ds, "z", Nz); defDim(ds, "zf", Nz+1)
    ds.attrib["U2"] = U₂; ds.attrib["V0"] = V0; ds.attrib["N2"] = Nᵢ²; ds.attrib["H"] = H
    defVar(ds, "x", Array{Float64}(xc), ("x",))
    defVar(ds, "z", Array{Float64}(zc), ("z",)); defVar(ds, "zf", Array{Float64}(zf), ("zf",))
    defVar(ds, "H_bathy", Hb, ("x",))
    defVar(ds, "w", w, ("x","zf"))
    defVar(ds, "u", u, ("x","z")); defVar(ds, "v", v, ("x","z"))
    defVar(ds, "b", b, ("x","z")); defVar(ds, "Gb", Gb, ("x","z"))
end
@info "DONE: wrote $nc"

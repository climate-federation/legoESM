# Reference generator for the Oceananigans `bickley_jet` fidelity oracle.
#
# A clean lat-lon-channel Bickley jet (barotropic shear instability) that maps
# directly to legoESM's create_regional_latlon_grid(periodic_x=True): a single
# free-surface layer, periodic in longitude, walls in latitude, on a unit sphere
# (radius=1, rotation_rate=1) -- following validation/bickley_jet/spherical_bickley_jet.jl
# but writing a NetCDF time series of (u, v, zeta) for the legoESM comparison.
#
# This is a DETERMINISTIC twin target: both codes start from the IDENTICAL
# Bickley IC (U=sech^2 jet + a fixed vortical perturbation) and evolve; the
# eddy field (relative vorticity) is compared at matched times -- a direct test
# of the vector-invariant / WENO momentum advection in the eddy regime (PR #559).
#
# Run (working env):
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_bickley_jet_reference.jl <out_dir> [stop_time] [dump_dt] [Nh]

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using NCDatasets
using Printf

out_dir   = length(ARGS) >= 1 ? ARGS[1] : "."
stop_time = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 100.0
dump_dt   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 10.0
Nh        = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 64
mkpath(out_dir)

Nφ = Int(Nh / 2)
grid = LatitudeLongitudeGrid(size = (Nh, Nφ, 1), radius = 1,
                             longitude = (-180, 180), latitude = (-80, 80),
                             z = (0, 1), halo = (7, 7, 7))

# Bickley jet + vortical perturbation (spherical_bickley_jet.jl).
Ψ(y) = - tanh(y)
U(y) = sech(y)^2
C(y, L) = sin(2π * y / L)
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2)
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x)

ϵ = 0.1; ℓ = 0.5; k = 0.5
dr(x) = deg2rad(x)
uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k)
cᵢ(x, y, z) = C(dr(y)*8, grid.Ly)

momentum_advection = WENOVectorInvariant(vorticity_order = 9)
free_surface = ImplicitFreeSurface(gravitational_acceleration = 1)
model = HydrostaticFreeSurfaceModel(grid; momentum_advection,
                                    tracer_advection = WENO(),
                                    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1),
                                    free_surface, tracers = :c, buoyancy = nothing)
set!(model, u = uᵢ, v = vᵢ, c = cᵢ)

gw = sqrt(model.free_surface.gravitational_acceleration)
Δt = 0.1 * Array(model.grid.Δxᶜᶠᵃ.parent)[5] / gw
simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

progress(sim) = @printf("t=%.1f iter=%d max|u|=%.3f max|eta|=%.3e\n",
                        time(sim), iteration(sim),
                        maximum(abs, model.velocities.u),
                        maximum(abs, model.free_surface.displacement))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

u, v = model.velocities.u, model.velocities.v
ζ = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v))
outputs = (u = u, v = v, zeta = ζ)
nc = joinpath(out_dir, "bickley_jet.nc")
simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs; filename = nc,
                 schedule = TimeInterval(dump_dt), overwrite_existing = true)

@info "Running Oceananigans bickley_jet -> $nc (stop=$stop_time, dump=$dump_dt, Nh=$Nh, dt=$Δt)"
run!(simulation)
@info "DONE: wrote $nc"

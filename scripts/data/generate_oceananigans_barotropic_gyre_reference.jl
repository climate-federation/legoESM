# Reference generator for the Oceananigans `barotropic_gyre` fidelity oracle.
#
# A faithful, NetCDF-emitting copy of
#   /tmp/ocn_depot/.../validation/barotropic_gyre/barotropic_gyre.jl
# (LatitudeLongitudeGrid 60x60x1, ImplicitFreeSurface g=0.1,
#  HydrostaticSphericalCoriolis EnstrophyConserving, cos wind stress,
#  linear bottom drag, constant horizontal viscosity, VectorInvariant momentum)
# but writing a NetCDF time series of (u, v, eta) that
# `legoesm.ocean.fidelity.oceananigans_runner.load_oceananigans_reference`
# ("barotropic_gyre") loads. Run in the isolated depot:
#
#   JULIA_DEPOT_PATH=/tmp/ocn_depot \
#   julia --project=/tmp/ocn_silvestri --startup-file=no \
#     scripts/data/generate_oceananigans_barotropic_gyre_reference.jl \
#     <output_dir> [stop_days] [dump_days]
#
# Writes <output_dir>/barotropic_gyre.nc. <output_dir> is normally
# `$LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF/barotropic_gyre` (or the
# ocean-fidelity cache's oceananigans/barotropic_gyre subdir).
#
# Mimicry-only glue (writer format, output dir) lives here in the harness; the
# numerics are the canonical Oceananigans validation setup, unchanged.

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Units
using NCDatasets
using Printf

out_dir   = length(ARGS) >= 1 ? ARGS[1] : "."
stop_days = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 365.0
dump_days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 30.0
mkpath(out_dir)

Nx = 60
Ny = 60

grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                             longitude = (-30, 30),
                             latitude = (15, 75),
                             z = (-4000, 0))

free_surface = ImplicitFreeSurface(gravitational_acceleration = 0.1)
coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConserving())

surface_wind_stress_parameters = (τ₀ = 1e-2, Lφ = grid.Ly, φ₀ = 15)
@inline surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)
surface_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress,
                                               parameters = surface_wind_stress_parameters)

μ = 1 / 60days
@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds -μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds -μ * fields.v[i, j, 1]
u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = μ)
v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = μ)

u_bcs = FieldBoundaryConditions(top = surface_wind_stress_bc, bottom = u_bottom_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_bottom_drag_bc)

νh₀ = 5e3 * (60 / grid.Nx)^2
horizontal_diffusivity = HorizontalScalarDiffusivity(ν = νh₀)

model = HydrostaticFreeSurfaceModel(grid; free_surface, coriolis,
                                    momentum_advection = VectorInvariant(),
                                    boundary_conditions = (u = u_bcs, v = v_bcs),
                                    closure = horizontal_diffusivity)

simulation = Simulation(model, Δt = 3600, stop_time = stop_days * days)

function progress(sim)
    @info @sprintf("t=%s  iter=%d  max|u|=%.3e",
                   prettytime(sim.model.clock.time), sim.model.clock.iteration,
                   maximum(abs, sim.model.velocities.u))
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(24 * 50))

# NOTE: the 2-D free-surface displacement (model.free_surface.displacement) is a
# singleton z-Face field that conflicts with the grid's z_aaf coordinate in the
# v0.110.x NetCDFWriter ("z_aaf already exists ... values differ"). The velocities
# (the barotropic flow) are written here; eta is a follow-up (separate writer).
outputs = (u = model.velocities.u,
           v = model.velocities.v)

nc_path = joinpath(out_dir, "barotropic_gyre.nc")
simulation.output_writers[:fields] =
    NetCDFWriter(model, outputs;
                 filename = nc_path,
                 schedule = TimeInterval(dump_days * days),
                 overwrite_existing = true)

@info "Running Oceananigans barotropic_gyre reference -> $nc_path " *
      "(stop=$(stop_days)d, dump every $(dump_days)d)"
run!(simulation)
@info "DONE: wrote $nc_path"

# Single-step TENDENCY reference for the Oceananigans-recipe wiring match.
#
# Dumps Oceananigans' HydrostaticFreeSurfaceModel momentum tendency Gu, Gv
# (= the summed RHS terms 1-8 of kernel_functions.jl, BEFORE the free-surface
# predictor-corrector) evaluated at a known IC, with NO time stepping. legoESM
# computes its own du_dt/dv_dt from the SAME IC and the two are compared
# term-by-term (Phase 1 of docs/ocean_fidelity/oceananigans_recipe_wiring_plan.md).
#
# Reuses the validated bickley grid/IC bridge (unit sphere, rotation=1, g=1,
# WENOVectorInvariant(9), ImplicitFreeSurface) so the staggering bridge is known.
#
# Run (working env):
#   JULIA_DEPOT_PATH=/tmp/ocn_j11_depot julia +1.10.11 --project=/tmp/ocn_j11_gen \
#     scripts/data/generate_oceananigans_tendency_reference.jl <out_dir> [Nh]

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Operators: ζ₃ᶠᶠᶜ
using Oceananigans.TimeSteppers: update_state!
using NCDatasets
using Printf

out_dir = length(ARGS) >= 1 ? ARGS[1] : "."
Nh      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
mkpath(out_dir)

Nφ = Int(Nh / 2)
grid = LatitudeLongitudeGrid(size = (Nh, Nφ, 1), radius = 1,
                             longitude = (-180, 180), latitude = (-80, 80),
                             z = (0, 1), halo = (7, 7, 7))

# Bickley jet + vortical perturbation (identical to the bickley reference deck).
U(y) = sech(y)^2
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2)
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x)
ϵ = 0.1; ℓ = 0.5; k = 0.5
dr(x) = deg2rad(x)
uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k)
cᵢ(x, y, z) = sin(2π * dr(y) * 8 / grid.Ly)

momentum_advection = WENOVectorInvariant(vorticity_order = 9)
free_surface = ImplicitFreeSurface(gravitational_acceleration = 1)
model = HydrostaticFreeSurfaceModel(grid; momentum_advection,
                                    tracer_advection = WENO(),
                                    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1),
                                    free_surface, tracers = :c, buoyancy = nothing)
set!(model, u = uᵢ, v = vᵢ, c = cᵢ)

# update_state! diagnoses w, pHY', closures AND computes the momentum tendency
# Gⁿ for the next step — exactly the Gu/Gv we want (terms 1-8, pre-free-surface).
update_state!(model)

u = interior(model.velocities.u)[:, :, 1]
v = interior(model.velocities.v)[:, :, 1]
Gu = interior(model.timestepper.Gⁿ.u)[:, :, 1]
Gv = interior(model.timestepper.Gⁿ.v)[:, :, 1]
ζf = Field(KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid,
                                                       model.velocities.u, model.velocities.v))
compute!(ζf)
ζ = interior(ζf)[:, :, 1]

# ISOLATED vorticity-flux node (the §5 residual target): Oceananigans'
# `horizontal_advection_U/V` = -v̂·ζᴿ (the vorticity-flux contribution to U_dot_∇u),
# evaluated DIRECTLY via KernelFunctionOperation on the materialized scheme. This is
# the single-step, non-chaotic per-node oracle for the C-grid vorticity-flux
# collocation (bias-velocity interp + vertex curl) — compared pointwise to legoESM's
# `diag_vortcor_u/v`. The advection contributes `-horizontal_advection_*` to G* (the
# tendency is minus the advection), so the legoESM comparison is to -hadv_*.
sch = model.advection.momentum
hadvU = Field(KernelFunctionOperation{Face, Center, Center}(
    Oceananigans.Advection.horizontal_advection_U, grid, sch,
    model.velocities.u, model.velocities.v))
hadvV = Field(KernelFunctionOperation{Center, Face, Center}(
    Oceananigans.Advection.horizontal_advection_V, grid, sch,
    model.velocities.u, model.velocities.v))
compute!(hadvU); compute!(hadvV)
hadv_u = interior(hadvU)[:, :, 1]   # at (Face,Center) = u-point, colocates with Gu
hadv_v = interior(hadvV)[:, :, 1]   # at (Center,Face) = v-point, colocates with Gv

@printf("tendency ref: Nh=%d  max|u|=%.4f max|Gu|=%.4e max|Gv|=%.4e max|hadvU|=%.4e max|hadvV|=%.4e\n",
        Nh, maximum(abs, u), maximum(abs, Gu), maximum(abs, Gv),
        maximum(abs, hadv_u), maximum(abs, hadv_v))

nc = joinpath(out_dir, "tendency.nc")
# u is (Face,Center); v is (Center,Face); Gu colocates with u, Gv with v;
# zeta is (Face,Face). Write each on its own dims.
NCDataset(nc, "c") do ds
    nxu, nyu = size(u);  defDim(ds, "xu", nxu); defDim(ds, "yu", nyu)
    nxv, nyv = size(v);  defDim(ds, "xv", nxv); defDim(ds, "yv", nyv)
    nxz, nyz = size(ζ);  defDim(ds, "xz", nxz); defDim(ds, "yz", nyz)
    defVar(ds, "u",  u,  ("xu","yu"))
    defVar(ds, "v",  v,  ("xv","yv"))
    defVar(ds, "Gu", Gu, ("xu","yu"))
    defVar(ds, "Gv", Gv, ("xv","yv"))
    defVar(ds, "zeta", ζ, ("xz","yz"))
    # Isolated vorticity-flux node (colocated with u/v respectively).
    defVar(ds, "hadv_u", hadv_u, ("xu","yu"))
    defVar(ds, "hadv_v", hadv_v, ("xv","yv"))
end
@info "DONE: wrote $nc"

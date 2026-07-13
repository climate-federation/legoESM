"""Bare-dycore stability probe: balanced warm-bubble IC, no physics.

Runs PlaneCompressibleEulerModel for N outer steps with:
  - Hydrostatically-balanced warm bubble (rho' = -rho_ref * theta' / theta_ref)
  - Sponge top, hyperdiff, Smag, mass fixer
  - --semi-implicit-acoustic, --acoustic-off-centering 0.1
  - NO surface flux / NO microphysics / NO radiation / NO tracers

Reports max|w|, max|theta'|, rho_min every 10 steps. If max|w| stays bounded
(<10 m/s) through the run, the dycore is stable — the long-run instability
is from the physics path. If it grows past ~step 70 like the full driver,
the dycore itself is the destabilizer.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_theta_ref_fn,
)

jax.config.update("jax_enable_x64", True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=48)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2000.0)
    p.add_argument("--H", type=float, default=33000.0)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-acoustic-substeps", type=int, default=48)
    p.add_argument("--sponge-width", type=float, default=10000.0)
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--hyperdiff", type=float, default=5.0e6)
    p.add_argument("--smag-cs", type=float, default=0.2)
    p.add_argument("--acoustic-off-centering", type=float, default=0.1)
    p.add_argument("--semi-implicit-acoustic", action="store_true",
                   default=True)
    p.add_argument("--no-semi-implicit-acoustic",
                   dest="semi_implicit_acoustic", action="store_false")
    p.add_argument("--bubble-theta-pert", type=float, default=0.5)
    p.add_argument("--no-bubble", action="store_true",
                   help="Run from pure rest state (no IC perturbation).")
    p.add_argument("--advection",
                   choices=["upwind1", "van_leer", "weno5"],
                   default="upwind1",
                   help="Horizontal advection scheme for theta/u/v/w. "
                        "iter-179/183 added van_leer (2nd-order TVD, "
                        "stencil 4) as the dycore's third choice; the "
                        "production driver default in run_rce_mpi_long.py "
                        "is now van_leer (iter-183 wall-time refresh). "
                        "This diag retains upwind1 as the cheapest "
                        "default for the bare-dycore stability sweep.")
    p.add_argument("--vertical-theta-diffusion", type=float, default=0.0,
                   help="Explicit vertical Laplacian diffusivity on "
                        "theta_prime [m^2/s]. Tames the buoyancy-driven "
                        "gravity-wave amplification at coarse dz.")
    p.add_argument("--implicit-buoyancy", action="store_true", default=False,
                   help="Enable Klemp-Wilhelmson 1978 implicit-buoyancy "
                        "treatment in the SI acoustic substep. Adds three "
                        "nearest-neighbour bands to the tridiagonal that "
                        "encode g/theta_0 * dtheta_ref/dz coupling between "
                        "w_new and theta_p_new. Closes the gravity-wave "
                        "amplification loop at coarse vertical resolution.")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"# bare-dycore stability probe: nx={args.nx} ny={args.ny} "
          f"nlev={args.nlev} dx={args.dx} dt={args.dt} steps={args.steps}")
    print(f"# SIA={args.semi_implicit_acoustic} beta={args.acoustic_off_centering} "
          f"n_acoustic={args.n_acoustic_substeps} hyperdiff={args.hyperdiff:.1e} "
          f"smag_cs={args.smag_cs} sponge_width={args.sponge_width}")
    print(f"# bubble theta_pert={args.bubble_theta_pert} K, no_bubble={args.no_bubble}")
    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    # Equilibrium CRM debug: T_v0 = SST (param renamed T_sfc -> T_v0).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=300.0, q_sfc=0.0224, z_t=15000.0, Gamma=6.7e-3,
    )
    hc = create_height_coordinate(
        n_levels=args.nlev, H=args.H, theta_ref_fn=theta_fn,
    )
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=args.semi_implicit_acoustic,
        acoustic_off_centering=args.acoustic_off_centering,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=args.n_acoustic_substeps,
        horizontal_advection_scheme=args.advection,
        vertical_theta_diffusion=args.vertical_theta_diffusion,
        implicit_buoyancy=args.implicit_buoyancy,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg)
    state = make_rest_state(grid, hc, dtype=jnp.float64)

    if not args.no_bubble:
        ny, nx = grid.ny, grid.nx
        jj = jnp.arange(ny); ii = jnp.arange(nx)
        yy, xx = jnp.meshgrid(jj, ii, indexing="ij")
        yc, xc = (ny - 1) / 2.0, (nx - 1) / 2.0
        r_cells = jnp.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
        r0 = 10.0
        horiz = jnp.where(
            r_cells < r0,
            0.5 * (1.0 + jnp.cos(jnp.pi * r_cells / r0)), 0.0)
        z = hc.z_full
        z_top_bubble = 1000.0
        vert = jnp.where(z < z_top_bubble,
                         0.5 * (1.0 + jnp.cos(jnp.pi * z / z_top_bubble)), 0.0)
        bubble_theta = args.bubble_theta_pert * horiz[:, :, None] * vert[None, None, :]
        bubble_rho = -hc.rho_ref * bubble_theta / hc.theta_ref
        state = state._replace(
            theta_prime=state.theta_prime.replace(data=bubble_theta),
            rho_prime=state.rho_prime.replace(data=bubble_rho),
        )

    print("# step,max|w|,max|theta'|,rho_min")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        state = model.step(state, dt=args.dt, physics_fn=None)
        if step % 10 == 0 or step == 1:
            max_w = float(jnp.max(jnp.abs(state.w.data)))
            max_tp = float(jnp.max(jnp.abs(state.theta_prime.data)))
            rho_min = float(jnp.min(hc.rho_ref + state.rho_prime.data))
            print(f"{step},{max_w:.4e},{max_tp:.4e},{rho_min:.4e}")
            if not jnp.isfinite(max_w):
                print("# BAIL: NaN")
                break
    wall = time.time() - t0
    print(f"# wall: {wall:.1f} s ({wall/args.steps:.3f} s/step)")


if __name__ == "__main__":
    main()

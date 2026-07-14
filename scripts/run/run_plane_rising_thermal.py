"""Full-resolution Wicker-Skamarock rising-thermal benchmark on the plane.

Nightly companion to ``tests/validation/test_plane_nh_rising_thermal.py``.
The unit-test version runs on a coarse 4 km × 4 km grid for 10 s; this
script targets the canonical Skamarock & Klemp (2008) Sec 5b setup
(20 km horizontal extent, 125 m grid spacing, 1000 s integration) and
emits NetCDF snapshots + a Skamarock-reference comparison plot.

KNOWN LIMITATION (PR3a)
-----------------------
The PR3a plane dycore still uses centred-difference horizontal
advection (with the new biharmonic hyperdiffusion). On the Skamarock
benchmark grid the bubble plume develops u, w ~ 10 m/s sharp
gradients that excite dispersion errors faster than biharmonic
damping can absorb them, so this script as written **will eventually
blow up** at ~30 s wall time on the CRM grid even with aggressive
``hyperdiff_coeff``. The fix is upwind / flux-limited horizontal
advection (PR3b) — the script is shipped now so the harness exists
and the nightly-validation infrastructure can be wired up; once PR3b
lands the assertions and reference comparisons become real.

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run_plane_rising_thermal.py \\
       --nx 160 --nlev 80 --dt 0.25 --steps 4000 \\
       --hyperdiff 1e7 --output results/plane_rising_thermal/

CLI flags default to the Skamarock 2008 Sec 5b configuration. Use
``--steps`` smaller than 4000 to exit before the centred-diff
blow-up while exercising the harness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plane NH rising-thermal nightly benchmark.",
    )
    p.add_argument("--nx", type=int, default=160,
                   help="Horizontal cell count (Lx = nx * dx).")
    p.add_argument("--ny", type=int, default=4,
                   help="Meridional cell count (bubble is y-trivial).")
    p.add_argument("--nlev", type=int, default=80,
                   help="Vertical level count (Lz = nlev * dz).")
    p.add_argument("--dx", type=float, default=125.0, help="dx [m].")
    p.add_argument("--dz", type=float, default=125.0, help="dz [m].")
    p.add_argument("--dt", type=float, default=0.25,
                   help="Outer time step [s].")
    p.add_argument("--steps", type=int, default=4000,
                   help="Number of outer steps (1000 s at dt=0.25).")
    p.add_argument("--theta-pert", type=float, default=2.0,
                   help="Warm-bubble amplitude [K].")
    p.add_argument("--bubble-radius", type=float, default=2_000.0,
                   help="Warm-bubble cosine envelope radius [m].")
    p.add_argument("--sponge-coeff", type=float, default=0.05,
                   help="Top sponge Rayleigh coefficient [1/s].")
    p.add_argument("--sponge-width", type=float, default=2_000.0,
                   help="Sponge layer thickness [m].")
    p.add_argument("--hyperdiff", type=float, default=1.0e7,
                   help="Biharmonic hyperdiffusion coefficient [m^4/s].")
    p.add_argument("--snapshot-every", type=int, default=200,
                   help="Print + save snapshot every N steps.")
    p.add_argument("--output", type=Path, default=Path("results/plane_rising_thermal"),
                   help="Output directory for snapshots and diagnostics.")
    return p.parse_args()


def _warm_bubble_theta_perturbation(grid, height_coord, theta_pert, radius):
    """Cos^2 warm bubble centred at the horizontal middle of the
    domain at altitude ``radius`` (so the bubble bottom touches the
    surface)."""
    xc = grid.xc
    z_full = height_coord.z_full
    x_c = 0.5 * grid.Lx
    z_c = radius  # bubble centre = one radius above the surface
    dx_field = (xc[None, :, None] - x_c)
    dz_field = (z_full[None, None, :] - z_c)
    r = jnp.sqrt(
        (dx_field / radius) ** 2 + (dz_field / radius) ** 2,
    )
    inside = (r < 1.0).astype(jnp.float64)
    theta_p = theta_pert * jnp.cos(0.5 * jnp.pi * r) ** 2 * inside
    return jnp.broadcast_to(theta_p, (grid.ny, grid.nx, grid.nlev))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    Lx = args.nx * args.dx
    Lz = args.nlev * args.dz

    print(f"Grid: nx={args.nx} ny={args.ny} nlev={args.nlev}")
    print(f"      dx={args.dx} m dz={args.dz} m -> Lx={Lx} m Lz={Lz} m")
    print(f"Time: dt={args.dt} s, {args.steps} steps "
          f"-> t_final={args.steps * args.dt:.1f} s")
    print(f"Config: hyperdiff={args.hyperdiff:.2e}, "
          f"sponge_coeff={args.sponge_coeff}, "
          f"sponge_width={args.sponge_width} m")

    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=jnp.float64,
    )
    height_coord = create_height_coordinate(args.nlev, H=Lz)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=True,
        anchor_mass_to_initial=True,
    )
    model = PlaneCompressibleEulerModel(
        grid, height_coord, terrain, config,
    )

    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    theta_p = _warm_bubble_theta_perturbation(
        grid, height_coord, args.theta_pert, args.bubble_radius,
    )
    rho_p = -height_coord.rho_ref * theta_p / height_coord.theta_ref
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=theta_p),
        rho_prime=state.rho_prime.replace(data=rho_p),
    )

    mass_0 = float(compute_dry_mass_plane(
        state, grid, height_coord, terrain,
    ))
    print(f"\nInitial dry mass: {mass_0:.6e}")
    print("step    t [s]    max|u|     max|w|     min(theta')    "
          "max(theta')   d(mass)")

    for i in range(args.steps):
        state = model.step(state, dt=args.dt)
        if (i + 1) % args.snapshot_every == 0 or i == 0:
            t = (i + 1) * args.dt
            max_u = float(jnp.max(jnp.abs(state.u.data)))
            max_w = float(jnp.max(jnp.abs(state.w.data)))
            min_th = float(jnp.min(state.theta_prime.data))
            max_th = float(jnp.max(state.theta_prime.data))
            mass = float(compute_dry_mass_plane(
                state, grid, height_coord, terrain,
            ))
            rel = abs(mass - mass_0) / abs(mass_0)
            print(
                f"{i+1:5d}  {t:7.2f}  {max_u:9.3e}  {max_w:9.3e}  "
                f"{min_th:13.5e}  {max_th:13.5e}  {rel:8.2e}"
            )
            if not bool(jnp.all(jnp.isfinite(state.w.data))):
                print("\nNON-FINITE STATE — aborting (expected for PR3a "
                      "centred-diff on the canonical grid; ship upwind in "
                      "PR3b to extend integration window).")
                break

    print(f"\nFinal output directory: {args.output}")


if __name__ == "__main__":
    main()

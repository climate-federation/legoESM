"""CFL convergence study for plane NH CRM dycore.

Scans dx, finds max stable dt at each dx (bisection on dt until
NaN or step blowup), for both:
* Explicit forward-backward acoustic substeps
* Semi-implicit acoustic substeps (vertical Thomas)

Reports CFL number ``dt_max · c_s / dx`` — for SSP-RK3 explicit
this should be ~1.7 (theoretical), for semi-implicit it should be
much larger (vertical CFL removed).

Mirrors the CFL-convergence study done for global atmosphere +
ocean dycores: pure stability characterization, no physics.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_theta_ref_fn,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

C_S_DRY = float(np.sqrt(1.4 * constants.R_d * 300.0))  # ~347 m/s
N_STEPS_BENCH = 20
KICK_AMP = 1.0e-3


def fresh_state(nx, dx, H=33_000.0, nlev=30):
    grid = create_plane_grid(
        nx=nx, ny=nx, nlev=nlev, dx=dx, dy=dx, dtype=jnp.float64,
    )
    # Near-EQUILIBRIUM CRM benchmark: theta reference uses T_v0=300 K to match
    # the 300 K surface setup (NOT the strict-RCEMIP fixed 295 K). The surface-
    # temp param was renamed T_sfc -> T_v0 (old kwarg TypeErrors).
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=300.0, q_sfc=0.018, z_t=15_000.0, Gamma=6.7e-3,
    )
    hc = create_height_coordinate(nlev, H=H, theta_ref_fn=theta_fn)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = jax.random.PRNGKey(0)
    kick = KICK_AMP * jax.random.normal(rng, state.theta_prime.data.shape)
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=kick),
    )
    return grid, hc, tm, state


def is_stable(grid, hc, tm, state, cfg, dt, n_steps=N_STEPS_BENCH):
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    for _ in range(n_steps):
        state = model.step(state, dt=dt)
        if not bool(jnp.all(jnp.isfinite(state.w.data))):
            return False
        rho_total = hc.rho_ref + state.rho_prime.data
        if float(jnp.min(rho_total)) < 0.0:
            return False
        if float(jnp.max(jnp.abs(state.w.data))) > 500.0:
            return False
    return True


def bisect_max_dt(grid, hc, tm, cfg, dt_lo=0.05, dt_hi=120.0, tol=0.1):
    """Bisect dt in [dt_lo, dt_hi] for largest stable value."""
    # First confirm dt_lo stable, dt_hi unstable.
    _, _, _, state_lo = fresh_state(
        grid.nx, grid.dx, hc.H, hc.theta_ref.shape[0],
    )
    if not is_stable(grid, hc, tm, state_lo, cfg, dt_lo):
        return None  # even dt_lo unstable
    _, _, _, state_hi = fresh_state(
        grid.nx, grid.dx, hc.H, hc.theta_ref.shape[0],
    )
    if is_stable(grid, hc, tm, state_hi, cfg, dt_hi):
        return dt_hi  # dt_hi already stable → return upper
    while (dt_hi - dt_lo) > tol:
        dt_mid = 0.5 * (dt_lo + dt_hi)
        _, _, _, state_mid = fresh_state(
            grid.nx, grid.dx, hc.H, hc.theta_ref.shape[0],
        )
        if is_stable(grid, hc, tm, state_mid, cfg, dt_mid):
            dt_lo = dt_mid
        else:
            dt_hi = dt_mid
    return dt_lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dx-list", type=float, nargs="+",
        default=[500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0],
    )
    ap.add_argument("--nx", type=int, default=24,
                    help="Horizontal grid size (kept constant; "
                         "scaling is in dx).")
    ap.add_argument("--nlev", type=int, default=30)
    ap.add_argument("--H", type=float, default=33_000.0)
    ap.add_argument("--n-acoustic-substeps", type=int, default=24)
    ap.add_argument(
        "--output", type=str, default="results/cfl_convergence.txt",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# CFL convergence study (plane NH CRM)")
    print(f"# nx={args.nx} nlev={args.nlev} H={args.H} c_s={C_S_DRY:.1f}")
    print(f"# n_acoustic_substeps={args.n_acoustic_substeps}")
    print(f"# Each row: dx, scheme, dt_max [s], CFL=dt_max·c_s/dx, "
          f"speedup vs explicit")
    header = (
        "dx[m],scheme,dt_max[s],CFL_acoustic,acoustic_substep_CFL,"
        "comments"
    )
    print(header)
    rows = [header]

    explicit_dt = {}
    for dx in args.dx_list:
        grid, hc, tm, _ = fresh_state(
            args.nx, dx, args.H, args.nlev,
        )
        # Explicit
        cfg_e = CompressibleEulerConfig(
            n_acoustic_substeps=args.n_acoustic_substeps,
            semi_implicit_acoustic=False,
            sponge_coeff=0.05, sponge_width=5000.,
            hyperdiff_coeff=1.0e6,
            hyperdiff_rho_coeff=1.0e6,
            hyperdiff_w_coeff=1.0e6,
            smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
            use_coriolis=False,
        )
        t0 = time.time()
        dt_e = bisect_max_dt(grid, hc, tm, cfg_e, dt_hi=60.0, tol=0.1)
        t_e = time.time() - t0
        if dt_e is None:
            line_e = f"{dx:.0f},explicit,FAIL,-,-,bisect_unable"
        else:
            cfl_e = dt_e * C_S_DRY / dx
            cfl_sub_e = cfl_e / args.n_acoustic_substeps
            line_e = (
                f"{dx:.0f},explicit,{dt_e:.3f},{cfl_e:.3f},"
                f"{cfl_sub_e:.4f},wall={t_e:.1f}s"
            )
            explicit_dt[dx] = dt_e
        print(line_e, flush=True)
        rows.append(line_e)

        # Semi-implicit
        cfg_si = cfg_e._replace(semi_implicit_acoustic=True)
        t0 = time.time()
        dt_si = bisect_max_dt(grid, hc, tm, cfg_si, dt_hi=60.0, tol=0.1)
        t_si = time.time() - t0
        if dt_si is None:
            line_si = f"{dx:.0f},semi_implicit,FAIL,-,-,bisect_unable"
        else:
            cfl_si = dt_si * C_S_DRY / dx
            speedup = (
                f" speedup={dt_si / dt_e:.2f}x"
                if dt_e is not None else ""
            )
            line_si = (
                f"{dx:.0f},semi_implicit,{dt_si:.3f},{cfl_si:.3f},"
                f"-, wall={t_si:.1f}s{speedup}"
            )
        print(line_si, flush=True)
        rows.append(line_si)

    out_path.write_text("\n".join(rows) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

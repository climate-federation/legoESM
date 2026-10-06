"""Bryan 1987 THC spinup: 500-1000 yr idealised hemispheric basin.

Reference
---------
Bryan, F. (1986/1987). "Parameter sensitivity of primitive equation
ocean general circulation models", J. Phys. Oceanogr. 17, 970-985.

Configuration
-------------
Idealised hemispheric sector basin (0-60 E, 0-70 N) with flat
bathymetry (4 km depth) on a lat-lon C-grid. CORE-II Normal-Year
Forcing (Large & Yeager 2009) drives the surface through the same
L&Y bulk-flux module the OMIP-2 driver uses, so the THC spinup is
forced by a repeating annual climatology.

Acceptance bars at equilibrium (500-1000 model years):

* MOC at the equator strengthens to 15-20 Sv (NADW analogue).
* Heat-content drift < 1 W/m^2 after 100 years.
* Stable single-cell equilibrium (no multiple-equilibrium drift).

Usage::

    # Smoke (1 day, exercises every code path):
    python scripts/run/ocean_long_runs/run_bryan_thc.py --smoke --output ...

    # Production (1000 years, cluster only):
    python scripts/run/ocean_long_runs/run_bryan_thc.py \\
        --years 1000 --resolution 36x36 \\
        --output results/ocean_long_runs/bryan_thc_1000yr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)


def _import_matrix_module():
    import importlib.util
    repo_root = Path(__file__).resolve().parents[3]
    matrix_path = repo_root / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "_rom_for_bryan", matrix_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rom_for_bryan"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_basin(resolution, *, H_max=4000.0, nlev=15,
                 lon_west=0.0, lon_east=60.0,
                 lat_south=0.0, lat_north=70.0,
                 scripts_dir=None):
    """Build the Bryan hemispheric basin on a regional lat-lon grid."""
    if scripts_dir is not None and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    matrix_mod = _import_matrix_module()
    tc = TestCase(
        case="bryan_thc", grid_type="latlon_regional",
        resolution=resolution, duration_days=365.0, quick_days=1.0,
        run_kwargs={
            "lon_west": lon_west, "lon_east": lon_east,
            "lat_south": lat_south, "lat_north": lat_north,
        },
    )
    # Bryan THC needs bottom drag + lateral viscosity to balance forcing.
    grid, z_coord, _, model, _, _, _ = _create_ocean_setup(
        tc, H_max=H_max, nlev=nlev,
        A_h=5.0e4, A_v=1.0e-4,
        bottom_drag_r=1.0e-3,
    )
    state = matrix_mod._create_rest_state(tc, grid, z_coord, H_max=H_max)
    return state, grid, z_coord, model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=str, default="24x24")
    p.add_argument("--years", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart-from", type=Path, default=None)
    p.add_argument("--dt", type=float, default=3600.0)
    p.add_argument("--diag-interval-years", type=int, default=10,
                   help="Emit RPE / energy / tracer diagnostics every N "
                        "years (default 10).")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parents[2] / "matrix"

    print(f"==> Building Bryan hemispheric basin @ {args.resolution}")
    state, grid, z_coord, model = _build_basin(
        args.resolution, scripts_dir=scripts_dir,
    )

    from legoesm.ocean import restart as _restart
    from legoesm.ocean.rpe import compute_rpe, rpe_drift_rate_per_m2
    from legoesm.ocean.budgets import (
        compute_energy_budget, compute_tracer_budget,
    )
    from legoesm.ocean.forcing import load_core2_nyf

    if args.restart_from is not None:
        print(f"==> Loading restart from {args.restart_from}")
        state = _restart.load_restart(args.restart_from, state)

    forcing = load_core2_nyf(n_time=12)   # monthly climatology

    from run_omip2 import step_with_omip2_forcing  # type: ignore

    rpe0 = compute_rpe(state, z_coord,
                       grid_type="latlon_regional", grid=grid)
    eb0 = compute_energy_budget(state, z_coord,
                                 grid_type="latlon_regional", grid=grid)
    tb0 = compute_tracer_budget(state, z_coord,
                                 grid_type="latlon_regional", grid=grid)
    print(f"   RPE_0 = {rpe0:.4e}  |  vol_0 = {tb0.volume:.3e}")

    dt = float(args.dt)
    n_years = 1 if args.smoke else args.years
    days_per_year = 1 if args.smoke else 365
    steps_per_year = int(days_per_year * 86400.0 / dt)

    yearly_diag: list[dict] = []
    rpe_prev = rpe0
    wall_t0 = time.time()
    for y in range(n_years):
        if y == 0 or (y + 1) % max(1, args.diag_interval_years) == 0 \
                or y == n_years - 1:
            print(f"==> Year {y + 1}/{n_years}")
        n_forc = forcing.u10.shape[0]
        for step in range(steps_per_year):
            idx_t = (step * n_forc) // max(1, steps_per_year)
            state = step_with_omip2_forcing(
                model, state, forcing=forcing, idx_t=idx_t, grid=grid,
                grid_type="latlon_regional", dt=dt,
            )
        state = jax.block_until_ready(state)

        if y == 0 or (y + 1) % max(1, args.diag_interval_years) == 0 \
                or y == n_years - 1:
            rpe_y = compute_rpe(state, z_coord,
                                 grid_type="latlon_regional", grid=grid)
            eb_y = compute_energy_budget(state, z_coord,
                                          grid_type="latlon_regional",
                                          grid=grid)
            tb_y = compute_tracer_budget(state, z_coord,
                                          grid_type="latlon_regional",
                                          grid=grid)
            rpe_flux = rpe_drift_rate_per_m2(
                rpe_prev, rpe_y,
                float(steps_per_year * dt * max(1, args.diag_interval_years)),
                eb0.area_total,
            )
            yearly_diag.append({
                "year": y + 1,
                "rpe": rpe_y, "rpe_flux_W_per_m2": rpe_flux,
                "KE": eb_y.KE, "APE": eb_y.APE,
                "volume": tb_y.volume,
                "heat_content": tb_y.heat_content,
                "salt_mass": tb_y.salt_mass,
            })
            rpe_prev = rpe_y
            _restart.save_restart(
                state, args.output / f"restart_year_{y + 1:05d}.npz",
                time_s=(y + 1) * steps_per_year * dt,
                step=(y + 1) * steps_per_year,
            )

    wall_s = time.time() - wall_t0
    summary = {
        "resolution": args.resolution,
        "years": n_years,
        "smoke": args.smoke,
        "wall_seconds": wall_s,
        "rpe_initial": rpe0,
        "area_total_m2": eb0.area_total,
        "yearly": yearly_diag,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWall time: {wall_s:.1f}s; summary -> {args.output}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

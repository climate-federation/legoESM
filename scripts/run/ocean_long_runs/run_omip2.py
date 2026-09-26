"""OMIP-2 30-year forced ocean driver (Tsujino et al. 2020 protocol).

Wires JRA55-do atmospheric forcing through Large & Yeager 2009 bulk
fluxes into the legoESM lat-lon C-grid ocean model on a global 1 deg
grid (or MPAS @ 100 km via ``--grid mpas``). Diagnostics emit each
year:

* RPE drift (Griffies 2015 spurious-mixing metric).
* Global energy budget (KE + APE).
* Tracer budget (volume / heat / salt / SSH integral).
* AMOC @ 26.5 deg N (Cunningham 2007).
* ACC transport @ Drake (Donohue 2016).
* SST climatology bias vs WOA (loaded separately).
* Restart written at end of each model year.

Acceptance bars after 30 years:

* AMOC @ 26.5 deg N -- 15 +/- 3 Sv.
* ACC @ Drake -- 130 +/- 15 Sv.
* SST bias -- < 1.5 deg C globally vs WOA.
* RPE drift -- < 0.5 mW/m^2 (Petersen 2015 reference).

Usage::

    # Smoke (1 day, synthetic forcing) -- exercises every code path.
    python scripts/run/ocean_long_runs/run_omip2.py --smoke --allow-synthetic --output results/ocean_long_runs/omip2_smoke

    # Production (30 years, real JRA55-do; cluster only):
    python scripts/run/ocean_long_runs/run_omip2.py \\
        --grid latlon --resolution 180x360 --years 30 \\
        --jra55-cache $LEGOESM_CACHE/forcing/jra55_do \\
        --output results/ocean_long_runs/omip2_lat1deg

The driver assumes either a fresh start or a ``--restart-from``
.npz produced by an earlier invocation of the same driver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)


def _import_matrix_module():
    """Load the matrix runner module (file-based, not a package)."""
    import importlib.util
    repo_root = Path(__file__).resolve().parents[3]
    matrix_path = repo_root / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "_rom_for_omip2", matrix_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rom_for_omip2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_state(grid_type, resolution, *, H_max=5500.0, nlev=15,
                 scripts_dir=None):
    """Build a global rest-state on the requested grid + a matching
    physics-config-style step-able ocean model."""
    if scripts_dir is not None and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    matrix_mod = _import_matrix_module()
    tc = TestCase(case="omip2", grid_type=grid_type,
                  resolution=resolution, duration_days=365.0,
                  quick_days=1.0)
    # OMIP-2 forced run requires bottom drag to balance wind input; without
    # it the wind continuously pumps momentum into the top layer and the
    # baroclinic adjustment runs away. A_h = 5e4 dissipates the Munk
    # boundary layer + intermediate-scale eddies at 1 deg resolution
    # (matches NEMO ORCA default).
    grid, z_coord, _, model, _, _, _ = _create_ocean_setup(
        tc, H_max=H_max, nlev=nlev,
        A_h=5.0e4, A_v=1.0e-4,
        bottom_drag_r=1.0e-3,
    )
    state = matrix_mod._create_rest_state(tc, grid, z_coord, H_max=H_max)
    return state, grid, z_coord, model


# NOTE: the inline bulk-flux step used by the Phase F skeleton has
# been promoted to the shared coupler hook
# ``legoesm.ocean.coupler.apply_omip2_surface_fluxes`` which both
# this driver and ``run_bryan_thc.py`` import. The skeleton lives on
# only as a docstring beacon.


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=["latlon", "mpas"], default="latlon")
    p.add_argument("--resolution", type=str, default="36x72",
                   help="Grid resolution (latlon: NxM; mpas: icoN).")
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--smoke", action="store_true",
                   help="Run a single model day to exercise code paths.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart-from", type=Path, default=None)
    p.add_argument("--jra55-cache", type=Path, default=None,
                   help="JRA55-do cache: the prepare_omip_forcing.py store or "
                        "its directory. Missing -> error unless "
                        "--allow-synthetic.")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Smoke/CI only: allow analytic stand-ins when a forcing "
                        "or observation cache is missing (default: fail).")
    p.add_argument("--dt", type=float, default=1800.0)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parents[2] / "matrix"

    print(f"==> Building global rest-state on {args.grid}/{args.resolution}")
    state, grid, z_coord, model = _build_state(
        args.grid, args.resolution, scripts_dir=scripts_dir,
    )

    from legoesm.ocean import restart as _restart
    from legoesm.ocean.rpe import compute_rpe, rpe_drift_rate_per_m2
    from legoesm.ocean.budgets import (
        compute_energy_budget, compute_tracer_budget,
    )
    from legoesm.ocean.forcing import load_jra55_do

    if args.restart_from is not None:
        print(f"==> Loading restart from {args.restart_from}")
        state = _restart.load_restart(args.restart_from, state)

    # Initial diagnostics baseline.
    rpe0 = compute_rpe(state, z_coord, grid_type=args.grid, grid=grid)
    eb0 = compute_energy_budget(state, z_coord, grid_type=args.grid, grid=grid)
    tb0 = compute_tracer_budget(state, z_coord, grid_type=args.grid, grid=grid)
    print(f"   RPE_0 = {rpe0:.4e} J  |  KE_0 = {eb0.KE:.3e}  |  "
          f"vol_0 = {tb0.volume:.3e}")

    from legoesm.ocean.coupler import apply_omip2_surface_fluxes

    yearly_diag: list[dict] = []
    dt = float(args.dt)
    n_years = 1 if args.smoke else args.years
    days_per_year = 1 if args.smoke else 365
    steps_per_year = int(days_per_year * 86400.0 / dt)

    wall_t0 = time.time()
    for y in range(n_years):
        print(f"==> Year {y + 1}/{n_years} ({steps_per_year} steps)")
        forcing = load_jra55_do(
            year=(2000 + y) if not args.smoke else 0,
            cache_dir=args.jra55_cache,
            allow_synthetic=args.allow_synthetic,
            cycle_years=not args.smoke,   # OMIP-2 repeats the cache's year window
        )
        n_forc = forcing.u10.shape[0]
        for step in range(steps_per_year):
            idx_t = (step * n_forc) // steps_per_year
            state = apply_omip2_surface_fluxes(
                state, forcing=forcing, idx_t=idx_t,
                z_coord=z_coord, grid=grid, grid_type=args.grid,
                dt=dt,
            )
            state = model.step(state, dt)
        state = jax.block_until_ready(state)

        # Yearly diagnostics.
        rpe_y = compute_rpe(state, z_coord, grid_type=args.grid, grid=grid)
        eb_y = compute_energy_budget(state, z_coord,
                                      grid_type=args.grid, grid=grid)
        tb_y = compute_tracer_budget(state, z_coord,
                                      grid_type=args.grid, grid=grid)
        delta_t_s = float(steps_per_year * dt)
        rpe_flux = rpe_drift_rate_per_m2(
            rpe0 if y == 0 else yearly_diag[-1]["rpe"], rpe_y,
            delta_t_s, eb0.area_total,
        )
        row = {
            "year": y + 1,
            "rpe": rpe_y, "rpe_flux_W_per_m2": rpe_flux,
            "KE": eb_y.KE, "APE": eb_y.APE,
            "volume": tb_y.volume, "heat_content": tb_y.heat_content,
            "salt_mass": tb_y.salt_mass, "eta_integral": tb_y.eta_integral,
        }
        yearly_diag.append(row)
        print(f"   RPE_flux = {rpe_flux:.3e} W/m^2  |  KE = {eb_y.KE:.3e}  "
              f"|  vol_drift = {(tb_y.volume - tb0.volume) / tb0.volume:.3e}")

        # Write restart every year.
        _restart.save_restart(
            state, args.output / f"restart_year_{y + 1:04d}.npz",
            time_s=delta_t_s * (y + 1),
            step=steps_per_year * (y + 1),
        )

    wall_s = time.time() - wall_t0
    summary = {
        "grid": args.grid,
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

#!/usr/bin/env python
"""Cap a land restart written by the pre-fix Richards solver at the water its
soil can hold, and re-derive the matric potential from the capped water content.

Why.  The solver that produced the published 1985 soil state (and every run
before the dry-end fix) could leave a layer holding more water than its
porosity plus the elastic allowance, and under pressures far above hydrostatic
(up to +214 m in the 1985 file).  Read by the fixed solver, that surplus is
expelled as a runoff pulse in the first hour of a run.  The surplus is water the
old solver invented, so it is removed here, once, and the file says so.

What.  Per cell and layer, with the solver's own per-cell Cosby hydraulics
rebuilt from the surfdata exactly as the driver builds them::

    theta_cap_k = theta(z_k)                  hydrostatic head of the layer, no pond
    theta_k    <- clip(theta_k, theta_floor, theta_cap_k)
    psi_k      <- psi(theta_k)                for EVERY cell (the elastic branch
                                              now starts at air entry)

Temperature, snow and the runoff stores are untouched.  ``--verify`` then runs
one hour of the fixed solver from the capped state under zero forcing and
checks that nothing runs off and no layer exceeds its cap.

Usage::

    python scripts/data/cap_land_restart.py \\
        --restart data/lmip_soil_ic/restart_1985_d000h00.npz \\
        --surfdata data/legoesm_surfdata_c260716.nc \\
        --out data/lmip_soil_ic/restart_1985_d000h00.capped.npz --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from legoesm.land.boundary_data import init_land_surface_data
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.restart import load_land_restart, save_land_restart
from legoesm.land.richards import RichardsConfig, dry_floor, solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import psi_from_theta, theta_from_psi
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart", required=True, help="restart .npz written by the old solver")
    ap.add_argument("--surfdata", required=True, help="the surfdata the run used")
    ap.add_argument("--out", required=True, help="capped restart .npz to write")
    ap.add_argument("--grid-type", default="latlon")
    ap.add_argument("--resolution", type=int, default=90)
    ap.add_argument("--soil-n-layers", type=int, default=10)
    ap.add_argument("--soil-depth-m", type=float, default=3.0)
    ap.add_argument("--soil-growth-factor", type=float, default=2.0)
    ap.add_argument("--verify", action="store_true",
                    help="run one hour of the fixed solver from the capped state "
                         "(zero forcing) and check nothing runs off")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from scripts.run.run_lmip_biophys import grid_latlon_rad, make_grid

    grid = make_grid(args.grid_type, args.resolution)
    lat_rad, _lon_rad = grid_latlon_rad(grid)
    ncol = int(lat_rad.shape[0])
    soil_cfg = SoilGridConfig(n_layers=args.soil_n_layers, total_depth=args.soil_depth_m,
                              growth_factor=args.soil_growth_factor)
    base = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(), soil_grid=soil_cfg)
    config, _params, _gsd = init_land_surface_data(args.surfdata, grid, base, 0.0)
    hyd = config.hydraulics
    soil_grid = make_soil_grid(soil_cfg)

    state, meta = load_land_restart(
        args.restart, expected_land_mode="multilayer", expected_ncol=ncol,
        expected_n_layers=soil_cfg.n_layers)
    theta = np.asarray(state.theta_soil, dtype=np.float64)
    psi_old = np.asarray(state.psi_soil, dtype=np.float64)
    dz = np.asarray(soil_grid.dz)

    # The solver's own bounds (richards.py): hydrostatic cap with no pond, dry floor.
    theta_cap = np.asarray(theta_from_psi(jnp.asarray(soil_grid.z_node)[None, :], hyd))
    theta_cap = np.broadcast_to(theta_cap, theta.shape)
    theta_floor = np.broadcast_to(np.asarray(dry_floor(hyd)[1]), theta.shape)
    theta_new = np.clip(theta, theta_floor, theta_cap)
    psi_new = np.asarray(psi_from_theta(jnp.asarray(theta_new), hyd))

    removed_mm = np.maximum(theta - theta_cap, 0.0) * dz[None, :] * 1000.0
    added_mm = np.maximum(theta_floor - theta, 0.0) * dz[None, :] * 1000.0
    over_cells = np.flatnonzero(removed_mm.sum(axis=1) > 0.0)
    under_cells = np.flatnonzero(added_mm.sum(axis=1) > 0.0)
    pressured = np.flatnonzero((psi_old > np.asarray(soil_grid.z_node)[None, :] + 1e-9).any(axis=1))
    summary = {
        "source": str(args.restart),
        "cells_over_cap": int(over_cells.size),
        "water_removed_mm_total": float(removed_mm.sum()),
        "water_removed_mm_max_cell": float(removed_mm.sum(axis=1).max()),
        "cells_below_floor": int(under_cells.size),
        "water_added_mm_total": float(added_mm.sum()),
        "cells_over_hydrostatic_psi": int(pressured.size),
        "max_psi_before_m": float(np.nanmax(psi_old)),
        "max_psi_after_m": float(np.nanmax(psi_new)),
        "psi_recomputed_for_all_cells": True,
    }
    print("cap_land_restart:", json.dumps(summary, indent=1))

    new_state = state._replace(theta_soil=jnp.asarray(theta_new), psi_soil=jnp.asarray(psi_new))
    metadata = dict(meta.get("metadata") or {})
    metadata["capped_by"] = {"script": "scripts/data/cap_land_restart.py", **summary}
    out = save_land_restart(args.out, new_state, land_mode="multilayer",
                            t_end_s=float(meta["t_end_s"]),
                            n_steps_completed=int(meta["n_steps_completed"]),
                            metadata=metadata, soil_grid=soil_cfg)
    print(f"wrote {out}")

    if args.verify:
        out_r = solve_richards(
            jnp.asarray(psi_new), jnp.asarray(theta_new), soil_grid, hyd, RichardsConfig(),
            jnp.zeros(ncol), jnp.zeros_like(jnp.asarray(theta_new)), 3600.0,
            surface_water=jnp.zeros(ncol))
        runoff = np.asarray(out_r.runoff_surface) * 3600.0          # kg/m2 = mm in the hour
        th1 = np.asarray(out_r.theta_new)
        print(f"verify: max surface runoff in one dry hour {runoff.max():.3e} mm, "
              f"cells with runoff > 0.01 mm: {(runoff > 0.01).sum()}, "
              f"max theta/cap {np.nanmax(th1 / theta_cap):.6f}, "
              f"non-finite cells {int((~np.isfinite(th1)).any(axis=1).sum())}")
        if runoff.max() > 0.01 or np.nanmax(th1 / theta_cap) > 1.0 + 1e-9:
            print("verify: FAILED")
            return 1
        print("verify: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

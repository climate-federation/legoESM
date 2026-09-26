"""Post-process AMOC / ACC / SST climate diagnostics from a long-run restart.

Reads the final ``restart_year_NNNN.npz`` from a long-run output
directory, rebuilds the grid + state, computes:

* AMOC streamfunction + value at 26.5 deg N (Cunningham 2007 RAPID).
* Barotropic streamfunction + ACC transport across Drake passage
  (Donohue 2016).
* SST bias vs the WOA annual-mean climatology (Locarnini 2018).

Writes a ``climate_diagnostics.json`` next to the restart + appends
a Markdown row to the long-run report.

Usage::

    python scripts/run/ocean_long_runs/postprocess_climate.py \\
        --run-dir results/ocean_long_runs/omip2_1deg_1yr \\
        --grid latlon --resolution 180x360 --H-max 5500 --nlev 15 \\
        --report docs/ocean/long_runs/results_omip2_local.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib.util

import numpy as np


def _matrix_module():
    repo_root = Path(__file__).resolve().parents[3]
    matrix_path = repo_root / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "_rom_for_postprocess", matrix_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_grid_and_state(grid_type, resolution, H_max, nlev, restart_path):
    scripts_dir = Path(__file__).resolve().parents[2] / "matrix"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    matrix_mod = _matrix_module()
    tc = TestCase("postprocess", grid_type, resolution, 1.0, 0.1)
    grid, z_coord, _, _, _, _, _ = _create_ocean_setup(
        tc, H_max=H_max, nlev=nlev,
    )
    template = matrix_mod._create_rest_state(tc, grid, z_coord, H_max=H_max)
    from legoesm.ocean.restart import load_restart
    state = load_restart(restart_path, template)
    return grid, z_coord, state


def _latest_restart(run_dir: Path) -> Path:
    restarts = sorted(run_dir.glob("restart_year_*.npz"))
    if not restarts:
        raise FileNotFoundError(f"No restart_year_*.npz under {run_dir}")
    return restarts[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--grid", choices=["latlon", "latlon_regional"],
                   default="latlon")
    p.add_argument("--resolution", type=str, required=True)
    p.add_argument("--H-max", type=float, default=5500.0)
    p.add_argument("--nlev", type=int, default=15)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    restart = _latest_restart(args.run_dir)
    print(f"==> Loading {restart}")
    grid, z_coord, state = _build_grid_and_state(
        args.grid, args.resolution, args.H_max, args.nlev, restart,
    )

    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.diagnostics_streamfunction import (
        moc_streamfunction, barotropic_streamfunction,
    )
    from legoesm.ocean.diagnostics_climate import (
        amoc_at_latitude, acc_transport, sst_climatology_bias,
    )
    from legoesm.ocean.forcing import load_woa_sst
    from legoesm import constants

    import jax.numpy as jnp
    h_k = np.asarray(compute_layer_thickness(
        jnp.asarray(state.eta.data),
        jnp.asarray(state.H_bathy.data),
        z_coord,
    ), dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    eta = np.asarray(state.eta.data, dtype=np.float64)
    H_bathy = np.asarray(state.H_bathy.data, dtype=np.float64)

    # AMOC streamfunction.
    psi_moc = moc_streamfunction(
        np.asarray(state.v.data, dtype=np.float64),
        h_k, eta, H_bathy, mask, grid,
    ) * 1.0e6   # back to m^3/s for the climate diagnostic
    lat_v = (np.degrees(np.asarray(grid.lat))
             - 0.5 * np.degrees(float(grid.dlat)))
    lat_v = np.concatenate([
        lat_v,
        [lat_v[-1] + np.degrees(float(grid.dlat))],
    ])
    depths = -np.asarray(z_coord.z_full_ref)
    amoc = amoc_at_latitude(psi_moc, lat_v, depths, target_lat=26.5)
    print(f"   AMOC @ 26.5 N  = {amoc.streamfunction_Sv:6.2f} Sv "
          f"(depth {amoc.depth_of_max_m:.0f} m)")

    # ACC transport.
    psi_bt = barotropic_streamfunction(
        np.asarray(state.u.data, dtype=np.float64),
        h_k, mask, grid,
    ) * 1.0e6
    lat_t = np.degrees(np.asarray(grid.lat))
    acc = acc_transport(psi_bt, lat_t)
    print(f"   ACC @ Drake     = {acc.transport_Sv:6.2f} Sv")

    # SST bias vs WOA: only against the real climatology on the model grid;
    # a synthetic stand-in would print an observational score that is not one.
    # Not scored (None) when the cache is missing or not on the model grid;
    # AMOC/ACC are still reported.
    sst_K = np.asarray(state.T.data)[..., 0] + constants.T_freeze
    bias = None
    try:
        sst_ref, _, _ = load_woa_sst(allow_synthetic=False)
    except FileNotFoundError as exc:
        print(f"   SST bias vs WOA = NOT SCORED ({exc})")
    else:
        if sst_ref.shape != sst_K.shape:
            print(f"   SST bias vs WOA = NOT SCORED (WOA cache on a "
                  f"{sst_ref.shape} grid, model SST on {sst_K.shape}; regrid "
                  "the WOA file to the model grid first)")
        else:
            area = np.asarray(grid.area)
            bias = sst_climatology_bias(sst_K, sst_ref, area, mask=mask)
            print(f"   SST bias vs WOA = {bias.bias_K:6.2f} K "
                  f"(RMSE {bias.rmse_K:.2f})")

    out = {
        "restart": str(restart),
        "grid": args.grid,
        "resolution": args.resolution,
        "amoc_Sv": amoc.streamfunction_Sv,
        "amoc_depth_m": amoc.depth_of_max_m,
        "acc_Sv": acc.transport_Sv,
        "sst_bias_K": None if bias is None else bias.bias_K,
        "sst_rmse_K": None if bias is None else bias.rmse_K,
    }
    json_path = args.run_dir / "climate_diagnostics.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n=> {json_path}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        # AMOC 15+/-3 Sv (RAPID), ACC 130+/-15 Sv (Donohue 2016),
        # SST bias < 1.5 K vs WOA.
        amoc_pass = abs(amoc.streamfunction_Sv - 15.0) <= 3.0
        acc_pass = abs(acc.transport_Sv - 130.0) <= 15.0
        if bias is None:
            sst_row = "| SST bias vs WOA | not scored | < 1.5 K | N/A |"
            rmse_row = "| SST RMSE | not scored | -- | -- |"
        else:
            sst_row = (f"| SST bias vs WOA | {bias.bias_K:.2f} K | < 1.5 K | "
                       f"{'PASS' if abs(bias.bias_K) < 1.5 else 'FAIL'} |")
            rmse_row = f"| SST RMSE | {bias.rmse_K:.2f} K | -- | -- |"
        md = [
            f"# Climate diagnostics -- {args.run_dir.name}",
            "",
            f"Restart: ``{restart.name}``",
            "",
            "| metric | value | acceptance | status |",
            "|---|---|---|---|",
            f"| AMOC @ 26.5 N | {amoc.streamfunction_Sv:.2f} Sv | 15 +/- 3 Sv | "
            f"{'PASS' if amoc_pass else 'FAIL'} |",
            f"| ACC @ Drake | {acc.transport_Sv:.2f} Sv | 130 +/- 15 Sv | "
            f"{'PASS' if acc_pass else 'FAIL'} |",
            sst_row,
            rmse_row,
        ]
        args.report.write_text("\n".join(md) + "\n")
        print(f"=> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

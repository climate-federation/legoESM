#!/usr/bin/env python
"""Bare-minimum forward LMIP land simulation — ERA5-forced, reusing main's infra.

This is a *forward* run (not calibration): it drives the 8-layer Richards
multilayer land column with ERA5 climatological forcing and the CLM5 surface map
(HWSD soil + PFT/LAI/cover), runs the STAGE-A soil-equilibrium spin-up + STAGE-B
seasonal cycle, and saves monthly land diagnostics.

Everything here is reused from the existing multilayer ERA5 calibrator
(``scripts/run/train_multilayer_land_era5.py``): the ERA5+CLM data loader, the
config/param assembly, the spin-up + physics forward (``forward_ml``, called with
``return_diag=True`` so it also returns the sensible/latent-heat fluxes and final
soil state).  The only thing this driver adds is a forward-only entry point + a
NetCDF writer.

Data (both public, NO credentials):
  * ERA5 npz:   python scripts/data/fetch_era5_hourly_climatology.py --out /tmp/era5_diurnal.npz
  * CLM surfdata: auto-downloaded by the loader (public UCAR URL) on first run.

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_lmip_era5.py \\
        --diurnal-npz /tmp/era5_diurnal.npz --n-sub 800 --out results/lmip_era5.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import scripts.run.train_multilayer_land_era5 as TML


def run(args) -> int:
    npz = Path(args.diurnal_npz)
    if not npz.exists():
        print(f"ERA5 forcing npz not found: {npz}\n"
              f"  build it (public GCS, no credentials):\n"
              f"    python scripts/data/fetch_era5_hourly_climatology.py --out {npz}")
        return 2

    # --- ERA5 forcing + CLM surface map (auto-downloads CLM surfdata). ---
    data = TML.load_training_data(str(npz), n_sub=args.n_sub, seed=args.seed)
    ncol = int(data["lat"].shape[0])

    # --- default params: untuned CLM5 table (constrain_ext maps into bounds). ---
    cp = TML.constrain_ext(TML.init_ext_params())

    print(f"forward LMIP: ncol={ncol} | NH={TML._NH} steps/month={TML._SPM} "
          f"| bulk={TML._BULK_SCHEME} | STAGE-A spin ({TML._EQ_STEPS}) + 2x STAGE-B ...")
    T, A, SH, LH, st = TML.forward_ml(cp, data, return_diag=True)
    T, A, SH, LH = map(np.asarray, (T, A, SH, LH))          # (12, ncol)

    # --- quick sanity vs ERA5 skin T (cos-lat weighted). ---
    skt = np.asarray(data["skt"]); w = np.asarray(data["w"])
    bias = float(np.sum(w[None] * (T - skt)) / np.sum(w) / 12)
    rmse = float(np.sqrt(np.sum(w[None] * (T - skt) ** 2) / np.sum(w) / 12))
    finite = bool(np.all(np.isfinite(T)))
    status = "PASS" if finite else "FAIL"
    print(f"vs ERA5 skin T: bias={bias:+.2f} K  RMSE={rmse:.2f} K  "
          f"(untuned CLM5 params) -> {status}")

    # --- save monthly diagnostics + final soil state. ---
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import xarray as xr
        lat = np.asarray(data["lat_deg"]); lon = np.asarray(data["lon_deg"])
        ds = xr.Dataset(
            {
                "T_sfc":  (("month", "col"), T),
                "shflx":  (("month", "col"), SH),
                "lhflx":  (("month", "col"), LH),
                "albedo": (("month", "col"), A),
                "skt_era5": (("month", "col"), skt),
                "T_soil_final": (("col", "layer"), np.asarray(st.T_soil)),
                "theta_soil_final": (("col", "layer"), np.asarray(st.theta_soil)),
            },
            coords={"month": np.arange(1, 13),
                    "lat": (("col",), lat), "lon": (("col",), lon)},
            attrs={"forcing": "ERA5 climatology", "surface": "CLM5 surfdata",
                   "params": "untuned CLM5", "spinup": "STAGE-A equilibrium + STAGE-B",
                   "bias_K": bias, "rmse_K": rmse},
        )
        ds.to_netcdf(out)
        print(f"wrote {out} ({ncol} land columns, 12 months)")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")

    return 0 if finite else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--diurnal-npz", default="/tmp/era5_diurnal.npz",
                    help="ERA5 forcing npz (scripts/data/fetch_era5_hourly_climatology.py)")
    ap.add_argument("--n-sub", type=int, default=800, help="number of land columns sampled")
    ap.add_argument("--seed", type=int, default=0, help="column-subsample seed")
    ap.add_argument("--out", default="results/lmip_era5.nc")
    return ap


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

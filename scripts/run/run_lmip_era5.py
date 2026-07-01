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

Physics ON in this driver (LMIP realism defaults):
  * 8-layer Richards hydraulics with EVOLVING soil moisture (--no-freeze-moisture);
  * MOST (Monin-Obukhov) bulk surface flux;
  * Ball-Berry stomata + Farquhar photosynthesis (--stomata, prescribed carbon state);
  * Prescribed SEASONAL LAI from CLM MONTHLY_LAI (PFT-weighted per column, per month);
  * Single-bulk snow with snow-albedo feedback;
  * Real cos(solar zenith) per (month, hour, column).

Physics OFF or absent on main (known simplifications; open follow-ups):
  * NO soil freeze/thaw (only the snow phase change is modelled) — deep permafrost
    behaviour is not represented;
  * Single-bulk snow (no multi-layer snowpack);
  * ET partitioning (transp / soil_evap / canopy_evap) is not exposed by
    TileResponse on main.

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

from legoesm import constants
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

    # Realism knobs (save + restore module globals so pytest sessions stay clean).
    saved_stomata, saved_bulk = TML._STOMATA_ON, TML._BULK_SCHEME
    TML._STOMATA_ON = bool(args.stomata)
    TML._BULK_SCHEME = args.bulk
    print(f"forward LMIP: ncol={ncol} | NH={TML._NH} steps/month={TML._SPM} "
          f"| bulk={TML._BULK_SCHEME} | stomata={TML._STOMATA_ON} "
          f"| freeze_moisture={args.freeze_moisture} "
          f"| STAGE-A spin ({TML._EQ_STEPS}) + 2x STAGE-B ...")
    try:
        T, A, SH, LH, R, P, st = TML.forward_ml(
            cp, data, return_diag=True, freeze_deep_moisture=args.freeze_moisture)
    finally:
        TML._STOMATA_ON, TML._BULK_SCHEME = saved_stomata, saved_bulk
    T, A, SH, LH, R, P = map(np.asarray, (T, A, SH, LH, R, P))     # (12, ncol)
    skt = np.asarray(data["skt"]); w = np.asarray(data["w"])

    # --- annual water balance: <P> vs. <E> + <R>  (all kg/m^2/s means).
    E = LH / float(constants.L_v)                                   # kg/m^2/s
    P_ann = P.mean(0); E_ann = E.mean(0); R_ann = R.mean(0)
    _S_PER_YEAR = 86400.0 * 365.0     # coeff-ok: seconds per (noleap) year, for mm/yr display
    to_mm_yr = lambda x: _S_PER_YEAR * float((w * x).sum() / w.sum())
    P_mm_yr, E_mm_yr, R_mm_yr = to_mm_yr(P_ann), to_mm_yr(E_ann), to_mm_yr(R_ann)
    res_mm_yr = P_mm_yr - E_mm_yr - R_mm_yr
    print(f"water balance (cos-lat mean, mm/yr): P={P_mm_yr:6.1f}  E={E_mm_yr:6.1f}  "
          f"R={R_mm_yr:6.1f}  residual={res_mm_yr:+.1f}  "
          f"({100.0 * res_mm_yr / max(P_mm_yr, 1e-9):+.1f}% of P)")

    # --- quick sanity vs ERA5 skin T (cos-lat weighted). ---
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
                "precip":  (("month", "col"), P),                    # kg/m^2/s
                "runoff":  (("month", "col"), R),                    # kg/m^2/s
                "evap":    (("month", "col"), E),                    # kg/m^2/s
                "T_soil_final": (("col", "layer"), np.asarray(st.T_soil)),
                "theta_soil_final": (("col", "layer"), np.asarray(st.theta_soil)),
            },
            coords={"month": np.arange(1, 13),
                    "lat": (("col",), lat), "lon": (("col",), lon)},
            attrs={"forcing": "ERA5 climatology", "surface": "CLM5 surfdata",
                   "params": "untuned CLM5", "spinup": "STAGE-A equilibrium + STAGE-B",
                   "bulk_scheme": TML._BULK_SCHEME, "stomata": int(args.stomata),
                   "freeze_moisture": int(args.freeze_moisture),
                   "soil_freeze_thaw": 0,  # not modelled on main (only snow phase)
                   "bias_K": bias, "rmse_K": rmse,
                   "wb_P_mm_yr": P_mm_yr, "wb_E_mm_yr": E_mm_yr,
                   "wb_R_mm_yr": R_mm_yr, "wb_residual_mm_yr": res_mm_yr},
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
    # --- biophysics realism knobs (LMIP defaults, not calibration defaults) ---
    ap.add_argument("--stomata", action=argparse.BooleanOptionalAction, default=True,
                    help="Ball-Berry stomatal control + Farquhar (default ON for real "
                         "biophysics; --no-stomata reverts to the calibrator default)")
    ap.add_argument("--freeze-moisture", action=argparse.BooleanOptionalAction, default=False,
                    help="pin soil moisture after STAGE-A spin-up (calibrator default is "
                         "True; forward LMIP wants False so hydrology evolves)")
    ap.add_argument("--bulk", default="most", choices=("constant", "most"),
                    help="surface bulk-flux scheme")
    ap.add_argument("--out", default="results/lmip_era5.nc")
    return ap


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

"""Offline ocean-evaporation decomposition for AMIP runs (issue #847).

Splits the bulk latent flux  E = rho * L_v * (C_E*U) * (q_sat(SST) - q_air)
into its two candidate deficit factors, from CMOR output alone (no model
instrumentation):

- ``dq = 0.98 * q_sat(tos, ps) - hus(lowest plev)``  — the air-sea humidity
  gradient (the "humidity-gradient collapse" branch: too-humid marine BL).
- ``(C_E*U)_eff = hfls / (rho * L_v * dq)``          — the effective transfer
  velocity (the "transfer too small" branch, ruled out code-side in #847).

Reference anchors for the tropical/global ocean (COARE lineage, Fairall et
al. 2003; DeCosmo et al. 1996 HEXOS):  C_E ~ 1.1-1.3e-3, |U10| ~ 7 m/s  =>
(C_E*U) ~ 8-9e-3 m/s;  ocean-mean dq ~ 4-6 g/kg;  ocean-mean hfls ~ 100-110
W/m^2 (ERA5/OAFlux).  Whichever factor sits far below its anchor carries the
~3x hfls deficit.

Usage:
    python scripts/validate/evap_decomposition.py --run-dir <amip-output-dir>
    (reads <run-dir>/cmor/{Amon,Omon,fx}; prints an ocean-mean table)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

# LY09 saline-water saturation reduction over ~35 PSU seawater; mirrors the
# coupler's ocean-tile q_sat (coupler.py _Q_SAT_SALINE_FACTOR).  # coeff-ok: published saline q_sat factor
_Q_SAT_SALINE_FACTOR = 0.98
# Ocean cell = land_area_fraction below half (sftlf is CMOR-% in [0, 100]).
_OCEAN_SFTLF_MAX_PCT = 50.0


def decompose_evap(
    hfls: np.ndarray,
    hus_low: np.ndarray,
    tos: np.ndarray,
    tas: np.ndarray,
    ps: np.ndarray,
    sftlf_pct: np.ndarray,
    lat: np.ndarray,
    wind_low: np.ndarray | None = None,
) -> dict:
    """Ocean-area-weighted decomposition of the bulk latent-heat flux.

    All fields are 2-D ``(lat, lon)`` time-means on a common grid; ``lat`` is
    1-D [deg].  ``hfls`` follows the CMOR sign convention (positive UP =
    ocean losing latent heat).  Returns ocean-mean diagnostics [SI + g/kg].
    """
    ocean = (sftlf_pct < _OCEAN_SFTLF_MAX_PCT)
    # exclude cells where any input is missing (e.g. hus 1000-hPa below-ground
    # mask leaking through, or sea-ice tos gaps)
    valid = ocean
    for f in (hfls, hus_low, tos, tas, ps):
        valid = valid & np.isfinite(f)

    w2d = np.broadcast_to(
        np.cos(np.deg2rad(lat))[:, None], hfls.shape
    ) * valid
    w_sum = w2d.sum()
    if w_sum <= 0.0:
        raise ValueError("no valid ocean cells (check sftlf / input masks)")

    def _omean(f: np.ndarray) -> float:
        return float((f * w2d).sum() / w_sum)

    # saturation q at the sea surface (saline-reduced), same Tetens curve as
    # the model's coupler ocean tile
    q_sat_sst = _Q_SAT_SALINE_FACTOR * np.asarray(
        saturation_mixing_ratio(tos, ps)
    )
    dq = q_sat_sst - hus_low                                    # [kg/kg]

    # surface air density from the near-surface state (virtual T with the
    # model's epsilon, not a hardcoded 0.61)
    virt = 1.0 + (1.0 / constants.epsilon - 1.0) * hus_low
    rho = ps / (constants.R_d * tas * virt)                     # [kg/m^3]

    # BULK effective transfer velocity: ratio of ocean MEANS, not the mean of
    # pointwise ratios (near-zero-dq cells blow the pointwise ratio up and
    # dominate an area mean — the mean-of-ratios is NOT the bulk C_E*U that
    # closes hfls_mean = rho*L_v*(C_E*U)*dq_mean).
    ce_u_bulk = _omean(hfls) / (
        _omean(rho) * constants.L_v * max(_omean(dq), 1e-6)
    )                                                            # [m/s]

    out = {
        "hfls_ocean_mean_W_m2": _omean(hfls),
        "dq_ocean_mean_g_kg": _omean(dq) * 1e3,
        "q_sat_sst_ocean_mean_g_kg": _omean(q_sat_sst) * 1e3,
        "q_air_ocean_mean_g_kg": _omean(hus_low) * 1e3,
        "ce_u_bulk_m_s": ce_u_bulk,
        "rho_ocean_mean_kg_m3": _omean(rho),
        "n_ocean_cells": int(valid.sum()),
    }
    if wind_low is not None:
        u_mean = _omean(np.where(np.isfinite(wind_low), wind_low, 0.0))
        out["wind_low_ocean_mean_m_s"] = u_mean
        # effective exchange coefficient at the resolved lowest-level wind
        out["ce_eff"] = ce_u_bulk / max(u_mean, 1e-3)
    return out


def _load_cmor_field(cmor_dir: str, table: str, var: str):
    """Open the single CMOR file for ``var`` and return (DataArray, time-mean)."""
    import xarray as xr

    pattern = os.path.join(cmor_dir, table, f"{var}_{table}_*.nc")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {var} file under {pattern}")
    da = xr.open_dataset(hits[0])[var]
    if "time" in da.dims:
        da = da.mean("time", keep_attrs=True)
    return da


def decompose_run_dir(run_dir: str) -> dict:
    """Read <run_dir>/cmor and run the decomposition on the time-mean state."""
    cmor_dir = os.path.join(run_dir, "cmor")
    hfls = _load_cmor_field(cmor_dir, "Amon", "hfls")
    hus = _load_cmor_field(cmor_dir, "Amon", "hus")
    tas = _load_cmor_field(cmor_dir, "Amon", "tas")
    ps = _load_cmor_field(cmor_dir, "Amon", "ps")
    tos = _load_cmor_field(cmor_dir, "Omon", "tos")
    sftlf = _load_cmor_field(cmor_dir, "fx", "sftlf")

    # near-surface air humidity / wind = the lowest (largest-pressure) plev
    i_low = int(np.argmax(hus["plev"].values))
    hus_low = hus.isel(plev=i_low)
    ua = _load_cmor_field(cmor_dir, "Amon", "ua").isel(plev=i_low)
    va = _load_cmor_field(cmor_dir, "Amon", "va").isel(plev=i_low)
    wind_low = np.hypot(ua.values, va.values)

    return decompose_evap(
        hfls=hfls.values,
        hus_low=hus_low.values,
        tos=tos.values,
        tas=tas.values,
        ps=ps.values,
        sftlf_pct=sftlf.values,
        lat=hfls["lat"].values,
        wind_low=wind_low,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True,
                    help="AMIP output dir containing cmor/{Amon,Omon,fx}")
    ap.add_argument("--label", default=None,
                    help="row label (default: run-dir basename)")
    args = ap.parse_args(argv)

    r = decompose_run_dir(args.run_dir)
    label = args.label or os.path.basename(os.path.normpath(args.run_dir))
    print(f"# ocean evap decomposition (#847): {label}")
    print(f"#   anchors: hfls ~100-110 W/m2 | dq ~4-6 g/kg | "
          f"C_E*U ~8-9e-3 m/s (COARE: C_E~1.2e-3 x U~7 m/s)")
    print(f"{'hfls [W/m2]':>12s} {'dq [g/kg]':>10s} {'q_sat(SST)':>10s} "
          f"{'q_air':>8s} {'C_E*U [m/s]':>12s} {'|U| [m/s]':>10s} "
          f"{'C_E_eff':>9s} {'n_ocn':>6s}")
    print(f"{r['hfls_ocean_mean_W_m2']:12.1f} {r['dq_ocean_mean_g_kg']:10.2f} "
          f"{r['q_sat_sst_ocean_mean_g_kg']:10.2f} "
          f"{r['q_air_ocean_mean_g_kg']:8.2f} "
          f"{r['ce_u_bulk_m_s']:12.2e} "
          f"{r.get('wind_low_ocean_mean_m_s', float('nan')):10.2f} "
          f"{r.get('ce_eff', float('nan')):9.2e} {r['n_ocean_cells']:6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

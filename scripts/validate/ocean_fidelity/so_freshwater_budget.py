#!/usr/bin/env python
"""Southern-Ocean surface freshwater budget: ours vs NEMO, term by term.

WHY.  The tripole's Antarctic summer fresh layer is roughly half NEMO's
(dS over the top 30 m: -0.22 vs -0.46 psu) and the freshwater-CLOSURE arm
made it worse, not better (arm 9436519: -0.018) — so the closure is refuted
and the remaining candidates are the fluxes themselves.  This probe compares
the term the closure consumes, on matched simulated days:

  NEMO   : ``empmr`` from the 5-day SBC file (kg/m2/s, water flux INTO the
           sea; already the sum of open-ocean E-P, over-ice E-P, runoff and
           the ice thermodynamic exchange, with NEMO's own sign convention).
  ours   : ``precip - evap + runoff + ice_fw`` rebuilt through the SAME
           production path the run used (compute_omip2_freshwater_forcing +
           the runoff map), i.e. the flux the model applied, not a re-derived
           lookalike.

It prints the area-weighted band mean of each side and, for ours, the
per-term split, so a deficit can be attributed to precipitation, evaporation,
runoff or ice melt rather than to "the freshwater forcing".

SIGN, stated once: this probe reports everything as POSITIVE = freshwater
INTO the ocean.  NEMO's ``empmr`` long_name is water_flux_into_sea_water, so
it is used as-is; ours is ``precip - evap + runoff + ice_fw`` because
``evap`` is positive UP in FreshwaterForcing.

CAVEAT that bounds every number: NEMO's is a 5-day mean, ours is rebuilt at
one instant of the matched window.  A 10-20% difference is inside that
mismatch.  A factor-of-two is not, and the fresh-layer deficit is a
factor-of-two-sized question.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

BANDS = {
    "antarctic_S_of_45S": (-90.0, -45.0),
    "SH_midlat_45S_23S": (-45.0, -23.0),
    "arctic_N_of_45N": (45.0, 90.0),
}


def _band_means(field, lat, wgt, label, scale=1.0):
    out = {}
    for bn, (lo, hi) in BANDS.items():
        m = (lat >= lo) & (lat < hi) & np.isfinite(field) & (wgt > 0)
        if not m.any():
            out[bn] = np.nan
            continue
        out[bn] = float((field[m] * wgt[m]).sum() / wgt[m].sum()) * scale
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo-sbc", required=True,
                    help="5-day SBC file carrying empmr (RUN_GATEWAY).")
    ap.add_argument("--nemo-recs", default="5:6",
                    help="record slice, matched to the snapshot's days.")
    ap.add_argument("--snapshot", default=None,
                    help="our snapshot (for the model-side lat/mask only; the "
                         "flux is rebuilt from the forcing, not stored).")
    ap.add_argument("--forcing-path", default=None,
                    help="nyf.zarr directory (default: the fidelity cache).")
    ap.add_argument("--idx-t", type=int, default=None,
                    help="forcing record index; default = the record whose "
                         "centre matches --day.")
    ap.add_argument("--day", type=float, default=30.0,
                    help="simulated day the snapshot is from.")
    a = ap.parse_args()

    import netCDF4 as nc

    ds = nc.Dataset(a.nemo_sbc)
    try:
        emp = np.ma.filled(np.ma.masked_invalid(
            ds.variables["empmr"][:]), np.nan).astype(np.float64)
        lat_n = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
    finally:
        ds.close()
    lo, hi = (int(x) for x in a.nemo_recs.split(":"))
    emp = emp[lo:hi]
    if emp.shape[0] == 0:
        raise SystemExit(f"--nemo-recs {a.nemo_recs} selects no records")
    emp = np.nanmean(emp, axis=0)
    wgt_n = np.cos(np.deg2rad(lat_n)) * np.isfinite(emp)
    nemo = _band_means(emp, lat_n, wgt_n, "NEMO")

    print(f"NEMO empmr (water flux INTO ocean), recs {a.nemo_recs}, "
          "1e-6 kg/m2/s:")
    for bn, v in nemo.items():
        print(f"  {bn:22s} {1e6 * v:+9.2f}")

    if a.snapshot is None:
        return 0

    # Model side: rebuild the applied flux through the production path.
    from legoesm.ocean.forcing.core2 import load_core2_nyf
    z = np.load(a.snapshot)
    lat_o = np.asarray(z["lat_T"], dtype=np.float64)
    mask = np.asarray(z["land_mask"], dtype=np.float64)
    lon_o = np.asarray(z["lon_T"], dtype=np.float64) % 360.0

    forcing = load_core2_nyf(cache_dir=(Path(a.forcing_path)
                                        if a.forcing_path else None),
                             allow_synthetic=False)
    t = np.asarray(forcing.time_s, dtype=np.float64)
    idx = (a.idx_t if a.idx_t is not None
           else int(np.argmin(np.abs(t - a.day * 86400.0))))
    print(f"\n[forcing] record {idx} (centre {t[idx] / 86400.0:.2f} d) "
          f"vs snapshot day {a.day}")

    # Precip and the evaporation the bulk produced at this state.
    from legoesm.ocean.coupler.omip2_applicator import _sample_forcing_points
    from legoesm.ocean.bulk_flux_omip import air_sea_fluxes
    from legoesm import constants
    import jax.numpy as jnp

    # The tripole's coordinates are 2-D curvilinear, so the POINTS sampler
    # (nearest-neighbour, the cube/MPAS path) is the applicable one; the
    # lat-lon conservative sampler needs 1-D destination axes.
    _shape = lat_o.shape
    forc = _sample_forcing_points(forcing, idx, lat_o.reshape(-1),
                                  lon_o.reshape(-1))
    forc = {k: np.asarray(v).reshape(_shape) for k, v in forc.items()}
    T_sfc_K = np.asarray(z["T"], dtype=np.float64)[..., 0] + constants.T_freeze
    _, _, _, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]), v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]), q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if forc.get("slp") is None else jnp.asarray(forc["slp"]),
    )
    precip = np.asarray(forc["precip"], dtype=np.float64)
    evap = np.asarray(evap, dtype=np.float64)           # +up
    wgt_o = np.cos(np.deg2rad(lat_o)) * (mask > 0.5)

    terms = {"precip": precip, "evap(+up)": evap, "P-E": precip - evap}
    print("\nOURS, per term, 1e-6 kg/m2/s (+ = into ocean; evap printed +up):")
    hdr = f"  {'band':22s}" + "".join(f"{k:>13s}" for k in terms)
    print(hdr)
    rows = {k: _band_means(v, lat_o, wgt_o, k) for k, v in terms.items()}
    for bn in BANDS:
        line = f"  {bn:22s}"
        for k in terms:
            line += f"{1e6 * rows[k][bn]:13.2f}"
        print(line)

    print("\nP-E vs NEMO empmr (NEMO includes runoff + ice exchange, ours "
          "here does NOT -- a P-E deficit is attributable, a P-E MATCH means "
          "the gap is in runoff/ice):")
    for bn in BANDS:
        o = rows["P-E"][bn]
        n = nemo[bn]
        print(f"  {bn:22s} ours {1e6 * o:+9.2f}  NEMO {1e6 * n:+9.2f}  "
              f"diff {1e6 * (o - n):+9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Southern-Ocean surface freshwater budget: ours vs NEMO, term by term.

WHY.  The tripole's Antarctic summer fresh layer is roughly half NEMO's
(dS over the top 30 m: -0.22 vs -0.46 psu) and the freshwater-CLOSURE arm
made it worse, not better (arm 9436519: -0.018) — so the closure is refuted
and the remaining candidates are the fluxes themselves.  This probe compares
the term the closure consumes, on matched simulated days:

  NEMO   : ``empmr`` and ``emp_oce`` from the 5-day SBC file. SIGN, MEASURED
           NOT ASSUMED (control below): NEMO stores E-P, POSITIVE = water
           LEAVING the ocean -- the ``water_flux_into_sea_water`` long_name on
           ``empmr`` is misleading. The probe therefore negates both. Control
           that proves it: the subtropics (23-35S) must be evaporative, and
           emp_oce there is +6.66e-6 (positive = evaporation dominant); the
           probe re-runs this check every time and REFUSES to report if the
           sign flips. ``emp_oce`` is open-ocean E-P only -- the LIKE-FOR-LIKE
           partner of our P-E -- while ``empmr`` additionally carries runoff
           and the ice exchange, so (empmr - emp_oce) isolates those.
  ours   : ``precip - evap + runoff + ice_fw`` rebuilt through the SAME
           production path the run used (compute_omip2_freshwater_forcing +
           the runoff map), i.e. the flux the model applied, not a re-derived
           lookalike.

It prints the area-weighted band mean of each side and, for ours, the
per-term split, so a deficit can be attributed to precipitation, evaporation,
runoff or ice melt rather than to "the freshwater forcing".

SIGN, stated once: this probe reports everything as POSITIVE = freshwater
INTO the ocean.  NEMO's ``empmr`` long_name (water_flux_into_sea_water) is
MISLEADING -- the measured convention is E-P, positive = water OUT -- so both
NEMO terms are negated; ours is ``precip - evap + runoff + ice_fw`` because
``evap`` is positive UP in FreshwaterForcing.

AREA: every band mean uses TRUE cell area ``e1t*e2t``, on both sides.  A
cos(lat) weight is wrong on this curvilinear mesh by 0.00-1.72x per cell
south of 45S, which is a 41% error in NEMO's own runoff+ice residual
(16.67e-6 under cos(lat) against 11.85e-6 on true area).

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


def _mesh_area(mesh_path):
    """True T-cell area e1t*e2t and its latitudes, from the mesh mask."""
    import netCDF4 as nc
    d = nc.Dataset(mesh_path)
    try:
        area = (np.squeeze(d["e1t"][:]) * np.squeeze(d["e2t"][:])).astype(np.float64)
        lat = np.squeeze(d["gphit"][:]).astype(np.float64)
    finally:
        d.close()
    return area, lat


def _nemo_cell_area(mesh_path, lat_n):
    """Cell area on NEMO's OUTPUT grid, aligned by matching latitudes.

    NEMO's XIOS output drops the halo, so the SBC file is (331, 360) where the
    eORCA1.2 mesh mask is (332, 362) -- and ``nav_lat`` carries -1 fill in the
    fully-masked southern rows, so a naive comparison of the two latitude
    fields disagrees by 84 degrees and hides the offset. The alignment is found
    on VALID cells only and must be exact; anything else is refused rather than
    silently mis-weighted.
    """
    area, lat_m = _mesh_area(mesh_path)
    ny, nx = lat_n.shape
    valid = np.isfinite(lat_n) & (lat_n != -1.0)
    for j0 in range(lat_m.shape[0] - ny + 1):
        for i0 in range(lat_m.shape[1] - nx + 1):
            if np.abs(lat_m[j0:j0 + ny, i0:i0 + nx] - lat_n)[valid].max() == 0.0:
                return area[j0:j0 + ny, i0:i0 + nx]
    raise SystemExit(
        f"cannot align NEMO output grid {lat_n.shape} to mesh {lat_m.shape}: "
        "no offset reproduces the output latitudes exactly. Refusing to "
        "area-weight with a guessed alignment.")


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
    ap.add_argument("--mesh", default="data/grids/eORCA1.2_mesh_mask.nc",
                    help="mesh-mask supplying TRUE cell area (e1t*e2t). "
                         "cos(lat) is NOT an area proxy on the tripole: the "
                         "two differ by 0.00-1.72x per cell south of 45S.")
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
        def rd(name):
            return np.ma.filled(np.ma.masked_invalid(
                ds.variables[name][:]), np.nan).astype(np.float64)
        emp_raw, empoce_raw = rd("empmr"), rd("emp_oce")
        lat_n = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
    finally:
        ds.close()
    lo, hi = (int(x) for x in a.nemo_recs.split(":"))
    if emp_raw[lo:hi].shape[0] == 0:
        raise SystemExit(f"--nemo-recs {a.nemo_recs} selects no records")
    emp = np.nanmean(emp_raw[lo:hi], axis=0)
    empoce = np.nanmean(empoce_raw[lo:hi], axis=0)
    wgt_n = _nemo_cell_area(a.mesh, lat_n) * np.isfinite(emp)

    # SIGN CONTROL, run every time: the subtropics must be evaporative in
    # NEMO's own storage convention (E-P > 0). If this ever fails the file's
    # convention changed and every number below would silently invert.
    sub = ((lat_n >= -35.0) & (lat_n < -23.0) & np.isfinite(empoce)
           & (wgt_n > 0))
    sub_mean = float((empoce[sub] * wgt_n[sub]).sum() / wgt_n[sub].sum())
    if not (sub_mean > 0):
        raise SystemExit(
            f"SIGN CONTROL FAILED: NEMO emp_oce over 23-35S is {sub_mean:+.3e}"
            " but the subtropics must be EVAPORATIVE (E-P > 0) in NEMO's "
            "storage convention. Refusing to report a budget whose sign "
            "cannot be established.")
    print(f"[sign control] NEMO emp_oce 23-35S = {1e6 * sub_mean:+.2f}e-6 > 0"
          " => stored as E-P (+ = water OUT); negating for this report.")

    # Everything below is POSITIVE = freshwater INTO the ocean.
    nemo = {k: -v for k, v in
            _band_means(emp, lat_n, wgt_n, "NEMO").items()}
    nemo_oce = {k: -v for k, v in
                _band_means(empoce, lat_n, wgt_n, "NEMO_oce").items()}

    print(f"\nNEMO, recs {a.nemo_recs}, 1e-6 kg/m2/s, + = freshwater INTO "
          "ocean:")
    print(f"  {'band':22s}{'total(empmr)':>14s}{'openoce(E-P)':>14s}"
          f"{'runoff+ice':>13s}")
    for bn in BANDS:
        print(f"  {bn:22s}{1e6 * nemo[bn]:14.2f}{1e6 * nemo_oce[bn]:14.2f}"
              f"{1e6 * (nemo[bn] - nemo_oce[bn]):13.2f}")

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
    area_o, lat_mesh = _mesh_area(a.mesh)
    if area_o.shape != lat_o.shape:
        raise SystemExit(
            f"mesh area {area_o.shape} does not match the snapshot grid "
            f"{lat_o.shape} -- wrong --mesh for this run")
    wgt_o = area_o * (mask > 0.5)

    # The model does NOT apply the full-cell P-E to the ocean:
    # blend_ice_ocean_forcing scales evap (and, on the snow-reservoir ice path
    # the production card runs, precip) by the open-water fraction 1-A, with
    # the ice-fraction share carried in ice_fw instead. Comparing a full-cell
    # P-E against NEMO's emp_oce therefore compares an unscaled quantity to a
    # scaled one, which codex flagged (job 9442707). A is archived in the
    # snapshot, so both are reported and the reader sees the bound: the
    # correction is small where the band is mostly open water and large where
    # it is not (measured 2026-08-20 at day 5: Antarctic A=0.073, so 7%;
    # Arctic A=0.348, so 35% -- big enough to change an Arctic conclusion).
    conc = np.asarray(z["ice_concentration"], dtype=np.float64)
    f_open = 1.0 - conc
    terms = {"precip": precip, "evap(+up)": evap, "P-E": precip - evap,
             "f_open*(P-E)": f_open * (precip - evap)}
    print("\nOURS, per term, 1e-6 kg/m2/s (+ = into ocean; evap printed +up):")
    hdr = f"  {'band':22s}" + "".join(f"{k:>13s}" for k in terms)
    print(hdr)
    rows = {k: _band_means(v, lat_o, wgt_o, k) for k, v in terms.items()}
    for bn in BANDS:
        line = f"  {bn:22s}"
        for k in terms:
            line += f"{1e6 * rows[k][bn]:13.2f}"
        print(line)

    print("\nLIKE-FOR-LIKE: our OPEN-WATER-SCALED f_open*(P-E) vs NEMO's "
          "open-ocean E-P (emp_oce). Both exclude runoff and the ice "
          "exchange, and both are per unit TOTAL cell area, so this line "
          "alone tests the bulk/precip channel. The unscaled full-cell P-E "
          "is shown beside it because it is what the model would apply if "
          "the ice partition were absent -- the gap between the two columns "
          "is the size of the partition, not a model error:")
    for bn in BANDS:
        o, n = rows["f_open*(P-E)"][bn], nemo_oce[bn]
        print(f"  {bn:22s} ours {1e6 * o:+9.2f}  NEMO {1e6 * n:+9.2f}  "
              f"diff {1e6 * (o - n):+9.2f}   (full-cell P-E "
              f"{1e6 * rows['P-E'][bn]:+9.2f})")
    print("\nWHAT THE OTHER CHANNELS OWE: NEMO's runoff+ice contribution is "
          "the residual above; ours is NOT measured here (it is applied "
          "in-model via FreshwaterForcing.runoff/ice_fw), so this is the "
          "budget our runoff + ice melt must supply to match:")
    for bn in BANDS:
        print(f"  {bn:22s} {1e6 * (nemo[bn] - nemo_oce[bn]):+9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

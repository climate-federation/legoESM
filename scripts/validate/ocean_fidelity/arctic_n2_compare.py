#!/usr/bin/env python
"""Arctic N² : ours vs NEMO, through the SAME routine, on a MATCHED window.

WHY.  The Arctic bias was shown to be a vertical redistribution, not an
inventory error (``arctic_salt_vs_volume.py``): our top 19 m varies by 0.018
psu where NEMO varies by 0.83, i.e. the column is mixed flat.  The surface TKE
boundary condition was then ELIMINATED as the cause -- our under-ice surface
value reproduces NEMO's ``rn_ebb*|tau|/rho0`` to four digits.  Since the TKE
mixing coefficient goes as ``l*sqrt(e)`` and ``e`` matches, what remains is the
LENGTH scale, whose dominant interior term is the buoyancy length
``l_b = sqrt(2e)/N``.

THIS PROBE measures ``N²`` for both models and the buoyancy length they imply,
which quantifies the suspected feedback: weak stratification -> long mixing
length -> more mixing -> weaker stratification.

MATCHED, deliberately:
  * both T/S go through the model's OWN ``compute_buoyancy_frequency_nemo_bn2``
    (NEMO ``eosbn2.F90`` bn2_t), so the instrument is identical on both sides;
  * NEMO's month-1 grid_T is used, NOT the grid_W ``bn2``, which is an ANNUAL
    MEAN -- Arctic winter mixing dwarfs summer, so an annual mean would flatter
    whichever side we wanted and prove nothing.

CAVEATS, stated rather than discovered later:
  * the ABSOLUTE N² depends on the S-EOS coefficient set; because the same
    routine and config are applied to both sides, the OURS-vs-NEMO contrast is
    the robust quantity, not the absolute value;
  * ours is an instantaneous day-30 state, NEMO's is a January monthly mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_HERE.parents[3] / "packages" / _p))

from global_tracer_content import (  # noqa: E402
    _native, load_mesh_depth_1d, load_mesh_latitude, load_mesh_longitude,
    load_mesh_metrics,
)


def _mesh_w_1d(mesh_mask_path):
    """Interface depths AND raw w-cell thicknesses from the mesh.

    ``e3w_1d`` is read rather than reconstructed as a depth difference. The
    closure demands the raw-mesh value and refuses the reconstruction as
    legacy opt-in, and it is right to: a sibling probe measured the two
    divisors differing by 6.6e-4 median and 2.9e-3 max on this very mesh.
    Same index convention as the depths -- interior interface i is NEMO's
    w-point jk=i+1, so both ladders drop their first entry together.
    """
    import netCDF4 as nc
    ds = nc.Dataset(mesh_mask_path)
    try:
        g = np.asarray(ds.variables["gdepw_1d"][:], dtype=np.float64).squeeze()
        if "e3w_1d" not in ds.variables:
            raise SystemExit(
                f"FATAL: {mesh_mask_path} carries no e3w_1d; the N2 closure "
                "requires raw-mesh w thicknesses")
        e = np.asarray(ds.variables["e3w_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    return g, e


def _n2(T, S, gdept, gdepw_int, e3w_int):
    """Model's own NEMO bn2. T/S level-LAST, per the function's contract."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(gdept), jnp.asarray(gdepw_int),
        e3w_int=jnp.asarray(e3w_int)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-month", type=int, default=1)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--arctic-lat", type=float, default=60.0)
    p.add_argument("--site", action="append", default=[], metavar="SPEC",
                   help="repeatable named box, NAME:LAT0,LAT1:LON0,LON1 "
                        "(degrees east, 0-360; LON0 > LON1 wraps the prime "
                        "meridian). Given one or more, the Arctic band is "
                        "replaced by these boxes; given none, behaviour is "
                        "exactly the Arctic band as before. Use the equals "
                        "form -- a negative latitude reads as a flag.")
    p.add_argument("--out-json", default=None,
                   help="write the per-site profiles here so downstream "
                        "plots read data rather than a scraped log")
    p.add_argument("--max-depth-m", type=float, default=120.0,
                   help="stop the per-level table below this depth. The "
                        "default suits the Arctic halocline this probe was "
                        "built for; a deep-convection site needs more.")
    a = p.parse_args()

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    lat = load_mesh_latitude(a.mesh_mask)
    lon = load_mesh_longitude(a.mesh_mask) % 360.0
    gdept = load_mesh_depth_1d(a.mesh_mask)
    gdepw, e3w = _mesh_w_1d(a.mesh_mask)
    gdepw_int = gdepw[1:]                      # interior interfaces, (nlev-1,)
    e3w_int = e3w[1:]                          # same convention, (nlev-1,)

    wet = tmask > 0.5
    has_water = wet.any(axis=0)
    regions = []
    for spec in a.site:
        nm, latspec, lonspec = spec.split(":")
        la0, la1 = (float(v) for v in latspec.split(","))
        lo0, lo1 = (float(v) % 360.0 for v in lonspec.split(","))
        inlat = (lat >= la0) & (lat <= la1)
        # lo0 > lo1 means the box wraps the prime meridian, which the
        # Greenland/Irminger sector does. Selecting with a plain AND there
        # returns an EMPTY box rather than an error, so the wrap is handled
        # explicitly and the cell count is printed for every site.
        inlon = ((lon >= lo0) | (lon <= lo1)) if lo0 > lo1 else \
                ((lon >= lo0) & (lon <= lo1))
        regions.append((nm, inlat & inlon & has_water))
    if not regions:
        regions = [(f"arctic>={a.arctic_lat:g}N",
                    (lat >= a.arctic_lat) & has_water)]
    for nm, sel in regions:
        if not sel.any():
            raise SystemExit(f"FATAL: region {nm!r} selects no wet cells")

    z = dict(np.load(a.snapshot))
    T_our = np.transpose(_native(z["T"]), (1, 2, 0))     # -> (nj, ni, nlev)
    S_our = np.transpose(_native(z["S"]), (1, 2, 0))
    tke = _native(z["tke"])                              # (nlev-1, nj, ni)

    import netCDF4 as nc
    ds = nc.Dataset(a.nemo_gridt)
    try:
        it = a.nemo_month - 1
        def v(n):
            return np.ma.filled(np.ma.masked_invalid(ds.variables[n][it]),
                                0.0).astype(np.float64)
        S_nemo = np.transpose(v("so"), (1, 2, 0))        # (75,331,360)->(nj,ni,75)
        T_nemo = np.transpose(v("to"), (1, 2, 0))
    finally:
        ds.close()
    for nm, arr in (("so", S_nemo), ("to", T_nemo)):
        if np.abs(arr).max() > 1.0e6:
            raise SystemExit(f"FATAL: {nm} carries fill values")
    if S_nemo.shape != S_our.shape:
        raise SystemExit(f"NEMO {S_nemo.shape} vs ours {S_our.shape}")

    n2_our = _n2(T_our, S_our, gdept, gdepw_int, e3w_int)          # (nj, ni, nlev-1)
    n2_nemo = _n2(T_nemo, S_nemo, gdept, gdepw_int, e3w_int)

    gd_i = 0.5 * (gdept[:-1] + gdept[1:])
    report = {}

    # AN INTERFACE IS ONLY USABLE IF BOTH CELLS TOUCHING IT ARE WET.
    #
    # The NEMO fields are read with fill values replaced by 0.0, so a cell
    # below the sea floor holds T=0, S=0 -- fresh and cold. N2 at the bottom
    # wet/dry interface therefore differences real water against fresh water,
    # which reads as violently UNSTABLE. Weighting by 0.5*(dV[k] + dV[k+1])
    # gives that interface HALF WEIGHT rather than none, so the garbage
    # survives into the mean.
    #
    # This is not hypothetical: before the mask, NEMO's volume-mean N2 in the
    # Weddell box went NEGATIVE from 190 m down to the bottom of the table --
    # a 350 m thick statically-unstable layer in a January monthly mean, which
    # no well-posed ocean produces. The Labrador box, being shallower and
    # having only 210 columns, was erratic at almost every depth. Both are the
    # sea floor showing through, and both are the SAME defect that made an
    # unmasked vertical-diffusion solve report a 414x variance ratio.
    iface_wet = wet[:-1] & wet[1:]                        # (nlev-1, nj, ni)
    dV_geom = e1t[None] * e2t[None] * e3t
    for nm, sel in regions:
        # Interface volume weights, this region only, wet interfaces only.
        dVi = 0.5 * (dV_geom[:-1] + dV_geom[1:]) * (iface_wet & sel[None])
        w = np.transpose(dVi, (1, 2, 0))
        n_col = int(sel.sum())
        n_if = int((iface_wet & sel[None]).sum())
        n_half = int(((wet[:-1] ^ wet[1:]) & sel[None]).sum())
        print(f"\n=== {nm}: {n_col} columns, {n_if} wet interfaces, "
              f"{n_half} sea-floor interfaces excluded ===")
        print("iface depth_m     N2_ours      N2_nemo    ratio   "
              "l_b_ours_m  l_b_withNemoN_m")
        rows = []
        for k in range(n2_our.shape[-1]):
            den = w[..., k].sum()
            if den <= 0:
                continue
            a_our = float((n2_our[..., k] * w[..., k]).sum() / den)
            a_nem = float((n2_nemo[..., k] * w[..., k]).sum() / den)
            e_k = float((tke[k] * dVi[k]).sum() / dVi[k].sum())
            def lb(n2, _e=e_k):
                return np.sqrt(2.0 * max(_e, 0.0)) / np.sqrt(n2) if n2 > 0 \
                    else np.inf
            print(f"{k:3d} {gd_i[k]:9.2f} {a_our:+.4e} {a_nem:+.4e} "
                  f"{(a_our / a_nem if a_nem != 0 else np.nan):7.3f} "
                  f"{lb(a_our):11.2f} {lb(a_nem):11.2f}")
            rows.append({"k": k, "depth_m": float(gd_i[k]),
                         "n2_ours": a_our, "n2_nemo": a_nem})
            if gd_i[k] > a.max_depth_m:
                break
        report[nm] = {"columns": n_col, "wet_interfaces": n_if,
                      "seafloor_excluded": n_half, "levels": rows}

    # A PARSED TABLE IS NOT DATA. Anything downstream -- a figure, a later
    # comparison -- reads this file, never a regex over the job log.
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(
            {"snapshot": a.snapshot, "nemo_gridt": a.nemo_gridt,
             "nemo_month": a.nemo_month, "sites": report}, indent=1))
        print(f"\n[report] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

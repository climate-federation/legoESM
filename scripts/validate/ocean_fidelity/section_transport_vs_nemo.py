"""Volume transport through a meridional section: ours vs NEMO, same formula.

WHY A SECTION AND NOT A STREAMFUNCTION. The barotropic streamfunction carries
an arbitrary reference constant, so comparing two models' psi fields means
first reproducing one model's integration convention inside the other -- a
place this campaign has already lost time to staggering mistakes. A section
integral has no reference constant: it is a sum of face transports, and both
models archive exactly that.

WHY DAY 30 IS ENOUGH. The barotropic mode adjusts in days, not decades, so at
day 30 it is already near its attractor. A large difference here is a solver
or topography defect rather than a transient -- unlike AMOC or the global
overturning, which are nowhere near equilibrium at day 30 and are deliberately
not touched by this probe.

WHAT IS COMPARED, and it is the same kind of quantity on both sides:
  ours   sum over depth of mass_flux_u * dy_u   [m2/s * m = m3/s]
  NEMO   sum over depth of uocetr_eff            [m3/s, archived directly]

TWO CONTROLS RUN BEFORE ANY TRANSPORT IS REPORTED, because both of the things
they check have produced confident wrong numbers in this campaign before.

  1. FILE-READING CONTROL. NEMO's uocetr_eff is the EFFECTIVE transport, which
     includes the eddy-induced (GM bolus) part; our mass_flux_u is the resolved
     flux. For a DEPTH-INTEGRATED transport that asymmetry should vanish, since
     the bolus streamfunction goes to zero at both the surface and the floor,
     so its depth integral is zero by construction. That is an assertion about
     NEMO's implementation, so it is MEASURED rather than assumed: the same
     depth-summed transport is rebuilt from uo * e3u * e2u and the two are
     compared. Agreement confirms both that the eddy part integrates out and
     that the file is being read correctly; disagreement means the comparison
     below is not like-for-like and the probe says so.

  2. STAGGERING CONTROL. Our u lives on a haloed (nj+1, ni+3) face array while
     NEMO's is the native (331, 360) frame. A face array does not slice the way
     a centre array does, and an off-by-one there is silent -- it looks exactly
     like a physics difference. So the land faces are cross-checked: every face
     NEMO's umask calls land must carry zero transport on our side too. The
     mismatch count is printed, and a nonzero count means the alignment is
     wrong and no number below can be trusted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SV = 1.0e-6          # m3/s -> Sv


def _m(ds, name):
    a = ds.variables[name][:]
    a = a.filled(np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)
    return np.asarray(a, dtype=np.float64).squeeze()


def _native_face(a):
    """Native frame for a u-FACE array.

    Centre arrays take [0:331, 1:361]. A u-FACE array does NOT: it takes
    [0:331, 2:362], one column further east, because face i sits between
    centres i and i+1 and our array carries an extra wrap column.

    That one column is not a detail. Using the centre slice left 1301 of
    NEMO's 55271 land faces carrying nonzero transport, which looks small
    enough to rationalise as a wet-domain difference -- and the transports it
    produced were physically plausible, right-signed and inside the expected
    range, which is exactly what makes this class of error dangerous. The
    offset scan settled it: this slice leaves EXACTLY ZERO land faces wet and
    every neighbour leaves 1268 or more. A sharp zero is the signature of a
    correct alignment; nothing else in the scan comes close.
    """
    return a[0:331, 2:362]


def _native_face_v(a):
    """Native frame for a v-FACE array.

    NOT assumed to be the u-face offset. A v-face sits between centres j and
    j+1 in the OTHER direction, and the haloed array's row padding need not
    mirror its column padding. The offset scan determines it the same way it
    determined the u one, and the land-mask control refuses to pass until it
    is right.
    """
    return a[1:332, 1:361]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--nemo-gridu", required=True)
    p.add_argument("--nemo-gridv", default=None,
                   help="required only when --zonal-section is used")
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--rec", type=int, required=True,
                   help="NEMO time record; the card derives it as DAY/5-1")
    p.add_argument("--section", action="append", required=True,
                   metavar="SPEC",
                   help="repeatable NAME:LON:LAT0,LAT1 (degrees east, and "
                        "LON may be negative). Use the equals form.")
    p.add_argument("--zonal-section", action="append", default=[],
                   metavar="SPEC",
                   help="repeatable NAME:LAT:LON0,LON1 -- a constant-LATITUDE "
                        "line, integrating NORTHWARD transport. The Florida "
                        "Current runs north through its strait, so a "
                        "meridional section cannot measure it. Equals form.")
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    import netCDF4 as nc

    z = dict(np.load(a.snapshot))
    mfu = np.asarray(z["mass_flux_u"], dtype=np.float64)   # (nj+1, ni+3, nlev)
    dyu = np.asarray(z["dy_u"], dtype=np.float64)          # (nj+1, ni+3)
    ours_face = _native_face(np.nansum(mfu, axis=-1) * dyu)   # m3/s, (331,360)

    du = nc.Dataset(a.nemo_gridu)
    try:
        nt = len(du.dimensions["time_counter"])
        if not 0 <= a.rec < nt:
            raise SystemExit(f"FATAL: rec {a.rec} outside 0..{nt - 1}")
        eff = _m(du, "uocetr_eff")[a.rec]                  # (75,331,360) m3/s
        uo = _m(du, "uo")[a.rec]
        e3u = _m(du, "e3u")[a.rec]
    finally:
        du.close()
    nemo_face = np.nansum(eff, axis=0)                     # (331,360) m3/s

    mm = nc.Dataset(a.mesh_mask)
    try:
        e2u = _m(mm, "e2u")[0:331, 1:361]
        umask = _m(mm, "umask")[:, 0:331, 1:361]
        mlat = _m(mm, "gphiu")[0:331, 1:361]
        mlon = _m(mm, "glamu")[0:331, 1:361] % 360.0
    finally:
        mm.close()

    # CONTROL 1 -- is the effective transport the same as the plain one, once
    # integrated over depth? If the GM bolus integrates to zero as it should,
    # these agree and our resolved flux is comparable to NEMO's effective one.
    rebuilt = np.nansum(uo * e3u, axis=0) * e2u
    both = np.isfinite(nemo_face) & np.isfinite(rebuilt) & (umask[0] > 0.5)
    num = float(np.abs(nemo_face[both] - rebuilt[both]).sum())
    den = float(np.abs(rebuilt[both]).sum())
    frac = num / den if den else float("nan")
    print(f"[control-1] depth-summed uocetr_eff vs uo*e3u*e2u: "
          f"relative difference {frac:.3e} over {int(both.sum())} faces")
    if frac > 0.02:
        print("[control-1] FAIL: the effective transport is NOT the plain "
              "transport once depth-integrated, so the eddy-induced part does "
              "NOT integrate out and our resolved flux is not comparable to "
              "it. Numbers below are NOT like-for-like.")

    # CONTROL 2 -- staggering. Every face NEMO calls land must be dry on ours.
    nemo_land = umask.max(axis=0) < 0.5
    ours_wet_on_land = int(np.count_nonzero(
        np.abs(np.nan_to_num(ours_face))[nemo_land] > 0.0))
    print(f"[control-2] faces NEMO calls land: {int(nemo_land.sum())}; "
          f"nonzero on ours: {ours_wet_on_land}")
    if ours_wet_on_land:
        # A nonzero count has TWO possible causes and they demand opposite
        # responses, so it is diagnosed rather than declared. An off-by-one
        # slice misaligns every coastline at once and should look catastrophic;
        # a genuine WET-DOMAIN DIFFERENCE (cells we call ocean and NEMO calls
        # land, a class this campaign has already found once in the closed
        # seas) leaves the alignment correct and shows up as a small residue.
        #
        # The discriminator: scan neighbouring slices. If the chosen offset is
        # a SHARP MINIMUM, the alignment is right and the residue is domain.
        # If a neighbour is better, the slice is simply wrong.
        full = np.nansum(mfu, axis=-1) * dyu
        print("[control-2] offset scan (di, dj -> faces nonzero on NEMO land):")
        best = None
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                cand = full[0 + dj:331 + dj, 2 + di:362 + di]
                if cand.shape != nemo_land.shape:
                    continue
                n = int(np.count_nonzero(
                    np.abs(np.nan_to_num(cand))[nemo_land] > 0.0))
                print(f"             ({di:+d},{dj:+d}) -> {n:7d}"
                      + ("   <- in use" if (di == 0 and dj == 0) else ""))
                if best is None or n < best[0]:
                    best = (n, di, dj)
        if best and (best[1], best[2]) != (0, 0):
            print(f"[control-2] FAIL: offset ({best[1]:+d},{best[2]:+d}) is "
                  f"better ({best[0]} vs {ours_wet_on_land}). The slice is "
                  "WRONG and every transport below compares different faces.")
        else:
            print("[control-2] alignment CONFIRMED: the slice in use is the "
                  "best of its neighbours, so the residue is a WET-DOMAIN "
                  "difference (cells we call ocean and NEMO calls land), not "
                  "a staggering error. Transports stand; the residue is its "
                  "own finding and is localised below.")
            jj, ii = np.nonzero(np.abs(np.nan_to_num(ours_face)) * nemo_land)
            mag = np.abs(ours_face[jj, ii])
            order = np.argsort(mag)[::-1][:8]
            print("[control-2] largest offenders (lat, lon, |transport| Sv):")
            for o in order:
                print(f"             {mlat[jj[o], ii[o]]:+7.2f} "
                      f"{mlon[jj[o], ii[o]]:7.2f}  {mag[o] * SV:.4f}")
            print(f"[control-2] residue carries "
                  f"{float(mag.sum()) * SV:.3f} Sv of |transport| in total")

    out = {"rec": a.rec, "control_rel_diff_eff_vs_plain": frac,
           "control_ours_nonzero_on_nemo_land": ours_wet_on_land,
           "sections": {}}
    for spec in a.section:
        nm, lonspec, latspec = spec.split(":")
        lon0 = float(lonspec) % 360.0
        la0, la1 = (float(v) for v in latspec.split(","))
        rows = (mlat >= la0) & (mlat <= la1)
        if not rows.any():
            raise SystemExit(f"FATAL: section {nm!r} spans no rows")
        # One i-column: a meridional section must be a single face column or
        # it is not a section. Pick the column whose longitude is closest to
        # the request WITHIN the latitude band, and report what was picked.
        dlon = np.abs((mlon - lon0 + 180.0) % 360.0 - 180.0)
        colscore = np.where(rows, dlon, np.nan)
        with np.errstate(invalid="ignore"):
            i = int(np.nanargmin(np.nanmean(colscore, axis=0)))
        sel = rows[:, i]
        wet = sel & (umask[:, :, i].max(axis=0) > 0.5)
        t_ours = float(np.nansum(ours_face[sel, i]) * SV)
        t_nemo = float(np.nansum(nemo_face[sel, i]) * SV)
        print(f"\n=== {nm}: i={i}, lon {mlon[sel, i].min():.2f}-"
              f"{mlon[sel, i].max():.2f}E, lat {mlat[sel, i].min():.2f} to "
              f"{mlat[sel, i].max():.2f}, {int(sel.sum())} rows "
              f"({int(wet.sum())} wet) ===")
        print(f"  transport ours {t_ours:+8.2f} Sv   NEMO {t_nemo:+8.2f} Sv   "
              f"difference {t_ours - t_nemo:+8.2f} Sv")
        out["sections"][nm] = {
            "i": i, "rows": int(sel.sum()), "wet_rows": int(wet.sum()),
            "lat0": float(mlat[sel, i].min()), "lat1": float(mlat[sel, i].max()),
            "lon_mean": float(mlon[sel, i].mean()),
            "transport_ours_Sv": t_ours, "transport_nemo_Sv": t_nemo,
            "ours_per_row_Sv": (ours_face[sel, i] * SV).tolist(),
            "nemo_per_row_Sv": (nemo_face[sel, i] * SV).tolist(),
            "lat_per_row": mlat[sel, i].tolist(),
        }

    if a.zonal_section:
        if not a.nemo_gridv:
            raise SystemExit("FATAL: --zonal-section needs --nemo-gridv")
        mfv = np.asarray(z["mass_flux_v"], dtype=np.float64)
        dxv = np.asarray(z["dx_v"], dtype=np.float64)
        full_v = np.nansum(mfv, axis=-1) * dxv
        ours_v = _native_face_v(full_v)
        dv = nc.Dataset(a.nemo_gridv)
        try:
            nemo_v = np.nansum(_m(dv, "vocetr_eff")[a.rec], axis=0)
        finally:
            dv.close()
        mv = nc.Dataset(a.mesh_mask)
        try:
            vmask = _m(mv, "vmask")[:, 0:331, 1:361]
            vlat = _m(mv, "gphiv")[0:331, 1:361]
            vlon = _m(mv, "glamv")[0:331, 1:361] % 360.0
        finally:
            mv.close()

        # The v-face slice gets its OWN alignment proof. Reusing the u answer
        # would be exactly the assumption that cost a retraction earlier in
        # this probe's life.
        v_land = vmask.max(axis=0) < 0.5
        nzero = int(np.count_nonzero(np.abs(np.nan_to_num(ours_v))[v_land] > 0))
        print(f"\n[control-2v] faces NEMO calls land: {int(v_land.sum())}; "
              f"nonzero on ours: {nzero}")
        if nzero:
            print("[control-2v] offset scan (di, dj -> nonzero on land):")
            best = None
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    cand = full_v[1 + dj:332 + dj, 1 + di:361 + di]
                    if cand.shape != v_land.shape:
                        continue
                    n = int(np.count_nonzero(
                        np.abs(np.nan_to_num(cand))[v_land] > 0.0))
                    print(f"              ({di:+d},{dj:+d}) -> {n:7d}"
                          + ("   <- in use" if (di == 0 and dj == 0) else ""))
                    if best is None or n < best[0]:
                        best = (n, di, dj)
            print(f"[control-2v] FAIL: best offset ({best[1]:+d},{best[2]:+d}) "
                  f"gives {best[0]}. Zonal transports below are NOT valid.")

        for spec in a.zonal_section:
            nm, latspec, lonspec = spec.split(":")
            lat0 = float(latspec)
            lo0, lo1 = (float(v) % 360.0 for v in lonspec.split(","))
            j = int(np.argmin(np.abs(np.nanmean(vlat, axis=1) - lat0)))
            cols = ((vlon[j] >= lo0) | (vlon[j] <= lo1)) if lo0 > lo1 else \
                   ((vlon[j] >= lo0) & (vlon[j] <= lo1))
            if not cols.any():
                raise SystemExit(f"FATAL: zonal section {nm!r} spans no cells")
            wetc = cols & (vmask[:, j, :].max(axis=0) > 0.5)
            t_o = float(np.nansum(ours_v[j, cols]) * SV)
            t_n = float(np.nansum(nemo_v[j, cols]) * SV)
            print(f"\n=== {nm}: j={j}, lat {vlat[j, cols].mean():.2f}, lon "
                  f"{vlon[j, cols].min():.2f}-{vlon[j, cols].max():.2f}E, "
                  f"{int(cols.sum())} cells ({int(wetc.sum())} wet) ===")
            print(f"  northward transport ours {t_o:+8.2f} Sv   NEMO "
                  f"{t_n:+8.2f} Sv   difference {t_o - t_n:+8.2f} Sv")
            out["sections"][nm] = {
                "orientation": "zonal", "j": j,
                "cells": int(cols.sum()), "wet_cells": int(wetc.sum()),
                "lat": float(vlat[j, cols].mean()),
                "transport_ours_Sv": t_o, "transport_nemo_Sv": t_n,
                "ours_per_row_Sv": (ours_v[j, cols] * SV).tolist(),
                "nemo_per_row_Sv": (nemo_v[j, cols] * SV).tolist(),
                "lat_per_row": vlon[j, cols].tolist(),
            }

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(out, indent=1))
        print(f"\n[report] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Equatorial SST bias against NEMO's five-day mean, per snapshot CLOCK PHASE.

Every equatorial SST comparison in this campaign pairs OUR 00:00 UTC snapshot
(13:20-18:00 local at 200-270E, the afternoon of the --dm2dc diurnal warm
layer) with NEMO's FIVE-DAY MEAN. This prints the same box biases for
snapshots taken at several UTC phases of one day, through the scorer's own
loaders and regrid (compare_omip_nemo), so the phase dependence of the
"warm lens" is measured rather than assumed. Also prints our own SST minus
our ~10 m temperature per phase: the diurnal warm layer's amplitude.

The three-way scorer refuses sub-daily snapshots by design (its oracle-record
gate); this probe exists only for the phase question and takes no verdict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0]))  # scripts/validate
from compare_omip_nemo import _BOXES, _load_legoesm, _load_nemo, regrid_curv_to_latlon  # noqa: E402

# The scorer's own box definitions (compare_omip_nemo._BOXES), not a copy.
BOXES = tuple((short, *_BOXES[key]) for short, key in
              (("nino3", "nino3_5S5N_150W90W"), ("nino34", "nino34_5S5N_170W120W"),
               ("eq_pacific", "eq_pacific_2S2N")))


def _box(tgt_lat, tgt_lon, lat0, lat1, lon0, lon1):
    la = (tgt_lat >= lat0) & (tgt_lat <= lat1)
    lo = (tgt_lon >= lon0) & (tgt_lon <= lon1)
    return la[:, None] & lo[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--res-deg", type=float, default=1.0)
    ap.add_argument("--sub-depth-m", type=float, default=10.0)
    ap.add_argument("--use-mean-fields", action="store_true",
                    help="read T_mean (the driver's --state-accumulate window "
                         "mean) instead of the instantaneous T when present")
    ap.add_argument("--mean", action="store_true",
                    help="also score the MEAN of all given snapshots (window-"
                         "matched against NEMO's 5-day mean when the snapshots "
                         "tile that window)")
    a = ap.parse_args()
    tgt_lat = -90.0 + a.res_deg / 2 + a.res_deg * np.arange(int(180 / a.res_deg))
    tgt_lon = a.res_deg / 2 + a.res_deg * np.arange(int(360 / a.res_deg))
    N = _load_nemo(a.nemo_gridt, a.rec)
    sstN, ocN = regrid_curv_to_latlon(N["sst"], N["lat"], N["lon"], N["mask"], tgt_lat, tgt_lon)
    # The oracle record's temporal contract, READ from the file, not assumed.
    import xarray as xr
    dsN = xr.open_dataset(a.nemo_gridt, decode_times=False)
    _tos = dsN["tos"]
    _tb = dsN["time_counter_bounds"].isel(time_counter=a.rec).values
    _span_d = float(_tb[1] - _tb[0]) / 86400.0
    if _tos.attrs.get("online_operation") != "average" or abs(_span_d - 5.0) > 1e-6:
        raise SystemExit(f"NEMO tos is not a 5-day average: op={_tos.attrs.get('online_operation')} "
                         f"span={_span_d} d -- the window-matching premise fails")
    if dsN["to"].attrs.get("online_operation") != "average":
        raise SystemExit("NEMO to is not an average")
    print(f"NEMO tos: online_operation={_tos.attrs.get('online_operation')} "
          f"interval_operation={_tos.attrs.get('interval_operation')} "
          f"interval_write={_tos.attrs.get('interval_write')}; record {a.rec} bounds "
          f"{(_tb - _tb[0] + (float(_tb[0]) % 86400)) / 86400.0} days rel. to record-0 start "
          f"(raw {_tb[0]:.0f}..{_tb[1]:.0f} s); NEMO temperature is CONSERVATIVE (TEOS-10), "
          "ours potential -- both started from the same IC numbers, and a fixed offset "
          "cancels between phases; it does NOT cancel in the absolute bias (unconverted here).")
    # NEMO's own 0-10 m stratification from its 5-day-mean 3-D field (to = the
    # model's top-level temperature at deptht ~9.82 m), same regrid.
    kN = int(np.argmin(np.abs(dsN["deptht"].values - a.sub_depth_m)))
    toN = np.asarray(dsN["to"].isel(time_counter=a.rec, deptht=kN).values, dtype=float)
    toN = np.where(np.isfinite(toN) & (np.abs(toN) < 1e3), toN, np.nan)
    subN, ocN_sub = regrid_curv_to_latlon(np.nan_to_num(toN), N["lat"], N["lon"], N["mask"] * np.isfinite(toN), tgt_lat, tgt_lon)
    print(f"NEMO sub-surface level {dsN['deptht'].values[kN]:.2f} m; NEMO tos - T_sub CONTRAST (background stratification included) box means: "
          + "  ".join(f"{b[0]} {float(np.sum((np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :])[m := ((ocN > 0.5) & (ocN_sub > 0.5) & np.isfinite(sstN) & np.isfinite(subN) & _box(tgt_lat, tgt_lon, *b[1:]))] * (sstN[m] - subN[m])) / np.sum((np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :])[m])):+.3f}" for b in BOXES))
    area = np.cos(np.deg2rad(tgt_lat))[:, None] * np.ones_like(tgt_lon)[None, :]
    print(f"NEMO {Path(a.nemo_gridt).name} record {a.rec}; target {a.res_deg} deg")
    hdr = f"{'snapshot':28s} {'day':>6s} " + "".join(f"{b[0]+' bias':>14s}" for b in BOXES) \
        + "".join(f"{b[0]+' SST-T'+str(int(a.sub_depth_m))+'ctr':>19s}" for b in BOXES) + f"{'n nino3':>9s}"
    print(hdr)
    # ONE declared support for every row (codex): ocean on both grids, NEMO
    # finite at the surface and the sub-surface level.  Each row must be
    # finite on all of it -- a row that is not is a blown-up state, not a
    # smaller box -- so no row is scored on its own private support.
    support = None
    members = []          # (day, weight_s or None, sstL, subL, label)
    for p in a.snapshots:
        L = _load_legoesm(p)
        s = np.load(p)
        z = np.asarray(s["z_center_ref"])
        k = int(np.argmin(np.abs(z - a.sub_depth_m)))
        if a.use_mean_fields and "T_mean" not in s.files:
            raise SystemExit(f"--use-mean-fields but {p} has no T_mean "
                             "(run the driver with --state-accumulate); no silent fallback")
        _T = np.asarray(s["T_mean" if a.use_mean_fields else "T"])
        # tos is a PLAIN mean (surface from T_mean); to is @toce_e3t/@e3t, so
        # the sub-surface level comes from T_mean_hw with its window h_mean.
        _Tsub = np.asarray(s["T_mean_hw"]) if a.use_mean_fields else _T
        _hsub = np.asarray(s["h_mean"])[..., k] if a.use_mean_fields else None
        if "time_days" not in s.files or not np.isfinite(float(s["time_days"])):
            raise SystemExit(f"{p}: no finite time_days; cannot place it in a window")
        day = float(s["time_days"])
        wsec = float(s["flux_mean_window_s"]) if a.use_mean_fields else None
        if a.use_mean_fields and not (np.isfinite(wsec) and wsec > 0):
            raise SystemExit(f"{p}: T_mean without a positive flux_mean_window_s")
        sstL, ocL = regrid_curv_to_latlon(_T[..., 0], L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
        subL, ocL_sub = regrid_curv_to_latlon(np.nan_to_num(_Tsub[..., k]), L["lat"], L["lon"],
                                              L["mask"] * np.isfinite(_Tsub[..., k]), tgt_lat, tgt_lon)
        hL = (regrid_curv_to_latlon(_hsub, L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)[0]
              if _hsub is not None else np.ones_like(subL))
        if support is None:
            # Coverage masks of BOTH regrids on BOTH sides (the regridder
            # extrapolates finite values outside its radius, codex).
            support = ((ocL > 0.5) & (ocN > 0.5) & (ocN_sub > 0.5) & (ocL_sub > 0.5)
                       & np.isfinite(sstN) & np.isfinite(subN))
            for name, la0, la1, lo0, lo1 in BOXES:
                bm = _box(tgt_lat, tgt_lon, la0, la1, lo0, lo1)
                if not (support & bm).any():
                    raise SystemExit(f"empty box {name}")
                print(f"support {name}: {100 * area[support & bm].sum() / area[bm & (ocN > 0.5)].sum():.1f}% "
                      f"of NEMO-ocean box area retained")
        bad = support & ~(np.isfinite(sstL) & np.isfinite(subL) & (ocL > 0.5) & (ocL_sub > 0.5))
        if bad.any():
            raise SystemExit(f"{p}: {int(bad.sum())} support cells non-finite/land in this row")
        members.append((day, wsec, sstL, subL,
                        Path(p).name + (" [T_mean]" if a.use_mean_fields else ""), hL))

    rows = list(members)
    if a.mean:
        days = [m[0] for m in members]
        if len(set(np.round(days, 4))) != len(days):
            raise SystemExit(f"duplicate snapshot days in the mean: {days}")
        order = np.argsort(days)
        if a.use_mean_fields:
            # Window means: duration-weighted, windows must tile [start, end]
            # with no gap or overlap (each window ends at its snapshot day).
            wts = np.array([members[o][1] for o in order]) / 86400.0
            ends = np.array([members[o][0] for o in order])
            starts = ends - wts
            gaps = starts[1:] - ends[:-1]
            if np.any(np.abs(gaps) > 1e-6):
                raise SystemExit(f"T_mean windows do not tile the interval: gaps(days)={gaps}")
            # ...and the tiled interval must BE the oracle record's window
            # (both endpoints; run day 0 = NEMO record-0 start).
            rec0, rec1 = 5.0 * a.rec, 5.0 * a.rec + 5.0
            if abs(starts[0] - rec0) > 1e-6 or abs(ends[-1] - rec1) > 1e-6:
                raise SystemExit(f"T_mean windows span d{starts[0]:.3f}-{ends[-1]:.3f} but NEMO "
                                 f"record {a.rec} is d{rec0:.0f}-{rec1:.0f}; not window-matched")
            label = f"WINDOW MEAN d{starts[0]:.2f}-{ends[-1]:.2f} ({len(order)} windows)"
        else:
            wts = np.ones(len(order))
            label = (f"SAMPLE MEAN of {len(order)} snaps d{min(days):.2f}-{max(days):.2f} "
                     "(equal weight; NOT a window mean)")
        sstM = sum(w * members[o][2] for w, o in zip(wts, order)) / wts.sum()
        # Sub-surface: duration x thickness weights (the hw mean of a union
        # of windows is Sum(T h dt)/Sum(h dt)); hL == 1 for instantaneous rows.
        subM = (sum(w * members[o][5] * members[o][3] for w, o in zip(wts, order))
                / sum(w * members[o][5] for w, o in zip(wts, order)))
        rows.append((float("nan"), None, sstM, subM, label, None))

    for day, _w, sstL, subL, label, _h in rows:
        row = f"{label:28s} {day:6.2f} "
        biases, dl, n3 = [], [], 0
        for name, la0, la1, lo0, lo1 in BOXES:
            m = support & _box(tgt_lat, tgt_lon, la0, la1, lo0, lo1)
            w = area[m]
            biases.append(float(np.sum(w * (sstL[m] - sstN[m])) / np.sum(w)))
            dl.append(float(np.sum(w * (sstL[m] - subL[m])) / np.sum(w)))
            if name == "nino3":
                n3 = int(m.sum())
        row += "".join(f"{b:+14.3f}" for b in biases) + "".join(f"{d:+19.3f}" for d in dl) + f"{n3:9d}"
        print(row)
    print(f"(z level used for the sub-surface temperature: {z[k]:.2f} m)")


if __name__ == "__main__":
    main()

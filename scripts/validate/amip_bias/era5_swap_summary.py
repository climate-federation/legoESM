#!/usr/bin/env python3
"""Summarise ERA5-swap radiation captures (nh_surface_replay.py capture --era5-pl).

For each capture and region: clear-sky surface downward LW of the model state
and with ERA5 temperature, humidity, or both swapped in; the round-trip
controls (area-weighted mean |change|, must be finite and <= 1 W/m2, recomputed
here so captures written before the gate existed are held to it too); and the
ERA5-minus-model temperature / humidity averaged over the lowest model layers
and a mid-troposphere band.  Weights: cell area over cells with f_land > 0.5
(land) or < 0.05 (ocean).

Usage: era5_swap_summary.py CAP.npz [CAP.npz ...]
"""
import json
import sys

import numpy as np

REGIONS = {"45-70N land": lambda la, fl: (la >= 45) & (la <= 70) & (fl > 0.5),
           "45-70N ocean": lambda la, fl: (la >= 45) & (la <= 70) & (fl < 0.05),
           "tropics": lambda la, fl: np.abs(la) <= 30}


def main(files):
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if "rad_clr_e5T_lw_dn_sfc" not in z.files:
            raise SystemExit(f"{f}: not an ERA5-swap capture")
        lat, fl, area = z["lat"], z["f_land"], z["area"]
        pf = z["e5_p_full"]
        dT, dq = z["e5_T_minus_model"], z["e5_q_minus_model"]
        base = z["rad_inst_clr_lw_dn_sfc"]
        print(f"\n{f.split('/')[-1]}  (run {meta['run']} day {meta['day']}, "
              f"control_fail in file: {meta.get('control_fail', 'n/a')})")
        for name, sel in REGIONS.items():
            m = sel(lat, fl)
            w = area * m
            mean = lambda x: float((x * w).sum() / w.sum())
            out = {k: mean(z[f"rad_{k}_lw_dn_sfc"]) for k in
                   ("inst_clr", "clr_e5T", "clr_e5q", "clr_e5Tq")}
            ctl = {}
            for k in ("clr_selfT", "clr_selfq"):
                d = z[f"rad_{k}_lw_dn_sfc"] - base
                ad = float((np.abs(d) * w).sum() / w.sum())
                ctl[k] = ad
                if not np.isfinite(ad) or ad > 1.0:
                    raise SystemExit(f"FATAL {f} {name}: control {k} mean |dDLW| {ad}")
            low = pf > 0.85e5                     # layers below 850 hPa
            mid = (pf > 4.0e4) & (pf < 7.0e4)
            lay = lambda x, mk: float(((x * mk).sum(1) / np.maximum(mk.sum(1), 1) * w).sum() / w.sum())
            beyond = float((z["e5_beyond_nodes"].mean(1) * w).sum() / w.sum())
            print(f"  {name}: clear-sky DLW model {out['inst_clr']:.1f} | ERA5 T {out['clr_e5T']:.1f} "
                  f"({out['clr_e5T'] - out['inst_clr']:+.1f}) | ERA5 q {out['clr_e5q']:.1f} "
                  f"({out['clr_e5q'] - out['inst_clr']:+.1f}) | both {out['clr_e5Tq']:.1f} "
                  f"({out['clr_e5Tq'] - out['inst_clr']:+.1f}) W/m2")
            print(f"     ERA5-model T below 850 hPa {lay(dT, low):+.2f} K, 400-700 hPa {lay(dT, mid):+.2f} K; "
                  f"q below 850 {1e3 * lay(dq, low):+.3f} g/kg, 400-700 {1e3 * lay(dq, mid):+.3f} g/kg; "
                  f"controls |dDLW| T {ctl['clr_selfT']:.2f} q {ctl['clr_selfq']:.2f}; "
                  f"layers beyond ERA5 nodes {100 * beyond:.1f} %")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

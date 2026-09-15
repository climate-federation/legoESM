#!/usr/bin/env python3
"""Arm-minus-control maps, with the arm's bias against the reference alongside.

One row per variable, four panels: control, arm, arm - control, arm - reference
(the ESMValTool reference bias_maps.py declares for that variable, on the same
calendar months; blank when the variable has none, e.g. hfls). Both runs must
publish the same months on the same grid.

    pair_maps.py --ctl pa_ctl --arm wv_sfcrain --out /path/dir
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
import bias_maps as bm  # noqa: E402

SPEC = {  # var: (scale, units, cmap for the fields, symmetric bias limit)
    "prw": (1.0, "kg/m2", "viridis", 8.0),
    "pr": (86400.0, "mm/day", "YlGnBu", 4.0),
    "evspsbl": (86400.0, "mm/day", "YlGnBu", 2.0),
    "hfls": (1.0, "W/m2", "YlOrRd", 40.0),
    "clt": (1.0, "%", "Greys_r", 40.0),
    "rsut": (1.0, "W/m2", "magma", 40.0),
    "tas": (1.0, "K", "coolwarm", 3.0),
}


def _load(run, var):
    d = rb._load_model(run, var)
    if d is None:
        return None
    return (np.asarray(d[var]).mean(axis=0), np.asarray(d.lat), np.asarray(d.lon),
            rb._month_labels(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--vars", nargs="+", default=list(SPEC))
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    rows = []
    for var in args.vars:
        c, a = _load(args.ctl, var), _load(args.arm, var)
        if c is None or a is None:
            print(f"  {var}: missing in one run -- skipped")
            continue
        (cv, lat, lon, months), (av, alat, alon, amonths) = c, a
        if cv.shape != av.shape or months != amonths:
            raise SystemExit(f"{var}: runs differ in grid or months ({months} vs {amonths})")
        scale, units, cmap, blim = SPEC[var]
        ref = bm._ref_clim(var, months, lat, lon, bm.SPEC[var][0]) if var in bm.SPEC else None
        rows.append((var, cv * scale, av * scale, None if ref is None else ref * scale,
                     lat, lon, units, cmap, blim))
    if not rows:
        raise SystemExit("nothing to plot")
    fig, axes = plt.subplots(len(rows), 4, figsize=(20, 2.9 * len(rows)),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    for i, (var, cv, av, ref, lat, lon, units, cmap, blim) in enumerate(rows):
        lo = float(min(np.nanpercentile(cv, 2), np.nanpercentile(av, 2)))
        hi = float(max(np.nanpercentile(cv, 98), np.nanpercentile(av, 98)))
        m0 = bm._panel(axes[i, 0], lon, lat, cv,
                       f"{args.ctl} {var}  global {bm._gm(cv, lat, lon):.3g} {units}", cmap, lo, hi, units)
        bm._panel(axes[i, 1], lon, lat, av,
                  f"{args.arm} {var}  global {bm._gm(av, lat, lon):.3g} {units}", cmap, lo, hi, units)
        plt.colorbar(m0, ax=axes[i, :2].tolist(), shrink=0.8, pad=0.01)
        d = av - cv
        m2 = bm._panel(axes[i, 2], lon, lat, d,
                       f"arm - ctl  global {bm._gm(d, lat, lon):+.3g}  tropics "
                       f"{rb.region_mean(d, lat, lon, (-20, 20, 0, 360)):+.3g} {units}",
                       "RdBu_r", -blim, blim, units)
        plt.colorbar(m2, ax=axes[i, 2], shrink=0.8, pad=0.01)
        if ref is not None:
            b = av - ref
            m3 = bm._panel(axes[i, 3], lon, lat, b,
                           f"arm - obs  global {bm._gm(b, lat, lon):+.3g}  tropics "
                           f"{rb.region_mean(b, lat, lon, (-20, 20, 0, 360)):+.3g} {units}",
                           "RdBu_r", -blim, blim, units)
            plt.colorbar(m3, ax=axes[i, 3], shrink=0.8, pad=0.01)
        else:
            axes[i, 3].set_title("no reference", fontsize=8)
            axes[i, 3].set_global()
    out = f"{args.out}/pair_{args.arm}_vs_{args.ctl}.png"
    fig.suptitle(f"{args.arm} vs {args.ctl} (published months, 4th column = arm minus reference)", fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

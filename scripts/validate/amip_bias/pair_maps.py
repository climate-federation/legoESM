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

SPEC = {  # var: (scale, units, cmap for the fields, symmetric bias limit, plain name)
    "prw": (1.0, "kg/m2", "viridis", 8.0, "column water vapour"),
    "pr": (86400.0, "mm/day", "YlGnBu", 4.0, "precipitation"),
    "evspsbl": (86400.0, "mm/day", "YlGnBu", 2.0, "surface evaporation"),
    "hfls": (1.0, "W/m2", "YlOrRd", 40.0, "latent heat flux"),
    "clt": (1.0, "%", "Greys_r", 40.0, "total cloud cover"),
    "rsut": (1.0, "W/m2", "magma", 40.0, "reflected sunlight (top of atmosphere)"),
    "tas": (1.0, "K", "coolwarm", 3.0, "2 m air temperature"),
}
REF_NAME = {"prw": "ERA5", "pr": "GPCP", "evspsbl": "ERA5", "clt": "ESACCI-CLOUD",
            "rsut": "CERES-EBAF", "tas": "ERA5"}


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
    ap.add_argument("--ctl-label", default=None, help="plain-language name of the control")
    ap.add_argument("--arm-label", default=None, help="plain-language name of the arm")
    ap.add_argument("--title", default=None)
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
        if (cv.shape != av.shape or months != amonths
                or not np.allclose(lat, alat) or not np.allclose(lon, alon)):
            raise SystemExit(f"{var}: runs differ in grid or months ({months} vs {amonths})")
        scale, units, cmap, blim, name = SPEC[var]
        ref = bm._ref_clim(var, months, lat, lon, bm.SPEC[var][0]) if var in bm.SPEC else None
        rows.append((var, cv * scale, av * scale, None if ref is None else ref * scale,
                     lat, lon, units, cmap, blim, name))
    if not rows:
        raise SystemExit("nothing to plot")
    fig, axes = plt.subplots(len(rows), 4, figsize=(20, 3.2 * len(rows)),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    ctl_label = args.ctl_label or args.ctl
    arm_label = args.arm_label or args.arm
    for i, (var, cv, av, ref, lat, lon, units, cmap, blim, name) in enumerate(rows):
        lo = float(min(np.nanpercentile(cv, 2), np.nanpercentile(av, 2)))
        hi = float(max(np.nanpercentile(cv, 98), np.nanpercentile(av, 98)))
        trop = (-20, 20, 0, 360)
        m0 = bm._panel(axes[i, 0], lon, lat, cv,
                       f"CONTROL: {name} [{units}]\nglobal mean {bm._gm(cv, lat, lon):.3g}",
                       cmap, lo, hi, units)
        bm._panel(axes[i, 1], lon, lat, av,
                  f"ARM: {name} [{units}]\nglobal mean {bm._gm(av, lat, lon):.3g}",
                  cmap, lo, hi, units)
        plt.colorbar(m0, ax=axes[i, :2].tolist(), shrink=0.8, pad=0.01)
        d = av - cv
        m2 = bm._panel(axes[i, 2], lon, lat, d,
                       f"CHANGE: arm minus control [{units}]\nglobal {bm._gm(d, lat, lon):+.3g}, "
                       f"tropics 20S-20N {rb.region_mean(d, lat, lon, trop):+.3g}",
                       "RdBu_r", -blim, blim, units)
        plt.colorbar(m2, ax=axes[i, 2], shrink=0.8, pad=0.01)
        if ref is not None:
            b = av - ref
            m3 = bm._panel(axes[i, 3], lon, lat, b,
                           f"REMAINING BIAS: arm minus {REF_NAME.get(var, 'reference')} [{units}]\n"
                           f"global {bm._gm(b, lat, lon):+.3g}, tropics {rb.region_mean(b, lat, lon, trop):+.3g}",
                           "RdBu_r", -blim, blim, units)
            plt.colorbar(m3, ax=axes[i, 3], shrink=0.8, pad=0.01)
        else:
            axes[i, 3].set_title(f"{name}: no observational reference", fontsize=8)
            axes[i, 3].set_global()
    out = f"{args.out}/pair_{args.arm}_vs_{args.ctl}.png"
    head = (args.title + "\n" if args.title else "") + (
        f"CONTROL = {ctl_label}\nARM = {arm_label}\n"
        "columns: control | arm | change (arm minus control) | remaining bias (arm minus observations); "
        "red = wetter / warmer / cloudier / brighter")
    fig.suptitle(head, fontsize=11, y=0.995, va="top")
    fig.subplots_adjust(top=0.93, hspace=0.35)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

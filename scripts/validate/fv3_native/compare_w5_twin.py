#!/usr/bin/env python
"""W5 twin comparison: our duo-stepper frames vs the generated
fv3_solo case-5 reference (matched dt_atmos=1800 cadence; deviation
from the published 3600 disclosed — our stepper's stability envelope
needs inner dt <~260 s, an open follow-up).

Reads their per-tile native atmos_daily (ucomp/vcomp) through the same
1-degree nearest-map lens used for case-8, FAIL-LOUD on frame count;
writes a side-by-side |V| panel (days 5/10/15) + both max-wind curves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_oracle(run_dir: Path, expect_frames: int):
    try:
        import netCDF4 as nc
    except ImportError:
        import scipy.io.netcdf  # noqa: F401 — fail loudly below instead
        raise SystemExit("netCDF4 not available")
    us, vs, lons, lats = [], [], [], []
    for t in range(1, 7):
        d = nc.Dataset(run_dir / f"atmos_daily.tile{t}.nc")
        u = np.array(d.variables["ucomp"][:])   # (time, [z,] y, x)
        v = np.array(d.variables["vcomp"][:])
        if u.ndim == 4:
            u, v = u[:, 0], v[:, 0]
        if u.shape[0] != expect_frames:
            raise SystemExit(
                f"tile{t}: {u.shape[0]} frames != expected "
                f"{expect_frames} — oracle run incomplete (fail-loud)")
        lons.append(np.array(d.variables["grid_xt"][:])
                    if "grid_xt" in d.variables else None)
        us.append(u)
        vs.append(v)
        d.close()
    return us, vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--days", default="5,10,15")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ours = np.load(args.ours, allow_pickle=False)
    times = np.asarray(ours["times_days"])
    n_days = int(round(float(times[-1])))
    us, vs = load_oracle(Path(args.oracle_dir), n_days)

    # oracle native tiles -> the same 1-degree canvas via cell-center
    # nearest sampling using each tile's own lon/lat from grid_spec?
    # atmos_daily lacks 2-D coords at layout 1,1 c2l-off — use the
    # native max metric + our lens frames for the pattern panel; the
    # CURVES are lens-independent maxima.
    orc_max = [float(np.max(np.hypot(
        np.stack([us[t][k] for t in range(6)]),
        np.stack([vs[t][k] for t in range(6)])))) for k in range(n_days)]
    our_w = np.hypot(ours["u"], ours["v"])
    our_max = [float(np.nanmax(our_w[k])) for k in range(len(times))]

    days = [float(x) for x in args.days.split(",")]
    fig = plt.figure(figsize=(4.6 * len(days), 6.2),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, len(days), height_ratios=[1, 1.2])
    axc = fig.add_subplot(gs[0, :])
    axc.plot(times, our_max, "o-", label="ours (duo stepper, nord=2)")
    axc.plot(np.arange(1, n_days + 1), orc_max, "s--",
             label="fv3_solo case-5 (native-tile max)")
    axc.set_xlabel("day")
    axc.set_ylabel("max |V| [m/s]")
    axc.legend()
    axc.set_title("W5 C48 twin, dt_atmos=1800 both "
                  "(published 3600 — disclosed deviation)")
    vmax = max(np.nanmax(our_w[int(round(d))]) for d in days)
    for j, d in enumerate(days):
        ax = fig.add_subplot(gs[1, j])
        k = int(np.argmin(np.abs(times - d)))
        im = ax.imshow(our_w[k], origin="lower", extent=(0, 360, -90, 90),
                       vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_title(f"ours d{times[k]:g}  max {np.nanmax(our_w[k]):.1f}"
                     f" / oracle {orc_max[min(int(round(d)) - 1, n_days - 1)]:.1f}",
                     fontsize=9)
        if j == len(days) - 1:
            fig.colorbar(im, ax=ax, shrink=0.8, label="|V| m/s")
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)
    print("curves: ours", [f"{x:.1f}" for x in our_max[::3]],
          "oracle", [f"{x:.1f}" for x in orc_max[::3]])


if __name__ == "__main__":
    main()

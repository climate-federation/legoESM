"""Ocean-WB D20 diagnostic figure: ORAS5 D20, model D20, model - ORAS5 (one sample, last lead),
and the area-weighted D20 error per lead day.  Reads ``d20_diagnostic.npz`` written by
``scripts/run/train_ocean_wb_physics.py``.  Colour-blind safe: viridis for depth, PuOr for the
difference, Okabe-Ito for lines.

    python scripts/plot/plot_ocean_wb_d20.py RUN_DIR [--sample 0]
"""
import argparse
from pathlib import Path

import numpy as np

OKABE_BLUE = "#0072B2"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--sample", type=int, default=0)
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.run_dir / "d20_diagnostic.npz")
    k, lead = a.sample, -1
    keep = d["scored"][k, lead]
    o = np.where(keep, d["oras5"][k, lead], np.nan)
    m = np.where(keep, d["model"][k, lead], np.nan)
    lim = float(np.nanpercentile(np.abs(m - o), 98)) if keep.any() else 1.0
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    vmax = float(np.nanpercentile(o, 99))
    fig = plt.figure(figsize=(13, 8))
    ext = (0, 360, -90, 90)
    for i, (f, title, cmap, vmin, vm) in enumerate((
            (o, "ORAS5 D20 [m]", "viridis", 0, vmax),
            (m, "model D20 [m]", "viridis", 0, vmax),
            (m - o, "model - ORAS5 [m]", "PuOr", -lim, lim))):
        ax = fig.add_subplot(2, 2, i + 1)
        im = ax.imshow(f, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vm, aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(d["lead_days"], d["mae_by_lead_m"], "o-", color=OKABE_BLUE)
    ax.set_xlabel("lead [days]")
    ax.set_ylabel("area-weighted |D20 error| [m]")
    ax.set_title("D20 error vs lead (scored: ORAS5 crosses 20 C)")
    fig.suptitle(f"{d['label']} | IC {d['dates'][k]}, lead {int(d['lead_days'][lead])} d, "
                 f"width {float(d['width_K'])} K")
    fig.tight_layout()
    out = a.run_dir / "d20_diagnostic.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()

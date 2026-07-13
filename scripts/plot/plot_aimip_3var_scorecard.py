"""Plain 2-panel AIMIP comparison: annual tas-anomaly time series (top) +
scorecard table (bottom) for classical / column_nn / sfno_physics, with an
optional dense overlay per variant when its CSV is present.

Anomaly = annual_global_mean_surfT_K minus that model's own 1979-2014 mean
(each variant has its own climatology, so absolute T is not comparable).
Reads results/aimip_fleet_paper/legoesm_<...>_annual.csv. Writes
results/aimip_fleet_paper/aimip_3var_scorecard.png.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLEET = Path("results/aimip_fleet_paper")
BASE_LO, BASE_HI = 1979, 2014  # anomaly reference window
# WeatherBench2 ERA5 reinit forcing effectively ends ~2022; the monthly-reinit
# hindcast is only constrained through then. 2023-2024 reinit clamps to stale
# ERA5 and the model free-runs those months -> a spurious ±3 K cliff that is a
# forcing-coverage artifact, not model skill. Cap the plot+scorecard there.
MAX_YEAR = 2022

# (label, csv-stem, color, linestyle). Dense entries are optional.
SERIES = [
    ("classical", "legoesm_classical_amip_r0_annual", "#9B2D6E", "--"),
    ("column_nn", "legoesm_column_nn_my_reinit_annual", "#E8720C", "-"),
    ("sfno_physics", "legoesm_sfno_physics_my_datactrl_annual", "#0E8C7E", "-"),
    ("column_nn (dense)", "legoesm_column_nn_dense_amip_reinit1mo_r0_annual",
     "#C42B1C", "-"),
    ("sfno_physics (dense)", "legoesm_sfno_physics_dense_amip_reinit1mo_r0_annual",
     "#337EBB", "-"),
]


def _load(stem: str) -> dict[int, float] | None:
    p = FLEET / f"{stem}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    key = "annual_global_mean_surfT_K"
    out = {}
    for r in rows:
        try:
            y = int(r["year"])
            if y > MAX_YEAR:
                continue  # drop the forcing-uncovered drift years
            out[y] = float(r[key])
        except (KeyError, ValueError):
            continue
    return out or None


def _anomaly(absT: dict[int, float]) -> dict[int, float]:
    base_yrs = [y for y in absT if BASE_LO <= y <= BASE_HI]
    ref = sum(absT[y] for y in base_yrs) / len(base_yrs)
    return {y: absT[y] - ref for y in absT}


def _stats(anom: dict[int, float]) -> tuple[int, int, int, float, float, float]:
    yrs = sorted(anom)
    n = len(yrs)
    mean = sum(anom.values()) / n
    rms = math.sqrt(sum(v * v for v in anom.values()) / n)
    late = [anom[y] for y in yrs if y >= 2013]
    rms_late = math.sqrt(sum(v * v for v in late) / len(late)) if late else float("nan")
    return yrs[0], yrs[-1], n, mean, rms, rms_late


def main() -> None:
    loaded = []
    for label, stem, color, ls in SERIES:
        absT = _load(stem)
        if absT is None:
            continue
        anom = _anomaly(absT)
        loaded.append((label, color, ls, anom, _stats(anom)))

    fig, (ax, axt) = plt.subplots(
        2, 1, figsize=(11, 9), height_ratios=[2.4, 1.0],
        gridspec_kw={"hspace": 0.28},
    )

    # --- time series ---
    ax.axhline(0, color="#c2cad6", lw=1.4, zorder=1)
    ax.axvline(2012.5, color="#c9a94b", lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(2012.7, ax.get_ylim()[1], "train → extrapolate",
            color="#b8791b", fontsize=9, style="italic", va="top")
    for label, color, ls, anom, _ in loaded:
        yrs = sorted(anom)
        ax.plot(yrs, [anom[y] for y in yrs], color=color, ls=ls, lw=2.2,
                label=label, zorder=3)
        ax.plot(yrs[-1], anom[yrs[-1]], "o", color=color, ms=5, zorder=4)
    ax.set_ylabel("T anomaly vs 1979–2014 (K)", fontsize=10.5)
    ax.set_title(f"AIMIP annual near-surface T anomaly — prescribed-SST, "
                 f"monthly-reinit hindcast (through {MAX_YEAR})",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper left")
    ax.grid(True, color="#edf0f4", lw=0.9)
    ax.set_xlim(1979, MAX_YEAR)
    for s in ax.spines.values():
        s.set_edgecolor("#cfd5dd")

    # --- scorecard table ---
    axt.axis("off")
    col = ["variant", "eval", "years", "mean bias (K)", "anom RMS (K)", "2013+ RMS (K)"]
    cells, colors = [], []
    for label, color, ls, anom, (y0, y1, n, mean, rms, rms_late) in loaded:
        mode = "free-run" if "classical" in label else "reinit 1mo"
        cells.append([
            label, mode, f"{y0}–{y1}", f"{mean:+.3f}", f"{rms:.3f}",
            "—" if math.isnan(rms_late) else f"{rms_late:.3f}",
        ])
        colors.append(color)
    tbl = axt.table(cellText=cells, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e6e9ee")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="#5c6675")
            cell.set_facecolor("#f6f7f9")
        elif c == 0:
            cell.set_text_props(fontweight="bold", color=colors[r - 1], ha="left")
            cell.set_facecolor("#ffffff")
        else:
            cell.set_facecolor("#ffffff")

    out = FLEET / "aimip_3var_scorecard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out} ({len(loaded)} variants: "
          f"{', '.join(l for l, *_ in loaded)})")


if __name__ == "__main__":
    main()

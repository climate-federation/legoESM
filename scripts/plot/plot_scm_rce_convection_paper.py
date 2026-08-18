#!/usr/bin/env python
"""Publication-quality figures for the SCM RCE convection intercomparison.

Reads the artifacts written by ``scripts/run/run_scm_rce_convection_tuning.py``
(``summary_table.csv`` + ``scheme_<name>.json``) plus the RCEMIP1 reference
bundle, and renders paper-ready vector PDFs (with PNG previews):

- ``fig_rce_profiles_T.pdf``   — small-multiples of equilibrium temperature
- ``fig_rce_profiles_qv.pdf``  — small-multiples of equilibrium water vapor
- ``fig_rce_rmse_summary.pdf`` — a-priori vs tuned RMSE bars (T and q_v)

Reference black solid, SCM a-priori red dashed, SCM a-posteriori red solid, on
a pressure ordinate (RCEMIP convention).  Text is embedded as editable
TrueType (``pdf.fonttype=42``) for downstream figure editing.

Run:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/plot/plot_scm_rce_convection_paper.py \
        --results-dir results/scm_rce_convection_tuning \
        --reference-dir results/rcemip1_small_wing_ocean
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

KG_KG_TO_G_KG = 1_000.0
PA_TO_HPA = 1.0e-2

REF_COLOR = "#111111"
SCM_COLOR = "#c0392b"
APRIORI_COLOR = "#8c8c8c"
TUNED_BAR = "#c0392b"


def _apply_paper_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.15,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    with (results_dir / "summary_table.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _load_scheme(results_dir: Path, scheme: str) -> dict[str, Any]:
    return json.loads((results_dir / f"scheme_{scheme}.json").read_text())


# The two tuning drivers name the untuned arm differently in their per-scheme
# checkpoints: run_scm_rce_convection_tuning.py writes "a_priori",
# run_scm_rce_convection_intercomparison.py writes "prior".  Read both rather
# than teaching one driver to duplicate the other's key — a silent {} here
# would drop the dashed a-priori curve from every panel and look like a
# plotting choice.
_APRIORI_KEYS = ("a_priori", "prior")


def _arm(rec: dict[str, Any], tuned: bool) -> dict[str, Any]:
    if tuned:
        return rec["tuned"]
    for key in _APRIORI_KEYS:
        if key in rec:
            return rec[key]
    raise KeyError(
        f"per-scheme checkpoint has no a-priori arm; expected one of "
        f"{_APRIORI_KEYS}, got {sorted(rec)}"
    )


def _ff(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _reference(reference_dir: Path, last_n: int):
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import WING_P_SFC
    from scripts.run.run_scm_rce_campaign import build_reference_profiles

    ref = build_reference_profiles(reference_dir, last_n)
    pressure_hpa = np.asarray(ref.sigma_full, dtype=float) * WING_P_SFC * PA_TO_HPA
    return ref, pressure_hpa


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))


def _profile_grid(
    out_base: Path,
    schemes: list[str],
    records: dict[str, dict[str, Any]],
    pressure_hpa: np.ndarray,
    ref_profile: np.ndarray,
    key: str,
    scale: float,
    xlabel: str,
    panel_title: str,
) -> None:
    import matplotlib.pyplot as plt

    n = len(schemes)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.1, 1.72 * nrows + 0.5),
        sharex=True, sharey=True, squeeze=False,
    )
    p_top = float(np.min(pressure_hpa))
    p_bot = float(np.max(pressure_hpa))
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (ax, scheme) in enumerate(zip(axes.flat, schemes)):
        ax.set_visible(True)
        rec = records[scheme]
        ax.plot(ref_profile * scale, pressure_hpa, color=REF_COLOR, lw=1.3,
                zorder=3)
        ap = _arm(rec, tuned=False).get(key)
        tu = _arm(rec, tuned=True).get(key)
        if ap:
            ax.plot(np.asarray(ap) * scale, pressure_hpa, color=SCM_COLOR,
                    lw=1.05, ls=(0, (4, 2)), zorder=4)
        if tu:
            ax.plot(np.asarray(tu) * scale, pressure_hpa, color=SCM_COLOR,
                    lw=1.15, zorder=5)
        ax.set_ylim(p_bot, p_top)
        ax.set_title(f"({letters[i]}) {scheme}", loc="left", fontsize=8)
        ax.tick_params(length=2.5)
    for r in range(nrows):
        ax = axes[r, 0]
        if ax.get_visible():
            ax.set_ylabel("pressure [hPa]")
    for c in range(ncols):
        # bottom-most visible axis in each column gets the x-label
        for r in range(nrows - 1, -1, -1):
            if axes[r, c].get_visible():
                axes[r, c].set_xlabel(xlabel)
                break
    # one shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=REF_COLOR, lw=1.3, label="RCEMIP1 reference"),
        Line2D([0], [0], color=SCM_COLOR, lw=1.05, ls=(0, (4, 2)),
               label="SCM a priori"),
        Line2D([0], [0], color=SCM_COLOR, lw=1.15, label="SCM a posteriori"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(panel_title, x=0.02, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    _save(fig, out_base)
    plt.close(fig)


def _rmse_summary(
    out_base: Path,
    rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    # sort by tuned temperature RMSE (ascending -> best on top)
    rows = sorted(rows, key=lambda r: _ff(r["tuned_T_rmse_K"]))
    schemes = [r["scheme"] for r in rows]
    y = np.arange(len(rows))[::-1]  # best at top
    h = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 0.42 * len(rows) + 1.1),
                             sharey=True)
    specs = [
        (axes[0], "apriori_T_rmse_K", "tuned_T_rmse_K",
         "temperature RMSE [K]", "(a)"),
        (axes[1], "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg",
         r"water-vapor RMSE [g kg$^{-1}$]", "(b)"),
    ]
    for ax, ak, tk, xlabel, letter in specs:
        ap = [_ff(r[ak]) for r in rows]
        tu = [_ff(r[tk]) for r in rows]
        ax.barh(y + h / 2, ap, height=h, color=APRIORI_COLOR,
                label="a priori", zorder=2)
        ax.barh(y - h / 2, tu, height=h, color=TUNED_BAR,
                label="a posteriori", zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_title(letter, loc="left", fontsize=9)
        ax.grid(axis="x", color="#e2e2e2", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(schemes)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def build_figures(
    results_dir: Path,
    reference_dir: Path,
    outdir: Path,
    last_reference_files: int = 5,
) -> list[Path]:
    _apply_paper_style()
    rows = _load_rows(results_dir)
    if not rows:
        raise SystemExit(f"No rows in {results_dir/'summary_table.csv'}")
    schemes = [r["scheme"] for r in rows]
    records = {s: _load_scheme(results_dir, s) for s in schemes}
    ref, pressure_hpa = _reference(reference_dir, last_reference_files)

    # Order the profile panels by tuned temperature RMSE for a legible read.
    order = sorted(schemes, key=lambda s: _ff(
        next(r["tuned_T_rmse_K"] for r in rows if r["scheme"] == s)))

    outdir.mkdir(parents=True, exist_ok=True)
    _profile_grid(
        outdir / "fig_rce_profiles_T", order, records, pressure_hpa,
        np.asarray(ref.T_ref, dtype=float), "T_profile", 1.0,
        "temperature [K]", "Equilibrium temperature vs RCEMIP1",
    )
    _profile_grid(
        outdir / "fig_rce_profiles_qv", order, records, pressure_hpa,
        np.asarray(ref.qv_ref, dtype=float), "qv_profile", KG_KG_TO_G_KG,
        r"water vapor [g kg$^{-1}$]", "Equilibrium water vapor vs RCEMIP1",
    )
    _rmse_summary(outdir / "fig_rce_rmse_summary", rows)
    return [
        outdir / "fig_rce_profiles_T.pdf",
        outdir / "fig_rce_profiles_qv.pdf",
        outdir / "fig_rce_rmse_summary.pdf",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results/scm_rce_convection_tuning"),
    )
    parser.add_argument(
        "--reference-dir", type=Path,
        default=Path("results/rcemip1_small_wing_ocean"),
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Default: <results-dir>/paper",
    )
    parser.add_argument("--last-reference-files", type=int, default=5)
    args = parser.parse_args(argv)
    outdir = args.outdir or (args.results_dir / "paper")
    paths = build_figures(
        args.results_dir, args.reference_dir, outdir,
        last_reference_files=args.last_reference_files,
    )
    print(f"[paper-figures] wrote {len(paths)} figures to {outdir}")
    for p in paths:
        print(f"    {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

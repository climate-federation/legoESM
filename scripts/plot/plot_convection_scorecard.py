"""Convection-scheme AMIP comparison scorecard (timeseries.npz → table + bars).

Ranks the convection-campaign runs (``conv_<scheme>/`` dirs from
``scripts/cluster/amip/amip_convection_campaign.sbatch``) against CERES /
GPCP / ERA5 global-mean targets, and names the most realistic scheme.

Each run's ``timeseries.npz`` already holds the driver's AREA-WEIGHTED
global-mean diagnostics per sample (``diagnostics.py`` uses
``area_weighted_mean``), so this script does no spatial reduction — it takes
the spun-up-window mean of each metric and grades it against the SAME
observational targets + acceptance bands the CMOR diagnostics plotter uses
(imported from ``plot_amip_cmor_diagnostics`` — NOT re-derived here).

Usage:
    python scripts/plot/plot_convection_scorecard.py \
        results/amip/convcmp --out results/amip/convcmp/scorecard.png \
        [--spinup-frac 0.5] [--schemes sbm tiedtke bechtold edmf]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

# Reuse the observational targets + acceptance bands from the CMOR plotter
# (sibling script, not an importable package) so the pass/fail criteria are
# identical across the two scorecards and defined in exactly one place.
_PLOTTER = Path(__file__).resolve().parent / "plot_amip_cmor_diagnostics.py"
_spec = importlib.util.spec_from_file_location("_amip_cmor_diag", _PLOTTER)
_diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag)

# timeseries.npz key -> (CMOR var, unit-scale already applied? , target, abs_tol)
# We map to the CMOR var so the target + tol come from the shared tables.
_FIELD_TARGET = {v[0]: (v[3], _diag._REALISM_ABS_TOL[v[0]])
                 for v in _diag.FIELD_TABLE}
# timeseries key -> CMOR var (timeseries stores SI: precip already mm/day,
# fluxes W/m2, CWV kg/m2 == mm).
_TS_TO_CMOR = {
    "T_low": "tas",
    "precip": "pr",
    "sw_up_toa": "rsut",
    "lw_up_toa": "rlut",
    "CWV": "prw",
    "hfls": "hfls",
    "hfss": "hfss",
}


def _spinup_mean(ts: dict, key: str, spinup_frac: float) -> float:
    v = np.asarray(ts[key], dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        return float("nan")
    i0 = int(spinup_frac * v.size)
    i0 = min(max(i0, 0), v.size - 1)
    return float(np.nanmean(v[i0:]))


def score_run(run_dir: str | Path, spinup_frac: float = 0.5) -> dict | None:
    """Grade one run's timeseries.npz vs the shared obs targets.

    Returns a dict with per-metric (value, target, tol, pass) plus derived
    planetary albedo and net TOA imbalance, the count of passing metrics,
    and a composite normalized error (mean |value-target|/tol over the
    graded metrics). ``None`` if the run has no readable timeseries.
    """
    p = Path(run_dir) / "timeseries.npz"
    if not p.exists():
        return None
    with np.load(p) as ts:
        ts = {k: ts[k] for k in ts.files}

    metrics: dict[str, dict] = {}
    for ts_key, cmor in _TS_TO_CMOR.items():
        if ts_key not in ts or cmor not in _FIELD_TARGET:
            continue
        val = _spinup_mean(ts, ts_key, spinup_frac)
        target, tol = _FIELD_TARGET[cmor]
        metrics[cmor] = {
            "value": val, "target": float(target), "tol": float(tol),
            "pass": bool(np.isfinite(val) and abs(val - target) <= tol),
        }

    # Derived planetary albedo (rsut / rsdt) and net TOA imbalance — use the
    # shared albedo/TOA acceptance bands, not FIELD_TABLE.
    rsut = _spinup_mean(ts, "sw_up_toa", spinup_frac)
    rsdt = _spinup_mean(ts, "rsdt", spinup_frac) if "rsdt" in ts else float("nan")
    rlut = _spinup_mean(ts, "lw_up_toa", spinup_frac)
    if np.isfinite(rsut) and np.isfinite(rsdt) and rsdt > 0:
        albedo = rsut / rsdt
        metrics["albedo"] = {
            "value": albedo, "target": _diag._ALBEDO_REF,
            "tol": _diag._ALBEDO_ABS_TOL,
            "pass": bool(abs(albedo - _diag._ALBEDO_REF) <= _diag._ALBEDO_ABS_TOL),
        }
        r_toa = rsdt - rsut - rlut
        metrics["R_TOA"] = {
            "value": r_toa, "target": 0.0, "tol": _diag._R_TOA_ABS_TOL,
            "pass": bool(abs(r_toa) <= _diag._R_TOA_ABS_TOL),
        }

    graded = [m for m in metrics.values() if np.isfinite(m["value"])]
    n_pass = sum(m["pass"] for m in graded)
    comp_err = (float(np.mean([abs(m["value"] - m["target"]) / m["tol"]
                               for m in graded]))
                if graded else float("inf"))
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "n_pass": n_pass,
        "n_graded": len(graded),
        "composite_error": comp_err,
        "final_day": (float(np.asarray(ts["days"])[-1])
                      if "days" in ts and np.asarray(ts["days"]).size else None),
    }


def rank_scheme_scores(scores: dict[str, dict]) -> list[tuple[str, dict]]:
    """Scheme name -> score dict, ranked best-first (more passes, then lower
    composite error)."""
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["n_pass"], kv[1]["composite_error"]),
    )


def _format_table(ranked: list[tuple[str, dict]]) -> str:
    cols = ["pr", "rsut", "rlut", "albedo", "R_TOA", "hfls", "prw", "tas"]
    head = f"{'scheme':<12} {'day':>4} {'pass':>5}  " + " ".join(
        f"{c:>7}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, s in ranked:
        cells = []
        for c in cols:
            m = s["metrics"].get(c)
            if m is None or not np.isfinite(m["value"]):
                cells.append(f"{'--':>7}")
            else:
                flag = "" if m["pass"] else "*"
                cells.append(f"{m['value']:>6.2f}{flag}")
        day = f"{s['final_day']:.0f}" if s["final_day"] is not None else "--"
        lines.append(
            f"{name:<12} {day:>4} {s['n_pass']:>2}/{s['n_graded']:<2}  "
            + " ".join(cells))
    lines.append("(* = outside the observational acceptance band)")
    return "\n".join(lines)


def plot_scorecard(ranked: list[tuple[str, dict]], out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["pr", "rsut", "rlut", "albedo", "hfls", "tas"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        names = [n for n, _ in ranked]
        vals = [ranked_s["metrics"].get(c, {}).get("value", np.nan)
                for _, ranked_s in ranked]
        m0 = next((s["metrics"][c] for _, s in ranked if c in s["metrics"]), None)
        colors = ["#2ca02c" if ranked_s["metrics"].get(c, {}).get("pass")
                  else "#d62728" for _, ranked_s in ranked]
        ax.bar(names, vals, color=colors)
        if m0 is not None:
            ax.axhline(m0["target"], color="k", ls="--", lw=1)
            ax.axhspan(m0["target"] - m0["tol"], m0["target"] + m0["tol"],
                       color="k", alpha=0.08)
        ax.set_title(c)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("AMIP convection-scheme scorecard vs CERES/GPCP/ERA5 "
                 "(green=within obs band)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("campaign_dir", type=Path,
                   help="dir holding conv_<scheme>/ subdirs")
    p.add_argument("--out", type=Path, default=None,
                   help="scorecard PNG path (default <campaign_dir>/scorecard.png)")
    p.add_argument("--schemes", nargs="*", default=None,
                   help="scheme subdir suffixes to score (default: all conv_*)")
    p.add_argument("--spinup-frac", type=float, default=0.5,
                   help="fraction of the record discarded as spin-up")
    args = p.parse_args(argv)

    if args.schemes:
        dirs = [args.campaign_dir / f"conv_{s}" for s in args.schemes]
    else:
        dirs = sorted(args.campaign_dir.glob("conv_*"))
    scores: dict[str, dict] = {}
    for d in dirs:
        s = score_run(d, spinup_frac=args.spinup_frac)
        if s is not None:
            scores[d.name.replace("conv_", "")] = s
    if not scores:
        p.error(f"no scoreable runs (conv_*/timeseries.npz) in {args.campaign_dir}")

    ranked = rank_scheme_scores(scores)
    print(_format_table(ranked))
    best, best_s = ranked[0]
    print(f"\nMost realistic: {best!r} "
          f"({best_s['n_pass']}/{best_s['n_graded']} metrics in band, "
          f"composite error {best_s['composite_error']:.2f})")

    out = args.out or (args.campaign_dir / "scorecard.png")
    plot_scorecard(ranked, out)
    print(f"[scorecard] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

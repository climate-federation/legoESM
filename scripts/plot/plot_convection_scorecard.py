"""Convection-scheme AMIP comparison scorecard (→ ranked table + bars).

Ranks the convection-campaign runs (``conv_<scheme>/`` dirs from
``scripts/cluster/amip/amip_convection_campaign.sbatch``) against CERES /
GPCP / ERA5 global-mean targets, and names the most realistic scheme.

Two scoring sources (``--source``):

* ``amon`` (DEFAULT) — grade each run's CMOR ``Amon`` monthly-mean
  climatology by REUSING ``plot_amip_cmor_diagnostics`` (its area-weighted
  ``compute_amip_diagnostics`` + ``amip_realism_scorecard``, spin-up dropped
  via ``spinup_frac``).  This is the ROBUST source for the chained production
  runs: the CMOR monthly accumulator is restored across chain links, so the
  climatology spans the whole run — whereas ``timeseries.npz`` holds only the
  sparse per-checkpoint samples of the FINAL link (its ``self.times`` list is
  not restored on restart), which for a chained run collapses to a handful of
  post-restart transient samples.  Requires ``cmor/Amon/*.nc`` (uses xarray).

* ``timeseries`` — grade the driver's lightweight ``timeseries.npz``
  area-weighted global means directly (xarray-free).  Valid for a SINGLE-link
  run (idealised / RCE benches); UNDER-SAMPLED for a chained multi-link run.

Both sources grade against the SAME observational targets + acceptance bands
(shared ``_amip_obs_targets`` — NOT re-derived here).

Usage:
    python scripts/plot/plot_convection_scorecard.py \
        results/amip/convcmp --out results/amip/convcmp/scorecard.png \
        [--source amon|timeseries] [--spinup-frac 0.5] [--tol-scale 1.0] \
        [--schemes sbm tiedtke bechtold edmf]
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

# Reuse the observational targets + acceptance bands from the dependency-light
# shared module (sibling script, not an importable package) so the pass/fail
# criteria are identical across the two scorecards and defined in one place —
# WITHOUT pulling the CMOR plotter's xarray import (this scorecard is
# timeseries-only).
_TARGETS = Path(__file__).resolve().parent / "_amip_obs_targets.py"
_spec = importlib.util.spec_from_file_location("_amip_obs_targets", _TARGETS)
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
    w = v[i0:]
    w = w[np.isfinite(w)]          # explicit finite mask (no nanmean warning)
    return float(np.mean(w)) if w.size else float("nan")


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
    if not graded:
        return None          # no finite metric -> run is not scoreable
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


def _final_day(run_dir: str | Path) -> float | None:
    """Latest simulated day from the run's ``checkpoint_day_NNNN.npz`` files, or
    ``None`` if absent.  Uses the NUMERIC max of the parsed day (not a
    lexicographic filename sort, which would rank ``..._10000`` before
    ``..._9999`` once the day count needs a fifth digit)."""
    import re
    days = [int(m.group(1))
            for f in Path(run_dir).glob("checkpoint_day_*.npz")
            if (m := re.search(r"checkpoint_day_0*(\d+)\.npz", f.name))]
    return float(max(days)) if days else None


def _load_cmor_plotter():
    """Lazy-load the sibling ``plot_amip_cmor_diagnostics`` module (pulls in
    xarray) — only when scoring from the ``amon`` source, so the ``timeseries``
    path stays xarray-free."""
    import importlib.util
    p = Path(__file__).resolve().parent / "plot_amip_cmor_diagnostics.py"
    spec = importlib.util.spec_from_file_location("plot_amip_cmor_diagnostics", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_run_amon(run_dir: str | Path, spinup_frac: float = 0.5,
                   tol_scale: float = 1.0) -> dict | None:
    """Grade one run's CMOR ``Amon`` climatology by reusing the CMOR plotter's
    area-weighted ``compute_amip_diagnostics`` + ``amip_realism_scorecard``
    (the ROBUST, cross-chain-link source — see the module docstring).

    ``n_pass``/``n_graded`` come from the shared scorecard and so INCLUDE the
    TOA-budget and large-scale STRUCTURE checks (equator-pole gradient, ITCZ);
    the per-metric table + ``composite_error`` cover the numeric-band checks
    (fields + albedo + net-TOA), which are the ones with a comparable
    normalized error.  Returns ``None`` if the run has no ``cmor/Amon/tas`` or
    no finite graded metric (mirrors ``score_run``)."""
    run_dir = Path(run_dir)
    mod = _load_cmor_plotter()
    try:
        diag = mod.compute_amip_diagnostics(run_dir, spinup_frac=spinup_frac)
    except FileNotFoundError:
        return None
    card = mod.amip_realism_scorecard(diag, tol_scale=tol_scale)

    metrics: dict[str, dict] = {}
    for var, d in card["fields"].items():
        metrics[var] = {"value": d["value"], "target": d["ref"],
                        "tol": d["abs_tol"], "pass": d["within"]}
    if card["budget"] is not None:
        alb, rt = card["budget"]["albedo"], card["budget"]["r_toa"]
        metrics["albedo"] = {"value": alb["value"], "target": alb["ref"],
                             "tol": alb["abs_tol"], "pass": alb["within"]}
        metrics["R_TOA"] = {"value": rt["value"], "target": rt["ref"],
                            "tol": rt["abs_tol"], "pass": rt["within"]}

    graded = [m for m in metrics.values()
              if np.isfinite(m["value"]) and m["tol"] > 0]
    if not graded:
        return None          # no finite band-checked metric -> not scoreable
    comp_err = float(np.mean([abs(m["value"] - m["target"]) / m["tol"]
                              for m in graded]))
    # A run MISSING a required field or the TOA budget (e.g. a crashed run that
    # wrote only some Amon vars) must NOT read as perfect: the shared card does
    # not count an absent field as a failed check, so fold each missing-required
    # item into n_graded as an (unpassed) failed check. This keeps
    # n_pass == n_graded true ONLY for a complete, all-in-band run, and makes the
    # ranker (sorts by -n_pass first) deprioritise an incomplete run.
    missing = list(card.get("missing_required", []))
    n_graded = int(card["n_checks"]) + len(missing)
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "n_pass": int(card["n_pass"]),
        "n_graded": n_graded,
        "composite_error": comp_err,
        "final_day": _final_day(run_dir),
        "structure": card.get("structure", {}),
        "missing_required": missing,
        "passed": bool(card.get("passed", False)),
    }


def score_run_auto(run_dir: str | Path, spinup_frac: float = 0.5,
                   tol_scale: float = 1.0) -> dict | None:
    """Score a run by its BEST available source: the robust CMOR ``amon``
    climatology when ``cmor/Amon`` is present (and scoreable), otherwise the
    ``timeseries.npz`` fallback — so a single-link / idealised run with no CMOR
    output is still graded instead of erroring."""
    run_dir = Path(run_dir)
    if (run_dir / "cmor" / "Amon").is_dir():
        s = score_run_amon(run_dir, spinup_frac=spinup_frac, tol_scale=tol_scale)
        if s is not None:
            return s
    return score_run(run_dir, spinup_frac=spinup_frac)


def rank_scheme_scores(scores: dict[str, dict]) -> list[tuple[str, dict]]:
    """Scheme name -> score dict, ranked best-first by, in order:

    1. MORE passing checks (``n_pass``);
    2. FEWER missing-required items — a COMPLETE run outranks an incomplete run
       (missing a required field / the TOA budget) on the same ``n_pass``, so a
       run that only wrote some Amon vars can't sneak ahead on a lower composite
       error over its handful of present metrics;
    3. LOWER composite error.

    ``missing_required`` is absent from a timeseries-source score (that path has
    no required-field concept) — treated as complete (``len == 0``)."""
    return sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["n_pass"],
                        len(kv[1].get("missing_required", [])),
                        kv[1]["composite_error"]),
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
    p.add_argument("--source", choices=["auto", "amon", "timeseries"],
                   default="auto",
                   help="scoring source: 'auto' (default; per-run: robust CMOR "
                        "'amon' climatology when cmor/Amon present, else the "
                        "'timeseries.npz' fallback), 'amon', or 'timeseries' "
                        "(sparse — single-link runs only)")
    p.add_argument("--spinup-frac", type=float, default=0.5,
                   help="leading fraction of the record discarded as spin-up")
    p.add_argument("--tol-scale", type=float, default=1.0,
                   help="scale every acceptance band (amon source; >1 widen)")
    args = p.parse_args(argv)
    if not (0.0 <= args.spinup_frac < 1.0):
        p.error(f"--spinup-frac must be in [0, 1), got {args.spinup_frac}")
    if args.tol_scale <= 0.0:
        p.error(f"--tol-scale must be > 0, got {args.tol_scale}")

    if args.schemes:
        dirs = [args.campaign_dir / f"conv_{s}" for s in args.schemes]
    else:
        dirs = sorted(args.campaign_dir.glob("conv_*"))
    scores: dict[str, dict] = {}
    for d in dirs:
        if args.source == "timeseries":
            s = score_run(d, spinup_frac=args.spinup_frac)
        else:
            scorer = score_run_amon if args.source == "amon" else score_run_auto
            s = scorer(d, spinup_frac=args.spinup_frac, tol_scale=args.tol_scale)
        if s is not None:
            scores[d.name.replace("conv_", "")] = s
    if not scores:
        _src = {"amon": "conv_*/cmor/Amon/tas_Amon_*.nc",
                "timeseries": "conv_*/timeseries.npz"}.get(
                    args.source, "conv_*/cmor/Amon/*.nc or conv_*/timeseries.npz")
        p.error(f"no scoreable runs ({_src}) in {args.campaign_dir}")

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

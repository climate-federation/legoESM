#!/usr/bin/env python
"""Summarize AIMIP lat-lon results into the deliverable comparison tables.

Two products the campaign asks for:

1. **Model-family comparison** — read one orchestrator run's
   ``aimip_latlon_scorecard.json`` (the classical / column_nn / sfno
   variants) and emit a markdown table of held-out RMSE for the prognostics
   AND the TOA/surface radiation fluxes, so the "which family wins" answer
   includes flux fidelity (the generalization criterion), not just state
   error.

2. **Best physics combination** — read every combo scorecard under a sweep
   root (``run_aimip_latlon_sweep.py`` output) and rank the classical
   physics combinations by a composite skill score, reporting the best
   convection x turbulence x GWD stack.

The composite score is the mean of each metric's RMSE normalized by the
median RMSE across the compared entries (dimensionless, lower = better), so
state variables (K, m/s, g/kg) and fluxes (W/m^2) combine without arbitrary
unit weights.  Flux and state groups are averaged then combined 50/50 so a
model cannot win on state error while radiating badly.

Usage:
    python summarize_aimip_latlon.py --run results/aimip_latlon/smoke_123
    python summarize_aimip_latlon.py --sweep results/aimip_latlon_sweep
    python summarize_aimip_latlon.py --run <dir> --sweep <dir> -o table.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# Metric groups (keys as emitted by run_aimip_latlon.evaluate()).
_STATE_KEYS = ["T", "u", "v", "q", "ps"]
_FLUX_KEYS = ["rsut", "olr", "sfc_net_sw", "sfc_net_lw"]
_ALL_KEYS = _STATE_KEYS + _FLUX_KEYS


def _load_scorecard(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _finite(x) -> bool:
    """True only for a real, finite number (rejects None / NaN / inf)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def _rmse(entry: dict, key: str):
    """Finite RMSE for metric ``key``, or None (None/NaN/inf -> None)."""
    m = entry.get("eval_metrics", {}).get(key)
    if not isinstance(m, dict):
        return None
    v = m.get("rmse")
    return v if _finite(v) else None


def _composite_scores(entries: dict[str, dict]) -> dict[str, float]:
    """Median-normalized composite skill per entry (lower = better).

    ``entries`` maps label -> scorecard variant/combo block.  The score is
    the 50/50 mean of the state-group and flux-group means, each metric's
    RMSE normalized by the positive median across entries.  A group counts
    only if ALL of its metrics are present and finite; otherwise the whole
    composite is ``inf``.  So a partial/errored entry can never out-rank a
    complete one (a composite over an unequal metric set is not a fair
    comparison), and the 50/50 weighting is never silently renormalized to
    one group.  ``evaluate()`` always emits all nine metrics, so a real
    complete run is rankable; a missing metric signals corruption and is
    treated as unrankable.
    """
    # Positive-finite median per metric (the normalizer); skip degenerate
    # (all-None / all-zero) metrics so we never divide by zero.
    medians = {}
    for k in _ALL_KEYS:
        vals = [v for v in (_rmse(e, k) for e in entries.values())
                if _finite(v) and v > 0.0]
        if vals:
            medians[k] = statistics.median(vals)

    def _grp(e, keys):
        # Require EVERY metric in the group to be present + finite — a
        # composite over an unequal metric set is not a fair comparison, so
        # a partial entry must be unrankable, not cheaply low-scored.
        if any(_rmse(e, k) is None for k in keys):
            return None
        # rmse/median where a positive median exists; an all-zero metric
        # (no median) is equal for everyone -> contributes 0.
        norm = [(_rmse(e, k) / medians[k]) if k in medians else 0.0
                for k in keys]
        return sum(norm) / len(norm)

    scores = {}
    for label, e in entries.items():
        state, flux = _grp(e, _STATE_KEYS), _grp(e, _FLUX_KEYS)
        if state is None or flux is None:
            scores[label] = float("inf")   # incomplete -> not rankable
        else:
            scores[label] = 0.5 * state + 0.5 * flux
    return scores


def _best(scores: dict[str, float]):
    """Label with the lowest FINITE composite, or None if none rankable."""
    finite = {l: s for l, s in scores.items() if math.isfinite(s)}
    return min(finite, key=finite.get) if finite else None


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(headers) + " |"
    sep += "\n|" + "|".join(["---"] * len(headers)) + "|"
    for r in rows:
        sep += "\n| " + " | ".join(r) + " |"
    return sep


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if _finite(x) else "—"


def summarize_run(scorecard_path: Path) -> str:
    """Markdown family-comparison table for one orchestrator run."""
    sc = _load_scorecard(scorecard_path)
    variants = {k: v for k, v in sc.items() if isinstance(v, dict)}
    scores = _composite_scores(variants)
    best = _best(scores)

    headers = ["model"] + _ALL_KEYS + ["composite"]
    rows = []
    for label in sorted(variants, key=lambda l: scores.get(l, float("inf"))):
        e = variants[label]
        failed = "eval_metrics" not in e
        tag = " (FAILED)" if failed else ""
        cells = [_fmt(_rmse(e, k)) for k in _ALL_KEYS]
        mark = " **★**" if label == best else ""
        rows.append([f"`{label}`{tag}{mark}"] + cells + [f"**{_fmt(scores[label])}**"])

    out = [f"### Model-family comparison — `{scorecard_path}`",
           "RMSE vs ERA5 (held-out); fluxes in W/m². ★ = best composite "
           "(50/50 state+flux, median-normalized; needs both groups; lower is "
           "better).", "", _md_table(headers, rows)]
    out.append(f"\n**Best family: `{best}`** (composite {scores[best]:.3f})."
               if best else "\n_No rankable family (all incomplete/failed)._")
    return "\n".join(out)


def summarize_sweep(sweep_root: Path) -> str:
    """Markdown best-physics-combo table over a sweep root."""
    sweep_root = Path(sweep_root)
    combos = {}
    for sc_path in sorted(sweep_root.glob("*/aimip_latlon_scorecard.json")):
        combo = sc_path.parent.name
        sc = _load_scorecard(sc_path)
        # classical sweep: one variant ("classical") per combo scorecard.
        block = sc.get("classical")
        if isinstance(block, dict):
            combos[combo] = block

    if not combos:
        return (f"### Best physics combination — `{sweep_root}`\n"
                "_No combo scorecards found "
                "(`<combo>/aimip_latlon_scorecard.json`)._")

    scores = _composite_scores(combos)
    best = _best(scores)
    headers = ["combo"] + _STATE_KEYS + _FLUX_KEYS + ["composite"]
    rows = []
    for combo in sorted(combos, key=lambda c: scores[c]):
        e = combos[combo]
        tag = " (FAILED)" if "eval_metrics" not in e else ""
        cells = [_fmt(_rmse(e, k)) for k in _ALL_KEYS]
        mark = " **★**" if combo == best else ""
        rows.append([f"`{combo}`{tag}{mark}"] + cells + [f"**{_fmt(scores[combo])}**"])

    tail = [f"### Best physics combination — `{sweep_root}`",
            "Classical-variant RMSE vs ERA5 per swept combo. ★ = best composite "
            "(needs both state and flux groups).", "", _md_table(headers, rows)]
    if best is None:
        tail.append("\n_No rankable combo (all incomplete/failed)._")
    else:
        sch = combos[best].get("schemes", {})
        combo_desc = (f"convection=`{sch.get('convection','?')}`, "
                      f"turbulence=`{sch.get('turbulence','?')}`, "
                      f"gwd=`{sch.get('gravity_wave_drag','?')}`, "
                      f"microphysics=`{sch.get('microphysics','?')}`, "
                      f"clouds=`{sch.get('clouds','?')}`, "
                      f"radiation=`{sch.get('radiation','?')}`")
        tail.append(f"\n**Best combination: `{best}`** — {combo_desc} "
                    f"(composite {scores[best]:.3f}).")
    return "\n".join(tail)


def main(argv=None):
    p = argparse.ArgumentParser(description="Summarize AIMIP lat-lon results")
    p.add_argument("--run", type=Path, default=None,
                   help="orchestrator run dir OR scorecard.json (family table)")
    p.add_argument("--sweep", type=Path, default=None,
                   help="sweep root dir (best-physics-combo table)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="write markdown here (default: stdout)")
    args = p.parse_args(argv)
    if not args.run and not args.sweep:
        p.error("pass --run and/or --sweep")

    parts = []
    if args.run:
        run = args.run
        if run.is_dir():
            run = run / "aimip_latlon_scorecard.json"
        parts.append(summarize_run(run))
    if args.sweep:
        parts.append(summarize_sweep(args.sweep))
    md = "\n\n".join(parts) + "\n"

    if args.output:
        args.output.write_text(md)
        print(f"Wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

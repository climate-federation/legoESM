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
import statistics
from pathlib import Path

# Metric groups (keys as emitted by run_aimip_latlon.evaluate()).
_STATE_KEYS = ["T", "u", "v", "q", "ps"]
_FLUX_KEYS = ["rsut", "olr", "sfc_net_sw", "sfc_net_lw"]
_ALL_KEYS = _STATE_KEYS + _FLUX_KEYS


def _load_scorecard(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _rmse(entry: dict, key: str):
    """RMSE for metric ``key`` from a variant/combo eval block, or None."""
    em = entry.get("eval_metrics", {})
    m = em.get(key)
    return None if m is None else m.get("rmse")


def _composite_scores(entries: dict[str, dict]) -> dict[str, float]:
    """Median-normalized composite skill per entry (lower = better).

    ``entries`` maps label -> scorecard variant/combo block.  Missing or
    errored entries (no eval_metrics) get ``inf``.
    """
    # Median per metric across the entries that have it (the normalizer).
    medians = {}
    for k in _ALL_KEYS:
        vals = [v for v in (_rmse(e, k) for e in entries.values())
                if v is not None and v == v]  # drop None/NaN
        if vals:
            medians[k] = statistics.median(vals) or 1.0

    scores = {}
    for label, e in entries.items():
        if "eval_metrics" not in e:
            scores[label] = float("inf")
            continue

        def _grp(keys):
            norm = [_rmse(e, k) / medians[k]
                    for k in keys
                    if k in medians and _rmse(e, k) is not None]
            return sum(norm) / len(norm) if norm else None

        state, flux = _grp(_STATE_KEYS), _grp(_FLUX_KEYS)
        parts = [p for p in (state, flux) if p is not None]
        scores[label] = sum(parts) / len(parts) if parts else float("inf")
    return scores


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(headers) + " |"
    sep += "\n|" + "|".join(["---"] * len(headers)) + "|"
    for r in rows:
        sep += "\n| " + " | ".join(r) + " |"
    return sep


def _fmt(x, nd=3):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def summarize_run(scorecard_path: Path) -> str:
    """Markdown family-comparison table for one orchestrator run."""
    sc = _load_scorecard(scorecard_path)
    variants = {k: v for k, v in sc.items() if isinstance(v, dict)}
    scores = _composite_scores(variants)
    best = min(scores, key=scores.get) if scores else None

    headers = ["model"] + _ALL_KEYS + ["composite"]
    rows = []
    for label in sorted(variants, key=lambda l: scores.get(l, float("inf"))):
        e = variants[label]
        if "eval_metrics" not in e:
            rows.append([f"{label} (FAILED)"] + ["—"] * len(_ALL_KEYS) + ["—"])
            continue
        cells = [_fmt(_rmse(e, k)) for k in _ALL_KEYS]
        comp = _fmt(scores[label])
        mark = " **★**" if label == best else ""
        rows.append([f"`{label}`{mark}"] + cells + [f"**{comp}**"])

    out = [f"### Model-family comparison — `{scorecard_path}`",
           "RMSE vs ERA5 (held-out); fluxes in W/m². ★ = best composite "
           "(state+flux, median-normalized, lower is better).", "",
           _md_table(headers, rows)]
    if best:
        out.append(f"\n**Best family: `{best}`** (composite {scores[best]:.3f}).")
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
    best = min(scores, key=scores.get)
    headers = ["combo"] + _STATE_KEYS + _FLUX_KEYS + ["composite"]
    rows = []
    for combo in sorted(combos, key=lambda c: scores[c]):
        e = combos[combo]
        cells = [_fmt(_rmse(e, k)) for k in _ALL_KEYS]
        mark = " **★**" if combo == best else ""
        rows.append([f"`{combo}`{mark}"] + cells + [f"**{_fmt(scores[combo])}**"])

    sch = combos[best].get("schemes", {})
    combo_desc = (f"convection=`{sch.get('convection','?')}`, "
                  f"turbulence=`{sch.get('turbulence','?')}`, "
                  f"gwd=`{sch.get('gravity_wave_drag','?')}`, "
                  f"microphysics=`{sch.get('microphysics','?')}`, "
                  f"radiation=`{sch.get('radiation','?')}`")
    return "\n".join([
        f"### Best physics combination — `{sweep_root}`",
        "Classical-variant RMSE vs ERA5 per swept combo. ★ = best composite.",
        "", _md_table(headers, rows),
        f"\n**Best combination: `{best}`** — {combo_desc} "
        f"(composite {scores[best]:.3f}).",
    ])


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

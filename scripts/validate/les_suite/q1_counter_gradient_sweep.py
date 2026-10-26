#!/usr/bin/env python
"""Q1a structural-ceiling sweep: counter-gradient layer vs surface heat flux.

Reads every LES artifact in a directory (the flux-sweep emitted by
``run_les_suite.py --q0 ...``), runs the counter-gradient diagnostic on each final
snapshot, and writes a JSON mapping surface flux ``Q0`` → whether/where a
counter-gradient layer exists — the tuning-independent ceiling on local closures
(LES_SUITE.md §7 Q1a). Fast + CPU-only (no tuning, no SCM).

Usage::

    python scripts/validate/les_suite/q1_counter_gradient_sweep.py \\
        --artifacts-dir results/les_suite/artifacts \\
        --output results/les_suite/q1_counter_gradient_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legoesm.atmosphere.les_suite.bridge import diagnostic_truth, load_artifact
from legoesm.atmosphere.les_suite.counter_gradient import diagnose_truth


def sweep(artifacts_dir: Path) -> list[dict]:
    """Q1a diagnostic for every artifact in ``artifacts_dir``, sorted by Q0."""
    rows: list[dict] = []
    for path in sorted(artifacts_dir.glob("*.npz")):
        try:
            art = load_artifact(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {path.name}: unreadable ({exc}); skipped", file=sys.stderr)
            continue
        if art.w_theta_s is None:
            continue  # not a prescribed-flux (dry CBL) artifact
        truth = diagnostic_truth(art)  # final snapshot, total (resolved+SGS) flux
        cg = diagnose_truth(truth)
        rows.append({
            "artifact": path.name,
            "case": art.case_name,
            "sgs": art.sgs,
            "Q0_K_m_s": float(art.w_theta_s[0]),
            "final_time_s": float(art.times_s[-1]),
            "has_counter_gradient_layer": cg.has_counter_gradient_layer,
            "layer_base_m": cg.layer_base_m,
            "layer_top_m": cg.layer_top_m,
            "counter_gradient_fraction": cg.counter_gradient_fraction,
        })
    rows.sort(key=lambda r: r["Q0_K_m_s"])
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-dir", type=Path,
                   default=Path("results/les_suite/artifacts"))
    p.add_argument("--output", type=Path,
                   default=Path("results/les_suite/q1_counter_gradient_sweep.json"))
    args = p.parse_args(argv)

    if not args.artifacts_dir.exists():
        print(f"error: {args.artifacts_dir} does not exist", file=sys.stderr)
        return 2
    rows = sweep(args.artifacts_dir)
    if not rows:
        print(f"error: no prescribed-flux CBL artifacts in {args.artifacts_dir}",
              file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    print(f"{'Q0[K m/s]':>10}  {'CG layer':>9}  {'base[m]':>8}  {'top[m]':>8}  frac")
    for r in rows:
        base = f"{r['layer_base_m']:.0f}" if r["layer_base_m"] is not None else "-"
        top = f"{r['layer_top_m']:.0f}" if r["layer_top_m"] is not None else "-"
        print(f"{r['Q0_K_m_s']:>10.3f}  {str(r['has_counter_gradient_layer']):>9}  "
              f"{base:>8}  {top:>8}  {r['counter_gradient_fraction']:.2f}")
    print(f"\n-> {args.output}  ({len(rows)} fluxes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

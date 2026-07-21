#!/usr/bin/env python
"""Generate the LES-suite scorecard (Q2 ranking + Q3 coefficient spread).

Reads the per-(closure, regime) tuned-result JSONs written by
``scripts/run/tune_scm_to_les.py`` (default: ``results/les_suite/tuned/*.json``),
looks up each case's regime from the LESCase registry, and assembles the Q2/Q3
scorecard via ``legoesm.atmosphere.les_suite.scorecard``.

Usage::

    python scripts/validate/les_suite/build_les_scorecard.py \\
        --tuned-dir results/les_suite/tuned \\
        --output results/les_suite/scorecard.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
from legoesm.atmosphere.les_suite.scorecard import assemble_scorecard, render_markdown


def _load_tuned(tuned_dir: Path) -> list[dict]:
    """Load tuned JSONs and attach each case's regime from the registry."""
    if not list_cases():
        register_default_catalog()
    records: list[dict] = []
    for path in sorted(tuned_dir.glob("*.json")):
        with open(path) as f:
            rec = json.load(f)
        case = rec.get("case")
        try:
            rec["regime"] = get_case(case).regime
        except Exception:  # noqa: BLE001 — unknown case: skip with a warning
            print(f"[warn] {path.name}: case {case!r} not in registry; skipped",
                  file=sys.stderr)
            continue
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tuned-dir", type=Path, default=Path("results/les_suite/tuned"))
    p.add_argument("--output", type=Path, default=Path("results/les_suite/scorecard.md"))
    args = p.parse_args(argv)

    if not args.tuned_dir.exists():
        print(f"error: {args.tuned_dir} does not exist (run tune_scm_to_les first)",
              file=sys.stderr)
        return 2
    records = _load_tuned(args.tuned_dir)
    if not records:
        print(f"error: no usable tuned records in {args.tuned_dir}", file=sys.stderr)
        return 1

    card = assemble_scorecard(records)
    md = render_markdown(card)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(md)
    print(f"\n-> {args.output}  ({len(records)} tuned records, "
          f"{len(card.rankings)} regimes, {len(card.spreads)} coefficients)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

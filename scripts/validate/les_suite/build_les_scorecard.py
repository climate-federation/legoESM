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
import re
import sys
from pathlib import Path

from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
from legoesm.atmosphere.les_suite.scorecard import assemble_scorecard, render_markdown

# scheme suffix on a tuned filename ``<artifact>__<scheme>__df.json`` — used only to
# recover per-flux provenance for LEGACY records that predate the tuner stamping
# ``artifact``/``q0`` into the JSON itself.
_Q0_RE = re.compile(r"__q0_([0-9]*\.?[0-9]+)")


def _backfill_provenance(rec: dict, path: Path) -> None:
    """Ensure ``artifact`` + ``q0`` are set so the scorecard groups per flux.

    New records carry them (written by ``tune_scm_to_les.py``). For legacy records
    they are recovered from the tuned filename ``<artifact>__<scheme>__df.json``:
    the artifact stem is everything before ``__<scheme>__df`` and the flux, if the
    stem is ``…__q0_<val>``, from that tag. A stem with no ``q0`` tag is a distinct
    (unlabelled) flux slice and keeps ``q0=None`` — never coerced to a guess.
    """
    if rec.get("artifact") is None:
        scheme = rec.get("scheme", "")
        stem = path.stem  # e.g. cbl_nieuwstadt__lasd__q0_0.02__smagorinsky__df
        suffix = f"__{scheme}__df"
        rec["artifact"] = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    if rec.get("q0") is None:
        m = _Q0_RE.search(rec["artifact"])
        if m:
            rec["q0"] = float(m.group(1))


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
        _backfill_provenance(rec, path)
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
          f"{len(card.rankings)} regimes, {len(card.per_flux)} regime×flux slices, "
          f"{len(card.spreads)} coefficients)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

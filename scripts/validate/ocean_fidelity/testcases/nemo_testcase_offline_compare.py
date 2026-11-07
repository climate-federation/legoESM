#!/usr/bin/env python3
"""Compare two SAVED gate reports offline, with the shared oracle-relative gate.

WHY THIS EXISTS.  ``--compare-to`` applies the shared before/after gate to the
report a run has JUST produced, so comparing two arms means running the second
one twice -- once to produce it, once to compare it.  On OVERFLOW that is a
quarter of an hour of the same arithmetic.  Every input the comparison needs is
already durable: each ``--output`` writes a report and a hashed ``.residuals``
sidecar carrying the per-cell oracle, candidate and residual field of every
certified row.

So this reads two saved pairs and calls the SAME
``ulp_move_gate.compare_gate_reports`` the in-run path calls.  It is not a
second comparison: the verdict, the 2-ulp bar, the first-over-bar criterion and
the plants all come from that shared module.

FAIL CLOSED.  ``load_residual_artifact`` verifies the sidecar's SHA-256 against
the hash the report recorded, so a report cannot be compared against a sidecar
that has drifted from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legoesm.ocean.fidelity.ulp_move_gate import (
    compare_gate_reports,
    comparison_exit_code,
    load_residual_artifact,
    plant_at_bar_to_debt,
    plant_cellwise_comparison,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="the BEFORE report")
    parser.add_argument("candidate", type=Path, help="the AFTER report")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows-matching",
                        help="restrict to rows whose name contains this")
    parser.add_argument(
        "--plant", choices=("worsen-3ulp", "improve", "at-bar-to-debt"),
        help="synthetic violation; every value but 'improve' must turn it red")
    args = parser.parse_args(argv)

    substring = args.rows_matching
    row_filter = None if substring is None else (lambda n: substring in n)
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    reference_fields = load_residual_artifact(reference, args.reference)
    candidate_fields = load_residual_artifact(candidate, args.candidate)
    if args.plant in {"worsen-3ulp", "improve"}:
        candidate_fields = plant_cellwise_comparison(
            candidate_fields, args.plant, row_filter=row_filter)
    elif args.plant == "at-bar-to-debt":
        plant_at_bar_to_debt(candidate, row_filter=row_filter)
    result = compare_gate_reports(
        reference, candidate,
        reference_fields=reference_fields,
        candidate_fields=candidate_fields,
        row_filter=row_filter)
    result.update({
        "reference": str(args.reference), "candidate": str(args.candidate),
        "plant": args.plant, "row_filter_substring": substring,
        "worktree": candidate.get("worktree"),
        "reference_worktree": reference.get("worktree", {}).get("commit"),
        "candidate_worktree": candidate.get("worktree", {}).get("commit"),
    })
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print(f"OFFLINE_ORACLE_RELATIVE_COMPARE {result['status']}: "
          f"rows={result['n_certified_rows_compared']} "
          f"max_worsening_ulps="
          f"{result['largest_oracle_residual_worsening_ulps']:.17g} "
          f"first_over_bar={result['first_over_bar_reference']!r}->"
          f"{result['first_over_bar_candidate']!r} plant={args.plant!r}")
    return comparison_exit_code(result)


if __name__ == "__main__":
    sys.exit(main())

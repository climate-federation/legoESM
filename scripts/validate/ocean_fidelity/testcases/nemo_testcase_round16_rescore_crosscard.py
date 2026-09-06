#!/usr/bin/env python3
"""Re-score Round-14 cross-card artifacts with the row-scale ULP criterion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legoesm.ocean.fidelity.ulp_move_gate import (
    compare_gate_reports,
    load_residual_artifact,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp

GATES = (
    "overflow_stage",
    "lock_stage",
    "overflow_trajectory",
    "lock_trajectory",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path) -> dict:
    results = {}
    for name in GATES:
        before_path = root / "baseline" / f"{name}.json"
        after_path = root / "current" / f"{name}.json"
        before = _load(before_path)
        after = _load(after_path)
        result = compare_gate_reports(
            before,
            after,
            reference_fields=load_residual_artifact(before, before_path),
            candidate_fields=load_residual_artifact(after, after_path),
        )
        result["reference"] = str(before_path)
        result["candidate"] = str(after_path)
        result["plant"] = None
        result["flagged_rows"] = [
            move for move in result["field_moves"]
            if move["n_cells_worse_than_bar"] > 0
        ]
        results[name] = result
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-round16-crosscard-row-scale-rescore-v1",
        "criterion": "cellwise oracle-relative, row-scale float64 ulp",
        "row_scale_ulp_definition": (
            "numpy.spacing(max(max(abs(float64(NEMO_row))), 1.0))"
        ),
        "total_flagged_rows": sum(
            len(result["flagged_rows"]) for result in results.values()
        ),
        "gates": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name in GATES:
        gate = result["gates"][name]
        print(
            f"ORACLE_RELATIVE_COMPARE {gate['status']}: "
            f"gate={name} rows={gate['n_certified_rows_compared']} "
            f"flagged_rows={len(gate['flagged_rows'])} "
            "max_worsening_row_scale_ulps="
            f"{gate['largest_oracle_residual_worsening_ulps']:.17g} "
            f"first_over_bar={gate['first_over_bar_reference']!r}->"
            f"{gate['first_over_bar_candidate']!r} plant=None"
        )
    print(f"ROUND16_RULE12_FLAGGED_ROWS {result['total_flagged_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

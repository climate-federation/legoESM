#!/usr/bin/env python3
"""Round 330: cross-tree Decision-122 gate for the TKE bottom-boundary fix.

The round-329 comparison card is the admitted control.  Round 330 changes one
shared implementation statement, so the two arms necessarily have different
commits.  This wrapper admits exactly those commits and delegates the ladder
comparison (including Decision 139's later-first handling) and the seven-card
decision to the already reviewed round-318/278 gates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round278_d122_landing_gate as r278,
    nemo_testcase_l4_orca2_round318_fold_d122_gate as r318,
)

GateError = r276.GateError
require = r276.require

BASELINE_COMMIT = "a48dd2956c6164257734a71f9b27162436cc28b4"
CANDIDATE_COMMIT = "ddd2821b71947f09fd2e1e5183b7381f9da1f3b9"
MEASUREMENT_FORMAT = "nemo-testcase-l4-orca2-round329-recordice-v1"
CANDIDATE_ARM = "tke_bottom_jpk_outside_solve"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(baseline_path: Path, candidate_path: Path, plant: str) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for name, doc in (("baseline", baseline), ("candidate", candidate)):
        require(doc.get("status") == "PASS_R276_SOURCE_LADDER",
                f"{name} source ladder is not admitted")
        require(doc.get("format") == MEASUREMENT_FORMAT,
                f"{name} measurement format moved")
        require(doc.get("arm") == "recordice",
                f"{name} is not the current comparison card")
    require(baseline["worktree"]["commit"] == BASELINE_COMMIT,
            "baseline commit moved")
    require(candidate["worktree"]["commit"] == CANDIDATE_COMMIT,
            "candidate commit moved")
    require(baseline["rung"] == candidate["rung"], "rung moved")
    require(baseline["label"] == candidate["label"], "label moved")
    require(baseline["selected_statements"] == candidate["selected_statements"],
            "comparison-card statements moved")

    # The inherited classifier requires same-tree arm metadata.  Normalize only
    # those two admission fields in deep copies; ladder rows remain untouched.
    control_doc = copy.deepcopy(baseline)
    candidate_doc = copy.deepcopy(candidate)
    control_doc["arm"] = "control"
    candidate_doc["arm"] = CANDIDATE_ARM
    control_doc["worktree"]["commit"] = CANDIDATE_COMMIT
    candidate_doc["selected_statements"] = [
        *candidate_doc["selected_statements"],
        "TKE full-depth mbkt+1=jpk remains outside the solve",
    ]

    original_read = Path.read_text
    docs = iter((control_doc, candidate_doc))

    def read_normalized(_self, *args, **kwargs):
        return json.dumps(next(docs))

    Path.read_text = read_normalized
    try:
        result = r318.classify_later_first(
            baseline_path, candidate_path, plant=plant)
    finally:
        Path.read_text = original_read
    result.update(
        format="nemo-testcase-l4-orca2-round330-tke-bottom-census-v1",
        baseline_commit=BASELINE_COMMIT,
        candidate_commit=CANDIDATE_COMMIT,
        baseline_sha256=_sha256(baseline_path),
        candidate_sha256=_sha256(candidate_path),
        admission_normalization=(
            "arm names and baseline worktree commit only; ladder rows untouched"),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--plant", choices=("none", "exact-loss"),
                        default="none")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            require(args.decide is None, "choose classify or decide")
            result = classify(*args.classify, plant=args.plant)
            if args.plant != "none":
                require(result["exact_row_losses"],
                        "exact-loss plant stayed green")
                raise GateError("exact-loss plant fired")
        else:
            require(args.decide is not None, "no gate operation selected")
            require(args.plant == "none", "plants apply only to classify")
            result = r278.decide(args.decide)
    except (GateError, r276.decision96.GateError, OSError, KeyError,
            TypeError, ValueError, StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"STATUS {result['status']}")
    return 0 if result.get("d122_c_and_d", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

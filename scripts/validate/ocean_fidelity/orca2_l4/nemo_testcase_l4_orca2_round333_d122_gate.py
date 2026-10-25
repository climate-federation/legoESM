#!/usr/bin/env python3
"""Same-tree Decision-122 arms for the stored tra_ldf Kmm divisor."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round332_rescore_gate as r332,
)

PAIRS = (
    ("rung0", "independent"), ("omt1", "independent"),
    ("omt2", "independent"), ("omt3", "independent"),
    ("omt4", "independent"), ("rung10", "given_nemo_entry"),
    ("rung10", "independent"),
)


class GateError(RuntimeError):
    """The same-tree arm or Decision-122 comparison is incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _legacy(original, counter: list[int]):
    def wrapped(*args, **kwargs):
        counter[0] += 1
        kwargs["redi_divisor_thickness_override"] = None
        return original(*args, **kwargs)
    return wrapped


def _counted(original, counter: list[int]):
    def wrapped(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)
    return wrapped


def measure(rung: str, label: str, arm: str, deck_root: Path,
            twin_a: Path, expect_commit: str) -> dict:
    require((rung, label) in PAIRS, "pair is outside Decision 122")
    require(arm in ("control", "candidate"), "unknown arm")
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    calls = [0]
    original = model_module.gm_redi_tracer_tendency_latlon
    model_module.gm_redi_tracer_tendency_latlon = (
        _legacy(original, calls) if arm == "control"
        else _counted(original, calls))
    try:
        result = r332.measure(rung, label, deck_root, twin_a, expect_commit)
    finally:
        model_module.gm_redi_tracer_tendency_latlon = original
    require((calls[0] > 0) == (rung == "rung10"),
            f"divisor call execution count moved: {calls[0]}")
    result.update({
        "format": "nemo-testcase-l4-orca2-round333-d122-arm-v1",
        "status": "PASS_R333_D122_ARM",
        "arm": arm,
        "legacy_divisor_calls": calls[0],
    })
    return result


def _row(rows: list[dict], key: tuple[int, str, str]) -> dict:
    return next(row for row in rows
                if (int(row["kt"]), row["checkpoint"], row["field"]) == key)


def classify(control_path: Path, candidate_path: Path) -> dict:
    control = json.loads(control_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    require(control.get("status") == candidate.get("status") == "PASS_R333_D122_ARM",
            "arm is not admitted")
    same_nonexecuting = (
        control["arm"] == candidate["arm"] == "candidate"
        and control["rung"] != "rung10"
        and control["legacy_divisor_calls"] == candidate["legacy_divisor_calls"] == 0)
    require((control["arm"] == "control" and candidate["arm"] == "candidate")
            or same_nonexecuting, "arm order moved")
    require((control["rung"], control["label"])
            == (candidate["rung"], candidate["label"]), "pair moved")
    require(control["worktree"]["commit"] == candidate["worktree"]["commit"],
            "arms are not from one committed tree")
    before, after = control["ladder"]["rows"], candidate["ladder"]["rows"]
    require(len(before) == len(after), "row count moved")
    pairs = list(zip(before, after, strict=True))
    require(all((a["kt"], a["checkpoint"], a["field"])
                == (b["kt"], b["checkpoint"], b["field"])
                for a, b in pairs), "row order moved")
    exact_losses = [
        [a["kt"], a["checkpoint"], a["field"]]
        for a, b in pairs if a["bit_identical"] and not b["bit_identical"]
    ]
    directions = Counter(r276.decision96._direction(a, b, "rms")
                         for a, b in pairs)
    old_first = next(row for row in before if row["classification"] == "OVER_BAR")
    target = (int(old_first["kt"]), old_first["checkpoint"], old_first["field"])
    first_direction = r276.decision96._direction(
        old_first, _row(after, target), "rms")
    passed = not exact_losses and first_direction in ("toward", "equal")
    return {
        "format": "nemo-testcase-l4-orca2-round333-d122-census-v1",
        "status": "PASS_R333_D122_CENSUS" if passed else "FAIL_R333_D122_CENSUS",
        "rung": candidate["rung"], "label": candidate["label"],
        "worktree_commit": candidate["worktree"]["commit"],
        "row_count": len(after), "exact_row_losses": exact_losses,
        "first_over_bar": {"target": list(target), "rms_direction": first_direction},
        "registered_census": {name: int(directions[name])
                              for name in ("toward", "away", "equal")},
        "d122_c_and_d": passed,
    }


def decide(paths: list[Path]) -> dict:
    docs = [json.loads(path.read_text()) for path in paths]
    require(sorted((doc["rung"], doc["label"]) for doc in docs) == sorted(PAIRS),
            "seven-pair set is incomplete")
    require(len({doc["worktree_commit"] for doc in docs}) == 1,
            "censuses mix commits")
    require(all(doc["status"] == "PASS_R333_D122_CENSUS"
                and doc["d122_c_and_d"] for doc in docs), "a pair failed")
    return {
        "format": "nemo-testcase-l4-orca2-round333-d122-decision-v1",
        "status": "PASS_R333_D122_DECISION",
        "worktree_commit": docs[0]["worktree_commit"],
        "cards": docs,
        "d122_c_and_d": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung")
    parser.add_argument("--label")
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            result = measure(args.rung, args.label, args.arm, args.deck_root,
                             args.twin_a, args.expect_commit)
        elif args.classify:
            result = classify(*args.classify)
        elif args.decide:
            result = decide(args.decide)
        else:
            raise GateError("no operation selected")
    except (GateError, r332.GateError, OSError, KeyError,
            TypeError, ValueError, StopIteration) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

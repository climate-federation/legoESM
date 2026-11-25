#!/usr/bin/env python3
"""Measure the G1 transfer on shipped ORCA2 against a same-tree OFF control."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_decision96_gate as decision96,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round269_g1_decision96_gate as round269,
)


class GateError(RuntimeError):
    """The shipped-card comparison is incomplete or outside Decision 96."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _rows(trajectory: dict) -> dict:
    rows = []
    first = None
    for checkpoint in trajectory["checkpoints"]:
        if first is None and checkpoint["first_non_bit_field"] is not None:
            first = {
                "kt": checkpoint["kt"],
                "checkpoint": checkpoint["checkpoint"],
                "field": checkpoint["first_non_bit_field"],
            }
        for field, score in checkpoint["rows"].items():
            rows.append({
                "kt": checkpoint["kt"],
                "checkpoint": checkpoint["checkpoint"],
                "field": field,
                **score,
            })
    require(len(rows) == 200, "shipped ORCA2 ladder is not kt=1..10 complete")
    require(first is not None, "shipped ORCA2 ladder has no first non-bit row")
    return {"rows": rows, "first_non_bit_checkpoint": first}


def measure(deck_root: Path, record_root: Path, candidate_path: Path,
            expect_commit: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"] == expect_commit,
            "round-270 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-270 measurement requires production JIT on CPU")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(candidate.get("status") == "LADDER_MEASURED",
            "candidate shipped-card ladder is not admitted")
    _, selected = rung10.card_fields(deck_root)
    control = round269._legacy_card(selected)
    baseline = rung10.candidate_trajectory(
        deck_root, record_root, control, max_step=10, bridge_ssh=True)
    return {
        "format": "nemo-testcase-l4-orca2-round270-g1-rung10-control-v1",
        "status": "PASS_R270_G1_RUNG10_CONTROL",
        "label": "given_nemo_entry",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "disabled_statements": [
            "een_fold_image_v_row", "rk3_stage1_ssh_nemo_literal",
            "barotropic_fold_top_v_face_ssh",
            "barotropic_flux_update_nemo_literal",
            "barotropic_substep_dt_nemo_literal",
            "barotropic_entry_depth_nemo_literal",
        ],
        "baseline": baseline,
    }


def classify(control_path: Path, candidate_path: Path, plant: str) -> dict:
    control = json.loads(control_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(control.get("status") == "PASS_R270_G1_RUNG10_CONTROL",
            "same-tree selector-off control is not admitted")
    require(candidate.get("status") == "LADDER_MEASURED",
            "candidate shipped-card ladder is not admitted")
    before = _rows(control["baseline"])
    after = _rows(candidate["candidate_trajectory"])
    if plant == "exact-loss":
        before = copy.deepcopy(before)
        after = copy.deepcopy(after)
        before["rows"][0]["bit_identical"] = True
        before["rows"][0]["unequal"] = 0
        after["rows"][0]["bit_identical"] = False
        after["rows"][0]["unequal"] = 1
    comparison = decision96._comparison(before, after)
    census = Counter(
        row["metrics"]["rms"]["direction"]
        for row in comparison["moved_rows"])
    counts = {name: int(census[name]) for name in ("toward", "away", "equal")}
    votes = counts["toward"] + counts["away"]
    majority = counts["toward"] * 2 > votes
    first = after["first_non_bit_checkpoint"]
    target = (first["kt"], first["checkpoint"], first["field"])
    direction = decision96._direction(
        decision96._row(before, target), decision96._row(after, target), "rms")
    require(majority, "rung-10 transfer lacks an RMS majority toward NEMO")
    require(direction in ("toward", "equal"), "first non-bit row moved away")
    require(not comparison["bit_identical_losses"], "an exact row left the bar")
    return {
        "format": "nemo-testcase-l4-orca2-round270-g1-rung10-decision96-v1",
        "status": "PASS_R270_G1_RUNG10_DECISION96",
        "label": "given_nemo_entry",
        "comparison": comparison,
        "rms_direction_census": counts,
        "strict_majority_toward": majority,
        "first_non_bit_rms_direction": direction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("none", "exact-loss"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = (
            measure(args.deck_root, args.record_root, args.candidate,
                    args.expect_commit)
            if args.measure else classify(args.control, args.candidate, args.plant)
        )
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            decision96.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

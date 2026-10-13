#!/usr/bin/env python3
"""Measure and classify the selected G1 statements on the ORCA2 OMT ladder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as omt2,
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as omt3,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_decision96_gate as decision96,
)

MODULES = {
    "rung0": rung0, "omt1": omt1, "omt2": omt2, "omt3": omt3, "omt4": omt4,
}
STATUSES = {
    "rung0": "PASS_RUNG0_TEN_STEP_LADDER",
    "omt1": "PASS_R211_OMT1_CARD_AND_LADDERS",
    "omt2": "PASS_R219_OMT2_CARD_AND_LADDERS",
    "omt3": "PASS_R221_OMT3_CARD_AND_LADDERS",
    "omt4": "PASS_R223_OMT4_CARD_AND_LADDERS",
}
LABELS = ("independent", "given_nemo_entry")


class GateError(RuntimeError):
    """The G1 transfer moved outside its registered Decision-96 predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_card(rung: str, deck_root: Path):
    module = MODULES[rung]
    card = getattr(module, f"build_{rung}_card")(deck_root)
    validator = getattr(module, f"validate_{rung}_card")
    if rung in ("rung0", "omt1"):
        validator(card)
    else:
        validator(deck_root, card)
    return card


def _legacy_card(card, *, plant: str = "none"):
    cfg = card.recipe.model_config
    barotropic = cfg.barotropic._replace(
        barotropic_fold_top_v_face_ssh=False,
        barotropic_flux_update_nemo_literal=False,
        barotropic_substep_dt_nemo_literal=False,
        barotropic_entry_depth_nemo_literal=False,
    )
    cfg = cfg._replace(
        een_fold_image_v_row=False,
        rk3_stage1_ssh_nemo_literal=False,
        barotropic=barotropic,
    )
    if plant == "retained-selector":
        cfg = cfg._replace(een_fold_image_v_row=True)
    values = (
        cfg.een_fold_image_v_row,
        cfg.rk3_stage1_ssh_nemo_literal,
        cfg.barotropic.barotropic_fold_top_v_face_ssh,
        cfg.barotropic.barotropic_flux_update_nemo_literal,
        cfg.barotropic.barotropic_substep_dt_nemo_literal,
        cfg.barotropic.barotropic_entry_depth_nemo_literal,
    )
    require(not any(values), f"legacy control retained a G1 selector: {values}")
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def measure(rung: str, deck_root: Path, twin_a: Path, label: str,
            candidate_report: Path, expect_commit: str, plant: str) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    require(rung in MODULES, f"unknown rung {rung}")
    require(label in LABELS, f"unknown label {label}")
    report = json.loads(candidate_report.read_text(encoding="utf-8"))
    require(report.get("status") == STATUSES[rung], "candidate ladder is not admitted")
    candidate_ladder = report if rung == "rung0" else report.get(label)
    require(candidate_ladder is not None, "candidate ladder label is absent")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-269 measurement requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-269 measurement requires production JIT on CPU")

    selected = _selected_card(rung, deck_root)
    card = _legacy_card(selected, plant=plant)
    entry = rung0.assemble_frame(twin_a, 1, 0)
    state = (
        rung0.bridge_entry(card, entry)
        if rung == "rung0" or label == "given_nemo_entry"
        else card.recipe.initial_state
    )
    row_count = int(candidate_ladder["row_count"])
    require(row_count % 20 == 0, "candidate row count is not a complete ladder")
    steps = tuple(range(1, row_count // 20 + 1))
    ladder = omt1._run_ladder(card, twin_a, state, label, steps=steps)
    return {
        "format": "nemo-testcase-l4-orca2-round269-g1-legacy-control-v1",
        "status": "PASS_R269_G1_LEGACY_CONTROL",
        "rung": rung,
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "candidate_report": {
            "path": str(candidate_report), "sha256": _sha256(candidate_report),
        },
        "disabled_statements": [
            "een_fold_image_v_row", "rk3_stage1_ssh_nemo_literal",
            "barotropic_fold_top_v_face_ssh",
            "barotropic_flux_update_nemo_literal",
            "barotropic_substep_dt_nemo_literal",
            "barotropic_entry_depth_nemo_literal",
        ],
        "baseline": ladder,
    }


def _row(document: dict, target: tuple[int, str, str]) -> dict:
    rows, _ = decision96._rows(document)
    for row in rows:
        if (row["kt"], row["checkpoint"], row["field"]) == target:
            return row
    raise GateError(f"missing ladder row {target}")


def classify(baseline_path: Path, candidate_path: Path, *, plant: str) -> dict:
    baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(baseline_doc.get("status") == "PASS_R269_G1_LEGACY_CONTROL",
            "legacy control is not admitted")
    label = baseline_doc["label"]
    require(candidate.get("status") == STATUSES[baseline_doc["rung"]],
            "candidate status moved")
    require(baseline_doc["rung"] == "rung0" or label in candidate,
            "candidate label moved")
    baseline = {label: copy.deepcopy(baseline_doc["baseline"])}
    if plant == "exact-loss":
        base_rows, _ = decision96._rows(baseline)
        cand_rows, _ = decision96._rows(candidate)
        base_rows[0]["bit_identical"] = True
        base_rows[0]["unequal"] = 0
        cand_rows[0]["bit_identical"] = False
        cand_rows[0]["unequal"] = 1
    comparison = decision96._comparison(baseline, candidate)
    counts = {}
    for metric in ("rms", "max_abs"):
        census = Counter(
            row["metrics"][metric]["direction"]
            for row in comparison["moved_rows"])
        counts[metric] = {name: int(census[name]) for name in ("toward", "away", "equal")}
    votes = counts["rms"]["toward"] + counts["rms"]["away"]
    majority = counts["rms"]["toward"] * 2 > votes
    candidate_ladder = candidate if baseline_doc["rung"] == "rung0" else candidate[label]
    first_key = candidate_ladder["first_non_bit_checkpoint"]
    require(first_key is not None, "candidate has no first non-bit checkpoint")
    first_target = (first_key["kt"], first_key["checkpoint"], first_key["field"])
    first_before = _row(baseline, first_target)
    first_after = _row(candidate, first_target)
    first_direction = decision96._direction(first_before, first_after, "rms")
    last_kt = max(row["kt"] for row in decision96._rows(candidate)[0])
    final_target = (last_kt, "stage3", "ssh")
    final_before = _row(baseline, final_target)
    final_after = _row(candidate, final_target)
    ssh_not_worse = float(final_after["max_abs"]) <= float(final_before["max_abs"])
    require(majority, "G1 transfer lacks a strict majority toward NEMO by RMS")
    require(first_direction in ("toward", "equal"), "first non-bit row moved away")
    require(not comparison["bit_identical_losses"], "an exact row left the bar")
    require(ssh_not_worse, "final stage-3 SSH maximum worsened")
    return {
        "format": "nemo-testcase-l4-orca2-round269-g1-decision96-v1",
        "status": "PASS_R269_G1_DECISION96",
        "rung": baseline_doc["rung"], "label": label,
        "comparison": comparison,
        "rms_direction_census": counts["rms"],
        "max_direction_census": counts["max_abs"],
        "strict_majority_toward": majority,
        "first_non_bit_rms_direction": first_direction,
        "final_stage3_ssh_max": {
            "kt": last_kt, "before": final_before["max_abs"],
            "after": final_after["max_abs"], "not_worse": ssh_not_worse,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung", choices=tuple(MODULES))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=("none", "retained-selector", "exact-loss"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.rung, args.deck_root, args.twin_a, args.label,
                         args.candidate_report, args.expect_commit)),
                    "measurement arguments are incomplete")
            require(args.plant in ("none", "retained-selector"), "bad measurement plant")
            result = measure(args.rung, args.deck_root, args.twin_a, args.label,
                             args.candidate_report, args.expect_commit, args.plant)
        else:
            require(args.classify is not None, "classification inputs are required")
            require(args.plant in ("none", "exact-loss"), "bad classification plant")
            result = classify(*args.classify, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, decision96.GateError, OSError, KeyError, TypeError,
            ValueError, StopIteration) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

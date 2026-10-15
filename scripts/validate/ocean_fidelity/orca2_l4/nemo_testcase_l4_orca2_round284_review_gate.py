#!/usr/bin/env python3
"""Round 284: round-282 review corrections and top-F-row D96 gate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round266_rhs_operator_gate as r266,
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round282_g1b_orca2_gate as r282,
)
from nemo_testcase_l2_gyre_phase3_gate import read_stage2_terms  # noqa: E402

LABELS = ("independent", "given_nemo_entry")
STAGE_ARMS = ("area_off", "area_on", "top_f_halo")
FLOOR = np.float64(2.0e-10)


class GateError(RuntimeError):
    """A frozen review correction or top-F predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _stage_card(card, arm: str):
    require(arm in STAGE_ARMS, f"unknown stage arm {arm!r}")
    cfg = card.recipe.model_config
    changed = cfg._replace(
        een_nemo_relative_vorticity=arm != "area_off",
        een_nemo_top_f_u_halo=arm == "top_f_halo",
    )
    return card._replace(recipe=card.recipe._replace(model_config=changed))


def measure_stage(deck_root: Path, frame_root: Path, record_root: Path,
                  label: str, arm: str, expect_commit: str) -> dict:
    require(label in LABELS, f"unknown label {label!r}")
    captured: dict[str, np.ndarray] = {}

    def capture(_card, parts, _state, _passivity, _vertex_mask):
        captured["after_vor_u"] = r266._native_u(parts["after_vor_u"].data)

    report = r266.measure(
        deck_root, frame_root, record_root, label, expect_commit,
        passive_trace_callback=capture,
        card_transform=lambda card: _stage_card(card, arm),
    )
    oracle = read_stage2_terms(
        record_root / "oracle_rkstage2_terms_kt00000001.bin",
        expected_dims=(94, 152, 31))["after_vorticity_u"][..., :30]
    candidate = captured["after_vor_u"]
    target = (19, 33, 3)
    require(candidate.shape == oracle.shape, "after-VOR array shape moved")
    value = np.float64(candidate[target])
    truth = np.float64(oracle[target])
    report.update({
        "format": "nemo-testcase-l4-orca2-round284-stage-boundary-v1",
        "round284_arm": arm,
        "target_19_33_3": {
            "candidate": float(value), "nemo": float(truth),
            "abs_error": float(abs(value - truth)),
        },
    })
    return report


def classify_stage(paths: list[Path], plant: bool = False) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_key = {(r["label"], r["round284_arm"]): r for r in reports}
    require(set(by_key) == {(label, arm) for label in LABELS for arm in STAGE_ARMS},
            "stage-boundary arm/label coverage moved")
    if plant:
        by_key[("independent", "area_on")]["target_19_33_3"]["abs_error"] = (
            by_key[("independent", "area_off")]["target_19_33_3"]["abs_error"])
    rows = {}
    for label in LABELS:
        off = by_key[(label, "area_off")]["target_19_33_3"]
        area = by_key[(label, "area_on")]["target_19_33_3"]
        halo = by_key[(label, "top_f_halo")]
        require(area["abs_error"] < off["abs_error"],
                f"{label}: F-area statement did not move [19,33,3] toward NEMO")
        require(all(halo["passivity"].values()), f"{label}: top-F trace is not passive")
        rows[label] = {
            "area_off": off, "area_on": area,
            "top_f_halo": halo["target_19_33_3"],
            "area_abs_error_delta": area["abs_error"] - off["abs_error"],
            "top_f_boundary": halo["boundary_rows"]["after_vorticity_u"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round284-stage-review-v1",
        "status": "PASS_R284_MISSING_FALSIFIER_AND_TOP_F_BOUNDARY",
        "labels": rows,
        "prediction_ledger": {
            "R284-P3": "CONFIRMED",
            "R284-P5-boundary": "MEASURED",
        },
    }


def classify_ladder(baseline_path: Path, candidate_path: Path,
                    plant: str = "none") -> dict:
    """Apply the unmodified round-276 predicate to A versus A+top-F."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(baseline["arm"] == "npcrv_f_area", "top-F baseline is not A")
    require(candidate["arm"] == "npcrv_f_area_top_f_halo",
            "top-F candidate is not A+C")
    control = copy.deepcopy(baseline)
    control["arm"] = "control"
    control["selected_statements"] = []
    result = r276.classify_documents(control, candidate, plant=plant)
    result["arm"] = "npcrv_top_f_u_halo"
    result["selected_statements"] = [r282.C]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-stage", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--arm", choices=STAGE_ARMS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-stage", nargs=6, type=Path)
    parser.add_argument("--classify-ladder", nargs=2, type=Path)
    parser.add_argument("--plant", choices=("none", "stage", "exact-loss"),
                        default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure_stage:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.arm, args.expect_commit)),
                    "stage measurement arguments incomplete")
            result = measure_stage(args.deck_root, args.frame_root, args.record_root,
                                   args.label, args.arm, args.expect_commit)
        elif args.classify_stage:
            result = classify_stage(args.classify_stage, plant=args.plant == "stage")
        else:
            require(args.classify_ladder is not None, "classification inputs missing")
            result = classify_ladder(
                *args.classify_ladder,
                plant="exact-loss" if args.plant == "exact-loss" else "none")
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r266.GateError, r276.GateError, OSError, KeyError,
            TypeError, ValueError, IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Split round 292's fold-image V-depth boundary on admitted operands."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
)

FLOOR = np.float64(2.0e-10)
OPERAND_ORDER = (
    "hv_0", "r1_e1e2v", "e1e2t", "eta_mid_owned",
    "north_e1e2t_halo", "north_eta_halo", "ssvmask",
)
ARMS = ("baseline", "raw_v_surface_mask", "mask_t_neighbor_pair")
PLANTS = (
    "none", "passivity", "source-order", "replay-calibration",
    "first-operand", "single-arm", "pair-closure", "locality",
    "consumer-closure", "one-bit",
)


class GateError(RuntimeError):
    """An admitted stream, operand split, or falsifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _complete(result: dict[str, object]) -> dict[str, object]:
    return result.get("complete", result)


def _score(left, right, support) -> dict[str, object]:
    return r228._difference(
        np.asarray(left, np.float64)[np.asarray(support, bool)],
        np.asarray(right, np.float64)[np.asarray(support, bool)],
    )


def measure_arm(deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
                expect_commit: str, arm: str) -> dict[str, object]:
    """Measure one arm in a fresh process to bound JAX compilation memory."""

    require(arm in ARMS, f"unknown arm {arm!r}")
    original_substeps = r245.SUBSTEPS
    r245.SUBSTEPS = (1,)
    try:
        report = r245.measure(
            deck_root, frame_root, twin_a, twin_b, "independent",
            expect_commit,
            raw_v_surface_mask=arm != "baseline",
            t_pivot_north_neighbor=arm == "mask_t_neighbor_pair")
    finally:
        r245.SUBSTEPS = original_substeps
    require(report["substeps"] == [1], f"{arm}: substep registry moved")
    require(report["label"] == "independent", f"{arm}: label moved")
    report["round293_arm"] = arm
    return report


def assemble(reports: list[dict[str, object]]) -> dict[str, object]:
    """Combine three independently stamped arm reports without recomputation."""

    require(len(reports) == len(ARMS), "exactly three arm reports are required")
    by_arm = {report.get("round293_arm"): report for report in reports}
    require(set(by_arm) == set(ARMS), "arm report coverage moved")
    for name, report in by_arm.items():
        require(report["substeps"] == [1], f"{name}: substep registry moved")
        require(report["label"] == "independent", f"{name}: label moved")
    commits = {report["worktree"]["commit"] for report in by_arm.values()}
    require(len(commits) == 1, "arm reports were measured on different trees")

    baseline = by_arm["baseline"]
    raw_mask = by_arm["raw_v_surface_mask"]
    pair = by_arm["mask_t_neighbor_pair"]
    base_row = baseline["rows"][0]
    mask_row = raw_mask["rows"][0]
    pair_row = pair["rows"][0]
    support = np.asarray(
        baseline["record_admission"]["defined_masks"]["v"], bool
        if "defined_masks" in baseline["record_admission"] else [], bool)
    # The inherited round-245 report already scores the direct consumer on
    # NEMO's admitted live V-face support.  Keep those source-owned scores and
    # do not reconstruct a second support convention here.
    del support

    return {
        "format": "nemo-testcase-l4-orca2-round293-v-depth-operands-v1",
        "status": "PASS_R293_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1_substep1",
        "execution": "production-jit-cpu-fp64-libm; round245 replay reused",
        "source": {
            "compiled_file": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90"),
            "lines": "509-520",
            "statement": (
                "zhvp2_e=hv_0+0.5*r1_e1e2v*"
                "(e1e2t(j)*zsshp2_e(j)+e1e2t(j+1)*"
                "zsshp2_e(j+1))*ssvmask"),
        },
        "worktree": baseline["worktree"],
        "record_admission": baseline["record_admission"],
        "operand_order": list(OPERAND_ORDER),
        "arms": list(ARMS),
        "passivity": {
            name: report["passivity"] for name, report in by_arm.items()
        },
        "operand_rows": baseline["operand_rows"],
        "baseline_replay_vs_production":
            base_row["baseline_replay_vs_production"],
        "baseline_vs_nemo": base_row["baseline_vs_nemo"],
        "cumulative_substitution": base_row["cumulative_substitution"],
        "direct_arms": {
            "baseline": base_row["pair"],
            "raw_v_surface_mask": mask_row["pair"],
            "mask_t_neighbor_pair": pair_row["pair"],
        },
        "one_bit_control": baseline["one_bit_control"],
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        first = next(iter(report["passivity"]["baseline"]))
        report["passivity"]["baseline"][first] = False
    elif plant == "source-order":
        report["operand_order"][0], report["operand_order"][1] = (
            report["operand_order"][1], report["operand_order"][0])
    elif plant == "replay-calibration":
        report["baseline_replay_vs_production"]["unequal"] = 1
    elif plant == "first-operand":
        report["operand_rows"]["hv_0"]["unequal"] = 0
    elif plant == "single-arm":
        _complete(report["direct_arms"]["raw_v_surface_mask"][
            "face_depth_v_mid"])["unequal"] = 0
    elif plant == "pair-closure":
        _complete(report["direct_arms"]["mask_t_neighbor_pair"][
            "face_depth_v_mid"])["unequal"] = 1
    elif plant == "locality":
        report["direct_arms"]["mask_t_neighbor_pair"][
            "face_depth_v_mid"]["interior"]["max_abs"] = 1.0
    elif plant == "consumer-closure":
        _complete(report["direct_arms"]["mask_t_neighbor_pair"][
            "metric_transport_v"])["unequal"] = 1
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R293_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] ==
            "independent_omt4_kt1_stage1_substep1", "claim label moved")
    require(tuple(report["operand_order"]) == OPERAND_ORDER,
            "compiled operand order moved")
    require(tuple(report["arms"]) == ARMS, "causal-arm registry moved")
    require(all(all(values.values())
                for values in report["passivity"].values()),
            "a diagnostic arm moved the completed state")
    require(report["baseline_replay_vs_production"]["unequal"] == 0,
            "source replay missed the executable face depth")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")

    operand_rows = report["operand_rows"]
    first = next(
        (name for name in ("hv_0", "r1_e1e2v", "e1e2t", "ssvmask")
         if operand_rows[name]["unequal"] > 0), None)
    require(first == "hv_0", f"first direct operand moved: {first}")

    raw_depth = report["direct_arms"]["raw_v_surface_mask"][
        "face_depth_v_mid"]
    pair_depth = report["direct_arms"]["mask_t_neighbor_pair"][
        "face_depth_v_mid"]
    pair_transport = report["direct_arms"]["mask_t_neighbor_pair"][
        "metric_transport_v"]
    require(_complete(raw_depth)["unequal"] > 0,
            "raw V surface mask alone unexpectedly closes the depth")
    require(_complete(pair_depth)["unequal"] == 0,
            "surface-mask/T-neighbour pair does not close the depth")
    require(_complete(pair_transport)["unequal"] == 0,
            "depth pair does not close the immediate V transport")
    require(pair_depth["interior"]["max_abs"] <= FLOOR,
            "paired depth movement is not fold-local")

    report["first_direct_operand"] = first
    report["cancelling_partner"] = {
        "held_statement": "raw hv_0 fold-image face depth",
        "partner": "raw V surface mask plus T-pivot north neighbour",
        "raw_mask_alone_depth_unequal": _complete(raw_depth)["unequal"],
        "pair_depth_unequal": _complete(pair_depth)["unequal"],
        "pair_transport_unequal": _complete(pair_transport)["unequal"],
    }
    report["prediction_ledger"] = {
        "R293-P1": "CONFIRMED",
        "R293-P2": "CONFIRMED",
        "R293-P3": "REFUTED_HV_0_IS_FIRST_NONBIT_OPERAND",
        "R293-P4": "CONFIRMED_MASK_T_NEIGHBOUR_PARTNER",
        "R293-P5": "CONFIRMED",
        "R293-P6": "CONFIRMED",
    }
    report["status"] = "HELD_R293_V_DEPTH_CANCELLING_PARTNER_NAMED"
    require(plant == "none", f"{plant} plant stayed green")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-arm", choices=ARMS)
    parser.add_argument("--assemble", nargs=3, type=Path)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure_arm:
            require(all((args.deck_root, args.frame_root, args.twin_a,
                         args.twin_b, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure_arm(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.expect_commit, args.measure_arm)
        elif args.assemble:
            result = assemble([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.assemble
            ])
        else:
            require(args.report_in is not None, "classification report missing")
            result = classify(json.loads(args.report_in.read_text()),
                              plant=args.plant)
    except (GateError, r245.GateError, OSError, KeyError, TypeError,
            ValueError, IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

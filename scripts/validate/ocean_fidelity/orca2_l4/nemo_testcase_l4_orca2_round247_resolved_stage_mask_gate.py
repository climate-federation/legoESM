#!/usr/bin/env python3
"""Gate round 247's resolved stage mask and complete atomic vector unit."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round241_compact_mask_atomic_gate as r241,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round246_atomic_pair_gate as r246,
)

LABELS = r246.LABELS
PLANTS = (
    "none", "label-coverage", "direct-closure", "completion", "vmask",
    "zfv", "correction-input",
)


class GateError(RuntimeError):
    """The frozen round-247 atomic candidate no longer qualifies."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "label-coverage":
        report["operands"].pop("given_nemo_entry")
    elif plant == "direct-closure":
        report["direct"]["independent"]["rows"][0]["pair"][
            "metric_transport_v"]["complete"]["unequal"] = 1
    elif plant == "completion":
        report["ladders"]["independent"]["last_completed_kt"] = 7
    elif plant == "vmask":
        report["operands"]["independent"]["operand_rows"]["vmask"][
            "unequal"] = 1
    elif plant == "zfv":
        report["operands"]["independent"]["zFv_signature"]["unequal"] = 1
    elif plant == "correction-input":
        report["operands"]["independent"]["correction_input_rows"][
            "vn_adv"]["unequal"] = 1

    base = r246.classify(report)
    require(set(report["operands"]) == set(LABELS),
            "both claim labels are required")
    for label, operands in report["operands"].items():
        for name, row in operands["operand_rows"].items():
            require(row["unequal"] == 0,
                    f"{label}: stage operand {name} is non-bit")
        for name, row in operands["correction_input_rows"].items():
            require(row["unequal"] == 0,
                    f"{label}: correction input {name} is non-bit")
        require(operands["zFv_signature"]["unequal"] == 0,
                f"{label}: completed zFv is non-bit")

    base["format"] = "nemo-testcase-l4-orca2-round247-resolved-stage-mask-v1"
    base["status"] = "PASS_R247_RESOLVED_STAGE_MASK_ATOMIC_KT10"
    base["predictions"] = {
        "R247-P1": "CONFIRMED_STAGE_MASK_AND_ZFV_EXACT_BOTH_LABELS",
        "R247-P2": "CONFIRMED_DIRECT_0_0_0_BOTH_LABELS",
        "R247-P3": "CONFIRMED_OMT4_COMPLETES_KT10_BOTH_LABELS",
        "R247-P4": "PENDING_FULL_DECISION96_GATES",
        "R247-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    return base


def measure_direct(args: argparse.Namespace) -> dict[str, object]:
    return r246.measure_direct(args)


def measure_atomic(args: argparse.Namespace) -> dict[str, object]:
    return r246.measure_atomic(args)


def measure_operands(args: argparse.Namespace) -> dict[str, object]:
    r241._policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.deck_root)
    omt4.validate_omt4_card(args.deck_root, card)
    with r246._resolved_candidate_stage_geometry(card):
        with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True,
        ):
            return r232.measure(
                args.deck_root, args.frame_root, args.record_root,
                args.label, args.expect_commit, resolved_stage_masks=True)


def _load(paths: list[Path]) -> dict[str, dict]:
    return {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in zip(LABELS, paths, strict=True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("direct", "atomic", "operands", "classify"),
        default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--direct-in", nargs=2, type=Path)
    parser.add_argument("--atomic-in", nargs=2, type=Path)
    parser.add_argument("--operands-in", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "direct":
            require(all((args.deck_root, args.frame_root, args.twin_a,
                         args.twin_b, args.label, args.expect_commit)),
                    "direct measurement arguments are incomplete")
            result = measure_direct(args)
        elif args.mode == "atomic":
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "atomic measurement arguments are incomplete")
            result = measure_atomic(args)
        elif args.mode == "operands":
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit)),
                    "operand measurement arguments are incomplete")
            result = measure_operands(args)
        else:
            require(all((args.direct_in, args.atomic_in, args.operands_in)),
                    "classification needs all six section reports")
            result = classify({
                "format": "nemo-testcase-l4-orca2-round247-input-v1",
                "execution": "production-jit-cpu-fp64-x64-libm",
                "direct": _load(args.direct_in),
                "ladders": _load(args.atomic_in),
                "operands": _load(args.operands_in),
            }, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r232.GateError, r241.GateError, r245.GateError, r246.GateError,
            rung0.GateError) as error:
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

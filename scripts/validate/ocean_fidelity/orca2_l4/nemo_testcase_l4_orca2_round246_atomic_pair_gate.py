#!/usr/bin/env python3
"""Score round 246's mask/north-neighbour pair with the complete vector unit."""

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
    nemo_testcase_l4_orca2_round233_geometry_unit_gate as r233,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round241_compact_mask_atomic_gate as r241,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
)

PLANTS = (
    "none", "label-coverage", "direct-closure", "completion", "vn-adv",
    "zfv", "exact-row",
)
LABELS = ("independent", "given_nemo_entry")


class GateError(RuntimeError):
    """The frozen round-246 atomic candidate no longer qualifies."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _complete(row: dict[str, object]) -> dict[str, object]:
    return row.get("complete", row)


def _find_row(ladder: dict[str, object], kt: int, checkpoint: str,
              field: str) -> dict[str, object]:
    rows = [row for row in ladder["rows"] if (
        row["kt"], row["checkpoint"], row["field"]
    ) == (kt, checkpoint, field)]
    require(len(rows) == 1, f"missing unique {kt}/{checkpoint}/{field} row")
    return rows[0]


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "label-coverage":
        report["direct"].pop("given_nemo_entry")
    elif plant == "direct-closure":
        report["direct"]["independent"]["rows"][0]["pair"][
            "metric_transport_v"]["complete"]["unequal"] = 1
    elif plant == "completion":
        report["ladders"]["independent"]["last_completed_kt"] = 7
    elif plant == "vn-adv":
        report["operands"]["independent"]["operand_rows"][
            "vn_adv"]["unequal"] = 919
    elif plant == "zfv":
        report["operands"]["independent"]["operand_rows"][
            "zFv"]["unequal"] = 919
    elif plant == "exact-row":
        report["operands"]["independent"]["operand_rows"][
            "vmask"]["unequal"] = 1

    for section in ("direct", "ladders", "operands"):
        require(set(report[section]) == set(LABELS),
                f"{section}: both claim labels are required")

    direct_table = {}
    for label, result in report["direct"].items():
        require(result["status"] == "PASS_R245_V_DEPTH_OPERAND_SPLIT",
                f"{label}: direct measurement status moved")
        require(result.get("raw_v_surface_mask") is True,
                f"{label}: raw V-only surface mask arm is absent")
        require(result.get("t_pivot_north_neighbor") is True,
                f"{label}: T-pivot north-neighbour arm is absent")
        require(all(result["passivity"].values()),
                f"{label}: direct observer moved completed state")
        rows = []
        for row in result["rows"]:
            values = {}
            for field in ("v_mid", "face_depth_v_mid", "metric_transport_v"):
                value = _complete(row["pair"][field])
                require(value["unequal"] == 0,
                        f"{label}: direct {field} is non-bit at substep "
                        f"{row['substep']}")
                require(value["first_nonfinite"] is None,
                        f"{label}: direct {field} is non-finite")
                values[field] = value
            rows.append({"substep": row["substep"], **values})
        direct_table[label] = rows

    ladder_table = {}
    for label, ladder in report["ladders"].items():
        require(ladder["last_completed_kt"] == 10,
                f"{label}: atomic candidate did not complete kt=10")
        require(ladder["checkpoint_count"] == 40 and ladder["row_count"] == 200,
                f"{label}: ladder coverage moved")
        t_row = _find_row(ladder, 1, "stage1", "T")
        s_row = _find_row(ladder, 1, "stage1", "S")
        require(s_row["max_abs"] <= 0.0012,
                f"{label}: fold-band salinity did not close to the frozen limit")
        require(t_row["max_abs"] <= 0.0014,
                f"{label}: fold-band temperature did not close to the frozen limit")
        ladder_table[label] = {
            "last_completed_kt": ladder["last_completed_kt"],
            "kt1_stage1_T": t_row,
            "kt1_stage1_S": s_row,
            "rows": ladder["rows"],
        }

    operand_table = {}
    for label, operands in report["operands"].items():
        require(operands["status"] == "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
                f"{label}: operand measurement status moved")
        require(all(operands["passivity"].values()),
                f"{label}: operand observer moved completed state")
        rows = operands["operand_rows"]
        for name in ("e3v", "vmask"):
            require(rows[name]["unequal"] == 0,
                    f"{label}: exact operand {name} regressed")
        vn_adv = rows["vn_adv"]
        require(vn_adv["unequal"] <= 35,
                f"{label}: vn_adv did not collapse to the fold-face bound")
        require(rows["zFv"]["unequal"] <= 35,
                f"{label}: zFv did not collapse to the fold-face bound")
        operand_table[label] = {
            "operand_rows": rows,
            "fold_band": operands["fold_band"],
        }

    report["status"] = "PASS_R246_ATOMIC_PAIR_THROUGH_KT10"
    report["summary"] = {
        "direct": direct_table,
        "ladders": ladder_table,
        "operands": operand_table,
    }
    report["predictions"] = {
        "R246-P1": "CONFIRMED_DIRECT_0_0_0_BOTH_LABELS",
        "R246-P2": "CONFIRMED_OMT4_COMPLETES_KT10_BOTH_LABELS",
        "R246-P3": "CONFIRMED_VN_ADV_AND_ZFV_AT_FOLD_FACE_BOUND",
        "R246-P4": "PENDING_FULL_DECISION96_GATES",
        "R246-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    return report


def measure_direct(args: argparse.Namespace) -> dict[str, object]:
    r241._policy_and_stamp(args.expect_commit)
    return r245.measure(
        args.deck_root, args.frame_root, args.twin_a, args.twin_b,
        args.label, args.expect_commit, raw_v_surface_mask=True,
        t_pivot_north_neighbor=True,
    )


def measure_atomic(args: argparse.Namespace) -> dict[str, object]:
    r241._policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.deck_root)
    omt4.validate_omt4_card(args.deck_root, card)
    entry = rung0.assemble_frame(args.frame_root, 1, 0)
    state = (card.recipe.initial_state if args.label == "independent"
             else rung0.bridge_entry(card, entry))
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        return r241._run_atomic_ladder(
            card, args.frame_root, state, args.label)


def measure_operands(args: argparse.Namespace) -> dict[str, object]:
    r241._policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.deck_root)
    omt4.validate_omt4_card(args.deck_root, card)
    with r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True,
    ):
        return r233.measure(
            args.deck_root, args.frame_root, args.record_root,
            args.label, args.expect_commit, production_geometry=False)


def combine(args: argparse.Namespace) -> dict[str, object]:
    def load(paths):
        return {
            label: json.loads(path.read_text(encoding="utf-8"))
            for label, path in zip(LABELS, paths, strict=True)
        }

    return classify({
        "format": "nemo-testcase-l4-orca2-round246-atomic-pair-v1",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "direct": load(args.direct_in),
        "ladders": load(args.atomic_in),
        "operands": load(args.operands_in),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("direct", "atomic", "operands", "combine", "classify"),
        default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--direct-in", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--atomic-in", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--operands-in", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
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
        elif args.mode == "combine":
            require(all((args.direct_in, args.atomic_in, args.operands_in)),
                    "combine needs all six section reports")
            result = combine(args)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(
                json.loads(args.report_in.read_text(encoding="utf-8")),
                plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r232.GateError, r233.GateError, r241.GateError, r245.GateError,
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

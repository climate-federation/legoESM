#!/usr/bin/env python3
"""Score round 246's mask/north-neighbour pair with the complete vector unit."""

from __future__ import annotations

import argparse
import contextlib
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
HELD_PLANTS = (
    "none", "label-coverage", "direct-closure", "boundary", "vmask", "zfv",
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
        report["operands"]["independent"]["correction_input_rows"][
            "vn_adv"]["unequal"] = 919
    elif plant == "zfv":
        report["operands"]["independent"]["zFv_signature"]["unequal"] = 919
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
        require(operands["status"] == "PASS_R232_V_TRANSPORT_OPERAND_SPLIT",
                f"{label}: operand measurement status moved")
        require(all(operands["passivity"].values()),
                f"{label}: operand observer moved completed state")
        require(all(row["unequal"] == 0
                    for row in operands["replay_checks"].values()),
                f"{label}: offline replay missed the production candidate")
        rows = operands["operand_rows"]
        for name in ("e1v", "e3v", "vv", "vmask"):
            require(rows[name]["unequal"] == 0,
                    f"{label}: exact operand {name} regressed")
        vn_adv = operands["correction_input_rows"]["vn_adv"]
        require(vn_adv["unequal"] <= 35,
                f"{label}: vn_adv did not collapse to the fold-face bound")
        require(operands["zFv_signature"]["unequal"] <= 35,
                f"{label}: zFv did not collapse to the fold-face bound")
        operand_table[label] = {
            "operand_rows": rows,
            "correction_input_rows": operands["correction_input_rows"],
            "zFv_signature": operands["zFv_signature"],
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


def classify_held(direct: dict[str, dict], atomic_logs: dict[str, str],
                  operands: dict[str, dict], *, plant: str = "none") -> dict:
    """Fail closed on the preregistered kt=8 refusal and its first open owner."""

    require(plant in HELD_PLANTS, f"unknown held plant {plant!r}")
    direct = copy.deepcopy(direct)
    atomic_logs = copy.deepcopy(atomic_logs)
    operands = copy.deepcopy(operands)
    if plant == "label-coverage":
        operands.pop("given_nemo_entry")
    elif plant == "direct-closure":
        direct["independent"]["rows"][0]["pair"]["metric_transport_v"][
            "complete"]["unequal"] = 1
    elif plant == "boundary":
        atomic_logs["independent"] = atomic_logs["independent"].replace(
            "raw-mesh e3w_int must contain only finite values > 0",
            "different refusal")
    elif plant == "vmask":
        operands["independent"]["operand_rows"]["vmask"]["unequal"] = 0
    elif plant == "zfv":
        operands["independent"]["zFv_signature"]["unequal"] = 0

    for section, values in (
        ("direct", direct), ("atomic logs", atomic_logs), ("operands", operands)
    ):
        require(set(values) == set(LABELS),
                f"{section}: both claim labels are required")

    direct_table = {}
    boundary_table = {}
    operand_table = {}
    refusal = "raw-mesh e3w_int must contain only finite values > 0"
    for label in LABELS:
        report = direct[label]
        require(report["status"] == "PASS_R245_V_DEPTH_OPERAND_SPLIT",
                f"{label}: direct report status moved")
        direct_rows = []
        for row in report["rows"]:
            counts = {
                name: _complete(value)["unequal"]
                for name, value in row["pair"].items()
            }
            require(all(value == 0 for value in counts.values()),
                    f"{label}: direct pair closure moved at substep {row['substep']}")
            direct_rows.append({"substep": row["substep"], **counts})
        direct_table[label] = direct_rows

        log = atomic_logs[label]
        for kt in range(1, 8):
            require(f"PROGRESS round241 {label} kt={kt}" in log,
                    f"{label}: missing completed kt={kt}")
        require(f"PROGRESS round241 {label} kt=8" not in log,
                f"{label}: candidate unexpectedly completed kt=8")
        require(refusal in log and "STATUS REFUSE none:" in log,
                f"{label}: registered live-W refusal moved")
        boundary_table[label] = {
            "last_completed_kt": 7,
            "first_refusing_kt": 8,
            "message": refusal,
        }

        report = operands[label]
        require(report["status"] == "PASS_R232_V_TRANSPORT_OPERAND_SPLIT",
                f"{label}: five-operand report status moved")
        require(all(report["passivity"].values()),
                f"{label}: operand observer moved completed state")
        require(all(row["unequal"] == 0
                    for row in report["replay_checks"].values()),
                f"{label}: offline replay missed the executable")
        rows = report["operand_rows"]
        for name in ("e1v", "e3v", "vv", "zvb"):
            require(rows[name]["unequal"] == 0,
                    f"{label}: upstream operand {name} is no longer exact")
        require(rows["vmask"]["unequal"] == 668
                and rows["vmask"]["max_abs"] == 1.0,
                f"{label}: first stage-mask debt moved")
        correction = report["correction_input_rows"]
        require(all(row["unequal"] == 0 for row in correction.values()),
                f"{label}: external correction is no longer exact")
        require(report["zFv_signature"]["unequal"] == 668,
                f"{label}: zFv debt moved")
        require(report["first_unequal_operand"] == "vmask",
                f"{label}: vmask is no longer the first non-bit operand")
        operand_table[label] = {
            "operand_rows": rows,
            "correction_input_rows": correction,
            "zFv_signature": report["zFv_signature"],
        }

    return {
        "format": "nemo-testcase-l4-orca2-round246-held-v1",
        "status": "HELD_R246_STAGE_VMASK_AND_KT8_LIVE_W",
        "direct": direct_table,
        "boundary": boundary_table,
        "operands": operand_table,
        "first_remaining_statement": {
            "operand": "stage-1 three-dimensional V mask",
            "candidate_boundary": "_nemo_stage_corrected_velocity / zFv product",
            "compiled_source": (
                "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:269-285"),
            "result": "668 mask and zFv slots unequal under both labels",
        },
        "predictions": {
            "R246-P1": "CONFIRMED_DIRECT_0_0_0_BOTH_LABELS",
            "R246-P2": "REFUTED_KT8_LIVE_W_REFUSAL_BOTH_LABELS",
            "R246-P3": "REFUTED_STAGE_VMASK_668_AND_ZFV_668",
            "R246-P4": "NOT_ACTIVATED_PREREQUISITE_R246-P2",
            "R246-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
        "open": (
            "Route the stage transport's three-dimensional V mask through "
            "the production D114 resolved mesh operands, then re-score the "
            "same atomic unit before any landing."),
    }


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


@contextlib.contextmanager
def _resolved_candidate_stage_geometry(card):
    """Route an equivalent rebuilt OMT-4 card through D114's raw operands."""

    import jax.numpy as jnp
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.vertical import (
        nemo_qco_live_face_geometry_cgrid,
        nemo_qco_resolved_mesh_operands,
    )

    original = model_module._nemo_ws_qco_stage_faces
    raw_u, raw_v = r233._raw_masks(card)
    raw_z_coord = card.recipe.z_coord

    def resolved(eta, h_ref, u_mask_3d, v_mask_3d, grid, *,
                 z_coord=None, include_reciprocals=False):
        del u_mask_3d, v_mask_3d, z_coord
        dtype = jnp.asarray(h_ref).dtype
        ops = nemo_qco_resolved_mesh_operands(
            raw_z_coord, grid, raw_u, raw_v, dtype,
            jnp.asarray(h_ref).shape[-1])
        return nemo_qco_live_face_geometry_cgrid(
            jnp.asarray(eta, dtype=dtype), ops.e3u_0, ops.e3v_0,
            ops.umask3, ops.vmask3, ops.hu_0, ops.hv_0,
            ops.area_t, ops.area_u, ops.area_v,
            include_reciprocals=include_reciprocals)

    require(getattr(raw_z_coord, "nemo_een_barotropic", None) is not None,
            "candidate has no raw NEMO geometry bundle")
    model_module._nemo_ws_qco_stage_faces = resolved
    try:
        yield
    finally:
        model_module._nemo_ws_qco_stage_faces = original


def measure_operands(args: argparse.Namespace) -> dict[str, object]:
    r241._policy_and_stamp(args.expect_commit)
    card = omt4.build_omt4_card(args.deck_root)
    omt4.validate_omt4_card(args.deck_root, card)
    with _resolved_candidate_stage_geometry(card):
        with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True,
        ):
            return r232.measure(
                args.deck_root, args.frame_root, args.record_root,
                args.label, args.expect_commit)


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
        "--mode", choices=("direct", "atomic", "operands", "combine",
                           "classify", "classify-held"),
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
    parser.add_argument("--atomic-log-in", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--plant", choices=tuple(dict.fromkeys((*PLANTS, *HELD_PLANTS))),
                        default="none")
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
        elif args.mode == "classify-held":
            require(all((args.direct_in, args.atomic_log_in, args.operands_in)),
                    "held classification needs all six reports/logs")
            result = classify_held(
                {label: json.loads(path.read_text(encoding="utf-8"))
                 for label, path in zip(LABELS, args.direct_in, strict=True)},
                {label: path.read_text(encoding="utf-8")
                 for label, path in zip(LABELS, args.atomic_log_in, strict=True)},
                {label: json.loads(path.read_text(encoding="utf-8"))
                 for label, path in zip(LABELS, args.operands_in, strict=True)},
                plant=args.plant)
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

#!/usr/bin/env python3
"""Gate the kt=1 stage-2 pivot-V entry and its tracer consequence."""

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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round256_fold_origin_gate as r256,
)

LABELS = r249.LABELS
PLANTS = (
    "none", "label-coverage", "production-route", "passivity",
    "stage1-reproduction", "source-order", "pivot-support",
    "association-closure", "stage2-reproduction", "causal-closure",
    "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-260 stage-2 predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _model(card, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)


def _rows(card, values, oracle) -> dict[str, dict[str, object]]:
    rows = {}
    for name, value in zip(("u", "v", "T", "S", "ssh"), values):
        native = r249._native(name, np.asarray(value, np.float64))
        expected = np.asarray(oracle[name], np.float64)
        rows[name] = {
            "complete": r249._score(native, expected),
            "fold_band": r249._score(native[-3:], expected[-3:]),
            "pivot_row": r249._score(native[-1:], expected[-1:]),
        }
    return rows


def measure(deck_root: Path, frame_root: Path, label: str,
            expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSLiveOperandTrace

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    base_hooks = r249._hooks(card, True)
    live_hooks = r249._hooks(card, True, live=True)
    ordinary_model = _model(card, base_hooks)
    live_model = _model(card, live_hooks)

    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
        baseline = jax.device_get(live_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(baseline, _NEMOWSLiveOperandTrace),
            "baseline live trace return type moved")
    passivity = passive._ordinary_state_equal(baseline.state_after, ordinary)
    require(all(passivity.values()), "baseline live trace moved completed state")

    stage1 = baseline.stage_outputs[0]
    associated = r256._associated_stage_values(card, stage1)
    eta1 = stage1[4]
    arm_hooks = live_hooks._replace(
        stage_entry_override=(2, *associated, eta1))
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        arm = jax.device_get(_model(card, arm_hooks).step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(arm, _NEMOWSLiveOperandTrace),
            "causal arm live trace return type moved")

    oracle_stage1 = rung0.assemble_frame(frame_root, 1, 1)
    oracle_stage2 = rung0.assemble_frame(frame_root, 1, 2)
    baseline_entry = (*baseline.stage_states[1][:4], stage1[4])
    arm_entry = (*arm.stage_states[1][:4], stage1[4])
    baseline_stage2 = baseline.stage_outputs[1]
    arm_stage2 = arm.stage_outputs[1]
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round260-stage2-pivot-v1",
        "status": "PASS_R260_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": passivity,
        "source_order": ["stage1_update", "post_stage_lbc_lnk", "stage2_transport", "stage2_tracer"],
        "baseline_stage1": _rows(card, stage1, oracle_stage1),
        "associated_stage1": _rows(card, (*associated, eta1), oracle_stage1),
        "baseline_stage2_entry": _rows(card, baseline_entry, oracle_stage1),
        "associated_stage2_entry": _rows(card, arm_entry, oracle_stage1),
        "baseline_stage2": _rows(card, baseline_stage2, oracle_stage2),
        "arm_stage2": _rows(card, arm_stage2, oracle_stage2),
        "one_bit_control": r249._score(one, np.nextafter(one, np.inf)),
        "compiled_source": {
            "velocity_update": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:468-482",
            "post_stage_association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802",
            "stage2_transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285",
            "stage2_tracer_update": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:674-700",
            "v_fold_rule": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721",
        },
    }


def classify(reports: list[dict[str, object]], *,
             plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "production-route":
        independent["operand_route"] = "retired-route"
    elif plant == "passivity":
        independent["passivity"]["T"] = False
    elif plant == "stage1-reproduction":
        independent["baseline_stage1"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "source-order":
        independent["source_order"][1:3] = reversed(independent["source_order"][1:3])
    elif plant == "pivot-support":
        independent["baseline_stage2_entry"]["v"]["pivot_row"]["support"] = 1
    elif plant == "association-closure":
        independent["associated_stage2_entry"]["v"]["pivot_row"]["max_abs"] = 1.0
    elif plant == "stage2-reproduction":
        independent["baseline_stage2"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "causal-closure":
        independent["arm_stage2"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summary = {}
    expected_order = ["stage1_update", "post_stage_lbc_lnk", "stage2_transport", "stage2_tracer"]
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R260_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-complete-private-unit",
                f"{label}: production route moved")
        require(all(report["passivity"].values()),
                f"{label}: live trace is not passive")
        require(report["source_order"] == expected_order,
                f"{label}: compiled source order moved")
        for name in ("T", "S"):
            require(report["baseline_stage1"][name]["fold_band"]["max_abs"]
                    <= float(r249.FLOOR),
                    f"{label}: stage-1 {name} no longer reproduces round 259")
        raw_v = report["baseline_stage2_entry"]["v"]["pivot_row"]
        fixed_v = report["associated_stage2_entry"]["v"]["pivot_row"]
        require(raw_v["support"] == 5400 and raw_v["max_abs"]
                == 0.027227136128767482,
                f"{label}: stage-2 pivot-V entry boundary moved")
        require(fixed_v["max_abs"] <= float(r249.FLOOR),
                f"{label}: literal V association did not close the entry")
        require(report["baseline_stage2"]["S"]["fold_band"]["max_abs"]
                == 6.817836451535307,
                f"{label}: stage-2 salinity boundary moved")
        for name in ("T", "S"):
            require(report["arm_stage2"][name]["fold_band"]["max_abs"]
                    <= float(r249.FLOOR),
                    f"{label}: stage-2 {name} did not close under the V association")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summary[label] = {
            "baseline_stage2_entry_v": raw_v,
            "associated_stage2_entry_v": fixed_v,
            "baseline_stage2": {name: report["baseline_stage2"][name]["fold_band"] for name in ("T", "S")},
            "arm_stage2": {name: report["arm_stage2"][name]["fold_band"] for name in ("T", "S")},
        }
    return {
        "format": "nemo-testcase-l4-orca2-round260-classification-v1",
        "status": "PASS_R260_POST_STAGE_V_OWNER",
        "labels": summary,
        "predictions": {
            "R260-P1": "CONFIRMED",
            "R260-P2": "CONFIRMED_PIVOT_V_ENTRY",
            "R260-P3": "CONFIRMED_AT_FLOOR",
            "R260-P4": "CONFIRMED_STAGE2_TRACER_CLOSURE",
            "R260-P5": "PENDING_COMPLETE_UNIT_INTEGRATION",
            "R260-P6": "PENDING_FULL_LANDING_GATES",
            "R260-P7": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.deck_root, args.frame_root, args.label,
                             args.expect_commit)
        else:
            require(args.classify is not None,
                    "classification JSONs are required")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

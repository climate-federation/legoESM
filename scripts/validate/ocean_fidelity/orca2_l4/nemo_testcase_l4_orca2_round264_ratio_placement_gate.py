#!/usr/bin/env python3
"""Reconcile OFF, retired, and compiled placements of stage ``r3va``."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    nemo_testcase_l4_orca2_round260_stage2_pivot_gate as r260,
)

LABELS = r249.LABELS
MODES = ("off", "round262", "nemo")
PLANTS = (
    "none", "label-coverage", "mode-coverage", "record-identity",
    "passivity", "off-reproduction", "retired-regression",
    "compiled-transport-isolation", "compiled-recovery", "matched-point",
    "source-order", "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-264 reconciliation predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(card, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)


def _hooks(card, *, live: bool, mode: str):
    require(mode in MODES, f"unknown ratio-placement mode {mode!r}")
    return r249._hooks(card, True, live=live)._replace(
        stage_v_ratio_association=("" if mode == "off" else mode))


def _record_identity(frame_root: Path) -> dict[str, str]:
    paths = []
    for kt_stage in ((1, 0), (1, 1), (1, 2)):
        kt, stage = kt_stage
        matches = sorted(frame_root.glob(
            f"oracle_r84_frame_rank*_kt{kt:08d}_s{stage}.bin"))
        if not matches:
            matches = sorted(frame_root.glob(
                f"*rank*_kt{kt:08d}*stage{stage}*.bin"))
        require(matches, f"missing admitted kt={kt} stage={stage} frame")
        paths.extend(matches)
    return {str(path): _sha256(path) for path in paths}


def _run_mode(card, state, forcing, oracle_stage1, oracle_stage2,
              mode: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSLiveOperandTrace

    ordinary_hooks = _hooks(card, live=False, mode=mode)
    live_hooks = _hooks(card, live=True, mode=mode)
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        ordinary = jax.device_get(_model(card, ordinary_hooks).step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
        traced = jax.device_get(_model(card, live_hooks).step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            f"{mode}: live trace return type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), f"{mode}: live trace moved completed state")

    stage1 = traced.stage_outputs[0]
    associated = r256._associated_stage_values(card, stage1)
    eta1 = stage1[4]
    arm_hooks = live_hooks._replace(
        stage_entry_override=(2, *associated, eta1))
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        arm = jax.device_get(_model(card, arm_hooks).step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(arm, _NEMOWSLiveOperandTrace),
            f"{mode}: matched-entry trace return type moved")

    associated_stage1 = (*associated, eta1)
    return {
        "mode": mode,
        "passivity": passivity,
        "raw_stage1": r260._rows(card, stage1, oracle_stage1),
        "matched_stage1": r260._rows(card, associated_stage1, oracle_stage1),
        "matched_stage2": r260._rows(card, arm.stage_outputs[1], oracle_stage2),
        "stage1_zFv": np.asarray(traced.stage_geometry[0][8], np.float64),
    }


def measure(deck_root: Path, frame_root: Path, label: str,
            expect_commit: str) -> dict[str, object]:
    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    oracle_stage1 = rung0.assemble_frame(frame_root, 1, 1)
    oracle_stage2 = rung0.assemble_frame(frame_root, 1, 2)
    modes = {
        mode: _run_mode(
            card, state, forcing, oracle_stage1, oracle_stage2, mode)
        for mode in MODES
    }
    off_zfv = modes["off"].pop("stage1_zFv")
    for mode in MODES:
        current = off_zfv if mode == "off" else modes[mode].pop("stage1_zFv")
        modes[mode]["stage1_zFv_vs_off"] = r249._score(current, off_zfv)
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round264-ratio-placement-v1",
        "status": "PASS_R264_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "record_identity": _record_identity(frame_root),
        "modes": modes,
        "compiled_order": [
            "dom_qco_r3c_RK3", "scalar_v_fold", "store_stage_ratio",
            "stage2_zvb", "stage2_zFv", "stage2_tracer",
        ],
        "compiled_source": {
            "ratio_build": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217",
            "ratio_association_store": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:163-180",
            "stage2_transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285",
            "fold_rule": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721",
        },
        "one_bit_control": r249._score(one, np.nextafter(one, np.inf)),
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
    if plant == "mode-coverage":
        independent["modes"].pop("nemo")
    elif plant == "record-identity":
        independent["record_identity"][next(iter(independent["record_identity"]))] = "0" * 64
    elif plant == "passivity":
        independent["modes"]["off"]["passivity"]["T"] = False
    elif plant == "off-reproduction":
        independent["modes"]["off"]["matched_stage2"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "retired-regression":
        independent["modes"]["round262"]["matched_stage2"]["S"]["fold_band"]["max_abs"] = 0.0
    elif plant == "compiled-transport-isolation":
        independent["modes"]["nemo"]["stage1_zFv_vs_off"]["unequal"] = 1
    elif plant == "compiled-recovery":
        independent["modes"]["nemo"]["matched_stage2"]["S"]["fold_band"]["max_abs"] = 2.0
    elif plant == "matched-point":
        independent["modes"]["nemo"]["matched_stage1"]["v"]["fold_band"]["max_abs"] = 1.0
    elif plant == "source-order":
        independent["compiled_order"][1:3] = reversed(independent["compiled_order"][1:3])
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    reference_records = by_label[LABELS[0]]["record_identity"]
    expected_order = [
        "dom_qco_r3c_RK3", "scalar_v_fold", "store_stage_ratio",
        "stage2_zvb", "stage2_zFv", "stage2_tracer",
    ]
    summaries = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R264_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-complete-private-unit",
                f"{label}: execution provenance moved")
        require(report["record_identity"] == reference_records,
                f"{label}: admitted record identity moved")
        require(set(report["modes"]) == set(MODES),
                f"{label}: mode coverage moved")
        require(report["compiled_order"] == expected_order,
                f"{label}: compiled source order moved")
        for mode in MODES:
            row = report["modes"][mode]
            require(all(row["passivity"].values()),
                    f"{label}/{mode}: observer moved completed state")
            require(row["matched_stage1"]["v"]["fold_band"]["max_abs"]
                    <= float(r249.FLOOR),
                    f"{label}/{mode}: stage-1 program point is unmatched")

        off = report["modes"]["off"]
        retired = report["modes"]["round262"]
        nemo = report["modes"]["nemo"]
        off_t = off["matched_stage2"]["T"]["fold_band"]["max_abs"]
        off_s = off["matched_stage2"]["S"]["fold_band"]["max_abs"]
        require(off_t == 6.435500132795369e-09
                and off_s == 1.3842782919937235e-08,
                f"{label}: round-260 OFF fingerprint did not reproduce")
        retired_s = retired["matched_stage2"]["S"]["fold_band"]["max_abs"]
        require(retired_s > 1.0 and retired_s > off_s * 1.0e6,
                f"{label}: retired placement did not reproduce O(1) regression")
        require(retired["stage1_zFv_vs_off"]["unequal"] > 0,
                f"{label}: retired placement no longer changes stage-1 transport")
        require(nemo["stage1_zFv_vs_off"]["unequal"] == 0,
                f"{label}: compiled placement changed stage-1 transport")
        for name, off_value in (("T", off_t), ("S", off_s)):
            recovered = nemo["matched_stage2"][name]["fold_band"]["max_abs"]
            require(recovered <= off_value,
                    f"{label}: compiled placement worsened stage-2 {name}")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summaries[label] = {
            mode: {
                "stage1_zFv_vs_off": report["modes"][mode]["stage1_zFv_vs_off"],
                "stage2_T": report["modes"][mode]["matched_stage2"]["T"]["fold_band"],
                "stage2_S": report["modes"][mode]["matched_stage2"]["S"]["fold_band"],
            }
            for mode in MODES
        }
    return {
        "format": "nemo-testcase-l4-orca2-round264-classification-v1",
        "status": "PASS_R264_RATIO_PLACEMENT_RECONCILIATION",
        "labels": summaries,
        "predictions": {
            "R264-P1": "CONFIRMED_ROUND260_OFF_REPRODUCED",
            "R264-P2": "CONFIRMED_RETIRED_PLACEMENT_REGRESSION",
            "R264-P3": "CONFIRMED_COMPILED_PLACEMENT_RECOVERY",
            "R264-P4": "PENDING_PREFIX_MEASUREMENT",
            "R264-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.label, args.expect_commit)
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

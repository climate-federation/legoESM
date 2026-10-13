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
    "passivity", "off-reproduction", "retired-closure",
    "transport-digest", "store-only-inertness", "matched-point",
    "source-order", "one-bit",
)
PREFIX_PLANTS = ("none", "prefix-labels", "prefix-order", "prefix-one-bit")


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


def measure(deck_root: Path, frame_root: Path, label: str, mode: str,
            expect_commit: str) -> dict[str, object]:
    require(label in LABELS, f"bad label {label!r}")
    require(mode in MODES, f"bad placement mode {mode!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    oracle_stage1 = rung0.assemble_frame(frame_root, 1, 1)
    oracle_stage2 = rung0.assemble_frame(frame_root, 1, 2)
    result = _run_mode(
        card, state, forcing, oracle_stage1, oracle_stage2, mode)
    stage1_zfv = result.pop("stage1_zFv")
    result["stage1_zFv_sha256"] = hashlib.sha256(
        np.ascontiguousarray(stage1_zfv).tobytes()).hexdigest()
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round264-ratio-placement-v1",
        "status": "PASS_R264_MEASUREMENT",
        "label": label,
        "mode": mode,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "record_identity": _record_identity(frame_root),
        "result": result,
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


def measure_prefix(deck_root: Path, frame_root: Path, label: str,
                   expect_commit: str) -> dict[str, object]:
    """Run the source-exact coupled placement without a materialising observer."""

    import jax

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    model = _model(card, _hooks(card, live=False, mode="round262"))
    completed = []
    refusal = None
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        for kt in (1, 2):
            try:
                state = jax.device_get(model.step(
                    state, card.dt_s, freshwater=forcing[0],
                    surface_forcing=forcing[1]))
            except ValueError as error:
                refusal = {"kt": kt, "message": str(error)}
                break
            leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]
            completed.append({
                "kt": kt,
                "arrays": len(leaves),
                "nonfinite": int(sum(
                    np.count_nonzero(~np.isfinite(leaf))
                    for leaf in leaves if np.issubdtype(leaf.dtype, np.inexact))),
            })
    return {
        "format": "nemo-testcase-l4-orca2-round264-prefix-v1",
        "status": "PASS_R264_PREFIX_MEASUREMENT",
        "label": label,
        "mode": "round262-coupled-source-placement",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm-no-observer",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "record_identity": _record_identity(frame_root),
        "completed": completed,
        "refusal": refusal,
        "one_bit_control": r249._score(
            np.asarray([1.0]), np.nextafter(np.asarray([1.0]), np.inf)),
    }


def classify_prefix(reports: list[dict[str, object]], *,
                    plant: str = "none") -> dict[str, object]:
    require(plant in PREFIX_PLANTS, f"unknown prefix plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "prefix-labels":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both prefix labels are required")
    independent = by_label["independent"]
    if plant == "prefix-order":
        independent["completed"] = []
    elif plant == "prefix-one-bit":
        independent["one_bit_control"]["unequal"] = 0
    summaries = {}
    reference_records = by_label[LABELS[0]]["record_identity"]
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R264_PREFIX_MEASUREMENT"
                and report["mode"] == "round262-coupled-source-placement",
                f"{label}: prefix provenance moved")
        require(report["record_identity"] == reference_records,
                f"{label}: prefix record identity moved")
        require([row["kt"] for row in report["completed"]] == [1, 2]
                and all(row["nonfinite"] == 0 for row in report["completed"]),
                f"{label}: source-exact candidate did not complete finite kt=1..2")
        require(report["refusal"] is None,
                f"{label}: pre-kt2 live-thickness refusal persists")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: prefix one-bit control did not fire")
        summaries[label] = {
            "completed": report["completed"],
            "refusal": report["refusal"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round264-prefix-classification-v1",
        "status": "PASS_R264_PREFIX_RECONCILIATION",
        "labels": summaries,
        "predictions": {
            "R264-P4": "CONFIRMED_KT1_KT2_FINITE",
            "R264-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def classify(reports: list[dict[str, object]], *,
             plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_key = {(report["label"], report["mode"]): report for report in reports}
    expected_keys = {(label, mode) for label in LABELS for mode in MODES}
    require(set(by_key) == expected_keys,
            "both labels and all three placement modes are required")
    independent = by_key[("independent", "off")]
    if plant == "mode-coverage":
        reports.pop()
        by_key = {(report["label"], report["mode"]): report for report in reports}
    elif plant == "record-identity":
        independent["record_identity"][next(iter(independent["record_identity"]))] = "0" * 64
    elif plant == "passivity":
        independent["result"]["passivity"]["T"] = False
    elif plant == "off-reproduction":
        independent["result"]["matched_stage2"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "retired-closure":
        by_key[("independent", "round262")]["result"]["matched_stage2"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "transport-digest":
        by_key[("independent", "round262")]["result"]["stage1_zFv_sha256"] = "0" * 64
    elif plant == "store-only-inertness":
        by_key[("independent", "nemo")]["result"]["matched_stage2"]["S"]["fold_band"]["max_abs"] = 2.0
    elif plant == "matched-point":
        by_key[("independent", "nemo")]["result"]["matched_stage1"]["v"]["fold_band"]["max_abs"] = 1.0
    elif plant == "source-order":
        independent["compiled_order"][1:3] = reversed(independent["compiled_order"][1:3])
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    require(set(by_key) == expected_keys,
            "both labels and all three placement modes are required")
    reference_records = by_key[(LABELS[0], MODES[0])]["record_identity"]
    expected_order = [
        "dom_qco_r3c_RK3", "scalar_v_fold", "store_stage_ratio",
        "stage2_zvb", "stage2_zFv", "stage2_tracer",
    ]
    summaries = {}
    for label in LABELS:
        for mode in MODES:
            report = by_key[(label, mode)]
            require(report["status"] == "PASS_R264_MEASUREMENT",
                    f"{label}/{mode}: measurement status moved")
            require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                    and report["operand_route"] ==
                    "production-d114-resolved-complete-private-unit",
                    f"{label}/{mode}: execution provenance moved")
            require(report["record_identity"] == reference_records,
                    f"{label}/{mode}: admitted record identity moved")
            require(report["compiled_order"] == expected_order,
                    f"{label}/{mode}: compiled source order moved")
            row = report["result"]
            require(all(row["passivity"].values()),
                    f"{label}/{mode}: observer moved completed state")
            require(row["matched_stage1"]["v"]["fold_band"]["max_abs"]
                    <= float(r249.FLOOR),
                    f"{label}/{mode}: stage-1 program point is unmatched")

        off = by_key[(label, "off")]["result"]
        retired = by_key[(label, "round262")]["result"]
        nemo = by_key[(label, "nemo")]["result"]
        off_t = off["matched_stage2"]["T"]["fold_band"]["max_abs"]
        off_s = off["matched_stage2"]["S"]["fold_band"]["max_abs"]
        require(off_t == 6.435500132795369e-09
                and off_s == 1.3842782919937235e-08,
                f"{label}: round-260 OFF fingerprint did not reproduce")
        for name in ("T", "S"):
            closed = retired["matched_stage2"][name]["fold_band"]["max_abs"]
            require(closed <= float(r249.FLOOR),
                    f"{label}: coupled round-262 placement did not close stage-2 {name}")
        for mode in ("round262", "nemo"):
            require(by_key[(label, mode)]["result"]["stage1_zFv_sha256"]
                    == off["stage1_zFv_sha256"],
                    f"{label}/{mode}: stage-1 transport digest moved")
        for name in ("T", "S"):
            store_only = nemo["matched_stage2"][name]["fold_band"]["max_abs"]
            require(store_only == off["matched_stage2"][name]["fold_band"]["max_abs"],
                    f"{label}: carry-store-only arm is no longer inert for stage-2 {name}")
        for mode in MODES:
            require(by_key[(label, mode)]["one_bit_control"]["unequal"] == 1,
                    f"{label}/{mode}: one-bit control did not fire")
        summaries[label] = {
            mode: {
                "stage1_zFv_sha256": by_key[(label, mode)]["result"]["stage1_zFv_sha256"],
                "stage2_T": by_key[(label, mode)]["result"]["matched_stage2"]["T"]["fold_band"],
                "stage2_S": by_key[(label, mode)]["result"]["matched_stage2"]["S"]["fold_band"],
            }
            for mode in MODES
        }
    return {
        "format": "nemo-testcase-l4-orca2-round264-classification-v1",
        "status": "PASS_R264_RATIO_PLACEMENT_RECONCILIATION",
        "labels": summaries,
        "predictions": {
            "R264-P1": "CONFIRMED_ROUND260_OFF_REPRODUCED",
            "R264-P2": "REFUTED_COUPLED_PLACEMENT_CLOSES_STAGE2",
            "R264-P3": "REFUTED_STORE_ONLY_IS_INERT",
            "R264-P4": "PENDING_PREFIX_MEASUREMENT",
            "R264-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--measure-prefix", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=6, type=Path)
    parser.add_argument("--classify-prefix", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS + PREFIX_PLANTS[1:], default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure_prefix:
            require(args.plant == "none", "prefix measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)),
                    "prefix measurement arguments are incomplete")
            result = measure_prefix(
                args.deck_root, args.frame_root, args.label,
                args.expect_commit)
        elif args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.label, args.mode,
                         args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.label, args.mode,
                args.expect_commit)
        elif args.classify_prefix is not None:
            result = classify_prefix([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify_prefix], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
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

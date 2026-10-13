#!/usr/bin/env python3
"""Split OMT-4's matched kt=1 stage-2 RHS in compiled source order."""

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
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hpg_gate as hpg_gate,
    nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate as r42,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round256_fold_origin_gate as r256,
    nemo_testcase_l4_orca2_round265_stage2_u_gate as r265,
)
from nemo_testcase_l2_gyre_phase3_gate import read_stage2_terms  # noqa: E402
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

LABELS = r249.LABELS
FLOOR = np.float64(2.0e-10)
BOUNDARY_ORDER = ("before_u", "after_hpg_u", "after_vorticity_u", "after_advection_u")
HPG_INPUT_ORDER = ("rhd", "e3w", "gdept_z0", "r1_e1u")
HPG_STATEMENT_ORDER = ("zhpi_u", "zuap_u", "sum_u")
PLANTS = (
    "none", "label-coverage", "record-identity", "passivity", "source-order",
    "round265-reproduction", "first-boundary", "self-replay", "leaf",
    "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-266 measurement predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(candidate, oracle) -> dict[str, object]:
    return r249._score(np.asarray(candidate, np.float64), np.asarray(oracle, np.float64))


def _masked_score(candidate, oracle, active) -> dict[str, object]:
    candidate = np.asarray(candidate, np.float64)
    oracle = np.asarray(oracle, np.float64)
    active = np.asarray(active, bool)
    require(candidate.shape == oracle.shape == active.shape,
            f"masked score shape moved: {candidate.shape} {oracle.shape} {active.shape}")
    require(bool(np.any(active)), "masked score support is empty")
    return _score(candidate[active], oracle[active])


def _native_u(value) -> np.ndarray:
    return r265._native_u(value)


def _owned_hpg(value: np.ndarray, *, three_d: bool = True) -> np.ndarray:
    value = np.asarray(value, np.float64)
    owned = value[2:-2, 2:-2]
    require(owned.shape[:2] == (148, 90), f"HPG owned shape moved: {owned.shape}")
    return owned[..., :30] if three_d else owned


def _candidate_owned(value: np.ndarray, *, three_d: bool = True) -> np.ndarray:
    value = np.asarray(value, np.float64)
    require(value.shape[:2] in ((148, 180), (148, 90)),
            f"candidate shape moved: {value.shape}")
    if value.shape[1] == 180:
        value = value[:, :90]
    return value[..., :30] if three_d else value


def _model(card, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)


def _hooks(card, *, live: bool, stage_entry=None):
    hooks = r265._hooks(card, live=live, stage_entry=stage_entry)
    return hooks


def _associated_stage(card, values):
    return r265._associated_stage(card, values)


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            label: str, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSLiveOperandTrace

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    oracle_stage2 = rung0.assemble_frame(frame_root, 1, 2)
    state = card.recipe.initial_state if label == "independent" else rung0.bridge_entry(card, entry)
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))

    terms_path = record_root / "oracle_rkstage2_terms_kt00000001.bin"
    hpg_operands_path = record_root / "oracle_rkstage2_hpg_operands_kt00000001.bin"
    hpg_literal_path = record_root / "oracle_rkstage2_hpg_literal_kt00000001.bin"
    terms = read_stage2_terms(terms_path, expected_dims=(94, 152, 31))
    hpg_operands = hpg_gate.read_operands(hpg_operands_path)
    hpg_literal = hpg_gate.read_literal(hpg_literal_path)

    arm = r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
    with arm:
        first = jax.device_get(_model(card, _hooks(card, live=True)).step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(first, _NEMOWSLiveOperandTrace), "source trace type moved")
    stage1_associated = _associated_stage(card, first.stage_outputs[0])
    stage_entry = (2, *stage1_associated)

    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        ordinary = jax.device_get(_model(
            card, _hooks(card, live=False, stage_entry=stage_entry)).step(
                state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
        traced = jax.device_get(_model(
            card, _hooks(card, live=True, stage_entry=stage_entry)).step(
                state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(traced, _NEMOWSLiveOperandTrace), "matched trace type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live trace moved completed state")

    parts = traced.operator_operands[1]
    require(parts is not None, "stage-2 operator trace is absent")
    candidate = {
        "before_u": np.zeros_like(terms["before_u"][..., :30]),
        "after_hpg_u": _native_u(parts["after_hpg_u"].data),
        "after_vorticity_u": _native_u(parts["after_vor_u"].data),
        "after_advection_u": _native_u(parts["after_adv_u"].data),
    }
    boundary_rows = {
        name: {
            "complete": _score(candidate[name], terms[name][..., :30]),
            "fold_band": _score(candidate[name][-3:], terms[name][-3:, ..., :30]),
        }
        for name in BOUNDARY_ORDER
    }

    stage2_state = state._replace(
        u=state.u.replace(data=stage1_associated[0]),
        v=state.v.replace(data=stage1_associated[1]),
        T=state.T.replace(data=stage1_associated[2]),
        S=state.S.replace(data=stage1_associated[3]),
        eta=state.eta.replace(data=stage1_associated[4]),
    )
    offline = _model(card, _hooks(card, live=False))
    inputs = r42._candidate_inputs(offline, stage2_state)
    literal = r42._literal_from_inputs(inputs, card.recipe.model_config.g, grid=card.recipe.grid)
    masks = phase3_gate.expected_masks(card)
    tmask = np.asarray(masks["T"], bool)[:, :90, :30]
    umask = np.asarray(masks["u"], bool)[:, :90, :30]
    umask2 = np.any(umask, axis=-1)
    hpg_input_rows = {
        "rhd": _masked_score(_candidate_owned(inputs["rhd"]),
                             _owned_hpg(hpg_operands["rhd"]), tmask),
        "e3w": _masked_score(_candidate_owned(inputs["e3w"]),
                             _owned_hpg(hpg_operands["e3w"]), tmask),
        "gdept_z0": _masked_score(_candidate_owned(inputs["gdept_z0"]),
                                  _owned_hpg(hpg_operands["gdept_z0"]), tmask),
        "r1_e1u": _masked_score(
            _candidate_owned(inputs["r1_e1u"], three_d=False),
            _owned_hpg(hpg_literal["r1_e1u"], three_d=False), umask2),
    }
    hpg_statement_rows = {
        name: _masked_score(
            _candidate_owned(_native_u(literal[name])),
            _owned_hpg(hpg_literal[name]),
            umask,
        )
        for name in HPG_STATEMENT_ORDER
    }
    hpg_live_calibration = _masked_score(
        _candidate_owned(_native_u(literal["sum_u"])),
        _candidate_owned(_native_u(parts["hpg_u"].data)),
        umask,
    )
    hpg_self_replay = hpg_gate.validate(record_root, plant=False)
    stage2_associated = _associated_stage(card, traced.stage_outputs[1])
    matched_stage2 = {
        "T": _score(r249._native("T", stage2_associated[2])[-3:],
                    np.asarray(oracle_stage2["T"], np.float64)[-3:]),
        "S": _score(r249._native("S", stage2_associated[3])[-3:],
                    np.asarray(oracle_stage2["S"], np.float64)[-3:]),
    }
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round266-rhs-operator-v1",
        "status": "PASS_R266_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": passivity,
        "boundary_order": list(BOUNDARY_ORDER),
        "boundary_rows": boundary_rows,
        "hpg_input_order": list(HPG_INPUT_ORDER),
        "hpg_statement_order": list(HPG_STATEMENT_ORDER),
        "hpg_input_rows": hpg_input_rows,
        "hpg_statement_rows": hpg_statement_rows,
        "hpg_live_calibration": hpg_live_calibration,
        "hpg_record_self_replay": hpg_self_replay,
        "matched_stage2": matched_stage2,
        "record_identity": {
            str(terms_path): _sha256(terms_path),
            str(hpg_operands_path): _sha256(hpg_operands_path),
            str(hpg_literal_path): _sha256(hpg_literal_path),
        },
        "compiled_source": {
            "operators": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-421",
            "advection": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynadv.f90:134-139",
            "hpg": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:404-464",
        },
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
    }


def _first(rows: dict[str, dict[str, object]], order) -> str | None:
    return next((name for name in order if rows[name]["max_abs"] > FLOOR), None)


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both labels are required")
    independent = by_label["independent"]
    if plant == "record-identity":
        independent["record_identity"][next(iter(independent["record_identity"]))] = "0" * 64
    elif plant == "passivity":
        independent["passivity"][next(iter(independent["passivity"]))] = False
    elif plant == "source-order":
        independent["boundary_order"][1:3] = reversed(independent["boundary_order"][1:3])
    elif plant == "round265-reproduction":
        independent["matched_stage2"]["S"]["max_abs"] = 1.0
    elif plant == "first-boundary":
        independent["boundary_rows"]["after_hpg_u"]["complete"]["max_abs"] = 0.0
    elif plant == "self-replay":
        independent["hpg_record_self_replay"]["result"] = "DEBT"
    elif plant == "leaf":
        independent["hpg_input_rows"]["rhd"]["max_abs"] = 0.0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    identity = by_label[LABELS[0]]["record_identity"]
    firsts = {}
    summaries = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R266_MEASUREMENT", f"{label}: status moved")
        require(report["record_identity"] == identity, f"{label}: record identity moved")
        require(all(report["passivity"].values()), f"{label}: trace is not passive")
        require(report["boundary_order"] == list(BOUNDARY_ORDER), f"{label}: source order moved")
        require(report["hpg_input_order"] == list(HPG_INPUT_ORDER)
                and report["hpg_statement_order"] == list(HPG_STATEMENT_ORDER),
                f"{label}: HPG order moved")
        require(all(report["matched_stage2"][name]["max_abs"] <= FLOOR for name in ("T", "S")),
                f"{label}: round-265 matched state moved")
        require(report["hpg_live_calibration"]["max_abs"] <= FLOOR,
                f"{label}: offline HPG does not reproduce the live component")
        require(report["hpg_record_self_replay"]["result"] == "AT_BAR",
                f"{label}: recorded HPG does not self-replay")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        complete = {name: report["boundary_rows"][name]["complete"] for name in BOUNDARY_ORDER}
        first = _first(complete, BOUNDARY_ORDER)
        require(first is not None, f"{label}: no RHS boundary")
        firsts[label] = first
        leaf_order = HPG_INPUT_ORDER + HPG_STATEMENT_ORDER
        leaf_rows = {**report["hpg_input_rows"], **report["hpg_statement_rows"]}
        leaf = _first(leaf_rows, leaf_order) if first == "after_hpg_u" else None
        require(first != "after_hpg_u" or leaf is not None,
                f"{label}: HPG moved but no HPG leaf moved")
        summaries[label] = {
            "first_non_floor": first,
            "first_hpg_leaf": leaf,
            "boundary_rows": report["boundary_rows"],
            "hpg_input_rows": report["hpg_input_rows"],
            "hpg_statement_rows": report["hpg_statement_rows"],
        }
    require(len(set(firsts.values())) == 1, "labels disagree on first boundary")
    leafs = {summaries[label]["first_hpg_leaf"] for label in LABELS}
    require(len(leafs) == 1, "labels disagree on HPG leaf")
    first = firsts[LABELS[0]]
    leaf = summaries[LABELS[0]]["first_hpg_leaf"]
    return {
        "format": "nemo-testcase-l4-orca2-round266-rhs-classification-v1",
        "status": "PASS_R266_RHS_OPERATOR_SPLIT",
        "first_non_floor": first,
        "first_hpg_leaf": leaf,
        "labels": summaries,
        "predictions": {
            "R266-P1": "CONFIRMED",
            "R266-P2": "CONFIRMED_HPG_FIRST" if first == "after_hpg_u" else "REFUTED",
            "R266-P3": "MEASURED_HPG_LEAF" if leaf is not None else "NOT_REACHED",
            "R266-P4": "NOT_REACHED_NO_SOURCE_ARM",
            "R266-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification reports required")
            result = classify(
                [json.loads(path.read_text(encoding="utf-8")) for path in args.classify],
                plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, hpg_gate.GateError, OSError, KeyError,
            TypeError, ValueError, IndexError) as error:
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

#!/usr/bin/env python3
"""Walk OMT-4's matched kt=1 stage-2 U update in compiled order."""

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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round256_fold_origin_gate as r256,
)
from nemo_testcase_l2_gyre_round11_composition import (  # noqa: E402
    read_stage2_composition,
    read_stage2_preupdate,
)

LABELS = r249.LABELS
FLOOR = np.float64(2.0e-10)
ORDER = (
    "Kbb_u", "Kmm_u", "Krhs_u", "raw_Kaa_u", "barotropic_Kaa_u",
    "post_mean_Kaa_u", "post_lbc_Kaa_u",
)
PLANTS = (
    "none", "label-coverage", "prefix-coverage", "record-identity",
    "passivity", "round264-reproduction", "source-order", "first-owner",
    "rhs-closure", "matched-point", "prefix-finite", "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-265 U-update predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_u(values) -> np.ndarray:
    values = np.asarray(values, np.float64)
    require(values.shape[0:2] == (148, 181),
            f"redundant U shape moved: {values.shape}")
    return values[:, 1:91, ...]


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _rows(candidate: dict[str, np.ndarray], oracle: dict[str, np.ndarray]) -> dict:
    rows = {}
    for name in ORDER:
        left = np.asarray(candidate[name], np.float64)
        right = np.asarray(oracle[name], np.float64)
        if left.ndim == 3:
            right = right[..., :left.shape[-1]]
        rows[name] = {
            "complete": _score(left, right),
            "fold_band": _score(left[-3:], right[-3:]),
        }
    return rows


def _model(card, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)


def _hooks(card, *, live: bool, stage_entry=None, rhs_override=None):
    hooks = r249._hooks(card, True, live=live)
    # The round-264 arm was promoted into production and its private hook was
    # subsequently removed.  Retain compatibility with the admitted probe on
    # older trees without inventing a replacement hook on the current tree.
    if "stage_v_ratio_association" in hooks._fields:
        hooks = hooks._replace(stage_v_ratio_association=True)
    if stage_entry is not None:
        hooks = hooks._replace(stage_entry_override=stage_entry)
    if rhs_override is not None:
        hooks = hooks._replace(stage2_momentum_rhs_override=rhs_override)
    return hooks


def _associated_stage(card, values):
    associated = r256._associated_stage_values(card, values)
    return (*associated, values[4])


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            label: str, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSLiveOperandTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    oracle_stage1 = rung0.assemble_frame(frame_root, 1, 1)
    oracle_stage2 = rung0.assemble_frame(frame_root, 1, 2)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))

    composition_path = record_root / "oracle_rkstage2_operands_kt00000001.bin"
    preupdate_path = record_root / "oracle_rkstage2_preupdate_kt00000001.bin"
    oracle = read_stage2_composition(
        composition_path, expected_dims=(94, 152, 31))
    preupdate = read_stage2_preupdate(
        preupdate_path, expected_dims=(94, 152, 31))

    arm = r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
    with arm:
        first = jax.device_get(_model(card, _hooks(card, live=True)).step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(first, _NEMOWSLiveOperandTrace),
            "source-placement trace type moved")
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
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "matched trace type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "matched trace moved the completed state")

    stage2_associated = _associated_stage(card, traced.stage_outputs[1])
    candidate = {
        "Kbb_u": _native_u(state.u.data),
        "Kmm_u": _native_u(traced.stage_states[1][0]),
        "Krhs_u": _native_u(traced.stage_rhs[1][0]),
        "raw_Kaa_u": _native_u(traced.stage_raw_velocities[1][0]),
        "barotropic_Kaa_u": _native_u(
            jnp.asarray(traced.barotropic_targets[0])[..., None])[..., 0],
        "post_mean_Kaa_u": _native_u(traced.stage_outputs[1][0]),
        "post_lbc_Kaa_u": _native_u(stage2_associated[0]),
    }
    oracle_rows = {
        **oracle, **preupdate,
        "post_lbc_Kaa_u": np.asarray(oracle_stage2["u"], np.float64)[:, :90],
    }
    rows = _rows(candidate, oracle_rows)

    rhs_u = jnp.asarray(traced.stage_rhs[1][0])
    rhs_u = rhs_u.at[:, 1:91, :].set(
        jnp.asarray(preupdate["Krhs_u"][..., :30]))
    rhs_override = (rhs_u, jnp.asarray(traced.stage_rhs[1][1]))
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        rhs_arm = jax.device_get(_model(
            card, _hooks(card, live=True, stage_entry=stage_entry,
                         rhs_override=rhs_override)).step(
                state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(rhs_arm, _NEMOWSLiveOperandTrace),
            "RHS substitution trace type moved")
    rhs_stage2_associated = _associated_stage(card, rhs_arm.stage_outputs[1])
    rhs_closure = {
        "raw_Kaa_u": _score(
            _native_u(rhs_arm.stage_raw_velocities[1][0]),
            oracle["raw_Kaa_u"][..., :30]),
        "post_mean_Kaa_u": _score(
            _native_u(rhs_arm.stage_outputs[1][0]),
            oracle["post_mean_Kaa_u"][..., :30]),
        "post_lbc_Kaa_u": _score(
            _native_u(rhs_stage2_associated[0]),
            oracle_stage2["u"][:, :90]),
    }
    matched_t = r249._score(
        r249._native("T", stage2_associated[2])[-3:],
        np.asarray(oracle_stage2["T"], np.float64)[-3:])
    matched_s = r249._score(
        r249._native("S", stage2_associated[3])[-3:],
        np.asarray(oracle_stage2["S"], np.float64)[-3:])
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round265-stage2-u-v1",
        "status": "PASS_R265_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": passivity,
        "source_order": list(ORDER),
        "rows": rows,
        "rhs_closure": rhs_closure,
        "matched_stage2": {"T": matched_t, "S": matched_s},
        "record_identity": {
            str(composition_path): _sha256(composition_path),
            str(preupdate_path): _sha256(preupdate_path),
            "stage1": hashlib.sha256(np.ascontiguousarray(
                oracle_stage1["u"]).tobytes()).hexdigest(),
            "stage2": hashlib.sha256(np.ascontiguousarray(
                oracle_stage2["u"]).tobytes()).hexdigest(),
        },
        "compiled_source": {
            "ratio_association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:163-180",
            "rhs": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:326-363",
            "update": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:365-370",
            "barotropic_correction": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:433-447",
            "association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802",
        },
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
    }


def classify(reports: list[dict[str, object]], prefixes: list[dict[str, object]],
             *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports, prefixes = copy.deepcopy(reports), copy.deepcopy(prefixes)
    if plant == "label-coverage":
        reports.pop()
    elif plant == "prefix-coverage":
        prefixes.pop()
    by_label = {report["label"]: report for report in reports}
    by_prefix = {report["label"]: report for report in prefixes}
    require(set(by_label) == set(LABELS), "both measurement labels are required")
    require(set(by_prefix) == set(LABELS), "both prefix labels are required")
    independent = by_label["independent"]
    if plant == "record-identity":
        independent["record_identity"]["stage1"] = "0" * 64
    elif plant == "passivity":
        independent["passivity"]["T"] = False
    elif plant == "round264-reproduction":
        independent["matched_stage2"]["S"]["max_abs"] = 1.0
    elif plant == "source-order":
        independent["source_order"][2:4] = reversed(
            independent["source_order"][2:4])
    elif plant == "first-owner":
        independent["rows"]["Krhs_u"]["fold_band"]["max_abs"] = 0.0
    elif plant == "rhs-closure":
        independent["rhs_closure"]["raw_Kaa_u"]["max_abs"] = 1.0
    elif plant == "matched-point":
        independent["rows"]["post_lbc_Kaa_u"]["fold_band"]["max_abs"] = 0.0
    elif plant == "prefix-finite":
        by_prefix["independent"]["completed"] = by_prefix["independent"]["completed"][:1]
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    identity = by_label[LABELS[0]]["record_identity"]
    summaries = {}
    firsts = {}
    for label in LABELS:
        report = by_label[label]
        prefix = by_prefix[label]
        require(report["status"] == "PASS_R265_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-complete-private-unit",
                f"{label}: provenance moved")
        require(report["record_identity"] == identity,
                f"{label}: admitted record identity moved")
        require(all(report["passivity"].values()),
                f"{label}: live trace moved completed state")
        require(report["source_order"] == list(ORDER),
                f"{label}: compiled source order moved")
        for name in ("T", "S"):
            require(report["matched_stage2"][name]["max_abs"] <= FLOOR,
                    f"{label}: round-264 stage-2 {name} closure moved")
        first = next((name for name in ORDER
                      if report["rows"][name]["fold_band"]["max_abs"] > FLOOR), None)
        require(first is not None, f"{label}: no valued stage-2 U boundary")
        firsts[label] = first
        require(first == "Krhs_u", f"{label}: completed RHS is not first ({first})")
        require(report["rhs_closure"]["raw_Kaa_u"]["max_abs"] <= FLOOR,
                f"{label}: recorded RHS did not close raw stage-2 U")
        require(report["rows"]["post_lbc_Kaa_u"]["fold_band"]["max_abs"]
                == 0.003382014663707235,
                f"{label}: matched completed-U boundary moved")
        require(prefix["status"] == "PASS_R264_PREFIX_MEASUREMENT"
                and [row["kt"] for row in prefix["completed"]] == [1, 2]
                and all(row["nonfinite"] == 0 for row in prefix["completed"])
                and prefix["refusal"] is None,
                f"{label}: source-complete prefix is not finite through kt=2")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summaries[label] = {
            "first_non_floor": first,
            "rows": report["rows"],
            "rhs_closure": report["rhs_closure"],
            "matched_stage2": report["matched_stage2"],
            "prefix": prefix["completed"],
        }
    require(len(set(firsts.values())) == 1,
            "claim labels disagree on first U operand")
    rhs_final = [
        summaries[label]["rhs_closure"]["post_lbc_Kaa_u"]["max_abs"]
        for label in LABELS
    ]
    closed = all(value <= FLOOR for value in rhs_final)
    return {
        "format": "nemo-testcase-l4-orca2-round265-stage2-u-classification-v1",
        "status": "PASS_R265_STAGE2_U_BOUNDARY",
        "first_non_floor": firsts[LABELS[0]],
        "labels": summaries,
        "predictions": {
            "R265-P1": "CONFIRMED",
            "R265-P2": "CONFIRMED_COMPLETED_RHS_BOUNDARY",
            "R265-P3": (
                "PENDING_RHS_OPERATOR_SPLIT_BOUNDARY_CLOSES"
                if closed else "REFUTED_CANCELLING_UNIT_REMAINS"),
            "R265-P4": (
                "NOT_REACHED_RHS_OPERATOR_SPLIT_REQUIRED"
                if closed else "NOT_REACHED_CANCELLING_UNIT"),
            "R265-P5": "CONFIRMED" if plant == "none" else "PLANT",
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
    parser.add_argument("--prefixes", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.label, args.expect_commit)
        else:
            require(args.classify is not None and args.prefixes is not None,
                    "classification and prefix reports are required")
            result = classify(
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.classify],
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.prefixes], plant=args.plant)
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

#!/usr/bin/env python3
"""Classify OMT-4's external-mode debt from the admitted passive streams."""

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
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1_gate,
    nemo_testcase_l4_orca2_round15_barotropic_solver_gate as r15,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round229_vector_unit_bisect as r229,
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean, read_ordered  # noqa: E402

FLOOR = np.float64(2.0e-10)
MEASURE_PLANTS = ("none", "observer-bit", "source-order")
CLASSIFY_PLANTS = ("none", "label-coverage", "owner-class")
STREAMS = (
    ("substeps", "oracle_bt_substeps_kt00000001.bin"),
    ("ordered", "oracle_bt_ordered_operands_kt00000001.bin"),
    ("drag", "oracle_bt_drag_operands_kt00000001.bin"),
    ("advmean", "oracle_bt_advmean_operands_kt00000001.bin"),
)
SOURCE_ORDER = (
    "slow_u", "slow_v",
    "eta_entry", "u_entry", "v_entry",
    "eta_mid", "u_mid", "v_mid",
    "face_depth_u_mid", "face_depth_v_mid",
    "metric_transport_u", "metric_transport_v",
    "eta_exit",
    "sum_u_entry", "sum_v_entry", "sum_u_exit", "sum_v_exit",
    "eta_pgf", "pgf_u", "pgf_v", "cor_u", "cor_v",
    "trd_u", "trd_v", "u_exit", "v_exit",
)


class GateError(RuntimeError):
    """The admitted record cannot support the frozen round-235 claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native(value, face: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if face == "u":
        return r97._native_u(value)
    if face == "v":
        return r97._native_v(value)
    require(face == "t", f"unknown face {face!r}")
    return value


def _regional_row(candidate, oracle) -> dict[str, object]:
    """Reuse the exact comparator, then split interior from the fold band."""

    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(candidate.shape == oracle.shape,
            f"regional comparison shape mismatch {candidate.shape} != {oracle.shape}")
    require(candidate.ndim == 2 and candidate.shape[0] == 148,
            f"regional comparison expects one native ORCA2 slab, got {candidate.shape}")
    return {
        "complete": r228._difference(candidate, oracle),
        "interior": r228._difference(candidate[:-3], oracle[:-3]),
        "fold_band": r228._difference(candidate[-3:], oracle[-3:]),
    }


def _admit_streams(twin_a: Path, twin_b: Path) -> list[dict[str, object]]:
    masks = o1_gate._defined_masks(twin_a)
    rows = []
    for family, name in STREAMS:
        left, right = twin_a / name, twin_b / name
        require(left.is_file() and right.is_file(), f"missing inherited {name}")
        left_header = phase1._parse_bt(left, family)
        right_header = phase1._parse_bt(right, family)
        exact = o1_gate._compare_hygiene_record(left, right, masks)
        require(exact["status"] == "EXACT_DEFINED_BYTES",
                f"{name}: twin defined payload differs")
        rows.append({
            "family": family,
            "name": name,
            "left": left_header,
            "right": right_header,
            "defined_twin_status": exact["status"],
            "defined_f64": exact["defined_f64"],
        })
    return rows


def _source_registry(trace, substeps, advmean):
    """Map one shared candidate trace onto the two admitted oracle streams."""

    registry = (
        ("slow_u", "slow_u", substeps["slow_u"], "u"),
        ("slow_v", "slow_v", substeps["slow_v"], "v"),
        ("eta_entry", "eta_entry", substeps["eta_entry"], "t"),
        ("u_entry", "u_entry", substeps["u_entry"], "u"),
        ("v_entry", "v_entry", substeps["v_entry"], "v"),
        ("eta_mid", "eta_mid", substeps["eta_mid"], "t"),
        ("u_mid", "u_mid", substeps["u_mid"], "u"),
        ("v_mid", "v_mid", substeps["v_mid"], "v"),
        ("face_depth_u_mid", "transport_face_depth_u", advmean["face_depth_u"], "u"),
        ("face_depth_v_mid", "transport_face_depth_v", advmean["face_depth_v"], "v"),
        ("metric_transport_u", "transport_metric_u", advmean["metric_u"], "u"),
        ("metric_transport_v", "transport_metric_v", advmean["metric_v"], "v"),
        ("eta_exit", "eta_exit", substeps["eta_exit"], "t"),
        ("sum_u_entry", "transport_sum_u_entry", advmean["sum_u_entry"], "u"),
        ("sum_v_entry", "transport_sum_v_entry", advmean["sum_v_entry"], "v"),
        ("sum_u_exit", "transport_sum_u_exit", advmean["sum_u_exit"], "u"),
        ("sum_v_exit", "transport_sum_v_exit", advmean["sum_v_exit"], "v"),
        ("eta_pgf", "eta_pgf", substeps["eta_pgf"], "t"),
        ("pgf_u", "pgf_u", substeps["pgf_u"], "u"),
        ("pgf_v", "pgf_v", substeps["pgf_v"], "v"),
        ("cor_u", "cor_u", substeps["cor_u"], "u"),
        ("cor_v", "cor_v", substeps["cor_v"], "v"),
        ("trd_u", "trd_u", substeps["trd_u"], "u"),
        ("trd_v", "trd_v", substeps["trd_v"], "v"),
        ("u_exit", "u_exit", substeps["u_exit"], "u"),
        ("v_exit", "v_exit", substeps["v_exit"], "v"),
    )
    require(tuple(row[0] for row in registry) == SOURCE_ORDER,
            "source registry moved")
    rows = []
    for substep in range(65):
        for order, (name, trace_name, oracle, face) in enumerate(registry):
            rows.append({
                "substep": substep + 1,
                "source_order": order,
                "name": name,
                "face": face,
                **_regional_row(
                    _native(trace[trace_name][substep], face),
                    np.asarray(oracle[substep], dtype=np.float64)),
            })
    return rows


def _first(rows, region: str):
    for row in rows:
        maximum = row[region]["max_abs"]
        if maximum is None or not np.isfinite(maximum) or maximum > FLOOR:
            return row
    return None


def measure(deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
            label: str, expect_commit: str, *, plant: str = "none") -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSBoundaryAssociationTrace,
        _NEMOWSRK3TestHooks,
    )

    require(plant in MEASURE_PLANTS, f"unknown plant {plant!r}")
    require(label in ("independent", "given_nemo_entry"), f"bad label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-235 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-235 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-235 measurement requires production JIT on CPU")

    admission = _admit_streams(twin_a, twin_b)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)
    advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    require(ordered["header"]["nrows"] == 2, "ordered record coverage moved")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    slow_override, raw_vmask = r229._slow_override(
        card, state, freshwater, surface)

    def run(expose: bool):
        hooks = _NEMOWSRK3TestHooks(
            expose_barotropic_substeps=expose,
            expose_barotropic_boundary_association=expose,
            barotropic_slow_forcing_override=slow_override,
            barotropic_vector_update_v_mask_override=raw_vmask,
            barotropic_atomic_fold_unit=True,
        )
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    ordinary = run(False)
    observed = run(True)
    require(isinstance(observed, _NEMOWSBoundaryAssociationTrace),
            "barotropic trace return type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary)
    if plant == "observer-bit":
        passivity["eta"] = False
    require(all(passivity.values()), "barotropic observer moved completed state")

    source_order = list(SOURCE_ORDER)
    if plant == "source-order":
        source_order[0], source_order[1] = source_order[1], source_order[0]
    require(tuple(source_order) == SOURCE_ORDER, "source-order plant fired")
    rows = _source_registry(observed.substeps, substeps, advmean)
    first_interior = _first(rows, "interior")
    first_fold = _first(rows, "fold_band")
    first_complete = _first(rows, "complete")
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round235-omt4-substep-v1",
        "status": "PASS_R235_OMT4_SUBSTEP_TABLE",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR),
        "record_admission": admission,
        "passivity": passivity,
        "source_order": source_order,
        "rows": rows,
        "first_complete": first_complete,
        "first_interior": first_interior,
        "first_fold_band": first_fold,
        "record_coverage": {
            "substeps": 65,
            "ordered_full_operand_substeps": 2,
            "exit_depth_after_substep_2": "UNMEASURED_WITH_SPEC",
        },
    }


def classify(reports: list[dict], *, plant: str = "none") -> dict:
    require(plant in CLASSIFY_PLANTS, f"unknown classify plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "claim-label coverage moved")

    endpoint = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R235_OMT4_SUBSTEP_TABLE",
                f"{label}: measurement status moved")
        require(all(report["passivity"].values()),
                f"{label}: passive trace moved")
        require(tuple(report["source_order"]) == SOURCE_ORDER,
                f"{label}: source order moved")
        interior = report["first_interior"]
        fold = report["first_fold_band"]
        if interior is not None:
            owner_class, first = "GLOBAL", interior
        elif fold is not None:
            owner_class, first = "FOLD_LOCAL", fold
        else:
            owner_class, first = "AT_FLOOR", None
        endpoint[label] = {
            "owner_class": owner_class,
            "first_substep": None if first is None else first["substep"],
            "first_operand": None if first is None else first["name"],
            "interior_max": None if first is None else first["interior"]["max_abs"],
            "fold_band_max": None if first is None else first["fold_band"]["max_abs"],
            "complete_max": None if first is None else first["complete"]["max_abs"],
            "argmax": None if first is None else first["complete"]["argmax"],
        }
    if plant == "owner-class":
        endpoint["independent"]["owner_class"] = "FOLD_LOCAL"
    require(endpoint["independent"]["owner_class"]
            == endpoint["given_nemo_entry"]["owner_class"],
            "the two labels disagree on owner class")
    require(endpoint["independent"]["first_substep"]
            == endpoint["given_nemo_entry"]["first_substep"],
            "the two labels disagree on first substep")
    require(endpoint["independent"]["first_operand"]
            == endpoint["given_nemo_entry"]["first_operand"],
            "the two labels disagree on first operand")
    require(plant == "none", f"{plant} plant stayed green")
    prediction = (
        "CONFIRMED_GLOBAL" if endpoint["independent"]["owner_class"] == "GLOBAL"
        else "REFUTED_NOT_GLOBAL"
    )
    return {
        "format": "nemo-testcase-l4-orca2-round235-classification-v1",
        "status": "HELD_R235_EXTERNAL_MODE_OWNER_CLASSIFIED",
        "rows": endpoint,
        "predictions": {
            "R235-P1": "CONFIRMED_PASSIVE_ADMITTED_RECORD",
            "R235-P2": prediction,
            "R235-P3": "CONFIRMED_SOURCE_ORDER",
            "R235-P4": "CONFIRMED_LABEL_AGREEMENT",
            "R235-P5": "CONFIRMED_MEASUREMENT_ONLY",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=MEASURE_PLANTS, default="none")
    parser.add_argument("--classify", nargs=2, type=Path, metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--classify-plant", choices=CLASSIFY_PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            reports = [json.loads(path.read_text(encoding="utf-8"))
                       for path in args.classify]
            result = classify(reports, plant=args.classify_plant)
        else:
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)),
                "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit, plant=args.plant)
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        planted = (args.classify_plant != "none" if args.classify
                   else args.plant != "none")
        print(f"STATUS {'PLANT-FIRED' if planted else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

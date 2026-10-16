#!/usr/bin/env python3
"""Split round 294's substep-1 vector-form V update offline."""

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
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1_gate,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round197_vector_v_update as r197,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round281_stage1_source_split as r281,
    nemo_testcase_l4_orca2_round294_raw_hv_substep as r294,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import (  # noqa: E402
    read_advmean,
    read_ordered,
)

FLOOR = np.float64(2.0e-10)
INPUT_ORDER = r197.INPUT_ORDER
PREDICTED_BIT = ("zv_spg", 35)
PREDICTED_VALUED = ("ssvmask", 35)
PLANTS = (
    "none", "passivity", "source-order", "parent-closure",
    "candidate-replay", "oracle-interior-replay", "signed-zero",
    "locality", "single-substitution", "one-bit",
)


class GateError(RuntimeError):
    """An admitted operand, source replay, or frozen conclusion moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _regional(left, right, support) -> dict[str, object]:
    return r294._regional(left, right, support)


def _scalar(left, right) -> dict[str, object]:
    return r281._exact_score(
        np.asarray([left], np.float64), np.asarray([right], np.float64),
        np.asarray([True]),
    )


def _term_regions(inputs: dict[str, object], reference_raw: np.ndarray,
                  support: np.ndarray) -> dict[str, object]:
    raw = r197._literal_terms(inputs)["raw_va_e"]
    return _regional(raw, reference_raw, support)


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            twin_a: Path, twin_b: Path, expect_commit: str,
            parent_report: Path) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_reference_face_depths,
        _nemo_ssh_avg_prep,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import north_fold_mask
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSBoundaryAssociationTrace,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-295 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 295 requires production JIT on CPU")

    parent = json.loads(parent_report.read_text(encoding="utf-8"))
    require(parent["status"] == "PASS_R294_MEASUREMENT",
            "round-294 parent measurement moved")
    require(parent["first_valued"]["name"] == "v_exit",
            "round-294 first valued row moved")
    require(parent["first_valued"]["regions"]["complete"]["unequal"] == 35,
            "round-294 V-update census moved")

    admission = r281.admission_gate.run(record_root, baseline_root)
    stream_admission = r235._admit_streams(twin_a, twin_b)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2)
    masks_raw = o1_gate._defined_masks(twin_a)["reduced2"]
    support = np.asarray(masks_raw["V"], bool).T
    require(support.shape == (148, 90), "rank-slab V support moved")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    cfg = card.recipe.model_config
    resolved = {
        "momentum_advection": cfg.momentum_advection,
        "literal_fold_unit_vector_form": (
            cfg.barotropic.barotropic_literal_fold_unit_vector_form),
        "completed_stage_top_v": cfg.rk3_fold_exchange_top_v_row,
    }
    require(resolved == {
        "momentum_advection": "vector_invariant",
        "literal_fold_unit_vector_form": False,
        "completed_stage_top_v": False,
    }, "OMT-4 resolved selector registry moved")

    state = card.recipe.initial_state
    dtype = jnp.asarray(state.eta.data).dtype
    grid = card.recipe.grid
    land = jnp.asarray(state.land_mask.data, dtype=dtype)
    prep = _nemo_ssh_avg_prep(
        jnp.asarray(state.H_bathy.data, dtype=dtype), land, grid, dtype,
        north_fold_mask(grid))
    _, raw_hv = _nemo_literal_reference_face_depths(card.recipe.z_coord, dtype)
    override = (prep[0], raw_hv)
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))

    def run(expose: bool):
        hooks = _NEMOWSRK3TestHooks(
            expose_barotropic_substeps=expose,
            expose_barotropic_boundary_association=expose,
            barotropic_reference_face_depth_override=override,
        )
        model = LatLonCGridOceanModel(
            grid, card.recipe.z_coord, cfg, _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    ordinary = run(False)
    observed = run(True)
    require(isinstance(observed, _NEMOWSBoundaryAssociationTrace),
            "substep trace type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary)
    trace = observed.substeps
    raw_bundle = card.recipe.z_coord.nemo_een_barotropic
    require(raw_bundle is not None, "OMT-4 card lacks raw NEMO masks")
    raw_surface_vmask = np.max(
        np.asarray(raw_bundle.vmask, np.float64), axis=-1)[:, :90]

    candidate_inputs = {
        "vn_e": r235._native(trace["v_entry"][0], "v"),
        "rDt_e": np.float64(card.dt_s / 65),
        "zv_spg": r235._native(trace["pgf_v"][0], "v"),
        "zv_trd": r235._native(trace["trd_v"][0], "v"),
        "zv_frc": r235._native(trace["slow_v"][0], "v"),
        "ssvmask": r235._native(state.v_mask.data, "v"),
    }
    reference_inputs = {
        "vn_e": np.asarray(substeps["v_entry"][0], np.float64),
        "rDt_e": np.float64(ordered["dt"]),
        "zv_spg": np.asarray(substeps["pgf_v"][0], np.float64),
        "zv_trd": np.asarray(substeps["trd_v"][0], np.float64),
        "zv_frc": np.asarray(substeps["slow_v"][0], np.float64),
        "ssvmask": raw_surface_vmask,
    }
    candidate_raw = r197._literal_terms(candidate_inputs)["raw_va_e"]
    reference_raw = r197._literal_terms(reference_inputs)["raw_va_e"]
    candidate_trace = r235._native(trace["v_exit"][0], "v")
    oracle_post = np.asarray(substeps["v_exit"][0], np.float64)
    candidate_replay = _regional(candidate_raw, candidate_trace, support)
    # NEMO's admitted v_exit is after lbc_lnk.  Away from the three northern
    # fold rows, the raw expression and recorded output are the same program
    # value and provide an independent calibration of the oracle replay.
    oracle_interior_replay = r281._exact_score(
        reference_raw[:-3], oracle_post[:-3], support[:-3])

    operand_rows = {}
    for name in INPUT_ORDER:
        operand_rows[name] = (
            _scalar(candidate_inputs[name], reference_inputs[name])
            if name == "rDt_e" else
            _regional(candidate_inputs[name], reference_inputs[name], support)
        )
    single = {}
    cumulative = {}
    accumulated = dict(candidate_inputs)
    for name in INPUT_ORDER:
        variant = dict(candidate_inputs)
        variant[name] = reference_inputs[name]
        single[name] = _term_regions(variant, reference_raw, support)
        accumulated[name] = reference_inputs[name]
        cumulative[name] = _term_regions(accumulated, reference_raw, support)

    control = r281._exact_score(
        np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
        np.asarray([True]))
    return {
        "format": "nemo-testcase-l4-orca2-round295-v-update-operands-v1",
        "status": "PASS_R295_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1_substep1",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm; raw-hv-only arm",
        "admission": admission,
        "stream_admission": stream_admission,
        "resolved_config": resolved,
        "passivity": passivity,
        "parent_round294": parent["first_valued"],
        "input_order": list(INPUT_ORDER),
        "candidate_replay": candidate_replay,
        "oracle_interior_replay": oracle_interior_replay,
        "operand_rows": operand_rows,
        "single_substitution": single,
        "cumulative_substitution": cumulative,
        "baseline_vs_reference_raw": _regional(
            candidate_raw, reference_raw, support),
        "reference_raw_vs_post_lbc": _regional(
            reference_raw, oracle_post, support),
        "one_bit_control": control,
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    registered_single = report.get("closing_single_substitution")
    if plant == "passivity":
        report["passivity"][next(iter(report["passivity"]))] = False
    elif plant == "source-order":
        report["input_order"][0], report["input_order"][1] = (
            report["input_order"][1], report["input_order"][0])
    elif plant == "parent-closure":
        report["parent_round294"]["regions"]["complete"]["unequal"] = 34
    elif plant == "candidate-replay":
        report["candidate_replay"]["complete"]["unequal"] = 1
    elif plant == "oracle-interior-replay":
        report["oracle_interior_replay"]["unequal"] = 1
    elif plant == "signed-zero":
        report["operand_rows"]["zv_spg"]["complete"]["max_abs"] = 1.0
    elif plant == "locality":
        report["operand_rows"]["ssvmask"]["interior"]["unequal"] = 1
    elif plant == "single-substitution":
        report["single_substitution"]["ssvmask"]["complete"]["unequal"] = 1
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R295_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] == "independent_omt4_kt1_stage1_substep1",
            "claim label moved")
    require(report["admission"]["status"] ==
            "PASS_R280_OMT4_STAGE1_ADMISSION", "record admission moved")
    require(all(report["passivity"].values()),
            "raw-hv observer moved the completed state")
    require(tuple(report["input_order"]) == INPUT_ORDER,
            "compiled vector-update operand order moved")
    require(report["parent_round294"]["name"] == "v_exit"
            and report["parent_round294"]["regions"]["complete"]["unequal"] == 35,
            "round-294 parent closure moved")
    require(report["candidate_replay"]["complete"]["unequal"] == 0,
            "candidate update replay missed the passive raw V exit")
    require(report["oracle_interior_replay"]["unequal"] == 0,
            "oracle update replay missed the post-LBC interior")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")

    rows = report["operand_rows"]
    first_bit = next(
        (name for name in INPUT_ORDER
         if (rows[name].get("unequal", 0) if name == "rDt_e"
             else rows[name]["complete"]["unequal"]) > 0), None)
    first_valued = next(
        (name for name in INPUT_ORDER
         if (rows[name].get("max_abs", 0.0) if name == "rDt_e"
             else rows[name]["complete"]["max_abs"]) > FLOOR), None)
    require(first_bit is not None, "all vector-update operands became bit-exact")
    require(first_valued is not None, "all vector-update operands are at the floor")
    require(rows["zv_spg"]["complete"]["max_abs"] == 0.0,
            "the pressure operand is no longer signed-zero-only")
    require(rows[first_valued]["interior"]["unequal"] == 0,
            "first valued operand is not fold-local")

    closing_single = next(
        (name for name in INPUT_ORDER
         if report["single_substitution"][name]["complete"]["unequal"] == 0),
        None)
    closing_cumulative = next(
        (name for name in INPUT_ORDER
         if report["cumulative_substitution"][name]["complete"]["unequal"] == 0),
        None)
    if registered_single is not None:
        require(registered_single == closing_single,
                "registered single-substitution closure moved")
    predicted_bit = (
        first_bit == PREDICTED_BIT[0]
        and rows[first_bit]["complete"]["unequal"] == PREDICTED_BIT[1])
    predicted_valued = (
        first_valued == PREDICTED_VALUED[0]
        and rows[first_valued]["complete"]["unequal"] == PREDICTED_VALUED[1])
    predicted_closure = closing_single == "ssvmask"
    report["first_nonbit_operand"] = first_bit
    report["first_valued_operand"] = first_valued
    report["closing_single_substitution"] = closing_single
    report["closing_cumulative_operand"] = closing_cumulative
    report["prediction_ledger"] = {
        "R295-P1": "CONFIRMED",
        "R295-P2": "CONFIRMED",
        "R295-P3": (
            "CONFIRMED_SIGNED_ZERO_THEN_SSVMASK"
            if predicted_bit and predicted_valued else
            f"REFUTED_BIT_{first_bit}_VALUED_{first_valued}"),
        "R295-P4": (
            "CONFIRMED_SINGLE_SSVMASK" if predicted_closure else
            f"REFUTED_SINGLE_{closing_single}_CUMULATIVE_{closing_cumulative}"),
        "R295-P5": "CONFIRMED_HELD_INSIDE_ATOMIC_FOLD_UNIT",
        "R295-P6": "CONFIRMED",
    }
    report["first_nonbit_statement"] = {
        "bit_operand": first_bit,
        "valued_operand": first_valued,
        "nemo_source": (
            "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
            "dynspg_ts.f90:666-678"),
        "status": "HELD_INSIDE_ATOMIC_FOLD_UNIT",
    }
    report["status"] = f"HELD_R295_VECTOR_V_{first_valued.upper()}"
    require(plant == "none", f"{plant} plant stayed green")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--parent-report", type=Path)
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.record_root, args.baseline_root,
                         args.twin_a, args.twin_b, args.expect_commit,
                         args.parent_report)), "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.record_root, args.baseline_root,
                args.twin_a, args.twin_b, args.expect_commit,
                args.parent_report)
        else:
            require(args.report_in is not None, "classification report missing")
            result = classify(
                json.loads(args.report_in.read_text(encoding="utf-8")),
                plant=args.plant)
    except (GateError, r281.GateError, r281.admission_gate.Refusal,
            r281.admission_gate.schema.Refusal, omt4.GateError,
            OSError, KeyError, TypeError, ValueError, IndexError) as error:
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

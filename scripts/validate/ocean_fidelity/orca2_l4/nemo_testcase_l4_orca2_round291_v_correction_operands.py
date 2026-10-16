#!/usr/bin/env python3
"""Split OMT-4's stage-1 V barotropic correction in compiled order."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round281_stage1_source_split as r281,
    nemo_testcase_l4_orca2_round283_stage1_v_operand_split as r283,
)

FLOOR = np.float64(2.0e-10)
SOURCE_ROW = 146
OPERAND_ORDER = (
    "external_target_v", "depth_weighted_sum", "r1_hv_0",
    "scalar_correction", "stage_vmask",
)
ROUND290_CALIBRATION = {
    "unequal": 1440,
    "count": 5400,
    "max_abs": 0.016207369589460653,
    "rms": 0.0035899206923965866,
}
PROGRAM_POINTS = {
    "raw_v": "after vector-form stage update, before barotropic correction",
    "corrected_v": "after barotropic correction, before lbc_lnk",
}
PLANTS = (
    "none", "passivity", "source-order", "target-control",
    "sum-control", "reciprocal-control", "scalar-control", "mask-control",
    "self-replay", "oracle-replay", "round290-calibration", "one-bit",
)


class GateError(RuntimeError):
    """An admitted operand, matched boundary, or control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right, support=None) -> dict[str, object]:
    return r281._exact_score(
        np.asarray(left, np.float64), np.asarray(right, np.float64), support)


def _source_components(field, target, thickness, reciprocal, mask):
    """Replay stprk3_stg.f90:525-550 with named intermediate arrays."""

    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as r
    from legoesm.ocean.dynamics.barotropic_common import ascending_level_sum

    @jax.jit
    def replay(f, t, h, inv, m):
        product = r(f * h)
        depth_sum = r(ascending_level_sum(product))
        weighted_sum = r(depth_sum * inv)
        correction = r(t - weighted_sum)
        corrected = r(f + r(correction[..., None] * m))
        return product, depth_sum, weighted_sum, correction, corrected

    return tuple(np.asarray(value, np.float64) for value in jax.device_get(
        replay(*(jnp.asarray(value) for value in (
            field, target, thickness, reciprocal, mask)))))


def _row_values(values):
    return np.asarray(values, np.float64)[SOURCE_ROW]


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_common import (
        nemo_reference_depth_reciprocal,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        compute_face_masks_3d,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-291 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 291 requires production JIT on CPU")

    admission = r281.admission_gate.run(record_root, baseline_root)
    oracle = r281._assemble_values(record_root)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    cfg = card.recipe.model_config
    require(cfg.rk3_stage_velocity_written_order,
            "OMT-4 no longer selects NEMO's written stage order")
    require(not cfg.rk3_fold_exchange_top_v_row,
            "Decision-125 top-V selector unexpectedly enabled")
    state = card.recipe.initial_state
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=r249._hooks(card, False))
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=r249._hooks(card, False, live=True))
    ordinary_after = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(observed, _NEMOWSLiveOperandTrace),
            "live operand trace type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary_after)

    candidate_field = r283._native_v(observed.stage_raw_velocities[0][1])
    candidate_corrected = r283._native_v(observed.stage_outputs[0][1])
    candidate_target = r283._native_v(observed.barotropic_targets[1])
    candidate_h = r283._native_v(observed.barotropic_correction_geometry[1])
    candidate_r1 = r283._native_v(observed.barotropic_correction_geometry[3])
    _, candidate_mask_redundant = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    candidate_mask = r283._native_v(candidate_mask_redundant)

    expected_field = np.asarray(oracle["pre_v"], np.float64)[..., :-1]
    expected_corrected = np.asarray(oracle["cor_v"], np.float64)[..., :-1]
    expected_target = np.asarray(oracle["ext_vb"], np.float64)
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 lacks raw NEMO face operands")
    expected_h = np.asarray(raw.e3v_0, np.float64)
    expected_mask = np.asarray(raw.vmask, np.float64)
    surface_mask = np.max(expected_mask, axis=-1)
    expected_r1 = np.asarray(jax.device_get(nemo_reference_depth_reciprocal(
        jnp.asarray(raw.hv_0), jnp.asarray(surface_mask))), np.float64)

    shape3 = (148, 180, 30)
    shape2 = (148, 180)
    require(all(value.shape == shape3 for value in (
        candidate_field, candidate_corrected, candidate_h, candidate_mask,
        expected_field, expected_corrected, expected_h, expected_mask,
    )), "three-dimensional correction operand shape moved")
    require(all(value.shape == shape2 for value in (
        candidate_target, candidate_r1, expected_target, expected_r1,
    )), "two-dimensional correction operand shape moved")

    candidate_parts = _source_components(
        candidate_field, candidate_target, candidate_h, candidate_r1,
        candidate_mask)
    expected_parts = _source_components(
        expected_field, expected_target, expected_h, expected_r1,
        expected_mask)
    names = ("product", "depth_sum", "weighted_sum", "correction", "corrected")
    candidate = dict(zip(names, candidate_parts, strict=True))
    expected = dict(zip(names, expected_parts, strict=True))

    # Cumulative substitutions preserve NEMO's statement order while holding
    # every later operand on the live candidate value.
    cumulative = {}
    substitutions = {
        "external_target_v": expected_target,
        "depth_weighted_sum": expected["depth_sum"],
        "r1_hv_0": expected_r1,
        "stage_vmask": expected_mask,
    }
    target, depth_sum, reciprocal, mask = (
        candidate_target, candidate["depth_sum"], candidate_r1, candidate_mask)

    def replay_from_aggregates(t, total, inv, stage_mask):
        from legoesm.core.source_rounding import nemo_source_round as r
        weighted = r(jnp.asarray(total) * jnp.asarray(inv))
        correction = r(jnp.asarray(t) - weighted)
        corrected = r(jnp.asarray(candidate_field) + r(
            correction[..., None] * jnp.asarray(stage_mask)))
        return np.asarray(jax.device_get(corrected), np.float64)

    for name in OPERAND_ORDER:
        if name == "external_target_v":
            target = substitutions[name]
        elif name == "depth_weighted_sum":
            depth_sum = substitutions[name]
        elif name == "r1_hv_0":
            reciprocal = substitutions[name]
        elif name == "scalar_correction":
            # It is an output boundary, not an independently substituted input.
            pass
        elif name == "stage_vmask":
            mask = substitutions[name]
        cumulative[name] = _score(
            _row_values(replay_from_aggregates(
                target, depth_sum, reciprocal, mask)),
            _row_values(expected_corrected))

    controls = {name: _score(
        np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]))
        for name in (
            "target", "sum", "reciprocal", "scalar", "mask", "one_bit")}

    return {
        "format": "nemo-testcase-l4-orca2-round291-v-correction-operands-v1",
        "status": "PASS_R291_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm; existing passive trace",
        "admission": admission,
        "passivity": passivity,
        "source_row": SOURCE_ROW,
        "program_points": dict(PROGRAM_POINTS),
        "operand_order": list(OPERAND_ORDER),
        "compiled_sources": {
            "external_target_v": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:521-528"),
            "depth_weighted_sum": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:525-527"),
            "r1_hv_0": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/domain.f90:193-215"),
            "scalar_correction": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:525-527"),
            "stage_vmask": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:547-550"),
        },
        "raw_operands": {
            "reference_e3v": _score(
                _row_values(candidate_h), _row_values(expected_h)),
            "external_target_v": _score(
                _row_values(candidate_target), _row_values(expected_target)),
            "r1_hv_0": _score(
                _row_values(candidate_r1), _row_values(expected_r1)),
            "stage_vmask": _score(
                _row_values(candidate_mask), _row_values(expected_mask)),
        },
        "statement_intermediates": {
            "depth_weighted_sum": _score(
                _row_values(candidate["depth_sum"]),
                _row_values(expected["depth_sum"])),
            "weighted_sum": _score(
                _row_values(candidate["weighted_sum"]),
                _row_values(expected["weighted_sum"])),
            "scalar_correction": _score(
                _row_values(candidate["correction"]),
                _row_values(expected["correction"])),
        },
        "replays": {
            "candidate_self": _score(
                _row_values(candidate["corrected"]),
                _row_values(candidate_corrected)),
            "oracle_self": _score(
                _row_values(expected["corrected"]),
                _row_values(expected_corrected)),
            "candidate_vs_oracle": _score(
                _row_values(candidate["corrected"]),
                _row_values(expected_corrected)),
        },
        "cumulative_substitutions": cumulative,
        "round290_calibration": dict(ROUND290_CALIBRATION),
        "controls": controls,
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        report["passivity"][next(iter(report["passivity"]))] = False
    elif plant == "source-order":
        report["operand_order"][0], report["operand_order"][1] = (
            report["operand_order"][1], report["operand_order"][0])
    elif plant.endswith("-control"):
        report["controls"][plant.removesuffix("-control")]["unequal"] = 0
    elif plant == "self-replay":
        report["replays"]["candidate_self"]["max_abs"] = 1.0
    elif plant == "oracle-replay":
        report["replays"]["oracle_self"]["max_abs"] = 1.0
    elif plant == "round290-calibration":
        report["round290_calibration"]["unequal"] -= 1
    elif plant == "one-bit":
        report["controls"]["one_bit"]["unequal"] = 0

    require(report["status"] == "PASS_R291_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] == "independent_omt4_kt1_stage1",
            "claim label moved")
    require(report["admission"]["status"] ==
            "PASS_R280_OMT4_STAGE1_ADMISSION", "record admission moved")
    require(all(report["passivity"].values()), "live trace is not passive")
    require(report["source_row"] == SOURCE_ROW, "source-row registry moved")
    require(report["program_points"] == PROGRAM_POINTS,
            "matched program-point registry moved")
    require(tuple(report["operand_order"]) == OPERAND_ORDER,
            "compiled operand order moved")
    require(report["round290_calibration"] == ROUND290_CALIBRATION,
            "round-290 calibration registry moved")
    for name, row in report["controls"].items():
        require(row["unequal"] == 1, f"{name} one-bit control did not fire")
    require(report["replays"]["candidate_self"]["max_abs"] <= FLOOR,
            "candidate correction self-replay moved")
    require(report["replays"]["oracle_self"]["max_abs"] <= FLOOR,
            "oracle correction replay moved")
    live = report["replays"]["candidate_vs_oracle"]
    require(all(live[name] == ROUND290_CALIBRATION[name]
                for name in ("unequal", "count", "max_abs", "rms")),
            "round-290 live-boundary calibration moved")

    rows = {
        "external_target_v": report["raw_operands"]["external_target_v"],
        "depth_weighted_sum": report["statement_intermediates"][
            "depth_weighted_sum"],
        "r1_hv_0": report["raw_operands"]["r1_hv_0"],
        "scalar_correction": report["statement_intermediates"][
            "scalar_correction"],
        "stage_vmask": report["raw_operands"]["stage_vmask"],
    }
    cumulative = report["cumulative_substitutions"]
    before = live["max_abs"]
    owner = None
    for name in OPERAND_ORDER:
        after = cumulative[name]["max_abs"]
        if before > FLOOR and after <= FLOOR:
            owner = name
            break
        before = after
    if owner is None:
        owner = next((name for name in OPERAND_ORDER
                      if rows[name]["max_abs"] > FLOOR), None)

    mask_prediction = (
        owner == "stage_vmask"
        and rows["stage_vmask"]["unequal"] == 1319
        and cumulative["stage_vmask"]["max_abs"] <= FLOOR)
    p2 = rows["external_target_v"]["max_abs"] <= FLOOR
    p3_aggregate = all(rows[name]["max_abs"] <= FLOOR for name in (
        "depth_weighted_sum", "r1_hv_0"))
    reference_e3v_exact = (
        report["raw_operands"]["reference_e3v"]["max_abs"] <= FLOOR)
    p4 = (rows["scalar_correction"]["max_abs"] <= FLOOR
          and report["replays"]["candidate_self"]["max_abs"] <= FLOOR)
    report["prediction_ledger"] = {
        "R291-P1": "CONFIRMED",
        "R291-P2": "CONFIRMED" if p2 else "REFUTED_TARGET_OWNS",
        "R291-P3": (
            "CONFIRMED" if p3_aggregate and reference_e3v_exact else
            "REFUTED_REFERENCE_E3V_NULL_SUM_AND_RECIPROCAL_EXACT"
            if p3_aggregate else "REFUTED_AGGREGATE_OR_RECIPROCAL_MOVED"),
        "R291-P4": "CONFIRMED" if p4 else "REFUTED_SCALAR_MOVED",
        "R291-P5": "CONFIRMED" if mask_prediction else "REFUTED",
        "R291-P6": "CONFIRMED",
    }
    if owner is None:
        report["first_nonbit_statement"] = {
            "boundary": "unresolved barotropic V correction",
            "status": "HELD",
        }
        report["status"] = "HELD_R291_UNRESOLVED"
    else:
        report["first_nonbit_statement"] = {
            "boundary": owner,
            "status": "HELD_CITED_STATEMENT",
            "nemo_source": report["compiled_sources"][owner],
        }
        report["status"] = (
            "HELD_R291_STAGE_MASK_CANCELLING_STATEMENT"
            if owner == "stage_vmask" else f"HELD_R291_{owner.upper()}")
    require(plant == "none", f"{plant} plant stayed green")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.record_root, args.baseline_root,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.record_root, args.baseline_root,
                             args.expect_commit)
        else:
            require(args.report_in is not None, "classification report missing")
            result = classify(json.loads(args.report_in.read_text(encoding="utf-8")),
                              plant=args.plant)
    except (GateError, r281.GateError, r281.admission_gate.Refusal,
            r281.admission_gate.schema.Refusal, r249.GateError, omt4.GateError,
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

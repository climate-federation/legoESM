#!/usr/bin/env python3
"""Split OMT-4's live stage-1 V source row at matched boundaries."""

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
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round281_stage1_source_split as r281,
    nemo_testcase_l4_orca2_round283_stage1_v_operand_split as r283,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)

FLOOR = np.float64(2.0e-10)
SOURCE_ROW = 146
ROUND289_CALIBRATION = {
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
    "none", "passivity", "program-order", "raw-boundary",
    "correction-calibration", "one-bit",
)


class GateError(RuntimeError):
    """An admitted record, matched boundary, or control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right) -> dict[str, object]:
    return r281._exact_score(
        np.asarray(left, np.float64), np.asarray(right, np.float64))


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-290 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 290 requires production JIT on CPU")

    admission = r281.admission_gate.run(record_root, baseline_root)
    oracle = r281._assemble_values(record_root)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    require(not card.recipe.model_config.rk3_fold_exchange_top_v_row,
            "Decision-125 top-V selector unexpectedly enabled")
    state = card.recipe.initial_state
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, False))
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, False, live=True))
    ordinary_after = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(observed, _NEMOWSLiveOperandTrace),
            "live operand trace type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary_after)

    raw_v = r283._native_v(observed.stage_raw_velocities[0][1])
    corrected_v = r283._native_v(observed.stage_outputs[0][1])
    nemo_raw = np.asarray(oracle["pre_v"], np.float64)[..., :-1]
    nemo_corrected = np.asarray(oracle["cor_v"], np.float64)[..., :-1]
    require(raw_v.shape == corrected_v.shape == nemo_raw.shape == (
        148, 180, 30), "matched stage-1 V shape moved")

    raw_row = raw_v[SOURCE_ROW]
    corrected_row = corrected_v[SOURCE_ROW]
    nemo_raw_row = nemo_raw[SOURCE_ROW]
    nemo_corrected_row = nemo_corrected[SOURCE_ROW]
    candidate_increment = corrected_row - raw_row
    nemo_increment = nemo_corrected_row - nemo_raw_row
    candidate_recomposed = raw_row + candidate_increment
    candidate_on_nemo_raw = nemo_raw_row + candidate_increment
    one = np.asarray([1.0], np.float64)

    return {
        "format": "nemo-testcase-l4-orca2-round290-v-source-boundary-v1",
        "status": "PASS_R290_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "worktree": stamp,
        "admission": admission,
        "execution": "production-jit-cpu-fp64-libm; existing passive trace",
        "source_row": SOURCE_ROW,
        "program_points": dict(PROGRAM_POINTS),
        "compiled_sources": {
            "raw_update": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:477-480"),
            "barotropic_correction": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:521-528"),
        },
        "passivity": passivity,
        "boundaries": {
            "raw_v_vs_nemo_pre_v": _score(raw_row, nemo_raw_row),
            "corrected_v_vs_nemo_cor_v": _score(
                corrected_row, nemo_corrected_row),
        },
        "increments": {
            "candidate_vs_nemo": _score(candidate_increment, nemo_increment),
            "candidate_recomposition": _score(
                candidate_recomposed, corrected_row),
            "candidate_on_nemo_raw_vs_nemo_cor": _score(
                candidate_on_nemo_raw, nemo_corrected_row),
        },
        "round289_calibration": dict(ROUND289_CALIBRATION),
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        report["passivity"][next(iter(report["passivity"]))] = False
    elif plant == "program-order":
        report["program_points"]["raw_v"] = PROGRAM_POINTS["corrected_v"]
    elif plant == "raw-boundary":
        report["source_row"] -= 1
    elif plant == "correction-calibration":
        report["round289_calibration"]["unequal"] -= 1
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R290_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] == "independent_omt4_kt1_stage1",
            "claim label moved")
    require(report["admission"]["status"] ==
            "PASS_R280_OMT4_STAGE1_ADMISSION", "record admission moved")
    require(all(report["passivity"].values()), "live trace is not passive")
    require(report["source_row"] == SOURCE_ROW, "source-row registry moved")
    require(report["program_points"] == PROGRAM_POINTS,
            "matched program-point registry moved")
    require(report["round289_calibration"] == ROUND289_CALIBRATION,
            "round-289 calibration registry moved")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")

    raw = report["boundaries"]["raw_v_vs_nemo_pre_v"]
    corrected = report["boundaries"]["corrected_v_vs_nemo_cor_v"]
    increment = report["increments"]["candidate_vs_nemo"]
    recomposed = report["increments"]["candidate_recomposition"]
    arm = report["increments"]["candidate_on_nemo_raw_vs_nemo_cor"]
    raw_valued = raw["max_abs"] > FLOOR
    correction_calibrated = all(
        corrected[name] == ROUND289_CALIBRATION[name]
        for name in ("unequal", "count", "max_abs", "rms"))
    increment_valued = increment["max_abs"] > FLOOR
    recompose_closes = recomposed["max_abs"] <= FLOOR
    arm_carries_correction = arm["max_abs"] > FLOOR

    if not correction_calibrated:
        boundary = "measurement disagreement with round 289"
        source = None
        status = "HELD_R290_CALIBRATION_DISAGREEMENT"
    elif raw_valued:
        boundary = "raw vector-form V update"
        source = report["compiled_sources"]["raw_update"]
        status = "HELD_R290_RAW_UPDATE"
    elif increment_valued and recompose_closes and arm_carries_correction:
        boundary = "barotropic V correction"
        source = report["compiled_sources"]["barotropic_correction"]
        status = "HELD_R290_BAROTROPIC_CORRECTION"
    else:
        boundary = "unresolved matched V boundary"
        source = None
        status = "HELD_R290_UNRESOLVED"

    report["first_nonbit_statement"] = {
        "boundary": boundary,
        "status": "HELD_CITED_STATEMENT" if source else "HELD",
        **({"nemo_source": source} if source else {}),
    }
    report["prediction_ledger"] = {
        "R290-P1": "CONFIRMED",
        "R290-P2": "CONFIRMED" if not raw_valued else "REFUTED_RAW_OWNS",
        "R290-P3": "CONFIRMED" if correction_calibrated else "REFUTED",
        "R290-P4": (
            "CONFIRMED" if increment_valued and recompose_closes
            and arm_carries_correction else "REFUTED"),
        "R290-P5": "CONFIRMED",
    }
    report["status"] = status
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

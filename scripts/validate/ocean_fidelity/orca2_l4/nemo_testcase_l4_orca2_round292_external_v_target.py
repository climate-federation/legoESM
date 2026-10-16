#!/usr/bin/env python3
"""Walk OMT-4's carried external-mode V target through its substeps."""

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
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round281_stage1_source_split as r281,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean  # noqa: E402

FLOOR = np.float64(2.0e-10)
SOURCE_ROW = 146
TARGET_CALIBRATION = {
    "unequal": 35,
    "count": 90,
    "max_abs": 0.01620736958946065,
}
SOURCE_ORDER = (
    "slow_v", "v_entry", "v_mid", "eta_mid", "face_depth_v_mid",
    "metric_transport_v", "eta_exit", "eta_pgf", "pgf_v", "cor_v",
    "trd_v", "v_exit",
)
FACE = {
    "slow_v": "V", "v_entry": "V", "v_mid": "V", "eta_mid": "T",
    "face_depth_v_mid": "V", "metric_transport_v": "V",
    "eta_exit": "T", "eta_pgf": "T", "pgf_v": "V", "cor_v": "V",
    "trd_v": "V", "v_exit": "V",
}
PLANTS = (
    "none", "passivity", "source-order", "target-calibration",
    "first-boundary", "locality", "endpoint-causality", "one-bit",
)


class GateError(RuntimeError):
    """An admitted stream, matched program point, or control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right, support) -> dict[str, object]:
    return r281._exact_score(
        np.asarray(left, np.float64), np.asarray(right, np.float64), support)


def _regional(left, right, support) -> dict[str, object]:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    support = np.asarray(support, bool)
    require(left.shape == right.shape == support.shape == (148, 90),
            "rank-slab shape or support moved")
    return {
        "complete": _score(left, right, support),
        "interior": _score(left[:-3], right[:-3], support[:-3]),
        "fold_band": _score(left[-3:], right[-3:], support[-3:]),
    }


def _first(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return next((row for row in rows
                 if row["regions"]["complete"]["max_abs"] > FLOOR), None)


def _source_rows(trace, substeps, advmean, masks) -> list[dict[str, object]]:
    registry = (
        ("slow_v", trace["slow_v"], substeps["slow_v"]),
        ("v_entry", trace["v_entry"], substeps["v_entry"]),
        ("v_mid", trace["v_mid"], substeps["v_mid"]),
        ("eta_mid", trace["eta_mid"], substeps["eta_mid"]),
        ("face_depth_v_mid", trace["transport_face_depth_v"],
         advmean["face_depth_v"]),
        ("metric_transport_v", trace["transport_metric_v"],
         advmean["metric_v"]),
        ("eta_exit", trace["eta_exit"], substeps["eta_exit"]),
        ("eta_pgf", trace["eta_pgf"], substeps["eta_pgf"]),
        ("pgf_v", trace["pgf_v"], substeps["pgf_v"]),
        ("cor_v", trace["cor_v"], substeps["cor_v"]),
        ("trd_v", trace["trd_v"], substeps["trd_v"]),
        ("v_exit", trace["v_exit"], substeps["v_exit"]),
    )
    require(tuple(row[0] for row in registry) == SOURCE_ORDER,
            "source registry moved")
    rows = []
    for substep in range(65):
        for order, (name, candidate, oracle) in enumerate(registry):
            face = FACE[name]
            rows.append({
                "substep": substep + 1,
                "source_order": order,
                "name": name,
                "face": face,
                "regions": _regional(
                    r235._native(candidate[substep], face.lower()),
                    np.asarray(oracle[substep], np.float64),
                    masks[face]),
            })
    return rows


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            twin_a: Path, twin_b: Path, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSBoundaryAssociationTrace,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-292 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 292 requires production JIT on CPU")

    admission = r281.admission_gate.run(record_root, baseline_root)
    stream_admission = r235._admit_streams(twin_a, twin_b)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    masks_raw = o1_gate._defined_masks(twin_a)["reduced2"]
    # ``mesh_mask`` is read in Fortran (i,j) order while every admitted
    # canonical stream is returned in model (j,i) order.
    masks = {
        name: np.asarray(masks_raw[name], bool).T for name in ("T", "V")
    }
    require(all(mask.shape == (148, 90) for mask in masks.values()),
            "canonical wet-mask orientation moved")
    oracle_stage = r281._assemble_values(record_root)

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
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))

    def run(hooks):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    ordinary = run(_NEMOWSRK3TestHooks())
    traced = run(_NEMOWSRK3TestHooks(
        expose_barotropic_substeps=True,
        expose_barotropic_boundary_association=True))
    live = run(_NEMOWSRK3TestHooks(expose_live_stage_operands=True))
    require(isinstance(traced, _NEMOWSBoundaryAssociationTrace),
            "substep trace type moved")
    require(isinstance(live, _NEMOWSLiveOperandTrace),
            "live target trace type moved")
    passivity = {
        "substeps": passive._ordinary_state_equal(traced.state_after, ordinary),
        "live_target": passive._ordinary_state_equal(live.state_after, ordinary),
    }

    rows = _source_rows(traced.substeps, substeps, advmean, masks)
    first = _first(rows)
    require(first is not None, "substep table is clean despite the non-bit target")

    candidate_trace_target = r235._native(
        traced.substeps["v_exit"][-1], "v")
    candidate_live_target = r235._native(live.barotropic_targets[1], "v")
    oracle_trace_target = np.asarray(substeps["v_exit"][-1], np.float64)
    oracle_live_target = np.asarray(oracle_stage["ext_vb"], np.float64)[:, :90]
    support_v = masks["V"]
    target = {
        "trace_vs_live_candidate": _regional(
            candidate_trace_target, candidate_live_target, support_v),
        "trace_vs_live_oracle": _regional(
            oracle_trace_target, oracle_live_target, support_v),
        "candidate_vs_oracle": _score(
            candidate_live_target[SOURCE_ROW],
            oracle_live_target[SOURCE_ROW], support_v[SOURCE_ROW]),
    }
    control = _score(
        np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
        np.asarray([True]))
    return {
        "format": "nemo-testcase-l4-orca2-round292-external-v-target-v1",
        "status": "PASS_R292_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm; existing passive traces",
        "admission": admission,
        "stream_admission": stream_admission,
        "passivity": passivity,
        "resolved_config": resolved,
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "first_valued": first,
        "target": target,
        "target_calibration": dict(TARGET_CALIBRATION),
        "one_bit_control": control,
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        report["passivity"]["substeps"][next(iter(
            report["passivity"]["substeps"]))] = False
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "target-calibration":
        report["target"]["candidate_vs_oracle"]["unequal"] -= 1
    elif plant == "first-boundary":
        report["first_valued"]["name"] = "v_exit"
    elif plant == "locality":
        report["first_valued"]["regions"]["interior"]["max_abs"] = 1.0
    elif plant == "endpoint-causality":
        report["target"]["trace_vs_live_candidate"]["complete"]["max_abs"] = 1.0
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R292_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] == "independent_omt4_kt1_stage1",
            "claim label moved")
    require(report["admission"]["status"] ==
            "PASS_R280_OMT4_STAGE1_ADMISSION", "record admission moved")
    require(all(all(group.values()) for group in report["passivity"].values()),
            "a diagnostic trace moved the completed state")
    require(tuple(report["source_order"]) == SOURCE_ORDER,
            "compiled source order moved")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")
    require(report["target_calibration"] == TARGET_CALIBRATION,
            "target calibration registry moved")
    measured_target = report["target"]["candidate_vs_oracle"]
    require(all(measured_target[name] == TARGET_CALIBRATION[name]
                for name in ("unequal", "count", "max_abs")),
            "round-291 target calibration did not reproduce on the rank slab")
    for side in ("trace_vs_live_candidate", "trace_vs_live_oracle"):
        require(report["target"][side]["complete"]["max_abs"] <= FLOOR,
                f"{side} program-point alignment moved")

    first = _first(report["rows"])
    require(first is not None, "no valued substep boundary found")
    require((first["substep"], first["source_order"], first["name"]) ==
            (report["first_valued"]["substep"],
             report["first_valued"]["source_order"],
             report["first_valued"]["name"]),
            "first-boundary registry moved")
    require(first["regions"]["interior"]["max_abs"] <= FLOOR,
            "first valued boundary is not fold-local")
    exit_row = next(row for row in report["rows"]
                    if row["substep"] == first["substep"]
                    and row["name"] == "v_exit")
    require(first["source_order"] <= exit_row["source_order"],
            "first boundary occurs after the substep exit")

    predicted = first["substep"] == 1 and first["name"] == "face_depth_v_mid"
    report["prediction_ledger"] = {
        "R292-P1": "CONFIRMED",
        "R292-P2": "CONFIRMED",
        "R292-P3": "CONFIRMED" if predicted else "REFUTED_EARLIER_OR_LATER_ROW",
        "R292-P4": "CONFIRMED",
        "R292-P5": "CONFIRMED",
        "R292-P6": "CONFIRMED",
    }
    report["first_nonbit_statement"] = {
        "substep": first["substep"],
        "name": first["name"],
        "nemo_source": {
            "face_depth_v_mid": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:517-519"),
            "metric_transport_v": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:530-536"),
            "v_exit": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:655-739"),
        }.get(first["name"],
              "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:445-739"),
        "status": "HELD_CITED_STATEMENT",
    }
    report["status"] = f"HELD_R292_{first['name'].upper()}"
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
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.record_root, args.baseline_root,
                         args.twin_a, args.twin_b, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.record_root, args.baseline_root,
                args.twin_a, args.twin_b, args.expect_commit)
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

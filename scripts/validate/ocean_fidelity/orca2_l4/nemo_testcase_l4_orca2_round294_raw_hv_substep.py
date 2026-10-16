#!/usr/bin/env python3
"""Continue OMT-4's external-substep walk with only raw NEMO hv_0."""

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
    nemo_testcase_l4_orca2_round292_external_v_target as r292,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_advmean  # noqa: E402

FLOOR = np.float64(2.0e-10)
PREDICTED_FIRST = (3, "face_depth_v_mid", 35)
PLANTS = (
    "none", "passivity", "source-order", "arm-isolation",
    "inherited-closure", "first-boundary", "locality", "one-bit",
)


class GateError(RuntimeError):
    """The arm, admitted stream, or source-ordered conclusion moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _regional(left, right, support) -> dict[str, object]:
    return r292._regional(left, right, support)


def _first(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return r292._first(rows)


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
            "round-294 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 294 requires production JIT on CPU")

    parent = json.loads(parent_report.read_text(encoding="utf-8"))
    require(parent["status"] == "PASS_R293_MEASUREMENT",
            "round-293 parent measurement moved")
    require(parent["direct_arms"]["raw_hv_only"]
            ["face_depth_v_mid"]["complete"]["unequal"] == 0,
            "round-293 raw-hv closure moved")

    admission = r281.admission_gate.run(record_root, baseline_root)
    stream_admission = r235._admit_streams(twin_a, twin_b)
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    advmean = read_advmean(
        twin_a / "oracle_bt_advmean_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)
    masks_raw = o1_gate._defined_masks(twin_a)["reduced2"]
    masks = {name: np.asarray(masks_raw[name], bool).T
             for name in ("T", "V")}
    require(all(mask.shape == (148, 90) for mask in masks.values()),
            "canonical wet-mask orientation moved")

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

    reconstructed_v = r235._native(prep[1], "v")
    raw_v = r235._native(raw_hv, "v")
    raw_oracle = np.asarray(card.recipe.z_coord.nemo_hv_0, np.float64)[:, :90]
    all_support = np.ones_like(raw_oracle, dtype=bool)
    arm_isolation = {
        "u_override_vs_production": r281._exact_score(
            np.asarray(override[0]), np.asarray(prep[0]),
            np.ones_like(np.asarray(prep[0]), dtype=bool)),
        "v_override_vs_raw_nemo": r281._exact_score(
            raw_v, raw_oracle, all_support),
        "v_override_vs_production": _regional(
            raw_v, reconstructed_v, all_support),
    }

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
    rows = r292._source_rows(observed.substeps, substeps, advmean, masks)
    first = _first(rows)
    require(first is not None, "raw-hv arm made the complete table clean")
    substep1_depth = next(
        row for row in rows
        if row["substep"] == 1 and row["name"] == "face_depth_v_mid")
    earlier_substep1 = [
        row for row in rows
        if row["substep"] == 1
        and row["source_order"] < substep1_depth["source_order"]
    ]
    control = r281._exact_score(
        np.asarray([0.0]), np.asarray([np.nextafter(0.0, 1.0)]),
        np.asarray([True]))
    return {
        "format": "nemo-testcase-l4-orca2-round294-raw-hv-substep-v1",
        "status": "PASS_R294_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-libm; raw-hv-only arm",
        "admission": admission,
        "stream_admission": stream_admission,
        "parent_round293": {
            "worktree": parent["worktree"],
            "raw_hv_depth": parent["direct_arms"]["raw_hv_only"]
            ["face_depth_v_mid"],
        },
        "resolved_config": resolved,
        "passivity": passivity,
        "arm_isolation": arm_isolation,
        "source_order": list(r292.SOURCE_ORDER),
        "rows": rows,
        "substep1_depth": substep1_depth,
        "substep1_predecessors": earlier_substep1,
        "first_valued": first,
        "predicted_first": {
            "substep": PREDICTED_FIRST[0],
            "name": PREDICTED_FIRST[1],
            "unequal": PREDICTED_FIRST[2],
        },
        "one_bit_control": control,
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "passivity":
        report["passivity"][next(iter(report["passivity"]))] = False
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "arm-isolation":
        report["arm_isolation"]["u_override_vs_production"]["unequal"] = 1
    elif plant == "inherited-closure":
        report["substep1_depth"]["regions"]["complete"]["unequal"] = 1
    elif plant == "first-boundary":
        report["first_valued"]["name"] = "v_exit"
    elif plant == "locality":
        report["first_valued"]["regions"]["interior"]["max_abs"] = 1.0
    elif plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0

    require(report["status"] == "PASS_R294_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] == "independent_omt4_kt1_stage1",
            "claim label moved")
    require(report["admission"]["status"] ==
            "PASS_R280_OMT4_STAGE1_ADMISSION", "record admission moved")
    require(all(report["passivity"].values()),
            "raw-hv observer moved the completed state")
    require(tuple(report["source_order"]) == r292.SOURCE_ORDER,
            "compiled source order moved")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")

    arm = report["arm_isolation"]
    require(arm["u_override_vs_production"]["unequal"] == 0,
            "raw-hv arm moved the U reference depth")
    require(arm["v_override_vs_raw_nemo"]["unequal"] == 0,
            "raw-hv arm does not carry NEMO's stored V depth")
    require(arm["v_override_vs_production"]["complete"]["unequal"] == 13,
            "raw-hv arm census moved")
    require(arm["v_override_vs_production"]["interior"]["unequal"] == 0,
            "raw-hv arm moved an interior V reference depth")

    require(report["substep1_depth"]["regions"]["complete"]["unequal"] == 0,
            "raw hv_0 no longer closes substep-1 midpoint V depth")
    require(all(row["regions"]["complete"]["max_abs"] <= FLOOR
                for row in report["substep1_predecessors"]),
            "an earlier substep-1 source row moved above the floor")
    first = _first(report["rows"])
    require(first is not None, "no valued row remains after raw hv_0")
    require((first["substep"], first["source_order"], first["name"]) ==
            (report["first_valued"]["substep"],
             report["first_valued"]["source_order"],
             report["first_valued"]["name"]),
            "first-boundary registry moved")
    require(first["regions"]["interior"]["max_abs"] <= FLOOR,
            "next valued boundary is not fold-local")

    measured = (
        first["substep"], first["name"],
        first["regions"]["complete"]["unequal"])
    prediction = "CONFIRMED" if measured == PREDICTED_FIRST else (
        "REFUTED_MEASURED_" + str(measured).replace(" ", ""))
    report["prediction_ledger"] = {
        "R294-P1": "CONFIRMED",
        "R294-P2": "CONFIRMED",
        "R294-P3": "CONFIRMED",
        "R294-P4": prediction,
        "R294-P5": (
            "NAMED_ROUND245_MASK_T_NEIGHBOUR_PAIR"
            if measured == PREDICTED_FIRST else "NOT_CREDITED"),
        "R294-P6": "CONFIRMED",
    }
    report["first_nonbit_statement"] = {
        "substep": first["substep"],
        "name": first["name"],
        "nemo_source": {
            "face_depth_v_mid": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:509-520"),
            "metric_transport_v": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:530-536"),
            "v_exit": (
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:666-678"),
        }.get(
            first["name"],
            "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
            "dynspg_ts.f90:445-739"),
        "status": "HELD_INSIDE_ATOMIC_FOLD_UNIT",
    }
    report["status"] = (
        f"HELD_R294_SUBSTEP{first['substep']}_{first['name'].upper()}")
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

#!/usr/bin/env python3
"""Replay independent ORCA2 kt=8 stage-1 RHS operators offline."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round171_rhs_accumulator_walk as r171,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round170_rhs8_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round84_rhs_walk as r84,
)


BOUNDARIES = ("after_hpg", "after_ldf", "after_vor", "after_keg", "after_zad")
FACES = ("u", "v")
ABS_EXPLOSIVE = np.float64(1.0e20)
REL_EXPLOSIVE = np.float64(1.0e12)
PLANTS = (
    "none", "rank-placement", "source-order", "trace-passivity",
    "completed-rhs", "first-boundary", "explosive-classification",
)


class GateError(RuntimeError):
    """The passive replay prerequisites or frozen classifier moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _state_rows(actual, expected) -> dict[str, bool]:
    return r166._state_rows(actual, expected)


def _bits_equal(actual, expected) -> bool:
    return bool(np.array_equal(np.asarray(actual), np.asarray(expected)))


def _explosive(row: dict[str, object]) -> bool:
    candidate = np.float64(row["candidate_max_abs"])
    reference = np.float64(row["reference_max_abs"])
    return bool(
        candidate >= ABS_EXPLOSIVE
        and candidate >= REL_EXPLOSIVE * max(reference, np.float64(1.0))
    )


def _first_nonbit(rows: dict[str, dict[str, dict]]) -> dict | None:
    for boundary in BOUNDARIES:
        for face in FACES:
            row = rows[face][boundary]
            if not row["bit_exact"]:
                return {"boundary": boundary, "face": face, **row}
    return None


def _first_explosive(rows: dict[str, dict[str, dict]], face: str) -> dict | None:
    previous = False
    for boundary in BOUNDARIES:
        row = rows[face][boundary]
        current = bool(row["explosive"])
        if current and not previous:
            return {"boundary": boundary, "face": face, **row}
        previous = current
    return None


def _stage1_tendency(model, state, surface, dt, *, components: bool):
    """Mirror the production stage-1 call, but do not advance the model."""

    import jax.numpy as jnp

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        compute_face_masks_3d,
        compute_frozen_geom_density,
    )
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate,
        compute_layer_thickness,
        nemo_ldf_reference_e3f,
        nemo_qco_live_vorticity_e3f_cgrid,
    )

    z_coord = model.z_coord
    config = model.config
    grid = model.grid
    require(isinstance(z_coord, OceanPartialCellCoordinate),
            "rung-0 stage-1 replay requires partial cells")
    geom = compute_frozen_geom_density(state, grid, z_coord, config)
    u_mask, v_mask = compute_face_masks_3d(z_coord.is_active, grid)
    u_mask = u_mask.astype(state.eta.data.dtype)
    v_mask = v_mask.astype(state.eta.data.dtype)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    face_h = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, u_mask, v_mask, grid)[:2]
    t_mask = jnp.asarray(z_coord.is_active)
    e3f = nemo_qco_live_vorticity_e3f_cgrid(
        state.eta.data, z_coord, state.eta.data.dtype, grid=grid,
        e3t_0=h_ref, tmask=t_mask,
        reference_e3f=nemo_ldf_reference_e3f(z_coord),
    )
    ldf_h = (geom[1], face_h[0], face_h[1], e3f, face_h[0], face_h[1])
    return model.tendencies(
        state, surface, dt=dt,
        precomputed_geom_density=geom,
        grid=grid,
        vertex_mask=model._vertex_mask,
        zad_continuity_dt=dt,
        momentum_flux_face_thickness=face_h,
        ldf_thickness_operands=ldf_h,
        nemo_operator_association=False,
        return_nemo_operator_components=components,
    )


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["admission"]["rank_coverage"] = "overlap"
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "trace-passivity":
        report["trace_passivity"]["1"]["state"]["T"] = False
    elif plant == "completed-rhs":
        report["offline_closure"]["plain_to_passive_u"] = False
    elif plant == "first-boundary":
        first = report["first_nonbit_accumulator"]
        require(first is not None, "first-boundary plant has no live selector")
        report["rows"][first["face"]][first["boundary"]]["bit_exact"] = True
    elif plant == "explosive-classification":
        report["rows"]["u"]["after_hpg"]["explosive"] = not report["rows"][
            "u"]["after_hpg"]["explosive"]

    require(
        report["admission"]["rank_coverage"] == "exactly-once"
        and len(report["admission"]["records"]) == 2,
        "rank-complete RHS record admission moved",
    )
    require(tuple(report["source_order"]) == BOUNDARIES,
            "compiled accumulator source order moved")
    require(len(report["trace_passivity"]) == 7, "trace passivity census moved")
    require(all(
        all(row["state"].values())
        and row["completed_rhs_u"]
        and row["completed_rhs_v"]
        for row in report["trace_passivity"].values()
    ), "live-operand trace moved the complete arm")
    require(all(report["offline_closure"].values()),
            "offline component instrument does not close bit-for-bit")
    require(
        report["one_ulp_control"]["differing_cells"] == 1
        and not report["one_ulp_control"]["bit_exact"],
        "one-ULP known-answer control did not fire",
    )
    for face in FACES:
        for boundary in BOUNDARIES:
            row = report["rows"][face][boundary]
            require(row["explosive"] == _explosive(row),
                    f"{face} {boundary} explosive classification moved")
            require(not row["reference_explosive"],
                    f"NEMO {face} {boundary} is explosive")

    first = _first_nonbit(report["rows"])
    require(first == report["first_nonbit_accumulator"],
            "first non-bit accumulator selector moved")
    first_u = _first_explosive(report["rows"], "u")
    require(first_u == report["first_explosive_u"],
            "first explosive U selector moved")
    report["prediction_dispositions"] = {
        "R172-P1": "CONFIRMED",
        "R172-P2": "CONFIRMED",
        "R172-P3": "CONFIRMED",
        "R172-P4": (
            "CONFIRMED" if first_u is not None
            and first_u["boundary"] == "after_vor" else "REFUTED"
        ),
        "R172-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND172_PASSIVE_RHS_REPLAY"
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-172 measurement requires its clean committed instrument")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-172 replay requires production JIT on CPU")

    admission = check_record.run(record_root, baseline_root)
    oracle, census = r171.assemble_record(record_root)
    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    live = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks._replace(expose_live_stage_operands=True))

    passivity: dict[str, dict[str, object]] = {}
    for kt in range(1, 8):
        ordinary_next = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        trace = jax.device_get(live.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        baro = r170._trace_at_kt8(card, state, freshwater, surface, hooks)
        trace_rhs = trace.slow_forcing_producer
        baro_rhs = baro.slow_forcing_operands
        row = {
            "state": _state_rows(trace.state_after, ordinary_next),
            "completed_rhs_u": _bits_equal(trace_rhs["rhs_u"], baro_rhs["du_dt"]),
            "completed_rhs_v": _bits_equal(trace_rhs["rhs_v"], baro_rhs["dv_dt"]),
        }
        require(all(row["state"].values()) and row["completed_rhs_u"]
                and row["completed_rhs_v"],
                f"live trace moved kt={kt}: {row}")
        passivity[str(kt)] = row
        state = ordinary_next
        print(f"PROGRESS round172 complete kt={kt}", file=sys.stderr, flush=True)

    passive = r170._trace_at_kt8(card, state, freshwater, surface, hooks)
    passive_rhs = passive.slow_forcing_operands
    offline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    offline.prime_step_caches(state)
    plain = jax.device_get(jax.jit(
        lambda source_state, forcing: _stage1_tendency(
            offline, source_state, forcing, card.dt_s, components=False)
    )(state, surface))
    with_components = jax.device_get(jax.jit(
        lambda source_state, forcing: _stage1_tendency(
            offline, source_state, forcing, card.dt_s, components=True)
    )(state, surface))
    component_total, _diagnostics, parts = with_components
    closure = {
        "plain_to_passive_u": _bits_equal(plain.du_dt.data, passive_rhs["du_dt"]),
        "plain_to_passive_v": _bits_equal(plain.dv_dt.data, passive_rhs["dv_dt"]),
        "components_to_plain_u": _bits_equal(component_total.du_dt.data, plain.du_dt.data),
        "components_to_plain_v": _bits_equal(component_total.dv_dt.data, plain.dv_dt.data),
        "production_association_u": _bits_equal(parts["after_ldf_u"].data, plain.du_dt.data),
        "production_association_v": _bits_equal(parts["after_ldf_v"].data, plain.dv_dt.data),
    }
    require(all(closure.values()), f"offline component closure moved: {closure}")

    accumulated = jax.device_get(jax.jit(r84.source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))
    masks = phase3_gate.expected_masks(card)
    active = {face: np.asarray(masks[face], dtype=bool) for face in FACES}
    live_rows = {
        "u": {boundary: r83.native_u(accumulated[f"{boundary}_u"])
              for boundary in BOUNDARIES},
        "v": {boundary: r83.native_v(accumulated[f"{boundary}_v"])
              for boundary in BOUNDARIES},
    }
    rows = {
        face: {
            boundary: r171.r93.score(
                live_rows[face][boundary], oracle[f"{boundary}_{face}"],
                active[face])
            for boundary in BOUNDARIES
        }
        for face in FACES
    }
    for face in FACES:
        for boundary in BOUNDARIES:
            row = rows[face][boundary]
            row["explosive"] = _explosive(row)
            row["reference_explosive"] = bool(
                np.float64(row["reference_max_abs"]) >= ABS_EXPLOSIVE)

    synthetic = np.zeros((2, 2, 2), dtype=np.float64)
    planted = synthetic.copy()
    planted[0, 0, 0] = np.nextafter(0.0, np.float64(np.inf))
    control = r171.r93.score(
        planted, synthetic, np.ones_like(synthetic, dtype=bool))
    raw = {
        "format": "nemo-testcase-l4-orca2-round172-passive-rhs-replay-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "admission": admission,
        "assembled_record": census,
        "completed_kt": 7,
        "kt": 8,
        "source_order": list(BOUNDARIES),
        "trace_passivity": passivity,
        "offline_closure": closure,
        "rows": rows,
        "first_nonbit_accumulator": _first_nonbit(rows),
        "first_explosive_u": _first_explosive(rows, "u"),
        "thresholds": {"absolute": float(ABS_EXPLOSIVE),
                       "relative": float(REL_EXPLOSIVE)},
        "one_ulp_control": control,
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.baseline_root, args.expect_commit)),
                    "measurement requires all record roots and commit")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None,
                    "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError,
            GateError, check_record.Refusal, r166.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND172_PASSIVE_RHS_REPLAY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

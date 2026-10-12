#!/usr/bin/env python3
"""Name the first current-tree OMT-4 stage-1 tracer writer on the T pivot."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_phase2l_tracer_gate as tracer_record,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round231_fold_transport_operand_gate as r231,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
)

LABELS = r249.LABELS
ORDER = (
    "entry_T", "entry_S", "zFu", "zFv", "zFw",
    "u_face_product_T", "v_face_product_T", "u_face_product_S",
    "v_face_product_S", "horizontal_divergence_T",
    "vertical_divergence_T", "after_advection_T", "after_advection_S",
    "after_sbc_T", "after_sbc_S", "qco_stage_T", "qco_stage_S",
    "post_stage_association",
)
PLANTS = (
    "none", "label-coverage", "support", "source-order", "reproduction",
    "first-owner", "recorded-closure", "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-257 predicate or passive-instrument invariant moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _physical(values) -> np.ndarray:
    """The rank-0 physical half of the northern T-pivot row."""

    return np.asarray(values, np.float64)[-1, :90, :30]


def _model(card, hooks):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )


def _run(card, state, forcing, hooks):
    return _model(card, hooks).step(
        state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1])


def _face_rows(tracer, p_u, p_v, grid):
    """Replay the two compiled CEN2 face-product statements."""

    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import centered_cell_to_uface
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points

    @jax.jit
    def program(q, zu, zv):
        u_sum = centered_cell_to_uface(q, nemo_source_sum=True)
        v_sum = interp_to_v_points(q, grid, nemo_source_sum=True)
        return sr(sr(0.5 * zu) * u_sum), sr(sr(0.5 * zv) * v_sum)

    return tuple(np.asarray(value, np.float64) for value in program(
        jnp.asarray(tracer), jnp.asarray(p_u), jnp.asarray(p_v)))


def _divergence_rows(flux_u, flux_v, flux_w, e3t, grid):
    """Replay the compiled horizontal and vertical divergence statements."""

    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round as sr

    @jax.jit
    def program(fu, fv, fw, h):
        du = sr(fu[:, 1:] - fu[:, :-1])
        dv = sr(fv[1:] - fv[:-1])
        horizontal = sr(-sr(sr(sr(du + dv) /
                                  jnp.asarray(grid.area_T)[..., None]) / h))
        dw = sr(fw[..., :-1] - fw[..., 1:])
        vertical = sr(-sr(sr(dw / jnp.asarray(grid.area_T)[..., None]) / h))
        return horizontal, vertical

    return tuple(np.asarray(value, np.float64) for value in program(
        jnp.asarray(flux_u), jnp.asarray(flux_v), jnp.asarray(flux_w),
        jnp.asarray(e3t)))


def _one_bit_control() -> dict[str, object]:
    value = np.asarray([1.0], np.float64)
    return _score(value, np.nextafter(value, np.inf))


def measure(deck_root: Path, frame_root: Path, operand_root: Path,
            label: str, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSLiveOperandTrace

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    base_hooks = r249._hooks(card, True)
    ordinary_model = _model(card, base_hooks)
    live_model = _model(card, r249._hooks(card, True, live=True))

    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
        traced = jax.device_get(live_model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live-stage trace return type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live-stage trace moved the trajectory")

    stage = traced.stage_outputs[0]
    current_t = np.asarray(stage[2], np.float64)
    current_s = np.asarray(stage[3], np.float64)
    oracle_stage = rung0.assemble_frame(frame_root, 1, 1)
    record = r231.assemble_record(operand_root)
    tracer = tracer_record.read_tracer(
        operand_root / tracer_record.TRACER_RECORD)

    candidate_pu_full = np.asarray(traced.stage_geometry[0][7], np.float64)
    candidate_pv_full = np.asarray(traced.stage_geometry[0][8], np.float64)
    candidate_pu = candidate_pu_full[:, 1:]
    candidate_pv = candidate_pv_full[1:]
    candidate_pw = (
        np.asarray(traced.stage_geometry[0][2], np.float64)
        * np.asarray(card.recipe.grid.area_T, np.float64)[..., None]
    )
    recorded_pu = candidate_pu.copy()
    recorded_pv = candidate_pv.copy()
    recorded_pw = candidate_pw.copy()
    recorded_pu[:, :90] = np.asarray(tracer["zFu"], np.float64)[..., :30]
    recorded_pv = np.asarray(record["zFv_after_trp"], np.float64)[..., :30]
    recorded_pw[:, :90] = np.asarray(tracer["zFw"], np.float64)
    recorded_pu_full = np.concatenate(
        [recorded_pu[:, -1:, :], recorded_pu], axis=1)
    recorded_pv_full = np.concatenate(
        [recorded_pv[-1:, :, :], recorded_pv], axis=0)

    entry_t = np.asarray(state.T.data, np.float64)
    entry_s = np.asarray(state.S.data, np.float64)
    nemo_entry_t = np.asarray(record["T_Kmm"], np.float64)[..., :30]
    nemo_entry_s = np.asarray(record["S_Kmm"], np.float64)[..., :30]
    h_stage = np.asarray(traced.stage_geometry[0][3], np.float64)

    cand_t_fu, cand_t_fv = _face_rows(
        entry_t, candidate_pu_full, candidate_pv_full, card.recipe.grid)
    nemo_t_fu, nemo_t_fv = _face_rows(
        nemo_entry_t, recorded_pu_full, recorded_pv_full, card.recipe.grid)
    cand_s_fu, cand_s_fv = _face_rows(
        entry_s, candidate_pu_full, candidate_pv_full, card.recipe.grid)
    nemo_s_fu, nemo_s_fv = _face_rows(
        nemo_entry_s, recorded_pu_full, recorded_pv_full, card.recipe.grid)

    # The record stores metric W transport.  Rebuild the source-written CEN2
    # W face products, including the live masks, through the production helper.
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _nemo_cen2_tracer_rhs
    tmask = np.asarray(card.recipe.z_coord.is_active, np.float64)
    candidate_t_rhs = np.asarray(_nemo_cen2_tracer_rhs(
        entry_t, candidate_pu_full, candidate_pv_full, candidate_pw,
        h_stage, tmask, card.recipe.grid), np.float64)
    recorded_t_rhs = np.asarray(_nemo_cen2_tracer_rhs(
        nemo_entry_t, recorded_pu_full, recorded_pv_full, recorded_pw,
        np.asarray(record["e3t_Kmm"], np.float64)[..., :30],
        np.asarray(record["tmask"], np.float64)[..., :30], card.recipe.grid),
        np.float64)
    candidate_s_rhs = np.asarray(_nemo_cen2_tracer_rhs(
        entry_s, candidate_pu_full, candidate_pv_full, candidate_pw,
        h_stage, tmask, card.recipe.grid), np.float64)
    recorded_s_rhs = np.asarray(_nemo_cen2_tracer_rhs(
        nemo_entry_s, recorded_pu_full, recorded_pv_full, recorded_pw,
        np.asarray(record["e3t_Kmm"], np.float64)[..., :30],
        np.asarray(record["tmask"], np.float64)[..., :30], card.recipe.grid),
        np.float64)

    # Publish write-only production boundaries after the ordinary step has
    # completed.  A second run with the admitted transports is the causal arm.
    boundary_hooks = base_hooks._replace(
        expose_tracer_stage1_boundary="after_advection")
    sbc_hooks = base_hooks._replace(expose_tracer_stage1_boundary="after_sbc")
    recorded_hooks = boundary_hooks._replace(
        stage1_tracer_transport_override=(recorded_pu, recorded_pv, recorded_pw))
    recorded_stage_hooks = base_hooks._replace(
        stage1_tracer_transport_override=(recorded_pu, recorded_pv, recorded_pw),
        expose_tracer_stage=1)
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        after_adv = jax.device_get(_run(card, state, forcing, boundary_hooks))
        after_sbc = jax.device_get(_run(card, state, forcing, sbc_hooks))
        recorded_after_adv = jax.device_get(_run(
            card, state, forcing, recorded_hooks))
        recorded_stage = jax.device_get(_run(
            card, state, forcing, recorded_stage_hooks))

    # Production CEN2 is the calibration endpoint for the offline helper.
    # The separate horizontal/vertical rows are diagnostic source-order splits.
    cand_t_h, cand_t_w = _divergence_rows(
        cand_t_fu, cand_t_fv,
        np.asarray(candidate_pw, np.float64), h_stage, card.recipe.grid)
    nemo_t_h, nemo_t_w = _divergence_rows(
        nemo_t_fu, nemo_t_fv,
        np.asarray(recorded_pw, np.float64),
        np.asarray(record["e3t_Kmm"], np.float64)[..., :30], card.recipe.grid)

    rows = {
        "entry_T": _score(_physical(entry_t), _physical(nemo_entry_t)),
        "entry_S": _score(_physical(entry_s), _physical(nemo_entry_s)),
        "zFu": _score(candidate_pu[-1, :90], recorded_pu[-1, :90]),
        "zFv": _score(candidate_pv[-1, :90], recorded_pv[-1, :90]),
        "zFw": _score(candidate_pw[-1, :90], recorded_pw[-1, :90]),
        "u_face_product_T": _score(
            cand_t_fu[-1, 1:91], nemo_t_fu[-1, 1:91]),
        "v_face_product_T": _score(
            cand_t_fv[-1, :90], nemo_t_fv[-1, :90]),
        "u_face_product_S": _score(
            cand_s_fu[-1, 1:91], nemo_s_fu[-1, 1:91]),
        "v_face_product_S": _score(
            cand_s_fv[-1, :90], nemo_s_fv[-1, :90]),
        "horizontal_divergence_T": _score(
            _physical(cand_t_h), _physical(nemo_t_h)),
        "vertical_divergence_T": _score(
            _physical(cand_t_w), _physical(nemo_t_w)),
        "after_advection_T": _score(
            _physical(after_adv.T.data),
            np.asarray(tracer["after_advection_T"])[-1, :, :30]),
        "after_advection_S": _score(
            _physical(after_adv.S.data),
            np.asarray(tracer["after_advection_S"])[-1, :, :30]),
        "after_sbc_T": _score(
            _physical(after_sbc.T.data),
            np.asarray(tracer["after_sbc_T"])[-1, :, :30]),
        "after_sbc_S": _score(
            _physical(after_sbc.S.data),
            np.asarray(tracer["after_sbc_S"])[-1, :, :30]),
        "qco_stage_T": _score(_physical(current_t), _physical(oracle_stage["T"])),
        "qco_stage_S": _score(_physical(current_s), _physical(oracle_stage["S"])),
        "post_stage_association": {"unmeasured": True},
    }

    return {
        "format": "nemo-testcase-l4-orca2-round257-tracer-writer-v1",
        "status": "PASS_R257_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved",
        "private_unit": True,
        "passivity": passivity,
        "support": "T-pivot physical half, j=147, i=0..89, k=0..29",
        "source_order": list(ORDER),
        "rows": rows,
        "helper_calibration": {
            "T": _score(_physical(candidate_t_rhs), _physical(after_adv.T.data)),
            "S": _score(_physical(candidate_s_rhs), _physical(after_adv.S.data)),
        },
        "recorded_transport_closure": {
            "after_advection_T": _score(
                _physical(recorded_after_adv.T.data),
                np.asarray(tracer["after_advection_T"])[-1, :, :30]),
            "after_advection_S": _score(
                _physical(recorded_after_adv.S.data),
                np.asarray(tracer["after_advection_S"])[-1, :, :30]),
            "qco_stage_T": _score(
                _physical(recorded_stage.T.data), _physical(oracle_stage["T"])),
            "qco_stage_S": _score(
                _physical(recorded_stage.S.data), _physical(oracle_stage["S"])),
        },
        "recorded_rhs": {
            "T": _score(_physical(recorded_t_rhs), _physical(candidate_t_rhs)),
            "S": _score(_physical(recorded_s_rhs), _physical(candidate_s_rhs)),
        },
        "current_tree_reproduction": {
            "T_max_abs": rows["qco_stage_T"]["max_abs"],
            "S_max_abs": rows["qco_stage_S"]["max_abs"],
            "round256_T": 0.17735225873138316,
            "round256_S": 3.283688696376231,
        },
        "one_bit_control": _one_bit_control(),
        "compiled_source": {
            "transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285",
            "cen2_horizontal": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90:149-161",
            "cen2_vertical": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90:202-227",
            "post_stage_association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {row["label"]: row for row in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "support":
        independent["support"] = "wrong support"
    elif plant == "source-order":
        independent["source_order"][2:4] = reversed(
            independent["source_order"][2:4])
    elif plant == "reproduction":
        independent["current_tree_reproduction"]["S_max_abs"] = 0.0
    elif plant == "first-owner":
        independent["rows"]["entry_T"]["max_abs"] = 1.0
    elif plant == "recorded-closure":
        independent["recorded_transport_closure"]["after_advection_S"][
            "max_abs"] = 4.0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    labels = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R257_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] == "production-d114-resolved"
                and report["private_unit"] is True,
                f"{label}: execution route moved")
        require(all(report["passivity"].values()),
                f"{label}: passive trace moved the state")
        require(report["support"] ==
                "T-pivot physical half, j=147, i=0..89, k=0..29",
                f"{label}: physical-half support moved")
        require(report["source_order"] == list(ORDER),
                f"{label}: source order moved")
        reproduction = report["current_tree_reproduction"]
        require(reproduction["T_max_abs"] == reproduction["round256_T"]
                and reproduction["S_max_abs"] == reproduction["round256_S"],
                f"{label}: round-256 current-tree boundary did not reproduce")
        require(report["helper_calibration"]["T"]["max_abs"] == 0.0
                and report["helper_calibration"]["S"]["max_abs"] == 0.0,
                f"{label}: literal helper did not reproduce production CEN2")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")

        measured_order = [name for name in ORDER
                          if not report["rows"][name].get("unmeasured", False)]
        first = next((name for name in measured_order
                      if report["rows"][name]["max_abs"] != 0.0), None)
        require(first is not None, f"{label}: no numerical owner was found")
        require(all(report["rows"][name]["max_abs"] == 0.0
                    for name in measured_order[:measured_order.index(first)]),
                f"{label}: an earlier numerical row was skipped")
        closure = report["recorded_transport_closure"]
        closure_max = max(float(closure[name]["max_abs"]) for name in closure)
        require(closure_max <= 0.0012,
                f"{label}: recorded transport did not close the boundary")
        labels[label] = {
            "first_numerical_statement": first,
            "first_row": report["rows"][first],
            "stage_boundary": {
                "T": report["rows"]["qco_stage_T"],
                "S": report["rows"]["qco_stage_S"],
            },
            "recorded_transport_closure": closure,
        }
    require(labels["independent"]["first_numerical_statement"] ==
            labels["given_nemo_entry"]["first_numerical_statement"],
            "claim labels disagree on the first numerical statement")
    first = labels["independent"]["first_numerical_statement"]
    return {
        "format": "nemo-testcase-l4-orca2-round257-classification-v1",
        "status": "HELD_R257_TRACER_WRITER_NAMED",
        "source_order": list(ORDER),
        "first_numerical_statement": first,
        "labels": labels,
        "predictions": {
            "R257-P1": "CONFIRMED",
            "R257-P2": "CONFIRMED",
            "R257-P3": "CONFIRMED" if first == "zFv" else "REFUTED",
            "R257-P4": "CONFIRMED_RECORDED_TRANSPORT_CLOSURE",
            "R257-P5": "CONFIRMED_MEASUREMENT_ONLY",
            "R257-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--operand-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.operand_root,
                args.label, args.expect_commit)), "measurement arguments are incomplete")
            result = measure(args.deck_root, args.frame_root, args.operand_root,
                             args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification JSONs are required")
            result = classify(
                [json.loads(path.read_text()) for path in args.classify],
                plant=args.plant)
    except (GateError, r231.GateError, r249.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

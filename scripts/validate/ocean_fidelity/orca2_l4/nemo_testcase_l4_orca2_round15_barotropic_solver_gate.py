#!/usr/bin/env python3
"""ORCA2 round-15 gate: check the last two inputs, then walk dynspg_ts.

The gate consumes the admitted rank-0 ``BTORD_2`` and ``SLOW_2`` records.
It first scores NEMO's sea-surface forcing and frozen drag coefficients at the
solver-copy boundary.  Only when those operands are bit-exact may it interpret
the production-JIT substep trace statement by statement.  A non-bit input is a
hard stop: walking the solver with a different input would misattribute an
inherited difference to its first consumer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))
_TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(_TESTCASES) not in sys.path:
    sys.path.insert(0, str(_TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round14_barotropic_owner_gate as round14,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "solver_copies_inputs": f"{_PP}/dynspg_ts.f90:287-291",
    "solver_initializes_entry": f"{_PP}/dynspg_ts.f90:355-364",
    "cold_history_initialization": f"{_PP}/dynspg_ts.f90:339-347",
    "predictor_weights": f"{_PP}/dynspg_ts.f90:460-493",
    "midstep_face_depth_and_transport": f"{_PP}/dynspg_ts.f90:505-536",
    "continuity_update": f"{_PP}/dynspg_ts.f90:550-558",
    "back_interpolation_and_pressure_gradient":
        f"{_PP}/dynspg_ts.f90:601-616",
    "coriolis_and_drag": f"{_PP}/dynspg_ts.f90:618-652",
    "vector_velocity_update": f"{_PP}/dynspg_ts.f90:666-679",
    "record_write_order": f"{_PP}/dynspg_ts.f90:755-779",
}

RANK0_DIMS = (94, 152)
RANK0_COLUMNS = 90
ROUND14_END_OF_STEP_SSH_M = 0.2448430937728719


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native(values: np.ndarray, stagger: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if stagger == "u":
        values = values[:, 1:, ...]
    elif stagger == "v":
        values = values[1:, :, ...]
    return values[:, :RANK0_COLUMNS, ...]


def _trace_native(trace: dict, name: str, stagger: str, rows: int = 2):
    values = np.asarray(trace[name], dtype=np.float64)[:rows]
    if stagger == "u":
        values = values[:, :, 1:]
    elif stagger == "v":
        values = values[:, 1:, :]
    return values[:, :, :RANK0_COLUMNS]


def _masks(card) -> dict[str, np.ndarray]:
    return {
        "t": np.asarray(card.recipe.initial_state.land_mask.data, dtype=bool)[
            :, :RANK0_COLUMNS],
        "u": _native(card.recipe.initial_state.u_mask.data, "u").astype(bool),
        "v": _native(card.recipe.initial_state.v_mask.data, "v").astype(bool),
    }


def _candidate_ordered(card, trace: dict) -> dict[str, np.ndarray]:
    two = 2
    stagger = {
        "u_entry": "u", "v_entry": "v",
        "u_history_b": "u", "v_history_b": "v",
        "u_history_bb": "u", "v_history_bb": "v",
        "eta_entry": "t", "eta_history_b": "t", "eta_history_bb": "t",
        "u_mid": "u", "v_mid": "v", "eta_mid": "t",
        "face_depth_u_mid": "u", "face_depth_v_mid": "v",
        "metric_transport_u": "u", "metric_transport_v": "v",
        "continuity_du": "t", "continuity_dv": "t",
        "continuity_divergence": "t", "continuity_forcing": "t",
        "eta_exit": "t", "face_ssh_u_exit": "u",
        "face_ssh_v_exit": "v", "eta_pgf": "t",
        "pgf_u": "u", "pgf_v": "v", "cor_u": "u", "cor_v": "v",
        "trd_u": "u", "trd_v": "v", "slow_u": "u", "slow_v": "v",
        "u_exit": "u", "v_exit": "v",
        "face_depth_u_exit": "u", "face_depth_v_exit": "v",
        "r1_face_depth_u_exit": "u", "r1_face_depth_v_exit": "v",
        "ffu_nw": "u", "ffu_ne": "u", "ffu_sw": "u", "ffu_se": "u",
        "ffv_sw": "v", "ffv_se": "v", "ffv_nw": "v", "ffv_ne": "v",
    }
    trace_names = {
        "face_depth_u_mid": "transport_face_depth_u",
        "face_depth_v_mid": "transport_face_depth_v",
        "metric_transport_u": "transport_metric_u",
        "metric_transport_v": "transport_metric_v",
        "eta_exit": "eta_exit",
    }
    out = {
        name: _trace_native(trace, trace_names.get(name, name), kind, two)
        for name, kind in stagger.items()
    }
    shape_t = (two, *out["eta_entry"].shape[1:])
    out.update({
        "metric_e2u": np.broadcast_to(
            _native(card.recipe.grid.dy_u, "u"), out["u_entry"].shape),
        "metric_e1v": np.broadcast_to(
            _native(card.recipe.grid.dx_v, "v"), out["v_entry"].shape),
        "r1_area": np.broadcast_to(
            1.0 / np.asarray(card.recipe.grid.area)[:, :RANK0_COLUMNS],
            shape_t),
        "r1_dx_u": np.broadcast_to(
            1.0 / _native(card.recipe.grid.dx_u, "u"),
            out["u_entry"].shape),
        "r1_dy_v": np.broadcast_to(
            1.0 / _native(card.recipe.grid.dy_v, "v"),
            out["v_entry"].shape),
    })
    return out


SOURCE_ORDER = (
    ("eta_entry", "t", "solver_initializes_entry"),
    ("u_entry", "u", "solver_initializes_entry"),
    ("v_entry", "v", "solver_initializes_entry"),
    ("u_history_b", "u", "cold_history_initialization"),
    ("v_history_b", "v", "cold_history_initialization"),
    ("u_history_bb", "u", "cold_history_initialization"),
    ("v_history_bb", "v", "cold_history_initialization"),
    ("eta_history_b", "t", "cold_history_initialization"),
    ("eta_history_bb", "t", "cold_history_initialization"),
    ("u_mid", "u", "predictor_weights"),
    ("v_mid", "v", "predictor_weights"),
    ("eta_mid", "t", "predictor_weights"),
    ("face_depth_u_mid", "u", "midstep_face_depth_and_transport"),
    ("face_depth_v_mid", "v", "midstep_face_depth_and_transport"),
    ("metric_transport_u", "u", "midstep_face_depth_and_transport"),
    ("metric_transport_v", "v", "midstep_face_depth_and_transport"),
    ("metric_e2u", "u", "midstep_face_depth_and_transport"),
    ("metric_e1v", "v", "midstep_face_depth_and_transport"),
    ("r1_area", "t", "continuity_update"),
    ("continuity_du", "t", "continuity_update"),
    ("continuity_dv", "t", "continuity_update"),
    ("continuity_divergence", "t", "continuity_update"),
    ("continuity_forcing", "t", "continuity_update"),
    ("eta_exit", "t", "continuity_update"),
    ("face_ssh_u_exit", "u", "back_interpolation_and_pressure_gradient"),
    ("face_ssh_v_exit", "v", "back_interpolation_and_pressure_gradient"),
    ("eta_pgf", "t", "back_interpolation_and_pressure_gradient"),
    ("r1_dx_u", "u", "back_interpolation_and_pressure_gradient"),
    ("r1_dy_v", "v", "back_interpolation_and_pressure_gradient"),
    ("pgf_u", "u", "back_interpolation_and_pressure_gradient"),
    ("pgf_v", "v", "back_interpolation_and_pressure_gradient"),
    ("cor_u", "u", "coriolis_and_drag"),
    ("cor_v", "v", "coriolis_and_drag"),
    ("trd_u", "u", "coriolis_and_drag"),
    ("trd_v", "v", "coriolis_and_drag"),
    ("slow_u", "u", "vector_velocity_update"),
    ("slow_v", "v", "vector_velocity_update"),
    ("u_exit", "u", "vector_velocity_update"),
    ("v_exit", "v", "vector_velocity_update"),
)


def run(deck_root: Path, root: Path, json_out: Path | None,
        plant: bool = False) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from nemo_testcase_l2_gyre_round14_advmean import read_ordered
    from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    ordered_path = root / "oracle_bt_ordered_operands_kt00000001.bin"
    slow_path = root / "oracle_slow_forcing_kt00000001.bin"
    require(ordered_path.is_file() and slow_path.is_file(),
            "the admitted ordered/slow records are missing")
    oracle = read_ordered(
        ordered_path, expected_dims=RANK0_DIMS, expected_nrows=2)
    slow = read_slow_forcing(slow_path, dims=(*RANK0_DIMS, 31))

    _, card = ladder.card_fields(deck_root)
    cfg = card.recipe.model_config
    resolved = {
        "dtype_eta": str(np.asarray(card.recipe.initial_state.eta.data).dtype),
        "barotropic_solver": cfg.barotropic.barotropic_solver,
        "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
        "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
        "barotropic_continuity_evaluation":
            cfg.barotropic.barotropic_continuity_evaluation,
        "barotropic_drag_substep": bool(cfg.barotropic_drag_substep),
        "outer_integrator": cfg.outer_integrator,
    }
    print("RESOLVED " + json.dumps(resolved, sort_keys=True))
    require(
        resolved["barotropic_solver"] == "explicit_substep"
        and resolved["n_barotropic_substeps"] == 65
        and resolved["barotropic_time_filter"] == "nemo_ab3am4"
        and resolved["barotropic_continuity_evaluation"] == "nemo_literal"
        and resolved["barotropic_drag_substep"]
        and resolved["outer_integrator"] == "forward_euler",
        "the resolved ORCA2 solver no longer matches the preregistration")

    entry = ladder.assemble_state_fields(root, 1, stage=None)
    state = round14._seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)

    print("STEP baseline production stage")
    baseline_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True))
    baseline = jax.device_get(baseline_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    oracle_ssh3 = ladder.read_state_frame(
        root / "oracle_stage_kt00000001_s3.bin", kt=1, stage=3)["ssh"]
    baseline_ssh = round14._ssh_row(baseline.stage_outputs, oracle_ssh3, 3)
    reproduction = {
        "round14_end_of_step_max_abs_m": ROUND14_END_OF_STEP_SSH_M,
        "measured_end_of_step_max_abs_m": baseline_ssh["max_abs_m"],
        "exactly_reproduced": bool(
            baseline_ssh["max_abs_m"] == ROUND14_END_OF_STEP_SSH_M),
        "relative_departure": float(abs(
            baseline_ssh["max_abs_m"] - ROUND14_END_OF_STEP_SSH_M)
            / ROUND14_END_OF_STEP_SSH_M),
    }
    require(reproduction["relative_departure"] <= 5.0e-5,
            "round 14's baseline sea-surface number did not reproduce")

    print("STEP production-JIT barotropic trace")
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    trace = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    substeps = trace.substeps
    masks = _masks(card)
    candidate_before = _candidate_ordered(card, substeps)

    input_rows = {
        "freshwater_forcing": round14.compare(
            candidate_before["continuity_forcing"][0],
            oracle["continuity_forcing"][0], masks["t"]),
        "drag_coefficient_u": round14.compare(
            _trace_native(substeps, "drag_coefficient_u", "u", 1)[0],
            slow["cd_u"], masks["u"]),
        "drag_coefficient_v": round14.compare(
            _trace_native(substeps, "drag_coefficient_v", "v", 1)[0],
            slow["cd_v"], masks["v"]),
    }
    print("INPUTS " + json.dumps(input_rows, sort_keys=True))

    # The freshwater replacement is already an identity.  Substitute the two
    # non-bit NEMO drag coefficients over rank 0's owned window, leaving rank 1
    # and every other solver input untouched.  The production trace exposes
    # NEMO's signed coefficients; the solver accepts legoESM's positive-rate
    # convention, hence the explicit minus sign on both sides.
    own_rate_u = -np.asarray(substeps["drag_coefficient_u"][0])
    own_rate_v = -np.asarray(substeps["drag_coefficient_v"][0])
    injected_rate_u, injected_rate_v = round14._inject(
        own_rate_u, own_rate_v, -slow["cd_u"], -slow["cd_v"])

    print("STEP no-op drag substitution control")
    noop_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_drag_rate_override=(own_rate_u, own_rate_v)))
    noop = jax.device_get(noop_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    noop_identical = bool(np.array_equal(
        np.asarray(noop.state_after_barotropic.eta.data),
        np.asarray(trace.state_after_barotropic.eta.data)))
    require(noop_identical,
            "drag substitution hook is not inert on legoESM's own rates")

    print("STEP recorded drag substitution trace")
    substituted_trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_drag_rate_override=(
                injected_rate_u, injected_rate_v)))
    substituted_trace = jax.device_get(substituted_trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    substituted_substeps = substituted_trace.substeps
    candidate = _candidate_ordered(card, substituted_substeps)
    landed_rows = {
        "drag_coefficient_u": round14.compare(
            _trace_native(
                substituted_substeps, "drag_coefficient_u", "u", 1)[0],
            slow["cd_u"], masks["u"]),
        "drag_coefficient_v": round14.compare(
            _trace_native(
                substituted_substeps, "drag_coefficient_v", "v", 1)[0],
            slow["cd_v"], masks["v"]),
    }
    require(all(row["bit_exact"] for row in landed_rows.values()),
            "recorded drag substitution did not land at the traced boundary")

    print("STEP recorded drag substitution production stage")
    substituted_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            barotropic_drag_rate_override=(
                injected_rate_u, injected_rate_v)))
    substituted = jax.device_get(substituted_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    substituted_ssh = round14._ssh_row(
        substituted.stage_outputs, oracle_ssh3, 3)
    baseline_field = np.asarray(baseline.stage_outputs[2][4])[
        :, :RANK0_COLUMNS]
    substituted_field = np.asarray(substituted.stage_outputs[2][4])[
        :, :RANK0_COLUMNS]
    arm_delta = np.abs(substituted_field - baseline_field)
    movement = {
        "max_abs_move_m": float(arm_delta.max()),
        "cells_moved": int(np.count_nonzero(arm_delta)),
        "scored_cells": int(arm_delta.size),
        "move_over_baseline_disagreement": float(
            arm_delta.max() / baseline_ssh["max_abs_m"]),
        "baseline_max_abs_m": baseline_ssh["max_abs_m"],
        "substituted_max_abs_m": substituted_ssh["max_abs_m"],
    }
    print("MEASURE recorded drag substitution moves end-of-step sea surface "
          + json.dumps(movement, sort_keys=True))

    # First non-bit boundary after the two requested input substitutions:
    # NEMO starts the kt=1 external loop from the recorded puu_b/pvv_b pair;
    # substitute that pair over the recorded half, then measure its causal
    # leverage before walking onward.
    entry_u, entry_v = round14._inject(
        np.asarray(state.uu_b.data), np.asarray(state.vv_b.data),
        oracle["u_entry"][0], oracle["v_entry"][0])
    state_entry = state._replace(
        uu_b=state.uu_b.replace(data=jnp.asarray(entry_u)),
        vv_b=state.vv_b.replace(data=jnp.asarray(entry_v)))
    entry_hooks = _NEMOWSRK3TestHooks(
        expose_barotropic_substeps=True,
        barotropic_drag_rate_override=(injected_rate_u, injected_rate_v))
    print("STEP recorded entry-velocity substitution trace")
    entry_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=entry_hooks).step(
            state_entry, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    entry_candidate = _candidate_ordered(card, entry_trace.substeps)

    print("STEP recorded entry-velocity substitution production stage")
    entry_stage = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            barotropic_drag_rate_override=(
                injected_rate_u, injected_rate_v))).step(
                    state_entry, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
    entry_stage_field = np.asarray(entry_stage.stage_outputs[2][4])[
        :, :RANK0_COLUMNS]
    entry_delta = np.abs(entry_stage_field - substituted_field)
    entry_movement = {
        "max_abs_move_m": float(entry_delta.max()),
        "cells_moved": int(np.count_nonzero(entry_delta)),
        "move_over_baseline_disagreement": float(
            entry_delta.max() / baseline_ssh["max_abs_m"]),
    }
    print("MEASURE recorded entry velocity moves end-of-step sea surface "
          + json.dumps(entry_movement, sort_keys=True))

    # The next source statement is the ll_init b/bb initialization.  Build a
    # one-variable history arm from the record's own raw arrays.  Rank 1 stays
    # on legoESM's own history because no rank-1 ordered record exists.
    def inject_t(full, rank0):
        out = np.array(full, dtype=np.float64, copy=True)
        require(out[:, :RANK0_COLUMNS].shape == rank0.shape,
                "T-point history injection window shape mismatch")
        out[:, :RANK0_COLUMNS] = rank0
        return out

    hist = entry_trace.substeps
    ub, vb = round14._inject(
        np.asarray(hist["u_history_b"][0]),
        np.asarray(hist["v_history_b"][0]),
        oracle["u_history_b"][0], oracle["v_history_b"][0])
    ubb, vbb = round14._inject(
        np.asarray(hist["u_history_bb"][0]),
        np.asarray(hist["v_history_bb"][0]),
        oracle["u_history_bb"][0], oracle["v_history_bb"][0])
    etab = inject_t(
        np.asarray(hist["eta_history_b"][0]), oracle["eta_history_b"][0])
    etabb = inject_t(
        np.asarray(hist["eta_history_bb"][0]), oracle["eta_history_bb"][0])
    raw_history = tuple(jnp.asarray(value) for value in (
        ub, ubb, vb, vbb, etab, etabb))

    print("STEP recorded cold-history substitution trace")
    history_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_drag_rate_override=(
                injected_rate_u, injected_rate_v),
            barotropic_raw_history_override=raw_history)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    history_candidate = _candidate_ordered(card, history_trace.substeps)

    print("STEP recorded cold-history substitution production stage")
    history_stage = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            barotropic_drag_rate_override=(
                injected_rate_u, injected_rate_v),
            barotropic_raw_history_override=raw_history)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    history_stage_field = np.asarray(history_stage.stage_outputs[2][4])[
        :, :RANK0_COLUMNS]
    history_delta = np.abs(history_stage_field - entry_stage_field)
    history_ssh = round14._ssh_row(
        history_stage.stage_outputs, oracle_ssh3, 3)
    history_movement = {
        "max_abs_move_m": float(history_delta.max()),
        "cells_moved": int(np.count_nonzero(history_delta)),
        "move_over_baseline_disagreement": float(
            history_delta.max() / baseline_ssh["max_abs_m"]),
        "combined_substituted_max_abs_m": history_ssh["max_abs_m"],
    }
    print("MEASURE recorded cold histories move end-of-step sea surface "
          + json.dumps(history_movement, sort_keys=True))

    inputs_ready = (
        input_rows["freshwater_forcing"]["bit_exact"]
        and all(row["bit_exact"] for row in landed_rows.values()))
    substitution = {
        "freshwater_replacement_is_identity":
            input_rows["freshwater_forcing"]["bit_exact"],
        "drag_replacement_is_identity": False,
        "noop_control_solver_eta_bit_exact": noop_identical,
        "replacement_landed": landed_rows,
        "end_of_step_sea_surface_movement": movement,
    }

    def score_walk(candidate_fields):
        rows = []
        first = None
        # Execution order is every statement in sub-step 1, then every
        # statement in sub-step 2.  Aggregating the two rows hid which loop
        # iteration first diverged (the original gate reported 8568/17136 U
        # entries without saying all 8568 belonged to only one sub-step).
        for substep in range(2):
            for name, stagger, citation in SOURCE_ORDER:
                row = round14.compare(
                    candidate_fields[name][substep],
                    oracle[name][substep], masks[stagger])
                row.update({
                    "substep": substep + 1,
                    "boundary": name,
                    "citation": CITATIONS[citation],
                })
                rows.append(row)
                if first is None and not row["bit_exact"]:
                    first = dict(row)
        return rows, first

    walk_rows = []
    first_non_bit = None
    entry_walk_rows = []
    first_after_entry = None
    history_walk_rows = []
    first_after_history = None
    if inputs_ready:
        walk_rows, first_non_bit = score_walk(candidate)
        entry_walk_rows, first_after_entry = score_walk(entry_candidate)
        history_walk_rows, first_after_history = score_walk(history_candidate)
    else:
        print("STOP_INPUT_SUBSTITUTION_FAILED: a recorded solver input did "
              "not reach the traced boundary; no downstream row is attributed")

    plant_result = {"requested": plant, "fires": None}
    if plant:
        channels = {}
        for name, source, active in (
                ("freshwater_forcing", oracle["continuity_forcing"][0],
                 masks["t"]),
                ("drag_coefficient_u", slow["cd_u"], masks["u"])):
            planted = np.array(source, copy=True)
            index = tuple(np.argwhere(
                active & np.isfinite(planted) & (planted != 0.0))[0])
            planted[index] = np.nextafter(planted[index], np.inf)
            row = round14.compare(planted, source, active)
            fired = bool(
                not row["bit_exact"] and row["differing_cells"] == 1
                and row["ulp_max"] == 1)
            channels[name] = {
                "fires": fired, "index": list(index), "row": row}
        fired = all(channel["fires"] for channel in channels.values())
        plant_result.update({"fires": fired, "channels": channels})
        print("PLANT FIRED" if fired else "PLANT DID NOT FIRE")
        require(fired, "one-ULP input-channel plant did not fire")

    result = {
        "gate": "nemo_testcase_l4_orca2_round15_barotropic_solver_gate",
        "label": "given NEMO's entry",
        "record_root": str(root),
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "resolved": resolved,
        "round14_reproduction": reproduction,
        "baseline_end_of_step_sea_surface": baseline_ssh,
        "substituted_end_of_step_sea_surface": substituted_ssh,
        "solver_input_rows": input_rows,
        "substitution": substitution,
        "entry_velocity_substitution": entry_movement,
        "cold_history_substitution": history_movement,
        "walk_status": (
            "WALKED_TO_FIRST_NON_BIT" if inputs_ready
            else "STOP_INPUT_SUBSTITUTION_FAILED"),
        "first_non_bit_statement": first_non_bit,
        "source_ordered_rows": walk_rows,
        "first_non_bit_after_entry_velocity_substitution": first_after_entry,
        "source_ordered_rows_after_entry_velocity_substitution":
            entry_walk_rows,
        "first_non_bit_after_cold_history_substitution": first_after_history,
        "source_ordered_rows_after_cold_history_substitution":
            history_walk_rows,
        "plant": plant_result,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.deck_root, args.record_root, args.json_out,
                     plant=args.plant)
    except GateError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "gate": result["gate"],
        "walk_status": result["walk_status"],
        "solver_input_rows": result["solver_input_rows"],
        "first_non_bit_statement": result["first_non_bit_statement"],
        "first_non_bit_after_entry_velocity_substitution":
            result["first_non_bit_after_entry_velocity_substitution"],
        "first_non_bit_after_cold_history_substitution":
            result["first_non_bit_after_cold_history_substitution"],
        "entry_velocity_substitution": result["entry_velocity_substitution"],
        "cold_history_substitution": result["cold_history_substitution"],
        "round14_reproduction": result["round14_reproduction"],
        "plant": result["plant"],
    }, indent=2, sort_keys=True))
    if args.plant:
        return 1 if result["plant"]["fires"] else 2
    return 0 if result["walk_status"] == "WALKED_TO_FIRST_NON_BIT" else 3


if __name__ == "__main__":
    raise SystemExit(main())

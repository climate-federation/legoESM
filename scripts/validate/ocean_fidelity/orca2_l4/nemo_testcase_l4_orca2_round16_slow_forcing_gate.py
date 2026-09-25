#!/usr/bin/env python3
"""ORCA2 round-16 gate: substitute slow forcing and replay vector update."""

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
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round15_barotropic_solver_gate as round15,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "vector_velocity_update": f"{_PP}/dynspg_ts.f90:666-679",
    "record_write_order": f"{_PP}/dynspg_ts.f90:755-779",
}

ROUND15_BASELINE_SSH_M = 0.2448430937728719
ROUND15_HISTORY_MOVE_M = 0.0
ROUND15_SLOW_U_MAX = 5.370080135032166e-12
ROUND15_SLOW_V_MAX = 2.2928581685638914e-11


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _inject_t(full: np.ndarray, rank0: np.ndarray) -> np.ndarray:
    result = np.array(full, dtype=np.float64, copy=True)
    require(
        result[:, :round15.RANK0_COLUMNS].shape == rank0.shape,
        "T-point rank-0 injection window shape mismatch",
    )
    result[:, :round15.RANK0_COLUMNS] = rank0
    return result


def _history_override(trace: dict, oracle: dict):
    import jax.numpy as jnp

    ub, vb = round14._inject(
        np.asarray(trace["u_history_b"][0]),
        np.asarray(trace["v_history_b"][0]),
        oracle["u_history_b"][0], oracle["v_history_b"][0],
    )
    ubb, vbb = round14._inject(
        np.asarray(trace["u_history_bb"][0]),
        np.asarray(trace["v_history_bb"][0]),
        oracle["u_history_bb"][0], oracle["v_history_bb"][0],
    )
    etab = _inject_t(
        np.asarray(trace["eta_history_b"][0]), oracle["eta_history_b"][0])
    etabb = _inject_t(
        np.asarray(trace["eta_history_bb"][0]), oracle["eta_history_bb"][0])
    return tuple(jnp.asarray(value) for value in (
        ub, ubb, vb, vbb, etab, etabb))


def _score_walk(candidate: dict, oracle: dict, masks: dict):
    rows = []
    first = None
    for substep in range(2):
        for name, stagger, citation in round15.SOURCE_ORDER:
            row = round14.compare(
                candidate[name][substep], oracle[name][substep],
                masks[stagger])
            row.update({
                "substep": substep + 1,
                "boundary": name,
                "citation": round15.CITATIONS[citation],
            })
            rows.append(row)
            if first is None and not row["bit_exact"]:
                first = dict(row)
    return rows, first


def _arithmetic_values(fields: dict, face: str, substep: int,
                       dt: np.float64, mask: np.ndarray):
    pgf = np.asarray(fields[f"pgf_{face}"][substep], dtype=np.float64)
    trend = np.asarray(fields[f"trd_{face}"][substep], dtype=np.float64)
    slow = np.asarray(fields[f"slow_{face}"][substep], dtype=np.float64)
    entry = np.asarray(fields[f"{face}_entry"][substep], dtype=np.float64)
    pgf_plus_trend = np.add(pgf, trend)
    rhs_plus_slow = np.add(pgf_plus_trend, slow)
    increment = np.multiply(dt, rhs_plus_slow)
    entry_plus_increment = np.add(entry, increment)
    masked_exit = np.multiply(entry_plus_increment, mask.astype(np.float64))
    return {
        "pgf_plus_trend": pgf_plus_trend,
        "rhs_plus_slow": rhs_plus_slow,
        "increment": increment,
        "entry_plus_increment": entry_plus_increment,
        "masked_exit": masked_exit,
    }


def _arithmetic_walk(candidate: dict, oracle: dict, masks: dict):
    dt = np.float64(oracle["dt"])
    rows = {}
    controls = {}
    first = None
    for face in ("u", "v"):
        mask = masks[face]
        candidate_values = _arithmetic_values(candidate, face, 0, dt, mask)
        oracle_values = _arithmetic_values(oracle, face, 0, dt, mask)
        face_rows = {}
        for boundary in (
            "pgf_plus_trend", "rhs_plus_slow", "increment",
            "entry_plus_increment", "masked_exit",
        ):
            row = round14.compare(
                candidate_values[boundary], oracle_values[boundary], mask)
            face_rows[boundary] = row
            if first is None and not row["bit_exact"]:
                first = {"face": face, "boundary": boundary, **row}
        replay_oracle = round14.compare(
            oracle_values["masked_exit"], oracle[f"{face}_exit"][0], mask)
        replay_candidate = round14.compare(
            candidate_values["masked_exit"],
            candidate[f"{face}_exit"][0], mask)
        controls[face] = {
            "oracle_replay_matches_record": replay_oracle,
            "candidate_replay_matches_trace": replay_candidate,
        }
        require(replay_oracle["bit_exact"],
                f"{face.upper()} arithmetic replay does not reproduce NEMO")
        require(replay_candidate["bit_exact"],
                f"{face.upper()} arithmetic replay does not reproduce legoESM")
        rows[face] = face_rows
    return rows, controls, first


def _hooks(_NEMOWSRK3TestHooks, *, expose_trace: bool, drag, history, slow=None):
    return _NEMOWSRK3TestHooks(
        expose_barotropic_substeps=expose_trace,
        expose_live_stage_operands=not expose_trace,
        barotropic_drag_rate_override=drag,
        barotropic_raw_history_override=history,
        barotropic_slow_forcing_override=slow,
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
        ordered_path, expected_dims=round15.RANK0_DIMS, expected_nrows=2)
    slow_record = read_slow_forcing(
        slow_path, dims=(*round15.RANK0_DIMS, 31))

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
        "recorded_substep_dt_s": oracle["dt"],
    }
    print("RESOLVED " + json.dumps(resolved, sort_keys=True))
    require(
        resolved["barotropic_solver"] == "explicit_substep"
        and resolved["n_barotropic_substeps"] == 65
        and resolved["barotropic_time_filter"] == "nemo_ab3am4"
        and resolved["barotropic_continuity_evaluation"] == "nemo_literal"
        and resolved["barotropic_drag_substep"]
        and resolved["outer_integrator"] == "forward_euler",
        "the resolved ORCA2 solver no longer matches the preregistration",
    )

    entry = ladder.assemble_state_fields(root, 1, stage=None)
    state = round14._seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    masks = round15._masks(card)

    print("STEP baseline trace for inherited drag operands")
    baseline_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True)).step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    own_rate_u = -np.asarray(
        baseline_trace.substeps["drag_coefficient_u"][0])
    own_rate_v = -np.asarray(
        baseline_trace.substeps["drag_coefficient_v"][0])
    injected_rate_u, injected_rate_v = round14._inject(
        own_rate_u, own_rate_v, -slow_record["cd_u"], -slow_record["cd_v"])
    drag_override = (injected_rate_u, injected_rate_v)

    entry_u, entry_v = round14._inject(
        np.asarray(state.uu_b.data), np.asarray(state.vv_b.data),
        oracle["u_entry"][0], oracle["v_entry"][0])
    state_entry = state._replace(
        uu_b=state.uu_b.replace(data=jnp.asarray(entry_u)),
        vv_b=state.vv_b.replace(data=jnp.asarray(entry_v)))

    print("STEP entry trace for inherited cold histories")
    entry_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_drag_rate_override=drag_override)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    history_override = _history_override(entry_trace.substeps, oracle)

    history_trace_hooks = _hooks(
        _NEMOWSRK3TestHooks, expose_trace=True,
        drag=drag_override, history=history_override)
    print("STEP inherited cold-history trace")
    history_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=history_trace_hooks).step(
            state_entry, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
    history_candidate = round15._candidate_ordered(
        card, history_trace.substeps)
    inherited_rows, inherited_first = _score_walk(
        history_candidate, oracle, masks)
    require(inherited_first is not None
            and inherited_first["substep"] == 1
            and inherited_first["boundary"] == "slow_u"
            and inherited_first["absolute_max"] == ROUND15_SLOW_U_MAX,
            "round 15's first active slow-forcing row did not reproduce")
    inherited_slow_v = next(
        row for row in inherited_rows
        if row["substep"] == 1 and row["boundary"] == "slow_v")
    require(inherited_slow_v["absolute_max"] == ROUND15_SLOW_V_MAX,
            "round 15's slow_v maximum did not reproduce")

    print("STEP inherited cold-history production stage")
    history_stage = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_hooks(
            _NEMOWSRK3TestHooks, expose_trace=False,
            drag=drag_override, history=history_override)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))

    own_slow_u = np.asarray(history_trace.substeps["slow_u"][0])
    own_slow_v = np.asarray(history_trace.substeps["slow_v"][0])
    own_slow = (own_slow_u, own_slow_v)
    recorded_slow_u, recorded_slow_v = round14._inject(
        own_slow_u, own_slow_v, oracle["slow_u"][0], oracle["slow_v"][0])
    recorded_slow = (recorded_slow_u, recorded_slow_v)

    print("STEP own-slow no-op controls")
    noop_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_hooks(
            _NEMOWSRK3TestHooks, expose_trace=True, drag=drag_override,
            history=history_override, slow=own_slow)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    noop_stage = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_hooks(
            _NEMOWSRK3TestHooks, expose_trace=False, drag=drag_override,
            history=history_override, slow=own_slow)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    noop_controls = {
        "u_exit_trace_bit_exact": bool(np.array_equal(
            np.asarray(noop_trace.substeps["u_exit"]),
            np.asarray(history_trace.substeps["u_exit"]))),
        "v_exit_trace_bit_exact": bool(np.array_equal(
            np.asarray(noop_trace.substeps["v_exit"]),
            np.asarray(history_trace.substeps["v_exit"]))),
        "end_of_step_ssh_bit_exact": bool(np.array_equal(
            np.asarray(noop_stage.stage_outputs[2][4]),
            np.asarray(history_stage.stage_outputs[2][4]))),
    }
    require(all(noop_controls.values()),
            "own slow-forcing substitution is not a no-op")

    print("STEP recorded slow-forcing trace")
    substituted_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_hooks(
            _NEMOWSRK3TestHooks, expose_trace=True, drag=drag_override,
            history=history_override, slow=recorded_slow)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    substituted_candidate = round15._candidate_ordered(
        card, substituted_trace.substeps)
    substituted_rows, substituted_first = _score_walk(
        substituted_candidate, oracle, masks)
    landed = {
        face: round14.compare(
            substituted_candidate[f"slow_{face}"][0],
            oracle[f"slow_{face}"][0], masks[face])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in landed.values()),
            "recorded slow forcing did not land at the traced boundary")

    before_slow = []
    for name, stagger, _ in round15.SOURCE_ORDER:
        if name == "slow_u":
            break
        before_slow.append({
            "boundary": name,
            **round14.compare(
                history_candidate[name][0], substituted_candidate[name][0],
                masks[stagger]),
        })
    require(all(row["bit_exact"] for row in before_slow),
            "slow-forcing substitution moved an earlier substep-1 boundary")

    print("STEP recorded slow-forcing production stage")
    substituted_stage = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_hooks(
            _NEMOWSRK3TestHooks, expose_trace=False, drag=drag_override,
            history=history_override, slow=recorded_slow)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    baseline_stage_field = np.asarray(history_stage.stage_outputs[2][4])[
        :, :round15.RANK0_COLUMNS]
    substituted_stage_field = np.asarray(substituted_stage.stage_outputs[2][4])[
        :, :round15.RANK0_COLUMNS]
    stage_delta = np.abs(substituted_stage_field - baseline_stage_field)
    oracle_ssh3 = ladder.read_state_frame(
        root / "oracle_stage_kt00000001_s3.bin", kt=1, stage=3)["ssh"]
    baseline_ssh = round14._ssh_row(history_stage.stage_outputs, oracle_ssh3, 3)
    substituted_ssh = round14._ssh_row(
        substituted_stage.stage_outputs, oracle_ssh3, 3)
    require(abs(baseline_ssh["max_abs_m"] - 0.24444708169024776) <= 0.0,
            "round 15's combined substituted SSH maximum did not reproduce")
    movement = {
        "max_abs_move_m": float(stage_delta.max()),
        "cells_moved": int(np.count_nonzero(stage_delta)),
        "scored_cells": int(stage_delta.size),
        "move_over_round15_baseline": float(
            stage_delta.max() / ROUND15_BASELINE_SSH_M),
        "inherited_max_abs_m": baseline_ssh["max_abs_m"],
        "substituted_max_abs_m": substituted_ssh["max_abs_m"],
    }

    inherited_arithmetic, inherited_replay, inherited_arithmetic_first = (
        _arithmetic_walk(history_candidate, oracle, masks))
    substituted_arithmetic, substituted_replay, substituted_arithmetic_first = (
        _arithmetic_walk(substituted_candidate, oracle, masks))

    plant_result = {"requested": plant, "fires": None}
    if plant:
        planted_native = np.array(oracle["slow_u"][0], copy=True)
        index = tuple(np.argwhere(
            masks["u"] & np.isfinite(planted_native)
            & (planted_native != 0.0))[0])
        planted_native[index] = np.nextafter(planted_native[index], np.inf)
        planted_u, planted_v = round14._inject(
            recorded_slow_u, recorded_slow_v,
            planted_native, oracle["slow_v"][0])
        print("STEP one-ULP slow-forcing plant trace")
        planted_trace = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_hooks(
                _NEMOWSRK3TestHooks, expose_trace=True, drag=drag_override,
                history=history_override, slow=(planted_u, planted_v))).step(
                    state_entry, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
        planted_candidate = round15._candidate_ordered(
            card, planted_trace.substeps)
        slow_row = round14.compare(
            planted_candidate["slow_u"][0], oracle["slow_u"][0], masks["u"])
        exit_row = round14.compare(
            planted_candidate["u_exit"][0],
            substituted_candidate["u_exit"][0], masks["u"])
        fires = bool(
            slow_row["differing_cells"] == 1
            and slow_row["ulp_max"] == 1
            and exit_row["differing_cells"] >= 1)
        plant_result.update({
            "fires": fires,
            "native_index": round15._json_index(index),
            "slow_row": slow_row,
            "exit_movement": exit_row,
        })
        print("PLANT FIRED" if fires else "PLANT DID NOT FIRE")
        require(fires, "one-ULP slow-forcing plant did not reach u_exit")

    result = {
        "gate": "nemo_testcase_l4_orca2_round16_slow_forcing_gate",
        "label": "given NEMO's entry",
        "record_root": str(root),
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "resolved": resolved,
        "round15_reproduction": {
            "history_movement_m": ROUND15_HISTORY_MOVE_M,
            "first_non_bit": inherited_first,
            "slow_v": inherited_slow_v,
            "combined_substituted_ssh_max_abs_m": baseline_ssh["max_abs_m"],
        },
        "noop_controls": noop_controls,
        "landed_slow_forcing": landed,
        "earlier_boundaries_unchanged": before_slow,
        "end_of_step_sea_surface_movement": movement,
        "first_non_bit_after_slow_forcing_substitution": substituted_first,
        "source_ordered_rows_after_slow_forcing_substitution":
            substituted_rows,
        "inherited_arithmetic_rows": inherited_arithmetic,
        "inherited_replay_controls": inherited_replay,
        "first_non_bit_inherited_arithmetic": inherited_arithmetic_first,
        "substituted_arithmetic_rows": substituted_arithmetic,
        "substituted_replay_controls": substituted_replay,
        "first_non_bit_substituted_arithmetic": substituted_arithmetic_first,
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
        result = run(
            args.deck_root, args.record_root, args.json_out, plant=args.plant)
    except GateError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "gate": result["gate"],
        "round15_reproduction": result["round15_reproduction"],
        "noop_controls": result["noop_controls"],
        "landed_slow_forcing": result["landed_slow_forcing"],
        "end_of_step_sea_surface_movement":
            result["end_of_step_sea_surface_movement"],
        "first_non_bit_inherited_arithmetic":
            result["first_non_bit_inherited_arithmetic"],
        "first_non_bit_substituted_arithmetic":
            result["first_non_bit_substituted_arithmetic"],
        "first_non_bit_after_slow_forcing_substitution":
            result["first_non_bit_after_slow_forcing_substitution"],
        "plant": result["plant"],
    }, indent=2, sort_keys=True))
    if args.plant:
        return 1 if result["plant"]["fires"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

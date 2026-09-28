#!/usr/bin/env python3
"""Gate GYRE kt=2 live WS-RK3 operands and raw barotropic-history substitution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_entry,
    require,
    score,
)
from nemo_testcase_l2_gyre_round46_kt2_stage_gate import (
    _owned2,
    _owned3,
    read_stage,
)
from nemo_testcase_l2_gyre_round48_bt_memory_gate import read_record


STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage")
MEMORY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48/oracle_bt_memory")
OPERATORS = {
    1: ("hpg", "ldf", "vor", "wzv", "keg", "zad", "transport"),
    2: ("hpg", "vor", "wzv", "keg", "zad", "transport"),
    3: ("hpg", "vor", "wzv", "keg", "zad", "ldf", "zdf", "transport"),
}


def _u(value):
    return np.asarray(value)[:, 1:, ...]


def _v(value):
    return np.asarray(value)[1:, ...]


def _raw_history(end):
    """Map NEMO east/north owned raw arrays to legoESM west/south faces."""
    arrays = end["arrays"]
    ub = _owned2(arrays["ub_e"])
    ubb = _owned2(arrays["ubb_e"])
    vb = _owned2(arrays["vb_e"])
    vbb = _owned2(arrays["vbb_e"])
    sshb = _owned2(arrays["sshb_e"])
    sshbb = _owned2(arrays["sshbb_e"])
    return (
        jnp.asarray(np.concatenate([ub[:, -1:], ub], axis=1)),
        jnp.asarray(np.concatenate([ubb[:, -1:], ubb], axis=1)),
        jnp.asarray(np.concatenate([np.zeros_like(vb[:1]), vb], axis=0)),
        jnp.asarray(np.concatenate([np.zeros_like(vbb[:1]), vbb], axis=0)),
        jnp.asarray(sshb),
        jnp.asarray(sshbb),
    )


def _masked_row(name, oracle, candidate, mask, *, stage, operator, field,
                plant=False):
    row = score(name, oracle, candidate, mask, plant=plant)
    row.update({"stage": stage, "operator": operator, "field": field})
    return row


def _captured_row(stage, field, value, reason):
    value = np.asarray(value)
    return {
        "name": f"GYRE-zco.kt2.s{stage}.transport.{field}",
        "stage": stage,
        "operator": "transport",
        "field": field,
        "status": "UNMEASURED",
        "reason": reason,
        "candidate_dtype": str(value.dtype),
        "candidate_shape": list(value.shape),
        "candidate_max_abs": float(np.max(np.abs(value))),
    }


def _operand_rows(trace, records, masks, *, plant, rho_0):
    rows = []
    for stage in (1, 2, 3):
        a = records[stage]["arrays"]
        state_u, state_v, state_T, state_S, state_ssh = trace.stage_states[stage - 1]
        operands = trace.operator_operands[stage - 1]
        geom = trace.stage_geometry[stage - 1]
        r3t, r3u, r3v = trace.stage_qco[stage - 1]
        tmask = _owned3(a["tmask"]) > 0.5
        umask = _owned3(a["umask"]) > 0.5
        vmask = _owned3(a["vmask"]) > 0.5
        tmask2 = tmask[..., 0]
        umask2 = umask[..., 0]
        vmask2 = vmask[..., 0]

        def add(operator, field, oracle, candidate, mask):
            rows.append(_masked_row(
                f"GYRE-zco.kt2.s{stage}.{operator}.{field}",
                oracle, candidate, mask, stage=stage, operator=operator,
                field=field,
                plant=(plant == "operand" and stage == 1
                       and operator == "hpg" and field == "u_Kmm")))

        # HPG receives Kmm T/S/ssh and the density/thickness materialized from
        # exactly that state (stp2d:141-145; stprk3_stg:431-452).
        add("hpg", "u_Kmm", _owned3(a["u_Kmm"]), _u(state_u), umask)
        add("hpg", "v_Kmm", _owned3(a["v_Kmm"]), _v(state_v), vmask)
        add("hpg", "T_Kmm", _owned3(a["T_Kmm"]), state_T, tmask)
        add("hpg", "S_Kmm", _owned3(a["S_Kmm"]), state_S, tmask)
        add("hpg", "ssh_Kmm", _owned2(a["ssh_Kmm"]), state_ssh, tmask2)
        add("hpg", "rhd", _owned3(a["rhd_in"]),
            operands["operand_rho_prime"] / rho_0, tmask)
        add("hpg", "e3t_Kmm", _owned3(a["e3t_Kmm"]),
            operands["operand_h_k"], tmask)

        # VOR/KEG/ZAD all read the same Kmm velocity. VOR and ZAD additionally
        # consume the canonical live face thickness; ZAD consumes ww.
        for operator in ("vor", "keg", "zad"):
            add(operator, "u_Kmm", _owned3(a["u_Kmm"]),
                _u(operands["operand_velocity_u"]), umask)
            add(operator, "v_Kmm", _owned3(a["v_Kmm"]),
                _v(operands["operand_velocity_v"]), vmask)
        for operator in ("vor", "zad"):
            add(operator, "e3u_Kmm", _owned3(a["e3u_Kmm"]),
                _u(operands["operand_momentum_h_u"]), umask)
            add(operator, "e3v_Kmm", _owned3(a["e3v_Kmm"]),
                _v(operands["operand_momentum_h_v"]), vmask)
        w_nlev = np.asarray(geom[2]).shape[-1]
        wmask = _owned3(a["wmask"], w_nlev) > 0.5
        add("zad", "ww", _owned3(a["ww"], w_nlev), geom[2], wmask)
        add("wzv", "u_Kmm", _owned3(a["u_Kmm"]), _u(state_u), umask)
        add("wzv", "v_Kmm", _owned3(a["v_Kmm"]), _v(state_v), vmask)
        add("wzv", "e3t_Kmm", _owned3(a["e3t_Kmm"]), geom[3], tmask)
        add("wzv", "e3u_Kmm", _owned3(a["e3u_Kmm"]), _u(geom[4]), umask)
        add("wzv", "e3v_Kmm", _owned3(a["e3v_Kmm"]), _v(geom[5]), vmask)
        add("wzv", "ww", _owned3(a["ww"], w_nlev), geom[2], wmask)
        add("wzv", "r1_Dt", np.asarray([a["r1_Dt"]]),
            np.asarray([1.0 / operands["operand_zad_continuity_dt"]]),
            np.ones(1, dtype=bool))

        # r3 fields are operands of the macro thickness families above.
        add("transport", "r3t_Kmm", _owned2(a["r3t_Kmm"]), r3t, tmask2)
        add("transport", "r3u_Kmm", _owned2(a["r3u_Kmm"]),
            _u(r3u)[..., 0], umask2)
        add("transport", "r3v_Kmm", _owned2(a["r3v_Kmm"]),
            _v(r3v)[..., 0], vmask2)

        # LDF executes only at stages 1 and 3. The live implementation's one
        # h_k operand is scored against NEMO's Kbb family, exposing any Kmm/Kbb
        # selection difference instead of hiding it in a given-input replay.
        if stage in (1, 3):
            add("ldf", "u_Kbb", _owned3(a["u_Kbb"]),
                _u(operands["operand_ldf_velocity_u"]), umask)
            add("ldf", "v_Kbb", _owned3(a["v_Kbb"]),
                _v(operands["operand_ldf_velocity_v"]), vmask)
            add("ldf", "e3t_Kbb", _owned3(a["e3t_Kbb"]),
                operands["operand_h_k"], tmask)

        # NEMO's kt=2 named stage bundle has no zFu/zFv/corrected-velocity
        # payload. Capture them but refuse a fabricated reconstruction.
        missing = "round-46 kt=2 stage record has no direct operand payload"
        rows.extend([
            _captured_row(stage, "zFu", _u(geom[7]), missing),
            _captured_row(stage, "zFv", _v(geom[8]), missing),
            _captured_row(stage, "corrected_u", _u(geom[9]), missing),
            _captured_row(stage, "corrected_v", _v(geom[10]), missing),
        ])

    return rows


def _state_rows(prefix, state, oracle_entry, masks):
    fields = lego_fields(state)
    rows = []
    for name in ("u", "v", "ssh"):
        reference = oracle_entry[name]
        if name != "ssh":
            reference = reference[..., :fields[name].shape[-1]]
        rows.append(score(
            f"{prefix}.{name}", reference, fields[name], masks[name]))
    return rows


def run(stage_root: Path, memory_root: Path, expect_commit: str,
        plant: str | None = None):
    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"].lower() == expected,
            f"producer commit mismatch: {stamp['commit']} != {expected}")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")

    records = {
        stage: read_stage(
            stage_root / f"oracle_momstage_kt00000002_s{stage}.bin")
        for stage in (1, 2, 3)
    }
    memory_end = read_record(
        memory_root / "oracle_bt_memory_kt00000001_end.bin")
    oracle_kt3 = read_entry(stage_root / "oracle_step_entry_kt00000003.bin")
    require(oracle_kt3["kt"] == 3, "oracle step-entry record is not kt=3")

    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    require((cfg.outer_integrator, cfg.momentum_time_integrator)
            == ("forward_euler", "rk3_ws"),
            "resolved GYRE path is not forward_euler/rk3_ws")
    require(str(card.recipe.initial_state.eta.data.dtype) == "float64",
            "state dtype is not float64")
    require(str(card.recipe.grid.area.dtype) == "float64",
            "grid dtype is not float64")

    base_model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    state0 = card.recipe.initial_state
    freshwater1, surface1 = _surface_forcings(card, state0, 1)
    state1 = base_model.step(
        state0, dt=card.dt_s, freshwater=freshwater1,
        surface_forcing=surface1)
    freshwater2, surface2 = _surface_forcings(card, state1, 2)
    baseline = base_model.step(
        state1, dt=card.dt_s, freshwater=freshwater2,
        surface_forcing=surface2)

    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True))
    trace = trace_model.step(
        state1, dt=card.dt_s, freshwater=freshwater2,
        surface_forcing=surface2)

    raw_history = list(_raw_history(memory_end))
    if plant == "history":
        planted = np.asarray(raw_history[0]).copy()
        planted[10, 10] += 1.0
        raw_history[0] = jnp.asarray(planted)
    substitute_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            barotropic_raw_history_override=tuple(raw_history)))
    substituted = substitute_model.step(
        state1, dt=card.dt_s, freshwater=freshwater2,
        surface_forcing=surface2)

    masks = expected_masks(card)
    operand_rows = _operand_rows(
        trace, records, masks, plant=plant, rho_0=cfg.rho_0)
    observer_rows = []
    baseline_fields = lego_fields(baseline)
    trace_fields = lego_fields(trace.state_after)
    for field in ("T", "S", "u", "v", "ssh"):
        observer_rows.append(score(
            f"trace_noninterference.{field}", baseline_fields[field],
            trace_fields[field], masks[field]))
    require(all(row["n_unequal"] == 0 for row in observer_rows),
            f"live trace perturbed production output: {observer_rows}")

    baseline_rows = _state_rows(
        "GYRE-zco.kt2.baseline", baseline, oracle_kt3, masks)
    substituted_rows = _state_rows(
        "GYRE-zco.kt2.nemo_raw_history", substituted, oracle_kt3, masks)
    substituted_fields = lego_fields(substituted)
    paired = []
    for before, after in zip(baseline_rows, substituted_rows, strict=True):
        field = before["name"].rsplit(".", 1)[-1]
        active = masks[field]
        before_value = baseline_fields[field][active]
        after_value = substituted_fields[field][active]
        paired.append({
            "field": field,
            "before_n_unequal": before["n_unequal"],
            "after_n_unequal": after["n_unequal"],
            "before_max_abs": before["absolute_max"],
            "after_max_abs": after["absolute_max"],
            "substitution_n_changed": int(np.count_nonzero(
                before_value.view(np.uint64) != after_value.view(np.uint64))),
            "substitution_max_abs": float(np.max(np.abs(
                after_value - before_value))),
        })
    history_confirmed = (
        any(row["after_max_abs"] < row["before_max_abs"] for row in paired)
        and all(row["after_max_abs"] <= row["before_max_abs"] for row in paired)
    )
    if plant == "history":
        require(any(row["after_max_abs"] > row["before_max_abs"] + 1e-6
                    for row in paired), "history plant did not worsen a row")

    ordered = [row for row in operand_rows if row["status"] != "UNMEASURED"]
    first = next((row for row in ordered if row["n_unequal"]), None)
    prediction = (
        first is not None
        and first["stage"] == 1
        and first["operator"] == "hpg"
        and first["field"] == "ssh_Kmm"
        and all(row["n_unequal"] == 0 for row in ordered[:ordered.index(first)])
    )
    if plant == "operand":
        planted = next(row for row in ordered
                       if row["stage"] == 1 and row["operator"] == "hpg"
                       and row["field"] == "u_Kmm")
        require(planted["absolute_max"] > 0.5,
                "operand plant did not move its scored active cell")

    return {
        "format": "nemo-testcase-l2-gyre-round51-live-operands-v1",
        "status": ("CONFIRMED" if prediction and history_confirmed
                   and plant is None else "REFUTED"),
        "prediction": "CONFIRMED" if prediction else "REFUTED",
        "first_unequal_handed_operand": (
            None if first is None else {
                key: first[key] for key in (
                    "stage", "operator", "field", "n_unequal", "absolute_max")
            }),
        "operand_rows": operand_rows,
        "trace_noninterference": observer_rows,
        "barotropic_memory_substitution": {
            "status": "CONFIRMED" if history_confirmed else "REFUTED",
            "rows": paired,
            "only_changed_input": (
                "six raw NEMO u/v/ssh b/bb histories at the production "
                "barotropic loop-entry test seam"),
        },
        "resolved": {
            "outer_integrator": cfg.outer_integrator,
            "momentum_time_integrator": cfg.momentum_time_integrator,
            "tracer_time_integrator": cfg.tracer_time_integrator,
            "momentum_advection": cfg.momentum_advection,
            "dt_s": float(card.dt_s),
            "state_dtype": str(state0.eta.data.dtype),
            "grid_dtype": str(card.recipe.grid.area.dtype),
        },
        "source": {
            "stage_calls": "stprk3.f90:200-215",
            "stage1_order": "stp2d.f90:141-175",
            "stage23_operands": "stprk3_stg.f90:276-345,431-480",
            "history_filter": "dynspg_ts.f90:456-489",
            "history_rotation": "dynspg_ts.f90:749-761",
            "legoesm_history_reconstruction": (
                "barotropic_latlon_cgrid.py:2053-2084"),
        },
        "worktree": stamp,
        "plant": plant,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--memory-root", type=Path, default=MEMORY_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("operand", "history", "stamp"))
    args = parser.parse_args(argv)
    report = run(
        args.stage_root, args.memory_root, args.expect_commit, args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

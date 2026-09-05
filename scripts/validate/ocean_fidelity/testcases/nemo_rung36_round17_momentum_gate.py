#!/usr/bin/env python3
"""Round-17 source-order momentum owner and continuous slab gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round as sr
from legoesm.grids.halo_latlon import set_meridionally_periodic
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_flux_form_barotropic_velocity_update,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_c1d_omip_l3_slab_ocean_card,
)

try:
    from scripts.validate.ocean_fidelity.testcases.nemo_rung36_exchange_gate import (
        GateError,
        _exchange,
        _fwb,
        _qsr,
        _ssm,
    )
    from scripts.validate.ocean_fidelity.testcases.nemo_rung36_ocean_gate import (
        DRAG_FIELDS,
        SPG_STATEMENT_FIELDS,
        STP_FIELDS,
        _records,
        _row,
    )
    from scripts.validate.ocean_fidelity.testcases.nemo_rung36_round16_tracer_gate import (
        _forcing,
    )
except ModuleNotFoundError:
    from nemo_rung36_exchange_gate import GateError, _exchange, _fwb, _qsr, _ssm
    from nemo_rung36_ocean_gate import (
        DRAG_FIELDS,
        SPG_STATEMENT_FIELDS,
        STP_FIELDS,
        _records,
        _row,
    )
    from nemo_rung36_round16_tracer_gate import _forcing

BAR = 1.0e-15
STATE_NAMES = ("u", "v", "temperature", "salinity", "ssh", "e3t")
SAMPLE_STEPS = (2, 3, 5, 6, 10, 100, 1000, 3000, 5000, 8760)
REQUIRED = (
    "oracle_si3_exchange_frames.bin",
    "oracle_rung36_fwb_frames.bin",
    "oracle_rung36_qsr_frames.bin",
    "oracle_rung36_ssm_frames.bin",
    "oracle_rung36_spg_statement_frames.bin",
    "oracle_rung36_drag_frames.bin",
    "oracle_rung36_stp2d_frames.bin",
)


def _require_streams(root: Path) -> None:
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise GateError("round-17 required stream(s) missing: " + ", ".join(missing))


def _statement_rows(root: Path, plant: bool):
    records = _records(
        root / "oracle_rung36_spg_statement_frames.bin",
        b"NEMO_L3SPGS_001 ", "9i", len(SPG_STATEMENT_FIELDS))
    if len(records) != 2 * 946:
        raise GateError("round-17 SPG statement stream must contain kt=1,2")
    if [head[5] for head, _ in records] != list(range(1, 947)) * 2:
        raise GateError("round-17 SPG cycle registry is not 1..946 at each kt")
    rows = []
    values = np.asarray([value for _, value in records])
    one = jnp.ones((values.shape[0],), dtype=jnp.float64)
    q = jnp.asarray(values)
    got_u = nemo_flux_form_barotropic_velocity_update(
        q[:, 1], q[:, 0], q[:, 3], q[:, 5], q[:, 7], q[:, 9],
        sr(one / q[:, 17]), q[:, 11], q[:, 13], q[:, 15], one,
        inverse_depth_after=q[:, 17])
    got_v = nemo_flux_form_barotropic_velocity_update(
        q[:, 2], q[:, 0], q[:, 4], q[:, 6], q[:, 8], q[:, 10],
        sr(one / q[:, 18]), q[:, 12], q[:, 14], q[:, 16], one,
        inverse_depth_after=q[:, 18])
    if plant:
        got_u = got_u.at[946].add(jnp.asarray(1.0e-8))
    rows.append(_row("kt1-2.SSH_SUBSTEP.ua_statement", got_u, values[:, 19]))
    rows.append(_row("kt1-2.SSH_SUBSTEP.va_statement", got_v, values[:, 20]))
    return rows, records


def _new_model(card, hooks: _NEMOWSRK3TestHooks):
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)


def _advance(root: Path, hooks: _NEMOWSRK3TestHooks, nstep: int):
    exchange = _exchange(root / "oracle_si3_exchange_frames.bin")
    fwb = _fwb(root / "oracle_rung36_fwb_frames.bin")
    qsr = _qsr(root / "oracle_rung36_qsr_frames.bin")
    instant, _before, _after, registry = _ssm(
        root / "oracle_rung36_ssm_frames.bin")
    card = build_c1d_omip_l3_slab_ocean_card()
    state = card.recipe.initial_state
    model = _new_model(card, hooks)
    aggregates = {
        name: {"count": 0, "bit_identical": 0, "max_abs": 0.0,
               "max_normalized": 0.0}
        for name in STATE_NAMES
    }
    samples = []
    first = None
    set_meridionally_periodic(card.meridionally_periodic)
    try:
        for index in range(min(nstep, card.n_steps - 1)):
            freshwater, surface = _forcing(exchange, fwb, qsr, index)
            state = model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
            got = np.asarray([
                state.u.data[0, 0, 0], state.v.data[0, 0, 0],
                state.T.data[0, 0, 0], state.S.data[0, 0, 0],
                state.eta.data[0, 0], np.float64(10.0) + state.eta.data[0, 0],
            ], dtype=np.float64)
            step = index + 2
            step_rows = []
            for field, value, oracle in zip(
                    STATE_NAMES, got, instant[index + 1, :6], strict=True):
                row = _row(f"kt{step}.PRE_SSM.{field}", value, oracle)
                stat = aggregates[field]
                stat["count"] += 1
                stat["bit_identical"] += row["bit_identical"]
                stat["max_abs"] = max(stat["max_abs"], row["max_abs"])
                stat["max_normalized"] = max(
                    stat["max_normalized"], row["max_normalized"])
                step_rows.append(row)
                if first is None and row["status"] == "DEBT":
                    first = row
            if step in SAMPLE_STEPS:
                samples.append({"step": step, "rows": step_rows})
    finally:
        set_meridionally_periodic(False)
    for stat in aggregates.values():
        stat["non_bit"] = stat["count"] - stat["bit_identical"]
    return aggregates, samples, first, registry


def _kt3(root: Path, hooks: _NEMOWSRK3TestHooks):
    aggregate, samples, first, _ = _advance(root, hooks, 2)
    kt3 = next(sample for sample in samples if sample["step"] == 3)
    return {row["name"].rsplit(".", 1)[-1]: row for row in kt3["rows"]}, first


def _raw_state_after(root: Path, hooks: _NEMOWSRK3TestHooks, steps: int):
    exchange = _exchange(root / "oracle_si3_exchange_frames.bin")
    fwb = _fwb(root / "oracle_rung36_fwb_frames.bin")
    qsr = _qsr(root / "oracle_rung36_qsr_frames.bin")
    card = build_c1d_omip_l3_slab_ocean_card()
    model = _new_model(card, hooks)
    state = card.recipe.initial_state
    set_meridionally_periodic(card.meridionally_periodic)
    try:
        for index in range(steps):
            freshwater, surface = _forcing(exchange, fwb, qsr, index)
            state = model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
    finally:
        set_meridionally_periodic(False)
    return state


def _owner_and_arms(root: Path, statement_records):
    kt2_first = next(value for head, value in statement_records
                     if head[0] == 2 and head[5] == 1)
    # Observe the superseded operand without bypassing any operator: the
    # private hook substitutes the pre-dynspg workspace only after the full
    # step has executed.  NEMO seeds from puu_b/vv_b(Kmm), dynspg_ts:484-493.
    kmm = _raw_state_after(root, _NEMOWSRK3TestHooks(), 1)
    exchange = _exchange(root / "oracle_si3_exchange_frames.bin")
    fwb = _fwb(root / "oracle_rung36_fwb_frames.bin")
    qsr = _qsr(root / "oracle_rung36_qsr_frames.bin")
    card = build_c1d_omip_l3_slab_ocean_card()
    expose_model = _new_model(card, _NEMOWSRK3TestHooks(
        expose_pre_barotropic_momentum=True))
    set_meridionally_periodic(card.meridionally_periodic)
    try:
        freshwater, surface = _forcing(exchange, fwb, qsr, 1)
        exposed = expose_model.step(
            kmm, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    finally:
        set_meridionally_periodic(False)
    kmm_values = (float(kmm.u.data[0, 0, 0]), float(kmm.v.data[0, 0, 0]))
    exposed_values = (
        float(exposed.u.data[0, 0, 0]), float(exposed.v.data[0, 0, 0]))
    owner_rows = [
        _row("kt2.dynspg.seed_u.Kmm", kmm_values[0], kt2_first[1]),
        _row("kt2.dynspg.seed_v.Kmm", kmm_values[1], kt2_first[2]),
        _row("kt2.dynspg.old_workspace_u", exposed_values[0], kt2_first[1]),
        _row("kt2.dynspg.old_workspace_v", exposed_values[1], kt2_first[2]),
    ]
    # Preserve signed operand evidence, since _row intentionally stores only
    # norms.  The exposed rows compare the kt3 substituted workspace against
    # PRE_SSM and are therefore read directly below in the report.
    old_u = float(exposed_values[0] - kt2_first[1])
    old_v = float(exposed_values[1] - kt2_first[2])
    arms = {}
    for name, hooks in (
        ("seed_weight_0", _NEMOWSRK3TestHooks(barotropic_kmm_seed_weight=0.0)),
        ("seed_weight_0p5", _NEMOWSRK3TestHooks(barotropic_kmm_seed_weight=0.5)),
        ("seed_weight_1", _NEMOWSRK3TestHooks(barotropic_kmm_seed_weight=1.0)),
        ("linear_drag", _NEMOWSRK3TestHooks(linearize_quadratic_bottom_drag=True)),
    ):
        rows, first = _kt3(root, hooks)
        arms[name] = {
            "u": rows["u"], "v": rows["v"],
            "first_over_bar": first,
        }
    return owner_rows, {
        "nemo_seed": {"u": float(kt2_first[1]), "v": float(kt2_first[2])},
        "superseded_workspace_signed_distance": {"u": old_u, "v": old_v},
        "source": "dynspg_ts.F90:484-493",
    }, arms


def _drag_association(root: Path):
    drag_records = _records(
        root / "oracle_rung36_drag_frames.bin", b"NEMO_L3DRG__001 ",
        "6i", len(DRAG_FIELDS))
    drag = np.asarray([value for _, value in drag_records])
    # rCdU_bot/top are signed T-point rates.  NEMO halves their combined
    # neighbour sum once (dynspg_ts.F90:1611-1612).
    bot, bot_e, top, top_e = map(jnp.asarray,
                                (drag[:, 4], drag[:, 5], drag[:, 1], drag[:, 2]))
    literal = sr(jnp.asarray(0.5) * sr(sr(bot_e + bot) + sr(top_e + top)))
    separate = sr(sr(jnp.asarray(0.5) * sr(bot_e + bot))
                  + sr(jnp.asarray(0.5) * sr(top_e + top)))
    different = np.flatnonzero(
        np.asarray(literal).view(np.uint64)
        != np.asarray(separate).view(np.uint64))
    nonzero_top = np.flatnonzero(drag[:, 1] != 0.0)
    stp = _records(
        root / "oracle_rung36_stp2d_frames.bin", b"NEMO_L3STP__001 ",
        "8i", len(STP_FIELDS))
    stp_pre = {head[0]: value for head, value in stp if head[5] == 0}
    rows = []
    if nonzero_top.size:
        index = int(nonzero_top[0])
        kt = index + 1
        rows.append(_row(
            f"kt{kt}.PRE_DYN_SPG_TS.combined_drag_u",
            np.asarray(literal)[index], stp_pre[kt][4]))
    return {
        "source": "dynspg_ts.F90:1611-1612; dynzdf.F90:302,478",
        "first_nonzero_top_drag_step": (
            int(nonzero_top[0] + 1) if nonzero_top.size else None),
        "literal_rows": rows,
        "separate_half_vs_literal_non_bit_steps": int(different.size),
        "first_moved_step": int(different[0] + 1) if different.size else None,
    }


def evaluate(root: Path, *, nstep: int = 8759, plant: bool = False):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    _require_streams(root)
    statement_rows, statements = _statement_rows(root, plant)
    owner_rows, owner, arms = _owner_and_arms(root, statements)
    aggregates, samples, first, registry = _advance(
        root, _NEMOWSRK3TestHooks(), nstep)
    source_first = next(
        (row for row in statement_rows if row["status"] == "DEBT"), None)
    first_over = source_first or first
    return {
        "verdict": "AT_BAR" if first_over is None else "STOP_FIRST_OVER_BAR",
        "bar": BAR,
        "backend": jax.default_backend(),
        "dtype": "float64",
        "transcendentals": "libm",
        "source_statement_rows": statement_rows,
        "owner_rows": owner_rows,
        "owner": owner,
        "one_variable_arms": arms,
        "trajectory_aggregates": aggregates,
        "trajectory_samples": samples,
        "first_over_bar": first_over,
        "drag_association": _drag_association(root),
        "time_level_registry": {
            "spg_statements": (
                "kt=1,2; header (kt,Kbb,Kmm,Krhs,Kaa,jn,nn_e,nvalue,bits); "
                "operand_1/2 are dynspg_ts entry un_e/vn_e at Kmm; "
                "operand_19/20 are post-source ua/va before LBC"),
            "continuous": registry,
        },
        "plant_binding": None if not plant else {
            "target": "kt1-2.SSH_SUBSTEP.ua_statement[kt2,jn1]",
            "red": source_first is not None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=8759)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root, nstep=args.steps, plant=args.plant)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.plant:
        return 1 if report["plant_binding"]["red"] else 2
    return 0 if report["verdict"] == "AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

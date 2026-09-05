#!/usr/bin/env python3
"""Round-16 source-order tracer and first-continuous-boundary gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round as sr
from legoesm.coupler.ocean_forcing import (
    NemoSI3ExchangeConfig,
    nemo_si3_exchange_forcing,
)
from legoesm.grids.halo_latlon import set_meridionally_periodic
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_c1d_omip_l3_slab_ocean_card,
)
from legoesm.ocean.physics.shortwave_penetration import nemo_rgb_one_layer_rhs
from legoesm.ocean.physics.surface_forcing.external import (
    nemo_tra_sbc_rk3_source,
)

try:
    from scripts.validate.ocean_fidelity.testcases.nemo_rung36_exchange_gate import (
        _exchange,
        _fwb,
        _qsr,
        _ssm,
    )
    from scripts.validate.ocean_fidelity.testcases.nemo_rung36_ocean_gate import (
        DRAG_FIELDS,
        _records,
        _row,
    )
except ModuleNotFoundError:
    from nemo_rung36_exchange_gate import _exchange, _fwb, _qsr, _ssm
    from nemo_rung36_ocean_gate import DRAG_FIELDS, _records, _row

BAR = 1.0e-15
TRACER_FIELDS = (
    "T_Kbb", "S_Kbb", "T_Krhs", "S_Krhs", "T_Kaa", "S_Kaa",
    "r3t_Kbb", "r3t_Kmm", "r3t_Kaa", "e3t_Kbb", "e3t_Kmm",
    "e3t_Kaa", "rDt", "qns", "qsr", "emp", "sfx", "tmask",
)
TRAZDF_FIELDS = (
    "tracer_Kbb", "Krhs", "e3t_Kbb", "e3t_Kmm", "e3t_Kaa",
    "p2dt", "zwi", "zws", "zwd", "zwt", "content", "output",
    "tmask", "avt",
)


def _source_rows(root: Path, plant: bool):
    records = _records(
        root / "oracle_rung36_tracer_owner_frames.bin",
        b"NEMO_L3TR16_001 ", "9i", len(TRACER_FIELDS))
    zdf = _records(
        root / "oracle_rung36_trazdf_owner_frames.bin",
        b"NEMO_L3TZ16_001 ", "9i", len(TRAZDF_FIELDS))
    if len(records) != 9 or len(zdf) != 2:
        raise ValueError("round-16 kt=1 tracer registry is incomplete")

    rows = []
    one = jnp.asarray(1.0, dtype=jnp.float64)
    r1rho = sr(one / jnp.asarray(NEMO_CONSTANTS_CONFIG.rho_0))
    r1cp = sr(one / jnp.asarray(NEMO_CONSTANTS_CONFIG.c_sw))
    r1rhocp = sr(r1rho * r1cp)
    by_stage_boundary = {(h[1], h[6]): values for h, values in records}
    for stage in (1, 2, 3):
        values = by_stage_boundary[(stage, 1)]
        got_t, got_s = nemo_tra_sbc_rk3_source(
            tendency_t=jnp.asarray(0.0), tendency_s=jnp.asarray(0.0),
            emp=jnp.asarray(values[15]), qns=jnp.asarray(values[13]),
            salt_flux_pss=jnp.asarray(values[16]),
            layer_thickness=jnp.asarray(values[10]),
            inverse_density=r1rho, inverse_heat_capacity=r1cp,
            temperature=jnp.asarray(values[0]), salinity=jnp.asarray(values[1]),
            stage=stage)
        if stage == 3:
            got_t = nemo_rgb_one_layer_rhs(
                got_t, jnp.asarray(values[14]), jnp.asarray(values[10]),
                r1rhocp)
            expected = by_stage_boundary[(3, 2)]
        else:
            expected = values
        if plant and stage == 1:
            got_t = got_t + jnp.asarray(1.0e-8)
        rows.extend([
            _row(f"kt1.stage{stage}.Krhs_T", got_t, expected[2]),
            _row(f"kt1.stage{stage}.Krhs_S", got_s, expected[3]),
        ])

    for (head, values), name in zip(zdf, ("T", "S"), strict=True):
        base = sr(jnp.asarray(values[2]) * jnp.asarray(values[0]))
        rhs = sr(sr(jnp.asarray(values[5]) * jnp.asarray(values[3]))
                 * jnp.asarray(values[1]))
        content = sr(base + rhs)
        output = sr(content / jnp.asarray(values[9]))
        rows.extend([
            _row(f"kt1.TRA_ZDF.{name}.content", content, values[10]),
            _row(f"kt1.TRA_ZDF.{name}.output", output, values[11]),
        ])
    return rows, {
        "tracer_records": len(records), "trazdf_records": len(zdf),
        "headers": "magic + 9 derived integers; nvalue/STORAGE_SIZE validated",
        "time_levels": (
            "kt=1; stage header registers Kbb/Kmm/Krhs/Kaa.  Stage 1/2 "
            "surface RHS precedes their Kaa content update; stage 3 registers "
            "POST_TRA_SBC, POST_TRA_QSR, POST_TRA_LDF, POST_TRA_ZDF, POST_LBC."),
    }


def _forcing(exchange, fwb, qsr, index):
    def scalar(name, source):
        return jnp.asarray(source[name][index:index + 1].reshape((1, 1)))
    return nemo_si3_exchange_forcing(
        qsr=scalar("qsr", exchange), qns=scalar("qns", fwb),
        emp=scalar("emp", fwb), sfx=scalar("sfx", exchange),
        utau=scalar("utau", exchange), vtau=scalar("vtau", exchange),
        chl=scalar("chl", qsr), rCdU_ice=scalar("rCdU_ice", exchange),
        snwice_fmass=scalar("snwice_fmass", exchange),
        config=NemoSI3ExchangeConfig())


def _trajectory(root: Path, nstep: int, source_scale: float):
    exchange = _exchange(root / "oracle_si3_exchange_frames.bin")
    fwb = _fwb(root / "oracle_rung36_fwb_frames.bin")
    qsr = _qsr(root / "oracle_rung36_qsr_frames.bin")
    instant, _before, _after, registry = _ssm(
        root / "oracle_rung36_ssm_frames.bin")
    card = build_c1d_omip_l3_slab_ocean_card()
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_surface_source_scale=source_scale))
    state = card.recipe.initial_state
    rows = []
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
            names = ("u", "v", "temperature", "salinity", "ssh", "e3t")
            step_rows = [
                _row(f"kt{index + 2}.PRE_SSM.{name}", got[i],
                     instant[index + 1, i])
                for i, name in enumerate(names)]
            rows.extend(step_rows)
            first = next((row for row in step_rows if row["status"] == "DEBT"), None)
            if first is not None:
                break
    finally:
        set_meridionally_periodic(False)
    return rows, first, registry


def _drag_association(root: Path):
    drag = np.asarray([values for _, values in _records(
        root / "oracle_rung36_drag_frames.bin", b"NEMO_L3DRG__001 ",
        "6i", len(DRAG_FIELDS))])
    raw = jnp.asarray(drag[:, 1])
    bottom = jnp.full_like(raw, 5.0e-5)
    old = sr(bottom + sr(-raw))
    literal = sr(-sr(jnp.asarray(0.5) * sr(
        sr(-bottom + -bottom) + sr(raw + raw))))
    changed = np.flatnonzero(
        np.asarray(old).view(np.uint64) != np.asarray(literal).view(np.uint64))
    nonzero = np.flatnonzero(np.asarray(raw) != 0.0)
    return {
        "nemo_statement": "dynspg_ts.F90:1611-1612",
        "kt1_inert": bool(raw[0] == 0.0),
        "first_nonzero_top_drag_step": int(nonzero[0] + 1) if nonzero.size else None,
        "old_vs_literal_non_bit_steps": int(changed.size),
        "first_changed_step": int(changed[0] + 1) if changed.size else None,
    }


def evaluate(root: Path, *, nstep: int, source_scale: float, plant: bool):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    source_rows, schema = _source_rows(root, plant)
    trajectory_rows, first, registry = _trajectory(root, nstep, source_scale)
    source_first = next((row for row in source_rows if row["status"] == "DEBT"), None)
    verdict = "AT_BAR" if source_first is None and first is None else "STOP_FIRST_OVER_BAR"
    if source_first is not None:
        owner_label = "PLANTED_SOURCE_ROW"
    elif first is None:
        owner_label = None
    elif source_scale != 1.0:
        owner_label = "TRACER_SURFACE_SOURCE_PRIVATE_ARM"
    else:
        owner_label = "STEP2_MOMENTUM_END_TO_PRE_SSM"
    return {
        "verdict": verdict, "bar": BAR, "backend": jax.default_backend(),
        "dtype": "float64", "source_rows": source_rows,
        "trajectory_rows": trajectory_rows, "first_over_bar": source_first or first,
        "owner_label": owner_label,
        "source_scale": source_scale, "plant": plant,
        "plant_binding": None if not plant else {
            "target": "kt1.stage1.Krhs_T",
            "red": source_rows[0]["status"] == "DEBT"},
        "schema_registry": schema, "ssm_registry": registry,
        "drag_association": _drag_association(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--steps", type=int, default=8760)
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.root, nstep=args.steps, source_scale=args.source_scale,
        plant=args.plant)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.plant:
        return 1 if report["plant_binding"]["red"] else 2
    return 0 if report["verdict"] == "AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

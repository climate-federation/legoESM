#!/usr/bin/env python3
"""Name the producer of the ICE_RHEO step-9 carried-state seed.

The probe starts from the oracle step-8 entry state and step-7 Prather
restart, advances the production card through completed step 8, and scores the
result against the oracle step-9 entry plus step-8 restart.  One-variable arms
replace only the surface-stress association or the five rheology outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice import ridging as ridging_module
from legoesm.ice.dynamics import SI3CGridAEVPState, si3_cgrid_aevp_solver
from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as recipe
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    ICE_RHEO_TRACERS,
    _forcing_for_state,
    build_ice_rheo_card,
    step_ice_rheo_card,
)
from legoesm.ice.transport import SI3_PRATHER_MOMENT_NAMES
from legoesm.ocean.fidelity.provenance import worktree_stamp

HERE = Path(__file__).resolve().parent
ORACLE_GATE = HERE / "nemo_si3_oracle_gate.py"
RUNG34_GATE = HERE / "nemo_si3_phase2_rung34_gate.py"
TRAJECTORY_GATE = HERE / "nemo_si3_phase2_rung34_trajectory_gate.py"
ROUND13_PROBE = HERE / "nemo_si3_phase2_round13_active_aevp_probe.py"
REPLAY = HERE / "nemo_si3_phase2_rung33_replay.py"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
RESTART_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/round8/oracle_active_restarts"
)
ROUND13_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/round13/run_with_init"
)

ENTRY_STEP = 8
TARGET_STEP = 9
ENTRY_RESTART_STEP = 7
HALO = 2
PLANT_OFFSET = 10
PLANT_MAGNITUDE = 1.0e-10
FP64_BITS = 64

ENTRY_FIELD_ORDER = (
    "u_ice",
    "v_ice",
    "stress1_i",
    "stress2_i",
    "stress12_i",
    "a_i",
    "v_i",
    "v_s",
    "v_ip",
    "v_il",
    "oa_i",
    "a_ip",
    *(f"e_s_l{level:02d}" for level in range(1, 6)),
    *(f"e_i_l{level:02d}" for level in range(1, 11)),
    *(f"szv_i_l{level:02d}" for level in range(1, 11)),
    "sv_i",
    "t_su",
    "open_water_area",
)


class SeedProbeError(RuntimeError):
    """Fail-closed Round-14 producer-probe error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SeedProbeError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load("nemo_si3_oracle_gate_round14", ORACLE_GATE)
rung34 = _load("nemo_si3_rung34_gate_round14", RUNG34_GATE)
trajectory = _load("nemo_si3_rung34_trajectory_round14", TRAJECTORY_GATE)
round13 = _load("nemo_si3_round13_probe_round14", ROUND13_PROBE)
replay = _load("nemo_si3_rung33_replay_round14", REPLAY)


def _physical(value: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.asarray(value)[HALO:-HALO, HALO:-HALO])


def _score(name: str, oracle, candidate) -> dict[str, object]:
    return round13._score(name, np.asarray(oracle), np.asarray(candidate))


def _nonzero(row: dict[str, object]) -> bool:
    return str(row["bitwise_nonzero_over_n"]).split()[0] != "0"


def _assert_plant(clean: dict[str, object], planted: dict[str, object]) -> bool:
    require(clean["name"] == planted["name"], "plant changed row identity")
    require(clean["status"] == "BIT-EXACT", "plant baseline is not exact")
    require(planted["status"] == "DEBT", "plant row did not become DEBT")
    require(not _nonzero(clean) and _nonzero(planted), "plant numerator did not move")
    return True


def _target_field(target: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "open_water_area":
        return 1.0 - np.asarray(rung34._oracle_field(target, "a_i"))
    return np.asarray(rung34._oracle_field(target, name))


def _candidate_fields(card, state) -> dict[str, np.ndarray]:
    fields = rung34._candidate_fields(card, state)
    fields["open_water_area"] = np.asarray(state.open_water_area)
    return fields


def _entry_rows(card, state, target: dict[str, np.ndarray]) -> list[dict[str, object]]:
    fields = _candidate_fields(card, state)
    require(set(fields) == set(ENTRY_FIELD_ORDER), "entry field registry drift")
    return [
        _score(
            f"entry.step9.{name}",
            _physical(_target_field(target, name)),
            _physical(fields[name]),
        )
        for name in ENTRY_FIELD_ORDER
    ]


def _moment_rows(state, restart_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with netCDF4.Dataset(restart_path) as dataset:
        for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
            for tracer_index, tracer in enumerate(ICE_RHEO_TRACERS):
                name = trajectory._moment_restart_name(moment, tracer)
                rows.append(
                    _score(
                        "entry_moment.step9." + name,
                        trajectory._restart_xy(dataset, name),
                        _physical(np.asarray(state.moments[moment_index])[..., tracer_index]),
                    )
                )
    require(len(rows) == 5 * len(ICE_RHEO_TRACERS), "moment registry drift")
    return rows


def _source_written_air_stress(state, ice_step_index):
    """Private one-variable arm for usrdef_sbc.F90:124-143."""

    r = nemo_source_round
    indices = jnp.arange(
        recipe._ICE_RHEO_HALO_WIDTH + 1,
        recipe._ICE_RHEO_HALO_WIDTH + recipe._ICE_RHEO_GRID_SIZE + 1,
        dtype=jnp.float64,
    )
    x = r(
        recipe._ICE_RHEO_DOMAIN_KM
        - r(r(recipe._ICE_RHEO_TWO * indices) * recipe._ICE_RHEO_RESOLUTION_KM)
    )
    y = x
    root = r(jnp.sqrt(r(recipe._ICE_RHEO_DOMAIN_KM * recipe._ICE_RHEO_KM_TO_M)))
    scale = r(recipe._ICE_RHEO_WIND_MAX_M_S / root)
    spinup = jnp.minimum(
        r(r(ice_step_index * recipe._ICE_RHEO_DT_S) / recipe._ICE_RHEO_SPINUP_S),
        recipe._ICE_RHEO_ONE,
    )
    wind_u_physical = jnp.broadcast_to(
        r(r(scale * x[:, None]) * spinup),
        (recipe._ICE_RHEO_GRID_SIZE, recipe._ICE_RHEO_GRID_SIZE),
    )
    wind_v_physical = jnp.broadcast_to(
        r(r(r(scale * y[None, :]) * recipe._ICE_RHEO_WIND_RATIO) * spinup),
        (recipe._ICE_RHEO_GRID_SIZE, recipe._ICE_RHEO_GRID_SIZE),
    )
    wind_u = recipe._periodic_halo(wind_u_physical)
    wind_v = recipe._periodic_halo(wind_v_physical)
    relative_u = r(
        wind_u
        - r(
            recipe._ICE_RHEO_RELATIVE_WIND
            * r(
                recipe._ICE_RHEO_HALF
                * r(jnp.roll(state.dynamics.u_ice_u, 1, axis=0) + state.dynamics.u_ice_u)
            )
        )
    )
    relative_v = r(
        wind_v
        - r(
            recipe._ICE_RHEO_RELATIVE_WIND
            * r(
                recipe._ICE_RHEO_HALF
                * r(jnp.roll(state.dynamics.v_ice_v, 1, axis=1) + state.dynamics.v_ice_v)
            )
        )
    )
    magnitude = r(jnp.sqrt(r(r(relative_u * relative_u) + r(relative_v * relative_v))))
    stress_scale = r(recipe._ICE_RHEO_AIR_DENSITY_KG_M3 * recipe._ICE_RHEO_AIR_DRAG)
    stress_u = r(r(stress_scale * magnitude) * relative_u)
    stress_v = r(r(stress_scale * magnitude) * relative_v)
    return recipe._replace_periodic_halo(stress_u), recipe._replace_periodic_halo(stress_v)


def _compiled_step(card, state):
    result = jax.jit(
        lambda current: step_ice_rheo_card(card, current, completed_steps=ENTRY_RESTART_STEP)
    )(state)
    jax.block_until_ready(result)
    return result


def _source_stress_arm(card, state):
    saved = recipe._ice_rheo_air_stress_impl
    try:
        recipe._ice_rheo_air_stress_impl = _source_written_air_stress
        return _compiled_step(card, state)
    finally:
        recipe._ice_rheo_air_stress_impl = saved


def _exact_dynamics_arm(
    card,
    state,
    target: dict[str, np.ndarray],
    *,
    ablate_ledger_rounding: bool = False,
):
    exact = SI3CGridAEVPState(
        *(jnp.asarray(target[name], dtype=jnp.float64) for name in (
            "u_ice",
            "v_ice",
            "stress1_i",
            "stress2_i",
            "stress12_i",
        ))
    )
    saved = recipe.si3_cgrid_aevp_solver
    saved_round = ridging_module.nemo_source_round
    try:
        recipe.si3_cgrid_aevp_solver = lambda *_args, **_kwargs: exact
        if ablate_ledger_rounding:
            ridging_module.nemo_source_round = lambda value: value
        return _compiled_step(card, state)
    finally:
        recipe.si3_cgrid_aevp_solver = saved
        ridging_module.nemo_source_round = saved_round


def _dynamics_rows(card, state, target: dict[str, np.ndarray]) -> list[dict[str, object]]:
    forcing = _forcing_for_state(card, state, ENTRY_STEP)
    result = jax.jit(
        lambda dynamics, applied: si3_cgrid_aevp_solver(
            dynamics, applied, card.metrics, card.dynamics_config
        )
    )(state.dynamics, forcing)
    jax.block_until_ready(result)
    names = ("u_ice", "v_ice", "stress1_i", "stress2_i", "stress12_i")
    return [
        _score(f"stage8.rheology.{name}", _physical(target[name]), _physical(value))
        for name, value in zip(names, result, strict=True)
    ]


def _setup_rows(card, state, setup: dict[str, np.ndarray], prefix: str):
    forcing = _forcing_for_state(card, state, TARGET_STEP)
    q = replay._setup(card, forcing)
    candidate = {
        **q,
        "u": np.asarray(state.dynamics.u_ice_u),
        "v": np.asarray(state.dynamics.v_ice_v),
        "stress1": np.asarray(state.dynamics.stress1_t),
        "stress2": np.asarray(state.dynamics.stress2_t),
        "stress12": np.asarray(state.dynamics.stress12_f),
        "u_b": np.asarray(state.dynamics.u_ice_u),
        "v_b": np.asarray(state.dynamics.v_ice_v),
        "base_u": np.zeros_like(np.asarray(state.dynamics.u_ice_u)),
        "base_v": np.zeros_like(np.asarray(state.dynamics.v_ice_v)),
    }
    require(set(round13.SETUP_REGISTRY) <= set(candidate), "setup candidate registry gap")
    return [
        _score(f"{prefix}.{name}", _physical(setup[name]), _physical(candidate[name]))
        for name in round13.SETUP_REGISTRY
    ]


def _arm_summary(
    baseline_rows: list[dict[str, object]], arm_rows: list[dict[str, object]]
) -> dict[str, object]:
    baseline = {str(row["name"]).split(".")[-1]: row for row in baseline_rows}
    arm = {str(row["name"]).split(".")[-1]: row for row in arm_rows}
    shared = sorted(set(baseline) & set(arm))
    moved = [
        name
        for name in shared
        if float(arm[name]["max_abs"]) < float(baseline[name]["max_abs"])
    ]
    exact = [name for name in shared if arm[name]["status"] == "BIT-EXACT"]
    return {
        "shared_rows": len(shared),
        "moved_toward_oracle": moved,
        "bit_exact_rows": exact,
        "all_shared_exact": len(exact) == len(shared),
    }


def run_probe(
    root: Path = ROOT,
    restart_root: Path = RESTART_ROOT,
    round13_root: Path = ROUND13_ROOT,
    *,
    plant: bool = False,
) -> tuple[dict[str, object], int]:
    require(jax.default_backend() == "cpu", "Round-14 seed probe requires CPU")
    entry_path = root / f"oracle_ice_step_entry_kt{ENTRY_STEP:08d}.bin"
    target_path = root / f"oracle_ice_step_entry_kt{TARGET_STEP:08d}.bin"
    entry_restart = restart_root / f"ICE_RHEO_OMIP_L3_{ENTRY_RESTART_STEP:08d}_restart_ice.nc"
    target_restart = restart_root / f"ICE_RHEO_OMIP_L3_{ENTRY_STEP:08d}_restart_ice.nc"
    setup_path = round13_root / "round13_aevp_setup.bin"
    for path in (entry_path, target_path, entry_restart, target_restart, setup_path):
        require(path.is_file(), f"missing Round-14 input: {path}")
    entry_header, entry = oracle_gate.read_frame(entry_path)
    target_header, target = oracle_gate.read_frame(target_path)
    require(entry_header["kt"] == ENTRY_STEP, "entry clock mismatch")
    require(target_header["kt"] == TARGET_STEP, "target clock mismatch")
    require(entry_header["storage_bits"] == target_header["storage_bits"] == FP64_BITS, "not fp64")
    with netCDF4.Dataset(root / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in rung34.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        from legoesm import constants

        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = build_ice_rheo_card(entry, mesh, ocean_temperature_k)
    state = card.initial_state._replace(moments=trajectory._restart_moments(entry_restart, card))
    candidate = _compiled_step(card, state)

    entry_rows = _entry_rows(card, candidate, target)
    moment_rows = _moment_rows(candidate, target_restart)
    first_nonexact = next((row for row in entry_rows if _nonzero(row)), None)
    require(first_nonexact is not None, "Round-13 seed disappeared")

    baseline_dyn_rows = _dynamics_rows(card, state, target)
    source_state = _source_stress_arm(card, state)
    source_rows = _entry_rows(card, source_state, target)
    exact_dynamics_state = _exact_dynamics_arm(card, state, target)
    exact_dynamics_rows = _entry_rows(card, exact_dynamics_state, target)
    exact_dynamics_moments = _moment_rows(exact_dynamics_state, target_restart)

    setup_header, setup = round13._read_setup(setup_path)
    require(setup_header["kt"] == TARGET_STEP, "setup clock mismatch")
    exact_target_card = build_ice_rheo_card(target, mesh, ocean_temperature_k)
    exact_setup_rows = _setup_rows(
        exact_target_card, exact_target_card.initial_state, setup, "setup.exact_oracle_entry"
    )
    candidate_setup_rows = _setup_rows(card, candidate, setup, "setup.candidate_entry")

    exact_dynamics_unrounded_state = _exact_dynamics_arm(
        card, state, target, ablate_ledger_rounding=True
    )
    exact_dynamics_unrounded_rows = _entry_rows(
        card, exact_dynamics_unrounded_state, target
    )

    mass_rows = []
    for name in ("snwice_mass", "snwice_mass_b"):
        value = _physical(target[name])
        mass_rows.append(
            {
                "name": f"missing_candidate_carry.{name}",
                "status": "UNMEASURED",
                "reason": (
                    "NEMO carries this field through iceupdate.F90:190-193; "
                    "the current legoESM card has no corresponding state leaf"
                ),
                "oracle_nonzero_over_n": f"{np.count_nonzero(value)} / {value.size}",
                "oracle_max_abs": float(np.max(np.abs(value))),
            }
        )

    source_summary = _arm_summary(entry_rows, source_rows)
    exact_dynamics_summary = _arm_summary(entry_rows, exact_dynamics_rows)
    exact_dynamics_all = all(
        row["status"] == "BIT-EXACT"
        for row in exact_dynamics_rows + exact_dynamics_moments
        if not str(row["name"]).endswith(("oa_i", "a_ip", "v_ip", "v_il"))
    )
    unrounded_summary = _arm_summary(
        exact_dynamics_rows, exact_dynamics_unrounded_rows
    )
    ledger_fields = (
        "v_i",
        "v_s",
        *(f"e_s_l{level:02d}" for level in range(1, 6)),
        *(f"e_i_l{level:02d}" for level in range(1, 11)),
        *(f"szv_i_l{level:02d}" for level in range(1, 11)),
    )
    rounded_by_name = {
        str(row["name"]).split(".")[-1]: row for row in exact_dynamics_rows
    }
    unrounded_by_name = {
        str(row["name"]).split(".")[-1]: row
        for row in exact_dynamics_unrounded_rows
    }
    ledger_confirmed = (
        all(rounded_by_name[name]["status"] == "BIT-EXACT" for name in ledger_fields)
        and any(unrounded_by_name[name]["status"] != "BIT-EXACT" for name in ledger_fields)
        and rounded_by_name["a_i"]["status"] == "BIT-EXACT"
        and not any(_nonzero(row) for row in exact_dynamics_moments)
    )
    if ledger_confirmed:
        producer = (
            "icedyn_rdgrft.F90:715-720,734-741,779-792,874-890 "
            "jpl=1 donor/survivor/receiver inventory statement association"
        )
        classification = "PRODUCER-NAMED"
    else:
        producer = "first surviving post-Prather ridging ledger operand"
        classification = "STATEMENT-OWNER-UNMEASURED"

    clean_plant_row = next(row for row in entry_rows if row["name"] == "entry.step9.a_i")
    plant_evidence = None
    if plant:
        planted = _candidate_fields(card, candidate)["a_i"].copy()
        planted[HALO + PLANT_OFFSET, HALO + PLANT_OFFSET] += PLANT_MAGNITUDE
        planted_row = _score(
            "entry.step9.a_i",
            _physical(_target_field(target, "a_i")),
            _physical(planted),
        )
        plant_evidence = {
            "clean": clean_plant_row,
            "planted": planted_row,
            "row_transition_bound": _assert_plant(clean_plant_row, planted_row),
        }

    report = {
        "format": "nemo-si3-phase2-round14-seed-producer-v1",
        "worktree": worktree_stamp(),
        "status": "DEBT" if plant else classification,
        "exit_code": 1 if plant or classification != "PRODUCER-NAMED" else 0,
        "execution": "CPU/fp64 production JIT; scalar-libm card",
        "clock": {
            "oracle_entry": ENTRY_STEP,
            "completed_step": ENTRY_STEP,
            "oracle_target_entry": TARGET_STEP,
        },
        "first_non_bit_exact_entry_field": first_nonexact,
        "entry_rows": entry_rows,
        "moment_rows": moment_rows,
        "stage8_rheology_rows": baseline_dyn_rows,
        "surface_stress_arm_rows": source_rows,
        "surface_stress_arm": source_summary,
        "surface_stress_prediction": "REFUTED",
        "exact_dynamics_arm_rows": exact_dynamics_rows,
        "exact_dynamics_arm_moment_rows": exact_dynamics_moments,
        "exact_dynamics_arm": {
            **exact_dynamics_summary,
            "all_entry_and_moment_rows_exact": exact_dynamics_all,
        },
        "private_unrounded_ridging_ledger_arm_rows": exact_dynamics_unrounded_rows,
        "private_unrounded_ridging_ledger_arm": unrounded_summary,
        "exact_oracle_entry_setup_rows": exact_setup_rows,
        "candidate_entry_setup_rows": candidate_setup_rows,
        "missing_candidate_carry_rows": mass_rows,
        "producer": producer,
        "predicates": {
            "surface_stress_prediction_confirmed": False,
            "ridging_inventory_ledger_prediction_confirmed": ledger_confirmed,
            "exact_dynamics_closes_downstream": exact_dynamics_all,
            "all_exact_oracle_setup_rows_exact": all(
                row["status"] == "BIT-EXACT" for row in exact_setup_rows
            ),
            "snwice_candidate_carry_present": False,
        },
        "plant": {
            "enabled": plant,
            "row": "entry.step9.a_i",
            "evidence": plant_evidence,
        },
    }
    return report, cast(int, report["exit_code"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--restart-root", type=Path, default=RESTART_ROOT)
    parser.add_argument("--round13-root", type=Path, default=ROUND13_ROOT)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report, code = run_probe(
        args.root, args.restart_root, args.round13_root, plant=args.plant
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.artifact:
        args.artifact.write_text(payload + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Written-order step-8 Prather moment discrimination for SI3 rung 3.4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import legoesm.ice.transport as ice_transport
from legoesm.core.precision import PrecisionPolicy, get_policy
from legoesm.ice.dynamics import si3_cgrid_aevp_solver
from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as recipe
from legoesm.ice.transport import (
    SI3_PRATHER_MOMENT_NAMES,
    advect_si3_prather_2d,
    si3_prather_pack_intensives,
)

from legoesm import constants

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
RESTART_ROOT = ROOT.parent / "round8" / "oracle_active_restarts"
HERE = Path(__file__).resolve().parent
TRAJECTORY_GATE = HERE / "nemo_si3_phase2_rung34_trajectory_gate.py"
ADV2D_REPLAY = HERE / "nemo_si3_phase2_adv2d_replay.py"
POINTWISE_BAR = 1.0e-15
ULP_LIMIT = 2
ENTRY_FRAME = 8
TARGET_FRAME = 9
ENTRY_RESTART = 7
TARGET_RESTART = 8
HALO = 2
PLANT_VELOCITY_M_S = 1.0e-10
PLANT_BINDING_ROW = "plant_delta.syye_l01"
_HASH_BLOCK_BYTES = 1024 * 1024
_MOMENT_ROW_COUNT = len(SI3_PRATHER_MOMENT_NAMES) * len(recipe.ICE_RHEO_TRACERS)


class MomentReplayError(RuntimeError):
    """Fail-closed moment replay error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MomentReplayError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trajectory = _load("nemo_si3_rung34_trajectory_moment_replay", TRAJECTORY_GATE)
adv2d_replay = _load("nemo_si3_adv2d_written_order_for_rung34", ADV2D_REPLAY)
gate = trajectory.gate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, adv2d_replay._ulp_distance(left, right))


def _run_written_order_arm(card, state, u_ice: np.ndarray, v_ice: np.ndarray):
    """Even-step y-then-x program at icedyn_adv_pra.F90:303-350."""

    x_step, y_step = adv2d_replay._numpy_prather_functions()
    # icedyn_adv_pra.F90:218-245 forms the extensive z0* work arrays only
    # after rheology, from the intensive category fields carried by NEMO.
    contents = np.asarray(state.contents).copy() * np.asarray(card.metrics.area_t)[..., None]
    moments = tuple(np.asarray(value).copy() for value in state.moments)
    area = np.asarray(card.metrics.area_t)
    wet = np.asarray(card.forcing_template.tmask_t, dtype=bool)
    after_y, moments_y, swept_area = y_step(
        contents,
        moments,
        v_ice * np.asarray(card.metrics.e1v),
        area,
        wet,
        card.dt_s,
        initial_area=area,
        first_sweep=True,
        halo_width=card.halo_width,
        subcycle_index=1,
        subcycles=card.subcycles,
    )
    after_x, moments_x, _ = x_step(
        after_y,
        moments_y,
        u_ice * np.asarray(card.metrics.e2u),
        area,
        wet,
        card.dt_s,
        initial_area=swept_area,
        first_sweep=False,
        halo_width=card.halo_width,
        subcycle_index=1,
        subcycles=card.subcycles,
    )
    return {
        "after_y": (after_y, moments_y),
        "after_x": (after_x, moments_x),
    }


def _make_jit_stepper(card):
    area = card.metrics.area_t
    wet = card.forcing_template.tmask_t.astype(bool)

    def advance(packed, moments, u_ice, v_ice):
        return advect_si3_prather_2d(
            packed,
            moments,
            u_ice,
            v_ice,
            area,
            wet,
            card.dt_s,
            dx=card.metrics.e1t,
            dy=card.metrics.e2t,
            ice_step_index=TARGET_RESTART,
            nn_fsbc=card.nn_fsbc,
            halo_width=card.halo_width,
            ice_volume_index=recipe.ICE_RHEO_TRACERS.index("v_i"),
            concentration_index=recipe.ICE_RHEO_TRACERS.index("a_i"),
            subcycles=card.subcycles,
        )

    return jax.jit(advance)


def _run_jit_arm(card, state, u_ice: np.ndarray, v_ice: np.ndarray, stepper):
    """Production-JIT even-step transport with prescribed U/V inputs."""

    contents = si3_prather_pack_intensives(state.contents, card.metrics.area_t)
    after_x, moments_x, _ = stepper(
        contents,
        state.moments,
        jnp.asarray(u_ice),
        jnp.asarray(v_ice),
    )
    jax.block_until_ready(moments_x)
    return {
        "after_x": (
            np.asarray(after_x),
            tuple(np.asarray(value) for value in moments_x),
        )
    }


def _moment_row(name: str, oracle: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    require(oracle.shape == candidate.shape, f"{name}: shape mismatch")
    require(oracle.dtype == candidate.dtype == np.float64, f"{name}: fp64 required")
    difference = np.abs(candidate - oracle)
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    maximum = float(difference[index])
    oracle_maximum = float(np.max(np.abs(oracle)))
    normalized = maximum / max(oracle_maximum, 1.0)
    relative = maximum / oracle_maximum if oracle_maximum > 0.0 else None
    ulp = _ulp_distance(candidate, oracle)
    nonzero = int(np.count_nonzero(candidate.view(np.uint64) != oracle.view(np.uint64)))
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= POINTWISE_BAR else "DEBT",
        "max_abs": maximum,
        "normalized_max_abs": normalized,
        "relative_max_abs": relative,
        "max_ulp": int(np.max(ulp)),
        "bitwise_nonzero_over_n": f"{nonzero} / {oracle.size}",
        "bitwise_nonzero_count": nonzero,
        "max_abs_index_xy": [int(value) for value in index],
    }


def _score_arm(stages, target_restart: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    moments = stages["after_x"][1]
    with netCDF4.Dataset(target_restart) as dataset:
        for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
            for tracer_index, tracer_name in enumerate(recipe.ICE_RHEO_TRACERS):
                restart_name = trajectory._moment_restart_name(moment, tracer_name)
                oracle = trajectory._restart_xy(dataset, restart_name)
                candidate = moments[moment_index][HALO:-HALO, HALO:-HALO, tracer_index]
                rows.append(_moment_row(restart_name, oracle, candidate))
    require(len(rows) == _MOMENT_ROW_COUNT, "step-8 replay lost a moment row")
    return rows


def _score_plant_delta(clean_stages, planted_stages) -> list[dict[str, object]]:
    """Score the plant against its clean JIT twin, independent of oracle debt."""

    rows: list[dict[str, object]] = []
    clean = clean_stages["after_x"][1]
    planted = planted_stages["after_x"][1]
    for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
        for tracer_index, tracer_name in enumerate(recipe.ICE_RHEO_TRACERS):
            name = "plant_delta." + trajectory._moment_restart_name(moment, tracer_name)
            clean_value = clean[moment_index][HALO:-HALO, HALO:-HALO, tracer_index]
            planted_value = planted[moment_index][HALO:-HALO, HALO:-HALO, tracer_index]
            rows.append(_moment_row(name, clean_value, planted_value))
    require(len(rows) == _MOMENT_ROW_COUNT, "JIT plant delta lost a moment row")
    return rows


def _first_arm_difference(baseline, oracle_input) -> dict[str, object] | None:
    # NEMO's argument/write order inside adv_y/adv_x is sx,sxx,sy,syy,sxy
    # (icedyn_adv_pra.F90:722-738,761-791,933-939).
    family_order = (("sx", 0), ("sxx", 2), ("sy", 1), ("syy", 3), ("sxy", 4))
    for stage_name in ("after_y", "after_x"):
        baseline_moments = baseline[stage_name][1]
        oracle_moments = oracle_input[stage_name][1]
        for tracer_index, tracer_name in enumerate(recipe.ICE_RHEO_TRACERS):
            for family, moment_index in family_order:
                left = baseline_moments[moment_index][
                    HALO:-HALO, HALO:-HALO, tracer_index
                ]
                right = oracle_moments[moment_index][
                    HALO:-HALO, HALO:-HALO, tracer_index
                ]
                ulp = _ulp_distance(left, right)
                if np.any(ulp > ULP_LIMIT):
                    index = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
                    return {
                        "stage": stage_name,
                        "relative_to_limiter": (
                            "after limiter; velocity first enters at "
                            "icedyn_adv_pra.F90:793"
                        ),
                        "tracer": tracer_name,
                        "moment": family,
                        "index_xy": [int(value) for value in index],
                        "max_ulp": int(ulp[index]),
                        "baseline": float(left[index]),
                        "oracle_velocity_arm": float(right[index]),
                    }
    return None


def _arm_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    debts = [row for row in rows if row["status"] == "DEBT"]
    owner = max(rows, key=lambda row: float(row["normalized_max_abs"]))
    over_two_ulp = [row for row in rows if int(row["max_ulp"]) > ULP_LIMIT]
    non_bit_exact = [row for row in rows if int(row["bitwise_nonzero_count"]) > 0]
    return {
        "status": "AT-BAR" if not debts else "DEBT",
        "debt_count": len(debts),
        "over_two_ulp_count": len(over_two_ulp),
        "byte_exact_count": len(rows) - len(non_bit_exact),
        "non_bit_exact_count": len(non_bit_exact),
        "owner": owner,
        "rows": rows,
    }


def run_replay(
    root: Path,
    restart_root: Path,
    *,
    jit_only: bool = False,
) -> dict[str, object]:
    require(jax.default_backend() == "cpu", "CPU backend required")
    _, entry = gate.oracle_gate.read_frame(
        root / f"oracle_ice_step_entry_kt{ENTRY_FRAME:08d}.bin"
    )
    _, target = gate.oracle_gate.read_frame(
        root / f"oracle_ice_step_entry_kt{TARGET_FRAME:08d}.bin"
    )
    with netCDF4.Dataset(root / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in gate.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = recipe.build_ice_rheo_card(entry, mesh, ocean_temperature_k)
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "fp64/scalar-libm policy required",
    )
    entry_restart = restart_root / (
        f"ICE_RHEO_OMIP_L3_{ENTRY_RESTART:08d}_restart_ice.nc"
    )
    target_restart = restart_root / (
        f"ICE_RHEO_OMIP_L3_{TARGET_RESTART:08d}_restart_ice.nc"
    )
    state = card.initial_state._replace(
        moments=trajectory._restart_moments(entry_restart, card)
    )
    oracle_u = np.asarray(target["u_ice"])
    oracle_v = np.asarray(target["v_ice"])
    planted_v = oracle_v.copy()
    planted_v[HALO + 10, HALO + 10] += PLANT_VELOCITY_M_S
    jit_stepper = _make_jit_stepper(card)
    jit_clean_stages = _run_jit_arm(card, state, oracle_u, oracle_v, jit_stepper)
    jit_planted_stages = _run_jit_arm(card, state, oracle_u, planted_v, jit_stepper)
    jit_oracle_uv = _arm_summary(_score_arm(jit_clean_stages, target_restart))
    jit_plant_delta = _arm_summary(
        _score_plant_delta(jit_clean_stages, jit_planted_stages)
    )
    plant_binding_row = next(
        row
        for row in jit_plant_delta["rows"]
        if row["name"] == PLANT_BINDING_ROW
    )
    require(
        plant_binding_row["status"] == "DEBT"
        and int(plant_binding_row["max_ulp"]) > ULP_LIMIT,
        f"JIT velocity plant did not move {PLANT_BINDING_ROW} over two ULP",
    )
    # Private one-variable ablation: trace the same production JIT program with
    # the source-statement guard replaced by identity.  This is evidence only;
    # it is not a card selector or public transport option.
    source_round = ice_transport.nemo_source_round
    try:
        ice_transport.nemo_source_round = lambda value: value
        unrounded_stepper = _make_jit_stepper(card)
        jit_unrounded_stages = _run_jit_arm(
            card, state, oracle_u, oracle_v, unrounded_stepper
        )
    finally:
        ice_transport.nemo_source_round = source_round
    jit_unrounded = _arm_summary(_score_arm(jit_unrounded_stages, target_restart))
    if jit_only:
        return {
            "gate": "nemo-si3-phase2-rung34-step8-moment-replay-jit-v1",
            "status": jit_oracle_uv["status"],
            "exit_code": 0 if jit_oracle_uv["non_bit_exact_count"] == 0 else 1,
            "backend": jax.default_backend(),
            "precision_policy": str(get_policy()),
            "execution_path": "production JIT (CPU, fp64, scalar-libm)",
            "jit_oracle_uv": jit_oracle_uv,
            "private_unrounded_ablation": jit_unrounded,
            "plant": {
                "velocity_delta_m_s": PLANT_VELOCITY_M_S,
                "jit_over_two_ulp_count": jit_plant_delta["over_two_ulp_count"],
                "delta_owner": jit_plant_delta["owner"],
                "binding_row": plant_binding_row,
                "exit_code": 1,
            },
        }

    forcing = recipe._forcing_for_state(card, state, TARGET_RESTART)
    dynamics = si3_cgrid_aevp_solver(
        state.dynamics, forcing, card.metrics, card.dynamics_config
    )
    lego_u = np.asarray(dynamics.u_ice_u)
    lego_v = np.asarray(dynamics.v_ice_v)

    velocity_rows = {
        "u_ice": gate._score(
            "u_ice",
            oracle_u[HALO:-HALO, HALO:-HALO],
            lego_u[HALO:-HALO, HALO:-HALO],
        ),
        "v_ice": gate._score(
            "v_ice",
            oracle_v[HALO:-HALO, HALO:-HALO],
            lego_v[HALO:-HALO, HALO:-HALO],
        ),
    }
    arms = {
        "baseline": _run_written_order_arm(card, state, lego_u, lego_v),
        "oracle_v": _run_written_order_arm(card, state, lego_u, oracle_v),
        "oracle_u": _run_written_order_arm(card, state, oracle_u, lego_v),
        "oracle_uv": _run_written_order_arm(card, state, oracle_u, oracle_v),
    }
    summaries = {
        name: _arm_summary(_score_arm(stages, target_restart))
        for name, stages in arms.items()
    }
    first_difference = _first_arm_difference(arms["baseline"], arms["oracle_uv"])

    planted = _run_written_order_arm(card, state, oracle_u, planted_v)
    planted_rows = _score_arm(planted, target_restart)
    clean_rows = summaries["oracle_uv"]["rows"]
    changed = [
        {
            "name": clean["name"],
            "clean_normalized_max_abs": clean["normalized_max_abs"],
            "planted_normalized_max_abs": plant["normalized_max_abs"],
        }
        for clean, plant in zip(clean_rows, planted_rows, strict=True)
        if plant["normalized_max_abs"] != clean["normalized_max_abs"]
    ]
    require(changed, "velocity plant did not reach any scored moment row")

    input_moment_hash = _array_sha256(*(np.asarray(value) for value in state.moments))
    # Source corrections, ridge/raft, and ice_cor receive no moment argument;
    # step_ice_rheo_card returns the transport tuple verbatim at recipe:664-704.
    return {
        "gate": "nemo-si3-phase2-rung34-step8-moment-replay-v1",
        "status": summaries["oracle_uv"]["status"],
        "exit_code": 0 if jit_oracle_uv["non_bit_exact_count"] == 0 else 1,
        "backend": jax.default_backend(),
        "precision_policy": str(get_policy()),
        "pointwise_bar": POINTWISE_BAR,
        "ulp_limit": ULP_LIMIT,
        "sweep_order": "even step 8: y then x",
        "velocity_rows": velocity_rows,
        "arms": summaries,
        "jit_oracle_uv": jit_oracle_uv,
        "private_unrounded_ablation": jit_unrounded,
        "first_baseline_vs_oracle_velocity_moment_difference": first_difference,
        "limiter_input_identity": {
            "status": "EXACT",
            "reason": (
                "both arms have identical contents/moments before the "
                "velocity-free y limiter"
            ),
            "source": "icedyn_adv_pra.F90:757-791",
            "moment_sha256": input_moment_hash,
        },
        "between_advection_moment_handling": {
            "status": "EXACTLY-UNCHANGED",
            "source": "nemo_rheo_testcase_recipe.py:646-704",
            "input_transport_moment_sha256": input_moment_hash,
        },
        "plant": {
            "velocity_delta_m_s": PLANT_VELOCITY_M_S,
            "changed_scored_moment_rows": len(changed),
            "first_changed_row": changed[0],
            "jit_over_two_ulp_count": jit_plant_delta["over_two_ulp_count"],
            "delta_owner": jit_plant_delta["owner"],
            "binding_row": plant_binding_row,
            "exit_code": 1,
        },
        "artifacts": {
            str(entry_restart): _sha256(entry_restart),
            str(target_restart): _sha256(target_restart),
            str(root / f"oracle_ice_step_entry_kt{ENTRY_FRAME:08d}.bin"): _sha256(
                root / f"oracle_ice_step_entry_kt{ENTRY_FRAME:08d}.bin"
            ),
            str(root / f"oracle_ice_step_entry_kt{TARGET_FRAME:08d}.bin"): _sha256(
                root / f"oracle_ice_step_entry_kt{TARGET_FRAME:08d}.bin"
            ),
        },
    }


def _selected_exit_code(report: dict[str, object], *, plant: bool) -> int:
    if plant:
        plant_report = cast(dict[str, object], report["plant"])
        binding_row = cast(dict[str, object], plant_report["binding_row"])
        require(
            binding_row["name"] == PLANT_BINDING_ROW
            and binding_row["status"] == "DEBT"
            and int(binding_row["max_ulp"]) > ULP_LIMIT,
            f"plant exit requested without binding {PLANT_BINDING_ROW}",
        )
        return int(plant_report["exit_code"])
    return int(report["exit_code"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--restart-root", type=Path, default=RESTART_ROOT)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--jit-only", action="store_true")
    args = parser.parse_args()
    report = run_replay(
        args.root.resolve(), args.restart_root.resolve(), jit_only=args.jit_only
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return _selected_exit_code(report, plant=args.plant)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MomentReplayError as exc:
        print(f"UNMEASURED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)

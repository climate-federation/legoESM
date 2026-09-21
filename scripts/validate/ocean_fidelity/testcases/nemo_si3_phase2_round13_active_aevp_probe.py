#!/usr/bin/env python3
"""Exact-input ICE_RHEO step-9 aEVP operand discriminator.

The oracle operands are WRITE-only fields from a copied NEMO configuration;
no shipped source or testcase is modified.  The candidate uses the shared
C-grid solver for its production-JIT endpoint and the independent source-order
replay for its registered internal statement boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
from legoesm.ice.dynamics import SI3CGridAEVPState, si3_cgrid_aevp_solver
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    _forcing_for_state,
    build_ice_rheo_card,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp

from legoesm import constants

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
PROBE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/round13/run_with_init"
)
HERE = Path(__file__).resolve().parent
ORACLE_GATE = HERE / "nemo_si3_oracle_gate.py"
REPLAY = HERE / "nemo_si3_phase2_rung33_replay.py"
FRAME_STEP = 9
TARGET_FRAME_STEP = 10
N_SUBCYCLES = 100
HALO_WIDTH = 2
POINTWISE_BAR = 1.0e-15
HASH_BLOCK_BYTES = 1024 * 1024
PLANT_INDEX = (2, 2)

OPERAND_REGISTRY = (
    "shear",
    "shear2",
    "divergence",
    "divergence2",
    "tension",
    "tension2",
    "delta",
    "delta_floor",
    "p_over_delta",
    "alpha_t",
    "inverse_alpha_t",
    "beta_t",
    "alpha_f",
    "inverse_alpha_f",
    "p_over_delta_f",
    "stress1",
    "stress2",
    "stress12",
    "force_u",
    "force_v",
    "cross_v_u",
    "cross_u_v",
    "tauo_u",
    "ocean_stress_u",
    "speed_u",
    "taub_u",
    "bottom_stress_u",
    "coriolis_u",
    "rhs_u",
    "beta_u",
    "denominator_u",
    "raw_u",
    "thin_u",
    "prehalo_u",
    "tauo_v",
    "ocean_stress_v",
    "speed_v",
    "taub_v",
    "bottom_stress_v",
    "coriolis_v",
    "rhs_v",
    "beta_v",
    "denominator_v",
    "raw_v",
    "thin_v",
    "prehalo_v",
    "u",
    "v",
)

SETUP_REGISTRY = (
    "u",
    "v",
    "stress1",
    "stress2",
    "stress12",
    "strength_t",
    "dt_over_mass_t",
    "mass_over_dt_u",
    "mass_over_dt_v",
    "drag_u",
    "drag_v",
    "slope_u",
    "slope_v",
    "tau_air_u",
    "tau_air_v",
    "zmsk",
    "fimask",
    "active_u",
    "active_v",
    "mass_mask_u",
    "mass_mask_v",
    "fast_u",
    "fast_v",
    "u_b",
    "v_b",
    "concentration_t",
    "ice_volume_t",
    "snow_volume_t",
    "pond_volume_t",
    "lid_volume_t",
    "ocean_u_u",
    "ocean_v_v",
    "ocean_v_u",
    "ocean_u_v",
    "mass_coriolis_t",
    "base_u",
    "base_v",
)

SETUP_LAYOUTS = (
    "fffff",
    "ffssss",
    "ssssff",
    "ssssff",
    "fffffff",
    "ffssfss",
)

MESH_FIELDS = (
    "e1t",
    "e2t",
    "e1u",
    "e2u",
    "e1v",
    "e2v",
    "e1f",
    "e2f",
    "tmask",
    "umask",
    "vmask",
)


class ActiveAEVPProbeError(RuntimeError):
    """A fail-closed active-regime probe contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveAEVPProbeError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load("nemo_si3_oracle_gate_round13", ORACLE_GATE)
replay = _load("nemo_si3_rung33_replay_round13", REPLAY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(stream) -> bytes:
    marker = stream.read(struct.calcsize("=i"))
    require(len(marker) == struct.calcsize("=i"), "truncated Fortran record marker")
    size = struct.unpack("=i", marker)[0]
    payload = stream.read(size)
    require(len(payload) == size, "truncated Fortran record payload")
    trailer = stream.read(struct.calcsize("=i"))
    require(len(trailer) == struct.calcsize("=i"), "truncated Fortran record trailer")
    require(struct.unpack("=i", trailer)[0] == size, "Fortran record markers disagree")
    return payload


def _full_field(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    value = np.frombuffer(payload, dtype=np.float64)
    require(value.size == shape[0] * shape[1], "full-grid operand has wrong size")
    return cast(np.ndarray, value.reshape(shape, order="F"))


def _unpack_layout(
    payload: bytes, layout: str, shape: tuple[int, int]
) -> list[np.ndarray]:
    values = np.frombuffer(payload, dtype=np.float64)
    result: list[np.ndarray] = []
    offset = 0
    for kind in layout:
        target_shape = shape if kind == "f" else (shape[0] - 2, shape[1] - 2)
        size = target_shape[0] * target_shape[1]
        value = values[offset : offset + size].reshape(target_shape, order="F")
        offset += size
        if kind == "s":
            expanded = np.zeros(shape, dtype=np.float64)
            expanded[1:-1, 1:-1] = value
            value = expanded
        result.append(value)
    require(offset == values.size, "setup-record layout does not consume its payload")
    return result


def _read_setup(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    with path.open("rb") as stream:
        jpi, jpj, kt, subcycles = struct.unpack("=4i", _record(stream))
        arrays: list[np.ndarray] = []
        for layout in SETUP_LAYOUTS:
            arrays.extend(_unpack_layout(_record(stream), layout, (jpi, jpj)))
        require(stream.read(1) == b"", "setup dump has unregistered trailing bytes")
    require(len(arrays) == len(SETUP_REGISTRY), "setup registry length mismatch")
    return {"jpi": jpi, "jpj": jpj, "kt": kt, "subcycles": subcycles}, dict(
        zip(SETUP_REGISTRY, arrays, strict=True)
    )


def _read_subcycle(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    with path.open("rb") as stream:
        jpi, jpj, kt, iteration = struct.unpack("=4i", _record(stream))
        arrays = [_full_field(_record(stream), (jpi, jpj)) for _ in OPERAND_REGISTRY]
        require(stream.read(1) == b"", "subcycle dump has unregistered trailing bytes")
    return {"jpi": jpi, "jpj": jpj, "kt": kt, "iteration": iteration}, dict(
        zip(OPERAND_REGISTRY, arrays, strict=True)
    )


def _ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    require(left.dtype == right.dtype == np.float64, "ULP comparison requires fp64")
    require(np.all(np.isfinite(left)) and np.all(np.isfinite(right)), "non-finite score input")
    sign = np.uint64(1 << 63)
    left_bits = left.view(np.uint64)
    right_bits = right.view(np.uint64)
    left_ordered = np.where(left_bits & sign, ~left_bits, left_bits | sign)
    right_ordered = np.where(right_bits & sign, ~right_bits, right_bits | sign)
    return cast(
        np.ndarray,
        np.where(
            left_ordered >= right_ordered,
            left_ordered - right_ordered,
            right_ordered - left_ordered,
        ),
    )


def _score(name: str, oracle: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    require(oracle.shape == candidate.shape, f"{name}: score shape mismatch")
    if np.array_equal(oracle, candidate):
        return {
            "name": name,
            "bitwise_nonzero_over_n": f"0 / {oracle.size}",
            "max_abs": 0.0,
            "normalized_max_abs": 0.0,
            "relative_max_abs": 0.0 if np.any(oracle) else None,
            "max_ulp": 0,
            "max_abs_index_xy": [0, 0],
            "status": "BIT-EXACT",
        }
    difference = np.abs(candidate - oracle)
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    oracle_max = float(np.max(np.abs(oracle)))
    max_abs = float(difference[index])
    ulp = _ulp_distance(oracle, candidate)
    return {
        "name": name,
        "bitwise_nonzero_over_n": f"{np.count_nonzero(difference)} / {difference.size}",
        "max_abs": max_abs,
        "normalized_max_abs": max_abs / max(oracle_max, 1.0),
        "relative_max_abs": None if oracle_max == 0.0 else max_abs / oracle_max,
        "max_ulp": int(np.max(ulp)),
        "max_abs_index_xy": [int(value) for value in index],
        "status": "BIT-EXACT" if max_abs == 0.0 else "DEBT",
    }


def _physical(value: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, value[HALO_WIDTH:-HALO_WIDTH, HALO_WIDTH:-HALO_WIDTH])


def _merge_aggregate(
    aggregates: dict[str, dict[str, object]], row: dict[str, object], iteration: int
) -> None:
    name = str(row["name"]).split(".")[-1]
    current = aggregates.setdefault(
        name,
        {
            "name": name,
            "nonexact_subcycles": 0,
            "first_nonexact_subcycle": None,
            "max_abs": 0.0,
            "max_normalized_abs": 0.0,
            "max_relative_abs": 0.0,
            "max_ulp": 0,
            "status": "BIT-EXACT",
        },
    )
    if row["status"] != "BIT-EXACT":
        current["nonexact_subcycles"] = int(current["nonexact_subcycles"]) + 1
        if current["first_nonexact_subcycle"] is None:
            current["first_nonexact_subcycle"] = iteration
        current["status"] = "DEBT"
    current["max_abs"] = max(float(current["max_abs"]), float(row["max_abs"]))
    current["max_normalized_abs"] = max(
        float(current["max_normalized_abs"]), float(row["normalized_max_abs"])
    )
    relative = row["relative_max_abs"]
    if relative is not None:
        current["max_relative_abs"] = max(float(current["max_relative_abs"]), float(relative))
    current["max_ulp"] = max(int(current["max_ulp"]), int(row["max_ulp"]))


def _assert_plant_transition(
    clean_row: dict[str, object], planted_row: dict[str, object]
) -> bool:
    """Require the named row, not merely the process status, to move red."""

    require(clean_row["name"] == planted_row["name"], "plant changed row identity")
    require(clean_row["status"] == "BIT-EXACT", "plant baseline row is not clean")
    require(planted_row["status"] == "DEBT", "plant did not move its named row")
    require(
        clean_row["bitwise_nonzero_over_n"] != planted_row["bitwise_nonzero_over_n"],
        "plant did not change the row numerator",
    )
    return True


def run_probe(
    root: Path = ROOT,
    probe_root: Path = PROBE_ROOT,
    *,
    plant: bool = False,
) -> tuple[dict[str, object], int]:
    require(jax.default_backend() == "cpu", "active aEVP probe requires CPU")
    frame_path = root / f"oracle_ice_step_entry_kt{FRAME_STEP:08d}.bin"
    target_path = root / f"oracle_ice_step_entry_kt{TARGET_FRAME_STEP:08d}.bin"
    setup_path = probe_root / "round13_aevp_setup.bin"
    init_path = root / "output.init_ice.nc"
    mesh_path = root / "mesh_mask.nc"
    for path in (frame_path, target_path, setup_path, init_path, mesh_path):
        require(path.is_file(), f"missing probe input: {path}")

    header, frame = oracle_gate.read_frame(frame_path)
    target_header, target = oracle_gate.read_frame(target_path)
    require(header["kt"] == FRAME_STEP, "oracle entry clock mismatch")
    require(target_header["kt"] == TARGET_FRAME_STEP, "oracle target clock mismatch")
    require(header["storage_bits"] == 64, "oracle entry is not fp64")
    with netCDF4.Dataset(mesh_path) as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in MESH_FIELDS}
    with netCDF4.Dataset(init_path) as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = build_ice_rheo_card(frame, mesh, ocean_temperature_k)
    require(card.dynamics_config.n_subcycles == N_SUBCYCLES, "aEVP subcycle count drift")
    require(str(card.initial_state.contents.dtype) == "float64", "card contents are not fp64")

    setup_header, setup = _read_setup(setup_path)
    require(setup_header["kt"] == FRAME_STEP, "setup dump clock mismatch")
    require(setup_header["subcycles"] == N_SUBCYCLES, "setup subcycle count mismatch")
    initial = replay._oracle_state(frame)
    input_rows = {
        name: _score(
            f"input.{name}",
            _physical(setup[name]),
            _physical(np.asarray(getattr(initial, name))),
        )
        for name in replay.ReplayState._fields
    }
    require(
        all(row["status"] == "BIT-EXACT" for row in input_rows.values()),
        "copied probe did not receive exact oracle rheology carries",
    )

    forcing = _forcing_for_state(card, card.initial_state, FRAME_STEP)
    q = replay._setup(card, forcing)
    for name in (
        "strength_t",
        "dt_over_mass_t",
        "mass_over_dt_u",
        "mass_over_dt_v",
        "drag_u",
        "drag_v",
        "slope_u",
        "slope_v",
        "tau_air_u",
        "tau_air_v",
        "zmsk",
        "fimask",
        "active_u",
        "active_v",
        "mass_mask_u",
        "mass_mask_v",
        "fast_u",
        "fast_v",
        "ocean_u_u",
        "ocean_v_v",
        "ocean_v_u",
        "ocean_u_v",
        "mass_coriolis_t",
        "base_u",
        "base_v",
    ):
        q[name] = setup[name]

    current = initial
    aggregates: dict[str, dict[str, object]] = {}
    selected_rows: dict[str, dict[str, dict[str, object]]] = {}
    dump_hashes: dict[str, str] = {}
    clean_plant_row: dict[str, object] | None = None
    planted_row: dict[str, object] | None = None
    scored_iterations = (1,) if plant else range(1, N_SUBCYCLES + 1)
    for iteration in scored_iterations:
        dump_path = probe_root / f"round13_aevp_subcycle_{iteration:03d}.bin"
        require(dump_path.is_file(), f"missing subcycle dump {iteration}")
        dump_header, oracle_operands = _read_subcycle(dump_path)
        require(dump_header["kt"] == FRAME_STEP, f"subcycle {iteration}: clock mismatch")
        require(dump_header["iteration"] == iteration, f"subcycle {iteration}: index mismatch")
        current, candidate_operands = replay._subcycle(current, initial, q, iteration - 1)
        cycle_rows: dict[str, dict[str, object]] = {}
        for name in OPERAND_REGISTRY:
            candidate = np.asarray(candidate_operands[name]).copy()
            clean_row = _score(
                f"subcycle{iteration:03d}.{name}",
                _physical(oracle_operands[name]),
                _physical(candidate),
            )
            row = clean_row
            if plant and iteration == 1 and name == "force_u":
                clean_plant_row = clean_row
                candidate[PLANT_INDEX] = np.nextafter(candidate[PLANT_INDEX], np.inf)
                row = _score(
                    f"subcycle{iteration:03d}.{name}",
                    _physical(oracle_operands[name]),
                    _physical(candidate),
                )
                planted_row = row
            cycle_rows[name] = row
            _merge_aggregate(aggregates, row, iteration)
        if iteration in (1, 2, 9, N_SUBCYCLES):
            selected_rows[str(iteration)] = cycle_rows
        dump_hashes[dump_path.name] = _sha256(dump_path)

    jax_state = SI3CGridAEVPState(*(jnp.asarray(value) for value in initial))
    jax_forcing = type(forcing)(*(jnp.asarray(value) for value in forcing))
    solve = jax.jit(
        lambda state, applied_forcing: si3_cgrid_aevp_solver(
            state,
            applied_forcing,
            card.metrics,
            card.dynamics_config,
        )
    )
    production = solve(jax_state, jax_forcing)
    jax.block_until_ready(production)
    final_dump_header, final_dump = _read_subcycle(
        probe_root / f"round13_aevp_subcycle_{N_SUBCYCLES:03d}.bin"
    )
    del final_dump_header
    endpoint_names = ("u", "v", "stress1", "stress2", "stress12")
    endpoint_rows = {
        name: _score(
            f"production_endpoint.{name}",
            _physical(final_dump[name]),
            _physical(np.asarray(value)),
        )
        for name, value in zip(endpoint_names, production, strict=True)
    }
    target_names = ("u_ice", "v_ice", "stress1_i", "stress2_i", "stress12_i")
    observer_rows = {
        name: _score(
            f"observer_effect.{name}",
            _physical(np.asarray(target[target_name])),
            _physical(final_dump[name]),
        )
        for name, target_name in zip(endpoint_names, target_names, strict=True)
    }

    operands_exact = all(row["status"] == "BIT-EXACT" for row in aggregates.values())
    endpoint_exact = all(row["status"] == "BIT-EXACT" for row in endpoint_rows.values())
    observer_exact = all(row["status"] == "BIT-EXACT" for row in observer_rows.values())
    if plant:
        require(clean_plant_row is not None and planted_row is not None, "plant row was not scored")
        _assert_plant_transition(clean_plant_row, planted_row)
    clean = operands_exact and endpoint_exact and observer_exact
    status = "BIT-EXACT" if clean and not plant else "DEBT"
    report = {
        "format": "nemo-si3-phase2-round13-active-aevp-probe-v1",
        "worktree": worktree_stamp(),
        "status": status,
        "execution": "CPU/fp64 production JIT; scalar-libm transcendental policy",
        "clock": {
            "oracle_entry_kt": FRAME_STEP,
            "completed_step": FRAME_STEP,
            "oracle_target_kt": TARGET_FRAME_STEP,
            "subcycles": N_SUBCYCLES,
        },
        "bar": POINTWISE_BAR,
        "bar_definition": "normalized max absolute error with scale=max(max|oracle|,1)",
        "relative_column": "diagnostic max absolute error/max|oracle|; not the gate bar",
        "dtype": {
            "oracle": "float64",
            "card_contents": str(card.initial_state.contents.dtype),
            "production_u": str(production.u_ice_u.dtype),
        },
        "source_registry": {
            "deformation_and_closure": "icedyn_rhg_evp.F90:392-489",
            "stress_divergence_and_cross_velocity": "icedyn_rhg_evp.F90:493-514",
            "even_velocity_order": "icedyn_rhg_evp.F90:532-634",
            "odd_velocity_order": "icedyn_rhg_evp.F90:638-741",
        },
        "input_rows": input_rows,
        "operand_aggregates": aggregates,
        "selected_subcycle_rows": selected_rows,
        "production_endpoint_rows": endpoint_rows,
        "observer_effect_rows": observer_rows,
        "predicates": {
            "all_scored_operands_exact": operands_exact,
            "clean_scope_is_all_48_operands_for_100_subcycles": not plant,
            "production_endpoint_5_of_5_exact": endpoint_exact,
            "instrumented_endpoint_matches_uninstrumented_next_entry": observer_exact,
            "preregistered_defect_prediction_confirmed": not operands_exact,
            "preregistered_amplification_arm_confirmed": operands_exact,
        },
        "plant": {
            "enabled": plant,
            "control_scope": "subcycle 1 only" if plant else "not applicable",
            "row": "subcycle001.force_u",
            "clean": clean_plant_row,
            "perturbed": planted_row,
            "row_transition_bound": bool(
                plant
                and clean_plant_row is not None
                and planted_row is not None
                and _assert_plant_transition(clean_plant_row, planted_row)
            ),
        },
        "artifacts": {
            str(frame_path): _sha256(frame_path),
            str(target_path): _sha256(target_path),
            str(setup_path): _sha256(setup_path),
            str(probe_root / "nemo.exe"): _sha256(probe_root / "nemo.exe"),
            "subcycle_dumps": dump_hashes,
        },
    }
    return report, 0 if status == "BIT-EXACT" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--probe-root", type=Path, default=PROBE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report, exit_code = run_probe(args.root, args.probe_root, plant=args.plant)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

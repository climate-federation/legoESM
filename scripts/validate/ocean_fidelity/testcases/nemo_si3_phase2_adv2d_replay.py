#!/usr/bin/env python3
"""Classify ICE_ADV2D's first DEBT with a NumPy written-order replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import types
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final")
OUTPUT = ROOT.parent / "phase2" / "nemo_si3_phase2_adv2d_replay.json"
REPO_ROOT = Path(__file__).resolve().parents[4]
GATE_PATH = Path(__file__).with_name("nemo_si3_phase2_adv2d_gate.py")
POINTWISE_BAR = 1.0e-15
PREPARE_STEPS = 15
DEBT_STEP = 16
ORACLE_INPUT_KT = 16
ORACLE_ENTRY_KT = 17
ULP_LIMIT = 2
TARGET_TRACER = "szv_i_l01"
PLANT_RELATIVE = 1.0e-6  # instrument control only; intentionally above fp64 roundoff
_SHA256_CHUNK_BYTES = 1024 * 1024  # streaming-I/O implementation choice
_H_BIG_MAXIMUM_CONCENTRATION = 0.15  # icedyn_adv_pra.F90:992,1000,1007
_ICE_MAXIMUM_THICKNESS_M = 99.0  # ICE_ADV2D namelist_ice_ref:46 rn_himax
_UINT64_SIGN_BIT_INDEX = 63  # IEEE-754 binary64 sign-bit position


class ReplayError(RuntimeError):
    """Fail-closed replay error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("nemo_si3_phase2_adv2d_gate_replay", GATE_PATH)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_SHA256_CHUNK_BYTES), b""):
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


def _clone_with_numpy(function, namespace: dict):
    clone = types.FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def _numpy_prather_functions():
    """Execute the production source-ordered graph with NumPy scalar ufuncs."""

    import legoesm.ice.transport as transport

    namespace = dict(transport.__dict__)
    namespace["jnp"] = np
    names = (
        "_si3_prather_limit_x",
        "_si3_prather_limit_y",
        "_si3_prather_merge_from_left",
        "_si3_prather_merge_from_right",
        "_si3_prather_merge_from_below",
        "_si3_prather_merge_from_above",
        "_si3_periodic_halo_xy",
    )
    for name in names:
        namespace[name] = _clone_with_numpy(getattr(transport, name), namespace)
    namespace["_si3_prather_x_substep"] = _clone_with_numpy(
        transport._si3_prather_x_substep, namespace
    )
    namespace["_si3_prather_y_substep"] = _clone_with_numpy(
        transport._si3_prather_y_substep, namespace
    )
    return namespace["_si3_prather_x_substep"], namespace["_si3_prather_y_substep"]


def _periodic_numpy(value: np.ndarray, halo: int) -> np.ndarray:
    interior = value[halo:-halo, halo:-halo, ...]
    padding = ((halo, halo), (halo, halo)) + ((0, 0),) * (value.ndim - 2)
    return cast(np.ndarray, np.pad(interior, padding, mode="wrap"))


def _hbig_and_halo_numpy(card, entry: np.ndarray, contents: np.ndarray, moments):
    """Post-split Hbig/halo order from icedyn_adv_pra.F90:355-367,405-479."""

    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    area = card.dx_m * card.dy_m
    v_index = ICE_ADV2D_TRACERS.index("v_i")
    a_index = ICE_ADV2D_TRACERS.index("a_i")
    entry_v = entry[..., v_index] / area
    entry_a = entry[..., a_index] / area
    entry_h = np.where(entry_a > 1.0e-10, entry_v / np.where(entry_a > 1.0e-10, entry_a, 1.0), 0.0)
    h_max = np.maximum(
        1.0e-20,
        np.max(
            np.stack(
                [
                    np.roll(np.roll(entry_h, di, axis=0), dj, axis=1)
                    for di in (-1, 0, 1)
                    for dj in (-1, 0, 1)
                ]
            ),
            axis=0,
        ),
    )
    volume = contents[..., v_index] / area
    concentration = contents[..., a_index] / area
    thickness = np.where(
        concentration > 0.0,
        volume / np.where(concentration > 0.0, concentration, 1.0),
        0.0,
    )
    active = np.zeros(concentration.shape, dtype=bool)
    halo = card.halo_width
    active[halo:-halo, halo:-halo] = True
    correct = (
        active
        & (volume > 0.0)
        & (concentration > 0.0)
        & (thickness > h_max)
        & (concentration < _H_BIG_MAXIMUM_CONCENTRATION)
    )
    concentration = np.where(
        correct,
        volume / np.minimum(h_max, _ICE_MAXIMUM_THICKNESS_M),
        concentration,
    )
    contents = np.asarray(contents).copy()
    contents[..., a_index] = concentration * area
    return _periodic_numpy(contents, halo), tuple(
        _periodic_numpy(value, halo) for value in moments
    )


def _numpy_transport_step(card, state, *, plant: bool = False):
    """Even-step y-then-x replay of icedyn_adv_pra.F90:253-351,722-943,499-719."""

    x_step, y_step = _numpy_prather_functions()
    entry = np.asarray(state.contents).copy()
    moments = tuple(np.asarray(value).copy() for value in state.moments)
    if plant:
        plane = moments[4][..., 0]
        location = np.unravel_index(int(np.argmax(np.abs(plane))), plane.shape)
        require(plane[location] != 0.0, "planted cross-moment operand is zero")
        moments[4][location + (0,)] *= 1.0 + PLANT_RELATIVE
    area_value = card.dx_m * card.dy_m
    area = np.full(entry.shape[:2], area_value, dtype=np.float64)
    wet = np.ones(entry.shape[:2], dtype=bool)
    u_transport = np.asarray(card.prescribed_u_ice) * card.dy_m
    v_transport = np.asarray(card.prescribed_v_ice) * card.dx_m
    after_y, moments_y, swept_area = y_step(
        entry,
        moments,
        v_transport,
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
        u_transport,
        area,
        wet,
        card.dt_s,
        initial_area=swept_area,
        first_sweep=False,
        halo_width=card.halo_width,
        subcycle_index=1,
        subcycles=card.subcycles,
    )
    contents, moments_out = _hbig_and_halo_numpy(card, entry, after_x, moments_x)
    return contents, moments_out, {"after_y": (after_y, moments_y), "after_x": (after_x, moments_x)}


def _finish_card_step(card, state, contents: np.ndarray, moments):
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
        apply_ice_adv2d_source_corrections,
        apply_ice_adv2d_zapsmall,
    )
    from legoesm.ice.transport import (
        si3_prather_pack_intensives,
        si3_prather_unpack_intensives,
    )

    cell_area = jnp.full(state.contents.shape[:2], card.dx_m * card.dy_m, dtype=jnp.float64)
    wet = jnp.ones(state.contents.shape[:2], dtype=bool)
    entry_intensives = si3_prather_unpack_intensives(state.contents, cell_area, wet)
    transported_intensives = si3_prather_unpack_intensives(
        jnp.asarray(contents), cell_area, wet
    )
    corrected = apply_ice_adv2d_source_corrections(
        card,
        entry_intensives,
        transported_intensives,
        entry_intensive_contents=entry_intensives,
        contents_are_intensive=True,
    )
    result = apply_ice_adv2d_zapsmall(
        card,
        state,
        corrected,
        tuple(jnp.asarray(value) for value in moments),
        contents_are_intensive=True,
    )
    return result._replace(contents=si3_prather_pack_intensives(result.contents, cell_area))


def _ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    require(left.dtype == right.dtype == np.float64, "ULP comparison requires fp64")
    require(
        bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right))),
        "non-finite ULP operand",
    )
    left_bits: np.ndarray = left.view(np.uint64)
    right_bits: np.ndarray = right.view(np.uint64)
    sign = np.uint64(1 << _UINT64_SIGN_BIT_INDEX)
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


def _field_row(name: str, oracle: np.ndarray, production: np.ndarray, replay: np.ndarray) -> dict:
    scale = max(1.0, float(np.max(np.abs(oracle))))
    prod_abs = np.abs(production - oracle)
    replay_abs = np.abs(replay - oracle)
    prod_index = np.unravel_index(int(np.argmax(prod_abs)), prod_abs.shape)
    return {
        "name": name,
        "production_max_abs": float(np.max(prod_abs)),
        "production_normalized_max_abs": float(np.max(prod_abs)) / scale,
        "production_max_ulp": int(np.max(_ulp_distance(production, oracle))),
        "replay_max_abs": float(np.max(replay_abs)),
        "replay_normalized_max_abs": float(np.max(replay_abs)) / scale,
        "replay_max_ulp": int(np.max(_ulp_distance(replay, oracle))),
        "production_max_abs_index_xy": [int(value) for value in prod_index],
        "oracle_at_production_max": float(oracle[prod_index]),
        "production_at_max": float(production[prod_index]),
        "replay_at_production_max": float(replay[prod_index]),
        "replay_ulp_at_production_max": int(
            _ulp_distance(replay, oracle)[prod_index]
        ),
    }


def _first_stage_difference(production_stages: dict, replay_stages: dict) -> dict | None:
    labels = ("content", "sx", "sy", "sxx", "syy", "sxy")
    for stage in ("after_y", "after_x"):
        prod_content, prod_moments = production_stages[stage]
        replay_content, replay_moments = replay_stages[stage]
        for label, prod, replay in zip(
            labels,
            (prod_content, *prod_moments),
            (replay_content, *replay_moments),
            strict=True,
        ):
            prod_array = np.asarray(prod)
            replay_array = np.asarray(replay)
            distance = _ulp_distance(prod_array, replay_array)
            if np.any(distance > ULP_LIMIT):
                index = np.unravel_index(int(np.argmax(distance)), distance.shape)
                return {
                    "stage": stage,
                    "operand": label,
                    "index": [int(value) for value in index],
                    "production": float(prod_array[index]),
                    "replay": float(replay_array[index]),
                    "ulp_distance": int(distance[index]),
                }
    return None


def run_replay(root: Path) -> dict:
    import legoesm.ice.transport as transport
    from legoesm.core.precision import PrecisionPolicy, get_policy
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
        ICE_ADV2D_TRACERS,
        build_ice_adv2d_card,
        step_ice_adv2d_card,
    )
    require(jax.default_backend() == "cpu", "CPU backend required")
    card = build_ice_adv2d_card(gate.oracle_surface_temperature_c(root))
    require(get_policy() == PrecisionPolicy.fp64(), "fp64 policy required")
    state = card.initial_state
    target_input_history = []
    for completed in range(PREPARE_STEPS):
        kt = completed + 1
        history_frame = gate._read_oracle_frame(root, kt=kt, card=card)
        history_oracle = np.asarray(
            gate._entry_tracer(history_frame, TARGET_TRACER), dtype=np.float64
        )
        history_candidate = np.asarray(
            gate.state_field(card, state, TARGET_TRACER), dtype=np.float64
        )
        history_row = _field_row(
            TARGET_TRACER,
            history_oracle,
            history_candidate,
            history_candidate,
        )
        target_input_history.append({"kt": kt, **history_row})
        state = step_ice_adv2d_card(card, state, completed_steps=completed)

    production = step_ice_adv2d_card(card, state, completed_steps=PREPARE_STEPS)
    replay_contents, replay_moments, replay_stages = _numpy_transport_step(card, state)
    replay = _finish_card_step(card, state, replay_contents, replay_moments)

    area_value = card.dx_m * card.dy_m
    area = jnp.full(state.contents.shape[:2], area_value, dtype=jnp.float64)
    wet = jnp.ones(state.contents.shape[:2], dtype=bool)
    after_y, moments_y, swept_area = transport._si3_prather_y_substep(
        state.contents,
        state.moments,
        card.prescribed_v_ice * card.dx_m,
        area,
        wet,
        card.dt_s,
        initial_area=area,
        first_sweep=True,
        halo_width=card.halo_width,
        subcycle_index=1,
        subcycles=card.subcycles,
    )
    after_x, moments_x, _ = transport._si3_prather_x_substep(
        after_y,
        moments_y,
        card.prescribed_u_ice * card.dy_m,
        area,
        wet,
        card.dt_s,
        initial_area=swept_area,
        first_sweep=False,
        halo_width=card.halo_width,
        subcycle_index=1,
        subcycles=card.subcycles,
    )
    production_stages = {
        "after_y": (after_y, moments_y),
        "after_x": (after_x, moments_x),
    }

    input_frame_path = root / f"oracle_ice_step_entry_kt{ORACLE_INPUT_KT:08d}.bin"
    input_frame = gate._read_oracle_frame(root, kt=ORACLE_INPUT_KT, card=card)
    input_rows = []
    for tracer in ICE_ADV2D_TRACERS:
        oracle_input = np.asarray(gate._entry_tracer(input_frame, tracer), dtype=np.float64)
        candidate_input = np.asarray(gate.state_field(card, state, tracer), dtype=np.float64)
        input_rows.append(
            _field_row(tracer, oracle_input, candidate_input, candidate_input)
        )

    frame_path = root / f"oracle_ice_step_entry_kt{ORACLE_ENTRY_KT:08d}.bin"
    frame = gate._read_oracle_frame(root, kt=ORACLE_ENTRY_KT, card=card)
    halo = card.halo_width
    rows = []
    for tracer in ICE_ADV2D_TRACERS:
        oracle = np.asarray(gate._entry_tracer(frame, tracer), dtype=np.float64)
        prod = np.asarray(gate.state_field(card, production, tracer), dtype=np.float64)
        replay_field = np.asarray(gate.state_field(card, replay, tracer), dtype=np.float64)
        rows.append(_field_row(tracer, oracle, prod, replay_field))

    target = next(row for row in rows if row["name"] == TARGET_TRACER)
    input_target = next(row for row in input_rows if row["name"] == TARGET_TRACER)
    initial_frame = gate._read_oracle_frame(root, kt=1, card=card)
    initial_oracle_salt = np.asarray(
        gate._entry_tracer(initial_frame, TARGET_TRACER), dtype=np.float64
    )
    initial_bulk_salt = np.asarray(card.initial_state.bulk_salt_diagnostic)[
        halo:-halo, halo:-halo
    ]
    reciprocal_nlay_i = np.float64(1.0) / np.float64(card.nlay_i)
    nemo_written_initial_salt = initial_bulk_salt * reciprocal_nlay_i
    initialization_order = _field_row(
        TARGET_TRACER,
        initial_oracle_salt,
        np.asarray(gate.state_field(card, card.initial_state, TARGET_TRACER)),
        nemo_written_initial_salt,
    )
    target_input_history.append({"kt": ORACLE_INPUT_KT, **input_target})
    first_input_over_two_ulp = next(
        (
            row
            for row in target_input_history
            if row["production_max_ulp"] > ULP_LIMIT
        ),
        None,
    )
    if (
        target["replay_ulp_at_production_max"] <= ULP_LIMIT
        and target["production_max_ulp"] > ULP_LIMIT
    ):
        classification = "RE-ASSOCIATION"
    elif (
        np.array_equal(np.asarray(production.contents), np.asarray(replay.contents))
        and all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(production.moments, replay.moments, strict=True)
        )
        and input_target["production_max_ulp"] > ULP_LIMIT
    ):
        classification = "INHERITED_STEP_ENTRY_DEBT"
    else:
        classification = "IMPLEMENTATION_OR_UNMEASURED_MOMENT_INPUT_DEBT"
    moment_rows = []
    for moment_name, prod_moment, replay_moment in zip(
        transport.SI3_PRATHER_MOMENT_NAMES,
        production.moments,
        replay.moments,
        strict=True,
    ):
        for tracer_index, tracer in enumerate(ICE_ADV2D_TRACERS):
            prod = np.asarray(prod_moment)[halo:-halo, halo:-halo, tracer_index]
            replay_value = np.asarray(replay_moment)[halo:-halo, halo:-halo, tracer_index]
            moment_rows.append(
                {
                    "name": f"{moment_name}.{tracer}",
                    "production_replay_max_abs": float(np.max(np.abs(prod - replay_value))),
                    "production_replay_max_ulp": int(np.max(_ulp_distance(prod, replay_value))),
                }
            )

    planted_contents, planted_moments, _ = _numpy_transport_step(card, state, plant=True)
    planted = _finish_card_step(card, state, planted_contents, planted_moments)
    normal_v = np.asarray(replay.contents)[..., 0]
    planted_v = np.asarray(planted.contents)[..., 0]
    planted_change = int(np.max(_ulp_distance(normal_v, planted_v)))
    require(planted_change > 0, "planted arithmetic-order control was vacuous")

    return {
        "classification": classification,
        "scope": "ICE_ADV2D_OMIP_L3_STEP16_WRITTEN_ORDER_REPLAY",
        "backend": jax.default_backend(),
        "precision_policy": "fp64",
        "pointwise_bar": POINTWISE_BAR,
        "ulp_limit": ULP_LIMIT,
        "candidate_input_after_completed_steps": PREPARE_STEPS,
        "replayed_completed_step": DEBT_STEP,
        "oracle_entry_kt": ORACLE_ENTRY_KT,
        "sweep_order": "y_then_x",
        "target": target,
        "target_input": input_target,
        "target_input_history": target_input_history,
        "first_target_input_over_two_ulp": first_input_over_two_ulp,
        "initial_salt_reciprocal_order": {
            "source": "iceistate.F90:357-360; r1_nlay_i is a stored reciprocal",
            **initialization_order,
        },
        "first_production_replay_stage_difference": _first_stage_difference(
            production_stages, replay_stages
        ),
        "tracer_rows": rows,
        "input_tracer_rows": input_rows,
        "moment_rows": moment_rows,
        "planted_control": {
            "status": "RED",
            "changed_v_i_max_ulp": planted_change,
        },
        "provenance": {
            "oracle_frame": str(frame_path),
            "oracle_frame_sha256": _sha256(frame_path),
            "oracle_input_frame": str(input_frame_path),
            "oracle_input_frame_sha256": _sha256(input_frame_path),
            "candidate_input_sha256": _array_sha256(
                np.asarray(state.contents),
                *(np.asarray(value) for value in state.moments),
            ),
            "production_output_sha256": _array_sha256(
                np.asarray(production.contents),
                *(np.asarray(value) for value in production.moments),
            ),
            "replay_output_sha256": _array_sha256(
                np.asarray(replay.contents),
                *(np.asarray(value) for value in replay.moments),
            ),
            "transport_source_sha256": _sha256(
                REPO_ROOT / "packages/ice/legoesm/ice/transport.py"
            ),
            "replay_source_sha256": _sha256(Path(__file__).resolve()),
        },
        "blind_spot": (
            "step-entry frames omit Prather moments; moment rows compare production "
            "to replay only, not either implementation to NEMO"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run_replay(args.run_dir.resolve())
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        print(f"UNMEASURED: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)

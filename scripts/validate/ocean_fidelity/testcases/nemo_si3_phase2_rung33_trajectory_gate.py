#!/usr/bin/env python3
"""Full fp64 trajectory/restart gate for SI3 lane-3 rung 3.3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import cast

import jax
import netCDF4
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

POINTWISE_BAR = 1.0e-15
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final")
HERE = Path(__file__).resolve().parent
PHASE1 = HERE / "nemo_si3_oracle_gate.py"
RUNG32 = HERE / "nemo_si3_phase2_adv2d_gate.py"
RUNG33 = HERE / "nemo_si3_phase2_rung33_gate.py"
REPLAY = HERE / "nemo_si3_phase2_rung33_replay.py"
GROWTH_STEPS = (1, 10, 50, 100, 200, 485)
DYNAMICS_FIELDS = (
    "u_ice",
    "v_ice",
    "stress1_i",
    "stress2_i",
    "stress12_i",
)


class TrajectoryGateError(RuntimeError):
    """Fail-closed trajectory or restart coverage error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrajectoryGateError(message)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load("nemo_si3_oracle_gate_rung33_trajectory", PHASE1)
rung32_gate = _load("nemo_si3_phase2_adv2d_gate_rung33_trajectory", RUNG32)
rung33_gate = _load("nemo_si3_phase2_rung33_gate_trajectory", RUNG33)
replay_gate = _load("nemo_si3_phase2_rung33_replay_trajectory", REPLAY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detailed_score(name: str, oracle, candidate) -> dict[str, object]:
    oracle = np.asarray(oracle)
    candidate = np.asarray(candidate)
    require(oracle.shape == candidate.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is not fp64")
    require(
        bool(np.all(np.isfinite(oracle)) and np.all(np.isfinite(candidate))),
        f"{name}: non-finite input",
    )
    absolute = np.abs(candidate - oracle)
    index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    oracle_max_abs = float(np.max(np.abs(oracle)))
    scale = max(1.0, oracle_max_abs)
    normalized = float(absolute[index]) / scale
    relative = (
        float(absolute[index]) / oracle_max_abs
        if oracle_max_abs > 0.0
        else (0.0 if float(absolute[index]) == 0.0 else None)
    )
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= POINTWISE_BAR else "DEBT",
        "normalized_max_abs": normalized,
        "relative_max_abs": relative,
        "max_abs": float(absolute[index]),
        "oracle_max_abs": oracle_max_abs,
        "scale": scale,
        "max_abs_index_xy": [int(value) for value in index],
        "oracle_at_max": float(oracle[index]),
        "candidate_at_max": float(candidate[index]),
        "nonzero_error_cells": int(np.count_nonzero(absolute)),
        "n": int(oracle.size),
        "bar": POINTWISE_BAR,
    }


def _frame(root: Path, kt: int, card) -> dict[str, np.ndarray]:
    return cast(
        dict[str, np.ndarray],
        rung32_gate._read_oracle_frame(root, kt=kt, card=card.base),
    )


def _restart_field(dataset: netCDF4.Dataset, name: str) -> np.ndarray:
    if name in DYNAMICS_FIELDS:
        return cast(np.ndarray, rung32_gate._restart_2d_field(dataset, name))
    return cast(np.ndarray, rung32_gate._restart_field(dataset, name))


def _oracle_boundary_field(source, name: str, *, restart: bool) -> np.ndarray:
    if restart:
        if name.startswith(("e_s_l", "e_i_l", "szv_i_l")):
            return cast(np.ndarray, rung32_gate._restart_tracer(source, name))
        if name == "t_surface":
            return _restart_field(source, "t_su")
        return _restart_field(source, name)
    return cast(np.ndarray, rung33_gate._oracle_field(source, name))


def _candidate_boundary_field(card, state, name: str) -> np.ndarray:
    return cast(np.ndarray, rung33_gate._candidate_field(card, state, name))


def _face_threshold_census(contents: np.ndarray, dynamics, card) -> dict[str, object]:
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    from legoesm import constants

    def field(name: str) -> np.ndarray:
        return contents[..., ICE_ADV2D_TRACERS.index(name)]

    concentration = field("a_i")
    mass = (
        constants.rho_snow * field("v_s")
        + constants.rho_ice * field("v_i")
        + constants.rho_water * (field("v_ip") + field("v_il"))
    )
    mass_u = 0.5 * (mass + np.roll(mass, -1, axis=0))
    mass_v = 0.5 * (mass + np.roll(mass, -1, axis=1))
    area_u = 0.5 * (concentration + np.roll(concentration, -1, axis=0))
    area_v = 0.5 * (concentration + np.roll(concentration, -1, axis=1))
    low_u = (mass_u <= 1.0) & (area_u <= 0.001)
    low_v = (mass_v <= 1.0) & (area_v <= 0.001)
    halo = card.base.halo_width
    physical = (slice(halo, -halo), slice(halo, -halo))
    low_u_physical = low_u[physical]
    low_v_physical = low_v[physical]
    u = np.asarray(dynamics.u_ice_u)[physical]
    v = np.asarray(dynamics.v_ice_v)[physical]
    return {
        "low_mass_u_faces": int(np.count_nonzero(low_u_physical)),
        "low_mass_v_faces": int(np.count_nonzero(low_v_physical)),
        "max_abs_u_on_low_mass_faces": float(np.max(np.abs(u[low_u_physical]), initial=0.0)),
        "max_abs_v_on_low_mass_faces": float(np.max(np.abs(v[low_v_physical]), initial=0.0)),
        "exact_zero_on_all_low_mass_faces": bool(
            np.all(u[low_u_physical] == 0.0) and np.all(v[low_v_physical] == 0.0)
        ),
    }


def _oracle_threshold_census(
    entry: dict[str, np.ndarray], velocity_frame, card
) -> dict[str, object]:
    from legoesm.ice.dynamics import SI3CGridAEVPState
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    area = card.base.dx_m * card.base.dy_m
    mapping: list[np.ndarray] = []
    for name in ICE_ADV2D_TRACERS:
        value = rung33_gate._oracle_field(entry, name)
        mapping.append(np.pad(value * area, card.base.halo_width, mode="wrap"))
    contents = np.stack(mapping, axis=-1)
    zero = np.zeros_like(entry["u_ice"])
    dynamics = SI3CGridAEVPState(
        np.asarray(velocity_frame["u_ice"]),
        np.asarray(velocity_frame["v_ice"]),
        zero,
        zero,
        zero,
    )
    return _face_threshold_census(contents, dynamics, card)


def _shape_census(value: np.ndarray) -> dict[str, object]:
    east = np.roll(value, -1, axis=0)
    west = np.roll(value, 1, axis=0)
    north = np.roll(value, -1, axis=1)
    south = np.roll(value, 1, axis=1)
    local_max = (value > east) & (value > west) & (value > north) & (value > south)
    local_min = (value < east) & (value < west) & (value < north) & (value < south)
    return {
        "maximum": float(np.max(value)),
        "minimum": float(np.min(value)),
        "negative_cells": int(np.count_nonzero(value < 0.0)),
        "strict_four_neighbor_local_maxima": int(np.count_nonzero(local_max)),
        "strict_four_neighbor_local_minima": int(np.count_nonzero(local_min)),
    }


def _stress_velocity_arithmetic(card, completed, oracle_frame) -> dict[str, object]:
    """Propagate the measured stress residual through :495-510 and the denominator."""

    from legoesm.ice.dynamics import _si3_stress_divergence
    from legoesm.ice.fidelity import nemo_adv2d_rhg_testcase_recipe as recipe

    oracle_stress = tuple(
        np.asarray(oracle_frame[name]) for name in ("stress1_i", "stress2_i", "stress12_i")
    )
    stress_error = tuple(
        np.asarray(value) - reference
        for value, reference in zip(completed.dynamics[2:], oracle_stress, strict=True)
    )
    force_u, force_v = (
        np.asarray(value)
        for value in _si3_stress_divergence(
            *(jax.numpy.asarray(value) for value in stress_error), card.metrics
        )
    )
    forcing = recipe._forcing_for_state(card.forcing_template, card.initial_state, card.base)
    setup = replay_gate._setup(card, type(forcing)(*(np.asarray(value) for value in forcing)))
    physical = (slice(2, -2), slice(2, -2))
    result: dict[str, object] = {
        "source_stencil": "icedyn_rhg_evp.F90:495-510",
        "momentum_denominator": "icedyn_rhg_evp.F90:667-668,719-720",
        "stress_max_abs": [float(np.max(np.abs(value[physical]))) for value in stress_error],
        "observed_velocity_max_abs": [
            float(
                np.max(
                    np.abs(
                        np.asarray(completed.dynamics[index])[physical]
                        - np.asarray(oracle_frame[name])[physical]
                    )
                )
            )
            for index, name in ((0, "u_ice"), (1, "v_ice"))
        ],
    }
    for component, force, active, mass_over_dt in (
        ("u", force_u, setup["active_u"] * setup["mass_mask_u"], setup["mass_over_dt_u"]),
        ("v", force_v, setup["active_v"] * setup["mass_mask_v"], setup["mass_over_dt_v"]),
    ):
        # beta >= 50 by :446/:468.  Omitting positive drag makes this a
        # conservative upper bound on the velocity response to stress error.
        denominator_lower_bound = mass_over_dt * 51.0
        mask = active.astype(bool)
        response_bound = np.where(
            mask,
            np.abs(force) / np.maximum(1.0e-20, denominator_lower_bound),
            0.0,
        )
        result[component] = {
            "stress_divergence_error_max_abs": float(np.max(np.abs(force[physical]))),
            "max_abs_force_over_minimum_active_denominator": float(
                np.max(response_bound[physical])
            ),
            "note": "upper bound uses beta floor 50 and omits nonnegative ocean-drag denominator",
        }
    return result


def _restart_roundtrip_control(card, state, completed_steps: int) -> dict[str, object]:
    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
        load_ice_adv2d_rhg_restart,
        save_ice_adv2d_rhg_restart,
        step_ice_adv2d_rhg_card,
    )

    with tempfile.TemporaryDirectory(prefix="si3-rung33-restart-") as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv2d_rhg_restart(path, card, state, completed_steps=completed_steps)
        restored, clock = load_ice_adv2d_rhg_restart(path, card)
        require(clock == completed_steps, "rung-3.3 restart clock changed")
        equal = all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree_util.tree_leaves(state),
                jax.tree_util.tree_leaves(restored),
                strict=True,
            )
        )
        require(equal, "rung-3.3 restart roundtrip changed a carry")
        direct = step_ice_adv2d_rhg_card(card, state, completed_steps=completed_steps)
        resumed = step_ice_adv2d_rhg_card(card, restored, completed_steps=clock)
        continuation_equal = all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(
                jax.tree_util.tree_leaves(direct),
                jax.tree_util.tree_leaves(resumed),
                strict=True,
            )
        )
        require(continuation_equal, "rung-3.3 split continuation changed")
        with np.load(path, allow_pickle=False) as archive:
            original = {name: archive[name].copy() for name in archive.files}
        controls: dict[str, str] = {}
        for label, mutation in (
            ("missing_stress", lambda p: p.pop("stress1_i")),
            (
                "retyped_stress",
                lambda p: p.__setitem__("stress1_i", p["stress1_i"].astype(np.float32)),
            ),
            ("missing_moment", lambda p: p.pop("moment_4")),
            (
                "retyped_moment",
                lambda p: p.__setitem__("moment_4", p["moment_4"].astype(np.float32)),
            ),
        ):
            payload = {name: value.copy() for name, value in original.items()}
            mutation(payload)
            np.savez(path, **payload)  # type: ignore[arg-type]
            try:
                load_ice_adv2d_rhg_restart(path, card)
            except ValueError:
                controls[label] = "RED_AS_REQUIRED"
            else:
                raise TrajectoryGateError(f"{label} restart control did not go red")
        for label, key in (("perturbed_stress", "stress1_i"), ("perturbed_moment", "moment_4")):
            payload = {name: value.copy() for name, value in original.items()}
            payload[key].flat[0] += 1.0
            np.savez(path, **payload)  # type: ignore[arg-type]
            changed, _ = load_ice_adv2d_rhg_restart(path, card)
            changed_leaf = changed.dynamics.stress1_t if key == "stress1_i" else changed.moments[4]
            original_leaf = state.dynamics.stress1_t if key == "stress1_i" else state.moments[4]
            require(
                not np.array_equal(np.asarray(changed_leaf), np.asarray(original_leaf)),
                f"{label} comparison control did not bind",
            )
            controls[label] = "RED_AS_REQUIRED"
    return {
        "status": "VERIFIED",
        "roundtrip_byte_identical": equal,
        "split_continuation_byte_identical": continuation_equal,
        "ordinary_state_arrays": 8,
        "packed_moment_leaves": 5,
        "unpacked_moment_arrays": 80,
        "controls": controls,
    }


def run_gate(root: Path = ROOT) -> tuple[dict[str, object], int]:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
        build_ice_adv2d_rhg_card,
        step_ice_adv2d_rhg_card,
    )
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS
    from legoesm.ice.transport import SI3_PRATHER_MOMENT_NAMES

    require(jax.default_backend() == "cpu", "rung-3.3 trajectory requires CPU")
    set_policy(PrecisionPolicy.fp64())
    _, entry_frame = oracle_gate.read_frame(
        root / "oracle_ice_step_entry_kt00000001.bin"
    )
    card = build_ice_adv2d_rhg_card(
        rung33_gate.oracle_surface_temperature_c(root), entry_frame
    )
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "rung-3.3 policy is not fp64/scalar-libm",
    )
    frame_paths = sorted(root.glob("oracle_ice_step_entry_kt*.bin"))
    require(len(frame_paths) == card.base.n_steps, "oracle frame count changed")
    fields = (*DYNAMICS_FIELDS, *ICE_ADV2D_TRACERS, "sv_i", "t_surface")
    rows: list[dict[str, object]] = []
    growth: dict[str, dict[str, dict[str, object]]] = {}
    first_divergence: dict[str, object] | None = None
    maximum_series: dict[str, dict[str, list[float]]] = {
        "oracle": {"a_i": [], "v_i": []},
        "legoesm": {"a_i": [], "v_i": []},
    }
    shape_endpoints: dict[str, dict[str, dict[str, object]]] = {}
    threshold: dict[str, list[dict[str, object]]] = {"oracle": [], "legoesm": []}
    velocity_range: dict[str, dict[str, float]] = {
        "oracle": {"u_min": np.inf, "u_max": -np.inf, "v_min": np.inf, "v_max": -np.inf},
        "legoesm": {"u_min": np.inf, "u_max": -np.inf, "v_min": np.inf, "v_max": -np.inf},
    }
    state = card.initial_state
    split_state = None
    stress_velocity_arithmetic = None
    previous_frame = _frame(root, 1, card)
    odd_step = jax.jit(
        lambda current: step_ice_adv2d_rhg_card(
            card, current, completed_steps=0
        )
    )
    even_step = jax.jit(
        lambda current: step_ice_adv2d_rhg_card(
            card, current, completed_steps=1
        )
    )

    for completed in range(1, card.base.n_steps + 1):
        entry_state = state
        state = (odd_step if completed % 2 else even_step)(state)
        jax.block_until_ready(state)
        if completed == 7:
            split_state = state
        restart_boundary = completed == card.base.n_steps
        dataset: netCDF4.Dataset | None = None
        source: object
        next_frame: dict[str, np.ndarray] | None = None
        if restart_boundary:
            dataset = netCDF4.Dataset(oracle_gate.run_files(root)["restart"])
            source = dataset
        else:
            next_frame = _frame(root, completed + 1, card)
            source = next_frame
        boundary_rows: dict[str, dict[str, object]] = {}
        for name in fields:
            row = _detailed_score(
                f"trajectory.post_step_{completed:08d}.{name}",
                _oracle_boundary_field(source, name, restart=restart_boundary),
                _candidate_boundary_field(card, state, name),
            )
            rows.append(row)
            boundary_rows[name] = row
            if first_divergence is None and row["status"] == "DEBT":
                first_divergence = row
        if completed == 1:
            stress_velocity_arithmetic = _stress_velocity_arithmetic(card, state, source)
        if completed in GROWTH_STEPS:
            growth[str(completed)] = {
                name: boundary_rows[name]
                for name in (
                    "u_ice",
                    "v_ice",
                    "stress1_i",
                    "stress2_i",
                    "stress12_i",
                    "a_i",
                    "v_i",
                    "v_s",
                )
            }
        for model, getter in (
            ("oracle", lambda name: _oracle_boundary_field(source, name, restart=restart_boundary)),
            ("legoesm", lambda name: _candidate_boundary_field(card, state, name)),
        ):
            for name in ("a_i", "v_i"):
                maximum_series[model][name].append(float(np.max(getter(name))))
            u = getter("u_ice")
            v = getter("v_ice")
            velocity_range[model]["u_min"] = min(velocity_range[model]["u_min"], float(np.min(u)))
            velocity_range[model]["u_max"] = max(velocity_range[model]["u_max"], float(np.max(u)))
            velocity_range[model]["v_min"] = min(velocity_range[model]["v_min"], float(np.min(v)))
            velocity_range[model]["v_max"] = max(velocity_range[model]["v_max"], float(np.max(v)))
            if completed in (1, card.base.n_steps):
                shape_endpoints.setdefault(model, {})[str(completed)] = {
                    name: _shape_census(getter(name)) for name in ("a_i", "v_i")
                }
        threshold["legoesm"].append(
            _face_threshold_census(np.asarray(entry_state.contents), state.dynamics, card)
        )
        if restart_boundary:
            assert dataset is not None
            velocity_frame = {
                "u_ice": np.pad(
                    _restart_field(dataset, "u_ice"),
                    int(card.base.halo_width),
                    mode="wrap",
                ),
                "v_ice": np.pad(
                    _restart_field(dataset, "v_ice"),
                    int(card.base.halo_width),
                    mode="wrap",
                ),
            }
        else:
            assert next_frame is not None
            velocity_frame = next_frame
        threshold["oracle"].append(_oracle_threshold_census(previous_frame, velocity_frame, card))
        if dataset is not None:
            dataset.close()
        if not restart_boundary:
            assert next_frame is not None
            previous_frame = next_frame

    require(stress_velocity_arithmetic is not None, "stress arithmetic was not measured")
    restart_path = oracle_gate.run_files(root)["restart"]
    moment_rows: list[dict[str, object]] = []
    with netCDF4.Dataset(restart_path) as dataset:
        discovered = {name for name in dataset.variables if name.startswith(("sx", "sy"))}
        expected = {
            rung32_gate._moment_name(moment, tracer)
            for moment in SI3_PRATHER_MOMENT_NAMES
            for tracer in ICE_ADV2D_TRACERS
        }
        require(discovered == expected, "oracle restart moment discovery mismatch")
        for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
            for tracer_index, tracer in enumerate(ICE_ADV2D_TRACERS):
                restart_name = rung32_gate._moment_name(moment, tracer)
                row = _detailed_score(
                    f"restart_moment.{restart_name}",
                    rung32_gate._restart_field(dataset, restart_name),
                    np.asarray(state.moments[moment_index])[2:-2, 2:-2, tracer_index],
                )
                rows.append(row)
                moment_rows.append(row)
    ordinary_restart_rows = [
        row for row in rows if str(row["name"]).startswith("trajectory.post_step_00000485")
    ]
    require(len(ordinary_restart_rows) == len(fields), "ordinary restart row loss")
    require(len(moment_rows) == 80, "Prather restart moment row loss")
    require(split_state is not None, "split restart checkpoint was not captured")
    restart_control = _restart_roundtrip_control(card, split_state, 7)
    replay, _ = replay_gate.run_replay(root)
    require(
        all(
            row["exact_zero_on_all_low_mass_faces"]
            for values in threshold.values()
            for row in values
        ),
        "documented low-mass ocean-velocity switch failed",
    )
    require(
        all(
            values["u_max"] > 0.0
            or values["u_min"] < 0.0
            or values["v_max"] > 0.0
            or values["v_min"] < 0.0
            for values in velocity_range.values()
        ),
        "documented rheology-generated velocity was not observed",
    )
    overall_status = rung33_gate._status_from_rows(rows)
    report = {
        "format": "nemo-si3-phase2-rung33-trajectory-v1",
        "worktree": worktree_stamp(),
        "status": overall_status,
        "bar": POINTWISE_BAR,
        "backend": jax.default_backend(),
        "precision_policy": "fp64",
        "transcendentals": "libm",
        "execution_path": "whole-step production JIT",
        "dtypes": sorted(
            {str(np.asarray(leaf).dtype) for leaf in jax.tree_util.tree_leaves(state)}
        ),
        "root": str(root),
        "trajectory": {
            "completed_steps": card.base.n_steps,
            "oracle_entry_frames_consumed": card.base.n_steps,
            "first_divergence": first_divergence,
            "growth": growth,
            "rows": rows,
            "debt_rows": sum(row["status"] == "DEBT" for row in rows),
            "at_bar_rows": sum(row["status"] == "AT-BAR" for row in rows),
        },
        "stress_replay": replay,
        "stress_velocity_arithmetic": stress_velocity_arithmetic,
        "restart": {
            "oracle_sha256": _sha256(restart_path),
            "ordinary_prognostic_rows": len(ordinary_restart_rows),
            "stress_rows": [
                row
                for row in ordinary_restart_rows
                if str(row["name"]).endswith(("stress1_i", "stress2_i", "stress12_i"))
            ],
            "moment_rows": moment_rows,
            "roundtrip_control": restart_control,
            "coverage": {
                "snwice_mass": (
                    "UNMEASURED: derived mass diagnostic, not candidate prognostic state"
                ),
                "snwice_mass_b": (
                    "UNMEASURED: derived before-level diagnostic, not candidate prognostic state"
                ),
            },
        },
        "phenomenology": {
            "source": "tests/ICE_ADV2D/EXPREF/README:48-55,64-66",
            "maximum_series": maximum_series,
            "shape_endpoints": shape_endpoints,
            "velocity_range": velocity_range,
            "low_mass_switch": {
                model: {
                    "all_boundaries_exact_zero": all(
                        row["exact_zero_on_all_low_mass_faces"] for row in values
                    ),
                    "max_abs_u": max(
                        cast(float, row["max_abs_u_on_low_mass_faces"]) for row in values
                    ),
                    "max_abs_v": max(
                        cast(float, row["max_abs_v_on_low_mass_faces"]) for row in values
                    ),
                    "min_u_face_count": min(cast(int, row["low_mass_u_faces"]) for row in values),
                    "min_v_face_count": min(cast(int, row["low_mass_v_faces"]) for row in values),
                }
                for model, values in threshold.items()
            },
            "classification": {
                "maximum_and_side_lobes": (
                    "MEASURED-UNCLASSIFIED: README supplies no numerical tolerance/predicate"
                ),
                "rheology_velocity": (
                    "VERIFIED: each model velocity range contains nonzero values"
                ),
                "low_mass_switch": (
                    "VERIFIED: all source-predicate faces are exactly at the zero "
                    "ocean velocity in both models"
                ),
            },
        },
        "oracle_hashes": {
            "first_frame": _sha256(frame_paths[0]),
            "last_frame": _sha256(frame_paths[-1]),
            "restart": _sha256(restart_path),
        },
    }
    return report, 0 if overall_status == "AT-BAR" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    report, code = run_gate(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.artifact:
        args.artifact.write_text(payload + "\n")
    print(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

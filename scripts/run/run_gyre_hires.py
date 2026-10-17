#!/usr/bin/env python3
"""Run a resolved GYRE deck with daily snapshots and daily means."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86_400
FIELD_NAMES = (
    "ssh_m",
    "surface_u_m_s",
    "surface_v_m_s",
    "surface_temperature_c",
    "barotropic_u_m_s",
    "barotropic_v_m_s",
)
FULL_STATE_FIELD_NAMES = (
    "temperature_c_3d",
    "salinity_psu_3d",
    "u_m_s_3d",
    "v_m_s_3d",
    "tke_m2_s2_3d",
    "layer_thickness_m_3d",
)
SCAN_PARITY_ATOL = 1.0e-12


def validate_physical_gpu(mode: str, physical_gpu: int | None, environ=None) -> None:
    """Require an explicit, provenance-bearing physical GPU selection."""
    environ = os.environ if environ is None else environ
    if mode == "gpu-native":
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise ValueError("gpu-native mode requires --physical-gpu with a non-negative index")
        visible = environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(physical_gpu):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must equal the explicit --physical-gpu index "
                f"({physical_gpu}), got {visible!r}"
            )
    elif mode == "cpu-certified":
        if physical_gpu is not None:
            raise ValueError("cpu-certified mode does not accept --physical-gpu")
    else:
        raise ValueError("mode must be 'gpu-native' or 'cpu-certified'")


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_hashes_atomic(path: Path, hashes: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    temporary.replace(path)


def resolved_daily_namelist(deck: str, days: int):
    """Resolve a named certified deck for an exact whole-day run length."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        GYRE_R4_SMAG_NAMELIST,
        GYRE_SHIPPED_NAMELIST,
    )

    if deck not in ("shipped", "r4", "r4_smag"):
        raise ValueError("deck must be 'shipped', 'r4', or 'r4_smag'")
    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    base = {
        "shipped": GYRE_SHIPPED_NAMELIST,
        "r4": GYRE_R4_NAMELIST,
        "r4_smag": GYRE_R4_SMAG_NAMELIST,
    }[deck]
    steps_per_day = SECONDS_PER_DAY / base.rn_Dt_s
    if not float(steps_per_day).is_integer():
        raise ValueError(f"rn_Dt={base.rn_Dt_s} does not divide one model day")
    cadence = int(steps_per_day)
    return base._replace(nn_itend=days * cadence, nn_write=cadence)


def _state_arrays(state):
    """Return cheap native-grid climate fields from one model state."""
    if state.uu_b is None or state.vv_b is None:
        raise ValueError("GYRE output requires prognostic barotropic velocity")
    return (
        state.eta.data,
        state.u.data[:, 1:, 0],
        state.v.data[1:, :, 0],
        state.T.data[..., 0],
        state.uu_b.data[:, 1:],
        state.vv_b.data[1:, :],
    )


def _mean_accumulators(state, card):
    """IOIPSL-order samples; surface T is accumulated as T*e3t plus e3t."""
    from legoesm.ocean.vertical import compute_layer_thickness

    values = list(_state_arrays(state))
    h_surface = compute_layer_thickness(state.eta.data, state.H_bathy.data, card.recipe.z_coord)[
        ..., 0
    ]
    values[3] = values[3] * h_surface
    values.append(h_surface)
    return tuple(values)


def _finalize_means(accumulators):
    """Convert mean(T*e3t), mean(e3t) to the NEMO-comparable T mean."""
    import jax.numpy as jnp

    means = list(accumulators[:-1])
    thickness = accumulators[-1]
    means[3] = jnp.where(thickness > 0.0, means[3] / thickness, 0.0)
    return tuple(means)


def parse_day_window(value: str | None, *, days: int) -> tuple[int, int] | None:
    """Parse an inclusive ``START:END`` model-day window."""
    if value is None:
        return None
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("full-state window must be START:END") from exc
    if start < 1 or end < start or end > days:
        raise ValueError(
            f"full-state window must satisfy 1 <= START <= END <= {days}")
    return start, end


def validate_full_state_request(
    *, days: int, per_step: bool, window: tuple[int, int] | None,
) -> None:
    """Require observer-free per-step integration for full-state means."""
    if window is not None and not per_step:
        raise ValueError(
            "full-state daily means require --per-step so diagnostic "
            "accumulation cannot change the compiled-day scan")
    if per_step and days <= 0:
        raise ValueError("per-step integration requires a positive day count")


def _full_state_arrays(state, card):
    """Return native-grid 3-D prognostics needed by the momentum budget."""
    from legoesm.ocean.vertical import compute_layer_thickness

    if state.tke is None:
        raise ValueError("GYRE full-state output requires prognostic TKE")
    return (
        state.T.data,
        state.S.data,
        state.u.data[:, 1:, :],
        state.v.data[1:, :, :],
        state.tke.data,
        compute_layer_thickness(
            state.eta.data, state.H_bathy.data, card.recipe.z_coord),
    )


def _full_mean_accumulators(state, card):
    """Return IOIPSL-compatible 3-D mean samples plus cell thickness."""
    temperature, salinity, u, v, tke, thickness = _full_state_arrays(
        state, card)
    return (
        temperature * thickness,
        salinity * thickness,
        u,
        v,
        tke,
        thickness,
    )


def _finalize_full_means(accumulators):
    """Convert mean tracer contents to thickness-weighted concentrations."""
    import jax.numpy as jnp

    temperature_content, salinity_content, u, v, tke, thickness = accumulators
    safe = jnp.where(thickness > 0.0, thickness, 1.0)
    return (
        jnp.where(thickness > 0.0, temperature_content / safe, 0.0),
        jnp.where(thickness > 0.0, salinity_content / safe, 0.0),
        u,
        v,
        tke,
        thickness,
    )


def cold_start_scan_state(model, card, state):
    """Run NEMO's required cold-start step, then stabilize the scan carry.

    The AB3/AM4 external mode cannot be seeded with zero history without
    changing NEMO's ``ll_init`` ramp.  Its first step therefore runs through
    the same public production dispatch as certified CPU mode; every remaining
    step, including every complete later day, is scan-safe.
    """
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings

    freshwater, surface = gyre_surface_forcings(card, state, 1)
    state = model.step(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    freshwater, surface = gyre_surface_forcings(card, state, 2)
    state = model.seed_scan_carry(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    return state, _mean_accumulators(state, card)


def build_scanned_day_mean_stepper(model, card, n_steps: int):
    """Compile one day; evaluate seasonal forcing at every scanned step."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("daily scan n_steps must be a positive integer")

    def scan_body(carry, values):
        state, means = carry
        kt, count = values
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state = model.step(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        sample = _mean_accumulators(state, card)
        old_count = count - jnp.asarray(1, dtype=count.dtype)
        means = tuple((mean * old_count + value) / count for mean, value in zip(means, sample))
        return (state, means), None

    @jax.jit
    def compiled(state, first_kt, means, initial_count):
        steps = jnp.arange(n_steps, dtype=jnp.int64) + first_kt
        counts = jnp.arange(1, n_steps + 1, dtype=state.eta.data.dtype) + initial_count
        (state, means), _ = jax.lax.scan(scan_body, (state, means), (steps, counts))
        return state, _finalize_means(means)

    def scanned(state, first_kt, initial_means=None, initial_count=0):
        means = (
            tuple(jnp.zeros_like(value) for value in _mean_accumulators(state, card))
            if initial_means is None
            else initial_means
        )
        return compiled(
            state, first_kt, means, jnp.asarray(initial_count, dtype=state.eta.data.dtype)
        )

    return scanned


def run_python_day(model, card, state, first_kt: int, n_steps: int):
    """Run the certified per-step dispatch and accumulate IOIPSL-order means."""
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("Python day n_steps must be a positive integer")
    means = tuple(jnp.zeros_like(value) for value in _mean_accumulators(state, card))
    for offset in range(n_steps):
        freshwater, surface = gyre_surface_forcings(card, state, first_kt + offset)
        state = model.step(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        count = float(offset + 1)
        means = tuple(
            (mean * (count - 1.0) + value) / count
            for mean, value in zip(means, _mean_accumulators(state, card))
        )
    return state, _finalize_means(means)


def run_python_day_with_full_state_means(
    model, card, state, first_kt: int, n_steps: int,
):
    """Run ordinary compiled steps; accumulate matched 3-D means outside JIT."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("Python full-state day n_steps must be a positive integer")
    cheap = tuple(np.zeros_like(np.asarray(value))
                  for value in _mean_accumulators(state, card))
    # The prognostic TKE field is None until the first model.step seeds it,
    # so the full-state template is built from the first post-step sample.
    full = None
    for offset in range(n_steps):
        freshwater, surface = gyre_surface_forcings(
            card, state, first_kt + offset)
        state = model.step(
            state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        count = float(offset + 1)
        cheap = tuple(
            (mean * (count - 1.0) + np.asarray(value)) / count
            for mean, value in zip(
                cheap, _mean_accumulators(state, card), strict=True)
        )
        sample = tuple(
            np.asarray(value) for value in _full_mean_accumulators(state, card))
        if full is None:
            full = tuple(np.zeros_like(value) for value in sample)
        full = tuple(
            (mean * (count - 1.0) + value) / count
            for mean, value in zip(full, sample, strict=True)
        )
    return state, _finalize_means(cheap), _finalize_full_means(full)


def prognostic_differences(left, right) -> dict[str, float | int | bool]:
    """Compare all fields emitted by the climate driver."""
    result: dict[str, float | int | bool] = {}
    all_finite = True
    for name, a_raw, b_raw in zip(FIELD_NAMES, _state_arrays(left), _state_arrays(right)):
        a = np.asarray(a_raw, dtype=np.float64)
        b = np.asarray(b_raw, dtype=np.float64)
        finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        all_finite = all_finite and finite
        result[f"{name}_finite"] = finite
        result[f"{name}_max_abs"] = float(np.max(np.abs(a - b)))
        result[f"{name}_unequal"] = int(np.count_nonzero(a != b))
    result["all_finite"] = all_finite
    result["all_bit_equal"] = bool(all(result[f"{name}_unequal"] == 0 for name in FIELD_NAMES))
    result["atol"] = SCAN_PARITY_ATOL
    result["within_atol"] = bool(
        all_finite and all(result[f"{name}_max_abs"] <= SCAN_PARITY_ATOL for name in FIELD_NAMES)
    )
    return result


def _forcing_differences(card, state, kt: int) -> dict[str, object]:
    """Test Python-step versus traced-step seasonal forcing bit identity."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_seasonal_cosines
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings

    def clock_and_phase(step):
        clock = nemo_source_round(
            jnp.asarray(step, dtype=jnp.float64) * jnp.asarray(card.dt_s, dtype=jnp.float64)
        )
        return (clock, *nemo_gyre_seasonal_cosines(clock))

    eager_clock = clock_and_phase(kt)
    traced_clock = jax.jit(clock_and_phase)(jnp.asarray(kt, dtype=jnp.int64))
    clock_rows = []
    for name, a_raw, b_raw in zip(
        ("clock_seconds", "seasonal_cosine_1", "seasonal_cosine_2"), eager_clock, traced_clock
    ):
        a = np.asarray(a_raw, dtype=np.float64)
        b = np.asarray(b_raw, dtype=np.float64)
        clock_rows.append(
            {
                "name": name,
                "n_unequal": int(np.count_nonzero(a != b)),
                "max_abs": float(np.max(np.abs(a - b))),
            }
        )

    eager = gyre_surface_forcings(card, state, kt)

    @jax.jit
    def traced(step):
        return gyre_surface_forcings(card, state, step)

    compiled = traced(jnp.asarray(kt, dtype=jnp.int64))
    eager_paths = jax.tree_util.tree_flatten_with_path(eager)[0]
    compiled_paths = jax.tree_util.tree_flatten_with_path(compiled)[0]
    rows = []
    for index, ((path, a_raw), (_, b_raw)) in enumerate(zip(eager_paths, compiled_paths)):
        a = np.asarray(a_raw, dtype=np.float64)
        b = np.asarray(b_raw, dtype=np.float64)
        rows.append(
            {
                "leaf": index,
                "path": jax.tree_util.keystr(path),
                "shape": list(a.shape),
                "n_unequal": int(np.count_nonzero(a != b)),
                "max_abs": float(np.max(np.abs(a - b))),
            }
        )
    return {
        "step": kt,
        "clock_and_phase_bit_equal": all(row["n_unequal"] == 0 for row in clock_rows),
        "clock_and_phase": clock_rows,
        "all_bit_equal": all(row["n_unequal"] == 0 for row in rows),
        "leaves": rows,
    }


def cpu_scan_parity(namelist, *, days: int = 2, commit: str | None = None) -> dict:
    """Compare a real GYRE compiled-day scan with the certified Python loop."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_gyre_zco_card

    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU scan parity requires the CPU backend")
    if type(days) is not int or days <= 0:
        raise ValueError("parity days must be a positive integer")
    if commit is not None and (
        len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_gyre_zco_card(namelist)
    steps_per_day = int(SECONDS_PER_DAY / card.dt_s)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    initial, _ = cold_start_scan_state(model, card, card.recipe.initial_state)
    forcing_check = _forcing_differences(card, initial, 2)
    prefix_comparisons = []
    for prefix_steps in (1, 2):
        prefix_loop, _ = run_python_day(model, card, initial, 2, prefix_steps)
        prefix_scan_step = build_scanned_day_mean_stepper(model, card, prefix_steps)
        prefix_scan, _ = prefix_scan_step(initial, 2)
        jax.block_until_ready(prefix_scan.eta.data)
        prefix_differences = prognostic_differences(prefix_loop, prefix_scan)
        prefix_fields = [name for name in FIELD_NAMES if prefix_differences[f"{name}_unequal"] > 0]
        prefix_comparisons.append(
            {
                "model_step": 1 + prefix_steps,
                "scan_steps": prefix_steps,
                "first_field": prefix_fields[0] if prefix_fields else None,
                "fields": prefix_fields,
                "differences": prefix_differences,
            }
        )
    differing_prefixes = [row for row in prefix_comparisons if row["fields"]]
    first_prefix = differing_prefixes[0] if differing_prefixes else None
    loop_state = initial
    loop_state, _ = run_python_day(model, card, loop_state, 2, steps_per_day - 1)
    for day in range(1, days):
        loop_state, _ = run_python_day(
            model, card, loop_state, day * steps_per_day + 1, steps_per_day
        )
    cold_tail = build_scanned_day_mean_stepper(model, card, steps_per_day - 1)
    scanned = build_scanned_day_mean_stepper(model, card, steps_per_day)
    scan_state, _ = cold_tail(initial, 2)
    for day in range(1, days):
        scan_state, _ = scanned(scan_state, day * steps_per_day + 1)
    jax.block_until_ready(scan_state.eta.data)
    differences = prognostic_differences(loop_state, scan_state)
    classification = (
        "TRANSCRIPTION"
        if not forcing_check["clock_and_phase_bit_equal"]
        else "COMPOSITION"
        if (first_prefix is not None or not forcing_check["all_bit_equal"])
        else "UNRESOLVED"
    )
    return {
        "schema": "legoesm-gyre-hires-scan-parity-v1",
        "legoesm_git_sha": commit,
        "backend": jax.default_backend(),
        "precision_policy": "fp64-libm-certified",
        "resolved_namelist": namelist._asdict(),
        "days": days,
        "steps": days * steps_per_day,
        "common_cold_start_steps": 1,
        "forcing_transcription": forcing_check,
        "prefix_comparisons": prefix_comparisons,
        "first_difference": {
            "step": first_prefix["model_step"] if first_prefix else None,
            "first_field": first_prefix["first_field"] if first_prefix else None,
            "fields": first_prefix["fields"] if first_prefix else [],
            "differences": first_prefix["differences"] if first_prefix else None,
        },
        "classification": classification,
        "disposition": (
            "measured composition debt; climate long run may proceed"
            if classification == "COMPOSITION"
            else "fix forcing transcription before the climate long run"
            if classification == "TRANSCRIPTION"
            else "first differing step not isolated"
        ),
        "differences": differences,
    }


def run_long(
    namelist, output: Path, *, days: int, mode: str, commit: str,
    physical_gpu: int | None = None, per_step: bool = False,
    full_state_window: tuple[int, int] | None = None,
) -> dict:
    """Integrate a resolved GYRE deck and atomically write each model day."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        validate_gyre_namelist,
    )

    validate_gyre_namelist(namelist)
    validate_physical_gpu(mode, physical_gpu)
    validate_full_state_request(
        days=days, per_step=per_step, window=full_state_window)
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    steps_per_day_f = SECONDS_PER_DAY / namelist.rn_Dt_s
    if not float(steps_per_day_f).is_integer():
        raise ValueError("resolved rn_Dt must divide one model day")
    steps_per_day = int(steps_per_day_f)
    if namelist.nn_itend != days * steps_per_day:
        raise ValueError("resolved nn_itend does not equal days times steps/day")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output}")

    if mode == "gpu-native":
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        certified_policy = False
        expected_backend = "gpu"
    else:
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        certified_policy = True
        expected_backend = "cpu"
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX x64 is disabled")
    if jax.default_backend() != expected_backend:
        raise RuntimeError(
            f"{mode} requires JAX backend {expected_backend}, got {jax.default_backend()}"
        )

    card = build_gyre_zco_card(namelist)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    cold_tail = None
    cold_sample = None
    if mode == "gpu-native" and not per_step:
        state, cold_sample = cold_start_scan_state(model, card, state)
        cold_tail = build_scanned_day_mean_stepper(model, card, steps_per_day - 1)
        scanned_day = build_scanned_day_mean_stepper(model, card, steps_per_day)
    else:
        scanned_day = None
    output.mkdir(parents=True)
    manifest = {
        "schema": "legoesm-gyre-hires-daily-v1",
        "legoesm_git_sha": commit,
        "mode": mode,
        "certified_policy": certified_policy,
        "certification_scope": (
            "CPU per-step ladder eligible for certification"
            if certified_policy
            else "NON-CERTIFIED climate run; native GPU transcendentals"
        ),
        "backend": jax.default_backend(),
        "physical_gpu_index": physical_gpu,
        "precision_policy": "fp64",
        "transcendentals": "native" if mode == "gpu-native" else "libm",
        "days": days,
        "steps_per_day": steps_per_day,
        "per_step": per_step,
        "integration": (
            "one required eager AB3/AM4 cold-start step, then compiled "
            "lax.scan days; seasonal forcing inside scan"
            if scanned_day is not None
            else "ordinary compiled model.step in a per-step Python loop"
        ),
        "sampling": ["end_of_day_snapshot", "daily_mean"],
        "namelist": namelist._asdict(),
        "fields": {
            "ssh_m": "T-point sea-surface height",
            "surface_u_m_s": "surface native U velocity",
            "surface_v_m_s": "surface native V velocity",
            "surface_temperature_c": "surface conservative temperature",
            "barotropic_u_m_s": "prognostic native U depth-mean velocity",
            "barotropic_v_m_s": "prognostic native V depth-mean velocity",
        },
        "daily_mean_convention": {
            "surface_temperature_c": (
                "thickness weighted: mean(T*e3t)/mean(e3t), matching NEMO IOIPSL votemper/vovvle3t"
            ),
            "temperature_c_3d_and_salinity_psu_3d": (
                "thickness weighted: mean(tracer*e3t)/mean(e3t), matching "
                "NEMO IOIPSL votemper/vosaline divided by vovvle3t"
            ),
            "other_fields": "arithmetic mean of per-step end states",
        },
        "full_state_window": (
            None if full_state_window is None else list(full_state_window)),
        "full_state_fields": (
            list(FULL_STATE_FIELD_NAMES)
            if full_state_window is not None else None
        ),
        "full_state_observer_contract": (
            "per-step model.step is compiled independently; 3-D mean "
            "accumulation is host-side and cannot alter the step graph"
            if full_state_window is not None else None
        ),
    }
    pending = output / "manifest.pending.json"
    hashes: dict[str, str] = {}
    _write_json_atomic(
        pending, manifest | {"status": "pending", "completed_days": 0, "daily_sha256": hashes}
    )

    for day in range(1, days + 1):
        first_kt = (day - 1) * steps_per_day + 1
        full_means = None
        if (full_state_window is not None
                and full_state_window[0] <= day <= full_state_window[1]):
            state, means, full_means = run_python_day_with_full_state_means(
                model, card, state, first_kt, steps_per_day)
        elif scanned_day is None:
            state, means = run_python_day(
                model, card, state, first_kt, steps_per_day)
        elif day == 1:
            assert cold_tail is not None and cold_sample is not None
            state, means = cold_tail(state, 2, initial_means=cold_sample, initial_count=1)
        else:
            state, means = scanned_day(state, first_kt)
        jax.block_until_ready(state.eta.data)
        arrays = {
            "schema": np.str_(manifest["schema"]),
            "legoesm_git_sha": np.str_(commit),
            "mode": np.str_(mode),
            "certified_policy": np.bool_(certified_policy),
            "physical_gpu_index": np.int64(-1 if physical_gpu is None else physical_gpu),
            "day": np.int64(day),
            "step": np.int64(day * steps_per_day),
            "dt_s": np.float64(card.dt_s),
            "daily_mean_sample_count": np.int64(steps_per_day),
        }
        for name, snapshot, mean in zip(FIELD_NAMES, _state_arrays(state), means):
            arrays[name] = np.asarray(snapshot, dtype=np.float64)
            arrays[f"{name}_daily_mean"] = np.asarray(mean, dtype=np.float64)
        if full_means is not None:
            for name, snapshot, mean in zip(
                FULL_STATE_FIELD_NAMES, _full_state_arrays(state, card),
                full_means, strict=True,
            ):
                arrays[name] = np.asarray(snapshot, dtype=np.float64)
                arrays[f"{name}_daily_mean"] = np.asarray(
                    mean, dtype=np.float64)
        name = f"day{day:04d}.npz"
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(output / name)
        hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
        _write_hashes_atomic(output / "daily.sha256", hashes)
        _write_json_atomic(
            pending,
            manifest | {"status": "pending", "completed_days": day, "daily_sha256": hashes},
        )
    completed = manifest | {
        "status": "complete",
        "completed_days": days,
        "daily_sha256": hashes,
    }
    _write_json_atomic(pending, completed)
    pending.replace(output / "manifest.json")
    return completed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=("shipped", "r4", "r4_smag"), required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("gpu-native", "cpu-certified"), required=True)
    parser.add_argument(
        "--physical-gpu", type=int,
        help="required physical device index for gpu-native mode; forbidden for CPU mode",
    )
    parser.add_argument(
        "--daily-means",
        action="store_true",
        help="required acknowledgement: emit daily means beside snapshots",
    )
    parser.add_argument(
        "--per-step", action="store_true",
        help="run ordinary compiled model.step calls outside a day scan",
    )
    parser.add_argument(
        "--full-state-window",
        help="inclusive START:END days for 3-D snapshots and daily means",
    )
    args = parser.parse_args(argv)
    if not args.daily_means:
        parser.error("--daily-means is required for this climate driver")
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        namelist = resolved_daily_namelist(args.deck, args.days)
        full_state_window = parse_day_window(
            args.full_state_window, days=args.days)
        manifest = run_long(
            namelist, args.output, days=args.days, mode=args.mode,
            commit=git_sha(), physical_gpu=args.physical_gpu,
            per_step=args.per_step, full_state_window=full_state_window,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        f"COMPLETED days={manifest['days']} backend={manifest['backend']} "
        f"certified_policy={manifest['certified_policy']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

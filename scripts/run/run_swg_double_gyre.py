#!/usr/bin/env python3
"""Run the parameterised SWG double gyre from rest with daily NPZ output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24
SCAN_PARITY_ATOL = 1.0e-15
GPU_TARGET_MS_PER_STEP = 5.0
METADATA_CHECKPOINT_DAYS = 25


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_hashes_atomic(path: Path, hashes: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    temporary.replace(path)


def resolved_daily_namelist(days: int, *, deck: str = "maddison",
                            surface: str | None = None):
    """Return a supported double-gyre deck resolved for a number of days.

    A rigid-lid-only deck (``*-rl``) requires ``surface="rigid_lid"``.
    """
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_DG50_RL_NAMELIST,
        SWG_MADDISON_NAMELIST,
        SWG_MADDISON_RL_NAMELIST,
    )
    from legoesm.ocean.fidelity.swg_online_decks import (
        SWG_DG50_NU80_NAMELIST,
        SWG_DG50_NU80_RL_NAMELIST,
        SWG_DG3125_NAMELIST,
        require_surface_for_deck,
    )

    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    decks = {
        "maddison": SWG_MADDISON_NAMELIST,
        "dg50": SWG_DG50_NAMELIST,
        "dg50-nu80": SWG_DG50_NU80_NAMELIST,
        "dg3125": SWG_DG3125_NAMELIST,
        "maddison-rl": SWG_MADDISON_RL_NAMELIST,
        "dg50-rl": SWG_DG50_RL_NAMELIST,
        "dg50-nu80-rl": SWG_DG50_NU80_RL_NAMELIST,
    }
    if deck not in decks:
        raise ValueError(f"unsupported double-gyre deck {deck!r}")
    require_surface_for_deck(deck, surface)
    selected = decks[deck]
    dt_s = selected.rn_Dt_s
    steps_per_day = SECONDS_PER_DAY / dt_s
    if not float(steps_per_day).is_integer():
        raise ValueError(f"rn_Dt={dt_s} does not divide one model day")
    cadence = int(steps_per_day)
    return selected._replace(
        nn_itend=days * cadence, nn_write=cadence, nn_stock=cadence)


def restart_state_from_npz(state, forcing, path: Path, *, dt_s: float,
                           perturb_amplitude_m: float | None = None,
                           perturb_seed: int = 0):
    """Restore native SWG prognostics and optional fixed-seed wet SSH noise."""
    import jax.numpy as jnp

    if perturb_amplitude_m is not None and (
            not np.isfinite(perturb_amplitude_m) or perturb_amplitude_m < 0.0):
        raise ValueError("perturb amplitude must be finite and non-negative")
    if type(perturb_seed) is not int or perturb_seed < 0:
        raise ValueError("perturb seed must be a non-negative integer")
    with np.load(path, allow_pickle=False) as saved:
        required = {"ssh_m", "u_m_s", "v_m_s", "day", "step", "dt_s"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"restart is missing keys {sorted(missing)}")
        if float(saved["dt_s"]) != float(dt_s):
            raise ValueError("restart timestep differs from resolved deck")
        ssh = np.asarray(saved["ssh_m"], dtype=np.float64).copy()
        u_native = np.asarray(saved["u_m_s"], dtype=np.float64)
        v_native = np.asarray(saved["v_m_s"], dtype=np.float64)
        source_day = int(saved["day"])
        source_step = int(saved["step"])
    expected = tuple(state.eta.data.shape)
    if ssh.shape != expected or u_native.shape != expected or v_native.shape != expected:
        raise ValueError(
            f"restart native shapes must all be {expected}, got "
            f"{ssh.shape}, {u_native.shape}, {v_native.shape}")
    if not all(np.all(np.isfinite(value)) for value in (ssh, u_native, v_native)):
        raise ValueError("restart contains non-finite prognostics")
    wet = np.asarray(state.land_mask.data) > 0.5
    if perturb_amplitude_m is not None and perturb_amplitude_m > 0.0:
        noise = np.random.default_rng(perturb_seed).standard_normal(ssh.shape)
        ssh[wet] += perturb_amplitude_m * noise[wet]
    u = np.zeros(state.u.data.shape, dtype=np.float64)
    v = np.zeros(state.v.data.shape, dtype=np.float64)
    u[:, 1:, 0] = u_native
    v[1:, :, 0] = v_native
    restored = state._replace(
        eta=state.eta.replace(data=jnp.asarray(ssh)),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        tau_x_prev=jnp.asarray(forcing.tau_i_native, dtype=state.eta.data.dtype),
        tau_y_prev=jnp.asarray(forcing.tau_j_native, dtype=state.eta.data.dtype),
    )
    return restored, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_day": source_day,
        "source_step": source_step,
        "perturb_amplitude_m": (
            None if perturb_amplitude_m is None else float(perturb_amplitude_m)),
        "perturb_seed": perturb_seed if perturb_amplitude_m is not None else None,
        "perturb_distribution": (
            "numpy.default_rng.standard_normal on wet T points"
            if perturb_amplitude_m is not None else None),
    }


def admit_listed_snapshot(path: Path) -> dict:
    """Refuse a trajectory file not content-pinned by its run manifest."""
    candidates = (path.parent / "manifest.json",
                  path.parent / "manifest.pending.json")
    manifest_path = next((item for item in candidates if item.is_file()), None)
    if manifest_path is None:
        raise ValueError(f"snapshot has no run manifest: {path}")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("daily_sha256", {}).get(path.name)
    if expected is None:
        raise ValueError(f"snapshot is not listed by manifest: {path.name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"snapshot hash differs from manifest: {actual} != {expected}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        "snapshot_sha256": actual,
        "manifest_status": manifest.get("status"),
    }


# Rigid-lid-only restart check (user decision 2026-10-18): the hourly-target
# program differs from a plain restart by round-off (max |du| 2.6e-15 m/s);
# the bitwise cause is deferred.  Explicit runs stay strictly bitwise.
LID_RESTART_TOLERANCE = {
    "kind": "max_abs_relative_to_field_max", "value": 1.0e-12,
    "applies_to": "rigid_lid"}


def compare_native_state_bitwise(
        state, path: Path, relative_tolerance: float | None = None) -> dict:
    """Compare a stepped fine state with an admitted uninterrupted snapshot.

    ``all_equal`` is always the bitwise flag.  ``within_tolerance`` equals it
    unless ``relative_tolerance`` is given, in which case every field needs
    max|actual - expected| <= relative_tolerance * max|expected|.
    """
    with np.load(path, allow_pickle=False) as saved:
        required = {"ssh_m", "u_m_s", "v_m_s"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"comparison snapshot is missing {sorted(missing)}")
        expected = {name: np.asarray(saved[key]) for name, key in (
            ("ssh_m", "ssh_m"), ("u_m_s", "u_m_s"),
            ("v_m_s", "v_m_s"))}
    actual = {
        "ssh_m": np.asarray(state.eta.data),
        "u_m_s": np.asarray(state.u.data[:, 1:, 0]),
        "v_m_s": np.asarray(state.v.data[1:, :, 0]),
    }
    result = {"all_equal": True, "within_tolerance": True}
    for name in actual:
        if actual[name].shape != expected[name].shape:
            raise ValueError(f"comparison shape differs for {name}")
        if actual[name].dtype != np.dtype("float64") \
                or expected[name].dtype != np.dtype("float64"):
            raise ValueError(f"bitwise comparison requires float64 {name}")
        unequal = int(np.count_nonzero(
            actual[name].view(np.uint64) != expected[name].view(np.uint64)))
        result[f"{name}_unequal"] = unequal
        result[f"{name}_max_abs"] = float(
            np.max(np.abs(actual[name] - expected[name])))
        result["all_equal"] = bool(result["all_equal"] and unequal == 0)
        scale = float(np.max(np.abs(expected[name])))
        max_abs = result[f"{name}_max_abs"]
        result[f"{name}_relative_max_abs"] = (
            max_abs / scale if scale > 0.0
            else 0.0 if max_abs == 0.0 else float("inf"))
        ok = (unequal == 0 if relative_tolerance is None
              else max_abs <= relative_tolerance * scale)
        result["within_tolerance"] = bool(result["within_tolerance"] and ok)
    return result


def seed_steady_forcing_carry(state, forcing):
    """Seed the fixed-stress history so a scan carry has one pytree shape."""
    import jax.numpy as jnp

    dtype = state.eta.data.dtype
    tau_u = jnp.asarray(forcing.tau_i_native, dtype=dtype)
    tau_v = jnp.asarray(forcing.tau_j_native, dtype=dtype)
    if state.tau_x_prev is not None or state.tau_y_prev is not None:
        raise ValueError("steady-forcing scan seed requires an unstepped state")
    return state._replace(tau_x_prev=tau_u, tau_y_prev=tau_v)


def build_scanned_stepper(model, dt_s: float, forcing, n_steps: int):
    """Return one JIT program for ``n_steps`` production SWE steps."""
    import jax

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("scan n_steps must be a positive integer")

    def scan_body(state, _):
        return model.step(state, dt_s, surface_forcing=forcing), None

    @jax.jit
    def scanned(state):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_steps)
        return state

    return scanned


def build_scanned_day_mean_stepper(model, dt_s: float, forcing, n_steps: int,
                                   transport_diagnostic=None):
    """Return one JIT program for exact state and live-transport means."""
    import jax
    import jax.numpy as jnp

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("daily-mean scan n_steps must be a positive integer")

    def scan_body(carry, count):
        state, mean_ssh, mean_u, mean_v, mean_qu, mean_qv, mean_phi = carry
        state = model.step(state, dt_s, surface_forcing=forcing)
        ssh = state.eta.data
        u = state.u.data[:, 1:, 0]
        v = state.v.data[1:, :, 0]
        # NEMO's executed dia_wri call sees the just-computed state, and
        # IOIPSL moycum uses (old_mean * old_count + sample) / new_count.
        old_count = count - jnp.asarray(1, dtype=count.dtype)
        mean_ssh = (mean_ssh * old_count + ssh) / count
        mean_u = (mean_u * old_count + u) / count
        mean_v = (mean_v * old_count + v) / count
        if transport_diagnostic is not None:
            qu, qv = transport_diagnostic(ssh, u, v)
            mean_qu = (mean_qu * old_count + qu) / count
            mean_qv = (mean_qv * old_count + qv) / count
        if state.rigid_lid_phi is not None:
            mean_phi = (mean_phi * old_count + state.rigid_lid_phi) / count
        return (state, mean_ssh, mean_u, mean_v, mean_qu, mean_qv, mean_phi), None

    @jax.jit
    def scanned(state):
        means = (
            jnp.zeros_like(state.eta.data),
            jnp.zeros_like(state.u.data[:, 1:, 0]),
            jnp.zeros_like(state.v.data[1:, :, 0]),
            jnp.zeros_like(state.u.data[:, 1:, 0]),
            jnp.zeros_like(state.v.data[1:, :, 0]),
            (jnp.zeros(()) if state.rigid_lid_phi is None
             else jnp.zeros_like(state.rigid_lid_phi)),
        )
        counts = jnp.arange(1, n_steps + 1, dtype=state.eta.data.dtype)
        (state, mean_ssh, mean_u, mean_v, mean_qu, mean_qv, mean_phi), _ = (
            jax.lax.scan(scan_body, (state, *means), counts))
        result = (mean_ssh, mean_u, mean_v)
        if transport_diagnostic is not None:
            result += (mean_qu, mean_qv)
        if state.rigid_lid_phi is not None:
            result += (mean_phi,)
        return state, result

    return scanned


def zero_ssh_restrictor(restrictor):
    """A restrictor whose sea level is exactly zero (rigid-lid targets).

    The fine lid ssh is identically zero, but the restrictors form sea level
    as volume/area minus the reference depth, which leaves 1e-13 m round-off
    (measured: box 1.8e-14, filtered 2.1e-13 m); that is not a sea level.
    """
    import jax.numpy as jnp

    def restrict(eta, u_native, v_native):
        ssh, u, v = restrictor(eta, u_native, v_native)
        return jnp.zeros_like(ssh), u, v

    return restrict


def build_dual_coarse_sampler(restrictors):
    """Build the dual-operator sampler used inside every retained hour."""
    if set(restrictors) != {"box", "filtered"}:
        raise ValueError("hourly sampler requires box and filtered restrictors")

    def sample(eta, u_native, v_native):
        arguments = (eta, u_native, v_native)
        return tuple(restrictors[name](*arguments)
                     for name in ("box", "filtered"))

    return sample


def build_scanned_hourly_target_stepper(
        model, dt_s: float, forcing, steps_per_hour: int, restrictors):
    """Compile one day while retaining two restricted hourly state streams."""
    import jax

    if type(steps_per_hour) is not int or steps_per_hour <= 0:
        raise ValueError("hourly target steps must be a positive integer")
    sample = build_dual_coarse_sampler(restrictors)

    def step_body(state, _):
        return model.step(state, dt_s, surface_forcing=forcing), None

    def hour_body(state, _):
        state, _ = jax.lax.scan(
            step_body, state, None, length=steps_per_hour)
        coarse = sample(
            state.eta.data, state.u.data[:, 1:, 0], state.v.data[1:, :, 0])
        return state, coarse

    @jax.jit
    def scanned(state):
        return jax.lax.scan(hour_body, state, None, length=HOURS_PER_DAY)

    return scanned


def assert_hourly_stepper_has_no_convolution(hourly_stepper, state) -> None:
    """Guard the complete stepped-and-sampled production lowering."""
    from legoesm.training.swg_online import assert_no_convolution_operations

    assert_no_convolution_operations(
        hourly_stepper.lower(state), label="production hourly target scan")


def run_python_steps(model, state, dt_s: float, forcing, n_steps: int):
    """Run the retained certified per-step production dispatch."""
    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("Python-loop n_steps must be a positive integer")
    for _ in range(n_steps):
        state = model.step(state, dt_s, surface_forcing=forcing)
    return state


def prognostic_differences(left, right) -> dict[str, float | int | bool]:
    """Compare native SWE prognostics after synchronising the two programs."""
    arrays = {
        "ssh_m": (left.eta.data, right.eta.data),
        "u_m_s": (left.u.data[:, 1:, 0], right.u.data[:, 1:, 0]),
        "v_m_s": (left.v.data[1:, :, 0], right.v.data[1:, :, 0]),
    }
    result: dict[str, float | int | bool] = {}
    all_finite = True
    for name, (first, second) in arrays.items():
        a = np.asarray(first, dtype=np.float64)
        b = np.asarray(second, dtype=np.float64)
        finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        all_finite = all_finite and finite
        result[f"{name}_finite"] = finite
        result[f"{name}_max_abs"] = float(np.max(np.abs(a - b)))
        result[f"{name}_unequal"] = int(np.count_nonzero(a != b))
    result["all_finite"] = all_finite
    result["atol"] = SCAN_PARITY_ATOL
    result["within_atol"] = bool(
        all_finite
        and all(result[f"{name}_max_abs"] <= SCAN_PARITY_ATOL
                for name in arrays))
    return result


def profile_step_modes(model, state, dt_s: float, forcing, n_steps: int) -> dict:
    """Time warm Python and scanned bursts from one fixed-stress state."""
    import jax

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("profile n_steps must be a positive integer")
    model.prime_step_caches(state)

    warm_start = time.perf_counter()
    warmed = model.step(state, dt_s, surface_forcing=forcing)
    jax.block_until_ready(warmed.eta.data)
    python_compile_s = time.perf_counter() - warm_start

    scanned = build_scanned_stepper(model, dt_s, forcing, n_steps)
    warm_start = time.perf_counter()
    warmed = scanned(state)
    jax.block_until_ready(warmed.eta.data)
    scan_compile_s = time.perf_counter() - warm_start

    started = time.perf_counter()
    python_state = run_python_steps(model, state, dt_s, forcing, n_steps)
    jax.block_until_ready(python_state.eta.data)
    python_s = time.perf_counter() - started

    started = time.perf_counter()
    scan_state = scanned(state)
    jax.block_until_ready(scan_state.eta.data)
    scan_s = time.perf_counter() - started
    differences = prognostic_differences(python_state, scan_state)
    python_ms = 1_000.0 * python_s / n_steps
    scan_ms = 1_000.0 * scan_s / n_steps
    return {
        "steps": n_steps,
        "python_compile_s": python_compile_s,
        "scan_compile_s": scan_compile_s,
        "python_wall_s": python_s,
        "scan_wall_s": scan_s,
        "python_ms_per_step": python_ms,
        "scan_ms_per_step": scan_ms,
        "wall_delta_ms_per_step": python_ms - scan_ms,
        "speedup": python_s / scan_s,
        "target_ms_per_step": GPU_TARGET_MS_PER_STEP,
        "target_met": scan_ms < GPU_TARGET_MS_PER_STEP,
        "differences": differences,
    }


def profile_python_steps(model, state, dt_s: float, forcing, n_steps: int) -> dict:
    """Time a warm certified production-step burst on the CPU."""
    import jax

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("profile n_steps must be a positive integer")
    model.prime_step_caches(state)
    started = time.perf_counter()
    warmed = model.step(state, dt_s, surface_forcing=forcing)
    jax.block_until_ready(warmed.eta.data)
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    profiled = run_python_steps(model, state, dt_s, forcing, n_steps)
    jax.block_until_ready(profiled.eta.data)
    wall_s = time.perf_counter() - started
    return {
        "program": "production-python-step-loop",
        "steps": n_steps,
        "python_compile_s": compile_s,
        "python_wall_s": wall_s,
        "python_ms_per_step": 1_000.0 * wall_s / n_steps,
    }


def numerical_health(state, namelist) -> dict[str, float | bool]:
    """Daily finite/thickness/Courant checks on native prognostics."""
    eta = np.asarray(state.eta.data, dtype=np.float64)
    u = np.asarray(state.u.data[:, 1:, 0], dtype=np.float64)
    v = np.asarray(state.v.data[1:, :, 0], dtype=np.float64)
    wet = np.asarray(state.land_mask.data) > 0.5
    u_wet = np.asarray(state.u_mask.data[:, 1:]) > 0.5
    v_wet = np.asarray(state.v_mask.data[1:, :]) > 0.5
    bathymetry = np.asarray(state.H_bathy.data, dtype=np.float64)
    spacing_m = namelist.rn_dx_m / namelist.nn_SWG
    return {
        "all_finite": bool(all(np.all(np.isfinite(field))
                               for field in (eta, u, v))),
        "minimum_total_layer_thickness_m": float(
            np.min((bathymetry + eta)[wet])),
        "maximum_u_courant": float(
            np.max(np.abs(u[u_wet])) * namelist.rn_Dt_s / spacing_m),
        "maximum_v_courant": float(
            np.max(np.abs(v[v_wet])) * namelist.rn_Dt_s / spacing_m),
    }


def balanced_vorticity_direction(state, namelist):
    """Build a unit-RMS-vorticity, zero-transport-divergence perturbation."""
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import relative_vorticity_cell_centre
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        recover_velocity_from_streamfunction,
    )
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        cgrid_geometry,
        live_face_thickness,
        native_face_transports,
        transport_divergence,
    )

    geometry = cgrid_geometry(namelist)
    wet = np.asarray(geometry["mask"]["T"], dtype=bool)
    jj, ii = np.where(wet)
    j0, j1 = int(jj.min()), int(jj.max())
    i0, i1 = int(ii.min()), int(ii.max())
    psi = np.zeros((wet.shape[0] + 1, wet.shape[1] + 1), dtype=np.float64)
    y = np.linspace(0.0, np.pi, j1 - j0 + 2)
    x = np.linspace(0.0, np.pi, i1 - i0 + 2)
    psi[j0:j1 + 2, i0:i1 + 2] = np.sin(y)[:, None] * np.sin(x)[None, :]

    eta = np.asarray(state.eta.data, dtype=np.float64)
    h_u_native = live_face_thickness(eta, geometry, "U")
    h_v_native = live_face_thickness(eta, geometry, "V")
    h_u = np.ones(state.u.data.shape[:2], dtype=np.float64)
    h_v = np.ones(state.v.data.shape[:2], dtype=np.float64)
    h_u[:, 1:] = h_u_native
    h_v[1:, :] = h_v_native
    u_mask = np.zeros_like(h_u)
    v_mask = np.zeros_like(h_v)
    u_mask[:, 1:] = np.asarray(geometry["mask"]["U"], dtype=np.float64)
    v_mask[1:, :] = np.asarray(geometry["mask"]["V"], dtype=np.float64)
    du, dv = recover_velocity_from_streamfunction(
        jnp.asarray(psi), jnp.asarray(1.0 / h_u), jnp.asarray(1.0 / h_v),
        geometry["grid"], u_mask=jnp.asarray(u_mask), v_mask=jnp.asarray(v_mask))
    zeta = np.asarray(relative_vorticity_cell_centre(
        du, dv, geometry["grid"]), dtype=np.float64)
    rms = float(np.sqrt(np.mean(zeta[wet] ** 2, dtype=np.float64)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError("balanced perturbation has zero/non-finite vorticity")
    du = np.asarray(du, dtype=np.float64) / rms
    dv = np.asarray(dv, dtype=np.float64) / rms
    native = {"T": eta, "U": du[:, 1:], "V": dv[1:, :]}
    divergence = transport_divergence(
        native_face_transports(native, geometry), geometry)
    max_divergence_per_amplitude = float(np.max(np.abs(divergence[wet])))
    if max_divergence_per_amplitude > 1.0e-8:
        raise RuntimeError(
            "balanced perturbation transport divergence is not roundoff-small")
    return du, dv, {
        "pattern": "closed-wall sin(pi*x)*sin(pi*y) F-point streamfunction",
        "normalization": "unit wet-T RMS relative vorticity",
        "transport_divergence_max_per_s-1_amplitude_m_s": (
            max_divergence_per_amplitude),
    }


def apply_balanced_vorticity_perturbation(state, namelist, amplitude_s_1: float):
    """Add a smooth non-divergent velocity perturbation at fixed SSH."""
    import jax.numpy as jnp

    if not np.isfinite(amplitude_s_1) or amplitude_s_1 <= 0.0:
        raise ValueError("balanced vorticity amplitude must be positive finite")
    du, dv, provenance = balanced_vorticity_direction(state, namelist)
    perturbed = state._replace(
        u=state.u.replace(data=state.u.data + jnp.asarray(
            amplitude_s_1 * du[..., None])),
        v=state.v.replace(data=state.v.data + jnp.asarray(
            amplitude_s_1 * dv[..., None])),
    )
    return perturbed, provenance | {
        "amplitude_s-1": float(amplitude_s_1),
        "ssh_perturbation_m": 0.0,
    }


def validate_operator_device(mode: str, physical_gpu: int | None, *, environ=None):
    """Require the operator to name the physical GPU selected for GPU mode."""
    environment = os.environ if environ is None else environ
    if mode == "gpu-native":
        if type(physical_gpu) is not int or physical_gpu < 0:
            raise ValueError("gpu-native mode requires explicit non-negative physical_gpu")
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        if visible != str(physical_gpu):
            raise RuntimeError(
                "gpu-native physical_gpu does not match CUDA_VISIBLE_DEVICES: "
                f"{physical_gpu} != {visible!r}")
    elif physical_gpu is not None:
        raise ValueError("physical_gpu is only valid in gpu-native mode")


def run_long(
        namelist, output: Path, *, days: int, mode: str, commit: str,
        physical_gpu: int | None, surface: str,
        profile_steps: int = 0, compare_days: int = 0,
        daily_means: bool = False,
        restart: Path | None = None,
        perturb_amplitude_m: float | None = None,
        perturb_kind: str = "white-ssh",
        profile_only: bool = False,
        hourly_targets: bool = False,
        verify_next: Path | None = None,
) -> dict:
    """Integrate one resolved SWG deck and write one native state per day."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_swg_card,
        swg_surface_wind,
        validate_swg_namelist,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    validate_swg_namelist(namelist)
    if mode not in ("gpu-native", "cpu-certified", "cpu-scanned"):
        raise ValueError(
            "mode must be 'gpu-native', 'cpu-certified', or 'cpu-scanned'")
    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    if type(profile_steps) is not int or profile_steps < 0:
        raise ValueError("profile_steps must be a non-negative integer")
    if type(compare_days) is not int or compare_days < 0:
        raise ValueError("compare_days must be a non-negative integer")
    if profile_only and profile_steps <= 0:
        raise ValueError("profile_only requires positive profile_steps")
    if profile_only and (compare_days or daily_means or hourly_targets):
        raise ValueError(
            "profile_only excludes comparison, daily means, and hourly targets")
    if mode == "cpu-certified" and compare_days:
        raise ValueError("scan comparison requires a scanned mode")
    if daily_means and mode != "gpu-native":
        raise ValueError("daily-mean output requires non-certified gpu-native mode")
    if hourly_targets and daily_means:
        raise ValueError("hourly targets and daily means are separate sampling modes")
    if hourly_targets and mode == "cpu-certified":
        raise ValueError("hourly targets require a scanned mode")
    if hourly_targets and namelist.nn_SWG != 16:
        raise ValueError("hourly coarse targets require the 6.25 km fine deck")
    if verify_next is not None and (not hourly_targets or restart is None):
        raise ValueError("verify_next requires hourly_targets and restart")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    steps_per_day_f = SECONDS_PER_DAY / namelist.rn_Dt_s
    if not float(steps_per_day_f).is_integer():
        raise ValueError("resolved rn_Dt must divide one model day")
    steps_per_day = int(steps_per_day_f)
    if namelist.nn_itend != days * steps_per_day:
        raise ValueError("resolved nn_itend does not equal days times steps/day")

    validate_operator_device(mode, physical_gpu)
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
            f"{mode} requires JAX backend {expected_backend}, got "
            f"{jax.default_backend()}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output}")

    card = build_swg_card(namelist, surface=surface)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64),
    )
    state = card.recipe.initial_state
    restart_provenance = None
    if perturb_amplitude_m is not None and restart is None:
        raise ValueError("--perturb requires --restart")
    if perturb_kind not in ("white-ssh", "balanced-vorticity"):
        raise ValueError("unknown perturbation kind")
    if perturb_kind != "white-ssh" and perturb_amplitude_m is None:
        raise ValueError("non-default perturbation kind requires --perturb")
    if (surface == "rigid_lid" and perturb_amplitude_m is not None
            and perturb_kind == "white-ssh"):
        raise ValueError(
            "white-ssh perturbation is a no-op under rigid_lid (ssh is "
            "projected out); use balanced-vorticity")
    if restart is not None:
        if hourly_targets:
            restart_admission = admit_listed_snapshot(restart)
        else:
            restart_admission = None
        state, restart_provenance = restart_state_from_npz(
            state, forcing, restart, dt_s=card.dt_s,
            perturb_amplitude_m=(
                perturb_amplitude_m if perturb_kind == "white-ssh" else None))
        if surface == "rigid_lid" and np.any(np.asarray(state.eta.data) != 0.0):
            raise ValueError(
                "rigid_lid restart requires a snapshot with identically zero ssh")
        restart_provenance["manifest_admission"] = restart_admission
        restart_provenance["perturb_kind"] = perturb_kind
        if perturb_kind == "balanced-vorticity":
            state, balanced_provenance = apply_balanced_vorticity_perturbation(
                state, namelist, perturb_amplitude_m)
            restart_provenance["balanced_perturbation"] = balanced_provenance
    next_admission = (
        admit_listed_snapshot(verify_next) if verify_next is not None else None)
    profile = None
    parity = None
    hourly_stepper = None
    hourly_restrictors = None
    if mode in ("gpu-native", "cpu-scanned"):
        if restart is None:
            state = seed_steady_forcing_carry(state, forcing)
        day_stepper = build_scanned_stepper(
            model, card.dt_s, forcing, steps_per_day)
        if daily_means:
            from legoesm.ocean.fidelity.swg_coarse_graining import (
                build_jax_native_transport_diagnostic,
            )
            transport_diagnostic = build_jax_native_transport_diagnostic(namelist)
            day_mean_stepper = build_scanned_day_mean_stepper(
                model, card.dt_s, forcing, steps_per_day,
                transport_diagnostic=transport_diagnostic)
        else:
            day_mean_stepper = None
        if hourly_targets:
            from legoesm.ocean.fidelity.nemo_testcase_recipe import (
                SWG_DG50_NAMELIST,
                SWG_MADDISON_NAMELIST,
            )
            from legoesm.ocean.fidelity.swg_coarse_graining import (
                build_jax_conservative_restrictor,
                build_jax_filtered_restrictor,
                cgrid_geometry,
            )

            steps_per_hour_f = SECONDS_PER_HOUR / card.dt_s
            if not float(steps_per_hour_f).is_integer():
                raise ValueError("fine timestep must divide one hour")
            source_geometry = cgrid_geometry(SWG_MADDISON_NAMELIST)
            target_geometry = cgrid_geometry(SWG_DG50_NAMELIST)
            sigma_source_cells = (
                float(target_geometry["spacing_m"])
                / float(source_geometry["spacing_m"]))
            hourly_restrictors = {
                "box": build_jax_conservative_restrictor(
                    source_geometry, target_geometry),
                "filtered": build_jax_filtered_restrictor(
                    source_geometry, target_geometry,
                    sigma_source_cells=sigma_source_cells),
            }
            if surface == "rigid_lid":
                hourly_restrictors = {
                    name: zero_ssh_restrictor(restrictor)
                    for name, restrictor in hourly_restrictors.items()}
            hourly_stepper = build_scanned_hourly_target_stepper(
                model, card.dt_s, forcing, int(steps_per_hour_f),
                hourly_restrictors)
            assert_hourly_stepper_has_no_convolution(hourly_stepper, state)
        if profile_steps:
            profile = profile_step_modes(
                model, state, card.dt_s, forcing, profile_steps)
        if compare_days:
            model.prime_step_caches(state)
            python_state = run_python_steps(
                model, state, card.dt_s, forcing,
                compare_days * steps_per_day)
            jax.block_until_ready(python_state.eta.data)
            scan_state = state
            for _ in range(compare_days):
                scan_state = day_stepper(scan_state)
            jax.block_until_ready(scan_state.eta.data)
            parity = {
                "days": compare_days,
                "steps": compare_days * steps_per_day,
                "differences": prognostic_differences(
                    python_state, scan_state),
            }
    else:
        day_stepper = None
        day_mean_stepper = None
        if profile_steps:
            profile = profile_python_steps(
                model, state, card.dt_s, forcing, profile_steps)
    output.mkdir(parents=True)
    manifest = {
        "schema": (
            "legoesm-swg-double-gyre-hourly-target-v2"
            if hourly_targets else
            "legoesm-swg-double-gyre-daily-v2"
            if daily_means else "legoesm-swg-double-gyre-daily-v1"),
        "legoesm_git_sha": commit,
        "mode": mode,
        "surface": surface,
        "certified_policy": certified_policy,
        "deck_certification": "NEMO ladder measured; see fidelity receipts",
        "backend": jax.default_backend(),
        "physical_gpu": physical_gpu,
        "precision_policy": "fp64",
        "transcendentals": "native" if mode == "gpu-native" else "libm",
        "days": days,
        "steps_per_day": steps_per_day,
        "integration": (
            "one-jitted-scan-per-day" if mode in ("gpu-native", "cpu-scanned")
            else "production-python-step-loop"),
        "sampling": ("hourly_dual_coarse_target" if hourly_targets else
                     "daily_mean" if daily_means else "daily_snapshot"),
        "available_sampling": (
            ["hourly_box_coarse_state", "hourly_filtered_coarse_state"]
            if hourly_targets else
            ["daily_mean", "end_of_day_snapshot"]
            if daily_means else ["end_of_day_snapshot"]),
        "profile": profile,
        "scan_vs_python": parity,
        "restart": restart_provenance,
        "restart_next_verification": (
            None if verify_next is None else {
                "path": str(verify_next.resolve()),
                "admission": next_admission,
                "tolerance": (LID_RESTART_TOLERANCE
                              if surface == "rigid_lid" else None),
                "result": None,
            }),
        "numerical_health": None,
        "namelist": namelist._asdict(),
        "fields": ({} if hourly_targets else {
            "ssh_m": "end-of-day snapshot; T points",
            "u_m_s": "end-of-day snapshot; native NEMO U record",
            "v_m_s": "end-of-day snapshot; native NEMO V record",
        }),
    }
    if daily_means:
        manifest["fields"].update({
            "ssh_daily_mean_m": "daily mean of post-step T-point SSH",
            "u_daily_mean_m_s": "daily mean of post-step native U velocity",
            "v_daily_mean_m_s": "daily mean of post-step native V velocity",
        })
    if hourly_targets:
        manifest["hourly_samples_per_day"] = HOURS_PER_DAY
        manifest["hourly_sampling"] = (
            "right-endpoint instantaneous state, restricted inside compiled scan")
        manifest["hourly_target_deck"] = "dg50"
        manifest["hourly_operators"] = {
            "box": (
                "exact nested T-volume and native face-transport restriction"),
            "filtered": (
                "fine cell-volume and face-transport Gaussian, sigma=8 fine "
                "cells, even scalar/tangential and odd normal wall parity, "
                "then exact nested restriction"),
        }
        manifest["fields"].update({
            "{operator}_ssh_initial_m": "coarse T-point state at segment t0",
            "{operator}_u_initial_m_s": "coarse native U state at segment t0",
            "{operator}_v_initial_m_s": "coarse native V state at segment t0",
            "{operator}_ssh_hourly_m": "24 coarse hourly T-point states",
            "{operator}_u_hourly_m_s": "24 coarse hourly native U states",
            "{operator}_v_hourly_m_s": "24 coarse hourly native V states",
        })
    if profile_only:
        completed = manifest | {
            "schema": "legoesm-swg-double-gyre-step-profile-v1",
            "profile_only": True,
            "status": "complete",
            "completed_days": 0,
            "daily_sha256": {},
        }
        _write_json_atomic(output / "manifest.json", completed)
        return completed
    pending_path = output / "manifest.pending.json"
    day_hashes: dict[str, str] = {}
    _write_json_atomic(
        pending_path,
        manifest | {"status": "pending", "completed_days": 0,
                    "daily_sha256": day_hashes})

    hourly_initial = None
    if hourly_restrictors is not None:
        initial_arguments = (
            state.eta.data, state.u.data[:, 1:, 0], state.v.data[1:, :, 0])
        hourly_initial = tuple(hourly_restrictors[name](*initial_arguments)
                               for name in ("box", "filtered"))

    for day in range(1, days + 1):
        means = None
        hourly = None
        if day_stepper is None:
            state = run_python_steps(
                model, state, card.dt_s, forcing, steps_per_day)
        elif hourly_stepper is not None:
            state, hourly = hourly_stepper(state)
        elif day_mean_stepper is not None:
            state, means = day_mean_stepper(state)
        else:
            state = day_stepper(state)
        jax.block_until_ready(state.eta.data)
        arrays = {
            "schema": np.str_(manifest["schema"]),
            "legoesm_git_sha": np.str_(commit),
            "mode": np.str_(mode),
            "certified_policy": np.bool_(certified_policy),
            "day": np.int64(day),
            "absolute_day": np.int64(
                day + (0 if restart_provenance is None else
                       restart_provenance["source_day"])),
            "step": np.int64(
                day * steps_per_day + (
                    0 if restart_provenance is None
                    else restart_provenance["source_step"])),
            "dt_s": np.float64(card.dt_s),
        }
        if hourly is None:
            arrays.update({
                "ssh_m": np.asarray(state.eta.data, dtype=np.float64),
                "u_m_s": np.asarray(state.u.data[:, 1:, 0], dtype=np.float64),
                "v_m_s": np.asarray(state.v.data[1:, :, 0], dtype=np.float64),
            })
            if state.rigid_lid_phi is not None:
                arrays["rigid_lid_phi_m2_s2"] = np.asarray(
                    state.rigid_lid_phi, dtype=np.float64)
        daily_health = numerical_health(state, namelist)
        previous_health = manifest["numerical_health"]
        manifest["numerical_health"] = {
            "all_finite": bool(
                daily_health["all_finite"] and
                (previous_health is None or previous_health["all_finite"])),
            "minimum_total_layer_thickness_m": min(
                daily_health["minimum_total_layer_thickness_m"],
                float("inf") if previous_health is None else
                previous_health["minimum_total_layer_thickness_m"]),
            "maximum_u_courant": max(
                daily_health["maximum_u_courant"],
                float("-inf") if previous_health is None else
                previous_health["maximum_u_courant"]),
            "maximum_v_courant": max(
                daily_health["maximum_v_courant"],
                float("-inf") if previous_health is None else
                previous_health["maximum_v_courant"]),
        }
        if means is not None:
            arrays.update({
                "daily_mean_sampling": np.str_(
                    "arithmetic mean of every post-step state in model day"),
                "snapshot_sampling": np.str_("end-of-day instantaneous state"),
                "daily_mean_sample_count": np.int64(steps_per_day),
                "ssh_daily_mean_m": np.asarray(means[0], dtype=np.float64),
                "u_daily_mean_m_s": np.asarray(means[1], dtype=np.float64),
                "v_daily_mean_m_s": np.asarray(means[2], dtype=np.float64),
                "u_daily_mean_transport_m3_s": np.asarray(
                    means[3], dtype=np.float64),
                "v_daily_mean_transport_m3_s": np.asarray(
                    means[4], dtype=np.float64),
            })
            if state.rigid_lid_phi is not None:
                arrays["rigid_lid_phi_daily_mean_m2_s2"] = np.asarray(
                    means[-1], dtype=np.float64)
        if hourly is not None:
            source_day = (0 if restart_provenance is None else
                          restart_provenance["source_day"])
            arrays.update({
                "hourly_sampling": np.str_(manifest["hourly_sampling"]),
                "hourly_offset_hours": np.arange(
                    1, HOURS_PER_DAY + 1, dtype=np.int64),
                "absolute_hour": (
                    (source_day + day - 1) * HOURS_PER_DAY
                    + np.arange(1, HOURS_PER_DAY + 1, dtype=np.int64)),
            })
            for operator_index, operator in enumerate(("box", "filtered")):
                operator_hourly = hourly[operator_index]
                for field_index, (field, unit) in enumerate((
                        ("ssh", "m"), ("u", "m_s"), ("v", "m_s"))):
                    arrays[f"{operator}_{field}_hourly_{unit}"] = np.asarray(
                        operator_hourly[field_index], dtype=np.float64)
                    if day == 1:
                        arrays[f"{operator}_{field}_initial_{unit}"] = np.asarray(
                            hourly_initial[operator_index][field_index],
                            dtype=np.float64)
        if day == 1 and verify_next is not None:
            comparison = compare_native_state_bitwise(
                state, verify_next,
                LID_RESTART_TOLERANCE["value"] if surface == "rigid_lid"
                else None)
            manifest["restart_next_verification"]["result"] = comparison
            if not comparison["within_tolerance"]:
                _write_json_atomic(
                    pending_path,
                    manifest | {"status": "refused", "completed_days": 0,
                                "daily_sha256": day_hashes})
                raise RuntimeError(
                    "restarted segment does not reproduce uninterrupted next day")
        name = f"day{day:04d}.npz"
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(output / name)
        day_hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if day % METADATA_CHECKPOINT_DAYS == 0 or day == days:
            _write_hashes_atomic(output / "daily.sha256", day_hashes)
            _write_json_atomic(
                pending_path,
                manifest | {"status": "pending", "completed_days": day,
                            "daily_sha256": day_hashes})
    completed = manifest | {
        "status": "complete", "completed_days": days,
        "daily_sha256": day_hashes,
    }
    _write_json_atomic(pending_path, completed)
    pending_path.replace(output / "manifest.json")
    return completed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deck", choices=("maddison", "dg50", "dg50-nu80", "dg3125",
                           "maddison-rl", "dg50-rl", "dg50-nu80-rl"),
        default="maddison")
    parser.add_argument(
        "--surface", choices=("explicit", "rigid_lid"), required=True,
        help="SWE RK3 free-surface treatment; the certified card is explicit")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--perturb", type=float,
                        help="perturbation amplitude: m for white SSH, s^-1 for balanced")
    parser.add_argument(
        "--perturb-kind", choices=("white-ssh", "balanced-vorticity"),
        default="white-ssh")
    parser.add_argument(
        "--mode", choices=("gpu-native", "cpu-certified", "cpu-scanned"),
        default="gpu-native")
    parser.add_argument(
        "--physical-gpu", type=int,
        help="required physical CUDA device for --mode gpu-native; no default")
    parser.add_argument(
        "--profile-steps", type=int, default=0,
        help="warm wall-clock profile of Python and scan programs")
    parser.add_argument(
        "--profile-only", action="store_true",
        help="write the warm step profile without integrating a model day")
    parser.add_argument(
        "--compare-days", type=int, default=0,
        help="compare the daily scan with the Python loop from one state")
    parser.add_argument(
        "--daily-means", action="store_true",
        help="also accumulate every post-step state inside the compiled day")
    parser.add_argument(
        "--hourly-targets", action="store_true",
        help="retain only conservative dg50 targets at each fine-model hour")
    parser.add_argument(
        "--verify-next", type=Path,
        help="admitted uninterrupted next-day snapshot required to match bitwise")
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        namelist = resolved_daily_namelist(
            args.days, deck=args.deck, surface=args.surface)
        manifest = run_long(
            namelist, args.output, days=args.days, mode=args.mode,
            commit=git_sha(), physical_gpu=args.physical_gpu,
            surface=args.surface,
            profile_steps=args.profile_steps,
            compare_days=args.compare_days, daily_means=args.daily_means,
            restart=args.restart, perturb_amplitude_m=args.perturb,
            perturb_kind=args.perturb_kind,
            profile_only=args.profile_only, hourly_targets=args.hourly_targets,
            verify_next=args.verify_next)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        f"COMPLETED days={manifest['days']} backend={manifest['backend']} "
        f"certified_policy={manifest['certified_policy']} "
        f"deck_certification={manifest['deck_certification']} "
        f"output={args.output.resolve()}")
    if manifest["profile"] is not None:
        profile = manifest["profile"]
        message = (
            f"PROFILE steps={profile['steps']} "
            f"python_ms_per_step={profile['python_ms_per_step']:.6f}")
        if "scan_ms_per_step" in profile:
            message += (
                f" scan_ms_per_step={profile['scan_ms_per_step']:.6f} "
                f"target_met={profile['target_met']}")
        print(message)
    if manifest["scan_vs_python"] is not None:
        parity = manifest["scan_vs_python"]
        print(
            f"SCAN_PARITY days={parity['days']} "
            f"within_atol={parity['differences']['within_atol']} "
            f"atol={parity['differences']['atol']:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

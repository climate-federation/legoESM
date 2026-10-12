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
SCAN_PARITY_ATOL = 1.0e-12
GPU_TARGET_MS_PER_STEP = 5.0


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_hashes_atomic(path: Path, hashes: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    temporary.replace(path)


def resolved_daily_namelist(days: int):
    """Return the user-selected Maddison deck resolved for a number of days."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    dt_s = SWG_MADDISON_NAMELIST.rn_Dt_s
    steps_per_day = SECONDS_PER_DAY / dt_s
    if not float(steps_per_day).is_integer():
        raise ValueError(f"rn_Dt={dt_s} does not divide one model day")
    cadence = int(steps_per_day)
    return SWG_MADDISON_NAMELIST._replace(
        nn_itend=days * cadence, nn_write=cadence, nn_stock=cadence)


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


def run_long(
        namelist, output: Path, *, days: int, mode: str, commit: str,
        profile_steps: int = 0, compare_days: int = 0,
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
    if mode not in ("gpu-native", "cpu-certified"):
        raise ValueError("mode must be 'gpu-native' or 'cpu-certified'")
    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    if type(profile_steps) is not int or profile_steps < 0:
        raise ValueError("profile_steps must be a non-negative integer")
    if type(compare_days) is not int or compare_days < 0:
        raise ValueError("compare_days must be a non-negative integer")
    if mode != "gpu-native" and (profile_steps or compare_days):
        raise ValueError("profiling and scan comparison require gpu-native mode")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    steps_per_day_f = SECONDS_PER_DAY / namelist.rn_Dt_s
    if not float(steps_per_day_f).is_integer():
        raise ValueError("resolved rn_Dt must divide one model day")
    steps_per_day = int(steps_per_day_f)
    if namelist.nn_itend != days * steps_per_day:
        raise ValueError("resolved nn_itend does not equal days times steps/day")

    if mode == "gpu-native":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("gpu-native mode requires CUDA_VISIBLE_DEVICES=1")
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

    card = build_swg_card(namelist)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64),
    )
    state = card.recipe.initial_state
    profile = None
    parity = None
    if mode == "gpu-native":
        state = seed_steady_forcing_carry(state, forcing)
        day_stepper = build_scanned_stepper(
            model, card.dt_s, forcing, steps_per_day)
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
    output.mkdir(parents=True)
    manifest = {
        "schema": "legoesm-swg-double-gyre-daily-v1",
        "legoesm_git_sha": commit,
        "mode": mode,
        "certified_policy": certified_policy,
        "deck_certification": "pending NEMO ladder acquisition",
        "backend": jax.default_backend(),
        "precision_policy": "fp64",
        "transcendentals": "native" if mode == "gpu-native" else "libm",
        "days": days,
        "steps_per_day": steps_per_day,
        "integration": (
            "one-jitted-scan-per-day" if mode == "gpu-native"
            else "production-python-step-loop"),
        "profile": profile,
        "scan_vs_python": parity,
        "namelist": namelist._asdict(),
        "fields": {
            "ssh_m": "T points",
            "u_m_s": "native NEMO U record (west redundant face removed)",
            "v_m_s": "native NEMO V record (south redundant face removed)",
        },
    }
    pending_path = output / "manifest.pending.json"
    day_hashes: dict[str, str] = {}
    _write_json_atomic(
        pending_path,
        manifest | {"status": "pending", "completed_days": 0,
                    "daily_sha256": day_hashes})

    for day in range(1, days + 1):
        if day_stepper is None:
            state = run_python_steps(
                model, state, card.dt_s, forcing, steps_per_day)
        else:
            state = day_stepper(state)
        jax.block_until_ready(state.eta.data)
        arrays = {
            "schema": np.str_(manifest["schema"]),
            "legoesm_git_sha": np.str_(commit),
            "mode": np.str_(mode),
            "certified_policy": np.bool_(certified_policy),
            "day": np.int64(day),
            "step": np.int64(day * steps_per_day),
            "dt_s": np.float64(card.dt_s),
            "ssh_m": np.asarray(state.eta.data, dtype=np.float64),
            "u_m_s": np.asarray(state.u.data[:, 1:, 0], dtype=np.float64),
            "v_m_s": np.asarray(state.v.data[1:, :, 0], dtype=np.float64),
        }
        name = f"day{day:04d}.npz"
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(output / name)
        day_hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
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
        "--mode", choices=("gpu-native", "cpu-certified"),
        default="gpu-native")
    parser.add_argument(
        "--profile-steps", type=int, default=0,
        help="warm wall-clock profile of Python and scan programs")
    parser.add_argument(
        "--compare-days", type=int, default=0,
        help="compare the daily scan with the Python loop from one state")
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        namelist = resolved_daily_namelist(args.days)
        manifest = run_long(
            namelist, args.output, days=args.days, mode=args.mode,
            commit=git_sha(), profile_steps=args.profile_steps,
            compare_days=args.compare_days)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        f"COMPLETED days={manifest['days']} backend={manifest['backend']} "
        f"certified_policy={manifest['certified_policy']} "
        f"deck_certification=pending output={args.output.resolve()}")
    if manifest["profile"] is not None:
        profile = manifest["profile"]
        print(
            f"PROFILE steps={profile['steps']} "
            f"python_ms_per_step={profile['python_ms_per_step']:.6f} "
            f"scan_ms_per_step={profile['scan_ms_per_step']:.6f} "
            f"target_met={profile['target_met']}")
    if manifest["scan_vs_python"] is not None:
        parity = manifest["scan_vs_python"]
        print(
            f"SCAN_PARITY days={parity['days']} "
            f"within_atol={parity['differences']['within_atol']} "
            f"atol={parity['differences']['atol']:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

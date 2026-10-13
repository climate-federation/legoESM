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


def resolved_daily_namelist(days: int, *, deck: str = "maddison"):
    """Return a supported double-gyre deck resolved for a number of days."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
    )

    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    decks = {"maddison": SWG_MADDISON_NAMELIST, "dg50": SWG_DG50_NAMELIST}
    if deck not in decks:
        raise ValueError(f"unsupported double-gyre deck {deck!r}")
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


def build_scanned_day_mean_stepper(model, dt_s: float, forcing, n_steps: int):
    """Return one JIT program for a day and its IOIPSL-order state means."""
    import jax
    import jax.numpy as jnp

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("daily-mean scan n_steps must be a positive integer")

    def scan_body(carry, count):
        state, mean_ssh, mean_u, mean_v = carry
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
        return (state, mean_ssh, mean_u, mean_v), None

    @jax.jit
    def scanned(state):
        means = (
            jnp.zeros_like(state.eta.data),
            jnp.zeros_like(state.u.data[:, 1:, 0]),
            jnp.zeros_like(state.v.data[1:, :, 0]),
        )
        counts = jnp.arange(1, n_steps + 1, dtype=state.eta.data.dtype)
        (state, mean_ssh, mean_u, mean_v), _ = jax.lax.scan(
            scan_body, (state, *means), counts)
        return state, (mean_ssh, mean_u, mean_v)

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


def run_long(
        namelist, output: Path, *, days: int, mode: str, commit: str,
        profile_steps: int = 0, compare_days: int = 0,
        daily_means: bool = False,
        restart: Path | None = None,
        perturb_amplitude_m: float | None = None,
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
    if mode != "gpu-native" and compare_days:
        raise ValueError("scan comparison requires gpu-native mode")
    if daily_means and mode != "gpu-native":
        raise ValueError("daily-mean output requires non-certified gpu-native mode")
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
    restart_provenance = None
    if perturb_amplitude_m is not None and restart is None:
        raise ValueError("--perturb requires --restart")
    if restart is not None:
        state, restart_provenance = restart_state_from_npz(
            state, forcing, restart, dt_s=card.dt_s,
            perturb_amplitude_m=perturb_amplitude_m)
    profile = None
    parity = None
    if mode == "gpu-native":
        if restart is None:
            state = seed_steady_forcing_carry(state, forcing)
        day_stepper = build_scanned_stepper(
            model, card.dt_s, forcing, steps_per_day)
        day_mean_stepper = (
            build_scanned_day_mean_stepper(
                model, card.dt_s, forcing, steps_per_day)
            if daily_means else None)
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
            "legoesm-swg-double-gyre-daily-v2"
            if daily_means else "legoesm-swg-double-gyre-daily-v1"),
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
        "sampling": "daily_mean" if daily_means else "daily_snapshot",
        "available_sampling": (
            ["daily_mean", "end_of_day_snapshot"]
            if daily_means else ["end_of_day_snapshot"]),
        "profile": profile,
        "scan_vs_python": parity,
        "restart": restart_provenance,
        "numerical_health": None,
        "namelist": namelist._asdict(),
        "fields": {
            "ssh_m": "end-of-day snapshot; T points",
            "u_m_s": "end-of-day snapshot; native NEMO U record",
            "v_m_s": "end-of-day snapshot; native NEMO V record",
        },
    }
    if daily_means:
        manifest["fields"].update({
            "ssh_daily_mean_m": "daily mean of post-step T-point SSH",
            "u_daily_mean_m_s": "daily mean of post-step native U velocity",
            "v_daily_mean_m_s": "daily mean of post-step native V velocity",
        })
    pending_path = output / "manifest.pending.json"
    day_hashes: dict[str, str] = {}
    _write_json_atomic(
        pending_path,
        manifest | {"status": "pending", "completed_days": 0,
                    "daily_sha256": day_hashes})

    for day in range(1, days + 1):
        means = None
        if day_stepper is None:
            state = run_python_steps(
                model, state, card.dt_s, forcing, steps_per_day)
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
            "step": np.int64(
                day * steps_per_day + (
                    0 if restart_provenance is None
                    else restart_provenance["source_step"])),
            "dt_s": np.float64(card.dt_s),
            "ssh_m": np.asarray(state.eta.data, dtype=np.float64),
            "u_m_s": np.asarray(state.u.data[:, 1:, 0], dtype=np.float64),
            "v_m_s": np.asarray(state.v.data[1:, :, 0], dtype=np.float64),
        }
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
            })
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
    parser.add_argument("--deck", choices=("maddison", "dg50"),
                        default="maddison")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--perturb", type=float,
                        help="fixed-seed wet T-point SSH noise amplitude [m]")
    parser.add_argument(
        "--mode", choices=("gpu-native", "cpu-certified"),
        default="gpu-native")
    parser.add_argument(
        "--profile-steps", type=int, default=0,
        help="warm wall-clock profile of Python and scan programs")
    parser.add_argument(
        "--compare-days", type=int, default=0,
        help="compare the daily scan with the Python loop from one state")
    parser.add_argument(
        "--daily-means", action="store_true",
        help="also accumulate every post-step state inside the compiled day")
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        namelist = resolved_daily_namelist(args.days, deck=args.deck)
        manifest = run_long(
            namelist, args.output, days=args.days, mode=args.mode,
            commit=git_sha(), profile_steps=args.profile_steps,
            compare_days=args.compare_days, daily_means=args.daily_means,
            restart=args.restart, perturb_amplitude_m=args.perturb)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        f"COMPLETED days={manifest['days']} backend={manifest['backend']} "
        f"certified_policy={manifest['certified_policy']} "
        f"deck_certification=pending output={args.output.resolve()}")
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

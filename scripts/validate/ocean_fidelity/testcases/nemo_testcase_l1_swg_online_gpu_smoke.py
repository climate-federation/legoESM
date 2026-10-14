#!/usr/bin/env python3
"""Fail-closed GPU smoke test for the SWG online-learning execution path."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical_gpu", type=int)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--worker", choices=("sampler", "symmetric", "template"),
        help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _memory_report(device) -> dict[str, int]:
    """Read the installed JAX Device.memory_stats diagnostic contract."""
    stats = device.memory_stats()
    if not isinstance(stats, dict) or "peak_bytes_in_use" not in stats:
        raise RuntimeError("GPU device does not expose peak_bytes_in_use")
    peak = int(stats["peak_bytes_in_use"])
    current = int(stats.get("bytes_in_use", -1))
    if peak < 0 or current < 0:
        raise RuntimeError("GPU memory statistics are unavailable")
    return {"peak_bytes_in_use": peak, "bytes_in_use": current}


def _load_restart(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as saved:
        required = ("ssh_m", "u_m_s", "v_m_s")
        missing = set(required) - set(saved.files)
        if missing:
            raise ValueError(f"restart is missing {sorted(missing)}")
        fields = tuple(np.asarray(saved[name], dtype=np.float64)
                       for name in required)
    if len({value.shape for value in fields}) != 1:
        raise ValueError("restart T/U/V native shapes differ")
    if not all(np.all(np.isfinite(value)) for value in fields):
        raise ValueError("restart contains non-finite fields")
    return fields


def _sampler_components():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
    )
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        build_jax_conservative_restrictor,
        build_jax_filtered_restrictor,
        cgrid_geometry,
    )

    source = cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = cgrid_geometry(SWG_DG50_NAMELIST)
    sigma = float(target["spacing_m"]) / float(source["spacing_m"])
    restrictors = {
        "box": build_jax_conservative_restrictor(source, target),
        "filtered": build_jax_filtered_restrictor(
            source, target, sigma_source_cells=sigma),
    }
    return source, target, restrictors


def _load_driver_module():
    import importlib.util

    driver_path = Path(__file__).parents[3] / "run" / "run_swg_double_gyre.py"
    spec = importlib.util.spec_from_file_location("swg_gpu_smoke_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("cannot load SWG double-gyre driver")
    spec.loader.exec_module(driver)
    return driver


def _sample_on_device(fields, device):
    import jax
    import jax.numpy as jnp
    from legoesm.training.swg_online import assert_no_convolution_operations

    # Import by path because scripts are intentionally not Python packages.
    driver = _load_driver_module()

    with jax.default_device(device):
        _, _, restrictors = _sampler_components()
        sampler = jax.jit(driver.build_dual_coarse_sampler(restrictors))
        arguments = tuple(jnp.asarray(value) for value in fields)
        lowered = sampler.lower(*arguments)
        assert_no_convolution_operations(lowered, label="hourly dual sampler")
        compiled = lowered.compile()
        analysis = compiled.memory_analysis()
        sampled = compiled(*arguments)
        jax.block_until_ready(sampled[1][0])
        memory = None if analysis is None else {
            name: int(getattr(analysis, name)) for name in (
                "argument_size_in_bytes", "output_size_in_bytes",
                "temp_size_in_bytes", "alias_size_in_bytes")}
        return (tuple(tuple(jax.device_get(value) for value in operator)
                      for operator in sampled), memory)


def _compare_samples(cpu_sample, gpu_sample, *,
                     label: str = "CPU/GPU") -> list[dict]:
    import numpy as np

    rows = []
    for operator, cpu_fields, gpu_fields in zip(
            ("box", "filtered"), cpu_sample, gpu_sample, strict=True):
        for field, cpu_value, gpu_value in zip(
                ("ssh_m", "u_m_s", "v_m_s"), cpu_fields, gpu_fields,
                strict=True):
            if not (np.all(np.isfinite(cpu_value))
                    and np.all(np.isfinite(gpu_value))):
                raise AssertionError(
                    f"{operator} {field} {label} sample is non-finite")
            difference = np.abs(np.asarray(gpu_value) - np.asarray(cpu_value))
            maximum = float(np.max(difference))
            atol = 2.1e-12 if field == "ssh_m" else 3.0e-14
            np.testing.assert_allclose(
                gpu_value, cpu_value, rtol=3.0e-14, atol=atol,
                err_msg=f"{operator} {field} {label} mismatch")
            rows.append({
                "comparison": label, "operator": operator, "field": field,
                "max_abs": maximum, "atol": atol, "rtol": 3.0e-14,
            })
    return rows


def _run_in_scan_sampler(driver, model, state, dt_s, forcing, restrictors):
    """Execute the production nested-scan sampler program on the device.

    Two fine steps per stored hour keep a real inner loop at 1/36 of the
    production cost.  Its last sample must equal the stand-alone sampler on
    the stepped state.  Round 5c's GPU fault depended on how XLA fused the
    filter with its surroundings, which differs between the two programs, so
    the stand-alone sample alone does not show that the scan runs.
    """
    import jax

    stepper = driver.build_scanned_hourly_target_stepper(
        model, dt_s, forcing, 2, restrictors)
    final, hourly = stepper(state)
    jax.block_until_ready(final.eta.data)
    in_scan = tuple(tuple(jax.device_get(field[-1]) for field in operator)
                    for operator in hourly)
    standalone = jax.jit(driver.build_dual_coarse_sampler(restrictors))(
        final.eta.data, final.u.data[:, 1:, 0], final.v.data[1:, :, 0])
    standalone = tuple(tuple(jax.device_get(field) for field in operator)
                       for operator in standalone)
    return _compare_samples(
        standalone, in_scan, label="in-scan/stand-alone")


def _state_from_coarse_arrays(card, coarse):
    import jax.numpy as jnp

    ssh, u_native, v_native = coarse
    state = card.recipe.initial_state
    u = jnp.zeros_like(state.u.data).at[:, 1:, 0].set(u_native)
    v = jnp.zeros_like(state.v.data).at[1:, :, 0].set(v_native)
    return state._replace(
        eta=state.eta.replace(data=jnp.asarray(ssh)),
        u=state.u.replace(data=u), v=state.v.replace(data=v))


def _run_worker(args) -> dict:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_surface_wind,
    )
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.training.swg_online import (
        BalancedNormalizers,
        SWGNetworkSpec,
        SWGScalarPotentialCNN,
        assert_network_gradients_live,
        assert_no_convolution_operations,
        build_closure_geometry,
        build_segment_fn,
        deterministic_trajectory_loss,
        swg_online_step,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("GPU smoke requires JAX x64")
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"GPU smoke selected backend {jax.default_backend()!r}, not 'gpu'")
    gpu = jax.devices("gpu")[0]
    driver = _load_driver_module()
    admission = driver.admit_listed_snapshot(args.restart)
    fields = _load_restart(args.restart)
    started = time.perf_counter()

    if args.worker == "sampler":
        with jax.default_device(gpu):
            fine_card = build_swg_card(SWG_MADDISON_NAMELIST)
            fine_model = LatLonCGridOceanModel(
                fine_card.recipe.grid, fine_card.recipe.z_coord,
                fine_card.recipe.model_config)
            fine_tau = swg_surface_wind(SWG_MADDISON_NAMELIST)
            fine_forcing = OceanSurfaceForcing(
                tau_i_native=jnp.asarray(fine_tau[0], dtype=jnp.float64),
                tau_j_native=jnp.asarray(fine_tau[1], dtype=jnp.float64))
            fine_state, _ = driver.restart_state_from_npz(
                fine_card.recipe.initial_state, fine_forcing, args.restart,
                dt_s=fine_card.dt_s)
            _, _, full_restrictors = _sampler_components()
            full_stepper = driver.build_scanned_hourly_target_stepper(
                fine_model, fine_card.dt_s, fine_forcing,
                int(3_600 / fine_card.dt_s), full_restrictors)
            driver.assert_hourly_stepper_has_no_convolution(
                full_stepper, fine_state)
            in_scan_comparisons = _run_in_scan_sampler(
                driver, fine_model, fine_state, fine_card.dt_s, fine_forcing,
                full_restrictors)
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("CPU reference device is unavailable")
        cpu_sample, cpu_compiled_memory = _sample_on_device(
            fields, cpu_devices[0])
        gpu_sample, gpu_compiled_memory = _sample_on_device(fields, gpu)
        result = {
            "task": "dual_hourly_sampler",
            "comparisons": (_compare_samples(cpu_sample, gpu_sample)
                            + in_scan_comparisons),
            "cpu_compiled_memory": cpu_compiled_memory,
            "gpu_compiled_memory": gpu_compiled_memory,
            "restart_admission": admission,
        }
    else:
        objective = args.worker
        with jax.default_device(gpu):
            _, _, restrictors = _sampler_components()
            sampler = jax.jit(driver.build_dual_coarse_sampler(restrictors))
            sampled = sampler(*(jnp.asarray(value) for value in fields))
            selected = sampled[0 if objective == "symmetric" else 1]

            card = build_swg_card(SWG_DG50_NAMELIST)
            geometry = build_closure_geometry(card)
            initial = _state_from_coarse_arrays(card, selected)
            network = SWGScalarPotentialCNN(
                SWGNetworkSpec(
                    hidden_channels=16, hidden_layers=4, kernel_size=3,
                    ssh_scale_m=1.0, velocity_scale_m_s=1.0,
                    potential_scale_m3_s2=1.0),
                key=jax.random.PRNGKey(20261013))
            tau_u, tau_v = swg_surface_wind(SWG_DG50_NAMELIST)
            forcing = OceanSurfaceForcing(
                tau_i_native=jnp.asarray(tau_u, dtype=jnp.float64),
                tau_j_native=jnp.asarray(tau_v, dtype=jnp.float64))
            stage_noise = jnp.zeros(
                (3,) + tuple(geometry.tmask.shape), dtype=jnp.float64)
            one_step = eqx.filter_jit(lambda net, state, noise, key, scale:
                swg_online_step(
                    net, state, noise, key, scale, card=card,
                    geometry=geometry, forcing=forcing))
            one_step_lowered = one_step.lower(
                network, initial, stage_noise, jax.random.PRNGKey(7),
                jnp.asarray(0.0))
            assert_no_convolution_operations(
                one_step_lowered, label="closure model step")

            _, segment = build_segment_fn(
                card, geometry, window_days=1, checkpoint=True)
            n_steps = 24 * int(3_600 / card.dt_s)
            noise = jnp.zeros(
                (n_steps, 3) + tuple(geometry.tmask.shape), dtype=jnp.float64)
            key = jax.random.PRNGKey(311)
            _, target = segment(None, initial, noise, key, jnp.asarray(0.0))
            target = tuple(value[None] for value in target)
            normalizers = BalancedNormalizers(1.0, 1.0, 1.0)

            def loss(net):
                _, predicted = segment(
                    net, initial, noise, key, jnp.asarray(0.0))
                return deterministic_trajectory_loss(
                    tuple(value[None] for value in predicted), target,
                    geometry, objective=objective,
                    beta_m1_s1=card.resolved_namelist.rn_beta_m_s,
                    basin_length_m=card.resolved_namelist.rn_domsiz_m,
                    normalizers=normalizers)

            measured = eqx.filter_jit(eqx.filter_value_and_grad(loss))
            lowered = measured.lower(network)
            assert_no_convolution_operations(
                lowered, label=f"{objective} reverse training objective")
            value, gradients = measured(network)
            jax.block_until_ready(value)
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise RuntimeError(
                    f"{objective} reverse objective is not positive finite")
            assert_network_gradients_live(gradients)
            result = {
                "task": f"reverse_24h_{objective}",
                "objective": float(value),
                "gradient_leaves": len(jax.tree.leaves(eqx.filter(
                    gradients, eqx.is_inexact_array))),
                "restart_admission": admission,
            }

    result.update({
        "backend": jax.default_backend(),
        "device": str(gpu),
        "wall_seconds": time.perf_counter() - started,
        "memory": _memory_report(gpu),
    })
    return result


def _run_parent(args) -> dict:
    if args.physical_gpu < 0:
        raise ValueError("physical_gpu must be non-negative")
    memory_fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    if not memory_fraction:
        raise ValueError("XLA_PYTHON_CLIENT_MEM_FRACTION must be set")
    script = Path(__file__).resolve()
    repo_root = script.parents[4]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root, check=True, capture_output=True, text=True)
    if status.stdout:
        raise RuntimeError("GPU smoke requires a clean committed worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True).stdout.strip()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    environment["JAX_ENABLE_X64"] = "1"
    environment.pop("JAX_PLATFORMS", None)
    results = []
    with tempfile.TemporaryDirectory(prefix="swgonl-gpu-smoke-") as directory:
        temporary = Path(directory)
        for task in ("sampler", "symmetric", "template"):
            worker_output = temporary / f"{task}.json"
            command = [
                sys.executable, str(script), str(args.physical_gpu),
                "--restart", str(args.restart), "--output", str(args.output),
                "--worker", task, "--worker-output", str(worker_output),
            ]
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"GPU smoke worker {task} exited {completed.returncode}")
            results.append(json.loads(worker_output.read_text()))
    return {
        "schema": "legoesm-swg-online-gpu-smoke-v2",
        "status": "complete",
        "physical_gpu": args.physical_gpu,
        "legoesm_git_sha": commit,
        "xla_python_client_mem_fraction": memory_fraction,
        "restart": str(args.restart.resolve()),
        "results": results,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker is not None:
            if args.worker_output is None:
                raise ValueError("worker mode requires --worker-output")
            report = _run_worker(args)
            args.worker_output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(
                f"GPU_SMOKE_TASK_PASS task={report['task']} "
                f"wall_seconds={report['wall_seconds']:.3f} "
                f"peak_bytes={report['memory']['peak_bytes_in_use']}")
            return 0
        report = _run_parent(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (AssertionError, OSError, RuntimeError, subprocess.SubprocessError,
            ValueError) as exc:
        print(f"GPU_SMOKE_FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"GPU_SMOKE_PASS output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

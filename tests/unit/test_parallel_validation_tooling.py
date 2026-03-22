"""Unit tests for scripts/run_parallel_validation.py logic."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_module():
    # The main script is now a deprecated wrapper; the actual logic lives
    # in scripts/legacy/run_parallel_validation.py.
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "legacy"
        / "run_parallel_validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_parallel_validation",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


rpv = _load_module()


def _base_args(**overrides):
    data = dict(
        scaling_backends="cpu",
        scaling_cpu_devices="1,2,3,6,24",
        scaling_gpu_devices="1,2,3,6",
        scaling_tpu_devices="1,2,3,6",
        scaling_workload="halo",
        scaling_dt=300.0,
        scaling_compile_time_max_s=30.0,
        strong_min_efficiency=0.12,
        weak_max_step_growth=2.5,
        weak_min_per_device_throughput_ratio=0.35,
        scaling_strong_grid=64,
        scaling_weak_base_grid=64,
        scaling_iterations=4,
        scaling_warmup=1,
    )
    data.update(overrides)
    return argparse.Namespace(**data)


def test_supported_cubedsphere_parallel_counts():
    for n in (1, 2, 3, 6, 24, 54, 96, 150):
        ok, reason = rpv._is_supported_cubedsphere_parallel_count(n)
        assert ok, f"expected n={n} to be supported, got: {reason}"


def test_unsupported_cubedsphere_parallel_counts():
    for n in (4, 5, 7, 8, 10, 12, 18, 48):
        ok, _ = rpv._is_supported_cubedsphere_parallel_count(n)
        assert not ok, f"expected n={n} to be unsupported"


def test_scaling_suite_supports_tpu_backend(monkeypatch, tmp_path):
    args = _base_args(
        scaling_backends="tpu",
        scaling_tpu_devices="1,6",
    )
    host_info = {"device_counts": {"cpu": 1, "gpu": 0, "tpu": 8}}

    calls = []

    def _fake_run_scaling_point(
        *,
        args,
        output_dir,
        backend,
        case_type,
        n_devices,
        grid_size,
    ):
        calls.append((backend, case_type, n_devices, grid_size))
        base_ms = 10.0
        # Good strong-scaling behavior for non-baseline points.
        step_ms = base_ms if n_devices == 1 else base_ms / min(float(n_devices), 6.0)
        return {
            "backend": backend,
            "case_type": case_type,
            "n_devices_requested": int(n_devices),
            "grid_size": int(grid_size),
            "command_status": "pass",
            "command_returncode": 0,
            "elapsed_s": 0.1,
            "log": str(output_dir / "logs" / "fake.log"),
            "worker": {
                "status": "pass",
                "compile_time_s": 1.0,
                "steady_ms_per_step": float(step_ms),
                "throughput_per_device_mcells_s": 1.0,
                "throughput_global_mcells_s": float(n_devices),
                "throughput_per_rank_mcells_s": float(n_devices),
                "n_devices": int(n_devices),
            },
        }

    monkeypatch.setattr(rpv, "_run_scaling_point", _fake_run_scaling_point)

    result = rpv._run_scaling_suite(args, tmp_path, host_info)
    assert result["status"] == "pass"
    assert len(result["backends"]) == 1
    backend = result["backends"][0]
    assert backend["backend"] == "tpu"
    assert backend["status"] == "pass"
    assert backend["requested_devices"] == [1, 6]
    assert calls
    assert all(c[0] == "tpu" for c in calls)


def test_scaling_suite_skips_tpu_when_unavailable(tmp_path):
    args = _base_args(scaling_backends="tpu")
    host_info = {"device_counts": {"cpu": 1, "gpu": 0, "tpu": 0}}
    result = rpv._run_scaling_suite(args, tmp_path, host_info)
    assert result["status"] == "skipped"
    assert len(result["backends"]) == 1
    backend = result["backends"][0]
    assert backend["backend"] == "tpu"
    assert backend["status"] == "skipped"
    assert "No local TPU devices detected" in backend["reason"]

"""Direct controls for the shared certified-card gradient gate."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_gradient_gate.py"
)
_GPU_SCRIPT = _SCRIPT.with_name("nemo_testcase_l2_gyre_gradgate_gpu_rows.sh")
_SPEC = importlib.util.spec_from_file_location("gyre_gradient_gate_tested", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
G = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = G
_SPEC.loader.exec_module(G)


def test_broken_vjp_plant_flips_passing_row():
    base = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    direction = jnp.asarray([3.0, -1.0], dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction)

    def loss(value):
        return jnp.sum(value * value)

    control = G.directional_derivatives(loss, base, direction)
    planted = G.directional_derivatives(loss, base, direction, plant=True)
    assert control["status"] == "CONFIRMED"
    assert planted["status"] == "REFUTED"
    assert planted["reverse"] == pytest.approx(2.0 * planted["forward"])


def test_scalar_line_reverse_equals_full_vjp_dot_direction():
    base = jnp.asarray([0.2, -0.4, 0.7], dtype=jnp.float64)
    direction = jnp.asarray([0.5, -0.1, 0.3], dtype=jnp.float64)

    def loss(value):
        return jnp.sum(jnp.sin(value) + value**3)

    full_gradient = jnp.asarray(jax.grad(loss)(base))
    expected = float(jnp.vdot(full_gradient, direction))
    row = G.directional_derivatives(loss, base, direction)
    assert row["reverse"] == pytest.approx(expected, rel=1.0e-13)


def test_scalar_line_supports_swg_three_field_pytree_and_plant():
    base = G.SWGPerturbation(*(
        jnp.asarray([value], dtype=jnp.float64)
        for value in (0.2, -0.4, 0.7)
    ))
    direction = G.SWGPerturbation(*(
        jnp.asarray([value], dtype=jnp.float64)
        for value in (0.5, -0.1, 0.3)
    ))

    def loss(value):
        return sum(jnp.sum(leaf**2) for leaf in value)

    control = G.directional_derivatives(loss, base, direction)
    planted = G.directional_derivatives(loss, base, direction, plant=True)
    assert control["status"] == "CONFIRMED"
    assert planted["status"] == "REFUTED"
    assert planted["reverse"] == pytest.approx(2.0 * planted["forward"])


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("none", 700.0),
        ("uniform", 110.0),
        ("binomial", 100.0 + 10.0 * 19.0 / 3.0),
    ],
)
def test_registered_360_step_memory_extrapolation(schedule, expected):
    assert G.extrapolate_360(schedule, 100.0, 110.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("none", 120.0),
        ("uniform", 110.0),
        ("binomial", 100.0 + 10.0 * 38.0 / 27.0),
    ],
)
def test_registered_swg_1440_step_memory_extrapolation(schedule, expected):
    assert G.extrapolate_memory(
        schedule, 720, 1440, 100.0, 110.0) == pytest.approx(expected)


def test_volume_weighted_mse_ignores_zero_volume_cells():
    pred = jnp.asarray([1.0, 3.0, 99.0], dtype=jnp.float64)
    target = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
    volume = jnp.asarray([1.0, 3.0, 0.0], dtype=jnp.float64)
    assert float(G.volume_weighted_mse(pred, target, volume)) == pytest.approx(3.25)


def test_schedule_primal_contract_records_bits_and_refuses_large_drift():
    exact = G.compare_scheduled_primal(
        (jnp.asarray([1.0]),), (jnp.asarray([1.0]),))
    assert exact["within_contract"]
    assert exact["bitwise"]

    roundoff = G.compare_scheduled_primal(
        (jnp.asarray([1.0]),), (jnp.asarray([1.0 + 1.0e-12]),))
    assert roundoff["within_contract"]
    assert not roundoff["bitwise"]

    drift = G.compare_scheduled_primal(
        (jnp.asarray([1.0]),), (jnp.asarray([1.0 + 1.0e-6]),))
    assert not drift["within_contract"]
    assert not drift["bitwise"]


def test_nemo_literal_dissl_guard_preserves_primal_bits_and_zero_vjp():
    """The approved production guard changes only the zero-TKE transpose."""
    values = jnp.asarray(
        [4.0, 0.0, np.nextafter(0.0, 1.0), -1.0, np.nan],
        dtype=jnp.float64,
    )
    divisors = jnp.asarray([2.0, 3.0, 5.0, 7.0, 11.0], dtype=jnp.float64)

    def reverse_safe(value, divisor):
        positive = value > 0.0
        safe = jnp.where(positive, value, 1.0)
        safe_divisor = jnp.where(positive, divisor, 1.0)
        return jnp.where(positive, jnp.sqrt(safe) / safe_divisor, 0.0)

    old_primal = np.asarray(jnp.sqrt(values) / divisors)
    new_primal = np.asarray(reverse_safe(values, divisors))
    old_finite = np.isfinite(old_primal)
    np.testing.assert_array_equal(
        new_primal[old_finite].view(np.uint64),
        old_primal[old_finite].view(np.uint64),
    )

    _, pullback = jax.vjp(reverse_safe, values, divisors)
    tke_vjp, divisor_vjp = map(
        np.asarray, pullback(jnp.ones_like(values)))
    assert np.isfinite(tke_vjp[1])
    assert tke_vjp[1] == 0.0
    assert np.isfinite(divisor_vjp[1])
    assert divisor_vjp[1] == 0.0

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    _, zero_pullback = jax.vjp(reverse_safe, zero, zero)
    zero_tke_vjp, zero_divisor_vjp = zero_pullback(
        jnp.asarray(1.0, dtype=jnp.float64))
    assert float(zero_tke_vjp) == 0.0
    assert float(zero_divisor_vjp) == 0.0

    source = inspect.getsource(tke_mod.tke_vertical_mixing)
    assert "_dissl_sqrt_input = jnp.where(" in source
    assert "_dissl_divisor = jnp.where(" in source
    assert "jnp.sqrt(_dissl_sqrt_input)" in source


def test_resource_cli_requires_a_schedule():
    with pytest.raises(SystemExit):
        G.parse_args([
            "resources", "--card", "GYRE-zco", "--steps", "1",
            "--expect-commit", "deadbeef",
            "--output", "/tmp/unused-gyre-gradient-row.json",
        ])


def test_gpu_cli_can_select_native_transcendentals_explicitly():
    args = G.parse_args([
        "agreement", "--card", "SWG", "--steps", "1",
        "--transcendentals", "native",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert args.transcendentals == "native"
    assert args.card == "SWG"


def test_card_dispatch_raises_on_unknown_before_building(monkeypatch):
    monkeypatch.setattr(G, "configure_runtime", lambda _policy: None)
    with pytest.raises(ValueError, match="unknown gradient card"):
        G.build_context("not-a-card")


def test_resource_classifier_is_fail_closed():
    valid = {
        "cold_wall_s": 2.0,
        "warm_wall_s": 1.0,
        "baseline_mib": 100.0,
        "peak_mib": 101.0,
        "samples": 2,
        "reverse_transform": G.COMPILED_REVERSE_TRANSFORM,
        "warm_reverse_timer_span": G.WARM_REVERSE_TIMER_SPAN,
    }
    assert G.classify_resources(1.0, True, True, **valid) == "CONFIRMED"
    assert G.classify_resources(
        float("nan"), True, True, **valid) == "REFUTED"
    assert G.classify_resources(1.0, False, True, **valid) == "REFUTED"
    assert G.classify_resources(1.0, True, False, **valid) == "REFUTED"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "warm_wall_s": 0.0}) == "REFUTED"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "peak_mib": 99.0}) == "VACUOUS"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "peak_mib": 100.0}) == "VACUOUS"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "samples": 1}) == "VACUOUS"
    assert G.classify_resources(
        1.0,
        True,
        True,
        **{
            **valid,
            "reverse_transform": None,
            "warm_reverse_timer_span": None,
        },
    ) == "RETRACTED"


def test_device_memory_stats_requires_allocator_peak_and_live_bytes():
    class Device:
        def memory_stats(self):
            return {
                "bytes_in_use": 2**20,
                "peak_bytes_in_use": 3 * 2**20,
                "num_allocs": np.int64(7),
            }

    assert G.device_memory_stats(Device()) == {
        "bytes_in_use": 2**20,
        "peak_bytes_in_use": 3 * 2**20,
        "num_allocs": 7,
    }

    class MissingPeak:
        def memory_stats(self):
            return {"bytes_in_use": 2**20}

    with pytest.raises(RuntimeError, match="peak_bytes_in_use"):
        G.device_memory_stats(MissingPeak())


def _gpu_resource_row(schedule, jax_increment, smi_increment):
    return {
        "card_selector": "SWG",
        "steps": 24,
        "backend": "gpu",
        "schedule": schedule,
        "status": "CONFIRMED",
        "loss": 1.0,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "cold_compile_and_reverse_wall_seconds": 2.0,
        "warm_reverse_wall_seconds": 1.0,
        "warm_reverse_timer_span": G.WARM_REVERSE_TIMER_SPAN,
        "reverse_transform": G.COMPILED_REVERSE_TRANSFORM,
        "memory_baseline_mib": 100.0,
        "memory_peak_mib": 100.0 + jax_increment,
        "memory_samples": 2,
        "jax_device_memory_increment_mib": jax_increment,
        "jax_device_peak_growth_mib": jax_increment,
        "nvidia_smi_memory_increment_mib": smi_increment,
        "provenance": {"commit": "deadbeef"},
    }


def test_gpu_resource_suite_requires_both_orderings_and_plant_flips():
    rows = [
        _gpu_resource_row("none", 900.0, 1000.0),
        _gpu_resource_row("uniform", 200.0, 250.0),
        _gpu_resource_row("binomial", 400.0, 500.0),
    ]
    control = G.classify_resource_suite(rows)
    planted = G.classify_resource_suite(rows, plant=True)
    assert control["status"] == "CONFIRMED"
    assert control["ordering"] == {"jax_allocator": True, "nvidia_smi": True}
    assert planted["status"] == "REFUTED"
    assert planted["ordering"] == {
        "jax_allocator": False,
        "nvidia_smi": False,
    }

    rows[1]["jax_device_peak_growth_mib"] = 0.0
    vacuous = G.classify_resource_suite(rows)
    assert vacuous["status"] == "VACUOUS"
    assert "peak did not advance" in vacuous["reason"]


def test_gpu_resource_suite_retracts_rows_without_timer_stamps():
    rows = [
        {
            "card_selector": "SWG",
            "steps": 24,
            "backend": "gpu",
            "schedule": schedule,
            "status": "CONFIRMED",
            "memory_increment_mib": increment,
        }
        for schedule, increment in (
            ("none", 0.0), ("uniform", 6.0), ("binomial", 4.0))
    ]
    row = G.classify_resource_suite(rows)
    assert row["status"] == "RETRACTED"
    assert "compiled-transform timer contract" in row["reason"]


def _cpu_scaling_row(schedule, steps, seconds):
    return {
        "card_selector": "SWG",
        "steps": steps,
        "backend": "cpu",
        "schedule": schedule,
        "status": "CONFIRMED",
        "loss": 1.0,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "cold_compile_and_reverse_wall_seconds": seconds + 1.0,
        "warm_reverse_wall_seconds": seconds,
        "warm_reverse_timer_span": G.WARM_REVERSE_TIMER_SPAN,
        "reverse_transform": G.COMPILED_REVERSE_TRANSFORM,
        "memory_baseline_mib": 100.0,
        "memory_peak_mib": 101.0,
        "memory_samples": 2,
        "provenance": {"commit": "deadbeef"},
    }


def test_cpu_scaling_suite_gates_frozen_predictions_and_constant_time_plant():
    rows = [
        _cpu_scaling_row("uniform", steps, 0.01 * steps + 0.1)
        for steps in G.SWG_SCALING_STEPS
    ]
    control = G.classify_scaling_suite(rows)
    planted = G.classify_scaling_suite(rows, plant=True)
    assert control["status"] == "CONFIRMED"
    assert control["fit_seconds_per_step"] == pytest.approx(0.01)
    assert control["checks"] == {
        "positive_fitted_seconds_per_step": True,
        "time_1440_exceeds_time_720": True,
        "uniform_1440_to_720_ratio_in_preregistered_band": True,
    }
    assert planted["status"] == "REFUTED"
    assert not any(planted["checks"].values())
    args = G.parse_args([
        "scaling-suite", "--card", "SWG", "--steps", "1440",
        "--schedule", "uniform", "--expect-commit", "deadbeef",
        "--output", "/tmp/unused-scaling-suite.json",
        *sum((["--row", f"/tmp/{step}.json"]
              for step in G.SWG_SCALING_STEPS), []),
    ])
    assert args.plant is None
    assert len(args.row) == 5


def test_resource_suite_executable_retracts_legacy_timing_rows(
        tmp_path, monkeypatch):
    paths = []
    for schedule in G.SCHEDULES:
        row = _gpu_resource_row(schedule, 10.0, 10.0)
        row.pop("reverse_transform")
        row.pop("warm_reverse_timer_span")
        path = tmp_path / f"legacy_{schedule}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "legacy_suite.json"
    monkeypatch.setattr(
        G, "require_measurement_stamp",
        lambda _commit: {"commit": "deadbeef", "dirty": False},
    )
    argv = [
        "resource-suite", "--card", "SWG", "--steps", "24",
        "--expect-commit", "deadbeef", "--output", str(output),
    ]
    for path in paths:
        argv.extend(("--row", str(path)))
    assert G.main(argv) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "RETRACTED"


def test_swg_cli_admits_registered_timing_rows_and_refuses_other_counts():
    agreement = G.parse_args([
        "agreement", "--card", "SWG", "--steps", "1440",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert agreement.steps == 1440
    for steps in (24, 96, 360, 720, 1440):
        args = G.parse_args([
            "resources", "--card", "SWG", "--steps", str(steps),
            "--schedule", "uniform", "--expect-commit", "deadbeef",
            "--output", "/tmp/unused-resource-row.json",
        ])
        assert args.steps == steps
    with pytest.raises(RuntimeError, match="must be one of"):
        G.measure_resources("SWG", 95, "uniform")


def test_gpu_runner_disables_preallocation_and_covers_direct_1440_rows():
    source = _GPU_SCRIPT.read_text(encoding="utf-8")
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in source
    assert "env -u XLA_PYTHON_CLIENT_ALLOCATOR" in source
    assert "XLA_PYTHON_CLIENT_ALLOCATOR=platform" not in source
    assert "for steps in 1 24 720 1440" in source
    assert "for steps in 24 720 1440" in source
    assert "--plant uniform_label_uses_none" in source
    assert "LAST_ROW_STATUS" in source
    assert "LAST_SUITE_STATUS" in source
    assert "SWG_GRADGATE_GPU_SUITE" in source

    self_test = subprocess.run(
        ["bash", str(_GPU_SCRIPT), "--self-test"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert self_test.returncode == 0, self_test.stderr
    assert (
        "statuses=REFUTED VACUOUS RETRACTED CONFIRMED" in self_test.stdout
    )


def test_gpu_allocator_contract_refuses_platform(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    G.require_gpu_allocator_instrument()
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    with pytest.raises(RuntimeError, match="refuse.*platform"):
        G.require_gpu_allocator_instrument()


def test_gpu_resource_two_pass_control_uses_fresh_sampler_on_cpu(monkeypatch):
    instances = []
    events = []

    class SingleUseCPUSampler:
        def __init__(self, backend):
            assert backend == "gpu"
            self.started = 0
            self.stopped = 0
            self.baseline_mib = 1.0
            self.peak_mib = 2.0
            self.samples = 2
            instances.append(self)

        def start(self):
            assert self.started == 0
            self.started += 1
            events.append(f"start-{len(instances) - 1}")

        def stop(self):
            assert self.started == 1
            assert self.stopped == 0
            self.stopped += 1
            events.append(f"stop-{len(instances) - 1}")

    monkeypatch.setattr(G, "PeakMemorySampler", SingleUseCPUSampler)
    reverse = jax.jit(jax.value_and_grad(lambda x: x**2))

    def after_cold():
        events.append("stats-after-cold")
        return {"peak_bytes_in_use": 123}

    cold, warm, cold_stats = G.measure_cold_and_warm_reverse(
        reverse, jnp.asarray(3.0), "gpu", after_cold=after_cold)

    assert len(instances) == 2
    assert cold.sampler is instances[0]
    assert warm.sampler is instances[1]
    assert all(item.started == item.stopped == 1 for item in instances)
    assert events == [
        "start-0", "stop-0", "stats-after-cold", "start-1", "stop-1"]
    assert cold_stats == {"peak_bytes_in_use": 123}
    assert float(cold.gradient) == pytest.approx(6.0)
    assert float(warm.gradient) == pytest.approx(6.0)


def test_compiled_reverse_warm_call_does_not_retrace():
    traces = 0

    def loss(x):
        nonlocal traces
        traces += 1
        return x**2

    reverse = G.build_compiled_reverse(loss)
    reverse(jnp.asarray(2.0))
    reverse(jnp.asarray(3.0))
    assert traces == 1


def test_resource_dispatch_rejects_unknown_card_before_step_lookup():
    with pytest.raises(ValueError, match="unknown gradient card"):
        G.measure_resources("NOT-A-CARD", 24, "uniform")


def test_measurement_policy_labels_refuse_cross_backend_combinations():
    assert G.measurement_policy_label("cpu", "libm") == (
        "CPU, certified fp64/libm bit-exact policy")
    assert G.measurement_policy_label("gpu", "native") == (
        "GPU, XLA transcendentals — not the certified bit-exact policy")
    with pytest.raises(ValueError, match="unsupported gradient"):
        G.measurement_policy_label("gpu", "libm")
    with pytest.raises(ValueError, match="unsupported gradient"):
        G.measurement_policy_label("cpu", "native")

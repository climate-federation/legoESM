"""Direct controls for the shared certified-card gradient gate."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import NamedTuple

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


def test_nonfinite_agreement_row_is_not_labeled_nonzero():
    row = G.classify_agreement(np.nan, np.nan, np.nan)
    assert row["finite"] is False
    assert row["nonzero"] is False
    assert row["status"] == "REFUTED"


def test_scalar_line_reverse_equals_full_vjp_dot_direction():
    base = jnp.asarray([0.2, -0.4, 0.7], dtype=jnp.float64)
    direction = jnp.asarray([0.5, -0.1, 0.3], dtype=jnp.float64)

    def loss(value):
        return jnp.sum(jnp.sin(value) + value**3)

    full_gradient = jnp.asarray(jax.grad(loss)(base))
    expected = float(jnp.vdot(full_gradient, direction))
    row = G.directional_derivatives(loss, base, direction)
    assert row["reverse"] == pytest.approx(expected, rel=1.0e-13)
    assert row["gradient_dot_abs_sum"] == pytest.approx(
        float(jnp.sum(jnp.abs(full_gradient * direction))))
    assert row["gradient_direction_norm_product"] == pytest.approx(
        float(jnp.linalg.norm(full_gradient) * jnp.linalg.norm(direction)))
    assert row["rev_fwd_scale_aware_error"] == pytest.approx(0.0, abs=1.0e-15)


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


def test_explicit_stepwise_vjp_matches_whole_window_gradient(monkeypatch):
    class Field(NamedTuple):
        data: jax.Array

        def replace(self, **changes):
            return self._replace(**changes)

    class State(NamedTuple):
        T: Field

    class Context(NamedTuple):
        initial_state: State
        card: object
        model: object
        surface_forcings: object

    class Card(NamedTuple):
        dt_s: float

    class Model:
        def _step_jitted(self, state, _dt_s, freshwater, surface):
            return State(Field(
                state.T.data * (1.0 + freshwater) + 0.01 * surface))

    def forcing(_card, state, kt):
        return 0.1 * kt, state.T.data**2

    context = Context(
        State(Field(jnp.asarray([0.2, -0.4], dtype=jnp.float64))),
        Card(1.0),
        Model(),
        forcing,
    )
    monkeypatch.setattr(
        G,
        "loss_at_state",
        lambda _reference, state: jnp.sum(jnp.sin(state.T.data)),
    )
    base = jnp.asarray([0.01, -0.02], dtype=jnp.float64)
    direction = jnp.asarray([0.6, -0.8], dtype=jnp.float64)
    row = G.explicit_stepwise_directional_derivatives(
        context, None, base, direction, 3)
    planted = G.explicit_stepwise_directional_derivatives(
        context, None, base, direction, 3, plant=True)

    def whole_window(value):
        state = G._apply_initial_perturbation(context, value)
        for kt in range(1, 4):
            state = G._production_step(context, state, kt)
        return G.loss_at_state(None, state)

    expected_gradient = jax.grad(whole_window)(base)
    assert row["reverse"] == pytest.approx(
        float(jnp.vdot(expected_gradient, direction)), rel=1e-13)
    assert row["gradient_dot_abs_sum"] == pytest.approx(
        float(jnp.sum(jnp.abs(expected_gradient * direction))), rel=1e-13)
    assert all(
        report["bitwise_equal"]
        for report in row["step_vjp_replay_reports"].values()
    )
    assert row["step_vjp_replays_bitwise"] is True
    assert row["step_vjp_replay_used_as_primal"] is False
    assert all(
        report["bitwise_equal"]
        for report in row["step_jvp_replay_reports"].values()
    )
    assert row["step_jvp_replays_bitwise"] is True
    assert row["step_jvp_replay_used_as_primal"] is False
    assert planted["status"] == "REFUTED"
    assert planted["reverse"] == pytest.approx(2.0 * planted["forward"])


def test_reverse_only_schedule_compares_against_forward_reference():
    base = jnp.asarray([0.2, -0.4, 0.7], dtype=jnp.float64)
    direction = jnp.asarray([0.5, -0.1, 0.3], dtype=jnp.float64)

    def forward_loss(value):
        return jnp.sum(jnp.sin(value) + value**3)

    def reverse_loss(value):
        return G.checkpointed_loop(
            lambda carry: carry,
            forward_loss(value),
            2,
            schedule="binomial",
        )

    row = G.directional_derivatives(
        forward_loss, base, direction, reverse_loss=reverse_loss)
    assert row["status"] == "CONFIRMED"


def _healthy_resource_row():
    return {
        "resource_schema": G.RESOURCE_SCHEMA,
        "reverse_transform": G.REVERSE_TRANSFORM,
        "warm_reverse_timer_span": G.WARM_TIMER_SPAN,
        "case": G.CASE,
        "schedule": "uniform",
        "backend": "cpu",
        "steps": 6,
        "gradient_finite": True,
        "gradient_nonzero": True,
        "loss": 1.0,
        "cold_compile_and_reverse_wall_seconds": 2.0,
        "warm_reverse_wall_seconds": 1.0,
        "memory_baseline_mib": 100.0,
        "memory_peak_mib": 110.0,
        "memory_increment_mib": 10.0,
        "memory_samples": 3,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("gradient_finite", False),
        ("gradient_nonzero", False),
        ("loss", np.nan),
        ("warm_reverse_wall_seconds", 0.0),
        ("memory_peak_mib", np.nan),
        ("memory_samples", 1),
    ],
)
def test_resource_rows_fail_closed(field, bad_value):
    row = _healthy_resource_row()
    row[field] = bad_value
    assert G.resource_row_passes(row) is False


def test_healthy_resource_row_passes():
    assert G.resource_row_passes(_healthy_resource_row()) is True


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
        ("none", 200.0),
        ("uniform", 110.0),
        ("binomial", 100.0 + 10.0 * 8.0 / 3.0),
    ],
)
def test_registered_60_step_memory_extrapolation(schedule, expected):
    assert G.extrapolate_steps(schedule, 100.0, 110.0, 60) == pytest.approx(
        expected)


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

    base = {"x": jnp.asarray([1.0, 2.0], dtype=jnp.float64)}
    last_bit = {"x": jnp.asarray(
        [np.nextafter(1.0, 2.0), 2.0], dtype=jnp.float64)}
    bad = {"x": jnp.asarray([1.0, 2.1], dtype=jnp.float64)}
    close = G.compare_scheduled_primal(base, last_bit)
    assert close["within_contract"] is True
    assert close["bitwise"] is False
    assert G.compare_scheduled_primal(base, bad)["within_contract"] is False


def test_state_bitwise_report_detects_last_bit_and_signed_zero():
    left = {
        "x": jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        "i": jnp.asarray([1], dtype=jnp.int32),
    }
    right = {
        "x": jnp.asarray(
            [np.nextafter(1.0, 2.0), -0.0], dtype=jnp.float64),
        "i": jnp.asarray([1], dtype=jnp.int32),
    }
    report = G.state_bitwise_report(left, right)
    assert report["bitwise_equal"] is False
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["path"] == "['x']"
    assert report["mismatches"][0]["max_abs"] == pytest.approx(
        np.nextafter(1.0, 2.0) - 1.0)
    assert G.states_bitwise_equal(left, right) is False
    assert G.states_bitwise_equal(left, left) is True


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


def test_agreement_cli_defaults_to_the_per_step_composition():
    args = G.parse_args([
        "agreement", "--card", "GYRE-zco", "--steps", "1",
        "--expect-commit", "deadbeef",
        "--output", "/tmp/unused-gyre-gradient-row.json",
    ])
    assert args.schedule == "per_step"
    args = G.parse_args([
        "agreement", "--card", "GYRE-zco", "--steps", "1",
        "--schedule", "uniform", "--expect-commit", "deadbeef",
        "--output", "/tmp/unused-gyre-gradient-row.json",
    ])
    assert args.schedule == "uniform"


def test_gpu_cli_can_select_native_transcendentals_explicitly():
    args = G.parse_args([
        "agreement", "--card", "SWG", "--steps", "1",
        "--transcendentals", "native",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert args.transcendentals == "native"
    assert args.card == "SWG"


def test_per_step_composition_is_agreement_only():
    args = G.parse_args([
        "agreement", "--card", "GYRE-zco", "--steps", "6",
        "--schedule", "per_step",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert args.schedule == "per_step"
    with pytest.raises(SystemExit):
        G.parse_args([
            "resources", "--card", "GYRE-zco", "--steps", "6",
            "--schedule", "per_step",
            "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
        ])


def test_time_scaling_suite_cli_takes_rows_without_steps():
    args = G.parse_args([
        "time-scaling-suite", "--card", "GYRE-zco",
        "--row", "/tmp/a.json", "--row", "/tmp/b.json",
        "--plant", "constant_time",
        "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
    ])
    assert [str(path) for path in args.row] == ["/tmp/a.json", "/tmp/b.json"]
    with pytest.raises(SystemExit):
        G.parse_args([
            "time-scaling-suite", "--card", "GYRE-zco", "--steps", "6",
            "--row", "/tmp/a.json", "--row", "/tmp/b.json",
            "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
        ])
    with pytest.raises(SystemExit):
        G.parse_args([
            "time-scaling-suite", "--card", "GYRE-zco", "--row", "/tmp/a.json",
            "--expect-commit", "deadbeef", "--output", "/tmp/unused-row.json",
        ])


def test_card_dispatch_raises_on_unknown_before_building(monkeypatch):
    monkeypatch.setattr(G, "configure_runtime", lambda _policy: None)
    with pytest.raises(ValueError, match="unknown gradient card"):
        G.build_context("not-a-card")


def test_peak_sampler_propagates_worker_failure(monkeypatch):
    sampler = G.PeakMemorySampler("cpu", interval_s=0.001)

    def fail():
        raise OSError("planted sampler read failure")

    sampler._read = fail
    sampler.start()
    sampler._thread.join(timeout=1.0)
    with pytest.raises(RuntimeError, match="peak-memory sampler failed"):
        sampler.stop()


def test_measurement_provenance_is_rechecked_after_row(monkeypatch):
    answers = iter(["changed-commit"])
    monkeypatch.setattr(G, "_git_output", lambda *_args: next(answers))
    with pytest.raises(RuntimeError, match="measurement commit"):
        G.require_unchanged_measurement_stamp(
            {"commit": "opening-commit", "dirty": False})


def test_cli_refuses_existing_output_before_measurement(tmp_path):
    output = tmp_path / "retained.json"
    output.write_text("retained\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite evidence"):
        G.main([
            "agreement", "--card", "GYRE-zco", "--steps", "1",
            "--schedule", "uniform",
            "--expect-commit", "deadbeef", "--output", str(output),
        ])
    assert output.read_text(encoding="utf-8") == "retained\n"


def test_resource_classifier_is_fail_closed():
    valid = {
        "cold_wall_s": 2.0,
        "warm_wall_s": 1.0,
        "baseline_mib": 100.0,
        "peak_mib": 101.0,
        "samples": 2,
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


def test_historical_uncompiled_resource_timing_is_executable_retraction():
    old = _healthy_resource_row()
    old.pop("resource_schema")
    old.pop("reverse_transform")
    old.pop("warm_reverse_timer_span")
    assert G.classify_resource_row(old) == "RETRACTED"
    assert G.resource_row_passes(old) is False


def test_time_scaling_classifier_and_constant_time_plant():
    six = _healthy_resource_row()
    sixty = {**six, "steps": 60, "warm_reverse_wall_seconds": 8.0}
    control = G.classify_time_scaling([six, sixty])
    planted = G.classify_time_scaling([six, sixty], plant=True)
    assert control["status"] == "CONFIRMED"
    assert control["slope_seconds_per_step"] > 0.0
    assert planted["status"] == "REFUTED"
    assert planted["slope_seconds_per_step"] == pytest.approx(0.0, abs=1e-15)


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


def test_gpu_resource_ordering_plant_flips():
    rows = [
        _gpu_resource_row("none", 900.0, 1000.0),
        _gpu_resource_row("uniform", 200.0, 250.0),
        _gpu_resource_row("binomial", 400.0, 500.0),
    ]
    assert G.classify_gpu_resource_suite(rows)["status"] == "CONFIRMED"
    assert G.classify_gpu_resource_suite(rows, plant=True)["status"] == "REFUTED"


def test_gpu_resource_suite_marks_old_pool_rows_vacuous():
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
    assert row["status"] == "VACUOUS"
    assert "jax_allocator" in row["reason"]


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


def test_gyre_card_admits_its_registered_sixty_step_rows():
    assert G.CARD_STEPS["GYRE-zco"] == (1, 6, 60)
    assert G.RESOURCE_STEPS["GYRE-zco"] == (1, 6, 60)
    with pytest.raises(RuntimeError, match="must be one of"):
        G.measure_resources("GYRE-zco", 7, "uniform")


def test_gpu_runner_disables_preallocation_and_covers_direct_1440_rows():
    source = _GPU_SCRIPT.read_text(encoding="utf-8")
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in source
    assert "env -u XLA_PYTHON_CLIENT_ALLOCATOR" in source
    assert "XLA_PYTHON_CLIENT_ALLOCATOR=platform" not in source
    assert "for steps in 1 24 720 1440" in source
    assert "for steps in 24 720 1440" in source
    assert "--plant uniform_label_uses_none" in source


def test_gpu_runner_keeps_the_gyre_card_rows():
    source = _GPU_SCRIPT.read_text(encoding="utf-8")
    assert "for steps in 1 6" in source
    assert 'agreement "$steps" per_step' in source
    assert 'args+=(--schedule "$schedule")' in source
    assert "gradgate_rounds/round7" in source
    assert "swggg_rounds/round3" in source


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


def test_resource_two_pass_control_uses_fresh_sampler(monkeypatch):
    instances = []

    class SingleUseSampler:
        def __init__(self, _backend):
            self.started = 0
            self.stopped = 0
            self.baseline_mib = 1.0
            self.peak_mib = 2.0
            self.samples = 2
            instances.append(self)

        def start(self):
            assert self.started == 0
            self.started += 1

        def stop(self):
            assert self.started == 1 and self.stopped == 0
            self.stopped += 1

    monkeypatch.setattr(G, "PeakMemorySampler", SingleUseSampler)
    reverse = jax.jit(jax.value_and_grad(lambda x: x**2))
    cold, warm, _ = G.measure_cold_and_warm_reverse(
        reverse, jnp.asarray(3.0), "cpu")
    assert cold.sampler is instances[0]
    assert warm.sampler is instances[1]
    assert all(item.started == item.stopped == 1 for item in instances)


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

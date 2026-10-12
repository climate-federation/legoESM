"""Direct controls for the shared certified-card gradient gate."""

from __future__ import annotations

import importlib.util
import inspect
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
    }
    assert G.classify_resources(1.0, True, True, **valid) == "CONFIRMED"
    assert G.classify_resources(
        float("nan"), True, True, **valid) == "REFUTED"
    assert G.classify_resources(1.0, False, True, **valid) == "REFUTED"
    assert G.classify_resources(1.0, True, False, **valid) == "REFUTED"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "warm_wall_s": 0.0}) == "REFUTED"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "peak_mib": 99.0}) == "REFUTED"
    assert G.classify_resources(
        1.0, True, True, **{**valid, "samples": 1}) == "REFUTED"


def test_measurement_policy_labels_refuse_cross_backend_combinations():
    assert G.measurement_policy_label("cpu", "libm") == (
        "CPU, certified fp64/libm bit-exact policy")
    assert G.measurement_policy_label("gpu", "native") == (
        "GPU, XLA transcendentals — not the certified bit-exact policy")
    with pytest.raises(ValueError, match="unsupported gradient"):
        G.measurement_policy_label("gpu", "libm")
    with pytest.raises(ValueError, match="unsupported gradient"):
        G.measurement_policy_label("cpu", "native")

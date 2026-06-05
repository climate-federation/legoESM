"""Stage-A2 Experiment harness: the contract + the gradient-check rung."""

from __future__ import annotations

import jax
import pytest

from legoesm.experiments import (
    Experiment,
    ExperimentResult,
    GradientCheckExperiment,
    MetricCheck,
    finite_difference_grad,
    relative_grad_error,
    single_column_thermo_gradient_check,
)


# ----------------------------- MetricCheck -----------------------------
def test_metric_check_below() -> None:
    passed, _ = MetricCheck(reference=1e-3, kind="below").evaluate(5e-4)
    assert passed
    passed, _ = MetricCheck(reference=1e-3, kind="below").evaluate(2e-3)
    assert not passed


def test_metric_check_above() -> None:
    assert MetricCheck(reference=0.9, kind="above").evaluate(0.95)[0]
    assert not MetricCheck(reference=0.9, kind="above").evaluate(0.5)[0]


def test_metric_check_close() -> None:
    assert MetricCheck(reference=1.0, kind="close", rtol=1e-3).evaluate(1.0005)[0]
    assert not MetricCheck(reference=1.0, kind="close", rtol=1e-6).evaluate(1.01)[0]


def test_metric_check_bad_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        MetricCheck(reference=0.0, kind="sideways").evaluate(0.0)


# ----------------------------- Experiment ------------------------------
class _Toy(Experiment):
    name = "toy"

    def __init__(self, value: float) -> None:
        self._value = value

    def run(self) -> dict:
        return {"err": self._value, "extra": 42.0}

    @property
    def checks(self) -> dict:
        return {"err": MetricCheck(reference=1e-3, kind="below")}


def test_experiment_evaluate_pass_and_fail() -> None:
    good = _Toy(1e-4).evaluate()
    assert isinstance(good, ExperimentResult) and good.passed
    assert good.metrics["extra"] == 42.0  # ungated metrics still recorded
    assert not _Toy(1.0).evaluate().passed


class _NoChecks(Experiment):
    def run(self) -> dict:
        return {"x": 0.0}

    @property
    def checks(self) -> dict:
        return {}


def test_active_experiment_with_no_checks_fails_closed() -> None:
    exp = _NoChecks()
    exp.name, exp.status = "empty_active", "active"
    res = exp.evaluate()
    assert not res.passed  # gates nothing -> must NOT pass open


def test_proposed_experiment_with_no_checks_does_not_fail() -> None:
    exp = _NoChecks()
    exp.name, exp.status = "empty_proposed", "proposed"
    assert exp.evaluate().passed  # proposed = runs, non-gating


def test_unknown_status_raises() -> None:
    exp = _NoChecks()
    exp.status = "maybe"
    with pytest.raises(ValueError, match="status"):
        exp.evaluate()


def test_experiment_missing_metric_fails_closed() -> None:
    class Missing(Experiment):
        name = "missing"

        def run(self) -> dict:
            return {}  # does not produce the gated metric

        @property
        def checks(self) -> dict:
            return {"err": MetricCheck(reference=1.0)}

    res = Missing().evaluate()
    assert not res.passed
    assert "did not produce" in res.report["err"]


# --------------------- gradient check (needs x64) ----------------------
_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="finite-difference gradient check needs JAX_ENABLE_X64=1",
)


@_needs_x64
def test_finite_difference_grad_matches_known() -> None:
    import jax.numpy as jnp

    # f(x) = sum(x**2) -> grad = 2x
    g = finite_difference_grad(lambda x: jnp.sum(x ** 2), jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(g, jnp.array([2.0, 4.0, 6.0]), atol=1e-5)


@_needs_x64
def test_relative_grad_error_small_for_smooth_fn() -> None:
    import jax.numpy as jnp

    err = relative_grad_error(lambda x: jnp.sum(jnp.sin(x)), jnp.array([0.3, 1.1, 2.0]))
    assert err < 1e-6


@_needs_x64
def test_gradient_check_experiment_passes_on_polynomial() -> None:
    import jax.numpy as jnp

    exp = GradientCheckExperiment(
        "poly", lambda x: jnp.sum(x ** 3), jnp.array([0.5, 1.0, 1.5]), tol=1e-6
    )
    assert exp.evaluate().passed


def test_gradient_check_fails_closed_without_x64(monkeypatch) -> None:
    """Without x64, a clear precondition error — not a misleading numeric failure."""
    import jax.numpy as jnp

    import legoesm.experiments.gradient_check as gc

    monkeypatch.setattr(gc.jax.config, "read", lambda _k: False)

    calls = []

    def f(x):
        calls.append(1)  # must NOT be reached — guard is fail-fast
        return jnp.sum(x ** 2)

    with pytest.raises(RuntimeError, match="64-bit|JAX_ENABLE_X64"):
        gc.relative_grad_error(f, jnp.array([1.0, 2.0]))
    assert not calls, "target fn evaluated in float32 before the x64 guard fired"


@_needs_x64
def test_single_column_thermo_gradient_rung_passes() -> None:
    """The first wired Experiment rung runs and passes (A2 exit criterion)."""
    result = single_column_thermo_gradient_check().evaluate()
    assert result.passed, result.summary()
    assert result.metrics["grad_rel_error"] < 1e-6

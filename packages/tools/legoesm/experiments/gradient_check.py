"""Finite-difference gradient checks as ``Experiment`` rungs (Stage A2).

The cheapest guardian of the project's defining invariant (D1: end-to-end
differentiability): take a scalar function of a state/column, and confirm its
reverse-mode autodiff gradient matches a central finite-difference gradient to a
tolerance.  Wrapped as an :class:`Experiment` so it carries a reference tolerance
and a pass/fail verdict like every other matrix cell.

Run with ``JAX_ENABLE_X64=1`` — the finite-difference reference needs 64-bit.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from legoesm.experiments.abstract import Experiment, MetricCheck


def _require_x64() -> None:
    """Fail CLOSED (clear precondition error) when 64-bit is off.

    JAX silently truncates ``jnp.float64`` to float32 when ``jax_enable_x64`` is
    disabled, which would turn a finite-difference check into a *misleading
    numerical* failure (~1e-3 relative error) instead of a precondition failure.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "finite-difference gradient checks require 64-bit precision; run with "
            "JAX_ENABLE_X64=1 (float32 finite differences are unreliable)."
        )


def finite_difference_grad(
    f: Callable, x, eps: float = 1e-4
):
    """Central finite-difference gradient of scalar ``f`` at ``x``.

    ``x`` may be a scalar or an array; the returned gradient has ``x``'s shape.
    Each element is perturbed by ``±eps`` independently (fine for the small
    columns these checks run on — this is a harness, not a hot loop).
    Requires ``JAX_ENABLE_X64=1``.
    """
    _require_x64()
    x = jnp.asarray(x, dtype=jnp.float64)
    flat = x.reshape(-1)
    grads = []
    for i in range(flat.size):
        plus = flat.at[i].add(eps).reshape(x.shape)
        minus = flat.at[i].add(-eps).reshape(x.shape)
        grads.append((f(plus) - f(minus)) / (2.0 * eps))
    return jnp.asarray(grads, dtype=jnp.float64).reshape(x.shape)


def relative_grad_error(f: Callable, x, eps: float = 1e-4) -> float:
    """Max relative error between autodiff and finite-difference gradients of ``f``.

    ``max|g_ad - g_fd| / (max|g_fd| + tiny)`` — a single scalar a tolerance can
    gate on.
    """
    _require_x64()  # FIRST — before f is ever evaluated, in either half
    x = jnp.asarray(x, dtype=jnp.float64)
    g_ad = jax.grad(f)(x)
    g_fd = finite_difference_grad(f, x, eps=eps)
    num = jnp.max(jnp.abs(g_ad - g_fd))
    den = jnp.max(jnp.abs(g_fd)) + 1e-30
    return float(num / den)


class GradientCheckExperiment(Experiment):
    """An ``Experiment`` that gates on the autodiff-vs-FD gradient error of ``fn``."""

    def __init__(
        self,
        name: str,
        fn: Callable,
        x0,
        *,
        tol: float = 1e-5,
        eps: float = 1e-4,
        description: str = "",
        status: str = "active",
    ) -> None:
        self.name = name
        self.description = description
        self.status = status
        self._fn = fn
        self._x0 = x0
        self._tol = tol
        self._eps = eps

    def run(self) -> dict[str, float]:
        return {
            "grad_rel_error": relative_grad_error(self._fn, self._x0, eps=self._eps)
        }

    @property
    def checks(self) -> dict[str, MetricCheck]:
        return {"grad_rel_error": MetricCheck(reference=self._tol, kind="below")}


def single_column_thermo_gradient_check() -> GradientCheckExperiment:
    """The first wired rung: differentiate a real thermo column end-to-end.

    ``d/dT sum(saturation_vapor_pressure(T))`` over a 20-level temperature column,
    autodiff vs central finite difference.  Exercises the shared, differentiable
    ``legoesm.thermo`` path on real model code (D1).
    """
    from legoesm.thermo import saturation_vapor_pressure

    # A 20-level column spanning a realistic tropospheric temperature range.
    t_column = jnp.linspace(220.0, 300.0, 20, dtype=jnp.float64)

    def column_saturation(t_col):
        return jnp.sum(saturation_vapor_pressure(t_col))

    return GradientCheckExperiment(
        name="single_column_saturation_grad",
        fn=column_saturation,
        x0=t_column,
        tol=1e-6,
        description=(
            "Autodiff vs central-FD gradient of column-summed saturation vapor "
            "pressure w.r.t. temperature (D1 differentiability of legoesm.thermo)."
        ),
    )

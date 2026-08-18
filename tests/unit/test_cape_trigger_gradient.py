"""The CAPE trigger must be trainable without changing a single answer.

``sigmoid`` saturates to exactly 1.0 in float64 above an argument of ~36.7, so
its derivative is exactly 0 there.  A deep-tropical column carries CAPE of a few
thousand J/kg against a threshold of tens, so every CAPE trigger sat far inside
that dead zone and ``d/d(cape_threshold)`` was exactly zero — which is why those
thresholds are declared ``tunable_tier 0`` (#1417).

They are not physically inert: in the 2026-08-16 SCM-RCE campaign ``dca``
improved its temperature-and-humidity score 65 % (6.06 -> 2.13) by tuning its
CAPE threshold alone, found by a derivative-free search *because* the gradient
could not see it.

So the fix has two obligations and this file gates both:

1. the FORWARD value is unchanged, bit for bit — no answer moves;
2. the gradient with respect to the threshold is finite and NON-ZERO across the
   physical CAPE range.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.convection._triggers import (
    _TRIGGER_GRADIENT_WIDENING,
    cape_trigger,
    smooth_step,
)

#: CAPE values spanning quiescent to vigorous deep tropics [J/kg].  The
#: measured values in the campaign's columns were 1641-6992.
CAPE_SAMPLES = (0.0, 50.0, 100.0, 500.0, 1641.0, 4679.0, 6992.0, 20_000.0)

#: Shipped-scale trigger sharpness [1/(J/kg)] and threshold [J/kg].
SHARPNESS = 0.01
THRESHOLD = 70.0


@pytest.mark.parametrize("cape", CAPE_SAMPLES)
def test_forward_value_is_bit_identical_to_the_hard_sigmoid(cape):
    """Obligation 1: the physics must not move.  Not 'close' — identical."""
    got = cape_trigger(jnp.asarray(cape), THRESHOLD, SHARPNESS)
    want = smooth_step(jnp.asarray(cape) - THRESHOLD, SHARPNESS)
    assert float(got) == float(want), (
        f"CAPE={cape}: straight-through changed the forward value "
        f"{float(got)!r} vs {float(want)!r}; it must only change the gradient")


@pytest.mark.parametrize("cape", CAPE_SAMPLES)
def test_gradient_wrt_threshold_is_finite_and_nonzero(cape):
    """Obligation 2: a trainer can move the parameter."""
    g = jax.grad(
        lambda thr: cape_trigger(jnp.asarray(cape), thr, SHARPNESS).sum()
    )(THRESHOLD)
    g = float(g)
    assert jnp.isfinite(g), f"CAPE={cape}: gradient is not finite ({g})"
    assert g != 0.0, (
        f"CAPE={cape}: d(trigger)/d(threshold) is exactly 0 — the parameter is "
        "invisible to a gradient trainer, which is the defect this guards")


def test_the_unfixed_trigger_really_was_dead():
    """The control that makes the test above non-vacuous.

    Without the straight-through path the gradient is EXACTLY zero at deep
    tropical CAPE.  If this ever stops being true the widening is unnecessary
    and can be revisited — but only on evidence, not by assumption.
    """
    g = jax.grad(
        lambda thr: smooth_step(jnp.asarray(4679.0) - thr, SHARPNESS).sum()
    )(THRESHOLD)
    assert float(g) == 0.0, (
        "the plain sigmoid is no longer saturated at CAPE=4679 J/kg; the "
        f"straight-through widening may be unnecessary (got {float(g)!r})")


def test_gradient_has_the_physically_correct_sign():
    """A surrogate gradient is only useful if it points the right way.

    RAISING the threshold suppresses convection, so the trigger must DECREASE:
    d(trigger)/d(threshold) < 0.
    """
    for cape in (500.0, 4679.0):
        g = float(jax.grad(
            lambda thr: cape_trigger(jnp.asarray(cape), thr, SHARPNESS).sum()
        )(THRESHOLD))
        assert g < 0.0, (
            f"CAPE={cape}: raising the CAPE threshold must suppress the "
            f"trigger, so the gradient must be negative; got {g!r}")


def test_widening_covers_the_physical_cape_range():
    """The widening factor is a declared number; check it actually spans the
    range it claims rather than merely being large."""
    reach = _TRIGGER_GRADIENT_WIDENING / SHARPNESS * 36.7
    assert reach > max(CAPE_SAMPLES), (
        f"the widened sigmoid saturates at a CAPE difference of {reach:.3g} "
        f"J/kg, inside the physical range (max sample {max(CAPE_SAMPLES)}); "
        "_TRIGGER_GRADIENT_WIDENING is too small to make the trigger trainable")


def test_gradient_is_nonzero_through_a_real_scheme():
    """End to end: not the helper in isolation but a scheme's own config field,
    differentiated the way a trainer would."""
    from legoesm.atmosphere.physics.convection.integration import (
        _get_convection_fn,
    )
    from scripts.run import run_scm_rce_campaign as camp

    import numpy as np

    n = 30
    p_half = np.linspace(5_000.0, 100_000.0, n + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    sigma = p_full / p_full[-1]
    T = 200.0 + 100.0 * sigma ** 0.5
    q_v = 0.018 * sigma ** 3

    cfg = camp.make_physics_config(convection="dca")
    _name, conv_fn, scheme_cfg = _get_convection_fn(cfg.convection)

    def loss(threshold):
        cfg_t = scheme_cfg._replace(cape_threshold=threshold)
        out = conv_fn(
            T=jnp.asarray(T)[None, :], q_v=jnp.asarray(q_v)[None, :],
            p_full=jnp.asarray(p_full)[None, :],
            p_half=jnp.asarray(p_half)[None, :], dt=600.0, config=cfg_t)
        # ConvectionOutput is a NamedTuple, i.e. a tuple, so an
        # `isinstance(out, tuple)` unwrap silently takes its FIRST FIELD.
        # Some leaves return (output, carry), others the bare output; test for
        # the field, not the type.
        if not hasattr(out, "dT_dt"):
            out = out[0]
        return jnp.sum(out.dT_dt) + jnp.sum(out.dq_v_dt)

    g = float(jax.grad(loss)(float(scheme_cfg.cape_threshold)))
    assert jnp.isfinite(g), f"non-finite gradient through dca ({g})"
    assert g != 0.0, (
        "dca's cape_threshold still has an exactly-zero gradient through the "
        "scheme; the straight-through trigger is not reaching it")

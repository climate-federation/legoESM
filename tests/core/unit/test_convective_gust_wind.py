"""Cold-pool gustiness from the surface precipitation rate.

The term exists because a cloud-resolving model with NO mean wind still
evaporates ~2.7 mm/day while a single column reaches ~0.7 — its fluxes ride on
convective cold-pool gusts.  These checks pin the properties that make the term
safe to switch on: exact-zero gating, a finite gradient at zero precipitation
(the hazard of any fractional power), the cap that breaks the
gust->flux->convection->gust loop, and the quadrature combination.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import apply_gustiness, convective_gust_wind

_MM_DAY = 1.0 / 86_400.0  # kg/m^2/s per mm/day


def _rates(mm_day):
    return jnp.asarray([m * _MM_DAY for m in mm_day], dtype=jnp.float64)


def test_disabled_by_default_coefficient_is_exactly_zero():
    """coeff=0 must return EXACT zeros, not a small number.

    The quadrature ``sqrt(u^2+v^2+g^2)`` is only byte-identical to the
    pre-existing behaviour if g is exactly 0, so this is what keeps every
    shipped run unchanged when the feature is off.
    """
    p = _rates([0.0, 1.0, 2.7, 50.0])
    g = convective_gust_wind(p, 0.0)
    assert jnp.array_equal(g, jnp.zeros_like(p))


def test_reference_rate_reproduces_the_coefficient():
    """At P = 1 mm/day the gust IS the coefficient — that is what makes the
    config field readable as 'gust wind at 1 mm/day [m/s]'."""
    g = float(convective_gust_wind(_rates([1.0]), 2.0)[0])
    assert g == pytest.approx(2.0, rel=1e-12), (
        "with no offset inside the power this identity is EXACT")


def test_monotone_and_cube_root_scaling():
    p = _rates([1.0, 8.0])
    g = convective_gust_wind(p, 1.0)
    assert float(g[1]) > float(g[0])
    # 8x the rain is 2x the gust for the 1/3 exponent.
    assert float(g[1]) / float(g[0]) == pytest.approx(2.0, rel=1e-3)


def test_gradient_is_finite_at_zero_precipitation():
    """A bare ratio**(1/3) has an INFINITE derivative at 0.

    The double-``where`` makes the gradient finite AND the value exactly zero
    there.  A first version used an additive offset inside the power instead;
    this test's dry-column sibling caught that it left a spurious 0.02 m/s gust
    (coefficient-scaled) in a column with no rain at all.
    """
    p = _rates([0.0, 0.5, 2.7])
    g = jax.grad(lambda x: convective_gust_wind(x, 2.0).sum())(p)
    assert bool(jnp.all(jnp.isfinite(g))), f"non-finite gradient: {g}"
    assert float(g[0]) == 0.0, "dry column must have an exactly zero subgradient"
    assert float(g[2]) > 0.0, "a raining column must still respond to precip"


def test_negative_precipitation_is_clamped_not_propagated():
    """An upstream sign error must not become a NaN gradient here."""
    p = jnp.asarray([-1.0 * _MM_DAY, 0.0], dtype=jnp.float64)
    g = convective_gust_wind(p, 2.0)
    # EXACT zero, not "small": a dry or sign-flipped column must not acquire a
    # coefficient-scaled gust out of the regularisation.
    assert float(g[0]) == 0.0
    assert float(g[1]) == 0.0
    assert bool(jnp.isfinite(jax.grad(
        lambda x: convective_gust_wind(x, 2.0).sum())(p)[0]))


def test_cap_bounds_the_feedback_loop():
    huge = _rates([10_000.0])
    assert float(convective_gust_wind(huge, 2.0, cap=5.0)[0]) == pytest.approx(
        5.0, rel=1e-6)
    # cap=0 means UNCAPPED, not "capped at zero" — the opposite reading would
    # silently disable the term.
    assert float(convective_gust_wind(huge, 2.0, cap=0.0)[0]) > 5.0


def test_jit_parity():
    p = _rates([0.0, 2.7, 20.0])
    eager = convective_gust_wind(p, 2.0, cap=6.0)
    jitted = jax.jit(lambda x: convective_gust_wind(x, 2.0, cap=6.0))(p)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted), rtol=1e-12)


def test_combines_in_quadrature_never_linearly():
    """The gust joins the resolved wind through apply_gustiness, so a 0.5 m/s
    mean wind with a 3 m/s gust gives sqrt(0.25+9), NOT 3.5."""
    gust = float(convective_gust_wind(_rates([2.7]), 2.0)[0])
    eff = float(apply_gustiness(jnp.asarray(0.5), jnp.asarray(0.0), gust))
    assert eff == pytest.approx(float(np.sqrt(0.25 + gust ** 2)), rel=1e-6)
    assert eff < 0.5 + gust

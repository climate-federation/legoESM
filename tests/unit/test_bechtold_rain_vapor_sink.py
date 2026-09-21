"""``distribute_rain_vapor_sink``: where the in-plume convective rain's vapour
is debited from the environment.

Non-vacuous: routing the "formation" scheme through the legacy vapour-mass
branch fails the formation-local, cap and counter-column tests (the sink
lands on every level instead of at the formation levels); reverting the
inactive denominators to a 1e-30 floor fails the float32 gradient test.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.bechtold import (
    _RAIN_SINK_CAPACITY_FRAC,
    distribute_rain_vapor_sink,
)

DT = 300.0
FORM = np.array([1.0e-7, 2.0e-7, 1.0e-7])   # kg/kg/s at levels 4..6
TOL = 1e-12                                  # well below every rate tested


def _column():
    """Two columns, 10 levels top-down, dp uniform 100 hPa, moist below.
    Column 0 rains at levels 4..6; column 1 is the counter-case: no rain."""
    nlev = 10
    dp = jnp.full((2, nlev), 1.0e4)
    q_v = jnp.asarray(np.stack([np.linspace(1e-4, 1.6e-2, nlev),
                                np.linspace(2e-4, 1.2e-2, nlev)]))
    dq_v_dt = jnp.zeros_like(q_v)
    form = np.zeros((2, nlev)); form[0, 4:7] = FORM
    return q_v, dq_v_dt, dp, jnp.asarray(form)


def _col_int(f, dp):
    return np.asarray(jnp.sum(f * dp, axis=-1) / constants.g)


def _check_pairing(sink, scale, form, dp):
    """Rain actually emitted (form * scale) integrates to the sink."""
    np.testing.assert_allclose(_col_int(sink, dp),
                               _col_int(form * np.asarray(scale)[:, None], dp),
                               rtol=1e-10, atol=TOL)


def test_formation_debits_only_where_rain_forms_and_is_column_exact():
    q_v, dq, dp, form = _column()
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    sink, scale = np.asarray(sink), np.asarray(scale)
    assert np.all(sink[0, :4] == 0.0) and np.all(sink[0, 7:] == 0.0)
    np.testing.assert_allclose(sink[0, 4:7], FORM, rtol=1e-12)
    assert np.all(sink[1] == 0.0) and scale[1] == 1.0      # counter-column untouched
    np.testing.assert_allclose(scale, 1.0, rtol=1e-12)
    _check_pairing(sink, scale, np.asarray(form), dp)


def test_legacy_spreads_by_vapour_mass_over_every_level():
    q_v, dq, dp, form = _column()
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "vapour_mass")
    w = np.asarray(q_v * dp); w = w / w.sum(1, keepdims=True)
    expect = _col_int(form, dp)[:, None] * constants.g * w / np.asarray(dp)
    np.testing.assert_allclose(np.asarray(sink), expect, rtol=1e-6, atol=TOL)
    np.testing.assert_allclose(scale, 1.0)
    assert (np.asarray(sink)[0] > 0).all() and np.all(np.asarray(sink)[1] == 0.0)


def test_capacity_cap_redistributes_within_support_then_reduces_rain():
    q_v, dq, dp, form = _column()
    # level 5 nearly dry: it cannot pay its 2e-7 kg/kg/s over one step
    q_v = q_v.at[:, 5].set(1.0e-6)
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    sink, scale = np.asarray(sink), np.asarray(scale)
    cap5 = _RAIN_SINK_CAPACITY_FRAC * 1.0e-6 / DT
    np.testing.assert_allclose(sink[0, 5], cap5, rtol=1e-12)
    # the excess went to levels 4 and 6 (the rest of the formation support)
    assert (sink[0, [4, 6]] > FORM[[0, 2]] + TOL).all()
    assert np.all(sink[0, :4] == 0.0) and np.all(sink[0, 7:] == 0.0)
    assert np.all(sink[1] == 0.0) and scale[1] == 1.0
    _check_pairing(sink, scale, np.asarray(form), dp)
    np.testing.assert_allclose(scale, 1.0, rtol=1e-12)   # enough slack: nothing lost
    assert (np.asarray(q_v) - DT * sink >= 0.0).all()    # no level goes negative
    # dry the whole support: the rain must shrink, never borrow from outside
    q_v2 = q_v.at[:, 4:7].set(1.0e-6)
    sink2, scale2 = distribute_rain_vapor_sink(form, q_v2, dq, dp, DT, "formation")
    sink2, scale2 = np.asarray(sink2), np.asarray(scale2)
    assert np.all(sink2[0, :4] == 0.0) and np.all(sink2[0, 7:] == 0.0)
    assert 0.0 < scale2[0] < 1.0 and scale2[1] == 1.0
    np.testing.assert_allclose(sink2[0, 4:7], cap5, rtol=1e-12)   # every level at its cap
    _check_pairing(sink2, scale2, np.asarray(form), dp)
    assert (np.asarray(q_v2) - DT * sink2 >= 0.0).all()


def test_capacity_sees_the_transport_tendency():
    q_v, dq, dp, form = _column()
    dq = dq.at[:, 5].set(-np.asarray(q_v)[0, 5] / DT)   # transport empties level 5
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    assert np.asarray(sink)[0, 5] == 0.0
    np.testing.assert_allclose(scale, 1.0, rtol=1e-12)


def test_float32_gradients_finite_for_zero_rain_and_exhausted_capacity():
    q_v, dq, dp, form = _column()
    q_v = q_v.at[0, 4:7].set(0.0)                     # column 0: capacity exhausted
    args = [jnp.asarray(a, jnp.float32) for a in (form, q_v, dq, dp)]

    def loss(form, q_v, dq, dp):
        sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
        return jnp.sum(sink * dp) + jnp.sum(scale)

    for fn in (jax.grad(loss, argnums=(0, 1, 2)), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))):
        grads = fn(*args)
        assert all(np.isfinite(np.asarray(g)).all() for g in grads)
    sink, scale = distribute_rain_vapor_sink(*args, DT, "formation")
    assert np.all(np.asarray(sink)[0] == 0.0) and np.asarray(scale)[0] == 0.0
    assert np.asarray(scale)[1] == 1.0


def test_unknown_scheme_and_bad_dt_raise():
    q_v, dq, dp, form = _column()
    with pytest.raises(ValueError, match="unknown rain_vapor_sink"):
        distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "bogus")
    with pytest.raises(ValueError, match="dt > 0"):
        distribute_rain_vapor_sink(form, q_v, dq, dp, 0.0, "formation")

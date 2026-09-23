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
    _RAIN_SINK_ZERO_FLUX,
    distribute_rain_vapor_sink,
)

DT = 300.0
FORM = np.array([1.0e-7, 2.0e-7, 1.0e-7])   # kg/kg/s at levels 4..6
RTOL = 1e-6                                  # holds in float32 and float64


def _column():
    """Two columns, 10 levels top-down, layer thickness growing 50 -> 140 hPa
    (so a missing thickness weight cannot hide), moist below.
    Column 0 rains at levels 4..6; column 1 is the counter-case: no rain."""
    nlev = 10
    dp = jnp.asarray(np.stack([np.linspace(5.0e3, 1.4e4, nlev),
                               np.linspace(1.4e4, 5.0e3, nlev)]))   # column 1 reversed
    q_v = jnp.asarray(np.stack([np.linspace(1e-4, 1.6e-2, nlev),
                                np.linspace(2e-4, 1.2e-2, nlev)]))
    dq_v_dt = jnp.zeros_like(q_v)
    form = np.zeros((2, nlev)); form[0, 4:7] = FORM
    return q_v, dq_v_dt, dp, jnp.asarray(form)


def _col_int(f, dp):
    return np.asarray(jnp.sum(f * dp, axis=-1) / constants.g)


def _expected(form, q_v, dq, dp, dt=DT):
    """Hand-computed take + redistribution, independent of the helper.
    Arithmetic in the inputs' own dtype so float32 rounding (e.g. an emptied
    level's ``q + dt*dq`` landing on exactly 0) matches the kernel's."""
    want = np.maximum(np.asarray(form), 0.0)
    q_v, dq = np.asarray(q_v), np.asarray(dq)
    cap = (_RAIN_SINK_CAPACITY_FRAC * np.maximum(q_v + q_v.dtype.type(dt) * dq, 0.0)
           / q_v.dtype.type(dt))
    take = np.minimum(want, cap)
    slack = np.where(want > 0, np.maximum(cap - take, 0.0), 0.0)
    dpn = np.asarray(dp)
    excess = np.sum((want - take) * dpn, 1, keepdims=True)
    slack_col = np.sum(slack * dpn, 1, keepdims=True)
    g = constants.g
    on = slack_col / g > _RAIN_SINK_ZERO_FLUX
    add = np.where(on, np.minimum(excess, slack_col) * slack
                   / np.where(on, slack_col, 1.0), 0.0)
    sink = take + add
    rain = np.sum(want * dpn, 1) / g; real = np.sum(sink * dpn, 1) / g
    on = rain > _RAIN_SINK_ZERO_FLUX
    # below the floor: no rain is emitted (scale 0) and nothing is debited
    return (np.where(on[:, None], sink, 0.0),
            np.where(on, real / np.where(on, rain, 1.0), 0.0))


def _check_pairing(sink, scale, form, q_v, dq, dp):
    """Sink and scale match the independent hand computation, and the rain
    actually emitted (form * scale) integrates to the sink."""
    e_sink, e_scale = _expected(form, q_v, dq, dp)
    np.testing.assert_allclose(sink, e_sink, rtol=RTOL, atol=0)
    np.testing.assert_allclose(scale, e_scale, rtol=RTOL, atol=0)
    np.testing.assert_allclose(_col_int(sink, dp),
                               _col_int(form * np.asarray(scale)[:, None], dp), rtol=RTOL)


def test_formation_debits_only_where_rain_forms_and_is_column_exact():
    q_v, dq, dp, form = _column()
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    sink, scale = np.asarray(sink), np.asarray(scale)
    assert np.all(sink[0, :4] == 0.0) and np.all(sink[0, 7:] == 0.0)
    np.testing.assert_allclose(sink[0, 4:7], FORM, rtol=RTOL)
    assert np.all(sink[1] == 0.0) and scale[1] == 0.0      # counter-column: no rain, no debit
    np.testing.assert_allclose(scale[0], 1.0, rtol=RTOL)
    _check_pairing(sink, scale, np.asarray(form), q_v, dq, dp)


def test_legacy_spreads_by_vapour_mass_over_every_level():
    q_v, dq, dp, form = _column()
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "vapour_mass")
    w = np.asarray(q_v * dp); w = w / w.sum(1, keepdims=True)
    expect = _col_int(form, dp)[:, None] * constants.g * w / np.asarray(dp)
    np.testing.assert_allclose(np.asarray(sink), expect, rtol=RTOL, atol=0)
    np.testing.assert_allclose(scale, 1.0)      # legacy returns before the has_rain mask: never rescales
    assert (np.asarray(sink)[0] > 0).all() and np.all(np.asarray(sink)[1] == 0.0)


def test_capacity_cap_redistributes_within_support_then_reduces_rain():
    q_v, dq, dp, form = _column()
    # level 5 nearly dry: it cannot pay its 2e-7 kg/kg/s over one step
    q_v = q_v.at[:, 5].set(1.0e-6)
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    sink, scale = np.asarray(sink), np.asarray(scale)
    cap5 = _RAIN_SINK_CAPACITY_FRAC * 1.0e-6 / DT
    np.testing.assert_allclose(sink[0, 5], cap5, rtol=RTOL)
    # the excess went to levels 4 and 6 (the rest of the formation support)
    assert (sink[0, [4, 6]] > FORM[[0, 2]] * (1 + RTOL)).all()
    assert np.all(sink[0, :4] == 0.0) and np.all(sink[0, 7:] == 0.0)
    assert np.all(sink[1] == 0.0) and scale[1] == 0.0
    _check_pairing(sink, scale, np.asarray(form), q_v, dq, dp)
    np.testing.assert_allclose(scale[0], 1.0, rtol=RTOL)   # enough slack: nothing lost
    assert (np.asarray(q_v) - DT * sink >= 0.0).all()    # no level goes negative
    # dry the whole support: the rain must shrink, never borrow from outside
    q_v2 = q_v.at[:, 4:7].set(1.0e-6)
    sink2, scale2 = distribute_rain_vapor_sink(form, q_v2, dq, dp, DT, "formation")
    sink2, scale2 = np.asarray(sink2), np.asarray(scale2)
    assert np.all(sink2[0, :4] == 0.0) and np.all(sink2[0, 7:] == 0.0)
    assert 0.0 < scale2[0] < 1.0 and scale2[1] == 0.0
    np.testing.assert_allclose(sink2[0, 4:7], cap5, rtol=RTOL)   # every level at its cap
    _check_pairing(sink2, scale2, np.asarray(form), q_v2, dq, dp)
    assert (np.asarray(q_v2) - DT * sink2 >= 0.0).all()


def test_capacity_sees_the_transport_tendency():
    q_v, dq, dp, form = _column()
    dq = dq.at[0, 5].set(-np.asarray(q_v)[0, 5] / DT)   # transport empties level 5
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    assert np.asarray(sink)[0, 5] == 0.0
    np.testing.assert_allclose(np.asarray(scale)[0], 1.0, rtol=RTOL)
    _check_pairing(np.asarray(sink), np.asarray(scale), np.asarray(form), q_v, dq, dp)


@pytest.mark.parametrize("trace", [0.0, 1e-30])
def test_float32_gradients_finite_for_zero_rain_and_exhausted_capacity(trace):
    """trace = 1e-30: a soft-gated whisper of formation in the no-rain column
    (a NORMAL float32 number whose 1/x**2 overflows the backward pass)."""
    q_v, dq, dp, form = _column()
    q_v = q_v.at[0, 4:7].set(0.0)                     # column 0: capacity exhausted
    form = form.at[1, 3].set(trace)
    args = [jnp.asarray(a, jnp.float32) for a in (form, q_v, dq, dp)]

    def loss(form, q_v, dq, dp):
        sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
        return jnp.sum(sink * dp) + jnp.sum(scale)

    for fn in (jax.grad(loss, argnums=(0, 1, 2)), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))):
        grads = fn(*args)
        assert all(np.isfinite(np.asarray(g)).all() for g in grads)
    sink, scale = distribute_rain_vapor_sink(*args, DT, "formation")
    if trace > 0.0:      # pin the precondition: a sub-floor trace, not the trivial form == 0
        col_flux = float(np.sum(np.asarray(args[0][1]) * np.asarray(args[3][1]))) / constants.g
        assert 0.0 < col_flux <= _RAIN_SINK_ZERO_FLUX
    assert np.all(np.asarray(sink)[0] == 0.0) and np.asarray(scale)[0] == 0.0
    # the sub-floor trace: scale 0 and sink 0 together, so rain == sink exactly
    assert np.all(np.asarray(sink)[1] == 0.0) and np.asarray(scale)[1] == 0.0


def test_unknown_scheme_and_bad_dt_raise():
    q_v, dq, dp, form = _column()
    with pytest.raises(ValueError, match="unknown rain_vapor_sink"):
        distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "bogus")
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="dt > 0"):
            distribute_rain_vapor_sink(form, q_v, dq, dp, bad, "formation")

"""``distribute_rain_vapor_sink``: where the in-plume convective rain's vapour
is debited from the environment.

Non-vacuous: routing the "formation" scheme through the legacy vapour-mass
branch makes three of these tests fail (the sink lands on every level
instead of at the formation levels); the end-to-end conservation test in
test_bechtold_column_conservation.py passes either way, as it must.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.bechtold import (
    _RAIN_SINK_CAPACITY_FRAC,
    distribute_rain_vapor_sink,
)

DT = 300.0


def _column():
    """Two columns, 10 levels top-down, dp uniform 100 hPa, moist below."""
    nlev = 10
    dp = jnp.full((2, nlev), 1.0e4)
    q_v = jnp.asarray(np.stack([np.linspace(1e-4, 1.6e-2, nlev),
                                np.linspace(1e-4, 1.6e-2, nlev)]))
    dq_v_dt = jnp.zeros_like(q_v)
    # rain forms at levels 4..6 (mid-troposphere) only
    form = np.zeros((2, nlev)); form[:, 4:7] = [1.0e-7, 2.0e-7, 1.0e-7]
    return q_v, dq_v_dt, dp, jnp.asarray(form)


def _col_int(f, dp):
    return np.asarray(jnp.sum(f * dp, axis=-1) / constants.g)


def test_formation_debits_only_where_rain_forms_and_is_column_exact():
    q_v, dq, dp, form = _column()
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    sink = np.asarray(sink)
    assert np.all(sink[:, :4] == 0.0) and np.all(sink[:, 7:] == 0.0)
    assert np.allclose(sink[:, 4:7], np.asarray(form)[:, 4:7])
    assert np.allclose(scale, 1.0)
    assert np.allclose(_col_int(sink, dp), _col_int(form, dp), rtol=1e-12)


def test_legacy_spreads_by_vapour_mass_over_every_level():
    q_v, dq, dp, form = _column()
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "vapour_mass")
    w = np.asarray(q_v * dp); w = w / w.sum(1, keepdims=True)
    expect = _col_int(form, dp)[:, None] * constants.g * w / np.asarray(dp)
    assert np.allclose(np.asarray(sink), expect, rtol=1e-6)
    assert np.allclose(scale, 1.0)
    assert (np.asarray(sink) > 0).all()


def test_capacity_cap_redistributes_within_support_then_reduces_rain():
    q_v, dq, dp, form = _column()
    # level 5 nearly dry: it cannot pay its 2e-7 kg/kg/s over one step
    q_v = q_v.at[:, 5].set(1.0e-6)
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    sink = np.asarray(sink)
    cap5 = _RAIN_SINK_CAPACITY_FRAC * 1.0e-6 / DT
    assert np.allclose(sink[:, 5], cap5)
    # the excess went to levels 4 and 6 (the rest of the formation support)
    assert (sink[:, [4, 6]] > np.asarray(form)[:, [4, 6]]).all()
    assert np.all(sink[:, :4] == 0.0) and np.all(sink[:, 7:] == 0.0)
    assert np.allclose(_col_int(sink, dp), _col_int(form, dp) * np.asarray(scale))
    assert np.allclose(scale, 1.0)   # enough slack: nothing lost
    # no level goes negative after the debit
    assert (np.asarray(q_v) - DT * sink >= 0.0).all()
    # dry the whole support: the rain must shrink, never borrow from outside
    q_v2 = q_v.at[:, 4:7].set(1.0e-6)
    sink2, scale2 = distribute_rain_vapor_sink(form, q_v2, dq, dp, DT, "formation")
    sink2 = np.asarray(sink2)
    assert np.all(sink2[:, :4] == 0.0) and np.all(sink2[:, 7:] == 0.0)
    assert (np.asarray(scale2) < 1.0).all()
    assert np.allclose(_col_int(sink2, dp), _col_int(form, dp) * np.asarray(scale2))
    assert (np.asarray(q_v2) - DT * sink2 >= 0.0).all()


def test_capacity_sees_the_transport_tendency():
    q_v, dq, dp, form = _column()
    dq = dq.at[:, 5].set(-np.asarray(q_v)[0, 5] / DT)   # transport empties level 5
    sink, scale = distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "formation")
    assert np.allclose(np.asarray(sink)[:, 5], 0.0)
    assert np.allclose(scale, 1.0)


def test_unknown_scheme_raises():
    q_v, dq, dp, form = _column()
    with pytest.raises(ValueError, match="unknown rain_vapor_sink"):
        distribute_rain_vapor_sink(form, q_v, dq, dp, DT, "bogus")

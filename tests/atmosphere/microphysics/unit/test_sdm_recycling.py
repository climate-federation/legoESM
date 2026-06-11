"""Unit tests for SDM recycling (ERF SuperDropletPC::Recycle port)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    SDMConfig,
    SuperDropletState,
    column_rainout,
    make_monodisperse,
    recycle_inactive,
)

_RHO_AS = 1770.0


def test_all_active_pass_through_bit_identical():
    st = make_monodisperse(16, 2.0e-5, 1.0e6)
    z = jnp.linspace(10.0, 100.0, 16)
    st2, z2 = recycle_inactive(st, z, random.PRNGKey(0), 5.0e5,
                               5.0e-8, 1.6, _RHO_AS, 0.0, 500.0)
    for a, b in zip(st, st2):
        assert jnp.array_equal(a, b)
    assert jnp.array_equal(z, z2)


def test_inactive_slots_become_fresh_aerosol():
    st = make_monodisperse(8, 2.0e-5, 1.0e6)
    active = jnp.asarray([1, 0, 1, 0, 0, 1, 1, 0], dtype=st.active.dtype)
    st = st._replace(active=active,
                     multiplicity=st.multiplicity * active)  # inactive: xi=0
    z = jnp.full((8,), 50.0)
    xi_r = 2.5e5
    st2, z2 = recycle_inactive(st, z, random.PRNGKey(1), xi_r,
                               5.0e-8, 1.6, _RHO_AS, 100.0, 400.0)
    inact = np.asarray(active) == 0
    # recycled: active, xi_recycle, wet radius == lognormal dry radius,
    # solute mass consistent, height within bounds
    assert np.all(np.asarray(st2.active)[inact] == 1.0)
    assert np.allclose(np.asarray(st2.multiplicity)[inact], xi_r)
    r = np.asarray(st2.radius)[inact]
    m_s = np.asarray(st2.solute_mass)[inact]
    assert np.allclose(m_s, 4.0 / 3.0 * np.pi * _RHO_AS * r**3, rtol=1e-12)
    assert np.all((r > 5.0e-9) & (r < 5.0e-7))     # sane lognormal range
    znew = np.asarray(z2)[inact]
    assert np.all((znew >= 100.0) & (znew <= 400.0))
    # untouched actives keep everything
    act = ~inact
    assert np.allclose(np.asarray(st2.radius)[act], 2.0e-5)
    assert np.allclose(np.asarray(z2)[act], 50.0)


def test_recycle_deterministic():
    st = make_monodisperse(8, 2.0e-5, 1.0e6)._replace(
        active=jnp.zeros((8,)))
    z = jnp.zeros((8,))
    a = recycle_inactive(st, z, random.PRNGKey(2), 1.0, 5e-8, 1.6, _RHO_AS, 0.0, 100.0)
    b = recycle_inactive(st, z, random.PRNGKey(2), 1.0, 5e-8, 1.6, _RHO_AS, 0.0, 100.0)
    assert jnp.array_equal(a[0].radius, b[0].radius)
    assert jnp.array_equal(a[1], b[1])


def test_rainout_then_recycle_repopulates_column():
    """Compose: rain the column out (all droplets deactivate at the surface),
    then recycle — the column repopulates with fresh aerosol aloft and the
    precipitation bucket keeps the rained-out water (a source/sink pair)."""
    cfg = SDMConfig(terminal_velocity="rogers_yau")
    st = make_monodisperse(32, 1.0e-4, 1.0e5)     # 100 um drops, fast fall
    z = jnp.linspace(1.0, 30.0, 32)
    final, z_f, precip, hist = column_rainout(
        st, z, 1.0, 9.0e4, 283.0, 1.0, 60, 1.0, cfg)
    assert float(jnp.sum(final.active)) == 0.0    # everything rained out
    assert float(precip) > 0.0

    st2, z2 = recycle_inactive(final, z_f, random.PRNGKey(3), 1.0e5,
                               5.0e-8, 1.6, _RHO_AS, 200.0, 1000.0)
    assert float(jnp.sum(st2.active)) == 32.0     # fully repopulated
    assert float(jnp.min(z2)) >= 200.0            # re-injected aloft
    # fresh aerosol is essentially dry: represented water tiny vs the rain
    from legoesm.atmosphere.physics.microphysics.sdm import represented_water_mass
    assert float(jnp.sum(represented_water_mass(st2))) < 1e-6 * float(precip)

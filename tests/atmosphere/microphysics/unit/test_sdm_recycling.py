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


def test_all_active_finite_pass_through_unchanged():
    """A fully active (finite-valued) ensemble passes through unchanged."""
    st = make_monodisperse(16, 2.0e-5, 1.0e6)
    z = jnp.linspace(10.0, 100.0, 16)
    st2, z2 = recycle_inactive(st, z, random.PRNGKey(0), 5.0e5,
                               5.0e-8, 1.6, _RHO_AS, 0.0, 500.0)
    for a, b in zip(st, st2):
        assert jnp.array_equal(a, b)
    assert jnp.array_equal(z, z2)


def test_inactive_slots_become_fresh_aerosol_with_exact_seed_water():
    st = make_monodisperse(8, 2.0e-5, 1.0e6)
    active = jnp.asarray([1, 0, 1, 0, 0, 1, 1, 0], dtype=st.active.dtype)
    # rained-out case: inactive slots keep a STALE xi > 0 (sedimentation does
    # not zero multiplicity) — recycle must overwrite it.
    st = st._replace(active=active)
    z = jnp.full((8,), 50.0)
    xi_r = 2.5e5
    st2, z2 = recycle_inactive(st, z, random.PRNGKey(1), xi_r,
                               5.0e-8, 1.6, _RHO_AS, 100.0, 400.0)
    inact = np.asarray(active) == 0
    assert np.all(np.asarray(st2.active)[inact] == 1.0)
    assert np.allclose(np.asarray(st2.multiplicity)[inact], xi_r)
    # ERF seed semantics: wet (water-equivalent) radius is EXACTLY 1e-15 m —
    # the represented water of a recycled droplet is xi*(4/3)pi*rho_w*1e-45.
    r = np.asarray(st2.radius)[inact]
    assert np.all(r == 1.0e-15)
    seed_water = xi_r * 4.0 / 3.0 * np.pi * constants.rho_water * (1.0e-15) ** 3
    from legoesm.atmosphere.physics.microphysics.sdm import represented_water_mass
    assert np.allclose(np.asarray(represented_water_mass(st2))[inact],
                       seed_water, rtol=1e-12)
    # solute mass is the lognormal dry-aerosol mass (NOT tied to the wet radius)
    m_s = np.asarray(st2.solute_mass)[inact]
    r_dry = (m_s / (4.0 / 3.0 * np.pi * _RHO_AS)) ** (1.0 / 3.0)
    assert np.all((r_dry > 1.0e-9) & (r_dry < 5.0e-6))
    znew = np.asarray(z2)[inact]
    assert np.all((znew >= 100.0) & (znew <= 400.0))
    # untouched actives keep everything
    act = ~inact
    assert np.allclose(np.asarray(st2.radius)[act], 2.0e-5)
    assert np.allclose(np.asarray(z2)[act], 50.0)


def test_recycled_dry_radii_follow_the_lognormal_mode():
    """Statistical check over many recycled slots: the implied dry radii have
    the requested median and log-std (a wrong sampler or a wet/dry mixup in
    solute mass would fail)."""
    n = 50_000
    st = make_monodisperse(n, 2.0e-5, 1.0e6)._replace(active=jnp.zeros((n,)))
    z = jnp.zeros((n,))
    st2, _ = recycle_inactive(st, z, random.PRNGKey(9), 1.0,
                              5.0e-8, 1.6, _RHO_AS, 0.0, 100.0)
    r_dry = (np.asarray(st2.solute_mass)
             / (4.0 / 3.0 * np.pi * _RHO_AS)) ** (1.0 / 3.0)
    assert np.exp(np.median(np.log(r_dry))) == pytest.approx(5.0e-8, rel=0.02)
    assert np.log(r_dry).std() == pytest.approx(np.log(1.6), rel=0.02)


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
    # recycled water is EXACTLY the ERF seed total (32 droplets, xi=1e5 each)
    from legoesm.atmosphere.physics.microphysics.sdm import represented_water_mass
    seed_total = 32 * 1.0e5 * 4.0 / 3.0 * np.pi * constants.rho_water * (1e-15) ** 3
    assert float(jnp.sum(represented_water_mass(st2))) == pytest.approx(
        seed_total, rel=1e-9)

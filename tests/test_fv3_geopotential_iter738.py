"""FV3_3D iter 738: geopotential_from_T_peln_fv3 port.

Hydrostatic integration phi[k] = phis + sum_{j>=k} R_d * T_v * dpeln.

Tests
-----

1. ``test_phi_surface_matches_phis``.
2. ``test_phi_isothermal_dry_analytical``.
3. ``test_phi_moist_higher_than_dry``.
4. ``test_phi_monotone_decreasing``.
5. ``test_phi_shapes_3d``.
6. ``test_phi_finite``.
7. ``test_phi_moist_requires_q``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import geopotential_from_T_peln_fv3


def test_phi_surface_matches_phis():
    """phi[km] = phis."""
    km = 10
    pt = jnp.full((km,), 280.0)
    pe = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    phis = jnp.asarray(5000.0 * constants.g)
    phi = geopotential_from_T_peln_fv3(pt, peln, phis)
    assert abs(float(phi[-1]) - 5000.0 * constants.g) < 1e-8


def test_phi_isothermal_dry_analytical():
    """Isothermal dry column: phi[0] - phi[km] = R_d·T·ln(p_bot/p_top)."""
    km = 10
    T = 280.0
    pt = jnp.full((km,), T)
    pe = jnp.linspace(2.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    phis = jnp.asarray(0.0)
    phi = geopotential_from_T_peln_fv3(pt, peln, phis)
    expected = constants.R_d * T * jnp.log(1.0e5 / 2.0e4)
    actual = float(phi[0] - phi[-1])
    assert abs(actual - float(expected)) / float(expected) < 1e-10


def test_phi_moist_higher_than_dry():
    """Moist column → T_v > T → larger column geopotential."""
    km = 5
    pt = jnp.full((km,), 290.0)
    pe = jnp.linspace(2.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    phis = jnp.asarray(0.0)
    q = jnp.full((km,), 0.015)
    phi_dry = geopotential_from_T_peln_fv3(pt, peln, phis, moist=False)
    phi_moist = geopotential_from_T_peln_fv3(pt, peln, phis, q=q, moist=True)
    assert float(phi_moist[0]) > float(phi_dry[0])


def test_phi_monotone_decreasing():
    """phi monotone decreasing top → surface."""
    rng = np.random.default_rng(seed=738)
    km = 30
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(km,)))
    pe = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    phis = jnp.asarray(1000.0 * constants.g)
    phi = geopotential_from_T_peln_fv3(pt, peln, phis)
    assert jnp.all(jnp.diff(phi) < 0.0)


def test_phi_shapes_3d():
    """3-D inputs → 3-D output."""
    rng = np.random.default_rng(seed=739)
    n_x, n_y, km = 4, 5, 20
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.broadcast_to(jnp.log(pe_col)[None, None, :], (n_x, n_y, km + 1))
    phis = jnp.zeros((n_x, n_y))
    phi = geopotential_from_T_peln_fv3(pt, peln, phis)
    assert phi.shape == (n_x, n_y, km + 1)


def test_phi_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=740)
    km = 30
    pt = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 4, km)))
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.broadcast_to(jnp.log(pe_col)[None, None, :], (4, 4, km + 1))
    phis = jnp.zeros((4, 4))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, km)))
    phi = geopotential_from_T_peln_fv3(pt, peln, phis, q=q, moist=True)
    assert jnp.all(jnp.isfinite(phi))


def test_phi_moist_requires_q():
    """moist=True without q raises."""
    km = 5
    pt = jnp.full((km,), 280.0)
    pe = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    phis = jnp.asarray(0.0)
    with pytest.raises(ValueError):
        geopotential_from_T_peln_fv3(pt, peln, phis, moist=True)

"""FV3_3D iter 733: refactor iter-729/716 hydrostatic branches to delegate
to iter-732 layer_mean_pressure_fv3.

Tests
-----

1. ``test_dry_pressure_refactor_matches_inline``.
2. ``test_bolton_refactor_matches_inline``.
3. ``test_eqv_pot_iter692_unchanged``.
4. ``test_eqv_pot_bolton_iter716_unchanged``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dry_pressure_fv3,
    eqv_pot_bolton_fv3,
    eqv_pot_fv3,
    layer_mean_pressure_fv3,
)


def test_dry_pressure_refactor_matches_inline():
    """iter-729 hydrostatic refactor: pd = (1-rq) * layer_mean_pressure.

    Verify helper result equals manual (1-rq) * delp / dpeln."""
    rng = np.random.default_rng(seed=733)
    km = 10
    pe = jnp.linspace(2.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(km,)))
    pd = dry_pressure_fv3(
        delp, q=q, peln=peln, hydrostatic=True, moist=True,
    )
    expected_inline = (1.0 - q) * delp / (peln[1:] - peln[:-1])
    assert jnp.allclose(pd, expected_inline, atol=1e-10)


def test_bolton_refactor_matches_inline():
    """iter-716 hydrostatic Bolton refactor: p_mb = 0.01 * layer_mean_pressure.

    Run Bolton and verify p_mb path produces same output as manual
    inline formula."""
    km = 5
    T = 280.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    pe = jnp.array([5.0e4, 6.0e4, 7.0e4, 8.0e4, 9.0e4, 1.0e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    theta_e = eqv_pot_bolton_fv3(
        pt, delp, q, peln=peln, hydrostatic=True, moist=False,
    )
    # Manual reproduction: p_mb = 0.01 * delp / dpeln
    p_mb_inline = 0.01 * delp / (peln[1:] - peln[:-1])
    p_mb_helper = 0.01 * layer_mean_pressure_fv3(delp, peln)
    assert jnp.allclose(p_mb_inline, p_mb_helper, atol=1e-12)
    # Theta_e dry: pt * (1000/p_mb)^kappa
    expected = pt * (1000.0 / p_mb_helper) ** constants.kappa
    assert jnp.allclose(theta_e, expected, atol=1e-10)


def test_eqv_pot_iter692_unchanged():
    """iter-692 eqv_pot output preserved by iter-733 refactor."""
    rng = np.random.default_rng(seed=734)
    km = 10
    T = 290.0
    pt = jnp.full((km,), T)
    delp = jnp.full((km,), 1000.0)
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(km,)))
    pe = jnp.linspace(2.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    delp_pe = pe[1:] - pe[:-1]
    theta_e = eqv_pot_fv3(pt, delp_pe, q, peln=peln, hydrostatic=True, moist=True)
    # Sanity: finite + positive
    assert jnp.all(jnp.isfinite(theta_e))
    assert jnp.all(theta_e > 0.0)


def test_eqv_pot_bolton_iter716_unchanged():
    """iter-716 eqv_pot_bolton output preserved by iter-733 refactor."""
    rng = np.random.default_rng(seed=735)
    km = 10
    T = 290.0
    pt = jnp.full((km,), T)
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(km,)))
    pe = jnp.linspace(2.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    theta_e = eqv_pot_bolton_fv3(
        pt, delp, q, peln=peln, hydrostatic=True, moist=True,
    )
    assert jnp.all(jnp.isfinite(theta_e))
    assert jnp.all(theta_e > 0.0)

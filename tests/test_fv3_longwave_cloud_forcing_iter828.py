"""FV3_3D iter 828: longwave_cloud_forcing_fv3 (TOA CRE_LW).

CRE_LW = ε·σ·(T_sfc^4 − T_cloud_top^4).

Tests
-----

1. ``test_lw_tropical_anvil``: T_cloud=200, T_sfc=300 → CRE ≈ 367 W/m².
2. ``test_lw_no_temperature_jump``: T_cloud=T_sfc → CRE=0.
3. ``test_lw_thin_cirrus_emissivity``: ε=0.5 halves CRE.
4. ``test_lw_transparent_zero``: ε=0 → CRE=0.
5. ``test_lw_monotone_dT``: ↑ΔT → ↑CRE.
6. ``test_lw_pairs_with_iter827``: net CRE = LW + SW realistic.
7. ``test_lw_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    longwave_cloud_forcing_fv3,
    shortwave_cloud_forcing_fv3,
)


def test_lw_tropical_anvil():
    """Tropical deep convection: T_cloud=200, T_sfc=300, ε=1.
    CRE_LW = σ·(300^4 - 200^4) = 5.67e-8·(8.1e9 - 1.6e9) ≈ 367 W/m²."""
    t_cloud = jnp.array([200.0])
    t_sfc = jnp.array([300.0])
    cre = longwave_cloud_forcing_fv3(t_cloud, t_sfc)
    expected = constants.sigma_sb * (300.0 ** 4 - 200.0 ** 4)
    np.testing.assert_allclose(np.asarray(cre), [expected], rtol=1e-12)
    assert 300.0 < float(cre[0]) < 400.0


def test_lw_no_temperature_jump():
    """T_cloud = T_sfc → CRE = 0."""
    t = jnp.array([290.0, 280.0, 270.0])
    cre = longwave_cloud_forcing_fv3(t, t)
    np.testing.assert_allclose(np.asarray(cre), jnp.zeros((3,)), atol=1e-12)


def test_lw_thin_cirrus_emissivity():
    """ε=0.5 halves CRE relative to ε=1.0."""
    t_cloud = jnp.array([220.0])
    t_sfc = jnp.array([300.0])
    cre_thick = longwave_cloud_forcing_fv3(t_cloud, t_sfc, emissivity=1.0)
    cre_thin = longwave_cloud_forcing_fv3(t_cloud, t_sfc, emissivity=0.5)
    np.testing.assert_allclose(
        np.asarray(cre_thin), 0.5 * np.asarray(cre_thick), rtol=1e-12
    )


def test_lw_transparent_zero():
    """ε=0 → CRE = 0 (cloud transparent to IR)."""
    t_cloud = jnp.array([200.0])
    t_sfc = jnp.array([300.0])
    cre = longwave_cloud_forcing_fv3(t_cloud, t_sfc, emissivity=0.0)
    np.testing.assert_allclose(np.asarray(cre), [0.0], atol=1e-12)


def test_lw_monotone_dT():
    """↑ΔT = T_sfc − T_cloud → ↑CRE."""
    t_sfc = jnp.full((4,), 300.0)
    t_cloud = jnp.array([290.0, 270.0, 240.0, 200.0])
    cre = longwave_cloud_forcing_fv3(t_cloud, t_sfc)
    assert jnp.all(jnp.diff(cre) > 0.0)


def test_lw_pairs_with_iter827():
    """Net cloud forcing CRE_net = CRE_SW + CRE_LW.
    Tropical anvil: large LW warming dominates over SW cooling."""
    t_cloud = jnp.array([200.0])
    t_sfc = jnp.array([300.0])
    cre_lw = longwave_cloud_forcing_fv3(t_cloud, t_sfc)
    alpha_c = jnp.array([0.6])
    alpha_clr = jnp.array([0.1])
    s_in = jnp.array([400.0])
    cre_sw = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    cre_net = cre_lw + cre_sw
    # Anvil: LW≈367, SW≈-200 → net≈+167 W/m² (net warming)
    assert float(cre_net[0]) > 0.0


def test_lw_shapes_finite():
    """3-D shapes preserved, finite, non-negative for T_cloud ≤ T_sfc."""
    rng = np.random.default_rng(seed=828)
    n_x, n_y = 6, 8
    t_sfc = jnp.asarray(rng.uniform(280.0, 305.0, size=(n_x, n_y)))
    t_cloud = jnp.asarray(rng.uniform(200.0, 280.0, size=(n_x, n_y)))
    eps = jnp.asarray(rng.uniform(0.0, 1.0, size=(n_x, n_y)))
    cre = longwave_cloud_forcing_fv3(t_cloud, t_sfc, eps)
    assert cre.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(cre))
    assert jnp.all(cre >= 0.0)  # T_cloud<T_sfc → positive

"""FV3_3D iter 827: shortwave_cloud_forcing_fv3.

CRE_SW = -S_in · (α_cloudy − α_clear).

Tests
-----

1. ``test_cre_sc_deck``: S=200, α_cloudy=0.5, α_clear=0.1 → -80 W/m².
2. ``test_cre_cloudy_equals_clear``: α_cloudy=α_clear → CRE=0.
3. ``test_cre_night``: S=0 → CRE=0.
4. ``test_cre_warming_dark_cloud``: α_cloudy<α_clear → CRE>0.
5. ``test_cre_monotone_in_cloudy``: ↑α_cloudy → ↓CRE.
6. ``test_cre_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import shortwave_cloud_forcing_fv3


def test_cre_sc_deck():
    """Sc deck: S=200, α_cloudy=0.5, α_clear=0.1 → CRE = -200·0.4 = -80."""
    alpha_c = jnp.array([0.5])
    alpha_clr = jnp.array([0.1])
    s_in = jnp.array([200.0])
    cre = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    np.testing.assert_allclose(np.asarray(cre), [-80.0], rtol=1e-12)


def test_cre_cloudy_equals_clear():
    """α_cloudy = α_clear → CRE = 0."""
    alpha_c = jnp.array([0.3])
    alpha_clr = jnp.array([0.3])
    s_in = jnp.array([500.0])
    cre = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    np.testing.assert_allclose(np.asarray(cre), [0.0], atol=1e-12)


def test_cre_night():
    """S_in=0 (night) → CRE=0."""
    alpha_c = jnp.array([0.5, 0.7, 0.3])
    alpha_clr = jnp.array([0.1, 0.2, 0.3])
    s_in = jnp.zeros((3,))
    cre = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    np.testing.assert_allclose(np.asarray(cre), jnp.zeros((3,)), atol=1e-15)


def test_cre_warming_dark_cloud():
    """α_cloudy < α_clear (dark cloud over bright surface) → CRE > 0."""
    alpha_c = jnp.array([0.2])
    alpha_clr = jnp.array([0.7])  # bright surface (snow / desert)
    s_in = jnp.array([500.0])
    cre = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    assert float(cre[0]) > 0.0
    np.testing.assert_allclose(np.asarray(cre), [250.0], rtol=1e-12)


def test_cre_monotone_in_cloudy():
    """↑α_cloudy → ↓CRE (more reflective cloud, larger cooling)."""
    alpha_c = jnp.array([0.2, 0.4, 0.6, 0.8])
    alpha_clr = jnp.full((4,), 0.1)
    s_in = jnp.full((4,), 200.0)
    cre = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    assert jnp.all(jnp.diff(cre) < 0.0)


def test_cre_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=827)
    n_x, n_y = 6, 8
    alpha_c = jnp.asarray(rng.uniform(0.1, 0.9, size=(n_x, n_y)))
    alpha_clr = jnp.asarray(rng.uniform(0.05, 0.5, size=(n_x, n_y)))
    s_in = jnp.asarray(rng.uniform(0.0, 1400.0, size=(n_x, n_y)))
    cre = shortwave_cloud_forcing_fv3(alpha_c, alpha_clr, s_in)
    assert cre.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(cre))

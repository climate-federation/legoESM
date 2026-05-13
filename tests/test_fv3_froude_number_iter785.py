"""FV3_3D iter 785: froude_number_fv3 (Fr = U/c).

Composes iter-784 ``gravity_wave_speed_fv3``.

Tests
-----

1. ``test_fr_zero_u``: U=0 → Fr=0.
2. ``test_fr_subcritical``: U<c → Fr<1.
3. ``test_fr_critical``: U=c → Fr=1.
4. ``test_fr_supercritical``: U>c → Fr>1.
5. ``test_fr_unstrat_floor``: c=0 → Fr huge but finite.
6. ``test_fr_composes_iter784``: (U, N, H) → c → Fr.
7. ``test_fr_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    froude_number_fv3,
    gravity_wave_speed_fv3,
)


def test_fr_zero_u():
    """U=0 → Fr=0."""
    U = jnp.array([0.0, 0.0, 0.0])
    c = jnp.array([10.0, 30.0, 100.0])
    Fr = froude_number_fv3(U, c)
    np.testing.assert_allclose(np.asarray(Fr), [0.0, 0.0, 0.0], atol=1e-15)


def test_fr_subcritical():
    """U=10, c=30 → Fr=1/3 < 1 (subcritical)."""
    U = jnp.array([10.0])
    c = jnp.array([30.0])
    Fr = froude_number_fv3(U, c)
    np.testing.assert_allclose(np.asarray(Fr), [1.0 / 3.0], rtol=1e-12)
    assert float(Fr[0]) < 1.0


def test_fr_critical():
    """U=c → Fr=1.0 (hydraulic critical / standing wave)."""
    U = jnp.array([30.0])
    c = jnp.array([30.0])
    Fr = froude_number_fv3(U, c)
    np.testing.assert_allclose(np.asarray(Fr), [1.0], rtol=1e-12)


def test_fr_supercritical():
    """U=50, c=30 → Fr=5/3 > 1 (wave-breaking regime)."""
    U = jnp.array([50.0])
    c = jnp.array([30.0])
    Fr = froude_number_fv3(U, c)
    np.testing.assert_allclose(np.asarray(Fr), [5.0 / 3.0], rtol=1e-12)
    assert float(Fr[0]) > 1.0


def test_fr_unstrat_floor():
    """c=0 (unstratified) → Fr huge but finite via c_floor."""
    U = jnp.array([10.0])
    c = jnp.array([0.0])
    Fr = froude_number_fv3(U, c, c_floor=1e-12)
    assert jnp.all(jnp.isfinite(Fr))
    # 10/1e-12 = 1e13
    assert float(Fr[0]) > 1e12


def test_fr_composes_iter784():
    """Pipeline (U, N, H) → c → Fr."""
    U = jnp.array([20.0])
    N = jnp.array([0.01])
    H = jnp.array([3000.0])
    c = gravity_wave_speed_fv3(N, H)  # = 30 m/s (atmospheric Kelvin)
    Fr = froude_number_fv3(U, c)
    # 20/30 ≈ 0.667 (subcritical)
    np.testing.assert_allclose(np.asarray(Fr), [20.0 / 30.0], rtol=1e-12)
    assert float(Fr[0]) < 1.0


def test_fr_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=785)
    n_x, n_y, km = 4, 5, 20
    U = jnp.asarray(rng.uniform(0.0, 50.0, size=(n_x, n_y, km)))
    c = jnp.asarray(rng.uniform(5.0, 100.0, size=(n_x, n_y, km)))
    Fr = froude_number_fv3(U, c)
    assert Fr.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(Fr))
    assert jnp.all(Fr >= 0.0)

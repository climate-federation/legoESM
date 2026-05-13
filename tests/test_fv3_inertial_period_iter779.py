"""FV3_3D iter 779: inertial_period_fv3 (2π/|f|).

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_inertial_pole``: lat=π/2 → T ≈ 11.97 h.
2. ``test_inertial_midlat_30``: lat=30° → T ≈ 23.93 h (sidereal day).
3. ``test_inertial_equator_floored``: lat=0 → huge but finite (floor).
4. ``test_inertial_hemisphere_symmetry``: T(+lat) = T(-lat).
5. ``test_inertial_composes_iter778``: matches 2π/|f| from iter-778.
6. ``test_inertial_shapes_finite``: 3-D shapes + finite.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    inertial_period_fv3,
)


def test_inertial_pole():
    """Pole: T = 2π/(2·Ω) ≈ 11.97 h ≈ 43083 s."""
    lat = jnp.array([jnp.pi / 2.0])
    T = inertial_period_fv3(lat)
    expected = 2.0 * np.pi / (2.0 * constants.Omega)
    np.testing.assert_allclose(np.asarray(T), [expected], rtol=1e-12)
    # Sanity: ~11.97 hours
    assert 43000.0 < float(T[0]) < 43200.0


def test_inertial_midlat_30():
    """30°N: |f| = Ω · sin(60°) wait — recompute.

    |f| = 2·Ω·sin(30°) = Ω.  So T = 2π/Ω = one sidereal day ≈ 86164 s.
    """
    lat = jnp.array([30.0])
    T = inertial_period_fv3(lat, units="deg")
    expected = 2.0 * np.pi / constants.Omega
    np.testing.assert_allclose(np.asarray(T), [expected], rtol=1e-12)
    # Sanity: ~24 hours
    assert 86000.0 < float(T[0]) < 86200.0


def test_inertial_equator_floored():
    """Equator: f=0 → T clamped huge but finite via f_floor."""
    lat = jnp.array([0.0])
    T = inertial_period_fv3(lat, f_floor=1e-12)
    assert jnp.all(jnp.isfinite(T))
    # 2π/1e-12 ≈ 6.28e12 s
    assert float(T[0]) > 1e12


def test_inertial_hemisphere_symmetry():
    """T(+lat) = T(-lat) since |f| absolute value."""
    lat_pos = jnp.array([10.0, 30.0, 50.0, 70.0])
    lat_neg = -lat_pos
    T_pos = inertial_period_fv3(lat_pos, units="deg")
    T_neg = inertial_period_fv3(lat_neg, units="deg")
    np.testing.assert_allclose(np.asarray(T_pos), np.asarray(T_neg), atol=1e-12)


def test_inertial_composes_iter778():
    """T = 2π/|f| matches iter-778 directly."""
    rng = np.random.default_rng(seed=779)
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2 + 0.1, jnp.pi / 2 - 0.1, size=(50,)))
    f = coriolis_parameter_fv3(lat)
    T_ref = 2.0 * jnp.pi / jnp.abs(f)
    T = inertial_period_fv3(lat)
    np.testing.assert_allclose(np.asarray(T), np.asarray(T_ref), rtol=1e-12)


def test_inertial_shapes_finite():
    """3-D shapes preserved, finite for non-pole inputs."""
    rng = np.random.default_rng(seed=780)
    n_x, n_y = 6, 8
    lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=(n_x, n_y)))  # avoid singular
    T = inertial_period_fv3(lat)
    assert T.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(T))
    assert jnp.all(T > 0.0)

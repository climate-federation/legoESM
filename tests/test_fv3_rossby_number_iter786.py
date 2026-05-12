"""FV3_3D iter 786: rossby_number_fv3 (Ro = U/(|f|·L)).

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_ro_synoptic_qg``: U=10, L=1000 km, |f|=1e-4 → Ro=0.1 (QG).
2. ``test_ro_tornado_inertial``: U=70, L=100 m, |f|=1e-4 → Ro=7000.
3. ``test_ro_zero_u``: U=0 → Ro=0.
4. ``test_ro_monotonic_inputs``: ↑U, ↓L, ↓f → ↑Ro.
5. ``test_ro_equator_floored``: f=0 → huge but finite.
6. ``test_ro_composes_iter778``: (U, L, lat) → f → Ro.
7. ``test_ro_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    rossby_number_fv3,
)


def test_ro_synoptic_qg():
    """Synoptic mid-lat: U=10, L=1000 km, |f|=1e-4 → Ro=0.1 (QG regime)."""
    U = jnp.array([10.0])
    L = jnp.array([1.0e6])
    f = jnp.array([1.0e-4])
    Ro = rossby_number_fv3(U, L, f)
    np.testing.assert_allclose(np.asarray(Ro), [0.1], rtol=1e-12)
    assert float(Ro[0]) < 0.5  # QG regime


def test_ro_tornado_inertial():
    """Tornado-scale: U=70, L=100 m, |f|=1e-4 → Ro=7000 (inertial)."""
    U = jnp.array([70.0])
    L = jnp.array([100.0])
    f = jnp.array([1.0e-4])
    Ro = rossby_number_fv3(U, L, f)
    np.testing.assert_allclose(np.asarray(Ro), [7000.0], rtol=1e-12)
    assert float(Ro[0]) > 100.0  # rotation negligible


def test_ro_zero_u():
    """U=0 → Ro=0."""
    U = jnp.array([0.0, 0.0, 0.0])
    L = jnp.array([1.0e5, 1.0e6, 1.0e7])
    f = jnp.array([1.0e-4, 1.0e-4, 1.0e-4])
    Ro = rossby_number_fv3(U, L, f)
    np.testing.assert_allclose(np.asarray(Ro), [0.0, 0.0, 0.0], atol=1e-15)


def test_ro_monotonic_inputs():
    """↑U → ↑Ro; ↓L → ↑Ro; ↓|f| → ↑Ro."""
    base_U = jnp.array([10.0, 10.0, 10.0])
    base_L = jnp.array([1.0e6, 1.0e6, 1.0e6])
    base_f = jnp.array([1.0e-4, 1.0e-4, 1.0e-4])
    Ro_base = rossby_number_fv3(base_U, base_L, base_f)

    # ↑U → ↑Ro
    Ro_U_hi = rossby_number_fv3(base_U + 5.0, base_L, base_f)
    assert jnp.all(Ro_U_hi > Ro_base)

    # ↓L → ↑Ro
    Ro_L_lo = rossby_number_fv3(base_U, base_L / 2.0, base_f)
    assert jnp.all(Ro_L_lo > Ro_base)

    # ↓|f| → ↑Ro
    Ro_f_lo = rossby_number_fv3(base_U, base_L, base_f / 2.0)
    assert jnp.all(Ro_f_lo > Ro_base)


def test_ro_equator_floored():
    """f=0 → Ro huge but finite via fL_floor."""
    U = jnp.array([10.0])
    L = jnp.array([1.0e6])
    f = jnp.array([0.0])
    Ro = rossby_number_fv3(U, L, f, fL_floor=1e-12)
    assert jnp.all(jnp.isfinite(Ro))
    # 10 / 1e-12 = 1e13
    assert float(Ro[0]) > 1e12


def test_ro_composes_iter778():
    """Pipeline (U, L, lat) → f → Ro."""
    U = jnp.array([10.0])
    L = jnp.array([1.0e6])
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    Ro = rossby_number_fv3(U, L, f)
    # |f|=2·Ω·sin(45°) ≈ 1.03e-4 → Ro = 10/(1.03e-4·1e6) ≈ 0.097
    assert 0.05 < float(Ro[0]) < 0.15


def test_ro_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=786)
    n_x, n_y, km = 4, 5, 20
    U = jnp.asarray(rng.uniform(1.0, 50.0, size=(n_x, n_y, km)))
    L = jnp.asarray(rng.uniform(1.0e3, 1.0e7, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(1e-5, 1.5e-4, size=(n_x, n_y, km)))
    Ro = rossby_number_fv3(U, L, f)
    assert Ro.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(Ro))
    assert jnp.all(Ro >= 0.0)

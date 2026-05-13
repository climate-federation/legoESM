"""FV3_3D iter 765: lcl_height_fv3 port.

z_LCL = z + (cp_d / g) · (T − T_LCL)

Completes LCL state triad (T_LCL, p_LCL, z_LCL) with iter-763
``lcl_temperature_fv3`` and iter-764 ``lcl_pressure_fv3``.

Tests
-----

1. ``test_lcl_z_dry_case``: T_LCL=T → z_LCL=z_parcel.
2. ``test_lcl_z_above_parcel``: cooling required → z_LCL>z_parcel.
3. ``test_lcl_z_with_iter763_iter764``: full triad cross-check.
4. ``test_lcl_z_shapes_3d``.
5. ``test_lcl_z_finite``.
6. ``test_lcl_z_overrides``: explicit cp_air + g overrides.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    lcl_height_fv3,
    lcl_pressure_fv3,
    lcl_temperature_fv3,
)


def test_lcl_z_dry_case():
    """T_LCL=T (no cooling) → z_LCL=z_parcel."""
    z = jnp.full((4,), 1000.0)
    T = jnp.full((4,), 290.0)
    t_lcl = T  # no lift
    z_lcl = lcl_height_fv3(z, T, t_lcl)
    np.testing.assert_allclose(np.asarray(z_lcl), np.asarray(z), atol=1e-10)


def test_lcl_z_above_parcel():
    """T_LCL<T → z_LCL>z_parcel (LCL above parcel source)."""
    z = jnp.full((3,), 500.0)
    T = jnp.full((3,), 290.0)
    t_lcl = jnp.full((3,), 280.0)  # cooled 10 K
    z_lcl = lcl_height_fv3(z, T, t_lcl)
    # Δz = cp/g · 10 K ≈ 1024 m
    assert jnp.all(z_lcl > z)
    expected = z + (constants.c_pd / constants.g) * 10.0
    np.testing.assert_allclose(np.asarray(z_lcl), np.asarray(expected), atol=1e-6)


def test_lcl_z_with_iter763_iter764():
    """Compose iter-763 + iter-764 + iter-765 for realistic supercell parcel."""
    z = jnp.array([10.0])
    T = jnp.array([295.0])
    p_pa = jnp.array([100_000.0])
    p_mb = p_pa / 100.0
    q = jnp.array([0.012])
    t_lcl = lcl_temperature_fv3(T, p_mb, q)
    p_lcl = lcl_pressure_fv3(T, p_pa, t_lcl)
    z_lcl = lcl_height_fv3(z, T, t_lcl)
    # Sanity: all three LCL state vars consistent with parcel above
    assert float(t_lcl[0]) < float(T[0])
    assert float(p_lcl[0]) < float(p_pa[0])
    assert float(z_lcl[0]) > float(z[0])
    # LCL height ~0.5-2.5 km above parcel for these conditions
    assert 100.0 < float(z_lcl[0]) - float(z[0]) < 3000.0


def test_lcl_z_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=765)
    n_x, n_y, km = 4, 5, 20
    z = jnp.asarray(rng.uniform(0.0, 5000.0, size=(n_x, n_y, km)))
    T = jnp.asarray(rng.uniform(240.0, 300.0, size=(n_x, n_y, km)))
    t_lcl = T - jnp.asarray(rng.uniform(0.0, 30.0, size=(n_x, n_y, km)))
    z_lcl = lcl_height_fv3(z, T, t_lcl)
    assert z_lcl.shape == (n_x, n_y, km)


def test_lcl_z_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=766)
    z = jnp.asarray(rng.uniform(0.0, 10_000.0, size=(8, 30)))
    T = jnp.asarray(rng.uniform(220.0, 305.0, size=(8, 30)))
    t_lcl = T - jnp.asarray(rng.uniform(0.0, 40.0, size=(8, 30)))
    z_lcl = lcl_height_fv3(z, T, t_lcl)
    assert jnp.all(jnp.isfinite(z_lcl))


def test_lcl_z_overrides():
    """Explicit cp_air + g overrides match defaults."""
    z = jnp.full((2,), 100.0)
    T = jnp.full((2,), 290.0)
    t_lcl = jnp.full((2,), 280.0)
    z_lcl_default = lcl_height_fv3(z, T, t_lcl)
    z_lcl_explicit = lcl_height_fv3(
        z, T, t_lcl, cp_air=constants.c_pd, g=constants.g
    )
    np.testing.assert_allclose(
        np.asarray(z_lcl_default), np.asarray(z_lcl_explicit), atol=1e-12
    )

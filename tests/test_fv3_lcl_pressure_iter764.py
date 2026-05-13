"""FV3_3D iter 764: lcl_pressure_fv3 port.

p_LCL = p · (T_LCL / T)^(1/κ)  (Poisson, θ conserved on dry adiabat).

Companion to iter-763 lcl_temperature_fv3.

Tests
-----

1. ``test_lcl_p_dry_case``: T_LCL=T → p_LCL=p.
2. ``test_lcl_p_below_p``: cooling required → p_LCL<p.
3. ``test_lcl_p_with_iter763``: cross-check with iter-763 T_LCL.
4. ``test_lcl_p_shapes_3d``.
5. ``test_lcl_p_finite``.
6. ``test_lcl_p_cappa_arg``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    lcl_pressure_fv3,
    lcl_temperature_fv3,
)


def test_lcl_p_dry_case():
    """T_LCL=T (no cooling) → p_LCL=p."""
    T = jnp.full((4,), 290.0)
    p = jnp.full((4,), 100_000.0)
    t_lcl = T  # no lift needed
    p_lcl = lcl_pressure_fv3(T, p, t_lcl)
    np.testing.assert_allclose(np.asarray(p_lcl), np.asarray(p), atol=1e-6)


def test_lcl_p_below_p():
    """T_LCL<T → p_LCL<p (LCL above parcel source)."""
    T = jnp.full((3,), 290.0)
    p = jnp.full((3,), 100_000.0)
    t_lcl = jnp.full((3,), 280.0)  # cooled 10 K
    p_lcl = lcl_pressure_fv3(T, p, t_lcl)
    assert jnp.all(p_lcl < p)


def test_lcl_p_with_iter763():
    """Compose with iter-763 T_LCL: realistic supercell parcel."""
    T = jnp.array([295.0])
    p_pa = jnp.array([100_000.0])
    p_mb = p_pa / 100.0
    q = jnp.array([0.012])
    t_lcl = lcl_temperature_fv3(T, p_mb, q)
    p_lcl = lcl_pressure_fv3(T, p_pa, t_lcl)
    # T_LCL well below T → p_LCL well below p
    assert float(t_lcl[0]) < float(T[0])
    assert float(p_lcl[0]) < float(p_pa[0])
    # LCL typically 80-95 kPa for these surface conditions
    assert 70_000.0 < float(p_lcl[0]) < 99_000.0


def test_lcl_p_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=764)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(240.0, 300.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 100_000.0, size=(n_x, n_y, km)))
    t_lcl = T - jnp.asarray(rng.uniform(0.0, 30.0, size=(n_x, n_y, km)))
    p_lcl = lcl_pressure_fv3(T, p, t_lcl)
    assert p_lcl.shape == (n_x, n_y, km)


def test_lcl_p_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=765)
    T = jnp.asarray(rng.uniform(220.0, 305.0, size=(8, 30)))
    p = jnp.asarray(rng.uniform(5_000.0, 105_000.0, size=(8, 30)))
    t_lcl = T - jnp.asarray(rng.uniform(0.0, 40.0, size=(8, 30)))
    p_lcl = lcl_pressure_fv3(T, p, t_lcl)
    assert jnp.all(jnp.isfinite(p_lcl))


def test_lcl_p_cappa_arg():
    """Explicit cappa override."""
    T = jnp.full((2,), 290.0)
    p = jnp.full((2,), 100_000.0)
    t_lcl = jnp.full((2,), 280.0)
    p_lcl_default = lcl_pressure_fv3(T, p, t_lcl)
    p_lcl_explicit = lcl_pressure_fv3(T, p, t_lcl, cappa=constants.kappa)
    np.testing.assert_allclose(np.asarray(p_lcl_default), np.asarray(p_lcl_explicit), atol=1e-10)

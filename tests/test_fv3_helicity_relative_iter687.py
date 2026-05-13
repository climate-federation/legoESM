"""FV3_3D iter 687: helicity_relative_fv3 port.

Faithful JAX port of FV3 ``helicity_relative`` (tools/
fv_diagnostics.F90:4811-4895).

Tests
-----

1. ``test_srh_zero_shear``.
2. ``test_srh_zero_wind``.
3. ``test_srh_outside_window_zero``.
4. ``test_srh_finite``.
5. ``test_srh_hydrostatic_requires_args``.
6. ``test_srh_idealized_linear_shear``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import helicity_relative_fv3


def _column_setup(km=20, dz=-500.0):
    delz = jnp.full((km,), dz)
    return delz


def test_srh_zero_shear():
    """Uniform wind in column → zero shear → SRH = 0."""
    km = 20
    delz = _column_setup(km)
    ua = jnp.full((km,), 10.0)
    va = jnp.full((km,), 5.0)
    srh = helicity_relative_fv3(ua, va, delz=delz, z_bot=0.0, z_top=3000.0)
    assert abs(float(srh)) < 1e-10


def test_srh_zero_wind():
    """All-zero wind → SRH = 0."""
    km = 20
    delz = _column_setup(km)
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    srh = helicity_relative_fv3(ua, va, delz=delz, z_bot=0.0, z_top=3000.0)
    assert abs(float(srh)) < 1e-12


def test_srh_outside_window_zero():
    """z_top ≤ z_bot → SRH = 0."""
    km = 20
    delz = _column_setup(km)
    rng = np.random.default_rng(seed=687)
    ua = jnp.asarray(rng.normal(size=(km,)))
    va = jnp.asarray(rng.normal(size=(km,)))
    srh = helicity_relative_fv3(ua, va, delz=delz, z_bot=3000.0, z_top=3000.0)
    assert abs(float(srh)) < 1e-12


def test_srh_finite():
    """No NaN/Inf on random 3-D field."""
    rng = np.random.default_rng(seed=687)
    km = 30
    ua = jnp.asarray(rng.normal(scale=10.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(4, 4, km)))
    delz = jnp.full((4, 4, km), -300.0)
    srh = helicity_relative_fv3(ua, va, delz=delz)
    assert jnp.all(jnp.isfinite(srh))
    assert srh.shape == (4, 4)


def test_srh_hydrostatic_requires_args():
    """hydrostatic=True without pt/q/peln raises ValueError."""
    km = 10
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    with pytest.raises(ValueError):
        helicity_relative_fv3(ua, va, hydrostatic=True)


def test_srh_idealized_linear_shear():
    """Linear veering wind: u(z) = z·c1, v(z) = 0 in [z_bot, z_top].

    Mean uc = (z_bot+z_top)/2 · c1, vc = 0.
    du_dz centered ≈ c1·dz at interior layers; dv_dz = 0.
    SRH = Σ (ua-uc)·dv_dz - (va-vc)·du_dz
        = - Σ 0·c1·dz = 0  (va≡0)
    So test reduces to: SRH=0 if va≡0 and only u shears.
    """
    km = 30
    delz = jnp.full((km,), -100.0)
    zh = jnp.linspace(km * 100.0 - 50.0, 50.0, km)  # top-down
    ua = 0.005 * zh
    va = jnp.zeros((km,))
    srh = helicity_relative_fv3(ua, va, delz=delz, z_bot=0.0, z_top=3000.0)
    assert abs(float(srh)) < 1e-9

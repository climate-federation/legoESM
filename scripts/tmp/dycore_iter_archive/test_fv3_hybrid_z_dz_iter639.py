"""FV3_3D iter 639: hybrid_z_dz port.

Faithful JAX port of FV3 ``hybrid_z_dz`` (tools/fv_eta.F90:1794-1855).
FV3 stretched vertical-coord with per-layer s_fac table.

Tests
-----

1. ``test_hybrid_z_dz_shape``.
2. ``test_hybrid_z_dz_positive``.
3. ``test_hybrid_z_dz_total_near_ztop``.
4. ``test_hybrid_z_dz_finite``.
5. ``test_hybrid_z_dz_km_too_small_raises``.
6. ``test_hybrid_z_dz_top_dominant``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import hybrid_z_dz


def test_hybrid_z_dz_shape():
    """Output shape (km,)."""
    km = 32
    dz = hybrid_z_dz(km, ztop=50.0e3, s_rate=1.06)
    assert dz.shape == (km,)


def test_hybrid_z_dz_positive():
    """All dz > 0."""
    dz = hybrid_z_dz(km=32, ztop=50.0e3, s_rate=1.06)
    assert jnp.all(dz > 0)


def test_hybrid_z_dz_total_near_ztop():
    """Σ dz ≈ ztop (sm1_edge preserves total thickness exactly)."""
    ztop = 50.0e3
    dz = hybrid_z_dz(km=32, ztop=ztop, s_rate=1.06)
    total = float(jnp.sum(dz))
    # sm1_edge is flux-form: total preserved.  ze[0] was set to ztop
    # explicitly so total = ztop - ze[km] = ztop - 0 = ztop.
    assert abs(total - ztop) / ztop < 1e-9, (
        f"sum(dz) = {total}, ztop = {ztop}"
    )


def test_hybrid_z_dz_finite():
    """No NaN/Inf."""
    dz = hybrid_z_dz(km=24, ztop=40.0e3, s_rate=1.05)
    assert jnp.all(jnp.isfinite(dz))


def test_hybrid_z_dz_km_too_small_raises():
    """km < 18 raises ValueError."""
    with pytest.raises(ValueError):
        hybrid_z_dz(km=17, ztop=30.0e3, s_rate=1.06)


def test_hybrid_z_dz_top_dominant():
    """Top layer (k=0) is the largest (due to top-multiplier stretching)."""
    dz = hybrid_z_dz(km=32, ztop=50.0e3, s_rate=1.06)
    # Top should be (much) larger than bottom
    assert float(dz[0]) > float(dz[-1])

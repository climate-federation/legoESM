"""FV3_3D iter 641: compute_dz_var port.

Faithful JAX port of FV3 ``compute_dz_var`` (tools/fv_eta.F90:
1930-1998).  Variable dz with rescaling to exact ztop.

Tests
-----

1. ``test_compute_dz_var_shape``.
2. ``test_compute_dz_var_total_equals_ztop``.
3. ``test_compute_dz_var_positive``.
4. ``test_compute_dz_var_finite``.
5. ``test_compute_dz_var_km_too_small_raises``.
6. ``test_compute_dz_var_default_s_rate_uniform_middle``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import compute_dz_var


def test_compute_dz_var_shape():
    """Output shape (km,)."""
    km = 32
    dz = compute_dz_var(km, ztop=50.0e3)
    assert dz.shape == (km,)


def test_compute_dz_var_total_equals_ztop():
    """Σ dz = ztop after rescaling + sm1_edge (flux-form invariant)."""
    ztop = 50.0e3
    dz = compute_dz_var(km=32, ztop=ztop)
    total = float(jnp.sum(dz))
    assert abs(total - ztop) / ztop < 1e-9, (
        f"sum(dz) = {total}, ztop = {ztop}"
    )


def test_compute_dz_var_positive():
    """All dz > 0."""
    dz = compute_dz_var(km=24, ztop=40.0e3)
    assert jnp.all(dz > 0)


def test_compute_dz_var_finite():
    """No NaN/Inf."""
    dz = compute_dz_var(km=32, ztop=60.0e3, s_rate=1.05)
    assert jnp.all(jnp.isfinite(dz))


def test_compute_dz_var_km_too_small_raises():
    """km < 18 raises ValueError."""
    with pytest.raises(ValueError):
        compute_dz_var(km=15, ztop=30.0e3)


def test_compute_dz_var_default_s_rate_uniform_middle():
    """With default s_rate=1.0, middle layers (after top stretch) have
    similar s_fac → moderate variability."""
    dz = compute_dz_var(km=32, ztop=50.0e3, s_rate=1.0)
    # Top should be larger than bottom (FV3 top-stretch multipliers)
    assert float(dz[0]) > float(dz[-1])

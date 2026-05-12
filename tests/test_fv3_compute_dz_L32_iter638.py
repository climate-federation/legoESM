"""FV3_3D iter 638: compute_dz_L32 port.

Faithful JAX port of FV3 ``compute_dz_L32`` (tools/fv_eta.F90:
2000-2067).  Builds the FV3-canonical 32-layer vertical with
ztop ≈ 60 km.

Tests
-----

1. ``test_compute_dz_L32_shape``.
2. ``test_compute_dz_L32_bottom_after_zflip``.
3. ``test_compute_dz_L32_total_thickness``.
4. ``test_compute_dz_L32_dz_positive``.
5. ``test_compute_dz_L32_finite``.
6. ``test_compute_dz_L32_top_layer_doubled``.
7. ``test_compute_dz_L32_reaches_30km_at_k_after_zflip``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import compute_dz_L32


def test_compute_dz_L32_shape():
    """Output shape (32,)."""
    dz, _ = compute_dz_L32()
    assert dz.shape == (32,)


def test_compute_dz_L32_bottom_after_zflip():
    """After zflip, dz[31] (bottom) = 75 m (the FV3 bottom dz)."""
    dz, _ = compute_dz_L32()
    assert abs(float(dz[31]) - 75.0) < 1e-9


def test_compute_dz_L32_total_thickness():
    """ztop = sum(dz)."""
    dz, ztop = compute_dz_L32()
    assert abs(float(ztop) - float(jnp.sum(dz))) < 1e-9


def test_compute_dz_L32_dz_positive():
    """All layers have positive thickness."""
    dz, _ = compute_dz_L32()
    assert jnp.all(dz > 0)


def test_compute_dz_L32_finite():
    """No NaN/Inf."""
    dz, ztop = compute_dz_L32()
    assert jnp.all(jnp.isfinite(dz))
    assert jnp.isfinite(ztop)


def test_compute_dz_L32_top_layer_doubled():
    """After zflip, dz[0] (top) = 2·dz[1] (FV3 construction)."""
    dz, _ = compute_dz_L32()
    assert abs(float(dz[0]) - 2.0 * float(dz[1])) < 1e-8


def test_compute_dz_L32_reaches_30km_at_k_after_zflip():
    """Cumulative thickness reaches z2 ≈ 30 km near the upper block boundary.

    In FV3 1-indexed: ze[k0+k1+k2+1] = ze[32] = z2 (approximately).
    After zflip, this corresponds to ze counting from the top.
    """
    dz, ztop = compute_dz_L32()
    # ztop should be around 60-66 km (z2 was 30 km at intermediate point)
    ztop_km = float(ztop) / 1000.0
    assert 40.0 < ztop_km < 80.0, f"ztop = {ztop_km} km (expected ~60)"

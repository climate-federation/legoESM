"""FV3_3D iter 637: compute_dz_L101 + set_external_eta ports.

Faithful JAX ports of FV3 helpers (tools/fv_eta.F90):
- ``set_external_eta`` (F90:788) — derive (ptop, ks) from ak/bk
- ``compute_dz_L101``  (F90:2069) — FV3 L101 vertical layer thicknesses

Tests
-----

1. ``test_set_external_eta_ptop``.
2. ``test_set_external_eta_ks_count``.
3. ``test_set_external_eta_all_hybrid``.
4. ``test_compute_dz_L101_shape``.
5. ``test_compute_dz_L101_uniform_bottom``.
6. ``test_compute_dz_L101_geometric_middle``.
7. ``test_compute_dz_L101_top_quadrupled``.
8. ``test_compute_dz_L101_ztop_approx_20km``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import compute_dz_L101, set_external_eta


def test_set_external_eta_ptop():
    """ptop = ak[0]."""
    ak = jnp.asarray([100.0, 200.0, 500.0, 1000.0, 5000.0])
    bk = jnp.asarray([0.0, 0.0, 0.1, 0.3, 1.0])
    ptop, _ = set_external_eta(ak, bk)
    assert abs(float(ptop) - 100.0) < 1e-10


def test_set_external_eta_ks_count():
    """ks = last level (0-indexed) where bk < eps."""
    ak = jnp.asarray([100.0, 200.0, 500.0, 1000.0, 5000.0])
    bk = jnp.asarray([0.0, 0.0, 1e-10, 0.3, 1.0])  # 3 pure-pressure levels
    _, ks = set_external_eta(ak, bk)
    # Last index with bk < eps (1e-7) is k=2 (bk[2]=1e-10 < 1e-7)
    assert ks == 2


def test_set_external_eta_all_hybrid():
    """All bk > eps → ks = -1 (no pure-pressure layers)."""
    ak = jnp.asarray([100.0, 200.0, 500.0])
    bk = jnp.asarray([0.001, 0.1, 1.0])  # all > eps=1e-7
    _, ks = set_external_eta(ak, bk)
    assert ks == -1


def test_compute_dz_L101_shape():
    """L101 vertical has 101 layers."""
    dz, _ = compute_dz_L101()
    assert dz.shape == (101,)


def test_compute_dz_L101_uniform_bottom():
    """Bottom 77 layers (k=24..100, 0-indexed) are uniform dz0=40 m."""
    dz, _ = compute_dz_L101()
    # FV3 first loop: do k=km, k0, -1 sets dz[k0=24..km-1=100] = dz0
    # 0-indexed: dz[24:101] = 40 m (but second loop overwrites dz[1:25])
    # So uniform region is dz[25:101] = 40 m
    assert jnp.allclose(dz[25:], 40.0, atol=1e-12)


def test_compute_dz_L101_geometric_middle():
    """Middle layers (k=1..24, 0-indexed) follow geometric stretch."""
    dz, _ = compute_dz_L101()
    stretch_f = 1.16
    # dz[k] = stretch_f * dz[k+1] for k in [1, 24]
    for k in range(1, 25):
        expected = float(stretch_f * dz[k + 1])
        assert abs(float(dz[k]) - expected) / max(expected, 1.0) < 1e-10, (
            f"k={k}: dz={float(dz[k])}, expected={expected}"
        )


def test_compute_dz_L101_top_quadrupled():
    """Top layer dz[0] = 4·dz[1]."""
    dz, _ = compute_dz_L101()
    assert abs(float(dz[0]) - 4.0 * float(dz[1])) < 1e-8


def test_compute_dz_L101_ztop_approx_20km():
    """ztop ≈ 20.3 km (FV3 documented value)."""
    _, ztop = compute_dz_L101()
    ztop_km = float(ztop) / 1000.0
    assert 18.0 < ztop_km < 22.0, f"ztop = {ztop_km} km (expected ~20.3)"

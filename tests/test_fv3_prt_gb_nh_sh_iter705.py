"""FV3_3D iter 705: prt_gb_nh_sh_fv3 port.

Faithful JAX port of FV3 ``prt_gb_nh_sh`` (tools/fv_diagnostics.F90:
4462-4509).  Lat-band area-weighted mean diagnostic.

Tests
-----

1. ``test_prt_gb_nh_sh_uniform_field``.
2. ``test_prt_gb_nh_sh_band_separation``.
3. ``test_prt_gb_nh_sh_empty_band_returns_minus_one``.
4. ``test_prt_gb_nh_sh_global_uniform``.
5. ``test_prt_gb_nh_sh_area_weighting``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import prt_gb_nh_sh_fv3


def test_prt_gb_nh_sh_uniform_field():
    """Uniform field 5.0 → all bands return 5.0 (or -1 if empty)."""
    n = 10
    lat = jnp.linspace(-jnp.pi / 2 * 0.9, jnp.pi / 2 * 0.9, n)
    a2 = jnp.full((n,), 5.0)
    area = jnp.ones((n,)) * 1000.0
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    assert abs(out["gb"] - 5.0) < 1e-12
    # NH/SH/eq bands all have ≥1 sample → 5.0
    assert abs(out["nh"] - 5.0) < 1e-12
    assert abs(out["sh"] - 5.0) < 1e-12
    assert abs(out["eq"] - 5.0) < 1e-12


def test_prt_gb_nh_sh_band_separation():
    """Field = lat_deg: each band mean = midpoint of its lat range.
    NH (20..80°): mean ≈ 50.
    SH (-80..-20°): mean ≈ -50.
    EQ (-20..20°): mean ≈ 0.
    """
    n = 181  # 1° spacing
    lat_deg = jnp.linspace(-89.5, 89.5, n)
    lat = lat_deg * jnp.pi / 180.0
    a2 = lat_deg
    # Uniform area (1° lat spacing → equal-area-band approximation OK for test)
    area = jnp.ones((n,))
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    # NH mean = average lat in [20, 80) ≈ 49.5 (uniform 1° points 20..79 inclusive)
    assert 49.0 < out["nh"] < 51.0
    assert -51.0 < out["sh"] < -49.0
    assert abs(out["eq"]) < 1.5  # equatorial mean near 0


def test_prt_gb_nh_sh_empty_band_returns_minus_one():
    """Single cell at lat=0 (eq only) → nh/sh return -1.0."""
    lat = jnp.array([0.0])
    a2 = jnp.array([3.7])
    area = jnp.array([2.0])
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    # eq band has 1 cell with area=2.0 > 1.0 → returns 3.7
    # nh/sh have no cells → area_band = 0 ≤ 1.0 → returns -1.0
    assert abs(out["eq"] - 3.7) < 1e-12
    assert out["nh"] == -1.0
    assert out["sh"] == -1.0


def test_prt_gb_nh_sh_global_uniform():
    """Global mean of uniform field = that field value (full sphere)."""
    rng = np.random.default_rng(seed=705)
    n = 200
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2, jnp.pi / 2, size=(n,)))
    a2 = jnp.full((n,), 42.0)
    area = jnp.asarray(rng.uniform(0.5, 2.0, size=(n,)))
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    assert abs(out["gb"] - 42.0) < 1e-12


def test_prt_gb_nh_sh_area_weighting():
    """Two cells at same lat band with different areas → weighted mean."""
    lat = jnp.array([0.0, 0.0])     # both equatorial
    a2 = jnp.array([10.0, 20.0])
    area = jnp.array([1.0, 3.0])    # 2:6 weights → mean = (10·1 + 20·3)/4 = 17.5
    out = prt_gb_nh_sh_fv3(a2, area, lat)
    assert abs(out["eq"] - 17.5) < 1e-12
    assert abs(out["gb"] - 17.5) < 1e-12

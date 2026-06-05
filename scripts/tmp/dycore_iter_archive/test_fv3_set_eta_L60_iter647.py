"""FV3_3D iter 647: set_eta_L60 port.

Faithful JAX port of FV3 L60 hardcoded a60/b60 hybrid-coord
table (tools/fv_eta.F90:45-85).  FV3 reference for 60-layer
baroclinic-instability tests, equivalent to NCEP GFS L64 (top
3 layers differ).

Tests
-----

1. ``test_set_eta_L60_shapes``.
2. ``test_set_eta_L60_ptop_300``.
3. ``test_set_eta_L60_bk_endpoints``.
4. ``test_set_eta_L60_ak_endpoints``.
5. ``test_set_eta_L60_bk_monotonic``.
6. ``test_set_eta_L60_ks_pure_pressure``.
7. ``test_set_eta_L60_drives_hydro_eq``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import (
    hydro_eq,
    set_eta_L60,
)


def test_set_eta_L60_shapes():
    """ak, bk shape (61,); ptop scalar; ks int."""
    ak, bk, ptop, ks = set_eta_L60()
    assert ak.shape == (61,)
    assert bk.shape == (61,)
    assert ptop.shape == ()
    assert isinstance(ks, int)


def test_set_eta_L60_ptop_300():
    """ptop = ak[0] = 300 Pa (FV3 L60 reference)."""
    _, _, ptop, _ = set_eta_L60()
    assert abs(float(ptop) - 300.0) < 1e-10


def test_set_eta_L60_bk_endpoints():
    """bk[0] = 0 (pure pressure top); bk[60] = 1 (pure sigma bottom)."""
    _, bk, _, _ = set_eta_L60()
    assert abs(float(bk[0])) < 1e-10
    assert abs(float(bk[60]) - 1.0) < 1e-10


def test_set_eta_L60_ak_endpoints():
    """ak[60] = 0 (pure sigma at surface)."""
    ak, _, _, _ = set_eta_L60()
    assert abs(float(ak[60])) < 1e-10


def test_set_eta_L60_bk_monotonic():
    """bk monotonically increasing."""
    _, bk, _, _ = set_eta_L60()
    diffs = bk[1:] - bk[:-1]
    assert jnp.all(diffs >= 0.0), (
        f"bk not monotonic: min diff = {float(jnp.min(diffs))}"
    )


def test_set_eta_L60_ks_pure_pressure():
    """ks counts pure-pressure layers; FV3 L60 has ~21 (first 21 bk = 0)."""
    _, _, _, ks = set_eta_L60()
    # 0-indexed: bk[0..20] = 0 → ks = 20 (last 0-index)
    assert 15 < ks < 25, f"ks = {ks}, expected ~20"


def test_set_eta_L60_drives_hydro_eq():
    """set_eta_L60 ak/bk → hydro_eq builds finite IC."""
    ak, bk, _, _ = set_eta_L60()
    hs = jnp.zeros((2, 2))
    ps, delp, pt = hydro_eq(ak, bk, hs, drym=1.0e5, mountain=False)
    assert jnp.all(jnp.isfinite(ps))
    assert jnp.all(jnp.isfinite(delp))
    assert jnp.all(jnp.isfinite(pt))
    assert jnp.all(delp > 0)
    assert jnp.all(pt >= 200.0 - 1e-10)

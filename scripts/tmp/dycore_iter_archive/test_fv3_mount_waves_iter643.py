"""FV3_3D iter 643: mount_waves port.

Faithful JAX port of FV3 ``mount_waves`` (tools/fv_eta.F90:
2346-2479, NO_UKMO_HB branch).  HIWPP mountain-wave hybrid-coord init.

Tests
-----

1. ``test_mount_waves_shapes``.
2. ``test_mount_waves_ak_bk_endpoints``.
3. ``test_mount_waves_pe_monotonic``.
4. ``test_mount_waves_pure_pressure_layers``.
5. ``test_mount_waves_finite``.
6. ``test_mount_waves_km_too_small_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import mount_waves


def test_mount_waves_shapes():
    """ak, bk shapes (km+1,); ks int; pint, ptop scalars."""
    km = 30
    ak, bk, ptop, ks, pint_out = mount_waves(km)
    assert ak.shape == (km + 1,)
    assert bk.shape == (km + 1,)
    assert isinstance(ks, int)
    assert ptop.shape == ()
    assert pint_out.shape == ()


def test_mount_waves_ak_bk_endpoints():
    """ak[0] = ptop; ak[km] = 0; bk[km] = 1; bk[0] = 0."""
    km = 30
    ak, bk, ptop, _, _ = mount_waves(km)
    assert abs(float(ak[0]) - float(ptop)) < 1e-10
    assert abs(float(ak[km])) < 1e-14
    assert abs(float(bk[km]) - 1.0) < 1e-14
    assert abs(float(bk[0])) < 1e-14


def test_mount_waves_pe_monotonic():
    """Pressure pe = ak + bk·p00 monotonically increases top→bottom."""
    km = 30
    p00 = 1.0e5
    ak, bk, _, _, _ = mount_waves(km)
    pe = ak + bk * p00
    diffs = pe[1:] - pe[:-1]
    assert jnp.all(diffs >= 0), f"pe not monotonic: min diff = {float(jnp.min(diffs))}"


def test_mount_waves_pure_pressure_layers():
    """Pure-pressure layers (k ≤ ks) have bk = 0."""
    km = 30
    ak, bk, _, ks, _ = mount_waves(km)
    if ks > 0:
        assert jnp.all(bk[:ks + 1] == 0.0)


def test_mount_waves_finite():
    """No NaN/Inf in outputs."""
    km = 30
    ak, bk, ptop, _, pint_out = mount_waves(km)
    assert jnp.all(jnp.isfinite(ak))
    assert jnp.all(jnp.isfinite(bk))
    assert jnp.isfinite(ptop)
    assert jnp.isfinite(pint_out)


def test_mount_waves_km_too_small_raises():
    """km < 23 raises ValueError."""
    with pytest.raises(ValueError):
        mount_waves(km=20)

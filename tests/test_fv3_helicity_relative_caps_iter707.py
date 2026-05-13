"""FV3_3D iter 707: helicity_relative_caps_fv3 port.

Faithful JAX port of FV3 ``helicity_relative_CAPS`` (tools/
fv_diagnostics.F90:4894-4967).  Variant of iter-687 SRH with
external user-supplied storm motion (uc, vc) — pairs with iter-689
Bunkers right-mover storm-motion predictor.

Tests
-----

1. ``test_srh_caps_zero_shear``.
2. ``test_srh_caps_zero_wind``.
3. ``test_srh_caps_matches_iter687_when_uc_equals_mean``.
4. ``test_srh_caps_with_bunkers``.
5. ``test_srh_caps_shapes_3d``.
6. ``test_srh_caps_hydrostatic_requires_args``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    bunkers_vector_fv3,
    helicity_relative_caps_fv3,
    helicity_relative_fv3,
)


def _column_setup(km=30, dz=-500.0):
    return jnp.full((km,), dz)


def test_srh_caps_zero_shear():
    """Uniform wind → no shear → SRH = 0 regardless of (uc, vc)."""
    km = 20
    delz = _column_setup(km)
    ua = jnp.full((km,), 10.0)
    va = jnp.full((km,), 5.0)
    uc = jnp.asarray(7.5)
    vc = jnp.asarray(-2.0)
    srh = helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz)
    assert abs(float(srh)) < 1e-10


def test_srh_caps_zero_wind():
    """All-zero wind → SRH = 0."""
    km = 20
    delz = _column_setup(km)
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    uc = jnp.asarray(0.0)
    vc = jnp.asarray(0.0)
    srh = helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz)
    assert abs(float(srh)) < 1e-12


def test_srh_caps_matches_iter687_when_uc_equals_mean():
    """iter-687 internally uses depth-weighted mean as (uc, vc).
    Passing that same mean to iter-707 should match iter-687
    output exactly."""
    km = 30
    delz = _column_setup(km)
    rng = np.random.default_rng(seed=707)
    ua = jnp.asarray(rng.normal(scale=10.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(km,)))

    # Compute depth-weighted mean wind in [0, 3000]
    dz = -delz
    cumsum = jnp.cumsum(dz[::-1])[::-1]
    zh_above = cumsum
    zh_below = zh_above - dz
    dz_eff = jnp.maximum(
        0.0, jnp.minimum(zh_above, 3000.0) - jnp.maximum(zh_below, 0.0)
    )
    total = jnp.sum(dz_eff)
    uc = jnp.sum(ua * dz_eff) / total
    vc = jnp.sum(va * dz_eff) / total

    srh_687 = helicity_relative_fv3(ua, va, delz=delz)
    srh_707 = helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz)
    assert abs(float(srh_687) - float(srh_707)) < 1e-9


def test_srh_caps_with_bunkers():
    """Pair iter-707 with iter-689 Bunkers (uc, vc) → finite SRH."""
    km = 30
    delz = _column_setup(km)
    rng = np.random.default_rng(seed=708)
    ua = jnp.asarray(rng.normal(scale=15.0, size=(km,)))
    va = jnp.asarray(rng.normal(scale=15.0, size=(km,)))
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    srh = helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz)
    assert jnp.isfinite(srh)


def test_srh_caps_shapes_3d():
    """3-D wind input + 2-D (uc, vc) → 2-D SRH output."""
    rng = np.random.default_rng(seed=709)
    n_x, n_y, km = 4, 5, 30
    ua = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    delz = jnp.full((n_x, n_y, km), -300.0)
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    srh = helicity_relative_caps_fv3(ua, va, uc, vc, delz=delz)
    assert srh.shape == (n_x, n_y)


def test_srh_caps_hydrostatic_requires_args():
    """hydrostatic=True without pt/q/peln raises."""
    km = 10
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    uc = jnp.asarray(0.0)
    vc = jnp.asarray(0.0)
    with pytest.raises(ValueError):
        helicity_relative_caps_fv3(ua, va, uc, vc, hydrostatic=True)

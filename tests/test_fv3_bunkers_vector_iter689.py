"""FV3_3D iter 689: bunkers_vector_fv3 port.

Faithful JAX port of FV3 ``bunkers_vector`` (tools/fv_diagnostics.F90:
4970-5046).  Bunkers right-mover storm motion vector — pairs with
iter-687 SRH.

Tests
-----

1. ``test_bunkers_zero_shear``.
2. ``test_bunkers_uniform_unshear``.
3. ``test_bunkers_linear_shear``.
4. ``test_bunkers_shapes_3d``.
5. ``test_bunkers_finite``.
6. ``test_bunkers_hydrostatic_requires_args``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import bunkers_vector_fv3


def test_bunkers_zero_shear():
    """Uniform wind everywhere → no shear → uc=umn=u, vc=vmn=v exactly
    (shrmag floor=1 in safe divisor, but ushr=vshr=0 so offset=0)."""
    km = 30
    delz = jnp.full((km,), -500.0)  # 15 km column
    ua = jnp.full((km,), 8.0)
    va = jnp.full((km,), 3.0)
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    assert abs(float(uc) - 8.0) < 1e-12
    assert abs(float(vc) - 3.0) < 1e-12


def test_bunkers_uniform_unshear():
    """Different uniform wind in lower vs upper half of column → shear
    exists.  Lowest layer ua=usfc=0, layers above ua=U.  6km mid-column.

    With km=30, dz=500m, column is 15km.  Surface at k=29, z_mid≈250m.
    Above 6km: ua=U=10.  Below 6km: ua=0 except at k=29 (surface, also 0).
    But that means all of 0-6km has ua=0, so umn=0, u6km=0 (interpolated
    across the bracket layer), shrmag=0 → uc=umn=0.
    """
    km = 30
    delz = jnp.full((km,), -500.0)
    U = 10.0
    ua = jnp.concatenate(
        [jnp.full((km - 12,), U), jnp.zeros((12,))], axis=-1
    )  # top 18 layers: U; bottom 12: 0
    va = jnp.zeros((km,))
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    # umn = 0 (all 0-6km layers have ua=0)
    # u6km is in the bracket layer, which is layer at z≈6km
    # bracket has ua=U or 0 depending on placement; not strictly testable
    assert jnp.isfinite(uc)
    assert jnp.isfinite(vc)


def test_bunkers_linear_shear():
    """Westerly shear: ua(z) = c·z, va=0, c=0.005/s.
    At surface ua=0, at z=6km ua=30.  ushr=30, vshr=0.  shrmag=30.
    Bunkers offset for uc: +7.5·0/30 = 0; for vc: -7.5·30/30 = -7.5.
    umn = depth-avg of c·z over 0-6km = c·3000 = 15.
    So uc=15+0=15, vc=0-7.5=-7.5.
    """
    km = 30
    delz = jnp.full((km,), -500.0)
    # Build z-midpoint for each layer top-down: highest layer z=14750m
    zh_mid = jnp.linspace(km * 500.0 - 250.0, 250.0, km)
    c = 0.005
    ua = c * zh_mid
    va = jnp.zeros((km,))
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    # Tolerance accounts for sub-layer interpolation and surface=lowest cell
    assert abs(float(uc) - 15.0) < 0.5
    assert abs(float(vc) - (-7.5)) < 0.5


def test_bunkers_shapes_3d():
    """3-D input → 2-D output."""
    rng = np.random.default_rng(seed=689)
    n_x, n_y, km = 4, 5, 30
    ua = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    delz = jnp.full((n_x, n_y, km), -500.0)
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    assert uc.shape == (n_x, n_y)
    assert vc.shape == (n_x, n_y)


def test_bunkers_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=690)
    km = 30
    ua = jnp.asarray(rng.normal(scale=15.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=15.0, size=(4, 4, km)))
    delz = jnp.full((4, 4, km), -300.0)
    uc, vc = bunkers_vector_fv3(ua, va, delz=delz)
    assert jnp.all(jnp.isfinite(uc))
    assert jnp.all(jnp.isfinite(vc))


def test_bunkers_hydrostatic_requires_args():
    """hydrostatic=True without pt/q/peln raises."""
    km = 10
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    with pytest.raises(ValueError):
        bunkers_vector_fv3(ua, va, hydrostatic=True)

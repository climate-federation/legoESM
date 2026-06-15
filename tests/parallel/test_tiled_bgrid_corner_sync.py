"""U3e: tile the POINTWISE local<->geographic corner conversion of the
BGRID_NE corner sync (task #3 cube np>6 stage).

synchronize_bgrid_ne_corner_geo = bgrid_ne_corner_to_geo (pointwise) ->
synchronize_corner_scalar (the CROSS-FACE/cross-tile sync, a later
increment) -> bgrid_ne_corner_from_geo (pointwise).  The two pointwise
conversions read only their own corner's (u,v,z*) so they tile with NO
halo; this pins them (exact inverse pair + per-tile == global).  The
cross-tile scalar sync (the d2a2c-corner-rounds pattern) remains the
deliberate hard piece.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import (
    bgrid_ne_corner_to_geo,
    bgrid_ne_corner_from_geo,
)


def _setup(kt=3, nl=6):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    return n, cd


def test_corner_geo_conversion_exact_inverse():
    """from_geo ∘ to_geo == identity (adjugate cancels det) on real cube
    corner z-metrics — the pair is an exact invertible frame change."""
    n, cd = _setup()
    z11, z12 = cd.cos_angle_corner, cd.sin_angle_corner
    z21, z22 = cd.z21_corner, cd.z22_corner
    rng = np.random.default_rng(7)
    u = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    v = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    ue, un = bgrid_ne_corner_to_geo(u, v, z11, z12, z21, z22)
    u2, v2 = bgrid_ne_corner_from_geo(ue, un, z11, z12, z21, z22)

    # INDEPENDENT inline oracle (codex U3e MED: a per-tile==global check is
    # helper-vs-helper and would miss a COHERENT but wrong paired transform;
    # pin both to_geo and from_geo against the literal FV3 z-matrix formula).
    z = [np.asarray(x) for x in (z11, z12, z21, z22)]
    z11n, z12n, z21n, z22n = z
    un_v, vn_v = np.asarray(u), np.asarray(v)
    eps = float(jnp.finfo(jnp.float32).eps)
    det = z11n * z22n - z21n * z12n
    inv = 1.0 / np.where(np.abs(det) > eps, det, 1.0)
    ue_o = (z11n * un_v + z21n * vn_v) * inv
    un_o = (z12n * un_v + z22n * vn_v) * inv
    np.testing.assert_allclose(np.asarray(ue), ue_o, atol=1e-12, rtol=1e-12,
                               err_msg="to_geo != FV3 oracle")
    np.testing.assert_allclose(np.asarray(un), un_o, atol=1e-12, rtol=1e-12)
    us_o = z22n * ue_o - z21n * un_o
    vs_o = -z12n * ue_o + z11n * un_o
    np.testing.assert_allclose(np.asarray(u2), us_o, atol=1e-12, rtol=1e-12,
                               err_msg="from_geo != FV3 oracle")
    np.testing.assert_allclose(np.asarray(v2), vs_o, atol=1e-12, rtol=1e-12)

    # Inverse identity: from_geo∘to_geo == (u,v).  The cube corner z-metrics
    # are FLOAT32 (probe job 8480531: z11=0.8660254=float32 √3/2, det∈[0.866,1]
    # everywhere — no degenerate corners), so adj(A)·A=det·I holds only to
    # ~float32 precision (~1.2e-7); the EXACT validation is the oracle above.
    np.testing.assert_allclose(np.asarray(u2), un_v, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(v2), vn_v, atol=2e-6, rtol=2e-6)
    # non-vacuity: the geo frame actually differs from the local frame.
    assert float(np.max(np.abs(np.asarray(ue) - un_v))) > 1e-6


def test_corner_geo_conversion_tiles_no_halo():
    """to_geo / from_geo on a per-tile corner block (nl+1) == the global
    result sliced to that block — pointwise, no halo, exact."""
    kt, nl = 3, 6
    n, cd = _setup(kt, nl)
    z11, z12 = cd.cos_angle_corner, cd.sin_angle_corner
    z21, z22 = cd.z21_corner, cd.z22_corner
    rng = np.random.default_rng(9)
    u = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    v = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    ue_g, un_g = bgrid_ne_corner_to_geo(u, v, z11, z12, z21, z22)
    us_g, vs_g = bgrid_ne_corner_from_geo(ue_g, un_g, z11, z12, z21, z22)

    for ti in range(kt):
        for tj in range(kt):
            ai, aj = ti * nl, tj * nl
            sl = (slice(None), slice(ai, ai + nl + 1), slice(aj, aj + nl + 1))
            # to_geo per-tile == global sliced
            ue_t, un_t = bgrid_ne_corner_to_geo(
                u[sl], v[sl], z11[sl], z12[sl], z21[sl], z22[sl])
            np.testing.assert_allclose(
                np.asarray(ue_t), np.asarray(ue_g)[sl], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                np.asarray(un_t), np.asarray(un_g)[sl], atol=1e-12, rtol=1e-12)
            # from_geo per-tile == global sliced (codex U3e MED: pin from_geo too)
            us_t, vs_t = bgrid_ne_corner_from_geo(
                ue_g[sl], un_g[sl], z11[sl], z12[sl], z21[sl], z22[sl])
            np.testing.assert_allclose(
                np.asarray(us_t), np.asarray(us_g)[sl], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                np.asarray(vs_t), np.asarray(vs_g)[sl], atol=1e-12, rtol=1e-12)

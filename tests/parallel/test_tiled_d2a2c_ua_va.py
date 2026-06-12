"""P4 phase-1b: d2a2c approach-C first A→C output (ua/va), tile-local.

Approach C: the cheap D→A step (d2a2c_d_to_a) runs in the global view;
the A→C is tiled.  ua/va are pure pointwise cell algebra from the
covariant utmp/vtmp (d2a2c_ua_va_local), so they tile with no halo.

Pins:
1. d2a2c_ua_va_local on the global d2a2c_d_to_a output EQUALS the
   production d2a2c_vect ua/va EXACTLY (ties the new helper to the live
   operator; also a regression check on the d2a2c_d_to_a extraction).
2. The tile-local helper assembled over tiles matches the global ua/va.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.core.fv3_sw_core import (
    d2a2c_vect,
    d2a2c_d_to_a,
    d2a2c_ua_va_local,
    d2a2c_uc_4th_local,
    d2a2c_vc_4th_local,
    d2a2c_ut_vt_local,
    d2a2c_interior_local,
)
from legoesm.parallel.mesh import (
    tiled_face_block, tiled_padded_block, staggered_tile_block)

N = 24
KT = 2
NL = N // KT
H = 2


@pytest.fixture(scope="module")
def setup():
    set_halo_backend("local")
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    rng = np.random.default_rng(9)
    u_d = jnp.asarray(rng.standard_normal((6, N, N + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, N + 1, N)))
    ua_g, va_g, *_ = d2a2c_vect(u_d, v_d, cdg)
    return cdg, u_d, v_d, np.asarray(ua_g), np.asarray(va_g)


def test_helper_matches_production_d2a2c(setup):
    """d2a2c_ua_va_local on the global D→A output == d2a2c_vect ua/va.
    Also exercises d2a2c_d_to_a (regression on the extraction)."""
    cdg, u_d, v_d, ua_g, va_g = setup
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    utmp_int = utmp_pad[:, H:-H, H:-H]
    vtmp_int = vtmp_pad[:, H:-H, H:-H]
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    ua_h, va_h = d2a2c_ua_va_local(utmp_int, vtmp_int, cos_sg5, rsin2)
    np.testing.assert_array_equal(np.asarray(ua_h), ua_g)
    np.testing.assert_array_equal(np.asarray(va_h), va_g)


@pytest.mark.parametrize("kt", (2, 3))
def test_tiled_ua_va_matches_global(setup, kt):
    cdg, u_d, v_d, ua_g, va_g = setup
    nl = N // kt
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    utmp_int = np.asarray(utmp_pad[:, H:-H, H:-H])
    vtmp_int = np.asarray(vtmp_pad[:, H:-H, H:-H])
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                ut = tiled_face_block(jnp.asarray(utmp_int[f]), ti, tj, nl, kt)
                vt = tiled_face_block(jnp.asarray(vtmp_int[f]), ti, tj, nl, kt)
                cs = tiled_face_block(cos_sg5[f], ti, tj, nl, kt)
                rs = tiled_face_block(rsin2[f], ti, tj, nl, kt)
                ua_t, va_t = d2a2c_ua_va_local(
                    ut[None], vt[None], cs[None], rs[None])
                want = ua_g[f, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
                np.testing.assert_array_equal(np.asarray(ua_t)[0], want)


def test_tiled_ua_va_in_shardmap(setup):
    if len(jax.devices()) < 24:
        pytest.skip("needs --xla_force_host_platform_device_count=24")
    cdg, u_d, v_d, ua_g, va_g = setup
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    utmp_int = utmp_pad[:, H:-H, H:-H]
    vtmp_int = vtmp_pad[:, H:-H, H:-H]
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    dev = np.array(jax.devices()[:24]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    co = P("face", "tile_i", "tile_j")
    sh = NamedSharding(mesh, co)
    args = [jax.device_put(x, sh) for x in (utmp_int, vtmp_int, cos_sg5, rsin2)]

    @partial(shard_map, mesh=mesh, in_specs=(co, co, co, co),
             out_specs=(co, co), check_vma=False)
    def _stage(ut, vt, cs, rs):
        ua, va = d2a2c_ua_va_local(ut, vt, cs, rs)
        return ua, va

    ua_t, va_t = _stage(*args)
    np.testing.assert_array_equal(np.asarray(ua_t), ua_g)
    np.testing.assert_array_equal(np.asarray(va_t), va_g)


def test_uc_vc_4th_core_matches_production_interior(setup):
    """The extracted 4th-order uc/vc core matches the production
    d2a2c_vect uc/vc in the face-interior 4th-order band [npt+1:n-npt]
    (regression on the uc_4th/vc_4th extraction + ties the core to the
    live operator).  d2a2c_vect returns ua,va,uc,vc,ut,vt."""
    cdg, u_d, v_d, _ua, _va = setup
    _, _, uc_g, vc_g, _, _ = d2a2c_vect(u_d, v_d, cdg)
    uc_g, vc_g = np.asarray(uc_g), np.asarray(vc_g)
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    uc4 = np.asarray(d2a2c_uc_4th_local(utmp_pad))   # (6, n+1, n)
    vc4 = np.asarray(d2a2c_vc_4th_local(vtmp_pad))   # (6, n, n+1)
    npt = min(4, N // 2)
    lo, hi = npt + 1, N - npt
    # Global overlays uc_4th[lo-1:hi-1] into uc[lo:hi].
    np.testing.assert_array_equal(uc_g[:, lo:hi, :], uc4[:, lo - 1:hi - 1, :])
    np.testing.assert_array_equal(vc_g[:, :, lo:hi], vc4[:, :, lo - 1:hi - 1])


@pytest.mark.parametrize("kt", (2, 3))
def test_uc_vc_4th_core_tiles_exact(setup, kt):
    """The 4th-order core tiles EXACTLY: d2a2c_uc_4th_local on a tile's
    h2-padded utmp block (tiled_padded_block) == the global 4th-order uc
    restricted to that tile's u-faces (the stencil is local; the padded
    slice supplies the exact halo window)."""
    cdg, u_d, v_d, _ua, _va = setup
    nl = N // kt
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    uc4_g = np.asarray(d2a2c_uc_4th_local(utmp_pad))   # (6, n+1, n)
    vc4_g = np.asarray(d2a2c_vc_4th_local(vtmp_pad))   # (6, n, n+1)
    for f in range(6):
        upf = utmp_pad[f]
        vpf = vtmp_pad[f]
        for ti in range(kt):
            for tj in range(kt):
                ub = tiled_padded_block(upf, ti, tj, nl, kt)   # (nl+4,nl+4)
                vb = tiled_padded_block(vpf, ti, tj, nl, kt)
                uc_t = np.asarray(d2a2c_uc_4th_local(ub[None]))[0]  # (nl+1,nl)
                vc_t = np.asarray(d2a2c_vc_4th_local(vb[None]))[0]  # (nl,nl+1)
                uc_w = uc4_g[f, ti * nl: ti * nl + nl + 1,
                             tj * nl:(tj + 1) * nl]
                vc_w = vc4_g[f, ti * nl:(ti + 1) * nl,
                             tj * nl: tj * nl + nl + 1]
                np.testing.assert_array_equal(uc_t, uc_w)
                np.testing.assert_array_equal(vc_t, vc_w)


def test_ut_vt_base_matches_production_interior(setup):
    """d2a2c_ut_vt_local (the pointwise ut/vt base) matches production
    d2a2c_vect ut/vt in the deep interior (away from the face-boundary
    + adjacent-strip overrides)."""
    cdg, u_d, v_d, _ua, _va = setup
    _, _, uc_g, vc_g, ut_g, vt_g = d2a2c_vect(u_d, v_d, cdg)
    ut_b, vt_b = d2a2c_ut_vt_local(
        uc_g, vc_g, u_d, v_d,
        cdg.cosa_u, cdg.rsin_u, cdg.cosa_v, cdg.rsin_v)
    npt = min(4, N // 2)
    lo, hi = npt + 1, N - npt
    np.testing.assert_array_equal(
        np.asarray(ut_b)[:, lo:hi, lo:hi], np.asarray(ut_g)[:, lo:hi, lo:hi])
    np.testing.assert_array_equal(
        np.asarray(vt_b)[:, lo:hi, lo:hi], np.asarray(vt_g)[:, lo:hi, lo:hi])


@pytest.mark.parametrize("kt", (2, 3))
def test_ut_vt_base_tiles_exact(setup, kt):
    """The pointwise ut/vt base tiles EXACTLY via staggered slicing."""
    cdg, u_d, v_d, _ua, _va = setup
    nl = N // kt
    _, _, uc_g, vc_g, _, _ = d2a2c_vect(u_d, v_d, cdg)
    uc_g, vc_g = jnp.asarray(uc_g), jnp.asarray(vc_g)
    ut_b, vt_b = d2a2c_ut_vt_local(
        uc_g, vc_g, u_d, v_d,
        cdg.cosa_u, cdg.rsin_u, cdg.cosa_v, cdg.rsin_v)
    ut_b, vt_b = np.asarray(ut_b), np.asarray(vt_b)
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                # i-staggered (n+1,n): axis 0
                uc_t = staggered_tile_block(uc_g[f], ti, tj, nl, 0)
                vd_t = staggered_tile_block(v_d[f], ti, tj, nl, 0)
                cau = staggered_tile_block(cdg.cosa_u[f], ti, tj, nl, 0)
                rsu = staggered_tile_block(cdg.rsin_u[f], ti, tj, nl, 0)
                # j-staggered (n,n+1): axis 1
                vc_t = staggered_tile_block(vc_g[f], ti, tj, nl, 1)
                ud_t = staggered_tile_block(u_d[f], ti, tj, nl, 1)
                cav = staggered_tile_block(cdg.cosa_v[f], ti, tj, nl, 1)
                rsv = staggered_tile_block(cdg.rsin_v[f], ti, tj, nl, 1)
                ut_t, vt_t = d2a2c_ut_vt_local(
                    uc_t[None], vc_t[None], ud_t[None], vd_t[None],
                    cau[None], rsu[None], cav[None], rsv[None])
                ut_w = ut_b[f, ti * nl: ti * nl + nl + 1, tj * nl:(tj + 1) * nl]
                vt_w = vt_b[f, ti * nl:(ti + 1) * nl, tj * nl: tj * nl + nl + 1]
                np.testing.assert_array_equal(np.asarray(ut_t)[0], ut_w)
                np.testing.assert_array_equal(np.asarray(vt_t)[0], vt_w)


def test_d2a2c_interior_tile_full_parity():
    """FULL tiled d2a2c on an INTERIOR tile == PRODUCTION d2a2c_vect.

    kt=3 (interior tile (1,1) touches no face edge -> pure 4th-order,
    no edge specials).  d2a2c_interior_local slices the CORRECT
    per-output windows (asymmetric -1 for uc/vc on their staggered
    axis); compared to the live d2a2c_vect outputs (NOT uc_4th) — the
    check that the prior off-by-one version failed.  Host
    slice-reassemble."""
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt           # 8
    npt = min(4, n // 2)   # 4
    assert nl >= npt + 1
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(13)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))

    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell

    ti = tj = 1
    a, b = ti * nl, (ti + 1) * nl
    for f in range(6):
        out = d2a2c_interior_local(
            utmp_pad[f], vtmp_pad[f], ti, tj, nl,
            staggered_tile_block(u_d[f], ti, tj, nl, 1)[None],   # (nl,nl+1)
            staggered_tile_block(v_d[f], ti, tj, nl, 0)[None],   # (nl+1,nl)
            tiled_face_block(cos_sg5[f], ti, tj, nl, kt)[None],
            tiled_face_block(rsin2[f], ti, tj, nl, kt)[None],
            staggered_tile_block(cdg.cosa_u[f], ti, tj, nl, 0)[None],
            staggered_tile_block(cdg.rsin_u[f], ti, tj, nl, 0)[None],
            staggered_tile_block(cdg.cosa_v[f], ti, tj, nl, 1)[None],
            staggered_tile_block(cdg.rsin_v[f], ti, tj, nl, 1)[None],
        )
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_array_equal(ua_t, ua_g[f, a:b, a:b])
        np.testing.assert_array_equal(va_t, va_g[f, a:b, a:b])
        np.testing.assert_array_equal(uc_t, uc_g[f, a:b + 1, a:b])
        np.testing.assert_array_equal(vc_t, vc_g[f, a:b, a:b + 1])
        np.testing.assert_array_equal(ut_t, ut_g[f, a:b + 1, a:b])
        np.testing.assert_array_equal(vt_t, vt_g[f, a:b, a:b + 1])

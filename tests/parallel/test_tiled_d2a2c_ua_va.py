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
    d2a2c_uc_c123_local,
    d2a2c_uc_edge_interp_local,
    d2a2c_vc_c123_local,
    d2a2c_vc_edge_interp_local,
    d2a2c_edge_w_local,
    d2a2c_edge_e_local,
    d2a2c_edge_s_local,
    d2a2c_edge_n_local,
    d2a2c_corner_local,
    d2a2c_global_fields,
    d2a2c_adjacent_strips,
    d2a2c_uc_tile_unified,
    d2a2c_vc_tile_unified,
    d2a2c_tile_unified,
    d2a2c_tile_strips,
    _A1 as _FV3_A1,
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


def test_uc_c123_edge_matches_production():
    """The one-sided C1/C2/C3 uc edge special matches production
    d2a2c_vect at face i=1 (W edge tile) and i=n-1 (E edge tile).
    kt=3: W tile (0,1), E tile (2,1).  First edge-special piece."""
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(21)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    uc_g = np.asarray(d2a2c_vect(u_d, v_d, cdg)[2])  # (6, n+1, n)
    utmp_pad, _ = d2a2c_d_to_a(u_d, v_d, cdg)
    tj = 1
    b = tj * nl
    for f in range(6):
        win_w = tiled_padded_block(utmp_pad[f], 0, tj, nl, kt)
        c1_w = np.asarray(d2a2c_uc_c123_local(win_w[None], False))[0]  # (nl,)
        np.testing.assert_array_equal(c1_w, uc_g[f, 1, b:b + nl])
        win_e = tiled_padded_block(utmp_pad[f], kt - 1, tj, nl, kt)
        c1_e = np.asarray(d2a2c_uc_c123_local(win_e[None], True))[0]
        np.testing.assert_array_equal(c1_e, uc_g[f, n - 1, b:b + nl])


def test_uc_edge_interp_west_matches_production():
    """edge_interpolate4 + upwind sin_sg uc at the W face boundary
    (i=0) matches production d2a2c_vect, tile-local (kt=3 W tile
    (0,1)).  The outer upwind sine is cross-face -> sin_sg pads are
    computed globally (pad_halo) and sliced per tile (approach C)."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(31)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    uc_g = np.asarray(d2a2c_vect(u_d, v_d, cdg)[2])

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad   # (6,n+4,n+4)
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdg.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdg.sin_sg[:, :, :, 0], interp_offsets=offsets)

    ti, tj = 0, 1
    b = tj * nl
    for f in range(6):
        ua_w = ua_pad[f, 0:nl + 4, b:b + nl + 4]
        dx_w = dxc_pad_x[f, 0:nl + 4, b:b + nl]
        se_w = se_pad_x[f, 0:nl + 2, b:b + nl + 2]
        sw_w = sw_pad_x[f, 0:nl + 2, b:b + nl + 2]
        uc_b = np.asarray(d2a2c_uc_edge_interp_local(
            ua_w[None], dx_w[None], se_w[None], sw_w[None], False))[0]
        np.testing.assert_allclose(
            uc_b, uc_g[f, 0, b:b + nl], rtol=0, atol=1e-12)


def test_uc_edge_interp_east_matches_production():
    """edge_interpolate4 + upwind uc at the E face boundary (i=n)
    matches production, tile-local (kt=3 E tile (2,1)).  The E stencil
    reads ua_pad[n+4] (OOB -> JAX clamps); the symmetric E-tile window
    [a:a+nl+4]=[n-nl:n+4] reproduces the same clamp."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(31)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    uc_g = np.asarray(d2a2c_vect(u_d, v_d, cdg)[2])

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5_pad = jnp.pad(cdg.cos_sg[:, :, :, 4],
                          [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(cdg.rsin2_cell, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdg.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdg.sin_sg[:, :, :, 0], interp_offsets=offsets)

    ti, tj = kt - 1, 1
    a, b = ti * nl, tj * nl
    for f in range(6):
        ua_w = ua_pad[f, a:a + nl + 4, b:b + nl + 4]
        dx_w = dxc_pad_x[f, a:a + nl + 4, b:b + nl]
        se_w = se_pad_x[f, a:a + nl + 2, b:b + nl + 2]
        sw_w = sw_pad_x[f, a:a + nl + 2, b:b + nl + 2]
        uc_b = np.asarray(d2a2c_uc_edge_interp_local(
            ua_w[None], dx_w[None], se_w[None], sw_w[None], True))[0]
        np.testing.assert_allclose(
            uc_b, uc_g[f, n, b:b + nl], rtol=0, atol=1e-12)


def test_vc_edge_specials_south_north_match_production():
    """vc C1/C2/C3 (j=1 S / j=n-1 N) and edge_interpolate4+upwind
    (j=0 S / j=n N) match production d2a2c_vect, tile-local at kt=3
    S tile (1,0) and N tile (1,2).  j-axis transpose of the uc W/E
    specials."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(41)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc_g = np.asarray(d2a2c_vect(u_d, v_d, cdg)[3])  # (6, n, n+1)

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5_pad = jnp.pad(cdg.cos_sg[:, :, :, 4],
                          [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(cdg.rsin2_cell, [(0, 0), (h, h), (h, h)], mode='edge')
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    offsets = grid.halo_interp_offsets
    sn_pad_y = pad_halo(cdg.sin_sg[:, :, :, 3], interp_offsets=offsets)
    ss_pad_y = pad_halo(cdg.sin_sg[:, :, :, 1], interp_offsets=offsets)

    # --- S tile (1,0): C1 at j=1, edge_interp at j=0 ---
    ti, tj = 1, 0
    a, b = ti * nl, tj * nl
    for f in range(6):
        win = tiled_padded_block(vtmp_pad[f], ti, tj, nl, kt)
        c1_s = np.asarray(d2a2c_vc_c123_local(win[None], False))[0]
        np.testing.assert_array_equal(c1_s, vc_g[f, a:a + nl, 1])
        va_w = va_pad[f, a:a + nl + 4, b:b + nl + 4]
        dy_w = dyc_pad_y[f, a:a + nl, b:b + nl + 4]
        sn_w = sn_pad_y[f, a:a + nl + 2, b:b + nl + 2]
        ss_w = ss_pad_y[f, a:a + nl + 2, b:b + nl + 2]
        vc_b = np.asarray(d2a2c_vc_edge_interp_local(
            va_w[None], dy_w[None], sn_w[None], ss_w[None], False))[0]
        np.testing.assert_allclose(vc_b, vc_g[f, a:a + nl, 0],
                                   rtol=0, atol=1e-12)

    # --- N tile (1,2): C1 at j=n-1, edge_interp at j=n ---
    ti, tj = 1, kt - 1
    a, b = ti * nl, tj * nl
    for f in range(6):
        win = tiled_padded_block(vtmp_pad[f], ti, tj, nl, kt)
        c1_n = np.asarray(d2a2c_vc_c123_local(win[None], True))[0]
        np.testing.assert_array_equal(c1_n, vc_g[f, a:a + nl, n - 1])
        va_w = va_pad[f, a:a + nl + 4, b:b + nl + 4]
        dy_w = dyc_pad_y[f, a:a + nl, b:b + nl + 4]
        sn_w = sn_pad_y[f, a:a + nl + 2, b:b + nl + 2]
        ss_w = ss_pad_y[f, a:a + nl + 2, b:b + nl + 2]
        vc_b = np.asarray(d2a2c_vc_edge_interp_local(
            va_w[None], dy_w[None], sn_w[None], ss_w[None], True))[0]
        np.testing.assert_allclose(vc_b, vc_g[f, a:a + nl, n],
                                   rtol=0, atol=1e-12)


def test_d2a2c_edge_w_full_parity():
    """Full W-edge-tile d2a2c == production d2a2c_vect (kt=3 W tile
    (0,1)), every output EXCEPT the adjacent-strip vt[0] (which needs a
    neighbour ut j-halo — stage-level).  Host slice-reassemble."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(51)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdg.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdg.sin_sg[:, :, :, 0], interp_offsets=offsets)

    ti, tj = 0, 1
    a, b = ti * nl, tj * nl
    for f in range(6):
        sb = lambda arr, ax: staggered_tile_block(arr[f], ti, tj, nl, ax)
        fb = lambda arr: tiled_face_block(arr[f], ti, tj, nl, kt)
        out = d2a2c_edge_w_local(
            utmp_pad[f], vtmp_pad[f], tj, nl, n,
            sb(u_d, 1)[None], sb(v_d, 0)[None],
            fb(cos_sg5)[None], fb(rsin2)[None],
            sb(cdg.cosa_u, 0)[None], sb(cdg.rsin_u, 0)[None],
            sb(cdg.cosa_v, 1)[None], sb(cdg.rsin_v, 1)[None],
            ua_pad[f], dxc_pad_x[f], se_pad_x[f], sw_pad_x[f])
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_allclose(ua_t, ua_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(va_t, va_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(ut_t, ut_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        # vt: exclude i=0 (the adjacent-strip; deferred).
        np.testing.assert_allclose(
            vt_t[1:nl], vt_g[f, a + 1:a + nl, b:b + nl + 1],
            rtol=0, atol=1e-12)


def test_d2a2c_edge_e_full_parity():
    """Full E-edge-tile d2a2c == production d2a2c_vect (kt=3 E tile
    (kt-1,1)), every output EXCEPT the adjacent-strip vt[nl-1] (which needs
    a neighbour ut j-halo — stage-level).  Host slice-reassemble."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(52)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdg.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdg.sin_sg[:, :, :, 0], interp_offsets=offsets)

    ti, tj = kt - 1, 1
    a, b = ti * nl, tj * nl
    for f in range(6):
        sb = lambda arr, ax: staggered_tile_block(arr[f], ti, tj, nl, ax)
        fb = lambda arr: tiled_face_block(arr[f], ti, tj, nl, kt)
        out = d2a2c_edge_e_local(
            utmp_pad[f], vtmp_pad[f], tj, nl, n,
            sb(u_d, 1)[None], sb(v_d, 0)[None],
            fb(cos_sg5)[None], fb(rsin2)[None],
            sb(cdg.cosa_u, 0)[None], sb(cdg.rsin_u, 0)[None],
            sb(cdg.cosa_v, 1)[None], sb(cdg.rsin_v, 1)[None],
            ua_pad[f], dxc_pad_x[f], se_pad_x[f], sw_pad_x[f])
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_allclose(ua_t, ua_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(va_t, va_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(ut_t, ut_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        # vt: exclude i=nl-1 (the E adjacent-strip; deferred).
        np.testing.assert_allclose(
            vt_t[0:nl - 1], vt_g[f, a:a + nl - 1, b:b + nl + 1],
            rtol=0, atol=1e-12)


@pytest.mark.parametrize("ti,tj", [(1, 1), (4, 4), (1, 4), (4, 1)])
def test_d2a2c_interior_small_tile_oob_parity(ti, tj):
    """INTERIOR-tile d2a2c == production for SMALL tiles (kt=6, nl=4 <
    npt+1) whose faces fall OUTSIDE the global 4th-order band [npt+1,n-npt).

    Non-vacuous gate for the 2nd-base + clamped-4th-overlay generalization:
    before it, d2a2c_interior_local used PURE 4th and diverged from
    production at the out-of-band faces.  Tiles (1,*) expose the low side
    (face 4 < npt+1=5); tiles (4,*) expose the high side (face 20 >=
    n-npt=20).  Interior tiles touch no global boundary, so no edge specials
    and no adjacent strip — every output matches fully."""
    set_halo_backend("local")
    n, kt = 24, 6
    nl = n // kt           # 4
    npt = min(4, n // 2)   # 4
    assert nl < npt + 1    # the regime the old precondition forbade
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(60 + ti * 6 + tj)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    a, b = ti * nl, tj * nl
    for f in range(6):
        out = d2a2c_interior_local(
            utmp_pad[f], vtmp_pad[f], ti, tj, nl,
            staggered_tile_block(u_d[f], ti, tj, nl, 1)[None],
            staggered_tile_block(v_d[f], ti, tj, nl, 0)[None],
            tiled_face_block(cos_sg5[f], ti, tj, nl, kt)[None],
            tiled_face_block(rsin2[f], ti, tj, nl, kt)[None],
            staggered_tile_block(cdg.cosa_u[f], ti, tj, nl, 0)[None],
            staggered_tile_block(cdg.rsin_u[f], ti, tj, nl, 0)[None],
            staggered_tile_block(cdg.cosa_v[f], ti, tj, nl, 1)[None],
            staggered_tile_block(cdg.rsin_v[f], ti, tj, nl, 1)[None],
        )
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_allclose(ua_t, ua_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(va_t, va_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(ut_t, ut_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vt_t, vt_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)


@pytest.mark.parametrize("side", ["W", "E"])
def test_d2a2c_edge_we_small_tile_vc_parity(side):
    """W/E-edge-tile d2a2c == production for a SMALL tile (kt=6, nl=4,
    tj=1) whose j-faces fall outside the 4th-order band — non-vacuous gate
    for the edge tiles' vc 2nd-base+clamped-4th-overlay fix (local j-face 0
    is global j=4 < npt+1=5, so production uses the 2nd base there).  All
    outputs except the deferred adjacent-strip vt are compared."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 6
    nl = n // kt           # 4
    npt = min(4, n // 2)   # 4
    assert nl < npt + 1
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(70 if side == "W" else 71)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))
    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdg.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdg.sin_sg[:, :, :, 0], interp_offsets=offsets)

    fn = d2a2c_edge_w_local if side == "W" else d2a2c_edge_e_local
    ti, tj = (0 if side == "W" else kt - 1), 1
    a, b = ti * nl, tj * nl
    for f in range(6):
        sb = lambda arr, ax: staggered_tile_block(arr[f], ti, tj, nl, ax)
        fb = lambda arr: tiled_face_block(arr[f], ti, tj, nl, kt)
        out = fn(
            utmp_pad[f], vtmp_pad[f], tj, nl, n,
            sb(u_d, 1)[None], sb(v_d, 0)[None],
            fb(cos_sg5)[None], fb(rsin2)[None],
            sb(cdg.cosa_u, 0)[None], sb(cdg.rsin_u, 0)[None],
            sb(cdg.cosa_v, 1)[None], sb(cdg.rsin_v, 1)[None],
            ua_pad[f], dxc_pad_x[f], se_pad_x[f], sw_pad_x[f])
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(ut_t, ut_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        # vt: exclude the deferred adjacent strip (W->local 0, E->local nl-1).
        keep = slice(1, nl) if side == "W" else slice(0, nl - 1)
        gkeep = (slice(a + 1, a + nl) if side == "W"
                 else slice(a, a + nl - 1))
        np.testing.assert_allclose(
            vt_t[keep], vt_g[f, gkeep, b:b + nl + 1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("side", ["S", "N"])
def test_d2a2c_edge_sn_full_parity(side):
    """Full S/N-edge-tile d2a2c == production d2a2c_vect (kt=3, ti=1),
    every output EXCEPT the adjacent-strip ut (which needs a neighbour vt
    i-halo — stage-level).  j-axis transpose of the W/E full-parity tests."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n, kt = 24, 3
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(53 if side == "S" else 54)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    offsets = grid.halo_interp_offsets
    sn_pad_y = pad_halo(cdg.sin_sg[:, :, :, 3], interp_offsets=offsets)
    ss_pad_y = pad_halo(cdg.sin_sg[:, :, :, 1], interp_offsets=offsets)

    fn = d2a2c_edge_s_local if side == "S" else d2a2c_edge_n_local
    ti, tj = 1, (0 if side == "S" else kt - 1)
    a, b = ti * nl, tj * nl
    for f in range(6):
        sb = lambda arr, ax: staggered_tile_block(arr[f], ti, tj, nl, ax)
        fb = lambda arr: tiled_face_block(arr[f], ti, tj, nl, kt)
        out = fn(
            utmp_pad[f], vtmp_pad[f], ti, nl, n,
            sb(u_d, 1)[None], sb(v_d, 0)[None],
            fb(cos_sg5)[None], fb(rsin2)[None],
            sb(cdg.cosa_u, 0)[None], sb(cdg.rsin_u, 0)[None],
            sb(cdg.cosa_v, 1)[None], sb(cdg.rsin_v, 1)[None],
            va_pad[f], dyc_pad_y[f], sn_pad_y[f], ss_pad_y[f])
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_allclose(ua_t, ua_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(va_t, va_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vt_t, vt_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        # ut: exclude the deferred adjacent strip (S->local j=0, N->j=nl-1).
        keep = slice(1, nl) if side == "S" else slice(0, nl - 1)
        gkeep = (slice(b + 1, b + nl) if side == "S"
                 else slice(b, b + nl - 1))
        np.testing.assert_allclose(
            ut_t[:, keep], ut_g[f, a:a + nl + 1, gkeep], rtol=0, atol=1e-12)


@pytest.mark.parametrize("kt,ti,tj", [
    (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),   # kt=2: ALL tiles are corners
    (4, 0, 0), (4, 3, 3), (4, 0, 3), (4, 3, 0),   # kt=4: the 4 face-corners
])
def test_d2a2c_corner_full_parity(kt, ti, tj):
    """Full corner-tile d2a2c == production d2a2c_vect, every output EXCEPT
    the two deferred adjacent strips (vt i-strip row + ut j-strip column,
    each a stage-level transport-wind halo).  kt=2 -> all 4 tiles are corners
    (the np=24 sub-face unlock); kt=4 -> the 4 face-corner tiles."""
    from legoesm.grids.halo import pad_halo
    set_halo_backend("local")
    n = 24
    nl = n // kt
    h = 2
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    grid = cdg.base
    rng = np.random.default_rng(80 + kt * 16 + ti * 4 + tj)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    ua_g, va_g, uc_g, vc_g, ut_g, vt_g = (
        np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg))

    utmp_pad, vtmp_pad = d2a2c_d_to_a(u_d, v_d, cdg)
    cos_sg5 = cdg.cos_sg[:, :, :, 4]
    rsin2 = cdg.rsin2_cell
    cos_sg5_pad = jnp.pad(cos_sg5, [(0, 0), (h, h), (h, h)], mode='edge')
    rsin2_pad = jnp.pad(rsin2, [(0, 0), (h, h), (h, h)], mode='edge')
    ua_pad = (utmp_pad - vtmp_pad * cos_sg5_pad) * rsin2_pad
    va_pad = (vtmp_pad - utmp_pad * cos_sg5_pad) * rsin2_pad
    dxc_pad_x = jnp.pad(grid.dx, [(0, 0), (h, h), (0, 0)], mode='edge')
    dyc_pad_y = jnp.pad(grid.dy, [(0, 0), (0, 0), (h, h)], mode='edge')
    offsets = grid.halo_interp_offsets
    se_pad_x = pad_halo(cdg.sin_sg[:, :, :, 2], interp_offsets=offsets)
    sw_pad_x = pad_halo(cdg.sin_sg[:, :, :, 0], interp_offsets=offsets)
    sn_pad_y = pad_halo(cdg.sin_sg[:, :, :, 3], interp_offsets=offsets)
    ss_pad_y = pad_halo(cdg.sin_sg[:, :, :, 1], interp_offsets=offsets)

    a, b = ti * nl, tj * nl
    i_strip = 0 if ti == 0 else nl - 1
    j_strip = 0 if tj == 0 else nl - 1
    for f in range(6):
        sb = lambda arr, ax: staggered_tile_block(arr[f], ti, tj, nl, ax)
        fb = lambda arr: tiled_face_block(arr[f], ti, tj, nl, kt)
        out = d2a2c_corner_local(
            utmp_pad[f], vtmp_pad[f], ti, tj, nl, n,
            sb(u_d, 1)[None], sb(v_d, 0)[None],
            fb(cos_sg5)[None], fb(rsin2)[None],
            sb(cdg.cosa_u, 0)[None], sb(cdg.rsin_u, 0)[None],
            sb(cdg.cosa_v, 1)[None], sb(cdg.rsin_v, 1)[None],
            ua_pad[f], dxc_pad_x[f], se_pad_x[f], sw_pad_x[f],
            va_pad[f], dyc_pad_y[f], sn_pad_y[f], ss_pad_y[f])
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (np.asarray(x)[0] for x in out)
        np.testing.assert_allclose(ua_t, ua_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(va_t, va_g[f, a:a + nl, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)
        # vt: exclude ONLY the deferred i-strip SEGMENT (production recomputes
        # vt[i_strip, j_face in [2,n-2]] -> local [2:nl+1] for tj=0, [0:nl-1]
        # for tj=kt-1).  Cells outside it (incl. the j-face override) ARE
        # checked: blank the segment in both, then compare the full arrays.
        vt_j0, vt_j1 = (2, nl + 1) if tj == 0 else (0, nl - 1)
        vt_cmp = np.asarray(vt_t).copy()
        vt_ref = np.asarray(vt_g[f, a:a + nl, b:b + nl + 1]).copy()
        vt_cmp[i_strip, vt_j0:vt_j1] = 0.0
        vt_ref[i_strip, vt_j0:vt_j1] = 0.0
        np.testing.assert_allclose(vt_cmp, vt_ref, rtol=0, atol=1e-12)
        # ut: exclude ONLY the deferred j-strip SEGMENT (ut[i_face in [2,n-2],
        # j_strip] -> local rows [2:nl+1] for ti=0, [0:nl-1] for ti=kt-1).
        ut_i0, ut_i1 = (2, nl + 1) if ti == 0 else (0, nl - 1)
        ut_cmp = np.asarray(ut_t).copy()
        ut_ref = np.asarray(ut_g[f, a:a + nl + 1, b:b + nl]).copy()
        ut_cmp[ut_i0:ut_i1, j_strip] = 0.0
        ut_ref[ut_i0:ut_i1, j_strip] = 0.0
        np.testing.assert_allclose(ut_cmp, ut_ref, rtol=0, atol=1e-12)


def _assemble_tiled_d2a2c(u_d, v_d, cdg, kt, unified=False):
    """Single-device reference for the tiled A->C stage: run every (ti,tj)
    sub-face tile's LOCAL kernel on the shared global fields, gather into the
    global staggered uc/vc/ut/vt, then apply the adjacent strips GLOBALLY (the
    single-device stand-in for the shard_map stage's deferred-strip neighbour
    halo).  Must equal d2a2c_vect bit-for-bit.  ``unified=True`` routes EVERY
    tile through the single flag/mask-driven d2a2c_tile_unified (the body the
    shard_map runs) instead of the per-type specific kernels."""
    n = cdg.n
    nl = n // kt
    npt = min(4, n // 2)
    F = d2a2c_global_fields(u_d, v_d, cdg)
    up, vp = F.utmp_pad, F.vtmp_pad
    cos_sg5, rsin2 = F.cos_sg5, F.rsin2
    uxp = jnp.pad(up, [(0, 0), (1, 0), (0, 0)], mode='edge')   # wide low-i
    vxp = jnp.pad(vp, [(0, 0), (0, 0), (1, 0)], mode='edge')   # wide low-j
    ua = np.zeros((6, n, n)); va = np.zeros((6, n, n))
    uc = np.zeros((6, n + 1, n)); vc = np.zeros((6, n, n + 1))
    ut = np.zeros((6, n + 1, n)); vt = np.zeros((6, n, n + 1))
    for f in range(6):
        xf = (F.ua_pad[f], F.dxc_pad_x[f], F.se_pad_x[f], F.sw_pad_x[f])
        yf = (F.va_pad[f], F.dyc_pad_y[f], F.sn_pad_y[f], F.ss_pad_y[f])
        for ti in range(kt):
            for tj in range(kt):
                a, b = ti * nl, tj * nl
                cb = (
                    staggered_tile_block(u_d[f], ti, tj, nl, 1)[None],
                    staggered_tile_block(v_d[f], ti, tj, nl, 0)[None],
                    tiled_face_block(cos_sg5[f], ti, tj, nl, kt)[None],
                    tiled_face_block(rsin2[f], ti, tj, nl, kt)[None],
                    staggered_tile_block(cdg.cosa_u[f], ti, tj, nl, 0)[None],
                    staggered_tile_block(cdg.rsin_u[f], ti, tj, nl, 0)[None],
                    staggered_tile_block(cdg.cosa_v[f], ti, tj, nl, 1)[None],
                    staggered_tile_block(cdg.rsin_v[f], ti, tj, nl, 1)[None],
                )
                lo_i, hi_i = ti == 0, ti == kt - 1
                lo_j, hi_j = tj == 0, tj == kt - 1
                if unified:
                    ub = uxp[f, a:a + nl + 5, b:b + nl + 4][None]
                    vb = vxp[f, a:a + nl + 4, b:b + nl + 5][None]
                    out = d2a2c_tile_unified(
                        ub, vb,
                        xf[0][a:a + nl + 4, b:b + nl + 4][None],
                        xf[1][a:a + nl + 4, b:b + nl][None],
                        xf[2][a:a + nl + 2, b:b + nl + 2][None],
                        xf[3][a:a + nl + 2, b:b + nl + 2][None],
                        yf[0][a:a + nl + 4, b:b + nl + 4][None],
                        yf[1][a:a + nl, b:b + nl + 4][None],
                        yf[2][a:a + nl + 2, b:b + nl + 2][None],
                        yf[3][a:a + nl + 2, b:b + nl + 2][None],
                        *cb, a, b, n, npt, lo_i, hi_i, lo_j, hi_j)
                elif (lo_i or hi_i) and (lo_j or hi_j):
                    out = d2a2c_corner_local(up[f], vp[f], ti, tj, nl, n,
                                             *cb, *xf, *yf)
                elif lo_i or hi_i:
                    fn = d2a2c_edge_w_local if lo_i else d2a2c_edge_e_local
                    out = fn(up[f], vp[f], tj, nl, n, *cb, *xf)
                elif lo_j or hi_j:
                    fn = d2a2c_edge_s_local if lo_j else d2a2c_edge_n_local
                    out = fn(up[f], vp[f], ti, nl, n, *cb, *yf)
                else:
                    out = d2a2c_interior_local(up[f], vp[f], ti, tj, nl, *cb)
                ua_t, va_t, uc_t, vc_t, ut_t, vt_t = (
                    np.asarray(x)[0] for x in out)
                ua[f, a:a + nl, b:b + nl] = ua_t
                va[f, a:a + nl, b:b + nl] = va_t
                uc[f, a:a + nl + 1, b:b + nl] = uc_t
                vc[f, a:a + nl, b:b + nl + 1] = vc_t
                ut[f, a:a + nl + 1, b:b + nl] = ut_t
                vt[f, a:a + nl, b:b + nl + 1] = vt_t
    ut_g, vt_g = d2a2c_adjacent_strips(
        jnp.asarray(uc), jnp.asarray(vc), jnp.asarray(ut), jnp.asarray(vt),
        cdg.cosa_u, cdg.cosa_v, n)
    return ua, va, uc, vc, np.asarray(ut_g), np.asarray(vt_g)


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("kt", [2, 3, 4])
def test_tiled_d2a2c_assemble_matches_production(kt, unified):
    """FULL tiled assembly (every (ti,tj) tile's kernel gathered + adjacent
    strips applied globally) == production d2a2c_vect, bit-for-bit.  The
    end-to-end proof that the tile kernels COMPOSE into the global A->C field
    — the single-device stand-in for the shard_map stage.  ``unified=False``
    routes the per-type specific kernels; ``unified=True`` routes EVERY tile
    through the single flag/mask-driven d2a2c_tile_unified (the device-uniform
    body the shard_map will run).  kt=2 routes only corners (np=24), kt=3/4
    exercise interior + edge + corner tiles."""
    set_halo_backend("local")
    n = 24
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(90 + kt)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    g = [np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg)]
    t = _assemble_tiled_d2a2c(u_d, v_d, cdg, kt, unified=unified)
    for name, tt, gg in zip(["ua", "va", "uc", "vc", "ut", "vt"], t, g):
        np.testing.assert_allclose(tt, gg, rtol=0, atol=1e-12,
                                   err_msg=f"{name} mismatch (kt={kt}, "
                                           f"unified={unified})")


def _tile_blocks(F, uxp, vxp, cdg, u_d, v_d, f, a, b, nl):
    """The (a,b)-anchored tile blocks the shard_map stage dynamic_slices
    (host-slice mirror of a per-tile ``dynamic_slice``)."""
    return (
        uxp[f, a:a + nl + 5, b:b + nl + 4][None],
        vxp[f, a:a + nl + 4, b:b + nl + 5][None],
        F.ua_pad[f, a:a + nl + 4, b:b + nl + 4][None],
        F.dxc_pad_x[f, a:a + nl + 4, b:b + nl][None],
        F.se_pad_x[f, a:a + nl + 2, b:b + nl + 2][None],
        F.sw_pad_x[f, a:a + nl + 2, b:b + nl + 2][None],
        F.va_pad[f, a:a + nl + 4, b:b + nl + 4][None],
        F.dyc_pad_y[f, a:a + nl, b:b + nl + 4][None],
        F.sn_pad_y[f, a:a + nl + 2, b:b + nl + 2][None],
        F.ss_pad_y[f, a:a + nl + 2, b:b + nl + 2][None],
        u_d[f, a:a + nl, b:b + nl + 1][None],
        v_d[f, a:a + nl + 1, b:b + nl][None],
        F.cos_sg5[f, a:a + nl, b:b + nl][None],
        F.rsin2[f, a:a + nl, b:b + nl][None],
        cdg.cosa_u[f, a:a + nl + 1, b:b + nl][None],
        cdg.rsin_u[f, a:a + nl + 1, b:b + nl][None],
        cdg.cosa_v[f, a:a + nl, b:b + nl + 1][None],
        cdg.rsin_v[f, a:a + nl, b:b + nl + 1][None],
    )


@pytest.mark.parametrize("kt", [2, 3])
def test_d2a2c_tile_strips_matches_production(kt):
    """Per-tile strip application (d2a2c_tile_strips fed 1-cell neighbour
    halos — the host-slice mirror of the stage's 4 ppermutes) makes the
    assembled tiled output equal production d2a2c_vect EXACTLY with NO
    global strip pass, and keeps the duplicated shared-staggered-face
    copies of neighbouring tiles bit-identical (the symmetric-exchange
    invariant downstream consumers rely on)."""
    set_halo_backend("local")
    n = 24
    nl = n // kt
    npt = min(4, n // 2)
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(70 + kt)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    g = [np.asarray(x) for x in d2a2c_vect(u_d, v_d, cdg)]
    F = d2a2c_global_fields(u_d, v_d, cdg)
    uxp = jnp.pad(F.utmp_pad, [(0, 0), (1, 0), (0, 0)], mode='edge')
    vxp = jnp.pad(F.vtmp_pad, [(0, 0), (0, 0), (1, 0)], mode='edge')

    # Pass 1: every tile's PRE-strip kernel output.
    pre = {}
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                a, b = ti * nl, tj * nl
                blocks = _tile_blocks(F, uxp, vxp, cdg, u_d, v_d, f, a, b, nl)
                pre[(f, ti, tj)] = d2a2c_tile_unified(
                    *blocks, a, b, n, npt,
                    ti == 0, ti == kt - 1, tj == 0, tj == kt - 1)

    # Pass 2: per-tile strips with halos sliced from the neighbours'
    # PRE-strip ut/vt (cell axes partition cleanly — no dedup question).
    zcol = jnp.zeros((1, nl + 1))
    post = {}
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                a, b = ti * nl, tj * nl
                ua_t, va_t, uc_t, vc_t, ut_t, vt_t = pre[(f, ti, tj)]
                ut_hi = (pre[(f, ti, tj + 1)][4][:, :, 0]
                         if tj + 1 < kt else zcol)
                ut_lo = (pre[(f, ti, tj - 1)][4][:, :, nl - 1]
                         if tj - 1 >= 0 else zcol)
                vt_hi = (pre[(f, ti + 1, tj)][5][:, 0, :]
                         if ti + 1 < kt else zcol)
                vt_lo = (pre[(f, ti - 1, tj)][5][:, nl - 1, :]
                         if ti - 1 >= 0 else zcol)
                ut_s, vt_s = d2a2c_tile_strips(
                    uc_t, vc_t, ut_t, vt_t, ut_lo, ut_hi, vt_lo, vt_hi,
                    cdg.cosa_u[f, a:a + nl + 1, b:b + nl][None],
                    cdg.cosa_v[f, a:a + nl, b:b + nl + 1][None],
                    a, b, n, nl,
                    ti == 0, ti == kt - 1, tj == 0, tj == kt - 1)
                post[(f, ti, tj)] = (ua_t, va_t, uc_t, vc_t, ut_s, vt_s)

    # Duplicated shared-face copies bit-identical (i-staggered: uc/ut;
    # j-staggered: vc/vt).
    for f in range(6):
        for ti in range(kt - 1):
            for tj in range(kt):
                for idx in (2, 4):   # uc, ut
                    lo = np.asarray(post[(f, ti, tj)][idx])[0]
                    hi = np.asarray(post[(f, ti + 1, tj)][idx])[0]
                    np.testing.assert_array_equal(
                        lo[nl, :], hi[0, :],
                        err_msg=f"i-shared face copy mismatch idx={idx} "
                                f"(f={f}, ti={ti}, tj={tj})")
        for ti in range(kt):
            for tj in range(kt - 1):
                for idx in (3, 5):   # vc, vt
                    lo = np.asarray(post[(f, ti, tj)][idx])[0]
                    hi = np.asarray(post[(f, ti, tj + 1)][idx])[0]
                    np.testing.assert_array_equal(
                        lo[:, nl], hi[:, 0],
                        err_msg=f"j-shared face copy mismatch idx={idx} "
                                f"(f={f}, ti={ti}, tj={tj})")

    # Assemble (lower tile owns the shared face) == d2a2c_vect EXACTLY,
    # with NO global strip pass.
    ua = np.zeros((6, n, n)); va = np.zeros((6, n, n))
    uc = np.zeros((6, n + 1, n)); vc = np.zeros((6, n, n + 1))
    ut = np.zeros((6, n + 1, n)); vt = np.zeros((6, n, n + 1))
    for f in range(6):
        for ti in reversed(range(kt)):
            for tj in reversed(range(kt)):
                a, b = ti * nl, tj * nl
                ua_t, va_t, uc_t, vc_t, ut_s, vt_s = (
                    np.asarray(x)[0] for x in post[(f, ti, tj)])
                ua[f, a:a + nl, b:b + nl] = ua_t
                va[f, a:a + nl, b:b + nl] = va_t
                uc[f, a:a + nl + 1, b:b + nl] = uc_t
                vc[f, a:a + nl, b:b + nl + 1] = vc_t
                ut[f, a:a + nl + 1, b:b + nl] = ut_s
                vt[f, a:a + nl, b:b + nl + 1] = vt_s
    for name, tt, gg in zip(["ua", "va", "uc", "vc", "ut", "vt"],
                            (ua, va, uc, vc, ut, vt), g):
        np.testing.assert_array_equal(
            tt, gg, err_msg=f"{name} mismatch (kt={kt}, in-tile strips)")


@pytest.mark.parametrize("kt,ti,tj", [
    (3, 1, 1), (3, 0, 1), (3, 2, 1), (3, 1, 0), (3, 1, 2),  # interior + edges
    (3, 0, 0), (3, 2, 2), (3, 0, 2), (3, 2, 0),             # kt=3 corners
    (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),             # kt=2 (all corners)
])
def test_d2a2c_uc_tile_unified_matches_production(kt, ti, tj):
    """The flag/mask-driven UNIFIED uc kernel (the GSPMD form the shard_map
    stage runs identically on every device) == production d2a2c_vect uc on
    the tile, for every tile position (interior, W/E/S/N edge, corner).
    The wide low-padded u-block feeds the no-shift 4th overlay; the boundary
    flags select C1/C2/C3 + edge_interpolate4 via jnp.where."""
    set_halo_backend("local")
    n = 24
    nl = n // kt
    npt = min(4, n // 2)
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(100 + kt * 16 + ti * 4 + tj)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    uc_g = np.asarray(d2a2c_vect(u_d, v_d, cdg)[2])   # (6, n+1, n)
    F = d2a2c_global_fields(u_d, v_d, cdg)
    # wide block: one extra low-i cell so [a-1:a+nl+4] is in-bounds for a=0.
    utmp_xpad = jnp.pad(F.utmp_pad, [(0, 0), (1, 0), (0, 0)], mode='edge')
    a, b = ti * nl, tj * nl
    is_lo_i, is_hi_i = ti == 0, ti == kt - 1
    for f in range(6):
        u_block = utmp_xpad[f, a:a + nl + 5, b:b + nl + 4][None]
        ua_block = F.ua_pad[f, a:a + nl + 4, b:b + nl + 4][None]
        dx_block = F.dxc_pad_x[f, a:a + nl + 4, b:b + nl][None]
        se_block = F.se_pad_x[f, a:a + nl + 2, b:b + nl + 2][None]
        sw_block = F.sw_pad_x[f, a:a + nl + 2, b:b + nl + 2][None]
        uc_t = np.asarray(d2a2c_uc_tile_unified(
            u_block, ua_block, dx_block, se_block, sw_block,
            a, n, npt, is_lo_i, is_hi_i))[0]
        np.testing.assert_allclose(uc_t, uc_g[f, a:a + nl + 1, b:b + nl],
                                   rtol=0, atol=1e-12)


@pytest.mark.parametrize("kt,ti,tj", [
    (3, 1, 1), (3, 0, 1), (3, 2, 1), (3, 1, 0), (3, 1, 2),  # interior + edges
    (3, 0, 0), (3, 2, 2), (3, 0, 2), (3, 2, 0),             # kt=3 corners
    (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),             # kt=2 (all corners)
])
def test_d2a2c_vc_tile_unified_matches_production(kt, ti, tj):
    """The flag/mask-driven UNIFIED vc kernel (j-axis transpose of the uc
    unified kernel) == production d2a2c_vect vc on the tile, for every tile
    position.  Wide low-j-padded v-block + flag-selected S/N specials."""
    set_halo_backend("local")
    n = 24
    nl = n // kt
    npt = min(4, n // 2)
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    rng = np.random.default_rng(200 + kt * 16 + ti * 4 + tj)
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc_g = np.asarray(d2a2c_vect(u_d, v_d, cdg)[3])   # (6, n, n+1)
    F = d2a2c_global_fields(u_d, v_d, cdg)
    vtmp_xpad = jnp.pad(F.vtmp_pad, [(0, 0), (0, 0), (1, 0)], mode='edge')
    a, b = ti * nl, tj * nl
    is_lo_j, is_hi_j = tj == 0, tj == kt - 1
    for f in range(6):
        v_block = vtmp_xpad[f, a:a + nl + 4, b:b + nl + 5][None]
        va_block = F.va_pad[f, a:a + nl + 4, b:b + nl + 4][None]
        dy_block = F.dyc_pad_y[f, a:a + nl, b:b + nl + 4][None]
        sn_block = F.sn_pad_y[f, a:a + nl + 2, b:b + nl + 2][None]
        ss_block = F.ss_pad_y[f, a:a + nl + 2, b:b + nl + 2][None]
        vc_t = np.asarray(d2a2c_vc_tile_unified(
            v_block, va_block, dy_block, sn_block, ss_block,
            b, n, npt, is_lo_j, is_hi_j))[0]
        np.testing.assert_allclose(vc_t, vc_g[f, a:a + nl, b:b + nl + 1],
                                   rtol=0, atol=1e-12)

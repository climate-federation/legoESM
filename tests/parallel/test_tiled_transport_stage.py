"""U3: the PPM i-sweep transport tiled under a REAL shard_map == the global
sweep (task #3 cube tiled np>6 stage).

Proves the approach-C tiling (U2/U2b: global pre-pad → per-tile slice →
ppm_transport_1d(external_halo=4, rd_prepadded=True)) holds INSIDE a
``(6, kt)`` ``(face, tile_i)`` shard_map (6*kt host devices) — the step
where the np>6 perf actually materialises.  i-sweep only, synthetic
field/courant (the cross-face/staggered FIELD faithfulness is U2b; this
pins the SHARD_MAP WIRING).  Run with
``--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv3_sw_core import ppm_transport_1d
from legoesm.parallel.tiled_transport import (
    make_tiled_transport_sweep_stage,
    make_tiled_transport_sweep_stage_2d,
    transport_sweep_tile,
    transport_sweep_tile_2d,
    transport_jsweep_tile_2d,
)


def test_transport_sweep_tile_body_host():
    """Non-shard CI coverage (codex U3 MEDIUM): the per-tile body
    ``transport_sweep_tile`` (the stage's exact slice + _ppm) reassembled
    over tiles == the global PPM sweep, WITHOUT a multi-device mesh — so the
    stage's slice logic is exercised even when the shard_map test below
    skips (below 6*kt host devices)."""
    kt, h3, n, m = 3, 4, 18, 5
    nl = n // kt
    rng = np.random.default_rng(11)
    field = jnp.asarray(rng.standard_normal((6, n, m)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)
    global_flux = np.asarray(
        ppm_transport_1d(field, courant, rd, 1, external_halo=0))
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")
    # a = t*nl is a Python int here (host); axis_index int32 in the stage.
    tiles = [np.asarray(transport_sweep_tile(vp_g, courant, rd_g, t * nl,
                                             nl, h3)) for t in range(kt)]
    for t in range(kt - 1):
        np.testing.assert_allclose(
            tiles[t][:, nl, :], tiles[t + 1][:, 0, :], atol=1e-12, rtol=1e-12,
            err_msg=f"host body shared interface mismatch t={t}")
    reassembled = np.concatenate(
        [tiles[t][:, :nl, :] for t in range(kt)]
        + [tiles[-1][:, nl:nl + 1, :]], axis=1)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg="transport_sweep_tile body reassembled != global PPM sweep")


@pytest.mark.parametrize("kt", [2, 4])
def test_tiled_transport_sweep_in_shardmap(kt):
    ndev = 6 * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    h3 = 4
    n = 24
    nl = n // kt
    m = 5
    rng = np.random.default_rng(7)
    field = jnp.asarray(rng.standard_normal((6, n, m)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)

    global_flux = np.asarray(
        ppm_transport_1d(field, courant, rd, 1, external_halo=0))  # (6,n+1,m)

    # Global pre-pad EXACTLY as ppm_transport_1d(external_halo=0) does —
    # these are the FACE-REPLICATED stage inputs.
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i"))
    stage = make_tiled_transport_sweep_stage(mesh, n, kt, h3=h3)
    flux_tiled = np.asarray(stage(vp_g, courant, rd_g))  # (6, kt*(nl+1), m)

    fb = flux_tiled.reshape(6, kt, nl + 1, m)
    # Shared-interface consistency (tile t's [nl] == tile t+1's [0]).
    for t in range(kt - 1):
        np.testing.assert_allclose(
            fb[:, t, nl, :], fb[:, t + 1, 0, :], atol=1e-12, rtol=1e-12,
            err_msg=f"shard_map shared interface mismatch (kt={kt} t={t})")
    # Reassemble (lower tile owns the shared interface) → global sweep.
    reassembled = np.concatenate(
        [fb[:, t, :nl, :] for t in range(kt)]
        + [fb[:, -1, nl:nl + 1, :]], axis=1)              # (6, n+1, m)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg=f"tiled-in-shardmap PPM sweep != global (kt={kt})")


# =====================================================================
# U3b: full 2-D (6, kt, kt) tiling — cross axis tiled + the j-sweep.
# =====================================================================

def test_transport_sweep_tile_2d_body_host():
    """Host CI coverage: the 2-D-tiled i-sweep body reassembled over
    (tile_i, tile_j) == the global i-sweep (cross axis sliced WITHOUT a halo
    — the i-sweep is independent per j-row)."""
    kt, h3, n = 3, 4, 18
    nl = n // kt
    rng = np.random.default_rng(13)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, n))) + 0.1)
    global_flux = np.asarray(
        ppm_transport_1d(field, courant, rd, 1, external_halo=0))  # (6,n+1,n)
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")
    cols = []
    for tj in range(kt):
        ti_tiles = [np.asarray(transport_sweep_tile_2d(
            vp_g, courant, rd_g, ti * nl, tj * nl, nl, h3)) for ti in range(kt)]
        col = np.concatenate(
            [ti_tiles[ti][:, :nl, :] for ti in range(kt)]
            + [ti_tiles[-1][:, nl:nl + 1, :]], axis=1)        # (6, n+1, nl)
        cols.append(col)
    reassembled = np.concatenate(cols, axis=2)                  # (6, n+1, n)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg="2-D i-sweep body reassembled != global PPM sweep")


def test_transport_jsweep_tile_2d_body_host():
    """Host CI coverage: the 2-D-tiled j-sweep (ytp_v, axis=2) body
    reassembled == the global j-sweep."""
    kt, h3, n = 3, 4, 18
    nl = n // kt
    rng = np.random.default_rng(17)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    courant = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, n))) + 0.1)
    global_flux = np.asarray(
        ppm_transport_1d(field, courant, rd, 2, external_halo=0))  # (6,n,n+1)
    vp_g = jnp.pad(field, [(0, 0), (0, 0), (h3, h3)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (0, 0), (1, 1)], mode="edge")
    rows = []
    for ti in range(kt):
        tj_tiles = [np.asarray(transport_jsweep_tile_2d(
            vp_g, courant, rd_g, ti * nl, tj * nl, nl, h3)) for tj in range(kt)]
        row = np.concatenate(
            [tj_tiles[tj][:, :, :nl] for tj in range(kt)]
            + [tj_tiles[-1][:, :, nl:nl + 1]], axis=2)         # (6, nl, n+1)
        rows.append(row)
    reassembled = np.concatenate(rows, axis=1)                  # (6, n, n+1)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg="2-D j-sweep body reassembled != global PPM sweep")


@pytest.mark.parametrize("sweep", ["i", "j"])
def test_tiled_transport_sweep_2d_in_shardmap(sweep):
    kt = 2
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    h3, n = 4, 24
    nl = n // kt
    rng = np.random.default_rng(23)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, n))) + 0.1)
    if sweep == "i":
        courant = jnp.asarray(rng.standard_normal((6, n + 1, n)))
        global_flux = np.asarray(
            ppm_transport_1d(field, courant, rd, 1, external_halo=0))
        vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
        rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")
    else:
        courant = jnp.asarray(rng.standard_normal((6, n, n + 1)))
        global_flux = np.asarray(
            ppm_transport_1d(field, courant, rd, 2, external_halo=0))
        vp_g = jnp.pad(field, [(0, 0), (0, 0), (h3, h3)], mode="edge")
        rd_g = jnp.pad(rd, [(0, 0), (0, 0), (1, 1)], mode="edge")

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_transport_sweep_stage_2d(mesh, n, kt, h3=h3, sweep=sweep)
    flux = np.asarray(stage(vp_g, courant, rd_g))

    if sweep == "i":
        fb = flux.reshape(6, kt, nl + 1, kt, nl)   # (f, ti, i, tj, j)
        cols = []
        for tj in range(kt):
            col = np.concatenate(
                [fb[:, ti, :nl, tj, :] for ti in range(kt)]
                + [fb[:, kt - 1, nl:nl + 1, tj, :]], axis=1)   # (6, n+1, nl)
            cols.append(col)
        reassembled = np.concatenate(cols, axis=2)              # (6, n+1, n)
    else:
        fb = flux.reshape(6, kt, nl, kt, nl + 1)   # (f, ti, i, tj, j)
        rows = []
        for ti in range(kt):
            row = np.concatenate(
                [fb[:, ti, :, tj, :nl] for tj in range(kt)]
                + [fb[:, ti, :, kt - 1, nl:nl + 1]], axis=2)    # (6, nl, n+1)
            rows.append(row)
        reassembled = np.concatenate(rows, axis=1)              # (6, n, n+1)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg=f"2-D tiled-in-shardmap {sweep}-sweep != global")


# =====================================================================
# U3d: REAL-Courant composition — the corner-cross sweep tile on the real
# _bgrid_ke_transport ytp_v inputs (real cdgrid vb + v_d_jhalo).  Shapes
# from job 8480355: ytp_v cross axis = n+1 (corners) ⇒ cross_nl = nl+1.
# =====================================================================

def test_u3d_real_ytp_v_corner_cross_tiling():
    """The corner-cross j-sweep tile (cross_nl=nl+1) reassembles bit-exactly
    to the global ytp_v of _bgrid_ke_transport, driven by the REAL corner
    Courant vb (U3c on real cross-face-halo'd uc/vc) and the REAL
    cross-face-halo'd D-wind v_d_jhalo.  Reassembly on BOTH axes: each
    non-last tile contributes its first nl rows/interfaces and the final
    shared boundary comes from the last tile (the boundary is duplicated
    across adjacent tiles but bit-identical — PPM is independent per cross
    row — so taking all-but-last from each tile is exact)."""
    import numpy as np
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.core.fv3_sw_core import (
        _pad_halo_uc_vc_new_via_neighbor_delta, _pad_halo_dgrid_for_ppm,
        bgrid_corner_courant_local,
    )
    from tests.test_cases.cosine_bell import cosine_bell_cubesphere

    kt, nl, h3, h_dg = 3, 6, 4, 2
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    state = cosine_bell_cubesphere(grid, cd)
    u_d, v_d = state.u_d, state.v_d

    # REAL corner Courant vb (from U3c on real cross-face-halo'd uc/vc).
    rng = np.random.default_rng(31)
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    uc_pad, vc_pad = _pad_halo_uc_vc_new_via_neighbor_delta(uc, vc, u_d, v_d, cd)
    dt5 = 0.5 * 1800.0
    vb, _ub = bgrid_corner_courant_local(
        uc_pad, vc_pad, cd.cosa_corner, cd.rsin2_corner, dt5)   # (6,n+1,n+1)

    # REAL cross-face-halo'd D-wind + rdy, EXACTLY as _bgrid_ke_transport.
    _u_ih, v_d_jhalo = _pad_halo_dgrid_for_ppm(u_d, v_d, cd, halo=h_dg)  # (6,n+1,n+2h)
    rdy = 1.0 / jnp.maximum(cd.dy_edge_x, 1.0e-30)              # (6,n+1,n)

    global_y = np.asarray(ppm_transport_1d(
        v_d_jhalo, vb, rdy, 2, external_halo=h_dg))            # (6,n+1,n+1)

    # Non-vacuity: cross-face halo differs from edge-replication; vb varies.
    assert float(jnp.max(jnp.abs(
        v_d_jhalo[:, :, :h_dg] - v_d_jhalo[:, :, h_dg:h_dg + 1]))) > 1e-9
    assert float(jnp.std(np.asarray(vb))) > 1e-9

    # Approach-C global pre-pad to h3 on the SWEEP axis=2 (gap=h3-h_dg),
    # depth-1 rd pad — then the corner-cross tile (cross_nl=nl+1).
    vp_g = jnp.pad(v_d_jhalo, [(0, 0), (0, 0), (h3 - h_dg, h3 - h_dg)],
                   mode="edge")                                # (6,n+1,n+2h3)
    rd_g = jnp.pad(rdy, [(0, 0), (0, 0), (1, 1)], mode="edge")  # (6,n+1,n+2)

    rows = []
    for ti in range(kt):                       # cross (axis=1, corner)
        cols = []
        for tj in range(kt):                   # sweep (axis=2, j)
            blk = np.asarray(transport_jsweep_tile_2d(
                vp_g, vb, rd_g, ti * nl, tj * nl, nl, h3=h3, cross_nl=nl + 1))
            r_hi = nl + 1 if ti == kt - 1 else nl   # cross corner ownership
            c_hi = nl + 1 if tj == kt - 1 else nl   # sweep interface ownership
            cols.append(blk[:, :r_hi, :c_hi])
        rows.append(np.concatenate(cols, axis=2))
    reassembled = np.concatenate(rows, axis=1)                 # (6,n+1,n+1)

    np.testing.assert_allclose(
        reassembled, global_y, atol=1e-12, rtol=1e-12,
        err_msg="U3d corner-cross ytp_v tiling != global _bgrid_ke ytp_v")


def test_u3d_real_xtp_u_corner_cross_tiling():
    """Symmetric counterpart of the ytp_v U3d test: the corner-cross i-sweep
    tile (cross_nl=nl+1, cross axis=2 = n+1 corners) reassembles bit-exactly
    to the global xtp_u of _bgrid_ke_transport, driven by the REAL corner
    Courant ub + the REAL cross-face-halo'd u_d_ihalo (shapes job 8480355:
    u_d_ihalo (6,n+2h,n+1), ub (6,n+1,n+1), rdx (6,n,n+1) — all cross
    axis=2 = n+1)."""
    import numpy as np
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.core.fv3_sw_core import (
        _pad_halo_uc_vc_new_via_neighbor_delta, _pad_halo_dgrid_for_ppm,
        bgrid_corner_courant_local,
    )
    from tests.test_cases.cosine_bell import cosine_bell_cubesphere

    kt, nl, h3, h_dg = 3, 6, 4, 2
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    state = cosine_bell_cubesphere(grid, cd)
    u_d, v_d = state.u_d, state.v_d

    rng = np.random.default_rng(41)
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    uc_pad, vc_pad = _pad_halo_uc_vc_new_via_neighbor_delta(uc, vc, u_d, v_d, cd)
    dt5 = 0.5 * 1800.0
    _vb, ub = bgrid_corner_courant_local(
        uc_pad, vc_pad, cd.cosa_corner, cd.rsin2_corner, dt5)   # (6,n+1,n+1)

    u_d_ihalo, _v = _pad_halo_dgrid_for_ppm(u_d, v_d, cd, halo=h_dg)  # (6,n+2h,n+1)
    rdx = 1.0 / jnp.maximum(cd.dx_edge_y, 1.0e-30)             # (6,n,n+1)

    global_x = np.asarray(ppm_transport_1d(
        u_d_ihalo, ub, rdx, 1, external_halo=h_dg))           # (6,n+1,n+1)

    assert float(jnp.max(jnp.abs(
        u_d_ihalo[:, :h_dg, :] - u_d_ihalo[:, h_dg:h_dg + 1, :]))) > 1e-9
    assert float(jnp.std(np.asarray(ub))) > 1e-9

    # Approach-C: pre-pad to h3 on the SWEEP axis=1; depth-1 rd pad on axis=1.
    vp_g = jnp.pad(u_d_ihalo, [(0, 0), (h3 - h_dg, h3 - h_dg), (0, 0)],
                   mode="edge")                                # (6,n+2h3,n+1)
    rd_g = jnp.pad(rdx, [(0, 0), (1, 1), (0, 0)], mode="edge")  # (6,n+2,n+1)

    cols = []
    for tj in range(kt):                       # cross (axis=2, corner)
        rows = []
        for ti in range(kt):                   # sweep (axis=1, i)
            blk = np.asarray(transport_sweep_tile_2d(
                vp_g, ub, rd_g, ti * nl, tj * nl, nl, h3=h3, cross_nl=nl + 1))
            r_hi = nl + 1 if ti == kt - 1 else nl   # sweep interface ownership
            c_hi = nl + 1 if tj == kt - 1 else nl   # cross corner ownership
            rows.append(blk[:, :r_hi, :c_hi])
        cols.append(np.concatenate(rows, axis=1))  # along sweep (axis=1)
    reassembled = np.concatenate(cols, axis=2)     # along cross (axis=2) -> (6,n+1,n+1)

    np.testing.assert_allclose(
        reassembled, global_x, atol=1e-12, rtol=1e-12,
        err_msg="U3d corner-cross xtp_u tiling != global _bgrid_ke xtp_u")


@pytest.mark.parametrize("sweep", ["i", "j"])
def test_u3d_real_courant_in_shardmap(sweep):
    """U3d under a REAL (6,kt,kt) shard_map at np24: the corner-cross stage
    (cross_nl=nl+1) on the REAL _bgrid_ke_transport inputs reassembles
    bit-exactly to the global xtp_u/ytp_v.  Proves the real-Courant tiling
    runs in the SPMD stage (not just host-body)."""
    kt = 2
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    import numpy as np
    from jax.sharding import Mesh
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.core.fv3_sw_core import (
        _pad_halo_uc_vc_new_via_neighbor_delta, _pad_halo_dgrid_for_ppm,
        bgrid_corner_courant_local,
    )
    from tests.test_cases.cosine_bell import cosine_bell_cubesphere

    nl, h3, h_dg = 12, 4, 2
    n = kt * nl                                  # 24
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    state = cosine_bell_cubesphere(grid, cd)
    u_d, v_d = state.u_d, state.v_d
    rng = np.random.default_rng(53)
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    uc_pad, vc_pad = _pad_halo_uc_vc_new_via_neighbor_delta(uc, vc, u_d, v_d, cd)
    vb, ub = bgrid_corner_courant_local(
        uc_pad, vc_pad, cd.cosa_corner, cd.rsin2_corner, 0.5 * 1800.0)
    u_d_ihalo, v_d_jhalo = _pad_halo_dgrid_for_ppm(u_d, v_d, cd, halo=h_dg)

    if sweep == "j":
        courant = vb
        rdy = 1.0 / jnp.maximum(cd.dy_edge_x, 1.0e-30)
        global_f = np.asarray(ppm_transport_1d(v_d_jhalo, vb, rdy, 2, external_halo=h_dg))
        vp_g = jnp.pad(v_d_jhalo, [(0, 0), (0, 0), (h3 - h_dg, h3 - h_dg)], mode="edge")
        rd_g = jnp.pad(rdy, [(0, 0), (0, 0), (1, 1)], mode="edge")
    else:
        courant = ub
        rdx = 1.0 / jnp.maximum(cd.dx_edge_y, 1.0e-30)
        global_f = np.asarray(ppm_transport_1d(u_d_ihalo, ub, rdx, 1, external_halo=h_dg))
        vp_g = jnp.pad(u_d_ihalo, [(0, 0), (h3 - h_dg, h3 - h_dg), (0, 0)], mode="edge")
        rd_g = jnp.pad(rdx, [(0, 0), (1, 1), (0, 0)], mode="edge")

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_transport_sweep_stage_2d(
        mesh, n, kt, h3=h3, sweep=sweep, cross_nl=nl + 1)
    flux = np.asarray(stage(vp_g, courant, rd_g))   # (6, kt*(nl+1), kt*(nl+1))

    fb = flux.reshape(6, kt, nl + 1, kt, nl + 1)    # (f, ti, a, tj, b)
    # axis-1 tile = tile_i, axis-2 tile = tile_j (both nl+1 per tile, corner).
    rows = []
    for ti in range(kt):
        cols = []
        for tj in range(kt):
            a_hi = nl + 1 if ti == kt - 1 else nl
            b_hi = nl + 1 if tj == kt - 1 else nl
            cols.append(fb[:, ti, :a_hi, tj, :b_hi])
        rows.append(np.concatenate(cols, axis=2))
    reassembled = np.concatenate(rows, axis=1)      # (6, n+1, n+1)
    np.testing.assert_allclose(
        reassembled, global_f, atol=1e-12, rtol=1e-12,
        err_msg=f"U3d real-Courant {sweep}-sweep in shard_map != global")


def test_stage_factory_rejects_d_sw3_boundary_fix():
    """codex 2026-07-11 F3: `make_tiled_transport_sweep_stage_2d` must
    REFUSE the d_sw3 boundary-fix args (per-tile edge masks not yet
    ported to the shard-map stage) — a silent no-forward would reproduce
    the pre-fix numerics under shard_map.  Host-side tiled helpers
    (`transport_sweep_tile_2d`/`transport_jsweep_tile_2d`) remain the
    supported route (they take per-tile `boundary_fix_edges`)."""
    n, kt = 8, 2
    dummy_dx = jnp.ones((6, n, n))
    with pytest.raises(NotImplementedError, match="edge masks"):
        make_tiled_transport_sweep_stage_2d(
            None, n, kt, apply_d_sw3_boundary_fix=True)
    with pytest.raises(NotImplementedError, match="edge masks"):
        make_tiled_transport_sweep_stage_2d(
            None, n, kt, boundary_fix_dx_field=dummy_dx)
    with pytest.raises(NotImplementedError, match="edge masks"):
        make_tiled_transport_sweep_stage_2d(
            None, n, kt, boundary_fix_edges=(True, False, False, False))

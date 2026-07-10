"""U4a: the d_sw1 ut/vt transport-velocity RECOMPUTE (the start of the FV3
D-grid stage) tiled via approach-C (task #3 cube np>6, next op after the
d_sw3 KE-transport capstone U3f).

ut/vt are re-derived from the UPDATED C-grid winds by a 4-cell box average of
the OTHER component (``_d_sw1_recompute_ut_vt`` duogrid Part 1) — DISTINCT
from the d2a2c ut/vt (which uses the D-grid wind directly).  The box stencil
is fully LOCAL once uc/vc carry their 1-cell cross halo (the GLOBAL
face-replicated ``_pad_halo_uc_vc_new_via_neighbor_delta`` pre-pad, proven in
U3f), so each tile only ``dynamic_slice``s — NO in-stage exchange.

Two checks (tight tol ``rtol=0, atol=1e-12`` — the tiled and global scalar
expressions match operand order, but JAX/XLA gives NO portable bit-exact
guarantee across differently shaped compilations: FMA contraction / codegen
can differ by ~1 ULP between the full-array and the per-tile op, so exact
equality is brittle — codex U4a LOW):
  * host-body parity (kt=3 → interior + edge + corner tiles), and
  * the real ``(6, kt, kt)`` shard_map stage at kt=2 (np24), SKIPPED below 24
    host devices (so CI on a small box still runs the host-body gate).
Both reassemble lower-tile-owns-the-shared-staggered-face to the global ut/vt,
and we ALSO assert the discarded duplicate shared faces match (which is what
makes the lower-owns trim valid).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv3_sw_core import (
    _d_sw1_recompute_ut_vt, _pad_halo_uc_vc_new_via_neighbor_delta,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.tiled_transport import (
    dsw1_ut_vt_tile_2d, make_tiled_dsw1_ut_vt_stage_2d,
)
from tests.test_cases.cosine_bell import cosine_bell_cubesphere


def _setup(kt, nl, seed):
    """Cube + updated C-grid winds + their global pre-pad + the duogrid-Part-1
    global ut/vt oracle.  ``u_d``/``v_d`` (the D-grid winds) serve as the OLD
    fields the NEW-corrected halo (ng>=3) carries the cross-face delta from."""
    n = kt * nl
    dt = 1800.0
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    assert grid.duogrid is not None and grid.duogrid.ng >= 3, (
        "test needs ng>=3 so _d_sw1_recompute_ut_vt takes the "
        "_pad_halo_uc_vc_new_via_neighbor_delta path")
    cd = create_cubed_sphere_cdgrid(grid)
    state = cosine_bell_cubesphere(grid, cd)
    u_d_old, v_d_old = state.u_d, state.v_d
    rng = np.random.default_rng(seed)
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))

    # Global oracle (duogrid Part 1 — returns at the `if use_duogrid` line).
    ut_g, vt_g = _d_sw1_recompute_ut_vt(uc, vc, cd, dt, u_d_old, v_d_old)
    # The SAME pre-pad the oracle used internally (ng>=3 + old fields present).
    uc_pad, vc_pad = _pad_halo_uc_vc_new_via_neighbor_delta(
        uc, vc, u_d_old, v_d_old, cd)
    return cd, uc, vc, uc_pad, vc_pad, np.asarray(ut_g), np.asarray(vt_g)


def _reassemble_and_check(get_tile, kt, nl, ut_g, vt_g, tag):
    """``get_tile(ti, tj) -> (ut_blk (6,nl+1,nl), vt_blk (6,nl,nl+1))`` untrimmed.
    Asserts (1) the duplicated shared staggered faces are BIT-identical between
    adjacent tiles (validity of lower-owns), then (2) the lower-owns reassembly
    equals the global ut/vt BIT-for-bit."""
    ut_blk, vt_blk = {}, {}
    for ti in range(kt):
        for tj in range(kt):
            u, v = get_tile(ti, tj)
            ut_blk[ti, tj] = np.asarray(u)
            vt_blk[ti, tj] = np.asarray(v)

    # (1) duplicate shared face bit-identical.  ut i-axis is staggered: tile
    # (ti,tj) i-face [nl] is the SAME global face as tile (ti+1,tj) i-face [0].
    for ti in range(kt - 1):
        for tj in range(kt):
            np.testing.assert_allclose(
                ut_blk[ti, tj][:, nl, :], ut_blk[ti + 1, tj][:, 0, :],
                rtol=0, atol=1e-12,
                err_msg=f"{tag}: ut shared i-face (ti={ti},tj={tj}) != neighbour")
    # vt j-axis is staggered: tile (ti,tj) j-face [nl] == tile (ti,tj+1) [0].
    for ti in range(kt):
        for tj in range(kt - 1):
            np.testing.assert_allclose(
                vt_blk[ti, tj][:, :, nl], vt_blk[ti, tj + 1][:, :, 0],
                rtol=0, atol=1e-12,
                err_msg=f"{tag}: vt shared j-face (ti={ti},tj={tj}) != neighbour")

    # (2) lower-owns reassembly.  ut: i-axis trim last tile to nl+1, j cell
    # full; vt: j-axis trim last tile, i cell full.
    ut_rows, vt_rows = [], []
    for ti in range(kt):
        ut_cols, vt_cols = [], []
        for tj in range(kt):
            a_hi = nl + 1 if ti == kt - 1 else nl
            b_hi = nl + 1 if tj == kt - 1 else nl
            ut_cols.append(ut_blk[ti, tj][:, :a_hi, :])
            vt_cols.append(vt_blk[ti, tj][:, :, :b_hi])
        ut_rows.append(np.concatenate(ut_cols, axis=2))
        vt_rows.append(np.concatenate(vt_cols, axis=2))
    ut_tiled = np.concatenate(ut_rows, axis=1)
    vt_tiled = np.concatenate(vt_rows, axis=1)

    np.testing.assert_allclose(
        ut_tiled, ut_g, rtol=0, atol=1e-12,
        err_msg=f"{tag}: tiled d_sw1 ut != global")
    np.testing.assert_allclose(
        vt_tiled, vt_g, rtol=0, atol=1e-12,
        err_msg=f"{tag}: tiled d_sw1 vt != global")


def test_dsw1_ut_vt_host_body_tiling():
    kt, nl = 3, 6
    cd, uc, vc, uc_pad, vc_pad, ut_g, vt_g = _setup(kt, nl, seed=71)

    def get_tile(ti, tj):
        return dsw1_ut_vt_tile_2d(
            uc, vc, uc_pad, vc_pad,
            cd.cosa_u, cd.rsin_u, cd.cosa_v, cd.rsin_v,
            ti * nl, tj * nl, nl)

    _reassemble_and_check(get_tile, kt, nl, ut_g, vt_g, "U4a host-body")


def test_dsw1_ut_vt_shard_map_stage_np24():
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, uc, vc, uc_pad, vc_pad, ut_g, vt_g = _setup(kt, nl, seed=72)
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_dsw1_ut_vt_stage_2d(mesh, cd, kt * nl, kt)
    ut_s, vt_s = stage(uc, vc, uc_pad, vc_pad)
    # gathered (6, kt*(nl+1), kt*nl) / (6, kt*nl, kt*(nl+1)) → block-index view.
    ut_fb = np.asarray(ut_s).reshape(6, kt, nl + 1, kt, nl)
    vt_fb = np.asarray(vt_s).reshape(6, kt, nl, kt, nl + 1)

    def get_tile(ti, tj):
        return ut_fb[:, ti, :, tj, :], vt_fb[:, ti, :, tj, :]

    _reassemble_and_check(get_tile, kt, nl, ut_g, vt_g, "U4a shard_map")

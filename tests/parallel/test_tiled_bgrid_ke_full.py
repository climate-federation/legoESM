"""U3f: the FULL d_sw3 B-grid KE transport (_bgrid_ke_transport) tiled via
approach-C (task #3 cube np>6 capstone).

Composition (all pieces proven separately): GLOBAL cross-face halos + corner
Courant (U3c bgrid_corner_courant_local) + GLOBAL corner scalar sync
(synchronize_bgrid_ne_corner_geo — cheap O(n), runs in the global view, U3f
simplification) + per-tile PPM sweeps (U3d transport_*_tile_2d, cross_nl=nl+1,
UNSYNCED Courant) + pointwise KE (Step 6).  Reassembles bit-exactly to the
global ke_corner.  Host-body (the shard_map wiring is the follow-up).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import (
    _bgrid_ke_transport, _pad_halo_uc_vc_new_via_old_delta,
    _pad_halo_dgrid_for_ppm, bgrid_corner_courant_local, _EPS,
)
from legoesm.grids.halo import synchronize_bgrid_ne_corner_geo
from legoesm.parallel.tiled_transport import (
    transport_sweep_tile_2d, transport_jsweep_tile_2d,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.test_cases.cosine_bell import cosine_bell_cubesphere


def test_bgrid_ke_transport_full_tiling():
    kt, nl, h3, h_dg = 3, 6, 4, 2
    n = kt * nl
    dt = 1800.0
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    state = cosine_bell_cubesphere(grid, cd)
    u_d, v_d = state.u_d, state.v_d
    rng = np.random.default_rng(61)
    uc = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    vc = jnp.asarray(rng.standard_normal((6, n, n + 1)))

    ke_global = np.asarray(_bgrid_ke_transport(u_d, v_d, uc, vc, cd, dt))  # (6,n+1,n+1)

    # --- GLOBAL pre-steps (approach-C; all cheap / face-replicable) ---
    uc_pad, vc_pad = _pad_halo_uc_vc_new_via_old_delta(uc, vc, u_d, v_d, cd)
    vb, ub = bgrid_corner_courant_local(uc_pad, vc_pad, cd.cosa_corner,
                                        cd.rsin2_corner, 0.5 * dt)
    # corner SCALAR sync (global; the U3f insight — not a tiled exchange).
    # _bgrid_ke_transport syncs (ubb=ub, vbbtemp=vb); mirror exactly.
    ubb, vbbtemp = synchronize_bgrid_ne_corner_geo(
        ub, vb, cd.cos_angle_corner, cd.sin_angle_corner,
        cd.z21_corner, cd.z22_corner, n)
    u_d_ihalo, v_d_jhalo = _pad_halo_dgrid_for_ppm(u_d, v_d, cd, halo=h_dg)
    rdx = 1.0 / jnp.maximum(cd.dx_edge_y, _EPS)
    rdy = 1.0 / jnp.maximum(cd.dy_edge_x, _EPS)
    # sweep-axis pre-pad to h3 (gap) + depth-1 rd pad
    vpy = jnp.pad(v_d_jhalo, [(0, 0), (0, 0), (h3 - h_dg, h3 - h_dg)], mode="edge")
    rdy_g = jnp.pad(rdy, [(0, 0), (0, 0), (1, 1)], mode="edge")
    vpx = jnp.pad(u_d_ihalo, [(0, 0), (h3 - h_dg, h3 - h_dg), (0, 0)], mode="edge")
    rdx_g = jnp.pad(rdx, [(0, 0), (1, 1), (0, 0)], mode="edge")

    ubb_n, vbb_n = np.asarray(ubb), np.asarray(vbbtemp)
    rows = []
    for ti in range(kt):
        cols = []
        for tj in range(kt):
            ai, aj = ti * nl, tj * nl
            # tiled sweeps (UNSYNCED Courant vb/ub), corner cross_nl=nl+1
            ty = np.asarray(transport_jsweep_tile_2d(
                vpy, vb, rdy_g, ai, aj, nl, h3=h3, cross_nl=nl + 1))   # (6,nl+1[i],nl+1[j])
            tx = np.asarray(transport_sweep_tile_2d(
                vpx, ub, rdx_g, ai, aj, nl, h3=h3, cross_nl=nl + 1))   # (6,nl+1[i],nl+1[j])
            sl = (slice(None), slice(ai, ai + nl + 1), slice(aj, aj + nl + 1))
            # Step 6 KE (pointwise): 0.5*(ubbtemp*vbbtemp + ubb*vbb),
            # ubbtemp=transported_y, vbb=transported_x.
            ke_tile = 0.5 * (ty * vbb_n[sl] + ubb_n[sl] * tx)
            a_hi = nl + 1 if ti == kt - 1 else nl
            b_hi = nl + 1 if tj == kt - 1 else nl
            cols.append(ke_tile[:, :a_hi, :b_hi])
        rows.append(np.concatenate(cols, axis=2))
    ke_tiled = np.concatenate(rows, axis=1)                            # (6,n+1,n+1)

    np.testing.assert_allclose(
        ke_tiled, ke_global, atol=1e-10, rtol=1e-10,
        err_msg="U3f full tiled _bgrid_ke_transport != global ke_corner")

"""U3c: tile the B-grid corner Courant (vb, ub) of _bgrid_ke_transport
(task #3 cube np>6 stage).

``bgrid_corner_courant_local`` is the pointwise Step 1/3 of
``_bgrid_ke_transport`` factored for approach-C tiling: the cross-face halo
of uc/vc runs in the GLOBAL view (a later piece), and each sub-face tile
computes its corner-block (vb, ub) from its slice of the globally-padded
uc_pad/vc_pad + corner metrics.  Being pure pointwise, the tiled slices
reassemble bit-exactly to the global corner Courant — proven here on the
REAL cube corner metrics (cosa_corner / rsin2_corner).

The cross-face halo tiling + the BGRID_NE corner sync are later increments;
this pins the corner-Courant pointwise core (mirrors d2a2c_ua_va_local).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import bgrid_corner_courant_local


def _setup(kt=3, nl=6):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return n, cdgrid


def test_corner_courant_tiling_real_metrics():
    kt, nl = 3, 6
    n, cdgrid = _setup(kt, nl)
    cosa = cdgrid.cosa_corner       # (6, n+1, n+1)
    rsina = cdgrid.rsin2_corner     # (6, n+1, n+1)
    rng = np.random.default_rng(5)
    # Globally cross-face-halo'd c-grid winds (random; the halo itself is a
    # later piece — this pins the pointwise corner-Courant tiling).
    uc_pad = jnp.asarray(rng.standard_normal((6, n + 1, n + 2)))   # j-padded uc
    vc_pad = jnp.asarray(rng.standard_normal((6, n + 2, n + 1)))   # i-padded vc
    dt5 = 0.5 * 1800.0

    vb_g, ub_g = bgrid_corner_courant_local(uc_pad, vc_pad, cosa, rsina, dt5)
    assert vb_g.shape == (6, n + 1, n + 1)
    # Non-vacuity: vb != ub (different cross-term sign) and metrics vary.
    assert float(jnp.max(jnp.abs(vb_g - ub_g))) > 1e-9
    assert float(jnp.std(np.asarray(cosa))) > 1e-9

    # Independent inline oracle (codex 2026-06-13 LOW): the tiled-vs-global
    # check alone is helper-vs-helper and would miss a vb/ub swap or a wrong
    # formula INSIDE the helper.  Recompute from the literal FV3 definitions
    # and pin both fields + guard against the swap.
    ucp, vcp = np.asarray(uc_pad), np.asarray(vc_pad)
    cs, rs = np.asarray(cosa), np.asarray(rsina)
    vc_sum_o = vcp[:, :-1, :] + vcp[:, 1:, :]
    uc_sum_o = ucp[:, :, :-1] + ucp[:, :, 1:]
    vb_o = dt5 * (vc_sum_o - uc_sum_o * cs) * rs
    ub_o = dt5 * (uc_sum_o - vc_sum_o * cs) * rs
    np.testing.assert_allclose(np.asarray(vb_g), vb_o, atol=1e-12, rtol=1e-12,
                               err_msg="vb != FV3 corner-Courant formula")
    np.testing.assert_allclose(np.asarray(ub_g), ub_o, atol=1e-12, rtol=1e-12,
                               err_msg="ub != FV3 corner-Courant formula")
    assert not np.allclose(np.asarray(vb_g), ub_o), "vb/ub swap not caught"

    def _tile(ti, tj):
        ai, aj = ti * nl, tj * nl
        uc_t = uc_pad[:, ai:ai + nl + 1, aj:aj + nl + 2]   # (6, nl+1, nl+2)
        vc_t = vc_pad[:, ai:ai + nl + 2, aj:aj + nl + 1]   # (6, nl+2, nl+1)
        cs_t = cosa[:, ai:ai + nl + 1, aj:aj + nl + 1]
        rs_t = rsina[:, ai:ai + nl + 1, aj:aj + nl + 1]
        return bgrid_corner_courant_local(uc_t, vc_t, cs_t, rs_t, dt5)

    # Reassemble the (n+1, n+1) corner block: each tile owns nl corner
    # rows/cols, the LAST tile owns the final +1 (shared-corner ownership).
    for idx, g in ((0, vb_g), (1, ub_g)):
        rows = []
        for ti in range(kt):
            cols = []
            for tj in range(kt):
                blk = np.asarray(_tile(ti, tj)[idx])    # (6, nl+1, nl+1)
                r_hi = nl + 1 if ti == kt - 1 else nl
                c_hi = nl + 1 if tj == kt - 1 else nl
                cols.append(blk[:, :r_hi, :c_hi])
            rows.append(np.concatenate(cols, axis=2))
        reassembled = np.concatenate(rows, axis=1)        # (6, n+1, n+1)
        np.testing.assert_allclose(
            reassembled, np.asarray(g), atol=1e-12, rtol=1e-12,
            err_msg=f"tiled corner Courant ({('vb', 'ub')[idx]}) != global")

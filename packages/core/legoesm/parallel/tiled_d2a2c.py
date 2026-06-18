"""Sub-face tiled FV3 d2a2c SPMD stage (P4 phase-1b).

Approach C: the cheap D→A step + global fields (:func:`d2a2c_global_fields`)
run in the global (face-replicated) view; the A→C compute is sharded over a
``(6, kt, kt)`` device mesh — each device ``lax.dynamic_slice``s its tile's
blocks from the replicated fields and runs the single device-uniform
:func:`d2a2c_tile_unified` (flag/mask-driven, codex-clean), so there is NO
dynamic wind-halo exchange.  With ``apply_strips=True`` (default) the two
adjacent strips are applied IN-STAGE: four 1-cell ``lax.ppermute`` halos on
the same-face tile axes feed :func:`d2a2c_tile_strips`, making the
reassembled output equal ``d2a2c_vect`` with NO global post-pass.  With
``apply_strips=False`` the strips are deferred and the caller applies
:func:`d2a2c_adjacent_strips` on the reassembled global staggered fields
(correctness proven bit-exact vs ``d2a2c_vect`` in
``tests/parallel/test_tiled_d2a2c_ua_va.py``).  Design:
``docs/scaling/d2a2c_spmd_stage_design.md``.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from legoesm.parallel.shard_map_compat import shard_map


def make_tiled_d2a2c_stage(mesh, cdgrid, kt: int, apply_strips: bool = True):
    """Build the sharded A→C d2a2c stage for a ``(6, kt, kt)`` mesh.

    mesh : ``jax.sharding.Mesh`` with axis names ``("face", "tile_i",
        "tile_j")`` and shape ``(6, kt, kt)``; ``cdgrid`` supplies the static
    staggered metrics (``cosa_u``/``rsin_u``/``cosa_v``/``rsin_v``), closed
    over since they never change per step.

    Returns ``stage(u_d, v_d, fields) -> (ua, va, uc, vc, ut, vt)`` where
    ``fields`` is the :func:`d2a2c_global_fields` output (face-replicated;
    its NamedTuple fields carry the per-step padded winds + static padded
    metrics).  Outputs are tile-sharded ``P("face", "tile_i", "tile_j")``
    with the staggered tile blocks DUPLICATING the shared boundary face
    between neighbouring tiles (gathered extent ``kt*(nl+1)``); duplicated
    copies are bit-identical (the strip exchange is symmetric).  With
    ``apply_strips=True`` (default) reassembly (lower tile owns the shared
    face) equals ``d2a2c_vect`` directly; with ``apply_strips=False`` the
    two adjacent strips are left as base and the caller must apply
    :func:`d2a2c_adjacent_strips` post-gather.
    """
    from legoesm.core.fv3_sw_core import (
        d2a2c_tile_strips,
        d2a2c_tile_unified,
    )

    n = cdgrid.n
    if n % kt:
        raise ValueError(f"n={n} not divisible by kt={kt}")
    nl = n // kt
    npt = min(4, n // 2)
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")
    # 1-cell strip halos: hi = value FROM tile t+1 (every source sends to
    # t-1), lo = FROM t-1 (sends to t+1); periodic wrap delivers garbage
    # at the face boundary, provably masked out by the [2, n-2] strip
    # range inside d2a2c_tile_strips.
    perm_hi = [(t, (t - 1) % kt) for t in range(kt)]
    perm_lo = [(t, (t + 1) % kt) for t in range(kt)]

    @partial(shard_map, mesh=mesh, in_specs=(fo,) * 18,
             out_specs=(co,) * 6, check_vma=False)
    def _stage(uxp, vxp, ua, dx, se, sw, va, dy, sn, ss,
               cos_sg5, rsin2, cosa_u, rsin_u, cosa_v, rsin_v, ud, vd):
        ti = jax.lax.axis_index("tile_i")
        tj = jax.lax.axis_index("tile_j")
        a, b = ti * nl, tj * nl

        def ds(arr, si, sj):
            return jax.lax.dynamic_slice(arr[0], (a, b), (si, sj))[None]

        cu_blk = ds(cosa_u, nl + 1, nl)
        cv_blk = ds(cosa_v, nl, nl + 1)
        out = d2a2c_tile_unified(
            ds(uxp, nl + 5, nl + 4), ds(vxp, nl + 4, nl + 5),
            ds(ua, nl + 4, nl + 4), ds(dx, nl + 4, nl),
            ds(se, nl + 2, nl + 2), ds(sw, nl + 2, nl + 2),
            ds(va, nl + 4, nl + 4), ds(dy, nl, nl + 4),
            ds(sn, nl + 2, nl + 2), ds(ss, nl + 2, nl + 2),
            ds(ud, nl, nl + 1), ds(vd, nl + 1, nl),
            ds(cos_sg5, nl, nl), ds(rsin2, nl, nl),
            cu_blk, ds(rsin_u, nl + 1, nl),
            cv_blk, ds(rsin_v, nl, nl + 1),
            a, b, n, npt, ti == 0, ti == kt - 1, tj == 0, tj == kt - 1)
        if not apply_strips:
            return out
        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = out
        # hi halo: every source sends its LOW edge (col/row 0) to t-1, so
        # receiver r holds tile r+1's cell b+nl / a+nl.  lo halo: sends
        # its HIGH edge (col/row nl-1) to t+1 -> receiver holds cell
        # b-1 / a-1.
        ut_hi = jax.lax.ppermute(ut_t[:, :, 0], "tile_j", perm_hi)
        ut_lo = jax.lax.ppermute(ut_t[:, :, nl - 1], "tile_j", perm_lo)
        vt_hi = jax.lax.ppermute(vt_t[:, 0, :], "tile_i", perm_hi)
        vt_lo = jax.lax.ppermute(vt_t[:, nl - 1, :], "tile_i", perm_lo)
        ut_t, vt_t = d2a2c_tile_strips(
            uc_t, vc_t, ut_t, vt_t, ut_lo, ut_hi, vt_lo, vt_hi,
            cu_blk, cv_blk, a, b, n, nl,
            ti == 0, ti == kt - 1, tj == 0, tj == kt - 1)
        return ua_t, va_t, uc_t, vc_t, ut_t, vt_t

    def stage(u_d, v_d, fields):
        # Wide low-padded covariant blocks: the kernel's -1 cell in-bounds
        # at a=0 (edge-pad; that cell is provably overwritten by the W/S
        # face specials — codex-verified).
        uxp = jnp.pad(fields.utmp_pad, [(0, 0), (1, 0), (0, 0)], mode='edge')
        vxp = jnp.pad(fields.vtmp_pad, [(0, 0), (0, 0), (1, 0)], mode='edge')
        return _stage(uxp, vxp, fields.ua_pad, fields.dxc_pad_x,
                      fields.se_pad_x, fields.sw_pad_x, fields.va_pad,
                      fields.dyc_pad_y, fields.sn_pad_y, fields.ss_pad_y,
                      fields.cos_sg5, fields.rsin2, cdgrid.cosa_u,
                      cdgrid.rsin_u, cdgrid.cosa_v, cdgrid.rsin_v, u_d, v_d)

    return stage

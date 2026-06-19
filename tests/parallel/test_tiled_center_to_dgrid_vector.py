"""np24 gate: tiled ``center_to_dgrid_vector`` (cc wind vector -> D-grid corners)
— the section-12c ``_vert_adv_uv_d`` lift the standalone momentum stage omits and
the full hydrostatic capstone must add (primitive_eq_cdgrid.py:957-977; the true
base-case du_d_dt = momentum(307-479) + center_to_dgrid_vector(vert_adv_uv_cc),
added UNCONDITIONALLY).

Base case (``use_fv3_a2b_ord4=False``): halo=1 ROTATING vector pad + the 4-pt
corner average.  Coord-FREE (pure geometry).  Reference = the global
``center_to_dgrid_vector`` op.  Output corner-staggered, so the per-tile corner
block is compared to the overlapping global staggered slice (the duplicated
shared corner face validates lower-owns-shared) to an FMA-robust relative
tolerance.  NOT a wall-clock measurement (np>6 anti-scales on Ginsburg); the
future-HW win is the capability.

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  kt=3 (nl=8 != 12) pins
the strip-ppermute direction + nl-dependent slice bugs.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.core.operators_cdgrid import center_to_dgrid_vector
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_center_to_dgrid_vector_stage_2d,
)

N = 24
NLEV = 6


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    u_cc = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    v_cc = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    return u_cc, v_cc


def _global_lift(u_cc, v_cc, cdgrid):
    # base case use_fv3_a2b_ord4=False (the default; the 2nd-order 4-pt path).
    u_d, v_d = center_to_dgrid_vector(u_cc, v_cc, cdgrid)
    return np.asarray(u_d), np.asarray(v_d)


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None, "base cut is non-duogrid"
    assert g.base.halo_interp_offsets is not None
    return g


def _compare_corner(u_t, v_t, u_g, v_g, kt, nl):
    """Per-tile corner block vs the overlapping global staggered (nl+1) slice."""
    blk = nl + 1
    wu = wv = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                gu = u_g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                tu = u_t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                wu = max(wu, float(np.max(np.abs(tu - gu))))
                gv = v_g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                tv = v_t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                wv = max(wv, float(np.max(np.abs(tv - gv))))
    su = float(np.max(np.abs(u_g))) + 1e-300
    sv = float(np.max(np.abs(v_g))) + 1e-300
    return wu / su, wv / sv


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_lift_matches_global(cdg, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    nl = N // KT
    u_cc, v_cc = _inputs(N, NLEV, 50 + KT)
    u_g, v_g = _global_lift(u_cc, v_cc, cdg)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_center_to_dgrid_vector_stage_2d(mesh, cdg, N, KT, NLEV)

    fw = NamedSharding(mesh, P("face", None, None, None))
    u_t, v_t = stage(jax.device_put(u_cc, fw), jax.device_put(v_cc, fw))

    ru, rv = _compare_corner(np.asarray(u_t), np.asarray(v_t), u_g, v_g, KT, nl)
    assert ru < 1e-10, f"u_d rel {ru:.3e} (kt={KT})"
    assert rv < 1e-10, f"v_d rel {rv:.3e} (kt={KT})"


def test_lift_compose_host_body(cdg):
    """Composition exactness WITHOUT 24 devices: slice the GLOBAL rotating vector
    pad per tile + the 4-pt corner average reassembled == global.  Isolates the
    slice arithmetic from the in-stage ppermute halo (which the np24 lane
    exercises via the real ndim=4 vector_body)."""
    from legoesm.grids.halo import pad_halo_vector_4d
    kt, nl = 3, N // 3
    u_cc, v_cc = _inputs(N, NLEV, 99)
    u_g, v_g = _global_lift(u_cc, v_cc, cdg)
    g = cdg.base
    u_pad_g, v_pad_g = pad_halo_vector_4d(
        u_cc, v_cc, g.cos_angle, g.sin_angle,
        g.cos_angle_padded, g.sin_angle_padded,
        interp_offsets=g.halo_interp_offsets, duogrid=None)
    u_pad_g = np.asarray(u_pad_g)
    v_pad_g = np.asarray(v_pad_g)
    blk = nl + 1
    wu = wv = 0.0
    for ti in range(kt):
        for tj in range(kt):
            a_i, a_j = ti * nl, tj * nl
            up = u_pad_g[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2]
            vp = v_pad_g[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2]
            ud = 0.25 * (up[:, :-1, :-1, :] + up[:, 1:, :-1, :]
                         + up[:, :-1, 1:, :] + up[:, 1:, 1:, :])
            vd = 0.25 * (vp[:, :-1, :-1, :] + vp[:, 1:, :-1, :]
                         + vp[:, :-1, 1:, :] + vp[:, 1:, 1:, :])
            gu = u_g[:, a_i:a_i + blk, a_j:a_j + blk]
            gv = v_g[:, a_i:a_i + blk, a_j:a_j + blk]
            wu = max(wu, float(np.max(np.abs(ud - gu))))
            wv = max(wv, float(np.max(np.abs(vd - gv))))
    ru = wu / (float(np.max(np.abs(u_g))) + 1e-300)
    rv = wv / (float(np.max(np.abs(v_g))) + 1e-300)
    assert ru < 1e-10, f"host-body u_d rel {ru:.3e}"
    assert rv < 1e-10, f"host-body v_d rel {rv:.3e}"

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
)
from legoesm.parallel.mesh import tiled_face_block

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

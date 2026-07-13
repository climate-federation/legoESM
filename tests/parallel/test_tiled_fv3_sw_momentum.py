"""np24 ASSEMBLY gate: the full tiled ``fv3_sw_tendencies`` MOMENTUM stage.

This is the genuine >6-device unlock the per-op tile kernels
(``test_tiled_{dgrid_vorticity,box_interps,al_gradient,velocity_transforms}``)
were building toward.  The assembly chains the production momentum path
   fv3_d2cc -> Bernoulli -> [B scalar halo] -> arakawa_lamb_gradient
   -> [u_cc/v_cc vector halo] -> corner winds -> dgrid_vorticity
   -> interp_corner_to_center -> du_cc/dv_cc -> [vector halo] -> D-grid project
on a ``(6, kt, kt)`` device mesh, with the three INTERMEDIATE halos
(B, the cc winds, the cc tendencies) exchanged IN-STAGE via the unwrapped
tiled pad bodies (``make_tiled_pad_body`` / ``make_tiled_pad_vector_body``) —
not slices of a global pre-pad (intermediates have none).

Gated by BIT-IDENTITY: each tile's ``(du_d_dt, dv_d_dt)`` block, compared to
the overlapping global ``fv3_sw_tendencies`` staggered slice (the duplicated
shared face validates lower-owns-shared), to a tight FMA-robust relative
tolerance.  NOT a wall-clock measurement (np24 on Ginsburg = CPU shard_map /
cross-node ppermute, both anti-scale here); the future-HW win is the
capability, the gate is correctness.

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  The kt=3 lane pins
the strip-ppermute DIRECTION (at kt=2 ``perm_hi == perm_lo``), and nl=8 != 12
catches any nl-dependent slice bug.
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
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_sw_momentum_stage_2d,
)

N = 24  # divisible by kt=2 (nl=12) and kt=3 (nl=8)


@pytest.fixture(scope="module")
def setup():
    set_halo_backend("local")
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert cdg.base.duogrid is None, "base cut is non-duogrid"
    assert cdg.base.halo_interp_offsets is not None, "needs interp offsets"
    rng = np.random.default_rng(414)
    # Random but finite state — bit-identity tests numerical equivalence, not
    # physics.  h positive-ish (height), winds O(1), h_s small topography.
    h = jnp.asarray(1.0e3 + rng.standard_normal((6, N, N)))
    u_d = jnp.asarray(rng.standard_normal((6, N, N + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, N + 1, N)))
    h_s = jnp.asarray(10.0 * rng.standard_normal((6, N, N)))
    # Global reference — BASE case (all optional terms off by default:
    # div_damp=0, hyperdiff_coeff=0, boundary_fix=False).
    _, du_g, dv_g = fv3_sw_tendencies(h, u_d, v_d, h_s, cdg)
    return cdg, h, u_d, v_d, h_s, np.asarray(du_g), np.asarray(dv_g)


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_momentum_matches_global(setup, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    NL = N // KT
    cdg, h, u_d, v_d, h_s, du_g, dv_g = setup

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_sw_momentum_stage_2d(mesh, cdg, N, KT)

    fo = P("face", None, None)
    sh_fo = NamedSharding(mesh, fo)
    h_sh = jax.device_put(h, sh_fo)
    u_sh = jax.device_put(u_d, sh_fo)
    v_sh = jax.device_put(v_d, sh_fo)
    hs_sh = jax.device_put(h_s, sh_fo)

    du_t, dv_t = stage(h_sh, u_sh, v_sh, hs_sh)
    du_t = np.asarray(du_t)   # (6, KT*NL, KT*(NL+1))
    dv_t = np.asarray(dv_t)   # (6, KT*(NL+1), KT*NL)

    blk_u_j = NL + 1          # du_d_dt: (nl, nl+1) staggered in j
    blk_v_i = NL + 1          # dv_d_dt: (nl+1, nl) staggered in i
    worst_du = worst_dv = 0.0
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                # du_d_dt (nl, nl+1): i centered, j staggered (shared j-edge).
                g_du = du_g[f, ti * NL: ti * NL + NL,
                            tj * NL: tj * NL + NL + 1]
                t_du = du_t[f, ti * NL:(ti + 1) * NL,
                            tj * blk_u_j:(tj + 1) * blk_u_j]
                worst_du = max(worst_du, float(np.max(np.abs(t_du - g_du))))
                # dv_d_dt (nl+1, nl): i staggered (shared i-row), j centered.
                g_dv = dv_g[f, ti * NL: ti * NL + NL + 1,
                            tj * NL: tj * NL + NL]
                t_dv = dv_t[f, ti * blk_v_i:(ti + 1) * blk_v_i,
                            tj * NL:(tj + 1) * NL]
                worst_dv = max(worst_dv, float(np.max(np.abs(t_dv - g_dv))))

    # FMA-robust relative tolerance: the assembly reorders the global pad's
    # contiguous arithmetic into per-tile ppermute receives (3 in-stage halos:
    # B scalar + 2 vector) -> O(1e-13) relative ULP drift, not algorithmic.
    scale_du = float(np.max(np.abs(du_g))) + 1e-300
    scale_dv = float(np.max(np.abs(dv_g))) + 1e-300
    rel_du = worst_du / scale_du
    rel_dv = worst_dv / scale_dv
    assert rel_du < 1e-10, (
        f"kt={KT} tiled du_d_dt vs global: abs={worst_du:.3e} "
        f"rel={rel_du:.3e}")
    assert rel_dv < 1e-10, (
        f"kt={KT} tiled dv_d_dt vs global: abs={worst_dv:.3e} "
        f"rel={rel_dv:.3e}")

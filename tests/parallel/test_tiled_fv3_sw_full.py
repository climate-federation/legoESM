"""np24 CAPSTONE gate: the FULL tiled fv3_sw_tendencies stage (all 3 outputs).

Composes the (separately-validated) tiled momentum assembly + mass-PPM
divergence into ONE shard_map stage returning (dh_dt, du_d_dt, dv_d_dt), with
the cc-wind VECTOR halo computed ONCE and shared between fv3_cc2c (-> the mass
PPM C-grid winds) and the momentum corner winds.  This is the genuine >6-device
production SW dycore tendency.

Gated by BIT-IDENTITY of ALL THREE tendencies vs the global fv3_sw_tendencies
(base case), to a tight FMA-robust relative tolerance.  NOT a wall-clock
measurement (np>6 anti-scales on Ginsburg); the future-HW win is the capability.

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
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
    make_tiled_fv3_sw_tendencies_stage_2d,
)

N = 24  # divisible by kt=2 (nl=12) and kt=3 (nl=8)


@pytest.fixture(scope="module")
def setup():
    set_halo_backend("local")
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert cdg.base.duogrid is None
    rng = np.random.default_rng(616)
    h = jnp.asarray(1.0e3 + rng.standard_normal((6, N, N)))
    u_d = jnp.asarray(rng.standard_normal((6, N, N + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, N + 1, N)))
    h_s = jnp.asarray(10.0 * rng.standard_normal((6, N, N)))
    dh_g, du_g, dv_g = fv3_sw_tendencies(h, u_d, v_d, h_s, cdg)  # base case
    return (cdg, h, u_d, v_d, h_s,
            np.asarray(dh_g), np.asarray(du_g), np.asarray(dv_g))


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_full_tendencies_match_global(setup, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    NL = N // KT
    cdg, h, u_d, v_d, h_s, dh_g, du_g, dv_g = setup

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_sw_tendencies_stage_2d(mesh, cdg, N, KT)

    fo = P("face", None, None)
    sh = NamedSharding(mesh, fo)
    dh_t, du_t, dv_t = stage(
        jax.device_put(h, sh), jax.device_put(u_d, sh),
        jax.device_put(v_d, sh), jax.device_put(h_s, sh))
    dh_t, du_t, dv_t = np.asarray(dh_t), np.asarray(du_t), np.asarray(dv_t)

    blk_u_j = NL + 1
    blk_v_i = NL + 1
    worst_h = worst_du = worst_dv = 0.0
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                # dh_dt: cc exact partition.
                gh = dh_g[f, ti * NL:(ti + 1) * NL, tj * NL:(tj + 1) * NL]
                th = dh_t[f, ti * NL:(ti + 1) * NL, tj * NL:(tj + 1) * NL]
                worst_h = max(worst_h, float(np.max(np.abs(th - gh))))
                # du_d_dt (nl, nl+1) j-staggered; dv_d_dt (nl+1, nl) i-staggered.
                gdu = du_g[f, ti * NL:ti * NL + NL, tj * NL:tj * NL + NL + 1]
                tdu = du_t[f, ti * NL:(ti + 1) * NL,
                           tj * blk_u_j:(tj + 1) * blk_u_j]
                worst_du = max(worst_du, float(np.max(np.abs(tdu - gdu))))
                gdv = dv_g[f, ti * NL:ti * NL + NL + 1, tj * NL:tj * NL + NL]
                tdv = dv_t[f, ti * blk_v_i:(ti + 1) * blk_v_i,
                           tj * NL:(tj + 1) * NL]
                worst_dv = max(worst_dv, float(np.max(np.abs(tdv - gdv))))

    def _rel(worst, ref):
        return worst / (float(np.max(np.abs(ref))) + 1e-300)

    rh, rdu, rdv = _rel(worst_h, dh_g), _rel(worst_du, du_g), _rel(worst_dv, dv_g)
    assert rh < 1e-10, f"kt={KT} dh_dt: abs={worst_h:.3e} rel={rh:.3e}"
    assert rdu < 1e-10, f"kt={KT} du_d_dt: abs={worst_du:.3e} rel={rdu:.3e}"
    assert rdv < 1e-10, f"kt={KT} dv_d_dt: abs={worst_dv:.3e} rel={rdv:.3e}"


def test_tiled_full_is_differentiable(setup):
    """End-to-end jax.grad through the tiled full stage (kt=2, np24).

    Bit-identity parity proves the FORWARD pass; it does NOT prove transpose
    safety of the in-stage halo ppermutes (codex audit MED).  This drives
    reverse-mode AD through the whole chain (scalar B halo + 2 vector halos +
    PPM + the *_core ops) — the ppermute VJP is a ppermute with the inverse
    permutation — and checks the gradient is finite and non-trivial.  End-to-end
    differentiability is a first-class repo goal."""
    if len(jax.devices()) < 24:
        pytest.skip("needs --xla_force_host_platform_device_count=24")
    cdg, h, u_d, v_d, h_s, *_ = setup

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:24]).reshape(6, 2, 2)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_sw_tendencies_stage_2d(mesh, cdg, N, 2)
    sh = NamedSharding(mesh, P("face", None, None))
    h_sh = jax.device_put(h, sh)
    u_sh = jax.device_put(u_d, sh)
    v_sh = jax.device_put(v_d, sh)
    hs_sh = jax.device_put(h_s, sh)

    def loss(hh, uu, vv, hs):
        dh, du, dv = stage(hh, uu, vv, hs)
        return jnp.sum(dh ** 2) + jnp.sum(du ** 2) + jnp.sum(dv ** 2)

    # Grad w.r.t. ALL FOUR inputs (not just h): the wind grads route through the
    # SHARED u_cc/v_cc vector halo, so this exercises that halo's transpose too
    # (codex audit MED — grad-wrt-h alone leaves u_d/v_d closed-over constants).
    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(h_sh, u_sh, v_sh, hs_sh)
    for name, g in zip(("h", "u_d", "v_d", "h_s"), grads):
        ga = np.asarray(g)
        assert np.all(np.isfinite(ga)), f"non-finite gradient w.r.t. {name}"
    # The wind gradients (gu, gv) flow through the shared vector halo; require
    # them non-trivial (h/h_s could be near-zero in a contrived case, winds not).
    assert max(float(np.max(np.abs(np.asarray(g)))) for g in grads[1:3]) > 0.0, \
        "wind gradients all-zero (vector-halo transpose broke)"

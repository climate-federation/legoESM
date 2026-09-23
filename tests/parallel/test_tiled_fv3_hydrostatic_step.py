"""np24 gate: the full tiled SSP-RK3 STEP (make_tiled_fv3_hydrostatic_step_stage_2d)
— the no-gather time step wrapping the hydrostatic tendency capstone on np=6*kt^2.

Reference = the SAME SSP-RK3 (timestepping/ssp_rk3.py) over the REAL base-cut
``fv3_hydrostatic_tendencies`` (config sponge_tau_sec=0; the _step_fv3 RK3 WITHOUT
the post-step _sync_dgrid_boundary/sponge).  Inputs are GENTLE (small winds, T~250,
p_s~1e5) so the per-RK3-stage T_min / p_floor clips inside
fv3_hydrostatic_tendencies stay no-ops (the tiled stage's base-cut tendency does
not clip) — the gate is then a pure numerical-equivalence test of the tiled RK3.

Gated by BIT-IDENTITY: the stepped u_d/v_d (corner-staggered) per-tile vs the
overlapping global slice; the stepped T/p_s (cc, exact partition) elementwise.
Tolerances admit the 3-stage accumulation of the documented cc XLA-fusion
reorder (~5e-8/stage, capstone diag 8512950): u_d/v_d corner stay tight; T/p_s cc
looser.  A real bug is O(1e-2)+.  NOT a wall-clock measurement (np>6 anti-scales
on Ginsburg).  Covers sigma + hybrid.

Host CPU devices = 6*kt^2: 24 (kt=2) / 54 (kt=3) / 96 (kt=4) / 216 (kt=6) /
384 (kt=8); ``XLA_FLAGS=--xla_force_host_platform_device_count=384`` covers all.
kt above 3 is what _VALIDATED_KT grows on.
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
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels
from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    fv3_hydrostatic_tendencies, CDGridPrimitiveEquationConfig,
)
from legoesm.parallel import tiled_production_cdgrid as tiled_mod
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_hydrostatic_step_stage_2d,
)

N = 48   # divisible by every validated kt {2,3,4,6,8}; nl >= 6 per tile
NLEV = 6
P_FLOOR = CDGridPrimitiveEquationConfig().p_floor
DT = 100.0   # s — a small step; gentle inputs keep the per-stage clips inactive


def _inputs(n, nlev, seed):
    """GENTLE finite state so no RK3 intermediate trips the T_min/p_floor clamps
    inside fv3_hydrostatic_tendencies (which the base-cut tiled tendency omits)."""
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(0.1 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(0.1 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(250.0 + 2.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 50.0 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e2 * rng.standard_normal((6, n, n)))
    return u_d, v_d, T, p_s, phis


def _global_step(u_d, v_d, T, p_s, phis, cdgrid, coord, dt):
    """The base-cut RK3 step = ssp_rk3_step over the real fv3_hydrostatic_
    tendencies (sponge off), i.e. _step_fv3 minus the post-step ops."""
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    state = FV3HydrostaticState(
        u_d=Field(data=u_d, name="u_d", dims=d3, units="m/s"),
        v_d=Field(data=v_d, name="v_d", dims=d3, units="m/s"),
        T=Field(data=T, name="T", dims=d3, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=phis, name="phis", dims=d2, units="m^2/s^2"))
    cfg = CDGridPrimitiveEquationConfig(sponge_tau_sec=0.0)

    def tendency_fn(s):
        tend = fv3_hydrostatic_tendencies(s, cdgrid.base, coord, cdgrid, cfg)
        return FV3HydrostaticState(
            u_d=s.u_d.replace(data=tend.du_d_dt.data),
            v_d=s.v_d.replace(data=tend.dv_d_dt.data),
            T=s.T.replace(data=tend.dT_dt.data),
            p_s=s.p_s.replace(data=tend.dp_s_dt.data),
            phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)))

    stepped = ssp_rk3_step(state, tendency_fn, dt)
    return (np.asarray(stepped.u_d.data), np.asarray(stepped.v_d.data),
            np.asarray(stepped.T.data), np.asarray(stepped.p_s.data))


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None
    return g


def _rel_corner(t, g, kt, nl):
    blk = nl + 1
    worst = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                gg = g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                tt = t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                worst = max(worst, float(np.max(np.abs(tt - gg))))
    return worst / (float(np.max(np.abs(g))) + 1e-300)


def _rel_cc(t, g):
    return float(np.max(np.abs(np.asarray(t) - g))) / (
        float(np.max(np.abs(g))) + 1e-300)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("KT", [2, 3, 4, 6, 8])
def test_tiled_step_matches_global(cdg, KT, hybrid, monkeypatch):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    # This test IS the bit-identity receipt that _VALIDATED_KT is grown on, so
    # it must run for a kt the deployment guard still refuses; the guard stays
    # in force everywhere else.
    monkeypatch.setattr(tiled_mod, "_VALIDATED_KT",
                        tiled_mod._VALIDATED_KT | {KT})
    nl = N // KT
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV, (80 if hybrid else 90) + KT)
    ug, vg, Tg, psg = _global_step(u_d, v_d, T, p_s, phis, cdg, coord, DT)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    step = make_tiled_fv3_hydrostatic_step_stage_2d(
        mesh, cdg, coord, N, KT, NLEV, p_floor=P_FLOOR, dt=DT)

    fw = NamedSharding(mesh, P("face", None, None, None))
    fo = NamedSharding(mesh, P("face", None, None))
    ut, vt, Tt, pst = step(
        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
        jax.device_put(T, fw), jax.device_put(p_s, fo),
        jax.device_put(phis, fo))

    r_u = _rel_corner(np.asarray(ut), ug, KT, nl)
    r_v = _rel_corner(np.asarray(vt), vg, KT, nl)
    r_T = _rel_cc(Tt, Tg)
    r_ps = _rel_cc(pst, psg)
    tag = f"(kt={KT}, hybrid={hybrid})"
    # stepped state = s0 + O(dt)*tendency, so the tendency's reorder is scaled by
    # ~dt and diluted by s0; u_d/v_d stay near bit-identity, T/p_s carry the
    # 3-stage cc-fusion reorder (capstone diag 8512950).  A real bug is O(1e-2)+.
    assert r_u < 1e-9, f"u_d rel {r_u:.3e} {tag}"
    assert r_v < 1e-9, f"v_d rel {r_v:.3e} {tag}"
    assert r_T < 1e-7, f"T rel {r_T:.3e} {tag}"
    assert r_ps < 1e-7, f"p_s rel {r_ps:.3e} {tag}"


def test_tiled_step_rejects_bad_dt(cdg):
    """Fail-loud factory guard on dt."""
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    coord = create_sigma_coordinate(NLEV)
    with pytest.raises(ValueError, match="dt must be"):
        make_tiled_fv3_hydrostatic_step_stage_2d(
            mesh, cdg, coord, N, 1, NLEV, p_floor=P_FLOOR, dt=0.0)

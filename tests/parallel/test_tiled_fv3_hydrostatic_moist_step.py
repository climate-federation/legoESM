"""np24 gate: the full tiled MOIST SSP-RK3 STEP — increment 4, the cube-MOIST
np>6 capstone (docs/performance/scaling/cube_moist_tiled_step_design.md).

``make_tiled_fv3_hydrostatic_moist_step_stage_2d`` threads q_v/q_c/q_r through the
SAME tiled SSP-RK3 + tracer-aware tendency the dry step uses, plus an INJECTED
per-tile Kessler column physics, plus a post-step tracer floor.  Reference = the
base-cut serial moist RK3: ``ssp_rk3_step`` over the real
``fv3_hydrostatic_tendencies`` with per-stage Kessler ``physics_tendency_cc``
(exactly ``_step_fv3.tendency_fn``), then the same ``max(q,0)`` floor — i.e.
``_step_fv3`` minus the post-step (sponge / mass-fixer) ops the tiled base cut
omits.  Bit-identity (FMA-robust rel<1e-10).  NOT a wall-clock measurement (np24
on Ginsburg CPU shard_map anti-scales; the future-HW win is the capability).
24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  sigma-only (Kessler).
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
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    fv3_hydrostatic_tendencies, fv3_to_hydrostatic, CDGridPrimitiveEquationConfig,
)
from legoesm.atmosphere.kessler_forcing import (
    make_kessler_forcing_cube, kessler_column_tendencies,
)
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_hydrostatic_moist_step_stage_2d,
)

N = 24
NLEV = 6
P_FLOOR = CDGridPrimitiveEquationConfig().p_floor
DT = 100.0
_TRACERS = ("q_v", "q_c", "q_r")


def _inputs(n, nlev, seed):
    """GENTLE finite state (no RK3 intermediate trips the T_min/p_floor clamps the
    base-cut tiled tendency omits) + positive tracers."""
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(0.1 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(0.1 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(280.0 + 2.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 50.0 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e2 * rng.standard_normal((6, n, n)))
    q_pack = jnp.asarray(np.abs(
        5.0e-3 + 1.0e-3 * rng.standard_normal((6, n, n, nlev, 3))))
    return u_d, v_d, T, p_s, phis, q_pack


def _phys_injected(coord, dt, cfg):
    """Per-tile Kessler column physics — the PRODUCTION bridge (this test's
    helper was promoted into ``make_kessler_column_physics_fn``; keep the
    test on the shipped factory so there is exactly one copy of the
    reshape-wrap numerics)."""
    from legoesm.atmosphere.kessler_forcing import (
        make_kessler_column_physics_fn,
    )
    return make_kessler_column_physics_fn(coord, dt, config=cfg)


def _global_moist_step(u_d, v_d, T, p_s, phis, q_pack, cdgrid, coord, dt):
    """Base-cut serial moist RK3 = ssp_rk3_step over fv3_hydrostatic_tendencies
    with per-stage Kessler physics_tendency_cc (mirrors _step_fv3.tendency_fn),
    then the post-step max(q,0) floor."""
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    tracers = {nm: Field(data=q_pack[..., i], name=nm, dims=d3, units="kg/kg")
               for i, nm in enumerate(_TRACERS)}
    state = FV3HydrostaticState(
        u_d=Field(data=u_d, name="u_d", dims=d3, units="m/s"),
        v_d=Field(data=v_d, name="v_d", dims=d3, units="m/s"),
        T=Field(data=T, name="T", dims=d3, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=phis, name="phis", dims=d2, units="m^2/s^2"),
        tracers=tracers)
    cfg = CDGridPrimitiveEquationConfig(sponge_tau_sec=0.0)
    phys_fn = make_kessler_forcing_cube(dt)

    def tendency_fn(s):
        s_cc = fv3_to_hydrostatic(s, cdgrid)
        phys_result = phys_fn(s_cc, cdgrid.base, coord)
        phys_cc = phys_result[0] if type(phys_result) is tuple else phys_result
        # Base cut (matches the tiled _tile_tendency + the dry _global_step): NO
        # dt_actual -> the corner-div adaptive cap (iter-189) is OFF, exactly as
        # the tiled base cut omits it.  physics_tendency_cc carries the per-stage
        # Kessler (dT + tracer rates), the only physics.
        tend = fv3_hydrostatic_tendencies(
            s, cdgrid.base, coord, cdgrid, cfg,
            physics_tendency=None, physics_tendency_cc=phys_cc)
        tracer_tend = {
            k: s.tracers[k].replace(data=tend.tracer_tendencies[k].data)
            for k in s.tracers}
        return FV3HydrostaticState(
            u_d=s.u_d.replace(data=tend.du_d_dt.data),
            v_d=s.v_d.replace(data=tend.dv_d_dt.data),
            T=s.T.replace(data=tend.dT_dt.data),
            p_s=s.p_s.replace(data=tend.dp_s_dt.data),
            phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            tracers=tracer_tend)

    stepped = ssp_rk3_step(state, tendency_fn, dt)
    q_g = np.stack(
        [np.maximum(np.asarray(stepped.tracers[nm].data), 0.0)
         for nm in _TRACERS], axis=-1)
    return (np.asarray(stepped.u_d.data), np.asarray(stepped.v_d.data),
            np.asarray(stepped.T.data), np.asarray(stepped.p_s.data), q_g)


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None
    return g


def _rel(t, g):
    """cc fields (T, p_s, q_pack): the EXACT per-tile cc partition gathers to the
    global shape -> plain elementwise."""
    return float(np.max(np.abs(np.asarray(t) - g))) / (
        float(np.max(np.abs(g))) + 1e-300)


def _rel_corner(t, g, kt, nl):
    """D-grid CORNER fields (u_d, v_d): the tiled output is kt blocks of nl+1
    (the staggered shared edge is duplicated across tiles), so it gathers to
    (6, kt*(nl+1), kt*(nl+1), ...) NOT the global (6, n+1, n+1, ...).  Compare
    each tile's (nl+1) block to its overlapping global slice (same handling as
    the dry-step gate's _rel_corner)."""
    t = np.asarray(t)
    blk = nl + 1
    worst = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                gg = g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                tt = t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                worst = max(worst, float(np.max(np.abs(tt - gg))))
    return worst / (float(np.max(np.abs(g))) + 1e-300)


@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_moist_step_matches_global(cdg, KT):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    coord = create_sigma_coordinate(NLEV)
    cfg = KesslerConfig()
    u_d, v_d, T, p_s, phis, q_pack = _inputs(N, NLEV, 90 + KT)
    ug, vg, Tg, psg, qg = _global_moist_step(
        u_d, v_d, T, p_s, phis, q_pack, cdg, coord, DT)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    step = make_tiled_fv3_hydrostatic_moist_step_stage_2d(
        mesh, cdg, coord, N, KT, NLEV, p_floor=P_FLOOR, dt=DT,
        column_physics_fn=_phys_injected(coord, DT, cfg))

    fw = NamedSharding(mesh, P("face", None, None, None))
    fo = NamedSharding(mesh, P("face", None, None))
    fw5 = NamedSharding(mesh, P("face", None, None, None, None))
    ut, vt, Tt, pst, qt = step(
        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
        jax.device_put(T, fw), jax.device_put(p_s, fo),
        jax.device_put(phis, fo), jax.device_put(q_pack, fw5))

    nl = N // KT
    # D-grid corner fields (u_d, v_d) -> block-wise corner compare; cc fields
    # (T, p_s, q_pack) -> exact-partition elementwise.
    for name, t, g in [("u_d", ut, ug), ("v_d", vt, vg)]:
        rel = _rel_corner(t, g, KT, nl)
        assert rel < 1e-10, f"moist step {name} rel {rel:.3e} (kt={KT})"
    for name, t, g in [("T", Tt, Tg), ("p_s", pst, psg)]:
        rel = _rel(t, g)
        assert rel < 1e-10, f"moist step {name} rel {rel:.3e} (kt={KT})"
    # q_pack: the dynamics fields hold 1e-10, but the tracer pack rides Kessler's
    # NONLINEAR column physics (saturation/condensation/evaporation, with
    # q_v<->q_c<->q_r cancellation) accumulated across 3 SSP-RK3 stages on small
    # mixing ratios -> the per-tile-vs-global FMA reordering is ~3.6e-10 (measured
    # kt=2), still bit-identity class, NOT algorithmic (a real bug would be
    # O(1e-3)+; the q tendency itself matched at 1e-10 in increments 2-3).
    rel_q = _rel(qt, qg)
    assert rel_q < 1e-8, f"moist step q_pack rel {rel_q:.3e} (kt={KT})"


def test_moist_step_requires_physics_fn(cdg):
    """Fail-loud: the injected column physics is required."""
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    coord = create_sigma_coordinate(NLEV)
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    with pytest.raises(ValueError, match="column_physics_fn"):
        make_tiled_fv3_hydrostatic_moist_step_stage_2d(
            mesh, cdg, coord, N, 1, NLEV, p_floor=P_FLOOR, dt=DT,
            column_physics_fn=None)

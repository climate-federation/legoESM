"""np24 gate: the BLOCKED-I/O persistent tiled step + the closed-loop adapter.

The np>6 PRODUCTION assembly: unlike the single-shot step stages
(face-replicated in, tile-sharded out — feeding back requires a full-cube
gather), ``make_tiled_fv3_hydrostatic_step_blocked_2d`` has input layout ==
output layout, so ``s = step(s)`` iterates with no per-step gather, and the
serial post-step dry-mass fixer runs IN-STAGE (``fix_mass=True``).

Reference = the SERIAL production ``model.step`` on the FV3 D-grid state
(the same multi-step trajectory the cs-spmd bench times at np<=6), with the
PRODUCTION default conservation config (use_conservation_fixer + fix_mass +
zero_mean_ps_tendency; anchor off -> both sides telescope the pre-step
mass).  Gentle inputs keep the serial per-stage T_min/p_floor clips inactive
(the tiled base-cut tendency omits them) — the stage-gate recipe.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
)
from legoesm.parallel.tiled_production_cdgrid import (
    expand_corners_to_blocks,
    make_tiled_fv3_hydrostatic_step_blocked_2d,
)
from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
    dedup_tiled_corners,
)

N, NLEV, KT = 8, 4, 2
NL = N // KT
DT = 60.0
N_STEPS = 3


def _mesh():
    # RAW (6, kt, kt) mesh + LOCAL halo backend — the stage-gate setup
    # (create_device_mesh would arm global halo state that reroutes the
    # serial reference's pads; see test_tiled_cc_step_adapter._mesh).
    from jax.sharding import Mesh
    if len(jax.devices()) < 6 * KT * KT:
        pytest.skip(f"needs {6 * KT * KT} devices "
                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
    set_halo_backend("local")
    dev = np.array(jax.devices()[: 6 * KT * KT]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


def _inputs(n, nlev, seed=7):
    """Gentle finite state (stage-gate recipe): no RK3 intermediate trips
    the serial T_min/p_floor clips the tiled base-cut tendency omits."""
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(0.1 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(0.1 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(250.0 + 2.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 50.0 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e2 * rng.standard_normal((6, n, n)))
    return u_d, v_d, T, p_s, phis


def _fv3_state(u_d, v_d, T, p_s, phis):
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    return FV3HydrostaticState(
        u_d=Field(data=u_d, name="u_d", dims=d3, units="m/s"),
        v_d=Field(data=v_d, name="v_d", dims=d3, units="m/s"),
        T=Field(data=T, name="T", dims=d3, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=phis, name="phis", dims=d2, units="m^2/s^2"))


def _rel_corner_blocked(t, g, kt, nl):
    """Worst per-tile rel diff of a BLOCKED corner field vs global corners."""
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
    return float(np.max(np.abs(np.asarray(t) - np.asarray(g)))) / (
        float(np.max(np.abs(np.asarray(g)))) + 1e-300)


# ---------------------------------------------------------------------------
# Pure layout tests (no mesh)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kt", [2, 3])
def test_expand_dedup_roundtrip(kt):
    nl = 4
    n = kt * nl
    rng = np.random.default_rng(3)
    g = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, 2)))
    blocked = expand_corners_to_blocks(g, kt, nl)
    assert blocked.shape == (6, kt * (nl + 1), kt * (nl + 1), 2)
    back = dedup_tiled_corners(blocked, kt, nl)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(g))


def test_expand_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expand_corners_to_blocks"):
        expand_corners_to_blocks(jnp.zeros((6, 8, 8, 1)), 2, 4)  # needs 9x9


# ---------------------------------------------------------------------------
# np24 closed-loop gate: multi-step blocked trajectory vs serial model.step
# with the PRODUCTION conservation config (in-stage telescoping fixer).
# ---------------------------------------------------------------------------

def test_blocked_loop_matches_serial_production_config():
    mesh = _mesh()
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert grid.duogrid is None
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=False, zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV)

    # Serial production trajectory (D-grid state end to end — the np<=6
    # cs-spmd bench path).
    ref = _fv3_state(u_d, v_d, T, p_s, phis)
    for _ in range(N_STEPS):
        ref = model.step(ref, DT)

    # Blocked closed loop: expand once, iterate, never gather in-loop.
    step = make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, coord, N, KT, NLEV,
        p_floor=float(cfg.p_floor), dt=DT,
        sponge_sigma=float(cfg.sponge_sigma),
        sponge_tau_sec=float(cfg.sponge_tau_sec),
        fix_mass=True)
    ub = expand_corners_to_blocks(u_d, KT, NL)
    vb = expand_corners_to_blocks(v_d, KT, NL)
    # jit ONCE (the production wrappers all jit the blocked step; an
    # eager shard_map call would re-lower per step — minutes each at 24
    # virtual devices).
    step_jit = jax.jit(lambda u, v, t, ps: step(u, v, t, ps, phis))
    s = (ub, vb, T, p_s)
    for _ in range(N_STEPS):
        s = step_jit(*s)
    ub3, vb3, T3, ps3 = s

    r_u = _rel_corner_blocked(np.asarray(ub3), np.asarray(ref.u_d.data),
                              KT, NL)
    r_v = _rel_corner_blocked(np.asarray(vb3), np.asarray(ref.v_d.data),
                              KT, NL)
    r_T = _rel_cc(T3, ref.T.data)
    r_ps = _rel_cc(ps3, ref.p_s.data)
    # x64 bounds: stage-gate single-step tolerances (corner 1e-9 / cc 1e-7,
    # the documented cc XLA-fusion reorder class) with 3-step accumulation
    # headroom.  A real halo/layout/fixer bug is orders of magnitude above.
    assert r_u < 3e-9, f"u_d rel {r_u:.3e}"
    assert r_v < 3e-9, f"v_d rel {r_v:.3e}"
    assert r_T < 3e-7, f"T rel {r_T:.3e}"
    assert r_ps < 3e-7, f"p_s rel {r_ps:.3e}"

    # Conservation: the in-stage telescoping fixer holds the global dry
    # mass at its initial value (the fixer's defining property).
    area = np.asarray(grid.area)
    mass0 = float(np.sum(np.asarray(p_s) * area))
    mass3 = float(np.sum(np.asarray(ps3) * area))
    assert abs(mass3 - mass0) / mass0 < 1e-12, (mass0, mass3)

    # Duplicated shared-face self-consistency across steps: adjacent tiles'
    # duplicated corner rows/cols must remain BIT-IDENTICAL after N steps
    # (the no-gather precondition).
    blk = NL + 1
    for arr in (np.asarray(ub3), np.asarray(vb3)):
        for i in range(1, KT):
            np.testing.assert_array_equal(
                arr[:, i * blk, :, :], arr[:, i * blk - 1, :, :])
            np.testing.assert_array_equal(
                arr[:, :, i * blk, :], arr[:, :, i * blk - 1, :])


def test_blocked_step_bit_identical_to_shipped_stage():
    """THE no-new-numerics gate: one blocked step == one shipped single-shot
    step stage, BITWISE, on a production-magnitude (Held-Suarez) state.

    The blocked variant re-plumbs LAYOUT only (state arrives tile-local
    instead of being sliced from face-replicated inputs); the tendency/RK3
    bodies are the shared builders.  Any numerical delta here is a bug.
    """
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.parallel.tiled_production_cdgrid import (
        make_tiled_fv3_hydrostatic_step_stage_2d,
    )

    mesh = _mesh()
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig()
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    hs = held_suarez_init(grid, coord)
    u_d0, v_d0 = center_to_dgrid_vector(hs.u.data, hs.v.data, model.cdgrid)

    kw = dict(p_floor=float(cfg.p_floor), dt=DT,
              sponge_sigma=float(cfg.sponge_sigma),
              sponge_tau_sec=float(cfg.sponge_tau_sec))
    stage = make_tiled_fv3_hydrostatic_step_stage_2d(
        mesh, cdgrid, coord, N, KT, NLEV, **kw)
    blocked = make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, coord, N, KT, NLEV, **kw, fix_mass=False)

    a = jax.jit(stage)(u_d0, v_d0, hs.T.data, hs.p_s.data, hs.phis.data)
    b = jax.jit(lambda u, v, t, ps, ph: blocked(u, v, t, ps, ph))(
        expand_corners_to_blocks(u_d0, KT, NL),
        expand_corners_to_blocks(v_d0, KT, NL),
        hs.T.data, hs.p_s.data, hs.phis.data)
    for nm, x, y in zip(("u_d", "v_d", "T", "p_s"), a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y),
                                      err_msg=nm)


def test_moist_blocked_step_bit_identical_to_shipped_moist_stage():
    """Moist twin of the no-new-numerics gate: one moist blocked step ==
    one shipped moist step stage, BITWISE, with nonzero tracers, a
    deterministic injected column physics, AND a negative-q input so the
    post-step ``max(q, 0)`` floor is exercised (codex: the moist blocked
    contract must not ship untested)."""
    from legoesm.parallel.tiled_production_cdgrid import (
        make_tiled_fv3_hydrostatic_moist_step_stage_2d,
    )

    mesh = _mesh()
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV, seed=11)
    rng = np.random.default_rng(12)
    # Small tracers with NEGATIVE excursions (floor must fire identically).
    q = jnp.asarray(1e-3 * rng.standard_normal((6, N, N, NLEV, 3)))
    assert float(jnp.min(q)) < 0.0

    def col_phys(T_t, p_s_t, qv, qc, qr):
        # Deterministic, column-local, tendency-shaped — same fn injected
        # into BOTH sides, so any output delta is layout plumbing.
        dT = 1e-4 * qv
        return dT, -0.1 * qv, 0.05 * qv - 0.02 * qc, 0.02 * qc - 0.01 * qr

    kw = dict(p_floor=100.0, dt=DT)
    stage = make_tiled_fv3_hydrostatic_moist_step_stage_2d(
        mesh, cdgrid, coord, N, KT, NLEV, **kw, column_physics_fn=col_phys)
    blocked = make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, coord, N, KT, NLEV, **kw, column_physics_fn=col_phys)

    a = jax.jit(stage)(u_d, v_d, T, p_s, phis, q)
    b = jax.jit(lambda u, v, t, ps, ph, qq: blocked(u, v, t, ps, ph, qq))(
        expand_corners_to_blocks(u_d, KT, NL),
        expand_corners_to_blocks(v_d, KT, NL), T, p_s, phis, q)
    for nm, x, y in zip(("u_d", "v_d", "T", "p_s", "q"), a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y),
                                      err_msg=nm)
    # The floor actually fired (post-step q is non-negative everywhere,
    # and the pre-step q had negative values).
    assert float(jnp.min(b[4])) >= 0.0


def test_adapter_moist_loop_kessler_matches_serial():
    """MOIST closed loop with the PRODUCTION Kessler bridge vs the serial
    ``model.step(physics_fn=make_kessler_forcing_cube)`` trajectory from
    the SAME entry conversion — the drivers' moist np>6 lane end to end
    (dynamics tracer advection + per-stage column Kessler + q floor +
    in-stage mass fixer)."""
    from legoesm.core.field import Field
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
        make_kessler_column_physics_fn, make_kessler_forcing_cube,
    )
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        _TILED_TRACERS, make_tiled_cc_loop,
    )

    mesh = _mesh()
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=False, zero_mean_ps_tendency=True)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    hs = held_suarez_init(grid, coord)
    rng = np.random.default_rng(21)
    d3 = ("face", "x", "y", "level")
    tracers = {
        nm: Field(data=jnp.asarray(np.abs(
                s + 1e-4 * rng.standard_normal((6, N, N, NLEV)))),
                  name=nm, dims=d3, units="kg/kg")
        for nm, s in zip(_TILED_TRACERS, (5e-3, 5e-4, 5e-5))
    }
    hs = hs._replace(tracers=tracers)

    # Serial: same entry conversion, then the production step with the
    # grid-space Kessler physics_fn (the np<=6 lane) — SAME shared column
    # core as the tiled bridge.
    u_d0, v_d0 = center_to_dgrid_vector(hs.u.data, hs.v.data, model.cdgrid)
    ref = _fv3_state(u_d0, v_d0, hs.T.data, hs.p_s.data, hs.phis.data)
    ref = ref._replace(tracers=tracers)
    phys_serial = make_kessler_forcing_cube(DT)
    for _ in range(N_STEPS):
        ref = model.step(ref, DT, physics_fn=phys_serial)

    col_fn = make_kessler_column_physics_fn(coord, DT)
    enter, step, exit_ = make_tiled_cc_loop(
        model, mesh, kt=KT, dt=DT, column_physics_fn=col_fn)
    step_jit = jax.jit(step)
    blk = enter(hs)
    for _ in range(N_STEPS):
        blk = step_jit(blk)

    # ABS bounds in the shipped adapter-gate class (see the dry loop gate's
    # rationale — the pre-existing face-corner wind term).
    def _abs(t, g):
        return float(np.max(np.abs(np.asarray(t) - np.asarray(g))))

    assert _abs(blk["T"], ref.T.data) < 1e-4
    assert _abs(blk["p_s"], ref.p_s.data) < 0.06
    for i, nm in enumerate(_TILED_TRACERS):
        d_q = _abs(blk["q_pack"][..., i], ref.tracers[nm].data)
        assert d_q < 1e-7, f"{nm} abs {d_q:.3e}"
    assert float(jnp.min(blk["q_pack"])) >= 0.0   # floor held

    # exit_ unpacks the advanced tracers (not the template's).
    out = exit_(blk, hs)
    for i, nm in enumerate(_TILED_TRACERS):
        np.testing.assert_array_equal(
            np.asarray(out.tracers[nm].data),
            np.asarray(blk["q_pack"][..., i]))


def test_adapter_loop_refuses_tracer_state():
    """The loop adapter is dry-only: silently freezing tracers while the
    serial step advances them is a divergence-by-omission (codex BLOCKER)."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        make_tiled_cc_loop,
    )

    mesh = _mesh()
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=False, zero_mean_ps_tendency=True)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    hs = held_suarez_init(grid, coord)
    enter, _, _ = make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)
    hs_q = hs._replace(tracers={"q_v": hs.T})   # any non-empty tracer dict
    with pytest.raises(NotImplementedError,
                       match="no column_physics_fn"):
        enter(hs_q)
    # Moist mode with the WRONG tracer set refuses too.
    enter_m, _, _ = make_tiled_cc_loop(
        model, mesh, kt=KT, dt=DT,
        column_physics_fn=lambda T, ps, qv, qc, qr: (
            0.0 * T, 0.0 * qv, 0.0 * qc, 0.0 * qr))
    with pytest.raises(NotImplementedError, match="exactly"):
        enter_m(hs_q)


def test_blocked_step_refuses_q_physics_mismatch():
    mesh = _mesh()
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = create_sigma_coordinate(NLEV)
    step = make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, coord, N, KT, NLEV,
        p_floor=100.0, dt=DT)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV)
    ub = expand_corners_to_blocks(u_d, KT, NL)
    vb = expand_corners_to_blocks(v_d, KT, NL)
    q = jnp.zeros((6, N, N, NLEV, 3))
    with pytest.raises(ValueError, match="q_pack must be passed iff"):
        step(ub, vb, T, p_s, phis, q)


# ---------------------------------------------------------------------------
# Adapter closed loop: enter/step/exit_ vs one serial step from the SAME
# entry conversion, plus the envelope refusals new to the loop.
# ---------------------------------------------------------------------------

def test_adapter_loop_matches_serial_from_same_entry():
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        make_tiled_cc_loop,
    )

    mesh = _mesh()
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=False, zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    hs = held_suarez_init(grid, coord)

    # Serial: the SAME entry conversion the loop's enter uses, then the
    # production model.step on the resulting D-grid state.
    u_d0, v_d0 = center_to_dgrid_vector(hs.u.data, hs.v.data, model.cdgrid)
    ref = _fv3_state(u_d0, v_d0, hs.T.data, hs.p_s.data, hs.phis.data)
    for _ in range(N_STEPS):
        ref = model.step(ref, DT)

    enter, step, exit_ = make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)
    step_jit = jax.jit(step)          # see the core test's jit note
    blk = enter(hs)
    for _ in range(N_STEPS):
        blk = step_jit(blk)

    # ABS bounds in the SHIPPED adapter gate's class (TILED_PARITY_ATOL
    # rationale, test_tiled_cc_step_adapter): on production-magnitude HS
    # states the tiled step differs from serial by a PRE-EXISTING
    # face-corner wind term of O(1e-6 abs) (level-decaying, step-constant;
    # bit-identity of blocked-vs-stage above proves the loop adds nothing).
    # Measured here @3 steps: u 1.8e-6 / T 2.9e-8 / p_s 1.2e-5 abs.
    def _abs(t, g):
        return float(np.max(np.abs(np.asarray(t) - np.asarray(g))))

    blk_u = np.asarray(blk["u_d"])
    ref_u = np.asarray(ref.u_d.data)
    d_u = 0.0
    b = NL + 1
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                gg = ref_u[f, ti * NL: ti * NL + b, tj * NL: tj * NL + b]
                tt = blk_u[f, ti * b:(ti + 1) * b, tj * b:(tj + 1) * b]
                d_u = max(d_u, float(np.max(np.abs(tt - gg))))
    d_T = _abs(blk["T"], ref.T.data)
    d_ps = _abs(blk["p_s"], ref.p_s.data)
    assert d_u < 2e-5, f"u_d abs {d_u:.3e}"
    assert d_T < 1e-4, f"T abs {d_T:.3e}"
    assert d_ps < 0.06, f"p_s abs {d_ps:.3e}"

    # exit_ reassembles a cc HydrostaticState (I/O path smoke).
    out = exit_(blk, hs)
    assert out.u.data.shape == hs.u.data.shape
    assert bool(jnp.all(jnp.isfinite(out.u.data)))


def test_adapter_loop_envelope_refusals():
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
        make_tiled_cc_loop,
    )

    mesh = _mesh()
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)

    # zero_mean per-stage active (fixer off) is outside the blocked base cut.
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=True)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    with pytest.raises(NotImplementedError, match="zero_mean_ps_tendency"):
        make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)

    # Ray_fast post-step damping refuses (new shared-envelope hardening).
    cfg = CDGridPrimitiveEquationConfig(rf_tau_days=10.0)
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    with pytest.raises(NotImplementedError, match="Ray_fast"):
        make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)

"""np24 gate: the TILED operator-split step + the tile-aware carry shard
(``driver/tiled_operator_split_step.py``).

Mirrors the lat-band lane's "2b" staging (sharded_operator_split_step): the
physics is a SHAPE-AGNOSTIC column-local MOCK ``step_unified`` (the real
PhysicsPipeline is decomposition-invariant per column, so the mock pins the
LAYOUT/composition contract; the real-pipeline end-to-end is the driver
integration follow-up).  The serial reference composes the SAME serial
primitives the compiled cube ``_single_step`` uses: cc->D-grid ->
base-cut SSP-RK3 dynamics -> corner->cc -> ``fix_ps_mass_target`` ->
``split_physics_single_rank`` -> ``finalize_split_step``.

Also gates: pack/unpack round-trip of the flattened per-column carry
leaves (the tile-aware PhysicsState shard), the moisture fixer's tiled
psum (conservation), and the envelope refusals.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.core.conservation import fix_ps_mass_target, global_area_sum
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
)
from legoesm.core.operators_cdgrid import (
    center_to_dgrid_vector, interp_corner_to_center)
from legoesm.driver.compiled_segments import (
    SegmentCarry, SegmentForcing, build_operator_split_statics,
    split_physics_single_rank, finalize_split_step, pack_carry)
from legoesm.driver.sharded_operator_split_step import need_rad_and_time
from legoesm.driver.tiled_operator_split_step import (
    make_tiled_operator_split_step, pack_carry_tiled,
    shard_tiled_split_carry, unpack_carry_tiled)

N, NLEV, KT = 8, 4, 2
NCOL = 6 * N * N
DT = 60.0
N_STEPS = 2
START_DAY = 0.0
# Cadence 1 is the ONLY value this lane accepts: build_tile_step_unified pins
# static_need_rad=True, which discards the per-step predicate, so >1 is refused
# by make_tiled_operator_split_step (see
# test_tiled_operator_split_refuses_rad_update_steps_above_one).  This was 2,
# which now raises -- and never exercised a held-radiation cadence anyway, since
# the predicate was being thrown away.
RAD_UPDATE_STEPS = 1


def _mesh():
    from jax.sharding import Mesh
    if len(jax.devices()) < 6 * KT * KT:
        pytest.skip(f"needs {6 * KT * KT} devices "
                    f"(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
    set_halo_backend("local")
    dev = np.array(jax.devices()[: 6 * KT * KT]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


def _mock_step_unified(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                       sst, sic, lat, lon, doy, sod, dt, *args, **kwargs):
    """Shape-agnostic, column-local, DETERMINISTIC physics mock.

    Depends on ``lat`` (validates the per-tile lat slicing) and ``need_rad``
    (validates the radiation cadence plumbing).  Positional layout matches
    the split_physics_single_rank call; held fields ride ``args[4:10]``.
    """
    held = args[4:10]
    o3 = args[2]
    lat_b = lat[..., None]
    # ``sst`` (surface-native) may arrive grid-shaped (tile) or flat
    # (serial production layout) — normalize against T's leading dims,
    # exactly how the adapter-flattened consumers see it.
    sst_b = sst.reshape(T.shape[:-1])[..., None]
    # ``o3_vmr`` is a COLUMN-format forcing leaf: the pipeline's radiation
    # consumers read it at (ncol, nlev) WITHOUT an adapter flatten (codex
    # round-3 High) — assert the tile body re-flattened it.
    assert o3.ndim == 2, f"o3_vmr must be column-format, got {o3.shape}"
    o3_g = o3.reshape(T.shape)
    phys_out = types.SimpleNamespace(
        dT_dt=(1e-4 * (300.0 - T) + 1e-2 * q_v * jnp.cos(lat_b)
               + 1e-6 * sst_b + 1e-3 * o3_g),
        dq_v_dt=-1e-3 * q_v + 1e-9 * jnp.sin(lat_b),
        dq_c_dt=5e-4 * q_v - 2e-4 * q_c,
        dq_r_dt=2e-4 * q_c - 1e-4 * q_r,
        dq_i_dt=None, dq_s_dt=None, dq_g_dt=None,
        dN_c_dt=None, dN_r_dt=None, dN_i_dt=None,
        conv_prog=0.9 * conv_prog,
        precip=1e-3 * jnp.abs(q_r[..., -1]),
        shflx=1e-2 * (T[..., -1] - 288.0),
        lhflx=None,
        w_land=None, snow=None,
        tke=None, qke=None, gwd_spectrum=None,
    )
    # Held refresh only when need_rad fires (mirrors step_unified's cond).
    held_new = tuple(
        jnp.where(need_rad, h + (i + 1.0), h) for i, h in enumerate(held))
    return phys_out, held_new


def _inputs(seed=3):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.1 * rng.standard_normal((6, N, N, NLEV)))
    v = jnp.asarray(0.1 * rng.standard_normal((6, N, N, NLEV)))
    T = jnp.asarray(250.0 + 2.0 * rng.standard_normal((6, N, N, NLEV)))
    p_s = jnp.asarray(1.0e5 + 50.0 * rng.standard_normal((6, N, N)))
    phis = jnp.asarray(1.0e2 * rng.standard_normal((6, N, N)))
    q_v = jnp.asarray(np.abs(5e-3 + 1e-3 * rng.standard_normal(
        (6, N, N, NLEV))))
    q_c = 0.1 * q_v
    q_r = 0.01 * q_v
    return u, v, T, p_s, phis, q_v, q_c, q_r


def _global_moisture(q_v, p_s, coord, grid):
    from legoesm.core.conservation import compute_global_moisture
    return compute_global_moisture(q_v, p_s, coord.dsigma, grid)


def _cc_state(u, v, T, p_s, phis):
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    from legoesm.core.state import HydrostaticState
    return HydrostaticState(
        u=Field(data=u, name="u", dims=d3, units="m/s"),
        v=Field(data=v, name="v", dims=d3, units="m/s"),
        T=Field(data=T, name="T", dims=d3, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=phis, name="phis", dims=d2, units="m^2/s^2"))


def _build(seed=3):
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig()   # defaults incl. the sponge
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    u, v, T, p_s, phis, q_v, q_c, q_r = _inputs(seed)
    state = _cc_state(u, v, T, p_s, phis)
    carry = pack_carry(
        state, q_v, q_c, q_r,
        conv_prog=jnp.zeros((NCOL,)),          # FLAT — exercises the pack
        held_dT_rad=jnp.zeros_like(T),
        held_sw_net_sfc=jnp.zeros_like(p_s),
        held_lw_net_sfc=jnp.zeros_like(p_s),
        held_sw_up_toa=jnp.zeros_like(p_s),
        held_lw_up_toa=jnp.zeros_like(p_s),
        held_sw_down_toa=jnp.zeros_like(p_s),
        step_index=0,
        target_mass=global_area_sum(p_s, grid),
        target_moisture=_global_moisture(q_v, p_s, coord, grid))

    # FLAT (ncol,) sst — the driver's per-column forcing layout; the tiled
    # factory must pack it grid-shaped (codex round-2), and the SERIAL
    # reference consumes the SAME packed layout (the column physics
    # flattens either way; the pack is layout-only).
    rng_f = np.random.default_rng(seed + 100)
    sst_flat = jnp.asarray(290.0 + rng_f.standard_normal(NCOL))
    o3_flat = jnp.asarray(np.abs(
        1e-6 + 1e-7 * rng_f.standard_normal((NCOL, NLEV))))
    forcing = SegmentForcing(**{
        nm: (sst_flat if nm == "sst"
             else o3_flat if nm == "o3_vmr" else None)
        for nm in SegmentForcing._fields})
    statics = build_operator_split_statics(
        step_unified=_mock_step_unified, forcing=None,
        lat=grid.grid_lat, lon=grid.grid_lon, dt=DT,
        tau_equator=None, tau_pole=None, sbm_tau_c=None, sbm_RH_ref=None,
        C_H=None, C_E=None, albedo_ice=None, albedo_ocean=None,
        ghg_vmr_override=None, hs_newtonian_relax=None,
        energy_consistent_moisture_clip=False,
        do_sat_adjust=True, fix_moisture=True,
        sigma_full=coord.sigma_full, dsigma=coord.dsigma, grid=grid,
        owned_mask=None, qv_smooth_coeff=0.0, fric_decay=0.9999,
        hyperdiffusion_3d=None)
    return model, cdgrid, coord, cfg, carry, statics, forcing


def _serial_split_step(carry, model, cdgrid, coord, cfg, statics):
    """The serial operator-split composition (the compiled cube
    ``_single_step`` semantics with the SAME base-cut dynamics the tiled
    lane runs: cc->D -> RK3 -> corner->cc, fixers externalized)."""
    u_d0, v_d0 = center_to_dgrid_vector(carry.u, carry.v, cdgrid)
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    st = FV3HydrostaticState(
        u_d=Field(data=u_d0, name="u_d", dims=d3, units="m/s"),
        v_d=Field(data=v_d0, name="v_d", dims=d3, units="m/s"),
        T=Field(data=carry.T, name="T", dims=d3, units="K"),
        p_s=Field(data=carry.p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=carry.phis, name="phis", dims=d2,
                   units="m^2/s^2"))

    def tendency_fn(s):
        tend = fv3_hydrostatic_tendencies(s, cdgrid.base, coord, cdgrid, cfg)
        return FV3HydrostaticState(
            u_d=s.u_d.replace(data=tend.du_d_dt.data),
            v_d=s.v_d.replace(data=tend.dv_d_dt.data),
            T=s.T.replace(data=tend.dT_dt.data),
            p_s=s.p_s.replace(data=tend.dp_s_dt.data),
            phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)))

    stepped = ssp_rk3_step(st, tendency_fn, DT)
    T_new = stepped.T.data
    u_new = interp_corner_to_center(stepped.u_d.data)
    v_new = interp_corner_to_center(stepped.v_d.data)
    p_s_new = fix_ps_mass_target(
        stepped.p_s.data, carry.target_mass, cdgrid.base)
    need_rad, doy, sod = need_rad_and_time(
        carry.step_index, START_DAY, DT, RAD_UPDATE_STEPS)
    lz = split_physics_single_rank(
        carry, T_new, u_new, v_new, p_s_new, need_rad, doy, sod, statics)
    return finalize_split_step(carry, lz, statics)


# ---------------------------------------------------------------------------
# Pack/shard round-trip (the tile-aware carry/PhysicsState layout)
# ---------------------------------------------------------------------------

def test_pack_unpack_carry_roundtrip():
    model, cdgrid, coord, cfg, carry, statics, forcing = _build()
    assert carry.conv_prog.shape == (NCOL,)           # flattened per-column
    packed = pack_carry_tiled(carry, N)
    assert packed.conv_prog.shape == (6, N, N)        # grid-shaped
    assert packed.T.shape == carry.T.shape            # grid-shaped untouched
    back = unpack_carry_tiled(packed, N, carry)
    for name in SegmentCarry._fields:
        a, b = getattr(back, name), getattr(carry, name)
        if a is None or not hasattr(a, "shape"):
            continue
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=name)


# ---------------------------------------------------------------------------
# np24 parity: tiled operator-split step vs the serial composition
# ---------------------------------------------------------------------------

def test_tiled_operator_split_matches_serial():
    import dataclasses

    mesh = _mesh()
    model, cdgrid, coord, cfg, carry0, statics, forcing = _build()

    # Serial reference consumes the ORIGINAL (production, flat-per-column)
    # forcing layout; the tile body re-flattens the column-format leaves,
    # so BOTH sides feed the mock identical column arrays (see the mock's
    # o3 layout assertion — the codex round-3 contract).
    statics_ref = dataclasses.replace(statics, forcing=forcing)
    ref = carry0
    for _ in range(N_STEPS):
        ref = _serial_split_step(ref, model, cdgrid, coord, cfg,
                                 statics_ref)

    step = make_tiled_operator_split_step(
        model, mesh, statics, fix_mass=True,
        rad_update_steps=RAD_UPDATE_STEPS, start_day=START_DAY, kt=KT)
    tc = shard_tiled_split_carry(carry0, mesh, N)
    for _ in range(N_STEPS):
        tc = step(tc, forcing)
    got = unpack_carry_tiled(tc, N, carry0)

    def _abs(name):
        a = np.asarray(getattr(got, name))
        b = np.asarray(getattr(ref, name))
        assert a.shape == b.shape, name
        return float(np.max(np.abs(a - b)))

    # Dynamics-affected fields: the documented tiled-vs-serial face-corner
    # wind class (O(1e-6 abs), the blocked-loop gates' bound) propagated
    # through the mock physics.  Everything else is column/pointwise and
    # tracks to fp-reorder.
    assert _abs("u") < 2e-5
    assert _abs("v") < 2e-5
    assert _abs("T") < 1e-4
    assert _abs("p_s") < 0.06
    for nm in ("q_v", "q_c", "q_r"):
        assert _abs(nm) < 1e-9, nm
    # Per-column physics carry + held-radiation cadence: mock updates are
    # exact functions of carry/need_rad — bitwise-comparable classes.
    assert _abs("conv_prog") < 1e-12
    for nm in ("held_dT_rad", "held_sw_up_toa", "held_lw_up_toa"):
        assert _abs(nm) < 1e-12, nm
    assert int(got.step_index) == int(ref.step_index) == N_STEPS
    # Accumulators (precip from q_r; flux from T) stay in the same classes.
    assert _abs("precip_accum") < 1e-12
    assert _abs("shflx_accum") < 1e-6

    # Conservation: the tiled moisture fixer (tile-psum global_area_sum)
    # held the target — global moisture equals the serial run's.
    from legoesm.core.conservation import compute_global_moisture
    m_t = float(compute_global_moisture(
        got.q_v, got.p_s, coord.dsigma, cdgrid.base))
    m_s = float(compute_global_moisture(
        ref.q_v, ref.p_s, coord.dsigma, cdgrid.base))
    assert abs(m_t - m_s) / abs(m_s) < 1e-9


def test_tiled_psum_branch_requires_explicit_scope():
    """The tiled-mesh psum branch must NOT fire from an armed mesh alone —
    a reduction outside the tiled shard_map body (writers, diagnostics)
    would emit an out-of-scope psum (codex).  Without the scope the helper
    falls through (returns None -> serial logic)."""
    from legoesm.core.conservation import (
        _spmd_lat_psum_or_none, tiled_reduction_scope)
    from legoesm.grids.halo import (
        get_halo_backend, get_mpi_topology, get_spmd_mesh,
        set_halo_backend, set_spmd_mesh)

    mesh = _mesh()
    prev = (get_halo_backend(), get_mpi_topology(), get_spmd_mesh())
    try:
        set_halo_backend("spmd")
        set_spmd_mesh(mesh)
        # Armed mesh, NO scope -> fall through (None), never psum.
        assert _spmd_lat_psum_or_none([jnp.asarray(1.0)]) is None
        # Inside the scope (and inside a shard_map, where the axes exist)
        # the branch fires — exercised end-to-end by the parity test's
        # moisture fixer; here we only pin the outside-scope inertness.
        with tiled_reduction_scope():
            pass
        assert _spmd_lat_psum_or_none([jnp.asarray(1.0)]) is None
    finally:
        set_spmd_mesh(prev[2])
        set_halo_backend(prev[0], prev[1])


def test_tiled_operator_split_refuses_land_ml():
    mesh = _mesh()
    model, cdgrid, coord, cfg, carry0, statics, forcing = _build()
    carry_land = carry0._replace(land_ml={"T_soil": jnp.zeros((NCOL, 4))})
    with pytest.raises(NotImplementedError, match="land_ml"):
        shard_tiled_split_carry(carry_land, mesh, N)
    step = make_tiled_operator_split_step(
        model, mesh, statics, fix_mass=True,
        rad_update_steps=1, start_day=0.0, kt=KT)
    with pytest.raises(NotImplementedError, match="land_ml"):
        step(carry_land, forcing)


def test_tiled_operator_split_envelope_refusals():
    mesh = _mesh()
    model, cdgrid, coord, cfg, carry0, statics, forcing = _build()
    import dataclasses
    with pytest.raises(NotImplementedError, match="qv_smooth_coeff"):
        make_tiled_operator_split_step(
            model, mesh,
            dataclasses.replace(statics, qv_smooth_coeff=1e4),
            fix_mass=True, rad_update_steps=1, start_day=0.0, kt=KT)
    with pytest.raises(NotImplementedError, match="owned_mask"):
        make_tiled_operator_split_step(
            model, mesh,
            dataclasses.replace(statics,
                                owned_mask=jnp.ones((6,))),
            fix_mass=True, rad_update_steps=1, start_day=0.0, kt=KT)


def test_tiled_operator_split_refuses_rad_update_steps_above_one():
    """``rad_update_steps > 1`` must be refused, not silently ignored.

    The step body computes the cadence predicate (``need_rad_and_time``) and
    passes it to ``statics.step_unified``, but ``build_tile_step_unified`` pins
    ``static_need_rad=True``, which DELETES the predicate and always takes the
    radiation branch -- so the knob ran a denser cadence than configured, with
    no error. Coupled tiled runs raise at the driver dispatch before reaching
    here, so the silent path was the UNCOUPLED tiled cube run.

    Non-vacuity: the identical call at ``rad_update_steps=1`` builds fine (the
    sibling tests above all use it), so the refusal is keyed on the knob and
    not on the fixture being unbuildable.
    """
    mesh = _mesh()
    model, cdgrid, coord, cfg, carry0, statics, forcing = _build()
    with pytest.raises(NotImplementedError, match="rad_update_steps"):
        make_tiled_operator_split_step(
            model, mesh, statics,
            fix_mass=True, rad_update_steps=2, start_day=0.0, kt=KT)
    # Same call, cadence 1 -> builds.
    assert make_tiled_operator_split_step(
        model, mesh, statics,
        fix_mass=True, rad_update_steps=1, start_day=0.0, kt=KT) is not None


def test_build_tile_step_unified_refuses_column_sharding():
    """shard_radiation_columns builds a GLOBAL column mesh against the
    pipeline ncol — nested inside the per-tile shard_map it is wrong or
    a false refusal (codex round-3 Medium)."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.tiled_operator_split_step import (
        build_tile_step_unified,
    )

    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    cfg = ExperimentConfig()
    cfg = cfg._replace(
        grid=cfg.grid._replace(grid_type="cubed_sphere"),
        radiation="gray", convection="none", turbulence="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        shard_radiation_columns=True)
    with pytest.raises(NotImplementedError,
                       match="shard_radiation_columns"):
        build_tile_step_unified(grid, coord, cfg, KT)

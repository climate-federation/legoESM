"""Resolved-wind moisture advection through the compiled segment (issue #771).

The cube AMIP path column-locked its moisture: ``_rebuild_state`` built the
dycore state WITHOUT tracers, so ``primitive_eq_cdgrid``'s tracer transport
(``if state.tracers:``) never ran — moisture moved only by physics column
tendencies + hyperdiffusion smoothing, and the local wet drift fed the
day-150/195 C48 blowup family.  These tests lock the fix at three seams:

1. ``_rebuild_state(advect_moisture=True)`` attaches the moisture tracers
   (and stays legacy-tracerless by default).
2. The compiled segment READS BACK the dycore-advected tracers: a mock model
   that mutates ``state.tracers`` must be visible in the segment result with
   the flag on, and invisible with it off (bit-identity for the legacy path).
3. The REAL cdgrid model advects an attached q_v blob with the resolved wind
   (and is a no-op under zero wind) — the mechanism the wiring relies on.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import (
    _rebuild_state,
    build_segment_fn,
    pack_carry,
    pack_forcing,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.physics_pipeline import PhysicsOutput
from legoesm.grids.cubed_sphere import create_cubed_sphere

from legoesm import constants

N_FACES, N, NLEV = 6, 4, 3
DT = 600.0
_GRID = create_cubed_sphere(N)


def _f3(name, fill):
    return Field(jnp.full((N_FACES, N, N, NLEV), fill, dtype=jnp.float32),
                 name=name, dims=("face", "x", "y", "level"), units="K")


def _f2(name, fill):
    return Field(jnp.full((N_FACES, N, N), fill, dtype=jnp.float32),
                 name=name, dims=("face", "x", "y"), units="Pa")


def _state():
    return HydrostaticState(u=_f3("u", 0.5), v=_f3("v", -0.5),
                            T=_f3("T", 280.0),
                            p_s=_f2("p_s", 101325.0), phis=_f2("phis", 0.0))


def _carry():
    s3, s2 = (N_FACES, N, N, NLEV), (N_FACES, N, N)
    return pack_carry(
        _state(),
        q_v=jnp.full(s3, 0.01), q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        held_dT_rad=jnp.zeros(s3),
        held_sw_net_sfc=jnp.zeros(s2), held_lw_net_sfc=jnp.zeros(s2),
        held_sw_up_toa=jnp.zeros(s2), held_lw_up_toa=jnp.zeros(s2),
        held_sw_down_toa=jnp.zeros(s2), step_index=0,
    )


class _TracerMutatingModel:
    """Mock dycore that DOUBLES every attached tracer — a visible marker for
    whether the segment reads the dycore-advected tracers back (#771)."""

    _state_type = HydrostaticState

    def step(self, state, dt):
        if state.tracers:
            state = state._replace(tracers={
                k: f.replace(data=2.0 * f.data)
                for k, f in state.tracers.items()})
        return state


def _mock_step_unified(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                       sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                       solar_weights, s_0, o3_vmr, aerosol_od,
                       held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                       held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                       **kwargs):
    zero3, zero2 = jnp.zeros_like(T), jnp.zeros_like(p_s)
    out = PhysicsOutput(
        dT_dt=zero3, dq_v_dt=zero3, dq_c_dt=zero3, dq_r_dt=zero3,
        precip=zero2,
        sw_net_sfc=zero2, lw_net_sfc=zero2, sw_up_toa=zero2,
        lw_up_toa=zero2, sw_down_toa=zero2,
        du_dt=zero3, dv_dt=zero3,
        dq_i_dt=zero3, dq_s_dt=zero3, dq_g_dt=zero3,
        dN_c_dt=zero3, dN_r_dt=zero3, dN_i_dt=zero3,
        conv_prog=conv_prog,
    )
    held = (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa)
    return out, held, kwargs.get("T_land")


def _segment_args(model, **over):
    args = dict(
        model=model, step_unified=_mock_step_unified, grid=_GRID,
        sigma_full=jnp.linspace(0.1, 1.0, NLEV),
        dsigma=jnp.full((NLEV,), 1.0 / NLEV), dt=DT,
        rad_update_steps=1, microphysics="none",
        fix_moisture=False, fix_mass=False,
        fric_decay=jnp.ones((NLEV,)), qv_smooth_coeff=0.0,
        lat=_GRID.lat, lon=_GRID.lon, start_day=0.0,
    )
    args.update(over)
    return args


_FORCING = pack_forcing(
    sst=jnp.full((N_FACES, N, N), 300.0), sic=jnp.zeros((N_FACES, N, N)),
    day_of_year=1.0, seconds_of_day=0.0, solar_weights=jnp.ones(14),
    s_0=constants.S_0, o3_vmr=jnp.zeros((N_FACES, N, N, NLEV)),
    aerosol_od=jnp.zeros((N_FACES, N, N)),
)


# --- seam 1: _rebuild_state -------------------------------------------------

def test_rebuild_state_attaches_moisture_tracers():
    st = _rebuild_state(_carry(), _TracerMutatingModel(),
                        advect_moisture=True)
    assert set(st.tracers) == {"q_v", "q_c", "q_r"}   # DM fields None -> absent
    np.testing.assert_allclose(np.asarray(st.tracers["q_v"].data), 0.01)


def test_rebuild_state_default_is_tracerless():
    st = _rebuild_state(_carry(), _TracerMutatingModel())
    assert st.tracers is None


# --- seam 2: the compiled segment reads the advected tracers back -----------

def test_segment_uses_dycore_advected_moisture():
    """Advect ON: the mock dycore doubles q_v -> the segment result must
    carry the doubled field (dycore transport feeds the physics update)."""
    run = build_segment_fn(**_segment_args(_TracerMutatingModel(),
                                           advect_moisture=True))
    res = run(_carry(), 1, _FORCING)
    # Level 0 (sigma=0.1) is far from saturation at 280 K, so the segment's
    # saturation adjustment leaves it untouched — the doubled value must
    # survive to the result.  (The lowest level gets sat-adjusted; that is
    # pre-existing segment behaviour, not the advection seam.)
    np.testing.assert_allclose(np.asarray(res.q_v[..., 0]), 0.02, rtol=1e-6)


def test_segment_legacy_path_ignores_dycore_tracers():
    """Advect OFF (default): the mutating mock must be invisible — moisture
    stays column-locked on the carry, byte-identical to the legacy path."""
    run = build_segment_fn(**_segment_args(_TracerMutatingModel()))
    res = run(_carry(), 1, _FORCING)
    np.testing.assert_allclose(np.asarray(res.q_v[..., 0]), 0.01, rtol=1e-6)


# --- seam 3: the real cdgrid dycore advects an attached tracer --------------

def test_cdgrid_step_advects_attached_tracer():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    n, nlev = 12, 8
    grid = create_cubed_sphere(n)
    model = CDGridPrimitiveEquationModel(grid, create_sigma_coordinate(nlev))

    def mkstate(u0):
        s3 = (6, n, n, nlev)
        x = jnp.arange(n)
        X, Y = jnp.meshgrid(x, x, indexing="ij")
        blob = 1e-3 * jnp.exp(-((X - n / 2) ** 2 + (Y - n / 2) ** 2) / 8.0)
        q = jnp.zeros(s3).at[0, :, :, -1].set(blob)
        def F(d, nm, dims, un):
            return Field(data=d, name=nm, dims=dims, units=un)
        d3, d2 = ("face", "x", "y", "level"), ("face", "x", "y")
        return HydrostaticState(
            u=F(jnp.full(s3, u0), "u", d3, "m/s"),
            v=F(jnp.zeros(s3), "v", d3, "m/s"),
            T=F(jnp.full(s3, 280.0), "T", d3, "K"),
            p_s=F(jnp.full((6, n, n), 1.0e5), "p_s", d2, "Pa"),
            phis=F(jnp.zeros((6, n, n)), "phis", d2, "m2/s2"),
            tracers={"q_v": F(q, "q_v", d3, "kg/kg")})

    # Wind on: the blob moves (advection is wired).  Wind off: bit-exact no-op.
    #
    # We deliberately do NOT assert mass conservation here: the transport is
    # ADVECTIVE form -(u·∇q), not discretely mass-conserving on the unequal-area
    # cube, and a plain jnp.mean is neither the conserved integral (∫ q·δp·dA)
    # nor area-weighted — a tight mean-tolerance check would be misleading (it
    # passes trivially because one 150 s step moves the blob << a cell).  Budget
    # closure is fix_moisture's job; a proper area+mass-weighted total-water
    # closure test belongs with the flux-form follow-up (#771).  Here we assert
    # only the wiring: the blob moved, and it stayed finite/bounded (no blow-up).
    s = mkstate(20.0)
    s1 = model.step(s, 150.0)
    dq = jnp.abs(s1.tracers["q_v"].data - s.tracers["q_v"].data)
    assert float(jnp.max(dq)) > 1e-8            # advection acted
    q1 = s1.tracers["q_v"].data
    assert bool(jnp.all(jnp.isfinite(q1)))      # no NaN/Inf blow-up
    assert float(jnp.max(q1)) < 5.0 * float(jnp.max(s.tracers["q_v"].data))

    s = mkstate(0.0)
    s1 = model.step(s, 150.0)
    # Zero wind: roundoff-only change (float32 default; ~1e-11 observed),
    # >2 orders below the wind-on signal.
    assert float(jnp.max(jnp.abs(
        s1.tracers["q_v"].data - s.tracers["q_v"].data))) < 1e-9


# --- driver gate -------------------------------------------------------------

def _fake_driver(grid_type, discretization, flag=True):
    fake = SimpleNamespace()
    fake.config = SimpleNamespace(
        moisture_advection=flag,
        grid=SimpleNamespace(grid_type=grid_type),
        dycore=SimpleNamespace(discretization=discretization),
    )
    return fake


def test_gate_on_for_cube_cdgrid():
    assert ModelDriver._moisture_advection_active(
        _fake_driver("cubed_sphere", "cdgrid")) is True


def test_gate_on_for_cube_cdgrid_aliases():
    # "centered"/"finite_volume" resolve to the cdgrid PE dycore, so they are
    # tracer-capable and must gate ON too (else an opt-in run on those aliases
    # silently drops to the legacy column-locked path, #771).
    for alias in ("centered", "finite_volume"):
        assert ModelDriver._moisture_advection_active(
            _fake_driver("cubed_sphere", alias)) is True


def test_gate_off_when_disabled_or_unsupported():
    assert ModelDriver._moisture_advection_active(
        _fake_driver("cubed_sphere", "cdgrid", flag=False)) is False
    assert ModelDriver._moisture_advection_active(
        _fake_driver("latlon", "latlon_cgrid")) is False
    # spectral on a cube is NOT tracer-capable here -> legacy path.
    assert ModelDriver._moisture_advection_active(
        _fake_driver("cubed_sphere", "spectral")) is False


def test_per_volume_number_densities_excluded_from_advection():
    # N_c/N_r are per-VOLUME [#/m^3]; advecting them with the mass-mixing-ratio
    # operator applies the wrong conservation law (#772 review), so they are
    # excluded — masses q_c/q_r still advect, the per-mass ice number N_i may.
    # This is a contract guard: if the set changes, the compiled_segments
    # readback fallback (non-advected tracers -> carry.<X>, NOT None, so a
    # double-moment opt-in run keeps its numbers column-locked instead of
    # dropping them) and the per-volume/per-mass units handling MUST be
    # revisited together.
    from legoesm.driver.compiled_segments import _ADVECTED_TRACER_NAMES
    assert "N_c" not in _ADVECTED_TRACER_NAMES
    assert "N_r" not in _ADVECTED_TRACER_NAMES
    assert {"q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_i"} <= set(
        _ADVECTED_TRACER_NAMES)

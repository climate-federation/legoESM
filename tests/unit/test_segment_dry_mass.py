"""Dry-mass convention on the compiled / training segment lanes (review
2026-10-10 F13).

Physics water now carries its mass (``segment_physics_water``: precipitation
lowers p_s, every moisture field is re-weighted) and the target-anchored fixer
holds the global DRY mass (``fix_segment_dry_mass``), the convention MPAS,
lat-lon and the duo lanes already use.  Before, physics water was applied at
fixed p_s and the fixer held the TOTAL p_s integral, so precipitated water
stayed behind as dry air.

Checked here through the real ``build_segment_fn`` scan (mock dynamics +
mock physics with a known water sink), x64:

* per column, one step without the fixer: the column DRY mass is unchanged
  and the column water changes by exactly the physics water tendency;
* globally, several steps with the fixer on and a dynamics that drifts the
  mass: the global dry mass returns to the target every step.

Each check carries its own non-vacuity line: the legacy update (fixed p_s,
total-mass target) violates it by many orders of magnitude more than the
tolerance.
"""

from __future__ import annotations

import functools
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.conservation import (
    SigmaLayers, compute_global_moisture, dycore_column_energy,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import (
    build_segment_fn, fix_segment_dry_mass, pack_carry, segment_dry_mass,
)
from legoesm.driver.physics_pipeline import PhysicsOutput
from tests.unit.test_compiled_segments import _FORCING, _GRID

jax.config.update("jax_enable_x64", True)

N, NLEV, DT = 4, 3, 600.0
DSIGMA = jnp.asarray([0.2, 0.3, 0.5])
SIGMA_FULL = jnp.asarray([0.1, 0.35, 0.75])
K_SINK = 1.0e-5            # vapour removal rate [1/s]
RAIN_FRAC = 0.5            # share of the removed vapour that stays as rain


def _state(seed=0, T=300.0):
    """T=300 K keeps every level subsaturated (no saturation adjustment);
    T=270 K makes every lowest level supersaturated (the tail's adjustment
    fires)."""
    rng = np.random.default_rng(seed)
    s3, s2 = (6, N, N, NLEV), (6, N, N)

    def f(a):
        return Field(jnp.asarray(a), name="x")
    return HydrostaticState(
        u=f(np.zeros(s3)), v=f(np.zeros(s3)), T=f(np.full(s3, T)),
        p_s=f(101325.0 + 2000.0 * rng.standard_normal(s2)),
        phis=f(np.zeros(s2)),
    ), jnp.asarray(rng.uniform(0.005, 0.015, s3))


class _DriftModel:
    """Dynamics stand-in: leaves the column untouched except a uniform
    relative p_s drift (a dycore mass error the fixer must undo)."""
    _state_type = HydrostaticState

    def __init__(self, drift):
        self._drift = drift

    def step(self, state, dt):
        return state._replace(
            p_s=state.p_s.replace(data=state.p_s.data * (1.0 + self._drift)))


def _sink_physics(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                  sst, sic, lat, lon, day_of_year, seconds_of_day, dt,
                  solar_weights, s_0, o3_vmr, aerosol_od,
                  held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                  held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                  k_sink=K_SINK, **kwargs):
    """Removes vapour at ``k_sink``; RAIN_FRAC of it stays as rain, the rest
    precipitates: a column water sink of (1-RAIN_FRAC)*k_sink*q_v."""
    z3, z2 = jnp.zeros_like(T), jnp.zeros_like(p_s)
    dqv = -k_sink * q_v
    out = PhysicsOutput(
        dT_dt=z3, dq_v_dt=dqv, dq_c_dt=z3, dq_r_dt=-RAIN_FRAC * dqv,
        precip=z2, sw_net_sfc=z2, lw_net_sfc=z2, sw_up_toa=z2, lw_up_toa=z2,
        sw_down_toa=z2, du_dt=z3, dv_dt=z3, dq_i_dt=z3, dq_s_dt=z3,
        dq_g_dt=z3, dN_c_dt=z3, dN_r_dt=z3, dN_i_dt=z3, conv_prog=conv_prog)
    held = (held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa)
    return out, held, kwargs.get("T_land")


def _run(n_steps, *, fix_mass, drift, seed=0, T=300.0, k_sink=K_SINK,
         target_moisture=None, model_cls=None, **seg_kw):
    state, q_v = _state(seed, T)
    z3, z2 = jnp.zeros_like(q_v), jnp.zeros_like(state.p_s.data)
    moist = {"q_v": q_v, "q_c": z3, "q_r": z3}
    target = segment_dry_mass(state.p_s.data, moist, _GRID, DSIGMA)
    carry = pack_carry(
        state, q_v, z3, z3, held_dT_rad=z3, held_sw_net_sfc=z2,
        held_lw_net_sfc=z2, held_sw_up_toa=z2, held_lw_up_toa=z2,
        held_sw_down_toa=z2, step_index=0,
        target_mass=target if fix_mass else None,
        target_moisture=target_moisture)
    run = build_segment_fn(
        model=(model_cls or _DriftModel)(drift),
        step_unified=functools.partial(_sink_physics, k_sink=k_sink),
        grid=_GRID,
        sigma_full=SIGMA_FULL, dsigma=DSIGMA, dt=DT, rad_update_steps=1,
        microphysics="none", fix_moisture=target_moisture is not None,
        fix_mass=fix_mass,
        fric_decay=jnp.ones((NLEV,)), qv_smooth_coeff=0.0,
        lat=_GRID.lat, lon=_GRID.lon, start_day=0.0, **seg_kw)
    return state, q_v, target, run.raw(carry, n_steps, _FORCING)


def _col(p_s, q):
    """(column dry pressure, column water pressure) on p_s*dsigma layers."""
    w = p_s * jnp.sum(DSIGMA * q, axis=-1)
    return p_s - w, w


def test_physics_water_leaves_column_dry_mass_untouched():
    state, q_v, _, out = _run(1, fix_mass=False, drift=0.0)
    ps0 = state.p_s.data
    dry0, w0 = _col(ps0, q_v)
    dry1, w1 = _col(out.p_s, out.q_v + out.q_c + out.q_r)
    np.testing.assert_allclose(np.asarray(dry1), np.asarray(dry0),
                               rtol=1e-14, atol=0.0)
    # column water change == the physics water tendency on the pre-physics
    # layer masses (the mass rides p_s): sink = (1-RAIN_FRAC)*K*q_v
    src = -DT * (1.0 - RAIN_FRAC) * K_SINK * ps0 * jnp.sum(DSIGMA * q_v, -1)
    np.testing.assert_allclose(np.asarray(w1 - w0), np.asarray(src),
                               rtol=1e-10, atol=0.0)
    # p_s fell by exactly the precipitated water
    np.testing.assert_allclose(np.asarray(out.p_s - ps0), np.asarray(src),
                               rtol=1e-10, atol=0.0)
    # NON-VACUITY: the legacy update (q += dt*dq at fixed p_s) turns the
    # precipitated water into dry air -- many orders above the tolerance.
    dry_legacy, _ = _col(ps0, q_v * (1.0 - DT * K_SINK)
                         + RAIN_FRAC * DT * K_SINK * q_v)
    rel = float(jnp.max(jnp.abs(dry_legacy - dry0) / dry0))
    assert rel > 1e-6, rel


def test_saturation_adjustment_rain_leaves_column_dry_mass_untouched():
    """The segment tail's saturation adjustment (microphysics='none') rains
    the supersaturated excess out WITH its mass: p_s falls by the column's
    excess and the column dry mass is unchanged."""
    state, q_v, _, out = _run(1, fix_mass=False, drift=0.0, T=270.0,
                              k_sink=0.0)
    ps0 = state.p_s.data
    dry0, w0 = _col(ps0, q_v)
    dry1, w1 = _col(out.p_s, out.q_v + out.q_c + out.q_r)
    rained = w0 - w1
    assert float(jnp.min(rained)) > 1.0, "the adjustment must fire (Pa)"
    np.testing.assert_allclose(np.asarray(dry1), np.asarray(dry0),
                               rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(np.asarray(ps0 - out.p_s), np.asarray(rained),
                               rtol=1e-10, atol=0.0)
    # column water budget closes: the rain is booked as precipitation
    # (kg/m2; the mock physics' own precip is zero)
    np.testing.assert_allclose(np.asarray(out.precip_accum * constants.g),
                               np.asarray(rained), rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("hard_floor", [False, True])
def test_positivity_floor_water_leaves_column_dry_mass_untouched(hard_floor):
    """A sink 1.5x the vapour drives every level negative: the column
    borrow (net-negative column) and the opt-in hard floor both CREATE
    vapour; it now arrives with its mass, so the column dry mass holds."""
    state, q_v, _, out = _run(
        1, fix_mass=False, drift=0.0, k_sink=1.5 / DT,
        energy_consistent_moisture_clip=hard_floor)
    dry0, w0 = _col(state.p_s.data, q_v)
    dry1, w1 = _col(out.p_s, out.q_v + out.q_c + out.q_r)
    np.testing.assert_allclose(np.asarray(dry1), np.asarray(dry0),
                               rtol=1e-14, atol=0.0)
    # NON-VACUITY: the raw update went negative and the floor clamped it
    assert float(jnp.max(jnp.abs(out.q_v))) < 1e-12


def test_moisture_fixer_water_leaves_column_dry_mass_untouched():
    """The tail's vapour fixer (fix_moisture) adds its water WITH its mass:
    the global vapour reaches the fixer's target, the column dry mass is
    unchanged (at fixed p_s the added water was booked as dry air)."""
    state, q_v = _state(4)
    tgt = 1.02 * float(compute_global_moisture(
        q_v, state.p_s.data, DSIGMA, _GRID))
    _, q_v0, _, out = _run(1, fix_mass=False, drift=0.0, seed=4,
                           target_moisture=jnp.asarray(tgt))
    dry0, _ = _col(state.p_s.data, q_v0)
    dry1, _ = _col(out.p_s, out.q_v + out.q_c + out.q_r)
    np.testing.assert_allclose(np.asarray(dry1), np.asarray(dry0),
                               rtol=1e-14, atol=0.0)
    got = float(compute_global_moisture(out.q_v, out.p_s, DSIGMA, _GRID))
    np.testing.assert_allclose(got, tgt, rtol=1e-12)
    # NON-VACUITY: the fixer really added water (>1% of the vapour)
    assert float(jnp.max(out.p_s - state.p_s.data)) > 1.0


def test_fixer_holds_global_dry_mass_through_the_segment():
    n, drift = 3, 2.0e-4
    state, q_v, target, out = _run(n, fix_mass=True, drift=drift, T=270.0)
    moist = {"q_v": out.q_v, "q_c": out.q_c, "q_r": out.q_r}
    got = segment_dry_mass(out.p_s, moist, _GRID, DSIGMA)
    # The fixer is exact up to its divisor: the cube grid's total area is a
    # storage-dtype (fp32) sum while the mass integral sums fp64 areas, so
    # each correction lands short by that relative mismatch (shared with the
    # total-mass fixer this replaced).  Bound = n steps x drift x mismatch.
    a64 = float(jnp.sum(_GRID.area.astype(jnp.float64)))
    mismatch = abs(float(_GRID.grid_total_area) - a64) / a64
    bound = 2.0 * n * drift * mismatch + 1e-13
    rel = abs(float(got) - float(target)) / float(target)
    assert rel <= bound, (rel, bound)
    # NON-VACUITY: the TOTAL-mass anchor this replaced holds sum(p_s*area),
    # so the precipitated water (several Pa/step here) would stay as dry air
    area = _GRID.area
    sink = float(jnp.sum(area * DT * n * (1.0 - RAIN_FRAC) * K_SINK
                         * state.p_s.data * jnp.sum(DSIGMA * q_v, -1)))
    assert sink / float(target) > 1e-6, sink / float(target)


def test_dry_fixer_moves_only_dry_air():
    """The fixer alone: the global dry mass lands on the target and every
    column's water mass (p_s * sum dsigma q per species) is unchanged."""
    state, q_v = _state(2)
    q_c = 0.1 * q_v
    moist = {"q_v": q_v, "q_c": q_c, "q_r": None}
    target = segment_dry_mass(state.p_s.data, moist, _GRID, DSIGMA)
    ps_drift = state.p_s.data * (1.0 + 3.0e-4)
    ps1, m1 = fix_segment_dry_mass(ps_drift, moist, target, _GRID, DSIGMA)
    assert m1["q_r"] is None
    for k in ("q_v", "q_c"):
        np.testing.assert_allclose(
            np.asarray(ps1 * jnp.sum(DSIGMA * m1[k], -1)),
            np.asarray(ps_drift * jnp.sum(DSIGMA * moist[k], -1)),
            rtol=1e-14, atol=0.0)
    a64 = float(jnp.sum(_GRID.area.astype(jnp.float64)))
    mismatch = abs(float(_GRID.grid_total_area) - a64) / a64
    rel = abs(float(segment_dry_mass(ps1, m1, _GRID, DSIGMA)) - float(target))
    assert rel / float(target) <= 2.0 * 3.0e-4 * mismatch + 1e-13
    # NON-VACUITY: the shift is real (several Pa of dry air removed)
    assert float(jnp.max(jnp.abs(ps1 - ps_drift))) > 1.0


def test_dry_mass_target_is_differentiable():
    """The fixer sits inside the training segment's grad: d(final p_s)/d(q_v)
    flows through the dry target and the re-weighting, finite and nonzero."""
    state, q_v = _state(1)
    z3, z2 = jnp.zeros_like(q_v), jnp.zeros_like(state.p_s.data)

    def loss(qv):
        moist = {"q_v": qv, "q_c": z3, "q_r": z3}
        tgt = segment_dry_mass(state.p_s.data, moist, _GRID, DSIGMA)
        carry = pack_carry(
            state, qv, z3, z3, held_dT_rad=z3, held_sw_net_sfc=z2,
            held_lw_net_sfc=z2, held_sw_up_toa=z2, held_lw_up_toa=z2,
            held_sw_down_toa=z2, step_index=0, target_mass=tgt)
        run = build_segment_fn(
            model=_DriftModel(1.0e-4), step_unified=_sink_physics,
            grid=_GRID, sigma_full=SIGMA_FULL, dsigma=DSIGMA, dt=DT,
            rad_update_steps=1, microphysics="none", fix_moisture=False,
            fix_mass=True, fric_decay=jnp.ones((NLEV,)), qv_smooth_coeff=0.0,
            lat=_GRID.lat, lon=_GRID.lon, start_day=0.0)
        return jnp.sum(run.raw(carry, 2, _FORCING).p_s)

    g = jax.grad(loss)(q_v)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


class _EnergyFixerDriftModel(_DriftModel):
    """The drift dycore with the energy fixer on (its own window closed; the
    mass drift stands for what it leaves): ``config`` / ``sigma_coord`` are
    what the segment's mass-fixer energy closure reads."""
    config = namedtuple("Cfg", "energy_fixer")(True)
    sigma_coord = SigmaLayers(DSIGMA)


def test_segment_mass_fixer_returns_its_energy_as_heat():
    """The segment's dry-mass fixer runs after the dycore step, OUTSIDE its
    energy window (review 2026-10-14 F15 V8): with the dycore fixer on, the
    step's energy equals the post-dynamics (drifted) state's -- the mass the
    fixer removes takes no energy with it."""
    drift = 2.0e-4
    state, _q, _t, out = _run(1, fix_mass=True, drift=drift, k_sink=0.0,
                              model_cls=_EnergyFixerDriftModel)
    sig, area = _EnergyFixerDriftModel.sigma_coord, _GRID.area.astype(
        jnp.float64)

    def E(T, p_s):
        ph = sig.pressure_at_half(p_s)
        e = dycore_column_energy(T, jnp.zeros_like(T),
                                 ph[..., 1:] - ph[..., :-1],
                                 jnp.zeros_like(p_s), p_s)
        return float(jnp.sum(e * area))

    e_dyn = E(state.T.data, state.p_s.data * (1.0 + drift))
    assert abs(E(out.T, out.p_s) - e_dyn) / e_dyn < 1e-12
    # NON-VACUITY: the fixer really removed the drift (~20 Pa of air)
    assert float(jnp.mean(state.p_s.data * (1.0 + drift) - out.p_s)) > 10.0

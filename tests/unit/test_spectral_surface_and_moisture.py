"""The spectral lane's surface: humidity into turbulence, a prescribed skin
temperature and the scene's own calendar into the rollout.

Three defects this locks down, all measured on the WeatherBench classical
training arm (2026-08-18):

1. The spectral turbulence bridge handed the scheme a hard-coded ZERO
   humidity column and threw the scheme's moisture tendency away, so the
   boundary layer moved momentum and heat but no water, and the closure's
   moist buoyancy / PDF cloud fraction were computed on a bone-dry column.
   The hydrostatic, MPAS and cubed-sphere bridges never had this — only
   the spectral one.
2. ``spectral_rollout`` REFUSED a forcing dict whenever radiation was
   sub-cycled, which is every classical training configuration.  With no
   forcing the bulk-flux surface temperature falls back to the lowest
   model level's own air temperature, i.e. the sensible heat flux is
   identically zero and the run has no surface energy exchange.
3. ...and radiation then ran on its module-default calendar — a spring
   equinox noon sun for every scene, whatever the scene's real date.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.integration import (
    make_turbulence_physics,
)
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.training.neural_gcm_spectral import (
    carry_to_spectral_state, spectral_rollout, spectral_state_to_carry,
)

from tests.unit.test_spectral_carry_tracer_set import _carry


def _state(n_max=5, nlev=4, q_v=1.0e-3):
    """A sheared, stably stratified column with a uniform humidity.

    The bare fixture is at rest and isothermal, which gives every bulk-flux
    scheme a vanishing exchange coefficient — the tendencies would then be
    rounding noise and no assertion on them could mean anything.
    """
    grid = create_gaussian_grid(n_max)
    sigma = create_sigma_coordinate(nlev)
    carry = _carry(grid.n_lat, grid.n_lon, nlev=nlev, extras=False)
    lapse = jnp.asarray([250.0, 260.0, 272.0, 285.0][:nlev])
    carry = carry._replace(
        u=jnp.full_like(carry.u, 10.0),
        T=jnp.broadcast_to(lapse, carry.T.shape).astype(carry.T.dtype),
        q_v=jnp.full_like(carry.q_v, q_v),
    )
    state = carry_to_spectral_state(carry, grid)
    return state, grid, sigma


# --------------------------------------------------------------------------
# 1. humidity in, moisture tendency out
# --------------------------------------------------------------------------

def _turb_tend(q_v):
    state, grid, sigma = _state(q_v=q_v)
    fn = make_turbulence_physics(
        TurbulenceConfig(scheme="louis"), "spectral_pe", dt=600.0)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    nlev = int(jnp.shape(sigma.sigma_full)[0])
    ps = init_physics_state(ncol, nlev, _dummy_cfg())
    out = fn(state, grid, sigma, phys_state=ps)
    tend = out[0] if isinstance(out, tuple) else out
    return tend


def _dummy_cfg():
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.gravity_wave_drag import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def test_turbulence_exports_a_moisture_tendency():
    """The bridge must return a q_v tendency; it used to return none at all,
    so surface evaporation and boundary-layer moisture mixing never reached
    the dycore."""
    tend = _turb_tend(1.0e-3)
    assert tend.tracers is not None, "no tracer tendency returned"
    assert "q_v" in tend.tracers
    dq = tend.tracers["q_v"]
    dq = dq.data if hasattr(dq, "data") else dq
    assert bool(jnp.all(jnp.isfinite(dq)))
    assert float(jnp.max(jnp.abs(dq))) > 0.0, "moisture tendency is all zero"


def test_the_scheme_actually_reads_the_state_humidity():
    """Non-vacuity: two states differing ONLY in q_v must give different
    turbulence tendencies.  With the old hard-coded zero column they were
    bit-identical."""
    dry = _turb_tend(0.0).tracers["q_v"]
    moist = _turb_tend(8.0e-3).tracers["q_v"]
    dry = np.asarray(dry.data if hasattr(dry, "data") else dry)
    moist = np.asarray(moist.data if hasattr(moist, "data") else moist)
    # The near-surface tendency is the discriminating one: evaporation into a
    # bone-dry column is far stronger than into a moist one.  Compared on a
    # RELATIVE scale — these are ~1e-8 kg/kg/s, so an absolute tolerance would
    # call any two of them equal.
    d_sfc, m_sfc = float(np.mean(dry[..., -1])), float(np.mean(moist[..., -1]))
    assert abs(d_sfc - m_sfc) > 0.1 * max(abs(d_sfc), abs(m_sfc)), (
        f"the turbulence tendency ignored the state humidity "
        f"(dry {d_sfc:.3e} vs moist {m_sfc:.3e})")


def test_other_tracers_are_mirrored_as_zero():
    """The combined wrapper sums per key, so every key of the state's tracer
    pytree must be present (zero) or the RK tree-map loses a leaf."""
    tend = _turb_tend(1.0e-3)
    state, _, _ = _state()
    assert set(tend.tracers) == set(state.tracers)
    for name, v in tend.tracers.items():
        if name == "q_v":
            continue
        arr = v.data if hasattr(v, "data") else v
        assert float(jnp.max(jnp.abs(arr))) == 0.0


# --------------------------------------------------------------------------
# 2 + 3. prescribed surface temperature and the scene's calendar
# --------------------------------------------------------------------------

def _marked_physics(coef):
    """Marker-carrying stub whose heating is proportional to the mean
    PRESCRIBED surface temperature it is handed.  A rollout that never
    anchors the surface leaves the override at its sentinel and heats by a
    completely different (large negative) amount, so the assertion below
    cannot pass by accident."""
    def _zero_tend(state, dT_mean):
        z3 = jnp.zeros_like(state.vor_hat.data)
        T_t = jnp.zeros_like(state.T_hat.data).at[0, :].set(dT_mean)
        return state._replace(
            vor_hat=state.vor_hat.replace(data=z3),
            div_hat=state.div_hat.replace(data=z3),
            T_hat=state.T_hat.replace(data=T_t),
            lnps_hat=state.lnps_hat.replace(
                data=jnp.zeros_like(state.lnps_hat.data)),
            phis_hat=state.phis_hat.replace(
                data=jnp.zeros_like(state.phis_hat.data)),
            tracers=None if state.tracers is None else {
                k: (v.replace(data=jnp.zeros_like(v.data))
                    if hasattr(v, "data") else jnp.zeros_like(v))
                for k, v in state.tracers.items()},
        )

    def fn(state, grid_, sigma_coord):
        return _zero_tend(state, 0.0)

    def fn_with_state(state, grid_, sigma_coord, phys_state, forcing=None):
        return (_zero_tend(state,
                           coef * jnp.mean(phys_state.surface_T_sfc_override)),
                phys_state)

    fn.with_phys_state = fn_with_state
    fn.init_phys_state = lambda ncol, nlev, dtype=None: init_physics_state(
        ncol, nlev, _dummy_cfg(), dtype=dtype)
    return fn


def _rad_from_calendar(coef):
    """Radiation stub heating in proportion to the seconds-of-day it is
    given, so the rollout's calendar is observable in the final state."""
    def rad(state, grid_, sigma_coord, *, sim_time_seconds=0.0, forcing=None):
        sod = 0.0 if forcing is None else forcing["seconds_of_day"]
        z3 = jnp.zeros_like(state.vor_hat.data)
        T_t = jnp.zeros_like(state.T_hat.data).at[0, :].set(coef * sod)
        return state._replace(
            vor_hat=state.vor_hat.replace(data=z3),
            div_hat=state.div_hat.replace(data=z3),
            T_hat=state.T_hat.replace(data=T_t),
            lnps_hat=state.lnps_hat.replace(
                data=jnp.zeros_like(state.lnps_hat.data)),
            phis_hat=state.phis_hat.replace(
                data=jnp.zeros_like(state.phis_hat.data)),
            tracers=None if state.tracers is None else {
                k: (v.replace(data=jnp.zeros_like(v.data))
                    if hasattr(v, "data") else jnp.zeros_like(v))
                for k, v in state.tracers.items()},
        )
    return rad


DT = 600.0
NORM = 1.0 / np.sqrt(4.0 * np.pi)     # SH mean mode -> grid mean


def _run(physics_fn, rad_fn, forcing, n_steps=6, interval=3):
    state, grid, sigma = _state()
    final = spectral_rollout(
        state, physics_fn, grid, sigma,
        SpectralPEConfig(semi_implicit=True), DT, n_steps,
        None, None,
        rad_physics_fn=rad_fn, rad_update_interval=interval,
        forcing_base=forcing,
    )
    return spectral_state_to_carry(final, grid, sigma), grid


def _forcing(grid, T_sfc, doy=244.0, sod=21600.0):
    ncol = int(grid.n_lat) * int(grid.n_lon)
    return {
        "T_sfc": jnp.full((ncol,), T_sfc),
        "sic": jnp.zeros((ncol,)),
        "day_of_year": jnp.asarray(doy),
        "seconds_of_day": jnp.asarray(sod),
    }


def test_forcing_is_accepted_alongside_subcycled_radiation():
    """This combination used to raise; every classical training arm needs it."""
    grid = create_gaussian_grid(5)
    out, _ = _run(_marked_physics(0.0), _rad_from_calendar(0.0),
                  _forcing(grid, 290.0))
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(out.T))))


def test_the_prescribed_surface_temperature_reaches_the_physics():
    """Two runs differing only in the prescribed skin temperature must differ
    by exactly the stub's response to it."""
    grid = create_gaussian_grid(5)
    coef = 1.0e-6
    warm, _ = _run(_marked_physics(coef), _rad_from_calendar(0.0),
                   _forcing(grid, 300.0))
    cold, _ = _run(_marked_physics(coef), _rad_from_calendar(0.0),
                   _forcing(grid, 280.0))
    dT = float(jnp.mean(jnp.asarray(warm.T)) - jnp.mean(jnp.asarray(cold.T)))
    # 6 steps of a constant tendency coef*(300-280) on the mean SH mode.
    assert dT == pytest.approx(6 * DT * coef * 20.0 * NORM, rel=1e-3)


def test_radiation_sees_the_scene_calendar_advancing():
    """Radiation must run on the scene's own clock, and that clock must move
    through the window: with a 3-step sub-cycle the two solves see
    seconds-of-day s and s + 3*dt, not the module default twice."""
    grid = create_gaussian_grid(5)
    coef = 1.0e-9
    sod = 21600.0
    out, _ = _run(_marked_physics(0.0), _rad_from_calendar(coef),
                  _forcing(grid, 290.0, sod=sod))
    base, _ = _run(_marked_physics(0.0), _rad_from_calendar(0.0),
                   _forcing(grid, 290.0, sod=sod))
    dT = float(jnp.mean(jnp.asarray(out.T)) - jnp.mean(jnp.asarray(base.T)))
    # steps 0-2 held the solve at sod; steps 3-5 at sod + 3*dt.
    expect = DT * coef * (3.0 * sod + 3.0 * (sod + 3.0 * DT)) * NORM
    assert dT == pytest.approx(expect, rel=1e-3)


def test_a_state_without_a_surface_slot_is_refused():
    """A silent drop of the anchor is what produced a run with no surface
    energy exchange; the rollout must say so instead."""
    grid = create_gaussian_grid(5)
    fn = _marked_physics(0.0)
    fn.init_phys_state = lambda ncol, nlev, dtype=None: jnp.asarray(0.0)

    def _bare(state, grid_, sigma_coord, phys_state, forcing=None):
        return fn(state, grid_, sigma_coord), phys_state
    fn.with_phys_state = _bare
    with pytest.raises(TypeError, match="surface_T_sfc_override"):
        _run(fn, _rad_from_calendar(0.0), _forcing(grid, 290.0))


# --------------------------------------------------------------------------
# 4. the classical arm actually hands its sample forcing to the rollout
# --------------------------------------------------------------------------

def test_the_classical_arm_passes_its_sample_forcing():
    """End to end: the WeatherBench classical builder used to DROP the
    sample's forcing (``uses_forcing = False``), so the two fixes above would
    have been unreachable from the arm that needs them.

    Probe: hand the rollout a prescribed surface field of the wrong length.
    A lane that forwards the forcing raises on the shape; a lane that drops
    it runs happily — so this fails if the wiring is reverted, without
    needing ERA5 or a full forecast.
    """
    from types import SimpleNamespace
    from pathlib import Path
    import yaml as _yaml

    from legoesm.training.scale_build import build_mode_components

    repo = Path(__file__).resolve().parents[2]
    yml = _yaml.safe_load(
        (repo / "config" / "wb" / "campaign" / "spectral_smoke.yaml").read_text())
    cfg = SimpleNamespace(mode="physics", training_core="spectral",
                          smoke=True, multi_step_hours=(6,), n_days=1)
    _m, grid, sigma, params, make_run_seg, _loss, _dt = build_mode_components(
        cfg, yml)

    nlev = int(jnp.shape(sigma.sigma_full)[0])
    carry = _carry(grid.n_lat, grid.n_lon, nlev=nlev, extras=False)
    bad = {
        "T_sfc": jnp.full((3,), 290.0),        # deliberately not ncol
        "sic": jnp.zeros((3,)),
        "day_of_year": jnp.asarray(244.0),
        "seconds_of_day": jnp.asarray(21600.0),
    }
    with pytest.raises(ValueError, match="prescribed surface temperature"):
        make_run_seg(params).raw(carry, 1, bad)


def test_a_warmer_prescribed_surface_warms_the_lowest_level():
    """The physical consequence, not just the wiring: with the surface
    anchored to a prescribed field, a surface 10 K warmer than the air must
    heat the lowest model level.  Before the fix the surface temperature WAS
    the lowest air temperature, so the sensible heat flux was identically
    zero and this difference was exactly 0.
    """
    from types import SimpleNamespace
    from pathlib import Path
    import yaml as _yaml

    from legoesm.training.scale_build import build_mode_components

    repo = Path(__file__).resolve().parents[2]
    yml = _yaml.safe_load(
        (repo / "config" / "wb" / "campaign" / "spectral_smoke.yaml").read_text())
    cfg = SimpleNamespace(mode="physics", training_core="spectral",
                          smoke=True, multi_step_hours=(6,), n_days=1)
    _m, grid, sigma, params, make_run_seg, _loss, _dt = build_mode_components(
        cfg, yml)

    nlev = int(jnp.shape(sigma.sigma_full)[0])
    carry = _carry(grid.n_lat, grid.n_lon, nlev=nlev, extras=False)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    T_air = float(jnp.asarray(carry.T)[..., -1].mean())

    def _run(T_sfc):
        fc = {
            "T_sfc": jnp.full((ncol,), T_sfc),
            "sic": jnp.zeros((ncol,)),
            "day_of_year": jnp.asarray(244.0),
            "seconds_of_day": jnp.asarray(21600.0),
        }
        return make_run_seg(params).raw(carry, 2, fc)

    cold = _run(T_air)
    warm = _run(T_air + 10.0)
    d_sfc = float(jnp.mean(jnp.asarray(warm.T)[..., -1]
                           - jnp.asarray(cold.T)[..., -1]))
    assert d_sfc > 1.0e-4, (
        f"a 10 K warmer surface changed the lowest level by {d_sfc:.3e} K — "
        "the surface heat flux is not reaching the atmosphere")


# --------------------------------------------------------------------------
# 5. the radiation g-point block size is reachable from the campaign config
# --------------------------------------------------------------------------

def _built_radiation_config(gpt_batch):
    """Capture the RadiationConfig the classical builder hands to RRTMGP."""
    from types import SimpleNamespace
    from pathlib import Path
    import yaml as _yaml

    import legoesm.atmosphere.physics.radiation.integration as rad_int
    from legoesm.training.scale_build import build_mode_components

    repo = Path(__file__).resolve().parents[2]
    yml = _yaml.safe_load(
        (repo / "config" / "wb" / "campaign" / "spectral_smoke.yaml").read_text())
    if gpt_batch is not None:
        yml["classical"] = dict(yml.get("classical", {}),
                                rrtmgp_gpoint_batch_size=gpt_batch)
    seen = {}
    real = rad_int.make_radiation_physics

    def spy(cfg, model_type, **kw):
        seen["cfg"] = cfg
        return real(cfg, model_type, **kw)

    rad_int.make_radiation_physics = spy
    try:
        cfg = SimpleNamespace(mode="physics", training_core="spectral",
                              smoke=True, multi_step_hours=(6,), n_days=1)
        _m, _g, _s, params, make_run_seg, _l, _dt = build_mode_components(cfg, yml)
        make_run_seg(params)
    finally:
        rad_int.make_radiation_physics = real
    return seen["cfg"]


def test_the_gpoint_block_size_reaches_rrtmgp():
    """The scratch this knob controls is 50.5 GiB at 16 and 44.7 GiB at 8 on
    the real training step, so a knob that did not reach the scheme would be
    the difference between a run that fits and one that does not."""
    assert _built_radiation_config(8).rrtmgp.gpoint_batch_size == 8
    assert _built_radiation_config(16).rrtmgp.gpoint_batch_size == 16


def test_a_zero_gpoint_block_size_is_refused():
    with pytest.raises(ValueError, match="rrtmgp_gpoint_batch_size"):
        _built_radiation_config(0)

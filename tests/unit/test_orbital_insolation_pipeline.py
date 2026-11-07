"""The production PhysicsPipeline must honour ExperimentConfig.orbital_insolation.

Codex adversarial-review (orbital AMIP-II) caught that the orbital flag was
threaded only into the standalone radiation `integration` module, NOT the
PhysicsPipeline radiation builders / diagnostics that the main cubed-sphere
AMIP path actually runs.  These tests pin the wiring end-to-end:

* ``build_physics_pipeline`` sets ``pipeline.orbit`` from the config flag,
* the pipeline's ``_toa_insolation`` diagnostic (the EXACT convention the
  gray/RRTMGP radiation builders use for the prescribed insolation) differs
  between January (perihelion) and July (aphelion) by the eccentricity
  ``(a/r)^2`` factor vs the circular-orbit baseline,
* a config with the flag OFF is bit-for-bit the circular baseline.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.solar import (
    OrbitalParameters,
    earth_orbit,
    earth_sun_distance_factor,
)
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere

S0 = float(constants.S_0)
N_CS = 4


def _sigma(nlev=6):
    class _S:
        sigma_full = jnp.linspace(0.1, 0.95, nlev)
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        dsigma = jnp.diff(jnp.linspace(0.05, 1.0, nlev + 1))

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
    return _S()


def _pipeline(orbital: bool, diurnal: bool = True):
    grid = create_cubed_sphere(N_CS)
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=diurnal, orbital_insolation=orbital)
    pipe = build_physics_pipeline(grid, _sigma(), cfg)
    return grid, pipe


def test_build_physics_pipeline_sets_orbit_from_config():
    _, pipe_on = _pipeline(orbital=True)
    _, pipe_off = _pipeline(orbital=False)
    assert isinstance(pipe_on.orbit, OrbitalParameters)
    assert pipe_on.orbit == earth_orbit()
    assert pipe_off.orbit is None


def _day_area_mean_toa(grid, pipe, day, diurnal=True):
    lat = grid.grid_lat
    lon = grid.grid_lon
    w = np.cos(np.asarray(lat))
    w = w / w.sum()
    if diurnal:
        tot = 0.0
        for h in range(24):
            ins = np.asarray(pipe._toa_insolation(
                lat, lon, float(day), h * 3600.0, S0))
            tot += (ins * w).sum()
        return tot / 24.0
    ins = np.asarray(pipe._toa_insolation(lat, lon, float(day), 0.0, S0))
    return (ins * w).sum()


def test_pipeline_toa_insolation_scales_by_eccentricity():
    """Perihelion (Jan) gets +3.4%, aphelion (Jul) -3.3% vs circular — the
    diurnal day+area-mean ratio equals the (a/r)^2 distance factor."""
    grid, pipe_on = _pipeline(orbital=True)
    _, pipe_off = _pipeline(orbital=False)
    orbit = earth_orbit()
    for day, lo, hi in ((3.0, 1.030, 1.040), (185.0, 0.962, 0.972)):
        circ = _day_area_mean_toa(grid, pipe_off, day)
        orb = _day_area_mean_toa(grid, pipe_on, day)
        ratio = orb / circ
        eccf = float(earth_sun_distance_factor(day, orbit))
        assert lo < ratio < hi, f"day {day}: ratio {ratio:.4f}"
        assert abs(ratio - eccf) < 2e-3


def test_pipeline_orbital_off_is_circular_baseline():
    """orbital_insolation=False ⇒ the diagnostic is bit-for-bit the legacy
    circular-orbit insolation (no silent change to existing runs)."""
    grid, pipe_off = _pipeline(orbital=False, diurnal=True)
    lat, lon = grid.grid_lat, grid.grid_lon
    # Legacy reference: a pipeline whose orbit attribute was never set.
    _, pipe_legacy = _pipeline(orbital=False, diurnal=True)
    pipe_legacy.orbit = None
    for h in (0.0, 21600.0, 43200.0):
        a = np.asarray(pipe_off._toa_insolation(lat, lon, 120.0, h, S0))
        b = np.asarray(pipe_legacy._toa_insolation(lat, lon, 120.0, h, S0))
        assert np.array_equal(a, b)


def test_rrtmgp_sw_flux_tracks_per_step_irradiance(monkeypatch):
    """Per-step irradiance (time-varying TSI from solar_source=file, passed as
    s_0) must scale the RRTMGP SW fluxes/heating, not only the rsdt diagnostic.
    The solver is built once with the static config S_0, so the SW output is
    rescaled by s_0/S_0 (* eccf).  A fake solver isolates the rescale from the
    gas-optics tables."""
    import legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp as rrtmgp_mod
    from legoesm.atmosphere.physics.radiation.output import RadiationOutput
    from legoesm.driver.physics_pipeline import _build_rrtmgp_radiation_fn

    class _FakeSolver:
        def solve_columns(self, *, T, **kw):
            ncol, nlev = T.shape
            flux = jnp.ones((ncol, nlev + 1))
            heat = jnp.ones((ncol, nlev))
            return RadiationOutput(
                lw_flux_up=flux, lw_flux_down=flux,
                sw_flux_up=flux, sw_flux_down=flux,
                heating_rate=heat, lw_heating_rate=heat, sw_heating_rate=heat,
            )

    monkeypatch.setattr(rrtmgp_mod.RRTMGP, "from_legoesm_config",
                        classmethod(lambda cls, cfg: _FakeSolver()))

    cfg = ExperimentConfig(radiation="rrtmgp", diurnal_cycle=True)
    rad_fn = _build_rrtmgp_radiation_fn(cfg)

    ncol, nlev = 2, 2
    T = jnp.full((ncol, nlev), 260.0)
    p_full = jnp.broadcast_to(jnp.array([3e4, 7e4]), (ncol, nlev))
    p_half = jnp.broadcast_to(jnp.array([1e4, 5e4, 1e5]), (ncol, nlev + 1))
    q_v = jnp.full((ncol, nlev), 1e-3)
    T_sfc = jnp.full((ncol,), 290.0)
    lat = jnp.array([0.1, -0.2])
    lon = jnp.array([0.0, 1.0])
    alb = jnp.full((ncol,), 0.1)
    emis = jnp.full((ncol,), 0.99)
    o3 = jnp.zeros((ncol, nlev))
    aer = jnp.zeros((ncol, nlev))
    sw = jnp.array([])  # size 0 -> no spectral weighting

    def _call(s0):
        return rad_fn(T, p_full, p_half, q_v, T_sfc, lat, lon, 80.0, 43200.0,
                      alb, emis, o3, aer, sw, s_0=s0)

    S0 = float(constants.S_0)
    base = _call(S0)
    hi = _call(2.0 * S0)   # a +100% TSI perturbation
    # SW fluxes + heating scale with the irradiance; the LW is untouched.
    assert np.allclose(np.asarray(hi.sw_flux_down)
                       / np.asarray(base.sw_flux_down), 2.0, rtol=1e-6)
    assert np.allclose(np.asarray(hi.sw_heating_rate)
                       / np.asarray(base.sw_heating_rate), 2.0, rtol=1e-6)
    assert np.allclose(np.asarray(hi.lw_flux_down),
                       np.asarray(base.lw_flux_down))
    # rsdt diagnostic moves too (consistency).
    assert np.allclose(np.asarray(hi.toa_insolation)
                       / np.asarray(base.toa_insolation), 2.0, rtol=1e-6)

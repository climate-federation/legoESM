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

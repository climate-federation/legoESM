"""Prognostic snow + snow-albedo feedback on the AMIP slab-land tile.

``--snow-albedo-feedback`` adds a prognostic snow-water (SWE) carry threaded
exactly like the soil-water bucket (``w_land``) through SegmentCarry /
PhysicsOutput / the driver, and brightens the land albedo when snow is present
(``legoesm.surface_albedo.land_albedo``).  Tests:

* ``_land_albedo_eff`` brightens the land albedo under snow and returns the
  static vegetation albedo when snow is absent or the feedback is off
  (byte-identical);
* the compiled physics step (``build_step_unified``) THREADS snow: it returns
  an updated ``PhysicsOutput.snow`` that melts under warm surface temps;
* with the feedback OFF, ``PhysicsOutput.snow`` is ``None`` (byte-identical
  legacy carry);
* the SegmentCarry / pack_carry plumbing carries snow.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.driver.compiled_segments import SegmentCarry, pack_carry
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere

N_CS = 4
NLEV = 6
SHAPE_2D = (6, N_CS, N_CS)


def _sigma(nlev=NLEV):
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


def _snow_pipeline(feedback=True):
    grid = create_cubed_sphere(N_CS)
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False, turbulence="louis",
                           surface_bulk_scheme="coare3")
    pipe = build_physics_pipeline(grid, _sigma(), cfg)
    pipe.f_land = jnp.full(SHAPE_2D, 0.5)
    pipe.albedo_land = jnp.full(SHAPE_2D, 0.20)
    pipe.rad_update_steps = 1
    pipe.slab_land_active = True
    pipe.surface_tiled = True
    pipe.surface_z0_land = 0.1
    pipe.snow_albedo_feedback = feedback
    return pipe, grid


class TestLandAlbedoEff:
    def test_snow_brightens_albedo(self):
        pipe, grid = _snow_pipeline(feedback=True)
        lat = grid.grid_lat
        snow_heavy = jnp.full(SHAPE_2D, 200.0)   # deep snow (>= crit)
        alb = np.asarray(pipe._land_albedo_eff(lat, snow_heavy))
        # Snow albedo (0.5-0.8) >> vegetation albedo 0.20.
        assert float(alb.mean()) > 0.20
        assert float(alb.mean()) > 0.45

    def test_no_snow_is_vegetation_albedo(self):
        pipe, grid = _snow_pipeline(feedback=True)
        alb = np.asarray(pipe._land_albedo_eff(grid.grid_lat,
                                               jnp.zeros(SHAPE_2D)))
        assert np.allclose(alb, 0.20, atol=1e-6)

    def test_feedback_off_returns_static_albedo(self):
        pipe, grid = _snow_pipeline(feedback=False)
        # Off ⇒ returns self.albedo_land unchanged (byte-identical), even with
        # deep snow supplied.
        alb = pipe._land_albedo_eff(grid.grid_lat, jnp.full(SHAPE_2D, 200.0))
        assert alb is pipe.albedo_land


def _run_step(pipe, snow, T_land_val=300.0, T_air=285.0):
    step_fn = pipe.build_step_unified()
    ad = pipe.adapter
    s2 = ad.shape_2d
    s3 = (*s2, NLEV)
    held3 = jnp.zeros(s3)
    held2 = jnp.zeros(s2)
    o3 = jnp.zeros((ad.ncol, NLEV))
    aer = jnp.zeros((ad.ncol, NLEV))
    return step_fn(
        jnp.bool_(True),
        jnp.full(s3, T_air), jnp.full(s2, 1.0e5),
        jnp.full(s3, 0.006), jnp.zeros(s3), jnp.zeros(s3),
        jnp.zeros((ad.ncol,)),
        jnp.full(s3, 5.0), jnp.zeros(s3),
        jnp.full(s2, 295.0), jnp.zeros(s2),
        jnp.full(s2, 0.4), jnp.full(s2, 1.0),
        100.0, 43200.0, 600.0,
        jnp.array([]), constants.S_0, o3, aer,
        held3, held2, held2, held2, held2, held2,
        T_land=jnp.full(s2, T_land_val),
        w_land=None,
        snow=snow,
    )


class TestCompiledStepThreadsSnow:
    def test_snow_threaded_and_melts_when_warm(self):
        """The compiled step returns PhysicsOutput.snow; a warm surface melts
        the initial snowpack (degree-day melt)."""
        pipe, _ = _snow_pipeline(feedback=True)
        snow0 = jnp.full(SHAPE_2D, 50.0)   # 50 kg/m^2 initial SWE
        # step_unified returns a 4-tuple (phys_out, held, T_land, land_ml).
        phys_out, *_ = _run_step(pipe, snow0, T_land_val=305.0)
        assert phys_out.snow is not None, "snow must be threaded on PhysicsOutput"
        snow_new = np.asarray(phys_out.snow)
        assert np.all(np.isfinite(snow_new))
        # Warm land (305 K > 273) with no snowfall ⇒ the pack melts.
        assert float(snow_new.mean()) < 50.0
        assert np.all(snow_new >= 0.0)

    def test_feedback_off_snow_is_none(self):
        """Off ⇒ PhysicsOutput.snow is None (byte-identical legacy carry)."""
        pipe, _ = _snow_pipeline(feedback=False)
        phys_out, *_ = _run_step(pipe, None)
        assert phys_out.snow is None


class TestSnowCarryPlumbing:
    def test_pack_carry_threads_snow(self):
        z2 = jnp.zeros(SHAPE_2D)
        state = _MinimalState(z2)
        common = dict(
            held_dT_rad=jnp.zeros((*SHAPE_2D, NLEV)),
            held_sw_net_sfc=z2, held_lw_net_sfc=z2,
            held_sw_up_toa=z2, held_lw_up_toa=z2, held_sw_down_toa=z2,
            step_index=jnp.int32(0),
        )
        snow = jnp.full(SHAPE_2D, 12.0)
        carry = pack_carry(state, z2, z2, z2, snow=snow, **common)
        assert isinstance(carry, SegmentCarry)
        assert carry.snow is not None
        assert np.allclose(np.asarray(carry.snow), 12.0)
        # None when not supplied (byte-identical legacy carry).
        carry0 = pack_carry(state, z2, z2, z2, **common)
        assert carry0.snow is None


class _MinimalState:
    """Duck ocean/atmo state exposing the fields pack_carry reads."""
    def __init__(self, arr):
        from legoesm.core.field import Field
        d3 = ("face", "x", "y", "level")
        d2 = ("face", "x", "y")
        arr3 = jnp.broadcast_to(arr[..., None], (*arr.shape, NLEV))
        self.u = Field(arr3, name="u", dims=d3, units="m/s")
        self.v = Field(arr3, name="v", dims=d3, units="m/s")
        self.T = Field(arr3, name="T", dims=d3, units="K")
        self.p_s = Field(arr, name="p_s", dims=d2, units="Pa")
        self.phis = Field(arr, name="phis", dims=d2, units="m2/s2")

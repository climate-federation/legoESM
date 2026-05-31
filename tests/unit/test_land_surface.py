"""Unit tests for the slab-land surface tile in the ModelDriver path.

Covers:
- land/ocean surface blending (``PhysicsPipeline._blend_land``);
- the semi-implicit slab-land energy budget (``_step_slab_land``):
  relaxation toward surface-energy-balance equilibrium and
  unconditional stability;
- ``compute_radiation_core`` with the land tile active vs inactive;
- the land-sea-mask loader (``grids.topography.load_land_fraction``).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.driver.config import ExperimentConfig
from legoesm.grids.cubed_sphere import create_cubed_sphere

NLEV = 6
N_CS = 4


def _sigma(nlev=NLEV):
    class _S:
        sigma_full = jnp.linspace(0.1, 0.95, nlev)
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        dsigma = jnp.diff(jnp.linspace(0.05, 1.0, nlev + 1))
    return _S()


def _pipeline(land=False, **cfg_overrides):
    """Build a gray-radiation pipeline, optionally with the land tile on."""
    grid = create_cubed_sphere(N_CS)
    sigma = _sigma()
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False, **cfg_overrides)
    pipe = build_physics_pipeline(grid, sigma, cfg)
    if land:
        shape_2d = (6, N_CS, N_CS)
        pipe.f_land = jnp.full(shape_2d, 1.0)        # all-land
        pipe.albedo_land = jnp.full(shape_2d, 0.20)
        pipe.rad_update_steps = 1
    return pipe, grid


# ---------------------------------------------------------------------------
# Surface blending
# ---------------------------------------------------------------------------

class TestBlendLand:
    def test_blend_endpoints_and_midpoint(self):
        pipe, _ = _pipeline(land=True)
        ocean = jnp.full((6, N_CS, N_CS), 290.0)
        land = jnp.full((6, N_CS, N_CS), 305.0)

        pipe.f_land = jnp.zeros((6, N_CS, N_CS))
        assert jnp.allclose(pipe._blend_land(ocean, land), ocean)

        pipe.f_land = jnp.ones((6, N_CS, N_CS))
        assert jnp.allclose(pipe._blend_land(ocean, land), land)

        pipe.f_land = jnp.full((6, N_CS, N_CS), 0.5)
        assert jnp.allclose(pipe._blend_land(ocean, land), 0.5 * (ocean + land))


# ---------------------------------------------------------------------------
# Slab-land energy budget
# ---------------------------------------------------------------------------

def _slab_inputs(pipe, T_land_val):
    shape_2d = (6, N_CS, N_CS)
    shape_3d = (*shape_2d, NLEV)
    T_land = jnp.full(shape_2d, T_land_val)
    sw_down = jnp.full(shape_2d, 400.0)
    lw_down = jnp.full(shape_2d, 340.0)
    T = jnp.full(shape_3d, 285.0)
    p_s = jnp.full(shape_2d, 1.0e5)
    q_v = jnp.full(shape_3d, 0.004)
    u = jnp.full(shape_3d, 4.0)
    v = jnp.zeros(shape_3d)
    return T_land, sw_down, lw_down, T, p_s, q_v, u, v


class TestSlabLand:
    def test_cold_land_warms(self):
        """A land surface far below equilibrium gains energy and warms."""
        pipe, _ = _pipeline(land=True)
        args = _slab_inputs(pipe, 250.0)
        T_new = pipe._step_slab_land(*args, dt=600.0)
        assert jnp.all(T_new > 250.0), "cold land should warm under net heating"
        assert jnp.all(jnp.isfinite(T_new))

    def test_relaxes_to_equilibrium(self):
        """Iterating the slab budget converges to a steady skin temperature."""
        pipe, _ = _pipeline(land=True)
        T_land, sw_down, lw_down, T, p_s, q_v, u, v = _slab_inputs(pipe, 250.0)
        for _ in range(400):
            T_land = pipe._step_slab_land(
                T_land, sw_down, lw_down, T, p_s, q_v, u, v, dt=600.0)
        T_final = T_land
        T_next = pipe._step_slab_land(
            T_final, sw_down, lw_down, T, p_s, q_v, u, v, dt=600.0)
        # Converged: consecutive steps barely move, and the result is a
        # physically plausible land skin temperature.
        assert jnp.max(jnp.abs(T_next - T_final)) < 0.05
        assert jnp.all((T_final > 230.0) & (T_final < 340.0))

    def test_semi_implicit_stable_for_large_dt(self):
        """The semi-implicit update stays bounded for a huge time step."""
        pipe, _ = _pipeline(land=True)
        args = _slab_inputs(pipe, 250.0)
        # dt 100x larger than a normal radiation step — explicit would blow up.
        T_new = pipe._step_slab_land(*args, dt=360000.0)
        assert jnp.all(jnp.isfinite(T_new))
        assert jnp.all((T_new > 230.0) & (T_new < 360.0))

    def test_equilibrium_independent_of_initial_temperature(self):
        """Hot and cold starts relax to the same SEB equilibrium."""
        pipe, _ = _pipeline(land=True)
        inp = _slab_inputs(pipe, 0.0)[1:]  # sw_down..v, drop T_land
        cold = jnp.full((6, N_CS, N_CS), 240.0)
        hot = jnp.full((6, N_CS, N_CS), 330.0)
        for _ in range(600):
            cold = pipe._step_slab_land(cold, *inp, dt=600.0)
            hot = pipe._step_slab_land(hot, *inp, dt=600.0)
        assert jnp.max(jnp.abs(cold - hot)) < 0.2


# ---------------------------------------------------------------------------
# compute_radiation_core with / without the land tile
# ---------------------------------------------------------------------------

def _rad_inputs():
    shape_2d = (6, N_CS, N_CS)
    shape_3d = (*shape_2d, NLEV)
    T = jnp.full(shape_3d, 270.0)
    p_s = jnp.full(shape_2d, 1.0e5)
    q_v = jnp.full(shape_3d, 0.003)
    sst = jnp.full(shape_2d, 290.0)
    sic = jnp.zeros(shape_2d)
    lat = jnp.full(shape_2d, 0.4)
    lon = jnp.full(shape_2d, 1.0)
    u = jnp.full(shape_3d, 3.0)
    v = jnp.zeros(shape_3d)
    return T, p_s, q_v, sst, sic, lat, lon, u, v


class TestRadiationCoreLand:
    def test_ocean_only_leaves_T_land_untouched(self):
        """With the land tile inactive, T_land is returned unchanged."""
        pipe, _ = _pipeline(land=False)
        T, p_s, q_v, sst, sic, lat, lon, u, v = _rad_inputs()
        T_land = jnp.full((6, N_CS, N_CS), 300.0)
        out = pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 1.0, 0.0,
            jnp.zeros(0), constants.S_0, None, None,
            u=u, v=v, dt=600.0, T_land=T_land,
        )
        assert len(out) == 7
        assert jnp.array_equal(out[6], T_land)

    def test_land_tile_updates_T_land(self):
        """With the land tile active the slab temperature evolves."""
        pipe, _ = _pipeline(land=True)
        T, p_s, q_v, sst, sic, lat, lon, u, v = _rad_inputs()
        T_land = jnp.full((6, N_CS, N_CS), 250.0)
        out = pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 1.0, 0.0,
            jnp.zeros(0), constants.S_0, None, None,
            u=u, v=v, dt=600.0, T_land=T_land,
        )
        T_land_new = out[6]
        assert jnp.all(jnp.isfinite(T_land_new))
        assert not jnp.array_equal(T_land_new, T_land)


# ---------------------------------------------------------------------------
# Land-sea-mask loader
# ---------------------------------------------------------------------------

class TestLoadLandFraction:
    def test_regrid_percent_mask(self, tmp_path):
        """A percent-valued mask is rescaled to [0,1] and regridded."""
        import xarray as xr

        lat = np.linspace(-89.0, 89.0, 90)
        lon = np.linspace(0.0, 358.0, 180)
        # Northern hemisphere = land (100%), southern = ocean (0%).
        mask = np.where(lat[:, None] > 0.0, 100.0, 0.0) * np.ones_like(lon)
        ds = xr.Dataset(
            {"sftlf": (("lat", "lon"), mask)},
            coords={"lat": lat, "lon": lon},
        )
        path = tmp_path / "sftlf_test.nc"
        ds.to_netcdf(path)

        from legoesm.grids.topography import load_land_fraction
        grid = create_cubed_sphere(N_CS)
        f_land = load_land_fraction(grid, str(path))

        assert f_land.shape == (6, N_CS, N_CS)
        # Percent → fraction: values must land in [0, 1].
        assert jnp.all((f_land >= 0.0) & (f_land <= 1.0))
        # The mask is non-trivial (both land and ocean cells present).
        assert float(jnp.max(f_land)) > 0.5
        assert float(jnp.min(f_land)) < 0.5


class TestStepUnifiedLand:
    def test_jitted_step_unified_threads_T_land(self):
        """The jitted step_unified advances T_land through its lax.cond."""
        pipe, _ = _pipeline(land=True)
        step_fn = pipe.build_step_unified()
        ad = pipe.adapter
        shape_2d = ad.shape_2d
        shape_3d = (*shape_2d, NLEV)

        T = jnp.full(shape_3d, 270.0)
        p_s = jnp.full(shape_2d, 1.0e5)
        q_v = jnp.full(shape_3d, 0.003)
        q_c = jnp.zeros(shape_3d)
        q_r = jnp.zeros(shape_3d)
        u = jnp.full(shape_3d, 3.0)
        v = jnp.zeros(shape_3d)
        sst = jnp.full(shape_2d, 290.0)
        sic = jnp.zeros(shape_2d)
        lat = jnp.full(shape_2d, 0.4)
        lon = jnp.full(shape_2d, 1.0)
        held_3d = jnp.zeros(shape_3d)
        held_2d = jnp.zeros(shape_2d)
        o3 = jnp.zeros((ad.ncol, NLEV))
        aerosol = jnp.zeros((ad.ncol, NLEV))
        T_land = jnp.full(shape_2d, 250.0)

        phys_out, new_held, T_land_new = step_fn(
            jnp.bool_(True),
            T, p_s, q_v, q_c, q_r, jnp.zeros((ad.ncol,)), u, v,
            sst, sic, lat, lon, 100.0, 43200.0, 600.0,
            jnp.array([]), constants.S_0, o3, aerosol,
            held_3d, held_2d, held_2d, held_2d, held_2d, held_2d,
            T_land=T_land,
        )
        assert jnp.all(jnp.isfinite(phys_out.dT_dt))
        assert jnp.all(jnp.isfinite(T_land_new))
        # Land tile active → the slab skin temperature must evolve.
        assert not jnp.array_equal(T_land_new, T_land)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

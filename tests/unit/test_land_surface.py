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
import jax
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

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
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
        assert len(out) == 8          # + land_ml_new (None on the slab path)
        assert jnp.array_equal(out[6], T_land)
        assert out[7] is None

    def test_land_tile_updates_T_land(self):
        """With the land tile active the slab temperature evolves."""
        pipe, _ = _pipeline(land=True)
        pipe.slab_land_active = True   # step T_land (not the passive blend-only mode)
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


class TestRadiationCoreSurfaceOverride:
    """Cluster-A feedback: a coupler-provided tile-blended surface albedo /
    skin temperature must OVERRIDE the static internal blend in
    compute_radiation_core (and be byte-identical when not provided)."""

    def _call(self, pipe, **extra):
        T, p_s, q_v, sst, sic, lat, lon, u, v = _rad_inputs()
        return pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 1.0, 0.0,
            jnp.zeros(0), constants.S_0, None, None,
            u=u, v=v, dt=600.0, T_land=None, **extra,
        )

    def test_none_override_byte_identical(self):
        """Explicit None overrides reproduce the no-override path exactly."""
        pipe, _ = _pipeline(land=False)
        base = self._call(pipe)
        out = self._call(pipe, sfc_albedo_override=None, sfc_T_override=None)
        for b, o in zip(base, out):
            if b is None:
                assert o is None
            else:
                assert jnp.array_equal(b, o)

    def test_albedo_override_changes_sw_net(self):
        """A brighter coupler albedo reflects more SW ⇒ less surface net SW."""
        pipe, _ = _pipeline(land=False)
        shape_2d = (6, N_CS, N_CS)
        base = self._call(pipe)                          # ocean albedo 0.06
        out = self._call(pipe, sfc_albedo_override=jnp.full(shape_2d, 0.6))
        # sw_net_sfc is tuple index 1.  Brighter surface ⇒ lower net SW.
        assert not jnp.allclose(base[1], out[1])
        assert bool(jnp.all(out[1] <= base[1] + 1e-9))

    def test_T_override_changes_lw_net(self):
        """A colder coupler skin temperature changes surface net LW."""
        pipe, _ = _pipeline(land=False)
        shape_2d = (6, N_CS, N_CS)
        base = self._call(pipe)                          # T_sfc ≈ sst 290 K
        out = self._call(pipe, sfc_T_override=jnp.full(shape_2d, 230.0))
        # lw_net_sfc is tuple index 2.
        assert not jnp.allclose(base[2], out[2])


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
        pipe.slab_land_active = True   # step T_land (not the passive blend-only mode)
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

        phys_out, new_held, T_land_new, _land_ml = step_fn(
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

    def test_jitted_step_unified_advances_multilayer_land(self):
        """End-to-end: the jitted step_unified runs the MULTILAYER coupler tile (not
        the slab) when the pipeline carries a land_ml state — the soil column advances
        through the segment, and rides the SegmentCarry.land_ml field (the refactor)."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.soil_grid import SoilGridConfig
        from legoesm.land.multilayer_land import init_multilayer_land_state
        from legoesm.land.surface_params import LandSurfaceParams
        pipe, _ = _pipeline(land=True)
        ad = pipe.adapter
        shape_2d = ad.shape_2d; shape_3d = (*shape_2d, NLEV); ncol = ad.ncol
        cfg = MultiLayerLandConfig(
            soil_grid=SoilGridConfig(n_layers=6, total_depth=2.0), bulk_scheme="most")
        c = lambda x: jnp.full(ncol, float(x))
        pipe.land_ml_cfg = cfg
        pipe.land_ml_lat = jnp.full(ncol, 0.4)
        pipe.land_ml_params = LandSurfaceParams(
            albedo_veg=c(0.2), emissivity=c(0.97), z0=c(0.1), W_max=c(150.0),
            C_soil=c(2.0e6), d_soil=c(1.0), root_depth=c(1.0), theta_wp=c(0.12),
            theta_fc=c(0.30), Vc_max25=c(50.0), LCMA=c(50.0), g1=c(9.0))
        land_ml = init_multilayer_land_state(ncol, cfg, T_init=285.0)
        step_fn = pipe.build_step_unified(static_need_rad=True)
        held_3d = jnp.zeros(shape_3d); held_2d = jnp.zeros(shape_2d)
        _po, _nh, _Tl, land_ml_new = step_fn(
            jnp.bool_(True),
            jnp.full(shape_3d, 288.0), jnp.full(shape_2d, 1.0e5),
            jnp.full(shape_3d, 0.006), jnp.zeros(shape_3d), jnp.zeros(shape_3d),
            jnp.zeros((ncol,)), jnp.full(shape_3d, 3.0), jnp.zeros(shape_3d),
            jnp.full(shape_2d, 290.0), jnp.zeros(shape_2d), jnp.full(shape_2d, 0.4),
            jnp.full(shape_2d, 1.0), 100.0, 43200.0, 600.0,
            jnp.array([]), constants.S_0, jnp.zeros((ncol, NLEV)),
            jnp.zeros((ncol, NLEV)),
            held_3d, held_2d, held_2d, held_2d, held_2d, held_2d,
            T_land=jnp.zeros(shape_2d), land_ml=land_ml,
        )
        # the multilayer state advanced (not the slab), stays finite + physical
        assert land_ml_new is not None
        assert jnp.all(jnp.isfinite(land_ml_new.T_soil))
        assert not jnp.array_equal(land_ml_new.T_soil, land_ml.T_soil)
        assert jnp.all((land_ml_new.theta_soil >= 0.0) & (land_ml_new.theta_soil <= 1.0))

    def test_tiled_surface_flux_changes_tendencies(self):
        """With louis + coare3 + an active land tile, enabling tiled surface
        fluxes (land Monin-Obukhov on the land tile, coare3 on the ocean
        tile) changes the BL tendencies vs running coare3 on the blended
        surface — proving the per-tile flux is wired into the compiled
        physics step (and that the land scheme genuinely differs)."""

        def _build(tiled):
            pipe, _ = _pipeline(land=True, turbulence="louis",
                                surface_bulk_scheme="coare3")
            pipe.slab_land_active = True
            pipe.surface_tiled = tiled
            pipe.surface_z0_land = 0.1
            # A land/ocean mosaic so BOTH tiles contribute to the blend.
            pipe.f_land = jnp.full((6, N_CS, N_CS), 0.5)
            return pipe

        def _run(pipe):
            step_fn = pipe.build_step_unified()
            ad = pipe.adapter
            shape_2d = ad.shape_2d
            shape_3d = (*shape_2d, NLEV)
            T = jnp.full(shape_3d, 285.0)
            p_s = jnp.full(shape_2d, 1.0e5)
            q_v = jnp.full(shape_3d, 0.006)
            q_c = jnp.zeros(shape_3d)
            q_r = jnp.zeros(shape_3d)
            u = jnp.full(shape_3d, 5.0)
            v = jnp.zeros(shape_3d)
            sst = jnp.full(shape_2d, 295.0)
            sic = jnp.zeros(shape_2d)
            lat = jnp.full(shape_2d, 0.4)
            lon = jnp.full(shape_2d, 1.0)
            held_3d = jnp.zeros(shape_3d)
            held_2d = jnp.zeros(shape_2d)
            o3 = jnp.zeros((ad.ncol, NLEV))
            aerosol = jnp.zeros((ad.ncol, NLEV))
            T_land = jnp.full(shape_2d, 300.0)  # warm land vs 295 K ocean
            # step_unified returns a 4-tuple since the differentiable land
            # refactor (physics_out, held, T_land_new, land_ml_new).
            phys_out, _, _, _ = step_fn(
                jnp.bool_(True),
                T, p_s, q_v, q_c, q_r, jnp.zeros((ad.ncol,)), u, v,
                sst, sic, lat, lon, 100.0, 43200.0, 600.0,
                jnp.array([]), constants.S_0, o3, aerosol,
                held_3d, held_2d, held_2d, held_2d, held_2d, held_2d,
                T_land=T_land,
            )
            return phys_out

        out_tiled = _run(_build(True))
        out_plain = _run(_build(False))

        assert jnp.all(jnp.isfinite(out_tiled.dT_dt))
        # Tiled land Monin-Obukhov flux differs from coare3-on-the-blend, so
        # the resulting BL temperature tendency must differ somewhere.
        assert not jnp.allclose(out_tiled.dT_dt, out_plain.dT_dt)

    def test_jitted_step_unified_threads_sfc_override(self):
        """step_unified threads the coupler-provided surface albedo / skin
        temperature override all the way into compute_radiation_core — the
        Cluster-A dynamic surface → radiation feedback."""
        pipe, _ = _pipeline(land=False)
        step_fn = pipe.build_step_unified(static_need_rad=True)
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

        def _run(**extra):
            return step_fn(
                jnp.bool_(True),
                T, p_s, q_v, q_c, q_r, jnp.zeros((ad.ncol,)), u, v,
                sst, sic, lat, lon, 100.0, 43200.0, 600.0,
                jnp.array([]), constants.S_0, o3, aerosol,
                held_3d, held_2d, held_2d, held_2d, held_2d, held_2d,
                T_land=None, **extra,
            )

        _, base_held, _, _ = _run()
        # held tuple: (dT_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
        #              sw_down_toa).  A much brighter coupler albedo lowers
        # surface net SW.
        _, alb_held, _, _ = _run(sfc_albedo_override=jnp.full(shape_2d, 0.6))
        assert not jnp.allclose(base_held[1], alb_held[1])
        assert bool(jnp.all(alb_held[1] <= base_held[1] + 1e-9))
        # A much colder coupler skin temperature changes surface net LW.
        _, T_held, _, _ = _run(sfc_T_override=jnp.full(shape_2d, 230.0))
        assert not jnp.allclose(base_held[2], T_held[2])
        # Override = None reproduces the baseline exactly (byte-identical).
        _, none_held, _, _ = _run(sfc_albedo_override=None, sfc_T_override=None)
        for b, n in zip(base_held, none_held):
            assert jnp.array_equal(b, n)


class TestMultilayerLandTile:
    """The PhysicsPipeline multilayer land tile (refactor enabling differentiable
    coupled calibration of the multilayer land params)."""

    def _setup(self, z0_val=0.1):
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.soil_grid import SoilGridConfig
        from legoesm.land.multilayer_land import init_multilayer_land_state
        from legoesm.land.surface_params import LandSurfaceParams
        pipe, _ = _pipeline(land=True)
        ncol = 6 * N_CS * N_CS
        cfg = MultiLayerLandConfig(
            soil_grid=SoilGridConfig(n_layers=6, total_depth=2.0),
            bulk_scheme="most")           # MOST -> z0 matters
        c = lambda v: jnp.full(ncol, v)   # keep traced (z0 is the grad target)
        pipe.land_ml_cfg = cfg
        pipe.land_ml_lat = jnp.zeros(ncol)
        pipe.land_ml_params = LandSurfaceParams(
            albedo_veg=c(0.2), emissivity=c(0.97), z0=c(z0_val), W_max=c(150.0),
            C_soil=c(2.0e6), d_soil=c(1.0), root_depth=c(1.0), theta_wp=c(0.12),
            theta_fc=c(0.30), Vc_max25=c(50.0), LCMA=c(50.0), g1=c(9.0))
        land_ml = init_multilayer_land_state(ncol, cfg, T_init=285.0)
        s2 = (6, N_CS, N_CS); s3 = (*s2, NLEV)
        atm = dict(T=jnp.full(s3, 288.0), p_s=jnp.full(s2, 1.0e5),
                   q_v=jnp.full(s3, 0.006), u=jnp.full(s3, 4.0), v=jnp.zeros(s3))
        sw = jnp.full(ncol, 400.0); lw = jnp.full(ncol, 340.0)
        return pipe, land_ml, atm, sw, lw, ncol

    def test_step_returns_finite_columnar(self):
        pipe, land_ml, atm, sw, lw, ncol = self._setup()
        land_new, T_sfc, albedo = pipe._step_multilayer_land_tile(
            land_ml, sw, lw, atm["T"], atm["p_s"], atm["q_v"], atm["u"], atm["v"],
            None, 1800.0)
        assert T_sfc.shape == (ncol,) and jnp.all(jnp.isfinite(T_sfc))
        assert jnp.all((albedo > 0.0) & (albedo < 1.0))
        assert land_new.T_soil.shape == land_ml.T_soil.shape

    def test_differentiable_wrt_z0(self):
        """jax.grad of the tile's surface T w.r.t. the roughness z0 — the capability
        the slab-embedded T_land cannot provide (the refactor's point)."""
        _, _, atm, sw, lw, ncol = self._setup()

        def mean_Tsfc(z0_val):
            pipe, land_ml, _a, _sw, _lw, _ = self._setup(z0_val=z0_val)
            _, T_sfc, _ = pipe._step_multilayer_land_tile(
                land_ml, sw, lw, atm["T"], atm["p_s"], atm["q_v"], atm["u"],
                atm["v"], None, 1800.0)
            return jnp.mean(T_sfc)

        g = jax.grad(mean_Tsfc)(0.1)
        assert jnp.isfinite(g) and abs(float(g)) > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

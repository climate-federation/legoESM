"""Tests for the physically-realistic standard-atmosphere initial condition.

Direct unit tests of ``standard_atmosphere_temperature`` (the leaf module) plus
a driver-level wiring test that ``ic="standard"`` yields an Earth-like column
water vapour on the lat-lon finite-volume grid — in contrast to the
uniform-300 K rest state, which gives ~6x too much.
"""

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio
from legoesm.diagnostics.column_integrals import column_water_vapor
from legoesm.atmosphere.standard_atmosphere import (
    StandardAtmosphereConfig,
    standard_atmosphere_temperature,
    standard_atmosphere_zonal_wind,
)
from legoesm.atmosphere.held_suarez import held_suarez_init_latlon
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    cgrid_latlon_hydrostatic_tendencies, hydrostatic_to_cgrid,
    CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
)


def _q_v_from_T(T, sigma, p_s, rh_init=0.7):
    """Reproduce the driver's moisture init (model_driver._init_state)."""
    p_full = p_s[..., None] * sigma.sigma_full
    q_sat = saturation_mixing_ratio(T, p_full)
    return jnp.minimum(rh_init * q_sat * sigma.sigma_full ** 2, q_sat)


class TestStandardProfile:
    """Direct unit tests of the analytic temperature profile."""

    def test_shape_latlon(self):
        sigma = create_sigma_coordinate(n_levels=12)
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 16)[:, None] * jnp.ones((16, 32))
        T = standard_atmosphere_temperature(lat, sigma.sigma_full)
        assert T.shape == (16, 32, 12)

    def test_shape_column_and_cubedsphere(self):
        sigma = create_sigma_coordinate(n_levels=5)
        # 1-D column layout (MPAS-like)
        lat_col = jnp.linspace(-1.5, 1.5, 7)
        assert standard_atmosphere_temperature(lat_col, sigma.sigma_full).shape == (7, 5)
        # cubed-sphere-like (6, n, n)
        lat_cs = jnp.zeros((6, 4, 4))
        assert standard_atmosphere_temperature(lat_cs, sigma.sigma_full).shape == (6, 4, 4, 5)

    def test_equator_warmer_than_pole_at_surface(self):
        sigma = create_sigma_coordinate(n_levels=10)
        lat = jnp.array([0.0, jnp.pi / 2])  # equator, pole
        T = standard_atmosphere_temperature(lat, sigma.sigma_full)
        # Lowest (near-surface) level, highest sigma index.
        T_sfc = T[:, -1]
        assert float(T_sfc[0]) > float(T_sfc[1])
        # Default 40 K equator-pole contrast (sigma slightly < 1 at surface).
        assert 30.0 < float(T_sfc[0] - T_sfc[1]) <= 40.0

    def test_decreases_with_height(self):
        sigma = create_sigma_coordinate(n_levels=20)
        lat = jnp.array([0.0])
        T = standard_atmosphere_temperature(lat, sigma.sigma_full)[0]
        # sigma_full is ordered top->surface, so T should be non-decreasing along
        # the level axis (cold aloft, warm near surface).
        assert float(T[0]) <= float(T[-1])
        assert float(T[-1]) - float(T[0]) > 30.0

    def test_stratosphere_floor(self):
        cfg = StandardAtmosphereConfig(T_strato_K=216.0)
        sigma = create_sigma_coordinate(n_levels=40)
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 9)
        T = standard_atmosphere_temperature(lat, sigma.sigma_full, cfg)
        assert float(jnp.min(T)) >= 216.0 - 1e-6

    def test_finite_and_bounded(self):
        sigma = create_sigma_coordinate(n_levels=30)
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 48)[:, None] * jnp.ones((48, 96))
        T = standard_atmosphere_temperature(lat, sigma.sigma_full)
        assert bool(jnp.all(jnp.isfinite(T)))
        assert float(jnp.min(T)) > 150.0 and float(jnp.max(T)) < 330.0

    def test_cwv_is_earthlike_and_meridional(self):
        """The whole point: realistic column water vapour with a tropics>poles
        gradient, ~6x lower than the uniform-300 K column."""
        g = create_latlon_grid(n_lat=48, radius=constants.R_earth,
                               omega=constants.Omega)
        sigma = create_sigma_coordinate(n_levels=30)
        p_s = jnp.full((g.n_lat, g.n_lon), constants.p_ref)

        T_std = standard_atmosphere_temperature(jnp.asarray(g.lat2d), sigma.sigma_full)
        cwv_std = column_water_vapor(_q_v_from_T(T_std, sigma, p_s), p_s, sigma.dsigma)

        # Earth-like global mean (obs ~13-25 kg/m^2; this dry cold-start ~11).
        mean_std = float(jnp.mean(cwv_std))
        assert 5.0 < mean_std < 40.0, f"standard-IC CWV mean {mean_std:.1f} unrealistic"

        # Tropics much moister than poles.
        eq = g.n_lat // 2
        cwv_eq = float(jnp.mean(cwv_std[eq - 1:eq + 1, :]))
        cwv_pole = float(jnp.mean(cwv_std[:2, :]))
        assert cwv_eq > cwv_pole + 5.0

        # Dramatically drier than the uniform-300 K column it replaces.
        T_uniform = jnp.full(T_std.shape, 300.0)
        cwv_uniform = column_water_vapor(
            _q_v_from_T(T_uniform, sigma, p_s), p_s, sigma.dsigma)
        assert float(jnp.mean(cwv_uniform)) > 3.0 * mean_std


class TestStandardZonalWind:
    """Thermal-wind-balanced zonal wind."""

    def _u(self, n_lat=48):
        g = create_latlon_grid(n_lat=n_lat, radius=constants.R_earth,
                               omega=constants.Omega)
        sigma = create_sigma_coordinate(n_levels=30)
        u = standard_atmosphere_zonal_wind(
            jnp.asarray(g.lat2d), sigma.sigma_full,
            constants.R_earth, constants.Omega)
        return g, sigma, u

    def test_shape_and_finite(self):
        g, sigma, u = self._u()
        assert u.shape == (g.n_lat, g.n_lon, sigma.n_levels)
        assert bool(jnp.all(jnp.isfinite(u)))

    def test_surface_wind_near_zero(self):
        # No troposphere integrated at the surface -> zero surface jet.
        g, sigma, u = self._u()
        assert float(jnp.max(jnp.abs(u[:, :, -1]))) < 1.0

    def test_realistic_jet_magnitude(self):
        # Mid-latitude westerly jet of order tens of m/s (not ~130 m/s).
        _, _, u = self._u()
        umax = float(jnp.max(jnp.abs(u)))
        assert 10.0 < umax < 60.0, f"jet max {umax:.1f} m/s unrealistic"
        assert float(jnp.min(u)) >= -1.0, "no spurious strong easterlies"

    def test_equatorial_taper_and_midlat_peak(self):
        # Weak winds at the equator (geostrophy invalid there); peak off-equator.
        g, _, u = self._u()
        latdeg = np.degrees(np.asarray(g.lat))
        utop = np.asarray(u[:, 0, 0])
        i_eq = int(np.argmin(np.abs(latdeg)))
        i_peak = int(np.argmax(np.abs(utop)))
        assert abs(utop[i_eq]) < 5.0, "equatorial jet not tapered"
        assert abs(latdeg[i_peak]) > 12.0, "jet peak should be off-equator"
        assert abs(utop[i_peak]) > abs(utop[i_eq]) + 10.0

    def test_finite_winds_at_cold_or_negative_pole_Tsfc(self):
        """A small T_sfc_equator drives the pole surface temperature toward (or
        below) zero; the fractional tropopause power must not NaN."""
        sigma = create_sigma_coordinate(n_levels=20)
        lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 19)
        for T_eq in (50.0, 40.0, 30.0):  # pole T_sfc = 10, 0, -10
            cfg = StandardAtmosphereConfig(T_sfc_equator_K=T_eq)
            u = standard_atmosphere_zonal_wind(
                lat, sigma.sigma_full, constants.R_earth, constants.Omega, cfg)
            assert bool(jnp.all(jnp.isfinite(u))), f"NaN wind at T_eq={T_eq}"

    def test_balanced_wind_reduces_startup_imbalance(self):
        """The balanced zonal wind must cut the cold-start meridional
        pressure-gradient tendency relative to a zero-wind standard IC."""
        g = create_latlon_grid(n_lat=48, radius=constants.R_earth,
                               omega=constants.Omega)
        sigma = create_sigma_coordinate(n_levels=30)
        T = standard_atmosphere_temperature(jnp.asarray(g.lat2d), sigma.sigma_full)
        u = standard_atmosphere_zonal_wind(
            jnp.asarray(g.lat2d), sigma.sigma_full,
            constants.R_earth, constants.Omega)
        st = held_suarez_init_latlon(g, sigma, T_init=300.0)

        def max_dv(state):
            _, dv, _, _, _ = cgrid_latlon_hydrostatic_tendencies(
                hydrostatic_to_cgrid(state, g), g, sigma)
            return float(jnp.max(jnp.abs(dv)))

        unbal = max_dv(st._replace(T=st.T.replace(data=T)))
        bal = max_dv(st._replace(
            T=st.T.replace(data=T),
            u=st.u.replace(data=jnp.broadcast_to(u, st.u.data.shape))))
        assert bal < unbal, "balanced wind did not reduce dv/dt"
        assert bal < 0.5 * unbal, f"weak balance: {unbal:.2e}->{bal:.2e}"

    def test_startup_no_blowup(self):
        """Cold start from the balanced standard IC must not blow up: step the
        lat-lon FV dycore and require finite, bounded winds (guards the
        geostrophic-adjustment shock the balanced wind is meant to tame).
        Manual 15-day res-48 + polar-filter run confirms the production case;
        this is the cheap in-CI regression guard."""
        g = create_latlon_grid(n_lat=24, radius=constants.R_earth,
                               omega=constants.Omega)
        sigma = create_sigma_coordinate(n_levels=10)
        T = standard_atmosphere_temperature(jnp.asarray(g.lat2d), sigma.sigma_full)
        u = standard_atmosphere_zonal_wind(
            jnp.asarray(g.lat2d), sigma.sigma_full,
            constants.R_earth, constants.Omega)
        st = held_suarez_init_latlon(g, sigma, T_init=300.0)
        st = st._replace(
            T=st.T.replace(data=T),
            u=st.u.replace(data=jnp.broadcast_to(u, st.u.data.shape)))

        model = CGridLatLonPrimitiveEquationModel(
            g, sigma, CGridLatLonPrimitiveEquationConfig(fix_mass=True), dt=300.0)
        for _ in range(20):
            st = model.step_with_physics(st, 300.0)
        assert bool(jnp.all(jnp.isfinite(st.T.data)))
        assert bool(jnp.all(jnp.isfinite(st.u.data)))
        # Winds stay physical — no runaway adjustment shock.
        assert float(jnp.max(jnp.abs(st.u.data))) < 80.0
        assert float(jnp.max(jnp.abs(st.v.data))) < 80.0


class TestDriverStandardIC:
    """Driver-level wiring: ic='standard' on the lat-lon FV grid."""

    def _build(self, ic, topography="flat"):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        from legoesm.driver.model_driver import ModelDriver
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=24, nlev=30),
            dycore=DycoreConfig(model_type="hydrostatic",
                                discretization="finite_volume", dt=300.0),
            dataset="analytical", radiation="gray", ic=ic,
            topography=topography,
        )
        # Per-call unique output dir: ModelDriver's default path is a
        # wall-clock-timestamped results/amip/... folder, so two _build calls in
        # the same second collide on the run-manifest provenance guard (and would
        # pollute the shared results/ tree). Isolate each driver.
        drv = ModelDriver(cfg, output_dir=tempfile.mkdtemp(prefix="std_ic_"))
        drv.setup()
        return drv

    def test_pressure_consistent_over_topography_via_driver(self):
        """End-to-end: with non-flat (gaussian) topography the driver's
        ic='standard' surface pressure must match the analytic hydrostatic
        reduction against the STANDARD surface temperature,
        p_s = p_ref*exp(-phis/(R_d*T_sfc(lat))) — not the uniform T_init the
        scaffold used, and not a flat p_s over terrain."""
        drv = self._build("standard", topography="gaussian")
        phis = np.asarray(drv.state.phis.data)
        p_s = np.asarray(drv.state.p_s.data)
        lat2d = np.asarray(drv.grid.lat2d)
        assert float(np.max(phis)) > 1.0e4, "gaussian topo gave no mountain"

        T_sfc = np.asarray(standard_atmosphere_temperature(
            jnp.asarray(lat2d), jnp.ones((1,)),
            StandardAtmosphereConfig(T_sfc_equator_K=300.0))[..., 0])
        p_s_expected = constants.p_ref * np.exp(-phis / (constants.R_d * T_sfc))

        # Over the mountain (where the bug was a ~33% over-pressure).
        mask = phis > 0.2 * float(np.max(phis))
        assert np.allclose(p_s[mask], p_s_expected[mask], rtol=1e-4), (
            f"p_s over terrain {p_s[mask].mean():.0f} != analytic "
            f"{p_s_expected[mask].mean():.0f}")
        # Flat region stays at the reference pressure.
        assert np.allclose(p_s[phis < 1e-6], constants.p_ref, rtol=1e-6)
        # Mountain sits off-equator, so the scale T is genuinely below T_init.
        i_mtn = np.unravel_index(int(np.argmax(phis)), phis.shape)
        assert float(T_sfc[i_mtn]) < 299.0

    def test_ic_uses_physics_pressure_convention_hybrid_topography(self):
        """The IC temperature/moisture must be built on the SAME pressure grid
        the production physics uses (p_full = p_s*sigma_full), so a hybrid+
        topography run does not start with the IC on one pressure grid and the
        first radiation/convection/saturation step on another.

        Over a mountain (hybrid), the model-convention grid p_s*sigma_full and
        the "true" hybrid pressure A*p_ref+B*p_s differ materially; the IC must
        track the former (what physics_pipeline / compiled_segments evaluate)."""
        from legoesm.thermo import saturation_mixing_ratio
        drv = self._build("standard", topography="gaussian")
        assert type(drv.sigma).__name__ == "HybridSigmaPressureCoordinate"
        phis = np.asarray(drv.state.phis.data)
        p_s = np.asarray(drv.state.p_s.data)
        T = np.asarray(drv.state.T.data)
        qv = np.asarray(drv.tracers["q_v"])
        lat2d = np.asarray(drv.grid.lat2d)
        sig_full = np.asarray(drv.sigma.sigma_full)
        i = np.unravel_index(int(np.argmax(phis)), phis.shape)  # mountain top

        sa = StandardAtmosphereConfig(T_sfc_equator_K=300.0)
        # (1) Temperature follows the sigma_full (physics-convention) profile,
        #     NOT the "true" hybrid local ratio (which would differ over terrain
        #     and disagree with the physics grid).
        T_physics = np.asarray(standard_atmosphere_temperature(
            jnp.asarray(lat2d[i]), jnp.asarray(sig_full), sa))
        sig_local = (np.asarray(drv.sigma.pressure_at_full(drv.state.p_s.data))[i]
                     / p_s[i])
        T_local = np.asarray(standard_atmosphere_temperature(
            jnp.asarray(lat2d[i]), jnp.asarray(sig_local), sa))
        assert np.allclose(T[i], T_physics, atol=1e-2)
        # The two conventions genuinely differ over terrain (so the test
        # distinguishes which one the IC uses).
        assert float(np.max(np.abs(T_physics - T_local))) > 2.0

        # (2) Moisture is consistent on the physics grid: RH = q_v/q_sat(T,
        #     p_s*sigma_full) <= RH_init and never supersaturated over the
        #     mountain — so the first physics step sees a consistent column.
        p_full_phys = p_s[i][..., None] * sig_full  # (nlev,)
        q_sat_phys = np.asarray(saturation_mixing_ratio(
            jnp.asarray(T[i]), jnp.asarray(p_full_phys)))
        assert np.all(np.isfinite(qv[i]))
        assert np.all(qv[i] <= q_sat_phys + 1e-9)
        rh = qv[i] / np.maximum(q_sat_phys, 1e-12)
        assert float(np.max(rh)) <= 0.7 + 1e-3

    def test_standard_ic_realistic_cwv_and_gradient(self):
        drv = self._build("standard")
        T = drv.state.T.data            # (n_lat, n_lon, nlev)
        n_lat = T.shape[0]

        # Meridional surface gradient: equator warmer than pole.
        eq = n_lat // 2
        assert float(jnp.mean(T[eq - 1:eq + 1, :, -1])) > \
            float(jnp.mean(T[:2, :, -1])) + 10.0
        # Cold stratosphere aloft.
        assert float(jnp.min(T)) < 230.0

        cwv = column_water_vapor(
            drv.tracers["q_v"], drv.state.p_s.data, drv.sigma.dsigma)
        assert 5.0 < float(jnp.mean(cwv)) < 40.0

        # Balanced jet wired into the state (not left at rest).
        umax = float(jnp.max(jnp.abs(drv.state.u.data)))
        assert 10.0 < umax < 60.0, f"standard-IC jet {umax:.1f} m/s not wired"

    def _build_cube(self, ic):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        from legoesm.driver.model_driver import ModelDriver
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=20),
            dycore=DycoreConfig(model_type="hydrostatic",
                                discretization="cdgrid", dt=300.0),
            dataset="analytical", radiation="gray", ic=ic,
        )
        # Unique output dir (see _build): avoid same-second run-manifest
        # collisions and shared-results pollution under the full suite.
        drv = ModelDriver(cfg, output_dir=tempfile.mkdtemp(prefix="std_ic_cube_"))
        drv.setup()
        return drv

    def test_standard_ic_on_cube_earthlike_lapse_and_cwv(self):
        """ic='standard' on the cubed-sphere applies the GRID-AGNOSTIC realistic
        T / p_s overlay (cold stratosphere + equator-pole surface gradient +
        Earth-like CWV) — the same fix that on the uniform ic='default' scaffold
        was missing (isothermal ~300 K column, CWV ~80).  The balanced jet is
        skipped on the cube (cube-local winds), which must NOT crash."""
        drv = self._build_cube("standard")
        T = drv.state.T.data            # (6, n, n, nlev)
        assert bool(jnp.all(jnp.isfinite(T)))
        # Cold stratosphere aloft — NOT the isothermal ~300 K default scaffold.
        assert float(jnp.min(T)) < 230.0
        # A real lapse rate: surface much warmer than the column top.
        assert float(jnp.mean(T[..., -1])) > float(jnp.mean(T[..., 0])) + 30.0
        # Imposed equator-pole surface gradient (uses geographic cube latitude).
        lat = np.asarray(drv.grid.lat)          # (6, n, n)
        Tsfc = np.asarray(T[..., -1])
        eq = np.abs(lat) < np.deg2rad(15.0)
        pole = np.abs(lat) > np.deg2rad(60.0)
        assert Tsfc[eq].mean() > Tsfc[pole].mean() + 8.0
        # Earth-like CWV (the uniform ic='default' scaffold gave ~80 kg/m^2).
        cwv = column_water_vapor(
            drv.tracers["q_v"], drv.state.p_s.data, drv.sigma.dsigma)
        assert 5.0 < float(jnp.mean(cwv)) < 40.0

    def test_standard_ic_cube_accepted_by_validation(self):
        """validate_strict ACCEPTS ic='standard' on cubed_sphere (the extension);
        the T/p_s overlay is grid-agnostic and the jet is skipped."""
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=20),
            dycore=DycoreConfig(model_type="hydrostatic",
                                discretization="cdgrid", dt=300.0),
            ic="standard",
        )
        cfg.validate_strict()   # must not raise

    def test_standard_rejected_for_spectral_at_validation(self):
        """ic='standard' must fail fast in validate_strict for spectral/Gaussian
        (SpectralHydrostaticState has T_hat, no grid-space T to override) — not
        crash mid-setup."""
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="gaussian", resolution=21, nlev=10),
            dycore=DycoreConfig(model_type="hydrostatic",
                                discretization="spectral", dt=600.0),
            ic="standard",
        )
        with pytest.raises(ValueError, match="grid_type"):
            cfg.validate_strict()

    def test_standard_accepted_for_latlon(self):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="latlon", resolution=16, nlev=10),
            dycore=DycoreConfig(model_type="hydrostatic",
                                discretization="finite_volume", dt=300.0),
            ic="standard",
        )
        cfg.validate_strict()  # must not raise

    def test_standard_accepted_for_cubed_sphere(self):
        # ic='standard' on cubed_sphere is ACCEPTED: the grid-agnostic realistic
        # T/p_s overlay is applied; the balanced jet is skipped because cube u/v
        # are cube-local components (config.py validate_strict accepts both
        # 'latlon' and 'cubed_sphere'). The former test_standard_rejected_for_
        # cubed_sphere was removed when the cube path was wired — it asserted a
        # contract that contradicts the shipped behaviour.
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=16, nlev=10),
            dycore=DycoreConfig(model_type="hydrostatic",
                                discretization="cdgrid", dt=300.0),
            ic="standard",
        )
        cfg.validate_strict()  # must not raise

    def test_standard_rejects_unphysical_t_init(self):
        from legoesm.driver.config import (
            ExperimentConfig, GridConfig, DycoreConfig,
        )
        for bad in (30.0, 400.0):
            cfg = ExperimentConfig(
                grid=GridConfig(grid_type="latlon", resolution=16, nlev=10),
                dycore=DycoreConfig(model_type="hydrostatic",
                                    discretization="finite_volume", dt=300.0),
                ic="standard", T_init=bad,
            )
            with pytest.raises(ValueError, match="equator surface temperature"):
                cfg.validate_strict()

    def test_standard_drier_than_default(self):
        cwv_std = column_water_vapor(
            (d := self._build("standard")).tracers["q_v"],
            d.state.p_s.data, d.sigma.dsigma)
        cwv_def = column_water_vapor(
            (d2 := self._build("default")).tracers["q_v"],
            d2.state.p_s.data, d2.sigma.dsigma)
        # Uniform-300 K default holds far more column water vapour.
        assert float(jnp.mean(cwv_def)) > 2.0 * float(jnp.mean(cwv_std))

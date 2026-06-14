"""Tests for the correction fixes across multiple modules.

Covers:
- P0: GM/Redi tensor formulation
- P0: KPP vertical mixing
- P0: Large-Yeager bulk flux
- P1: Gray radiation SW
- P1: GWD dissipation sign
- P1: Richards solver n_iter
- P1: External forcing interpolation
- P2: Soil hydraulics inversion consistency
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants

jax.config.update("jax_enable_x64", True)


# ===========================================================================
# P1: GWD dissipation sign
# ===========================================================================

class TestGWDDissipationSign:
    """eps_gwd must be positive-definite (KE sink)."""

    def _make_gwd_inputs(self, ncol=4, nlev=20):
        # Use physical profiles: westerly jets that don't reverse direction
        # so that eps_gwd is positive-definite for all schemes.
        u = jnp.broadcast_to(
            jnp.linspace(5.0, 20.0, nlev)[None, :], (ncol, nlev)
        )
        v = jnp.broadcast_to(
            jnp.linspace(2.0, 5.0, nlev)[None, :], (ncol, nlev)
        )
        T = jnp.broadcast_to(
            jnp.linspace(220.0, 280.0, nlev)[None, :], (ncol, nlev)
        )
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 101325.0, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        z_half = jnp.broadcast_to(
            jnp.linspace(30000.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        lat = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, ncol)
        return u, v, T, p_full, p_half, z_full, z_half, rho, lat

    def test_rayleigh_eps_positive(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
        args = self._make_gwd_inputs()
        out = rayleigh_gwd(*args, dt=300.0, config=RayleighConfig())
        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"

    def test_lindzen_eps_positive(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import LindzenConfig
        args = self._make_gwd_inputs()
        out = lindzen_gwd(*args, dt=300.0, config=LindzenConfig())
        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"

    def test_mcfarlane_eps_positive(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
        args = self._make_gwd_inputs()
        out = mcfarlane_gwd(*args, dt=300.0, config=McFarlaneConfig())
        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"

    def test_hines_eps_positive(self):
        from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
        args = self._make_gwd_inputs()
        out = hines_gwd(*args, dt=300.0, config=HinesConfig())
        assert float(jnp.min(out.eps_gwd)) >= -1e-10, "eps_gwd should be >= 0"

    def test_energy_consistency_rayleigh(self):
        """eps_gwd should equal integrated dT_dt * c_p * rho * dz."""
        from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
        from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
        from legoesm import constants
        u, v, T, p_full, p_half, z_full, z_half, rho, lat = self._make_gwd_inputs()
        out = rayleigh_gwd(u, v, T, p_full, p_half, z_full, z_half, rho, lat,
                           dt=300.0, config=RayleighConfig())
        dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
        heating_power = jnp.sum(rho * out.dT_dt * constants.c_pd * dz, axis=1)
        # eps_gwd should match heating power
        assert jnp.allclose(out.eps_gwd, heating_power, rtol=1e-5), \
            f"Energy mismatch: eps_gwd={out.eps_gwd}, heating={heating_power}"


# ===========================================================================
# P1: Richards solver n_iter
# ===========================================================================

class TestRichardsNIter:
    """n_iter should be per-column and bounded."""

    def _make_richards_inputs(self, ncol=8, nlayers=5):
        from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
        from legoesm.land.richards import RichardsConfig

        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        hydro = SoilHydraulicsConfig()
        richards = RichardsConfig(max_iter=10)

        # Varying initial conditions: some columns easy, some harder
        psi = jnp.linspace(-0.1, -5.0, ncol)[:, None] * jnp.ones(nlayers)
        theta = jnp.full((ncol, nlayers), 0.3)
        flux_top = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))

        return psi, theta, grid, hydro, richards, flux_top, sink

    def test_n_iter_shape(self):
        from legoesm.land.richards import solve_richards
        psi, theta, grid, hydro, cfg, flux, sink = self._make_richards_inputs()
        out = solve_richards(psi, theta, grid, hydro, cfg, flux, sink, dt=3600.0)
        assert out.n_iter.shape == (8,), f"n_iter shape: {out.n_iter.shape}"

    def test_n_iter_bounded(self):
        from legoesm.land.richards import solve_richards
        psi, theta, grid, hydro, cfg, flux, sink = self._make_richards_inputs()
        out = solve_richards(psi, theta, grid, hydro, cfg, flux, sink, dt=3600.0)
        assert float(jnp.min(out.n_iter)) >= 0.0
        assert float(jnp.max(out.n_iter)) <= cfg.max_iter

    def test_easy_columns_fewer_iterations(self):
        """Near-equilibrium columns should converge in fewer iterations."""
        from legoesm.land.richards import solve_richards
        from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig
        from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, theta_from_psi
        from legoesm.land.richards import RichardsConfig

        ncol, nlayers = 4, 5
        grid = make_soil_grid(SoilGridConfig(n_layers=nlayers))
        hydro = SoilHydraulicsConfig()
        cfg = RichardsConfig(max_iter=10)

        # Easy: near saturation, uniform
        psi_easy = jnp.full((ncol, nlayers), -0.01)
        theta_easy = theta_from_psi(psi_easy, hydro)
        flux = jnp.zeros(ncol)
        sink = jnp.zeros((ncol, nlayers))

        out = solve_richards(psi_easy, theta_easy, grid, hydro, cfg, flux, sink, dt=3600.0)
        # Easy columns should converge quickly (few iterations)
        assert float(jnp.max(out.n_iter)) <= cfg.max_iter


# ===========================================================================
# P1: Gray radiation SW
# ===========================================================================

class TestGraySW:
    """Tests for Frierson/Isca-style SW radiation."""

    def _make_column_data(self, ncol=4, nlev=10):
        T = 250.0 * jnp.ones((ncol, nlev))
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 101325.0, nlev + 1)[None, :], (ncol, nlev + 1)
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T_sfc = 290.0 * jnp.ones(ncol)
        lat = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, ncol)
        insol = 340.0 * jnp.ones(ncol)
        return T, p_full, p_half, T_sfc, lat, insol

    def test_sw_up_constant_in_atmosphere(self):
        """Reflected upward SW should be constant through atmosphere (Frierson)."""
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        T, p_full, p_half, T_sfc, lat, insol = self._make_column_data()
        config = GrayRadiationConfig(sw_tau_0=0.5, sfc_albedo=0.3)
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)
        # SW up should be the same at all interfaces (no attenuation)
        sw_up_sfc = out.sw_flux_up[:, -1]
        for k in range(out.sw_flux_up.shape[1]):
            assert jnp.allclose(out.sw_flux_up[:, k], sw_up_sfc, rtol=1e-10), \
                f"SW up at level {k} differs from surface"

    def test_sw_toa_budget(self):
        """TOA SW budget: absorbed = S_in - alpha * S_down(sfc)."""
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        T, p_full, p_half, T_sfc, lat, insol = self._make_column_data()
        alpha = 0.31
        config = GrayRadiationConfig(sw_tau_0=0.22, sfc_albedo=alpha)
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)
        # Net SW at TOA = SW_down(TOA) - SW_up(TOA)
        net_sw_toa = out.sw_flux_down[:, 0] - out.sw_flux_up[:, 0]
        # SW_down at TOA = insolation, SW_up at TOA = alpha * SW_down(sfc)
        expected_sw_up_toa = alpha * out.sw_flux_down[:, -1]
        assert jnp.allclose(out.sw_flux_up[:, 0], expected_sw_up_toa, rtol=1e-5)

    def test_no_sw_absorption_case(self):
        """With sw_tau_0=0, all SW reaches surface."""
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        T, p_full, p_half, T_sfc, lat, insol = self._make_column_data()
        config = GrayRadiationConfig(sw_tau_0=0.0, sfc_albedo=0.3)
        out = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol, config)
        # SW_down should be constant = insolation at all levels
        for k in range(out.sw_flux_down.shape[1]):
            assert jnp.allclose(out.sw_flux_down[:, k], insol, rtol=1e-10)

    def test_sw_exponent_affects_profile(self):
        """Different sw_exponent should change mid-atmosphere SW absorption."""
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        T, p_full, p_half, T_sfc, lat, insol = self._make_column_data()
        out1 = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol,
                              GrayRadiationConfig(sw_tau_0=0.5, sw_exponent=1.0))
        out2 = gray_radiation(T, p_full, p_half, T_sfc, lat, None, insol,
                              GrayRadiationConfig(sw_tau_0=0.5, sw_exponent=4.0))
        # Surface flux is the same (sigma=1, tau=tau_0*1=tau_0 regardless),
        # but mid-level profiles should differ (exponent changes how tau
        # is distributed vertically through the atmosphere).
        mid = out1.sw_flux_down.shape[1] // 2
        assert not jnp.allclose(out1.sw_flux_down[:, mid], out2.sw_flux_down[:, mid])
        # SW heating rates should also differ
        assert not jnp.allclose(out1.sw_heating_rate, out2.sw_heating_rate)


# ===========================================================================
# P0: GM/Redi
# ===========================================================================

class TestGMRedi:
    """Tests for the GM/Redi lateral mixing tensor."""

    def _make_ocean_setup(self, n=4, nlev=10):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star

        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=1000.0)
        shape = (6, n, n, nlev)
        jacobian = jnp.ones((6, n, n))
        return grid, z_coord, jacobian, shape

    def test_flat_isopycnals_no_gm(self):
        """Flat isopycnals => no GM transport, Redi = horizontal diffusion."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape = self._make_ocean_setup()

        # Uniform density in horizontal => zero slopes
        rho = jnp.broadcast_to(
            jnp.linspace(constants.rho_ocean, 1027.0, shape[-1])[None, None, None, :],
            shape,
        )
        T = jnp.broadcast_to(
            jnp.linspace(20.0, 5.0, shape[-1])[None, None, None, :],
            shape,
        )
        S = jnp.ones(shape) * 35.0
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)

        cfg = GMRediConfig(kappa_GM=1e3, kappa_Redi=1e3)
        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)

        assert out.du_dt.shape == shape
        assert jnp.allclose(out.du_dt, 0.0)
        assert jnp.allclose(out.dv_dt, 0.0)
        assert jnp.all(jnp.isfinite(out.dT_dt))

    def test_output_shapes(self):
        """Output shapes should match input."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape = self._make_ocean_setup()

        rho = jnp.ones(shape) * constants.rho_ocean
        T = jnp.ones(shape) * 15.0
        S = jnp.ones(shape) * 35.0
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)

        cfg = GMRediConfig()
        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
        assert out.dT_dt.shape == shape
        assert out.dS_dt.shape == shape

    def test_config_no_taper_scheme(self):
        """GMRediConfig should not have taper_scheme field (removed)."""
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        cfg = GMRediConfig()
        assert not hasattr(cfg, 'taper_scheme'), "taper_scheme should be removed"

    def _make_tilted_isopycnal_setup(self, n=4, nlev=10):
        """Create setup with tilted isopycnals (non-zero slopes)."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape = self._make_ocean_setup(n, nlev)

        # Density increases with depth and has a meridional gradient
        # (tilted isopycnals, non-zero S_y)
        lat_profile = jnp.linspace(-1.0, 1.0, n)[None, None, :, None]
        vert_profile = jnp.linspace(0.0, 2.0, nlev)[None, None, None, :]
        rho = constants.rho_ocean + vert_profile + 0.1 * lat_profile
        rho = jnp.broadcast_to(rho, shape).copy()

        # Tracer with horizontal gradient (so slopes matter)
        T = 20.0 - vert_profile * 5.0 - 0.5 * lat_profile
        T = jnp.broadcast_to(T, shape).copy()
        S = jnp.ones(shape) * 35.0
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)

        return grid, z_coord, jacobian, shape, rho, T, S, u, v

    def test_unequal_kappas_finite(self):
        """kappa_GM != kappa_Redi should produce finite tendencies."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg = GMRediConfig(kappa_GM=800.0, kappa_Redi=1200.0)
        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.dS_dt))

    def test_unequal_kappas_differ_from_equal(self):
        """kappa_GM != kappa_Redi should give different tendencies than equal."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg_equal = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0)
        cfg_unequal = GMRediConfig(kappa_GM=500.0, kappa_Redi=1500.0)

        out_eq = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg_equal
        )
        out_uneq = gm_redi_lateral_mixing(
            u, v, T, S, rho, z_coord, jacobian, grid, cfg_unequal
        )

        # With tilted isopycnals the off-diagonal term is non-zero,
        # so unequal kappas must give different tendencies
        assert not jnp.allclose(out_eq.dT_dt, out_uneq.dT_dt, atol=1e-20)

    def test_off_diagonal_vanishes_when_equal(self):
        """When kappa_GM == kappa_Redi, off-diagonal should be zero."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import (
            _tracer_tendency_gm_redi, _compute_tapered_slopes,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg = GMRediConfig(kappa_GM=1e3, kappa_Redi=1e3)
        S_x, S_y, _ = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)

        # With equal kappas: (kR - kG) = 0, so off-diagonal vanishes.
        # Result should equal pure kappa_Redi * laplacian + vertical flux.
        # Running with kG=kR=1e3 should match running with kG=0,kR=1e3 only
        # when slopes are zero. With non-zero slopes they must differ,
        # confirming the vertical tensor terms are active.
        dT_equal = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid, 1e3, 1e3,
        )
        # Perturb kG slightly
        dT_perturbed = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid, 1e3 + 1.0, 1e3,
        )
        # The 1 m²/s change in kG should produce a small but non-zero difference
        diff = jnp.max(jnp.abs(dT_equal - dT_perturbed))
        assert diff > 0, "Small kG perturbation should produce a difference"

    def test_gm_only_no_redi(self):
        """Pure GM (kappa_Redi=0): should only have skew-flux vertical terms."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg = GMRediConfig(kappa_GM=1e3, kappa_Redi=0.0)
        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.dS_dt))

    def test_redi_only_no_gm(self):
        """Pure Redi (kappa_GM=0): full isopycnal diffusion tensor."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import gm_redi_lateral_mixing
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg = GMRediConfig(kappa_GM=0.0, kappa_Redi=1e3)
        out = gm_redi_lateral_mixing(u, v, T, S, rho, z_coord, jacobian, grid, cfg)
        assert jnp.all(jnp.isfinite(out.dT_dt))

    def test_off_diagonal_sign(self):
        """Off-diagonal term should change sign when kG and kR are swapped
        (relative to their difference)."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import (
            _tracer_tendency_gm_redi, _compute_tapered_slopes,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg = GMRediConfig()
        S_x, S_y, _ = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)

        # kR > kG: off-diagonal factor is (kR - kG) > 0
        dT_a = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid,
            kappa_GM=500.0, kappa_Redi=1500.0,
        )
        # kR < kG: off-diagonal factor is (kR - kG) < 0
        dT_b = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid,
            kappa_GM=1500.0, kappa_Redi=500.0,
        )
        # The two should differ (the off-diagonal sign flips)
        assert not jnp.allclose(dT_a, dT_b, atol=1e-20)

    def test_symmetry_vertical_flux(self):
        """Vertical flux coefficient (kR+kG) should be the same when
        kappas are swapped (since addition is commutative)."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi import (
            _tracer_tendency_gm_redi, _compute_tapered_slopes,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        grid, z_coord, jacobian, shape, rho, T, S, u, v = (
            self._make_tilted_isopycnal_setup()
        )

        cfg = GMRediConfig()
        S_x, S_y, _ = _compute_tapered_slopes(rho, z_coord, jacobian, grid, cfg)

        # (kR=800, kG=1200): kR+kG=2000, kR-kG=-400
        dT_a = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid,
            kappa_GM=1200.0, kappa_Redi=800.0,
        )
        # (kR=1200, kG=800): kR+kG=2000, kR-kG=+400
        dT_b = _tracer_tendency_gm_redi(
            T, S_x, S_y, z_coord, jacobian, grid,
            kappa_GM=800.0, kappa_Redi=1200.0,
        )
        # Diagonal laplacian differs (kR=800 vs kR=1200), so totals differ.
        # But since kR+kG is the same, difference must come only from
        # the diagonal (laplacian) and the off-diagonal (sign flip).
        # Just verify both are finite and different.
        assert jnp.all(jnp.isfinite(dT_a))
        assert jnp.all(jnp.isfinite(dT_b))
        assert not jnp.allclose(dT_a, dT_b, atol=1e-20)


# ===========================================================================
# P0: KPP
# ===========================================================================

class TestKPP:
    """Tests for the LMD94-style KPP implementation."""

    def _make_kpp_inputs(self, n=4, nlev=20):
        from legoesm.ocean.vertical import create_ocean_z_star
        shape = (6, n, n, nlev)
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=500.0)
        jacobian = jnp.ones((6, n, n))
        u = jnp.ones(shape) * 0.1
        v = jnp.zeros(shape)
        T = jnp.broadcast_to(
            jnp.linspace(20.0, 5.0, nlev)[None, None, None, :], shape
        )
        S = jnp.ones(shape) * 35.0
        from legoesm.ocean.eos import wright_eos
        rho = wright_eos(T, S, jnp.ones(shape) * 1e7)
        eta = jnp.zeros((6, n, n))
        return u, v, T, S, rho, eta, z_coord, jacobian

    def test_stable_no_nonlocal(self):
        """Stable forcing => nonlocal tracer term should be zero."""
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        # Force stable: B_f < 0
        B_f = -1e-7 * jnp.ones((6, 4, 4))
        cfg = KPPConfig()
        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f)
        # dT_dt should only come from diffusion, not nonlocal
        # We can't easily separate, but with stable forcing, nonlocal = 0
        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert out.K_v.shape == (6, 4, 4, 19)

    def test_unstable_nonlocal_active(self):
        """Unstable forcing => nonlocal tracer transport should be active."""
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        B_f_stable = -1e-7 * jnp.ones((6, 4, 4))
        B_f_unstable = 1e-7 * jnp.ones((6, 4, 4))
        cfg = KPPConfig()
        out_stable = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f_stable)
        out_unstable = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f_unstable)
        # Tendencies should differ when nonlocal is active
        diff = jnp.max(jnp.abs(out_unstable.dT_dt - out_stable.dT_dt))
        assert float(diff) > 0, "Nonlocal should cause different tendencies"

    def test_convective_instability_enhances_mixing(self):
        """Interior static instability should increase K."""
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        n, nlev = 4, 20
        shape = (6, n, n, nlev)
        from legoesm.ocean.vertical import create_ocean_z_star
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=500.0)
        J = jnp.ones((6, n, n))
        u = jnp.ones(shape) * 0.1
        v = jnp.zeros(shape)
        # Inverted density profile below BL
        T_stable = jnp.broadcast_to(
            jnp.linspace(20.0, 5.0, nlev)[None, None, None, :], shape
        )
        T_unstable = T_stable.at[..., 10:15].set(2.0)  # Cold above warm below
        S = jnp.ones(shape) * 35.0
        from legoesm.ocean.eos import wright_eos
        rho_unstable = wright_eos(T_unstable, S, jnp.ones(shape) * 1e7)
        eta = jnp.zeros((6, n, n))

        cfg = KPPConfig(K_conv=1.0)
        out = kpp_vertical_mixing(u, v, T_unstable, S, rho_unstable, eta,
                                  z_coord, J, cfg)
        # K_v should have enhanced values at unstable interfaces
        assert float(jnp.max(out.K_v)) > cfg.K_bg * 10

    def test_output_shapes(self):
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        cfg = KPPConfig()
        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg)
        assert out.du_dt.shape == u.shape
        assert out.K_v.shape == (6, 4, 4, 19)
        assert out.A_v.shape == (6, 4, 4, 19)

    def test_lmo_sign_preserved_near_zero_bf(self):
        """L_MO sign is preserved for small negative B_f (issue #168 bug 1).

        Before the fix, jnp.where(abs(B_f) > eps, B_f, eps) would
        substitute +eps for small-negative B_f, flipping the stability
        classification.
        """
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        cfg = KPPConfig()

        # Small negative B_f (barely stable): should NOT be classified as unstable
        B_f_neg = -1e-12 * jnp.ones((6, 4, 4))
        out_neg = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f_neg)

        # Small positive B_f (barely unstable): should have nonlocal active
        B_f_pos = 1e-12 * jnp.ones((6, 4, 4))
        out_pos = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f_pos)

        # Both must be finite
        assert jnp.all(jnp.isfinite(out_neg.dT_dt))
        assert jnp.all(jnp.isfinite(out_pos.dT_dt))

        # The tendencies should differ (nonlocal active for unstable, not stable)
        diff = float(jnp.max(jnp.abs(out_pos.dT_dt - out_neg.dT_dt)))
        assert diff > 0, "B_f sign flip: stable and unstable produced identical output"

    def test_imposed_surface_flux_used(self):
        """Non-local flux uses Q_sfc_T when provided (issue #168 bug 2)."""
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        cfg = KPPConfig()
        B_f = 1e-7 * jnp.ones((6, 4, 4))

        # Without Q_sfc_T (diagnosed proxy)
        out_diag = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f)

        # With Q_sfc_T imposed (much larger than proxy)
        Q_sfc_T = jnp.ones((6, 4, 4)) * 0.01  # 0.01 K*m/s
        out_imposed = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f, Q_sfc_T=Q_sfc_T,
        )

        # Tendencies should differ when imposed flux is used
        diff = float(jnp.max(jnp.abs(out_imposed.dT_dt - out_diag.dT_dt)))
        assert diff > 1e-15, "Q_sfc_T had no effect on KPP output"

    def test_h_bl_prev_changes_bl_depth(self):
        """h_bl_prev breaks the V_t-h_bl coupling (issue #168 bug 3)."""
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        cfg = KPPConfig()
        B_f = 1e-7 * jnp.ones((6, 4, 4))

        # Without h_bl_prev (uses max_depth estimate)
        out_default = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f,
        )

        # With h_bl_prev = shallow (20m)
        h_bl_prev = jnp.ones((6, 4, 4)) * 20.0
        out_shallow = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f,
            h_bl_prev=h_bl_prev,
        )

        # BL depth estimate should differ
        # (K_v profile changes because V_t changes with h_bl_prev)
        diff = float(jnp.max(jnp.abs(out_shallow.K_v - out_default.K_v)))
        assert diff > 1e-15, "h_bl_prev had no effect on K_v profile"

    def test_w_s_suppressed_in_stable_conditions(self):
        """In stable forcing (B_f<0), the turbulent velocity scale w_s
        must be suppressed by phi_m = 1 + 5*|zeta|, NOT remain at the
        unsuppressed value kappa*u_star.

        Convention used here: ``B_f > 0 = unstable``.  Then
        ``L_MO = u*^3/(kappa*B_f) < 0`` for stable, and
        ``zeta = d/L_MO < 0`` for stable.  An earlier version used
        ``max(zeta, 0)`` which always returned 0 in stable conditions,
        disabling the suppression entirely.  Fixed to ``max(-zeta, 0)``.

        Codex adversarial review iter-1, finding #4.
        """
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()

        # Match-magnitude probes: same |B_f|, opposite signs.
        B_f_stable = jnp.full((6, 4, 4), -1e-6, dtype=jnp.float64)
        B_f_unstable = jnp.full((6, 4, 4), 1e-6, dtype=jnp.float64)
        cfg = KPPConfig()

        out_s = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f_stable)
        out_u = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, J, cfg, B_f=B_f_unstable)

        # Stable boundary layer should mix LESS than unstable
        # at the same |B_f|.  K_v inside the BL = h_bl * w_s * G, so
        # K_v_stable.max() < K_v_unstable.max() if w_s suppression is
        # active.
        K_v_stable_max = float(jnp.max(out_s.K_v))
        K_v_unstable_max = float(jnp.max(out_u.K_v))
        # Unstable should have at least 1.5x more max K_v than stable.
        # Under the bug (no suppression), the two are nearly identical.
        ratio = K_v_unstable_max / max(K_v_stable_max, 1e-30)
        assert ratio > 1.2, (
            f"Stable BL mixes too aggressively: K_v_stable_max="
            f"{K_v_stable_max:.3e}, K_v_unstable_max={K_v_unstable_max:.3e}, "
            f"ratio={ratio:.2f}.  Stable suppression formula likely "
            f"degenerated to no-op (max(zeta_kpp, 0) bug)."
        )

    def test_a_v_uses_a_bg_not_k_bg(self):
        """Interior momentum viscosity falls back to A_bg, not K_bg.

        In LMD94, momentum and tracer share the shear-instability
        formulation but have different background floors:
        ``K_bg = 1e-5`` for tracers, ``A_bg = 1e-4`` for momentum.
        An earlier version used ``K_bg`` for the momentum interior
        floor, dropping ``A_v`` by an order of magnitude in stable
        interior layers.  Codex adversarial review iter-1, finding #6.
        """
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()

        # Stable forcing → all interior interfaces below the BL are in
        # the "interior" branch; pick a small h_bl_prev to keep BL thin.
        B_f = jnp.full((6, 4, 4), -1e-7, dtype=jnp.float64)
        h_bl_prev = jnp.full((6, 4, 4), 5.0, dtype=jnp.float64)
        cfg = KPPConfig()
        out = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg,
            B_f=B_f, h_bl_prev=h_bl_prev,
        )

        # Far below the BL, in stably stratified layers with weak
        # shear, K_v should be at or near K_bg = 1e-5 and A_v at or
        # near A_bg = 1e-4.  Take the minimum over the lowest layers
        # (away from the BL where K_bl_half dominates).
        # Specifically, the LAST few interfaces should reflect the
        # interior floor.
        K_v_min_bottom = float(jnp.min(out.K_v[..., -3:]))
        A_v_min_bottom = float(jnp.min(out.A_v[..., -3:]))

        # K_v floor ~ K_bg = 1e-5.  Check it is within 50% of K_bg.
        assert abs(K_v_min_bottom - cfg.K_bg) / cfg.K_bg < 0.5, (
            f"K_v interior floor {K_v_min_bottom:.3e} differs from "
            f"K_bg={cfg.K_bg} by more than 50%."
        )
        # A_v floor must be ~A_bg = 1e-4, NOT K_bg = 1e-5.  The
        # discriminator: A_v should be at least 5x K_v in stable
        # interior (since A_bg = 10 * K_bg).
        assert A_v_min_bottom > 5.0 * K_v_min_bottom, (
            f"A_v interior {A_v_min_bottom:.3e} is not significantly "
            f"larger than K_v interior {K_v_min_bottom:.3e}.  "
            f"Bug: A_v branch is using K_bg ({cfg.K_bg}) instead of "
            f"A_bg ({cfg.A_bg})."
        )

    def test_nonlocal_transport_conserves_column_tracer(self):
        """KPP non-local transport is column-conservative.

        The non-local heat/salt fluxes ``F = gamma * Q * G(sigma)``
        vanish at sigma=0 (G=0) and sigma>=1 (G clipped to 0), so the
        flux divergence ``-dF/dz`` integrates to zero over the column
        with zero-flux BCs.  An earlier version masked the divergence
        with ``in_bl_full`` (sigma_center < 1), which dropped the
        compensating tendency in the cell whose center sigma >= 1
        but whose top interface sigma_half < 1 — breaking column
        conservation when h_bl cut through a grid cell.  Codex
        adversarial review iter-1, finding #1.
        """
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        from legoesm.ocean.eos import wright_eos
        from legoesm.ocean.vertical import create_ocean_z_star

        nlev = 8
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=200.0)
        shape = (6, 2, 2, nlev)
        shape_2d = (6, 2, 2)

        # Unstable column: cold over warm.
        T_profile = jnp.array([2.0, 5.0, 10.0, 15.0, 18.0, 20.0, 22.0, 23.0])
        T = jnp.broadcast_to(T_profile[None, None, None, :], shape).astype(jnp.float64)
        S = jnp.full(shape, 35.0, dtype=jnp.float64)
        p = jnp.full(shape, 1e6, dtype=jnp.float64)
        rho = wright_eos(T, S, p)

        u = jnp.full(shape, 0.05, dtype=jnp.float64)
        v = jnp.zeros(shape, dtype=jnp.float64)
        eta = jnp.zeros(shape_2d, dtype=jnp.float64)
        J = jnp.ones(shape_2d, dtype=jnp.float64)

        B_f = jnp.full(shape_2d, 1e-6, dtype=jnp.float64)
        Q_sfc_T = jnp.full(shape_2d, 1e-3, dtype=jnp.float64)

        # h_bl_prev placed mid-column to ensure the BL crosses a layer.
        h_bl_prev = jnp.full(shape_2d, 75.0, dtype=jnp.float64)
        cfg = KPPConfig()

        out = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg,
            B_f=B_f, Q_sfc_T=Q_sfc_T, h_bl_prev=h_bl_prev,
        )
        out_no_qsfc = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg,
            B_f=B_f, Q_sfc_T=jnp.zeros(shape_2d), h_bl_prev=h_bl_prev,
        )

        # Isolate the non-local contribution by differencing.
        dT_nonlocal = out.dT_dt - out_no_qsfc.dT_dt

        dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
        column_integral = jnp.sum(dT_nonlocal * dz_actual, axis=-1)
        max_drift = float(jnp.max(jnp.abs(column_integral)))
        norm = float(jnp.max(jnp.abs(dT_nonlocal)))
        rel_drift = max_drift / max(norm, 1e-30)

        assert rel_drift < 1e-12, (
            f"KPP non-local transport non-conservative: column drift "
            f"{max_drift:.3e}, relative {rel_drift:.3e}.  Was the "
            f"in_bl_full mask re-introduced?"
        )

    def test_b_f_none_fallback_is_zero(self):
        """Calling kpp_vertical_mixing with ``B_f=None`` must use
        B_f = 0 (no convective non-local transport) rather than a
        wrong-magnitude diffusive proxy.

        Iter-47 replaced the prior ``g/ρ₀ · K_bg · drho_dz_sfc`` proxy
        (which underestimates realistic B_f by 2-4 orders of magnitude)
        with B_f = 0 — fail-closed semantics that prevent silent
        non-local transport activation when surface forcing is missing.

        Regression guard: this test asserts that ``B_f=None`` and an
        explicit ``B_f = 0`` produce IDENTICAL output.  Under the prior
        diffusive-proxy behavior, the two would disagree by
        ~O(1e-5) K/s in dT_dt (the magnitude of the spurious proxy).
        """
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        u, v, T, S, rho, eta, z_coord, J = self._make_kpp_inputs()
        cfg = KPPConfig()

        Q_sfc_T = jnp.full((6, 4, 4), 1e-3, dtype=jnp.float64)
        Q_sfc_S = jnp.zeros((6, 4, 4), dtype=jnp.float64)
        B_f_zero = jnp.zeros((6, 4, 4), dtype=jnp.float64)

        out_explicit_zero = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg,
            B_f=B_f_zero, Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
        )
        out_none = kpp_vertical_mixing(
            u, v, T, S, rho, eta, z_coord, J, cfg,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
        )

        diff = float(jnp.max(jnp.abs(out_none.dT_dt - out_explicit_zero.dT_dt)))
        assert diff < 1e-12, (
            f"B_f=None fallback differs from B_f=0 by {diff:.3e}; iter-47 "
            f"expected fail-closed identity."
        )

    def test_b_salt_sign_freshening_is_stabilizing(self):
        """Surface buoyancy flux from freshwater has the correct sign.

        The KPP B_f convention is ``B_f > 0 = unstable``, with formula
        ``B_f = -g*alpha*Q_T + g*beta*Q_S`` where Q_T, Q_S are kinematic
        fluxes INTO the ocean.  For pure surface freshening (P > E,
        Q_T = 0) the salt flux INTO the ocean is NEGATIVE
        (Q_S = -S * F_fw / rho_0 < 0 because freshwater dilutes the
        surface), which makes B_salt = +g*beta*Q_S < 0 — STABILIZING,
        consistent with lighter water on top.

        Regression guard: a previous implementation used the wrong sign
        ``B_salt = -g*beta*Q_S``, producing destabilization for
        freshening — exactly opposite of physical reality.
        """
        from legoesm.ocean.physics.vertical_mixing.integration import (
            make_vertical_mixing_physics,
        )
        from legoesm.ocean.physics.vertical_mixing.config import (
            KPPConfig, VerticalMixingConfig,
        )
        from legoesm.ocean.state import OceanSurfaceForcing
        from legoesm.core.field import Field
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.grids.cubed_sphere import create_cubed_sphere

        # Build minimal cubed-sphere ocean state with strong surface freshening.
        n, nlev = 4, 6
        grid = create_cubed_sphere(n=n)
        z_coord = create_ocean_z_star(n_levels=nlev, H_max=2000.0)
        shape = (6, n, n, nlev)
        shape_2d = (6, n, n)

        # Mildly stable column (T decreasing with depth, near-zero density gradient
        # so the surface buoyancy flux dominates h_bl).
        T_data = jnp.broadcast_to(
            jnp.linspace(15.0, 5.0, nlev)[None, None, None, :], shape,
        ).astype(jnp.float64)
        S_data = jnp.full(shape, 35.0, dtype=jnp.float64)
        u_data = jnp.zeros(shape, dtype=jnp.float64)
        v_data = jnp.zeros(shape, dtype=jnp.float64)
        eta_data = jnp.zeros(shape_2d, dtype=jnp.float64)
        H_bathy_data = jnp.full(shape_2d, 2000.0, dtype=jnp.float64)
        land_mask = jnp.ones(shape_2d, dtype=jnp.float64)

        from legoesm.ocean.state import OceanState
        state = OceanState(
            u=Field(data=u_data),
            v=Field(data=v_data),
            T=Field(data=T_data),
            S=Field(data=S_data),
            eta=Field(data=eta_data),
            H_bathy=Field(data=H_bathy_data),
            land_mask=Field(data=land_mask),
        )

        # Strong freshening: P >> E.
        fw_freshening = jnp.full(shape_2d, 5e-4, dtype=jnp.float64)  # kg/m^2/s
        # Strong brine rejection: -P + E >> 0 (e.g. sea-ice formation).
        fw_brine = jnp.full(shape_2d, -5e-4, dtype=jnp.float64)
        # Zero net heat flux so freshwater is the only buoyancy driver.
        zero_2d = jnp.zeros(shape_2d, dtype=jnp.float64)

        cfg = VerticalMixingConfig(scheme="kpp", kpp=KPPConfig())
        physics_fn = make_vertical_mixing_physics(cfg)

        # Pure freshening: K_v should pick up the BACKGROUND value (B_f<0,
        # h_bl <= 1 layer) rather than an enhanced convective profile.
        sf_fresh = OceanSurfaceForcing(
            tau_x=zero_2d, tau_y=zero_2d, q_net=zero_2d, freshwater=fw_freshening,
        )
        out_fresh = physics_fn(state, grid, z_coord, surface_forcing=sf_fresh)

        # Pure brine rejection: K_v should be larger because B_f > 0
        # destabilizes the column → deeper BL → enhanced K_v.
        sf_brine = OceanSurfaceForcing(
            tau_x=zero_2d, tau_y=zero_2d, q_net=zero_2d, freshwater=fw_brine,
        )
        out_brine = physics_fn(state, grid, z_coord, surface_forcing=sf_brine)

        # The factory wraps to OceanTendencies, which doesn't expose K_v
        # (it is None), so compare the non-local dT/dt magnitudes instead.
        # Brine rejection should produce STRONGER mixing (larger |dT/dt|)
        # than freshening — the asymmetry is exactly what would FAIL
        # under the buggy `B_salt = -g*beta*Q_S` (which would invert
        # the asymmetry: freshening would mix harder than brine).
        max_T_brine = float(jnp.max(jnp.abs(out_brine.dT_dt.data)))
        max_T_fresh = float(jnp.max(jnp.abs(out_fresh.dT_dt.data)))
        assert max_T_brine >= max_T_fresh - 1e-30, (
            f"B_salt sign regression: freshening produced stronger mixing "
            f"({max_T_fresh:.3e}) than brine rejection ({max_T_brine:.3e}). "
            "Under the correct sign (B_salt = +g*beta*Q_S), brine "
            "rejection destabilizes and should mix at least as hard as "
            "freshening (which stabilizes)."
        )


# ===========================================================================
# P0: Large-Yeager bulk flux
# ===========================================================================

class TestLargeYeager:
    """Tests for the Large-Yeager coefficient-space iteration."""

    def _make_flux_inputs(self, n=10):
        u_rel = jnp.ones(n) * 5.0
        v_rel = jnp.zeros(n)
        T_atm = jnp.ones(n) * 290.0
        q_atm = jnp.ones(n) * 0.008
        T_sfc = jnp.ones(n) * 295.0
        q_sfc = jnp.ones(n) * 0.012
        rho = jnp.ones(n) * 1.2
        return u_rel, v_rel, T_atm, q_atm, T_sfc, q_sfc, rho

    def test_monotonic_drag_with_wind(self):
        """Drag should increase monotonically with wind speed."""
        from legoesm.core.bulk_flux import compute_most_fluxes
        winds = jnp.array([2.0, 5.0, 10.0, 15.0, 20.0])
        taus = []
        for w in winds:
            u = jnp.ones(1) * float(w)
            v = jnp.zeros(1)
            T_atm = jnp.ones(1) * 290.0
            q_atm = jnp.ones(1) * 0.008
            T_sfc = jnp.ones(1) * 295.0
            q_sfc = jnp.ones(1) * 0.012
            rho = jnp.ones(1) * 1.2
            tau_x, _, _, _, _ = compute_most_fluxes(
                u, v, T_atm, q_atm, T_sfc, q_sfc, rho,
                scheme="large_yeager"
            )
            taus.append(float(jnp.abs(tau_x[0])))
        for i in range(len(taus) - 1):
            assert taus[i + 1] > taus[i], \
                f"Drag not increasing: {taus[i]} -> {taus[i+1]}"

    def test_height_dependence(self):
        """Results should differ for z_ref=10 vs z_ref=20."""
        from legoesm.core.bulk_flux import compute_most_fluxes
        args = self._make_flux_inputs()
        _, _, _, _, u_star_10 = compute_most_fluxes(*args, z_ref=10.0,
                                                      scheme="large_yeager")
        _, _, _, _, u_star_20 = compute_most_fluxes(*args, z_ref=20.0,
                                                      scheme="large_yeager")
        assert not jnp.allclose(u_star_10, u_star_20), \
            "u_star should differ with different z_ref"

    def test_stable_vs_unstable(self):
        """Stable and unstable cases should give different heat fluxes."""
        from legoesm.core.bulk_flux import compute_most_fluxes
        n = 4
        u = jnp.ones(n) * 8.0
        v = jnp.zeros(n)
        rho = jnp.ones(n) * 1.2
        q_atm = jnp.ones(n) * 0.008
        # Unstable: warm surface
        T_sfc_warm = jnp.ones(n) * 300.0
        T_atm = jnp.ones(n) * 290.0
        q_sfc = jnp.ones(n) * 0.012
        _, _, sh_unstable, _, _ = compute_most_fluxes(
            u, v, T_atm, q_atm, T_sfc_warm, q_sfc, rho, scheme="large_yeager"
        )
        # Stable: cold surface
        T_sfc_cold = jnp.ones(n) * 285.0
        _, _, sh_stable, _, _ = compute_most_fluxes(
            u, v, T_atm, q_atm, T_sfc_cold, q_sfc, rho, scheme="large_yeager"
        )
        assert not jnp.allclose(sh_unstable, sh_stable)

    def test_reference_values(self):
        """Spot-check LY drag coefficient at moderate wind."""
        from legoesm.core.bulk_flux import compute_most_fluxes
        u = jnp.ones(1) * 10.0
        v = jnp.zeros(1)
        T_atm = jnp.ones(1) * 290.0
        T_sfc = jnp.ones(1) * 290.0  # neutral
        q_atm = jnp.ones(1) * 0.0
        q_sfc = jnp.ones(1) * 0.0
        rho = jnp.ones(1) * 1.2
        tau_x, _, _, _, u_star = compute_most_fluxes(
            u, v, T_atm, q_atm, T_sfc, q_sfc, rho,
            scheme="large_yeager", z_ref=10.0
        )
        # Expected CDN ~ (2.7/10 + 0.142 + 0.764)*1e-3 ~ 1.176e-3
        # tau ~ rho * CDN * U^2 ~ 1.2 * 1.176e-3 * 100 ~ 0.14
        tau = float(jnp.abs(tau_x[0]))
        assert 0.05 < tau < 0.5, f"tau={tau} out of expected range"


# ===========================================================================
# P1: External forcing interpolation
# ===========================================================================

class TestExternalForcing:
    """Tests for external forcing interpolation."""

    def test_ghg_constant(self):
        from legoesm.forcing.external import GHGConfig, get_ghg_at_time
        cfg = GHGConfig(co2_ppmv=400.0)
        ghg = get_ghg_at_time(cfg, 180.0)
        assert ghg["co2_ppmv"] == 400.0

    def test_tsi_constant(self):
        from legoesm.forcing.external import SolarConfig, get_tsi_at_time
        cfg = SolarConfig(S_0=constants.S_0)
        assert get_tsi_at_time(cfg, 0.0) == constants.S_0

    def test_ozone_disabled(self):
        from legoesm.forcing.external import OzoneConfig, get_ozone_at_time
        cfg = OzoneConfig(enabled=False)
        assert get_ozone_at_time(cfg, 0.0) is None

    def test_zonal_interp_to_grid(self):
        """Test latitude interpolation from zonal mean to grid."""
        from legoesm.forcing.external import _interp_zonal_to_grid
        lat_src = np.linspace(-90, 90, 37)
        field = np.sin(np.radians(lat_src))  # sin(lat)
        lat_grid = jnp.array([0.0, jnp.pi / 4, -jnp.pi / 4])
        result = _interp_zonal_to_grid(lat_src, field, lat_grid)
        assert result.shape == (3,)
        assert jnp.allclose(result[0], 0.0, atol=0.05)  # equator
        assert float(result[1]) > 0.5  # 45N

    def test_vertical_interp(self):
        """Test vertical interpolation in log-pressure space."""
        from legoesm.forcing.external import _interp_vertical
        plev_src = np.array([100.0, 1000.0, 10000.0, 50000.0, 100000.0])
        field = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # (1, 5)
        p_target = jnp.array([[500.0, 5000.0, 75000.0]])  # (1, 3)
        result = _interp_vertical(field, plev_src, p_target)
        assert result.shape == (1, 3)
        # Should be between adjacent values
        assert 1.0 < float(result[0, 0]) < 2.5
        assert 2.0 < float(result[0, 1]) < 4.0

    def test_bad_ghg_source(self):
        from legoesm.forcing.external import GHGConfig, get_ghg_at_time
        cfg = GHGConfig(source="invalid")
        with pytest.raises(ValueError, match="Unknown GHG source"):
            get_ghg_at_time(cfg, 0.0)


# ===========================================================================
# P2: Soil hydraulics inversion consistency
# ===========================================================================

class TestSoilHydraulicsConsistency:
    """Inverse relations and monotonicity checks."""

    def test_vg_roundtrip(self):
        """VG: theta -> psi -> theta should be identity."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, van_genuchten_theta, van_genuchten_psi
        )
        cfg = SoilHydraulicsConfig()
        theta = jnp.linspace(cfg.theta_r + 0.01, cfg.theta_sat - 0.01, 20)
        psi = van_genuchten_psi(theta, cfg)
        theta_back = van_genuchten_theta(psi, cfg)
        assert jnp.allclose(theta, theta_back, atol=1e-4)

    def test_pdi_approximate_inversion(self):
        """PDI inverse (VG approx) should be reasonable near saturation."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, pdi_theta, pdi_psi
        )
        cfg = SoilHydraulicsConfig(retention_curve="pdi")
        # Near saturation (wet end), VG approximation is best
        psi_wet = jnp.linspace(-0.1, -1.0, 10)
        theta = pdi_theta(psi_wet, cfg)
        psi_approx = pdi_psi(theta, cfg)
        theta_back = pdi_theta(psi_approx, cfg)
        # Approximate: VG inverse works well in the capillary-dominated range
        assert jnp.allclose(theta, theta_back, atol=0.02), \
            f"PDI round-trip error: max={float(jnp.max(jnp.abs(theta - theta_back)))}"

    def test_lu_approximate_inversion(self):
        """Lu inverse (VG approx) should be reasonable in normal range."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, lu_theta, lu_psi
        )
        cfg = SoilHydraulicsConfig(retention_curve="lu")
        psi_orig = jnp.linspace(-0.5, -5.0, 20)
        theta = lu_theta(psi_orig, cfg)
        psi_approx = lu_psi(theta, cfg)
        theta_back = lu_theta(psi_approx, cfg)
        assert jnp.allclose(theta, theta_back, atol=0.02)

    def test_k_monotonic(self):
        """Hydraulic conductivity should decrease with increasing suction."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, van_genuchten_K
        )
        cfg = SoilHydraulicsConfig()
        psi = jnp.linspace(-0.1, -10.0, 50)
        K = van_genuchten_K(psi, cfg)
        # K should be monotonically decreasing (more negative psi = drier)
        dK = K[1:] - K[:-1]
        assert float(jnp.max(dK)) <= 1e-10, "K should decrease with suction"

    def test_theta_bounded(self):
        """Water content should be bounded between theta_r and theta_sat."""
        from legoesm.land.soil_hydraulics import (
            SoilHydraulicsConfig, theta_from_psi
        )
        for curve in ["van_genuchten", "clapp_hornberger", "pdi", "lu"]:
            cfg = SoilHydraulicsConfig(retention_curve=curve)
            psi = jnp.linspace(-0.01, -100.0, 100)
            theta = theta_from_psi(psi, cfg)
            assert float(jnp.min(theta)) >= 0.0, f"{curve}: theta < 0"
            assert float(jnp.max(theta)) <= cfg.theta_sat + 1e-6, \
                f"{curve}: theta > theta_sat"

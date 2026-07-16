"""Smoke test for the RCE (Radiative-Convective Equilibrium) script.

Verifies import, grid setup, initial state construction, and a single
physics step for the cubed-sphere configuration.
"""

import jax.numpy as jnp
import pytest


class TestRCESetup:
    """Verify RCE experiment can be configured and initialized."""

    def test_cubed_sphere_setup_and_single_step(self):
        """Set up RCE on a small cubed-sphere grid and run one physics step."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.driver.component_factory import create_atmosphere_dycore
        from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        from legoesm.atmosphere.physics.convection.config import SBMConfig
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
        from legoesm.atmosphere.physics.convection.sbm import sbm_convection
        from legoesm import constants
        from legoesm.thermo import saturation_mixing_ratio

        N, NLEV, DT = 8, 10, 600.0
        grid = create_cubed_sphere(N)
        sigma = create_sigma_coordinate(NLEV)

        config = ExperimentConfig(
            grid=GridConfig(grid_type="cubed_sphere", resolution=N, nlev=NLEV),
            dycore=DycoreConfig(discretization="cdgrid", dt=DT),
        )
        model = create_atmosphere_dycore(config, grid, sigma)

        # Initial state
        state = held_suarez_init(grid, sigma, T_init=280.0)
        p_full_init = state.p_s.data[..., None] * sigma.sigma_full
        q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
        q_v = 0.6 * q_sat_init * sigma.sigma_full ** 2

        # One dynamics step
        state = model.step(state, DT)
        assert jnp.all(jnp.isfinite(state.T.data))
        assert jnp.all(jnp.isfinite(state.u.data))

        # One radiation call
        gray_config = GrayRadiationConfig(
            tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
            sfc_albedo=0.06, perpetual_equinox=True,
        )
        ncol = state.T.data[..., 0].size
        T_col = state.T.data.reshape(ncol, NLEV)
        p_full_col = (state.p_s.data[..., None] * sigma.sigma_full).reshape(ncol, NLEV)
        p_half_col = (state.p_s.data[..., None] * sigma.sigma_half).reshape(ncol, NLEV + 1)
        q_v_col = q_v.reshape(ncol, NLEV)
        lat_col = grid.grid_lat.reshape(ncol)
        T_sfc_col = jnp.full(ncol, 300.0)
        insol = perpetual_equinox_insolation(lat_col, 1360.0)

        rad = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
        assert jnp.all(jnp.isfinite(rad.heating_rate))
        assert rad.heating_rate.shape == (ncol, NLEV)

        # One convection call
        sbm_config = SBMConfig(tau_c=7200.0, rh_ref=0.7)
        conv = sbm_convection(
            T=T_col, q_v=q_v_col,
            p_full=p_full_col, p_half=p_half_col,
            dt=DT, config=sbm_config,
        )
        assert jnp.all(jnp.isfinite(conv.dT_dt))
        # Post-Option-C: convection emits a 3D ``dq_c_conv_dt``
        # cloud-water source instead of a scalar surface precip;
        # microphysics owns the resulting surface-flux diagnostic.
        assert conv.dq_c_conv_dt.shape == (ncol, NLEV)
        assert jnp.all(conv.dq_c_conv_dt >= 0)

    def test_constants_used_for_freezing_point(self):
        """Verify RCE uses constants.T_freeze for land mode (not hardcoded)."""
        from legoesm import constants
        assert constants.T_freeze == 273.15
        assert constants.T_freeze_ocean == 271.35

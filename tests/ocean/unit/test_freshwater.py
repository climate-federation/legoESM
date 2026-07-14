"""Tests for freshwater closure in the MPAS ocean model.

Tests FreshwaterForcing construction, virtual salt flux, free-surface
mass terms, coupler integration, and conservation with freshwater.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    zero_freshwater,
    net_freshwater_flux,
    physical_net_freshwater_flux,
    freshwater_eta_tendency,
    virtual_salt_flux,
    virtual_salt_flux_from_net,
    normalize_freshwater_net,
    normalized_virtual_salt_flux,
    freshwater_from_coupler,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0)


@pytest.fixture
def state0(mesh, z_coord):
    return rest_state_mpas_ocean(
        mesh, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=500.0, land_lat_threshold=85.0,
    )


# ============================================================================
# FreshwaterForcing construction
# ============================================================================

class TestFreshwaterForcing:

    def test_zero_freshwater_shape(self):
        fw = zero_freshwater(100)
        assert fw.precip.shape == (100,)
        assert fw.evap.shape == (100,)
        assert fw.runoff.shape == (100,)
        assert fw.ice_fw.shape == (100,)

    def test_zero_freshwater_values(self):
        fw = zero_freshwater(10)
        assert jnp.all(fw.precip == 0.0)
        assert jnp.all(fw.evap == 0.0)
        assert jnp.all(fw.runoff == 0.0)
        assert jnp.all(fw.ice_fw == 0.0)

    def test_net_flux_zero(self):
        fw = zero_freshwater(10)
        F = net_freshwater_flux(fw)
        assert jnp.allclose(F, 0.0)

    def test_net_flux_precip_only(self):
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.ones(n) * 1e-5,
            evap=jnp.zeros(n),
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        F = net_freshwater_flux(fw)
        assert jnp.allclose(F, 1e-5)

    def test_net_flux_all_terms(self):
        n = 5
        P = jnp.ones(n) * 3e-5   # precip in
        E = jnp.ones(n) * 1e-5   # evap out (positive up)
        R = jnp.ones(n) * 0.5e-5  # runoff in
        M = jnp.ones(n) * 0.5e-5  # ice melt in
        fw = FreshwaterForcing(precip=P, evap=E, runoff=R, ice_fw=M)
        F = net_freshwater_flux(fw)
        # F = P - E + R + M = 3e-5 - 1e-5 + 0.5e-5 + 0.5e-5 = 3e-5
        assert jnp.allclose(F, 3e-5)

    def test_net_flux_evap_dominant(self):
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.zeros(n),
            evap=jnp.ones(n) * 2e-5,
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        F = net_freshwater_flux(fw)
        assert jnp.all(F < 0)  # net loss of freshwater

    def test_physical_net_excludes_restoring(self):
        """physical_net_freshwater_flux (the KPP/vmix buoyancy contract):
        P - E + R + M, EXCLUDING the numerical SSS-restoring channel —
        net_freshwater_flux minus exactly the restoring term."""
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.full(n, 3e-5), evap=jnp.full(n, 1e-5),
            runoff=jnp.full(n, 0.5e-5), ice_fw=jnp.full(n, 0.5e-5),
            restoring=jnp.full(n, 2e-5),
        )
        F_phys = physical_net_freshwater_flux(fw)
        assert jnp.allclose(F_phys, 3e-5)                    # P - E + R + M
        assert jnp.allclose(net_freshwater_flux(fw) - fw.restoring, F_phys)
        # Without restoring the two nets agree.
        fw0 = fw._replace(restoring=jnp.zeros(n))
        assert jnp.allclose(physical_net_freshwater_flux(fw0),
                            net_freshwater_flux(fw0))


# ============================================================================
# Eta tendency from freshwater
# ============================================================================

class TestEtaTendency:

    def test_eta_tendency_shape(self):
        fw = zero_freshwater(10)
        deta = freshwater_eta_tendency(fw, rho_0=constants.rho_ocean)
        assert deta.shape == (10,)

    def test_eta_tendency_zero(self):
        fw = zero_freshwater(10)
        deta = freshwater_eta_tendency(fw, rho_0=constants.rho_ocean)
        assert jnp.allclose(deta, 0.0)

    def test_eta_tendency_precip_positive(self):
        """Precipitation should raise the free surface."""
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.ones(n) * 1e-3,  # 1 mm/s
            evap=jnp.zeros(n),
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        deta = freshwater_eta_tendency(fw, rho_0=constants.rho_ocean)
        assert jnp.all(deta > 0)
        # deta/dt = P / rho_0 = 1e-3 / 1025 ≈ 9.76e-7 m/s
        expected = 1e-3 / constants.rho_ocean
        assert jnp.allclose(deta, expected, rtol=1e-10)

    def test_eta_tendency_evap_negative(self):
        """Evaporation should lower the free surface."""
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.zeros(n),
            evap=jnp.ones(n) * 2e-3,
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        deta = freshwater_eta_tendency(fw, rho_0=constants.rho_ocean)
        assert jnp.all(deta < 0)


# ============================================================================
# Virtual salt flux
# ============================================================================

class TestVirtualSaltFlux:

    def test_vsf_shape(self):
        fw = zero_freshwater(10)
        dz = jnp.ones(10) * 20.0
        dS = virtual_salt_flux(fw, S_ref=35.0, dz_0=dz, rho_0=constants.rho_ocean)
        assert dS.shape == (10,)

    def test_vsf_zero_freshwater(self):
        fw = zero_freshwater(10)
        dz = jnp.ones(10) * 20.0
        dS = virtual_salt_flux(fw, S_ref=35.0, dz_0=dz, rho_0=constants.rho_ocean)
        assert jnp.allclose(dS, 0.0)

    def test_vsf_precip_dilutes(self):
        """Precipitation (freshwater in) should decrease salinity."""
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.ones(n) * 1e-4,
            evap=jnp.zeros(n),
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        dz = jnp.ones(n) * 20.0
        dS = virtual_salt_flux(fw, S_ref=35.0, dz_0=dz, rho_0=constants.rho_ocean)
        assert jnp.all(dS < 0)  # salinity decreases

    def test_vsf_evap_concentrates(self):
        """Evaporation (freshwater out) should increase salinity."""
        n = 5
        fw = FreshwaterForcing(
            precip=jnp.zeros(n),
            evap=jnp.ones(n) * 1e-4,
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        dz = jnp.ones(n) * 20.0
        dS = virtual_salt_flux(fw, S_ref=35.0, dz_0=dz, rho_0=constants.rho_ocean)
        assert jnp.all(dS > 0)  # salinity increases

    def test_vsf_formula(self):
        """Check the virtual salt flux formula: dS/dt = -S_ref * F_fw / (rho_0 * dz_0)."""
        n = 3
        P = 2e-4
        S_ref = 35.0
        rho_0 = constants.rho_ocean
        dz_0 = 20.0
        fw = FreshwaterForcing(
            precip=jnp.ones(n) * P,
            evap=jnp.zeros(n),
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        dS = virtual_salt_flux(fw, S_ref=S_ref, dz_0=jnp.ones(n) * dz_0, rho_0=rho_0)
        expected = -S_ref * P / (rho_0 * dz_0)
        assert jnp.allclose(dS, expected, rtol=1e-10)

    def test_vsf_thin_layer_amplifies(self):
        """Thinner surface layer => stronger salinity response."""
        n = 3
        fw = FreshwaterForcing(
            precip=jnp.ones(n) * 1e-4,
            evap=jnp.zeros(n),
            runoff=jnp.zeros(n),
            ice_fw=jnp.zeros(n),
        )
        dS_thick = virtual_salt_flux(fw, S_ref=35.0, dz_0=jnp.ones(n) * 50.0, rho_0=constants.rho_ocean)
        dS_thin = virtual_salt_flux(fw, S_ref=35.0, dz_0=jnp.ones(n) * 10.0, rho_0=constants.rho_ocean)
        assert jnp.all(jnp.abs(dS_thin) > jnp.abs(dS_thick))


# ============================================================================
# freshwater_from_coupler
# ============================================================================

class TestFreshwaterFromCoupler:

    def test_basic_from_coupler(self):
        n = 10
        precip = jnp.ones(n) * 1e-4
        lhflx = jnp.ones(n) * 50.0  # W/m2
        L_v = constants.L_v
        mask = jnp.ones(n)
        fw = freshwater_from_coupler(precip, lhflx, L_v, ocean_mask=mask)
        assert fw.precip.shape == (n,)
        assert jnp.allclose(fw.precip, 1e-4)
        assert jnp.allclose(fw.evap, 50.0 / L_v)
        assert jnp.allclose(fw.runoff, 0.0)
        assert jnp.allclose(fw.ice_fw, 0.0)

    def test_ocean_mask(self):
        n = 10
        precip = jnp.ones(n) * 1e-4
        lhflx = jnp.ones(n) * 50.0
        mask = jnp.zeros(n).at[:5].set(1.0)  # only first 5 are ocean
        fw = freshwater_from_coupler(precip, lhflx, constants.L_v, ocean_mask=mask)
        assert jnp.all(fw.precip[5:] == 0.0)
        assert jnp.all(fw.precip[:5] > 0.0)

    def test_with_runoff(self):
        n = 10
        precip = jnp.zeros(n)
        lhflx = jnp.zeros(n)
        runoff_sfc = jnp.ones(n) * 1e-5
        runoff_sub = jnp.ones(n) * 0.5e-5
        fw = freshwater_from_coupler(
            precip, lhflx, constants.L_v,
            runoff_surface=runoff_sfc,
            runoff_subsurface=runoff_sub,
        )
        assert jnp.allclose(fw.runoff, 1.5e-5)

    def test_with_ice(self):
        """Ice melting should produce positive freshwater."""
        from legoesm.core.field import Field
        from legoesm.ice.state import SeaIceState

        n = 10
        shape = (n,)
        h_old = jnp.ones(shape) * 1.0  # 1m ice
        h_new = jnp.ones(shape) * 0.8  # melted to 0.8m
        conc = jnp.ones(shape) * 0.5

        ice_old = SeaIceState(
            h_ice=Field(data=h_old, name="h_ice", dims=("nCells",), units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=("nCells",), units="K"),
            concentration=Field(data=conc, name="conc", dims=("nCells",), units="1"),
        )
        ice_new = SeaIceState(
            h_ice=Field(data=h_new, name="h_ice", dims=("nCells",), units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T_ice", dims=("nCells",), units="K"),
            concentration=Field(data=conc, name="conc", dims=("nCells",), units="1"),
        )

        from legoesm.ice.config import SeaIceConfig
        ice_cfg = SeaIceConfig()

        fw = freshwater_from_coupler(
            jnp.zeros(n), jnp.zeros(n), constants.L_v,
            ice_state_old=ice_old, ice_state_new=ice_new,
            ice_config=ice_cfg, dt=3600.0,
        )
        # Melting (h decreases) should give positive ice_fw
        assert jnp.all(fw.ice_fw > 0)

    def test_ice_fw_thickness_only_melt(self):
        """Thickness-only melt (fixed concentration) → correct mass change."""
        from legoesm.core.field import Field
        from legoesm.ice.state import SeaIceState
        from legoesm.ice.config import SeaIceConfig

        n, dt = 5, 3600.0
        shape = (n,)
        rho_ice = SeaIceConfig().rho_ice
        h_old, h_new, conc = 2.0, 1.5, 0.8

        ice_old = SeaIceState(
            h_ice=Field(data=jnp.full(shape, h_old), name="h", dims=("c",), units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T", dims=("c",), units="K"),
            concentration=Field(data=jnp.full(shape, conc), name="A", dims=("c",), units="1"),
        )
        ice_new = SeaIceState(
            h_ice=Field(data=jnp.full(shape, h_new), name="h", dims=("c",), units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T", dims=("c",), units="K"),
            concentration=Field(data=jnp.full(shape, conc), name="A", dims=("c",), units="1"),
        )

        fw = freshwater_from_coupler(
            jnp.zeros(n), jnp.zeros(n), constants.L_v,
            ice_state_old=ice_old, ice_state_new=ice_new,
            ice_config=SeaIceConfig(), dt=dt,
        )
        expected = rho_ice * (h_old - h_new) * conc / dt
        assert jnp.allclose(fw.ice_fw, expected, rtol=1e-10)

    def test_ice_fw_concentration_only(self):
        """Concentration change at fixed thickness → correct mass change."""
        from legoesm.core.field import Field
        from legoesm.ice.state import SeaIceState
        from legoesm.ice.config import SeaIceConfig

        n, dt = 5, 3600.0
        shape = (n,)
        rho_ice = SeaIceConfig().rho_ice
        h, A_old, A_new = 1.0, 0.8, 0.6

        ice_old = SeaIceState(
            h_ice=Field(data=jnp.full(shape, h), name="h", dims=("c",), units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T", dims=("c",), units="K"),
            concentration=Field(data=jnp.full(shape, A_old), name="A", dims=("c",), units="1"),
        )
        ice_new = SeaIceState(
            h_ice=Field(data=jnp.full(shape, h), name="h", dims=("c",), units="m"),
            T_ice=Field(data=jnp.full(shape, 260.0), name="T", dims=("c",), units="K"),
            concentration=Field(data=jnp.full(shape, A_new), name="A", dims=("c",), units="1"),
        )

        fw = freshwater_from_coupler(
            jnp.zeros(n), jnp.zeros(n), constants.L_v,
            ice_state_old=ice_old, ice_state_new=ice_new,
            ice_config=SeaIceConfig(), dt=dt,
        )
        expected = rho_ice * h * (A_old - A_new) / dt
        assert jnp.allclose(fw.ice_fw, expected, rtol=1e-10)

    def test_ice_fw_multi_category(self):
        """Multi-category ice: total areal mass change is summed."""
        from legoesm.core.field import Field
        from legoesm.ice.state import SeaIceState
        from legoesm.ice.config import SeaIceConfig

        n, n_cat, dt = 5, 3, 3600.0
        shape = (n,)
        rho_ice = SeaIceConfig().rho_ice

        # Multi-cat: shape (n, n_cat)
        h_old = jnp.ones((n, n_cat)) * jnp.array([0.5, 1.0, 2.0])
        h_new = jnp.ones((n, n_cat)) * jnp.array([0.4, 0.9, 1.8])
        A_old = jnp.ones((n, n_cat)) * 0.3
        A_new = jnp.ones((n, n_cat)) * 0.3

        ice_old = SeaIceState(
            h_ice=Field(data=h_old, name="h", dims=("c", "cat"), units="m"),
            T_ice=Field(data=jnp.full((n, n_cat), 260.0), name="T", dims=("c", "cat"), units="K"),
            concentration=Field(data=A_old, name="A", dims=("c", "cat"), units="1"),
        )
        ice_new = SeaIceState(
            h_ice=Field(data=h_new, name="h", dims=("c", "cat"), units="m"),
            T_ice=Field(data=jnp.full((n, n_cat), 260.0), name="T", dims=("c", "cat"), units="K"),
            concentration=Field(data=A_new, name="A", dims=("c", "cat"), units="1"),
        )

        fw = freshwater_from_coupler(
            jnp.zeros(n), jnp.zeros(n), constants.L_v,
            ice_state_old=ice_old, ice_state_new=ice_new,
            ice_config=SeaIceConfig(), dt=dt,
        )
        # Should be shape (n,) — summed over categories
        assert fw.ice_fw.shape == (n,)
        mass_old = rho_ice * jnp.sum(h_old * A_old, axis=-1)
        mass_new = rho_ice * jnp.sum(h_new * A_new, axis=-1)
        expected = -(mass_new - mass_old) / dt
        assert jnp.allclose(fw.ice_fw, expected, rtol=1e-10)


# ============================================================================
# Config fields
# ============================================================================

class TestConfig:

    def test_default_freshwater_closure(self):
        config = MPASOceanConfig()
        assert config.freshwater_closure == "virtual_salt_flux"
        assert config.S_ref == 35.0

    def test_none_closure(self):
        config = MPASOceanConfig(freshwater_closure="none")
        assert config.freshwater_closure == "none"

    def test_real_freshwater_closure(self):
        config = MPASOceanConfig(freshwater_closure="real_freshwater")
        assert config.freshwater_closure == "real_freshwater"


# ============================================================================
# Integration with ocean model
# ============================================================================

class TestOceanModelWithFreshwater:

    def test_step_no_freshwater(self, mesh, z_coord, state0):
        """Model step without freshwater should work as before."""
        config = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
            freshwater_closure="none",
        )
        model = MPASOceanModel(mesh, z_coord, config)
        state1 = model.step(state0, 60.0)
        assert jnp.all(jnp.isfinite(state1.eta.data))
        assert jnp.all(jnp.isfinite(state1.S.data))

    def test_step_with_zero_freshwater(self, mesh, z_coord, state0):
        """Step with zero freshwater = step without freshwater."""
        config = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        fw = zero_freshwater(mesh.nCells)
        state_fw = model.step(state0, 60.0, freshwater=fw)
        state_no = model.step(state0, 60.0)
        # Should be identical
        assert jnp.allclose(state_fw.eta.data, state_no.eta.data, atol=1e-15)
        assert jnp.allclose(state_fw.S.data, state_no.S.data, atol=1e-15)

    def test_step_with_precip_raises_eta(self, mesh, z_coord, state0):
        """Precipitation should raise the free surface."""
        config = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        mask = state0.land_mask.data

        # Uniform 1 mm/s precip over ocean
        fw = FreshwaterForcing(
            precip=jnp.ones(mesh.nCells) * 1e-3 * mask,
            evap=jnp.zeros(mesh.nCells),
            runoff=jnp.zeros(mesh.nCells),
            ice_fw=jnp.zeros(mesh.nCells),
        )

        state_no = model.step(state0, 60.0)
        state_fw = model.step(state0, 60.0, freshwater=fw)

        # Mean ocean eta should be higher with precip
        ocean_cells = mask > 0.5
        eta_mean_no = jnp.mean(state_no.eta.data[ocean_cells])
        eta_mean_fw = jnp.mean(state_fw.eta.data[ocean_cells])
        assert eta_mean_fw > eta_mean_no

    def test_step_with_precip_decreases_S(self, mesh, z_coord, state0):
        """Precipitation should decrease top-layer salinity via virtual salt flux."""
        config = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        mask = state0.land_mask.data

        fw = FreshwaterForcing(
            precip=jnp.ones(mesh.nCells) * 1e-3 * mask,
            evap=jnp.zeros(mesh.nCells),
            runoff=jnp.zeros(mesh.nCells),
            ice_fw=jnp.zeros(mesh.nCells),
        )

        state_no = model.step(state0, 60.0)
        state_fw = model.step(state0, 60.0, freshwater=fw)

        # Mean top-layer S should be lower with precip
        ocean_cells = mask > 0.5
        S_top_no = jnp.mean(state_no.S.data[ocean_cells, 0])
        S_top_fw = jnp.mean(state_fw.S.data[ocean_cells, 0])
        assert S_top_fw < S_top_no

    def test_step_with_evap_increases_S(self, mesh, z_coord, state0):
        """Evaporation should increase top-layer salinity."""
        config = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        mask = state0.land_mask.data

        fw = FreshwaterForcing(
            precip=jnp.zeros(mesh.nCells),
            evap=jnp.ones(mesh.nCells) * 1e-3 * mask,
            runoff=jnp.zeros(mesh.nCells),
            ice_fw=jnp.zeros(mesh.nCells),
        )

        state_no = model.step(state0, 60.0)
        state_fw = model.step(state0, 60.0, freshwater=fw)

        ocean_cells = mask > 0.5
        S_top_no = jnp.mean(state_no.S.data[ocean_cells, 0])
        S_top_fw = jnp.mean(state_fw.S.data[ocean_cells, 0])
        assert S_top_fw > S_top_no

    def test_multi_step_stability(self, mesh, z_coord, state0):
        """10 steps with moderate freshwater should remain stable."""
        config = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=True,
        )
        model = MPASOceanModel(mesh, z_coord, config)
        mask = state0.land_mask.data

        # Typical precip rate: ~3 mm/day = 3.5e-8 kg/m2/s
        fw = FreshwaterForcing(
            precip=jnp.ones(mesh.nCells) * 3.5e-8 * mask,
            evap=jnp.ones(mesh.nCells) * 2.0e-8 * mask,
            runoff=jnp.ones(mesh.nCells) * 0.5e-8 * mask,
            ice_fw=jnp.zeros(mesh.nCells),
        )

        state = state0
        for _ in range(10):
            state = model.step(state, 60.0, freshwater=fw)

        assert jnp.all(jnp.isfinite(state.eta.data))
        assert jnp.all(jnp.isfinite(state.T.data))
        assert jnp.all(jnp.isfinite(state.S.data))
        assert jnp.all(jnp.isfinite(state.u.data))

    def test_freshwater_none_closure_ignores_forcing(self, mesh, z_coord, state0):
        """With freshwater_closure='none', forcing should be ignored."""
        config_none = MPASOceanConfig(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
            freshwater_closure="none",
        )
        model = MPASOceanModel(mesh, z_coord, config_none)
        mask = state0.land_mask.data

        # Large freshwater forcing
        fw = FreshwaterForcing(
            precip=jnp.ones(mesh.nCells) * 1e-2 * mask,
            evap=jnp.zeros(mesh.nCells),
            runoff=jnp.zeros(mesh.nCells),
            ice_fw=jnp.zeros(mesh.nCells),
        )

        state_fw = model.step(state0, 60.0, freshwater=fw)
        state_no = model.step(state0, 60.0)

        # With closure="none", freshwater should be completely ignored
        assert jnp.allclose(state_fw.eta.data, state_no.eta.data, atol=1e-15)
        assert jnp.allclose(state_fw.S.data, state_no.S.data, atol=1e-15)


# ============================================================================
# Integration with latlon C-grid ocean model
# ============================================================================

@pytest.fixture
def ll_grid():
    return create_latlon_grid(n_lat=18, n_lon=36)


@pytest.fixture
def ll_z_coord():
    return create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0,
    )


@pytest.fixture
def ll_state0(ll_grid, ll_z_coord):
    return rest_state_latlon_cgrid_ocean(
        ll_grid, ll_z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=500.0, land_lat_threshold=85.0,
    )


class TestLatLonCGridOceanModelWithFreshwater:

    def test_step_no_freshwater(self, ll_grid, ll_z_coord, ll_state0):
        """Model step without freshwater should work as before."""
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
            freshwater_closure="none",
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config)
        state1 = model.step(ll_state0, 60.0)
        assert jnp.all(jnp.isfinite(state1.eta.data))
        assert jnp.all(jnp.isfinite(state1.S.data))

    def test_step_with_zero_freshwater(self, ll_grid, ll_z_coord, ll_state0):
        """Step with zero freshwater = step without freshwater."""
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config)
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon
        fw = FreshwaterForcing(
            precip=jnp.zeros((n_lat, n_lon)),
            evap=jnp.zeros((n_lat, n_lon)),
            runoff=jnp.zeros((n_lat, n_lon)),
            ice_fw=jnp.zeros((n_lat, n_lon)),
        )
        state_fw = model.step(ll_state0, 60.0, freshwater=fw)
        state_no = model.step(ll_state0, 60.0)
        assert jnp.allclose(state_fw.eta.data, state_no.eta.data, atol=1e-15)
        assert jnp.allclose(state_fw.S.data, state_no.S.data, atol=1e-15)

    def test_step_with_precip_raises_eta(self, ll_grid, ll_z_coord, ll_state0):
        """Precipitation should raise the free surface."""
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config)
        mask = ll_state0.land_mask.data
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon

        fw = FreshwaterForcing(
            precip=jnp.ones((n_lat, n_lon)) * 1e-3 * mask,
            evap=jnp.zeros((n_lat, n_lon)),
            runoff=jnp.zeros((n_lat, n_lon)),
            ice_fw=jnp.zeros((n_lat, n_lon)),
        )

        state_no = model.step(ll_state0, 60.0)
        state_fw = model.step(ll_state0, 60.0, freshwater=fw)

        ocean_cells = mask > 0.5
        eta_mean_no = jnp.mean(state_no.eta.data[ocean_cells])
        eta_mean_fw = jnp.mean(state_fw.eta.data[ocean_cells])
        assert eta_mean_fw > eta_mean_no

    def test_step_with_precip_decreases_S(self, ll_grid, ll_z_coord, ll_state0):
        """Precipitation should decrease top-layer salinity via virtual salt flux."""
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config)
        mask = ll_state0.land_mask.data
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon

        fw = FreshwaterForcing(
            precip=jnp.ones((n_lat, n_lon)) * 1e-3 * mask,
            evap=jnp.zeros((n_lat, n_lon)),
            runoff=jnp.zeros((n_lat, n_lon)),
            ice_fw=jnp.zeros((n_lat, n_lon)),
        )

        state_no = model.step(ll_state0, 60.0)
        state_fw = model.step(ll_state0, 60.0, freshwater=fw)

        ocean_cells = mask > 0.5
        S_top_no = jnp.mean(state_no.S.data[ocean_cells, 0])
        S_top_fw = jnp.mean(state_fw.S.data[ocean_cells, 0])
        assert S_top_fw < S_top_no

    def test_step_with_evap_increases_S(self, ll_grid, ll_z_coord, ll_state0):
        """Evaporation should increase top-layer salinity."""
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config)
        mask = ll_state0.land_mask.data
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon

        fw = FreshwaterForcing(
            precip=jnp.zeros((n_lat, n_lon)),
            evap=jnp.ones((n_lat, n_lon)) * 1e-3 * mask,
            runoff=jnp.zeros((n_lat, n_lon)),
            ice_fw=jnp.zeros((n_lat, n_lon)),
        )

        state_no = model.step(ll_state0, 60.0)
        state_fw = model.step(ll_state0, 60.0, freshwater=fw)

        ocean_cells = mask > 0.5
        S_top_no = jnp.mean(state_no.S.data[ocean_cells, 0])
        S_top_fw = jnp.mean(state_fw.S.data[ocean_cells, 0])
        assert S_top_fw > S_top_no

    def test_multi_step_stability(self, ll_grid, ll_z_coord, ll_state0):
        """10 steps with moderate freshwater should remain stable."""
        config = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=True,
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config)
        mask = ll_state0.land_mask.data
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon

        fw = FreshwaterForcing(
            precip=jnp.ones((n_lat, n_lon)) * 3.5e-8 * mask,
            evap=jnp.ones((n_lat, n_lon)) * 2.0e-8 * mask,
            runoff=jnp.ones((n_lat, n_lon)) * 0.5e-8 * mask,
            ice_fw=jnp.zeros((n_lat, n_lon)),
        )

        state = ll_state0
        for _ in range(10):
            state = model.step(state, 60.0, freshwater=fw)

        assert jnp.all(jnp.isfinite(state.eta.data))
        assert jnp.all(jnp.isfinite(state.T.data))
        assert jnp.all(jnp.isfinite(state.S.data))
        assert jnp.all(jnp.isfinite(state.u.data))

    def test_freshwater_none_closure_ignores_forcing(
        self, ll_grid, ll_z_coord, ll_state0,
    ):
        """With freshwater_closure='none', forcing should be ignored."""
        config_none = LatLonCGridOceanConfig.from_flat(
            A_h=1e3, n_barotropic_substeps=5,
            use_conservation_fixer=False,
            freshwater_closure="none",
        )
        model = LatLonCGridOceanModel(ll_grid, ll_z_coord, config_none)
        mask = ll_state0.land_mask.data
        n_lat, n_lon = ll_grid.n_lat, ll_grid.n_lon

        fw = FreshwaterForcing(
            precip=jnp.ones((n_lat, n_lon)) * 1e-2 * mask,
            evap=jnp.zeros((n_lat, n_lon)),
            runoff=jnp.zeros((n_lat, n_lon)),
            ice_fw=jnp.zeros((n_lat, n_lon)),
        )

        state_fw = model.step(ll_state0, 60.0, freshwater=fw)
        state_no = model.step(ll_state0, 60.0)

        assert jnp.allclose(state_fw.eta.data, state_no.eta.data, atol=1e-15)
        assert jnp.allclose(state_fw.S.data, state_no.S.data, atol=1e-15)


# ============================================================================
# Differentiability
# ============================================================================

class TestDifferentiability:

    def test_vsf_differentiable(self):
        """Virtual salt flux should be differentiable w.r.t. precip."""
        n = 5

        def loss(precip_val):
            fw = FreshwaterForcing(
                precip=jnp.ones(n) * precip_val,
                evap=jnp.zeros(n),
                runoff=jnp.zeros(n),
                ice_fw=jnp.zeros(n),
            )
            dz = jnp.ones(n) * 20.0
            dS = virtual_salt_flux(fw, S_ref=35.0, dz_0=dz, rho_0=constants.rho_ocean)
            return jnp.sum(dS ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(1e-4)
        assert jnp.isfinite(g)
        assert g != 0.0

    def test_eta_tendency_differentiable(self):
        """Eta tendency should be differentiable."""
        n = 5

        def loss(precip_val):
            fw = FreshwaterForcing(
                precip=jnp.ones(n) * precip_val,
                evap=jnp.zeros(n),
                runoff=jnp.zeros(n),
                ice_fw=jnp.zeros(n),
            )
            deta = freshwater_eta_tendency(fw, rho_0=constants.rho_ocean)
            return jnp.sum(deta ** 2)

        grad_fn = jax.grad(loss)
        g = grad_fn(1e-4)
        assert jnp.isfinite(g)
        assert g != 0.0


# ============================================================================
# Coupler adapter
# ============================================================================

class TestCouplerAdapter:

    def test_compute_mpas_freshwater_import(self):
        from legoesm.coupler.mpas_adapter import compute_mpas_freshwater
        assert callable(compute_mpas_freshwater)

    def test_compute_mpas_freshwater_basic(self):
        from legoesm.coupler.mpas_adapter import compute_mpas_freshwater
        from legoesm.core.coupling_fields import AtmToSurface, SurfaceToAtm

        n = 10
        z = jnp.zeros(n)
        ones = jnp.ones(n)

        atm = AtmToSurface(
            sw_down=z, lw_down=z,
            precip_total=ones * 1e-4,
            precip_snow=z,
            T_lowest=z, q_lowest=z, u_lowest=z, v_lowest=z,
            p_lowest=z, p_surface=z, rho_lowest=z,
            cos_zenith=z, co2_ppmv=z, has_radiation=z, has_precipitation=ones,
        )
        sfc = SurfaceToAtm(
            T_sfc=z, T_rad=z, albedo=z, emissivity=z, z0=z,
            q_surface=z, shflx=z,
            lhflx=ones * 100.0,  # 100 W/m2
            tau_x=z, tau_y=z, lw_up=z,
            u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z,
            # Coupler-conservation channels added in iter-13/14/15/16.
            # Set freshwater/ocean_heat/stress to zero, but populate
            # surface_mass_flux from lhflx so the F22 path can use it.
            freshwater_flux=z,
            ocean_heat_extraction=z,
            ocean_stress_x=z, ocean_stress_y=z,
            surface_mass_flux=ones * 100.0 / constants.L_v,
            salt_flux=z,
            river_runoff_flux=z,
            ice_lake_freshwater_flux=z,
        )
        mask = ones

        fw = compute_mpas_freshwater(atm, sfc, mask, L_v=constants.L_v)
        assert fw.precip.shape == (n,)
        assert jnp.allclose(fw.precip, 1e-4)
        assert jnp.allclose(fw.evap, 100.0 / constants.L_v)


class TestFreshwaterSaltNormalization:
    """Global-salt conservation via area-mean removal of the freshwater flux.

    Without normalization an unbalanced ∮(P-E+R) drifts the global-mean salinity
    even when volume is conserved; ``normalize_freshwater_net`` removes the
    area-weighted ocean mean so the virtual-salt closure conserves global salt.
    """

    def test_normalize_freshwater_net_zero_area_mean(self):
        n = 50
        key = jax.random.PRNGKey(0)
        F = jax.random.normal(key, (n,)) + 2.0      # nonzero-mean flux
        area = 1.0 + jax.random.uniform(jax.random.PRNGKey(1), (n,))
        mask = (jnp.arange(n) % 5 != 0).astype(F.dtype)   # 20% land
        Fn = normalize_freshwater_net(F, area, mask)
        # Area-weighted ocean integral of the normalized flux is ~0.
        integ = float(jnp.sum(Fn * area * mask))
        assert abs(integ) < 1e-9, f"normalized flux integral {integ} != 0"
        # Land cells: the subtraction is masked, so the land flux is F - F_mean*0
        # = F (untouched); only ocean cells are corrected.
        assert jnp.allclose(Fn[mask < 0.5], F[mask < 0.5])

    def test_apply_virtual_salt_normalize_conserves_global_salt(self):
        """apply_freshwater_virtual_salt_top(normalize=True) -> the top-layer
        salt-mass tendency integrates to ~0 over the ocean; normalize=False does
        NOT (the budget is unbalanced)."""
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            apply_freshwater_virtual_salt_top,
        )
        n, nlev = 60, 4
        # Net freshwater = precip only -> strictly positive, nonzero global mean
        # (the unbalanced case that drifts salinity).
        P = 1e-4 * (1.0 + jax.random.uniform(jax.random.PRNGKey(2), (n,)))
        fw = FreshwaterForcing(precip=P, evap=jnp.zeros(n), runoff=jnp.zeros(n),
                               ice_fw=jnp.zeros(n), restoring=jnp.zeros(n))
        area = 1.0 + jax.random.uniform(jax.random.PRNGKey(3), (n,))
        mask = (jnp.arange(n) % 7 != 0).astype(P.dtype)
        h_top = jnp.full((n,), 10.0)
        rho_0, S_ref = 1025.0, 35.0

        def salt_rate(normalize):
            dS = jnp.zeros((n, nlev))
            dS = apply_freshwater_virtual_salt_top(
                dS, fw, S_ref, h_top, rho_0, mask,
                area=area, normalize=normalize)
            # salt-mass rate per area = dS_top * rho_0 * h_top; integrate over ocean
            return float(jnp.sum(dS[:, 0] * rho_0 * h_top * area * mask))

        r_norm = salt_rate(True)
        r_raw = salt_rate(False)
        assert abs(r_norm) < 1e-6, f"normalized salt rate {r_norm} != 0"
        assert abs(r_raw) > 1e-3, f"raw salt rate {r_raw} should be nonzero"

    def test_apply_virtual_salt_normalize_requires_area(self):
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            apply_freshwater_virtual_salt_top,
        )
        n = 10
        fw = zero_freshwater(n)
        with pytest.raises(ValueError):
            apply_freshwater_virtual_salt_top(
                jnp.zeros((n, 3)), fw, 35.0, jnp.full((n,), 10.0), 1025.0,
                jnp.ones(n), normalize=True)  # area=None -> error

    def test_virtual_salt_flux_from_net_matches_wrapper(self):
        n = 20
        P = 1e-4 * jnp.ones(n)
        fw = FreshwaterForcing(precip=P, evap=jnp.zeros(n), runoff=jnp.zeros(n),
                               ice_fw=jnp.zeros(n), restoring=jnp.zeros(n))
        h = jnp.full((n,), 10.0)
        a = virtual_salt_flux(fw, 35.0, h, 1025.0)
        b = virtual_salt_flux_from_net(net_freshwater_flux(fw), 35.0, h, 1025.0)
        assert jnp.allclose(a, b)


class TestNormalizedVirtualSaltFluxHelper:
    """The shared global-salt-conserving helper used by BOTH the MPAS and lat-lon
    cores (factored so the OMIP correction is implemented once)."""

    def test_helper_conserves_global_salt(self):
        n = 80
        P = 1e-4 * (1.0 + jax.random.uniform(jax.random.PRNGKey(5), (n,)))
        fw = FreshwaterForcing(precip=P, evap=jnp.zeros(n), runoff=jnp.zeros(n),
                               ice_fw=jnp.zeros(n), restoring=jnp.zeros(n))
        area = 1.0 + jax.random.uniform(jax.random.PRNGKey(6), (n,))
        mask = (jnp.arange(n) % 6 != 0).astype(P.dtype)
        h = jnp.full((n,), 10.0)
        dS = normalized_virtual_salt_flux(fw, 35.0, h, 1025.0, area, mask)
        wet = mask * (h > 1e-3)
        # salt-mass rate = dS * rho * h; integrates to ~0 over the wet ocean.
        rate = float(jnp.sum(dS * 1025.0 * h * area * wet))
        assert abs(rate) < 1e-6, f"helper salt rate {rate} != 0"
        # vs the raw flux: nonzero.
        raw = virtual_salt_flux(fw, 35.0, h, 1025.0)
        assert abs(float(jnp.sum(raw * 1025.0 * h * area * wet))) > 1e-3

    def test_helper_excludes_restoring_channel(self):
        """The restoring channel passes through UN-normalized (it is a local
        relaxation, not a globally-redistributed physical freshwater flux)."""
        n = 40
        rstr = 1e-4 * jnp.ones(n)   # uniform restoring (nonzero mean)
        fw = FreshwaterForcing(precip=jnp.zeros(n), evap=jnp.zeros(n),
                               runoff=jnp.zeros(n), ice_fw=jnp.zeros(n),
                               restoring=rstr)
        area = jnp.ones(n)
        mask = jnp.ones(n)
        h = jnp.full((n,), 10.0)
        dS = normalized_virtual_salt_flux(fw, 35.0, h, 1025.0, area, mask)
        # Physical net is 0 -> normalization leaves only the restoring; dS equals
        # the raw restoring-only virtual salt (restoring NOT zeroed by the mean).
        dS_raw = virtual_salt_flux_from_net(rstr, 35.0, h, 1025.0)
        assert jnp.allclose(dS, dS_raw)


class TestLatLonSaltNormalizationWiring:
    """The lat-lon config gained ``normalize_freshwater``; the core routes through
    the same shared helper.  Smoke: a freshwater step runs + stays finite with the
    flag on (the analytic conservation is covered by the helper test above)."""

    def test_latlon_normalize_freshwater_step_runs(self):
        nlat, nlon = 24, 48
        grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)
        z = create_ocean_z_star(n_levels=4, H_max=4000.0)
        st = rest_state_latlon_cgrid_ocean(grid, z, H_max=4000.0)
        # STRONGLY NON-UNIFORM precip (SH-heavy) so normalization removes a
        # nonzero global mean -> a clear, above-float32-noise difference from the
        # raw path; large amplitude (2e-3) so a single step exceeds float32 eps
        # at S~35 (~4e-6).
        lat2d = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None], (nlat, nlon))
        precip = jnp.where(lat2d < 0.0, 2.0e-3, 0.0).astype(jnp.float64)
        fw = FreshwaterForcing(
            precip=precip, evap=jnp.zeros((nlat, nlon)),
            runoff=jnp.zeros((nlat, nlon)), ice_fw=jnp.zeros((nlat, nlon)),
            restoring=jnp.zeros((nlat, nlon)))

        def step(normalize):
            cfg = LatLonCGridOceanConfig.from_flat(
                freshwater_closure="virtual_salt_flux", normalize_freshwater=normalize)
            return LatLonCGridOceanModel(grid, z, cfg).step(st, 600.0, freshwater=fw)

        st_on = step(True)
        st_off = step(False)
        assert bool(jnp.isfinite(st_on.S.data).all())
        # The flag MUST change the surface salinity (normalization removes the
        # nonzero global mean of the flux), by well above float32 rounding noise.
        dS = float(jnp.max(jnp.abs(st_on.S.data[..., 0] - st_off.S.data[..., 0])))
        assert dS > 1e-4, f"normalize_freshwater had no effect on latlon S (dS={dS})"

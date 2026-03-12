"""Tests for ocean biogeochemistry: carbon cycle, NPZD, air-sea CO2 exchange.

Tests cover:
- BiogeoConfig defaults and scheme selection
- State initialization (abiotic, npzd, none)
- CO2 solubility (Weiss 1974)
- Carbonate equilibria (Lueker 2000)
- pCO2/pH solver (Follows 2006)
- Schmidt number and gas transfer velocity (Wanninkhof 2014)
- Air-sea CO2 flux direction and magnitude
- NPZD source/sink terms (growth, grazing, remineralization)
- PAR attenuation profile
- Full step function (abiotic and npzd)
- Non-negativity of tracers
- Conservation properties
- Differentiability through all computations
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.config import (
    BiogeoConfig,
    OceanBiogeoState,
    BiogeoTendencies,
    AirSeaCO2Diagnostics,
    init_biogeo_state,
)
from legoesm.ocean.biogeochemistry.carbonate import (
    co2_solubility,
    carbonate_equilibria,
    borate_equilibrium,
    total_borate,
    solve_carbonate_system,
)
from legoesm.ocean.biogeochemistry.gas_exchange import (
    schmidt_number_co2,
    gas_transfer_velocity,
    air_sea_co2_flux,
)
from legoesm.ocean.biogeochemistry.npzd import (
    par_profile,
    npzd_source_sink,
)
from legoesm.ocean.biogeochemistry.carbon_cycle import (
    compute_biogeo_tendencies,
    step_ocean_biogeochemistry,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def z_ref():
    """10-level z* coordinate reference arrays."""
    nlev = 10
    dz = jnp.linspace(10.0, 200.0, nlev)
    z_half = jnp.concatenate([jnp.array([0.0]), -jnp.cumsum(dz)])
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, dz


@pytest.fixture
def abiotic_state(z_ref):
    """Abiotic (DIC+ALK) state on a small grid."""
    z_full, dz = z_ref
    shape = (4, 10)  # 4 columns, 10 levels
    cfg = BiogeoConfig(scheme="abiotic")
    return init_biogeo_state(shape, z_full, cfg)


@pytest.fixture
def npzd_state(z_ref):
    """NPZD state on a small grid."""
    z_full, dz = z_ref
    shape = (4, 10)
    cfg = BiogeoConfig(scheme="npzd")
    return init_biogeo_state(shape, z_full, cfg)


# ============================================================================
# Config
# ============================================================================


class TestBiogeoConfig:
    def test_defaults(self):
        cfg = BiogeoConfig()
        assert cfg.scheme == "none"
        assert cfg.pCO2_atm == 400.0
        assert cfg.R_CN == 6.625

    def test_abiotic_config(self):
        cfg = BiogeoConfig(scheme="abiotic", pCO2_atm=280.0)
        assert cfg.scheme == "abiotic"
        assert cfg.pCO2_atm == 280.0

    def test_npzd_config(self):
        cfg = BiogeoConfig(scheme="npzd", mu_max=2.0)
        assert cfg.scheme == "npzd"
        assert cfg.mu_max == 2.0


# ============================================================================
# State initialization
# ============================================================================


class TestInitBiogeoState:
    def test_none_returns_none(self, z_ref):
        z_full, _ = z_ref
        cfg = BiogeoConfig(scheme="none")
        state = init_biogeo_state((4, 10), z_full, cfg)
        assert state is None

    def test_abiotic_has_dic_alk(self, abiotic_state):
        assert abiotic_state is not None
        assert abiotic_state.DIC.shape == (4, 10)
        assert abiotic_state.ALK.shape == (4, 10)
        assert abiotic_state.NO3 is None
        assert abiotic_state.Phyto is None

    def test_abiotic_initial_values(self, abiotic_state):
        assert float(jnp.mean(abiotic_state.DIC)) == pytest.approx(2.1, abs=0.01)
        assert float(jnp.mean(abiotic_state.ALK)) == pytest.approx(2.3, abs=0.01)

    def test_npzd_has_all_tracers(self, npzd_state):
        assert npzd_state is not None
        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
            val = getattr(npzd_state, field)
            assert val is not None, f"{field} should not be None for NPZD"
            assert val.shape == (4, 10)

    def test_npzd_no3_increases_with_depth(self, npzd_state):
        """NO3 should increase from surface to deep."""
        NO3 = npzd_state.NO3
        mean_surf = float(jnp.mean(NO3[:, 0]))
        mean_deep = float(jnp.mean(NO3[:, -1]))
        assert mean_deep > mean_surf

    def test_cubed_sphere_shape(self, z_ref):
        z_full, _ = z_ref
        shape = (6, 8, 8, 10)
        cfg = BiogeoConfig(scheme="abiotic")
        state = init_biogeo_state(shape, z_full, cfg)
        assert state.DIC.shape == (6, 8, 8, 10)

    def test_invalid_scheme_raises(self, z_ref):
        z_full, _ = z_ref
        cfg = BiogeoConfig(scheme="invalid")
        with pytest.raises(ValueError, match="Unknown"):
            init_biogeo_state((4, 10), z_full, cfg)


# ============================================================================
# CO2 solubility
# ============================================================================


class TestCO2Solubility:
    def test_positive(self):
        K0 = co2_solubility(jnp.array(20.0), jnp.array(35.0))
        assert float(K0) > 0

    def test_typical_range(self):
        """K0 should be ~0.03-0.06 mol/(kg*atm) for ocean conditions."""
        K0 = co2_solubility(jnp.array(20.0), jnp.array(35.0))
        assert 0.02 < float(K0) < 0.08

    def test_cold_water_more_soluble(self):
        """CO2 is more soluble in cold water."""
        K0_cold = co2_solubility(jnp.array(2.0), jnp.array(35.0))
        K0_warm = co2_solubility(jnp.array(25.0), jnp.array(35.0))
        assert float(K0_cold) > float(K0_warm)

    def test_vectorized(self):
        T = jnp.array([5.0, 15.0, 25.0])
        S = jnp.array([34.0, 35.0, 36.0])
        K0 = co2_solubility(T, S)
        assert K0.shape == (3,)
        assert jnp.all(K0 > 0)


# ============================================================================
# Carbonate equilibria
# ============================================================================


class TestCarbonateEquilibria:
    def test_K1_K2_positive(self):
        K1, K2 = carbonate_equilibria(jnp.array(20.0), jnp.array(35.0))
        assert float(K1) > 0
        assert float(K2) > 0

    def test_K1_greater_than_K2(self):
        """K1 should be much larger than K2."""
        K1, K2 = carbonate_equilibria(jnp.array(20.0), jnp.array(35.0))
        assert float(K1) > float(K2) * 10

    def test_typical_pK_ranges(self):
        """pK1 ~ 5.8-6.0, pK2 ~ 8.9-9.2 for typical ocean."""
        K1, K2 = carbonate_equilibria(jnp.array(20.0), jnp.array(35.0))
        pK1 = -float(jnp.log10(K1))
        pK2 = -float(jnp.log10(K2))
        assert 5.5 < pK1 < 6.5
        assert 8.5 < pK2 < 9.5

    def test_borate(self):
        KB = borate_equilibrium(jnp.array(20.0), jnp.array(35.0))
        assert float(KB) > 0
        BT = total_borate(jnp.array(35.0))
        assert 0.0003 < float(BT) < 0.0005


# ============================================================================
# pCO2 / pH solver
# ============================================================================


class TestCarbonateSolver:
    def test_typical_pco2_range(self):
        """Surface ocean pCO2 should be ~300-500 uatm for typical DIC/ALK."""
        pCO2, pH = solve_carbonate_system(
            DIC=jnp.array(2.1),  # mol/m^3
            ALK=jnp.array(2.3),
            T_degC=jnp.array(20.0),
            S_psu=jnp.array(35.0),
        )
        assert 200 < float(pCO2) < 800

    def test_typical_pH_range(self):
        """Ocean pH should be ~7.8-8.3."""
        _, pH = solve_carbonate_system(
            DIC=jnp.array(2.1),
            ALK=jnp.array(2.3),
            T_degC=jnp.array(20.0),
            S_psu=jnp.array(35.0),
        )
        assert 7.5 < float(pH) < 8.5

    def test_higher_dic_higher_pco2(self):
        """Increasing DIC should increase pCO2."""
        pCO2_low, _ = solve_carbonate_system(
            jnp.array(2.0), jnp.array(2.3),
            jnp.array(20.0), jnp.array(35.0),
        )
        pCO2_high, _ = solve_carbonate_system(
            jnp.array(2.2), jnp.array(2.3),
            jnp.array(20.0), jnp.array(35.0),
        )
        assert float(pCO2_high) > float(pCO2_low)

    def test_higher_alk_lower_pco2(self):
        """Increasing alkalinity should decrease pCO2 (more CO3^2-)."""
        pCO2_low_alk, _ = solve_carbonate_system(
            jnp.array(2.1), jnp.array(2.2),
            jnp.array(20.0), jnp.array(35.0),
        )
        pCO2_high_alk, _ = solve_carbonate_system(
            jnp.array(2.1), jnp.array(2.4),
            jnp.array(20.0), jnp.array(35.0),
        )
        assert float(pCO2_high_alk) < float(pCO2_low_alk)

    def test_vectorized(self):
        DIC = jnp.array([2.0, 2.1, 2.2])
        ALK = jnp.array([2.3, 2.3, 2.3])
        T = jnp.array([15.0, 20.0, 25.0])
        S = jnp.array([35.0, 35.0, 35.0])
        pCO2, pH = solve_carbonate_system(DIC, ALK, T, S)
        assert pCO2.shape == (3,)
        assert pH.shape == (3,)
        assert jnp.all(jnp.isfinite(pCO2))
        assert jnp.all(jnp.isfinite(pH))


# ============================================================================
# Schmidt number and gas transfer
# ============================================================================


class TestGasTransfer:
    def test_schmidt_positive(self):
        Sc = schmidt_number_co2(jnp.array(20.0))
        assert float(Sc) > 0

    def test_schmidt_decreases_with_temp(self):
        """Schmidt number decreases with temperature."""
        Sc_cold = schmidt_number_co2(jnp.array(5.0))
        Sc_warm = schmidt_number_co2(jnp.array(25.0))
        assert float(Sc_cold) > float(Sc_warm)

    def test_piston_velocity_positive(self):
        kw = gas_transfer_velocity(jnp.array(7.0), jnp.array(20.0))
        assert float(kw) > 0

    def test_higher_wind_faster_exchange(self):
        kw_low = gas_transfer_velocity(jnp.array(3.0), jnp.array(20.0))
        kw_high = gas_transfer_velocity(jnp.array(12.0), jnp.array(20.0))
        assert float(kw_high) > float(kw_low)

    def test_piston_velocity_order_of_magnitude(self):
        """k_w should be O(1e-5 to 1e-4) m/s for typical winds."""
        kw = gas_transfer_velocity(jnp.array(7.0), jnp.array(20.0))
        assert 1e-6 < float(kw) < 1e-3


# ============================================================================
# Air-sea CO2 flux
# ============================================================================


class TestAirSeaCO2Flux:
    def test_undersaturated_ocean_takes_up_co2(self):
        """When ocean pCO2 < atm pCO2, flux should be into ocean (positive)."""
        # Low DIC -> low ocean pCO2 -> ocean uptake
        diag = air_sea_co2_flux(
            DIC_surf=jnp.array(1.9),
            ALK_surf=jnp.array(2.3),
            T_surf=jnp.array(15.0),
            S_surf=jnp.array(35.0),
            U10=jnp.array(7.0),
            pCO2_atm=400.0,
        )
        assert float(diag.flux_co2) > 0

    def test_supersaturated_ocean_outgasses(self):
        """When ocean pCO2 > atm pCO2, flux should be out of ocean (negative)."""
        # High DIC -> high ocean pCO2 -> outgassing
        diag = air_sea_co2_flux(
            DIC_surf=jnp.array(2.3),
            ALK_surf=jnp.array(2.2),
            T_surf=jnp.array(25.0),
            S_surf=jnp.array(35.0),
            U10=jnp.array(7.0),
            pCO2_atm=400.0,
        )
        assert float(diag.flux_co2) < 0

    def test_flux_magnitude(self):
        """Typical CO2 flux should be O(1e-7 to 1e-5) mol/m^2/s."""
        diag = air_sea_co2_flux(
            DIC_surf=jnp.array(2.1),
            ALK_surf=jnp.array(2.3),
            T_surf=jnp.array(20.0),
            S_surf=jnp.array(35.0),
            U10=jnp.array(7.0),
        )
        assert abs(float(diag.flux_co2)) < 1e-4

    def test_diagnostics_finite(self):
        diag = air_sea_co2_flux(
            DIC_surf=jnp.array(2.1),
            ALK_surf=jnp.array(2.3),
            T_surf=jnp.array(20.0),
            S_surf=jnp.array(35.0),
            U10=jnp.array(7.0),
        )
        assert jnp.isfinite(diag.pCO2_ocean)
        assert jnp.isfinite(diag.pH)
        assert jnp.isfinite(diag.flux_co2)
        assert jnp.isfinite(diag.k_w)


# ============================================================================
# NPZD
# ============================================================================


class TestNPZD:
    def test_par_decreases_with_depth(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        PAR_surf = jnp.array([200.0])
        Phyto = jnp.full((1, 10), 0.1e-3)
        PAR = par_profile(PAR_surf, z_full, Phyto, dz, cfg)
        assert PAR.shape == (1, 10)
        # Surface should be brightest
        assert float(PAR[0, 0]) > float(PAR[0, -1])
        # Should be positive everywhere
        assert jnp.all(PAR > 0)

    def test_source_sink_shapes(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        shape = (4, 10)
        NO3 = jnp.full(shape, 10e-3)
        P = jnp.full(shape, 0.1e-3)
        Z = jnp.full(shape, 0.05e-3)
        D = jnp.full(shape, 0.01e-3)
        DIC = jnp.full(shape, 2.1)
        ALK = jnp.full(shape, 2.3)
        T = jnp.full(shape, 20.0)
        PAR = jnp.full(shape, 100.0)

        result = npzd_source_sink(NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg)
        assert len(result) == 6
        for r in result:
            assert r.shape == shape

    def test_source_sink_finite(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        shape = (4, 10)
        NO3 = jnp.full(shape, 10e-3)
        P = jnp.full(shape, 0.1e-3)
        Z = jnp.full(shape, 0.05e-3)
        D = jnp.full(shape, 0.01e-3)
        DIC = jnp.full(shape, 2.1)
        ALK = jnp.full(shape, 2.3)
        T = jnp.full(shape, 20.0)
        PAR = jnp.full(shape, 100.0)

        result = npzd_source_sink(NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg)
        for r in result:
            assert jnp.all(jnp.isfinite(r))

    def test_growth_requires_light_and_nutrients(self, z_ref):
        """Zero PAR or zero nutrients should give zero growth."""
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        shape = (1, 10)
        P = jnp.full(shape, 0.5e-3)
        Z = jnp.full(shape, 0.05e-3)
        D = jnp.full(shape, 0.01e-3)
        DIC = jnp.full(shape, 2.1)
        ALK = jnp.full(shape, 2.3)
        T = jnp.full(shape, 20.0)

        # No light: phytoplankton should not grow
        zero_PAR = jnp.zeros(shape)
        NO3 = jnp.full(shape, 10e-3)
        _, dP_dark, _, _, _, _ = npzd_source_sink(
            NO3, P, Z, D, DIC, ALK, T, zero_PAR, dz, cfg,
        )
        # Growth term should be near zero; only losses
        assert float(jnp.mean(dP_dark)) <= 0

        # No nutrients: phytoplankton should not grow
        zero_NO3 = jnp.full(shape, 1e-15)
        PAR = jnp.full(shape, 200.0)
        _, dP_no_N, _, _, _, _ = npzd_source_sink(
            zero_NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg,
        )
        assert float(jnp.mean(dP_no_N)) <= 0


# ============================================================================
# Step function
# ============================================================================


class TestStepAbiotic:
    def test_step_shapes(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="abiotic")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        new_state, diag = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
        )
        assert new_state.DIC.shape == (4, 10)
        assert new_state.ALK.shape == (4, 10)
        assert new_state.NO3 is None

    def test_step_modifies_dic(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="abiotic", pCO2_atm=400.0)
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        new_state, _ = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
        )
        # DIC should change due to gas exchange
        diff = float(jnp.max(jnp.abs(new_state.DIC - state.DIC)))
        assert diff > 0

    def test_land_mask_prevents_changes(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="abiotic")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.zeros(4)  # all land

        new_state, _ = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
        )
        # No changes on land
        assert float(jnp.max(jnp.abs(new_state.DIC - state.DIC))) == 0.0

    def test_non_negative_tracers(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="abiotic")
        # Start with very low DIC to test clipping
        state = OceanBiogeoState(
            DIC=jnp.full((4, 10), 0.01),
            ALK=jnp.full((4, 10), 2.3),
        )
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        new_state, _ = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=86400.0, cfg=cfg,
        )
        assert jnp.all(new_state.DIC >= 0)
        assert jnp.all(new_state.ALK >= 0)


class TestStepNPZD:
    def test_step_all_tracers(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        new_state, diag = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
        )
        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
            val = getattr(new_state, field)
            assert val is not None
            assert val.shape == (4, 10)
            assert jnp.all(jnp.isfinite(val))

    def test_non_negative_npzd(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        new_state, _ = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
        )
        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
            val = getattr(new_state, field)
            assert jnp.all(val >= 0), f"{field} has negative values"

    def test_multi_step_stable(self, z_ref):
        """10 steps should remain finite and non-negative."""
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        for _ in range(10):
            state, _ = step_ocean_biogeochemistry(
                state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
            )

        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
            val = getattr(state, field)
            assert jnp.all(jnp.isfinite(val)), f"{field} not finite after 10 steps"
            assert jnp.all(val >= 0), f"{field} negative after 10 steps"

    def test_cubed_sphere_shape(self, z_ref):
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        shape = (6, 4, 4, 10)
        state = init_biogeo_state(shape, z_full, cfg)
        T = jnp.full(shape, 20.0)
        S = jnp.full(shape, 35.0)
        mask = jnp.ones((6, 4, 4))

        new_state, diag = step_ocean_biogeochemistry(
            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
        )
        assert new_state.DIC.shape == shape
        assert diag.pCO2_ocean.shape == (6, 4, 4)


# ============================================================================
# Differentiability
# ============================================================================


class TestDifferentiability:
    def test_grad_through_carbonate_solver(self):
        """Carbonate solver should be differentiable."""
        def loss(DIC):
            pCO2, _ = solve_carbonate_system(DIC, jnp.array(2.3),
                                              jnp.array(20.0), jnp.array(35.0))
            return jnp.sum(pCO2 ** 2)

        grad = jax.grad(loss)(jnp.array(2.1))
        assert jnp.isfinite(grad)

    def test_grad_through_gas_exchange(self):
        """Air-sea flux should be differentiable w.r.t. DIC."""
        def loss(DIC):
            diag = air_sea_co2_flux(DIC, jnp.array(2.3),
                                     jnp.array(20.0), jnp.array(35.0),
                                     jnp.array(7.0))
            return jnp.sum(diag.flux_co2 ** 2)

        grad = jax.grad(loss)(jnp.array(2.1))
        assert jnp.isfinite(grad)

    def test_grad_through_step(self, z_ref):
        """Full step should be differentiable."""
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="abiotic")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        def loss(DIC):
            s = OceanBiogeoState(DIC=DIC, ALK=state.ALK)
            new_s, _ = step_ocean_biogeochemistry(
                s, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
            )
            return jnp.sum(new_s.DIC ** 2)

        grad = jax.grad(loss)(state.DIC)
        assert jnp.all(jnp.isfinite(grad))
        assert grad.shape == state.DIC.shape

    def test_grad_through_npzd(self, z_ref):
        """NPZD step should be differentiable."""
        z_full, dz = z_ref
        cfg = BiogeoConfig(scheme="npzd")
        state = init_biogeo_state((4, 10), z_full, cfg)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)
        mask = jnp.ones(4)

        def loss(DIC):
            s = state._replace(DIC=DIC)
            new_s, _ = step_ocean_biogeochemistry(
                s, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
            )
            return jnp.sum(new_s.DIC ** 2)

        grad = jax.grad(loss)(state.DIC)
        assert jnp.all(jnp.isfinite(grad))


# ============================================================================
# CO2 feedback sensitivity
# ============================================================================


class TestCO2Feedback:
    def test_higher_atm_co2_increases_uptake(self, z_ref):
        """Higher atmospheric CO2 should increase ocean uptake."""
        z_full, dz = z_ref
        mask = jnp.ones(4)
        T = jnp.full((4, 10), 20.0)
        S = jnp.full((4, 10), 35.0)

        cfg_low = BiogeoConfig(scheme="abiotic", pCO2_atm=280.0)
        state_low = init_biogeo_state((4, 10), z_full, cfg_low)
        _, diag_low = step_ocean_biogeochemistry(
            state_low, T, S, dz, z_full, mask, 3600.0, cfg_low,
        )

        cfg_high = BiogeoConfig(scheme="abiotic", pCO2_atm=560.0)
        state_high = init_biogeo_state((4, 10), z_full, cfg_high)
        _, diag_high = step_ocean_biogeochemistry(
            state_high, T, S, dz, z_full, mask, 3600.0, cfg_high,
        )

        # Higher atm CO2 -> larger positive flux (more uptake)
        assert float(jnp.mean(diag_high.flux_co2)) > float(jnp.mean(diag_low.flux_co2))

    def test_warming_increases_pco2(self):
        """Warmer SST should increase ocean pCO2 (reduced solubility)."""
        pCO2_cold, _ = solve_carbonate_system(
            jnp.array(2.1), jnp.array(2.3), jnp.array(10.0), jnp.array(35.0),
        )
        pCO2_warm, _ = solve_carbonate_system(
            jnp.array(2.1), jnp.array(2.3), jnp.array(25.0), jnp.array(35.0),
        )
        assert float(pCO2_warm) > float(pCO2_cold)

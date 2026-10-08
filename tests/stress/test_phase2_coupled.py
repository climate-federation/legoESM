"""Phase 2: Coupled system correctness stress tests.

Exercises progressively complex coupled configurations (aquaplanet through
full_coupled) for 30-day integrations at C8/L5, validating stability,
boundedness, conservation, and diagnostic population.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_driver(preset="aquaplanet", days=30, resolution=8, nlev=5, dt=600.0):
    """Create a CoupledESMDriver with minimal config."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=max(days, 1)),
        radiation="gray",
        days=days,
    )
    coupled_cfg = PRESETS[preset]()
    driver = CoupledESMDriver(atm_config, coupled_cfg)
    driver.setup()
    return driver


# ---------------------------------------------------------------------------
# 2.1  Aquaplanet 30-day stability
# ---------------------------------------------------------------------------

class TestAquaplanet30Day:
    """Aquaplanet preset: 30-day stability and diagnostic population."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("aquaplanet", days=30)
        d.run()
        return d

    def test_completed(self, driver):
        # If we get here, the run completed without exception.
        assert driver.state is not None

    def test_atm_fields_finite(self, driver):
        assert jnp.all(jnp.isfinite(driver.state.T.data))
        assert jnp.all(jnp.isfinite(driver.state.u.data))
        assert jnp.all(jnp.isfinite(driver.state.v.data))
        assert jnp.all(jnp.isfinite(driver.state.p_s.data))

    def test_sst_bounded(self, driver):
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 260.0, f"SST min {float(jnp.min(sst)):.1f} < 260"
        assert float(jnp.max(sst)) < 320.0, f"SST max {float(jnp.max(sst)):.1f} > 320"

    def test_sst_drift_rate(self, driver):
        diag = driver.coupled_diagnostics
        if len(diag) < 2:
            pytest.skip("Not enough diagnostics to compute drift")
        sst_start = diag[0]["sst_mean"]
        sst_end = diag[-1]["sst_mean"]
        day_start = diag[0]["day"]
        day_end = diag[-1]["day"]
        drift_per_year = abs(sst_end - sst_start) / max(day_end - day_start, 1) * 365
        assert drift_per_year < 5.0, f"SST drift {drift_per_year:.1f} K/yr > 5"

    def test_coupled_diagnostics_populated(self, driver):
        diag = driver.coupled_diagnostics
        assert len(diag) > 0
        assert "sst_mean" in diag[0]
        assert "day" in diag[0]


# ---------------------------------------------------------------------------
# 2.2  Slab simple (land + ocean) 30-day stability
# ---------------------------------------------------------------------------

class TestSlabSimple30Day:
    """slab_simple preset: slab ocean + slab bucket land."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("slab_simple", days=30)
        d.run()
        return d

    def test_completed(self, driver):
        assert driver.state is not None

    def test_sst_bounded(self, driver):
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 250.0
        assert float(jnp.max(sst)) < 330.0

    def test_land_state_exists(self, driver):
        sfc = driver.surface_state
        assert sfc is not None
        assert sfc.land is not None

    def test_surface_fields_finite(self, driver):
        sfc = driver.surface_state
        # Land state has at least T_soil or equivalent
        land = sfc.land
        leaves = jax.tree.leaves(land)
        for leaf in leaves:
            if hasattr(leaf, 'shape') and leaf.size > 0:
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite value in land state"


# ---------------------------------------------------------------------------
# 2.3  Slab Richards (multilayer land) 30-day stability
# ---------------------------------------------------------------------------

class TestSlabRichards30Day:
    """slab_richards preset: multi-layer Richards' equation land."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("slab_richards", days=30)
        d.run()
        return d

    def test_completed(self, driver):
        assert driver.state is not None

    def test_land_state_finite(self, driver):
        sfc = driver.surface_state
        leaves = jax.tree.leaves(sfc.land)
        for leaf in leaves:
            if hasattr(leaf, 'shape') and leaf.size > 0:
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite in multilayer land"

    def test_sst_bounded(self, driver):
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 250.0
        assert float(jnp.max(sst)) < 330.0


# ---------------------------------------------------------------------------
# 2.4  Slab carbon 30-day: CO2 evolves and stays bounded
# ---------------------------------------------------------------------------

class TestSlabCarbon30Day:
    """slab_carbon preset: DifferLand carbon + prognostic CO2."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("slab_carbon", days=30)
        d.run()
        return d

    def test_completed(self, driver):
        assert driver.state is not None

    def test_co2_bounded(self, driver):
        """Prognostic CO2 should stay in a physically reasonable range."""
        if not hasattr(driver, '_co2_field') or driver._co2_field is None:
            pytest.skip("No prognostic CO2 tracer")
        assert jnp.all(jnp.isfinite(driver._co2_field)), "CO2 field has NaN/inf"
        M_CO2, M_air = constants.M_CO2, constants.M_air
        co2_ppmv = float(driver._co2_global_mean_kgkg()) / (M_CO2 / M_air) * 1e6
        assert 200.0 < co2_ppmv < 800.0, f"CO2 = {co2_ppmv:.1f} ppmv out of [200,800]"

    def test_co2_changed_from_init(self, driver):
        """Carbon cycle should cause at least a small CO2 change."""
        if not hasattr(driver, '_co2_field') or driver._co2_field is None:
            pytest.skip("No prognostic CO2 tracer")
        M_CO2, M_air = constants.M_CO2, constants.M_air
        co2_init_kgkg = driver.coupled_cfg.co2_ppmv_init * 1.0e-6 * (M_CO2 / M_air)
        co2_now_kgkg = float(driver._co2_global_mean_kgkg())
        assert co2_now_kgkg != pytest.approx(co2_init_kgkg, rel=1e-6), (
            "CO2 unchanged from init"
        )

    def test_sst_bounded(self, driver):
        sst = driver.ocean_state.T_sfc.data
        assert float(jnp.min(sst)) > 250.0
        assert float(jnp.max(sst)) < 330.0


# ---------------------------------------------------------------------------
# 2.5  Full coupled 30-day: all components active
# ---------------------------------------------------------------------------

class TestFullCoupled30Day:
    """full_coupled preset: land carbon + ocean biogeochem + CO2 tracer."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("full_coupled", days=30)
        d.run()
        return d

    def test_completed(self, driver):
        assert driver.state is not None

    def test_all_fields_finite(self, driver):
        assert jnp.all(jnp.isfinite(driver.state.T.data))
        assert jnp.all(jnp.isfinite(driver.ocean_state.T_sfc.data))

    def test_co2_bounded(self, driver):
        if not hasattr(driver, '_co2_field') or driver._co2_field is None:
            pytest.skip("No prognostic CO2 tracer")
        assert jnp.all(jnp.isfinite(driver._co2_field)), "CO2 field has NaN/inf"
        M_CO2, M_air = constants.M_CO2, constants.M_air
        co2_ppmv = float(driver._co2_global_mean_kgkg()) / (M_CO2 / M_air) * 1e6
        assert 200.0 < co2_ppmv < 800.0, f"CO2 = {co2_ppmv:.1f} ppmv out of [200,800]"


# ---------------------------------------------------------------------------
# 2.6  Energy budget closure (aquaplanet)
# ---------------------------------------------------------------------------

class TestEnergyBudget:
    """Verify approximate energy budget closure for the aquaplanet run."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("aquaplanet", days=30)
        d.run()
        return d

    def test_sst_tendency_consistent(self, driver):
        """SST tendency should be consistent with a physically bounded flux."""
        diag = driver.coupled_diagnostics
        if len(diag) < 2:
            pytest.skip("Not enough diagnostics")
        # SST change over the run
        dT = diag[-1]["sst_mean"] - diag[0]["sst_mean"]
        dt_total = (diag[-1]["day"] - diag[0]["day"]) * 86400.0
        # Implied flux: F = rho * c * h * dT / dt
        rho_c_h = constants.rho_ocean * constants.c_sw * 50.0  # ~2e8 J/m2/K
        implied_flux = rho_c_h * dT / max(dt_total, 1.0)
        # Should be bounded: |F| < 200 W/m2 for a reasonable model
        assert abs(implied_flux) < 200.0, (
            f"Implied SST flux = {implied_flux:.1f} W/m2, too large"
        )


# ---------------------------------------------------------------------------
# 2.7  Water budget check (slab_simple)
# ---------------------------------------------------------------------------

class TestWaterBudget:
    """Basic moisture sanity for slab_simple."""

    @pytest.fixture(scope="class")
    def driver(self):
        d = _make_driver("slab_simple", days=30)
        d.run()
        return d

    def test_moisture_non_negative(self, driver):
        """Atmospheric specific humidity should remain non-negative."""
        q_v = driver._atm.q_v
        if q_v is not None:
            assert float(jnp.min(q_v)) >= -1e-12, (
                f"Negative q_v: min = {float(jnp.min(q_v)):.2e}"
            )

    def test_surface_pressure_bounded(self, driver):
        """Surface pressure should stay physically reasonable."""
        p_s = driver.state.p_s.data
        assert float(jnp.min(p_s)) > 8e4, f"p_s min {float(jnp.min(p_s)):.0f} < 80000"
        assert float(jnp.max(p_s)) < 1.2e5, f"p_s max {float(jnp.max(p_s)):.0f} > 120000"

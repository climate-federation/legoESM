"""Phase 1C: Radiation with GHG concentration stress tests."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.forcing.external import GHGConfig, get_ghg_at_time


class TestRadiationGHG:
    """Radiation and GHG forcing stress tests (1C)."""

    # -----------------------------------------------------------------
    # 1C.1 -- Gray radiation optical depth sensitivity
    # -----------------------------------------------------------------
    def test_gray_radiation_co2_sensitivity(self):
        """Doubling LW optical depth should reduce OLR (stronger greenhouse).

        Uses a dry atmosphere (q_v=None) with a realistic lapse-rate
        temperature profile so that trapping radiation at depth (where T
        is warmer) and emitting from higher, colder levels produces a
        measurable OLR reduction when optical depth increases.
        """
        ncol = 10
        nlev = 20

        # Build a realistic atmosphere with lapse rate.
        p_sfc = 1e5
        p_top = 100.0
        p_half = jnp.broadcast_to(
            jnp.linspace(p_top, p_sfc, nlev + 1)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

        # Temperature with ~6.5 K/km lapse rate: warm at surface, cold aloft.
        # Use sigma-based profile: T = T_sfc - dT * (1 - sigma)
        sigma_full = p_full / p_sfc  # (ncol, nlev)
        T_sfc_val = 290.0
        T = T_sfc_val - 80.0 * (1.0 - sigma_full)  # ~210K at top, 290K at surface
        T = jnp.maximum(T, 200.0)  # floor at 200K

        sfc_temperature = jnp.full((ncol,), T_sfc_val)
        lat = jnp.linspace(-jnp.pi / 3, jnp.pi / 3, ncol)
        insolation = jnp.full((ncol,), 340.0)

        # Standard optical depth (dry atmosphere: q_v=None)
        cfg_standard = GrayRadiationConfig(tau_equator=7.2)
        out_standard = gray_radiation(
            T, p_full, p_half, sfc_temperature, lat, None, insolation, cfg_standard,
        )

        # Doubled optical depth
        cfg_doubled = GrayRadiationConfig(tau_equator=14.4)
        out_doubled = gray_radiation(
            T, p_full, p_half, sfc_temperature, lat, None, insolation, cfg_doubled,
        )

        # OLR = upward LW flux at TOA (interface index 0)
        olr_standard = out_standard.lw_flux_up[:, 0]
        olr_doubled = out_doubled.lw_flux_up[:, 0]

        assert jnp.all(jnp.isfinite(olr_standard)), "OLR(standard) not finite"
        assert jnp.all(jnp.isfinite(olr_doubled)), "OLR(doubled) not finite"

        # More greenhouse effect -> less OLR
        assert jnp.all(olr_doubled < olr_standard), (
            "OLR with doubled optical depth should be less than standard"
        )

        # Difference should be meaningful (> 5 W/m2 averaged across columns)
        diff = float(jnp.mean(olr_standard - olr_doubled))
        assert diff > 5.0, (
            f"Mean OLR reduction = {diff:.2f} W/m2, expected > 5 W/m2"
        )

    # -----------------------------------------------------------------
    # 1C.2 -- GHG forcing constant mode
    # -----------------------------------------------------------------
    def test_ghg_forcing_constant(self):
        """Constant GHG config should return the specified values."""
        config = GHGConfig(source="constant", co2_ppmv=400.0)
        result = get_ghg_at_time(config, day=0.0)

        assert result["co2_ppmv"] == 400.0, (
            f"Expected co2_ppmv=400.0, got {result['co2_ppmv']}"
        )
        assert result["ch4_ppbv"] > 0, "ch4_ppbv should be positive"
        assert result["n2o_ppbv"] > 0, "n2o_ppbv should be positive"
        assert np.isfinite(result["co2_ppmv"]), "co2_ppmv not finite"
        assert np.isfinite(result["ch4_ppbv"]), "ch4_ppbv not finite"
        assert np.isfinite(result["n2o_ppbv"]), "n2o_ppbv not finite"

    # -----------------------------------------------------------------
    # 1C.3 -- GHG interpolation from synthetic file
    # -----------------------------------------------------------------
    def test_ghg_interpolation_synthetic(self, tmp_path):
        """Linear interpolation of GHG concentrations from a file."""
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not available")

        # Create a synthetic NetCDF with three time points
        times = np.array([0.0, 365.0, 730.0])
        co2 = np.array([300.0, 350.0, 400.0])
        ch4 = np.array([1800.0, 1800.0, 1800.0])
        n2o = np.array([320.0, 320.0, 320.0])

        ds = xr.Dataset(
            {
                "co2_ppmv": ("time", co2),
                "ch4_ppbv": ("time", ch4),
                "n2o_ppbv": ("time", n2o),
            },
            coords={"time": times},
        )
        nc_path = str(tmp_path / "ghg_test.nc")
        ds.to_netcdf(nc_path)
        ds.close()

        config = GHGConfig(source="file", path=nc_path)

        # Interpolate at midpoint of first year
        result = get_ghg_at_time(config, day=182.5)

        # Linear interpolation between 300 and 350 at day 182.5 / 365 ~ 325
        expected_co2 = 300.0 + (350.0 - 300.0) * (182.5 / 365.0)
        assert abs(result["co2_ppmv"] - expected_co2) < 1.0, (
            f"Expected CO2 ~ {expected_co2:.1f}, got {result['co2_ppmv']:.1f}"
        )
        assert abs(result["ch4_ppbv"] - 1800.0) < 1.0, (
            f"CH4 should be ~1800, got {result['ch4_ppbv']:.1f}"
        )
        assert abs(result["n2o_ppbv"] - 320.0) < 1.0, (
            f"N2O should be ~320, got {result['n2o_ppbv']:.1f}"
        )

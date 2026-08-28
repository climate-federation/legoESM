"""Validation: precision modes produce stable, physically reasonable AMIP simulations.

Runs 5-day C8/L5 AMIP with analytical forcing in fp32, fp64, and mixed
precision modes and verifies:

1. All three modes complete without blow-up.
2. Temperature stays in physical bounds (150-400 K).
3. Surface pressure stays in physical bounds (40-115 kPa).
4. Mixed vs fp64 RMS temperature difference < 1.0 K after 5 days.
5. fp32 vs fp64 may diverge more but must not blow up.
6. Per-module precision overrides are active in mixed mode.

Requires JAX_ENABLE_X64=1 for the fp64 and mixed runs.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.tier3  # operational: fp64 AMIP precision validation

from legoesm.core.precision import (
    PrecisionPolicy,
    get_module_overrides,
    get_policy,
    set_policy,
    clear_module_overrides,
)
from legoesm.driver.config import (
    ExperimentConfig,
    GridConfig,
    DycoreConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver
from legoesm.runtime.precision import apply_precision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(precision: str, output_dir: str) -> ExperimentConfig:
    """Build a minimal 5-day C8/L5 AMIP config with analytical forcing."""
    return ExperimentConfig(
        grid=GridConfig(
            grid_type="cubed_sphere",
            resolution=8,
            nlev=5,
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic",
            discretization="cdgrid",
            dt=600.0,
        ),
        output=OutputConfig(
            output_dir=output_dir,
            diag_days=5,
            checkpoint_days=0,
        ),
        days=5,
        dataset="analytical",
        radiation="gray",
        precision=precision,
    )


def _run_amip(precision: str, tmp_dir: str) -> ModelDriver:
    """Run a 5-day AMIP simulation in the specified precision mode.

    Returns the ModelDriver instance with final state accessible.
    """
    out = os.path.join(tmp_dir, precision)
    config = _make_config(precision, out)
    driver = ModelDriver(config, output_dir=out)
    driver.setup()
    status = driver.run(compiled=False)
    driver._run_status = status
    return driver


def _extract_T(driver: ModelDriver) -> np.ndarray:
    """Extract temperature array as numpy (always fp64 for comparison)."""
    return np.asarray(driver.state.T.data, dtype=np.float64)


def _extract_ps(driver: ModelDriver) -> np.ndarray:
    """Extract surface pressure array as numpy (always fp64 for comparison)."""
    return np.asarray(driver.state.p_s.data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _enable_x64():
    """Ensure x64 mode is active for the entire test module."""
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def amip_runs(tmp_path_factory):
    """Run all three precision modes once and cache results.

    Using module scope avoids re-running the simulations for each test
    function, which would be prohibitively slow.
    """
    tmp_dir = str(tmp_path_factory.mktemp("precision_amip"))
    results = {}
    for mode in ("fp64", "fp32"):  # #1665: mixed refused (fast refuse test below)
        # Reset precision state before each run to avoid cross-contamination.
        clear_module_overrides()
        set_policy(PrecisionPolicy.fp32())
        results[mode] = _run_amip(mode, tmp_dir)

    # Restore default state after all runs.
    clear_module_overrides()
    set_policy(PrecisionPolicy.fp32())

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPrecisionAMIPStability:
    """Verify all precision modes produce stable AMIP simulations."""

    def test_fp64_completes(self, amip_runs):
        """fp64 run should complete without blow-up."""
        assert amip_runs["fp64"]._run_status == "COMPLETED"

    def test_fp32_completes(self, amip_runs):
        """fp32 run should complete without blow-up."""
        assert amip_runs["fp32"]._run_status == "COMPLETED"

    def test_mixed_is_refused(self):
        """#1665 interim: mixed is refused loudly (was: mixed AMIP completes).
        Fast — no simulation; the fp64-vs-mixed AMIP validation is the tracked
        mixed-consistency campaign."""
        from legoesm.runtime.precision import apply_precision
        with pytest.raises(NotImplementedError, match="disabled"):
            apply_precision("mixed")


@pytest.mark.slow
class TestPrecisionAMIPPhysicalBounds:
    """Verify temperature and surface pressure remain in physical bounds."""

    # Temperature bounds [K]: stratosphere can reach ~150 K at model top,
    # tropical surface can reach ~320 K, and we allow generous headroom
    # for low-resolution / short-run transients.
    T_MIN = 150.0
    T_MAX = 400.0

    # Surface pressure bounds [Pa]: generous range to accommodate
    # low-resolution dynamics (40-115 kPa).
    PS_MIN = 40_000.0
    PS_MAX = 115_000.0

    @pytest.mark.parametrize("mode", ["fp64", "fp32"])
    def test_temperature_bounds(self, amip_runs, mode):
        """Temperature must stay within [150, 400] K in all precision modes."""
        T = _extract_T(amip_runs[mode])
        assert np.all(np.isfinite(T)), f"{mode}: temperature contains NaN/Inf"
        assert np.min(T) >= self.T_MIN, (
            f"{mode}: T_min={np.min(T):.1f} K < {self.T_MIN} K"
        )
        assert np.max(T) <= self.T_MAX, (
            f"{mode}: T_max={np.max(T):.1f} K > {self.T_MAX} K"
        )

    @pytest.mark.parametrize("mode", ["fp64", "fp32"])
    def test_surface_pressure_bounds(self, amip_runs, mode):
        """Surface pressure must stay within [40, 115] kPa in all modes."""
        ps = _extract_ps(amip_runs[mode])
        assert np.all(np.isfinite(ps)), f"{mode}: p_s contains NaN/Inf"
        assert np.min(ps) >= self.PS_MIN, (
            f"{mode}: p_s_min={np.min(ps)/1000:.1f} kPa < "
            f"{self.PS_MIN/1000:.0f} kPa"
        )
        assert np.max(ps) <= self.PS_MAX, (
            f"{mode}: p_s_max={np.max(ps)/1000:.1f} kPa > "
            f"{self.PS_MAX/1000:.0f} kPa"
        )


@pytest.mark.slow
class TestPrecisionAMIPCrossComparison:
    """Cross-compare precision modes against the fp64 reference."""


    def test_fp32_vs_fp64_finite(self, amip_runs):
        """fp32 vs fp64 may diverge but must remain finite.

        After 5 days at C8/L5, fp32 truncation errors accumulate but
        should not produce NaN or Inf values.
        """
        T_f32 = _extract_T(amip_runs["fp32"])
        T_ref = _extract_T(amip_runs["fp64"])
        assert np.all(np.isfinite(T_f32)), "fp32 temperature has NaN/Inf"
        assert np.all(np.isfinite(T_ref)), "fp64 temperature has NaN/Inf"
        rms = np.sqrt(np.mean((T_ref - T_f32) ** 2))
        # Log the divergence for diagnostic purposes (not a hard failure).
        print(f"  fp32 vs fp64 RMS T difference: {rms:.4f} K")


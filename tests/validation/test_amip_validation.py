"""Tests for the AMIP climatology validation pipeline.

Validates:
- Validation pipeline runs on ModelDriver output
- All diagnostic checks produce finite results
- Report formatting works
- Short AMIP run passes basic physical bounds
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tier3  # operational: full AMIP, real forcing

from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig, OutputConfig
from legoesm.driver.model_driver import ModelDriver
from evaluations.amip_validation import (
    validate_amip_diagnostics,
    print_validation_report,
    ValidationResult,
)


@pytest.mark.slow
class TestAMIPValidation:

    @pytest.fixture(scope="class")
    def amip_output(self, tmp_path_factory):
        """Run a 10-day C8/L5 AMIP and return the output directory."""
        out = tmp_path_factory.mktemp("amip_validation")
        cfg = ExperimentConfig(
            grid=GridConfig(resolution=8, nlev=5),
            dycore=DycoreConfig(dt=600.0, fix_mass=True),
            output=OutputConfig(diag_days=2, output_dir=str(out)),
            days=10,
            dataset="analytical",
            precision="fp64",
        )
        driver = ModelDriver(cfg, output_dir=out)
        driver.setup()
        status = driver.run()
        assert status == "COMPLETED"
        return out

    def test_validation_runs(self, amip_output):
        """validate_amip_diagnostics produces a report."""
        report = validate_amip_diagnostics(amip_output)
        assert report.n_total >= 5
        assert all(np.isfinite(r.value) for r in report.results)

    def test_temperature_in_bounds(self, amip_output):
        """Mean temperature is physically reasonable."""
        report = validate_amip_diagnostics(amip_output)
        T_checks = [r for r in report.results if "T" in r.name]
        assert len(T_checks) >= 1
        for r in T_checks:
            assert 150.0 < r.value < 400.0, f"{r.name} = {r.value}K out of bounds"

    def test_pressure_in_bounds(self, amip_output):
        """Surface pressure is physically reasonable."""
        report = validate_amip_diagnostics(amip_output)
        ps_checks = [r for r in report.results if "p_s" in r.name]
        for r in ps_checks:
            assert 900.0 < r.value < 1100.0, f"{r.name} = {r.value}hPa"

    def test_report_formatting(self, amip_output):
        """Report string contains expected sections."""
        report = validate_amip_diagnostics(amip_output)
        text = print_validation_report(report)
        assert "AMIP Climatology Validation" in text
        assert "PASS" in text or "FAIL" in text
        assert "checks passed" in text

    def test_energy_budget_checked(self, amip_output):
        """Energy budget residual is validated."""
        report = validate_amip_diagnostics(amip_output)
        energy_checks = [r for r in report.results if "Energy" in r.name]
        assert len(energy_checks) >= 1

    def test_moisture_budget_checked(self, amip_output):
        """Moisture budget residual is validated."""
        report = validate_amip_diagnostics(amip_output)
        moisture_checks = [r for r in report.results if "Moisture" in r.name or "moisture" in r.name]
        assert len(moisture_checks) >= 1

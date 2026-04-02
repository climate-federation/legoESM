"""Smoke tests for scripts/plot_amip.py."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pytest

# Add scripts/ to path so we can import plot_amip
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from plot_amip import load_timeseries, plot_amip


@pytest.fixture()
def fake_run_dir(tmp_path):
    """Create a fake AMIP output directory with synthetic timeseries."""
    n = 20
    data = {
        "days": np.arange(n, dtype=np.float64),
        "T_atm": 250.0 + np.random.default_rng(0).standard_normal(n),
        "T_low": 280.0 + np.random.default_rng(1).standard_normal(n),
        "sst": 290.0 + np.random.default_rng(2).standard_normal(n),
        "CWV": 25.0 + np.random.default_rng(3).standard_normal(n),
        "precip": 3.0 + np.random.default_rng(4).standard_normal(n),
        "max_wind": 30.0 + np.random.default_rng(5).standard_normal(n),
        "sw_up_toa": 100.0 * np.ones(n),
        "lw_up_toa": 240.0 * np.ones(n),
        "sw_net_sfc": 180.0 * np.ones(n),
        "lw_net_sfc": -50.0 * np.ones(n),
        "energy_toa_net": np.zeros(n),
        "dry_mass_ps": 101325.0 * np.ones(n),
        "moisture_residual": np.zeros(n),
        "energy_residual": np.zeros(n),
        "sigma": np.linspace(0.05, 0.95, 10),
        "profiles_T": np.tile(np.linspace(200, 300, 10), (n, 1)),
        "profiles_qv": np.tile(np.linspace(0.1, 10, 10), (n, 1)),
    }
    np.savez(tmp_path / "timeseries.npz", **data)
    return tmp_path


class TestLoadTimeseries:
    def test_loads_from_npz(self, fake_run_dir):
        data = load_timeseries(fake_run_dir)
        assert "days" in data
        assert "T_atm" in data
        assert len(data["days"]) == 20

    def test_loads_from_incremental_chunks(self, tmp_path):
        """Incremental chunk loading works when timeseries.npz is absent."""
        incr_dir = tmp_path / "timeseries_incremental"
        incr_dir.mkdir()
        for i in range(3):
            np.savez(
                incr_dir / f"chunk_{i:04d}.npz",
                days=np.arange(i * 5, (i + 1) * 5, dtype=np.float64),
                T_atm=np.ones(5) * (250.0 + i),
            )
        data = load_timeseries(tmp_path)
        assert len(data["days"]) == 15
        assert len(data["T_atm"]) == 15


class TestPlotAmip:
    def test_generates_png(self, fake_run_dir):
        """plot_amip produces a diagnostics.png file."""
        out = plot_amip(fake_run_dir, show=False)
        assert out.exists()
        assert out.name == "diagnostics.png"

"""Contract tests for the ppermute latency/bandwidth microbenchmark.

Runs on CPU virtual devices — the FIT logic and the refusals are what these
gate, not the hardware constants.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

from bench_ppermute_microbench import fit_latency_bandwidth  # noqa: E402


def test_fit_recovers_known_latency_and_bandwidth():
    """t = 12 us + bytes / 25 GB/s must round-trip through the fit."""
    lat_us, bw_gbs = 12.0, 25.0
    sizes = np.array([1 << k for k in range(18, 24)], dtype=float)
    times = lat_us + sizes / (bw_gbs * 1e9) * 1e6      # bytes -> us
    fit_lat, fit_bw = fit_latency_bandwidth(sizes, times)
    assert fit_lat == pytest.approx(lat_us, rel=1e-6, abs=1e-6)
    assert fit_bw == pytest.approx(bw_gbs, rel=1e-6)


def test_fit_is_not_fooled_by_a_pure_latency_curve():
    """A flat curve (all latency) must yield an enormous, not negative, BW."""
    sizes = np.array([1 << k for k in range(18, 24)], dtype=float)
    times = np.full_like(sizes, 30.0)
    _, fit_bw = fit_latency_bandwidth(sizes, times)
    assert fit_bw > 1e3 or np.isnan(fit_bw), fit_bw


def test_single_device_is_refused():
    """A latency/bandwidth fit from one device would be meaningless."""
    import subprocess
    import sys
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "bench" / "bench_ppermute_microbench.py")
    out = subprocess.run(
        [sys.executable, str(script), "--n-devices", "1"],
        capture_output=True, text=True, timeout=300,
        env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin",
             "PYTHONPATH": ":".join(sys.path)},
    )
    assert out.returncode != 0
    assert "needs >=2 devices" in (out.stderr + out.stdout)

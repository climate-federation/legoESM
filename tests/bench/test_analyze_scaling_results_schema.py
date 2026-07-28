"""Contract tests for the scaling-results analyzer's handling of the
2026-07-24 ``scaling_diagnostics`` schema fix.

The fixed ``profile_halo_exchange`` reports ``bandwidth_gb_s=None`` for a
world_size==1 profile (a local device-memory pad, no inter-rank transfer),
and ``estimate_overlap_potential`` reports ``measurement_valid=False`` with a
``None`` speedup when the standalone-halo probe is not a component of the step
(it once timed 23 s of eager dispatch inside a 46 ms step). These tests pin
that ``analyze_scaling_results.py`` consumes those ``None``/flag values without
crashing and without inventing a verdict.
"""
import importlib.util
from pathlib import Path

_MODULE_PATH = (Path(__file__).resolve().parents[2]
                / "scripts" / "bench" / "analyze_scaling_results.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_scaling_results", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = _load()


def _halo_data(bandwidths, local_pad_only):
    """Build a run dict mimicking a profile_halo_exchange output block."""
    profile = {}
    for label, bw in bandwidths.items():
        profile[label] = {
            "mean_us": 12.0,
            "p95_us": 15.0,
            "bandwidth_gb_s": bw,
            "local_pad_only": local_pad_only,
            "bytes_per_exchange": 4096,
            "std_us": 1.0,
        }
    return {"halo_profile": profile}


def test_local_pad_profile_reports_none_peak_and_does_not_crash():
    """world_size==1: every bandwidth is None -> peak is None, no TypeError."""
    data = _halo_data({"scalar_3d": None, "scalar_4d": None,
                       "vector_4d": None}, local_pad_only=True)
    out = analyze.analyze_halo_exchange(data)
    assert out["peak_bandwidth_gb_s"] is None
    # The assessment must SAY there is no inter-rank bandwidth, not claim a
    # "very low bandwidth" verdict fabricated from a coerced 0.
    assert "no inter-rank bandwidth" in out["assessment"].lower()
    assert out["exchanges"]["scalar_3d"]["bandwidth_gb_s"] is None


def test_peak_skips_none_but_uses_real_bandwidths():
    """Mixed None + real values: peak is the max of the real ones."""
    data = _halo_data({"scalar_3d": None, "scalar_4d": 12.5,
                       "vector_4d": 8.0}, local_pad_only=False)
    out = analyze.analyze_halo_exchange(data)
    assert out["peak_bandwidth_gb_s"] == 12.5


def test_bottleneck_ranking_tolerates_none_peak_bandwidth():
    """rank_bottlenecks must not crash when peak bandwidth is None."""
    analyses = {
        "phase_breakdown": {"comm_total_pct": 40.0},
        "halo_exchange": {"peak_bandwidth_gb_s": None},
    }
    # Would previously raise: None < 5.0. Now the halo-bandwidth bottleneck
    # is simply not inferred from an unmeasured link.
    bottlenecks = analyze.rank_bottlenecks(analyses)
    assert not any(b["category"] == "halo_bandwidth" for b in bottlenecks)


def test_invalid_overlap_measurement_is_refused_not_reported():
    """measurement_valid=False -> no numeric speedup, verdict refuses."""
    data = {"diag_rank0": {"extra": {"overlap_estimate": {
        "mean_step_ms": 46.0,
        "mean_halo_ms": 23000.0,
        "halo_fraction_pct": 50000.0,
        "measurement_valid": False,
        "theoretical_speedup_pct": None,
        "invalid_reason": "standalone halo exceeds the full step",
    }}}}
    out = analyze.analyze_overlap(data)
    assert out["measurement_valid"] is False
    assert out["theoretical_speedup_pct"] is None
    # The honest reason is surfaced, and NO "overlap potential" verdict is
    # manufactured from the bogus 50000% fraction.
    assert "standalone halo exceeds the full step" in out["assessment"]
    assert "overlap potential" not in out["assessment"].lower()


def test_valid_overlap_measurement_still_reports_verdict():
    """A valid measurement still produces the normal fraction-based verdict."""
    data = {"diag_rank0": {"extra": {"overlap_estimate": {
        "mean_step_ms": 10.0,
        "mean_halo_ms": 4.0,
        "halo_fraction_pct": 40.0,
        "measurement_valid": True,
        "theoretical_speedup_pct": 40.0,
    }}}}
    out = analyze.analyze_overlap(data)
    assert out["measurement_valid"] is True
    assert out["theoretical_speedup_pct"] == 40.0
    assert "overlap potential" in out["assessment"].lower()

"""Contract tests for `profile_halo_exchange` (scaling diagnosis phase 4).

The Levante campaign (2026-07-24) found the profiler timing the EAGER pad
path — ~20 s per "exchange" on a GPU node, pure op-by-op dispatch, and it
reported a "bandwidth" for a single-process run where nothing crosses a rank
boundary. These tests pin the fixed contract so the instrument cannot
silently go back to measuring dispatch overhead.
"""
import pytest

jax = pytest.importorskip("jax")

from legoesm.parallel.scaling_diagnostics import profile_halo_exchange

LABELS = ("scalar_3d", "scalar_4d", "vector_4d")


@pytest.fixture(scope="module")
def profile():
    return profile_halo_exchange(16, 4, halo=1, n_warmup=2, n_iters=5,
                                 rank=0, world_size=1)


def test_all_labels_measured_without_error(profile):
    for label in LABELS:
        assert label in profile, f"missing {label}"
        assert "error" not in profile[label], profile[label].get("error")


def test_jitted_and_fast(profile):
    """A compiled pad is microseconds; the eager path was ~20 s (2e7 us)."""
    for label in LABELS:
        entry = profile[label]
        assert entry["jitted"] is True
        assert entry["mean_us"] < 1e5, (
            f"{label}: {entry['mean_us']} us — that is dispatch/compile "
            f"overhead, not a padded exchange (the pre-fix defect)")


def test_single_process_refuses_a_bandwidth_number(profile):
    """world_size==1 is a local device-memory pad: BW is a category error."""
    for label in LABELS:
        entry = profile[label]
        assert entry["local_pad_only"] is True
        assert entry["bandwidth_gb_s"] is None
        assert "no inter-rank" in entry["bandwidth_reason"]


def test_dtype_bytes_match_the_x64_setting(profile):
    """Byte accounting must follow the actual dtype, not assume float64."""
    expected = 8 if jax.config.jax_enable_x64 else 4
    for label in LABELS:
        assert profile[label]["dtype_bytes"] == expected


def test_overlap_estimator_refuses_impossible_fractions():
    """A standalone halo cannot exceed the step it is part of.

    Pre-fix the estimator timed an eager pad and reported a 49451 %
    "theoretical speedup" (Levante job 26454084). Feed it a deliberately
    slow halo probe and assert the verdict is refused, not published.
    """
    import time as _time

    from legoesm.parallel.scaling_diagnostics import estimate_overlap_potential

    def fast_step(s, dt):
        return s

    def slow_halo(s):
        _time.sleep(0.002)      # 2 ms >> the ~0 ms no-op step
        return s

    out = estimate_overlap_potential(fast_step, slow_halo, {"x": 1.0}, 1.0,
                                     n_iters=3)
    assert out["measurement_valid"] is False
    assert out["theoretical_speedup_pct"] is None
    assert out["overlap_potential_ms"] is None
    assert "exceeds the full step" in out["invalid_reason"]


if __name__ == "__main__":  # pragma: no cover - manual run
    p = profile_halo_exchange(16, 4, halo=1, n_warmup=2, n_iters=5,
                              rank=0, world_size=1)
    for lab in LABELS:
        print(lab, p[lab]["mean_us"], "us", p[lab]["bandwidth_gb_s"])

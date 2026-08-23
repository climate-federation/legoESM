"""Direct unit tests for the eta wave-field twin instrument
(``scripts/validate/ocean_fidelity/dino_1226/eta_wave_twin.py``).

The instrument decides what we believe about the wave comparison, so its
masking, its locus partition and its spectral estimator are tested here on
synthetic data with KNOWN answers -- including the synthetic-violation checks
that prove the mask control is not vacuous.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def ewt():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.dino_1226.eta_wave_twin as m
        return m
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def _basin(ny=9, nx=7):
    """A tiny closed basin: land ring, wet interior."""
    wet = np.zeros((ny, nx), dtype=bool)
    wet[1:-1, 1:-1] = True
    lat = np.linspace(-4.0, 4.0, ny)[:, None] * np.ones((1, nx))
    return wet, lat


def test_masked_stats_ignores_dry_cells(ewt):
    wet, _ = _basin()
    diff = np.zeros(wet.shape)
    diff[2, 2] = 0.5           # wet
    diff[0, 0] = 1000.0        # dry
    st = ewt.masked_stats(diff, wet)
    assert st["max_abs_m"] == pytest.approx(0.5)
    assert (st["argmax_j"], st["argmax_i"]) == (2, 2)


def test_plant_dry_violation_is_inert_on_masked_stats(ewt):
    wet, _ = _basin()
    rng = np.random.default_rng(0)
    eta = rng.normal(size=wet.shape)
    poisoned = ewt.plant_dry_violation(eta, wet)
    assert ewt.masked_stats(eta, wet) == ewt.masked_stats(poisoned, wet)
    # ...and NOT inert on the unmasked one -- otherwise the control proves
    # nothing at all.
    assert np.max(np.abs(poisoned)) > 1e5


def test_plant_dry_violation_refuses_an_all_wet_frame(ewt):
    """A control that cannot fire must abort, not silently pass."""
    wet = np.ones((4, 4), dtype=bool)
    with pytest.raises(SystemExit, match="cannot fire"):
        ewt.plant_dry_violation(np.zeros((4, 4)), wet)


def test_masked_stats_would_move_without_masking(ewt):
    """Synthetic violation: remove the mask and the statistic MUST change.

    This is the non-vacuity proof for every masked number the probe reports.
    """
    wet, _ = _basin()
    diff = np.zeros(wet.shape)
    diff[2, 2] = 0.5
    diff[0, 0] = 1000.0
    masked = ewt.masked_stats(diff, wet)["max_abs_m"]
    unmasked = ewt.masked_stats(diff, np.ones_like(wet))["max_abs_m"]
    assert masked == pytest.approx(0.5)
    assert unmasked == pytest.approx(1000.0)


def test_locus_partition_is_a_disjoint_cover_of_the_wet_domain(ewt):
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    stack = np.stack([r["wall"], r["equator"], r["interior"]])
    assert np.array_equal(stack.any(axis=0), wet)          # covers
    assert stack.sum(axis=0).max() <= 1                     # disjoint
    # every wet cell touching land is in the wall band
    assert r["wall"][1, 1] and r["wall"][1, 5]
    # the equator band is the |lat| <= 2 strip away from the walls
    assert r["equator"].any()
    assert np.all(np.abs(lat[r["equator"]]) <= 2.0)


def test_locus_shares_sum_to_one_and_localise(ewt):
    wet, lat = _basin(ny=21, nx=11)
    r = ewt.locus_partition(wet, lat)
    diff = np.zeros(wet.shape)
    diff[r["wall"]] = 1.0
    sh = ewt.locus_shares(diff, r)
    assert sh["wall"] == pytest.approx(1.0)
    assert sum(sh.values()) == pytest.approx(1.0)


def test_locus_shares_of_an_exactly_zero_field_is_zero_not_nan(ewt):
    wet, lat = _basin()
    r = ewt.locus_partition(wet, lat)
    sh = ewt.locus_shares(np.zeros(wet.shape), r)
    assert all(v == 0.0 for v in sh.values())


def test_spectra_recovers_a_known_sinusoid(ewt):
    """Known-answer control for the estimator, before it is used on model data."""
    dt = 2700.0
    n = 512
    period_h = 8.0
    t = np.arange(n) * dt
    amp = 0.37
    x = amp * np.sin(2 * np.pi * t / (period_h * 3600.0)) + 5.0
    f, a = ewt.spectra(x, dt)
    peak = np.argmax(a[1:]) + 1
    assert 1.0 / (f[peak] * 3600.0) == pytest.approx(period_h, rel=0.05)
    assert a[peak] == pytest.approx(amp, rel=0.05)
    # The constant offset must not appear at zero frequency.  The bound is
    # 1e-3 of the amplitude, not machine zero: the series is de-meaned
    # exactly, so the DC bin holds only Hann leakage from the sinusoid not
    # spanning a whole number of periods (measured ~5e-7 relative here).
    assert a[0] < 1e-3 * amp


def test_load_side_rejects_nan(ewt, tmp_path):
    p = tmp_path / "bad.npz"
    eta = np.zeros((3, 4, 5))
    eta[1, 1, 1] = np.nan
    np.savez(p, eta=eta, t_seconds=np.arange(3.0))
    with pytest.raises(SystemExit, match="NaN"):
        ewt.load_side(str(p), "x")


def test_load_side_rejects_a_mismatched_time_axis(ewt, tmp_path):
    p = tmp_path / "bad2.npz"
    np.savez(p, eta=np.zeros((3, 4, 5)), t_seconds=np.arange(2.0))
    with pytest.raises(SystemExit, match="t_seconds"):
        ewt.load_side(str(p), "x")


def test_provenance_stamps_sha_clock_and_dtype(ewt):
    prov = ewt.provenance({"source": "test"})
    for key in ("git_sha", "git_dirty", "utc", "dtype", "numpy", "argv"):
        assert key in prov
    assert prov["dtype"] == "float64"
    assert prov["source"] == "test"

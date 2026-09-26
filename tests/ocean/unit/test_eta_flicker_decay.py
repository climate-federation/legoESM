"""Unit tests for the 2-step-flicker decay probe.

Each test is built so that it FAILS if the behaviour it names is removed --
a probe whose controls cannot fail is worse than no probe, because it lends
its authority to whatever it printed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE_DIR = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "scripts", "validate", "ocean_fidelity",
    "dino_1226"))


def _load():
    """Import the probe without importing its heavyweight sibling's CLI."""
    if _PROBE_DIR not in sys.path:
        sys.path.insert(0, _PROBE_DIR)
    spec = importlib.util.spec_from_file_location(
        "eta_flicker_decay", os.path.join(_PROBE_DIR, "eta_flicker_decay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("netCDF4")
efd = _load()


def _grid(ny=8, nx=6):
    wet = np.ones((ny, nx), dtype=bool)
    wet[0, :] = False                       # a land row so "wall" is non-empty
    area = np.where(wet, 1.0, 0.0)
    return wet, area


def test_region_amplitude_recovers_a_known_nyquist_amplitude():
    """x[n] = A*(-1)**n must come back as A, not 2A and not A/2."""
    wet, area = _grid()
    amp_true = 0.37
    n = 12
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    field = amp_true * sign * np.ones((n,) + wet.shape)
    alt = efd.ewt.two_dt_component(field)
    got = efd.region_amplitude(alt, wet, area)
    assert np.allclose(got, amp_true, rtol=1e-12), got


def test_region_amplitude_annihilates_a_smooth_series():
    """A linear-in-time field has no 2-step content at all."""
    wet, area = _grid()
    t = np.arange(20.0)[:, None, None]
    field = 3.0 + 0.5 * t * np.ones((1,) + wet.shape)
    alt = efd.ewt.two_dt_component(field)
    assert efd.region_amplitude(alt, wet, area).max() < 1e-12


def test_region_amplitude_is_area_weighted_not_a_cell_count():
    """Two cells with a 100:1 area ratio must not count equally.

    Fails if the weighting is dropped: the unweighted answer differs.
    """
    wet = np.ones((1, 2), dtype=bool)
    area = np.array([[100.0, 1.0]])
    n = 6
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    field = sign * np.array([[1.0, 10.0]])
    alt = efd.ewt.two_dt_component(field)
    got = efd.region_amplitude(alt, wet, area)[0]
    weighted = np.sqrt((100.0 * 1.0 + 1.0 * 100.0) / 101.0)
    unweighted = np.sqrt((1.0 + 100.0) / 2.0)
    assert np.isclose(got, weighted, rtol=1e-12)
    assert not np.isclose(got, unweighted, rtol=1e-3)


def test_region_amplitude_refuses_an_empty_region():
    wet, area = _grid()
    empty = np.zeros_like(wet)
    n = 6
    alt = efd.ewt.two_dt_component(np.zeros((n,) + wet.shape))
    with pytest.raises(SystemExit):
        efd.region_amplitude(alt, empty, np.zeros_like(area))


def test_leakage_floor_is_small_against_a_real_nyquist_mode():
    """The [1,2,1]/4 smoother annihilates the mode, so the floor must not
    report the mode back as leakage."""
    wet, area = _grid()
    n = 24
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    field = 1.0 + sign * np.ones((1,) + wet.shape)
    floor = efd.region_leakage_floor(field, wet, area, slice(0, 8))
    assert floor < 1e-10, floor


def test_leakage_floor_tracks_its_window():
    """The floor must be measurable over a late window, not only the first 8.

    The review blocker: a floor taken over the launch transient and then
    subtracted from a late-time amplitude is a mismatched-window comparison,
    and it reversed the sign of a conclusion.  Here a field whose fast
    content dies away must give a LARGER early floor than late floor; a
    window-blind implementation returns the same number twice.
    """
    wet, area = _grid()
    n = 40
    t = np.arange(n)[:, None, None]
    fast = np.cos(2.0 * np.pi * t / 5.0) * np.exp(-t / 6.0)
    field = fast * np.ones((1,) + wet.shape)
    early = efd.region_leakage_floor(field, wet, area, slice(0, 8))
    late = efd.region_leakage_floor(field, wet, area, slice(20, None))
    assert early > 5.0 * late, (early, late)


def test_leakage_floor_is_region_restricted():
    """A mode planted in one region must not raise the OTHER region's floor."""
    wet, area = _grid()
    n = 40
    t = np.arange(n)[:, None, None]
    band_a = np.zeros_like(wet)
    band_a[1, :] = True
    band_b = np.zeros_like(wet)
    band_b[2:, :] = True
    field = np.zeros((n,) + wet.shape)
    field += (np.cos(2.0 * np.pi * t / 5.0) * band_a[None, :, :])
    fa = efd.region_leakage_floor(field, band_a, area, slice(0, 8))
    fb = efd.region_leakage_floor(field, band_b, area, slice(0, 8))
    assert fa > 1e-3 and fb < 1e-12, (fa, fb)


def test_leakage_floor_demeans_in_time():
    """A large constant offset must not create a floor out of the padding.

    This is the earned trap: zero-padded convolution on a field with a big
    mean manufactured a floor 700x its own signal.  Fails if the de-meaning
    in ``region_leakage_floor`` is removed.
    """
    wet, area = _grid()
    n = 24
    rng = np.random.default_rng(0)
    slow = np.cumsum(rng.normal(0.0, 1e-6, size=(n, 1, 1)), axis=0) * np.ones(
        (1,) + wet.shape)
    quiet = efd.region_leakage_floor(slow, wet, area, slice(0, 8))
    offset = efd.region_leakage_floor(slow + 0.83, wet, area, slice(0, 8))
    assert np.isclose(quiet, offset, rtol=1e-9), (quiet, offset)


def test_fit_decay_recovers_a_known_exponential():
    n = np.arange(120.0)
    amp = 0.2 + 1.5 * np.exp(-n / 9.0)
    fit = efd.fit_decay(amp)
    assert fit["converged"]
    assert np.isclose(fit["tau_steps"], 9.0, rtol=0.15), fit
    assert np.isclose(fit["steady_C_m"], 0.2, rtol=0.15), fit
    assert fit["r2"] > 0.99


def test_fit_decay_flags_an_unidentifiable_tau():
    """A series that does not decay inside the window must not quote a tau."""
    n = np.arange(40.0)
    amp = 0.5 + 0.5 * np.exp(-n / 500.0)     # essentially flat over 40 samples
    fit = efd.fit_decay(amp)
    assert fit["converged"]
    assert not fit["fit_usable"], fit


def test_fit_usable_is_false_for_a_flat_noisy_series():
    """The degenerate case the first gate returned True for.

    A series with NO transient at all must not be reported as a usable decay
    fit -- the optimizer simply never moves the amplitude off its starting
    guess, and the old gate (``A > 0``) accepted 1e-10 as a transient.
    """
    rng = np.random.default_rng(3)
    amp = 1.0 + rng.normal(0.0, 0.02, size=160)
    fit = efd.fit_decay(amp)
    assert fit["converged"]
    assert not fit["fit_usable"], fit


def test_fit_usable_is_true_for_a_clean_exponential():
    n = np.arange(160.0)
    rng = np.random.default_rng(4)
    amp = 0.2 + 1.5 * np.exp(-n / 9.0) + rng.normal(0.0, 0.002, size=160)
    fit = efd.fit_decay(amp)
    assert fit["fit_usable"], fit


def test_ratio_returns_none_rather_than_nan_or_inf():
    """``json.dump`` defaults to allow_nan=True, so a NaN ratio would be
    written silently and later read as a number."""
    assert efd._ratio(2.0, 1.0) == 2.0
    assert efd._ratio(1.0, 0.0) is None
    assert efd._ratio(1.0, -1.0) is None
    assert efd._ratio(np.nan, 1.0) is None
    assert efd._ratio(np.inf, 1.0) is None


def test_shared_constant_land_poison_is_vacuous_for_a_nyquist_statistic():
    """Why this probe cannot reuse ``plant_dry_violation`` unchanged.

    Both models write a CONSTANT (zero, verified: dry-cell eta is exactly 0.0
    at every step in both artifacts) on dry cells, and the 2-step operator
    annihilates a constant exactly.  So replacing that constant with a
    different constant is invisible to an UNMASKED Nyquist statistic -- the
    shared control would pass on a probe that averages land.  Reproduced here
    on the real data's pattern: dry cells constant in time.

    This test pins the fact, so nobody "simplifies" the alternating poison
    back to the shared helper.
    """
    wet, _ = _grid()
    n = 10
    rng = np.random.default_rng(1)
    eta = rng.normal(size=(n,) + wet.shape)
    eta[..., ~wet] = 0.0                     # what both models actually write
    poisoned = efd.ewt.plant_dry_violation(eta, wet)
    unmasked = np.abs(efd.ewt.two_dt_component(eta)).mean()
    unmasked_p = np.abs(efd.ewt.two_dt_component(poisoned)).mean()
    assert unmasked_p == unmasked, (
        "a constant dry poison DOES move the unmasked Nyquist statistic on "
        "constant-in-time dry cells; if this ever becomes true the shared "
        "helper could serve as the control after all")


def test_land_control_is_non_vacuous_with_the_alternating_poison():
    """The alternating poison IS visible to an unmasked Nyquist statistic.

    Fails if the shared helper's alternating branch loses its sign flip, which is
    exactly the way this control would silently become unable to fire.
    """
    wet, area = _grid()
    n = 10
    rng = np.random.default_rng(1)
    eta = rng.normal(size=(n,) + wet.shape)
    poisoned = efd.ewt.plant_dry_violation(eta, wet, alternating=True)
    masked = efd.region_amplitude(efd.ewt.two_dt_component(eta), wet, area)
    masked_p = efd.region_amplitude(
        efd.ewt.two_dt_component(poisoned), wet, area)
    assert np.array_equal(masked, masked_p)
    unmasked = np.abs(efd.ewt.two_dt_component(eta)).mean()
    unmasked_p = np.abs(efd.ewt.two_dt_component(poisoned)).mean()
    assert unmasked_p > 1.0e4 * unmasked, (unmasked, unmasked_p)


def test_alternating_poison_refuses_an_all_wet_grid():
    wet = np.ones((4, 4), dtype=bool)
    with pytest.raises(SystemExit):
        efd.ewt.plant_dry_violation(np.zeros((6, 4, 4)), wet, alternating=True)


def _regions(wet):
    regions = {"all": wet,
               "wall": np.zeros_like(wet),
               "interior": np.zeros_like(wet)}
    regions["wall"][1, :] = True
    regions["interior"][2:, :] = True
    return regions


def test_self_check_planted_mode_is_attributed_to_the_wall_not_the_interior():
    wet, area = _grid()
    regions = _regions(wet)
    n = 16
    rng = np.random.default_rng(2)
    eta = rng.normal(0.0, 1e-9, size=(n,) + wet.shape)
    out = efd._self_check(eta, wet, regions, area)
    assert out["land_poison_identical"]
    assert 0.95 <= out["plant_large"]["recovery_ratio"] <= 1.05, out
    for key in ("plant_large", "plant_at_measurement_scale"):
        assert out[key]["interior_untouched_partition_disjoint"], (key, out[key])


def test_measurement_scale_plant_is_reported_not_gated():
    """A sign-coherent pre-existing wall mode reads ~1.73, and that is a
    FINDING not an instrument failure -- so it must be reported, not aborted.

    Fails if anyone re-gates the measurement-scale plant to [0.8, 1.2].
    """
    wet, area = _grid()
    regions = _regions(wet)
    n = 16
    sign = ((-1.0) ** np.arange(n))[:, None, None]
    # a wall field that is ALREADY a coherent Nyquist mode of amplitude 1e-6
    eta = 1e-6 * sign * regions["wall"][None, :, :] * np.ones((n,) + wet.shape)
    out = efd._self_check(eta, wet, regions, area)
    r = out["plant_at_measurement_scale"]["recovery_ratio"]
    assert 1.5 <= r <= 2.0, r
    assert out["plant_at_measurement_scale"]["sign_coherent_field_suspected"]


def test_decorrelation_lag_measures_what_its_name_says():
    rng = np.random.default_rng(11)
    white = rng.normal(size=400)
    assert efd.decorrelation_lag(white) <= 2
    smooth = np.convolve(rng.normal(size=460), np.ones(30) / 30.0,
                         mode="valid")
    assert efd.decorrelation_lag(smooth) > 5


def test_bootstrap_uses_the_measured_lag_and_independent_resampling():
    rng = np.random.default_rng(12)
    a = 2.0 + rng.normal(0.0, 0.1, size=120)
    b = 1.0 + rng.normal(0.0, 0.1, size=120)
    out = efd.block_bootstrap_ratio_ci(a, b)
    assert out["block"] == out["measured_decorrelation_lag"]
    assert out["resampling"] == "independent per side"
    lo, hi = out["ci90"]
    assert lo < 2.0 < hi, out


def test_land_control_goes_red_when_the_region_mask_admits_land():
    """The blocker: the production ``area`` is already zero on land, so no
    change inside the probe could turn the land control red.  ``_self_check``
    now runs it against UNMASKED weights, leaving the region mask as the only
    defence -- so a mask that admits dry cells MUST fail it.
    """
    wet, area = _grid()
    n = 16
    rng = np.random.default_rng(5)
    eta = rng.normal(0.0, 1e-9, size=(n,) + wet.shape)
    eta[..., ~wet] = 0.0

    good = efd._self_check(eta, wet, _regions(wet), area)
    assert good["land_poison_identical"], "the honest mask must PASS"

    broken = _regions(wet)
    broken["all"] = np.ones_like(wet)          # admits the land row
    bad = efd._self_check(eta, wet, broken, area)
    assert not bad["land_poison_identical"], (
        "a region mask that admits land must fail the land control")


def test_shared_reduction_is_bit_identical_to_the_published_expression():
    """The refactor must not move a number the campaign has already quoted.

    ``weighted_rms_series`` replaced a per-sample loop that the shipped
    ``two_dt_mode``/``two_dt_leakage_floor`` JSONs were produced by.  A
    vectorised ``sum(..., axis=1)`` agrees to 12+ digits but not bit-for-bit
    (pairwise summation runs in a different order), so the reduction keeps the
    loop.  This test goes red if anyone vectorises it again.
    """
    rng = np.random.default_rng(7)
    ny, nx, n = 31, 17, 24
    mask = rng.random((ny, nx)) > 0.3
    area = rng.random((ny, nx)) * 1e10
    alt = rng.normal(size=(n, ny, nx))
    w = area[mask]
    reference = np.array([
        float(np.sqrt(np.sum(w * alt[i][mask] ** 2) / np.sum(w)))
        for i in range(n)])
    got = efd.ewt.weighted_rms_series(alt, mask, area)
    assert np.array_equal(got, reference), np.abs(got - reference).max()

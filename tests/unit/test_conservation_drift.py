"""Unit tests for ``legoesm.diagnostics.conservation_drift``.

Pins the iter-78/80/83/87 baseline-zero floor convention shared by the
atmosphere, ocean, and HS+RRTMGP cross-grid drivers.  See the module
docstring of ``conservation_drift.py`` for the full history.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.diagnostics.conservation_drift import (
    DEFAULT_MIN_BASELINE,
    compute_relative_drift,
    relative_drift_series,
)


class TestComputeRelativeDriftScalar:
    """Scalar drift helper used by atmosphere & HS+RRTMGP runners."""

    def test_default_floor_is_one(self):
        assert DEFAULT_MIN_BASELINE == pytest.approx(1.0)

    def test_returns_zero_for_empty_input(self):
        assert compute_relative_drift([]) == 0.0

    def test_returns_zero_for_single_element(self):
        assert compute_relative_drift([42.0]) == 0.0

    def test_normal_positive_baseline_relative_drift(self):
        # mass-like baseline (~5e+19) with 1e-3 relative drift
        baseline = 5.0e19
        drift_abs = 5.0e16
        result = compute_relative_drift([baseline, baseline + drift_abs])
        assert result == pytest.approx(drift_abs / baseline)
        # Result is the dimensionless relative drift, ≪ 1
        assert 0.0 < result < 1.0

    def test_zero_baseline_returns_absolute_drift(self):
        # Rest-state pathology: baseline = 0, machine-precision rounding
        # in the timeseries could be 1e-17.  With the old 1e-30 floor
        # this would amplify to 1e+13.  With the 1.0 floor it becomes
        # the absolute drift in natural units (1e-17).
        result = compute_relative_drift([0.0, 1.0e-17])
        assert result == pytest.approx(1.0e-17, abs=1e-25)
        # Critically: the result must NOT be order-of-magnitude inflated
        assert result < 1.0e-10

    def test_negative_baseline_uses_absolute_value(self):
        # Stratified ocean heat could be negative (relative to a high
        # reference temperature).  The denominator must be |baseline|.
        result = compute_relative_drift([-1.0e25, -1.0e25 + 1.0e22])
        assert result == pytest.approx(1.0e22 / 1.0e25)

    def test_subnormal_baseline_uses_floor(self):
        # baseline < min_baseline → fall through to absolute drift.
        # 0.5 < 1.0 floor, so denominator is 1.0 (not 0.5).
        result = compute_relative_drift([0.5, 0.7])
        assert result == pytest.approx(0.2)  # absolute drift, denom=1.0

    def test_at_baseline_floor_uses_baseline(self):
        # |baseline| > min_baseline by a hair → use baseline.
        result = compute_relative_drift([2.0, 2.5])
        assert result == pytest.approx(0.25)  # 0.5/2.0

    def test_custom_min_baseline_argument(self):
        # User supplies their own floor (e.g., for a unit-system that
        # is not bridged to SI).  baseline=0.5, floor=0.1 → use 0.5.
        result = compute_relative_drift([0.5, 0.7], min_baseline=0.1)
        assert result == pytest.approx(0.4)  # 0.2/0.5

    def test_negative_min_baseline_raises(self):
        with pytest.raises(ValueError, match="min_baseline"):
            compute_relative_drift([1.0, 2.0], min_baseline=-1.0)

    def test_zero_min_baseline_raises(self):
        with pytest.raises(ValueError, match="min_baseline"):
            compute_relative_drift([1.0, 2.0], min_baseline=0.0)

    def test_inf_min_baseline_raises(self):
        # iter-89 self-review: silently accepting ``inf`` would make
        # every drift evaluate to 0 (since denom = inf).  The helper
        # must reject this with a clear ValueError so a regression
        # passing inf to the helper would fail loudly.
        with pytest.raises(ValueError, match="min_baseline"):
            compute_relative_drift([1.0, 2.0], min_baseline=float("inf"))

    def test_neg_inf_min_baseline_raises(self):
        with pytest.raises(ValueError, match="min_baseline"):
            compute_relative_drift([1.0, 2.0], min_baseline=float("-inf"))

    def test_nan_min_baseline_raises(self):
        # iter-89 self-review: silently accepting NaN would make
        # every drift evaluate to NaN.  Fail-fast at the boundary.
        with pytest.raises(ValueError, match="min_baseline"):
            compute_relative_drift([1.0, 2.0], min_baseline=float("nan"))

    def test_accepts_numpy_array(self):
        arr = np.array([5.0e19, 5.0e19 + 5.0e16])
        result = compute_relative_drift(arr)
        assert result == pytest.approx(1.0e-3)

    def test_accepts_python_list(self):
        result = compute_relative_drift([5.0e19, 5.0e19 + 5.0e16])
        assert result == pytest.approx(1.0e-3)

    def test_decreasing_series_returns_positive_magnitude(self):
        # A sign-flipping regression in the numerator (forgetting abs)
        # would silently produce a negative drift.  The contract is
        # that drift is always >= 0.
        result = compute_relative_drift([5.0e19, 4.5e19])  # decreasing
        assert result == pytest.approx(0.1)  # 0.5e19/5.0e19 = 0.1
        assert result >= 0.0

    def test_only_first_and_last_matter(self):
        # The intermediate samples are ignored — only [0] and [-1].
        # This pins the "scalar drift" semantics distinct from
        # ``relative_drift_series``.
        result_with_middle = compute_relative_drift(
            [10.0, 1e9, 1e9, 1e9, 11.0]
        )
        result_endpoints = compute_relative_drift([10.0, 11.0])
        assert result_with_middle == pytest.approx(result_endpoints)


class TestRelativeDriftSeries:
    """Series version used by ocean run_ocean_test_matrix.py."""

    def test_empty_input_returns_empty(self):
        result = relative_drift_series([])
        assert result.shape == (0,)

    def test_single_element_returns_zero_drift(self):
        result = relative_drift_series([7.0])
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0)

    def test_first_element_always_zero(self):
        # By construction (values - values[0]) / denom: the first
        # entry is always 0.  This is what makes the cross-variant
        # plot axhline(0) meaningful.
        result = relative_drift_series([5.0e19, 5.0e19 + 1e16, 5.0e19 - 1e16])
        assert result[0] == pytest.approx(0.0)

    def test_normal_positive_baseline(self):
        # Ocean volume ~ 1e+18 m³ with 1e+15 m³ drift over 3 samples.
        baseline = 1.0e18
        series = [baseline, baseline + 5.0e14, baseline - 1.0e15]
        result = relative_drift_series(series)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(5.0e-4)
        assert result[2] == pytest.approx(-1.0e-3)

    def test_zero_baseline_returns_absolute_series(self):
        # iter-78/80 rest-state pathology: baseline = 0 ⇒ each entry
        # is the absolute deviation in natural units (NOT a 1e+13
        # spurious value).
        result = relative_drift_series([0.0, 1.0e-17, 5.0e-17])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0e-17, abs=1e-25)
        assert result[2] == pytest.approx(5.0e-17, abs=1e-25)
        # Critical: the spurious 1e+13 magnitude must NOT appear
        assert np.max(np.abs(result)) < 1.0e-10

    def test_returns_float64(self):
        result = relative_drift_series([1.0, 2.0])
        assert result.dtype == np.float64

    def test_returns_numpy_array(self):
        result = relative_drift_series([1.0, 2.0])
        assert isinstance(result, np.ndarray)

    def test_length_preserved(self):
        for n in (1, 2, 5, 100):
            series = [float(i) + 5.0e19 for i in range(n)]
            result = relative_drift_series(series)
            assert result.shape == (n,)

    def test_negative_min_baseline_raises(self):
        with pytest.raises(ValueError, match="min_baseline"):
            relative_drift_series([1.0, 2.0], min_baseline=-0.5)

    def test_zero_min_baseline_raises(self):
        with pytest.raises(ValueError, match="min_baseline"):
            relative_drift_series([1.0, 2.0], min_baseline=0.0)

    def test_inf_min_baseline_raises(self):
        # iter-89: same defensive validation as the scalar version.
        with pytest.raises(ValueError, match="min_baseline"):
            relative_drift_series([1.0, 2.0], min_baseline=float("inf"))

    def test_nan_min_baseline_raises(self):
        with pytest.raises(ValueError, match="min_baseline"):
            relative_drift_series([1.0, 2.0], min_baseline=float("nan"))

    def test_custom_min_baseline_argument(self):
        # baseline=0.5, floor=0.1 → use 0.5.  series=[0.5, 0.7] → [0, 0.4]
        result = relative_drift_series([0.5, 0.7], min_baseline=0.1)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.4)

    def test_consistency_with_scalar_at_endpoints(self):
        # The series version's last value (in absolute magnitude)
        # must agree with the scalar version's drift.
        series = [5.0e19, 5.5e19, 6.0e19, 5.5e19]
        scalar = compute_relative_drift(series)
        s_series = relative_drift_series(series)
        assert abs(s_series[-1]) == pytest.approx(scalar)

    def test_oscillating_series_first_entry_zero(self):
        # Used by the ocean test matrix's axhline(0) reference.
        result = relative_drift_series(
            [1.0e18, 1.05e18, 0.95e18, 1.10e18, 0.90e18]
        )
        assert result[0] == pytest.approx(0.0)


class TestNonFiniteValueInputs:
    """iter-90 codex MEDIUM-5: ``values`` may contain inf or NaN
    when a diagnostic blows up.  Pin the contract: the helper
    propagates non-finite values to the output (instead of silently
    masking them) so downstream consumers see a clear "broken
    diagnostic" signal rather than a plausible-looking number.
    """

    def test_inf_endpoint_returns_inf(self):
        # CFL violation produces inf at last sample.
        result = compute_relative_drift([1.0e19, float("inf")])
        assert result == float("inf")

    def test_nan_endpoint_returns_nan(self):
        result = compute_relative_drift([1.0e19, float("nan")])
        assert result != result  # NaN != NaN

    def test_nan_baseline_returns_nan(self):
        # iter-90 codex MEDIUM-3: NaN baseline propagates because
        # max(NaN, 1.0) returns NaN under Python's stdlib max.
        # Documenting this contract; not a regression vs legacy.
        result = compute_relative_drift([float("nan"), 1.0e19])
        assert result != result  # NaN

    def test_inf_in_middle_does_not_affect_scalar(self):
        # The scalar API only reads [0] and [-1] — middle inf is
        # ignored.  Pins the "endpoints only" contract.
        result = compute_relative_drift(
            [1.0e19, float("inf"), 1.01e19]
        )
        assert result == pytest.approx(1.0e-2)

    def test_inf_in_middle_does_affect_series(self):
        # The series API computes per-step drift, so an inf in the
        # middle will appear as inf in the series.
        result = relative_drift_series(
            [1.0e19, float("inf"), 1.01e19]
        )
        assert result[0] == pytest.approx(0.0)
        assert result[1] == float("inf")
        assert result[2] == pytest.approx(1.0e-2, rel=1e-9)


class TestEmptyArrayAliasing:
    """iter-90 codex LOW-6: ``relative_drift_series([])`` must
    return a fresh empty array — not the same object as
    ``np.asarray([])`` — so a caller can't accidentally mutate
    their input through the returned alias.
    """

    def test_empty_returns_fresh_array(self):
        original = np.asarray([], dtype=np.float64)
        result = relative_drift_series(original)
        # Both empty, but result is a separate allocation.
        assert result.size == 0
        assert result is not original

    def test_empty_caller_mutation_does_not_affect_input(self):
        # iter-90: even though both are empty (so neither can hold
        # values), the contract is they are distinct objects.
        # Resizing the result must not affect input.
        empty_input: list[float] = []
        result = relative_drift_series(empty_input)
        # result is a numpy array, empty_input is a Python list.
        # The contract is just: result is a fresh ndarray.
        assert isinstance(result, np.ndarray)


class TestSingleElementSeriesShape:
    """iter-90 codex LOW-7: the series API preserves the input
    length even for length-1 inputs (returns ``[0.0]``), unlike
    the scalar API which returns ``0.0``.
    """

    def test_single_element_returns_length_one(self):
        result = relative_drift_series([42.0])
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.0)

    def test_scalar_api_for_single_element(self):
        # Scalar always returns 0.0 for short series.
        assert compute_relative_drift([42.0]) == 0.0

    def test_apis_diverge_only_on_short_input(self):
        # For length >= 2, the series last element (in absolute
        # value) matches the scalar result.
        for series in (
            [5.0e19, 5.5e19],
            [-1.0e19, -1.05e19],
            [1.34e18, 1.34e18, 1.34e18 + 1e15],
        ):
            scalar = compute_relative_drift(series)
            arr = relative_drift_series(series)
            assert arr.shape == (len(series),)
            assert arr[0] == pytest.approx(0.0)
            assert abs(arr[-1]) == pytest.approx(scalar, rel=1e-9)


class TestRegressionPathologyDirect:
    """Direct regression tests for the iter-78/80/83/87 pathology.

    These tests intentionally use the historical pathological inputs
    (baseline = 0 with machine-precision rounding) so that any future
    regression that re-introduces the 1e-30 floor would flip these
    tests with an instantly-recognizable failure message.
    """

    def test_iter78_rest_state_volume_pathology_is_fixed(self):
        # Original iter-78 cube ocean rest_state run reported
        # -2.83e+13 m volume drift (display-only bug, not physics).
        # Reproduce the pathological input and verify the helper
        # returns a sane absolute drift.
        machine_eps_drift = 1.0e-17
        result = compute_relative_drift([0.0, machine_eps_drift])
        # Old behaviour: ~1e+13 spurious.  Fixed behaviour: ~1e-17.
        assert result < 1.0  # NOT inflated
        assert result == pytest.approx(machine_eps_drift, abs=1e-25)

    def test_iter83_atmosphere_rest_state_pathology_is_fixed(self):
        # The iter-83 audit-target: _compute_drift on a quiescent
        # diagnostic with baseline 0.
        result = compute_relative_drift([0.0, 1.0e-15])
        assert result < 1.0
        assert result == pytest.approx(1.0e-15, abs=1e-25)

    def test_iter87_hs_rrtmgp_mass_pathology_is_fixed(self):
        # HS+RRTMGP mass_drift: in production mass[0] ~ 5e+19, but
        # the helper must remain safe for any baseline including 0.
        result = compute_relative_drift([0.0, 2.0e-17])
        assert result < 1.0
        assert result == pytest.approx(2.0e-17, abs=1e-25)

    def test_production_atmosphere_mass_baseline_unchanged(self):
        # Sanity: when the baseline is huge (~5e+19 Pa·m²), the
        # iter-88 floor must not affect the answer.
        baseline = 5.0e19
        # 1 ppm relative drift
        result = compute_relative_drift([baseline, baseline * (1 + 1e-6)])
        assert result == pytest.approx(1.0e-6, rel=1e-9)

    def test_production_ocean_volume_baseline_unchanged(self):
        # Ocean volume ~ 1.34e+18 m³.  A 1e-9 relative drift must
        # come back as 1e-9, not affected by the 1.0 floor.
        baseline = 1.34e18
        result = compute_relative_drift(
            [baseline, baseline * (1 + 1e-9)]
        )
        assert result == pytest.approx(1.0e-9, rel=1e-6)

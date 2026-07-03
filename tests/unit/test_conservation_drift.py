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

    def test_inf_endpoint_returns_nan(self):
        # iter-152 (codex iter-151 review HIGH-1): the all-finite
        # check now returns NaN for ANY non-finite sample
        # (including endpoints).  Pre-iter-152 this returned
        # ``inf``; the contract was tightened so the downstream
        # gate's ``not isfinite(drift)`` branch handles both
        # NaN and Inf uniformly via a single "non-finite" path.
        result = compute_relative_drift([1.0e19, float("inf")])
        assert result != result  # NaN (was inf pre-iter-152)

    def test_nan_endpoint_returns_nan(self):
        result = compute_relative_drift([1.0e19, float("nan")])
        assert result != result  # NaN != NaN

    def test_nan_baseline_returns_nan(self):
        # iter-90 codex MEDIUM-3 + iter-152 update: NaN baseline
        # surfaces as NaN.  Pre-iter-152 propagated via
        # ``max(NaN, 1.0)`` → NaN; post-iter-152 the all-finite
        # check fails first and returns NaN explicitly.  Same
        # observable result.
        result = compute_relative_drift([float("nan"), 1.0e19])
        assert result != result  # NaN

    def test_inf_in_middle_does_not_affect_scalar(self):
        # iter-152 (codex iter-151 review HIGH-1): renamed test
        # contract — pre-iter-152 the scalar API only read
        # ``arr[0]`` and ``arr[-1]``, so a middle Inf was IGNORED,
        # letting a transient blowup silently PASS the drift gate
        # (the codex finding).  Post-iter-152 the helper checks
        # ``np.all(np.isfinite(arr))`` and returns NaN, surfacing
        # the transient.  Old test name kept for git-blame-ability.
        result = compute_relative_drift(
            [1.0e19, float("inf"), 1.01e19]
        )
        assert result != result  # NaN (was 1.0e-2 pre-iter-152)

    def test_inf_in_middle_does_affect_series(self):
        # iter-153 (codex iter-152 review MEDIUM-2): the series
        # API now propagates non-finite samples to the WHOLE
        # output array (mirroring the iter-152 scalar fix), not
        # just the affected element.  Pre-iter-153 the series
        # was ``[0, inf, 1e-2]`` so a caller using
        # ``abs(series[-1])`` could still silently PASS the
        # transient case.  Post-iter-153 the entire array is NaN
        # if ANY sample is non-finite, surfacing the broken
        # diagnostic uniformly.
        import numpy as np
        result = relative_drift_series(
            [1.0e19, float("inf"), 1.01e19]
        )
        assert result.shape == (3,)
        assert np.all(np.isnan(result)), (
            "iter-153: any non-finite sample must produce all-NaN "
            "output array.")


class TestIter176SingleSampleNonFinite:
    """iter-176 (codex iter-176 review MEDIUM-1): single-sample
    non-finite series must return NaN, not 0.0.

    Pre-iter-176 contract:
        compute_relative_drift([nan]) -> 0.0   # WRONG silent pass
        compute_relative_drift([inf]) -> 0.0   # WRONG silent pass

    The bug was ordering — the ``len(values) < 2: return 0.0``
    early-return ran BEFORE the all-finite check, so any non-
    finite single-sample input got the "no-drift-to-compute"
    path instead of being surfaced as NaN.

    Post-iter-176 contract:
        compute_relative_drift([nan]) -> NaN
        compute_relative_drift([inf]) -> NaN
        compute_relative_drift([])    -> 0.0  (empty stays as is)
        compute_relative_drift([1.0]) -> 0.0  (finite single-sample
                                              still returns 0.0
                                              — no series to compare)

    Impact: a truncated/degraded run with only one non-finite
    diagnostic sample (e.g. blow-up at step 0 with no further
    samples collected) was reported as ``drift=0.0`` and PASSED
    downstream gating.  This was a real silent-pass.
    """

    def test_single_nan_returns_nan(self):
        result = compute_relative_drift([float("nan")])
        assert result != result  # NaN != NaN

    def test_single_inf_returns_nan(self):
        result = compute_relative_drift([float("inf")])
        assert result != result  # NaN

    def test_single_neg_inf_returns_nan(self):
        result = compute_relative_drift([float("-inf")])
        assert result != result  # NaN

    def test_single_finite_returns_zero(self):
        # iter-176 must NOT regress the finite single-sample case
        # — that one still returns 0.0 (no drift to compute when
        # there's only one sample).
        result = compute_relative_drift([1.0e19])
        assert result == 0.0

    def test_empty_still_returns_zero(self):
        # iter-176 preserves the empty-series contract: 0.0
        # (nothing to be non-finite about).
        result = compute_relative_drift([])
        assert result == 0.0


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


class TestIter92AuditFollowupDelegation:
    """iter-92 audit followup to codex iter-90 review.

    iter-90 codex caught HIGH-1 (script ocean ``_compute_drift``).
    iter-91 audited and caught two more in the package.  iter-92
    audited even more aggressively and caught four MORE missed
    callsites:

    * ``src/legoesm/diagnostics/precision_drift.py:406`` — production
      energy-drift-rate normalization with ``max(abs(energy_prev), 1e-30)``.
    * ``scripts/matrix/run_sea_ice_test_matrix.py`` — 6 ``vol_drift`` callsites
      with ``max(vol_X, 1e-20)`` floors (deferred in iter-91).
    * ``tests/validation/bench_spectral_pe.py:323`` — JW06 KE timeseries
      with ``max(abs(KE_ts[0]), 1e-30)`` floor.
    * ``tests/validation/bench_spectral_pe.py:317`` — JW06 mass timeseries
      with NO floor at all (would NaN for mass=0).

    These tests pin the iter-92 fixes by directly probing the
    target functions / source for the right delegation pattern.
    """

    def test_precision_drift_uses_compute_relative_drift(self):
        """``precision_health_report``'s energy-drift block must
        delegate to ``compute_relative_drift``.
        """
        import inspect
        import re
        from legoesm.diagnostics.precision_drift import precision_health_report
        src = inspect.getsource(precision_health_report)
        # Strip docstrings + line comments first.
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "compute_relative_drift" in code_only, (
            "iter-92: ``precision_health_report``'s energy-drift "
            "computation must delegate to "
            "``compute_relative_drift`` from "
            "``legoesm.diagnostics.conservation_drift``."
        )
        assert "1e-30" not in code_only, (
            "iter-92: legacy 1e-30 floor must not appear in the "
            "active code of ``precision_health_report``."
        )

    def test_sea_ice_matrix_uses_compute_relative_drift(self):
        """``run_sea_ice_test_matrix.py`` no longer inlines
        ``max(vol_X, 1e-20)`` — all 6 vol_drift callsites delegate.
        """
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        sea_ice_path = scripts_dir / "matrix" / "run_sea_ice_test_matrix.py"
        text = sea_ice_path.read_text()
        # Strip line comments to test ACTIVE code only.
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        # 1) Helper IS imported and called.
        assert "from legoesm.diagnostics.conservation_drift import compute_relative_drift" in code_only, (
            "iter-92: ``run_sea_ice_test_matrix.py`` must import "
            "``compute_relative_drift``."
        )
        # 2) Legacy 1e-20 vol_drift floor must be gone — count
        # occurrences of the inline pattern across all variants.
        # ``r_xy = jnp.sqrt(... + 1e-20)`` for distance is still
        # legitimate (NaN-gradient guard); we only flag the
        # ``vol_drift = abs(.) / max(., 1e-20)`` form.
        import re
        bad_pattern = re.compile(
            r"vol_drift\s*=\s*abs\([^)]*\)\s*/\s*max\([^)]*1e-20\)"
        )
        bad_matches = bad_pattern.findall(code_only)
        assert len(bad_matches) == 0, (
            f"iter-92: found {len(bad_matches)} remaining inline "
            f"``vol_drift = abs(.) / max(., 1e-20)`` patterns in "
            f"``run_sea_ice_test_matrix.py``: {bad_matches}.  All "
            f"must delegate to ``compute_relative_drift``."
        )
        # 3) At least 6 ``compute_relative_drift([..., ...])``
        # callsites for vol_drift.
        good_matches = re.findall(
            r"compute_relative_drift\(\[vol_\w+,\s*vol_\w+\]\)",
            code_only,
        )
        assert len(good_matches) >= 6, (
            f"iter-92: expected ≥6 ``compute_relative_drift`` "
            f"vol_drift call sites, found {len(good_matches)}: "
            f"{good_matches}"
        )

    def test_bench_spectral_sw_uses_helper(self):
        """``bench_spectral_sw.py`` Williamson SW conservation
        plots and TC2/TC5 scalar drifts use the shared helper.
        """
        from pathlib import Path
        bench_path = Path(__file__).resolve().parent.parent.parent / \
                     "tests" / "validation" / "bench_spectral_sw.py"
        text = bench_path.read_text()
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        # Plotting timeseries (mass, energy, enstrophy) — 3 sites.
        assert code_only.count("relative_drift_series(") >= 3, (
            "iter-93: ``bench_spectral_sw.py`` conservation plot "
            "must use ``relative_drift_series`` for mass/energy/"
            "enstrophy (3 callsites)."
        )
        # Scalar drifts (TC2 + TC5: mass + energy = 4 callsites).
        assert code_only.count("compute_relative_drift(") >= 4, (
            "iter-93: ``bench_spectral_sw.py`` TC2/TC5 must use "
            "``compute_relative_drift`` for mass_drift / "
            "energy_drift (4 callsites)."
        )
        # No remaining inline ``/ abs(X[0])`` no-floor patterns.
        import re
        bad = re.findall(
            r"/ abs\([a-zA-Z_]+\[0\]\)", code_only,
        )
        assert len(bad) == 0, (
            f"iter-93: found {len(bad)} remaining ``/ abs(X[0])`` "
            f"no-floor patterns in bench_spectral_sw.py: {bad}"
        )

    def test_w2_w5_cosine_bell_uses_helper(self):
        """``run_w2_w5_cosine_bell_iter1030.py`` mass drift uses
        the shared helper.
        """
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.parent / \
               "scripts" / "run" / "run_w2_w5_cosine_bell_iter1030.py"
        text = path.read_text()
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "compute_relative_drift([mass_init, mass_final])" in code_only, (
            "iter-93: ``run_w2_w5_cosine_bell_iter1030.py`` must "
            "compute mass drift via the shared helper."
        )

    def test_advection_convergence_2d_uses_helper(self):
        """``run_advection_convergence_2d.py`` mass drift uses
        the shared helper.
        """
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent.parent / \
               "scripts" / "run" / "run_advection_convergence_2d.py"
        text = path.read_text()
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "compute_relative_drift([mass_init, mass_final])" in code_only, (
            "iter-93: ``run_advection_convergence_2d.py`` must "
            "compute mass drift via the shared helper."
        )

    def test_bench_spectral_pe_uses_relative_drift_series(self):
        """``bench_spectral_pe.py`` JW06 conservation plot uses the
        shared helper, not an inline ``max(abs(KE_ts[0]), 1e-30)``.
        """
        from pathlib import Path
        bench_path = Path(__file__).resolve().parent.parent.parent / \
                     "tests" / "validation" / "bench_spectral_pe.py"
        text = bench_path.read_text()
        # Strip line comments so we test ACTIVE CODE only — the
        # iter-92 commit message + comments mention the legacy
        # formula for history, which would false-positive a naive
        # literal-text grep.
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "relative_drift_series(mass_ts)" in code_only, (
            "iter-92: ``bench_spectral_pe.py`` mass conservation "
            "plot must use ``relative_drift_series``."
        )
        assert "relative_drift_series(KE_ts)" in code_only, (
            "iter-92: ``bench_spectral_pe.py`` KE conservation "
            "plot must use ``relative_drift_series``."
        )
        # Active code (non-comment) check: the legacy 1e-30 floor
        # is gone from the conservation plotting code.
        assert "/ max(abs(KE_ts[0]), 1e-30)" not in code_only, (
            "iter-92: the legacy 1e-30 KE floor must not appear "
            "in active code."
        )
        assert "/ abs(mass_ts[0])" not in code_only, (
            "iter-92: the unfloored ``/ abs(mass_ts[0])`` "
            "(NaN-on-zero) must not appear in active code — "
            "replaced by ``relative_drift_series(mass_ts)``."
        )


class TestStructuralRegressionNoNewIter78Pathology:
    """iter-94 forward-looking structural regression test.

    iter-90/91/92/93 each found 1+ missed copies of the iter-78
    pathology pattern (``max(abs(x[0]), 1e-30)`` and variants)
    that earlier iterations had not migrated.  This pattern is
    insidious: it's only pathological for rest-state baselines,
    so production runs with non-zero baselines hide the bug, and
    each new copy gets reviewed in isolation rather than against
    the canonical helper.

    This structural test scans ``scripts/``, ``src/legoesm/``,
    and ``tests/`` for *active code* (comments stripped) matching
    the pathological pattern, and asserts ZERO matches.  Any
    future PR that introduces a new inline copy will fail this
    test, so a reviewer / next-iteration audit will be alerted
    *before* the copy ships.

    The test deliberately NOT-checks for legitimate epsilon uses
    (``jnp.sqrt(... + 1e-30)`` for NaN-grad guards, weighted
    means with positive weights, log-of-zero protection, etc.) —
    it only matches the specific iter-78 *baseline-zero
    normalization* shape.
    """

    @staticmethod
    def _strip_comments_and_docstrings(text: str) -> str:
        """Remove ``#`` line comments and triple-quoted strings.

        This is intentionally simple — it does not parse Python.
        The goal is to eliminate the most common false-positive
        sources (history comments, audit-mention docstrings) so
        the regex sees only active code.
        """
        import re
        # Strip ``"""..."""`` and ``'''...'''`` blocks (incl.
        # multi-line, non-greedy).
        no_triple_dbl = re.sub(r'"""[\s\S]*?"""', "", text)
        no_triple = re.sub(r"'''[\s\S]*?'''", "", no_triple_dbl)
        # Strip line comments.
        out_lines = []
        for line in no_triple.splitlines():
            # Inline-comment stripping is risky (strings can contain
            # ``#``); for our regex this rarely matters since the
            # pathological pattern has no ``#`` in it.  Just drop
            # whole-line comments.
            if line.lstrip().startswith("#"):
                continue
            out_lines.append(line)
        return "\n".join(out_lines)

    @staticmethod
    def _walk_python_files():
        """Yield (path, text) for *.py under scripts/, src/legoesm/,
        tests/, excluding __pycache__.
        """
        from pathlib import Path
        from tests.legoesm_paths import legoesm_root_paths
        repo_root = Path(__file__).resolve().parent.parent.parent
        # Carve-aware: scan every legoesm namespace root (the substrate may live
        # outside src/legoesm) plus scripts and tests.
        roots = [
            repo_root / "scripts",
            repo_root / "tests",
            *legoesm_root_paths(),
        ]
        # Files that legitimately contain the pattern as DATA
        # (test assertions checking string presence/absence,
        # the canonical helper module itself).
        exempt_files = {
            repo_root / "src" / "legoesm" / "diagnostics" / "conservation_drift.py",
            repo_root / "tests" / "unit" / "test_conservation_drift.py",
            repo_root / "tests" / "test_atmosphere_cross_grid_plots.py",
            repo_root / "tests" / "test_ocean_cross_grid_plots.py",
        }
        for root in roots:
            for path in root.rglob("*.py"):
                if path in exempt_files:
                    continue
                if "__pycache__" in path.parts:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                yield path, text

    def test_iter_106_regex_catches_decimal_form(self):
        """iter-106 codex iter-104 MEDIUM-5: pin that the
        broadened regex actually distinguishes both
        ``1e-30`` and ``1.0e-30`` decimal forms from the
        canonical helper expression ``max(abs(x), 1.0)``.
        """
        import re
        pat = re.compile(
            r"max\(\s*abs\([^)]*\[0\][^)]*\)\s*,\s*"
            r"\d(?:\.\d+)?e-\d{2,}\s*\)"
        )
        # Should match (iter-78 pathology):
        for bad in [
            "max(abs(mass[0]), 1e-30)",
            "max(abs(mass[0]), 1.0e-30)",
            "max(abs(values[0]), 1.5e-25)",
            "max(abs(energy[0]), 1.0e-30)",
        ]:
            assert pat.search(bad) is not None, (
                f"iter-106: regex must catch {bad!r}"
            )
        # Should NOT match (canonical or different form):
        for good in [
            "max(abs(values[0]), 1.0)",  # canonical helper
            "max(abs(x), 1e-10)",  # no [0] subscript
            "abs(values[0]) + 1.0e-30",  # different operator
        ]:
            assert pat.search(good) is None, (
                f"iter-106: regex must NOT match {good!r}"
            )

    def test_no_inline_max_abs_x_zero_1e30_pattern(self):
        """No ``max(abs(X[0]), 1e-XX)`` (or ``1.0e-XX`` decimal
        form) for tiny epsilon in active code.  The canonical
        helper's ``DEFAULT_MIN_BASELINE = 1.0`` is the single
        source of truth for the floor.

        iter-106 codex iter-104 MEDIUM-5: the original iter-94
        regex used ``1e-\\d{2,}`` and missed the ``1.0e-30``
        form found in
        ``tests/validation/run_dycore_progression_suite.py``.
        Now matches both forms via ``1(?:\\.\\d+)?e-\\d{2,}``.
        """
        import re
        # Match: max(abs(<anything ending in [0])>), Xe-NN)
        # where:
        #   X is "1" or "1.0" / "1.5" etc (decimal form)
        #   NN >= 10 (so 1e-10, 1e-12, 1e-20, 1e-30, 1.0e-30, ...)
        # Catches the iter-78 family but NOT
        # ``max(abs(x), 1.0)`` (the helper's canonical form, no
        # ``e-`` exponent suffix).
        pat = re.compile(
            r"max\(\s*abs\([^)]*\[0\][^)]*\)\s*,\s*"
            r"\d(?:\.\d+)?e-\d{2,}\s*\)"
        )
        offenders = []
        for path, text in self._walk_python_files():
            code_only = self._strip_comments_and_docstrings(text)
            for match in pat.finditer(code_only):
                # Find the line number of the match.
                line_num = code_only[:match.start()].count("\n") + 1
                offenders.append(
                    f"{path.relative_to(path.parents[3])}:~{line_num}  "
                    f"{match.group(0)}"
                )
        assert offenders == [], (
            "iter-94 structural regression: found inline "
            "``max(abs(X[0]), 1e-XX)`` patterns (the iter-78 "
            "baseline-zero pathology) in active code.  Use "
            "``compute_relative_drift`` from "
            "``legoesm.diagnostics.conservation_drift`` instead "
            "of inlining the floor:\n"
            + "\n".join(f"  - {o}" for o in offenders)
        )

    def test_no_inline_abs_x_zero_plus_eps_pattern(self):
        """No ``abs(X[0]) + 1e-XX`` (or ``1.0e-XX`` decimal form)
        denominator pattern.

        This is the iter-87 variant — the HS+RRTMGP runner used
        ``/ abs(mass_vals[0] + 1e-30)`` which has the same iter-78
        pathology.

        iter-106 codex iter-104 MEDIUM-5: broadened regex to
        also match ``1.0e-30`` decimal form.
        """
        import re
        # Match: / (abs(...) + Xe-NN) or / abs(...) + Xe-NN
        pat = re.compile(
            r"/\s*\(?\s*abs\([^)]*\[0\][^)]*\)\s*\+\s*"
            r"\d(?:\.\d+)?e-\d{2,}\s*\)?"
        )
        offenders = []
        for path, text in self._walk_python_files():
            code_only = self._strip_comments_and_docstrings(text)
            for match in pat.finditer(code_only):
                line_num = code_only[:match.start()].count("\n") + 1
                offenders.append(
                    f"{path.relative_to(path.parents[3])}:~{line_num}  "
                    f"{match.group(0)}"
                )
        assert offenders == [], (
            "iter-94 structural regression: found inline "
            "``/ (abs(X[0]) + 1e-XX)`` patterns (iter-87 variant "
            "of the iter-78 pathology) in active code:\n"
            + "\n".join(f"  - {o}" for o in offenders)
        )

    def test_no_inline_max_x_zero_1e20_pattern(self):
        """No ``max(X_init, 1e-20)`` time-zero-baseline denominator
        (iter-92 sea-ice variant).

        Tightening note: the regex requires the variable name to
        match a *time-zero baseline* identifier (``_init``,
        ``_initial``, ``_before``, ``_baseline``, ``_zero``) NOT a
        *norm/scale* identifier (``_scale``, ``_norm``, ``_max``,
        ``_ref``, ``_f``).  This avoids false positives on:

        * ``max(T_scale, 1e-30)`` — relative tendency
          normalization where the denominator is a norm of the
          field, NOT a time-zero baseline.  Found in
          ``scripts/diagnose_redi_residual_at_t0.py`` and
          ``tests/ocean/unit/test_gm_redi_eady_physics.py``;
          legitimate divide-by-zero guard for a degenerate
          (uniform-zero) field.
        * ``max(hl_f, 1e-30)`` — regression-test comparison
          of live vs committed file values
          (``tests/unit/test_cdgrid_fv3_regression.py``); not a
          time-zero baseline.
        """
        import re
        pat = re.compile(
            r"max\(\s*"
            # Variable name MUST end in _init / _initial / _before /
            # _baseline / _zero — names that semantically imply
            # "time-zero baseline" and trigger the iter-78 bug.
            r"\w+(?:_init|_initial|_before|_baseline|_zero)"
            # iter-106: also match decimal form ``1.0e-XX`` (codex
            # iter-104 MEDIUM-5).
            r"\s*,\s*\d(?:\.\d+)?e-(?:1[5-9]|2\d|3\d|4\d)\s*\)"
        )
        offenders = []
        for path, text in self._walk_python_files():
            code_only = self._strip_comments_and_docstrings(text)
            for match in pat.finditer(code_only):
                line_num = code_only[:match.start()].count("\n") + 1
                offenders.append(
                    f"{path.relative_to(path.parents[3])}:~{line_num}  "
                    f"{match.group(0)}"
                )
        assert offenders == [], (
            "iter-94 structural regression: found inline "
            "``max(X_init, 1e-XX)`` patterns (iter-92 sea-ice "
            "variant of the iter-78 pathology) in active code:\n"
            + "\n".join(f"  - {o}" for o in offenders)
        )

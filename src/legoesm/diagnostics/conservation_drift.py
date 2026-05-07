"""Relative drift of scalar conservation diagnostics with baseline-zero safety.

A common pathology in conservation timeseries: when the baseline value
``values[0]`` is exactly zero or near-machine-epsilon (e.g., a rest-state
ocean run where the volume anomaly baseline is 0, or any pristine-quiescent
diagnostic where the time-zero value is identically 0), normalizing as
``|x - x[0]| / abs(x[0])`` is undefined; using ``abs(x[0]) + eps`` with
``eps = 1e-30`` divides ~1e-17 floating-point rounding by the floor and
manufactures spurious 1e+13–1e+30-magnitude "drift" values that masquerade
as catastrophic conservation failures.

This module provides a single floor convention shared by the atmosphere,
ocean, and HS+RRTMGP cross-grid drivers:

* When the baseline magnitude is **below ``min_baseline``** (default 1.0
  in the diagnostic's natural units), report **absolute** drift in those
  units.  This handles rest-state runs where ``x[0] = 0`` cleanly.
* When the baseline is **above ``min_baseline``** (production runs:
  atmosphere mass ~ 5e+19 Pa·m², ocean volume ~ 1.34e+18 m³, ocean
  heat ~ 1e+25 J), report dimensionless **relative** drift.

The piecewise behaviour is intentional — see iter-80 _save_conservation
docstring for the full rationale.

History
-------
* iter-78: cube ocean ``rest_state`` reported -2.83e+13 m volume drift.
* iter-80: traced the spurious magnitude to a 1e-30 denominator floor in
  ``scripts/run_ocean_test_matrix.py:_save_conservation`` and replaced
  it with a 1.0 floor inline.
* iter-83: same pathology spotted in
  ``scripts/run_atmosphere_test_matrix.py:_compute_drift`` (denominator
  ``max(abs(x), 1e-30)``); fixed with the same 1.0 floor inline.
* iter-87: same pathology in
  ``scripts/run_held_suarez_rrtmgp_4grids.py`` (denominator
  ``abs(x[0]) + 1e-30``); fixed inline.
* iter-88: factored the floor logic into this shared helper so every
  current and future caller inherits the iter-80 floor convention from
  one place.  The matrix runners now delegate to ``compute_relative_drift``
  / ``relative_drift_series``.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Default baseline floor (in the diagnostic's natural physical units).
# All currently-tracked conservation diagnostics in legoESM have either
# ``|x[0]| = 0`` (rest state) or ``|x[0]| >> 1`` (production: mass in
# Pa·m², volume in m³, heat in J, salt in kg).  Setting the floor at 1.0
# makes the function piecewise:
#   |x[0]| <= 1.0:  return absolute drift in natural units
#   |x[0]| >  1.0:  return dimensionless relative drift
# See module docstring for the iter-78/80/83/87 history.
DEFAULT_MIN_BASELINE: float = 1.0


def _validate_min_baseline(min_baseline: float) -> None:
    """Reject non-finite or non-positive ``min_baseline`` values.

    iter-89 self-review: silently accepting ``inf`` would make every
    drift evaluate to 0 (since ``denom = inf``); silently accepting
    ``nan`` would make every drift NaN.  Both cases are pure user
    error and a fail-fast ValueError is preferable to a silent
    misleading number.
    """
    if not math.isfinite(min_baseline):
        raise ValueError(
            f"min_baseline must be a finite positive float, "
            f"got {min_baseline!r}"
        )
    if min_baseline <= 0.0:
        raise ValueError(
            f"min_baseline must be > 0, got {min_baseline!r}"
        )


def compute_relative_drift(
    values: Sequence[float],
    *,
    min_baseline: float = DEFAULT_MIN_BASELINE,
) -> float:
    """Scalar drift between the first and last samples of ``values``.

    Returns ``|values[-1] - values[0]| / max(|values[0]|, min_baseline)``.

    Empty / single-element behaviour
    --------------------------------
    For ``len(values) < 2`` returns ``0.0`` (drift is undefined for a
    single sample).

    Non-finite inputs
    -----------------
    iter-90 codex review (MEDIUM-3, MEDIUM-5):
    * If ``values[0]`` is NaN, ``abs(NaN) = NaN`` and ``max(NaN, x)``
      returns NaN under Python's stdlib ``max`` — so the result is
      NaN.  This signals "diagnostic was already broken" rather than
      silently masking it.
    * If ``values[-1]`` is NaN or inf, the result is NaN or inf,
      again a deliberate "broken diagnostic" signal.
    Callers wanting a different convention should pre-clean their
    series.  This is consistent with the legacy inline behaviour at
    all three iter-80/83/87 callsites.

    Sign
    ----
    Always returns ``>= 0`` (uses ``abs`` on the numerator).  See
    ``relative_drift_series`` for the signed per-step series.

    Parameters
    ----------
    values
        Conservation diagnostic timeseries (Python list, tuple, or
        ``np.ndarray``).  Coerced to float64 internally.
    min_baseline
        Floor on the denominator magnitude.  Must be a finite
        positive float (rejected by ``_validate_min_baseline``
        otherwise).  Default 1.0 — see module docstring.

    See module docstring for the iter-78/80/83/87 history of the
    ``min_baseline`` floor.
    """
    _validate_min_baseline(min_baseline)
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    # iter-152 (codex iter-151 review HIGH-1): if ANY sample
    # in the series is non-finite (NaN or Inf), surface it as a
    # NaN result so the downstream gate fails explicitly.  Pre-
    # iter-152 the function only inspected arr[0] and arr[-1],
    # so a series like ``[1.0, NaN, 1.0]`` would yield drift=0.0
    # and PASS the gate even though the simulation went transiently
    # unstable mid-run.  This was a real silent-pass failure mode.
    if not np.all(np.isfinite(arr)):
        return float("nan")
    denom = max(abs(float(arr[0])), float(min_baseline))
    return float(abs(arr[-1] - arr[0]) / denom)


def relative_drift_series(
    values: Sequence[float],
    *,
    min_baseline: float = DEFAULT_MIN_BASELINE,
) -> np.ndarray:
    """Per-step drift series ``values - values[0]`` normalized by the baseline.

    Returns a ``np.ndarray`` of the same length as ``values`` (always
    1-D, dtype float64, always a fresh copy — never aliased to the
    input).  Each element is
    ``(values[i] - values[0]) / max(|values[0]|, min_baseline)``.

    Empty / single-element behaviour
    --------------------------------
    iter-90 codex review (LOW-7): for ``len(values) == 0`` returns an
    empty float64 array (shape ``(0,)``).  For ``len(values) == 1``
    returns ``np.zeros(1)`` (since ``values[0] - values[0] = 0``).
    This differs from ``compute_relative_drift`` which collapses both
    cases to scalar 0.0 — the series API preserves the input shape.

    Sign
    ----
    iter-90 codex review (MEDIUM-4): the series IS signed
    (``arr - arr[0]`` not ``abs(arr - arr[0])``) so that callers can
    distinguish drift in opposite directions in their plots — the
    ocean ``_save_conservation`` panel uses this to show whether
    volume is gaining or losing mass over time.  To compare against
    ``compute_relative_drift``'s scalar result, take
    ``abs(result[-1])``.

    Non-finite inputs
    -----------------
    Same NaN/inf propagation rules as ``compute_relative_drift`` —
    see that function's docstring.

    Parameters
    ----------
    values
        Conservation diagnostic timeseries (Python list, tuple, or
        ``np.ndarray``).
    min_baseline
        See ``compute_relative_drift``.

    Used by ``scripts/run_ocean_test_matrix.py:_save_conservation`` to
    emit a per-timestep drift trace for plotting.
    """
    _validate_min_baseline(min_baseline)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        # iter-90 codex review (LOW-6): return a fresh empty array
        # so callers can't accidentally mutate their input via the
        # returned alias.
        return np.empty(0, dtype=np.float64)
    denom = max(abs(float(arr[0])), float(min_baseline))
    return (arr - arr[0]) / denom


def apply_drift_tolerance(
    ok: bool, notes: str, drift: float, tol: float,
    *, label: str, n_samples: int | None = None,
) -> tuple[bool, str]:
    """Apply a drift PASS tolerance to a test case.

    iter-127 (codex iter-126-followup LOW-5/maintainability):
    centralized version of the helper introduced in iter-117
    for the atmosphere matrix runner (HS) and ported in iter-123/
    124/125/126 to the ocean matrix monolithic + modular
    runners.  Pre-iter-127 there were 2 byte-near-identical
    copies (one in each runner) plus a third in the
    atmosphere matrix.  iter-127 centralizes here so future
    fixes flow through a single source of truth.

    Behaviour:

    * If the input series has fewer than ``n_samples=2`` samples
      (when ``n_samples`` is provided), fail with a clear
      ``[FAIL: <label> series has only N sample(s); ...]``
      annotation (iter-120 fix).
    * If ``drift`` is non-finite (NaN or Inf), fail with a
      ``[FAIL: <label> drift is non-finite ...]`` annotation
      (iter-118 fix).
    * If ``drift > tol``, fail with a
      ``[FAIL: <label> drift X.YYe-ZZ > tolerance ...]``
      annotation (iter-117 baseline behaviour).
    * Idempotent on already-failed runs (``ok=False`` short-
      circuits all of the above).

    The atmosphere matrix runner ``_apply_mass_drift_tolerance``
    delegates to this helper specialized for ``label="mass"``
    (post-iter-128 codex iter-127-followup LOW-1).  The two ocean
    runners (monolithic ``scripts/run_ocean_test_matrix.py`` and
    modular ``scripts/ocean_test_matrix/timeloop.py``) likewise
    delegate via thin ``_apply_drift_tolerance`` wrappers.

    Parameters
    ----------
    ok
        Current PASS state of the test case.  False short-
        circuits (idempotent).
    notes
        Per-test-case ``notes`` string that gets appended to.
    drift
        Computed drift magnitude.
    tol
        PASS tolerance.  Drift > tol or non-finite fails.
    label
        Diagnostic label (e.g., ``"mass"``, ``"eta"``,
        ``"T"``, ``"S_integral"``).  Keyword-only per
        iter-124 LOW-4.
    n_samples
        Optional length of the underlying series.  When
        provided and < 2, fail with a "too few samples"
        annotation.

    Returns
    -------
    (ok, notes): tuple of updated values.
    """
    import numpy as _np
    # iter-152 (codex iter-151 review MEDIUM-3): validate the
    # tolerance itself is finite — otherwise ``drift > NaN`` is
    # always False and a finite-but-broken drift would silently
    # PASS.  Raise on tol=NaN/Inf rather than fail-quietly so
    # callers get a clear ``ValueError`` traceback at the
    # invocation site.
    if not _np.isfinite(tol):
        raise ValueError(
            f"apply_drift_tolerance: tol must be finite, got "
            f"{tol!r} (label={label!r})"
        )
    if ok and n_samples is not None and n_samples < 2:
        ok = False
        notes += (
            f" [FAIL: {label} series has only {n_samples} "
            f"sample(s); need >= 2 for a valid drift]"
        )
        return ok, notes
    if ok and (not _np.isfinite(drift) or drift > tol):
        ok = False
        if not _np.isfinite(drift):
            notes += (
                f" [FAIL: {label} drift is non-finite "
                f"({drift!r})]"
            )
        else:
            notes += (
                f" [FAIL: {label} drift {drift:.2e} > "
                f"tolerance {tol:.0e}]"
            )
    return ok, notes


def apply_value_threshold(
    ok: bool, notes: str, value: float, threshold: float,
    *, label: str, op: str = "le", units: str = "",
    n_samples: int | None = None,
) -> tuple[bool, str]:
    """Apply a non-drift PASS threshold (overshoot, sign-check, etc.).

    iter-128 (codex iter-127-followup MEDIUM-2/3): companion to
    ``apply_drift_tolerance`` for tests that gate on non-drift
    quantities — e.g., the documented Stommel
    ``overshoot < 0.1 PSU`` and ``undershoot < 0.1 PSU`` thresholds
    in the "Stommel Gyre Tracer" section, and the Overflow /
    Lock-Exchange ``pe_rel_final < 0`` sign check in their
    respective Validation Thresholds blocks (see
    ``docs/ocean_experiments_reference.md``; iter-131 codex
    iter-130-followup LOW-2: removed stale line numbers).

    iter-129 (codex iter-128-followup LOW-1/3/MEDIUM-1):
      * Removed ``op="lt_zero"`` (one-iteration-old API; replaced
        by the more general ``op="lt"`` with explicit ``threshold``
        argument).  This eliminates the silent ignoring of
        ``threshold`` that LOW-3 flagged.
      * Added ``op="lt"`` (strict less-than): fail if
        ``value >= threshold``.  Stommel uses this with
        ``threshold=0.1`` to match the documented
        ``< 0.1 PSU`` strict bound (LOW-1).
      * Added ``n_samples`` kwarg (parity with
        ``apply_drift_tolerance``); when provided and < 2,
        fail explicitly so missing/single-sample diagnostics
        cannot silently pass via a default-zero placeholder
        (MEDIUM-1).

    Behaviour:

    * ``op="le"`` (less-than-or-equal): fail if ``value > threshold``
      (or non-finite).  Used for overshoot/undershoot/absolute
      magnitude tests where the boundary value passes.
    * ``op="lt"`` (strict less-than): fail if ``value >= threshold``
      (or non-finite).  Used for sign checks (e.g., RPE must
      strictly decrease) and any documented ``< X`` strict
      bound.
    * ``op="ge"`` (greater-than-or-equal): fail if ``value < threshold``
      (or non-finite).  Used for lower-bound tests (e.g., Phillips
      ``eta_growth >= 0.8``).  Added iter-131 (codex iter-130-followup
      HIGH-1) for range checks.
    * Idempotent on already-failed runs (``ok=False`` short-
      circuits).

    Returns
    -------
    (ok, notes): tuple of updated values.
    """
    import numpy as _np
    # iter-152 (codex iter-151 review MEDIUM-3): validate the
    # threshold itself is finite — otherwise comparisons against
    # NaN/Inf threshold are always False and a finite-but-broken
    # value would silently PASS.  Raise rather than fail-quietly.
    if not _np.isfinite(threshold):
        raise ValueError(
            f"apply_value_threshold: threshold must be finite, got "
            f"{threshold!r} (label={label!r}, op={op!r})"
        )
    if not ok:
        return ok, notes
    if n_samples is not None and n_samples < 2:
        ok = False
        notes += (
            f" [FAIL: {label} series has only {n_samples} "
            f"sample(s); need >= 2 for a valid threshold check]"
        )
        return ok, notes
    if not _np.isfinite(value):
        ok = False
        notes += (
            f" [FAIL: {label} is non-finite ({value!r})]"
        )
        return ok, notes
    if op == "le":
        if value > threshold:
            ok = False
            notes += (
                f" [FAIL: {label}={value:.3g}{units} > "
                f"threshold {threshold:.3g}{units}]"
            )
    elif op == "lt":
        if value >= threshold:
            ok = False
            notes += (
                f" [FAIL: {label}={value:.3g}{units} >= "
                f"threshold {threshold:.3g}{units}; "
                f"expected strictly less-than]"
            )
    elif op == "ge":
        if value < threshold:
            ok = False
            notes += (
                f" [FAIL: {label}={value:.3g}{units} < "
                f"threshold {threshold:.3g}{units}; "
                f"expected greater-than-or-equal]"
            )
    else:
        raise ValueError(
            f"apply_value_threshold: unknown op {op!r} "
            f"(expected 'le', 'lt', or 'ge')"
        )
    return ok, notes

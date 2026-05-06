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

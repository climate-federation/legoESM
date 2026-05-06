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
  atmosphere mass ~ 5e+19 Pa·m², ocean volume ~ 10²⁰ m³, ocean heat
  ~ 10²⁵ J), report dimensionless **relative** drift.

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


def compute_relative_drift(
    values: Sequence[float],
    *,
    min_baseline: float = DEFAULT_MIN_BASELINE,
) -> float:
    """Scalar drift between the first and last samples of ``values``.

    Returns ``|values[-1] - values[0]| / max(|values[0]|, min_baseline)``.

    For series shorter than 2 entries returns ``0.0`` (drift is undefined
    for a single sample).  ``min_baseline`` must be strictly positive.

    See module docstring for the iter-78/80/83/87 history of the
    ``min_baseline`` floor.
    """
    if min_baseline <= 0.0:
        raise ValueError(
            f"min_baseline must be > 0, got {min_baseline!r}"
        )
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

    Returns a ``np.ndarray`` of the same length as ``values`` (always at
    least 1-D, dtype float64).  Each element is
    ``(values[i] - values[0]) / max(|values[0]|, min_baseline)``.

    Used by ``scripts/run_ocean_test_matrix.py:_save_conservation`` to
    emit a per-timestep drift trace for plotting.  ``min_baseline``
    defaults to 1.0 — see module docstring.
    """
    if min_baseline <= 0.0:
        raise ValueError(
            f"min_baseline must be > 0, got {min_baseline!r}"
        )
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    denom = max(abs(float(arr[0])), float(min_baseline))
    return (arr - arr[0]) / denom

"""Shared utilities for ocean unit tests.

Convergence-rate helper used by the per-term validation tests landed
under Phase 1 of the Adcroft follow-up plan.  See
``docs/ocean/per_term_test_methodology.md`` for why we assert on
convergence rate (not absolute L2 error).
"""

from __future__ import annotations

from typing import Sequence

import math


def convergence_rate(
    errors: Sequence[float],
    resolutions: Sequence[int],
) -> list[float]:
    """Compute pairwise convergence rates from a resolution sweep.

    For a scheme of formal order ``p`` and a sequence of error norms
    ``e_i`` at resolutions ``n_i`` (where ``n_i`` is some "resolution
    number" — e.g. grid points per side, refinement level, or inverse
    grid spacing — that is *larger* on finer grids), the expected
    relation is

        e_{i+1} / e_i  =  (n_i / n_{i+1})^p

    so the empirical rate between consecutive pairs is

        r_i  =  log2(e_i / e_{i+1}) / log2(n_{i+1} / n_i).

    Returns one rate per *pair* of successive entries, so the output
    list has length ``len(errors) - 1``.

    Parameters
    ----------
    errors
        Error norms at increasing resolution.  Must be strictly
        positive.
    resolutions
        Matching resolution counts; strictly increasing.  Same length
        as ``errors``.

    Returns
    -------
    list[float]
        Pairwise rates.  An ideal ``p``-th-order scheme gives a list
        of values close to ``p``; degradation toward 1 typically signals
        limiter activation (TVD) or a non-smooth feature dominating
        the error.

    Examples
    --------
    >>> errors = [1.0, 0.25, 0.0625, 0.015625]   # halves each refinement
    >>> resolutions = [16, 32, 64, 128]
    >>> rates = convergence_rate(errors, resolutions)
    >>> all(abs(r - 2.0) < 1e-12 for r in rates)
    True
    """
    if len(errors) != len(resolutions):
        raise ValueError(
            f"errors and resolutions length mismatch: "
            f"{len(errors)} vs {len(resolutions)}"
        )
    if len(errors) < 2:
        raise ValueError(
            "convergence_rate needs at least two (error, resolution) pairs."
        )
    for i, e in enumerate(errors):
        if e <= 0.0:
            raise ValueError(f"errors[{i}] = {e!r}; must be strictly positive.")
    for i in range(1, len(resolutions)):
        if resolutions[i] <= resolutions[i - 1]:
            raise ValueError(
                "resolutions must be strictly increasing; got "
                f"{resolutions[i-1]} -> {resolutions[i]} at index {i}."
            )

    rates: list[float] = []
    for i in range(len(errors) - 1):
        ratio_err = errors[i] / errors[i + 1]
        ratio_res = resolutions[i + 1] / resolutions[i]
        rates.append(math.log2(ratio_err) / math.log2(ratio_res))
    return rates


def assert_convergence_rate_at_least(
    errors: Sequence[float],
    resolutions: Sequence[int],
    expected_order: float,
    tolerance: float = 0.15,
) -> None:
    """Assert that every pairwise rate is at least ``expected_order - tolerance``.

    A 2nd-order scheme is expected to produce rates near 2.0; we
    accept anything ``>= 2 - tolerance``.  Tolerance of 0.15 (the
    default) is the standard "within 7-8% of nominal order" check.

    Raises
    ------
    AssertionError
        With a message reporting each rate against the threshold so
        the failing pair is immediately visible.
    """
    rates = convergence_rate(errors, resolutions)
    threshold = expected_order - tolerance
    failing = [(i, r) for i, r in enumerate(rates) if r < threshold]
    if failing:
        msg_lines = [
            f"Convergence-rate assertion failed: expected each rate >= "
            f"{threshold:.3f} (order {expected_order:.2f} - tolerance "
            f"{tolerance:.3f}).",
            f"  resolutions: {list(resolutions)}",
            f"  errors:      {[f'{e:.3e}' for e in errors]}",
            f"  pairwise rates: {[f'{r:.3f}' for r in rates]}",
            f"  failing pairs (index, rate): {failing}",
        ]
        raise AssertionError("\n".join(msg_lines))

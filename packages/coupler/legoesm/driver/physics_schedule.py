"""Outer-step physics-call schedule arithmetic for the CRM driver.

Single source of truth for "how often does the radiation tick (or any
held-tendency slow physics) fire?". Lives in ``legoesm.driver`` so:

  1. ``scripts/run/run_rce_mpi_long.py`` and the production-scale slow
     tests in ``tests/atmosphere/nonhydrostatic/integration/`` both
     import the SAME ``radiation_call_schedule`` instead of each
     reimplementing the
     ``max(1, round(rad_call_interval_s / dt))`` + the
     ``1 + (total_steps - 1) // every`` formulas (Codex iter-39/40/41
     review trail flagged the inline-arithmetic pattern as a
     silent-divergence risk).
  2. Future drivers (long-run wrappers, batch sweeps) reuse the
     same logic without copy-paste.

The schedule matches the driver loop ``if (step - 1) % every == 0`` at
``step in [1, total_steps]``, so the very first step always fires
(when radiation is not disabled), and subsequent fires land at
``1 + k * every`` for integer ``k``.
"""
from __future__ import annotations

import math
import sys
from typing import NamedTuple


class RadiationCallSchedule(NamedTuple):
    """Result of ``radiation_call_schedule(rad_call_interval_s, dt, total_steps)``.

    Fields:
        every_steps:
            Number of outer steps between radiation refreshes.
            Always >= 1 (sub-dt intervals are clamped up).
        num_calls:
            How many times the radiation tendency is recomputed
            across the full outer loop ``step in [1, total_steps]``.
            Zero when ``total_steps <= 0``.
        fire_step_indices:
            Lazy ``range`` of 1-based outer-step indices at which
            the radiation tendency fires. For total_steps=60, dt=5,
            interval=60: every_steps=12 → range(1, 61, 12), which
            yields (1, 13, 25, 37, 49). iter-42 Codex MEDIUM fix:
            uses ``range`` instead of ``tuple`` so a 1M-step run
            doesn't allocate a million ints just to count them.
            Callers comparing against a tuple should call
            ``tuple(schedule.fire_step_indices)``.
    """

    every_steps: int
    num_calls: int
    fire_step_indices: range


def radiation_call_every_steps(rad_call_interval_s: float, dt: float) -> int:
    """Outer-step gap between radiation refreshes.

    Mirrors the driver line
    ``rad_call_every_steps = max(1, int(round(rad_call_interval_s / dt)))``.
    Clamped to >= 1 so a sub-dt interval still fires every outer step
    (which is the only safe behaviour without sub-step interpolation).
    """
    # iter-42 Codex 2nd-pass MEDIUM: reject NaN/inf BEFORE the
    # ordering / round-trip guards. NaN compares False against
    # anything so ``dt <= 0.0`` would silently pass for ``dt=nan``
    # and then ``round(rad/nan)`` raises an opaque ValueError;
    # ``inf`` propagates into ``round(inf) → OverflowError``.
    if not math.isfinite(dt):
        raise ValueError(
            f"radiation_call_every_steps: dt must be finite, got {dt!r}"
        )
    if not math.isfinite(rad_call_interval_s):
        raise ValueError(
            f"radiation_call_every_steps: rad_call_interval_s must "
            f"be finite, got {rad_call_interval_s!r}"
        )
    if dt <= 0.0:
        raise ValueError(
            f"radiation_call_every_steps: dt must be positive, got {dt!r}"
        )
    if rad_call_interval_s < 0.0:
        raise ValueError(
            f"radiation_call_every_steps: rad_call_interval_s must be "
            f"non-negative, got {rad_call_interval_s!r}"
        )
    return max(1, int(round(rad_call_interval_s / dt)))


def radiation_call_schedule(
    rad_call_interval_s: float,
    dt: float,
    total_steps: int,
) -> RadiationCallSchedule:
    """Compute the full radiation-call schedule for an outer loop.

    Arguments mirror the driver's CLI / loop control. ``total_steps``
    is the loop upper bound (Python ``range(1, total_steps + 1)``).
    A non-positive ``total_steps`` yields zero calls (used by smoke
    tests with ``--days 0``).

    Does not consider ``--no-radiation``: that flag short-circuits
    the entire schedule branch in the driver, yielding zero calls.
    Callers should special-case ``--no-radiation`` themselves rather
    than poking this helper.
    """
    if not isinstance(total_steps, int):
        raise TypeError(
            f"radiation_call_schedule: total_steps must be int, got "
            f"{type(total_steps).__name__}"
        )
    # iter-42 Codex 2nd-pass MEDIUM: ``range(1, total_steps + 1, ...)``
    # overflows when ``total_steps == sys.maxsize``. Hard-cap below
    # that boundary — no realistic ESM run gets even close (a
    # 10-million-step run would still be 4 orders of magnitude below
    # sys.maxsize on a 64-bit system), so this is a fail-loudly guard
    # against logic bugs (passing the wrong int, negative-step
    # underflow producing a huge unsigned value, etc.).
    if total_steps >= sys.maxsize:
        raise ValueError(
            f"radiation_call_schedule: total_steps={total_steps} too "
            f"large (>= sys.maxsize={sys.maxsize}). This is likely a "
            f"logic bug — no realistic ESM run exceeds 10^9 steps."
        )
    every = radiation_call_every_steps(rad_call_interval_s, dt)
    if total_steps <= 0:
        return RadiationCallSchedule(
            every_steps=every,
            num_calls=0,
            fire_step_indices=range(1, 1, every),
        )
    # iter-42 Codex MEDIUM fix: O(1) memory + O(1) num_calls. The
    # closed-form 1 + (total_steps - 1) // every avoids materializing
    # a tuple of fire indices for multi-million-step production runs.
    fires = range(1, total_steps + 1, every)
    num_calls = 1 + (total_steps - 1) // every
    return RadiationCallSchedule(
        every_steps=every,
        num_calls=num_calls,
        fire_step_indices=fires,
    )

"""Unit tests for ``legoesm.driver.physics_schedule``.

Pins the radiation-call schedule arithmetic that the production
driver (``scripts/run_rce_mpi_long.py``) and the production-scale
slow tests (``tests/atmosphere/nonhydrostatic/integration/
test_plane_crm_end_to_end_smoke.py``) both depend on. iter-42
extracted this from inline duplication to ``legoesm.driver`` so
the test pyramid is rooted at one canonical formula.
"""
from __future__ import annotations

import pytest

from legoesm.driver import (
    RadiationCallSchedule,
    radiation_call_every_steps,
    radiation_call_schedule,
)


# ---------------------------------------------------------------------------
# radiation_call_every_steps
# ---------------------------------------------------------------------------


def test_every_steps_basic():
    """Driver line equivalent: max(1, round(60/5)) = 12."""
    assert radiation_call_every_steps(60.0, 5.0) == 12


def test_every_steps_sub_dt_clamped_to_one():
    """sub-dt intervals (rad < dt) clamp to firing every outer step.

    iter-42 contract: the only safe behaviour without sub-step
    interpolation. A bug here would have radiation firing zero
    times because ``rad_interval / dt = 0`` would round to 0.
    """
    assert radiation_call_every_steps(2.0, 5.0) == 1
    assert radiation_call_every_steps(0.5, 5.0) == 1
    assert radiation_call_every_steps(0.0, 5.0) == 1


def test_every_steps_huge_interval():
    """A huge interval (e.g. 1e9 s, the historical "disable" pattern
    that Codex iter-39 HIGH#2 found doesn't actually disable
    radiation) returns a huge step gap but still finite, so the
    driver still fires radiation ONCE at step 1.
    """
    every = radiation_call_every_steps(1.0e9, 5.0)
    assert every == 200_000_000
    sched = radiation_call_schedule(1.0e9, 5.0, total_steps=60)
    assert sched.num_calls == 1
    assert tuple(sched.fire_step_indices) == (1,)


def test_every_steps_rounds_to_nearest():
    """Driver uses int(round(...)) — 28/5 = 5.6 → 6, 27/5 = 5.4 → 5."""
    assert radiation_call_every_steps(28.0, 5.0) == 6
    assert radiation_call_every_steps(27.0, 5.0) == 5


def test_every_steps_rejects_invalid_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        radiation_call_every_steps(60.0, 0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        radiation_call_every_steps(60.0, -1.0)


def test_every_steps_rejects_negative_interval():
    with pytest.raises(ValueError, match="non-negative"):
        radiation_call_every_steps(-1.0, 5.0)


def test_every_steps_rejects_nan_dt():
    """iter-42 Codex 2nd-pass MEDIUM: NaN dt would silently bypass
    the ``dt <= 0`` guard (NaN compares False vs anything) and then
    fail in round() with an opaque error."""
    with pytest.raises(ValueError, match="dt must be finite"):
        radiation_call_every_steps(60.0, float("nan"))


def test_every_steps_rejects_inf_dt():
    with pytest.raises(ValueError, match="dt must be finite"):
        radiation_call_every_steps(60.0, float("inf"))


def test_every_steps_rejects_nan_interval():
    with pytest.raises(ValueError, match="rad_call_interval_s must be finite"):
        radiation_call_every_steps(float("nan"), 5.0)


def test_every_steps_rejects_inf_interval():
    """A literal float('inf') interval would cause round(inf)
    OverflowError in CPython. Reject at the boundary."""
    with pytest.raises(ValueError, match="rad_call_interval_s must be finite"):
        radiation_call_every_steps(float("inf"), 5.0)


# ---------------------------------------------------------------------------
# radiation_call_schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rad_interval_s, dt, total_steps, expected_every, "
    "expected_calls, expected_fires",
    [
        # iter-39 production-scale slow test (132x132x30, 60 steps,
        # rad-interval=60): 5 calls at 1, 13, 25, 37, 49.
        (60.0, 5.0, 60, 12, 5, (1, 13, 25, 37, 49)),
        # iter-16 short smoke (12x12, 34 steps, rad-interval=30):
        # 6 calls at 1, 7, 13, 19, 25, 31.
        (30.0, 5.0, 34, 6, 6, (1, 7, 13, 19, 25, 31)),
        # Every-step fire (rad_interval = dt).
        (5.0, 5.0, 10, 1, 10, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        # Sub-dt interval clamped to every-step.
        (1.0, 5.0, 4, 1, 4, (1, 2, 3, 4)),
        # Huge interval — only step 1 fires.
        (1.0e9, 5.0, 60, 200_000_000, 1, (1,)),
        # iter-42 Codex LOW: total_steps=1 boundary
        # (one outer step → exactly one fire at step 1).
        (60.0, 5.0, 1, 12, 1, (1,)),
        # total_steps=0 → empty schedule (smoke tests with --days 0).
        (60.0, 5.0, 0, 12, 0, ()),
    ],
)
def test_schedule_table(
    rad_interval_s, dt, total_steps,
    expected_every, expected_calls, expected_fires,
):
    schedule = radiation_call_schedule(rad_interval_s, dt, total_steps)
    assert isinstance(schedule, RadiationCallSchedule)
    assert schedule.every_steps == expected_every
    assert schedule.num_calls == expected_calls
    # iter-42 Codex MEDIUM: fire_step_indices is now a lazy range
    # (O(1) memory for multi-million-step runs); compare via tuple
    # since range == tuple is False.
    assert tuple(schedule.fire_step_indices) == expected_fires


def test_schedule_first_fire_always_at_step_one():
    """Driver loop ``range(1, total_steps + 1)`` + ``(step-1) % every == 0``
    guarantees step 1 always fires (when radiation is on). Codex iter-39
    HIGH#2 exposed this — same property must hold across every interval.
    """
    for rad_interval in (1.0, 5.0, 30.0, 60.0, 600.0, 1.0e9):
        sched = radiation_call_schedule(rad_interval, 5.0, total_steps=10)
        first_fire = next(iter(sched.fire_step_indices))
        assert first_fire == 1, (
            f"first fire at step {first_fire} != 1 "
            f"for rad_interval={rad_interval}"
        )


def test_schedule_memory_stays_constant_for_huge_total_steps():
    """iter-42 Codex MEDIUM: fire_step_indices is a lazy ``range``,
    so a 1M-step run does NOT allocate a million ints. Verified by
    checking the type is range (range materializes lazily).
    """
    sched = radiation_call_schedule(60.0, 5.0, total_steps=10_000_000)
    assert isinstance(sched.fire_step_indices, range)
    # The num_calls field is computed in O(1) without materializing.
    assert sched.num_calls == 1 + (10_000_000 - 1) // sched.every_steps


def test_schedule_num_calls_matches_fire_count():
    """``num_calls == len(fire_step_indices)`` as an internal invariant
    — protects callers who use one but not the other.
    """
    for rad_interval, dt, n in [
        (60.0, 5.0, 60), (30.0, 5.0, 34), (5.0, 5.0, 10),
        (1.0e9, 5.0, 60), (60.0, 1.0, 1000),
    ]:
        sched = radiation_call_schedule(rad_interval, dt, n)
        assert sched.num_calls == len(sched.fire_step_indices)


def test_schedule_rejects_non_int_total_steps():
    with pytest.raises(TypeError, match="total_steps must be int"):
        radiation_call_schedule(60.0, 5.0, 60.0)  # type: ignore[arg-type]


def test_schedule_rejects_sys_maxsize_total_steps():
    """iter-42 Codex 2nd-pass MEDIUM: range(1, sys.maxsize + 1, ...)
    overflows. Fail loudly so a logic bug (wrong int, unsigned
    underflow) trips early instead of producing a confusing
    OverflowError deep inside CPython."""
    import sys
    with pytest.raises(ValueError, match="too large"):
        radiation_call_schedule(60.0, 5.0, sys.maxsize)

"""
JAX/Python translation of the CLM time manager module.

Provides date arithmetic, calendar-day computation, and timestep
queries used throughout the multilayer canopy code.

Calendar conventions (preserved verbatim from Fortran):

- ``calkindflag = 'GREGORIAN'`` — Gregorian calendar with standard
  leap-year rules.
- Calendar day 1.000 = 0Z on 1 January of the current year.
- The *current date* is the date at the **end** of the current
  timestep (``itim * dtstep`` elapsed seconds from the start date).
- Dates are stored as ``yyyymmdd`` integers, e.g. 19960701.
- Time of day is stored as integer seconds past 0Z.

Public API
----------
- :data:`dtstep` / :data:`itim` — module-level mutable time state.
- :func:`get_step_size`
- :func:`get_nstep`
- :func:`isleap`
- :func:`get_curr_date`
- :func:`get_curr_time`
- :func:`get_curr_calday`
- :func:`is_end_curr_day`
- :func:`is_end_curr_month`

Private helpers
---------------
- :func:`_get_prev_date`
- :func:`_get_prev_calday`

Original Fortran module: clm_time_manager
"""

from __future__ import annotations

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level time state — Fortran public module variables
# ---------------------------------------------------------------------------

dtstep: int = 0  # Model timestep (s); Fortran: integer, public :: dtstep
itim: int = 0  # Current model timestep number; Fortran: integer, public :: itim

# Private simulation date state
start_date_ymd: int = 0  # Start date in yyyymmdd format
start_date_tod: int = 0  # Start time of day (s past 0Z)
curr_date_ymd: int = 0  # Current date in yyyymmdd format (end of timestep)
curr_date_tod: int = 0  # Current time of day (s past 0Z)

# ---------------------------------------------------------------------------
# Calendar tables — Fortran parameter arrays
# ---------------------------------------------------------------------------

calkindflag: str = "GREGORIAN"

# Days in each month, normal and leap — Fortran: mday(12), mdayleap(12)
_mday: tuple = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_mdayleap: tuple = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Cumulative days at end of month (0-based index 0..12) — Fortran: mdaycum(0:12), mdayleapcum(0:12)
_mdaycum: tuple = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
_mdayleapcum: tuple = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366)


# ---------------------------------------------------------------------------
# Public: get_step_size
# ---------------------------------------------------------------------------


def get_step_size() -> int:
    """
    Return the model timestep in seconds.

    Mirrors Fortran function ``get_step_size`` (lines 82-88).

    Returns:
        ``dtstep``.
    """
    return dtstep


# ---------------------------------------------------------------------------
# Public: get_nstep
# ---------------------------------------------------------------------------


def get_nstep() -> int:
    """
    Return the current model timestep number.

    Mirrors Fortran function ``get_nstep`` (lines 90-96).

    Returns:
        ``itim``.
    """
    return itim


# ---------------------------------------------------------------------------
# Public: isleap
# ---------------------------------------------------------------------------


def isleap(year: int, calendar: str) -> bool:
    """
    Return ``True`` if ``year`` is a leap year.

    Mirrors Fortran function ``isleap`` (lines 98-119).

    Leap-year rules for ``'GREGORIAN'`` calendar:

    .. code-block:: none

        divisible by   4   → leap (29 days in Feb)
        divisible by 100   → not leap (28 days)
        divisible by 400   → leap  (29 days)

    For any other calendar value (e.g. ``'NOLEAP'``) the function
    always returns ``False``.

    Args:
        year: Four-digit year (e.g. 1996).
        calendar: Calendar type string (``'GREGORIAN'`` or ``'NOLEAP'``).

    Returns:
        ``True`` iff ``year`` is a leap year.
    """
    result = False  # Fortran: isleap = .false.
    if calendar.strip() == "GREGORIAN":
        if year % 4 == 0:
            result = True  # Every 4 years → leap
            if year % 100 == 0:
                result = False  # Every 100 years → not leap
                if year % 400 == 0:
                    result = True  # Every 400 years → leap
    return result


# ---------------------------------------------------------------------------
# Public: get_curr_date
# ---------------------------------------------------------------------------


def get_curr_date() -> tuple:
    """
    Return date components valid at the end of the current timestep.

    Mirrors Fortran subroutine ``get_curr_date`` (lines 121-181).

    Algorithm (Fortran lines 139-180):

    1. Compute elapsed seconds: ``nsecs = itim * dtstep``.
    2. Compute elapsed days: ``ndays = (nsecs + start_date_tod) // 86400``.
    3. Estimate elapsed years from elapsed days (using 365 or 366 per
       year depending on whether the start year is a leap year).
    4. Compute day-of-year: ``ndays = ndays % (365 or 366)``.
    5. Compute time-of-day: ``tod = (nsecs + start_date_tod) % 86400``.
    6. Initialise ``mcyear, mcmnth, mcday`` from start date + offsets.
    7. Loop (replacing Fortran ``go to 10``) to roll over month and
       year boundaries until ``mcday <= days_in_month``.
    8. Update ``curr_date_ymd``.

    Returns:
        Tuple ``(yr, mon, day, tod)`` where

        - ``yr``  — year (e.g. 1996)
        - ``mon`` — month (1–12)
        - ``day`` — day of month (1–31)
        - ``tod`` — time of day (seconds past 0Z, 0–86399)
    """
    global curr_date_ymd, curr_date_tod

    mcyear = start_date_ymd // 10000

    nsecs = itim * dtstep
    ndays = (nsecs + start_date_tod) // 86400

    if isleap(mcyear, calkindflag):
        nyears = ndays // 366
    else:
        nyears = ndays // 365

    # Day of current year
    if isleap(mcyear, calkindflag):
        ndays = ndays % 366
    else:
        ndays = ndays % 365

    tod = (nsecs + start_date_tod) % 86400

    # Initialise current year, month, day from start date + offsets
    mcyear = start_date_ymd // 10000 + nyears
    mcmnth = (start_date_ymd % 10000) // 100
    mcday = (start_date_ymd % 100) + ndays

    # Roll over month/year boundaries — replaces Fortran "go to 10"
    while True:
        days_per_month = _mdayleap[mcmnth - 1] if isleap(mcyear, calkindflag) else _mday[mcmnth - 1]
        if mcday > days_per_month:
            mcday -= days_per_month
            mcmnth += 1
            if mcmnth == 13:
                mcyear += 1
                mcmnth = 1
        else:
            break

    curr_date_ymd = mcyear * 10000 + mcmnth * 100 + mcday

    yr = curr_date_ymd // 10000
    mon = (curr_date_ymd % 10000) // 100
    day = curr_date_ymd % 100

    return yr, mon, day, tod


# ---------------------------------------------------------------------------
# Private: _get_prev_date
# ---------------------------------------------------------------------------


def _get_prev_date() -> tuple:
    """
    Return date components valid at the beginning of the current timestep.

    Mirrors Fortran private subroutine ``get_prev_date`` (lines 183-244).

    Identical logic to :func:`get_curr_date` but uses
    ``nsecs = (itim - 1) * dtstep`` (Fortran line 205).

    Returns:
        Tuple ``(yr, mon, day, tod)``.
    """
    mcyear = start_date_ymd // 10000

    nsecs = (itim - 1) * dtstep  # Fortran: nsecs = (itim-1)*dtstep
    ndays = (nsecs + start_date_tod) // 86400

    if isleap(mcyear, calkindflag):
        nyears = ndays // 366
    else:
        nyears = ndays // 365

    if isleap(mcyear, calkindflag):
        ndays = ndays % 366
    else:
        ndays = ndays % 365

    tod = (nsecs + start_date_tod) % 86400

    mcyear = start_date_ymd // 10000 + nyears
    mcmnth = (start_date_ymd % 10000) // 100
    mcday = (start_date_ymd % 100) + ndays

    while True:
        days_per_month = _mdayleap[mcmnth - 1] if isleap(mcyear, calkindflag) else _mday[mcmnth - 1]
        if mcday > days_per_month:
            mcday -= days_per_month
            mcmnth += 1
            if mcmnth == 13:
                mcyear += 1
                mcmnth = 1
        else:
            break

    date_ymd = mcyear * 10000 + mcmnth * 100 + mcday

    yr = date_ymd // 10000
    mon = (date_ymd % 10000) // 100
    day = date_ymd % 100

    return yr, mon, day, tod


# ---------------------------------------------------------------------------
# Public: get_curr_time
# ---------------------------------------------------------------------------


def get_curr_time() -> tuple:
    """
    Return elapsed time components at the end of the current timestep.

    Mirrors Fortran subroutine ``get_curr_time`` (lines 246-264).

    Returns:
        Tuple ``(days, seconds)`` where ``days`` is the number of whole
        elapsed days since the start date and ``seconds`` is the
        remaining partial-day seconds.
    """
    nsecs = itim * dtstep
    days = (nsecs + start_date_tod) // 86400
    seconds = (nsecs + start_date_tod) % 86400
    return days, seconds


# ---------------------------------------------------------------------------
# Private: _get_prev_calday
# ---------------------------------------------------------------------------


def _get_prev_calday() -> float:
    """
    Return calendar day at the beginning of the current timestep.

    Mirrors Fortran private function ``get_prev_calday`` (lines 296-335).

    Calendar day 1.000 = 0Z on January 1 of the current year.

    Includes the Gregorian-calendar hack for day 366/367 boundary
    (Fortran lines 320-327):

    .. code-block:: none

        if 366 < calday <= 367 and GREGORIAN: calday -= 1

    Returns:
        Floating-point calendar day in [1, 366].
    """
    yr, mon, day, tod = _get_prev_date()

    if isleap(yr, calkindflag):
        calday = float(_mdayleapcum[mon - 1]) + float(day) + float(tod) / 86400.0
    else:
        calday = float(_mdaycum[mon - 1]) + float(day) + float(tod) / 86400.0

    # Gregorian calendar hack — Fortran lines 320-327
    if 366.0 < calday <= 367.0 and calkindflag.strip() == "GREGORIAN":
        calday -= 1.0

    if calday < 1.0 or calday > 366.0:
        print(f"{iulog}: get_prev_calday error: out of bounds")
        endrun()

    return calday


# ---------------------------------------------------------------------------
# Public: get_curr_calday
# ---------------------------------------------------------------------------


def get_curr_calday(offset: int = 0) -> float:
    """
    Return the calendar day at (or relative to) the end of the current
    timestep.

    Mirrors Fortran function ``get_curr_calday`` (lines 266-294).

    Calendar day 1.000 = 0Z on 1 January of the current year.

    **Offset semantics** (Fortran lines 271-293):

    .. code-block:: none

        offset < 0 : return calday at beginning of timestep
                     (delegates to _get_prev_calday)
        offset > 0 : NOT SUPPORTED — calls endrun
        offset == 0: return calday at end of current timestep

    **Gregorian hack** (Fortran lines 284-290):

    .. code-block:: none

        if 366 < calday <= 367 and GREGORIAN: calday -= 1

    This keeps calendar days within [1, 366] for compatibility with
    ``shr_orb_decl``.

    Args:
        offset: Offset from current time in seconds.  Only ``< 0``
            (previous step) and ``== 0`` (current step) are supported.

    Returns:
        Floating-point calendar day in [1, 366].
    """
    if offset < 0:
        # Return calendar day at beginning of timestep — Fortran lines 273-275
        calday = _get_prev_calday()

    elif offset > 0:
        # Not implemented — Fortran lines 277-279
        print(f"{iulog}: get_curr_calday error: offset > 0")
        endrun()
        calday = 0.0  # unreachable; silences type checkers

    else:
        # Current timestep: end-of-step date — Fortran lines 281-293
        yr, mon, day, tod = get_curr_date()

        if isleap(yr, calkindflag):
            calday = float(_mdayleapcum[mon - 1]) + float(day) + float(tod) / 86400.0
        else:
            calday = float(_mdaycum[mon - 1]) + float(day) + float(tod) / 86400.0

        # Gregorian hack — Fortran lines 284-290
        if 366.0 < calday <= 367.0 and calkindflag.strip() == "GREGORIAN":
            calday -= 1.0

        if calday < 1.0 or calday > 366.0:
            print(f"{iulog}: get_curr_calday error: out of bounds")
            endrun()

    return calday


# ---------------------------------------------------------------------------
# Public: is_end_curr_day
# ---------------------------------------------------------------------------


def is_end_curr_day() -> bool:
    """
    Return ``True`` if the current timestep is the last in the current day.

    Mirrors Fortran function ``is_end_curr_day`` (lines 337-353).

    A timestep ends the day when ``tod == 0`` (midnight, 0Z) at the
    **end** of the step, meaning the step just crossed into the next
    calendar day.

    Returns:
        ``True`` iff ``tod == 0``.
    """
    _yr, _mon, _day, tod = get_curr_date()
    return tod == 0


# ---------------------------------------------------------------------------
# Public: is_end_curr_month
# ---------------------------------------------------------------------------


def is_end_curr_month() -> bool:
    """
    Return ``True`` if the current timestep is the last in the current month.

    Mirrors Fortran function ``is_end_curr_month`` (lines 355-371).

    A timestep ends the month when ``day == 1 and tod == 0``, i.e. the
    step has crossed into the first second of the first day of the next
    month.

    Returns:
        ``True`` iff ``day == 1 and tod == 0``.
    """
    _yr, _mon, day, tod = get_curr_date()
    return day == 1 and tod == 0

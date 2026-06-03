"""Time conversion utilities for legoESM simulations.

Provides a single source of truth for converting fractional simulation
days into day-of-year and seconds-of-day, used across AMIP scripts,
analytical forcing, and tests.

For OMIP-style runs that need to map between absolute (year, month, day)
and simulation days since a reference epoch (e.g. 1958-01-01 for the
OMIP-2 forcing window), use :func:`date_to_day` and :func:`day_to_date`.
These currently support a ``noleap`` calendar (365 days/year — the
OMIP-2 protocol convention for forcing pre-processed by dropping leap
days). A ``gregorian`` extension hook is reserved in the API.
"""

from __future__ import annotations

# Noleap (365-day) calendar tables.  Fixed module-level constants so
# helpers below stay pure arithmetic.
NOLEAP_DAYS_PER_MONTH: tuple[int, ...] = (
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
)
NOLEAP_DAYS_PER_YEAR: int = sum(NOLEAP_DAYS_PER_MONTH)  # 365
# Cumulative day-of-year at the start of each month (0-indexed):
# Jan -> 0, Feb -> 31, Mar -> 59, ..., Dec -> 334.  Length 13 so
# NOLEAP_MONTH_STARTS[12] == 365 simplifies the search in day_to_date.
NOLEAP_MONTH_STARTS: tuple[int, ...] = (
    0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
)


def day_to_calendar(day: float) -> tuple[float, float]:
    """Convert simulation day to day-of-year and seconds-of-day.

    Uses a perpetual 365-day calendar with no leap years.

    Parameters
    ----------
    day : float
        Fractional simulation day (e.g., 0.5 = noon on day 0).

    Returns
    -------
    day_of_year : float
        Day of year in [1, 366) (1-indexed, wraps every 365 days).
    seconds_of_day : float
        Seconds elapsed within the current day [0, 86400).
    """
    day_of_year = day % 365.0 + 1.0
    seconds_of_day = (day * 86400.0) % 86400.0
    return day_of_year, seconds_of_day


# ---------------------------------------------------------------------------
# Absolute-date <-> simulation-day conversions for OMIP-style runs
# ---------------------------------------------------------------------------

def is_feb_29(year: int, month: int, day_of_month: int) -> bool:
    """Return ``True`` if the given Gregorian date is February 29.

    Used for filtering leap days out of source forcing data (e.g.
    JRA55-do at Gregorian dates) when staging onto a noleap calendar
    per OMIP §2.2 protocol.

    The check is purely on the (month, day) pair — leap-year validity
    is not enforced because by construction Feb 29 only appears in the
    source data on actual leap years.
    """
    del year  # Unused; signature accepts year for caller convenience.
    return month == 2 and day_of_month == 29


def noleap_day_of_year(month: int, day_of_month: int) -> int:
    """Return the 1-based day of year on the noleap calendar.

    Parameters
    ----------
    month : int
        Calendar month, 1..12.
    day_of_month : int
        Day of month, 1..NOLEAP_DAYS_PER_MONTH[month-1].

    Returns
    -------
    int
        Day of year in [1, 365].

    Raises
    ------
    ValueError
        If ``month`` is outside [1, 12], if ``day_of_month`` is outside
        the valid range for that month on the noleap calendar, or if
        ``(month, day_of_month) == (2, 29)`` (no Feb 29 on noleap).
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in [1, 12], got {month}")
    max_day = NOLEAP_DAYS_PER_MONTH[month - 1]
    if not 1 <= day_of_month <= max_day:
        raise ValueError(
            f"day_of_month {day_of_month} out of range for month {month} "
            f"on noleap calendar (max {max_day})"
        )
    return NOLEAP_MONTH_STARTS[month - 1] + day_of_month


def date_to_day(
    year: int,
    month: int,
    day_of_month: int,
    hour: float = 0.0,
    ref_year: int = 1958,
    calendar: str = "noleap",
) -> float:
    """Map absolute date to fractional simulation day since
    ``ref_year-01-01 00:00``.

    Parameters
    ----------
    year, month, day_of_month : int
        Calendar date.
    hour : float
        UTC hour within the day, [0, 24). Default 0 (midnight).
    ref_year : int
        Reference year. ``date_to_day(ref_year, 1, 1, 0.0)`` returns 0.0.
        Default 1958 (OMIP-2 forcing-window start).
    calendar : str
        ``"noleap"`` (default) — 365 days/year. ``"gregorian"`` not yet
        implemented (raises ``NotImplementedError``).

    Returns
    -------
    float
        Fractional days since ``ref_year-01-01 00:00`` on the chosen
        calendar.

    Raises
    ------
    ValueError
        If the date is invalid on the chosen calendar (e.g. Feb 29 on
        noleap, or month/day out of range).
    NotImplementedError
        If ``calendar`` is anything other than ``"noleap"``.
    """
    if calendar != "noleap":
        raise NotImplementedError(
            f"calendar={calendar!r} not yet supported; only 'noleap'."
        )
    doy = noleap_day_of_year(month, day_of_month)  # 1-based, raises on Feb 29
    return (year - ref_year) * NOLEAP_DAYS_PER_YEAR + (doy - 1) + hour / 24.0


def day_to_date(
    day: float,
    ref_year: int = 1958,
    calendar: str = "noleap",
) -> tuple[int, int, int, float]:
    """Inverse of :func:`date_to_day`.

    Parameters
    ----------
    day : float
        Fractional simulation day since ``ref_year-01-01 00:00``.
        Must be ``>= 0`` (negative dates are not supported).
    ref_year : int
        Reference year. Default 1958.
    calendar : str
        Only ``"noleap"`` is supported.

    Returns
    -------
    (year, month, day_of_month, hour) : tuple
        Calendar date with ``hour`` in [0, 24).
    """
    if calendar != "noleap":
        raise NotImplementedError(
            f"calendar={calendar!r} not yet supported; only 'noleap'."
        )
    if day < 0.0:
        raise ValueError(f"day must be non-negative; got {day}")

    day_int = int(day)              # whole days since ref
    hour = (day - day_int) * 24.0   # fractional remainder in [0, 24)

    year = ref_year + day_int // NOLEAP_DAYS_PER_YEAR
    doy0 = day_int % NOLEAP_DAYS_PER_YEAR  # 0-based day of year, [0, 364]

    # Find month such that NOLEAP_MONTH_STARTS[m-1] <= doy0 < NOLEAP_MONTH_STARTS[m].
    # Linear scan is fine — 12 iterations max, called only at I/O boundaries.
    for m in range(1, 13):
        if doy0 < NOLEAP_MONTH_STARTS[m]:
            month = m
            day_of_month = doy0 - NOLEAP_MONTH_STARTS[m - 1] + 1
            return year, month, day_of_month, hour
    # Unreachable: doy0 < 365 by construction.
    raise AssertionError(f"day_to_date internal: doy0={doy0} not found")

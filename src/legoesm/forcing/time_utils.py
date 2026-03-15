"""Time conversion utilities for legoESM simulations.

Provides a single source of truth for converting fractional simulation
days into day-of-year and seconds-of-day, used across AMIP scripts,
analytical forcing, and tests.
"""

from __future__ import annotations


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

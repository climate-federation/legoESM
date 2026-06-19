"""Tests for ``legoesm.forcing.time_utils``.

Covers:
- Backward compatibility of the existing ``day_to_calendar`` API.
- Round-trip ``date_to_day`` <-> ``day_to_date`` on the noleap calendar.
- Leap-day rejection (Feb 29 is invalid on noleap).
- Edge cases at year and month boundaries.
- Behaviour at the OMIP-2 reference epoch (1958-01-01).
"""

from __future__ import annotations

import pytest
from legoesm.forcing.time_utils import (
    NOLEAP_DAYS_PER_MONTH,
    NOLEAP_DAYS_PER_YEAR,
    NOLEAP_MONTH_STARTS,
    daily_forcing_bucket,
    date_to_day,
    day_to_calendar,
    day_to_date,
    is_feb_29,
    noleap_day_of_year,
)

# ============================================================================
# Calendar table constants — sanity
# ============================================================================

def test_noleap_year_is_365_days():
    assert NOLEAP_DAYS_PER_YEAR == 365
    assert sum(NOLEAP_DAYS_PER_MONTH) == 365


def test_noleap_month_starts_consistent_with_days_per_month():
    cum = 0
    for m, days in enumerate(NOLEAP_DAYS_PER_MONTH):
        assert NOLEAP_MONTH_STARTS[m] == cum
        cum += days
    # Sentinel at index 12 == 365.
    assert NOLEAP_MONTH_STARTS[12] == 365


# ============================================================================
# daily_forcing_bucket — floor (not int) semantics pin
# ============================================================================

@pytest.mark.parametrize(
    "day,bucket",
    [
        (0.0, 0),
        (0.997, 0),
        (1.0, 1),
        (1.5, 1),
        # The discriminating cases: int() truncation-toward-zero would
        # return 0 for these, putting a negative fractional day (restart
        # chain crossing day 0, pre-reference epoch) in the WRONG daily
        # bucket and sampling a different seasonal SST/ozone than the
        # straight run (FIX_RESTART_TIME; the chain-consistency
        # integration tests CANNOT distinguish floor from int because
        # straight and chained runs bucket identically either way —
        # this unit pin is the only test that bites on the semantics).
        (-0.003, -1),
        (-0.5, -1),
        (-1.0, -1),
        (-1.0000001, -2),
    ],
)
def test_daily_forcing_bucket_floor_semantics(day, bucket):
    got = daily_forcing_bucket(day)
    assert got == bucket, (
        f"daily_forcing_bucket({day}) = {got}, want {bucket} "
        f"(floor, not truncation-toward-zero)"
    )
    assert isinstance(got, int)


# ============================================================================
# Backward compatibility — existing day_to_calendar must not regress
# ============================================================================

def test_day_to_calendar_unchanged_at_zero():
    doy, sod = day_to_calendar(0.0)
    assert doy == 1.0
    assert sod == 0.0


def test_day_to_calendar_unchanged_at_noon_day_zero():
    doy, sod = day_to_calendar(0.5)
    assert doy == 1.5  # original behaviour: doy = day % 365 + 1
    assert sod == 43200.0


def test_day_to_calendar_wraps_at_year_end():
    # day=365 wraps to doy=1 (start of next "year")
    doy, _ = day_to_calendar(365.0)
    assert doy == 1.0


# ============================================================================
# is_feb_29
# ============================================================================

def test_is_feb_29_detects_leap_day():
    assert is_feb_29(2016, 2, 29) is True
    assert is_feb_29(2000, 2, 29) is True


def test_is_feb_29_rejects_non_feb_29_dates():
    assert is_feb_29(2016, 2, 28) is False
    assert is_feb_29(2016, 3, 1) is False
    assert is_feb_29(2016, 2, 1) is False


def test_is_feb_29_year_argument_is_ignored():
    """Documented contract: ``is_feb_29`` checks only (month, day).

    Year is unused because by construction Feb 29 only appears in real
    Gregorian source data on actual leap years; the function is purely
    a (2, 29)-pair detector for filtering source records, not a
    leap-year validator.
    """
    assert is_feb_29(2017, 2, 29) is True   # synthetic input, but (2,29) trips
    assert is_feb_29(2016, 2, 29) is True
    assert is_feb_29(2000, 2, 29) is True


# ============================================================================
# noleap_day_of_year
# ============================================================================

def test_noleap_day_of_year_jan_1_is_1():
    assert noleap_day_of_year(1, 1) == 1


def test_noleap_day_of_year_dec_31_is_365():
    assert noleap_day_of_year(12, 31) == 365


def test_noleap_day_of_year_feb_28_is_59():
    # Jan has 31, so Feb 28 is day 31 + 28 = 59.
    assert noleap_day_of_year(2, 28) == 59


def test_noleap_day_of_year_mar_1_is_60():
    # On noleap, Mar 1 immediately follows Feb 28.
    assert noleap_day_of_year(3, 1) == 60


def test_noleap_day_of_year_rejects_feb_29():
    with pytest.raises(ValueError, match="day_of_month 29 out of range"):
        noleap_day_of_year(2, 29)


def test_noleap_day_of_year_rejects_invalid_month():
    with pytest.raises(ValueError, match="month must be in"):
        noleap_day_of_year(0, 1)
    with pytest.raises(ValueError, match="month must be in"):
        noleap_day_of_year(13, 1)


def test_noleap_day_of_year_rejects_invalid_day_of_month():
    with pytest.raises(ValueError, match="out of range"):
        noleap_day_of_year(4, 31)  # April has 30 days
    with pytest.raises(ValueError, match="out of range"):
        noleap_day_of_year(1, 0)


# ============================================================================
# date_to_day
# ============================================================================

def test_date_to_day_at_reference_epoch_is_zero():
    assert date_to_day(1958, 1, 1, ref_year=1958) == 0.0


def test_date_to_day_one_year_later_is_365():
    assert date_to_day(1959, 1, 1, ref_year=1958) == 365.0


def test_date_to_day_hour_offset():
    # 1958-01-01 12:00 == day 0.5
    assert date_to_day(1958, 1, 1, hour=12.0, ref_year=1958) == 0.5
    # 1958-01-02 06:00 == day 1.25
    assert date_to_day(1958, 1, 2, hour=6.0, ref_year=1958) == 1.25


def test_date_to_day_omip2_endpoints():
    # OMIP-2 forcing window: 1958-01-01 to 2018-12-31, 61 years.
    # On noleap (no Feb 29 dropped from this calendar), 61*365 = 22265 days.
    end = date_to_day(2018, 12, 31, ref_year=1958)
    assert end == 61 * 365 - 1  # 22264.0 (Dec 31 of year 60 since ref)


def test_date_to_day_rejects_feb_29_on_noleap():
    with pytest.raises(ValueError, match="out of range"):
        date_to_day(2016, 2, 29, ref_year=1958)


def test_date_to_day_rejects_unknown_calendar():
    with pytest.raises(NotImplementedError, match="not yet supported"):
        date_to_day(1958, 1, 1, calendar="gregorian")


# ============================================================================
# day_to_date — inverse of date_to_day
# ============================================================================

def test_day_to_date_at_zero_is_reference_epoch():
    assert day_to_date(0.0, ref_year=1958) == (1958, 1, 1, 0.0)


def test_day_to_date_at_year_boundary():
    # day 365 on noleap = next year, Jan 1.
    assert day_to_date(365.0, ref_year=1958) == (1959, 1, 1, 0.0)


def test_day_to_date_within_first_year():
    # day 30 = Jan 31; day 31 = Feb 1.
    assert day_to_date(30.0, ref_year=1958) == (1958, 1, 31, 0.0)
    assert day_to_date(31.0, ref_year=1958) == (1958, 2, 1, 0.0)


def test_day_to_date_fractional_hour():
    y, m, d, h = day_to_date(0.5, ref_year=1958)
    assert (y, m, d) == (1958, 1, 1)
    assert h == 12.0


def test_day_to_date_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        day_to_date(-0.1)


def test_day_to_date_rejects_unknown_calendar():
    with pytest.raises(NotImplementedError):
        day_to_date(0.0, calendar="gregorian")


# ============================================================================
# Round-trip date <-> day across the OMIP-2 window
# ============================================================================

@pytest.mark.parametrize("year", [1958, 1959, 1980, 2000, 2018])
@pytest.mark.parametrize(
    "month_day_hour",
    [
        (1, 1, 0.0),
        (1, 1, 12.0),
        (3, 15, 6.5),
        (7, 4, 18.0),
        (12, 31, 23.5),
        (2, 28, 0.0),    # last day of Feb on noleap
        (3, 1, 0.0),     # first day of March (immediately after Feb 28)
    ],
)
def test_round_trip_date_to_day_to_date(year, month_day_hour):
    month, day, hour = month_day_hour
    sim_day = date_to_day(year, month, day, hour=hour, ref_year=1958)
    y2, m2, d2, h2 = day_to_date(sim_day, ref_year=1958)
    assert (y2, m2, d2) == (year, month, day)
    assert h2 == pytest.approx(hour, abs=1e-9)


def test_round_trip_dense_first_year():
    """Sample every day of the first noleap year and round-trip."""
    for doy in range(365):
        y, m, d, h = day_to_date(float(doy), ref_year=1958)
        recovered = date_to_day(y, m, d, hour=h, ref_year=1958)
        assert recovered == float(doy), (
            f"day={doy} -> ({y},{m},{d},{h}) -> {recovered}"
        )


# ============================================================================
# Consistency between noleap_day_of_year and date_to_day
# ============================================================================

def test_doy_matches_date_to_day_within_year():
    # date_to_day for (year=ref, month, day) should equal doy - 1.
    for month in range(1, 13):
        for day in range(1, NOLEAP_DAYS_PER_MONTH[month - 1] + 1):
            sim_day = date_to_day(1958, month, day, ref_year=1958)
            doy = noleap_day_of_year(month, day)
            assert sim_day == float(doy - 1), (
                f"({month},{day}): sim_day={sim_day}, doy-1={doy-1}"
            )


# ============================================================================
# Forward-compat hook: gregorian path raises until implemented
# ============================================================================

def test_gregorian_calendar_not_implemented():
    """Document that Gregorian support is reserved API surface."""
    with pytest.raises(NotImplementedError):
        date_to_day(2000, 1, 1, calendar="gregorian")
    with pytest.raises(NotImplementedError):
        day_to_date(0.0, calendar="gregorian")

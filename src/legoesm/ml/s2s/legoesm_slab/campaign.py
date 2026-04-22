"""Campaign helpers for initialized pure-physics legoESM slab runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence


def generate_semimonthly_start_times(
    year: int,
    *,
    months: Sequence[int] | None = None,
    days: Sequence[int] = (1, 15),
    hour: int = 0,
) -> list[str]:
    """Return ISO timestamps for the requested semimonthly schedule."""
    selected_months = list(months) if months is not None else list(range(1, 13))
    start_times: list[str] = []
    for month in selected_months:
        for day in days:
            start = datetime(int(year), int(month), int(day), int(hour), 0, 0)
            start_times.append(start.strftime("%Y-%m-%dT%H:%M:%S"))
    return start_times


def _days_tag(days: Sequence[int]) -> str:
    return "_".join(str(int(day)) for day in days)


def case_dir_for_start(
    *,
    base_output_dir: str | Path,
    start_time: str,
    forecast_days: int,
) -> Path:
    """Return the canonical case directory for one initialization time."""
    del forecast_days
    start_stamp = datetime.fromisoformat(start_time).strftime("%Y%m%d")
    return Path(base_output_dir) / f"inference_{start_stamp}"


def campaign_dir_name(
    *,
    year: int,
    days: Sequence[int] = (1, 15),
    forecast_days: int = 42,
) -> str:
    """Return the canonical campaign directory name."""
    return f"campaign_{int(year)}_days{_days_tag(days)}_rollout{int(forecast_days)}_legoesm_slab"


def campaign_dir_for_year(
    *,
    base_output_dir: str | Path,
    year: int,
    days: Sequence[int] = (1, 15),
    forecast_days: int = 42,
) -> Path:
    """Return the output directory for one semimonthly campaign year."""
    return Path(base_output_dir) / campaign_dir_name(
        year=int(year),
        days=days,
        forecast_days=int(forecast_days),
    )


def surface_forcing_filename(*, start_time: str, forecast_days: int) -> str:
    """Return the canonical prepared forcing filename."""
    start_stamp = datetime.fromisoformat(start_time).strftime("%Y%m%d")
    return f"surface_forcing_{start_stamp}_{int(forecast_days)}d.nc"


def filter_start_times_to_year(
    start_times: Sequence[str],
    *,
    forecast_days: int,
    year: int,
) -> list[str]:
    """Keep only starts whose full forecast horizon remains inside one year."""
    year_end = datetime(int(year) + 1, 1, 1)
    filtered: list[str] = []
    for start_time in start_times:
        start_dt = datetime.fromisoformat(start_time)
        if start_dt.year != int(year):
            continue
        if start_dt.toordinal() + int(forecast_days) < year_end.toordinal():
            filtered.append(start_time)
    return filtered


__all__ = [
    "campaign_dir_for_year",
    "campaign_dir_name",
    "case_dir_for_start",
    "filter_start_times_to_year",
    "generate_semimonthly_start_times",
    "surface_forcing_filename",
]

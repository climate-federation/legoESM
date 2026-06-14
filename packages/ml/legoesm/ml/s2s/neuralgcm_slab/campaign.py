"""Campaign helpers for multi-initialization NeuralGCM slab-ocean runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence


def checkpoint_tag(checkpoint: str) -> str:
    """Return a short, filesystem-friendly checkpoint tag."""
    stem = Path(checkpoint).stem
    replacements = {
        "stochastic_1_4_deg": "stoch1p4",
        "deterministic_2_8_deg": "det2p8",
    }
    return replacements.get(stem, stem.replace("_", ""))


def generate_semimonthly_start_times(
    year: int,
    *,
    months: Sequence[int] | None = None,
    days: Sequence[int] = (1, 15),
    hour: int = 0,
) -> list[str]:
    """Return ISO timestamps for the requested monthly initialization schedule."""
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
    checkpoint: str,
    forecast_days: int,
) -> Path:
    """Return the canonical case directory for one initialization time."""
    del checkpoint, forecast_days
    start_stamp = datetime.fromisoformat(start_time).strftime("%Y%m%d")
    return Path(base_output_dir) / f"inference_{start_stamp}"


def campaign_dir_name(
    *,
    year: int,
    days: Sequence[int] = (1, 15),
    forecast_days: int = 42,
) -> str:
    return f"campaign_{int(year)}_days{_days_tag(days)}_rollout{int(forecast_days)}_fixedsst"


def campaign_dir_for_year(
    *,
    base_output_dir: str | Path,
    year: int,
    days: Sequence[int] = (1, 15),
    forecast_days: int = 42,
) -> Path:
    return Path(base_output_dir) / campaign_dir_name(
        year=int(year),
        days=days,
        forecast_days=int(forecast_days),
    )


def center_compare_dir_name(*, years: Sequence[int], forecast_days: int = 42) -> str:
    years_tag = "_".join(str(int(year)) for year in years)
    return f"campaign_center_compare_{years_tag}_rollout{int(forecast_days)}_fixedsst"


def center_compare_dir(
    *,
    base_output_dir: str | Path,
    years: Sequence[int],
    forecast_days: int = 42,
) -> Path:
    return Path(base_output_dir) / center_compare_dir_name(
        years=years,
        forecast_days=int(forecast_days),
    )


def surface_forcing_filename(*, start_time: str, forecast_days: int) -> str:
    start_stamp = datetime.fromisoformat(start_time).strftime("%Y%m%d")
    return f"surface_forcing_{start_stamp}_{int(forecast_days)}d.nc"


def filter_start_times_to_year(
    start_times: Sequence[str],
    *,
    forecast_days: int,
    year: int,
) -> list[str]:
    """Keep only starts whose full forecast horizon remains inside the given year."""
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
    "center_compare_dir",
    "center_compare_dir_name",
    "checkpoint_tag",
    "filter_start_times_to_year",
    "generate_semimonthly_start_times",
    "surface_forcing_filename",
]

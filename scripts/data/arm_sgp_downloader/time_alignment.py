"""Place native records on a common 30-minute grid labelled by interval end."""
import numpy as np
import pandas as pd

GRID = "30min"
MIN_COVERAGE = 0.8
SUMMED = ("precip",)  # an amount per record, so an interval total rather than a mean


def to_grid(table, native_seconds):
    """Average within each interval only; intervals short of MIN_COVERAGE stay missing, nothing is interpolated."""
    if native_seconds >= 1800:
        grid = table.copy()
        grid.index = grid.index.ceil(GRID)
        return grid[~grid.index.duplicated(keep="first")]
    need = MIN_COVERAGE * 1800 / native_seconds
    bins = table.index.ceil(GRID)
    values = table[[c for c in table if not c.startswith("qc_")]]
    grouped = values.groupby(bins)
    mean = grouped.mean()
    for name in SUMMED:
        if name in mean:
            mean[name] = grouped[name].sum()
    mean = mean.where(grouped.count() >= need)
    fails = table[[c for c in table if c.startswith("qc_fail_")]]
    raw = table[[c for c in table if c.startswith("qc_") and not c.startswith("qc_fail_")]]
    if not fails.empty:
        mean = mean.join(fails.groupby(bins).mean().rename(columns=lambda c: c + "_fraction"))
    if not raw.empty:
        mean = mean.join((raw.fillna(0) != 0).groupby(bins).mean().rename(columns=lambda c: c + "_fraction_nonzero"))
    return mean


def axis(tables):
    starts = [t.index.min() for t in tables if len(t)]
    stops = [t.index.max() for t in tables if len(t)]
    return pd.date_range(min(starts), max(stops), freq=GRID)


def native_seconds(table):
    if len(table) < 2:  # no step to measure, so nothing to average
        return 1800.0
    return float(np.median(np.diff(table.index.values).astype("timedelta64[s]").astype(float)))

"""CHATS 30-min tower observations loader (Zenodo 17426258).

Reads the multi-level Canopy Horizontal Array Turbulence Study (CHATS)
30-minute-averaged tower data as distributed on Zenodo as record
``17426258``.  Companion dataset to:

    Bonan, G. B., Burns, S. P., Patton, E. G. (2026):
    "Multi-layer canopy modelling of the CHATS walnut orchard".
    Agricultural and Forest Meteorology 378: 110960.

Dataset provenance / conventions come from the Zenodo README PDF
(``README_CHATS_30min_data.pdf``) that ships alongside the CSV files:

* Columns are comma-separated, no header row, 151 columns per row.
* Missing values are sentineled as ``1e+36`` (all variables).
* Timestamps in columns 0..5 (Year/Month/Day/Hour/Minute/Sec) are UTC
  and mark the *center* of the 30-minute averaging interval, i.e. the
  first row of a UTC day is ``YYYY-MM-DD 00:15:00`` and the last is
  ``YYYY-MM-DD 23:45:00``.  Column 6 is the matching decimal
  day-of-year (UTC).
* Profile heights (above ground) are taken verbatim from README section
  3; see the ``*_HEIGHTS`` module constants below.

This module intentionally has only ``pandas`` + ``numpy`` as
dependencies (no JAX): it is an offline data-prep utility for
single-column CHATS runs.  It exposes two entry points:

* :func:`load_chats_obs` -- one month CSV -> :class:`ChatsObs`.
* :func:`load_chats_obs_both_months` -- concatenate April + May 2007.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Height grids (metres above ground); values are exact per README section 3.
# ---------------------------------------------------------------------------

# 12 heights shared by Ta, RH, and sh profiles (columns 16..51).
TA_HEIGHTS: np.ndarray = np.array(
    [1.55, 3.11, 4.54, 6.07, 7.65, 9.04, 10.04, 11.12, 13.57, 17.52, 22.55, 28.51],
    dtype=np.float64,
)

# 13 heights shared by ws, wd, ustar, Qh, and (u_u, v_v, w_w) variances.
# Note the 12.54 m insertion between 11.13 m and 14.09 m (README section 3).
WIND_HEIGHTS: np.ndarray = np.array(
    [
        1.54, 3.11, 4.54, 6.07, 7.62, 9.02, 10.03, 11.13,
        12.54, 14.09, 18.04, 23.07, 29.03,
    ],
    dtype=np.float64,
)

# 6 heights that carry latent-heat-flux (Qe) measurements.
QE_HEIGHTS: np.ndarray = np.array(
    [1.54, 4.54, 7.62, 10.03, 14.09, 23.07],
    dtype=np.float64,
)

# Missing-value sentinel used across every variable in the CSVs.
_MISSING_SENTINEL: float = 1e36

# Fixed 151-column layout (0-indexed, half-open slices).
_N_COLS: int = 151
_SLICE_TIME = slice(0, 6)          # Year, Month, Day, Hour, Minute, Sec
_COL_DOY = 6
_SLICE_RAD = slice(7, 15)          # 8 radiation columns
_COL_P_1M = 15
_SLICE_TA = slice(16, 28)          # 12 heights
_SLICE_RH = slice(28, 40)          # 12 heights
_SLICE_SH = slice(40, 52)          # 12 heights
_SLICE_WS = slice(52, 65)          # 13 heights
_SLICE_WD = slice(65, 78)          # 13 heights
_SLICE_USTAR = slice(78, 91)       # 13 heights
_SLICE_QH = slice(91, 104)         # 13 heights
_SLICE_QE = slice(104, 110)        # 6 heights
_COL_GSOIL_CORR = 110
_COL_GSOIL_RAW = 111
# 112..150 (39 cols) hold u_u, v_v, w_w variances at 13 heights each; not
# exposed by the public API yet -- kept as raw columns so downstream code
# can trivially pull them via `df.iloc[:, 112:151]` if needed later.


@dataclass
class ChatsObs:
    """Container of CHATS 30-min observations for a contiguous period.

    All arrays share a leading time axis of length ``n_time``.  Profile
    arrays have shape ``(n_time, n_heights)`` where ``n_heights`` matches
    the corresponding ``*_heights`` vector.  Missing samples are
    ``np.nan`` (already unmasked from the ``1e+36`` on-disk sentinel).
    """

    time: pd.DatetimeIndex          # UTC, center of 30-min interval
    doy: np.ndarray                 # decimal DOY (UTC), 1-based per README

    # Above-canopy scalar radiation + pressure
    Rsw_in_16m: np.ndarray          # W/m^2
    Rsw_out_16m: np.ndarray         # W/m^2
    Rlw_in_16m: np.ndarray          # W/m^2
    Rlw_out_16m: np.ndarray         # W/m^2
    Rsw_in_2m: np.ndarray           # W/m^2
    Rsw_out_2m: np.ndarray          # W/m^2
    Rlw_in_2m: np.ndarray           # W/m^2
    Rlw_out_2m: np.ndarray          # W/m^2
    P_1m: np.ndarray                # barometric pressure (Pa)

    # Ta / RH / sh profiles on the 12-height grid
    Ta_heights: np.ndarray          # (12,) metres AGL
    Ta: np.ndarray                  # (n_time, 12), K
    RH: np.ndarray                  # (n_time, 12), %
    sh: np.ndarray                  # (n_time, 12), g/kg

    # Wind + turbulence profiles on the 13-height grid
    wind_heights: np.ndarray        # (13,) metres AGL
    ws: np.ndarray                  # (n_time, 13), m/s
    wd: np.ndarray                  # (n_time, 13), deg
    ustar: np.ndarray               # (n_time, 13), m/s
    Qh: np.ndarray                  # (n_time, 13), W/m^2

    # Latent-heat-flux subset (6 heights)
    Qe_heights: np.ndarray          # (6,) metres AGL
    Qe: np.ndarray                  # (n_time, 6), W/m^2

    # Soil heat flux
    Gsoil_corr: np.ndarray          # W/m^2, gap/storage-corrected
    Gsoil_raw: np.ndarray           # W/m^2, raw plate reading


def _read_csv_no_header(csv_path: Path) -> pd.DataFrame:
    """Read a single CHATS month CSV, masking the missing sentinel."""
    df = pd.read_csv(
        csv_path,
        header=None,
        na_values=[_MISSING_SENTINEL],
        # A comfortable equality tolerance for the sentinel: CSV strings
        # round-trip to exactly 1e+36 as a double, so equality matching
        # via `na_values` is safe here.
    )
    if df.shape[1] != _N_COLS:
        raise ValueError(
            f"CHATS CSV {csv_path} has {df.shape[1]} columns, expected {_N_COLS}"
        )
    return df


def _time_index_from_ymdhms(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Assemble a UTC ``DatetimeIndex`` from the Y/M/D/h/m/s columns."""
    parts = df.iloc[:, _SLICE_TIME].astype(int)
    parts.columns = ["year", "month", "day", "hour", "minute", "second"]
    # ``pd.to_datetime`` on a DataFrame returns a ``Series``; wrap it as an
    # explicit ``DatetimeIndex`` so downstream positional indexing and
    # ``.append`` behave as advertised in :class:`ChatsObs`.
    return pd.DatetimeIndex(pd.to_datetime(parts, utc=True))


def _profile(df: pd.DataFrame, cols: slice) -> np.ndarray:
    """Extract a 2-D profile block, returning a contiguous float64 array."""
    return np.ascontiguousarray(df.iloc[:, cols].to_numpy(dtype=np.float64))


def _scalar(df: pd.DataFrame, col: int) -> np.ndarray:
    """Extract a single column as a 1-D float64 array."""
    return np.ascontiguousarray(df.iloc[:, col].to_numpy(dtype=np.float64))


def _obs_from_dataframe(df: pd.DataFrame, time: pd.DatetimeIndex) -> ChatsObs:
    """Slice a raw 151-column DataFrame into a :class:`ChatsObs`."""
    rad = _profile(df, _SLICE_RAD)  # (n_time, 8)
    return ChatsObs(
        time=time,
        doy=_scalar(df, _COL_DOY),
        Rsw_in_16m=rad[:, 0],
        Rsw_out_16m=rad[:, 1],
        Rlw_in_16m=rad[:, 2],
        Rlw_out_16m=rad[:, 3],
        Rsw_in_2m=rad[:, 4],
        Rsw_out_2m=rad[:, 5],
        Rlw_in_2m=rad[:, 6],
        Rlw_out_2m=rad[:, 7],
        P_1m=_scalar(df, _COL_P_1M),
        Ta_heights=TA_HEIGHTS.copy(),
        Ta=_profile(df, _SLICE_TA),
        RH=_profile(df, _SLICE_RH),
        sh=_profile(df, _SLICE_SH),
        wind_heights=WIND_HEIGHTS.copy(),
        ws=_profile(df, _SLICE_WS),
        wd=_profile(df, _SLICE_WD),
        ustar=_profile(df, _SLICE_USTAR),
        Qh=_profile(df, _SLICE_QH),
        Qe_heights=QE_HEIGHTS.copy(),
        Qe=_profile(df, _SLICE_QE),
        Gsoil_corr=_scalar(df, _COL_GSOIL_CORR),
        Gsoil_raw=_scalar(df, _COL_GSOIL_RAW),
    )


def load_chats_obs(csv_path: Path) -> ChatsObs:
    """Load one month of CHATS 30-min observations from ``csv_path``.

    Parameters
    ----------
    csv_path
        Path to a monthly CSV shipped in Zenodo record 17426258, e.g.
        ``chats_30min_data_2007_05_ver_250801.csv``.

    Returns
    -------
    ChatsObs
        Time-ordered container; ``1e+36`` sentinels are masked to
        ``np.nan``.
    """
    df = _read_csv_no_header(Path(csv_path))
    time = _time_index_from_ymdhms(df)
    return _obs_from_dataframe(df, time)


def _concat_along_time(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concatenate two arrays along the leading (time) axis."""
    return np.concatenate([a, b], axis=0)


def load_chats_obs_both_months(dir_path: Path) -> ChatsObs:
    """Load April + May 2007 CHATS CSVs from ``dir_path`` and concatenate.

    The two files are read in calendar order and stacked along the time
    axis; height grids (which are identical between months) are carried
    through unchanged.  Useful for spinup + analysis runs that need the
    full published period as one continuous record.
    """
    dir_path = Path(dir_path)
    apr = load_chats_obs(dir_path / "chats_30min_data_2007_04_ver_250801.csv")
    may = load_chats_obs(dir_path / "chats_30min_data_2007_05_ver_250801.csv")
    return ChatsObs(
        time=apr.time.append(may.time),
        doy=_concat_along_time(apr.doy, may.doy),
        Rsw_in_16m=_concat_along_time(apr.Rsw_in_16m, may.Rsw_in_16m),
        Rsw_out_16m=_concat_along_time(apr.Rsw_out_16m, may.Rsw_out_16m),
        Rlw_in_16m=_concat_along_time(apr.Rlw_in_16m, may.Rlw_in_16m),
        Rlw_out_16m=_concat_along_time(apr.Rlw_out_16m, may.Rlw_out_16m),
        Rsw_in_2m=_concat_along_time(apr.Rsw_in_2m, may.Rsw_in_2m),
        Rsw_out_2m=_concat_along_time(apr.Rsw_out_2m, may.Rsw_out_2m),
        Rlw_in_2m=_concat_along_time(apr.Rlw_in_2m, may.Rlw_in_2m),
        Rlw_out_2m=_concat_along_time(apr.Rlw_out_2m, may.Rlw_out_2m),
        P_1m=_concat_along_time(apr.P_1m, may.P_1m),
        Ta_heights=apr.Ta_heights.copy(),
        Ta=_concat_along_time(apr.Ta, may.Ta),
        RH=_concat_along_time(apr.RH, may.RH),
        sh=_concat_along_time(apr.sh, may.sh),
        wind_heights=apr.wind_heights.copy(),
        ws=_concat_along_time(apr.ws, may.ws),
        wd=_concat_along_time(apr.wd, may.wd),
        ustar=_concat_along_time(apr.ustar, may.ustar),
        Qh=_concat_along_time(apr.Qh, may.Qh),
        Qe_heights=apr.Qe_heights.copy(),
        Qe=_concat_along_time(apr.Qe, may.Qe),
        Gsoil_corr=_concat_along_time(apr.Gsoil_corr, may.Gsoil_corr),
        Gsoil_raw=_concat_along_time(apr.Gsoil_raw, may.Gsoil_raw),
    )


__all__ = [
    "ChatsObs",
    "TA_HEIGHTS",
    "WIND_HEIGHTS",
    "QE_HEIGHTS",
    "load_chats_obs",
    "load_chats_obs_both_months",
]

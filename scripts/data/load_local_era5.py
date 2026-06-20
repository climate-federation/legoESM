"""Open LOCAL NCAR-RDA ERA5 (ll025 NetCDF) as one dataset for ``load_era5_slice(ds=)``.

The compare-reanalysis pipeline normally loads ERA5 from a WeatherBench Zarr over the
network (``gs://``).  On an HPC filesystem that mirrors the NCAR-RDA ERA5 collection
``d633006`` but has NO outbound network/gcsfs, this adapter opens the per-variable
NetCDF files — pressure-level T/U/V/Q and single-level surface-pressure + SST, all on
the regular 0.25° lat-lon (``ll025``) grid — and MERGES them into ONE
``xarray.Dataset`` matching the loader's contract (lat/lon dims, ``resolve_var``-findable
variable names).  Feed it through the SAME extraction → regrid → interp chain OFFLINE::

    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice
    cfg = TrainingERA5Config(surface_variables=("surface_pressure", "skin_temperature"))
    ds = open_local_era5_dataset("/glade/derecho/scratch/ashford/ERA5", "20170901")
    slice0 = load_era5_slice(cfg, time_idx=0, ds=ds)      # REAL ERA5, no Zarr/network

Archive layout (``e5.oper.an.{pl,sfc}.<grib_code>.ll025{sc,uv}.<start>_<end>.nc``):
pressure-level files are DAILY chunks (``YYYYMMDD00_YYYYMMDD23``), single-level files are
MONTHLY chunks (``YYYYMM0100_YYYYMM<last>23``); the monthly ``sfc`` times are aligned to
the daily ``pl`` times.  Verified iter 408/409 against d633006 Sept-2017 (a physically
sane slice: T∈[183.8, 316.9] K, p_s∈[498, 1038] hPa).  This is a MIMICRY adapter for one
archive's file convention — it lives in ``scripts/data/`` (a harness), not the model.
"""

from __future__ import annotations

import glob
from typing import Any

# RDA file metadata variables that are NOT the geophysical field (skip when picking it).
_META_VARS = frozenset(
    {"utc_date", "weight", "zero", "a_half", "a_model", "b_half", "b_model"})

# (GRIB param code in the RDA filename → the short name ``resolve_var`` finds): the
# pressure-level prognostics + the two single-level surface fields the compare needs.
_PL_VARS: tuple[tuple[str, str], ...] = (
    ("128_130_t", "t"), ("128_131_u", "u"),
    ("128_132_v", "v"), ("128_133_q", "q"),
)
_SFC_VARS: tuple[tuple[str, str], ...] = (
    ("128_134_sp", "sp"),       # surface_pressure
    ("128_034_sstk", "skt"),    # skin_temperature (SST proxy)
)


def _open_one(path: str, short: str) -> Any:
    """Open one RDA NetCDF, rename its single geophysical variable to ``short``."""
    import xarray as xr

    ds = xr.open_dataset(path)
    real = [v for v in ds.data_vars if str(v).lower() not in _META_VARS]
    if len(real) != 1:
        raise ValueError(
            f"open_local_era5_dataset: expected exactly ONE geophysical variable in "
            f"{path}, found {list(ds.data_vars)} (after dropping metadata {sorted(_META_VARS)}).")
    return ds.rename({real[0]: short})[[short]]


def _find(data_dir: str, kind: str, code: str, date: str, *, monthly: bool) -> str:
    """Glob the single RDA ll025 file for ``kind``/``code`` covering ``date``."""
    chunk = (date[:6] + "0100") if monthly else (date + "00")
    pattern = f"{data_dir}/e5.oper.an.{kind}.{code}.ll025*.{chunk}_*.nc"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"open_local_era5_dataset: no ERA5 file matching {pattern!r} — check the "
            f"data_dir, the date (YYYYMMDD), and that the NCAR-RDA ll025 {kind} {code} "
            "chunk exists locally.")
    if len(matches) > 1:
        # One chunk per (kind, code, month/day) is the d633006 convention; an AMBIGUOUS
        # match (e.g. a partial re-download alongside the original) would otherwise
        # silently pick the alphabetically-first file → a wrong/partial reference
        # (codex-review iter 414).  Fail loud instead.
        raise ValueError(
            f"open_local_era5_dataset: {len(matches)} files match {pattern!r} "
            f"(expected exactly one chunk): {matches}. Remove the duplicate/partial.")
    return matches[0]


def open_local_era5_dataset(data_dir: str, date: str) -> Any:
    """Merge the local NCAR-RDA ll025 ERA5 for ``date`` (``YYYYMMDD``) into one dataset.

    Returns an ``xarray.Dataset`` with ``t``/``u``/``v``/``q`` (level, latitude,
    longitude) + ``sp``/``skt`` (latitude, longitude) over the day's 24 hourly times —
    ready for ``load_era5_slice(config, time_idx, ds=...)``.  Raises ``FileNotFoundError``
    (a missing required chunk) or ``ValueError`` (a malformed file) rather than silently
    dropping a field.
    """
    import xarray as xr

    pl = xr.merge(
        [_open_one(_find(data_dir, "pl", c, date, monthly=False), s) for c, s in _PL_VARS],
        compat="override")
    sfc = xr.merge(
        [_open_one(_find(data_dir, "sfc", c, date, monthly=True), s) for c, s in _SFC_VARS],
        compat="override")
    # The single-level files are MONTHLY chunks; align them to the day's pl times so the
    # merged dataset has one consistent time axis (load_era5_slice isel's into it).
    sfc = sfc.sel(time=pl.time)
    # join="override" forces pl's lat/lon labels onto sfc — SAFE only because both come
    # from the SAME ll025 archive (identical grid); it would mis-LABEL (not re-grid) sfc
    # if the two ever used different lat/lon grids (codex-review iter 414).
    return xr.merge([pl, sfc], compat="override", join="override")

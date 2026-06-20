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


#: GRIB param codes for the AMIP lower-boundary fields in the d633006 sfc archive.
_AMIP_SST_CODE = "128_034_sstk"   # SST [K]
_AMIP_SIC_CODE = "128_031_ci"     # sea-ice concentration [fraction, 0-1]


def build_era5_amip_forcing(
    data_dir: str, date: str, out_path: str, *, hour_stride: int = 24,
    **overrides: Any,
) -> Any:
    """Build a memory-safe subsampled AMIP forcing file from LOCAL ERA5 + return its config.

    The FORCING-side twin of :func:`open_local_era5_dataset`, so a fully-OFFLINE realistic
    AMIP run (``ModelDriver``, ``dataset="custom"``) can use REAL ERA5 lower boundary
    conditions with no network.  The single-level ``sstk``/``ci`` chunks are MONTHLY and
    HOURLY (~720 steps × 721×1440 ≈ 6 GB/field) — feeding the raw file to
    ``load_amip_forcing`` OOMs (verified iter 419) — and ERA5 SST/SIC vary negligibly within
    a day, so this subsamples to every ``hour_stride``-th step (00Z daily by default),
    reading ONLY those strided slices (memory-safe), and writes ONE combined NetCDF
    (``SSTK`` [K] + ``CI`` [0-1 fraction], with ``latitude``/``longitude``/``time``) to
    ``out_path``.  ``out_path`` is a runtime artifact (gitignored), NOT committed.

    ``date`` is ``YYYYMMDD`` (selects that month's chunk).  Returns a ``dataset="custom"``
    :class:`AMIPForcingConfig` (Kelvin SST ⇒ ``sst_offset=0``; fraction SIC ⇒
    ``sic_scale=1``; SIC read from the same combined file); ``overrides`` pass through (e.g.
    ``T_ice=...``).  ``load_amip_forcing``'s units guard cross-checks the K/fraction
    conventions, so a wrong offset/scale still fails loud.  Verified iter 419 against
    d633006 Sept-2017.
    """
    import xarray as xr
    from legoesm.forcing.amip import AMIPForcingConfig

    if hour_stride < 1:
        raise ValueError(
            f"build_era5_amip_forcing: hour_stride must be >= 1, got {hour_stride}.")
    # Both _find calls FIRST so a missing chunk fails loud before any open/write.
    sst_file = _find(data_dir, "sfc", _AMIP_SST_CODE, date, monthly=True)
    sic_file = _find(data_dir, "sfc", _AMIP_SIC_CODE, date, monthly=True)
    sel = slice(0, None, hour_stride)
    # .isel(time=sel).load() reads ONLY the strided slices for the NCAR-RDA ll025 layout
    # (time stored contiguously, not HDF5-chunked) — the memory-safe step that avoids the
    # ~6 GB/field OOM.  (A file HDF5-chunked along time could still read full chunks per
    # selected index.)  Context-managed so the HDF5 file descriptors close after load.
    with xr.open_dataset(sst_file) as ds_sst:
        sst = ds_sst["SSTK"].isel(time=sel).load()
    with xr.open_dataset(sic_file) as ds_sic:
        sic = ds_sic["CI"].isel(time=sel).load()
    if int(sst.sizes["time"]) < 2:
        raise ValueError(
            f"build_era5_amip_forcing: hour_stride={hour_stride} leaves "
            f"{int(sst.sizes['time'])} time step(s); the time-interpolated AMIP forcing "
            "needs >= 2. Use a smaller stride.")
    # Same archive ⇒ the SPATIAL grids must match EXACTLY (a silent mis-LABEL, not a regrid,
    # if they ever differed — same-size-but-different-values would slip past a shape check),
    # so verify lat/lon by value and fail loud.  Time is force-aligned: the two files share
    # the same hourly steps, and an encoding nicety should not block the build.
    for axis in ("latitude", "longitude"):
        if not sst[axis].equals(sic[axis]):
            raise ValueError(
                f"build_era5_amip_forcing: SST and CI {axis} grids differ — both must be "
                "the same d633006 ll025 chunk (this would silently mis-label, not regrid).")
    sic = sic.assign_coords(time=sst["time"])
    xr.Dataset({"SSTK": sst, "CI": sic}).to_netcdf(out_path)

    base = dict(
        dataset="custom", path=str(out_path),
        sst_var="SSTK", sic_var="CI",          # SIC lives in the same combined file
        time_var="time", lat_var="latitude", lon_var="longitude",
        sst_offset=0.0,     # SSTK is already in Kelvin
        sic_scale=1.0,      # ERA5 CI is already a [0, 1] fraction
    )
    base.update(overrides)
    return AMIPForcingConfig(**base)


def apply_amip_forcing_to_config(experiment_config: Any, forcing_config: Any) -> Any:
    """Inject an ``AMIPForcingConfig``'s fields into an ``ExperimentConfig`` so a custom
    forcing (e.g. from :func:`build_era5_amip_forcing`) drives the AMIP ``ModelDriver`` run.

    Returns a NEW config (``ExperimentConfig._replace``) — the two structs carry the same
    forcing field names except ``AMIPForcingConfig.path`` → ``ExperimentConfig.forcing_path``.
    This is the field map that turns "I built an ERA5 forcing file" into "the campaign runs
    AMIP with it", with no error-prone hand-editing of ``sst_var``/``sst_offset``/… in the
    config JSON.  The downstream ``load_amip_forcing`` units guard still cross-checks the
    K/fraction conventions, so a wrong offset/scale fails loud.  Copies EVERY forcing field
    the two structs share — including the surface BCs ``T_ice``/``albedo_ice``/
    ``albedo_ocean`` — so a ``build_era5_amip_forcing(..., T_ice=…)`` override is NOT
    silently dropped (codex-review iter 421).
    """
    return experiment_config._replace(
        dataset=forcing_config.dataset,
        forcing_path=forcing_config.path,
        sic_path=forcing_config.sic_path,
        sst_var=forcing_config.sst_var,
        sic_var=forcing_config.sic_var,
        time_var=forcing_config.time_var,
        lat_var=forcing_config.lat_var,
        lon_var=forcing_config.lon_var,
        sst_offset=forcing_config.sst_offset,
        sic_scale=forcing_config.sic_scale,
        T_ice=forcing_config.T_ice,
        albedo_ice=forcing_config.albedo_ice,
        albedo_ocean=forcing_config.albedo_ocean,
    )


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

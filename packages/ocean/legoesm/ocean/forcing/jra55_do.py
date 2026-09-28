"""JRA55-do reanalysis loader for OMIP-2-style forced ocean runs.

Reference
---------
Tsujino, H., et al. (2020). "JRA-55 based surface dataset for driving
ocean-sea-ice models (JRA55-do)", Ocean Modelling 130, 79-139.

Tsujino, H., et al. (2020). "Evaluation of global ocean-sea-ice
model simulations based on the experimental protocols of the Ocean
Model Intercomparison Project phase 2 (OMIP-2)", Geosci. Model Dev.
13, 3643-3708.

Spec
----
JRA55-do v1.5 provides 3-hourly atmospheric forcing on a 0.5625-deg
grid from 1958 to present. The seven OMIP-2 forcing channels carried
by this loader are:

* ``u10``, ``v10``      [m/s]       3-hourly 10-m wind components
* ``T_air``             [K]         3-hourly 10-m air temperature
* ``q_air``             [kg/kg]     3-hourly 10-m specific humidity
* ``sw_down``           [W/m^2]     3-hourly downward shortwave
* ``lw_down``           [W/m^2]     3-hourly downward longwave
* ``precip``            [kg/m^2/s]  3-hourly precipitation flux
* ``runoff``            [kg/m^2/s]  daily continental runoff flux

The loader caches each calendar year as a zarr store under
``$LEGOESM_CACHE/forcing/jra55_do/<year>.zarr`` (resolved via
``legoesm.ocean.fidelity.cache.sub("forcing")`` to match the existing
pattern). When the cache is missing the loader raises; smoke runs opt in
to a deterministic synthetic climatology with ``allow_synthetic=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure

logger = logging.getLogger(__name__)


# JRA55-do native grid: 0.5625 deg ~ 640 lon x 320 lat.
JRA55_NLON: int = 640
JRA55_NLAT: int = 320
JRA55_FREQ_HOURS: int = 3       # 3-hourly atmospheric channels


class OceanForcing(NamedTuple):
    """Seven-channel ocean atmospheric forcing on a regular lat-lon grid.

    Arrays have shape ``(n_time, n_lat, n_lon)`` with units documented
    on each field.  ``snow``/``slp`` are OPTIONAL NEMO-parity channels
    (solid precipitation for the q_ns snow-fusion/heat-content terms;
    sea-level pressure for moist-air density + Goff saturation humidity).
    ``None`` on forcing sets built before the 2026-06 schema extension —
    consumers fall back to snow=0 / slp=standard-atmosphere.
    """
    lon: np.ndarray            # (n_lon,) deg E in [0, 360)
    lat: np.ndarray            # (n_lat,) deg N in [-90, 90]
    time_s: np.ndarray         # (n_time,) seconds since the year start
    u10: np.ndarray            # m/s
    v10: np.ndarray            # m/s
    T_air: np.ndarray          # K
    q_air: np.ndarray          # kg/kg
    sw_down: np.ndarray        # W/m^2
    lw_down: np.ndarray        # W/m^2
    precip: np.ndarray         # kg/m^2/s (TOTAL = rain + snow)
    runoff: np.ndarray         # kg/m^2/s
    snow: Optional[np.ndarray] = None   # kg/m^2/s solid precipitation
    slp: Optional[np.ndarray] = None    # Pa sea-level pressure


def _cache_dir() -> Path:
    from legoesm.ocean.fidelity import cache as _cache
    return _cache.sub("forcing") / "jra55_do"


def synthetic_ocean_forcing(year: int, *,
                             n_time: int = 12,
                             nlon: int = JRA55_NLON,
                             nlat: int = JRA55_NLAT) -> OceanForcing:
    """Deterministic monthly climatology with physically-reasonable bounds.

    Used as the fallback when the on-disk JRA55-do cache is missing.
    Channel ranges match the OMIP-2 evaluation paper's diagnostics:

    * |u10|, |v10| -- 10 m/s zonal mean, sin(2 lat) meridional belt
    * T_air        -- 263 K poles, 300 K equator, +/- 5 K seasonal cycle
    * q_air        -- saturation at T_air * 80 % RH
    * sw_down      -- 200 cos(lat) * (1 + sin(2 pi t)) W/m^2
    * lw_down      -- 300 W/m^2 globally uniform
    * precip       -- 3e-5 kg/m^2/s tropical band, zero in deserts
    * runoff       -- zero (synthetic mode, no continental mask)
    """
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    lat = np.linspace(-90.0, 90.0, nlat, dtype=np.float64)
    time_s = np.linspace(
        0.0, 365.0 * 86400.0, n_time, endpoint=False, dtype=np.float64,
    )

    LAT_R = np.radians(lat)[None, :, None]
    LON_R = np.radians(lon)[None, None, :]
    cos_lat = np.cos(LAT_R)
    sin_2lat = np.sin(2.0 * LAT_R)
    t_year = (time_s / (365.0 * 86400.0))[:, None, None]
    seasonal = np.sin(2.0 * np.pi * t_year)

    out_shape = (n_time, nlat, nlon)
    u10 = np.broadcast_to(10.0 * sin_2lat * np.ones_like(LON_R), out_shape).copy()
    v10 = np.broadcast_to(2.0 * np.sin(LAT_R) * np.cos(LON_R), out_shape).copy()
    T_air = np.broadcast_to(
        283.0 + 17.0 * cos_lat - 5.0 * seasonal + 0.0 * LON_R,
        out_shape,
    ).copy()
    # Canonical saturation vapour pressure (legoesm.thermo; no re-derived
    # Magnus/Tetens coefficients).  Specific-humidity conversion uses
    # constants.epsilon: q = eps e / (p - (1 - eps) e).
    e_s = np.asarray(saturation_vapor_pressure(jnp.asarray(T_air))) / 100.0  # hPa
    p_sfc = 1013.25                                      # hPa
    q_sat = constants.epsilon * e_s / (p_sfc - (1.0 - constants.epsilon) * e_s)
    q_air = 0.8 * q_sat                                  # 80 % RH

    sw_down = np.maximum(
        200.0 * cos_lat * (1.0 + 0.5 * seasonal) + 0.0 * LON_R, 0.0,
    )
    sw_down = np.broadcast_to(sw_down, out_shape).copy()
    lw_down = np.full(out_shape, 300.0, dtype=np.float64)
    precip = np.broadcast_to(
        3e-5 * np.exp(-(np.degrees(LAT_R) / 10.0) ** 2) + 0.0 * LON_R,
        out_shape,
    ).copy()
    runoff = np.zeros_like(precip)

    return OceanForcing(
        lon=lon, lat=lat, time_s=time_s,
        u10=u10.astype(np.float64),
        v10=v10.astype(np.float64),
        T_air=T_air.astype(np.float64),
        q_air=q_air.astype(np.float64),
        sw_down=sw_down.astype(np.float64),
        lw_down=lw_down.astype(np.float64),
        precip=precip.astype(np.float64),
        runoff=runoff.astype(np.float64),
    )


# Plausible near-surface air temperature in Kelvin; a Celsius field (~-70..50)
# or a corrupt one falls outside.  Trust boundary of the forcing cache.
_TAS_KELVIN_MIN_K = 150.0
_TAS_KELVIN_MAX_K = 350.0


def _load_from_builder_cache(store: Path, year: int,
                             cycle_years: bool) -> OceanForcing:
    """Slice one noleap year out of the multi-year CMOR-named cache written
    by ``legoesm.forcing.jra55_do.build_jra55_cache``.

    ``cycle_years=True`` maps ``year`` onto the cache's own window
    ``year_start + (year - year_start) % n_years`` (the OMIP-2 protocol repeats
    its forcing cycle); otherwise a year outside the window raises
    ``FileNotFoundError``.  The store's layout (calendar, cadence, record
    count, coordinates) and every returned field are checked; a mismatch
    raises ``ValueError`` rather than feeding time-shifted or non-finite
    forcing to the model.
    """
    import xarray as xr
    from legoesm.forcing.jra55_do import RECORDS_PER_DAY
    from legoesm.forcing.time_utils import NOLEAP_DAYS_PER_YEAR

    ds = xr.open_zarr(store)
    a = ds.attrs
    missing = [k for k in ("year_start", "year_end", "ref_year") if k not in a]
    if missing:
        raise ValueError(f"{store}: malformed JRA55-do cache: missing attrs {missing}")
    y0, y1 = int(a["year_start"]), int(a["year_end"])
    per_year = NOLEAP_DAYS_PER_YEAR * RECORDS_PER_DAY
    n_time = ds.sizes["time"]
    problems = [
        msg for bad, msg in (
            (a.get("calendar") != "noleap", f"calendar={a.get('calendar')!r}, expected 'noleap'"),
            (int(a.get("records_per_day", -1)) != RECORDS_PER_DAY,
             f"records_per_day={a.get('records_per_day')}, expected {RECORDS_PER_DAY}"),
            (int(a["ref_year"]) != y0, f"ref_year={a['ref_year']} != year_start={y0}"),
            (not (int(a.get("n_records", -1)) == n_time == (y1 - y0 + 1) * per_year),
             f"n_records={a.get('n_records')}, time axis {n_time}, expected "
             f"{(y1 - y0 + 1) * per_year} for {y0}-{y1}"),
            (not np.array_equal(ds["time"].values, np.arange(n_time)),
             "time axis is not the consecutive record index 0..n-1"),
        ) if bad
    ]
    if problems:
        raise ValueError(f"{store}: malformed JRA55-do cache: " + "; ".join(problems))
    requested = year
    if cycle_years:
        year = y0 + (year - y0) % (y1 - y0 + 1)
    elif not y0 <= year <= y1:
        raise FileNotFoundError(
            f"JRA55-do year {year} is outside the cached window {y0}-{y1} of {store}")
    start = (year - y0) * per_year
    ds = ds.isel(time=slice(start, start + per_year))

    def f(name):
        if name in ("lat", "lon"):
            dims = (name,)
        elif tuple(ds[name].dims) != ("time", "lat", "lon"):
            raise ValueError(
                f"{store}: {name!r} has dims {tuple(ds[name].dims)}, expected "
                f"('time', 'lat', 'lon') -- a transposed field would be read as "
                f"(time, lat, lon) without error")
        arr = np.asarray(ds[name].values, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError(f"{store}: non-finite values in {name!r} for year {year}")
        return arr

    lat, lon = f("lat"), f("lon")
    if lat.min() < -90.0 or lat.max() > 90.0 or lon.min() < 0.0 or lon.max() >= 360.0:
        raise ValueError(f"{store}: lat/lon not in degrees [-90, 90] x [0, 360)")
    logger.info("JRA55-do: requested year %d -> forcing year %d read from %s",
                requested, year, store)
    prra, prsn = f("prra"), f("prsn")
    tas = f("tas")
    if not (_TAS_KELVIN_MIN_K < tas.min() and tas.max() < _TAS_KELVIN_MAX_K):
        raise ValueError(
            f"{store}: tas spans {tas.min():.1f}..{tas.max():.1f}, outside "
            f"{_TAS_KELVIN_MIN_K}..{_TAS_KELVIN_MAX_K} K -- Celsius or corrupt")
    return OceanForcing(
        lon=lon, lat=lat,
        time_s=np.arange(per_year, dtype=np.float64)
        * (86400.0 / RECORDS_PER_DAY),
        u10=f("uas"), v10=f("vas"), T_air=tas, q_air=f("huss"),
        sw_down=f("rsds"), lw_down=f("rlds"),
        precip=prra + prsn, runoff=f("friver"),
        snow=prsn, slp=f("psl"),
    )


def load_jra55_do(year: int, *, cache_dir: Optional[Path] = None,
                  allow_synthetic: bool = False,
                  cycle_years: bool = False) -> OceanForcing:
    """Load one calendar year of JRA55-do forcing.

    Looks for ``<cache_dir>/<year>.zarr`` first, then for the multi-year
    cache written by ``scripts/data/prepare_omip_forcing.py`` (``cache_dir``
    may name that store directly or the directory holding it under its
    default filename). Raises ``FileNotFoundError`` when neither holds the
    year, unless ``allow_synthetic=True`` (smoke runs only), which returns
    :func:`synthetic_ocean_forcing` with a warning.  ``cycle_years=True``
    (multi-year cache only) wraps ``year`` into the cache's own year window,
    as the OMIP-2 protocol repeats its forcing cycle.
    """
    root = Path(cache_dir) if cache_dir is not None else _cache_dir()
    zarr_path = root / f"{year}.zarr"
    if not zarr_path.exists():
        from legoesm.forcing.jra55_do import JRA55DoConfig
        store = root if root.suffix == ".zarr" else (
            root / JRA55DoConfig.cache_filename)
        if store.exists():
            try:
                return _load_from_builder_cache(store, year, cycle_years)
            except FileNotFoundError:
                if not allow_synthetic:
                    raise
    if zarr_path.exists():
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "JRA55-do real-data load requires xarray + zarr; install "
                "with ``pip install xarray zarr``"
            ) from exc
        ds = xr.open_zarr(zarr_path)
        return OceanForcing(
            lon=np.asarray(ds.lon.values, dtype=np.float64),
            lat=np.asarray(ds.lat.values, dtype=np.float64),
            time_s=np.asarray(ds.time_s.values, dtype=np.float64),
            u10=np.asarray(ds.u10.values, dtype=np.float64),
            v10=np.asarray(ds.v10.values, dtype=np.float64),
            T_air=np.asarray(ds.T_air.values, dtype=np.float64),
            q_air=np.asarray(ds.q_air.values, dtype=np.float64),
            sw_down=np.asarray(ds.sw_down.values, dtype=np.float64),
            lw_down=np.asarray(ds.lw_down.values, dtype=np.float64),
            precip=np.asarray(ds.precip.values, dtype=np.float64),
            runoff=np.asarray(ds.runoff.values, dtype=np.float64),
        )
    if not allow_synthetic:
        raise FileNotFoundError(
            f"JRA55-do cache missing: {zarr_path}; populate via "
            f"``scripts/data/prepare_omip_forcing.py`` (see "
            f"https://climate.mri-jma.go.jp/pub/ocean/JRA55-do/)."
        )
    # Loud (not silent) fallback: synthetic analytic forcing is NOT the
    # OMIP-2 protocol — it must never be mistaken for a real JRA55-do run.
    logger.warning(
        "JRA55-do cache missing at %s — falling back to SYNTHETIC analytic "
        "forcing. This is NOT the OMIP-2 protocol; results are not "
        "OMIP-comparable (allow_synthetic=True was requested). Populate the "
        "cache with scripts/data/prepare_omip_forcing.py for real runs.",
        zarr_path,
    )
    return synthetic_ocean_forcing(year)


__all__ = [
    "OceanForcing",
    "JRA55_NLON",
    "JRA55_NLAT",
    "JRA55_FREQ_HOURS",
    "synthetic_ocean_forcing",
    "load_jra55_do",
]

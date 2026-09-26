"""Convert ERA5 / WeatherBench2 data to legoESM model state.

Handles the full pipeline:
1. Load ERA5 from GCS or local Zarr cache
2. Horizontal regridding (lat-lon → spectral Gaussian or cubed-sphere)
3. Vertical interpolation (pressure levels → sigma/hybrid)
4. State assembly into SegmentCarry for the compiled dycore

Also provides a local Zarr cache to avoid repeated GCS downloads.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.ml.channel_packing import WB2_PRESSURE_LEVELS
from legoesm.ml.data.era5_loader import (
    WB2_ERA5_ZARR,
    ERA5Config,
    create_era5_dataset,
)
from legoesm.thermo import (
    specific_condensate_to_mixing_ratio,
    specific_humidity_to_mixing_ratio,
)

# Canonical long ERA5/WeatherBench variable name → its short ECMWF/GRIB alias.
# Used BIDIRECTIONALLY by resolve_var: a request for either form finds the other.
_ERA5_VAR_ALIASES = {
    'temperature': 't', 'u_component_of_wind': 'u',
    'v_component_of_wind': 'v', 'specific_humidity': 'q',
    'surface_pressure': 'sp', 'skin_temperature': 'skt',
    'sea_surface_temperature': 'sst', '2m_temperature': 't2m',
    'geopotential': 'z',
    'geopotential_at_surface': 'z_sfc',
    'specific_cloud_liquid_water_content': 'clwc',
    'specific_cloud_ice_water_content': 'ciwc',
    'land_sea_mask': 'lsm',
    'mean_surface_sensible_heat_flux': 'msshf',
    'mean_surface_latent_heat_flux': 'mslhf',
    'mean_eastward_turbulent_surface_stress': 'metss',
    'mean_northward_turbulent_surface_stress': 'mntss',
    'mean_surface_downward_short_wave_radiation_flux': 'msdwswrf',
    'mean_surface_downward_long_wave_radiation_flux': 'msdwlwrf',
    'mean_surface_net_short_wave_radiation_flux': 'msnswrf',
    'mean_surface_net_long_wave_radiation_flux': 'msnlwrf',
}


def resolve_var(ds, name):
    """Find a variable in the dataset by ``name`` or an equivalent alias.

    BIDIRECTIONAL: a long-name request (``"temperature"``) finds a short-named
    store variable (``"t"``) AND a short-name request (``"t"``) finds a long-named
    store variable — so the loader is robust to either the WeatherBench/ARCO-ERA5
    long-name convention or the classic ECMWF/GRIB short-name convention.  Returns
    the matched store key, or ``None`` if neither ``name`` nor any alias is present.
    """
    if name in ds:
        return name
    candidates = []
    short = _ERA5_VAR_ALIASES.get(name)          # name is a long key → its short
    if short is not None:
        candidates.append(short)
    candidates += [long for long, s in _ERA5_VAR_ALIASES.items()
                   if s == name]                  # name is a short alias → its long(s)
    for cand in candidates:
        if cand in ds:
            return cand
    # ERA5-style invariant stores often carry the SURFACE geopotential under
    # the bare short name 'z' (the same ECMWF/GRIB code as the 3-D
    # geopotential).  Treat 'z' as surface geopotential ONLY when the variable
    # is 2-D — i.e. carries NO level dimension: a 'z' on a level axis is the
    # 3-D geopotential, not phis.  Objects without per-variable dims metadata
    # (e.g. plain sets in unit tests) safely fall through to None.
    if name in ("geopotential_at_surface", "z_sfc") and "z" in ds:
        try:
            dims = ds["z"].dims
        except (TypeError, KeyError, AttributeError):
            return None
        if not any(d in ("level", "pressure_level") for d in dims):
            return "z"
    return None


from legoesm.training.vertical_interp import (
    interp_pressure_to_sigma,
    interp_pressure_to_hybrid,
)

logger = logging.getLogger(__name__)


def _hybrid_p_s_floor(sigma, dp_floor: float = 100.0) -> float:
    """Minimum surface pressure for all hybrid layer thicknesses >= dp_floor.

    The L40 hybrid coordinate develops near-zero or negative layer thicknesses
    (dp = dA*p_ref + dB*p_s < dp_floor) when p_s << p_ref.  This returns the
    smallest p_s that keeps every level's dp above dp_floor Pa.
    """
    import numpy as _np
    dA = _np.asarray(sigma.dA)
    dB = _np.asarray(sigma.dB)
    p_ref = float(sigma.p_ref)
    # dp(k) = dA[k]*p_ref + dB[k]*p_s >= dp_floor
    # → p_s >= (dp_floor - dA[k]*p_ref) / dB[k]  when dB[k] > 0
    p_s_per_level = _np.where(dB > 0, (dp_floor - dA * p_ref) / dB, 0.0)
    return float(_np.max(p_s_per_level))


def _apply_phis_hydrostatic_adjustment(
    phis_raw,
    phis_smooth,
    p_s,
    T_sfc,
    sigma,
    is_hybrid: bool,
    dp_floor: float = 100.0,
):
    """Reconcile ``p_s`` with a SMOOTHED ``phis`` (+ optional hybrid floor).

    Grid-AGNOSTIC: operates elementwise over arbitrary leading spatial dims
    (cube ``(6, n, n)`` or lat-lon ``(n_lat, n_lon)``), so the cube and lat-lon
    ERA5 carries share ONE copy of the barometric/floor numerics.  The grid
    smoothing itself (cube vs Gaussian) is done by the caller; this only does
    the hydrostatic reconciliation that must follow it.

    Sign convention (z UP; ``phis = g*z`` surface geopotential [m^2/s^2];
    pressure increases downward):

    * Smoothing lowers terrain peaks, so ``delta = phis_raw - phis_smooth >= 0``
      where a peak was cut.  Descending from the higher RAW surface to the lower
      SMOOTHED surface is ``dPhi = -delta < 0``; the hydrostatic relation
      ``dln_p = -dPhi / (R_d T)`` then gives ``dln_p = +delta/(R_d T) > 0`` — so
      ``p_s`` INCREASES: ``p_s_corrected = p_s * exp(+delta/(R_d T))``.
      (Lowering terrain raises surface pressure. ✓)
    * Hybrid floor: where layers would become degenerate, raise ``p_s`` to
      ``p_s_floor`` and LOWER ``phis`` by the barometric equivalent
      ``R_d*T*ln(p_s_floor/p_s_corrected)`` so the split-PGF cancellation stays
      consistent with the raised ``p_s``.

    Returns ``(phis_adjusted, p_s_adjusted)`` with the same shapes as inputs.
    """
    # >= 0 where smoothing lowered terrain; may be NEGATIVE where a caller
    # passes a spectrally round-tripped target whose Gibbs overshoot exceeds
    # the raw peak — the barometric relation is exact for either sign.
    delta_phis = phis_raw - phis_smooth
    p_s_corrected = p_s * jnp.exp(delta_phis / (constants.R_d * T_sfc))
    if is_hybrid:
        p_s_floor = _hybrid_p_s_floor(sigma, dp_floor=dp_floor)
        # Only ever RAISE p_s toward the floor (ln_ratio >= 0); lower phis to match.
        ln_ratio = jnp.maximum(0.0, jnp.log(p_s_floor / p_s_corrected))
        phis_adjusted = phis_smooth - constants.R_d * T_sfc * ln_ratio
        p_s_adjusted = jnp.maximum(p_s_corrected, p_s_floor)
    else:
        phis_adjusted = phis_smooth
        p_s_adjusted = p_s_corrected
    return phis_adjusted, p_s_adjusted


# Module-level cache for regridding weights (expensive to recompute)
_CS_WEIGHT_CACHE: dict[tuple, object] = {}


def _coord_fingerprint(arr) -> tuple:
    """Content fingerprint ``(shape, hash(bytes))`` of a coordinate array.

    Keyed on the FULL array contents, not just shape + endpoints: two source grids
    with the same bounding box but different INTERIOR spacing (e.g. a uniform lat-lon
    grid vs a Gaussian-quadrature grid of the same extent — the very confusion this
    path fixes) must NOT collide.  Fingerprinting the target grid's coordinates too
    makes the cache robust to ``id()`` reuse after a grid is garbage-collected.
    """
    a = np.ascontiguousarray(np.asarray(arr))
    return (a.shape, hash(a.tobytes()))


def _get_cs_weights(src_lat, src_lon, grid):
    """Get or compute cached ERA5-lat-lon → cubed-sphere regridding weights.

    Built from the ACTUAL ERA5 lat/lon grid (``src_lat``/``src_lon``, 1-D radians) —
    NOT a Gaussian proxy of it, whose quadrature latitudes + differing latitude count
    mis-index the uniform ERA5 data (the regrid pulled near-antipodal latitudes; #cs).
    Cached on the CONTENT fingerprint of both source coords and the target grid
    coords (collision- and GC-safe; see ``_coord_fingerprint``).
    """
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    key = (_coord_fingerprint(src_lat), _coord_fingerprint(src_lon),
           _coord_fingerprint(grid.lat), _coord_fingerprint(grid.lon))
    if key not in _CS_WEIGHT_CACHE:
        from legoesm.grids.regridding import compute_latlon_to_cs_weights
        _CS_WEIGHT_CACHE[key] = compute_latlon_to_cs_weights(src_lat, src_lon, grid)
    return _CS_WEIGHT_CACHE[key]


def open_era5_zarr(zarr_path: str):
    """Open an ERA5 Zarr store with dimension normalization.

    Public wrapper around the ERA5 loader's internal helpers.
    """
    import xarray as xr
    if zarr_path.startswith("gs://"):
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(zarr_path)
        ds = xr.open_zarr(store, chunks=None)
    else:
        ds = xr.open_zarr(zarr_path, chunks=None)
    # Normalize dimension names
    rename = {}
    for short, long in [("lat", "latitude"), ("lon", "longitude")]:
        if long in ds.dims and short not in ds.dims:
            rename[long] = short
    if rename:
        ds = ds.rename(rename)
    return ds


# Extended config with surface variables needed for dycore IC + forcing
# Public ARCO-ERA5 store (Analysis-Ready Cloud-Optimized ERA5 on GCS,
# anon-readable).  Source of the radiation-flux TARGETS: the default WB2
# state store's ``mean_*_radiation_flux`` variables are NaN at every
# analysis time (probe 8533800 — 0/20 sampled times populated), but
# ARCO-ERA5 carries the same ERA5 fields with clean W/m² mean-rate fluxes
# (probe 8533818).  Same 0.25° 1440×721 grid as the WB2 state store.
ARCO_ERA5_ZARR = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)


class TrainingERA5Config(NamedTuple):
    """ERA5 config extended with surface variables for dycore training."""
    zarr_store: str = WB2_ERA5_ZARR
    pressure_variables: tuple = (
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    )
    surface_variables: tuple = (
        "surface_pressure",
        "skin_temperature",           # SST proxy
        "geopotential_at_surface",    # surface geopotential for phis [m2/s2]
    )
    levels: tuple = WB2_PRESSURE_LEVELS
    time_range: tuple = ("1979-01-01", "2020-12-31")
    dt_hours: int = 6
    local_cache_dir: str = ""     # empty = no cache
    # Radiation-flux targets for AIMIP TOA + surface flux supervision.
    # When True, ``load_era5_slice`` also loads ERA5 TOA/surface radiation
    # and derives the four model-comparable fluxes (rsut, OLR, surface net
    # SW, surface net LW) so the target carry's ``held_*`` fields hold real
    # observations instead of zeros.  Default False → byte-identical legacy.
    load_radiation_fluxes: bool = False
    # Radiation-flux TARGETS come from a SEPARATE store: the WB2 state
    # store's flux vars are all-NaN, so fluxes are read from ARCO-ERA5
    # (clean ``mean_*_radiation_flux`` in W/m², same 0.25° grid).  Set to
    # "" to read fluxes from the state ``zarr_store`` instead (only valid
    # if that store actually populates them).
    flux_zarr: str = ARCO_ERA5_ZARR
    # Cloud liquid + cloud ice for the initial condition.  OFF by default: the
    # WeatherBench2 store carries neither (only ``total_cloud_cover``), so every
    # sample would otherwise start bone dry of condensate and no microphysics
    # parameter could ever influence a short forecast.  ARCO-ERA5 has both, on
    # the same 0.25 degree grid and a superset of the WB2 pressure levels, so it
    # is read as a SECOND store exactly the way the radiation fluxes already are.
    load_cloud_condensate: bool = False
    cloud_zarr: str = ARCO_ERA5_ZARR
    # ARCO ``mean_*_radiation_flux`` are W/m² mean rates → divide by 1.0
    # (no-op).  A store accumulating J/m² over the hour would need 3600.0.
    flux_accum_seconds: float = 1.0
    # Prescribed ERA5 surface fluxes (turbulent stress, sensible/latent heat,
    # upwelling SW/LW) read from the same store as the radiation fluxes
    # (flux_zarr). load_surface_fluxes implies the land-sea mask below.
    load_surface_fluxes: bool = False
    # Read only the static land_sea_mask (0..1); implied by
    # load_surface_fluxes.
    load_land_frac: bool = False


# ---------------------------------------------------------------------------
# Local Zarr cache
# ---------------------------------------------------------------------------

_CACHE_STORE_NAME = "era5_training_cache.zarr"
# The completeness marker is written LAST, INSIDE the store dir, so it exists
# iff the ``to_zarr`` finished.  Read-side keys on THIS, never on ``.zarr``
# existence: a walltime-killed build leaves the dir + fill-value (NaN) chunks
# but no marker, which the old existence-only check silently read as complete
# (#942/#985 — silent data corruption, not an error).
_CACHE_MARKER_NAME = ".cache_complete.json"


def _read_cache_marker(cache_path: Path) -> dict | None:
    """Return the completeness-marker dict, or ``None`` when the cache is
    absent / interrupted (no marker => it must be rebuilt)."""
    marker = cache_path / _CACHE_MARKER_NAME
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text())
    except (OSError, ValueError):
        return None


def _cache_is_complete(
    cache_path: Path,
    expected_n_time: int | None,
    expected_fingerprint: str | None = None,
) -> bool:
    """A cached store is trusted only if it carries the marker AND matches the
    request: the snapshot count AND (for window-scoped caches) the fingerprint
    of the exact selected timestamps + source-store + variable config.  The
    fingerprint guard stops two DIFFERENT window sets with the same snapshot
    count (e.g. distinct chunks in the same year span) from silently reusing
    each other's store — a data-integrity failure, not just a stale read."""
    marker = _read_cache_marker(cache_path)
    if marker is None:
        return False
    if expected_n_time is not None and int(marker.get("n_time", -1)) != int(
        expected_n_time
    ):
        logger.warning(
            f"ERA5 cache {cache_path} holds n_time={marker.get('n_time')} but "
            f"{expected_n_time} snapshots were requested; rebuilding."
        )
        return False
    if expected_fingerprint is not None and str(
        marker.get("fingerprint", "")
    ) != str(expected_fingerprint):
        logger.warning(
            f"ERA5 cache {cache_path} fingerprint {marker.get('fingerprint')!r} "
            f"!= requested {expected_fingerprint!r}; rebuilding (different "
            f"window selection or variable set)."
        )
        return False
    return True


def wait_for_cache(
    cache_dir: str | Path,
    expected_n_time: int | None,
    expected_fingerprint: str | None = None,
    *,
    timeout_s: float = 3600.0,
    poll_s: float = 5.0,
) -> Path:
    """Block until the cache under ``cache_dir`` is complete, then return its path.

    Filesystem-based coordination for the multi-rank case: rank 0 BUILDS the
    window cache while the other ranks call this to WAIT for the completeness
    marker to appear (the write is atomic, so the marker flips true exactly when
    the store is ready).  Crucially this issues NO MPI collective, so it is safe
    to call from the background prefetch thread while the main thread is running
    gradient allreduces on ``COMM_WORLD`` — an MPI barrier there would interleave
    with those allreduces and deadlock (codex #985).  Raises ``TimeoutError`` if
    the builder never finishes (e.g. rank 0 died).
    """
    import time

    cache_path = Path(cache_dir) / _CACHE_STORE_NAME
    start = time.monotonic()
    while not _cache_is_complete(cache_path, expected_n_time, expected_fingerprint):
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"ERA5 cache {cache_path} not complete after {timeout_s}s "
                f"(the rank-0 builder may have failed)."
            )
        time.sleep(poll_s)
    return cache_path


def selection_fingerprint(
    time_selection: Sequence[int], config: TrainingERA5Config
) -> str:
    """Short stable hash of the EXACT window selection + source-store + variable
    config.  Used as the window-cache identity so two different selections (even
    with the same snapshot count / year span) never collide on one store."""
    import hashlib

    src = repr(
        (
            tuple(int(i) for i in time_selection),
            config.zarr_store,
            tuple(config.pressure_variables),
            tuple(config.surface_variables),
            tuple(config.levels),
        )
    )
    return hashlib.blake2b(src.encode(), digest_size=8).hexdigest()


def ensure_local_cache(
    config: TrainingERA5Config,
    cache_dir: str | Path,
    years: tuple[int, int] = (2015, 2020),
    *,
    time_selection: Sequence[int] | None = None,
    fingerprint: str | None = None,
) -> Path:
    """Materialise a subset of ERA5 to a local Zarr store, atomically.

    Two scoping modes:

    * ``time_selection=None`` (default): cache the FULL ``years`` span with all
      configured variables/levels (the year-span behaviour).
    * ``time_selection=[abs_idx, ...]``: WINDOW-scoped — cache only the given
      absolute time indices of the remote store (the snapshots the training
      windows actually touch, +lead spillover), preserving the real ``time``
      coordinate so the reader can select them back by timestamp.  ~0.5 TB
      instead of ~10 TB for the AIMIP T106 workload (#985).

    Correctness (#942/#985): the build is atomic (write to a ``.building`` tmp
    dir, then ``os.replace`` into place) and gated by a completeness marker
    written last, so an interrupted build is rebuilt — never read as
    fill-value NaNs.  Subsequent calls with a matching, complete store skip the
    download.

    Returns the path to the local Zarr store.
    """
    cache_path = Path(cache_dir) / _CACHE_STORE_NAME
    expected_n_time = (
        len(time_selection) if time_selection is not None else None
    )

    if _cache_is_complete(cache_path, expected_n_time, fingerprint):
        logger.info(f"Using cached ERA5 at {cache_path}")
        return cache_path

    if time_selection is not None:
        # Window-scoped: subset the remote store to exactly the requested
        # absolute snapshots (keeping the real ``time`` coord + only the
        # pressure/surface/static vars the reader consumes).
        logger.info(
            f"Downloading {len(time_selection)} window-scoped ERA5 snapshots "
            f"to {cache_path}..."
        )
        ds_full = open_era5_zarr(config.zarr_store)
        ds_sub = ds_full.isel(time=list(int(i) for i in time_selection))
        keep: list[str] = []
        for name in (
            list(config.pressure_variables)
            + list(config.surface_variables)
            + ["geopotential_at_surface"]
        ):
            r = resolve_var(ds_sub, name)
            if r is not None and r in ds_sub and r not in keep:
                keep.append(r)
        ds = ds_sub[keep]
        # Keep only the CONFIGURED pressure levels: the reader selects these
        # anyway, so caching every level of a 37-level source would inflate the
        # window-cache footprint ~3x for no benefit (codex #985).
        _ldim = next(
            (d for d in ("level", "pressure_level") if d in ds.dims), None
        )
        if _ldim is not None:
            _have = set(np.asarray(ds[_ldim].values).tolist())
            _want = [lv for lv in config.levels if lv in _have]
            if _want:
                ds = ds.sel({_ldim: _want})
    else:
        logger.info(f"Downloading ERA5 {years[0]}-{years[1]} to {cache_path}...")
        era5_cfg = ERA5Config(
            zarr_store=config.zarr_store,
            variables=config.pressure_variables,
            levels=config.levels,
            time_range=(f"{years[0]}-01-01", f"{years[1]}-12-31"),
            dt_hours=config.dt_hours,
        )
        ds = create_era5_dataset(era5_cfg)

        # Also grab surface variables
        ds_full = open_era5_zarr(config.zarr_store)
        ds_full = ds_full.sel(
            time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31")
        )
        for svar in config.surface_variables:
            resolved = [v for v in [resolve_var(ds_full, svar)] if v]
            for r in resolved:
                if r in ds_full and r not in ds:
                    ds[r] = ds_full[r]

    # --- Atomic write: build to a tmp store, mark complete, then swap in. -----
    # A kill mid-``to_zarr`` leaves only ``tmp_path`` (no marker at the final
    # path), so the next call rebuilds instead of reading a torn store.
    # PER-PROCESS tmp dir. A FIXED ``.building`` path is not actually atomic
    # when two processes build the same cache key concurrently: both write into
    # the one temp dir, and the loser's os.replace dies with
    # "OSError: [Errno 39] Directory not empty: ...zarr.building" while the
    # other can see its store vanish mid-read (zarr FileNotFoundError on a
    # group node). Both failures were observed 2026-08-05 when three AIMIP
    # variants sharing base_t106_allyears started within minutes of each other
    # — they request different SUBSETS but hash to the SAME window key, so the
    # differing snapshot counts hid the collision.
    #
    # The PID suffix makes each builder's temp dir private; the final
    # ``os.replace`` onto the shared path stays atomic, so a late finisher
    # simply replaces an equivalent complete store. Stale dirs from a killed
    # job are swept below (own-PID only is not enough — a dead PID's dir would
    # leak), guarded to this process's own prefix so a CONCURRENT builder's
    # live temp dir is never deleted.
    import os as _os

    tmp_path = Path(cache_dir) / f"{_CACHE_STORE_NAME}.building.{_os.getpid()}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    # CHUNKED WRITE. A single ``to_zarr`` over a fancy multi-thousand-index
    # ``isel`` did not stream: the .building store sat at 12 KB (metadata only)
    # for 11 minutes while xarray/dask built one enormous graph, which is why
    # the window cache was abandoned and the data lever capped at ~1,400
    # snapshots (2026-07-27). Writing the time axis in blocks keeps each graph
    # small, streams to disk, and shows progress — which is what lets the
    # training set grow past that cap.
    # Drop the SOURCE encoding before writing. The remote store carries zarr-v2
    # numcodecs compressors (Blosc); handing those to a zarr-v3 writer raises
    # "Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'>". Letting
    # the writer choose its own codecs is correct here — we are re-encoding a
    # subset, not preserving byte layout.
    ds = ds.copy()
    for _v in list(ds.variables):
        ds[_v].encoding = {}
    _tdim = "time" if "time" in ds.dims else None
    _n_t = int(ds.sizes.get(_tdim, 0)) if _tdim else 0
    _block = 64
    if _tdim is None or _n_t <= _block:
        ds.to_zarr(str(tmp_path), mode="w")
    else:
        ds.isel({_tdim: slice(0, _block)}).to_zarr(str(tmp_path), mode="w")
        for _s in range(_block, _n_t, _block):
            _e = min(_s + _block, _n_t)
            ds.isel({_tdim: slice(_s, _e)}).to_zarr(
                str(tmp_path), mode="a", append_dim=_tdim)
            logger.info("  cached %d/%d snapshots", _e, _n_t)
    n_time = int(ds.sizes.get("time", 0))
    # Marker LAST, inside the tmp store, so it is present iff the write finished.
    (tmp_path / _CACHE_MARKER_NAME).write_text(
        json.dumps(
            {
                "n_time": n_time,
                "years": list(years),
                "windowed": time_selection is not None,
                "fingerprint": fingerprint,
            }
        )
    )
    # Atomic swap: drop any stale (incomplete) final store, then rename.
    # ponytail: rmtree+os.replace over a 2-phase commit — the tiny gap between
    # them can only ever cost a rebuild (correct), never a silent-NaN read.
    if cache_path.exists():
        shutil.rmtree(cache_path)
    os.replace(tmp_path, cache_path)
    logger.info(f"Cached ERA5 to {cache_path} ({n_time} snapshots)")
    return cache_path


# ---------------------------------------------------------------------------
# ERA5 → model state conversion
# ---------------------------------------------------------------------------

class ERA5Slice(NamedTuple):
    """A single ERA5 time slice with all needed fields."""
    T: np.ndarray          # (n_lat, n_lon, n_plev) temperature [K]
    u: np.ndarray          # (n_lat, n_lon, n_plev) zonal wind [m/s]
    v: np.ndarray          # (n_lat, n_lon, n_plev) meridional wind [m/s]
    q: np.ndarray          # (n_lat, n_lon, n_plev) specific humidity [kg/kg]
    p_s: np.ndarray        # (n_lat, n_lon) surface pressure [Pa]
    sst: np.ndarray        # (n_lat, n_lon) skin/SST temperature [K]
    phis: np.ndarray       # (n_lat, n_lon) surface geopotential [m²/s²]
    lat: np.ndarray        # (n_lat,) latitude [rad]
    lon: np.ndarray        # (n_lon,) longitude [rad]
    plev_Pa: np.ndarray    # (n_plev,) pressure levels [Pa], ascending
    # Optional radiation-flux targets [W/m²] (None unless
    # ``config.load_radiation_fluxes``).  Conventions match the model's
    # ``SegmentCarry.held_*`` fields:
    #   rsut       = TOA outgoing (reflected) SW, positive up
    #   olr        = TOA outgoing LW (OLR/rlut), positive up
    #   sfc_net_sw = surface net SW (down − up), positive down
    #   sfc_net_lw = surface net LW (down − up), positive down (usually <0)
    rsut: np.ndarray = None        # (n_lat, n_lon)
    olr: np.ndarray = None         # (n_lat, n_lon)
    sfc_net_sw: np.ndarray = None  # (n_lat, n_lon)
    sfc_net_lw: np.ndarray = None  # (n_lat, n_lon)
    # Optional cloud condensate, SPECIFIC contents [kg/kg of moist air], on the
    # same (n_lat, n_lon, n_plev) grid as ``q`` (None unless
    # ``config.load_cloud_condensate``).  The carry builders convert them to
    # dry-air mixing ratios, which is what the microphysics consumes.
    q_c: np.ndarray = None         # (n_lat, n_lon, n_plev) cloud liquid
    q_i: np.ndarray = None         # (n_lat, n_lon, n_plev) cloud ice
    # --- prescribed surface boundary planes (n_lat, n_lon); legoESM sign
    # conventions, None unless the matching load_* flag is set ---
    sfc_shf: np.ndarray = None      # sensible heat flux [W/m^2], positive UP
    sfc_lhf: np.ndarray = None      # latent heat flux [W/m^2], positive UP
    sfc_tau_x: np.ndarray = None    # eastward turbulent stress ON THE ATMOSPHERE [Pa]
    sfc_tau_y: np.ndarray = None    # northward turbulent stress ON THE ATMOSPHERE [Pa]
    sfc_sw_up: np.ndarray = None    # upwelling SW at the surface [W/m^2], positive UP
    sfc_sw_down: np.ndarray = None  # downwelling SW at the surface [W/m^2], positive DOWN
    sfc_lw_up: np.ndarray = None    # upwelling LW at the surface [W/m^2], positive UP
    land_frac: np.ndarray = None    # static land-sea fraction, dimensionless [0..1]


def _assert_required_era5_vars(ds_t, ds) -> None:
    """Upfront preflight: report ALL missing REQUIRED ERA5 variables in ONE error.

    The per-field loaders (``_get_3d`` / ``_get_2d``) each raise on the FIRST unresolvable
    required variable, so an operator preparing a real-ERA5 zarr with several mis-named
    fields would iterate error-by-error.  This lists every missing required variable (with
    its accepted alias) at once, so the whole naming pass is fixed in one go.  Required =
    the 3D state T/u/v/q + the surface pressure; SST + surface geopotential stay OPTIONAL
    (zero-filled if absent, so they are NOT flagged here).
    """
    required = ("temperature", "u_component_of_wind", "v_component_of_wind",
                "specific_humidity", "surface_pressure")
    missing = [n for n in required
               if resolve_var(ds_t, n) is None and resolve_var(ds, n) is None]
    if missing:
        listed = ", ".join(
            f"{n!r} (alias {_ERA5_VAR_ALIASES.get(n, '—')!r})" for n in missing)
        raise ValueError(
            f"load_era5_slice: REQUIRED ERA5 variable(s) {listed} not found in the "
            f"store; available variables: {sorted(map(str, ds_t.data_vars))}. Provide "
            "them (or an accepted alias) in --era5-zarr — a missing required field must "
            "NOT silently load as zeros (it would corrupt the whole compare).")


def load_era5_slice(
    config: TrainingERA5Config, time_idx: int, *, ds: Any = None,
    flux_ds: Any = None, cloud_ds: Any = None,
) -> ERA5Slice:
    """Load a single ERA5 time slice with all fields needed for IC + forcing.

    Parameters
    ----------
    config : TrainingERA5Config
    time_idx : int
        Time index into the dataset.
    ds : xarray.Dataset, optional
        A PRE-OPENED ERA5 dataset (lat/lon dims, ``resolve_var``-findable variables on
        a ``level`` axis).  When given, the Zarr open from ``config`` is BYPASSED — used
        by a local-archive adapter (e.g. per-variable NetCDF merged into one dataset)
        to feed REAL ERA5 through the SAME extraction/regrid chain WITHOUT a Zarr store
        or network.  ``None`` (default) opens the configured store as before.
    flux_ds : xarray.Dataset, optional
        A PRE-OPENED flux store (see ``config.flux_zarr``); pass it when
        looping over many snapshots so the flux zarr is opened once.
        Consulted by ``load_radiation_fluxes``, ``load_surface_fluxes`` and
        (as the fallback behind the state store) ``load_land_frac``.
    cloud_ds : xarray.Dataset, optional
        A PRE-OPENED cloud-condensate store (see ``config.cloud_zarr``); same
        reason.  Only consulted when ``config.load_cloud_condensate`` is True.

    Returns
    -------
    ERA5Slice with all fields on the native ERA5 lat-lon grid.
    """
    store = config.local_cache_dir if config.local_cache_dir else config.zarr_store
    if ds is None:
        ds = open_era5_zarr(store)
    else:
        # Normalize a PRE-OPENED ds the same way open_era5_zarr does, so a local-archive
        # adapter may pass the raw ``latitude``/``longitude`` dims (the NetCDF/CF
        # convention) without renaming them itself.
        _rename = {long: short for short, long in (("lat", "latitude"),
                                                   ("lon", "longitude"))
                   if long in ds.dims and short not in ds.dims}
        if _rename:
            ds = ds.rename(_rename)

    # Bounds-check the time index up front so a typo'd ``--era5-time-idx`` /
    # held-out index gives a CAMPAIGN-specific message (with the store's actual time
    # count) instead of xarray's generic "index N is out of bounds for axis 0".  Same
    # ``IndexError`` type (callers/tests catching it are unaffected); negative indices
    # are allowed exactly as xarray would (valid range [-n, n-1]).  ``time``-dim
    # absent ⇒ skip and let ``isel`` raise (a differently-named time axis).
    n_time = ds.sizes.get("time")
    if n_time is not None and not (-n_time <= time_idx < n_time):
        raise IndexError(
            f"era5 time index {time_idx} is out of range: the ERA5 store has "
            f"{n_time} time(s) (valid 0..{n_time - 1} or -{n_time}..-1). Pick an "
            "in-range --era5-time-idx / held-out index.")

    # Select time
    ds_t = ds.isel(time=time_idx)

    # Fail loud + COMPLETE on a mis-prepared store: report every missing required variable
    # at once (the per-field _get_* below still raise as a backstop / on other errors).
    _assert_required_era5_vars(ds_t, ds)

    # Extract lat/lon
    lat = np.deg2rad(ds_t.lat.values.astype(np.float64))
    lon = np.deg2rad(ds_t.lon.values.astype(np.float64))

    # Pressure levels (hPa → Pa, ensure ascending)
    level_dim = "level" if "level" in ds.dims else "pressure_level"
    plev_hPa = np.array(config.levels, dtype=np.float64)
    plev_Pa = np.sort(plev_hPa * 100.0)  # ascending in Pa

    def _missing(name):
        return (
            f"load_era5_slice: REQUIRED ERA5 variable {name!r} not found in the "
            f"store (tried alias {_ERA5_VAR_ALIASES.get(name, '—')!r}); available "
            f"variables: {sorted(map(str, ds_t.data_vars))}. Fix the store / "
            "--era5-zarr or the variable naming — a missing required field must NOT "
            "silently load as zeros (it would corrupt the whole compare)."
        )

    def _get_3d(name, *, required=True, src=None, flip_lat=False):
        """Extract a 3D variable as (lat, lon, level), levels ascending in pressure.

        A REQUIRED but unresolvable variable RAISES (never silently zero-fills — a
        zeros T/u/v/q would corrupt the bias and make the loop 'correct' garbage).

        ``src`` reads from a SECOND store (the cloud-condensate store) through the
        identical level selection and ascending-pressure reordering, so the two
        stores can never end up with their levels paired differently.  ``flip_lat``
        reverses the latitude axis of that second store to the state store's sense.
        """
        src = ds_t if src is None else src
        resolved = resolve_var(src, name)
        if resolved is None:
            if required:
                raise ValueError(_missing(name))
            return np.zeros((len(lat), len(lon), len(plev_Pa)), dtype=np.float32)
        data = src[resolved].sel({level_dim: list(config.levels)}).values
        if data.ndim == 3:
            dims = list(src[resolved].dims)
            spatial = {"lat", "lon", "latitude", "longitude"}
            level_axis = next((i for i, d in enumerate(dims) if d not in spatial), 0)
            if level_axis != 2:
                data = np.moveaxis(data, level_axis, -1)
        # Reorder the level axis to ASCENDING pressure, matching ``plev_Pa =
        # np.sort(plev_hPa)`` for ANY level order — not just a monotonic config.levels.
        # The coordinate is robustly sorted, so the DATA (selected in config.levels
        # order) must be reordered the SAME way; a mere ``[::-1]`` flip only matches when
        # config.levels is monotonic, so a non-monotonic list (e.g. [1000, 850, 500, 700,
        # 200]) would silently pair each level's data with the WRONG pressure in the
        # vertical interp.  ``argsort`` == reversal for the descending WB2 default, so
        # this is behavior-preserving there.
        order = np.argsort(plev_hPa)
        data = data[..., order]
        if flip_lat:
            data = data[::-1]
        return data.astype(np.float32)

    def _get_2d(name, *, required=False):
        """Extract a 2D surface variable as (lat, lon).

        ``required`` (e.g. ``surface_pressure``) RAISES on an unresolvable variable;
        optional surface fields (skin temperature, surface geopotential) keep the
        zero-fill so an IC missing them still loads.
        """
        # Try time-selected dataset first, then full dataset for static fields
        resolved = resolve_var(ds_t, name)
        if resolved is None:
            resolved = resolve_var(ds, name)
            if resolved is None:
                if required:
                    raise ValueError(_missing(name))
                return np.zeros((len(lat), len(lon)), dtype=np.float32)
            data = ds[resolved].values
        else:
            data = ds_t[resolved].values
        # Drop singleton dimensions and take first slice of any extra dims
        data = data.squeeze()
        while data.ndim > 2:
            data = data[0]
        return data.astype(np.float32)

    def _has(name):
        return resolve_var(ds_t, name) is not None or resolve_var(ds, name) is not None

    def _get_sst():
        """Skin/SST surface-temperature forcing with a physical fallback chain.

        ``skin_temperature`` (defined everywhere) when the store carries it; else
        ``sea_surface_temperature`` (NaN over land) gap-filled with
        ``2m_temperature``; else ``2m_temperature`` alone (skin proxy).  The
        legacy zero-fill (0 K!) is the LAST resort and warns loudly: the WB2
        6h zarr has no skin_temperature, and the silent 0 K SST forcing sent
        the WB scale-trainer surface fluxes into a sick regime (#797 bug 7).
        """
        if _has("skin_temperature"):
            return _get_2d("skin_temperature")
        if _has("sea_surface_temperature"):
            sst = _get_2d("sea_surface_temperature")
            if _has("2m_temperature"):
                t2m = _get_2d("2m_temperature")
                return np.where(np.isfinite(sst), sst, t2m).astype(np.float32)
            fill = float(np.nanmean(sst))
            return np.nan_to_num(sst, nan=fill).astype(np.float32)
        if _has("2m_temperature"):
            return _get_2d("2m_temperature")
        logger.warning(
            "ERA5 store has none of skin_temperature/sea_surface_temperature/"
            "2m_temperature; sst zero-filled (0 K) — unusable as SST forcing")
        return np.zeros((len(lat), len(lon)), dtype=np.float32)

    def _get_phis():
        """Surface geopotential phis [m²/s²]; zero-fill is LAST resort + LOUD.

        Resolves ``geopotential_at_surface`` / ``z_sfc`` / a 2-D ``z``
        (``resolve_var``'s dimension-checked short alias).  A store lacking all
        of them keeps the legacy zero-fill so idealized ICs still load, but
        warns loudly (matching ``_get_sst``): real ERA5 surface pressure
        (~600 hPa over Tibet) combined with phis=0 (flat topography) yields a
        grossly NON-HYDROSTATIC initial condition that the dycore cannot
        balance.
        """
        if _has("geopotential_at_surface"):
            return _get_2d("geopotential_at_surface")
        logger.warning(
            "ERA5 store has no surface geopotential ('geopotential_at_surface'"
            " / 'z_sfc' / 2-D 'z'); phis zero-filled (flat topography) — with"
            " real ERA5 surface pressure (~600 hPa over Tibet) this produces a"
            " grossly non-hydrostatic initial condition")
        return np.zeros((len(lat), len(lon)), dtype=np.float32)

    # Optional radiation-flux targets (TOA + surface) for AIMIP flux
    # supervision.  Derive the four model-comparable fluxes:
    #   rsut       = top_downward_SW − top_net_SW   (reflected up, +up)
    #   OLR        = −top_net_LW                     (TOA net LW = −OLR)
    #   sfc_net_sw = surface_net_SW                  (down − up, +down)
    #   sfc_net_lw = surface_net_LW                  (down − up, +down)
    # ``flux_accum_seconds`` (default 1.0) converts an accumulated-J/m²
    # store to W/m²; it is a no-op for the W/m² ARCO store.
    rsut = olr = sfc_net_sw = sfc_net_lw = None
    sfc_shf = sfc_lhf = sfc_tau_x = sfc_tau_y = None
    sfc_sw_up = sfc_sw_down = sfc_lw_up = None
    land_frac = None
    # One shared flux-store context: the radiation fluxes, the prescribed
    # surface fluxes and the static land-sea mask all read from the same
    # (flux_zarr) store, which is opened at most once per slice.
    # A land mask that the STATE store already carries (WB2 does) must not
    # cost a remote flux-store open, so the store is opened lazily: the flux
    # readers force it, the mask reader only when the state store lacks it.
    _want_land = config.load_land_frac or config.load_surface_fluxes
    if (config.load_radiation_fluxes or config.load_surface_fluxes
            or _want_land):
        fzarr = config.flux_zarr or store
        _flux_ctx = {"ds": flux_ds, "ds_t": None, "flip": None}

        def _flux_store():
            if _flux_ctx["ds"] is None:
                _flux_ctx["ds"] = (open_era5_zarr(fzarr) if config.flux_zarr
                                   else ds)
            if _flux_ctx["ds_t"] is None:
                _fds = _flux_ctx["ds"]
                # EXACT timestamp: a prescribed boundary condition from the
                # wrong hour is a silent forcing error, so no nearest-match.
                _t_state = np.asarray(ds_t.time.values).reshape(-1)[0]
                try:
                    _flux_ctx["ds_t"] = _fds.sel(time=_t_state)
                except KeyError as e:
                    raise ValueError(
                        f"flux store {fzarr} has no snapshot at the state "
                        f"time {_t_state}; the WB2 6-hourly times must be "
                        "a subset of the flux store's times.") from e
                flux_lat_deg = np.asarray(_fds.lat.values, dtype=np.float64)
                state_lat_deg = np.rad2deg(lat)
                _flux_ctx["flip"] = (
                    np.sign(flux_lat_deg[1] - flux_lat_deg[0])
                    != np.sign(state_lat_deg[1] - state_lat_deg[0]))
                # Same COORDINATES, not just the same shape and sense: a
                # shifted longitude origin would prescribe every plane at
                # the wrong location and nothing downstream could tell.
                _fl = flux_lat_deg[::-1] if _flux_ctx["flip"] else flux_lat_deg
                _flon = np.asarray(_fds.lon.values, dtype=np.float64)
                _slon = np.rad2deg(lon)
                if (_fl.shape != state_lat_deg.shape
                        or _flon.shape != _slon.shape
                        or not np.allclose(_fl, state_lat_deg, atol=1e-6)
                        or not np.allclose(_flon, _slon, atol=1e-6)):
                    raise ValueError(
                        f"flux store {fzarr} grid coordinates differ from the "
                        "state store's (lat/lon values, not only the shape); "
                        "refusing to prescribe fluxes at the wrong locations.")
            return _flux_ctx["ds"], _flux_ctx["ds_t"], _flux_ctx["flip"]

        def _flux_2d(name, flag):
            flux_ds, fds_t, flip_lat = _flux_store()
            r = resolve_var(fds_t, name)
            src = fds_t
            if r is None:
                r = resolve_var(flux_ds, name)
                src = flux_ds
            if r is None:
                raise ValueError(
                    f"{flag}=True but flux variable {name!r} "
                    f"is absent from {fzarr}.")
            d = np.asarray(src[r].values).squeeze()
            while d.ndim > 2:
                d = d[0]
            if d.shape != (len(lat), len(lon)):
                raise ValueError(
                    f"flux field {name!r} grid {d.shape} != state grid "
                    f"{(len(lat), len(lon))}; flux_zarr must match the state "
                    f"store resolution (both 0.25° ERA5).")
            if flip_lat:
                d = d[::-1]
            return d.astype(np.float32)

        if config.load_radiation_fluxes:
            acc = np.float32(config.flux_accum_seconds)
            toa_dn_sw = _flux_2d("mean_top_downward_short_wave_radiation_flux", "load_radiation_fluxes")
            toa_net_sw = _flux_2d("mean_top_net_short_wave_radiation_flux", "load_radiation_fluxes")
            toa_net_lw = _flux_2d("mean_top_net_long_wave_radiation_flux", "load_radiation_fluxes")
            sfc_net_sw_v = _flux_2d("mean_surface_net_short_wave_radiation_flux", "load_radiation_fluxes")
            sfc_net_lw_v = _flux_2d("mean_surface_net_long_wave_radiation_flux", "load_radiation_fluxes")
            rsut = (toa_dn_sw - toa_net_sw) / acc
            olr = (-toa_net_lw) / acc
            sfc_net_sw = sfc_net_sw_v / acc
            sfc_net_lw = sfc_net_lw_v / acc

        if config.load_surface_fluxes:
            # ARCO accumulations -> mean rates, divided by acc exactly like
            # the radiation block above (W/m^2 for heat/radiation, N/m^2 for
            # stress).
            #
            # SIGN CONVENTIONS (ERA5 -> legoESM):
            # * ERA5 mean surface heat fluxes are POSITIVE DOWNWARD; legoESM's
            #   shflx/lhflx are POSITIVE UPWARD:
            #   shf = -mean_surface_sensible_heat_flux,
            #   lhf = -mean_surface_latent_heat_flux.
            # * ERA5 turbulent surface stress is the stress the atmosphere
            #   exerts ON THE SURFACE (positive eastward for eastward wind);
            #   legoESM's tau_x/tau_y in surface_layer.compute_surface_fluxes
            #   are the stress ON THE ATMOSPHERE (tau = -rho*Cd*|U|*u,
            #   opposite sign to the wind):
            #   tau_x = -mean_eastward_turbulent_surface_stress,
            #   tau_y = -mean_northward_turbulent_surface_stress.
            # * ERA5 surface net radiation = down - up (positive down), so the
            #   upwelling fields are sw_up = sw_down - sw_net and
            #   lw_up = lw_down - lw_net, both POSITIVE UPWARD.
            acc = np.float32(config.flux_accum_seconds)
            _mssfhf = _flux_2d("mean_surface_sensible_heat_flux", "load_surface_fluxes")
            _mslhf = _flux_2d("mean_surface_latent_heat_flux", "load_surface_fluxes")
            _ewss = _flux_2d("mean_eastward_turbulent_surface_stress", "load_surface_fluxes")
            _nsss = _flux_2d("mean_northward_turbulent_surface_stress", "load_surface_fluxes")
            _swd = _flux_2d("mean_surface_downward_short_wave_radiation_flux", "load_surface_fluxes")
            _swn = _flux_2d("mean_surface_net_short_wave_radiation_flux", "load_surface_fluxes")
            _lwd = _flux_2d("mean_surface_downward_long_wave_radiation_flux", "load_surface_fluxes")
            _lwn = _flux_2d("mean_surface_net_long_wave_radiation_flux", "load_surface_fluxes")
            sfc_shf = -_mssfhf / acc
            sfc_lhf = -_mslhf / acc
            sfc_tau_x = -_ewss / acc
            sfc_tau_y = -_nsss / acc
            sfc_sw_up = (_swd - _swn) / acc
            sfc_sw_down = _swd / acc
            sfc_lw_up = (_lwd - _lwn) / acc
            # A non-finite plane would become a zero flux (classical anchor)
            # or a NaN input (learned arm) downstream, where nothing can
            # raise; ERA5 has none, so a NaN here is a store defect.
            for _nm, _arr in (("sfc_shf", sfc_shf), ("sfc_lhf", sfc_lhf),
                              ("sfc_tau_x", sfc_tau_x),
                              ("sfc_tau_y", sfc_tau_y),
                              ("sfc_sw_up", sfc_sw_up),
                              ("sfc_sw_down", sfc_sw_down),
                              ("sfc_lw_up", sfc_lw_up)):
                if not np.all(np.isfinite(_arr)):
                    raise ValueError(
                        f"load_surface_fluxes=True: {_nm} has "
                        f"{int((~np.isfinite(_arr)).sum())} non-finite "
                        f"values at time index {time_idx} in {fzarr}; "
                        "refusing to prescribe a broken boundary condition.")

        if _want_land:
            # Static land-sea mask (0..1): state store first, then the flux
            # store; absent everywhere is a configuration error.
            _r = resolve_var(ds_t, "land_sea_mask")
            _src, _from_flux_store, flip_lat = ds_t, False, False
            if _r is None:
                _r = resolve_var(ds, "land_sea_mask")
                if _r is not None:
                    _src = ds
            if _r is None:
                flux_ds, fds_t, flip_lat = _flux_store()
                _r = resolve_var(fds_t, "land_sea_mask")
                if _r is not None:
                    _src, _from_flux_store = fds_t, True
                else:
                    _r = resolve_var(flux_ds, "land_sea_mask")
                    if _r is not None:
                        _src, _from_flux_store = flux_ds, True
            if _r is None:
                raise ValueError(
                    "load_land_frac=True (directly or implied by "
                    "load_surface_fluxes=True) but the static land-sea mask "
                    "('land_sea_mask') is absent from both the state store "
                    f"({store}) and the flux store ({fzarr}).")
            _v = _src[_r]
            while _v.ndim > 2:
                # static field: drop any leading (time) axis, first slice
                _v = _v[0]
            _d = np.asarray(_v.values).squeeze()
            if _d.shape != (len(lat), len(lon)):
                raise ValueError(
                    f"land_sea_mask grid {_d.shape} != state grid "
                    f"{(len(lat), len(lon))}.")
            if _from_flux_store and flip_lat:
                _d = _d[::-1]
            if not np.all(np.isfinite(_d)):
                raise ValueError(
                    "land_sea_mask has non-finite values; refusing to feed "
                    "a broken land fraction to the model.")
            land_frac = _d.astype(np.float32)

    # --- optional cloud condensate for the initial condition ---------------
    # Read as a SECOND store (the WB2 state store has no cloud water at all),
    # at the SAME timestamp, the SAME pressure levels and through the SAME
    # ``_get_3d`` level handling, so liquid, ice and humidity can never end up
    # paired with different pressures.  A missing variable RAISES: silently
    # zero-filling is exactly the condition this option exists to remove, and a
    # run that thinks it has cloud water and does not would be worse than one
    # that never asked.
    q_c_spec = q_i_spec = None
    if config.load_cloud_condensate:
        if cloud_ds is None:
            cloud_ds = (open_era5_zarr(config.cloud_zarr)
                        if config.cloud_zarr else ds)
        cds_t = cloud_ds.sel(time=ds_t.time.values, method="nearest")
        # ``nearest`` has no tolerance of its own: a store missing the analysis
        # hour would hand back a field from another day and nothing would say
        # so.  ARCO is hourly and the WB2 6-hourly times are a subset of it, so
        # the match is EXACT or the stores do not belong together.
        _want = np.asarray(ds_t.time.values, dtype="datetime64[ns]")
        _got = np.asarray(cds_t.time.values, dtype="datetime64[ns]")
        if _got != _want:
            raise ValueError(
                f"cloud_zarr has no field at {_want}; nearest is {_got}. The "
                "condensate store must cover the state store's analysis times.")
        # Same 0.25 degree ERA5 grid and 0..360 longitude origin as the state
        # store, so only the latitude sense can differ (ARCO is N->S, WB2 may
        # be S->N) — mirroring the radiation-flux alignment above.
        _cloud_lat = np.asarray(
            (cloud_ds.lat if "lat" in cloud_ds.dims else cloud_ds.latitude).values,
            dtype=np.float64)
        _state_lat = np.rad2deg(lat)
        if len(_cloud_lat) != len(_state_lat):
            raise ValueError(
                f"cloud_zarr latitude size {len(_cloud_lat)} != state store "
                f"{len(_state_lat)}; both must be the same ERA5 grid.")
        _cflip = (np.sign(_cloud_lat[1] - _cloud_lat[0])
                  != np.sign(_state_lat[1] - _state_lat[0]))
        # Same size and same sense is not the same GRID: a half-cell offset, or
        # a 0..360 versus -180..180 longitude origin, loads condensate that is
        # geographically displaced and entirely plausible-looking.
        if not np.allclose(np.sort(_cloud_lat), np.sort(_state_lat), atol=1e-4):
            raise ValueError(
                "cloud_zarr latitudes differ from the state store's beyond "
                "1e-4 degrees; the two must be the same ERA5 grid.")
        _cloud_lon = np.asarray(
            (cloud_ds.lon if "lon" in cloud_ds.dims else cloud_ds.longitude).values,
            dtype=np.float64)
        _state_lon = np.rad2deg(lon)
        if (len(_cloud_lon) != len(_state_lon)
                or not np.allclose(_cloud_lon, _state_lon, atol=1e-4)):
            raise ValueError(
                f"cloud_zarr longitudes ({_cloud_lon[:2]}...{_cloud_lon[-1:]}) "
                f"differ from the state store's "
                f"({_state_lon[:2]}...{_state_lon[-1:]}); a different origin "
                "or offset would load geographically displaced condensate.")
        # Levels and units, not just the horizontal grid: a store whose level
        # coordinate is labelled in Pa rather than hPa, or whose condensate is
        # a density [kg/m^3] rather than a specific content [kg/kg], selects
        # and converts without complaint and is wrong by orders of magnitude.
        _clev = np.asarray(cds_t[level_dim].values, dtype=np.float64)
        _want_lev = np.asarray(config.levels, dtype=np.float64)
        if not np.all(np.isin(_want_lev, _clev)):
            raise ValueError(
                f"cloud_zarr level coordinate {_clev[:4]}... does not contain "
                f"the configured levels {_want_lev[:4]}...; check its units "
                "(hPa vs Pa).")
        for _name in ("specific_cloud_liquid_water_content",
                      "specific_cloud_ice_water_content"):
            _r = resolve_var(cds_t, _name)
            _u = (cds_t[_r].attrs.get("units", "") if _r is not None else "")
            if _u and _u.replace(" ", "").replace("**", "").lower() not in (
                    "kgkg-1", "kg/kg", "kgkg^-1", "1", "kg kg-1".replace(" ", "")):
                raise ValueError(
                    f"cloud_zarr {_name!r} has units {_u!r}; this path expects "
                    "a SPECIFIC content in kg/kg, not a density.")

        q_c_spec = _get_3d("specific_cloud_liquid_water_content",
                           src=cds_t, flip_lat=_cflip)
        q_i_spec = _get_3d("specific_cloud_ice_water_content",
                           src=cds_t, flip_lat=_cflip)

    return ERA5Slice(
        T=_get_3d("temperature"),
        u=_get_3d("u_component_of_wind"),
        v=_get_3d("v_component_of_wind"),
        q=_get_3d("specific_humidity"),
        p_s=_get_2d("surface_pressure", required=True),
        sst=_get_sst(),
        phis=_get_phis(),  # optional (loud-warned zero-fill); already in m²/s²
        lat=lat,
        lon=lon,
        plev_Pa=plev_Pa,
        rsut=rsut,
        olr=olr,
        sfc_net_sw=sfc_net_sw,
        sfc_net_lw=sfc_net_lw,
        q_c=q_c_spec,
        q_i=q_i_spec,
        sfc_shf=sfc_shf, sfc_lhf=sfc_lhf,
        sfc_tau_x=sfc_tau_x, sfc_tau_y=sfc_tau_y,
        sfc_sw_up=sfc_sw_up, sfc_sw_down=sfc_sw_down, sfc_lw_up=sfc_lw_up,
        land_frac=land_frac,
    )


def load_era5_time_mean(
    config: TrainingERA5Config, time_indices, *, ds: Any = None
) -> ERA5Slice:
    """Time-MEAN ERA5 reference: the element-wise average over ``time_indices`` of each
    field, so the time-mean MODEL state (``run_to_column_mean``) is compared to a
    time-mean ERA5 CLIMATOLOGY rather than a single synoptic snapshot (which injects
    weather noise into the bias).

    Coords (``lat``/``lon``/``plev_Pa``) are identical across times (kept from the
    first slice).  A SINGLE index returns :func:`load_era5_slice` unchanged
    (byte-identical to the old single-time behaviour).  NaN-PROPAGATING (a running SUM,
    not ``nanmean``) — consistent with the raw single-slice extraction: an SST-over-land
    cell is NaN in every slice, so the mean is NaN there too, no worse than a single
    slice.

    Memory: an INCREMENTAL running sum (float64) holds only ~ONE slice's worth + the
    current slice, NOT all ``N`` slices at once — so averaging a full-res global ERA5
    over many times (a monthly/seasonal climatology) does not OOM.  The sum is float64
    (no float32 precision loss over many times); each field is divided by ``N`` and cast
    BACK to the first slice's dtype.  Reuses ``load_era5_slice`` per time (a one-time
    campaign-start load).  An OUT-OF-RANGE time index (the requested window exceeds the
    store's times) FAILS LOUD with a clear, actionable error (vs a cryptic xarray
    ``IndexError`` mid-load).  Empty ``time_indices`` ⇒ raise."""
    indices = list(time_indices)
    if not indices:
        raise ValueError("load_era5_time_mean: time_indices must be non-empty.")

    def _load(i):
        try:
            return load_era5_slice(config, int(i), ds=ds)
        except IndexError as e:
            raise ValueError(
                f"load_era5_time_mean: ERA5 time index {int(i)} is out of range — the "
                "requested time window exceeds the store's available times (campaign "
                "CLI: lower --era5-n-times or --era5-time-idx). "
                f"Underlying: {e}") from e

    if len(indices) == 1:
        return _load(indices[0])
    data_fields = ("T", "u", "v", "q", "p_s", "sst", "phis")
    first = None
    acc = None
    for i in indices:
        sl = _load(i)
        if acc is None:
            first = sl
            acc = {name: getattr(sl, name).astype(np.float64) for name in data_fields}
        else:
            for name in data_fields:
                acc[name] = acc[name] + getattr(sl, name)   # running sum (slice discarded)
    n = float(len(indices))
    return first._replace(**{
        name: (acc[name] / n).astype(getattr(first, name).dtype) for name in data_fields})


def _era5_held_fluxes(era5: ERA5Slice, regrid_2d_fn, shape_2d):
    """Regrid ERA5 radiation-flux targets onto the model grid for the carry.

    Returns ``(held_sw_up_toa, held_lw_up_toa, held_sw_net_sfc,
    held_lw_net_sfc)`` — i.e. (rsut, OLR, surface net SW, surface net LW) —
    each mapped to the model 2D layout by ``regrid_2d_fn`` (a grid-specific
    callback: Gaussian/lat-lon interpolation or cubed-sphere ``regrid_scalar``).
    When the slice carries no fluxes (``load_radiation_fluxes=False``) returns
    four ``jnp.zeros(shape_2d)`` — byte-identical to the legacy zero-fill.

    ``held_sw_down_toa`` (rsdt) is intentionally NOT set from ERA5: it is
    prescribed insolation that the dycore computes each step, and the flux
    loss does not penalize it.  One shared implementation so the spectral /
    lat-lon / cubed-sphere builders never re-derive flux-target packing.
    """
    if era5.rsut is None:
        z = jnp.zeros(shape_2d)
        return z, z, z, z
    return (
        jnp.asarray(regrid_2d_fn(era5.rsut)),
        jnp.asarray(regrid_2d_fn(era5.olr)),
        jnp.asarray(regrid_2d_fn(era5.sfc_net_sw)),
        jnp.asarray(regrid_2d_fn(era5.sfc_net_lw)),
    )


def prognostic_carry_seeds(
    microphysics: str,
    turbulence: str,
    shape_3d,
):
    """Extra ``pack_carry`` kwargs seeding the conditional prognostic carries.

    The warm-rain carry (``q_v``/``q_c``/``q_r`` + diagnostic turbulence)
    needs nothing beyond the three microphysics slots every ERA5 carry
    already passes, so for ``kessler``/``sundqvist`` + a diagnostic
    turbulence scheme this returns ``{}`` and the carry pytree is
    BYTE-IDENTICAL to the legacy warm-rain carry (``q_i``…``N_i``/``tke``/
    ``qke`` stay ``None`` ⇒ the ``lax.scan`` carry structure and the
    ``_dm_upd(None tendency)`` path are unchanged).  Seeding non-``None``
    extras would flip the carry structure, so the warm-rain branch must
    return ``{}``.

    Two conditions add seeds (mirroring the production driver's
    ``ModelDriver`` IC seeding, model_driver.py): a double-moment /
    bin microphysics scheme that writes more than the three warm-rain
    tracer slots gets ``q_i``…``N_i`` seeded as ``jnp.zeros(shape_3d)``;
    a STATEFUL turbulence scheme (``carries_energy`` — tke / mynn25 /
    clubb* / edmf) gets its prognostic energy carry (``tke`` or ``qke``)
    seeded as ``jnp.zeros(shape_3d)``.  All double-moment schemes guard
    their mean-size / fall-speed divides with ``jnp.where(q>eps,…)`` /
    ``safe_divide`` / number floors, so a zero seed is forward- and
    gradient-safe (produces zero tendencies at ``t=0``).

    Parameters
    ----------
    microphysics : str
        Microphysics scheme name (``ExperimentConfig.microphysics``).
    turbulence : str
        Turbulence scheme name (``ExperimentConfig.turbulence``).
    shape_3d : tuple
        Model 3-D field shape ``(..., nlev)`` — the shape of ``q_v``.

    Returns
    -------
    dict
        Keyword arguments to splat into :func:`pack_carry`.
    """
    from legoesm.driver.physics_pipeline import (
        required_microphysics_tracer_slots,
    )
    from legoesm.atmosphere.physics.turbulence.integration import (
        turbulence_scheme_traits,
    )

    seeds: dict = {}

    # Double-moment / bin microphysics: seed the hydrometeor + number
    # carries the scheme writes beyond the warm-rain [q_v, q_c, q_r]
    # slots.  Slot layout (see validate_microphysics_tracer_slots):
    # [3]=q_i [4]=q_s [5]=q_g [6]=N_c [7]=N_r [8]=N_i.  ``>3`` is the
    # warm-rain guard: kessler / sundqvist (3 slots) and SDM's default
    # condensation-only path (2 slots) keep these ``None``.
    if required_microphysics_tracer_slots(microphysics) > 3:
        for _name in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
            seeds[_name] = jnp.zeros(shape_3d)

    # Stateful turbulence: seed the prognostic energy carry (tke or qke).
    # Diagnostic schemes (louis / smagorinsky / ysu / holtslag_boville /
    # vreman) report carries_energy=False ⇒ no seed, carry unchanged.
    # NB the tke/qke carry is stored FLATTENED per-column (ncol, nlev) — NOT
    # the grid-shaped (n_lat, n_lon, nlev) layout the microphysics tracers
    # use — so a grid-shaped seed fails the scheme's carry-shape check
    # (issue #405/#413: "expected (ncol, nlev)").  ncol = product of the
    # horizontal dims (n_lat*n_lon for lat-lon, 6*n*n for cubed-sphere).
    _traits = turbulence_scheme_traits(turbulence)
    if _traits.carries_energy:
        _ncol = int(np.prod(shape_3d[:-1]))
        seeds[_traits.energy_field] = jnp.zeros((_ncol, shape_3d[-1]))

    return seeds


def _fill_below_ground(field_plev, plev_Pa, p_s):
    """Replace BELOW-GROUND pressure levels with the lowest above-ground value.

    A pressure-level reanalysis still carries values at levels that lie under
    the terrain — 1000 hPa over a 950 hPa plateau — and they are an
    extrapolation, not a measurement.  The vertical interpolation holds the
    lowest source level constant downward, so those fill values would be blended
    into the model's near-surface levels: over every elevated land point the
    initial condition would gain cloud water that ERA5 never reported there.

    Each column's below-ground levels are overwritten with the value at its
    lowest ABOVE-ground level, which is what the interpolation would have done
    if the fill levels simply did not exist.  Applied to the CONDENSATE only:
    humidity and temperature have always travelled the unmasked path, and
    changing them here would move an existing result for a reason unrelated to
    cloud water.
    """
    plev = jnp.asarray(plev_Pa)                       # (n_plev,) ascending
    above = plev <= jnp.asarray(p_s)[..., None]       # (..., n_plev)
    field = jnp.asarray(field_plev)
    # Ascending pressure ⇒ the last True is the lowest above-ground level.
    # A column entirely below ground (p_s under the top level) cannot happen
    # for a real surface pressure, but clip keeps the gather in range anyway.
    last = jnp.clip(jnp.sum(above, axis=-1) - 1, 0, plev.shape[0] - 1)
    lowest = jnp.take_along_axis(field, last[..., None], axis=-1)
    return jnp.where(above, field, lowest)


def _with_ice(seeds: dict, q_i_model) -> dict:
    """Put the ERA5 cloud ice into the carry's ice slot, when it has one.

    The crystal NUMBER is seeded with it.  Ice mass with a zero number is not
    merely incomplete: radiation runs before microphysics, and it diagnoses the
    crystal size by inverting the ice size distribution, so a zero number gives
    an effective radius of hundreds of metres — finite, plausible-looking, and
    radiatively almost inert.  The number that puts the crystals at the
    radiation module's own default size comes from that module, so the two
    stay each other's inverse.
    """
    if "q_i" in seeds:
        from legoesm.atmosphere.physics.clouds.cloud_fraction import (
            initial_ice_number_from_mass,
        )
        seeds["q_i"] = q_i_model
        if "N_i" in seeds:
            seeds["N_i"] = initial_ice_number_from_mass(q_i_model)
    return seeds


def _condensate_model_fields(era5, regrid, vinterp, q_spec_model, shape_3d,
                             microphysics):
    """Cloud liquid and cloud ice on model levels, as dry-air mixing ratios.

    ``regrid`` and ``vinterp`` are the builder's OWN horizontal regrid and its
    pressure-to-sigma interpolation, applied in that order, so the condensate
    travels the identical path as humidity.  They are separate arguments
    because the order matters twice over: the below-ground mask needs the
    SOURCE grid (it compares source pressure levels against the source surface
    pressure), while the vertical interpolation needs the MODEL grid (it
    interpolates against the model's surface pressure).  Handing a source-grid
    field to the vertical step is a shape error at best and a silent
    mispairing at worst.

    Returns ``(q_c, q_i)``, both zeros when the slice carries no condensate
    (``config.load_cloud_condensate`` off, or a store that has none).

    The cloud-ice NUMBER is seeded with the ice mass (see :func:`_with_ice`);
    the cloud-DROPLET number is deliberately left at zero, and the asymmetry is
    not an oversight.  Morrison reads its own specified constant droplet number
    wherever the prognostic one is unphysical, so the microphysics is already
    correct.  Radiation is not: it reads the tracer number, and with zero it
    falls back on its droplet-size clip, giving the seeded liquid an effective
    radius of 40 um on the first radiation call instead of the configured 10 um
    (measured) — about four times too little extinction, for one call, until
    activation fills the number in.  Seeding the scheme's constant instead
    would fix that and introduce something worse: that constant is a TRAINABLE
    parameter, so the initial condition would become a stale function of a
    value training moves, baked in before the first step and never updated.
    The ice has no such problem — its seed comes from a fixed radiation
    constant, not a trainable one.

    A scheme with no ice slot cannot hold ``q_i`` at all.  The ice is then
    DROPPED, with a warning naming how much: handing it to the liquid slot would
    put supercooled water at 220 K into a scheme with no ice physics, and
    silently discarding it is the exact failure this option exists to remove.
    """
    if era5.q_c is None or era5.q_i is None:
        return jnp.zeros(shape_3d), jnp.zeros(shape_3d)
    from legoesm.driver.physics_pipeline import (
        required_microphysics_tracer_slots,
    )

    def _to_model(field_native):
        # mask on the SOURCE grid, then regrid, then to model levels.
        return vinterp(regrid(
            _fill_below_ground(field_native, era5.plev_Pa, era5.p_s)))

    q_c = specific_condensate_to_mixing_ratio(_to_model(era5.q_c), q_spec_model)
    q_i = specific_condensate_to_mixing_ratio(_to_model(era5.q_i), q_spec_model)
    if required_microphysics_tracer_slots(microphysics) <= 3:
        logger.warning(
            "ERA5 cloud ice DROPPED: microphysics %r carries no ice slot "
            "(mean ice mixing ratio in the discarded field: %.3e kg/kg). "
            "Select an ice-capable scheme to use it.",
            microphysics, float(jnp.mean(q_i)))
        q_i = jnp.zeros(shape_3d)
    return q_c, q_i

def era5_to_spectral_carry(
    era5: ERA5Slice,
    grid,
    sigma,
    microphysics: str = "none",
    turbulence: str = "none",
    smoothing_passes: int = 4,
):
    """Convert ERA5 slice to SegmentCarry on a spectral (Gaussian) grid.

    The ERA5 lat-lon data is directly analyzed into spherical harmonics
    (since the Gaussian grid is also lat-lon), then vertically
    interpolated to model sigma levels.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : GaussianGrid
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    SegmentCarry
    """
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry

    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
    sigma_full = np.asarray(sigma.sigma_full)

    # ERA5 is on 1440x721 lat-lon; regrid to Gaussian grid via simple
    # nearest-neighbor or linear interpolation in lat-lon space
    T_ll, u_ll, v_ll, q_ll, p_s_ll = regrid_latlon_to_gaussian(
        era5, grid,
    )

    # Smooth the regridded ERA5 orography + hydrostatically reconcile p_s —
    # the SAME treatment the cube / lat-lon / MPAS carries already apply
    # (mirrors era5_to_latlon_carry; the spectral Gaussian grid IS a lat-lon
    # grid in grid space, so smooth_phis_gaussian applies directly).  Raw
    # regridded ERA5 phis (peaks ~5.6e4 m^2/s^2) with an unreconciled p_s
    # drives an unbalanced pressure-gradient force at step ~0; the barometric
    # correction + hybrid p_s floor live in the SHARED
    # _apply_phis_hydrostatic_adjustment (not re-implemented here).
    from legoesm.grids.topography import smooth_phis_gaussian
    phis_ll_raw = regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid)
    phis_ll_smooth = smooth_phis_gaussian(
        phis_ll_raw, smoothing_passes=smoothing_passes)
    # Reconcile against the terrain the DYNAMICS actually feel: the spectral
    # core reads phis only through its truncation (carry_to_spectral_state:
    # phis_hat = sh_analysis(phis)), so the effective surface is the
    # ROUND-TRIPPED field, Gibbs ringing included — not the grid-space
    # smoothed one.  Reconciling to the grid-space field left every ingested
    # state ~850 Pa RMS off the model's balanced manifold; the model adjusted
    # there within one 1800 s step, and because targets ride this same
    # ingestion, that standing gap was 87% of the WB training loss.  Measured
    # 2026-08-26 over 8 seasonal scenes: the one-step ps adjustment matches
    # the barometric response to (grid phis − round-tripped phis) at
    # correlation +0.996 per scene (+0.998 mean field, 46 Pa unexplained of
    # 836).  The truncation is idempotent, so the carry's phis and the
    # spectral core's phis_hat now describe the same surface.
    from legoesm.grids.gaussian import sh_analysis, sh_synthesis
    phis_ll_model = sh_synthesis(
        grid, sh_analysis(grid, jnp.asarray(phis_ll_smooth, jnp.float64)))
    # T_sfc proxy = ERA5 T at the highest pressure level (plev_Pa ascending →
    # last index = nearest to surface), matching the lat-lon carry.
    _T_sfc_ll = jnp.asarray(T_ll)[..., -1]
    phis_jax, p_s_jax = _apply_phis_hydrostatic_adjustment(
        jnp.asarray(phis_ll_raw), phis_ll_model,
        jnp.asarray(p_s_ll), _T_sfc_ll, sigma, _is_hybrid,
    )

    # Vertical interpolation: pressure levels → model levels.
    # Use TRUE hybrid pressure p(k) = A(k)*p_ref + B(k)*p_s to avoid
    # the sigma approximation error over steep terrain (see cubed-sphere
    # path comment for details).  Uses the RECONCILED p_s from above.
    plev = jnp.asarray(era5.plev_Pa)
    sigma_f = jnp.asarray(sigma_full)
    # Model TRUE full-level pressures (hybrid-correct; iter 339): interp the
    # ERA5 reference to these, not pure-sigma sigma*p_s, so it lands on the model's
    # actual levels.  Pure-sigma: pressure_at_full == sigma*p_s (byte-identical).
    p_full = sigma.pressure_at_full(p_s_jax)

    T_model = interp_pressure_to_sigma(jnp.asarray(T_ll), plev, p_s_jax, sigma_f, p_full=p_full)
    u_model = interp_pressure_to_sigma(jnp.asarray(u_ll), plev, p_s_jax, sigma_f, p_full=p_full)
    v_model = interp_pressure_to_sigma(jnp.asarray(v_ll), plev, p_s_jax, sigma_f, p_full=p_full)
    # ERA5 q is SPECIFIC HUMIDITY (mass vapor / mass moist air); the legoesm
    # physics path treats q_v as MASS MIXING RATIO (mass vapor / mass dry air —
    # the convention saturation_mixing_ratio + the physics modules consume).
    # Convert at the ERA5 boundary via the CANONICAL thermo helper r = q/(1−q)
    # (clips q below 1 to guard the division).  In the tropical PBL (q ≈ 0.025)
    # the bias from skipping this conversion is ~3% of q.
    q_spec_model = interp_pressure_to_sigma(
        jnp.asarray(q_ll), plev, p_s_jax, sigma_f, p_full=p_full)
    q_model = specific_humidity_to_mixing_ratio(q_spec_model)

    # Build HydrostaticState (phis_jax / p_s_jax already smoothed + reconciled)
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    state = HydrostaticState(
        u=Field(u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(p_s_jax, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(phis_jax, name="phis", dims=dims_2d, units="m2/s2"),
    )

    # Pack into SegmentCarry (held flux targets from ERA5 when loaded,
    # zeros otherwise — see _era5_held_fluxes)
    shape_3d = T_model.shape
    shape_2d = p_s_jax.shape

    rsut_m, olr_m, snsw_m, snlw_m = _era5_held_fluxes(
        era5, lambda f: regrid_2d_to_gaussian(f, era5.lat, era5.lon, grid), shape_2d,
    )
    q_c_model, q_i_model = _condensate_model_fields(
        era5,
        lambda f: regrid_3d_to_gaussian(f, era5.lat, era5.lon, grid),
        lambda f: interp_pressure_to_sigma(
            jnp.asarray(f), plev, p_s_jax, sigma_f, p_full=p_full),
        q_spec_model, shape_3d, microphysics,
    )
    return pack_carry(
        state,
        q_v=q_model,
        q_c=q_c_model,
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=snsw_m,
        held_lw_net_sfc=snlw_m,
        held_sw_up_toa=rsut_m,
        held_lw_up_toa=olr_m,
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
        **_with_ice(prognostic_carry_seeds(microphysics, turbulence, shape_3d),
                    q_i_model),
    )


def era5_to_cubedsphere_carry(
    era5: ERA5Slice,
    grid,
    sigma,
    target_phis=None,
    microphysics: str = "none",
    turbulence: str = "none",
    smoothing_passes: int = 4,
    edge_blend_strength: float = 0.3,
):
    """Convert ERA5 slice to SegmentCarry on a cubed-sphere grid.

    Uses KD-tree regridding from lat-lon to cubed-sphere, then
    vertical interpolation to model sigma levels.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : CubedSphereGrid
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
    target_phis : array-like, optional
        Accepted for caller compatibility (the driver passes the model's
        ETOPO surface geopotential here).  CURRENTLY NOT APPLIED: the IC
        dynamics are initialised on the *smoothed ERA5* orography below
        (the validated behaviour — job 25918469), while the CMOR ``orog``
        field separately reports the ETOPO mountain mask
        (``model_driver._setup_diagnostics``).  Wiring this through to place
        the dynamics on ``target_phis`` (with a barometric p_s adjustment
        from ERA5 orography to ETOPO) would change the IC and is a
        deliberate, revalidation-gated change intentionally NOT made here.

    Returns
    -------
    SegmentCarry
    """
    # ``target_phis`` is intentionally unused — see the parameter docstring.
    del target_phis
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.grids.regridding import regrid_scalar

    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
    sigma_full = np.asarray(sigma.sigma_full)
    # Build the regrid weights from the ACTUAL ERA5 lat/lon grid (not a Gaussian
    # proxy — see _get_cs_weights / compute_latlon_to_cs_weights for the bug fixed).
    weights = _get_cs_weights(era5.lat, era5.lon, grid)

    # Regrid 3D fields
    def _regrid_3d(field_ll):
        """Regrid (n_lat, n_lon, n_plev) → (6, n, n, n_plev)."""
        # Flatten spatial dims for regrid_scalar
        flat = field_ll.reshape(-1, field_ll.shape[-1])
        return regrid_scalar(jnp.asarray(flat), weights)

    T_cs = _regrid_3d(era5.T)
    q_cs = _regrid_3d(era5.q)

    # Regrid winds then rotate from geographic (east, north) to local panel
    # (x, y) coordinates.  ERA5 u/v are in geographic frame; the cubed-sphere
    # dycore expects winds in the local panel frame.  On equatorial faces the
    # angle is ~0 so the rotation is a no-op; on polar faces it can be ±90°,
    # which is exactly where the ~100 m/s polar-vortex jet would otherwise be
    # placed in the wrong direction, triggering immediate numerical blowup.
    u_cs_geo = _regrid_3d(era5.u)  # (6, n, n, n_plev) in geographic frame
    v_cs_geo = _regrid_3d(era5.v)
    _angle = jnp.asarray(grid.angle)[..., None]  # (6, n, n, 1) → broadcasts
    _cos_a = jnp.cos(_angle)
    _sin_a = jnp.sin(_angle)
    u_cs = _cos_a * u_cs_geo + _sin_a * v_cs_geo
    v_cs = -_sin_a * u_cs_geo + _cos_a * v_cs_geo

    # Regrid 2D fields
    p_s_cs = regrid_scalar(jnp.asarray(era5.p_s.ravel()), weights)
    phis_cs_raw = regrid_scalar(jnp.asarray(era5.phis.ravel()), weights)

    # Apply topography smoothing to match load_real_topography defaults
    # (topo_smoothing=4, topo_edge_blend=0.3).  Without this, raw ERA5 phis
    # has steep gradients near cubed-sphere face boundaries: the northern
    # Tibet slope (~37°N) sits only 3-4 cells from face 1's polar edge, and
    # the Arakawa-Lamb gradient scheme amplifies face-boundary gradient errors
    # to O(dx^-1) magnitude.  With raw ERA5 phis differences of ~50 kJ/kg,
    # this creates spurious ~0.4 m/s² PGF that drives blowup in ~1–5 days
    # even from rest.
    # edge_blend_width now defaults to TopographyConfig's 2 (was silently 1);
    # driver wires smoothing_passes/edge_blend_strength from cfg.topo_*.
    from legoesm.grids.topography import smooth_phis_cubed_sphere
    phis_cs_smooth = smooth_phis_cubed_sphere(
        phis_cs_raw, smoothing_passes=smoothing_passes,
        edge_blend_strength=edge_blend_strength)

    # Hydrostatically reconcile p_s with the smoothed phis (barometric p_s
    # correction + hybrid p_s floor).  Shared with the lat-lon carry via
    # ``_apply_phis_hydrostatic_adjustment`` — see that helper for the full
    # sign-convention + barometric derivation and the degenerate-hybrid-layer
    # rationale (Tibet: 19/40 levels underground, dp = −1 Pa at the arch peak).
    # T_sfc proxy = ERA5 T at 1000 hPa (plev_Pa ascending → last index = surface).
    _T_sfc_cs = T_cs[..., -1]  # (6, n, n) — 1000 hPa, nearest to surface
    phis_cs, p_s_cs = _apply_phis_hydrostatic_adjustment(
        phis_cs_raw, phis_cs_smooth, p_s_cs, _T_sfc_cs, sigma, _is_hybrid,
    )

    # Vertical interpolation.
    # CRITICAL: for hybrid sigma-pressure coordinates, use the TRUE level
    # pressure p(k) = A(k)*p_ref + B(k)*p_s — NOT the sigma approximation
    # (A+B)*p_s.  Over steep terrain (Tibet, Andes) where p_s << p_ref, the
    # sigma approximation misplaces upper levels by 100-180 hPa, introducing
    # ~20 K temperature errors and ~70 m/s wind imbalances that cause
    # immediate numerical blowup.
    plev = jnp.asarray(era5.plev_Pa)
    sigma_f = jnp.asarray(sigma_full)
    # Model TRUE full-level pressures (hybrid-correct; iter 339): interp the
    # ERA5 reference to these, not pure-sigma sigma*p_s, so it lands on the model's
    # actual levels.  Pure-sigma: pressure_at_full == sigma*p_s (byte-identical).
    p_full = sigma.pressure_at_full(p_s_cs)
    T_model = interp_pressure_to_sigma(T_cs, plev, p_s_cs, sigma_f, p_full=p_full)
    u_model = interp_pressure_to_sigma(u_cs, plev, p_s_cs, sigma_f, p_full=p_full)
    v_model = interp_pressure_to_sigma(v_cs, plev, p_s_cs, sigma_f, p_full=p_full)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING RATIO
    # r = q/(1−q) (canonical thermo helper; see era5_to_spectral_carry).
    q_spec_model = interp_pressure_to_sigma(
        q_cs, plev, p_s_cs, sigma_f, p_full=p_full)
    q_model = specific_humidity_to_mixing_ratio(q_spec_model)
    q_c_model, q_i_model = _condensate_model_fields(
        era5, _regrid_3d,
        lambda f: interp_pressure_to_sigma(
            f, plev, p_s_cs, sigma_f, p_full=p_full),
        q_spec_model, T_model.shape, microphysics,
    )

    if logger.isEnabledFor(logging.INFO):
        import jax as _jax
        _hT = _jax.device_get(T_model)
        _hu = _jax.device_get(u_model)
        _hv = _jax.device_get(v_model)
        _hps = _jax.device_get(p_s_cs)
        _hphis = _jax.device_get(phis_cs)
        _hphis_raw = _jax.device_get(phis_cs_raw)
        _p_s_floor_val = _hybrid_p_s_floor(sigma, dp_floor=100.0) if _is_hybrid else 0.0
        logger.info(
            f"  ERA5→CS IC: T=[{float(_hT.min()):.0f},{float(_hT.max()):.0f}]K "
            f"u=[{float(_hu.min()):.0f},{float(_hu.max()):.0f}]m/s "
            f"v=[{float(_hv.min()):.0f},{float(_hv.max()):.0f}]m/s "
            f"p_s=[{float(_hps.min()):.0f},{float(_hps.max()):.0f}]Pa "
            f"phis=[{float(_hphis.min()):.0f},{float(_hphis.max()):.0f}]m2/s2 "
            f"(raw phis peak={float(_hphis_raw.max()):.0f}, "
            f"p_s_floor={_p_s_floor_val:.0f}Pa)"
        )

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    state = HydrostaticState(
        u=Field(u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(p_s_cs, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(phis_cs, name="phis", dims=dims_2d, units="m2/s2"),
    )

    shape_3d = T_model.shape
    shape_2d = p_s_cs.shape

    rsut_m, olr_m, snsw_m, snlw_m = _era5_held_fluxes(
        era5, lambda f: regrid_scalar(jnp.asarray(f.ravel()), weights), shape_2d,
    )
    return pack_carry(
        state,
        q_v=q_model,
        q_c=q_c_model,
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=snsw_m,
        held_lw_net_sfc=snlw_m,
        held_sw_up_toa=rsut_m,
        held_lw_up_toa=olr_m,
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
        **_with_ice(prognostic_carry_seeds(microphysics, turbulence, shape_3d),
                    q_i_model),
    )


_VORONOI_WEIGHT_CACHE: dict[tuple, object] = {}


def _get_voronoi_weights(src_lat_rad, src_lon_rad, mesh):
    """Cached ERA5-lat-lon → MPAS-cell inverse-distance regridding weights.

    Keyed on the CONTENT fingerprint of both the source coords AND the mesh cell
    coords (``_coord_fingerprint``), exactly as the cubed-sphere path: keying on
    shape + endpoints alone COLLIDES on two source grids that share an extent but
    differ in INTERIOR spacing (a uniform lat-lon grid vs a Gaussian grid of the
    same bounds), and an ``id(mesh)`` key is unsafe after the mesh is
    garbage-collected and its id reused.  Fingerprinting the mesh cells makes the
    cache collision- and GC-safe (see #cs / iter 109's cubed-sphere fix).
    """
    src_lat = np.asarray(src_lat_rad)
    src_lon = np.asarray(src_lon_rad)
    key = (
        _coord_fingerprint(src_lat), _coord_fingerprint(src_lon),
        _coord_fingerprint(mesh.latCell), _coord_fingerprint(mesh.lonCell),
    )
    if key not in _VORONOI_WEIGHT_CACHE:
        from legoesm.grids.regridding import compute_latlon_to_voronoi_weights
        _VORONOI_WEIGHT_CACHE[key] = compute_latlon_to_voronoi_weights(
            src_lat, src_lon,
            np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
        )
    return _VORONOI_WEIGHT_CACHE[key]


def era5_to_latlon_carry(
    era5: ERA5Slice,
    grid,
    sigma,
    microphysics: str = "none",
    turbulence: str = "none",
    smoothing_passes: int = 4,
):
    """Convert ERA5 slice to a SegmentCarry on the lat-lon C-grid.

    The driver-level lat-lon state stores u, v, T at CELL CENTRES
    ``(n_lat, n_lon, nlev)`` — the Arakawa C-grid face staggering
    (u at lon interfaces, v at lat interfaces) is internal to the
    ``CGridLatLonPrimitiveEquationModel`` step, which re-staggers the
    cell-centred winds on the first integration.  So the IC carry is
    built exactly like the spectral path: regrid the ERA5 lat-lon
    fields onto the model lat-lon grid (a plain 2-D interpolation,
    since both are lat-lon), vertically interpolate to sigma, and
    convert ERA5 specific humidity to the model's mixing-ratio
    convention.  No face interpolation is applied here — doing so
    would double-stagger against the dycore's own re-staggering.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : LatLonGrid  (exposes ``.lat`` / ``.lon`` in radians)
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate

    Returns
    -------
    SegmentCarry
    """
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry

    sigma_full = np.asarray(sigma.sigma_full)

    # ERA5 lat-lon → model lat-lon grid (reuses the generic
    # grid.lat/grid.lon interpolator shared with the Gaussian path).
    T_ll, u_ll, v_ll, q_ll, p_s_ll = regrid_latlon_to_gaussian(era5, grid)

    plev = jnp.asarray(era5.plev_Pa)

    # Smooth the regridded ERA5 orography — Gaussian-grid analogue of the cube
    # carry's smooth_phis_cubed_sphere.  Raw ERA5 phis (peaks ~5.6e4 m^2/s^2)
    # regridded to a coarse 2° lat-lon grid drives an unbalanced
    # pressure-gradient force that blows up the dycore at step ~0; smoothing +
    # the hydrostatic p_s reconciliation below is the SAME treatment the cube
    # carry already applies (factored into _apply_phis_hydrostatic_adjustment).
    from legoesm.grids.topography import smooth_phis_gaussian
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
    phis_ll_raw = regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid)
    phis_ll_smooth = smooth_phis_gaussian(
        phis_ll_raw, smoothing_passes=smoothing_passes)

    # Hydrostatically reconcile p_s with the smoothed phis (+ hybrid p_s floor).
    # T_sfc proxy = ERA5 T at 1000 hPa (plev_Pa ascending → last index = surface).
    _T_sfc_ll = jnp.asarray(T_ll)[..., -1]
    phis_jax, p_s_jax = _apply_phis_hydrostatic_adjustment(
        jnp.asarray(phis_ll_raw), jnp.asarray(phis_ll_smooth),
        jnp.asarray(p_s_ll), _T_sfc_ll, sigma, _is_hybrid,
    )

    # Vertical interpolation (hybrid-aware, mirroring the cube/Gaussian paths).
    # CRITICAL for HybridSigmaPressureCoordinate: use the TRUE level pressure
    # p(k) = A(k)*p_ref + B(k)*p_s, NOT the (A+B)*p_s sigma approximation, which
    # over steep terrain misplaces upper levels by 100-180 hPa.
    if _is_hybrid:
        _A = jnp.asarray(sigma.A_full)
        _B = jnp.asarray(sigma.B_full)
        _p_ref = float(sigma.p_ref)

        def _vinterp(f):
            return interp_pressure_to_hybrid(
                jnp.asarray(f), plev, p_s_jax, _A, _B, _p_ref)
    else:
        sigma_f = jnp.asarray(sigma_full)

        def _vinterp(f):
            return interp_pressure_to_sigma(jnp.asarray(f), plev, p_s_jax, sigma_f)

    T_model = _vinterp(T_ll)
    u_model = _vinterp(u_ll)
    v_model = _vinterp(v_ll)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING RATIO
    # r = q/(1−q) (canonical thermo helper; consistent with the spectral,
    # Gaussian, and cube carries above — #565 unified this path onto it).
    q_spec_model = _vinterp(q_ll)
    q_model = specific_humidity_to_mixing_ratio(q_spec_model)
    q_c_model, q_i_model = _condensate_model_fields(
        era5,
        lambda f: regrid_3d_to_gaussian(f, era5.lat, era5.lon, grid),
        _vinterp, q_spec_model, T_model.shape, microphysics)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    state = HydrostaticState(
        u=Field(u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(p_s_jax, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(phis_jax, name="phis", dims=dims_2d, units="m2/s2"),
    )

    shape_3d = T_model.shape
    shape_2d = p_s_jax.shape
    rsut_m, olr_m, snsw_m, snlw_m = _era5_held_fluxes(
        era5, lambda f: regrid_2d_to_gaussian(f, era5.lat, era5.lon, grid), shape_2d,
    )
    return pack_carry(
        state,
        q_v=q_model,
        q_c=q_c_model,
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=snsw_m,
        held_lw_net_sfc=snlw_m,
        held_sw_up_toa=rsut_m,
        held_lw_up_toa=olr_m,
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
        **_with_ice(prognostic_carry_seeds(microphysics, turbulence, shape_3d),
                    q_i_model),
    )


class MPASCarry(NamedTuple):
    """ERA5 initial condition on an MPAS/Voronoi mesh.

    Unlike the cube/lat-lon ``SegmentCarry`` (which stores cell-centred
    ``u``/``v``), the MPAS hydrostatic state carries the horizontal wind as
    the **edge-normal** component on mesh edges, and has no separate ``v``.
    """
    u: jnp.ndarray      # (nEdges, nlev) edge-normal velocity [m/s]
    T: jnp.ndarray      # (nCells, nlev) temperature [K]
    p_s: jnp.ndarray    # (nCells,) surface pressure [Pa]
    phis: jnp.ndarray   # (nCells,) surface geopotential [m^2/s^2]
    q_v: jnp.ndarray    # (nCells, nlev) water vapour mixing ratio [kg/kg]


def era5_to_mpas_carry(
    era5: ERA5Slice,
    mesh,
    sigma,
    smoothing_passes: int = 4,
):
    """Convert an ERA5 slice to an initial condition on an MPAS/Voronoi mesh.

    The MPAS hydrostatic dycore stores the prognostic horizontal velocity as
    the **edge-normal** component ``u`` on mesh edges (no cell-centred ``v``),
    with scalars (``T``/``p_s``/``phis``/tracers) at cell centres.  This
    builds that state from ERA5:

    1. Horizontal regrid (KD-tree inverse-distance) of the regular ERA5
       lat-lon fields onto the mesh — scalars to cell centres
       (``latCell``/``lonCell``), winds to edge midpoints
       (``latEdge``/``lonEdge``).
    2. Edge-normal projection of the geographic ERA5 (east, north) winds
       via ``angleEdge`` (eastward = 0):
       ``u_n = u_east·cos(angleEdge) + v_north·sin(angleEdge)`` — the inverse
       of the ``reconstruct_cell_velocity`` the physics uses.
    3. Vertical interpolation from ERA5 pressure levels to the model
       sigma/hybrid levels (true level pressure ``A·p_ref + B·p_s`` for
       hybrid), with a hybrid ``p_s`` floor over high terrain.
    4. Specific humidity → mixing ratio (``r = q/(1−q)``), matching the
       cube/lat-lon convention.

    Parameters
    ----------
    era5 : ERA5Slice
    mesh : VoronoiMesh
        Exposes ``latCell``/``lonCell``/``latEdge``/``lonEdge``/``angleEdge``
        (radians), ``nCells``/``nEdges``, and the cell adjacency
        ``cellsOnCell``/``nEdgesOnCell`` used for phis smoothing.
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
    smoothing_passes : int
        Laplacian smoothing passes applied to the regridded ERA5 surface
        geopotential before it is used as ``phis`` (default 4, matching the
        cubed-sphere path).  ``0`` disables smoothing.  Raw ERA5 phis on a
        coarse Voronoi mesh produces O(dx^-1) spurious pressure-gradient force
        over steep terrain that drives a wind runaway / blowup within days.

    Returns
    -------
    MPASCarry
    """
    from legoesm.grids.regridding import (
        compute_latlon_to_voronoi_weights, regrid_scalar,
    )
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate

    _is_hybrid = isinstance(sigma, HybridSigmaPressureCoordinate)
    sigma_full = np.asarray(sigma.sigma_full)

    era5_lat = np.asarray(era5.lat)  # radians
    era5_lon = np.asarray(era5.lon)  # radians

    # Horizontal regrid weights: ERA5 lat-lon -> cell centres / edge midpoints.
    cell_w = compute_latlon_to_voronoi_weights(
        era5_lat, era5_lon,
        np.asarray(mesh.latCell), np.asarray(mesh.lonCell),
    )
    edge_w = compute_latlon_to_voronoi_weights(
        era5_lat, era5_lon,
        np.asarray(mesh.latEdge), np.asarray(mesh.lonEdge),
    )

    # Scalars at cells (nCells, n_plev) / (nCells,).
    T_cell = regrid_scalar(jnp.asarray(era5.T), cell_w)
    q_cell = regrid_scalar(jnp.asarray(era5.q), cell_w)
    p_s_cell = regrid_scalar(jnp.asarray(era5.p_s), cell_w)
    phis_cell = regrid_scalar(jnp.asarray(era5.phis), cell_w)

    # Winds at edge midpoints (geographic east/north), then project to the
    # edge normal.  ERA5 u/v are in the geographic frame; angleEdge gives the
    # edge-normal direction relative to local east.
    u_east_edge = regrid_scalar(jnp.asarray(era5.u), edge_w)   # (nEdges, n_plev)
    v_north_edge = regrid_scalar(jnp.asarray(era5.v), edge_w)
    angle = jnp.asarray(mesh.angleEdge)[:, None]               # (nEdges, 1)
    u_n_edge = u_east_edge * jnp.cos(angle) + v_north_edge * jnp.sin(angle)

    # Surface pressure at edges (for placing the wind levels over terrain).
    p_s_edge = regrid_scalar(jnp.asarray(era5.p_s), edge_w)

    # Smooth the raw regridded ERA5 surface geopotential on the mesh before
    # using it as phis.  Raw ERA5 phis retains grid-scale roughness over steep
    # terrain (Himalaya/Andes/Antarctica); the TRiSK pressure-gradient
    # amplifies those cell-to-cell gradients to O(dx^-1) spurious force, which
    # drives a localized wind runaway / blowup within days from the ERA5 IC.
    # This mirrors the cubed-sphere path (era5_to_cubedsphere_carry), which was
    # already hardened against the identical failure.  A barometric p_s
    # correction keeps each column hydrostatically consistent with the (lowered)
    # terrain gradients: smoothing lowers dB_dx, so without raising p_s where
    # terrain was smoothed down, the split-PGF correction term would no longer
    # cancel.  p_s_new = p_s · exp[(phis_raw − phis_smooth) / (R_d · T_sfc)],
    # from hydrostatic Δln_p = −ΔΦ / (R_d · T); T_sfc proxy = ERA5 T at 1000 hPa
    # (plev ascending -> last index).
    if smoothing_passes > 0:
        from legoesm.grids.topography import smooth_phis_voronoi
        phis_cell_raw = phis_cell
        phis_cell = smooth_phis_voronoi(
            phis_cell, mesh.cellsOnCell, mesh.nEdgesOnCell,
            smoothing_passes=smoothing_passes,
        )
        _T_sfc = T_cell[..., -1]
        _delta_phis = phis_cell_raw - phis_cell  # > 0 where terrain was lowered
        p_s_cell = p_s_cell * jnp.exp(_delta_phis / (constants.R_d * _T_sfc))

    # Hybrid p_s floor over high terrain: raise p_s where the hybrid layers
    # would become degenerate (dp < dp_floor), and lower phis by the
    # barometric equivalent so the split-PGF cancellation is preserved.
    if _is_hybrid:
        p_s_floor = _hybrid_p_s_floor(sigma, dp_floor=100.0)
        _T_sfc = T_cell[..., -1]  # 1000 hPa (plev ascending -> last index)
        _ln_ratio = jnp.maximum(0.0, jnp.log(p_s_floor / p_s_cell))
        phis_cell = phis_cell - constants.R_d * _T_sfc * _ln_ratio
        p_s_cell = jnp.maximum(p_s_cell, p_s_floor)
        p_s_edge = jnp.maximum(p_s_edge, p_s_floor)

    # Vertical interpolation to model levels.
    plev = jnp.asarray(era5.plev_Pa)
    if _is_hybrid:
        _A = jnp.asarray(sigma.A_full)
        _B = jnp.asarray(sigma.B_full)
        _p_ref = float(sigma.p_ref)

        def _vinterp_cell(field):
            return interp_pressure_to_hybrid(field, plev, p_s_cell, _A, _B, _p_ref)

        def _vinterp_edge(field):
            return interp_pressure_to_hybrid(field, plev, p_s_edge, _A, _B, _p_ref)
    else:
        sigma_f = jnp.asarray(sigma_full)

        def _vinterp_cell(field):
            return interp_pressure_to_sigma(field, plev, p_s_cell, sigma_f)

        def _vinterp_edge(field):
            return interp_pressure_to_sigma(field, plev, p_s_edge, sigma_f)

    T_model = _vinterp_cell(T_cell)
    u_model = _vinterp_edge(u_n_edge)
    # ERA5 q is SPECIFIC HUMIDITY; legoesm physics expects MIXING RATIO.
    q_specific = jnp.clip(jnp.maximum(_vinterp_cell(q_cell), 0.0), 0.0, 0.99)
    q_model = q_specific / (1.0 - q_specific)

    if logger.isEnabledFor(logging.INFO):
        import jax as _jax
        _hT = _jax.device_get(T_model)
        _hu = _jax.device_get(u_model)
        _hps = _jax.device_get(p_s_cell)
        _hphis = _jax.device_get(phis_cell)
        logger.info(
            f"  ERA5->MPAS IC: T=[{float(_hT.min()):.0f},{float(_hT.max()):.0f}]K "
            f"u_n=[{float(_hu.min()):.0f},{float(_hu.max()):.0f}]m/s "
            f"p_s=[{float(_hps.min()):.0f},{float(_hps.max()):.0f}]Pa "
            f"phis=[{float(_hphis.min()):.0f},{float(_hphis.max()):.0f}]m2/s2"
        )

    return MPASCarry(
        u=u_model,
        T=T_model,
        p_s=p_s_cell,
        phis=phis_cell,
        q_v=q_model,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def regrid_latlon_to_gaussian(era5: ERA5Slice, grid):
    """Regrid ERA5 lat-lon fields to the model's Gaussian grid.

    Uses scipy linear interpolation for simplicity.  This runs at
    init time (outside JIT), so numpy/scipy are fine.

    Returns T, u, v, q (n_lat, n_lon, n_plev) and p_s (n_lat, n_lon)
    on the Gaussian grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    era5_lat = np.asarray(era5.lat)  # radians
    era5_lon = np.asarray(era5.lon)  # radians
    gauss_lat = np.asarray(grid.lat)  # radians
    gauss_lon = np.asarray(grid.lon)  # radians

    # Build target mesh
    lat_g, lon_g = np.meshgrid(gauss_lat, gauss_lon, indexing='ij')

    def _interp_3d(field):
        return regrid_3d_to_gaussian(field, era5_lat, era5_lon, grid)

    def _interp_2d(field):
        interp = RegularGridInterpolator(
            (era5_lat, era5_lon), field,
            method='linear', bounds_error=False, fill_value=None,
        )
        return interp((lat_g, lon_g)).astype(np.float32)

    return (
        _interp_3d(era5.T),
        _interp_3d(era5.u),
        _interp_3d(era5.v),
        _interp_3d(era5.q),
        _interp_2d(era5.p_s),
    )


def regrid_3d_to_gaussian(field_3d, era5_lat, era5_lon, grid):
    """Regrid a (n_lat, n_lon, n_plev) field from ERA5 lat-lon to Gaussian.

    Shared by :func:`regrid_latlon_to_gaussian`'s state fields and by the cloud
    condensate, so a field added later cannot reach the vertical interpolation
    still on the source grid while the surface pressure it is interpolated
    against is already on the model grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    era5_lat = np.asarray(era5_lat)
    era5_lon = np.asarray(era5_lon)
    gauss_lat = np.asarray(grid.lat)
    gauss_lon = np.asarray(grid.lon)
    lat_g, lon_g = np.meshgrid(gauss_lat, gauss_lon, indexing="ij")
    field_3d = np.asarray(field_3d)
    n_plev = field_3d.shape[-1]
    out = np.zeros((len(gauss_lat), len(gauss_lon), n_plev), dtype=np.float32)
    for k in range(n_plev):
        interp = RegularGridInterpolator(
            (era5_lat, era5_lon), field_3d[:, :, k],
            method="linear", bounds_error=False, fill_value=None,
        )
        out[:, :, k] = interp((lat_g, lon_g))
    return out


def regrid_2d_to_gaussian(field_2d, era5_lat, era5_lon, grid):
    """Regrid a 2D field from ERA5 lat-lon to Gaussian grid."""
    from scipy.interpolate import RegularGridInterpolator

    gauss_lat = np.asarray(grid.lat)
    gauss_lon = np.asarray(grid.lon)
    lat_g, lon_g = np.meshgrid(gauss_lat, gauss_lon, indexing='ij')

    interp = RegularGridInterpolator(
        (np.asarray(era5_lat), np.asarray(era5_lon)),
        np.asarray(field_2d),
        method='linear', bounds_error=False, fill_value=None,
    )
    return interp((lat_g, lon_g)).astype(np.float32)


def load_era5_ic(
    zarr_path: str,
    year: int,
    month: int = 1,
    day: int = 1,
    hour: int = 0,
) -> ERA5Slice:
    """Load a single ERA5 time slice for use as AMIP initial conditions.

    Unlike ``load_era5_slice``, this function does not require a
    ``TrainingERA5Config``.  It auto-detects available pressure levels
    from the Zarr store and selects the timestamp nearest to the
    requested date.

    Parameters
    ----------
    zarr_path : str
        Path to a local Zarr store or GCS URI containing ERA5 data.
    year, month, day, hour : int
        Target datetime for IC (default: 1 January of *year* at 00:00 UTC).

    Returns
    -------
    ERA5Slice
        Single time slice ready to pass to ``era5_to_cubedsphere_carry``
        or ``era5_to_spectral_carry``.
    """
    import pandas as pd

    ds = open_era5_zarr(zarr_path)

    # --- locate nearest time index ---
    times = ds.time.values
    try:
        target = pd.Timestamp(year=year, month=month, day=day, hour=hour)
        time_series = pd.DatetimeIndex(times)
        time_idx = int(np.argmin(np.abs(time_series - target)))
    except Exception:
        # cftime objects (e.g. noleap calendar)
        import cftime
        target_cf = cftime.datetime(year, month, day, hour)
        diffs = np.array(
            [abs((t - target_cf).total_seconds()) for t in times],
            dtype=np.float64,
        )
        time_idx = int(np.argmin(diffs))

    # --- auto-detect pressure levels ---
    level_dim = "level" if "level" in ds.dims else "pressure_level"
    levels_hPa = tuple(
        int(v) for v in sorted(ds[level_dim].values.tolist())
    )

    cfg = TrainingERA5Config(
        zarr_store=zarr_path,
        levels=levels_hPa,
        local_cache_dir="",
    )
    return load_era5_slice(cfg, time_idx)

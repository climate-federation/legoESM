"""CORE-II Normal Year Forcing (Large & Yeager 2009).

Reference
---------
Large, W. G., & Yeager, S. G. (2009). "The global climatology of an
interannually varying air-sea flux data set", Climate Dynamics 33,
341-364.

Griffies, S. M., et al. (2009). "Coordinated Ocean-ice Reference
Experiments (COREs)", Ocean Modelling 26, 1-46.

What
----
A repeating annual climatology with daily resolution suitable for
multi-century spinups (Bryan 1987 / OMIP-1). Carries the same seven
channels as ``jra55_do`` so the loader interface is interchangeable.

Like the JRA55-do loader this falls back to the deterministic
synthetic climatology when the on-disk cache is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .jra55_do import (
    OceanForcing,
    synthetic_ocean_forcing,
)
from .jra55_do import (
    _cache_dir as _jra_cache_dir,
)

logger = logging.getLogger(__name__)


def core2_nyf_cache_dir() -> Path:
    """Default CORE-II NYF cache.

    ``core2_nyf_mod`` is built from the Large & Yeager bias-corrected fields
    (U_10_MOD, V_10_MOD, T_10_MOD, Q_10_MOD, SWDN_MOD, LWDN_MOD, PRC_MOD) --
    the ones NEMO ORCA1's namelist reads.  The older ``core2_nyf`` was built
    from the RAW fields, whose equatorial winds are 33% weaker (the known
    NCEP-1 trade-wind bias the _MOD correction exists to remove); forcing with
    them gave roughly half NEMO's equatorial wind stress and a +2.5 C nino3
    bias that fell to +1.49 C on the corrected forcing at matched day 20.

    The raw cache is deliberately left on disk rather than overwritten, so an
    older run can still be reproduced by passing ``cache_dir`` explicitly.
    """
    root = _jra_cache_dir().parent
    corrected = root / "core2_nyf_mod"
    if corrected.exists():
        return corrected
    # Fall back to the raw cache only if the corrected one was never built,
    # and say so -- silently forcing with 33%-weak equatorial winds is the
    # defect this default exists to prevent.
    logger.warning(
        "CORE-II cache %s not found; falling back to the RAW-wind cache at "
        "%s. Its equatorial winds are ~33%% weaker than the fields NEMO "
        "reads. Rebuild with scripts/data/build_core2_nyf_zarr.py "
        "--wind-variant mod.", corrected, root / "core2_nyf")
    return root / "core2_nyf"


def _nino3_wind_speed(ds) -> float:
    """Time-and-area mean 10 m wind speed over nino3 (5S-5N, 150W-90W).

    Provenance only: the corrected (``_MOD``) CORE-II fields give ~6.3 m/s
    here and the raw ones ~4.7 m/s, so this single number identifies which
    cache a run was forced with.
    """
    lat = np.asarray(ds.lat.values, dtype=np.float64)
    lon = np.asarray(ds.lon.values, dtype=np.float64) % 360.0
    jj = np.where((lat >= -5.0) & (lat <= 5.0))[0]
    ii = np.where((lon >= 210.0) & (lon <= 270.0))[0]
    if jj.size == 0 or ii.size == 0:
        return float("nan")
    u = np.asarray(ds.u10.values, dtype=np.float64)[:, jj][:, :, ii]
    v = np.asarray(ds.v10.values, dtype=np.float64)[:, jj][:, :, ii]
    w = np.cos(np.deg2rad(lat[jj]))[None, :, None]
    return float((np.hypot(u, v) * w).sum() / (np.ones_like(u) * w).sum())


def core2_nyf_path(cache_dir: Path | None = None) -> Path:
    """Resolve the ``nyf.zarr`` archive :func:`load_core2_nyf` would read.

    Exposed so a caller can RECORD which archive a run actually used.  With
    ``cache_dir`` unset the location comes from :func:`core2_nyf_cache_dir`,
    i.e. from the environment / home directory, so two runs launched with
    identical command lines can read DIFFERENT forcing (codex r8):
    ``run_omip_core2`` folds this resolved path into its restart configuration
    fingerprint rather than the raw ``--forcing-path`` argument, which is
    ``None`` in exactly that case.

    Single source of truth: :func:`load_core2_nyf` resolves through this
    function, so the recorded path cannot drift from the loaded one.
    """
    root = Path(cache_dir) if cache_dir is not None else core2_nyf_cache_dir()
    return root / "nyf.zarr"


def load_core2_nyf(*, cache_dir: Path | None = None,
                   allow_synthetic: bool = True,
                   n_time: int = 365) -> OceanForcing:
    """Load the CORE-II Normal Year (NYF) climatology.

    NYF is a perpetual year, so the loader returns a single ``OceanForcing``
    with ``n_time`` daily snapshots (default 365). The seasonal cycle is
    one year long.
    """
    zarr_path = core2_nyf_path(cache_dir)
    if zarr_path.exists():
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "CORE-II NYF real-data load requires xarray + zarr"
            ) from exc
        ds = xr.open_zarr(zarr_path)
        # Provenance, printed rather than logged so it lands in every run log
        # next to the other [setup] lines.  The nino3 wind speed is the number
        # that distinguishes the corrected cache (~6.3 m/s) from the raw one
        # (~4.7 m/s), so a silent fallback is visible in the log itself.
        print(f"[forcing] CORE-II NYF cache: {zarr_path} "
              f"(nino3 mean |U10| = {_nino3_wind_speed(ds):.3f} m/s)")
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
            # NEMO-parity channels (2026-06 schema): absent on old caches.
            snow=(np.asarray(ds.snow.values, dtype=np.float64)
                  if "snow" in ds else None),
            slp=(np.asarray(ds.slp.values, dtype=np.float64)
                 if "slp" in ds else None),
        )
    if not allow_synthetic:
        raise FileNotFoundError(
            f"CORE-II NYF cache missing: {zarr_path}; populate from "
            f"https://www.earthsystemgrid.org/dataset/ucar.cgd.artmip.core2.html"
        )
    # Loud (not silent) fallback: synthetic analytic forcing is NOT the
    # OMIP-1 / CORE-II protocol — never mistake it for a real NYF run.
    logger.warning(
        "CORE-II NYF cache missing at %s — falling back to SYNTHETIC analytic "
        "forcing. This is NOT the OMIP-1 protocol; results are not "
        "OMIP-comparable. Pass allow_synthetic=False to fail loudly, or build "
        "the cache (scripts/data/build_core2_nyf_zarr.py).",
        zarr_path,
    )
    # Synthetic climatology with daily resolution.
    return synthetic_ocean_forcing(0, n_time=n_time)


__all__ = ["core2_nyf_cache_dir", "core2_nyf_path", "load_core2_nyf"]

#!/usr/bin/env python
"""Build the gridded satellite-SIF ``--sif-obs`` NetCDF for the Stage-B SIF carbon
calibration (``scripts/run/train_carbon_params.py --with-sif``).

Science-grade data-prep FOLLOW-UP promised by
:func:`legoesm.land.carbon.sif_observations.load_gridded_sif` and
:mod:`legoesm.land.carbon.sif_forward`: fetch a PUBLIC gridded satellite SIF product,
reduce it to a growing-season climatology, regrid it onto the surfdata cover grid, and
write it in the MODEL's emitted-photon-flux units so the ``sif`` calibration term compares
like with like.  Mirrors the ERA5 land-forcing builder
(``scripts/data/build_land_forcing_climatology.py``): a PURE reduction/conversion
(:func:`reduce_sif_to_climatology`, unit-tested offline) separated from the networked
fetch wrapper (:func:`_fetch_caltech_sif740`).

Output contract (what ``load_gridded_sif`` reads via ``--sif-obs``)
------------------------------------------------------------------
A NetCDF with dims ``(lat, lon)`` on the SURFDATA cover grid (read from ``--surf-path``
via the canonical :func:`legoesm.land.surface_data.sources.clm5_surfdata.read_clm5_cover_veg`
so ``nlat*nlon`` == the cover ``ncell`` in the SAME row-major orientation), one variable
``sif`` [umol m-2 s-1] (the model's emitted top-of-canopy fluorescence PHOTON FLUX, the
:func:`legoesm.land.canopy.sif.leaf_sif` unit), with NaN preserved over unobserved cells
(ocean, polar night, persistent cloud) -- ``load_gridded_sif`` excludes NaN from the
per-archetype cover-weighted mean, so a gap is NEVER fabricated to 0.

Data source + access (probed reachable from a Ginsburg compute node 2026-07-09)
------------------------------------------------------------------------------
Caltech **TROPOMI SIF at 740 nm**, gridded (Koehler & Frankenberg), streamed by anonymous
FTP from ``ftp://fluo.gps.caltech.edu/data/tropomi/gridded/SIF740/``.  The 1-degree monthly
files ``TROPOMI-SIF740nm_01-YYYY--12-YYYY_1deg_1-monthly.nc`` (dims ``(lat=180, lon=360,
time=12)``; lat -89.5..89.5, lon -179.5..179.5; ``_FillValue`` -999 -> NaN) are ~6 MB each,
so a multi-year monthly climatology is a cheap compute-node download.  The default source
variable is ``sif_dc`` -- the length-of-day / daily-corrected SIF, the community-standard
daily-representative productivity signal (``sif`` = the raw ~13:30 overpass value is
selectable via ``--source-var``).

UNITS: radiance -> emitted photon flux (documented, absolute scale absorbed)
---------------------------------------------------------------------------
The product is a spectral radiance at 740 nm [mW m-2 sr-1 nm-1]; the model emits a
broadband fluorescence PHOTON FLUX [umol m-2 s-1].  :func:`radiance_to_photon_flux` applies
a documented linear conversion -- Lambertian hemispheric integration (x pi sr), an effective
far-red emission bandwidth (740 nm spectral value -> band-integrated emission), mW->W, and
the photon energy ``E = h c / lambda`` (via :data:`legoesm.constants.h_planck` /
``c_light`` / ``N_A``) -> genuine umol m-2 s-1.  The ABSOLUTE factor is uncertain (escape
probability, emission-spectrum shape, BRDF) and -- as
:mod:`legoesm.land.canopy.sif` / :mod:`legoesm.land.carbon.sif_observations` both state --
is ABSORBED by the tier-1 calibration knobs ``max_electron_yield`` / ``escape_probability``
(fesc).  So the calibration signal is the SPATIAL PATTERN, not the absolute magnitude.  To
make the ``sif`` MSE term SCALE-ROBUST (and comparable to the SOC term), the default
``--rescale`` renormalises the converted field by a single global factor so its
area-weighted mean equals :data:`_SIF_REFERENCE_MEAN_UMOL` (the model's O(1-10) emitted-SIF
scale) -- a PATTERN-PRESERVING rescale (one multiplicative constant, spatial structure
untouched).  ``--no-rescale`` writes the raw absolute physical conversion instead.  Either
way the file is genuine umol m-2 s-1, honouring the ``load_gridded_sif`` unit contract.

Aggregation: growing season, not annual mean
--------------------------------------------
SIF is a productivity signal, so the physically-right per-cell aggregate is the LOCAL
GROWING SEASON, not an annual mean diluted by dormant / polar-night months.
:func:`growing_season_mean` averages, per cell, the months whose SIF is at least
:data:`_GROWING_SEASON_ACTIVE_FRACTION` of that cell's own annual maximum (a standard
amplitude-threshold phenology definition), NaN-aware.  ``--aggregate annual`` selects a
plain annual mean instead.  This matches the model SIF forward, which samples a
REPRESENTATIVE GROWING-SEASON canopy (``archetype_forcing.REF_DOY`` = peak insolation).

Login-node policy: the FTP fetch is a networked compute-node job -- run via ``sbatch`` /
``srun``, never on the login node.

Run (compute node):
    PYTHONPATH=$REPO:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':') \
    python scripts/data/build_sif_observations.py \
        --surf-path data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc \
        --out results/land_forcing/tropomi_sif740_growingseason_2deg.nc \
        --years 2018 2019 2020
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from contextlib import contextmanager

import numpy as np
import xarray as xr
from legoesm.grids.regridding import conservative_regrid_latlon

from legoesm import constants


@contextmanager
def _quiet_nan():
    """Suppress the EXPECTED all-NaN-slice / empty-slice / invalid-compare warnings
    from the NaN-aware reductions over fully-unobserved cells (ocean, polar night,
    persistent cloud -- ~75 % of the raw 1-deg grid).  Those cells are INTENTIONALLY
    preserved as NaN (never fabricated), so the warning is noise, not an error."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

# --- Caltech TROPOMI SIF740 gridded public store (anonymous FTP) ---
_FTP_HOST = "fluo.gps.caltech.edu"
_FTP_BASE = "/data/tropomi/gridded/SIF740"
# Per-year 1-degree monthly file; {res} in {1deg, 0_2deg, 0_0833333deg}.
_FILE_TEMPLATE = "TROPOMI-SIF740nm_01-{year}--12-{year}_{res}_1-monthly.nc"
_DEFAULT_RES = "1deg"
# Daily-corrected SIF (length-of-day corrected) is the daily-representative productivity
# signal; "sif" is the raw ~13:30 overpass value.  Both are mW/m^2/sr/nm at 740 nm.
_DEFAULT_SOURCE_VAR = "sif_dc"
# Retrieval fill sentinel: values at/below this are missing (product _FillValue -999);
# decode_cf usually masks them to NaN, but we re-mask defensively.  A guard, not a datum.
_FILL_SENTINEL = -900.0

# --- radiance -> emitted photon-flux conversion (740 nm far-red SIF) ---
# Photon energy E = h c / lambda; photon flux [mol/m^2/s] = irradiance[W/m^2] / E.
_SIF_WAVELENGTH_NM = 740.0        # TROPOMI SIF retrieval wavelength [nm]
_NM_TO_M = 1.0e-9                  # nm -> m (exact)
_MW_TO_W = 1.0e-3                  # mW -> W (exact)
_UMOL_PER_MOL = 1.0e6             # mol -> umol (exact)
# Effective far-red fluorescence-emission bandwidth: converts the 740 nm SPECTRAL radiance
# [per nm] to the band-integrated far-red emission.  ~ Gaussian sigma*sqrt(2*pi) for a
# far-red peak of FWHM ~ 25 nm (sigma = FWHM/2.355 ~= 10.6 nm -> 10.6*2.5066 ~= 26.6 nm).
# APPROXIMATE (the far-red emission shape varies); the absolute scale it sets is absorbed
# by the calibration knobs (fesc / max_electron_yield) -- only the spatial pattern is used.
_SIF_FARRED_EFFECTIVE_WIDTH_NM = 26.6

# --- growing-season aggregation + pattern-normalisation references ---
# A cell's growing season = months with SIF >= this fraction of the cell's annual max
# (amplitude-threshold phenology; White et al. 1997 use 50 % of amplitude for onset).
_GROWING_SEASON_ACTIVE_FRACTION = 0.5
# Pattern-preserving rescale target: the model's emitted-SIF photon-flux scale is O(1-10)
# umol/m^2/s (leaf_sif output; sif_observations.synthetic base ~4-8), so normalising the
# observed field's area-weighted mean here keeps the sif MSE on the SOC term's scale and
# lets fesc/max_electron_yield absorb the (uncertain) absolute conversion factor.
_SIF_REFERENCE_MEAN_UMOL = 5.0

# --- output contract ---
_OUT_VAR = "sif"                  # first name load_gridded_sif auto-detects
_OUT_DIMS = ("lat", "lon")


# ===========================================================================
# PURE reduction / conversion (no network / no I/O -> unit-tested offline)
# ===========================================================================
def photon_flux_per_watt(wavelength_nm: float) -> float:
    """umol photons / m^2 / s per (W / m^2) at ``wavelength_nm``.

    ``n_photons = irradiance / E_photon`` with ``E_photon = h c / lambda`` [J];
    in umol, ``= irradiance * lambda * 1e6 / (h c N_A)``.  Uses the exact SI
    :data:`legoesm.constants.h_planck` / ``c_light`` / ``N_A`` (no hardcoded
    fundamental constants).
    """
    lam_m = float(wavelength_nm) * _NM_TO_M
    hc_na = constants.h_planck * constants.c_light * constants.N_A   # [J·m/mol]
    return lam_m * _UMOL_PER_MOL / hc_na


def radiance_to_photon_flux(
    radiance_mw,
    *,
    wavelength_nm: float = _SIF_WAVELENGTH_NM,
    effective_width_nm: float = _SIF_FARRED_EFFECTIVE_WIDTH_NM,
):
    """SIF spectral radiance [mW m-2 sr-1 nm-1] -> emitted photon flux [umol m-2 s-1].

    Linear, per-cell, sign-preserving (a positive scale; retrieval-noise negatives stay
    negative -- never clamped/fabricated).  Factors, all documented above:

        photon_flux = radiance[mW/m^2/sr/nm]
                      * pi [sr]                 (Lambertian hemispheric integration)
                      * effective_width_nm      (740 nm spectral -> band-integrated)
                      * 1e-3                     (mW -> W)
                      * photon_flux_per_watt(lambda)   (W/m^2 -> umol/m^2/s)

    The ABSOLUTE constant is approximate and is absorbed by the calibration knobs
    (``max_electron_yield`` / ``fesc``); the calibration uses the spatial pattern.
    """
    k = (constants.PI * float(effective_width_nm) * _MW_TO_W
         * photon_flux_per_watt(wavelength_nm))
    return np.asarray(radiance_mw, dtype=np.float64) * k


def average_years(monthly_by_year) -> np.ndarray:
    """Stack of per-year ``(12, nlat, nlon)`` monthly means -> one ``(12, nlat, nlon)``
    monthly climatology, NaN-aware (a month observed in SOME years keeps that partial mean;
    a month never observed stays NaN)."""
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in monthly_by_year], axis=0)
    if stack.ndim != 4 or stack.shape[1] != 12:
        raise ValueError(
            f"monthly_by_year must stack to (n_year, 12, nlat, nlon); got {stack.shape}.")
    with _quiet_nan():
        return np.nanmean(stack, axis=0)


def growing_season_mean(
    monthly, *, active_fraction: float = _GROWING_SEASON_ACTIVE_FRACTION,
) -> np.ndarray:
    """Per-cell growing-season mean of a ``(12, nlat, nlon)`` monthly field.

    Where the cell has a POSITIVE annual maximum (a real seasonal amplitude), the growing
    season is the months whose value is at least ``active_fraction`` of that maximum (an
    amplitude-threshold phenology definition) and the result is the NaN-aware mean over
    those months.  A ``frac * max`` threshold only defines a growing season when ``max > 0``;
    for a cell whose finite months are ALL <= 0 (all-negative retrieval noise over barren /
    non-vegetated ground) there is no productive season, so it FALLS BACK to the plain
    finite-month mean -- a small near-zero / negative value that faithfully reflects "no
    productivity" rather than being silently dropped to NaN.  A cell that is all-NaN (never
    observed) stays NaN.  Nothing is floored to 0 (no fabrication); ``load_gridded_sif``
    excludes the remaining NaN gaps from the per-archetype mean.
    """
    m = np.asarray(monthly, dtype=np.float64)
    if m.ndim != 3 or m.shape[0] != 12:
        raise ValueError(f"monthly must be (12, nlat, nlon); got {m.shape}.")
    # active_fraction must be in (0, 1]: it is a fraction of the cell's annual MAX, so >1
    # would put the threshold ABOVE the max and silently drop even a positive cell to NaN
    # (has_season True but no active month); <=0 makes every finite month active (no
    # amplitude selection).  Reject out-of-range instead of dropping observed cells.
    if not (0.0 < float(active_fraction) <= 1.0):
        raise ValueError(
            f"active_fraction must be in (0, 1]; got {active_fraction}. A value >1 "
            f"thresholds above the annual max and would drop positive observed cells.")
    with _quiet_nan():
        cell_max = np.nanmax(m, axis=0)                      # (nlat, nlon); NaN if all-NaN
    thresh = float(active_fraction) * cell_max               # broadcast per cell
    active = np.isfinite(m) & (m >= thresh[None, :, :])      # (12, nlat, nlon) bool
    masked = np.where(active, m, np.nan)
    with _quiet_nan():
        gs = np.nanmean(masked, axis=0)                      # growing-season mean (max>0)
        finite_mean = np.nanmean(m, axis=0)                  # fallback = finite-month mean
    # max > 0 -> a real amplitude -> growing-season mean (finite: the max month itself is
    # always active since max >= frac*max for max >= 0).  Otherwise (all-<=0 finite, or
    # all-NaN) fall back to the finite-month mean: preserves observed-but-unproductive cells
    # (all-NaN stays NaN because finite_mean is NaN there).
    has_season = cell_max > 0.0                              # False for NaN and for max<=0
    return np.where(has_season, gs, finite_mean)


def annual_mean(monthly) -> np.ndarray:
    """Plain per-cell annual mean of a ``(12, nlat, nlon)`` field (NaN-aware)."""
    m = np.asarray(monthly, dtype=np.float64)
    if m.ndim != 3 or m.shape[0] != 12:
        raise ValueError(f"monthly must be (12, nlat, nlon); got {m.shape}.")
    with _quiet_nan():
        return np.nanmean(m, axis=0)


def area_weighted_nanmean(field, lat) -> float:
    """cos(lat)-area-weighted mean over FINITE cells of a ``(nlat, nlon)`` field.

    True spherical cell area ~ cos(lat); missing (NaN) cells are excluded from both the
    weighted sum and the weight total, so the mean is over the observed area only.
    """
    f = np.asarray(field, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] != lat.size:
        raise ValueError(
            f"field {f.shape} must be (nlat, nlon) with nlat == lat.size {lat.size}.")
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, f.shape[1]))
    finite = np.isfinite(f)
    wsum = float(np.sum(w[finite]))
    if wsum <= 0.0:
        return float("nan")
    return float(np.sum(w[finite] * f[finite]) / wsum)


def rescale_to_reference_mean(
    field, lat, *, reference_umol: float = _SIF_REFERENCE_MEAN_UMOL,
):
    """Pattern-preserving rescale so the field's area-weighted mean == ``reference_umol``.

    Returns ``(scaled_field, factor)`` where ``scaled = field * factor`` and ``factor =
    reference / area_weighted_nanmean(field)``.  A SINGLE global multiplicative constant --
    the SPATIAL PATTERN (and every NaN gap) is untouched -- so the calibration ``sif`` MSE
    fits the pattern while the absolute scale rides on ``reference_umol`` (itself absorbed
    by fesc / max_electron_yield).  Raises if the observed mean is non-finite / non-positive
    (no valid data to anchor the scale).
    """
    mean = area_weighted_nanmean(field, lat)
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(
            f"cannot rescale: observed area-weighted mean is {mean} (<=0 or non-finite); "
            f"the converted SIF field has no positive observed signal to anchor the scale.")
    factor = float(reference_umol) / mean
    return np.asarray(field, dtype=np.float64) * factor, factor


def reduce_sif_to_climatology(
    monthly_by_year,
    src_lat,
    src_lon,
    tgt_lat,
    tgt_lon,
    *,
    aggregate: str = "growing_season",
    active_fraction: float = _GROWING_SEASON_ACTIVE_FRACTION,
    wavelength_nm: float = _SIF_WAVELENGTH_NM,
    effective_width_nm: float = _SIF_FARRED_EFFECTIVE_WIDTH_NM,
    rescale: bool = True,
    reference_umol: float = _SIF_REFERENCE_MEAN_UMOL,
):
    """Pure pipeline: per-year monthly SIF radiance -> the ``--sif-obs`` Dataset.

    Steps (all NaN-aware; every gap preserved):
      1. average the per-year ``(12, nlat_s, nlon_s)`` arrays over years (:func:`average_years`);
      2. per-cell aggregate to a 2-D map on the SOURCE grid -- growing-season
         (:func:`growing_season_mean`) or ``annual`` (:func:`annual_mean`);
      3. conservatively regrid to the target grid
         (:func:`legoesm.grids.regridding.conservative_regrid_latlon`; NaN-aware,
         orientation- and longitude-convention-agnostic so the -180..180 source maps onto a
         0..360 or descending target seamlessly);
      4. convert radiance -> emitted photon flux [umol/m^2/s] (:func:`radiance_to_photon_flux`);
      5. optionally pattern-preserving rescale to ``reference_umol``
         (:func:`rescale_to_reference_mean`).

    Steps 3-4 are linear so their order is immaterial; step 2 is per-cell nonlinear (its
    own annual max) so it precedes the regrid; step 5 (a global normaliser) is last.
    Returns ``(xarray.Dataset var 'sif' (lat, lon) [umol/m^2/s], provenance dict)``.
    """
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64)
    tgt_lon = np.asarray(tgt_lon, dtype=np.float64)
    if tgt_lat.ndim != 1 or tgt_lon.ndim != 1:
        raise ValueError("tgt_lat / tgt_lon must be 1-D degree coordinate vectors.")
    if aggregate not in ("growing_season", "annual"):
        raise ValueError(
            f"aggregate must be 'growing_season' or 'annual'; got {aggregate!r}.")

    monthly = average_years(monthly_by_year)                 # (12, nlat_s, nlon_s)
    if aggregate == "growing_season":
        src_map = growing_season_mean(monthly, active_fraction=active_fraction)
    else:
        src_map = annual_mean(monthly)

    regridded = conservative_regrid_latlon(
        src_map, src_lat, src_lon, tgt_lat, tgt_lon)         # (nlat_t, nlon_t), NaN gaps
    sif = radiance_to_photon_flux(
        regridded, wavelength_nm=wavelength_nm, effective_width_nm=effective_width_nm)

    factor = 1.0
    raw_mean = area_weighted_nanmean(sif, tgt_lat)
    if rescale:
        sif, factor = rescale_to_reference_mean(
            sif, tgt_lat, reference_umol=reference_umol)

    ds = xr.Dataset(
        data_vars={_OUT_VAR: (_OUT_DIMS, np.asarray(sif, dtype=np.float64))},
        coords={"lat": ("lat", tgt_lat), "lon": ("lon", tgt_lon)},
    )
    ds[_OUT_VAR].attrs.update(
        units="umol m-2 s-1",
        long_name="emitted top-of-canopy SIF photon flux (model units)",
        note=("TROPOMI SIF740 radiance -> emitted photon flux; absolute scale absorbed by "
              "calibration fesc/max_electron_yield -- calibrate on the SPATIAL PATTERN"))
    ds["lat"].attrs.update(units="degrees_north", long_name="latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="longitude")
    provenance = {
        "aggregate": aggregate,
        "active_fraction": float(active_fraction),
        "wavelength_nm": float(wavelength_nm),
        "effective_width_nm": float(effective_width_nm),
        "photon_flux_per_watt": photon_flux_per_watt(wavelength_nm),
        "rescaled": bool(rescale),
        "reference_umol": (float(reference_umol) if rescale else None),
        "rescale_factor": float(factor),
        "raw_converted_area_mean_umol": float(raw_mean),
    }
    return ds, provenance


# ===========================================================================
# Networked fetch wrapper + surfdata target grid (NOT unit-tested offline)
# ===========================================================================
def _fetch_year_file(ftp, remote_path: str, tmpdir: str) -> str:
    local = os.path.join(tmpdir, os.path.basename(remote_path))
    with open(local, "wb") as fh:
        ftp.retrbinary(f"RETR {remote_path}", fh.write)
    return local


def _fetch_caltech_sif740(
    years,
    *,
    resolution: str = _DEFAULT_RES,
    source_var: str = _DEFAULT_SOURCE_VAR,
    host: str = _FTP_HOST,
    base: str = _FTP_BASE,
    tmpdir: str | None = None,
):
    """Fetch the Caltech TROPOMI SIF740 per-year 1-degree monthly files over anon FTP.

    Downloads ``TROPOMI-SIF740nm_01-YYYY--12-YYYY_{res}_1-monthly.nc`` for each year, reads
    ``source_var`` (dims ``(lat, lon, time=12)``, ``_FillValue`` -999 -> NaN) and moves the
    month axis to the front -> a per-year ``(12, nlat, nlon)`` list.  Thin networked wrapper
    (mirrors the ERA5 builder's ``_fetch_arco``); its only reduction is the axis move + fill
    mask, both trivial -- the science lives in the tested pure helpers.

    Returns ``(monthly_by_year, src_lat, src_lon)`` in the product's native units
    (mW/m^2/sr/nm) and grid (lat -89.5..89.5, lon -179.5..179.5).
    """
    import ftplib

    made_tmp = tmpdir is None
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="sif740_")
    monthly_by_year: list[np.ndarray] = []
    src_lat = src_lon = None
    ftp = ftplib.FTP(host, timeout=180)
    try:
        ftp.login()
        for year in years:
            fname = _FILE_TEMPLATE.format(year=int(year), res=resolution)
            remote = f"{base}/{fname}"
            print(f"# fetching {remote}")
            local = _fetch_year_file(ftp, remote, tmpdir)
            ds = xr.open_dataset(local, decode_times=False)
            try:
                if source_var not in ds:
                    raise KeyError(
                        f"source var {source_var!r} not in {fname}; "
                        f"available: {sorted(ds.data_vars)}")
                arr = np.asarray(ds[source_var].values, dtype=np.float64)  # (lat, lon, 12)
                lat = np.asarray(ds["lat"].values, dtype=np.float64)
                lon = np.asarray(ds["lon"].values, dtype=np.float64)
            finally:
                ds.close()
            if arr.ndim != 3 or arr.shape[-1] != 12:
                raise ValueError(
                    f"{fname}[{source_var}] expected (lat, lon, 12); got {arr.shape}.")
            arr = np.moveaxis(arr, -1, 0)                    # -> (12, lat, lon)
            arr = np.where(arr <= _FILL_SENTINEL, np.nan, arr)   # defensive fill mask
            if src_lat is None:
                src_lat, src_lon = lat, lon
            elif lat.size != src_lat.size or lon.size != src_lon.size:
                raise ValueError(
                    f"{fname} grid {lat.size}x{lon.size} != first year "
                    f"{src_lat.size}x{src_lon.size}; use one --resolution.")
            monthly_by_year.append(arr)
            os.remove(local)
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()
        if made_tmp:
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
    if not monthly_by_year:
        raise SystemExit("no SIF years fetched; pass --years.")
    print(f"# fetched {len(monthly_by_year)} year(s) of "
          f"({source_var}) -> (12, {src_lat.size}, {src_lon.size}) monthly")
    return monthly_by_year, src_lat, src_lon


def read_target_grid(surf_path: str):
    """1-D ``(lat, lon)`` [deg] of the surfdata cover grid via the CANONICAL
    :func:`legoesm.land.surface_data.sources.clm5_surfdata.read_clm5_cover_veg`.

    The SAME reader ``build_global_carbon_ic._load_clm5_cover_soil`` (the trainer's
    ``clm5_surfdata`` cover path) uses -- so the written SIF grid matches the cover
    ``ncell = nlat*nlon`` in the SAME orientation and ``load_gridded_sif``'s row-major
    reshape aligns cell-for-cell (no re-derived grid).
    """
    from legoesm.land.surface_data.sources.clm5_surfdata import read_clm5_cover_veg

    clm = read_clm5_cover_veg(surf_path)
    return (np.asarray(clm["lat"], dtype=np.float64),
            np.asarray(clm["lon"], dtype=np.float64))


def _print_sanity(ds: xr.Dataset, provenance: dict) -> None:
    a = np.asarray(ds[_OUT_VAR].values, dtype=np.float64)
    finite = np.isfinite(a)
    n_fin = int(finite.sum())
    print(f"# {_OUT_VAR} [umol/m2/s]: finite {n_fin}/{a.size} "
          f"({100.0 * n_fin / a.size:.1f}% land/observed)  "
          f"min {np.nanmin(a):.4g}  mean {np.nanmean(a):.4g}  max {np.nanmax(a):.4g}")
    print(f"# provenance: {provenance}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surf-path", "--surfdata", dest="surf_path", type=str, required=True,
                   help="CLM5 surfdata NetCDF defining the target cover grid "
                        "(SAME file passed to train_carbon_params.py --surf-path)")
    p.add_argument("--out", type=str,
                   default="results/land_forcing/tropomi_sif740_growingseason.nc",
                   help="output NetCDF (dims lat,lon; var 'sif' [umol/m2/s]); pass to "
                        "train_carbon_params.py --with-sif --sif-obs <this>")
    p.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020],
                   help="years to average into the monthly climatology "
                        "(Caltech SIF740 covers 2018..2021)")
    p.add_argument("--resolution", type=str, default=_DEFAULT_RES,
                   choices=("1deg", "0_2deg", "0_0833333deg"),
                   help="source-file resolution token (coarser is a smaller download; the "
                        "conservative regrid handles any source resolution)")
    p.add_argument("--source-var", type=str, default=_DEFAULT_SOURCE_VAR,
                   choices=("sif_dc", "sif"),
                   help="Caltech variable: sif_dc = daily/length-of-day corrected "
                        "(default, daily-representative); sif = raw ~13:30 overpass")
    p.add_argument("--aggregate", type=str, default="growing_season",
                   choices=("growing_season", "annual"),
                   help="per-cell temporal aggregate (default growing_season)")
    p.add_argument("--active-fraction", type=float,
                   default=_GROWING_SEASON_ACTIVE_FRACTION,
                   help="growing-season = months >= this fraction of the cell's annual max")
    p.add_argument("--no-rescale", dest="rescale", action="store_false",
                   help="write the raw absolute radiance->photon-flux conversion instead "
                        "of the pattern-preserving rescale to the model-scale mean")
    p.add_argument("--reference-umol", type=float, default=_SIF_REFERENCE_MEAN_UMOL,
                   help="target area-weighted mean [umol/m2/s] for the pattern rescale")
    p.add_argument("--host", type=str, default=_FTP_HOST)
    p.add_argument("--base", type=str, default=_FTP_BASE)
    return p


def main(argv=None):
    """Fetch the Caltech SIF740 climatology, reduce/convert/regrid, and write the NetCDF."""
    args = build_arg_parser().parse_args(argv)

    tgt_lat, tgt_lon = read_target_grid(args.surf_path)
    print(f"# target cover grid from {args.surf_path}: "
          f"({tgt_lat.size} lat, {tgt_lon.size} lon) -> ncell {tgt_lat.size * tgt_lon.size}")

    monthly_by_year, src_lat, src_lon = _fetch_caltech_sif740(
        args.years, resolution=args.resolution, source_var=args.source_var,
        host=args.host, base=args.base)

    ds, provenance = reduce_sif_to_climatology(
        monthly_by_year, src_lat, src_lon, tgt_lat, tgt_lon,
        aggregate=args.aggregate, active_fraction=args.active_fraction,
        rescale=args.rescale, reference_umol=args.reference_umol)
    ds.attrs.update(
        title="TROPOMI SIF740 growing-season climatology on the surfdata cover grid",
        source=f"ftp://{args.host}{args.base} ({args.source_var})",
        history=(f"scripts/data/build_sif_observations.py; years={list(args.years)}, "
                 f"res={args.resolution}, aggregate={args.aggregate}, "
                 f"rescale={args.rescale}, ref={args.reference_umol} umol/m2/s"),
    )
    _print_sanity(ds, provenance)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ds.to_netcdf(out)
    print(f"# wrote {out}  (dims {dict(ds.sizes)}; var '{_OUT_VAR}' [umol/m2/s])")
    return out


if __name__ == "__main__":
    main()

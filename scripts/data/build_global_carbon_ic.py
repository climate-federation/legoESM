#!/usr/bin/env python
"""Build the global land-carbon initial-condition ("finidat") from real maps.

Stage A of the global carbon IC map
(``docs/superpowers/specs/2026-07-07-global-carbon-ic-map-design.md``): sample
the (PFT x climate) space that occurs globally, spin each sampled archetype to a
verified semi-analytic soil-carbon equilibrium, and assign each grid cell the
cover-weighted mix of its archetypes' equilibria -- so a global DifferLand run
starts *at equilibrium* instead of cold, without spinning up every cell.

This driver only *wires* the already-committed Stage-A pipeline (Tasks 2-5); it
implements no new numerics:

    reduce_climatology_to_features   (climate_features.py)  -- monthly clim -> features
    build_archetypes                 (global_init.py)       -- (PFT x climate) k-means
    equilibrate_archetypes           (global_init.py)       -- semi-analytic spin-up
    map_to_grid                      (global_init.py)       -- cover-weighted pool mix

Cover / soil presets (``--surfdata-preset``)
--------------------------------------------
* **``clm5_surfdata`` (default)** -- a RAW CLM5 surfdata NetCDF (e.g.
  ``surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc``).  Read via the canonical
  :func:`legoesm.land.surface_data.sources.clm5_surfdata.read_clm5_cover_veg`,
  which does the CLM landunit reconstruction (``natpft=15`` natural + ``cft=2``
  crop -> the 17 ``CLM5_PFT_NAMES`` in order) and reads the 2-D ``LATIXY`` /
  ``LONGXY`` coordinates.  Kept on the file's NATIVE grid (no regridding of
  cover).  Per-cell USDA soil-texture class comes from the raw file's topsoil
  ``PCT_SAND`` / ``PCT_CLAY`` via :func:`legoesm.land.soil_texture.usda_texture_index`.
  (The AMIP assembler :func:`~legoesm.land.global_surface_data.load_global_surface_data`
  CANNOT read this raw file: it needs ``BULK_DENSITY`` / ``DZSOI`` and 1-D
  ``lsmlat`` / ``lsmlon`` coordinate variables that a raw CLM5 surfdata lacks --
  measured Task-8h.  It also reads ``PCT_NAT_PFT`` directly (15, no crops)
  without the landunit reconstruction.  So ``clm5_surfdata`` uses the canonical
  reader, not the AMIP assembler.)
* **``legoesm_surfdata``** -- the harmonized ``legoesm_surfdata`` NetCDF (CLM5
  cover/PFT + HWSD v2.0 soil, built by ``scripts/data/build_legoesm_surfdata.py``)
  read via :func:`~legoesm.land.global_surface_data.load_global_surface_data`,
  which CONSERVATIVELY regrids onto a ``--resolution-deg`` target grid.  The
  17-PFT axis passes through :func:`_align_pft_axis`.

The PFT axis is made explicit and CORRECT for both presets by
:func:`_align_pft_axis`: exactly the 17-entry ``CLM5_PFT_NAMES`` layout the
equilibration indexes, zero-padding the two crop columns if a natural-only
(15-PFT) source is supplied, and raising on any other count -- never a silent
mismatch.

Climate (two modes)
-------------------
* **``--climate-from-latitude`` (zonal DEMONSTRATION)** -- builds the monthly
  ``(ncell, 12)`` T / precip / SW / net-radiation from each cell's LATITUDE,
  reusing the committed ``lmip_forcing`` latitude->feature pieces
  (:func:`~legoesm.land.lmip_forcing.latitude_mean_annual_temp_k` /
  ``latitude_seasonal_amp_k`` / ``latitude_daily_mean_sw_w`` -- MAT / seasonal
  amplitude / daily-mean insolation) plus a documented idealized zonal
  precipitation profile and a net-radiation fraction.  No climatology NetCDF.
  The climate axis is LATITUDE-ONLY (a prominent banner is printed) pending a
  real assembled ERA5 monthly land-forcing climatology.
* **``--climatology <nc>`` (science-grade)** -- read the four monthly land fields
  from a NetCDF and regrid onto the cover grid with the repo's
  :func:`legoesm.grids.regridding.conservative_regrid_latlon`.  Kept intact for
  when such a file exists; this driver never fabricates a real climatology.

``--dry-run-synthetic`` fabricates a tiny world in-process (a handful of cells,
3 PFTs) and runs the FULL pipeline + writes both ``.npz`` files, so the driver
is exercised end-to-end WITHOUT any data files.  Every real-data loader/assembler
import is deferred to FUNCTION scope so the dry run -- and importing this module
-- never requires the surface-data packages or the data files.

Login-node policy: the equilibration JIT-compiles the coupled land+carbon model,
so run this via ``sbatch``/``srun`` on a compute node, never on the login node.

Outputs (both under ``<--output>/``)
------------------------------------
* ``global_carbon_ic.npz`` -- per-pool ``(ncell,)`` CarbonState + ``lat``/``lon``
  + ``dominant_pft`` / ``pft_present`` (the "there" PFT ids) + cover weights.
* ``archetypes.npz`` -- the ``ArchetypeTable`` + per-archetype equilibrium pools
  + the QC bundle (GPP/NPP/SOC/biomass/drift), the reusable ``(PFT,climate)->pools``
  lookup.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from legoesm.land.carbon.climate_features import reduce_climatology_to_features
from legoesm.land.carbon.global_init import (
    build_archetypes,
    dropped_cover_fraction,
    equilibrate_archetypes,
    map_to_grid,
)

from legoesm import constants

# --- global regular-lat-lon grid geometry (degrees; grid layout, not physics) ---
_LAT_SPAN_DEG = 180.0          # pole-to-pole latitude span [deg]
_LON_SPAN_DEG = 360.0          # full longitude span [deg]

# --- synthetic-world PFT indices (into CLM5_PFT_NAMES) for --dry-run-synthetic ---
_PFT_TROPICAL_TREE = 4         # broadleaf_evergreen_tropical (woody)
_PFT_TEMPERATE_TREE = 7        # broadleaf_deciduous_temperate (woody)
_PFT_C3_GRASS = 13             # c3_grass (herbaceous)

# --- zonal DEMONSTRATION climate (--climate-from-latitude) ---
# Calendar (year length / NH seasonal peak) used to EXPAND the latitude-derived
# annual mean + amplitude into a 12-month T cycle.  Matches the NH-phased
# convention of legoesm.land.climate_forcing.make_climatological_forcing (July
# peak), which the archetype equilibration re-imposes anyway -- the monthly
# sampling phase only sets the clustering features (mean, half-range).
_YEAR_DAYS = 365.0             # calendar year length [day]
_SEASONAL_PEAK_DOY = 200.0     # NH day-of-year of the seasonal T maximum (~July)
# Idealized zonal-mean precipitation [kg/m2/s]: an ITCZ peak at the equator plus
# mid-latitude storm-track maxima near +-50 deg on a small everywhere-base,
# reproducing the wet-tropics / dry-subtropics / wet-midlat / dry-pole structure
# of the observed zonal mean (order-of-magnitude ~ Adler et al. 2003 GPCP;
# ~4.5 mm/day tropics, ~1.7 mm/day midlat).  DEMONSTRATION magnitudes, not a fit.
_PRECIP_BASE = 3.0e-6          # everywhere floor [kg/m2/s] (~0.26 mm/day)
_PRECIP_ITCZ_PEAK = 5.2e-5     # ITCZ excess at the equator [kg/m2/s] (~4.5 mm/day)
_PRECIP_ITCZ_WIDTH_DEG = 12.0  # ITCZ Gaussian half-width [deg]
_PRECIP_STORM_PEAK = 2.0e-5    # storm-track excess [kg/m2/s] (~1.7 mm/day)
_PRECIP_STORM_LAT_DEG = 50.0   # storm-track centre |latitude| [deg]
_PRECIP_STORM_WIDTH_DEG = 15.0  # storm-track Gaussian half-width [deg]
# Surface net radiation as a fraction of down-welling SW (drives ONLY the
# PET / aridity clustering feature).  Annual-mean surface net radiation over land
# is ~0.5-0.6 of down-SW (Trenberth et al. 2009 global energy budget); one
# fraction is adequate for the aridity feature in this DEMONSTRATION climate.
_NETRAD_OVER_SW_LAND = 0.55    # [-] land net-radiation / down-SW fraction

# --- persistent-cache policy (compile cache + equilibrium RESULT cache) --------
# Both caches MIRROR scripts/run/train_carbon_params.py (the Stage-B calibration
# trainer) so the caching policy never drifts between the two carbon drivers; the
# shared XLA compile cache is legoesm.ml.training.configure_jax_compilation_cache.
_CACHE_MIN_COMPILE_SECS_DEFAULT = 1.0   # only cache XLA compiles slower than this [s]
# BUMP _EQUILIBRIUM_CACHE_VERSION on ANY change to the archetype spin-up / coupled-
# carbon-step physics -- equilibrate_archetypes / iter_archetype_batches /
# _spinup_batch, run_semi_analytic_spinup, make_archetype_step_fn /
# carbon_cycle.step_carbon -- OR the DEFAULT CarbonConfig / MultiLayerLandConfig
# parameter VALUES those consume: the cached (eq, qc) is the DEFAULT-parameter
# equilibrium, so a changed default silently changes it and this INPUT-keyed cache
# would otherwise serve a STALE result.  Mirrors _PRECOMPUTE_CACHE_VERSION in the
# trainer and the XLA cache's HLO-in-key safety.
# PARAMS CAVEAT: the key intentionally hashes ONLY the archetype table + spin
# config, NOT any CarbonConfig, because this build always equilibrates at the
# PRODUCTION DEFAULT CarbonConfig (equilibrate_archetypes takes no params).  If a
# `--tuned-params` argument is EVER added here, the tuned params change eq, so the
# key MUST then also hash them (add a params digest term) -- otherwise two
# different parameterisations would alias to one digest and serve a wrong cache.
# REGIME: the numeric equilibrium is precision- and backend-dependent, so the cache
# is scoped to ONE regime rather than cross-serving float32/float64 or CPU/GPU
# results (which would break the byte-identical contract).  main() PINS x64 before
# any JAX op (like the trainer's train()), so precision is invariant; the result
# cache is additionally namespaced by jax.default_backend() (a `<backend>/`
# subdir), so a CPU-built cache never serves a GPU run or vice versa.  A jaxlib
# upgrade that alters bits on the SAME backend is the user's responsibility (bump
# the version), mirroring the trainer's documented same-code/backend assumption.
_EQUILIBRIUM_CACHE_VERSION = "v1"
# The QC bundle equilibrate_archetypes returns, in CANONICAL order.  The result
# cache requires EXACTLY these members on load (a file missing one is treated as
# corrupt and recomputed, never served as a partial hit); keep in sync with
# equilibrate_archetypes (bump _EQUILIBRIUM_CACHE_VERSION if this set changes).
_QC_KEYS = ("gpp", "npp", "som_kgC", "biomass_kgC", "drift_frac_per_yr")


class GlobalCarbonInputs:
    """Per-cell driver inputs on a common ``(ncell,)`` land vector.

    A tiny plain container (not a config): the shared hand-off between the
    real-data / synthetic loaders and the archetype pipeline.
    """

    __slots__ = (
        "pft_weights", "monthly_t_k", "monthly_precip", "monthly_sw",
        "monthly_netrad", "soil_class", "land_mask", "cell_lat_deg",
        "cell_lon_deg", "n_pft",
    )

    def __init__(self, pft_weights, monthly_t_k, monthly_precip, monthly_sw,
                 monthly_netrad, soil_class, land_mask, cell_lat_deg,
                 cell_lon_deg):
        self.pft_weights = np.asarray(pft_weights, float)      # (ncell, n_pft)
        self.monthly_t_k = np.asarray(monthly_t_k, float)      # (ncell, 12)
        self.monthly_precip = np.asarray(monthly_precip, float)
        self.monthly_sw = np.asarray(monthly_sw, float)
        self.monthly_netrad = np.asarray(monthly_netrad, float)
        self.soil_class = np.asarray(soil_class, dtype=object)  # (ncell,) str
        self.land_mask = np.asarray(land_mask, bool)           # (ncell,)
        self.cell_lat_deg = np.asarray(cell_lat_deg, float)    # (ncell,)
        self.cell_lon_deg = np.asarray(cell_lon_deg, float)    # (ncell,)
        self.n_pft = int(self.pft_weights.shape[1])


# ===========================================================================
# Synthetic world (--dry-run-synthetic): no files, no assembler, no grid
# ===========================================================================
def build_synthetic_inputs() -> GlobalCarbonInputs:
    """Fabricate a tiny (6-cell, 3-PFT) world exercising the full pipeline.

    Two woody PFTs (tropical + temperate tree) and one herbaceous PFT (C3 grass),
    with pure and mixed cells, so ``equilibrate_archetypes`` sees BOTH a woody and
    a herbaceous ``(is_woody, soil_class)`` group.  A NH-phased seasonal T cycle
    gives each cell a well-defined seasonal amplitude; precip/SW/net-radiation are
    per-cell constants.  Everything is land; soil is a single texture so the two
    groups are exactly {woody, herbaceous}.
    """
    from legoesm.land.surface_params import N_PFT_CLM5

    n_pft = int(N_PFT_CLM5)
    # (pft indices+weights, mean T [K], seasonal half-amp [K], precip [kg/m2/s],
    #  SW-down [W/m2], net radiation [W/m2]) per cell.
    cells = [
        ({_PFT_TROPICAL_TREE: 1.0},                    298.0, 2.0, 6.0e-5, 230.0, 110.0),
        ({_PFT_TEMPERATE_TREE: 1.0},                   283.0, 12.0, 2.5e-5, 190.0, 80.0),
        ({_PFT_C3_GRASS: 1.0},                         288.0, 8.0, 3.0e-5, 210.0, 90.0),
        ({_PFT_TROPICAL_TREE: 0.5, _PFT_TEMPERATE_TREE: 0.5}, 293.0, 6.0, 4.0e-5, 210.0, 95.0),
        ({_PFT_TEMPERATE_TREE: 0.6, _PFT_C3_GRASS: 0.4}, 285.0, 10.0, 3.0e-5, 195.0, 85.0),
        ({_PFT_TROPICAL_TREE: 1.0},                    300.0, 1.5, 7.0e-5, 235.0, 115.0),
    ]
    ncell = len(cells)
    months = np.arange(12)
    # NH-phased annual T cycle: reduce_climatology_to_features recovers the
    # half-amplitude from 0.5*(max-min).
    seasonal_shape = np.sin(2.0 * np.pi * months / 12.0)

    pft_weights = np.zeros((ncell, n_pft))
    monthly_t = np.zeros((ncell, 12))
    monthly_pr = np.zeros((ncell, 12))
    monthly_sw = np.zeros((ncell, 12))
    monthly_nr = np.zeros((ncell, 12))
    for c, (wmap, mean_t, amp, pr, sw, nr) in enumerate(cells):
        for p, wt in wmap.items():
            pft_weights[c, p] = wt
        monthly_t[c] = mean_t + amp * seasonal_shape
        monthly_pr[c] = pr
        monthly_sw[c] = sw
        monthly_nr[c] = nr

    soil_class = np.array(["loam"] * ncell, dtype=object)
    land_mask = np.ones(ncell, bool)
    # A small idealised lat/lon spread (metadata only; unused by the pipeline).
    cell_lat = np.linspace(-30.0, 60.0, ncell)
    cell_lon = np.linspace(0.0, 300.0, ncell)
    return GlobalCarbonInputs(
        pft_weights, monthly_t, monthly_pr, monthly_sw, monthly_nr,
        soil_class, land_mask, cell_lat, cell_lon)


# ===========================================================================
# Real-data world: canonical surface-data assembler + climatology NetCDF
# ===========================================================================
def _fill_nonfinite(a: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the field's finite mean.

    Ocean / no-data cells (dropped by ``conservative_regrid_latlon``) return NaN;
    those cells are excluded from clustering by the land mask anyway, but
    ``build_archetypes`` standardises features over ALL cells, so a NaN would
    poison the global mean/std.  Filling with the finite mean keeps the
    standardisation well-defined without touching any land-cell value.
    """
    a = np.asarray(a, float)
    bad = ~np.isfinite(a)
    if bad.any():
        fill = float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
        a = np.where(bad, fill, a)
    return a


def _load_monthly_climatology(args, tgt_lat_deg, tgt_lon_deg):
    """Read a monthly-climatology NetCDF and regrid to the target grid.

    Returns ``(monthly_t_k, monthly_precip, monthly_sw, monthly_netrad)`` each
    ``(ncell, 12)`` on the target grid's flattened ``(i_lat, i_lon)`` (row-major)
    ordering -- the SAME ordering ``load_global_surface_data`` produces, so PFT /
    soil / climate share one ``(ncell,)`` vector.  Uses the repo's conservative
    lat-lon regridder; no hand-rolled regridding.

    The file must expose the four fields (variable names configurable) shaped
    ``(time=12, lat, lon)`` with 1-D ``lat``/``lon`` coordinates in degrees.  T is
    Kelvin unless ``--clim-t-in-celsius`` is set (then ``constants.T_freeze`` is
    added).
    """
    import xarray as xr
    from legoesm.grids.regridding import conservative_regrid_latlon

    tgt_lat_deg = np.asarray(tgt_lat_deg, float)
    tgt_lon_deg = np.asarray(tgt_lon_deg, float)
    n_lat, n_lon = tgt_lat_deg.size, tgt_lon_deg.size

    ds = xr.open_dataset(args.climatology)
    try:
        src_lat = np.asarray(ds[args.clim_lat].values, float)
        src_lon = np.asarray(ds[args.clim_lon].values, float)

        def _regrid(var_name):
            da = ds[var_name]
            other = [d for d in da.dims
                     if d not in (args.clim_lat, args.clim_lon)]
            if len(other) != 1:
                raise ValueError(
                    f"climatology variable {var_name!r} must have exactly one "
                    f"non-spatial (time) dimension; got dims {da.dims}.")
            # (lat, lon, time) -> conservative regrid -> (n_lat, n_lon, 12).
            arr = da.transpose(args.clim_lat, args.clim_lon, other[0]).values
            arr = np.asarray(arr, float)
            if arr.shape[-1] != 12:
                raise ValueError(
                    f"climatology variable {var_name!r} time axis has "
                    f"{arr.shape[-1]} entries; expected 12 monthly means.")
            out = conservative_regrid_latlon(
                arr, src_lat, src_lon, tgt_lat_deg, tgt_lon_deg)
            return _fill_nonfinite(out).reshape(n_lat * n_lon, 12)

        monthly_t = _regrid(args.clim_t_var)
        monthly_pr = _regrid(args.clim_precip_var)
        monthly_sw = _regrid(args.clim_sw_var)
        monthly_nr = _regrid(args.clim_netrad_var)
    finally:
        ds.close()

    if args.clim_t_in_celsius:
        monthly_t = monthly_t + constants.T_freeze
    return monthly_t, monthly_pr, monthly_sw, monthly_nr


def _align_pft_axis(pft_weights, source: str) -> np.ndarray:
    """Return per-cell cover on the EXACT 17-entry ``CLM5_PFT_NAMES`` layout.

    The input MUST already be per-GRIDCELL cover fraction (each column is the
    cell's cover fraction of that PFT, so the columns sum to the cell's
    vegetated land fraction, NOT to 1).  The equilibration indexes cover columns
    as ``CLM5_PFT_NAMES`` (bare + 14 natural veg + 2 crops).

    * A source already on 17 gridcell-fraction PFTs passes through.
    * A natural-only 15-column source zero-pads the two crop columns
      (``CLM5_PFT_NAMES[15:17]`` = ``crop_c3`` / ``crop_c4``) -- **but only when
      those 15 columns are already gridcell fractions**.  Raw CLM
      ``PCT_NAT_PFT`` is percent WITHIN the natural-veg landunit (sums to 100 %
      of natveg, not of the gridcell); feeding it here would silently treat
      every cell as fully natural-vegetated.  Raw CLM landunit cover MUST go
      through
      :func:`legoesm.land.surface_data.sources.clm5_surfdata.reconstruct_clm5_pft_frac`
      (natveg/crop-scaled to 17 gridcell fractions) FIRST -- this branch does
      NOT rescale (F8).
    * Any other count is a real mismatch and RAISES -- never a silent
      mis-alignment.
    """
    from legoesm.land.surface_params import N_PFT_CLM5

    pw = np.asarray(pft_weights, float)
    if pw.ndim != 2:
        raise ValueError(f"{source} pft_weights must be 2-D (ncell, n_pft); "
                         f"got shape {pw.shape}.")
    n_pft = pw.shape[1]
    if n_pft == N_PFT_CLM5:
        return pw
    if n_pft == N_PFT_CLM5 - 2:
        # Natural-only source: CLM5_PFT_NAMES[0:15] are the natural PFTs,
        # [15:17] the two (absent) crops -> zero-pad the crop columns.  REQUIRES
        # gridcell-fraction input: raw within-natveg PCT_NAT_PFT must be
        # reconstruct_clm5_pft_frac'd to 17 gridcell fractions first (this branch
        # does NOT natveg-scale -- see the function docstring, F8).
        return np.concatenate([pw, np.zeros((pw.shape[0], 2), float)], axis=1)
    raise ValueError(
        f"{source} PFT cover has {n_pft} columns; cannot reconcile with the "
        f"{N_PFT_CLM5}-entry CLM5_PFT_NAMES layout the equilibration indexes "
        f"(need {N_PFT_CLM5}, or {N_PFT_CLM5 - 2} natural-only to zero-pad crops).")


def _zonal_precip_profile(cell_lat_deg) -> np.ndarray:
    """Idealized zonal-mean precipitation rate [kg/m2/s] from latitude [deg].

    Base floor + an equatorial ITCZ Gaussian (signed latitude, peak at 0) + a
    mid-latitude storm-track Gaussian at ``+-_PRECIP_STORM_LAT_DEG`` (|latitude|,
    both hemispheres).  DEMONSTRATION profile (documented module constants), not
    an observational fit.
    """
    lat = np.asarray(cell_lat_deg, float)
    itcz = _PRECIP_ITCZ_PEAK * np.exp(-(lat / _PRECIP_ITCZ_WIDTH_DEG) ** 2)
    storm = _PRECIP_STORM_PEAK * np.exp(
        -((np.abs(lat) - _PRECIP_STORM_LAT_DEG) / _PRECIP_STORM_WIDTH_DEG) ** 2)
    return _PRECIP_BASE + itcz + storm


def zonal_monthly_climate(cell_lat_deg):
    """Build the ``(ncell, 12)`` zonal DEMONSTRATION monthly climatology.

    Reuses the committed ``lmip_forcing`` latitude->feature pieces -- MAT
    (:func:`~legoesm.land.lmip_forcing.latitude_mean_annual_temp_k`), seasonal
    half-amplitude (``latitude_seasonal_amp_k``) and daily-mean insolation
    (``latitude_daily_mean_sw_w``) -- so no solar-geometry / temperature numerics
    are re-derived.  T is the latitude MAT + amplitude expanded over 12 mid-month
    days with the NH July-peaked seasonal cosine; SW is the latitude/day
    daily-mean insolation (a real seasonal SW cycle); precip is the documented
    zonal profile (:func:`_zonal_precip_profile`); net radiation is a documented
    fraction of down-SW (drives only the PET / aridity clustering feature).

    Returns ``(monthly_t_k, monthly_precip, monthly_sw, monthly_netrad)`` each
    ``(ncell, 12)``, matching what ``reduce_climatology_to_features`` expects.
    """
    # Deferred (function-scope) import: keep module import numpy-only.
    import jax
    import jax.numpy as jnp

    from legoesm.land.lmip_forcing import (
        latitude_daily_mean_sw_w,
        latitude_mean_annual_temp_k,
        latitude_seasonal_amp_k,
    )

    lat_deg = np.asarray(cell_lat_deg, float)
    ncell = lat_deg.shape[0]
    lat_rad = np.deg2rad(lat_deg)
    doy_months = (np.arange(12) + 0.5) * (_YEAR_DAYS / 12.0)        # (12,) mid-month

    # --- T: latitude MAT + amplitude, expanded over 12 months (NH July peak) ---
    mat = np.asarray(latitude_mean_annual_temp_k(jnp.asarray(lat_rad)), float)   # (ncell,)
    amp = np.asarray(latitude_seasonal_amp_k(jnp.asarray(lat_rad)), float)       # (ncell,)
    seasonal = np.cos(2.0 * np.pi * (doy_months - _SEASONAL_PEAK_DOY) / _YEAR_DAYS)
    monthly_t = mat[:, None] + amp[:, None] * seasonal[None, :]                  # (ncell, 12)

    # --- SW: latitude/day daily-mean insolation (one vmap over all cell-months) ---
    lat_flat = np.repeat(lat_rad, 12)                              # (ncell*12,)
    day_flat = np.tile(doy_months, ncell)                          # (ncell*12,)
    sw_flat = jax.vmap(latitude_daily_mean_sw_w)(
        jnp.asarray(lat_flat), jnp.asarray(day_flat))
    monthly_sw = np.asarray(sw_flat, float).reshape(ncell, 12)

    # --- precip (documented zonal profile) + net radiation (SW fraction) ---
    monthly_pr = np.broadcast_to(
        _zonal_precip_profile(lat_deg)[:, None], (ncell, 12)).copy()
    monthly_nr = _NETRAD_OVER_SW_LAND * monthly_sw
    return monthly_t, monthly_pr, monthly_sw, monthly_nr


def _print_zonal_banner() -> None:
    """Prominent banner: the climate axis is latitude-only, not real ERA5."""
    bar = "=" * 74
    print(bar)
    print("  ZONAL DEMONSTRATION CLIMATE  (--climate-from-latitude)")
    print("  Monthly T / SW derive from each cell's LATITUDE via the committed")
    print("  lmip_forcing latitude baseline + solar geometry; precip / net-rad")
    print("  follow documented idealized zonal profiles.  The climate axis is")
    print("  LATITUDE-ONLY -- NOT a real ERA5 climatology.  Use --climatology <nc>")
    print("  for a science-grade build once a monthly land-forcing file exists.")
    print(bar)


def _load_clm5_cover_soil(args) -> dict:
    """Real CLM5 cover (17-PFT) + topsoil texture on the file's NATIVE grid.

    Uses the canonical :func:`read_clm5_cover_veg` (LATIXY/LONGXY coords + the
    natpft=15 + cft=2 -> 17 ``CLM5_PFT_NAMES`` landunit reconstruction) for cover,
    and reads the raw file's topsoil ``PCT_SAND`` / ``PCT_CLAY`` (already percent)
    for the per-cell USDA soil-texture class.  No regridding (native grid).
    """
    import xarray as xr

    from legoesm.land.soil_texture import USDA_TEXTURES, usda_texture_index
    from legoesm.land.surface_data.sources.clm5_surfdata import read_clm5_cover_veg

    surf = args.surf_path
    if not surf:
        raise SystemExit(
            "--surf-path (raw CLM5 surfdata NetCDF) is required for "
            "--surfdata-preset clm5_surfdata, or pass --dry-run-synthetic.")

    clm = read_clm5_cover_veg(surf)
    lat = np.asarray(clm["lat"], float)                # (nlat,)
    lon = np.asarray(clm["lon"], float)                # (nlon,)
    n_lat, n_lon = lat.size, lon.size
    pft_frac_pct = np.asarray(clm["pft_frac"], float)  # (17, nlat, nlon) % of gridcell
    f_land_pct = np.asarray(clm["f_land"], float)      # (nlat, nlon) %

    # Per-cell (ncell,) row-major (i_lat, i_lon), 17-PFT fractional cover.
    pft_weights = np.moveaxis(pft_frac_pct, 0, -1).reshape(n_lat * n_lon, -1) / 100.0
    pft_weights = _align_pft_axis(pft_weights, "clm5_surfdata")

    # Topsoil (layer 0) sand/clay [percent] from the raw file -> USDA class.
    ds = xr.open_dataset(surf, decode_times=False)
    try:
        sand_pct = np.asarray(ds["PCT_SAND"].values, float)[0].ravel()   # (ncell,)
        clay_pct = np.asarray(ds["PCT_CLAY"].values, float)[0].ravel()
    finally:
        ds.close()
    tex_idx = np.asarray(usda_texture_index(sand_pct, clay_pct))
    soil_class = np.array([USDA_TEXTURES[int(i)] for i in tex_idx], dtype=object)

    land_mask = f_land_pct.ravel() > 0.0
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    return dict(
        pft_weights=pft_weights, soil_class=soil_class, land_mask=land_mask,
        cell_lat=lat2d.ravel(), cell_lon=lon2d.ravel(),
        tgt_lat_1d=lat, tgt_lon_1d=lon)


def _load_legoesm_cover_soil(args) -> dict:
    """Harmonized ``legoesm_surfdata`` cover + soil via the AMIP assembler.

    Conservatively regrids onto a ``--resolution-deg`` target grid with
    :func:`~legoesm.land.global_surface_data.load_global_surface_data`; the 17-PFT
    axis passes through :func:`_align_pft_axis`.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.land.global_surface_data import (
        get_surfdata_preset, load_global_surface_data,
    )
    from legoesm.land.soil_texture import USDA_TEXTURES, usda_texture_index

    surf = args.surf_path
    if not surf:
        raise SystemExit(
            "--surf-path (harmonized legoesm_surfdata NetCDF) is required for "
            "--surfdata-preset legoesm_surfdata; build it with "
            "scripts/data/build_legoesm_surfdata.py or pass --dry-run-synthetic.")

    res = float(args.resolution_deg)
    n_lat = int(round(_LAT_SPAN_DEG / res))
    n_lon = int(round(_LON_SPAN_DEG / res))
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    preset = get_surfdata_preset("legoesm_surfdata")._replace(
        surf_path=surf, landuse_path=surf, veg_path=(args.veg_path or surf))
    gsd = load_global_surface_data(preset, grid)

    # gsd.pft_frac is renormalized to sum-to-1 over PFTs per cell (ignoring the
    # land fraction, global_surface_data._renorm_pft), so a cell with f_land < 1
    # would otherwise carry FULL land-area carbon.  Scale by the gridcell land
    # fraction so this preset expresses per-GRIDCELL cover, matching the
    # clm5_surfdata path (whose reconstruct_clm5_pft_frac already returns
    # percent-of-gridcell) (F1).
    pft_cover = (np.asarray(gsd.pft_frac[0], float)
                 * np.asarray(gsd.f_land[0], float)[:, None])
    pft_weights = _align_pft_axis(pft_cover, "legoesm_surfdata")
    # Topsoil (layer 0) sand/clay [fraction -> percent] -> USDA class.
    sand_pct = np.asarray(gsd.sand_frac[:, 0], float) * 100.0
    clay_pct = np.asarray(gsd.clay_frac[:, 0], float) * 100.0
    tex_idx = np.asarray(usda_texture_index(sand_pct, clay_pct))
    soil_class = np.array([USDA_TEXTURES[int(i)] for i in tex_idx], dtype=object)
    land_mask = np.asarray(gsd.f_land[0], float) > 0.0

    return dict(
        pft_weights=pft_weights, soil_class=soil_class, land_mask=land_mask,
        cell_lat=np.rad2deg(np.asarray(grid.lat2d, float)).ravel(),
        cell_lon=np.rad2deg(np.asarray(grid.lon2d, float)).ravel(),
        tgt_lat_1d=np.rad2deg(np.asarray(grid.lat, float)),
        tgt_lon_1d=np.rad2deg(np.asarray(grid.lon, float)))


def load_real_inputs(args) -> GlobalCarbonInputs:
    """Load the real PFT / soil cover + climate onto one ``(ncell,)`` vector.

    Cover / soil dispatch on ``--surfdata-preset`` (``clm5_surfdata`` = raw CLM5
    native grid via :func:`_load_clm5_cover_soil`; ``legoesm_surfdata`` =
    harmonized + regridded via :func:`_load_legoesm_cover_soil`).  Climate is
    either the zonal DEMONSTRATION (``--climate-from-latitude``,
    :func:`zonal_monthly_climate`) or a real monthly-climatology NetCDF
    (``--climatology``, :func:`_load_monthly_climatology`).
    """
    if args.surfdata_preset == "clm5_surfdata":
        cov = _load_clm5_cover_soil(args)
    elif args.surfdata_preset == "legoesm_surfdata":
        cov = _load_legoesm_cover_soil(args)
    else:
        raise SystemExit(
            f"--surfdata-preset {args.surfdata_preset!r} unknown; use "
            f"'clm5_surfdata' or 'legoesm_surfdata'.")

    if args.climate_from_latitude:
        _print_zonal_banner()
        monthly_t, monthly_pr, monthly_sw, monthly_nr = zonal_monthly_climate(
            cov["cell_lat"])
    else:
        if not args.climatology:
            raise SystemExit(
                "a climate source is required: pass --climate-from-latitude "
                "(zonal DEMONSTRATION) or --climatology <nc> (real monthly-mean "
                "T/precip/SW/netrad), or --dry-run-synthetic.")
        monthly_t, monthly_pr, monthly_sw, monthly_nr = _load_monthly_climatology(
            args, cov["tgt_lat_1d"], cov["tgt_lon_1d"])

    return GlobalCarbonInputs(
        cov["pft_weights"], monthly_t, monthly_pr, monthly_sw, monthly_nr,
        cov["soil_class"], cov["land_mask"], cov["cell_lat"], cov["cell_lon"])


# ===========================================================================
# QC summary + outputs
# ===========================================================================
def _print_qc_summary(table, qc, n_arch) -> None:
    """Print archetype count and per-PFT SOC / biomass ranges from the QC bundle."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES

    som = np.asarray(qc["som_kgC"], float)
    bio = np.asarray(qc["biomass_kgC"], float)
    drift = np.asarray(qc["drift_frac_per_yr"], float)
    pft_id = np.asarray(table.pft_id, int)

    print(f"[global_carbon_ic] {n_arch} archetypes across "
          f"{len(np.unique(pft_id))} PFTs")
    print(f"[global_carbon_ic] SOC  range {som.min():8.2f} .. {som.max():8.2f} kgC/m2")
    print(f"[global_carbon_ic] biomass range {bio.min():8.2f} .. {bio.max():8.2f} kgC/m2")
    print(f"[global_carbon_ic] |drift| max {np.abs(drift).max():.2e} /yr "
          f"(verify-segment total-C)")
    print("[global_carbon_ic] per-PFT SOC / biomass [kgC/m2] ranges:")
    for p in np.unique(pft_id):
        sel = pft_id == p
        name = CLM5_PFT_NAMES[int(p)] if int(p) < len(CLM5_PFT_NAMES) else f"pft{p}"
        print(f"    {name:36s} n={int(sel.sum()):3d}  "
              f"SOC [{som[sel].min():7.2f}, {som[sel].max():7.2f}]  "
              f"biomass [{bio[sel].min():7.2f}, {bio[sel].max():7.2f}]")


def _write_archetypes_npz(path, table, eq, qc, pft_names, *,
                          n_layers, soil_depth, dt, res_deg):
    """Write ``archetypes.npz``: ``ArchetypeTable`` + eq pools + QC + the
    SOIL-COLUMN GEOMETRY.

    The geometry (``n_layers`` / ``soil_depth`` / ``dt`` / ``resolution_deg``) is
    persisted so the drift validator re-integrates each archetype on the SAME
    soil column the map equilibrated on -- it defaults the validator CLI to these
    and errors on a conflicting override (never a mismatched column).
    """
    arch = {}
    for f in table._fields:
        vals = np.asarray(getattr(table, f))
        # soil_class is a string/object column -> store as fixed-width unicode
        # (no object pickling in the npz).
        arch[f] = vals.astype("U40") if vals.dtype == object else vals
    for f in eq._fields:
        arch[f"eq_{f}"] = np.asarray(getattr(eq, f), float)
    for k, v in qc.items():
        arch[f"qc_{k}"] = np.asarray(v, float)
    arch["pft_names"] = np.asarray(pft_names, dtype="U40")
    # --- soil-column geometry (Task-8h geometry persistence) ---
    arch["n_layers"] = np.asarray(int(n_layers))
    arch["soil_depth"] = np.asarray(float(soil_depth))
    arch["dt"] = np.asarray(float(dt))
    arch["resolution_deg"] = np.asarray(float(res_deg))
    np.savez(path, **arch)
    return path


def _write_outputs(out_dir, inputs, grid_state, table, eq, qc, w_min, res_deg,
                   *, n_layers, soil_depth, dt):
    """Write ``global_carbon_ic.npz`` (finidat) + ``archetypes.npz`` (lookup)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finidat_path = out / "global_carbon_ic.npz"
    archetypes_path = out / "archetypes.npz"

    pft_weights = inputs.pft_weights
    pft_present = pft_weights >= w_min                              # (ncell, n_pft) "there"
    has_cover = pft_weights.max(axis=1) > 0.0
    dominant_pft = np.where(has_cover, pft_weights.argmax(axis=1), -1).astype(np.int64)

    from legoesm.land.surface_params import CLM5_PFT_NAMES
    pft_names = np.asarray(CLM5_PFT_NAMES[:inputs.n_pft], dtype="U40")

    # --- finidat: per-pool (ncell,) CarbonState + geometry + PFT ids ---
    finidat = {f: np.asarray(getattr(grid_state, f), float) for f in grid_state._fields}
    finidat.update(
        lat=inputs.cell_lat_deg, lon=inputs.cell_lon_deg,
        land_mask=inputs.land_mask,
        dominant_pft=dominant_pft, pft_present=pft_present,
        pft_weights=pft_weights, pft_names=pft_names,
        resolution_deg=np.asarray(float(res_deg)),
        n_layers=np.asarray(int(n_layers)),
        soil_depth=np.asarray(float(soil_depth)),
        dt=np.asarray(float(dt)),
    )
    np.savez(finidat_path, **finidat)

    # --- archetypes: ArchetypeTable + equilibrium pools + QC + geometry ---
    _write_archetypes_npz(
        archetypes_path, table, eq, qc, pft_names,
        n_layers=n_layers, soil_depth=soil_depth, dt=dt, res_deg=res_deg)

    return finidat_path, archetypes_path


# ===========================================================================
# Deterministic equilibrium RESULT cache (driver-level; global_init stays PURE)
# ===========================================================================
def _check_cached_array(a, name: str, n_arch: int):
    """Validate a cached-npz member: numeric dtype + per-archetype ``(n_arch,)``.

    Raised ``ValueError`` propagates into ``_load_or_equilibrate``'s cache-read
    ``except`` so a structurally-invalid file (wrong shape, or an object/complex
    dtype ``jnp.asarray`` would otherwise ``TypeError`` on) is DROPPED and
    recomputed -- never served as a silently-wrong hit.  Returns the array.
    """
    a = np.asarray(a)
    if a.dtype.kind not in "fiu" or a.shape != (n_arch,):
        raise ValueError(
            f"cached {name!r}: dtype {a.dtype} shape {a.shape} "
            f"(expected a numeric (n_arch={n_arch},) array)")
    return a


def _equilibrium_cache_key(table, spin: dict) -> str:
    """Stable SHA-256 digest over the equilibration's INPUTS.

    The ``equilibrate_archetypes`` output (``(eq, qc)``) is DETERMINISTIC given
    the ``ArchetypeTable`` + the spin config, evaluated at the PRODUCTION DEFAULT
    ``CarbonConfig`` (this build passes no tuned params).  Key components, in a
    FIXED and documented order (any reorder changes the digest, so it must never
    be reordered):

      1. ``_EQUILIBRIUM_CACHE_VERSION`` -- the stale-physics guard (bump on any
         spin-up / coupled-carbon-step or default-parameter change; see its
         definition, including the ``--tuned-params`` caveat).
      2. the six NUMERIC ``ArchetypeTable`` fields in declaration order
         (``pft_id, mat_k, map_yr, t_seasonal_amp_k, aridity, sw_mean_w``), each
         as ``np.ascontiguousarray(...).tobytes()`` -- tagged with the field name,
         dtype, and shape so two distinct tables can never alias to one digest.
      3. the string ``soil_class`` field as ``"|".join(...)`` (per-archetype
         texture keys; ``|`` is not a soil-class token).
      4. the spin config ``(n_spinup, n_verify, dt, n_layers, soil_depth)``
         (``repr`` on the floats keeps full precision stable).

    Mirrors ``train_carbon_params._precompute_cache_key``; the only differences
    are the domain-separation prefix and the version constant, because the two
    caches store DIFFERENT artifacts and must never collide on a shared dir.
    """
    h = hashlib.sha256()
    h.update(b"GLOBAL_CARBON_EQUILIBRIUM")
    h.update(_EQUILIBRIUM_CACHE_VERSION.encode())
    for name in ("pft_id", "mat_k", "map_yr", "t_seasonal_amp_k", "aridity",
                 "sw_mean_w"):
        arr = np.ascontiguousarray(getattr(table, name))
        h.update(f"|{name}:{arr.dtype}:{arr.shape}|".encode())
        h.update(arr.tobytes())
    soil_class = "|".join(str(s) for s in np.asarray(table.soil_class).ravel())
    h.update(b"|soil_class|")
    h.update(soil_class.encode("utf-8"))
    spin_key = (f"|spin|{spin['n_spinup']}|{spin['n_verify']}|{spin['dt']!r}|"
                f"{spin['n_layers']}|{spin['soil_depth']!r}|")
    h.update(spin_key.encode("utf-8"))
    return h.hexdigest()


def _load_or_equilibrate(table, spin: dict, *, cache_dir: str, rebuild: bool):
    """Deterministic RESULT cache around :func:`equilibrate_archetypes`.

    ``equilibrate_archetypes`` cold-compiles + spins ONE coupled-land-model graph
    per ``(is_woody, is_evergreen, soil_class)`` group (~19 min/group on a diverse
    real archetype set); its ``(eq, qc)`` is a PURE function of the archetype
    table + spin config at the default ``CarbonConfig``.  We cache that result to
    ``<cache_dir>/<backend>/<key>.npz`` (key from :func:`_equilibrium_cache_key`;
    ``<backend>`` = :func:`jax.default_backend` -- see REGIME below) so a re-run
    with the same inputs reloads the WHOLE result in ~1 s -- skipping every
    per-group compile + spin -- instead of re-computing it.

    ``equilibrate_archetypes`` (in ``global_init``) stays PURE (no disk I/O): the
    cache is a driver concern and lives here.  ``rebuild=True``
    (``--rebuild-equilibrium``) forces a recompute + overwrite; an empty
    ``cache_dir`` disables the cache (recompute every launch, unchanged behavior).

    A cache HIT reconstructs the SAME ``(eq, qc)`` the compute returns EXACTLY --
    ``eq`` as a :class:`CarbonState` of per-archetype pools and ``qc`` as the QC
    dict of ``(n_arch,)`` arrays -- so the downstream ``map_to_grid`` and the two
    written ``.npz`` files are byte-identical to a from-scratch build.

    REGIME: the numeric equilibrium is precision- and backend-dependent.  main()
    PINS x64 (so precision is invariant), and the cache path is namespaced by
    ``jax.default_backend()`` so a CPU-built cache never serves a GPU run (or vice
    versa) -- either would break the byte-identical contract.

    A cache file that is corrupt / truncated (a writer killed under an old
    non-atomic version, a disk fault) OR STRUCTURALLY INCOMPLETE (missing an eq
    pool or a QC member, wrong shape/dtype -- validated on read) is DROPPED and
    recomputed rather than crashing every future run OR being served as a partial
    hit (self-healing).
    """
    # Deferred (function-scope) imports keep module import numpy-only and match
    # the equilibration's own deferred-import discipline.
    import jax
    import jax.numpy as jnp

    from legoesm.land.carbon.config import CarbonState

    n_arch = int(np.asarray(table.pft_id).shape[0])
    key = _equilibrium_cache_key(table, spin)
    key8 = key[:8]
    # Namespace by backend so a CPU cache and a GPU cache never cross-serve (their
    # equilibria can differ in the last bits); x64 is pinned in main(), so a
    # `<backend>/` subdir fully scopes the regime.
    path = (Path(cache_dir) / jax.default_backend() / f"{key}.npz"
            if cache_dir else None)

    if path is not None and path.exists() and not rebuild:
        t0 = time.time()
        try:
            with np.load(path) as z:
                # Require EXACTLY the eight equilibrium pools and the five QC
                # members, each a numeric (n_arch,) array (validated), rebuilt in
                # canonical order -- a KeyError (missing member) / ValueError (bad
                # shape or dtype) drops the file and recomputes (never a partial
                # hit).  jnp.asarray happens only AFTER validation, so a bad-dtype
                # array can't slip through as an uncaught TypeError.
                eq = CarbonState(**{
                    p: jnp.asarray(_check_cached_array(z[f"eq_{p}"], f"eq_{p}", n_arch))
                    for p in CarbonState._fields})
                qc = {k: _check_cached_array(z[f"qc_{k}"], f"qc_{k}", n_arch)
                      for k in _QC_KEYS}
        except (OSError, ValueError, KeyError, EOFError, TypeError,
                zipfile.BadZipFile) as exc:
            print(f"[global_carbon_ic] equilibrium cache {key8} unreadable "
                  f"({type(exc).__name__}); recomputing", flush=True)
            try:
                path.unlink()
            except OSError:
                pass
        else:
            print(f"[global_carbon_ic] equilibrium CACHE HIT ({key8}) loaded in "
                  f"{time.time() - t0:.1f}s", flush=True)
            return eq, qc

    # MISS (absent / unreadable / --rebuild-equilibrium) or cache disabled: compute.
    t0 = time.time()
    eq, qc = equilibrate_archetypes(
        table, n_spinup=spin["n_spinup"], n_verify=spin["n_verify"],
        dt=spin["dt"], n_layers=spin["n_layers"], soil_depth=spin["soil_depth"])
    elapsed = time.time() - t0
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save the eight equilibrium pools (eq_<field>) + the five QC members
        # (qc_<key>, canonical order); reading qc[k] for the fixed _QC_KEYS keeps
        # save + load symmetric and fails LOUD if the compute ever drops a member.
        arrays = {f"eq_{p}": np.asarray(getattr(eq, p)) for p in eq._fields}
        arrays.update({f"qc_{k}": np.asarray(qc[k]) for k in _QC_KEYS})
        # Write to an EXCLUSIVELY-created unique temp file in the same directory,
        # then os.replace (ATOMIC within a dir) onto the final path.  tempfile
        # guarantees the temp name is unique even across nodes/containers sharing a
        # PID on the advertised SHARED filesystem (os.getpid() alone is not), so
        # concurrent writers own distinct complete files and a reader only ever
        # sees a complete file or none -- never a truncated npz.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f"{path.stem}.", suffix=".tmp.npz")
        os.close(fd)  # np.savez reopens by name
        tmp = Path(tmp_name)
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        print(f"[global_carbon_ic] equilibrated {elapsed:.1f}s (saved cache {key8})",
              flush=True)
    else:
        print(f"[global_carbon_ic] equilibrated {elapsed:.1f}s (cache disabled)",
              flush=True)
    return eq, qc


# ===========================================================================
# CLI + orchestration
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- archetype / spin-up controls (design defaults) ---
    p.add_argument("--resolution-deg", type=float, default=1.0,
                   help="target regular lat-lon resolution [deg] (real-data path)")
    p.add_argument("--k-per-pft", type=int, default=12,
                   help="climate archetypes per PFT (upper bound; k-means)")
    p.add_argument("--w-min", type=float, default=0.05,
                   help="min cover weight for a (cell,PFT) pair to be occupied")
    p.add_argument("--n-spinup", type=int, default=200,
                   help="transient spin-up years before the analytic reset")
    p.add_argument("--n-verify", type=int, default=40,
                   help="verification years after the analytic reset")
    p.add_argument("--dt", type=float, default=3600.0,
                   help="sub-daily spin-up timestep [s]")
    p.add_argument("--n-layers", type=int, default=10, help="soil layers")
    p.add_argument("--soil-depth", type=float, default=3.0,
                   help="soil column depth [m]")
    p.add_argument("--seed", type=int, default=0, help="base k-means RNG seed")
    p.add_argument("--output", type=str, default="results/global_carbon_ic",
                   help="output DIRECTORY for the two .npz files")
    # --- real-data cover / soil inputs (ignored under --dry-run-synthetic) ---
    p.add_argument("--surfdata-preset", type=str, default="clm5_surfdata",
                   choices=("clm5_surfdata", "legoesm_surfdata"),
                   help="cover/soil source: raw CLM5 surfdata (native grid) or "
                        "harmonized legoesm_surfdata (regridded to --resolution-deg)")
    p.add_argument("--surf-path", type=str, default="",
                   help="surfdata NetCDF path (raw CLM5 file, or harmonized "
                        "legoesm_surfdata for --surfdata-preset legoesm_surfdata)")
    p.add_argument("--veg-path", type=str, default="",
                   help="monthly-veg NetCDF (legoesm_surfdata preset only; "
                        "defaults to --surf-path)")
    # --- climate inputs ---
    p.add_argument("--climate-from-latitude", action="store_true",
                   help="ZONAL DEMONSTRATION climate: build the monthly T/precip/"
                        "SW/netrad from each cell's latitude (no climatology NetCDF)")
    p.add_argument("--climatology", type=str, default="",
                   help="monthly-climatology NetCDF (12-month T/precip/SW/netrad); "
                        "science-grade climate (omit with --climate-from-latitude)")
    p.add_argument("--clim-t-var", type=str, default="tas",
                   help="climatology 2 m air-temperature variable name")
    p.add_argument("--clim-precip-var", type=str, default="pr",
                   help="climatology precipitation-rate variable name [kg/m2/s]")
    p.add_argument("--clim-sw-var", type=str, default="rsds",
                   help="climatology down-shortwave variable name [W/m2]")
    p.add_argument("--clim-netrad-var", type=str, default="netrad",
                   help="climatology surface net-radiation variable name [W/m2]")
    p.add_argument("--clim-lat", type=str, default="lat",
                   help="climatology latitude coordinate name")
    p.add_argument("--clim-lon", type=str, default="lon",
                   help="climatology longitude coordinate name")
    p.add_argument("--clim-t-in-celsius", action="store_true",
                   help="climatology T is in Celsius (add constants.T_freeze)")
    p.add_argument("--dry-run-synthetic", action="store_true",
                   help="fabricate a tiny world and run the full pipeline with no files")
    # --- persistent caches (mirror scripts/run/train_carbon_params.py) ---
    p.add_argument("--compilation-cache-dir",
                   default=os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
                   help="dir for JAX's PERSISTENT on-disk compilation cache so the "
                        "~19-min COLD compile of the per-(is_woody,is_evergreen,"
                        "soil_class)-group coupled-land-model graphs is written ONCE "
                        "and reused across launches (a re-run with the same code/backend "
                        "drops to seconds). MUST be a SHARED-filesystem path when a "
                        "compute node writes it. Defaults to $JAX_COMPILATION_CACHE_DIR; "
                        "empty = disabled (unchanged behavior).")
    p.add_argument("--cache-min-compile-secs", type=float,
                   default=_CACHE_MIN_COMPILE_SECS_DEFAULT,
                   help="(--compilation-cache-dir) cache only XLA compiles slower than "
                        f"this [s] (default {_CACHE_MIN_COMPILE_SECS_DEFAULT:g}); the "
                        "per-group coupled-land graphs are the slow ones, so keep this "
                        "small or the cache stays empty.")
    p.add_argument("--equilibrium-cache-dir",
                   default=os.environ.get("CARBON_EQUILIBRIUM_CACHE_DIR", ""),
                   help="dir for the deterministic RESULT cache of the archetype "
                        "equilibration: its (eq, qc) is a PURE function of the archetype "
                        "table + spin config at the DEFAULT CarbonConfig, cached per "
                        "(table + spin) key so a re-run reloads the whole result in ~1 s "
                        "instead of re-spinning every group. Defaults to "
                        "$CARBON_EQUILIBRIUM_CACHE_DIR; empty = disabled (always "
                        "recompute, unchanged behavior).")
    p.add_argument("--rebuild-equilibrium", action="store_true",
                   help="force a recompute + overwrite of the equilibrium RESULT cache "
                        "(--equilibrium-cache-dir). Also bump _EQUILIBRIUM_CACHE_VERSION "
                        "when the spin-up physics or default params change (the key is "
                        "over inputs, not code).")
    return p


def main(argv=None):
    """Build the global carbon IC and write both .npz files; returns their paths."""
    args = build_arg_parser().parse_args(argv)

    # PIN x64 BEFORE any JAX op: the semi-analytic carbon spin-up is a float64
    # pipeline (like the calibration trainer's train(), which pins the same), and
    # pinning here makes the equilibrium precision INVARIANT regardless of the
    # JAX_ENABLE_X64 env -- so the deterministic result cache can never serve a
    # float32-built equilibrium into a float64 run (or vice versa), which would
    # violate its byte-identical contract.  Must precede every JAX op below.
    import jax
    jax.config.update("jax_enable_x64", True)

    # Enable JAX's PERSISTENT on-disk compilation cache BEFORE the first JAX op
    # (zonal_monthly_climate's vmap on the real path; the per-group coupled-land
    # graph in equilibrate_archetypes). The equilibration cold-compiles a separate
    # coupled-land-model XLA graph per (is_woody, is_evergreen, soil_class) group --
    # ~19 min/group on a diverse real archetype set -- so a shared on-disk cache
    # turns every re-run's compile into a seconds-long disk reuse. Empty dir =>
    # no-op (default behavior). Deferred import keeps module import numpy-only.
    # Mirrors scripts/run/train_carbon_params.py.
    from legoesm.ml.training import configure_jax_compilation_cache
    compile_cache_dir = configure_jax_compilation_cache(
        args.compilation_cache_dir, args.cache_min_compile_secs)
    if compile_cache_dir is not None:
        print(f"[global_carbon_ic] JAX persistent compilation cache: "
              f"{compile_cache_dir} (caching compiles > "
              f"{args.cache_min_compile_secs:g}s) — per-group coupled-land-model "
              "compiles are written once and reused across launches.", flush=True)

    if args.dry_run_synthetic:
        print("[global_carbon_ic] --dry-run-synthetic: fabricating a tiny world")
        inputs = build_synthetic_inputs()
    else:
        inputs = load_real_inputs(args)

    ncell = inputs.pft_weights.shape[0]
    print(f"[global_carbon_ic] {ncell} cells, {inputs.n_pft} PFTs, "
          f"{int(inputs.land_mask.sum())} land cells")

    # Stage A: climate features -> archetypes.
    features = reduce_climatology_to_features(
        inputs.monthly_t_k, inputs.monthly_precip, inputs.monthly_sw,
        inputs.monthly_netrad)
    table, cell_id, cell_w = build_archetypes(
        inputs.pft_weights, features, inputs.soil_class, inputs.land_mask,
        k_per_pft=args.k_per_pft, w_min=args.w_min, seed=args.seed)
    n_arch = int(np.asarray(table.pft_id).shape[0])
    print(f"[global_carbon_ic] built {n_arch} archetypes "
          f"(k_per_pft={args.k_per_pft}, w_min={args.w_min})")

    # Audit the land cover the map silently drops: PFT fractions below w_min get
    # no archetype (zero carbon), and many small fractions can sum to material
    # area.  Print the loss so it is visible (F3; a known Stage-A approximation
    # documented in docs/land/carbon_equilibrium_audit.md).
    mean_drop, max_drop = dropped_cover_fraction(
        inputs.pft_weights, inputs.land_mask, w_min=args.w_min)
    print(f"[global_carbon_ic] dropped sub-w_min vegetated cover (omitted from "
          f"the map): mean {mean_drop * 100.0:.2f}%, max {max_drop * 100.0:.2f}% "
          f"of land area (known Stage-A approximation)")

    # Stage B: spin every archetype to a verified equilibrium, THROUGH the
    # deterministic result cache -- a re-run with the same archetype table + spin
    # config reloads the (eq, qc) in ~1 s instead of re-paying the per-group cold
    # compile + spin-up (the load-bearing shield; see _load_or_equilibrate).
    print(f"[global_carbon_ic] equilibrating (n_spinup={args.n_spinup}, "
          f"n_verify={args.n_verify}, dt={args.dt}s, n_layers={args.n_layers}, "
          f"soil_depth={args.soil_depth}m) ...")
    spin = {
        "n_spinup": args.n_spinup, "n_verify": args.n_verify, "dt": args.dt,
        "n_layers": args.n_layers, "soil_depth": args.soil_depth,
    }
    eq, qc = _load_or_equilibrate(
        table, spin, cache_dir=args.equilibrium_cache_dir,
        rebuild=args.rebuild_equilibrium)

    # Stage C: cover-weighted map of archetype equilibria onto the grid.
    grid_state = map_to_grid(cell_id, cell_w, eq)

    _print_qc_summary(table, qc, n_arch)

    finidat_path, archetypes_path = _write_outputs(
        args.output, inputs, grid_state, table, eq, qc, args.w_min,
        float(args.resolution_deg),
        n_layers=args.n_layers, soil_depth=args.soil_depth, dt=args.dt)
    print(f"[global_carbon_ic] wrote {finidat_path}")
    print(f"[global_carbon_ic] wrote {archetypes_path}")
    return finidat_path, archetypes_path


if __name__ == "__main__":
    main()

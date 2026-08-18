"""Global CLM reference surface map → default plant + soil land parameters.

Reads a CLM (CESM) ``surfdata`` file — the community reference dataset — and maps
it onto the model grid:

* **PFT classification + spatial variation**: ``PCT_NAT_PFT`` (natural PFT %) +
  ``PCT_NATVEG`` / ``PCT_CROP`` → per-column fractions over the 17 CLM5 PFTs, fed
  to the existing :class:`~legoesm.land.param_providers.PFTParamProvider` so the
  vegetation parameters (albedo, z0, root depth, Vc_max, …) come from the CLM5 PFT
  table weighted by the real land-cover map.
* **Reference soil parameters**: ``PCT_SAND`` / ``PCT_CLAY`` (root-zone mean) →
  USDA texture (:mod:`legoesm.land.soil_texture`) → van-Genuchten curve →
  per-column wilting point / field capacity, which override the PFT-table
  ``theta_wp`` / ``theta_fc`` (those are soil, not vegetation, properties).

The full van-Genuchten retention parameters (``theta_sat``/``alpha_vg``/``n_vg``/
``K_sat``) are also returned per column for callers that thread a spatial soil
hydraulics field into the Richards solver (follow-up; the solver currently reads a
single ``MultiLayerLandConfig.hydraulics``).

Surfdata is a large NetCDF; fetch once with :func:`download_clm_surfdata` (cached).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

from legoesm import constants
from legoesm.land.surface_params import (
    LandSurfaceParams, clm5_pft_table, CLM5_PFT_NAMES, PARAM_NAMES)
from legoesm.land import soil_texture

# Present-day CLM5 16-PFT surface dataset, 1.9x2.5 deg (CESM public inputdata).
_SURFDATA_URL = (
    "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/"
    "surfdata_map/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000_c170706.nc")
_N_PFT = len(CLM5_PFT_NAMES)            # 17
_ROOTZONE_LAYERS = 5                    # top ~5 CLM soil layers ≈ root zone
_I_CROP_C3 = CLM5_PFT_NAMES.index("crop_c3")
# --- Differentiably-calibrated DEFAULT per-PFT land parameters (PR: land-tuning) ---
# Tuned against ERA5 (skin temperature + forecast albedo) with end-to-end gradients
# under PHYSICAL per-PFT bounds (scripts/run/train_land_params_era5.py), so every
# value is realistic: forests dark (~0.14), bare/desert bright (~0.38), grass/crop
# mid; rooting depth forests deep / grass shallow.  CLM5 PFT order (17).
# Reduces the global ERA5 land T bias +3.1 -> +1.8 K (RMSE 4.4 -> 3.0).  CAVEAT:
# tuned to OFFLINE monthly ERA5 forcing; re-tune with a coupled diurnal cycle for a
# fully coupled run.  Set CLMSurfaceParamProvider(tuned=False) for the raw CLM5 table.
# 2026-08 DUAL-TARGET recalibration (ERA5 latent heat + skin T, MONTHLY-bias-first
# objective; scripts/run/train_land_params_era5.py, 400 it, full monthly forcing):
# monthly skin-T bias 3.46 -> 1.70 K, monthly LE bias 7.08 -> 3.15 W/m2; annual
# per-cell (all 5551 land cells): skin-T bias +3.33 -> +1.61 K, LE -5.9 -> -1.3.
# root_depth is NOT slab-trainable (no gradient path: slab stress is beta(W/W_max))
# -> kept from the earlier bake.  See docs/land/land_dual_target_calibration_runbook.md.
_TUNED_PFT_ALBEDO = (0.3902, 0.1391, 0.1390, 0.1592, 0.1592, 0.1595, 0.1696, 0.1700,
                     0.1700, 0.2184, 0.2191, 0.2191, 0.2280, 0.2394, 0.2482, 0.2548, 0.2359)
_TUNED_PFT_EMISSIVITY = (0.9869, 0.9885, 0.9863, 0.9889, 0.9888, 0.9888, 0.9888, 0.9880,
                         0.9851, 0.9888, 0.9888, 0.9889, 0.9792, 0.9821, 0.9889, 0.9875, 0.9806)
_TUNED_PFT_ROOT_DEPTH = (0.10, 2.00, 1.50, 1.50, 1.50, 1.80, 1.50, 1.50, 1.20,
                         0.80, 0.80, 0.80, 0.50, 0.50, 0.50, 0.50, 0.50)
_TUNED_PFT_WMAX = (282.1, 308.3, 301.5, 274.4, 312.2, 310.8, 310.7, 309.5, 249.5,
                   311.7, 306.8, 290.3, 240.3, 302.6, 311.4, 310.3, 306.1)
# tuned snow/ice + bulk parameters (config-level; applied on the CLM default path).
TUNED_GLACIER_ALBEDO = 0.7318     # snow-free ice-sheet base albedo
TUNED_SNOW_ALBEDO_MAX = 0.8388    # LandAlbedoConfig.alpha_snow_max
TUNED_CH = 0.004575              # LandConfig.Ch_land / Cd_land bulk transfer

# --- MULTILAYER (8-layer Richards) default per-PFT + snow/ice parameters ----------
# Calibrated against 24-hour ERA5 with the coupled diurnal surface (MOST exchange ->
# roughness z0 trainable) and the PER-CELL TEXTURE soil thermal model (k_solid/C_solid
# from sand/clay x a per-PFT scale -> seasonal cycle); plus albedo/emissivity/root and
# the PLANT btran water-stress thresholds theta_wp/theta_fc (distinct from the soil
# van-Genuchten retention).  Full-grid: RMSE 3.13 K, bias +0.44, seasonal-amp RMS 4.05.
# Selected via CLMSurfaceParamProvider(variant="multilayer"); slab unchanged.  (17 PFTs.)
# 2026-07 recalibration (v4): 24-h ERA5, MINI-BATCH SGD over the FULL global land grid
# (~5.4k cells), 80/20 train/test split, corrected albedo physics (Niu-Yang tanh snow
# cover + trainable snow albedo + CLM dry-soil brightening) AND the per-cell CLM
# soil-COLOUR bare-soil albedo (bright deserts).  HELD-OUT (1087 never-trained cells):
# skin-T RMSE 2.63 -> 2.26 K, bias +0.53 -> +0.30 K; albedo RMSE 0.110 -> 0.104, bias
# +0.011 -> +0.006.  Regional albedo bias vs the first tune: Antarctica -0.146 -> -0.061,
# NH>55 -0.060 -> +0.019, Sahara -0.066 -> -0.007 (soil colour), Tibet -0.088 -> -0.106
# (worse: the correct dark Tibetan soil now shows through the sub-grid-snow deficit).
# (pft_ch is inert under MOST -> kept from the constant-Ch bake.)
# 2026-07 recalibration (v5): re-tuned WITH the field-capacity gravity-drainage limiter
# on (RichardsConfig.fc_drain_saturation=0.5, #862), warm-started from the v4 tune via the
# new --init-json path (#891) — the correct "beat the current best" baseline (from the
# baked prior the offline loss is flat).  Full-grid controlled eval (era5_hourly, limiter
# on): global skin-T bias +0.682 -> +0.589 K, N-America JJA +1.365 -> +1.188, Sahara
# +2.072 -> +1.877, T-RMSE 2.478 -> 2.470; cost albedo RMSE +0.0026 (the optimizer's
# endorsed trade-off).  Soil moisture unchanged (structural, not param-tunable).  smscale +
# elevation-band params left at v4 defaults (worth only +0.001 K, not baked here).
# 2026-08 recalibration (v6, DUAL TARGET): ERA5 latent heat + skin T + soil moisture,
# MONTHLY-bias-first objective (train_multilayer_land_era5.py, mini-batch full grid,
# 900 it, warm-started from v5; docs/land/land_dual_target_calibration_runbook.md).
# Farquhar stomata ON -> Vc_max25/g1/LCMA calibrated for the first time (new
# _TUNED_PFT_{VCMAX,G1,LCMA}_MULTILAYER below).  Full-grid annual per-cell:
# skin-T bias +1.74 -> +0.36 K (RMSE 2.26 -> 1.53), albedo -0.047 -> -0.004,
# LE -12.4 -> -0.3 W/m2; monthly-mean biases T 0.38 K / LE 1.0 W/m2.  HELD-OUT
# (1110 never-trained cells): skin-T RMSE 2.53 -> 2.05 K, bias +1.61 -> +0.34 K.
_TUNED_PFT_ALBEDO_MULTILAYER = (0.3000, 0.1186, 0.1079, 0.0951, 0.1123, 0.1535, 0.1597, 0.1700, 0.1700, 0.2071, 0.2457, 0.1597, 0.1746, 0.1763, 0.1717, 0.1695, 0.1898)
_TUNED_PFT_EMISSIVITY_MULTILAYER = (0.9797, 0.9558, 0.9461, 0.9437, 0.9490, 0.9747, 0.9648, 0.9587, 0.9530, 0.9816, 0.9839, 0.9551, 0.9557, 0.9576, 0.9610, 0.9662, 0.9693)
_TUNED_PFT_ROOT_DEPTH_MULTILAYER = (0.082, 1.887, 0.996, 0.927, 1.193, 1.973, 1.429, 1.557, 0.832, 0.800, 0.497, 0.507, 0.386, 0.448, 0.379, 0.399, 0.508)  # const-ok: baked per-PFT root-depth calibration table (v5 multilayer tuning, #892), not a physical constant — same class as the annotated-by-budget _TUNED_PFT_*_MULTILAYER siblings above
# per-PFT roughness length z0 [m] (drives the MOST surface exchange -> tall forests
# rough ~1-2 m, grass/crop/bare smooth ~0.02-0.23 m).  Calibrated under MOST (the
# coupled diurnal-surface default); the constant-bulk fallback ignores it.
_TUNED_PFT_Z0_MULTILAYER = (0.0060, 1.6345, 2.1374, 2.5714, 2.3211, 1.9324, 2.2620, 2.0291, 1.3602, 0.0473, 0.6631, 0.6884, 0.2739, 0.0666, 0.1521, 0.1177, 0.0936)
# per-PFT bulk heat/moisture exchange coefficient [-] (constant-bulk fallback only;
# inert under the MOST default -> retained from the constant-Ch calibration).
_TUNED_PFT_CH_MULTILAYER = (0.003113, 0.005482, 0.004555, 0.005126, 0.005625, 0.005629, 0.005195, 0.005527, 0.004722, 0.002040, 0.005474, 0.004351, 0.005020, 0.004351, 0.005545, 0.004121, 0.003000)
# per-PFT calibration SCALE on the per-cell texture-derived soil thermal properties
# (k_solid / C_solid from sand/clay, Oleson 2013): the per-cell texture sets the
# spatial pattern, the per-PFT scale sets the magnitude.  k_scale ~0.2-0.5 brings the
# physical mineral k (3-9 W/m/K) down to the effective seasonal-cycle value.
_TUNED_PFT_KSCALE_MULTILAYER = (0.1500, 0.1605, 0.3553, 0.1336, 0.2352, 0.2513, 0.2254, 0.2378, 0.2261, 0.1409, 0.1323, 0.1929, 0.2530, 0.1299, 0.1460, 0.1927, 0.1543)
_TUNED_PFT_CSCALE_MULTILAYER = (0.5125, 0.6662, 0.8402, 0.4603, 0.5173, 0.6205, 0.7375, 0.7221, 0.7893, 0.5085, 0.3963, 0.6233, 0.7059, 0.3781, 0.4921, 0.6094, 0.4964)
# per-PFT PLANT btran water-stress thresholds (wilting / field capacity) [m3/m3]
_TUNED_PFT_WP_MULTILAYER = (0.0703, 0.0656, 0.0708, 0.0700, 0.1159, 0.0610, 0.0720, 0.0734, 0.0772, 0.0749, 0.0564, 0.0695, 0.0620, 0.0658, 0.0616, 0.0660, 0.0780)
_TUNED_PFT_FC_MULTILAYER = (0.1342, 0.1409, 0.1611, 0.1579, 0.2911, 0.1245, 0.1663, 0.1712, 0.1817, 0.1886, 0.1056, 0.1563, 0.1311, 0.1443, 0.1295, 0.1532, 0.1826)
TUNED_GLACIER_ALBEDO_MULTILAYER = 0.7981    # snow-free ice-sheet base (raised: ERA5 ~0.85)
TUNED_SNOW_ALBEDO_MAX_MULTILAYER = 0.8173
# aged-snow albedo floor, tanh snow-cover SWE half-scale [kg/m2], snow-albedo age
# e-folding [days], and the CLM dry-soil albedo brightening (deserts) — the snow/soil
# albedo processes the v3 recalibration made trainable to close the high-lat / ice-sheet
# / desert albedo bias.  snow_dcrit is now the Niu-Yang tanh half-cover scale (NOT the
# old linear full-cover depth).
TUNED_SNOW_ALBEDO_MIN_MULTILAYER = 0.6841
TUNED_SNOW_DCRIT_MULTILAYER = 20.2123       # kg/m2 for full snow cover
TUNED_SNOW_TAU_DAYS_MULTILAYER = 11.2605   # snow-albedo age e-folding [days]
TUNED_SOIL_DRY_BOOST_MULTILAYER = 0.1515   # CLM dry-soil albedo brightening
# scale on the per-cell CLM soil-COLOUR bare-soil albedo (~1 -> the raw MODIS-calibrated
# soil colour is right; gives the model CLM's bright-desert skill).
TUNED_SOIL_ALB_SCALE_MULTILAYER = 1.3393
# deep-ice thermal-inertia boost on glacier cells (on top of pure-ice C=rho_ice*c_pi):
# parameterises the large thermal mass of a deep ice sheet that the finite 3 m soil
# column under-represents -> damps the polar seasonal over-amplitude.  Calibrated:
# fixes the Antarctica seasonal-amplitude bias +2.62 -> -0.02 K.
TUNED_GLACIER_CBOOST_MULTILAYER = 4.3888
# scalar fallback Ch (PFT mean) for the rare no-grid path where the per-cell map
# (clm_multilayer_ch) cannot be built; the per-cell value is used when a grid exists.
TUNED_CH_MULTILAYER = float(np.mean(_TUNED_PFT_CH_MULTILAYER))
# Farquhar canopy conductance (v6 dual-target LE calibration): EFFECTIVE-conductance
# values fitted to monthly ERA5 latent heat under the offline fixed-humidity scheme —
# g1 ~10 is ~2x the Medlyn physiological envelope (part monthly-VPD convexity, part
# conductance absorbing aerodynamic/humidity error; GLM review 2026-08-18).  Do NOT
# reuse as photosynthetic capacity in the carbon cycle, and do NOT enable coupled
# without a controlled A/B (#741 over-transpiration, predicted worse here).  Coupled
# consumption is TRIPLE-gated: stomata.enabled AND carbon.scheme=="differland" AND a
# prescribed carbon state, none of which AMIP sets by default.  Index 0 = bare soil.
_TUNED_PFT_VCMAX_MULTILAYER = (0.00, 76.12, 70.07, 76.23, 64.34, 64.80, 45.75, 70.77, 63.87, 43.31, 61.90, 56.41, 61.44, 63.45, 40.55, 66.44, 43.48)
_TUNED_PFT_G1_MULTILAYER = (1.000, 9.950, 10.273, 11.280, 10.351, 9.627, 10.046, 10.637, 10.796, 5.728, 9.964, 9.354, 8.207, 7.642, 7.143, 8.279, 5.116)
_TUNED_PFT_LCMA_MULTILAYER = (67.54, 87.52, 87.49, 88.00, 44.80, 61.56, 58.92, 56.08, 63.96, 42.19, 62.64, 70.94, 44.78, 49.51, 34.23, 46.48, 54.02)
# btran needs theta_fc > theta_wp per PFT; PFT-weighting (a convex combination) then
# preserves the ordering for every mixed cell, so the stress range never inverts.
assert all(fc > wp for wp, fc in zip(_TUNED_PFT_WP_MULTILAYER, _TUNED_PFT_FC_MULTILAYER)), \
    "multilayer plant theta_fc must exceed theta_wp for every PFT"
# every baked per-PFT tuple must have exactly _N_PFT entries (a wrong-length paste is
# the bake's main footgun -> a load-time tripwire instead of a deep matmul error).
assert all(len(t) == _N_PFT for t in (
    _TUNED_PFT_ALBEDO_MULTILAYER, _TUNED_PFT_EMISSIVITY_MULTILAYER,
    _TUNED_PFT_ROOT_DEPTH_MULTILAYER, _TUNED_PFT_Z0_MULTILAYER, _TUNED_PFT_CH_MULTILAYER,
    _TUNED_PFT_KSCALE_MULTILAYER, _TUNED_PFT_CSCALE_MULTILAYER,
    _TUNED_PFT_WP_MULTILAYER, _TUNED_PFT_FC_MULTILAYER,
    _TUNED_PFT_VCMAX_MULTILAYER, _TUNED_PFT_G1_MULTILAYER,
    _TUNED_PFT_LCMA_MULTILAYER)), \
    f"every _TUNED_PFT_*_MULTILAYER tuple must have {_N_PFT} entries"

# Per-variant lookup: snow-free per-PFT (albedo, emissivity, root_depth) columns +
# glacier ice base albedo.  The slab W_max is reused for both (W_max is the bucket
# store of the 1-layer slab; the Richards multilayer ignores it -> it gets no
# gradient in the multilayer calibration, so there is nothing distinct to bake).
_VARIANT_TUNED = {
    "slab": (_TUNED_PFT_ALBEDO, _TUNED_PFT_EMISSIVITY, _TUNED_PFT_ROOT_DEPTH,
             TUNED_GLACIER_ALBEDO),
    "multilayer": (_TUNED_PFT_ALBEDO_MULTILAYER, _TUNED_PFT_EMISSIVITY_MULTILAYER,
                   _TUNED_PFT_ROOT_DEPTH_MULTILAYER, TUNED_GLACIER_ALBEDO_MULTILAYER),
}

# Snow-free albedo of glacier / ice-sheet ice used as the snow-free BASE over
# glacier cells so ice sheets stay bright when summer snow melts (snow feedback
# layers on top) instead of exposing dark bare soil — the "Greenland problem".
_GLACIER_ALBEDO = TUNED_GLACIER_ALBEDO


def _surfdata_candidates(cache: str) -> list[str]:
    """Ordered local paths to probe for the CLM surfdata before any network I/O.

    ``cache`` (node-local /tmp by default) first, then the ``LEGOESM_CLM_SURFDATA``
    env override, then shared-filesystem ``data/clm/surfdata_*.nc`` under the repo
    root and the CWD.  A fresh compute node has an empty /tmp, and worktrees do
    not carry the (gitignored) ``data/`` tree — without these probes every such
    run fell through to the UCAR SVN download, which currently dies with an SSL
    hostname-mismatch and killed whole SLURM jobs at startup (#869/#847 probes).
    """
    import glob
    cands = [cache, os.environ.get("LEGOESM_CLM_SURFDATA", "")]
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
    for base in (repo_root, os.getcwd()):
        # sorted() -> deterministic pick if several surfdata files coexist
        cands.extend(sorted(glob.glob(os.path.join(base, "data", "clm",
                                                   "surfdata_*.nc"))))
    return [c for c in cands if c]


def download_clm_surfdata(cache: str = "/tmp/clm_surfdata.nc") -> str:
    """Return a local CLM surfdata path, downloading to ``cache`` as a last resort.

    Probes local/shared-filesystem candidates first (see
    :func:`_surfdata_candidates`); only if none exists is the UCAR SVN download
    attempted.  A download failure raises an actionable error instead of a bare
    urllib traceback.
    """
    for cand in _surfdata_candidates(cache):
        if os.path.exists(cand):
            return cand
    import urllib.request
    try:
        urllib.request.urlretrieve(_SURFDATA_URL, cache)
    except Exception as exc:
        raise RuntimeError(
            f"CLM surfdata not found locally and the download failed ({exc!r}).\n"
            f"Tried: {_surfdata_candidates(cache)}\n"
            f"URL: {_SURFDATA_URL}\n"
            "Fix: pass --clm-surfdata-path (run scripts), set LEGOESM_CLM_SURFDATA "
            "to an existing surfdata NetCDF, or place one under <repo>/data/clm/."
        ) from exc
    return cache


def _nearest_regrid(src_lat, src_lon, field, tgt_lat_deg, tgt_lon_deg):
    """Nearest-neighbour regrid ``field[...,nlat,nlon]`` (last two axes on the
    source grid) to target columns (1-D tgt_lat/lon in degrees)."""
    src_lat = np.asarray(src_lat); src_lon = np.asarray(src_lon) % 360.0
    tlat = np.asarray(tgt_lat_deg); tlon = np.asarray(tgt_lon_deg) % 360.0
    jlat = np.abs(src_lat[None, :] - tlat[:, None]).argmin(axis=1)
    # Longitude is periodic: use the modular (great-circle-in-lon) distance so a
    # target near the 0/360 seam picks the true nearest source cell across the
    # wrap rather than a within-hemisphere cell up to one grid spacing farther.
    dlon = np.abs(src_lon[None, :] - tlon[:, None])
    dlon = np.minimum(dlon, 360.0 - dlon)
    jlon = dlon.argmin(axis=1)
    return np.asarray(field)[..., jlat, jlon]      # (..., ncol)


# --- CLM soil-colour broadband albedo (Oleson et al. 2013, CLM Tech Note, Table 3.3) ---
# 20 soil-colour classes.  CLM sets the soil colour so the resulting soil albedo matches
# the SATELLITE-OBSERVED (MODIS) albedo, which is why real CLM nails bright deserts.  We
# adopt the same per-cell SATURATED broadband albedo as the model's bare-soil base (the
# spatial pattern); the wet<->dry range is added on top by the moisture-dependent
# dry-soil brightening.  Broadband = 0.5*(visible + near-infrared).
_SOIL_ALBSAT_VIS = np.array([.25,.23,.21,.20,.19,.18,.17,.16,.15,.14,.13,.12,.11,.10,.09,.08,.07,.06,.05,.04])
_SOIL_ALBSAT_NIR = np.array([.50,.46,.42,.40,.38,.36,.34,.32,.30,.28,.26,.24,.22,.20,.18,.16,.14,.12,.10,.08])
_SOIL_ALB_SAT = 0.5 * (_SOIL_ALBSAT_VIS + _SOIL_ALBSAT_NIR)   # per class (index 0..19)


def soil_color_albedo(color_class):
    """Per-cell snow-free bare-soil broadband albedo from the CLM soil-colour class
    (1..20) — the spatially-varying 'dark forest soil vs bright desert soil' pattern that
    a single per-PFT bare-soil albedo cannot represent.  Returns the SATURATED value; the
    dynamic dry brightening is applied separately (see surface_albedo.dry_soil_brightening)."""
    idx = np.clip(np.asarray(color_class).astype(int), 1, 20) - 1  # coeff-ok: CLM soil-colour classes are 1..20 (table-size index clamp)
    return _SOIL_ALB_SAT[idx]


def load_clm_surface(path: str, tgt_lat_deg, tgt_lon_deg) -> dict:
    """Map a CLM surfdata file to the target columns.

    Parameters
    ----------
    path : str — CLM surfdata NetCDF (see :func:`download_clm_surfdata`).
    tgt_lat_deg, tgt_lon_deg : 1-D arrays — target column centres [deg].

    Returns
    -------
    dict with ``pft_fractions`` (ncol, 17), ``theta_wp`` / ``theta_fc`` (ncol,),
    ``texture_index`` (ncol,), the per-column VG arrays
    (``theta_sat``/``alpha_vg``/``n_vg``/``K_sat``/``theta_r``), and ``lai`` (ncol,)
    — the prescribed per-cell LAI climatology (or ``None`` if the surfdata lacks
    ``MONTHLY_LAI``/``PCT_CFT``).
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    slat = ds["LATIXY"].values[:, 0] if "LATIXY" in ds else ds["lsmlat"].values
    slon = ds["LONGXY"].values[0, :] if "LONGXY" in ds else ds["lsmlon"].values

    pct_nat = ds["PCT_NAT_PFT"].values            # (n_natpft, nlat, nlon), % of natveg
    pct_natveg = ds["PCT_NATVEG"].values          # (nlat, nlon), % of gridcell
    pct_crop = ds["PCT_CROP"].values              # (nlat, nlon), % of gridcell
    pct_glacier = ds["PCT_GLACIER"].values        # (nlat, nlon), % of gridcell (ice sheet)
    n_nat = pct_nat.shape[0]
    # --- Spatial LAI climatology (REUSE the canonical CLM5 crop-split reader) ---
    # The two-leaf canopy needs a per-cell LAI; rather than re-read MONTHLY_LAI and
    # PFT-weight with a bespoke (crop_c3-only) cover map, reuse the ONE canonical
    # PCT_CFT-aware 17-PFT weight reconstruction shared with read_clm5_cover_veg
    # (surface_data.sources.clm5_surfdata) so crop cells get the correct c3/c4 split.
    # The per-cell LAI is the grid-cell-mean over that 17-PFT cover of the annual-mean
    # per-PFT MONTHLY_LAI; bare/lake/glacier cover carries LAI~0 so barren cells end
    # up ~0 (no spurious over-shading canopy).  Gated on MONTHLY_LAI + PCT_CFT being
    # present: a minimal/synthetic surfdata lacking them yields None -> the canopy
    # falls back to its scalar default.  Weighting on the native grid then
    # nearest-regridding the scalar is identical to per-PFT-regrid-then-weight.
    # Canonical PCT_CFT-aware 17-PFT cover (shared with read_clm5_cover_veg), used
    # for BOTH the per-cell parameter weighting (fr, below) and the LAI so crop cells
    # get a CONSISTENT c3/c4 split.  None when the surfdata lacks PCT_CFT (minimal /
    # synthetic files) -> fr falls back to the crop_c3-only cover and LAI to None
    # (canopy scalar default).
    pft_frac_pct = None
    if "PCT_CFT" in ds:
        from legoesm.land.surface_data.sources.clm5_surfdata import (
            reconstruct_clm5_pft_frac)
        pft_frac_pct = reconstruct_clm5_pft_frac(   # (n_pft, nlat, nlon) % of gridcell
            pct_natveg, pct_crop, pct_nat, ds["PCT_CFT"].values)
    lai_native = None
    if "MONTHLY_LAI" in ds and pft_frac_pct is not None:
        lai_annual = np.asarray(
            ds["MONTHLY_LAI"].values, dtype=np.float64).mean(axis=0)  # (n_pft, nlat, nlon)
        if lai_annual.shape[0] != _N_PFT:
            raise ValueError(
                f"MONTHLY_LAI has {lai_annual.shape[0]} PFTs, expected {_N_PFT} "
                "(CLM5 17-PFT ordering aligned with the reconstructed cover).")
        lai_native = np.sum(
            (pft_frac_pct / 100.0) * lai_annual, axis=0)   # (nlat, nlon) grid-cell mean
    # root-zone mean sand/clay (top layers)
    sand = ds["PCT_SAND"].values[:_ROOTZONE_LAYERS].mean(0)   # (nlat, nlon)
    clay = ds["PCT_CLAY"].values[:_ROOTZONE_LAYERS].mean(0)
    soil_color = ds["SOIL_COLOR"].values if "SOIL_COLOR" in ds.variables else None
    # Sub-grid elevation std [m] (drives the elevation-band snow scheme); zero
    # (flat) when the surfdata predates STD_ELEV, e.g. a synthetic test map.
    std_elev = ds["STD_ELEV"].values if "STD_ELEV" in ds.variables else None

    # regrid to target columns
    pct_nat_c = _nearest_regrid(slat, slon, pct_nat, tgt_lat_deg, tgt_lon_deg)  # (n_nat, ncol)
    natveg_c = _nearest_regrid(slat, slon, pct_natveg, tgt_lat_deg, tgt_lon_deg)
    crop_c = _nearest_regrid(slat, slon, pct_crop, tgt_lat_deg, tgt_lon_deg)
    glac_c = _nearest_regrid(slat, slon, pct_glacier, tgt_lat_deg, tgt_lon_deg)
    sand_c = _nearest_regrid(slat, slon, sand, tgt_lat_deg, tgt_lon_deg)
    clay_c = _nearest_regrid(slat, slon, clay, tgt_lat_deg, tgt_lon_deg)
    lai_c = (_nearest_regrid(slat, slon, lai_native, tgt_lat_deg, tgt_lon_deg)
             if lai_native is not None else None)              # (ncol,) or None
    pft_frac_pct_c = (
        _nearest_regrid(slat, slon, pft_frac_pct, tgt_lat_deg, tgt_lon_deg)  # (17, ncol) %
        if pft_frac_pct is not None else None)
    ncol = natveg_c.shape[0]
    # per-cell CLM soil-colour bare-soil albedo (fallback to a mid class if the surfdata
    # predates SOIL_COLOR, e.g. a synthetic test map).
    soil_alb_c = (soil_color_albedo(_nearest_regrid(slat, slon, soil_color,
                                                    tgt_lat_deg, tgt_lon_deg))
                  if soil_color is not None else np.full(ncol, float(_SOIL_ALB_SAT[14])))
    std_elev_c = (_nearest_regrid(slat, slon, std_elev, tgt_lat_deg, tgt_lon_deg)
                  if std_elev is not None else np.zeros(ncol))

    # PFT fractions over the 17 CLM5 classes.  With PCT_CFT present, reuse the ONE
    # canonical crop-split cover (so the per-cell params and the LAI weight by the
    # SAME 17-PFT split, incl. crop_c3/crop_c4); else fall back to the crop_c3-only
    # cover for surfdata lacking PCT_CFT.
    if pft_frac_pct_c is not None:
        fr = pft_frac_pct_c.T / 100.0                          # (ncol, 17) fraction of gridcell
    else:
        fr = np.zeros((ncol, _N_PFT))
        fr[:, :n_nat] = (pct_nat_c.T / 100.0) * (natveg_c[:, None] / 100.0)
        fr[:, _I_CROP_C3] += crop_c / 100.0
    # normalise per column (bare-soil floor keeps the weighted avg well-defined
    # where the gridcell is non-vegetated land — lakes/glacier/urban remainder).
    fr[:, 0] += np.maximum(1.0 - fr.sum(1), 0.0)
    fr = fr / np.maximum(fr.sum(1, keepdims=True), 1e-12)

    tex = soil_texture.usda_texture_index(jnp.asarray(sand_c), jnp.asarray(clay_c))
    vg = soil_texture.vg_params_from_index(tex)
    wp, fc = soil_texture.wilting_field_capacity(vg)
    return dict(pft_fractions=jnp.asarray(fr), texture_index=np.asarray(tex),
                glacier_frac=jnp.asarray(np.clip(glac_c / 100.0, 0.0, 1.0)),
                # per-cell %sand/%clay (root-zone mean) -> per-cell soil thermal props
                pct_sand=jnp.asarray(sand_c), pct_clay=jnp.asarray(clay_c),
                # per-cell CLM soil-colour bare-soil albedo (spatial 'bright desert' map)
                soil_albedo=jnp.asarray(soil_alb_c),
                # sub-grid elevation std [m] (elevation-band snow scheme)
                std_elev=jnp.asarray(np.maximum(std_elev_c, 0.0)),
                # per-cell prescribed LAI climatology on the target columns (None if
                # the surfdata lacks MONTHLY_LAI/PCT_CFT) -> LandSurfaceParams.LAI
                lai=(jnp.asarray(lai_c) if lai_c is not None else None),
                theta_wp=wp, theta_fc=fc, **{k: vg[k] for k in vg})


class CLMSurfaceParamProvider(eqx.Module):
    """Land params from the CLM reference map: PFT-weighted CLM5 vegetation params
    with the wilting-point / field-capacity overridden by the reference soil map.

    Mirrors ``PFTParamProvider`` (``__call__() -> LandSurfaceParams``) so it drops
    into ``make_coupler(land_param_provider=...)`` unchanged."""
    pft_fractions: jax.Array          # (ncol, 17)
    soil_theta_wp: jax.Array          # (ncol,)
    soil_theta_fc: jax.Array          # (ncol,)
    glacier_frac: jax.Array           # (ncol,) ice-sheet fraction [0,1]
    raw_table: jax.Array              # (17, 12) per-PFT parameter table (tuned or CLM5)
    _glacier_albedo: float = eqx.field(static=True)   # snow-free ice base albedo
    plant_theta_wp: jax.Array = None  # (ncol,) PFT-weighted PLANT btran wilting (or None)
    plant_theta_fc: jax.Array = None  # (ncol,) PFT-weighted PLANT btran field cap (or None)
    soil_albedo: jax.Array = None     # (ncol,) per-cell CLM soil-colour bare-soil albedo
    _soil_alb_scale: float = eqx.field(static=True, default=1.0)   # tuned scale on it
    lai: jax.Array = None             # (ncol,) prescribed per-cell LAI climatology (or None)

    def __init__(self, pft_fractions, soil_theta_wp, soil_theta_fc, glacier_frac,
                 tuned: bool = True, variant: str = "slab", soil_albedo=None, lai=None):
        self.pft_fractions = pft_fractions
        self.soil_theta_wp = soil_theta_wp
        self.soil_theta_fc = soil_theta_fc
        self.glacier_frac = glacier_frac
        self._glacier_albedo = TUNED_GLACIER_ALBEDO
        self.plant_theta_wp = None
        self.plant_theta_fc = None
        self.soil_albedo = soil_albedo
        self._soil_alb_scale = 1.0
        self.lai = lai
        table = np.asarray(clm5_pft_table())
        if tuned:   # overwrite the calibrated per-PFT columns (physical bounds)
            if variant not in _VARIANT_TUNED:
                raise ValueError(
                    f"unknown tuned variant {variant!r}; expected one of "
                    f"{sorted(_VARIANT_TUNED)}")
            alb, emis, root, glac_alb = _VARIANT_TUNED[variant]
            table = table.copy()
            table[:, PARAM_NAMES.index("albedo_veg")] = alb
            table[:, PARAM_NAMES.index("emissivity")] = emis
            table[:, PARAM_NAMES.index("root_depth")] = root
            table[:, PARAM_NAMES.index("W_max")] = _TUNED_PFT_WMAX
            self._glacier_albedo = glac_alb
            if variant == "multilayer":  # PLANT btran thresholds override the soil map
                self.plant_theta_wp = pft_fractions @ jnp.asarray(_TUNED_PFT_WP_MULTILAYER)
                self.plant_theta_fc = pft_fractions @ jnp.asarray(_TUNED_PFT_FC_MULTILAYER)
                # calibrated MOST roughness (drives the coupled diurnal exchange)
                table[:, PARAM_NAMES.index("z0")] = _TUNED_PFT_Z0_MULTILAYER
                self._soil_alb_scale = TUNED_SOIL_ALB_SCALE_MULTILAYER
                # v6 Farquhar canopy conductance (dual-target LE calibration);
                # active only when the stomata path runs, harmless otherwise
                table[:, PARAM_NAMES.index("Vc_max25")] = _TUNED_PFT_VCMAX_MULTILAYER
                table[:, PARAM_NAMES.index("g1")] = _TUNED_PFT_G1_MULTILAYER
                table[:, PARAM_NAMES.index("LCMA")] = _TUNED_PFT_LCMA_MULTILAYER
        self.raw_table = jnp.asarray(table)

    def __call__(self) -> LandSurfaceParams:
        vals = self.pft_fractions @ self.raw_table          # (ncol, 12) PFT-weighted
        params = {name: vals[:, i] for i, name in enumerate(PARAM_NAMES)}
        # theta_wp/theta_fc: the PLANT btran thresholds (multilayer variant) when
        # calibrated, else the reference-soil van-Genuchten wilting/field capacity.
        if self.plant_theta_wp is not None:
            params["theta_wp"] = self.plant_theta_wp
            params["theta_fc"] = self.plant_theta_fc
        else:
            params["theta_wp"] = self.soil_theta_wp
            params["theta_fc"] = self.soil_theta_fc
        # Bare soil (PFT 0): use the per-cell CLM soil-COLOUR albedo (spatial desert/soil
        # pattern) x the tuned scale, instead of the single per-PFT bare value — CLM's
        # bright-desert skill, from the MODIS-calibrated soil colour.
        if self.soil_albedo is not None:
            bare = self.pft_fractions[:, 0]
            alb0 = self.raw_table[0, PARAM_NAMES.index("albedo_veg")]
            params["albedo_veg"] = (params["albedo_veg"] - bare * alb0
                                    + bare * self.soil_albedo * self._soil_alb_scale)
        # Glacier / ice-sheet cells: blend the snow-free base albedo toward ice so
        # ice sheets stay bright when summer snow melts (the snow feedback layers on
        # top of this base) instead of exposing dark bare soil — the Greenland fix.
        fg = self.glacier_frac
        params["albedo_veg"] = (1.0 - fg) * params["albedo_veg"] + fg * self._glacier_albedo
        # Prescribed per-cell LAI climatology (PFT-weighted CLM MONTHLY_LAI with the
        # crop c3/c4 split; barren/glacier cells ~0) -> the two-leaf canopy reads this
        # instead of a spurious uniform LAI.  None-safe: LAI stays None for a surfdata
        # without MONTHLY_LAI (the canopy then uses its scalar default).
        if self.lai is not None:
            params["LAI"] = jnp.maximum(jnp.asarray(self.lai), 0.0)
        return LandSurfaceParams(**params)


def clm_hydraulics_config(surface_map: dict):
    """Per-column :class:`SoilHydraulicsConfig` from a :func:`load_clm_surface`
    result — the reference-soil van-Genuchten retention map, shaped ``(ncol, 1)``
    so it broadcasts over soil layers in the Richards/thermal solvers (the solver
    is per-column heterogeneous; per-layer would need ``(ncol, n_layers)``).

    Drop into ``MultiLayerLandConfig(hydraulics=...)`` for spatially-varying soil
    hydrology; the non-VG fields keep their (scalar) defaults."""
    from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
    col = lambda k: jnp.asarray(surface_map[k]).reshape(-1, 1)
    return SoilHydraulicsConfig(
        theta_r=col("theta_r"), theta_sat=col("theta_sat"),
        alpha_vg=col("alpha_vg"), n_vg=col("n_vg"), K_sat=col("K_sat"))


def multilayer_thermal_arrays(pft_fractions, pct_sand, pct_clay, glacier_frac,
                              kscale, cscale, glacier_cboost):
    """Per-cell soil thermal (k_solid, C_soil) as the texture base x per-PFT scale,
    blended toward ICE on glacier-fraction cells.  Pure (arrays in -> arrays out) so
    the offline calibrator and the bake share ONE definition (no re-derivation).

    The glacier blend fixes the polar over-amplitude: an ice sheet is a deep, large
    thermal mass, but the finite 3 m column under-damps the seasonal wave, so the
    glacier heat capacity carries a calibrated deep-ice boost (>1) on top of the pure-
    ice C = rho_ice * c_pi; conductivity blends toward k_ice."""
    from legoesm.land.soil_texture import (
        soil_solid_conductivity, soil_solid_heat_capacity)
    pft = jnp.asarray(pft_fractions)
    k_soil = soil_solid_conductivity(pct_sand, pct_clay) * (pft @ jnp.asarray(kscale))
    c_soil = soil_solid_heat_capacity(pct_sand, pct_clay) * (pft @ jnp.asarray(cscale))
    fg = jnp.clip(jnp.asarray(glacier_frac), 0.0, 1.0)   # guard regrid boundary values
    k_ice = constants.k_ice_default
    c_ice = constants.rho_ice * constants.c_pi * glacier_cboost
    k_eff = (1.0 - fg) * k_soil + fg * k_ice
    c_eff = (1.0 - fg) * c_soil + fg * c_ice
    return k_eff, c_eff


def clm_multilayer_thermal_config(surface_map: dict):
    """Per-column :class:`SoilThermalConfig` (shaped ``(ncol, 1)``): per-cell texture
    soil thermal inertia x per-PFT scale, blended toward ICE on glacier cells (see
    :func:`multilayer_thermal_arrays`).  Drop into ``MultiLayerLandConfig(thermal=)``."""
    from legoesm.land.soil_thermal import SoilThermalConfig
    k_eff, c_eff = multilayer_thermal_arrays(
        surface_map["pft_fractions"], surface_map["pct_sand"], surface_map["pct_clay"],
        surface_map["glacier_frac"], _TUNED_PFT_KSCALE_MULTILAYER,
        _TUNED_PFT_CSCALE_MULTILAYER, TUNED_GLACIER_CBOOST_MULTILAYER)
    return SoilThermalConfig(C_soil=c_eff.reshape(-1, 1), k_solid=k_eff.reshape(-1, 1))


def clm_multilayer_ch(surface_map: dict):
    """PFT-weighted, ERA5-calibrated per-cell bulk heat/moisture exchange coefficient
    (ncol,) for the multilayer CLM default path (``LandConfig.Ch_land``/``Cd_land``)."""
    return jnp.asarray(surface_map["pft_fractions"]) @ jnp.asarray(_TUNED_PFT_CH_MULTILAYER)


def clm_surface_provider(tgt_lat_deg, tgt_lon_deg, surfdata_path: str | None = None,
                         variant: str = "slab") -> CLMSurfaceParamProvider:
    """Build the default CLM PFT + reference-soil parameter provider for the given
    target columns (downloads the surfdata file if ``surfdata_path`` is None).

    ``variant`` selects the baked tuned set: ``"slab"`` (default, 1-layer slab land)
    or ``"multilayer"`` (8-layer Richards land)."""
    path = surfdata_path or download_clm_surfdata()
    m = load_clm_surface(path, tgt_lat_deg, tgt_lon_deg)
    return CLMSurfaceParamProvider(m["pft_fractions"], m["theta_wp"], m["theta_fc"],
                                   m["glacier_frac"], variant=variant, lai=m.get("lai"))


def clm_multilayer_setup(surface_map: dict, base_config=None, variant: str = "multilayer"):
    """The faithful CLM-default MULTILAYER land setup from a :func:`load_clm_surface`
    map: per-column ``(LandSurfaceParams, MultiLayerLandConfig)``.

    Composes the already-factored pieces (no re-derivation): PFT-weighted veg params
    (:class:`CLMSurfaceParamProvider`), reference-soil van-Genuchten hydraulics
    (:func:`clm_hydraulics_config`), and per-cell texture x per-PFT-scale soil thermal
    inertia blended toward ice on glacier cells (:func:`clm_multilayer_thermal_config`).
    ``base_config`` (a ``MultiLayerLandConfig``) supplies the non-spatial defaults
    (soil grid, Richards, carbon, stomata); only ``hydraulics``/``thermal`` are
    overwritten with the spatial maps.  Pure — the model driver and any calibrator
    share ONE definition."""
    from legoesm.land import MultiLayerLandConfig
    base = base_config if base_config is not None else MultiLayerLandConfig()
    provider = CLMSurfaceParamProvider(
        surface_map["pft_fractions"], surface_map["theta_wp"], surface_map["theta_fc"],
        surface_map["glacier_frac"], variant=variant,
        soil_albedo=surface_map.get("soil_albedo"), lai=surface_map.get("lai"))
    # Calibrated SNOW albedo (fresh/aged brightness + cover threshold + age decay) baked
    # ONCE here so every consumer (the AMIP segment model_driver, the CMIP coupled driver,
    # complexity) gets the full 2026-07 recalibration — clm_multilayer_setup is the single
    # multilayer-land setup, so the snow feedback belongs here, not duplicated per driver.
    land_albedo = base.land_albedo._replace(
        alpha_snow_max=TUNED_SNOW_ALBEDO_MAX_MULTILAYER,
        alpha_snow_min=TUNED_SNOW_ALBEDO_MIN_MULTILAYER,
        snow_depth_crit=TUNED_SNOW_DCRIT_MULTILAYER,
        tau_snow_decay=TUNED_SNOW_TAU_DAYS_MULTILAYER * 86400.0,  # days -> s
        soil_dry_albedo_boost=TUNED_SOIL_DRY_BOOST_MULTILAYER)    # CLM dry-soil (deserts)
    cfg = base._replace(
        hydraulics=clm_hydraulics_config(surface_map),
        thermal=clm_multilayer_thermal_config(surface_map),
        land_albedo=land_albedo, snow_albedo_feedback=True)
    return provider(), cfg


# ===========================================================================
# Transient land-use cover (LULC): re-weight the vegetation params by year
# ===========================================================================
# The CLM map above bakes a single-year PFT cover.  For a transient land-use run
# the annual cover comes from a harmonized ``legoesm_surfdata`` (LUH2/HYDE/... —
# see ``surface_data/``), whose ``pft_frac(year, npft, lat, lon)`` is nearest-
# regridded onto the SAME model columns as ``load_clm_surface`` (identical
# ``_nearest_regrid`` targets => cell-for-cell aligned).  Only the PFT-weighted
# VEGETATION params (albedo/z0/root/emissivity/stomata + the plant btran
# thresholds) re-derive from the year's cover; per-cell SOIL hydraulics/thermal/
# Ch and the prescribed LAI stay frozen from the base map (land use changes
# vegetation, not soil texture; transient LAI is a documented follow-up).


def load_transient_cover_on_columns(surfdata_path, tgt_lat_deg, tgt_lon_deg):
    """Read a transient ``legoesm_surfdata`` cover and regrid it onto columns.

    Returns ``(cover, years)`` with ``cover`` shape ``(nyear, ncol, 17)`` — a
    per-column within-land PFT composition summing to 1 (bare-soil floor for empty
    columns), matching :func:`load_clm_surface`'s ``fr`` so the two are
    interchangeable in :class:`CLMSurfaceParamProvider` — and integer ``years``.
    """
    import xarray as xr

    ds = xr.open_dataset(surfdata_path, decode_times=False)
    try:
        slat = np.asarray(ds["lat"].values, dtype=np.float64)
        slon = np.asarray(ds["lon"].values, dtype=np.float64)
        years = np.rint(np.asarray(ds["year"].values)).astype(int)
        # Transpose by NAME to the documented (year, npft, lat, lon): a file stored
        # with a different axis order (or where lat happens to be length 17) would
        # otherwise be silently mis-indexed by the positional regrid below.
        da = ds["pft_frac"].transpose("year", "npft", "lat", "lon")
        pft = np.asarray(da.values, dtype=np.float64)               # (nyear,17,nlat,nlon) %
        if pft.shape[1] != _N_PFT:
            raise ValueError(
                f"transient cover has {pft.shape[1]} PFTs on the npft axis; expected "
                f"the CLM5 {_N_PFT}-PFT axis (same order as load_clm_surface).")
    finally:
        ds.close()

    cov = _nearest_regrid(slat, slon, pft, tgt_lat_deg, tgt_lon_deg)  # (nyear,17,ncol)
    cov = np.moveaxis(cov, 1, -1) / 100.0                            # (nyear,ncol,17) frac
    # Per-column within-land composition (sum 1), bare-soil floor where the cover
    # is short of a full cell — identical to load_clm_surface's normalisation.
    cov = np.nan_to_num(cov, nan=0.0)
    short = np.maximum(1.0 - cov.sum(axis=-1), 0.0)                  # (nyear,ncol)
    cov[..., 0] += short
    cov = cov / np.maximum(cov.sum(axis=-1, keepdims=True), 1e-12)
    return jnp.asarray(cov), jnp.asarray(years, dtype=float)


def clm_provider_rebuild(surface_map: dict, variant: str = "multilayer",
                         *, include_soil_albedo: bool = True):
    """Return ``rebuild(fracs) -> CLMSurfaceParamProvider``: the SAME per-cell soil,
    glacier, soil-colour albedo and prescribed LAI as ``surface_map`` (frozen),
    with only the PFT cover ``fracs`` (ncol, 17) swapped in.

    Used to re-weight the vegetation params at a new land-use year.  The PFT-
    weighted params (incl. the plant btran wilting/field-capacity thresholds)
    re-derive from ``fracs``; soil texture stays fixed.  ``include_soil_albedo``
    mirrors the consumer's non-transient provider so enabling transient cover with
    an unchanged slice is a no-op: ``clm_multilayer_setup`` uses the soil-colour
    albedo (True), the coupled ``clm_surface_provider`` omits it (pass False)."""
    _soil_albedo = surface_map.get("soil_albedo") if include_soil_albedo else None

    def rebuild(fracs):
        return CLMSurfaceParamProvider(
            jnp.asarray(fracs), surface_map["theta_wp"], surface_map["theta_fc"],
            surface_map["glacier_frac"], variant=variant,
            soil_albedo=_soil_albedo, lai=surface_map.get("lai"))
    return rebuild


class TransientCoverProvider(eqx.Module):
    """Land-param provider whose PFT cover varies by calendar year.

    Wraps a reference provider ``base`` (used when ``year`` is None) plus an annual
    cover series; ``__call__(year=...)`` re-weights the vegetation params at that
    year via a ``rebuild`` closure (soil frozen).  Mirrors the provider protocol
    (callable -> ``LandSurfaceParams``) so it drops into
    ``make_coupler(land_param_provider=...)`` unchanged; ``year_varying`` marks it
    so the coupler forwards the per-segment year."""
    base: eqx.Module
    cover: jax.Array                                  # (nyear, ncol, 17)
    years: jax.Array                                  # (nyear,)
    _rebuild: object = eqx.field(static=True)         # fracs -> provider
    year_varying: bool = eqx.field(static=True, default=True)

    def __call__(self, *, year=None):
        if year is None:
            return self.base()
        from legoesm.land.global_surface_data import interp_annual
        fracs = interp_annual(self.cover, self.years, jnp.asarray(float(year)))
        return self._rebuild(fracs)()

    def at_year(self, year):
        """The reference provider re-baked at ``year`` (for drivers that swap the
        whole provider rather than pass a year each call, e.g. the AMIP attribute)."""
        from legoesm.land.global_surface_data import interp_annual
        fracs = interp_annual(self.cover, self.years, jnp.asarray(float(year)))
        return self._rebuild(fracs)

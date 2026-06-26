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
_TUNED_PFT_ALBEDO = (0.3802, 0.1367, 0.1384, 0.1568, 0.1568, 0.1584, 0.1684, 0.1700,
                     0.1700, 0.2130, 0.2165, 0.2166, 0.2345, 0.2424, 0.2412, 0.2463, 0.1800)
_TUNED_PFT_EMISSIVITY = (0.9844, 0.9845, 0.9843, 0.9844, 0.9844, 0.9842, 0.9844, 0.9845,
                         0.9845, 0.9842, 0.9839, 0.9842, 0.9850, 0.9849, 0.9843, 0.9848, 0.9600)
_TUNED_PFT_ROOT_DEPTH = (0.10, 2.00, 1.50, 1.50, 1.50, 1.80, 1.50, 1.50, 1.20,
                         0.80, 0.80, 0.80, 0.50, 0.50, 0.50, 0.50, 0.50)
_TUNED_PFT_WMAX = (240.6, 282.9, 283.1, 281.6, 284.1, 280.4, 276.2, 281.6, 287.7,
                   280.8, 262.3, 282.4, 262.4, 277.5, 277.6, 280.1, 150.0)
# tuned snow/ice + bulk parameters (config-level; applied on the CLM default path).
TUNED_GLACIER_ALBEDO = 0.7192     # snow-free ice-sheet base albedo
TUNED_SNOW_ALBEDO_MAX = 0.7844    # LandAlbedoConfig.alpha_snow_max
TUNED_CH = 0.004425              # LandConfig.Ch_land / Cd_land bulk transfer

# --- MULTILAYER (8-layer Richards) default per-PFT + snow/ice parameters ----------
# A SEPARATE calibration from the slab set above: the deep-soil forward freezes the
# annual-mean soil moisture and trains the EXTENDED per-PFT set against 24-hour ERA5
# forcing (scripts/run/train_multilayer_land_era5.py) -- besides albedo/emissivity/
# root it tunes the bulk exchange Ch, the soil thermal inertia (C_soil + k_solid ->
# seasonal-cycle amplitude/phase) and the PLANT btran water-stress thresholds
# theta_wp/theta_fc (distinct from the reference-soil van-Genuchten retention).
# Full-grid validation: bias +0.81 K, RMSE 3.28, seasonal-amplitude bias -0.49 K.
# Selected via CLMSurfaceParamProvider(variant="multilayer"); the slab set stays the
# default so slab runs do not regress.  CLM5 PFT order (17).
_TUNED_PFT_ALBEDO_MULTILAYER = (0.2980, 0.1219, 0.1337, 0.1500, 0.1377, 0.1486, 0.1585,
                                0.1700, 0.1700, 0.1548, 0.2018, 0.2112, 0.2207, 0.1961,
                                0.1961, 0.1759, 0.1800)
_TUNED_PFT_EMISSIVITY_MULTILAYER = (0.9593, 0.9618, 0.9543, 0.9681, 0.9581, 0.9581,
                                    0.9581, 0.9614, 0.9555, 0.9624, 0.9614, 0.9651,
                                    0.9584, 0.9581, 0.9581, 0.9581, 0.9600)
_TUNED_PFT_ROOT_DEPTH_MULTILAYER = (0.10, 2.06, 1.34, 1.52, 1.57, 1.90, 1.58, 1.52, 1.00,
                                    0.96, 0.77, 0.57, 0.38, 0.53, 0.53, 0.53, 0.50)
# per-PFT roughness length z0 [m] (drives the MOST surface exchange -> tall forests
# rough ~1-2 m, grass/crop/bare smooth ~0.03-0.26 m).  Calibrated under MOST (the
# coupled diurnal-surface default); the constant-bulk fallback ignores it.
_TUNED_PFT_Z0_MULTILAYER = (0.0051, 0.7293, 1.0641, 0.9487, 2.1092, 1.6183, 2.1092, 0.8978,
                            1.1558, 0.0428, 0.0908, 0.2603, 0.1046, 0.0342, 0.0346, 0.0699, 0.0600)
# per-PFT bulk heat/moisture exchange coefficient [-] (constant-bulk fallback only;
# inert under the MOST default -> retained from the constant-Ch calibration).
_TUNED_PFT_CH_MULTILAYER = (0.003113, 0.005482, 0.004555, 0.005126, 0.005625, 0.005629,
                            0.005195, 0.005527, 0.004722, 0.002040, 0.005474, 0.004351,
                            0.005020, 0.004351, 0.005545, 0.004121, 0.003000)
# per-PFT soil thermal inertia: mineral heat capacity [J/m3/K] + solid conductivity [W/m/K]
_TUNED_PFT_CSOIL_MULTILAYER = (2074110., 2688855., 3239029., 2247210., 1976438., 1847718.,
                               1976438., 2190729., 2183373., 1303132., 2266332., 2730470.,
                               3519647., 1847718., 1976438., 1976438., 2000000.)
_TUNED_PFT_KSOLID_MULTILAYER = (1.965, 2.090, 1.799, 1.560, 1.908, 1.907, 1.908, 2.130,
                                1.749, 1.457, 2.072, 1.340, 1.571, 1.907, 1.908, 1.908, 2.000)
# per-PFT PLANT btran water-stress thresholds (wilting / field capacity) [m3/m3]
_TUNED_PFT_WP_MULTILAYER = (0.1025, 0.1057, 0.0811, 0.0872, 0.1282, 0.1282, 0.1282, 0.1105,
                            0.0899, 0.1722, 0.0955, 0.0707, 0.0710, 0.1066, 0.1066, 0.1066, 0.1000)
_TUNED_PFT_FC_MULTILAYER = (0.2057, 0.2677, 0.2010, 0.2201, 0.3185, 0.3187, 0.3185, 0.2694,
                            0.2389, 0.3944, 0.2378, 0.1586, 0.1608, 0.2673, 0.2671, 0.2671, 0.2500)
TUNED_GLACIER_ALBEDO_MULTILAYER = 0.6528
TUNED_SNOW_ALBEDO_MAX_MULTILAYER = 0.8275
# scalar fallback Ch (PFT mean) for the rare no-grid path where the per-cell map
# (clm_multilayer_ch) cannot be built; the per-cell value is used when a grid exists.
TUNED_CH_MULTILAYER = float(np.mean(_TUNED_PFT_CH_MULTILAYER))
# btran needs theta_fc > theta_wp per PFT; PFT-weighting (a convex combination) then
# preserves the ordering for every mixed cell, so the stress range never inverts.
assert all(fc > wp for wp, fc in zip(_TUNED_PFT_WP_MULTILAYER, _TUNED_PFT_FC_MULTILAYER)), \
    "multilayer plant theta_fc must exceed theta_wp for every PFT"
# every baked per-PFT tuple must have exactly _N_PFT entries (a wrong-length paste is
# the bake's main footgun -> a load-time tripwire instead of a deep matmul error).
assert all(len(t) == _N_PFT for t in (
    _TUNED_PFT_ALBEDO_MULTILAYER, _TUNED_PFT_EMISSIVITY_MULTILAYER,
    _TUNED_PFT_ROOT_DEPTH_MULTILAYER, _TUNED_PFT_Z0_MULTILAYER, _TUNED_PFT_CH_MULTILAYER,
    _TUNED_PFT_CSOIL_MULTILAYER, _TUNED_PFT_KSOLID_MULTILAYER,
    _TUNED_PFT_WP_MULTILAYER, _TUNED_PFT_FC_MULTILAYER)), \
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


def download_clm_surfdata(cache: str = "/tmp/clm_surfdata.nc") -> str:
    """Download the CLM surfdata file to ``cache`` (skip if present). Returns path."""
    if not os.path.exists(cache):
        import urllib.request
        urllib.request.urlretrieve(_SURFDATA_URL, cache)
    return cache


def _nearest_regrid(src_lat, src_lon, field, tgt_lat_deg, tgt_lon_deg):
    """Nearest-neighbour regrid ``field[...,nlat,nlon]`` (last two axes on the
    source grid) to target columns (1-D tgt_lat/lon in degrees)."""
    src_lat = np.asarray(src_lat); src_lon = np.asarray(src_lon) % 360.0
    tlat = np.asarray(tgt_lat_deg); tlon = np.asarray(tgt_lon_deg) % 360.0
    jlat = np.abs(src_lat[None, :] - tlat[:, None]).argmin(axis=1)
    jlon = np.abs(src_lon[None, :] - tlon[:, None]).argmin(axis=1)
    return np.asarray(field)[..., jlat, jlon]      # (..., ncol)


def load_clm_surface(path: str, tgt_lat_deg, tgt_lon_deg) -> dict:
    """Map a CLM surfdata file to the target columns.

    Parameters
    ----------
    path : str — CLM surfdata NetCDF (see :func:`download_clm_surfdata`).
    tgt_lat_deg, tgt_lon_deg : 1-D arrays — target column centres [deg].

    Returns
    -------
    dict with ``pft_fractions`` (ncol, 17), ``theta_wp`` / ``theta_fc`` (ncol,),
    ``texture_index`` (ncol,), and the per-column VG arrays
    (``theta_sat``/``alpha_vg``/``n_vg``/``K_sat``/``theta_r``).
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
    # root-zone mean sand/clay (top layers)
    sand = ds["PCT_SAND"].values[:_ROOTZONE_LAYERS].mean(0)   # (nlat, nlon)
    clay = ds["PCT_CLAY"].values[:_ROOTZONE_LAYERS].mean(0)

    # regrid to target columns
    pct_nat_c = _nearest_regrid(slat, slon, pct_nat, tgt_lat_deg, tgt_lon_deg)  # (n_nat, ncol)
    natveg_c = _nearest_regrid(slat, slon, pct_natveg, tgt_lat_deg, tgt_lon_deg)
    crop_c = _nearest_regrid(slat, slon, pct_crop, tgt_lat_deg, tgt_lon_deg)
    glac_c = _nearest_regrid(slat, slon, pct_glacier, tgt_lat_deg, tgt_lon_deg)
    sand_c = _nearest_regrid(slat, slon, sand, tgt_lat_deg, tgt_lon_deg)
    clay_c = _nearest_regrid(slat, slon, clay, tgt_lat_deg, tgt_lon_deg)
    ncol = natveg_c.shape[0]

    # PFT fractions over the 17 CLM5 classes: natural PFTs 0..n_nat-1 weighted by
    # the gridcell natural-veg fraction; crops -> crop_c3 slot.
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

    def __init__(self, pft_fractions, soil_theta_wp, soil_theta_fc, glacier_frac,
                 tuned: bool = True, variant: str = "slab"):
        self.pft_fractions = pft_fractions
        self.soil_theta_wp = soil_theta_wp
        self.soil_theta_fc = soil_theta_fc
        self.glacier_frac = glacier_frac
        self._glacier_albedo = TUNED_GLACIER_ALBEDO
        self.plant_theta_wp = None
        self.plant_theta_fc = None
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
        # Glacier / ice-sheet cells: blend the snow-free base albedo toward ice so
        # ice sheets stay bright when summer snow melts (the snow feedback layers on
        # top of this base) instead of exposing dark bare soil — the Greenland fix.
        fg = self.glacier_frac
        params["albedo_veg"] = (1.0 - fg) * params["albedo_veg"] + fg * self._glacier_albedo
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


def clm_multilayer_thermal_config(surface_map: dict):
    """Per-column :class:`SoilThermalConfig` with the PFT-weighted, ERA5-calibrated
    soil thermal inertia (mineral heat capacity + solid conductivity), shaped
    ``(ncol, 1)`` so it broadcasts over the soil layers — the multilayer-variant
    seasonal-cycle calibration.  Drop into ``MultiLayerLandConfig(thermal=...)``."""
    from legoesm.land.soil_thermal import SoilThermalConfig
    pft = jnp.asarray(surface_map["pft_fractions"])              # (ncol, 17)
    pw1 = lambda t: (pft @ jnp.asarray(t)).reshape(-1, 1)
    return SoilThermalConfig(C_soil=pw1(_TUNED_PFT_CSOIL_MULTILAYER),
                             k_solid=pw1(_TUNED_PFT_KSOLID_MULTILAYER))


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
                                   m["glacier_frac"], variant=variant)

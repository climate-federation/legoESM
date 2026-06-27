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
# Calibrated against 24-hour ERA5 with the coupled diurnal surface (MOST exchange ->
# roughness z0 trainable) and the PER-CELL TEXTURE soil thermal model (k_solid/C_solid
# from sand/clay x a per-PFT scale -> seasonal cycle); plus albedo/emissivity/root and
# the PLANT btran water-stress thresholds theta_wp/theta_fc (distinct from the soil
# van-Genuchten retention).  Full-grid: RMSE 3.13 K, bias +0.44, seasonal-amp RMS 4.05.
# Selected via CLMSurfaceParamProvider(variant="multilayer"); slab unchanged.  (17 PFTs.)
_TUNED_PFT_ALBEDO_MULTILAYER = (0.2993, 0.1146, 0.1308, 0.1478, 0.1397, 0.1492, 0.1598,
                                0.1700, 0.1700, 0.1430, 0.1997, 0.2114, 0.2098, 0.1978,
                                0.1995, 0.1794, 0.1800)
_TUNED_PFT_EMISSIVITY_MULTILAYER = (0.9597, 0.9556, 0.9559, 0.9640, 0.9597, 0.9589,
                                    0.9597, 0.9642, 0.9583, 0.9510, 0.9597, 0.9631,
                                    0.9597, 0.9589, 0.9597, 0.9597, 0.9600)
_TUNED_PFT_ROOT_DEPTH_MULTILAYER = (0.10, 2.23, 1.32, 1.50, 1.45, 1.85, 1.44, 1.45, 1.09,
                                    0.63, 0.76, 0.57, 0.42, 0.52, 0.48, 0.48, 0.50)
# per-PFT roughness length z0 [m] (drives the MOST surface exchange -> tall forests
# rough ~1-2 m, grass/crop/bare smooth ~0.02-0.23 m).  Calibrated under MOST (the
# coupled diurnal-surface default); the constant-bulk fallback ignores it.
_TUNED_PFT_Z0_MULTILAYER = (0.0051, 1.2662, 0.9216, 0.7534, 2.0903, 1.5671, 2.0903, 0.8359,
                            1.1281, 0.0296, 0.1138, 0.2255, 0.0561, 0.0323, 0.0337, 0.0681, 0.0600)
# per-PFT bulk heat/moisture exchange coefficient [-] (constant-bulk fallback only;
# inert under the MOST default -> retained from the constant-Ch calibration).
_TUNED_PFT_CH_MULTILAYER = (0.003113, 0.005482, 0.004555, 0.005126, 0.005625, 0.005629,
                            0.005195, 0.005527, 0.004722, 0.002040, 0.005474, 0.004351,
                            0.005020, 0.004351, 0.005545, 0.004121, 0.003000)
# per-PFT calibration SCALE on the per-cell texture-derived soil thermal properties
# (k_solid / C_solid from sand/clay, Oleson 2013): the per-cell texture sets the
# spatial pattern, the per-PFT scale sets the magnitude.  k_scale ~0.2-0.5 brings the
# physical mineral k (3-9 W/m/K) down to the effective seasonal-cycle value.
_TUNED_PFT_KSCALE_MULTILAYER = (0.3396, 0.3052, 0.3507, 0.3678, 0.3396, 0.3321, 0.3396,
                                0.3999, 0.2854, 0.2789, 0.3396, 0.2692, 0.2983, 0.3321,
                                0.3396, 0.3396, 0.3500)
_TUNED_PFT_CSCALE_MULTILAYER = (0.8916, 0.7625, 0.9174, 0.9087, 0.8916, 0.8656, 0.8916,
                                0.9877, 0.8414, 0.6982, 0.8916, 1.0472, 1.0335, 0.8656,
                                0.8916, 0.8916, 0.9000)
# per-PFT PLANT btran water-stress thresholds (wilting / field capacity) [m3/m3]
_TUNED_PFT_WP_MULTILAYER = (0.0896, 0.1195, 0.0852, 0.0961, 0.1066, 0.1094, 0.1066, 0.1092,
                            0.1057, 0.1932, 0.0896, 0.0709, 0.0804, 0.0917, 0.0896, 0.0896, 0.1000)
_TUNED_PFT_FC_MULTILAYER = (0.1759, 0.2981, 0.2093, 0.2405, 0.2675, 0.2746, 0.2675, 0.2744,
                            0.2696, 0.4390, 0.2211, 0.1638, 0.1932, 0.2272, 0.2211, 0.2211, 0.2500)
TUNED_GLACIER_ALBEDO_MULTILAYER = 0.6458
TUNED_SNOW_ALBEDO_MAX_MULTILAYER = 0.7985
# deep-ice thermal-inertia boost on glacier cells (on top of pure-ice C=rho_ice*c_pi):
# parameterises the large thermal mass of a deep ice sheet that the finite 3 m soil
# column under-represents -> damps the polar seasonal over-amplitude.  Calibrated:
# fixes the Antarctica seasonal-amplitude bias +2.62 -> -0.02 K.
TUNED_GLACIER_CBOOST_MULTILAYER = 7.474
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
    _TUNED_PFT_KSCALE_MULTILAYER, _TUNED_PFT_CSCALE_MULTILAYER,
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
                # per-cell %sand/%clay (root-zone mean) -> per-cell soil thermal props
                pct_sand=jnp.asarray(sand_c), pct_clay=jnp.asarray(clay_c),
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
                                   m["glacier_frac"], variant=variant)

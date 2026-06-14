"""Build per-column land-model inputs from a regridded :class:`GlobalSurfaceData`.

Turns the harmonized surface data (on a model grid) into the per-column arrays the
multilayer-land + two-leaf-canopy model consumes for a global run:

  - :func:`build_canopy_params` -> :class:`CanopyLandParams` (LAI, canopy height,
    soil-colour background albedo, PFT photosynthesis/aerodynamic params, C4
    flag), one value per column from the **dominant PFT**;
  - :func:`build_soil_hydraulics` -> a per-column :class:`SoilHydraulicsConfig`
    (Cosby pedotransfer from column-mean sand/clay), with a fallback texture where
    HWSD has no soil (sand seas / ice).

The canopy PFT tables (:data:`PFT_VCMAX25_C3` etc.) are keyed by biome type
(ENF/EBF/.../GRA/CRO) with three climate-zone columns ordered
``[boreal, temperate, tropical]``; CLM5's 17 PFTs carry the climate zone in the
name, so :data:`_CLM5_TO_BIOME` maps each CLM5 PFT to ``(biome, zone_index,
is_c4)``.  Host-side; not traced.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.land.surface_params import (
    CLM5_PFT_NAMES, N_PFT_CLM5, clm5_pft_table, array_to_params, PARAM_NAMES,
)
from legoesm.land.canopy.config import (
    CanopyLandParams,
    PFT_VCMAX25_C3,
    PFT_VCMAX25_C4,
    PFT_AERO_PARAMS,
    PFT_CANOPY_HEIGHT,
)
from legoesm.land.soil_albedo import soil_albedo, soil_albedo_broadband
from legoesm.land.pedotransfer import cosby_hydraulic_params
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.global_surface_data import interp_monthly

# zone index into the 3-element biome lists: [boreal, temperate, tropical]
_BOREAL, _TEMPERATE, _TROPICAL = 0, 1, 2

# CLM5 PFT name -> (biome key, climate-zone index, is_c4).  ``None`` biome = bare.
_CLM5_TO_BIOME: dict[str, tuple] = {
    "bare_soil": (None, _TEMPERATE, False),
    "needleleaf_evergreen_temperate": ("ENF", _TEMPERATE, False),
    "needleleaf_evergreen_boreal": ("ENF", _BOREAL, False),
    "needleleaf_deciduous_boreal": ("DNF", _BOREAL, False),
    "broadleaf_evergreen_tropical": ("EBF", _TROPICAL, False),
    "broadleaf_evergreen_temperate": ("EBF", _TEMPERATE, False),
    "broadleaf_deciduous_tropical": ("DBF", _TROPICAL, False),
    "broadleaf_deciduous_temperate": ("DBF", _TEMPERATE, False),
    "broadleaf_deciduous_boreal": ("DBF", _BOREAL, False),
    "broadleaf_evergreen_shrub": ("SHR", _TEMPERATE, False),
    "broadleaf_deciduous_temperate_shrub": ("SHR", _TEMPERATE, False),
    "broadleaf_deciduous_boreal_shrub": ("SHR", _BOREAL, False),
    "c3_arctic_grass": ("GRA", _BOREAL, False),
    "c3_grass": ("GRA", _TEMPERATE, False),
    "c4_grass": ("GRA", _TEMPERATE, True),
    "crop_c3": ("CRO", _TEMPERATE, False),
    "crop_c4": ("CRO", _TEMPERATE, True),
}


def _pft_lookup_arrays() -> dict[str, np.ndarray]:
    """Per-CLM5-PFT (length-17) lookups built from the canopy biome tables."""
    vc3 = np.zeros(N_PFT_CLM5); vc4 = np.zeros(N_PFT_CLM5)
    rz0m = np.zeros(N_PFT_CLM5); rd = np.zeros(N_PFT_CLM5)
    hc = np.zeros(N_PFT_CLM5); fc4 = np.zeros(N_PFT_CLM5); is_veg = np.zeros(N_PFT_CLM5)
    for i, name in enumerate(CLM5_PFT_NAMES):
        biome, zone, c4 = _CLM5_TO_BIOME[name]
        if biome is None:                      # bare soil: no canopy
            continue
        is_veg[i] = 1.0
        fc4[i] = 1.0 if c4 else 0.0
        if c4:
            vc4[i] = PFT_VCMAX25_C4.get(biome, PFT_VCMAX25_C4["GRA"])[zone]
        else:
            vc3[i] = PFT_VCMAX25_C3[biome][zone]
        rz0m[i] = PFT_AERO_PARAMS[biome]["rz0m"]
        rd[i] = PFT_AERO_PARAMS[biome]["rd"]
        hc[i] = PFT_CANOPY_HEIGHT[biome]
    return {"vc3": vc3, "vc4": vc4, "rz0m": rz0m, "rd": rd,
            "hc": hc, "fc4": fc4, "is_veg": is_veg}


def dominant_pft_index(gsd) -> np.ndarray:
    """Dominant PFT index per column from the (year-mean) ``pft_frac``."""
    mean_frac = np.asarray(jnp.mean(gsd.pft_frac, axis=0))   # (ncol, npft)
    return np.argmax(mean_frac, axis=-1)                     # (ncol,)


def _cover1d(a):
    """A cover fraction as ``(ncol,)`` (squeeze a leading single-year axis)."""
    a = np.asarray(a)
    return a[0] if a.ndim == 2 else a


def glacier_mask(gsd) -> np.ndarray:
    """Boolean ``(ncol,)``: columns where glacier is the dominant surface type.

    Greenland / Antarctica / mountain ice: ``f_glacier`` exceeds both the
    soil/veg (``f_land``) and lake fractions.  These are treated as ice surfaces
    (no vegetation, high albedo), not bare soil.
    """
    f_g = _cover1d(gsd.f_glacier)
    return (f_g > _cover1d(gsd.f_land)) & (f_g > _cover1d(gsd.f_lake)) & (f_g > 0.0)


def build_canopy_params(
    gsd,
    day_of_year: float,
    theta_top: jnp.ndarray,
    *,
    tgc_C: float = 25.0,
    glacier_alb_vis: float = 0.70,
    glacier_alb_nir: float = 0.50,
) -> CanopyLandParams:
    """Per-column :class:`CanopyLandParams` from surface data at ``day_of_year``.

    ``theta_top`` (ncol,) is the top-layer volumetric water content (for the
    moisture-dependent soil background albedo).  Vegetation structure (LAI,
    canopy height) is the dominant PFT's value; photosynthesis/aerodynamic params
    come from the PFT biome tables; ``ALB_VIS``/``ALB_NIR`` are the soil-colour
    background albedo.

    Glacier-dominant columns (:func:`glacier_mask`) are set to a bare **ice
    surface**: no vegetation (LAI=0, FNonVeg=1) and a high snow/ice albedo
    (``glacier_alb_vis``/``glacier_alb_nir``) instead of the soil background.
    """
    dom = dominant_pft_index(gsd)                            # (ncol,)
    ncol = dom.shape[0]
    lut = _pft_lookup_arrays()

    # LAI / canopy height of the dominant PFT at this day-of-year.
    lai_m = np.asarray(interp_monthly(gsd.lai_monthly, jnp.asarray(float(day_of_year))))
    htop_m = np.asarray(interp_monthly(gsd.htop_monthly, jnp.asarray(float(day_of_year))))
    cols = np.arange(ncol)
    LAI = lai_m[cols, dom]
    hc_surf = htop_m[cols, dom]
    hc_default = lut["hc"][dom]
    hc = np.where(np.isfinite(hc_surf) & (hc_surf > 0.0), hc_surf, hc_default)

    is_veg = lut["is_veg"][dom]
    LAI = np.where(is_veg > 0.0, np.nan_to_num(LAI, nan=0.0), 0.0)

    _av, _an = soil_albedo(jnp.asarray(np.asarray(gsd.soil_color)), jnp.asarray(theta_top))
    alb_vis, alb_nir = np.asarray(_av), np.asarray(_an)

    # Glacier columns: ice surface — no vegetation, high snow/ice albedo.
    ice = glacier_mask(gsd)
    is_veg = np.where(ice, 0.0, is_veg)
    LAI = np.where(ice, 0.0, LAI)
    alb_vis = np.where(ice, glacier_alb_vis, alb_vis)
    alb_nir = np.where(ice, glacier_alb_nir, alb_nir)

    full = lambda v: jnp.full(ncol, v)
    return CanopyLandParams(
        LAI=jnp.asarray(LAI),
        hc=jnp.asarray(np.maximum(hc, 0.1)),
        fC4=jnp.asarray(lut["fc4"][dom]),
        FNonVeg=jnp.asarray(1.0 - is_veg),       # bare-dominant columns -> non-veg
        CI=full(0.75), kn=full(0.3),
        Vcmax25_C3_leaf=jnp.asarray(lut["vc3"][dom]),
        Vcmax25_C4_leaf=jnp.asarray(lut["vc4"][dom]),
        m_C3=full(9.0), m_C4=full(4.0), b0_C3=full(0.01), b0_C4=full(0.04),
        alf=full(0.3), TgC=full(float(tgc_C)),
        ALB_VIS=jnp.asarray(alb_vis), ALB_NIR=jnp.asarray(alb_nir),
        emissivity=full(0.97),
        rz0m=jnp.asarray(np.where(is_veg > 0.0, lut["rz0m"][dom], 0.01)),
        rd=jnp.asarray(np.where(is_veg > 0.0, lut["rd"][dom], 0.0)),
    )


def build_soil_hydraulics(
    gsd,
    *,
    fallback_sand_pct: float = 92.0,    # default to sand where HWSD has no soil
    fallback_clay_pct: float = 3.0,
    base: SoilHydraulicsConfig = SoilHydraulicsConfig(),
) -> SoilHydraulicsConfig:
    """Per-column Clapp-Hornberger :class:`SoilHydraulicsConfig` (Cosby pedotransfer).

    Texture is the column mean over soil layers; columns where HWSD has no soil
    (all-NaN) fall back to ``fallback_*`` (a sandy default).  Hydraulic params are
    returned as ``(ncol, 1)`` arrays that broadcast against the model's
    ``(ncol, n_layer)`` soil state.
    """
    import warnings

    sand = np.asarray(gsd.sand_frac) * 100.0                 # (ncol, nlayer) percent
    clay = np.asarray(gsd.clay_frac) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN cols -> handled below
        sand_col = np.nanmean(sand, axis=1)
        clay_col = np.nanmean(clay, axis=1)
    bad = ~np.isfinite(sand_col) | ~np.isfinite(clay_col)
    sand_col = np.where(bad, fallback_sand_pct, sand_col)
    clay_col = np.where(bad, fallback_clay_pct, clay_col)

    p = cosby_hydraulic_params(jnp.asarray(sand_col), jnp.asarray(clay_col))
    col = lambda a: jnp.asarray(a)[:, None]                  # (ncol, 1)
    return base._replace(
        retention_curve="clapp_hornberger",
        theta_sat=col(p.theta_sat), psi_sat=col(p.psi_sat),
        b_ch=col(p.b_ch), K_sat=col(p.K_sat), theta_r=0.0,
    )


# ===========================================================================
# Scheme-agnostic adapters: GlobalSurfaceData -> per-scheme land parameters
# ===========================================================================
def build_land_surface_params(
    gsd,
    day_of_year: float,
    theta_top: jnp.ndarray,
    *,
    glacier_albedo: float = 0.6,
):
    """Per-column :class:`LandSurfaceParams` for the SLAB / multilayer-SimpleSEB
    schemes, from the dominant CLM5 PFT.

    Physical parameters (emissivity, z0, W_max, C_soil, d_soil, root_depth,
    theta_wp/fc, Vcmax25, LCMA, g1) are the dominant PFT's row of the CLM5 table;
    ``albedo_veg`` is the soil-colour background blended with the PFT veg albedo by
    canopy cover ``1-exp(-0.5*LAI)``, with glacier columns set to ice albedo.
    """
    dom = dominant_pft_index(gsd)                       # (ncol,) CLM5 17-PFT axis
    ncol = dom.shape[0]
    params = array_to_params(jnp.asarray(np.asarray(clm5_pft_table())[dom]), PARAM_NAMES)

    lai_m = np.asarray(interp_monthly(gsd.lai_monthly, jnp.asarray(float(day_of_year))))
    LAI = np.nan_to_num(lai_m[np.arange(ncol), dom], nan=0.0)
    soil_bg = np.asarray(
        soil_albedo_broadband(jnp.asarray(np.asarray(gsd.soil_color)), jnp.asarray(theta_top)))
    f_veg = 1.0 - np.exp(-0.5 * LAI)
    alb = np.asarray(params.albedo_veg) * f_veg + soil_bg * (1.0 - f_veg)
    alb = np.where(glacier_mask(gsd), glacier_albedo, alb)
    return params._replace(albedo_veg=jnp.asarray(alb))


def surface_data_to_land_params(gsd, surface_scheme, day_of_year, theta_top):
    """Dispatch to the right per-column land-params object for ``surface_scheme``.

    ``TwoLeafCanopyConfig`` -> :class:`CanopyLandParams`; ``SimpleSEBConfig`` (slab
    or multilayer) -> :class:`LandSurfaceParams`.  (clm-ml is an external plugin
    with its own input contract; feed it ``gsd`` directly.)
    """
    from legoesm.land.canopy import CanopyConfig
    if isinstance(surface_scheme, CanopyConfig):
        return build_canopy_params(gsd, day_of_year, theta_top)
    return build_land_surface_params(gsd, day_of_year, theta_top)


def init_land_surface_data(surfdata_path, grid, land_config, day_of_year, *, theta_top=None):
    """Load the surfdata, regrid to ``grid``, and adapt to ``land_config``'s scheme.

    The single entry a driver calls at simulation start.  Returns
    ``(land_config, land_params, gsd)``: for a multilayer config the returned
    config also carries the Cosby soil hydraulics derived from the surfdata
    (global-mean texture for now; see solve_richards shape-hardening TODO).
    ``theta_top`` (top-layer wetness for the soil-colour albedo) defaults to a
    nominal 0.2 when no state exists yet.
    """
    import warnings
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.pedotransfer import soil_hydraulics_config_from_texture
    from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data

    cfg_sd = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=surfdata_path)
    gsd = load_global_surface_data(cfg_sd, grid)
    ncol = int(np.asarray(gsd.soil_color).shape[0])
    if theta_top is None:
        theta_top = jnp.full(ncol, 0.2)

    if isinstance(land_config, MultiLayerLandConfig):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_sand = float(np.nanmean(np.asarray(gsd.sand_frac) * 100.0))
            mean_clay = float(np.nanmean(np.asarray(gsd.clay_frac) * 100.0))
        land_config = land_config._replace(
            hydraulics=soil_hydraulics_config_from_texture(mean_sand, mean_clay))

    land_params = surface_data_to_land_params(
        gsd, land_config.surface_scheme, day_of_year, theta_top)
    return land_config, land_params, gsd


# ===========================================================================
# Reconciliation with the authoritative land-sea mask
# ===========================================================================
# The driver's land-sea mask (sftlf / ERA5 lsm / topography) decides which cells
# are land — NOT the surfdata.  Wherever the mask says land but the surfdata has
# no coverage (small islands, coast mismatch, HWSD/ice gaps), the land model must
# still get *finite* parameters.  These helpers fill such cells (and any residual
# NaN) with a bare-soil fallback, so every column is valid regardless of the mask.
def _bare_land_surface_params(ncol: int):
    """Bare-soil :class:`LandSurfaceParams` (CLM5 PFT 0 row) broadcast to ncol."""
    row = np.asarray(clm5_pft_table())[0]                    # bare_soil (12,)
    return array_to_params(jnp.broadcast_to(jnp.asarray(row), (ncol, row.shape[0])), PARAM_NAMES)


def _bare_canopy_params(ncol: int) -> CanopyLandParams:
    """Bare (no-vegetation) :class:`CanopyLandParams` broadcast to ncol."""
    full = lambda v: jnp.full(ncol, v)
    return CanopyLandParams(
        LAI=full(0.0), hc=full(0.1), fC4=full(0.0), FNonVeg=full(1.0),
        CI=full(0.75), kn=full(0.3), Vcmax25_C3_leaf=full(0.0), Vcmax25_C4_leaf=full(0.0),
        m_C3=full(9.0), m_C4=full(4.0), b0_C3=full(0.01), b0_C4=full(0.04),
        alf=full(0.3), TgC=full(25.0), ALB_VIS=full(0.2), ALB_NIR=full(0.3),
        emissivity=full(0.96), rz0m=full(0.01), rd=full(0.0),
    )


def surfdata_covered(gsd) -> np.ndarray:
    """Boolean ``(ncol,)``: columns the surfdata actually covers (finite pft_frac)."""
    pft = np.asarray(gsd.pft_frac)
    pft0 = pft[0] if pft.ndim == 3 else pft                  # (ncol, npft)
    return np.isfinite(pft0.sum(axis=-1))


def fill_land_param_gaps(land_params, gsd):
    """Replace surfdata-uncovered (and any non-finite) columns with a bare fallback.

    Reconciles the surfdata with the driver's authoritative land mask: a column is
    kept only where the surfdata covers it *and* the value is finite, otherwise it
    falls back to bare soil.  Works for :class:`CanopyLandParams` or
    :class:`LandSurfaceParams`.
    """
    import jax
    covered = jnp.asarray(surfdata_covered(gsd))            # (ncol,)
    ncol = covered.shape[0]
    fb = (_bare_canopy_params(ncol) if isinstance(land_params, CanopyLandParams)
          else _bare_land_surface_params(ncol))

    def _fill(v, f):
        v = jnp.asarray(v)
        keep = covered.reshape(covered.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))

    return jax.tree.map(_fill, land_params, fb)

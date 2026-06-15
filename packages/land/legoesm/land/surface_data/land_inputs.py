"""Build per-column land-model inputs from a regridded :class:`GlobalSurfaceData`.

Turns the harmonized surface data (on a model grid) into the per-column arrays the
multilayer-land + two-leaf-canopy model consumes for a global run:

  - :func:`build_canopy_params` -> :class:`CanopyLandParams` (LAI, canopy height,
    soil-colour background albedo, PFT photosynthesis/aerodynamic params, C4
    flag), one value per column from the **dominant PFT**;
  - :func:`build_soil_hydraulics` -> a per-(col, layer) :class:`SoilHydraulicsConfig`
    (Cosby pedotransfer from the surfdata's depth-remapped sand/clay profile),
    with a per-cell fallback texture where HWSD has no soil (sand seas / ice).

The canopy PFT tables (:data:`PFT_VCMAX25_C3` etc.) are keyed by biome type
(ENF/EBF/.../GRA/CRO) with three climate-zone columns ordered
``[boreal, temperate, tropical]``; CLM5's 17 PFTs carry the climate zone in the
name, so :data:`_CLM5_TO_BIOME` maps each CLM5 PFT to ``(biome, zone_index,
is_c4)``.  Host-side; not traced.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.land.surface_params import (
    CLM5_PFT_NAMES, N_PFT_CLM5, clm5_pft_table, array_to_params, PARAM_NAMES,
    LandSurfaceParams,
)
from legoesm.land.param_providers import PFTParamProvider
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

# --- Canopy / Ball-Berry column defaults (CLM5 tech note; Ball-Berry 1987;
#     Bonan 2019 two-leaf canopy) — fill CanopyLandParams columns the surfdata /
#     PFT tables do not prescribe per-column. ---
_CI_DEFAULT = 0.75          # clumping index [-]
_KN_DEFAULT = 0.3           # nitrogen extinction coefficient [-]
_ALF_DEFAULT = 0.3          # quantum yield [mol CO2 / mol photon]
_M_C3 = 9.0                 # Ball-Berry slope, C3 [-]
_M_C4 = 4.0                 # Ball-Berry slope, C4 [-]
_B0_C3 = 0.01              # Ball-Berry intercept, C3 [mol m-2 s-1]
_B0_C4 = 0.04             # Ball-Berry intercept, C4 [mol m-2 s-1]
_TGC_DEFAULT_C = 25.0      # growth-temperature default for Vcmax acclimation [°C]
_HC_MIN_M = 0.1            # minimum canopy height [m]
_EMISS_VEG = 0.97          # vegetated-surface emissivity [-]
_EMISS_BARE = 0.96         # bare-soil emissivity [-]
_RZ0M_BARE = 0.01          # bare-surface z0m/hc ratio [-]
_ALB_VIS_BARE = 0.2        # bare fallback visible albedo [-]
_ALB_NIR_BARE = 0.3        # bare fallback NIR albedo [-]

# --- Glacier ice-surface albedo (snow/ice; CLM glacier landunit) ---
_GLACIER_ALB_VIS = 0.70
_GLACIER_ALB_NIR = 0.50

# --- Soil-texture fallback where HWSD has no soil (sandy default) ---
_FALLBACK_SAND_PCT = 92.0
_FALLBACK_CLAY_PCT = 3.0

# --- Nominal top-layer wetness for soil-colour albedo before state exists ---
_THETA_TOP_DEFAULT = 0.2

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
    tgc_C: float = _TGC_DEFAULT_C,
    glacier_alb_vis: float = _GLACIER_ALB_VIS,
    glacier_alb_nir: float = _GLACIER_ALB_NIR,
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
        hc=jnp.asarray(np.maximum(hc, _HC_MIN_M)),
        fC4=jnp.asarray(lut["fc4"][dom]),
        FNonVeg=jnp.asarray(1.0 - is_veg),       # bare-dominant columns -> non-veg
        CI=full(_CI_DEFAULT), kn=full(_KN_DEFAULT),
        Vcmax25_C3_leaf=jnp.asarray(lut["vc3"][dom]),
        Vcmax25_C4_leaf=jnp.asarray(lut["vc4"][dom]),
        m_C3=full(_M_C3), m_C4=full(_M_C4), b0_C3=full(_B0_C3), b0_C4=full(_B0_C4),
        alf=full(_ALF_DEFAULT), TgC=full(float(tgc_C)),
        ALB_VIS=jnp.asarray(alb_vis), ALB_NIR=jnp.asarray(alb_nir),
        emissivity=full(_EMISS_VEG),
        rz0m=jnp.asarray(np.where(is_veg > 0.0, lut["rz0m"][dom], _RZ0M_BARE)),
        rd=jnp.asarray(np.where(is_veg > 0.0, lut["rd"][dom], 0.0)),
    )


def build_soil_hydraulics(
    gsd,
    *,
    fallback_sand_pct: float = _FALLBACK_SAND_PCT,   # sand where HWSD has no soil
    fallback_clay_pct: float = _FALLBACK_CLAY_PCT,
    base: SoilHydraulicsConfig = SoilHydraulicsConfig(),
) -> SoilHydraulicsConfig:
    """Per-(col, layer) Clapp-Hornberger :class:`SoilHydraulicsConfig` (Cosby).

    Runs the Cosby (1984) pedotransfer per (col, layer) using the surfdata
    texture profile, which the loader already remapped to the model's
    :class:`SoilGrid` via :func:`_remap_soil_layers`.  Hydraulic params come
    back as ``(ncol, n_layer)`` arrays that align cell-for-cell with the
    Richards solver's soil state — ``slice_layer`` picks the right layer for
    single-layer call sites (``K_top`` / ``K_bot``).  Any (col, layer) where
    HWSD has no soil (NaN) falls back to ``fallback_*`` (a sandy default) in
    just that cell, so a column with partial coverage keeps its real layers.
    """
    sand = np.asarray(gsd.sand_frac) * 100.0                 # (ncol, n_layer) percent
    clay = np.asarray(gsd.clay_frac) * 100.0
    bad = ~np.isfinite(sand) | ~np.isfinite(clay)
    sand = np.where(bad, fallback_sand_pct, sand)
    clay = np.where(bad, fallback_clay_pct, clay)

    p = cosby_hydraulic_params(jnp.asarray(sand), jnp.asarray(clay))  # (ncol, n_layer)
    return base._replace(
        retention_curve="clapp_hornberger",
        theta_sat=jnp.asarray(p.theta_sat), psi_sat=jnp.asarray(p.psi_sat),
        b_ch=jnp.asarray(p.b_ch), K_sat=jnp.asarray(p.K_sat), theta_r=0.0,
    )


# ===========================================================================
# Scheme-agnostic adapters: GlobalSurfaceData -> per-scheme land parameters
# ===========================================================================
# The SLAB / multilayer-SimpleSEB schemes consume a ``LandSurfaceParams``
# produced through the land/dev *param-provider* architecture
# (``legoesm.land.param_providers``), NOT a bespoke adapter.  The surface-data
# path is a thin provider that COMPOSES the existing differentiable
# ``PFTParamProvider`` (CLM5 table weighted by surfdata PFT fractions; the table
# stays trainable) and overrides only ``albedo_veg`` with the surfdata-derived
# value (soil-colour background blended with the PFT veg albedo, glacier ice).
# Glacier ice albedo for surface-data columns flagged as glacier-dominant.
_GLACIER_ALBEDO_DEFAULT = 0.6


class SurfaceDataParamProvider(eqx.Module):
    """``LandSurfaceParams`` provider backed by regridded surface data.

    Wraps a :class:`~legoesm.land.param_providers.PFTParamProvider` (so the CLM5
    parameter table remains trainable and gradients flow exactly as for the
    prescribed-PFT path) and applies a surfdata ``albedo_veg`` override:
    the soil-colour background blended with the PFT veg albedo by canopy cover
    ``1-exp(-0.5*LAI)``, with glacier-dominant columns set to ice albedo.

    Observational arrays (``soil_bg``, ``f_veg``, ``is_glacier``) are
    ``stop_gradient``-ed in ``__call__`` — they are boundary data, not knobs —
    mirroring how ``PFTParamProvider`` treats ``pft_fractions``.  ``__call__``
    accepts an optional ``features`` arg (ignored) so it is drop-in compatible
    with the coupler's ``land_param_provider(features)`` call site.
    """
    pft_provider: PFTParamProvider
    soil_bg: jax.Array              # (ncol,) broadband soil-colour albedo
    f_veg: jax.Array               # (ncol,) canopy cover fraction 1-exp(-0.5*LAI)
    is_glacier: jax.Array          # (ncol,) 1.0 where glacier-dominant
    glacier_albedo: float = eqx.field(static=True)

    def __call__(self, features=None) -> LandSurfaceParams:
        lp = self.pft_provider()
        soil_bg = jax.lax.stop_gradient(self.soil_bg)
        f_veg = jax.lax.stop_gradient(self.f_veg)
        is_glacier = jax.lax.stop_gradient(self.is_glacier)
        alb = lp.albedo_veg * f_veg + soil_bg * (1.0 - f_veg)
        alb = jnp.where(is_glacier > 0.0, self.glacier_albedo, alb)
        return lp._replace(albedo_veg=alb)


def surface_data_param_provider(
    gsd,
    day_of_year: float,
    theta_top: jnp.ndarray,
    *,
    glacier_albedo: float = _GLACIER_ALBEDO_DEFAULT,
) -> SurfaceDataParamProvider:
    """Build a :class:`SurfaceDataParamProvider` from regridded surface data.

    PFT weights are the (year-mean) ``pft_frac`` (uncovered columns -> zero; the
    mask-reconciliation pass :func:`fill_land_param_gaps` handles them).  The
    canopy-cover LAI for the albedo blend is the PFT-weighted column LAI, so it is
    consistent with the PFT-weighted parameters.
    """
    fracs = np.nan_to_num(np.asarray(jnp.mean(gsd.pft_frac, axis=0)), nan=0.0)  # (ncol,npft)
    # Zero-cover columns (no PFT info: ocean/ice/desert gaps) -> bare soil (PFT 0),
    # matching the dominant-PFT fallback (argmax of all-zeros = bare_soil) so they
    # get the valid bare-soil table row (nonzero C_soil/W_max) instead of an
    # all-zero ``fracs @ table`` row that would divide-by-zero in the surface step.
    zero_cover = fracs.sum(axis=-1) < 1e-6
    fracs[zero_cover, :] = 0.0
    fracs[zero_cover, 0] = 1.0
    pft_provider = PFTParamProvider.from_defaults(jnp.asarray(fracs))

    lai_m = np.asarray(interp_monthly(gsd.lai_monthly, jnp.asarray(float(day_of_year))))
    lai_col = np.nan_to_num(np.sum(lai_m * fracs, axis=-1), nan=0.0)            # (ncol,)
    soil_bg = np.asarray(
        soil_albedo_broadband(jnp.asarray(np.asarray(gsd.soil_color)), jnp.asarray(theta_top)))
    f_veg = 1.0 - np.exp(-0.5 * lai_col)
    return SurfaceDataParamProvider(
        pft_provider=pft_provider,
        soil_bg=jnp.asarray(soil_bg),
        f_veg=jnp.asarray(f_veg),
        is_glacier=jnp.asarray(glacier_mask(gsd).astype(float)),
        glacier_albedo=float(glacier_albedo),
    )


def surface_data_to_land_params(gsd, surface_scheme, day_of_year, theta_top):
    """Dispatch to the right per-column land-params object for ``surface_scheme``.

    ``TwoLeafCanopyConfig`` -> :class:`CanopyLandParams` (built directly — land/dev
    has no canopy provider).  ``SimpleSEBConfig`` (slab or multilayer) ->
    :class:`LandSurfaceParams` materialized from :class:`SurfaceDataParamProvider`.
    For coupler use, prefer :func:`surface_data_param_provider` and pass the
    provider to ``make_coupler(land_param_provider=...)``.  (clm-ml is an external
    plugin with its own input contract; feed it ``gsd`` directly.)
    """
    from legoesm.land.canopy import CanopyConfig
    if isinstance(surface_scheme, CanopyConfig):
        return build_canopy_params(gsd, day_of_year, theta_top)
    return surface_data_param_provider(gsd, day_of_year, theta_top)()


def init_land_surface_data(surfdata_path, grid, land_config, day_of_year, *, theta_top=None):
    """Load the surfdata, regrid to ``grid``, and adapt to ``land_config``'s scheme.

    The single entry a driver calls at simulation start.  Returns
    ``(land_config, land_params, gsd)``: for a multilayer config the returned
    config also carries the per-(col, layer) Cosby soil hydraulics derived from
    the surfdata via :func:`build_soil_hydraulics` (params are ``(ncol, n_layer)``
    arrays that align with the model's soil state cell-for-cell).
    ``theta_top`` (top-layer wetness for the soil-colour albedo) defaults to a
    nominal 0.2 when no state exists yet.
    """
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data

    cfg_sd = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=surfdata_path)
    gsd = load_global_surface_data(cfg_sd, grid)
    ncol = int(np.asarray(gsd.soil_color).shape[0])
    if theta_top is None:
        theta_top = jnp.full(ncol, _THETA_TOP_DEFAULT)

    if isinstance(land_config, MultiLayerLandConfig):
        land_config = land_config._replace(
            hydraulics=build_soil_hydraulics(gsd, base=land_config.hydraulics))

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
        LAI=full(0.0), hc=full(_HC_MIN_M), fC4=full(0.0), FNonVeg=full(1.0),
        CI=full(_CI_DEFAULT), kn=full(_KN_DEFAULT),
        Vcmax25_C3_leaf=full(0.0), Vcmax25_C4_leaf=full(0.0),
        m_C3=full(_M_C3), m_C4=full(_M_C4), b0_C3=full(_B0_C3), b0_C4=full(_B0_C4),
        alf=full(_ALF_DEFAULT), TgC=full(_TGC_DEFAULT_C),
        ALB_VIS=full(_ALB_VIS_BARE), ALB_NIR=full(_ALB_NIR_BARE),
        emissivity=full(_EMISS_BARE), rz0m=full(_RZ0M_BARE), rd=full(0.0),
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
    covered = jnp.asarray(surfdata_covered(gsd))            # (ncol,)
    ncol = covered.shape[0]
    fb = (_bare_canopy_params(ncol) if isinstance(land_params, CanopyLandParams)
          else _bare_land_surface_params(ncol))

    def _fill(v, f):
        v = jnp.asarray(v)
        keep = covered.reshape(covered.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))

    return jax.tree.map(_fill, land_params, fb)


# ===========================================================================
# JAX-native per-step LAI / albedo updater (for lax.scan time loops)
# ===========================================================================
def _gap_fill_tree(land_params, fb, covered_jnp):
    """Mask-fill helper: same logic as :func:`fill_land_param_gaps` but with
    pre-computed (JAX-side) ``covered`` mask, so it runs inside a ``lax.scan``
    body without a host roundtrip."""
    def _fill(v, f):
        v = jnp.asarray(v)
        keep = covered_jnp.reshape(covered_jnp.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))
    return jax.tree.map(_fill, land_params, fb)


def make_step_land_params_updater(gsd, surface_scheme):
    """Build a JAX-pure ``(theta_top, doy) -> land_params`` closure for use
    inside a ``lax.scan`` time loop.

    Splits :func:`surface_data_to_land_params` + :func:`fill_land_param_gaps`
    into a static precompute (PFT lookups, soil_color, glacier / surfdata-
    covered masks, bare-fallback templates — host-side, runs once) and a
    JAX-pure per-step updater that ingests the only inputs that change with
    time: ``theta_top`` (top-layer wetness; varies with the soil state) and
    ``doy`` (day-of-year, advances each step).  Everything else is captured
    once.

    Inside the updater:
      - LAI / canopy-height are picked from the monthly climatology via
        :func:`interp_monthly` at the traced ``doy`` and the precomputed
        dominant-PFT (canopy scheme) or PFT-fraction weights (SEB scheme).
      - Soil-background albedo is recomputed from the precomputed soil colour
        and the traced ``theta_top``.
      - Glacier override (ice albedo + zero LAI) and surfdata-gap fill are
        applied with precomputed masks.

    Returns a function with the same output type as
    :func:`surface_data_to_land_params` for the given scheme.
    """
    from legoesm.land.canopy import CanopyConfig
    is_canopy = isinstance(surface_scheme, CanopyConfig)

    # Inputs that don't change across steps (cast once to JAX).
    lai_monthly = jnp.asarray(gsd.lai_monthly)
    htop_monthly = jnp.asarray(gsd.htop_monthly)
    soil_color = jnp.asarray(np.asarray(gsd.soil_color))
    glacier_col = jnp.asarray(glacier_mask(gsd).astype(np.float64))
    covered_jnp = jnp.asarray(surfdata_covered(gsd))

    if is_canopy:
        dom = dominant_pft_index(gsd)
        ncol = int(dom.shape[0])
        lut = _pft_lookup_arrays()
        dom_idx = jnp.asarray(dom.astype(np.int32))
        is_veg_dom = lut["is_veg"][dom]              # (ncol,) np
        is_veg_col = jnp.asarray(is_veg_dom.astype(np.float64))
        hc_default = jnp.asarray(lut["hc"][dom])
        fC4 = jnp.asarray(lut["fc4"][dom])
        Vcmax25_C3 = jnp.asarray(lut["vc3"][dom])
        Vcmax25_C4 = jnp.asarray(lut["vc4"][dom])
        rz0m = jnp.asarray(np.where(is_veg_dom > 0.0, lut["rz0m"][dom], _RZ0M_BARE))
        rd = jnp.asarray(np.where(is_veg_dom > 0.0, lut["rd"][dom], 0.0))
        bare_fb = _bare_canopy_params(ncol)
        full = lambda v: jnp.full(ncol, v)

        def _update_canopy(theta_top: jnp.ndarray, doy: jnp.ndarray):
            """Return ``(CanopyLandParams, lai_col)``: ``lai_col`` is the
            per-column dominant-PFT LAI used in the params, useful as a
            diagnostic in the scan body."""
            lai_m = interp_monthly(lai_monthly, doy)                # (ncol, npft)
            htop_m = interp_monthly(htop_monthly, doy)
            LAI = jnp.take_along_axis(lai_m, dom_idx[:, None], axis=1)[:, 0]
            hc_surf = jnp.take_along_axis(htop_m, dom_idx[:, None], axis=1)[:, 0]
            LAI = jnp.where(is_veg_col > 0.0,
                            jnp.where(jnp.isfinite(LAI), LAI, 0.0), 0.0)
            hc = jnp.where(jnp.isfinite(hc_surf) & (hc_surf > 0.0),
                           hc_surf, hc_default)
            hc = jnp.maximum(hc, _HC_MIN_M)
            av, an = soil_albedo(soil_color, theta_top)
            ice = glacier_col > 0.0
            is_veg = jnp.where(ice, 0.0, is_veg_col)
            LAI = jnp.where(ice, 0.0, LAI)
            av = jnp.where(ice, _GLACIER_ALB_VIS, av)
            an = jnp.where(ice, _GLACIER_ALB_NIR, an)
            lp = CanopyLandParams(
                LAI=LAI, hc=hc, fC4=fC4, FNonVeg=1.0 - is_veg,
                CI=full(_CI_DEFAULT), kn=full(_KN_DEFAULT),
                Vcmax25_C3_leaf=Vcmax25_C3, Vcmax25_C4_leaf=Vcmax25_C4,
                m_C3=full(_M_C3), m_C4=full(_M_C4),
                b0_C3=full(_B0_C3), b0_C4=full(_B0_C4),
                alf=full(_ALF_DEFAULT), TgC=full(_TGC_DEFAULT_C),
                ALB_VIS=av, ALB_NIR=an,
                emissivity=full(_EMISS_VEG), rz0m=rz0m, rd=rd,
            )
            lp_filled = _gap_fill_tree(lp, bare_fb, covered_jnp)
            return lp_filled, lp_filled.LAI

        return _update_canopy

    # ---- SEB / slab path: PFT-weighted LAI -> albedo_veg blend ----
    fracs = np.nan_to_num(np.asarray(jnp.mean(gsd.pft_frac, axis=0)), nan=0.0)
    zero_cover = fracs.sum(axis=-1) < 1e-6
    fracs[zero_cover, :] = 0.0
    fracs[zero_cover, 0] = 1.0
    pft_provider = PFTParamProvider.from_defaults(jnp.asarray(fracs))
    base_lp = pft_provider()                                 # static base LandSurfaceParams
    fracs_jnp = jnp.asarray(fracs)
    ncol = int(fracs.shape[0])
    bare_fb = _bare_land_surface_params(ncol)

    def _update_seb(theta_top: jnp.ndarray, doy: jnp.ndarray):
        """Return ``(LandSurfaceParams, lai_col)``: ``lai_col`` is the
        PFT-weighted column LAI driving the albedo blend, useful as a
        diagnostic in the scan body."""
        lai_m = interp_monthly(lai_monthly, doy)              # (ncol, npft)
        lai_col = jnp.sum(jnp.where(jnp.isfinite(lai_m), lai_m, 0.0) * fracs_jnp,
                          axis=-1)
        soil_bg = soil_albedo_broadband(soil_color, theta_top)
        f_veg = 1.0 - jnp.exp(-0.5 * lai_col)
        alb = base_lp.albedo_veg * f_veg + soil_bg * (1.0 - f_veg)
        alb = jnp.where(glacier_col > 0.0, _GLACIER_ALBEDO_DEFAULT, alb)
        lp = base_lp._replace(albedo_veg=alb)
        return _gap_fill_tree(lp, bare_fb, covered_jnp), lai_col

    return _update_seb

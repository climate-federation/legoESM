"""Host-side builders: ``GlobalSurfaceData`` -> per-column land-model inputs.

This is the consumer-side counterpart to the producer in
:mod:`legoesm.land.surface_data`.  Turns a regridded
:class:`~legoesm.land.global_surface_data.GlobalSurfaceData` (NetCDF read +
regridded to the model grid at simulation start) into the per-column arrays
the multilayer-land + two-leaf-canopy model consumes for a global run:

  - :func:`build_canopy_params` -> :class:`CanopyLandParams` (LAI, canopy height,
    soil-colour background albedo, PFT photosynthesis/aerodynamic params, C4
    flag), one value per column from the **dominant PFT**;
  - :func:`build_soil_hydraulics` -> a per-(col, layer) :class:`SoilHydraulicsConfig`
    (Cosby pedotransfer from the surfdata's depth-remapped sand/clay profile),
    with a per-cell fallback texture where HWSD has no soil (sand seas / ice);
  - :func:`surface_data_param_provider` -> a differentiable
    :class:`SurfaceDataParamProvider` wrapping the existing
    :class:`PFTParamProvider` for the SimpleSEB / slab path.

The simulation-start entry is :func:`init_land_surface_data`, which loads
the surfdata, regrids it to the model grid, attaches the texture-derived
hydraulics to a multilayer config, and returns the per-scheme land_params.

The canopy PFT tables (:data:`PFT_VCMAX25_C3` etc.) are keyed by biome type
(ENF/EBF/.../GRA/CRO) with three climate-zone columns ordered
``[boreal, temperate, tropical]``; CLM5's 17 PFTs carry the climate zone in the
name, so ``_CLM5_TO_BIOME`` in :mod:`._internals` maps each CLM5 PFT to
``(biome, zone_index, is_c4)``.  Host-side; not traced.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.land.surface_params import LandSurfaceParams
from legoesm.land.param_providers import PFTParamProvider
from legoesm.land.canopy.config import CanopyLandParams
from legoesm.land.soil_albedo import (
    soil_albedo_bounds, soil_albedo_broadband, wet_soil_albedo)
from legoesm.land.pedotransfer import cosby_hydraulic_params
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.global_surface_data import interp_monthly

from legoesm.land.boundary_data._internals import (
    CI_DEFAULT, KN_DEFAULT, ALF_DEFAULT,
    M_C3, M_C4, B0_C3, B0_C4,
    TGC_DEFAULT_C, HC_MIN_M,
    EMISS_VEG, RZ0M_BARE,
    GLACIER_ALB_VIS, GLACIER_ALB_NIR, GLACIER_ALBEDO_DEFAULT,
    FALLBACK_SAND_PCT, FALLBACK_CLAY_PCT,
    THETA_TOP_DEFAULT,
    pft_lookup_arrays,
    cover1d,
)


# ===========================================================================
# Coverage / mask inspectors
# ===========================================================================
def cover_fracs(gsd, year=None) -> jnp.ndarray:
    """Per-column PFT cover ``(ncol, npft)`` from the transient ``pft_frac``.

    ``year=None`` (default) returns the year-mean — the legacy static snapshot the
    surfdata builders used.  A concrete ``year`` returns the transient slice at that
    calendar year via :func:`legoesm.land.global_surface_data.interp_annual`
    (single-year surfdata -> the one slice), so a driver can request a dated cover
    (e.g. its run start year) instead of collapsing a multi-century series to a mean.
    The single cover-collapse used by every ``surface_data_*`` builder below.
    """
    if year is None:
        return jnp.mean(gsd.pft_frac, axis=0)               # (ncol, npft)
    from legoesm.land.global_surface_data import interp_annual
    return interp_annual(gsd.pft_frac, gsd.years, jnp.asarray(float(year)))


def dominant_pft_index(gsd, year=None) -> np.ndarray:
    """Dominant PFT index per column from the ``pft_frac`` (year-mean, or ``year``)."""
    return np.argmax(np.asarray(cover_fracs(gsd, year)), axis=-1)    # (ncol,)


def glacier_mask(gsd) -> np.ndarray:
    """Boolean ``(ncol,)``: columns where glacier is the dominant surface type.

    Greenland / Antarctica / mountain ice: ``f_glacier`` exceeds both the
    soil/veg (``f_land``) and lake fractions.  These are treated as ice surfaces
    (no vegetation, high albedo), not bare soil.
    """
    f_g = cover1d(gsd.f_glacier)
    return (f_g > cover1d(gsd.f_land)) & (f_g > cover1d(gsd.f_lake)) & (f_g > 0.0)


# ===========================================================================
# Canopy path: GlobalSurfaceData -> CanopyLandParams
# ===========================================================================
def build_canopy_params(
    gsd,
    day_of_year: float,
    theta_top: jnp.ndarray,
    *,
    tgc_C: float = TGC_DEFAULT_C,
    glacier_alb_vis: float = GLACIER_ALB_VIS,
    glacier_alb_nir: float = GLACIER_ALB_NIR,
    pft_root_params: dict | None = None,
    year=None,
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

    ``pft_root_params`` optionally supplies per-PFT (length-17, CLM5 order)
    root-zone water-uptake parameters — keys ``root_depth`` [m], ``theta_wp``,
    ``theta_fc`` [m3/m3] — applied at the DOMINANT PFT, consistent with every
    other field here, which is the dominant PFT's value.  (The coupled
    ``CLMSurfaceParamProvider`` instead PFT-fraction-WEIGHTS such tables;
    blending them would be incoherent in a column whose canopy, LAI, height and
    roughness are all a single PFT's.)  ``None`` leaves the per-column fields
    unset, so ``multilayer_land._get`` falls back to the scalar
    ``MultiLayerLandConfig`` values — bit-identical to the pre-existing
    behaviour.
    """
    dom = dominant_pft_index(gsd, year)                     # (ncol,)
    ncol = dom.shape[0]
    lut = pft_lookup_arrays()

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

    # Per-column dry/saturated soil-colour bounds; the land step re-evaluates the
    # band albedos from the live top-layer water (soil_albedo.rewet_soil_bands).
    # ``theta_top`` here only sets the build-time ALB_VIS/ALB_NIR.
    # Glacier columns: ice surface — no vegetation, dry == sat == the ice
    # albedo, so the wetness relation returns it exactly.
    ice = glacier_mask(gsd)
    is_veg = np.where(ice, 0.0, is_veg)
    LAI = np.where(ice, 0.0, LAI)
    bounds = [np.asarray(b) for b in
              soil_albedo_bounds(jnp.asarray(np.asarray(gsd.soil_color)))]
    dry_vis, dry_nir, sat_vis, sat_nir = (
        jnp.asarray(np.where(ice, g, b)) for g, b in zip(
            (glacier_alb_vis, glacier_alb_nir, glacier_alb_vis, glacier_alb_nir),
            bounds))
    _theta = jnp.asarray(theta_top)
    alb_vis = wet_soil_albedo(dry_vis, sat_vis, _theta)
    alb_nir = wet_soil_albedo(dry_nir, sat_nir, _theta)

    # Optional per-column root-zone params (dominant PFT).  Absent => the scalar
    # MultiLayerLandConfig values via multilayer_land._get.
    _root_kw = {}
    if pft_root_params is not None:
        _root_kw = {k: jnp.asarray(np.asarray(v, dtype=np.float64)[dom])
                    for k, v in pft_root_params.items()}

    full = lambda v: jnp.full(ncol, v)
    return CanopyLandParams(
        LAI=jnp.asarray(LAI),
        hc=jnp.asarray(np.maximum(hc, HC_MIN_M)),
        fC4=jnp.asarray(lut["fc4"][dom]),
        FNonVeg=jnp.asarray(1.0 - is_veg),       # bare-dominant columns -> non-veg
        CI=full(CI_DEFAULT), kn=full(KN_DEFAULT),
        Vcmax25_C3_leaf=jnp.asarray(lut["vc3"][dom]),
        Vcmax25_C4_leaf=jnp.asarray(lut["vc4"][dom]),
        m_C3=full(M_C3), m_C4=full(M_C4), b0_C3=full(B0_C3), b0_C4=full(B0_C4),
        alf=full(ALF_DEFAULT), TgC=full(float(tgc_C)),
        ALB_VIS=jnp.asarray(alb_vis), ALB_NIR=jnp.asarray(alb_nir),
        emissivity=full(EMISS_VEG),
        rz0m=jnp.asarray(np.where(is_veg > 0.0, lut["rz0m"][dom], RZ0M_BARE)),
        rd=jnp.asarray(np.where(is_veg > 0.0, lut["rd"][dom], 0.0)),
        ALB_VIS_DRY=dry_vis, ALB_VIS_SAT=sat_vis,
        ALB_NIR_DRY=dry_nir, ALB_NIR_SAT=sat_nir,
        **_root_kw,
    )


# ===========================================================================
# Soil hydraulics: per-(col, layer) Clapp-Hornberger via Cosby pedotransfer
# ===========================================================================
def build_soil_hydraulics(
    gsd,
    *,
    fallback_sand_pct: float = FALLBACK_SAND_PCT,   # sand where HWSD has no soil
    fallback_clay_pct: float = FALLBACK_CLAY_PCT,
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
# SEB / slab path: GlobalSurfaceData -> SurfaceDataParamProvider -> LandSurfaceParams
# ===========================================================================
# The SLAB / multilayer-SimpleSEB schemes consume a ``LandSurfaceParams``
# produced through the land/dev *param-provider* architecture
# (``legoesm.land.param_providers``), NOT a bespoke adapter.  The surface-data
# path is a thin provider that COMPOSES the existing differentiable
# ``PFTParamProvider`` (CLM5 table weighted by surfdata PFT fractions; the table
# stays trainable) and overrides only ``albedo_veg`` with the surfdata-derived
# value (soil-colour background blended with the PFT veg albedo, glacier ice).


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
    fc4: jax.Array                 # (ncol,) PFT-weighted C4 area fraction [0,1]
    glacier_albedo: float = eqx.field(static=True)

    def __call__(self, features=None) -> LandSurfaceParams:
        lp = self.pft_provider()
        soil_bg = jax.lax.stop_gradient(self.soil_bg)
        f_veg = jax.lax.stop_gradient(self.f_veg)
        is_glacier = jax.lax.stop_gradient(self.is_glacier)
        # C4 fraction is boundary data (PFT flags), not a knob -> stop_gradient,
        # mirroring soil_bg / f_veg / pft_fractions.
        fc4 = jax.lax.stop_gradient(self.fc4)
        alb = lp.albedo_veg * f_veg + soil_bg * (1.0 - f_veg)
        alb = jnp.where(is_glacier > 0.0, self.glacier_albedo, alb)
        return lp._replace(albedo_veg=alb, fC4=fc4)


def surface_data_param_provider(
    gsd,
    day_of_year: float,
    theta_top: jnp.ndarray,
    *,
    glacier_albedo: float = GLACIER_ALBEDO_DEFAULT,
    year=None,
) -> SurfaceDataParamProvider:
    """Build a :class:`SurfaceDataParamProvider` from regridded surface data.

    PFT weights are the ``pft_frac`` at ``year`` (or the year-mean when ``year`` is
    None; uncovered columns -> zero, handled by the mask-reconciliation pass
    :func:`fill_land_param_gaps`).  The canopy-cover LAI for the albedo blend is the
    PFT-weighted column LAI, so it is consistent with the PFT-weighted parameters.
    """
    fracs = np.nan_to_num(np.asarray(cover_fracs(gsd, year)), nan=0.0)          # (ncol,npft)
    # Zero-cover columns (no PFT info: ocean/ice/desert gaps) -> bare soil (PFT 0),
    # matching the dominant-PFT fallback (argmax of all-zeros = bare_soil) so they
    # get the valid bare-soil table row (nonzero C_soil/W_max) instead of an
    # all-zero ``fracs @ table`` row that would divide-by-zero in the surface step.
    zero_cover = fracs.sum(axis=-1) < 1e-6
    fracs[zero_cover, :] = 0.0
    fracs[zero_cover, 0] = 1.0
    pft_provider = PFTParamProvider.from_defaults(jnp.asarray(fracs))

    # Sanitise BEFORE the PFT-weighted sum: a NaN in a zero-weight PFT slot would
    # otherwise poison the whole column via ``NaN * 0 == NaN`` (same guard as
    # ``prescribed_canopy_structure`` and ``step_updater._update_seb``).
    lai_m = np.nan_to_num(
        np.asarray(interp_monthly(gsd.lai_monthly, jnp.asarray(float(day_of_year)))), nan=0.0)
    lai_col = np.nan_to_num(np.sum(lai_m * fracs, axis=-1), nan=0.0)            # (ncol,)
    soil_bg = np.asarray(
        soil_albedo_broadband(jnp.asarray(np.asarray(gsd.soil_color)), jnp.asarray(theta_top)))
    f_veg = 1.0 - np.exp(-0.5 * lai_col)
    # Big-leaf C4 flag from the DOMINANT PFT (0/1), matching build_canopy_params'
    # ``lut["fc4"][dom]``. A big leaf is a single photosynthetic pathway, so the
    # column runs pure C3 or pure C4 with its (PFT-weighted) Vcmax — not a
    # blend of both branches sharing one capacity. Continuous sub-grid C3/C4
    # mixing with separate C3/C4 capacities is the two-leaf canopy's role.
    fc4_col = np.asarray(pft_lookup_arrays()["fc4"])[dominant_pft_index(gsd)]   # (ncol,) 0/1
    return SurfaceDataParamProvider(
        pft_provider=pft_provider,
        soil_bg=jnp.asarray(soil_bg),
        f_veg=jnp.asarray(f_veg),
        is_glacier=jnp.asarray(glacier_mask(gsd).astype(float)),
        fc4=jnp.asarray(fc4_col),
        glacier_albedo=float(glacier_albedo),
    )


# ===========================================================================
# Scheme-agnostic dispatcher + simulation-start entry
# ===========================================================================
def prescribed_canopy_structure(gsd, day_of_year, year=None):
    """PFT-weighted monthly canopy structure ``(LAI, SAI, htop)`` — each
    ``(ncol,)`` — for the CLM-ML canopy's PRESCRIBED-LAI mode.

    Reuses the same PFT weights as the albedo blend in
    :func:`surface_data_param_provider` (``pft_frac`` at ``year`` or the year-mean,
    zero-cover columns collapsed to the bare-soil PFT row) so the prescribed LAI
    here is identical to the ``lai_col`` that drives the surfdata albedo.

    - ``LAI``/``SAI`` are area-additive (total leaf/stem area per unit ground =
      ``sum_pft frac * pft_value``), so the PFT-weighted sum is the physical
      column aggregate.
    - ``htop`` is a height (intensive); the PFT-fraction-weighted mean is
      returned as the effective single-canopy top height for the column.

    ``hbot`` is intentionally NOT returned: the CLM-ML interface derives the
    bottom-of-canopy height as ``CLMMLCanopyConfig.hbot_frac * htop`` (Bonan et
    al. 2021 GMD default), so it already tracks the prescribed ``htop`` and a
    separate prescribed ``hbot`` would be populated-but-unread.
    """
    fracs = np.nan_to_num(np.asarray(cover_fracs(gsd, year)), nan=0.0)          # (ncol,npft)
    zero_cover = fracs.sum(axis=-1) < 1e-6
    fracs[zero_cover, :] = 0.0
    fracs[zero_cover, 0] = 1.0   # uncovered -> bare-soil PFT row (LAI/SAI/height ~ 0)
    doy = jnp.asarray(float(day_of_year))

    def _wcol(monthly):
        # Sanitise the monthly field BEFORE the PFT-weighted sum: a NaN in a
        # zero-weight PFT slot would otherwise poison the whole column via
        # ``NaN * 0 == NaN`` (matches the ``jnp.where(isfinite, ., 0)`` guard in
        # ``step_updater._update_seb``).
        m = np.nan_to_num(np.asarray(interp_monthly(monthly, doy)), nan=0.0)  # (ncol,npft)
        return jnp.asarray(np.sum(m * fracs, axis=-1))                        # (ncol,)

    return (_wcol(gsd.lai_monthly), _wcol(gsd.sai_monthly),
            _wcol(gsd.htop_monthly))


def surface_data_to_land_params(gsd, surface_scheme, day_of_year, theta_top, *,
                                year=None, glacier_alb=None,
                                pft_root_params=None):
    """Dispatch to the right per-column land-params object for ``surface_scheme``.

    ``TwoLeafCanopyConfig`` -> :class:`CanopyLandParams` (built directly — land/dev
    has no canopy provider).  ``CLMMLCanopyConfig`` -> :class:`LandSurfaceParams`
    from :class:`SurfaceDataParamProvider` PLUS the prescribed PFT-weighted monthly
    canopy structure (``LAI``/``SAI``/``htop`` from
    :func:`prescribed_canopy_structure`) so the CLM-ML scheme runs on real
    vegetation structure instead of its scalar fallbacks (``hbot`` stays derived
    as ``hbot_frac * htop`` inside the CLM-ML interface).  ``SimpleSEBConfig``
    (slab or multilayer) -> :class:`LandSurfaceParams` (albedo/PFT params only;
    it does not read the canopy-structure fields).

    ``glacier_alb`` is an optional ``(vis, nir)`` ice-surface albedo pair that
    overrides the uncalibrated :data:`GLACIER_ALB_VIS`/:data:`GLACIER_ALB_NIR`
    default (values come from the run config, e.g. the LMIP
    ``physics.glacier_albedo_vis``/``_nir`` keys).  ``None`` keeps the default
    (the call is then bit-identical to the pre-override behaviour).
    """
    from legoesm.land.canopy import CanopyConfig
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    if isinstance(surface_scheme, CanopyConfig):
        _gk = {} if glacier_alb is None else {
            "glacier_alb_vis": float(glacier_alb[0]),
            "glacier_alb_nir": float(glacier_alb[1])}
        return build_canopy_params(gsd, day_of_year, theta_top, year=year,
                                   pft_root_params=pft_root_params, **_gk)
    # SEB / CLM-ML path: the provider takes a single BROADBAND ice albedo, which
    # is the 0.5/0.5 vis-NIR integral the canopy pair collapses to (see
    # GLACIER_ALBEDO_DEFAULT == 0.5*(GLACIER_ALB_VIS + GLACIER_ALB_NIR)).
    _bk = {} if glacier_alb is None else {
        "glacier_albedo": 0.5 * (float(glacier_alb[0]) + float(glacier_alb[1]))}
    lp = surface_data_param_provider(gsd, day_of_year, theta_top, year=year, **_bk)()
    if isinstance(surface_scheme, CLMMLCanopyConfig):
        lai, sai, htop = prescribed_canopy_structure(gsd, day_of_year, year)
        lp = lp._replace(LAI=lai, SAI=sai, htop=htop)
    return lp


def init_land_surface_data(surfdata_path, grid, land_config, day_of_year, *,
                           theta_top=None, year=None, glacier_alb=None,
                           pft_root_params=None):
    """Load the surfdata, regrid to ``grid``, and adapt to ``land_config``'s scheme.

    The single entry a driver calls at simulation start.  Returns
    ``(land_config, land_params, gsd)``: for a multilayer config the returned
    config also carries the per-(col, layer) Cosby soil hydraulics derived from
    the surfdata via :func:`build_soil_hydraulics` (params are ``(ncol, n_layer)``
    arrays that align with the model's soil state cell-for-cell).
    ``theta_top`` (top-layer wetness for the soil-colour albedo) defaults to a
    nominal 0.2 when no state exists yet.  ``year`` selects the transient cover
    slice for the returned ``land_params`` (None -> the legacy year-mean); the
    returned ``gsd`` still carries the full ``pft_frac``/``years`` series so a
    driver can rebuild params at other years.
    ``glacier_alb`` is an optional ``(vis, nir)`` ice-albedo override forwarded
    to :func:`surface_data_to_land_params`; ``None`` = the uncalibrated default.

    NOTE: a driver that also builds a PER-STEP params updater
    (:func:`~legoesm.land.boundary_data.make_step_land_params_updater`) must pass
    the SAME ``glacier_alb`` there — the updater rebuilds ``CanopyLandParams``
    every step, so an override applied only here would be discarded after step 0.
    """
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data

    cfg_sd = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=surfdata_path)
    # Remap the surfdata soil profile onto the CONFIG's soil grid, not the loader
    # default.  build_soil_hydraulics below derives per-(col, layer) hydraulics
    # from gsd.sand_frac/clay_frac, so the remap target MUST match
    # ``land_config.soil_grid`` — otherwise a non-default SoilGridConfig silently
    # produces (ncol, 8) hydraulics against an (ncol, n_layers) soil state.
    _soil_grid = None
    if isinstance(land_config, MultiLayerLandConfig):
        from legoesm.land.soil_grid import make_soil_grid
        _soil_grid = make_soil_grid(land_config.soil_grid)
    gsd = load_global_surface_data(cfg_sd, grid, soil_grid=_soil_grid)
    ncol = int(np.asarray(gsd.soil_color).shape[0])
    if theta_top is None:
        theta_top = jnp.full(ncol, THETA_TOP_DEFAULT)

    if isinstance(land_config, MultiLayerLandConfig):
        land_config = land_config._replace(
            hydraulics=build_soil_hydraulics(gsd, base=land_config.hydraulics))

    land_params = surface_data_to_land_params(
        gsd, land_config.surface_scheme, day_of_year, theta_top, year=year,
        glacier_alb=glacier_alb, pft_root_params=pft_root_params)
    return land_config, land_params, gsd

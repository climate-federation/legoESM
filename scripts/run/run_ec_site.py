#!/usr/bin/env python
"""Standalone offline two-leaf-canopy run at eddy-covariance flux-tower sites.

Drives legoESM's two-leaf canopy from a DifferBESS v2 per-site driver NetCDF
(read via ``legoesm.land.boundary_data.ec_site``) and compares the modelled
GPP / LE / H against the tower observations bundled in the driver.

Modes
-----
``diagnostic`` (this stage): prescribe the soil skin temperature + root-zone
moisture stress from the driver and evaluate the canopy energy/carbon/water
closure independently at each timestep (``jax.vmap`` over time, chunked to bound
memory) — the apples-to-apples analogue of DifferBESS's vmapped forward pass.
(``prognostic`` — running the full multilayer soil — is added in a later stage.)

Validation is against the observed fluxes only, restricted to the adapter's
``valid`` timesteps (every driving input genuinely observed) AND finite obs.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/run_ec_site.py \
        --driver-nc <DifferBESS>/data/sitelevel/nc/US-Ton_driver_v2.nc \
                    <DifferBESS>/data/sitelevel/nc/US-MMS_driver_v2.nc \
        --mode diagnostic --out diagnostics/ec_site
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data.ec_site import read_ec_site_driver, ECSiteDriver
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.richards import RichardsConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import compute_two_leaf_canopy_fluxes
from legoesm.land.surface_scheme.patch_mosaic import (
    ClmmlMosaicConfig,
    PatchMosaicConfig,
    area_weight_series,
    compute_mosaic_canopy_fluxes,
    savanna_clmml_two_patch,
    savanna_two_patch,
)
from legoesm.land.canopy.config import (
    CLMMLCanopyConfig,
    VALID_CLM_ML_STOMATAL_MODELS,
    VALID_CLM_ML_TURBULENCE_SCHEMES,
)

# Stem area index supplied to the CLM-ML multilayer canopy [m2/m2].  The EC
# driver carries LAI only (the two-leaf canopy has no stem-area term), so the
# multilayer canopy — whose plant-area profile is LAI + SAI — needs a value;
# 0.5 is the CLM temperate-forest stem/dead-leaf area typical of the offline
# sites.  It is a real +8% plant area relative to the two-leaf arm; see the
# canopy-scheme comparison notes in docs/ec_site_offline_run.md.
_CLMML_SAI = 0.5

# Soil presets for the offline land config.  ``default`` = legoESM loam defaults
# (6.4 m free-draining column).  The site presets set texture-appropriate
# hydraulics + a 2 m column; ``siltloam``/``sandyloam`` match US-MMS soil
# (Clapp & Hornberger / Cosby texture classes).  A zero-flux bottom (no deep
# drainage) keeps the column from bleeding to wilting at a site with a shallow
# restrictive layer / water table.
_SOIL_HYDRAULICS = {
    "siltloam": dict(theta_sat=0.485, theta_r=0.05, K_sat=7.2e-7, b_ch=5.30,
                     psi_sat=-0.786),
    "sandyloam": dict(theta_sat=0.41, theta_r=0.041, K_sat=7.2e-6, b_ch=4.74,
                      psi_sat=-0.218),
}
_SOIL_FC_WP = {"siltloam": (0.33, 0.13), "sandyloam": (0.21, 0.10)}


# ---------------------------------------------------------------------------
# Per-site tower + hydrology metadata for the offline EC-site validation.
# Single source of truth for the site-level physical settings that were
# previously passed by hand / environment overrides, so a site run is
# reproducible from this one table.
#   z_ref [m]      : wind and flux measurement height = FLUXNET BADM
#                    Reference_height_v (from
#                    flux_tower_site_attributes/<SITE>_*_ReferenceHeight.nc).
#                    Anchors the Monin-Obukhov surface-layer profile / u* at the
#                    true tower height rather than a generic 10 m default.
#   root_depth [m] : root-density e-folding depth (root_frac = exp(-z/root_depth));
#                    deepened for phreatophytic vegetation that taps deep soil /
#                    weathered-bedrock water through the dry season (US-Ton blue
#                    oaks, AU-How savanna).  1.0 m elsewhere.
#   soil_depth_m   : soil-column depth [m]; deepened so the deep roots reach the
#                    retained deep-column water (0 => model default ~6.4 m column).
# Aerodynamic roughness (rz0m/rd) is deliberately NOT here: it is assigned
# per-PFT by the driver reader (canopy.config.PFT_AERO_PARAMS), so the tower
# height is the only site-specific aerodynamic input.  The Ball-Berry intercept
# stress (stress_b0) is set globally (cuticular conductance persists on live
# leaves; a dormant/deciduous canopy self-limits via LAI -> 0), not per site.
# root_depth reflects the rooting depth the site's soil profile actually permits:
# a deep glacial-till loam (US-MMS) roots deeper than a shallow montane podzol
# (DE-Obe) or an annual grassland (US-Var), while phreatophytes tapping deep soil /
# weathered-bedrock water through the dry season (US-Ton oaks, AU-How savanna) root
# far deeper.  These are the per-site values validated against the FLUXNET fluxes.
EC_SITE_PHYSICS: dict[str, dict[str, float]] = {
    "US-MMS": {"z_ref": 46.0, "root_depth": 2.0},                     # deep loam, DBF
    "DE-Hai": {"z_ref": 43.5, "root_depth": 1.5},                     # tall DBF (hc~33 m)
    "DE-Obe": {"z_ref": 30.0, "root_depth": 1.0},                     # shallow montane ENF
    "US-Ton": {"z_ref": 23.5, "root_depth": 5.0, "soil_depth_m": 10.0},  # phreatophyte oak
    "US-Var": {"z_ref": 2.0,  "root_depth": 1.0},                     # shallow annual grass
    "AU-How": {"z_ref": 23.0, "root_depth": 5.0, "soil_depth_m": 10.0},  # phreatophyte savanna
}
# Generic fallbacks for a site not in the table (matches the model/CLI defaults).
_EC_SITE_DEFAULTS = {"z_ref": 10.0, "root_depth": 1.0, "soil_depth_m": 0.0}


# DifferBESS IGBP (0-indexed) + Koppen climate -> CLM4.5 multilayer-canopy PFT
# index (1-based, MLpftconMod: 1 NET-temperate, 2 NET-boreal, 3 NDT-boreal,
# 4 BET-tropical, 5 BET-temperate, 6 BDT-tropical, 7 BDT-temperate, 8 BDT-boreal,
# 9 BES, 10 BDS-temperate, 11 BDS-boreal, 13 C3-grass, 14 C4-grass, 15 crop).
# Two-leaf is site-aware via the driver CanopyLandParams; CLM-ML takes a single
# PFT, so it MUST be derived from the site's IGBP+climate (not a fixed default 7),
# or CLM-ML runs every site as broadleaf-deciduous and loses skill.  IGBP->PFT-name
# category matches the reader's _IGBP_TO_PFT (WSA/SAV->savanna tree; CSH/OSH->shrub).
_CLM_PFT_BY_CLIMATE: dict[int, dict[str, int]] = {
    0: {"tropical": 1,  "temperate": 1,  "boreal": 2},   # ENF -> NET
    1: {"tropical": 4,  "temperate": 5,  "boreal": 5},   # EBF -> BET
    2: {"tropical": 3,  "temperate": 3,  "boreal": 3},   # DNF -> NDT boreal
    3: {"tropical": 6,  "temperate": 7,  "boreal": 8},   # DBF -> BDT
    4: {"tropical": 6,  "temperate": 7,  "boreal": 8},   # MF  -> BDT (approx)
    5: {"tropical": 9,  "temperate": 10, "boreal": 11},  # CSH -> BES(trop)/BDS
    6: {"tropical": 9,  "temperate": 10, "boreal": 11},  # OSH -> BES(trop)/BDS
    7: {"tropical": 6,  "temperate": 7,  "boreal": 8},   # WSA -> BDT (savanna tree)
    8: {"tropical": 6,  "temperate": 7,  "boreal": 8},   # SAV -> BDT
    9: {"tropical": 13, "temperate": 13, "boreal": 13},  # GRA -> C3 grass
    10: {"tropical": 15, "temperate": 15, "boreal": 15}, # CRO -> crop
    12: {"tropical": 15, "temperate": 15, "boreal": 15}, # CNM/CRO -> crop
}
_CLM_PFT_DEFAULT = 7  # broadleaf-deciduous-temperate; fallback for an unmapped IGBP


def clm_pft_for_site(igbp: int, climate: str) -> int:
    """CLM4.5 multilayer-canopy PFT index for a site's ``(IGBP, climate)``.

    Falls back to :data:`_CLM_PFT_DEFAULT` (BDT-temperate) for an IGBP absent from
    the table, and to the temperate column for an unrecognised climate string.
    """
    by_clim = _CLM_PFT_BY_CLIMATE.get(int(igbp))
    if by_clim is None:
        return _CLM_PFT_DEFAULT
    return by_clim.get(climate, by_clim.get("temperate", _CLM_PFT_DEFAULT))


def _site_mean_vcmax25(d: ECSiteDriver) -> float | None:
    """Valid-mean of the driver's per-site C3 Vcmax25 [umol/m2/s], or None.

    This is the SAME per-site photosynthetic capacity two-leaf consumes
    (``CanopyLandParams.Vcmax25_C3_leaf``); injected into CLM-ML's
    ``vcmax25_override`` so the multilayer canopy uses the site value instead of
    the generic MLpftcon PFT-table constant.  A scalar (site mean) — the seasonal
    cycle two-leaf also uses is a documented follow-up (needs per-step injection).
    """
    v = np.asarray(d.canopy_params.Vcmax25_C3_leaf, dtype=float).ravel()
    v = v[np.isfinite(v) & (v > 0.0)]
    return float(v.mean()) if v.size else None


def ec_site_physics(site: str) -> dict[str, float]:
    """Tower height + root/column depth for ``site`` (BADM-anchored; see table).

    Returns a dict with keys ``z_ref``/``root_depth``/``soil_depth_m``, filling
    the generic defaults (10 m reference height, 1 m rooting, model-default
    column) for any key a site does not override — and for a site absent from
    :data:`EC_SITE_PHYSICS` entirely.
    """
    return {**_EC_SITE_DEFAULTS, **EC_SITE_PHYSICS.get(site, {})}


def _build_land_config(canopy_config: TwoLeafCanopyConfig, soil: str,
                       bottom_bc: str, depth_m: float,
                       k_sat_decay_m: float = 0.0,
                       soil_evap_resistance_exp: float = 2.0,
                       root_depth: float = 1.0,
                       z_ref: float = 10.0,
                       texture: tuple[float, float] | None = None,
                       interception: bool = False,
                       plant_wilting_point: float | None = None
                       ) -> MultiLayerLandConfig:
    """Assemble the multilayer land config for the offline EC-site run.

    ``k_sat_decay_m`` (>0) enables the Niu-2005 depth-decaying K_sat(z) retention;
    ``soil_evap_resistance_exp`` sets the S_top**exp bare-soil evaporation throttle
    on the canopy path (0 => Kelvin-h_r only, the pre-fix behaviour).
    ``texture`` = (pct_sand, pct_clay): build per-site van-Genuchten hydraulics
    (Carsel-Parrish, via :mod:`legoesm.land.soil_texture`) + the matching
    theta_fc/theta_wp, overriding the ``soil`` preset — the physically-correct
    per-site soil water-holding instead of loam-for-all.
    """
    kw = dict(surface_scheme=canopy_config,
              soil_evap_resistance_exp=soil_evap_resistance_exp,
              root_depth=root_depth, z_ref=z_ref)
    if interception:
        from legoesm.land.canopy.interception import InterceptionConfig
        kw["interception"] = InterceptionConfig()
    if plant_wilting_point is not None:
        # Separate PLANT wilting point (transpiration extraction) from the soil
        # wilting point — deep-rooted vegetation extracts below the soil cutoff.
        kw["theta_wp_plant"] = float(plant_wilting_point)
    if depth_m > 0:
        kw["soil_grid"] = SoilGridConfig(total_depth=depth_m)
    if bottom_bc != "free_drainage":
        kw["richards"] = RichardsConfig(bottom_bc=bottom_bc)
    if texture is not None:
        from legoesm.land import soil_texture as _st
        sand, clay = float(texture[0]), float(texture[1])
        vg = _st.vg_params_from_index(int(_st.usda_texture_index(sand, clay)))
        wp, fc = _st.wilting_field_capacity(vg)
        kw["hydraulics"] = SoilHydraulicsConfig(
            retention_curve="van_genuchten",
            theta_r=float(vg["theta_r"]), theta_sat=float(vg["theta_sat"]),
            alpha_vg=float(vg["alpha_vg"]), n_vg=float(vg["n_vg"]),
            K_sat=float(vg["K_sat"]), k_sat_decay_m=k_sat_decay_m)
        kw["theta_fc"], kw["theta_wp"] = float(fc), float(wp)
    elif soil != "default" and soil != "auto":
        h = dict(_SOIL_HYDRAULICS[soil])
        h["k_sat_decay_m"] = k_sat_decay_m
        kw["hydraulics"] = SoilHydraulicsConfig(**h)
        fc, wp = _SOIL_FC_WP[soil]
        kw["theta_fc"], kw["theta_wp"] = fc, wp
    elif k_sat_decay_m > 0.0:
        kw["hydraulics"] = SoilHydraulicsConfig(k_sat_decay_m=k_sat_decay_m)
    return MultiLayerLandConfig(**kw)


_DEFAULT_TEXTURE_CSV = "scripts/cluster/ec_site/ec_site_soil_texture.csv"


def _texture_lookup(site: str, csv_path: str) -> tuple[float, float] | None:
    """(pct_sand, pct_clay) for ``site`` from the texture table, or None if absent
    (caller then falls back to the loam default)."""
    import csv as _csv
    if not os.path.isfile(csv_path):
        return None
    key = (site.replace("_driver_v2_gapfree.nc", "")
               .replace("_driver_v2.nc", ""))
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            if row["site"] == key:
                return float(row["sand_pct"]), float(row["clay_pct"])
    return None


# gC per umol CO2 (carbon molar mass conversion); matches the canopy's own
# ``GPP = (Agross_Sun + Agross_Sh) * 12.0e-6`` (GROSS) so model<->obs GPP
# units round-trip.
_GC_PER_UMOL_CO2 = 12.0e-6
_DEFAULT_CHUNK = 8760            # timesteps per vmap chunk (bounds memory)

# gC per umol CO2 (carbon molar mass conversion); matches the canopy's own
# ``GPP = (Agross_Sun + Agross_Sh) * 12.0e-6`` (GROSS) so model<->obs GPP
# units round-trip.
_GC_PER_UMOL_CO2 = 12.0e-6
_DEFAULT_CHUNK = 8760            # timesteps per vmap chunk (bounds memory)
# Production wind-speed floor [m/s].  The coupler / land step ALWAYS drive the
# canopy with sqrt(u^2 + v^2 + U_min^2) (multilayer_land.py; coupler U_min default
# = 1.0), so the canopy never sees raw calm wind.  We mirror that here for a
# faithful offline run rather than passing the raw tower wind.
_U_MIN = 1.0


def _diagnostic_fluxes(d: ECSiteDriver, canopy_config: TwoLeafCanopyConfig,
                       land_config: MultiLayerLandConfig, chunk: int,
                       mosaic: "PatchMosaicConfig | None" = None):
    """vmap the canopy over time with the soil state PRESCRIBED from the driver.

    Returns (gpp_gC, le_wm2, h_wm2, t_surface) each shape (n_time,).  When
    ``mosaic`` is given, each timestep runs the N-patch mosaic (tree + grass +
    ... tiles, area-weighted) instead of the single blended canopy.
    """
    def _step(T_soil_t, forcing_t, params_t, w_frac_t):
        # Mirror production: floor wind as sqrt(u^2 + v^2 + U_min^2) (the canopy
        # never sees raw calm wind in the coupled model).
        wind_t = jnp.sqrt(forcing_t.u_lowest ** 2 + forcing_t.v_lowest ** 2
                          + _U_MIN ** 2)
        kw = dict(
            T_soil_top=T_soil_t, forcing=forcing_t,
            canopy_config=canopy_config, land_config=land_config,
            canopy_params=params_t, w_frac_rz=w_frac_t, wind_speed=wind_t,
            wind_dir_x=jnp.ones_like(wind_t), wind_dir_y=jnp.zeros_like(wind_t),
            # Diagnostic: hold the soil skin temperature fixed at the prescribed
            # value (no soil-thermal evolution) so this isolates the canopy.
            soil_thermal_fn=lambda G, dt_: T_soil_t,
            dt=d.dt_s,
            LAI_override=params_t.LAI, TgC_override=params_t.TgC,
        )
        out = (compute_two_leaf_canopy_fluxes(**kw) if mosaic is None
               else compute_mosaic_canopy_fluxes(mosaic=mosaic, **kw))
        return out.gpp, out.lhflx, out.shflx, out.T_surface

    vstep = jax.jit(jax.vmap(_step))
    n = int(d.forcing.T_lowest.shape[0])
    gpp, le, h, ts = [], [], [], []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        f = jax.tree_util.tree_map(lambda a: a[sl], d.forcing)
        p = jax.tree_util.tree_map(lambda a: a[sl], d.canopy_params)
        g, l, hh, t = vstep(d.T_soil_top[sl], f, p, d.w_frac_rz[sl])
        gpp.append(np.asarray(g).ravel()); le.append(np.asarray(l).ravel())
        h.append(np.asarray(hh).ravel()); ts.append(np.asarray(t).ravel())
    return (np.concatenate(gpp), np.concatenate(le),
            np.concatenate(h), np.concatenate(ts))


def _clmml_mosaic_prognostic(d: ECSiteDriver, mosaic: ClmmlMosaicConfig, *,
                             soil: str, bottom_bc: str, soil_depth_m: float,
                             k_sat_decay_m: float, soil_evap_resistance_exp: float,
                             z_ref: float, texture, interception: bool,
                             plant_wilting_point: float | None,
                             clmml_turbulence: str, clmml_stomatal: str,
                             u_min: float, nudge_tau_days: float, clmml_sai: float):
    """Run the CLM-ML mosaic OUTER-loop: one prognostic column per tile, then
    area-weight the flux series.

    Each tile is an independent single-PFT CLM-ML column with its OWN rooting
    depth (hence its own prognostic water stress) — the deep-tree vs shallow-grass
    phenology the single canopy cannot represent.  The tiles do NOT share soil
    water (independent columns, no inter-patch competition), a documented v1
    simplification of a shared-column mosaic.

    Returns the same 8-tuple as :func:`_prognostic_fluxes`
    ``(gpp_gC, le, h, T_surface, reverted, ts_soil, swc_soil, ustar)`` so the
    caller's output/metrics schema is preserved: the soil-state / ustar / skin-T
    diagnostics are area-weighted across tiles and ``reverted`` is the per-step
    max (a step is flagged reverted if ANY tile rolled it back).
    """
    fracs = [p.frac for p in mosaic.patches]
    gpps, les, hs, tss, revs, tsoils, swcs, ustars = ([], [], [], [], [], [], [], [])
    for p in mosaic.patches:
        cc = CLMMLCanopyConfig(
            pft_clm=int(p.pft_clm), turbulence_scheme=clmml_turbulence,
            stomatal_model=clmml_stomatal,
            vcmax25_override=p.vcmax25_override).validate()
        lc = _build_land_config(
            cc, soil, bottom_bc, soil_depth_m, k_sat_decay_m=k_sat_decay_m,
            soil_evap_resistance_exp=soil_evap_resistance_exp,
            root_depth=p.root_depth_m, z_ref=z_ref, texture=texture,
            interception=interception, plant_wilting_point=plant_wilting_point)
        g, l, h, ts, rev, tsoil, swc, ustar = _prognostic_fluxes(
            d, cc, lc, u_min, nudge_tau_days=nudge_tau_days, clmml_sai=clmml_sai)
        gpps.append(g); les.append(l); hs.append(h); tss.append(ts)
        revs.append(rev); tsoils.append(tsoil); swcs.append(swc); ustars.append(ustar)
    return _aggregate_prognostic_tiles(
        list(zip(gpps, les, hs, tss, revs, tsoils, swcs, ustars)), fracs)


def _aggregate_prognostic_tiles(tile_results, fracs):
    """Area-weight a list of per-tile ``_prognostic_fluxes`` 8-tuples into one grid
    8-tuple ``(gpp, le, h, T_surface, reverted, ts_soil, swc_soil, ustar)``.

    Flux / soil-state / ustar / skin-T series are area-weighted; ``reverted`` is the
    per-step max (a step is reverted if ANY tile rolled it back).  A diagnostic that
    is ``None`` on any tile aggregates to ``None`` (schema preserved).
    """
    cols = list(zip(*tile_results))     # [gpps, les, hs, tss, revs, tsoils, swcs, ustars]
    def _aw(series):
        return None if any(s is None for s in series) \
            else area_weight_series(series, fracs)
    revs = cols[4]
    reverted = None if any(r is None for r in revs) \
        else np.maximum.reduce([np.asarray(r) for r in revs])
    return (_aw(cols[0]), _aw(cols[1]), _aw(cols[2]), _aw(cols[3]),
            reverted, _aw(cols[5]), _aw(cols[6]), _aw(cols[7]))


def _two_leaf_mosaic_prognostic(d: ECSiteDriver, mosaic: "PatchMosaicConfig", *,
                                soil: str, bottom_bc: str, soil_depth_m: float,
                                k_sat_decay_m: float, soil_evap_resistance_exp: float,
                                z_ref: float, texture, interception: bool,
                                plant_wilting_point: float | None, stress_b0: bool,
                                root_depth: float, u_min: float,
                                nudge_tau_days: float):
    """Two-leaf mosaic OUTER-loop (prognostic): one two-leaf column per tile with
    its OWN rooting depth (deep tree vs shallow grass) → its own prognostic
    water-stress, then area-weight.  This is the big-leaf analogue of the CLM-ML
    outer-loop and, unlike the diagnostic INNER-loop mosaic (which shares one
    prescribed ``w_frac_rz``), it captures the dry-season water-stress handoff.
    Independent columns → no inter-patch water competition (documented v1).
    """
    fracs = [p.frac for p in mosaic.patches]
    results = []
    for p in mosaic.patches:
        cc = TwoLeafCanopyConfig(max_iters=30, stress_b0=stress_b0)
        rd = root_depth if p.root_depth_m is None else float(p.root_depth_m)
        lc = _build_land_config(
            cc, soil, bottom_bc, soil_depth_m, k_sat_decay_m=k_sat_decay_m,
            soil_evap_resistance_exp=soil_evap_resistance_exp, root_depth=rd,
            z_ref=z_ref, texture=texture, interception=interception,
            plant_wilting_point=plant_wilting_point)
        # Per-tile canopy params: pathway (fC4) + Vcmax scales; LAI scale.
        dd = _scale_two_leaf_params(d, p.vcmax_c3_scale, p.vcmax_c4_scale, 1.0, 1.0)
        cp = dd.canopy_params
        if p.fc4 is not None:
            cp = cp._replace(fC4=jnp.full_like(cp.fC4, float(p.fc4)))
        if p.lai_scale != 1.0:
            cp = cp._replace(LAI=cp.LAI * p.lai_scale)
        dd = dd._replace(canopy_params=cp)
        results.append(_prognostic_fluxes(
            dd, cc, lc, u_min, nudge_tau_days=nudge_tau_days))
    return _aggregate_prognostic_tiles(results, fracs)


def _is_float_leaf(x) -> bool:
    """True for a floating-point array leaf.

    The carried state is not uniformly floating point: the CLM-ML canopy state
    (``state.canopy_state.mlcanopy``) carries integer topology arrays, so the
    finiteness scan must look at the float leaves only.
    """
    return (hasattr(x, "dtype")
            and jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact))


def _is_array_leaf(x) -> bool:
    """True for any array leaf (float OR int).

    Used by the rollback: a failed step must revert EVERY array leaf, not just
    the float ones, or the canopy state ends up a hybrid of old float
    prognostics and new integer topology/counters.  ``jnp.where`` is dtype-
    agnostic, so this reverts int arrays too; only genuinely non-array leaves
    (None, Python scalars) are carried as-is.
    """
    return hasattr(x, "dtype") and hasattr(x, "shape")


def _any_nonfinite(tree) -> jnp.ndarray:
    """Scalar bool: True if any FLOAT leaf of ``tree`` holds a non-finite value."""
    flags = [jnp.any(~jnp.isfinite(leaf))
             for leaf in jax.tree_util.tree_leaves(tree) if _is_float_leaf(leaf)]
    return jnp.any(jnp.stack(flags)) if flags else jnp.asarray(False)


def _clmml_land_params(cp, sai: float):
    """``CanopyLandParams`` + the structural fields the CLM-ML canopy reads.

    The EC driver builds the two-leaf ``CanopyLandParams`` (``hc``, ``LAI``);
    the multilayer canopy reads ``htop`` / ``SAI`` / ``LAI``.  Rather than
    duplicate the reader, widen the same per-step pytree with ``htop <- hc``
    and a constant ``SAI``.

    NOTE: the two canopy arms are NOT byte-identical in structure — the
    multilayer canopy has a stem-area term the two-leaf big-leaf canopy has
    none of, so ``SAI`` is real extra plant area (+SAI/(LAI+SAI) at US-MMS)
    that only CLM-ML sees.  The *site meteorology, soil, wind floor and scoring
    mask* are identical; the canopy structural parameterisation differs by
    construction.  ``sai`` is exposed as a knob (``--clmml-sai``) so it can be
    set to a site/PFT value or tuned; it is recorded in the output metadata.
    """
    import collections
    cls = collections.namedtuple("ClmMlLandParams", cp._fields + ("SAI", "htop"))
    return cls(*cp, SAI=jnp.full_like(cp.LAI, sai), htop=cp.hc)


def _scale_two_leaf_params(d: ECSiteDriver, vcmax_c3_scale: float,
                           vcmax_c4_scale: float, m_c3_scale: float,
                           m_c4_scale: float) -> ECSiteDriver:
    """Independently scale the two-leaf canopy's C3 (tree) and C4 (grass) params.

    Savanna / woody-savanna sites (US-Ton, AU-How, US-SRM) are a MIX of sparse C3
    trees (overstory) and C4 (or C3) grass (understory); the driver blends them by
    the per-timestep ``fC4`` fraction with a shared ``Vcmax25`` / Ball-Berry slope
    per pathway.  These knobs let the tree (C3) and grass (C4) photosynthetic
    capacity (``Vcmax25``) and stomatal slope (``m``) be tuned separately — e.g.
    down-weight an over-productive grass understory without touching the trees.

    Scales the DRIVER canopy params (two-leaf arm only; CLM-ML reads its own
    ``MLpftcon`` Vcmax / ``g1_BB``).  A scale of 1.0 is a no-op.  NOTE: this scales
    the DifferBESS-provided values, so a scaled run is a SENSITIVITY run, not a
    like-for-like DifferBESS comparison.  It shifts flux MAGNITUDE/bias, not the
    temporal correlation (the savanna GPP-``r`` gap is a two-source phenology
    limitation of the single canopy, which no scalar knob fixes).
    """
    cp = d.canopy_params
    if vcmax_c3_scale != 1.0 or vcmax_c4_scale != 1.0:
        cp = cp._replace(Vcmax25_C3_leaf=cp.Vcmax25_C3_leaf * vcmax_c3_scale,
                         Vcmax25_C4_leaf=cp.Vcmax25_C4_leaf * vcmax_c4_scale)
    if m_c3_scale != 1.0 or m_c4_scale != 1.0:
        cp = cp._replace(m_C3=cp.m_C3 * m_c3_scale, m_C4=cp.m_C4 * m_c4_scale)
    return d._replace(canopy_params=cp)


def _prognostic_fluxes(d: ECSiteDriver,
                       canopy_config: TwoLeafCanopyConfig | CLMMLCanopyConfig,
                       land_config: MultiLayerLandConfig, U_min: float,
                       nudge_tau_days: float = 0.0, clmml_sai: float = _CLMML_SAI):
    """Integrate the FULL multilayer land forward in time (``lax.scan``).

    Unlike diagnostic mode (per-step ``vmap`` with the soil PRESCRIBED), the soil
    moisture (Richards) and soil temperature evolve prognostically and the
    moisture stress ``w_frac_rz`` is computed from the carried ``theta`` — the
    driver's ``T_soil_top`` / ``w_frac_rz`` are used only for the initial state.
    The canopy is invoked inside the land step, which also applies the ``U_min``
    wind floor (so the raw forcing wind is passed through unchanged).

    Soil-moisture nudging (``nudge_tau_days > 0``): after each step the prognostic
    ``theta`` is relaxed toward the OBSERVED volumetric soil moisture with an
    e-folding timescale ``tau`` (``theta += dt/tau * (theta_obs - theta)``).  The
    energy balance, canopy and carbon stay fully prognostic (so H is still solved,
    unlike diagnostic mode), but the water stress tracks observed soil moisture
    rather than the free-running soil hydrology — a standard land-flux evaluation
    technique that isolates the surface flux physics from soil-hydrology drift.

    NaN handling — because the soil state is CARRIED, a single non-finite update
    would otherwise poison every subsequent step.  The carry is therefore made
    atomic: if any state leaf goes non-finite on a step, the *entire* state is
    reverted to the previous step's state (``jnp.where(reverted, old, new)``), so
    a bad step cannot propagate.  That step's emitted fluxes are left non-finite
    and surface via ``model_nan`` in the skill report; ``reverted`` counts how
    many steps were rolled back.

    Returns (gpp_gC, le_wm2, h_wm2, t_surface, reverted) each shape (n_time,).
    """
    is_clmml = isinstance(canopy_config, CLMMLCanopyConfig)
    # Canopy-param C3/C4 scaling (the GPP / Bowen levers) is applied once in
    # ``run()`` before mode dispatch (``_scale_two_leaf_params``) so it reaches
    # BOTH diagnostic and prognostic mode.

    # --- initial soil state from the driver's first finite soil obs ---
    # Initialise from the OBSERVED volumetric soil moisture directly (theta_soil
    # = SWC/100), NOT by inverting w_frac_rz: the reader derives w_frac_rz with
    # its own theta_wp/theta_fc, which differ from this config's, so a round-trip
    # would bias the IC.  theta is clipped to this config's hydraulic range.
    Ts = np.asarray(d.T_soil_top).ravel()
    T_init = float(Ts[np.isfinite(Ts)][0])                       # [K]
    th = np.asarray(d.theta_soil).ravel()
    th0 = th[np.isfinite(th)]
    theta_r = float(land_config.hydraulics.theta_r)
    theta_sat = float(land_config.hydraulics.theta_sat)
    theta_init = float(np.clip(th0[0], theta_r + 1e-3, theta_sat - 1e-3))
    state0 = init_multilayer_land_state(
        1, land_config, T_init=T_init, theta_init=theta_init,
        TgC_init=T_init - constants.T_freeze)

    # Static nudging weight (dt and tau are compile-time constants -> no retrace).
    nudge_alpha = (min(d.dt_s / (nudge_tau_days * 86400.0), 1.0)
                   if nudge_tau_days > 0 else 0.0)
    # Nudge only SUB-SURFACE layers (z > 5 cm): the root zone tracks observed
    # moisture (transpiration stress) while the thin surface layer dries freely
    # between rains, so the (top-layer-based) bare-soil evaporation is realistic.
    _zc = np.cumsum(np.asarray(make_soil_grid(land_config.soil_grid).dz))
    nudge_layer_mask = jnp.asarray(_zc >= 0.05)
    # Model layer nearest the ~5 cm shallowest soil sensor, for the soil-state
    # (T / moisture) evaluation against the driver's shallowest TS/SWC.
    _znode = np.asarray(make_soil_grid(land_config.soil_grid).z_node)
    _i5 = int(np.argmin(np.abs(_znode - 0.05)))

    # The CLM-ML multilayer canopy needs the site latitude (host-side CLM solar
    # geometry); the two-leaf canopy takes its radiation straight from the driver.
    lat_arg = jnp.asarray([np.rad2deg(d.lat_rad)]) if is_clmml else None

    def _scan_step(state, xs):
        forcing_t, params_t, doy_t, theta_obs_t = xs
        new_state, _resp, _carbon, out = step_multilayer_land_with_diagnostics(
            state, forcing_t, land_config, U_min, d.dt_s,
            lat=lat_arg, carbon_state=None, doy=doy_t, land_params=params_t)
        # The CLM-ML cold-start step grows the carry (canopy_state None ->
        # CanopyState), so there is no same-structure previous state to revert
        # to; that step is taken unguarded and every later step is guarded.
        if (jax.tree_util.tree_structure(new_state)
                == jax.tree_util.tree_structure(state)):
            reverted = _any_nonfinite(new_state)
            # Revert EVERY array leaf (int topology included) so a rolled-back
            # step cannot leave a hybrid of old float prognostics + new integer
            # counters; only non-array leaves (None) are carried unchanged.
            safe_state = jax.tree_util.tree_map(
                lambda n, o: jnp.where(reverted, o, n) if _is_array_leaf(n) else n,
                new_state, state)
        else:
            # Structure grew (CLM-ML cold start: canopy_state None -> CanopyState).
            # There is no same-shape previous state to revert to, so a poisoned
            # cold-start state would become the "old" state and every later
            # rollback would restore NaNs forever.  Guard it explicitly instead.
            if bool(_any_nonfinite(new_state)):
                raise FloatingPointError(
                    "CLM-ML cold-start step produced a non-finite land state; "
                    "the vertical structure or first flux solve failed. Check "
                    "the canopy config (layering, PFT, htop/LAI) — a cold-start "
                    "NaN cannot be rolled back and would poison the whole run.")
            reverted = jnp.asarray(False)
            safe_state = new_state
        if nudge_alpha > 0.0:
            # Relax theta toward observed SWC (all layers; only where obs finite),
            # then make psi consistent.  Keeps water stress realistic while the
            # energy/canopy/carbon stay prognostic.
            th = safe_state.theta_soil
            obs = theta_obs_t[:, None]
            th_nudged = jnp.where(
                jnp.isfinite(obs) & nudge_layer_mask[None, :],
                jnp.clip(th + nudge_alpha * (obs - th), theta_r, theta_sat),
                th)
            safe_state = safe_state._replace(
                theta_soil=th_nudged,
                psi_soil=psi_from_theta(th_nudged, land_config.hydraulics))
        # On revert the post-flux soil update was rejected, so the step's
        # diagnostics are NOT trustworthy even when `out` itself is finite
        # (the failure is downstream of the surface flux calc).  Mask the
        # emitted fluxes to NaN so they count as model_nan and are excluded
        # from skill metrics rather than silently scored.
        nan = jnp.array(jnp.nan, dtype=jnp.float64)
        masked = lambda v: jnp.where(reverted, nan, v)
        # friction velocity u* from the modelled momentum stress:
        # |tau| = rho * u*^2  ->  u* = sqrt(|tau|/rho).
        u_star = jnp.sqrt(
            jnp.sqrt(out.tau_x ** 2 + out.tau_y ** 2)
            / jnp.maximum(forcing_t.rho_lowest, 1e-3))
        emit = (masked(out.gpp), masked(out.lhflx), masked(out.shflx),
                masked(out.T_surface), reverted.astype(jnp.float64),
                safe_state.T_soil[:, _i5], safe_state.theta_soil[:, _i5],
                masked(u_star))
        return safe_state, emit

    params = (_clmml_land_params(d.canopy_params, clmml_sai) if is_clmml
              else d.canopy_params)
    xs = (d.forcing, params, d.doy, jnp.asarray(d.theta_soil))
    if is_clmml:
        # CLM-ML runs EAGERLY, one Python step at a time.  Its interface reads
        # per-step canopy structure on the host (``float(land_params.LAI[i])``,
        # ``htop``), so a time-varying-LAI rollout cannot be traced yet — the
        # jit-traceable forward path assumes a fixed canopy structure.
        # ponytail: eager loop; switch to lax.scan once the per-step structural
        # reads are de-hosted (CLM-ML de-host stage S2+).  Physics, config,
        # driver and scoring are identical to the two-leaf arm either way.
        n_steps = int(d.forcing.T_lowest.shape[0])
        state, emits = state0, []
        for i in range(n_steps):
            xi = jax.tree_util.tree_map(lambda a: a[i], xs)
            state, e = _scan_step(state, xi)
            emits.append(jax.tree_util.tree_map(np.asarray, e))
            if (i + 1) % 200 == 0:
                print(f"  clmml step {i + 1}/{n_steps}", flush=True)
        gpp, le, h, ts, reverted, ts_soil, swc_soil, ustar = (
            np.stack(v) for v in zip(*emits))
    else:
        run = jax.jit(lambda s0, x: jax.lax.scan(_scan_step, s0, x))
        _final, (gpp, le, h, ts, reverted, ts_soil, swc_soil, ustar) = run(state0, xs)
    rav = lambda a: np.asarray(a).ravel()
    return (rav(gpp), rav(le), rav(h), rav(ts), rav(reverted),
            rav(ts_soil), rav(swc_soil), rav(ustar))


def _skill(model: np.ndarray, obs: np.ndarray, valid: np.ndarray) -> dict:
    """RMSE / bias / Pearson r / count over valid + jointly-finite samples."""
    m, o = np.asarray(model).ravel(), np.asarray(obs).ravel()
    vmask = np.asarray(valid).ravel()
    # Steps the model itself failed (non-finite) on a valid forcing step — reported
    # separately so the masked-out failures are not hidden in an optimistic skill.
    model_nan = int((vmask & ~np.isfinite(m)).sum())
    mask = vmask & np.isfinite(m) & np.isfinite(o)
    k = int(mask.sum())
    if k < 2:
        return dict(n=k, model_nan=model_nan, rmse=float("nan"),
                    bias=float("nan"), r=float("nan"))
    mm, oo = m[mask], o[mask]
    rmse = float(np.sqrt(np.mean((mm - oo) ** 2)))
    bias = float(np.mean(mm - oo))
    r = float(np.corrcoef(mm, oo)[0, 1]) if np.std(mm) > 0 and np.std(oo) > 0 else float("nan")
    return dict(n=k, model_nan=model_nan, rmse=rmse, bias=bias, r=r)


def _write_output(out_dir: str, d: ECSiteDriver, model: dict,
                  mode: str = "diagnostic",
                  reverted: np.ndarray | None = None,
                  score_valid: np.ndarray | None = None,
                  extra_attrs: dict | None = None) -> str:
    import xarray as xr
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{d.site_id}_ec_{mode}.nc")
    t = np.arange(model["gpp_umol"].shape[0])
    data = {
        "gpp_mod": ("time", model["gpp_umol"]), "gpp_obs": ("time", d.obs["gpp_umol"]),
        "le_mod": ("time", model["le_wm2"]), "le_obs": ("time", d.obs["le_wm2"]),
        "h_mod": ("time", model["h_wm2"]), "h_obs": ("time", d.obs["h_wm2"]),
        "valid": ("time", d.valid.astype("i1")),
    }
    if "ts_soil" in model:
        # prognostic soil STATE: top (~5 cm) model soil T / moisture vs the driver's
        # SHALLOWEST observed TS / SWC.
        data["ts_mod"] = ("time", model["ts_soil"])
        data["ts_obs"] = ("time", np.asarray(d.T_soil_top).ravel())
        data["swc_mod"] = ("time", model["swc_soil"])
        data["swc_obs"] = ("time", np.asarray(d.theta_soil).ravel())
        data["ustar_mod"] = ("time", model["ustar"])
        data["ustar_obs"] = ("time", np.asarray(d.obs["ustar"]).ravel())
    if reverted is not None:
        # prognostic: per-step flag, 1 where the soil update was NaN-reverted.
        data["reverted"] = ("time", np.asarray(reverted).astype("i1"))
    if d.met_filled is not None:
        # forcing provenance: 1 where an atmospheric forcing field was gap-filled.
        data["met_filled"] = ("time", np.asarray(d.met_filled).astype("i1"))
    if d.soil_filled is not None:
        data["soil_filled"] = ("time", np.asarray(d.soil_filled).astype("i1"))
    if score_valid is not None:
        # the EXACT mask used for the reported skill metrics, so downstream users
        # can reproduce the metric sample set from the file alone.
        data["score_valid"] = ("time", np.asarray(score_valid).astype("i1"))
    ds = xr.Dataset(
        data,
        coords={"time": t},
        attrs=dict(site=d.site_id, pft=d.pft, climate=d.climate, igbp=int(d.igbp),
                   lat_deg=float(np.rad2deg(d.lat_rad)), lon_deg=float(d.lon_deg),
                   dt_s=float(d.dt_s), mode=mode,
                   gpp_units="umolCO2/m2/s", le_units="W/m2", h_units="W/m2",
                   **(extra_attrs or {})),
    )
    ds.to_netcdf(path)
    return path


def _slice_driver(d: ECSiteDriver, start: int, k: int) -> ECSiteDriver:
    """``[start : start+k]`` timestep view of a driver (quick runs / smoke test)."""
    sl = slice(start, start + k)
    return d._replace(
        forcing=jax.tree_util.tree_map(lambda a: a[sl], d.forcing),
        canopy_params=jax.tree_util.tree_map(lambda a: a[sl], d.canopy_params),
        T_soil_top=d.T_soil_top[sl], w_frac_rz=d.w_frac_rz[sl],
        theta_soil=d.theta_soil[sl],
        wind_speed=d.wind_speed[sl], valid=d.valid[sl], doy=d.doy[sl],
        obs={kk: vv[sl] for kk, vv in d.obs.items()},
        met_filled=(None if d.met_filled is None else d.met_filled[sl]),
        soil_filled=(None if d.soil_filled is None else d.soil_filled[sl]),
    )


def _best_year_slice(driver_nc: str, d) -> tuple[int, int, int]:
    """Return ``(start_step, n_steps, year)`` for the calendar year with the most
    genuinely-observed flux steps.

    A single contiguous calendar year spans BOTH the growing and non-growing
    season (required for a prognostic free-run, whose soil state is carried
    step-to-step), and is ~12-24x shorter than a full multi-decade FLUXNET record
    — the dominant cost of the offline scan.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    t = pd.DatetimeIndex(xr.open_dataset(driver_nc)["time"].values)
    valid = np.asarray(d.valid).ravel()
    le = np.asarray(d.obs["le_wm2"]).ravel()
    good = valid & np.isfinite(le)
    years = t.year.to_numpy()
    n = min(len(years), good.size)
    years, good = years[:n], good[:n]
    counts = {int(y): int(good[years == y].sum()) for y in np.unique(years)}
    best = max(counts, key=counts.get)
    idx = np.nonzero(years == best)[0]
    return int(idx[0]), int(len(idx)), best


def run_site(driver_nc: str, mode: str, out_dir: str, chunk: int,
             max_steps: int | None = None, start_step: int = 0,
             soil: str = "default", bottom_bc: str = "free_drainage",
             soil_depth_m: float | None = None, nudge_tau_days: float = 0.0,
             k_sat_decay_m: float = 0.0, soil_evap_resistance_exp: float = 2.0,
             root_depth: float | None = None, z_ref: float | None = None,
             stress_b0: bool = False, select_best_year: bool = False,
             texture_csv: str = _DEFAULT_TEXTURE_CSV,
             canopy: str = "two_leaf", clm_pft: int | None = None,
             clmml_sai: float = _CLMML_SAI, u_min: float = _U_MIN,
             stomatal_m_scale: float = 1.0, vcmax_scale: float = 1.0,
             clmml_turbulence: str = "rsl_bonan", spinup_steps: int = 0,
             interception: bool = False, clmml_stomatal: str = "ball_berry",
             clmml_vcmax25: float | None = None,
             plant_wilting_point: float | None = None,
             vcmax_c3_scale: float | None = None,
             vcmax_c4_scale: float | None = None,
             stomatal_m_c3_scale: float | None = None,
             stomatal_m_c4_scale: float | None = None,
             mosaic: str = "none", tree_frac: float = 0.4,
             savanna_grass_pft: int = 14, savanna_grass_root_m: float = 0.5,
             savanna_grass_fc4: float = 1.0) -> dict:
    if mode not in ("diagnostic", "prognostic"):
        raise ValueError(f"mode {mode!r} not supported (diagnostic|prognostic)")
    if canopy not in ("two_leaf", "clmml"):
        raise ValueError(f"canopy {canopy!r} not supported (two_leaf|clmml)")
    # N-patch mosaic (tree + grass tiles) for savanna sites.  Fail early on an
    # unknown selection or an unsupported combination (dispatch hardening).  The
    # concrete mosaic configs are built lower down, once the site root depth is
    # resolved: the two-leaf mosaic runs INNER-loop (patches on the canopy ncol
    # axis) in diagnostic mode; the CLM-ML mosaic runs OUTER-loop (one prognostic
    # column per tile, area-weighted) because the CLM-ML forward is single-PFT.
    if mosaic not in ("none", "savanna"):
        raise ValueError(f"mosaic {mosaic!r} not supported (none|savanna)")
    if mosaic == "savanna":
        # two_leaf: diagnostic -> INNER-loop mosaic (shares one prescribed
        # w_frac_rz); prognostic -> OUTER-loop mosaic (per-tile rooting => per-tile
        # water stress, the real dry-season handoff).  Both modes supported.
        if canopy == "clmml" and mode != "prognostic":
            raise ValueError(
                "mosaic=savanna with --canopy clmml requires --mode prognostic")
        # An explicit --clmml-vcmax25 (if given) overrides the TREE tile's Vcmax
        # (see the clmml_mosaic build below); grass keeps its PFT-table value.
    mosaic_cfg: PatchMosaicConfig | None = None
    clmml_mosaic: ClmmlMosaicConfig | None = None
    if canopy == "clmml" and mode != "prognostic":
        # Diagnostic mode vmaps the canopy over timesteps; the CLM-ML interface
        # mutates CLM module globals per step and cannot be vmapped.
        raise ValueError("--canopy clmml requires --mode prognostic")
    d = read_ec_site_driver(driver_nc)
    # Site-level physics from the consolidated table (tower height, phreatophyte
    # root/column depth).  An explicit non-None argument (CLI override) wins; a
    # None falls back to the per-site table value for reproducibility.
    phys = ec_site_physics(d.site_id)
    z_ref = phys["z_ref"] if z_ref is None else z_ref
    root_depth = phys["root_depth"] if root_depth is None else root_depth
    soil_depth_m = phys["soil_depth_m"] if soil_depth_m is None else soil_depth_m
    if select_best_year:
        start_step, max_steps, _yr = _best_year_slice(driver_nc, d)
        print(f"  select-best-year: {_yr} (steps {start_step}..{start_step + max_steps}, "
              f"{max_steps} of {int(d.forcing.T_lowest.shape[0])})")
    # Soil-moisture spin-up: prepend ``spinup_steps`` before the evaluation window
    # so the prognostic soil column reaches a realistic state before scoring.
    # Essential at seasonally-dry (Mediterranean) sites: a mid-summer cold start
    # initialises the whole column at the dry observed surface moisture, which for
    # a phreatophyte (US-Ton oaks tapping deep winter-recharged water) is at/below
    # wilting -> root-zone stress w_frac_rz = 0 -> Vcmax * 0 -> GPP identically
    # zero.  Starting earlier fills the deep root zone from the wet season and the
    # summer GPP recovers.  The spin-up steps run but are trimmed from the scored
    # output (prognostic only; diagnostic mode has no carried soil state).
    n_spinup = 0
    if spinup_steps > 0 and mode == "prognostic":
        n_spinup = min(spinup_steps, start_step)
        start_step -= n_spinup
        if max_steps is not None:
            max_steps += n_spinup
        print(f"  spin-up: {n_spinup} steps before the evaluation window "
              f"(trimmed from scoring)")
    if max_steps is not None or start_step:
        n = int(d.forcing.T_lowest.shape[0])
        k = (n - start_step) if max_steps is None else max_steps
        d = _slice_driver(d, start_step, k)
    texture = None
    if soil == "auto":
        texture = _texture_lookup(d.site_id, texture_csv)
        if texture is None:
            print(f"  texture=auto: {d.site_id} not in {texture_csv}; using loam default")
        else:
            print(f"  texture=auto: sand={texture[0]:.0f}% clay={texture[1]:.0f}%")
    # stress_b0=False keeps the Ball-Berry cuticular intercept b0 unstressed:
    # the leaf cuticle keeps leaking under drought, so a live (evergreen /
    # phreatophytic) canopy sustains a baseline transpiration, while a
    # deciduous / senescent canopy self-limits because its LAI -> 0.  This is
    # the physically-general default for the offline sites (see EC_SITE_PHYSICS).
    if canopy == "two_leaf":
        canopy_config = TwoLeafCanopyConfig(max_iters=30, stress_b0=stress_b0)
    elif canopy == "clmml":
        # CLM-ML multilayer canopy.  num_ml_steps=None derives the canopy
        # sub-step from dtime_ml_target_s (300 s), so an hourly EC driver
        # sub-cycles instead of running the stiff canopy-air storage term at
        # the host step (PR #1240).
        # turbulence_scheme selects the canopy-airspace scheme: "rsl_bonan"
        # (Harman-Finnigan roughness sublayer, default) or "most" (Monin-Obukhov
        # only).  RSL's within-canopy wind profile takes log((z-d)/(ztop-d)),
        # which goes to a math-domain error for a very tall canopy where a layer
        # height drops below the displacement height — "most" avoids that.
        # PARITY with two-leaf (a richer canopy must get the SAME site inputs, not
        # generic PFT-table constants, or it loses skill by construction of the
        # harness).  (a) PFT from the site IGBP+climate, not a fixed default 7;
        # (b) Vcmax25 = the driver's per-site value (what two-leaf uses), not the
        # MLpftcon table constant.  An explicit CLI value still wins.
        pft_resolved = (clm_pft_for_site(d.igbp, d.climate)
                        if clm_pft is None else int(clm_pft))
        vcmax_resolved = (_site_mean_vcmax25(d)
                          if clmml_vcmax25 is None else clmml_vcmax25)
        print(f"  clmml: pft={pft_resolved} (IGBP={int(d.igbp)}, {d.climate}), "
              f"Vcmax25={vcmax_resolved!r}, stomatal={clmml_stomatal}")
        canopy_config = CLMMLCanopyConfig(
            pft_clm=pft_resolved, turbulence_scheme=clmml_turbulence,
            stomatal_model=clmml_stomatal,
            vcmax25_override=vcmax_resolved).validate()
    else:
        raise ValueError(f"canopy {canopy!r} not supported (two_leaf|clmml)")
    land_config = _build_land_config(
        canopy_config, soil, bottom_bc, soil_depth_m,
        k_sat_decay_m=k_sat_decay_m,
        soil_evap_resistance_exp=soil_evap_resistance_exp,
        root_depth=root_depth, z_ref=z_ref, texture=texture,
        interception=interception, plant_wilting_point=plant_wilting_point)

    # Independent C3 (tree) / C4 (grass) canopy-param tuning for savanna sites.
    # A per-pathway flag (``--vcmax-c3-scale`` etc.) wins; otherwise the combined
    # ``--vcmax-scale`` / ``--stomatal-m-scale`` applies to both pathways.  Applied
    # here (before mode dispatch) so it reaches diagnostic AND prognostic; two-leaf
    # only (CLM-ML reads its own MLpftcon Vcmax / g1_BB).
    if canopy == "two_leaf":
        eff_vc3 = vcmax_scale if vcmax_c3_scale is None else vcmax_c3_scale
        eff_vc4 = vcmax_scale if vcmax_c4_scale is None else vcmax_c4_scale
        eff_mc3 = stomatal_m_scale if stomatal_m_c3_scale is None else stomatal_m_c3_scale
        eff_mc4 = stomatal_m_scale if stomatal_m_c4_scale is None else stomatal_m_c4_scale
        d = _scale_two_leaf_params(d, eff_vc3, eff_vc4, eff_mc3, eff_mc4)

    # Build the concrete savanna mosaic now that the site root depth is resolved:
    # the tree tile keeps the site (deep phreatophyte) rooting; the grass tile is
    # shallow-rooted.  Two-leaf -> inner-loop (diagnostic); CLM-ML -> outer-loop.
    if mosaic == "savanna":
        if canopy == "two_leaf":
            # tree tile keeps the site (deep) rooting; grass tile is shallow.  The
            # per-tile root depth only bites in the prognostic OUTER-loop.
            mosaic_cfg = savanna_two_patch(
                tree_frac=tree_frac, grass_fc4=savanna_grass_fc4,
                tree_root_m=float(root_depth), grass_root_m=savanna_grass_root_m)
        else:                                              # clmml (prognostic)
            # Tree tile PFT from the site IGBP when --clm-pft is left to derive.
            tree_pft = (clm_pft_for_site(d.igbp, d.climate)
                        if clm_pft is None else int(clm_pft))
            # Inject the driver site Vcmax (what two-leaf uses) into the tree tile
            # for parity; grass keeps its PFT table value unless overridden.
            _site_vcmax = (_site_mean_vcmax25(d)
                           if clmml_vcmax25 is None else clmml_vcmax25)
            clmml_mosaic = savanna_clmml_two_patch(
                tree_frac=tree_frac, tree_pft=tree_pft, grass_pft=savanna_grass_pft,
                tree_root_m=float(root_depth), grass_root_m=savanna_grass_root_m,
                tree_vcmax25=_site_vcmax)

    reverted = None
    ts_soil = swc_soil = ustar = None
    if mode == "diagnostic":
        gpp_gC, le, h, _ = _diagnostic_fluxes(d, canopy_config, land_config, chunk,
                                              mosaic=mosaic_cfg)
    elif mosaic_cfg is not None:
        # Two-leaf prognostic OUTER-loop mosaic (per-tile rooting -> per-tile water
        # stress).  Preserves the prognostic 8-tuple output schema.
        gpp_gC, le, h, _ts, reverted, ts_soil, swc_soil, ustar = _two_leaf_mosaic_prognostic(
            d, mosaic_cfg, soil=soil, bottom_bc=bottom_bc, soil_depth_m=soil_depth_m,
            k_sat_decay_m=k_sat_decay_m,
            soil_evap_resistance_exp=soil_evap_resistance_exp, z_ref=z_ref,
            texture=texture, interception=interception,
            plant_wilting_point=plant_wilting_point, stress_b0=stress_b0,
            root_depth=float(root_depth), u_min=u_min, nudge_tau_days=nudge_tau_days)
    elif clmml_mosaic is not None:
        # CLM-ML outer-loop mosaic: run each tile as its own prognostic column and
        # area-weight (soil-state / ustar diagnostics area-weighted; reverted =
        # per-step any-tile max) so the prognostic output schema is preserved.
        gpp_gC, le, h, _ts, reverted, ts_soil, swc_soil, ustar = _clmml_mosaic_prognostic(
            d, clmml_mosaic, soil=soil, bottom_bc=bottom_bc,
            soil_depth_m=soil_depth_m, k_sat_decay_m=k_sat_decay_m,
            soil_evap_resistance_exp=soil_evap_resistance_exp, z_ref=z_ref,
            texture=texture, interception=interception,
            plant_wilting_point=plant_wilting_point,
            clmml_turbulence=clmml_turbulence, clmml_stomatal=clmml_stomatal,
            u_min=u_min, nudge_tau_days=nudge_tau_days, clmml_sai=clmml_sai)
    else:
        gpp_gC, le, h, _ts, reverted, ts_soil, swc_soil, ustar = _prognostic_fluxes(
            d, canopy_config, land_config, u_min, nudge_tau_days=nudge_tau_days,
            clmml_sai=clmml_sai)
    # Trim the spin-up window from BOTH the model arrays and the driver ``d``
    # (obs/valid/provenance) so scoring, output and provenance all cover the
    # evaluation window only, while the carried soil state that reaches it was
    # spun up.  Keep the two in lock-step so score_valid still aligns.
    if n_spinup > 0:
        _tr = lambda a: None if a is None else a[n_spinup:]
        gpp_gC, le, h = _tr(gpp_gC), _tr(le), _tr(h)
        reverted, ts_soil = _tr(reverted), _tr(ts_soil)
        swc_soil, ustar = _tr(swc_soil), _tr(ustar)
        d = _slice_driver(d, n_spinup, int(d.forcing.T_lowest.shape[0]) - n_spinup)

    model = {
        "gpp_umol": gpp_gC / _GC_PER_UMOL_CO2,   # gC/m2/s -> umolCO2/m2/s (obs units)
        "le_wm2": le, "h_wm2": h,
    }
    if ts_soil is not None:
        model["ts_soil"] = ts_soil       # top (~5 cm) model soil T [K]
        model["swc_soil"] = swc_soil     # top (~5 cm) model soil moisture [m3/m3]
        model["ustar"] = ustar           # friction velocity [m/s]
    # Score on genuinely-OBSERVED forcing: when running a gap-free driver, exclude
    # steps whose forcing was gap-filled so skill is not credited to synthesised
    # forcing.  Atmospheric fill (met_filled) is per-step forcing in BOTH modes.
    # Soil fill (soil_filled) is per-step forcing ONLY in diagnostic mode (TS/SWC
    # prescribed); in prognostic mode the soil is integrated, so soil fill matters
    # only for the initial state (reported separately below).  For a standard
    # driver_v2 (flags are None) this is a no-op.  Provenance is persisted to out.
    score_valid = d.valid
    if d.met_filled is not None:
        score_valid = score_valid & (np.asarray(d.met_filled).ravel() == 0)
    if mode == "diagnostic" and d.soil_filled is not None:
        score_valid = score_valid & (np.asarray(d.soil_filled).ravel() == 0)
    metrics = {
        "GPP": _skill(model["gpp_umol"], d.obs["gpp_umol"], score_valid),
        "LE": _skill(model["le_wm2"], d.obs["le_wm2"], score_valid),
        "H": _skill(model["h_wm2"], d.obs["h_wm2"], score_valid),
    }
    # Prognostic soil STATE evaluation: top (~5 cm) model soil T / moisture vs the
    # driver's SHALLOWEST observed TS / SWC, scored only where those obs are genuine
    # (soil_filled == 0), so gap-filled soil obs never credit the state skill.
    if "ts_soil" in model:
        soil_score = score_valid
        if d.soil_filled is not None:
            soil_score = soil_score & (np.asarray(d.soil_filled).ravel() == 0)
        metrics["TS"] = _skill(model["ts_soil"], np.asarray(d.T_soil_top), soil_score)
        metrics["SWC"] = _skill(model["swc_soil"], np.asarray(d.theta_soil), soil_score)
        # friction velocity: modelled u* (from momentum stress) vs observed USTAR.
        metrics["USTAR"] = _skill(model["ustar"], np.asarray(d.obs["ustar"]), score_valid)
    # Tag the output with the canopy arm so a two-leaf and a CLM-ML run of the
    # same site/mode do not overwrite each other.
    extra_attrs = {"canopy": canopy}
    if canopy == "clmml":
        extra_attrs.update(clm_pft=clm_pft, clmml_sai=float(clmml_sai),
                           clmml_turbulence=clmml_turbulence,
                           nlevmlcan=canopy_config.nlevmlcan,
                           nlayer_above=canopy_config.nlayer_above)
    path = _write_output(out_dir, d, model,
                         mode=mode if canopy == "two_leaf" else f"{mode}_{canopy}",
                         reverted=reverted, score_valid=score_valid,
                         extra_attrs=extra_attrs)
    forcing_note = ("" if d.met_filled is None else
                    f"; scored on observed forcing only "
                    f"({100 * (d.met_filled == 0).mean():.0f}% of steps)")
    print(f"\n=== {d.site_id}  (PFT={d.pft}, {d.climate}; mode={mode}; "
          f"valid {100 * d.valid.mean():.0f}% of {d.valid.size} steps{forcing_note}) ===")
    if reverted is not None:
        nrev = int(np.nansum(reverted))
        print(f"  prognostic soil: {nrev} step(s) NaN-reverted "
              f"({100 * nrev / reverted.size:.3f}% of run)")
        # The soil IC is taken from the first finite soil obs (step 0 on a
        # gap-free driver); report whether that value was itself gap-filled.
        if d.soil_filled is not None:
            ic_src = "FILLED" if int(np.asarray(d.soil_filled).ravel()[0]) else "observed"
            print(f"  prognostic soil: initial state from {ic_src} TS/SWC at step 0")
    for flux, mtr in metrics.items():
        print(f"  {flux:3s}  n={mtr['n']:>7d}  model_nan={mtr['model_nan']:>5d}  "
              f"RMSE={mtr['rmse']:8.3f}  bias={mtr['bias']:+8.3f}  r={mtr['r']:.3f}")
    print(f"  -> {path}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--driver-nc", nargs="+", required=True,
                    help="one or more DifferBESS <SITE>_driver_v2.nc paths")
    ap.add_argument("--mode", default="diagnostic",
                    choices=["diagnostic", "prognostic"])
    ap.add_argument("--out", default="diagnostics/ec_site")
    ap.add_argument("--chunk", type=int, default=_DEFAULT_CHUNK,
                    help="timesteps per vmap chunk (memory control)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="limit to N timesteps from --start-step (quick runs)")
    ap.add_argument("--start-step", type=int, default=0,
                    help="first timestep index to run (skip early gappy record)")
    ap.add_argument("--soil", default="default",
                    choices=["default", "siltloam", "sandyloam", "auto"],
                    help="prognostic hydraulics: a preset, or 'auto' = per-site "
                         "van-Genuchten from the --texture-csv table")
    ap.add_argument("--texture-csv", default=_DEFAULT_TEXTURE_CSV,
                    help="site->(sand,clay) table for --soil auto "
                         "(build with scripts/data/build_ec_site_texture.py)")
    ap.add_argument("--bottom-bc", default="free_drainage",
                    choices=["free_drainage", "zero_flux"],
                    help="Richards bottom boundary condition (prognostic)")
    ap.add_argument("--soil-depth-m", type=float, default=None,
                    help="total soil column depth [m] (default: per-site "
                         "EC_SITE_PHYSICS table, else legoESM ~6.4 m column)")
    ap.add_argument("--z-ref", type=float, default=None,
                    help="wind/flux reference (tower) height [m] for the "
                         "Monin-Obukhov surface layer (default: per-site BADM "
                         "Reference_height_v from EC_SITE_PHYSICS, else 10 m)")
    ap.add_argument("--stress-b0", dest="stress_b0", action="store_true",
                    default=False,
                    help="down-regulate the Ball-Berry cuticular intercept b0 by "
                         "soil-moisture stress too (legacy); default leaves b0 "
                         "unstressed so live canopies keep a baseline transpiration")
    ap.add_argument("--nudge-tau-days", type=float, default=0.0,
                    help="prognostic mode: relax soil moisture toward observed SWC "
                         "with this e-folding timescale [days] (0 = free-running)")
    ap.add_argument("--k-sat-decay-m", type=float, default=0.0,
                    help="Niu-2005 K_sat(z)=K0*exp(-z/L) retention decay length L [m] "
                         "(0 = uniform K, no depth decay)")
    ap.add_argument("--soil-evap-resistance-exp", type=float, default=2.0,
                    help="S_top**exp bare-soil evaporation throttle on the canopy path "
                         "(0 = Kelvin-h_r only; 2 = #671 default; 3-4 = stronger)")
    ap.add_argument("--root-depth", type=float, default=None,
                    help="root e-folding depth [m] (deeper => more deep-water "
                         "access; default: per-site EC_SITE_PHYSICS table, else 1 m)")
    ap.add_argument("--canopy", default="two_leaf", choices=["two_leaf", "clmml"],
                    help="surface canopy scheme: the two-leaf big-leaf canopy "
                         "(default) or the CLM-ML multilayer canopy (requires the "
                         "'canopy' extra; --mode prognostic only, runs eagerly)")
    ap.add_argument("--clm-pft", type=int, default=None,
                    help="CLM PFT index for --canopy clmml.  Default (None) DERIVES "
                         "the PFT from the site IGBP+climate (clm_pft_for_site) so "
                         "CLM-ML is site-appropriate like two-leaf; pass an int to "
                         "override (7 = broadleaf-deciduous-temperate, 13 = C3 "
                         "grass, 14 = C4 grass)")
    ap.add_argument("--clmml-turbulence", default="rsl_bonan",
                    choices=list(VALID_CLM_ML_TURBULENCE_SCHEMES),
                    help="CLM-ML canopy-airspace turbulence scheme: rsl_bonan "
                         "(roughness sublayer, default) or most (Monin-Obukhov; "
                         "avoids the RSL wind-profile domain error on very tall "
                         "canopies)")
    ap.add_argument("--clmml-stomatal", default="ball_berry",
                    choices=list(VALID_CLM_ML_STOMATAL_MODELS),
                    help="CLM-ML leaf stomatal model: ball_berry (default, matches "
                         "the two-leaf empirical closure — CLM g1_BB=9/g0_BB=0.01 "
                         "equal the driver m_C3=9/b0=0.01), medlyn, or wue")
    ap.add_argument("--clmml-vcmax25", type=float, default=None,
                    help="per-site CLM-ML Vcmax25 [umol/m2/s].  Default (None) uses "
                         "the driver's site-mean Vcmax25 (the value two-leaf uses) "
                         "instead of the generic MLpftcon table; applies under any "
                         "stomatal model")
    ap.add_argument("--clmml-sai", type=float, default=_CLMML_SAI,
                    help="stem area index [m2/m2] for the CLM-ML canopy (the "
                         "two-leaf arm has no stem-area term); tunable")
    ap.add_argument("--stomatal-m-scale", type=float, default=1.0,
                    help="scale the two-leaf Ball-Berry slope m (transpiration "
                         "per assimilation); >1 => more LE, less H at fixed GPP")
    ap.add_argument("--vcmax-scale", type=float, default=1.0,
                    help="scale the two-leaf Vcmax25 (photosynthetic capacity); "
                         ">1 => more GPP. The GPP knob, tuned separately from gs")
    # Independent C3 (tree) / C4 (grass) tuning for savanna sites (US-Ton, AU-How,
    # US-SRM): sparse trees + grass understory blended by the driver fC4 fraction.
    # A per-pathway flag overrides the combined --vcmax-scale/--stomatal-m-scale
    # for that pathway; default None => the combined value applies to both.
    ap.add_argument("--vcmax-c3-scale", type=float, default=None,
                    help="scale ONLY the C3 (tree/overstory) Vcmax25; overrides "
                         "--vcmax-scale for the C3 pathway (savanna tuning)")
    ap.add_argument("--vcmax-c4-scale", type=float, default=None,
                    help="scale ONLY the C4 (grass/understory) Vcmax25; overrides "
                         "--vcmax-scale for the C4 pathway (savanna tuning)")
    ap.add_argument("--stomatal-m-c3-scale", type=float, default=None,
                    help="scale ONLY the C3 (tree) Ball-Berry slope m; overrides "
                         "--stomatal-m-scale for the C3 pathway")
    ap.add_argument("--stomatal-m-c4-scale", type=float, default=None,
                    help="scale ONLY the C4 (grass) Ball-Berry slope m; overrides "
                         "--stomatal-m-scale for the C4 pathway")
    # N-patch canopy mosaic for savanna sites (US-Ton, AU-How, US-SRM): run the
    # canopy per tile (C3 tree + C4 grass) and area-weight, instead of one blended
    # canopy.  Diagnostic-mode, two-leaf only for now.
    ap.add_argument("--mosaic", default="none", choices=["none", "savanna"],
                    help="canopy mosaic: 'savanna' = 2 patches (C3 tree overstory "
                         "+ C4 grass understory), area-weighted; diagnostic + "
                         "two_leaf only")
    ap.add_argument("--tree-frac", type=float, default=0.4,
                    help="woody (tree) area fraction for --mosaic savanna; grass "
                         "is the residual 1-tree_frac")
    ap.add_argument("--savanna-grass-pft", type=int, default=14,
                    help="CLM PFT index for the grass tile of the CLM-ML savanna "
                         "mosaic (default 14 = C4 grass; 13 = C3 grass for a "
                         "Mediterranean savanna); tree tile uses --clm-pft/IGBP")
    ap.add_argument("--savanna-grass-root-m", type=float, default=0.5,
                    help="grass-tile root depth [m] for the savanna mosaic "
                         "(shallow); the tree tile uses the site root depth")
    ap.add_argument("--savanna-grass-fc4", type=float, default=1.0,
                    help="grass-tile C4 fraction for the two-leaf savanna mosaic: "
                         "1.0 = C4 grass (tropical savanna), 0.0 = C3 grass "
                         "(Mediterranean, e.g. US-Ton)")
    ap.add_argument("--u-min", type=float, default=_U_MIN,
                    help="wind-speed floor [m/s]; canopy sees "
                         "sqrt(u^2+v^2+u_min^2). Shared by both arms; lower => "
                         "less aerodynamic conductance => less sensible heat")
    ap.add_argument("--select-best-year", action="store_true",
                    help="run only the calendar year with the most observed flux "
                         "steps (contiguous, spans both seasons) — ~12-24x faster "
                         "than the full multi-decade record")
    ap.add_argument("--plant-wilting-point", type=float, default=None,
                    help="PLANT wilting point [m3/m3] for root-zone transpiration, "
                         "SEPARATE from the soil wilting point; below the soil "
                         "value = deep-rooted extraction (e.g. phreatophytes)")
    ap.add_argument("--canopy-interception", dest="interception",
                    action="store_true", default=False,
                    help="enable the shared canopy-water interception scheme "
                         "(two-leaf path): rain is intercepted, drips as "
                         "throughfall, and the wet leaf evaporates from the store")
    ap.add_argument("--spinup-steps", type=int, default=0,
                    help="prognostic soil spin-up: run this many steps before the "
                         "evaluation window (trimmed from scoring) so the soil "
                         "column reaches a realistic state. Needed at seasonally-"
                         "dry sites (e.g. US-Ton) where a mid-summer cold start "
                         "puts the root zone below wilting and zeros GPP")
    args = ap.parse_args()
    for nc in args.driver_nc:
        run_site(nc, args.mode, args.out, args.chunk,
                 max_steps=args.max_steps, start_step=args.start_step,
                 soil=args.soil, bottom_bc=args.bottom_bc,
                 soil_depth_m=args.soil_depth_m, nudge_tau_days=args.nudge_tau_days,
                 k_sat_decay_m=args.k_sat_decay_m,
                 soil_evap_resistance_exp=args.soil_evap_resistance_exp,
                 root_depth=args.root_depth, z_ref=args.z_ref,
                 stress_b0=args.stress_b0, select_best_year=args.select_best_year,
                 texture_csv=args.texture_csv,
                 canopy=args.canopy, clm_pft=args.clm_pft,
                 clmml_sai=args.clmml_sai, u_min=args.u_min,
                 stomatal_m_scale=args.stomatal_m_scale,
                 vcmax_scale=args.vcmax_scale,
                 vcmax_c3_scale=args.vcmax_c3_scale,
                 vcmax_c4_scale=args.vcmax_c4_scale,
                 stomatal_m_c3_scale=args.stomatal_m_c3_scale,
                 stomatal_m_c4_scale=args.stomatal_m_c4_scale,
                 mosaic=args.mosaic, tree_frac=args.tree_frac,
                 savanna_grass_pft=args.savanna_grass_pft,
                 savanna_grass_root_m=args.savanna_grass_root_m,
                 savanna_grass_fc4=args.savanna_grass_fc4,
                 clmml_turbulence=args.clmml_turbulence,
                 clmml_stomatal=args.clmml_stomatal,
                 clmml_vcmax25=args.clmml_vcmax25,
                 plant_wilting_point=args.plant_wilting_point,
                 spinup_steps=args.spinup_steps,
                 interception=args.interception)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

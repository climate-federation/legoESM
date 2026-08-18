"""Differentiable calibration of the MULTILAYER (8-layer Richards) land vs ERA5.

Companion to the slab calibrator (``scripts/run/train_land_params_era5.py``): the
same bounded, physical per-PFT + snow/ice parameter set, but the forward run is the
8-layer soil-thermal + Richards soil-moisture column (``step_multilayer_land``) with
the CLM per-column van-Genuchten soil.

The deep soil column needs years-to-decades of spin-up to equilibrate, which an
offline AD calibration cannot afford.  The forward is therefore restructured around a
SOIL-EQUILIBRIUM TRICK (the "annual value across the whole soil layer"):

  STAGE A  Equilibrate the column under CONSTANT ANNUAL-MEAN forcing.  Constant
           boundary conditions have a true fixed point reached in ~the column's
           thermal diffusion time (months, not years), so the deep soil equilibrates
           cheaply and the training gradients stop being dominated by un-equilibrated
           drift.  Every layer is initialised at the ERA5 ANNUAL-MEAN SKIN T.

  freeze   The annual-mean SOIL MOISTURE is then frozen for the seasonal pass.  The
           offline forcing (representative days, 4 synoptic hours) is too coarse to
           resolve the wet-tropics water balance — thin sandy CLM cells drain to
           wilting in a few steps, transpiration collapses, and the surface runs away
           to ~48 C, although the real wet tropics are ENERGY-limited (perennially
           wet).  Prescribing the annual-mean theta keeps evapotranspiration supply
           realistic per cell (deserts stay dry, rainforest stays wet) while the
           surface-ENERGY parameters are calibrated; the seasonal THERMAL cycle still
           evolves freely on top.

  STAGE B  Two seasonal years with subdaily forcing.  The first (unscored) settles the
           near-surface seasonal wave from the annual-mean equilibrium; only the
           second is scored, so the monthly means are free of first-cycle transients.
           Soil moisture is FROZEN at the Stage-A annual equilibrium (_FREEZE_FROM=0):
           evolving it under the under-resolved offline forcing desiccates the top
           layer -> ET collapse -> runaway.

The DEFAULT surface exchange is MOST (``--bulk most``), matching the coupled diurnal
land default — so the roughness z0 is calibrated (the key diurnal-coupling param; the
2 m T/q vs 10 m wind height split that hurt earlier MOST runs turned out NOT to help,
default z_ref=10 fits best).  Trainable per-PFT (17): albedo, emissivity, root depth,
roughness z0, soil thermal inertia (C_soil + k_solid -> seasonal-cycle amplitude/phase),
and the PLANT water-stress thresholds theta_wp/theta_fc (CLM btran, DISTINCT from the
soil van-Genuchten retention).  Ch has no gradient path under MOST (used only under
``--bulk constant``) and is frozen out of the trainable set.  The Farquhar
photosynthesis params (Vc_max25/g1/LCMA) are ON by default (2026-08-17): the loss is
the ERA5 DUAL TARGET (latent heat + skin temperature, shared with the slab
calibrator), and the latent-heat term directly constrains the canopy-conductance
chain that skin T alone could not (--no-stomata restores the legacy energy-only
mode).

STRICT RULE: no inert parameters, ever.  Mode-inactive params are frozen out of the
trainable pytree (``_inactive_keys``), and the first training step asserts every
remaining leaf carries loss gradient (``assert_no_inert``) — a zero-gradient
parameter aborts the run instead of silently pretending to be calibrated.

Result (ERA5 skin T, land, MOST default + 24-h forcing, full-grid): RMSE 3.15 K, bias
+0.42 K, seasonal-amplitude bias -0.73 K; z0 physically structured (forests ~1-2 m,
grass/crop/bare ~0.03-0.16 m).

Writes the recommended tuned parameters to ``results/land_tuned_multilayer.json`` —
it does NOT mutate production defaults.  The values baked as the multilayer default
live in ``legoesm.land.clm_surface_map`` (``*_MULTILAYER`` constants).

Run: PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/run/train_multilayer_land_era5.py
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.surface_albedo import LandAlbedoConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.stomata import StomataConfig
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.snow_bands import ElevationSnowBandConfig, band_elevation_anomalies
from legoesm.land.surface_params import LandSurfaceParams

# Reuse the slab calibrator's constrained-parameter machinery (bounds, sigmoid
# constrain/inverse, raw-param init, the CLM5 PFT table and its column index map) so
# the two calibrators share one bounded parameter definition.
import scripts.run.train_land_params_era5 as S

_NH = 4              # synoptic hours (0,6,12,18 UTC) -> resolve the diurnal cycle
_DAYS = 12           # representative days per month (keeps the AD scan tractable)
_DT = (24 / _NH) * 3600.0   # 6 h
_SPM = _DAYS * _NH   # steps per month
_EQ_STEPS = 600      # ~150 days of annual-mean spin -> deep-soil equilibrium
_N_LAYERS = 8
_SOIL_DEPTH_M = 3.0
_SOIL_GROWTH = 1.5
_FREEZE_FROM = 0     # pin ALL soil-moisture layers at the Stage-A annual equilibrium.
                     # The under-resolved offline forcing desiccates the top layer
                     # (bare-soil evaporation) -> ET collapse -> runaway when moisture
                     # evolves; the per-cell frozen equilibrium still spans the stress
                     # band (dry deserts / wet tropics), so the plant water-stress
                     # thresholds (theta_wp/theta_fc) remain trainable on real data.
_N_PFT = S._N_PFT
_PI = S._PI
_TBL = np.asarray(S._TABLE)              # (17,12) CLM5 init values per column
# Surface-exchange / photosynthesis mode (set by the CLI).  MOST surface exchange
# (matches the coupled DIURNAL default -> z0 trainable, RMSE ~3.1 vs constant-Ch 3.3).
_BULK_SCHEME = "most"
# Farquhar stomata ON by default (user directive 2026-08-17): with the ERA5
# latent-heat target in the loss the canopy conductance chain (Vc_max25/g1/LCMA
# -> stomatal resistance -> transpiration) is directly constrained, so the
# photosynthesis params are UNFROZEN.  --no-stomata restores the legacy
# energy-only mode (which would leave them inert — and the inert-parameter gate
# then requires them out of the trainable set, handled by _inactive_keys).
_STOMATA_ON = True
# Sub-grid elevation-band snow (legoesm.land.snow_bands): banded precip phase /
# melt / permanent snow from the CLM STD_ELEV map.  Off by default (legacy
# cell-mean snowpack); enable with --elev-bands.
_ELEV_BANDS_ON = False
# Loss weights (CLI-tunable).  lam_amp raised from 0.5 -> 1.5: the residual is
# dominated by the seasonal-cycle AMPLITUDE (mid-lats under, Antarctica over), and the
# per-PFT thermal inertia has head-room the low weight wasn't exploiting.
_LAM_ALB = 300.0
_LAM_PFT = 2.0
_LAM_AMP = 1.5
# Soil-moisture loss weight.  theta is O(0.1-0.5) m3/m3 vs T's O(10-50) K, so a unit
# MSE on theta is ~1e4x smaller; the weight lifts the (annual-mean, 0-28cm root-zone)
# soil-moisture term to a non-negligible fraction of the temperature term (smse~1e-2 *
# 3e3 ~ 30 vs tmse~O(100-1000)) so it actually constrains the porosity scale + plant
# water-stress thresholds without swamping the skin-T fit.  Tune via --lam-sm.
_LAM_SM = 3.0e3
# Global skin-T BIAS penalty (area-weighted mean of T_model - T_era, squared).  The
# tmse term penalises the space-time RMSE, which a structured warm bias can survive; this
# term drives the GLOBAL-MEAN skin-T bias toward 0 without distorting the spatial fit (the
# RMSE term still holds the pattern).  Default 0 = OFF (no change to existing behaviour);
# raise via --lam-tbias to explicitly target a low global skin-T bias.
_LAM_TBIAS = 0.0
# Latent-heat loss weight (ERA5 dual target: evaporation + skin T).  LE MSE is
# O(10^2-10^3) (W/m2)^2 vs skin-T MSE O(10) K^2; 0.02 puts the two on the same
# footing.  This is the term that constrains the unfrozen canopy-conductance
# chain (Vc_max25/g1/LCMA) and the plant water-stress thresholds.
_LAM_LE = 0.02
# MONTHLY-mean bias penalties (user 2026-08-17: "RMSE reduction is challenging —
# natural variability. Reducing mean bias, including monthly, is a MUST").
# Per-month area-weighted mean error, squared, averaged over the 12 months —
# targets the seasonal-cycle bias directly while the RMSE terms hold the
# spatial pattern.  Monthly T bias ~1.5 K and weight 30 puts the term at ~70
# vs tmse ~6; monthly LE bias ~10 W/m2 and 0.3 puts it at ~30.
_LAM_TBIAS_MON = 30.0
_LAM_LEBIAS_MON = 0.3
# Per-cell gradient-norm cap for the pre-train pathological-cell filter.  Healthy land
# cells have a per-cell |grad| ~ 1e1-1e3 (logged p90 ~ 3e3); a near-singular stiff-clay/
# saturated cell whose MOST flux backward is approaching the overflow reads 1e5-1e41.
# 1e6 sits in the empty gap between the two populations (see the logged distribution) —
# dropping the whole near-singular tail (not just the outright-NaN cells) keeps the
# CLIPPED gradient DIRECTION clean, so Adam does not overshoot on the few cells whose
# huge-but-finite gradient would otherwise dominate the global-norm clip.  See
# _drop_nan_grad_cells.
_GRAD_NORM_CAP = 1.0e6


# --------------------------------------------------------------------------- #
# EXTENDED per-PFT parameter set (multilayer-specific)                        #
# --------------------------------------------------------------------------- #
# Beyond the slab calibrator's albedo/emissivity/root, the multilayer forward
# calibrates (all per-PFT, 17):
#   pft_z0                     surface-exchange roughness (active because the bulk
#                              scheme is MOST; a constant Ch would give z0 no gradient)
#   pft_vcmax/pft_g1/pft_lcma  photosynthesis / stomatal conductance (active via the
#                              Farquhar carbon path with a prescribed LAI = C_fol/LCMA)
#   pft_wp/pft_fcgap           PLANT water-stress thresholds (CLM-style btran; theta_fc
#                              = wp + gap so the range never inverts), DISTINCT from the
#                              soil van-Genuchten retention that drives Richards drainage
#   pft_csoil/pft_ksolid       soil thermal inertia -> seasonal-cycle amplitude / phase
#                              (PFT-weighted SoilThermalConfig heat capacity + solid
#                              conductivity, per-cell (ncol,1) over the layers)
# Soil moisture EVOLVES (Richards) through the seasonal cycle so the water-stress
# response is seasonal.  The slab lp.C_soil/d_soil are unused by the multilayer thermal
# solver (the per-layer SoilThermalConfig replaces them); W_max and a constant Ch are
# likewise dropped — none carries a gradient in the MOST/Richards/Farquhar forward.
# Widen the per-PFT snow-free albedo ceiling for the BRIGHT surfaces the first tune
# could not reach: bare soil / deserts (ERA5 Sahara ~0.37 vs the old 0.40 HI) and the
# sparse/short PFTs.  Index 0 is bare_soil; 10-16 are the grass/crop/shrub columns.
_ML_PFT_ALB_HI = np.asarray(S._PFT_ALB_HI, dtype=float).copy()
_ML_PFT_ALB_HI[0] = 0.50                       # bare soil / desert
_ML_PFT_ALB_HI[[10, 11, 12, 13, 14, 15]] = np.maximum(
    _ML_PFT_ALB_HI[[10, 11, 12, 13, 14, 15]], 0.35)   # grass/crop/shrub headroom

BOUNDS_EXT = dict(
    pft_alb=(S._PFT_ALB_LO, _ML_PFT_ALB_HI), pft_emis=(0.94, 0.99),
    pft_root=(S._PFT_ROOT_LO, S._PFT_ROOT_HI),
    pft_ch=(2.0e-3, 6.0e-3),                           # per-PFT bulk exch (constant bulk)
    pft_z0=(5e-3, 3.0),                                # roughness (MOST); forests saturated 2.0
    pft_vcmax=(0.0, 80.0), pft_lcma=(20.0, 90.0), pft_g1=(1.0, 12.0),
    pft_wp=(0.05, 0.30), pft_fcgap=(0.03, 0.30),       # theta_fc_plant = wp + gap
    # per-PFT SCALE on the per-cell texture-derived van-Genuchten POROSITY (theta_sat):
    # texture sets the spatial pattern, the scale sets the per-PFT magnitude so the
    # equilibrium root-zone soil moisture can be pulled toward ERA5 (the soil-moisture
    # target is what makes the retention trainable — skin T alone could not constrain it).
    pft_smscale=(0.7, 1.3),
    # per-PFT snow-cover masking scale (CLM-style canopy snow burial): forests hide
    # ground snow (<1), open tundra/grass may whiten faster than the global
    # snow_depth_crit implies (>1); 1.0 = legacy full Niu-Yang cover.  The NH>55
    # albedo mosaic (forests +0.05 too bright, tundra/grass -0.05..-0.09 too dark)
    # is exactly the signature this closes.
    pft_snowmask=(0.2, 1.2),
    # per-PFT SCALE on the per-cell texture-derived soil thermal k_solid / C_solid
    # (texture sets the spatial pattern; the scale sets the per-PFT magnitude)
    pft_kscale=(0.1, 1.5), pft_cscale=(0.3, 2.0),
    th_glacier_cboost=(1.0, 15.0),                     # deep-ice inertia boost (glacier)
    glac_alb=(0.55, 0.90),          # snow-free ice-sheet base (raised: ERA5 Antarctica ~0.85)
    # Snow albedo (global scalars, blended over the snow-cover fraction f_snow =
    # tanh(snow_depth/snow_depth_crit), Niu-Yang 2007).  The offline model was too DARK
    # over snowy high-lat / Tibet / Antarctica cells (large negative albedo bias): the
    # OLD linear cover form sat perennial-snow cells at f_snow~0.5.  With the saturating
    # tanh form a SMALL snow_dcrit (SWE half-cover scale) makes a thin pack cover the
    # cell; snow_min lifts the aged-snow floor, snow_tau_days slows the age decay.
    snow_max=(0.60, 0.92),          # fresh-snow albedo (fresh snow ~0.85-0.9)
    snow_min=(0.45, 0.75),          # aged/melting-snow albedo floor (raised: was too dark)
    snow_dcrit=(3.0, 40.0),         # SWE [kg/m2] half-cover scale (tanh; LOWER=brighter)
    snow_tau_days=(1.0, 20.0),      # snow-albedo age e-folding [days]
    soil_dry_boost=(0.0, 0.16),     # CLM dry-soil albedo brightening (deserts); 0=off
    soil_alb_scale=(0.6, 1.5),      # scale on the per-cell CLM soil-colour bare-soil albedo
    # --- High-elevation snow/ice closures (gaps 1-4; active with --elev-bands) ---
    elev_lapse=(4.5e-3, 8.0e-3),    # band T-downscaling lapse rate [K/m]
    elev_sw_grad=(0.0, 1.2e-4),     # SW-down elevation gradient [1/m] (thinner air aloft)
    elev_lw_lapse=(0.0, 6.0e-2),    # LW-down elevation lapse [W/m2/m] (colder air aloft)
    glac_ice_alb=(0.15, 0.45),      # exposed ablation-zone glacier-ice albedo (dark ice)
    snow_zenith=(0.0, 0.6))         # BATS solar-zenith snow brightening weight (gap 3)


def constrain_ext(p: dict) -> dict:
    return {k: BOUNDS_EXT[k][0] + (BOUNDS_EXT[k][1] - BOUNDS_EXT[k][0]) * jax.nn.sigmoid(v)
            for k, v in p.items()}


def _inv_ext(v, k):
    lo, hi = BOUNDS_EXT[k]
    v = np.clip(v, np.asarray(lo) + 1e-6, np.asarray(hi) - 1e-6)
    return np.log((v - lo) / (hi - v))


def init_ext_params() -> dict:
    """Raw (unconstrained) params initialised at the CLM5 table defaults (clamped into
    the extended physical bounds)."""
    col = lambda name: _TBL[:, _PI[name]]
    full = lambda v: np.full(_N_PFT, v)
    fc0 = np.clip(col("theta_fc"), 0.12, 0.42); wp0 = np.clip(col("theta_wp"), 0.05, 0.29)
    return {k: jnp.asarray(a) for k, a in dict(
        pft_alb=_inv_ext(col("albedo_veg"), "pft_alb"),
        pft_emis=_inv_ext(full(0.96), "pft_emis"),
        pft_root=_inv_ext(np.clip(col("root_depth"), S._PFT_ROOT_LO + 1e-3,
                                  S._PFT_ROOT_HI - 1e-3), "pft_root"),
        pft_ch=_inv_ext(full(3.0e-3), "pft_ch"),
        pft_z0=_inv_ext(np.clip(col("z0"), 5e-3 + 1e-4, 2.0 - 1e-3), "pft_z0"),
        pft_vcmax=_inv_ext(col("Vc_max25"), "pft_vcmax"),
        pft_lcma=_inv_ext(col("LCMA"), "pft_lcma"),
        pft_g1=_inv_ext(col("g1"), "pft_g1"),
        pft_wp=_inv_ext(wp0, "pft_wp"),
        pft_fcgap=_inv_ext(np.clip(fc0 - wp0, 0.04, 0.29), "pft_fcgap"),
        pft_smscale=_inv_ext(full(1.0), "pft_smscale"),    # start at texture porosity
        pft_snowmask=_inv_ext(full(1.0), "pft_snowmask"),  # start at legacy full cover
        # init scales so texture*scale ~ the previous effective inertia (texture
        # k_base~5 -> k_scale~0.35 gives k_solid~1.8; C_base~2.3e6 -> c_scale~0.9)
        pft_kscale=_inv_ext(full(0.35), "pft_kscale"),
        pft_cscale=_inv_ext(full(0.9), "pft_cscale"),
        th_glacier_cboost=jnp.asarray(_inv_ext(5.0, "th_glacier_cboost")),
        glac_alb=_inv_ext(0.70, "glac_alb"), snow_max=_inv_ext(0.82, "snow_max"),
        snow_min=_inv_ext(0.55, "snow_min"),          # aged-snow floor
        snow_dcrit=_inv_ext(15.0, "snow_dcrit"),      # tanh SWE half-cover scale [kg/m2]
        snow_tau_days=_inv_ext(5.0, "snow_tau_days"), # snow-albedo age e-folding [days]
        soil_dry_boost=_inv_ext(0.11, "soil_dry_boost"),  # CLM dry-soil brightening
        soil_alb_scale=_inv_ext(1.0, "soil_alb_scale"),   # start at the raw CLM soil colour
        # High-elevation snow/ice closures (physical defaults; refined by the re-tune)
        elev_lapse=_inv_ext(6.0e-3, "elev_lapse"),        # CESM/CLM glacier MEC lapse
        elev_sw_grad=_inv_ext(3.0e-5, "elev_sw_grad"),    # ~3 %/km clear-sky SW gradient
        elev_lw_lapse=_inv_ext(2.9e-2, "elev_lw_lapse"),  # ~29 W/m2/km LW-down lapse
        glac_ice_alb=_inv_ext(0.30, "glac_ice_alb"),      # CLM exposed glacier-ice albedo
        snow_zenith=_inv_ext(0.2, "snow_zenith"),         # BATS zenith brightening weight
    ).items()}


def baked_init_params() -> dict:
    """Raw (unconstrained) params WARM-STARTED at the production baked multilayer
    defaults (the ERA5-calibrated ``_TUNED_*_MULTILAYER`` bake in ``clm_surface_map``).

    Refining these — instead of climbing from the CLM5 prior — lets the re-tune keep the
    well-tuned production skill and only adjust for the elevation-band snow now in the
    forward.  The Farquhar/smscale keys (not part of the offline bake) stay at CLM5.
    """
    from legoesm.land import clm_surface_map as C
    p = dict(init_ext_params())
    wp = np.asarray(C._TUNED_PFT_WP_MULTILAYER); fc = np.asarray(C._TUNED_PFT_FC_MULTILAYER)
    over = dict(
        pft_alb=_inv_ext(np.asarray(C._TUNED_PFT_ALBEDO_MULTILAYER), "pft_alb"),
        pft_emis=_inv_ext(np.asarray(C._TUNED_PFT_EMISSIVITY_MULTILAYER), "pft_emis"),
        pft_root=_inv_ext(np.asarray(C._TUNED_PFT_ROOT_DEPTH_MULTILAYER), "pft_root"),
        pft_z0=_inv_ext(np.asarray(C._TUNED_PFT_Z0_MULTILAYER), "pft_z0"),
        pft_ch=_inv_ext(np.asarray(C._TUNED_PFT_CH_MULTILAYER), "pft_ch"),
        pft_kscale=_inv_ext(np.asarray(C._TUNED_PFT_KSCALE_MULTILAYER), "pft_kscale"),
        pft_cscale=_inv_ext(np.asarray(C._TUNED_PFT_CSCALE_MULTILAYER), "pft_cscale"),
        pft_wp=_inv_ext(wp, "pft_wp"),
        pft_fcgap=_inv_ext(np.clip(fc - wp, 0.04, 0.29), "pft_fcgap"),
        glac_alb=_inv_ext(C.TUNED_GLACIER_ALBEDO_MULTILAYER, "glac_alb"),
        snow_max=_inv_ext(C.TUNED_SNOW_ALBEDO_MAX_MULTILAYER, "snow_max"),
        snow_min=_inv_ext(C.TUNED_SNOW_ALBEDO_MIN_MULTILAYER, "snow_min"),
        snow_dcrit=_inv_ext(C.TUNED_SNOW_DCRIT_MULTILAYER, "snow_dcrit"),
        snow_tau_days=_inv_ext(C.TUNED_SNOW_TAU_DAYS_MULTILAYER, "snow_tau_days"),
        soil_dry_boost=_inv_ext(C.TUNED_SOIL_DRY_BOOST_MULTILAYER, "soil_dry_boost"),
        soil_alb_scale=_inv_ext(C.TUNED_SOIL_ALB_SCALE_MULTILAYER, "soil_alb_scale"),
        th_glacier_cboost=_inv_ext(C.TUNED_GLACIER_CBOOST_MULTILAYER, "th_glacier_cboost"),
        # v6 Farquhar bake (dual-target LE calibration)
        pft_vcmax=_inv_ext(np.asarray(C._TUNED_PFT_VCMAX_MULTILAYER), "pft_vcmax"),
        pft_g1=_inv_ext(np.asarray(C._TUNED_PFT_G1_MULTILAYER), "pft_g1"),
        pft_lcma=_inv_ext(np.asarray(C._TUNED_PFT_LCMA_MULTILAYER), "pft_lcma"),
    )
    p.update({k: jnp.asarray(v) for k, v in over.items()})
    return p


def json_init_params(path: str) -> dict:
    """Raw (unconstrained) params warm-started from a saved CONSTRAINED tuned JSON
    (the inverse of the ``constrain_ext`` applied at save time), layered over the baked
    raw defaults so any key absent from an older checkpoint keeps its baked value.

    Lets a re-tune REFINE an existing tuned checkpoint (e.g. ``land_tuned_allgaps.json``,
    the current production params) instead of restarting from the baked/CLM5 prior — the
    correct warm start when the point of the re-tune is to beat the CURRENT best.
    """
    p = baked_init_params()
    with open(path) as f:
        cp = json.load(f)
    for k, v in cp.items():
        if k in BOUNDS_EXT:
            p[k] = jnp.asarray(_inv_ext(np.asarray(v, dtype=float), k))
    return p


# --------------------------------------------------------------------------- #
# Pure (testable, differentiable) calibration core                            #
# --------------------------------------------------------------------------- #
def _ml_land_params(cp, data):
    """Per-column LandSurfaceParams from the constrained EXTENDED per-PFT params.
    theta_wp/theta_fc are the PLANT (btran) stress thresholds (per-PFT, trainable),
    NOT the soil retention; W_max is unused by the Richards forward (kept as the CLM5
    value for completeness)."""
    pw = lambda k: data["pft"] @ cp[k]                  # PFT-weighted per cell
    # Bare-soil (PFT 0) uses the per-cell CLM soil-COLOUR albedo (spatially-varying
    # desert/soil pattern) x a trainable scale, instead of one global bare-soil value;
    # vegetated PFTs keep their per-PFT albedo.  This is what gives our model CLM's
    # bright-desert skill (CLM's soil colour is calibrated to observed/MODIS albedo).
    bare = data["pft"][:, 0]
    veg_alb = pw("pft_alb") - bare * cp["pft_alb"][0]
    alb = (1 - data["fg"]) * (veg_alb + bare * data["soil_albedo"] * cp["soil_alb_scale"]) \
        + data["fg"] * cp["glac_alb"]
    wp = pw("pft_wp"); fc = wp + pw("pft_fcgap")
    n = data["lat"].shape[0]
    cst = lambda name: jnp.full(n, float(_TBL[:, _PI[name]].mean()))  # unused slab cols
    return LandSurfaceParams(
        albedo_veg=alb, emissivity=pw("pft_emis"), z0=pw("pft_z0"),
        W_max=cst("W_max"), C_soil=cst("C_soil"), d_soil=cst("d_soil"),
        root_depth=pw("pft_root"), theta_wp=wp, theta_fc=fc, Vc_max25=pw("pft_vcmax"),
        LCMA=pw("pft_lcma"), g1=pw("pft_g1"))


def build_multilayer_cfg(cp, data):
    """Assemble (config, land_params, hydraulics, carbon_state) for the multilayer
    forward from the constrained extended params + the per-cell CLM soil map.  Surface
    exchange + stomata follow the module mode (_BULK_SCHEME / _STOMATA_ON): the default
    is a constant per-PFT Ch with soil-only beta; --bulk most / --stomata switch on the
    MOST (z0) and Farquhar (Vc_max25/g1/LCMA) paths.  Soil thermal inertia is per-PFT."""
    col = lambda k: data["vg_" + k].reshape(-1, 1)
    # Per-PFT POROSITY scale on the texture theta_sat (trainable via the soil-moisture
    # target).  Clamp theta_sat above BOTH theta_r and the PLANT field capacity (wp+gap):
    # the van-Genuchten retention needs theta_sat > theta_r, and btran (theta-wp)/(fc-wp)
    # needs the column able to saturate above field capacity — without the fc floor a
    # scale-down (smscale<1, sandy cells) could push porosity below fc and break btran /
    # drive an inconsistent hydraulic state -> Richards NaN (codex).
    sm_scale = (data["pft"] @ cp["pft_smscale"]).reshape(-1, 1)
    plant_fc = (data["pft"] @ (cp["pft_wp"] + cp["pft_fcgap"])).reshape(-1, 1)
    theta_sat_s = jnp.maximum(col("theta_sat") * sm_scale,
                              jnp.maximum(col("theta_r") + 0.05, plant_fc + 0.02))
    hyd = SoilHydraulicsConfig(theta_r=col("theta_r"), theta_sat=theta_sat_s,
                               alpha_vg=col("alpha_vg"), n_vg=col("n_vg"), K_sat=col("K_sat"))
    # Soil thermal inertia: per-cell TEXTURE (sand/clay) x per-PFT scale, blended
    # toward ICE on glacier cells (deep-ice inertia boost).  Reuse the SHARED
    # definition so the calibrator and the bake never diverge.
    from legoesm.land.clm_surface_map import multilayer_thermal_arrays
    k_eff, c_eff = multilayer_thermal_arrays(
        data["pft"], data["pct_sand"], data["pct_clay"], data["fg"],
        cp["pft_kscale"], cp["pft_cscale"], cp["th_glacier_cboost"])
    thermal = SoilThermalConfig(k_solid=k_eff.reshape(-1, 1), C_soil=c_eff.reshape(-1, 1))
    ch = data["pft"] @ cp["pft_ch"]              # per-PFT bulk exchange coefficient
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=_N_LAYERS, total_depth=_SOIL_DEPTH_M,
                                 growth_factor=_SOIL_GROWTH),
        hydraulics=hyd, thermal=thermal,
        bulk_scheme=_BULK_SCHEME, Ch_land=ch, Cd_land=ch,  # MOST uses z0; constant uses ch
        stomata=StomataConfig(enabled=_STOMATA_ON),  # Farquhar -> Vc_max25/g1/LCMA active
        carbon=CarbonConfig(scheme="differland" if _STOMATA_ON else "none"),
        snow_albedo_feedback=True,
        # Sub-grid elevation-band snow: banded precip phase / melt gating /
        # permanent snow from the CLM STD_ELEV sub-grid topography (module toggle).
        elev_bands=(ElevationSnowBandConfig(
            band_dz=band_elevation_anomalies(data["std_elev"]),
            lapse_rate_K_m=cp["elev_lapse"],
            sw_elev_grad_per_m=cp["elev_sw_grad"],
            lw_elev_lapse_W_m2_per_m=cp["elev_lw_lapse"],
            alpha_glacier_ice=cp["glac_ice_alb"])
            if _ELEV_BANDS_ON else None),
        land_albedo=LandAlbedoConfig(
            snow_cover_scale=data["pft"] @ cp["pft_snowmask"],  # canopy snow masking
            alpha_snow_max=cp["snow_max"], alpha_snow_min=cp["snow_min"],
            snow_depth_crit=cp["snow_dcrit"],
            tau_snow_decay=cp["snow_tau_days"] * 86400.0,    # days -> seconds
            snow_zenith_factor=cp["snow_zenith"],            # BATS zenith brightening (gap 3)
            soil_dry_albedo_boost=cp["soil_dry_boost"]))     # CLM dry-soil brightening
    # Carbon state is PRESCRIBED (fixed climatological leaf carbon -> fixed LAI), the
    # same decoupling as the soil-moisture trick: it activates the photosynthesis /
    # stomatal-conductance parameters (Vc_max25, g1, LCMA via LAI = C_fol/LCMA) so they
    # are trainable, without paying for a multi-decade carbon-pool spin-up.
    cs0 = init_carbon_state((data["lat"].shape[0],), cfg.carbon)
    return cfg, _ml_land_params(cp, data), hyd, cs0


def forward_ml(cp, data):
    """Monthly-mean T_sfc + surface albedo (12, ncol) under constrained params."""
    n = data["lat"].shape[0]
    cfg, lp, hyd, cs0 = build_multilayer_cfg(cp, data)
    st0 = init_multilayer_land_state(n, cfg, T_init=280.0)
    st0 = jax.tree.map(lambda x: x.astype(jnp.float64), st0)
    # Init the whole column: thermal at the ERA5 annual-mean skin T, moisture at the
    # per-cell field capacity (the library default seeds a uniform theta=0.209, which
    # is below the wilting point of clay/tropical soils -> ET shuts off from step 0).
    theta0 = jnp.clip(data["fc"].astype(jnp.float64),
                      hyd.theta_r[:, 0] + 1e-3, hyd.theta_sat[:, 0] - 1e-3)
    theta_col = jnp.broadcast_to(theta0[:, None], st0.theta_soil.shape)
    st0 = st0._replace(
        T_soil=jnp.broadcast_to(data["t0"].astype(jnp.float64)[:, None], st0.T_soil.shape),
        theta_soil=theta_col,
        psi_soil=psi_from_theta(theta_col, hyd))  # keep matric head consistent

    # Diurnal sampling (NH hours, DT timestep, SPM steps/month) is taken from the
    # module constants set by ``load_training_data`` (STATIC compile-time values, not
    # traced pytree leaves) so the same forward serves the 4-synoptic-hour and the
    # full 24-hour hourly forcing.
    nh, spm, dt = _NH, _SPM, _DT
    fstack = jax.tree.map(lambda *xs: jnp.stack(xs), *data["forc"])  # leaves (12, NH, ncol)
    nstep = 12 * spm

    f64 = lambda t: jax.tree.map(
        lambda x: x.astype(jnp.float64)
        if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x, t)

    def step(s, f):
        # carbon_state is held FIXED at cs0 (prescribed LAI) — the returned, evolved
        # carbon pools are discarded so no carbon spin-up is needed.  Force float64 on
        # the returned state: the MOST/Farquhar path can emit a float32 field, which
        # would break the lax.scan carry (input float64 != output float32).
        s2, r, _ = step_multilayer_land(s, f, cfg, 1.0, dt, lat=data["lat"], doy=15.0,
                                        land_params=lp, carbon_state=cs0)
        return f64(s2), r

    # --- STAGE A: equilibrate the column under CONSTANT annual-mean forcing -------
    f_ann = jax.tree.map(lambda x: x.mean((0, 1)), fstack)   # (ncol,) per field

    @jax.checkpoint
    def eq_body(s, _):
        s2, _ = step(s, f_ann)
        return s2, None
    st, _ = jax.lax.scan(eq_body, st0, None, length=_EQ_STEPS)

    # PARTIAL moisture freeze: pin the DEEP soil layers (the slow supply reservoir) at
    # their Stage-A annual equilibrium, let only the shallow ROOT-ZONE layers evolve.
    # Fully-evolving moisture under the under-resolved offline forcing drains the thin
    # sandy tropical cells to wilting -> ET collapse -> +50 C runaway (untrainable);
    # fully-frozen moisture is stable but kills the SEASONAL water-stress signal.  The
    # split keeps the deep root supply wet (energy-limited tropics stay supplied) while
    # the shallow layers dry down and recharge seasonally -> a real, trainable btran
    # cycle (a dry season warms the surface).
    deep = st.theta_soil[:, _FREEZE_FROM:]
    deep_psi = st.psi_soil[:, _FREEZE_FROM:]
    # Root-zone (0-28cm = ERA5 swvl1+2) soil moisture from the Stage-A EQUILIBRIUM,
    # saved BEFORE Stage B so it is the documented frozen-equilibrium value regardless
    # of the freeze span (moisture is pinned, so it does not drift in Stage B anyway).
    rz_w = data["rz_w"]
    W_sm = (st.theta_soil[:, :rz_w.shape[0]] * rz_w).sum(1) / rz_w.sum()

    # --- STAGE B: seasonal years with subdaily forcing -> monthly means ----------
    @jax.checkpoint     # remat per step -> bounded backward memory
    def body(carry, k):
        s, Tsum, Asum, Wsum, Lsum = carry
        month = k // spm; hour = k % nh
        f = jax.tree.map(lambda x: x[month, hour], fstack)
        s2, r = step(s, f)
        s2 = s2._replace(
            theta_soil=s2.theta_soil.at[:, _FREEZE_FROM:].set(deep),
            psi_soil=s2.psi_soil.at[:, _FREEZE_FROM:].set(deep_psi))  # pin deep reservoir
        # Albedo is INSOLATION-WEIGHTED (weight = sw_down): with the real diurnal sun
        # (gap 3) the zenith brightening makes a night albedo spuriously bright, but
        # night carries ~0 shortwave, so weighting by sw_down gives the physically
        # meaningful effective (SW-budget) albedo and excludes the dark hours.
        Tsum = Tsum.at[month].add(r.T_sfc)
        # Clip the SW weight >= 0: ERA5 ssrd can carry small negative accumulation
        # artefacts, which would otherwise divide a negative Asum by the 1e-6 Wsum floor
        # into a huge negative albedo.  Matches the ssrd>=0 clip on the target.
        _sw_w = jnp.maximum(f.sw_down, 0.0)
        Asum = Asum.at[month].add(r.albedo * _sw_w)
        Wsum = Wsum.at[month].add(_sw_w)
        Lsum = Lsum.at[month].add(r.lhflx)
        return (s2, Tsum, Asum, Wsum, Lsum), None

    # First seasonal pass is an unscored spin; only the second year is scored so the
    # monthly means are free of first-cycle thermal/snow transients.
    z = lambda: jnp.zeros((12, n), dtype=st.T_soil.dtype)
    (st, _, _, _, _), _ = jax.lax.scan(body, (st, z(), z(), z(), z()), jnp.arange(nstep))
    (st, Tsum, Asum, Wsum, Lsum), _ = jax.lax.scan(
        body, (st, z(), z(), z(), z()), jnp.arange(nstep))
    return Tsum / spm, Asum / jnp.maximum(Wsum, 1e-6), W_sm, Lsum / spm


def loss_ml(p, data, lam_alb=None, lam_pft=None, lam_amp=None, lam_sm=None, lam_tbias=None,
            lam_le=None, lam_tbias_mon=None, lam_lebias_mon=None):
    # weights default to the module globals (CLI-tunable) so the jitted
    # value_and_grad picks up an updated lam_amp without re-partialling.
    lam_alb = _LAM_ALB if lam_alb is None else lam_alb
    lam_pft = _LAM_PFT if lam_pft is None else lam_pft
    lam_amp = _LAM_AMP if lam_amp is None else lam_amp
    lam_sm = _LAM_SM if lam_sm is None else lam_sm
    lam_tbias = _LAM_TBIAS if lam_tbias is None else lam_tbias
    lam_le = _LAM_LE if lam_le is None else lam_le
    lam_tbias_mon = _LAM_TBIAS_MON if lam_tbias_mon is None else lam_tbias_mon
    lam_lebias_mon = _LAM_LEBIAS_MON if lam_lebias_mon is None else lam_lebias_mon
    cp = constrain_ext(p)
    T, A, W, L = forward_ml(cp, data)
    w = data["w"][None, :]
    tmse = jnp.sum(w * (T - data["skt"]) ** 2) / jnp.sum(w) / 12
    # Albedo MSE is INSOLATION-weighted (area w x monthly SW): a polar-night month-cell
    # (no sun -> the model's zenith albedo and the ERA5 target are both undefined) must
    # not enter the fit.  Falls back to uniform monthly weight if alb_wt is absent.
    # Broadcast to (12, ncol) so the numerator sum (over months x cells) matches the
    # denominator sum — else the uniform fallback under-counts the denominator 12x.
    _awt = (w * data["alb_wt"] if "alb_wt" in data
            else jnp.broadcast_to(w, A.shape))
    amse = jnp.sum(_awt * (A - data["alb"]) ** 2) / jnp.maximum(jnp.sum(_awt), 1e-12)
    # soil moisture: model root-zone equilibrium vs ERA5 annual-mean swvl (0-28cm).
    # FINITE-MASKED: a diverged forward (NaN W) or a missing target must not poison the
    # loss/gradient via a NaN that survives the lam_sm weight (codex: 0*NaN == NaN).
    # lam_sm == 0 is a STATIC (Python) flag, so the SM path is skipped entirely when off.
    if lam_sm > 0.0:
        sm_ok = jnp.isfinite(W) & jnp.isfinite(data["sm"])
        # Sanitise the INPUTS before the squared difference (not the output): a NaN that
        # reaches (W - sm)**2 poisons the reverse-mode VJP even inside a where() that
        # discards it (the standard JAX where-NaN-gradient gotcha) — replace masked
        # cells with 0 BEFORE the diff so no NaN ever enters the graph.
        W_s = jnp.where(sm_ok, W, 0.0)
        sm_s = jnp.where(sm_ok, data["sm"], 0.0)
        wsm = data["w"] * sm_ok
        smse = jnp.sum(wsm * (W_s - sm_s) ** 2) / (jnp.sum(wsm) + 1e-9)
    else:
        smse = jnp.zeros((), tmse.dtype)
    # Latent heat vs ERA5 slhf_wm2 (positive-up) — the second leg of the dual
    # target.  Same finite-mask + input-sanitise pattern as the SM term (a NaN
    # reaching the diff poisons the VJP even where()-discarded); lam_le == 0 is a
    # STATIC flag, so the LE path is skipped entirely when off.
    if lam_le > 0.0 or lam_lebias_mon > 0.0:
        le_ok = jnp.isfinite(L) & jnp.isfinite(data["le"])
        L_s = jnp.where(le_ok, L, 0.0)
        le_t = jnp.where(le_ok, data["le"], 0.0)
        wle = w * le_ok
        lemse = jnp.sum(wle * (L_s - le_t) ** 2) / (jnp.sum(wle) + 1e-9)
        # per-month area-weighted mean LE bias, squared, averaged over months
        mb_le = jnp.sum(wle * (L_s - le_t), axis=1) / (jnp.sum(wle, axis=1) + 1e-9)
        lebias_mon = jnp.mean(mb_le ** 2)
    else:
        lemse = jnp.zeros((), tmse.dtype)
        lebias_mon = jnp.zeros((), tmse.dtype)
    # per-month area-weighted mean skin-T bias (the seasonal-cycle bias target)
    mb_t = jnp.sum(w * (T - data["skt"]), axis=1) / jnp.sum(w)
    tbias_mon = jnp.mean(mb_t ** 2)
    ann = (T - data["skt"]).mean(0)
    oh = data["dom_onehot"] * data["w"][:, None]
    pb = (oh * ann[:, None]).sum(0) / (oh.sum(0) + 1e-9)
    present = (data["dom_onehot"].sum(0) > 0).astype(ann.dtype)
    ppft = jnp.sum(present * pb ** 2) / jnp.sum(present + 1e-9)
    # seasonal-amplitude term: penalise model monthly amplitude away from ERA5
    amp = (T.max(0) - T.min(0)) - (data["skt"].max(0) - data["skt"].min(0))
    samp = jnp.sum(data["w"] * amp ** 2) / jnp.sum(data["w"])
    # Global (area-weighted) skin-T bias [K].  ``ann`` is the per-cell annual bias above.
    gbias = jnp.sum(data["w"] * ann) / jnp.sum(data["w"])
    loss = (tmse + lam_alb * amse + lam_pft * ppft + lam_amp * samp + lam_sm * smse
            + lam_tbias * gbias ** 2 + lam_le * lemse
            + lam_tbias_mon * tbias_mon + lam_lebias_mon * lebias_mon)
    return loss, (tmse, amse, ppft, samp, smse, gbias, lemse, tbias_mon, lebias_mon)


def _params_dict(p):
    return {k: np.asarray(v).tolist() for k, v in constrain_ext(p).items()}


def _select_cells(data: dict, idx) -> dict:
    """Slice the training dict to a subset of land cells (``idx`` into the cell axis):
    forc/skt/alb carry the cell axis last, ``rz_w`` is soil-layer weights (no cell
    axis), everything else is ``(ncol, ...)`` on axis 0."""
    idx = np.asarray(idx)
    out = {}
    for k, v in data.items():
        if k == "forc":
            out[k] = [jax.tree.map(lambda x: x[:, idx], f) for f in v]
        elif k in ("skt", "alb", "alb_wt", "le"):   # (12, ncol) monthly targets/weights
            out[k] = v[:, idx]
        elif k == "rz_w":
            out[k] = v
        else:
            out[k] = v[idx]
    return out


def _drop_nan_grad_cells(data: dict, p: dict) -> np.ndarray:
    """One-time: keep only land cells whose PER-CELL loss gradient is finite.

    A few stiff-clay, near-saturated cells evolve over the seasonal Stage-B run into a
    regime where the implicit MOST surface-flux fixed-point iteration's reverse mode
    overflows f64 — a finite FORWARD but a NaN BACKWARD — and a single such cell poisons
    the whole batch gradient.  They are <1% of land and their per-PFT parameters are
    pinned by the many other cells sharing the PFT, so excluding them (LOGGED, not
    silently) leaves the calibration well-posed and faithful (the Richards+MOST scheme
    is unchanged; only pathological training samples are dropped).  The shared
    ``thomas_solve`` adjoint is already stable (custom_vjp); this covers the residual
    MOST-iteration cells without altering any production surface-flux gradient."""
    n = int(data["lat"].shape[0])
    gfn = jax.jit(lambda q, d: jax.grad(lambda qq: loss_ml(qq, d)[0])(q))
    norms = np.empty(n)
    for c in range(n):
        g = gfn(p, _select_cells(data, [c]))
        norms[c] = float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in g.values())))
    # A cell is "pathological" if its per-cell gradient is non-finite OR astronomically
    # large: both are the SAME near-singular MOST-iteration backward (a barely-finite
    # 1e17 norm is the same overflow one rounding step away from NaN).  Healthy cells
    # sit at ~1e2-1e5, so _GRAD_NORM_CAP cleanly separates them; the cap is logged with
    # the norm distribution so the cut is auditable, not a silent heuristic.
    bad = np.where(~np.isfinite(norms) | (norms > _GRAD_NORM_CAP))[0]
    keep = np.setdiff1d(np.arange(n), bad)
    if bad.size:
        fin = norms[np.isfinite(norms)]
        dom = np.asarray(data["dom_onehot"])[bad].argmax(1)
        nv = np.asarray(data["vg_n_vg"])[bad]
        print(f"# pre-filter: dropped {bad.size}/{n} pathological cells (NaN/overflow "
              f"MOST-iteration backward at stiff-clay/saturated state); dom-PFTs "
              f"{sorted(set(int(d) for d in dom))}, n_vg in [{nv.min():.3f},{nv.max():.3f}]; "
              f"per-cell |grad| p50={np.percentile(fin,50):.1e} p90={np.percentile(fin,90):.1e} "
              f"max={fin.max():.1e}, cap={_GRAD_NORM_CAP:.0e}", flush=True)
    return np.asarray(keep, dtype=int)


def _split_cells(data: dict, frac: float, seed: int):
    """Random train/test split of the sampled cells -> (train_data, test_data).  Uses
    seed+1 so the split is independent of the cell-sampling RNG (seed)."""
    n = int(data["lat"].shape[0])
    idx = np.random.default_rng(seed + 1).permutation(n)
    ntest = max(1, int(round(frac * n)))
    return _select_cells(data, idx[ntest:]), _select_cells(data, idx[:ntest])


def _metrics(cp: dict, data: dict):
    """Area-weighted skin-T / albedo RMSE + bias for a constrained param set."""
    T, A, _, _ = forward_ml(cp, data)
    w = data["w"]; sw = float(jnp.sum(w))
    tr = float(jnp.sqrt(jnp.sum(w[None] * (T - data["skt"]) ** 2) / sw / 12))
    tb = float(jnp.sum(w[None] * (T - data["skt"])) / sw / 12)
    ar = float(jnp.sqrt(jnp.sum(w[None] * (A - data["alb"]) ** 2) / sw / 12))
    ab = float(jnp.sum(w[None] * (A - data["alb"])) / sw / 12)
    return tr, tb, ar, ab


def _report_test(tuned: dict, test_data: dict):
    """Held-out generalisation: initial (CLM5) vs trained, on cells NEVER trained on."""
    ti, tbi, ai, abi = _metrics(constrain_ext(init_ext_params()), test_data)
    tc, tbc, ac, abc = _metrics({k: jnp.asarray(v) for k, v in tuned.items()}, test_data)
    print(f"# HELD-OUT TEST ({int(test_data['lat'].shape[0])} cells, never trained):",
          flush=True)
    print(f"#   skin-T RMSE {ti:.3f} -> {tc:.3f} K   bias {tbi:+.3f} -> {tbc:+.3f} K",
          flush=True)
    print(f"#   albedo RMSE {ai:.4f} -> {ac:.4f}   bias {abi:+.4f} -> {abc:+.4f}", flush=True)


def _inactive_keys() -> set:
    """Params with NO gradient path in the current mode — excluded from the trainable
    set (strict no-inert-parameters rule; everything left must carry gradient, gated
    by ``assert_no_inert`` on the first training step).  They are still baked into
    the config (frozen at init) so the forward and the saved JSON are complete."""
    ks = {"pft_ch"} if _BULK_SCHEME == "most" else {"pft_z0"}
    if not _STOMATA_ON:
        ks |= {"pft_vcmax", "pft_g1", "pft_lcma"}
    if not _ELEV_BANDS_ON:
        ks |= {"elev_lapse", "elev_sw_grad", "elev_lw_lapse", "glac_ice_alb"}
    # snow_zenith is inert in EVERY current mode — the gate caught it on the
    # first full-grid dual-target run: no compute_land_albedo call site (SEB,
    # multilayer post-step, snow bands) passes cos_zenith, so the BATS zenith
    # brightening never executes.  Frozen until that wiring exists (a production
    # albedo change, out of calibration scope).
    ks.add("snow_zenith")
    return ks


def train(data, n_iter=250, lr=3e-2, ckpt_path=None, ckpt_every=50, clip=1.0,
          prefilter=True, batch=0, init_params=None):
    p = init_ext_params() if init_params is None else init_params
    # The per-cell pre-filter is O(ncol) single-cell gradients — cheap at n_sub~200 but
    # the bottleneck at full-grid (~5.5k cells).  On the AD-stable 24-h forcing almost
    # no cell is pathological, so --no-prefilter skips it and leans on the per-step
    # gradient sanitisation + best-checkpoint instead.
    if prefilter:
        keep = _drop_nan_grad_cells(data, p)    # exclude NaN-backward cells (logged)
        if keep.size < int(data["lat"].shape[0]):
            data = _select_cells(data, keep)
    # Mini-batch SGD: a single full-grid (~5k-cell) gradient is dominated by the global
    # snow/soil scalars aggregated over every snowy cell AND is biased whenever the
    # per-step sanitiser zeros a transiently-singular cell, so it wanders instead of
    # descending.  Stepping on a small RANDOM cell batch each iter keeps every step in
    # the stable small-sample regime while still visiting EVERY cell across the run.
    ncol_tr = int(data["lat"].shape[0])
    use_batch = 0 < batch < ncol_tr
    _brng = np.random.default_rng(20260701)          # mini-batch sampler (fixed seed)
    # STRICT no-inert-parameters rule: mode-inactive keys (no gradient path) are
    # FROZEN at init and removed from the optimised pytree; the first computed
    # gradient then asserts every remaining leaf is live (S.assert_no_inert).
    frozen = {k: p[k] for k in _inactive_keys() if k in p}
    p = {k: v for k, v in p.items() if k not in frozen}
    if frozen:
        print(f"# frozen (no gradient path in this mode): {sorted(frozen)}", flush=True)
    loss_full = lambda q, d: loss_ml({**q, **frozen}, d)
    vg = jax.jit(jax.value_and_grad(loss_full, has_aux=True))
    # Full-training-set score for best-checkpoint + logging (a mini-batch loss varies
    # cell-to-cell and cannot rank iterates).  Evaluated only at the log cadence.
    fscore = jax.jit(lambda q: loss_full(q, data)) if use_batch else None
    # Gradient clipping: the initial loss is large (extreme high-latitude / desert /
    # ice cells contribute a huge skin-T error + seasonal-amplitude term), so unclipped
    # Adam at lr~3e-2 overshoots into a Richards/MOST-unstable parameter region and NaNs
    # within ~20 iters (independent of the soil-moisture target — the SM term is masked).
    # clip_by_global_norm bounds the step so the optimiser stays stable from the large-
    # loss start; clip<=0 disables it (the legacy bare-Adam path).
    tx = optax.adam(lr) if clip <= 0.0 else optax.chain(
        optax.clip_by_global_norm(clip), optax.adam(lr))
    opt = tx; state = opt.init(p)
    n_san = 0
    best_p, best_l = p, float("inf")     # keep the LOWEST-loss params, not the latest
    for it in range(n_iter):
        bdata = _select_cells(data, _brng.choice(ncol_tr, batch, replace=False)) \
            if use_batch else data
        (l, aux), g = vg(p, bdata)                    # step on the (mini-)batch gradient
        if it == 0:
            # Inert gate on the FULL-data init gradient, never a mini-batch (a
            # snow/glacier param absent from one 300-cell batch is not inert —
            # codex/GLM).  In batch mode this costs one extra full-data grad eval
            # at init; in full-batch mode g already is the full-data gradient.
            S.assert_no_inert(vg(p, data)[1] if use_batch else g)
        # Dynamic safety net.  The static pre-filter only removes cells pathological at
        # the INIT params; a parameter step can push a previously-healthy cell into the
        # stiff-clay/saturated regime whose MOST-iteration backward overflows to NaN/inf.
        # Zero any non-finite gradient COMPONENT before the optimiser sees it — otherwise
        # one NaN flows through Adam's moments and poisons EVERY parameter permanently.
        # The huge-but-finite case is handled by clip_by_global_norm; this only catches
        # NaN/inf.  Per-PFT parameter structure means a transient singular cell at worst
        # freezes its own PFT's update for that step, not the whole optimisation.
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in g.values()):
            g = jax.tree.map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), g)
            n_san += 1
        log = (it % 20 == 0 or it == n_iter - 1)
        # Score + rank on the FULL training set at the log/checkpoint cadence (the batch
        # loss is too noisy to rank iterates); p here is the PRE-update iterate.
        if not use_batch:
            score, (tm, am, pp, sa, sm, gb, lm, tbm, lbm) = float(l), aux
        elif log or (ckpt_path and it > 0 and it % ckpt_every == 0):
            fl, (tm, am, pp, sa, sm, gb, lm, tbm, lbm) = fscore(p); score = float(fl)
        else:
            score = None
        if score is not None and np.isfinite(score) and score < best_l:
            best_l, best_p = score, p
        upd, state = opt.update(g, state); p = optax.apply_updates(p, upd)
        if log:
            print(f"# it {it:3d} loss {score:.3f} T-RMSE {float(jnp.sqrt(tm)):.3f} "
                  f"T-bias {float(gb):+.3f} T-mbias {float(jnp.sqrt(tbm)):.3f} "
                  f"LE-RMSE {float(jnp.sqrt(lm)):.2f} LE-mbias {float(jnp.sqrt(lbm)):.2f} "
                  f"alb-RMSE {float(jnp.sqrt(am)):.4f} sm-RMSE {float(jnp.sqrt(sm)):.4f} "
                  f"perPFT {float(jnp.sqrt(pp)):.3f} seas-amp {float(jnp.sqrt(sa)):.3f}",
                  flush=True)
        # periodic checkpoint so a long (slow per-iter) run is interruptible and the
        # BEST-so-far params (not a late, possibly-diverged iterate) are captured.
        if ckpt_path and it > 0 and it % ckpt_every == 0:
            with open(ckpt_path, "w") as f:
                json.dump(_params_dict({**best_p, **frozen}), f, indent=2)
            print(f"# checkpoint -> {ckpt_path} (best loss {best_l:.3f} @ it {it})",
                  flush=True)
        # Early stop: the MOST/thermal cliff makes a too-large step blow the seasonal-
        # amplitude term up ~10x; once the loss runs away there is no recovery (the
        # sanitised gradient is biased near the cliff), so cut the run and keep best_p.
        if score is not None and np.isfinite(score) and score > 5.0 * best_l and it > 10:
            print(f"# early stop @ it {it}: loss {score:.1f} ran away from best "
                  f"{best_l:.3f} (MOST/thermal cliff); keeping best params", flush=True)
            break
    if n_san:
        print(f"# {n_san}/{n_iter} steps had a non-finite gradient component sanitised "
              f"(transient MOST-iteration singular cells); clip handled the rest",
              flush=True)
    print(f"# returning BEST params (loss {best_l:.3f})", flush=True)
    return _params_dict({**best_p, **frozen})


# --------------------------------------------------------------------------- #
# ERA5 + CLM training data (network/file)                                      #
# --------------------------------------------------------------------------- #
def load_training_data(diurnal_npz: str, n_sub: int, seed: int = 0,
                       days: int = _DAYS) -> dict:
    """Diurnal-resolved training data: per-month forcing keeps the ``NH`` sampled
    hours (leaves ``(NH, ncol)``); targets are the monthly mean over those hours.

    ``NH`` is read from the npz (4 synoptic hours or the full 24-hour climatology),
    so the same loader serves both forcings; ``days`` representative days/month sets
    the steps-per-month (smaller for the 24-hour forcing to keep the AD scan
    tractable)."""
    # Publish the diurnal-sampling structure as STATIC module constants (forward_ml
    # reads them as compile-time values; they must NOT be traced jit-arg leaves).
    global _NH, _DT, _SPM
    from legoesm.land.clm_surface_map import load_clm_surface, download_clm_surfdata
    D = np.load(diurnal_npz)
    nh = int(D["hours"].size) if "hours" in D else 4
    _NH, _DT, _SPM = nh, 86400.0 / nh, days * nh
    lat1, lon1 = D["lat"], D["lon"]; nlat, nlon = lat1.size, lon1.size
    land = D["lsm"].reshape(nlat, nlon) > 0.5; lidx = np.where(land.ravel())[0]
    latc = np.deg2rad(np.broadcast_to(lat1[:, None], (nlat, nlon))).ravel()[lidx]
    lonc = lon1[lidx % nlon]
    cmap = load_clm_surface(download_clm_surfdata(), np.rad2deg(latc), lonc)
    sub = np.random.default_rng(seed).choice(lidx.size, size=min(n_sub, lidx.size),
                                             replace=False)
    g = lambda k: D[k].reshape(12, nh, -1)[:, :, lidx][:, :, sub]       # (12, nh, ncol)
    _hours = np.asarray(D["hours"]) if "hours" in D else None
    return _pack(g, latc[sub], cmap, sub, lonc=lonc[sub], hours=_hours, nh=nh)


# --- Solar geometry (gap 3: real diurnal cos(zenith) for the zenith snow albedo) ---
# Mid-month day-of-year for the 12 months (Cooper 1969 declination).
_MONTH_DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def _solar_cos_zenith(lat_rad, lon_deg, hours_utc, month) -> np.ndarray:
    """cos(solar zenith) for one month, shape ``(nh, ncol)`` (clipped >= 0).

    ``mu = sin(lat) sin(delta) + cos(lat) cos(delta) cos(H)`` with the Cooper (1969)
    solar declination ``delta`` and the LOCAL hour angle ``H`` from the UTC hour plus
    the cell longitude (local solar time = UTC + lon/15).  Used to give the offline
    calibration a real diurnal sun so the BATS zenith snow brightening is active and
    calibratable (the old constant 0.5 placeholder left it inert)."""
    doy = _MONTH_DOY[month]
    delta = np.deg2rad(23.45) * np.sin(2.0 * np.pi * (284 + doy) / 365.0)
    lst = hours_utc[:, None] + lon_deg[None, :] / 15.0          # (nh, ncol) local solar hr
    H = np.deg2rad(15.0 * (lst - 12.0))
    mu = (np.sin(lat_rad)[None, :] * np.sin(delta)
          + np.cos(lat_rad)[None, :] * np.cos(delta) * np.cos(H))
    return np.maximum(mu, 0.0)


def _pack(g, latc, cmap, sub, lonc=None, hours=None, nh=_NH) -> dict:
    """Assemble the training dict from a per-key getter ``g(key) -> (12, nh, ncol)``."""
    _hours = hours if hours is not None else np.arange(nh) * (24.0 / nh)
    T2, D2, SP = g("2m_temperature"), g("2m_dewpoint_temperature"), g("surface_pressure")
    PR = np.maximum(g("precip_kgms"), 0)
    zc = lambda v: jnp.full((nh, latc.size), v)
    forc = [AtmToSurface(
        sw_down=jnp.asarray(g("ssrd_wm2")[m]), lw_down=jnp.asarray(g("strd_wm2")[m]),
        precip_total=jnp.asarray(PR[m]),
        precip_snow=jnp.where(jnp.asarray(T2[m]) < constants.T_freeze, jnp.asarray(PR[m]), 0.0),
        T_lowest=jnp.asarray(T2[m]),
        q_lowest=saturation_mixing_ratio(jnp.asarray(D2[m]), jnp.asarray(SP[m])),
        u_lowest=jnp.asarray(g("10m_u_component_of_wind")[m]),
        v_lowest=jnp.asarray(g("10m_v_component_of_wind")[m]),
        p_lowest=0.99 * jnp.asarray(SP[m]), p_surface=jnp.asarray(SP[m]),
        rho_lowest=jnp.asarray(SP[m]) / (constants.R_d * jnp.asarray(T2[m])),
        cos_zenith=(jnp.asarray(_solar_cos_zenith(latc, lonc, _hours, m))
                    if lonc is not None else zc(0.5)),
        co2_ppmv=zc(412.0), has_radiation=zc(1.0),
        has_precipitation=zc(1.0)) for m in range(12)]
    # ERA5/ARCO fields load as float32; cast forcing to float64 so the float64 soil
    # state and the MOST flux loop share one dtype (the fori_loop carry rejects a
    # float32/float64 mix that the constant-bulk path silently tolerated).
    forc = [jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), f) for f in forc]
    pft = np.asarray(cmap["pft_fractions"])[sub]
    dom = pft.argmax(1); oh = np.zeros((latc.size, 17)); oh[np.arange(latc.size), dom] = 1.0
    skt = g("skin_temperature").mean(1)                       # (12, ncol)
    # INSOLATION-WEIGHTED forecast albedo (weight = ssrd), matching the model's
    # insolation-weighted albedo in forward_ml so the zenith brightening (gap 3) is
    # compared on the SW-budget-relevant effective albedo, not a night-inflated mean.
    _ssrd = np.maximum(g("ssrd_wm2"), 0.0)                     # (12, nh, ncol)
    _ssrd_month = np.sum(_ssrd, axis=1)                        # (12, ncol) monthly SW total
    alb = np.clip(np.sum(g("forecast_albedo") * _ssrd, axis=1)
                  / np.maximum(_ssrd_month, 1e-6), 0.05, 0.85)
    # latent-heat target: ERA5 slhf_wm2 (positive-up, monthly mean over the sampled
    # hours) — the evaporation leg of the dual target.  Legacy npz without the field
    # yields all-NaN, dropped by the loss finite-mask.
    try:
        le = g("slhf_wm2").mean(1)                             # (12, ncol)
    except KeyError:
        le = np.full((12, latc.size), np.nan)
    # soil-moisture target: ERA5 swvl1 (0-7cm) + swvl2 (7-28cm), depth-weighted to a
    # single 0-28cm root-zone value, then the ANNUAL mean (the frozen column's signal).
    # Backward-compat: an OLD npz without soil moisture yields an all-NaN target, which
    # the loss finite-mask drops (the SM term then contributes nothing) so legacy inputs
    # still run (with --lam-sm 0 to silence, or simply ignored cell-by-cell).
    try:
        sm1 = g("swvl1").mean(1); sm2 = g("swvl2").mean(1)   # (12, ncol)
        sm = ((7.0 * sm1 + 21.0 * sm2) / 28.0).mean(0)       # (ncol,)
    except KeyError:
        sm = np.full(latc.shape[0], np.nan)
    # model root-zone weights: each soil layer's overlap with the top 0.28 m, so the
    # model's depth-weighted theta matches the ERA5 0-28cm target on the same grid.
    _grid = make_soil_grid(SoilGridConfig(n_layers=_N_LAYERS, total_depth=_SOIL_DEPTH_M,
                                          growth_factor=_SOIL_GROWTH))
    _dz = np.asarray(_grid.dz); _bot = np.cumsum(_dz)
    _ov = np.clip(0.28 - (_bot - _dz), 0.0, _dz)              # layer∩[0,0.28m]
    _K = int((_ov > 1e-9).sum())
    # per-cell longitude [deg], SAME (subsampled/permuted) order as every other
    # field — plotters must use this, not a re-derived unpermuted index map
    lon = lonc if lonc is not None else np.zeros_like(latc)
    data = dict(forc=forc, lat=jnp.asarray(latc), lon=jnp.asarray(lon),
                pft=jnp.asarray(pft),
                sm=jnp.asarray(sm), rz_w=jnp.asarray(_ov[:_K]),
                fg=jnp.asarray(np.asarray(cmap["glacier_frac"])[sub]),
                wp=jnp.asarray(np.asarray(cmap["theta_wp"])[sub]),
                fc=jnp.asarray(np.asarray(cmap["theta_fc"])[sub]),
                pct_sand=jnp.asarray(np.asarray(cmap["pct_sand"])[sub]),
                pct_clay=jnp.asarray(np.asarray(cmap["pct_clay"])[sub]),
                # per-cell CLM soil-colour bare-soil albedo (spatial desert/soil pattern)
                soil_albedo=jnp.asarray(np.asarray(cmap["soil_albedo"])[sub]),
                # sub-grid elevation std [m] (elevation-band snow scheme)
                std_elev=jnp.asarray(np.asarray(cmap["std_elev"])[sub]),
                skt=jnp.asarray(skt), alb=jnp.asarray(alb), le=jnp.asarray(le),
                # per-(month,cell) monthly SW total -> insolation weight for the albedo
                # loss so polar-night months (no sun, garbage albedo) don't bias amse.
                alb_wt=jnp.asarray(_ssrd_month),
                # init the whole soil column at the ERA5 annual-mean skin T
                t0=jnp.asarray(skt.mean(0)), dom_onehot=jnp.asarray(oh),
                w=jnp.cos(jnp.asarray(latc)))
    for k in ("theta_r", "theta_sat", "alpha_vg", "n_vg", "K_sat"):
        data["vg_" + k] = jnp.asarray(np.asarray(cmap[k])[sub])
    return data


def main():
    global _BULK_SCHEME, _STOMATA_ON, _ELEV_BANDS_ON, _LAM_AMP, _LAM_SM, _LAM_PFT, _LAM_ALB
    global _LAM_TBIAS, _LAM_LE, _LAM_TBIAS_MON, _LAM_LEBIAS_MON
    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lam-tbias", type=float, default=_LAM_TBIAS,
                    help="global skin-T BIAS penalty (area-weighted mean bias squared); "
                         "raise to drive the global-mean warm/cold bias toward 0 without "
                         "distorting the spatial fit (the RMSE term holds the pattern)")
    ap.add_argument("--lam-amp", type=float, default=_LAM_AMP,
                    help="seasonal-amplitude loss weight (raise to tighten the "
                         "seasonal cycle at some cost to the annual-mean fit)")
    ap.add_argument("--lam-sm", type=float, default=_LAM_SM,
                    help="soil-moisture loss weight (ERA5 annual-mean 0-28cm swvl vs "
                         "the model root-zone equilibrium; trains the porosity scale)")
    ap.add_argument("--lam-pft", type=float, default=_LAM_PFT,
                    help="per-PFT CLM5-prior regularisation weight (LOWER it to let the "
                         "params leave the prior and cut skin-T/albedo RMSE; the default "
                         "2.0 keeps the prior term ~equal to the skin-T term)")
    ap.add_argument("--lam-alb", type=float, default=_LAM_ALB,
                    help="albedo loss weight (RAISE it to force the albedo fit; the "
                         "default makes the albedo term small vs skin-T so albedo barely "
                         "moves)")
    ap.add_argument("--diurnal-npz", default="/tmp/era5_diurnal.npz",
                    help="ERA5 monthly-diurnal climatology (4-synoptic-hour "
                         "fetch_era5_diurnal.py, or 24-h fetch_era5_hourly_climatology.py)")
    ap.add_argument("--n-sub", type=int, default=800, help="land columns to train on")
    ap.add_argument("--days", type=int, default=_DAYS,
                    help="representative days/month (use fewer with 24-h forcing)")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient global-norm clip (stabilises the large-loss start; "
                         "<=0 disables = legacy bare-Adam)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bulk", choices=["constant", "most"], default=_BULK_SCHEME,
                    help="surface exchange: 'most' (default, matches the coupled "
                         "diurnal model -> z0 trainable) or 'constant' (per-PFT Ch)")
    ap.add_argument("--elev-bands", action="store_true",
                    help="enable the sub-grid elevation-band snow scheme (banded "
                         "precip phase/melt + permanent snow from CLM STD_ELEV)")
    ap.add_argument("--no-stomata", action="store_true",
                    help="disable Farquhar stomata (DEFAULT ON: the ERA5 latent-heat "
                         "target constrains Vc_max25/g1/LCMA; disabling freezes them "
                         "at CLM5 and drops them from the trainable set)")
    ap.add_argument("--lam-le", type=float, default=_LAM_LE,
                    help="latent-heat loss weight (ERA5 slhf_wm2 dual target; needs an "
                         "npz fetched with mean_surface_latent_heat_flux; 0 disables)")
    ap.add_argument("--lam-tbias-mon", type=float, default=_LAM_TBIAS_MON,
                    help="MONTHLY skin-T mean-bias penalty (seasonal-cycle bias is the "
                         "primary calibration target; RMSE fights natural variability)")
    ap.add_argument("--lam-lebias-mon", type=float, default=_LAM_LEBIAS_MON,
                    help="MONTHLY latent-heat mean-bias penalty (0 disables)")
    ap.add_argument("--out", default="results/land_tuned_multilayer.json")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of sampled cells held OUT of training to report a "
                         "generalisation test RMSE (0.2 = 80/20 train/test; 0 = off)")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="skip the O(ncol) per-cell NaN-gradient pre-filter (the "
                         "bottleneck at full-grid; safe on the AD-stable 24-h forcing "
                         "where the per-step sanitisation catches the rare bad cell)")
    ap.add_argument("--batch", type=int, default=0,
                    help="mini-batch SGD cell count per step (0 = full batch).  Use for "
                         "full-grid training: a single ~5k-cell gradient wanders (global "
                         "scalars + sanitiser bias), a random ~300-cell batch each step "
                         "stays in the stable regime yet visits every cell across the run")
    ap.add_argument("--keep-file", default=None,
                    help="cache the pre-filter keep-indices (.npy): computed+saved on the "
                         "first run (do it on CPU — the per-cell filter is slow on GPU), "
                         "loaded on later runs so a GPU full-grid run gets the CLEAN "
                         "pre-filtered gradient without paying the per-cell scan")
    ap.add_argument("--init-from", choices=["clm5", "baked"], default="clm5",
                    help="warm start: 'clm5' (default, the CLM5 prior) or 'baked' (the "
                         "production _TUNED_*_MULTILAYER params -> REFINE the well-tuned "
                         "model with the elevation bands active, instead of climbing "
                         "from the prior)")
    ap.add_argument("--init-json", default=None,
                    help="warm start from a saved CONSTRAINED tuned JSON (e.g. "
                         "results/land_tuned_allgaps.json), inverse-constrained to raw "
                         "and layered over the baked defaults; overrides --init-from so a "
                         "re-tune REFINES the current best instead of the baked prior")
    args = ap.parse_args()
    _BULK_SCHEME, _STOMATA_ON = args.bulk, not args.no_stomata
    _ELEV_BANDS_ON = args.elev_bands
    _LAM_AMP = args.lam_amp
    _LAM_SM = args.lam_sm
    _LAM_TBIAS = args.lam_tbias
    _LAM_LE = args.lam_le
    _LAM_TBIAS_MON = args.lam_tbias_mon
    _LAM_LEBIAS_MON = args.lam_lebias_mon
    _LAM_PFT = args.lam_pft
    _LAM_ALB = args.lam_alb
    data = load_training_data(args.diurnal_npz, args.n_sub, args.seed, args.days)
    # A missing LE field must not silently degrade the dual target to skin-T-only
    # (codex): the all-NaN fallback zeroes the LE term through the finite-mask.
    if _LAM_LE > 0.0 and not bool(np.isfinite(np.asarray(data["le"])).any()):
        raise SystemExit(
            "npz has no slhf_wm2 (latent-heat target): re-fetch with "
            "scripts/data/fetch_era5_hourly_climatology.py, or pass --lam-le 0 "
            "to explicitly train skin-T-only")
    # Pre-filter, cached: compute the per-cell NaN-gradient keep-set once (on CPU) and
    # reuse it so a GPU full-grid run trains on the CLEAN set without the slow per-cell
    # scan.  Filtering happens BEFORE the train/test split so both partitions are clean.
    prefilter = not args.no_prefilter
    if args.keep_file and os.path.exists(args.keep_file):
        keep = np.load(args.keep_file)
        data = _select_cells(data, keep); prefilter = False
        print(f"# loaded {keep.size} keep-cells from {args.keep_file} (skip per-cell filter)",
              flush=True)
    elif args.keep_file:
        keep = _drop_nan_grad_cells(data, init_ext_params())
        np.save(args.keep_file, keep)
        data = _select_cells(data, keep); prefilter = False
        print(f"# computed + saved {keep.size}/{args.n_sub} keep-cells -> {args.keep_file}",
              flush=True)
    test_data = None
    if args.holdout > 0.0:
        data, test_data = _split_cells(data, args.holdout, args.seed)
        print(f"# train/test split: {int(data['lat'].shape[0])} train / "
              f"{int(test_data['lat'].shape[0])} test (holdout {args.holdout:.0%}, "
              f"seed {args.seed})", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.init_json:
        init_params = json_init_params(args.init_json)
        print(f"# warm-start: refining tuned checkpoint {args.init_json}", flush=True)
    else:
        init_params = baked_init_params() if args.init_from == "baked" else None
        if init_params is not None:
            print("# warm-start: refining the production baked multilayer params", flush=True)
    tuned = train(data, n_iter=args.iters, lr=args.lr, ckpt_path=args.out, clip=args.clip,
                  prefilter=prefilter, batch=args.batch, init_params=init_params)
    with open(args.out, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"# recommended tuned params -> {args.out} (does NOT mutate production defaults)")
    if test_data is not None:
        _report_test(tuned, test_data)


if __name__ == "__main__":
    main()

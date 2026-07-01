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
soil van-Genuchten retention).  Ch is inert under MOST (used only under ``--bulk
constant``).  The Farquhar photosynthesis params (Vc_max25/g1/LCMA) are wired in but OFF
by default (``--stomata``): the offline single column has no atmospheric feedback to
stabilise them (~3 -> ~12 K RMSE), so they stay at physical CLM5 defaults and are
calibrated only in the coupled model (where they ARE active).

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
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.surface_albedo import LandAlbedoConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.carbon.stomata import StomataConfig
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
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
# Surface-exchange / photosynthesis mode (set by the CLI).  DEFAULT = the stable
# offline config: a constant (per-PFT) bulk coefficient and soil-only beta.  The
# faithful MOST exchange (makes z0 trainable) and Farquhar stomata (make Vc_max25/g1/
# LCMA trainable) are reachable via --bulk most / --stomata, but the crude offline
# single-column forcing cannot constrain them — they roughly TRIPLE the skin-T RMSE
# (~3.7 -> ~12 K) and need the coupled diurnal atmosphere to calibrate.
# MOST surface exchange (matches the coupled DIURNAL default -> z0 trainable, RMSE ~3.1
# vs constant-Ch 3.3).  Farquhar stomata are OFF for the OFFLINE calibration: with no
# atmospheric feedback the offline single column cannot constrain Vc_max25/g1/LCMA (they
# triple the RMSE), so they stay at physical CLM5 defaults — they ARE active in the
# coupled model (land_diurnal_surface).  --stomata re-enables them for experiments.
_BULK_SCHEME = "most"
_STOMATA_ON = False
# Loss weights (CLI-tunable).  lam_amp raised from 0.5 -> 1.5: the residual is
# dominated by the seasonal-cycle AMPLITUDE (mid-lats under, Antarctica over), and the
# per-PFT thermal inertia has head-room the low weight wasn't exploiting.
_LAM_ALB = 300.0
_LAM_PFT = 2.0
_LAM_AMP = 1.5


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
BOUNDS_EXT = dict(
    pft_alb=(S._PFT_ALB_LO, S._PFT_ALB_HI), pft_emis=(0.94, 0.99),
    pft_root=(S._PFT_ROOT_LO, S._PFT_ROOT_HI),
    pft_ch=(2.0e-3, 6.0e-3),                           # per-PFT bulk exch (constant bulk)
    pft_z0=(5e-3, 3.0),                                # roughness (MOST); forests saturated 2.0
    pft_vcmax=(0.0, 80.0), pft_lcma=(20.0, 90.0), pft_g1=(1.0, 12.0),
    pft_wp=(0.05, 0.30), pft_fcgap=(0.03, 0.30),       # theta_fc_plant = wp + gap
    # per-PFT SCALE on the per-cell texture-derived soil thermal k_solid / C_solid
    # (texture sets the spatial pattern; the scale sets the per-PFT magnitude)
    pft_kscale=(0.1, 1.5), pft_cscale=(0.3, 2.0),
    th_glacier_cboost=(1.0, 15.0),                     # deep-ice inertia boost (glacier)
    glac_alb=(0.45, 0.75), snow_max=(0.60, 0.85))


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
        # init scales so texture*scale ~ the previous effective inertia (texture
        # k_base~5 -> k_scale~0.35 gives k_solid~1.8; C_base~2.3e6 -> c_scale~0.9)
        pft_kscale=_inv_ext(full(0.35), "pft_kscale"),
        pft_cscale=_inv_ext(full(0.9), "pft_cscale"),
        th_glacier_cboost=jnp.asarray(_inv_ext(5.0, "th_glacier_cboost")),
        glac_alb=_inv_ext(0.55, "glac_alb"), snow_max=_inv_ext(0.80, "snow_max"),
    ).items()}


# --------------------------------------------------------------------------- #
# Pure (testable, differentiable) calibration core                            #
# --------------------------------------------------------------------------- #
def _ml_land_params(cp, data):
    """Per-column LandSurfaceParams from the constrained EXTENDED per-PFT params.
    theta_wp/theta_fc are the PLANT (btran) stress thresholds (per-PFT, trainable),
    NOT the soil retention; W_max is unused by the Richards forward (kept as the CLM5
    value for completeness)."""
    pw = lambda k: data["pft"] @ cp[k]                  # PFT-weighted per cell
    alb = (1 - data["fg"]) * pw("pft_alb") + data["fg"] * cp["glac_alb"]
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
    hyd = SoilHydraulicsConfig(theta_r=col("theta_r"), theta_sat=col("theta_sat"),
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
        land_albedo=LandAlbedoConfig(alpha_snow_max=cp["snow_max"]))
    # Carbon state is PRESCRIBED (fixed climatological leaf carbon -> fixed LAI), the
    # same decoupling as the soil-moisture trick: it activates the photosynthesis /
    # stomatal-conductance parameters (Vc_max25, g1, LCMA via LAI = C_fol/LCMA) so they
    # are trainable, without paying for a multi-decade carbon-pool spin-up.
    cs0 = init_carbon_state((data["lat"].shape[0],), cfg.carbon)
    return cfg, _ml_land_params(cp, data), hyd, cs0


def forward_ml(cp, data, return_diag: bool = False):
    """Monthly-mean T_sfc + surface albedo (12, ncol) under constrained params.

    With ``return_diag=True`` also returns monthly-mean sensible + latent heat
    fluxes and the final (last-scored-month) soil state — the extra land
    diagnostics a forward LMIP run wants (the calibration loss ignores them, so
    the default 2-tuple return is unchanged).  See ``scripts/run/run_lmip_era5.py``.
    """
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

    # --- STAGE B: seasonal years with subdaily forcing -> monthly means ----------
    @jax.checkpoint     # remat per step -> bounded backward memory
    def body(carry, k):
        s, Tsum, Asum, SHsum, LHsum = carry
        month = k // spm; hour = k % nh
        f = jax.tree.map(lambda x: x[month, hour], fstack)
        s2, r = step(s, f)
        s2 = s2._replace(
            theta_soil=s2.theta_soil.at[:, _FREEZE_FROM:].set(deep),
            psi_soil=s2.psi_soil.at[:, _FREEZE_FROM:].set(deep_psi))  # pin deep reservoir
        Tsum = Tsum.at[month].add(r.T_sfc); Asum = Asum.at[month].add(r.albedo)
        SHsum = SHsum.at[month].add(r.shflx); LHsum = LHsum.at[month].add(r.lhflx)
        return (s2, Tsum, Asum, SHsum, LHsum), None

    # First seasonal pass is an unscored spin; only the second year is scored so the
    # monthly means are free of first-cycle thermal/snow transients.
    z = lambda: jnp.zeros((12, n), dtype=st.T_soil.dtype)
    (st, *_), _ = jax.lax.scan(body, (st, z(), z(), z(), z()), jnp.arange(nstep))
    (st, Tsum, Asum, SHsum, LHsum), _ = jax.lax.scan(
        body, (st, z(), z(), z(), z()), jnp.arange(nstep))
    if return_diag:
        return Tsum / spm, Asum / spm, SHsum / spm, LHsum / spm, st
    return Tsum / spm, Asum / spm


def loss_ml(p, data, lam_alb=None, lam_pft=None, lam_amp=None):
    # weights default to the module globals (CLI-tunable) so the jitted
    # value_and_grad picks up an updated lam_amp without re-partialling.
    lam_alb = _LAM_ALB if lam_alb is None else lam_alb
    lam_pft = _LAM_PFT if lam_pft is None else lam_pft
    lam_amp = _LAM_AMP if lam_amp is None else lam_amp
    cp = constrain_ext(p)
    T, A = forward_ml(cp, data)
    w = data["w"][None, :]
    tmse = jnp.sum(w * (T - data["skt"]) ** 2) / jnp.sum(w) / 12
    amse = jnp.sum(w * (A - data["alb"]) ** 2) / jnp.sum(w) / 12
    ann = (T - data["skt"]).mean(0)
    oh = data["dom_onehot"] * data["w"][:, None]
    pb = (oh * ann[:, None]).sum(0) / (oh.sum(0) + 1e-9)
    present = (data["dom_onehot"].sum(0) > 0).astype(ann.dtype)
    ppft = jnp.sum(present * pb ** 2) / jnp.sum(present + 1e-9)
    # seasonal-amplitude term: penalise model monthly amplitude away from ERA5
    amp = (T.max(0) - T.min(0)) - (data["skt"].max(0) - data["skt"].min(0))
    samp = jnp.sum(data["w"] * amp ** 2) / jnp.sum(data["w"])
    return tmse + lam_alb * amse + lam_pft * ppft + lam_amp * samp, (tmse, amse, ppft, samp)


def _params_dict(p):
    return {k: np.asarray(v).tolist() for k, v in constrain_ext(p).items()}


def train(data, n_iter=250, lr=3e-2, ckpt_path=None, ckpt_every=50):
    p = init_ext_params()
    vg = jax.jit(jax.value_and_grad(loss_ml, has_aux=True))
    opt = optax.adam(lr); state = opt.init(p)
    for it in range(n_iter):
        (l, (tm, am, pp, sa)), g = vg(p, data)
        upd, state = opt.update(g, state); p = optax.apply_updates(p, upd)
        if it % 20 == 0 or it == n_iter - 1:
            print(f"# it {it:3d} loss {float(l):.3f} T-RMSE {float(jnp.sqrt(tm)):.3f} "
                  f"alb-RMSE {float(jnp.sqrt(am)):.4f} perPFT {float(jnp.sqrt(pp)):.3f} "
                  f"seas-amp {float(jnp.sqrt(sa)):.3f}", flush=True)
        # periodic checkpoint so a long (slow per-iter) run is interruptible and the
        # converged params are captured before the final iteration.
        if ckpt_path and it > 0 and it % ckpt_every == 0:
            with open(ckpt_path, "w") as f:
                json.dump(_params_dict(p), f, indent=2)
            print(f"# checkpoint -> {ckpt_path} (it {it})", flush=True)
    return _params_dict(p)


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
    return _pack(g, latc[sub], lonc[sub], cmap, sub, nh)


def _pack(g, latc, lonc, cmap, sub, nh=_NH) -> dict:
    """Assemble the training dict from a per-key getter ``g(key) -> (12, nh, ncol)``."""
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
        cos_zenith=zc(0.5), co2_ppmv=zc(412.0), has_radiation=zc(1.0),
        has_precipitation=zc(1.0)) for m in range(12)]
    # ERA5/ARCO fields load as float32; cast forcing to float64 so the float64 soil
    # state and the MOST flux loop share one dtype (the fori_loop carry rejects a
    # float32/float64 mix that the constant-bulk path silently tolerated).
    forc = [jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), f) for f in forc]
    pft = np.asarray(cmap["pft_fractions"])[sub]
    dom = pft.argmax(1); oh = np.zeros((latc.size, 17)); oh[np.arange(latc.size), dom] = 1.0
    skt = g("skin_temperature").mean(1)                       # (12, ncol)
    alb = np.clip(g("forecast_albedo").mean(1), 0.05, 0.85)
    data = dict(forc=forc, lat=jnp.asarray(latc), pft=jnp.asarray(pft),
                fg=jnp.asarray(np.asarray(cmap["glacier_frac"])[sub]),
                wp=jnp.asarray(np.asarray(cmap["theta_wp"])[sub]),
                fc=jnp.asarray(np.asarray(cmap["theta_fc"])[sub]),
                pct_sand=jnp.asarray(np.asarray(cmap["pct_sand"])[sub]),
                pct_clay=jnp.asarray(np.asarray(cmap["pct_clay"])[sub]),
                skt=jnp.asarray(skt), alb=jnp.asarray(alb),
                # init the whole soil column at the ERA5 annual-mean skin T
                t0=jnp.asarray(skt.mean(0)), dom_onehot=jnp.asarray(oh),
                lat_deg=jnp.rad2deg(jnp.asarray(latc)), lon_deg=jnp.asarray(lonc),
                w=jnp.cos(jnp.asarray(latc)))
    for k in ("theta_r", "theta_sat", "alpha_vg", "n_vg", "K_sat"):
        data["vg_" + k] = jnp.asarray(np.asarray(cmap[k])[sub])
    return data


def main():
    global _BULK_SCHEME, _STOMATA_ON, _LAM_AMP
    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lam-amp", type=float, default=_LAM_AMP,
                    help="seasonal-amplitude loss weight (raise to tighten the "
                         "seasonal cycle at some cost to the annual-mean fit)")
    ap.add_argument("--diurnal-npz", default="/tmp/era5_diurnal.npz",
                    help="ERA5 monthly-diurnal climatology (4-synoptic-hour "
                         "fetch_era5_diurnal.py, or 24-h fetch_era5_hourly_climatology.py)")
    ap.add_argument("--n-sub", type=int, default=800, help="land columns to train on")
    ap.add_argument("--days", type=int, default=_DAYS,
                    help="representative days/month (use fewer with 24-h forcing)")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bulk", choices=["constant", "most"], default=_BULK_SCHEME,
                    help="surface exchange: 'most' (default, matches the coupled "
                         "diurnal model -> z0 trainable) or 'constant' (per-PFT Ch)")
    ap.add_argument("--stomata", action="store_true",
                    help="enable Farquhar stomata (makes Vc_max25/g1/LCMA trainable; "
                         "degrades the offline fit — needs the coupled model)")
    ap.add_argument("--out", default="results/land_tuned_multilayer.json")
    args = ap.parse_args()
    _BULK_SCHEME, _STOMATA_ON = args.bulk, args.stomata
    _LAM_AMP = args.lam_amp
    data = load_training_data(args.diurnal_npz, args.n_sub, args.seed, args.days)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tuned = train(data, n_iter=args.iters, lr=args.lr, ckpt_path=args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"# recommended tuned params -> {args.out} (does NOT mutate production defaults)")


if __name__ == "__main__":
    main()

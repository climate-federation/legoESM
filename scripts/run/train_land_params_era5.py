"""Differentiable calibration of land PFT + snow/ice parameters against ERA5.

End-to-end gradients (jax.grad) through a multi-year ERA5-forced slab-land run tune
GLOBAL, physically-bounded land parameters to minimise the land surface-temperature
AND albedo bias vs ERA5 (skin temperature + forecast albedo). Trainable, per-PFT
where it makes sense:

    pft_alb (17)      per-PFT snow-free albedo         (physical per-PFT bounds)
    pft_emis (17)     per-PFT emissivity               [0.94, 0.99]
    pft_wmax (17)     bucket capacity [kg/m2]          [60, 320]
    pft_snowmask (17) canopy snow-cover masking        [0.2, 1.2] (1 = legacy)
    glac_alb          glacier/ice-sheet ice albedo     [0.45, 0.85]
    snow_max/min      fresh / aged snow albedo         [0.60,0.90] / [0.45,0.75]
    snow_dcrit        tanh SWE half-cover scale        [3, 60] kg/m2
    snow_tau_days     snow-albedo age e-folding        [1, 20] d
    ch                bulk heat-transfer coefficient   [2e-3, 5e-3]

(Roughness z0 is intentionally NOT trained: the constant bulk-flux scheme fixes the
exchange coefficient, so z0 has zero gradient; switch to the MOST scheme to calibrate
z0 per PFT.)

Loss = cos-lat-weighted MSE(T_sfc, skin_T) + lam_alb·MSE(albedo, forecast_albedo)
+ lam_le·MSE(lhflx, ERA5 slhf_wm2) + lam_pft·(per-dominant-PFT mean-bias)² — the
ERA5 DUAL TARGET (evaporation + skin temperature) shared with the multilayer
calibrator; the latent-heat term is what constrains the bucket capacity W_max
(the slab's root-zone-storage stand-in) and the rooting depth.  The per-PFT term
drives EACH PFT class's mean bias toward zero so no PFT compensates another.
Optimiser = Adam (these are scalar physics params; MUON's orthogonalisation is
for matrices).

STRICT RULE: no inert parameters, ever — the first training step asserts every
trainable leaf carries loss gradient (``assert_no_inert``); a zero-gradient
parameter aborts the run instead of silently pretending to be calibrated.

Result (~2° land, 2-yr spin, vs ERA5 skin T): global bias +3.1→+0.9 K, RMSE
4.4→2.1 K; per-PFT mean-bias RMS 3.2→0.9 K. Writes the tuned params to
``results/land_tuned_params.json`` (RECOMMENDED values — does NOT mutate production
config defaults; tuned to OFFLINE monthly forcing, so a coupled run with a live
diurnal cycle should re-tune).

Run: PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python scripts/run/train_land_params_era5.py
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
from legoesm.core.field import Field
from legoesm.land import LandState
from legoesm.land.config import LandConfig
from legoesm.surface_albedo import LandAlbedoConfig
from legoesm.land.slab_land import step_land
from legoesm.land.surface_params import (
    LandSurfaceParams, clm5_pft_table, PARAM_NAMES)

_TABLE = np.asarray(clm5_pft_table())          # (17,12) CLM5 PFT params
_PI = {n: i for i, n in enumerate(PARAM_NAMES)}
_N_PFT = 17
# PHYSICAL per-PFT snow-free albedo bounds (lo, hi) in CLM5 PFT order, so the
# calibration cannot over-brighten vegetation to cancel the offline-forcing warm
# bias (literature broadband ranges): bare/soil bright, forests dark, grass/crop mid.
_PFT_ALB_LO = np.array([0.25, 0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09,
                        0.12, 0.12, 0.12, 0.14, 0.15, 0.15, 0.14, 0.14])
_PFT_ALB_HI = np.array([0.40, 0.14, 0.14, 0.16, 0.16, 0.16, 0.17, 0.17, 0.17,
                        0.22, 0.22, 0.22, 0.24, 0.25, 0.25, 0.26, 0.26])
# per-PFT rooting depth [m] (water uptake): forests deep, grass/crop shallow, bare ~0.
_PFT_ROOT_LO = np.array([0.05, 0.8, 0.8, 0.6, 1.0, 1.0, 0.8, 0.8, 0.6,
                         0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3])
_PFT_ROOT_HI = np.array([0.3, 3.0, 3.0, 2.5, 4.0, 4.0, 3.0, 3.0, 2.5,
                         2.0, 2.0, 2.0, 1.2, 1.5, 1.5, 1.5, 1.5])
# per-PFT bucket capacity [kg/m2] (plant-available water store).
_PFT_WMAX_LO, _PFT_WMAX_HI = 60.0, 320.0
# constrained parameter bounds (lo, hi); per-PFT params are length-17 vectors.
# NOTE: pft_root is NOT a slab trainable — the slab water stress is beta(W/W_max)
# only, root_depth is never consumed on the slab path (the inert-parameter gate
# caught it with an identically-zero gradient).  The multilayer calibrator trains
# it (real root-zone uptake); the slab's storage knob is pft_wmax.
BOUNDS = dict(pft_alb=(_PFT_ALB_LO, _PFT_ALB_HI), pft_emis=(0.94, 0.99),
              pft_wmax=(_PFT_WMAX_LO, _PFT_WMAX_HI),
              glac_alb=(0.45, 0.85), snow_max=(0.60, 0.90), ch=(2.0e-3, 5.0e-3),
              # snow-albedo scalars unfrozen for the NH dark bias (previously only
              # snow_max trained; min/cover-scale/age at LandAlbedoConfig defaults)
              snow_min=(0.45, 0.75), snow_dcrit=(3.0, 60.0), snow_tau_days=(1.0, 20.0),
              # per-PFT snow-cover masking (canopy snow burial; 1 = legacy cover)
              pft_snowmask=(0.2, 1.2))
_STEPS_PER_MONTH = 120                          # 6-h steps over ~30 days
_DT = 6 * 3600.0
# Latent-heat loss weight: LE MSE is O(10^2-10^3) (W/m2)^2 vs skin-T MSE O(10) K^2,
# so 0.02 puts the two terms of the ERA5 dual target (evaporation + skin T) on the
# same footing.  The bucket capacity W_max (root-zone storage stand-in) and root
# depth get their gradient primarily through this term.
_LAM_LE = 0.02


def assert_no_inert(g: dict) -> None:
    """STRICT RULE (user directive 2026-08-17): no inert parameters, ever.

    Every leaf in the trainable pytree must carry loss gradient; a parameter the
    forward never consumes silently pretends to be calibrated.  Called on the
    FULL-data init gradient of every training run (never a mini-batch — a
    seasonally/spatially gated param absent from one batch is not inert).

    Raises on (a) an identically-zero leaf (inert) and (b) an all-non-finite
    leaf — the reviewers' case: an all-NaN gradient would be zeroed by the
    downstream sanitiser every step, i.e. silently inert while looking live.
    A PARTIALLY zero per-PFT vector (some components live) passes with a printed
    warning: components for PFTs absent from the sample are legitimately zero
    and stay pinned at their prior."""
    dead = [k for k, v in g.items() if bool(jnp.all(v == 0.0))]
    poisoned = [k for k, v in g.items() if not bool(jnp.any(jnp.isfinite(v)))]
    if dead or poisoned:
        raise ValueError(
            "no-inert-parameters gate: "
            + (f"identically-zero gradient: {sorted(dead)}; " if dead else "")
            + (f"all-non-finite gradient (sanitiser would zero it every step): "
               f"{sorted(poisoned)}; " if poisoned else "")
            + "remove the key from the trainable set, enable the physics path "
              "that consumes it, or fix the init state (run the pre-filter)")
    for k, v in g.items():
        nz = int(jnp.sum(v == 0.0)); tot = int(jnp.asarray(v).size)
        if 0 < nz and tot > 1 and nz > tot // 2:
            print(f"# no-inert gate: {k} has {nz}/{tot} zero-gradient components "
                  f"(PFTs absent from the sample stay at the prior)", flush=True)


# --------------------------------------------------------------------------- #
# Pure (testable, differentiable) calibration core                            #
# --------------------------------------------------------------------------- #
def constrain(p: dict) -> dict:
    return {k: BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) * jax.nn.sigmoid(v)
            for k, v in p.items()}


def _inv(v, k):
    """Inverse-sigmoid of value(s) v into raw space for bounded param k (lo/hi may
    be scalars or per-PFT arrays)."""
    lo, hi = BOUNDS[k]
    v = np.clip(v, np.asarray(lo) + 1e-4, np.asarray(hi) - 1e-4)
    return np.log((v - lo) / (hi - v))


def init_raw_params() -> dict:
    """Raw (unconstrained) params initialised at the CLM5 defaults (clamped into the
    physical bounds)."""
    return dict(
        pft_alb=jnp.asarray(_inv(_TABLE[:, _PI["albedo_veg"]], "pft_alb")),  # (17,)
        pft_emis=jnp.asarray(np.full(_N_PFT, _inv(0.96, "pft_emis"))),
        pft_wmax=jnp.asarray(np.full(_N_PFT, _inv(150.0, "pft_wmax"))),
        glac_alb=jnp.asarray(_inv(0.55, "glac_alb")),
        snow_max=jnp.asarray(_inv(0.80, "snow_max")),
        snow_min=jnp.asarray(_inv(0.50, "snow_min")),
        snow_dcrit=jnp.asarray(_inv(20.0, "snow_dcrit")),
        snow_tau_days=jnp.asarray(_inv(5.0, "snow_tau_days")),
        pft_snowmask=jnp.asarray(np.full(_N_PFT, _inv(1.0, "pft_snowmask"))),
        ch=jnp.asarray(_inv(3.0e-3, "ch")))


def _land_params(cp, data):
    n = data["lat"].shape[0]
    alb = (1 - data["fg"]) * (data["pft"] @ cp["pft_alb"]) + data["fg"] * cp["glac_alb"]
    emis = data["pft"] @ cp["pft_emis"]
    other = data["pft"] @ jnp.asarray(_TABLE)
    lp = LandSurfaceParams(
        albedo_veg=alb, emissivity=emis, z0=other[:, _PI["z0"]],
        W_max=data["pft"] @ cp["pft_wmax"], C_soil=other[:, _PI["C_soil"]],
        d_soil=other[:, _PI["d_soil"]], root_depth=other[:, _PI["root_depth"]],
        theta_wp=data["wp"], theta_fc=data["fc"], Vc_max25=other[:, _PI["Vc_max25"]],
        LCMA=other[:, _PI["LCMA"]], g1=other[:, _PI["g1"]])
    cfg = LandConfig(snow_albedo_feedback=True, Ch_land=cp["ch"], Cd_land=cp["ch"],
                     land_albedo=LandAlbedoConfig(
                         alpha_snow_max=cp["snow_max"], alpha_snow_min=cp["snow_min"],
                         snow_depth_crit=cp["snow_dcrit"],
                         tau_snow_decay=cp["snow_tau_days"] * 86400.0,
                         snow_cover_scale=data["pft"] @ cp["pft_snowmask"]))
    return lp, cfg


def forward(cp, data, n_spin_years: int = 2):
    """Monthly mean T_sfc, surface albedo and latent heat (12, ncol) under
    constrained params."""
    n = data["lat"].shape[0]
    lp, cfg = _land_params(cp, data)

    def st(a):
        F = lambda d, u: Field(d, name="x", dims=("c",), units=u)
        return LandState(T_soil=F(a[0], "K"), W_bucket=F(a[1], "kg/m2"),
                         snow_depth=F(a[2], "kg/m2"), snow_age=F(a[3], "s"))

    @jax.checkpoint
    def run_month(a, f):
        def body(c, _):
            ar, ts, ab, le = c
            ns, r, _ = step_land(st(ar), f, cfg, 1.0, _DT, lat=data["lat"], doy=15.0,
                                 land_params=lp)
            return ((ns.T_soil.data, ns.W_bucket.data, ns.snow_depth.data,
                     ns.snow_age.data), ts + r.T_sfc, ab + r.albedo,
                    le + r.lhflx), None
        (a, ts, ab, le), _ = jax.lax.scan(
            body, (a, jnp.zeros((n,)), jnp.zeros((n,)), jnp.zeros((n,))),
            None, length=_STEPS_PER_MONTH)
        return a, ts / _STEPS_PER_MONTH, ab / _STEPS_PER_MONTH, le / _STEPS_PER_MONTH

    a = (data["t0"].astype(jnp.float64), jnp.full((n,), 75.0),
         jnp.zeros((n,)), jnp.zeros((n,)))
    Tm = Am = Lm = None
    for _ in range(n_spin_years):
        Tm, Am, Lm = [], [], []
        for m in range(12):
            a, tm, am, lm = run_month(a, data["forc"][m])
            Tm.append(tm); Am.append(am); Lm.append(lm)
    return jnp.stack(Tm), jnp.stack(Am), jnp.stack(Lm)


# MONTHLY-mean bias penalties (user 2026-08-17: "RMSE reduction is challenging —
# natural variability. Reducing mean bias, including monthly, is a MUST").
_LAM_TBIAS_MON = 30.0
_LAM_LEBIAS_MON = 0.3


def loss_fn(p, data, lam_alb=300.0, lam_pft=2.0, lam_le=_LAM_LE, n_spin_years=2,
            lam_tbias_mon=_LAM_TBIAS_MON, lam_lebias_mon=_LAM_LEBIAS_MON):
    cp = constrain(p)
    T, A, L = forward(cp, data, n_spin_years)
    w = data["w"][None, :]
    tmse = jnp.sum(w * (T - data["skt"]) ** 2) / jnp.sum(w) / 12
    amse = jnp.sum(w * (A - data["alb"]) ** 2) / jnp.sum(w) / 12
    # ERA5 dual target: latent heat (positive-up slhf_wm2) next to skin T.
    # Finite-masked on BOTH sides (a non-finite model L must not poison the
    # gradient any more than a missing target); inputs sanitised BEFORE the diff
    # (where-NaN-gradient gotcha).  lam_le == 0 is a STATIC flag: term skipped.
    if lam_le > 0.0 or lam_lebias_mon > 0.0:
        le_ok = jnp.isfinite(L) & jnp.isfinite(data["le"])
        L_s = jnp.where(le_ok, L, 0.0); le_t = jnp.where(le_ok, data["le"], 0.0)
        wle = w * le_ok
        lemse = jnp.sum(wle * (L_s - le_t) ** 2) / (jnp.sum(wle) + 1e-9)
        # per-month area-weighted mean LE bias, squared, averaged over months
        mb_le = jnp.sum(wle * (L_s - le_t), axis=1) / (jnp.sum(wle, axis=1) + 1e-9)
        lebias_mon = jnp.mean(mb_le ** 2)
    else:
        lemse = jnp.zeros((), tmse.dtype)
        lebias_mon = jnp.zeros((), tmse.dtype)
    # per-month area-weighted mean skin-T bias (seasonal-cycle bias target)
    mb_t = jnp.sum(w * (T - data["skt"]), axis=1) / jnp.sum(w)
    tbias_mon = jnp.mean(mb_t ** 2)
    ann = (T - data["skt"]).mean(0)
    oh = data["dom_onehot"] * data["w"][:, None]
    pft_bias = (oh * ann[:, None]).sum(0) / (oh.sum(0) + 1e-9)
    present = (data["dom_onehot"].sum(0) > 0).astype(ann.dtype)
    ppft = jnp.sum(present * pft_bias ** 2) / jnp.sum(present + 1e-9)
    return (tmse + lam_alb * amse + lam_pft * ppft + lam_le * lemse
            + lam_tbias_mon * tbias_mon + lam_lebias_mon * lebias_mon,
            (tmse, amse, ppft, lemse, tbias_mon, lebias_mon))


def train(data, n_iter=80, lr=3e-2, lam_le=_LAM_LE, lam_tbias_mon=_LAM_TBIAS_MON,
          lam_lebias_mon=_LAM_LEBIAS_MON):
    p = init_raw_params()
    # weight kwargs are STATIC (the loss branches on them in Python); passing
    # them as traced jit kwargs raises TracerBoolConversionError.
    vg = jax.jit(jax.value_and_grad(loss_fn, has_aux=True),
                 static_argnames=("lam_le", "n_spin_years",
                                  "lam_tbias_mon", "lam_lebias_mon"))
    opt = optax.adam(lr); state = opt.init(p)
    for it in range(n_iter):
        (l, (tm, am, pp, lm, tbm, lbm)), g = vg(
            p, data, lam_le=lam_le, lam_tbias_mon=lam_tbias_mon,
            lam_lebias_mon=lam_lebias_mon)
        if it == 0:
            assert_no_inert(g)          # strict no-inert-parameters gate
        upd, state = opt.update(g, state); p = optax.apply_updates(p, upd)
        if it % 10 == 0 or it == n_iter - 1:
            print(f"# it {it:3d} loss {float(l):.3f} T-RMSE {float(jnp.sqrt(tm)):.3f} "
                  f"T-mbias {float(jnp.sqrt(tbm)):.3f} "
                  f"LE-RMSE {float(jnp.sqrt(lm)):.2f} LE-mbias {float(jnp.sqrt(lbm)):.2f} "
                  f"alb-RMSE {float(jnp.sqrt(am)):.4f} perPFT-bias-RMS {float(jnp.sqrt(pp)):.3f}")
    return {k: np.asarray(v).tolist() for k, v in constrain(p).items()}


# --------------------------------------------------------------------------- #
# ERA5 + CLM training data (network/file)                                     #
# --------------------------------------------------------------------------- #
def load_training_data(diurnal_npz: str, n_sub: int, seed: int = 0) -> dict:
    from legoesm.land.clm_surface_map import load_clm_surface, download_clm_surfdata
    D = np.load(diurnal_npz)
    lat1, lon1 = D["lat"], D["lon"]; nlat, nlon = lat1.size, lon1.size
    land = D["lsm"].reshape(nlat, nlon) > 0.5
    lidx = np.where(land.ravel())[0]
    latc = np.deg2rad(np.broadcast_to(lat1[:, None], (nlat, nlon))).ravel()[lidx]
    lonc = lon1[lidx % nlon]
    nh = D["hours"].size
    mh = lambda k: D[k].reshape(12, nh, -1)[:, :, lidx].mean(1)
    cmap = load_clm_surface(download_clm_surfdata(), np.rad2deg(latc), lonc)
    sub = np.random.default_rng(seed).choice(lidx.size, size=min(n_sub, lidx.size),
                                             replace=False)
    return _pack(mh, latc, np.asarray(cmap["pft_fractions"]),
                 np.asarray(cmap["glacier_frac"]), np.asarray(cmap["theta_wp"]),
                 np.asarray(cmap["theta_fc"]), sub, lonc=lonc)


def _pack(mh, latc, pft, fg, wp, fc, idx, lonc=None):
    z = lambda v: jnp.full((idx.size,), v)
    T2 = mh("2m_temperature"); D2 = mh("2m_dewpoint_temperature")
    SP = mh("surface_pressure"); PR = np.maximum(mh("precip_kgms"), 0)
    forc = [AtmToSurface(
        sw_down=jnp.asarray(mh("ssrd_wm2")[m, idx]), lw_down=jnp.asarray(mh("strd_wm2")[m, idx]),
        precip_total=jnp.asarray(PR[m, idx]),
        precip_snow=jnp.where(jnp.asarray(T2[m, idx]) < constants.T_freeze,
                              jnp.asarray(PR[m, idx]), 0.0),
        T_lowest=jnp.asarray(T2[m, idx]),
        q_lowest=saturation_mixing_ratio(jnp.asarray(D2[m, idx]), jnp.asarray(SP[m, idx])),
        u_lowest=jnp.asarray(mh("10m_u_component_of_wind")[m, idx]),
        v_lowest=jnp.asarray(mh("10m_v_component_of_wind")[m, idx]),
        p_lowest=0.99 * jnp.asarray(SP[m, idx]), p_surface=jnp.asarray(SP[m, idx]),
        rho_lowest=jnp.asarray(SP[m, idx]) / (constants.R_d * jnp.asarray(T2[m, idx])),
        cos_zenith=z(0.5), co2_ppmv=z(412.0), has_radiation=z(1.0),
        has_precipitation=z(1.0)) for m in range(12)]
    dom = np.argmax(pft[idx], axis=1)
    oh = np.zeros((idx.size, _N_PFT)); oh[np.arange(idx.size), dom] = 1.0
    # LE target (positive-up W/m2, monthly mean); legacy npz without the field
    # yields all-NaN -> the loss finite-mask drops the term.
    try:
        le = mh("slhf_wm2")[:, idx]
    except KeyError:
        le = np.full((12, idx.size), np.nan)
    # per-cell longitude [deg] in the SAME subsampled order as every other field
    lon = (lonc[idx] if lonc is not None else np.zeros(idx.size))
    return dict(forc=forc, lat=jnp.asarray(latc[idx]), lon=jnp.asarray(lon),
                pft=jnp.asarray(pft[idx]),
                fg=jnp.asarray(fg[idx]), wp=jnp.asarray(wp[idx]), fc=jnp.asarray(fc[idx]),
                le=jnp.asarray(le),
                skt=jnp.asarray(mh("skin_temperature")[:, idx]),
                alb=jnp.asarray(np.clip(mh("forecast_albedo")[:, idx], 0.05, 0.85)),
                t0=jnp.asarray(T2[0, idx]), dom_onehot=jnp.asarray(oh),
                w=jnp.cos(jnp.asarray(latc[idx])))


def main():
    jax.config.update("jax_enable_x64", True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diurnal-npz", default="/tmp/era5_diurnal.npz",
                   help="ERA5 monthly-diurnal climatology (scripts/tmp/fetch_era5_diurnal.py)")
    p.add_argument("--n-sub", type=int, default=1100, help="land columns to train on")
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--lam-le", type=float, default=_LAM_LE,
                   help="latent-heat loss weight (ERA5 slhf_wm2 dual target; needs an "
                        "npz fetched with mean_surface_latent_heat_flux)")
    p.add_argument("--lam-tbias-mon", type=float, default=_LAM_TBIAS_MON,
                   help="MONTHLY skin-T mean-bias penalty (seasonal-cycle bias is the "
                        "primary target; RMSE fights natural variability)")
    p.add_argument("--lam-lebias-mon", type=float, default=_LAM_LEBIAS_MON,
                   help="MONTHLY latent-heat mean-bias penalty (0 disables)")
    p.add_argument("--out", default="results/land_tuned_params.json")
    args = p.parse_args()
    data = load_training_data(args.diurnal_npz, args.n_sub)
    # A missing LE field must not silently degrade the dual target to skin-T-only
    # (codex): an all-NaN target zeroes the LE term through the finite-mask, so a
    # run would CLAIM a dual-target calibration it never performed.
    if args.lam_le > 0.0 and not bool(np.isfinite(np.asarray(data["le"])).any()):
        raise SystemExit(
            "npz has no slhf_wm2 (latent-heat target): re-fetch with "
            "scripts/data/fetch_era5_hourly_climatology.py, or pass --lam-le 0 "
            "to explicitly train skin-T-only")
    tuned = train(data, n_iter=args.iters, lam_le=args.lam_le,
                  lam_tbias_mon=args.lam_tbias_mon, lam_lebias_mon=args.lam_lebias_mon)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"# recommended tuned params -> {args.out} (does NOT mutate production defaults)")


if __name__ == "__main__":
    main()

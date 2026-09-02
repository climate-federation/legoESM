"""Calibrated-vs-calibrated tuner: two-leaf +/- P-hydro at the dry EC sites.

Protocol: ``docs/land/phydro_calibrated_comparison_prereg.md`` (pre-registered
before this file existed; both design reviews folded in).  One process tunes
ONE (arm, init) pair by Adam through the prognostic multilayer land scan:

* arm ``beta_theta``: tunes the shared soil/root set
  {root_depth, theta_wp, theta_fc(=wp+gap), soil_evap_litter_resistance_s_m}
* arm ``phydro``: tunes the same four PLUS
  {gplant_mol, minlwp_mpa, gamma_cost, root_biomass_gm2}

Reused machinery (pre-impl search): driver reading + per-site config
construction from ``run_ec_site`` (imported, not copied); the bounded-sigmoid
raw-parameter pattern, per-step ``jax.checkpoint`` scan and inert-leaf gate
from ``train_multilayer_land_era5`` / ``train_land_params_era5``.  Parameters
are spliced into the config INSIDE the loss (traced leaves, SegmentForcing
doctrine) by direct ``_replace`` — the shared ``apply_param_overrides`` layer
is host-side (SystemExit validation) and cannot trace; bounds here are
enforced structurally by the sigmoid transform to the SAME ``__param_spec__``
bounds.

Optimizer: DERIVATIVE-FREE (per-seed Nelder-Mead on the sigmoid-bounded raw
vector).  Gradient descent was the original design; the pre-registered FD gate
REFUTED it before any tuning (measured 2026-09-02, US-Whs 2-train-year window):
the phydro backward pass explodes (analytic dL/draw ~ 1.9e11 vs FD -44 — BPTT
explosion through the year-long scan with the 40-step unrolled hydraulic inner
loop, exactly the GLM design-review Q5 channel), and the beta arm's analytic/FD
ratio was 0.082.  Per the repo AD-verification rule a gradient path that fails
its check must not produce results, so the tuner optimises the SAME loss
forward-only.  The forward still runs WITHOUT the harness NaN-revert; a
non-finite candidate loss is assigned a large penalty (recorded per eval),
and a seed whose BEST point is penalised is marked FAILED.

Usage (one slurm task per invocation)::

    python scripts/run/tune_ec_stress_arms.py --write-year-split   # once, obs-only
    python scripts/run/tune_ec_stress_arms.py --arm phydro --init-id 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "run"))

_spec = importlib.util.spec_from_file_location(
    "_ec", os.path.join(_ROOT, "scripts", "run", "run_ec_site.py"))
_ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ec)

from legoesm import constants  # noqa: E402
from legoesm.land.multilayer_land import (  # noqa: E402
    init_multilayer_land_state, step_multilayer_land_with_diagnostics)

SITES = ["US-SRM", "US-Whs", "US-Ton", "US-Var", "FR-Pue"]
DRV = "/burg-archive/glab/users/jf3423/DifferBESS/data/sitelevel/nc"
OUT_DIR = os.path.join(_ROOT, "results", "ec_stress_tuning")
SPLIT_JSON = os.path.join(OUT_DIR, "year_split.json")
N_TRAIN_YEARS = 2
MIN_TEST_COVER = 0.20

# Tuned parameters: name -> (bounds_lo, bounds_hi), matching __param_spec__
# (land.multilayer / land.phydro).  theta_fc is derived (= theta_wp + gap).
SHARED = {
    "root_depth": (0.33, 3.0),
    "theta_wp": (0.0495, 0.45),
    "fc_gap": (0.02, 0.45),   # theta_fc = theta_wp + gap; keeps fc>wp
    "soil_evap_litter_resistance_s_m": (0.0, 400.0),
}
PHYDRO = {
    "gplant_mol": (5e-4, 2e-2),
    "minlwp_mpa": (-5.0, -0.5),
    "gamma_cost": (0.1, 10.0),
    "root_biomass_gm2": (50.0, 2000.0),
}
DEFAULTS = {
    "root_depth": 1.0, "theta_wp": None, "fc_gap": None,
    "soil_evap_litter_resistance_s_m": 300.0,
    "gplant_mol": 4e-3, "minlwp_mpa": -2.0, "gamma_cost": 1.0,
    "root_biomass_gm2": 500.0,
}


def _bounds(arm):
    b = dict(SHARED)
    if arm == "phydro":
        b.update(PHYDRO)
    return b


def constrain(raw, arm):
    b = _bounds(arm)
    return {k: b[k][0] + (b[k][1] - b[k][0]) * jax.nn.sigmoid(raw[k])
            for k in b}


def _inv(v, lo, hi):
    x = np.clip((v - lo) / (hi - lo), 1e-4, 1 - 1e-4)
    return float(np.log(x / (1 - x)))


def default_raw(arm, site_defaults):
    """Raw vector at the defaults (init-id 0).  theta_wp/fc defaults are the
    per-site texture values; a single global init uses their mean."""
    b = _bounds(arm)
    out = {}
    for k, (lo, hi) in b.items():
        v = DEFAULTS[k]
        if v is None:
            v = site_defaults[k]
        out[k] = _inv(v, lo, hi)
    return out


def seeded_raw(arm, seed):
    rng = np.random.default_rng(seed)
    return {k: float(rng.uniform(-2.0, 2.0)) for k in _bounds(arm)}


# ---------------------------------------------------------------------------
# Year split (obs-only; frozen once)
# ---------------------------------------------------------------------------

def _year_index(d):
    """Calendar-year label per timestep from the driver's doy stream."""
    doy = np.asarray(d.doy).ravel()
    wrap = np.where(np.diff(doy) < -180)[0]
    year = np.zeros(doy.size, dtype=int)
    for w in wrap:
        year[w + 1:] += 1
    return year


def _joint_valid(d):
    """Joint GPP+LE validity from OBS/forcing masks only (codex: never LE-only,
    never model output)."""
    v = np.asarray(d.valid).ravel().astype(bool)
    if d.met_filled is not None:
        v &= np.asarray(d.met_filled).ravel() == 0
    v &= np.isfinite(np.asarray(d.obs["gpp_umol"]).ravel())
    v &= np.isfinite(np.asarray(d.obs["le_wm2"]).ravel())
    return v


def write_year_split():
    os.makedirs(OUT_DIR, exist_ok=True)
    table = {}
    for site in SITES:
        d = _ec.read_ec_site_driver(os.path.join(DRV, f"{site}_driver_v2_gapfree.nc"))
        yr = _year_index(d)
        jv = _joint_valid(d)
        cover = {int(y): float(jv[yr == y].mean()) for y in np.unique(yr)
                 if (yr == y).sum() > 17000}          # full years only
        ranked = sorted(cover, key=lambda y: (-cover[y], y))
        train = sorted(ranked[:N_TRAIN_YEARS])
        test = sorted(y for y in ranked[N_TRAIN_YEARS:]
                      if cover[y] >= MIN_TEST_COVER)
        table[site] = {"train": train, "test": test, "coverage": cover}
        print(f"{site}: train={train} test={test}")
    with open(SPLIT_JSON, "w") as fh:
        json.dump(table, fh, indent=1)
    print(f"-> {SPLIT_JSON}")


# ---------------------------------------------------------------------------
# Per-site assembly (configs + train slices), reusing run_ec_site pieces
# ---------------------------------------------------------------------------

def build_site(site, arm, years):
    d = _ec.read_ec_site_driver(os.path.join(DRV, f"{site}_driver_v2_gapfree.nc"))
    tex = _ec._texture_lookup(d.site_id, _ec._DEFAULT_TEXTURE_CSV)
    phys = _ec.ec_site_physics(d.site_id)
    canopy = _ec.TwoLeafCanopyConfig(
        max_iters=30, stomatal_model="medlyn",
        capacity_scheme="p_model", g1_source="p_model").validate()
    lc = _ec._build_land_config(
        canopy, "auto", "free_drainage", phys.get("soil_depth_m", 0.0),
        root_depth=phys["root_depth"],
        z_ref=phys["z_ref"], texture=tex)
    if arm == "phydro":
        lc = lc._replace(transpiration_stress="phydro")

    yr = _year_index(d)
    sel = np.isin(yr, years)
    sl = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a)[sel]),
                                (d.forcing, d.canopy_params, d.doy))
    jv = _joint_valid(d)[sel]
    obs_g = np.asarray(d.obs["gpp_umol"]).ravel()[sel]
    obs_l = np.asarray(d.obs["le_wm2"]).ravel()[sel]
    # obs arrays carry NaN outside the mask; zero them so masked residuals
    # cannot inject NaN into the loss (weights are zero there anyway).
    obs_g = np.where(jv, obs_g, 0.0)
    obs_l = np.where(jv, obs_l, 0.0)
    var_g = float(np.var(obs_g[jv])) if jv.any() else 1.0
    var_l = float(np.var(obs_l[jv])) if jv.any() else 1.0

    Ts = np.asarray(d.T_soil_top).ravel()
    T_init = float(Ts[np.isfinite(Ts)][0])
    th = np.asarray(d.theta_soil).ravel()
    th0 = th[np.isfinite(th)]
    theta_r = float(lc.hydraulics.theta_r)
    theta_sat = float(lc.hydraulics.theta_sat)
    theta_init = float(np.clip(th0[0], theta_r + 1e-3, theta_sat - 1e-3))
    state0 = init_multilayer_land_state(
        1, lc, T_init=T_init, theta_init=theta_init,
        TgC_init=T_init - constants.T_freeze)
    return dict(lc=lc, xs=sl, state0=state0, dt=d.dt_s,
                mask=jnp.asarray(jv.astype(np.float64)),
                obs_g=jnp.asarray(obs_g), obs_l=jnp.asarray(obs_l),
                var_g=var_g, var_l=var_l,
                site_defaults={"theta_wp": float(lc.theta_wp),
                               "fc_gap": float(lc.theta_fc - lc.theta_wp)})


def apply_params(lc, cp, arm):
    lc = lc._replace(
        root_depth=cp["root_depth"], theta_wp=cp["theta_wp"],
        theta_fc=cp["theta_wp"] + cp["fc_gap"],
        soil_evap_litter_resistance_s_m=cp["soil_evap_litter_resistance_s_m"])
    if arm == "phydro":
        lc = lc._replace(phydro=lc.phydro._replace(
            gplant_mol=cp["gplant_mol"], minlwp_mpa=cp["minlwp_mpa"],
            gamma_cost=cp["gamma_cost"],
            root_biomass_gm2=cp["root_biomass_gm2"]))
    return lc


def site_loss_fn(site_pack, arm):
    """Return loss(raw_params) for ONE site (train years), AD-safe forward
    (NO NaN-revert, NO nudging)."""
    lc0, xs, state0, dt = (site_pack["lc"], site_pack["xs"],
                           site_pack["state0"], site_pack["dt"])
    mask, og, ol = site_pack["mask"], site_pack["obs_g"], site_pack["obs_l"]
    vg, vl = site_pack["var_g"], site_pack["var_l"]

    def loss(raw):
        cp = constrain(raw, arm)
        lc = apply_params(lc0, cp, arm)

        @jax.checkpoint
        def step(state, x):
            f, p, doy = x
            s2, _r, _c, out = step_multilayer_land_with_diagnostics(
                state, f, lc, 1.0, dt, lat=None, carbon_state=None,
                doy=doy, land_params=p)
            return s2, (out.gpp[0], out.lhflx[0])

        _s, (gpp, le) = jax.lax.scan(step, state0, xs)
        # out.gpp is gC m-2 s-1 (canopy GPP = Agross * 12e-6); obs are
        # umolCO2 m-2 s-1 -> same conversion the harness applies.
        gpp = gpp / _ec._GC_PER_UMOL_CO2
        n = jnp.maximum(mask.sum(), 1.0)
        mse_g = jnp.sum(mask * (gpp - og) ** 2) / n / vg
        mse_l = jnp.sum(mask * (le - ol) ** 2) / n / vl
        return 0.5 * mse_g + 0.5 * mse_l

    return loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-year-split", action="store_true")
    ap.add_argument("--arm", choices=["beta_theta", "phydro"])
    ap.add_argument("--init-id", type=int, default=0,
                    help="0 = defaults init; 1..5 = random seeds")
    ap.add_argument("--evals", type=int, default=80,
                    help="Nelder-Mead function-evaluation budget")
    ap.add_argument("--sites", nargs="+", default=SITES)
    args = ap.parse_args()

    if args.write_year_split:
        write_year_split()
        return 0
    if not args.arm:
        raise SystemExit("--arm required (or --write-year-split)")

    with open(SPLIT_JSON) as fh:
        split = json.load(fh)

    packs = {s: build_site(s, args.arm, split[s]["train"])
             for s in args.sites}
    losses = {s: site_loss_fn(p, args.arm) for s, p in packs.items()}

    def total_loss(raw):
        return jnp.mean(jnp.stack([losses[s](raw) for s in args.sites]))

    site_defaults = {
        "theta_wp": float(np.mean([p["site_defaults"]["theta_wp"]
                                   for p in packs.values()])),
        "fc_gap": float(np.mean([p["site_defaults"]["fc_gap"]
                                 for p in packs.values()])),
    }
    raw = (default_raw(args.arm, site_defaults) if args.init_id == 0
           else seeded_raw(args.arm, args.init_id))

    keys = sorted(_bounds(args.arm))
    jit_loss = jax.jit(total_loss)
    _PENALTY = 1.0e3   # non-finite forward -> recorded penalty, never masked

    n_eval = [0]
    history = []

    def f_vec(x):
        rawd = {k: float(v) for k, v in zip(keys, x)}
        v = float(jit_loss(rawd))
        finite = np.isfinite(v)
        v = v if finite else _PENALTY
        n_eval[0] += 1
        history.append({"eval": n_eval[0], "loss": v, "finite": bool(finite),
                        **{k: float(c) for k, c in
                           constrain(rawd, args.arm).items()}})
        print(f"eval {n_eval[0]:3d} loss={v:.5f} finite={finite}", flush=True)
        return v

    from scipy.optimize import minimize
    x0 = np.array([raw[k] for k in keys])
    res = minimize(f_vec, x0, method="Nelder-Mead",
                   options={"maxfev": args.evals, "xatol": 1e-2,
                            "fatol": 1e-4, "adaptive": True})
    raw = {k: float(v) for k, v in zip(keys, res.x)}
    best = min(history, key=lambda h: h["loss"])
    status = ("ok" if best["finite"] else "FAILED_best_is_penalised")
    print(f"best eval={best['eval']} loss={best['loss']:.5f} status={status}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "arm": args.arm, "init_id": args.init_id, "status": status,
        "optimizer": "nelder-mead (AD refuted by the FD gate; see docstring)",
        "eval_history": history,
        "raw": {k: float(v) for k, v in raw.items()},
        "constrained": {k: float(v)
                        for k, v in constrain(raw, args.arm).items()},
        "sites": args.sites, "year_split": SPLIT_JSON,
        "evals": args.evals,
    }
    path = os.path.join(OUT_DIR, f"{args.arm}_init{args.init_id}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"-> {path}  status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

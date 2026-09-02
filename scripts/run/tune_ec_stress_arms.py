"""Calibrated-vs-calibrated tuner: two-leaf +/- P-hydro at the dry EC sites.

Protocol: ``docs/land/phydro_calibrated_comparison_prereg.md`` (pre-registered
before this file existed; both design reviews folded in).  One process tunes
ONE (arm, init) pair through the prognostic multilayer land scan (5 site
forwards in parallel worker processes):

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
    python scripts/run/tune_ec_stress_arms.py --arm phydro --init-id 3 --evaluate
    python scripts/run/tune_ec_stress_arms.py --arm phydro --init-id -1 --evaluate  # untuned defaults
    python scripts/run/tune_ec_stress_arms.py --verdict

``--evaluate`` scores one (arm, init) cell on the TEST years only, one forward
per site-year, and writes ``<arm>_init<k>_test.json`` (per-site per-year joint
loss).  ``--verdict`` applies the pre-registered rule mechanically to every
test file present: seed SELECTION on train loss; HEADLINE = the selected
seeds paired per-site-per-year (A_tuned - B_tuned)/A_tuned with a 5% relative
margin and >= 4/5 site sign consistency (all five sites required, any
penalised paired site-year withholds the verdict); matched-seed population
and seed spread reported as secondary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

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
SPLIT_JSON = os.path.join(_ROOT, "docs", "land", "phydro_calibrated_year_split.json")  # committed (prereg)
N_TRAIN_YEARS = 2
MIN_TEST_COVER = 0.10   # prereg deviation 2: 0.20 left US-Whs with no test year

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
    if jv.sum() < 2:
        raise SystemExit(f"{site} years={years}: <2 joint-valid samples")
    var_g = float(np.var(obs_g[jv]))
    var_l = float(np.var(obs_l[jv]))

    # IC from the SELECTED slice only (codex P0: whole-driver first value
    # could come from a held-out year)
    Ts = np.asarray(d.T_soil_top).ravel()[sel]
    T_init = float(Ts[np.isfinite(Ts)][0])
    th = np.asarray(d.theta_soil).ravel()[sel]
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


def _result_path(arm, init_id, suffix=""):
    tag = "default" if init_id < 0 else f"init{init_id}"
    return os.path.join(OUT_DIR, f"{arm}_{tag}{suffix}.json")


def evaluate(arm, init_id, split, sites):
    """Score one cell on TEST years: one forward per (site, year)."""
    if init_id < 0:
        # untuned defaults = what the harness runs: PER-SITE texture wp/fc,
        # everything else at config defaults (raw resolved per site below)
        raw = None
        train_loss = None
    else:
        with open(_result_path(arm, init_id)) as fh:
            tuned = json.load(fh)
        if tuned["status"] != "ok":
            raise SystemExit(f"{arm} init{init_id} status={tuned['status']}; "
                             "not evaluated")
        raw = tuned["raw"]
        train_loss = min(h["loss"] for h in tuned["eval_history"] if h["finite"])
    per_site = {}
    for s in sites:
        per_site[s] = {}
        for y in split[s]["test"]:
            pack = build_site(s, arm, [y])
            r = raw if raw is not None else default_raw(arm, pack["site_defaults"])
            v = float(jax.jit(site_loss_fn(pack, arm))(r))
            per_site[s][str(y)] = v if np.isfinite(v) else None
            print(f"{arm} init{init_id} {s} {y}: loss={v:.5f}", flush=True)
    out = {"arm": arm, "init_id": init_id, "train_loss": train_loss,
           "constrained": ({k: float(v) for k, v in constrain(raw, arm).items()}
                           if raw is not None else "per-site defaults"),
           "test_loss": per_site}
    path = _result_path(arm, init_id, "_test")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"-> {path}")


def verdict(margin=0.05, min_sites_consistent=4):
    """Pre-registered verdict rule, applied to whatever test files exist."""
    import glob
    cells = {}
    for p in glob.glob(os.path.join(OUT_DIR, "*_test.json")):
        with open(p) as fh:
            r = json.load(fh)
        cells.setdefault(r["arm"], {})[r["init_id"]] = r

    def flat(r):
        # a non-finite test forward is penalised (+inf), never masked
        return {(s, y): (np.inf if v is None else v)
                for s, ys in r["test_loss"].items() for y, v in ys.items()}

    def site_mean(r):
        return float(np.mean(list(flat(r).values())))

    def paired(ra, rb):
        """Relative paired differences (A-B)/A per site-year, >0 = B better;
        both penalised -> 0, one penalised -> +/-1 (capped)."""
        fa, fb = flat(ra), flat(rb)
        keys = sorted(set(fa) & set(fb))
        rel = {}
        for k in keys:
            if np.isinf(fa[k]) and np.isinf(fb[k]):
                rel[k] = 0.0
            elif np.isinf(fb[k]):
                rel[k] = -1.0
            elif np.isinf(fa[k]):
                rel[k] = 1.0
            else:
                rel[k] = (fa[k] - fb[k]) / fa[k]
        return rel

    def rule(rel):
        med = float(np.median(list(rel.values())))
        sites = sorted({s for s, _y in rel})
        sign = {s: int(np.sign(np.median([v for (ss, _y), v in rel.items()
                                          if ss == s]))) for s in sites}
        n_pos = sum(v > 0 for v in sign.values())
        n_neg = sum(v < 0 for v in sign.values())
        return med, n_pos, n_neg

    summary = {}
    for arm, by_init in cells.items():
        tuned = {k: r for k, r in by_init.items() if k >= 0}
        # selection over ALL ok tuning artifacts (not only those already
        # scored on test), then require the selected seed's test artifact
        train_all = {}
        for p in glob.glob(os.path.join(OUT_DIR, f"{arm}_init*.json")):
            if p.endswith("_test.json"):
                continue
            with open(p) as fh:
                t = json.load(fh)
            if t["status"] == "ok":
                train_all[t["init_id"]] = min(
                    h["loss"] for h in t["eval_history"] if h["finite"])
        sel = min(train_all, key=train_all.get) if train_all else None
        if sel is not None and sel not in tuned:
            raise SystemExit(f"{arm}: best-train seed {sel} has no test "
                             "artifact; run --evaluate for it first")
        summary[arm] = {
            "default_test": site_mean(by_init[-1]) if -1 in by_init else None,
            "tuned_seed_test": {k: site_mean(r) for k, r in tuned.items()},
            "tuned_seed_train": {k: r["train_loss"] for k, r in tuned.items()},
            "selected_seed": sel,
            "selected_test": site_mean(tuned[sel]) if sel is not None else None,
            "seed_median_test": (float(np.median([site_mean(r) for r in tuned.values()]))
                                 if tuned else None),
            "n_penalised_site_years": {k: int(sum(np.isinf(v) for v in flat(r).values()))
                                       for k, r in by_init.items()},
        }
    print(json.dumps(summary, indent=1))
    a, b = summary.get("beta_theta", {}), summary.get("phydro", {})
    if a.get("selected_seed") is None or b.get("selected_seed") is None:
        print("VERDICT: incomplete (need >=1 ok tuned seed per arm)")
        return
    ra = cells["beta_theta"][a["selected_seed"]]
    rb = cells["phydro"][b["selected_seed"]]
    rel = paired(ra, rb)
    covered = sorted({s for s, _y in rel})
    if covered != sorted(SITES):
        print(f"VERDICT: INCOMPLETE — paired test coverage only for {covered}; "
              f"the >=4/5 rule needs all of {SITES}")
        return
    n_pen = sum(np.isinf(flat(ra)[k]) or np.isinf(flat(rb)[k]) for k in rel)
    if n_pen:
        print(f"VERDICT: WITHHELD — {n_pen} paired site-years penalised "
              "(non-finite forward) in the selected seeds")
        return
    med, n_pos, n_neg = rule(rel)
    # site-level cluster bootstrap of the median (sites are the independent
    # units; seeds fixed at the selected ones)
    sites = sorted({s for s, _y in rel})
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        pick = rng.choice(sites, size=len(sites), replace=True)
        vals = [v for s in pick for (ss, _y), v in rel.items() if ss == s]
        boots.append(np.median(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    from scipy.stats import binomtest
    p_sign = binomtest(max(n_pos, n_neg), len(sites), 0.5, alternative="greater").pvalue
    print(f"HEADLINE selected seeds A={a['selected_seed']} B={b['selected_seed']}: "
          f"median (A-B)/A = {med:+.4f} [site-bootstrap 95% {lo:+.4f},{hi:+.4f}, "
          f"descriptive only at n={len(sites)}] over {len(rel)} site-years; "
          f"sites B better={n_pos} A better={n_neg} (exact sign test p={p_sign:.3f})")
    # secondary: seed population, matched seeds (same init id both arms)
    common = sorted(k for k in cells["beta_theta"] if k >= 0 and k in cells["phydro"])
    pop = [rule(paired(cells["beta_theta"][k], cells["phydro"][k]))[0] for k in common]
    if pop:
        print(f"SECONDARY matched-seed medians (A-B)/A: "
              + " ".join(f"seed{k}={m:+.4f}" for k, m in zip(common, pop))
              + f"  -> median over seeds {np.median(pop):+.4f}")
    win = med > margin and n_pos >= min_sites_consistent
    lose = med < -margin and n_neg >= min_sites_consistent
    print("VERDICT: " + ("P-hydro WINS (tuned vs tuned)" if win else
                         "beta_theta WINS (tuned vs tuned)" if lose else
                         "no demonstrated advantage of the hydraulic optimum "
                         "over the tuned empirical stress at these sites"))


_SIMPLEX_WIDTH_RAW = 1.0


def _site_worker(conn, site, arm, years):
    import traceback
    try:
        pack = build_site(site, arm, years)
        f = jax.jit(site_loss_fn(pack, arm))
        conn.send(("ok", pack["site_defaults"]))
        while True:
            raw = conn.recv()
            if raw is None:
                return
            conn.send(("ok", float(f(raw))))
    except BaseException:
        conn.send(("err", f"{site}: {traceback.format_exc()}"))
        raise


def _recv(conn):
    tag, payload = conn.recv()
    if tag != "ok":
        raise SystemExit(f"site worker failed:\n{payload}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-year-split", action="store_true")
    ap.add_argument("--evaluate", action="store_true",
                    help="score (arm, init) on TEST years; init -1 = untuned")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--arm", choices=["beta_theta", "phydro"])
    ap.add_argument("--init-id", type=int, default=0, choices=range(-1, 6),
                    help="0 = defaults init (global-mean texture wp/fc); "
                         "1..5 = random seeds; -1 = untuned per-site "
                         "defaults (--evaluate only)")
    ap.add_argument("--evals", type=int, default=80,
                    help="Nelder-Mead function-evaluation budget")
    ap.add_argument("--sites", nargs="+", default=SITES)
    args = ap.parse_args()

    if args.write_year_split:
        write_year_split()
        return 0
    if args.verdict:
        verdict()
        return 0
    if not args.arm:
        raise SystemExit("--arm required (or --write-year-split / --verdict)")

    with open(SPLIT_JSON) as fh:
        split = json.load(fh)

    if args.evaluate:
        evaluate(args.arm, args.init_id, split, args.sites)
        return 0
    if args.init_id < 0:
        raise SystemExit("--init-id -1 only valid with --evaluate")

    # One worker process per site: the 5 site forwards are independent, so
    # each eval costs one site-forward of wall-clock instead of five.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    workers = []
    for s in args.sites:
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_site_worker,
                        args=(child, s, args.arm, split[s]["train"]))
        p.start()
        workers.append((parent, p))
    site_defaults_all = [_recv(c) for c, _p in workers]

    def total_loss(raw):
        for c, _p in workers:
            c.send(raw)
        return float(np.mean([_recv(c) for c, _p in workers]))

    site_defaults = {
        k: float(np.mean([d[k] for d in site_defaults_all]))
        for k in ("theta_wp", "fc_gap")}
    raw = (default_raw(args.arm, site_defaults) if args.init_id == 0
           else seeded_raw(args.arm, args.init_id))

    keys = sorted(_bounds(args.arm))

    n_eval = [0]
    history = []
    t0 = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.arm}_init{args.init_id}.json")

    def write_artifact(status, res=None):
        """Written after EVERY eval (status 'running') so a walltime kill
        still leaves the best point so far; final call sets the verdict."""
        finite_hist = [h for h in history if h["finite"]]
        best = min(finite_hist, key=lambda h: h["loss"]) if finite_hist else None
        raw_best = best["raw"] if best else None
        out = {
            "arm": args.arm, "init_id": args.init_id, "status": status,
            "optimizer": "nelder-mead (AD refuted by the FD gate; see docstring)",
            "converged": bool(res.success) if res is not None else None,
            "scipy_message": str(res.message) if res is not None else None,
            "n_nonfinite": len(history) - len(finite_hist),
            "n_evals": len(history), "evals": args.evals,
            "best_eval": best["eval"] if best else None,
            "best_loss": best["loss"] if best else None,
            "raw": raw_best,
            "constrained": ({k: float(v) for k, v in
                             constrain(raw_best, args.arm).items()}
                            if raw_best else None),
            "eval_history": history,
            "sites": args.sites, "year_split": SPLIT_JSON,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, path)
        return best

    def f_vec(x):
        rawd = {k: float(v) for k, v in zip(keys, x)}
        v = total_loss(rawd)
        finite = bool(np.isfinite(v))
        n_eval[0] += 1
        history.append({"eval": n_eval[0], "loss": v if finite else None,
                        "finite": finite, "raw": rawd,
                        **{k: float(c) for k, c in
                           constrain(rawd, args.arm).items()}})
        print(f"eval {n_eval[0]:3d} loss={v:.5f} finite={finite} "
              f"t={time.time() - t0:.0f}s", flush=True)
        write_artifact("running")
        # non-finite forward -> +inf: always the worst vertex, never a
        # candidate optimum (a finite loss can exceed any fixed penalty)
        return v if finite else np.inf

    from scipy.optimize import minimize
    x0 = np.array([raw[k] for k in keys])
    # initial simplex of width 1.0 in raw space (~25% of a parameter's range
    # at mid-range); scipy's default 5%-of-x0 steps cannot move materially
    simplex = np.vstack([x0] + [x0 + _SIMPLEX_WIDTH_RAW * np.eye(len(keys))[i]
                                for i in range(len(keys))])
    res = minimize(f_vec, x0, method="Nelder-Mead",
                   options={"maxfev": args.evals, "xatol": 1e-2,
                            "fatol": 1e-4, "adaptive": True,
                            "initial_simplex": simplex})
    for c, p in workers:
        c.send(None)
        p.join()
    status = ("ok" if any(h["finite"] for h in history)
              else "FAILED_no_finite_eval")
    best = write_artifact(status, res) or {"eval": -1, "loss": float("nan")}
    print(f"best eval={best['eval']} loss={best['loss']:.5f} status={status} "
          f"nonfinite={sum(not h['finite'] for h in history)}/{len(history)} "
          f"converged={res.success} ({res.message})")
    print(f"-> {path}  status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

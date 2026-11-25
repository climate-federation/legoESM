#!/usr/bin/env python
"""Round-2 equation search: screen, fit and score closed-form phi candidates on the cached benchmark-v5 samples.

Candidates (JSON list) give, per kind m or h, a stable expression in z = zeta > 0 and an unstable expression in
x = -zeta > 0, both jax.numpy expressions of named parameters (namespace: jnp, z or x, the parameters).
Fit and score copy candidate_formulations / standard_formulations exactly: log(phi_obs/phi) least squares over
samples with phi_obs > 0, each site weighted 1/sqrt(n_site) over both regimes of the kind, bounded trust region,
fit on odd UTC days; score = per-site gradient log-RMSE on even days over samples where obs and model share a sign;
rank among the 12 standard schemes' even-day scores read from the v5 json (1 = better than all of them).
--check reproduces dyer1974 and J3_global v5 scores from the cache before anything else is trusted.
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)

PANELS = ("m_stable", "m_unstable", "h_stable", "h_unstable")
ZETA_DOMAIN = 10.0
GRID = np.logspace(-6, 1, 701)  # 1e-6 .. 10 per side
FINE_GRID = np.logspace(-6, 1, 70001)
KINK_FRACTION = 0.5
FAIL_RESIDUAL = 1e3
MAX_PARAMS_PER_SIDE = 3
N_START_FRACTIONS = (0.25, 0.5, 0.75)
N_BOOT = 2000
SCORE_TIE = 1e-9  # log-RMSE differences below this are rounding, not a win
DYER = {"m": {"stable": "1 + 5*z", "unstable": "(1 + 16*x)**-0.25", "params": {}},
        "h": {"stable": "1 + 5*z", "unstable": "(1 + 16*x)**-0.5", "params": {}}}


def build(spec):
    """(f(zeta, p) for both sides, param names, per-side param names)."""
    names = list(spec["params"])
    used = {s: [n for n in names if _uses(spec[s], n)] for s in ("stable", "unstable")}
    st = eval("lambda z, " + ", ".join(names or ["_u"]) + ": " + spec["stable"], {"jnp": jnp})  # noqa: S307
    un = eval("lambda x, " + ", ".join(names or ["_u"]) + ": " + spec["unstable"], {"jnp": jnp})  # noqa: S307

    def f(zeta, p):
        p = list(p) or [0.0]
        pos = jnp.where(zeta > 0, zeta, 1.0)
        neg = jnp.where(zeta <= 0, -zeta, 1.0)  # zeta = 0 evaluates the unstable form at x = 0
        return jnp.where(zeta > 0, st(pos, *p), un(neg, *p))
    f.sides = (st, un)
    return f, names, used


def _uses(expr, name):
    import re
    return re.search(rf"\b{name}\b", expr) is not None


def screen(spec, p, kind):
    """Mechanical screen at fitted parameters. Returns dict of checks, 'pass', and asymptotic exponents."""
    f, names, used = build(spec)
    g = jax.vmap(jax.grad(lambda z: f(z, p)))
    out = {}
    for side, sign in (("stable", 1.0), ("unstable", -1.0)):
        z = sign * GRID
        v = np.asarray(jax.vmap(lambda q: f(q, p))(jnp.asarray(z)))
        s = np.asarray(g(jnp.asarray(z)))
        slope_x = s * sign  # d phi / d |zeta|
        out[f"{side}_finite_positive"] = bool(np.all(np.isfinite(v)) and np.all(v > 0))
        out[f"{side}_monotone"] = bool(np.all(slope_x >= -1e-12) if side == "stable" else np.all(slope_x <= 1e-12))
        lead = np.asarray(g(jnp.asarray(sign * np.array([1e-7, 1e-9, 1e-11]))))
        out[f"{side}_finite_slope_at_0"] = bool(np.all(np.isfinite(lead)) and abs(lead[2] - lead[1]) <= max(abs(lead[1] - lead[0]), 1e-6))
        # C1 by refinement: a slope jump keeps its size on a 100x finer grid, a smooth slope change shrinks 100x
        fine = np.asarray(g(jnp.asarray(sign * FINE_GRID)))
        step = np.abs(np.diff(fine))
        cell = np.searchsorted(GRID, FINE_GRID[1:], side="right") - 1
        max_fine = np.zeros(len(GRID) - 1)
        np.maximum.at(max_fine, np.clip(cell, 0, len(GRID) - 2), step)
        coarse = np.abs(np.diff(s))
        kink = (coarse > 1e-6 * (1 + np.abs(s[1:]))) & (max_fine > KINK_FRACTION * coarse)
        out[f"{side}_c1"] = bool(np.all(np.isfinite(fine)) and not kink.any())
        out[f"{side}_at_0"] = float(v[0])
        out[f"{side}_exponent_at_10"] = float(np.log(v[-1] / v[-10]) / np.log(np.abs(z[-1]) / np.abs(z[-10])))
        out[f"{side}_n_params"] = len(used[side])
    st, un = f.sides
    pp = list(p) or [0.0]
    out["stable_at_0"], out["unstable_at_0"] = float(st(0.0, *pp)), float(un(0.0, *pp))  # exactly at zeta = 0
    out["continuous_at_0"] = abs(out["stable_at_0"] - out["unstable_at_0"]) < 1e-9
    out["phi0_ok"] = kind == "h" or abs(out["stable_at_0"] - 1.0) < 1e-5
    out["pass"] = all(out[k] for k in out if k.endswith(("positive", "monotone", "slope_at_0", "c1"))) \
        and out["continuous_at_0"] and out["phi0_ok"] \
        and all(out[f"{s}_n_params"] <= MAX_PARAMS_PER_SIDE for s in ("stable", "unstable"))
    return out


def load(npz):
    d = np.load(npz)
    sites = sorted({k.split("|")[0] for k in d.files})
    get = lambda s, k, par, f: d[f"{s}|{k}|{par}|{f}"] if f"{s}|{k}|{par}|{f}" in d.files else np.empty(0)  # noqa: E731
    return {s: {k: {par: (get(s, k, par, "zeta"), get(s, k, par, "phi")) for par in (0, 1)} for k in "mh"} for s in sites}


def fit(spec, kind, data, min_count, restrict=None):
    """Global fit over all sites of one kind (both regimes), odd days. restrict: 'stable'/'unstable' to fit one side."""
    f, names, used = build(spec)
    if not names:
        return np.zeros(0), {"status": "no params"}
    if restrict is not None:
        idle = [n for n in names if n not in used[restrict]]
        if idle:
            raise ValueError(f"parameters {idle} are not used on the fitted {restrict} side")
    blocks = []
    for s, v in data.items():
        z, ph = v[kind][1]
        sel = ph > 0
        if restrict == "stable":
            sel &= z > 0
        elif restrict == "unstable":
            sel &= z < 0
        if sel.sum() >= min_count:
            blocks.append((jnp.asarray(z[sel]), np.log(ph[sel]), 1.0 / np.sqrt(sel.sum())))
    fj = jax.jit(f)

    def res(p):
        out = []
        for z, lph, w in blocks:
            r = lph - np.log(np.asarray(fj(z, jnp.asarray(p))))
            out.append(np.where(np.isfinite(r), r * w, FAIL_RESIDUAL))
        return np.concatenate(out)
    if not blocks:
        return np.full(len(names), np.nan), {"success": False, "reason": "no data"}
    lo = np.array([spec["params"][n][1] for n in names], float)
    hi = np.array([spec["params"][n][2] for n in names], float)
    starts = [np.array([spec["params"][n][0] for n in names], float)] + [lo + q * (hi - lo) for q in N_START_FRACTIONS]
    best = None
    for x0 in starts:  # multi-start (codex review 1): keep the lowest converged cost
        r = least_squares(res, np.clip(x0, lo, hi), bounds=(lo, hi))
        if r.success and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        return np.full(len(names), np.nan), {"success": False, "reason": "no start converged"}
    at_bound = [bool(abs(x - a) < 1e-6 * max(1, abs(a)) or abs(x - b) < 1e-6 * max(1, abs(b))) for x, a, b in zip(best.x, lo, hi)]
    return best.x, {"success": True, "cost": float(best.cost), "at_bound": at_bound, "n_starts": len(starts)}


def log_rmse(obs, pred):
    ok = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[ok], pred[ok]
    same = o * p > 0
    return float(np.sqrt(np.mean(np.log(o[same] / p[same]) ** 2))) if same.any() else np.nan


def score(spec, p, kind, data, min_count, parity=0):
    f, _, _ = build(spec)
    fj = jax.jit(f)
    out = {}
    for s, v in data.items():
        z, ph = v[kind][parity]
        for regime, sel in (("stable", z > 0), ("unstable", z < 0)):
            if sel.sum() < min_count:
                continue
            out.setdefault(f"{kind}_{regime}", {})[s] = log_rmse(ph[sel], np.asarray(fj(jnp.asarray(z[sel]), jnp.asarray(p))))
    return out


def rank(scores, bench):
    """Per panel: median rank among the 12 standard schemes, wins, median gain vs dyer and vs J3_global."""
    std = bench["standard"]
    out = {}
    for panel, per_site in scores.items():
        ranks, wins, vd, vj = [], 0, [], []
        for s, v in per_site.items():
            sc = bench["sites"][s]["offset_corrected"].get(panel, {})
            if not np.isfinite(sc.get("dyer1974", {}).get("gradient", {}).get("log_rmse", np.nan)) or not np.isfinite(v):
                continue
            stdv = [sc[k]["gradient"]["log_rmse"] for k in std if np.isfinite(sc[k]["gradient"]["log_rmse"])]
            r = 1 + sum(x < v for x in stdv)
            ranks.append(r)
            wins += r == 1
            vd.append(v - sc["dyer1974"]["gradient"]["log_rmse"])
            vj.append(v - sc["J3_global"]["gradient"]["log_rmse"])
        if ranks:
            rng = np.random.default_rng(0)
            ci = lambda v: np.percentile([np.median(rng.choice(v, len(v))) for _ in range(N_BOOT)], [2.5, 97.5]).tolist()  # noqa: E731
            out[panel] = {"median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)), "wins": wins, "n": len(ranks),
                          "med_vs_dyer": float(np.median(vd)), "med_vs_J3g": float(np.median(vj)),
                          "frac_sites_better_than_dyer": float(np.mean(np.array(vd) < -SCORE_TIE)),
                          "ci95_vs_dyer": ci(np.array(vd)), "ci95_vs_J3g": ci(np.array(vj))}
    return out


def check(data, bench, min_count):
    """Reproduce v5 even-day scores of dyer1974 and J3_global from the cache."""
    j3 = bench["global_fit"]["J3"]
    j3m = {"stable": "1 + a_inf*z + (a0 - a_inf)*z/(1 + z/zc)", "unstable": "(1 + 16*x)**-0.25",
           "params": {"a0": [j3["a0"], 0, 99], "a_inf": [j3["a_inf"], 0, 99], "zc": [j3["zc"], 0, 99]}}
    j3h = {"stable": "pr0 + a_inf*z + (a0 - a_inf)*z/(1 + z/zc)", "unstable": "pr0*(1 + 16*x)**-0.5",
           "params": {**j3m["params"], "pr0": [j3["pr0"], 0, 9]}}
    worst, n, seen = 0.0, 0, set()
    for name, kind, spec in (("dyer1974", "m", DYER["m"]), ("dyer1974", "h", DYER["h"]), ("J3_global", "m", j3m), ("J3_global", "h", j3h)):
        p = [spec["params"][n][0] for n in spec["params"]]
        for panel, per_site in score(spec, p, kind, data, min_count).items():
            for s, v in per_site.items():
                seen.add((name, panel, s))
                ref = bench["sites"][s]["offset_corrected"].get(panel, {}).get(name, {}).get("gradient", {}).get("log_rmse", np.nan)
                if np.isfinite(ref) or np.isfinite(v):
                    d = abs(v - ref) / max(abs(ref), 1e-12)
                    worst = max(worst, d) if np.isfinite(d) else np.inf  # a score present on one side only fails
                    n += 1
                    if not abs(v - ref) <= 1e-6 * abs(ref):
                        print("MISMATCH", name, panel, s, v, ref)
    for s, v in bench["sites"].items():  # a benchmark score the cache never produced also fails
        for panel in PANELS:
            for name in ("dyer1974", "J3_global"):
                ref = v.get("offset_corrected", {}).get(panel, {}).get(name, {}).get("gradient", {}).get("log_rmse", np.nan)
                if np.isfinite(ref) and (name, panel, s) not in seen:
                    print("MISSING", name, panel, s)
                    worst = np.inf
    print(f"CHECK worst relative difference vs v5 json: {worst:.2e} over {n} scores")
    return worst if n else np.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--candidates")
    ap.add_argument("--out")
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    np.seterr(all="ignore")
    data = load(a.samples)
    bench = json.load(open(a.bench))
    if a.check or a.candidates:  # ranking reads the benchmark's scores, so the cache must reproduce them first
        if not check(data, bench, a.min_count) <= 1e-6:
            raise SystemExit("cache does not reproduce v5")
    if not a.candidates:
        return
    cands = json.load(open(a.candidates))
    result = {}
    for c in cands:
        name, kind = c["name"], c["kind"]
        entry = {"kind": kind, "spec": c, "author": c.get("author", "")}
        try:
            p, diag = fit(c, kind, data, a.min_count, restrict=c.get("restrict"))
            entry.update(params=dict(zip(c["params"], map(float, p))), fit=diag)
            if not diag["success"]:
                raise RuntimeError(f"fit failed: {diag}")
            entry["screen"] = screen(c, p, kind)
            entry["scores"] = score(c, p, kind, data, a.min_count)
            entry["rank"] = rank(entry["scores"], bench)
        except Exception as e:  # a candidate that does not evaluate is a rejection, recorded
            entry["error"] = repr(e)
        result[name] = entry
        print(name, kind, entry.get("params"), entry.get("screen", {}).get("pass"),
              {k: (v["median_rank"], round(v["med_vs_dyer"], 4), round(v["med_vs_J3g"], 4)) for k, v in entry.get("rank", {}).items()},
              entry.get("error", ""), flush=True)
    json.dump(result, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

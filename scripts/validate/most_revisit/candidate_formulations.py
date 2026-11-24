#!/usr/bin/env python
"""Co-scientist survivors J3, J17, J5 fitted to towers and scored beside the standard model schemes (benchmark v3).

Forms and pairings from coscientist/cs_final.md and the J3/J17/J5 reflection prompts:
  J3  stable phi_m = 1 + a_inf z + (a0 - a_inf) z/(1 + z/zc); phi_h = Pr0 + (phi_m - 1);
      unstable Businger-Dyer gamma 16, heat Pr0 (1 - 16 z)^(-1/2) so phi_h is continuous at 0.
  J17 unstable phi_m = (1 + 16x)^(-1/4) (1 + x/lambda)^(-1/12), x = -z; no heat form; rest Dyer 1974.
  J5  stable phi_m = 1 + b z + a z^2 exp(-z), b >= max_z[-a (2z - z^2) e^-z] (monotone); rest Dyer 1974.
Fits (GLM+codex vote, coscientist/vote_bench3*): least squares of log(phi_obs/phi) over single samples with
phi_obs > 0, bounded trust region; global fit weights every site equally; J3 momentum first, then Pr0 on heat.
Odd UTC days fit, even days score; every scheme is scored on the same even-day samples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import flux_gradient_sites as fs
import standard_formulations as sf

GAMMA = 16.0  # Businger-Dyer unstable coefficient used by every pairing below
# extrema of (2z - z^2) e^-z on [0, 10): stationary points 2 -+ sqrt(2) and the end points
J5_EXTREMA = np.array([0.0, 2.0 - np.sqrt(2.0), 2.0 + np.sqrt(2.0), sf.ZETA_DOMAIN])
J5_SHAPE = (2 * J5_EXTREMA - J5_EXTREMA**2) * np.exp(-J5_EXTREMA)
COND_MAX = 1e8  # Jacobian condition number above which a fitted parameter set is marked unidentified
BOUND_TOL = 1e-6  # relative distance to a box bound counted as "at the bound" (constrained fit)
assert jax.config.jax_enable_x64, "slope probes at 1e-12 need float64 (set by standard_formulations at import)"
MONOTONE_GRID = np.logspace(-4, np.log10(sf.ZETA_DOMAIN * 0.999), 400)

# name: (parameter names, start, lower, upper, kinds fitted, regime fitted)
FORMS = {
    "J3": (("a0", "a_inf", "zc", "pr0"), (5.0, 3.0, 1.0, 1.0), (0.5, 0.0, 0.05, 0.5), (20.0, 20.0, 20.0, 1.5)),
    "J17": (("lambda",), (1.0,), (0.05,), (100.0,)),
    "J5": (("a", "b_extra"), (0.0, 5.0), (-20.0, 0.0), (20.0, 20.0)),
}


def j5_b(a, b_extra):
    """b = max_z[-a (2z - z^2) e^-z] + b_extra over 0 <= z < 10 keeps phi_m monotone (codex vote); exact extrema."""
    return float(np.max(-a * J5_SHAPE)) + b_extra


def _dyer_m(z):
    return jnp.power(jnp.maximum(1.0 - GAMMA * z, 1.0), -0.25)


def make(form, p):
    """(phi_m, phi_h) scalar JAX functions for one parameter set."""
    if form == "J3":
        a0, a_inf, zc, pr0 = p

        def stable_m(z):
            return 1.0 + a_inf * z + (a0 - a_inf) * z / (1.0 + z / zc)

        return (lambda z: jnp.where(z > 0, stable_m(jnp.maximum(z, 0.0)), _dyer_m(z)),
                lambda z: jnp.where(z > 0, pr0 + stable_m(jnp.maximum(z, 0.0)) - 1.0,
                                    pr0 * jnp.power(jnp.maximum(1.0 - GAMMA * z, 1.0), -0.5)))
    if form == "J17":
        (lam,) = p
        x = lambda z: jnp.maximum(-z, 0.0)  # noqa: E731
        return (lambda z: jnp.where(z > 0, 1.0 + 5.0 * z, (1 + GAMMA * x(z)) ** -0.25 * (1 + x(z) / lam) ** (-1.0 / 12.0)),
                sf._DYER["h"])
    if form == "J5":
        a, b_extra = p
        b = j5_b(a, b_extra)
        zp = lambda z: jnp.maximum(z, 0.0)  # noqa: E731
        return (lambda z: jnp.where(z > 0, 1.0 + b * zp(z) + a * zp(z) ** 2 * jnp.exp(-zp(z)), _dyer_m(z)), sf._DYER["h"])
    raise ValueError(f"unknown form {form!r}")


def register(name, form, p, note):
    m, h = make(form, p)
    sf.SCHEMES[name] = (m, h, sf.KAPPA, note, "coscientist/cs_final.md")
    if form in ("J17", "J5"):  # heat is legoESM's Dyer, which carries its +-10 psi clip
        sf.BRANCH_POINTS[name] = {"h": sf.BRANCH_POINTS["dyer1974"]["h"]}
    for kind in "mh":
        sf._VECTOR.pop((name, kind), None)


def _residuals(form, p, data):
    """data: one list of (kind, zeta, phi) blocks per site; each site's blocks share weight 1/sqrt(n_site).

    A non-finite model value is a failure, scored as a large residual, never as a perfect fit."""
    m, h = make(form, p)  # fitted branches are elementwise, so arrays go straight through without a recompile
    out = []
    for blocks in data:
        n = sum(len(z) for _, z, _ in blocks)
        for kind, zeta, ph in blocks:
            model = np.asarray((m if kind == "m" else h)(jnp.asarray(zeta)))
            r = np.log(ph / model)
            out.append(np.where(np.isfinite(r), r / np.sqrt(max(n, 1)), FAIL_RESIDUAL))  # penalty outside the site weight
    return np.concatenate(out) if out else np.zeros(1)


FAIL_RESIDUAL = 1e3


def _subset(samples, kind, regime):
    """samples: {kind: (zeta, phi)} of one site; positive phi in the regime."""
    z, ph = samples[kind]
    sel = (ph > 0) & ((z > 0) if regime == "stable" else (z < 0))
    return kind, z[sel], ph[sel]


def _solve(fun, start, lo, hi):
    """Bounded least squares; None unless the optimizer reports success. Adds identifiability diagnostics."""
    r = least_squares(fun, start, bounds=(lo, hi))
    if not r.success:
        return None
    _, sv, vt = np.linalg.svd(r.jac, full_matrices=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    dof = max(len(r.fun) - len(r.x), 1)
    with np.errstate(divide="ignore"):
        cov = (vt.T / sv**2) @ vt * (2 * r.cost / dof)  # SVD form, no squaring of the condition number
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    at_bound = (np.abs(r.x - np.asarray(lo)) <= BOUND_TOL * np.maximum(1.0, np.abs(lo))) | \
               (np.abs(r.x - np.asarray(hi)) <= BOUND_TOL * np.maximum(1.0, np.abs(hi)))
    se = np.where(at_bound, np.nan, se)  # one-sided optimum: no normal-theory SE
    interior = ~at_bound
    identified = bool(cond < COND_MAX and np.all(se[interior] < np.abs(r.x[interior])))
    return r.x, {"status": int(r.status), "cost": float(r.cost), "cond": cond, "identified": identified,
                 "at_bound": at_bound.tolist(), "constrained": bool(at_bound.any()), "se": se.tolist()}


def fit(form, site_samples, min_count):
    """site_samples: list of {kind: (zeta, phi)} (one per site). Returns params dict (with diagnostics) or None."""
    names, start, lo, hi = FORMS[form]
    momentum_regime = "unstable" if form == "J17" else "stable"
    data = [[d] for d in (_subset(s, "m", momentum_regime) for s in site_samples) if len(d[1]) >= min_count]
    if not data:
        return None
    diag = {}
    if form == "J3":
        r = _solve(lambda q: _residuals(form, np.r_[q, 1.0], data), start[:3], lo[:3], hi[:3])
        if r is None:
            return None
        heat = [[b for b in (_subset(s, "h", "stable"), _subset(s, "h", "unstable")) if len(b[1])] for s in site_samples]
        heat = [blocks for blocks in heat if sum(len(b[1]) for b in blocks) >= min_count]
        if not heat:
            return None
        r2 = _solve(lambda q: _residuals(form, np.r_[r[0], q], heat), start[3:], lo[3:], hi[3:])
        if r2 is None:
            return None
        x, diag = np.r_[r[0], r2[0]], {"momentum": r[1], "heat": r2[1]}
    else:
        r = _solve(lambda q: _residuals(form, q, data), start, lo, hi)
        if r is None:
            return None
        x, diag = r[0], {"momentum": r[1]}
    out = dict(zip(names, map(float, x)))
    if form == "J5":
        out["b"] = j5_b(out["a"], out["b_extra"])
    out["diagnostics"] = diag
    return out


def strict(name, kind):
    """User's strict reading: finite one-sided slopes at 0, phi monotone on each side, plus the C1 check elsewhere."""
    c = sf.continuity(name, kind)
    f = sf.SCHEMES[name][0 if kind == "m" else 1]
    g = jax.jit(jax.vmap(jax.grad(f)))
    slopes = {side: np.asarray(g(jnp.asarray(sign * MONOTONE_GRID))) for side, sign in (("stable", 1.0), ("unstable", -1.0))}
    monotone = {side: bool(np.all(s >= 0) or np.all(s <= 0)) for side, s in slopes.items()}
    finite = all(_slope_converges(g, sign) for sign in (1.0, -1.0))
    return {"finite_slope_at_0": finite, "monotone": monotone, "phi_continuous_at_0": c["zeta0"]["phi_continuous"],
            "c1_elsewhere": all(b["c1"] for b in c["branches"]),
            "pass": finite and all(monotone.values()) and c["zeta0"]["phi_continuous"] and all(b["c1"] for b in c["branches"]),
            "slope_at_0": c["zeta0"]["slope_algebraic"]}


def _slope_converges(g, sign):
    """Numerical finite-slope test at 0: algebraic slopes at 1e-6, 1e-9, 1e-12 must converge (successive differences
    shrink). Detects power-law divergence z^-e for e above about 3e-3; slower divergence (e.g. log log 1/z) is not
    detectable by sampling and would pass. Every form scored here has a closed form checked by hand in the tests."""
    s = np.asarray(g(jnp.asarray(sign * SLOPE_PROBE)))
    if not np.all(np.isfinite(s)):
        return False
    d1, d2 = abs(s[1] - s[0]), abs(s[2] - s[1])
    return bool(d2 <= d1 or d2 <= SLOPE_PROBE_ABS)


SLOPE_PROBE = np.array([1e-6, 1e-9, 1e-12])
SLOPE_PROBE_ABS = 1e-6  # both differences below this: slope already constant


def samples_for_fit(pairs_):
    return {kind: (np.concatenate([p["zeta"][p["keep"]] for p in plist]) if plist else np.empty(0),
                   np.concatenate([p["phi"][p["keep"]] for p in plist]) if plist else np.empty(0))
            for kind, plist in pairs_.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ustar", type=float, required=True)
    ap.add_argument("--min-count", type=int, required=True)
    ap.add_argument("--max-zg", type=float, required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    paths = [p for p in sorted(glob.glob(f"{a.tables}/*.nc")) if os.path.basename(p)[:-3] not in fs.EXCLUDED]
    odd = {}
    for path in paths:
        ds = fs.open_site(path)
        odd[os.path.basename(path)[:-3]] = samples_for_fit(sf.site_pairs(ds, a.min_ustar, True, a.max_zg, day_parity=1)[0])
    standard = list(sf.SCHEMES)
    result = {"args": vars(a), "global_fit": {}, "site_fit": {}, "sites": {}, "strict": {}}
    for form in FORMS:
        p = fit(form, list(odd.values()), a.min_count)
        result["global_fit"][form] = p
        if p is not None:
            register(f"{form}_global", form, [p[k] for k in FORMS[form][0]], "global fit, odd days of all sites")
    for path in paths:
        name = os.path.basename(path)[:-3]
        ds = fs.open_site(path)
        result["site_fit"][name] = {}  # fit uses odd days, scores below use even days
        for form in FORMS:
            p = fit(form, [odd[name]], a.min_count)
            result["site_fit"][name][form] = p
            sf.SCHEMES.pop(f"{form}_site", None)
            if p is not None:
                register(f"{form}_site", form, [p[k] for k in FORMS[form][0]], "per-site fit, odd days")
                result["strict"].setdefault(f"{form}_site", {})[name] = {k: strict(f"{form}_site", k) for k in "mh"}
        site = {}
        for arm, corr in (("offset_corrected", True), ("uncorrected", False)):
            pairs_, flux = sf.site_pairs(ds, a.min_ustar, corr, a.max_zg, day_parity=0)
            site[arm] = sf.score_site(pairs_, flux, a.min_count)
            if corr:
                site["inversion"] = sf.score_inversion(pairs_, flux, a.min_count)
        result["sites"][name] = site
        print(name, {f: result["site_fit"][name][f] for f in FORMS}, flush=True)
    for form in FORMS:
        sf.SCHEMES.pop(f"{form}_site", None)
    for s in sf.SCHEMES:
        result["strict"][s] = {k: strict(s, k) for k in "mh"}
    result["schemes"] = {k: {"kappa": v[2], "used_in": v[3], "source": v[4]} for k, v in sf.SCHEMES.items()}
    for form in FORMS:
        result["schemes"][f"{form}_site"] = {"kappa": sf.KAPPA, "used_in": "per-site fit, odd days", "source": "coscientist/cs_final.md"}
    result["bulk_only"] = {"louis1979": {"kappa": sf.KAPPA, "used_in": "ED2/BRAMS LEAF, older NWP", "source": "ED2 R-utils/sfclyrfun.r"}}
    result["standard"] = standard
    zeta = np.concatenate([-np.logspace(1, -3, 80), np.logspace(-3, 1, 80)])
    result["curves"] = {"zeta": zeta.tolist(), **{s: {k: sf.phi(s, k, zeta).tolist() for k in "mh"} for s in sf.SCHEMES}}
    with open(a.out, "w") as h:
        json.dump(result, h, indent=1, default=float)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

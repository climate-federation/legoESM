#!/usr/bin/env python
"""Does eta = z/zi explain flux-gradient residuals beyond zeta = z/L? Tower test of the co-scientist z/zi forms.

Protocol is benchmark v5 (candidate_formulations.py): same towers, filters, offset-corrected heat, odd UTC days fit,
even days scored, gradient log-RMSE, sites never pooled, stable and unstable separate. Baseline is the J3 global fit
(stable J3; unstable branch Dyer 1974, heat Pr0 x Dyer). Choices voted (coscientist/zi_test/vote1_*):
  diagnostic: held-out r = ln(phi_obs/phi_base) on ln(eta) with z/L-bin fixed effects, only z/L x eta cells (eta
  half-decade) with n >= min_count; samples with eta >= 1 dropped from every arm; outer factors that do not vanish at
  zeta = 0 fitted jointly over both regimes; placebo = same form with eta shuffled within site x regime (fit and score days); limit = same form with eta off, refitted; sondes
  matched within +-1.5 h of the interval midpoint; ERA5 bias correction multiplicative (median ERA5/sonde per site
  and sign of H, measured on odd days only); global and per-site fits.
zi arms: era5 (all sites); on sonde sites: era5, era5 ratio-corrected; on sonde-matched samples: era5 and sonde.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import zlib

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

import candidate_formulations as cf
import flux_gradient_sites as fs
import standard_formulations as sf

SONDE_WINDOW_S = 5400  # +-1.5 h, vote V6
SONDE_MAX_KM = 25.0  # same distance limit as the zi comparison
ETA_CELLS = 10.0 ** np.arange(-4.0, 0.01, 0.5)  # half-decade cells, eta < 1
SEED = 0
BASE = "J3_global"
FAIL = 1e3
F64 = np.float64


def base(kind, zeta):
    return sf.phi(BASE, kind, zeta)


def _x(z):
    return np.maximum(-z, 0.0)


# name: (regimes fitted, kinds, one shared parameter set over m+h, parameter names, start, lower, upper, phi(kind, zeta, eta, p, pr0))
def _h2(kind, z, e, p, pr0):
    return base(kind, z / (1.0 + e / 10.0 ** p[0]))


def _hc(kind, z, e, p, pr0):
    return base(kind, z) * np.exp(p[0] * e)


def _h8(kind, z, e, p, pr0):
    return (1.0 + p[0] * _x(z) * (1.0 - e ** p[1])) ** (-1.0 / 3.0)


def _h9(kind, z, e, p, pr0):
    mu = _x(z) / e
    gamma = p[0] + (p[1] - p[0]) / (1.0 + (mu / 10.0 ** p[2]) ** 2)
    return (1.0 + gamma * _x(z)) ** -0.25 if kind == "m" else pr0 * (1.0 + gamma * _x(z)) ** (-1.0 / 3.0)


def _h1(kind, z, e, p, pr0):
    return pr0 * (1.0 + p[0] * _x(z)) ** (-1.0 / 3.0) * np.exp(-p[1] * _x(z) * e / (e + _x(z)))


def _e1(kind, z, e, p, pr0):
    return (1.0 + p[0] * _x(z)) ** (-1.0 / 3.0) * np.exp(-p[1] * e * _x(z) / (e + _x(z)))


def _j19(kind, z, e, p, pr0):
    return (1.0 - e) ** (p[0] / 2.0) * base(kind, z * (1.0 - e) ** (-p[1]))


def _zl2(kind, z, e, p, pr0):
    """z/L-only control, 2 parameters, unstable: (1 + g|z|)^-p (momentum), Pr0 (1 + g|z|)^-p (heat)."""
    return (pr0 if kind == "h" else 1.0) * (1.0 + p[0] * _x(z)) ** (-p[1])


FORMS = {
    "H2_cap": (("unstable",), "mh", True, ("log10_eta_c",), (-1.0,), (-4.0,), (3.0,), _h2),
    "HC_exp": (("stable", "unstable"), "mh", False, ("c",), (0.0,), (-20.0,), (20.0,), _hc),
    "H8": (("unstable",), "m", False, ("a", "p"), (16.0, 1.0), (0.1, 0.05), (200.0, 5.0), _h8),
    "H9": (("unstable",), "mh", False, ("gamma_c", "gamma_r", "log10_mu_t"), (16.0, 16.0, 1.0), (0.1, 0.1, -1.0),
           (200.0, 200.0, 3.0), _h9),
    "H1": (("unstable",), "h", False, ("g", "a"), (16.0, 0.0), (1.0, 0.0), (200.0, 50.0), _h1),
    "E1": (("unstable",), "m", False, ("g", "a"), (16.0, 0.0), (1.0, 0.0), (200.0, 50.0), _e1),
    "J19": (("stable", "unstable"), "mh", False, ("a", "c"), (0.0, 0.0), (-10.0, -5.0), (10.0, 5.0), _j19),
    "zL2": (("unstable",), "mh", False, ("g", "p"), (16.0, 0.25), (0.1, 0.05), (200.0, 2.0), _zl2),
}
ETA_FREE = {"zL2"}  # control forms that do not read eta (no placebo)


# ---------------------------------------------------------------- zi on the samples

def era5_zi(name, times, zi_dir):
    e = xr.open_dataset(f"{zi_dir}/{name}_era5.nc", decode_times=False)
    t = e.time.values.astype(np.int64)
    k = np.clip(np.searchsorted(t, times), 0, t.size - 1)
    assert np.array_equal(t[k], times), f"{name}: ERA5 zi time axis does not cover the table"
    return e.zi.values.astype(F64)[k]


def sonde_file(name, zi_dir):
    for cand in (name, name.split("_flux")[0]):
        p = f"{zi_dir}/{cand}_sondes.nc"
        if os.path.exists(p):
            return p
    return None


def sonde_zi(path, mid):
    """Nearest valid sonde zi within the window of each interval midpoint (s since 1970); NaN otherwise."""
    s = xr.open_dataset(path)
    t = s.launch_time.values.astype("datetime64[s]").astype(np.int64)
    zi = s.zi.values.astype(F64)
    ok = np.isfinite(zi) & (s.distance_km.values <= SONDE_MAX_KM) & (zi > 0)
    t, zi = t[ok], zi[ok]
    order = np.argsort(t)
    t, zi = t[order], zi[order]
    out = np.full(mid.size, np.nan)
    if t.size == 0:
        return out
    k = np.clip(np.searchsorted(t, mid), 1, t.size - 1) if t.size > 1 else np.zeros(mid.size, int)
    lo, hi = np.clip(k - 1, 0, t.size - 1), k
    pick = np.where(np.abs(t[hi] - mid) < np.abs(t[lo] - mid), hi, lo)
    near = np.abs(t[pick] - mid) <= SONDE_WINDOW_S
    out[near] = zi[pick][near]
    return out


# ---------------------------------------------------------------- samples

def site_samples(ds, a, parity):
    """Per kind: concatenated kept samples of every pair, with time and heat-flux sign."""
    pairs_, flux = sf.site_pairs(ds, a.min_ustar, True, a.max_zg, day_parity=parity)
    t = ds.time.values.astype(np.int64)
    out = {}
    for kind, plist in pairs_.items():
        if not plist:
            continue
        cat = lambda k: np.concatenate([p[k][p["keep"]] for p in plist])  # noqa: E731
        idx = np.concatenate([np.nonzero(p["keep"])[0] for p in plist])
        zeta, inv_l = cat("zeta"), cat("inv_l")
        out[kind] = {"zeta": zeta, "phi": cat("phi"), "z_g": zeta / inv_l, "t": t[idx],
                     "hpos": flux["h"][idx] > 0}
    return out


def attach_eta(s, zi):
    s["eta"] = s["z_g"] / zi


def regime_sel(s, regime):
    z = s["zeta"]
    return np.isfinite(s["eta"]) & (s["eta"] < 1.0) & (s["eta"] > 0) & (s["phi"] > 0) & \
        ((z > 0) if regime == "stable" else (z < 0))


# ---------------------------------------------------------------- diagnostic

def fe_slope(zeta, eta, r, min_count, day=None):
    """OLS slope of r on ln(eta) with z/L-bin fixed effects; cells (z/L bin x eta half-decade) with n >= min_count.

    se: day-clustered (samples of one UTC day share zi, fluxes and weather); se_iid kept for reference."""
    zb = np.digitize(zeta, fs.EDGES)
    eb = np.digitize(eta, ETA_CELLS)
    cell = zb * 100 + eb
    u, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_count
    if keep.sum() < min_count:
        return None
    x, y, g = np.log(eta[keep]), r[keep], zb[keep]
    xd, yd = x.copy(), y.copy()
    for b in np.unique(g):
        m = g == b
        xd[m] -= x[m].mean()
        yd[m] -= y[m].mean()
    sxx = float(np.sum(xd**2))
    if sxx <= 0:
        return None
    slope = float(np.sum(xd * yd) / sxx)
    res = yd - slope * xd
    n, k = int(keep.sum()), len(np.unique(g)) + 1
    se_iid = float(np.sqrt(np.sum(res**2) / max(n - k, 1) / sxx))
    if day is None:
        se = se_iid
    else:
        _, di = np.unique(day[keep], return_inverse=True)
        sc = np.bincount(di, weights=xd * res)
        nc = sc.size
        cr1 = nc / max(nc - 1, 1) * (n - 1) / max(n - k, 1)  # CR1 small-sample factor incl. fixed effects
        se = float(np.sqrt(np.sum(sc**2) * cr1) / sxx) if nc > 1 else np.nan
    partial_r2 = float(1.0 - np.sum(res**2) / np.sum(yd**2)) if np.sum(yd**2) > 0 else np.nan
    prof = [[float(lo), float(hi), int(m.sum()), float(np.median(yd[m]))]
            for lo, hi in zip(ETA_CELLS[:-1], ETA_CELLS[1:]) for m in [(eta[keep] >= lo) & (eta[keep] < hi)] if m.sum() >= min_count]
    return {"n": n, "slope": slope, "se": se, "se_iid": se_iid, "partial_r2": partial_r2, "profile": prof,
            "ln_eta_iqr": float(np.subtract(*np.percentile(x, [75, 25])))}


def fe_split(zeta, z_g, zi, r, min_count, day):
    """Discriminates z/zi scaling from a separate height or zi effect: r on ln z and ln zi jointly, z/L-bin fixed
    effects, same cells as fe_slope. Pure z/zi scaling predicts b_z = -b_zi. Day-clustered CR1 covariance."""
    eta = z_g / zi
    zb, eb = np.digitize(zeta, fs.EDGES), np.digitize(eta, ETA_CELLS)
    _, inv, cnt = np.unique(zb * 100 + eb, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_count
    if keep.sum() < min_count:
        return None
    X = np.c_[np.log(z_g[keep]), np.log(zi[keep])]
    y, g = r[keep].copy(), zb[keep]
    for b in np.unique(g):
        m = g == b
        X[m] -= X[m].mean(0)
        y[m] -= y[m].mean()
    xtx = X.T @ X
    if np.linalg.cond(xtx) > 1e10:
        return None
    beta = np.linalg.solve(xtx, X.T @ y)
    res = y - X @ beta
    _, di = np.unique(day[keep], return_inverse=True)
    sc = np.c_[np.bincount(di, X[:, 0] * res), np.bincount(di, X[:, 1] * res)]
    n, k, nc = int(keep.sum()), len(np.unique(g)) + 2, sc.shape[0]
    if nc < 2 or n <= k:
        return None  # clustered covariance undefined
    inv_ = np.linalg.inv(xtx)
    cov = inv_ @ (sc.T @ sc) @ inv_ * nc / max(nc - 1, 1) * (n - 1) / max(n - k, 1)
    se = np.sqrt(np.diag(cov))
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    return {"n": n, "b_z": float(beta[0]), "se_z": float(se[0]), "b_zi": float(beta[1]), "se_zi": float(se[1]),
            "b_z_plus_b_zi": float(beta.sum()), "se_sum": float(np.sqrt(cov.sum())), "corr_lnz_lnzi": corr,
            "sd_lnz": float(X[:, 0].std())}


# ---------------------------------------------------------------- sensor bias vs height (all level pairs)

def pair_means(pairs_, kind, regime, min_count, bin_mean=None):
    """Per level pair: mean residual at fixed z/L against the INTERVAL-AVERAGED baseline (integral of phi_J3 over
    ln z from z1 to z2, divided by ln(z2/z1)), so wide and narrow pairs are judged on equal terms. z/L-bin means
    (bin_mean, pooled over both splits by the caller) removed. Pairs with fewer than min_count samples dropped.
    Returns (rows, raw) where raw feeds the pooled bin means."""
    raw = []
    for p in pairs_.get(kind, []):
        k = p["keep"]
        z, ph = p["zeta"][k], p["phi"][k]
        sel = (ph > 0) & ((z > 0) if regime == "stable" else (z < 0))
        z1, z2, il = p["z1"][k][sel], p["z2"][k][sel], p["inv_l"][k][sel]
        with np.errstate(all="ignore"):
            r = np.log(ph[sel] / (sf.integral(BASE, kind, il, z1, z2) / np.log(z2 / z1)))
        ok = np.isfinite(r)
        raw.append((tuple(p["levels"]), np.log(p["z_g"]), z[sel][ok], r[ok]))
    if bin_mean is None:
        return [], raw
    out = []
    for levels, lnz, z, r in raw:
        if r.size >= min_count:
            rd = r - np.array([bin_mean.get(b, np.nan) for b in np.digitize(z, fs.EDGES)])
            out.append({"levels": levels, "ln_z": float(lnz), "n": int(r.size), "mean": float(np.nanmean(rd))})
    return out, raw


def pooled_bin_means(*raws):
    z = np.concatenate([x[2] for raw in raws for x in raw]) if any(raws) else np.empty(0)
    r = np.concatenate([x[3] for raw in raws for x in raw]) if any(raws) else np.empty(0)
    zb = np.digitize(z, fs.EDGES)
    return {b: float(r[zb == b].mean()) for b in np.unique(zb)}


def _design(rows, model, n_levels):
    one = np.ones((len(rows), 1))
    lnz = np.array([[r["ln_z"]] for r in rows])
    sens = np.zeros((len(rows), n_levels))
    for k, r in enumerate(rows):  # an additive level bias enters a pair difference as bias(upper) - bias(lower)
        sens[k, r["levels"][1]] += 1.0
        sens[k, r["levels"][0]] -= 1.0
    sens = sens[:, 1:]  # level 0 is the reference
    return {"height": np.hstack([one, lnz]), "sensor": np.hstack([one, sens]), "both": np.hstack([one, lnz, sens])}[model]


def pair_test(odd, even, n_levels):
    """Fit pair means on odd days, predict even-day pair means. Weighted held-out R2 per model; pairs in both splits."""
    keys = sorted(set(r["levels"] for r in odd) & set(r["levels"] for r in even))
    if len(keys) < 3:
        return None
    o = sorted([r for r in odd if r["levels"] in keys], key=lambda r: r["levels"])
    e = sorted([r for r in even if r["levels"] in keys], key=lambda r: r["levels"])
    w = np.sqrt(np.array([r["n"] for r in o], float))
    ye = np.array([r["mean"] for r in e])
    we = np.array([r["n"] for r in e], float)
    sst = float(np.sum(we * (ye - np.average(ye, weights=we)) ** 2))
    out = {"pairs": [list(k) for k in keys], "n_pairs": len(keys), "even_means": ye.tolist(),
           "ln_z": [r["ln_z"] for r in e], "sst": sst}
    for m in ("height", "sensor", "both"):
        X = _design(o, m, n_levels)
        X = X[:, np.any(X != 0, axis=0)] if m != "height" else X
        beta, *_ = np.linalg.lstsq(X * w[:, None], np.array([r["mean"] for r in o]) * w, rcond=None)
        Xe = _design(e, m, n_levels)
        Xe = Xe[:, np.any(_design(o, m, n_levels) != 0, axis=0)] if m != "height" else Xe
        pred = Xe @ beta
        sse = float(np.sum(we * (ye - pred) ** 2))
        out[m] = {"params": X.shape[1], "sse": sse, "r2": float(1 - sse / sst) if sst > 0 else np.nan}
    out["sensor_saturated"] = out["sensor"]["params"] >= len(keys)  # fits any persistent pair pattern; not a test
    out["both_saturated"] = out["both"]["params"] >= len(keys)
    return out


# ---------------------------------------------------------------- fits

def groups(form):
    """Kinds fitted together: one shared parameter set for joint forms, else one fit per kind (vote V4)."""
    return [FORMS[form][1]] if FORMS[form][2] else list(FORMS[form][1])


LIMIT_ETA = 1e-300  # eta -> 0 control, refitted; this small so eta**p vanishes even at p = 0.05 (H8)


def eta_for(name, kind, regime, eta, mode, parity):
    """real eta; placebo = permutation within site x regime x kind (fit and score days permuted independently);
    limit = eta switched off."""
    if mode == "placebo":
        return np.random.default_rng([SEED, zlib.crc32(name.encode()), ord(kind), len(regime), parity]).permutation(eta)
    if mode == "limit":
        return np.full_like(eta, LIMIT_ETA)
    return eta


def _blocks(sites, form, mode, kinds, parity=1):
    """One list of (kind, regime, zeta, eta, phi) blocks per site for the form's regimes and the given kinds."""
    regimes = FORMS[form][0]
    data = []
    for name, s in sites:
        blocks = []
        for kind in kinds:
            if kind not in s:
                continue
            for regime in regimes:
                sel = regime_sel(s[kind], regime)
                eta = eta_for(name, kind, regime, s[kind]["eta"][sel], mode, parity)
                blocks.append((kind, regime, s[kind]["zeta"][sel], eta, s[kind]["phi"][sel]))
        data.append(blocks)
    return data


def model(form, kind, zeta, eta, p, pr0):
    with np.errstate(all="ignore"):
        return FORMS[form][7](kind, zeta, eta, p, pr0)


def _residuals(form, p, data, pr0):
    out = []
    for blocks in data:
        n = sum(len(b[2]) for b in blocks)
        for kind, _, z, e, ph in blocks:
            with np.errstate(all="ignore"):
                r = np.log(ph / model(form, kind, z, e, p, pr0))
            out.append(np.where(np.isfinite(r), r / np.sqrt(max(n, 1)), FAIL))
    return np.concatenate(out) if out else np.zeros(1)


def fit(form, data, pr0, min_count):
    data = [b for b in ([blk for blk in d if len(blk[2])] for d in data) if sum(len(x[2]) for x in b) >= min_count]
    if not data:
        return None
    _, _, _, names, start, lo, hi, _ = FORMS[form]
    r = least_squares(lambda q: _residuals(form, q, data, pr0), start, bounds=(lo, hi))
    if not r.success:
        return None
    at_bound = [bool(abs(x - l) <= 1e-6 * max(1, abs(l)) or abs(x - h) <= 1e-6 * max(1, abs(h))) for x, l, h in zip(r.x, lo, hi)]
    return {"p": r.x.tolist(), "names": list(names), "cost": float(r.cost), "at_bound": at_bound}


def score(form, p, s, kind, regime, pr0, name="", mode="real"):
    """Held-out log-RMSE; any non-finite prediction marks the score invalid (primary), finite-only kept as secondary."""
    sel = regime_sel(s[kind], regime)
    z, ph = s[kind]["zeta"][sel], s[kind]["phi"][sel]
    e = eta_for(name, kind, regime, s[kind]["eta"][sel], mode, 0)
    pred = base(kind, z) if form == "base" else model(form, kind, z, e, p, pr0)
    with np.errstate(all="ignore"):
        lr = np.log(ph / pred)
    ok = np.isfinite(lr)
    finite = float(np.sqrt(np.mean(lr[ok] ** 2))) if ok.any() else np.nan
    return {"n": int(sel.sum()), "n_fail": int((~ok).sum()), "log_rmse": finite if ok.all() else np.nan,
            "log_rmse_finite_only": finite}


# ---------------------------------------------------------------- strict smoothness at fixed eta

STRICT_ETAS = (0.003, 0.03, 0.1, 0.3, 0.9)
STRICT_GRID = np.logspace(-4, np.log10(sf.ZETA_DOMAIN * 0.999), 400)


def strict(form, p, pr0, kind):
    """phi continuous at 0, finite one-sided slopes (difference quotients converge), monotone on each fitted side."""
    out = []
    regimes = FORMS[form][0]
    for e in STRICT_ETAS:
        f = lambda z: model(form, kind, np.asarray(z, F64), np.full(np.shape(z), e), p, pr0)  # noqa: E731
        sides = {}
        for regime, sign in (("stable", 1.0), ("unstable", -1.0)):
            g = STRICT_GRID * sign
            ph = f(g) if regime in regimes else base(kind, g)
            d = np.diff(ph)
            sides[regime] = bool(np.all(np.isfinite(ph)) and (np.all(d >= -1e-12) or np.all(d <= 1e-12)))
        v0p = f(np.array([1e-12])) if "stable" in regimes else base(kind, np.array([1e-12]))
        v0m = f(np.array([-1e-12])) if "unstable" in regimes else base(kind, np.array([-1e-12]))
        cont = bool(abs(v0p[0] - v0m[0]) <= 1e-6)
        fin = True
        for regime, sign in (("stable", 1.0), ("unstable", -1.0)):
            h = np.array([1e-4, 1e-6, 1e-8]) * sign
            ff = f if regime in regimes else (lambda z: base(kind, z))
            q = (ff(h) - ff(np.zeros(3) + 1e-14 * sign)) / h
            # quotients at h = 1e-4, 1e-6, 1e-8 must converge geometrically; a log-divergent slope keeps a constant
            # per-decade step. Floor 1e-6 is round-off of the 1e-8 quotient; log terms below ~2e-7 z ln z pass.
            d1, d2 = abs(q[1] - q[0]), abs(q[2] - q[1])
            fin &= bool(np.all(np.isfinite(q)) and d2 <= max(0.1 * d1, 1e-6))
        out.append({"eta": e, "continuous_at_0": cont, "finite_slope_at_0": fin, "monotone": sides,
                    "pass": cont and fin and all(sides.values())})
    return {"pass": all(o["pass"] for o in out), "per_eta": out}


# ---------------------------------------------------------------- driver

def run_arm(arm, fit_sites, score_sites, pr0, a):
    """fit_sites/score_sites: [(name, samples)] with eta attached. Returns diagnostic, fits and scores."""
    res = {"n_sites": len(score_sites), "diagnostic": {}, "global_fit": {}, "site_fit": {}, "scores": {}}
    for name, s in score_sites:
        d = res["diagnostic"][name] = {}
        for kind in s:
            for regime in ("stable", "unstable"):
                sel = regime_sel(s[kind], regime)
                z, e, ph = s[kind]["zeta"][sel], s[kind]["eta"][sel], s[kind]["phi"][sel]
                day = s[kind]["t"][sel] // 86400
                with np.errstate(all="ignore"):
                    r = np.log(ph / base(kind, z))
                ok = np.isfinite(r)
                d[f"{kind}_{regime}"] = fe_slope(z[ok], e[ok], r[ok], a.min_count, day[ok]) if ok.sum() >= a.min_count else None
                zg = s[kind]["z_g"][sel]
                d[f"{kind}_{regime}_split"] = (fe_split(z[ok], zg[ok], zg[ok] / e[ok], r[ok], a.min_count, day[ok])
                                               if ok.sum() >= a.min_count else None)
    if a.diag_only:
        return res
    for form in FORMS:
        for mode in (("real",) if form in ETA_FREE else ("real", "placebo", "limit")):
            tag = form + ("" if mode == "real" else "_" + mode)
            for grp in groups(form):
                g = fit(form, _blocks(fit_sites, form, mode, grp), pr0, a.min_count)
                res["global_fit"][f"{tag}:{grp}"] = g
                if g and mode == "real":
                    g["strict"] = {k: strict(form, np.array(g["p"]), pr0, k) for k in grp}
                for name, s in score_sites:
                    fs_ = dict(fit_sites).get(name)
                    sf_ = (fit(form, _blocks([(name, fs_)], form, mode, grp), pr0, a.min_count)
                           if (fs_ and a.site_fits) else None)
                    if sf_ and mode == "real":
                        sf_["strict"] = {k: strict(form, np.array(sf_["p"]), pr0, k)["pass"] for k in grp}
                    res["site_fit"].setdefault(name, {})[f"{tag}:{grp}"] = sf_
                    sc = res["scores"].setdefault(name, {})
                    for kind in grp:
                        if kind not in s:
                            continue
                        for regime in FORMS[form][0]:
                            c = sc.setdefault(f"{kind}_{regime}", {})
                            if "base" not in c:
                                c["base"] = score("base", None, s, kind, regime, pr0)
                            if g:
                                c[tag + "_global"] = score(form, np.array(g["p"]), s, kind, regime, pr0, name, mode)
                            if sf_:
                                c[tag + "_site"] = score(form, np.array(sf_["p"]), s, kind, regime, pr0, name, mode)
        print(arm, form, "done", flush=True)
    return res


# ---------------------------------------------------------------- figures (Okabe-Ito)

OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
PANELS = ("m_stable", "m_unstable", "h_stable", "h_unstable")


def plot(path, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(path))
    arm = res["arms"]["era5_all"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for a_, key in zip(ax.ravel(), PANELS):
        slopes = []
        for site, d in arm["diagnostic"].items():
            x = d.get(key)
            if not x or not x["profile"]:
                continue
            pr = np.array(x["profile"])
            a_.plot(np.sqrt(pr[:, 0] * pr[:, 1]), pr[:, 3], color="#999999", lw=0.8)
            slopes.append(x["slope"])
        pos = sum(s_ > 0 for s_ in slopes)
        a_.axhline(0, color=OI[0], lw=0.8)
        a_.set_xscale("log")
        a_.set_title(f"{key}: slope d r / d ln(z/zi) median {np.median(slopes) if slopes else np.nan:+.3f}, {pos}/{len(slopes)} > 0",
                     fontsize=9)
        a_.set_ylabel("residual at fixed z/L, ln(phi_obs/phi_J3)")
    for a_ in ax[1]:
        a_.set_xlabel("z/zi (ERA5)")
    fig.suptitle("Held-out residual of the best z/L form against z/zi, one grey line per tower")
    fig.tight_layout()
    fig.savefig(prefix + "_residual.png", dpi=130)
    plt.close(fig)

    forms = [f for f in FORMS]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a_, key in zip(ax.ravel(), PANELS):
        ylab = []
        for i, f in enumerate(forms):
            for j, (tag, col) in enumerate(((f + "_global", OI[5]), (f + "_placebo_global", OI[1]),
                                            (f + "_limit_global", OI[3]))):
                d = [v[key][tag]["log_rmse"] - v[key]["base"]["log_rmse"] for v in arm["scores"].values()
                     if key in v and tag in v[key] and np.isfinite(v[key][tag]["log_rmse"])]
                bad = sum(1 for v in arm["scores"].values() if key in v and tag in v[key] and not np.isfinite(v[key][tag]["log_rmse"]))
                if bad:
                    a_.annotate(f"{bad} invalid", (1.0, i + 0.2 * j), xycoords=("axes fraction", "data"), fontsize=6, ha="right")
                if d:
                    a_.plot(d, np.full(len(d), i + 0.2 * j), "o", ms=3, color=col, alpha=0.7)
                    a_.plot(np.median(d), i + 0.2 * j, "|", ms=14, color=OI[0])
            ylab.append(f)
        a_.axvline(0, color=OI[0], lw=0.8)
        a_.set_yticks(range(len(forms)))
        a_.set_yticklabels(ylab, fontsize=8)
        a_.set_title(key, fontsize=9)
        a_.set_xlabel("held-out log-RMSE minus J3 baseline (negative = better)")
    ax[0, 0].plot([], [], "o", color=OI[5], label="z/zi form, global fit")
    ax[0, 0].plot([], [], "o", color=OI[1], label="same form, z/zi shuffled (placebo)")
    ax[0, 0].plot([], [], "o", color=OI[3], label="same form, z/zi switched off, refitted")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Candidate scores per tower, ERA5 zi, even days")
    fig.tight_layout()
    fig.savefig(prefix + "_scores.png", dpi=130)
    plt.close(fig)

    arms = ("era5_sondesites", "era5_ratio_corrected", "era5_on_sonde_samples", "sonde")
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for a_, key in zip(ax.ravel(), PANELS):
        sites = res["sonde_sites"]
        for k, (an, col) in enumerate(zip(arms, OI[1:])):
            y = [(res["arms"][an]["diagnostic"].get(s_, {}).get(key) or {}) for s_ in sites]
            a_.errorbar(np.arange(len(sites)) + 0.15 * k, [v.get("slope", np.nan) for v in y],
                        yerr=[v.get("se", np.nan) for v in y], fmt="o", color=col, ms=4, label=an)
        a_.axhline(0, color=OI[0], lw=0.8)
        a_.set_xticks(range(len(sites)))
        a_.set_xticklabels(sites, rotation=60, fontsize=7)
        a_.set_title(key, fontsize=9)
        a_.set_ylabel("slope d r / d ln(z/zi)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Sensitivity of the z/zi residual slope to the zi source (sonde sites)")
    fig.tight_layout()
    fig.savefig(prefix + "_zi_source.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", nargs=2, metavar=("JSON", "PREFIX"), help="only draw the figures from a result file")
    ap.add_argument("--tables")
    ap.add_argument("--zi-dir")
    ap.add_argument("--out")
    ap.add_argument("--min-ustar", type=float)
    ap.add_argument("--min-count", type=int)
    ap.add_argument("--max-zg", type=float)
    ap.add_argument("--no-site-fits", dest="site_fits", action="store_false")
    ap.add_argument("--diag-only", action="store_true", help="diagnostics only, no candidate fits")
    ap.add_argument("--pair-test", help="write the all-level-pairs sensor-bias test here, then stop")
    a = ap.parse_args()
    if a.plot:
        return plot(*a.plot)
    np.seterr(all="ignore")
    paths = [p for p in sorted(glob.glob(f"{a.tables}/*.nc")) if os.path.basename(p)[:-3] not in fs.EXCLUDED]
    odd_v5, odd, even = {}, {}, {}
    for path in paths:
        name = os.path.basename(path)[:-3]
        ds = fs.open_site(path)
        odd_v5[name] = cf.samples_for_fit(sf.site_pairs(ds, a.min_ustar, True, a.max_zg, day_parity=1)[0])
        if not os.path.exists(f"{a.zi_dir}/{name}_era5.nc"):
            print("no ERA5 zi, not scored:", name, flush=True)
            continue
        mid_off = int(ds.attrs["averaging_minutes"]) * 30
        for store, parity in ((odd, 1), (even, 0)):
            s = site_samples(ds, a, parity)
            for kind in s:
                s[kind]["zi_era5"] = era5_zi(name, s[kind]["t"], a.zi_dir)
                sp = sonde_file(name, a.zi_dir)
                s[kind]["zi_sonde"] = sonde_zi(sp, s[kind]["t"] - mid_off) if sp else np.full(s[kind]["t"].size, np.nan)
            store[name] = s
    # baseline J3 global fit exactly as benchmark v5 (all v5 sites, odd days, no eta filter)
    j3 = cf.fit("J3", list(odd_v5.values()), a.min_count)
    cf.register(BASE, "J3", [j3[k] for k in cf.FORMS["J3"][0]], "global fit, odd days of all sites")
    pr0 = j3["pr0"]
    if a.pair_test:
        pt = {"baseline": {k: v for k, v in j3.items() if k != "diagnostics"}, "sites": {}}
        for path in paths:
            name = os.path.basename(path)[:-3]
            ds = fs.open_site(path)
            sp = {par: sf.site_pairs(ds, a.min_ustar, True, a.max_zg, day_parity=par, all_pairs=True)[0] for par in (0, 1)}
            for kind, nlev in (("m", ds.sizes["lev_u"]), ("h", ds.sizes["lev_t"])):
                for regime in ("stable", "unstable"):
                    _, ro = pair_means(sp[1], kind, regime, a.min_count)
                    _, re_ = pair_means(sp[0], kind, regime, a.min_count)
                    bm = pooled_bin_means(ro, re_)  # same target definition on both splits (GLM review)
                    t = pair_test(pair_means(sp[1], kind, regime, a.min_count, bm)[0],
                                  pair_means(sp[0], kind, regime, a.min_count, bm)[0], nlev)
                    if t:
                        pt["sites"].setdefault(name, {})[f"{kind}_{regime}"] = t
            print(name, "pair test done", flush=True)
        with open(a.pair_test, "w") as h:
            json.dump(pt, h, indent=1, default=float)
        print("WROTE", a.pair_test)
        return
    result = {"args": vars(a), "baseline": {k: v for k, v in j3.items() if k != "diagnostics"}, "arms": {},
              "sonde_sites": [], "ratio": {}}

    def with_eta(store, names, key, mask_sonde=False):
        out = []
        for n in names:
            s = {}
            for kind, v in store[n].items():
                w = dict(v)
                zi = v[key] if isinstance(key, str) else key(n, kind, v)
                if mask_sonde:
                    zi = np.where(np.isfinite(v["zi_sonde"]), zi, np.nan)
                attach_eta(w, zi)
                s[kind] = w
            out.append((n, s))
        return out

    names = sorted(even)
    sonde_names = [n for n in names if any(np.isfinite(v["zi_sonde"]).sum() >= a.min_count for v in even[n].values())]
    result["sonde_sites"] = sonde_names
    for n in sonde_names:  # multiplicative ERA5/sonde ratio per site and sign of H, odd days only
        v = np.concatenate([np.c_[x["zi_era5"], x["zi_sonde"], x["hpos"], x["t"]] for x in odd[n].values()])
        _, first = np.unique(v[:, 3], return_index=True)  # one row per time stamp, not per level pair
        v = v[first]
        ok = np.isfinite(v[:, 0]) & np.isfinite(v[:, 1]) & (v[:, 0] > 0)
        result["ratio"][n] = {lab: (float(np.median(v[ok & sel, 0] / v[ok & sel, 1])) if (ok & sel).sum() >= a.min_count else np.nan,
                                    int((ok & sel).sum()))
                              for lab, sel in (("H>0", v[:, 2] > 0), ("H<=0", v[:, 2] <= 0))}

    def corrected(n, kind, v):
        r = result["ratio"][n]
        ratio = np.where(v["hpos"], r["H>0"][0], r["H<=0"][0])
        return v["zi_era5"] / ratio

    arms = {
        "era5_all": (with_eta(odd, names, "zi_era5"), with_eta(even, names, "zi_era5")),
        "era5_sondesites": (with_eta(odd, sonde_names, "zi_era5"), with_eta(even, sonde_names, "zi_era5")),
        "era5_ratio_corrected": (with_eta(odd, sonde_names, corrected), with_eta(even, sonde_names, corrected)),
        "era5_on_sonde_samples": (with_eta(odd, sonde_names, "zi_era5", True), with_eta(even, sonde_names, "zi_era5", True)),
        "sonde": (with_eta(odd, sonde_names, "zi_sonde"), with_eta(even, sonde_names, "zi_sonde")),
    }
    for arm, (fit_s, score_s) in arms.items():
        result["arms"][arm] = run_arm(arm, fit_s, score_s, pr0, a)
        with open(a.out, "w") as h:
            json.dump(result, h, indent=1, default=float)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

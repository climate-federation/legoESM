#!/usr/bin/env python
"""Self-correlation tests for round-2 winners (votes R1-V1, S1 in coscientist/round2/votes.md).

Statistic: held-out advantage of a curved (saturating) stable form over the same family made linear, i.e. median over
sites of [even-day log-RMSE(curved) - log-RMSE(linear)], both fitted on odd days. Computed on real data and on
data whose flux rows were permuted, so phi and zeta share only the algebraic coupling.

permute u*: within site, day parity and stability sign, whole 30-min timestamps are permuted (coherent across level
  pairs); each sample takes u* of its partner time, H and LE kept, so B is unchanged and 1/L scales by (u*/u*')^3;
  phi_m scales by u*/u*', phi_h by u*'/u*.
permute H : same, each sample takes H of its partner time (same sign), u* and LE kept; B' = B + (H' - H)/(rho c_p)
  (the latent part of B is unchanged), 1/L' = 1/L * B'/B; phi_h scales by H/H'.
After permutation the v5 sample filters are re-applied: |z2/L| < 10, heat |zeta| >= 0.1, stability sign unchanged.
split     : zeta from the next sonic above the lowest (independent u*, H, LE), phi from the lowest flux, on the
            subset where both exist; compared with own-flux zeta on the same subset.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import flux_gradient_sites as fs
import round2 as r2
from flux_gradient import KAPPA
from legoesm import constants

N_SEEDS = 20
PAIRS = {  # (curved, linear) per kind; unstable side Dyer in both so only the stable shape differs
    "h": ({"name": "h_sat", "stable": "1 + a*z/(1 + z/zc)", "unstable": "(1 + 16*x)**-0.5", "params": {"a": [3, 0, 20], "zc": [0.5, 0.01, 20]}},
          {"name": "h_lin", "stable": "1 + b*z", "unstable": "(1 + 16*x)**-0.5", "params": {"b": [5, 0, 20]}}),
    "m": ({"name": "m_sqrt", "stable": "1 + b*z/jnp.sqrt(1 + z/zc)", "unstable": "(1 + 16*x)**-0.25", "params": {"b": [4, 0, 20], "zc": [0.5, 0.01, 50]}},
          {"name": "m_lin", "stable": "1 + b*z", "unstable": "(1 + 16*x)**-0.25", "params": {"b": [5, 0, 20]}}),
}


def raw(npz):
    d = np.load(npz)
    out = {}
    for k in d.files:
        s, kind, par, f = k.split("|")
        out.setdefault((s, kind, int(par)), {})[f] = d[k]
    return out


def permute(rawd, var, rng):
    """{site: {kind: {parity: (zeta, phi)}}} after a coherent timestamp permutation of u* or H."""
    out = {}
    for s in sorted({k[0] for k in rawd}):
        for par in (0, 1):
            rows = [rawd[(s, k, par)] for k in "mh" if (s, k, par) in rawd]
            if not rows:
                continue
            t = np.concatenate([r["time"] for r in rows])
            u = np.concatenate([r["ustar"] for r in rows])
            h = np.concatenate([r["h"] for r in rows])
            il = np.concatenate([r["inv_l"] for r in rows])
            times, first = np.unique(t, return_index=True)
            new = {}
            for sign in (1, -1):
                sel = np.where(np.sign(il[first]) == sign)[0]
                if var == "h":
                    sel = sel[np.sign(h[first][sel]) == -sign]
                perm = rng.permutation(sel)
                new.update(zip(times[sel], first[perm]))
            for k in "mh":
                v = rawd.get((s, k, par))
                if v is None:
                    continue
                j = np.array([new.get(x, -1) for x in v["time"]], dtype=int)
                ok = j >= 0
                jj = np.where(ok, j, 0)
                u0, h0, il0 = v["ustar"], v["h"], v["inv_l"]
                if var == "ustar":
                    u1 = u[jj]
                    il1 = il0 * (u0 / u1) ** 3
                    ph = v["phi"] * ((u0 / u1) if k == "m" else (u1 / u0))
                else:
                    h1 = h[jj]
                    b0 = -il0 * v["theta"] * u0**3 / KAPPA / constants.g
                    b1 = b0 + (h1 - h0) / (v["rho"] * constants.c_pd)
                    il1 = il0 * b1 / b0
                    ph = v["phi"] * ((h0 / h1) if k == "h" else 1.0)
                z = v["zeta"] * il1 / il0
                keep = ok & np.isfinite(z) & np.isfinite(ph) & (np.sign(z) == np.sign(v["zeta"])) & (np.abs(v["z2"] * il1) < r2.ZETA_DOMAIN)
                if k == "h":
                    keep &= np.abs(z) >= fs.MIN_HEAT_ABS_ZETA
                out.setdefault(s, {"m": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2},
                                   "h": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2}})[k][par] = (z[keep], ph[keep])
    return out


def split_second_level(rawd):
    """Two datasets on one sample subset (second sonic present and passing QC): zeta from the lowest flux (own) or from
    the next sonic up (split); phi always from the lowest flux. Filters re-applied to the split zeta."""
    own, split, n = {}, {}, {}
    for (s, k, par), v in rawd.items():
        z2 = v["zeta"] * v["inv_l2"] / v["inv_l"]
        keep = np.isfinite(z2) & (np.sign(z2) == np.sign(v["zeta"])) & (np.abs(v["z2"] * v["inv_l2"]) < r2.ZETA_DOMAIN)
        if k == "h":
            keep &= np.abs(z2) >= fs.MIN_HEAT_ABS_ZETA
        empty = {"m": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2}, "h": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2}}
        own.setdefault(s, {kk: dict(vv) for kk, vv in empty.items()})[k][par] = (v["zeta"][keep], v["phi"][keep])
        split.setdefault(s, {kk: dict(vv) for kk, vv in empty.items()})[k][par] = (z2[keep], v["phi"][keep])
        n[f"{s}|{k}|{par}"] = int(keep.sum())
    return own, split, n


def advantage(data, kind, min_count):
    """Per panel: median over sites of held-out log-RMSE(curved) - log-RMSE(linear), and the fitted curved params."""
    curved, linear = PAIRS[kind]
    res, sc = {}, {}
    for spec in (curved, linear):
        p, diag = r2.fit(spec, kind, data, min_count)
        if not diag["success"]:
            return {"error": diag}
        res[spec["name"]] = dict(zip(spec["params"], map(float, p)))
        sc[spec["name"]] = r2.score(spec, p, kind, data, min_count)
    out = {"params": res}
    for panel in sc[curved["name"]]:
        d = [sc[curved["name"]][panel][s] - sc[linear["name"]][panel].get(s, np.nan) for s in sc[curved["name"]][panel]]
        d = np.array([x for x in d if np.isfinite(x)])
        out[panel] = {"median_diff": float(np.median(d)) if len(d) else np.nan, "n_sites": int(len(d))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=30)
    a = ap.parse_args()
    np.seterr(all="ignore")
    rawd = raw(a.samples)
    real = r2.load(a.samples)
    result = {"real": {k: advantage(real, k, a.min_count) for k in "mh"}}
    print("real", result["real"], flush=True)
    own, split, n = split_second_level(rawd)
    result["split"] = {"n": n, "own_flux": {k: advantage(own, k, a.min_count) for k in "mh"},
                       "split_flux": {k: advantage(split, k, a.min_count) for k in "mh"}}
    print("split", result["split"]["own_flux"], result["split"]["split_flux"], flush=True)
    for var in ("ustar", "h"):
        runs = []
        for seed in range(N_SEEDS):
            data = permute(rawd, var, np.random.default_rng(seed))
            runs.append({k: advantage(data, k, a.min_count) for k in "mh"})
            print(var, seed, runs[-1], flush=True)
        result[f"perm_{var}"] = runs
    json.dump(result, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

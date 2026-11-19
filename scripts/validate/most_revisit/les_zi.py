#!/usr/bin/env python
"""Sensor-free check of the tower height / z-over-zi term on the Heisel-Chamecki 2024 convective LES (mean profiles).

Per case (never pooled): phi_m = kappa z |dU/dz| / u*, phi_h = kappa z dtheta/dz / theta*, surface u*, Q; L from u*,
Q, theta at the first level; zi = height of the minimum total heat flux. Gradients from adjacent staggered mean levels,
placed at the flux height between them. Residual r = ln(phi / phi_J3(z/L)), J3 global benchmark-v5 fit (unstable
branch = Dyer, heat Pr0 x Dyer). Within one case z/L and z/zi are both proportional to z, so the fixed-z/L test
compares cases inside z/L bins: case-mean r on ln z and ln zi with bin fixed effects (towers: b_z > 0).
Near-wall artefact check: grid convergence of the same numbers on the 200-512 point grids (cases A, C, E).
Voted choices (coscientist/zi_test/vote2_*): see NEAR_WALL_LEVELS, ETA_MAX, scales, shear below.
Cheng DNS Sh5/Sh20 are not used: the files hold 3-D filtered u, v, w and SGS stress with no vertical coordinate and no
temperature, so neither z, L nor zi can be formed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

import candidate_formulations as cf
import flux_gradient as fg
import flux_gradient_sites as fs
import standard_formulations as sf
from legoesm import constants

KAPPA = fg.KAPPA
SGS_SHARE_MAX = 0.3  # vote V1: drop every flux level up to the highest one near the wall where the SGS share of the
# momentum flux exceeds 30% (only the contiguous near-wall block; z/zi <= ETA_MAX above it)
ETA_MAX = 0.1  # vote V2: surface layer z/zi <= 0.1, the tower range
J3_V5 = [4.025317781641760, 0.47095884193497345, 1.2235740991725885, 1.0014241766478238]  # standard_benchmark_v5.json
BASE = "J3_global"
CONVERGENCE_GRIDS = (200, 300, 400, 512)


def load(path):
    a = np.loadtxt(path, skiprows=1)
    zm, U, V, th, zf = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    uw, vw, wt = a[:, 5] + a[:, 6], a[:, 7] + a[:, 8], a[:, 9] + a[:, 10]
    with np.errstate(all="ignore"):
        share = np.hypot(a[:, 6], a[:, 8]) / np.hypot(uw, vw)  # SGS share of the momentum flux at flux levels
    assert zf[0] == 0.0 and np.all(np.diff(zm) > 0) and np.all((zf[1:] > zm[:-1]) & (zf[1:] < zm[1:]))
    ustar = float((uw[0] ** 2 + vw[0] ** 2) ** 0.25)  # vote V3: surface fluxes, as on towers
    q0 = float(wt[0])
    theta_star = -q0 / ustar
    length = float(-ustar**3 * th[0] / (KAPPA * constants.g * q0))
    zi = float(zf[np.argmin(wt)])
    k = np.arange(1, zm.size)
    z = zf[1:]
    dz = np.diff(zm)
    shear = np.hypot(np.diff(U), np.diff(V)) / dz  # vote V4: vector shear magnitude
    dth = np.diff(th) / dz
    sl = share[1:]
    first = 0
    while first < sl.size and sl[first] > SGS_SHARE_MAX:  # contiguous near-wall block only
        first += 1
    keep = (np.arange(z.size) >= first) & (z / zi <= ETA_MAX)
    zeta = z / length
    out = {"first_kept_z": float(z[first]) if first < z.size else np.nan, "levels_dropped": first, "ustar": ustar, "Q": q0, "L": length, "zi": zi, "minus_zi_over_L": -zi / length, "dz": float(dz[0]),
           "n_levels": int(zm.size), "z": z[keep], "zeta": zeta[keep], "eta": (z / zi)[keep],
           "phi_m": (KAPPA * z * shear / ustar)[keep], "phi_h": (KAPPA * z * dth / theta_star)[keep]}
    with np.errstate(all="ignore"):
        for kind in "mh":
            out["r_" + kind] = np.log(out["phi_" + kind] / sf.phi(BASE, kind, out["zeta"]))
    return out


def within_slope(c, kind):
    """OLS slope of r on ln z inside one case (z/L and z/zi both proportional to z here)."""
    ok = np.isfinite(c["r_" + kind])
    if ok.sum() < 3:
        return None
    x, y = np.log(c["z"][ok]), c["r_" + kind][ok]
    b = np.polyfit(x, y, 1)[0]
    return {"slope": float(b), "n": int(ok.sum()), "r_mean": float(y.mean())}


ZETA_GRID_PER_DECADE = 10


def _cross_rows(cases, kind, drop=None):
    """Each case's r interpolated (in ln|z/L|) to common z/L points shared by >= 3 cases: exact fixed z/L.
    Rows (zeta point, case, r, ln z = ln(zeta L), ln zi)."""
    use = {c: d for c, d in cases.items() if c != drop}
    rng = {}
    for c, d in use.items():
        ok = np.isfinite(d["r_" + kind])
        if ok.sum() >= 2:
            lz = np.log(np.abs(d["zeta"][ok]))
            rng[c] = (lz.min(), lz.max(), lz, d["r_" + kind][ok])
    if len(rng) < 3:
        return []
    lo = min(v[0] for v in rng.values())
    hi = max(v[1] for v in rng.values())
    grid = np.arange(np.floor(lo * ZETA_GRID_PER_DECADE / np.log(10)), np.ceil(hi * ZETA_GRID_PER_DECADE / np.log(10)) + 1)
    grid = grid * np.log(10) / ZETA_GRID_PER_DECADE
    rows = []
    for g in grid:
        inside = [c for c, v in rng.items() if v[0] <= g <= v[1]]
        if len(inside) < 3:
            continue
        for c in inside:
            lz, r = rng[c][2], rng[c][3]
            o = np.argsort(lz)
            rows.append((g, c, float(np.interp(g, lz[o], r[o])), float(g + np.log(abs(use[c]["L"]))), float(np.log(use[c]["zi"]))))
    return rows


def _fe_fit(rows):
    pts = np.array([r[0] for r in rows])
    y = np.array([r[2] for r in rows])
    X = np.array([[r[3], r[4]] for r in rows])
    yd, Xd = y.copy(), X.copy()
    for g in np.unique(pts):
        m = pts == g
        yd[m] -= y[m].mean()
        Xd[m] -= X[m].mean(0)
    rank = int(np.linalg.matrix_rank(Xd))
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0] if rank == 2 else np.full(2, np.nan)
    eta = Xd[:, 0] - Xd[:, 1]  # within-point demeaned ln(z/zi)
    b_eta = float(np.sum(eta * yd) / np.sum(eta**2)) if np.sum(eta**2) > 0 else np.nan
    cond = float(np.linalg.cond(Xd.T @ Xd)) if rank == 2 else np.inf
    corr = float(np.corrcoef(Xd[:, 0], Xd[:, 1])[0, 1]) if rank == 2 else np.nan
    return {"b_z": float(beta[0]), "b_zi": float(beta[1]), "b_eta": b_eta, "rank": rank, "cond": cond,
            "corr_lnz_lnzi": corr, "n_rows": len(rows), "n_points": int(len(np.unique(pts)))}


def cross_case(cases, kind):
    """Fixed-z/L test across cases (each case computed on its own profile): r on ln z and ln zi with z/L-point
    fixed effects, plus r on ln(z/zi) alone; case-deletion (jackknife over cases) for uncertainty and leverage."""
    rows = _cross_rows(cases, kind)
    if len(rows) < 4:
        return None
    out = _fe_fit(rows)
    jack = {}
    for c in cases:
        rr = _cross_rows(cases, kind, drop=c)
        if len(rr) >= 4:
            jack[c] = _fe_fit(rr)
    complete = len(jack) == len(cases)
    for k in ("b_z", "b_zi", "b_eta"):
        v = np.array([j[k] for j in jack.values()])
        if not (complete and np.all(np.isfinite(v))):  # a failed deletion withholds the uncertainty
            out[k + "_jack_se"], out[k + "_drop_range"] = np.nan, [np.nan, np.nan]
            continue
        n = v.size
        out[k + "_jack_se"] = float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2))) if n > 2 else np.nan
        out[k + "_drop_range"] = [float(v.min()), float(v.max())] if n else [np.nan, np.nan]
    out["drop_one"] = {c: {k: j[k] for k in ("b_z", "b_zi", "b_eta")} for c, j in jack.items()}
    return out


def within_common(cases, kind, z_lo, z_hi):
    """Within-case slope on a fixed physical height window, so grids are compared on common support."""
    out = {}
    for c, d in cases.items():
        m = (d["z"] >= z_lo[c]) & (d["z"] <= z_hi[c]) & np.isfinite(d["r_" + kind])
        out[c] = (float(np.polyfit(np.log(d["z"][m]), d["r_" + kind][m], 1)[0]), int(m.sum())) if m.sum() >= 3 else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--les-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    cf.register(BASE, "J3", J3_V5, "benchmark v5 global fit")
    files = {}
    for p in sorted(glob.glob(f"{a.les_dir}/profiles_case*_*.txt")):
        case, grid = re.match(r"profiles_case(\w+?)_(\d+)\.txt", os.path.basename(p)).groups()
        files[(case, int(grid))] = p
    res = {"choices": {"sgs_share_max": SGS_SHARE_MAX, "eta_max": ETA_MAX, "baseline": "J3 v5 global"},
           "cases": {}, "within": {}, "cross_case": {}, "convergence": {}}
    full = {c: load(p) for (c, g), p in files.items() if g == 512}
    for c, d in full.items():
        res["cases"][c] = {k: d[k] for k in ("ustar", "Q", "L", "zi", "minus_zi_over_L", "dz", "n_levels",
                                             "levels_dropped", "first_kept_z")}
        res["cases"][c]["n_used"] = int(d["z"].size)
        res["within"][c] = {k: within_slope(d, k) for k in "mh"}
    res["within_note"] = "NOT a z/zi test: z, z/L and z/zi are collinear inside one case"
    res["cross_case"]["512_all"] = {k: cross_case(full, k) for k in "mh"}
    for g in CONVERGENCE_GRIDS:
        sub = {c: load(p) for (c, gg), p in files.items() if gg == g and c in ("A", "C", "E")}
        res["convergence"][g] = {"dropped": {c: [d["levels_dropped"], d["first_kept_z"]] for c, d in sub.items()},
                                 "within": {c: {k: within_slope(d, k) for k in "mh"} for c, d in sub.items()},
                                 "cross_ACE": {k: cross_case(sub, k) for k in "mh"},
                                 "profiles": {c: {"z": d["z"].tolist(), "r_m": d["r_m"].tolist(), "r_h": d["r_h"].tolist()}
                                              for c, d in sub.items()}}
    grids = {g: {c: load(p) for (c, gg), p in files.items() if gg == g and c in ("A", "C", "E")} for g in CONVERGENCE_GRIDS}
    z_lo = {c: max(grids[g][c]["z"].min() for g in CONVERGENCE_GRIDS) for c in ("A", "C", "E")}
    z_hi = {c: min(grids[g][c]["z"].max() for g in CONVERGENCE_GRIDS) for c in ("A", "C", "E")}
    res["convergence_common_support"] = {"z_lo": z_lo, "z_hi": z_hi, **{str(g): {k: within_common(grids[g], k, z_lo, z_hi)
                                                                                for k in "mh"} for g in CONVERGENCE_GRIDS}}
    res["profiles_512"] = {c: {k: d[k].tolist() for k in ("z", "zeta", "eta", "r_m", "r_h")} for c, d in full.items()}
    with open(a.out, "w") as h:
        json.dump(res, h, indent=1, default=float)
    plot(res, a.figure)
    print("WROTE", a.out, a.figure)


OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    cases = sorted(res["profiles_512"], key=lambda c: res["cases"][c]["minus_zi_over_L"])
    for j, kind in enumerate("mh"):
        for i, c in enumerate(cases):
            p = res["profiles_512"][c]
            ax[0, j].plot(p["eta"], p["r_" + kind], "-o", ms=2, color=OI[1 + i % 7],
                          label=f"{c}: -zi/L {res['cases'][c]['minus_zi_over_L']:.1f}")
        ax[0, j].set(xscale="log", xlabel="z/zi", ylabel=f"ln(phi_{kind} / phi_J3(z/L))",
                     title=f"phi_{kind}, 512 grid, near-wall SGS block > {SGS_SHARE_MAX} dropped, z/zi <= {ETA_MAX}")
        ax[0, j].axhline(0, color=OI[0], lw=0.8)
        for i, g in enumerate(CONVERGENCE_GRIDS):
            for c, p in res["convergence"][g]["profiles"].items():
                ax[1, j].plot(p["z"], p["r_" + kind], ["-", "--", ":", "-."][i], color=OI[{"A": 1, "C": 2, "E": 3}[c]],
                              label=f"{c} {g}" if j == 0 else None)
        ax[1, j].set(xscale="log", xlabel="z (m)", ylabel=f"ln(phi_{kind} / phi_J3)",
                     title=f"grid convergence, cases A C E (line style = grid)")
        ax[1, j].axhline(0, color=OI[0], lw=0.8)
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

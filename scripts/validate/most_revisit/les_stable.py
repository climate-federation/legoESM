#!/usr/bin/env python
"""Stable flux-gradient formulas against stable-boundary-layer LES mean profiles (sensor-free check of the towers).

Data: Bon, Cal & Meyers 2024 (BLM) steady SBLs with subsidence (doi:10.48804/SPJL3Z, CC BY-NC: never commit) and
Huusko et al. 2025 GABLS1 (Nek5000 two resolutions + NCAR LES reference; doi:10.17043/huusko-2025-les-nek5000-1.1).
Per run (never pooled): phi from adjacent-level differences of the mean profiles placed between the levels, scaled by
surface u*, q0 (les_zi.profile_case, same near-wall SGS rule); h = h_tau from the total momentum-flux profile with the
95% criterion of Bon et al. Each formula scored by log-RMSE of phi over the kept levels (tower statistic, the scheme's
own kappa convention), fixed coefficients. Votes: coscientist/les_stable/votes.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import candidate_formulations as cf
import les_zi as lz
import round2
import standard_formulations as sf

KAPPA = lz.KAPPA
H_TAU_FRACTION = 0.05  # Bon et al. 2024 Sect. 3.2: tau(z) = 0.05 u*^2, h_tau = z / 0.95
ETA_SENSITIVITY = 0.2  # vote V2: labelled sensitivity arm
STANDARD = ("dyer1974", "businger1971", "hogstrom1988", "beljaars_holtslag1991", "grachev2007_sheba", "gryanik2020")
RESOLUTION_REPEATS = {"S6f": "S6", "S7f": "S7", "T4f": "T4"}  # vote V7: checks, not cases
NO_PROFILES = ("WSBL-L19", "VSBL-L19")  # bulk values only in the archive


def h_tau(zf, uw, vw, ustar):
    """First height where the total stress magnitude falls to 0.05 u*^2, divided by 0.95 (linear interpolation)."""
    tau = np.hypot(uw, vw) / ustar**2
    i = int(np.argmax(tau <= H_TAU_FRACTION))
    assert i > 0 and tau[i] <= H_TAU_FRACTION, "stress never falls to the h_tau threshold"
    z = zf[i - 1] + (H_TAU_FRACTION - tau[i - 1]) * (zf[i] - zf[i - 1]) / (tau[i] - tau[i - 1])
    return float(z / (1 - H_TAU_FRACTION))


def _case(zm, U, V, th, zf, uw, vw, wt, share, ustar, q0, theta0, eta_max):
    """Means at zm; total fluxes and SGS share at zf, the midpoints of zm (gradient heights)."""
    assert np.allclose(zf, 0.5 * (zm[1:] + zm[:-1])), "fluxes must sit midway between mean levels"
    dz = np.diff(zm)
    h = h_tau(zf, uw, vw, ustar)
    out = lz.profile_case(zf, np.hypot(np.diff(U), np.diff(V)) / dz, np.diff(th) / dz, share, ustar, q0, theta0, h,
                          eta_max=eta_max, local=(uw, vw, wt))
    out["h_tau"] = h
    return out


def load_bon(folder, row, eta_max=lz.ETA_MAX):
    cc = np.genfromtxt(os.path.join(folder, "profiles_cc.csv"), delimiter=",", names=True)
    st = np.genfromtxt(os.path.join(folder, "profiles_st.csv"), delimiter=",", names=True)
    zm, zf = cc["z"], st["z_st"][1:-1]  # edge faces carry zeroed derivatives by construction
    mid = lambda a: 0.5 * (a[1:] + a[:-1])  # noqa: E731
    sgs_u, sgs_v, sgs_t = st["tau_xz_sgs"][1:-1], st["tau_yz_sgs"][1:-1], st["q_z_sgs"][1:-1]
    uw, vw, wt = mid(cc["uw_res"]) + sgs_u, mid(cc["vw_res"]) + sgs_v, mid(cc["wtheta_res"]) + sgs_t
    with np.errstate(all="ignore"):
        share = np.hypot(sgs_u, sgs_v) / np.hypot(uw, vw)
    out = _case(zm, cc["u"], cc["v"], cc["theta"], zf, uw, vw, wt, share, float(row["u_star"]), float(row["q_0"]),
                float(row["theta_ref"]), eta_max)
    keep = np.isin(zf, out["z"])
    provided = np.hypot(st["dudz"][1:-1], st["dvdz"][1:-1])[keep]  # vote V5: instrument check, reported
    out["check_provided_phi_m"] = float(np.nanmax(np.abs(KAPPA * zf[keep] * provided / out["ustar"] / out["phi_m"] - 1)))
    out["check_provided_phi_h"] = float(np.nanmax(np.abs(KAPPA * zf[keep] * st["dthetadz"][1:-1][keep]
                                                         / (-out["Q"] / out["ustar"]) / out["phi_h"] - 1)))
    out["published"] = {k: float(row[k]) for k in ("u_star", "q_0", "L_O", "h_tau", "h_theta")}
    return out


def load_nek(path, eta_max=lz.ETA_MAX):
    """Nek5000: y is vertical, v vertical velocity, w meridional; fluxes = resolved - SGS (nutot dU/dy)."""
    import xarray as xr
    d = xr.open_dataset(path)
    _, first = np.unique(d["y"].values, return_index=True)  # element-boundary nodes appear twice
    d = d.isel(y=np.sort(first))
    y = d["y"].values
    uw, vw, wt = d["uv"].values - d["nutotdudy"].values, d["vw"].values - d["nutotdwdy"].values, \
        d["tv"].values - d["xitotdtdy"].values
    with np.errstate(all="ignore"):
        share = np.hypot(d["nutotdudy"].values, d["nutotdwdy"].values) / np.hypot(uw, vw)
    ustar = float(np.hypot(d["nutotdudy"].values[0], d["nutotdwdy"].values[0]) ** 0.5)  # wall stress (vote V6)
    q0 = float(wt[0])
    mid = lambda a: 0.5 * (a[1:] + a[:-1])  # noqa: E731
    return _case(y, d["u"].values, d["w"].values, d["t"].values, mid(y), mid(uw), mid(vw), mid(wt), mid(share),
                 ustar, q0, float(d["t"].values[0]), eta_max)


def read_xy(path):
    """NCAR .xy: '#k "NAME' headers then (value, coordinate) rows; first block per name kept."""
    blocks, name, rows, consts = {}, None, [], {}
    for line in open(path):
        if line.startswith('#k "'):
            if name and rows and name not in blocks:
                blocks[name] = np.array(rows)
            tag = line[4:].strip().strip('"').strip()
            if "=" in tag:
                k, v = tag.split("=")
                try:
                    consts[k.strip()] = float(v)
                except ValueError:  # text constants such as CASE
                    pass
            name, rows = tag, []
        elif line.startswith("#k"):  # bare '#k' ends a block (axis-marker rows follow it)
            if name and rows and name not in blocks:
                blocks[name] = np.array(rows)
            name, rows = None, []
        elif name and not line.startswith("#") and line.strip():
            rows.append([float(x) for x in line.split()])
    if name and rows and name not in blocks:
        blocks[name] = np.array(rows)
    return blocks, consts


def load_ncar(path, eta_max=lz.ETA_MAX):
    b, c = read_xy(path)
    zm, zf = b["UXYM"][:, 1], b["UWTOT"][:, 1]
    sel = (zf > zm[0]) & (zf < zm[-1])
    zf = zf[sel]
    uw, vw, wt = b["UWTOT"][sel, 0], b["VWTOT"][sel, 0], b["WTTOT"][sel, 0]
    with np.errstate(all="ignore"):
        share = np.hypot(b["UWSGS"][sel, 0], b["VWSGS"][sel, 0]) / np.hypot(uw, vw)
    out = _case(zm, b["UXYM"][:, 0], b["VXYM"][:, 0], b["TXYM"][:, 0], zf, uw, vw, wt, share,
                c["Ustar_bar"], c["Qstar_bar"], float(b["TXYM"][0, 0]), eta_max)
    phim = b["PHIM"]  # NCAR's own phi_m against z/zi (vote V5 check)
    own = np.interp(out["eta"] * out["h_tau"] / c["Zi_les_bar"], phim[:, 1], phim[:, 0])
    out["check_ncar_phi_m"] = float(np.nanmax(np.abs(out["phi_m"] / own - 1)))
    out["published"] = {"u_star": c["Ustar_bar"], "q_0": c["Qstar_bar"], "L_O": c["Amo_bar"], "zi": c["Zi_les_bar"]}
    return out


def register_formulas(winners_path):
    """J3 v5 global and the round-2 stable winners; unspecified kind falls back to Dyer."""
    cf.register("J3_global", "J3", lz.J3_V5, "benchmark v5 global fit")
    names = []
    w = json.load(open(winners_path))
    dyer = sf.SCHEMES["dyer1974"]
    for kind, entries in w.items():
        for i, (label, spec, p) in enumerate(entries):
            if "not beyond noise" in label:
                continue
            f, _, _ = round2.build(spec)
            g = (lambda f_, p_: (lambda z: f_(z, p_)))(f, list(p))
            name = f"round2_{kind}{i}"
            sf.SCHEMES[name] = (g, dyer[1], KAPPA, label, "coscientist/round2/winners.json") if kind == "m" else \
                (dyer[0], g, KAPPA, label, "coscientist/round2/winners.json")
            for k in "mh":
                sf._VECTOR.pop((name, k), None)
            names.append((name, kind))
    return names


def score(case, formulas, arm=""):
    """log-RMSE per formula over kept stable levels; observed phi and zeta in each scheme's kappa convention."""
    out = {}
    for kind in "mh":
        phi, zeta = case["phi_" + kind + arm], case["zeta" + arm]
        sel = (zeta > 0) & (phi > 0) & np.isfinite(phi)  # one level set for every formula
        res, n_valid = {}, {}
        for name, only in formulas:
            if only not in (None, kind):
                continue
            s = sf.SCHEMES[name][2] / KAPPA
            st = sf._stats(phi[sel] * s, sf.phi(name, kind, zeta[sel] * s))
            res[name] = st.get("log_rmse", np.nan)
            n_valid[name] = int(round(st["n"] * (1 - st["frac_sign_mismatch"]))) if st["n"] else 0
        order = sorted((v, k) for k, v in res.items() if np.isfinite(v))
        out[kind] = {"log_rmse": res, "n": int(sel.sum()), "n_scored": n_valid, "best": order[0][1] if order else None,
                     "rank": {k: i + 1 for i, (_, k) in enumerate(order)},
                     "zeta_range": [float(zeta[sel].min()), float(zeta[sel].max())] if sel.any() else None}
    return out


def paired(runs, formulas, ref=("J3_global", "dyer1974")):
    """Per-run paired log-RMSE differences (formula - ref) over primary-arm runs: median and count of runs better."""
    use = [c for c in runs if c not in RESOLUTION_REPEATS and not c.startswith("huusko_nek")]
    out = {}
    for kind in "mh":
        for name, only in formulas:
            if only not in (None, kind):
                continue
            for r in ref:
                d = np.array([runs[c]["score"][kind]["log_rmse"].get(name, np.nan) - runs[c]["score"][kind]["log_rmse"][r]
                              for c in use])
                d = d[np.isfinite(d)]
                out[f"{kind}|{name}|vs_{r}"] = {"median": float(np.median(d)) if d.size else np.nan,
                                               "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
                                               "tie": int((d == 0).sum()), "n_runs": int(d.size)}
    return out


def resolution_pair(base, fine):
    """Fine-run phi interpolated to the base run's kept heights (ponytail: interpolates phi, not re-differenced
    fine gradients over coarse intervals). Domain height and averaging window also differ for T4f."""
    out = {}
    for kind in "mh":
        zf, pf = fine["z"], fine["phi_" + kind]
        m = (base["z"] >= zf.min()) & (base["z"] <= zf.max())
        out[kind] = {"z": base["z"][m].tolist(), "phi_base": base["phi_" + kind][m].tolist(),
                     "phi_fine": np.interp(base["z"][m], zf, pf).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bon", required=True)
    ap.add_argument("--huusko", required=True)
    ap.add_argument("--winners", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    formulas = [(n, None) for n in STANDARD] + [("J3_global", None)] + register_formulas(a.winners)
    rows = {r["CaseName"]: r for r in csv.DictReader(open(os.path.join(a.bon, "flowproperties_overview.csv")))}
    loaders = {c: (lambda c_=c, e=lz.ETA_MAX: load_bon(os.path.join(a.bon, f"Case_{c_}"), rows[c_], e))
               for c in rows if c not in NO_PROFILES}
    for n in (64, 25):
        loaders[f"huusko_nek{n}"] = (lambda n_=n, e=lz.ETA_MAX: load_nek(f"{a.huusko}/stable/stable_{n_}_TKE_profiles.nc", e))
    loaders["huusko_ncar"] = lambda e=lz.ETA_MAX: load_ncar(f"{a.huusko}/reference/stable.xy", e)
    res = {"votes": "coscientist/les_stable/votes.md", "formulas": [f[0] for f in formulas], "runs": {}}
    cases = {}
    for c, ld in loaders.items():
        d = ld()
        cases[c] = d
        sens = ld(e=ETA_SENSITIVITY)
        res["runs"][c] = {"scales": {k: d[k] for k in ("ustar", "Q", "L", "h_tau", "levels_dropped", "first_kept_z")},
                          "h_over_L": d["h_tau"] / d["L"], "n_kept": int(d["z"].size),
                          "published": d.get("published"),
                          "checks": {k: v for k, v in d.items() if k.startswith("check_")},
                          "score": score(d, formulas), "score_eta_0p2": score(sens, formulas),
                          "score_local": score(d, formulas, "_loc"),
                          "profile": {k: d[k].tolist() for k in ("z", "zeta", "eta", "phi_m", "phi_h", "r_m", "r_h")}}
    fit_cases = {c: d for c, d in cases.items() if c not in RESOLUTION_REPEATS and not c.startswith("huusko_nek")}
    res["cross_case_z_over_h"] = {k: lz.cross_case(fit_cases, k) for k in "mh"}
    res["paired"] = paired(res["runs"], formulas)
    res["resolution_pairs"] = {f: resolution_pair(cases[b], cases[f]) for f, b in RESOLUTION_REPEATS.items()}
    res["cross_case_note"] = "Bon runs + NCAR GABLS1; f-runs and Nek5000 repeats excluded (vote V7)"
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    plot(res, formulas, a.figure)
    print("WROTE", a.out, a.figure)


def plot(res, formulas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oi = lz.OI
    runs = [c for c in res["runs"] if c not in RESOLUTION_REPEATS]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    zg = np.linspace(1e-3, 0.6, 200)
    styles = ["-", "--", ":", "-."]
    for j, kind in enumerate("mh"):
        for i, c in enumerate(runs):
            p = res["runs"][c]["profile"]
            ax[j].plot(p["zeta"], p["phi_" + kind], "o", ms=3, color=oi[1 + i % 7], alpha=0.8, label=c if j == 0 else None)
        for i, (name, only) in enumerate(f for f in formulas if f[1] in (None, kind)):
            s = sf.SCHEMES[name][2] / KAPPA  # curve in the kappa 0.4 coordinates of the observations
            ax[j].plot(zg, sf.phi(name, kind, zg * s) / s, styles[i % 4], color=oi[0] if name in ("J3_global", "dyer1974") else "0.5",
                       lw=1.6 if name in ("J3_global", "dyer1974") else 0.8, label=name if j == 1 else None)
        ax[j].set(xlabel="z/L", ylabel=f"phi_{kind}", title=f"phi_{kind}: stable LES, z/h <= {lz.ETA_MAX}, near-wall SGS block dropped")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

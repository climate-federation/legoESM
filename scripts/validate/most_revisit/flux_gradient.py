#!/usr/bin/env python
"""Observed Monin-Obukhov flux-gradient functions from tower profiles and eddy fluxes.

phi_m = kappa z / u* dU/dz and phi_h = kappa z / theta* dtheta/dz, with
theta* = -(w'theta')/u* and local scaling: fluxes and L are taken at the height
of the gradient. Gradients use two levels at the geometric-mean height,
dX/dz = (X2 - X1) / (z ln(z2/z1)), exact for a logarithmic profile.

Falkenberg (FESSTVaL 2021, doi:10.25592/uhhfdm.14516) is the first reader; a new
tower only needs a loader that returns the same columns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import numpy as np

from legoesm import constants
from legoesm.core.bulk_flux import psi_h, psi_m

KAPPA = constants.kappa_von_karman
SECONDS_PER_AVERAGE = 1800
VIRTUAL_COEFF = 1.0 / constants.epsilon - 1.0


def log_gradient(x1, x2, z1, z2):
    """Return (z_g, dX/dz at z_g) for two levels, z_g the geometric mean."""
    z_g = np.sqrt(z1 * z2)
    return z_g, (x2 - x1) / (z_g * np.log(z2 / z1))


def potential_temperature_offset(z):
    """Dry-adiabatic offset [K] between height z and the local surface."""
    return constants.g / constants.c_pd * z


def obukhov_length(ustar, w_theta_v, theta_v):
    """L = -u*^3 theta_v / (kappa g w'theta_v'); positive when stable."""
    return -ustar**3 * theta_v / (KAPPA * constants.g * w_theta_v)


def phi_observed(gradient, scale, z):
    """kappa z / scale * gradient, for scale = u* or theta*."""
    return KAPPA * z * gradient / scale


def phi_model(zeta, scheme, kind):
    """phi = 1 - zeta dpsi/dzeta from the model's own psi functions."""
    import jax
    import jax.numpy as jnp

    psi = psi_m if kind == "m" else psi_h
    dpsi = jax.vmap(jax.grad(lambda x: psi(x, scheme)))
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    return np.asarray(1.0 - zeta * dpsi(zeta))


def binned_median(zeta, phi, edges, min_count):
    """Median phi with bootstrap-free quartiles per zeta bin; NaN below min_count."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (zeta >= lo) & (zeta < hi) & np.isfinite(phi)
        n = int(sel.sum())
        if n < min_count:
            rows.append((float(lo), float(hi), n, np.nan, np.nan, np.nan, np.nan))
            continue
        q1, q2, q3 = np.percentile(phi[sel], [25, 50, 75])
        rows.append((float(lo), float(hi), n, float(np.median(zeta[sel])),
                     float(q1), float(q2), float(q3)))
    return rows


def _read_daily(pattern, names):
    import netCDF4

    out = {k: [] for k in names}
    times = []
    heights = None
    for path in sorted(glob.glob(pattern)):
        with netCDF4.Dataset(path) as d:
            times.append(np.asarray(d["time"][:], dtype=float))
            for k in names:
                out[k].append(np.ma.filled(np.ma.asarray(d[k][:], dtype=float), np.nan))
            if "height2" in d.variables:
                heights = np.asarray(d["height2"][:], dtype=float)
    if not times:
        raise FileNotFoundError(pattern)
    return np.concatenate(times), {k: np.concatenate(v) for k, v in out.items()}, heights


def to_half_hour(t, x, flag, samples=3):
    """Mean of good values in each half hour ending at the stamp; `samples` per half hour."""
    if np.unique(t).size != t.size:
        raise ValueError("duplicate time stamps would pass as complete half hours")
    x = np.where(flag == 1, x, np.nan)
    key = np.ceil(t / SECONDS_PER_AVERAGE).astype(np.int64)
    keys = np.unique(key)
    idx = np.searchsorted(keys, key)
    shape = (keys.size,) + x.shape[1:]
    total, count = np.zeros(shape), np.zeros(shape)
    good = np.isfinite(x)
    np.add.at(total, idx, np.where(good, x, 0.0))
    np.add.at(count, idx, good)
    # Every sample must be present, so a half hour is never a partial mean.
    return keys * SECONDS_PER_AVERAGE, np.where(count == samples, total / np.maximum(count, 1), np.nan)


def load_falkenberg(root):
    """Half-hourly profile and flux arrays keyed by mast and sensor."""
    mast = {}
    for name, sub, tag in (("low", "10m_mast", "mett01_l1_any_v02"),
                           ("high", "99m_mast", "mett00_l1_any_v01")):
        t, v, z = _read_daily(
            f"{root}/falkenberg_tower/level1/{sub}/*/sups_rao_{tag}_*.nc",
            ["ta", "ta_flag", "hur", "hur_flag", "wspeed", "wspeed_flag"])
        t30, ta = to_half_hour(t, v["ta"], v["ta_flag"])
        _, rh = to_half_hour(t, v["hur"], v["hur_flag"])
        _, ws = to_half_hour(t, v["wspeed"], v["wspeed_flag"])
        mast[name] = {"t": t30, "z": z, "ta": ta, "hur": rh, "wspeed": ws}
    flux = {}
    for name, tag in (("2m", "turb00"), ("50m", "turb01"), ("90m", "turb02")):
        t, v, _ = _read_daily(
            f"{root}/falkenberg_turb/level2/*/sups_rao_{tag}_l2_any_v00_*.nc",
            ["ustar", "ustar_flag", "hfss", "hfss_flag", "hfls", "hfls_flag", "tsonic", "wvfs", "wspeed", "wdir"])
        flux[name] = {"t": t, **v}
    tp, vp, _ = _read_daily(f"{root}/falkenberg_basic/level1/*/sups_rao_mets00_l1_pa_v01_*.nc",
                            ["pa", "pa_flag"])
    t30, pa = to_half_hour(tp, vp["pa"], vp["pa_flag"])
    for f in flux.values():
        f["pa"] = np.interp(f["t"], t30, pa, left=np.nan, right=np.nan)
    return mast, flux


def phi_samples(mast, flux, z1, z2, min_ustar, min_abs_dtheta):
    """Per-half-hour zeta, phi_m and phi_h for one level pair and one flux height."""
    t = np.intersect1d(mast["t"], flux["t"])
    im, jf = np.searchsorted(mast["t"], t), np.searchsorted(flux["t"], t)
    k1, k2 = (int(np.argmin(np.abs(mast["z"] - z))) for z in (z1, z2))
    th1 = mast["ta"][im, k1] + potential_temperature_offset(z1)
    th2 = mast["ta"][im, k2] + potential_temperature_offset(z2)
    z_g, dth = log_gradient(th1, th2, z1, z2)
    _, du = log_gradient(mast["wspeed"][im, k1], mast["wspeed"][im, k2], z1, z2)
    ustar, hfss, ts = flux["ustar"][jf], flux["hfss"][jf], flux["tsonic"][jf]
    good = (flux["ustar_flag"][jf] == 3) & (flux["hfss_flag"][jf] == 3) & (ustar > min_ustar)
    rho = flux["pa"][jf] / (constants.R_d * ts)
    w_theta = hfss / (rho * constants.c_pd)
    w_q = flux["wvfs"][jf] / rho
    L = obukhov_length(ustar, w_theta + VIRTUAL_COEFF * ts * w_q, ts)
    zeta = np.where(good, z_g / L, np.nan)
    theta_star = -w_theta / ustar
    phi_m = phi_observed(du, ustar, z_g)
    phi_h = phi_observed(dth, theta_star, z_g)
    phi_h = np.where(np.abs(th2 - th1) >= min_abs_dtheta, phi_h, np.nan)
    return z_g, zeta, phi_m, phi_h


def log_interp(x1, x2, z1, z2, z):
    """Value at z on the straight line through two levels in ln z."""
    return x1 + (x2 - x1) * np.log(z / z1) / np.log(z2 / z1)


def instrument_controls(mast, flux):
    """Checks the phi tables depend on: level labels, time stamps, offsets, sample loss."""
    out = {}
    for tag, m, f, z1, z2, zs in (("2m", "low", "2m", 2.0, 4.0, 2.3),
                                  ("50m", "high", "50m", 40.0, 60.0, 50.3),
                                  ("90m", "high", "90m", 80.0, 98.0, 90.3)):
        a, b = mast[m], flux[f]
        t = np.intersect1d(a["t"], b["t"])
        im, jf = np.searchsorted(a["t"], t), np.searchsorted(b["t"], t)
        k1, k2 = (int(np.argmin(np.abs(a["z"] - z))) for z in (z1, z2))
        cup = log_interp(a["wspeed"][im, k1], a["wspeed"][im, k2], z1, z2, zs)
        sonic = b["wspeed"][jf]
        lags = {}
        for lag in range(-3, 4):
            x, y = (cup[3 + lag:cup.size - 3 + lag], sonic[3:sonic.size - 3])
            ok = np.isfinite(x) & np.isfinite(y)
            lags[lag] = float(np.corrcoef(x[ok], y[ok])[0, 1])
        ok = np.isfinite(cup) & np.isfinite(sonic) & (sonic > 2.0)
        out[f"wind_{tag}"] = {"n": int(ok.sum()), "cup_over_sonic": float(np.median(cup[ok] / sonic[ok])),
                              "lag_corr": lags}
    pairs = [(1, 2), (2, 4), (1, 4), (4, 8), (4, 10), (2, 8), (0.5, 1)]
    out["near_neutral_phi_m"] = {}
    for z1, z2 in pairs:
        _, zeta, pm, _ = phi_samples(mast["low"], flux["2m"], z1, z2, 0.1, 0.0)
        ok = (np.abs(zeta) < 0.02) & np.isfinite(pm)
        out["near_neutral_phi_m"][f"{z1}-{z2}"] = {"n": int(ok.sum()), "median": float(np.median(pm[ok])) if ok.any() else None}
    t = np.intersect1d(mast["low"]["t"], flux["2m"]["t"])
    sector = (flux["2m"]["wdir"][np.searchsorted(flux["2m"]["t"], t)] // 45).astype(float)
    out["near_neutral_phi_m_by_sector"] = {}
    for z1, z2 in ((0.5, 1), (1, 4), (4, 10)):
        _, zeta, pm, _ = phi_samples(mast["low"], flux["2m"], z1, z2, 0.1, 0.0)
        ok = (np.abs(zeta) < 0.02) & np.isfinite(pm)
        out["near_neutral_phi_m_by_sector"][f"{z1}-{z2}"] = {
            f"{int(k) * 45}-{int(k) * 45 + 45}": [int((ok & (sector == k)).sum()),
                                                  float(np.median(pm[ok & (sector == k)])) if (ok & (sector == k)).any() else None]
            for k in range(8)}
    out["theta_offset_neutral"] = {}
    for m, f in (("low", "2m"), ("high", "50m")):
        a, b = mast[m], flux[f]
        t = np.intersect1d(a["t"], b["t"])
        im, jf = np.searchsorted(a["t"], t), np.searchsorted(b["t"], t)
        sel = (b["ustar"][jf] > 0.4) & (np.abs(b["hfss"][jf]) < 30.0)
        z = a["z"]
        th = a["ta"][im] + potential_temperature_offset(z)[None, :]
        out["theta_offset_neutral"][m] = {"z": z.tolist(), "n": int(sel.sum()),
                                          "median_minus_lowest": np.median((th - th[:, :1])[sel], axis=0).tolist()}
    b = flux["2m"]
    stable, ok_u, ok_h = b["hfss"] < 0, b["ustar_flag"] == 3, b["hfss_flag"] == 3
    out["stable_loss_2m"] = {"all": int(b["hfss"].size), "hfss_negative": int(stable.sum()),
                             "ustar_flag_good": int((stable & ok_u).sum()),
                             "both_flags_good": int((stable & ok_u & ok_h).sum()),
                             "ustar_gt_0.1": int((stable & ok_u & ok_h & (b["ustar"] > 0.1)).sum()),
                             "hfss_flag_values": {str(v): int(((b["hfss_flag"] == v) & stable).sum())
                                                  for v in np.unique(b["hfss_flag"][np.isfinite(b["hfss_flag"])])}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ustar", type=float, required=True)
    ap.add_argument("--min-abs-dtheta", type=float, required=True)
    ap.add_argument("--min-count", type=int, required=True)
    ap.add_argument("--controls", help="write instrument controls here instead of phi tables")
    a = ap.parse_args()
    mast, flux = load_falkenberg(a.root)
    if a.controls:
        with open(a.controls, "w") as f:
            json.dump(instrument_controls(mast, flux), f, indent=1)
        return
    edges = np.array([-10, -3, -1, -0.5, -0.2, -0.1, -0.05, -0.02,
                      0.02, 0.05, 0.1, 0.2, 0.5, 1, 3, 10])
    pairs = [("low", "2m", 1.0, 4.0), ("high", "50m", 40.0, 60.0), ("high", "90m", 80.0, 98.0)]
    result = {"args": vars(a), "git": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(), "pairs": {}}
    for m, f, z1, z2 in pairs:
        z_g, zeta, pm, ph = phi_samples(mast[m], flux[f], z1, z2, a.min_ustar, a.min_abs_dtheta)
        result["pairs"][f"{z1:g}-{z2:g}m"] = {
            "z_g": float(z_g), "n_zeta": int(np.isfinite(zeta).sum()),
            "n_phi_m": int((np.isfinite(zeta) & np.isfinite(pm)).sum()),
            "n_phi_h": int((np.isfinite(zeta) & np.isfinite(ph)).sum()),
            "phi_m": binned_median(zeta, pm, edges, a.min_count),
            "phi_h": binned_median(zeta, ph, edges, a.min_count)}
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

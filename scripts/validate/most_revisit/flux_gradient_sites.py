#!/usr/bin/env python
"""Observed phi_m and phi_h against z/L for every flux-profile table, per site and level pair.

Rules: 30-min tables, measured eddy flux, both levels above canopy, z-d in zeta and phi,
lowest flux level as the surface flux, sites never pooled. Physical filters live here only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import xarray as xr

from flux_gradient import KAPPA, VIRTUAL_COEFF, binned_median, log_gradient, obukhov_length, phi_model, phi_observed, potential_temperature_offset
from legoesm import constants

MAX_ABS_H = 800.0
MAX_USTAR = 2.0
MAX_WIND = 40.0
EDGES = np.array([-10, -3, -1, -0.5, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 3, 10])
SCHEMES = ("dyer1974", "beljaars_holtslag1991", "grachev2007_sheba", "gryanik2020")
ASSUMED_DISPLACEMENT_FRACTION = 2.0 / 3.0  # no published d found for Hyytiala or CHATS
OFFSET_MAX_ABS_H = 10.0
OFFSET_MIN_USTAR = 0.3
MIN_HEAT_ABS_ZETA = 0.1  # below this, temperature differences are within sensor offset error at most pairs
EXCLUDED = {"ibox_flag1": "sensitivity copy of ibox_kolsass",
            "ibox_kolsass": "flat-terrain rule: Alpine valley floor",
            "fino1": "sonic temperature only, offsets vary with wind direction; 61 m in a wake in sectors 150-300"}


def clean_flux(f, min_ustar):
    """Flux fields with spikes and weak-turbulence times blanked."""
    ok = (np.abs(f["h"]) <= MAX_ABS_H) & (f["ustar"] >= min_ustar) & (f["ustar"] <= MAX_USTAR)
    return ok


def buoyancy_flux(h, le, t, p):
    """w'theta_v' from H, with the moisture term where a latent flux exists."""
    rho = p / (constants.R_d * t)
    moist = np.where(np.isfinite(le), VIRTUAL_COEFF * t * le / (rho * constants.L_v), 0.0)
    return h / (rho * constants.c_pd) + moist


def pairs(n, widest):
    out = [(i, i + 1) for i in range(n - 1)]
    return out + ([(0, n - 1)] if widest and n > 2 else [])


def pair_samples(z, x, i, j, canopy, d):
    """Effective mean height, gradient, and difference for one level pair."""
    z1, z2 = z[:, i], z[:, j]
    above = (z1 > canopy) & (z2 > z1) & (z1 - d > 0)
    a, b = np.where(above, z1 - d, np.nan), np.where(above, z2 - d, np.nan)
    z_g, grad = log_gradient(x[:, i], x[:, j], a, b)
    return z_g, grad, x[:, j] - x[:, i]


def neutral_offset(diff, h, ustar):
    """Median level difference under strong wind with almost no heat flux: sensor offset.

    Selecting on H, not z/L, because only a vanishing heat flux implies a vanishing gradient.
    """
    sel = (np.abs(h) < OFFSET_MAX_ABS_H) & (ustar > OFFSET_MIN_USTAR) & np.isfinite(diff)
    return float(np.median(diff[sel])) if sel.sum() >= 30 else np.nan


def displacement(ds):
    """Recorded d, else the assumed fraction of canopy height, else zero for a bare site."""
    d, canopy = float(ds.attrs["displacement_m"]), float(ds.attrs["canopy_height_m"])
    if np.isfinite(d):
        return d
    return ASSUMED_DISPLACEMENT_FRACTION * canopy if np.isfinite(canopy) else 0.0


def site_samples(ds, flux_level, min_ustar, min_dtheta, widest, offset_correct):
    """Rows of (kind, z1, z2, z_g, zeta, phi) as dict keyed by pair label."""
    canopy = float(ds.attrs["canopy_height_m"]) if np.isfinite(float(ds.attrs["canopy_height_m"])) else 0.0
    d = displacement(ds)
    f = {k: ds[k].values[:, flux_level].astype(float) for k in ("ustar", "h", "le", "t_flux")}
    p = ds.pressure.values.astype(float)
    good = clean_flux(f, min_ustar)
    w_tv = buoyancy_flux(f["h"], f["le"], f["t_flux"], p)
    length = obukhov_length(f["ustar"], w_tv, f["t_flux"])
    rho = p / (constants.R_d * f["t_flux"])
    theta_star = -(f["h"] / (rho * constants.c_pd)) / f["ustar"]
    wind = np.where(ds.wind.values <= MAX_WIND, ds.wind.values, np.nan).astype(float)
    z_u, z_t = ds.z_u.values.astype(float), ds.z_t.values.astype(float)
    theta = ds.temp.values.astype(float) + potential_temperature_offset(z_t)
    out = {}
    for kind, z, x in (("m", z_u, wind), ("h", z_t, theta)):
        for i, j in pairs(x.shape[1], widest):
            z_g, grad, diff = pair_samples(z, x, i, j, canopy, d)
            if not np.isfinite(z_g).any():
                continue
            zeta = np.where(good, z_g / length, np.nan)
            if kind == "m":
                phi = phi_observed(grad, f["ustar"], z_g)
            else:
                off = neutral_offset(diff, f["h"], f["ustar"])
                if offset_correct:
                    if not np.isfinite(off):
                        continue
                    grad = grad - off / (z_g * np.log((z[:, j] - d) / (z[:, i] - d)))
                phi = phi_observed(grad, theta_star, z_g)
                # corrected difference in every arm, so arms differ only by the correction
                phi = np.where(np.abs(diff - (off if np.isfinite(off) else 0.0)) >= min_dtheta, phi, np.nan)
            key = f"{kind}:{np.nanmedian(z[:, i]):.4g}-{np.nanmedian(z[:, j]):.4g}"
            out[key] = (float(np.nanmedian(z_g)), zeta, phi)
    return out


def lowest_flux_level(ds):
    """Lowest sonic above the canopy: an in-canopy flux is not a surface-layer flux."""
    z = np.nanmedian(ds.z_f.values, axis=0)
    canopy = float(ds.attrs.get("canopy_height_m", 0.0))
    return int(np.nanargmin(np.where(z > canopy, z, np.nan))) if np.isfinite(canopy) else int(np.nanargmin(z))


def nearest_flux_level(ds, z_g):
    return int(np.nanargmin(np.abs(np.log(np.nanmedian(ds.z_f.values, axis=0) / z_g))))


def nearest_samples(ds, min_ustar, min_dtheta):
    """Each pair against the flux level closest to its height; no offset correction."""
    out = {}
    for level in range(ds.sizes["lev_f"]):
        s = site_samples(ds, level, min_ustar, min_dtheta, False, False)
        out.update({k: v for k, v in s.items() if nearest_flux_level(ds, v[0]) == level})
    return out


def open_site(path):
    ds = xr.open_dataset(path, decode_times=False)
    if "ustar_suspect_from_s" in ds.attrs:
        ds = ds.sel(time=ds.time < ds.attrs["ustar_suspect_from_s"])
    return ds


def summarise(samples, min_count, max_zg):
    """Per pair and per site (pooled over pairs at or below max_zg) binned medians."""
    rows, pooled = {}, {"m": ([], []), "h": ([], [])}
    for key, (z_g, zeta, phi) in samples.items():
        ok = np.isfinite(zeta) & np.isfinite(phi)
        rows[key] = {"z_g": z_g, "n": int(ok.sum()), "bins": binned_median(zeta, phi, EDGES, min_count)}
        if z_g <= max_zg:
            keep = ok & (np.abs(zeta) >= MIN_HEAT_ABS_ZETA) if key[0] == "h" else ok
            pooled[key[0]][0].append(zeta[keep])
            pooled[key[0]][1].append(phi[keep])
    site = {}
    for kind, (a, b) in pooled.items():
        if a:
            a, b = np.concatenate(a), np.concatenate(b)
            site[kind] = {"n": int(a.size), "bins": binned_median(a, b, EDGES, min_count)}
    return {"pairs": rows, "site": site}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ustar", type=float, required=True)
    ap.add_argument("--min-abs-dtheta", type=float, required=True)
    ap.add_argument("--sensitivity-dtheta", type=float, required=True)
    ap.add_argument("--min-count", type=int, required=True)
    ap.add_argument("--max-zg", type=float, required=True)
    a = ap.parse_args()
    result = {"args": vars(a), "sites": {}}
    for path in sorted(glob.glob(f"{a.tables}/*.nc")):
        name = os.path.basename(path)[:-3]
        if name in EXCLUDED:
            continue
        ds = open_site(path)
        low = lowest_flux_level(ds)
        arms = {"primary": (low, False, False), "widest": (low, True, False), "offset_corrected": (low, False, True)}
        result["sites"][name] = {"z_flux": float(np.nanmedian(ds.z_f.values[:, low])), "canopy": float(ds.attrs["canopy_height_m"]),
                                 "displacement": displacement(ds)}
        for arm, (level, widest, off) in arms.items():
            s = site_samples(ds, level, a.min_ustar, a.min_abs_dtheta, widest, off)
            result["sites"][name][arm] = summarise(s, a.min_count, a.max_zg)
        s = site_samples(ds, low, a.min_ustar, a.sensitivity_dtheta, False, True)
        result["sites"][name]["dtheta_filtered"] = summarise(s, a.min_count, a.max_zg)
        result["sites"][name]["nearest_flux"] = summarise(nearest_samples(ds, a.min_ustar, a.min_abs_dtheta), a.min_count, a.max_zg)
        print(name, {k: v["n"] for k, v in result["sites"][name]["primary"]["site"].items()}, flush=True)
    zeta = 0.5 * (EDGES[:-1] + EDGES[1:])
    result["models"] = {s: {k: phi_model(zeta, s, k).tolist() for k in "mh"} for s in SCHEMES}
    result["models"]["zeta"] = zeta.tolist()
    with open(a.out, "w") as h:
        json.dump(result, h, indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Weather- and climate-model stability functions scored against tower flux-profile data.

Same samples as flux_gradient_sites.py (v4 settings): adjacent above-canopy level pairs,
lowest above-canopy eddy flux, heat offset-corrected, sites never pooled, z/L > 0 and < 0 apart.
Three scores per scheme: (a) gradient space phi_obs/phi_scheme, (b) integrated space,
level differences predicted from measured fluxes, (c) the models' direction, fluxes
predicted from the measured level differences (dry, bracketed root of the 1/L equation).
Louis 1979 is a bulk-Richardson scheme with no phi: it enters score (c) only, in a two-level form.
continuity() reports phi and dphi/dzeta at zeta = 0 and at every branch point.
Side decisions voted by GLM and codex (coscientist/vote_bench*).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import flux_gradient_sites as fs
from flux_gradient import KAPPA
from legoesm import constants
from legoesm.core.bulk_flux import psi_h, psi_h_coare, psi_m, psi_m_coare

jax.config.update("jax_enable_x64", True)

GAUSS_NODES = 8
BISECTION_STEPS = 60
ROOT_SCAN_POINTS = 64  # residual sampled across the domain; the root nearest neutral is bracketed
ZETA_DOMAIN = 10.0  # legoESM psi clips z/L at +-10, so phi is undefined beyond for those schemes; same domain for all
MAX_PAIR_HEIGHT_RATIO = 2.0  # wind and temperature pairs matched only if z_g within this factor
BUSINGER_KAPPA = 0.35  # Businger et al. 1971 own calibration
SECONDS_PER_DAY = 86400.0


def _phi_from_psi(psi):
    """Scalar phi = 1 - zeta dpsi/dzeta, by automatic differentiation of a model's psi."""
    dpsi = jax.grad(psi)
    return lambda z: 1.0 - z * dpsi(z)


def _split(zeta, stable, unstable):
    """Branches evaluated on safe arguments only, so neither feeds NaN to value or gradient."""
    neg = jnp.where(zeta > 0, -1.0, zeta)
    pos = jnp.where(zeta > 0, zeta, 1.0)
    return jnp.where(zeta > 0, stable(pos), unstable(neg))


def _fairall_unstable(z, conv, third):
    """Free-convective psi blended into the Kansas form by z^2/(1+z^2); WRF codes the third as 0.33, MYNN-SFC as 1/3."""
    sqrt3 = jnp.sqrt(3.0)
    y = (1.0 - conv * z) ** third
    psic = 1.5 * jnp.log((y**2 + y + 1.0) / 3.0) - sqrt3 * jnp.arctan((2.0 * y + 1.0) / sqrt3) + jnp.pi / sqrt3
    return psic


def _wrf_psi_m(z, third):
    # WRF revised MM5 (Jimenez et al. 2012), MMM-physics sf_sfclayrev.F90; MYNN-SFC psi_opt=0 same with third = 1/3
    def stable(p):
        return -6.1 * jnp.log(p + (1.0 + p**2.5) ** (1.0 / 2.5))

    def unstable(n):
        x = (1.0 - 16.0 * n) ** 0.25
        psik = 2 * jnp.log(0.5 * (1 + x)) + jnp.log(0.5 * (1 + x * x)) - 2.0 * jnp.arctan(x) + jnp.pi / 2
        return (psik + n**2 * _fairall_unstable(n, 10.0, third)) / (1 + n**2)
    return _split(z, stable, unstable)


def _wrf_psi_h(z, third):
    def stable(p):
        return -5.3 * jnp.log(p + (1.0 + p**1.1) ** (1.0 / 1.1))

    def unstable(n):
        y = (1.0 - 16.0 * n) ** 0.5
        return (2.0 * jnp.log((1 + y) / 2.0) + n**2 * _fairall_unstable(n, 34.0, third)) / (1 + n**2)
    return _split(z, stable, unstable)


def _gfs_psi(z, a0, a1, b1, b2, free):
    # MYNN-SFC psi_opt=1 "gfs" forms (module_sf_mynnsfc_driver.F90 psi*_full_gfs); stable alpha4 = 20
    def stable(p):
        a = jnp.sqrt(1.0 + 20.0 * p)
        return -a + jnp.log(a + 1.0)

    def unstable(n):
        near = jnp.where(n >= -0.5, n, -0.5)
        h = jnp.where(n < -0.5, -n, 1.0)
        return jnp.where(n >= -0.5, (a0 + a1 * near) * near / (1.0 + (b1 + b2 * near) * near), free(h))
    return _split(z, stable, unstable)


def _gfs_psi_m(z):
    return _gfs_psi(z, -3.975, 12.32, -7.755, 6.041, lambda h: jnp.log(h) + 2.0 * jnp.sqrt(1.0 / jnp.sqrt(h)) - 0.8776)


def _gfs_psi_h(z):
    return _gfs_psi(z, -7.941, 24.75, -8.705, 7.899, lambda h: jnp.log(h) + 0.5 / jnp.sqrt(h) + 1.386)


def _kansas(a, gamma, beta, exponent):
    """Businger 1971 / Hogstrom 1988 linear-stable, power-unstable phi."""
    return lambda z: jnp.where(z >= 0, a + beta * z, a * jnp.power(jnp.maximum(1.0 - gamma * z, 1.0), exponent))


def _zeng(kind):
    """Zeng et al. 1998 as coded in CTSM FrictionVelocityMod: matched free-convection and very-stable branches."""
    def phi(z):
        stable = jnp.where(z <= 1.0, 1.0 + 5.0 * z, 5.0 + z)
        if kind == "m":
            unst = jnp.where(z >= -1.574, jnp.power(jnp.maximum(1 - 16 * z, 1.0), -0.25),
                             1.14 * 0.333 * jnp.power(jnp.maximum(-z, 1e-12), 0.333))  # d/dln of CTSM 1.14 (-zeta)^0.333
        else:
            unst = jnp.where(z >= -0.465, jnp.power(jnp.maximum(1 - 16 * z, 1.0), -0.5),
                             0.8 * 0.333 * jnp.power(jnp.maximum(-z, 1e-12), -0.333))  # d/dln of CTSM -0.8 (-zeta)^-0.333
        return jnp.where(z >= 0, stable, unst)
    return phi


def _repo(scheme):
    return {"m": _phi_from_psi(lambda z: psi_m(z, scheme)), "h": _phi_from_psi(lambda z: psi_h(z, scheme))}


def _holtslag_debruin(z):
    """Holtslag & De Bruin 1988 stable phi_m = phi_h = 1 + a z + b z (1 + c - d z) exp(-d z), a=0.7 b=0.75 c=5 d=0.35
    (Gryanik et al. 2020 JAS appendix A1); stable-only scheme, unstable side Businger-Dyer gamma 16 as legoESM's BH91."""
    stable = 1.0 + 0.7 * z + 0.75 * z * (1.0 + 5.0 - 0.35 * z) * jnp.exp(-0.35 * jnp.maximum(z, 0.0))
    return jnp.where(z > 0, stable, _DYER["m"](z))


def _holtslag_debruin_h(z):
    return jnp.where(z > 0, _holtslag_debruin(z), _DYER["h"](z))


_DYER = _repo("dyer1974")


# name: (phi_m, phi_h, kappa, models using it, source of coefficients)
SCHEMES = {
    "businger1971": (_kansas(1.0, 15.0, 4.7, -0.25), _kansas(0.74, 9.0, 4.7, -0.5), BUSINGER_KAPPA,
                     "Kansas original; reference form", "Hogstrom 1988 Tables VI-VII (original Kansas, k=0.35)"),
    "hogstrom1988": (_kansas(1.0, 19.3, 6.0, -0.25), _kansas(0.95, 11.6, 7.8, -0.5), KAPPA,
                     "recalibrated Kansas", "Hogstrom 1988 eqs 12-13, Table VI-VII modified Kansas"),
    "dyer1974": (*_DYER.values(), KAPPA, "CESM ocean (Large-Pond), IFS unstable branch", "legoESM bulk_flux"),
    "holtslag_debruin1988": (_holtslag_debruin, _holtslag_debruin_h, KAPPA, "Andreas 2002 polar recommendation",
                             "Gryanik et al. 2020 appendix eqs A1-A2"),
    "beljaars_holtslag1991": (*_repo("beljaars_holtslag1991").values(), KAPPA, "ECMWF IFS stable", "legoESM bulk_flux"),
    "grachev2007_sheba": (*_repo("grachev2007_sheba").values(), KAPPA, "Arctic studies; COARE 3.6 stable", "legoESM bulk_flux"),
    "gryanik2020": (*_repo("gryanik2020").values(), KAPPA, "CliMA SurfaceFluxes", "legoESM bulk_flux"),
    "coare30": (_phi_from_psi(lambda z: psi_m_coare(z, "dyer1974", 0.0)), _phi_from_psi(lambda z: psi_h_coare(z, "dyer1974", 0.0)),
                KAPPA, "COARE 3.0 ocean (NEMO, AeroBulk)", "legoESM AeroBulk port"),
    "wrf_revmm5": (_phi_from_psi(lambda z: _wrf_psi_m(z, 0.33)), _phi_from_psi(lambda z: _wrf_psi_h(z, 0.33)), KAPPA,
                   "WRF/MPAS revised MM5 surface layer", "MMM-physics sf_sfclayrev.F90"),
    "mynn_sfc": (_phi_from_psi(lambda z: _wrf_psi_m(z, 1.0 / 3.0)), _phi_from_psi(lambda z: _wrf_psi_h(z, 1.0 / 3.0)), KAPPA,
                 "WRF/MPAS/RAP MYNN surface layer psi_opt=0", "MYNN-SFC module_sf_mynnsfc_driver.F90"),
    "gfs": (_phi_from_psi(_gfs_psi_m), _phi_from_psi(_gfs_psi_h), KAPPA, "NCEP GFS forms (MYNN-SFC psi_opt=1)",
            "MYNN-SFC module_sf_mynnsfc_driver.F90"),
    "zeng1998_clm": (_zeng("m"), _zeng("h"), KAPPA, "CESM/CTSM land (CLM)", "CTSM FrictionVelocityMod.F90"),
}


_VECTOR = {}


def phi(name, kind, zeta):
    z = np.asarray(zeta, dtype=float)
    key = (name, kind)
    if key not in _VECTOR:
        _VECTOR[key] = jax.jit(jax.vmap(SCHEMES[name][0 if kind == "m" else 1]))
    return np.asarray(_VECTOR[key](jnp.asarray(z.ravel()))).reshape(z.shape)


# Branch points inside the domain besides zeta = 0, read off each implementation; +-ZETA_DOMAIN is legoESM's psi clip.
_CLIP = [-ZETA_DOMAIN, ZETA_DOMAIN]
BRANCH_POINTS = {
    "dyer1974": {"m": _CLIP, "h": _CLIP}, "beljaars_holtslag1991": {"m": _CLIP, "h": _CLIP},
    "grachev2007_sheba": {"m": _CLIP, "h": _CLIP}, "gryanik2020": {"m": _CLIP, "h": _CLIP},
    "holtslag_debruin1988": {"m": [-ZETA_DOMAIN], "h": [-ZETA_DOMAIN]},
    "zeng1998_clm": {"m": [-1.574, 1.0], "h": [-0.465, 1.0]}, "gfs": {"m": [-0.5], "h": [-0.5]},
}
CONTINUITY_EPS = 1e-9   # one-sided evaluation offset from a branch point
CONTINUITY_STEP = 1e-6  # one-sided finite-difference step
VALUE_TOL = 1e-6        # |phi jump| counted as continuous
SLOPE_TOL = 1e-3        # |dphi/dzeta jump| / max(1, |slope|) counted as continuous


def _one_sided(f, b):
    g = jax.grad(f)
    lo, hi = b - CONTINUITY_EPS, b + CONTINUITY_EPS
    h = CONTINUITY_STEP
    return {"phi": (float(f(lo)), float(f(hi))),
            "slope_algebraic": (float(g(lo)), float(g(hi))),
            "slope_numeric": (float((f(lo) - f(lo - h)) / h), float((f(hi + h) - f(hi)) / h))}


def continuity(name, kind):
    """phi must be continuous everywhere; a slope change is allowed at zeta = 0 only, C1 required elsewhere."""
    f = SCHEMES[name][0 if kind == "m" else 1]
    zero = _one_sided(f, 0.0)
    out = {"zeta0": {**zero, "phi_jump": zero["phi"][1] - zero["phi"][0],
                     "phi_continuous": abs(zero["phi"][1] - zero["phi"][0]) <= VALUE_TOL}, "branches": []}
    for b in BRANCH_POINTS.get(name, {}).get(kind, []):
        r = _one_sided(f, b)
        dv = r["phi"][1] - r["phi"][0]
        ds = r["slope_algebraic"][1] - r["slope_algebraic"][0]
        r.update(point=b, phi_jump=dv, slope_jump=ds, phi_continuous=abs(dv) <= VALUE_TOL,
                 c1=abs(dv) <= VALUE_TOL and abs(ds) <= SLOPE_TOL * max(1.0, abs(r["slope_algebraic"][1])))
        out["branches"].append(r)
    out["pass"] = out["zeta0"]["phi_continuous"] and all(r["c1"] for r in out["branches"])
    return out


# --- Louis 1979 bulk-Richardson scheme (bulk only); b, C*m, C*h, d from ED2 R-utils/sfclyrfun.r (Louis 1979 eqs) ---
LOUIS_B, LOUIS_CM, LOUIS_CH, LOUIS_D = 5.0, 7.5, 5.0, 5.0


def louis_f(ri, zratio, kind):
    """Louis 1979 Fm or Fh for bulk Richardson number ri; zratio plays z/z0 (two-level form: z2/z1)."""
    a2 = KAPPA**2 / jnp.log(zratio) ** 2
    ee = zratio ** (1.0 / 3.0) - 1.0
    c = (LOUIS_CM if kind == "m" else LOUIS_CH) * LOUIS_B * a2 * ee * jnp.sqrt(ee * jnp.abs(ri))
    k = 2.0 if kind == "m" else 3.0
    unstable = 1.0 - k * LOUIS_B * ri / (1.0 + k * c)
    root = jnp.sqrt(1.0 + LOUIS_D * jnp.maximum(ri, 0.0))
    stable = 1.0 / (1.0 + k * LOUIS_B * ri / root) if kind == "m" else 1.0 / (1.0 + k * LOUIS_B * ri * root)
    return jnp.where(ri > 0, stable, unstable)


def louis_continuity(kind, zratio=4.0):
    return _one_sided(lambda r: louis_f(r, zratio, kind), 0.0)


def louis_invert(du, dt, zu, zt, theta):
    """Two-level Louis: level 1 stands in for the surface and z2/z1 for z/z0 (wind pair); heat difference moved
    to the wind pair's log interval for Ri. u* = a dU sqrt(Fm), theta* = kappa dtheta / ln(zt2/zt1) Fh / sqrt(Fm)."""
    lnu, lnt = np.log(zu[1] / zu[0]), np.log(zt[1] / zt[0])
    ri = constants.g * (zu[1] - zu[0]) * (dt * lnu / lnt) / (theta * du**2)
    fm = np.asarray(louis_f(jnp.asarray(ri), jnp.asarray(zu[1] / zu[0]), "m"))
    fh = np.asarray(louis_f(jnp.asarray(ri), jnp.asarray(zu[1] / zu[0]), "h"))
    us = KAPPA * du / lnu * np.sqrt(fm)
    ts = KAPPA * dt / lnt * fh / np.sqrt(fm)
    bad = ~(np.isfinite(us) & np.isfinite(ts) & (fm > 0))
    return np.where(bad, np.nan, us), np.where(bad, np.nan, ts), bad


def integral(name, kind, inv_l, z1, z2):
    """Integral of phi(z/L) d ln z from z1 to z2 by Gauss-Legendre in ln z."""
    x, w = np.polynomial.legendre.leggauss(GAUSS_NODES)
    a, b = np.log(z1)[..., None], np.log(z2)[..., None]
    lnz = 0.5 * (b - a) * x + 0.5 * (a + b)
    vals = phi(name, kind, np.exp(lnz) * inv_l[..., None])
    return 0.5 * (b - a)[..., 0] * np.sum(w * vals, axis=-1)


def site_pairs(ds, min_ustar, offset_correct, max_zg, day_parity=None, all_pairs=False):
    """Per kind, adjacent pairs at or below max_zg: heights, differences, scales, and the phi of site_samples.

    day_parity 0 or 1 keeps only even or odd UTC days of the sample end time (s since 1970).
    all_pairs: every level pair i < j (not only adjacent), for the sensor-bias test; same per-pair arithmetic, no
    cross-check against site_samples (which holds adjacent pairs only)."""
    low = fs.lowest_flux_level(ds)
    canopy = float(ds.attrs["canopy_height_m"]) if np.isfinite(float(ds.attrs["canopy_height_m"])) else 0.0
    d = fs.displacement(ds)
    f = {k: ds[k].values[:, low].astype(float) for k in ("ustar", "h", "le", "t_flux")}
    p = ds.pressure.values.astype(float)
    good = fs.clean_flux(f, min_ustar)
    length = fs.obukhov_length(f["ustar"], fs.buoyancy_flux(f["h"], f["le"], f["t_flux"], p), f["t_flux"])
    rho = p / (constants.R_d * f["t_flux"])
    theta_star = -(f["h"] / (rho * constants.c_pd)) / f["ustar"]
    wind = np.where(ds.wind.values <= fs.MAX_WIND, ds.wind.values, np.nan).astype(float)
    z_u, z_t = ds.z_u.values.astype(float), ds.z_t.values.astype(float)
    theta = ds.temp.values.astype(float) + fs.potential_temperature_offset(z_t)
    reference = fs.site_samples(ds, low, min_ustar, 0.0, False, offset_correct)
    day = np.floor(ds.time.values.astype(float) / SECONDS_PER_DAY) % 2
    in_split = np.ones(ds.sizes["time"], bool) if day_parity is None else day == day_parity
    # the neutral offset is calibrated on odd (fit) days for both splits, so scored days never calibrate themselves
    offset_days = in_split if day_parity is None else day == 1
    out = {"m": [], "h": []}
    for kind, z, x, scale in (("m", z_u, wind, f["ustar"]), ("h", z_t, theta, theta_star)):
        level_pairs = ([(i, j) for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])] if all_pairs
                       else fs.pairs(x.shape[1], False))
        for i, j in level_pairs:
            key = f"{kind}:{np.nanmedian(z[:, i]):.4g}-{np.nanmedian(z[:, j]):.4g}"
            if not all_pairs and (key not in reference or reference[key][0] > max_zg):
                continue
            z_g, _, diff = fs.pair_samples(z, x, i, j, canopy, d)
            if all_pairs and not np.nanmedian(z_g) <= max_zg:
                continue
            above = np.isfinite(z_g)
            z1, z2 = np.where(above, z[:, i] - d, np.nan), np.where(above, z[:, j] - d, np.nan)
            if kind == "h" and offset_correct:
                offset = fs.neutral_offset(np.where(offset_days, diff, np.nan), f["h"], f["ustar"])
                if not np.isfinite(offset) and np.isfinite(diff[in_split]).any():
                    warnings.warn(f"{ds.attrs.get('site', '?')} {key}: no neutral offset from the calibration days, heat pair dropped")
                diff = diff - offset
            zeta = np.where(good, z_g / length, np.nan)
            phi_obs = KAPPA * diff / (np.log(z2 / z1) * scale)
            if key in reference:
                ref_zeta, ref_phi = reference[key][1], reference[key][2]
                assert np.allclose(zeta, ref_zeta, equal_nan=True), key
                if day_parity is None:
                    assert np.allclose(phi_obs, ref_phi, equal_nan=True, rtol=1e-9), key
            keep = np.isfinite(zeta) & np.isfinite(phi_obs)
            keep &= np.abs(z2 / length) < ZETA_DOMAIN
            if kind == "h":
                keep &= np.abs(zeta) >= fs.MIN_HEAT_ABS_ZETA
            keep &= in_split
            out[kind].append({"key": key, "levels": (i, j), "z_g": float(np.nanmedian(z_g)), "z1": z1, "z2": z2, "diff": diff,
                              "scale": scale, "inv_l": 1.0 / length, "zeta": zeta, "phi": phi_obs, "keep": keep})
    flux = {"ustar": f["ustar"], "h": f["h"], "rho": rho, "theta": f["t_flux"], "good": good, "inv_l": 1.0 / length}
    return out, flux


def _stats(obs, pred):
    """Errors over all finite pairs; ratio and log error only where observation and prediction share a sign."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() == 0:
        return {"n": 0}
    o, p = obs[ok], pred[ok]
    same = (o * p) > 0
    r = {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean((o - p) ** 2))), "bias": float(np.mean(o - p)),
         "frac_sign_mismatch": float(1 - same.mean())}
    r["median_ratio"] = float(np.median(o[same] / p[same])) if same.any() else np.nan
    r["log_rmse"] = float(np.sqrt(np.mean(np.log(o[same] / p[same]) ** 2))) if same.any() else np.nan
    return r


def score_site(pairs_, flux, min_count):
    res = {}
    for kind, plist in pairs_.items():
        if not plist:
            continue
        cat = {k: np.concatenate([p[k][p["keep"]] for p in plist]) for k in ("z1", "z2", "diff", "scale", "inv_l", "zeta", "phi")}
        for regime, sel in (("stable", cat["zeta"] > 0), ("unstable", cat["zeta"] < 0)):
            if sel.sum() < min_count:
                continue
            c = {k: v[sel] for k, v in cat.items()}
            for name, spec in SCHEMES.items():
                s = spec[2] / KAPPA  # observed phi and zeta in the scheme's own kappa convention
                grad = _stats(c["phi"] * s, phi(name, kind, c["zeta"] * s))
                bins = []
                for lo, hi in zip(fs.EDGES[:-1], fs.EDGES[1:]):
                    b = (c["zeta"] >= lo) & (c["zeta"] < hi)
                    if b.sum() >= min_count:
                        bins.append([float(lo), float(hi), _stats(c["phi"][b] * s, phi(name, kind, c["zeta"][b] * s))])
                pred = c["scale"] / spec[2] * integral(name, kind, c["inv_l"] * s, c["z1"], c["z2"])
                res.setdefault(f"{kind}_{regime}", {})[name] = {"gradient": grad, "bins": bins, "integrated": _stats(c["diff"], pred)}
    return res


def invert(name, du, dt, zu, zt, theta):
    """Models' direction: u*, theta* from level differences, dry, root of the 1/L equation bracketed inside ZETA_DOMAIN.

    1/L = kappa g theta* / (u*^2 theta) with theta* = -w'theta'/u* (L > 0 stable); the root lies on the side of
    sign(dt). Returns (u*, theta*, no_root): NaN fluxes and no_root=True where no root exists inside the domain.
    """
    kappa = SCHEMES[name][2]

    def residual(x):
        us = kappa * du / integral(name, "m", x, *zu)
        ts = kappa * dt / integral(name, "h", x, *zt)
        return kappa * constants.g * ts / (us**2 * theta) - x, us, ts

    edge = np.sign(dt) * ZETA_DOMAIN / np.maximum(zu[1], zt[1])
    lo, hi = np.zeros_like(du), edge.copy()
    found = dt == 0
    prev = np.zeros_like(du)
    for k in range(1, ROOT_SCAN_POINTS + 1):  # first sign change walking out from neutral
        x = edge * k / ROOT_SCAN_POINTS
        change = ~found & (np.sign(residual(x)[0]) != np.sign(dt))
        lo, hi = np.where(change, prev, lo), np.where(change, x, hi)
        found |= change
        prev = x
    no_root = ~found
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        r_mid = residual(mid)[0]
        toward_edge = np.sign(r_mid) == np.sign(dt)  # residual keeps the sign of dt between 0 and the root
        lo, hi = np.where(toward_edge, mid, lo), np.where(toward_edge, hi, mid)
    _, us, ts = residual(0.5 * (lo + hi))
    bad = no_root | ~np.isfinite(us) | ~np.isfinite(ts)
    return np.where(bad, np.nan, us), np.where(bad, np.nan, ts), no_root


def score_inversion(pairs_, flux, min_count):
    """Each wind pair matched to the temperature pair nearest in height (ratio <= MAX_PAIR_HEIGHT_RATIO).

    Samples are those kept by both pairs in the gradient score; a non-positive wind difference is a data
    condition and is excluded (counted), not scored as a scheme failure.
    """
    res = {}
    for pm in pairs_["m"]:
        if not pairs_["h"]:
            break
        ph = min(pairs_["h"], key=lambda p: abs(np.log(p["z_g"] / pm["z_g"])))
        if max(ph["z_g"], pm["z_g"]) / min(ph["z_g"], pm["z_g"]) > MAX_PAIR_HEIGHT_RATIO:
            continue
        both = pm["keep"] & ph["keep"] & np.isfinite(pm["diff"]) & np.isfinite(ph["diff"])
        sel = both & (pm["diff"] > 0)
        zeta_obs = pm["zeta"][sel]
        args = (pm["diff"][sel], ph["diff"][sel], (pm["z1"][sel], pm["z2"][sel]), (ph["z1"][sel], ph["z2"][sel]), flux["theta"][sel])
        out = {name: invert(name, *args) for name in SCHEMES}
        out["louis1979"] = louis_invert(*args)
        solved_by_all = np.all([np.isfinite(o[0]) for o in out.values()], axis=0)  # same samples for every scheme
        for name, (us, ts, no_root) in out.items():
            h_pred = -flux["rho"][sel] * constants.c_pd * us * ts
            for regime, r in (("stable", zeta_obs > 0), ("unstable", zeta_obs < 0)):
                if r.sum() < min_count:
                    continue
                c = r & solved_by_all
                res.setdefault(f"{pm['key']}|{ph['key']}", {}).setdefault(regime, {})[name] = {
                    "n_nonpositive_wind_difference_excluded": int((both & (pm["diff"] <= 0)).sum()),
                    "no_root_frac": float(np.mean(no_root[r])),
                    "ustar": _stats(flux["ustar"][sel][r], us[r]), "h": _stats(flux["h"][sel][r], h_pred[r]),
                    "common": ({"ustar": _stats(flux["ustar"][sel][c], us[c]), "h": _stats(flux["h"][sel][c], h_pred[c])}
                               if c.sum() >= min_count else {"n": int(c.sum())})}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ustar", type=float, required=True)
    ap.add_argument("--min-count", type=int, required=True)
    ap.add_argument("--max-zg", type=float, required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    result = {"args": vars(a), "schemes": {k: {"kappa": v[2], "used_in": v[3], "source": v[4]} for k, v in SCHEMES.items()},
              "bulk_only": {"louis1979": {"kappa": KAPPA, "used_in": "ED2/BRAMS LEAF, older NWP", "source": "ED2 R-utils/sfclyrfun.r"}},
              "continuity": {k: {kind: continuity(k, kind) for kind in "mh"} for k in SCHEMES},
              "louis_continuity_in_ri": {kind: louis_continuity(kind) for kind in "mh"}, "sites": {}}
    for path in sorted(glob.glob(f"{a.tables}/*.nc")):
        name = os.path.basename(path)[:-3]
        if name in fs.EXCLUDED:
            continue
        ds = fs.open_site(path)
        site = {}
        for arm, corr in (("offset_corrected", True), ("uncorrected", False)):
            pairs_, flux = site_pairs(ds, a.min_ustar, corr, a.max_zg)
            site[arm] = score_site(pairs_, flux, a.min_count)
            if corr:
                site["inversion"] = score_inversion(pairs_, flux, a.min_count)
        result["sites"][name] = site
        print(name, {k: v.get("dyer1974", {}).get("gradient", {}).get("n") for k, v in site["offset_corrected"].items()}, flush=True)
    zeta = np.concatenate([-np.logspace(1, -3, 80), np.logspace(-3, 1, 80)])
    result["curves"] = {"zeta": zeta.tolist(), **{s: {k: phi(s, k, zeta).tolist() for k in "mh"} for s in SCHEMES}}
    with open(a.out, "w") as h:
        json.dump(result, h, indent=1, default=float)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

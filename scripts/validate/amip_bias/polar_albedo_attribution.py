#!/usr/bin/env python3
"""Which surface owns the polar clear-sky shortwave deficit: a replay.

The polar TOA bias measured on the arms is 75 % CLEAR-SKY (rsutcs -18.7 of
rsut -24.8 W/m2 at 60-90), which points at the surface.  Three surfaces are
candidates -- land snow (albedo 0.521 wherever snow lies), prescribed sea ice
(0.65) and open ocean (0.06) -- and area alone cannot rank them, because the
incident shortwave and the atmospheric transmission differ by cell.

So: replay the run's OWN clear-sky radiation on a saved checkpoint and perturb
ONE surface albedo at a time, holding everything else byte-identical.  Reported
is the polar-cap change in top-of-atmosphere clear-sky upward shortwave, which
is exactly the quantity the CERES comparison uses.

CONTROL FIRST (this probe is untrusted code until it passes): the unperturbed
replay is compared against the run's OWN published rsutcs for the same cap.
The replay is a fixed-day diurnal quadrature on a single checkpoint while the
published value is a monthly mean over a partly different window, so they are
NOT expected to agree to a fraction of a W/m2; the control asserts only that
the replay is in the right REGIME (within `--control-tol`, default 15 W/m2).
A replay outside that is reported and the run ABORTS rather than emitting
attribution numbers from a broken instrument.

The perturbations are ABSOLUTE albedo values, not deltas, so the reported
number answers "what would the cap reflect if this surface had the observed
albedo", which is the question the attribution asks.

Usage:
  polar_albedo_attribution.py <run> [--day 86] [--n-times 8]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

_VAL = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_VAL))
_spec = importlib.util.spec_from_file_location("cloud_layers", _VAL / "cloud_layers.py")
cl = importlib.util.module_from_spec(_spec)
sys.modules["cloud_layers"] = cl
_spec.loader.exec_module(cl)
rb = cl.rb

# Observed surface albedos for the perturbations, with provenance.
OBS_SNOW_ARCTIC = 0.70      # snow-covered tundra, broadband
OBS_SNOW_ANTARCTIC = 0.82   # Antarctic plateau, broadband annual
OBS_ICE = 0.80              # Arctic sea ice with snow cover, March


def land_fraction(exp, lat, lon):
    import xarray as xr
    lmf = exp.get("land_mask_path") or exp.get("land_mask_file")
    if not lmf or not pathlib.Path(str(lmf)).exists():
        raise SystemExit(f"FATAL: land mask {lmf!r} unreadable")
    d = xr.open_dataset(lmf, decode_times=False)
    v = "sftlf" if "sftlf" in d else list(d.data_vars)[0]
    m = np.asarray(d[v].values).squeeze()
    if m.max() > 1.5:
        m = m / 100.0
    mlat = np.asarray(d["lat"].values)
    mlon = np.asarray(d["lon"].values) % 360.0
    ii = np.abs(mlat[None, :] - lat[:, None]).argmin(axis=1)
    jj = np.abs(((mlon[None, :] - lon[:, None] + 180) % 360) - 180).argmin(axis=1)
    return m[ii, jj]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--day", type=int, default=86)
    ap.add_argument("--n-times", type=int, default=8,
                    help="local-time quadrature points over the diurnal cycle")
    ap.add_argument("--control-tol", type=float, default=15.0,
                    help="max |replay - published| cap-mean rsutcs [W/m2]")
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.surface_albedo import (
        LandAlbedoConfig, snow_albedo, snow_cover_fraction)
    from legoesm import constants

    rundir = f"{rb.ROOT}/{args.run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat_deg, lon_deg, area = cl.mesh_coords(exp)
    z = np.load(f"{rundir}/checkpoint_day_{args.day:04d}.npz", allow_pickle=True)
    order = cl.cell_order(z, lat_deg.size)

    def col(name):
        return np.asarray(z[name])[order]

    vg = np.asarray(z["meta_vgrid"], dtype=np.float64)
    ps = col("p_s")
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = col("T")
    q_v = col("trc_q_v")
    T_sfc = T[:, -1]

    # The run's RESOLVED surface albedos, exactly as the driver blends them:
    # alpha = (1-f_land) * [sic*alpha_ice + (1-sic)*alpha_ocean] + f_land*alpha_land
    cfg = LandAlbedoConfig()
    if exp.get("land_calibrated_physics"):
        from legoesm.land.clm_surface_map import biophysics_lmip_albedo_scalars
        cfg = cfg._replace(**biophysics_lmip_albedo_scalars())
    swe = col("land_ml_snow_depth")
    age = col("land_ml_snow_age")
    f_snow = np.asarray(snow_cover_fraction(swe, cfg))
    a_snow = np.asarray(snow_albedo(age, cfg))
    a_land = cfg.alpha_veg_highlat * (1 - f_snow) + a_snow * f_snow
    f_land = land_fraction(exp, lat_deg, lon_deg)
    a_ice = float(exp["albedo_ice"])
    a_ocn = float(exp["albedo_ocean"])
    # The prescribed sea-ice CONCENTRATION is not carried in the checkpoint, so
    # the sea tile is treated as fully ice-covered POLEWARD of 60 and fully
    # open equatorward -- stated, and an upper bound on the ice term.
    polar = np.abs(lat_deg) >= 60.0
    sic = np.where(polar, 1.0, 0.0)

    def blended(a_land_v, a_ice_v, a_ocn_v):
        sea = sic * a_ice_v + (1.0 - sic) * a_ocn_v
        return (1.0 - f_land) * sea + f_land * a_land_v

    # Diurnal quadrature at the checkpoint's own day of year.
    day = float(z["day"]) if "day" in z.files else float(args.day)
    doy = 1.0 + day
    decl = -23.44 * np.cos(2 * np.pi * (doy + 10.0) / 365.0) * np.pi / 180.0
    latr = np.deg2rad(lat_deg)
    hours = (np.arange(args.n_times) * 24.0 / args.n_times
             + 24.0 / (2 * args.n_times))
    cosz_t = [np.clip(np.sin(latr) * np.sin(decl)
                      + np.cos(latr) * np.cos(decl) * np.cos((h - 12.0) * np.pi / 12.0),
                      0.0, 1.0) for h in hours]

    # CLEAR SKY: no cloud paths passed at all (include_clouds is irrelevant
    # when the kwargs are absent, but it is set False so a future default
    # cannot inject cloud).
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(
        gpoint_batch_size=16, gpoint_checkpoint=False, include_clouds=False))

    def cap_rsutcs(alb, mask):
        w = area[mask]
        tot = np.zeros(mask.sum())
        for mu in cosz_t:
            mu_j = jnp.asarray(np.maximum(mu, 1e-4))
            out = solver.solve_columns(
                T=jnp.asarray(T), p_full=jnp.asarray(p_full),
                p_half=jnp.asarray(p_half), sfc_temperature=jnp.asarray(T_sfc),
                q_v=jnp.asarray(q_v), cos_zenith=mu_j,
                sfc_albedo=jnp.asarray(alb), sfc_emissivity=0.97)
            up = np.asarray(out.sw_flux_up[:, 0])
            tot = tot + np.where(mu[mask] > 0.0, up[mask], 0.0)
        return float((tot / len(cosz_t) * w).sum() / w.sum())

    caps = {"Arctic 60-90N": lat_deg >= 60.0, "Antarctic 60-90S": lat_deg <= -60.0}

    # --- CONTROL: does the replay reproduce the run's own clear-sky number? ---
    print(f"=== {args.run} day {args.day}: clear-sky replay control ===")
    pub = rb._load_model(args.run, "rsutcs")
    ok = True
    base = {}
    alb0 = blended(a_land, a_ice, a_ocn)
    for name, m in caps.items():
        base[name] = cap_rsutcs(alb0, m)
        if pub is None:
            print(f"  {name}: replay {base[name]:7.2f}   (run publishes no rsutcs)")
            continue
        plat = np.asarray(pub.lat)
        pv = np.asarray(pub["rsutcs"]).mean(axis=0)
        sel = (plat >= 60.0) if "Arctic" in name else (plat <= -60.0)
        wl = np.cos(np.deg2rad(plat[sel]))
        published = float((pv[sel].mean(axis=1) * wl).sum() / wl.sum())
        d = base[name] - published
        flag = "OK" if abs(d) <= args.control_tol else "FAIL"
        ok &= abs(d) <= args.control_tol
        print(f"  {name}: replay {base[name]:7.2f}  published {published:7.2f}  "
              f"diff {d:+6.2f}  [{flag}]")
    if not ok:
        raise SystemExit(
            f"FATAL: the replay does not reproduce the run's clear-sky regime "
            f"(tolerance {args.control_tol} W/m2). Attribution numbers from a "
            f"replay that fails its own control would be meaningless.")

    # --- ATTRIBUTION: one surface at a time, to its observed albedo ---
    print(f"\n=== change in cap-mean clear-sky reflected SW [W/m2] when ONE "
          f"surface takes its observed albedo ===")
    print(f"{'perturbation':38s} {'Arctic':>9s} {'Antarctic':>11s}")
    rows = []
    for label, la, ia in (
            (f"land snow -> obs ({OBS_SNOW_ARCTIC}/{OBS_SNOW_ANTARCTIC})", None, None),
            (f"sea ice {a_ice} -> {OBS_ICE}", a_land, OBS_ICE),
            ("both", None, OBS_ICE),
    ):
        vals = []
        for name, m in caps.items():
            obs_snow = OBS_SNOW_ARCTIC if "Arctic" in name else OBS_SNOW_ANTARCTIC
            la_v = (np.where(f_snow > 0.5, obs_snow, a_land)
                    if la is None else la)
            ia_v = a_ice if ia is None else ia
            vals.append(cap_rsutcs(blended(la_v, ia_v, a_ocn), m) - base[name])
        rows.append((label, vals))
        print(f"{label:38s} {vals[0]:+9.2f} {vals[1]:+11.2f}")
    (_, land_v), (_, ice_v), (_, both_v) = rows
    print(f"\n{'additivity residual (both - land - ice)':38s} "
          f"{both_v[0] - land_v[0] - ice_v[0]:+9.2f} "
          f"{both_v[1] - land_v[1] - ice_v[1]:+11.2f}")
    print(f"{'land share of the two-surface total':38s} "
          f"{land_v[0] / max(land_v[0] + ice_v[0], 1e-9):9.2f} "
          f"{land_v[1] / max(land_v[1] + ice_v[1], 1e-9):11.2f}")
    print("\nThe CERES comparison needs +18.7 W/m2 more clear-sky reflection at "
          "60-90 (both caps, monthly mean).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Polar surface albedo from a checkpoint, through the model's OWN albedo code.

The polar TOA shortwave deficit measured on the arms is 75 % CLEAR-SKY
(rsutcs -18.7 of rsut -24.8 W/m2 at 60-90), which points at the surface, not
the cloud.  This rebuilds the land albedo the radiation was given -- snow
cover from the checkpoint's SWE, snow albedo from its age, through
``legoesm.surface_albedo`` -- and prints the snow state next to it, so the
albedo can be attributed to (a) too little snow, (b) too-fast age decay, or
(c) a too-dark albedo ceiling.  Sea-ice and open-ocean albedo are printed for
the same cells from the prescribed AMIP ice fraction.

Reference values quoted for comparison (literature, not model output):
fresh dry snow broadband 0.80-0.85; Antarctic plateau annual 0.80-0.83;
Arctic sea ice with snow 0.75-0.85, bare melting ice 0.5-0.65; tundra in
snow 0.6-0.75.

Usage: _probe_polar_albedo.py <run> [--day 86]
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

BANDS = {"Arctic 60-90N": (60.0, 90.0), "Antarctic 60-90S": (-90.0, -60.0),
         "NH 30-60N": (30.0, 60.0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--day", type=int, default=86)
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.surface_albedo import (
        LandAlbedoConfig, snow_albedo, snow_cover_fraction, land_albedo,
    )

    rundir = f"{rb.ROOT}/{args.run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat, lon, area = cl.mesh_coords(exp)
    z = np.load(f"{rundir}/checkpoint_day_{args.day:04d}.npz", allow_pickle=True)
    order = cl.cell_order(z, lat.size)
    swe = np.asarray(z["land_ml_snow_depth"])[order]
    age = np.asarray(z["land_ml_snow_age"])[order]

    # REAL land fraction, from the run's own land-mask file -- NOT "the land
    # model has soil state here" (the land tile carries state on cells that are
    # mostly ocean, so theta_soil > 0 called the whole polar cap land).
    import xarray as xr
    lmf = exp.get("land_mask_path") or exp.get("land_mask_file")
    if not lmf or not pathlib.Path(str(lmf)).exists():
        raise SystemExit(f"FATAL: land_mask_path {lmf!r} unreadable; the polar "
                         "land/ocean split cannot be rebuilt")
    d = xr.open_dataset(lmf, decode_times=False)
    v = "sftlf" if "sftlf" in d else list(d.data_vars)[0]
    m = np.asarray(d[v].values).squeeze()
    if m.max() > 1.5:
        m = m / 100.0
    mlat = np.asarray(d["lat"].values); mlon = np.asarray(d["lon"].values) % 360.0
    ii = np.abs(mlat[None, :] - lat[:, None]).argmin(axis=1)
    jj = np.abs(((mlon[None, :] - lon[:, None] + 180) % 360) - 180).argmin(axis=1)
    f_land = m[ii, jj]
    print(f"  land_mask_file = {lmf}  ({v}); global land area fraction "
          f"{float((f_land * area).sum() / area.sum()):.3f} (obs 0.29)")

    # RESOLVED land-albedo config, not the library defaults: a calibrated run
    # (``land_calibrated_physics``) replaces the snow scalars, and reading the
    # defaults instead reports parameters the run never used (codex review,
    # 2026-09-09 - the run's tau is 3.67 d, not the library's 5 d, and its
    # snow_depth_crit is 15.4, not 50).
    cfg = LandAlbedoConfig()
    if exp.get("land_calibrated_physics"):
        from legoesm.land.clm_surface_map import biophysics_lmip_albedo_scalars
        cfg = cfg._replace(**biophysics_lmip_albedo_scalars())
        print("  land_calibrated_physics=true -> LMIP-calibrated snow scalars")
    else:
        print("  land_calibrated_physics=false -> LandAlbedoConfig defaults")
    # Per-cell snow-free base albedo, as the run built it (surfdata / CLM map),
    # NOT the latitude-band vegetation value.
    base_map = None
    _sd = exp.get("surfdata_path") or exp.get("clm_surfdata_path")
    print(f"  snow-free base: {'surfdata map ' + str(_sd) if _sd else 'latitude band %.2f' % cfg.alpha_veg_highlat}")
    # NB the run's per-column ``snow_cover_scale`` map (built from the surfdata)
    # is NOT rebuilt here; f_snow below is therefore the unscaled tanh curve and
    # is an UPPER bound on the run's snow cover.
    print(f"=== {args.run} day {args.day}: land albedo config (resolved defaults) ===")
    for k in ("alpha_snow_max", "alpha_snow_min", "tau_snow_decay",
              "snow_depth_crit", "snow_zenith_factor", "alpha_veg_highlat"):
        print(f"  {k} = {getattr(cfg, k)}")
    if any(exp.get(k) is not None for k in ("land_albedo_alpha_snow_max",)):
        print("  NB run overrides some albedo fields; see experiment_config.json")

    f_snow = np.asarray(snow_cover_fraction(swe, cfg))
    a_snow = np.asarray(snow_albedo(age, cfg))          # diffuse, no zenith
    theta0 = np.asarray(z["land_ml_theta_soil"])[order][:, 0]

    # DISTRIBUTIONS, not means of a nonlinear function: the area-weighted mean
    # of alpha(age) is NOT alpha(mean age) (the decay is convex, so the mean
    # albedo of a band containing snow-free cells reads far brighter than the
    # snow itself).  Report the SNOW-COVERED subset and its percentiles.
    print(f"\n{'band':18s} {'cells':>6s} {'snowy':>6s}  "
          f"{'SWE p50':>8s} {'age p50 d':>9s} {'f_snow p50':>10s} "
          f"{'a_snow p10/p50/p90':>20s} {'a_land p50':>10s}")
    for name, (lo, hi) in BANDS.items():
        band = (lat >= lo) & (lat <= hi) & (f_land > 0.5)
        snowy = band & (swe > 10.0)          # cells the snow albedo governs
        if not snowy.any():
            print(f"{name:18s}   (no snow-covered land columns)")
            continue
        base = cfg.alpha_veg_highlat
        a_land = base * (1 - f_snow) + a_snow * f_snow
        q = lambda x, p: float(np.percentile(x[snowy], p))
        print(f"{name:18s} {band.sum():6d} {snowy.sum():6d}  "
              f"{q(swe, 50):8.1f} {q(age, 50) / 86400.0:9.1f} {q(f_snow, 50):10.3f} "
              f"{q(a_snow, 10):6.3f}/{q(a_snow, 50):.3f}/{q(a_snow, 90):.3f} "
              f"{q(a_land, 50):10.3f}")

    print("\n=== snow-albedo sensitivity on SNOW-COVERED land cells (median a_land) ===")
    print(f"{'variant':34s} {'Arctic':>8s} {'Antarctic':>10s}")
    for label, c in (
            ("resolved default", cfg),
            ("alpha_snow_max 0.85", cfg._replace(alpha_snow_max=0.85)),
            ("alpha_snow_min 0.65", cfg._replace(alpha_snow_min=0.65)),
            ("tau_snow_decay 15 d", cfg._replace(tau_snow_decay=15 * 86400.0)),
            ("tau_snow_decay 250 d (cold-snow eff.)",
             cfg._replace(tau_snow_decay=250 * 86400.0)),
            ("snow_depth_crit 10", cfg._replace(snow_depth_crit=10.0)),
    ):
        row = []
        for lo, hi in ((60.0, 90.0), (-90.0, -60.0)):
            snowy = (lat >= lo) & (lat <= hi) & (f_land > 0.5) & (swe > 10.0)
            if not snowy.any():
                row.append(float("nan"))
                continue
            fs = np.asarray(snow_cover_fraction(swe[snowy], c))
            a = np.asarray(snow_albedo(age[snowy], c))
            row.append(float(np.median(cfg.alpha_veg_highlat * (1 - fs) + a * fs)))
        print(f"{label:34s} {row[0]:8.3f} {row[1]:10.3f}")

    # The BATS zenith brightening the RUN applies (multilayer_land.py passes
    # cos_zenith) but the diffuse numbers above omit.  Bracket it over the
    # polar March range of mu rather than inventing the run's instantaneous mu.
    print("\n=== BATS zenith brightening the run applies (a_snow median) ===")
    print(f"{'mu = cos(zenith)':20s} {'Arctic':>8s} {'Antarctic':>10s}")
    for mu in (1.0, 0.5, 0.3, 0.2, 0.1):
        row = []
        for lo, hi in ((60.0, 90.0), (-90.0, -60.0)):
            snowy = (lat >= lo) & (lat <= hi) & (f_land > 0.5) & (swe > 10.0)
            a = np.asarray(snow_albedo(age[snowy], cfg,
                                       cos_zenith=np.full(snowy.sum(), mu)))
            row.append(float(np.median(a)))
        print(f"{mu:<20.2f} {row[0]:8.3f} {row[1]:10.3f}")


    # LAND vs OCEAN/SEA-ICE: which surface carries the polar clear-sky deficit.
    # The AMIP lane blends alpha = (1-f_land)*[sic*alpha_ice + (1-sic)*alpha_ocean]
    # + f_land*alpha_land (model_driver: blended_surface_albedo), with
    # alpha_ice/alpha_ocean the RESOLVED config scalars.
    a_ice = float(exp["albedo_ice"]); a_ocn = float(exp["albedo_ocean"])
    print(f"\n=== polar surface budget: land vs ocean/ice "
          f"(alpha_ice {a_ice}, alpha_ocean {a_ocn} from the run's config) ===")
    print(f"{'band':18s} {'land area':>10s} {'a_land':>8s} {'a_sea(ice)':>11s} "
          f"{'obs a_land':>11s} {'obs a_ice':>10s} {'land share of gap':>18s}")
    OBS = {"Arctic 60-90N": (0.70, 0.80), "Antarctic 60-90S": (0.82, 0.80),
           "NH 30-60N": (0.55, 0.80)}
    for name, (lo, hi) in BANDS.items():
        m = (lat >= lo) & (lat <= hi)
        w = area[m]
        fl = f_land[m]
        land_share = float((fl * w).sum() / w.sum())
        snowy = m & (f_land > 0.5) & (swe > 10.0)
        if not snowy.any():
            continue
        fs = np.asarray(snow_cover_fraction(swe[snowy], cfg))
        al = float(np.median(cfg.alpha_veg_highlat * (1 - fs)
                             + np.asarray(snow_albedo(age[snowy], cfg)) * fs))
        ol, oi = OBS[name]
        # sea albedo assumed fully ice-covered at the poles (upper bound on the
        # sea term); the gap each surface contributes, area-weighted
        gap_land = land_share * (ol - al)
        gap_sea = (1 - land_share) * (oi - a_ice)
        tot = gap_land + gap_sea
        print(f"{name:18s} {land_share:10.2f} {al:8.3f} {a_ice:11.3f} "
              f"{ol:11.2f} {oi:10.2f} "
              f"{(gap_land / tot if tot else float('nan')):18.2f}")
    print("  NB the sea column assumes 100% ice cover at 60-90 (upper bound on"
          " the sea term); the prescribed sea-ice CONCENTRATION is not read here.")


if __name__ == "__main__":
    main()

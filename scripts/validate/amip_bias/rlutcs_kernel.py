#!/usr/bin/env python3
"""Is the free-tropospheric COLD bias big enough to explain the clear-sky OLR
deficit, or is something else missing?

The 12-month AMIP is -19.6 W/m2 in `rlutcs` GLOBALLY (-8.4 even at the poles),
and `profile_rh_split.py` measures a cold bias peaking at -5.2 K at 700 hPa
together with a mid/upper-tropospheric moist bias. Both push clear-sky OLR
down. Before either is treated as the cause, the arithmetic has to close: a
temperature bias of that size and shape must actually be worth ~20 W/m2.

This measures it instead of asserting it. The model's own RRTMGP is run
clear-sky on real production columns three times:

    base            model state as it is
    +dT             temperature warmed to ERA5 by the measured bias profile
    +dq             humidity scaled to ERA5 by the measured ratio profile

and the two deltas are reported against the -19.6 W/m2 to be explained.

READ THIS BEFORE QUOTING THE NUMBER. This is a SENSITIVITY of the radiation
operator to a prescribed profile perturbation, NOT a prediction of what the
coupled model would do: correcting the temperature in a live run changes the
circulation, the humidity and the clouds too. The claim it can support is
narrow and it is the one in question — "a bias of this size and shape is / is
not worth ~20 W/m2 of clear-sky OLR".

Two deliberate choices, stated because they bound the answer:

* The perturbation profile is a GLOBAL MEAN, so any covariance between the
  bias and the local state is discarded. Running the profile and the columns
  from DIFFERENT runs was flagged in review as a category error; measured, it
  is not one here — cldF_fsd day 1825 gives +3.82 / +8.07 / +12.80 and
  ref1979's own day 360 gives +3.82 / +8.16 / +12.89, so the split is robust
  to the state it is evaluated on. Quote the same-run numbers anyway.
* The correction is held at the IDENTITY above the top measured level
  (100 hPa). The model is +11 K warm there, and applying ERA5's cooling would
  push OLR DOWN, so the temperature figure below is an UPPER bound on what a
  full temperature correction buys.
* Clear-sky means `include_clouds=False`, matching how `rsutcs`/`rlutcs` are
  produced. The cloud fields are still loaded and simply not passed.

Usage:
    rlutcs_kernel.py [--run <run dir name>] [--day N] [--ncol N]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_HARNESS = "/work/bd1083/b309178/diffESM/legoesm_pg/_albedo_hifi/harness_hifi.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bias_profiles(ref_run):
    """Global-mean (dT[K], q_model/q_ERA5) on the CMOR plev axis, below-ground
    cells masked — the same reduction ``profile_rh_split`` prints."""
    prs = _load(str(pathlib.Path(__file__).with_name("profile_rh_split.py")),
                "prs")
    rb = prs.rb
    mt, mq = rb._load_model(ref_run, "ta"), rb._load_model(ref_run, "hus")
    if mt is None or mq is None:
        raise SystemExit(f"{ref_run}: no ta/hus CMOR output to build the "
                         "perturbation profile from")
    months = rb._month_labels(mt)
    mlat = np.asarray(mt.lat, dtype=np.float64)
    mlon = np.asarray(mt.lon, dtype=np.float64) % 360.0
    plev_all = np.asarray(mt.plev, dtype=np.float64)
    keep = plev_all >= prs.P_FLOOR
    plev = plev_all[keep]
    Tm = np.asarray(mt["ta"]).mean(axis=0)[keep]
    qm = np.asarray(mq["hus"]).mean(axis=0)[keep]
    Te = prs._era5_on_model("ta", months, plev, mlat, mlon)
    qe = prs._to_mixing_ratio(prs._era5_on_model("hus", months, plev, mlat, mlon))

    ps_m = np.asarray(rb._load_model(ref_run, "ps")["ps"]).min(axis=0)
    ps_e = np.min([rb.bin_to_model(*prs._era5_surface_pressure([m]), mlat, mlon,
                                   label="ERA5 ps")
                   for m in sorted(set(months))], axis=0)
    below = plev[:, None, None] > np.minimum(ps_m, ps_e)[None, :, :]

    box = rb.REGIONS["GLOBAL"]
    dT = np.array([rb.region_mean(Te[k] - Tm[k], mlat, mlon, box, ~below[k])
                   for k in range(plev.size)])
    ratio = np.array([rb.region_mean(qe[k], mlat, mlon, box, ~below[k])
                      / rb.region_mean(qm[k], mlat, mlon, box, ~below[k])
                      for k in range(plev.size)])
    return plev, dT, ratio


def _on_columns(plev, prof, p_full, fill):
    """Interpolate a profile given on ``plev`` onto each column's p_full.

    Linear in log-p. OUTSIDE the plev range the profile is held at ``fill``
    (not extrapolated): above the top plev row the correction is unmeasured,
    and extrapolating a +11 K upper-level bias into the model's lid would
    manufacture the answer.
    """
    lp, lq = np.log(plev[::-1]), prof[::-1]
    out = np.interp(np.log(np.asarray(p_full)), lp, lq, left=fill, right=fill)
    return out


_REPO = pathlib.Path(__file__).resolve().parents[3]


def _strip_flags(argv, drop):
    """``argv`` without the flags in ``drop`` ({flag: n_values})."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in drop:
            i += 1 + drop[argv[i]]
            continue
        out.append(argv[i])
        i += 1
    return out


def build_run_driver(run_dir, first_ckpt, out_dir, manifest="run_manifest.json"):
    """The run's OWN driver: ``run_amip.main`` on the manifest's argv, restarted
    from ``first_ckpt``, with the time loop replaced by a capture.

    Only the output folder moves (to ``out_dir``) and ``--distributed`` is
    dropped (single process); every physics/forcing setting is the run's.
    """
    import json
    import os
    man = json.load(open(run_dir / manifest))
    argv = _strip_flags(list(man["run"]["argv"][1:]),
                        {"--distributed": 0, "--output": 1,
                         "--restart-from": 1})
    argv += ["--output", str(out_dir), "--restart-from", str(first_ckpt)]
    from legoesm.driver.model_driver import ModelDriver
    got = {}

    def _capture(self, start_step=0, start_day=None, compiled=True):
        got["driver"] = self
        return "COMPLETED"

    ModelDriver.run = _capture
    ra = _load(str(_REPO / "scripts/run/run_amip.py"), "run_amip_probe")
    cwd = os.getcwd()
    os.chdir(_REPO)
    try:
        ra.main(argv)
    except SystemExit as e:
        if "driver" not in got:
            raise
        print(f"(run_amip.main exited {e.code} after setup; driver captured)")
    finally:
        os.chdir(cwd)
    return got["driver"], man


def driver_columns(drv, man, ckpt, day):
    """Clear-sky radiation inputs at checkpoint ``ckpt``, assembled the way the
    MPAS time loop assembles them (surface anchor, external forcing, tracers)."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.forcing.surface_utils import (
        blend_surface_temperature, land_lapse_adjusted_surface_temperature)
    from legoesm.atmosphere.physics.radiation import integration as RI

    drv.load_checkpoint(ckpt)
    rc = man["config"]["resolved_config"]
    st = drv.state
    T = jnp.asarray(st.T.data)
    ncol, nlev = T.shape
    ps = jnp.asarray(st.p_s.data).reshape(-1)
    p_full = drv.sigma.pressure_at_full(ps).reshape(ncol, nlev)
    p_half = drv.sigma.pressure_at_half(ps).reshape(ncol, nlev + 1)
    q_v = RI._extract_tracer_columns(
        st, ncol, nlev,
        rho_col=RI._rho_for_number_conversion(st, ncol, nlev, T, p_full))[0]

    fday = float(int(day))
    ck = np.load(ckpt, allow_pickle=True)
    sst, sic = drv.get_sst_sic(fday)
    sst = jnp.asarray(sst).reshape(-1)
    sic = jnp.asarray(sic).reshape(-1)
    ice = (jnp.asarray(ck["ice_T_skin"]) if rc["ice_skin_prognostic"]
           else rc["T_ice"])
    f_land = jnp.asarray(drv._f_land).reshape(-1)
    ts = blend_surface_temperature(sst, sic, ice)
    lapse = float(rc["land_lapse_K_per_km"]) * 1.0e-3
    if lapse > 0.0:
        ts = land_lapse_adjusted_surface_temperature(
            ts, f_land, jnp.asarray(st.phis.data).reshape(-1) / constants.g,
            lapse)
    if rc["use_multilayer_land"]:
        ts = (1.0 - f_land) * ts + f_land * jnp.asarray(ck["land_held_T_skin"])

    p_s_o, lat_o = drv._owned_p_s_and_lat()
    o3, aer, ghg = drv._precompute_external_forcing(fday, p_s_o,
                                                    jnp.asarray(lat_o))
    aer_lw = drv._aerosol_lw_od if drv._aerosol_lw_active else None
    o3_top = getattr(drv, "_o3_top_vmr", None)
    lat = jnp.asarray(lat_o).reshape(-1)
    lon = jnp.asarray(drv._owned_lon()).reshape(-1)
    area = np.asarray(drv._coeff_grid().areaCell, dtype=np.float64).reshape(-1)
    assert area.size == ncol, (area.size, ncol)
    return dict(T=T, q_v=q_v, p_full=p_full, p_half=p_half, T_sfc=ts,
                o3=o3, o3_top=o3_top, aer=aer, aer_lw=aer_lw,
                ghg=None if ghg is None else {k: jnp.asarray(v)
                                              for k, v in ghg.items()},
                lat=lat, lon=lon, w=area / area.sum(), day=fday)


def driver_rlutcs(drv, man, s):
    """Global-mean clear-sky OLR through the model's own backend call, with the
    run's recorded RadiationConfig switched to its clear-sky variant exactly as
    the MPAS factory builds it (cloud_scheme='none', cloud_config=None)."""
    from legoesm.driver.restart import physics_config_from_record
    from legoesm.atmosphere.physics.radiation import integration as RI
    rad = physics_config_from_record(
        man["config"]["resolved_physics"]["radiation"])
    clr = rad._replace(cloud_scheme="none", cloud_config=None)
    doy, sod = drv._calendar_for_radiation(s["day"])
    insol, cos_sza, f_day, eccf = RI._compute_insolation(
        s["lat"], clr, lon=s["lon"], day_of_year=doy, seconds_of_day=sod)

    def olr(T, q_v):
        out = RI._call_radiation_backend(
            radiation_config=clr, eccf=eccf, T=T, p_full=s["p_full"],
            p_half=s["p_half"], sfc_temperature=s["T_sfc"], lat=s["lat"],
            q_v=q_v, insolation=insol, cos_sza=cos_sza, f_day=f_day,
            lon=s["lon"], o3_vmr_override=s["o3"], o3_top_vmr=s["o3_top"],
            aerosol_od=s["aer"], aerosol_lw_od=s["aer_lw"],
            ghg_vmr_override=s["ghg"])
        x = np.asarray(out.lw_flux_up[:, 0], dtype=np.float64)
        assert np.isfinite(x).all(), "clear-sky OLR non-finite -- FATAL"
        return x
    return olr


def cmor_global(run, var):
    """Window-mean global mean of a CMOR field with the regional tool's own
    cos-lat weighting (the same reduction its GLOBAL row uses)."""
    prs = _load(str(pathlib.Path(__file__).with_name("profile_rh_split.py")),
                "prs")
    rb = prs.rb
    md = rb._load_model(run, var)
    months = sorted(set(rb._month_labels(md)))
    f = np.asarray(md[var]).mean(axis=0)
    return months, rb.region_mean(f, np.asarray(md.lat), np.asarray(md.lon),
                                  rb.REGIONS["GLOBAL"])


def ts_bias_grid(ref_run):
    """(mlat, mlon, ERA5 skin T minus model ts [K], land mask) on the run's
    CMOR grid, same months as the run -- the regional tool's own ts reference."""
    prs = _load(str(pathlib.Path(__file__).with_name("profile_rh_split.py")),
                "prs")
    rb = prs.rb
    md = rb._load_model(ref_run, "ts")
    if md is None:
        raise SystemExit(f"{ref_run}: no ts CMOR output")
    months = rb._month_labels(md)
    mlat = np.asarray(md.lat, dtype=np.float64)
    mlon = np.asarray(md.lon, dtype=np.float64) % 360.0
    tm = np.asarray(md["ts"]).mean(axis=0)
    te = rb._ref_clim("ts", months, mlat, mlon)
    return mlat, mlon, te - tm, rb._land_mask(ref_run, mlat, md.lon.values), rb


def nearest_cell(lat_rad, lon_rad, mlat, mlon):
    """Index (i_lat, i_lon) of the CMOR cell containing each model column."""
    la = np.rad2deg(np.asarray(lat_rad, dtype=np.float64))
    lo = np.rad2deg(np.asarray(lon_rad, dtype=np.float64)) % 360.0
    assert np.abs(la).max() <= 90.0 + 1e-6, "column latitude not in radians"
    il = np.abs(la[:, None] - mlat[None, :]).argmin(axis=1)
    dlo = np.abs(((lo[:, None] - mlon[None, :]) + 180.0) % 360.0 - 180.0)
    return il, dlo.argmin(axis=1)


def bin_columns(x, w, il, io, shape):
    """Area-weighted mean of column values ``x`` in each CMOR cell, and the
    mask of cells that hold at least one column (small polar cells may not;
    they are left OUT of region means, never filled)."""
    num = np.zeros(shape); den = np.zeros(shape)
    np.add.at(num, (il, io), x * w); np.add.at(den, (il, io), w)
    has = den > 0
    return np.where(has, num / np.where(has, den, 1.0), 0.0), has


def main_driver(args):
    import os
    root = pathlib.Path(os.environ["LEGOESM_AMIP_RUNS"])
    run_dir = root / args.driver_run
    days = [int(d) for d in args.days.split(",")]
    ck = [run_dir / f"checkpoint_day_{d:04d}.npz" for d in days]
    for c in ck:
        assert c.exists(), f"missing {c}"
    drv, man = build_run_driver(run_dir, ck[0], pathlib.Path(args.out_dir),
                                args.manifest)
    plev, dT, ratio = bias_profiles(args.ref_run)
    print(f"perturbation profile from {args.ref_run}:")
    for p, t, r in zip(plev, dT, ratio):
        print(f"   {p / 100:7.0f} hPa   dT(ERA5-model) {t:+6.2f} K   "
              f"q_ERA5/q_model {r:5.2f}")
    import jax.numpy as jnp
    mlat, mlon, dTs_grid, land, rb = ts_bias_grid(args.ref_run)
    labs = ("base", "+T", "+q", "+Ts", "T+q", "T+q+Ts")
    cols, rows = [], []
    for d, c in zip(days, ck):
        s = driver_columns(drv, man, c, d)
        olr = driver_rlutcs(drv, man, s)
        il, io = nearest_cell(s["lat"], s["lon"], mlat, mlon)
        dTs = jnp.asarray(dTs_grid[il, io])
        pf = np.asarray(s["p_full"])
        Tw = s["T"] + jnp.asarray(_on_columns(plev, dT, pf, 0.0))
        qd = s["q_v"] * jnp.asarray(_on_columns(plev, ratio, pf, 1.0))
        base_ts = s["T_sfc"]
        x = [olr(s["T"], s["q_v"]), olr(Tw, s["q_v"]), olr(s["T"], qd)]
        s["T_sfc"] = base_ts + dTs
        x += [olr(s["T"], s["q_v"])]
        s["T_sfc"] = base_ts
        x += [olr(Tw, qd)]
        s["T_sfc"] = base_ts + dTs
        x += [olr(Tw, qd)]
        s["T_sfc"] = base_ts
        binned = [bin_columns(v, s["w"], il, io, dTs_grid.shape) for v in x]
        has = binned[0][1]
        cols.append([b[0] for b in binned])
        r = [float(v @ s["w"]) for v in x]
        rows.append(r)
        print(f"day {d:3d}: base {r[0]:8.2f}  " + "  ".join(
            f"{l} {v - r[0]:+6.2f}" for l, v in zip(labs[1:], r[1:])) + " W/m2")
    m = np.mean(np.asarray(rows), axis=0)
    months, cm = cmor_global(args.driver_run, "rlutcs")
    print(f"\nBASE CHECK: offline clear-sky OLR, mean of checkpoint days "
          f"{days} = {m[0]:.2f} W/m2; CMOR rlutcs mean over months {months} "
          f"(regional-tool global weighting) = {cm:.2f}; "
          f"diff {m[0] - cm:+.2f} (tolerance {args.tol} W/m2) -> "
          f"{'PASS' if abs(m[0] - cm) <= args.tol else 'FAIL'}")
    tgt = args.target
    print(f"rlutcs error to explain {tgt:+.2f} W/m2; correction must supply "
          f"{-tgt:+.2f}")
    for lab, i in zip(labs[1:], range(1, len(labs))):
        print(f"  {lab:8s} d = {m[i] - m[0]:+6.2f} W/m2  "
              f"({100 * (m[i] - m[0]) / -tgt:5.1f} % of target)")
    g = np.mean(np.asarray(cols), axis=0)
    print(f"\n{int((~has).sum())} of {has.size} CMOR cells hold no model "
          "column and are left out of the region means below.")
    print("\nregional d(clear-sky OLR) [W/m2], same region boxes and land "
          "masks as regional_bias.py:")
    print(f"{'region':<18}" + "".join(f"{l:>9}" for l in labs[1:]))
    for name, box in rb.REGIONS.items():
        v = rb._region_valid(name, land)
        v = has if v is None else (v & has)
        print(f"{name:<18}" + "".join(
            f"{rb.region_mean(g[i] - g[0], mlat, mlon, box, v):9.2f}"
            for i in range(1, len(labs))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None,
                    help="Run whose columns are used (default: the harness's "
                         "own run, which is what it was reviewed against).")
    ap.add_argument("--ref-run", default="ref1979",
                    help="Run the bias PROFILE is measured from.")
    ap.add_argument("--day", type=int, default=1155)
    ap.add_argument("--ncol", type=int, default=2048)
    ap.add_argument("--n-tod", type=int, default=4)
    ap.add_argument("--res", type=int, default=5,
                    help="Icosahedral level of --run's mesh (res6 = 40962 cells).")
    ap.add_argument("--target", type=float, default=-19.6,
                    help="The rlutcs error to be explained [W/m2].")
    ap.add_argument("--driver-run", default=None,
                    help="Run folder under LEGOESM_AMIP_RUNS: build that run's "
                         "own driver from its manifest and use its columns.")
    ap.add_argument("--days", default="5,10,15,20,25,30",
                    help="Checkpoint days averaged in --driver-run mode.")
    ap.add_argument("--out-dir", default=None,
                    help="Scratch output folder for the rebuilt driver.")
    ap.add_argument("--manifest", default="run_manifest.json",
                    help="Manifest file name inside the run folder (a restarted "
                         "run keeps link 1's as run_manifest.link1.json).")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="Base-check tolerance vs CMOR rlutcs [W/m2].")
    args = ap.parse_args(argv)
    if args.driver_run:
        return main_driver(args)

    hh = _load(_HARNESS, "hh")
    if args.run:
        hh.RUN = f"{hh.RUN.rsplit('/', 1)[0]}/{args.run}"
    hh.RES = args.res
    plev, dT, ratio = bias_profiles(args.ref_run)
    print(f"perturbation profile from {args.ref_run} (global mean, "
          f"below-ground masked):")
    for p, t, r in zip(plev, dT, ratio):
        print(f"   {p / 100:7.0f} hPa   dT(ERA5-model) {t:+6.2f} K   "
              f"q_ERA5/q_model {r:5.2f}")

    s = hh.load_state(args.day, ncol=args.ncol)
    w = np.asarray(s["w"], dtype=np.float64)
    p_full = np.asarray(s["p_full"], dtype=np.float64)

    # fill=0.0 for dT and 1.0 for the ratio: outside the measured range the
    # perturbation is the IDENTITY, never an extrapolation.
    dT_col = _on_columns(plev, dT, p_full, 0.0)
    r_col = _on_columns(plev, ratio, p_full, 1.0)

    def olrcs(state):
        flux, _ = hh.diurnal(state, n_tod=args.n_tod, include_clouds=False,
                            n_sub=1)
        return hh.gm(flux[2], w)          # rlut with clouds off = rlutcs

    import jax.numpy as jnp
    base = dict(s)
    warm = dict(s, T=jnp.asarray(np.asarray(s["T"]) + dT_col))
    dry = dict(s, q_v=jnp.asarray(np.asarray(s["q_v"]) * r_col))
    both = dict(s, T=warm["T"], q_v=dry["q_v"])

    o_base = olrcs(base)
    o_warm = olrcs(warm)
    o_dry = olrcs(dry)
    o_both = olrcs(both)

    print(f"\ncolumns: {args.ncol} from {hh.RUN.rsplit('/', 1)[-1]} "
          f"day {args.day}, {args.n_tod} times of day, clear sky")
    print(f"  base rlutcs                        {o_base:8.2f} W/m2")
    print(f"  + temperature correction to ERA5   {o_warm:8.2f}  "
          f"(d = {o_warm - o_base:+6.2f})")
    print(f"  + humidity correction to ERA5      {o_dry:8.2f}  "
          f"(d = {o_dry - o_base:+6.2f})")
    print(f"  + both                             {o_both:8.2f}  "
          f"(d = {o_both - o_base:+6.2f})")
    print(f"\nrlutcs error to explain: {args.target:+.1f} W/m2 "
          f"(model - obs), so a correction must supply {-args.target:+.1f}.")
    print(f"  temperature alone covers {100 * (o_warm - o_base) / -args.target:5.1f} %")
    print(f"  humidity alone covers    {100 * (o_dry - o_base) / -args.target:5.1f} %")
    print(f"  both together cover      {100 * (o_both - o_base) / -args.target:5.1f} %")
    print("\nSENSITIVITY OF THE RADIATION OPERATOR, not a coupled prediction: "
          "correcting T in a live run also moves the circulation, the humidity "
          "and the clouds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

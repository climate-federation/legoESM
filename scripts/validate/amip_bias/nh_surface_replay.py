#!/usr/bin/env python3
"""Northern-winter land surface radiation split and momentum handoff, replayed on
the run's own code path from a checkpoint, scored against ERA5 at the same 2001
dates.

capture: relaunches the run with its own arguments from ``checkpoint_day_N``
(``mpas_onestep_param_grad.launch_argv``, the same relaunch heating_budget.py
uses), lets the driver take ONE real step (so the land bootstrap step runs and
the step-1 forcing carries the land skin, albedo and fluxes), and captures the
step-1 state.  Recorded, all from the model's own functions:

* LAND (``_step_multilayer_land_impl``, wrapped; jax.debug.callback): the
  bootstrap land call's forcing sw/lw down, canopy sw_net / lw_net / lw_up,
  H, LE, G, skin, albedo, roughness z0, the land model's MOST stress
  tau_x/tau_y (rho u*^2), held mask, lowest-level wind/rho/height.
* BULK (``surface_layer.surface_fluxes_at_lowest_level``, wrapped): the stress
  and u* that the turbulence (CLUBB) receives over the whole cell, from the
  run's surface config (large_yeager_cesm, 0.5 m/s minwind), when the
  turbulence module is called eagerly on the captured step-1 state.
* RADIATION (``_call_radiation_backend``, wrapped): the radiation module called
  eagerly on the step-1 state (00Z + one step): instantaneous surface and TOA
  fluxes; then the SAME backend re-called (same columns, same cloud inputs) in
  the model's daily-mean insolation mode (``_compute_insolation`` with the
  diurnal cycle off) for a daily-mean shortwave, and with zero condensate and
  zero cloud fraction for clear sky.  Clouds are the 00Z snapshot.

score: land cells (driver land fraction > 0.5, ERA5 lsm > 0.5, ERA5 snow < 9.9 m
w.e. i.e. no glaciers), 45-70N / Siberia / Canada / forest / open, weights area x
land fraction, against ERA5 (nh_surface_replay_era5.sh): daily-mean ssrd/ssr,
00Z-hour strd/str, 00Z-hour ewss/nsss stress, 00Z 10 m wind and ERA5 model
levels 133-137 interpolated to the model's lowest-level height.

Run with the run's code on PYTHONPATH, JAX_PLATFORMS=cpu.  Prints numbers only.

Usage:
  nh_surface_replay.py capture <run> --day N --out DIR
  nh_surface_replay.py score <run> --days N ... --out DIR --era5 DIR --era5-an DIR
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LAND_KEYS = ("sw_down", "lw_down", "u_lowest", "v_lowest", "rho_lowest", "T_lowest",
             "z_lowest")
SURF_KEYS = ("sw_net", "lw_net", "lw_up", "shflx", "lhflx", "G_soil", "T_surface",
             "albedo", "z0", "tau_x", "tau_y", "held")
SURF_OPT_KEYS = ("tau_mag", "converged", "n_iters")


class _Captured(BaseException):
    pass



def _era5_field(path, var, date):
    """One ERA5 00Z field from a cdo -R regular Gaussian netCDF; checks the
    variable name and the timestamp, returns (values, lat, lon, plev or None)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        if var not in ds.data_vars:
            raise SystemExit(f"FATAL: {path} has no {var} ({list(ds.data_vars)})")
        tt = np.asarray(ds["time"].values)
        if tt.size != 1 or str(tt[0])[:19] != f"{date}T00:00:00":
            raise SystemExit(f"FATAL: {path} time {tt} is not {date} 00Z")
        da = ds[var].squeeze("time", drop=True)
        da = da.transpose(*[d for d in ("plev", "lat", "lon") if d in da.dims])
        plev = np.asarray(da["plev"], dtype=np.float64) if "plev" in da.dims else None
        return (np.asarray(da.values, dtype=np.float64),
                np.asarray(da["lat"], dtype=np.float64),
                np.asarray(da["lon"], dtype=np.float64) % 360.0, plev)


def era5_columns(e5dir, date, lat_deg, lon_deg):
    """ERA5 00Z T [K] and specific humidity [kg/kg] profiles on its 37 levels
    [Pa], nearest ERA5 column to each model cell, with a SURFACE node at the
    ERA5 surface pressure (2 m temperature; specific humidity from the 2 m
    dewpoint, Tetens over water as ECMWF defines it).  ERA5 levels at or below
    its own surface (extrapolated by ECMWF) are not used."""
    from legoesm import constants
    from legoesm.thermo import saturation_vapor_pressure
    T, la, lo, plev = _era5_field(f"{e5dir}/e5pl_130_{date}.nc", "T", date)
    q, la2, lo2, plev2 = _era5_field(f"{e5dir}/e5pl_133_{date}.nc", "Q", date)
    if not (np.array_equal(plev, plev2) and np.array_equal(la, la2) and np.array_equal(lo, lo2)):
        raise SystemExit("FATAL: ERA5 T and Q files are on different grids")
    if not (np.all(np.diff(plev) > 0) and plev.min() > 0.0
            and 9.0e4 <= plev.max() <= 1.1e5 and plev.min() < 1.0e3):
        raise SystemExit(f"FATAL: ERA5 plev not ascending Pa 1..1000 hPa: {plev[[0, -1]]}")
    sfc = {}
    for code, name in ((134, "SP"), (167, "T2M"), (168, "D2M")):
        v, la3, lo3, _ = _era5_field(f"{e5dir}/e5sf_{code}_{date}.nc", name, date)
        if not (np.array_equal(la3, la) and np.array_equal(lo3, lo)):
            raise SystemExit(f"FATAL: ERA5 {name} is not on the profile grid")
        sfc[name] = v
    sp, t2, d2 = sfc["SP"], sfc["T2M"], sfc["D2M"]
    i = np.abs(la[:, None] - lat_deg[None, :]).argmin(0)
    dlon = np.abs(((lo[:, None] - (lon_deg[None, :] % 360.0)) + 180.0) % 360.0 - 180.0)
    j = dlon.argmin(0)
    Tc, qc = T[:, i, j].T, q[:, i, j].T                       # (ncol, nplev)
    psc, t2c, d2c = sp[i, j], t2[i, j], d2[i, j]
    e2 = np.asarray(saturation_vapor_pressure(d2c), dtype=np.float64)
    q2 = constants.epsilon * e2 / (psc - (1.0 - constants.epsilon) * e2)
    for name, v in (("T", Tc), ("Q", qc), ("SP", psc), ("T2M", t2c), ("q2", q2)):
        if not np.isfinite(v).all():
            raise SystemExit(f"FATAL: non-finite ERA5 {name}")
    if not (5.0e4 < psc.min() and psc.max() < 1.1e5):
        raise SystemExit(f"FATAL: ERA5 surface pressure out of range {psc.min()}..{psc.max()}")
    return dict(T=Tc, q=qc, plev=plev, sp=psc, t2=t2c, q2=q2,
                dlat=np.abs(la[i] - lat_deg))


def column_interp(prof, plev, sfc_p, sfc_x, p_full):
    """Per column: nodes = ERA5 levels above its surface, plus (sfc_p, sfc_x);
    linear in log p to p_full [Pa]; constant beyond the end nodes.  Returns the
    profile and the mask of model layers beyond the end nodes."""
    out = np.empty_like(p_full)
    for c in range(p_full.shape[0]):
        above = plev < sfc_p[c]
        xp = np.log(np.append(plev[above], sfc_p[c]))
        fp = np.append(prof[c, above], sfc_x[c])
        out[c] = np.interp(np.log(p_full[c]), xp, fp)
    beyond = (p_full > sfc_p[:, None]) | (p_full < plev[0])
    return out, beyond

def capture(a):
    import mpas_onestep_param_grad as H
    t0 = time.time()
    out = Path(a.out)
    scratch = out / f"_launch_{a.run}_d{a.day:04d}{a.tag or ''}"
    scratch.mkdir(parents=True, exist_ok=True)
    argv = H.launch_argv(a.run, a.day, scratch)
    while "--distributed" in argv:          # single CPU process
        argv.remove("--distributed")
    if a.config:
        argv[argv.index("--config") + 1] = a.config
    argv += list(a.extra or [])            # what-if flags, appended last
    tag = getattr(a, "tag", "") or ""

    import jax
    import jax.numpy as jnp
    import legoesm.atmosphere.physics.combined as combined
    import legoesm.atmosphere.physics.radiation.integration as rad_int
    import legoesm.atmosphere.physics.turbulence.surface_layer as sl
    import legoesm.land.multilayer_land as mll
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationModel
    from legoesm.driver.model_driver import ModelDriver

    flags = {"land": False, "atm": False}
    rec: dict = {}

    # --- land: the bootstrap call after step 0 (jitted -> host callback) ---
    real_impl = mll._step_multilayer_land_impl

    def land_cb(*vals):
        if flags["land"] and "land" not in rec:
            rec["land"] = {k: np.asarray(v) for k, v in
                           zip(LAND_KEYS + SURF_KEYS + SURF_OPT_KEYS, vals)}

    @functools.wraps(real_impl)
    def impl(state, forcing, *x, **k):
        res = real_impl(state, forcing, *x, **k)
        so = res[3]
        held = so.held if so.held is not None else jnp.zeros_like(so.shflx)
        z = forcing.z_lowest if forcing.z_lowest is not None else jnp.full_like(so.shflx, jnp.nan)
        def _g(n):   # optional fields (older code / non-iterating schemes): NaN
            v = getattr(so, n, None)
            return (jnp.full_like(so.shflx, jnp.nan) if v is None
                    else jnp.asarray(v, dtype=so.shflx.dtype).reshape(so.shflx.shape))
        vals = [getattr(forcing, n) for n in LAND_KEYS[:-1]] + [z] + \
               [getattr(so, n) for n in SURF_KEYS[:-1]] + [held] + \
               [_g(n) for n in SURF_OPT_KEYS]
        jax.debug.callback(land_cb, *vals)
        return res
    mll._step_multilayer_land_impl = impl

    # --- bulk stress handed to the turbulence ---
    real_sfl = sl.surface_fluxes_at_lowest_level

    def sfl_cb(*vals):
        if flags["atm"]:
            rec["bulk"] = dict(zip(("tx", "ty", "sh", "lh", "us", "u", "v", "T", "T_sfc",
                                    "rho", "z_low"), (np.asarray(v) for v in vals)))

    @functools.wraps(real_sfl)
    def sfl(u, v, T, q_v, T_sfc, q_sfc, rho, config, z_low):
        r = real_sfl(u, v, T, q_v, T_sfc, q_sfc, rho, config, z_low)
        jax.debug.callback(sfl_cb, *r, u, v, T, T_sfc, rho,
                           z_low if z_low is not None else jnp.full_like(u, jnp.nan))
        rec["bulk_cfg"] = config
        return r
    sl.surface_fluxes_at_lowest_level = sfl

    # --- radiation backend: record the call made on the captured state ---
    real_backend = rad_int._call_radiation_backend
    bsig = inspect.signature(real_backend)

    def spy_backend(*x, **k):
        if flags["atm"] and "rad_args" not in rec:
            rec["rad_args"] = bsig.bind(*x, **k).arguments
        return real_backend(*x, **k)
    rad_int._call_radiation_backend = spy_backend

    # --- module builders (radiation, turbulence) and the full-physics build ---
    fns: dict = {}

    def spy_builder(name, real):
        @functools.wraps(real)
        def build(*x, **k):
            fn = real(*x, **k)
            fns.setdefault(name, []).append(fn)
            return fn
        return build
    combined.make_radiation_physics = spy_builder("rad", combined.make_radiation_physics)
    combined.make_turbulence_physics = spy_builder("turb", combined.make_turbulence_physics)

    drv: dict = {}
    real_run = ModelDriver.run

    def run(self, *x, **k):
        drv["d"] = self
        return real_run(self, *x, **k)
    ModelDriver.run = run

    real_step = MPASPrimitiveEquationModel.step
    cap: dict = {}

    def step(self, state, dt_, physics_fn=None, forcing=None, phys_state=None):
        if not cap.get("stepped"):
            cap["stepped"] = True
            r = real_step(self, state, dt_, physics_fn=physics_fn, forcing=forcing,
                          phys_state=phys_state)
            flags["land"] = True       # the land bootstrap call follows this step
            return r
        cap.update(model=self, state=state, dt=dt_, forcing=forcing, phys_state=phys_state)
        raise _Captured()
    MPASPrimitiveEquationModel.step = step

    # run_amip from the RUN'S checkout (cwd), matching the library on PYTHONPATH.
    _ra = Path.cwd() / "scripts" / "run"
    if not (_ra / "run_amip.py").exists():
        raise SystemExit(f"run from the run's checkout: no {_ra}/run_amip.py")
    sys.path.insert(0, str(_ra))
    import run_amip
    print(f"run_amip from {run_amip.__file__}; argv tail {argv[-8:]}", flush=True)
    try:
        run_amip.main(argv)
    except _Captured:
        pass
    else:
        raise SystemExit("FATAL: driver returned without a second model.step")
    jax.effects_barrier()
    if "land" not in rec:
        raise SystemExit("FATAL: the land bootstrap call was never recorded")
    print(f"captured step 1 ({time.time() - t0:.0f}s)", flush=True)
    if getattr(a, "land_only", False):
        # Only the land bootstrap call's outputs (what-if comparisons of the
        # land scheme on identical inputs); no turbulence / radiation replay.
        _mesh = cap["model"].mesh
        out_arr = {f"land_{k}": v for k, v in rec["land"].items()}
        out_arr["lat"] = np.rad2deg(np.asarray(_mesh.latCell, dtype=np.float64))
        out_arr["area"] = np.asarray(_mesh.areaCell, dtype=np.float64)
        out_arr["f_land"] = np.asarray(drv["d"]._f_land, dtype=np.float64).reshape(-1)
        f = out / f"cap_{a.run}_d{a.day:04d}{tag}.npz"
        np.savez(f, meta=json.dumps({"run": a.run, "day": a.day, "argv": argv,
                                     "land_only": True}), **out_arr)
        print(f"wrote {f} ({time.time() - t0:.0f}s)")
        return 0

    model, state = cap["model"], cap["state"]
    mesh, sig = model.mesh, model.sigma_coord
    forcing, phys_state = cap["forcing"], cap["phys_state"]
    d = drv["d"]
    flags["atm"] = True

    def call(fn):
        kw = {"phys_state": phys_state}
        if getattr(fn, "_wants_forcing", False):
            kw["forcing"] = forcing
        return fn(state, mesh, sig, **kw)

    # turbulence: the LAST built turbulence fn is the production one (same as rad below)
    call(fns["turb"][-1])
    jax.effects_barrier()
    if "bulk" not in rec:
        raise SystemExit("FATAL: turbulence never called the surface-layer helper")
    print(f"turbulence eager ({time.time() - t0:.0f}s)", flush=True)
    rad_fns = fns["rad"]
    call(rad_fns[-1])
    if "rad_args" not in rec:
        raise SystemExit("FATAL: radiation backend not called on the captured state")
    print(f"radiation eager ({time.time() - t0:.0f}s)", flush=True)

    ra = dict(rec["rad_args"])
    cfg = ra["radiation_config"]
    if not getattr(cfg, "diurnal_cycle", False) or ra.get("cos_sza") is None:
        raise SystemExit("FATAL: production radiation is not on the diurnal path")

    def solve(**over):
        k = dict(ra)
        k.update(over)
        o = real_backend(**k)
        return {"sw_dn_sfc": np.asarray(o.sw_flux_down[:, -1]),
                "sw_up_sfc": np.asarray(o.sw_flux_up[:, -1]),
                "lw_dn_sfc": np.asarray(o.lw_flux_down[:, -1]),
                "lw_up_sfc": np.asarray(o.lw_flux_up[:, -1]),
                "olr": np.asarray(o.lw_flux_up[:, 0]),
                "sw_up_toa": np.asarray(o.sw_flux_up[:, 0]),
                "insol": np.asarray(o.toa_insolation)}
    z0c = lambda v: None if v is None else jnp.zeros_like(v)
    clear = dict(q_cloud=z0c(ra.get("q_cloud")), q_ice=z0c(ra.get("q_ice")),
                 cloud_fraction_override=z0c(ra.get("cloud_fraction_override")),
                 conv_precip=z0c(ra.get("conv_precip")),
                 conv_mass_flux_up=z0c(ra.get("conv_mass_flux_up")),
                 conv_icwmr=z0c(ra.get("conv_icwmr")))
    # The model's own daily-mean insolation branch, at the run's own calendar
    # (forcing day_of_year, transient TSI).  The gray sub-config defaults to
    # perpetual equinox, which that branch honours -- switch it off, or the
    # "daily mean" is an equinox (first attempt: 70N got 148 W/m2 in December).
    doy = float(np.asarray(forcing["day_of_year"]))
    cfg_dm = cfg._replace(diurnal_cycle=False,
                          gray=cfg.gray._replace(perpetual_equinox=False))
    ins, cz, f_day, eccf = rad_int._compute_insolation(
        ra["lat"], cfg_dm, ra.get("lon"), day_of_year=doy)
    assert cz is None and f_day is not None
    S0 = cfg.rrtmgp.S_0
    tsi = forcing.get("tsi")
    if tsi is not None:
        ins = ins * (tsi / S0)
    # CONTROL: the daily-mean TOA insolation must equal the mean of the model's
    # own diurnal insolation over 48 half-hours of the same day.
    acc = 0.0
    for h in range(48):
        i_h = rad_int._compute_insolation(ra["lat"], cfg, ra.get("lon"),
                                          day_of_year=np.floor(doy),
                                          seconds_of_day=1800.0 * h + 900.0)[0]
        acc = acc + i_h / 48.0
    if tsi is not None:
        acc = acc * (tsi / S0)
    lat_c = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
    bnd = (lat_c >= 40) & (lat_c <= 70)
    d_ins = float(np.mean(np.abs(np.asarray(ins) - np.asarray(acc))[bnd]))
    print(f"insolation control 40-70N: daily-mean branch {float(np.mean(np.asarray(ins)[bnd])):.2f} "
          f"vs 48-sample diurnal mean {float(np.mean(np.asarray(acc)[bnd])):.2f} W/m2, "
          f"mean |diff| {d_ins:.2f}", flush=True)
    if d_ins > 2.0:
        raise SystemExit("FATAL: daily-mean insolation does not match the diurnal mean")
    dm = dict(cos_sza=None, insolation=ins, f_day=f_day, eccf=eccf)
    arms = {"inst": {}, "inst_clr": clear, "dmean": dm, "dmean_clr": {**dm, **clear},
            # cloud phase split at the 00Z instant: the same cloud fraction,
            # with only ice (no liquid) or only liquid (no ice)
            "inst_noliq": {"q_cloud": z0c(ra.get("q_cloud"))},
            "inst_noice": {"q_ice": z0c(ra.get("q_ice")),
                           "conv_icwmr": z0c(ra.get("conv_icwmr"))}}
    if a.era5_pl:
        # Clear-sky radiation with ONE atmospheric input swapped to the ERA5
        # 00Z analysis of the same date (skin, clouds, gases unchanged), and
        # CONTROL arms: the model's own T and q taken to the ERA5 levels and
        # back (with the model's own lowest layer as the surface node), so the
        # interpolation error is measured on the same columns and gated below.
        if getattr(getattr(cfg, "cloud_config", None), "cap_floor_on", False):
            raise SystemExit("FATAL: polar cloud floor is on; zeroed clouds are not clear sky")
        if ra.get("ml_ozone_coefs") is not None and ra.get("o3_vmr_override") is None:
            raise SystemExit("FATAL: ML ozone reads T; a T swap would also change ozone")
        date = (dt.date(2001, 1, 1) + dt.timedelta(days=a.day)).isoformat()
        lat_d = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
        lon_d = np.rad2deg(np.asarray(mesh.lonCell, dtype=np.float64))
        pf = np.asarray(ra["p_full"], dtype=np.float64)
        if not (np.isfinite(pf).all() and (pf > 0.0).all() and np.all(np.diff(pf, axis=1) > 0)
                and 5.0e4 < pf[:, -1].min() and pf[:, -1].max() < 1.1e5):
            raise SystemExit("FATAL: p_full is not finite, top-to-bottom, in Pa")
        T_m = np.asarray(ra["T"], dtype=np.float64)
        q_m = np.asarray(ra["q_v"], dtype=np.float64)
        e5 = era5_columns(a.era5_pl, date, lat_d, lon_d)
        T_e5, beyond = column_interp(e5["T"], e5["plev"], e5["sp"], e5["t2"], pf)
        q_e5, _ = column_interp(e5["q"], e5["plev"], e5["sp"], e5["q2"], pf)
        # model -> ERA5 levels -> model, surface node = the model's lowest layer
        to_pl = lambda X: np.stack([np.interp(np.log(e5["plev"]), np.log(pf[c]), X[c])
                                    for c in range(pf.shape[0])])
        T_rt, _ = column_interp(to_pl(T_m), e5["plev"], pf[:, -1], T_m[:, -1], pf)
        q_rt, _ = column_interp(to_pl(q_m), e5["plev"], pf[:, -1], q_m[:, -1], pf)
        cast = lambda v, ref: jnp.asarray(v, dtype=np.asarray(ref).dtype)
        arms.update({
            "clr_e5T": {**clear, "T": cast(T_e5, ra["T"])},
            "clr_e5q": {**clear, "q_v": cast(q_e5, ra["q_v"])},
            "clr_e5Tq": {**clear, "T": cast(T_e5, ra["T"]), "q_v": cast(q_e5, ra["q_v"])},
            "clr_selfT": {**clear, "T": cast(T_rt, ra["T"])},
            "clr_selfq": {**clear, "q_v": cast(q_rt, ra["q_v"])}})
        out_arr_e5 = {"e5_T_minus_model": T_e5 - T_m, "e5_q_minus_model": q_e5 - q_m,
                      "e5_selfT_minus_model": T_rt - T_m, "e5_selfq_minus_model": q_rt - q_m,
                      "e5_beyond_nodes": beyond, "e5_p_full": pf, "e5_dlat": e5["dlat"]}
        print(f"ERA5 swap arms for {date} added; model layers beyond ERA5 nodes: "
              f"{100 * beyond.mean():.2f} %", flush=True)
    else:
        out_arr_e5 = {}
    # Column condensate handed to the radiation (grid-mean paths, kg/m2) and
    # its cloud fraction: where the cloud water is, and how much the solver gets.
    from legoesm import constants
    radin = {}
    _dp = np.abs(np.diff(np.asarray(ra["p_half"], dtype=np.float64), axis=1))
    for _k in ("q_cloud", "q_ice", "conv_icwmr"):
        if ra.get(_k) is not None:
            radin[f"radin_{_k}_path"] = (
                np.asarray(ra[_k], dtype=np.float64) * _dp).sum(1) / constants.g
    if ra.get("cloud_fraction_override") is not None:
        # The CLUBB (liquid) fraction carried in -- NOT the radiation's total
        # cloud fraction, which compute_cloud_properties builds from it plus
        # the CAM6 ice fraction and convective cloud (below).
        _cf = np.asarray(ra["cloud_fraction_override"], dtype=np.float64)
        radin["radin_cfclubb_max"] = _cf.max(1)
    # The cloud field the solver actually got: the backend's own call.
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties)
    from legoesm.diagnostics.cloud_overlap import maximum_random_overlap
    _cc = cfg.cloud_config if cfg.cloud_config is not None else CloudConfig(
        scheme=cfg.cloud_scheme)
    _ph = ra["p_half"]
    _cp = compute_cloud_properties(
        T=ra["T"], p_full=ra["p_full"], q_v=ra["q_v"],
        dp=_ph[:, 1:] - _ph[:, :-1], config=_cc, q_cloud=ra.get("q_cloud"),
        q_ice=ra.get("q_ice"), n_ice=ra.get("n_ice"), n_cloud=ra.get("n_cloud"),
        conv_precip=ra.get("conv_precip"),
        cloud_fraction_override=ra.get("cloud_fraction_override"),
        lat=ra["lat"], conv_mass_flux_up=ra.get("conv_mass_flux_up"),
        conv_icwmr=ra.get("conv_icwmr"), p_half=_ph)
    _cfr = np.asarray(_cp.cloud_fraction, dtype=np.float64)
    _p = np.asarray(ra["p_full"], dtype=np.float64)
    radin["rad_cf_layer_max"] = _cfr.max(1)
    radin["rad_cf_total_maxrand"] = np.asarray(maximum_random_overlap(_cp.cloud_fraction))
    radin["rad_cf_low_maxrand"] = np.asarray(maximum_random_overlap(
        jnp.where(jnp.asarray(_p) > 68000.0, _cp.cloud_fraction, 0.0)))
    radin["rad_lwp"] = np.asarray(_cp.lwp, dtype=np.float64).sum(1)
    radin["rad_iwp"] = np.asarray(_cp.iwp, dtype=np.float64).sum(1)
    if a.snow_arms:
        # CAM6 radiates falling snow with its OWN cover, path and size and snow
        # optics (micro_mg_cam.F90 cldfsnow / icswp; rrtmg radiation.F90
        # snow_cloud_get_rad_props_lw, cam_cesm2_1_rel).  The radiation here
        # has no snow condensate (the prognostic snow tracer is never passed to
        # it), so these arms ADD the snow (snow+graupel) mass to the cloud ice
        # handed to the backend.  A SENSITIVITY, not a bound on CAM6: the added
        # mass gets ice optics and, with fixed n_ice, a larger ice radius, and
        # it may change the backend's ice cover (audited below); snow in
        # zero-cover layers stays invisible where CAM gives it 0.25 cover.
        # Graupel is beyond CAM6 MG2 (no graupel there): "snowg" is a mass
        # sensitivity only.
        _tr = cap["state"].tracers
        _qs = jnp.asarray(_tr["q_s"].data if hasattr(_tr["q_s"], "data") else _tr["q_s"])
        _qg = jnp.asarray(_tr["q_g"].data if hasattr(_tr["q_g"], "data") else _tr["q_g"])
        _qi = jnp.asarray(ra["q_ice"])
        if _qs.shape != _qi.shape or _qg.shape != _qi.shape:
            raise SystemExit(f"FATAL: snow/graupel {_qs.shape}/{_qg.shape} vs ice {_qi.shape}")
        # Same state, same level order: the cloud-ice tracer must match the
        # ice the backend was handed (else the snow lands on the wrong levels).
        if getattr(_cc, "cap_floor_on", False):
            raise SystemExit("FATAL: polar cloud floor is on; the cover audit would miss it")
        if "q_i" not in _tr:
            raise SystemExit("FATAL: no q_i tracer to check the radiation's q_ice against")
        _qit = np.asarray(_tr["q_i"].data if hasattr(_tr["q_i"], "data") else _tr["q_i"],
                          dtype=np.float64)
        _qin = np.asarray(_qi, dtype=np.float64)
        # element-wise (column AND level AND scale): the snow tracer comes from
        # the same state as the cloud-ice tracer, so if q_i lines up with the
        # radiation's q_ice cell by cell, q_s does too.
        if _qit.shape != _qin.shape:
            raise SystemExit(f"FATAL: q_i tracer {_qit.shape} vs radiation q_ice {_qin.shape}")
        _rel = float(np.abs(_qit - _qin).max() / max(np.abs(_qin).max(), 1e-30))
        print(f"snow arms: q_i tracer vs radiation q_ice, max |diff| / max {_rel:.2e}", flush=True)
        if not (np.isfinite(_rel) and _rel < 1e-3 and _qin.max() > 0.0):
            raise SystemExit("FATAL: cloud-ice tracer does not match the radiation's q_ice cell by cell")
        _dpn = np.abs(np.diff(np.asarray(_ph, dtype=np.float64), axis=1))
        for _n, _q in (("snow", _qi + _qs.astype(_qi.dtype)),
                       ("snowg", _qi + (_qs + _qg).astype(_qi.dtype))):
            arms[f"inst_{_n}"] = {"q_ice": _q}
            _cps = compute_cloud_properties(
                T=ra["T"], p_full=ra["p_full"], q_v=ra["q_v"],
                dp=_ph[:, 1:] - _ph[:, :-1], config=_cc, q_cloud=ra.get("q_cloud"),
                q_ice=_q, n_ice=ra.get("n_ice"), n_cloud=ra.get("n_cloud"),
                conv_precip=ra.get("conv_precip"),
                cloud_fraction_override=ra.get("cloud_fraction_override"),
                lat=ra["lat"], conv_mass_flux_up=ra.get("conv_mass_flux_up"),
                conv_icwmr=ra.get("conv_icwmr"), p_half=_ph)
            radin[f"rad_iwp_{_n}"] = np.asarray(_cps.iwp, dtype=np.float64).sum(1)
            radin[f"rad_cf_total_maxrand_{_n}"] = np.asarray(
                maximum_random_overlap(_cps.cloud_fraction))
            radin[f"rad_cf_layer_change_{_n}"] = np.abs(
                np.asarray(_cps.cloud_fraction, dtype=np.float64) - _cfr).max(1)
            _w_ice = np.asarray(_cps.iwp, dtype=np.float64)
            radin[f"rad_reff_ice_iwpw_{_n}"] = (
                (np.asarray(_cps.r_eff_ice, dtype=np.float64) * _w_ice).sum(1)
                / np.maximum(_w_ice.sum(1), 1e-30))
        _w0 = np.asarray(_cp.iwp, dtype=np.float64)
        radin["rad_reff_ice_iwpw"] = ((np.asarray(_cp.r_eff_ice, dtype=np.float64) * _w0).sum(1)
                                      / np.maximum(_w0.sum(1), 1e-30))
        radin["radin_q_snow_path"] = (np.asarray(_qs, dtype=np.float64) * _dpn).sum(1) / constants.g
        radin["radin_q_graupel_path"] = (np.asarray(_qg, dtype=np.float64) * _dpn).sum(1) / constants.g
    out_arr = {}
    for n, o in arms.items():
        for k, v in solve(**o).items():
            out_arr[f"rad_{n}_{k}"] = v
        print(f"radiation arm {n} ({time.time() - t0:.0f}s)", flush=True)
    if a.snow_arms:
        # CANARY: the arms must reach the solver (an ignored override would
        # read as "snow does not matter").
        for _n in ("snow", "snowg"):
            _d = np.abs(out_arr[f"rad_inst_{_n}_lw_dn_sfc"] - out_arr["rad_inst_lw_dn_sfc"]).max()
            _g = float((radin[f"rad_iwp_{_n}"] - radin["rad_iwp"]).sum())
            print(f"snow arm {_n}: max |dDLW| {_d:.3f} W/m2, added radiative ice "
                  f"{_g:.4g} kg/m2 summed over columns", flush=True)
            if not (_d > 0.0 and _g > 0.0):
                raise SystemExit(f"FATAL: snow arm {_n} did not reach the radiation")
    control_fail: list = []
    if a.era5_pl:
        # CONTROL GATE: the round-trip arms must reproduce clear-sky surface
        # downward LW over 45-70N land within 1 W/m2 (area-weighted), else the
        # interpolation error is the size of the effect and the swap is void.
        _lat = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
        _fl = np.asarray(d._f_land, dtype=np.float64).reshape(-1)
        _w = np.asarray(mesh.areaCell, dtype=np.float64) * ((_lat >= 45) & (_lat <= 70) & (_fl > 0.5))
        _base = out_arr["rad_inst_clr_lw_dn_sfc"]
        for _arm in ("clr_selfT", "clr_selfq"):
            _dd = out_arr[f"rad_{_arm}_lw_dn_sfc"] - _base
            _d = float((_dd * _w).sum() / _w.sum())
            _ad = float((np.abs(_dd) * _w).sum() / _w.sum())     # no cancellation
            print(f"control {_arm}: 45-70N land clear-sky DLW change mean {_d:+.3f}, "
                  f"mean |.| {_ad:.3f} W/m2", flush=True)
            if not np.isfinite(_ad) or _ad > 1.0:
                control_fail.append(f"{_arm}: mean |dDLW| {_ad} W/m2 (limit 1, must be finite)")
    out_arr["rad_sfc_albedo"] = np.broadcast_to(
        np.asarray(ra["sfc_albedo_override"]), out_arr["rad_inst_olr"].shape).copy()
    out_arr["rad_cos_sza"] = np.asarray(ra["cos_sza"])
    out_arr.update(radin)
    out_arr.update(out_arr_e5)
    for k, v in rec["land"].items():
        out_arr[f"land_{k}"] = v
    for k, v in rec["bulk"].items():
        out_arr[f"bulk_{k}"] = v
    bc = rec["bulk_cfg"]
    out_arr["lat"] = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
    out_arr["lon"] = np.rad2deg(np.asarray(mesh.lonCell, dtype=np.float64)) % 360
    out_arr["area"] = np.asarray(mesh.areaCell, dtype=np.float64)
    out_arr["f_land"] = np.asarray(d._f_land, dtype=np.float64).reshape(-1)
    meta = {"run": a.run, "day": a.day, "bulk_scheme": bc.bulk_scheme,
            "z_ref_model_level": bool(getattr(bc, "z_ref_model_level", False)),
            "argv": argv}
    f = out / f"cap_{a.run}_d{a.day:04d}{tag}.npz"
    meta["argv_tail"] = argv[-8:]
    meta["control_fail"] = control_fail
    np.savez(f, meta=json.dumps(meta), **out_arr)
    print(f"wrote {f} ({time.time() - t0:.0f}s)")
    if control_fail:   # written first so the failing arrays can be inspected
        raise SystemExit("FATAL: ERA5-swap control failed: " + "; ".join(control_fail))
    return 0


# ----------------------------------------------------------------- scoring ---
REG = {"45-70N": (45, 70, 0, 360), "Siberia": (50, 70, 60, 140),
       "Canada": (50, 70, 220, 300)}
# ERA5 L137 full-level heights [m] above ground, levels 133..137 (ECMWF L137 table,
# standard atmosphere).  ponytail: fixed table, use per-column geopotential if the
# lowest-level wind ever becomes the deciding number.
_REF: dict = {}
H_ROOT = "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs"
ML_Z = np.array([106.54, 79.30, 53.92, 30.96, 10.00])


def _e5(path, var, date, ii, jj, nrec_ok=1, ref=None):
    import xarray as xr
    v = xr.open_dataset(path)[var].squeeze(drop=True)
    for c in ("lat", "lon"):                 # same grid as the lsm file, or the
        rv = (ref or _REF)[c]
        if v[c].size != rv.size or not np.allclose(v[c].values, rv, atol=1e-3):
            raise SystemExit(f"FATAL: {path} {c} grid differs from e5_172")
    t = v["time"].values.astype("datetime64[D]").astype(str)
    k = np.flatnonzero(t == date)
    if k.size != nrec_ok:
        raise SystemExit(f"FATAL: {path} has {k.size} records for {date}")
    x = np.asarray(v.isel(time=int(k[0])).values, dtype=np.float64)
    return x[..., ii, jj]


def score(a):
    import xarray as xr
    from nh_winter_land_state import nearest_index, wmean
    rows: dict = {}
    for day in a.days:
        z = dict(np.load(Path(a.out) / f"cap_{a.run}_d{day:04d}.npz", allow_pickle=True))
        lat, lon, area, fl = z["lat"], z["lon"], z["area"], z["f_land"]
        # the driver packs the land step onto cells with f_land > 0 (model_driver
        # _land_pack_idx_np); scatter back, NaN elsewhere
        pidx = np.nonzero(fl > 0.0)[0]
        for k in [k for k in z if k.startswith("land_")]:
            if z[k].shape[0] != lat.size:
                if z[k].shape[0] != pidx.size:
                    raise SystemExit(f"FATAL: {k} has {z[k].shape[0]} columns, "
                                     f"pack index {pidx.size}")
                full_ = np.full(lat.size, np.nan)
                full_[pidx] = z[k]
                z[k] = full_
        date = (dt.date(2001, 1, 1) + dt.timedelta(day)).isoformat()
        sel = lat > 25
        e = xr.open_dataset(f"{a.era5_an}/e5_172.nc")
        _REF.update(lat=e["lat"].values, lon=e["lon"].values)
        ii, jj = nearest_index(e, lat[sel], lon[sel])
        full = functools.partial(_put, sel)
        lsm = full(np.asarray(e["var172"].isel(time=0).values)[ii, jj])
        sd0 = full(_e5(f"{a.era5_an}/e5_141.nc", "var141", date, ii, jj))
        land = (fl > 0.5) & (lsm > 0.5) & (sd0 < 9.9)
        E = lambda p, var, kind: full(_e5(f"{a.era5}/e5{kind}_{p}.nc", var, date, ii, jj))
        e5 = {"ssrd_d": E(169, "var169", "d") / 86400, "ssr_d": E(176, "var176", "d") / 86400,
              "strd_d": E(175, "var175", "d") / 86400, "str_d": E(177, "var177", "d") / 86400,
              "sshf_d": -E(146, "var146", "d") / 86400, "strd_h": E(175, "var175", "h") / 3600,
              "str_h": E(177, "var177", "h") / 3600, "sshf_h": -E(146, "var146", "h") / 3600,
              "ewss": E(180, "var180", "h") / 3600, "nsss": E(181, "var181", "h") / 3600,
              "u10": E(165, "var165", "a"), "v10": E(166, "var166", "a"),
              "fsr": E(244, "var244", "a")}
        # ERA5 model levels 133..137 -> model lowest-level height (log-z interpolation)
        eml = xr.open_dataset(f"{a.era5}/e5ml_131.nc")      # spectral -> its own grid
        mref = {"lat": eml["lat"].values, "lon": eml["lon"].values}
        im, jm = nearest_index(eml, lat[sel], lon[sel])
        um = _e5(f"{a.era5}/e5ml_131.nc", "u", date, im, jm, ref=mref)
        vm = _e5(f"{a.era5}/e5ml_132.nc", "v", date, im, jm, ref=mref)
        # IFS spectral u/v are WIND IMAGES u*cos(lat) (first look: level 137
        # read 2.7 m/s against a 5.3 m/s 10 m wind at 45-70N); divide it out.
        coslat = np.cos(np.deg2rad(mref["lat"][im]))[None, :]
        spd = np.hypot(um, vm) / coslat               # (5, nsel), level order 133..137
        e5["wind_lev137"] = full(spd[-1])
        zl = z["land_z_lowest"][sel]
        lz = np.log(ML_Z[::-1])
        e5["wind_zlow"] = full(np.array([np.interp(np.log(np.clip(q, 10, 106.5)), lz,
                                                   spd[::-1, c]) for c, q in enumerate(zl)]))
        m = {}
        rho = z["land_rho_lowest"]
        m["sw_dn_d"] = z["rad_dmean_sw_dn_sfc"]
        m["sw_abs_d"] = z["rad_dmean_sw_dn_sfc"] - z["rad_dmean_sw_up_sfc"]
        m["sw_dn_d_clr"] = z["rad_dmean_clr_sw_dn_sfc"]
        m["lw_dn_inst"] = z["rad_inst_lw_dn_sfc"]
        m["lw_dn_inst_clr"] = z["rad_inst_clr_lw_dn_sfc"]
        m["lw_dn_landforcing"] = z["land_lw_down"]
        m["lw_net_land"] = z["land_lw_net"]
        m["lw_up_land"] = z["land_lw_up"]
        m["albedo_rad"] = z["rad_sfc_albedo"]
        m["H_land"] = z["land_shflx"]
        m["G_land"] = z["land_G_soil"]
        tl = np.hypot(z["land_tau_x"], z["land_tau_y"])
        tb = np.hypot(z["bulk_tx"], z["bulk_ty"])
        m["tau_land"], m["tau_bulk"] = tl, tb
        m["ustar_land"] = np.sqrt(tl / rho)
        m["ustar_bulk_returned"] = z["bulk_us"]
        m["ustar_bulk_tau"] = np.sqrt(tb / z["bulk_rho"])
        m["z0_land"] = z["land_z0"]
        wl = np.hypot(z["bulk_u"], z["bulk_v"])
        m["wind_low"] = wl
        m["z_low"] = z["bulk_z_low"]
        m["floor_bulk"] = (wl < 0.5).astype(float)
        m["held"] = z["land_held"].astype(float)
        m["olr"] = z["rad_inst_olr"]
        e5["tau"] = np.hypot(e5["ewss"], e5["nsss"])
        e5["ustar"] = np.sqrt(e5["tau"] / rho)
        e5["wind10"] = np.hypot(e5["u10"], e5["v10"])
        e5["sw_abs_d"] = e5["ssr_d"]
        e5["albedo_d"] = 1 - e5["ssr_d"] / np.maximum(e5["ssrd_d"], 1e-3)
        if a.surfdata:
            sdat = xr.open_dataset(a.surfdata)
            tree = (sdat["PCT_NATVEG"].values / 100.0
                    * sdat["PCT_NAT_PFT"].values[1:9].sum(0) / 100.0)
            si = np.abs(sdat["LATIXY"].values[:, 0][None, :] - lat[:, None]).argmin(1)
            sj = np.abs(((sdat["LONGXY"].values[0][None, :] - lon[:, None] + 180) % 360)
                        - 180).argmin(1)
            tf = tree[si, sj]
        else:
            tf = np.full(lat.size, np.nan)
        regs = {k: land & (lat >= r[0]) & (lat <= r[1]) & (lon >= r[2]) & (lon <= r[3])
                for k, r in REG.items()}
        regs["forest>50%"] = regs["45-70N"] & (tf > 0.5)
        regs["open<10%"] = regs["45-70N"] & (tf < 0.1)
        for rn, rm in regs.items():
            w = area * fl * rm
            for k, v in m.items():
                rows.setdefault((rn, "m", k), []).append(wmean(v, w))
            for k, v in e5.items():
                rows.setdefault((rn, "e", k), []).append(wmean(v, w))
            r_ = tb[rm] / np.maximum(tl[rm], 1e-6)
            rows.setdefault((rn, "m", "tau_ratio_median"), []).append(float(np.median(r_)))
            rows.setdefault((rn, "m", "ustar_ratio_median"), []).append(
                float(np.median(np.sqrt(r_))))
            rows.setdefault((rn, "m", "bulk_tau_ge_land_frac"), []).append(
                float(np.mean(tb[rm] >= tl[rm])))
    # CONTROL (regime): replay OLR vs the run's published December rlut over the
    # same 45-70N land box (sftlf >= 50 on the CMOR grid; centre-registered files).
    import glob
    fr = sorted(glob.glob(f"{H_ROOT}/{a.run}/cmor/Amon/rlut_*_200112-200112.nc"))
    fs = sorted(glob.glob(f"{H_ROOT}/{a.run}/cmor/fx/sftlf_*.nc"))
    if fr and fs:
        r_ = xr.open_dataset(fr[0])["rlut"].isel(time=0)
        s_ = xr.open_dataset(fs[0])["sftlf"]
        mk = ((r_.lat >= 45) & (r_.lat <= 70) & (s_ >= 50)).values
        w_ = np.broadcast_to(np.cos(np.deg2rad(r_.lat.values))[:, None], mk.shape) * mk
        pub = float((r_.values * w_).sum() / w_.sum())
        rep = float(np.mean(rows[("45-70N", "m", "olr")]))
        print(f"control OLR 45-70N land: replay 00Z {rep:.1f} vs published Dec-mean rlut "
              f"{pub:.1f} W/m2 (diff {rep - pub:+.1f}; regime check, tolerance 15)")
        if abs(rep - pub) > 15:
            raise SystemExit("FATAL: replay radiation is not in the run's regime")
    lw_f = np.mean(rows[("45-70N", "m", "lw_dn_landforcing")])
    lw_r = np.mean(rows[("45-70N", "m", "lw_dn_inst")])
    print(f"control DLW 45-70N land: the run's own in-loop step-0 radiation (land forcing) "
          f"{lw_f:.1f} vs replay step-1 re-solve {lw_r:.1f} W/m2")
    print(f"{a.run} days {a.days} (00Z-state replays; '_d' = daily-mean insolation on the "
          f"00Z clouds vs ERA5 daily mean; others 00Z instants vs ERA5 00Z / the hour ending 00Z)")
    for rn in list(REG) + ["forest>50%", "open<10%"]:
        print(f"--- {rn}")
        for (r, s, k), v in rows.items():
            if r == rn:
                print(f"  {'model' if s == 'm' else 'ERA5 '} {k:24s} {np.mean(v):10.4f}  "
                      f"(days: {' '.join(f'{x:.3g}' for x in v)})")
    return 0


def zref(a):
    """How often the two-leaf canopy lifts its reference height above the level
    the driver hands it (z_ref = max(z_low, displa + 10 z0m + 2)), from the
    captured land call: z0m and z_lowest are the land's own; displa is rebuilt
    with the land's compute_aerodynamics from the run's static hc/rz0m/rd and
    the LAI that reproduces the captured z0m (egvf inverted from z0m; checked
    against the prescribed-LAI rebuild)."""
    import jax.numpy as jnp
    from nh_winter_land_state import build_driver
    from legoesm.land.canopy.stability import compute_aerodynamics, _Z0MG_BARE
    d = build_driver(a.run, a.scratch)
    lp = d.physics.land_ml_params
    lp_n = np.asarray(d._f_land).size
    hc, rz0m, rd = (np.asarray(getattr(lp, k), np.float64).reshape(-1) for k in ("hc", "rz0m", "rd"))
    lai = np.asarray(lp.LAI, np.float64).reshape(-1)
    if hc.size != lp_n:
        raise SystemExit(f"FATAL: params have {hc.size} columns, grid {lp_n}")
    z0p, dp = (np.asarray(x) for x in compute_aerodynamics(
        jnp.asarray(hc), jnp.asarray(lai), jnp.asarray(rz0m), jnp.asarray(rd)))
    for day in a.days:
        z = np.load(Path(a.out) / f"cap_{a.run}_d{day:04d}.npz", allow_pickle=True)
        lat, fl = z["lat"], z["f_land"]
        pidx = np.nonzero(fl > 0.0)[0]
        z0, zl = np.full(lat.size, np.nan), np.full(lat.size, np.nan)
        z0[pidx], zl[pidx] = z["land_z0"], z["land_z_lowest"]
        top = np.log(np.maximum(hc * rz0m, _Z0MG_BARE) / _Z0MG_BARE)
        egvf = np.where(top > 1e-6, np.log(z0 / _Z0MG_BARE) / np.where(top > 1e-6, top, 1.0),
                        (1 - np.exp(-np.clip(lai, 0, 2))) / (1 - np.exp(-2.0)))
        displa = hc * rd * egvf
        lift = displa + 10.0 * z0 + 2.0
        m = (fl > 0.5) & (lat >= 45) & (lat <= 70)
        print(f"day {day}: control z0m rebuilt from prescribed LAI vs captured, 45-70N land: "
              f"max|d| {np.nanmax(np.abs(z0p - z0)[m]):.2e} m, egvf range "
              f"[{np.nanmin(egvf[m]):.3f}, {np.nanmax(egvf[m]):.3f}]")
        for name, mm in (("45-70N land", m), ("forest (z0m>0.3 m)", m & (z0 > 0.3))):
            up = mm & (lift > zl)
            r = np.log(lift / z0) / np.log(zl / z0)
            print(f"  {name}: n={mm.sum()} lifted {up.sum()} ({up.sum() / mm.sum():.1%}); "
                  f"z_low median {np.median(zl[mm]):.1f} m; canopy lift height median "
                  f"{np.median(lift[mm]):.1f} max {np.max(lift[mm]):.1f} m, max lift/z_low "
                  f"{np.max((lift / zl)[mm]):.2f}; lift-z_low in lifted cells median "
                  f"{np.median((lift - zl)[up]) if up.any() else 0:.1f} max "
                  f"{np.max((lift - zl)[up]) if up.any() else 0:.1f} m; "
                  f"ln(zref/z0)/ln(zlow/z0) median {np.median(r[up]) if up.any() else 1:.3f} "
                  f"max {np.max(r[up]) if up.any() else 1:.3f}")
    return 0


def _put(sel, x):
    out = np.full(sel.shape, np.nan)
    out[sel] = x
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("run")
    c.add_argument("--day", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--config", default=None, help="replace the logged --config deck")
    c.add_argument("--tag", default="", help="suffix of the capture file name")
    c.add_argument("--snow-arms", action="store_true",
                   help="add radiation arms with snow (and snow+graupel) mass added to the cloud ice")
    c.add_argument("--era5-pl", default=None,
                   help="dir of ERA5 00Z e5pl_130/133_<date>.nc (era5pl/extract.sh): "
                        "adds clear-sky arms with ERA5 T, q, both, and a round-trip control")
    c.add_argument("--land-only", action="store_true",
                   help="save only the land bootstrap outputs (no atmosphere replay)")
    c.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="run_amip flags appended to the launch argv (what-if)")
    s = sp.add_parser("score")
    s.add_argument("run")
    s.add_argument("--days", type=int, nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--era5", required=True, help="nh_surface_replay_era5.sh output")
    s.add_argument("--era5-an", required=True, help="nh_winter_era5_extract.sh output")
    s.add_argument("--surfdata", default=None)
    zr = sp.add_parser("zref")
    zr.add_argument("run")
    zr.add_argument("--days", type=int, nargs="+", required=True)
    zr.add_argument("--out", required=True)
    zr.add_argument("--scratch", required=True)
    a = ap.parse_args(argv)
    return {"capture": capture, "score": score, "zref": zref}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

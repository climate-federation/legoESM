#!/usr/bin/env python
"""Zonal-mean temperature-tendency budget BY PROCESS on a production MPAS AMIP state.

Question: which process holds the upper troposphere / lower stratosphere warm
(dd_ctl: +7 K global at 250 hPa, +12 K at 40-60N and over the SH polar cap,
April 1979)?  A CMOR month cannot say: only the total tendency is carried.

How the production setup is reused, not re-derived: the run is relaunched with
its own arguments from a checkpoint (``mpas_onestep_param_grad.launch_argv``),
the five module builders in ``combined`` are wrapped so that each module's
``dT_dt`` is recorded when the FULL-radiation physics function is called
eagerly on the captured first-step state, and the dynamics term is the
residual of one un-jitted model step: (T1 - T0)/dt - sum(physics).
Instantaneous, one step: radiation / convection / turbulence / GWD are smooth
in time, the dynamics row is a single-step snapshot (gravity-wave noise averages
out in the zonal mean over ~41k columns but is NOT a monthly mean).

Reference: ERA5 monthly climatology (1979-2014, the run's calendar month) on
its own pressure levels, zonally averaged, interpolated in latitude and log-p
to every column -- so the model-minus-ERA5 row covers the WHOLE column,
including the levels above the 100 hPa CMOR lid.

Writes <out>/heating_budget_<run>_d<day>.npz and .png; prints the global-mean
K/day per process and the temperature bias at reference pressures.  Not a
test: prints, exit 1 on failure.
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpas_onestep_param_grad as H  # noqa: E402  (launch_argv, ROOT, _Captured)

ERA5 = "/work/bd1179/b309141/climateeval_input/reanalysis_ERA5/mon"
MODULES = {
    "radiation": "make_radiation_physics",
    "convection": "make_convection_physics",
    "turbulence": "make_turbulence_physics",
    "microphysics": "make_microphysics_physics",
    "gwd": "make_gwd_physics",
}
P_REF_HPA = (1000, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 5, 3)
BANDS = np.arange(-90.0, 90.1, 5.0)


def era5_zonal_clim(month: int):
    import xarray as xr
    fs = sorted(glob.glob(f"{ERA5}/ta/*.nc"))
    d = xr.open_dataset(fs[0])
    v = d["ta"].sel(time=slice("1979-01-01", "2014-12-31"))
    zm = v.isel(time=(v["time"].dt.month == month)).mean(("time", "lon")).load()
    plev = np.asarray(zm["plev"], dtype=np.float64)
    lat = np.asarray(zm["lat"], dtype=np.float64)
    arr = np.asarray(zm.transpose("plev", "lat"), dtype=np.float64)
    o = np.argsort(lat)
    return plev, lat[o], arr[:, o]


def era5_on_columns(lat_deg, p_full, month):
    """ERA5 zonal-mean T at every (column, level) by lat then log-p interpolation."""
    plev, elat, arr = era5_zonal_clim(month)
    at_lat = np.stack([np.interp(lat_deg, elat, arr[k]) for k in range(plev.size)])
    o = np.argsort(plev)
    lp, at_lat = np.log(plev[o]), at_lat[o]
    out = np.empty_like(p_full)
    for i in range(p_full.shape[0]):
        out[i] = np.interp(np.log(p_full[i]), lp, at_lat[:, i])
    return out


def band_mean(field, lat_deg, area):
    """Area-weighted mean per 5-degree band -> (nband, nlev)."""
    idx = np.clip(np.digitize(lat_deg, BANDS) - 1, 0, BANDS.size - 2)
    out = np.full((BANDS.size - 1, field.shape[1]), np.nan)
    for b in range(BANDS.size - 1):
        m = idx == b
        if m.any():
            out[b] = (field[m] * area[m, None]).sum(0) / area[m].sum()
    return out


# Cloud-optics lever arms (#1521).  Each arm multiplies named inputs of the
# production radiation backend's ALL-SKY call (the clear-sky companion call has
# no condensate and is left alone); everything else is the production call.
#   liq xk   : q_cloud and n_cloud x k -> liquid water x k at ~fixed radius
#   ice x2   : q_ice and n_ice x 2 (CAM6 ice cover depends on q_ice: combined)
#   Nc xk    : n_cloud x k -> radius x ~k^(-1/3) at fixed water
#   cover xk : CLUBB cloud_fraction_override x k (clipped to 1) with q_cloud,
#              q_ice, n_cloud, n_ice x k -> more cover at ~fixed in-cloud water
LEVER_ARMS = (("ctl", {}), ("liq x1.0", {"q_cloud": 1.0, "n_cloud": 1.0}),
              ("liq x0.5", {"q_cloud": 0.5, "n_cloud": 0.5}),
              ("liq x1.76", {"q_cloud": 1.76, "n_cloud": 1.76}),
              ("liq x2.27", {"q_cloud": 2.27, "n_cloud": 2.27}),
              ("ice x2", {"q_ice": 2.0, "n_ice": 2.0}),
              ("Nc x0.5", {"n_cloud": 0.5}), ("Nc x2", {"n_cloud": 2.0}),
              ("cover x1.07", {"cloud_fraction_override": 1.07, "q_cloud": 1.07,
                               "q_ice": 1.07, "n_cloud": 1.07, "n_ice": 1.07}))
BANDS_LEVER = {"global": (-90.0, 90.0), "30S-30N": (-30.0, 30.0)}


def scale_inputs(bound: dict, factors: dict) -> dict:
    """Copy of the backend arguments with the named inputs multiplied; the
    cloud-fraction override is clipped to [0, 1] after scaling."""
    out = dict(bound)
    for name, k in factors.items():
        if out.get(name) is None:
            raise SystemExit(f"lever input {name!r} is None in the captured call")
        v = out[name] * k
        if name == "cloud_fraction_override":
            v = v.clip(0.0, 1.0)
        out[name] = v
    return out


def gate(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")
    print(f"gate ok: {msg}")


def cloud_levers(fns, state, mesh, sig, forcing, phys_state, rad_int, real_backend,
                 n_times):
    """Diurnal-mean TOA rsut/rlut per lever arm on the captured state.

    The sun is moved by the MODEL's own radiation function: ``forcing``
    'seconds_of_day' is set to each quadrature hour and the production
    radiation physics_fn is called, so declination, orbit, distance factor and
    hour angle are the model's (no astronomy here).  The backend is wrapped to
    scale the arm's inputs on the all-sky call and record its outputs.
    """
    import inspect
    import jax
    if forcing is None or forcing.get("seconds_of_day") is None:
        raise SystemExit("captured forcing has no seconds_of_day: cannot move the sun")
    fn = fns["radiation"]
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    lat = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
    w = {b: np.where((lat >= lo) & (lat <= hi), area, 0.0) for b, (lo, hi) in BANDS_LEVER.items()}
    w = {b: v / v.sum() for b, v in w.items()}
    sig_bound = inspect.signature(real_backend)
    orig_ccp = rad_int.compute_cloud_properties
    rec = {}

    def ccp(*a, **kw):
        out = orig_ccp(*a, **kw)
        rec.setdefault("props", []).append(out)
        return out

    def run(factors, sod):
        rec.clear()

        def backend(*a, **kw):
            b = sig_bound.bind(*a, **kw).arguments
            if isinstance(b.get("T"), jax.core.Tracer):
                raise SystemExit("cloud_levers reached a traced radiation call; "
                                 "the arms must run eagerly")
            allsky = b.get("q_cloud") is not None
            if allsky:
                b = scale_inputs(b, factors)
                rec.setdefault("props", [])
            out = real_backend(**b)
            if allsky:
                rec["allsky"] = out
                rec["cf_in"] = b.get("cloud_fraction_override")
            return out
        rad_int._call_radiation_backend, rad_int.compute_cloud_properties = backend, ccp
        try:
            f = dict(forcing)
            if sod is not None:
                f["seconds_of_day"] = np.asarray(sod, dtype=np.asarray(forcing["seconds_of_day"]).dtype)
            kw = {"forcing": f} if getattr(fn, "_wants_forcing", False) else {}
            if getattr(fn, "_wants_phys_state_ro", False):
                kw["phys_state"] = phys_state
            fn(state, mesh, sig, **kw)
        finally:
            rad_int._call_radiation_backend, rad_int.compute_cloud_properties = real_backend, orig_ccp
        if "allsky" not in rec or len(rec["props"]) != 1:
            raise SystemExit(f"expected one all-sky solve with one cloud-property call, "
                             f"got {len(rec.get('props', []))}")
        o, p = rec["allsky"], rec["props"][0]
        lwp = np.asarray(p.lwp, dtype=np.float64)
        col = lwp.sum(1)
        rl = np.asarray(p.r_eff_liq, dtype=np.float64)
        return {
            "rsut": np.asarray(o.sw_flux_up[:, 0], dtype=np.float64),
            "rlut": np.asarray(o.lw_flux_up[:, 0], dtype=np.float64),
            "rsdt": np.asarray(o.sw_flux_down[:, 0], dtype=np.float64),
            "lwp": float((col * w["global"]).sum()),
            "iwp": float((np.asarray(p.iwp, dtype=np.float64).sum(1) * w["global"]).sum()),
            "reff": float(((lwp * rl).sum(1) * w["global"]).sum() / max((col * w["global"]).sum(), 1e-30)),
            "cf": np.asarray(p.cloud_fraction, dtype=np.float64),
        }

    # I1: the wrapper with no scaling at the captured time is the production call.
    real_out = {}

    def spy(*a, **kw):
        out = real_backend(*a, **kw)
        b = sig_bound.bind(*a, **kw).arguments
        if isinstance(b.get("T"), jax.core.Tracer):
            raise SystemExit("cloud_levers reached a traced radiation call; "
                             "the arms must run eagerly")
        if b.get("q_cloud") is not None:
            real_out["o"] = out
        return out
    rad_int._call_radiation_backend = spy
    kw = {"forcing": forcing} if getattr(fn, "_wants_forcing", False) else {}
    if getattr(fn, "_wants_phys_state_ro", False):
        kw["phys_state"] = phys_state
    fn(state, mesh, sig, **kw)
    rad_int._call_radiation_backend = real_backend
    r0 = run({}, None)
    d_sw = float(np.abs(r0["rsut"] - np.asarray(real_out["o"].sw_flux_up[:, 0])).max())
    d_lw = float(np.abs(r0["rlut"] - np.asarray(real_out["o"].lw_flux_up[:, 0])).max())
    gate(d_sw < 1e-6 and d_lw < 1e-6,
         f"I1 replay = production call (max|d rsut| {d_sw:.1e}, max|d rlut| {d_lw:.1e} W/m2)")

    def diurnal(factors, n):
        """Mean over n evenly spaced UTC hours (every column sees n local
        times); cloud properties do not depend on the sun, so the first
        hour's are kept and asserted identical at every hour."""
        acc = None
        for h in (np.arange(n) + 0.5) * 24.0 / n:
            r = run(factors, h * 3600.0)
            if acc is None:
                acc = dict(r)
                for k in ("rsut", "rlut", "rsdt"):
                    acc[k] = r[k] / n
            else:
                if not np.array_equal(acc["cf"], r["cf"]):
                    raise SystemExit("cloud fraction changed with the hour of day")
                for k in ("rsut", "rlut", "rsdt"):
                    acc[k] = acc[k] + r[k] / n
        return acc

    # Quadrature check: n vs 2n on ctl, and global daily-mean rsdt printed.
    a_n = diurnal({}, n_times)
    a_half = diurnal({}, n_times // 2)
    g = lambda acc, k, b="global": float((acc[k] * w[b]).sum())
    drift = abs(g(a_n, "rsut") - g(a_half, "rsut"))
    print(f"quadrature: rsdt {g(a_n, 'rsdt'):.2f} ({n_times} pts) vs {g(a_half, 'rsdt'):.2f} ({n_times // 2} pts) W/m2")
    gate(drift < 1.0, f"ctl rsut {n_times} vs {n_times // 2} points differs by {drift:.3f} W/m2 (< 1.0)")

    print(f"\n{n_times}-point diurnal mean, frozen captured state; only differences between arms are the measurement")
    hdr = (f"{'arm':>12s} {'rsut':>8s} {'d rsut':>7s} {'d 30S-30N':>9s} {'rlut':>8s} {'d rlut':>7s} "
           f"{'r_eff um':>8s} {'LWP g/m2':>8s} {'IWP g/m2':>8s} {'d cf mean':>9s}")
    print(hdr)
    base = None
    rows = {}
    for name, fac in LEVER_ARMS:
        acc = a_n if name == "ctl" else diurnal(fac, n_times)
        row = {"rsut": g(acc, "rsut"), "rsut_t": g(acc, "rsut", "30S-30N"), "rlut": g(acc, "rlut"),
               "reff": acc["reff"], "lwp": acc["lwp"], "iwp": acc["iwp"], "cf": acc["cf"]}
        rows[name] = row
        if base is None:
            base = row
        dcf = float(((row["cf"] - base["cf"]).mean(1) * w["global"]).sum())
        print(f"{name:>12s} {row['rsut']:8.3f} {row['rsut'] - base['rsut']:+7.3f} "
              f"{row['rsut_t'] - base['rsut_t']:+9.3f} {row['rlut']:8.3f} {row['rlut'] - base['rlut']:+7.3f} "
              f"{row['reff'] * 1e6:8.2f} {row['lwp'] * 1e3:8.2f} {row['iwp'] * 1e3:8.2f} {dcf:+9.5f}")
    # Hard gates on what each arm is allowed to change.
    ident = rows["liq x1.0"]
    gate(abs(ident["rsut"] - base["rsut"]) < 1e-9 and abs(ident["rlut"] - base["rlut"]) < 1e-9,
         "identity arm liq x1.0 equals ctl")
    for name, fac in LEVER_ARMS:
        r = rows[name]
        if name.startswith("liq") and name != "liq x1.0":
            k = fac["q_cloud"]
            gate(abs(r["lwp"] / base["lwp"] - k) < 0.01 * k, f"{name}: LWP ratio {r['lwp'] / base['lwp']:.4f} = {k} +-1%")
            gate(abs(r["reff"] / base["reff"] - 1.0) < 0.05, f"{name}: r_eff ratio {r['reff'] / base['reff']:.4f} within 5%")
        if name.startswith(("liq", "Nc")):
            gate(np.array_equal(r["cf"], base["cf"]), f"{name}: cloud fraction unchanged")
        if name.startswith("Nc"):
            gate(abs(r["lwp"] / base["lwp"] - 1.0) < 1e-9, f"{name}: LWP unchanged")
        if name.startswith("ice"):
            k = fac["q_ice"]
            gate(abs(r["iwp"] / base["iwp"] - k) < 0.01 * k, f"{name}: IWP ratio {r['iwp'] / base['iwp']:.4f} = {k} +-1%")
    ln_r = np.log(rows["Nc x0.5"]["reff"] / rows["Nc x2"]["reff"])
    print(f"\nd rsut / d ln r_eff (Nc x0.5 vs x2): "
          f"{(rows['Nc x0.5']['rsut'] - rows['Nc x2']['rsut']) / ln_r:+.2f} W/m2 per ln unit "
          f"(radius moved {rows['Nc x2']['reff'] * 1e6:.2f} -> {rows['Nc x0.5']['reff'] * 1e6:.2f} um)")
    return 0

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dd_ctl")
    ap.add_argument("--day", type=int, default=90)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rad-detail", action="store_true",
                    help="LW/SW split of the top layers on the model T and on ERA5 T, "
                         "plus the ozone the radiation was given and the TOA fluxes")
    ap.add_argument("--jacobian", nargs="*", type=int, default=None,
                    metavar="LEVEL",
                    help="radiative Jacobian: perturb each listed model level by "
                         "--jac-delta K with the clouds FROZEN and record the net "
                         "heating response in every level.  Separates the local "
                         "Planck damping from the part supplied by neighbouring "
                         "layers, which the whole-profile swap cannot.  Default "
                         "levels are the UTLS band.")
    ap.add_argument("--jac-delta", type=float, default=1.0,
                    help="perturbation amplitude [K]; the response is reported per K "
                         "and is also run at half amplitude as a linearity check")
    ap.add_argument("--gwd-only", action="store_true",
                    help="only the gravity-wave-drag momentum tendency: deposition per layer, "
                         "global shares and the 5-degree-band profiles (saved to npz)")
    ap.add_argument("--cloud-levers", action="store_true",
                    help="#1521: re-solve the captured production radiation call with "
                         "liquid water, ice water or droplet number scaled (LEVER_ARMS), "
                         "diurnally averaged; prints rsut/rlut per arm")
    ap.add_argument("--n-times", type=int, default=8)
    args = ap.parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.out or (H.ROOT / "_tools" / "heating_budget"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / f"_launch_{args.run}"
    scratch.mkdir(exist_ok=True)
    argv_run = H.launch_argv(args.run, args.day, scratch)

    import jax
    import jax.numpy as jnp
    import legoesm.atmosphere.physics.combined as combined
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel)

    rec: dict[str, np.ndarray] = {}
    fns: dict = {}
    rad_calls: list = []

    def spy_builder(name, real):
        @functools.wraps(real)
        def build(*a, **kw):
            fn = real(*a, **kw)

            @functools.wraps(fn)          # keeps _wants_forcing / _wants_phys_state_ro
            def wrapped(*fa, **fkw):
                res = fn(*fa, **fkw)
                t = res if hasattr(res, "dT_dt") else res[0]   # tendencies are NamedTuples
                if not isinstance(t.dT_dt.data, jax.core.Tracer):   # setup traces under jit
                    rec[name] = np.asarray(t.dT_dt.data, dtype=np.float64)
                return res
            fns[name] = wrapped
            return wrapped
        return build

    for name, attr in MODULES.items():
        setattr(combined, attr, spy_builder(name, getattr(combined, attr)))
    import legoesm.atmosphere.physics.radiation.integration as rad_int
    real_backend = rad_int._call_radiation_backend

    def spy_backend(*a, **kw):
        out = real_backend(*a, **kw)
        if isinstance(out.lw_heating_rate, jax.core.Tracer):          # setup traces under jit
            return out
        rad_calls.append((np.asarray(out.lw_heating_rate, dtype=np.float64),
                          np.asarray(out.sw_heating_rate, dtype=np.float64),
                          np.asarray(out.lw_flux_up[:, 0], dtype=np.float64),
                          np.asarray(out.sw_flux_up[:, 0], dtype=np.float64)))
        return out
    rad_int._call_radiation_backend = spy_backend
    real_make_physics = combined.make_physics
    builds: list[tuple] = []

    def spy_make_physics(config, *a, **kw):
        builds.append((config, a, dict(kw)))
        return real_make_physics(config, *a, **kw)
    cap: dict = {}

    def spy_step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        cap.update(model=self, state=state, dt=dt, physics_fn=physics_fn,
                   forcing=forcing, phys_state=phys_state)
        raise H._Captured()
    combined.make_physics = spy_make_physics
    MPASPrimitiveEquationModel.step = spy_step
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run"))
    import run_amip
    try:
        run_amip.main(argv_run)
    except H._Captured:
        pass
    else:
        raise SystemExit("driver returned without ever calling model.step")
    full = [b for b in builds if b[2].get("need_rad", True) is not False]
    if not full:
        raise SystemExit("no full-radiation make_physics call captured")
    phys_cfg, pargs, pkw = full[-1]
    model, state, DT = cap["model"], cap["state"], float(cap["dt"])
    forcing, phys_state = cap["forcing"], cap["phys_state"]
    mesh, sig = model.mesh, model.sigma_coord
    print(f"captured dt={DT} builds={len(builds)} ({time.time() - t0:.0f}s)", flush=True)

    pf_full = real_make_physics(phys_cfg, *pargs, **pkw)
    T0 = np.asarray(state.T.data, dtype=np.float64)
    if args.cloud_levers:
        return cloud_levers(fns, state, mesh, sig, forcing, phys_state, rad_int,
                            real_backend, n_times=args.n_times)
    if args.gwd_only:
        from legoesm import constants
        fn = fns["gwd"]
        kw = {"phys_state": phys_state}
        if getattr(fn, "_wants_forcing", False):
            kw["forcing"] = forcing
        res = fn(state, mesh, sig, **kw)
        t = res if hasattr(res, "dT_dt") else res[0]
        du = np.asarray(t.du_dt.data, dtype=np.float64).reshape(-1, T0.shape[1]) * 86400.0
        p_s = np.asarray(state.p_s.data, dtype=np.float64)
        dsig = np.asarray(sig.dsigma, dtype=np.float64)[None, :]
        on_edges = du.shape[0] != T0.shape[0]
        print(f"GWD tendency on {'edges' if on_edges else 'cells'}: {du.shape}")
        if on_edges:                                        # edge-normal tendency (MPAS bridge)
            from legoesm.grids.voronoi import reconstruct_cell_velocity
            ue, vn = reconstruct_cell_velocity(jnp.asarray(du), mesh)
            du_east, dv_north = np.asarray(ue, dtype=np.float64), np.asarray(vn, dtype=np.float64)
        else:
            du_east, dv_north = du, np.zeros_like(du)
        lat_deg = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
        area = np.asarray(mesh.areaCell, dtype=np.float64)
        dp = p_s[:, None] * dsig
        du = du_east                                        # zonal force from here on
        dep = np.hypot(du_east, dv_north) * dp / constants.g / 86400.0   # |momentum sink| per layer [Pa]
        w = area / area.sum()
        tot = dep.sum(1)
        share_top1 = (dep[:, 0] * w).sum() / max((tot * w).sum(), 1e-30)
        share_top2 = (dep[:, :2].sum(1) * w).sum() / max((tot * w).sum(), 1e-30)
        print(f"\nGWD deposition: global-mean column sink {(tot * w).sum() * 1e3:.4f} mPa; "
              f"share in top layer {share_top1:.3f}, top two {share_top2:.3f}")
        zdu, zdep = band_mean(du, lat_deg, area), band_mean(dep, lat_deg, area)
        p_mid = band_mean(p_s[:, None] * np.asarray(sig.sigma_full, dtype=np.float64)[None, :],
                          lat_deg, area) / 100.0
        np.savez(out_dir / f"gwd_{args.run}_d{args.day:04d}.npz", bands=BANDS,
                 du_dt=zdu, deposition=zdep, p_hpa=p_mid, dt=DT)
        for lo, hi, name in ((-90, -60, "60-90S"), (60, 90, "60-90N")):
            sel = (BANDS[:-1] >= lo) & (BANDS[:-1] < hi)
            cw = np.cos(np.deg2rad(0.5 * (BANDS[:-1] + BANDS[1:])))[sel]
            cw = cw / cw.sum()
            prof, pp = (zdu[sel] * cw[:, None]).sum(0), (p_mid[sel] * cw[:, None]).sum(0)
            print(f"\n{name} GWD zonal du/dt [m/s/day] by level (p hPa):")
            print("  " + " ".join(f"{pp[k]:6.1f}:{prof[k]:+6.2f}" for k in range(min(14, prof.size))))
        return 0
    rec.clear()
    pf_full(state, mesh, sig, phys_state=phys_state, forcing=forcing)
    missing = [m for m in MODULES if m not in rec]
    if missing:
        raise SystemExit(f"modules never called: {missing}")
    phys = {k: v.reshape(T0.shape) for k, v in rec.items()}
    print(f"physics eager ({time.time() - t0:.0f}s)", flush=True)

    target_mass = model._target_mass
    if model.config.fix_mass and model.config.anchor_mass_to_initial and target_mass is None:
        target_mass = model.compute_mass(state)
    step_raw = type(model)._step_jit.__wrapped__
    new, _pout, _sfc, _led = step_raw(model, state, DT, pf_full, target_mass,
                                      forcing, phys_state)
    T1 = np.asarray(new.T.data, dtype=np.float64)
    total = (T1 - T0) / DT
    dyn = total - sum(phys.values())
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import vertical_del4_T_tendency
    nu4 = float(model.config.nu_vert4_T)
    vfilt = np.asarray(vertical_del4_T_tendency(jnp.asarray(T0), nu4, jnp.asarray(sig.dsigma)),
                       dtype=np.float64) if nu4 > 0 else np.zeros_like(T0)
    dyn = dyn - vfilt          # dynamics row = advection + adiabatic + horizontal mixing
    print(f"one step eager ({time.time() - t0:.0f}s), nu_vert4_T={nu4:g}", flush=True)

    # Split the dynamics row into the three terms the dycore itself assembles
    # (horizontal advection, the theta-form vertical term, the surface-pressure
    # adiabatic term).  These are the dycore's own arrays, not a re-derivation:
    # mpas_hydrostatic_tendencies returns them on request, which no production
    # path does.  Called WITHOUT physics so the terms are the dry dynamics only.
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        mpas_hydrostatic_tendencies)
    _tend_full, _terms = mpas_hydrostatic_tendencies(
        state, mesh, sig, model.config, None, DT, return_thermo_terms=True)
    dyn_terms = {
        "dyn_horiz_adv": np.asarray(_terms.horiz_adv, dtype=np.float64),
        "dyn_horiz_diff": np.asarray(_terms.horiz_diff, dtype=np.float64),
        "dyn_vert_adv": np.asarray(_terms.vert_adv, dtype=np.float64),
        "dyn_adiabatic": np.asarray(_terms.adiabatic_ps, dtype=np.float64),
    }
    # Coordinate vertical velocity as a descent rate in pressure units, so the
    # cap mean can be read against the winter residual circulation's ~0.5-1
    # hPa/day (GLM's closure test): if the model descends several times faster
    # than that, the vertical warming is a circulation problem; if it descends
    # at a physical rate while the tendency is 1-2 K/day, the vertical operator
    # itself is.  sigma_dot is at interfaces; average to full levels.
    _sd = np.asarray(_terms.sigma_dot, dtype=np.float64)
    if _sd.shape[1] == T0.shape[1] + 1:
        _sd = 0.5 * (_sd[:, :-1] + _sd[:, 1:])
    # The pressure velocity is NOT p_s*sigma_dot alone: in a sigma coordinate
    # omega = p_s*sigma_dot + sigma*(dp_s/dt + u.grad p_s).  Both reviewers
    # flagged that dropping the surface-pressure terms lets a spinning-down
    # global p_s bump read as "descent", which is exactly the failure mode in a
    # cold start.  The material dp_s/dt the dycore reports already contains the
    # advective part (it is the flux-form tendency), so it is added directly.
    _p_s = np.asarray(state.p_s.data, dtype=np.float64)
    _sig = np.asarray(sig.sigma_full, dtype=np.float64)[None, :]
    _dps = np.asarray(_tend_full.dp_s_dt.data, dtype=np.float64)[:, None]
    _omega = _sd * _p_s[:, None] + _sig * _dps
    extra_rows = {"omega_hpa_day": _omega * 86400.0 / 100.0,
                  # the piece that was being reported before, kept so the two
                  # can be compared directly
                  "omega_sigmadot_only_hpa_day": _sd * _p_s[:, None] * 86400.0 / 100.0,
                  "omega_ps_term_hpa_day": _sig * _dps * 86400.0 / 100.0}
    # NOT in dyn_terms: these are not K/day tendencies and must not enter the
    # closure sum below.
    print(f"horizontal T diffusion K_h = {float(model.config.K_h):.3e} m2/s "
          f"({'ACTIVE' if float(model.config.K_h) > 0 else 'off'})", flush=True)
    _closure = dyn - sum(dyn_terms.values())
    print(f"dynamics split closure: max |dyn - (horiz+vert+adiab)| = "
          f"{np.abs(_closure).max() * 86400.0:.3f} K/day "
          f"(the remainder is the dycore's mass fixer and the p_s tendency's "
          f"effect on T, which the residual row keeps)", flush=True)
    dyn_terms["dyn_residual"] = _closure

    lat_deg = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    p_s = np.asarray(state.p_s.data, dtype=np.float64)
    p_full = p_s[:, None] * np.asarray(sig.sigma_full, dtype=np.float64)[None, :]
    run_cfg = json.load(open(H.ROOT / args.run / "experiment_config.json"))
    month = int(((int(run_cfg.get("start_month", 1)) - 1 + args.day // 30) % 12) + 1)
    Te = era5_on_columns(lat_deg, p_full, month)

    # The same three dycore terms evaluated on ERA5's temperature field with the
    # model's own winds.  The model's advection warming the cap could be a
    # RESPONSE to the warm anomaly already there (a warm cap weakens the inflow
    # gradient) rather than its cause; on a reanalysis temperature field that
    # anomaly is absent, so a warming that survives is the flow and the operator,
    # not the anomaly.  Winds, pressure and the mesh are the model's throughout.
    _state_era5 = state._replace(
        T=state.T.replace(data=jnp.asarray(Te, dtype=state.T.data.dtype)))
    _, _terms_e = mpas_hydrostatic_tendencies(
        _state_era5, mesh, sig, model.config, None, DT, return_thermo_terms=True)
    dyn_terms.update({
        "era5T_horiz_adv": np.asarray(_terms_e.horiz_adv, dtype=np.float64),
        "era5T_horiz_diff": np.asarray(_terms_e.horiz_diff, dtype=np.float64),
        "era5T_vert_adv": np.asarray(_terms_e.vert_adv, dtype=np.float64),
        "era5T_adiabatic": np.asarray(_terms_e.adiabatic_ps, dtype=np.float64),
    })
    w = area / area.sum()
    if args.rad_detail:
        # Radiation rebuilds cloud properties from the temperature it is handed
        # (compute_cloud_properties(T=T, ...)), and cloud fraction depends on T
        # through saturation, so swapping the temperature ALSO swaps the clouds
        # and the difference is not a pure temperature response (codex).  Record
        # the cloud properties the model-temperature call produces and replay
        # them, in call order, into the ERA5-temperature call.
        import legoesm.atmosphere.physics.radiation.integration as _radint
        _cloud_tape: list = []
        _cloud_mode = {"m": "record"}
        _orig_ccp = _radint.compute_cloud_properties

        def _ccp(*a, **kw):
            if _cloud_mode["m"] == "replay" and _cloud_tape:
                return _cloud_tape.pop(0)
            out = _orig_ccp(*a, **kw)
            if _cloud_mode["m"] == "record":
                _cloud_tape.append(out)
            return out

        _radint.compute_cloud_properties = _ccp

        def rad_on(T_arr):
            fn = fns["radiation"]
            kw = {"forcing": forcing} if getattr(fn, "_wants_forcing", False) else {}
            if getattr(fn, "_wants_phys_state_ro", False):
                kw["phys_state"] = phys_state
            rad_calls.clear()
            st = state._replace(T=state.T.replace(data=jnp.asarray(T_arr, dtype=state.T.data.dtype)))
            fn(st, mesh, sig, **kw)
            lw, sw, olr, rsu = rad_calls[0]            # first call = all-sky
            print(f"    TOA global mean: rlut {float((olr * w).sum()):.2f}  rsut {float((rsu * w).sum()):.2f} W/m2")
            return lw.reshape(T0.shape) * 86400.0, sw.reshape(T0.shape) * 86400.0
        _cloud_mode["m"] = "record"
        lw_m, sw_m = rad_on(T0)
        _recorded = list(_cloud_tape)

        if args.jacobian is not None:
            levels = args.jacobian or [9, 10, 11, 12, 13, 14]
            cap = lat_deg >= 60.0
            wc = area[cap] / area[cap].sum()
            base = (lw_m + sw_m)
            print(f"\nradiative Jacobian, clouds frozen, {args.jac_delta:g} K "
                  f"perturbation, 60-90N area-weighted [K/day per K]")
            print("  perturbed | response in each level (rows = perturbed, cols = level)")
            hdr = "   p_pert |" + "".join(f"{np.average(p_full[cap, L], weights=wc)/100:8.0f}"
                                          for L in levels)
            print(hdr)
            jac = np.zeros((len(levels), len(levels)))
            for a_i, L in enumerate(levels):
                for amp, store in ((args.jac_delta, True), (0.5 * args.jac_delta, False)):
                    Tp = T0.copy()
                    Tp[:, L] += amp
                    _cloud_mode["m"] = "replay"
                    _cloud_tape[:] = list(_recorded)
                    lw_p, sw_p = rad_on(Tp)
                    resp = ((lw_p + sw_p) - base) / amp
                    if store:
                        row = [float(np.average(resp[cap, M], weights=wc)) for M in levels]
                        jac[a_i] = row
                    else:
                        half = [float(np.average(resp[cap, M], weights=wc)) for M in levels]
                print(f"{np.average(p_full[cap, L], weights=wc)/100:9.0f} |"
                      + "".join(f"{v:8.4f}" for v in jac[a_i])
                      + f"   (half-amplitude diagonal {half[a_i]:+.4f})")
            diag = np.diag(jac)
            print(f"\n  local damping -J_ii: " + " ".join(f"{-v:.4f}" for v in diag))
            print(f"  column sum per perturbed layer (local + what neighbours return): "
                  + " ".join(f"{v:.4f}" for v in jac.sum(axis=1)))
            np.savez(out_dir / f"rad_jacobian_{args.run}_d{args.day:04d}.npz",
                     jac=jac, levels=np.asarray(levels),
                     p_hpa=np.asarray([np.average(p_full[cap, L], weights=wc) / 100
                                       for L in levels]))
            _cloud_mode["m"] = "off"

        # clouds FROZEN at the model-temperature values: a pure temperature response
        _cloud_mode["m"] = "replay"
        _cloud_tape[:] = list(_recorded)
        lw_e, sw_e = rad_on(Te)
        # and again with clouds free, so the cloud share of the response is visible
        _cloud_mode["m"] = "off"
        lw_e_freecld, sw_e_freecld = rad_on(Te)
        _radint.compute_cloud_properties = _orig_ccp
        print(f"    cloud tape: {len(_recorded)} call(s) recorded and replayed")
        o3 = np.asarray(forcing["o3_vmr"], dtype=np.float64).reshape(T0.shape) * 1e6
        rad_bands = {"lw_model": lw_m, "sw_model": sw_m, "lw_era5": lw_e, "sw_era5": sw_e,
                     "lw_era5_freecld": lw_e_freecld, "sw_era5_freecld": sw_e_freecld,
                     "o3_ppmv": o3}
        gmp = (p_full * w[:, None]).sum(0) / 100.0
        print("\nradiation detail, global mean, K/day (model T | ERA5 T in the same columns), o3 ppmv given to radiation")
        print("  p[hPa]  T_mod  T_era   LW_m   SW_m  net_m |  LW_e   SW_e  net_e | o3")
        for k in list(range(0, 5)) + [int(np.argmin(np.abs(gmp - x))) for x in (150, 250)]:
            g = lambda f: float((f[:, k] * w).sum())  # noqa: E731
            print(f"  {gmp[k]:6.1f} {g(T0):6.1f} {g(Te):6.1f} {g(lw_m):+6.2f} {g(sw_m):+6.2f} {g(lw_m + sw_m):+6.2f} |"
                  f" {g(lw_e):+6.2f} {g(sw_e):+6.2f} {g(lw_e + sw_e):+6.2f} | {g(o3):5.2f}")

    rows = {**{k: v * 86400.0 for k, v in phys.items()},
            "vert_del4": vfilt * 86400.0, "dynamics": dyn * 86400.0,
            **{k: v * 86400.0 for k, v in dyn_terms.items()},
            **extra_rows,
            "total": total * 86400.0}
    zm = {k: band_mean(v, lat_deg, area) for k, v in rows.items()}
    zm["T_model"] = band_mean(T0, lat_deg, area)
    zm["T_era5"] = band_mean(Te, lat_deg, area)
    zm["p_hpa"] = band_mean(p_full, lat_deg, area) / 100.0
    if args.rad_detail:
        zm.update({k: band_mean(v, lat_deg, area) for k, v in rad_bands.items()})
    gm = {k: (v * w[:, None]).sum(0) for k, v in rows.items()}
    gm_p = (p_full * w[:, None]).sum(0) / 100.0
    gm_bias = ((T0 - Te) * w[:, None]).sum(0)
    lp = np.log(gm_p)
    order = np.argsort(lp)
    print(f"\n{args.run} day {args.day} (ERA5 month {month}): global-mean K/day per process")
    hdr = "   p[hPa]   T-ERA5 " + " ".join(f"{k:>12s}" for k in rows)
    print(hdr)
    for pr in P_REF_HPA:
        x = np.log(pr)
        if x < lp[order[0]] or x > lp[order[-1]]:
            continue
        line = f"{pr:8.0f} {np.interp(x, lp[order], gm_bias[order]):+8.2f} "
        line += " ".join(f"{np.interp(x, lp[order], gm[k][order]):+12.3f}" for k in rows)
        print(line)
    np.savez(out_dir / f"heating_budget_{args.run}_d{args.day:04d}.npz",
             bands=BANDS, month=month, dt=DT, **zm)
    # Per-cell descent rate and terrain, so a cap-mean descent can be checked
    # for whether it is cap-WIDE or concentrated near topography — a band mean
    # cannot tell those apart, and they implicate different causes.
    np.savez(out_dir / f"omega_cells_{args.run}_d{args.day:04d}.npz",
             omega_hpa_day=extra_rows["omega_hpa_day"],
             omega_sigmadot_only=extra_rows["omega_sigmadot_only_hpa_day"],
             omega_ps_term=extra_rows["omega_ps_term_hpa_day"], lat_deg=lat_deg,
             lon_deg=np.rad2deg(np.asarray(mesh.lonCell, dtype=np.float64)),
             area=area, p_full=p_full,
             phis=np.asarray(state.phis.data, dtype=np.float64))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(rows) + ["T_model - T_era5"]
    zm["T_model - T_era5"] = zm["T_model"] - zm["T_era5"]
    ncol_p = 3
    nrow_p = -(-len(keys) // ncol_p)      # grew when the dynamics split landed
    fig, axs = plt.subplots(nrow_p, ncol_p, figsize=(16, 3.7 * nrow_p),
                            constrained_layout=True)
    latc = 0.5 * (BANDS[1:] + BANDS[:-1])
    pc = np.nanmean(zm["p_hpa"], axis=0)
    for a in axs.ravel()[len(keys):]:
        a.axis("off")
    for a, k in zip(axs.ravel(), keys):
        lim = 12.0 if k.startswith("T_model") else max(0.5, np.nanpercentile(np.abs(zm[k]), 98))
        m = a.contourf(latc, pc, zm[k].T, np.linspace(-lim, lim, 25), cmap="RdBu_r",
                       extend="both")
        a.set_yscale("log")
        a.invert_yaxis()
        a.set_title(k + (" [K]" if k.startswith("T_model") else " [K/day]"), fontsize=9)
        fig.colorbar(m, ax=a, shrink=0.8)
    fig.suptitle(f"{args.run} day {args.day}: zonal-mean dT/dt by process (one step) "
                 f"and T bias vs ERA5 month {month}", fontsize=10)
    png = out_dir / f"heating_budget_{args.run}_d{args.day:04d}.png"
    fig.savefig(png, dpi=110)
    print(f"wrote {png} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

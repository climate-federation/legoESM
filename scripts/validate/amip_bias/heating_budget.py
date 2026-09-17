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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dd_ctl")
    ap.add_argument("--day", type=int, default=90)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rad-detail", action="store_true",
                    help="LW/SW split of the top layers on the model T and on ERA5 T, "
                         "plus the ozone the radiation was given and the TOA fluxes")
    ap.add_argument("--gwd-only", action="store_true",
                    help="only the gravity-wave-drag momentum tendency: deposition per layer, "
                         "global shares and the 5-degree-band profiles (saved to npz)")
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
    _, _terms = mpas_hydrostatic_tendencies(
        state, mesh, sig, model.config, None, DT, return_thermo_terms=True)
    dyn_terms = {
        "dyn_horiz_adv": np.asarray(_terms.horiz_adv, dtype=np.float64),
        "dyn_horiz_diff": np.asarray(_terms.horiz_diff, dtype=np.float64),
        "dyn_vert_adv": np.asarray(_terms.vert_adv, dtype=np.float64),
        "dyn_adiabatic": np.asarray(_terms.adiabatic_ps, dtype=np.float64),
    }
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
    w = area / area.sum()
    if args.rad_detail:
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
        lw_m, sw_m = rad_on(T0)
        lw_e, sw_e = rad_on(Te)
        o3 = np.asarray(forcing["o3_vmr"], dtype=np.float64).reshape(T0.shape) * 1e6
        rad_bands = {"lw_model": lw_m, "sw_model": sw_m, "lw_era5": lw_e, "sw_era5": sw_e,
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

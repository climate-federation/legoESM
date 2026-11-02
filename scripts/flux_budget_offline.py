"""Offline surface heat budget of the legoESM JRA55-do forcing, component by component.

legoESM (jra55_to_atm_surface + ocean_tile_response with the CouplerConfig of run_omip.py) vs an
independent Large & Yeager (2004) reference (CORE-II / OMIP-2: albedo 0.066, emissivity 1,
rho_a 1.22, T/q shifted 2 m -> 10 m).  Options: --sst-run (seasonal SST from a run's restarts),
--sea-ice-clim (flux x (1 - f_ice) as the scan block), --ref-hfds (compare with a CMIP hfds).
"""
import argparse
import ast
import importlib
import inspect
import re
import time
from pathlib import Path

import numpy as np

SIGMA = 5.67e-8
KAPPA, GRAV = 0.4, 9.81
R_E = 6.371e6


def _psi_m(z):
    x = np.sqrt(np.sqrt(np.maximum(1 - 16 * z, 1e-12)))
    un = 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + np.pi / 2
    return np.where(z < 0, un, -5 * z)


def _psi_h(z):
    x = np.sqrt(np.sqrt(np.maximum(1 - 16 * z, 1e-12)))
    return np.where(z < 0, 2 * np.log((1 + x * x) / 2), -5 * z)


def ly04(u, v, t2, q2, sst, zt=2.0, n_iter=5):
    rho_a, cp_a, L_v = 1.22, 1000.5, 2.5e6
    U = np.maximum(np.hypot(u, v), 0.5)
    qs = 0.98 * 640380.0 / rho_a * np.exp(-5107.4 / sst)
    th = t2 + 0.0098 * zt
    th10, q10 = th.copy(), q2.copy()
    cdn = lambda W: (2.7 / W + 0.142 + 0.0764 * W) / 1e3  # noqa: E731
    cd_n = cdn(U); rt = np.sqrt(cd_n)
    stab = (th - sst > 0).astype(float)
    cd, ch, ce = cd_n.copy(), (18.0 * stab + 32.7 * (1 - stab)) * rt / 1e3, 34.6 * rt / 1e3
    for _ in range(n_iter):
        cd_rt = np.sqrt(cd)
        ustar = cd_rt * U
        tstar = ch / cd_rt * (th10 - sst)
        qstar = ce / cd_rt * (q10 - qs)
        tv = th10 * (1 + 0.608 * q10)
        bstar = GRAV * (tstar / tv + qstar / (q10 + 1 / 0.608))
        z10 = np.clip(KAPPA * bstar * 10.0 / ustar ** 2, -10, 10)
        ztt = np.clip(KAPPA * bstar * zt / ustar ** 2, -10, 10)
        shift = np.log(zt / 10.0) - _psi_h(ztt) + _psi_h(z10)
        th10 = th - tstar / KAPPA * shift
        q10 = q2 - qstar / KAPPA * shift
        U10N = np.maximum(U / (1 + rt * (-_psi_m(z10)) / KAPPA), 0.5)
        cd_n = cdn(U10N); rt = np.sqrt(cd_n)
        stab = (z10 > 0).astype(float)
        ch_n, ce_n = (18.0 * stab + 32.7 * (1 - stab)) * rt / 1e3, 34.6 * rt / 1e3
        cd = (rt / (1 + rt * (-_psi_m(z10)) / KAPPA)) ** 2
        xx = -_psi_h(z10) / KAPPA
        ch = ch_n / (1 + ch_n * xx / rt) * np.sqrt(cd / cd_n)
        ce = ce_n / (1 + ce_n * xx / rt) * np.sqrt(cd / cd_n)
    SH = -rho_a * cp_a * ch * U * (th10 - sst)
    LH = -rho_a * L_v * ce * U * (q10 - qs)
    return SH, LH


def coupler_config(run_omip):
    src = Path(run_omip).read_text()
    tree = ast.parse(src)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "id", getattr(n.func, "attr", "")) == "CouplerConfig")
    kw, skipped = {}, []
    for k in call.keywords:
        try:
            kw[k.arg] = ast.literal_eval(k.value)
        except Exception:  # noqa: BLE001
            skipped.append(f"{k.arg}={ast.get_source_segment(src, k.value)}")
    CC = None
    for mod in ("legoesm.coupler.coupler", "legoesm.coupler.config", "legoesm.coupler"):
        try:
            CC = getattr(importlib.import_module(mod), "CouplerConfig")
            break
        except Exception:  # noqa: BLE001
            pass
    if CC is None:
        raise SystemExit("CouplerConfig not importable")
    cfg = CC(**kw)
    print(f"CouplerConfig from {run_omip}: {kw}")
    if skipped:
        print(f"  NOT literal (defaults used instead): {skipped}")
    lines = [l.strip() for l in src.splitlines() if re.search(r'"(lat|lon)_2d"\s*\]?\s*[:=]|(lat|lon)_2d\s*=', l)]
    return cfg, " ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--sea-ice-clim", default=None)
    ap.add_argument("--ref-hfds", default=None)
    ap.add_argument("--ref-years", type=int, default=20)
    ap.add_argument("--sst-obs", default=None)
    ap.add_argument("--sst-run", default=None)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--run-omip", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--ref-year", type=int, default=1900)
    ap.add_argument("--lat-units", choices=["auto", "deg", "rad"], default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import jax.numpy as jnp
    import xarray as xr
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.forcing.jra55_do import jra55_to_atm_surface, load_jra55_slice

    cfg, latsrc = coupler_config(a.run_omip)
    g = np.load(a.grid)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    rad_in_grid = np.abs(lat).max() <= np.pi / 2 + 1e-6
    latd, lond = (np.rad2deg(lat), np.rad2deg(lon)) if rad_in_grid else (lat, lon)
    use_rad = (a.lat_units == "rad") or (a.lat_units == "auto" and ("rad" in latsrc.lower() or "grid.lat" in latsrc))
    LAd, LOd = np.meshgrid(latd, lond, indexing="ij")
    lat2d = np.deg2rad(LAd) if use_rad else LAd
    lon2d = np.deg2rad(LOd) if use_rad else LOd

    r = np.load(a.restart)
    ocean = r["land_mask"] > 0.5
    sstC = np.where(ocean, np.asarray(r["T"], float)[..., 0], 10.0)
    sstK = jnp.asarray(sstC + 273.15)
    snaps = None
    if a.sst_run:
        import glob
        fs = sorted(glob.glob(f"{a.sst_run}/restart_day*.npz"))
        days = np.array([float(Path(f).stem.replace("restart_day", "")) for f in fs])
        last = days.max()
        sel = [(d, f) for d, f in zip(days, fs) if d > last - 365 + 1e-6]
        snaps = sorted(((d % 365.0, np.where(ocean, np.asarray(np.load(f)["T"], float)[..., 0], 10.0)) for d, f in sel),
                       key=lambda t: t[0])
        print(f"  SST from {len(snaps)} snapshots of {a.sst_run} (days {[int(d) for d, _ in sel]})")

    obs_m = None
    if a.sst_obs:
        from scipy.interpolate import RegularGridInterpolator as _RGI2
        _o = xr.open_dataset(a.sst_obs)
        _v = _o["tos"] if "tos" in _o else _o[next(iter(_o.data_vars))]
        _t = _v["time"].dt
        _ms = [_v.sel(time=(_t.year == (1991 if m <= 4 else 1990)) & (_t.month == m)).squeeze("time", drop=True) for m in range(1, 13)]
        _la_n = next(d for d in _ms[0].dims if d.lower().startswith("lat"))
        _lo_n = next(d for d in _ms[0].dims if d.lower().startswith("lon"))
        _la = np.asarray(_ms[0][_la_n], float); _lo = np.mod(np.asarray(_ms[0][_lo_n], float), 360.0)
        _oi, _oj = np.argsort(_la), np.argsort(_lo)
        _la, _lo = _la[_oi], _lo[_oj]
        _lop = np.concatenate([[_lo[-1] - 360], _lo, [_lo[0] + 360]])
        _qq = np.stack([np.clip(LAd, _la.min(), _la.max()), np.mod(LOd, 360.0)], axis=-1)
        obs_m = []
        for mm in _ms:
            v = np.asarray(mm.transpose(_la_n, _lo_n).values, float)[_oi][:, _oj]
            if np.nanmedian(v) > 100:
                v = v - 273.15
            v = np.maximum(np.nan_to_num(v, nan=-1.8), -1.8)
            vp = np.concatenate([v[:, -1:], v, v[:, :1]], axis=1)
            obs_m.append(_RGI2((_la, _lop), vp, bounds_error=False, fill_value=-1.8)(_qq))
        obs_m = np.stack(obs_m)
        print(f"  observed SST from {a.sst_obs}: RYF 1990/91 months, range {obs_m.min():.1f}..{obs_m.max():.1f} C")

    def _run_at(doy):
        ds = np.array([d for d, _ in snaps])
        ext_d = np.concatenate([[ds[-1] - 365.0], ds, [ds[0] + 365.0]])
        ext_f = [snaps[-1][1]] + [f for _, f in snaps] + [snaps[0][1]]
        i = min(max(int(np.searchsorted(ext_d, doy % 365.0, side="right") - 1), 0), len(ext_d) - 2)
        w1 = (doy % 365.0 - ext_d[i]) / (ext_d[i + 1] - ext_d[i])
        return (1 - w1) * ext_f[i] + w1 * ext_f[i + 1]

    def sst_at(doy):
        if obs_m is not None:
            pos = (np.mod(doy, 365.0) + 1.0 - 15.5) / (365.0 / 12.0)
            m0 = int(np.floor(pos)); fr = pos - m0
            return np.where(ocean, (1 - fr) * obs_m[m0 % 12] + fr * obs_m[(m0 + 1) % 12], 10.0)
        if snaps is None:
            return sstC
        ds = np.array([d for d, _ in snaps])
        ext_d = np.concatenate([[ds[-1] - 365.0], ds, [ds[0] + 365.0]])
        ext_f = [snaps[-1][1]] + [f for _, f in snaps] + [snaps[0][1]]
        i = int(np.searchsorted(ext_d, doy % 365.0, side="right") - 1)
        i = min(max(i, 0), len(ext_d) - 2)
        w1 = (doy % 365.0 - ext_d[i]) / (ext_d[i + 1] - ext_d[i])
        return (1 - w1) * ext_f[i] + w1 * ext_f[i + 1]
    zer = jnp.zeros_like(sstK)
    lat_e = np.concatenate([[-90.0], 0.5 * (latd[1:] + latd[:-1]), [90.0]])
    area = (R_E ** 2 * (np.sin(np.deg2rad(lat_e[1:])) - np.sin(np.deg2rad(lat_e[:-1]))) * np.deg2rad(1.0))[:, None] * np.ones((1, lond.size))
    w = np.where(ocean, area, 0.0)

    z = xr.open_zarr(a.cache)
    nrec = z.sizes["time"]
    recs = np.arange(0, nrec, a.stride)
    names = ["sst_used", "sst_run_mean", "sw_net", "lw_down", "lw_up", "shflx", "lhflx", "q_net", "ref_sh", "ref_lh", "ref_q_net", "q_net_ice", "f_ice"]
    ice = None
    if a.sea_ice_clim:
        _d = xr.open_dataset(a.sea_ice_clim)["siconc"].transpose("month", "lat", "lon")
        _la = np.asarray(_d["lat"], float); _lo = np.mod(np.asarray(_d["lon"], float), 360.0)
        _v = np.nan_to_num(np.asarray(_d.values, float))[:, :, np.argsort(_lo)]; _lo = np.sort(_lo)
        from scipy.interpolate import RegularGridInterpolator as _RGI
        _lop = np.concatenate([[_lo[-1] - 360], _lo, [_lo[0] + 360]])
        _vp = np.concatenate([_v[:, :, -1:], _v, _v[:, :, :1]], axis=2)
        _q = np.stack([np.clip(LAd, _la.min(), _la.max()), np.mod(LOd, 360.0)], axis=-1)
        ice = np.clip(np.stack([_RGI((_la, _lop), _vp[m], bounds_error=False, fill_value=0.0)(_q) for m in range(12)]), 0, 1)
    acc = {k: np.zeros(ocean.shape) for k in names}
    t0 = time.time()
    for n, rec in enumerate(recs):
        day = rec / 8.0
        sstC = sst_at(day)
        sstK = jnp.asarray(sstC + 273.15)
        acc["sst_used"] += sstC
        if obs_m is not None and snaps is not None:
            acc["sst_run_mean"] += _run_at(day)
        slc = load_jra55_slice(a.cache, day, ref_year=a.ref_year, cycle=True)
        atm = jra55_to_atm_surface(slc, jnp.asarray(lat2d), jnp.asarray(lon2d), day, ref_year=a.ref_year, co2_ppmv=400.0)
        tr = ocean_tile_response(atm, sstK, zer, zer, cfg)
        f = {k: np.asarray(v, float) for k, v in dict(sw_down=atm.sw_down, alb=tr.albedo, lw_down=atm.lw_down, lw_up=tr.lw_up,
                                                      shflx=tr.shflx, lhflx=tr.lhflx, u=atm.u_lowest, v=atm.v_lowest,
                                                      T=atm.T_lowest, q=atm.q_lowest).items()}
        sw_net = f["sw_down"] * (1 - f["alb"])
        qn = sw_net + f["lw_down"] - f["lw_up"] - f["shflx"] - f["lhflx"]
        acc["sw_net"] += sw_net; acc["lw_down"] += f["lw_down"]; acc["lw_up"] += f["lw_up"]
        acc["shflx"] += f["shflx"]; acc["lhflx"] += f["lhflx"]; acc["q_net"] += qn
        if ice is not None:
            pos = (np.mod(day, 365.0) + 1.0 - 15.5) / (365.0 / 12.0)
            m0 = int(np.floor(pos)); fr = pos - m0
            fi = (1 - fr) * ice[m0 % 12] + fr * ice[(m0 + 1) % 12]
            acc["f_ice"] += fi
            acc["q_net_ice"] += (1 - fi) * qn
        sh, lh = ly04(f["u"], f["v"], f["T"], f["q"], sstC + 273.15)
        acc["ref_sh"] += sh; acc["ref_lh"] += lh
        acc["ref_q_net"] += f["sw_down"] * (1 - 0.066) + f["lw_down"] - SIGMA * (sstC + 273.15) ** 4 - sh - lh
        if n % 100 == 0:
            print(f"  record {rec}/{nrec}  ({time.time() - t0:.0f} s)", flush=True)
    for k in acc:
        acc[k] /= len(recs)

    gm = lambda f, m=ocean: np.sum(f * w * m) / np.sum(w * m)  # noqa: E731
    LA = LAd
    bands = {"global": ocean, "S of 50S": ocean & (LA < -50), "50S-20S": ocean & (LA >= -50) & (LA < -20),
             "20S-20N": ocean & (np.abs(LA) <= 20), "20N-50N": ocean & (LA > 20) & (LA <= 50), "N of 50N": ocean & (LA > 50)}
    print(f"\nannual-mean ocean fluxes [W m-2], {len(recs)} records; SST from {a.sst_obs or a.sst_run or a.restart}; + = into ocean")
    print("  " + f"{'':26s}" + "".join(f"{b:>10s}" for b in bands))
    rows = [("legoESM SW net", acc["sw_net"]), ("legoESM LW net", acc["lw_down"] - acc["lw_up"]),
            ("legoESM SH (up)", acc["shflx"]), ("legoESM LH (up)", acc["lhflx"]),
            ("legoESM Q net", acc["q_net"]), ("ref Q net", acc["ref_q_net"])]
    if ice is not None:
        rows.append(("legoESM Q net x(1-f_ice)", acc["q_net_ice"]))
    ref = None
    if a.ref_hfds:
        import glob as _glob
        _fs = sorted(_glob.glob(a.ref_hfds))
        if not _fs:
            raise SystemExit(f"no file matches --ref-hfds {a.ref_hfds}")
        rd = (xr.open_dataset(_fs[0], decode_times=False) if len(_fs) == 1 else
              xr.open_mfdataset(_fs, decode_times=False, combine="nested", concat_dim="time", data_vars="minimal",
                                coords="minimal", compat="override"))
        hv = rd["hfds"]
        tdim = next((d for d in hv.dims if d.lower().startswith("time")), None)
        if tdim:
            hv = hv.isel({tdim: slice(-12 * a.ref_years, None)}).mean(tdim)
        names_ = list(rd.coords) + list(rd.variables)
        rlat = next(rd[n] for n in names_ if n.lower() in ("lat", "latitude", "nav_lat", "tlat"))
        rlon = next(rd[n] for n in names_ if n.lower() in ("lon", "longitude", "nav_lon", "tlong", "tlon"))
        rla, rlo, hval = np.asarray(rlat, float), np.asarray(rlon, float), np.asarray(hv.values, float)
        if rla.ndim == 1 and hval.ndim == 2 and rla.size == hval.shape[0]:
            rla, rlo = np.meshgrid(rla, rlo, indexing="ij")
        ok_ = np.isfinite(hval) & np.isfinite(rla) & np.isfinite(rlo)
        from scipy.spatial import cKDTree
        xyz = lambda la_, lo_: np.stack([np.cos(np.deg2rad(la_)) * np.cos(np.deg2rad(lo_)),  # noqa: E731
                                         np.cos(np.deg2rad(la_)) * np.sin(np.deg2rad(lo_)), np.sin(np.deg2rad(la_))], -1)
        _, ii = cKDTree(xyz(rla[ok_], rlo[ok_])).query(xyz(LAd.ravel(), LOd.ravel()))
        ref = hval[ok_][ii].reshape(LAd.shape)
        print(f"  reference hfds: {len(_fs)} files, mean of the last {a.ref_years} years")
        rows.append(("reference hfds", ref))
        qm0 = acc["q_net_ice"] if ice is not None else acc["q_net"]
        rows.append(("legoESM - ref hfds", qm0 - ref))
    for name, f in rows:
        print(f"  {name:26s}" + "".join(f"{gm(f, m):10.2f}" for m in bands.values()))
    qm = acc["q_net_ice"] if ice is not None else acc["q_net"]
    if obs_m is not None and snaps is not None:
        print("\n  annual-mean SST [C]   band         run  observed   run-obs")
        for lo_ in range(-80, 90, 10):
            mb = ocean & (LA >= lo_) & (LA < lo_ + 10)
            if mb.sum() > 20:
                print(f"                       {lo_:+4d}..{lo_ + 10:+4d} {gm(acc['sst_run_mean'], mb):8.2f} "
                      f"{gm(acc['sst_used'], mb):9.2f} {gm(acc['sst_run_mean'] - acc['sst_used'], mb):8.2f}")
    if ref is not None:
        print("\n  zonal means [W m-2]   band       legoESM  reference   diff")
        for lo_ in range(-80, 90, 10):
            mb = ocean & (LA >= lo_) & (LA < lo_ + 10) & np.isfinite(ref)
            if mb.sum() > 20:
                print(f"                       {lo_:+4d}..{lo_ + 10:+4d} {gm(qm, mb):9.1f} {gm(ref, mb):9.1f} {gm(qm - ref, mb):7.1f}")
    np.savez(a.out + ".npz", lat=latd, lon=lond, ocean=ocean, ref=(ref if ref is not None else np.zeros(1)), **acc)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)
    msk = lambda f: np.ma.masked_where(~ocean, f)  # noqa: E731
    panels = [(ax[0, 0], qm, "legoESM Q net" + (" x (1-f_ice)" if ice is not None else ""), TwoSlopeNorm(0, -150, 150))]
    if ref is not None:
        panels += [(ax[0, 1], ref, "reference hfds", TwoSlopeNorm(0, -150, 150)),
                   (ax[1, 0], qm - ref, "legoESM - reference hfds", TwoSlopeNorm(0, -80, 80))]
    else:
        panels += [(ax[0, 1], acc["ref_q_net"], "LY04 reference Q net", TwoSlopeNorm(0, -150, 150)),
                   (ax[1, 0], acc["q_net"] - acc["ref_q_net"], "legoESM - LY04", TwoSlopeNorm(0, -30, 30))]
    if ice is not None:
        panels += [(ax[1, 1], acc["f_ice"], "annual-mean f_ice", TwoSlopeNorm(0.5, 0, 1))]
    for axx, f, t, nm in panels:
        pc = axx.pcolormesh(lond, latd, msk(f), cmap="RdBu_r", norm=nm, shading="auto")
        axx.set_title(t); fig.colorbar(pc, ax=axx, shrink=0.85)
    fig.savefig(a.out + ".png", dpi=150)
    print(f"wrote {a.out}.png and {a.out}.npz")


if __name__ == "__main__":
    main()

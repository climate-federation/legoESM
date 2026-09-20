#!/usr/bin/env python3
"""Polar-cap water and isolation numbers from MPAS checkpoints against ERA5.

Two views of the same question -- is the Arctic dry because water is removed
or because it never arrives -- read straight from checkpoints (full 3D state)
so that every day with a checkpoint can be scored, not only published months:

  profile   q_v, ERA5 q, their ratio, RH w.r.t. liquid (model and ERA5) and the
            T bias at reference pressures, area-weighted over lat >= --lat-lo.
  loop      the self-isolation loop's five numbers per checkpoint: the
            surface-pressure dome (85-90N minus 60-70N, ocean cells, model
            minus ERA5), the 925 hPa water ratio and RH_liq over >= 75N, the
            lowest-layer T bias over >= 75N, and the 850 hPa zonal wind over
            70-90N next to ERA5's.

ERA5 = monthly climatology 1979-2014 of the run's month (ta, hus, ps, ua),
hus converted to mixing ratio; the model is interpolated in log-pressure per
column to each reference pressure and a level below a column's surface is
left out of that column's mean (no extrapolation).  Cell order is taken from
the checkpoint's own column index (cloud_layers.cell_order).

Usage:
  cap_water.py profile <run>:<day> [...] [--lat-lo 75]
  cap_water.py loop    <run>:<day> [...]
"""
from __future__ import annotations

import argparse
import glob
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

PLEV = np.array([1000, 925, 850, 700, 600, 500, 400, 300], float) * 100.0
ERA5_YEARS = ("1979-01-01", "2014-12-31")


def columns_to_plev(p_full, field, plev):
    """Interpolate a top-down (ncol, nlev) field in log-p to ``plev`` per
    column; NaN where the target lies below that column's lowest level."""
    out = np.full((p_full.shape[0], np.size(plev)), np.nan)
    lt = np.log(np.atleast_1d(plev))
    for i in range(p_full.shape[0]):
        ok = lt <= np.log(p_full[i, -1])
        out[i, ok] = np.interp(lt[ok], np.log(p_full[i]), field[i])
    return out


def area_mean(x, area, mask):
    """Area-weighted mean of the finite entries of ``x`` inside ``mask``
    (x may be (ncol,) or (ncol, k); returns a scalar or (k,))."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        ok = mask & np.isfinite(x)
        if not ok.any():
            raise SystemExit("FATAL: empty area mean")
        return float((x[ok] * area[ok]).sum() / area[ok].sum())
    return np.array([area_mean(x[:, k], area, mask) for k in range(x.shape[1])])


def era5_month(var, month):
    """ERA5 monthly climatology of ``var`` for ``month``, lat ascending,
    lon in [0, 360)."""
    import xarray as xr
    from heating_budget import ERA5
    files = sorted(glob.glob(f"{ERA5}/{var}/*.nc"))
    if not files:
        raise SystemExit(f"FATAL: no ERA5 monthly {var} under {ERA5}")
    d = xr.open_dataset(files[0])[var].sel(time=slice(*ERA5_YEARS))
    d = d[d.time.dt.month == month].mean("time")
    d = d.rename({("lat" if "lat" in d.dims else "latitude"): "lat",
                  ("lon" if "lon" in d.dims else "longitude"): "lon"})
    if float(d.lon.max()) <= 180.0:
        d = d.assign_coords(lon=(d.lon % 360)).sortby("lon")
    return d.sortby("lat")


def on_cells(d, lat, lon):
    import xarray as xr
    return d.sel(lat=xr.DataArray(lat, dims="c"), lon=xr.DataArray(lon, dims="c"),
                 method="nearest").values



def load_state(run, day):
    z = np.load(f"{rb.ROOT}/{run}/checkpoint_day_{day:04d}.npz", allow_pickle=True)
    exp = json.load(open(f"{rb.ROOT}/{run}/experiment_config.json"))
    lat, lon, area = cl.mesh_coords(exp)
    order = cl.cell_order(z, lat.size)
    from legoesm import constants
    if "day" not in z.files or abs(float(z["day"]) - day) > 1e-6:
        raise SystemExit(f"FATAL: {run} day {day}: checkpoint day stamp mismatch")
    st = {k: np.asarray(z[k], dtype=np.float64)[order] for k in ("T", "p_s", "trc_q_v")}
    vg = np.asarray(z["meta_vgrid"], dtype=np.float64)
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * st["p_s"][:, None]
    if not np.all(np.diff(p_half, axis=1) > 0):
        raise SystemExit(f"FATAL: {run} day {day}: pressure not increasing top-down")
    st["p_full"] = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    for k in ("T", "trc_q_v"):
        if not np.isfinite(st[k]).all() or st[k].shape != st["p_full"].shape:
            raise SystemExit(f"FATAL: {run} day {day}: bad {k}")
    st["u_edge"] = np.asarray(z["u"]) if "u" in z.files else None
    st["month"] = int(((int(exp.get("start_month", 1)) - 1 + day // 30) % 12) + 1)
    st["exp"] = exp
    return st, lat, lon, area, order


def profile(args):
    from legoesm.thermo import saturation_mixing_ratio
    specs = [(s.split(":")[0], int(s.split(":")[1])) for s in args.specs]
    states = [load_state(r, d) for r, d in specs]
    lat, lon, area = states[0][1:4]
    months = {s[0]["month"] for s in states}
    if len(months) != 1:
        raise SystemExit(f"FATAL: checkpoints span different months {months}; one table per month")
    month = months.pop()
    ta, hus = era5_month("ta", month), era5_month("hus", month)
    plev_e = ta.plev.values
    Te = np.stack([on_cells(ta.sel(plev=p), lat, lon) for p in plev_e], 1)     # (ncol, nplev_e)
    qe = np.stack([on_cells(hus.sel(plev=p), lat, lon) for p in plev_e], 1)
    qe = qe / (1.0 - qe)                                                        # specific -> mixing ratio
    o = np.argsort(plev_e)
    Te = columns_to_plev(np.broadcast_to(plev_e[o], (lat.size, o.size)), Te[:, o], PLEV)
    qe = columns_to_plev(np.broadcast_to(plev_e[o], (lat.size, o.size)), qe[:, o], PLEV)
    mask = lat >= args.lat_lo
    print(f"cap >= {args.lat_lo:g}N vs ERA5 month {month} clim (area-weighted on the model's cells; "
          f"model interpolated in log-p per column)")
    print(f"{'hPa':>5s} {'T_ERA5':>7s} {'q_ERA5':>7s} {'RH_ERA5':>7s} | "
          + " ".join(f"{s:>30s}" for s in args.specs))
    rows = []
    for st, *_ in states:
        T = columns_to_plev(st["p_full"], st["T"], PLEV)
        q = columns_to_plev(st["p_full"], st["trc_q_v"], PLEV)
        rh = q / np.asarray(saturation_mixing_ratio(np.where(np.isfinite(T), T, 250.0), PLEV[None, :]))
        rows.append((area_mean(T, area, mask), area_mean(q, area, mask), area_mean(rh, area, mask)))
    Tm, qm = area_mean(Te, area, mask), area_mean(qe, area, mask)
    rhe = area_mean(qe / np.asarray(saturation_mixing_ratio(np.where(np.isfinite(Te), Te, 250.0), PLEV[None, :])), area, mask)
    for k, p in enumerate(PLEV):
        line = f"{p/100:5.0f} {Tm[k]:7.1f} {qm[k]*1e3:7.3f} {rhe[k]:7.2f} | "
        line += " ".join(f"dT{r[0][k]-Tm[k]:+6.1f} q{r[1][k]*1e3:6.3f} ({r[1][k]/qm[k]:4.2f}x) RH{r[2][k]:5.2f}"
                         for r in rows)
        print(line)
    return 0


def loop(args):
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.grids.voronoi import reconstruct_cell_velocity
    from legoesm.grids.factory import create_grid
    import jax.numpy as jnp
    import xarray as xr
    print(f"{'run:day':>14s} {'dome[hPa]':>10s} {'q925/ERA5':>10s} {'RH925':>6s} {'dT_low[K]':>10s} "
          f"{'u850 70-90N':>12s} {'ERA5 u850':>10s}")
    cache = {}
    for spec in args.specs:
        run, day = spec.split(":")
        st, lat, lon, area, order = load_state(run, int(day))
        m = st["month"]
        if m not in cache:
            sf = xr.open_dataset(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_*.nc")[0])["sftlf"]
            ocean = on_cells(sf.sortby("lat"), lat, lon) < 50
            pse = on_cells(era5_month("ps", m), lat, lon)
            hus = era5_month("hus", m).sel(plev=92500.0, method="nearest")
            qe = on_cells(hus, lat, lon); qe = qe / (1.0 - qe)
            ua = era5_month("ua", m).sel(plev=85000.0, method="nearest").mean("lon").sel(lat=slice(70, 90))
            ta = era5_month("ta", m)
            Te_low = None
            cache[m] = (ocean, pse, qe, float(ua.weighted(np.cos(np.deg2rad(ua.lat))).mean()), ta)
            mesh = create_grid("mpas", resolution=int(st["exp"]["grid"]["resolution"]))
        ocean, pse, qe, uref, ta = cache[m]
        dome = (area_mean(st["p_s"] - pse, area, (lat >= 85) & ocean)
                - area_mean(st["p_s"] - pse, area, (lat >= 60) & (lat < 70) & ocean)) / 100.0
        q925 = columns_to_plev(st["p_full"], st["trc_q_v"], 92500.0)[:, 0]
        T925 = columns_to_plev(st["p_full"], st["T"], 92500.0)[:, 0]
        cap = lat >= 75
        qr = area_mean(q925, area, cap) / area_mean(np.where(np.isfinite(q925), qe, np.nan), area, cap)
        rh = area_mean(q925 / np.asarray(saturation_mixing_ratio(np.where(np.isfinite(T925), T925, 250.0), 92500.0)), area, cap)
        from heating_budget import era5_on_columns
        Te = era5_on_columns(lat, st["p_full"], m)
        dT = area_mean(st["T"][:, -1] - Te[:, -1], area, cap)
        if st["u_edge"] is None:
            raise SystemExit(f"FATAL: {spec}: checkpoint carries no edge wind")
        ue, _ = reconstruct_cell_velocity(jnp.asarray(st["u_edge"]), mesh)
        ue = np.asarray(ue, dtype=np.float64)[order]
        u850 = columns_to_plev(st["p_full"], ue, 85000.0)[:, 0]
        u = area_mean(u850, area, lat >= 70)
        print(f"{spec:>14s} {dome:+10.1f} {qr:10.2f} {rh:6.2f} {dT:+10.1f} {u:+12.1f} {uref:+10.1f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile"); p.add_argument("specs", nargs="+"); p.add_argument("--lat-lo", type=float, default=75.0)
    p.set_defaults(fn=profile)
    l = sub.add_parser("loop"); l.add_argument("specs", nargs="+"); l.set_defaults(fn=loop)
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

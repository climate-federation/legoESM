"""Physics diagnostics of a legoESM lat-lon run against WOA temperature.

Reads one run_omip restart (restart_dayNNNNNN.npz) and the WOA annual temperature,
and writes <out>/physics_diag_dayNNNNNN.{png,pdf} with
  a  SST - WOA
  b  maximum current speed over depth (where the flow is unrealistic)
  c  zonal-mean temperature - WOA (0-1500 m)
  d  equatorial Pacific (2S-2N) temperature section, model (colour) vs WOA (contours)
  e  vertical velocity at ~100 m (m/day)
  f  where |u| > 2 m/s: depth of the sea floor and number of wet levels
and prints band-mean SST biases and the 20 fastest cells (lat, lon, level, H, speed).

Usage:
  python scripts/diag_physics_1deg.py \
     --restart <run>/latlon/180x360/restart_day000900.npz \
     --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg.npz \
     --woa-t /work/bb1596/b384036/diffESM/data/woa13v2_thetao_annual_degC.nc \
     --out /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/physics_diag
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

LAND = "#d9d6cf"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "savefig.dpi": 250})


def load_woa_T(path, latd, lond, z):
    import xarray as xr
    from scipy.interpolate import interp1d
    da = xr.open_dataset(path, decode_times=False)
    var = next(v for v in da.data_vars if da[v].ndim >= 3)
    da = da[var]
    t = [d for d in da.dims if d.lower().startswith("time")]
    if t:
        da = da.mean(t[0])
    latn = next(d for d in da.dims if d.lower().startswith("lat"))
    lonn = next(d for d in da.dims if d.lower().startswith("lon"))
    zn = next(d for d in da.dims if d not in (latn, lonn))
    da = da.transpose(zn, latn, lonn).sortby(latn)
    da = da.assign_coords({lonn: np.mod(da[lonn].values, 360.0)}).sortby(lonn)
    first, last = da.isel({lonn: [0]}), da.isel({lonn: [-1]})
    da = xr.concat([last.assign_coords({lonn: last[lonn].values - 360.0}), da,
                    first.assign_coords({lonn: first[lonn].values + 360.0})], dim=lonn)
    lq = np.mod(lond, 360.0); o = np.argsort(lq)
    hor = da.interp({latn: latd, lonn: lq[o]}).values[:, :, np.argsort(o)]
    zs = np.abs(np.asarray(da[zn].values, float))
    out = interp1d(zs, hor, axis=0, bounds_error=False, fill_value=np.nan)(z)
    out = np.moveaxis(out, 0, -1)
    if np.nanmedian(out) > 100:          # Kelvin
        out = out - 273.15
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--woa-t", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    g = np.load(a.grid)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    z = np.abs(np.asarray(g["z"], float)); zh = np.abs(np.asarray(g["z_half"], float))
    r = np.load(a.restart)
    ocean = r["land_mask"] > 0.5
    H = r["H_bathy"]
    act = (H[..., None] > zh[:-1][None, None, :] + 1e-6) & ocean[..., None]
    T = np.where(act, r["T"], np.nan)
    u = 0.5 * (r["u"][:, :-1, :] + r["u"][:, 1:, :])
    v = 0.5 * (r["v"][:-1] + r["v"][1:])
    sp = np.where(act, np.sqrt(u ** 2 + v ** 2), np.nan)
    w = np.where(act, r["w"], np.nan) * 86400.0
    Tw = np.where(act, load_woa_T(a.woa_t, lat, lon, z), np.nan)
    day = Path(a.restart).stem.replace("restart_day", "")
    LA = lat[:, None] * np.ones_like(ocean, float)

    print(f"restart {a.restart} (day {int(day)})")
    print("SST - WOA by latitude band [degC]:")
    for lo, hi in [(-90, -50), (-50, -30), (-30, -10), (-10, 10), (10, 30), (30, 50), (50, 90)]:
        b = (LA >= lo) & (LA < hi) & ocean
        print(f"  {lo:4d}..{hi:3d}: model {np.nanmean(T[..., 0][b]):6.2f}  WOA {np.nanmean(Tw[..., 0][b]):6.2f}  "
              f"bias {np.nanmean((T - Tw)[..., 0][b]):+6.2f}")
    k100 = int(np.argmin(np.abs(z - 100)))
    print(f"global-mean T - WOA by depth: " + ", ".join(
        f"{z[k]:.0f} m {np.nanmean((T - Tw)[..., k]):+.2f}" for k in (0, k100, int(np.argmin(np.abs(z - 500))),
                                                                       int(np.argmin(np.abs(z - 1500))),
                                                                       int(np.argmin(np.abs(z - 3000))))))
    for thr in (1.0, 2.0, 5.0, 10.0):
        print(f"  cells with |u| > {thr:4.1f} m/s: {int(np.nansum(sp > thr))}")
    flat = np.nan_to_num(sp, nan=-1).ravel()
    top = np.argsort(flat)[::-1][:20]
    print("20 fastest cells:  lat    lon  level  H[m]  wet levels  speed[m/s]")
    for idx in top:
        i, j, k = np.unravel_index(idx, sp.shape)
        print(f"               {lat[i]:6.1f} {lon[j]:6.1f}  {k:4d}  {H[i, j]:6.0f}  {int(act[i, j].sum()):4d}  {flat[idx]:8.2f}")

    lon_e = np.concatenate([lon - 0.5, [lon[-1] + 0.5]]); lat_e = np.clip(np.concatenate([lat - 0.5, [lat[-1] + 0.5]]), -90, 90)
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 8.2), constrained_layout=True)

    def mp(ax, f, title, cmap, norm, unit):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad(LAND)
        pc = ax.pcolormesh(lon_e, lat_e, np.where(ocean, f, np.nan), cmap=cm, norm=norm, rasterized=True)
        ax.set_title(title, loc="left"); ax.set_ylim(-80, 90); ax.set_xticks([0, 90, 180, 270, 360])
        fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01).set_label(unit)

    dsst = (T - Tw)[..., 0]
    mp(axs[0, 0], dsst, "a  SST − WOA", "RdBu_r", TwoSlopeNorm(0, -6, 6), "°C")
    smax = np.nanmax(sp, axis=-1)
    mp(axs[0, 1], smax, "b  maximum current speed over depth", "magma_r", LogNorm(0.05, max(2.0, np.nanmax(smax))), "m/s")
    ax = axs[1, 0]
    zm = np.nanmean(T - Tw, axis=1)
    kk = z <= 1500
    pc = ax.pcolormesh(lat, z[kk], zm[:, kk].T, cmap="RdBu_r", norm=TwoSlopeNorm(0, -5, 5), shading="auto")
    ax.invert_yaxis(); ax.set_title("c  zonal-mean T − WOA", loc="left"); ax.set_xlabel("latitude"); ax.set_ylabel("depth [m]")
    fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01).set_label("°C")
    ax = axs[1, 1]
    eq = (np.abs(lat) <= 2)
    jj = (lon >= 120) & (lon <= 280)
    kk = z <= 500
    Tm = np.nanmean(T[eq][:, jj][..., kk], axis=0); Tobs = np.nanmean(Tw[eq][:, jj][..., kk], axis=0)
    pc = ax.pcolormesh(lon[jj], z[kk], Tm.T, cmap="viridis", vmin=8, vmax=30, shading="auto")
    cs = ax.contour(lon[jj], z[kk], Tobs.T, levels=[12, 16, 20, 24, 28], colors="white", linewidths=0.8)
    ax.clabel(cs, fontsize=6, fmt="%d")
    ax.invert_yaxis(); ax.set_title("d  equatorial Pacific T (2°S–2°N); WOA contours", loc="left")
    ax.set_xlabel("longitude"); fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01).set_label("°C")
    mp(axs[2, 0], w[..., k100], f"e  vertical velocity at {z[k100]:.0f} m", "RdBu_r", TwoSlopeNorm(0, -5, 5), "m/day")
    ax = axs[2, 1]
    fast = smax > 2.0
    ax.scatter(H[fast & ocean], act.sum(-1)[fast & ocean], s=6, c="#e34948", label="|u| > 2 m/s")
    ax.scatter(H[ocean & ~fast][::20], act.sum(-1)[ocean & ~fast][::20], s=2, c="#8a8a85", label="other (1 in 20)")
    ax.set_xscale("log"); ax.set_xlabel("sea-floor depth H [m]"); ax.set_ylabel("wet levels")
    ax.set_title("f  where the flow is too fast", loc="left"); ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"legoESM 1° physics vs WOA, model day {int(day)}", x=0.01, ha="left", fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"physics_diag_day{day}.{ext}")
    print(f"wrote {out}/physics_diag_day{day}.png")


if __name__ == "__main__":
    main()

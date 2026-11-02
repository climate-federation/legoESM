"""Compare legoESM physics runs (restart snapshots) with WOA and with each other.

For each run (restart_dayNNNNNN.npz, all on the same grid and the same day):
  * GM/Redi settings found in the run's run_config.json (to confirm GM/Redi was really on)
  * horizontal-mean T and S bias vs WOA at selected depths, and the RMS T error
  * T bias at the level nearest 460 m by basin
  * volume-mean T and S (drift away from WOA)
  * Atlantic MOC: maximum below 300 m at 26.5 N and between 30 and 60 N [Sv]
  * global MOC south of 40 S: maximum (Deacon cell) and minimum [Sv]
  * Drake Passage transport (68 W, sea floor to surface) [Sv]
  * mean upwelling at the level nearest 100 m, equatorial Pacific 5 S - 5 N [m/day]
  * mixed-layer depth (T criterion: 0.2 degC below the value at the level nearest 10 m),
    mean by band; note the restart is an instantaneous snapshot (day 3650 = end of December)
Figure: zonal-mean T bias (0-2000 m) for each run and the difference of the last two runs;
Atlantic MOC for each run; the global-mean T bias profile of all runs.

Usage:
  python compare_physics_runs.py \
     --run noGM=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/phys_spinup10y_dt300/latlon/180x360/restart_day003650.npz \
     --run GM=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/phys_gm10y_dt300/latlon/180x360/restart_day003650.npz \
     --woa /work/bb1596/b384036/diffESM/data/woa_on_model_grid_1deg_mc.nc \
     --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg_mc.npz \
     --out /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/paper_figs/phys_gm_vs_nogm
"""
import argparse
import json
from pathlib import Path

import numpy as np

R_E = 6.371e6


def basin_masks(latd, lond, ocean):
    """Same geographic basins as evaluate_marbl_3d.py."""
    LA, LO = np.meshgrid(latd, np.mod(lond, 360.0), indexing="ij")
    so, arc = ocean & (LA < -35), ocean & (LA >= 66)
    mid = ocean & ~so & ~arc
    bands = [(-35, 8, 290.0), (8, 9, 281.0), (9, 11, 276.0), (11, 14, 274.0), (14, 17, 271.0),
             (17, 19, 265.0), (19, 31, 262.0), (31, 66, 260.0)]
    div = np.full(LA.shape, 290.0)
    for lo, hi, d in bands:
        div = np.where((LA >= lo) & (LA < hi), d, div)
    east_lim = np.where(LA >= 30, 45.0, 20.0)
    atl = mid & ((LO >= div) | (LO < east_lim))
    ind = mid & ~atl & (LO >= 20) & (LO < 147) & (LA < 31) & ~((LO >= 100) & (LA > -9))
    pac = mid & ~atl & ~ind
    return {"N Atlantic": atl & (LA >= 0), "S Atlantic": atl & (LA < 0),
            "N Pacific": pac & (LA >= 0), "S Pacific": pac & (LA < 0),
            "Indian": ind, "Southern Ocean": so, "Arctic": arc, "Global": ocean}


def gm_settings(restart):
    p = Path(restart).parent / "run_config.json"
    if not p.exists():
        return "run_config.json not found"
    try:
        c = json.load(open(p))
    except Exception as e:  # noqa: BLE001
        return f"run_config.json unreadable ({e})"
    found = {}

    def walk(d, pre=""):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{pre}{k}.")
        elif any(s in pre.lower() for s in ("gm", "redi", "kappa", "eddy", "skew", "isopyc")):
            found[pre[:-1]] = d
    walk(c)
    return ", ".join(f"{k}={v}" for k, v in found.items()) or "no GM/Redi key in run_config.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=restart_dayNNNNNN.npz (repeat)")
    ap.add_argument("--woa", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import xarray as xr

    g = np.load(a.grid)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() <= np.pi / 2 + 1e-6:
        lat, lon = np.rad2deg(lat), np.rad2deg(lon)
    z, zh = np.abs(np.asarray(g["z"], float)), np.abs(np.asarray(g["z_half"], float))
    nlat, nlon, nz = lat.size, lon.size, z.size
    dlat, dlon = np.deg2rad(np.median(np.diff(lat))), np.deg2rad(np.median(np.diff(lon)))
    lat_e = np.concatenate([[-90.0], 0.5 * (lat[1:] + lat[:-1]), [90.0]])
    area = (R_E ** 2 * (np.sin(np.deg2rad(lat_e[1:])) - np.sin(np.deg2rad(lat_e[:-1]))) * dlon)[:, None] * np.ones((1, nlon))

    w = xr.open_dataset(a.woa)
    def woa(v):
        da = w[v]
        if "time" in da.dims:
            da = da.mean("time")
        da = da.transpose(*[d for d in da.dims if d.startswith("lat")], *[d for d in da.dims if d.startswith("lon")], ...)
        return np.asarray(da.values, float)
    Tw, Sw = woa("T"), woa("S")
    assert Tw.shape == (nlat, nlon, nz), f"WOA T shape {Tw.shape} != grid {(nlat, nlon, nz)}"

    runs = []
    for spec in a.run:
        name, path = spec.split("=", 1)
        r = np.load(path)
        ocean = r["land_mask"] > 0.5
        H = np.asarray(r["H_bathy"], float)
        dz = np.clip(H[..., None] - zh[None, None, :-1], 0.0, (zh[1:] - zh[:-1])[None, None, :]) * ocean[..., None]
        act = dz > 1e-3
        runs.append(dict(name=name, path=path, r=r, ocean=ocean, dz=dz, act=act,
                         T=np.where(act, r["T"], np.nan), S=np.where(act, r["S"], np.nan)))
    ocean, dz, act = runs[0]["ocean"], runs[0]["dz"], runs[0]["act"]
    vol = dz * area[..., None]
    basins = basin_masks(lat, lon, ocean)
    LA = lat[:, None] * np.ones((1, nlon))
    LO = np.mod(lon, 360.0)[None, :] * np.ones((nlat, 1))
    ok_w = act & np.isfinite(Tw) & np.isfinite(Sw)

    def hmean(f, k, m=None):
        sel = ok_w[..., k] & np.isfinite(f[..., k]) & (ocean if m is None else m)
        return np.sum(f[..., k][sel] * area[sel]) / max(np.sum(area[sel]), 1.0)

    depths = [10, 50, 100, 200, 300, 460, 700, 1000, 1500, 2000, 3000, 4000]
    ks = sorted({int(np.argmin(np.abs(z - d))) for d in depths})
    k460 = int(np.argmin(np.abs(z - 460))); k100 = int(np.argmin(np.abs(z - 100))); k10 = int(np.argmin(np.abs(z - 10)))

    print("== GM/Redi settings (run_config.json)")
    for ru in runs:
        print(f"  {ru['name']:8s}: {gm_settings(ru['path'])}")

    print("\n== horizontal-mean bias vs WOA   T [degC] | S [psu]   (RMS T error in brackets)")
    print("  depth[m] " + "".join(f"{ru['name']:>30s}" for ru in runs))
    for k in ks:
        row = f"  {z[k]:7.0f}  "
        for ru in runs:
            bT, bS = hmean(ru["T"] - Tw, k), hmean(ru["S"] - Sw, k)
            rms = np.sqrt(hmean((ru["T"] - Tw) ** 2, k))
            row += f"{bT:+8.2f} | {bS:+6.3f}  [{rms:5.2f}]    "
        print(row)

    print(f"\n== T bias [degC] at {z[k460]:.0f} m by basin")
    print("  basin           " + "".join(f"{ru['name']:>10s}" for ru in runs))
    for b, m in basins.items():
        print(f"  {b:15s} " + "".join(f"{hmean(ru['T'] - Tw, k460, m):+10.2f}" for ru in runs))

    print("\n== volume-mean T, S (WOA on the same cells)")
    sel = ok_w
    print(f"  WOA      T {np.sum(Tw[sel] * vol[sel]) / np.sum(vol[sel]):7.3f}   S {np.sum(Sw[sel] * vol[sel]) / np.sum(vol[sel]):7.3f}")
    for ru in runs:
        s2 = sel & np.isfinite(ru["T"])
        print(f"  {ru['name']:8s} T {np.sum(ru['T'][s2] * vol[s2]) / np.sum(vol[s2]):7.3f}   "
              f"S {np.sum(ru['S'][s2] * vol[s2]) / np.sum(vol[s2]):7.3f}")

    # --- circulation
    atl_cells = basins["N Atlantic"] | basins["S Atlantic"] | basins["Arctic"]
    lat_f = np.arange(nlat + 1) * np.rad2deg(dlat) + lat_e[0]          # v-face latitudes
    print("\n== circulation")
    print("  quantity                         " + "".join(f"{ru['name']:>10s}" for ru in runs))
    rows = {k: [] for k in ("AMOC max 26.5N  [Sv]", "AMOC max 30-60N [Sv]", "SO MOC max <40S [Sv]",
                            "SO MOC min <40S [Sv]", "Drake Passage   [Sv]", "w100 EqPac      [m/d]",
                            "MLD 50-70S      [m]", "MLD 30S-30N     [m]", "MLD 40-70N      [m]")}
    for ru in runs:
        r = ru["r"]
        v = np.asarray(r["v"], float)                                    # (nlat+1, nlon, nz) on faces
        dzf = np.zeros(v.shape)
        dzf[1:-1] = np.minimum(ru["dz"][:-1], ru["dz"][1:])
        dx = R_E * np.cos(np.deg2rad(lat_f))[:, None] * dlon
        V = np.nan_to_num(v) * dzf * dx[..., None]                       # m3/s per face cell
        def moc(mask_faces):
            Vz = np.sum(V * mask_faces[..., None], axis=1)               # (nf, nz)
            return np.cumsum(Vz, axis=1) / 1e6                           # psi at the bottom of each level, from the surface
        fa = np.zeros((nlat + 1, nlon), bool)
        fa[1:-1] = atl_cells[1:] & atl_cells[:-1]
        fa &= (lat_f[:, None] > -34)
        psiA, psiG = moc(fa), moc(np.ones((nlat + 1, nlon), bool))
        ru["psiA"] = psiA
        deep = zh[1:] > 300
        i265 = int(np.argmin(np.abs(lat_f - 26.5)))
        b3060 = (lat_f >= 30) & (lat_f <= 60)
        bso = lat_f < -40
        rows["AMOC max 26.5N  [Sv]"].append(psiA[i265, deep].max())
        rows["AMOC max 30-60N [Sv]"].append(psiA[b3060][:, deep].max())
        rows["SO MOC max <40S [Sv]"].append(psiG[bso].max())
        rows["SO MOC min <40S [Sv]"].append(psiG[bso].min())
        u = np.asarray(r["u"], float)                                    # (nlat, nlon+1, nz) on west faces
        jdp = int(np.argmin(np.abs(np.mod(np.arange(u.shape[1]) * np.rad2deg(dlon) + lon[0] - np.rad2deg(dlon) / 2, 360) - 292.0)))
        jw, je = (jdp - 1) % nlon, jdp % nlon
        dzu = np.minimum(ru["dz"][:, jw], ru["dz"][:, je])
        band = (lat > -70) & (lat < -52)
        rows["Drake Passage   [Sv]"].append(np.sum(np.nan_to_num(u[band, jdp]) * dzu[band] * R_E * dlat) / 1e6)
        wv = np.asarray(r["w"], float)[..., k100] * 86400.0 if "w" in r.files else np.full(ocean.shape, np.nan)
        eq = basins["N Pacific"] | basins["S Pacific"]
        eq &= (np.abs(LA) <= 5) & ru["act"][..., k100]
        rows["w100 EqPac      [m/d]"].append(np.sum(wv[eq] * area[eq]) / np.sum(area[eq]))
        T = ru["T"]
        mld = np.full(ocean.shape, np.nan)
        below = (T < (T[..., k10:k10 + 1] - 0.2)) & ru["act"]
        below[..., :k10 + 1] = False
        has = below.any(-1)
        kfirst = np.argmax(below, axis=-1)
        mld[has] = z[kfirst[has]]
        ru["mld"] = mld
        for key, (lo, hi) in (("MLD 50-70S      [m]", (-70, -50)), ("MLD 30S-30N     [m]", (-30, 30)),
                              ("MLD 40-70N      [m]", (40, 70))):
            m = ocean & (LA >= lo) & (LA <= hi) & np.isfinite(mld)
            rows[key].append(np.sum(mld[m] * area[m]) / np.sum(area[m]))
    for k, vals in rows.items():
        print(f"  {k:32s} " + "".join(f"{x:10.2f}" for x in vals))
    print("  (references: AMOC 26.5N ~17 Sv RAPID; Drake Passage ~137-173 Sv; MLD is a December snapshot)")

    # --- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.size": 8})
    n = len(runs)
    ncol = n + (1 if n >= 2 else 0)
    fig, ax = plt.subplots(2, ncol, figsize=(3.6 * ncol, 6.0), constrained_layout=True)
    kz = z <= 2000
    def zm(f):
        m = ok_w & np.isfinite(f)
        return np.nansum(np.where(m, f, 0) * area[..., None], 1) / np.maximum(np.sum(m * area[..., None], 1), 1)
    zb = [np.where(np.sum(ok_w, 1) > 0, zm(ru["T"] - Tw), np.nan) for ru in runs]
    nb = TwoSlopeNorm(0, -4, 4)
    for i, ru in enumerate(runs):
        pc = ax[0, i].pcolormesh(lat, z[kz], zb[i][:, kz].T, cmap="RdBu_r", norm=nb, shading="auto")
        ax[0, i].set_title(f"{ru['name']}: zonal-mean T - WOA"); ax[0, i].invert_yaxis()
        ax[0, i].set_ylabel("depth [m]")
    fig.colorbar(pc, ax=ax[0, :n], shrink=0.8, label="degC")
    if n >= 2:
        pc = ax[0, n].pcolormesh(lat, z[kz], (zb[-1] - zb[-2])[:, kz].T, cmap="PuOr_r", norm=TwoSlopeNorm(0, -2, 2), shading="auto")
        ax[0, n].set_title(f"{runs[-1]['name']} - {runs[-2]['name']}"); ax[0, n].invert_yaxis()
        fig.colorbar(pc, ax=ax[0, n], shrink=0.8, label="degC")
    for i, ru in enumerate(runs):
        p = ru["psiA"].T.copy()
        p[:, lat_f < -34] = np.nan
        pc = ax[1, i].contourf(lat_f, zh[1:], p, levels=np.arange(-24, 26, 2), cmap="RdBu_r", extend="both")
        ax[1, i].contour(lat_f, zh[1:], p, levels=[0], colors="k", linewidths=0.5)
        ax[1, i].set_xlim(-34, 70); ax[1, i].invert_yaxis(); ax[1, i].set_title(f"{ru['name']}: Atlantic MOC [Sv]")
        ax[1, i].set_ylabel("depth [m]")
    fig.colorbar(pc, ax=ax[1, :n], shrink=0.8, label="Sv")
    if n >= 2:
        for ru in runs:
            prof = [hmean(ru["T"] - Tw, k) for k in range(nz)]
            ax[1, n].plot(prof, z, label=ru["name"])
        ax[1, n].axvline(0, color="k", lw=0.5); ax[1, n].invert_yaxis(); ax[1, n].set_yscale("symlog", linthresh=200)
        ax[1, n].set_xlabel("horizontal-mean T - WOA [degC]"); ax[1, n].legend(frameon=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
        print(f"wrote {a.out}.{ext}")


if __name__ == "__main__":
    main()

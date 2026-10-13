"""Does each OMIP grid answer the SAME equatorial wind the same way?  Day 5..30.

All three lanes are driven by the same CORE-II winds through the same bulk
formula, so off the equator their wind-driven (Ekman) poleward transport in the
top 50 m should agree to sampling noise whatever their mixing: that transport
is set by the stress and f alone.  A lane that carries markedly less poleward
transport than the others is not receiving, or not keeping, the wind's push.

Per 5-day mark, per source, on the common 1-degree grid of
``omip_pacbias_grid_profiles`` (same remap, NEMO level centres, one wet
footprint per mark = finite in every source):

  V_N, V_S  meridional transport 0-50 m [m^2/s], mean over the wet
            (row, lon) cells of 3.5..5.5 N / S, 180-260E (one shared footprint,
            so the weighting is identical for every source; poleward = +N, -S)
  div_ek    V_N - V_S (two-sided poleward export, the equatorial upwelling feed)
  u_sec     0-20 m zonal velocity, |lat|<=1, 210-270E
  u_euc     max zonal velocity below 30 m, |lat|<=1, 180-270E, and its depth

SAMPLING.  Lanes use INSTANTANEOUS velocity (FESOM stores no mean velocity) at
the same marks, so the three lanes are like for like; inertial oscillations
(period ~6-9 days at 3.5-5.5 deg) are not averaged out within a mark, so read
the six-mark mean, not one mark.  NEMO is a 5-DAY MEAN (record d/5-1) and is a
reference only, not like for like.  Tripole velocities are grid-relative;
ORCA1 is geographic in the tropics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omip_pacbias_grid_profiles import (  # noqa: E402
    DAYS, GLAT, GLON, _f, band_weights, remap3, to_levels, wet3_from)

LON_EK = (180.0, 260.0)
LAT_EK = (3.5, 5.5)
Z_EK = 50.0


def transports(u, v, zN, zi):
    """u, v (nz, GLAT, GLON) on NEMO level centres zN -> dict of scalars."""
    bw = band_weights(zi, 0.0, Z_EK)[: zN.size][:, None, None]
    lo = (GLON >= LON_EK[0]) & (GLON <= LON_EK[1])
    used = bw[:, 0, 0] > 0
    V = np.sum(np.where(used[:, None, None], v, 0.0) * bw, axis=0) \
        + np.where(np.all(np.isfinite(v[used]), axis=0), 0.0, np.nan)  # NaN if any layer missing
    rN = (GLAT >= LAT_EK[0]) & (GLAT <= LAT_EK[1])
    rS = (GLAT <= -LAT_EK[0]) & (GLAT >= -LAT_EK[1])
    VN = float(np.nanmean(V[np.ix_(rN, lo)])); VS = float(np.nanmean(V[np.ix_(rS, lo)]))
    eq = np.abs(GLAT) <= 1.0
    l3 = (GLON >= 210.0) & (GLON <= 270.0)
    bs = band_weights(zi, 0.0, 20.0)[: zN.size]
    ue = np.nanmean(u[:, eq][:, :, l3], axis=(1, 2))
    sec = float(np.sum(ue[bs > 0] * bs[bs > 0]) / np.sum(bs[bs > 0]))   # NaN if a 0-20 m layer is missing
    le = (GLON >= 180.0) & (GLON <= 270.0)
    prof = np.nanmean(u[:, eq][:, :, le], axis=1)          # (nz, lon)
    prof = np.where(zN[:, None] > 30.0, prof, np.nan)
    if not np.isfinite(prof).any():
        raise ValueError("no finite equatorial velocity below 30 m")
    k, i = np.unravel_index(np.nanargmax(prof), prof.shape)
    return {"V_N": VN, "V_S": VS, "div_ek": VN - VS, "u_sec": sec,
            "u_euc": float(prof[k, i]), "z_euc": float(zN[k]), "lon_euc": float(GLON[le][i])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--nemo-prefix", required=True)
    p.add_argument("--trp", required=True)
    p.add_argument("--mpas", required=True)
    p.add_argument("--fesom", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    import netCDF4 as nc
    T = nc.Dataset(a.nemo_prefix + "grid_T.nc"); U = nc.Dataset(a.nemo_prefix + "grid_U.nc")
    Vn = nc.Dataset(a.nemo_prefix + "grid_V.nc"); W = nc.Dataset(a.nemo_prefix + "grid_W.nc")
    nlat, nlon = _f(T["nav_lat"][:]), _f(T["nav_lon"][:])
    zN = _f(T["deptht"][:]); kz = int(np.searchsorted(zN, 300.0)) + 1; zN = zN[:kz]
    zi = _f(W["depthw"][:])[: kz + 1]
    assert abs(zi[0]) < 1e-6 and np.all(np.diff(zi) > 0)

    def snap(lane, d):
        f = a.results / lane / f"snapshot_day{d:04d}.npz"
        if not f.exists() and d == DAYS[-1]:
            f = a.results / lane / "snapshot_final.npz"
        s = np.load(f)                       # missing mark -> FileNotFoundError, never a stand-in
        if "time_days" in s.files and abs(float(s["time_days"]) - d) > 0.6:
            raise ValueError(f"{f}: time_days {float(s['time_days'])} is not mark {d}")
        return s

    out = {"provenance": {"argv": sys.argv, "lanes": [a.trp, a.mpas, a.fesom],
                          "sampling": "lanes instantaneous; NEMO 5-day mean"}, "marks": {}}
    marks = []
    for d in DAYS:
        try:
            [snap(l, d) for l in (a.trp, a.mpas, a.fesom)]
        except FileNotFoundError as e:
            print(f"SKIP mark {d}: {e.filename} missing")
            continue
        marks.append(d)
        rec = d // 5 - 1
        G = {}
        toN = np.moveaxis(_f(T["to"][rec, :kz]), 0, -1); wetN = np.isfinite(toN)
        uo = np.nan_to_num(np.moveaxis(_f(U["uo"][rec, :kz]), 0, -1))
        vo = np.nan_to_num(np.moveaxis(_f(Vn["vo"][rec, :kz]), 0, -1))
        uN = 0.5 * (uo + np.roll(uo, 1, axis=1)); vN = 0.5 * (vo + np.roll(vo, 1, axis=0))
        G["NEMO_mean"] = (remap3(uN, nlat, nlon, wetN), remap3(vN, nlat, nlon, wetN))
        t = snap(a.trp, d); zt = np.abs(t["z_center_ref"])[:kz]
        assert np.allclose(zt, zN, atol=0.5)
        wt = wet3_from(t["H_bathy"], zt, t["T"][..., :kz].shape) & (t["land_mask"] > 0.5)[..., None]
        uT = 0.5 * (t["u"][:, :-1, :kz] + t["u"][:, 1:, :kz]); vT = 0.5 * (t["v"][:-1, :, :kz] + t["v"][1:, :, :kz])
        G["tripole"] = (remap3(uT, t["lat_T"], t["lon_T"], wt), remap3(vT, t["lat_T"], t["lon_T"], wt))
        m = snap(a.mpas, d); zm = np.abs(m["z_center_ref"])[:kz]
        assert np.allclose(zm, zN, atol=0.5)
        wm = wet3_from(m["H_bathy"], zm, m["T"][..., :kz].shape) & (m["land_mask"] > 0.5)[..., None]
        G["MPAS"] = (remap3(m["u_east"][:, :kz], m["lat_T"], m["lon_T"], wm),
                     remap3(m["v_north"][:, :kz], m["lat_T"], m["lon_T"], wm))
        fz = snap(a.fesom, d); zf = np.abs(fz["z_center_ref"]); nf = zf.size
        wf = wet3_from(fz["H_bathy"], zf, fz["T"][..., :nf].shape) & (fz["land_mask"] > 0.5)[..., None]
        rf = lambda x: to_levels(remap3(x[..., :nf], fz["lat_T"], fz["lon_T"], wf), zf, zN)
        G["FESOM"] = (rf(fz["u_east"]), rf(fz["v_north"]))
        common = np.all([np.isfinite(x) for uv in G.values() for x in uv], axis=0)
        out["marks"][d] = {lab: transports(np.where(common, u, np.nan), np.where(common, v, np.nan), zN, zi)
                           for lab, (u, v) in G.items()}
        print(d, json.dumps({k: {q: round(x, 3) for q, x in r.items()} for k, r in out["marks"][d].items()}))
    labs = list(out["marks"][marks[0]])
    out["marks_used"] = marks
    out["six_mark_mean"] = {lab: {q: float(np.mean([out["marks"][d][lab][q] for d in marks]))
                                  for q in out["marks"][marks[0]][lab]} for lab in labs}
    print("MEAN", json.dumps({k: {q: round(x, 3) for q, x in r.items()} for k, r in out["six_mark_mean"].items()}))
    (a.out / "eq_wind_response.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    qs = (("V_N", "0-50 m transport 3.5-5.5N [m2/s]"), ("V_S", "0-50 m transport 3.5-5.5S [m2/s]"),
          ("u_sec", "surface current 0-20 m, eq [m/s]"), ("u_euc", "undercurrent core [m/s]"))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axi, (q, lab) in zip(ax.ravel(), qs):
        for s in labs:
            axi.plot(marks, [out["marks"][d][s][q] for d in marks], "o-" if s != "NEMO_mean" else "k--", label=s)
        axi.set_title(lab); axi.set_xlabel("day"); axi.axhline(0, c="0.6", lw=0.5)
    ax[0, 0].legend()
    fig.suptitle("Same winds, three grids: lanes instantaneous, NEMO 5-day mean (180-260E)")
    fig.savefig(a.out / "eq_wind_response.png", dpi=110)
    print("PNG", a.out / "eq_wind_response.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

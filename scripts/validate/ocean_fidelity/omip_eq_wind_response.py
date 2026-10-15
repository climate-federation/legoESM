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
  U_col_W/E 0-400 m zonal transport [m^2/s], |lat|<=1, mean over 160-210E /
            210-260E (NaN at a longitude if any layer is missing).  Vertical
            mixing only moves momentum between depths; a column difference
            must come from pressure gradient, lateral friction or advection
  deta_WE   sea-surface height west minus east [cm] across 160-260E, from a
            least-squares line through the |lat|<=1 mean (+ = eastward push)
  F_tot/F_ssh  0-400 m zonal pressure force per unit area / rho0 [1e-5 m2/s2],
            |lat|<=1, 160-260E (+ = eastward), from the hydrostatic pressure
            of each lane's own T, S, eta through ONE equation of state
            (Wright, the tripole/MPAS one) so lanes differ only by structure;
            F_ssh = the sea-surface part alone, F_tot - F_ssh = density part
  ug_N/u_N, ug_S/u_S  50-300 m mean geostrophic vs actual zonal velocity
            [m/s] at 2N / 2S (rows 1.5-2.5), 160-260E
  A_x, A_y, A_z, Cor, lapU  0-400 m zonal momentum terms per unit area of the
            box |lat|<=1, 160-260E [1e-5 m2/s2], + = eastward gain, flux form:
            A_x = (int u^2 dz at 160E - at 260E) / Lx; A_y = (int u v dz at
            1S - at 1N) / Ly; A_z = u w at 400 m, w = int_0^400 div(u, v) dz
            (rigid lid, + up); Cor = f int v dz; lapU = curvature of the 0-400 m
            transport, i.e. lateral friction PER 1000 m2/s of viscosity
            (snapshots carry no viscosity).  U_box = box-mean 0-400 m transport.
            All from the 1-degree remap: an estimate, not the native operator.

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
    DAYS, GLAT, GLON, _f, band_weights, remap, remap3, to_levels, wet3_from)

LON_EK = (180.0, 260.0)
LAT_EK = (3.5, 5.5)
Z_EK = 50.0
Z_COL = 400.0
LON_COL = ((160.0, 210.0), (210.0, 260.0))
LON_SLOPE = (160.0, 260.0)


def transports(u, v, zN, zi, eta=None):
    """u, v (nz, GLAT, GLON) on NEMO level centres zN, optional eta (GLAT, GLON)
    -> dict of scalars."""
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
    bc = band_weights(zi, 0.0, Z_COL)[: zN.size]
    if bc.sum() < Z_COL - 1e-6:
        raise ValueError(f"levels reach only {bc.sum():.1f} m < {Z_COL} m")
    col = np.nanmean(np.sum(u[bc > 0][:, eq] * bc[bc > 0, None, None], axis=0), axis=0)  # full columns only
    r = {"V_N": VN, "V_S": VS, "div_ek": VN - VS, "u_sec": sec,
         "u_euc": float(prof[k, i]), "z_euc": float(zN[k]), "lon_euc": float(GLON[le][i])}
    for tag, (l0, l1) in zip(("U_col_W", "U_col_E"), LON_COL):
        c = col[(GLON >= l0) & (GLON <= l1)]
        if not np.isfinite(c).any():
            raise ValueError(f"no full 0-{Z_COL:.0f} m column at {l0}-{l1}E")
        r[tag] = float(np.nanmean(c))
    if eta is not None:
        ls = (GLON >= LON_SLOPE[0]) & (GLON <= LON_SLOPE[1])
        e = np.nanmean(eta[eq][:, ls], axis=0); x = GLON[ls]; ok = np.isfinite(e)
        if ok.sum() < 0.8 * ok.size:
            raise ValueError(f"equatorial eta finite at only {ok.sum()}/{ok.size} longitudes")
        slope = np.polyfit(x[ok], e[ok], 1)[0]            # m per degree
        r["deta_WE"] = float(-slope * (LON_SLOPE[1] - LON_SLOPE[0]) * 100.0)
    return r


def pressure_terms(T, S, eta, u, zN, zi):
    """Hydrostatic zonal pressure force and off-equator geostrophy.

    T, S, u (nz, GLAT, GLON) on level centres zN (interfaces zi); eta (GLAT, GLON)
    [m].  Pressure at a centre = g*rho0*eta + g * int_0^z (rho - rho0) dz'."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.eos import wright_eos
    g, r0, R = float(constants.g), float(constants.rho_ocean), float(constants.R_earth)
    nz = zN.size
    dz = np.diff(zi)[:nz]
    p_hyd = r0 * g * np.broadcast_to(zN[:, None, None], T.shape)
    b = g * (np.asarray(wright_eos(jnp.asarray(T), jnp.asarray(S), jnp.asarray(p_hyd))) - r0)
    bc = band_weights(zi, 0.0, Z_COL)[:nz]

    def p_at(zc):                                               # Pa at depths zc (nz,)
        return g * r0 * eta[None] + np.cumsum(b * dz[:, None, None], axis=0) \
            - b * dz[:, None, None] + b * (zc - zi[:nz])[:, None, None]
    P = p_at(zN)                                                # NaN where any input is
    Pc = p_at(np.where(bc < dz, zi[:nz] + 0.5 * bc, zN))        # clipped layer: overlap midpoint
    eq = np.abs(GLAT) <= 1.0
    ls = (GLON >= LON_SLOPE[0]) & (GLON <= LON_SLOPE[1])
    full = np.all(np.isfinite(Pc[bc > 0][:, eq]), axis=(0, 1))  # every |lat|<=1 row, full 0-Z_COL
    x = np.deg2rad(GLON[ls]) * R
    out = {}
    for tag, F in (("F_tot", Pc), ("F_ssh", g * r0 * np.broadcast_to(eta[None], P.shape))):
        Fe = np.where(full[None], np.mean(F[:, eq], axis=1), np.nan)[:, ls]   # (nz, lon)
        ok = np.all(np.isfinite(Fe[bc > 0]), axis=0)
        if ok.sum() < 0.8 * ok.size:
            raise ValueError(f"{tag}: full equatorial columns at only {ok.sum()}/{ok.size} longitudes")
        slope = np.array([np.polyfit(x[ok], Fe[k, ok], 1)[0] for k in np.flatnonzero(bc > 0)])
        out[tag] = float(-np.sum(slope * bc[bc > 0]) / r0 * 1e5)
    bg = band_weights(zi, 50.0, 300.0)[:nz]
    dy = np.deg2rad(1.0) * R
    for tag, r_lo, r_hi in (("N", 1.5, 2.5), ("S", -1.5, -2.5)):
        i_lo, i_hi = (int(np.argmin(np.abs(GLAT - v))) for v in (r_lo, r_hi))
        f = 2.0 * float(constants.Omega) * np.sin(np.deg2rad(0.5 * (r_lo + r_hi)))
        dPdy = (P[:, i_hi] - P[:, i_lo]) / (dy * np.sign(r_hi - r_lo))
        ug = -dPdy / (f * r0)
        ua = 0.5 * (u[:, i_lo] + u[:, i_hi])
        both = np.isfinite(ug[bg > 0]).all(0) & np.isfinite(ua[bg > 0]).all(0) & ls
        if both.sum() < 0.8 * ls.sum():
            raise ValueError(f"geostrophy {tag}: full columns at only {both.sum()}/{ls.sum()} longitudes")
        w = bg[bg > 0, None] / bg.sum()
        out["ug_" + tag] = float(np.mean(np.sum(ug[bg > 0][:, both] * w, axis=0)))
        out["u_" + tag] = float(np.mean(np.sum(ua[bg > 0][:, both] * w, axis=0)))
    return out


def momentum_terms(u, v, zN, zi, fields=None):
    """Flux-form 0-Z_COL zonal momentum terms over |lat|<=1, LON_SLOPE (see module doc).

    u, v (nz, GLAT, GLON) on level centres zN, interfaces zi; NaN = dry/missing."""
    from legoesm import constants
    R, Om = float(constants.R_earth), float(constants.Omega)
    nz = zN.size
    bc = band_weights(zi, 0.0, Z_COL)[:nz]
    k = bc > 0
    col = lambda q: np.sum(q[k] * bc[k, None, None], axis=0)      # NaN if any layer missing
    U, V, UU, UV = col(u), col(v), col(u * u), col(u * v)
    if fields is not None:
        fields.update(U=U, UV=UV)
    eq = np.abs(GLAT) <= 1.0
    l0, l1 = LON_SLOPE
    ls = (GLON >= l0) & (GLON <= l1)
    rN, rS = np.flatnonzero(eq).max(), np.flatnonzero(eq).min()
    iW, iE = int(np.flatnonzero(ls).min()), int(np.flatnonzero(ls).max())
    Lx = np.deg2rad(l1 - l0) * R
    Ly = np.deg2rad(GLAT[rN] - GLAT[rS] + 1.0) * R
    face_x = lambda i: np.mean(0.5 * (UU[eq, i - 1] + UU[eq, i]))         # 2-row mean at a lon face
    face_y = lambda r0, r1: np.mean(0.5 * (UV[r0, ls] + UV[r1, ls]))      # lon mean at a lat face
    out = {"A_x": (face_x(iW) - face_x(iE + 1)) / Lx,
           "A_y": (face_y(rS - 1, rS) - face_y(rN, rN + 1)) / Ly}
    lat = np.deg2rad(GLAT)[:, None]
    dx = np.deg2rad(1.0) * R * np.cos(lat)
    dy = np.deg2rad(1.0) * R
    div = np.gradient(U, axis=1) / dx + np.gradient(V * np.cos(lat), axis=0) / (dy * np.cos(lat))
    j = int(np.searchsorted(zN, Z_COL))                          # zN[j-1] < Z_COL <= zN[j]
    if not 0 < j < nz:
        raise ValueError(f"level centres {zN[0]:.1f}-{zN[-1]:.1f} m do not bracket {Z_COL} m")
    wgt = (Z_COL - zN[j - 1]) / (zN[j] - zN[j - 1])
    u400 = (1.0 - wgt) * u[j - 1] + wgt * u[j]                   # linear in depth to Z_COL
    box = lambda q: q[eq][:, ls]
    f = 2.0 * Om * np.sin(lat)
    lap = np.gradient(np.gradient(U, axis=1) / dx, axis=1) / dx + np.gradient(np.gradient(U, axis=0) / dy, axis=0) / dy
    for tag, q in (("A_z", u400 * div), ("Cor", f * V), ("lapU", 1.0e3 * lap), ("U_box", U)):
        b = box(q)
        if np.isfinite(b).sum() < 0.8 * b.size:
            raise ValueError(f"{tag}: finite at only {np.isfinite(b).sum()}/{b.size} box columns")
        out[tag] = float(np.nanmean(b))
    for tag in ("A_x", "A_y"):
        if not np.isfinite(out[tag]):
            raise ValueError(f"{tag}: a box face has a missing column")
    return {t: (x if t == "U_box" else x * 1e5) for t, x in out.items()}


def main(argv=None):
    global Z_COL
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--nemo-prefix", required=True)
    p.add_argument("--trp", required=True)
    p.add_argument("--mpas", required=True)
    p.add_argument("--fesom", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--z-col", type=float, default=Z_COL, help="column transport depth [m]")
    a = p.parse_args(argv)
    Z_COL = a.z_col
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # EOS in fp64
    a.out.mkdir(parents=True, exist_ok=True)
    import netCDF4 as nc
    T = nc.Dataset(a.nemo_prefix + "grid_T.nc"); U = nc.Dataset(a.nemo_prefix + "grid_U.nc")
    Vn = nc.Dataset(a.nemo_prefix + "grid_V.nc"); W = nc.Dataset(a.nemo_prefix + "grid_W.nc")
    nlat, nlon = _f(T["nav_lat"][:]), _f(T["nav_lon"][:])
    zN = _f(T["deptht"][:]); kz = int(np.searchsorted(zN, Z_COL)) + 1; zN = zN[:kz]
    sshv = [k for k in ("zos", "ssh", "sossheig") if k in T.variables]
    if not sshv:
        raise KeyError(f"no sea-surface height in grid_T: {sorted(T.variables)}")
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

    out = {"provenance": {"argv": sys.argv, "lanes": [a.trp, a.mpas, a.fesom], "nemo_ssh": sshv[0],
                          "sampling": "lanes instantaneous; NEMO 5-day mean"}, "marks": {}}
    marks = []
    MAPS = {}
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
        E = {"NEMO_mean": remap(_f(T[sshv[0]][rec]), nlat, nlon, wetN[..., 0])}
        t = snap(a.trp, d); zt = np.abs(t["z_center_ref"])[:kz]
        if not np.allclose(zt, zN, rtol=2e-3, atol=0.5):
            raise ValueError(f"tripole levels differ from NEMO: {np.c_[zt, zN][~np.isclose(zt, zN, rtol=2e-3, atol=0.5)][:5]}")
        wt = wet3_from(t["H_bathy"], zt, t["T"][..., :kz].shape) & (t["land_mask"] > 0.5)[..., None]
        uT = 0.5 * (t["u"][:, :-1, :kz] + t["u"][:, 1:, :kz]); vT = 0.5 * (t["v"][:-1, :, :kz] + t["v"][1:, :, :kz])
        G["tripole"] = (remap3(uT, t["lat_T"], t["lon_T"], wt), remap3(vT, t["lat_T"], t["lon_T"], wt))
        E["tripole"] = remap(t["eta"], t["lat_T"], t["lon_T"], wt[..., 0])
        TS = {"tripole": tuple(remap3(t[q][..., :kz], t["lat_T"], t["lon_T"], wt) for q in ("T", "S"))}
        m = snap(a.mpas, d); zm = np.abs(m["z_center_ref"])[:kz]
        assert np.allclose(zm, zN, rtol=2e-3, atol=0.5)
        wm = wet3_from(m["H_bathy"], zm, m["T"][..., :kz].shape) & (m["land_mask"] > 0.5)[..., None]
        G["MPAS"] = (remap3(m["u_east"][:, :kz], m["lat_T"], m["lon_T"], wm),
                     remap3(m["v_north"][:, :kz], m["lat_T"], m["lon_T"], wm))
        E["MPAS"] = remap(m["eta"], m["lat_T"], m["lon_T"], wm[..., 0])
        TS["MPAS"] = tuple(remap3(m[q][..., :kz], m["lat_T"], m["lon_T"], wm) for q in ("T", "S"))
        fz = snap(a.fesom, d); zf = np.abs(fz["z_center_ref"]); nf = zf.size
        wf = wet3_from(fz["H_bathy"], zf, fz["T"][..., :nf].shape) & (fz["land_mask"] > 0.5)[..., None]
        rf = lambda x: to_levels(remap3(x[..., :nf], fz["lat_T"], fz["lon_T"], wf), zf, zN)
        G["FESOM"] = (rf(fz["u_east"]), rf(fz["v_north"]))
        E["FESOM"] = remap(fz["eta"], fz["lat_T"], fz["lon_T"], wf[..., 0])
        TS["FESOM"] = (rf(fz["T"]), rf(fz["S"]))
        common = np.all([np.isfinite(x) for uv in G.values() for x in uv], axis=0)
        ecom = common[0] & np.all([np.isfinite(x) for x in E.values()], axis=0)
        out["marks"][d] = {lab: transports(np.where(common, u, np.nan), np.where(common, v, np.nan), zN, zi,
                                           eta=np.where(ecom, E[lab], np.nan))
                           for lab, (u, v) in G.items()}
        for lab, (u, v) in G.items():
            fl = {}
            out["marks"][d][lab].update(momentum_terms(
                np.where(common, u, np.nan), np.where(common, v, np.nan), zN, zi, fields=fl))
            MAPS.setdefault(lab, []).append(fl)
        pcom = common & np.all([np.isfinite(x) for ts in TS.values() for x in ts], axis=0) & ecom[None]
        for lab, (Tl, Sl) in TS.items():
            out["marks"][d][lab].update(pressure_terms(
                np.where(pcom, Tl, np.nan), np.where(pcom, Sl, np.nan), E[lab],
                np.where(pcom, G[lab][0], np.nan), zN, zi))
        print(d, json.dumps({k: {q: round(x, 3) for q, x in r.items()} for k, r in out["marks"][d].items()}))
    labs = list(out["marks"][marks[0]])
    out["marks_used"] = marks
    out["six_mark_mean"] = {lab: {q: float(np.mean([out["marks"][d][lab][q] for d in marks]))
                                  for q in out["marks"][marks[0]][lab]} for lab in labs}
    late = [d for d in marks if d >= 10]
    out["dUdt_late_1e5"] = {lab: float(np.polyfit(np.array(late) * 86400.0,
                                                  [out["marks"][d][lab]["U_box"] for d in late], 1)[0] * 1e5)
                            for lab in labs} if len(late) >= 2 else {}
    print("dU_box/dt days", late, json.dumps({k: round(x, 3) for k, x in out["dUdt_late_1e5"].items()}))
    print("MEAN", json.dumps({k: {q: round(x, 3) for q, x in r.items()} for k, r in out["six_mark_mean"].items()}))
    (a.out / "eq_wind_response.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    qs = (("V_N", "0-50 m transport 3.5-5.5N [m2/s]"), ("V_S", "0-50 m transport 3.5-5.5S [m2/s]"),
          ("u_sec", "surface current 0-20 m, eq [m/s]"), ("u_euc", "undercurrent core [m/s]"),
          ("U_col_W", "0-400 m zonal transport 160-210E [m2/s]"),
          ("U_col_E", "0-400 m zonal transport 210-260E [m2/s]"),
          ("deta_WE", "SSH west minus east 160-260E [cm]"))
    fig, ax = plt.subplots(4, 2, figsize=(11, 13), constrained_layout=True)
    ax[3, 1].axis("off")
    for axi, (q, lab) in zip(ax.ravel(), qs):
        for s in labs:
            axi.plot(marks, [out["marks"][d][s][q] for d in marks], "o-" if s != "NEMO_mean" else "k--", label=s)
        axi.set_title(lab); axi.set_xlabel("day"); axi.axhline(0, c="0.6", lw=0.5)
    ax[0, 0].legend()
    fig.suptitle("Same winds, three grids: lanes instantaneous, NEMO 5-day mean (180-260E)")
    fig.savefig(a.out / "eq_wind_response.png", dpi=110)
    print("PNG", a.out / "eq_wind_response.png")
    fig, ax = plt.subplots(len(labs) + 1, 2, figsize=(13, 3.0 * (len(labs) + 1)), constrained_layout=True)
    for j, (key, ttl, vm) in enumerate((("U", "0-400 m zonal transport [m2/s]", 80.0),
                                         ("UV", "0-400 m int u v dz [m3/s2]", 4.0))):
        for i, s in enumerate(labs):
            fld = np.nanmean([fl[key] for fl in MAPS[s]], axis=0)
            pc = ax[i, j].pcolormesh(GLON, GLAT, fld, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="nearest")
            ax[i, j].add_patch(plt.Rectangle((LON_SLOPE[0], -1.0), LON_SLOPE[1] - LON_SLOPE[0], 2.0, fill=False, ec="k"))
            ax[i, j].set_title(f"{s}: {ttl}, mean of days {marks}", fontsize=9)
            fig.colorbar(pc, ax=ax[i, j], shrink=0.8)
    terms = ("F_tot", "A_x", "A_y", "A_z", "Cor", "lapU")
    lanes = [s for s in labs if s != "NEMO_mean"]
    x = np.arange(len(terms))
    for k, s in enumerate(lanes):
        ax[-1, 0].bar(x + 0.25 * (k - 1), [out["six_mark_mean"][s][t] for t in terms], 0.25, label=s)
    ax[-1, 0].set_xticks(x, ["pressure", "adv W-E", "adv S-N", "adv 400m", "Coriolis", "friction/1000"])
    ax[-1, 0].axhline(0, c="0.5", lw=0.5); ax[-1, 0].legend(fontsize=8)
    ax[-1, 0].set_title("box |lat|<=1, 160-260E, 0-400 m: momentum terms [1e-5 m2/s2], + = eastward gain", fontsize=9)
    for s in labs:
        ax[-1, 1].plot(marks, [out["marks"][d][s]["U_box"] for d in marks], "o-" if s != "NEMO_mean" else "k--", label=s)
    ax[-1, 1].set_title("box-mean 0-400 m zonal transport [m2/s]", fontsize=9); ax[-1, 1].set_xlabel("day")
    ax[-1, 1].legend(fontsize=8)
    fig.suptitle("Equatorial momentum: lanes instantaneous, NEMO 5-day mean (1-degree remap)")
    fig.savefig(a.out / "eq_momentum.png", dpi=100)
    print("PNG", a.out / "eq_momentum.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

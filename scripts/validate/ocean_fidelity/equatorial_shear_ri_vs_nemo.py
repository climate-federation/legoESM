#!/usr/bin/env python
"""Cold-tongue near-surface mixing: shear^2, N^2, Ri and diffusivity profiles,
ours vs NEMO, box mean over an equatorial longitude window.

WHY.  On the 5-day WINDOW MEAN the cold tongue (220-240E, |lat|<=2) is +1.6 K
too warm above 16 m and -1.2 K too cold at 40-60 m at day 30, with the same
shape at 1/13 the amplitude at day 5: heat is not carried across 15-60 m as
NEMO carries it.  Two chains produce that: (a) weaker shear (a weaker
undercurrent) -> less shear-driven TKE below the mixed layer; (b) the same
shear but a closure that mixes less at the same Ri.  They separate on the
profiles: (a) shows S^2 lower than NEMO's at 20-60 m with Ri higher; (b) shows
S^2 and Ri matching NEMO while K_H is lower.

CAVEATS PRINTED WITH THE NUMBERS.  NEMO's uo/vo/avt/bn2 are 5-day means; the
shear^2 of a mean velocity is a LOWER bound on the mean shear^2, and a mean
avt is not the avt of the mean state.  Ours: u/v/K_H are the 00 UTC snapshot
(afternoon in this box, the low phase of the diurnal deep cycle), T/S the
window mean when --use-mean-fields.  So the comparison is for ORDER and SHAPE;
it does not close a budget.  Staggering: u/v are averaged onto the box as
point clouds (U/V-point offsets are half a cell, negligible for a 20-degree
box mean); everything vertical is on each model's OWN levels, which are the
same 75 NEMO levels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_REPO / "packages" / _p))
sys.path.insert(0, str(_REPO))


def _box_mean(field, w, box):
    """Area-weighted box mean along the column axis, NaN-aware -> (nlev,)."""
    f = field[box]
    ww = w[box][:, None] * np.isfinite(f)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(np.where(np.isfinite(f), f, 0.0) * w[box][:, None], axis=0) / ww.sum(axis=0)


def _shear2(u, v, zc):
    """(du/dz)^2 + (dv/dz)^2 on the interior interfaces between cell centres."""
    dz = np.diff(zc)
    du = np.diff(u, axis=-1) / dz
    dv = np.diff(v, axis=-1) / dz
    return du ** 2 + dv ** 2


def _n2(T, S, gdept, gdepw_int):
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept), jnp.asarray(gdepw_int),
        eos_form="teos10", e3w_source="depth_difference"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--use-mean-fields", action="store_true",
                    help="our T/S from the 5-day window mean (T_mean_hw/S_mean_hw); refused if absent")
    ap.add_argument("--nemo-dir", required=True, type=Path, help="directory of ORCA1_5d_*_grid_{T,U,V,W}.nc")
    ap.add_argument("--nemo-stem", default="ORCA1_5d_20000101_20000331")
    ap.add_argument("--rec", type=int, required=True)
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=26)
    ap.add_argument("--n2-threshold", type=float, default=-1e-12,
                    help="EVD trigger threshold on N2 [1/s2]; must equal the run's enhanced_diffusion.n2_threshold")
    ap.add_argument("--rn-evd", type=float, default=100.0, help="NEMO rn_evd [m2/s], for the implied EVD fraction")
    a = ap.parse_args()
    import netCDF4 as nc

    # ---- ours -------------------------------------------------------------
    s = np.load(a.snapshot)
    if a.use_mean_fields:
        if "T_mean_hw" not in s.files:
            raise SystemExit(f"{a.snapshot}: --use-mean-fields but no T_mean_hw")
        T, S = np.asarray(s["T_mean_hw"], float), np.asarray(s["S_mean_hw"], float)
        tsrc = "T_mean_hw/S_mean_hw (5-day window mean)"
    else:
        T, S = np.asarray(s["T"], float), np.asarray(s["S"], float)
        tsrc = "T/S instantaneous 00 UTC"
    if a.use_mean_fields and "u_mean" in s.files:
        u, v = np.asarray(s["u_mean"], float), np.asarray(s["v_mean"], float)
        usrc = "u_mean/v_mean (5-day window mean, same statistic as NEMO uo/vo)"
    else:
        u, v = np.asarray(s["u"], float), np.asarray(s["v"], float)
        usrc = "u/v instantaneous 00 UTC (NEMO uo/vo are 5-day means: S2 of a mean is a lower bound)"
    KH = np.asarray(s["K_H_diag"], float)
    lat, lon = np.asarray(s["lat_T"], float), np.asarray(s["lon_T"], float) % 360.0
    wet = np.asarray(s["land_mask"], float) > 0.5
    zc = np.abs(np.asarray(s["z_center_ref"], float)); zw = np.abs(np.asarray(s["z_interface_ref"], float))
    area = np.asarray(s["cell_area"], float)
    nlev = zc.size
    print(f"[ours] {a.snapshot.name}: T {T.shape} u {u.shape} v {v.shape} K_H_diag {KH.shape} "
          f"zc {zc.shape} zw {zw.shape}; T/S source = {tsrc}; u/v source = {usrc}; K_H = instantaneous 00 UTC")
    # C-grid: u on (ny, nx+1) east faces, v on (ny+1, nx) north faces ->
    # average the two faces of each cell onto the T point.
    if u.shape == (T.shape[0], T.shape[1] + 1, nlev):
        u = 0.5 * (u[:, :-1] + u[:, 1:])
    if v.shape == (T.shape[0] + 1, T.shape[1], nlev):
        v = 0.5 * (v[:-1] + v[1:])
    if u.shape != T.shape or v.shape != T.shape:
        raise SystemExit(f"u/v shape {u.shape}/{v.shape} != T shape {T.shape}: staggering not handled")
    if zw.size == nlev - 1:
        zw_int = zw                       # interior interfaces only
    elif zw.size == nlev + 1:
        zw_int = zw[1:nlev]
    else:
        raise SystemExit(f"z_interface_ref has {zw.size} entries for {nlev} levels")
    box = wet & (np.abs(lat) <= a.lat_halfwidth) & (lon >= a.lon_lo) & (lon < a.lon_hi)
    box_o = box.ravel(); w_o = area.ravel()
    Tf, Sf, uf, vf = (x.reshape(-1, nlev) for x in (T, S, u, v))
    Tf = np.where(np.abs(Tf) < 1e-6, np.nan, Tf)
    KHf = KH.reshape(box_o.size, -1)
    # K_H staggering: nlev+1 -> W levels incl. the surface (index k = top of
    # cell k); nlev -> assumed W level at the top of cell k (NEMO avt(jk)).
    if KHf.shape[1] == nlev + 1:
        KH_w = KHf[:, 1:nlev]          # interior interfaces 1..nlev-1
        kh_note = "K_H_diag has nlev+1 W levels; interior = [1:nlev]"
    elif KHf.shape[1] == nlev - 1:
        KH_w = KHf                     # interior interfaces, same as z_interface_ref
        kh_note = "K_H_diag has nlev-1 levels = the interior interfaces (z_interface_ref)"
    elif KHf.shape[1] == nlev:
        KH_w = KHf[:, 1:nlev]
        kh_note = "K_H_diag has nlev levels, read as W levels with index k at the top of cell k (NEMO avt convention); interior = [1:nlev]"
    else:
        raise SystemExit(f"K_H_diag shape {KHf.shape}")
    print(f"[ours] {kh_note}")
    S2_o = _shear2(uf, vf, zc)
    N2_o = _n2(np.nan_to_num(Tf), np.nan_to_num(Sf), zc, zw_int)
    wet_i = np.isfinite(Tf[:, :-1]) & np.isfinite(Tf[:, 1:])
    S2_o = np.where(wet_i, S2_o, np.nan); N2_o = np.where(wet_i, N2_o, np.nan); KH_w = np.where(wet_i, KH_w, np.nan)

    # ---- NEMO -------------------------------------------------------------
    def _nemo(grid, var):
        d = nc.Dataset(a.nemo_dir / f"{a.nemo_stem}_grid_{grid}.nc")
        x = np.ma.filled(np.ma.masked_invalid(d.variables[var][a.rec]), np.nan).astype(float)  # (nlev, y, x)
        la = np.asarray(d.variables["nav_lat"][:], float); lo = np.asarray(d.variables["nav_lon"][:], float) % 360.0
        dep = np.asarray(d.variables[[k for k in d.variables if k.startswith("depth") and not k.endswith("bounds")][0]][:], float)
        return np.transpose(x, (1, 2, 0)), la, lo, dep
    toN, laT, loT, gdept = _nemo("T", "to"); soN = _nemo("T", "so")[0]
    uoN, laU, loU, _ = _nemo("U", "uo"); voN, laV, loV, _ = _nemo("V", "vo")
    avtN, laW, loW, depthw = _nemo("W", "avt"); bn2N = _nemo("W", "bn2")[0]
    print(f"[NEMO] record {a.rec} (5-day means): to {toN.shape} uo {uoN.shape} avt {avtN.shape}; "
          f"deptht[:3]={gdept[:3]} depthw[:3]={depthw[:3]}")
    if not np.allclose(gdept[:a.n_levels], zc[:a.n_levels], rtol=0.02, atol=0.02):
        raise SystemExit(f"levels differ: NEMO {gdept[:4]} ours {zc[:4]}")
    nlN = gdept.size
    def _flat(x, la, lo):
        b = (np.abs(la) <= a.lat_halfwidth) & (lo >= a.lon_lo) & (lo < a.lon_hi) & np.isfinite(x[..., 0]) & (np.abs(x[..., 0]) > 0)
        return x.reshape(-1, x.shape[-1]), b.ravel(), np.cos(np.radians(la)).ravel()
    toF, boxT, wT = _flat(toN, laT, loT); soF = soN.reshape(-1, nlN)
    uoF, boxU, wU = _flat(uoN, laU, loU); voF, boxV, wV = _flat(voN, laV, loV)
    avtF = avtN.reshape(-1, nlN); bn2F = bn2N.reshape(-1, nlN)
    # NEMO shear on T columns: average the two U neighbours is a half-cell
    # shift; for a 20-deg box mean we take U-point and V-point clouds directly.
    S2u = np.diff(uoF, axis=-1) ** 2 / np.diff(gdept) ** 2
    S2v = np.diff(voF, axis=-1) ** 2 / np.diff(gdept) ** 2
    N2_N_recomp = _n2(np.nan_to_num(toF), np.nan_to_num(soF), gdept, depthw[1:nlN])
    wet_iN = np.isfinite(toF[:, :-1]) & np.isfinite(toF[:, 1:])
    N2_N_recomp = np.where(wet_iN, N2_N_recomp, np.nan)
    bn2_int = np.where(wet_iN, bn2F[:, 1:nlN], np.nan)      # avt/bn2(jk) at the top of cell jk
    avt_int = np.where(wet_iN, avtF[:, 1:nlN], np.nan)

    # ---- table ------------------------------------------------------------
    m = lambda f, w, b: _box_mean(f, w, b)
    S2_o_m = m(S2_o, w_o, box_o); N2_o_m = m(N2_o, w_o, box_o); KH_m = m(KH_w, w_o, box_o)
    KH_med = np.nanmedian(KH_w[box_o], axis=0)
    S2_N_m = m(S2u, wU, boxU) + m(S2v, wV, boxV)
    N2_N_m = m(N2_N_recomp, wT, boxT); bn2_m = m(bn2_int, wT, boxT)
    avt_m = m(avt_int, wT, boxT); avt_med = np.nanmedian(avt_int[boxT], axis=0)
    # Instability occupancy: ours = area fraction of box interfaces with N2 below the
    # EVD threshold in THIS snapshot (instantaneous T/S only -- on window means the
    # N2 of the mean is not the instability); NEMO = (mean avt - median avt)/rn_evd,
    # the window fraction EVD must have fired IF the median is the TKE-only value.
    unst_o = m(np.where(np.isfinite(N2_o), (N2_o < a.n2_threshold).astype(float), np.nan), w_o, box_o)
    evd_N = np.clip((avt_m - avt_med) / a.rn_evd, 0.0, 1.0)
    # Among OUR unstable interfaces: fraction where the model's own diagnosed K_H
    # exceeds 10 m2/s (EVD engaged on this state) and the median N2 (a value near
    # the threshold = instability generated after the solve; ~1e-5 = unmixed).
    _u = (N2_o < a.n2_threshold) & np.isfinite(N2_o) & box_o[:, None]
    with np.errstate(invalid="ignore"):
        kh_hi_unst = np.where(_u.sum(0) > 0, ((KH_w > 10.0) & _u).sum(0) / np.maximum(_u.sum(0), 1), np.nan)
    n2_med_unst = np.array([np.nanmedian(N2_o[_u[:, k], k]) if _u[:, k].any() else np.nan for k in range(N2_o.shape[1])])
    if a.use_mean_fields:
        unst_o[:] = np.nan; kh_hi_unst[:] = np.nan; n2_med_unst[:] = np.nan
    print(f"\n=== Box {a.lon_lo:.0f}-{a.lon_hi:.0f}E |lat|<={a.lat_halfwidth}: ours {int(box_o.sum())} cols, "
          f"NEMO {int(boxT.sum())} T-cols; interior interfaces (top of cell k+1) ===")
    print("Ri = box-mean N2 / box-mean S2 (a ratio of means, not the mean Ri). NEMO S2 from 5-day-MEAN u,v (lower bound); "
          "ours S2 from the 00 UTC snapshot. bn2 = NEMO's own file field; N2 = recomputed with the same TEOS-10 routine on both sides.")
    hdr = (f"{'k':>3}{'z_w m':>8} | {'S2 ours':>9}{'S2 NEMO':>9} | {'N2 ours':>9}{'N2 NEMO':>9}{'bn2file':>9} | "
           f"{'Ri ours':>8}{'Ri NEMO':>8} | {'KH ours':>9}{'KHmed':>9}{'avt NEMO':>9}{'avtmed':>9}")
    print(hdr)
    for k in range(min(a.n_levels, nlev - 1)):
        Ri_o = N2_o_m[k] / S2_o_m[k] if S2_o_m[k] > 0 else np.nan
        Ri_N = N2_N_m[k] / S2_N_m[k] if S2_N_m[k] > 0 else np.nan
        print(f"{k + 1:>3}{zw_int[k]:>8.2f} | {S2_o_m[k]:>9.2e}{S2_N_m[k]:>9.2e} | {N2_o_m[k]:>9.2e}{N2_N_m[k]:>9.2e}{bn2_m[k]:>9.2e} | "
              f"{Ri_o:>8.2f}{Ri_N:>8.2f} | {KH_m[k]:>9.2e}{KH_med[k]:>9.2e}{avt_m[k]:>9.2e}{avt_med[k]:>9.2e} | "
              f"{unst_o[k]:>10.3f}{evd_N[k]:>11.3f}{kh_hi_unst[k]:>11.3f}{n2_med_unst[k]:>11.2e}")
    # Turbocline depth (NEMO mldkz5: first depth where avt < 5e-4 m2/s), the
    # oracle's window MEAN of an hourly depth. Ours per snapshot from K_H_diag
    # on this state; a 4-clock-phase mean of it is the daily mean. Also NEMO's
    # depth-of-the-MEAN-avt for the reduction caveat (mean of depths != depth
    # of mean).
    # zdfmxl.F90:145-152: search from the bottom up to nlb10 (the first W level
    # below 10 m), so the shallowest level AT OR BELOW 10 m with avt < avt_c;
    # levels above 10 m never count; a column never below the threshold
    # reports its bottom depth (imld = mbkt+1). Dry interfaces mark the bottom.
    kz_thr = 5e-4
    def _turbocline(K_w, zw, box_mask, w):
        below = np.where(np.isfinite(K_w), K_w < kz_thr, True)         # dry = bottom
        below[:, zw < 10.0] = False                                    # nlb10 floor
        first = np.argmax(below, axis=1)                               # shallowest at/below 10 m
        never = ~below.any(axis=1)
        depth = np.where(never, zw[-1], zw[np.minimum(first, zw.size - 1)])
        depth = np.where(np.isfinite(K_w[:, 0]), depth, np.nan)
        q = np.nanpercentile(depth[box_mask], [10, 25, 50, 75, 90])
        print(f"    turbocline per-column quantiles 10/25/50/75/90%: {q[0]:.1f} {q[1]:.1f} {q[2]:.1f} {q[3]:.1f} {q[4]:.1f} m; "
              f"columns {int(np.isfinite(depth[box_mask]).sum())}")
        return _box_mean(depth[:, None], w, box_mask)[0]
    tc_o = _turbocline(KH_w, zw_int, box_o, w_o)
    tc_N_of_mean = _turbocline(avt_int, depthw[1:nlN], boxT, wT)
    d5 = nc.Dataset(a.nemo_dir / f"{a.nemo_stem}_grid_T.nc")
    if "mldkz5" in d5.variables:
        mk = np.ma.filled(np.ma.masked_invalid(d5.variables["mldkz5"][a.rec]), np.nan).astype(float).ravel()
        tc_N = _box_mean(mk[:, None], wT, boxT)[0]
        print(f"\n[turbocline avt<5e-4] ours (this snapshot) {tc_o:.2f} m | NEMO mldkz5 window mean {tc_N:.2f} m "
              f"| NEMO depth of the MEAN avt {tc_N_of_mean:.2f} m (reduction caveat)")
    else:
        print(f"\n[turbocline avt<5e-4] ours {tc_o:.2f} m | NEMO grid_T has no mldkz5; depth of the mean avt {tc_N_of_mean:.2f} m")

    # box-mean u profile too: is the undercurrent itself weaker?
    u_o = m(uf, w_o, box_o); u_N = m(uoF, wU, boxU)
    print(f"\n{'k':>3}{'z_c m':>8}{'u ours':>9}{'u NEMO':>9}   (zonal velocity box mean, m/s; ours 00 UTC, NEMO 5-day mean)")
    for k in range(0, min(a.n_levels + 20, nlev), 2):
        print(f"{k + 1:>3}{zc[k]:>8.1f}{u_o[k]:>9.3f}{u_N[k]:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

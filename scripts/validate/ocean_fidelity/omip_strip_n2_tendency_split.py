#!/usr/bin/env python
"""Split the tendency of stratification at one interface into DIFFUSIVE and RESIDUAL parts, ours vs NEMO.

For the equatorial every-step strip (our ``--trd-columns`` dump vs NEMO's 1ts strip output) and a
linear buoyancy proxy b = g (alpha T - beta S), the buoyancy jump across interface k
(between levels k and k+1), db_k = b_k - b_{k+1}, changes each step by

    total      = (db^n - db^{n-1}) / dt                       (observed)
    diffusive  = div_k - div_{k+1},  div_m = (F_{m-1} - F_m) / dz_m,  F_i = K_i (b_i - b_{i+1}) / dzh_i
    residual   = total - diffusive                            (advection + solar + surface + K33 + ...)

with the post-step state and the K used in that step's backward-Euler solve (exact for an implicit
solve on that K). Columns are PAIRED (same physical columns both models), so per-column ours-minus-NEMO
differences are reported, not medians of ratios.  Prints numbers only, no verdict.

NEMO side: the 1ts strip trend fields do not close against the 1ts T (per-step corr ~0) and the
offline K*grad estimator is ill-conditioned under EVD (K=100 on a float32-output, nearly mixed
column), so NEMO's split uses its HOURLY-MEAN trend file (trd1h: ttrd/strd_zdf = diffusive;
totad+ldf+qsr+bbl = residual), checked against the 1ts instantaneous change of the jump over each
hour.  Ours: offline backward-Euler diffusive from the dumped K, checked against ``ttrd_zdf_mean``.
"""
from __future__ import annotations

import argparse

import numpy as np

from legoesm import constants

ALPHA, BETA = 3.0e-4, 7.5e-4   # linear proxy, same as the omip_nemo strip probes


def interface_split(X, K, dz, dzh, dt, k):
    """X (nt, ncol, nz) post-step tracer/buoyancy, K (nt, ncol, nz-1) interface coefficient used in
    that step, dz (nt|1, ncol|1, nz), dzh (nt|1, ncol|1, nz-1).  Returns (total, diffusive) of
    d(X_k - X_{k+1})/dt for steps 1..nt-1, each (nt-1, ncol)."""
    F = K * (X[..., :-1] - X[..., 1:]) / dzh                    # downward flux at interfaces 0..nz-2
    Fp = np.concatenate([np.zeros_like(F[..., :1]), F], -1)      # F_{-1} = 0 (surface flux is residual)
    div = (Fp[..., :-1] - Fp[..., 1:]) / dz[..., :-1]            # levels 0..nz-2
    dif = div[..., k] - div[..., k + 1]
    d = X[..., k] - X[..., k + 1]
    return (d[1:] - d[:-1]) / dt, dif[1:]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ours", required=True, help="run dir with snapshot_day0001.npz + snapshot_final.npz")
    p.add_argument("--nemo-prefix", required=True, help=".../ORCA1_1ts_..._eqs1ts_ (T.nc appended)")
    p.add_argument("--nemo-trd1h", required=True, help="RUN_DT150_TRD .../ORCA1_1h_*_trd1h_T.nc")
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--k", type=int, default=8, help="interface index (8 = 12.8 m)")
    p.add_argument("--hours", type=int, default=16)
    p.add_argument("--h0", type=int, default=4, help="attribution window start hour")
    p.add_argument("--h1", type=int, default=12, help="attribution window end hour")
    a = p.parse_args(argv)
    import netCDF4 as nc

    g = constants.g
    zs = [np.load(a.ours + f) for f in ("/snapshot_day0001.npz", "/snapshot_final.npz")]
    cj, ci = zs[0]["col_j"], zs[0]["col_i"]
    sj, si = cj - 180, ci - 153
    ok = (sj >= 0) & (sj < 13) & (si >= 0) & (si < 31)
    O = {key: np.concatenate([z[key] for z in zs]).astype(np.float64)[:, ok] for key in ("col_T", "col_S", "col_K")}
    nz = O["col_T"].shape[-1]
    zi = np.concatenate([[0.0], np.abs(zs[0]["z_interface_ref"])])[: nz + 1]
    zc = np.abs(zs[0]["z_center_ref"])[:nz]
    dz_o, dzh_o = np.diff(zi)[None, None], np.diff(zc)[None, None]

    T = nc.Dataset(a.nemo_prefix + "T.nc")
    ns = min(O["col_T"].shape[0], T["votemper"].shape[0])
    f = lambda v: np.ma.filled(v, np.nan).astype(np.float64)
    pick = lambda ds, v, kmax: f(ds[v][:ns, :kmax])[:, :, sj[ok], si[ok]].transpose(0, 2, 1)
    NT, NS = pick(T, "votemper", nz), pick(T, "vosaline", nz)
    for name, arr in (("our col_T", O["col_T"]), ("NEMO T", NT)):
        print(f"[dtype] {name} {arr.dtype}")
    O = {kk: v[:ns] for kk, v in O.items()}
    wet = np.all(np.isfinite(NT[0, :, : a.k + 3]), -1) & np.all(O["col_T"][0, :, : a.k + 3] != 0, -1)
    print(f"columns {ok.sum()} wet {wet.sum()} steps {ns} interface k={a.k} z={zi[a.k + 1]:.2f} m")

    k = a.k
    # --- instrument checks ---
    _, difT_o = interface_split(O["col_T"], O["col_K"], dz_o, dzh_o, a.dt, k)
    Fo = O["col_K"] * (O["col_T"][..., :-1] - O["col_T"][..., 1:]) / dzh_o
    Fo = np.concatenate([np.zeros_like(Fo[..., :1]), Fo], -1)
    divT_o = (Fo[..., :-1] - Fo[..., 1:]) / dz_o[..., :-1]
    n1 = int(zs[0]["ttrd_zdf_n_steps"])
    trd = zs[0]["ttrd_zdf_mean"][cj[ok], ci[ok], :]
    for lev in (k, k + 1):
        off = divT_o[:n1, wet, lev].mean(0)
        print(f"[check ours] level {lev}: offline day-1 mean diffusive dT/dt median {np.median(off):+.3e} "
              f"vs model ttrd_zdf_mean {np.median(trd[wet, lev]):+.3e}  (per-col corr "
              f"{np.corrcoef(off, trd[wet, lev])[0, 1]:.3f}, per-col |rel err| p95 "
              f"{np.percentile(np.abs(off - trd[wet, lev]) / np.maximum(np.abs(trd[wet, lev]), 1e-12), 95):.1e})")
    # --- buoyancy split, paired per column, hourly sums of d(db) ---
    bo = g * (ALPHA * O["col_T"] - BETA * O["col_S"])
    bn = g * (ALPHA * NT - BETA * NS)
    to, do = interface_split(bo, O["col_K"], dz_o, dzh_o, a.dt, k)
    ro = to - do
    H = nc.Dataset(a.nemo_trd1h)
    hf = lambda v: np.where(np.abs(f(H[v][:, k:k + 2])) < 1e15, f(H[v][:, k:k + 2]), np.nan)[:, :, cj[ok], ci[ok] - 1]
    jump = lambda x: x[:, 0] - x[:, 1]                           # (nh, ncol)
    dif_n = g * (ALPHA * jump(hf("ttrd_zdf")) - BETA * jump(hf("strd_zdf"))) * 3600.0
    res_n = g * (ALPHA * sum(jump(hf(v)) for v in ("ttrd_totad", "ttrd_ldf", "ttrd_qsr", "ttrd_bbl"))
                 - BETA * sum(jump(hf(v)) for v in ("strd_totad", "strd_ldf", "strd_bbl"))) * 3600.0
    bj = lambda tv, sv: g * (ALPHA * jump(hf(tv)) - (BETA * jump(hf(sv)) if sv else 0.0)) * 3600.0
    res_parts = {"totad": bj("ttrd_totad", "strd_totad"), "qsr": bj("ttrd_qsr", None),
                 "ldf+bbl": bj("ttrd_ldf", "strd_ldf") + bj("ttrd_bbl", "strd_bbl")}
    spd = int(round(3600.0 / a.dt))
    dbo, dbn = (bo[..., k] - bo[..., k + 1])[:, wet], (bn[..., k] - bn[..., k + 1])[:, wet]
    print("hourly change of the buoyancy jump db (m/s2), median over wet columns; ours / NEMO / paired diff")
    print(" hr  | db ours  db NEMO ratio | dDIFF ours  NEMO  o-n | dRESID ours  NEMO  o-n | NEMO check: 1ts d(db) vs trd1h sum")
    for h in range(min(a.hours, (ns - 1) // spd)):
        # row r = step r+1; to/do[r] = change into step r+2; hour h = steps h*spd+1 .. (h+1)*spd
        r0, r1 = max(h * spd - 1, 0), (h + 1) * spd - 1
        s = slice(r0, r1)
        sd = lambda x: x[s][:, wet].sum(0) * a.dt
        Do, Ro = sd(do), sd(ro)
        Dn, Rn = dif_n[h, wet], res_n[h, wet]
        e = r1
        chk_n = dbn[r1] - dbn[r0]
        md = lambda x: float(np.median(x))
        print(f"h{h + 1:02d} | {md(dbo[e]):.2e} {md(dbn[e]):.2e} {md(dbo[e] / dbn[e]):5.2f} | "
              f"{md(Do):+.2e} {md(Dn):+.2e} {md(Do - Dn):+.2e} | {md(Ro):+.2e} {md(Rn):+.2e} {md(Ro - Rn):+.2e} | "
              f"{md(chk_n):+.2e} {md(Dn + Rn):+.2e} corr {np.corrcoef(chk_n, Dn + Rn)[0, 1]:.2f}")

    # Medians do not add across hours; the attribution is the per-column CUMULATIVE split over a window.
    for h0, h1 in ((a.h0, a.h1), (a.h0, 8), (8, a.h1)):
        r0, r1 = max(h0 * spd - 1, 0), h1 * spd - 1
        Do_c, Ro_c = do[r0:r1][:, wet].sum(0) * a.dt, ro[r0:r1][:, wet].sum(0) * a.dt
        Dn_c, Rn_c = dif_n[h0:h1, wet].sum(0), res_n[h0:h1, wet].sum(0)
        gap = (dbo[r1] - dbo[r0]) - (dbn[r1] - dbn[r0])
        dd, dr = Do_c - Dn_c, Ro_c - Rn_c
        print(f"[window h{h0:02d}-h{h1:02d}] change of db: ours mean {np.mean(dbo[r1] - dbo[r0]):+.2e} NEMO mean "
              f"{np.mean(dbn[r1] - dbn[r0]):+.2e}; gap (o-n) mean {np.mean(gap):+.2e} median {np.median(gap):+.2e}")
        print(f"    gap split: DIFFUSIVE mean {np.mean(dd):+.2e} median {np.median(dd):+.2e} | RESIDUAL mean "
              f"{np.mean(dr):+.2e} median {np.median(dr):+.2e} | cols |dd|>|dr|: {np.mean(np.abs(dd) > np.abs(dr)):.2f}"
              f" | |.|-weighted DIFF share {np.abs(dd).sum() / (np.abs(dd).sum() + np.abs(dr).sum()):.2f}"
              f" | rms dd {np.sqrt(np.mean(dd ** 2)):.2e} dr {np.sqrt(np.mean(dr ** 2)):.2e}")
        clo = (dbn[r1] - dbn[r0]) - (Dn_c + Rn_c)
        print(f"    NEMO per-column cumulative closure: median |1ts change - trd1h sum| / |1ts change| "
              f"{np.median(np.abs(clo) / np.maximum(np.abs(dbn[r1] - dbn[r0]), 1e-12)):.3f}")
        # Shortwave absorption matches NEMO at 0-25 m (ratio 0.99-1.00), so the RESIDUAL gap is the
        # gap in advection (NEMO totad = horizontal + vertical) + ldf + bbl.
        print("    NEMO RESIDUAL parts (mean, rms over cols): " + " | ".join(
            f"{nm} {np.mean(v[h0:h1, wet].sum(0)):+.2e} rms {np.sqrt(np.mean(v[h0:h1, wet].sum(0) ** 2)):.2e}"
            for nm, v in res_parts.items()))
        print(f"    advective gap (ours RES - NEMO qsr) - NEMO totad: mean "
              f"{np.mean(Ro_c - res_parts['qsr'][h0:h1, wet].sum(0) - res_parts['totad'][h0:h1, wet].sum(0)):+.2e}"
              f" rms {np.sqrt(np.mean((Ro_c - res_parts['qsr'][h0:h1, wet].sum(0) - res_parts['totad'][h0:h1, wet].sum(0)) ** 2)):.2e}"
              f" vs rms total gap {np.sqrt(np.mean(gap ** 2)):.2e}")
        print(f"    components: ours DIFF {np.mean(Do_c):+.2e} RES {np.mean(Ro_c):+.2e} | NEMO DIFF {np.mean(Dn_c):+.2e} "
              f"RES {np.mean(Rn_c):+.2e}")


if __name__ == "__main__":
    main()

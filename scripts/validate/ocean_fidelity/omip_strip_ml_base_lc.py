"""Profile-shape and Langmuir-depth comparison, ours vs NEMO, equatorial strip
(probe; pre-registered in omip_nemo/_tkereslc_rerun_PREREG.md, 2026-10-12).

Part 1 (hourly, h0-h48, paired per column):
  zstep  = depth of the interface with the largest N2 in the top ZTOP m
           (linear-EOS proxy a=3e-4, b=7.5e-4, identical on both sides, as
           omip_strip_n2_seed / omip_strip_xplant_score);
  zturb  = turbocline: first interface (from the top) whose APPLIED tracer
           diffusivity falls below AVT_C (NEMO zdfmxl avt_c = 5e-4 m2/s;
           ours col_K, NEMO avt incl. EVD + IWM).
  Printed: medians ours | NEMO, paired median ours-NEMO with IQR, and the
  share of columns whose zstep sits at the K12 interface (the 12.8 m one of
  the 10-10 seed probe).
Part 2 (Langmuir, window H0-H1): from each model's own Langmuir increment
  (ours col_e_lc; NEMO en_lc_k - en_in_k), zdftke.F90:358-370 applies
  dt*zus3*(rn_lc*sin(pi z/h_lc))**3/h_lc on interfaces with z < h_lc, so
  h_lc = depth of the first interface below a positive run whose increment
  is zero (exact on the w-ladder).  Inverting the increment at the
  shallowest active interface gives zus3, and taum_implied ~ zus**2 (the
  zcsd constant cancels in the ours/NEMO ratio).  Instrument check (NEMO):
  implied taum/|utau| must be >= 1 and close to 1 where |vtau| is small.
  Depths are the reference gdepw_1d (NEMO uses gdepw(Kmm); z* offset tiny).
No verdict printed.
usage: omip_strip_ml_base_lc.py OUR_RUN NEMO_EN2_DIR [H0 H1]
"""
import sys
import numpy as np
from legoesm import constants

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omip_strip_tke_ledger as L   # noqa: E402

ALPHA, BETA, DT = 3.0e-4, 7.5e-4, 150.0   # coeff-ok: linear-EOS proxy shared with omip_strip_xplant_score
RN_LC = 0.25                              # coeff-ok: ORCA1 namelist_cfg rn_lc (both cards)
AVT_C = 5.0e-4                            # coeff-ok: NEMO zdfmxl avt_c turbocline threshold
ZTOP = 40.0
PREFIX = "ORCA1_1ts_20000101_20000102_eqs1ts_"


def step_depth(T, S, zt, ztop):
    """depth of the max linear-proxy N2 interface among those above ztop.
    T,S (..., nlev); interface i lies between T-levels i, i+1 at the
    midpoint depth.  Returns (..., ) depth [m] and the interface index."""
    dz = np.diff(zt)
    n2 = constants.g * (ALPHA * (T[..., :-1] - T[..., 1:]) - BETA * (S[..., :-1] - S[..., 1:])) / dz
    zi = 0.5 * (zt[:-1] + zt[1:])
    n2 = np.where(zi <= ztop, n2, -np.inf)
    i = np.argmax(n2, axis=-1)
    return zi[i], i


def turbocline(K, zw, avt_c):
    """first interface (top-down) with K < avt_c; deepest interface if none."""
    below = K < avt_c
    i = np.where(np.any(below, -1), np.argmax(below, -1), K.shape[-1] - 1)
    return zw[i]


def lc_depth(inc, zw):
    """h_lc from a Langmuir increment profile (..., nk) on interface depths
    zw: depth of the first zero-increment interface after the positive run
    from the top.  NaN when no positive increment (zus3 = 0 or h_lc above
    the first interface); deepest interface + one spacing if positive all
    the way down.  Also returns the index of the first positive interface."""
    pos = inc > 0
    any_pos = np.any(pos, -1)
    first = np.argmax(pos, -1)
    after = pos | (np.arange(inc.shape[-1]) < first[..., None])
    stop = ~after
    has_stop = np.any(stop, -1)
    j = np.argmax(stop, -1)
    deep = zw[-1] + (zw[-1] - zw[-2])
    h = np.where(has_stop, zw[np.minimum(j, len(zw) - 1)], deep)
    return np.where(any_pos, h, np.nan), first


def implied_us(inc, h, first, zw):
    """zus from the increment at the first active interface: inc = dt*zus^3*
    (rn_lc sin(pi z/h))^3/h  ->  zus = (inc*h/(dt*(rn_lc sin)^3))^(1/3)."""
    a = np.take_along_axis(inc, first[..., None], -1)[..., 0]
    z = zw[first]
    w = RN_LC * np.sin(np.pi * z / h)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.cbrt(a * h / (DT * w ** 3))


def _q(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "   nan [   nan,   nan] n=0"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:6.2f} [{q[0]:6.2f},{q[2]:6.2f}] n={x.size}"


def main(argv):
    h0, h1 = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (27, 38)
    import netCDF4 as nc
    a = [np.load(f"{argv[0]}/{fn}") for fn in ("snapshot_day0001.npz", "snapshot_final.npz")]
    g = lambda key: np.concatenate([x[key] for x in a]).astype(np.float64)
    cj, ci = a[0]["col_j"], a[0]["col_i"]
    sj, si = cj - L.J0, ci - L.I0
    keep = (sj >= L.BUF) & (sj < 13 - L.BUF) & (si >= L.BUF) & (si < 31 - L.BUF)
    To, So, Ko = g("col_T")[:, keep], g("col_S")[:, keep], g("col_K")[:, keep]
    LCo = g("col_e_lc")[:, keep]
    sj, si = sj[keep], si[keep]
    nlev = To.shape[2]
    ds = {f: nc.Dataset(f"{argv[1]}/{PREFIX}{f}.nc") for f in ("T", "W", "2D")}

    def n(f, v, sl):
        x = np.asarray(ds[f][v][(slice(None), sl)], dtype=np.float64)
        x = x[..., sj, si].transpose(0, 2, 1)
        x[np.abs(x) > 1e15] = np.nan
        return x
    Tn, Sn = n("T", "votemper", slice(0, nlev)), n("T", "vosaline", slice(0, nlev))
    Kn = n("W", "avt", slice(1, nlev))                         # file w i+1 = our interface i
    LCn = n("W", "en_lc_k", slice(1, nlev)) - n("W", "en_in_k", slice(1, nlev))
    utau = np.asarray(ds["2D"]["utau"][:], dtype=np.float64)[:, sj, si]
    m = nc.Dataset(L.MESH)
    zt = np.asarray(m["gdept_1d"][0], dtype=np.float64)[:nlev]
    zw = np.asarray(m["gdepw_1d"][0], dtype=np.float64)[1:nlev]
    nt = min(To.shape[0], Tn.shape[0])
    print(f"steps ours {To.shape[0]} NEMO {Tn.shape[0]} -> using {nt}; dtype {To.dtype}/{Tn.dtype}")
    wet = np.all(np.isfinite(Tn[0, :, :12]), 1) & np.all(To[0, :, :12] != 0, 1)
    k12 = int(np.argmin(np.abs(0.5 * (zt[:-1] + zt[1:]) - 12.8)))
    print(f"{wet.sum()} wet interior cols; K12 interface {k12} at {0.5 * (zt[k12] + zt[k12 + 1]):.2f} m")
    print("\nPart 1  h | zstep ours | NEMO | o-N median [IQR] | share at K12 o/N || zturb ours | NEMO | o-N median [IQR]")
    for h in range(0, 49):
        s = max(h * 24 - 1, 0)
        if s >= nt:
            break
        zo, io = step_depth(To[s, wet], So[s, wet], zt, ZTOP)
        zn, inn = step_depth(Tn[s, wet], Sn[s, wet], zt, ZTOP)
        to = turbocline(Ko[s, wet], zw, AVT_C)
        tn = turbocline(Kn[s, wet], zw, AVT_C)
        print(f"h{h:02d} | {np.median(zo):6.2f} | {np.median(zn):6.2f} | {_q(zo - zn)} | "
              f"{np.mean(io == k12):.2f}/{np.mean(inn == k12):.2f} || "
              f"{np.median(to):6.2f} | {np.median(tn):6.2f} | {_q(to - tn)}")
    s0, s1 = h0 * 24, min(h1 * 24, nt)
    hlo, fo = lc_depth(LCo[s0:s1, wet], zw)
    hln, fn = lc_depth(LCn[s0:s1, wet], zw)
    uo = implied_us(LCo[s0:s1, wet], hlo, fo, zw)
    un = implied_us(LCn[s0:s1, wet], hln, fn, zw)
    print(f"\nPart 2  Langmuir, steps {s0}..{s1 - 1} (h{h0}-h{h1}), all wet col x step samples")
    print(f"  samples with a positive increment: ours {np.mean(np.isfinite(hlo)):.3f}  NEMO {np.mean(np.isfinite(hln)):.3f}")
    print(f"  h_lc [m] ours {_q(hlo)}")
    print(f"  h_lc [m] NEMO {_q(hln)}")
    print(f"  h_lc ours-NEMO (paired) {_q(hlo - hln)}")
    print(f"  share h_lc <= 12.8 m: ours {np.nanmean(hlo <= 12.8):.3f}  NEMO {np.nanmean(hln <= 12.8):.3f}")
    print(f"  share (ours h_lc < 12.8 m AND NEMO h_lc > 12.8 m) among NEMO>12.8: "
          f"{np.nanmean((hlo < 12.8)[hln > 12.8]):.3f}")
    print(f"  implied taum ratio ours/NEMO (zus^2) {_q((uo / un) ** 2)}")
    tau_n = un ** 2 / (2.0 * L_CSD())
    print(f"  CHECK NEMO implied taum / |utau| (>=1, ~1 if vtau small) {_q(tau_n / np.abs(utau[s0:s1, wet]))}")
    print(f"  NEMO implied taum [N/m2] {_q(tau_n)}  ours {_q(uo ** 2 / (2.0 * L_CSD()))}")


def L_CSD():
    """zcsd = 0.5*0.016**2/(rhoa*cdrag), zdftke.F90:243 (rhoa=1.22, cdrag=1.5e-3,
    zdftke.F90 locals) -- only used to express the NEMO check in N/m2."""
    return 0.5 * 0.016 ** 2 / (1.22 * 1.5e-3)  # coeff-ok: zdftke.F90:243 zcsd literal constants


if __name__ == "__main__":
    main(sys.argv[1:])

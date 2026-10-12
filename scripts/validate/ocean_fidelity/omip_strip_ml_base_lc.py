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
  is zero (exact on the w-ladder; censored above the first / below the last
  saved interface are reported separately and never inverted).  Inverting the increment at the
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
ZTURB = 100.0
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
    """first interface (top-down) with K < avt_c among the interfaces given;
    +inf (censored: deeper than the window) if none crosses.  Callers pass
    only the validated window (codex 2026-10-12: searching dry rows turned
    ours' zeros vs NEMO's NaNs into a fake 70 vs 100 m difference)."""
    below = K < avt_c
    return np.where(np.any(below, -1), zw[np.argmax(below, -1)], np.inf)


def lc_depth(inc, zw):
    """h_lc from a Langmuir increment profile (..., nk) on interface depths
    zw: depth of the first zero-increment interface after the positive run
    from the top.  Censored cases (codex 2026-10-12): no positive increment
    -> h_lc <= zw[0] (NEMO's strict z < h test leaves every saved interface
    untouched) OR zus3 = 0 (zero stress, zdftke.F90:359-362): returned as
    zw[0] with status -1 and excluded from depth shares by the caller; positive down to the last
    saved interface -> h_lc beyond the dump, returned as +inf, status +1.
    An interior zero inside the positive run -> status 2 (cutoff unknown).
    Any NaN in the profile (diagnostic unavailable) -> status -3, h NaN.
    Resolution-limited cutoffs (see resolution_limited) are status 3.
    Only status 0 is an exact depth.  Also returns the first positive index."""
    pos = inc > 0
    any_pos = np.any(pos, -1)
    first = np.argmax(pos, -1)
    after = pos | (np.arange(inc.shape[-1]) < first[..., None])
    stop = ~after
    has_stop = np.any(stop, -1)
    j = np.argmax(stop, -1)
    h = np.where(has_stop, zw[np.minimum(j, len(zw) - 1)], np.inf)
    h = np.where(any_pos, h, zw[0])
    # a positive increment BELOW the first stop = an interior zero inside the
    # run (float32 drop-out): the cutoff is unknown -> status 2, never exact
    gap = has_stop & np.any(pos & (np.arange(inc.shape[-1]) > j[..., None]), -1)
    status = np.where(~any_pos, -1, np.where(gap, 2, np.where(has_stop, 0, 1)))
    missing = np.any(np.isnan(inc), -1)                 # diagnostic unavailable -> status -3
    status = np.where(missing, -3, status)
    h = np.where(missing, np.nan, h)
    return h, first, status


def resolution_limited(inc, h, first, status, zw, e_in, eps):
    """True where an exact-looking cutoff could be a rounding drop-out: if
    the true h_lc were the NEXT interface, the increment at the observed
    first-zero interface would be dt*zus^3*(rn_lc sin(pi z0/z1))^3/z1;
    when that is below 2 ulp of the stored entry TKE (eps*e_in, NEMO writes
    float32) the dump cannot tell a zero from it (codex 2026-10-12)."""
    j = np.searchsorted(zw, np.where(status == 0, h, zw[0]))
    last = j >= len(zw) - 1           # cutoff at the last saved interface: no deeper one to test
    jj = np.clip(j, 0, len(zw) - 2)
    us = implied_us(inc, np.where(status == 0, h, np.nan), first, zw)
    z0 = zw[np.clip(j, 0, len(zw) - 1)]
    z1 = zw[jj + 1]
    pred = DT * us ** 3 * (RN_LC * np.sin(np.pi * z0 / z1)) ** 3 / z1
    e0 = np.take_along_axis(e_in, np.clip(j, 0, len(zw) - 1)[..., None], -1)[..., 0]
    # every positive increment in the run must sit well above the dump's
    # rounding of the entry TKE, or the run itself (and the zus inverted from
    # it) may be rounding noise (codex round 6)
    noisy = np.any((inc > 0) & (inc < 10.0 * eps * np.abs(e_in)), -1)
    with np.errstate(invalid="ignore"):
        return (status == 0) & (last | noisy | (pred < 2.0 * eps * np.abs(e0)))


def implied_us(inc, h, first, zw):
    """zus from the increment at the first active interface: inc = dt*zus^3*
    (rn_lc sin(pi z/h))^3/h  ->  zus = (inc*h/(dt*(rn_lc sin)^3))^(1/3)."""
    a = np.take_along_axis(inc, first[..., None], -1)[..., 0]
    z = zw[first]
    w = RN_LC * np.sin(np.pi * z / h)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.cbrt(a * h / (DT * w ** 3))


def profile_us(inc, h, zw, sin_min=0.3):
    """zus inverted at EVERY active interface (sin(pi z/h) >= sin_min):
    returns (zus at the max-increment interface, max/min ratio of the
    per-level zus).  The ratio is 1 for an increment with NEMO's shape;
    >1 flags a shape/level mismatch (GLM 2026-10-12: a scalar or level port
    defect is degenerate with taum in a one-level inversion)."""
    s = np.sin(np.pi * zw / h[..., None])
    act = (inc > 0) & (zw < h[..., None]) & (s >= sin_min)
    with np.errstate(invalid="ignore", divide="ignore"):
        us = np.cbrt(inc * h[..., None] / (DT * (RN_LC * s) ** 3))
    us = np.where(act, us, np.nan)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(us, -1) / np.nanmin(us, -1)
    k = np.argmax(np.where(act, inc, -np.inf), -1)
    top = np.take_along_axis(us, k[..., None], -1)[..., 0]
    return top, spread


def gap_share(inc):
    """share of profiles whose positive run has an interior zero (a float32
    drop-out inside the run would make lc_depth read h_lc too shallow)."""
    pos = inc > 0
    last = inc.shape[-1] - 1 - np.argmax(pos[..., ::-1], -1)
    first = np.argmax(pos, -1)
    k = np.arange(inc.shape[-1])
    inside = (k >= first[..., None]) & (k <= last[..., None])
    return np.mean(np.any(inside & ~pos, -1)[np.any(pos, -1)])


def share(x, cond):
    """mean of cond over samples where x is not NaN (+inf = censored deep counts)."""
    ok = ~np.isnan(x)
    return float(np.mean(cond[ok])) if ok.any() else float("nan")


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
    LCo, EIo = g("col_e_lc")[:, keep], g("col_e_in")[:, keep]
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
    Kkn = n("W", "avt_k", slice(1, nlev))                      # NEMO closure only (no EVD, no IWM)
    EIn = n("W", "en_in_k", slice(1, nlev))
    LCn = n("W", "en_lc_k", slice(1, nlev)) - EIn
    utau = np.asarray(ds["2D"]["utau"][:], dtype=np.float64)[:, sj, si]
    utau[np.abs(utau) > 1e15] = np.nan                        # EN2 writes utau as all-fill: check then reads n=0
    m = nc.Dataset(L.MESH)
    zt = np.asarray(m["gdept_1d"][0], dtype=np.float64)[:nlev]
    zw = np.asarray(m["gdepw_1d"][0], dtype=np.float64)[1:nlev]
    nt = min(To.shape[0], Tn.shape[0])
    print(f"steps ours {To.shape[0]} NEMO {Tn.shape[0]} -> using {nt}; dtype {To.dtype}/{Tn.dtype}")
    ktop = int(np.searchsorted(zt, ZTOP)) + 2          # T-levels feeding interfaces above ZTOP
    kw = int(np.searchsorted(zw, ZTURB, side="right"))  # turbocline window = interfaces with zw <= ZTURB
    wet = (np.all(np.isfinite(Tn[:nt, :, :ktop]) & np.isfinite(Sn[:nt, :, :ktop]), (0, 2))
           & np.all(np.isfinite(Kn[:nt, :, :kw]), (0, 2))
           & np.all((To[:nt, :, :ktop] != 0) & np.isfinite(To[:nt, :, :ktop]) & np.isfinite(So[:nt, :, :ktop]), (0, 2))
           & np.all(np.isfinite(Ko[:nt, :, :kw]), (0, 2)))
    k12 = int(np.argmin(np.abs(0.5 * (zt[:-1] + zt[1:]) - 12.8)))
    print(f"{wet.sum()} wet interior cols; K12 interface {k12} at {0.5 * (zt[k12] + zt[k12 + 1]):.2f} m")
    print("\nPart 1  h | zstep ours | NEMO | o-N median [IQR] | share at K12 o/N || zturb ours | NEMO | o-N median [IQR] | censored share")
    for h in range(0, 49):
        s = max(h * 24 - 1, 0)
        if s >= nt:
            break
        zo, io = step_depth(To[s, wet], So[s, wet], zt, ZTOP)
        zn, inn = step_depth(Tn[s, wet], Sn[s, wet], zt, ZTOP)
        to = turbocline(Ko[s, wet, :kw], zw[:kw], AVT_C)
        tn = turbocline(Kn[s, wet, :kw], zw[:kw], AVT_C)
        cens = f"{np.mean(np.isinf(to)):.2f}/{np.mean(np.isinf(tn)):.2f}"
        to, tn = np.where(np.isinf(to), np.nan, to), np.where(np.isinf(tn), np.nan, tn)
        print(f"h{h:02d} | {np.median(zo):6.2f} | {np.median(zn):6.2f} | {_q(zo - zn)} | "
              f"{np.mean(io == k12):.2f}/{np.mean(inn == k12):.2f} || "
              f"{np.nanmedian(to):6.2f} | {np.nanmedian(tn):6.2f} | {_q(to - tn)} | >{ZTURB:.0f}m o/N {cens}")
    print(f"\nPart 1b  h | turbocline vs NEMO CLOSURE avt_k: ours | NEMO avt_k | o-N [IQR] || median K at 12.8 / 17.9 m: ours | avt_k | avt  [1e-4 m2/s]")
    k128, k179 = int(np.argmin(np.abs(zw - 12.8))), int(np.argmin(np.abs(zw - 17.9)))
    for h in range(0, 49, 2):
        s = max(h * 24 - 1, 0)
        if s >= nt:
            break
        to = turbocline(Ko[s, wet, :kw], zw[:kw], AVT_C)
        tk = turbocline(Kkn[s, wet, :kw], zw[:kw], AVT_C)
        to, tk = np.where(np.isinf(to), np.nan, to), np.where(np.isinf(tk), np.nan, tk)
        km = lambda K, k: np.median(K[s, wet, k]) * 1e4
        print(f"h{h:02d} | {np.nanmedian(to):6.2f} | {np.nanmedian(tk):6.2f} | {_q(to - tk)} || "
              f"{km(Ko, k128):7.2f} {km(Kkn, k128):7.2f} {km(Kn, k128):7.2f} | {km(Ko, k179):7.2f} {km(Kkn, k179):7.2f} {km(Kn, k179):7.2f}")
    s0, s1 = h0 * 24, min(h1 * 24, nt)
    hlo, fo, sto = lc_depth(LCo[s0:s1, wet], zw)
    hln, fn, stn = lc_depth(LCn[s0:s1, wet], zw)
    sto = np.where(resolution_limited(LCo[s0:s1, wet], hlo, fo, sto, zw, EIo[s0:s1, wet],
                                      np.finfo(np.float64).eps), 3, sto)
    stn = np.where(resolution_limited(LCn[s0:s1, wet], hln, fn, stn, zw, EIn[s0:s1, wet],
                                      np.finfo(np.float32).eps), 3, stn)
    hlo = np.where(sto == -1, np.nan, hlo); hln = np.where(stn == -1, np.nan, hln)   # no increment: h unknown
    both = (sto == 0) & (stn == 0)
    hx_o = np.where(sto == 0, hlo, np.nan); hx_n = np.where(stn == 0, hln, np.nan)
    uo = implied_us(LCo[s0:s1, wet], hx_o, fo, zw)
    un = implied_us(LCn[s0:s1, wet], hx_n, fn, zw)
    print(f"\nPart 2  Langmuir, steps {s0}..{s1 - 1} (h{h0}-h{h1}), all wet col x step samples")
    for lab, st in (("ours", sto), ("NEMO", stn)):
        print(f"  {lab} status share: missing {np.mean(st == -3):.3f}  no increment (h<=first iface or zero stress) {np.mean(st == -1):.3f}  exact {np.mean(st == 0):.3f}  beyond dump {np.mean(st == 1):.3f}  gap {np.mean(st == 2):.3f}  resolution-limited {np.mean(st == 3):.3f}")
    print(f"  h_lc [m] (exact only) ours {_q(hx_o)}")
    print(f"  h_lc [m] (exact only) NEMO {_q(hx_n)}")
    print(f"  h_lc ours-NEMO (paired, both exact) {_q(np.where(both, hlo - hln, np.nan))}")
    hlo = np.where(np.isin(sto, (2, 3)), np.nan, hlo); hln = np.where(np.isin(stn, (2, 3)), np.nan, hln)   # cutoff unknown
    print(f"  share h_lc <= 12.8 m (censored included, gaps/missing excluded): ours {share(hlo, hlo <= 12.8):.3f}  NEMO {share(hln, hln <= 12.8):.3f}")
    print(f"  share ours h_lc < 12.8 m among samples with NEMO h_lc > 12.8 m: "
          f"{share(hlo[hln > 12.8], hlo[hln > 12.8] < 12.8):.3f}  (n={np.sum(~np.isnan(hlo) & (hln > 12.8))})")
    print("  NOTE: 'implied amplitude' = zus^2 from inverting each model's own increment with NEMO's formula;"
          " a z-independent port defect is degenerate with taum here (GLM 2026-10-12) -- not a measured stress.")
    print(f"  implied amplitude ratio ours/NEMO (zus^2, first active interface; monitor only) {_q((uo / un) ** 2)}")
    po, spo = profile_us(LCo[s0:s1, wet], hx_o, zw)
    pn, spn = profile_us(LCn[s0:s1, wet], hx_n, zw)
    amp = (po / pn) ** 2
    print(f"  implied amplitude ratio ours/NEMO (zus^2, max-increment interface, both exact) {_q(amp)}")
    print(f"  per-level zus spread max/min (1 = NEMO shape): ours {_q(spo)}  NEMO {_q(spn)}")
    print(f"  share spread <= 1.01: ours {np.nanmean(spo[np.isfinite(spo)] <= 1.01):.3f}  NEMO {np.nanmean(spn[np.isfinite(spn)] <= 1.01):.3f}")
    ok = np.isfinite(amp)
    if ok.sum() > 30:
        edges = np.percentile(hln[ok], [0, 33.3, 66.7, 100])
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = ok & (hln >= lo) & (hln <= hi)
            print(f"  amplitude ratio for NEMO h_lc in [{lo:5.1f},{hi:5.1f}] m: {_q(amp[b])}")
    print(f"  positive run with interior zero (float32 drop-out): ours {gap_share(LCo[s0:s1, wet]):.4f}  NEMO {gap_share(LCn[s0:s1, wet]):.4f}")
    tau_n = un ** 2 / (2.0 * L_CSD())
    print(f"  CHECK NEMO implied taum / |utau| (>=1, ~1 if vtau small) {_q(tau_n / np.abs(utau[s0:s1, wet]))}")
    print(f"  NEMO implied taum [N/m2] {_q(tau_n)}  ours implied {_q(uo ** 2 / (2.0 * L_CSD()))}")


def L_CSD():
    """zcsd = 0.5*0.016**2/(rhoa*cdrag), zdftke.F90:243 (rhoa=1.22, cdrag=1.5e-3,
    zdftke.F90 locals) -- only used to express the NEMO check in N/m2."""
    return 0.5 * 0.016 ** 2 / (1.22 * 1.5e-3)  # coeff-ok: zdftke.F90:243 zcsd literal constants


if __name__ == "__main__":
    main(sys.argv[1:])

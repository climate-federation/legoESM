"""Onset of the low-K gap: per-term TKE budget, mixing length, Prandtl and Ri at
the gap interface, ours vs NEMO, equatorial strip (probe).

Cohort: our columns that carry a low-K gap (first interface with K < THR lies
above the deepest interface <= 40 m with K >= THR; omip_strip_ml_base_lc) at
hour HG, each taken at ITS gap interface kg (the smallest K above the active
base at HG).  NEMO is read at the same column and interface (paired).
Per 1-hour window (end-of-hour indexing, as the ledger): cumulative per-term
TKE increments from each model's own discretisation (omip_strip_tke_ledger:
between, lc, P, B, D, diff, floor, etau; the window sum telescopes to the e
change), and at the window end: e, step-entry length l = sqrt(e_in)/dissl_in
(omip_strip_mixing_length), inverse Prandtl K_H/K_M, N2 = -estrat/K_H,
S2 = eshear/K_M(s-1), Ri = N2/S2 (omip_strip_prandtl_offline conventions).
Ours closure K = applied - below-ML background (BG_A/BG_K, as the Prandtl
probe).  Median and IQR of ours-NEMO over the cohort.  No verdict printed.
usage: omip_strip_gap_onset.py OUR_RUN NEMO_EN2_DIR [HG THR H0 H1]
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omip_strip_tke_ledger as L          # noqa: E402
import omip_strip_ml_base_lc as M           # noqa: E402
import omip_strip_prandtl_offline as PR     # noqa: E402

PREFIX = "ORCA1_1ts_20000101_20000102_eqs1ts_"
C_EPS, E_BG = 0.7, 1.0e-10


def cohort_gap_interface(K, zw, thr, zmax):
    """(gap mask, gap interface index) for K (col, k): gap = first K < thr above
    the deepest K >= thr within zmax; kg = argmin K above that active base."""
    t = M.turbocline(K, zw, thr)
    da = M.deepest_active(K, zw, thr, zmax)
    g = ~np.isnan(da) & (t < da)
    above = zw[None, :] < np.where(np.isnan(da), -np.inf, da)[:, None]
    kg = np.argmin(np.where(above, K, np.inf), -1)
    return g, kg


def _iq(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "      nan [      nan,      nan]"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:9.2e} [{q[0]:9.2e},{q[2]:9.2e}]"


def main(argv):
    hg = int(argv[2]) if len(argv) > 2 else 12
    thr = float(argv[3]) if len(argv) > 3 else 1e-4
    h0, h1 = (int(argv[4]), int(argv[5])) if len(argv) > 5 else (4, 12)
    import netCDF4 as nc
    lo, le_end, lo_sol, cj, ci = L.ours_ledger(argv[0])
    a = [np.load(f"{argv[0]}/{fn}") for fn in ("snapshot_day0001.npz", "snapshot_final.npz")]
    g = lambda k: np.concatenate([x[k] for x in a]).astype(np.float64)
    sj, si = cj - L.J0, ci - L.I0
    keep = (sj >= L.BUF) & (sj < 13 - L.BUF) & (si >= L.BUF) & (si < 31 - L.BUF)
    lo = {k: v[:, keep] for k, v in lo.items()}
    le_end, lo_sol = le_end[:, keep], lo_sol[:, keep]
    Ko_app, Ao_app = g("col_K")[:, keep], g("col_A")[:, keep]
    e_in_o, ediss_o = g("col_e_in")[:, keep], g("col_ediss")[:, keep]
    est_o, sh_o = g("col_estrat")[:, keep], g("col_eshear")[:, keep]
    sj, si = sj[keep], si[keep]
    nk = le_end.shape[2]
    W = f"{argv[1]}/{PREFIX}W.nc"
    ln, ne_end, ne_pfl, ne_lc, avm, depw = L.nemo_ledger(W, sj, si, nk)
    d = nc.Dataset(W)

    def n(v):
        x = np.asarray(d[v][:, 1:nk + 1], dtype=np.float64)[:, :, sj, si].transpose(0, 2, 1)
        x[np.abs(x) > 1e15] = np.nan
        return x
    avtk, avmk, est_n, sh_n = n("avt_k"), n("avm_k"), n("estrat_k"), n("eshear_k")
    en_in, dl_in = n("en_in_k"), n("dissl_in_k")
    zw = np.asarray(d["depthw"][1:nk + 1], dtype=np.float64)

    KH = Ko_app - PR.BG_K
    KM = Ao_app - PR.BG_A
    dissl_o = 2.0 * ediss_o / (C_EPS * np.where(lo_sol > 0, lo_sol, np.nan))
    l_o = np.sqrt(np.maximum(e_in_o, E_BG)) / dissl_o
    l_n = np.sqrt(en_in) / np.where(dl_in > 0, dl_in, np.nan)
    n2_o = -est_o / np.where(KH > 0, KH, np.nan)
    n2_n = -est_n / np.where(avtk > 0, avtk, np.nan)

    sg = hg * 24 - 1
    gap, kg = cohort_gap_interface(Ko_app[sg], zw, thr, M.ZTOP)
    ng_nemo, _ = cohort_gap_interface(avtk[sg], zw, thr, M.ZTOP)
    cols = np.where(gap)[0]
    k = kg[cols]
    print(f"{argv[0]}: {keep.sum()} interior cols; cohort = ours gap at h{hg} (thr {thr:.0e}): {cols.size} cols "
          f"(NEMO gap there: {ng_nemo[cols].sum()}); gap interface depth median {np.median(zw[k]):.2f} m "
          f"[{np.percentile(zw[k], 25):.2f},{np.percentile(zw[k], 75):.2f}]")
    pick = lambda X, s: X[s, cols, k]
    # instrument: telescoping identity on the cohort, both sides, over the whole window
    s0, s1 = h0 * 24, h1 * 24
    for lab, led, e in (("ours", lo, le_end), ("NEMO", ln, ne_end)):
        tot = sum(np.nansum(led[t][s0:s1, cols, k], axis=0) for t in L.TERMS)
        de = e[s1 - 1, cols, k] - e[s0 - 1, cols, k]
        print(f"identity {lab}: max |sum(terms) - de| / max|de| = {np.nanmax(np.abs(tot - de)) / np.nanmax(np.abs(de)):.2e}")
    for c in (0, cols.size // 2):          # two rows printed for a hand check
        s = 8 * 24 - 1
        print(f"hand-check col {cols[c]} k {k[c]} ({zw[k[c]]:.2f} m) step {s}: ours "
              + " ".join(f"{t}={lo[t][s, cols[c], k[c]]:.3e}" for t in L.TERMS)
              + f" | e(s)-e(s-1)={le_end[s, cols[c], k[c]] - le_end[s - 1, cols[c], k[c]]:.3e}")
        print(f"   NEMO " + " ".join(f"{t}={ln[t][s, cols[c], k[c]]:.3e}" for t in L.TERMS)
              + f" | e(s)-e(s-1)={ne_end[s, cols[c], k[c]] - ne_end[s - 1, cols[c], k[c]]:.3e}")

    print(f"\nPer 1-h window at the gap interface: median [IQR] of ours-NEMO (cumulative, m2/s2); NEMO median in ()")
    for h in range(h0, h1):
        a0, a1 = h * 24, (h + 1) * 24
        print(f"-- h{h}-h{h + 1}")
        for t in L.TERMS:
            co = np.nansum(lo[t][a0:a1, cols, k], axis=0)
            cn = np.nansum(ln[t][a0:a1, cols, k], axis=0)
            print(f"   {t:7s} o-N {_iq(co - cn)}  (NEMO {np.median(cn):9.2e}, ours {np.median(co):9.2e})")
        s = a1 - 1
        eo, en_ = pick(le_end, s), pick(ne_end, s)
        print(f"   e end    ours {np.median(eo):9.2e} NEMO {np.median(en_):9.2e}  col ratio {np.median(eo / en_):6.2f}")
        for up in (1, 2, 3):
            ku = np.maximum(k - up, 0)
            ro = le_end[s, cols, ku] / ne_end[s, cols, ku]
            print(f"   e at kg-{up} ({np.median(zw[ku]):5.2f} m) col ratio o/N {np.median(ro):6.2f}  ours {np.median(le_end[s, cols, ku]):9.2e} NEMO {np.median(ne_end[s, cols, ku]):9.2e}")
        print(f"   l entry  ours {np.nanmedian(pick(l_o, s)):9.3f} NEMO {np.nanmedian(pick(l_n, s)):9.3f}  col ratio {np.nanmedian(pick(l_o, s) / pick(l_n, s)):6.2f}")
        pro = pick(KH, s) / pick(KM, s)
        prn = pick(avtk, s) / pick(avmk, s)
        print(f"   1/Pr     ours {np.nanmedian(pro):6.3f} NEMO {np.nanmedian(prn):6.3f}")
        s2o = pick(sh_o, s) / pick(KM, s - 1)
        s2n = pick(sh_n, s) / pick(avmk, s - 1)
        no, nn = pick(n2_o, s), pick(n2_n, s)
        print(f"   N2       ours {np.nanmedian(no):9.2e} NEMO {np.nanmedian(nn):9.2e}  col ratio {np.nanmedian(no / nn):6.2f}")
        print(f"   S2       ours {np.nanmedian(s2o):9.2e} NEMO {np.nanmedian(s2n):9.2e}  col ratio {np.nanmedian(s2o / s2n):6.2f}")
        print(f"   Ri       ours {np.nanmedian(no / s2o):6.3f} NEMO {np.nanmedian(nn / s2n):6.3f}  (ri_cri {PR.RI_CRI:.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])

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


def face_vs_tpoint(U, V, A, dz, j, i, kint):
    """avm-weighted shear production at T(j,i), interface kint, NEMO face form
    (zdfsh2.F90:80-94, now-shear squared, wet interior) vs the T-point form:
      face  = 0.25 * sum_f (A_nb + A_c) * s_f
      tpnt  = A_c * 0.5 * sum_f s_f
    with s_f = ((x_k - x_k+1)/dz)^2 at the four faces u(i-1), u(i), v(j-1), v(j)
    (u(i) east of T(i), v(j) north of T(j)).  Grid arrays (..., ny, nx, k)."""
    su = ((U[..., kint] - U[..., kint + 1]) / dz[kint]) ** 2
    sv = ((V[..., kint] - V[..., kint + 1]) / dz[kint]) ** 2
    Ak = A[..., kint]
    ac = Ak[..., j, i]
    face = 0.25 * ((Ak[..., j, i + 1] + ac) * su[..., j, i] + (ac + Ak[..., j, i - 1]) * su[..., j, i - 1]
                   + (Ak[..., j + 1, i] + ac) * sv[..., j, i] + (ac + Ak[..., j - 1, i]) * sv[..., j - 1, i])
    tpnt = ac * 0.5 * (su[..., j, i] + su[..., j, i - 1] + sv[..., j, i] + sv[..., j - 1, i])
    return face, tpnt


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

    # Effective production viscosity: eshear / S2_face, S2_face from the
    # C-grid face velocities at the column's own (j,i) faces, the SAME crude
    # operator on both sides (ours col_u/col_v post-solve, NEMO uoce/voce);
    # neither equals NEMO's avm-weighted p_sh2, only the ratio is meant.
    uo, vo = g("col_u")[:, keep], g("col_v")[:, keep]
    def nuv(f, v):
        x = np.asarray(nc.Dataset(f"{argv[1]}/{PREFIX}{f}.nc")[v][:, :nk + 1], dtype=np.float64)
        x = x[:, :, sj, si].transpose(0, 2, 1); x[np.abs(x) > 1e15] = np.nan
        return x
    un, vn = nuv("U", "uoce"), nuv("V", "voce")
    zt = np.asarray(nc.Dataset(L.MESH)["gdept_1d"][0], dtype=np.float64)[:nk + 1]
    dzt = np.diff(zt)
    s2f = lambda U, V: ((U[..., :-1] - U[..., 1:]) ** 2 + (V[..., :-1] - V[..., 1:]) ** 2) / dzt ** 2
    S2o_f, S2n_f = s2f(uo, vo), s2f(un, vn)
    print("\nEffective production viscosity at the gap interface (window-end step): median")
    print("  h | eshear ours NEMO | S2_face ours NEMO (col ratio) | nu_eff=eshear/S2_face ours NEMO (col ratio) | A_applied(s-1) ours | avm_k(s-1) NEMO | e col ratio")
    for h in range(h0, h1):
        s = (h + 1) * 24 - 1
        po, pn = pick(sh_o, s), pick(sh_n, s)
        so, sn = pick(S2o_f, s), pick(S2n_f, s)
        nuo, nun = po / so, pn / sn
        print(f"  h{h + 1:02d} | {np.nanmedian(po):9.2e} {np.nanmedian(pn):9.2e} | {np.nanmedian(so):9.2e} {np.nanmedian(sn):9.2e} "
              f"({np.nanmedian(so / sn):5.2f}) | {np.nanmedian(nuo):9.2e} {np.nanmedian(nun):9.2e} ({np.nanmedian(nuo / nun):5.2f}) | "
              f"{np.nanmedian(pick(Ao_app, s - 1)):9.2e} | {np.nanmedian(pick(avmk, s - 1)):9.2e} | {np.nanmedian(pick(le_end, s) / pick(ne_end, s)):5.2f}")

    # NEMO's TKE-diffusion floor MAX(avm(k)+avm(k+-1), 2e-5) (zdftke.F90:407-409)
    # with p_avm = avm_k entering the step: share of cohort samples where it binds
    print("\nNEMO TKE-diffusion floor binding at the gap interface (p_avm = avm_k(s-1)):")
    for h in range(h0, min(h1, 8)):
        a0, a1 = h * 24, (h + 1) * 24
        am = avmk[a0 - 1:a1 - 1, cols]                     # (steps, cohort, k)
        kk = k[None, :]
        up = np.take_along_axis(am, np.minimum(kk + 1, nk - 1)[..., None], -1)[..., 0] + np.take_along_axis(am, kk[..., None], -1)[..., 0]
        lw = np.take_along_axis(am, np.maximum(kk - 1, 0)[..., None], -1)[..., 0] + np.take_along_axis(am, kk[..., None], -1)[..., 0]
        print(f"  h{h}-h{h + 1}: upper-face sum < 2e-5 in {np.nanmean(up < 2e-5):.2f}, lower-face in {np.nanmean(lw < 2e-5):.2f}; "
              f"median sums {np.nanmedian(up):.2e} / {np.nanmedian(lw):.2e}")
    # First divergence per column: first step after the reference hour HREF where
    # e_ours/e_NEMO < 0.9 at kg; cumulative ours-NEMO per term from HREF to that step.
    href = 2
    sref = href * 24
    ratio = le_end[sref:, cols, k] / ne_end[sref:, cols, k]
    hit = ratio < 0.9
    has = np.any(hit, 0)
    first = np.argmax(hit, 0)
    lead = {t: 0 for t in L.TERMS}
    contrib = {t: [] for t in L.TERMS}
    for c in np.where(has)[0]:
        s1c = sref + first[c] + 1
        dd = {t: np.nansum(lo[t][sref:s1c, cols[c], k[c]]) - np.nansum(ln[t][sref:s1c, cols[c], k[c]]) for t in L.TERMS}
        de = (ne_end[s1c - 1, cols[c], k[c]] - ne_end[sref - 1, cols[c], k[c]])
        for t in L.TERMS:
            contrib[t].append(dd[t] / max(abs(de), 1e-30))
        lead[min(dd, key=dd.get)] += 1
    print(f"\nFirst divergence (e ours/NEMO < 0.9 at kg after h{href}): {has.sum()} of {cols.size} cols; "
          f"onset hour median {np.median(href + (first[has] + 1) / 24):.2f} [{np.percentile(href + (first[has] + 1) / 24, 25):.2f},"
          f"{np.percentile(href + (first[has] + 1) / 24, 75):.2f}]")
    print("  most-negative cumulative ours-NEMO term at onset (count): " + ", ".join(f"{t} {n}" for t, n in lead.items() if n))
    print("  median contribution / |NEMO de| since h2: " + ", ".join(f"{t} {np.median(v):+.2f}" for t, v in contrib.items() if v and np.any(np.asarray(v) != 0)))

    # (1) NEMO's 2e-5 TKE-diffusion floor, measured on NEMO's own operator:
    # cumulative floor part = direct(2e-5) - direct(0) at kg, h2 -> each column's onset
    avm_prev = np.concatenate([avmk[:1] * np.nan, avmk[:-1]])
    d_fl = L.nemo_diff_direct(ne_pfl, avm_prev, sj, si, nk, avm_floor=2e-5)
    d_nf = L.nemo_diff_direct(ne_pfl, avm_prev, sj, si, nk, avm_floor=0.0)
    fl = []
    for c in np.where(has)[0]:
        s1c = sref + first[c] + 1
        de = ne_end[s1c - 1, cols[c], k[c]] - ne_end[sref - 1, cols[c], k[c]]
        fl.append(np.nansum(d_fl[sref:s1c, cols[c], k[c]] - d_nf[sref:s1c, cols[c], k[c]]) / max(abs(de), 1e-30))
    print(f"\nNEMO TKE-diffusion floor contribution at kg, h{href} -> onset, / |NEMO de|: median {np.median(fl):+.3f} "
          f"[{np.percentile(fl, 25):+.3f},{np.percentile(fl, 75):+.3f}]  (ours-NEMO total at onset = -0.1 by construction)")
    # (2) face-summed vs T-point avm weighting of shear production, on the strip grid
    def grid(X, jj, ii):
        G = np.full((X.shape[0], 13, 31, X.shape[2]), np.nan)
        G[:, jj, ii, :] = X
        return G
    ja, ia = (a[0]["col_j"] - L.J0), (a[0]["col_i"] - L.I0)
    uo_all, vo_all = g("col_u"), g("col_v")
    A_all = np.maximum(g("col_A") - 2.0e-6, 1.4e-6)   # closure K_M ~ applied - (A_v 1.4e-6 + IWM ~0.6e-6), floored at kappaM_min
    Uo, Vo, Ao_g = grid(uo_all, ja, ia), grid(vo_all, ja, ia), grid(A_all, ja, ia)
    dN = nc.Dataset(f"{argv[1]}/{PREFIX}U.nc")
    Un = np.asarray(dN["uoce"][:, :nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    Vn = np.asarray(nc.Dataset(f"{argv[1]}/{PREFIX}V.nc")["voce"][:, :nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    Un[np.abs(Un) > 1e15] = np.nan; Vn[np.abs(Vn) > 1e15] = np.nan
    An = np.asarray(d["avm_k"][:, 1:nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    An[np.abs(An) > 1e15] = np.nan
    shN = np.asarray(d["eshear_k"][:, 1:nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    zt = np.asarray(nc.Dataset(L.MESH)["gdept_1d"][0], dtype=np.float64)
    dzt = np.diff(zt)
    # instrument check on NEMO: face form with avm_k(s-1) and now-shear vs NEMO's own eshear_k(s)
    ss = np.arange(sref, sref + 3 * 24)
    fN, tN, aN = [], [], []
    for c in range(cols.size):
        jj, ii = sj[cols[c]], si[cols[c]]
        f_, t_ = face_vs_tpoint(Un[ss], Vn[ss], An[ss - 1], dzt, jj, ii, k[c])
        fN.append(f_); tN.append(t_); aN.append(shN[ss, jj, ii, k[c]])
    fN, tN, aN = map(np.concatenate, (fN, tN, aN))
    ok = np.isfinite(fN) & np.isfinite(aN) & (aN > 0)
    print(f"CHECK NEMO face form vs eshear_k (h{href}-h{href + 3}, cohort kg): corr(log) {np.corrcoef(np.log(fN[ok]), np.log(aN[ok]))[0, 1]:.3f} "
          f"median ratio {np.median(fN[ok] / aN[ok]):.3f}; T-point form corr(log) {np.corrcoef(np.log(tN[ok] + 1e-30), np.log(aN[ok]))[0, 1]:.3f} "
          f"median ratio {np.median(tN[ok] / aN[ok]):.3f}  n={ok.sum()}")
    print("face/T-point production ratio at kg (window-end step), ours state | NEMO state; predicted ours P under face weighting / NEMO P (col median)")
    for h in range(href, min(h1, 8)):
        s = (h + 1) * 24 - 1
        ro, rn, pr = [], [], []
        for c in range(cols.size):
            jj, ii = sj[cols[c]], si[cols[c]]
            fo_, to_ = face_vs_tpoint(Uo[s], Vo[s], Ao_g[s - 1], dzt, jj, ii, k[c])
            fn_, tn_ = face_vs_tpoint(Un[s], Vn[s], An[s - 1], dzt, jj, ii, k[c])
            ro.append(fo_ / to_); rn.append(fn_ / tn_)
            pr.append(sh_o[s, cols[c], k[c]] * fo_ / to_ / sh_n[s, cols[c], k[c]])
        print(f"  h{h + 1:02d}: ours {np.nanmedian(ro):6.2f} | NEMO {np.nanmedian(rn):6.2f} | ours-face/NEMO P {np.nanmedian(pr):6.2f}  "
              f"(ours-tpoint/NEMO P {np.nanmedian(pick(sh_o, s) / pick(sh_n, s)):6.2f})")

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

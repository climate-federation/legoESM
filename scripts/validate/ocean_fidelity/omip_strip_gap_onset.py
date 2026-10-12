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
Ours closure K_H = applied - 6.04e-7 (IWM + K_v), closure K_M = max(applied -
2.0e-6 (A_v + IWM), kappaM_min); used for 1/Pr, N2, S2, Ri only, on faces
with K_M > 1e-5 (as the Prandtl probe).  Median and IQR of ours-NEMO over the cohort.  No verdict printed.
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
BG_A_NONCLOSURE = 2.0e-6     # coeff-ok: A_v 1.4e-6 + below-ML IWM viscosity ~0.6e-6 (2026-10-10 strip measurement, 3.40e-6 total)
KAPPA_M_MIN = 1.4e-6         # coeff-ok: resolved TKEConfig.kappaM_min of the run


def cohort_gap_interface(K, zw, thr, zmax):
    """(gap mask, gap interface index) for K (col, k): gap = first K < thr above
    the deepest K >= thr within zmax; kg = argmin K above that active base."""
    t = M.turbocline(K, zw, thr)
    da = M.deepest_active(K, zw, thr, zmax)
    g = ~np.isnan(da) & (t < da)
    above = zw[None, :] < np.where(np.isnan(da), -np.inf, da)[:, None]
    kg = np.argmin(np.where(above, K, np.inf), -1)
    return g, kg


def face_nb(U, Ub, V, Vb, A, dz, j, i, kint):
    """avm-weighted shear production at T(j,i), interface kint, NEMO zdfsh2
    face form (zdfsh2.F90:76-95, now x before shear, uniform-metric
    approximation, wet interior) and the same face shears with the T-point
    avm only:  face = 0.25*sum_f (A_nb + A_c)*s_f ;  tp = A_c*0.5*sum_f s_f,
    s_f = (x_k - x_k+1)(xb_k - xb_k+1)/dz^2 at faces u(i-1), u(i), v(j-1), v(j)
    (u(i) east of T(i), v(j) north of T(j)).  Grid arrays (..., ny, nx, k)."""
    su = (U[..., kint] - U[..., kint + 1]) * (Ub[..., kint] - Ub[..., kint + 1]) / dz[kint] ** 2
    sv = (V[..., kint] - V[..., kint + 1]) * (Vb[..., kint] - Vb[..., kint + 1]) / dz[kint] ** 2
    Ak = A[..., kint]
    ac = Ak[..., j, i]
    face = 0.25 * ((Ak[..., j, i + 1] + ac) * su[..., j, i] + (ac + Ak[..., j, i - 1]) * su[..., j, i - 1]
                   + (Ak[..., j + 1, i] + ac) * sv[..., j, i] + (ac + Ak[..., j - 1, i]) * sv[..., j - 1, i])
    tp = ac * 0.5 * (su[..., j, i] + su[..., j, i - 1] + sv[..., j, i] + sv[..., j - 1, i])
    return face, tp


def west_to_east(X, axis):
    """legoESM lat-lon C-grid stores u[j, i] = WEST face of cell (j, i) and
    v[j, i] = SOUTH face (latlon_cgrid_operators.py:4825; production centres
    with u(i)+u(i+1)); NEMO's u(i)/v(j) are the EAST/NORTH faces.  Returns X
    re-indexed to east/north faces: out[..., i] = X[..., i+1] (last row NaN)."""
    out = np.full_like(X, np.nan)
    src = [slice(None)] * X.ndim; dst = [slice(None)] * X.ndim
    src[axis] = slice(1, None); dst[axis] = slice(0, -1)
    out[tuple(dst)] = X[tuple(src)]
    return out


def onset_steps(ratio, thr=0.9):
    """ratio (steps, cols) with ratio[0] = the reference record.  Returns
    (has, first): has = column agrees at the reference (>= thr) and later
    drops below thr; first = index into ratio[1:] of the first drop."""
    hit = ratio[1:] < thr
    has = np.any(hit, 0) & (ratio[0] >= thr)
    return has, np.argmax(hit, 0)


def ratio_gate(r, cover_min=0.9, lo=0.9, hi=1.1):
    """(pass, coverage, quartiles): coverage = finite-positive share; pass
    requires coverage >= cover_min and the IQR inside [lo, hi]."""
    r = np.asarray(r, dtype=np.float64).ravel()
    good = np.isfinite(r) & (r > 0)
    cov = float(good.mean()) if r.size else 0.0
    q = np.percentile(r[good], [25, 50, 75]) if good.any() else np.full(3, np.nan)
    return bool(cov >= cover_min and q[0] > lo and q[2] < hi), cov, q


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

    KH = Ko_app - PR.BG_K                                   # K below ML 6.04e-7 = IWM + K_v (closure floor 1e-10)
    KM = np.maximum(Ao_app - BG_A_NONCLOSURE, KAPPA_M_MIN)   # A below ML 3.4e-6 = closure floor 1.4e-6 + A_v 1.4e-6 + IWM ~0.6e-6
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
        tot = sum(np.sum(led[t][s0:s1, cols, k], axis=0) for t in L.TERMS if t != "between") \
            + np.sum(led["between"][max(s0, 1):s1, cols, k], axis=0)   # 'between' is NaN only at step 0
        print(f"identity {lab}: non-finite column windows {int(np.sum(~np.isfinite(tot)))} of {tot.size}")
        de = e[s1 - 1, cols, k] - e[s0 - 1, cols, k]
        okt = np.isfinite(tot) & np.isfinite(de)
        print(f"identity {lab}: max |sum(terms) - de| / max|de| = {np.max(np.abs(tot - de)[okt]) / np.max(np.abs(de)[okt]):.2e} over {okt.sum()} complete columns")
    for c in (0, cols.size // 2):          # two rows printed for a hand check
        s = 8 * 24 - 1
        print(f"hand-check col {cols[c]} k {k[c]} ({zw[k[c]]:.2f} m) step {s}: ours "
              + " ".join(f"{t}={lo[t][s, cols[c], k[c]]:.3e}" for t in L.TERMS)
              + f" | e(s)-e(s-1)={le_end[s, cols[c], k[c]] - le_end[s - 1, cols[c], k[c]]:.3e}")
        print(f"   NEMO " + " ".join(f"{t}={ln[t][s, cols[c], k[c]]:.3e}" for t in L.TERMS)
              + f" | e(s)-e(s-1)={ne_end[s, cols[c], k[c]] - ne_end[s - 1, cols[c], k[c]]:.3e}")

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
    sref = href * 24                                     # first step after h2; end-of-h2 record = sref-1
    ratio = le_end[sref - 1:, cols, k] / ne_end[sref - 1:, cols, k]   # ratio[0] = end of h2
    has, first = onset_steps(ratio)                      # onset step = sref + first
    groups = {"P+B": ("P", "B"), "D+diff": ("D", "diff"), "lc": ("lc",), "between+floor+etau": ("between", "floor", "etau")}
    lead = {gname: 0 for gname in groups}
    contrib = {gname: [] for gname in list(groups) + ["total"]}
    incomplete = 0
    onset_cols = []
    for c in np.where(has)[0]:
        s1c = sref + first[c] + 1
        segs = {t: (lo[t][sref:s1c, cols[c], k[c]], ln[t][sref:s1c, cols[c], k[c]]) for t in L.TERMS}
        if not all(np.all(np.isfinite(o)) and np.all(np.isfinite(n_)) for o, n_ in segs.values()):
            incomplete += 1
            continue
        dd = {t: np.sum(o) - np.sum(n_) for t, (o, n_) in segs.items()}
        de = abs(ne_end[s1c - 1, cols[c], k[c]] - ne_end[sref - 1, cols[c], k[c]])
        gd = {gname: sum(dd[t] for t in ts) for gname, ts in groups.items()}
        for gname, v in gd.items():
            contrib[gname].append(v / max(de, 1e-30))
        contrib["total"].append(sum(dd.values()) / max(de, 1e-30))
        lead[min(gd, key=gd.get)] += 1
        onset_cols.append(c)
    oc = np.asarray(onset_cols, dtype=int)
    oh = href + (first[oc] + 1) / 24 if oc.size else np.array([np.nan])
    print(f"\nFirst divergence (e ours/NEMO < 0.9 at kg after h{href}, >= 0.9 at end of h{href}): {oc.size} of {cols.size} cols "
          f"({incomplete} more dropped for incomplete windows); onset hour median {np.median(oh):.2f} [{np.percentile(oh, 25):.2f},{np.percentile(oh, 75):.2f}]")
    print("  most-negative cumulative ours-NEMO term GROUP at onset (count): " + ", ".join(f"{t} {n}" for t, n in lead.items()))
    print("  median group contribution / |NEMO de| since end of h2: " + ", ".join(f"{t} {np.median(v):+.2f}" for t, v in contrib.items() if v))

    # (1) NEMO's 2e-5 TKE-diffusion floor, measured on NEMO's own operator:
    # cumulative floor part = direct(2e-5) - direct(0) at kg, h2 -> each column's onset
    avm_prev = np.concatenate([avmk[:1] * np.nan, avmk[:-1]])
    d_fl = L.nemo_diff_direct(ne_pfl, avm_prev, sj, si, nk, avm_floor=2e-5)
    d_nf = L.nemo_diff_direct(ne_pfl, avm_prev, sj, si, nk, avm_floor=0.0)
    fl, nbad = [], 0
    for c in onset_cols:
        s1c = sref + first[c] + 1
        seg = d_fl[sref:s1c, cols[c], k[c]] - d_nf[sref:s1c, cols[c], k[c]]
        if not np.all(np.isfinite(seg)):
            nbad += 1
            continue
        de = ne_end[s1c - 1, cols[c], k[c]] - ne_end[sref - 1, cols[c], k[c]]
        fl.append(np.sum(seg) / max(abs(de), 1e-30))
    fl = np.asarray(fl)
    print(f"\nNEMO TKE-diffusion floor, LOCAL operator contribution on NEMO's own solution at kg, h{href} -> onset, / |NEMO de| "
          f"(not a counterfactual solve): n={fl.size} (dropped {nbad} with non-finite samples)"
          + (f" median {np.median(fl):+.3f} [{np.percentile(fl, 25):+.3f},{np.percentile(fl, 75):+.3f}]" if fl.size else ""))
    # (2) avm weighting of shear production, on the strip grid.
    def grid(X, jj, ii):
        G = np.full((X.shape[0], 13, 31, X.shape[2]), np.nan)
        G[:, jj, ii, :] = X
        return G
    ja, ia = (a[0]["col_j"] - L.J0), (a[0]["col_i"] - L.I0)
    Uo = west_to_east(grid(g("col_u"), ja, ia), axis=2)       # -> east faces, as NEMO's u(i)
    Vo = west_to_east(grid(g("col_v"), ja, ia), axis=1)       # -> north faces, as NEMO's v(j)
    Ao_app_g = grid(g("col_A"), ja, ia)
    SHo = grid(g("col_eshear"), ja, ia)
    zt = np.asarray(nc.Dataset(L.MESH)["gdept_1d"][0], dtype=np.float64)
    dzt = np.diff(zt)[:nk]
    # ours 'squared_centered': centre-averaged velocities, then differenced; production
    # at step s uses the state entering s = post-solve s-1 (codex).  Our effective
    # production viscosity is READ from our own record: nu_o = eshear(s) / S2c(s-1),
    # so no closure/background guess enters (codex).
    uc = 0.5 * (Uo + np.roll(Uo, 1, axis=2)); vc = 0.5 * (Vo + np.roll(Vo, 1, axis=1))
    S2c = ((uc[..., :-1] - uc[..., 1:]) ** 2 + (vc[..., :-1] - vc[..., 1:]) ** 2) / dzt ** 2   # (t, y, x, nk)
    S2c_prev = np.concatenate([S2c[:1] * np.nan, S2c[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        NUo = np.where(S2c_prev > 1e-12, SHo / S2c_prev, np.nan)
    dN = nc.Dataset(f"{argv[1]}/{PREFIX}U.nc")
    Un = np.asarray(dN["uoce"][:, :nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    Vn = np.asarray(nc.Dataset(f"{argv[1]}/{PREFIX}V.nc")["voce"][:, :nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    Un[np.abs(Un) > 1e15] = np.nan; Vn[np.abs(Vn) > 1e15] = np.nan
    An = np.asarray(d["avm_k"][:, 1:nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    An[np.abs(An) > 1e15] = np.nan
    shN = np.asarray(d["eshear_k"][:, 1:nk + 1], dtype=np.float64).transpose(0, 2, 3, 1)
    shN[np.abs(shN) > 1e15] = np.nan

    # instrument gate (NEMO): face form with step-entry shear squared -- ORCA1EN2 is an
    # RK3 build, stprk3.f90:155 zdf_phy(Nbb,Nbb), so NEMO's now and before coincide --
    # avm_k(s-1) vs eshear_k(s); coverage >= 90% finite-positive or FAIL, and the
    # dependent production comparison is withheld on FAIL (codex)
    ss = np.arange(sref, sref + 3 * 24)
    rr = []
    for c in range(cols.size):
        jj, ii = sj[cols[c]], si[cols[c]]
        f_, _ = face_nb(Un[ss - 1], Un[ss - 1], Vn[ss - 1], Vn[ss - 1], An[ss - 1], dzt, jj, ii, k[c])
        rr.append(f_ / shN[ss, jj, ii, k[c]])
    gate, cover, q = ratio_gate(np.concatenate(rr))
    print(f"GATE NEMO face form (step-entry shear^2) / eshear_k at cohort kg h{href}-h{href + 3} (every step): coverage {cover:.3f}, "
          f"IQR {q[0]:.3f}-{q[2]:.3f} -> {'PASS' if gate else 'FAIL'} (coverage >= 0.9, IQR within 0.9-1.1)")
    # our side: reconstructed centred production with our own nu reproduces eshear by
    # construction; the substantive check is that nu_o is a smooth closure field:
    # report nu_o vs applied A (must be <= A, ~ A - background where turbulent)
    # our nu_eff = eshear(s)/S2c(s-1) must be a physical production viscosity:
    # >= kappaM_min (1.4e-6) and <= the applied viscosity A(s-1) (closure +
    # background + IWM), per sample, over the reported hours and the cohort's
    # face neighbours (codex: a misaligned reconstruction lands outside)
    hrs = range(href, min(h1, 9))

    def nu_gate(sx, jj_, ii_, kk_):
        """share of samples with nu_eff in [kappaM_min, A_applied] (5% tol)."""
        nu = NUo[sx, jj_, ii_, kk_]; aa = Ao_app_g[sx - 1, jj_, ii_, kk_]
        return float(np.mean(np.isfinite(nu) & (nu >= 1.4e-6 * 0.95) & (nu <= aa * 1.05)))
    nu_pass = True
    for h in hrs:
        s = (h + 1) * 24 - 1
        cen = nu_gate(s, sj[cols], si[cols], k)
        nb = np.mean([nu_gate(s, sj[cols] + dj, si[cols] + di, k) for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0))])
        ok_h = cen >= 0.9 and nb >= 0.9
        nu_pass &= ok_h
        print(f"GATE ours nu_eff in [1.4e-6, A_applied] h{h + 1:02d}: centres {cen:.3f} neighbours {nb:.3f} -> {'PASS' if ok_h else 'FAIL'} (each >= 0.9)")
    gates = []
    for h in hrs:
        s = (h + 1) * 24 - 1
        rh = [face_nb(Un[s - 1], Un[s - 1], Vn[s - 1], Vn[s - 1], An[s - 1], dzt, sj[c_], si[c_], k_)[0] / shN[s, sj[c_], si[c_], k_]
              for c_, k_ in zip(cols, k)]
        ok_h, cov_h, q_h = ratio_gate(rh)
        gates.append(ok_h)
        print(f"  NEMO gate h{h + 1:02d}: coverage {cov_h:.3f} IQR {q_h[0]:.3f}-{q_h[2]:.3f} -> {'PASS' if ok_h else 'FAIL'}")
    gate = gate and nu_pass and all(gates)
    if not gate:
        print("  production comparison WITHHELD (an instrument gate failed)")
    # second viscosity source, independent of the shear time level (codex round 5:
    # production evaluates shear on the in-solve working state, not dumped)
    KMc = np.maximum(Ao_app_g - BG_A_NONCLOSURE, KAPPA_M_MIN)           # (t, y, x, k), closure estimate
    KMc_prev = np.concatenate([KMc[:1] * np.nan, KMc[:-1]])
    agree = []
    for h in hrs:
        s = (h + 1) * 24 - 1
        nu = NUo[s, sj[cols], si[cols], k]; km = KMc_prev[s, sj[cols], si[cols], k]
        big = km > 2e-5
        if big.any():
            agree.append(nu[big] / km[big])
    agree = np.concatenate(agree) if agree else np.array([np.nan])
    print(f"consistency nu_eff(record) / closure estimate where K_M > 2e-5, h{href + 1}-h{max(hrs) + 1}: median {np.nanmedian(agree):.2f} "
          f"[{np.nanpercentile(agree, 25):.2f},{np.nanpercentile(agree, 75):.2f}] n={np.isfinite(agree).sum()}")
    if gate:
        print("Production at kg (window-end step), median over cohort, / NEMO eshear_k (col median):")
        print("  h | ours recorded (centred, T-pt nu) | ours face-geometry, T-pt nu | face-summed nu=record | face-summed nu=closure est || NEMO face/T-pt ratio")
        for h in range(href, min(h1, 9)):
            s = (h + 1) * 24 - 1
            r_rec, r_tp, r_fc, r_fk, r_n = [], [], [], [], []
            for c in range(cols.size):
                jj, ii = sj[cols[c]], si[cols[c]]
                fo_, to_ = face_nb(Uo[s - 1], Uo[s - 1], Vo[s - 1], Vo[s - 1], NUo[s], dzt, jj, ii, k[c])
                fk_, tk_ = face_nb(Uo[s - 1], Uo[s - 1], Vo[s - 1], Vo[s - 1], KMc_prev[s], dzt, jj, ii, k[c])
                fn_, tn_ = face_nb(Un[s - 1], Un[s - 1], Vn[s - 1], Vn[s - 1], An[s - 1], dzt, jj, ii, k[c])
                ref = shN[s, jj, ii, k[c]]
                r_rec.append(SHo[s, jj, ii, k[c]] / ref); r_tp.append(to_ / ref); r_fc.append(fo_ / ref)
                r_fk.append(fk_ / ref); r_n.append(fn_ / tn_)
            cv = lambda x: np.mean(np.isfinite(np.asarray(x)))
            print(f"  h{h + 1:02d} | {np.nanmedian(r_rec):6.2f} | {np.nanmedian(r_tp):6.2f} | {np.nanmedian(r_fc):6.2f} | {np.nanmedian(r_fk):6.2f} || {np.nanmedian(r_n):6.2f}"
                  f"   (coverage {cv(r_rec):.2f}/{cv(r_tp):.2f}/{cv(r_fc):.2f}/{cv(r_fk):.2f})")
        print("  (closure-est column = absolute face-geometry production weighted by the closure estimate max(A(s-1)-2e-6, kappaM_min): no inferred viscosity)")
    # selection-bias control (GLM): unconditional face/T-pt ratio, all interior columns,
    # interfaces 10-25 m; each hour gated on both instruments over the SAME samples
    kk_rng = np.where((zw >= 10.0) & (zw <= 25.0))[0]
    jin, iin = np.meshgrid(np.arange(L.BUF, 13 - L.BUF), np.arange(L.BUF, 31 - L.BUF), indexing="ij")
    print("Unconditional face/T-pt production ratio, all interior columns, interfaces 10-25 m, ours weighted by the closure estimate: ours | NEMO (median [p25,p75]), per-hour gated")
    for h in (4, 6, 8, 12, 18, 30):
        s = h * 24 - 1
        ro, rn_, gN, gO = [], [], [], []
        for kint in kk_rng:
            f1, t1 = face_nb(Uo[s - 1], Uo[s - 1], Vo[s - 1], Vo[s - 1], KMc_prev[s], dzt, jin, iin, kint)
            f2, t2 = face_nb(Un[s - 1], Un[s - 1], Vn[s - 1], Vn[s - 1], An[s - 1], dzt, jin, iin, kint)
            ro.append((f1 / t1).ravel()); rn_.append((f2 / t2).ravel())
            gN.append((f2 / shN[s, jin, iin, kint]).ravel())
            gO.append(np.full(jin.size, float(np.mean(np.isfinite(KMc_prev[s, jin, iin, kint])))))   # closure-estimate weights: finite coverage
        okN, covN, qN = ratio_gate(np.concatenate(gN))
        shareO = float(np.mean(np.concatenate(gO)))
        if not (okN and shareO >= 0.9):
            print(f"  h{h:02d}: WITHHELD (NEMO gate {'PASS' if okN else 'FAIL'} cov {covN:.2f}; ours weights finite {shareO:.2f})")
            continue
        ro, rn_ = np.concatenate(ro), np.concatenate(rn_)
        jok = np.isfinite(ro) & (ro > 0) & np.isfinite(rn_) & (rn_ > 0)      # same surviving samples on both sides
        ro, rn_ = ro[jok], rn_[jok]
        print(f"  h{h:02d}: ours {np.median(ro):6.2f} [{np.percentile(ro, 25):.2f},{np.percentile(ro, 75):.2f}] | "
              f"NEMO {np.median(rn_):6.2f} [{np.percentile(rn_, 25):.2f},{np.percentile(rn_, 75):.2f}]  (gates: NEMO IQR {qN[0]:.3f}-{qN[2]:.3f}, ours weights finite {shareO:.2f})")

    print(f"\nPer 1-h window at the gap interface: median [IQR] of ours-NEMO (cumulative, m2/s2); NEMO median in ()")
    for h in range(h0, h1):
        a0, a1 = h * 24, (h + 1) * 24
        print(f"-- h{h}-h{h + 1}")
        for t in L.TERMS:
            co = np.sum(lo[t][a0:a1, cols, k], axis=0)     # NaN propagates: incomplete window -> excluded
            cn = np.sum(ln[t][a0:a1, cols, k], axis=0)
            print(f"   {t:7s} o-N {_iq(co - cn)}  (NEMO {np.median(cn):9.2e}, ours {np.median(co):9.2e})")
        s = a1 - 1
        eo, en_ = pick(le_end, s), pick(ne_end, s)
        print(f"   e end    ours {np.median(eo):9.2e} NEMO {np.median(en_):9.2e}  col ratio {np.median(eo / en_):6.2f}")
        for up in (1, 2, 3):
            ku = np.maximum(k - up, 0)
            ro = le_end[s, cols, ku] / ne_end[s, cols, ku]
            print(f"   e at kg-{up} ({np.median(zw[ku]):5.2f} m) col ratio o/N {np.median(ro):6.2f}  ours {np.median(le_end[s, cols, ku]):9.2e} NEMO {np.median(ne_end[s, cols, ku]):9.2e}")
        print(f"   l entry  ours {np.nanmedian(pick(l_o, s)):9.3f} NEMO {np.nanmedian(pick(l_n, s)):9.3f}  col ratio {np.nanmedian(pick(l_o, s) / pick(l_n, s)):6.2f}")
        okm = ((pick(KM, s) > 1e-5) & (pick(KM, s - 1) > 1e-5) & (pick(avmk, s) > 1e-5) & (pick(avmk, s - 1) > 1e-5))
        pro = np.where(okm, pick(KH, s) / pick(KM, s), np.nan)
        prn = np.where(okm, pick(avtk, s) / pick(avmk, s), np.nan)
        print(f"   [Pr/N2/S2/Ri on faces with K_M > 1e-5 both sides, s and s-1: n={okm.sum()} of {okm.size}]")
        print(f"   1/Pr     ours {np.nanmedian(pro):6.3f} NEMO {np.nanmedian(prn):6.3f}")
        s2o = pick(sh_o, s) / pick(KM, s - 1)
        s2n = pick(sh_n, s) / pick(avmk, s - 1)
        no, nn = np.where(okm, pick(n2_o, s), np.nan), np.where(okm, pick(n2_n, s), np.nan)
        s2o, s2n = np.where(okm, s2o, np.nan), np.where(okm, s2n, np.nan)
        print(f"   N2       ours {np.nanmedian(no):9.2e} NEMO {np.nanmedian(nn):9.2e}  col ratio {np.nanmedian(no / nn):6.2f}")
        print(f"   S2       ours {np.nanmedian(s2o):9.2e} NEMO {np.nanmedian(s2n):9.2e}  col ratio {np.nanmedian(s2o / s2n):6.2f}")
        print(f"   Ri       ours {np.nanmedian(no / s2o):6.3f} NEMO {np.nanmedian(nn / s2n):6.3f}  (ri_cri {PR.RI_CRI:.3f})")


if __name__ == "__main__":
    main(sys.argv[1:])

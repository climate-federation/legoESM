"""Cumulative per-term TKE ledger, ours vs NEMO, equatorial strip (probe).

Each model's ledger is built in its OWN applied discretisation, per step:
  between = e_in(s) - e_end(s-1)           (anything outside tke_tke)
  lc      = Langmuir increment             (ours: col_e_lc; NEMO: en_lc - en_in)
  P, B    = dt*shear, dt*applied buoyancy
  D       = dt*c_eps*dissl_in*(0.5 e_lc - 1.5 e_new)   (NEMO 1.5/0.5 split)
  diff    = e_new - e_lc - (P + B + D)     (the e-diffusion increment)
  floor   = e_sol - e_new                  (NEMO e_new = en_pfl; ours 0, see below)
  etau    = e_end - e_sol
Ours applies the emin floor inside its solver, so columns where EITHER side
is floor-bound at the target interface (or its neighbours) in the window are
dropped.  Sums over the window telescope to the cumulative e change; the
identity is enforced per column.  Independent check (NEMO only): diff is
rebuilt from NEMO's own tridiagonal operator (avm_k of the previous step,
mesh e3t_0/e3w_0) and compared with the residual.  No verdict printed.

usage: omip_strip_tke_ledger.py OUR_RUN NEMO_EN2_W [H0 H1]
"""
import sys
import numpy as np

J0, I0, BUF = 180, 153, 3
DT, C_EPS, EMIN = 150.0, 0.7, 1.0e-10
MESH = '/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc'
DEPTHS_M = (5.8, 12.8, 17.9)
TERMS = ("between", "lc", "P", "B", "D", "diff", "floor", "etau")
MERGED_LC = False


def ours_ledger(run):
    a, b = np.load(f"{run}/snapshot_day0001.npz"), np.load(f"{run}/snapshot_final.npz")
    g = lambda k: np.concatenate([a[k], b[k]], axis=0).astype(np.float64)
    e_in, lc, e_sol, e_end = g("col_e_in"), g("col_e_lc"), g("col_e_sol"), g("col_e")
    P, B, ediss = DT * g("col_eshear"), DT * g("col_bapplied"), g("col_ediss")
    cd = np.where(e_sol > 0, 2.0 * ediss / e_sol, 0.0)      # c_eps*dissl_used
    if np.all(np.isnan(lc)):
        # Langmuir merged into the RHS source (coupling 'separate'): no 0.5
        # dissipation add-back on it, and it lands in diff -> read diff as
        # diff+lc and compare with NEMO's diff+lc (MERGED_LC flag).
        print("OURS Langmuir not separable: ours diff = diff + lc")
        lc = np.zeros_like(e_sol)
        global MERGED_LC
        MERGED_LC = True
    e_lc = e_in + lc
    D = DT * cd * (0.5 * e_lc - 1.5 * e_sol)
    led = dict(lc=lc, P=P, B=B, D=D, diff=e_sol - e_lc - P - B - D,
               floor=np.zeros_like(e_sol), etau=e_end - e_sol)
    led["between"] = np.concatenate([np.full_like(e_in[:1], np.nan), e_in[1:] - e_end[:-1]])
    return led, e_end, e_sol, a["col_j"], a["col_i"]


def nemo_ledger(path, sj, si, nk):
    import netCDF4 as nc
    d = nc.Dataset(path)
    def g(v):
        x = np.asarray(d[v][:, 1:nk + 1], dtype=np.float64)[:, :, sj, si].transpose(0, 2, 1)
        x[np.abs(x) > 1e15] = np.nan
        return x
    e_in, e_lc, e_sol, e_pfl, e_end = g("en_in_k"), g("en_lc_k"), g("en_sol_k"), g("en_pfl_k"), g("en_k")
    P, B, dl = DT * g("eshear_k"), DT * g("bapp_k"), g("dissl_in_k")
    D = DT * C_EPS * dl * (0.5 * e_lc - 1.5 * e_pfl)
    led = dict(lc=e_lc - e_in, P=P, B=B, D=D, diff=e_pfl - e_lc - P - B - D,
               floor=e_sol - e_pfl, etau=e_end - e_sol)
    led["between"] = np.concatenate([np.full_like(e_in[:1], np.nan), e_in[1:] - e_end[:-1]])
    depw = np.asarray(d["depthw"][1:nk + 1])
    avm = g("avm_k")
    return led, e_end, e_pfl, e_lc, avm, depw


def nemo_diff_direct(e_new, avm_prev, sj, si, nk):
    """-(zd_up (e_k+1 - e_k) + zd_lw (e_k-1 - e_k)), zdftke.F90 matrix rows,
    on our interface index k (NEMO w-level k+1); e3 from mesh e3*_0."""
    import netCDF4 as nc
    m = nc.Dataset(MESH)
    e3t = np.asarray(m["e3t_0"][0])[:, sj + J0, si + I0].T    # (col, 75)
    e3w = np.asarray(m["e3w_0"][0])[:, sj + J0, si + I0].T
    out = np.full_like(e_new, np.nan)
    # our interface k = file w-index k+1 = Fortran jk = k+2; k=0 would need
    # the surface Dirichlet row (not in our arrays), so start at k=1.
    # Fortran e3t(jk) -> 0-based k+1, e3t(jk-1) -> k, e3w(jk) -> k+1.
    for k in range(1, nk - 1):
        up = -0.5 * DT * np.maximum(avm_prev[:, :, k + 1] + avm_prev[:, :, k], 2e-5) / (e3t[:, k + 1] * e3w[:, k + 1])
        lw = -0.5 * DT * np.maximum(avm_prev[:, :, k] + avm_prev[:, :, k - 1], 2e-5) / (e3t[:, k] * e3w[:, k + 1])
        out[:, :, k] = -(up * (e_new[:, :, k + 1] - e_new[:, :, k]) + lw * (e_new[:, :, k - 1] - e_new[:, :, k]))
    return out


def main(argv):
    h0, h1 = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (4, 14)
    lo, le_end, lo_sol, cj, ci = ours_ledger(argv[0])
    sj, si = cj - J0, ci - I0
    keep = (sj >= BUF) & (sj < 13 - BUF) & (si >= BUF) & (si < 31 - BUF)
    lo = {k: v[:, keep] for k, v in lo.items()}; le_end, lo_sol = le_end[:, keep], lo_sol[:, keep]
    sj, si = sj[keep], si[keep]
    nk = le_end.shape[2]
    ln, ne_end, ne_pfl, ne_lc, avm, depw = nemo_ledger(argv[1], sj, si, nk)
    s0, s1 = h0 * 24, h1 * 24            # steps s0..s1-1 = hours h0+1..h1 (end-of-hour indexing as onset probe)
    # gates
    gate_e = np.nanmax(np.abs(ln["between"][1:])) / np.nanmax(np.abs(ne_end))
    print(f"gate NEMO en_in(s) vs en_k(s-1): max |diff|/max e = {gate_e:.2e}")
    gate_o = np.nanmax(np.abs(lo["between"][1:])) / np.nanmax(np.abs(le_end))
    print(f"gate OURS e_in(s) vs e(s-1): max |diff|/max e = {gate_o:.2e}")
    nfl = np.mean(ne_pfl[s0:s1] < EMIN)
    print(f"NEMO floor binds (en_pfl < emin) on {nfl:.4f} of window samples")
    ddir = nemo_diff_direct(ne_pfl, np.concatenate([avm[:1] * np.nan, avm[:-1]]), sj, si, nk)
    for z in DEPTHS_M:
        k = int(np.argmin(np.abs(depw - z)))
        r, dd = ln["diff"][s0:s1, :, k], ddir[s0:s1, :, k]
        ok = np.isfinite(r) & np.isfinite(dd)
        cc = np.corrcoef(r[ok], dd[ok])[0, 1]
        sl = np.sum(r[ok] * dd[ok]) / np.sum(dd[ok] ** 2)
        print(f"check NEMO diff residual vs direct operator @ {depw[k]:.1f} m: corr {cc:.4f} slope {sl:.4f} n {ok.sum()}")
    if MERGED_LC:
        ln["diff"] = ln["diff"] + ln["lc"]; ln["lc"] = np.zeros_like(ln["lc"])
    print(f"window steps {s0}..{s1 - 1} (h{h0 + 1}-h{h1}), {keep.sum()} interior cols; units m2/s2 cumulative")
    for z in DEPTHS_M:
        k = int(np.argmin(np.abs(depw - z)))
        sl = slice(max(k - 1, 0), k + 2)
        fb = (np.any(lo_sol[s0:s1, :, sl] <= EMIN * 1.0001, axis=(0, 2))
              | np.any(ne_pfl[s0:s1, :, sl] <= EMIN * 1.0001, axis=(0, 2)))
        dE_o = le_end[s1 - 1, :, k] - le_end[s0 - 1, :, k]
        dE_n = ne_end[s1 - 1, :, k] - ne_end[s0 - 1, :, k]
        co = {t: np.sum(lo[t][s0:s1, :, k], axis=0) for t in TERMS}
        cn = {t: np.sum(ln[t][s0:s1, :, k], axis=0) for t in TERMS}
        res_o = np.abs(sum(co.values()) - dE_o); res_n = np.abs(sum(cn.values()) - dE_n)
        good = ~fb & (res_o <= 1e-6 * np.abs(dE_o).max() + 1e-14) & (res_n <= 1e-6 * np.abs(dE_n).max() + 1e-14)
        good &= np.all([np.isfinite(co[t]) & np.isfinite(cn[t]) for t in TERMS], axis=0)
        print(f"\n@ {depw[k]:.1f} m: cols used {good.sum()} (floor-dropped {fb.sum()}, identity-dropped {(~fb & ~good).sum()})")
        print(f"  e at window start  ours {np.median(le_end[s0 - 1, good, k]):.3e}  NEMO {np.median(ne_end[s0 - 1, good, k]):.3e}")
        # downward e flux into the layer below the T-level above interface k:
        # sum_{j>=k} diff_j * e3w_j (deep e ~ emin, so the bottom flux ~ 0),
        # from each model's own residual -- no diffusivity needed (GLM test)
        e3w = np.asarray(__import__("netCDF4").Dataset(MESH)["e3w_0"][0])[1:nk + 1, sj + J0, si + I0].T
        fo = np.sum(np.nansum(lo["diff"][s0:s1, :, k:] * e3w[None, :, k:], axis=2), axis=0)
        fn = np.sum(np.nansum(ln["diff"][s0:s1, :, k:] * e3w[None, :, k:], axis=2), axis=0)
        for lab, msk in (("used cols", good), ("ALL cols", np.ones_like(good))):
            q = np.percentile((fo - fn)[msk], [25, 75])
            print(f"  e flux into layer below (m3/s2 cum) [{lab}, n={msk.sum()}]: ours {np.median(fo[msk]):.3e} "
                  f"NEMO {np.median(fn[msk]):.3e} o-N {np.median((fo - fn)[msk]):.3e} IQR [{q[0]:.3e},{q[1]:.3e}]")
        co["local"] = co["P"] + co["B"] + co["D"]; cn["local"] = cn["P"] + cn["B"] + cn["D"]
        print(f"  {'term':8s} {'ours':>11s} {'NEMO':>11s} {'ours-NEMO':>11s} {'IQR(o-N)':>23s}")
        for t in TERMS + ("local", "dE"):
            o = dE_o[good] if t == "dE" else co[t][good]
            n = dE_n[good] if t == "dE" else cn[t][good]
            q = np.percentile(o - n, [25, 75])
            print(f"  {t:8s} {np.median(o):11.3e} {np.median(n):11.3e} {np.median(o - n):11.3e}  [{q[0]:10.3e},{q[1]:10.3e}]")


if __name__ == "__main__":
    main(sys.argv[1:])

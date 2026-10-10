"""NEMO's Prandtl function applied offline to each model's OWN saved state (probe).

NEMO zdftke (nn_pdl=1): Ri = rn2b * p_avm / (p_sh2 + rn_bshear) (0 if rn2b<=0),
pdlr = max(0.1, ri_cri / max(ri_cri, Ri)), ri_cri = 2/(2 + rn_ediss/rn_ediff),
p_avm = avm_k entering the step, avt_k = max(pdlr*zav, avtb).
Inputs per interface and step:
  NEMO  rn2 = -estrat_k/avt_k, p_avm = avm_k(s-1), p_sh2 = eshear_k(s);
        actual ratio avt_k/avm_k(s)  -> INSTRUMENT CHECK (formula vs itself)
  ours  closure K_H, K_M ~ col_K - BG_K, col_A - BG_A (IWM + backgrounds,
        measured below the mixed layer), N2 = -col_estrat/K_H,
        p_avm = K_M(s-1), p_sh2 = col_eshear(s); actual ratio K_H/K_M(s).
Only faces where both closures are well above the backgrounds (K_M > 1e-5).
Median over interior columns, last step of each hour.  No verdict printed.
usage: omip_strip_prandtl_offline.py OUR_RUN NEMO_EN2_W
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omip_strip_tke_ledger as L   # noqa: E402

RI_CRI = 2.0 / (2.0 + 0.7 / 0.1)
BSHEAR = 1.0e-20
BG_A, BG_K = 3.40e-6, 6.04e-7     # ours below-ML viscosity/diffusivity (h04 strip median)
DEPTHS_M = (5.8, 12.8, 17.9)


def pdlr(n2, avm_prev, sh2):
    ri = np.where(n2 > 0, n2 * avm_prev / (sh2 + BSHEAR), 0.0)
    return np.maximum(0.1, RI_CRI / np.maximum(RI_CRI, ri))


def main(argv):
    a = [np.load(f"{argv[0]}/{fn}") for fn in ("snapshot_day0001.npz", "snapshot_final.npz")]
    g = lambda k: np.concatenate([x[k] for x in a]).astype(np.float64)
    cj, ci = a[0]["col_j"], a[0]["col_i"]
    sj, si = cj - L.J0, ci - L.I0
    keep = (sj >= L.BUF) & (sj < 13 - L.BUF) & (si >= L.BUF) & (si < 31 - L.BUF)
    KH = g("col_K")[:, keep] - BG_K
    KM = g("col_A")[:, keep] - BG_A
    n2_o = -g("col_estrat")[:, keep] / np.where(KH > 0, KH, np.nan)
    sh_o = g("col_eshear")[:, keep]
    import netCDF4 as nc
    d = nc.Dataset(argv[1]); nk = KH.shape[2]
    def n(v):
        x = np.asarray(d[v][:, 1:nk + 1], dtype=np.float64)[:, :, sj[keep], si[keep]].transpose(0, 2, 1)
        x[np.abs(x) > 1e15] = np.nan
        return x
    avtk, avmk, est, shn = n("avt_k"), n("avm_k"), n("estrat_k"), n("eshear_k")
    n2_n = -est / np.where(avtk > 0, avtk, np.nan)
    depw = np.asarray(d["depthw"][1:nk + 1])
    ks = [int(np.argmin(np.abs(depw - z))) for z in DEPTHS_M]
    print(f"{argv[0]}: {keep.sum()} cols; depths " + " ".join(f"{depw[k]:.1f}" for k in ks))
    print("hr | per depth: ours actual K/A | NEMO-formula on OURS | NEMO actual | NEMO-formula on NEMO | n_ours n_nemo")
    for h in range(1, KH.shape[0] // 24 + 1):
        s = h * 24 - 1
        cells = []
        for k in ks:
            oo = (KM[s, :, k] > 1e-5) & (KM[s - 1, :, k] > 1e-5) & np.isfinite(n2_o[s, :, k])
            on = (avmk[s, :, k] > 1e-5) & (avmk[s - 1, :, k] > 1e-5) & np.isfinite(n2_n[s, :, k])
            if oo.sum() < 5 or on.sum() < 5:
                cells.append("   (closure at floor)                 ")
                continue
            ro = np.median(KH[s, oo, k] / KM[s, oo, k])
            fo = np.median(pdlr(n2_o[s, oo, k], KM[s - 1, oo, k], sh_o[s, oo, k]))
            rn = np.median(avtk[s, on, k] / avmk[s, on, k])
            fn = np.median(pdlr(n2_n[s, on, k], avmk[s - 1, on, k], shn[s, on, k]))
            cells.append(f"{ro:5.2f} {fo:5.2f} {rn:5.2f} {fn:5.2f} {oo.sum():3d} {on.sum():3d}")
        print(f"h{h:02d} | " + " | ".join(cells))
    # Richardson split at the middle depth: median N2 and S2 = p_sh2/p_avm,
    # each side's own state; per-column ratios on faces valid on both sides
    k = ks[1]
    print(f"\nRi split @ {depw[k]:.1f} m: median N2 ours, NEMO | S2 ours, NEMO | Ri ours, NEMO | col-ratio N2 o/N, S2 o/N")
    for h in range(1, KH.shape[0] // 24 + 1):
        s = h * 24 - 1
        ok = ((KM[s - 1, :, k] > 1e-5) & (avmk[s - 1, :, k] > 1e-5) & np.isfinite(n2_o[s, :, k])
              & np.isfinite(n2_n[s, :, k]) & (sh_o[s, :, k] > 0) & (shn[s, :, k] > 0))
        if ok.sum() < 5:
            continue
        s2o = sh_o[s, ok, k] / KM[s - 1, ok, k]; s2n = shn[s, ok, k] / avmk[s - 1, ok, k]
        no, nn = n2_o[s, ok, k], n2_n[s, ok, k]
        print(f"h{h:02d} | {np.median(no):.2e} {np.median(nn):.2e} | {np.median(s2o):.2e} {np.median(s2n):.2e} | "
              f"{np.median(no / s2o):5.2f} {np.median(nn / s2n):5.2f} | {np.median(no / nn):5.2f} {np.median(s2o / s2n):5.2f}  n={ok.sum()}")


if __name__ == "__main__":
    main(sys.argv[1:])

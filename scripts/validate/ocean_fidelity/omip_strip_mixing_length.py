"""Hour-by-hour TKE mixing length, ours vs NEMO, equatorial strip (probe).

Step-entry length l = sqrt(e_entry) / dissl_entry on both sides (nn_mxl=2:
one length):
  NEMO  dissl_in_k (saved dissl entering tke_tke), en_in_k        (ORCA1EN2)
  ours  dissl_used = 2*ediss/(c_eps*e_sol) is the solve's linearisation
        sqrt(max(e_pre, e_bg))/l_eps_used (tke.py), so
        l = sqrt(max(e_in, e_bg)) / dissl_used.
Also the buoyancy length sqrt(2 e)/N from each side's own e and N2 (NEMO
avt_k-free: N2 recomputed from the column T/S is NOT done here -- we use
l_buoy only where both sides report it) -- omitted; prints l and e.
Median over interior wet columns, last step of each hour; floor-bound
samples (e <= 1.0001*emin either side) excluded.  No verdict printed.

usage: omip_strip_mixing_length.py OUR_RUN NEMO_EN2_W
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omip_strip_tke_ledger as L   # noqa: E402

C_EPS, E_BG = 0.7, 1.0e-10
DEPTHS_M = (5.8, 12.8, 17.9)


def main(argv):
    a = [np.load(f"{argv[0]}/{fn}") for fn in ("snapshot_day0001.npz", "snapshot_final.npz")]
    g = lambda k: np.concatenate([x[k] for x in a]).astype(np.float64)
    cj, ci = a[0]["col_j"], a[0]["col_i"]
    sj, si = cj - L.J0, ci - L.I0
    keep = (sj >= L.BUF) & (sj < 13 - L.BUF) & (si >= L.BUF) & (si < 31 - L.BUF)
    e_in, e_sol, ediss = (g(k)[:, keep] for k in ("col_e_in", "col_e_sol", "col_ediss"))
    dissl_o = 2.0 * ediss / (C_EPS * np.where(e_sol > 0, e_sol, np.nan))
    l_o = np.sqrt(np.maximum(e_in, E_BG)) / dissl_o
    import netCDF4 as nc
    d = nc.Dataset(argv[1])
    nk = e_in.shape[2]
    def n(v):
        x = np.asarray(d[v][:, 1:nk + 1], dtype=np.float64)[:, :, sj[keep], si[keep]].transpose(0, 2, 1)
        x[np.abs(x) > 1e15] = np.nan
        return x
    en_in, dl_in = n("en_in_k"), n("dissl_in_k")
    l_n = np.sqrt(en_in) / np.where(dl_in > 0, dl_in, np.nan)
    depw = np.asarray(d["depthw"][1:nk + 1])
    ks = [int(np.argmin(np.abs(depw - z))) for z in DEPTHS_M]
    bad = (e_in <= L.EMIN * 1.0001) | (en_in <= L.EMIN * 1.0001)
    print(f"{argv[0]}: {keep.sum()} interior cols; depths " + " ".join(f"{depw[k]:.1f}" for k in ks))
    print("hr | per depth: l ours [m]  l NEMO [m]  ratio(med of col ratios)  e ratio  n")
    for h in range(1, e_in.shape[0] // 24 + 1):
        s = h * 24 - 1
        cells = []
        for k in ks:
            ok = ~bad[s, :, k] & np.isfinite(l_o[s, :, k]) & np.isfinite(l_n[s, :, k])
            if not ok.any():
                cells.append("   (all floor-bound)        ")
                continue
            cells.append(f"{np.median(l_o[s, ok, k]):6.2f} {np.median(l_n[s, ok, k]):6.2f} "
                         f"{np.median(l_o[s, ok, k] / l_n[s, ok, k]):5.2f} "
                         f"{np.median(e_in[s, ok, k] / en_in[s, ok, k]):5.2f} {ok.sum():3d}")
        print(f"h{h:02d} | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])

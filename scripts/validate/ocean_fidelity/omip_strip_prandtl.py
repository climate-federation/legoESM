"""Hour-by-hour viscosity, diffusivity and their ratio, ours vs NEMO (probe).

ours  col_A (viscosity), col_K (heat diffusivity): assembled closure + IWM +
      backgrounds, as applied by the solve
NEMO  avm, avt (closure + IWM; avt also EVD) -- faces where EVD fired
      (avt - avt_k > 50) are excluded on both sides.
Interior strip columns, last step of each hour, median of per-column ratios.
No verdict printed.   usage: omip_strip_prandtl.py OUR_RUN NEMO_EN2_W
"""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omip_strip_tke_ledger as L   # noqa: E402

DEPTHS_M = (5.8, 12.8, 17.9)


def main(argv):
    a = [np.load(f"{argv[0]}/{fn}") for fn in ("snapshot_day0001.npz", "snapshot_final.npz")]
    g = lambda k: np.concatenate([x[k] for x in a]).astype(np.float64)
    cj, ci = a[0]["col_j"], a[0]["col_i"]
    sj, si = cj - L.J0, ci - L.I0
    keep = (sj >= L.BUF) & (sj < 13 - L.BUF) & (si >= L.BUF) & (si < 31 - L.BUF)
    K, A = g("col_K")[:, keep], g("col_A")[:, keep]
    import netCDF4 as nc
    d = nc.Dataset(argv[1]); nk = K.shape[2]
    def n(v):
        x = np.asarray(d[v][:, 1:nk + 1], dtype=np.float64)[:, :, sj[keep], si[keep]].transpose(0, 2, 1)
        x[np.abs(x) > 1e15] = np.nan
        return x
    avt, avm, avtk = n("avt"), n("avm"), n("avt_k")
    evd = (avt - avtk) > 50.0
    depw = np.asarray(d["depthw"][1:nk + 1])
    ks = [int(np.argmin(np.abs(depw - z))) for z in DEPTHS_M]
    print(f"{argv[0]}: {keep.sum()} cols; depths " + " ".join(f"{depw[k]:.1f}" for k in ks))
    print("hr | per depth: K/avt  A/avm  (K/A)/(avt/avm)  NEMO avt/avm  n")
    for h in range(1, K.shape[0] // 24 + 1):
        s = h * 24 - 1
        cells = []
        for k in ks:
            ok = ~evd[s, :, k] & np.isfinite(avt[s, :, k]) & (avt[s, :, k] > 0) & (A[s, :, k] > 0)
            rk = K[s, ok, k] / avt[s, ok, k]; ra = A[s, ok, k] / avm[s, ok, k]
            pr = (K[s, ok, k] / A[s, ok, k]) / (avt[s, ok, k] / avm[s, ok, k])
            cells.append(f"{np.median(rk):5.2f} {np.median(ra):5.2f} {np.median(pr):5.2f} "
                         f"{np.median(avt[s, ok, k] / avm[s, ok, k]):5.2f} {ok.sum():3d}")
        print(f"h{h:02d} | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])

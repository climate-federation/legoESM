"""Hour-by-hour TKE budget, ours vs NEMO, in the equatorial strip (probe).

Ours: --trd-columns col_e/col_eshear/col_estrat/col_ediss per step, defined as
NEMO's en_k/eshear_k/estrat_k/ediss_k (tke.py TKEOutput.budget docstring).
NEMO: RUN_DT150_EN eqs1ts_W (every step).  Index map (memory, codex-checked):
ours (j, i) = strip (j - J0, i - I0); our interface k = NEMO w-level k + 1.
Prints, per hour and depth, the median over interior wet columns of
ours/NEMO for each term (last step of the hour).  No verdict printed.

usage: omip_strip_tke_budget_onset.py RUN_DIR [NEMO_W]
"""
import sys
import numpy as np

NEMO_W = ('/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/'
          'RUN_DT150_EN/ORCA1_1ts_20000101_20000102_eqs1ts_W.nc')
J0, I0, BUF = 180, 153, 3
TERMS = (("e", "en_k"), ("shear", "eshear_k"), ("strat", "estrat_k"), ("diss", "ediss_k"))
DEPTHS_M = (5.8, 12.8, 17.9, 24.6)
STEPS_PER_HOUR = 24


def load_ours(run):
    a = np.load(f"{run}/snapshot_day0001.npz")
    b = np.load(f"{run}/snapshot_final.npz")
    cat = lambda k: np.concatenate([a[k], b[k]], axis=0)
    out = {t: cat("col_" + ("e" if t == "e" else "e" + t)) for t, _ in TERMS}
    return out, a["col_j"], a["col_i"]


def pair(ours, cj, ci, nemo_path):
    import netCDF4 as nc
    d = nc.Dataset(nemo_path)
    depw = np.asarray(d["depthw"][:])
    sj, si = cj - J0, ci - I0
    nj, ni = d["en_k"].shape[2:]
    keep = (sj >= BUF) & (sj < nj - BUF) & (si >= BUF) & (si < ni - BUF)
    nk = ours["e"].shape[2]
    res = {}
    for t, v in TERMS:
        x = np.asarray(d[v][:, 1:nk + 1], dtype=np.float64)
        x[np.abs(x) > 1e15] = np.nan
        res[t] = (ours[t][:, keep, :].astype(np.float64),
                  x[:, :, sj[keep], si[keep]].transpose(0, 2, 1))
    return res, depw[1:nk + 1], int(keep.sum())


def main(argv):
    ours, cj, ci = load_ours(argv[0])
    res, depw, ncol = pair(ours, cj, ci, argv[1] if len(argv) > 1 else NEMO_W)
    nt = res["e"][0].shape[0]
    if res["e"][1].shape[0] != nt:
        raise SystemExit(f"step mismatch ours {nt} NEMO {res['e'][1].shape[0]}")
    ks = [int(np.argmin(np.abs(depw - z))) for z in DEPTHS_M]
    print(f"{argv[0]}: {nt} steps, {ncol} interior cols; depths "
          + " ".join(f"{depw[k]:.1f}" for k in ks))
    print("median ours/NEMO (last step of hour) per depth: " + " | ".join(t for t, _ in TERMS))
    for h in range(1, nt // STEPS_PER_HOUR + 1):
        s = h * STEPS_PER_HOUR - 1
        cells = []
        for k in ks:
            r = []
            for t, _ in TERMS:
                o, n = res[t][0][s, :, k], res[t][1][s, :, k]
                ok = np.isfinite(o) & np.isfinite(n) & (np.abs(n) > 0)
                if not ok.any():
                    raise SystemExit(f"no finite pairs h{h} k{k} {t}")
                r.append(np.median(o[ok] / n[ok]))
            cells.append(" ".join(f"{x:6.2f}" for x in r))
        print(f"h{h:02d} | " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])

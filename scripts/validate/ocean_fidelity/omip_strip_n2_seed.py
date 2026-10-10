"""Where does our 12.8 m stratification go? Thermal/haline split and a
vertical-diffusion vs remainder budget of the jump across the interface,
ours vs NEMO, equatorial strip (probe).

Part 1 (hourly): N2 at interface k (below T-level k) = g*(a dT - b dS)/dz,
linear-EOS proxy (a=3e-4, b=7.5e-4, same on both sides, as the xplant
scorer), split into thermal and haline parts.
Part 2 (window): change of the jumps dT = T_k - T_k+1, dS likewise, over
[H0, H1], and the part explained by vertical diffusion, from each model's
own applied diffusivity (ours col_K, NEMO avt incl. EVD; salt uses the same
coefficient) and mesh e3t_0/e3w_0, backward-Euler fluxes on the post-step
profile, dt = 150 s.  Remainder = everything else (advection, lateral
mixing, solar -- night window -> solar ~0).  Converted to N2 units.
Median over interior wet columns; ours-NEMO with IQR.  No verdict printed.
usage: omip_strip_n2_seed.py OUR_RUN NEMO_EN2_DIR [K H0 H1]
"""
import sys
import numpy as np
from legoesm import constants

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import omip_strip_tke_ledger as L   # noqa: E402

ALPHA, BETA, DT = 3.0e-4, 7.5e-4, 150.0   # coeff-ok: linear-EOS proxy shared with omip_strip_xplant_score
PREFIX = "ORCA1_1ts_20000101_20000102_eqs1ts_"


def main(argv):
    k = int(argv[2]) if len(argv) > 2 else 8
    h0, h1 = (int(argv[3]), int(argv[4])) if len(argv) > 4 else (4, 9)
    import netCDF4 as nc
    a = [np.load(f"{argv[0]}/{fn}") for fn in ("snapshot_day0001.npz", "snapshot_final.npz")]
    g = lambda key: np.concatenate([x[key] for x in a]).astype(np.float64)
    cj, ci = a[0]["col_j"], a[0]["col_i"]
    sj, si = cj - L.J0, ci - L.I0
    keep = (sj >= L.BUF) & (sj < 13 - L.BUF) & (si >= L.BUF) & (si < 31 - L.BUF)
    To, So, Ko = g("col_T")[:, keep], g("col_S")[:, keep], g("col_K")[:, keep]
    sj, si = sj[keep], si[keep]
    def n(fn, v, nk):
        x = np.asarray(nc.Dataset(f"{argv[1]}/{PREFIX}{fn}.nc")[v][:, :nk], dtype=np.float64)
        x = x[:, :, sj, si].transpose(0, 2, 1); x[np.abs(x) > 1e15] = np.nan
        return x
    nlev = To.shape[2]
    Tn, Sn = n("T", "votemper", nlev), n("T", "vosaline", nlev)
    Kn = n("W", "avt", nlev)[:, :, 1:]               # w-level i+1 = our interface i
    m = nc.Dataset(L.MESH)
    zt = np.asarray(m["gdept_1d"][0])
    e3t = np.asarray(m["e3t_0"][0])[:nlev, sj + L.J0, si + L.I0].T
    e3w = np.asarray(m["e3w_0"][0])[1:nlev, sj + L.J0, si + L.I0].T
    wet = np.all(np.isfinite(Tn[0, :, :k + 3]), 1) & np.all(To[0, :, :k + 3] != 0, 1)
    dz = zt[k + 1] - zt[k]
    G = constants.g
    print(f"{argv[0]}: interface below T-level {k} (z {zt[k]:.1f}->{zt[k + 1]:.1f} m), {wet.sum()} wet interior cols")
    print("Part 1 (end of hour): median N2 total / thermal / haline  ours | NEMO   [1e-5 s-2]; col ratio total o/N")
    for h in range(0, 16):
        s = max(h * 24 - 1, 0)
        out = []
        for T, S in ((To, So), (Tn, Sn)):
            nt = G * ALPHA * (T[s, wet, k] - T[s, wet, k + 1]) / dz
            ns = -G * BETA * (S[s, wet, k] - S[s, wet, k + 1]) / dz
            out.append((nt + ns, nt, ns))
        r = np.median(out[0][0] / out[1][0])
        print(f"h{h:02d} | " + " | ".join(" ".join(f"{np.median(x) * 1e5:6.2f}" for x in o) for o in out) + f" | {r:5.2f}")

    def budget(T, K):
        """cumulative change of the jump T_k - T_k+1 over the window and its
        vertical-diffusion part (sum over steps, backward-Euler fluxes)."""
        s0, s1 = h0 * 24 - 1, h1 * 24 - 1
        D = lambda s, i: K[s, :, i] * (T[s, :, i] - T[s, :, i + 1]) / e3w[:, i]    # down-gradient flux, + downward
        diff = np.zeros(T.shape[1])
        for s in range(s0 + 1, s1 + 1):
            dTk = (D(s, k - 1) - D(s, k)) / e3t[:, k]
            dTk1 = (D(s, k) - D(s, k + 1)) / e3t[:, k + 1]
            diff += DT * (dTk - dTk1)
        tot = (T[s1, :, k] - T[s1, :, k + 1]) - (T[s0, :, k] - T[s0, :, k + 1])
        return tot, diff
    print(f"\nPart 2: window h{h0}-h{h1}, change of the 12.8 m jump in N2 units [1e-5 s-2]: total | vertical diffusion | remainder")
    res = {}
    for lab, (T, S, K) in (("ours", (To, So, Ko)), ("NEMO", (Tn, Sn, Kn))):
        tT, dTd = budget(T, K); tS, dSd = budget(S, K)
        f = lambda x, y: (G * ALPHA * x - G * BETA * y) / dz * 1e5
        res[lab] = dict(tot=f(tT, tS), diff=f(dTd, dSd), rem=f(tT - dTd, tS - dSd),
                        totT=G * ALPHA * tT / dz * 1e5, totS=-G * BETA * tS / dz * 1e5)
        r = res[lab]
        print(f"  {lab:4s} total {np.median(r['tot'][wet]):7.2f} (thermal {np.median(r['totT'][wet]):6.2f}, haline {np.median(r['totS'][wet]):6.2f})"
              f" | diffusion {np.median(r['diff'][wet]):7.2f} | remainder {np.median(r['rem'][wet]):7.2f}")
    for key in ("tot", "diff", "rem", "totT", "totS"):
        d = (res["ours"][key] - res["NEMO"][key])[wet]
        q = np.percentile(d, [25, 75])
        print(f"  ours-NEMO {key:5s} median {np.median(d):7.2f}  IQR [{q[0]:7.2f},{q[1]:7.2f}]")


if __name__ == "__main__":
    main(sys.argv[1:])

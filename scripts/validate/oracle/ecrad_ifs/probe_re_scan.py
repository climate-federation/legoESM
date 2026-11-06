"""Scan along h (re*(1+h)) for the worst SW kernel elements found by probe_re_fd.py at
sharpness 100: residual from a least-squares quadratic of (a) the element's inputs
od/ssa/g (built at sharpness 100) and (b) calc_ref_trans_sw outputs evaluated on THOSE
SAME inputs at sharpness 100 and at sharpness inf.  Jitter in (a) => inputs; jitter only
in (b, 100) => the kernel's smooth forms.

Provenance (vote C10): copied unchanged from /burg/glab/users/pg2328/oracle_audit/ifs_port/ecrad/agents/end_to_end/probe_re_scan.py (end-to-end agent's
diagnosis of the FD-vs-AD finding of vote C7; sha256 prefix a3f08f29b539fc52); only this
paragraph is added.
"""
import sys
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.argv = [sys.argv[0], "100"]
import importlib.util
spec = importlib.util.spec_from_file_location("p", __file__.replace("probe_re_scan.py", "probe_re_fd_lib.py"))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
ts = p.ts
hs = np.linspace(-3e-6, 3e-6, 25)
cases = {"liq": [(4, 103, 84), (2, 118, 83), (6, 124, 85)], "ice": [(6, 85, 85), (4, 68, 85), (7, 95, 81)]}
mu = np.asarray(p.inp["cos_sza"])
for which, els in cases.items():
    g = jax.jit(p.along(p.kernels, which))
    path = [g(float(h)) for h in hs]
    for (c, l, k) in els:
        ins = {q: np.array([float(o[f"sw.{q}"][c, l, k]) for o in path]) for q in ("od", "ssa", "g")}
        print(f"== re_{which} element {(c, l, k)} od={ins['od'][12]:.4e} 1-ssa={1 - ins['ssa'][12]:.3e} g={ins['g'][12]:.4f} mu0={mu[c]:.3f}")
        for q, v in ins.items():
            r = v - np.polyval(np.polyfit(hs, v, 2), hs)
            print(f"   input {q:4s}: |slope|={abs(np.polyfit(hs, v, 1)[0]):.2e} max|resid|={np.abs(r).max():.2e} (rel {np.abs(r).max() / abs(v[12]):.1e})")
        kern = {S: np.stack([np.asarray(x) for x in jax.jit(lambda o, w, gg, S=S: ts.calc_ref_trans_sw(
            mu[c], o, w, gg, S))(*(jnp.asarray(ins[q]) for q in ("od", "ssa", "g")))], axis=1) for S in (100.0, np.inf)}
        jit_path = np.array([[float(o[f"sw.{n}"][c, l, k]) for n in ("Rd", "Td", "Rdir", "Tdd", "Tdir")] for o in path])
        print(f"   bitwise kernel(s=100) == kernel(s=inf) along the path: {np.array_equal(kern[100.0], kern[np.inf])}; "
              f"max|kernel(s=100) - full jitted probe path| = {np.abs(kern[100.0] - jit_path).max():.1e}")
        # fp64 vs 60-digit evaluation of Rd, Td (Meador-Weaver, k^2 above the floor here)
        import mpmath
        mpmath.mp.dps = 60
        hp = []
        for i in range(len(hs)):
            od, w, gg, m0 = (mpmath.mpf(float(v)) for v in (ins["od"][i], ins["ssa"][i], ins["g"][i], mu[c]))
            f = mpmath.mpf("0.75") * gg
            g1, g2 = 2 - w * (mpmath.mpf("1.25") + f), w * (mpmath.mpf("0.75") - f)
            kk = mpmath.sqrt((g1 - g2) * (g1 + g2))
            e = mpmath.exp(-kk * od)
            rtf = 1 / (kk + g1 + (kk - g1) * e * e)
            hp.append([float(g2 * (1 - e * e) * rtf), float(2 * kk * e * rtf)])
        hp = np.array(hp)
        for j, n in enumerate(("Rd", "Td")):
            r = hp[:, j] - np.polyval(np.polyfit(hs, hp[:, j], 2), hs)
            print(f"   60-digit {n}: max|resid|={np.abs(r).max():.2e}; max|fp64 - 60-digit|={np.abs(kern[100.0][:, j] - hp[:, j]).max():.2e}")
        for S in (100.0, np.inf):
            outs = kern[S]
            for j, n in enumerate(("Rd", "Td", "Rdir", "Tdd", "Tdir")):
                v = outs[:, j]
                r = v - np.polyval(np.polyfit(hs, v, 2), hs)
                sl = abs(np.polyfit(hs, v, 1)[0])
                print(f"   kernel s={S:<5} {n:4s}: value={v[12]:.6e} |slope|={sl:.2e} max|resid|={np.abs(r).max():.2e} "
                      f"=> FD err at h=1e-6 ~ {np.abs(r).max() / 1e-6 / max(sl, 1e-300):.1e} rel")
print("SCAN_DONE")

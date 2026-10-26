"""Run the WRF v4.8.0 Tiedtke oracle and write the legoESM oracle fixture.

usage: python make_fixture.py <columns.npz> <build dir> <out.npz>

<columns.npz> are the 21 tier-2 audit columns (15 tier-1 ZM columns + 6
mixed-boundary-layer variants) with the audit's two forcings F1 / F2
(5 mm/day large-scale moisture convergence over the lowest 300 hPa; F2 adds
3 mm/day surface evaporation as a PBL tendency over the lowest 100 hPa and as
qfx), w = 0, znu = p/p_s, dt = 300 s.  The driver's stdin is built exactly as
the audit harness built it; <build dir> holds tdriver.x / tdriver_chk.x from
build.sh.  The fixture stores the inputs of ``tiedtke_wrf`` (top-down,
specific humidity, geopotential formed as WRF's cu_tiedtke forms it from
dz8w) and the oracle outputs.
"""
import json
import subprocess
import sys

import numpy as np

from legoesm import constants as C

DT = 300.0
SHA = "06d4240ae989cc3e50af412bb472df3d9048783c"


def oracle_stdin(c, fk):
    """Audit harness stage ``inputs`` (bottom-up WRF arrays, r = q/(1-q))."""
    T, q, pf, ph, u, v = (c[k] for k in ("T", "q", "p_full", "p_half", "u", "v"))
    ncol, nlev = T.shape
    xland = np.where(c["land"] > 0.5, 1.0, 2.0)
    pi = (pf / C.p_ref) ** C.kappa
    rho, dz = c["rho_lego"], c["dz"]
    qvften, qvpblten, qfx = c[f"{fk}_qvften"], c[f"{fk}_qvpblten"], c[f"{fk}_qfx"]
    lines = [f"{ncol} {nlev} {DT!r}"]
    for i in range(ncol):
        lines.append(f"{float(xland[i])!r} {float(qfx[i])!r}")
        sig = pf[i] / ph[i, -1]
        for kk in range(nlev):
            k = nlev - 1 - kk
            r = q[i, k] / (1.0 - q[i, k])
            vals = [u[i, k], v[i, k], T[i, k], r, 0.0, 0.0, pi[i, k], rho[i, k],
                    qvften[i, k], qvpblten[i, k], dz[i, k], pf[i, k], sig[k]]
            lines.append(" ".join(repr(float(x)) for x in vals))
        for kk in range(nlev + 1):
            lines.append(f"{float(ph[i, nlev - kk])!r} 0.0")
    return "\n".join(lines) + "\n"


def parse(text):
    cols, cur = [], None
    for line in text.splitlines():
        if line.startswith("COL "):
            cur = {}
            cols.append(cur)
        elif line.startswith("S ") and cur is not None:
            _, k, val = line.split()
            cur[k] = float(val)
        elif line.startswith("A ") and cur is not None:
            p = line.split()
            cur[p[1]] = np.array([float(x) for x in p[2:]])
    return cols


def kernel_inputs(stdin):
    """Re-read the driver stdin and form what cu_tiedtke hands to tiecnv:
    flip to top-down, q = r/(1+r), geopotential g*zl from dz8w (zi/zl loops of
    cu_tiedtke), omega = -0.5*g*rho*(w_k + w_k+1)."""
    it = iter(stdin.splitlines())
    ncol, nlev, _ = next(it).split()
    ncol, nlev = int(ncol), int(nlev)
    out = {k: [] for k in ("T", "q_v", "u", "v", "p_full", "p_half", "geopotential",
                           "omega", "dq_v_dt_dyn", "dq_v_dt_pbl", "sigma")}
    qfx, land = [], []
    for _ in range(ncol):
        xl, qf = (float(x) for x in next(it).split())
        land.append(int(abs(xl - 2.0)) == 1)
        qfx.append(qf)
        rows = np.array([[float(x) for x in next(it).split()] for _ in range(nlev)])
        half = np.array([[float(x) for x in next(it).split()] for _ in range(nlev + 1)])
        u, v, t, r, _qc, _qi, _pi, rho, qvf, qvbl, dz, pcps, znu = rows.T
        p8w, w = half.T
        zi = np.zeros(nlev)
        for k in range(1, nlev):
            zi[k] = zi[k - 1] + dz[k - 1]
        zl = np.zeros(nlev)
        for k in range(1, nlev):
            zl[k - 1] = (zi[k] + zi[k - 1]) * 0.5
        zl[nlev - 1] = 2.0 * zi[nlev - 1] - zl[nlev - 2]
        dot = -0.5 * C.g * rho * (w[:-1] + w[1:])
        f = lambda a: a[::-1]
        out["T"].append(f(t)); out["q_v"].append(f(r / (1.0 + r)))
        out["u"].append(f(u)); out["v"].append(f(v)); out["p_full"].append(f(pcps))
        out["p_half"].append(f(p8w)); out["geopotential"].append(f(C.g * zl))
        out["omega"].append(f(dot)); out["dq_v_dt_dyn"].append(f(qvf))
        out["dq_v_dt_pbl"].append(f(qvbl)); out["sigma"].append(f(znu))
    res = {k: np.array(a) for k, a in out.items()}
    res["surface_moisture_flux"] = np.array(qfx)
    res["land"] = np.array(land)
    return res


def main(columns, build, out):
    c = dict(np.load(columns))
    fx = {"names": c["names"]}
    for fk in ("F1", "F2"):
        stdin = oracle_stdin(c, fk)
        res = [subprocess.run([f"{build}/{exe}"], input=stdin, capture_output=True,
                              text=True, check=True).stdout
               for exe in ("tdriver.x", "tdriver_chk.x")]
        assert res[0] == res[1], "bounds-checked oracle differs"
        cols = parse(res[0])
        for k, a in kernel_inputs(stdin).items():
            fx[f"{fk}/in/{k}"] = a
        for k in ("dtdt", "dqdt", "dcdt", "dudt", "dvdt", "mfu", "mfd"):
            fx[f"{fk}/out/{k}"] = np.array([o[k] for o in cols])
        for k in ("rqccuten", "rqicuten"):   # WRF arrays are bottom-up
            fx[f"{fk}/out/{k}"] = np.array([o[k][::-1] for o in cols])
        for k in ("prec", "mfub", "ldcum", "ktype", "kcbot", "kctop", "icum"):
            fx[f"{fk}/out/{k}"] = np.array([o[k] for o in cols])
        print(fk, "oracle-active", int(fx[f"{fk}/out/ldcum"].sum()), "of", len(cols))
    fx["meta"] = json.dumps(dict(
        oracle="WRF v4.8.0 phys/module_cu_tiedtke.F", sha=SHA,
        md5="1d9127ddd05b9e314ae95b847043e155", variant="V1: -DLEGOCONST -DLEGOSAT fp64",
        dt=DT, compiler=subprocess.run(["gfortran", "--version"], capture_output=True,
                                       text=True).stdout.splitlines()[0]))
    np.savez_compressed(out, **fx)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])

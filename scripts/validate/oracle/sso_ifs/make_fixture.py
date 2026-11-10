"""Run the OpenIFS 49r1 Lott-Miller subgrid-orography drag oracle and write the fixture.

usage: python make_fixture.py <columns dir> <build dir> <out.npz>

<columns dir> holds the tier-2 audit columns ``columns_{L32,L137,STRESS}.npz``
(the same 24 atmosphere columns as the IFS convection oracle); their T, u, v,
full/half pressure and full-level geopotential above the surface are used.
Each column is run with every SSO parameter set in ``SSO`` (vote V1a), chosen
so that every branch fires somewhere (off below 50 m and on just above it, blocking, critical level,
Richardson-limited breaking, GTENLIM clip; see the fixture's integer dump);
the last set is deliberately extreme (sigma 0.3) to reach the upper-level
GTENLIM clip and the model-top region.
<build dir> holds ``oracle.x`` from build.sh.

Arms (vote V3a): L = legoESM g, R_d, c_pd (CONST_MODE 1), N = 49r1 native;
C = as L with NAMGWD GTENLIM = 1e-3 m s-2 (clip coverage).
The fixture stores the inputs exactly as the driver read them (text round trip)
and, per arm, SOTEU, SOTEV, SOBETA, the gamma GWSETUP leaves, the TOFD
coefficient PTOFDC (VDFTOFDC on the same column with PSIGFLT), and the integers
NKTOPG NGWDTOP NGWDLIM KKCRITH KCRIT KKENVH KKNU KKNU2.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C

GROUPS = ("L32", "L137", "STRESS")
# (std m, gamma, theta rad, sigma, filtered-orography std m for TOFD)
SSO = ((30.0, 0.6, 0.3, 0.02, 5.0),
       (55.0, 0.6, 0.3, 0.02, 10.0),
       (250.0, 0.15, -1.2, 0.004, 40.0),
       (700.0, 0.6, 0.3, 0.02, 150.0),
       (1500.0, 1.0, 1.5, 0.06, 400.0),
       (1500.0, 0.15, 0.3, 0.06, 0.0),
       (700.0, 1.0, -1.2, 0.004, 250.0),
       (2500.0, 0.15, 0.3, 0.3, 600.0))
ARMS = {"L": dict(CONST_MODE=1, CRG=C.g, CRD=C.R_d, CRCPD=C.c_pd), "N": dict(CONST_MODE=2),
        "C": dict(CONST_MODE=1, CRG=C.g, CRD=C.R_d, CRCPD=C.c_pd)}
# NAMGWD per arm; arm C lowers the 49r1 namelist GTENLIM so the upper-level
# tendency clip (never reached at the 80 m/s/h code value) is exercised.
NAMGWD = {"L": "", "N": "", "C": "GTENLIM=1.0E-3"}
INTS = ("NKTOPG", "NGWDTOP", "NGWDLIM", "KKCRITH", "KCRIT", "KKENVH", "KKNU", "KKNU2")


def _r(x):
    """The value the driver reads back from its '%.17e' text."""
    return np.array([float(f"{float(v):.17e}") for v in np.ravel(x)]).reshape(np.shape(x))


def inputs(d):
    """Columns x SSO sets of one group, rounded through the driver's text."""
    ncol = d["T"].shape[0]
    x = {k: [] for k in ("names", "std", "gamma", "theta", "sigma", "sigflt", "p_half", "p_full",
                         "geo_full", "T", "u", "v")}
    for i in range(ncol):
        for j, (s, g, th, sg, sf) in enumerate(SSO):
            x["names"].append(f"{d['names'][i]}_sso{j}")
            for k, v in (("std", s), ("gamma", g), ("theta", th), ("sigma", sg), ("sigflt", sf)):
                x[k].append(_r(v))
            for k in ("p_half", "p_full", "geo_full", "T", "u", "v"):
                x[k].append(_r(d[k][i]))
    return {k: np.array(v) for k, v in x.items()}


def oracle_text(x):
    n, nlev = x["T"].shape
    out = [f"{n} {nlev}"]
    fmt = lambda a: " ".join(f"{v:.17e}" for v in np.ravel(a))
    for i in range(n):
        out.append(f"COL {x['names'][i]}")
        out.append(fmt([x["std"][i], x["gamma"][i], x["theta"][i], x["sigma"][i], x["sigflt"][i]]))
        for k in ("p_half", "p_full", "geo_full", "T", "u", "v"):
            out.append(fmt(x[k][i]))
    return "\n".join(out) + "\n"


def parse(path):
    lines = Path(path).read_text().splitlines()
    n, nlev = (int(v) for v in lines[0].split())
    res, k = {}, 1
    for _ in range(n):
        name = lines[k].split()[1]
        rows = [np.array([float(v) for v in lines[k + 1 + j].split()]) for j in range(5)]
        ints = [int(v) for v in lines[k + 6].split()[1:]]
        res[name] = dict(soteu=rows[0], sotev=rows[1], sobeta=rows[2], gamma_out=rows[3][0],
                         tofdc=rows[4], **{nm.lower(): iv for nm, iv in zip(INTS, ints)})
        k += 7
    return res


def run(build, x, arm, tag):
    build = Path(build)
    inf, outf, nml = build / f"in_{tag}.txt", build / f"out_{tag}.txt", build / f"run_{tag}.nml"
    inf.write_text(oracle_text(x))
    body = ", ".join(f"{k}={v!r}" for k, v in ARMS[arm].items())
    nml.write_text(f"&RUN {body}, INFILE='{inf}', OUTFILE='{outf}' /\n&NAMSCEN /\n&NAMVDF /\n&NAMGWD {NAMGWD[arm]} /\n")
    r = subprocess.run([str(build / "oracle.x"), str(nml)], capture_output=True, text=True)
    if r.returncode != 0 or "DRIVER_DONE" not in r.stdout:
        raise SystemExit(f"oracle failed {tag}: {r.stdout[-400:]} {r.stderr[-400:]}")
    res = parse(outf)
    if list(res) != list(x["names"]):
        raise SystemExit(f"column order/names differ {tag}")
    return res, r.stdout


def main(cols, build, outp):
    fx = {}
    for g in GROUPS:
        x = inputs(np.load(Path(cols) / f"columns_{g}.npz", allow_pickle=True))
        for k, v in x.items():
            fx[f"{g}_{k}"] = v
        for arm in ARMS:
            res, log = run(build, x, arm, f"{arm}_{g}")
            fx[f"{arm}_{g}_log"] = np.array(log)
            for f in ("soteu", "sotev", "sobeta", "gamma_out", "tofdc") + tuple(i.lower() for i in INTS):
                fx[f"{arm}_{g}_{f}"] = np.array([res[nm][f] for nm in x["names"]])
    for arm, a in ARMS.items():
        for k, v in a.items():
            fx[f"{arm}_const_{k}"] = np.array(v)
    np.savez_compressed(outp, **fx)
    print("FIXTURE_OK", outp)


if __name__ == "__main__":
    main(*sys.argv[1:4])

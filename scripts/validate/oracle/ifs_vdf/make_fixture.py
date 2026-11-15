"""Run the OpenIFS 49r1 VDF column-kernel oracle and write the fixture.

usage: python make_fixture.py <columns dir> <build dir> <out.npz>

<columns dir> holds the tier-2 audit columns ``columns_{L32,L137,STRESS}.npz``
(the same 24 atmosphere columns as the IFS convection and SSO oracles).  Each
column runs with every surface set in ``SURF`` (downward-positive sensible and
latent heat flux W m-2, friction velocity m s-1; one filtered-orography
standard deviation per set for VDFTOFDC) so that stable, dry
convective, cloudy and shallow-cumulus boundary layers all occur; z0m is
1e-3 m over sea and 0.1 m over land.  Liquid water is the column's total
condensate (ice 0), the longwave heating rate is its radiative heating (SW 0),
and PQE is its advective plus other moisture tendency.
<build dir> holds ``oracle.x`` from build.sh.

Arms: N = 49r1 native constants (CONST_MODE 2); L = legoESM g, R_d, c_pd, R_v,
L_v (CONST_MODE 1).  The fixture stores the inputs exactly as the driver read
them (text round trip) and, per arm, every VDFDPBL/VDFHGHTN/VDFEXCU output the
driver prints.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C

GROUPS = ("L32", "L137", "STRESS")
# (SHF_down W m-2, LHF_down W m-2, u* m s-1, wind factor); None = the column's own flux.
# Sets 4-5 are targeted (vote 3, codex note): a weak sensible flux between the
# two VDFEXCU cumulus thresholds, and a weakly stable, windy column whose
# stable PBL height exceeds the 3000 m that caps the 300 m length (or has no
# Ri crossing at all, VDFDPBL's 1000 m default); set 6's 30x winds leave
# some columns with no Ri crossing below the model top.
SURF = ((None, None, 0.30, 1.0),
        (20.0, 0.0, 0.08, 1.0),
        (-150.0, -50.0, 0.50, 1.0),
        (-10.0, -250.0, 0.25, 1.0),
        (-2.0, -200.0, 0.25, 1.0),
        (2.0, 0.0, 0.60, 4.0),
        (2.0, 0.0, 0.60, 30.0))
# filtered-orography standard deviation (m) for VDFTOFDC, one per surface set
SIGFLT = (0.0, 40.0, 250.0, 600.0, 100.0, 1000.0, 0.0)
ARMS = {"N": dict(CONST_MODE=2),
        "L": dict(CONST_MODE=1, CRG=C.g, CRD=C.R_d, CRCPD=C.c_pd, CRV=C.R_v, CRLVTT=C.L_v)}
PTSPHY = 900.0
ARRAYS = ("p_half", "p_full", "geo_full", "geo_half", "T", "q", "u", "v", "ql", "qi", "hrlw",
          "hrsw", "qe")
SCAL = ("kmfl", "khfl", "kqfl", "pblh", "zinv", "zcldbase", "eis", "khvfl")
INTS = ("kinv", "kcbase", "kctop", "kpbltype", "njkt2", "njkt6")
ROWS = ("ri", "mflx", "slguh", "qtuh", "uuh", "vuh", "cfm", "cfh", "kh", "km", "tofdc")


def _r(x):
    """The value the driver reads back from its '%.17e' text."""
    return np.array([float(f"{float(v):.17e}") for v in np.ravel(x)]).reshape(np.shape(x))


def inputs(d):
    ncol = d["T"].shape[0]
    x = {k: [] for k in ("names", "land", "shf", "lhf", "ustar", "z0m", "sigflt") + ARRAYS}
    for i in range(ncol):
        for j, (shf, lhf, ust, wf) in enumerate(SURF):
            x["names"].append(f"{d['names'][i]}_sfc{j}")
            land = float(d["land"][i])
            x["land"].append(_r(land))
            x["shf"].append(_r(-d["shf_up"][i] if shf is None else shf))
            x["lhf"].append(_r(-d["lhf_up"][i] if lhf is None else lhf))
            x["ustar"].append(_r(ust))
            x["z0m"].append(_r(0.1 if land > 0.5 else 1.0e-3))
            x["sigflt"].append(_r(SIGFLT[j]))
            src = dict(p_half=d["p_half"][i], p_full=d["p_full"][i], geo_full=d["geo_full"][i],
                       geo_half=d["geo_half"][i], T=d["T"][i], q=d["q"][i], u=wf * d["u"][i],
                       v=wf * d["v"][i], ql=d["lit"][i], qi=np.zeros_like(d["lit"][i]),
                       hrlw=d["dT_rad"][i], hrsw=np.zeros_like(d["dT_rad"][i]),
                       qe=d["dq_adv"][i] + d["dq_other"][i])
            for k in ARRAYS:
                x[k].append(_r(src[k]))
    return {k: np.array(v) for k, v in x.items()}


def oracle_text(x):
    n, nlev = x["T"].shape
    out = [f"{n} {nlev}"]
    fmt = lambda a: " ".join(f"{v:.17e}" for v in np.ravel(a))
    for i in range(n):
        out.append(f"COL {x['names'][i]}")
        out.append(fmt([x[k][i] for k in ("land", "shf", "lhf", "ustar", "z0m", "sigflt")]))
        for k in ARRAYS:
            out.append(fmt(x[k][i]))
    return "\n".join(out) + "\n"


def parse(path):
    lines = Path(path).read_text().splitlines()
    n, _ = (int(v) for v in lines[0].split())
    res, k = {}, 1
    for _ in range(n):
        name = lines[k].split()[1]
        sc = [float(v) for v in lines[k + 1].split()[1:]]
        it = [int(v) for v in lines[k + 2].split()[1:]]
        rows = [np.array([float(v) for v in lines[k + 3 + j].split()]) for j in range(len(ROWS))]
        res[name] = {**dict(zip(SCAL, sc)), **dict(zip(INTS, it)), **dict(zip(ROWS, rows))}
        k += 3 + len(ROWS)
    return res


def run(build, x, arm, tag):
    build = Path(build)
    inf, outf, nml = build / f"in_{tag}.txt", build / f"out_{tag}.txt", build / f"run_{tag}.nml"
    inf.write_text(oracle_text(x))
    body = ", ".join(f"{k}={v!r}" for k, v in ARMS[arm].items())
    nml.write_text(f"&RUN {body}, PTSPHY={PTSPHY!r}, INFILE='{inf}', OUTFILE='{outf}' /\n"
                   "&NAMSCEN /\n&NAMVDF /\n&NAMGWD /\n")
    r = subprocess.run([str(build / "oracle.x"), str(nml)], capture_output=True, text=True)
    if r.returncode != 0 or "DRIVER_DONE" not in r.stdout:
        raise SystemExit(f"oracle failed {tag}: {r.stdout[-400:]} {r.stderr[-400:]}")
    res = parse(outf)
    if list(res) != list(x["names"]):
        raise SystemExit(f"column order/names differ {tag}")
    return res, r.stdout


def main(cols, build, outp):
    fx = {"ptsphy": np.array(PTSPHY)}
    for g in GROUPS:
        x = inputs(np.load(Path(cols) / f"columns_{g}.npz", allow_pickle=True))
        for k, v in x.items():
            fx[f"{g}_{k}"] = v
        for arm in ARMS:
            res, log = run(build, x, arm, f"{arm}_{g}")
            fx[f"{arm}_{g}_log"] = np.array(log)
            for f in SCAL + INTS + ROWS:
                fx[f"{arm}_{g}_{f}"] = np.array([res[nm][f] for nm in x["names"]])
    np.savez_compressed(outp, **fx)
    print("FIXTURE_OK", outp)


if __name__ == "__main__":
    main(*sys.argv[1:4])

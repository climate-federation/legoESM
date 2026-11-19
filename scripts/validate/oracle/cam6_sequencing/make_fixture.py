"""Generate tests/atmosphere/oracle_fixtures/cam6_sequencing.npz by RUNNING the CAM6
physics-sequencing oracle.

Oracle: ``oracle_seq.F90`` -- verbatim extracts of ESCOMP/CAM cam_cesm2_1_rel_60
(a03b84b7) physics_update / physics_ptend_scale / qneg3 arithmetic and the physpkg
tphysbc (deep convection, macmic loop, radiation) / tphysac (gw_tend) control flow,
over two 1800 s physics steps (radiation computed on the first, held on the second).
The unit under test is the ORDER/time-splitting glue, so the schemes are analytic
stubs identical to ``stubs.py``.  Mode flags: rad_post gw_post dlf_clubb qneg_on
(1 = CAM).  Stored: "1100" (CAM order for radiation and GWD: the order legoESM now
runs), "0000" (the old legoESM order: radiation and GWD on the pre-physics state;
control) and "1111" (all four CAM features, recorded).

Columns: five analytic soundings (tropical deep, subtropical Sc, midlatitude, polar
dry, an edge column at q = qmin).  Constants c_pd, L_v, L_f, g = legoESM's, written
into the oracle input.

Self-checks (every run): the 1100 and 0000 runs differ (the order is visible); every
output is finite.

usage (compute node):
    bash build.sh <CAM@cam_cesm2_1_rel_60> <build dir>
    python make_fixture.py <build dir>/oracle_seq <CAM checkout> <out.npz>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants as C

ORACLE, CAM, OUT = sys.argv[1:4]
MODES = ("1100", "0000", "1111")
FIELDS = ("T", "U", "Q", "QC", "QI")
NMAC, ZTODT = 3, 1800.0


def f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


ncol, nlev = 5, 32
ph = 225.0 + (1.0e5 - 225.0) * (np.arange(nlev + 1) / nlev) ** 1.6   # top->bottom
pm = 0.5 * (ph[1:] + ph[:-1])
dp = ph[1:] - ph[:-1]
cols = [  # Tsfc, Ttp, q0, (qc, plo, phi), (qi, plo, phi), u0, du
    (300.0, 195.0, 0.018, (2e-4, 6e4, 9e4), (5e-5, 2e4, 4e4), 5.0, 10.0),
    (290.0, 200.0, 0.010, (4e-4, 9e4, 9.5e4), (0.0, 0, 0), 3.0, 0.0),
    (285.0, 215.0, 0.008, (1e-4, 7e4, 8.5e4), (1e-4, 3e4, 5e4), 10.0, 30.0),
    (250.0, 215.0, 0.0008, (0.0, 0, 0), (2e-5, 6e4, 9e4), 2.0, 0.0),
    (270.0, 210.0, 0.0, (0.0, 0, 0), (0.0, 0, 0), 0.0, 0.0),
]
T = np.zeros((ncol, nlev))
q, qc, qi, u = (np.zeros_like(T) for _ in range(4))
for i, (ts, ttp, q0, c, ice, u0, du) in enumerate(cols):
    T[i] = np.maximum(ts * (pm / 1e5) ** 0.19, ttp)
    q[i] = q0 * (pm / 1e5) ** 3
    if i == 1:
        q[i] = np.where(pm > 9e4, q0, 0.002 * (pm / 1e5) ** 3)
    q[i] = np.maximum(q[i], 1e-12)
    qc[i] = np.where((pm >= c[1]) & (pm <= c[2]), c[0], 0.0)
    qi[i] = np.where((pm >= ice[1]) & (pm <= ice[2]), ice[0], 0.0)
    u[i] = u0 + du * (1.0 - pm / 1e5)
q0ref = 0.9 * q

lines = [f"{ncol} {nlev} {NMAC}"] + [repr(float(v)) for v in (C.c_pd, C.L_v, C.L_f, C.g, ZTODT)]  # latent-ok: CAM6 oracle constant
lines += [f17(pm), f17(dp)]
for i in range(ncol):
    lines += [f17(a[i]) for a in (T, q, qc, qi, u, q0ref)]

out = {}
with tempfile.TemporaryDirectory() as td:
    fin = Path(td) / "seq_in.txt"
    fin.write_text("\n".join(lines) + "\n")
    for m in MODES:
        fout = Path(td) / f"seq_out_{m}.txt"
        r = subprocess.run([ORACLE, str(fin), str(fout), *m], cwd=td, capture_output=True,
                           text=True, check=True)
        assert "ORACLE_SEQ_DONE" in r.stdout, r.stdout + r.stderr
        o = {k: np.zeros((ncol, nlev)) for k in FIELDS}
        prec = np.zeros((2, ncol))
        for line in fout.read_text().splitlines():
            tok = line.split()
            if tok[0] == "PREC":
                prec[int(tok[1]), int(tok[2]) - 1] = float(tok[3])
            else:
                o[tok[0]][int(tok[1]) - 1] = np.array(tok[2:], np.float64)
        o["PREC"] = prec
        for k, v in o.items():
            assert np.all(np.isfinite(v)), (m, k)
            out[f"{k}_{m}"] = v

assert np.max(np.abs(out["T_1100"] - out["T_0000"])) > 0.0       # the order is visible

meta = {
    "oracle": "oracle_seq.F90: CAM cam_cesm2_1_rel_60 physics_update/ptend_scale/qneg3 + "
              "physpkg tphysbc/tphysac order, analytic stub schemes (stubs.py)",
    "repo": "https://github.com/ESCOMP/CAM",
    "sha": subprocess.run(["git", "-C", CAM, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip(),
    "generator": "scripts/validate/oracle/cam6_sequencing/{build.sh,make_fixture.py,oracle_seq.F90,stubs.py}",
    "modes": "rad_post gw_post dlf_clubb qneg_on (1 = CAM)", "nmac": NMAC, "ztodt": ZTODT,
}
np.savez(OUT, meta=json.dumps(meta), T=T, q=q, qc=qc, qi=qi, u=u, q0ref=q0ref, pm=pm, dp=dp,
         cp=C.c_pd, Lv=C.L_v, g=C.g, **out)  # latent-ok: CAM6 oracle constant
print("FIXTURE_OK", OUT, meta["sha"])

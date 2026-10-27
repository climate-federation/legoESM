"""Generate the SBM oracle fixture by RUNNING Isca's qe_moist_convection.

Oracle: ExeClim/Isca tag ``v1.0`` (commit
5868cfd863b41823a3eeb30754ce06feba7528ec),
``src/atmos_param/qe_moist_convection/qe_moist_convection.F90``
(``convection_scheme = 'SIMPLE_BETTS_MILLER'``), built by ``build.sh`` against
FMS stubs carrying legoESM's constants and Tetens curve (``stubs_common.F90``,
sat_mode 1).  Namelist ``tau_bm = 7200, rhbm = 0.7``; ``dt = 1800 s``.

Columns: the 19 columns of the 2026-10-19 tier-2 audit
(``oracle_audit/sbm_dca_adjustment_convection/inputs/columns.npz``: 3 MPAS AMIP
day-1 checkpoint columns + 12 constructed soundings from the tier-1 ZM audit +
4 constructed there), level 0 = model top, q = specific humidity.

Usage (CPU node; gfortran 8.5.0)::

    bash scripts/validate/oracle/sbm_qe/build.sh <Isca v1.0> <build dir>
    python scripts/validate/oracle/sbm_qe/make_fixture.py --build <build dir> \\
        --columns <columns.npz> --isca <Isca v1.0> \\
        --out tests/atmosphere/oracle_fixtures/sbm_qe.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DT = 1800.0
TAU_BM = 7200.0
RHBM = 0.7
SAT_MODE = 1   # stubs_common.F90: legoESM Tetens
ORACLE = dict(repo="ExeClim/Isca", tag="v1.0",
              sha="5868cfd863b41823a3eeb30754ce06feba7528ec",
              file="src/atmos_param/qe_moist_convection/qe_moist_convection.F90")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_input(path, c):
    ncol, nlev = c["T"].shape
    with open(path, "w") as f:
        f.write(f"{ncol} {nlev} {SAT_MODE} {DT!r}\n")
        for i in range(ncol):
            for a in (c["T"][i], c["q"][i], c["p_full"][i], c["p_half"][i]):
                f.write(" ".join(repr(float(x)) for x in a) + "\n")


def read_output(path):
    lines = Path(path).read_text().split("\n")
    cols = []
    i = 1
    while i < len(lines) and lines[i].startswith("COL"):
        h = lines[i].split()
        sc = [float(x) for x in lines[i + 1].split()]
        arr = [np.array([float(x) for x in lines[i + 2 + j].split()]) for j in range(6)]
        cols.append(dict(flag=int(h[2]), klzb=int(h[3]), klzb_diag=int(h[4]),
                         precip=sc[0], cape=sc[1], cin=sc[2], cape_diag=sc[3],
                         dT=arr[0], dq=arr[1], Tref=arr[2], qref=arr[3], Tp=arr[4], rp=arr[5]))
        i += 8
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--isca", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = dict(np.load(a.columns))
    with tempfile.TemporaryDirectory() as d:
        write_input(Path(d) / "in.txt", c)
        (Path(d) / "input.nml").write_text(
            f"&qe_moist_convection_nml\n tau_bm = {TAU_BM!r}, rhbm = {RHBM!r}\n/\n")
        r = subprocess.run([str(Path(a.build) / "drv_sbm.x"), "in.txt", "out.txt"],
                           cwd=d, capture_output=True, text=True, check=True)
        assert "DRV_SBM_OK" in r.stdout, r.stdout + r.stderr
        cols = read_output(Path(d) / "out.txt")
    for col in cols:   # the diagnostic wrapper ran the scheme's own parcel routine
        assert col["cape"] == col["cape_diag"] and col["klzb"] == col["klzb_diag"]
    isca_file = Path(a.isca) / ORACLE["file"]
    head = subprocess.run(["git", "-C", a.isca, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == ORACLE["sha"], head
    gf = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    meta = dict(ORACLE, oracle_file_sha256=_sha256(isca_file), gfortran=gf,
                flags="-O0 -cpp -fdefault-real-8 -fdefault-double-8 -fno-fast-math",
                patch="patch_qe.py (diagnostic wrapper qe_diag_parcel only)",
                constants="legoESM (stubs_common.F90 default)", saturation="legoESM Tetens (sat_mode 1)",
                dt=DT, tau_bm=TAU_BM, rhbm=RHBM, columns_sha256=_sha256(a.columns),
                columns="oracle_audit/sbm_dca_adjustment_convection/inputs/columns.npz")
    out = {k: c[k] for k in ("T", "q", "p_full", "p_half")}
    out["names"] = np.array([str(n) for n in c["names"]])
    for k in ("flag", "klzb", "precip", "cape"):
        out[f"ora_{k}"] = np.array([col[k] for col in cols])
    for k in ("dT", "dq", "Tp", "rp"):
        out[f"ora_{k}"] = np.stack([col[k] for col in cols])
    out["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(a.out, **out)
    print("wrote", a.out, "flags", out["ora_flag"].tolist())


if __name__ == "__main__":
    main()

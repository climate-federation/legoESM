"""Generate tests/atmosphere/oracle_fixtures/holtslag_boville.npz by RUNNING
the Holtslag-Boville oracle (E3SM v3.0.1 hb_diff.F90 + pbl_utils.F90, then
CAM cam_cesm2_1_rel_60 compute_vdiff as vertical_diffusion_tend sets it up
for eddy_scheme='HB').

usage:
  bash build.sh <E3SM> <CAM> <build>
  python make_fixture.py --inputs <columns.npz> --driver <build>/hb_driver \
      --out tests/atmosphere/oracle_fixtures/holtslag_boville.npz

``--inputs`` is the tier-2 audit column set (21 L32 columns; real MPAS
checkpoint columns + constructed soundings, with CAM-FV geopotential heights
and prescribed surface fluxes; built by the audit's make_inputs.py, sha256
recorded in ``meta``).  Constants handed to the oracle's init routines are
legoESM's (``legoesm.constants``), so both sides use identical constants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants as C

IN_FIELDS = ("T", "q", "u", "v", "p_full", "p_half", "zm", "zi")
SFC_FIELDS = ("shflx", "lhflx", "qflx", "taux", "tauy")
ORACLE = {
    "hb_diff": dict(repo="E3SM-Project/E3SM", tag="v3.0.1",
                    sha="d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84",
                    files=["components/eam/src/physics/cam/hb_diff.F90",
                           "components/eam/src/physics/cam/pbl_utils.F90"]),
    "solver": dict(repo="ESCOMP/CAM", tag="cam_cesm2_1_rel_60",
                   sha="a03b84b7c4e34f965b115686f22a043b85739e56",
                   files=["src/physics/cam/diffusion_solver.F90",
                          "src/physics/cam/vdiff_lu_solver.F90",
                          "src/utils/coords_1d.F90",
                          "src/utils/linear_1d_operators.F90"]),
}


def constants_vector(dt):
    return [C.g, C.kappa_vk, C.c_pd, C.R_d, C.R_v / C.R_d - 1.0, dt]


def write_inputs(path, d):
    def wr(f, a):
        f.write(" ".join(repr(float(x)) for x in np.ravel(a)) + "\n")
    ncol, nlev = d["T"].shape
    with open(path, "w") as f:
        f.write(f"{ncol} {nlev}\n")
        wr(f, constants_vector(float(d["dt"])))
        wr(f, d["pref_mid"])
        for c in range(ncol):
            for k in IN_FIELDS + ("th",):
                wr(f, d[k][c])
            wr(f, [d["shflx"][c], d["qflx"][c], d["taux"][c], d["tauy"][c]])


def parse_outputs(path, ncol):
    lines = open(path).read().split("\n")
    keys1 = ("kvm", "kvh", "kvq", "cgh", "cgs", "tke", "ri",
             "u_out", "v_out", "s_out", "q_out", "dtk")
    o = {k: [] for k in ("ustar", "obklen", "pblh", "tpert", "qpert") + keys1}
    i = 0
    for c in range(ncol):
        assert lines[i].split() == ["COL", str(c + 1)], lines[i]
        i += 1
        s = np.array(lines[i].split(), float); i += 1
        for k, val in zip(("ustar", "obklen", "pblh", "tpert", "qpert"), s):
            o[k].append(val)
        for k in keys1:
            o[k].append(np.array(lines[i].split(), float)); i += 1
    return {f"ora_{k}": np.array(v) for k, v in o.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = np.load(a.inputs)
    d = {k: np.asarray(src[k], float) for k in IN_FIELDS + SFC_FIELDS + ("pref_mid",)}
    d["dt"] = float(src["dt"])
    # oracle input th = T * (p0/pmid)^cappa  (vertical_diffusion_tend th = t*exner)
    d["th"] = d["T"] / (d["p_full"] / C.p_ref) ** C.kappa
    ncol = d["T"].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        write_inputs(fin, d)
        r = subprocess.run([a.driver, str(fin), str(fout)], capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0 and "ORACLE_OK" in r.stdout, r
        ora = parse_outputs(fout, ncol)
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran,
                constants=dict(zip(("g", "vk", "cpair", "rair", "zvir", "dt"),
                                   constants_vector(d["dt"]))),
                inputs_sha256=hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
                names=[str(n) for n in src["names"]],
                setup="eddy_scheme=HB, do_iss=F, no TMS/Beljaars/molec, qmincg=1e-12, phis=0")
    np.savez_compressed(a.out, meta=json.dumps(meta), dt=d["dt"],
                        **{f"in_{k}": v for k, v in d.items() if k != "dt"}, **ora)
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

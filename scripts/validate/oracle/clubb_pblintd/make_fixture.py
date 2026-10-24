"""Generate tests/atmosphere/oracle_fixtures/clubb_pblintd_cam.npz by RUNNING
the CAM6 post-CLUBB PBL-height block (clubb_intr.F90:2594-2617: calc_ustar,
calc_obklen, hb_diff::pblintd; CAM cam_cesm2_1_rel_60,
a03b84b7c4e34f965b115686f22a043b85739e56).

usage:
  bash build.sh <CAM checkout> <build>
  python make_fixture.py --driver <build>/pblintd_driver \
      --out tests/atmosphere/oracle_fixtures/clubb_pblintd_cam.npz

Input columns are the committed Holtslag-Boville oracle fixture's 21 L32
columns (``tests/atmosphere/oracle_fixtures/holtslag_boville.npz``: state,
heights, reference pressures and prescribed surface stress / sensible heat /
water flux; sha256 recorded in ``meta``).  ``exner`` handed to the oracle is
``1/exner_function(p_full)``, the mapping the port uses.  Constants handed to
the oracle's init routines are legoESM's (``legoesm.constants``).
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

SRC = (Path(__file__).resolve().parents[4] / "tests" / "atmosphere"
       / "oracle_fixtures" / "holtslag_boville.npz")
IN_FIELDS = ("T", "q", "u", "v", "p_full", "zm", "zi", "exner")
ORACLE = dict(repo="ESCOMP/CAM", tag="cam_cesm2_1_rel_60",
              sha="a03b84b7c4e34f965b115686f22a043b85739e56",
              files=["src/physics/cam/clubb_intr.F90 (2594-2617, copied into "
                     "pblintd_driver.F90)", "src/physics/cam/hb_diff.F90",
                     "src/physics/cam/pbl_utils.F90"])
OUT_KEYS = ("ustar", "obklen", "kbfs", "pblh", "wstar")


def constants_vector():
    return [C.g, C.kappa_vk, C.c_pd, C.R_d, C.R_v / C.R_d - 1.0]


def build_inputs(src):
    d = {k: np.asarray(src[f"in_{k}"], float)
         for k in ("T", "q", "u", "v", "p_full", "p_half", "zm", "zi",
                   "taux", "tauy", "shflx", "qflx", "pref_mid")}
    d["exner"] = 1.0 / (d["p_full"] / C.p_ref) ** C.kappa
    return d


def write_inputs(path, d):
    def wr(f, a):
        f.write(" ".join(repr(float(x)) for x in np.ravel(a)) + "\n")
    ncol, nlev = d["T"].shape
    with open(path, "w") as f:
        f.write(f"{ncol} {nlev}\n")
        wr(f, constants_vector())
        wr(f, d["pref_mid"])
        for c in range(ncol):
            for k in IN_FIELDS:
                wr(f, d[k][c])
            wr(f, [d["taux"][c], d["tauy"][c], d["shflx"][c], d["qflx"][c]])


def parse_outputs(path, ncol):
    rows = [np.array(ln.split(), float)
            for ln in open(path).read().split("\n") if ln.strip()]
    assert len(rows) == ncol, (len(rows), ncol)
    a = np.array(rows)
    return {f"ora_{k}": a[:, j] for j, k in enumerate(OUT_KEYS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = np.load(SRC)
    d = build_inputs(src)
    ncol = d["T"].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        write_inputs(fin, d)
        r = subprocess.run([a.driver, str(fin), str(fout)],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0 and "ORACLE_OK" in r.stdout, r
        ora = parse_outputs(fout, ncol)
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran,
                constants=dict(zip(("g", "vk", "cpair", "rair", "zvir"),
                                   constants_vector())),
                inputs=str(SRC.relative_to(SRC.parents[3])),
                inputs_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
                names=json.loads(str(src["meta"]))["names"])
    np.savez_compressed(a.out, meta=json.dumps(meta),
                        **{f"in_{k}": v for k, v in d.items()}, **ora)
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

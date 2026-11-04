"""Generate tests/atmosphere/oracle_fixtures/zm_detrain_cam.npz by RUNNING the
CAM6 detrainment of ZM condensate (clubb_intr.F90:2345-2386, CAM
cam_cesm2_1_rel_60, a03b84b7c4e34f965b115686f22a043b85739e56).

usage:
  bash build.sh <CAM checkout> <build>
  python make_fixture.py --driver <build>/detrain_driver \
      --out tests/atmosphere/oracle_fixtures/zm_detrain_cam.npz

Input columns are the committed ZM oracle fixture's 15 L32 columns
(``tests/atmosphere/oracle_fixtures/zm_pblh_cam.npz``): temperature ``T``,
the oracle's own detrained condensate ``oracle_dlf`` and the layer thickness
``diff(p_half)``; sha256 recorded in ``meta``.  latice and gravit handed to
the oracle are legoESM's (``legoesm.constants``).
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
       / "oracle_fixtures" / "zm_pblh_cam.npz")
ORACLE = dict(repo="ESCOMP/CAM", tag="cam_cesm2_1_rel_60",
              sha="a03b84b7c4e34f965b115686f22a043b85739e56",
              files=["src/physics/cam/clubb_intr.F90 (2345-2386, #included "
                     "verbatim by driver.F90)"])
OUT_KEYS = ("dq_c", "dq_i", "dN_c", "dN_i", "s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = np.load(SRC)
    T = np.asarray(src["T"], float)
    dlf = np.asarray(src["oracle_dlf"], float)
    pdel = np.diff(np.asarray(src["p_half"], float), axis=1)
    ncol, nlev = T.shape

    def wr(f, x):
        f.write(" ".join(repr(float(v)) for v in np.ravel(x)) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        with open(fin, "w") as f:
            f.write(f"{ncol} {nlev}\n")
            wr(f, [C.L_f, C.g])
            for c in range(ncol):
                wr(f, T[c]); wr(f, dlf[c]); wr(f, pdel[c])
        r = subprocess.run([a.driver, str(fin), str(fout)],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0 and "ORACLE_OK" in r.stdout, r
        rows = [np.array(ln.split(), float)
                for ln in fout.read_text().split("\n") if ln.strip()]
    assert len(rows) == 6 * ncol, (len(rows), ncol)
    ora = {f"ora_{k}": np.array([rows[6 * c + j] for c in range(ncol)])
           for j, k in enumerate(OUT_KEYS)}
    det = np.array([rows[6 * c + 5] for c in range(ncol)])
    ora["ora_det_s"], ora["ora_det_ice"] = det[:, 0], det[:, 1]
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran,
                constants=dict(latice=C.L_f, gravit=C.g, cpair=C.c_pd),
                inputs=str(SRC.relative_to(SRC.parents[3])),
                inputs_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest())
    np.savez_compressed(a.out, meta=json.dumps(meta), names=src["names"],
                        in_T=T, in_dlf=dlf, in_pdel=pdel, **ora)
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

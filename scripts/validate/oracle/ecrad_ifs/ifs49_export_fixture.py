"""Pack one validated ifs49_driver.x run into the IFS-interface oracle fixture.

Reads the text dumps of a run directory (ECRAD49_DUMP format, dumpio.py), the
driver inputs (ifs49_make_inputs.py format) and the harness namelist, and
writes ONE compressed npz (vote IF7-ifs_interface):

  dump:<name>   every dump array, Fortran shape (column first)
  in:<name>     every driver input array, Fortran shape
  nml:<NAME>    the &HARNESS scalars the port needs (NINDAT_H, NSTEP_H,
                TSTEP_H, NSTEPS, DMU0, DTS)

No value is derived here.  NaN anywhere is fatal; the chk_* control dumps
(deliberately NaN: members the 49r1 path never sets) are not exported.

usage: python -I ifs49_export_fixture.py <run dir>/base_a <run dir>/inputs <base_a.nam> <out.npz>
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dumpio import read_arrays  # noqa: E402

_NML_KEYS = ("NINDAT_H", "NSTEP_H", "TSTEP_H", "NSTEPS", "DMU0", "DTS")


def _read_input(path):
    with open(path) as f:
        h = f.readline().split()
        vals = np.array([float(v) for v in f.read().split()], dtype=np.float64)
    shape = tuple(int(n) for n in h[2:2 + int(h[1])])
    return vals.reshape(shape, order="F") if shape else vals[0]


def _read_nml(path):
    text = open(path).read()
    out = {}
    for k in _NML_KEYS:
        m = re.search(rf"^\s*{k}\s*=\s*([0-9.eEdD+-]+)", text, re.M)
        if m is None:
            sys.exit(f"{k} missing from {path}")
        out[k] = float(m.group(1).replace("d", "e").replace("D", "e"))
    return out


def main(dump_dir, input_dir, nml, out):
    z = {}
    for p in sorted(glob.glob(os.path.join(dump_dir, "*.txt"))):
        if os.path.basename(p).startswith("chk_"):
            continue
        for name, (a, _lb) in read_arrays(p).items():
            a = np.asarray(a)
            if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
                sys.exit(f"non-finite value in dump {name}")
            z["dump:" + name] = a
    for p in sorted(glob.glob(os.path.join(input_dir, "*.txt"))):
        name = os.path.basename(p)[:-4]
        if name == "columns":
            z["in:columns"] = np.array([int(v) for v in open(p).read().split()], dtype=np.int64)
            continue
        z["in:" + name] = _read_input(p)
        if not np.all(np.isfinite(z["in:" + name])):
            sys.exit(f"non-finite value in input {name}")
    for k, v in _read_nml(nml).items():
        if not np.isfinite(v):
            sys.exit(f"non-finite namelist value {k}")
        z["nml:" + k] = np.float64(v)
    np.savez_compressed(out, **z)
    print(f"EXPORT_OK {len(z)} arrays -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])

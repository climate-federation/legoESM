"""Export every dataset of a CLOUDSC-dwarf HDF5 file to one text file per dataset.

Format (read by textio_hdf5_file_mod.F90): line 1 = rank and the Fortran-order
extents; then one value per line, Fortran (column-major) order, reals as %.17e
(round-trips fp64 exactly), integers/logicals as integers.
h5py reports the C-order shape, which is the Fortran shape reversed, so the
C-order flattening of the h5py array IS the Fortran-order sequence.
usage: export_h5.py <file.h5> <out_dir>
"""
import sys
from pathlib import Path

import h5py
import numpy as np


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f:
        for name in f:
            a = np.asarray(f[name])
            fshape = a.shape[::-1]
            if a.dtype.kind == "f":
                vals = "\n".join(f"{v:.17e}" for v in a.ravel())
            elif a.dtype.kind in "iub":
                vals = "\n".join(str(int(v)) for v in a.ravel())
            else:
                raise SystemExit(f"unsupported dtype {a.dtype} for {name}")
            head = " ".join(str(n) for n in (len(fshape),) + fshape)
            (out / f"{name}.txt").write_text(head + "\n" + vals + "\n")
            back = np.array([float(x) for x in vals.split()], np.float64)
            if a.dtype.kind == "f" and back.tobytes() != a.ravel().astype(np.float64).tobytes():
                raise SystemExit(f"text round-trip not exact for {name}")
    print("EXPORT_OK", len(list(out.glob("*.txt"))), "datasets")


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Write a copy of an ecRad input file with one variable changed.

usage: python -I make_variant.py IN.nc OUT.nc VAR (nan|x2)
  nan: every element of VAR set to NaN (proves VAR is unused when the output
       is bitwise unchanged); x2: VAR doubled (proves VAR is used when the
       output changes).
"""
import shutil
import sys

import netCDF4
import numpy as np


def main(src, dst, var, how):
    shutil.copyfile(src, dst)
    with netCDF4.Dataset(dst, "a") as f:
        f.set_auto_mask(False)
        x = np.asarray(f[var][:], dtype=np.float64)
        f[var][:] = np.full_like(x, np.nan) if how == "nan" else 2.0 * x
        if how not in ("nan", "x2"):
            raise SystemExit(f"unknown change {how}")


if __name__ == "__main__":
    main(*sys.argv[1:5])

"""Read the text arrays written by the 49r1 ecRad oracle harness.

Format (tables49.F90 and ECRAD49_DUMP): one header line
"name kind rank lb1 ub1 ... lbR ubR" (kind r or i), then the values one per
line in Fortran (column-major) order. read_arrays() returns
{name: (array, lower_bounds)}; arrays are float64/int64 in Fortran order with
the Fortran shape, and lower_bounds keeps the Fortran index base of each dim.
"""
import numpy as np


def read_arrays(path):
    out = {}
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        h = lines[i].split()
        name, kind, rank = h[0], h[1], int(h[2])
        b = [int(x) for x in h[3:3 + 2 * rank]]
        lb, ub = b[0::2], b[1::2]
        shape = tuple(u - l + 1 for l, u in zip(lb, ub))
        n = int(np.prod(shape)) if shape else 1
        vals = lines[i + 1:i + 1 + n]
        if len(vals) != n:
            raise ValueError(f"{path}: {name} has {len(vals)} values, header says {n}")
        conv = float if kind == "r" else int
        a = np.array([conv(v.replace("D", "E")) for v in vals],
                     dtype=np.float64 if kind == "r" else np.int64)
        out[name] = (a.reshape(shape, order="F") if shape else a[0], tuple(lb))
        i += 1 + n
    return out

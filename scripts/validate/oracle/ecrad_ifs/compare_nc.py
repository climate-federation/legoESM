"""Element-wise comparison of two ecRad output files.

usage: python -I compare_nc.py A.nc B.nc
Prints, per shared variable, the max |a-b| and the max element relative
difference |a-b|/max(|a|,|b|) (pairs that are both zero count as 0). NaN or Inf in
either file is FATAL. Variables present in only one file are listed.
"""
import sys

import netCDF4
import numpy as np


def compare(path_a, path_b):
    a, b = netCDF4.Dataset(path_a), netCDF4.Dataset(path_b)
    a.set_auto_mask(False)
    b.set_auto_mask(False)
    only = sorted(set(a.variables) ^ set(b.variables))
    worst = 0.0
    rows = []
    for v in sorted(set(a.variables) & set(b.variables)):
        for f, p in ((a, path_a), (b, path_b)):
            if f[v].dtype.kind == "f" and f[v].dtype.itemsize != 8:
                raise SystemExit(f"{v} in {p} is {f[v].dtype}, not float64")
            if any(k in f[v].ncattrs() for k in ("scale_factor", "add_offset")):
                raise SystemExit(f"{v} in {p} is packed")
        x = np.asarray(a[v][:], dtype=np.float64)
        y = np.asarray(b[v][:], dtype=np.float64)
        if x.shape != y.shape:
            raise SystemExit(f"shape mismatch {v}: {x.shape} vs {y.shape}")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise SystemExit(f"non-finite value in {v}")
        d = np.abs(x - y)
        den = np.maximum(np.abs(x), np.abs(y))
        rel = np.divide(d, den, out=np.zeros_like(d), where=den > 0)
        r = float(rel.max()) if rel.size else 0.0
        worst = max(worst, r)
        rows.append((v, float(d.max()) if d.size else 0.0, r))
    return rows, only, worst


if __name__ == "__main__":
    rows, only, worst = compare(sys.argv[1], sys.argv[2])
    for v, dmax, r in rows:
        print(f"{v:42s} maxabs={dmax:.3e} max_elem_rel={r:.3e}")
    if only:
        print("only in one file:", " ".join(only))
    print(f"WORST_ELEM_REL {worst:.3e}")

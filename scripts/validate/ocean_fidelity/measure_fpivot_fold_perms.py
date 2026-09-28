#!/usr/bin/env python
"""Measure the F-pivot north-fold index maps of an ORCA mesh IN THIS MODEL'S
STAGGERING, from coordinate coincidences (no convention assumed).

Layout measured: the halo-inclusive mesh (e.g. eORCA1.2, 332x362) whose stored
top row is the duplicated fold-halo row.  After stripping that row (the
``--tripole-strip-north-rows 1`` layout) the stored top T row is NEMO's
``jpj-1`` and the fold line is the NORTH face of that row.  Our conventions:

* T (j, i)      = NEMO T(i, j)
* U face (j, k) = NEMO u(k-1, j)   (our face k lies WEST of cell k; face 0 wraps)
* v (j+1, i)    = NEMO v(i, j)     (our v has a prepended south row)
* q (j+1, k)    = NEMO F(k-1, j)

With the halo row stripped (n_lat = N-1, N = mesh rows), the maps are read as

* T ghost row above the top row      : T_mesh[N-1]   == T_mesh[N-2][P_T]
* U ghost row above the top row      : U_ours[N-1]   == U_ours[N-2][P_U]
* v fold line (our v[n_lat], self)   : V_mesh[N-2]   == V_mesh[N-2][P_V]
* v ghost row (our v[n_lat+1])       : V_mesh[N-1]   == V_mesh[N-3][P_V]
* q fold line (our q[n_lat], self)   : F_ours[N-2]   == F_ours[N-2][P_F]
* q ghost row (our q[n_lat+1])       : F_ours[N-1]   == F_ours[N-3][P_F]

A map "fits" when every compared pair is the SAME physical point (lon mod 360
and lat to 1e-6 deg).  The two cyclic E-W halo columns are excluded (their
source NEMO column is 0 or n-1) as are self-paired points on self rows.
At most ONE mismatching column is tolerated, and it is printed.
Prints one line per (relation, candidate) with the max coordinate mismatch.
"""
from __future__ import annotations

import sys

import netCDF4 as nc
import numpy as np


def _wrap_dlon(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def _west_face(a):
    """NEMO (.., n) east-of-cell array -> our west-face (.., n) layout."""
    return np.concatenate([a[:, -1:], a[:, :-1]], axis=1)


CANDIDATES = {
    "n-1-i": lambda n: np.arange(n - 1, -1, -1),
    "(n-i)%n": lambda n: (n - np.arange(n)) % n,
    "(n-2-i)%n": lambda n: (n - 2 - np.arange(n)) % n,
    "(n+1-i)%n": lambda n: (n + 1 - np.arange(n)) % n,
}


def measure(mesh: str, tol: float = 1e-6) -> dict:
    ds = nc.Dataset(mesh)

    def rd(name):
        a = np.asarray(ds[name][:], dtype=float)
        return a[0] if a.ndim == 3 else a

    pts = {
        "T": (rd("glamt"), rd("gphit"), 0),
        "U": (_west_face(rd("glamu")), _west_face(rd("gphiu")), 1),
        "V": (rd("glamv"), rd("gphiv"), 0),
        "F": (_west_face(rd("glamf")), _west_face(rd("gphif")), 1),
    }
    N, n = pts["T"][0].shape
    rel = {  # name -> (point, dst row, src row, self_row)
        "T ghost": ("T", N - 1, N - 2, False),
        "U ghost": ("U", N - 1, N - 2, False),
        "V fold self": ("V", N - 2, N - 2, True),
        "V ghost": ("V", N - 1, N - 3, False),
        "F fold self": ("F", N - 2, N - 2, True),
        "F ghost": ("F", N - 1, N - 3, False),
    }
    idx = np.arange(n)
    out = {}
    print(f"mesh={mesh} shape=({N},{n})  (tol {tol:g} deg)")
    for name, (pt, jd, js, self_row) in rel.items():
        LO, LA, shift = pts[pt]
        # NEMO source column of our column c is c - shift; drop cyclic halo.
        src = idx - shift
        interior = (src >= 1) & (src <= n - 2)
        fits = []
        for cname, cfn in CANDIDATES.items():
            p = cfn(n)
            keep = interior & interior[p]
            if self_row:
                keep &= p != idx
            d = np.maximum(_wrap_dlon(LO[jd], LO[js][p]),
                           np.abs(LA[jd] - LA[js][p]))
            bad = np.flatnonzero(keep & (d >= tol))
            # One stored-coordinate outlier is tolerated and PRINTED: the
            # eORCA1.2 halo row holds glam = 73.0 exactly at column 360 (a
            # rounded placeholder; its gphi matches), for T and V alike.
            ok = bad.size <= 1
            if ok:
                fits.append(cname)
            extra = (f" outlier col {bad[0]}: lon {LO[jd][bad[0]]} vs "
                     f"{LO[js][p][bad[0]]}" if bad.size == 1 else "")
            print(f"  {name:12s} {cname:10s} max|d|={d[keep].max():.2e} "
                  f"n_bad={bad.size}{'  **FIT' if ok else ''}{extra}")
        out[name] = fits
    return out


def main() -> int:
    mesh = sys.argv[1] if len(sys.argv) > 1 else "data/grids/eORCA1.2_mesh_mask.nc"
    res = measure(mesh)
    print("RESULT:", res)
    return 0 if all(len(v) == 1 for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

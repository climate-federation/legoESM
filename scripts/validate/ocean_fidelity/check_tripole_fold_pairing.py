#!/usr/bin/env python
"""Does the detected tripole fold pair the RIGHT cells on this mesh?

WHY (2026-08-26): the eORCA025 8-GPU full card goes NaN by step 23 while the
same card is clean at 1 degree (CPU 8-band single-process AND 2-process GPU,
bit-identical), so the remaining suspects are 1/4-degree-mesh-specific.  NEMO
ORCA1 folds at a T-point pivot; ORCA025 in NEMO is an F-POINT pivot mesh
(jperio=6).  Our loader models both meshes as T-folds with two wrap-origin
candidates, auto-detected by fold-row LATITUDE symmetry — a test an F-pivot
mesh can pass approximately while the actual partner pairing is off by half a
cell.  The sharp test needs no model run: under the CORRECT pairing, a fold
row cell and its partner are the SAME physical point, so their longitudes
must agree (mod 360) and their latitudes match tightly.  A half-cell pivot
error shows up as a systematic ~dlon/2 longitude offset.

Prints, for each candidate permutation: fold-row max |dlat|, and the median /
p95 |dlon| between paired cells (wrap-aware), plus the same for row j_max-1
against the fold partner ROW pairing the operators use.
"""
from __future__ import annotations

import sys

import netCDF4 as nc
import numpy as np


def _wrap_dlon(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def main() -> int:
    mesh = sys.argv[1] if len(sys.argv) > 1 else "data/grids/eORCA025_mesh_mask.nc"
    ds = nc.Dataset(mesh)

    def rd(name):
        v = ds[name][:]
        a = np.asarray(v)
        return a[0] if a.ndim == 3 else a

    glamt, gphit = rd("glamt"), rd("gphit")
    tmask = np.asarray(ds["tmaskutil"][:]).squeeze() if "tmaskutil" in ds.variables else None
    n_lat, n_lon = gphit.shape
    print(f"mesh={mesh}  shape=({n_lat},{n_lon})")
    j = n_lat - 1
    lat_f, lon_f = gphit[j], glamt[j]

    perms = {
        "n_lon-1-i": np.arange(n_lon - 1, -1, -1),
        "(n_lon-i)%n_lon": (n_lon - np.arange(n_lon)) % n_lon,
    }
    for name, p in perms.items():
        dlat = np.abs(lat_f - lat_f[p])
        dlon = _wrap_dlon(lon_f, lon_f[p])
        # exclude self-paired columns (fixed points of the permutation)
        m = p != np.arange(n_lon)
        print(f"[fold row j={j}] perm {name:16s} max|dlat|={dlat[m].max():.4f} "
              f"median|dlon|={np.median(dlon[m]):.4f}  p95|dlon|="
              f"{np.percentile(dlon[m], 95):.4f} deg")
        # The operators' fold reads row j-1 as the partner of row j (T-fold:
        # row j duplicates row j-1 mirrored).  Same-point test across rows:
        dlat2 = np.abs(gphit[j - 1] - gphit[j][p])
        dlon2 = _wrap_dlon(glamt[j - 1], glamt[j][p])
        print(f"  [row {j - 1} vs folded row {j}] max|dlat|={dlat2[m].max():.4f} "
              f"median|dlon|={np.median(dlon2[m]):.4f} p95|dlon|="
              f"{np.percentile(dlon2[m], 95):.4f}")
    # Grid spacing for scale: half-cell error at the fold ~ dlon_cell/2.
    dlon_cell = np.median(_wrap_dlon(glamt[j, 1:], glamt[j, :-1]))
    print(f"[scale] median fold-row cell dlon = {dlon_cell:.4f} deg "
          f"(half-cell = {dlon_cell / 2:.4f})")
    if tmask is not None:
        print(f"[mask] fold-row wet cells: {int((tmask[j] > 0.5).sum())}/{n_lon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

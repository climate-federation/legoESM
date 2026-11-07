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

    # ROW-PAIR x PERM SCAN over all four point types (T/U/V/F): report every
    # EXACT physical-coincidence mapping, so the fold implementation is read
    # off the MESH rather than assumed from a convention name.  NEMO T-pivot
    # (lbc_nfd 'T' case, ihls=1): halo row ipj <- row ipj-2 permuted; pivot
    # row ipj-1 is SELF-symmetric (right half mirrors left).  A de-haloed
    # mesh stores the pivot row as its top row.
    pts = {"T": ("glamt", "gphit"), "U": ("glamu", "gphiu"),
           "V": ("glamv", "gphiv"), "F": ("glamf", "gphif")}
    offs = {"(n_lon-i)%n_lon": lambda n: (n - np.arange(n)) % n,
            "n_lon-1-i": lambda n: np.arange(n - 1, -1, -1),
            "(n_lon-i-2)%n_lon": lambda n: (n - np.arange(n) - 2) % n,
            "n_lon-2-i": lambda n: (n - 2 - np.arange(n)) % n}
    print("\n# exact-coincidence scan (max|dlat|, p50|dlon| in deg; '**' = exact <1e-6)")
    for pt, (ln, lt) in pts.items():
        if ln not in ds.variables:
            continue
        LO, LA = rd(ln), rd(lt)
        for (ja, jb) in ((j, j), (j, j - 1), (j - 1, j - 1), (j - 1, j - 2)):
            for oname, ofn in offs.items():
                pp = ofn(n_lon)
                m = pp != np.arange(n_lon) if ja == jb else np.ones(n_lon, bool)
                if not m.any():
                    continue
                dla = np.abs(LA[ja] - LA[jb][pp])[m].max()
                dlo = np.median(_wrap_dlon(LO[ja], LO[jb][pp])[m])
                tag = " **" if (dla < 1e-6 and dlo < 1e-6) else ""
                if dla < 0.02 and dlo < 0.02 or tag:
                    print(f"  {pt}: row{ja}<-row{jb} perm {oname:18s} "
                          f"max|dlat|={dla:.2e} p50|dlon|={dlo:.2e}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

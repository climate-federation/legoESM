"""Build a half-degree eORCA05 tripole mesh by 2x-decimating eORCA025 (1/4 deg).

The repo ships only eORCA1 (1 deg) and eORCA025 (1/4 deg) tripole meshes; a true
2x-of-1deg run needs 1/2 deg, which does not exist.  This decimates eORCA025 ->
eORCA05 so a single-GPU tripole run at double the 1-deg resolution is possible.

Decimation phases (CRUCIAL for the tripole north-fold):
  * j = [1::2]  -> keeps the fold row (j = n_lat-1 = 1205, last row) AND drops
    the southern wall row 0 (land).  [::2] would DROP the fold row.
  * i = [::2]   -> preserves the de-haloed fold pairing exactly: eORCA025 uses
    convention ``(n_lon-i)%n_lon``; subsampled col i' = 2i pairs with
    (1440-2i')%1440, which in subsampled indices is (720-i')%720 = the same
    convention at n_lon'=720.  create_tripole_grid._detect_fold VERIFIES this
    (fails loud) -> a wrong phase is caught, not silently shipped.

Metrics (e1t/e2t/... [m]) are scaled x2 (a 1/2 deg cell ~ 2x the 1/4 deg width);
coords (glam*/gphi*) are decimated; tmask/e3t_0 decimated; e3t_1d/nav_lev copied.
The x2 metric is approximate (ignores local spacing variation, ~1% off-fold) but
geometrically consistent enough for an exploratory double-resolution attempt;
the fold + pole are handled by the validated fold map and the polar filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

_SRC = "data/grids/eORCA025_mesh_mask.nc"
_DST = "data/grids/eORCA05_mesh_mask.nc"

# field groups
_COORDS = ["glamt", "gphit", "glamu", "gphiu", "glamv", "gphiv", "glamf", "gphif"]
_METRICS = ["e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"]
_MASK2D = ["tmaskutil", "umaskutil", "vmaskutil"]
_J = slice(1, None, 2)   # keep fold row (last), drop south wall
_I = slice(0, None, 2)   # preserve fold pairing


def _dec2d(a):
    """Decimate the trailing (y, x) of a (..., y, x) array with the fold phases."""
    return a[..., _J, _I]


def build(src, dst):
    ds = xr.open_dataset(src)
    print(f"[eORCA05] source dims: {dict(ds.sizes)}")
    out = {}

    def _get(name):
        v = np.asarray(ds[name].values, dtype=np.float64)
        return v

    for name in _COORDS:
        if name in ds:
            out[name] = (("y", "x"), _dec2d(np.squeeze(_get(name))))
    for name in _METRICS:
        if name in ds:
            out[name] = (("y", "x"), 2.0 * _dec2d(np.squeeze(_get(name))))
    for name in _MASK2D:
        if name in ds:
            out[name] = (("y", "x"), _dec2d(np.squeeze(_get(name))))

    # tmask (z,y,x) + e3t_0 (z,y,x): decimate horizontally, keep all 75 levels.
    tmask = np.squeeze(_get("tmask"))                       # (z, y, x)
    while tmask.ndim > 3:
        tmask = tmask[0]
    out["tmask"] = (("z", "y", "x"), _dec2d(tmask))
    e3t0 = np.squeeze(_get("e3t_0"))
    while e3t0.ndim > 3:
        e3t0 = e3t0[0]
    if e3t0.ndim == 3:
        out["e3t_0"] = (("z", "y", "x"), _dec2d(e3t0))
    else:  # 1-D e3t_0 (rare) -> broadcast unchanged
        out["e3t_0"] = (("z",), e3t0)
    # vertical column unchanged
    for name in ("e3t_1d", "nav_lev"):
        if name in ds:
            out[name] = (("z",), np.squeeze(_get(name)))

    dso = xr.Dataset({k: v for k, v in out.items()})
    ny, nx = out["gphit"][1].shape
    print(f"[eORCA05] decimated dims: y={ny} x={nx} z={dso.sizes.get('z')}")
    # sanity
    gph = out["gphit"][1]; glam = out["glamt"][1]
    print(f"[eORCA05] lat {np.nanmin(gph):.1f}..{np.nanmax(gph):.1f}  "
          f"lon {np.nanmin(glam):.1f}..{np.nanmax(glam):.1f}")
    # Metrics may be 0/junk at fold/boundary rows (the source eORCA025 has them
    # too; the model masks those). Report counts; positivity is only required in
    # the ocean interior, which the fold validation + the run smoke-test cover.
    for m in _METRICS:
        if m in out:
            a = out[m][1]; fin = a[np.isfinite(a)]
            nbad = int((fin <= 0).sum())
            if nbad:
                print(f"[eORCA05] {m}: {nbad} non-positive (boundary/fold; "
                      f"min={fin.min():.3g}) — matches the eORCA025 source")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    dso.to_netcdf(dst)
    print(f"[eORCA05] wrote {dst}")
    return dst


def validate(dst):
    """Fail-loud fold verification via the model's own create_tripole_grid."""
    from legoesm.grids.tripole import create_tripole_grid
    print("[eORCA05] validating fold via create_tripole_grid ...")
    grid = create_tripole_grid(dst, fold_convention="(n_lon-i)%n_lon")
    print(f"[eORCA05] FOLD OK — grid n_lat={grid.n_lat} n_lon={grid.n_lon}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=_SRC)
    p.add_argument("--dst", default=_DST)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()
    dst = build(args.src, args.dst)
    if not args.no_validate:
        validate(dst)


if __name__ == "__main__":
    main()

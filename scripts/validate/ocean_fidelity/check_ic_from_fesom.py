"""Acceptance check for ``init_ocean_from_fesom_mesh`` on a real target geometry.

Builds T/S for the horizontal geometry stored in a ``run_omip`` snapshot
(``lat_T``, ``lon_T``, ``land_mask``, ``H_bathy``, ``z_center_ref``) from a FESOM2-JAX
mesh's cached initial field, writes the result in the same snapshot layout (so
``probe_ic_density_walls.py --orca-snapshot`` ranks its adjacent-cell density
jumps), and prints: the node-area-weighted FESOM surface means against the
wet-area-weighted target surface means, and the count and worst value of
static density inversions (in-situ density at the deeper level's pressure,
level k+1 lighter than level k by more than ``--inversion-tol`` kg/m3).

Sampling depths are the REFERENCE level centres, not the run's partial-cell
centroids: the run itself resamples the bottom cell at its true centroid, so
this check bounds the horizontal stitching, not the bottom-cell placement.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from legoesm import constants
from legoesm.ocean.eos import wright_eos
from legoesm.ocean.init_woa import init_ocean_from_fesom_mesh


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry-snapshot", required=True, help="run_omip snapshot_final.npz giving the target geometry")
    p.add_argument("--fesom-mesh", required=True)
    p.add_argument("--out", required=True, help="npz in the snapshot layout, for probe_ic_density_walls.py")
    p.add_argument("--inversion-tol", type=float, default=0.1)
    a = p.parse_args()

    s = np.load(a.geometry_snapshot, allow_pickle=False)
    lat, lon, wet, H, zc = s["lat_T"], s["lon_T"], s["land_mask"] > 0.5, s["H_bathy"], s["z_center_ref"]
    grid = SimpleNamespace(lat2d=np.deg2rad(lat), lon2d=np.deg2rad(lon))
    z_coord = SimpleNamespace(n_levels=int(zc.size), z_full_ref=-np.abs(zc))
    T, S = init_ocean_from_fesom_mesh(grid, z_coord, a.fesom_mesh, wet_mask=wet)
    T, S = np.asarray(T), np.asarray(S)

    # surface means: FESOM node-area-weighted vs target wet-area-weighted, the
    # target cell area from the great-circle spacing between T-points (the
    # snapshot carries no e1t/e2t; this is the probe's own spacing estimate)
    m = a.fesom_mesh
    area = np.load(os.path.join(m, "area.npy"), mmap_mode="r")[:, 0]
    Tn = np.load(os.path.join(m, "T_ic.npy"), mmap_mode="r")[:, 0]
    Sn = np.load(os.path.join(m, "S_ic.npy"), mmap_mode="r")[:, 0]
    fT, fS = float((Tn * area).sum() / area.sum()), float((Sn * area).sum() / area.sum())
    def _gc(la1, lo1, la2, lo2):
        la1, lo1, la2, lo2 = (np.deg2rad(x) for x in (la1, lo1, la2, lo2))
        c = np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2)
        return constants.R_earth * np.arccos(np.clip(c, -1.0, 1.0))
    dxe = np.empty_like(lat); dxe[:, :-1] = _gc(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:]); dxe[:, -1] = dxe[:, -2]
    dxn = np.empty_like(lat); dxn[:-1, :] = _gc(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :]); dxn[-1, :] = dxn[-2, :]
    w = dxe * dxn * wet
    tT, tS = float((T[..., 0] * w).sum() / w.sum()), float((S[..., 0] * w).sum() / w.sum())
    print(f"surface mean T: FESOM {fT:.4f} target {tT:.4f} (diff {tT - fT:+.4f} K); "
          f"S: FESOM {fS:.4f} target {tS:.4f} (diff {tS - fS:+.4f} psu)")

    # static inversions over active wet cells, both levels at the deeper pressure
    depth = np.abs(zc)
    n_inv, worst = 0, 0.0
    for k in range(depth.size - 1):
        act = wet & (depth[k + 1] < H)
        if not act.any():
            break
        p_pa = constants.rho_ocean * constants.g * depth[k + 1]
        r_up = np.asarray(wright_eos(T[..., k], S[..., k], p_pa))
        r_dn = np.asarray(wright_eos(T[..., k + 1], S[..., k + 1], p_pa))
        inv = np.where(act, r_up - r_dn, 0.0)
        n_inv += int((inv > a.inversion_tol).sum())
        worst = max(worst, float(inv.max()))
    print(f"static inversions > {a.inversion_tol} kg/m3: {n_inv} cells; worst {worst:.3f} kg/m3")

    np.savez(a.out, T=T, S=S, land_mask=s["land_mask"], lat_T=lat, lon_T=lon, H_bathy=H,
             z_center_ref=zc, _time_s=np.float64(0.0), _step=np.int64(0))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

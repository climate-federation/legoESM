"""Add geographic cell-centred velocity (u_east, v_north) to an MPAS OMIP
snapshot written before the runner stored it.

Rebuilds the mesh EXACTLY as the driver does (same builder, same level and
Lloyd count), asserts the snapshot's cell coordinates match the rebuilt mesh
(so the edge ordering is the run's), applies the canonical Perot
``reconstruct_cell_velocity`` to the stored edge-normal ``u`` and writes a
sibling ``<name>_ugeo.npz`` with every original field plus the two new ones.
The original is never modified.

Usage (compute node):
    python scripts/validate/ocean_fidelity/mpas_snapshot_add_u_east.py \
        --snapshot results/omip_nemo/mpas_unified_180d/snapshot_day0030.npz --level 7
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--lloyd", type=int, default=20)
    a = p.parse_args()
    from scripts.run.run_omip_core2 import (
        _MESH, _NEMO_DOMAIN_CFG, _load_nemo_e3t_1d, build_mpas_ocean)
    from legoesm.ocean.init_mpas import reconstruct_cell_velocity

    src = Path(a.snapshot)
    snap = dict(np.load(src))
    if "u_east" in snap:
        raise SystemExit(f"{src} already carries u_east")
    dz = _load_nemo_e3t_1d(_NEMO_DOMAIN_CFG)
    grid, _z, _model, _state, _H = build_mpas_ocean(
        int(dz.size), 5500.0, _MESH, level=a.level, lloyd_iterations=a.lloyd,
        partial_cell=True, dz_ref_override=dz)
    lat_mesh = np.degrees(np.asarray(grid.latCell))
    lat_snap = np.asarray(snap["lat_T"]).ravel()
    if lat_snap.shape != lat_mesh.shape:
        raise SystemExit(f"cell count {lat_snap.shape} != rebuilt mesh {lat_mesh.shape}")
    dlat = float(np.max(np.abs(lat_snap - lat_mesh)))
    if dlat > 1e-6:
        raise SystemExit(f"rebuilt mesh does not match the snapshot (max |dlat| {dlat:.3e} deg)")
    u = np.asarray(snap["u"], dtype=np.float64)
    if u.shape[0] != int(np.asarray(grid.nEdges)):
        raise SystemExit(f"u {u.shape} is not edge-based for {int(grid.nEdges)} edges")
    ue, vn = reconstruct_cell_velocity(u, grid)
    snap["u_east"] = np.asarray(ue, dtype=np.float64)
    snap["v_north"] = np.asarray(vn, dtype=np.float64)
    out = src.with_name(src.stem + "_ugeo.npz")
    np.savez(out, **snap)
    k = int(np.nanargmax(np.abs(snap["u_east"][:, 0])))
    print(f"wrote {out}: u_east {snap['u_east'].shape}, surface max |u_east| "
          f"{abs(snap['u_east'][k, 0]):.3f} m/s at lat {lat_mesh[k]:.1f}; mesh match {dlat:.1e} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

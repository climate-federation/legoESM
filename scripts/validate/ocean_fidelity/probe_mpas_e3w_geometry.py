"""Which MPAS cells give a non-positive live W-spacing (``e3w_int``)?

The level-8 real-FW arm (job 9631490) died at step 1 with
``raw-mesh e3w_int must contain only finite values > 0`` while the level-7
card with byte-identical flags runs.  This builds the ocean EXACTLY as the
driver does (same builder, same NEMO 75-level ladder, partial cells) and
evaluates the same geometry call the TKE bridge makes, on the initial state,
then lists the offending cells with their bathymetry, mask and eta so the
defect is localised before any code is touched.

Usage (compute node):
    python scripts/validate/ocean_fidelity/probe_mpas_e3w_geometry.py --level 8
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
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--lloyd", type=int, default=20)
    p.add_argument("--no-partial-cell", action="store_true")
    p.add_argument("--nemo-monthly-init", nargs=2, default=None,
                   help="T S NEMO monthly IC files: also count non-finite wet "
                        "cells the IC regrid leaves on this mesh")
    a = p.parse_args()

    from scripts.run.run_omip_core2 import (
        _MESH, _NEMO_DOMAIN_CFG, _load_nemo_e3t_1d, build_mpas_ocean)
    from legoesm.ocean.eos import (
        nemo_bn2_live_geometry, nemo_bn2_live_ladders, nemo_r3t_stretch)

    dz = _load_nemo_e3t_1d(_NEMO_DOMAIN_CFG)
    grid, z_coord, model, state, H_bathy = build_mpas_ocean(
        int(dz.size), 5500.0, _MESH, level=a.level, lloyd_iterations=a.lloyd,
        partial_cell=not a.no_partial_cell, dz_ref_override=dz)
    eta = np.asarray(state.eta.data)
    H = np.asarray(state.H_bathy.data)
    wet = np.asarray(state.land_mask.data if hasattr(state, "land_mask")
                     else grid.land_mask) > 0.5
    print(f"level {a.level}: nCells={H.size} wet={int(wet.sum())} "
          f"H range wet [{H[wet].min():.2f}, {H[wet].max():.2f}] "
          f"H<=0 on wet: {int((H[wet] <= 0).sum())}  eta finite: {np.isfinite(eta).all()}")
    raw = getattr(z_coord, "nemo_e3w_0", None)
    print(f"z_coord.nemo_e3w_0: {None if raw is None else (np.asarray(raw).shape, float(np.asarray(raw).min()))}  "
          f"mesh_reference={getattr(z_coord, 'nemo_e3w_mesh_reference', None)}")
    stretch = np.asarray(nemo_r3t_stretch(z_coord, state.eta.data, state.H_bathy.data))
    print(f"stretch (1+eta/H): finite={np.isfinite(stretch).all()} min={np.nanmin(stretch):.4g} "
          f"n<=0={int((stretch <= 0).sum())} n_nonfinite={int((~np.isfinite(stretch)).sum())}")
    _, _, e3w = nemo_bn2_live_geometry(z_coord, state.eta.data, state.H_bathy.data)
    e3w = np.asarray(e3w)
    bad = ~(np.isfinite(e3w) & (e3w > 0)).all(axis=-1)
    print(f"e3w_int shape {e3w.shape}; bad columns {int(bad.sum())} "
          f"(wet {int((bad & wet).sum())}, land {int((bad & ~wet).sum())})")
    lat = np.degrees(np.asarray(grid.latCell)); lon = np.degrees(np.asarray(grid.lonCell))
    for c in np.flatnonzero(bad)[:20]:
        col = e3w[c]
        k = np.flatnonzero(~(np.isfinite(col) & (col > 0)))
        print(f"  cell {c:7d} lat {lat[c]:7.2f} lon {lon[c]:7.2f} wet={wet[c]} "
              f"H={H[c]:.3f} eta={eta[c]:.3g} stretch={stretch[c]:.4g} "
              f"bad k={k[:5].tolist()} e3w[k0]={col[k[0]]:.3g}")
    if a.nemo_monthly_init:
        from scripts.run.run_omip_core2 import _MESH as _M
        from legoesm.ocean.forcing.nemo_native_fields import (
            load_nemo_monthly_init_ts, nemo_src_tmask_for)
        T_ic, S_ic = load_nemo_monthly_init_ts(
            a.nemo_monthly_init[0], a.nemo_monthly_init[1], lat, lon,
            n_levels=int(z_coord.n_levels), month=1,
            src_tmask=nemo_src_tmask_for(_M, a.nemo_monthly_init[0]))
        T_ic = np.asarray(T_ic); S_ic = np.asarray(S_ic)
        z_half = np.abs(np.asarray(z_coord.z_half_ref))
        active = z_half[None, 1:] <= H[:, None] + 1e-6   # (nCells, nlev)
        active[:, 0] = True
        colmask = active & wet[:, None]
        badT = colmask & ~np.isfinite(T_ic); badS = colmask & ~np.isfinite(S_ic)
        print(f"IC on level {a.level}: wet active cells {int(colmask.sum())}; "
              f"non-finite T {int(badT.sum())} (columns {int(badT.any(axis=1).sum())}), "
              f"S {int(badS.sum())} (columns {int(badS.any(axis=1).sum())}); "
              f"surface T range {np.nanmin(T_ic[wet,0]):.2f}..{np.nanmax(T_ic[wet,0]):.2f} "
              f"S {np.nanmin(S_ic[wet,0]):.2f}..{np.nanmax(S_ic[wet,0]):.2f}")
        for c in np.flatnonzero(badT.any(axis=1) | badS.any(axis=1))[:15]:
            k = np.flatnonzero(badT[c] | badS[c])
            print(f"  IC cell {c:7d} lat {lat[c]:7.2f} lon {lon[c]:7.2f} H={H[c]:.1f} "
                  f"bad levels {k[:6].tolist()} of {int(active[c].sum())}")
        bad = bad | badT.any(axis=1) | badS.any(axis=1)
    return 1 if bad.any() else 0


if __name__ == "__main__":
    raise SystemExit(main())

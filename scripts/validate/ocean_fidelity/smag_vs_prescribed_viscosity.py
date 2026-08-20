#!/usr/bin/env python
"""Is the Smagorinsky term masking the prescribed equatorial viscosity?

WHY.  We replaced a uniform A_h = 1e5 with NEMO's prescribed 3-D field, which
ramps to ~1000 m2/s within a couple of degrees of the equator -- a ~100x
reduction that was expected to restore the Equatorial Undercurrent.  With the
Smagorinsky backstop ON (``--C-smag-lap 3.0``) the day-30 nino3 bias moved by
0.07 C against its control (+3.12 vs +3.19), i.e. essentially nothing, while
the Smagorinsky-OFF twin had moved the equatorial circulation a great deal
before going unstable.

The obvious candidate is that the flow-adaptive term supplies, locally, the
very viscosity the prescribed field removes.  ``A_smag = (C_smag * Delta)**2 *
|D|`` with ``Delta = sqrt(cell area)`` and NO 1/pi normalisation, so at
C_smag = 3 on a 1-degree cell the prefactor is ~1e11 and a strain rate of
1e-6 /s already yields ~1e5 m2/s.

This calls the model's OWN ``smagorinsky_viscosity_cgrid`` on an archived
snapshot -- not a re-derived lookalike -- and compares it, per latitude band,
against the prescribed field the run was configured with.  Reports the RATIO,
because that is what decides whether the prescribed ramp is visible to the
momentum equation at all.

CONFIRMS the masking hypothesis if A_smag >> A_prescribed within a few degrees
of the equator (say a median ratio above ~5).  REFUTES it if A_smag is at or
below the prescribed value there, in which case the ramp is being applied and
its failure to move nino3 has some other cause.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--mesh", type=Path,
                    default=Path("data/grids/eORCA1.2_mesh_mask.nc"))
    ap.add_argument("--visc-file", type=Path, default=None,
                    help="NEMO eddy_viscosity_3D.nc the run was given; "
                         "omit to compare against --A-h-uniform instead.")
    ap.add_argument("--A-h-uniform", type=float, default=None,
                    help="uniform background the control used [m2/s].")
    ap.add_argument("--C-smag", type=float, required=True)
    ap.add_argument("--level", type=int, default=0, help="model level to report.")
    ap.add_argument("--nlev", type=int, default=75)
    ap.add_argument("--H-max", type=float, default=5902.06)
    ap.add_argument("--dt", type=float, default=150.0,
                    help="the run's timestep; the Smagorinsky CFL cap scales "
                         "as 1/dt so this must match the arm.")
    ap.add_argument("--smag-cfl-safety", type=float, default=0.125,
                    help="the run's --smag-cfl-safety.")
    a = ap.parse_args()

    import netCDF4 as nc
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        smagorinsky_viscosity_cgrid,
    )

    z = np.load(a.snapshot)
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    land = np.asarray(z["land_mask"], dtype=np.float64)

    d = nc.Dataset(a.mesh)
    try:
        e1 = np.squeeze(d["e1t"][:]).astype(np.float64)
        e2 = np.squeeze(d["e2t"][:]).astype(np.float64)
    finally:
        d.close()

    # The model's own operator needs the grid object carrying the metrics it
    # uses, so this calls the DRIVER's own build_tripole -- the same
    # constructor the run used -- rather than a duck-typed stand-in, so a
    # metric convention cannot silently differ between here and the model.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.run.run_omip_core2 import build_tripole
    grid = build_tripole(int(a.nlev), float(a.H_max), str(a.mesh),
                         min_levels=2, partial_cell=True)
    if not isinstance(grid, tuple):
        grid_obj = grid
    else:  # some builders return (grid, extras)
        grid_obj = grid[0]
    grid = grid_obj

    u = jnp.asarray(np.asarray(z["u"], dtype=np.float64))
    v = jnp.asarray(np.asarray(z["v"], dtype=np.float64))
    A_smag_raw = np.asarray(smagorinsky_viscosity_cgrid(u, v, grid, a.C_smag))

    # THE MODEL CAPS IT. ocean_pe_latlon_cgrid applies laplacian_smag_cfl_cap
    # (safety * area * cos^2(lat) / dt) before the viscous tendency, so the raw
    # coefficient is NOT what the momentum equation sees. Reporting the
    # uncapped value would overstate the ratio wherever the cap binds.
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        laplacian_smag_cfl_cap,
    )
    cap_h, _cap_q = laplacian_smag_cfl_cap(grid, a.dt, a.smag_cfl_safety)
    cap_h = np.asarray(cap_h)
    A_smag = np.minimum(
        A_smag_raw,
        cap_h[..., None] if A_smag_raw.ndim == 3 else cap_h)
    if A_smag.ndim == 3:
        A_smag = A_smag[..., a.level]
        A_smag_raw = A_smag_raw[..., a.level]
    _bind = float(np.mean(A_smag < A_smag_raw - 1e-9) * 100.0)
    print(f"[cap] laplacian_smag_cfl_cap(safety={a.smag_cfl_safety}, "
          f"dt={a.dt}) binds on {_bind:.2f}% of cells at level {a.level}; "
          "the capped value is what the momentum equation uses.")

    if a.visc_file is not None:
        dv = nc.Dataset(a.visc_file)
        try:
            key = "ahmt_3d" if "ahmt_3d" in dv.variables else "ahmf_3d"
            pres = np.squeeze(dv[key][:]).astype(np.float64)
        finally:
            dv.close()
        if pres.ndim == 3:
            pres = pres[0]
        label = f"{a.visc_file.name}:{key}"
    elif a.A_h_uniform is not None:
        pres = np.full_like(A_smag, a.A_h_uniform)
        label = f"uniform {a.A_h_uniform:g}"
    else:
        raise SystemExit("pass --visc-file or --A-h-uniform")

    area = e1 * e2
    if pres.shape != A_smag.shape:
        # NEMO's prescribed field is on the OUTPUT grid (halo dropped), the
        # model is on the mesh grid. Reuse the ONE shared aligner so this
        # probe and the freshwater budget cannot disagree about the offset.
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "so_freshwater_budget",
            Path(__file__).resolve().parent / "so_freshwater_budget.py")
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        dv = nc.Dataset(a.visc_file)
        try:
            lat_out = np.asarray(dv["nav_lat"][:], dtype=np.float64)
        finally:
            dv.close()
        # Align against the MESH latitudes, not the snapshot's: the snapshot
        # round-trip perturbs them by ~1e-14 deg, and the mesh is the grid the
        # model fields are indexed on anyway.
        _d = nc.Dataset(a.mesh)
        try:
            lat_mesh = np.squeeze(_d["gphit"][:]).astype(np.float64)
        finally:
            _d.close()
        sj, si = _m.align_output_to_mesh(lat_mesh, lat_out)
        print(f"[align] NEMO field {pres.shape} -> mesh slice "
              f"[{sj.start}:{sj.stop}, {si.start}:{si.stop}] (exact latitude "
              "match on valid cells)")
        A_smag = A_smag[sj, si]
        lat = lat[sj, si]
        land = land[sj, si]
        area = area[sj, si]
    wet = (land > 0.5) & np.isfinite(A_smag) & np.isfinite(pres) & (pres > 0)
    print(f"A_smag from the model operator (C_smag={a.C_smag}), level "
          f"{a.level}, vs prescribed {label}\n")
    print(f"  {'band':18s}{'A_smag':>12s}{'prescribed':>12s}{'ratio':>9s}")
    for lo, hi, nm in ((-1, 1, "eq |lat|<1"), (-2, 2, "eq |lat|<2"),
                       (-5, 5, "eq |lat|<5"), (-10, 10, "|lat|<10"),
                       (20, 40, "midlat 20-40N")):
        m = wet & (lat >= lo) & (lat < hi)
        if not m.any():
            continue
        w = area[m]
        s = float(np.sum(A_smag[m] * w) / np.sum(w))
        p = float(np.sum(pres[m] * w) / np.sum(w))
        print(f"  {nm:18s}{s:12.4g}{p:12.4g}{s / p:9.2f}")
    print("\nRatio >> 1 near the equator means the flow-adaptive term, not the "
          "prescribed field, sets the viscosity the momentum equation sees --\n"
          "so lowering the prescribed background there cannot do anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

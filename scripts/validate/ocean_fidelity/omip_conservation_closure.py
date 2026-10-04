#!/usr/bin/env python
"""Canonical conservation-closure probe for the OMIP latlon lane.

Why this exists
---------------
``docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md`` (issue #1492)
Phase 0.1 makes budget closure the primary localisation instrument, on the
FESOM2-JAX argument that *"a drift ... cannot originate inside a kernel and
must instead arise where two kernels are joined. Checking which of the
conservation budgets fails to close then locates the defect."*  Phase 0.3
additionally requires ONE COMMITTED probe per row with locked conventions --
"a row without a canonical probe is UNMEASURED regardless of history".

This module is that probe for the FESOM2-matched CORE2 OMIP row.  It replaces
a set of throwaway inline probes whose numbers were, by 0.3's standard,
unmeasured -- and one of which carried a real argument-order bug before it was
caught.

Scope, stated honestly
----------------------
A snapshot pair gives STATE, not applied boundary fluxes, so this computes the
**redistribution-vs-source** decomposition rather than the full
``d(int Q dV)/dt - sum(applied fluxes)`` residual of 0.1:

* a CONSERVATIVE vertical operator (diffusion, advection between two cells)
  moves tracer between levels and leaves the COLUMN integral unchanged;
* a SOURCE changes the column integral.

That distinction is what localises the defect, and it is decisive for the
2-step blowup this row is chasing: a diffusion instability produces a large
dipole whose two-cell sum is CONSERVED, so a dipole whose sum GROWS is not a
diffusion instability.  Extending to the full 0.1 residual requires the run to
export its applied surface/bottom fluxes; that is not available from
``snapshot_final.npz`` today and is NOT faked here.

Conventions (locked; changing any of these invalidates comparison)
------------------------------------------------------------------
* fp64 throughout (``JAX_ENABLE_X64=1`` is set before jax import; numpy f8).
* WET = ``land_mask > 0.5`` AND ``|z_full_ref| < H_bathy`` (3-D, per level).
* Global integrals are **AREA-WEIGHTED** with ``R^2 dlon dlat cos(lat)``.
  An unweighted global mean on a lat-lon grid is invalid -- meridian
  convergence over-counts polar cells -- and an earlier inline probe made
  exactly that error.
* Column integrals use reference thicknesses ``dz_ref`` (z*, eta ~ 0 at the
  step counts this row examines); the eta stretch is NOT applied, so column
  integrals are comparable between two snapshots but are not absolute volumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("JAX_ENABLE_X64", "1")


def cell_area(lat_deg_2d, dlon_deg: float = 1.0, dlat_deg: float = 1.0):
    """Area [m^2] of each lat-lon cell: ``R^2 * dlon * dlat * cos(lat)``.

    The ``cos(lat)`` factor is the whole point -- see the module docstring.
    """
    from legoesm import constants

    lat = np.asarray(lat_deg_2d, dtype=np.float64)
    return (constants.R_earth ** 2
            * np.deg2rad(dlon_deg) * np.deg2rad(dlat_deg)
            * np.cos(np.deg2rad(lat)))


def wet_mask_3d(land_mask, H_bathy, z_full_ref):
    """3-D wet mask: ocean column AND above the local seafloor."""
    m2 = np.asarray(land_mask, dtype=np.float64) > 0.5
    zc = np.abs(np.asarray(z_full_ref, dtype=np.float64))
    return (zc[None, None, :] < np.asarray(H_bathy)[..., None]) & m2[..., None]


def column_integral(field, dz_ref, wet3):
    """``int q dz`` per column [field units * m], zero on dry cells."""
    w = np.where(wet3, np.asarray(dz_ref, dtype=np.float64)[None, None, :], 0.0)
    return (np.asarray(field, dtype=np.float64) * w).sum(axis=-1)


def global_integral(field, dz_ref, wet3, area):
    """Area-weighted global integral ``int q dV`` [field units * m^3]."""
    return float((column_integral(field, dz_ref, wet3)
                  * np.asarray(area, dtype=np.float64)).sum())


def closure_report(T0, S0, T1, S1, dz_ref, wet3, area, *, top_n: int = 5):
    """Redistribution-vs-source decomposition between two states.

    Returns a dict with, for each of T and S: the area-weighted global
    integral at both times and its relative change, plus the columns whose
    integral moved most (a conservative operator moves NONE of them).
    """
    out: dict = {}
    for name, A0, A1 in (("T", T0, T1), ("S", S0, S1)):
        g0 = global_integral(A0, dz_ref, wet3, area)
        g1 = global_integral(A1, dz_ref, wet3, area)
        c0 = column_integral(A0, dz_ref, wet3)
        c1 = column_integral(A1, dz_ref, wet3)
        dcol = c1 - c0
        wet_col = wet3.any(axis=-1)
        dcol_wet = np.where(wet_col, np.abs(dcol), -np.inf)
        flat = np.argsort(dcol_wet, axis=None)[::-1][:top_n]
        worst = [
            {"index": [int(a) for a in np.unravel_index(f, dcol.shape)],
             "delta_column_integral": float(dcol.ravel()[f])}
            for f in flat
        ]
        out[name] = {
            "global_integral_t0": g0,
            "global_integral_t1": g1,
            "relative_change": (g1 - g0) / abs(g0) if g0 != 0.0 else float("nan"),
            "max_abs_column_delta": float(np.nanmax(dcol_wet[np.isfinite(dcol_wet)]))
            if np.isfinite(dcol_wet).any() else 0.0,
            "worst_columns": worst,
        }
    return out


def two_cell_sum_delta(T0, S0, T1, S1, dz_ref, j, i, k0, k1):
    """Change in the ``sum(q*dz)`` of ONE adjacent cell pair.

    The discriminator this row turns on: a conservative vertical operator
    leaves this EXACTLY zero (it moves tracer from one cell to the other), so
    a non-zero value means a SOURCE, not a diffusion instability.
    """
    dz = np.asarray(dz_ref, dtype=np.float64)
    sl = slice(k0, k1 + 1)
    res = {}
    for name, A0, A1 in (("T", T0, T1), ("S", S0, S1)):
        a = float((np.asarray(A0)[j, i, sl] * dz[sl]).sum())
        b = float((np.asarray(A1)[j, i, sl] * dz[sl]).sum())
        res[name] = {"t0": a, "t1": b, "delta": b - a}
    return res


def _git_sha() -> str:
    from legoesm.io.git_provenance import git_provenance
    return git_provenance(_PROJECT_ROOT).commit or "unknown"


def load_dz_ref(path) -> np.ndarray:
    """Layer thicknesses from a FESOM-style z-axis dump or a plain dz list."""
    raw = np.loadtxt(path, comments="#", dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError(f"{path}: expected a 1-D thickness/interface list")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{path}: non-finite entry")
    # Validate SIGNS ON THE RAW INPUT, before any abs().  Taking abs() first
    # silently rewrites a bad -5.0 into a plausible 5.0, which is how a
    # corrupt axis would sail through the positivity check below.
    if np.all(raw >= 0.0):
        z = raw
    elif np.all(raw <= 0.0):
        z = -raw                       # FESOM-style negative depths
    else:
        raise ValueError(
            f"{path}: mixed-sign axis {raw[:5]}... — expected all-positive "
            "thicknesses/depths or all-negative depths, not both."
        )
    # A leading 0 with strictly increasing values = INTERFACES (n+1 entries for
    # n layers); anything else is already a thickness list.
    dz = np.diff(z) if (z[0] == 0.0 and np.all(np.diff(z) > 0)) else z
    if not np.all(dz > 0):
        raise ValueError(f"{path}: non-positive layer thickness in {dz[:5]}...")
    return dz


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True,
                   help="snapshot_final.npz from the run under test")
    p.add_argument("--ic-snapshot", default=None,
                   help="Optional second npz to use as t0 instead of --woa-t/--woa-s.")
    p.add_argument("--woa-t", default=None, help="IC temperature NetCDF (t0)")
    p.add_argument("--woa-s", default=None, help="IC salinity NetCDF (t0)")
    p.add_argument("--dz-ref-file", required=True,
                   help="Layer thicknesses / interfaces (e.g. core2_dz.txt)")
    p.add_argument("--pair", default=None, metavar="j,i,k0,k1",
                   help="Also report the two-cell-sum delta for this cell pair.")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_woa import init_ocean_from_woa
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    d = np.load(args.snapshot)
    T1 = np.asarray(d["T"], dtype=np.float64)
    S1 = np.asarray(d["S"], dtype=np.float64)
    dz = load_dz_ref(args.dz_ref_file)
    z_coord = create_z_star_from_thicknesses(dz)
    wet3 = wet_mask_3d(d["land_mask"], d["H_bathy"], z_coord.z_full_ref)
    area = cell_area(d["lat_T"] if np.asarray(d["lat_T"]).ndim == 2
                     else np.broadcast_to(np.asarray(d["lat_T"])[:, None],
                                          T1.shape[:2]))

    if args.ic_snapshot:
        d0 = np.load(args.ic_snapshot)
        T0 = np.asarray(d0["T"], dtype=np.float64)
        S0 = np.asarray(d0["S"], dtype=np.float64)
    else:
        if not (args.woa_t and args.woa_s):
            raise SystemExit("need --ic-snapshot, or both --woa-t and --woa-s")
        grid = create_latlon_grid(T1.shape[0], T1.shape[1])
        T0, S0 = init_ocean_from_woa(grid, z_coord, args.woa_t, args.woa_s)
        T0 = np.asarray(T0, dtype=np.float64)
        S0 = np.asarray(S0, dtype=np.float64)

    rep = closure_report(T0, S0, T1, S1, dz, wet3, area)
    rep["_provenance"] = {
        "probe": str(Path(__file__).relative_to(_PROJECT_ROOT)),
        "git_sha": _git_sha(),
        "snapshot": str(args.snapshot),
        "ic": str(args.ic_snapshot or f"{args.woa_t} / {args.woa_s}"),
        "dz_ref_file": str(args.dz_ref_file),
        "step_in_snapshot": int(d["_step"]) if "_step" in d.files else None,
        "x64": os.environ.get("JAX_ENABLE_X64"),
        "area_weighted": True,
    }
    if args.pair:
        j, i, k0, k1 = (int(v) for v in args.pair.split(","))
        rep["two_cell_sum"] = two_cell_sum_delta(T0, S0, T1, S1, dz,
                                                 j, i, k0, k1)

    for name in ("T", "S"):
        r = rep[name]
        print(f"{name}: global int {r['global_integral_t0']:.8e} -> "
              f"{r['global_integral_t1']:.8e}  rel {r['relative_change']:+.4e}")
        print(f"   max |column delta| = {r['max_abs_column_delta']:.4e}"
              f"   worst {r['worst_columns'][0]['index']}")
    if "two_cell_sum" in rep:
        for name, v in rep["two_cell_sum"].items():
            verdict = ("SOURCE (not conservative)" if abs(v["delta"]) > 1e-9
                       else "conserved")
            print(f"two-cell sum {name}: {v['t0']:.4f} -> {v['t1']:.4f} "
                  f"delta {v['delta']:+.4f}  [{verdict}]")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

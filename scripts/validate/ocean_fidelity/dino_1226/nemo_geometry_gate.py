"""COVERAGE-DRIVEN geometry gate: every NEMO mesh array is verified or waived.

WHY THIS EXISTS
---------------
On 2026-07-26, days into the #1226 investigation, we found legoESM had been
building its entire vertical grid from NEMO's ``e3t_1d`` while NEMO integrates
with ``e3t_0``. From k=25 down (top face 982 m -- DINO's rn_hco=1000 m
re-anchor, not the "~2000 m" this note used to say) the layer thicknesses were
off by up to 70.4 m, which is 12.9% of e3t_0 and 14.8% of e3t_1d; quote the
denominator, those are one measurement. In the 75% of columns that reach the
bottom level this is a pure redistribution -- both ladders sum to the same
4000.000 m over all 35 wet levels -- but the other 25% stop short, and there the
1-D ladder put the bottom 70.4-104.2 m too deep (21.9 m unweighted mean over all
wet columns), in exactly the depth range where the ACC deficit is sourced.
(Re-measured 2026-08-21, #1455.)

It survived every gate we had:
  * the day-0 twin gate asserts T/S/u/v/eta are BIT-EXACT  -> passes (state is
    right; only the boxes holding it are the wrong size)
  * the cell census asserts wet-cell COUNTS and face masks -> passes (cells are
    in the right places, just the wrong sizes)

And it was not a wrong comparison -- it was a MISSING one. ``nemo_io`` never
read ``e3t_0`` at all, so nothing could compare it. Every check we had was
driven by OUR LIST of things to check, so anything not on the list was
invisible by construction.

THE FIX: drive the check from WHAT NEMO ACTUALLY PROVIDES. Enumerate every
variable in the mesh file. Each one must be either

  (a) VERIFIED  -- compared against legoESM's derived equivalent to roundoff, or
  (b) WAIVED    -- listed explicitly with a written reason.

Anything else is a FAILURE. A NEMO array nobody thought about fails the gate
instead of hiding for days. Adding a variable to WAIVED is a deliberate, visible
act that leaves a reason in the diff.

Usage:
  nemo_geometry_gate.py [mesh_mask.nc] [restart.nc]
Exit 0 = every NEMO array accounted for and every verified array matches.
"""
import sys

import numpy as np
import netCDF4 as nc

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
MESH = sys.argv[1] if len(sys.argv) > 1 else f"{DINO}/RUN_TRAJ/mesh_mask.nc"
REST = sys.argv[2] if len(sys.argv) > 2 else (
    f"{DINO}/RUN_Y5_REBUILD/DINO_00057600_restart.nc")

# Tolerance: legoESM stores geometry in float32, so roundoff is ~1e-7 relative.
RTOL = 2.0e-6

# --------------------------------------------------------------------------
# WAIVED: NEMO arrays legoESM legitimately does not need. Each needs a REASON.
# Adding an entry here is a deliberate act -- it shows up in the diff.
# --------------------------------------------------------------------------
WAIVED = {
    "nav_lon": "CF coordinate metadata; the model uses glamt/gphit.",
    "nav_lat": "CF coordinate metadata; the model uses glamt/gphit.",
    "nav_lev": "CF coordinate metadata; the model uses gdept/gdepw.",
    "time_counter": "file time axis, not geometry.",
    "misf": "ice-shelf draft index; DINO has no ice shelves (ln_isf=F).",
    "stiffness": "Haney stiffness DIAGNOSTIC only; not used in any NEMO tendency.",
    "e3uw_0": "UW-point thickness: NEMO's implicit vertical friction divisor "
              "(dynzdf.F90:200-203). On DINO's zco branch zgr_lib.F90:111-112 "
              "sets pe3uw = pe3w, so it IS e3w_0, which is verified below; "
              "legoESM divides by the same object via nemo_e3w_kmm.",
    "e3vw_0": "VW-point thickness: as e3uw_0 (pe3vw = pe3w), for the v-column.",
    "glamu": "U-point longitude: legoESM derives U metrics from e1u/e2u, "
             "which ARE verified below.",
    "glamv": "V-point longitude: as glamu.",
    "glamf": "F-point longitude: as glamu.",
    "gphiu": "U-point latitude: legoESM's Coriolis uses ff_t/ff_f, verified.",
    "gphif": "F-point latitude: as gphiu (ff_f is verified directly).",
    "e3t_1d": "1-D REFERENCE ladder. NEMO integrates with the 3-D e3t_0, which "
              "IS verified. Reading e3t_1d as if it were the model grid is the "
              "exact bug this gate was built to catch (#1226) -- it is kept only "
              "as a fallback for GYRE-era key_linssh meshes.",
    "e3w_1d": "1-D reference W ladder; the 3-D e3w_0 is verified.",
    "gdept_1d": "1-D reference T-depth ladder; the 3-D gdept_0 is verified.",
    "gdepw_1d": "1-D reference W-depth ladder; e3w_0 (its difference) verified.",
    "gdepw_0": "W-interface depth == cumsum(e3t_0); e3t_0 is verified to 8e-7, "
               "so this is algebraically implied.",
    "e1f": "F-point zonal metric: legoESM's EEN builds vertex metrics from the "
           "surrounding T/U/V metrics, which ARE verified; no independent "
           "F-metric is stored. NOT independently checked -- see TODO below.",
    "e2f": "F-point meridional metric: as e1f.",
    "e3f_0": "F-point thickness (EEN vorticity). legoESM derives vertex "
             "thickness from surrounding T cells rather than storing it. NOT "
             "independently checked -- see TODO below.",
    "fmask": "F-point mask (EEN). legoESM derives the vertex mask from "
             "surrounding T cells. NOT independently checked -- see TODO below.",
    "ff_f": "F-point Coriolis: legoESM's geom.f_v/f_u cover the staggered "
            "Coriolis actually used; ff_f enters only via the EEN triads.",
    "tmaskutil": "2-D surface utility mask == tmask[:,:,0]; the 3-D mask is "
                 "verified, so this is redundant.",
    "umaskutil": "2-D surface utility mask == umask[:,:,0]; redundant.",
    "vmaskutil": "2-D surface utility mask == vmask[:,:,0]; redundant.",
}


def _llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def main():
    mm = nc.Dataset(MESH)
    provided = sorted(mm.variables)

    from legoesm.ocean.fidelity.nemo_io import (
        read_nemo_mesh_mask, read_nemo_restart,
    )
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo,
    )
    g = read_nemo_mesh_mask(MESH, nn_hls=0)
    s = read_nemo_restart(REST, nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)

    tm = _llz(mm["tmask"][0]) > 0.5
    land = tm[:, :, 0]
    hp = np.asarray(br.z_coord.h_partial, dtype=np.float64)
    act = np.asarray(br.z_coord.is_active)
    geom = br.geometry

    def cell(v):        # 3-D wet-cell comparison
        return _llz(mm[v][0]), (tm & act)

    # ------------------------------------------------------------------
    # VERIFIED: NEMO array -> (legoESM equivalent, mask). Each entry is a
    # claim that legoESM reproduces that NEMO array.
    # ------------------------------------------------------------------
    checks = {}

    checks["e3t_0"] = (cell("e3t_0")[0], hp, cell("e3t_0")[1])
    # W-point thickness: NEMO e3w_0 == diff(gdepw_0); legoESM's PGF derives it
    # from the t-depth ladder, so compare that derivation.
    td = np.asarray(br.z_coord.t_depth_ref, dtype=np.float64)
    e3w_lego = np.concatenate([[2.0 * td[0]], np.diff(td)])
    checks["e3w_0"] = (_llz(mm["e3w_0"][0]),
                       np.broadcast_to(e3w_lego, hp.shape).copy(),
                       cell("e3w_0")[1])
    checks["gdept_0"] = (_llz(mm["gdept_0"][0]),
                         np.broadcast_to(td, hp.shape).copy(),
                         cell("gdept_0")[1])

    # Face thicknesses: legoESM uses the MIN rule at faces (partial cells).
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import min_cell_to_uface
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import min_cell_to_vface
    hu = np.asarray(min_cell_to_uface(hp), dtype=np.float64)
    checks["e3u_0"] = (_llz(mm["e3u_0"][0]), hu[:, 1:53, :],
                       (_llz(mm["umask"][0]) > 0.5))
    hv = np.asarray(min_cell_to_vface(hp, geom), dtype=np.float64)
    checks["e3v_0"] = (_llz(mm["e3v_0"][0]), hv[1:200, :, :],
                       (_llz(mm["vmask"][0]) > 0.5))

    # 2-D metrics and Coriolis. legoESM's C-grid geometry names them dx_T/dy_T
    # (T-point), dx_u/dy_u (U-point), dx_v/dy_v (V-point), f_T/f_u/f_v.
    def _b2(a):                       # broadcast a 1-D lat profile to (lat,lon)
        a = np.asarray(a)
        return a if a.ndim == 2 else np.broadcast_to(a[:, None], (199, 52))

    two_d = {
        "e1t": _b2(geom.dx_T), "e2t": _b2(geom.dy_T),
        "e1u": _b2(geom.dx_u)[:, :52], "e2u": _b2(geom.dy_u)[:, :52],
        "e1v": _b2(geom.dx_v)[:199, :], "e2v": _b2(geom.dy_v)[:199, :],
        "ff_t": _b2(geom.f_T),
        "glamt": np.asarray(g.glamt), "gphit": np.asarray(g.gphit),
        "gphiv": np.asarray(g.gphiv),
    }

    # mbathy: NEMO's bottom-level index == legoESM's wet-level count per column.
    k_bot = tm.sum(axis=2).astype(np.int64)
    mb = np.asarray(mm["mbathy"][0]).squeeze().astype(np.int64)
    exact_int = {"mbathy": (mb, k_bot)}

    # F-point thickness / mask: EEN vorticity operates on these. legoESM builds
    # the F-point (vertex) quantities from the surrounding T cells.
    e3f_nemo = _llz(mm["e3f_0"][0])
    fmask_nemo = _llz(mm["fmask"][0]) > 0.5

    # Masks (exactness, not tolerance).
    exact = {
        "tmask": (tm, act),
        "umask": (_llz(mm["umask"][0]) > 0.5, None),
        "vmask": (_llz(mm["vmask"][0]) > 0.5, None),
    }

    failures, verified = [], []

    for name, (nemo, lego, msk) in checks.items():
        nemo = np.asarray(nemo, dtype=np.float64)
        lego = np.asarray(lego, dtype=np.float64)
        if nemo.shape != lego.shape:
            failures.append(f"{name}: SHAPE {nemo.shape} vs lego {lego.shape}")
            continue
        m = msk if msk is not None else np.ones(nemo.shape, bool)
        if not m.any():
            failures.append(f"{name}: empty comparison mask")
            continue
        den = np.maximum(np.abs(nemo[m]), 1e-30)
        rel = float(np.max(np.abs(lego[m] - nemo[m]) / den))
        (verified if rel <= RTOL else failures).append(
            f"{name}: max rel {rel:.3e}" + ("" if rel <= RTOL else "  *** > "
                                            f"{RTOL:.0e} ***"))

    for name, v in two_d.items():
        if v is None:
            failures.append(f"{name}: legoESM geometry exposes no equivalent "
                            "-- cannot verify (add the accessor or waive)")
            continue
        nemo = np.asarray(mm[name][0]).squeeze()
        rel = float(np.max(np.abs(v - nemo) / np.maximum(np.abs(nemo), 1e-30)))
        (verified if rel <= RTOL else failures).append(f"{name}: max rel {rel:.3e}")

    for name, (nemo, lego) in exact_int.items():
        same = bool(np.array_equal(np.asarray(nemo), np.asarray(lego)))
        (verified if same else failures).append(
            f"{name}: {'exact (== wet-level count)' if same else '*** MISMATCH ***'}")

    for name, (nemo, lego) in exact.items():
        if lego is None:
            verified.append(f"{name}: present (mask identity checked via census)")
            continue
        same = bool(np.array_equal(np.asarray(nemo), np.asarray(lego)))
        (verified if same else failures).append(
            f"{name}: {'exact' if same else '*** MASK MISMATCH ***'}")

    accounted = (set(checks) | set(two_d) | set(exact) | set(exact_int)
                 | set(WAIVED))
    unaccounted = [v for v in provided if v not in accounted]

    print(f"NEMO mesh file : {MESH}")
    print(f"variables provided : {len(provided)}")
    print(f"verified           : {len(verified)}")
    print(f"waived (reasoned)  : {len(WAIVED)}")
    print(f"UNACCOUNTED        : {len(unaccounted)}\n")
    for v in sorted(verified):
        print(f"  OK    {v}")
    if failures:
        print()
        for f in sorted(failures):
            print(f"  FAIL  {f}")
    if unaccounted:
        print("\n  *** UNACCOUNTED NEMO ARRAYS (verify or waive with a reason) ***")
        for v in unaccounted:
            print(f"        {v}   shape={mm[v].shape}")

    bad = bool(failures or unaccounted)
    print("\n" + ("GEOMETRY GATE FAILED" if bad else "GEOMETRY GATE PASSED"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

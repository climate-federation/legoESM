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
    p.add_argument("--dt", type=float, default=75.0)
    p.add_argument("--snapshot", default=None,
                   help="snapshot npz of a blowing-up run: print the structure "
                        "of the max-|u| mode (edge profile, both cells' columns, "
                        "the cells' other edges)")
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
    lat = np.degrees(np.asarray(grid.latCell)); lon = np.degrees(np.asarray(grid.lonCell))
    from legoesm import constants
    dc = np.asarray(grid.dcEdge); dv = np.asarray(grid.dvEdge); ar = np.asarray(grid.areaCell)
    print(f"mesh quality: dcEdge min/median/max {dc.min():.0f}/{np.median(dc):.0f}/{dc.max():.0f} m, "
          f"dvEdge min/median {dv.min():.0f}/{np.median(dv):.0f} m, "
          f"areaCell min/median/max ratio {ar.min()/np.median(ar):.3f}/{ar.max()/np.median(ar):.3f}; "
          f"n(dcEdge<0.2*median)={int((dc < 0.2*np.median(dc)).sum())}, n(dvEdge<0.05*median)={int((dv < 0.05*np.median(dv)).sum())}")
    # Time-stepping CFL at the card's dt: barotropic gravity wave c=sqrt(gH)
    # per edge (deeper of the two cells) and the explicit Laplacian number.
    coe = np.asarray(grid.cellsOnEdge)
    coe = coe if coe.shape[0] == 2 else coe.T
    He = np.maximum(H[coe[0]], H[coe[1]])
    c_bt = np.sqrt(constants.g * He)
    cfl = c_bt * a.dt / dc
    ie = int(np.argmax(cfl))
    lat_e = np.degrees(np.asarray(grid.latEdge)); lon_e = np.degrees(np.asarray(grid.lonEdge)) % 360
    print(f"CFL(dt={a.dt}s): barotropic c*dt/dcEdge max {cfl.max():.3f} at edge {ie} "
          f"({lat_e[ie]:.2f}N, {lon_e[ie]:.2f}E, H={He[ie]:.0f} m, dc={dc[ie]:.0f} m); "
          f"n(cfl>0.5)={int((cfl > 0.5).sum())}, n(cfl>1)={int((cfl > 1).sum())}; "
          f"Laplacian A_h=1e5 dt/dc^2 max {1e5 * a.dt / dc.min() ** 2:.4f}")
    for name, (la, lo) in {"Gibraltar 36N 354.4E": (36.0, 354.4), "Alboran 35.9N 358.9E": (35.93, 358.92),
                           "BlackSea 44.5N 34.5E": (44.5, 34.5), "Bosporus 41.1N 29E": (41.1, 29.0)}.items():
        d = np.hypot(lat - la, (lon % 360 - lo + 180) % 360 - 180)
        near = d < 0.6
        print(f"  {name:22s} cells<0.6deg {int(near.sum()):3d} wet {int((near & wet).sum()):3d} "
              f"H wet max {H[near & wet].max() if (near & wet).any() else 0:.0f} m")
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
        # Deep-IC horizontal spread at the level-8 blowup sites: adjacent cells
        # filled from different NEMO source columns show up as a T/S jump.
        zc = np.abs(np.asarray(z_coord.z_full_ref))
        for name, (la, lo) in {"Alboran 35.9N 358.9E": (35.93, 358.92),
                               "BlackSea 44.5N 34.5E": (44.5, 34.5)}.items():
            near = (np.hypot(lat - la, (lon % 360 - lo + 180) % 360 - 180) < 0.6) & wet
            print(f"deep IC spread at {name} ({int(near.sum())} wet cells): level  depth  nact  T min/max  S min/max")
            for k in range(30, int(z_coord.n_levels), 4):
                act = near & active[:, k]
                if act.sum() < 2:
                    continue
                print(f"   k={k:2d} {zc[k]:7.0f} m  n={int(act.sum()):2d}  "
                      f"T {T_ic[act, k].min():6.2f}/{T_ic[act, k].max():6.2f}  "
                      f"S {S_ic[act, k].min():6.2f}/{S_ic[act, k].max():6.2f}")
    # TRiSK tangential-velocity weights: reconstruct a uniform flow (exact on a
    # perfect hex mesh, O(10%) on a Lloyd mesh); a weight blow-up shows as
    # |error| >> 1 at specific edges and as a large sum|w|.
    W = np.asarray(grid.weightsOnEdge); EoE = np.asarray(grid.edgesOnEdge)
    if W.shape[0] != dc.size:
        W, EoE = W.T, EoE.T                      # -> (nEdges, maxEdges2)
    ang = np.asarray(grid.angleEdge)
    nx_, ny_ = np.cos(ang), np.sin(ang); tx_, ty_ = -np.sin(ang), np.cos(ang)
    valid = EoE >= 0
    worst = 0.0
    for name, (Ux, Uy) in (("east", (1.0, 0.0)), ("north", (0.0, 1.0))):
        ue = Ux * nx_ + Uy * ny_
        vt_true = Ux * tx_ + Uy * ty_
        vt = np.where(valid, W * ue[np.where(valid, EoE, 0)], 0.0).sum(axis=1)
        err = np.abs(vt - vt_true)
        ie = int(np.argmax(err)); worst = max(worst, float(err.max()))
        print(f"TRiSK weights, uniform {name} flow: |v_t err| max {err.max():.3g} at edge {ie} "
              f"({lat_e[ie]:.2f}N {lon_e[ie]:.2f}E), p99 {np.percentile(err, 99):.3g}, "
              f"n(err>0.5)={int((err > 0.5).sum())}")
    sw = np.abs(np.where(valid, W, 0.0)).sum(axis=1)
    print(f"sum|w| per edge: median {np.median(sw):.3f} max {sw.max():.3f} at edge {int(np.argmax(sw))} "
          f"({lat_e[int(np.argmax(sw))]:.2f}N {lon_e[int(np.argmax(sw))]:.2f}E); n(sum|w|>3)={int((sw > 3).sum())}")
    if a.snapshot:
        zs = np.load(a.snapshot)
        u = np.asarray(zs["u"]); eta = np.asarray(zs["eta"]) if "eta" in zs.files else None
        Ts = np.asarray(zs["T"]); Ss = np.asarray(zs["S"])
        ie, k = np.unravel_index(int(np.nanargmax(np.abs(u))), u.shape)
        c0, c1 = int(coe[0, ie]), int(coe[1, ie])
        eoc = np.asarray(grid.edgesOnCell); eoc = eoc if eoc.shape[0] == H.size else eoc.T
        print(f"SNAPSHOT {a.snapshot}: max|u| {abs(u[ie, k]):.3g} at edge {ie} level {k} "
              f"({lat_e[ie]:.2f}N {lon_e[ie]:.2f}E), cells {c0},{c1} H={H[c0]:.0f}/{H[c1]:.0f} m "
              f"wet={wet[c0]}/{wet[c1]} eta={None if eta is None else (round(float(eta[c0]),4), round(float(eta[c1]),4))}")
        ks = range(max(0, k - 6), min(u.shape[1], k + 7))
        print("  level   u_edge      T c0     T c1     S c0     S c1")
        for kk in ks:
            print(f"  {kk:5d} {u[ie, kk]:+10.3g} {Ts[c0, kk]:8.3f} {Ts[c1, kk]:8.3f} {Ss[c0, kk]:8.3f} {Ss[c1, kk]:8.3f}")
        for c in (c0, c1):
            es = [int(e) for e in eoc[c] if 0 <= int(e) < u.shape[0]]
            print(f"  cell {c} edges at level {k}: " + " ".join(f"{u[e, k]:+.3g}" for e in es))
        # vertical column max profile: where in the column does |u| live
        col = np.abs(u[ie]); print(f"  |u| edge profile max at level {int(np.argmax(col))}, "
                                   f"top {col[0]:.3g}, mid {col[len(col)//2]:.3g}, bottom-active {col[np.flatnonzero(col)[-1]] if col.any() else 0:.3g}")
        # Per-term momentum tendency on this state (no physics: dynamics only).
        import jax.numpy as jnp
        from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
        st = state._replace(
            u=state.u.replace(data=jnp.asarray(u)), T=state.T.replace(data=jnp.asarray(Ts)),
            S=state.S.replace(data=jnp.asarray(Ss)),
            eta=state.eta.replace(data=jnp.asarray(eta if eta is not None else np.zeros(H.size))))
        _, terms = mpas_ocean_baroclinic_tendencies(
            st, model.mesh, z_coord, model.config, physics_fn=None, term_diagnostics=True)
        names = ("grad_B", "pv_flux", "visc", "vert_adv_u", "du_dt_full")
        print("  per-term tendency at the max edge [m/s^2] (x dt=75 -> m/s per step):")
        print("  level " + " ".join(f"{n:>11s}" for n in names) + "        w_e")
        we = np.asarray(terms["w_e"])
        for kk in ks:
            print(f"  {kk:5d} " + " ".join(f"{float(np.asarray(terms[n])[ie, kk]):+11.3e}" for n in names)
                  + f" {float(we[ie, kk]):+11.3e}")
        print("  global max |term| and location:")
        for n in names:
            t = np.abs(np.asarray(terms[n])); j, kk = np.unravel_index(int(np.nanargmax(t)), t.shape)
            print(f"    {n:11s} {t[j, kk]:.3e} at edge {j} ({lat_e[j]:.2f}N {lon_e[j]:.2f}E) level {kk}")
    return 1 if bad.any() else 0


if __name__ == "__main__":
    raise SystemExit(main())

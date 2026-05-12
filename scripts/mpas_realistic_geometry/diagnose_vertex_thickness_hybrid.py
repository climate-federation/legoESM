"""Diagnose where ``vertex_thickness_hybrid`` triggers its min-rule
fallback on the ETOPO+ico4 mesh.

Hypothesis (audit 2026-05-04):
    The ``alpha=0.5`` switch ``use_min = h_min < 0.5*h_max`` makes
    ``h_v`` artificially small at deep partial-cell step vertices,
    amplifying ``q = ζ/h_v`` by O(h_max/h_min) at exactly the
    topographic steps where the bottom-trapped instability lives.

This script answers:
    1. What fraction of (vertex, level) pairs trigger the min branch?
    2. What is the distribution of h_min/h_kite at triggered points?
    3. What is the spatial distribution (per latitude band, by depth)?
    4. Is the trigger concentrated in the deepest active level per
       vertex (which is where ETOPO's partial-cell step lives)?

Mirrors the §8f run config: subdivision_level=4 (2562 cells, ~480 km),
ETOPO 1deg, n_levels=20, H_max=5500 m, smoothing_passes=2,
r_factor_max=0.2.

Run::

    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/\\
        diagnose_vertex_thickness_hybrid.py \\
        --etopo-path data/bathymetry/etopo_1deg.nc
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    BIG_H,
    kite_area_vertex_thickness,
    min_cell_to_vertex,
)
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


def diagnose(args):
    print(f"[diag-vth] subdivision={args.subdivision} "
          f"H_max={args.H_max} m, n_levels={args.n_levels}, "
          f"alpha={args.alpha}")
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    z_coord = create_ocean_z_star(n_levels=args.n_levels, H_max=args.H_max)

    if args.etopo_path:
        bathy_cfg = BathymetryConfig(
            source="file", path=args.etopo_path,
            H_max=args.H_max, H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            enforce_straits=True,
            r_factor_max=args.r_factor_max,
            depth_is_negative=True,
        )
    else:
        bathy_cfg = None

    if bathy_cfg is not None:
        state = rest_state_mpas_ocean(
            mesh, z_coord, bathymetry=bathy_cfg,
        )
    else:
        state = rest_state_mpas_ocean(mesh, z_coord, H_max=args.H_max)

    H_bathy = state.H_bathy.data
    pc_coord = create_partial_cell_coordinate(z_coord, H_bathy)
    h_partial = pc_coord.h_partial  # (nCells, nlev)
    is_active = pc_coord.is_active   # (nCells, nlev) bool
    print(f"[diag-vth] nCells={mesh.nCells}, nVertices={mesh.nVertices}, "
          f"nlev={args.n_levels}")
    print(f"[diag-vth] active cells (level 0): "
          f"{int(jnp.sum(is_active[:, 0]))} / {mesh.nCells}")
    print(f"[diag-vth] active (cell, level) pairs: "
          f"{int(jnp.sum(is_active))} / {mesh.nCells * args.n_levels}")
    print(f"[diag-vth] H_bathy range = "
          f"[{float(jnp.min(jnp.where(is_active[:, 0], H_bathy, jnp.inf))):.0f}, "
          f"{float(jnp.max(H_bathy)):.0f}] m")

    # --------------------------------------------------------------
    # Replicate the vertex_thickness_hybrid internals so we can
    # extract h_min, h_max, h_kite, and the use_min mask separately.
    # --------------------------------------------------------------
    cov = mesh.cellsOnVertex                    # (3, nVertices)
    cov_valid = cov >= 0
    cov_safe = jnp.maximum(cov, 0)
    h_gathered = h_partial[cov_safe]            # (3, nVertices, nlev)
    valid = cov_valid[:, :, None] & (h_gathered > 0.0)

    h_for_max = jnp.where(valid, h_gathered, 0.0)
    h_max = jnp.max(h_for_max, axis=0)          # (nVertices, nlev)
    h_for_min = jnp.where(valid, h_gathered, BIG_H)
    h_min_raw = jnp.min(h_for_min, axis=0)
    any_wet = jnp.any(valid, axis=0)
    h_min = jnp.where(any_wet, h_min_raw, 0.0)

    h_kite = kite_area_vertex_thickness(h_partial, mesh)

    use_min = (h_min < args.alpha * h_max) & any_wet
    n_triggered = int(jnp.sum(use_min))
    n_active = int(jnp.sum(any_wet))
    n_total = mesh.nVertices * args.n_levels
    print()
    print("=== Trigger statistics ===")
    print(f"  active (vertex, level) pairs: {n_active}/{n_total} "
          f"({100*n_active/n_total:.1f}%)")
    print(f"  triggered (use_min)         : {n_triggered} "
          f"({100*n_triggered/max(n_active,1):.2f}% of active)")
    if n_triggered == 0:
        print("  → hybrid never triggers. Audit hypothesis: FALSIFIED on this grid.")
        return

    # --------------------------------------------------------------
    # Distribution of the amplification ratio at triggered points.
    # q_ratio = h_kite / h_v_hybrid = h_kite / h_min   (when use_min)
    # i.e. how much q is amplified by the hybrid switch vs pure kite.
    # --------------------------------------------------------------
    h_kite_safe = jnp.maximum(h_kite, 1.0e-10)
    h_min_safe  = jnp.maximum(h_min,  1.0e-10)
    q_ratio_field = jnp.where(use_min, h_kite_safe / h_min_safe, 1.0)
    q_amp_at_triggered = q_ratio_field[use_min]
    h_min_at_triggered = h_min[use_min]
    h_max_at_triggered = h_max[use_min]
    h_kite_at_triggered = h_kite[use_min]

    pcts = [10, 25, 50, 75, 90, 95, 99, 100]
    qa = np.array(q_amp_at_triggered)
    print()
    print("=== q amplification at triggered points (h_kite / h_min) ===")
    print(f"  mean   : {float(qa.mean()):.2f}×")
    print(f"  median : {float(np.median(qa)):.2f}×")
    for p in pcts:
        print(f"  p{p:>3}  : {float(np.percentile(qa, p)):.2f}×")
    print(f"  >  2× : {int(np.sum(qa > 2)):>6} / {len(qa)} "
          f"({100*np.sum(qa > 2)/len(qa):.1f}%)")
    print(f"  >  5× : {int(np.sum(qa > 5)):>6} / {len(qa)} "
          f"({100*np.sum(qa > 5)/len(qa):.1f}%)")
    print(f"  > 10× : {int(np.sum(qa > 10)):>6} / {len(qa)} "
          f"({100*np.sum(qa > 10)/len(qa):.1f}%)")
    print(f"  > 25× : {int(np.sum(qa > 25)):>6} / {len(qa)} "
          f"({100*np.sum(qa > 25)/len(qa):.1f}%)")

    # --------------------------------------------------------------
    # Per-level: how many triggered, mean amp, and where is the max.
    # --------------------------------------------------------------
    print()
    print("=== Per-level trigger fraction (active subset) ===")
    print(f"  level | active_v | triggered |  frac% | mean_amp | max_amp |  z_top")
    print(f"  ------+----------+-----------+--------+----------+---------+--------")
    z_top = 0.0
    for k in range(args.n_levels):
        active_k = int(jnp.sum(any_wet[:, k]))
        trig_k = int(jnp.sum(use_min[:, k]))
        if trig_k == 0:
            mean_amp = 0.0
            max_amp = 0.0
        else:
            qak = np.array(q_ratio_field[:, k])[np.array(use_min[:, k])]
            mean_amp = float(qak.mean())
            max_amp = float(qak.max())
        dz_k = float(z_coord.dz_ref[k])
        print(f"  {k:5d} | {active_k:8d} | {trig_k:9d} | "
              f"{100*trig_k/max(active_k,1):6.2f} | "
              f"{mean_amp:8.2f}× | {max_amp:7.2f}× | "
              f"{z_top:6.0f}")
        z_top += dz_k

    # --------------------------------------------------------------
    # Bottom-level concentration: at each vertex, find the deepest
    # active level. Is the trigger biased toward there?
    # --------------------------------------------------------------
    bot_lev_vtx = jnp.argmax(any_wet[:, ::-1], axis=1)  # from bottom up
    bot_lev_vtx = (args.n_levels - 1) - bot_lev_vtx     # absolute level
    bot_active = jnp.any(any_wet, axis=1)
    n_bot_active = int(jnp.sum(bot_active))
    # use_min at the bottom-most active level of each vertex
    use_min_bot = use_min[jnp.arange(mesh.nVertices), bot_lev_vtx] & bot_active
    n_bot_trig = int(jnp.sum(use_min_bot))
    # use_min at any non-bottom level of each vertex
    # (vertex with at least one trigger above its bottom level)
    bot_mask = jnp.zeros((mesh.nVertices, args.n_levels), dtype=jnp.bool_)
    bot_mask = bot_mask.at[
        jnp.arange(mesh.nVertices), bot_lev_vtx
    ].set(True)
    use_min_above = use_min & (~bot_mask)
    vtx_has_trig_above = jnp.any(use_min_above, axis=1)
    n_above_trig_vtx = int(jnp.sum(vtx_has_trig_above))
    print()
    print("=== Vertical concentration ===")
    print(f"  vertices with any active level   : {n_bot_active}")
    print(f"  vertices triggering at bot lev   : {n_bot_trig} "
          f"({100*n_bot_trig/max(n_bot_active,1):.1f}%)")
    print(f"  vertices triggering above bot    : {n_above_trig_vtx} "
          f"({100*n_above_trig_vtx/max(n_bot_active,1):.1f}%)")

    # --------------------------------------------------------------
    # Latitudinal distribution: where does the trigger land?
    # --------------------------------------------------------------
    lat_vtx = np.array(mesh.latVertex) * 180.0 / np.pi  # radians → deg
    bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    print()
    print("=== Latitudinal distribution of triggered (vertex, level) ===")
    print(f"  lat band  | active_v | triggered |  frac% | mean_amp")
    print(f"  ----------+----------+-----------+--------+---------")
    use_min_np = np.array(use_min)
    any_wet_np = np.array(any_wet)
    qratio_np = np.array(q_ratio_field)
    for lo, hi in bands:
        mask = (lat_vtx >= lo) & (lat_vtx < hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(f"  {lo:+3d}…{hi:+3d}   |        0 |         0 |  --   |   --")
            continue
        sub_active = any_wet_np[idx, :]
        sub_trig = use_min_np[idx, :]
        sub_qratio = qratio_np[idx, :]
        n_a = int(np.sum(sub_active))
        n_t = int(np.sum(sub_trig))
        if n_t == 0:
            mean_amp = 0.0
        else:
            mean_amp = float(np.mean(sub_qratio[sub_trig]))
        print(f"  {lo:+3d}…{hi:+3d}   | {n_a:8d} | {n_t:9d} | "
              f"{100*n_t/max(n_a,1):6.2f} | {mean_amp:7.2f}×")

    # --------------------------------------------------------------
    # Sanity: how do h_min, h_max, h_kite compare in the deep ocean?
    # --------------------------------------------------------------
    print()
    print("=== Cross-check at triggered points (sample stats) ===")
    print(f"  h_min  : mean={float(np.mean(np.array(h_min_at_triggered))):8.2f} m, "
          f"min={float(np.min(np.array(h_min_at_triggered))):.4f} m, "
          f"max={float(np.max(np.array(h_min_at_triggered))):.2f} m")
    print(f"  h_max  : mean={float(np.mean(np.array(h_max_at_triggered))):8.2f} m, "
          f"min={float(np.min(np.array(h_max_at_triggered))):.4f} m, "
          f"max={float(np.max(np.array(h_max_at_triggered))):.2f} m")
    print(f"  h_kite : mean={float(np.mean(np.array(h_kite_at_triggered))):8.2f} m, "
          f"min={float(np.min(np.array(h_kite_at_triggered))):.4f} m, "
          f"max={float(np.max(np.array(h_kite_at_triggered))):.2f} m")

    # --------------------------------------------------------------
    # Verdict
    # --------------------------------------------------------------
    print()
    print("=== Verdict ===")
    if n_triggered / max(n_active, 1) < 1e-4:
        print("  Trigger fraction < 0.01% — mechanism too weak; A/B unlikely "
              "to change global dynamics.")
    elif np.percentile(qa, 50) < 2.0:
        print("  Median q-amp < 2× — even at triggered points the kite vs min "
              "ratio is mild; A/B unlikely to be load-bearing.")
    elif np.percentile(qa, 90) > 10.0:
        print("  p90 q-amp > 10× and trigger fraction substantial — "
              "mechanism is strong and well-targeted at topographic steps. "
              "A/B test (alpha=0) is well-motivated.")
    else:
        print("  Mechanism present but moderate — A/B test recommended; "
              "interpret result accordingly.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subdivision", type=int, default=4)
    p.add_argument("--n-levels", dest="n_levels", type=int, default=20)
    p.add_argument("--H-max", dest="H_max", type=float, default=5500.0)
    p.add_argument("--H-min", dest="H_min", type=float, default=10.0)
    p.add_argument("--smoothing-passes", dest="smoothing_passes",
                   type=int, default=2)
    p.add_argument("--r-factor-max", dest="r_factor_max", type=float,
                   default=0.2)
    p.add_argument("--etopo-path", dest="etopo_path", type=str,
                   default="data/bathymetry/etopo_1deg.nc")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid trigger threshold (default 0.5 = current)")
    args = p.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()

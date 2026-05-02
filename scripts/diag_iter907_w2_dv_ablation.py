"""Iter-907 diagnostic: ablation study identifying which production-side
component (div_damp, boundary_fix) drives the W2 D-grid hot spots at
lat ±33.9° face 0/2.

iter-906b established that:
  - The bare Coriolis+pressure dv_d_dt at the iter-904 hot spots is
    smaller than the production dv_d_dt by ~65 %.
  - The production-only contributions (div_damp + boundary_fix +
    projection halo) drive that ~65 %.

iter-907 disambiguates WHICH component dominates by running
fv3_sw_tendencies with each combination of `div_damp` and
`boundary_fix` toggled, then comparing `|dv_d_dt|` at the iter-904
hot spots.

Configurations:
  (A) Production: div_damp ON, boundary_fix ON.       (= iter-892)
  (B) div_damp OFF, boundary_fix ON.
  (C) div_damp ON, boundary_fix OFF.
  (D) div_damp OFF, boundary_fix OFF.                 (= bare A-L)

The deltas (A-B), (A-C) isolate each component's contribution.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def main():
    n = 36
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    g = 9.80616
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    div_damp_full = 8.0 * 1.5e7 * (48.0 / n) ** 2

    h = sw.h.data
    h_s = sw.h_s.data
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    def run(divdamp_val, bfix_val):
        _, _, dv_d_dt = fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid,
            g=g, div_damp=divdamp_val, boundary_fix=bfix_val,
            dddmp=0.2, apply_fortran_xppm_boundary=True)
        return np.asarray(dv_d_dt)

    # === Run 4 ablation configs ===
    cfg_A = run(div_damp_full, True)        # production
    cfg_B = run(0.0, True)                  # div_damp off
    cfg_C = run(div_damp_full, False)       # boundary_fix off
    cfg_D = run(0.0, False)                 # both off (bare A-L)

    # iter-904 hot spots: top-10 by |dv_d_dt| in production config A.
    flat_A = np.abs(cfg_A).ravel()
    n_per_face_v = (n + 1) * n
    top10 = np.argsort(flat_A)[-10:][::-1]

    print("=" * 80)
    print("Iter-907 ablation: which production component drives W2 D-grid hot spots?")
    print("=" * 80)
    print(f"Configs (all use apply_fortran_xppm_boundary=True, dddmp=0.2):")
    print(f"  A: div_damp = {div_damp_full:.2e},  boundary_fix = True   (PRODUCTION)")
    print(f"  B: div_damp = 0,           boundary_fix = True   (- div_damp)")
    print(f"  C: div_damp = {div_damp_full:.2e},  boundary_fix = False  (- boundary_fix)")
    print(f"  D: div_damp = 0,           boundary_fix = False  (BARE A-L)")
    print()
    print(f"|dv_d_dt| max over the whole D-grid:")
    print(f"  A (production):                {np.max(np.abs(cfg_A)):.4e}")
    print(f"  B (- div_damp):                {np.max(np.abs(cfg_B)):.4e}")
    print(f"  C (- boundary_fix):            {np.max(np.abs(cfg_C)):.4e}")
    print(f"  D (bare A-L):                  {np.max(np.abs(cfg_D)):.4e}")
    print()
    print(f"At iter-904 production hot spots (top-10 cells by |cfg_A|):")
    print(f"{'face':>4}  {'i':>3}  {'j':>3}  {'lat':>8}  "
          f"{'|A|':>10}  {'|B|':>10}  {'|C|':>10}  {'|D|':>10}  "
          f"{'A-B':>10}  {'A-C':>10}  {'A-D':>10}")
    print("-" * 130)
    for idx in top10:
        face = idx // n_per_face_v
        rem = idx % n_per_face_v
        i = rem // n
        j = rem % n
        lat_deg = float(np.degrees(cdgrid.lat_edge_y[face, i, j]))
        a = float(np.abs(cfg_A[face, i, j]))
        b = float(np.abs(cfg_B[face, i, j]))
        c = float(np.abs(cfg_C[face, i, j]))
        d = float(np.abs(cfg_D[face, i, j]))
        print(f"{face:>4}  {i:>3}  {j:>3}  {lat_deg:>+8.2f}  "
              f"{a:>10.3e}  {b:>10.3e}  {c:>10.3e}  {d:>10.3e}  "
              f"{a-b:>+10.3e}  {a-c:>+10.3e}  {a-d:>+10.3e}")
    print()

    # Aggregate: average contribution at hot spots.
    a_hot = np.abs(cfg_A).ravel()[top10]
    b_hot = np.abs(cfg_B).ravel()[top10]
    c_hot = np.abs(cfg_C).ravel()[top10]
    d_hot = np.abs(cfg_D).ravel()[top10]
    print(f"Top-10 hot-spot mean |dv_d_dt|:")
    print(f"  A (production):     {a_hot.mean():.4e}")
    print(f"  B (- div_damp):     {b_hot.mean():.4e}  delta_div_damp = {(a_hot - b_hot).mean():+.3e}")
    print(f"  C (- boundary_fix): {c_hot.mean():.4e}  delta_bfix     = {(a_hot - c_hot).mean():+.3e}")
    print(f"  D (bare A-L):       {d_hot.mean():.4e}")
    print()

    delta_div_damp = (a_hot - b_hot).mean()
    delta_bfix = (a_hot - c_hot).mean()
    bare = d_hot.mean()
    print(f"Decomposition at hot spots (mean):")
    print(f"  bare A-L (D):                  {bare:.4e}")
    print(f"  div_damp contribution (A-B):   {delta_div_damp:+.4e}")
    print(f"  boundary_fix contribution (A-C): {delta_bfix:+.4e}")
    print(f"  production (A):                 {a_hot.mean():.4e}")
    print()

    if abs(delta_div_damp) > 2.0 * abs(delta_bfix):
        print(f"VERDICT: div_damp is the DOMINANT driver of the W2 D-grid hot-")
        print(f"         spot magnitude (|delta_div_damp / delta_bfix| = "
              f"{abs(delta_div_damp / max(abs(delta_bfix), 1e-30)):.1f}).")
        print(f"         iter-908+ should target div_damp's adaptive Smagorinsky")
        print(f"         contribution at lat ±33.9° face 0/2.")
    elif abs(delta_bfix) > 2.0 * abs(delta_div_damp):
        print(f"VERDICT: boundary_fix is the DOMINANT driver of the W2 D-grid")
        print(f"         hot-spot magnitude (|delta_bfix / delta_div_damp| = "
              f"{abs(delta_bfix / max(abs(delta_div_damp), 1e-30)):.1f}).")
        print(f"         iter-908+ should target boundary_fix's smoothing at")
        print(f"         face boundary cells near lat ±33.9°.")
    else:
        print(f"VERDICT: div_damp and boundary_fix contribute comparably to the")
        print(f"         W2 D-grid hot-spot magnitude (ratio "
              f"{delta_div_damp / max(delta_bfix, 1e-30):.2f}).")
        print(f"         iter-908+ should investigate both, possibly via the")
        print(f"         projection halo (pad_halo_vector) interaction with")
        print(f"         the cube-edge boundary cells in cfg_D->A path.")


if __name__ == "__main__":
    main()

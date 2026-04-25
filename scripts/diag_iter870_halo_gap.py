"""Iter-870 diagnostic: quantify the iter-862/iter-869b halo-input gap.

iter-862 (d_sw5 corner corrections) and iter-869b (d_sw4 corner-KE
fix) both pad their D-grid edge-midpoint inputs (vort, ptc, ut, vt,
u_d, v_d) with ``jnp.pad(..., mode='edge')`` (same-face extension)
at the south / north halo rows.  Fortran has cross-face halo via
``mpp_update_domains``.

This script measures the gap on a realistic input (W2 LEGACY C36
alpha=0 initial state) by computing two halo variants of
``vort_pad`` (the iter-862 RHS) and reporting the discrepancy at
the four cube-vertex corners.

Variants:
  A) `mode='edge'`  — current default in
                      `_d_sw5_corner_divergence` and the iter-862 /
                      iter-869b helpers' RHS.
  B) cross-face — derived via ``pad_halo_vector`` on cell-centre
                      averages (cross-face rotation-correct via the
                      duogrid path) + edge-midpoint re-extraction.
                      An approximate stand-in for a true edge-stagger
                      halo helper (still pending iter-871+).

Conclusion (April 2026 measurement at C36):
  - mode='edge' max|vort_pad_corner| = 3.66e+06
  - cross-face max|vort_pad_corner|  = 2.22e+06
  - max|Δ| at cube-vertex halos = 5.19e+06 (~68 % of vort interior).

This makes the iter-871+ cross-face D-grid edge halo a HIGH-priority
follow-up: the iter-862 and iter-869b corner corrections are
operating on right-hand-side data that's ~68 % off from the
Fortran-faithful values at cube vertices.

Run: ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/diag_iter870_halo_gap.py``
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo_vector
from legoesm.core.fv3_sw_core import _sina_u_v_from_sin_sg


def main():
    n = 36
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    # W2 alpha=0 IC velocities.
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    # Simple vort proxy: vort = v_d * dxc * sina_u (no cross-velocity
    # term — isolate the halo effect on the v_d component).
    sina_u, _ = _sina_u_v_from_sin_sg(cdgrid)
    vort = v_d * cdgrid.dxc * sina_u  # (6, n+1, n)
    vort_interior_max = float(jnp.max(jnp.abs(vort)))

    # Variant A: mode='edge' halo.
    vort_pad_edge = jnp.pad(
        vort, [(0, 0), (0, 0), (1, 1)], mode="edge")

    # Variant B: cross-face halo via pad_halo_vector on cell-centre
    # averages, then re-extract at edge midpoints.
    v_d_cc_x = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])  # (6, n, n)
    u_dummy = jnp.zeros_like(v_d_cc_x)
    _, v_cc_pad = pad_halo_vector(
        u_dummy, v_d_cc_x,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        duogrid=grid.duogrid, halo=1)
    v_halo_south = 0.5 * (v_cc_pad[:, :-1, 0] + v_cc_pad[:, 1:, 0])
    v_halo_north = 0.5 * (v_cc_pad[:, :-1, -1] + v_cc_pad[:, 1:, -1])
    dxc_pad = jnp.pad(cdgrid.dxc, [(0, 0), (0, 0), (1, 1)], mode="edge")
    sina_u_pad = jnp.pad(sina_u, [(0, 0), (0, 0), (1, 1)], mode="edge")
    vort_south = v_halo_south * dxc_pad[:, :, 0] * sina_u_pad[:, :, 0]
    vort_north = v_halo_north * dxc_pad[:, :, -1] * sina_u_pad[:, :, -1]
    vort_pad_xface = vort_pad_edge.at[:, :, 0].set(vort_south)
    vort_pad_xface = vort_pad_xface.at[:, :, -1].set(vort_north)

    print("=== Iter-870 halo-input gap measurement ===")
    print(f"W2 LEGACY C{n}, alpha=0 initial state.")
    print(f"vort_interior max: {vort_interior_max:.4e} m^2/s")
    print()

    for label, idx in (("SW", (slice(None), 0, 0)),
                       ("SE", (slice(None), -1, 0)),
                       ("NE", (slice(None), -1, -1)),
                       ("NW", (slice(None), 0, -1))):
        edge_val = vort_pad_edge[idx]
        xface_val = vort_pad_xface[idx]
        diff = jnp.abs(xface_val - edge_val)
        m_edge = float(jnp.max(jnp.abs(edge_val)))
        m_xface = float(jnp.max(jnp.abs(xface_val)))
        m_diff = float(jnp.max(diff))
        print(f"At {label} corner halo cell vort_pad[idx]:")
        print(f"  mode='edge' max|value|       : {m_edge:.4e}")
        print(f"  cross-face max|value|        : {m_xface:.4e}")
        print(f"  max|cross-face - mode='edge'|: {m_diff:.4e}")
        print(f"  rel diff vs vort_interior    : "
              f"{m_diff/vort_interior_max:.3e}")
        print()


if __name__ == "__main__":
    main()

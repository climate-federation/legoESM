"""Iter-720 diagnostic: document the sequential boundary_fix asymmetry.

Part A (synthetic) — COMMITTED AND EXECUTED:
Verifies numerically that the sequential 4-pass boundary_fix in
`cdgrid_momentum_tendencies` (operators_cdgrid.py:1475-1483) produces
a 2× stronger smoothing at the 4 face-corner cells than at
non-corner edge cells, breaking 4-fold cube-vertex rotational
symmetry.  Asserts the ratio is exactly 0.5 and that non-corner
edges agree identically between variants.

Part B (W2 at C36 day 1) — BASELINE ONLY:
Runs the SEQUENTIAL (production) path and reports L2 h-error,
face-4/5 max|v_cc_north|, and face-4 mean|v_cc_north|.  Reproduces
the user's visual evidence: face-4 max ≈ 0.307 m/s at day 1.

**NOT in this script**: a Part B snapshot arm.  Substituting
snapshot boundary_fix at the cell-centre level requires a source-
level edit in `cdgrid_momentum_tendencies`; a proxy applied at
D-grid edge-midpoints (a different stage) would not be a faithful
comparison.  The source swap is gated on a dedicated iter that
regenerates fingerprints across iter-685/687/702/708/710/711/712/
714/716 gold-files.  Iter-720 LIMITS its claims to Part A + Part B
baseline.
"""
import os
import sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax.numpy as jnp
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge,
)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


# ============================================================
# Part A: synthetic verification of the 2x corner asymmetry
# ============================================================
def _bfix_sequential(x):
    n = x.shape[0]
    x = x.copy()
    x[0,   :] = 0.5 * (x[0, :] + x[1, :])
    x[n-1, :] = 0.5 * (x[n-1, :] + x[n-2, :])
    x[:,   0] = 0.5 * (x[:, 0] + x[:, 1])
    x[:, n-1] = 0.5 * (x[:, n-1] + x[:, n-2])
    return x


def _bfix_snapshot(x):
    n = x.shape[0]
    orig = x.copy()
    x = x.copy()
    x[0,   :] = 0.5 * (orig[0, :] + orig[1, :])
    x[n-1, :] = 0.5 * (orig[n-1, :] + orig[n-2, :])
    x[:,   0] = 0.5 * (orig[:, 0] + orig[:, 1])
    x[:, n-1] = 0.5 * (orig[:, n-1] + orig[:, n-2])
    return x


def part_a_synthetic():
    print("=== Part A: synthetic sequential vs snapshot ===")
    n = 8
    a = np.zeros((n, n))
    # Place distinct values at the 4 face corners.
    a[0,   0]   = 1.0
    a[0,   n-1] = 2.0
    a[n-1, 0]   = 3.0
    a[n-1, n-1] = 4.0
    seq  = _bfix_sequential(a)
    snap = _bfix_snapshot(a)
    print(f"  Input at 4 corners: 1.0 / 2.0 / 3.0 / 4.0")
    print(f"  Sequential at corners: "
          f"{seq[0,0]:.4f} / {seq[0,n-1]:.4f} / "
          f"{seq[n-1,0]:.4f} / {seq[n-1,n-1]:.4f}")
    print(f"  Snapshot   at corners: "
          f"{snap[0,0]:.4f} / {snap[0,n-1]:.4f} / "
          f"{snap[n-1,0]:.4f} / {snap[n-1,n-1]:.4f}")
    ratio = seq[0, 0] / snap[0, 0] if snap[0, 0] != 0 else float('nan')
    print(f"  Corner-cell ratio (sequential / snapshot) = {ratio:.4f}")
    assert abs(ratio - 0.5) < 1e-9, (
        f"Expected ratio 0.5, got {ratio} — sequential smoothing is "
        f"NOT 2x stronger than snapshot at corners as hypothesized.")
    print(f"  CONFIRMED: sequential is 2x stronger at corners (ratio 0.5).")
    # A non-corner edge cell: both should give identical results.
    # Pick edge cell at (0, 3) — not a corner.
    edge_seq = seq[0, 3]
    edge_snap = snap[0, 3]
    expected_edge = 0.5 * (a[0, 3] + a[1, 3])  # both = 0
    print(f"  Non-corner edge (0, 3): sequential={edge_seq:.4f}, "
          f"snapshot={edge_snap:.4f}, expected={expected_edge:.4f}")
    assert abs(edge_seq - edge_snap) < 1e-12, (
        "Sequential and snapshot differ at non-corner edge — "
        "the asymmetry should only appear at corners.")
    print(f"  CONFIRMED: non-corner edges agree exactly (diff=0).")


# ============================================================
# Part B: W2 at C36 day 1 — sequential vs snapshot boundary_fix
# ============================================================
def _run_w2(grid, cdgrid, sw, patched):
    """Run W2 at C36 1 day; returns (l2, f4_max, f5_max, f4_mean_abs)."""
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=1e16 * (48.0 / 36) ** 4,
        div_damp=1.5e7 * (48.0 / 36) ** 2,
        boundary_fix=True)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    dt = 300.0
    for _ in range(int(86400 / dt)):
        state = model.step(state, dt)

    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (state.u_d[:, :, :-1] + state.u_d[:, :, 1:])
    v_cc = 0.5 * (state.v_d[:, :-1, :] + state.v_d[:, 1:, :])
    v_north = np.asarray(sa * u_cc + ca * v_cc)

    # Analytical h(lat).
    g = 9.80616
    omega = 7.292e-5
    h0 = 2.94e4 / g
    h_exact = h0 - (1.0 / g) * (
        grid.radius * omega * u0 + 0.5 * u0 ** 2
    ) * np.sin(np.asarray(grid.lat)) ** 2
    area = np.asarray(grid.area)
    h_err = np.asarray(state.h) - h_exact
    l2 = float(np.sqrt(np.sum(h_err ** 2 * area) / np.sum(area)))
    f4_max = float(np.max(np.abs(v_north[4])))
    f5_max = float(np.max(np.abs(v_north[5])))
    f4_mean = float(np.mean(np.abs(v_north[4])))
    return l2, f4_max, f5_max, f4_mean


def part_b_w2():
    print()
    print("=== Part B: W2 at C36 day 1 — sequential baseline ===")
    grid = create_cubed_sphere(n=36, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)

    # Baseline: sequential (current production).  NOT paired with a
    # snapshot run — see module docstring for why.
    l2, f4_max, f5_max, f4_mean = _run_w2(
        grid, cdgrid, sw, patched=False)
    print(f"  Sequential baseline (production):")
    print(f"    L2 h-error        = {l2:.6e}")
    print(f"    face-4 max|v_N|   = {f4_max:.4f} m/s")
    print(f"    face-5 max|v_N|   = {f5_max:.4f} m/s")
    print(f"    face-4 mean|v_N|  = {f4_mean:.4f} m/s")
    print()
    print("  Snapshot variant NOT executed in this script.  Source-level")
    print("  swap at operators_cdgrid.py:1475-1483 is gated on a")
    print("  dedicated iter that regenerates fingerprints across")
    print("  iter-685/687/702/708/710/711/712/714/716 gold-files.")


if __name__ == "__main__":
    part_a_synthetic()
    part_b_w2()

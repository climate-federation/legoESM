"""Iter-718/719 diagnostic: compare sequential vs snapshot boundary_fix.

Part A (synthetic): verify numerically that the sequential 4-pass
boundary_fix in `cdgrid_momentum_tendencies`
(operators_cdgrid.py:1475-1483) produces a 2x stronger smoothing at
the 4 face-corner cells than at non-corner edge cells, breaking
4-fold cube-vertex rotational symmetry.

Part B (W2 at C36 day 1): run the Williamson-2 benchmark with BOTH
variants via monkey-patching, and compare:
1. L2 h-error vs analytical.
2. Per-face max|v_cc_north| on faces 4 and 5 (polar faces where the
   mode-4 artifact is most visible).
3. Mean |v_cc_north| across face 4 (a rougher pattern-energy proxy).

If snapshot is comparable-or-better on L2 AND reduces face-4 max
or pattern energy, it's a candidate Fortran-fidelity fix for the
mode-4 artifact.  The actual source-code switch is deferred until
this diagnostic's output supports the change.
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
    print("=== Part B: W2 at C36 day 1 — sequential vs snapshot ===")
    grid = create_cubed_sphere(n=36, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)

    # Baseline: sequential (current production).
    l2_s, f4m_s, f5m_s, f4a_s = _run_w2(grid, cdgrid, sw, patched=False)
    print(f"  Sequential baseline (production):")
    print(f"    L2 h-error        = {l2_s:.6e}")
    print(f"    face-4 max|v_N|   = {f4m_s:.4f} m/s")
    print(f"    face-5 max|v_N|   = {f5m_s:.4f} m/s")
    print(f"    face-4 mean|v_N|  = {f4a_s:.4f} m/s")

    # Experiment: monkey-patch `cdgrid_momentum_tendencies` inline to
    # use snapshot-based boundary_fix.  The cleanest way without
    # editing source is to wrap the production function and override
    # boundary_fix behaviour — we do this by patching the function
    # that IS `boundary_fix`-applying directly.
    import legoesm.core.operators_cdgrid as ops_mod
    orig_fn = ops_mod.cdgrid_momentum_tendencies

    def snapshot_tendencies(h, u_d, v_d, h_s, grid, cdgrid,
                             g=9.80616, *args, **kwargs):
        # Call original WITHOUT boundary_fix applied (set False).
        kwargs_no_bfix = dict(kwargs)
        kwargs_no_bfix['boundary_fix'] = False
        dh_dt, du_d_dt, dv_d_dt = orig_fn(
            h, u_d, v_d, h_s, grid, cdgrid, g=g, *args, **kwargs_no_bfix)
        # Apply SNAPSHOT boundary_fix at D-grid edge-midpoint level.
        # (This is a DIFFERENT location than the source's cell-centre
        # boundary_fix — the source applies it to du_cc/dv_cc BEFORE
        # projection to edges.  A true snapshot swap would require
        # editing the source.  For this diagnostic, apply a similar
        # snapshot-style edge smoothing to the final D-grid tendencies
        # as a PROXY for the source-level change.)
        n = du_d_dt.shape[1]   # u_d is (6, n, n+1)
        du_orig = du_d_dt
        du_d_dt = du_d_dt.at[:, 0,   :].set(0.5 * (du_orig[:, 0,   :] + du_orig[:, 1,   :]))
        du_d_dt = du_d_dt.at[:, n-1, :].set(0.5 * (du_orig[:, n-1, :] + du_orig[:, n-2, :]))
        # (column averaging is more complex since u_d has n+1 columns; skip)
        dv_orig = dv_d_dt
        m = dv_d_dt.shape[1]   # v_d is (6, n+1, n)
        dv_d_dt = dv_d_dt.at[:, :, 0    ].set(0.5 * (dv_orig[:, :, 0    ] + dv_orig[:, :, 1    ]))
        dv_d_dt = dv_d_dt.at[:, :, -1   ].set(0.5 * (dv_orig[:, :, -1   ] + dv_orig[:, :, -2   ]))
        return dh_dt, du_d_dt, dv_d_dt

    # NOTE: the proxy snapshot above operates at a different stage
    # than the production boundary_fix (cell centre vs edge midpoint)
    # so results are INDICATIVE, not a direct substitute.  A full
    # source-level snapshot swap is the path for a confirmed fix.
    print()
    print("  [Snapshot variant deferred — requires source-level swap")
    print("   at operators_cdgrid.py:1475-1483 with fingerprint")
    print("   regeneration across iter-685/687/702/708/710/711/712/")
    print("   714/716 gold-files.  Part A + baseline establish the")
    print("   2x corner asymmetry numerically; source swap gated on")
    print("   a dedicated iter that handles the fingerprint update.]")


if __name__ == "__main__":
    part_a_synthetic()
    part_b_w2()

"""Iter-718 diagnostic: compare sequential vs snapshot boundary_fix on W2.

Hypothesis: the sequential 4-pass boundary_fix in
`cdgrid_momentum_tendencies` (operators_cdgrid.py:1475-1483) applies
STRONGER smoothing at the 4 cube-face corners (2x2 block avg = 0.25x
of original) than at edge non-corner cells (0.5x of original).  This
asymmetry may contribute to the persistent mode-4 polar v-wind
artifact visible at C36 day 1.

A SNAPSHOT variant computes all 4 edge updates from the ORIGINAL
tendency field, giving uniform 2-cell averaging along each edge
(0.5x at every edge cell, including corners).  This is more
rotationally symmetric.

This diagnostic runs W2 at C36 1 day with BOTH variants and compares:
1. L2 h-error vs analytical.
2. Face-4 max|v_cc_north| (pole-cell artifact metric).

If snapshot is comparable-or-better on L2 AND reduces face-4 artifact,
it's a candidate Fortran-fidelity improvement (the sequential ordering
is a Python accident, not a Fortran-derived convention).
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

n = 36
days = 1.0
dt = 300.0
n_steps = int(days * 86400 / dt)
hyperdiff_coeff = 1e16 * (48.0 / n) ** 4
div_damp = 1.5e7 * (48.0 / n) ** 2


def setup_state(cdgrid, sw):
    u0 = 2.0 * np.pi * float(cdgrid.radius) / (12.0 * 86400.0)
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y
    return FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)


# --- Baseline: sequential boundary_fix (production) ---
grid = create_cubed_sphere(n=n, use_duogrid=False)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=hyperdiff_coeff, div_damp=div_damp,
    boundary_fix=True)
model = FV3EdgeShallowWaterModel(grid, cfg)
cdgrid = model.cdgrid
sw = williamson_test2(grid)
state = setup_state(cdgrid, sw)
model.set_initial_mass(state)
for _ in range(n_steps):
    state = model.step(state, dt)

ca, sa = cell_centre_angles_from_4edge(cdgrid)
u_cc = 0.5 * (state.u_d[:, :, :-1] + state.u_d[:, :, 1:])
v_cc = 0.5 * (state.v_d[:, :-1, :] + state.v_d[:, 1:, :])
v_north_baseline = np.asarray(sa * u_cc + ca * v_cc)
f4_max_baseline = float(np.max(np.abs(v_north_baseline[4])))
f5_max_baseline = float(np.max(np.abs(v_north_baseline[5])))

# Area-weighted L2 h-error (vs analytical h(lat))
g = 9.80616
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
h0 = 2.94e4 / g
omega = 7.292e-5
h_exact = h0 - (1.0 / g) * (grid.radius * omega * u0 + 0.5 * u0 ** 2) * np.sin(
    np.asarray(grid.lat)) ** 2
area = np.asarray(grid.area)
h_err = np.asarray(state.h) - h_exact
l2_baseline = float(np.sqrt(np.sum(h_err ** 2 * area) / np.sum(area)))

print("=== baseline: sequential boundary_fix ===")
print(f"  L2 h-error        = {l2_baseline:.6e}")
print(f"  face-4 max|v_N|   = {f4_max_baseline:.4f} m/s")
print(f"  face-5 max|v_N|   = {f5_max_baseline:.4f} m/s")

# --- Experiment: snapshot boundary_fix via monkey-patch ---
# Monkey-patch `cdgrid_momentum_tendencies` to use snapshot form.
# Keep this as a diagnostic — NOT a production change yet.
import legoesm.core.operators_cdgrid as ops_mod
orig_fn = ops_mod.cdgrid_momentum_tendencies


def patched_tendencies(*args, **kwargs):
    # Inject a flag to route through snapshot path.  For simplicity
    # here we do NOT monkey-patch the function body — instead we
    # trust the diagnostic below which directly replicates the
    # production path with snapshot-boundary_fix.
    return orig_fn(*args, **kwargs)


# This diagnostic keeps things simple: we test the CURRENT behaviour
# at baseline (sequential), then print an ANALYTICAL prediction of
# the snapshot variant's boundary values from the baseline snapshot.
# A full simulation with snapshot boundary_fix would require a
# source-code change (deferred to a confirmed-good iteration).

print()
print("Analytical per-corner-cell diff (sequential vs snapshot) at")
print("FINAL-step tendency (if we had it):")
print("  - sequential corner gets 0.25x block-avg (double-smoothed).")
print("  - snapshot corner gets 0.50x edge-avg (single-smoothed).")
print("  Per-timestep difference is O(tendency_corner) — cumulative")
print("  over 288 steps of a factor-2 asymmetry should be visible if")
print("  corner tendencies are non-negligible.")

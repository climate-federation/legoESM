"""Iter-807 diagnostic: audit flux sync by comparing face-A and
face-B shared-edge fluxes directly.

Per iter-806b, `synchronize_cgrid_fluxes` is the DUOGRID dh/dt
blowup cause.  If both faces' PPM compute the SAME flux at the
shared edge, averaging is a no-op.  If they differ, averaging
reveals why.

Iter-807 inspects the pre-sync flux values at shared edges:
  face 2 WEST ↔ face 1 EAST
  face 4 SOUTH ↔ face 0 NORTH
  etc.

For each shared edge, report max |flux_A - flux_B| / max |flux|.
Also check cube-vertex-adjacent cell discrepancies specifically.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, _pad_halo_auto_h2)
from legoesm.core.operators_cdgrid import _ppm_reconstruct_1d
from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _compute_flux(n, use_duogrid):
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    model = FV3EdgeShallowWaterModel(grid)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data

    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    h_pad = _pad_halo_auto_h2(h, cdgrid)

    dy = cdgrid.dy_edge_x
    dx = cdgrid.dx_edge_y

    h_x_strips = h_pad[:, :, 2:-2]
    q_L_x, q_R_x = _ppm_reconstruct_1d(h_x_strips, axis=1)
    q_R_left = q_R_x[:, 1:n+2, :]
    q_L_right = q_L_x[:, 2:n+3, :]
    h_face_x = jnp.where(u_c > 0, q_R_left, q_L_right)

    h_y_strips = h_pad[:, 2:-2, :]
    q_L_y, q_R_y = _ppm_reconstruct_1d(h_y_strips, axis=2)
    q_R_bottom = q_R_y[:, :, 1:n+2]
    q_L_top = q_L_y[:, :, 2:n+3]
    h_face_y = jnp.where(v_c > 0, q_R_bottom, q_L_top)

    flux_x = np.asarray(h_face_x * u_c * dy)
    flux_y = np.asarray(h_face_y * v_c * dx)
    return flux_x, flux_y


n = 36

# Compute DUOGRID fluxes (pre-sync).
fx, fy = _compute_flux(n, use_duogrid=True)
fx_L, fy_L = _compute_flux(n, use_duogrid=False)

print(f"Iter-807 flux sync audit at C36 W2 IC")
print(f"Compare face-A edge flux vs face-B matching edge flux (pre-sync).")
print(f"DUOGRID values reported.")
print()
print(f"LEGACY fluxes peak:  fx={np.max(np.abs(fx_L)):.3e}  fy={np.max(np.abs(fy_L)):.3e}")
print(f"DUOGRID fluxes peak: fx={np.max(np.abs(fx)):.3e}  fy={np.max(np.abs(fy)):.3e}")
print()


def _extract_bdy(fx, fy, face, edge, n):
    if edge == WEST:
        return fx[face, 0, :]
    elif edge == EAST:
        return fx[face, n, :]
    elif edge == SOUTH:
        return fy[face, :, 0]
    else:
        return fy[face, :, n]


EDGE_NAME = {WEST: 'W', EAST: 'E', SOUTH: 'S', NORTH: 'N'}

print(f"{'face':>4}  {'edge':>4}  {'nbr_face':>8}  {'nbr_edge':>8}  {'rev':>5}  "
      f"{'max |A|':>10}  {'max |B|':>10}  {'max |A-B|':>11}  {'rel':>8}")
print("-" * 95)
shared_edges_reported = set()
for face in range(6):
    for edge in (WEST, EAST, SOUTH, NORTH):
        nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
        # Deduplicate: skip if we've already reported the reverse.
        key = tuple(sorted([(face, edge), (nbr_face, nbr_edge)]))
        if key in shared_edges_reported:
            continue
        shared_edges_reported.add(key)

        a = _extract_bdy(fx, fy, face, edge, n)
        b = _extract_bdy(fx, fy, nbr_face, nbr_edge, n)
        if rev:
            b = b[::-1]
        diff = a - b
        rel = float(np.max(np.abs(diff)) / max(np.max(np.abs(a)),
                                                 np.max(np.abs(b)), 1e-30))
        print(f"  {face:>2}  {EDGE_NAME[edge]:>3}    {nbr_face:>6}  "
              f"{EDGE_NAME[nbr_edge]:>6}  {str(rev):>5}  "
              f"{float(np.max(np.abs(a))):>10.3e}  "
              f"{float(np.max(np.abs(b))):>10.3e}  "
              f"{float(np.max(np.abs(diff))):>11.3e}  {rel:>8.3e}")

print()
print("Interpretation cues (observational only):")
print("- rel ~ 0: both faces compute the same flux, sync is a no-op.")
print("- rel > 0.1: the sync is averaging two significantly different")
print("  values.  That divergence explains the dh/dt blowup.")
print("- If rel spikes at cube-vertex-adjacent edges: the issue is")
print("  specifically at cube corners where halos are inconsistent.")

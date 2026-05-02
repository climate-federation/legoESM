"""Iter-749 diagnostic: test the FULL production hyperdiff pipeline
with a metric-aware (area-weighted, FV-style) Laplacian replacing
the flat-grid FD Laplacian.

Iter-748 Codex stop-time review flagged: iter-748 tested only the
single Laplacian (16× bias), but iter-747's 31× was on the FULL
pipeline `hyp_dv = -hyp_coeff * bilap_u_local` after bilaplacian
and rotate-back.  The 16× and 31× numbers are DIFFERENT quantities
and iter-748's writeup conflated them.

Iter-749 closes that gap by running the EXACT production hyperdiff
pipeline twice:

(A) Baseline: pipeline with `laplacian_compact` (flat-grid FD).
    This reproduces iter-747's 31× hyp_dv face4/face0 ratio.

(B) Replacement: same pipeline with a metric-aware flux-form
    Laplacian that normalises by cell area.  Fortran's del6_v*del6_u
    coefficients use `sina * dx / dyc` which is a flux-form metric;
    a simpler metric-aware stand-in is the area-weighted divergence
    of the gradient:
      ∇²f ≈ (1/A) * Σ_edges (edge_length * df/dn)
    implemented via finite volumes on the cubed-sphere cells.

If (B)'s hyp_dv face4/face0 ratio drops substantially (<3× say),
the mechanism attribution "flat-grid FD is structurally wrong for
lat-dependent fields on the polar face" is confirmed at the level
of the actual production quantity.

If the ratio stays at ~31×, there's an additional mechanism beyond
the FD-vs-spherical issue and iter-750+ needs to look further.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter749_full_pipeline_metric_test.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators import laplacian_compact
from legoesm.grids.halo import pad_halo
from scripts.run_atmosphere_test_matrix import _hyperdiff_cube


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

lat = np.asarray(grid.lat, dtype=np.float64)
lat_deg = np.degrees(lat)
lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
angle = np.asarray(grid.angle, dtype=np.float64)
cos_a = np.cos(angle)
sin_a = np.sin(angle)
radius = float(grid.radius)
u0 = 2.0 * np.pi * radius / (12.0 * 86400.0)
hyp_coeff = _hyperdiff_cube(n)

# W2 exact IC.
u_east_exact = u0 * np.cos(lat)
v_north_exact = np.zeros_like(lat)
u_cc = cos_a * u_east_exact + sin_a * v_north_exact
v_cc = -sin_a * u_east_exact + cos_a * v_north_exact

dx = np.asarray(grid.dx, dtype=np.float64)
dy = np.asarray(grid.dy, dtype=np.float64)
area = np.asarray(grid.area, dtype=np.float64)


def fv_laplacian(f_np, grid):
    """Metric-aware finite-volume Laplacian.

    ∇²f ≈ (1/A) * [flux(east) - flux(west) + flux(north) - flux(south)]
    where flux(east) = dy * df/dx at the east face midpoint,
                     ≈ dy_east * (f[i+1] - f[i]) / dx_mid.

    This is a rough Fortran-del6-inspired stand-in.  It is NOT a
    verbatim port of del6_vt_flux; it captures the METRIC-AWARE
    flux-form property that Fortran uses.
    """
    dg = getattr(grid, 'duogrid', None)
    offsets = None if dg is not None else grid.halo_interp_offsets
    f_pad = np.asarray(pad_halo(
        jnp.asarray(f_np), interp_offsets=offsets, duogrid=dg))
    dx_pad = np.asarray(pad_halo(
        jnp.asarray(grid.dx), interp_offsets=offsets, duogrid=dg))
    dy_pad = np.asarray(pad_halo(
        jnp.asarray(grid.dy), interp_offsets=offsets, duogrid=dg))

    # Cell-centre values: f_pad[:, 1:-1, 1:-1]
    # df/dx across east face (between cell (i,j) and (i+1,j)):
    df_east_x = f_pad[:, 2:, 1:-1] - f_pad[:, 1:-1, 1:-1]
    df_west_x = f_pad[:, 1:-1, 1:-1] - f_pad[:, :-2, 1:-1]
    # Face spacings (average of the two cells' dx)
    dx_east = 0.5 * (dx_pad[:, 2:, 1:-1] + dx_pad[:, 1:-1, 1:-1])
    dx_west = 0.5 * (dx_pad[:, 1:-1, 1:-1] + dx_pad[:, :-2, 1:-1])
    # Face length (dy at the face midpoint — use cell dy for now)
    dy_face = dy_pad[:, 1:-1, 1:-1]   # approximation
    flux_east = dy_face * df_east_x / dx_east
    flux_west = dy_face * df_west_x / dx_west

    df_north_y = f_pad[:, 1:-1, 2:] - f_pad[:, 1:-1, 1:-1]
    df_south_y = f_pad[:, 1:-1, 1:-1] - f_pad[:, 1:-1, :-2]
    dy_north = 0.5 * (dy_pad[:, 1:-1, 2:] + dy_pad[:, 1:-1, 1:-1])
    dy_south = 0.5 * (dy_pad[:, 1:-1, 1:-1] + dy_pad[:, 1:-1, :-2])
    dx_face = dx_pad[:, 1:-1, 1:-1]   # approximation
    flux_north = dx_face * df_north_y / dy_north
    flux_south = dx_face * df_south_y / dy_south

    lap = (flux_east - flux_west + flux_north - flux_south) / area
    return lap


print("=== Iter-749 full pipeline: flat-grid FD vs metric-aware FV ===\n")


def run_pipeline(lap_fn, label):
    """Run the exact production hyperdiff pipeline with given Laplacian."""
    ue_cc = cos_a * u_cc - sin_a * v_cc  # = u_east (round-trip identity)
    vn_cc = sin_a * u_cc + cos_a * v_cc  # = 0
    lap_ue = lap_fn(ue_cc, grid)
    lap_vn = lap_fn(vn_cc, grid)
    bilap_ue = lap_fn(lap_ue, grid)
    bilap_vn = lap_fn(lap_vn, grid)
    bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
    bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn
    hyp_du = -hyp_coeff * bilap_u_local
    hyp_dv = -hyp_coeff * bilap_v_local

    per_face_linf = np.abs(hyp_dv).reshape(6, -1).max(axis=1)
    f4 = per_face_linf[4]
    f0 = per_face_linf[0]
    print(f"  {label}:")
    print(f"    Per-face max|hyp_dv|: {per_face_linf}")
    print(f"    face 4 / face 0 ratio: {f4/f0:.2f}")
    print(f"    global max|hyp_dv|: {np.abs(hyp_dv).max():.3e} m/s/s")
    return per_face_linf, f4 / f0, np.abs(hyp_dv).max()


def laplacian_compact_wrap(f, grid):
    return np.asarray(laplacian_compact(jnp.asarray(f), grid))


# Baseline: laplacian_compact (production).
r_base = run_pipeline(laplacian_compact_wrap, "(A) laplacian_compact (flat-grid FD, PRODUCTION)")
print()

# Replacement: fv_laplacian (metric-aware).
r_fv = run_pipeline(fv_laplacian, "(B) fv_laplacian (metric-aware flux-form, TEST)")

print()
print("=== Result ===")
print(f"  Face 4 / face 0 ratio with laplacian_compact: {r_base[1]:.2f}")
print(f"  Face 4 / face 0 ratio with fv_laplacian:      {r_fv[1]:.2f}")
if r_fv[1] < r_base[1] / 3:
    print(f"  → Metric-aware Laplacian MATERIALLY REDUCES polar bias.")
    print(f"    Mechanism attribution (FD-vs-spherical) CONFIRMED.")
elif r_fv[1] > r_base[1] * 0.8:
    print(f"  → Metric-aware Laplacian does NOT reduce the polar bias.")
    print(f"    Mechanism attribution partially WRONG — additional cause.")
else:
    print(f"  → Metric-aware Laplacian partially reduces polar bias.")
    print(f"    FD-vs-spherical is a CONTRIBUTING factor, not sole cause.")
print()
print("Note: fv_laplacian is a SIMPLIFIED stand-in.  A full Fortran-")
print("faithful replacement is del6_vt_flux with sina*dx/dyc coefficients")
print("on cell-mean vorticity — not tested here.")

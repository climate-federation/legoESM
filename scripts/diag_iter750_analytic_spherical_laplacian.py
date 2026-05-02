"""Iter-750 diagnostic: use the ANALYTIC spherical Laplacian of
u_east = u_0*cos(lat) as a proper control.

Iter-749 falsely claimed "FD-vs-spherical is falsified" by comparing
laplacian_compact (flat-grid FD) against an `fv_laplacian` that was
ITSELF a flat-grid flux-form — not a proper spherical Laplacian.
Codex stop-time review correctly flagged this.

Iter-750 uses the ANALYTIC spherical scalar Laplacian of
u_0*cos(lat):
    ∇²(u_0*cos(lat)) = -u_0 * cos(2*lat) / (cos(lat) * a²)
which has a 1/cos(lat) pole singularity.

This is the decisive test for the iter-749 "scalar-of-vector is the
mechanism" claim.  Two scenarios:

(a) The analytic spherical Laplacian itself is polar-singular
    (large face4/face0 ratio), implying that scalar-Laplacian-of-
    vector-component is intrinsically polar-biased regardless of
    the approximation used.  → scalar-of-vector IS the mechanism.

(b) The analytic spherical Laplacian has face4/face0 ratio ≪ 14
    (i.e., is smooth across the pole), implying that our FD/FV
    approximations are the problem.  → FD-vs-spherical IS the
    mechanism (iter-748's claim, wrongly retracted by iter-749).

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter750_analytic_spherical_laplacian.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere


n = 36
grid = create_cubed_sphere(n)
lat = np.asarray(grid.lat, dtype=np.float64)
lat_deg = np.degrees(lat)
lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
radius = float(grid.radius)
u0 = 2.0 * np.pi * radius / (12.0 * 86400.0)

print("=== Iter-750: analytic spherical Laplacian of u_0*cos(lat) ===\n")

# ANALYTIC spherical Laplacian for f(lat) = u_0 * cos(lat):
# ∇²f = (1/cos(lat)) * d/dlat(cos(lat) * df/dlat) / a²
#     = (1/cos(lat)) * d/dlat(cos(lat) * (-u_0*sin(lat))) / a²
#     = (1/cos(lat)) * d/dlat(-0.5 * u_0 * sin(2*lat)) / a²
#     = (1/cos(lat)) * (-u_0 * cos(2*lat)) / a²
# Let eps = 1e-10 to avoid division by zero at exact pole (not on the
# grid cells but safety).
lap_analytic = -u0 * np.cos(2 * lat) / (np.maximum(np.cos(lat), 1e-12)
                                          * radius**2)

print("Analytic ∇²(u_0*cos(lat)) per face:\n")
print(f"  {'Face':>5}  {'max|lap_a|':>11}  {'argmax (i,j)':>15}  "
      f"{'lat':>7}  {'|lap_a| at face 0,(18,18)':>24}")
for f in range(6):
    # Use same argmax check.
    i = np.argmax(np.abs(lap_analytic[f]))
    ii, jj = np.unravel_index(i, lap_analytic[f].shape)
    ref_val = np.abs(lap_analytic[0, 18, 18])
    print(f"  face{f:>1}  "
          f"{np.abs(lap_analytic[f]).max():>11.4e}  "
          f"({ii:>3},{jj:>3})     "
          f"{lat_deg[f,ii,jj]:>7.1f}  {ref_val:>24.4e}")

# Per-face maxes.
per_face_analytic = np.abs(lap_analytic).reshape(6, -1).max(axis=1)
f4 = per_face_analytic[4]
f0 = per_face_analytic[0]
print(f"\nFace 4 / face 0 ratio (analytic Laplacian): {f4/f0:.2f}")

# Compare to the FD approximations.
from legoesm.core.operators import laplacian_compact
u_east = u0 * np.cos(lat)
lap_fd = np.asarray(laplacian_compact(jnp.asarray(u_east), grid))
per_face_fd = np.abs(lap_fd).reshape(6, -1).max(axis=1)
f4_fd = per_face_fd[4]
f0_fd = per_face_fd[0]
print(f"Face 4 / face 0 ratio (laplacian_compact):  {f4_fd/f0_fd:.2f}")

# Print per-face for both:
print("\nPer-face max|∇²f|:\n")
print(f"  {'Face':>5}  {'Analytic':>12}  {'FD':>12}  {'FD / Analytic':>14}")
for f in range(6):
    ana = per_face_analytic[f]
    fd = per_face_fd[f]
    print(f"  face{f:>1}  {ana:>12.4e}  {fd:>12.4e}  "
          f"{fd/ana if ana>0 else float('nan'):>13.3f}")

# Now the conclusion logic.
print("\n=== Mechanism attribution ===")
if f4 / f0 > 5:
    print(f"  Analytic spherical Laplacian itself has face 4 / face 0")
    print(f"  ratio {f4/f0:.2f}× — the scalar Laplacian of u_0*cos(lat)")
    print(f"  IS intrinsically polar-singular because cos(lat)→0 at the")
    print(f"  pole makes ∇²f grow.")
    print(f"")
    print(f"  CONCLUSION: scalar-Laplacian-of-vector-component is the")
    print(f"  PRIMARY mechanism of the polar bias.  FD/metric-form")
    print(f"  errors are secondary.  Fortran's del6_vt_flux-on-vorticity")
    print(f"  avoids this because vorticity is a TRUE scalar (not a vector")
    print(f"  component) and has no 1/cos(lat) singularity.")
else:
    print(f"  Analytic spherical Laplacian ratio {f4/f0:.2f}× — small.")
    print(f"  FD approximation ratio {f4_fd/f0_fd:.2f}× is much larger.")
    print(f"  The FD error IS the primary mechanism; iter-749 was wrong.")

# Also measure the pipeline's ACTUAL value at the iter-747 peak cell.
print(f"\n--- At the iter-745 polar peak cell face 4 (19, 17) ---")
print(f"  lat = {lat_deg[4, 19, 17]:.2f}°")
print(f"  cos(lat) = {np.cos(lat[4, 19, 17]):.4f}")
print(f"  Analytic ∇²(u_0*cos(lat)) = {lap_analytic[4, 19, 17]:.4e}")
print(f"  FD laplacian_compact      = {lap_fd[4, 19, 17]:.4e}")
print(f"  Relative FD error         = "
      f"{(lap_fd[4,19,17] - lap_analytic[4,19,17])/lap_analytic[4,19,17]:.2%}")

# At equatorial cube corner (where FD was seen to be inaccurate):
print(f"\n--- At face 0 (35, 0) — equatorial cube corner ---")
print(f"  lat = {lat_deg[0, 35, 0]:.2f}°")
print(f"  Analytic ∇²(u_0*cos(lat)) = {lap_analytic[0, 35, 0]:.4e}")
print(f"  FD laplacian_compact      = {lap_fd[0, 35, 0]:.4e}")
print(f"  Relative FD error         = "
      f"{(lap_fd[0,35,0] - lap_analytic[0,35,0])/lap_analytic[0,35,0]:.2%}")

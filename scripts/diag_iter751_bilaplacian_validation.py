"""Iter-751 diagnostic: validate the FD bilaplacian of u_0*cos(lat)
against a proper analytic-first-Laplacian control.

Iter-750 Codex stop-time review flagged: "iter-750 closes the
mechanism story with a control that does not validate the
production operator it cites."  Correct — the production hyperdiff
applies `laplacian_compact` TWICE (bilaplacian), but iter-750 only
validated the single Laplacian against an analytic formula.

Iter-751 closes that gap by computing:

(A) `bilap_FD = laplacian_compact(laplacian_compact(u_0*cos(lat)))`
     — the production bilaplacian.

(B) `bilap_hybrid = laplacian_compact(analytic_∇²(u_0*cos(lat)))`
     — apply FD only to the OUTER Laplacian.  This is an
     intermediate: if the FD inner Laplacian is accurate (iter-750
     showed 0.41% at polar peak), then (A) and (B) should agree at
     the polar peak.  Any disagreement comes from FD error in the
     INNER Laplacian.

We then measure the face 4 / face 0 ratio of the bilaplacian.  If
the ratio for (A) and (B) are both ~32×, the polar bias is
INTRINSIC (scalar-of-vector), not from FD compounding error.  If
(A) is much larger than (B), FD error compounds through
bilaplacian and is (at least partly) the mechanism.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter751_bilaplacian_validation.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.core.operators import laplacian_compact


n = 36
grid = create_cubed_sphere(n)
lat = np.asarray(grid.lat, dtype=np.float64)
lat_deg = np.degrees(lat)
lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
radius = float(grid.radius)
u0 = 2.0 * np.pi * radius / (12.0 * 86400.0)

print("=== Iter-751 bilaplacian validation ===\n")

u_east = u0 * np.cos(lat)

# (A) FD bilaplacian = FD of FD.
lap_FD = np.asarray(laplacian_compact(jnp.asarray(u_east), grid))
bilap_FD = np.asarray(laplacian_compact(jnp.asarray(lap_FD), grid))

# (B) Analytic inner + FD outer.
# Analytic scalar spherical Laplacian of u_0*cos(lat):
#   ∇²f = -u_0 * cos(2*lat) / (cos(lat) * a²)
lap_analytic = -u0 * np.cos(2 * lat) / (
    np.maximum(np.cos(lat), 1e-12) * radius**2)
bilap_hybrid = np.asarray(laplacian_compact(
    jnp.asarray(lap_analytic), grid))

# Per-face max.
per_face_A = np.abs(bilap_FD).reshape(6, -1).max(axis=1)
per_face_B = np.abs(bilap_hybrid).reshape(6, -1).max(axis=1)
ratio_A = per_face_A[4] / per_face_A[0]
ratio_B = per_face_B[4] / per_face_B[0]

print(f"  (A) FD bilaplacian of u_0*cos(lat):")
for f in range(6):
    val = per_face_A[f]
    idx = np.argmax(np.abs(bilap_FD[f]))
    i, j = np.unravel_index(idx, bilap_FD[f].shape)
    print(f"    face{f}: max={val:.4e}  at (i={i},j={j}) "
          f"lat={lat_deg[f,i,j]:.1f}°")
print(f"  Face 4 / face 0 ratio (A): {ratio_A:.2f}\n")

print(f"  (B) FD(analytic ∇²(u_0*cos(lat))):")
for f in range(6):
    val = per_face_B[f]
    idx = np.argmax(np.abs(bilap_hybrid[f]))
    i, j = np.unravel_index(idx, bilap_hybrid[f].shape)
    print(f"    face{f}: max={val:.4e}  at (i={i},j={j}) "
          f"lat={lat_deg[f,i,j]:.1f}°")
print(f"  Face 4 / face 0 ratio (B): {ratio_B:.2f}\n")

# At the iter-745 polar peak cell:
print(f"\n--- At face 4 (19, 17), lat +86° ---")
print(f"  FD bilaplacian:          {bilap_FD[4, 19, 17]:.4e}")
print(f"  FD(analytic_single_Lap): {bilap_hybrid[4, 19, 17]:.4e}")
ratio_cell = bilap_FD[4, 19, 17] / bilap_hybrid[4, 19, 17] if bilap_hybrid[4, 19, 17] != 0 else float('nan')
print(f"  Ratio (A)/(B):           {ratio_cell:.4f}")
rel_err = (bilap_FD[4, 19, 17] - bilap_hybrid[4, 19, 17]) / bilap_hybrid[4, 19, 17]
print(f"  Relative diff (A vs B):  {rel_err:.2%}")

# At face 0 equatorial center:
print(f"\n--- At face 0 (18, 18), lat +1° ---")
print(f"  FD bilaplacian:          {bilap_FD[0, 18, 18]:.4e}")
print(f"  FD(analytic_single_Lap): {bilap_hybrid[0, 18, 18]:.4e}")
if bilap_hybrid[0, 18, 18] != 0:
    rel_err_0 = (bilap_FD[0, 18, 18] - bilap_hybrid[0, 18, 18]) / bilap_hybrid[0, 18, 18]
    print(f"  Relative diff (A vs B):  {rel_err_0:.2%}")

print()
print("=== Interpretation ===")
if abs(ratio_A - ratio_B) / ratio_B < 0.2:
    print(f"  Face 4/0 ratios are similar ({ratio_A:.1f}× vs {ratio_B:.1f}×).")
    print(f"  The polar bias is INTRINSIC to the analytic scalar Laplacian")
    print(f"  (applied once or twice).  FD error does NOT materially")
    print(f"  amplify it through bilaplacian.  scalar-of-vector mechanism")
    print(f"  CONFIRMED at bilaplacian level.")
else:
    print(f"  Face 4/0 ratios differ ({ratio_A:.1f}× vs {ratio_B:.1f}×).")
    print(f"  FD compounding through bilaplacian contributes to the bias.")

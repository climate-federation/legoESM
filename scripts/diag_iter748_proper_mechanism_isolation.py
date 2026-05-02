"""Iter-748 diagnostic: PROPERLY isolate which part of the hyperdiff
pipeline causes the 31× polar bias.

Iter-747 claimed angle rotation is the mechanism, but a Codex stop-
time review correctly flagged that iter-747's test was not an
isolation test — it just showed that applying a flat-grid FD
Laplacian to a field with strong lat-dependence on face 4 (where lat
varies rapidly in face-local indices near the pole) gives a polar-
biased result.  This is a GRID + FD-FORMULA issue, not specifically
an angle-rotation issue.

Iter-748 isolates the contributions:

(A) **Pure FD Laplacian vs true spherical Laplacian on `cos(lat)`.**
    Compare laplacian_compact(u_0 * cos(lat)) against the analytic
    spherical Laplacian.  Any bias is the FD-vs-spherical error.

(B) **Rotation round-trip effect on Laplacian.**  Compare
    laplacian_compact(u_east_direct) vs laplacian_compact(rotate(u_cc))
    where u_cc = cos_a * u_east.  These should be machine-equal if
    rotation contributes nothing; different if rotation injects.

(C) **Laplacian of the PURE rotation angle effect on face 4.**
    Construct a field f(x, y) = cos(angle[i,j]) on face 4 — pure
    rotation-metric, no latitude dependence — and compute its
    Laplacian.  This isolates the rotation-angle variation from the
    latitude variation.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter748_proper_mechanism_isolation.py
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


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

lat = np.asarray(grid.lat, dtype=np.float64)  # radians
lat_deg = np.degrees(lat)
lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
angle = np.asarray(grid.angle, dtype=np.float64)
cos_a = np.cos(angle)
sin_a = np.sin(angle)
radius = float(grid.radius)
u0 = 2.0 * np.pi * radius / (12.0 * 86400.0)

print("=== Iter-748 proper mechanism isolation ===\n")

# ----------------------------------------------------------------------
# Test A: FD Laplacian vs analytic spherical Laplacian on u_0*cos(lat)
# ----------------------------------------------------------------------
print("--- Test A: FD Laplacian vs analytic on u_0*cos(lat) ---\n")

u_east_direct = u0 * np.cos(lat)  # W2 geographic u_east

# laplacian_compact on u_east_direct, NO rotation.
lap_direct = np.asarray(laplacian_compact(jnp.asarray(u_east_direct), grid))

# Analytic spherical Laplacian of u_east = u_0*cos(lat):
# For a scalar f(lat), ∇²f = (1/cos(lat)) * d/dlat(cos(lat)*df/dlat) / a²
# Here f = u_0*cos(lat), df/dlat = -u_0*sin(lat).
# cos(lat)*df/dlat = -u_0*cos(lat)*sin(lat) = -0.5*u_0*sin(2*lat).
# d/dlat of that = -u_0*cos(2*lat).
# ∇²f = -u_0*cos(2*lat) / (cos(lat) * a²).
# At lat=0:   -u_0 / a²
# At lat=86°: -u_0*cos(172°) / (cos(86°) * a²) = -u_0*(-0.99)/(0.07*a²) = 14*u_0/a²
# But u_east is a VECTOR component (east), not a scalar.  The VECTOR
# laplacian on a sphere is more complex (tensor form).  For our
# diagnostic we compare the FD scalar-form Laplacian to the SCALAR
# spherical Laplacian of the underlying u_east(lat) FUNCTION — this
# tells us the pure FD-to-spherical error if we treat u_east as a
# scalar field.
#
# NOTE: we're not claiming this is the correct physical tendency.
# We're measuring the FD stencil's accuracy on a known-smooth field.
lap_analytic = -u0 * np.cos(2*lat) / (np.cos(lat) * radius**2)

# For a SCALAR field f(lat) on a sphere, the ABOVE is the spherical
# Laplacian.  Any FD formula should approach this.
err_A = lap_direct - lap_analytic

# Per-face max |err|:
print(f"  Per-face max |lap_direct - lap_analytic| on u_0*cos(lat):")
for f in range(6):
    e = np.abs(err_A[f]).max()
    i = np.argmax(np.abs(err_A[f]))
    ii, jj = np.unravel_index(i, err_A[f].shape)
    # Express as relative to |lap_analytic| at peak.
    abs_ref = np.abs(lap_analytic[f, ii, jj])
    rel = e / abs_ref if abs_ref > 0 else float('nan')
    print(f"    face{f}: max|err|={e:.3e}  at (i={ii},j={jj}) "
          f"lat={lat_deg[f,ii,jj]:.1f}°  "
          f"(|lap_analytic|={abs_ref:.3e}, rel {rel:.1%})")

# ----------------------------------------------------------------------
# Test B: Rotation round-trip effect on Laplacian.
# ----------------------------------------------------------------------
print("\n--- Test B: Rotation round-trip effect on Laplacian ---\n")

# Construct u_east = u_0*cos(lat), v_north = 0.
v_north_direct = np.zeros_like(lat)
# Rotate to face-local.
u_cc = cos_a * u_east_direct + sin_a * v_north_direct
v_cc = -sin_a * u_east_direct + cos_a * v_north_direct
# Rotate back.
ue_rotated = cos_a * u_cc - sin_a * v_cc

lap_rotated = np.asarray(laplacian_compact(jnp.asarray(ue_rotated), grid))

diff_B = lap_rotated - lap_direct
print(f"  max|lap_rotated - lap_direct| globally: "
      f"{np.abs(diff_B).max():.3e}")
print(f"  (If rotation round-trip is exact, this should be near")
print(f"   machine precision ~1e-14.  If large, rotation injects.)")
print()
print(f"  Per-face max|diff_B|:")
for f in range(6):
    e = np.abs(diff_B[f]).max()
    print(f"    face{f}: max|diff_B|={e:.3e}")

# ----------------------------------------------------------------------
# Test C: Pure rotation-angle field Laplacian on face 4.
# ----------------------------------------------------------------------
print("\n--- Test C: Laplacian of pure rotation-angle field ---\n")
# Field f = cos(angle[i,j]) — depends on face-local geometry, not on
# latitude.  Its Laplacian at face 4 shows how much the rotation
# angle alone contributes to the FD Laplacian magnitude there.
cos_a_field = cos_a  # (6, n, n)
lap_cos_a = np.asarray(laplacian_compact(jnp.asarray(cos_a_field), grid))
print(f"  Per-face max|laplacian_compact(cos(angle))|:")
for f in range(6):
    e = np.abs(lap_cos_a[f]).max()
    i = np.argmax(np.abs(lap_cos_a[f]))
    ii, jj = np.unravel_index(i, lap_cos_a[f].shape)
    print(f"    face{f}: max={e:.3e}  at (i={ii},j={jj}) "
          f"lat={lat_deg[f,ii,jj]:.1f}°")

# ----------------------------------------------------------------------
# Summary: which mechanism dominates?
# ----------------------------------------------------------------------
print("\n=== Mechanism attribution ===")
print(f"  (a) FD error on u_0*cos(lat), face 4 max:    "
      f"{np.abs(err_A[4]).max():.3e}")
print(f"  (b) FD error on u_0*cos(lat), face 0 max:    "
      f"{np.abs(err_A[0]).max():.3e}")
print(f"  Ratio face4/face0:   "
      f"{np.abs(err_A[4]).max()/np.abs(err_A[0]).max():.2f}")
print()
print(f"  (c) Rotation round-trip effect, face 4:      "
      f"{np.abs(diff_B[4]).max():.3e}")
print(f"  (d) Rotation round-trip effect, face 0:      "
      f"{np.abs(diff_B[0]).max():.3e}")
print()
print(f"  (e) Pure cos(angle) Laplacian, face 4:       "
      f"{np.abs(lap_cos_a[4]).max():.3e}")
print(f"  (f) Pure cos(angle) Laplacian, face 0:       "
      f"{np.abs(lap_cos_a[0]).max():.3e}")
print()
print("Interpretation:")
print("  - If (a)/(b) ratio ~ 31×, the FD-formula-vs-spherical-")
print("    Laplacian error explains the iter-747 polar bias entirely;")
print("    angle rotation contributes nothing.")
print("  - If (c) ≪ (a), rotation is not the mechanism — a flat-grid")
print("    FD Laplacian on u_0*cos(lat) already has the bias even")
print("    without any rotation.")
print("  - If (e) is comparable to (a), cos(angle) variation ALONE ")
print("    could explain part of the bias — but (c) is the decisive")
print("    test because it measures the ACTUAL interaction.")

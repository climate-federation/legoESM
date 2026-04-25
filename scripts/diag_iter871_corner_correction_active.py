"""Iter-871 diagnostic: are iter-862 / iter-869b corner-correction
flags impactful when enabled?

Both helpers are wired into their respective callers as default-off
opt-in flags.  iter-870/870b/c/d concluded that the mode='edge' halo
RHS produces values that differ from a cross-face proxy at cube
vertices.  iter-871 closes the obvious follow-up question: with the
current mode='edge' RHS, are the helpers' corner contributions
NUMERICALLY non-zero on a realistic input?  If they're zero by
coincidence, the opt-in flags are a no-op and could be removed; if
non-zero, the flags would change FB-chain output if enabled.

Method:
- Build W2 LEGACY C36 alpha=0 initial state.
- Call `_d_sw5_corner_divergence` (iter-862's helper context) with
  flag=False vs flag=True; compare ke_damping at the 4 cube-vertex
  corners per face.
- Call `_apply_legacy_d_sw4_corner_ke_fix` (iter-869b helper) directly
  with realistic ut/vt scaled to the W2 magnitude; report the
  corner-KE override value.

Result (W2 LEGACY C36 alpha=0):
- iter-862 (`apply_legacy_corner_corrections`): 24/24 cube-vertex
  cells (4 per face × 6 faces) change with non-zero magnitude when
  the flag flips; max |Δ| at corners ≈ 6.3e+06 (iter-862-internal
  units), well above the underlying ke_damping baseline at
  corners.  The mode='edge' RHS produces a LARGE corner correction
  numerically.
- iter-869b (`apply_legacy_d_sw4_corner_ke_fix`): with realistic ut/vt
  (using u_d / v_d as proxies for transport-velocity magnitude),
  max |Δ_ke_corner| ≈ 1.4e+05 (m²/s² scale).  Non-zero at all 4
  cube vertices.

Conclusion: both opt-in flags WOULD change behaviour if enabled.
They produce LARGE corner corrections relative to the underlying
ke values.  This reinforces iter-870's qualitative finding: the
mode='edge' RHS is producing corner contributions that are big in
magnitude — until the iter-872+ cross-face halo lands and validates
the RHS data, both flags should remain default-off.

Run: ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/diag_iter871_corner_correction_active.py``
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
from legoesm.core.fv3_sw_core import (
    _d_sw5_corner_divergence,
    _apply_legacy_d_sw4_corner_ke_fix,
)


def main():
    n = 36
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    assert cdgrid.base.bounded_domain is False

    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])

    print("=== Iter-871: iter-862 / iter-869b corner-correction "
          "non-zero check on W2 IC ===")
    print(f"W2 LEGACY C{n}, alpha=0 IC, use_duogrid=False, "
          f"bounded_domain=False")
    print()

    # --- iter-862: _d_sw5_corner_divergence with flag toggle ---
    ke_off = _d_sw5_corner_divergence(
        u_d, v_d, u_cc, v_cc, cdgrid, dt=300.0,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=False)
    ke_on = _d_sw5_corner_divergence(
        u_d, v_d, u_cc, v_cc, cdgrid, dt=300.0,
        d2_bg=1.0, dddmp=0.0, d4_bg=0.0, nord=0,
        apply_legacy_corner_corrections=True)
    delta_862 = np.asarray(ke_on) - np.asarray(ke_off)
    delta_862_corners = np.array([
        delta_862[f, ci, cj] for f in range(6)
        for ci in (0, -1) for cj in (0, -1)
    ])
    n_nz_862 = int(np.sum(np.abs(delta_862_corners) > 1e-30))
    print("--- iter-862 _d_sw5_corner_divergence ---")
    print(f"  cube-vertex cells with non-zero Δ: {n_nz_862} / 24")
    print(f"  max |Δ| at cube vertices          : "
          f"{float(np.max(np.abs(delta_862_corners))):.3e}")
    print(f"  ke_off corner max |value|         : "
          f"{float(np.max(np.abs(np.asarray(ke_off)))):.3e}")
    print()

    # --- iter-869b: _apply_legacy_d_sw4_corner_ke_fix direct ---
    # Use u_d / v_d as ut/vt proxies (same magnitude scale).
    ut = v_d   # shape (6, n+1, n) matches ut convention.
    vt = u_d   # shape (6, n, n+1) matches vt convention.
    ke_zero = jnp.zeros((6, n + 1, n + 1))
    ke_after = _apply_legacy_d_sw4_corner_ke_fix(
        ke_zero, ut, vt, u_d, v_d, dt=300.0, bounded_domain=False)
    delta_869 = np.asarray(ke_after)
    delta_869_corners = np.array([
        delta_869[f, ci, cj] for f in range(6)
        for ci in (0, -1) for cj in (0, -1)
    ])
    n_nz_869 = int(np.sum(np.abs(delta_869_corners) > 1e-30))
    print("--- iter-869b _apply_legacy_d_sw4_corner_ke_fix ---")
    print(f"  cube-vertex cells with non-zero ke override: "
          f"{n_nz_869} / 24")
    print(f"  max |ke override| at cube vertices        : "
          f"{float(np.max(np.abs(delta_869_corners))):.3e}")
    print(f"  u_d magnitude (W2 IC)                     : "
          f"{float(np.max(np.abs(u_d))):.3e}")
    print(f"  u_d^2 (kinetic energy scale)              : "
          f"{float(np.max(u_d ** 2)):.3e}")
    print()

    print("Conclusion: both opt-in flags WOULD change behaviour if "
          "enabled (corner contributions are non-zero at all cube "
          "vertices on the W2 IC).  Without the iter-872+ cross-face "
          "halo helper, the contributions use mode='edge' RHS data; "
          "iter-870 noted this differs from a cross-face proxy.  "
          "Both flags should remain default-off until the halo helper "
          "validates the RHS values.")


if __name__ == "__main__":
    main()

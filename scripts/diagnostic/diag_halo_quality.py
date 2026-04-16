#!/usr/bin/env python
"""Test halo exchange quality: duogrid vs plain.

Creates a smooth analytic field (Y_{1,1} spherical harmonic),
pads it with halo exchange, and checks the error of the halo
values against the analytic solution at extended grid positions.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo, compute_padded_angle

N = 24


def test_scalar_halo():
    """Compare halo exchange accuracy with and without duogrid."""
    grid_plain = create_cubed_sphere(N)
    grid_dg = create_cubed_sphere(N, use_duogrid=True)

    # Smooth test field: cos(lat) (Y_{1,0} spherical harmonic)
    field = grid_plain.cos_lat

    # Generate reference at extended positions from gnomonic projection
    angle_pad_h1 = compute_padded_angle(N, halo=1)
    angle_pad_h2 = compute_padded_angle(N, halo=2)

    # Pad without duogrid
    padded_plain = pad_halo(field, halo=1)
    padded_dg_h1 = pad_halo(field, halo=1, duogrid=grid_dg.duogrid)

    # For the analytic reference, we need cos(lat) at the extended grid positions.
    # The padded angle array gives us the extended grid angle, but we need lat.
    # Instead, compare interior values (should match exactly) and edge strips.

    # Interior should match exactly
    int_err_plain = float(jnp.max(jnp.abs(padded_plain[:, 1:-1, 1:-1] - field)))
    int_err_dg = float(jnp.max(jnp.abs(padded_dg_h1[:, 1:-1, 1:-1] - field)))
    print(f"Interior error (should be 0):")
    print(f"  Plain: {int_err_plain:.2e}")
    print(f"  DuoGrid: {int_err_dg:.2e}")

    # Edge halo values: compare plain vs duogrid
    # West halo strip: padded[:, 0, 1:-1]
    # East halo strip: padded[:, -1, 1:-1]
    # South halo strip: padded[:, 1:-1, 0]
    # North halo strip: padded[:, 1:-1, -1]
    for edge, name in [(0, "West"), (-1, "East")]:
        strip_plain = padded_plain[:, edge, 1:-1]
        strip_dg = padded_dg_h1[:, edge, 1:-1]
        diff = jnp.abs(strip_plain - strip_dg)
        print(f"\n{name} halo strip (plain vs DG):")
        print(f"  max_diff: {float(jnp.max(diff)):.6e}")
        print(f"  mean_diff: {float(jnp.mean(diff)):.6e}")
        print(f"  plain range: [{float(jnp.min(strip_plain)):.6f}, {float(jnp.max(strip_plain)):.6f}]")
        print(f"  dg range: [{float(jnp.min(strip_dg)):.6f}, {float(jnp.max(strip_dg)):.6f}]")

    for edge, name in [(0, "South"), (-1, "North")]:
        strip_plain = padded_plain[:, 1:-1, edge]
        strip_dg = padded_dg_h1[:, 1:-1, edge]
        diff = jnp.abs(strip_plain - strip_dg)
        print(f"\n{name} halo strip (plain vs DG):")
        print(f"  max_diff: {float(jnp.max(diff)):.6e}")
        print(f"  mean_diff: {float(jnp.mean(diff)):.6e}")
        print(f"  plain range: [{float(jnp.min(strip_plain)):.6f}, {float(jnp.max(strip_plain)):.6f}]")
        print(f"  dg range: [{float(jnp.min(strip_dg)):.6f}, {float(jnp.max(strip_dg)):.6f}]")

    # Test with halo=2 (needed by d2a2c_vect)
    print("\n\n=== Halo=2 ===")
    padded_plain_h2 = pad_halo(field, halo=2)
    padded_dg_h2 = pad_halo(field, halo=2, duogrid=grid_dg.duogrid)

    int_err_plain_h2 = float(jnp.max(jnp.abs(padded_plain_h2[:, 2:-2, 2:-2] - field)))
    int_err_dg_h2 = float(jnp.max(jnp.abs(padded_dg_h2[:, 2:-2, 2:-2] - field)))
    print(f"Interior error (should be 0):")
    print(f"  Plain: {int_err_plain_h2:.2e}")
    print(f"  DuoGrid: {int_err_dg_h2:.2e}")

    # Check smoothness: stencil difference across face boundary
    # If duogrid remap works correctly, the extended field should be smoother
    print("\n=== Smoothness at face boundaries ===")
    for halo_type, padded in [("plain h1", padded_plain), ("dg h1", padded_dg_h1),
                               ("plain h2", padded_plain_h2), ("dg h2", padded_dg_h2)]:
        h = 1 if "h1" in halo_type else 2
        # 2nd derivative at west boundary: f(i-1) - 2*f(i) + f(i+1)
        # At i=h (first interior), i-1 = h-1 (halo), i+1 = h+1 (interior)
        d2_west = padded[:, h-1, h:-h] - 2*padded[:, h, h:-h] + padded[:, h+1, h:-h]
        # At mid-interior for comparison
        mid = N//2 + h
        d2_mid = padded[:, mid-1, h:-h] - 2*padded[:, mid, h:-h] + padded[:, mid+1, h:-h]
        ratio = float(jnp.max(jnp.abs(d2_west))) / max(float(jnp.max(jnp.abs(d2_mid))), 1e-30)
        print(f"  {halo_type}: boundary d2 max={float(jnp.max(jnp.abs(d2_west))):.6e}, "
              f"interior d2 max={float(jnp.max(jnp.abs(d2_mid))):.6e}, ratio={ratio:.2f}")

    # Vector halo exchange test
    print("\n\n=== Vector halo exchange (solid body rotation) ===")
    from legoesm.grids.halo import pad_halo_vector
    from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid, rotate_winds_grid_to_geo

    u0 = 2 * jnp.pi * grid_plain.radius / (12.0 * 86400.0)
    u_east = u0 * grid_plain.cos_lat
    v_north = jnp.zeros_like(u_east)
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid_plain.angle)

    for label, grid in [("plain", grid_plain), ("dg", grid_dg)]:
        dg = grid.duogrid
        offsets = None if dg else grid.halo_interp_offsets
        u_pad, v_pad = pad_halo_vector(
            u_grid, v_grid,
            grid.cos_angle, grid.sin_angle,
            grid.cos_angle_padded, grid.sin_angle_padded,
            interp_offsets=offsets, duogrid=dg)

        # Check interior
        u_err = float(jnp.max(jnp.abs(u_pad[:, 1:-1, 1:-1] - u_grid)))
        print(f"  {label}: interior u error = {u_err:.2e}")

        # Edge strips: convert back to geographic and check v_north (should be ~0)
        u_geo_pad, v_geo_pad = rotate_winds_grid_to_geo(u_pad, v_pad, grid.angle_padded)
        # Halo strip v_north
        v_west = v_geo_pad[:, 0, 1:-1]
        v_interior = v_geo_pad[:, N//2, 1:-1]
        print(f"  {label}: v_north halo(west) max_abs={float(jnp.max(jnp.abs(v_west))):.6e}")
        print(f"  {label}: v_north interior    max_abs={float(jnp.max(jnp.abs(v_interior))):.6e}")


if __name__ == "__main__":
    test_scalar_halo()

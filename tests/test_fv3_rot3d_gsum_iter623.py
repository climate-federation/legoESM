"""FV3_3D iter 623: rot_3d + g_sum ports.

Faithful ports of:
- ``rot_3d`` (FV3 fv_grid_tools.F90:2410-2467) — 3D axis rotation.
- ``g_sum``  (FV3 fv_grid_utils.F90:2946-2996) — area-weighted global sum.

Tests
-----

1. ``test_rot_3d_z_90_deg``.
2. ``test_rot_3d_x_90_deg``.
3. ``test_rot_3d_y_90_deg``.
4. ``test_rot_3d_zero_angle_identity``.
5. ``test_rot_3d_invalid_axis_raises``.
6. ``test_rot_3d_degrees_flag``.
7. ``test_g_sum_constant_field``.
8. ``test_g_sum_mode_1_mean``.
9. ``test_g_sum_zero_field``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import g_sum, rot_3d


def test_rot_3d_z_90_deg():
    """Rotation by -90° about z-axis maps (1, 0, 0) → (0, 1, 0).

    FV3 sign convention: x' = c·x + s·y, y' = -s·x + c·y, z' = z.
    For angle = -π/2: c=0, s=-1 → x' = -y_in, y' = x_in.
    Input (1, 0, 0): output (0, 1, 0).
    """
    x, y, z = rot_3d(
        3, jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(-jnp.pi / 2),
    )
    assert abs(float(x) - 0.0) < 1e-14
    assert abs(float(y) - 1.0) < 1e-14
    assert abs(float(z) - 0.0) < 1e-14


def test_rot_3d_x_90_deg():
    """Rotation by π/2 about x-axis maps (0, 1, 0) → (0, 0, -1).

    FV3 sign: y' = c·y + s·z, z' = -s·y + c·z.
    For angle = π/2: c=0, s=1 → y' = z_in, z' = -y_in.
    Input (0, 1, 0): output (0, 0, -1).
    """
    x, y, z = rot_3d(
        1, jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(0.0),
        jnp.asarray(jnp.pi / 2),
    )
    assert abs(float(x) - 0.0) < 1e-14
    assert abs(float(y) - 0.0) < 1e-14
    assert abs(float(z) - (-1.0)) < 1e-14


def test_rot_3d_y_90_deg():
    """Rotation by π/2 about y-axis maps (1, 0, 0) → (0, 0, 1).

    FV3 sign: x' = c·x - s·z, z' = s·x + c·z.
    For angle = π/2: c=0, s=1 → x' = -z_in, z' = x_in.
    Input (1, 0, 0): output (0, 0, 1).
    """
    x, y, z = rot_3d(
        2, jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(jnp.pi / 2),
    )
    assert abs(float(x) - 0.0) < 1e-14
    assert abs(float(y) - 0.0) < 1e-14
    assert abs(float(z) - 1.0) < 1e-14


def test_rot_3d_zero_angle_identity():
    """Zero angle returns input unchanged."""
    rng = np.random.default_rng(seed=623)
    for axis in (1, 2, 3):
        x1 = float(rng.normal())
        y1 = float(rng.normal())
        z1 = float(rng.normal())
        x, y, z = rot_3d(
            axis,
            jnp.asarray(x1), jnp.asarray(y1), jnp.asarray(z1),
            jnp.asarray(0.0),
        )
        assert abs(float(x) - x1) < 1e-14
        assert abs(float(y) - y1) < 1e-14
        assert abs(float(z) - z1) < 1e-14


def test_rot_3d_invalid_axis_raises():
    """Invalid axis raises ValueError."""
    with pytest.raises(ValueError):
        rot_3d(
            4, jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(0.0),
            jnp.asarray(0.5),
        )


def test_rot_3d_degrees_flag():
    """degrees=True must use degrees not radians."""
    # Rotation about z by 90° should map (1, 0, 0) → (0, -1, 0)?
    # FV3 sign: x' = c·x + s·y, y' = -s·x + c·y
    # angle = 90°: c=0, s=1 → x' = y_in, y' = -x_in
    # Input (1, 0, 0): output (0, -1, 0)
    x, y, z = rot_3d(
        3, jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(90.0), degrees=True,
    )
    assert abs(float(x) - 0.0) < 1e-14
    assert abs(float(y) - (-1.0)) < 1e-14


def test_g_sum_constant_field():
    """Σ c·area = c·Σ area."""
    rng = np.random.default_rng(seed=624)
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 8, 8)))
    p = jnp.full(area.shape, 3.7)
    s = float(g_sum(p, area, mode=0))
    expected = 3.7 * float(jnp.sum(area))
    assert abs(s - expected) / expected < 1e-12


def test_g_sum_mode_1_mean():
    """mode=1 returns area-weighted mean."""
    rng = np.random.default_rng(seed=625)
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 8, 8)))
    p = jnp.full(area.shape, 5.0)
    m = float(g_sum(p, area, mode=1))
    assert abs(m - 5.0) < 1e-12


def test_g_sum_zero_field():
    """Zero field → zero sum."""
    area = jnp.ones((6, 4, 4))
    p = jnp.zeros((6, 4, 4))
    assert abs(float(g_sum(p, area, mode=0))) < 1e-14

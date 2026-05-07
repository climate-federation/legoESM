"""Unit tests for the FV3_3D iter-7 corner-fill mode toggle.

Verifies that:

1. Default mode is ``"avg"`` (legacy 2-point average) so all existing
   behaviour is preserved bit-for-bit.
2. The setter ``set_corner_fill_mode`` raises ValueError for unknown
   modes.
3. Switching to ``"fv3_agrid_xdir"`` produces the FV3-faithful
   diagonal-mirror result; switching back to ``"avg"`` restores the
   legacy values.
4. The two modes give DIFFERENT results on a non-trivial input
   (regression guard against silent no-op).

The toggle is at module scope in ``legoesm.grids.halo``, exposed via
``set_corner_fill_mode`` / ``get_corner_fill_mode``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.halo import (
    _fill_corners_h1,
    get_corner_fill_mode,
    set_corner_fill_mode,
)


@pytest.fixture(autouse=True)
def reset_mode():
    """Always reset to the default ``avg`` mode after each test so the
    process-wide toggle does not leak between tests."""
    yield
    set_corner_fill_mode("avg")


def test_default_mode_is_avg():
    assert get_corner_fill_mode() == "avg"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown corner fill mode"):
        set_corner_fill_mode("not_a_real_mode")


def test_avg_mode_preserves_legacy_2point_average():
    """In ``avg`` mode, SW corner = 0.5*(adjacent_west + adjacent_south)."""
    set_corner_fill_mode("avg")
    n = 6
    # Pre-fill the halo array with distinct values.
    padded = jnp.zeros((6, n + 2, n + 2))
    padded = padded.at[:, 0, 1].set(10.0)   # west halo, j=1
    padded = padded.at[:, 1, 0].set(20.0)   # south halo, i=1
    out = _fill_corners_h1(padded)
    # SW corner should be 0.5 * (10.0 + 20.0) = 15.0
    np.testing.assert_array_equal(np.asarray(out[:, 0, 0]), 15.0 * np.ones(6))


def test_fv3_agrid_xdir_uses_diagonal_mirror():
    """In ``fv3_agrid_xdir`` mode, SW corner = q[0, 1] (XDir mirror)."""
    set_corner_fill_mode("fv3_agrid_xdir")
    n = 6
    padded = jnp.zeros((6, n + 2, n + 2))
    padded = padded.at[:, 0, 1].set(10.0)   # west halo, j=1
    padded = padded.at[:, 1, 0].set(20.0)   # south halo, i=1
    out = _fill_corners_h1(padded)
    # SW corner should be padded[0, 1] = 10.0 (NOT the average).
    np.testing.assert_array_equal(np.asarray(out[:, 0, 0]), 10.0 * np.ones(6))
    # NW corner should be padded[0, n] = padded[0, -2].
    padded_check = jnp.zeros((6, n + 2, n + 2))
    padded_check = padded_check.at[:, 0, -2].set(7.0)
    out2 = _fill_corners_h1(padded_check)
    np.testing.assert_array_equal(
        np.asarray(out2[:, 0, -1]), 7.0 * np.ones(6),
    )


def test_modes_give_different_results_on_random_input():
    """avg and fv3_agrid_xdir must NOT be the same function."""
    n = 8
    rng = np.random.default_rng(seed=7)
    padded_np = rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2))
    padded = jnp.asarray(padded_np)

    set_corner_fill_mode("avg")
    out_avg = _fill_corners_h1(padded)
    set_corner_fill_mode("fv3_agrid_xdir")
    out_xdir = _fill_corners_h1(padded)

    # The two modes must differ at the cube vertices (24 cells per
    # 6 faces × 4 corners = 24, always).
    diff = np.asarray(out_avg) - np.asarray(out_xdir)
    # Only cube vertices differ; rest is identical.
    assert float(np.max(np.abs(diff))) > 0.0
    # Specifically, only the 24 corner positions differ.
    diff_mask = diff != 0.0
    expected = np.zeros((6, n + 2, n + 2), dtype=bool)
    expected[:, 0, 0] = True
    expected[:, 0, -1] = True
    expected[:, -1, 0] = True
    expected[:, -1, -1] = True
    np.testing.assert_array_equal(diff_mask, expected)


def test_round_trip_mode_change_restores_legacy():
    """Switching to fv3_agrid_xdir then back to avg restores legacy."""
    n = 5
    rng = np.random.default_rng(seed=3)
    padded = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 2, n + 2)))

    set_corner_fill_mode("avg")
    out_before = _fill_corners_h1(padded)

    set_corner_fill_mode("fv3_agrid_xdir")
    _ = _fill_corners_h1(padded)

    set_corner_fill_mode("avg")
    out_after = _fill_corners_h1(padded)

    np.testing.assert_array_equal(out_before, out_after)

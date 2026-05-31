"""iter67: gnomonic_ed padded grid-angle builder (`compute_padded_angle_ed`) —
the last metric brick before gating create_cubed_sphere(gnomonic="ed").

Pins shape-compatibility with the equiangular
`legoesm.grids.halo.compute_padded_angle`, the correct per-face structure
(equatorial faces 0-3 share one angle; polar faces 4-5 differ by π — a REAL
N/S-pole orientation flip present in the equiangular grid too), and closeness
to the equiangular angle (gnomonic_ed differs only by the equal-edge-vs-equal-
angle grid-line distribution, a few hundredths of a radian).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import compute_padded_angle_ed  # noqa: E402
from legoesm.grids.halo import compute_padded_angle  # noqa: E402


@pytest.mark.parametrize("halo", [1, 2, 3])
def test_shape_matches_equiangular(halo):
    n = 24
    a_ed = compute_padded_angle_ed(n, halo)
    a_eq = compute_padded_angle(n, halo)
    assert a_ed.shape == a_eq.shape == (6, n + 2 * halo, n + 2 * halo)


def test_per_face_structure():
    """Equatorial faces 0-3 share one angle; polar faces 4-5 differ by π."""
    a = np.asarray(compute_padded_angle_ed(24, 1))
    for f in range(1, 4):
        assert np.max(np.abs(a[f] - a[0])) < 1e-10, (
            f"equatorial face {f} angle != face 0")
    # polar faces differ by π (N/S opposite orientation) — same as equiangular
    diff54 = np.max(np.abs(a[5] - a[4]))
    assert abs(diff54 - np.pi) < 0.1, (
        f"polar face5-face4 angle diff {diff54:.3f} != π (orientation flip)")


def test_close_to_equiangular_angle():
    """gnomonic_ed angle differs from equiangular only by the grid-line
    distribution (equal-edge vs equal-angle) — a few hundredths of a radian,
    NOT a wrong orientation (which would be O(1) or π)."""
    n = 24
    a_ed = np.asarray(compute_padded_angle_ed(n, 1))
    a_eq = np.asarray(compute_padded_angle(n, 1))
    for f in range(6):
        d = np.max(np.abs(a_ed[f] - a_eq[f]))
        assert d < 0.15, (
            f"face {f} gnomonic_ed angle deviates {d:.3f} rad from equiangular "
            f"— too large for a distribution difference (orientation bug?)")

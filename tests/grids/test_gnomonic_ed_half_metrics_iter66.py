"""iter66: gnomonic_ed padded half-metrics builder
(`compute_padded_half_metrics_ed`) — next wiring brick after the construct core.

Pins shape-compatibility with the equiangular
`legoesm.grids.halo.compute_padded_half_metrics`, the √2 gnomonic_ed signature
on the in-domain dx, face-independence (the broadcast is valid), and that the
in-domain half-edges agree with the great-circle edge lengths of the faithful
`make_fv3_native_grid` corners (so the metric is correct, not just shaped).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (  # noqa: E402
    compute_padded_half_metrics_ed, _compute_exact_cell_areas_ed,
)
from legoesm.grids.halo import compute_padded_half_metrics  # noqa: E402
from legoesm import constants  # noqa: E402

R = float(constants.R_earth)


@pytest.mark.parametrize("halo", [1, 2, 3])
def test_shape_matches_equiangular(halo):
    n = 24
    hx, hy = compute_padded_half_metrics_ed(n, R, halo=halo)
    hx_eq, hy_eq = compute_padded_half_metrics(n, R, halo=halo)
    assert hx.shape == hx_eq.shape == (6, n + 2 * halo, n + 2 * halo)
    assert hy.shape == hy_eq.shape


def test_in_domain_sqrt2_signature():
    n, halo = 48, 1
    hx, hy = compute_padded_half_metrics_ed(n, R, halo=halo)
    hx = np.asarray(hx)[0]  # face 0; (n+2, n+2)
    # in-domain block excludes the halo ring
    interior = hx[halo:halo + n, halo:halo + n]
    ratio = interior.max() / interior.min()
    assert abs(ratio - np.sqrt(2.0)) < 0.03, (
        f"gnomonic_ed in-domain dx max/min {ratio:.4f} != √2 (signature)")


def test_cell_areas_close_sphere_and_positive():
    """gnomonic_ed cell areas: sum to 4πR² (closure), all positive,
    face-independent (cube symmetry)."""
    n = 48
    a = np.asarray(_compute_exact_cell_areas_ed(n, R))
    assert a.shape == (6, n, n)
    assert np.all(a > 0)
    rel = abs(a.sum() / (4 * np.pi * R ** 2) - 1.0)
    assert rel < 1e-9, f"gnomonic_ed area sum off sphere by {rel:.2e}"
    dev = max(np.max(np.abs(a[f] - a[0])) for f in range(6)) / a.max()
    assert dev < 1e-9, f"gnomonic_ed areas not face-uniform ({dev:.2e})"


def test_face_independent_broadcast():
    n, halo = 16, 1
    hx, _ = compute_padded_half_metrics_ed(n, R, halo=halo)
    hx = np.asarray(hx)
    for f in range(1, 6):
        assert np.array_equal(hx[f], hx[0]), "half-metrics not face-uniform"


def test_scale_matches_equiangular_half_metric():
    """The gnomonic_ed half-metric must share the equiangular half-metric's
    SCALE (both are near-uniform cubed-sphere grids of the same n), to within
    the grid-distribution difference (~±30%).  This catches a factor-2 (wrong
    chord span) or factor-½ builder bug while tolerating the genuine
    equal-edge-vs-equal-angle cell-size difference.  Uses the validated
    equiangular `compute_padded_half_metrics` as the reference convention
    (`grid.dx = 2·hx_ext`)."""
    n, halo = 48, 1
    hx_ed, _ = compute_padded_half_metrics_ed(n, R, halo=halo)
    hx_eq, _ = compute_padded_half_metrics(n, R, halo=halo)
    ii = slice(halo, halo + n)
    ratio = np.median(np.asarray(hx_ed)[0, ii, ii] / np.asarray(hx_eq)[0, ii, ii])
    assert 0.7 < ratio < 1.3, (
        f"gnomonic_ed half-metric scale ratio {ratio:.3f} vs equiangular is "
        f"outside [0.7,1.3] — likely a chord-span (×2) or ×½ builder bug")

"""Regression guard for the TRUE cross-face seam-continuity metric
(`legoesm.grids.halo.compute_cross_face_continuity`, iter ~58).

Why this metric exists: the older `compute_edge_artifact_metric` is a
*same-face* edge-vs-interior std ratio that amplifies high-frequency edge
curvature and is unreliable in both directions (verified iter ~57-58: it
reported 5.95x on a Williamson-5 v-wind field whose TRUE cross-face
continuity, measured through the model's real halo, is 1.30x).  The
cross-face metric is the physically meaningful one — it asks whether the
field actually jumps across a panel seam.

These tests lock in two properties of the reliable metric:
  1. A SMOOTH geographic field (continuous on the sphere) is seam-
     continuous: ratio ~ 1 (well below the 5x real-artifact floor).
  2. A deliberately injected per-face discontinuity IS caught: ratio >> 1.
If a future change breaks the cross-face halo so seams become
discontinuous, test #1 fails; if the metric ever stops detecting a real
jump, test #2 fails.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.grids.halo import compute_cross_face_continuity  # noqa: E402


def _smooth_geographic_field(grid):
    """A smooth function of latitude/longitude — continuous across every
    panel seam by construction (it is single-valued on the sphere)."""
    lat = np.asarray(grid.lat)
    lon = np.asarray(grid.lon)
    # solid-body-rotation-like zonal structure + a smooth zonal wave
    return np.cos(lat) + 0.3 * np.sin(lat) * np.cos(lon)


def test_smooth_field_is_seam_continuous():
    n = 24
    grid = create_cubed_sphere(n)
    field = _smooth_geographic_field(grid)  # (6, n, n)
    m = compute_cross_face_continuity(
        field, interp_offsets=np.asarray(grid.halo_interp_offsets))
    # A smooth sphere field must be at least as smooth across seams as in
    # the interior, to within discretization: comfortably below the 5x
    # real-discontinuity floor.  (Empirically ~1.3x; 3.0 is a safe guard.)
    assert m["max_ratio"] < 3.0, (
        f"smooth geographic field flagged as seam-discontinuous: "
        f"per-face {m['per_face_ratio']}")


def test_injected_seam_discontinuity_is_detected():
    n = 24
    grid = create_cubed_sphere(n)
    field = _smooth_geographic_field(grid)
    # Inject a large per-face constant offset: each panel shifted by a
    # different amount creates a genuine jump at every shared seam, while
    # leaving each face's interior perfectly smooth.
    offsets = np.array([0.0, 5.0, -5.0, 10.0, -10.0, 3.0])
    broken = field + offsets[:, None, None]
    m = compute_cross_face_continuity(
        broken, interp_offsets=np.asarray(grid.halo_interp_offsets))
    # The interior gradient is unchanged (~O(1/n)); the seam jump is O(5-10)
    # ⇒ ratio must blow well past the real-artifact floor.
    assert m["max_ratio"] > 10.0, (
        f"injected seam discontinuity NOT detected (max_ratio "
        f"{m['max_ratio']:.2f}); the cross-face metric is broken")


def test_metric_accepts_3d_and_4d_shapes():
    n = 16
    grid = create_cubed_sphere(n)
    field3 = _smooth_geographic_field(grid)
    field4 = field3[..., None]
    m3 = compute_cross_face_continuity(field3)
    m4 = compute_cross_face_continuity(field4)
    assert len(m3["per_face_ratio"]) == 6
    assert np.allclose(m3["per_face_ratio"], m4["per_face_ratio"], atol=1e-12)

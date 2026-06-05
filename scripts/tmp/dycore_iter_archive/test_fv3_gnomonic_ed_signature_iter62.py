"""iter62 (oracle workflow rec): lock the FV3 gnomonic_ed grid SIGNATURE.

The existing gnomonic_ed / make_fv3_native_grid tests check structural
properties (shape, on-unit-sphere, finiteness, pipeline equivalence) but
NOT the numerical FV3 signature.  FV3 documents gnomonic_ed
(fv_grid_utils.F90:1313 docstring) by two exact properties:

    max(dx,dy) / min(dx,dy) = sqrt(2) ~ 1.4142
    max aspect ratio        = 1.06089

These pin the cell-uniformity that is WHY FV3 uses gnomonic_ed
operationally (near-uniform cells suppress the grid-scale corner modes
that a more-distorted grid develops at high resolution).  A regression in
the port (e.g. silently falling back to equiangular, whose corner aspect
is ~1.40) would break these.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import make_fv3_native_grid  # noqa: E402


def _gcd(la1, lo1, la2, lo2):
    dl = lo2 - lo1
    dla = la2 - la1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dl / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _edge_lengths(lon, lat):
    """Great-circle i-edge and j-edge lengths from (6, n+1, n+1) corners."""
    dx = _gcd(lat[:, :-1, :], lon[:, :-1, :], lat[:, 1:, :], lon[:, 1:, :])
    dy = _gcd(lat[:, :, :-1], lon[:, :, :-1], lat[:, :, 1:], lon[:, :, 1:])
    return dx, dy


@pytest.mark.parametrize("im", [48, 96])
def test_gnomonic_ed_sqrt2_dx_ratio(im):
    lon, lat = make_fv3_native_grid(im, grid_type=0)
    dx, dy = _edge_lengths(np.asarray(lon), np.asarray(lat))
    ratio = max(dx.max(), dy.max()) / min(dx.min(), dy.min())
    assert abs(ratio - np.sqrt(2.0)) < 0.02, (
        f"gnomonic_ed max/min edge ratio {ratio:.4f} != FV3 sqrt(2)~1.4142 "
        f"(a regression to equiangular gives ~1.40 on dx but the cell "
        f"aspect signature below is the discriminating check)")


@pytest.mark.parametrize("im", [48, 96])
def test_gnomonic_ed_max_aspect_1_06(im):
    lon, lat = make_fv3_native_grid(im, grid_type=0)
    dx, dy = _edge_lengths(np.asarray(lon), np.asarray(lat))
    # cell aspect = mean i-edge / mean j-edge per cell
    dxc = 0.5 * (dx[:, :, :-1] + dx[:, :, 1:])
    dyc = 0.5 * (dy[:, :-1, :] + dy[:, 1:, :])
    aspect = np.maximum(dxc, dyc) / np.minimum(dxc, dyc)
    # FV3 published max aspect = 1.06089.  Equiangular would be ~1.40 here —
    # this is the signature that distinguishes gnomonic_ed from equiangular.
    assert aspect.max() < 1.10, (
        f"gnomonic_ed max cell aspect {aspect.max():.4f} exceeds the FV3 "
        f"1.06089 signature (>1.10 ⇒ not the equal-edge gnomonic grid; "
        f"equiangular gives ~1.40)")
    assert aspect.max() > 1.02, (
        f"gnomonic_ed max aspect {aspect.max():.4f} suspiciously uniform "
        f"(<1.02) — expected ~1.06")

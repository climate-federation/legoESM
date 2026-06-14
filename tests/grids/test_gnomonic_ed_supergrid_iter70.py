"""iter70: gnomonic_ed supergrid nodes (`gnomonic_ed_supergrid_lonlat`) — the
first building block of the gated gnomonic_ed C-D grid rework.

Pins shape, on-sphere, and REFINEMENT-CONSISTENCY: the supergrid even nodes
must reproduce the gnomonic_ed corners (so cdgrid metrics derived from the
supergrid stay consistent with the A-grid), and the √2 equal-edge signature.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (  # noqa: E402
    gnomonic_ed_supergrid_lonlat, make_fv3_native_grid,
    gnomonic_ed_remap_to_create,
)


def _gcd(la1, lo1, la2, lo2):
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def test_shape_and_on_sphere():
    n = 24
    lon, lat = (np.asarray(a) for a in gnomonic_ed_supergrid_lonlat(n))
    assert lon.shape == (6, 2 * n + 1, 2 * n + 1)
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))
    assert np.all(np.abs(lat) <= np.pi / 2 + 1e-9)


def test_even_nodes_match_corners():
    """Refinement consistency: supergrid[::2, ::2] == gnomonic_ed corners."""
    n = 24
    lon_sg, lat_sg = (np.asarray(a) for a in gnomonic_ed_supergrid_lonlat(n))
    lon_c, lat_c = (np.asarray(a) for a in gnomonic_ed_remap_to_create(
        *make_fv3_native_grid(n, grid_type=0)))
    sep = _gcd(lat_c, lon_c, lat_sg[:, ::2, ::2], lon_sg[:, ::2, ::2])
    assert sep.max() < 1e-10, (
        f"supergrid even nodes diverge from corners by {sep.max():.2e} rad — "
        f"cdgrid metrics would be inconsistent with the A-grid")


def test_sqrt2_signature():
    n = 48
    lon, lat = (np.asarray(a) for a in gnomonic_ed_supergrid_lonlat(n))
    # i-edge node spacing on face 0 mid-row → √2 max/min over the half-grid
    dx = _gcd(lat[0, :-1, n], lon[0, :-1, n], lat[0, 1:, n], lon[0, 1:, n])
    ratio = dx.max() / dx.min()
    assert 1.3 < ratio < 1.55, f"supergrid spacing ratio {ratio:.3f} not ~√2"

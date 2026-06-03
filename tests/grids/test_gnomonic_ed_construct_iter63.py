"""iter63: `_gnomonic_ed_construct(theta_w)` — generalized gnomonic_ed face-2
construction, the reusable core for the halo-extended gnomonic_ed grid.

Pins (a) bit-for-bit equality with `gnomonic_ed(im)` on the standard W-edge
distribution (so the generalization is a faithful refactor, not a new grid),
and (b) that a halo-extended `theta_w` (spanning beyond ±α) yields a finite,
smooth grid whose in-domain block still matches `gnomonic_ed`.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.grids.cubed_sphere import (  # noqa: E402
    gnomonic_ed, _gnomonic_ed_construct,
)

_RSQ3 = 1.0 / np.sqrt(3.0)
_ALPHA = np.arcsin(_RSQ3)


def _standard_theta(im):
    dely = 2.0 * _ALPHA / im
    return -_ALPHA + dely * jnp.arange(im + 1, dtype=jnp.float64)


@pytest.mark.parametrize("im", [12, 24, 48])
def test_bit_match_gnomonic_ed(im):
    lon_g, lat_g = gnomonic_ed(im)
    lon_c, lat_c = _gnomonic_ed_construct(_standard_theta(im))
    assert float(jnp.max(jnp.abs(lon_g - lon_c))) < 1e-13
    assert float(jnp.max(jnp.abs(lat_g - lat_c))) < 1e-13


def test_halo_extended_grid_finite_and_consistent():
    """theta_w extended by `ext` cells on each side: output finite + the
    interior (im+1)x(im+1) block bit-matches the unextended construction."""
    im, ext = 24, 3
    dely = 2.0 * _ALPHA / im
    j = jnp.arange(-ext, im + 1 + ext, dtype=jnp.float64)
    theta_ext = -_ALPHA + dely * j  # spans beyond ±α into the halo
    lon_e, lat_e = _gnomonic_ed_construct(theta_ext)
    lon_e, lat_e = np.asarray(lon_e), np.asarray(lat_e)
    assert np.all(np.isfinite(lon_e)) and np.all(np.isfinite(lat_e))
    # interior block [ext:ext+im+1, ext:ext+im+1] == unextended construction
    lon_i, lat_i = _gnomonic_ed_construct(_standard_theta(im))
    sub_lon = lon_e[ext:ext + im + 1, ext:ext + im + 1]
    sub_lat = lat_e[ext:ext + im + 1, ext:ext + im + 1]
    assert np.max(np.abs(sub_lon - np.asarray(lon_i))) < 1e-10, (
        "halo-extended interior block diverges from the unextended grid")
    assert np.max(np.abs(sub_lat - np.asarray(lat_i))) < 1e-10

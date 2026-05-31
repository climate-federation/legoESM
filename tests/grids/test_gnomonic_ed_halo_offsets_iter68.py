"""iter68: gnomonic_ed halo interpolation offsets (`compute_halo_interp_offsets_ed`).

The cross-face fractional-index correction for the FV3 gnomonic_ed grid, the
gating blocker codex flagged (must NOT reuse the equiangular offsets).  Computed
by position-matching each face's first-halo cell to its neighbour's in-domain
edge strip on the actual extended gnomonic_ed centres.

These tests pin the contract + that the offsets are genuinely gnomonic_ed-
specific (differ from equiangular), bounded, and improve halo continuity vs no
correction.  NOTE: the offset is a SUB-CELL refinement; cross-face linear-interp
error dominates field tests, so equiangular-vs-gnomonic_ed sub-cell optimality
is below the field-test noise floor — final validation is the gated dynamical
run (W5 C96 eigenmode).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.grids.halo import (  # noqa: E402
    compute_halo_interp_offsets_ed, compute_halo_interp_offsets, pad_halo_4d,
)
from legoesm.grids.cubed_sphere import _compute_gnomonic_ed_lonlat  # noqa: E402


def test_contract_and_bounded():
    n = 24
    off = np.asarray(compute_halo_interp_offsets_ed(n))
    assert off.shape == (6, 4, n)
    assert np.all(np.isfinite(off))
    # corner mismatch is sub-cell; gnomonic_ed's near-uniform cells give a
    # smaller correction than equiangular (whose maxabs is ~0.5).
    assert np.abs(off).max() < 0.5, "offset exceeds half a cell — suspicious"


def test_gnomonic_ed_specific():
    """Must differ from the equiangular offsets (codex's gating concern: using
    equiangular offsets on a gnomonic_ed grid is the wrong geometry)."""
    n = 24
    off_ed = np.asarray(compute_halo_interp_offsets_ed(n))
    off_eq = np.asarray(compute_halo_interp_offsets(n))
    assert np.abs(off_ed - off_eq).max() > 0.05, (
        "gnomonic_ed offsets coincide with equiangular — not grid-specific")


def test_offsets_help_vs_none():
    """Using the gnomonic_ed offsets in pad_halo improves cross-seam continuity
    of a smooth field vs no correction (sign/scale sanity)."""
    n = 24
    off = jnp.asarray(compute_halo_interp_offsets_ed(n))
    lon, lat = (np.asarray(a) for a in _compute_gnomonic_ed_lonlat(n))
    fld = np.sin(lat) + 0.3 * np.cos(lat) * np.cos(lon)

    def seam(offsets):
        fp = np.asarray(pad_halo_4d(jnp.asarray(fld[..., None]), halo=1,
                                    interp_offsets=offsets))[..., 0]
        rs = []
        for f in range(6):
            a = fp[f]
            gi = np.sqrt(np.mean(np.concatenate([
                (a[2:-1, 1:-1] - a[1:-2, 1:-1]).ravel(),
                (a[1:-1, 2:-1] - a[1:-1, 1:-2]).ravel()]) ** 2))
            sm = np.concatenate([a[1, 1:-1] - a[0, 1:-1], a[-1, 1:-1] - a[-2, 1:-1],
                                 a[1:-1, 1] - a[1:-1, 0], a[1:-1, -1] - a[1:-1, -2]])
            rs.append(np.sqrt(np.mean(sm ** 2)) / (gi + 1e-30))
        return max(rs)

    assert seam(off) <= seam(None) + 1e-6, (
        "gnomonic_ed offsets worsen seam continuity vs no offset (sign error?)")

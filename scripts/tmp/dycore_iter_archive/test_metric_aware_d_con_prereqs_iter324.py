"""FV3_3D iter 324: pin grid prerequisites for the metric-aware
d_con form (FV3 ``sw_core.F90:1980``).

iter-238 audit documented a gap: legoESM's d_con KE→heat formula
uses ``ΔKE = u·du + 0.5·du²`` (orthogonal-grid simplification),
while FV3's form is metric-aware:

    heat = -damp * rsin2 * (
        sum(ub², vb²) + 2*(gx+gy fluxes) - cosa_s * cross_terms
    )

requiring ``rsin2`` and ``cosa_s`` (cell-centre non-orthogonality
metrics) and ``rdx``/``rdy`` (FV3 A-grid widths).

The iter-238 comment claimed these grid fields were not available;
iter-324 audit confirms they ARE available as
``CubedSphereCDGrid.{cosa_cell, rsin2_cell, rdxa, rdya}``.  This
test pins their existence so future refactors don't drop them
without exposing the broken metric-aware port path.

Tests
-----

1. ``test_cosa_cell_present`` — FV3 ``cosa_s`` analogue.
2. ``test_rsin2_cell_present`` — FV3 ``rsin2`` analogue.
3. ``test_rdxa_rdya_present`` — FV3 ``rdx``/``rdy`` A-grid analogues.
4. ``test_metric_shapes_match_cell_centres`` — ``(6, n, n)`` shape
   on a small cube.
5. ``test_rsin2_cell_positive`` — ``rsin2_cell > 0`` everywhere
   (sin² > 0 sanity).
6. ``test_cosa_cell_bounded`` — ``|cosa_cell| ≤ 0.6`` (FV3
   non-orthogonality bound; iter-266 already pins this for
   corners + cells, iter-324 pins the cell-centre value
   specifically).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid, create_cubed_sphere_cdgrid,
)


@pytest.fixture(scope="module")
def small_cdgrid():
    n = 8
    grid = create_cubed_sphere(n)
    return create_cubed_sphere_cdgrid(grid)


def test_cosa_cell_present(small_cdgrid):
    """``CubedSphereCDGrid.cosa_cell`` exists (FV3 ``cosa_s``)."""
    assert hasattr(small_cdgrid, "cosa_cell"), (
        "CubedSphereCDGrid must expose cosa_cell (FV3 cosa_s) — "
        "prerequisite for the iter-238/324 metric-aware d_con port."
    )


def test_rsin2_cell_present(small_cdgrid):
    """``CubedSphereCDGrid.rsin2_cell`` exists (FV3 ``rsin2``)."""
    assert hasattr(small_cdgrid, "rsin2_cell"), (
        "CubedSphereCDGrid must expose rsin2_cell (FV3 rsin2) — "
        "prerequisite for the iter-238/324 metric-aware d_con port."
    )


def test_rdxa_rdya_present(small_cdgrid):
    """``rdxa`` / ``rdya`` exist (FV3 ``rdx``/``rdy`` at A-grid)."""
    for field in ("rdxa", "rdya"):
        assert hasattr(small_cdgrid, field), (
            f"CubedSphereCDGrid must expose {field} (FV3 A-grid "
            f"cell width inverse) — prerequisite for the "
            f"iter-238/324 metric-aware d_con port."
        )


def test_metric_shapes_match_cell_centres(small_cdgrid):
    """All four metric fields share cell-centre shape (6, n, n)."""
    n = small_cdgrid.n
    expected = (6, n, n)
    for field in ("cosa_cell", "rsin2_cell", "rdxa", "rdya"):
        arr = getattr(small_cdgrid, field)
        assert arr.shape == expected, (
            f"{field}.shape = {arr.shape}; expected {expected} "
            f"(cell-centre layout)."
        )


def test_rsin2_cell_positive(small_cdgrid):
    """``rsin2_cell > 0`` everywhere (sin² > 0 sanity)."""
    rsin2 = np.asarray(small_cdgrid.rsin2_cell)
    assert np.all(rsin2 > 0.0), (
        "rsin2_cell must be strictly positive (sin² > 0); a zero "
        "or negative value indicates a singularity at the cube "
        "vertex that would NaN the metric-aware d_con port."
    )


def test_cosa_cell_bounded(small_cdgrid):
    """``|cosa_cell| ≤ 0.6`` (FV3 non-orthogonality bound).
    iter-266 pins this for corners + cells; iter-324 pins it for
    cells specifically (the metric-aware d_con form uses
    cell-centre cosa_s)."""
    cosa = np.asarray(small_cdgrid.cosa_cell)
    abs_max = float(np.max(np.abs(cosa)))
    assert abs_max <= 0.6, (
        f"max|cosa_cell| = {abs_max:.4f} > 0.6 — exceeds FV3's "
        f"cubed-sphere non-orthogonality bound; the metric-aware "
        f"d_con port (iter-238/324) would amplify near-singular "
        f"cosa_s contributions in the cross-term."
    )

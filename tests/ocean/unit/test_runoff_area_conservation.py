"""Area-conservative runoff renormalisation helpers
(`run_omip_core2._load_nemo_cell_area_m2`, `._area_conservative_scale`).

Why: the OMIP runoff was IDW-regridded WITHOUT area-integral conservation, so the
global freshwater input ∫runoff·dA differed by target grid — MPAS came out ~0.5 PSU
too fresh vs tripole ~0.1.  The fix area-weights the regridded runoff so every grid
receives the SAME source total (NEMO's exact e1t·e2t metric).  These tests pin:
the scale that forces ∫field·dA over wet cells to a target integral (incl. the
all-dry no-op), and — when the NEMO domain_cfg is present — that the source area
metric is the runoff-grid shape with a physical global total (~Earth surface).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from legoesm import constants


def _scale():
    from scripts.run.run_omip_core2 import _area_conservative_scale
    return _area_conservative_scale


def test_area_conservative_scale_hits_target_integral():
    """After scaling, ∫field·area over the wet cells equals the target integral."""
    scale = _scale()
    rng = np.random.default_rng(0)
    field = rng.uniform(0.0, 3.0, size=(20, 30))
    area = rng.uniform(0.5, 2.0, size=(20, 30))
    wet = rng.uniform(size=(20, 30)) > 0.3
    target = 12345.6
    out = scale(field, area, wet, target)
    got = float((out * area * wet).sum())
    assert abs(got - target) / target < 1e-12
    # only the wet integral is pinned; the field is uniformly scaled everywhere
    ratio = out[field > 0] / field[field > 0]
    assert np.allclose(ratio, ratio.flat[0])


def test_area_conservative_scale_noop_on_all_dry_or_zero():
    """An all-zero (or all-dry) field has nothing to scale → returned unchanged,
    and NO divide-by-zero."""
    scale = _scale()
    area = np.ones((4, 4))
    zero = np.zeros((4, 4))
    out = scale(zero, area, np.ones((4, 4), dtype=bool), 999.0)
    assert np.array_equal(out, zero)
    field = np.ones((4, 4))
    dry = np.zeros((4, 4), dtype=bool)            # nothing wet
    out2 = scale(field, area, dry, 999.0)
    assert np.array_equal(np.asarray(out2), field)


def test_nemo_cell_area_metric_shape_and_total():
    """When the NEMO domain_cfg is available: e1t·e2t has the Dai-Trenberth runoff
    grid shape (331, 360), is strictly positive, and its GLOBAL total is the Earth
    surface area (~5.1e14 m²) — i.e. the eORCA1 metric, NOT the buggy gradient
    approximation (whose ±180-seam blow-up gave cells up to ~7e13 m² each)."""
    from scripts.run.run_omip_core2 import _load_nemo_cell_area_m2, _NEMO_DOMAIN_CFG
    if not os.path.exists(_NEMO_DOMAIN_CFG):
        pytest.skip(f"NEMO domain_cfg not present: {_NEMO_DOMAIN_CFG}")
    A = _load_nemo_cell_area_m2()
    assert A.shape == (331, 360)
    assert (A > 0).all()
    total = float(A.sum())
    earth = 4.0 * np.pi * (constants.R_earth ** 2)     # ~5.10e14 m^2
    assert abs(total - earth) / earth < 0.05           # eORCA1 tiles the globe
    assert float(A.max()) < 1.0e11                     # no dateline-seam blow-up cell


def test_voronoi_runoff_spread_conserves_and_widens():
    """The MPAS runoff coastal-spread path: a single-cell river spike, after the
    Voronoi neighbour-average smooth (ocean-masked) + the area-conservative
    renorm, MUST (a) spread to neighbours (the over-concentration fix) and
    (b) preserve the area-integral exactly (the conservation guarantee).
    Mirrors load_runoff_monthly's MPAS branch on a synthetic ring mesh."""
    from legoesm.ocean.bathymetry import laplacian_smooth_voronoi
    scale = _scale()
    # ring of N cells; each cell's 2 neighbours are c-1, c+1 (periodic)
    N = 12
    coc = np.stack([(np.arange(N) - 1) % N, (np.arange(N) + 1) % N], axis=0)  # (2,N)
    nec = np.full(N, 2, dtype=np.int64)
    area = np.full(N, 3.0)                      # uniform cell area
    ocean = np.ones(N, dtype=bool)
    Rm = np.zeros(N); Rm[0] = 10.0             # all discharge in one cell
    F_src = float((Rm * area * ocean).sum())   # the source integral to preserve
    for _ in range(4):                         # MPAS spread passes
        sm = np.asarray(laplacian_smooth_voronoi(Rm, coc, nec, 1))
        Rm = np.where(ocean, sm, 0.0)
    Rm = scale(Rm, area, ocean, F_src)
    # (b) conservation: the area-integral is exactly the source total
    assert abs(float((Rm * area * ocean).sum()) - F_src) / F_src < 1e-12
    # (a) widened: the spike spread to neighbours and the peak dropped
    assert Rm[1] > 0 and Rm[N - 1] > 0
    assert Rm[0] < 10.0
    assert int((Rm > 1e-9).sum()) >= 5         # spread to a coastal band

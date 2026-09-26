"""gridscale_speckle: the growth exponent must separate a smooth field from
grid-scale noise BEFORE it is quoted on a run.

A smooth large-scale pattern must score near 2 (differentiable field), and a
field whose variance is entirely at the cell scale must score near 0.  A test
that only checked ordering would pass for any monotone statistic, so both
limits are pinned with tolerances.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_P = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate"
      / "amip_bias" / "gridscale_speckle.py")
SUBDIVISION = 4          # 2562 cells: enough pairs, seconds to build


@pytest.fixture(scope="module")
def gs():
    spec = importlib.util.spec_from_file_location("gridscale_speckle", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mesh_and_pairs(gs):
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(SUBDIVISION)
    p1, p2 = gs.ring_pairs(mesh)
    return mesh, p1, p2


def test_rings_are_disjoint_and_ordered(gs, mesh_and_pairs):
    mesh, p1, p2 = mesh_and_pairs
    s1, s2 = gs.separation(mesh, p1), gs.separation(mesh, p2)
    assert s1.mean() < s2.mean()
    one = {tuple(sorted(e)) for e in p1.T}
    two = {tuple(sorted(e)) for e in p2.T}
    assert not (one & two)
    assert all(i != j for i, j in two)


def test_smooth_field_scores_near_two(gs, mesh_and_pairs):
    mesh, p1, p2 = mesh_and_pairs
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    smooth = np.cos(lat) * np.sin(2.0 * lon) + 0.5 * np.sin(3.0 * lat)
    p, *_ = gs.structure_slope(smooth, mesh, p1, p2)
    assert p == pytest.approx(2.0, abs=0.25)


def test_cell_scale_noise_scores_near_zero(gs, mesh_and_pairs):
    mesh, p1, p2 = mesh_and_pairs
    noise = np.random.default_rng(0).normal(size=np.asarray(mesh.latCell).shape)
    p, *_ = gs.structure_slope(noise, mesh, p1, p2)
    assert abs(p) < 0.2


def test_noise_on_top_of_a_smooth_field_lowers_the_exponent(gs, mesh_and_pairs):
    """The operational case: same large-scale field, speckle added."""
    mesh, p1, p2 = mesh_and_pairs
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    smooth = np.cos(lat) * np.sin(2.0 * lon)
    rng = np.random.default_rng(1)
    dirty = smooth + 0.2 * rng.normal(size=smooth.shape)
    clean_p, *_ = gs.structure_slope(smooth, mesh, p1, p2)
    dirty_p, *_ = gs.structure_slope(dirty, mesh, p1, p2)
    assert dirty_p < clean_p - 0.5


def test_constant_field_is_refused(gs, mesh_and_pairs):
    mesh, p1, p2 = mesh_and_pairs
    with pytest.raises(SystemExit, match="constant"):
        gs.structure_slope(np.ones(np.asarray(mesh.latCell).shape), mesh, p1, p2)


def test_extremum_fraction_separates_noise_from_patchiness(gs, mesh_and_pairs):
    """The second instrument: a smooth field and a PATCHY but smooth field both
    score low; cell-scale noise scores near the random-mesh value 2/7."""
    mesh, _, _ = mesh_and_pairs
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    smooth = np.cos(lat) * np.sin(2.0 * lon)
    assert gs.extremum_fraction(smooth, mesh) < 0.05
    # Patchy: the same smooth field, zeroed outside isolated blobs.  Physical
    # intermittency must NOT register as noise.
    patchy = np.where(smooth > 0.6, smooth, 0.0)
    assert gs.extremum_fraction(patchy, mesh) < 0.10
    noise = np.random.default_rng(2).normal(size=smooth.shape)
    assert gs.extremum_fraction(noise, mesh) == pytest.approx(2.0 / 7.0, abs=0.06)

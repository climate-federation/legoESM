"""Subgrid swap of the SWG twin start: the coarse (symmetric-objective) field is kept, the subgrid is replaced."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tw():
    mod = _load(ROOT / "scripts/run/run_swg_subgrid_twins.py")
    restrict = _load(mod.EVAL).truth_restrictor("symmetric")
    mod.restrict, mod.source, mod.target = restrict, restrict.keywords["source"], restrict.keywords["target"]
    return mod


def _div_free(tw, seed):
    """Random closed-wall non-divergent fine U/V (psi noise, zero on the walls)."""
    sj0, sj1, si0, si1 = tw._wet_bounds(tw.source["mask"]["T"])
    psi = np.random.default_rng(seed).normal(size=(sj1 - sj0 + 2, si1 - si0 + 2))
    return tw.uv_from_vertex_streamfunction(psi, tw.source)


def _rel(a, b):
    return max(np.abs(x - y).max() for x, y in zip(a, b)) / max(np.abs(y).max() for y in b)


def test_prolongation_is_a_right_inverse_of_the_restriction(tw):
    cu, cv = tw.restrict_uv(tw.restrict, *_div_free(tw, 0))
    back = tw.restrict_uv(tw.restrict, *tw.prolong_faces(cu, cv, tw.source, tw.target))
    assert _rel(back, (cu, cv)) < 1e-12


def test_box_swap_keeps_the_coarse_field_and_replaces_the_subgrid(tw):
    a, b, c = (_div_free(tw, s) for s in (1, 2, 3))
    s_ab = tw.swap_subgrid(tw.restrict, tw.source, tw.target, *a, *b)
    assert _rel(tw.restrict_uv(tw.restrict, *s_ab), tw.restrict_uv(tw.restrict, *a)) < 1e-12
    # the keep field's own subgrid is gone: swapping it for c's first changes nothing
    a2 = tw.swap_subgrid(tw.restrict, tw.source, tw.target, *a, *c)
    s_a2b = tw.swap_subgrid(tw.restrict, tw.source, tw.target, *a2, *b)
    assert _rel(s_a2b, s_ab) < 1e-12
    assert _rel(s_ab, a) > 0.1  # and the donor's subgrid is in


def test_psi_swap_is_nondivergent_keeps_the_coarse_field_and_replaces_the_subgrid(tw):
    a, b, c = (_div_free(tw, s) for s in (4, 5, 6))
    dx = tw.source["spacing_m"]
    s_ab = tw.swap_subgrid_psi(tw.source, tw.target, *a, *b)
    assert np.abs(tw.divergence(*s_ab, dx)).max() < 1e-12 * np.abs(s_ab[0]).max() / dx
    assert _rel(tw.restrict_uv(tw.restrict, *s_ab), tw.restrict_uv(tw.restrict, *a)) < 1e-12
    a2 = tw.swap_subgrid_psi(tw.source, tw.target, *a, *c)
    assert _rel(tw.swap_subgrid_psi(tw.source, tw.target, *a2, *b), s_ab) < 1e-12
    assert _rel(s_ab, a) > 0.1


def test_streamfunction_round_trip_reproduces_a_nondivergent_field(tw):
    u, v = _div_free(tw, 7)
    psi, north, v_err = tw.vertex_streamfunction(u, v, tw.source)
    ub, vb = tw.uv_from_vertex_streamfunction(psi, tw.source)
    assert _rel((ub, vb), (u, v)) < 1e-10 and v_err < 1e-10 * np.abs(v).max()

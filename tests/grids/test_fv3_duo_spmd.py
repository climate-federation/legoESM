"""Duo SPMD depth census: decode self-test, the measured census, the gate.

The census is the SPMD ring design's load-bearing number (every cross-face
read must land inside the gathered ring), so this file pins it and proves
the instrument can fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_duo_spmd import (
    _self_test,
    assert_ring_width_covers,
    census_cross_face_depth,
)

N, NG = 12, 3


@pytest.fixture(scope="module")
def tab():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.grids.fv3_duo_halos import build_jax_duo_halo_tables
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    return build_jax_duo_halo_tables(ctx["ectx"], ctx["gs6"], nq=1)


def test_census_matches_the_measured_depth(tab):
    """C48 measured 7 (jobs 9493433-38); the builders' strip arithmetic is
    n-independent (codex-read: neighbor maps are affine, halo widths come
    from ng/ngp), so N=12 at the same ng=3/ngp=4 must census the same.
    A different number here is a FINDING about n-dependence, not a
    tolerance to relax."""
    assert census_cross_face_depth(tab) == 7


def test_gate_refuses_a_narrow_ring(tab):
    with pytest.raises(ValueError, match="does not cover"):
        assert_ring_width_covers(tab, 7)   # == depth: no margin, refused
    assert assert_ring_width_covers(tab, 8) == 7


def test_decode_self_test_can_fail(tab):
    """Synthetic violation (non-vacuity): a layout whose idx LIES must be
    caught by the self-test -- otherwise a broken decode could bless a
    broken ring width."""
    class _Lying:
        shapes = tab.lay_a.shapes
        bases = tab.lay_a.bases
        total = tab.lay_a.total

        @staticmethod
        def idx(k, face, i0, j0):
            return tab.lay_a.idx(k, face, i0, j0) + 1   # off by one

    with pytest.raises(AssertionError, match="self-test"):
        _self_test(_Lying())


def test_census_refuses_an_unknown_op_family(tab):
    """A table op without 'dst' must be a hard error, never a skip."""
    from legoesm.grids.fv3_duo_spmd import _max_cross_depth

    class _Alien:
        pass

    with pytest.raises(TypeError, match="fail closed"):
        _max_cross_depth(_Alien(), tab.lay_a)

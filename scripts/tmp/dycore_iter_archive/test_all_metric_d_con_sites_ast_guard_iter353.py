"""FV3_3D iter 353: AST regression guard for the metric-aware
d_con wiring at ALL 8 PE+NH d_con sites.

Iter-340 covered the damp_v_d_con sites (PE iter-338 + NH iter-
339).  iter-347-352 added metric wiring at 6 more sites
(corner_div, div_damp, A_h × PE + NH).  iter-353 pins the
``use_fv3_metric_aware_d_con`` gate + cosa_cell + rsin2_cell
reference at all 6 new sites.

Tests
-----

1-6. PE+NH corner_div / div_damp / A_h metric branches contain
     gate ``if config.use_fv3_metric_aware_d_con:`` + reference
     to ``cosa_cell`` + ``rsin2_cell``.

Catches refactors that drop the metric branch from any site.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


PE_SRC = legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")
NH_SRC = legoesm_source_path("atmosphere/dynamics/gcm/compressible_euler_cdgrid.py")


@pytest.fixture(scope="module")
def pe_source():
    return PE_SRC.read_text()


@pytest.fixture(scope="module")
def nh_source():
    return NH_SRC.read_text()


def _count_metric_gates(src: str) -> int:
    """Count ``if config.use_fv3_metric_aware_d_con:`` +
    ``if self.config.use_fv3_metric_aware_d_con:`` gates."""
    pat = (
        r"if\s+(?:self\.)?config\.use_fv3_metric_aware_d_con\s*:"
    )
    return len(re.findall(pat, src))


def test_pe_has_four_metric_gates(pe_source):
    """PE source has 4 metric gates (damp_v + corner_div +
    div_damp + A_h)."""
    n = _count_metric_gates(pe_source)
    assert n == 4, (
        f"PE source should have 4 metric d_con gates (damp_v +"
        f" corner_div + div_damp + A_h); found {n}."
    )


def test_nh_has_four_metric_gates(nh_source):
    """NH source has 4 metric gates (damp_v + corner_div +
    div_damp + A_h)."""
    n = _count_metric_gates(nh_source)
    assert n == 4, (
        f"NH source should have 4 metric d_con gates; found {n}."
    )


def test_pe_metric_iter_markers_present(pe_source):
    """All 4 PE iter markers (338, 347, 349, 351) present."""
    for marker in ("iter 338", "iter 347", "iter 349", "iter 351"):
        assert marker in pe_source, (
            f"PE source missing iter marker {marker!r}."
        )


def test_nh_metric_iter_markers_present(nh_source):
    """All 4 NH iter markers (339, 348, 350, 352) present."""
    for marker in ("iter 339", "iter 348", "iter 350", "iter 352"):
        assert marker in nh_source, (
            f"NH source missing iter marker {marker!r}."
        )


def test_cosa_rsin2_used_at_each_metric_site(pe_source, nh_source):
    """Each metric-gate branch references cosa_cell + rsin2_cell."""
    for src, label in [(pe_source, "PE"), (nh_source, "NH")]:
        # Split into gate blocks; each must reference both
        # cosa_cell + rsin2_cell.
        gate_positions = [
            m.start() for m in re.finditer(
                r"if\s+(?:self\.)?config\.use_fv3_metric_aware_d_con\s*:",
                src,
            )
        ]
        assert len(gate_positions) == 4, f"{label} gate count != 4"
        # Look at next ~3000 chars after each gate to find the
        # impl.
        for pos in gate_positions:
            block = src[pos:pos + 3000]
            assert "cosa_cell" in block, (
                f"{label} metric branch at pos {pos} missing "
                f"cosa_cell ref."
            )
            assert "rsin2_cell" in block, (
                f"{label} metric branch at pos {pos} missing "
                f"rsin2_cell ref."
            )

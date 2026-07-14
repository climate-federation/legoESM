"""FV3_3D iter 485: AST regression guard for the 4 user-facing
legoESM-min-edge factories.

Sites pinned:
* iter-467 ``make_legoesm_nh_min_edge_config``
* iter-468 ``make_legoesm_pe_min_edge_config``
* iter-483 ``make_legoesm_nh_min_edge_aggressive_config``
* iter-484 ``make_legoesm_pe_min_edge_aggressive_config``

For each: function definition + presence of the 3 (or 4)
override overrides.  Catches a silent regression where a
maintainer drops or accidentally changes one of these
factory overrides.

Tests
-----

1-4: function defs present.
5-6: NH+PE min_edge factories override 3 hurting flags.
7-8: NH+PE aggressive factories add the d2_bg=5e-2 boost.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


NH_SRC = legoesm_source_path(
    "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
)
PE_SRC = legoesm_source_path(
    "atmosphere/dynamics/gcm/primitive_eq_cdgrid.py"
)


@pytest.fixture(scope="module")
def nh_text():
    return NH_SRC.read_text()


@pytest.fixture(scope="module")
def pe_text():
    return PE_SRC.read_text()


def test_nh_min_edge_factory_def_present(nh_text):
    assert re.search(
        r"def\s+make_legoesm_nh_min_edge_config\s*\(",
        nh_text,
    )


def test_nh_min_edge_aggressive_factory_def_present(nh_text):
    assert re.search(
        r"def\s+make_legoesm_nh_min_edge_aggressive_config\s*\(",
        nh_text,
    )


def test_pe_min_edge_factory_def_present(pe_text):
    assert re.search(
        r"def\s+make_legoesm_pe_min_edge_config\s*\(",
        pe_text,
    )


def test_pe_min_edge_aggressive_factory_def_present(pe_text):
    assert re.search(
        r"def\s+make_legoesm_pe_min_edge_aggressive_config\s*\(",
        pe_text,
    )


def _factory_body(text, name):
    m = re.search(
        rf"def\s+{name}\s*\([^)]*\)[^:]*:\s*([\s\S]+?)(?=\ndef\s|\Z)",
        text,
    )
    assert m is not None
    return m.group(1)


def test_nh_min_edge_overrides_3_flags(nh_text):
    body = _factory_body(nh_text, "make_legoesm_nh_min_edge_config")
    assert "use_fv3_metric_aware_d_con=False" in body, (
        "iter-466 metric_aware_d_con override missing."
    )
    assert "heat_source_del2_iters=0" in body, (
        "iter-466 heat_source_del2 override missing."
    )
    assert "d_con_top_zero_levels=0" in body, (
        "iter-466 d_con_top_zero override missing."
    )


def test_pe_min_edge_overrides_3_flags(pe_text):
    body = _factory_body(pe_text, "make_legoesm_pe_min_edge_config")
    assert "use_fv3_metric_aware_d_con=False" in body
    assert "heat_source_del2_iters=0" in body
    assert "d_con_top_zero_levels=0" in body


def test_nh_aggressive_adds_d2_bg_boost(nh_text):
    body = _factory_body(
        nh_text, "make_legoesm_nh_min_edge_aggressive_config",
    )
    assert "use_fv3_metric_aware_d_con=False" in body
    assert "heat_source_del2_iters=0" in body
    assert "d_con_top_zero_levels=0" in body
    assert "corner_div_damp_d2_bg=5e-2" in body, (
        "iter-481 d2_bg=5e-2 boost missing."
    )


def test_pe_aggressive_adds_d2_bg_boost(pe_text):
    body = _factory_body(
        pe_text, "make_legoesm_pe_min_edge_aggressive_config",
    )
    assert "use_fv3_metric_aware_d_con=False" in body
    assert "heat_source_del2_iters=0" in body
    assert "d_con_top_zero_levels=0" in body
    assert "corner_div_damp_d2_bg=5e-2" in body

"""FV3_3D iter 460: AST regression guard for iter-457 NH +
iter-458 PE heat_source del-2 smoothing wirings + iter-459
factory default.

Catches silent drop of:
* config field definitions (NH + PE: heat_source_del2_iters,
  heat_source_del2_coeff)
* wiring site in slow_tendencies that calls laplacian_compact_3d
  iteratively on _d_con_sum
* factory default heat_source_del2_iters=2 (FV3 nf_ke at nord=1)
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


NH_SRC = legoesm_source_path("atmosphere/dynamics/gcm/compressible_euler_cdgrid.py")
PE_SRC = legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")


@pytest.fixture(scope="module")
def nh_text():
    return NH_SRC.read_text()


@pytest.fixture(scope="module")
def pe_text():
    return PE_SRC.read_text()


def test_nh_config_field_iters_present(nh_text):
    assert re.search(
        r"heat_source_del2_iters\s*:\s*int\s*=\s*0", nh_text,
    )


def test_nh_config_field_coeff_present(nh_text):
    assert re.search(
        r"heat_source_del2_coeff\s*:\s*float\s*=\s*0\.20", nh_text,
    )


def test_pe_config_field_iters_present(pe_text):
    assert re.search(
        r"heat_source_del2_iters\s*:\s*int\s*=\s*0", pe_text,
    )


def test_pe_config_field_coeff_present(pe_text):
    assert re.search(
        r"heat_source_del2_coeff\s*:\s*float\s*=\s*0\.20", pe_text,
    )


def test_nh_smoothing_wired_in_slow_tendencies(nh_text):
    """NH slow_tendencies has the del-2 loop applied to
    ``_d_con_sum`` under the iters>0 gate."""
    pat = (
        r"if\s+config\.heat_source_del2_iters\s*>\s*0\s*:"
        r"[\s\S]{0,500}?_cd_hs\s*="
        r"[\s\S]{0,300}?for\s+_\s+in\s+range\("
        r"config\.heat_source_del2_iters\)"
        r"[\s\S]{0,300}?_d_con_sum\s*=\s*_d_con_sum\s*\+"
    )
    assert re.search(pat, nh_text), (
        "iter-457 NH heat_source del-2 smoothing wiring missing."
    )


def test_pe_smoothing_wired_in_slow_tendencies(pe_text):
    pat = (
        r"if\s+config\.heat_source_del2_iters\s*>\s*0\s*:"
        r"[\s\S]{0,500}?_cd_hs_pe\s*="
        r"[\s\S]{0,300}?for\s+_\s+in\s+range\("
        r"config\.heat_source_del2_iters\)"
        r"[\s\S]{0,300}?_d_con_sum\s*=\s*_d_con_sum\s*\+"
    )
    assert re.search(pat, pe_text), (
        "iter-458 PE heat_source del-2 smoothing wiring missing."
    )


def test_nh_factory_default_iters_2(nh_text):
    m = re.search(
        r"def\s+make_fv3_faithful_nh_config[\s\S]+?defaults\s*=\s*dict\(([\s\S]+?)\)",
        nh_text,
    )
    assert m is not None
    block = m.group(1)
    assert "heat_source_del2_iters=2" in block, (
        "iter-459 NH factory default heat_source_del2_iters=2 missing."
    )


def test_pe_factory_default_iters_2(pe_text):
    m = re.search(
        r"def\s+make_fv3_faithful_pe_config[\s\S]+?defaults\s*=\s*dict\(([\s\S]+?)\)",
        pe_text,
    )
    assert m is not None
    block = m.group(1)
    assert "heat_source_del2_iters=2" in block, (
        "iter-459 PE factory default heat_source_del2_iters=2 missing."
    )

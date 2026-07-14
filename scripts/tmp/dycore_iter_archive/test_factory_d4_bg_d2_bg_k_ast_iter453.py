"""FV3_3D iter 453: AST regression guard for iter-451 factory
``d4_bg=0.16`` default + iter-452 ``d2_bg_k*=0`` rollback.

Catches silent regression:
* If a future change accidentally restores ``d2_bg_k1=4.0`` or
  ``d2_bg_k2=2.0`` in factory defaults, this test fires —
  reminds maintainer of the iter-452 normalization mismatch.
* If a future change drops the iter-451 ``d4_bg=0.16``, this
  test fires.

These are source-level AST guards on the factory ``defaults =
dict(...)`` block.

Tests
-----

1. ``test_nh_factory_has_d4_bg_016``.
2. ``test_pe_factory_has_d4_bg_016``.
3. ``test_nh_factory_does_not_set_d2_bg_k1`` — iter-452 rollback.
4. ``test_pe_factory_does_not_set_d2_bg_k1``.
5. ``test_nh_factory_does_not_set_d2_bg_k2``.
6. ``test_pe_factory_does_not_set_d2_bg_k2``.
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


def _factory_defaults_block(text: str) -> str:
    """Extract the ``defaults = dict(...)`` block from
    make_fv3_faithful_*_config."""
    m = re.search(
        r"def\s+make_fv3_faithful_\w+_config[\s\S]+?defaults\s*=\s*dict\(([\s\S]+?)\)",
        text,
    )
    assert m is not None, "could not locate factory defaults block"
    return m.group(1)


@pytest.fixture(scope="module")
def nh_defaults():
    return _factory_defaults_block(NH_SRC.read_text())


@pytest.fixture(scope="module")
def pe_defaults():
    return _factory_defaults_block(PE_SRC.read_text())


def test_nh_factory_has_d4_bg_016(nh_defaults):
    assert "corner_div_damp_d4_bg=0.16" in nh_defaults, (
        "iter-451 NH factory default d4_bg=0.16 missing."
    )


def test_pe_factory_has_d4_bg_016(pe_defaults):
    assert "corner_div_damp_d4_bg=0.16" in pe_defaults, (
        "iter-451 PE factory default d4_bg=0.16 missing."
    )


def test_nh_factory_does_not_set_d2_bg_k1(nh_defaults):
    """iter-452: ``d2_bg_k1`` should NOT be set in factory
    defaults (causes blow-up due to scale mismatch)."""
    pat = r"corner_div_damp_d2_bg_k1\s*=\s*[\d.]+"
    matches = re.findall(pat, nh_defaults)
    assert not matches, (
        f"iter-452 NH factory must NOT set corner_div_damp_"
        f"d2_bg_k1 — FV3 normalization mismatch causes blow-"
        f"up.  Found: {matches}."
    )


def test_pe_factory_does_not_set_d2_bg_k1(pe_defaults):
    pat = r"corner_div_damp_d2_bg_k1\s*=\s*[\d.]+"
    matches = re.findall(pat, pe_defaults)
    assert not matches, (
        f"iter-452 PE factory must NOT set corner_div_damp_"
        f"d2_bg_k1.  Found: {matches}."
    )


def test_nh_factory_does_not_set_d2_bg_k2(nh_defaults):
    pat = r"corner_div_damp_d2_bg_k2\s*=\s*[\d.]+"
    matches = re.findall(pat, nh_defaults)
    assert not matches, (
        f"iter-452 NH factory must NOT set corner_div_damp_"
        f"d2_bg_k2.  Found: {matches}."
    )


def test_pe_factory_does_not_set_d2_bg_k2(pe_defaults):
    pat = r"corner_div_damp_d2_bg_k2\s*=\s*[\d.]+"
    matches = re.findall(pat, pe_defaults)
    assert not matches, (
        f"iter-452 PE factory must NOT set corner_div_damp_"
        f"d2_bg_k2.  Found: {matches}."
    )

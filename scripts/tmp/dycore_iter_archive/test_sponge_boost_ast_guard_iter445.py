"""FV3_3D iter 445: AST regression guard for iter-438 through
443 FV3 sponge boost wirings (corner-div d2_bg_k1/k2 + damp_w +
damp_v).

Catches silent fidelity regressions where a future refactor
drops a wiring while keeping the config field.

Sites pinned
------------

PE (primitive_eq_cdgrid.py):
* iter-438: ``corner_div_damp_d2_bg_k1`` override at PE
  corner-div site.
* iter-439: ``corner_div_damp_d2_bg_k2`` override at PE
  corner-div site (with 0.01 / 0.05 thresholds).
* iter-443: ``use_fv3_sponge_damp_v`` linear scaling at PE
  damp_v site.

NH (compressible_euler_cdgrid.py):
* iter-440: ``_apply_top_sponge_damp_boost`` helper called at
  BOTH NH corner-div sites.
* iter-441: ``use_fv3_sponge_damp_w`` linear scaling at NH
  damp_w site.
* iter-442: ``use_fv3_sponge_damp_v`` linear scaling at NH
  damp_v site.

Plus config-field presence checks.
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


def test_pe_config_d2_bg_k1_present(pe_text):
    assert re.search(
        r"corner_div_damp_d2_bg_k1\s*:\s*float\s*=\s*0\.0", pe_text,
    )


def test_pe_config_d2_bg_k2_present(pe_text):
    assert re.search(
        r"corner_div_damp_d2_bg_k2\s*:\s*float\s*=\s*0\.0", pe_text,
    )


def test_pe_config_sponge_damp_v_present(pe_text):
    assert re.search(
        r"use_fv3_sponge_damp_v\s*:\s*bool\s*=\s*False", pe_text,
    )


def test_nh_config_d2_bg_k1_present(nh_text):
    assert re.search(
        r"corner_div_damp_d2_bg_k1\s*:\s*float\s*=\s*0\.0", nh_text,
    )


def test_nh_config_d2_bg_k2_present(nh_text):
    assert re.search(
        r"corner_div_damp_d2_bg_k2\s*:\s*float\s*=\s*0\.0", nh_text,
    )


def test_nh_config_sponge_damp_w_present(nh_text):
    assert re.search(
        r"use_fv3_sponge_damp_w\s*:\s*bool\s*=\s*False", nh_text,
    )


def test_nh_config_sponge_damp_v_present(nh_text):
    assert re.search(
        r"use_fv3_sponge_damp_v\s*:\s*bool\s*=\s*False", nh_text,
    )


def test_pe_iter438_439_446_call_shared_helper(pe_text):
    """iter-438/439/446: PE corner-div site now calls the
    shared ``apply_top_sponge_damp_boost`` helper for k=0/k=1/
    k=2 sponge overrides.

    Catches a refactor that drops the call while leaving the
    config fields declared.
    """
    pat = (
        r"apply_top_sponge_damp_boost"
        r"[\s\S]{0,500}?"
        r"_damp_corner\s*,\s*_da_min_c\s*,\s*"
        r"config\.corner_div_damp_d2_bg\s*,\s*"
        r"config\.corner_div_damp_d2_bg_k1\s*,\s*"
        r"config\.corner_div_damp_d2_bg_k2"
    )
    assert re.search(pat, pe_text), (
        "iter-438/439/446 PE call to shared helper missing."
    )


def test_shared_helper_definition(pe_text):
    """The shared helper itself must exist (in core/)."""
    from legoesm.core.fv3_sponge_boost import (
        apply_top_sponge_damp_boost,
    )
    assert callable(apply_top_sponge_damp_boost)


def test_nh_iter440_helper_defined(nh_text):
    assert re.search(
        r"def\s+_apply_top_sponge_damp_boost\s*\(", nh_text,
    ), "iter-440 NH _apply_top_sponge_damp_boost helper missing."


def test_nh_iter440_helper_called_twice(nh_text):
    matches = re.findall(
        r"_apply_top_sponge_damp_boost\s*\(", nh_text,
    )
    # 1 definition + 2 calls = 3 occurrences
    assert len(matches) >= 3, (
        f"iter-440 helper called {len(matches)-1} time(s); "
        f"expected 2 calls (one per NH corner-div site)."
    )


def test_nh_iter441_damp_w_scaling_wired(nh_text):
    """iter-441/447: damp_w sponge boost via shared helper with
    factor=1.0, apply_at_k2=True."""
    pat = (
        r"if\s+self\.config\.use_fv3_sponge_damp_w\s*:"
        r"[\s\S]{0,500}?apply_top_sponge_field_scale"
        r"[\s\S]{0,500}?factor\s*=\s*1\.0"
        r"[\s\S]{0,500}?apply_at_k2\s*=\s*True"
    )
    assert re.search(pat, nh_text), (
        "iter-441/447 NH use_fv3_sponge_damp_w shared-helper "
        "call missing (factor=1.0, apply_at_k2=True)."
    )


def test_nh_iter442_damp_v_scaling_wired(nh_text):
    """iter-442/447: damp_v sponge boost via shared helper with
    factor=0.5, apply_at_k2=False."""
    pat = (
        r"if\s+self\.config\.use_fv3_sponge_damp_v\s*:"
        r"[\s\S]{0,500}?apply_top_sponge_field_scale"
        r"[\s\S]{0,500}?factor\s*=\s*0\.5"
        r"[\s\S]{0,500}?apply_at_k2\s*=\s*False"
    )
    assert re.search(pat, nh_text), (
        "iter-442/447 NH use_fv3_sponge_damp_v shared-helper "
        "call missing (factor=0.5, apply_at_k2=False)."
    )


def test_pe_iter443_damp_v_scaling_wired(pe_text):
    """iter-443/447: PE damp_v sponge boost via shared helper."""
    pat = (
        r"if\s+self\.config\.use_fv3_sponge_damp_v\s*:"
        r"[\s\S]{0,500}?apply_top_sponge_field_scale"
        r"[\s\S]{0,500}?factor\s*=\s*0\.5"
        r"[\s\S]{0,500}?apply_at_k2\s*=\s*False"
    )
    assert re.search(pat, pe_text), (
        "iter-443/447 PE use_fv3_sponge_damp_v shared-helper "
        "call missing."
    )

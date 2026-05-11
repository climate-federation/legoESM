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
from pathlib import Path

import pytest


NH_SRC = (
    Path(__file__).resolve().parents[1]
    / "src" / "legoesm" / "atmosphere" / "dynamics"
    / "compressible_euler_cdgrid.py"
)
PE_SRC = (
    Path(__file__).resolve().parents[1]
    / "src" / "legoesm" / "atmosphere" / "dynamics"
    / "primitive_eq_cdgrid.py"
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


def test_pe_iter438_k1_override_wired(pe_text):
    """iter-438 PE k=0 override at corner-div site."""
    pat = (
        r"if\s+config\.corner_div_damp_d2_bg_k1\s*>\s*0\.0\s*:"
        r"[\s\S]{0,600}?_damp_k1\s*=\s*_da_min_c\s*\*\s*jnp\.maximum"
        r"[\s\S]{0,400}?_damp_corner\s*=\s*jnp\.where"
    )
    assert re.search(pat, pe_text), (
        "iter-438 PE corner_div_damp_d2_bg_k1 override missing."
    )


def test_pe_iter439_k2_override_wired(pe_text):
    """iter-439 PE k=1/k=2 override with 0.01 / 0.05 thresholds."""
    pat = (
        r"if\s+config\.corner_div_damp_d2_bg_k2\s*>\s*0\.01\s*:"
        r"[\s\S]{0,400}?_damp_k2\s*=\s*_da_min_c\s*\*\s*jnp\.maximum"
        r"[\s\S]{0,500}?"
        r"if\s+config\.corner_div_damp_d2_bg_k2\s*>\s*0\.05\s*:"
        r"[\s\S]{0,200}?0\.2\s*\*\s*config\.corner_div_damp_d2_bg_k2"
    )
    assert re.search(pat, pe_text), (
        "iter-439 PE k=1 (>0.01) + k=2 (>0.05) overrides missing."
    )


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
    pat = (
        r"if\s+self\.config\.use_fv3_sponge_damp_w\s*:"
        r"[\s\S]{0,800}?_scale_k1\s*="
        r"[\s\S]{0,200}?dw\s*=\s*jnp\.where"
    )
    assert re.search(pat, nh_text), (
        "iter-441 NH use_fv3_sponge_damp_w scaling missing."
    )


def test_nh_iter442_damp_v_scaling_wired(nh_text):
    pat = (
        r"if\s+self\.config\.use_fv3_sponge_damp_v\s*:"
        r"[\s\S]{0,800}?_boosted_k1_v\s*=\s*0\.5\s*\*"
        r"[\s\S]{0,400}?du_normal\s*=\s*jnp\.where"
    )
    assert re.search(pat, nh_text), (
        "iter-442 NH use_fv3_sponge_damp_v 0.5 scaling missing."
    )


def test_pe_iter443_damp_v_scaling_wired(pe_text):
    pat = (
        r"if\s+self\.config\.use_fv3_sponge_damp_v\s*:"
        r"[\s\S]{0,800}?_boosted_pk1_v\s*=\s*0\.5\s*\*"
        r"[\s\S]{0,400}?du_normal\s*=\s*jnp\.where"
    )
    assert re.search(pat, pe_text), (
        "iter-443 PE use_fv3_sponge_damp_v 0.5 scaling missing."
    )

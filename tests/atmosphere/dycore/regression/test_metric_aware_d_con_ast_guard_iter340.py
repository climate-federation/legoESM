"""FV3_3D iter 340: AST regression guard for iter-338 + iter-339
metric-aware d_con wiring on the PE + NH ``damp_v_d_con`` sites.

Mirrors the iter-327 / iter-329 AST guard pattern.

Tests
-----

1. ``test_pe_metric_flag_gate`` — PE source has ``if
   self.config.use_fv3_metric_aware_d_con:`` gate.
2. ``test_pe_metric_uses_cosa_rsin2`` — PE source references
   ``cosa_cell`` + ``rsin2_cell`` inside the metric block.
3. ``test_nh_metric_flag_gate`` — NH source has same gate.
4. ``test_nh_metric_uses_cosa_rsin2`` — NH source references
   ``cosa_cell`` + ``rsin2_cell`` inside the metric block.
5. ``test_pe_metric_else_branch_present`` — legacy simpler form
   in else branch (bit-for-bit baseline preserved).
6. ``test_nh_metric_else_branch_present`` — same for NH.
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


def test_pe_metric_flag_gate(pe_source):
    pat = r"if\s+self\.config\.use_fv3_metric_aware_d_con\s*:"
    assert re.search(pat, pe_source), (
        "iter-338 PE metric gate missing — flag is silent no-op."
    )


def test_pe_metric_uses_cosa_rsin2(pe_source):
    """PE metric impl branch references cosa_cell + rsin2_cell."""
    # Find the IMPL block (gate condition referencing self.config),
    # not the config-field docstring.
    match = re.search(
        r"if\s+self\.config\.use_fv3_metric_aware_d_con\s*:.*?"
        r"^\s+else\s*:",
        pe_source, re.DOTALL | re.MULTILINE,
    )
    assert match is not None, "Could not find PE metric impl branch"
    block = match.group(0)
    assert "cosa_cell" in block, (
        "PE metric impl must reference cosa_cell (FV3 cosa_s)."
    )
    assert "rsin2_cell" in block, (
        "PE metric impl must reference rsin2_cell (FV3 rsin2)."
    )


def test_nh_metric_flag_gate(nh_source):
    pat = r"if\s+self\.config\.use_fv3_metric_aware_d_con\s*:"
    assert re.search(pat, nh_source), (
        "iter-339 NH metric gate missing — flag is silent no-op."
    )


def test_nh_metric_uses_cosa_rsin2(nh_source):
    """NH metric impl branch references cosa_cell + rsin2_cell."""
    match = re.search(
        r"if\s+self\.config\.use_fv3_metric_aware_d_con\s*:.*?"
        r"^\s+else\s*:",
        nh_source, re.DOTALL | re.MULTILINE,
    )
    assert match is not None, "Could not find NH metric impl branch"
    block = match.group(0)
    assert "cosa_cell" in block, (
        "NH metric impl must reference cosa_cell."
    )
    assert "rsin2_cell" in block, (
        "NH metric impl must reference rsin2_cell."
    )


def test_pe_metric_else_branch_present(pe_source):
    """Legacy iter-208 simpler form preserved in else branch."""
    pat = (
        r"else\s*:[^a-zA-Z]+"
        r"dKE_corner\s*=\s*\(\s*"
        r"u_corner\s*\*\s*du_corner"
    )
    assert re.search(pat, pe_source, re.DOTALL), (
        "iter-338 PE else branch (legacy iter-208 form) missing — "
        "flag=False no longer reproduces baseline."
    )


def test_nh_metric_else_branch_present(nh_source):
    """Legacy iter-209 simpler form preserved in else branch."""
    pat = (
        r"else\s*:[^a-zA-Z]+"
        r"dKE_cc\s*=\s*\(\s*"
        r"u_cc_new\s*\*\s*du_cc"
    )
    assert re.search(pat, nh_source, re.DOTALL), (
        "iter-339 NH else branch (legacy iter-209 form) missing — "
        "flag=False no longer reproduces baseline."
    )

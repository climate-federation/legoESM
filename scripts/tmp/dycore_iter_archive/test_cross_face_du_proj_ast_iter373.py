"""FV3_3D iter 373: AST regression guard for iter-370 cross-face
du projection wiring (PE + NH).

Mirrors iter-327/329/334/340/353/361/362 pattern.

Tests
-----

1. PE source has ``if self.config.use_fv3_cross_face_du_proj:``
   gate.
2. NH source has same gate.
3. PE source uses ``pad_halo_4d`` (or alias) on du_normal +
   dv_normal inside the gate.
4. NH source uses ``_pad_halo_4d_module`` on du_normal +
   dv_normal inside the gate.
5. Both sources preserve legacy ``jnp.pad(..., mode='edge')``
   else branch for bit-for-bit baseline.
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


def test_pe_cross_face_gate(pe_source):
    pat = r"if\s+self\.config\.use_fv3_cross_face_du_proj\s*:"
    assert re.search(pat, pe_source), (
        "iter-370 PE gate missing — flag is silent no-op."
    )


def test_nh_cross_face_gate(nh_source):
    pat = r"if\s+self\.config\.use_fv3_cross_face_du_proj\s*:"
    assert re.search(pat, nh_source), (
        "iter-370 NH gate missing — flag is silent no-op."
    )


def test_pe_cross_face_uses_pad_halo_4d(pe_source):
    """PE iter-370 branch references pad_halo_4d (or alias)
    applied to du_normal + dv_normal."""
    pat_du = r"_pad_h4\s*\(\s*du_normal"
    pat_dv = r"_pad_h4\s*\(\s*dv_normal"
    assert re.search(pat_du, pe_source), (
        "PE iter-370: pad_halo_4d(du_normal) missing."
    )
    assert re.search(pat_dv, pe_source), (
        "PE iter-370: pad_halo_4d(dv_normal) missing."
    )


def test_nh_cross_face_uses_pad_halo_4d(nh_source):
    """NH iter-370 branch references _pad_halo_4d_module on
    du_normal + dv_normal."""
    pat_du = r"_pad_halo_4d_module\s*\(\s*du_normal"
    pat_dv = r"_pad_halo_4d_module\s*\(\s*dv_normal"
    assert re.search(pat_du, nh_source), (
        "NH iter-370: _pad_halo_4d_module(du_normal) missing."
    )
    assert re.search(pat_dv, nh_source), (
        "NH iter-370: _pad_halo_4d_module(dv_normal) missing."
    )


def test_pe_legacy_mode_edge_present(pe_source):
    """PE else branch keeps jnp.pad(mode='edge') for bit-for-bit."""
    pat = (
        r"else\s*:[\s\S]{0,300}?du_pad\s*=\s*jnp\.pad\("
        r"[\s\S]{0,300}?du_normal[\s\S]{0,300}?mode\s*=\s*[\"']edge[\"']"
    )
    assert re.search(pat, pe_source, re.DOTALL), (
        "PE iter-370 else branch lost mode='edge' baseline."
    )


def test_nh_legacy_mode_edge_present(nh_source):
    """NH else branch keeps jnp.pad(mode='edge') for bit-for-bit."""
    pat = (
        r"else\s*:[\s\S]{0,300}?du_pad\s*=\s*jnp\.pad\("
        r"[\s\S]{0,300}?du_normal[\s\S]{0,300}?mode\s*=\s*[\"']edge[\"']"
    )
    assert re.search(pat, nh_source, re.DOTALL), (
        "NH iter-370 else branch lost mode='edge' baseline."
    )

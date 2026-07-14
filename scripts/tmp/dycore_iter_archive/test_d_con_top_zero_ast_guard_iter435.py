"""FV3_3D iter 435: AST regression guard for iter-431/432/433
``d_con_top_zero_levels`` mask wiring at all sites.

iter-431 wired at NH post-acoustic damp_v.
iter-432 wired at NH post-acoustic damp_w + NH aggregate sum.
iter-433 wired at PE post-acoustic damp_v + PE aggregate sum.

If a future refactor drops a mask application at any of these
5 sites, the flag still exists but is partially honored — a
silent fidelity regression.  iter-435 pins each wiring site
via AST regex so removal trips the guard at CI time.

Tests
-----

1. ``test_nh_damp_v_mask_present`` — iter-431 NH damp_v site.
2. ``test_nh_damp_w_mask_present`` — iter-432 NH damp_w site.
3. ``test_nh_aggregate_mask_present`` — iter-432 NH aggregate.
4. ``test_pe_damp_v_mask_present`` — iter-433 PE damp_v site.
5. ``test_pe_aggregate_mask_present`` — iter-433 PE aggregate.
6. ``test_nh_config_field_present`` — config field defined.
7. ``test_pe_config_field_present`` — config field defined.
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


def test_nh_config_field_present(nh_text):
    assert re.search(
        r"d_con_top_zero_levels\s*:\s*int\s*=\s*0", nh_text,
    ), "iter-431 NH config field d_con_top_zero_levels missing."


def test_pe_config_field_present(pe_text):
    assert re.search(
        r"d_con_top_zero_levels\s*:\s*int\s*=\s*0", pe_text,
    ), "iter-433 PE config field d_con_top_zero_levels missing."


def test_nh_damp_v_mask_present(nh_text):
    """iter-431: NH damp_v post-acoustic d_con site has mask."""
    pat = (
        r"if\s+self\.config\.d_con_top_zero_levels\s*>\s*0\s*:"
        r"[\s\S]{0,400}?_d_con_mask\s*=\s*jnp\.where"
        r"[\s\S]{0,200}?dtheta_p\s*=\s*dtheta_p\s*\*\s*_d_con_mask"
    )
    assert re.search(pat, nh_text), (
        "iter-431 NH damp_v d_con_top_zero_levels mask wiring "
        "missing from compressible_euler_cdgrid.py."
    )


def test_nh_damp_w_mask_present(nh_text):
    """iter-432: NH damp_w post-acoustic d_con site has mask."""
    pat = (
        r"if\s+self\.config\.d_con_top_zero_levels\s*>\s*0\s*:"
        r"[\s\S]{0,400}?_d_con_mask_w\s*=\s*jnp\.where"
        r"[\s\S]{0,200}?dtheta_p\s*=\s*dtheta_p\s*\*\s*_d_con_mask_w"
    )
    assert re.search(pat, nh_text), (
        "iter-432 NH damp_w d_con_top_zero_levels mask wiring "
        "missing from compressible_euler_cdgrid.py."
    )


def test_nh_aggregate_mask_present(nh_text):
    """iter-432: NH slow-tendency aggregate _d_con_sum has mask."""
    pat = (
        r"if\s+config\.d_con_top_zero_levels\s*>\s*0\s*:"
        r"[\s\S]{0,400}?_d_con_mask_s\s*=\s*jnp\.where"
        r"[\s\S]{0,200}?_d_con_sum\s*=\s*_d_con_sum\s*\*\s*_d_con_mask_s"
    )
    assert re.search(pat, nh_text), (
        "iter-432 NH aggregate _d_con_sum d_con_top_zero_levels "
        "mask wiring missing from compressible_euler_cdgrid.py."
    )


def test_pe_damp_v_mask_present(pe_text):
    """iter-433: PE damp_v post-acoustic d_con site has mask."""
    pat = (
        r"if\s+self\.config\.d_con_top_zero_levels\s*>\s*0\s*:"
        r"[\s\S]{0,400}?_d_con_mask_v\s*=\s*jnp\.where"
        r"[\s\S]{0,200}?dT\s*=\s*dT\s*\*\s*_d_con_mask_v"
    )
    assert re.search(pat, pe_text), (
        "iter-433 PE damp_v d_con_top_zero_levels mask wiring "
        "missing from primitive_eq_cdgrid.py."
    )


def test_pe_aggregate_mask_present(pe_text):
    """iter-433: PE slow-tendency aggregate _d_con_sum has mask."""
    pat = (
        r"if\s+config\.d_con_top_zero_levels\s*>\s*0\s*:"
        r"[\s\S]{0,400}?_d_con_mask_sp\s*=\s*jnp\.where"
        r"[\s\S]{0,200}?_d_con_sum\s*=\s*_d_con_sum\s*\*\s*_d_con_mask_sp"
    )
    assert re.search(pat, pe_text), (
        "iter-433 PE aggregate _d_con_sum d_con_top_zero_levels "
        "mask wiring missing from primitive_eq_cdgrid.py."
    )

"""FV3_3D iter 450: AST regression guard for iter-448 NH + iter-
449 PE Rayleigh friction wirings.

Catches silent drop of the Ray_fast wiring while the config
fields stay declared.

Sites pinned
------------

NH (compressible_euler_cdgrid.py):
* config field ``rf_tau_days: float = 0.0``
* config field ``rf_cutoff_pa: float = 3000.0``
* call ``compute_rff_profile`` at end of step()
* u/v/w multiplied by rff

PE (primitive_eq_cdgrid.py):
* same config fields
* same rff call at end of step()
* u_d/v_d multiplied by rff (PE has no w)

Plus helper module presence check.
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


def test_helper_module_importable():
    from legoesm.core.fv3_rayleigh_fast import (
        compute_rff_profile, pfull_from_exner,
    )
    assert callable(compute_rff_profile)
    assert callable(pfull_from_exner)


def test_nh_config_rf_tau_present(nh_text):
    assert re.search(
        r"rf_tau_days\s*:\s*float\s*=\s*0\.0", nh_text,
    )


def test_nh_config_rf_cutoff_present(nh_text):
    assert re.search(
        r"rf_cutoff_pa\s*:\s*float\s*=\s*3000\.0", nh_text,
    )


def test_pe_config_rf_tau_present(pe_text):
    assert re.search(
        r"rf_tau_days\s*:\s*float\s*=\s*0\.0", pe_text,
    )


def test_pe_config_rf_cutoff_present(pe_text):
    assert re.search(
        r"rf_cutoff_pa\s*:\s*float\s*=\s*3000\.0", pe_text,
    )


def test_nh_rf_wired_in_step(nh_text):
    """NH step() applies Ray_fast to u, v, w when rf_tau_days>0."""
    pat = (
        r"if\s+self\.config\.rf_tau_days\s*>\s*0\.0\s*:"
        r"[\s\S]{0,400}?compute_rff_profile"
        r"[\s\S]{0,1000}?u=state_new\.u\.replace"
        r"[\s\S]{0,200}?v=state_new\.v\.replace"
        r"[\s\S]{0,800}?w=state_new\.w\.replace"
    )
    assert re.search(pat, nh_text), (
        "iter-448 NH Ray_fast wiring missing or out of order."
    )


def test_pe_rf_wired_in_step(pe_text):
    """PE step() applies Ray_fast to u_d, v_d when rf_tau_days>0."""
    pat = (
        r"if\s+self\.config\.rf_tau_days\s*>\s*0\.0\s*:"
        r"[\s\S]{0,400}?compute_rff_profile"
        r"[\s\S]{0,800}?u_d=state_new\.u_d\.replace"
        r"[\s\S]{0,200}?v_d=state_new\.v_d\.replace"
    )
    assert re.search(pat, pe_text), (
        "iter-449 PE Ray_fast wiring missing or out of order."
    )

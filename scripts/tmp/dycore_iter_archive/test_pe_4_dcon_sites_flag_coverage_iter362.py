"""FV3_3D iter 362: AST flag-coverage guard for all 4 PE d_con
sites.  PE counterpart of NH iter-361.

PE has 4 d_con sites (damp_v, corner_div, div_damp, A_h) — no
damp_w (PE is hydrostatic).  PE has 1 fidelity flag for d_con
(metric); cv + dynamic Exner are NH-only per iter-331/343.

Tests
-----

1. ``test_pe_has_4_metric_gates`` — exactly 4 PE metric d_con
   gates.
2. ``test_pe_does_not_have_cv_selector`` — no
   ``constants.c_vd if config.use_fv3_d_con_cv`` pattern in PE
   source (cv flag is NH-only).
3. ``test_pe_does_not_have_exner_eff`` — no ``_exner_eff_b`` /
   ``_exner_eff_dv`` etc in PE source (PE uses actual T).
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


PE_SRC = legoesm_source_path("atmosphere/dynamics/gcm/primitive_eq_cdgrid.py")


@pytest.fixture(scope="module")
def pe_source():
    return PE_SRC.read_text()


def test_pe_has_4_metric_gates(pe_source):
    """PE has 4 metric d_con gates (damp_v + corner_div +
    div_damp + A_h)."""
    pat = (
        r"if\s+(?:self\.)?config\.use_fv3_metric_aware_d_con\s*:"
    )
    n = len(re.findall(pat, pe_source))
    assert n == 4, f"Expected 4 PE metric gates; found {n}."


def test_pe_does_not_have_cv_selector(pe_source):
    """No cv-vs-cp selector in PE source (PE uses cp_air which
    is FV3-faithful for hydrostatic; cv flag is NH-only per
    iter-331/343 asymmetry guard)."""
    pat = (
        r"constants\.c_vd\s+if\s+(?:self\.)?config\.use_fv3_d_con_cv"
    )
    n = len(re.findall(pat, pe_source))
    assert n == 0, (
        f"PE source has {n} cv selectors; should be 0 (NH-only "
        f"flag).  iter-331/343 asymmetry violated."
    )


def test_pe_does_not_have_exner_eff(pe_source):
    """No ``_exner_eff_b`` etc in PE source (PE prognostic is
    actual T, no Exner factor)."""
    for name in ("_exner_eff_b", "_exner_eff_dv", "_exner_eff_dw"):
        assert name not in pe_source, (
            f"PE source has ``{name}`` — Exner factor wiring "
            f"leaked into PE (should be NH-only)."
        )

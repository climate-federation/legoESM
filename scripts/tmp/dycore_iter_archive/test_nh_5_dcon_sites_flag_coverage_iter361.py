"""FV3_3D iter 361: AST guard verifying every NH d_con site
respects both ``use_fv3_metric_aware_d_con`` + ``use_fv3_d_con_cv``
+ ``use_fv3_dynamic_exner`` flags.

For each of 5 NH d_con sites (damp_v, damp_w, corner_div,
div_damp, A_h):

* metric gate ``if (self.)config.use_fv3_metric_aware_d_con:``
* cv selector ``constants.c_vd if (self.)config.use_fv3_d_con_cv``
* dynamic Exner ``_exner_eff_b`` (slow-tendency) or ``_pi_prime_dv/dw``
  + ``_exner_eff_dv/dw`` (post-step)

Catches refactors that drop a flag from a single site.

Tests
-----

1. ``test_nh_has_5_metric_gates`` — exactly 4 metric gates in NH
   source (damp_v + corner_div + div_damp + A_h; damp_w_d_con
   doesn't get metric since w is scalar).
2. ``test_nh_has_5_cv_selectors`` — 5 ``c_vd if ... use_fv3_d_con_cv``
   patterns (one per d_con site).
3. ``test_nh_dyn_exner_at_all_slow_sites`` — ``_exner_eff_b``
   appears in 3 places (corner_div, div_damp, A_h).
4. ``test_nh_dyn_exner_at_post_acoustic_sites`` —
   ``_exner_eff_dv`` and ``_exner_eff_dw`` both present.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


NH_SRC = legoesm_source_path(
    "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
)


@pytest.fixture(scope="module")
def nh_source():
    return NH_SRC.read_text()


def test_nh_has_4_metric_gates(nh_source):
    """NH has 4 metric d_con gates (damp_v, corner_div, div_damp,
    A_h).  damp_w is scalar -- no metric form."""
    pat = (
        r"if\s+(?:self\.)?config\.use_fv3_metric_aware_d_con\s*:"
    )
    n = len(re.findall(pat, nh_source))
    assert n == 4, f"Expected 4 NH metric gates; found {n}."


def test_nh_has_5_cv_selectors(nh_source):
    """NH has 5 cv-vs-cp selectors (one per d_con site).  Counts
    distinct assignment-targets ``_cx_<X>`` AND inline post-acoustic
    selectors ``_cx_dv`` / ``_cx_dw``."""
    pat = (
        r"constants\.c_vd\s+if\s+(?:self\.)?config\.use_fv3_d_con_cv"
    )
    n = len(re.findall(pat, nh_source))
    assert n == 5, (
        f"Expected 5 NH cv-vs-cp selectors (5 d_con sites); "
        f"found {n}."
    )


def test_nh_dyn_exner_at_slow_sites(nh_source):
    """``_exner_eff_b`` used at all 3 slow-tendency d_con sites."""
    pat = r"_exner_eff_b"
    n = len(re.findall(pat, nh_source))
    # 1 assignment + 3 uses = 4 minimum
    assert n >= 4, (
        f"Expected ``_exner_eff_b`` at >= 4 places (1 assign + 3 "
        f"d_con denominators); found {n}."
    )


def test_nh_dyn_exner_at_post_acoustic_sites(nh_source):
    """Post-acoustic d_con sites (damp_v + damp_w) have
    ``_exner_eff_dv`` and ``_exner_eff_dw``."""
    for name in ("_exner_eff_dv", "_exner_eff_dw"):
        assert name in nh_source, (
            f"NH source missing ``{name}`` — iter-337 wiring "
            f"dropped at post-acoustic site."
        )

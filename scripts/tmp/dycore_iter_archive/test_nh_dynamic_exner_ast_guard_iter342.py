"""FV3_3D iter 342: AST regression guard for iter-336 + iter-337
dynamic Exner wiring on the NH 3D path.

Mirrors iter-327 / iter-329 / iter-340 AST guard pattern.

Tests
-----

1. ``test_slow_tendency_dynamic_exner_helper`` — slow_tendencies
   fn computes ``_exner_eff_b`` via the gate
   ``if config.use_fv3_dynamic_exner:`` (iter-336).
2. ``test_slow_tendency_exner_eff_b_used_in_dcon_sites`` —
   the 3 slow-tendency d_con denominators use
   ``_exner_eff_b`` (not the old ``_exner_ref_b_*`` constant).
3. ``test_damp_v_post_acoustic_dynamic_exner`` — damp_v_d_con
   post-step site computes ``_exner_eff_dv`` via the same gate
   (iter-337).
4. ``test_damp_w_post_acoustic_dynamic_exner`` — damp_w_d_con
   post-step site computes ``_exner_eff_dw`` via the same gate.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


SRC_PATH = legoesm_source_path(
    "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
)


@pytest.fixture(scope="module")
def nh_source():
    return SRC_PATH.read_text()


def test_slow_tendency_dynamic_exner_helper(nh_source):
    """Slow_tendencies computes ``_exner_eff_b`` via dynamic Exner
    gate (iter-336)."""
    pat = (
        r"if\s+config\.use_fv3_dynamic_exner\s*:\s*\n\s+"
        r"_exner_eff_b\s*=\s*\("
    )
    assert re.search(pat, nh_source), (
        "iter-336 dynamic Exner gate + _exner_eff_b assignment "
        f"missing from {SRC_PATH.name}.  Slow-tendency d_con "
        f"sites no longer respect use_fv3_dynamic_exner=True."
    )


def test_slow_tendency_exner_eff_b_used_in_dcon_sites(nh_source):
    """All 3 slow-tendency d_con denominators use ``_exner_eff_b``."""
    # Search for ``/ (_cx_<X> * _exner_eff_b)`` pattern.
    pat = r"/\s*\(\s*_cx_\w+\s*\*\s*_exner_eff_b\s*\)"
    matches = re.findall(pat, nh_source)
    assert len(matches) >= 3, (
        f"Expected >= 3 slow-tendency d_con denominators using "
        f"_exner_eff_b; found {len(matches)}.  iter-336 wiring "
        f"may have dropped a site (corner_div / div_damp / A_h)."
    )


def test_damp_v_post_acoustic_dynamic_exner(nh_source):
    """damp_v_d_con post-step site has the iter-337 dynamic Exner
    gate."""
    pat = (
        r"if\s+self\.config\.use_fv3_dynamic_exner\s*:\s*\n\s+"
        r"_pi_prime_dv\s*=\s*compute_exner_perturbation"
    )
    assert re.search(pat, nh_source), (
        "iter-337 damp_v_d_con dynamic Exner gate + "
        "_pi_prime_dv = compute_exner_perturbation(...) missing — "
        "post-acoustic damp_v site no longer respects the flag."
    )


def test_aggregate_cap_uses_dyn_exner(nh_source):
    """iter-397: aggregate delt_max cap respects dyn_exner flag."""
    pat = (
        r"if\s+config\.use_fv3_dynamic_exner\s*:[\s\S]{0,200}?"
        r"_sf_b\s*=\s*_sponge_factor[\s\S]{0,200}?"
        r"_exner_eff_b"
    )
    assert re.search(pat, nh_source), (
        "iter-397 aggregate-cap dyn_exner wiring missing."
    )


def test_damp_v_cap_uses_dyn_exner(nh_source):
    """iter-398: damp_v post-acoustic cap respects dyn_exner."""
    pat = (
        r"if\s+self\.config\.use_fv3_dynamic_exner\s*:[\s\S]{0,200}?"
        r"delt_theta_b\s*=[\s\S]{0,200}?_exner_eff_dv"
    )
    assert re.search(pat, nh_source), (
        "iter-398 damp_v post-acoustic cap dyn_exner wiring "
        "missing."
    )


def test_damp_w_cap_uses_dyn_exner(nh_source):
    """iter-398: damp_w post-acoustic cap respects dyn_exner."""
    pat = (
        r"if\s+self\.config\.use_fv3_dynamic_exner\s*:[\s\S]{0,200}?"
        r"delt_theta_b\s*=[\s\S]{0,200}?_exner_eff_dw"
    )
    assert re.search(pat, nh_source), (
        "iter-398 damp_w post-acoustic cap dyn_exner wiring "
        "missing."
    )


def test_damp_w_post_acoustic_dynamic_exner(nh_source):
    """damp_w_d_con post-step site has the iter-337 dynamic Exner
    gate."""
    pat = (
        r"if\s+self\.config\.use_fv3_dynamic_exner\s*:\s*\n\s+"
        r"_pi_prime_dw\s*=\s*compute_exner_perturbation"
    )
    assert re.search(pat, nh_source), (
        "iter-337 damp_w_d_con dynamic Exner gate + "
        "_pi_prime_dw = compute_exner_perturbation(...) missing — "
        "post-acoustic damp_w site no longer respects the flag."
    )

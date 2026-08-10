"""Gates for ``scripts/validate/check_ifs_supersaturation_cap.py``.

The checker is an INSTRUMENT: if it cannot fail, its "all assertions hold"
output is worthless.  These tests exercise its own comparison logic and prove
non-vacuity by breaking the thing it checks and demanding a failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ifs_cap_checker",
        REPO_ROOT / "scripts" / "validate" / "check_ifs_supersaturation_cap.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load()


def test_independent_transcription_is_not_the_shipped_helper(checker):
    """The section-1 comparison is only meaningful if the two sides are two
    implementations.  Guard against someone 'simplifying' the reference into a
    call to the code under test."""
    src = (REPO_ROOT / "scripts" / "validate"
           / "check_ifs_supersaturation_cap.py").read_text()
    body = src.split("def _gsam_rh_homo")[1].split("\ndef ")[0]
    assert "homogeneous_freezing_rh_factor" not in body


def test_gsam_transcription_matches_published_anchors(checker):
    """2.583 - T/207.8, floored at 1, gated at 235 K (gSAM cloud.f90)."""
    T = np.array([190.0, 215.0, 234.9, 235.0, 260.0])
    got = checker._gsam_rh_homo(T)
    assert got[0] == pytest.approx(2.583 - 190.0 / 207.8)
    assert got[1] == pytest.approx(2.583 - 215.0 / 207.8)
    assert got[2] == pytest.approx(2.583 - 234.9 / 207.8)
    assert got[3] == 1.0        # gate is strict <235
    assert got[4] == 1.0
    assert np.all(got >= 1.0)


def test_qci_gate_withdraws_the_allowance(checker):
    T = np.full(5, 200.0)
    q_ice = np.array([0.0, 1e-12, 1e-9, 1e-8, 1e-6])
    got = checker._gsam_rh_homo(T, q_ice)
    # < 1e-8 keeps the allowance; >= 1e-8 withdraws it.
    assert np.all(got[:3] > 1.0)
    assert got[3] == 1.0
    assert got[4] == 1.0


def test_section_1_passes_on_the_shipped_helper(checker):
    assert checker.section_1_formula() == []


def test_section_1_fails_when_the_shipped_helper_is_wrong(checker, monkeypatch):
    """Synthetic violation: perturb the helper and demand section 1 notices."""
    import legoesm.thermo as thermo

    def _wrong(T, q_ice=None, *, enabled=True, q_ice_threshold=1.0e-8):
        return jnp.ones_like(jnp.asarray(T)) * 1.25   # a plausible-looking cap

    monkeypatch.setattr(thermo, "homogeneous_freezing_rh_factor", _wrong)
    monkeypatch.setattr(checker.thermo, "homogeneous_freezing_rh_factor", _wrong)
    failures = checker.section_1_formula()
    assert failures, "section 1 accepted a wrong ramp"


def test_column_helper_builds_the_backend_shapes(checker):
    q_v, (T, hyd, p_full, p_half, rho, dz) = checker._column(215.0, 20_000.0, 1e-4)
    assert q_v.shape == T.shape == p_full.shape == rho.shape == dz.shape == (1, 1)
    assert p_half.shape == (1, 2)
    assert hyd.q_c.shape == (1, 1)
    # rho from the ideal gas law on the SAME T/p the caller asked for.
    from legoesm import constants
    assert float(rho[0, 0]) == pytest.approx(
        20_000.0 / (constants.R_d * 215.0), rel=1e-12)


def test_section_2_reports_both_flags_as_load_bearing(checker):
    """Not a physics claim — only that neither flag is inert, which is what
    makes the lane table in section 3 non-vacuous."""
    assert checker.section_2_load_bearing() == []


def test_section_3_finds_the_cap_live_in_all_three_lanes(checker):
    assert checker.section_3_reachable() == []

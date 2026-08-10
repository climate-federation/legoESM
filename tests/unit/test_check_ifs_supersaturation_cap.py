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


def test_section_2_covers_every_ice_scheme(checker):
    """A cap that reaches morrison and nothing else must not pass: section 3
    only reads config fields, so an ignored flag in thompson or p3 would be
    invisible without this."""
    assert set(checker.ICE_SCHEMES) == {"morrison", "thompson", "p3"}
    seen = []
    real = checker._on_off

    def _spy(scheme, field, q_v, rest):
        seen.append((scheme, field))
        return real(scheme, field, q_v, rest)

    checker._on_off = _spy
    try:
        checker.section_2_load_bearing()
    finally:
        checker._on_off = real
    for scheme in checker.ICE_SCHEMES:
        assert (scheme, checker.ICE_FIELD) in seen, (scheme, seen)
        assert (scheme, checker.LIQUID_FIELD) in seen, (scheme, seen)


def test_section_2_fails_when_a_scheme_ignores_the_flag(checker):
    """SYNTHETIC VIOLATION: a scheme whose ON and OFF tendencies are identical
    is exactly the 'flag declared but never read' defect section 2 exists to
    catch."""
    real = checker._on_off

    def _inert(scheme, field, q_v, rest):
        on, off = real(scheme, field, q_v, rest)
        return (off, off) if scheme == "p3" else (on, off)


    checker._on_off = _inert
    try:
        failures = checker.section_2_load_bearing()
    finally:
        checker._on_off = real
    assert any("p3" in f for f in failures), failures


def test_section_2_fails_on_a_barely_nonzero_liquid_effect(checker):
    """The LIQUID half keeps a relative floor (its effect is 36 % of the warm-
    cell tendency), so a 1e-9 relative difference must be rejected."""
    real = checker._on_off

    def _noise(scheme, field, q_v, rest):
        on, off = real(scheme, field, q_v, rest)
        if field == checker.LIQUID_FIELD and off != 0.0:
            return off * (1.0 - 1.0e-9), off
        return on, off

    checker._on_off = _noise
    try:
        failures = checker.section_2_load_bearing()
    finally:
        checker._on_off = real
    assert failures, "a 1e-9 relative difference was accepted as load-bearing"
    assert all("floor" in f for f in failures), failures


def test_section_2_fails_when_the_allowance_survives_its_own_gate(checker):
    """SYNTHETIC VIOLATION and the reason the ice half needs no magnitude
    threshold: an implementation that applies the ramp UNCONDITIONALLY still
    produces a difference below the qci gate, and is caught only by the
    controls above the gate and above 235 K."""
    real = checker._on_off
    calls = {"n": 0}

    def _always_on(scheme, field, q_v, rest):
        on, off = real(scheme, field, q_v, rest)
        if field == checker.ICE_FIELD:
            calls["n"] += 1
            # every ice call, control or not, reports a difference
            return off * 0.5, off
        return on, off

    checker._on_off = _always_on
    try:
        failures = checker.section_2_load_bearing()
    finally:
        checker._on_off = real
    assert calls["n"] >= 3 * len(checker.ICE_SCHEMES), calls
    assert failures, "an unconditional allowance passed the gate controls"
    assert any("withdrawal gate" in f or "ramp cutoff" in f
               for f in failures), failures


def test_section_3_finds_the_cap_live_in_all_three_lanes(checker):
    assert checker.section_3_reachable() == []


def test_section_3_fails_when_a_lane_drops_the_requested_guard(checker):
    """SYNTHETIC VIOLATION: a lane that accepts --hard-saturation-adjustment
    and hands the scheme a config with it still False."""
    real_lanes = checker.LANES
    label, fn = real_lanes[1]          # the SCM lane

    def _drops_guard(scheme, *, guard=False):
        return fn(scheme, guard=False)

    checker.LANES = (real_lanes[0], (label, _drops_guard), real_lanes[2])
    try:
        failures = checker.section_3_reachable()
    finally:
        checker.LANES = real_lanes
    assert failures, "a lane that silently dropped the guard passed"
    assert all("flag is dropped" in f for f in failures), failures


def test_section_3_fails_when_a_lane_turns_the_ice_allowance_off(checker):
    """SYNTHETIC VIOLATION: the ice allowance ships ON; a lane that disables it
    changes the physics of every cold cloud and must be reported."""
    real_lanes = checker.LANES
    label, fn = real_lanes[0]          # the CRM lane

    def _ice_off(scheme, *, guard=False):
        sub = fn(scheme, guard=guard)
        return sub._replace(**{checker.ICE_FIELD: False})

    checker.LANES = ((label, _ice_off),) + real_lanes[1:]
    try:
        failures = checker.section_3_reachable()
    finally:
        checker.LANES = real_lanes
    assert failures, "a lane with the ice allowance disabled passed"
    assert all(checker.ICE_FIELD in f for f in failures), failures

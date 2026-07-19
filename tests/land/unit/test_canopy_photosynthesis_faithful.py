"""Farquhar-von Caemmerer-Berry (FvCB) photosynthesis ORACLE-FAITHFULNESS tests.

``canopy/photosynthesis`` implements the canonical FvCB C3 + Collatz C4 leaf
biochemistry (Bonan 2019 ch. 11 / CLM5 §2.9).  The existing
test_canopy_photosynthesis.py is behavioral (An>0 in light, ==0 in dark, Vcmax
peaks near 25 C, Jmax bound, C4 constants, mixing) — it never pins the Ac/Aj/Ap/
J/co-limitation closed FORMS against an independent reimplementation.

These pin the FvCB ALGEBRA to round-off (rel 1e-9) against an independent scalar
oracle (reusing the shared JAX temperature helpers, see below):
  * C3 Rubisco-limited  Ac = Vcmax (Ci-Gamma*) / (Ci + Kc (1 + O/Ko))   [11.28]
  * C3 light-limited    Aj = (J/4)(Ci-Gamma*) / (Ci + 2 Gamma*)          [11.29]
  * J = smaller root of theta_j J^2 - (I_PSII+Jmax) J + I_PSII Jmax       [11.23]
  * product-limited Ap = 0.5 Vcmax + the two smaller-root co-limitations  [11.32-33]
  * C4 Ac=Vcmax(T), Aj=alpha APAR, Ap=kp Ci + fH/fL deactivation          [11.69-71]

The kinetic T-responses (Kc/Ko/Gamma*/Vcmax/Jmax peaked-Arrhenius) come from the
already-pinned leaf_biophysics Bernacchi/Kattge-Knorr block (test_leaf_biophysics),
reused here as upstream inputs so the pin isolates the FvCB structure proper.  The
Vcmax/Jmax activation energies (which live in THIS module, not leaf_biophysics)
are transcribed as oracle LITERALS and canaried separately.  Non-vacuity: a
monkeypatched-O2 competitive-inhibition canary and the Ci=Gamma* compensation
point.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants                                       # noqa: E402
from legoesm.land.leaf_biophysics import (                         # noqa: E402
    KC25_UMOL_MOL, KO25_UMOL_MOL, GAMMA_STAR25_UMOL_MOL, O2_UMOL_MOL,
    HA_KC, HA_KO, HA_GAMMA, arrhenius_factor, peaked_arrhenius_factor,
)
from legoesm.land.canopy import photosynthesis as photo            # noqa: E402
from legoesm.land.canopy.photosynthesis import (                   # noqa: E402
    c3_assimilation, c4_assimilation,
    _PHI_PSII, _THETA_J, _THETA_CJA_C3, _THETA_IP_C3,
    _HA_VCMAX, _HA_JMAX, _HD_VCMAX, _HD_JMAX,
    _DS_VCMAX_INTERCEPT, _DS_VCMAX_SLOPE, _DS_JMAX_INTERCEPT, _DS_JMAX_SLOPE,
    _JV_RATIO_INTERCEPT, _JV_RATIO_SLOPE,
)

# Vcmax/Jmax T-response activation energies (Kattge & Knorr 2007 / CLM5 Table
# 2.9.2), transcribed as INDEPENDENT oracle literals (the module's _HA_*/_HD_*
# are canaried against these in test_fvcb_constants_match_bonan_clm5).
_O_HA_VCMAX, _O_HA_JMAX = 72000.0, 50000.0
_O_HD_VCMAX = _O_HD_JMAX = 200000.0

# FvCB PSII quantum yield + the three colimitation-quadratic curvatures (Bonan /
# CLM5 §2.9), transcribed as INDEPENDENT oracle literals so the oracle is NOT
# sourced from the SUT.  The module's _PHI_PSII/_THETA_J/_THETA_CJA_C3/
# _THETA_IP_C3 are canaried against these in test_fvcb_constants_match_bonan_clm5.
_O_PHI_PSII = 0.85
_O_THETA_J = 0.7
_O_THETA_CJA_C3 = 0.98
_O_THETA_IP_C3 = 0.95


def _smaller_root(theta, A, B):
    """Independent smaller root of theta x^2 - (A+B) x + A B = 0."""
    s = A + B
    return (s - math.sqrt(max(s * s - 4.0 * theta * A * B, 1e-12))) / (2.0 * theta)


def _fvcb_c3_oracle(Tf, Ci, APAR, Vcmax25, TgC):
    """Independent FvCB C3 (Bonan ch. 11): returns (a_gross, Rd).

    The kinetic T-responses (Kc/Ko/Gamma*/Vcmax/Jmax) are taken from the pinned
    leaf_biophysics Arrhenius helpers (shared upstream, NOT the FvCB algebra
    under test); everything else — I_PSII, the J quadratic, Ac/Aj/Ap, the two
    co-limitations, and the Atkin Rd — is reimplemented here from Bonan ch. 11.
    """
    TgC_a = min(max(TgC, 11.0), 35.0)
    dS_v = 668.39 - 1.07 * TgC_a
    dS_j = 659.70 - 0.75 * TgC_a
    Jmax25 = (2.59 - 0.035 * TgC_a) * Vcmax25
    # Upstream (pinned) T-responses.
    Kc = float(KC25_UMOL_MOL) * float(arrhenius_factor(jnp.array(Tf), HA_KC))
    Ko = float(KO25_UMOL_MOL) * float(arrhenius_factor(jnp.array(Tf), HA_KO))
    Gs = float(GAMMA_STAR25_UMOL_MOL) * float(arrhenius_factor(jnp.array(Tf), HA_GAMMA))
    Vcmax = Vcmax25 * float(peaked_arrhenius_factor(jnp.array(Tf), _O_HA_VCMAX, _O_HD_VCMAX, dS_v))
    Jmax = Jmax25 * float(peaked_arrhenius_factor(jnp.array(Tf), _O_HA_JMAX, _O_HD_JMAX, dS_j))
    # Atkin (2008) dark respiration, reimplemented.
    T_C = min(max(Tf - float(constants.T_freeze), 5.0), 45.0)
    Q10 = 3.22 - 0.046 * T_C
    Rd = 10.0 ** (-0.00794 * (TgC_a - 25.0)) * (0.015 * Vcmax25) * Q10 ** ((T_C - 25.0) / 10.0)
    # FvCB algebra (Bonan eq. 11.23-11.33).
    I_PSII = 0.5 * _O_PHI_PSII * APAR
    J = _smaller_root(_O_THETA_J, I_PSII, Jmax)
    Ci_s = max(Ci, 1e-3)
    Ac = max(Vcmax * (Ci_s - Gs) / (Ci_s + Kc * (1.0 + float(O2_UMOL_MOL) / Ko)), 0.0)
    Aj = max((J / 4.0) * (Ci_s - Gs) / (Ci_s + 2.0 * Gs), 0.0)
    Ap = 0.5 * Vcmax
    Ai = _smaller_root(_O_THETA_CJA_C3, Ac, Aj)
    A = _smaller_root(_O_THETA_IP_C3, Ai, Ap)
    return A, Rd


# --- C3 full closed-form pin ---------------------------------------------------


@pytest.mark.parametrize("Tf,Ci,APAR,Vcmax25,TgC", [
    (298.15, 280.0, 1500.0, 60.0, 20.0),   # ambient, light-saturated
    (293.15, 150.0, 800.0, 45.0, 15.0),    # Rubisco-leaning (low Ci)
    (303.15, 400.0, 400.0, 70.0, 25.0),    # light-leaning (high Ci, low PAR)
    (288.15, 250.0, 2000.0, 50.0, 11.0),   # cool growth acclimation edge
])
def test_c3_assimilation_matches_fvcb_oracle(Tf, Ci, APAR, Vcmax25, TgC):
    """c3_assimilation a_gross + Rd match the independent Bonan ch. 11 FvCB
    reimplementation to round-off (Ac/Aj/Ap, J quadratic, both co-limitations)."""
    a_o, rd_o = _fvcb_c3_oracle(Tf, Ci, APAR, Vcmax25, TgC)
    r = c3_assimilation(jnp.array(Tf), jnp.array(Ci), jnp.array(APAR),
                        jnp.array(Vcmax25), jnp.array(TgC))
    assert float(r.a_gross) == pytest.approx(a_o, rel=1e-9, abs=0.0)
    assert float(r.rd) == pytest.approx(rd_o, rel=1e-9, abs=0.0)


def test_c3_rubisco_michaelis_o2_is_live(monkeypatch):
    """Non-vacuity of Ac = Vcmax(Ci-Gs)/(Ci + Kc(1+O/Ko)): halving the module O2
    (photo._OI) directly in a Rubisco-limited state RAISES the PRODUCTION
    a_gross — proving c3_assimilation actually uses the competitive-inhibition
    1+O/Ko denominator, not just Vcmax(Ci-Gs)."""
    # Rubisco-limited state: low Ci, saturating light.
    args = (jnp.array(298.15), jnp.array(150.0), jnp.array(2000.0),
            jnp.array(60.0), jnp.array(20.0))
    a_full = float(c3_assimilation(*args).a_gross)
    monkeypatch.setattr(photo, "_OI", photo._OI * 0.5)   # restored after the test
    a_half_o2 = float(c3_assimilation(*args).a_gross)
    assert a_half_o2 > a_full * 1.001            # less O2 competition -> higher Ac


def test_c3_compensation_point_zero_gross_at_ci_equals_gamma_star():
    """At Ci = Gamma* both Ac and Aj numerators (Ci-Gamma*) vanish, so a_gross
    collapses to zero — down to the smaller-root sqrt(eps=1e-12) co-limitation
    guard (|A| ~ sqrt(1e-12)/(2 theta) < 1e-6).  The UN-CLIPPED net A-Rd is then
    ~ -Rd (the public c3_photosynthesis floors this to An=0).  The production
    a_gross matches the independent oracle (same guard) to within abs=1e-12."""
    Tf, APAR, Vcmax25, TgC = 298.15, 1500.0, 60.0, 20.0
    Gs = float(photo.co2_compensation_point(jnp.array(Tf)))
    r = c3_assimilation(jnp.array(Tf), jnp.array(Gs), jnp.array(APAR),
                        jnp.array(Vcmax25), jnp.array(TgC))
    a_o, _ = _fvcb_c3_oracle(Tf, Gs, APAR, Vcmax25, TgC)
    assert float(r.a_gross) == pytest.approx(a_o, rel=0.0, abs=1e-12)   # within the shared eps guard
    assert abs(float(r.a_gross)) < 1e-6                        # gross ~0 at Ci=Gamma*
    assert float(r.rd) > 0.0
    # The public net rate floors the ~ -Rd un-clipped value to exactly 0.
    An = float(photo.c3_photosynthesis(
        jnp.array(Tf), jnp.array(Gs), jnp.array(APAR), jnp.array(Vcmax25),
        jnp.array(101325.0), jnp.array(0.3), jnp.array(TgC)))
    assert An == 0.0


# --- C4 (Collatz 1992) closed-form pin -----------------------------------------


def _fvcb_c4_oracle(Tf, Ci, APAR, Vcmax25):
    """Independent Collatz (1992)/Bonan §11.7 C4: (a_gross, Rd)."""
    item = (Tf - float(constants.T_freeze) - 25.0) / 10.0   # T_ref = 25 degC
    q10 = 2.0 ** item
    fH = 1.0 + math.exp(0.3 * (Tf - 313.15))
    fL = 1.0 + math.exp(0.2 * (288.15 - Tf))
    Vcmax = Vcmax25 * q10 / (fH * fL)
    Rd = (0.025 * Vcmax25) * q10 / (1.0 + math.exp(1.3 * (Tf - 328.15)))
    kp = (0.02 * Vcmax25) * q10
    Ac = Vcmax
    Aj = 0.05 * APAR
    Ap = kp * Ci
    Ai = _smaller_root(0.80, Ac, Aj)
    A = _smaller_root(0.95, Ai, Ap)
    return A, Rd


@pytest.mark.parametrize("Tf,Ci,APAR,Vcmax25", [
    (303.15, 150.0, 1500.0, 40.0),
    (298.15, 200.0, 800.0, 35.0),
    (310.15, 120.0, 1800.0, 50.0),
])
def test_c4_assimilation_matches_collatz_oracle(Tf, Ci, APAR, Vcmax25):
    """c4_assimilation a_gross + Rd match the independent Collatz/Bonan §11.7
    reimplementation (Q10, fH/fL deactivation, Ac=Vcmax/Aj=alpha APAR/Ap=kp Ci)."""
    a_o, rd_o = _fvcb_c4_oracle(Tf, Ci, APAR, Vcmax25)
    r = c4_assimilation(jnp.array(Tf), jnp.array(Ci), jnp.array(APAR),
                        jnp.array(Vcmax25))
    assert float(r.a_gross) == pytest.approx(a_o, rel=1e-9, abs=0.0)
    assert float(r.rd) == pytest.approx(rd_o, rel=1e-9, abs=0.0)


# --- FvCB constant canaries ----------------------------------------------------


def test_fvcb_constants_match_bonan_clm5():
    """The FvCB colimitation curvatures + PSII quantum yield equal both the
    independent oracle literals (_O_*, used to compute the oracle above) and
    their Bonan/CLM5 values — closing the loop so the oracle is not sourced from
    the SUT (I_PSII = 0.5 phi_PSII APAR; theta_j/theta_cja/theta_ip)."""
    assert _PHI_PSII == _O_PHI_PSII == 0.85
    assert _THETA_J == _O_THETA_J == 0.7
    assert _THETA_CJA_C3 == _O_THETA_CJA_C3 == 0.98
    assert _THETA_IP_C3 == _O_THETA_IP_C3 == 0.95
    assert photo._ALPHA_C4 == 0.05      # C4 quantum yield (Bonan §11.7)
    assert photo._KP25_FRAC == 0.02     # kp25/Vcmax25 (PEP carboxylase)


def test_vcmax_jmax_activation_energies_match_oracle_literals():
    """Every module Vcmax/Jmax T-response constant that the oracle above supplies
    as a LITERAL is canaried against the module value, closing the loop so the
    oracle's T-response is not circular with the SUT: the peaked-Arrhenius
    activation/deactivation energies (via _O_HA_*/_O_HD_*, Kattge & Knorr 2007 /
    CLM5 Table 2.9.2) AND the Kattge & Knorr acclimation coefficients (dS_v =
    668.39 - 1.07 TgC, dS_j = 659.70 - 0.75 TgC, Jmax25/Vcmax25 = 2.59 - 0.035 TgC)
    that the oracle hardcodes."""
    assert _HA_VCMAX == _O_HA_VCMAX == 72000.0
    assert _HA_JMAX == _O_HA_JMAX == 50000.0
    assert _HD_VCMAX == _O_HD_VCMAX == 200000.0
    assert _HD_JMAX == _O_HD_JMAX == 200000.0
    # Kattge & Knorr (2007) acclimation coefficients (oracle-hardcoded literals).
    assert _DS_VCMAX_INTERCEPT == 668.39 and _DS_VCMAX_SLOPE == 1.07
    assert _DS_JMAX_INTERCEPT == 659.70 and _DS_JMAX_SLOPE == 0.75
    assert _JV_RATIO_INTERCEPT == 2.59 and _JV_RATIO_SLOPE == 0.035


# --- AD-safety -----------------------------------------------------------------


def test_c3_c4_grad_finite_incl_compensation_point():
    """grad of a_gross wrt EACH of (Vcmax25, Ci, APAR) is finite for C3 —
    including at Ci=Gamma* (Ac/Aj numerator zero) and Ci=0 (Ci_safe guard) — and
    wrt (Vcmax25, Ci, APAR) for C4.  The smaller-root sqrt eps-guard and the
    max(.,0) floors must not NaN the VJP."""
    Tf, APAR, Vcmax25, TgC = 298.15, 1500.0, 60.0, 20.0
    Gs = float(photo.co2_compensation_point(jnp.array(Tf)))
    for Ci in (0.0, Gs, 280.0):
        gv = jax.grad(lambda v: c3_assimilation(
            jnp.array(Tf), jnp.array(Ci), jnp.array(APAR), v, jnp.array(TgC)).a_gross)(jnp.array(Vcmax25))
        gc = jax.grad(lambda c: c3_assimilation(
            jnp.array(Tf), c, jnp.array(APAR), jnp.array(Vcmax25), jnp.array(TgC)).a_gross)(jnp.array(Ci))
        ga = jax.grad(lambda a: c3_assimilation(
            jnp.array(Tf), jnp.array(Ci), a, jnp.array(Vcmax25), jnp.array(TgC)).a_gross)(jnp.array(APAR))
        assert all(bool(jnp.isfinite(g)) for g in (gv, gc, ga)), (Ci, gv, gc, ga)
    # C4: grad wrt each of Vcmax25, Ci (product-limited kp Ci), APAR (alpha APAR).
    c4 = lambda t, ci, ap, v: c4_assimilation(t, ci, ap, v).a_gross
    g4v = jax.grad(lambda v: c4(jnp.array(305.0), jnp.array(150.0), jnp.array(1500.0), v))(jnp.array(40.0))
    g4c = jax.grad(lambda c: c4(jnp.array(305.0), c, jnp.array(1500.0), jnp.array(40.0)))(jnp.array(150.0))
    g4a = jax.grad(lambda a: c4(jnp.array(305.0), jnp.array(150.0), a, jnp.array(40.0)))(jnp.array(1500.0))
    assert all(bool(jnp.isfinite(g)) for g in (g4v, g4c, g4a)), (g4v, g4c, g4a)


def test_c3_c4_grad_finite_float32():
    """Same VJP-finiteness at the compensation point + Ci=0 in float32 — the
    module and every other test force x64, which would mask a float32-only
    NaN-gradient trap (underflowing sqrt/power in the smaller-root eps guard or
    the peaked-Arrhenius exp).  Exercised explicitly with x64 OFF."""
    prev_x64 = jax.config.read("jax_enable_x64")   # restore entry state, not a hardcoded True
    jax.config.update("jax_enable_x64", False)
    try:
        Tf, APAR, Vcmax25, TgC = 298.15, 1500.0, 60.0, 20.0
        Gs = float(photo.co2_compensation_point(jnp.array(Tf, jnp.float32)))
        for Ci in (0.0, Gs, 280.0):
            gv = jax.grad(lambda v: c3_assimilation(
                jnp.array(Tf, jnp.float32), jnp.array(Ci, jnp.float32),
                jnp.array(APAR, jnp.float32), v, jnp.array(TgC, jnp.float32)
            ).a_gross)(jnp.array(Vcmax25, jnp.float32))
            gc = jax.grad(lambda c: c3_assimilation(
                jnp.array(Tf, jnp.float32), c, jnp.array(APAR, jnp.float32),
                jnp.array(Vcmax25, jnp.float32), jnp.array(TgC, jnp.float32)
            ).a_gross)(jnp.array(Ci, jnp.float32))
            assert bool(jnp.isfinite(gv)) and bool(jnp.isfinite(gc)), (Ci, gv, gc)
        g4 = jax.grad(lambda v: c4_assimilation(
            jnp.array(305.0, jnp.float32), jnp.array(150.0, jnp.float32),
            jnp.array(1500.0, jnp.float32), v).a_gross)(jnp.array(40.0, jnp.float32))
        assert bool(jnp.isfinite(g4)), g4
    finally:
        jax.config.update("jax_enable_x64", prev_x64)

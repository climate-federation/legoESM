"""Faithfulness pins for the MOST surface-layer stability functions psi_m/psi_h.

Target: ``psi_m`` / ``psi_h`` (with the selectable stable-regime
``stability_scheme``) and ``psi_m_coare`` / ``psi_h_coare`` in
``legoesm.core.bulk_flux``.

Most-trustful sources
---------------------
- Businger et al. (1971) / Dyer (1974): the unstable (zeta<0) branch, shared by
  every scheme:  psi_m = 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 atan(x) + pi/2 with
  x=(1-16 zeta)^{1/4};  psi_h = 2 ln((1+y)/2) with y=(1-16 zeta)^{1/2}.
- Dyer (1974) stable: psi = -5 zeta (historical linear default).
- Beljaars & Holtslag (1991), J. Appl. Meteorol. 30, 327-341: stable land form.
- Grachev et al. (2007), Boundary-Layer Meteorol. 124, 315-333: SHEBA strong-
  stable form (Eqs. 9/12 momentum, 10/13 heat).
- Gryanik et al. (2020), J. Atmos. Sci. 77, 2687-2716: modified-SHEBA form.
- Fairall et al. (2003) COARE 3.0, J. Climate 16, 571-591: the COARE psi.

The empirical fit coefficients below are typed directly from those papers (they
are scheme parameters, not physical constants); they equal the module's
documented, CliMA-cross-checked values.

Certification (test-only)
-------------------------
1. Closed-form pins of psi_m/psi_h for each stability_scheme + COARE, to
   round-off vs an independent reimplementation typed from the reference.
2. ANTI-CIRCULARITY: for the three complex stable schemes the integrated psi_m
   is cross-checked against the NUMERICAL integral of the paper's dimensionless
   gradient phi_m (psi = int_0^zeta (1-phi)/x dx) -- phi is a different, simpler
   primitive, so a wrong analytic integration in the module is caught even
   though the closed-form pin re-derives the same expression.
3. Physical canaries: neutral continuity psi(0)=0; stable-branch suppression
   (psi<0 for zeta>0); the near-neutral slope dpsi_m/dzeta|_{0+} = -a_m from the
   flux-profile relation; the +/-10 zeta clip; dispatch hardening on an unknown
   scheme; finite, nonzero gradients on both branches.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.bulk_flux import (
    psi_h,
    psi_h_coare,
    psi_m,
    psi_m_coare,
    validate_stability_scheme,
)

jax.config.update("jax_enable_x64", True)

# --- published fit coefficients (typed from the references above) ---
_BETA = 5.0                                   # Dyer (1974) linear stable slope
_BH_A, _BH_B, _BH_C, _BH_D = 1.0, 0.667, 5.0, 0.35        # Beljaars-Holtslag 1991
_GRY_A_M, _GRY_B_M = 5.0, 0.3                 # Gryanik 2020 momentum
_GRY_A_H, _GRY_B_H, _GRY_PR0 = 5.0, 0.4, 0.98            # Gryanik 2020 heat
_GRA_A_M, _GRA_B_M = 5.0, 5.0 / 6.5           # Grachev 2007 momentum (b_m=a_m/6.5)

_SCHEMES = ("dyer1974", "beljaars_holtslag1991", "grachev2007_sheba", "gryanik2020")


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _close(got, ref, rtol=1e-12, atol=0.0):
    np.testing.assert_allclose(np.asarray(got), ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Independent oracles typed from the papers.
# ---------------------------------------------------------------------------
def _bd_psi_m(z):
    x = (1.0 - 16.0 * z) ** 0.25
    return (2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x**2) / 2.0)
            - 2.0 * np.arctan(x) + np.pi / 2.0)


def _bd_psi_h(z):
    y = (1.0 - 16.0 * z) ** 0.5
    return 2.0 * np.log((1.0 + y) / 2.0)


def _gryanik_psi_m(z):
    return -3.0 * (_GRY_A_M / _GRY_B_M) * (np.cbrt(1.0 + _GRY_B_M * z) - 1.0)


def _gryanik_psi_h(z):
    return -_GRY_PR0 * (_GRY_A_H / _GRY_B_H) * np.log1p(_GRY_B_H * z)


def _bh_psi_m(z):
    return -(_BH_A * z + _BH_B * (z - _BH_C / _BH_D) * np.exp(-_BH_D * z) + _BH_B * _BH_C / _BH_D)


def _bh_psi_h(z):
    return -((1.0 + 2.0 * _BH_A * z / 3.0) ** 1.5
             + _BH_B * (z - _BH_C / _BH_D) * np.exp(-_BH_D * z)
             + _BH_B * _BH_C / _BH_D - 1.0)


def _coare_psi_m(z):
    z = float(z)
    if z < 0.0:
        x = (1.0 - 15.0 * z) ** 0.25
        pk = (2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x**2) / 2.0)
              - 2.0 * np.arctan(x) + np.pi / 2.0)
        y = (1.0 - 10.15 * z) ** (1.0 / 3.0)
        s3 = math.sqrt(3.0)
        pc = (1.5 * np.log((1.0 + y + y * y) / 3.0)
              - s3 * np.arctan((1.0 + 2.0 * y) / s3) + np.pi / s3)
        f = z * z / (1.0 + z * z)
        return (1.0 - f) * pk + f * pc
    c = min(50.0, 0.35 * z)
    return -((1.0 + z) + 0.6667 * (z - 14.28) * np.exp(-c) + 8.525)


def _coare_psi_h(z):
    z = float(z)
    if z < 0.0:
        x = (1.0 - 15.0 * z) ** 0.5
        pk = 2.0 * np.log((1.0 + x) / 2.0)
        y = (1.0 - 34.15 * z) ** (1.0 / 3.0)
        s3 = math.sqrt(3.0)
        pc = (1.5 * np.log((1.0 + y + y * y) / 3.0)
              - s3 * np.arctan((1.0 + 2.0 * y) / s3) + np.pi / s3)
        f = z * z / (1.0 + z * z)
        return (1.0 - f) * pk + f * pc
    c = min(50.0, 0.35 * z)
    return -((1.0 + 2.0 * z / 3.0) ** 1.5 + 0.6667 * (z - 14.28) * np.exp(-c) + 8.525)


# Dimensionless gradient phi_m typed from the papers (the primitive of psi_m);
# psi_m(zeta) = int_0^zeta (1 - phi_m(x))/x dx.
def _gryanik_phi_m(z):
    return 1.0 + _GRY_A_M * z / (1.0 + _GRY_B_M * z) ** (2.0 / 3.0)


def _bh_phi_m(z):
    return 1.0 + _BH_A * z + _BH_B * z * (1.0 + _BH_C - _BH_D * z) * np.exp(-_BH_D * z)


def _grachev_phi_m(z):
    return 1.0 + _GRA_A_M * z * (1.0 + z) ** (1.0 / 3.0) / (1.0 + _GRA_B_M * z)


def _grachev_phi_h(z):
    # Grachev et al. (2007) Eq. 10: phi_h = 1 + (a_h z + b_h z^2)/(1 + c_h z + z^2),
    # a_h=5, b_h=5, c_h=3 (phi_h(0)=1, no neutral-Prandtl prefactor).
    return 1.0 + (5.0 * z + 5.0 * z * z) / (1.0 + 3.0 * z + z * z)


def _psi_from_phi(phi, zt, n=200000):
    """psi(zt) = int_0^zt (1 - phi(x))/x dx by the trapezoid rule."""
    xs = np.linspace(1e-11, zt, n)
    return float(np.trapezoid((1.0 - phi(xs)) / xs, xs))


_ZU = np.array([-3.0, -1.0, -0.3, -0.05])          # unstable (zeta < 0)
# The near-neutral 1e-8 point (> the module's 1e-10 stable floor) makes the
# closed-form pins sensitive to a log1p-vs-log(1+x) precision bug in psi_h.
_ZS = np.array([1e-8, 0.05, 0.5, 2.0, 8.0])        # stable (zeta > 0)


# ---------------------------------------------------------------------------
# 1. Unstable branch (Businger-Dyer) -- shared by every scheme.
# ---------------------------------------------------------------------------
def test_unstable_businger_dyer_shared_all_schemes():
    ref_m = _bd_psi_m(_ZU)
    ref_h = _bd_psi_h(_ZU)
    for sc in _SCHEMES:
        np.testing.assert_allclose(np.asarray(psi_m(_a(_ZU), sc)), ref_m, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(psi_h(_a(_ZU), sc)), ref_h, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Stable closed forms.
# ---------------------------------------------------------------------------
def test_dyer1974_stable_linear():
    np.testing.assert_allclose(np.asarray(psi_m(_a(_ZS), "dyer1974")), -_BETA * _ZS, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(psi_h(_a(_ZS), "dyer1974")), -_BETA * _ZS, rtol=1e-12)


def test_gryanik2020_stable_closed_form():
    _close(psi_m(_a(_ZS), "gryanik2020"), _gryanik_psi_m(_ZS))
    _close(psi_h(_a(_ZS), "gryanik2020"), _gryanik_psi_h(_ZS))


def test_beljaars_holtslag1991_stable_closed_form():
    _close(psi_m(_a(_ZS), "beljaars_holtslag1991"), _bh_psi_m(_ZS))
    _close(psi_h(_a(_ZS), "beljaars_holtslag1991"), _bh_psi_h(_ZS))


def test_coare_psi_both_branches():
    z = np.concatenate([_ZU, _ZS])
    ref_m = np.array([_coare_psi_m(v) for v in z])
    ref_h = np.array([_coare_psi_h(v) for v in z])
    _close(psi_m_coare(_a(z)), ref_m)
    _close(psi_h_coare(_a(z)), ref_h)


# ---------------------------------------------------------------------------
# 3. Anti-circularity: psi_m == numerical integral of the paper's phi_m.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scheme, phi",
    [
        ("gryanik2020", _gryanik_phi_m),
        ("beljaars_holtslag1991", _bh_phi_m),
        ("grachev2007_sheba", _grachev_phi_m),
    ],
)
def test_stable_psi_m_matches_phi_integral(scheme, phi):
    # The integrated closed-form psi_m must equal the numerical integral of the
    # independently-typed dimensionless gradient phi_m -- catches a wrong analytic
    # integration even where the closed-form pin re-derives the same expression.
    for zt in (0.3, 1.0, 4.0):
        got = float(psi_m(_a([zt]), scheme)[0])
        ref = _psi_from_phi(phi, zt)
        np.testing.assert_allclose(got, ref, rtol=1e-6)


def test_grachev_psi_h_matches_phi_integral():
    # Grachev's integrated psi_h is too intricate to re-type non-circularly; pin it
    # QUANTITATIVELY against the numerical integral of the paper's Eq. 10 phi_h
    # (a different primitive) -- this catches a wrong a_h/b_h/c_h or a doubled psi_h.
    for zt in (0.3, 1.0, 4.0):
        got = float(psi_h(_a([zt]), "grachev2007_sheba")[0])
        ref = _psi_from_phi(_grachev_phi_h, zt)
        np.testing.assert_allclose(got, ref, rtol=1e-6)


def test_grachev_stable_monotone_and_negative():
    ph = np.asarray(psi_h(_a(_ZS), "grachev2007_sheba"))
    pm = np.asarray(psi_m(_a(_ZS), "grachev2007_sheba"))
    assert np.all(ph < 0.0) and np.all(pm < 0.0)              # stable suppression
    assert np.all(np.diff(ph) < 0.0) and np.all(np.diff(pm) < 0.0)   # monotone decreasing


# ---------------------------------------------------------------------------
# 4. Cross-scheme physical canaries.
# ---------------------------------------------------------------------------
def test_neutral_continuity_all_schemes():
    # At zeta=0 the stable branch is evaluated at the 1e-10 floor, so psi ~ -5e-10.
    # The tight 1e-9 tolerance would fail if that safety floor were materially
    # enlarged (a looser atol would silently accept a broken floor).
    for sc in _SCHEMES:
        np.testing.assert_allclose(float(psi_m(_a([0.0]), sc)[0]), 0.0, atol=1e-9)
        np.testing.assert_allclose(float(psi_h(_a([0.0]), sc)[0]), 0.0, atol=1e-9)


def test_stable_branch_suppresses_flux_all_schemes():
    # zeta>0 must give psi<0 (raises the log-law denominator ln(z/z0)-psi -> lower
    # exchange coefficient); a sign flip would spuriously ENHANCE stable fluxes.
    for sc in _SCHEMES:
        assert np.all(np.asarray(psi_m(_a(_ZS), sc)) < 0.0)
        assert np.all(np.asarray(psi_h(_a(_ZS), sc)) < 0.0)


def test_near_neutral_slope_matches_a_m():
    # Flux-profile relation: near zeta=0, phi_m ~ 1 + a_m zeta so psi_m ~ -a_m zeta.
    # Gryanik/Grachev/Dyer have a_m=5; BH91's slope is -(a+b(1+c))=-5.002.
    expect = {"dyer1974": -5.0, "gryanik2020": -5.0, "grachev2007_sheba": -5.0,
              "beljaars_holtslag1991": -(_BH_A + _BH_B * (1.0 + _BH_C))}
    for sc, want in expect.items():
        slope = float(jax.grad(lambda z: psi_m(_a([z]), sc)[0])(1e-8))
        np.testing.assert_allclose(slope, want, rtol=1e-5)


def test_zeta_clip_saturates():
    # zeta is clipped to [-10, 10]; beyond that psi is constant. Cover psi_m AND
    # psi_h for every selectable scheme, plus both COARE functions (a dropped clip
    # in any one would otherwise pass unnoticed).
    for sc in _SCHEMES:
        for fn in (psi_m, psi_h):
            _close(float(fn(_a([50.0]), sc)[0]), float(fn(_a([10.0]), sc)[0]))
            _close(float(fn(_a([-50.0]), sc)[0]), float(fn(_a([-10.0]), sc)[0]))
    for fn in (psi_m_coare, psi_h_coare):
        _close(float(fn(_a([50.0]))[0]), float(fn(_a([10.0]))[0]))
        _close(float(fn(_a([-50.0]))[0]), float(fn(_a([-10.0]))[0]))


# ---------------------------------------------------------------------------
# 5. Dispatch hardening.
# ---------------------------------------------------------------------------
def test_dispatch_hardening_unknown_scheme_raises():
    with pytest.raises((ValueError, KeyError)):
        validate_stability_scheme("not_a_scheme")
    with pytest.raises((ValueError, KeyError)):
        psi_m(_a([1.0]), "bogus_scheme")
    with pytest.raises((ValueError, KeyError)):
        psi_h(_a([1.0]), "bogus_scheme")


# ---------------------------------------------------------------------------
# 6. Differentiability (finite AND nonzero on both branches, every scheme).
# ---------------------------------------------------------------------------
def test_differentiable_both_branches_all_schemes():
    for sc in _SCHEMES:
        for z0 in (-1.5, 1.5):
            gm = float(jax.grad(lambda z: psi_m(_a([z]), sc)[0])(z0))
            gh = float(jax.grad(lambda z: psi_h(_a([z]), sc)[0])(z0))
            assert np.isfinite(gm) and abs(gm) > 0.0
            assert np.isfinite(gh) and abs(gh) > 0.0

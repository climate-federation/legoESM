"""P3 / warm-rain ACCRETION oracle-faithfulness tests.

The accretion rate wired into P3 (and Morrison's non-KK2000 path, Seifert-Beheng,
Thompson) is the simplified bilinear SURROGATE :func:`accretion`::

    dq_c_ac = k_ac * q_c * q_r * rho * gamma_norm        (k_ac default 5.25)

It is NOT faithful to the published Khairoutdinov-Kogan (2000) accretion used by
gSAM P3 and SAM-M2005.  The gSAM P3 oracle (``module_mp_p3.f90:3615``, the
default iparam=3 / KK2000 path; in a single column with the sub-grid cloud/precip
fractions off, ``SCF=iSCF=SPF=iSPF=1`` and ``dum2=SPF-SPF_clr=1``) is::

    qcacc = 67 * (q_c * iSCF * q_r * iSPF)^1.15 * dum2  ->  67 * (q_c*q_r)^1.15

a pure mixing-ratio power law (verified against the on-disk Fortran; identical to
the SAM-M2005 form ``module_mp_graupel.f90:1952``).  That faithful form exists in
this repo as :func:`accretion_kk2000` and is pinned here against the P3 oracle.

DEPARTURES of the wired surrogate from the KK2000 oracle, each canaried below:
  (1) coefficient ``k_ac`` (5.25) vs the published 67;
  (2) BILINEAR ``q_c*q_r`` (product exponent 1) vs KK2000 ``(q_c*q_r)^1.15``;
  (3) a spurious ``rho`` factor — KK2000 has NO density dependence;
  (4) an optional ``gamma_norm`` PSD-shape modifier absent from KK2000 (P3 passes
      the default 1.0, so P3 itself does not incur (4) — verified end-to-end).

The faithful :func:`accretion_kk2000` reproduced here is the SINGLE-COLUMN,
fractions-off MASS-rate reduction of the Fortran — the algebraic iparam=3 branch
ABOVE its ``qc,qr >= qsmall=1e-14`` gate.  DEPARTURES vs the full Fortran:
(a) it OMITS the qsmall gate, returning a positive rate below 1e-14 where the
Fortran returns 0 — CANARIED in ``test_accretion_kk2000_omits_qsmall_gate``;
(b) the number tie ``ncacc = qcacc*nc/qc`` is the caller's responsibility and is
NOT reproduced or pinned by this helper (a scope statement, not a canaried
departure).  ``safe_pow`` clips negatives and keeps the fractional power AD-safe
(at ``q_c*q_r=0`` the ``^1.15`` branch already has a finite ZERO slope —
``d/dx x^1.15 = 1.15*x^0.15 -> 0`` — so it is not a singular-slope fix).

The end-to-end pin drives ``p3_microphysics`` and, by DIFFERENCING the cloud-water
tendency over two rain values in a warm ice-free column with autoconversion off,
isolates the accretion contribution and shows P3 uses the surrogate (linear in
q_r, proportional to rho, gamma_norm=1), not the KK2000 1.15-power.  The
isolation preconditions (no ice tendency, condensation is a source so the
cloud-evaporation sink is zero, and the accretion sink is far below the
donor-limiter threshold q_c/dt) are ASSERTED, not assumed.

Precision: the warm-rain helpers carry no precision policy; x64 is forced at
import (mirroring test_p3_cooper_faithful.py) for the round-off pins.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    accretion,
    accretion_kk2000,
)
from legoesm.atmosphere.physics.microphysics.config import P3Config
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.thermo import saturation_mixing_ratio

jax.config.update("jax_enable_x64", True)

# gSAM P3 KK2000 accretion oracle constants (module_mp_p3.f90:3615), typed from
# the Fortran, NOT the module — identical to SAM-M2005 module_mp_graupel.f90:1952.
_O_KK2000_ACC_COEFF = 67.0
_O_KK2000_ACC_EXP = 1.15
_O_K_AC_DEFAULT = 5.25          # P3Config.k_ac default (surrogate coefficient)
_O_QSMALL = 1.0e-14            # Fortran accretion gate qc,qr >= qsmall (module_mp_p3.f90)


# --- surrogate FORM pinned to round-off -------------------------------------

def test_accretion_surrogate_form_matches_reimpl():
    q_c = jnp.array([1.0e-3, 2.0e-3, 0.5e-3])
    q_r = jnp.array([2.0e-4, 1.0e-4, 3.0e-4])
    rho = jnp.array([1.0, 0.9, 1.1])
    k_ac = _O_K_AC_DEFAULT
    rate = np.asarray(accretion(q_c, q_r, rho, k_ac))
    expected = k_ac * np.asarray(q_c) * np.asarray(q_r) * np.asarray(rho)  # bilinear
    np.testing.assert_allclose(rate, expected, rtol=1e-12, atol=0.0)


def test_accretion_negative_inputs_clip_to_zero():
    # q_c<0 or q_r<0 -> clipped to 0 (no spurious negative accretion).
    assert float(accretion(jnp.array(-1.0e-3), jnp.array(2.0e-4),
                           jnp.array(1.0), _O_K_AC_DEFAULT)) == 0.0
    assert float(accretion(jnp.array(1.0e-3), jnp.array(-2.0e-4),
                           jnp.array(1.0), _O_K_AC_DEFAULT)) == 0.0


def test_accretion_gamma_norm_is_a_departure_from_kk2000():
    """DEPARTURE (4): the generic helper multiplies by a ``gamma_norm`` PSD-shape
    modifier absent from KK2000 — a non-unit value rescales the rate linearly.
    P3 passes the default 1.0 (so P3 does not incur this departure; the
    end-to-end pin below confirms the P3 contribution carries no gamma factor)."""
    args = (jnp.array(1.0e-3), jnp.array(2.0e-4), jnp.array(1.0), _O_K_AC_DEFAULT)
    base = float(accretion(*args))
    assert float(accretion(*args, gamma_norm=2.5)) == pytest.approx(2.5 * base, rel=1e-12)
    assert float(accretion(*args, gamma_norm=1.0)) == pytest.approx(base, rel=1e-12)


# --- DEPARTURE canaries vs the KK2000 oracle --------------------------------

def _kk2000_oracle(q_c, q_r):
    return _O_KK2000_ACC_COEFF * (np.asarray(q_c) * np.asarray(q_r)) ** _O_KK2000_ACC_EXP


def test_departure_bilinear_vs_kk2000_power():
    """Scaling q_c*q_r by 2 scales the BILINEAR surrogate by exactly 2, but the
    KK2000 oracle by 2^1.15 — the product-exponent departure."""
    q_c, q_r, rho = jnp.array(1.0e-3), jnp.array(2.0e-4), jnp.array(1.0)
    s1 = float(accretion(q_c, q_r, rho, _O_K_AC_DEFAULT))
    s2 = float(accretion(q_c, 2.0 * q_r, rho, _O_K_AC_DEFAULT))
    assert s2 / s1 == pytest.approx(2.0, rel=1e-12)              # surrogate: linear
    o1, o2 = float(_kk2000_oracle(q_c, q_r)), float(_kk2000_oracle(q_c, 2.0 * q_r))
    assert o2 / o1 == pytest.approx(2.0 ** _O_KK2000_ACC_EXP, rel=1e-12)   # oracle: ^1.15
    assert not np.isclose(2.0, 2.0 ** _O_KK2000_ACC_EXP, rtol=1e-3)


def test_departure_spurious_rho_factor():
    """The surrogate is proportional to rho (2x rho -> 2x rate); the KK2000 oracle
    is a pure mixing-ratio rate, INDEPENDENT of rho."""
    q_c, q_r = jnp.array(1.0e-3), jnp.array(2.0e-4)
    r1 = float(accretion(q_c, q_r, jnp.array(1.0), _O_K_AC_DEFAULT))
    r2 = float(accretion(q_c, q_r, jnp.array(2.0), _O_K_AC_DEFAULT))
    assert r2 / r1 == pytest.approx(2.0, rel=1e-12)             # surrogate ∝ rho
    # oracle has no rho argument at all — its rate is unchanged by density.


def test_departure_coefficient():
    """At a reference state (rho=1) the surrogate coefficient 5.25 differs from the
    published 67, and the combined form differs materially from the oracle."""
    assert P3Config().k_ac == _O_K_AC_DEFAULT
    assert P3Config().k_ac != _O_KK2000_ACC_COEFF
    q_c, q_r = jnp.array(1.0e-3), jnp.array(2.0e-4)
    surrogate = float(accretion(q_c, q_r, jnp.array(1.0), _O_K_AC_DEFAULT))
    oracle = float(_kk2000_oracle(q_c, q_r))
    assert not np.isclose(surrogate, oracle, rtol=1e-2)


# --- the FAITHFUL form pinned to the P3 oracle ------------------------------

def test_accretion_kk2000_matches_p3_oracle():
    """accretion_kk2000 == 67*(q_c*q_r)^1.15 (module_mp_p3.f90:3615) to round-off,
    across a range of states — a cross-source check that the repo's faithful form
    matches the on-disk P3 Fortran (identical to SAM-M2005:1952)."""
    q_c = jnp.array([1.0e-3, 2.0e-3, 5.0e-4, 3.0e-3])
    q_r = jnp.array([2.0e-4, 1.0e-4, 4.0e-4, 5.0e-5])
    got = np.asarray(accretion_kk2000(q_c, q_r))
    np.testing.assert_allclose(got, _kk2000_oracle(q_c, q_r), rtol=1e-12, atol=0.0)
    # coefficient + exponent canaries (a wrong power would fail the ratio test).
    r = float(accretion_kk2000(jnp.array(1.0e-3), jnp.array(4.0e-4))) / \
        float(accretion_kk2000(jnp.array(1.0e-3), jnp.array(2.0e-4)))
    assert r == pytest.approx(2.0 ** _O_KK2000_ACC_EXP, rel=1e-12)


def test_accretion_kk2000_is_rho_independent():
    """The faithful form takes NO rho argument — a pure mixing-ratio rate (the
    defining contrast with the surrogate's spurious rho factor)."""
    import inspect
    assert "rho" not in inspect.signature(accretion_kk2000).parameters


def test_accretion_kk2000_omits_qsmall_gate():
    """DEPARTURE (a): the Fortran gates accretion on qc,qr >= qsmall=1e-14 and
    returns 0 below it; accretion_kk2000 omits the gate, so for a tiny-but-positive
    input it returns a POSITIVE rate where the gated oracle returns exactly 0."""
    q_c, q_r = jnp.array(1.0e-15), jnp.array(2.0e-4)          # q_c below qsmall
    helper = float(accretion_kk2000(q_c, q_r))
    assert helper > 0.0                                      # helper: no gate
    gated_oracle = 0.0 if (1.0e-15 < _O_QSMALL or 2.0e-4 < _O_QSMALL) \
        else float(_kk2000_oracle(q_c, q_r))
    assert gated_oracle == 0.0                               # Fortran: gated to 0
    assert helper != gated_oracle                            # the documented departure


# --- end-to-end: P3 wires the SURROGATE -------------------------------------

def _warm_column(q_r_val, q_c_val=2.0e-3, rho_val=1.0):
    """Warm (T=290 K, ice-free), supersaturated single column."""
    ncol, nlev = 1, 1
    z = jnp.zeros((ncol, nlev))
    T = jnp.full((ncol, nlev), 290.0)
    q_v = jnp.full((ncol, nlev), 2.0e-2)             # supersaturated -> condensation source
    p_full = jnp.full((ncol, nlev), 7.0e4)
    p_half = jnp.full((ncol, nlev + 1), 7.0e4)
    rho = jnp.full((ncol, nlev), rho_val)
    dz = jnp.full((ncol, nlev), 200.0)
    hydro = HydrometeorState(
        q_c=jnp.full((ncol, nlev), q_c_val), q_r=jnp.full((ncol, nlev), q_r_val),
        q_i=z, q_s=z, q_g=z,
        N_c=jnp.full((ncol, nlev), 1.0e8), N_r=jnp.full((ncol, nlev), 1.0e6),
        N_i=z,
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


def _assert_isolation_preconditions(cfg, q_c, q_r_max, rho, dt):
    """The Δdq_c_dt = -Δdq_c_ac identity holds only if (asserted, not assumed):
    autoconversion is off; condensation is a SOURCE (cloud-evaporation sink 0);
    the accretion sink is below the donor limiter q_c/dt so qc_scale==1; and the
    ice phase does not touch dq_c (riming/deposition ∝ input q_i=N_i=0)."""
    assert cfg.k_au == 0.0                                    # no autoconversion
    # Supersaturated at T=290 K, p=7e4 Pa -> condensation source, cond_evap_sink=0.
    q_sat = float(saturation_mixing_ratio(jnp.array(290.0), jnp.array(7.0e4)))
    assert 2.0e-2 > q_sat                                     # q_v (=2e-2) > q_sat
    # Largest accretion sink far below the donor-limiter threshold q_c/dt -> qc_scale=1.
    assert cfg.k_ac * q_c * q_r_max * rho * dt < 0.05 * q_c


def test_p3_accretion_contribution_is_the_surrogate():
    """DIFFERENCE dq_c_dt over two rain values in a warm ice-free column with
    autoconversion off: only accretion depends on q_r, so
    Δdq_c_dt = -Δdq_c_ac = -k_ac*q_c*rho*Δq_r — the SURROGATE (linear in q_r,
    proportional to rho), NOT the KK2000 1.15-power."""
    cfg = P3Config()._replace(k_au=0.0)              # kill autoconversion
    dt, q_c, rho = 1.0, 2.0e-3, 1.0
    q_r1, q_r2 = 1.0e-4, 2.0e-4
    _assert_isolation_preconditions(cfg, q_c, q_r2, rho, dt)
    out1 = p3_microphysics(*_warm_column(q_r1, q_c, rho), dt, cfg)
    out2 = p3_microphysics(*_warm_column(q_r2, q_c, rho), dt, cfg)
    # The ice tendency is q_r-INDEPENDENT (nucleation ∝ T only; riming/deposition
    # ∝ input N_i=q_i=0), so it is identical in both states and cancels in the
    # difference — it cannot contaminate the isolated accretion contribution.
    assert float(out1.dq_i_dt[0, 0]) == float(out2.dq_i_dt[0, 0])
    d_dqc = float(out2.dq_c_dt[0, 0] - out1.dq_c_dt[0, 0])
    expected = -cfg.k_ac * q_c * rho * (q_r2 - q_r1)  # surrogate accretion contribution
    assert d_dqc == pytest.approx(expected, rel=1e-9)
    # KK2000 would give -(67*(q_c*q_r2)^1.15 - 67*(q_c*q_r1)^1.15): materially different.
    kk = -(float(_kk2000_oracle(q_c, q_r2)) - float(_kk2000_oracle(q_c, q_r1)))
    assert not np.isclose(d_dqc, kk, rtol=1e-2)


def test_p3_accretion_contribution_scales_with_rho():
    """The isolated accretion contribution scales with rho (surrogate ∝ rho),
    confirming the spurious density factor is live in the wired P3 path."""
    cfg = P3Config()._replace(k_au=0.0)
    dt, q_c = 1.0, 2.0e-3
    q_r1, q_r2 = 1.0e-4, 2.0e-4

    def iso(rho_val):
        o1 = p3_microphysics(*_warm_column(q_r1, q_c, rho_val), dt, cfg)
        o2 = p3_microphysics(*_warm_column(q_r2, q_c, rho_val), dt, cfg)
        return float(o2.dq_c_dt[0, 0] - o1.dq_c_dt[0, 0])

    assert iso(2.0) / iso(1.0) == pytest.approx(2.0, rel=1e-9)


# --- differentiability ------------------------------------------------------

def test_accretion_grads_finite():
    def total(k):
        return jnp.sum(accretion(jnp.array([1.0e-3]), jnp.array([2.0e-4]),
                                 jnp.array([1.0]), k))
    g = jax.grad(total)(_O_K_AC_DEFAULT)
    assert bool(jnp.isfinite(g)) and float(g) > 0.0
    # faithful form's grad at q_r=0 is finite (the ^1.15 branch has zero slope
    # there: 1.15*x^0.15 -> 0; safe_pow additionally guards negative inputs).
    gk = jax.grad(lambda q: accretion_kk2000(jnp.array(1.0e-3), q))(jnp.array(0.0))
    assert bool(jnp.isfinite(gk))

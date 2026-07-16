"""VP/EVP/mEVP sea-ice rheology ORACLE-FAITHFULNESS tests.

``ice/rheology`` implements the Hibler (1979) viscous-plastic constitutive law
with the Hunke-Dukowicz (1997) EVP and Bouillon (2013)/Kimmritz (2015) mEVP
stress relaxations.  The existing ice tests are behavioral: they pin only the
rest state (sigma = -P/2 I) and signs for vp_stress, the Hibler form at two
(h, A) points for ice_strength, and a grad-finite / rtol-1e-3 convergence check
for the relaxations — never the general constitutive FORMS against an
independent reimplementation.

These pin the rheology ALGEBRA to round-off (rel 1e-9) against an independent
scalar oracle:
  * ice_strength      P = P* h exp(-C(1-A))                       (Hibler 1979)
  * delta_deformation Delta = sqrt(max(div^2 + shear/e^2, dmin^2))(Hibler 1979)
  * vp_stress         zeta=P/2Delta, eta=zeta/e^2, sigma_ij=...    (Hibler/HD97)
  * evp_stress_update sigma_new=(sigma+E sigma_vp)/(1+E)           (HD97)
  * mevp_stress_update sigma^(p+1)=(1-1/a)sigma^p+(1/a)sigma_vp    (Bouillon/Kimmritz)

plus the DEFINING VP yield-curve identity (stress lies on the ellipse), the
sigma=-P/2 I rest state, the EVP/mEVP fixed-point convergence to sigma_vp, and
the zero-strain gradient (the AD-safe Delta_min sqrt-ARGUMENT floor).

Independence: the P*/C/e/Delta_min/T_evp/N_evp/alpha coefficients are local _O_*
literals canaried against the SeaIceConfig defaults (config == _O_* == value).
Departures (the Delta_min sqrt-arg floor, the mEVP alpha<1 guard) are reproduced
/ probed by dedicated canaries.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the rel-1e-9 pins; restore the process-entry state in
    finally so selecting a single test never leaks x64 into another module."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


from legoesm.ice.rheology import (                                   # noqa: E402
    ice_strength, delta_deformation, vp_stress,
    evp_stress_update, mevp_stress_update,
)
from legoesm.ice.config import SeaIceConfig                          # noqa: E402

# Independent oracle literals (canaried in test_rheology_constants_match_cice).
_O_P_STAR = 2.75e4       # [N/m^2] Hibler ice-strength parameter (CICE)
_O_C_STRENGTH = 20.0     # Hibler strength concentration decay
_O_E_YIELD = 2.0         # yield-curve eccentricity
_O_DELTA_MIN = 2.0e-9    # [1/s] deformation-rate regulariser
_O_T_EVP = 0.36          # EVP damping ratio (CICE default)
_O_N_EVP = 120           # EVP subcycle count
_O_ALPHA_MEVP = 500.0    # mEVP relaxation parameter (CICE/FESOM)


def _ice_strength_oracle(h, A, Pstar, C):
    return Pstar * h * math.exp(-C * (1.0 - A))


def _delta_oracle(e11, e22, e12, e, dmin):
    div = e11 + e22
    shear = (e11 - e22) ** 2 + 4.0 * e12 ** 2
    return math.sqrt(max(div ** 2 + shear / e ** 2, dmin ** 2))


def _vp_stress_oracle(e11, e22, e12, P, Delta, e):
    zeta = P / (2.0 * Delta)
    eta = zeta / e ** 2
    tr = e11 + e22
    s11 = 2.0 * eta * e11 + (zeta - eta) * tr - P / 2.0
    s22 = 2.0 * eta * e22 + (zeta - eta) * tr - P / 2.0
    s12 = 2.0 * eta * e12
    return s11, s22, s12


def _evp_oracle(s, svp, T_evp, N_evp):
    E = 1.0 / (2.0 * T_evp * float(N_evp))
    return (s + E * svp) / (1.0 + E)


def _mevp_oracle(s, svp, alpha):
    ia = 1.0 / alpha
    return (1.0 - ia) * s + ia * svp


# Test ice strength, from the ORACLE (not production ice_strength) so the whole
# VP/EVP/mEVP oracle chain is independent of the SUT; ice_strength itself is
# pinned separately in test_ice_strength_matches_hibler_oracle.
_P_TEST = _ice_strength_oracle(2.0, 0.95, _O_P_STAR, _O_C_STRENGTH)


def _delta_of(e11, e22, e12):
    return _delta_oracle(e11, e22, e12, _O_E_YIELD, _O_DELTA_MIN)


def _a(x):
    return jnp.array(float(x))


# --- ice_strength (Hibler 1979) ------------------------------------------------

@pytest.mark.parametrize("h,A", [
    (2.0, 1.0),     # A=1 -> P = P* h exp(0)
    (1.5, 0.9),     # typical pack
    (3.0, 0.5),     # low concentration -> strong exp decay
    (0.3, 0.15),    # thin marginal ice
])
def test_ice_strength_matches_hibler_oracle(h, A):
    """ice_strength matches P = P* h exp(-C(1-A)) to round-off over a general
    (h, A) sweep (the existing suite pins only two points)."""
    got = float(ice_strength(_a(h), _a(A)))
    exp = _ice_strength_oracle(h, A, _O_P_STAR, _O_C_STRENGTH)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


# --- delta_deformation (Hibler 1979) -------------------------------------------

@pytest.mark.parametrize("e11,e22,e12", [
    (1e-6, -5e-7, 3e-7),     # general
    (2e-6, 2e-6, 0.0),       # pure divergence -> Delta = |div|
    (1e-6, -1e-6, 0.0),      # pure shear (no div) -> Delta = sqrt(shear)/e
    (0.0, 0.0, 0.0),         # zero strain -> Delta = Delta_min (floor engages)
])
def test_delta_deformation_matches_oracle(e11, e22, e12):
    """delta_deformation matches the Hibler invariant (incl the Delta_min floor
    at zero strain) to round-off."""
    got = float(delta_deformation(_a(e11), _a(e22), _a(e12)))
    exp = _delta_oracle(e11, e22, e12, _O_E_YIELD, _O_DELTA_MIN)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


def test_delta_pure_divergence_and_shear_limits():
    """Non-vacuity: pure divergence gives Delta=|e11+e22|; pure shear gives
    Delta=sqrt((e11-e22)^2+4e12^2)/e — the two invariant limits are distinct."""
    d_div = float(delta_deformation(_a(2e-6), _a(2e-6), _a(0.0)))
    assert d_div == pytest.approx(4e-6, rel=1e-9, abs=0.0)        # |div|
    d_shr = float(delta_deformation(_a(1e-6), _a(-1e-6), _a(0.0)))
    assert d_shr == pytest.approx(math.sqrt(4e-12) / _O_E_YIELD, rel=1e-9, abs=0.0)  # sqrt(shear)/e


# --- vp_stress (Hibler 1979 / Hunke-Dukowicz 1997) -----------------------------

@pytest.mark.parametrize("e11,e22,e12", [
    (1e-6, -5e-7, 3e-7),
    (2e-6, 1e-6, -8e-7),
    (-1e-6, -2e-6, 5e-7),
])
def test_vp_stress_matches_oracle(e11, e22, e12):
    """All three vp_stress components match the VP constitutive oracle
    (zeta=P/2Delta, eta=zeta/e^2) to round-off.  P and Delta are ORACLE-computed
    (independent of the SUT) and fed as inputs to both sides."""
    P = _P_TEST
    Delta = _delta_of(e11, e22, e12)
    s = vp_stress(_a(e11), _a(e22), _a(e12), _a(P), _a(Delta))
    exp = _vp_stress_oracle(e11, e22, e12, P, Delta, _O_E_YIELD)
    for g, e in zip(s, exp):
        assert float(g) == pytest.approx(e, rel=1e-9, abs=0.0)


def test_vp_stress_lies_on_yield_ellipse():
    """The DEFINING VP property: with an un-regularized Delta the stress lies
    EXACTLY on the yield ellipse
        ((sigma_I + P/2)/(P/2))^2 + (sigma_II e/(P/2))^2 = 1,
    sigma_I = (s11+s22)/2 (mean), sigma_II = sqrt(((s11-s22)/2)^2 + s12^2).
    This is what makes it a plastic yield curve — never checked today."""
    e11, e22, e12 = 3e-5, -1e-5, 2e-5           # large strain -> Delta >> Delta_min
    P = _P_TEST
    Delta = _delta_of(e11, e22, e12)
    assert Delta > _O_DELTA_MIN * 1e3            # genuinely un-regularized
    s11, s22, s12 = (float(x) for x in vp_stress(_a(e11), _a(e22), _a(e12),
                                                 _a(P), _a(Delta)))
    sigma_I = 0.5 * (s11 + s22)
    sigma_II = math.sqrt((0.5 * (s11 - s22)) ** 2 + s12 ** 2)
    ellipse = ((sigma_I + P / 2.0) / (P / 2.0)) ** 2 \
        + (sigma_II * _O_E_YIELD / (P / 2.0)) ** 2
    assert ellipse == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_vp_stress_rest_state_is_isotropic_pressure():
    """At zero strain the viscous terms vanish and sigma = -P/2 I (isotropic
    compressive pressure), independent of the (floored) Delta."""
    P = _P_TEST
    Delta = _delta_of(0.0, 0.0, 0.0)            # = Delta_min (floor)
    s11, s22, s12 = (float(x) for x in vp_stress(_a(0.0), _a(0.0), _a(0.0),
                                                 _a(P), _a(Delta)))
    assert s11 == pytest.approx(-P / 2.0, rel=1e-9, abs=0.0)
    assert s22 == pytest.approx(-P / 2.0, rel=1e-9, abs=0.0)
    assert s12 == 0.0


# --- EVP / mEVP relaxation -----------------------------------------------------

_STRAIN = (1e-6, -5e-7, 3e-7)


def _vp_target(P):
    Delta = _delta_of(*_STRAIN)                 # independent oracle Delta
    return _vp_stress_oracle(*_STRAIN, P, Delta, _O_E_YIELD)


def test_evp_update_matches_hd97_oracle():
    """One EVP subcycle matches sigma_new=(sigma+E sigma_vp)/(1+E),
    E=1/(2 T_evp N_evp), to round-off."""
    P = _P_TEST
    s0 = (100.0, -50.0, 20.0)
    out = evp_stress_update(*[_a(x) for x in s0], *[_a(x) for x in _STRAIN],
                            _a(P), _O_E_YIELD, _O_T_EVP, 1.0, _O_N_EVP)
    svp = _vp_target(P)
    for g, s_old, s_vp in zip(out, s0, svp):
        assert float(g) == pytest.approx(_evp_oracle(s_old, s_vp, _O_T_EVP, _O_N_EVP),
                                         rel=1e-9, abs=0.0)


def test_mevp_update_matches_bouillon_kimmritz_oracle():
    """One mEVP pseudo-step matches sigma^(p+1)=(1-1/a)sigma^p+(1/a)sigma_vp."""
    P = _P_TEST
    s0 = (100.0, -50.0, 20.0)
    out = mevp_stress_update(*[_a(x) for x in s0], *[_a(x) for x in _STRAIN],
                             _a(P), _O_E_YIELD, _O_ALPHA_MEVP)
    svp = _vp_target(P)
    for g, s_old, s_vp in zip(out, s0, svp):
        assert float(g) == pytest.approx(_mevp_oracle(s_old, s_vp, _O_ALPHA_MEVP),
                                         rel=1e-9, abs=0.0)


@pytest.mark.parametrize("scheme", ["evp", "mevp"])
def test_relaxation_converges_to_vp_fixed_point(scheme):
    """Non-vacuity: iterating the relaxation with a FIXED strain drives the
    stress to the sigma_vp fixed point (the whole purpose of the subcycle), and
    the sigma_vp target itself is the pinned VP oracle.  Run inside a jitted
    fori_loop over the REAL production update (mEVP's 0.998 factor needs many
    iterations to converge)."""
    P = _a(_P_TEST)
    strain = tuple(_a(x) for x in _STRAIN)

    def body(_, s):
        if scheme == "evp":
            return tuple(evp_stress_update(*s, *strain, P, _O_E_YIELD,
                                           _O_T_EVP, 1.0, _O_N_EVP))
        return tuple(mevp_stress_update(*s, *strain, P, _O_E_YIELD, _O_ALPHA_MEVP))

    s0 = (_a(100.0), _a(-50.0), _a(20.0))
    s = jax.lax.fori_loop(0, 8000, body, s0)
    svp = _vp_target(_P_TEST)
    for g, s_vp in zip(s, svp):
        assert float(g) == pytest.approx(s_vp, rel=1e-6, abs=1e-6)


def test_mevp_alpha_below_one_raises():
    """Departure guard: mEVP with alpha<1 (anti-relaxation past the VP target,
    incl the alpha=0 divide-by-zero) raises ValueError."""
    args = (*[_a(0.0) for _ in range(3)], *[_a(x) for x in _STRAIN],
            _a(_P_TEST), _O_E_YIELD)
    for bad in (0.5, 0.0, -1.0):
        with pytest.raises(ValueError):
            mevp_stress_update(*args, bad)


# --- constant canaries ---------------------------------------------------------

def test_rheology_constants_match_cice():
    """The SeaIceConfig rheology defaults equal both the independent oracle
    literals and the published CICE/Hibler values (config == _O_* == value)."""
    c = SeaIceConfig()
    assert c.P_star == _O_P_STAR == 2.75e4
    assert c.C_strength == _O_C_STRENGTH == 20.0
    assert c.e_yield == _O_E_YIELD == 2.0
    assert c.Delta_min == _O_DELTA_MIN == 2.0e-9
    assert c.T_evp == _O_T_EVP == 0.36
    assert c.N_evp == _O_N_EVP == 120
    assert c.alpha_mevp == _O_ALPHA_MEVP == 500.0


# --- AD-safety (the Delta_min sqrt-ARGUMENT floor) -----------------------------

def test_rheology_grad_finite_at_zero_strain_x64_and_float32():
    """The documented DEPARTURE: delta_deformation floors the sqrt ARGUMENT at
    Delta_min^2 (not the result), so the VP/EVP/mEVP stress gradients stay finite
    at the zero-strain rest state — where the naive max(sqrt(max(.,0)),Delta_min)
    form NaNs the VJP (sqrt'(0)=inf * 0 selector).  Checked in x64 AND float32."""
    P = 1.0e4

    def _check():
        # delta_deformation grad wrt e11 at (0,0,0): finite (max picks the floor).
        gd = jax.grad(lambda x: delta_deformation(x, _a(0.0), _a(0.0)))(_a(0.0))
        assert bool(jnp.isfinite(gd))
        # vp_stress via delta at zero strain: grad of sigma_11 wrt e11 finite.
        def s11_of(x):
            Delta = delta_deformation(x, _a(0.0), _a(0.0))
            return vp_stress(x, _a(0.0), _a(0.0), _a(P), Delta)[0]
        assert bool(jnp.isfinite(jax.grad(s11_of)(_a(0.0))))
        # evp + mevp updates (call delta+vp internally) at zero strain.
        def evp_of(x):
            return evp_stress_update(_a(0.0), _a(0.0), _a(0.0), x, _a(0.0), _a(0.0),
                                     _a(P), _O_E_YIELD, _O_T_EVP, 1.0, _O_N_EVP)[0]
        def mevp_of(x):
            return mevp_stress_update(_a(0.0), _a(0.0), _a(0.0), x, _a(0.0), _a(0.0),
                                      _a(P), _O_E_YIELD, _O_ALPHA_MEVP)[0]
        assert bool(jnp.isfinite(jax.grad(evp_of)(_a(0.0))))
        assert bool(jnp.isfinite(jax.grad(mevp_of)(_a(0.0))))

    _check()
    jax.config.update("jax_enable_x64", False)
    _check()   # autouse fixture restores the entry state afterwards

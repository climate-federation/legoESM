"""Thompson (2008) mass-weighted SNOW FALL SPEED oracle-faithfulness tests.

``_thompson_snow.snow_fall_speed`` reproduces the gSAM/WRF Thompson ``vts``
closed form (module_mp_thompson.f90:2751-2762) — a faithful closed-form
ALGEBRA, in the active/uncapped/un-floored regime, of::

    xDs = smoc/smob                          ! M3/M2
    Mrat = 1/xDs                             ! M2/M3
    t1_vts = Kap0*csg(4) *(Mrat*Lam0+fv_s)^-cse(4)
    t2_vts = Kap1*Mrat^mu_s*csg(10)*(Mrat*Lam1+fv_s)^-cse(10)
    t3_vts = Kap0*csg(1) *(Mrat*Lam0)^-cse(1)
    t4_vts = Kap1*Mrat^mu_s*csg(7) *(Mrat*Lam1)^-cse(7)
    vts = rhof*av_s*(t1_vts+t2_vts)/(t3_vts+t4_vts)

with cse(1)=bm_s+1, cse(4)=bm_s+bv_s+1, cse(7)=bm_s+mu_s+1,
cse(10)=bm_s+mu_s+bv_s+1 (module_mp_thompson.f90:476-487), csg(i)=Γ(cse(i)),
rhof=√(rho_not/rho), rho_not=p_std/(R·298).  The surrounding JAX code adds
DELIBERATE numerical departures that are NOT the oracle (the q_s>QS_SMALL
activation gate, the 0..5 m/s speed cap, the Field-fit tc clamp, and safe_pow
floors) — those are documented in the production docstring and are NOT pinned
here; these tests pin only the ``vts`` closed form and its constants.

To make the pin INDEPENDENT of production the whole PSD chain is recomputed
here from hardcoded gSAM constants: M2 = ρ·q_s/am_s, tc = min(−0.1, T−T_freeze),
M3 via an independent Field-2005 polynomial, Mrat = M2/M3 — so a regression in
any of _snow_moments' M2 / Field coefficients / M3 / ratio also fails.

The existing tests/unit/test_thompson_snow.py checks only physical bounds
(0<V<5, monotone, density SIGN) — never the gSAM formula or its coefficients.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._thompson_snow import (
    snow_fall_speed,
    _snow_moments,
    _SA, _SB,
    _AM_S, _BM_S, _AV_S, _BV_S, _FV_S,
    _MU_S, _KAP0, _KAP1, _LAM0, _LAM1, _RHO_NOT,
)


jax.config.update("jax_enable_x64", True)

# gSAM/WRF Thompson-2008 snow constants (module_mp_thompson.f90:73-103, :129).
# Hardcoded here as the INDEPENDENT oracle values (not imported from legoESM).
_G_AM_S, _G_BM_S = 0.069, 2.0
_G_AV_S, _G_BV_S, _G_FV_S = 40.0, 0.55, 100.0
_G_MU_S = 0.6357
_G_KAP0, _G_KAP1 = 490.6, 17.46
_G_LAM0, _G_LAM1 = 20.78, 3.29
# Field et al. (2005) universal-moment coefficients (module_mp_thompson.f90:238-244).
_G_SA = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
         0.31255, 0.000204, 0.003199, 0.0, -0.015952)
_G_SB = (0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
         0.060366, 0.000079, 0.000594, 0.0, -0.003577)
# gSAM rho_not is the Fortran literal p_std/(R·298) with p_std=101325 and the
# dry-air gas constant R (module_mp_thompson.f90). Built here from those literals
# so the oracle stays INDEPENDENT of legoESM constants; that gSAM R numerically
# coincides with constants.R_d (cross-checked in test_thompson_snow_constants_match_gsam).
_G_RHO_NOT = 101325.0 / (287.05 * 298.0)  # const-ok: gSAM oracle literal, not R_d/p_ref


def _gsam_field_moment(p, m2, tc):
    """Independent NumPy Field-2005 p-th snow moment from the 2nd moment ``m2``
    and ``tc`` [°C] (module_mp_thompson.f90:238-267). ``M_p = 10^loga · m2^b``."""
    loga = (_G_SA[0] + _G_SA[1] * tc + _G_SA[2] * p + _G_SA[3] * tc * p
            + _G_SA[4] * tc * tc + _G_SA[5] * p * p + _G_SA[6] * tc * tc * p
            + _G_SA[7] * tc * p * p + _G_SA[8] * tc * tc * tc + _G_SA[9] * p * p * p)
    b = (_G_SB[0] + _G_SB[1] * tc + _G_SB[2] * p + _G_SB[3] * tc * p
         + _G_SB[4] * tc * tc + _G_SB[5] * p * p + _G_SB[6] * tc * tc * p
         + _G_SB[7] * tc * p * p + _G_SB[8] * tc * tc * tc + _G_SB[9] * p * p * p)
    return (10.0 ** loga) * m2 ** b


def _gsam_mrat(q_s, rho, T):
    """Independent gSAM snow ``Mrat = M2/M3`` (module_mp_thompson.f90 smob/smoc).

    M2 = ρ·q_s/am_s (smob), tc = min(−0.1, T−T_freeze) (gSAM clamps only the
    upper end; the production −55 °C lower clamp is inactive for the test T's),
    M3 = independent Field-2005 3rd moment (smoc), Mrat = M2/M3 = 1/xDs."""
    tc = min(-0.1, float(T) - float(constants.T_freeze))
    m2 = rho * q_s / _G_AM_S
    m3 = _gsam_field_moment(3.0, m2, tc)
    return m2 / m3


def _gsam_vts(Mrat, rho):
    """Independent NumPy transcription of gSAM ``vts``
    (module_mp_thompson.f90:2751-2762). ``Mrat`` = M2/M3."""
    cse1 = _G_BM_S + 1.0                      # cse(1)
    cse4 = _G_BM_S + _G_BV_S + 1.0            # cse(4)
    cse7 = _G_BM_S + _G_MU_S + 1.0            # cse(7)
    cse10 = _G_BM_S + _G_MU_S + _G_BV_S + 1.0  # cse(10)
    csg1, csg4, csg7, csg10 = (
        math.gamma(cse1), math.gamma(cse4), math.gamma(cse7), math.gamma(cse10))
    t1 = _G_KAP0 * csg4 * (Mrat * _G_LAM0 + _G_FV_S) ** (-cse4)
    t2 = _G_KAP1 * Mrat ** _G_MU_S * csg10 * (Mrat * _G_LAM1 + _G_FV_S) ** (-cse10)
    t3 = _G_KAP0 * csg1 * (Mrat * _G_LAM0) ** (-cse1)
    t4 = _G_KAP1 * Mrat ** _G_MU_S * csg7 * (Mrat * _G_LAM1) ** (-cse7)
    rhof = math.sqrt(_G_RHO_NOT / rho)
    return rhof * _G_AV_S * (t1 + t2) / (t3 + t4)


# --- The vts fall-speed integral matches the gSAM formula ----------------------


def test_snow_fall_speed_matches_gsam_vts_formula():
    """snow_fall_speed(q_s, rho, T) == the gSAM vts closed form to round-off,
    at physical snow amounts where V is below the 5 m/s aggregate cap. Both the
    fall-speed integral AND the PSD moment ratio are checked against a fully
    independent oracle (hardcoded sa/sb + independent Field polynomial), so this
    pins av_s/bv_s/fv_s/mu_s/Kap0/Kap1/Lam0/Lam1, the cse(1/4/7/10) gamma
    structure, AND the M2/Field/M3/ratio chain in _snow_moments."""
    for q_s, rho, T in ((1.0e-4, 0.8, 263.0), (5.0e-4, 1.0, 258.0),
                        (2.0e-4, 0.5, 250.0), (1.0e-3, 1.1, 268.0)):
        mrat = _gsam_mrat(q_s, rho, T)                     # fully independent
        expected = _gsam_vts(mrat, rho)
        got = float(snow_fall_speed(
            jnp.array(q_s), jnp.array(rho), jnp.array(T)))
        assert got == pytest.approx(expected, rel=1e-9), f"q_s={q_s} T={T}"
        assert 0.0 < got < 5.0                             # uncapped physical regime
        # Certify the production PSD ratio equals the independent oracle Mrat, so
        # the shared moment ratio is itself pinned (not merely reused).
        _, _, ratio = _snow_moments(
            jnp.array(q_s), jnp.array(rho), jnp.array(T))
        assert float(ratio) == pytest.approx(mrat, rel=1e-9), f"q_s={q_s} T={T}"


def test_snow_fall_speed_bv_s_exponent_is_load_bearing():
    """A regression to the wrong fall-speed exponent bv_s would change the
    numerator moment order p_v=bm_s+bv_s. The gSAM bv_s=0.55 gives a materially
    different answer than a spherical bv_s (e.g. 0.41) — demonstrating the pin
    is sensitive to the exponent (i.e. the formula test above is non-vacuous)."""
    q_s, rho, T = 3.0e-4, 0.9, 255.0
    mrat = _gsam_mrat(q_s, rho, T)
    v_gsam = _gsam_vts(mrat, rho)

    def vts_with_bv(bv):
        cse4 = _G_BM_S + bv + 1.0
        cse10 = _G_BM_S + _G_MU_S + bv + 1.0
        cse1, cse7 = _G_BM_S + 1.0, _G_BM_S + _G_MU_S + 1.0
        t1 = _G_KAP0 * math.gamma(cse4) * (mrat * _G_LAM0 + _G_FV_S) ** (-cse4)
        t2 = _G_KAP1 * mrat ** _G_MU_S * math.gamma(cse10) * (mrat * _G_LAM1 + _G_FV_S) ** (-cse10)
        t3 = _G_KAP0 * math.gamma(cse1) * (mrat * _G_LAM0) ** (-cse1)
        t4 = _G_KAP1 * mrat ** _G_MU_S * math.gamma(cse7) * (mrat * _G_LAM1) ** (-cse7)
        return math.sqrt(_G_RHO_NOT / rho) * _G_AV_S * (t1 + t2) / (t3 + t4)

    assert v_gsam == pytest.approx(vts_with_bv(0.55), rel=1e-12)
    assert abs(v_gsam - vts_with_bv(0.41)) / v_gsam > 0.02   # exponent matters


# --- Every Thompson-2008 snow constant matches the gSAM/WRF value ---------------


def test_thompson_snow_constants_match_gsam():
    """The mass-size, fall-speed, and PSD-shape constants equal the gSAM/WRF
    module_mp_thompson.f90 values (:73-77, :88-89, :100-102)."""
    assert _AM_S == _G_AM_S and _BM_S == _G_BM_S
    assert _AV_S == _G_AV_S and _BV_S == _G_BV_S and _FV_S == _G_FV_S
    assert _MU_S == _G_MU_S
    assert _KAP0 == _G_KAP0 and _KAP1 == _G_KAP1
    assert _LAM0 == _G_LAM0 and _LAM1 == _G_LAM1
    # Production _RHO_NOT (built from legoESM constants) equals the INDEPENDENT
    # gSAM literal oracle — certifying gSAM's R coincides with constants.R_d.
    assert _RHO_NOT == pytest.approx(_G_RHO_NOT, rel=1e-12)


def test_thompson_field2005_sa_sb_arrays_match_gsam():
    """The production Field et al. (2005) universal-moment coefficients sa/sb
    equal the independent oracle arrays (module_mp_thompson.f90:238-244)."""
    assert tuple(_SA) == _G_SA
    assert tuple(_SB) == _G_SB


# --- Density correction magnitude + AD-safety ----------------------------------


def test_snow_fall_speed_density_correction_magnitude():
    """The (rho0/rho)^1/2 correction magnitude (not just its sign) matches gSAM:
    V scales as sqrt(rho_not/rho) at the PSD ratio for each rho."""
    q_s, T = 3.0e-4, 255.0
    for rho in (0.4, 0.7, 1.0, 1.25):
        mrat = _gsam_mrat(q_s, rho, T)
        got = float(snow_fall_speed(jnp.array(q_s), jnp.array(rho), jnp.array(T)))
        assert got == pytest.approx(_gsam_vts(mrat, rho), rel=1e-9)


def test_snow_fall_speed_ad_safe():
    """jax.grad of the fall speed wrt q_s is finite in x64, including at exactly
    q_s=0 where the moments are evaluated on the floored (_QS_SMALL) input."""
    g = jax.grad(lambda q: snow_fall_speed(
        q, jnp.array(0.9), jnp.array(255.0)))(jnp.array(3.0e-4))
    assert bool(jnp.isfinite(g))
    g0 = jax.grad(lambda q: snow_fall_speed(
        q, jnp.array(0.9), jnp.array(255.0)))(jnp.array(0.0))
    assert bool(jnp.isfinite(g0))


def test_snow_fall_speed_grad_finite_float32():
    """The zero-snow gradient must be finite in float32 — the default
    finite-volume dtype. Without the floored-moment fix, den ∝ lam0^-(bm_s+1)
    underflows to +inf in f32 and the activation-gate `where` returns a NaN VJP;
    x64 masks this. This is the regression guard for that production fix."""
    _was = getattr(jax.config, "jax_enable_x64", False)
    jax.config.update("jax_enable_x64", False)
    try:
        rho = jnp.asarray(0.9, dtype=jnp.float32)
        T = jnp.asarray(255.0, dtype=jnp.float32)
        for q in (0.0, 1.0e-13, 3.0e-4):
            qf = jnp.asarray(q, dtype=jnp.float32)
            g = jax.grad(lambda x: snow_fall_speed(x, rho, T))(qf)
            assert g.dtype == jnp.float32
            assert bool(jnp.isfinite(g)), f"NaN/inf grad at q_s={q} in float32"
        # The value at zero snow is still exactly zero.
        v0 = snow_fall_speed(jnp.asarray(0.0, jnp.float32), rho, T)
        assert float(v0) == 0.0
    finally:
        jax.config.update("jax_enable_x64", _was)

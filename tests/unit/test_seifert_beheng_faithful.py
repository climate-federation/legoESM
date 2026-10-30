"""Seifert & Beheng (2001) warm-rain oracle-faithfulness tests.

Pins the PUBLISHED SB2001 universal functions ``autoconversion_sb2001`` /
``accretion_sb2001`` (``_warm_rain.py``) against the gSAM M2005 IRAIN=1 oracle
(``MICRO_M2005/module_mp_graupel.f90``)::

    autoconv (:1835-1844)   tau    = 1 − q_c/(q_c+q_r)
                            phi_au = 600·tau^0.68·(1 − tau^0.68)^3
                            PRC    = 9.44e9/(20·2.6e-7)·(nu+2)(nu+4)/(nu+1)^2
                                     ·(rho·q_c/1000)^4/(N_c/1e6)^2
                                     ·(1 + phi_au/(1−tau)^2)·1000/rho
    accretion (:1960-1962)  phi_ac = (tau/(tau+5e-4))^4
                            PRA    = 5.78e3·rho/1000·q_c·q_r·phi_ac

The gSAM ``nc3d`` is per-MASS; legoESM ``N_c`` is per-VOLUME [1/m^3], so gSAM's
``rho·nc3d/1e6`` becomes ``N_c/1e6`` (identical conversion in both the reference
and the production code — see ``autoconversion_kk2000``).

Also CANARIES the DEPARTURE of the shipped-default simplified proxies
(``autoconversion_sb``: q_c^2·sigmoid; ``accretion``: bilinear, no tau) so any
future "faithful-ization" of those flips a red test and forces the
``seifert_beheng.py`` docstring to be updated in the same change.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics._warm_rain import (
    autoconversion_sb,
    autoconversion_sb2001,
    accretion,
    accretion_sb2001,
)
from legoesm.atmosphere.physics.microphysics.config import SeifertBehengConfig


jax.config.update("jax_enable_x64", True)
_RAMP_TOP = SeifertBehengConfig._field_defaults["sb2001_autoconv_ramp_top"]
_AC_TOPS = dict(
    qr_ramp_top=SeifertBehengConfig._field_defaults["sb2001_accretion_qr_ramp_top"],
    qc_ramp_top=SeifertBehengConfig._field_defaults["sb2001_accretion_qc_ramp_top"])


# --- Independent numpy transcriptions of the gSAM IRAIN=1 oracle ---------------


def _gsam_pra_irain1(q_c, q_r, rho):
    """gSAM accretion PRA (module_mp_graupel.f90:1960-1962)."""
    tau = 1.0 - q_c / (q_c + q_r)
    phi_ac = (tau / (tau + 5.0e-4)) ** 4
    return 5.78e3 * rho / 1000.0 * q_c * q_r * phi_ac


# --- Autoconversion vs the RUNNING oracle -------------------------------------
# PRC/NPRC1/PRA with gSAM's diagnosed droplet shape are pinned against the
# compiled gSAM M2005 by tests/atmosphere/microphysics/unit/
# test_sb2001_warm_rain_oracle.py (the fixed-nu re-derivation tests that lived
# here were superseded when the default became gSAM's diagnosed shape).


# --- Accretion: matches gSAM IRAIN=1 to machine precision ----------------------


def test_accretion_sb2001_matches_gsam_irain1():
    q_c, q_r, rho = 8.0e-4, 2.0e-4, 1.1
    pra = accretion_sb2001(jnp.array(q_c), jnp.array(q_r), jnp.array(rho), **_AC_TOPS)
    assert float(pra) == pytest.approx(_gsam_pra_irain1(q_c, q_r, rho), rel=1e-9)


def test_accretion_sb2001_phi_ac_suppresses_at_low_rain_fraction():
    """phi_ac=(tau/(tau+5e-4))^4 → 0 as rain fraction tau → 0, so the faithful
    accretion is progressively suppressed relative to the tau-blind simplified
    bilinear form as q_r shrinks (the SB2001 'accretion needs rain' behaviour)."""
    q_c, rho = 1.0e-3, 1.1
    cfg = SeifertBehengConfig()
    ratios = []
    for q_r in (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7):
        faith = float(accretion_sb2001(
            jnp.array(q_c), jnp.array(q_r), jnp.array(rho), **_AC_TOPS))
        simple = float(accretion(
            jnp.array(q_c), jnp.array(q_r), jnp.array(rho), cfg.k_ac))
        ratios.append(faith / simple)
    # Strictly decreasing suppression as tau → 0, and always < the tau→1 kernel
    # ratio 5.78/k_ac.
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert ratios[0] < 5.78 / cfg.k_ac


# --- CANARY: the shipped-default proxies ARE a departure (non-vacuous) ---------


def _loglog_slope(fn, q1, q2):
    return math.log(fn(q2) / fn(q1)) / math.log(q2 / q1)


def test_simplified_autoconv_is_a_departure_qc_squared_not_qc_fourth():
    """SB2001 PRC ∝ q_c^4 (at fixed N_c, tau); the simplified ``autoconversion_sb``
    is ∝ q_c^2 in its onset-saturated regime. Pinning the two log-log slopes
    proves the proxy is a genuine (not cosmetic) departure — if someone rewires
    ``autoconversion_sb`` to the faithful form this canary goes red."""
    N_c, rho = 1.0e6, 1.1        # clean N_c so x_c ≫ x_* ⇒ simplified onset ≈ 1
    # The faithful law at 100 cm^-3: at 1 cm^-3 gSAM's droplet-slope clamp
    # resets N_c in proportion to q_c, which turns q_c^4/N_c^2 into q_c^2.
    N_c_faithful = 1.0e8
    cfg = SeifertBehengConfig()

    def faithful(q_c):
        # Hold tau fixed at 0.2 (q_r = 0.25·q_c) ⇒ only the q_c^4 factor varies.
        dq, _, _ = autoconversion_sb2001(
            jnp.array(q_c), jnp.array(0.25 * q_c),
            jnp.array(N_c_faithful), jnp.array(rho), ramp_top=_RAMP_TOP)
        return float(dq)

    def simplified(q_c):
        dq, _, _ = autoconversion_sb(
            jnp.array(q_c), jnp.array(N_c), jnp.array(rho),
            cfg.k_au, cfg.x_star, cfg.autoconversion_sharpness)
        return float(dq)

    slope_f = _loglog_slope(faithful, 2.0e-3, 4.0e-3)
    slope_s = _loglog_slope(simplified, 2.0e-3, 4.0e-3)
    assert slope_f == pytest.approx(4.0, abs=0.05)     # published q_c^4
    assert slope_s == pytest.approx(2.0, abs=0.1)      # proxy q_c^2
    assert abs(slope_f - slope_s) > 1.5                # departure is real


# --- Morrison dispatch reaches the faithful path -------------------------------


def _morrison_column(scheme, **kw):
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

    ncol, nlev = 1, 4
    T = jnp.full((ncol, nlev), 295.0)
    q_v = jnp.full((ncol, nlev), 0.012)
    p_full = jnp.full((ncol, nlev), 9.0e4)
    p_half = jnp.full((ncol, nlev + 1), 9.0e4)
    rho = jnp.full((ncol, nlev), 1.1)
    dz = jnp.full((ncol, nlev), 200.0)
    z = jnp.zeros((ncol, nlev))
    hydro = HydrometeorState(
        q_c=jnp.full((ncol, nlev), 6.0e-4), q_r=jnp.full((ncol, nlev), 1.0e-4),
        q_i=z, q_s=z, q_g=z,
        N_c=jnp.full((ncol, nlev), 1.0e8), N_r=jnp.full((ncol, nlev), 1.0e3),
        N_i=z,
    )
    return morrison_microphysics(
        T, q_v, hydro, p_full, p_half, rho, dz, 20.0,
        MorrisonConfig(warm_rain_scheme=scheme, **kw),
    )


def test_morrison_warm_rain_scheme_sb2001_selects_faithful_and_differs():
    """``warm_rain_scheme="seifert_beheng_sb2001"`` runs, stays finite, and gives
    a materially different rain tendency than kk2000 -- proving the faithful
    path is actually reached, not dead code.  ``"seifert_beheng"`` (formerly a
    proxy) now runs the SAME law: bitwise identical outputs."""
    out_faith = _morrison_column("seifert_beheng_sb2001")
    out_sb = _morrison_column("seifert_beheng")
    out_kk = _morrison_column("kk2000")
    assert bool(jnp.all(jnp.isfinite(out_faith.dq_r_dt)))
    for f in ("dT_dt", "dq_v_dt", "dq_c_dt", "dq_r_dt", "dN_r_dt"):
        assert bool(jnp.array_equal(getattr(out_faith, f), getattr(out_sb, f))), f
    assert not bool(jnp.allclose(out_faith.dq_r_dt, out_kk.dq_r_dt, rtol=1e-3))


@pytest.mark.parametrize("field,value", [
    ("k_au", 1.0e3), ("x_star", 3.0e-10), ("k_ac", 6.0),
    ("autoconversion_sharpness", 100.0)])
def test_morrison_seifert_beheng_refuses_proxy_fields(field, value):
    with pytest.raises(ValueError, match="no counterpart"):
        _morrison_column("seifert_beheng", **{field: value})


def test_morrison_warm_rain_scheme_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown warm_rain_scheme"):
        _morrison_column("seifert_beheng_typo")


# --- AD-safety at the cold-start / no-condensate corners -----------------------


def test_sb2001_warm_rain_ad_safe_at_zero():
    """jax.grad through both faithful rates stays finite at q_c=0, q_r=0 and at
    the tau=0/1 boundaries (safe_pow + floored denominators)."""
    N_c, rho = jnp.array(1.0e8), jnp.array(1.1)
    g_au_qc = jax.grad(
        lambda q: autoconversion_sb2001(q, jnp.array(1.0e-4), N_c, rho,
                                        ramp_top=_RAMP_TOP)[0]
    )(jnp.array(0.0))
    g_au_qr = jax.grad(
        lambda q: autoconversion_sb2001(jnp.array(5.0e-4), q, N_c, rho,
                                        ramp_top=_RAMP_TOP)[0]
    )(jnp.array(0.0))
    g_ac_qc = jax.grad(
        lambda q: accretion_sb2001(q, jnp.array(1.0e-4), rho, **_AC_TOPS)
    )(jnp.array(0.0))
    g_ac_qr = jax.grad(
        lambda q: accretion_sb2001(jnp.array(5.0e-4), q, rho, **_AC_TOPS)
    )(jnp.array(0.0))
    assert all(bool(jnp.isfinite(g)) for g in (g_au_qc, g_au_qr, g_ac_qc, g_ac_qr))

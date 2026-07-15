"""P3 Cooper (1986) ice-nucleation oracle-faithfulness tests.

Cooper nucleation is the ONE p3.py-native process whose closed-form base curve
has a coefficient-level gSAM P3 oracle (module_mp_p3.f90:3084-3095): every other
P3 ice rate (deposition, riming, aggregation, melting, ice fall speed) is
computed from an interpolated LOOKUP TABLE (f1pr02..f1pr14) in the Fortran, so
legoESM's algebraic surrogates for those have no coefficient-level counterpart.

Oracle (module_mp_p3.f90:3088-3092, mi0 at :233); T_f = freezing point::

    dum   = 0.005*exp(0.304*(T_f-T))*1000*inv_rho          ! Cooper base [# kg^-1]
    dum   = min(dum, 100.e3*inv_rho*SCF)                    ! scheme-1 cap 100/L·SCF
    N_nuc = max(0, (dum - sum(nitot))*odt)                  ! relaxation, odt=1/dt
    Q_nuc = max(0, (dum - sum(nitot))*mi0*odt)              ! seed-mass source
    mi0   = 4/3*pi*900*(IceNucleiRadius)^3, radius=1e-6 m   ! seed mass [kg]

These tests DRIVE ``p3_microphysics`` (the existing test_cooper_ice_base.py only
checks two config coefficients and an order-of-magnitude curve computed straight
from config — it never calls the scheme). Isolation: q_c=q_r=q_i=0 and T in the
nucleating band ⇒ aggregation ∝ N_i, melt ∝ q_i=0, deposition ∝ N_i^(1/3)·q_i,
and (with agg_coeff=dep_coeff=0) NO other term reaches dN_i_dt/dq_i_dt, so they
equal the nucleation tendencies — provided vapour is NON-LIMITING (both are
donor-scaled by qv_scale at p3.py:378-381; the tiny seed mass keeps qv_scale≈1
at the states chosen here).

The base-curve FORM (N_i0=5.0, cooper_a=0.304, per-kg rho-divide,
(target-N_i)/dt relaxation, seed-mass source) is DERIVED from gSAM's below-cap
Cooper base expression (shared by both P3 nucleation schemes) and matches it
wherever Cooper is selected, SCF=1, vapour is non-limiting, and the value is
below the cap. It is NOT a gated gSAM-OUTPUT match; the three DELTAS below are
canaried against SCHEME 1 (documented departures, not bit-identity):
(1) cap unconditional 500/L vs gSAM 100/L·SCF (scheme 1) / 150/L·SCF (scheme 2);
(2) gate -8 C smooth with NO supersaturation requirement vs gSAM -15 C hard AND
supi>=0.05; (3) seed density 917 vs 900 kg/m^3.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.p3 import (
    p3_microphysics,
    _M_I0,
)
from legoesm.atmosphere.physics.microphysics.config import P3Config
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState


jax.config.update("jax_enable_x64", True)

# gSAM P3 Cooper oracle constants (module_mp_p3.f90:3088-3092, :233).
_GSAM_COOPER_BASE = 5.0        # 0.005 /L * 1000 = 5 /m^3
_GSAM_COOPER_A = 0.304         # per K
_GSAM_CAP_S1_PER_M3 = 1.0e5    # 100.e3 = 100 /L (scheme 1, times SCF)
_GSAM_CAP_S2_PER_M3 = 1.5e5    # 150.e3 = 150 /L (scheme 2, times SCF)
_GSAM_SEED_RHO = 900.0         # kg/m^3 hardcoded in mi0
_ICE_NUCLEUS_RADIUS = 1.0e-6   # m (gSAM default IceNucleiRadius)
_GSAM_MI0 = 4.0 / 3.0 * math.pi * _GSAM_SEED_RHO * _ICE_NUCLEUS_RADIUS ** 3


def _column(T_val, rho_val=0.9, q_v_val=1.0e-4, N_i_val=0.0):
    """Isolated single-column state: no condensate, T in the nucleating band."""
    ncol, nlev = 1, 3
    z = jnp.zeros((ncol, nlev))
    T = jnp.full((ncol, nlev), T_val)
    q_v = jnp.full((ncol, nlev), q_v_val)
    p_full = jnp.full((ncol, nlev), 7.0e4)
    p_half = jnp.full((ncol, nlev + 1), 7.0e4)
    rho = jnp.full((ncol, nlev), rho_val)
    dz = jnp.full((ncol, nlev), 200.0)
    hydro = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((ncol, nlev), N_i_val),
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


def _gsam_cooper_target_per_kg(T, rho):
    """gSAM Cooper N_nuc target [1/kg] BELOW the cap: 5*exp(0.304*(T_f-T))/rho
    (T_f = constants.T_freeze). Independent of legoESM config."""
    return _GSAM_COOPER_BASE * np.exp(
        _GSAM_COOPER_A * (constants.T_freeze - T)) / rho


# --- Base-curve FORM matches the gSAM Cooper expression (below cap) ------------


def test_p3_cooper_base_curve_form_matches_gsam_expression():
    """legoESM's nucleation-number rate equals the gSAM-DERIVED base-curve form
    ``5*exp(0.304*(T_f-T))/rho / dt`` at safely-cold T where the smooth gate is
    unity to machine precision (1-f_ice < 1e-20) and vapour is non-limiting.
    This pins the FORM/coefficients, NOT a gated gSAM-output match: at these T
    gSAM's own hard T<-15 C + supersaturation gate differs (see the gate canary).
    """
    cfg = P3Config()
    dt, rho = 30.0, 0.9
    for T_val in (245.0, 248.0, 252.0, 255.0):   # <= 255 K: f_ice = 1 - O(1e-22)
        out = p3_microphysics(*_column(T_val, rho_val=rho), dt, cfg)
        expected = _gsam_cooper_target_per_kg(T_val, rho) / dt
        assert float(out.dN_i_dt[0, 0]) == pytest.approx(expected, rel=1e-9), \
            f"T={T_val}"


def test_p3_cooper_coefficients_match_gsam():
    """The base 5.0 /m^3 and exponent 0.304 /K are the gSAM Cooper constants."""
    cfg = P3Config()
    assert cfg.N_i0 == _GSAM_COOPER_BASE           # 0.005/L * 1000
    assert cfg.cooper_a == _GSAM_COOPER_A


def test_p3_cooper_relaxes_toward_target_minus_existing_Ni():
    """The rate is ``(target - N_i)/dt`` (gSAM ``(dum - sum(nitot))*odt``), NOT
    ``target/dt`` — a nonzero N_i below target must reduce the rate. Isolated
    with agg_coeff=dep_coeff=0 and q_i=0 so only nucleation drives dN_i_dt.
    (legoESM uses clip(dt,1); dt=30 here so it equals gSAM's odt=1/dt.)"""
    cfg = P3Config()._replace(agg_coeff=0.0, dep_coeff=0.0)
    dt, rho, T_val = 30.0, 0.9, 252.0
    target = _gsam_cooper_target_per_kg(T_val, rho)      # below cap
    N_i_val = 0.4 * target                               # nonzero, below target
    out = p3_microphysics(*_column(T_val, rho_val=rho, N_i_val=N_i_val), dt, cfg)
    expected = (target - N_i_val) / dt                   # f_ice ≈ 1 at 252 K
    assert float(out.dN_i_dt[0, 0]) == pytest.approx(expected, rel=1e-9)
    # Sanity: strictly less than the N_i=0 (target/dt) rate — proves the -N_i term.
    out0 = p3_microphysics(*_column(T_val, rho_val=rho, N_i_val=0.0), dt, cfg)
    assert float(out.dN_i_dt[0, 0]) < float(out0.dN_i_dt[0, 0])


def test_p3_cooper_mass_source_is_independent_seed_mass_times_number():
    """dq_i_dt = dN_i_dt * m_i0 with m_i0 computed INDEPENDENTLY as
    4/3*pi*rho_ice*(1e-6)^3 (not read from the module)."""
    dt, rho, T_val = 30.0, 0.9, 252.0
    out = p3_microphysics(*_column(T_val, rho_val=rho), dt, P3Config())
    m_i0_ref = 4.0 / 3.0 * math.pi * constants.rho_ice * _ICE_NUCLEUS_RADIUS ** 3
    expected_dN = _gsam_cooper_target_per_kg(T_val, rho) / dt
    assert float(out.dN_i_dt[0, 0]) == pytest.approx(expected_dN, rel=1e-9)
    assert float(out.dq_i_dt[0, 0]) == pytest.approx(
        expected_dN * m_i0_ref, rel=1e-9)


def test_p3_cooper_number_is_per_kg_rho_divide():
    """The target is per-MASS: halving rho doubles dN_i (gSAM inv_rho / legoESM
    /rho). Confirms the per-volume→per-mass convention."""
    dt = 30.0
    out_hi = p3_microphysics(*_column(252.0, rho_val=1.2), dt, P3Config())
    out_lo = p3_microphysics(*_column(252.0, rho_val=0.6), dt, P3Config())
    assert float(out_lo.dN_i_dt[0, 0]) == pytest.approx(
        2.0 * float(out_hi.dN_i_dt[0, 0]), rel=1e-9)


# --- CANARY the three documented departures from the gSAM oracle ---------------


def test_p3_cooper_cap_is_500_per_litre_unconditional_not_gsam_scf():
    """DELTA 1: legoESM caps at N_i_nuc_max=5e5 /m^3 (500 /L), UNCONDITIONAL;
    gSAM caps at 100 /L·SCF (scheme 1) or 150 /L·SCF (scheme 2) — a smaller cap
    scaled by cloud fraction SCF that legoESM omits."""
    cfg = P3Config()
    assert cfg.N_i_nuc_max == 5.0e5
    assert cfg.N_i_nuc_max > _GSAM_CAP_S1_PER_M3        # exceeds scheme-1 100/L
    assert cfg.N_i_nuc_max > _GSAM_CAP_S2_PER_M3        # exceeds scheme-2 150/L
    # At a very cold T the uncapped base overshoots the cap ⇒ dN_i saturates at
    # N_i_nuc_max/rho/dt (f_ice≈1, vapour non-limiting).
    T_cold, rho = 200.0, 0.5
    out = p3_microphysics(*_column(T_cold, rho_val=rho), 30.0, cfg)
    capped = cfg.N_i_nuc_max / rho / 30.0
    assert float(out.dN_i_dt[0, 0]) == pytest.approx(capped, rel=1e-9)
    assert capped > _GSAM_CAP_S2_PER_M3 / rho / 30.0   # above even gSAM 150/L·(SCF=1)


def test_p3_cooper_seed_density_is_917_not_gsam_900():
    """DELTA 3: legoESM seed mass uses constants.rho_ice (917); gSAM mi0 uses
    900 kg/m^3 at its default 1-µm nucleus radius ⇒ ~1.9% heavier seed."""
    m_i0_ref = 4.0 / 3.0 * math.pi * constants.rho_ice * _ICE_NUCLEUS_RADIUS ** 3
    assert _M_I0 == pytest.approx(m_i0_ref, rel=1e-12)
    assert _M_I0 / _GSAM_MI0 == pytest.approx(constants.rho_ice / 900.0, rel=1e-9)
    assert constants.rho_ice / 900.0 == pytest.approx(1.0189, abs=1e-4)


def test_p3_cooper_gate_warmer_and_has_no_supersaturation_requirement():
    """DELTA 2: legoESM's smooth gate (cooper_T_act=265 K = -8 C, no S_i check)
    nucleates where gSAM P3's HARD gate (T<258.15 K = -15 C AND supi>=0.05)
    would not: (a) at a warmer T=262 K (> -15 C), and (b) in ICE-SUBSATURATED
    air (q_v→0 ⇒ S_i<0)."""
    cfg = P3Config()
    assert cfg.cooper_T_act == 265.0                       # -8 C, vs gSAM 258.15
    out_warm = p3_microphysics(*_column(262.0), 30.0, cfg)
    assert float(out_warm.dN_i_dt[0, 0]) > 0.0             # gSAM would be OFF (>-15 C)
    out_subsat = p3_microphysics(*_column(252.0, q_v_val=1.0e-8), 30.0, cfg)
    assert float(out_subsat.dN_i_dt[0, 0]) > 0.0           # gSAM would need supi>=0.05


# --- AD-safety across the kink (T_freeze), the cap onset, and the capped tail --


def test_p3_cooper_nucleation_ad_safe_across_regimes():
    """jax.grad of the nucleation rate wrt T is finite at the max(T_f-T,0) kink
    (T_freeze), at the cap onset (~235.3 K for the default curve), deep in the
    capped tail, and at an interior point — not just one interior sample."""
    def dNi(Tval):
        return p3_microphysics(*_column(Tval), 30.0, P3Config()).dN_i_dt[0, 0]

    for T_probe in (constants.T_freeze, 252.0, 235.3, 210.0):
        g = jax.grad(dNi)(jnp.array(T_probe))
        assert bool(jnp.isfinite(g)), f"T={T_probe}"

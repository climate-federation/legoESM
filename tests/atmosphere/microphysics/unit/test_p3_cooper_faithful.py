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

legoESM now carries FAITHFUL scheme-1 semantics (departures closed 2026-07-17):
base curve (N_i0=5.0, cooper_a=0.304, per-kg rho-divide), scheme-1 cap
100/L·SCF at SCF=1 (this column scheme has no SCPF cloud fraction; gSAM's
scpf_ON=.false. default also runs SCF=1), the nucleation gate T < -15 C AND
supi >= 0.05, the (target-N_i)/dt relaxation, and the oracle seed mass
mi0 = 4/3*pi*900*(1e-6)^3.  The ONLY structural delta left is smoothing: the
hard Fortran gates become sigmoids (ice_sigmoid_sharpness on T,
cooper_supi_sharpness on supi) so the scheme stays differentiable, plus the
JAX guards (rho floor, clip(dt,1) floor, exponent cap) absent from the raw
Fortran.  Tests pin the faithful pieces at rel 1e-9 in the sharp-gate interior
(gate factors within 1e-13 of unity) and pin the smooth-gate midpoint
semantics explicitly.
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


def _column(T_val, rho_val=0.9, q_v_val=None, N_i_val=0.0, supi=0.2):
    """Isolated single-column state: no condensate, T in the nucleating band.

    ``q_v_val=None`` (default) sets vapour to ``(1+supi)*q_sat_i`` so the
    oracle's ``supi >= 0.05`` nucleation gate is satisfied deep in its
    interior (supi=0.2 puts the smooth gate within 1e-13 of unity) while
    staying non-limiting for the tiny seed-mass sink.  Pass an explicit
    ``q_v_val`` to probe the subsaturated (gate-off) branch.
    """
    from legoesm.thermo import saturation_mixing_ratio_ice

    ncol, nlev = 1, 3
    z = jnp.zeros((ncol, nlev))
    T = jnp.full((ncol, nlev), T_val)
    p_full = jnp.full((ncol, nlev), 7.0e4)
    if q_v_val is None:
        q_sat_i = saturation_mixing_ratio_ice(T, p_full)
        q_v = (1.0 + supi) * q_sat_i
    else:
        q_v = jnp.full((ncol, nlev), q_v_val)
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
    """legoESM's nucleation-number rate equals the gSAM base-curve form
    ``5*exp(0.304*(T_f-T))/rho / dt`` in the ORACLE gate's interior (T well
    below -15 C AND supi=0.2 >> 0.05, where both smooth gate factors are
    within ~1e-13 of unity) with vapour non-limiting.  This is now a genuine
    gated-oracle match: gSAM's hard gate is also ON at every probed state.
    """
    cfg = P3Config()
    dt, rho = 30.0, 0.9
    for T_val in (242.0, 245.0, 248.0, 252.0):   # <= 252 K: gate = 1 - O(1e-13)
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
    """dq_i_dt = dN_i_dt * m_i0 with the ORACLE m_i0 computed INDEPENDENTLY as
    4/3*pi*900*(1e-6)^3 (module_mp_p3.f90:233 — not read from the module).

    ``abs=0.0`` is load-bearing: the compared masses are ~1e-13 kg/kg/s, far
    below ``pytest.approx``'s default ``abs=1e-12`` floor, which silently
    accepted a 1.9% (917 vs 900) seed-mass error before."""
    dt, rho, T_val = 30.0, 0.9, 252.0
    out = p3_microphysics(*_column(T_val, rho_val=rho), dt, P3Config())
    m_i0_ref = 4.0 / 3.0 * math.pi * _GSAM_SEED_RHO * _ICE_NUCLEUS_RADIUS ** 3
    expected_dN = _gsam_cooper_target_per_kg(T_val, rho) / dt
    assert float(out.dN_i_dt[0, 0]) == pytest.approx(
        expected_dN, rel=1e-9, abs=0.0)
    assert float(out.dq_i_dt[0, 0]) == pytest.approx(
        expected_dN * m_i0_ref, rel=1e-9, abs=0.0)
    # Canary against the OLD wrong seed: the 917-based mass must NOT match.
    m_i0_917 = 4.0 / 3.0 * math.pi * constants.rho_ice * _ICE_NUCLEUS_RADIUS ** 3
    assert float(out.dq_i_dt[0, 0]) != pytest.approx(
        expected_dN * m_i0_917, rel=1e-9, abs=0.0)


def test_p3_cooper_number_is_per_kg_rho_divide():
    """The target is per-MASS: halving rho doubles dN_i (gSAM inv_rho / legoESM
    /rho). Confirms the per-volume→per-mass convention."""
    dt = 30.0
    out_hi = p3_microphysics(*_column(252.0, rho_val=1.2), dt, P3Config())
    out_lo = p3_microphysics(*_column(252.0, rho_val=0.6), dt, P3Config())
    assert float(out_lo.dN_i_dt[0, 0]) == pytest.approx(
        2.0 * float(out_hi.dN_i_dt[0, 0]), rel=1e-9)


# --- The three former departures, now pinned FAITHFUL --------------------------


def test_p3_cooper_cap_is_gsam_scheme1_100_per_litre():
    """FAITHFUL CAP: N_i_nuc_max equals gSAM scheme-1's 100 /L·SCF at SCF=1
    (module_mp_p3.f90:3090; this column scheme has no SCPF cloud fraction and
    gSAM's scpf_ON=.false. default also runs SCF=1).  At a very cold T the
    uncapped base overshoots the cap ⇒ dN_i saturates at N_i_nuc_max/rho/dt
    (gate factors = 1 - O(1e-13), vapour non-limiting)."""
    cfg = P3Config()
    assert cfg.N_i_nuc_max == _GSAM_CAP_S1_PER_M3        # 100 /L, scheme 1
    T_cold, rho = 200.0, 0.5
    out = p3_microphysics(*_column(T_cold, rho_val=rho), 30.0, cfg)
    capped = cfg.N_i_nuc_max / rho / 30.0
    assert float(out.dN_i_dt[0, 0]) == pytest.approx(capped, rel=1e-9)
    # Below-cap states must NOT saturate (cap binds only past the crossover).
    out_warm = p3_microphysics(*_column(252.0, rho_val=rho), 30.0, cfg)
    assert float(out_warm.dN_i_dt[0, 0]) < capped


def test_p3_cooper_seed_mass_is_gsam_mi0():
    """FAITHFUL SEED: m_i0 equals gSAM's mi0 = 4/3*pi*900*(1e-6)^3
    (module_mp_p3.f90:233 — the oracle hardcodes a 900 kg/m^3 nucleus density,
    NOT constants.rho_ice = 917; the ~1.9% heavier legoESM seed was the old
    departure)."""
    # abs=0.0: _M_I0 ~ 3.8e-15 sits far below approx's default abs=1e-12
    # floor, which would make this pin vacuous.
    assert _M_I0 == pytest.approx(_GSAM_MI0, rel=1e-12, abs=0.0)
    assert _M_I0 < 4.0 / 3.0 * math.pi * constants.rho_ice * _ICE_NUCLEUS_RADIUS ** 3


def test_p3_cooper_gate_matches_gsam_cold_and_supersaturated_only():
    """FAITHFUL GATE (smoothed): nucleation requires T < 258.15 K (-15 C) AND
    supi >= 0.05 (module_mp_p3.f90:3084).  (a) warm T=262 K (supersaturated) is
    OFF; (b) ice-SUBSATURATED air at 252 K is OFF; (c) cold+supersaturated is
    ON; (d) the smooth-gate midpoints sit at half the interior rate."""
    cfg = P3Config()
    assert cfg.cooper_T_nuc == pytest.approx(constants.T_freeze - 15.0)
    assert cfg.cooper_supi_min == 0.05
    dt, rho = 30.0, 0.9

    on = float(p3_microphysics(*_column(252.0, rho_val=rho), dt, cfg).dN_i_dt[0, 0])
    assert on > 0.0

    # (a) warm, supersaturated: T-sigmoid(5*(258.15-262)) ~ 4e-9 ⇒ suppressed
    # by >= 1e8 relative to its own ungated base rate.
    warm = float(p3_microphysics(*_column(262.0, rho_val=rho), dt, cfg).dN_i_dt[0, 0])
    warm_base = _gsam_cooper_target_per_kg(262.0, rho) / dt
    assert warm < 1.0e-6 * warm_base

    # (b) cold but ice-subsaturated (q_v -> 0 ⇒ supi ~ -1): supi-sigmoid
    # (200*(-1.05)) ~ e^-210 ~ 1e-92 — nucleation is dead to all purposes.
    subsat = float(p3_microphysics(
        *_column(252.0, rho_val=rho, q_v_val=1.0e-8), dt, cfg).dN_i_dt[0, 0])
    assert subsat < 1.0e-60

    # (d) midpoint semantics: at supi = supi_min exactly the supi factor is
    # 1/2; at T = cooper_T_nuc exactly the T factor is 1/2.
    mid_supi = float(p3_microphysics(
        *_column(252.0, rho_val=rho, supi=cfg.cooper_supi_min),
        dt, cfg).dN_i_dt[0, 0])
    base_252 = _gsam_cooper_target_per_kg(252.0, rho) / dt
    assert mid_supi == pytest.approx(0.5 * base_252, rel=1e-6)
    mid_T = float(p3_microphysics(
        *_column(float(cfg.cooper_T_nuc), rho_val=rho), dt, cfg).dN_i_dt[0, 0])
    base_midT = _gsam_cooper_target_per_kg(float(cfg.cooper_T_nuc), rho) / dt
    assert mid_T == pytest.approx(0.5 * base_midT, rel=1e-6)


# --- AD-safety across the kink (T_freeze), the cap onset, and the capped tail --


def test_p3_cooper_nucleation_ad_safe_across_regimes():
    """jax.grad of the nucleation rate wrt T is finite at the max(T_f-T,0) kink
    (T_freeze), at the cap onset (~240.6 K for the 100/L cap), deep in the
    capped tail, and at an interior point — not just one interior sample."""
    def dNi(Tval):
        return p3_microphysics(*_column(Tval), 30.0, P3Config()).dN_i_dt[0, 0]

    for T_probe in (constants.T_freeze, 252.0, 240.6, 210.0):
        g = jax.grad(dNi)(jnp.array(T_probe))
        assert bool(jnp.isfinite(g)), f"T={T_probe}"

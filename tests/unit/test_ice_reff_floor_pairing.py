"""Ice effective radius must pair the M2005 PSD with TRACER ice only (#1520).

``compute_cloud_properties`` hands the radiation an ice water path that
includes the DIAGNOSTIC sub-grid condensate floor (``cf·q_c_diagnostic·f_ice``)
on the sundqvist/xu_randall schemes.  The prognostic ``N_i`` tracer knows
nothing about that floor mass, so building ``EFFI = 1.5/LAMI`` with
``LAMI ~ (N_i/q_i_floored)^(1/3)`` produced metre-scale "effective radii"
wherever the floor dominates and ``N_i`` is 0/tiny — on the #1520 production
checkpoint ~34% of the radiative IWP was pinned at the RRTMGP ice-LUT 180 µm
diameter ceiling (92% of that pinned mass floor-injected), leaving the ice
~3x optically too thin in both SW and LW.

The fix mirrors the liquid branch's dead-``N_c`` fallback: PSD radius for the
TRACER mass, the configured constant ``r_eff_ice`` for the floor-injected
mass, combined by extinction (τ ∝ IWP/r_eff ⇒ mass-weighted harmonic mean).

These tests pin the RELATIONSHIPS, not recorded numbers:

* pure floor ice (no tracer pair at all) radiates at the configured constant;
* a pure tracer cell (no floor deficit) keeps the M2005 PSD radius EXACTLY;
* a mixed cell satisfies the extinction identity
  ``m_tot/r_blend = m_psd/r_psd + m_flr/r_const``;
* the blend is bracketed by its two member radii;
* the 'resolved' scheme (no floor) is bit-preserved;
* the branch stays finite under ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import pytest
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    _R_EFF_ICE_PSD_COEFF,
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.thermo import saturation_mixing_ratio

NCOL, NLEV = 1, 4

# Cold column: T = 220 K < T_ice_only = 233.15 K, so the diagnostic floor is
# 100% ice (f_ice_diag = 1) and the liquid branch is inert.
T_COLD = 220.0
P_FULL = 3.0e4  # 300 hPa

# sundqvist with defaults: rh_crit = 0.77.  RH = 0.95 gives a solid stratiform
# fraction cf = 1 - sqrt(0.05/0.23) ≈ 0.53, hence a floor deficit
# cf·q_c_diagnostic ≈ 5e-4 kg/kg in a condensate-free cell.
CFG = CloudConfig(scheme="sundqvist")
RH = 0.95


def _cold_column(q_ice, n_ice, cfg=CFG):
    sh = np.full((NCOL, NLEV), 1.0)
    T = jnp.asarray(sh * T_COLD)
    p_full = jnp.asarray(sh * P_FULL)
    q_sat = saturation_mixing_ratio(T, p_full)
    return compute_cloud_properties(
        T=T,
        p_full=p_full,
        q_v=RH * q_sat,
        dp=jnp.asarray(sh * 1.0e4),
        config=cfg,
        q_cloud=jnp.asarray(sh * 0.0),
        q_ice=jnp.asarray(sh * q_ice),
        n_ice=jnp.asarray(sh * n_ice),
        n_cloud=jnp.asarray(sh * 0.0),
    )


def _psd_radius(q_i, n_i, cfg=CFG):
    """The M2005 radius the code should assign to a live tracer pair [m]."""
    lami = (cfg.rho_cloud_ice * np.pi * n_i / q_i) ** (1.0 / 3.0)
    return _R_EFF_ICE_PSD_COEFF / lami


# --------------------------------------------------------------- pure floor

def test_floor_only_ice_radiates_at_the_configured_constant():
    """Dead tracer pair (q_i = N_i = 0) + a live floor => r_eff_ice must be
    the configured constant, NOT a PSD built against the 1e-15 number floor.

    Pre-#1520 this returned 1.5·(cf·q_c_diag/(ρ_ci·π·1e-15))^(1/3) ~ 10 m.
    """
    props = _cold_column(q_ice=0.0, n_ice=0.0)
    iwp = float(props.iwp[0, 0])
    assert iwp > 0.0, "floor did not inject radiative ice — test is vacuous"
    r = float(props.r_eff_ice[0, 0])
    assert r == pytest.approx(CFG.r_eff_ice, rel=1e-12), (
        f"floor-injected ice got r_eff={r*1e6:.3g} um instead of the "
        f"configured {CFG.r_eff_ice*1e6:.3g} um"
    )


def test_floor_plus_dead_tiny_number_stays_near_the_constant():
    """N_i tiny-but-nonzero (the dominant #1520 population, median 8.5e-7/kg)
    with sub-QSMALL tracer ice: the sub-QSMALL tracer mass has no PSD standing
    and the whole radiative mass must stay at the constant radius."""
    props = _cold_column(q_ice=1.0e-15, n_ice=1.0e-6)
    r = float(props.r_eff_ice[0, 0])
    assert r == pytest.approx(CFG.r_eff_ice, rel=1e-9)


# ----------------------------------------------------------- pure tracer PSD

def test_tracer_dominated_cell_keeps_the_m2005_psd_radius():
    """A tracer pair big enough to exceed the floor (no deficit) must keep the
    PSD radius exactly — the fix may not touch consistent live pairs."""
    q_i, n_i = 2.0e-3, 5.0e5  # 2 g/kg, 5e5 /kg -> r_eff ~ 16 um, above floor
    props = _cold_column(q_ice=q_i, n_ice=n_i)
    r = float(props.r_eff_ice[0, 0])
    assert r == pytest.approx(_psd_radius(q_i, n_i), rel=1e-12)


def test_resolved_scheme_is_bit_preserved():
    """'resolved' has no diagnostic floor: the blend must never engage and the
    PSD radius must come out bit-identical to the direct formula."""
    cfg = CloudConfig(scheme="resolved")
    q_i, n_i = 1.0e-4, 2.0e5
    props = _cold_column(q_ice=q_i, n_ice=n_i, cfg=cfg)
    r = float(props.r_eff_ice[0, 0])
    lami = (cfg.rho_cloud_ice * jnp.pi
            * jnp.maximum(jnp.asarray(n_i), 1.0e-15)
            / jnp.maximum(jnp.asarray(q_i), 1.0e-15)) ** (1.0 / 3.0)
    expected = float(_R_EFF_ICE_PSD_COEFF / jnp.clip(lami, 1.0e-30))
    assert r == expected  # bit-preserved, not approx


# ------------------------------------------------------------- the mixed cell

def test_mixed_cell_satisfies_the_extinction_identity():
    """τ ∝ IWP/r_eff must be conserved by the blend:
    m_tot/r_blend == m_psd/r_psd + m_flr/r_const."""
    q_i, n_i = 1.0e-4, 1.0e5  # live pair, but below the ~5e-4 floor => deficit
    props = _cold_column(q_ice=q_i, n_ice=n_i)
    r_blend = float(props.r_eff_ice[0, 0])
    r_psd = _psd_radius(q_i, n_i)

    # Construction gives the radiative masses in closed form (constant scheme,
    # convective cloud off, f_ice_diag = 1 at 220 K): the floor target is
    # F = cf·q_c_diagnostic and the deficit tops the total up to F, so
    #   m_psd = q_i (tracer),   m_flr = F − q_i.
    cf = float(props.cloud_fraction[0, 0])
    F = cf * CFG.q_c_diagnostic
    assert F > q_i, "construction broken: tracer exceeds floor, no deficit"
    w_psd = q_i / F
    w_flr = (F - q_i) / F
    lhs = 1.0 / r_blend
    rhs = w_psd / r_psd + w_flr / CFG.r_eff_ice
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_blend_is_bracketed_by_its_members():
    q_i, n_i = 1.0e-4, 1.0e3  # tiny number: PSD radius >> const
    props = _cold_column(q_ice=q_i, n_ice=n_i)
    r_blend = float(props.r_eff_ice[0, 0])
    r_psd = _psd_radius(q_i, n_i)
    assert min(CFG.r_eff_ice, r_psd) <= r_blend <= max(CFG.r_eff_ice, r_psd)


# ------------------------------------------------------------------ AD safety

def test_grad_through_the_blend_is_finite():
    sh = np.full((NCOL, NLEV), 1.0)
    T = jnp.asarray(sh * T_COLD)
    p_full = jnp.asarray(sh * P_FULL)
    q_sat = saturation_mixing_ratio(T, p_full)

    def scalar_reff(q_i_scalar, n_i_scalar):
        props = compute_cloud_properties(
            T=T, p_full=p_full, q_v=RH * q_sat,
            dp=jnp.asarray(sh * 1.0e4), config=CFG,
            q_cloud=jnp.asarray(sh * 0.0),
            q_ice=jnp.asarray(sh) * q_i_scalar,
            n_ice=jnp.asarray(sh) * n_i_scalar,
            n_cloud=jnp.asarray(sh * 0.0),
        )
        return jnp.sum(props.r_eff_ice)

    for q0, n0 in [(0.0, 0.0), (1.0e-15, 1.0e-6), (1.0e-4, 1.0e5),
                   (2.0e-3, 5.0e5)]:
        g = jax.grad(scalar_reff, argnums=(0, 1))(q0, n0)
        assert np.isfinite(float(g[0])), (q0, n0)
        assert np.isfinite(float(g[1])), (q0, n0)

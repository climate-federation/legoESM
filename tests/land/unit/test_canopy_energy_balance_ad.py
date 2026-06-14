"""AD-stability + sign tests for the canopy energy balance (DifferBESS Apr-13).

The latent-heat fluxes are written in conductance form so reverse/forward-mode
AD uses the product rule and stays finite in the limiting regimes:
- leaf:  g_lh   = gs / (gs*Rb + 1)         finite as gs -> 0 (closed stomata)
- soil:  g_soil = fStress / raw_soil       finite as fStress -> 0 (dry soil)

These are algebraically identical to the old quotient forms in the
well-conditioned regime, which we also check.  Sensible heat ``H`` is left
sign-unconstrained (negative H = downward/nocturnal flux).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.land.canopy.energy_balance import (
    leaf_energy_balance_bt,
    soil_energy_balance_bt,
)


# ---- representative well-conditioned leaf state -----------------------------
_LEAF = dict(
    ASW=jnp.array(250.0), ALW=jnp.array(-40.0), Tf=jnp.array(300.0),
    Ps=jnp.array(101325.0), Ca=jnp.array(400.0), Tc=jnp.array(299.0),
    q_f=jnp.array(0.022), q_c=jnp.array(0.015), RH_c=jnp.array(0.6),
    VPD_c=jnp.array(1000.0), lam=jnp.array(2.5e6), Cp=jnp.array(1005.0),
    rhoa=jnp.array(1.2), Rb=jnp.array(40.0), m=jnp.array(9.0), b0=jnp.array(0.01),
)

_SOIL = dict(
    Ts=jnp.array(305.0), Tc=jnp.array(300.0), q_s=jnp.array(0.025),
    q_c=jnp.array(0.015), lam=jnp.array(2.5e6), rhoa=jnp.array(1.2),
    Cp=jnp.array(1005.0), rah_soil=jnp.array(50.0), raw_soil=jnp.array(50.0),
    ASW_soil=jnp.array(120.0), ALW_soil=jnp.array(-25.0),
)


def test_leaf_conductance_identity():
    """gs/(gs*Rb+1) == 1/(Rb + 1/gs) — reformulation is forward-identical."""
    gs = jnp.array([0.001, 0.01, 0.1, 1.0]); Rb = jnp.array(40.0)
    assert jnp.allclose(gs / (gs * Rb + 1.0), 1.0 / (Rb + 1.0 / gs), rtol=1e-10)


def test_leaf_le_matches_conductance_form():
    Rn, LE, H, Tf_new, gs, Ci = leaf_energy_balance_bt(An=jnp.array(12.0), **_LEAF)
    num = _LEAF["lam"] * _LEAF["rhoa"] * (_LEAF["q_f"] - _LEAF["q_c"])
    assert jnp.allclose(LE, num * gs / (gs * _LEAF["Rb"] + 1.0), rtol=1e-6)
    assert jnp.allclose(LE, num / (_LEAF["Rb"] + 1.0 / gs), rtol=1e-6)


def test_leaf_le_grad_finite_dark_and_light():
    """grad of leaf LE wrt the Ball-Berry slope is finite (incl. dark/closed)."""
    def le_of_m(m, An):
        _, LE, *_ = leaf_energy_balance_bt(
            An=An, **{**_LEAF, "m": m})
        return LE
    for An in (jnp.array(0.0), jnp.array(15.0)):
        g = jax.grad(le_of_m)(jnp.array(9.0), An)
        assert jnp.isfinite(g)


def test_soil_le_zero_and_grad_finite_at_zero_fstress():
    """LE_soil -> 0 as fStress -> 0, with a finite derivative num/raw_soil."""
    def le_of_fstress(fStress):
        _, LE, _, _ = soil_energy_balance_bt(fStress=fStress, **_SOIL)
        return LE

    LE0 = le_of_fstress(jnp.array(0.0))
    assert jnp.allclose(LE0, 0.0, atol=1e-12)

    g = jax.grad(le_of_fstress)(jnp.array(0.0))
    expected = (_SOIL["lam"] * _SOIL["rhoa"] * (_SOIL["q_s"] - _SOIL["q_c"])
                / _SOIL["raw_soil"])
    assert jnp.isfinite(g)
    assert jnp.allclose(g, expected, rtol=1e-6)


def test_soil_le_matches_old_quotient_in_wet_regime():
    """For fStress in (0,1], num*fStress/raw_soil == num/(raw_soil + Rsoil)."""
    fStress = jnp.array(0.4)
    _, LE, _, _ = soil_energy_balance_bt(fStress=fStress, **_SOIL)
    num = _SOIL["lam"] * _SOIL["rhoa"] * (_SOIL["q_s"] - _SOIL["q_c"])
    Rsoil = _SOIL["raw_soil"] * (1.0 / fStress - 1.0)  # old dryness resistance
    assert jnp.allclose(LE, num / (_SOIL["raw_soil"] + Rsoil), rtol=1e-6)


def test_sensible_heat_can_be_negative():
    """H < 0 is allowed (downward / nocturnal sensible heat) — no sign clamp."""
    # Soil cooler than canopy air -> downward H.
    _, _, H_soil, _ = soil_energy_balance_bt(
        **{**_SOIL, "Ts": jnp.array(297.0), "Tc": jnp.array(302.0),
           "fStress": jnp.array(0.2)})
    assert float(H_soil) < 0.0

    # Nocturnal leaf: net radiative loss, leaf cools below canopy air.
    _, _, H_leaf, _, _, _ = leaf_energy_balance_bt(
        An=jnp.array(0.0),
        **{**_LEAF, "ASW": jnp.array(0.0), "ALW": jnp.array(-80.0)})
    assert float(H_leaf) < 0.0

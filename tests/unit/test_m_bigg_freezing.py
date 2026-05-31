"""SAM Bigg (1953) immersion freezing of supercooled rain (iter-24).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:3251-3257) freezes supercooled
rain at the Bigg rate::

    X      = exp(AIMM·(T₀−T)) − 1          (AIMM=0.66/K, BIMM=100)
    MNUCCR = 20·π²·ρ_w·BIMM·(N_r/ρ)·X / LAMR^6    (mass)
    NNUCCR = π·N_r·BIMM·X / LAMR^3                (number)

legoESM previously had NO rain freezing — supercooled rain stayed liquid
below 0 °C, missing the latent heat of fusion and wrong-phasing cold updrafts.
Frozen rain joins SNOW (legoESM has no graupel category).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    rain_freezing_bigg,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _mnuccr(T, q_r=1.0e-3, N_r=1.0e4, p=5.0e4, dt=None):
    rho = p / (constants.R_d * T)
    m, n = rain_freezing_bigg(
        jnp.asarray(q_r), jnp.asarray(N_r), jnp.asarray(T),
        jnp.asarray(rho), _CFG, dt=dt)
    return float(m), float(n)


def _hand(T, q_r=1.0e-3, N_r=1.0e4, p=5.0e4):
    rho = p / (constants.R_d * T)
    lamr = (math.pi * constants.rho_water * N_r / (rho * q_r)) ** (1.0 / 3.0)
    lamr = min(max(lamr, _CFG.lamr_min), _CFG.lamr_max)
    x = math.exp(_CFG.bigg_aimm * (constants.T_freeze - T)) - 1.0
    m = (20 * math.pi ** 2 * constants.rho_water * _CFG.bigg_bimm
         * (N_r / rho) * x / lamr ** 6)
    n = math.pi * N_r * _CFG.bigg_bimm * x / lamr ** 3
    return m, n


def test_freezing_matches_sam_formula():
    m, n = _mnuccr(258.0)                     # −15 °C, unclamped
    mh, nh = _hand(258.0)
    assert m == pytest.approx(mh, rel=1e-6)
    assert n == pytest.approx(nh, rel=1e-6)


def test_freezing_increases_steeply_with_supercooling():
    """Bigg rate ∝ exp(AIMM·ΔT): much faster at −20 than −10 °C."""
    m10, _ = _mnuccr(263.0, dt=None)
    m20, _ = _mnuccr(253.0, dt=None)
    assert m20 > 100.0 * m10                  # exp(0.66·10) ≈ 735× steeper


def test_no_freezing_above_freezing():
    assert _mnuccr(275.0) == (0.0, 0.0)
    assert _mnuccr(273.15) == (0.0, 0.0)      # exactly 0 °C ⇒ ΔT=0 ⇒ 0


def test_no_freezing_without_rain():
    assert _mnuccr(258.0, q_r=0.0, N_r=0.0) == (0.0, 0.0)


def test_freezing_donor_clamped():
    m, n = _mnuccr(243.0, dt=20.0)            # −30 °C ⇒ huge raw rate
    assert m <= 1.0e-3 / 20.0 + 1e-18
    assert n <= 1.0e4 / 20.0 + 1e-9


def _morrison_dT_dqr_dqs_dNr(scheme, T=258.0):
    qsat = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(5.0e4)))
    rho = 5.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), 1.0e-3), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=jnp.full((1, 1), 1.0e4), N_i=z,
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), qsat), hm,    # liquid-saturated
        jnp.full((1, 1), 5.0e4), jnp.full((1, 2), 5.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), 20.0,
        MorrisonConfig(rain_freeze_scheme=scheme),
    )
    return (float(out.dT_dt[0, 0]), float(out.dq_r_dt[0, 0]),
            float(out.dq_s_dt[0, 0]), float(out.dN_r_dt[0, 0]),
            float(out.dq_g_dt[0, 0]))


def test_freezing_conserves_water_and_releases_latent_heat():
    """bigg−none difference isolates freezing: SAM freezes supercooled rain to
    GRAUPEL (iter-34), so rain mass → graupel mass (Δdq_r = −Δdq_g), snow is
    untouched, and the latent heat of fusion warms the air (ΔdT > 0)."""
    dT_b, dqr_b, dqs_b, dNr_b, dqg_b = _morrison_dT_dqr_dqs_dNr("bigg")
    dT_n, dqr_n, dqs_n, dNr_n, dqg_n = _morrison_dT_dqr_dqs_dNr("none")
    # rain → graupel (mass conserved between the two categories at T<0)
    assert (dqr_b - dqr_n) == pytest.approx(-(dqg_b - dqg_n), abs=1e-12)
    assert (dqr_b - dqr_n) < 0.0               # rain lost to freezing
    # frozen rain no longer goes to snow
    assert (dqs_b - dqs_n) == pytest.approx(0.0, abs=1e-12)
    # latent heat of fusion warms (ΔdT = +L_f·freeze/c_p)
    assert dT_b > dT_n
    # rain number removed by freezing
    assert dNr_b < dNr_n


def test_freezing_ad_safe():
    def loss(q_r0):
        rho = 5.0e4 / (constants.R_d * 258.0)
        m, n = rain_freezing_bigg(
            q_r0.reshape(1, 1), jnp.full((1, 1), 1.0e4), jnp.asarray(258.0),
            jnp.asarray(rho), _CFG, dt=20.0)
        return jnp.sum(m + n)

    for q0 in (0.0, 1.0e-3):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))


def test_unknown_rain_freeze_scheme_raises():
    qsat = float(saturation_mixing_ratio(jnp.asarray(258.0), jnp.asarray(5.0e4)))
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), 1.0e-3), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=jnp.full((1, 1), 1.0e4), N_i=z,
    )
    with pytest.raises(ValueError, match="Unknown rain_freeze_scheme"):
        morrison_microphysics(
            jnp.full((1, 1), 258.0), jnp.full((1, 1), qsat), hm,
            jnp.full((1, 1), 5.0e4), jnp.full((1, 2), 5.0e4),
            jnp.full((1, 1), 0.67), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(rain_freeze_scheme="bigg_typo"),
        )

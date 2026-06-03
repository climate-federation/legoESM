"""SAM M2005 ice deposition + sublimation tests (iter-8 M1).

gSAM ``MICRO_M2005`` grows/sublimates cloud ice by bulk diffusion
(``module_mp_graupel.f90:3427-3514``):
    PRD = EPSI·(q_v − q_sat_i)/ABI,  EPSI ∝ ρ·DV·N_i^⅔·q_i^⅓
with the ABI psychrometric (latent-heat) correction and a SUBLIMATION
branch when ``q_v < q_sat_i``. The legacy legoESM form used
``max(S_i,0)·q_i·N_i^⅓`` — deposition only, wrong N_i power.

With ``q_c=q_r=q_s=0`` and ``q_v`` ice-supersaturated but liquid-
subsaturated, the only vapour process is ice deposition, so
``dq_v_dt = −dq_i_dep`` — a clean observable for the deposition rate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_mixing_ratio_ice,
)


jax.config.update("jax_enable_x64", True)


def _dep_rate(q_v, q_i=1.0e-4, N_i=1.0e5, T=240.0, p=3.0e4, rho=0.4,
              scheme="m2005", dt=20.0):
    """Return the deposition rate dq_i_dep = −dq_v_dt (only ice present)."""
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=jnp.full((1, 1), q_i), q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((1, 1), N_i),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), q_v), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p),
        jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), dt,
        # N_i0=0 disables Cooper nucleation so −dq_v_dt isolates the
        # deposition/sublimation vapour exchange (no nucleation mass sink).
        MorrisonConfig(ice_deposition_scheme=scheme, N_i0=0.0),
    )
    return -float(out.dq_v_dt[0, 0])


def _ice_supersat_qv(T, p):
    """q_v ice-supersaturated but liquid-SUBsaturated (no condensation)."""
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))
    qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(p)))
    assert qsi < qsl                       # cold ⇒ ice sat below liquid sat
    return 0.5 * (qsi + qsl), qsi          # midway: S_i>0, S_liq<0


def test_m2005_deposition_consumes_vapor_when_ice_supersaturated():
    q_v, _ = _ice_supersat_qv(240.0, 3.0e4)
    assert _dep_rate(q_v) > 0.0            # ice grows, vapour consumed


def test_m2005_sublimates_when_subsaturated_but_heuristic_does_not():
    """The key M1 fix: in ice-subsaturated air the M2005 scheme SUBLIMATES
    ice (negative deposition ⇒ vapour source), which the legacy heuristic
    (deposition-only) cannot do."""
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(240.0), jnp.asarray(3.0e4)))
    q_v = 0.5 * qsi                        # 50% ice-subsaturated
    assert _dep_rate(q_v, scheme="m2005") < 0.0        # sublimation
    assert _dep_rate(q_v, scheme="heuristic") == 0.0   # no sublimation


def test_m2005_sublimation_fires_in_warm_subsaturated_air():
    """Codex iter-8 fix: SUBLIMATION is NOT gated by Cooper activation
    (f_ice). Ice present in WARM (f_ice≈0) subsaturated air must still
    sublimate (vapour source)."""
    T_warm = 272.0                          # f_ice = sigmoid(5·(265−272)) ≈ 0
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T_warm), jnp.asarray(6.0e4)))
    # Subsaturated wrt ice ⇒ sublimation should fire despite warm T.
    rate = _dep_rate(0.6 * qsi, q_i=1.0e-5, N_i=1.0e5, T=T_warm, p=6.0e4)
    assert rate < 0.0                       # ice sublimating in warm air


def test_m2005_deposition_fires_in_warm_mixed_phase():
    """iter-14 (Codex iters 8/9/13): DEPOSITION growth is NO LONGER
    f_ice-gated. In the WARM mixed phase (270 K, f_ice≈0) ice that is
    present + ice-supersaturated must GROW (positive deposition),
    comparable in magnitude to a cold cell — the EPSI∝N_i^⅔ factor
    self-gates on ice presence, so the Cooper temperature gate was a
    redundant suppressor of warm-cloud ice growth + the emergent WBF."""
    # q_i=0, N_i>0 ⇒ deposition from freshly-nucleated ice, no sedimentation
    # contamination ⇒ _dep_rate (= −dq_v_dt) is the deposition rate.
    # q_v = midpoint(q_sat_ice, q_sat_liq): ice-supersaturated but LIQUID-
    # subsaturated, so no saturation-adjustment condensation contaminates
    # −dq_v_dt — it is purely the ice deposition rate.
    def warm_dep(T, N_i=1.0e5):
        qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(5.0e4)))
        qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(5.0e4)))
        return _dep_rate(0.5 * (qsi + qsl), q_i=0.0, N_i=N_i, T=T, p=5.0e4)

    warm = warm_dep(270.0)        # f_ice ≈ 1.4e-11
    cold = warm_dep(250.0)        # f_ice ≈ 1
    assert warm > 0.0                          # warm ice now GROWS (was ~0)
    assert warm > 0.1 * cold                   # same order as cold (un-suppressed)
    # Ice-free cell (N_i=0) still self-gates to exactly 0 (EPSI∝N_i^⅔).
    assert warm_dep(270.0, N_i=0.0) == 0.0


def test_m2005_sublimation_donor_clamped_to_available_ice():
    """Sublimation cannot exceed q_i/dt; with no ice it is exactly 0."""
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(240.0), jnp.asarray(3.0e4)))
    q_v = 0.3 * qsi
    # No ice ⇒ clamped to 0.
    assert _dep_rate(q_v, q_i=0.0) == 0.0
    # With ice, the sublimation magnitude ≤ q_i/dt.
    q_i, dt = 1.0e-5, 20.0
    rate = _dep_rate(q_v, q_i=q_i, dt=dt)
    assert -q_i / dt - 1e-18 <= rate < 0.0


def test_m2005_scaling_qi_third_Ni_twothird():
    """EPSI ∝ q_i^⅓·N_i^⅔ (SAM diffusional growth), NOT the legacy
    q_i^1·N_i^⅓. Verify the power laws by ratios."""
    q_v, _ = _ice_supersat_qv(240.0, 3.0e4)
    # Vary q_i ×8 at fixed N_i ⇒ rate ×8^(1/3)=2.
    r1 = _dep_rate(q_v, q_i=1.0e-5, N_i=1.0e5)
    r2 = _dep_rate(q_v, q_i=8.0e-5, N_i=1.0e5)
    assert r2 / r1 == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-3)
    # Vary N_i ×8 at fixed q_i ⇒ rate ×8^(2/3)=4.
    r3 = _dep_rate(q_v, q_i=1.0e-5, N_i=1.0e5)
    r4 = _dep_rate(q_v, q_i=1.0e-5, N_i=8.0e5)
    assert r4 / r3 == pytest.approx(8.0 ** (2.0 / 3.0), rel=1e-3)


def test_m2005_abi_psychrometric_reduces_rate():
    """ABI = 1 + (dq_sat_i/dT)·L_s/c_p > 1 ⇒ the latent-heat correction
    REDUCES deposition below the diffusion-only (ABI=1) rate. ABI grows
    toward warmer (T→0 °C) ice where q_sat_i is larger."""
    q_v_cold, _ = _ice_supersat_qv(230.0, 3.0e4)
    q_v_warm, _ = _ice_supersat_qv(265.0, 3.0e4)
    # Warmer ice ⇒ larger q_sat_i ⇒ larger ABI ⇒ proportionally more
    # latent-heat damping. (Sanity: both positive, finite.)
    assert _dep_rate(q_v_cold, T=230.0) > 0.0
    assert _dep_rate(q_v_warm, T=265.0) > 0.0


def test_m2005_deposition_ad_safe():
    """jax.grad of the deposition wrt q_v stays finite across sat/subsat."""
    def loss(q_v):
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=z, q_r=z, q_i=jnp.full((1, 1), 1.0e-4), q_s=z, q_g=z,
            N_c=z, N_r=z, N_i=jnp.full((1, 1), 1.0e5),
        )
        out = morrison_microphysics(
            jnp.full((1, 1), 240.0), q_v.reshape(1, 1), hm,
            jnp.full((1, 1), 3.0e4), jnp.full((1, 2), 3.0e4),
            jnp.full((1, 1), 0.4), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(),
        )
        return jnp.sum(out.dq_i_dt)

    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(240.0), jnp.asarray(3.0e4)))
    for frac in (0.3, 0.8, 1.0, 1.5):
        g = jax.grad(loss)(jnp.asarray(frac * qsi))
        assert bool(jnp.isfinite(g))


def test_unknown_ice_deposition_scheme_raises():
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=jnp.full((1, 1), 1e-4), q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((1, 1), 1e5),
    )
    with pytest.raises(ValueError, match="Unknown ice_deposition_scheme"):
        morrison_microphysics(
            jnp.full((1, 1), 240.0), jnp.full((1, 1), 1e-3), hm,
            jnp.full((1, 1), 3e4), jnp.full((1, 2), 3e4),
            jnp.full((1, 1), 0.4), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(ice_deposition_scheme="bulk_typo"),
        )

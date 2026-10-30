"""Emergent Wegener-Bergeron-Findeisen tests (iter-13 M2).

gSAM MICRO_M2005 has NO explicit Bergeron rate: in the mixed phase the
ice deposition (PRD draws vapour toward ice saturation) plus the
saturation adjustment (evaporates cloud water as q_v drops below liquid
saturation) together convert cloud water to ice — deposition-rate-limited.
legoESM's legacy explicit ``bergeron_rate`` over-glaciates (~50× faster);
the faithful default is ``wbf_scheme="emergent"``.
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
from legoesm.thermo import saturation_specific_humidity


def _run(wbf, q_c=5.0e-4, q_i=1.0e-5, N_i=1.0e5, T=260.0, p=6.0e4,
         berg_rate=1.0e-3):
    """Return (dq_c_dt, dq_i_dt) in a mixed-phase, liquid-saturated cell."""
    qsl = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
    q_v = qsl                              # liquid-saturated ⇒ ice-supersat
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), q_c), q_r=z,
        q_i=jnp.full((1, 1), q_i), q_s=z, q_g=z,
        N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=jnp.full((1, 1), N_i),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), q_v), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p),
        jnp.full((1, 1), 0.7), jnp.full((1, 1), 300.0), 20.0,
        MorrisonConfig(wbf_scheme=wbf, bergeron_rate=berg_rate),
    )
    return float(out.dq_c_dt[0, 0]), float(out.dq_i_dt[0, 0])


jax.config.update("jax_enable_x64", True)


def test_emergent_wbf_converts_cloud_to_ice():
    """Mixed-phase, no explicit Bergeron: cloud water still glaciates
    (dq_c<0, dq_i>0) via deposition + saturation adjustment."""
    dq_c, dq_i = _run("emergent")
    assert dq_c < 0.0                       # cloud water sink
    assert dq_i > 0.0                       # ice gains


def test_emergent_is_deposition_limited_vs_heuristic():
    """The emergent (SAM-faithful) glaciation is much SLOWER than the
    legacy explicit-rate heuristic, which forces conversion far beyond
    what the deposition physics supports."""
    dqc_em, _ = _run("emergent")
    dqc_he, _ = _run("bergeron_heuristic")
    assert abs(dqc_em) < abs(dqc_he)        # emergent slower
    assert abs(dqc_he) > 10.0 * abs(dqc_em)  # heuristic ≫ deposition rate


def test_no_wbf_without_ice():
    """No ice (and nucleation off via N_i0=0) ⇒ no deposition surface ⇒
    no emergent glaciation of cloud water."""
    qsl = float(saturation_specific_humidity(jnp.asarray(260.0), jnp.asarray(6.0e4)))
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), 5.0e-4), q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=z,
    )
    out = morrison_microphysics(
        jnp.full((1, 1), 260.0), jnp.full((1, 1), qsl), hm,
        jnp.full((1, 1), 6.0e4), jnp.full((1, 2), 6.0e4),
        jnp.full((1, 1), 0.7), jnp.full((1, 1), 300.0), 20.0,
        MorrisonConfig(wbf_scheme="emergent", N_i0=0.0),
    )
    # No ice source ⇒ ice tendency ≈ 0 (no WBF glaciation).
    assert abs(float(out.dq_i_dt[0, 0])) < 1.0e-12


def test_heuristic_scheme_still_available():
    """The legacy explicit Bergeron remains selectable + bergeron_rate
    scales it."""
    dqc_lo, _ = _run("bergeron_heuristic", berg_rate=1.0e-3)
    dqc_hi, _ = _run("bergeron_heuristic", berg_rate=5.0e-3)
    assert abs(dqc_hi) > abs(dqc_lo)        # rate scales the conversion


def test_emergent_wbf_ad_safe():
    """jax.grad through the emergent (zero-Bergeron) mixed-phase path is
    finite wrt cloud water."""
    qsl = float(saturation_specific_humidity(jnp.asarray(260.0), jnp.asarray(6.0e4)))

    def loss(q_c):
        z = jnp.zeros((1, 1))
        hm = HydrometeorState(
            q_c=q_c.reshape(1, 1), q_r=z,
            q_i=jnp.full((1, 1), 1.0e-5), q_s=z, q_g=z,
            N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=jnp.full((1, 1), 1.0e5),
        )
        out = morrison_microphysics(
            jnp.full((1, 1), 260.0), jnp.full((1, 1), qsl), hm,
            jnp.full((1, 1), 6.0e4), jnp.full((1, 2), 6.0e4),
            jnp.full((1, 1), 0.7), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(),
        )
        return jnp.sum(out.dq_i_dt)

    g = jax.grad(loss)(jnp.asarray(5.0e-4))
    assert bool(jnp.isfinite(g))


def test_emergent_wbf_no_spurious_water_source():
    """Codex iter-13 order-of-operations check: the deposition + saturation
    adjustment coupling routes cloud-water mass to ice WITHOUT a spurious
    water source. With tiny ice (negligible sedimentation) the total-water
    tendency ≈ 0, and the residual MATCHES the heuristic scheme (so the
    emergent path adds no new imbalance — only re-partitions the rate)."""
    T, p = 260.0, 6.0e4
    qsl = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), 5.0e-4), q_r=z,
        q_i=jnp.full((1, 1), 1.0e-7), q_s=z, q_g=z,    # tiny ⇒ sed ≈ 0
        N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=jnp.full((1, 1), 1.0e4),
    )

    def total(wbf):
        out = morrison_microphysics(
            jnp.full((1, 1), T), jnp.full((1, 1), qsl), hm,
            jnp.full((1, 1), p), jnp.full((1, 2), p),
            jnp.full((1, 1), 0.7), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(wbf_scheme=wbf),
        )
        return float(
            out.dq_v_dt[0, 0] + out.dq_c_dt[0, 0] + out.dq_r_dt[0, 0]
            + out.dq_i_dt[0, 0] + out.dq_s_dt[0, 0]
        )

    tot_em = total("emergent")
    tot_he = total("bergeron_heuristic")
    assert abs(tot_em) < 1.0e-9                 # no spurious source/sink
    assert tot_em == pytest.approx(tot_he, abs=1e-12)  # same as heuristic


def test_unknown_wbf_scheme_raises():
    with pytest.raises(ValueError, match="Unknown wbf_scheme"):
        _run("bergeron_typo")

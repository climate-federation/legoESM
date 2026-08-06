"""Ice-NUMBER sinks: melting, sublimation, PSD-consistency (SAM semantics).

Regression for the century3/5 N_i surface pile (2026-07-28): sedimented
crystals melted/sublimated their MASS while their NUMBER accumulated forever
(century5 Ni^max argmax at 277 K, surface level; century3 ended at 1e193).
Number must leave WITH the mass, and ORPHAN number (q_i ~ 0, N_i > 0) must be
cleared by the lami-consistency adjustment the rates already assume.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState


def _column(T0, q_i0, n_i0, q_v0=1.0e-4, nlev=4):
    ncol = 1
    T = jnp.full((ncol, nlev), T0)
    q_v = jnp.full((ncol, nlev), q_v0)
    z = jnp.linspace(3000.0, 100.0, nlev)[None, :]
    p_full = jnp.linspace(7.0e4, 9.8e4, nlev)[None, :]
    p_half = jnp.linspace(6.8e4, 1.0e5, nlev + 1)[None, :]
    from legoesm import constants
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 800.0)
    zeros = jnp.zeros((ncol, nlev))
    hyd = HydrometeorState(
        q_c=zeros, q_r=zeros, q_i=jnp.full((ncol, nlev), q_i0),
        q_s=zeros, q_g=zeros, N_c=zeros, N_r=zeros,
        N_i=jnp.full((ncol, nlev), n_i0),
    )
    return T, q_v, hyd, p_full, p_half, rho, dz


def test_melting_removes_number_with_mass():
    """Above freezing, ice number must decay alongside the melting mass —
    the exact century5 surface-pile configuration."""
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        278.0, q_i0=1.0e-5, n_i0=1.0e8)
    out = morrison_microphysics(T, q_v, hyd, p_full, p_half, rho, dz,
                                75.0, MorrisonConfig())
    assert float(jnp.max(out.dN_i_dt)) < 0.0, (
        "warm ice column must LOSE number (melting number sink missing)")


def test_orphan_number_is_cleared_by_consistency_adjustment():
    """q_i ~ 0 with huge N_i (the trap state): mass-proportional sinks vanish,
    so only the lami-consistency adjustment can clear it."""
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        278.0, q_i0=0.0, n_i0=1.0e12)
    out = morrison_microphysics(T, q_v, hyd, p_full, p_half, rho, dz,
                                75.0, MorrisonConfig())
    n_new = 1.0e12 + 75.0 * np.asarray(out.dN_i_dt)
    assert n_new.max() < 1.0e12 * 1e-3, (
        f"orphan number persists: {n_new.max():.2e} (consistency adjustment "
        "missing or too weak)")


def test_cold_saturated_ice_keeps_its_number():
    """Non-vacuity partner: a COLD, ice-supersaturated, PSD-consistent cell
    must NOT have its number stripped by the new sinks."""
    from legoesm.thermo import saturation_mixing_ratio_ice
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        228.0, q_i0=1.0e-5, n_i0=1.0e6)
    q_sat = saturation_mixing_ratio_ice(T, p_full)
    out = morrison_microphysics(T, q_sat * 1.05, hyd, p_full, p_half, rho,
                                dz, 75.0, MorrisonConfig())
    dN = np.asarray(out.dN_i_dt)
    n_new = 1.0e6 + 75.0 * dN
    assert n_new.min() > 0.5e6, (
        "cold consistent ice lost most of its number — sinks over-firing")


def test_grad_finite_through_number_sinks():
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        278.0, q_i0=1.0e-6, n_i0=1.0e8)

    def loss(n0):
        h = hyd._replace(N_i=jnp.full_like(hyd.N_i, n0))
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz,
                                    75.0, MorrisonConfig())
        return jnp.sum(out.dN_i_dt ** 2)

    g = jax.grad(loss)(1.0e8)
    assert np.isfinite(float(g))


def test_clear_air_creates_no_number():
    """Codex round 1: a q_i floor in the consistency lower bound INVENTED
    number in clear air (broke Cooper tests).  q_i=N_i=0 subsaturated air
    must emit exactly zero dN_i_dt."""
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        260.0, q_i0=0.0, n_i0=0.0, q_v0=1.0e-6)
    out = morrison_microphysics(T, q_v, hyd, p_full, p_half, rho, dz,
                                75.0, MorrisonConfig())
    np.testing.assert_array_equal(np.asarray(out.dN_i_dt), 0.0)


def test_post_step_lami_within_bounds():
    """Codex round 1: the limiter must act on the POST-step state (SAM
    in-place reset), not the old one — an old-state adjustment alternated
    LAMI between 0 and lami_min across steps.  After one step from a
    lami_min-consistent state, post-step LAMI must sit inside
    [lami_min, lami_max] wherever post-step mass is above QSMALL."""
    cfg = MorrisonConfig()
    q_i0 = 1.0e-5
    n_min = float(cfg.lami_min) ** 3 * q_i0 / (np.pi * cfg.rho_cloud_ice)
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        250.0, q_i0=q_i0, n_i0=n_min)
    out = morrison_microphysics(T, q_v, hyd, p_full, p_half, rho, dz,
                                75.0, cfg)
    q_new = np.asarray(hyd.q_i + 75.0 * out.dq_i_dt)
    n_new = np.asarray(hyd.N_i + 75.0 * out.dN_i_dt)
    m = q_new > 1.0e-14
    lami = (np.pi * cfg.rho_cloud_ice * n_new[m] / q_new[m]) ** (1.0 / 3.0)
    assert (lami >= float(cfg.lami_min) * (1 - 1e-6)).all()
    assert (lami <= float(cfg.lami_max) * (1 + 1e-6)).all()


def test_melted_ice_number_reaches_rain():
    """The melted crystal number must arrive in the rain number budget
    (per-volume), mirroring the snow/graupel channels."""
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        278.0, q_i0=1.0e-5, n_i0=1.0e8)
    out = morrison_microphysics(T, q_v, hyd, p_full, p_half, rho, dz,
                                75.0, MorrisonConfig())
    assert float(jnp.max(out.dN_r_dt)) > 0.0, (
        "melting ice must source rain NUMBER (NMLTR channel)")


def test_grad_finite_at_zero_ice_mass():
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        260.0, q_i0=0.0, n_i0=1.0e10)

    def loss(n0):
        h = hyd._replace(N_i=jnp.full_like(hyd.N_i, n0))
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz,
                                    75.0, MorrisonConfig())
        return jnp.sum(out.dN_i_dt ** 2)

    assert np.isfinite(float(jax.grad(loss)(1.0e10)))

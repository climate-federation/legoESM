"""Physically-coherent per-column promotion of the CLUBB-lite eddy-diffusivity
coefficient ``C_K`` (``K_m = C_K·l·√(wp2)``) — the coefficient the LES
eddy-diffusivity diagnosis directly informs (docs/COMPARE_REANALYSIS.md).

Verifies the promotion is non-breaking (a uniform per-column ``C_K`` reproduces
the scalar default exactly) and reaches the physics (a per-column ``C_K``
produces per-column-varying ``K_m``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

jax.config.update("jax_enable_x64", True)


def _inputs(ncol=3, nlev=12):
    rng = np.random.default_rng(0)
    p_half = np.linspace(2.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_full = jnp.asarray(np.tile(np.linspace(15000.0, 50.0, nlev), (ncol, 1)))
    z_half = jnp.asarray(np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)))
    exner = (p_full / constants.p_ref) ** constants.kappa
    theta = 290.0 + 3e-3 * np.asarray(z_full)
    T = jnp.asarray(theta * exner)
    u = jnp.asarray(8.0 + 4.0 * rng.standard_normal((ncol, nlev)))
    v = jnp.asarray(2.0 * rng.standard_normal((ncol, nlev)))
    q_v = jnp.asarray(2e-3 + 4e-3 * rng.random((ncol, nlev)))
    rho = jnp.asarray(p_full) / (constants.R_d * T)
    return dict(
        u=u, v=v, T=T, q_v=q_v, tke=jnp.full((ncol, nlev), 0.4),
        p_full=jnp.asarray(p_full), p_half=jnp.asarray(p_half),
        z_full=z_full, z_half=z_half, T_sfc=T[:, -1] + 1.0, q_sfc=q_v[:, -1],
        rho=rho, dt=300.0,
    )


def test_uniform_per_column_ck_matches_scalar():
    """A uniform (ncol,) C_K reproduces the scalar default EXACTLY (no
    regression from the broadcast_column_param wrap)."""
    kw = _inputs()
    ncol = kw["T"].shape[0]
    cfg = CLUBBLiteConfig()
    out_scalar, _ = clubb_lite_turbulence(**kw, config=cfg)

    cfg_uniform = cfg._replace(C_K=jnp.full((ncol,), cfg.C_K))
    out_uniform, _ = clubb_lite_turbulence(**kw, config=cfg_uniform)

    # A uniform per-column C_K is a strict no-op vs the scalar path (each row is
    # scaled by the identical value) ⇒ BYTE-identical, not just close.
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Km), np.asarray(out_scalar.Km))
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Kh), np.asarray(out_scalar.Kh))


def test_per_column_ck_changes_km_per_column():
    """A per-column C_K scales each column's K_m proportionally (the LES
    correction reaches the GCM diffusivity)."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    out_base, _ = clubb_lite_turbulence(**kw, config=cfg)

    # Double C_K in column 1 only.
    ck = jnp.array([cfg.C_K, 2.0 * cfg.C_K, cfg.C_K])
    out_pc, _ = clubb_lite_turbulence(**kw, config=cfg._replace(C_K=ck))

    Km_base = np.asarray(out_base.Km)
    Km_pc = np.asarray(out_pc.Km)
    # Column 0 and 2 unchanged; column 1 K_m doubled (Km = C_K·l·sqrt(wp2),
    # linear in C_K at fixed l, wp2 — same inputs).
    np.testing.assert_allclose(Km_pc[0], Km_base[0], rtol=1e-12)
    np.testing.assert_allclose(Km_pc[2], Km_base[2], rtol=1e-12)
    np.testing.assert_allclose(Km_pc[1], 2.0 * Km_base[1], rtol=1e-12)


def test_promotion_registered_and_applies():
    """C_K is registered promotable and apply_feedback_to_scheme splices it."""
    from legoesm.training.promotable_params import (
        PROMOTABLE_FIELDS,
        apply_feedback_to_scheme,
    )

    assert "clubb_lite_C_K" in PROMOTABLE_FIELDS
    cfg = CLUBBLiteConfig()
    new = apply_feedback_to_scheme(cfg, "clubb_lite_C_K", jnp.array([0.3, 0.5, 0.7]))
    assert new.C_K.shape == (3,)
    np.testing.assert_allclose(np.asarray(new.C_K), [0.3, 0.5, 0.7])


def test_uniform_per_column_prt_matches_scalar():
    """A uniform (ncol,) Pr_t reproduces the scalar default EXACTLY (the
    broadcast_column_param wrap on Pr_t is a no-op for the scalar path)."""
    kw = _inputs()
    ncol = kw["T"].shape[0]
    cfg = CLUBBLiteConfig()
    out_scalar, _ = clubb_lite_turbulence(**kw, config=cfg)

    cfg_uniform = cfg._replace(Pr_t=jnp.full((ncol,), cfg.Pr_t))
    out_uniform, _ = clubb_lite_turbulence(**kw, config=cfg_uniform)
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Kh), np.asarray(out_scalar.Kh))
    # Km is independent of Pr_t (Kh = Km/Pr_t) ⇒ unchanged either way.
    np.testing.assert_array_equal(
        np.asarray(out_uniform.Km), np.asarray(out_scalar.Km))


def test_per_column_prt_changes_kh_per_column():
    """A per-column Pr_t scales each column's K_h inversely (Kh = Km/Pr_t); Km
    is unaffected — the LES Prandtl-number correction reaches the GCM."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    out_base, _ = clubb_lite_turbulence(**kw, config=cfg)

    prt = jnp.array([cfg.Pr_t, 2.0 * cfg.Pr_t, cfg.Pr_t])   # double Pr_t in col 1
    out_pc, _ = clubb_lite_turbulence(**kw, config=cfg._replace(Pr_t=prt))

    Kh_base, Kh_pc = np.asarray(out_base.Kh), np.asarray(out_pc.Kh)
    np.testing.assert_allclose(Kh_pc[0], Kh_base[0], rtol=1e-12)
    np.testing.assert_allclose(Kh_pc[2], Kh_base[2], rtol=1e-12)
    np.testing.assert_allclose(Kh_pc[1], 0.5 * Kh_base[1], rtol=1e-12)  # Kh halved
    # Km is unchanged by Pr_t.
    np.testing.assert_array_equal(np.asarray(out_pc.Km), np.asarray(out_base.Km))


def test_prt_promotion_registered_and_applies():
    from legoesm.training.promotable_params import (
        PROMOTABLE_FIELDS,
        apply_feedback_to_scheme,
    )

    assert "clubb_lite_Pr_t" in PROMOTABLE_FIELDS
    new = apply_feedback_to_scheme(
        CLUBBLiteConfig(), "clubb_lite_Pr_t", jnp.array([0.7, 0.9, 1.1]))
    assert new.Pr_t.shape == (3,)
    np.testing.assert_allclose(np.asarray(new.Pr_t), [0.7, 0.9, 1.1])


def test_uniform_per_column_ceps_matches_scalar():
    """A uniform (ncol,) C_eps reproduces the scalar default EXACTLY."""
    kw = _inputs()
    ncol = kw["T"].shape[0]
    cfg = CLUBBLiteConfig()
    _, wp2_scalar = clubb_lite_turbulence(**kw, config=cfg)
    _, wp2_uniform = clubb_lite_turbulence(
        **kw, config=cfg._replace(C_eps=jnp.full((ncol,), cfg.C_eps)))
    np.testing.assert_array_equal(np.asarray(wp2_uniform), np.asarray(wp2_scalar))


def test_per_column_ceps_changes_wp2_per_column():
    """A per-column C_eps changes the wp2 dissipation per column (reaches the body)."""
    kw = _inputs()
    cfg = CLUBBLiteConfig()
    _, wp2_base = clubb_lite_turbulence(**kw, config=cfg)
    ceps = jnp.array([cfg.C_eps, 2.0 * cfg.C_eps, cfg.C_eps])  # double in col 1
    _, wp2_pc = clubb_lite_turbulence(**kw, config=cfg._replace(C_eps=ceps))
    np.testing.assert_array_equal(np.asarray(wp2_pc)[0], np.asarray(wp2_base)[0])
    assert not np.allclose(np.asarray(wp2_pc)[1], np.asarray(wp2_base)[1])  # more diss

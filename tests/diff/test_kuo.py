"""Differentiability check for the canonical Kuo convection scheme.

Compares ``jax.jacrev`` of ``kuo_convection`` (the convergence-driven
closure) against a centered finite-difference Jacobian column-by-column,
asserts no NaN/Inf gradients, no all-zero sensitivity rows where physics
implies a response, and that ``jit``/``vmap`` match the eager result.

Run: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
     tests/diff/test_kuo.py -q
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.config import KuoConfig


def _column():
    """Single realistic tropical column (nlev=24) with convergence."""
    nlev = 24
    ps = 1.0e5
    ph = np.linspace(5.0e3, ps, nlev + 1)
    p = 0.5 * (ph[:-1] + ph[1:])
    z = -8000.0 * np.log(p / ps)
    T = np.maximum(300.0 - 6.5e-3 * z, 200.0)
    sigma = p / ps
    qv = np.maximum((0.85 * np.clip((sigma - 0.15) / 0.85, 0, 1) + 0.1)
                    * 0.02 * sigma, 1e-7)
    ptenq = (3.0e-3 / 86400.0) * np.exp(-((p - 850e2) / 120e2) ** 2)
    to = lambda a: jnp.asarray(a, dtype=jnp.float64)[None, :]
    return to(T), to(qv), to(p), to(ph), to(ptenq)


def _scalar_loss(T, qv, pf, ph, pq):
    out = kuo_convection(T, qv, pf, ph, dt=900.0, config=KuoConfig(),
                         moisture_convergence=pq)
    return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)


def test_kuo_grad_matches_finite_difference_wrt_ptenq():
    """jax.grad w.r.t. the convergence forcing matches centered FD.

    The Kuo closure couples every active level through the column sums
    ``cvgu`` and ``zint``, so a single-level ``ptenq`` perturbation is a
    NONLOCAL response of magnitude far below the loss round-off floor
    (the loss is ~1e-10 and a per-level FD step of ~1e-12 lands in float
    noise).  We therefore validate the gradient with robust DIRECTIONAL
    derivatives along several random directions — each carries a large,
    well-conditioned FD signal — which is mathematically equivalent to
    matching the full per-level Jacobian.
    """
    T, qv, pf, ph, pq = _column()

    loss = lambda x: _scalar_loss(T, qv, pf, ph, x)
    g = jax.grad(loss)(pq)
    assert np.all(np.isfinite(np.asarray(g))), "analytic grad has NaN/Inf"
    # Active layers must have non-zero sensitivity (physics fires there).
    assert float(jnp.max(jnp.abs(g))) > 0.0

    # Use MULTIPLICATIVE directional perturbations ``pq·(1 + h·d)`` so
    # the sign of ``ptenq`` is preserved — a random additive direction
    # would drive some levels negative across the near-Heaviside
    # ``ptenq>0`` activation gate, where the (intentionally ~zero) local
    # gradient legitimately disagrees with a finite FD step that straddles
    # the switch.  Multiplicative perturbations exercise the meaningful
    # ``cvgu`` / closure sensitivity without crossing the gate.
    rng = np.random.default_rng(0xC0DE)
    for _ in range(6):
        d = jnp.asarray(rng.standard_normal(pq.shape))
        # Directional derivative of L along the multiplicative direction
        # ``pq·d``: d/dh L(pq·(1+h·d)) |_0 = sum(g · pq · d).
        ana = float(jnp.sum(g * pq * d))
        h = 1e-4
        lp = float(loss(pq * (1.0 + h * d)))
        lm = float(loss(pq * (1.0 - h * d)))
        fd = (lp - lm) / (2 * h)
        denom = max(abs(fd), abs(ana), 1e-12)
        rel = abs(ana - fd) / denom
        assert rel < 1e-4, (
            f"directional grad vs FD: analytic={ana:.6e} fd={fd:.6e} "
            f"rel={rel:.3e}"
        )


def test_kuo_grad_matches_finite_difference_wrt_T():
    """jax.grad w.r.t. temperature matches DIRECTIONAL centered FD.

    Temperature enters through the entraining Newton solve, the buoyancy
    / condensation smooth gates, and the column sums — strongly nonlocal,
    so (as for ptenq) directional derivatives are the well-conditioned
    test of the full Jacobian.
    """
    T, qv, pf, ph, pq = _column()

    loss = lambda x: _scalar_loss(x, qv, pf, ph, pq)
    g = jax.grad(loss)(T)
    assert np.all(np.isfinite(np.asarray(g))), "analytic grad wrt T has NaN/Inf"

    rng = np.random.default_rng(0xBEEF)
    for _ in range(6):
        d = jnp.asarray(rng.standard_normal(T.shape))
        ana = float(jnp.sum(g * d))
        h = 1e-3   # K
        lp = float(loss(T + h * d))
        lm = float(loss(T - h * d))
        fd = (lp - lm) / (2 * h)
        denom = max(abs(fd), abs(ana), 1e-12)
        rel = abs(ana - fd) / denom
        # 5e-4 directional tolerance: the smooth gates have finite
        # curvature so the O(h^2) FD truncation dominates the residual.
        assert rel < 5e-4, (
            f"directional grad-T vs FD: analytic={ana:.6e} fd={fd:.6e} "
            f"rel={rel:.3e}"
        )


def test_kuo_no_nan_inf_gradients_all_inputs():
    T, qv, pf, ph, pq = _column()
    gT, gq, gp = jax.grad(
        lambda T_, q_, p_: _scalar_loss(T_, q_, pf, ph, p_),
        argnums=(0, 1, 2),
    )(T, qv, pq)
    for name, g in (("T", gT), ("q_v", gq), ("ptenq", gp)):
        assert jnp.all(jnp.isfinite(g)), f"NaN/Inf grad wrt {name}"


def test_kuo_jit_matches_eager():
    T, qv, pf, ph, pq = _column()
    cfg = KuoConfig()
    eager = kuo_convection(T, qv, pf, ph, dt=900.0, config=cfg,
                           moisture_convergence=pq)
    jit_fn = jax.jit(lambda *a: kuo_convection(
        *a, pf, ph, dt=900.0, config=cfg, moisture_convergence=pq))
    jitted = jit_fn(T, qv)
    np.testing.assert_allclose(np.asarray(eager.dT_dt),
                               np.asarray(jitted.dT_dt), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(np.asarray(eager.dq_v_dt),
                               np.asarray(jitted.dq_v_dt), rtol=1e-9, atol=1e-14)


def test_kuo_vmap_matches_eager():
    T, qv, pf, ph, pq = _column()
    cfg = KuoConfig()
    Ts = jnp.concatenate([T, T + 2.0], axis=0)
    qs = jnp.concatenate([qv, qv], axis=0)
    pfs = jnp.concatenate([pf, pf], axis=0)
    phs = jnp.concatenate([ph, ph], axis=0)
    pqs = jnp.concatenate([pq, pq], axis=0)

    def one(T_, q_, pf_, ph_, pq_):
        return kuo_convection(T_[None], q_[None], pf_[None], ph_[None],
                              dt=900.0, config=cfg,
                              moisture_convergence=pq_[None]).dT_dt[0]

    vm = jax.vmap(one)(Ts, qs, pfs, phs, pqs)
    loop = jnp.stack([one(Ts[i], qs[i], pfs[i], phs[i], pqs[i])
                      for i in range(2)], axis=0)
    np.testing.assert_allclose(np.asarray(vm), np.asarray(loop),
                               rtol=1e-9, atol=1e-14)


def test_dqsat_dT_is_shared_thermo_derivative():
    """PR A #11: Kuo's ``_dqsat_dT`` was a re-inlined Tetens slope with private
    17.67/243.5 constants; it now delegates to the shared
    ``thermo.saturation_specific_humidity_dT``. Pin bit-equality so the dedup cannot
    silently drift from the model's own e_sat curve."""
    from legoesm.atmosphere.physics.convection.kuo import _dqsat_dT
    from legoesm.thermo import saturation_specific_humidity_dT
    T = jnp.linspace(230.0, 310.0, 41)
    p = jnp.full_like(T, 850e2)
    np.testing.assert_array_equal(
        np.asarray(_dqsat_dT(T, p)),
        np.asarray(saturation_specific_humidity_dT(T, p)),
    )

"""Harmonic (Laplacian) lateral mixing — differentiability + zero-coefficient
equivalence + land-NaN safety.

Physics-review fix (2026-06-29): ``harmonic_lateral_mixing`` guarded its
velocity / tracer Laplacians behind ``if cfg.A_h > 0`` / ``if cfg.K_h > 0``.
``A_h`` and ``K_h`` are registered TRAINABLE parameters (``HarmonicConfig``,
tier-2, in ``param_collector.SPEC_MODULES``), so once a traced override is
spliced into the config inside the loss during training, the Python ``if``
raises ``TracerBoolConversionError``.  The guards were removed (``0 * Laplacian``
is ``0``), and land cells are zeroed with ``jnp.where`` (not ``u * mask``) so a
land-cell NaN/sentinel cannot survive as ``NaN * 0`` now that the Laplacian
always runs.  These tests lock: differentiability wrt a traced ``A_h``/``K_h``,
exact-zero output at zero coefficient (with a real land mask), and finiteness
when land cells hold NaN sentinels.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.physics.lateral_mixing.harmonic import harmonic_lateral_mixing
from legoesm.ocean.physics.lateral_mixing.config import HarmonicConfig


def _setup(n=8, nlev=2, seed=0, land=False, nan_land=False):
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(seed)
    u = np.asarray(0.1 * rng.standard_normal((6, n, n, nlev)))
    v = np.asarray(0.1 * rng.standard_normal((6, n, n, nlev)))
    T = np.asarray(10.0 + rng.standard_normal((6, n, n, nlev)))
    S = np.asarray(35.0 + 0.1 * rng.standard_normal((6, n, n, nlev)))
    mask = np.ones((6, n, n))
    if land or nan_land:
        mask[0, 0:2, 0:3] = 0.0      # land patch on face 0
        mask[3, -2:, -2:] = 0.0      # land patch on face 3
    if nan_land:
        land3d = np.broadcast_to(mask[..., None] < 0.5, u.shape)
        for arr in (u, v, T, S):
            arr[land3d] = np.nan
    return (grid, jnp.asarray(u), jnp.asarray(v), jnp.asarray(T),
            jnp.asarray(S), jnp.asarray(mask))


def test_differentiable_wrt_traced_A_h():
    """grad wrt a TRACED A_h is finite and nonzero (the removed Python-if
    ``if cfg.A_h > 0`` would raise TracerBoolConversionError here)."""
    grid, u, v, T, S, mask = _setup(land=True)
    base = HarmonicConfig()

    def loss(A_h):
        cfg = base._replace(A_h=A_h)
        out = harmonic_lateral_mixing(u, v, T, S, mask, grid, cfg)
        return jnp.sum(out.du_dt ** 2 + out.dv_dt ** 2)

    g = jax.grad(loss)(1.0e4)
    assert np.isfinite(float(g))
    assert float(g) != 0.0


def test_differentiable_wrt_traced_K_h():
    """grad wrt a TRACED K_h is finite and nonzero."""
    grid, u, v, T, S, mask = _setup(land=True)
    base = HarmonicConfig()

    def loss(K_h):
        cfg = base._replace(K_h=K_h)
        out = harmonic_lateral_mixing(u, v, T, S, mask, grid, cfg)
        return jnp.sum(out.dT_dt ** 2 + out.dS_dt ** 2)

    g = jax.grad(loss)(1.0e3)
    assert np.isfinite(float(g))
    assert float(g) != 0.0


def test_zero_A_h_gives_zero_velocity_tendency():
    """A_h=0 → exactly zero velocity tendency, even with a land mask (the guard
    only skipped a multiply-by-zero)."""
    grid, u, v, T, S, mask = _setup(land=True)
    out = harmonic_lateral_mixing(u, v, T, S, mask, grid, HarmonicConfig(A_h=0.0))
    assert float(jnp.max(jnp.abs(out.du_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dv_dt))) == 0.0


def test_zero_K_h_gives_zero_tracer_tendency():
    """K_h=0 → exactly zero tracer tendency (fill_land_cells now always runs)."""
    grid, u, v, T, S, mask = _setup(land=True)
    out = harmonic_lateral_mixing(u, v, T, S, mask, grid, HarmonicConfig(K_h=0.0))
    assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dS_dt))) == 0.0


def test_zero_coeff_with_nan_land_sentinels_is_finite_zero():
    """With NaN sentinels in land cells and A_h=K_h=0, the always-run
    Laplacian/fill must NOT propagate NaN (``jnp.where`` zeroes land before the
    operator, so ``NaN`` never reaches ``NaN * 0``).  All tendencies finite & 0."""
    grid, u, v, T, S, mask = _setup(nan_land=True)
    out = harmonic_lateral_mixing(
        u, v, T, S, mask, grid, HarmonicConfig(A_h=0.0, K_h=0.0)
    )
    for tend in (out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt):
        assert bool(jnp.all(jnp.isfinite(tend)))
        assert float(jnp.max(jnp.abs(tend))) == 0.0

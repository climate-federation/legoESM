"""Spectral wind round-trip idempotency across truncations (#976).

The WB2 eval driver (#951) runs ``era5 -A-> state -S-> carry -A-> state``, i.e.
one MORE analysis pass than training/probe ever do.  With the plain Bourke
``oc2``/``dmu`` analysis (``vordiv_from_uv_3d``) that extra pass is NOT the
inverse of the streamfunction/velocity-potential wind synthesis at the
truncation boundary: it amplifies pole-row wind error ~×21 per pass at T85
(and worse at higher truncation), rolling ``p_s`` to NaN and masking all of
``z500``.  The #858-era round-trip contract only held at low truncation.

``vordiv_from_uv_exact_3d`` replaces that analysis with the exact per-``m``
least-squares left-inverse of the synthesis, so ``S(A(x)) == x`` on the
band-limited subspace — pole rows included — at EVERY truncation, and the
composed round trip is idempotent.

These tests FAIL on the old oc2/dmu analysis (gain ~×21 at T85, pole-row error
O(10) m/s and growing) and PASS on the exact inverse (error at machine
precision, no growth), for n_max in {21, 42, 85, 106}, Ask #2.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    uv_from_vordiv_3d,
    vordiv_from_uv_exact_3d,
)

_COS_LAT_MIN = 1.0e-6
_POLE_LAT_DEG = 85.0     # "pole-most rows" band the issue calls out (±88.9° etc.)
_TARGET_UMAX = 80.0      # normalize synthetic winds to a realistic ~80 m/s peak


def _synth_winds(grid, vor_hat, div_hat):
    """Spectral (vor, div) -> grid (u, v), matching spectral_pe_to_grid."""
    u_cos, v_cos = uv_from_vordiv_3d(grid, vor_hat, div_hat)
    cos = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    return u_cos / cos, v_cos / cos


def _band_limited_winds(grid, nlev=1, seed=1):
    """Winds that live EXACTLY in the resolvable spectral subspace: draw a
    smooth red-spectrum (vor, div), synthesize once, normalize to ~80 m/s."""
    rng = np.random.default_rng(seed)
    n_sh = grid.ms.shape[0]
    ls = np.asarray(grid.ls)
    scale = np.where(ls > 0, 1.0 / (ls + 1.0) ** 2, 0.0)

    def _coeffs():
        c = (rng.standard_normal((n_sh, nlev))
             + 1j * rng.standard_normal((n_sh, nlev))) * scale[:, None]
        c[np.asarray(grid.ms) == 0] = c[np.asarray(grid.ms) == 0].real
        return jnp.asarray(c)

    u, v = _synth_winds(grid, _coeffs(), _coeffs())
    s = _TARGET_UMAX / float(np.max(np.abs(np.asarray(u))))
    return u * s, v * s


def _pole_mask(grid):
    return np.abs(np.degrees(np.asarray(grid.lat))) >= _POLE_LAT_DEG


@pytest.mark.parametrize("n_max", [21, 42, 85, 106])
def test_exact_analysis_inverts_synthesis(n_max):
    """A(S(x)) recovers x to machine precision for band-limited winds — the
    exact-inverse property, pole rows included (#976 Ask #1)."""
    grid = create_gaussian_grid(n_max)
    # Start from a KNOWN spectral state, synthesize, analyze back.
    rng = np.random.default_rng(0)
    n_sh = grid.ms.shape[0]
    ls = np.asarray(grid.ls)
    scale = np.where(ls > 0, 1.0 / (ls + 1.0) ** 2, 0.0)[:, None]
    ms = np.asarray(grid.ms)

    def _ref():
        c = (rng.standard_normal((n_sh, 1))
             + 1j * rng.standard_normal((n_sh, 1))) * scale
        c[ms == 0] = c[ms == 0].real   # m=0 must be real for a real field
        c[ls == 0] = 0.0               # no vor/div constant mode
        return jnp.asarray(c)

    vor0 = _ref()
    div0 = _ref()

    u, v = _synth_winds(grid, vor0, div0)
    vor1, div1 = vordiv_from_uv_exact_3d(grid, u, v)

    scl = float(np.max(np.abs(np.asarray(vor0)))) + 1e-30
    assert float(np.max(np.abs(np.asarray(vor1 - vor0)))) / scl < 1e-9
    assert float(np.max(np.abs(np.asarray(div1 - div0)))) / scl < 1e-9


@pytest.mark.parametrize("n_max", [21, 42, 85, 106])
def test_wind_roundtrip_idempotent_including_poles(n_max):
    """S(A(S(A(x)))) vs S(A(x)) wind error is at machine precision, and does
    NOT grow pass-to-pass, INCLUDING the pole-most rows (#976 Ask #2).

    On the old oc2/dmu analysis the pass-2/pass-1 gain is ~×21 at T85 and the
    pole-row error is O(10) m/s; this asserts both away.
    """
    grid = create_gaussian_grid(n_max)
    u0, v0 = _band_limited_winds(grid)

    def rt(u, v):
        vor, div = vordiv_from_uv_exact_3d(grid, u, v)
        return _synth_winds(grid, vor, div)

    u1, v1 = rt(u0, v0)          # S(A(x))
    u2, v2 = rt(u1, v1)          # S(A(S(A(x))))

    du1 = np.abs(np.asarray(u1 - u0))
    du2 = np.abs(np.asarray(u2 - u1))
    m1 = float(du1.max())
    m2 = float(du2.max())

    # (a) round trip reproduces the input to ~1e-8 of the ~80 m/s signal
    assert m1 < 1e-6 * _TARGET_UMAX, f"S(A(x)) err {m1:.3e} m/s at T{n_max}"
    # (b) idempotent: the second pass adds no growth (gain ~1, never ~21)
    assert m2 < 5.0 * max(m1, 1e-12), f"pass-2 gain blew up at T{n_max}"
    # (c) pole-most rows are NOT special — same machine-precision bound there
    pole = _pole_mask(grid)
    pole_err = float(du2[pole].max()) if pole.any() else 0.0
    assert pole_err < 1e-6 * _TARGET_UMAX, (
        f"pole-row err {pole_err:.3e} m/s at T{n_max}")


@pytest.mark.parametrize("n_max", [21, 85])
def test_carry_state_carry_roundtrip_idempotent(n_max):
    """The exact #976 eval-path repro on the public API: iterating
    ``spectral_state_to_carry(carry_to_spectral_state(.))`` does not amplify
    winds (esp. at the poles).  This is what blew up the WB2 driver."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_state_to_carry,
    )

    nlev = 4
    grid = create_gaussian_grid(n_max)
    sigma = create_sigma_coordinate(nlev)
    u, v = _band_limited_winds(grid, nlev=nlev, seed=3)

    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    s3, s2 = (n_lat, n_lon, nlev), (n_lat, n_lon)
    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    T = jnp.asarray(np.broadcast_to(
        250.0 + 40.0 * np.linspace(0.0, 1.0, nlev), s3).copy())
    state_h = HydrostaticState(
        u=Field(u, name="u", dims=d3, units="m/s"),
        v=Field(v, name="v", dims=d3, units="m/s"),
        T=Field(T, name="T", dims=d3, units="K"),
        p_s=Field(jnp.full(s2, 1.0e5), name="p_s", dims=d2, units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=d2, units="m2/s2"),
    )
    carry0 = pack_carry(
        state_h, q_v=jnp.zeros(s3), q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        held_dT_rad=jnp.zeros(s3), held_sw_net_sfc=jnp.zeros(s2),
        held_lw_net_sfc=jnp.zeros(s2), held_sw_up_toa=jnp.zeros(s2),
        held_lw_up_toa=jnp.zeros(s2), held_sw_down_toa=jnp.zeros(s2),
        step_index=0,
    )

    def rt(c):
        return spectral_state_to_carry(
            carry_to_spectral_state(c, grid), grid, sigma)

    c1 = rt(carry0)
    c2 = rt(c1)

    du1 = np.abs(np.asarray(c1.u) - np.asarray(carry0.u))
    du2 = np.abs(np.asarray(c2.u) - np.asarray(c1.u))
    m1, m2 = float(du1.max()), float(du2.max())
    # winds stay bounded and idempotent (old code: c1 err ~150, gain ~21)
    assert m1 < 1e-4 * _TARGET_UMAX, f"carry rt1 err {m1:.3e} m/s at T{n_max}"
    assert m2 < 5.0 * max(m1, 1e-12), f"carry rt gain blew up at T{n_max}"
    pole = _pole_mask(grid)
    if pole.any():
        assert float(du2[pole].max()) < 1e-4 * _TARGET_UMAX


def test_exact_analysis_is_differentiable():
    """vordiv_from_uv_exact_3d stays reverse-mode differentiable (grad flows
    through the FFT + matmul + scatter; the pinv operators are constants)."""
    grid = create_gaussian_grid(21)
    u, v = _band_limited_winds(grid, nlev=2, seed=7)

    def loss(u_, v_):
        vor, div = vordiv_from_uv_exact_3d(grid, u_, v_)
        return jnp.real(jnp.vdot(vor, vor) + jnp.vdot(div, div))

    g_u, g_v = jax.grad(loss, argnums=(0, 1))(u, v)
    assert np.all(np.isfinite(np.asarray(g_u)))
    assert np.all(np.isfinite(np.asarray(g_v)))
    assert float(np.max(np.abs(np.asarray(g_u)))) > 0.0


def test_exact_analysis_jittable():
    """The exact analysis traces under jax.jit (grid a closure constant) and
    matches the eager result to machine precision."""
    grid = create_gaussian_grid(21)
    u, v = _band_limited_winds(grid, nlev=3, seed=11)

    eager = vordiv_from_uv_exact_3d(grid, u, v)
    jitted = jax.jit(lambda u_, v_: vordiv_from_uv_exact_3d(grid, u_, v_))(u, v)
    for a, b in zip(eager, jitted):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)


def test_exact_analysis_first_call_under_trace_does_not_poison_cache():
    """JIT-FIRST ordering: the operator cache must never retain a trace's
    constants. When the first-ever call for a grid happens inside jit/grad,
    caching the in-trace ``jnp.asarray`` results stores that trace's
    DynamicJaxprTracers; every later trace or eager call then dies with
    UnexpectedTracerError at the einsum that consumes them (2026-08-16: this
    killed the jax_debug_nans re-trace during the WB sample-17 NaN hunt —
    the sibling jittable test runs eager first and could never see it)."""
    from legoesm.grids import gaussian as _g

    grid = create_gaussian_grid(21)
    u, v = _band_limited_winds(grid, nlev=2, seed=13)

    # Force the poisoning ordering: nothing cached, first call inside a trace.
    _g._VORDIV_PINV_CACHE.clear()
    jit1 = jax.jit(lambda u_, v_: vordiv_from_uv_exact_3d(grid, u_, v_))(u, v)
    # Second trace and eager call both must survive and agree.
    jit2 = jax.jit(lambda u_, v_: vordiv_from_uv_exact_3d(grid, u_, v_))(u, v)
    eager = vordiv_from_uv_exact_3d(grid, u, v)
    for a, b in zip(jit1, jit2):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)
    for a, b in zip(jit1, eager):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)

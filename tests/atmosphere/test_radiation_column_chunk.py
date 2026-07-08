"""Acceptance gates for RRTMGP column-chunking (``RRTMGP.solve_columns_chunked``).

Radiation columns are physically INDEPENDENT (the k-distribution and two-stream
solve have no horizontal coupling), so chunking the column axis and mapping the
solver over fixed-size blocks with ``jax.lax.map`` must be:

  1. NUMERICALLY EXACT vs the unchunked solve — the whole point is that it only
     caps the (highly super-linear) XLA compile time, it does NOT change the
     answer; and
  2. fully reverse-mode differentiable (``lax.map`` is scan-based), with the
     chunked gradient matching the unchunked gradient.

These are the hard gates; run under ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP


def _make_inputs(ncol, nlev, seed=0):
    """Smooth, physical multi-column input with every column DISTINCT.

    Distinct per-column temperatures/zenith angles are deliberate: if the
    reshape ever reordered columns, a distinct-column input would change the
    answer and trip the exactness gate (an all-identical-column input would
    silently pass a reorder bug).
    """
    rng = np.random.default_rng(seed)
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    t_off = jnp.asarray(rng.uniform(-8.0, 8.0, size=(ncol, 1)))
    T = jnp.broadcast_to(
        jnp.linspace(210.0, 295.0, nlev)[None, :], (ncol, nlev),
    ) + t_off
    sfc_temperature = 295.0 + t_off[:, 0]
    q_v = jnp.broadcast_to(
        jnp.linspace(3.0e-6, 8.0e-3, nlev)[None, :], (ncol, nlev),
    )
    cos_zenith = jnp.asarray(rng.uniform(0.2, 0.9, size=(ncol,)))
    return dict(
        T=T, p_full=p_full, p_half=p_half,
        sfc_temperature=sfc_temperature, q_v=q_v, cos_zenith=cos_zenith,
    )


@pytest.fixture(scope="module")
def solver():
    # Empty gas files => jax-rrtmgp bundled default optics tables.
    return RRTMGP.from_legoesm_config(RRTMGPConfig())


@pytest.mark.parametrize("chunk", [2, 4])
def test_chunked_forward_is_exact(solver, chunk):
    """Every RadiationOutput leaf from the chunked solve equals the unchunked
    solve to atol=1e-10 (rtol=0) — numerically exact, not an approximation."""
    ncol, nlev = 8, 12
    inp = _make_inputs(ncol, nlev)
    ref = solver.solve_columns(**inp)
    got = solver.solve_columns_chunked(column_chunk_size=chunk, **inp)

    ref_leaves = jax.tree_util.tree_leaves(ref)
    got_leaves = jax.tree_util.tree_leaves(got)
    assert ref_leaves, "no array leaves in RadiationOutput"
    assert len(got_leaves) == len(ref_leaves)
    for r, g in zip(ref_leaves, got_leaves):
        assert g.shape == r.shape
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(r), rtol=0.0, atol=1e-10,
        )


def test_chunked_gradient_finite_and_matches(solver):
    """jax.grad of a scalar (sum of sw_flux_up) w.r.t. T through the chunked
    path is finite and matches the unchunked gradient (AD-safe: lax.map is
    scan-based)."""
    ncol, nlev = 8, 12
    inp = _make_inputs(ncol, nlev, seed=1)
    T0 = inp["T"]
    rest = {k: v for k, v in inp.items() if k != "T"}

    def loss_unchunked(T):
        return jnp.sum(solver.solve_columns(T=T, **rest).sw_flux_up)

    def loss_chunked(T):
        out = solver.solve_columns_chunked(column_chunk_size=4, T=T, **rest)
        return jnp.sum(out.sw_flux_up)

    g_ref = jax.grad(loss_unchunked)(T0)
    g_chunk = jax.grad(loss_chunked)(T0)

    assert jnp.all(jnp.isfinite(g_chunk))
    assert g_chunk.shape == T0.shape
    np.testing.assert_allclose(
        np.asarray(g_chunk), np.asarray(g_ref), rtol=0.0, atol=1e-8,
    )


def test_chunk_size_must_divide_ncol(solver):
    """A chunk size that does not divide ncol is a hard error, not a silent
    reshape corruption."""
    inp = _make_inputs(ncol=8, nlev=12)
    with pytest.raises(ValueError, match="divide"):
        solver.solve_columns_chunked(column_chunk_size=3, **inp)


def test_chunk_disabled_is_plain_solve(solver):
    """chunk<=0 or chunk>=ncol is an exact passthrough to solve_columns."""
    ncol, nlev = 6, 12
    inp = _make_inputs(ncol, nlev)
    ref = solver.solve_columns(**inp)
    for k in (0, ncol, ncol + 4):
        got = solver.solve_columns_chunked(column_chunk_size=k, **inp)
        for r, g in zip(jax.tree_util.tree_leaves(ref),
                        jax.tree_util.tree_leaves(got)):
            np.testing.assert_allclose(
                np.asarray(g), np.asarray(r), rtol=0.0, atol=1e-10,
            )


def test_optical_field_only_rejected(solver):
    """The per-g-point optics-only tuple return has no (n_block, chunk) column
    axis, so chunking must reject it rather than corrupt the reshape."""
    inp = _make_inputs(ncol=8, nlev=12)
    with pytest.raises(ValueError, match="optics-only"):
        solver.solve_columns_chunked(
            column_chunk_size=4, sw_optical_field_only=True, **inp,
        )


def test_solar_spectral_fraction_never_chunked(solver):
    """A per-g-point ``solar_spectral_fraction`` (leading axis n_gpt, NOT ncol)
    must be broadcast to every block, never split — even in the degenerate
    ncol == n_gpt_sw case a pure shape heuristic would misclassify.

    Non-vacuous regression: with ncol == n_gpt_sw, an unguarded ``shape[0] ==
    ncol`` classifier would split the weight vector so each block sees a short
    vector and ``solve_columns`` raises on the wrong g-point length; the
    by-name exclusion keeps it static and the answer exact.
    """
    n = int(solver.optics_lib.n_gpt_sw)
    spf = next((p for p in range(2, n + 1) if n % p == 0), n)  # smallest factor
    chunk = n // spf  # largest proper divisor (fewest blocks); 1 iff n prime
    if chunk <= 1 or chunk >= n:
        pytest.skip(f"n_gpt_sw={n} is prime; no interior chunk divides it")
    ncol, nlev = n, 6  # ncol == n_gpt_sw is the collision trigger
    inp = _make_inputs(ncol, nlev)
    weights = jnp.asarray(np.linspace(0.5, 1.5, n))  # arbitrary positive weights
    ref = solver.solve_columns(solar_spectral_fraction=weights, **inp)
    got = solver.solve_columns_chunked(
        column_chunk_size=chunk, solar_spectral_fraction=weights, **inp,
    )
    for r, g in zip(jax.tree_util.tree_leaves(ref),
                    jax.tree_util.tree_leaves(got)):
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(r), rtol=0.0, atol=1e-10,
        )


def test_per_column_config_surface_fallback_rejected():
    """Active chunking + a per-column (ncol,) CONFIG surface fallback with NO
    explicit override must RAISE, not silently break.

    solve_columns reads ``self._config.sfc_albedo`` at GLOBAL ncol via
    ``_resolve_surface_field`` when the override is None; only explicit
    per-call overrides are chunked, so a per-column config array would be read
    at global ncol inside each chunk-local block → size mismatch. The guard
    fails loudly and directs the caller to pass the field explicitly.
    """
    ncol, nlev = 8, 12
    inp = _make_inputs(ncol, nlev)
    # config.sfc_albedo is a (ncol,) array; inp carries NO sfc_albedo override.
    cfg = RRTMGPConfig(sfc_albedo=jnp.full((ncol,), 0.1))
    solver2 = RRTMGP.from_legoesm_config(cfg)
    with pytest.raises(ValueError, match="config fallback"):
        solver2.solve_columns_chunked(column_chunk_size=4, **inp)


def test_optics_only_passthrough_when_disabled(solver):
    """column_chunk_size=0 is a TRUE passthrough even with an optics-only flag:
    it returns exactly the plain optics-only solve_columns result (a per-g-point
    tuple), NOT a ValueError (the disabled check precedes the optics rejection).
    """
    ncol, nlev = 8, 12
    inp = _make_inputs(ncol, nlev)
    ref = solver.solve_columns(sw_optical_field_only=True, **inp)
    got = solver.solve_columns_chunked(
        column_chunk_size=0, sw_optical_field_only=True, **inp,
    )
    ref_leaves = jax.tree_util.tree_leaves(ref)
    got_leaves = jax.tree_util.tree_leaves(got)
    assert ref_leaves and len(got_leaves) == len(ref_leaves)
    for r, g in zip(ref_leaves, got_leaves):
        np.testing.assert_allclose(
            np.asarray(g), np.asarray(r), rtol=0.0, atol=1e-10,
        )

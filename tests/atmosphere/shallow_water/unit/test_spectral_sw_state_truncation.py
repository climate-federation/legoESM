"""Regression: spectral-SW 2/3-rule must truncate the STATE, not only the
tendency (2026-07-12 audit).

The dealiasing mask applied to the tendencies holds masked modes CONSTANT
(d/dt = 0), so any upper-third power already in the state — e.g. the
Williamson-5 conical mountain imprinted on prognostic phi = g(h - h_s),
or arbitrary user ICs — was frozen forever instead of removed.  The fix
truncates the state after each step via ``_apply_filter``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
    SpectralSWConfig,
    SpectralShallowWaterModel,
    williamson_test2_spectral,
)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="spectral tests need JAX_ENABLE_X64=1",
)


def test_upper_third_state_mode_is_removed_not_frozen():
    grid = create_gaussian_grid(21)
    cfg = SpectralSWConfig(
        hyperdiff_coeff=0.0,          # isolate the truncation
        spectral_filter_order=0,      # no exponential filter
        dealiasing_fraction=0.667,
    )
    model = SpectralShallowWaterModel(grid, cfg)
    state = williamson_test2_spectral(grid)

    ls = np.asarray(grid.ls)
    n_cut = int(np.floor(cfg.dealiasing_fraction * grid.n_max))
    idx_hi = int(np.argmax(ls == grid.n_max))      # above the 2/3 band
    idx_lo = int(np.argmax(ls == 4))               # well-resolved mode

    amp = 1e-7
    vor = state.vor_hat.data.at[idx_hi].set(amp)
    state = state._replace(vor_hat=state.vor_hat.replace(data=vor))
    assert ls[idx_hi] > n_cut

    lo_before = complex(state.vor_hat.data[idx_lo])
    state1 = model.step(state, 300.0)

    # Fixed behavior: the upper-third state mode is exactly zeroed by the
    # post-step truncation (0/1 mask).  Pre-fix it stayed frozen at amp.
    assert float(jnp.abs(state1.vor_hat.data[idx_hi])) == 0.0

    # Resolved modes are untouched by the truncation (only dynamics moved
    # them): the mask is exactly 1 below the cutoff.
    lo_after = complex(state1.vor_hat.data[idx_lo])
    assert np.isfinite(lo_after.real) and np.isfinite(lo_after.imag)
    # W2 is a steady solution: the resolved mode must not be zeroed.
    assert abs(lo_after) > 0.1 * abs(lo_before) if abs(lo_before) > 0 else True


def test_truncation_noop_for_band_limited_state():
    """A state already inside the 2/3 band steps identically whether the
    state-truncation multiply runs or not (mask == 1 there)."""
    grid = create_gaussian_grid(21)
    cfg = SpectralSWConfig(
        hyperdiff_coeff=0.0, spectral_filter_order=0,
        dealiasing_fraction=0.667,
    )
    model = SpectralShallowWaterModel(grid, cfg)
    state = williamson_test2_spectral(grid)
    # W2 init is smooth/low-n; verify its upper-third power is roundoff.
    ls = np.asarray(grid.ls)
    n_cut = int(np.floor(cfg.dealiasing_fraction * grid.n_max))
    hi = ls > n_cut
    for f in (state.vor_hat.data, state.div_hat.data, state.phi_hat.data):
        assert float(jnp.max(jnp.abs(f[hi]))) < 1e-10 * max(
            1.0, float(jnp.max(jnp.abs(f)))
        )
    out = model.step(state, 300.0)
    # Truncated step of a band-limited state stays band-limited and finite.
    assert bool(jnp.all(jnp.isfinite(out.phi_hat.data)))
    assert float(jnp.max(jnp.abs(out.vor_hat.data[hi]))) == 0.0


def test_filter_every_step_false_matches_unfiltered_step():
    """With spectral_filter_every_step=False the post-step exponential
    filter must NOT run: one step equals the filter-order-0 step exactly
    (the 2/3-rule state truncation still applies in both).

    Regression for the colliding-modons erasure: the order-8/cutoff-0.65
    per-step filter compounds to e^-16 at n=10 over a 100-day T21 run.
    """
    grid = create_gaussian_grid(21)
    cfg_off = SpectralSWConfig(spectral_filter_order=0)
    cfg_init_only = SpectralSWConfig(
        spectral_filter_order=8, spectral_filter_every_step=False)
    cfg_every = SpectralSWConfig(spectral_filter_order=8)

    state = williamson_test2_spectral(grid)
    # Resolved mid-band mode (where the per-step filter bites) in BOTH
    # wind fields, plus an upper-third mode in both — the latter must be
    # removed by the 2/3-rule state truncation REGARDLESS of the
    # every-step flag (codex review: guard against the flag accidentally
    # disabling the truncation or skipping divergence).
    ls = np.asarray(grid.ls)
    idx_mid = int(np.argmax(ls == 10))
    idx_hi = int(np.argmax(ls == grid.n_max))
    vor = state.vor_hat.data.at[idx_mid].set(1e-6).at[idx_hi].set(1e-6)
    div = state.div_hat.data.at[idx_mid].set(1e-6).at[idx_hi].set(1e-6)
    state = state._replace(
        vor_hat=state.vor_hat.replace(data=vor),
        div_hat=state.div_hat.replace(data=div))

    out_off = SpectralShallowWaterModel(grid, cfg_off).step(state, 300.0)
    out_init = SpectralShallowWaterModel(grid, cfg_init_only).step(state, 300.0)
    out_every = SpectralShallowWaterModel(grid, cfg_every).step(state, 300.0)

    for fld in ("vor_hat", "div_hat", "phi_hat"):
        np.testing.assert_array_equal(
            np.asarray(getattr(out_init, fld).data),
            np.asarray(getattr(out_off, fld).data))
    # 2/3-rule state truncation still zeroes the upper-third mode in
    # every config, including every_step=False.
    for out in (out_off, out_init, out_every):
        assert float(jnp.abs(out.vor_hat.data[idx_hi])) == 0.0
        assert float(jnp.abs(out.div_hat.data[idx_hi])) == 0.0
    # ... while the every-step config really does damp the mid-band
    # mode in both wind fields and the init-only config does not.
    for fld in ("vor_hat", "div_hat"):
        assert float(jnp.abs(getattr(out_every, fld).data[idx_mid])) < float(
            jnp.abs(getattr(out_init, fld).data[idx_mid]))


def test_filter_initial_state_include_winds():
    """include_winds=True filters vor/div once; default leaves them exact."""
    grid = create_gaussian_grid(21)
    cfg = SpectralSWConfig(
        spectral_filter_order=8, spectral_filter_every_step=False)
    model = SpectralShallowWaterModel(grid, cfg)
    state = williamson_test2_spectral(grid)
    ls = np.asarray(grid.ls)
    n_cut = int(np.floor(cfg.dealiasing_fraction * grid.n_max))
    idx_band = int(np.argmax(ls == n_cut))     # top of the resolved band
    idx_hi = int(np.argmax(ls == grid.n_max))  # upper third (dealiased)
    vor = state.vor_hat.data.at[idx_band].set(1e-6).at[idx_hi].set(1e-6)
    state = state._replace(vor_hat=state.vor_hat.replace(data=vor))

    out_default = model.filter_initial_state(state)
    np.testing.assert_array_equal(
        np.asarray(out_default.vor_hat.data), np.asarray(state.vor_hat.data))

    out_winds = model.filter_initial_state(state, include_winds=True)
    # In-band mode: exponential filter only, exp(-alpha*(14/21)^8) ~ 0.983.
    ratio = float(jnp.abs(out_winds.vor_hat.data[idx_band])
                  / jnp.abs(state.vor_hat.data[idx_band]))
    assert 0.95 < ratio < 0.999
    # Upper-third mode: the 2/3-rule band limit zeroes it EXACTLY (iter-2
    # codex review — attenuated-only modes would contaminate the first
    # nonlinear step's stages).
    assert float(jnp.abs(out_winds.vor_hat.data[idx_hi])) == 0.0
    # phi filtered in both paths (unchanged phi-filter semantics).
    assert not np.array_equal(
        np.asarray(out_winds.phi_hat.data), np.asarray(state.phi_hat.data))

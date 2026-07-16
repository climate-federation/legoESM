"""WB scale trainer: the spectral (semi-implicit) training core (#817 blocker 1).

The `--training-core spectral` selector routes the WB training rollout through
the differentiable Gaussian/spectral PE core with the Hoskins–Simmons
semi-implicit step (#829's ``_make_spectral_integrator``) instead of the
explicit lat-lon C-grid — the design fix for the exploding training adjoint
(~x1.3/step on the explicit core) and the pole-cell dt collapse.

These tests validate the WIRING at a tiny truncation (T10/L8):
  * ``spectral_state_to_carry`` inverts ``carry_to_spectral_state`` (smooth
    fields round-trip; tracers exactly);
  * ``build_mode_components(training_core='spectral')`` returns an SI-enabled
    core whose ``.raw`` runs finite and is reverse-mode differentiable;
  * the YAML ``spectral:`` block drives the SpectralPEConfig (semi_implicit
    default True — the point of the core — and overridable).

The adjoint-damping DEMONSTRATION deliberately lives at the target fine grid
(the #829 T10 dt-sweep proved SI gives no benefit at coarse truncation where
the gravity-wave CFL never binds), so these tests assert wiring, not damping.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate


def _synthetic_carry(grid, nlev):
    """Smooth, balanced-ish SegmentCarry on the Gaussian grid (ERA5 stand-in)."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.driver.compiled_segments import pack_carry

    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    lat = np.asarray(grid.lat)[:, None]                      # (n_lat, 1)
    shape_3d = (n_lat, n_lon, nlev)
    shape_2d = (n_lat, n_lon)

    u = jnp.asarray(np.broadcast_to(
        20.0 * np.cos(lat)[..., None] * np.linspace(0.4, 1.0, nlev),
        shape_3d).copy())
    v = jnp.zeros(shape_3d)
    T = jnp.asarray(np.broadcast_to(
        250.0 + 40.0 * np.linspace(0.0, 1.0, nlev), shape_3d).copy())
    p_s = jnp.full(shape_2d, 1.0e5)
    phis = jnp.zeros(shape_2d)
    q_v = jnp.asarray(np.broadcast_to(
        5e-3 * np.linspace(0.1, 1.0, nlev), shape_3d).copy())

    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    state = HydrostaticState(
        u=Field(u, name="u", dims=d3, units="m/s"),
        v=Field(v, name="v", dims=d3, units="m/s"),
        T=Field(T, name="T", dims=d3, units="K"),
        p_s=Field(p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(phis, name="phis", dims=d2, units="m2/s2"),
    )
    return pack_carry(
        state, q_v=q_v, q_c=jnp.zeros(shape_3d), q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )


def _yml(n_max=10, nlev=8, **spectral_over):
    spec = {"n_max": n_max, "dt": 1800.0}
    spec.update(spectral_over)
    return {
        "n_lat": 32, "n_lon": 64, "nlev": nlev,
        "dt": 300.0,               # lat-lon core dt; spectral core ignores it
        "spectral": spec,
        "loss": {},
    }


def _cfg(mode="neural_gcm"):
    return SimpleNamespace(
        mode=mode, training_core="spectral", smoke=False,
        multi_step_hours=(6, 12),
    )


def test_spectral_state_to_carry_roundtrip():
    """carry -> spectral -> carry recovers the physical fields (SH round-trip
    on band-limited-ish smooth inputs) and passes tracers through exactly."""
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_state_to_carry,
    )

    nlev = 8
    grid = create_gaussian_grid(10)
    sigma = create_sigma_coordinate(nlev)
    carry = _synthetic_carry(grid, nlev)

    state = carry_to_spectral_state(carry, grid)
    back = spectral_state_to_carry(state, grid, sigma)

    # Smooth (large-scale) fields survive the SH analysis/synthesis round-trip
    # to a small relative tolerance (T10 truncation of already-smooth input).
    assert np.allclose(np.asarray(back.T), np.asarray(carry.T), atol=0.5)
    assert np.allclose(np.asarray(back.u), np.asarray(carry.u), atol=0.5)
    assert np.allclose(np.asarray(back.p_s), np.asarray(carry.p_s), rtol=1e-3)
    # Tracers ride the grid (no spectral truncation) -> exact.
    assert np.array_equal(np.asarray(back.q_v), np.asarray(carry.q_v))
    assert np.all(np.isfinite(np.asarray(back.v)))


def test_spectral_pe_config_defaults_and_overrides():
    """The YAML spectral block drives SpectralPEConfig; semi_implicit defaults
    True (the adjoint-bounding point of the core) and is overridable."""
    from legoesm.training.scale_build import _spectral_pe_config

    pe = _spectral_pe_config(_yml())
    assert pe.semi_implicit is True
    assert pe.hyperdiff_coeff == 2.5e15 and pe.spectral_filter_strength == 0.01

    pe_off = _spectral_pe_config(_yml(semi_implicit=False, si_substeps=2))
    assert pe_off.semi_implicit is False and pe_off.si_substeps == 2


@pytest.mark.parametrize("mode", ["neural_gcm", "physics"])
def test_spectral_core_raw_runs_finite_and_differentiable(mode):
    """build_mode_components(training_core='spectral'): the returned
    ``make_run_seg(params).raw(ic, n_steps, forcing)`` integrates the SI
    spectral core to a finite state AND reverse-mode differentiates to finite
    gradients — the two properties the exploding-adjoint blocker denies the
    explicit lat-lon core at scale."""
    import equinox as eqx

    from legoesm.training.losses import combined_loss
    from legoesm.training.scale_build import build_mode_components

    yml = _yml()
    cfg = _cfg(mode)
    model, grid, sigma, params, make_run_seg, loss_config, dt = (
        build_mode_components(cfg, yml))
    assert dt == 1800.0                       # spectral dt, no pole clamp
    assert int(grid.n_max) == 10

    nlev = int(yml["nlev"])
    ic = _synthetic_carry(grid, nlev)
    target = _synthetic_carry(grid, nlev)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    forcing = {
        "T_sfc": jnp.full((ncol,), 300.0),
        "sic": jnp.zeros((ncol,)),
        "day_of_year": jnp.asarray(80.0),
        "seconds_of_day": jnp.asarray(0.0),
    }
    n_steps = 3

    # Forward: finite rolled-out physical fields.
    pred = make_run_seg(params).raw(ic, n_steps, forcing)
    for name in ("u", "v", "T", "p_s", "q_v"):
        assert bool(jnp.all(jnp.isfinite(getattr(pred, name)))), name

    # Reverse mode: finite loss and finite, not-all-zero gradients.
    sigma_full = jnp.asarray(sigma.sigma_full)
    arr, static = eqx.partition(params, eqx.is_inexact_array)

    def loss_fn(a):
        trainable = eqx.combine(a, static)
        p = make_run_seg(trainable).raw(ic, n_steps, forcing)
        return combined_loss(p, target, sigma_full, grid=grid,
                             config=loss_config)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(arr)
    assert bool(jnp.isfinite(loss))
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in leaves)


def test_unknown_training_core_raises():
    from legoesm.training.scale_build import build_mode_components
    with pytest.raises(ValueError, match="unknown training_core"):
        build_mode_components(
            SimpleNamespace(mode="physics", training_core="bogus",
                            smoke=False, multi_step_hours=()),
            _yml())


def test_era5_forcing_calendar_convention():
    """codex #817 adversarial finding: spectral_rollout advances
    ``doy_eff = day_of_year + (t + seconds_of_day)/86400`` and wraps doy to
    [1, 366), so the sample forcing must carry the INTEGER 1-based day with the
    intra-day fraction in seconds_of_day alone.  The original 0-based
    fractional doy sent Jan 1 00Z to day 365 (a year off) and double-counted
    the hours."""
    from legoesm.training.scale_build import era5_time_to_forcing_calendar

    # Jan 1 00Z -> day 1, 0 s (NOT day 0 -> wrapped to 365).
    doy, sod = era5_time_to_forcing_calendar(np.datetime64("2010-01-01T00"), 2010)
    assert doy == 1.0 and sod == 0.0
    # Jan 1 06Z -> day 1 + 21600 s (fraction rides seconds, not doy).
    doy, sod = era5_time_to_forcing_calendar(np.datetime64("2010-01-01T06"), 2010)
    assert doy == 1.0 and sod == 21600.0
    # Feb 2 12Z -> day 33 + 43200 s.
    doy, sod = era5_time_to_forcing_calendar(np.datetime64("2010-02-02T12"), 2010)
    assert doy == 33.0 and sod == 43200.0
    # Consistency with the rollout's advance: doy + sod/86400 is the fractional
    # 1-based day the declination sees at step 0.
    assert doy + sod / 86400.0 == 33.5

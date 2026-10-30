"""``spectral_rollout(adjoint_steps=k)``: same forecast, gradient from the last k steps only."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.training.neural_gcm_spectral import (
    _scan_adjoint_window, carry_to_spectral_state, spectral_rollout,
    spectral_state_to_carry,
)
from tests.unit.test_spectral_carry_tracer_set import _carry
from tests.unit.test_spectral_phys_state_thread import _rad_zero, _stub_physics


def _toy(a, k, n=7):
    # x <- a*x + idx: the step index must continue across the cut.
    step = lambda x, i: (a * x + i, None)
    return _scan_adjoint_window(step, jnp.asarray(1.0), jnp.arange(n, dtype=float),
                                None, k)[0]


def _toy_ref(a, k, n=7):
    x = jnp.asarray(1.0)
    for i in range(n):
        if i == n - k:
            x = jax.lax.stop_gradient(x)
        x = a * x + i
    return x


@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_scan_window_forward_and_gradient(k):
    assert float(_toy(1.1, k)) == float(_toy(1.1, 0))
    g, g_ref = jax.grad(_toy)(1.1, k), jax.grad(_toy_ref)(1.1, k)
    np.testing.assert_allclose(float(g), float(g_ref), rtol=1e-14)


def test_scan_window_cut_changes_gradient():
    # Non-vacuity: a 1-step window is NOT the full gradient.
    assert abs(float(jax.grad(_toy)(1.1, 1)) - float(jax.grad(_toy)(1.1, 0))) > 1.0


def _setup():
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    state0 = carry_to_spectral_state(_carry(grid.n_lat, grid.n_lon, extras=False), grid)
    return grid, sigma, state0


def _roll(coef, k, grid, sigma, state0, n=6):  # out-of-range check only
    fn = _stub_physics(None, None, heat_per_unit_state=coef)
    final = spectral_rollout(
        state0, fn, grid, sigma, SpectralPEConfig(semi_implicit=True), 600.0, n,
        None, None, rad_physics_fn=_rad_zero, rad_update_interval=3,
        adjoint_steps=k)
    return jnp.mean(jnp.asarray(spectral_state_to_carry(final, grid, sigma).T) ** 2)


def _rad_live(state, grid_, sigma_coord, sim_time_seconds=0.0, phys_state=None,
              forcing=None):
    # Radiation that depends on the clock, the per-step forcing calendar AND
    # the physics carry, so a broken cadence, clock or hand-off shows.
    out = _rad_zero(state, grid_, sigma_coord)
    sod = 0.0 if forcing is None else forcing["seconds_of_day"]
    T_t = jnp.zeros_like(state.T_hat.data).at[0, :].set(
        1e-9 * (jnp.asarray(sim_time_seconds) + 600.0) + 1e-7 * phys_state
        + 1e-10 * jnp.asarray(sod))
    return out._replace(T_hat=out.T_hat.replace(data=T_t))


_CFG = SpectralPEConfig(semi_implicit=True)


_FORCING = {"day_of_year": jnp.asarray(10.0), "seconds_of_day": jnp.asarray(3600.0),
            "T_sfc": None, "sic": None}  # no surface override on the stub carry


def _roll_rad(fn, state0, grid, sigma, n, **kw):
    return spectral_rollout(state0, fn, grid, sigma, _CFG, 600.0, n, None, None,
                            rad_physics_fn=_rad_live, rad_update_interval=3,
                            forcing_base=_FORCING, **kw)


def _T2(state, grid, sigma):
    return jnp.mean(jnp.asarray(spectral_state_to_carry(state, grid, sigma).T) ** 2)


def _single(coef, k, grid, sigma, state0):
    fn = _stub_physics(None, None, heat_per_unit_state=coef)
    return _T2(_roll_rad(fn, state0, grid, sigma, 7, adjoint_steps=k), grid, sigma)


def _two_leg(coef_suffix, grid, sigma, state0, coef_prefix=1e-5, k=3):
    pre = _stub_physics(None, None, heat_per_unit_state=coef_prefix)
    suf = _stub_physics(None, None, heat_per_unit_state=coef_suffix)
    mid, ps, rad = _roll_rad(pre, state0, grid, sigma, 7 - k,
                             return_phys_state=True, return_rad_tendency=True)
    out = _roll_rad(suf, mid, grid, sigma, k, phys_state_in=ps,
                    rad_tendency_in=rad, step_offset=7 - k)
    return _T2(out, grid, sigma)


def test_two_leg_chain_equals_single_rollout_with_live_radiation():
    grid, sigma, state0 = _setup()
    one = float(_single(1e-5, 0, grid, sigma, state0))
    assert float(_two_leg(1e-5, grid, sigma, state0)) == one
    # Non-vacuity: restarting the clock/cadence in leg 2 changes the forecast.
    pre = _stub_physics(None, None, heat_per_unit_state=1e-5)
    mid, ps = _roll_rad(pre, state0, grid, sigma, 4, return_phys_state=True)
    restarted = _T2(_roll_rad(pre, mid, grid, sigma, 3, phys_state_in=ps),
                    grid, sigma)
    assert float(restarted) != one
    # ...and so does a continued leg that keeps the cache but restarts the clock.
    mid, ps, rad = _roll_rad(pre, state0, grid, sigma, 4,
                             return_phys_state=True, return_rad_tendency=True)
    no_clock = spectral_rollout(
        mid, pre, grid, sigma, _CFG, 600.0, 3, None, None,
        rad_physics_fn=_rad_live, rad_update_interval=3,
        forcing_base=_FORCING, phys_state_in=ps, rad_tendency_in=rad,
        step_offset=1)
    assert float(_T2(no_clock, grid, sigma)) != one


def test_window_forecast_unchanged_and_gradient_is_the_suffix_derivative():
    grid, sigma, state0 = _setup()
    assert float(_single(1e-5, 3, grid, sigma, state0)) == float(
        _single(1e-5, 0, grid, sigma, state0))
    g3 = float(jax.grad(_single)(1e-5, 3, grid, sigma, state0))
    g0 = float(jax.grad(_single)(1e-5, 0, grid, sigma, state0))
    np.testing.assert_allclose(
        g3, float(jax.grad(_two_leg)(1e-5, grid, sigma, state0)), rtol=1e-10)
    np.testing.assert_allclose(
        float(jax.grad(_single)(1e-5, 7, grid, sigma, state0)), g0, rtol=1e-12)
    assert abs(g3 - g0) > 1e-3 * abs(g0)


def test_markerless_gated_branch_window():
    grid, sigma, state0 = _setup()

    def f(c, k):
        fn = _stub_physics(None, None)
        del fn.with_phys_state, fn.init_phys_state
        rad = lambda s, g_, sc, sim_time_seconds=0.0: _rad_live(
            s, g_, sc, sim_time_seconds * c, 0.0)
        out = spectral_rollout(state0, fn, grid, sigma, _CFG, 600.0, 6, None,
                               None, rad_physics_fn=rad, rad_update_interval=3,
                               adjoint_steps=k)
        return _T2(out, grid, sigma)

    assert float(f(1.0, 2)) == float(f(1.0, 0))
    assert abs(float(jax.grad(f)(1.0, 2)) - float(jax.grad(f)(1.0, 0))) > 0.0


def test_chaining_kwargs_refused_where_they_cannot_be_exact():
    grid, sigma, state0 = _setup()
    fn = _stub_physics(None, None)
    with pytest.raises(ValueError, match="rad_tendency_in"):    # no cache
        _roll_rad(fn, state0, grid, sigma, 2, step_offset=2,
                  phys_state_in=jnp.asarray(0.0))
    with pytest.raises(ValueError, match="STATEFUL"):          # no radiation
        spectral_rollout(state0, fn, grid, sigma, _CFG, 600.0, 2, None, None,
                         return_phys_state=True, return_rad_tendency=True)
    anchored = _CFG._replace(fix_mass=True, anchor_mass_to_initial=True)
    mid, ps, rad = spectral_rollout(      # leg 1 alone anchors to the true start
        state0, fn, grid, sigma, anchored, 600.0, 2, None, None,
        rad_physics_fn=_rad_live, return_phys_state=True,
        return_rad_tendency=True)
    with pytest.raises(ValueError, match="anchor"):
        spectral_rollout(mid, fn, grid, sigma, anchored, 600.0, 2, None, None,
                         rad_physics_fn=_rad_live, phys_state_in=ps,
                         rad_tendency_in=rad, step_offset=2)


def test_adjoint_steps_out_of_range_raises():
    grid, sigma, state0 = _setup()
    with pytest.raises(ValueError, match="adjoint_steps"):
        _roll(1e-5, 7, grid, sigma, state0)

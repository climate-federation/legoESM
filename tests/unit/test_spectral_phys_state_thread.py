"""The spectral training rollout carries prognostic PHYSICS state.

The combined spectral physics wrapper has always returned the updated
prognostic physics memory (CLUBB's wp2/TKE, Bechtold's convective
organization, the GWD spectrum) and the training lane DISCARDED it — every
stateful scheme ran memoryless, which left the WeatherBench classical arm
with its turbulence energy at the floor forever (effectively no
boundary-layer mixing; 2026-08-17 scene-17 dissection, the same
absent-component class as the nine-species fix). ``spectral_rollout`` now
threads a PhysicsState through its scan carry when the physics callable
carries the ``with_phys_state`` / ``init_phys_state`` markers; markerless
callables (the learned arms) keep the legacy stateless body verbatim.
"""
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
    carry_to_spectral_state, spectral_rollout, spectral_state_to_carry,
)

# Reuse the minimal-carry builder from the tracer-set tests.
from tests.unit.test_spectral_carry_tracer_set import _carry


def _stub_physics(grid, sigma, *, heat_per_unit_state=1.0e-6):
    """A marker-carrying stub: its 'physics state' is a scalar counter that
    grows by 1 each step, and its T tendency is proportional to the COUNTER
    — so the final temperature encodes how much memory was accumulated.
    Memoryless evaluation (counter always 0) would heat by 0; threaded
    memory heats by sum(k)·dt·coef. The stub returns zero spectral
    tendencies except T_hat's mean mode.
    """
    def zero_tend_like(state, dT_mean):
        z3 = jnp.zeros_like(state.vor_hat.data)
        T_t = jnp.zeros_like(state.T_hat.data)
        # Mean (l=0,m=0) spectral mode is index 0 by construction of the
        # packed SH layout; heating the mean mode is enough to observe.
        T_t = T_t.at[0, :].set(dT_mean)
        return state._replace(
            vor_hat=state.vor_hat.replace(data=z3),
            div_hat=state.div_hat.replace(data=z3),
            T_hat=state.T_hat.replace(data=T_t),
            lnps_hat=state.lnps_hat.replace(
                data=jnp.zeros_like(state.lnps_hat.data)),
            phis_hat=state.phis_hat.replace(
                data=jnp.zeros_like(state.phis_hat.data)),
            tracers=None if state.tracers is None else {
                k: (v.replace(data=jnp.zeros_like(v.data))
                    if hasattr(v, "data") else jnp.zeros_like(v))
                for k, v in state.tracers.items()},
        )

    def fn(state, grid_, sigma_coord):
        return zero_tend_like(state, 0.0)

    def fn_with_state(state, grid_, sigma_coord, phys_state, forcing=None):
        counter = phys_state
        return (zero_tend_like(state, heat_per_unit_state * counter),
                counter + 1.0)

    fn.with_phys_state = fn_with_state
    fn.init_phys_state = lambda ncol, nlev, dtype=None: jnp.asarray(0.0)
    return fn


def _rad_zero(state, grid_, sigma_coord, **kw):
    z3 = jnp.zeros_like(state.vor_hat.data)
    return state._replace(
        vor_hat=state.vor_hat.replace(data=z3),
        div_hat=state.div_hat.replace(data=z3),
        T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
        lnps_hat=state.lnps_hat.replace(
            data=jnp.zeros_like(state.lnps_hat.data)),
        phis_hat=state.phis_hat.replace(
            data=jnp.zeros_like(state.phis_hat.data)),
        tracers=None if state.tracers is None else {
            k: (v.replace(data=jnp.zeros_like(v.data))
                if hasattr(v, "data") else jnp.zeros_like(v))
            for k, v in state.tracers.items()},
    )


def _setup(n_steps, physics_fn):
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    carry = _carry(grid.n_lat, grid.n_lon, extras=False)
    state0 = carry_to_spectral_state(carry, grid)
    final = spectral_rollout(
        state0, physics_fn, grid, sigma,
        SpectralPEConfig(semi_implicit=True), 600.0, n_steps,
        None, None,
        rad_physics_fn=_rad_zero, rad_update_interval=3,
    )
    return spectral_state_to_carry(final, grid, sigma), grid, sigma


def test_marked_physics_state_accumulates_across_steps():
    """With the markers, the counter memory grows 0,1,2,... and the
    T tendency (∝ counter) heats accordingly — a memoryless lane (counter
    stuck at its init, today's discarded-state behavior) would not heat at
    all. Non-vacuity: the markerless run of the SAME stub stays cold."""
    fn = _stub_physics(None, None, heat_per_unit_state=1.0e-5)
    out_stateful, grid, _ = _setup(6, fn)

    bare = _stub_physics(None, None, heat_per_unit_state=1.0e-5)
    del bare.with_phys_state, bare.init_phys_state    # markerless twin
    out_stateless, _, _ = _setup(6, bare)

    dT = float(jnp.mean(jnp.asarray(out_stateful.T))
               - jnp.mean(jnp.asarray(out_stateless.T)))
    # counters 0..5 over 6 steps at dt=600: sum(k)*dt*coef = 15*600*1e-5
    # = 0.09 K on the SH mean COEFFICIENT; the grid-mean carries the
    # orthonormal-harmonic normalization 1/sqrt(4*pi) (measured 0.2821x,
    # matching to 0.1%).
    expected = 0.09 / np.sqrt(4.0 * np.pi)
    assert dT == pytest.approx(expected, rel=0.05)


def test_markerless_physics_fn_keeps_legacy_path():
    """No markers -> the legacy 2-tuple scan carry runs and the rollout is
    unchanged (finite, cold for the zero-tendency stub)."""
    bare = _stub_physics(None, None)
    del bare.with_phys_state, bare.init_phys_state
    out, _, _ = _setup(4, bare)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(out.T))))


def test_gradient_flows_through_the_threaded_carry():
    """The physics memory rides the scan carry inside jax.checkpoint; the
    adjoint through it must be finite and NONZERO (a zero gradient would
    mean the thread is disconnected)."""
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    carry = _carry(grid.n_lat, grid.n_lon, extras=False)
    state0 = carry_to_spectral_state(carry, grid)

    def loss(coef):
        fn = _stub_physics(None, None, heat_per_unit_state=coef)
        final = spectral_rollout(
            state0, fn, grid, sigma,
            SpectralPEConfig(semi_implicit=True), 600.0, 5,
            None, None,
            rad_physics_fn=_rad_zero, rad_update_interval=2,
        )
        back = spectral_state_to_carry(final, grid, sigma)
        return jnp.mean(jnp.asarray(back.T) ** 2)

    g = jax.grad(loss)(1.0e-5)
    assert np.isfinite(float(g)) and abs(float(g)) > 0.0


def test_segmented_rollout_equals_single_when_state_is_chained():
    """Codex P0: a chained multi-segment loss must be able to carry the
    physics memory across segments. rollout(6) == rollout(3) -> carry ps
    -> rollout(3, phys_state_in=ps), EXACTLY (same math, same order),
    while the unchained pair (state reset at the seam) differs —
    non-vacuity of the seam."""
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    carry = _carry(grid.n_lat, grid.n_lon, extras=False)
    state0 = carry_to_spectral_state(carry, grid)
    fn = _stub_physics(None, None, heat_per_unit_state=1.0e-5)
    kw = dict(rad_physics_fn=_rad_zero, rad_update_interval=3)
    cfgpe = SpectralPEConfig(semi_implicit=True)

    single = spectral_rollout(
        state0, fn, grid, sigma, cfgpe, 600.0, 6, None, None, **kw)

    mid, ps_mid = spectral_rollout(
        state0, fn, grid, sigma, cfgpe, 600.0, 3, None, None,
        return_phys_state=True, **kw)
    # NOTE the rad cache is legitimately re-initialized at the seam (the
    # radiation interval divides 3 here so the refresh lands on the seam
    # anyway) — the physics-memory chain is what this test pins.
    chained = spectral_rollout(
        mid, fn, grid, sigma, cfgpe, 600.0, 3, None, None,
        phys_state_in=ps_mid,
        sim_time_offset_seconds=3 * 600.0, **kw)
    np.testing.assert_allclose(
        np.asarray(chained.T_hat.data), np.asarray(single.T_hat.data),
        rtol=1e-12)

    reset = spectral_rollout(
        mid, fn, grid, sigma, cfgpe, 600.0, 3, None, None,
        sim_time_offset_seconds=3 * 600.0, **kw)
    assert not np.allclose(
        np.asarray(reset.T_hat.data), np.asarray(single.T_hat.data),
        rtol=1e-12)


def test_state_kwargs_on_markerless_fn_raise():
    """Silently ignoring a supplied physics state would reintroduce the
    memoryless pathology — refuse loudly."""
    grid = create_gaussian_grid(5)
    sigma = create_sigma_coordinate(4)
    carry = _carry(grid.n_lat, grid.n_lon, extras=False)
    state0 = carry_to_spectral_state(carry, grid)
    bare = _stub_physics(None, None)
    del bare.with_phys_state, bare.init_phys_state
    with pytest.raises(ValueError, match="markers"):
        spectral_rollout(
            state0, bare, grid, sigma, SpectralPEConfig(semi_implicit=True),
            600.0, 2, None, None,
            rad_physics_fn=_rad_zero, rad_update_interval=2,
            return_phys_state=True)

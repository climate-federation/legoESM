"""CFL diagnostic + adaptive-dt helper tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.shared.cfl_diagnostic import (
    acoustic_courant_horizontal,
    assert_courant_below,
    column_sound_speed_upper_bound,
    compute_courant_numbers_plane,
    suggest_stable_dt,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

from legoesm import constants

jax.config.update("jax_enable_x64", True)


def _setup(dx=2_000.0, dz_levels=10, H=10_000.0):
    grid = create_plane_grid(
        nx=4, ny=4, nlev=dz_levels, dx=dx, dy=dx, dtype=jnp.float64,
    )
    hc = create_height_coordinate(dz_levels, H=H)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    return state, hc, grid


def test_rest_state_courant_is_acoustic_only():
    """At rest, u = v = w = 0 → horizontal + vertical Courant = 0,
    only acoustic is non-zero."""
    state, hc, grid = _setup()
    courant = compute_courant_numbers_plane(state, hc, grid, dt=1.0)
    assert float(courant.horizontal_advective) == 0.0
    assert float(courant.vertical_advective) == 0.0
    assert float(courant.acoustic) > 0.0


def test_horizontal_courant_matches_analytic_u_dt_over_dx():
    """Inject constant u; expect courant_h = u·dt/dx."""
    state, hc, grid = _setup(dx=1_000.0)
    u_val = 50.0
    dt = 5.0
    state = state._replace(
        u=state.u.replace(data=jnp.full(state.u.data.shape, u_val)),
    )
    courant = compute_courant_numbers_plane(state, hc, grid, dt=dt)
    expected_c_h = u_val * dt / grid.dx
    np.testing.assert_allclose(
        float(courant.horizontal_advective), expected_c_h, rtol=1.0e-12,
    )


def test_vertical_courant_matches_analytic_w_dt_over_dz():
    state, hc, grid = _setup(dx=1_000.0, dz_levels=8, H=8_000.0)
    w_val = 10.0
    dt = 5.0
    state = state._replace(
        w=state.w.replace(data=jnp.full(state.w.data.shape, w_val)),
    )
    courant = compute_courant_numbers_plane(state, hc, grid, dt=dt)
    dz_min = float(jnp.min(hc.dz))
    expected_c_v = w_val * dt / dz_min
    np.testing.assert_allclose(
        float(courant.vertical_advective), expected_c_v, rtol=1.0e-12,
    )


def test_acoustic_courant_matches_sound_speed_formula():
    state, hc, grid = _setup(dx=1_000.0, dz_levels=8, H=8_000.0)
    dt = 1.0
    courant = compute_courant_numbers_plane(state, hc, grid, dt=dt)
    gamma = constants.c_pd / constants.c_vd
    # Codex iter-1: sound speed uses (θ_ref + |θ'|) · exner_ref_max.
    # At rest θ' = 0 so total reduces to ref.
    theta_total_max = float(jnp.max(
        hc.theta_ref + jnp.abs(state.theta_prime.data),
    ))
    T_max = theta_total_max * float(jnp.max(hc.exner_ref))
    c_sound = (gamma * constants.R_d * T_max) ** 0.5
    dz_min = float(jnp.min(hc.dz))
    delta_min = min(grid.dx, grid.dy, dz_min)
    expected_c_a = c_sound * dt / delta_min
    np.testing.assert_allclose(
        float(courant.acoustic), expected_c_a, rtol=1.0e-12,
    )


def test_acoustic_courant_scaled_by_n_acoustic_substeps():
    """Codex iter-1: split-explicit substepping divides dt by
    n_acoustic_substeps before checking acoustic CFL."""
    state, hc, grid = _setup(dx=1_000.0)
    dt = 12.0
    n_sub = 6
    c_no_sub = compute_courant_numbers_plane(
        state, hc, grid, dt=dt, n_acoustic_substeps=1,
    )
    c_with_sub = compute_courant_numbers_plane(
        state, hc, grid, dt=dt, n_acoustic_substeps=n_sub,
    )
    np.testing.assert_allclose(
        float(c_with_sub.acoustic),
        float(c_no_sub.acoustic) / n_sub,
        rtol=1.0e-12,
    )


def test_acoustic_courant_picks_up_theta_perturbation():
    """Codex iter-1: warm θ' should raise c_sound. Inject θ'=30K and
    confirm acoustic Courant grows vs rest state."""
    state, hc, grid = _setup(dx=1_000.0)
    c_rest = compute_courant_numbers_plane(state, hc, grid, dt=1.0)
    state_warm = state._replace(
        theta_prime=state.theta_prime.replace(
            data=jnp.full(state.theta_prime.data.shape, 30.0),
        ),
    )
    c_warm = compute_courant_numbers_plane(state_warm, hc, grid, dt=1.0)
    assert float(c_warm.acoustic) > float(c_rest.acoustic), (
        f"Warm θ' did not raise acoustic Courant: rest={float(c_rest.acoustic):.3e}, "
        f"warm={float(c_warm.acoustic):.3e}"
    )


def test_assert_courant_below_raises_on_breach():
    """Inject u that gives c_h > 1."""
    state, hc, grid = _setup(dx=100.0)
    u_val = 500.0  # 5 m per cell per second
    state = state._replace(
        u=state.u.replace(data=jnp.full(state.u.data.shape, u_val)),
    )
    dt = 1.0  # → c_h = 5.0
    courant = compute_courant_numbers_plane(state, hc, grid, dt=dt)
    with pytest.raises(ValueError, match="Horizontal advective Courant"):
        assert_courant_below(courant, horizontal_max=1.0)


def test_assert_courant_below_passes_under_bounds():
    state, hc, grid = _setup()
    courant = compute_courant_numbers_plane(state, hc, grid, dt=0.5)
    # Rest state; acoustic ~ 340·0.5/2000 = 0.085. Pass.
    assert_courant_below(
        courant, horizontal_max=1.0, vertical_max=1.0, acoustic_max=1.0,
    )


def test_suggest_stable_dt_zero_winds_returns_acoustic_limit():
    """Rest state: dt suggestion bounded by acoustic CFL only."""
    state, hc, grid = _setup(dx=1_000.0)
    dt_suggested = suggest_stable_dt(
        state, hc, grid, courant_target=0.8,
    )
    gamma = constants.c_pd / constants.c_vd
    theta_total_max = float(jnp.max(
        hc.theta_ref + jnp.abs(state.theta_prime.data),
    ))
    T_max = theta_total_max * float(jnp.max(hc.exner_ref))
    c_sound = (gamma * constants.R_d * T_max) ** 0.5
    dz_min = float(jnp.min(hc.dz))
    delta_min = min(grid.dx, grid.dy, dz_min)
    expected = 0.8 * delta_min / c_sound
    np.testing.assert_allclose(dt_suggested, expected, rtol=1.0e-10)


def test_suggest_stable_dt_scales_with_n_acoustic_substeps():
    """Codex iter-1: dt suggestion scales LINEARLY with substep count
    (acoustic substepping lifts the outer-dt bound)."""
    state, hc, grid = _setup(dx=1_000.0)
    dt1 = suggest_stable_dt(state, hc, grid, n_acoustic_substeps=1)
    dt6 = suggest_stable_dt(state, hc, grid, n_acoustic_substeps=6)
    np.testing.assert_allclose(dt6, dt1 * 6.0, rtol=1.0e-10)


def test_suggest_stable_dt_with_strong_horizontal_wind():
    """When |u| > c_s, horizontal advective CFL is the bottleneck."""
    state, hc, grid = _setup(dx=1_000.0)
    u_val = 500.0  # very fast
    state = state._replace(
        u=state.u.replace(data=jnp.full(state.u.data.shape, u_val)),
    )
    dt_suggested = suggest_stable_dt(
        state, hc, grid, courant_target=0.5,
    )
    # Horizontal advective: dt = 0.5 · dx / |u| = 0.5 · 1000 / 500 = 1.
    assert dt_suggested < 1.0 + 1.0e-12


def test_compute_courant_rejects_zero_n_substeps():
    """Codex iter-2: validate n_acoustic_substeps >= 1."""
    state, hc, grid = _setup()
    with pytest.raises(ValueError, match="n_acoustic_substeps"):
        compute_courant_numbers_plane(
            state, hc, grid, dt=1.0, n_acoustic_substeps=0,
        )
    with pytest.raises(ValueError, match="n_acoustic_substeps"):
        suggest_stable_dt(state, hc, grid, n_acoustic_substeps=-3)


def test_compute_courant_is_jit_compilable():
    """Pure JAX → must JIT-compile cleanly."""
    state, hc, grid = _setup()
    fn = jax.jit(
        lambda s: compute_courant_numbers_plane(s, hc, grid, dt=1.0),
    )
    out = fn(state)
    assert bool(jnp.all(jnp.isfinite(out.horizontal_advective)))
    assert bool(jnp.all(jnp.isfinite(out.vertical_advective)))
    assert bool(jnp.all(jnp.isfinite(out.acoustic)))


def test_column_sound_speed_upper_bound_matches_formula():
    """c_s = sqrt(gamma*R_d*T_max), T_max = max(theta_ref+|theta'|)*max(exner_ref);
    theta_prime=None uses the REST bound (theta_ref only) (iter 103)."""
    _, hc, _ = _setup(dx=1_000.0)
    gamma = constants.c_pd / constants.c_vd
    # rest bound (theta_prime None)
    t_rest = float(jnp.max(hc.theta_ref)) * float(jnp.max(hc.exner_ref))
    np.testing.assert_allclose(
        float(column_sound_speed_upper_bound(hc)),
        (gamma * constants.R_d * t_rest) ** 0.5, rtol=1e-12)
    # a warm perturbation (+5 K) raises c_s above the rest bound.
    theta_prime = jnp.full(hc.theta_ref.shape, 5.0)
    assert float(column_sound_speed_upper_bound(hc, theta_prime)) > \
        float(column_sound_speed_upper_bound(hc))


def test_acoustic_courant_horizontal_uses_dx_not_dz():
    """The horizontal acoustic Courant uses min(dx,dy) — NOT the small dz that
    compute_courant_numbers_plane's `acoustic` uses — so a semi-implicit-vertical
    dycore is not falsely over-constrained by a thin surface layer (iter 103)."""
    state, hc, grid = _setup(dx=2_000.0, dz_levels=10, H=10_000.0)
    dz_min = float(jnp.min(hc.dz))
    assert dz_min < float(grid.dx)                    # vertical IS the smaller spacing
    c_s = float(column_sound_speed_upper_bound(hc))
    dt, n_sub = 4.0, 6
    c_horiz = float(acoustic_courant_horizontal(
        hc, grid, dt, n_acoustic_substeps=n_sub))
    np.testing.assert_allclose(
        c_horiz, c_s * (dt / n_sub) / float(grid.dx), rtol=1e-12)
    # the full (delta_min) acoustic is LARGER (it uses the smaller dz).
    full = compute_courant_numbers_plane(
        state, hc, grid, dt=dt, n_acoustic_substeps=n_sub)
    assert float(full.acoustic) > c_horiz


def test_acoustic_courant_horizontal_substep_scaling_and_guard():
    _, hc, grid = _setup(dx=1_000.0)
    c1 = float(acoustic_courant_horizontal(hc, grid, dt=6.0, n_acoustic_substeps=1))
    c6 = float(acoustic_courant_horizontal(hc, grid, dt=6.0, n_acoustic_substeps=6))
    np.testing.assert_allclose(c6, c1 / 6.0, rtol=1e-12)
    with pytest.raises(ValueError, match=">= 1"):
        acoustic_courant_horizontal(hc, grid, dt=1.0, n_acoustic_substeps=0)

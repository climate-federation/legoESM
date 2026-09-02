"""Red-capable tests for the NEMO-literal final ZDF applications."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.experiments.dino import DINOConfig, DINO_RECIPES
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_nemo_momentum,
    implicit_vertical_diffusion_nemo_tracer_pair,
    implicit_vertical_diffusion_ocean,
    implicit_vertical_diffusion_ocean_momentum_dispatch,
    implicit_vertical_diffusion_ocean_pair,
    implicit_vertical_diffusion_ocean_tracer_pair_dispatch,
)
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_full_step_coordinate,
    create_ocean_z_star,
)


def _hand_solve(lower, diagonal, upper, rhs):
    diagonal = np.array(diagonal, copy=True)
    work = np.array(rhs, copy=True)
    for k in range(1, rhs.shape[-1]):
        diagonal[..., k] = (diagonal[..., k]
                            - lower[..., k] * upper[..., k - 1]
                            / diagonal[..., k - 1])
    for k in range(1, rhs.shape[-1]):
        work[..., k] = (work[..., k]
                        - lower[..., k] / diagonal[..., k - 1]
                        * work[..., k - 1])
    work[..., -1] = work[..., -1] / diagonal[..., -1]
    for k in range(rhs.shape[-1] - 2, -1, -1):
        work[..., k] = ((work[..., k] - upper[..., k] * work[..., k + 1])
                        / diagonal[..., k])
    return work


def test_literal_momentum_matches_hand_written_unequal_depth_case():
    rhs = np.array([[1.25, -0.5, 2.0, 0.75]], dtype=np.float64)
    avm = np.array([[0.07, 0.11, 0.23]], dtype=np.float64)
    dz = np.array([[1.5, 2.75, 4.0, 7.25]], dtype=np.float64)
    dzh = np.array([[1.8, 3.2, 5.5]], dtype=np.float64)
    wet = np.ones_like(rhs, dtype=bool)
    extra = np.array([[0.0, 0.0, 0.0, 0.031]], dtype=np.float64)
    dt = 13.0
    lower_coeff = -(0.5 * dt) * (2.0 * avm) / (dz[..., 1:] * dzh)
    upper_coeff = -(0.5 * dt) * (2.0 * avm) / (dz[..., :-1] * dzh)
    lower = np.concatenate([np.zeros_like(rhs[..., :1]), lower_coeff], axis=-1)
    upper = np.concatenate([upper_coeff, np.zeros_like(rhs[..., :1])], axis=-1)
    diagonal = 1.0 - lower - upper + extra
    expected = _hand_solve(lower, diagonal, upper, rhs)
    actual = np.asarray(implicit_vertical_diffusion_nemo_momentum(
        jnp.asarray(rhs), jnp.asarray(avm), jnp.asarray(dz), jnp.asarray(dzh),
        dt, jnp.asarray(wet), extra_diag=jnp.asarray(extra)))
    # XLA may fuse the scalar-looking coefficient expression; the frozen
    # oracle bar is 1e-15 and this independent host loop agrees within 1 ULP.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-16)

    # Red control: the historical normalised matrix is numerically distinct.
    legacy = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(rhs), jnp.asarray(avm), jnp.asarray(dz), jnp.asarray(dzh),
        dt, extra_diag=jnp.asarray(extra)))
    assert np.any(legacy != expected)


def test_literal_tracer_pair_matches_hand_content_system():
    t_content = np.array([[4.0, 9.0, -1.5]], dtype=np.float64)
    s_content = np.array([[8.0, -3.0, 2.5]], dtype=np.float64)
    k = np.array([[0.13, 0.29]], dtype=np.float64)
    e3t = np.array([[1.25, 3.5, 8.0]], dtype=np.float64)
    e3w = np.array([[2.0, 5.0]], dtype=np.float64)
    wet = np.ones_like(t_content, dtype=bool)
    dt = 7.0
    coeff = -dt * k / e3w
    lower = np.concatenate([np.zeros_like(t_content[..., :1]), coeff], -1)
    upper = np.concatenate([coeff, np.zeros_like(t_content[..., :1])], -1)
    diagonal = e3t - lower - upper
    expected_t = _hand_solve(lower, diagonal, upper, t_content)
    expected_s = _hand_solve(lower, diagonal, upper, s_content)
    actual_t, actual_s = implicit_vertical_diffusion_nemo_tracer_pair(
        jnp.asarray(t_content), jnp.asarray(s_content), jnp.asarray(k),
        jnp.asarray(e3t), jnp.asarray(e3w), dt, jnp.asarray(wet))
    np.testing.assert_array_equal(np.asarray(actual_t), expected_t)
    np.testing.assert_array_equal(np.asarray(actual_s), expected_s)

    # Wrong-e3t control must be capable of failing.
    wrong_t, _ = implicit_vertical_diffusion_nemo_tracer_pair(
        jnp.asarray(t_content), jnp.asarray(s_content), jnp.asarray(k),
        jnp.asarray(e3t + np.array([[0.0, 0.25, 0.0]])),
        jnp.asarray(e3w), dt, jnp.asarray(wet))
    assert np.any(np.asarray(wrong_t) != expected_t)


def test_literal_solvers_jit_and_ad():
    rhs = jnp.array([[1.0, 2.0, 4.0]], dtype=jnp.float64)
    k = jnp.array([[0.1, 0.2]], dtype=jnp.float64)
    dz = jnp.array([[2.0, 3.0, 5.0]], dtype=jnp.float64)
    dzh = jnp.array([[2.5, 4.0]], dtype=jnp.float64)
    wet = jnp.ones_like(rhs, dtype=bool)
    eager = implicit_vertical_diffusion_nemo_momentum(
        rhs, k, dz, dzh, 2.0, wet)
    compiled = jax.jit(implicit_vertical_diffusion_nemo_momentum,
                       static_argnums=(4,))(rhs, k, dz, dzh, 2.0, wet)
    np.testing.assert_array_equal(np.asarray(compiled), np.asarray(eager))
    grad = jax.grad(lambda x: jnp.sum(
        implicit_vertical_diffusion_nemo_momentum(
            x, k, dz, dzh, 2.0, wet)))(rhs)
    assert np.isfinite(np.asarray(grad)).all()

    pair_jit = jax.jit(implicit_vertical_diffusion_nemo_tracer_pair,
                       static_argnums=(5,))
    t, s = pair_jit(rhs * dz, 2.0 * rhs * dz, k, dz, dzh, 2.0, wet)
    assert np.isfinite(np.asarray(t)).all()
    assert np.isfinite(np.asarray(s)).all()


def test_selector_scope_defaults_only_on_two_nemo_cards():
    selected = {
        name for name, values in DINO_RECIPES.items()
        if values.get("zdf_implicit_solver_evaluation", "shared_thomas")
        == "nemo_literal"
    }
    assert selected == {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    assert DINOConfig().zdf_implicit_solver_evaluation == "shared_thomas"
    for name, values in DINO_RECIPES.items():
        resolved = DINOConfig(**values)
        expected = "nemo_literal" if name in selected else "shared_thomas"
        assert resolved.zdf_implicit_solver_evaluation == expected


def test_literal_shape_guards_are_loud():
    field = jnp.ones((1, 3), dtype=jnp.float64)
    with pytest.raises(ValueError, match="avm_face"):
        implicit_vertical_diffusion_nemo_momentum(
            field, jnp.ones((1, 3)), field, jnp.ones((1, 2)), 1.0,
            jnp.ones_like(field, dtype=bool))


def test_dispatch_legacy_is_exact_and_selector_typos_are_loud():
    field = jnp.array([[1.0, 2.0, 4.0]], dtype=jnp.float64)
    second = 2.0 * field
    k = jnp.array([[0.1, 0.2]], dtype=jnp.float64)
    dz = jnp.array([[2.0, 3.0, 5.0]], dtype=jnp.float64)
    dzh = jnp.array([[2.5, 4.0]], dtype=jnp.float64)
    wet = jnp.ones_like(field, dtype=bool)
    direct = implicit_vertical_diffusion_ocean(field, k, dz, dzh, 2.0)
    dispatched = implicit_vertical_diffusion_ocean_momentum_dispatch(
        field, k, dz, dzh, 2.0, wet, evaluation="shared_thomas")
    np.testing.assert_array_equal(np.asarray(dispatched), np.asarray(direct))
    direct_pair = implicit_vertical_diffusion_ocean_pair(
        field, second, k, dz, dzh, 2.0)
    dispatch_pair = implicit_vertical_diffusion_ocean_tracer_pair_dispatch(
        field, second, field * dz, second * dz, k, dz, dzh, 2.0, wet,
        evaluation="shared_thomas")
    for actual, expected in zip(dispatch_pair, direct_pair):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    with pytest.raises(ValueError, match="unknown ZDF momentum"):
        implicit_vertical_diffusion_ocean_momentum_dispatch(
            field, k, dz, dzh, 2.0, wet, evaluation="nemo_litreal")
    with pytest.raises(ValueError, match="unknown ZDF tracer"):
        implicit_vertical_diffusion_ocean_tracer_pair_dispatch(
            field, second, field * dz, second * dz, k, dz, dzh, 2.0, wet,
            evaluation="nemo_litreal")


def test_literal_dry_rows_are_finite_and_masked():
    field = jnp.array([[1.0, 2.0, 9.0]], dtype=jnp.float64)
    wet = jnp.array([[True, True, False]])
    dz = jnp.array([[2.0, 3.0, 0.0]], dtype=jnp.float64)
    dzh = jnp.array([[2.5, 0.0]], dtype=jnp.float64)
    k = jnp.array([[0.1, 0.0]], dtype=jnp.float64)
    momentum = implicit_vertical_diffusion_nemo_momentum(
        field, k, dz, dzh, 2.0, wet)
    tracer, _ = implicit_vertical_diffusion_nemo_tracer_pair(
        field * dz, field * dz, k, dz, dzh, 2.0, wet)
    assert np.isfinite(np.asarray(momentum)).all()
    assert np.isfinite(np.asarray(tracer)).all()
    assert float(momentum[0, -1]) == 0.0
    assert float(tracer[0, -1]) == 0.0


def test_literal_single_level_applies_diagonal_and_dry_mask():
    field = jnp.array([[6.0], [9.0]], dtype=jnp.float64)
    wet = jnp.array([[True], [False]])
    empty = jnp.empty((2, 0), dtype=jnp.float64)
    extra = jnp.array([[2.0], [7.0]], dtype=jnp.float64)
    actual = implicit_vertical_diffusion_nemo_momentum(
        field, empty, jnp.array([[3.0], [0.0]]), empty, 4.0, wet,
        extra_diag=extra)
    np.testing.assert_array_equal(np.asarray(actual), np.array([[2.0], [0.0]]))


def test_production_literal_route_uses_full_step_active_masks():
    """The model route must not broadcast a surface mask into dry rock."""
    grid = create_latlon_grid(6, 8)
    zref = create_ocean_z_star(n_levels=5, H_max=3000.0)
    z_coord = create_full_step_coordinate(
        zref, jnp.full((6, 8), 3, dtype=jnp.int32))
    H_bathy = jnp.sum(z_coord.h_partial, axis=-1)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy)
    active_t = z_coord.is_active
    active_u, active_v = compute_face_masks_3d(active_t, grid)
    # Poison the dry slots: a correct production mask both avoids 0/0 in the
    # matrix and preserves these model-level below-bottom sentinels.
    state = state._replace(
        T=state.T.replace(data=jnp.where(active_t, state.T.data, 77.0)),
        S=state.S.replace(data=jnp.where(active_t, state.S.data, 88.0)),
        u=state.u.replace(data=jnp.where(active_u, state.u.data, 99.0)),
        v=state.v.replace(data=jnp.where(active_v, state.v.data, 111.0)),
    )
    cfg = LatLonCGridOceanConfig.from_flat(
        bottom_drag_scheme="legacy", bottom_drag_r=0.0,
        A_v=1.0e-3, K_v=1.0e-4, A_h=0.0, K_h=0.0,
        implicit_vertical_mixing=True,
        zdf_implicit_solver_evaluation="nemo_literal",
    )
    out = LatLonCGridOceanModel(grid, z_coord, cfg)._apply_implicit_vertical_mixing(
        state, 1800.0, surface_forcing=None)

    for field in (out.T.data, out.S.data, out.u.data, out.v.data):
        assert np.isfinite(np.asarray(field)).all()
    np.testing.assert_array_equal(
        np.asarray(out.T.data)[~np.asarray(active_t)],
        np.asarray(state.T.data)[~np.asarray(active_t)])
    np.testing.assert_array_equal(
        np.asarray(out.S.data)[~np.asarray(active_t)],
        np.asarray(state.S.data)[~np.asarray(active_t)])
    np.testing.assert_array_equal(
        np.asarray(out.u.data)[~np.asarray(active_u, dtype=bool)],
        np.asarray(state.u.data)[~np.asarray(active_u, dtype=bool)])
    np.testing.assert_array_equal(
        np.asarray(out.v.data)[~np.asarray(active_v, dtype=bool)],
        np.asarray(state.v.data)[~np.asarray(active_v, dtype=bool)])

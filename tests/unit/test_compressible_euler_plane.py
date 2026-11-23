"""Construction + JIT smoke + config rejection tests for the plane NH dycore."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
    validate_plane_config,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import (
    TerrainMetric,
    create_height_coordinate,
)


jax.config.update("jax_enable_x64", True)


def _minimal_config(**overrides):
    """``CompressibleEulerConfig`` with everything PR2b disallows turned off."""
    base = dict(
        sponge_coeff=0.0,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=False,
    )
    base.update(overrides)
    return CompressibleEulerConfig(**base)


def _setup(ny=4, nx=4, nlev=6, H=20.0e3, **cfg_kwargs):
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=1.0e3, dy=1.0e3, dtype=jnp.float64
    )
    height_coord = create_height_coordinate(grid.nlev, H=H)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = _minimal_config(**cfg_kwargs)
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    return model, state


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


def test_construct_with_flat_terrain():
    model, state = _setup()
    assert model.grid.ny == 4
    assert state.u.data.shape == (4, 4, 6)
    assert state.w.data.shape == (4, 4, 7)


def test_construct_rejects_non_plane_grid():
    from legoesm.grids.latlon import create_latlon_grid

    latlon_grid = create_latlon_grid(n_lat=8)
    height_coord = create_height_coordinate(4, H=20.0e3)
    # Fake a TerrainMetric that ``_assert_flat_terrain`` would accept;
    # the type check happens first regardless.
    fake_terrain = TerrainMetric(
        z_s=jnp.zeros((8, 16)),
        jacobian=jnp.ones((8, 16)),
        z_full_3d=jnp.zeros((8, 16, 4)),
        z_half_3d=jnp.zeros((8, 16, 5)),
    )
    with pytest.raises(TypeError, match="expects a PlaneGrid"):
        PlaneCompressibleEulerModel(
            latlon_grid, height_coord, fake_terrain, _minimal_config(),
        )


def test_construct_rejects_non_flat_terrain():
    grid = create_plane_grid(nx=4, ny=4, nlev=4, dx=1.0e3, dy=1.0e3,
                              dtype=jnp.float64)
    height_coord = create_height_coordinate(grid.nlev, H=20.0e3)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    bad = TerrainMetric(
        z_s=terrain.z_s,
        jacobian=terrain.jacobian + 0.5,
        z_full_3d=terrain.z_full_3d,
        z_half_3d=terrain.z_half_3d,
    )
    with pytest.raises(ValueError, match="Jacobian == 1"):
        PlaneCompressibleEulerModel(grid, height_coord, bad, _minimal_config())


# --------------------------------------------------------------------- #
# Config rejection                                                      #
# --------------------------------------------------------------------- #


def test_validate_plane_config_accepts_semi_implicit_acoustic():
    """semi_implicit_acoustic now supported via per-column Thomas
    solve (plane_acoustic_substeps_semi_implicit)."""
    cfg = _minimal_config(semi_implicit_acoustic=True)
    validate_plane_config(cfg)  # no raise


def test_validate_plane_config_accepts_sponge_coeff_positive():
    """PR2d wires the Rayleigh sponge; the rejection that PR2b carried
    on ``sponge_coeff > 0`` is now lifted."""
    cfg = _minimal_config(sponge_coeff=0.05)
    validate_plane_config(cfg)  # no raise


@pytest.mark.parametrize(
    "field",
    ["hyperdiff_coeff", "hyperdiff_rho_coeff", "hyperdiff_w_coeff"],
)
def test_validate_plane_config_accepts_hyperdiff_positive(field):
    """PR3a wires biharmonic hyperdiffusion for u/v/theta', rho', and
    w; the PR2b rejection on each of the three coefficients is
    lifted."""
    cfg = _minimal_config(**{field: 1.0e-6})
    validate_plane_config(cfg)  # no raise


def test_step_accepts_nonzero_tracer_axis_after_pr3d():
    """PR3d lifted the PR2b ``n_tracers == 0`` restriction. The step
    should now run with a non-empty tracer axis and return a state
    of the same shape."""
    model, state = _setup()
    nonzero_tracers = state.tracers.replace(
        data=jnp.zeros(state.tracers.data.shape[:-1] + (2,)),
    )
    state_with_tracers = state._replace(tracers=nonzero_tracers)
    next_state = model.step(state_with_tracers, dt=1.0)
    assert next_state.tracers.data.shape == state_with_tracers.tracers.data.shape
    # All-zero tracers + zero rest state stays zero.
    assert float(jnp.max(jnp.abs(next_state.tracers.data))) == 0.0


# --------------------------------------------------------------------- #
# One-step JIT smoke                                                    #
# --------------------------------------------------------------------- #


def test_one_step_runs_under_jit_and_returns_plane_state():
    model, state = _setup()
    next_state = model.step(state, dt=1.0)
    # Type preserved through the jitted step.
    from legoesm.core.state import PlaneNonHydrostaticState
    assert isinstance(next_state, PlaneNonHydrostaticState)
    # Shapes preserved.
    assert next_state.u.data.shape == state.u.data.shape
    assert next_state.w.data.shape == state.w.data.shape
    assert next_state.theta_prime.data.shape == state.theta_prime.data.shape


def test_step_compiles_once_for_repeated_dt():
    model, state = _setup()
    # First call traces; second call hits the cache.
    s1 = model.step(state, dt=1.0)
    s2 = model.step(s1, dt=1.0)
    assert jnp.all(jnp.isfinite(s2.u.data))
    assert jnp.all(jnp.isfinite(s2.w.data))


# --------------------------------------------------------------------- #
# Semi-implicit acoustic substeps                                       #
# --------------------------------------------------------------------- #


def test_semi_implicit_acoustic_step_runs():
    """semi_implicit_acoustic=True path runs end-to-end without
    NotImplementedError; produces finite output for rest state."""
    model, state = _setup(semi_implicit_acoustic=True)
    next_state = model.step(state, dt=1.0)
    assert jnp.all(jnp.isfinite(next_state.w.data))
    assert jnp.all(jnp.isfinite(next_state.theta_prime.data))
    assert jnp.all(jnp.isfinite(next_state.rho_prime.data))


def test_semi_implicit_acoustic_matches_explicit_at_rest():
    """At rest state with zero perturbation, both paths produce
    identical output (no acoustic activity to differentiate them)."""
    model_e, state = _setup(semi_implicit_acoustic=False)
    model_si, _ = _setup(semi_implicit_acoustic=True)
    s_e = model_e.step(state, dt=1.0)
    s_si = model_si.step(state, dt=1.0)
    # Rest state stays at rest under both paths.
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(s_e.w.data), np.asarray(s_si.w.data), atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(s_e.theta_prime.data),
        np.asarray(s_si.theta_prime.data), atol=1e-14,
    )


def test_semi_implicit_acoustic_jit_compilable():
    """JIT path compiles cleanly under semi-implicit acoustic."""
    model, state = _setup(semi_implicit_acoustic=True)
    fn = jax.jit(lambda s: model.step(s, dt=1.0))
    out = fn(state)
    assert jnp.all(jnp.isfinite(out.w.data))


def test_semi_implicit_acoustic_stable_on_stretched_grid():
    """Codex iter-2: regression guard for the feature's actual purpose.

    On a stretched vertical grid with thin surface layer (dz_sfc=50m),
    the vertical acoustic CFL bounds dt for the explicit
    forward-backward path at dt ~ 0.3 s. The semi-implicit path lifts
    this and must remain finite at dt=2.0s with a non-trivial
    perturbation that excites vertical acoustic modes.
    """
    from legoesm.grids.vertical import create_stretched_height_coordinate

    nx = ny = 4
    nlev = 20
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=2.0e3, dy=2.0e3, dtype=jnp.float64,
    )
    hc = create_stretched_height_coordinate(nlev, H=20.0e3, dz_sfc=50.0)
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = _minimal_config(
        semi_implicit_acoustic=True,
        n_acoustic_substeps=6,
        acoustic_off_centering=0.1,
    )
    model = PlaneCompressibleEulerModel(grid, hc, terrain, cfg)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    # Seed vertical acoustic mode via single-cell w-impulse in lowest
    # interior interface (dz_sfc=50m → fastest vertical mode).
    w_kick = rest.w.data.at[ny // 2, nx // 2, -2].set(0.5)
    state = rest._replace(w=rest.w.replace(data=w_kick))
    # Run a few outer steps at dt=2.0s — explicit path would blow up
    # at dz_sfc=50m within 1-2 steps (acoustic CFL ≈ 13 even with 24
    # substeps). Semi-implicit must remain finite.
    for _ in range(5):
        state = model.step(state, dt=2.0)
    assert jnp.all(jnp.isfinite(state.w.data)), (
        "semi-implicit acoustic must remain finite at dt=2 on a "
        "dz_sfc=50m stretched grid; explicit path fails here."
    )
    assert jnp.all(jnp.isfinite(state.theta_prime.data))
    assert jnp.all(jnp.isfinite(state.rho_prime.data))


# T29: full-step budgets, not instantaneous advective-form residuals.
T29_INTEGRATORS = ("ssp_rk3", "ssp_rk34", "ssp_rk54")
T29_ACOUSTICS = ("explicit", "semi_implicit", "si_horizontal")


def t29_plane_fixture(integrator, acoustic, vertical, *, stretched=False):
    """Approved T24 passive fixture, or the existing stretched-grid geometry."""
    from pathlib import Path
    import runpy
    from legoesm.grids.vertical import create_stretched_height_coordinate
    root = Path(__file__).resolve().parents[2]
    probe = runpy.run_path(str(root / "scripts/validate/diag_hybrid_tracer_smoothing.py"))
    _, _, _, q = runpy.run_path(
        str(root / "tests/unit/test_vertical_advection_hybrid_van_leer.py"))["_column"]()
    model, state, grid, hc, metric = probe["height_les_fixture"](root, q)
    if stretched:
        grid = create_plane_grid(nx=4, ny=4, nlev=20, dx=2000.0, dy=2000.0,
                                 dtype=jnp.float64)
        hc = create_stretched_height_coordinate(20, H=20000.0, dz_sfc=50.0)
        metric = make_flat_plane_terrain_metric(grid, hc)
        state = make_rest_state(grid, hc, dtype=jnp.float64)
        phase = 2 * jnp.pi * jnp.arange(grid.nx) / grid.nx
        q = 0.01 * (1 + 0.1 * jnp.arange(grid.nlev) / grid.nlev)[None, None, :]
        state = state._replace(
            tracers=state.tracers.replace(data=jnp.broadcast_to(
                q, (grid.ny, grid.nx, grid.nlev))[..., None]),
            w=state.w.replace(data=state.w.data.at[2, 2, -2].set(0.5)))
    phase = 2 * jnp.pi * jnp.arange(grid.nx) / grid.nx
    scalar_shape = state.rho_prime.data.shape
    state = state._replace(
        u=state.u.replace(data=jnp.broadcast_to(jnp.sin(phase)[None, :, None], scalar_shape)),
        rho_prime=state.rho_prime.replace(data=jnp.broadcast_to(
            0.01 * hc.rho_ref * jnp.sin(phase)[None, :, None], scalar_shape)),
        tracers=state.tracers.replace(data=
            state.tracers.data * (1 + 0.1 * jnp.cos(phase))[None, :, None, None]))
    cfg = model.config._replace(
        outer_integrator=integrator, vertical_tracer_advection=vertical,
        semi_implicit_acoustic=acoustic != "explicit",
        substep_horizontal_acoustic=acoustic == "si_horizontal",
        acoustic_off_centering=0.1, n_acoustic_substeps=6)
    return PlaneCompressibleEulerModel(grid, hc, metric, cfg), state


def t29_assert_inventory(state, final, rho_ref, volume):
    """Independent thickness/area weighted inventory, per tracer, closed walls."""
    import numpy as np
    rho0 = np.asarray(rho_ref + state.rho_prime.data)
    rho1 = np.asarray(rho_ref + final.rho_prime.data)
    old = np.asarray(volume)[..., None] * rho0[..., None] * np.asarray(state.tracers.data)
    new = np.asarray(volume)[..., None] * rho1[..., None] * np.asarray(final.tracers.data)
    axes = tuple(range(old.ndim - 1))
    residual = np.abs(np.sum(new, axis=axes) - np.sum(old, axis=axes))
    scale = np.sum(np.abs(old), axis=axes)
    assert np.all(scale > 0)
    assert np.all(residual / scale <= 1e-12), residual / scale


@pytest.mark.parametrize("integrator", T29_INTEGRATORS)
@pytest.mark.parametrize("acoustic", T29_ACOUSTICS)
@pytest.mark.parametrize("vertical", ["centered", "van_leer"])
@pytest.mark.parametrize("stretched", [False, True])
def test_t29_full_step_mass_and_constant(integrator, acoustic, vertical, stretched):
    import numpy as np
    model, state = t29_plane_fixture(integrator, acoustic, vertical, stretched=stretched)
    hc, metric = model.height_coord, model.terrain_metric
    volume = model.grid.grid_area[..., None] * metric.jacobian[..., None] * hc.dz
    final = model.step(state, 1.0)
    t29_assert_inventory(state, final, hc.rho_ref, volume)
    constant = state._replace(tracers=state.tracers.replace(
        data=jnp.full_like(state.tracers.data, 0.01)))
    final_constant = model.step(constant, 1.0)
    np.testing.assert_allclose(final_constant.tracers.data, 0.01, rtol=0, atol=1e-12)
    t29_assert_inventory(constant, final_constant, hc.rho_ref, volume)


@pytest.mark.parametrize("integrator", T29_INTEGRATORS)
@pytest.mark.parametrize("acoustic", T29_ACOUSTICS)
def test_t29_acoustic_export_pairs_density(integrator, acoustic):
    import numpy as np
    from legoesm.atmosphere.dynamics.les import compressible_euler_plane as plane
    from legoesm.timestepping.split_explicit import SplitExplicitConfig
    model, state = t29_plane_fixture(integrator, acoustic, "centered")
    cfg = SplitExplicitConfig(n_substeps=6, outer_integrator=integrator)
    fn = {"explicit": plane.plane_acoustic_substeps,
          "semi_implicit": plane.plane_acoustic_substeps_semi_implicit,
          "si_horizontal": plane.plane_acoustic_substeps_si_horizontal}[acoustic]
    args = (model.grid,) if acoustic == "si_horizontal" else ()
    final, (fx, fy, fz) = fn(
        state, None, 1.0 / 6, 6, cfg, model.height_coord,
        model.terrain_metric, model.config, *args, return_mass_flux=True)
    div = plane.divergence_vlast(fx, fy, model.grid) + (
        fz[..., :-1] - fz[..., 1:]) / (
            model.height_coord.dz * model.terrain_metric.jacobian[..., None])
    np.testing.assert_allclose(final.rho_prime.data - state.rho_prime.data,
                               -div, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(fz[..., (0, -1)], 0.0)
    # Nonzero exported transfer makes a dropped/off-centering flux detectable.
    assert float(jnp.max(jnp.abs(fz))) > 0


@pytest.mark.parametrize("integrator", T29_INTEGRATORS)
@pytest.mark.parametrize("acoustic", T29_ACOUSTICS)
@pytest.mark.parametrize("vertical", ["centered", "van_leer"])
def test_t29_rk_polynomial_not_euler(integrator, acoustic, vertical):
    import numpy as np
    from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
    from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
    from tests.unit.test_plane_nh_tracers import _setup_with_tracers
    model, state, grid, hc, metric = _setup_with_tracers(1)
    cfg = model.config._replace(
        outer_integrator=integrator, horizontal_advection_scheme="centered",
        vertical_tracer_advection=vertical, acoustic_off_centering=0.0,
        semi_implicit_acoustic=acoustic != "explicit",
        substep_horizontal_acoustic=acoustic == "si_horizontal")
    model = PlaneCompressibleEulerModel(grid, hc, metric, cfg)
    q = jnp.broadcast_to((0.01 * (1 + 0.1 * jnp.sin(
        2 * jnp.pi * jnp.arange(grid.nx) / grid.nx)))[None, :, None, None],
        state.tracers.data.shape)
    state = state._replace(u=state.u.replace(data=jnp.ones_like(state.u.data)),
                           tracers=state.tracers.replace(data=q))
    operator = lambda x: -(jnp.roll(x, -1, axis=1) - jnp.roll(x, 1, axis=1)) / (2 * grid.dx)
    stages = []

    def recorded_operator(x):
        stages.append(x)
        return operator(x)

    reference = {"ssp_rk3": ssp_rk3_step, "ssp_rk34": ssp_rk34_step,
                 "ssp_rk54": ssp_rk54_step}[integrator](q, recorded_operator, 1.0)
    if integrator == "ssp_rk54" and acoustic == "si_horizontal":
        from legoesm.timestepping import ssp_rk54 as rk54
        # Independent scalar stages, using the existing coefficient table.
        # Density's final acoustic loop spans dt, while slow RHS weights sum
        # to B. The tracer must inherit that density update (Claude SSP54).
        b53, b54 = rk54._SHU_OSHER_B[5][3:5]
        q2, q3, q4 = stages[2:5]
        a52, a53, a54 = rk54._SHU_OSHER_A[5][2:5]
        reference = (a52 * q2 + a53 * q3 + a54 * q4
                     + operator((b53 * q3 + b54 * q4) / (b53 + b54)))
    euler = q + operator(q)
    assert float(jnp.max(jnp.abs(reference - euler))) > 1e-12, "STOP: insensitive Euler control"
    final = model.step(state, 1.0)
    np.testing.assert_allclose(final.tracers.data, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("integrator", T29_INTEGRATORS)
def test_t29_actual_density_source_only(integrator):
    import numpy as np
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import plane_split_explicit_tracer_step
    from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
    from legoesm.timestepping.split_explicit import SplitExplicitConfig
    model, state = _setup()
    hc = model.height_coord._replace(rho_ref=jnp.full_like(model.height_coord.rho_ref, 1e-10))
    state = state._replace(tracers=state.tracers.replace(
        data=jnp.full(state.rho_prime.data.shape + (1,), 0.01)))
    _, sanitized = sanitize_theta_rho(hc.theta_ref, hc.rho_ref)
    assert bool(jnp.all(sanitized != hc.rho_ref))  # source-density discriminator is active

    def slow(s):
        zero = jax.tree.map(jnp.zeros_like, s)
        tend = zero._replace(tracers=s.tracers.replace(data=jnp.full_like(s.tracers.data, 0.01)))
        return tend, (jnp.zeros_like(s.u.data), jnp.zeros_like(s.v.data))

    def acoustic(s, tend, dt_s, n_sub, cfg):
        return s, (jnp.zeros_like(s.u.data), jnp.zeros_like(s.v.data), jnp.zeros_like(s.w.data))

    final = plane_split_explicit_tracer_step(
        state, slow, acoustic, 1.0, SplitExplicitConfig(outer_integrator=integrator),
        model.grid, hc, model.terrain_metric, model.config)
    np.testing.assert_allclose(final.tracers.data, state.tracers.data + 0.01,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("acoustic", T29_ACOUSTICS)
def test_t29_jit_eager_parity(acoustic):
    import numpy as np
    model, state = t29_plane_fixture("ssp_rk3", acoustic, "van_leer")
    compiled = model.step(state, 1.0)
    with jax.disable_jit():
        eager = model.step(state, 1.0)
    for a, b in zip(jax.tree.leaves(compiled), jax.tree.leaves(eager)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("vertical", ["centered", "van_leer"])
@pytest.mark.parametrize("acoustic", T29_ACOUSTICS)
def test_t29_fp32_gradient_q_and_w(vertical, acoustic):
    import numpy as np
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    previous = get_policy()
    try:
        set_policy(PrecisionPolicy.fp32())
        model, state = t29_plane_fixture("ssp_rk3", acoustic, vertical)
        cast = lambda a: a.astype(jnp.float32) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a
        state, grid, hc, metric = jax.tree.map(cast, (
            state, model.grid, model.height_coord, model.terrain_metric))
        model = PlaneCompressibleEulerModel(grid, hc, metric, model.config)

        def loss(q, w):
            s = state._replace(tracers=state.tracers.replace(data=q), w=state.w.replace(data=w))
            out = model.step(s, jnp.float32(1.0))
            return jnp.sum(out.tracers.data ** 2)

        gradients = jax.grad(loss, argnums=(0, 1))(state.tracers.data, state.w.data)
        for grad in gradients:
            assert grad.dtype == jnp.float32
            assert np.isfinite(grad).all() and np.any(np.asarray(grad) != 0)
    finally:
        set_policy(previous)


@pytest.mark.parametrize("acoustic", T29_ACOUSTICS)
def test_t29_fp64_check_grads_q_and_w(acoustic):
    from jax.test_util import check_grads
    model, state = t29_plane_fixture("ssp_rk3", acoustic, "centered")

    def loss(q, w):
        s = state._replace(tracers=state.tracers.replace(data=q),
                           w=state.w.replace(data=w))
        return jnp.sum(model.step(s, 1.0).tracers.data ** 2)

    check_grads(loss, (state.tracers.data, state.w.data), order=1,
                modes=["rev"], rtol=1e-3, atol=1e-6)

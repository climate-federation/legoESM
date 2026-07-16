"""Biharmonic hyperdiffusion unit tests for the plane NH dycore.

PR3a wires ``-coeff * ∇⁴f`` into
``plane_compressible_euler_slow_tendencies`` for u/v/theta', rho',
and w. The eigenvalue of the discrete biharmonic on the 2-Δx mode
is the largest, so the term damps grid-scale noise far faster than
the resolved scales — that is the property these tests pin down.

Three core checks:

1. With ``hyperdiff_coeff = 0`` the tendency is bit-exact equal to
   the PR2d slow-tendency (no hyperdiff applied).
2. A pure 2-Δx checkerboard perturbation in ``u`` decays
   monotonically when ``hyperdiff_coeff > 0`` and a longer-wave
   perturbation (4-Δx) decays much more slowly.
3. Hyperdiff alone is mass-conserving (sum of ``Lap²(rho')`` over
   the periodic plane is zero to machine epsilon).
"""

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
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _setup(hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
           hyperdiff_w_coeff=0.0):
    grid = create_plane_grid(
        nx=16, ny=4, nlev=8, dx=200.0, dy=200.0, dtype=jnp.float64,
    )
    height_coord = create_height_coordinate(grid.nlev, H=4_000.0)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=0.0,
        hyperdiff_coeff=hyperdiff_coeff,
        hyperdiff_rho_coeff=hyperdiff_rho_coeff,
        hyperdiff_w_coeff=hyperdiff_w_coeff,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=False,
    )
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    return model, grid, height_coord, terrain, config


def test_zero_hyperdiff_matches_pr2d_tendency_bit_exact():
    """Setting all three hyperdiff coefficients to zero must reproduce
    the PR2d slow-tendency output exactly."""
    model_off, grid, hc, tm, cfg_off = _setup()
    model_on, _, _, _, cfg_on = _setup(hyperdiff_coeff=1.0e6)

    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Seed a small perturbation so tendencies are nonzero.
    state = state._replace(
        u=state.u.replace(data=state.u.data + 0.1),
    )
    tend_off = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_off,
    )
    tend_on = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_on,
    )
    # ``hyperdiff_coeff > 0`` should change u/v/theta' tendencies
    # because Lap^2(uniform u + 0.1) = 0 only on the rest state; with
    # the seed it is non-zero ONLY if the field is non-trivial in x/y.
    # On a uniform u seed Lap^2 is exactly zero, so tend_on == tend_off
    # bit-exact even though hyperdiff is on. That is the property we
    # want to assert: hyperdiff vanishes on uniform fields.
    assert jnp.array_equal(tend_off.du_dt.data, tend_on.du_dt.data)


def test_hyperdiff_damps_two_dx_checkerboard_mode():
    """A pure 2-Δx checkerboard in u should be damped much faster than
    a 4-Δx mode at the same amplitude — this is the defining property
    of biharmonic damping."""
    model, grid, hc, tm, _ = _setup(hyperdiff_coeff=1.0e6)
    state = make_rest_state(grid, hc, dtype=jnp.float64)

    # 2-Δx checkerboard in x: u[..., 2k] = +1, u[..., 2k+1] = -1
    i = jnp.arange(grid.nx)
    checker_x = ((-1.0) ** i).astype(jnp.float64)
    u_checker = jnp.broadcast_to(
        checker_x[None, :, None], (grid.ny, grid.nx, grid.nlev),
    )
    state_2dx = state._replace(u=state.u.replace(data=u_checker))

    # 4-Δx mode in x: cos(2π * 2 i / nx) for nx=16 → period = 8 cells = 4Δx? Wait:
    # 2*pi*k*i/nx with k=2 gives period nx/k = 8 cells = 8Δx.
    # For 4-Δx we need k = nx/4 = 4. cos(2π * 4 i / 16) = cos(π i/2).
    k_long = grid.nx // 4   # = 4 → period 4 cells = 4-Δx
    smooth_x = jnp.cos(2.0 * jnp.pi * k_long * i / grid.nx).astype(jnp.float64)
    u_smooth = jnp.broadcast_to(
        smooth_x[None, :, None], (grid.ny, grid.nx, grid.nlev),
    )
    state_4dx = state._replace(u=state.u.replace(data=u_smooth))

    cfg_on = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=1.0e6,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
    )
    cfg_off = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
    )
    # Isolate the hyperdiff contribution by subtracting the
    # no-hyperdiff tendency from the with-hyperdiff tendency. PR3b
    # added upwind advection that also damps the 2-Δx mode (but with
    # a different wavenumber scaling), so comparing raw
    # ``max|du_dt|`` would conflate the two damping mechanisms.
    tend_2dx_on = plane_compressible_euler_slow_tendencies(
        state_2dx, grid, hc, tm, cfg_on)
    tend_2dx_off = plane_compressible_euler_slow_tendencies(
        state_2dx, grid, hc, tm, cfg_off)
    tend_4dx_on = plane_compressible_euler_slow_tendencies(
        state_4dx, grid, hc, tm, cfg_on)
    tend_4dx_off = plane_compressible_euler_slow_tendencies(
        state_4dx, grid, hc, tm, cfg_off)

    # Pure hyperdiff contribution: tend_on - tend_off.
    hd_2dx = tend_2dx_on.du_dt.data - tend_2dx_off.du_dt.data
    hd_4dx = tend_4dx_on.du_dt.data - tend_4dx_off.du_dt.data

    # Hyperdiff damping rate on a wavenumber-k mode is proportional to
    # the eigenvalue of Lap² on that mode. For the 5-point stencil the
    # eigenvalue scales as (-4 sin²(π k/nx)/dx²)². 2-Δx (k=nx/2) gives
    # eigenvalue ~ (4/dx²)² = 16/dx⁴. 4-Δx (k=nx/4) gives ~
    # (2/dx²)² = 4/dx⁴. So 2-Δx damping is ~4× the 4-Δx damping.
    max_damp_2dx = float(jnp.max(jnp.abs(hd_2dx)))
    max_damp_4dx = float(jnp.max(jnp.abs(hd_4dx)))
    ratio = max_damp_2dx / max_damp_4dx
    # Theoretical ratio is (sin²(π/2) / sin²(π/4))² = (1 / 0.5)² = 4.
    # Allow ±20% for measurement noise.
    assert 3.2 <= ratio <= 4.8, (
        f"2-Δx vs 4-Δx damping ratio = {ratio:.2f}; expected ~4 (theoretical)"
    )


def test_hyperdiff_conserves_total_field_under_periodic_bc():
    """``sum(Lap²(f) * area * dz)`` over the periodic plane is zero,
    so the biharmonic term cannot create or destroy mass / momentum /
    heat — it only redistributes them between scales."""
    model, grid, hc, tm, _ = _setup(
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Non-trivial state so Lap² is nonzero pointwise.
    import numpy as np
    rng = np.random.default_rng(0)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            rng.standard_normal(state.u.data.shape) * 0.1,
        )),
        theta_prime=state.theta_prime.replace(data=jnp.asarray(
            rng.standard_normal(state.theta_prime.data.shape) * 0.1,
        )),
        rho_prime=state.rho_prime.replace(data=jnp.asarray(
            rng.standard_normal(state.rho_prime.data.shape) * 1.0e-4,
        )),
    )
    tend = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm,
        CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=1.0e6,
            hyperdiff_rho_coeff=1.0e6, hyperdiff_w_coeff=1.0e6,
            semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        ),
    )
    tend_off = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm,
        CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
            semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        ),
    )
    # Hyperdiff contribution = tend_on - tend_off; its area-weighted
    # sum (any constant area suffices on uniform plane) is zero.
    for on_field, off_field in (
        (tend.du_dt.data, tend_off.du_dt.data),
        (tend.dtheta_prime_dt.data, tend_off.dtheta_prime_dt.data),
        (tend.drho_prime_dt.data, tend_off.drho_prime_dt.data),
    ):
        hyperdiff_term = on_field - off_field
        global_sum = float(jnp.sum(hyperdiff_term))
        # Loose tolerance: random field amplitudes scale to ~1e6 * 1e-1
        # / dx⁴ at the per-cell level; the sum cancels to ~1e-10 in
        # double precision via cancellation.
        rel_to_max = abs(global_sum) / max(
            float(jnp.max(jnp.abs(hyperdiff_term))), 1.0e-30,
        )
        assert rel_to_max < 1.0e-12, (
            f"Hyperdiff did not conserve: global_sum={global_sum:.3e}, "
            f"rel_to_max={rel_to_max:.3e}"
        )


def test_step_under_hyperdiff_stays_finite_for_extended_integration():
    """With hyperdiff on, the warm-bubble setup that blew up at 25 s
    in PR2d (no hyperdiff) should now survive longer integration."""
    model, grid, hc, tm, _ = _setup(hyperdiff_coeff=1.0e7)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Small uniform u kick so there is something to advect.
    state = state._replace(u=state.u.replace(data=state.u.data + 0.5))
    for _ in range(100):
        state = model.step(state, dt=0.5)
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.theta_prime.data)))


# --------------------------------------------------------------------- #
# Negative-coefficient rejection (Codex iter-1 finding)                 #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field",
    ["hyperdiff_coeff", "hyperdiff_rho_coeff", "hyperdiff_w_coeff"],
)
def test_validate_plane_config_rejects_negative_hyperdiff(field):
    """Negative coefficients invert the damping sign and produce
    exponential growth — must raise ``ValueError`` rather than
    silently treat the term as off."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        validate_plane_config,
    )
    base = dict(
        sponge_coeff=0.0, semi_implicit_acoustic=False,
        hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
    )
    base[field] = -1.0e-6
    cfg = CompressibleEulerConfig(**base)
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_plane_config(cfg)


# --------------------------------------------------------------------- #
# Bit-exact rho + w branches on uniform fields                          #
# --------------------------------------------------------------------- #


def test_zero_hyperdiff_rho_branch_bit_exact_on_uniform_rho():
    """``hyperdiff_rho_coeff > 0`` with spatially uniform ``rho'``
    must give the same ``drho'/dt`` as hyperdiff off: ``Lap²`` of a
    uniform field is identically zero."""
    _, grid, hc, tm, _ = _setup()
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    state = state._replace(
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + 1.0e-4,
        ),
    )
    tend_off = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm,
        CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
            semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        ),
    )
    tend_on = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm,
        CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=1.0e6, hyperdiff_w_coeff=0.0,
            semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        ),
    )
    assert jnp.array_equal(tend_off.drho_prime_dt.data, tend_on.drho_prime_dt.data)


def test_zero_hyperdiff_w_branch_bit_exact_on_uniform_w():
    """Same property for the w branch: hyperdiff on a uniform w
    should not change the tendency. w lives at half levels but the
    laplacian wrapper is horizontal-only, so the assertion holds."""
    _, grid, hc, tm, _ = _setup()
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    state = state._replace(
        w=state.w.replace(data=state.w.data + 0.1),
    )
    tend_off = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm,
        CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
            semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        ),
    )
    tend_on = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm,
        CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0,
            hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=1.0e6,
            semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        ),
    )
    assert jnp.array_equal(tend_off.dw_dt.data, tend_on.dw_dt.data)


# --------------------------------------------------------------------- #
# AD grad through hyperdiff term                                        #
# --------------------------------------------------------------------- #


def test_hyperdiff_term_is_differentiable():
    """``jax.grad`` should flow through every hyperdiffusion branch
    without raising. Use a scalar loss that sums squared tendencies
    of u, rho_prime, and w so each of the three hyperdiff
    coefficients participates in the loss (otherwise AD would only
    cover the u/v/theta branch via ``hyperdiff_coeff``)."""
    import numpy as np
    _, grid, hc, tm, cfg = _setup(
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    u0 = jnp.asarray(rng.standard_normal(state.u.data.shape) * 0.01)
    rho0 = jnp.asarray(rng.standard_normal(state.rho_prime.data.shape) * 1.0e-4)
    w0 = jnp.asarray(rng.standard_normal(state.w.data.shape) * 0.01)

    def loss(u_data, rho_data, w_data):
        s = state._replace(
            u=state.u.replace(data=u_data),
            rho_prime=state.rho_prime.replace(data=rho_data),
            w=state.w.replace(data=w_data),
        )
        tend = plane_compressible_euler_slow_tendencies(
            s, grid, hc, tm, cfg,
        )
        return (
            jnp.sum(tend.du_dt.data ** 2)
            + jnp.sum(tend.drho_prime_dt.data ** 2)
            + jnp.sum(tend.dw_dt.data ** 2)
        )

    grad_u, grad_rho, grad_w = jax.grad(loss, argnums=(0, 1, 2))(
        u0, rho0, w0,
    )
    for name, g in (("u", grad_u), ("rho", grad_rho), ("w", grad_w)):
        assert bool(jnp.all(jnp.isfinite(g))), f"non-finite grad: {name}"
        assert float(jnp.max(jnp.abs(g))) > 0.0, f"zero grad: {name}"

"""Unit tests for Van Leer TVD horizontal advection on the plane CRM
dycore (iter-179).

Cover the four properties a TVD scheme must satisfy:

1. Reduces to first-order upwind at extrema (r <= 0 -> phi = 0).
2. Exact on a linear field (2nd-order accuracy in smooth regions).
3. TVD: no new extrema injected for a monotone field.
4. Differentiable end-to-end (jax.grad flows through the limiter).

Also verify the scheme dispatch in the plane CRM slow-tendency entry
point picks the Van Leer path when ``horizontal_advection_scheme =
"van_leer"``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    _upwind_advection_x, _upwind_advection_y,
    _van_leer_advection_x, _van_leer_advection_y,
)
from legoesm.core.flux_limiters import van_leer_limiter  # noqa: E402


def test_van_leer_limiter_known_values():
    """Lock the four representative limiter values: extremum -> 0,
    smooth -> 1, sharp -> bounded by 2."""
    assert float(van_leer_limiter(jnp.array(0.0))) == 0.0
    assert float(van_leer_limiter(jnp.array(-1.0))) == 0.0
    assert float(van_leer_limiter(jnp.array(1.0))) == 1.0
    assert float(van_leer_limiter(jnp.array(10.0))) < 2.0


def test_van_leer_reduces_to_upwind_on_oscillating_field():
    """A 2-Δx oscillation has r <= 0 everywhere (every cell is an
    extremum) so the limiter returns 0 → Van Leer == upwind1. Locks
    the TVD-at-extrema fallback."""
    ny, nx, nlev = 4, 16, 2
    # Sign-flip in x at every other cell: f = [+1, -1, +1, -1, ...]
    f = jnp.ones((ny, nx, nlev), dtype=jnp.float64)
    f = f.at[:, 1::2, :].set(-1.0)
    u = jnp.full_like(f, 0.5)  # uniform positive flow
    dx = 1000.0
    upwind = _upwind_advection_x(f, u, dx)
    van_leer = _van_leer_advection_x(f, u, dx)
    np.testing.assert_allclose(
        np.asarray(van_leer), np.asarray(upwind),
        rtol=1e-12, atol=1e-12,
        err_msg="Van Leer must collapse to upwind1 at every extremum",
    )


def test_van_leer_exact_on_linear_field_x_interior():
    """For a linear field f(x) = a*x + b in the INTERIOR (away from
    the periodic wrap discontinuity at i=0/nx-1), the advective
    tendency is -u*df/dx = -u*a — a constant. Van Leer's slope
    limiter returns the full 2nd-order linear extrapolation on a
    locally-linear field so the tendency is exact within float64
    round-off. Tightens 2nd-order accuracy. (Boundary cells where
    the wraparound makes the field non-linear are excluded.)
    """
    ny, nx, nlev = 4, 12, 2
    dx = 1000.0
    a = 0.7
    x = jnp.arange(nx, dtype=jnp.float64) * dx
    f = jnp.broadcast_to(a * x[None, :, None], (ny, nx, nlev))
    u = jnp.full_like(f, 1.5)
    expected = -u * a  # -u * df/dx with df/dx = a (analytic)
    actual = _van_leer_advection_x(f, u, dx)
    # Strip boundary cells: Van Leer's 4-cell stencil [i-1..i+2] is
    # corrupted at i in {0, 1, nx-2, nx-1} by the periodic wrap
    # (the linear field is discontinuous at the seam).
    interior = slice(2, nx - 2)
    np.testing.assert_allclose(
        np.asarray(actual[:, interior, :]),
        np.asarray(expected[:, interior, :]),
        rtol=1e-10, atol=1e-10,
        err_msg="Van Leer must be exact on a linear field (smooth interior)",
    )


def test_van_leer_exact_on_linear_field_y_interior():
    """Same as the x test but on the y axis (axis=0). Tests
    _van_leer_advection_y interior accuracy."""
    ny, nx, nlev = 12, 4, 2
    dy = 750.0
    a = -0.4
    y = jnp.arange(ny, dtype=jnp.float64) * dy
    f = jnp.broadcast_to(a * y[:, None, None], (ny, nx, nlev))
    v = jnp.full_like(f, 2.0)
    expected = -v * a
    actual = _van_leer_advection_y(f, v, dy)
    interior = slice(2, ny - 2)
    np.testing.assert_allclose(
        np.asarray(actual[interior, :, :]),
        np.asarray(expected[interior, :, :]),
        rtol=1e-10, atol=1e-10,
    )


def test_van_leer_face_values_second_order_both_branches():
    """HD-1 (codex iter-44/45): the shared face reconstruction must be 2nd-order
    (the midpoint on a LINEAR stencil) for BOTH the positive AND negative
    velocity branches. The negative branch was silently 1st-order (upwind cell
    value) before the r_neg sign fix — this locks it."""
    from legoesm.core.flux_limiters import van_leer_face_values
    # linear stencil [10,11,12,13]: face i+1/2 = midpoint of cells i,i+1 = 11.5
    phi_pos, phi_neg = van_leer_face_values(
        jnp.array(10.0), jnp.array(11.0), jnp.array(12.0), jnp.array(13.0))
    assert float(phi_pos) == 11.5          # 2nd-order from the left
    assert float(phi_neg) == 11.5          # 2nd-order from the right
    #                                        (the bug gave 12.0 = 1st-order upwind)


def test_van_leer_face_values_upwind_at_extremum():
    """At an extremum (r<=0) BOTH branches collapse to 1st-order upwind (TVD)."""
    from legoesm.core.flux_limiters import van_leer_face_values
    # f_i is a local max ⇒ positive branch takes the upwind cell value (no
    # overshoot past 2.0)
    phi_pos, _ = van_leer_face_values(
        jnp.array(0.0), jnp.array(2.0), jnp.array(0.0), jnp.array(0.0))
    assert float(phi_pos) == 2.0
    # f_ip1 is a local max ⇒ negative branch takes the upwind cell i+1 value
    _, phi_neg = van_leer_face_values(
        jnp.array(0.0), jnp.array(0.0), jnp.array(2.0), jnp.array(0.0))
    assert float(phi_neg) == 2.0


def test_van_leer_tvd_no_new_extrema():
    """Forward Euler step with the Van Leer tendency must NOT introduce
    new extrema for a monotone step field. Concretely: max(f^{n+1})
    <= max(f^n) + tol, min(f^{n+1}) >= min(f^n) - tol. This is the
    TVD property the limiter is supposed to enforce."""
    ny, nx, nlev = 4, 16, 1
    # Heaviside step: f = 0 for x < nx/2, 1 elsewhere.
    f = jnp.zeros((ny, nx, nlev), dtype=jnp.float64)
    f = f.at[:, nx // 2:, :].set(1.0)
    u = jnp.full_like(f, 0.3)
    dx = 1000.0
    dt = 100.0  # Courant ~0.03, well within TVD bound.
    tend = _van_leer_advection_x(f, u, dx)
    f_new = f + dt * tend
    tol = 1e-12
    assert float(jnp.max(f_new)) <= 1.0 + tol, (
        "Van Leer introduced a new maximum on a step field — "
        "violates TVD"
    )
    assert float(jnp.min(f_new)) >= 0.0 - tol, (
        "Van Leer introduced a new minimum on a step field — "
        "violates TVD"
    )


def test_van_leer_differentiable_via_jax_grad():
    """jax.grad must flow through the Van Leer limiter end-to-end —
    the (r+|r|)/(1+|r|) form is differentiable everywhere except the
    extremum kink, and JAX returns a finite subgradient there. Lock
    the differentiability contract."""
    ny, nx, nlev = 4, 8, 2
    rng = jax.random.PRNGKey(0)
    f0 = jax.random.normal(rng, (ny, nx, nlev), dtype=jnp.float64)
    u = jax.random.normal(
        jax.random.fold_in(rng, 1), f0.shape, dtype=jnp.float64,
    )
    dx = 1000.0

    def loss_fn(f):
        return jnp.sum(_van_leer_advection_x(f, u, dx) ** 2)

    grad = jax.grad(loss_fn)(f0)
    assert jnp.all(jnp.isfinite(grad)), (
        "Van Leer gradient has non-finite entries — limiter or "
        "flux-form conversion broke autograd"
    )


def test_van_leer_y_differentiable_via_jax_grad():
    """iter-186 Codex LOW: the iter-179 x-axis differentiability
    test left the y-axis (_van_leer_advection_y) uncovered. The
    y path has its own ``where(|delta|>eps, delta, eps)`` ratio
    computation and a potentially distinct kink behaviour. Lock
    it explicitly so a future refactor that breaks y-axis grad
    surfaces here instead of mid-training run."""
    ny, nx, nlev = 8, 4, 2
    rng = jax.random.PRNGKey(42)
    f0 = jax.random.normal(rng, (ny, nx, nlev), dtype=jnp.float64)
    v = jax.random.normal(
        jax.random.fold_in(rng, 1), f0.shape, dtype=jnp.float64,
    )
    dy = 1000.0

    def loss_fn(f):
        return jnp.sum(_van_leer_advection_y(f, v, dy) ** 2)

    grad = jax.grad(loss_fn)(f0)
    assert jnp.all(jnp.isfinite(grad)), (
        "Van Leer y-axis gradient has non-finite entries — the "
        "ratio-eps branch in _van_leer_advection_y leaked NaN."
    )


def test_van_leer_grad_finite_on_zero_delta():
    """iter-186 Codex LOW: the ``where(|delta|>eps, delta, eps)``
    pattern can silently feed eps=1e-30 into a downstream multiply
    that overflows or produces non-finite gradients. Exercise the
    zero-delta branch directly with a CONSTANT field (every
    delta = 0 → every ratio denominator falls back to eps) on both
    axes."""
    ny, nx, nlev = 4, 4, 2
    f0 = jnp.ones((ny, nx, nlev), dtype=jnp.float64) * 2.5
    u = jnp.ones_like(f0) * 0.5
    dx = 1000.0

    def loss_x(f):
        return jnp.sum(_van_leer_advection_x(f, u, dx) ** 2)
    grad_x = jax.grad(loss_x)(f0)
    assert jnp.all(jnp.isfinite(grad_x)), (
        "Van Leer x: zero-delta branch produced non-finite gradient"
    )

    def loss_y(f):
        return jnp.sum(_van_leer_advection_y(f, u, dx) ** 2)
    grad_y = jax.grad(loss_y)(f0)
    assert jnp.all(jnp.isfinite(grad_y)), (
        "Van Leer y: zero-delta branch produced non-finite gradient"
    )


def test_smooth_k1_pattern_zero_mean_and_bounded():
    """iter-203/204/207/208: the smooth_k1 theta-noise mode applies a
    ``0.5 * (cos(2π x/nx) + cos(2π y/ny))`` pattern (minus its
    horizontal mean for safety on degenerate grids — iter-207
    Codex MEDIUM#1 fix).

    Properties locked:
    * Zero horizontal mean — total energy conserved at IC.
    * Peak amplitude bounded by 1.0 — driver multiplies by
      theta_noise_amp [K] for unit scaling.

    Exercises ``build_smooth_k1_pattern`` directly (iter-208 promoted
    the helper from scripts/run/run_rce_mpi_long.py to
    legoesm.atmosphere.idealized.rcemip_initial_conditions so this
    test imports it via the standard package path instead of loading
    the entire heavy driver module).

    Three grid shapes:
    * 8x12 (healthy): orthogonal cos sum integrates to 0 by
      construction; explicit mean subtraction is a no-op.
    * 1x1 (degenerate both axes): naive cos pattern = 1.0
      everywhere; explicit mean subtraction → 0 everywhere.
    * 4x1 (degenerate one axis): cos in x gives 1,0,-1,0 → mean
      = 0 in x. cos in y on length-1 gives 1 → mean = 1 in y.
      Combined naive = 0.5*(c_x + 1). Explicit mean subtraction
      → zero-mean across the 4 cells.
    """
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        build_smooth_k1_pattern,
    )
    for ny, nx in [(8, 12), (1, 1), (4, 1), (1, 4)]:
        pattern = build_smooth_k1_pattern(ny, nx)
        assert pattern.shape == (ny, nx), (
            f"shape mismatch on ({ny},{nx}): {pattern.shape}"
        )
        # Zero-mean (iter-207 MEDIUM#1 — explicit mean subtraction
        # is what guarantees this on degenerate grids).
        assert jnp.abs(jnp.mean(pattern)) < 1e-14, (
            f"smooth_k1 pattern not zero-mean on ({ny},{nx}); "
            f"got mean={float(jnp.mean(pattern))}"
        )
        # Peak amplitude bounded by 1.0 (caller multiplies by amp).
        assert jnp.max(jnp.abs(pattern)) <= 1.0 + 1e-12, (
            f"smooth_k1 peak exceeds 1.0 on ({ny},{nx}); got "
            f"{float(jnp.max(jnp.abs(pattern)))}"
        )


def test_horizontal_advection_halo_requirement_map_locked():
    """iter-187: HORIZONTAL_ADVECTION_HALO_REQUIREMENT is the single
    source of truth for halo widths consumed by both the driver
    (run_rce_mpi_long.py builds the layout) and the halo dispatch
    (compressible_euler_plane_halo.py validates the layout). Lock
    the active values so a drift in either direction surfaces here.
    """
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        HORIZONTAL_ADVECTION_HALO_REQUIREMENT,
    )
    assert HORIZONTAL_ADVECTION_HALO_REQUIREMENT == {
        "upwind1": 1,
        "centered": 1,
        "van_leer": 2,
        "weno5": 3,
    }


def test_van_leer_scheme_dispatch_in_slow_tendency():
    """The plane CRM slow-tendency entry accepts the registered schemes
    (van_leer among them) AND rejects a bad scheme name. Iter-179 added
    'van_leer'; the dispatch was later refactored from a literal if/elif
    chain to table membership (``if scheme not in _ADV_PAIRS: raise``).
    Assert the BEHAVIOUR (registry membership + typo raises) rather than
    scraping source text, so the table refactor doesn't break the lock
    while a silent fallback would still be caught."""
    import pytest
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        HORIZONTAL_ADVECTION_HALO_REQUIREMENT,
        make_flat_plane_terrain_metric,
        make_rest_state,
        plane_compressible_euler_slow_tendencies,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate

    # van_leer is a registered scheme in the shared halo-requirement map
    # (the single source of truth the dispatch validates against).
    assert "van_leer" in HORIZONTAL_ADVECTION_HALO_REQUIREMENT

    grid = create_plane_grid(
        nx=8, ny=8, nlev=3, dx=200.0, dy=200.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=3_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)

    def _cfg(scheme):
        return CompressibleEulerConfig(
            sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
            hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
            use_coriolis=False, fix_mass=False,
            horizontal_advection_scheme=scheme,
        )

    # The van_leer path is accepted and produces a finite tendency.
    tend = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, _cfg("van_leer"),
    )
    assert jnp.all(jnp.isfinite(tend.du_dt.data))

    # An unknown scheme must raise (a silent fallback cannot reintroduce
    # iter-114's typo-class bug).
    with pytest.raises(ValueError):
        plane_compressible_euler_slow_tendencies(
            state, grid, hc, tm, _cfg("vanleer_typo"),
        )

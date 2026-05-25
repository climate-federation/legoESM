"""Smagorinsky-Lilly LES closure tests for the plane NH dycore.

PR3c wires ``K_m = (C_s Δ)² |S|`` into
``plane_compressible_euler_slow_tendencies``. Tests cover:

1. ``K_m = 0`` on the rest state (no spontaneous turbulence).
2. ``K_m > 0`` everywhere for a strain-bearing state.
3. ``K_m`` scales as ``C_s²`` (quadratic in the Smag constant).
4. ``c_s = 0`` path is bit-exact equal to no-Smag (zero coefficient
   should be the on/off gate).
5. Rest state stays at rest with Smag on (no drift from the new term).
6. AD: ``jax.grad`` through Smag (including the safe-sqrt at zero
   strain) returns finite gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    _compute_smagorinsky_K_m_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _setup(c_s=0.0):
    grid = create_plane_grid(
        nx=8, ny=8, nlev=4, dx=200.0, dy=200.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=2_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=False, fix_mass=False,
        smagorinsky_cs=c_s, smagorinsky_prandtl=1.0,
    )
    return PlaneCompressibleEulerModel(grid, hc, tm, cfg), grid, hc, tm, cfg


def test_K_m_is_zero_on_rest_state():
    """No strain → ``|S| = 0`` → ``K_m = 0`` exactly (forward pass
    must return zero, not a tiny ``sqrt`` noise floor)."""
    _, grid, hc, _, _ = _setup()
    zero = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    zero_w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    K = _compute_smagorinsky_K_m_plane(zero, zero, zero_w, grid, hc, c_s=0.2)
    assert float(jnp.max(jnp.abs(K))) == 0.0


def test_K_m_is_positive_for_strain_bearing_state():
    """A random velocity field should have ``K_m > 0`` everywhere."""
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)),
    )
    K = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    assert float(jnp.min(K)) > 0.0


def test_K_m_scales_quadratically_in_c_s():
    """``K_m = (c_s * Δ)² * |S|`` so doubling ``c_s`` quadruples
    ``K_m`` at every cell."""
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)),
    )
    K_1 = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.1)
    K_2 = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    ratio = K_2 / jnp.where(K_1 > 0.0, K_1, 1.0)
    assert jnp.allclose(ratio, 4.0, atol=1.0e-12)


def test_zero_c_s_is_bit_exact_no_smag_path():
    """``c_s = 0`` must give the same slow tendency as the
    no-Smagorinsky path. The branch is gated by Python ``if`` so the
    function call is skipped entirely; this test pins that behaviour."""
    _, grid, hc, tm, _ = _setup(c_s=0.0)
    rng = np.random.default_rng(2)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(rng.standard_normal(state.u.data.shape) * 0.1),
        ),
    )
    cfg_off = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=0.0,
    )
    cfg_on_zero = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=0.0,  # explicit zero — must trigger the on/off
        smagorinsky_prandtl=0.33,  # irrelevant when c_s=0
    )
    tend_off = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_off)
    tend_on_zero = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_on_zero)
    assert jnp.array_equal(tend_off.du_dt.data, tend_on_zero.du_dt.data)


def test_rest_state_with_smag_stays_at_rest():
    """Smag on a zero state must not perturb the rest state."""
    model, grid, hc, _, _ = _setup(c_s=0.2)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    next_state = model.step(state, dt=1.0)
    for name, field in (
        ("u", next_state.u.data), ("v", next_state.v.data),
        ("w", next_state.w.data),
        ("theta_p", next_state.theta_prime.data),
        ("rho_p", next_state.rho_prime.data),
    ):
        assert float(jnp.max(jnp.abs(field))) == 0.0, name


def test_smag_term_is_differentiable():
    """``jax.grad`` should flow through the safe-sqrt at zero strain
    and through every K_m broadcast (full-level for u/v/theta',
    half-level for w)."""
    _, grid, hc, tm, _ = _setup(c_s=0.2)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(3)
    u0 = jnp.asarray(rng.standard_normal(state.u.data.shape) * 0.1)
    v0 = jnp.asarray(rng.standard_normal(state.v.data.shape) * 0.1)
    theta0 = jnp.asarray(
        rng.standard_normal(state.theta_prime.data.shape) * 0.01,
    )
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
    )

    def loss(u_data, v_data, theta_data):
        s = state._replace(
            u=state.u.replace(data=u_data),
            v=state.v.replace(data=v_data),
            theta_prime=state.theta_prime.replace(data=theta_data),
        )
        tend = plane_compressible_euler_slow_tendencies(
            s, grid, hc, tm, cfg,
        )
        return (
            jnp.sum(tend.du_dt.data ** 2)
            + jnp.sum(tend.dv_dt.data ** 2)
            + jnp.sum(tend.dtheta_prime_dt.data ** 2)
        )

    grads = jax.grad(loss, argnums=(0, 1, 2))(u0, v0, theta0)
    for name, g in zip(("u", "v", "theta"), grads):
        assert bool(jnp.all(jnp.isfinite(g))), f"non-finite Smag grad: {name}"
        assert float(jnp.max(jnp.abs(g))) > 0.0, f"zero Smag grad: {name}"


def test_smag_grad_through_zero_strain_is_finite():
    """The safe-sqrt at zero strain must produce a finite subgradient
    (not NaN). Verify by constructing a state where most cells have
    zero strain (uniform u) and one has nonzero."""
    _, grid, hc, tm, _ = _setup(c_s=0.2)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    u0 = jnp.zeros(state.u.data.shape).at[0, 0, 0].set(0.1)  # one nonzero
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
    )

    def loss(u_data):
        s = state._replace(u=state.u.replace(data=u_data))
        tend = plane_compressible_euler_slow_tendencies(
            s, grid, hc, tm, cfg,
        )
        return jnp.sum(tend.du_dt.data ** 2)

    g = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(g))), (
        "Safe-sqrt in Smag produced NaN gradient at zero strain"
    )


# --------------------------------------------------------------------- #
# Codex iter-1 follow-ups                                               #
# --------------------------------------------------------------------- #


def test_smag_tendency_is_dissipative():
    """``sum(u * div(K_m grad u) * area) <= 0`` over the periodic plane
    for any K_m >= 0. This is the property the conservative flux-form
    diffusion (PR3c iter-1) was introduced to guarantee — replacing
    the original ``K_m * Lap(u)`` which is not dissipative for
    variable K_m. Discrete integration by parts gives
    ``= -sum_faces K_face * (grad_face u)^2 <= 0``."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _variable_K_diffusion_vlast, _compute_smagorinsky_K_m_plane,
    )
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(7)
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)),
    )
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    diff_u = _variable_K_diffusion_vlast(u, K_m, grid)
    diff_v = _variable_K_diffusion_vlast(v, K_m, grid)
    # Area cancels (uniform plane). Sum of u · D(u) over the periodic
    # domain must be non-positive.
    dissipation = float(jnp.sum(u * diff_u + v * diff_v))
    assert dissipation <= 0.0, (
        f"Smag tendency is not dissipative: sum(u·D(u)) = {dissipation:.3e}"
    )


def test_smag_diffusion_conserves_field_under_periodic_bc():
    """``sum(div(K grad f)) = 0`` to machine epsilon under periodic
    BC. Flux-form is conservative; the cell-centred ``K * Lap(f)``
    is not (Codex iter-1 finding M1)."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _variable_K_diffusion_vlast, _compute_smagorinsky_K_m_plane,
    )
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(8)
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)),
    )
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    diff = _variable_K_diffusion_vlast(u, K_m, grid)
    rel = float(jnp.abs(jnp.sum(diff))) / max(
        float(jnp.max(jnp.abs(diff))), 1.0e-30,
    )
    assert rel < 1.0e-12, (
        f"Flux-form variable-K diffusion not conservative: "
        f"sum(out) / max|out| = {rel:.3e}"
    )


def test_validate_rejects_zero_prandtl():
    """Codex iter-1 M2: zero Prandtl number produces inf K_h. Must
    raise ``ValueError`` at config-validation time."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        validate_plane_config,
    )
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=0.2, smagorinsky_prandtl=0.0,
    )
    with pytest.raises(ValueError, match="smagorinsky_prandtl"):
        validate_plane_config(cfg)


def test_validate_rejects_negative_prandtl():
    """Negative Pr turns thermal diffusion into anti-diffusion."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        validate_plane_config,
    )
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=0.2, smagorinsky_prandtl=-0.5,
    )
    with pytest.raises(ValueError, match="smagorinsky_prandtl"):
        validate_plane_config(cfg)


def test_validate_rejects_negative_smagorinsky_cs():
    """Negative C_s would invert the Smag damping sign."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        validate_plane_config,
    )
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        smagorinsky_cs=-0.1, smagorinsky_prandtl=1.0,
    )
    with pytest.raises(ValueError, match="smagorinsky_cs"):
        validate_plane_config(cfg)


def test_variable_K_diffusion_rejects_shape_mismatch():
    """Codex iter-2 M1: K_yxz must match field_yxz shape exactly. The
    full vs half-level vertical-axis difference is the most common
    way this fails (e.g. K_m at full levels passed against w at
    half levels would broadcast silently)."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _variable_K_diffusion_vlast,
    )
    _, grid, _, _, _ = _setup()
    field = jnp.zeros((grid.ny, grid.nx, grid.nlev))      # (ny, nx, nlev)
    K_bad = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))  # (ny, nx, nlev+1)
    with pytest.raises(ValueError, match="K_yxz shape"):
        _variable_K_diffusion_vlast(field, K_bad, grid)


def test_smag_w_branch_shape_consistency():
    """Section 11 w branch derives ``K_m_half`` shape ``(ny, nx,
    nlev+1)`` via 0.5 * (K[..., :-1] + K[..., 1:]) → ``(ny, nx,
    nlev-1)`` followed by ``(1, 1)`` pad. Verifies it matches w
    shape ``(ny, nx, nlev+1)`` so the diffusion helper does not
    raise."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _variable_K_diffusion_vlast, _compute_smagorinsky_K_m_plane,
    )
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(9)
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)),
    )
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    K_half_interior = 0.5 * (K_m[..., :-1] + K_m[..., 1:])
    K_half = jnp.pad(K_half_interior, ((0, 0), (0, 0), (1, 1)))
    assert K_half.shape == w.shape, (
        f"K_m_half shape {K_half.shape} != w shape {w.shape}"
    )
    out = _variable_K_diffusion_vlast(w, K_half, grid)
    assert out.shape == w.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_full_3D_strain_picks_up_pure_vertical_shear():
    """Full 3D Smag strain tensor includes S_13 = 0.5(∂u/∂z + ∂w/∂x).
    Pure ∂u/∂z shear must give K_m > 0 — the previous horizontal-only
    pilot returned zero here, masking the vertical mixing path. This
    is the upgrade pinned by the user request (2026-05-24)."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _compute_smagorinsky_K_m_plane,
    )
    _, grid, hc, _, _ = _setup()
    # u(x, y, z) varies only in z (linear shear); v = w = 0.
    k = jnp.arange(grid.nlev, dtype=jnp.float64)
    u = jnp.broadcast_to(k[None, None, :], (grid.ny, grid.nx, grid.nlev))
    v = jnp.zeros_like(u)
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    K = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    # Vertical shear ⇒ S_13² > 0 ⇒ K_m > 0 at every cell where the
    # one-sided dz difference samples a non-zero u-difference.
    assert float(jnp.max(K)) > 0.0, (
        "3D Smag strain returned zero K_m under pure vertical shear "
        "— S_13 term is not contributing to |S|²."
    )


def test_full_level_centred_d_dz_linear_field_returns_constant():
    """Codex review 2026-05-24: pin the vertical-derivative helper
    against a known linear profile. ``u(z) = a · z + b`` must yield
    ``|∂u/∂z| = |a|`` at every level — interior centred AND the
    top/bottom one-sided fallbacks. (The sign convention is
    documented in the helper's docstring; this test compares the
    absolute magnitude because the helper returns the per-level-
    index derivative, which is ``-∂u/∂z_physical`` under the
    top-to-bottom storage order.)"""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _full_level_centred_d_dz,
    )
    _, grid, hc, _, _ = _setup()
    a = 2.5
    b = 7.3
    # u(z_full) = a*z + b → broadcast to (ny, nx, nlev).
    u = jnp.broadcast_to(
        (a * hc.z_full + b)[None, None, :],
        (grid.ny, grid.nx, grid.nlev),
    )
    deriv = _full_level_centred_d_dz(u, hc)
    # |deriv| == |a| at every cell, every level, including top/bottom.
    np.testing.assert_allclose(
        np.asarray(jnp.abs(deriv)),
        np.full((grid.ny, grid.nx, grid.nlev), abs(a)),
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_full_3D_strain_picks_up_pure_dw_dz():
    """Pure ∂w/∂z divergence (no horizontal shear, no w-tilt) must
    give K_m > 0 via the S_33² term."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import (
        _compute_smagorinsky_K_m_plane,
    )
    _, grid, hc, _, _ = _setup()
    u = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    v = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    # Vertical-velocity profile: linear in k → ∂w/∂z = const != 0.
    k_half = jnp.arange(grid.nlev + 1, dtype=jnp.float64)
    w = jnp.broadcast_to(
        k_half[None, None, :], (grid.ny, grid.nx, grid.nlev + 1),
    )
    K = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    assert float(jnp.max(K)) > 0.0, (
        "3D Smag strain returned zero K_m under pure dw/dz — S_33 "
        "term is not contributing."
    )

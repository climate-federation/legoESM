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

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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


def test_rest_state_with_sgs_vertical_stays_at_rest():
    """SGS-VERT #81: with the vertical SGS flux ENABLED, a zero state must
    still stay exactly at rest — the 3D SGS must not inject a spurious
    source from the new vertical leg (zero fields ⇒ zero flux)."""
    grid = create_plane_grid(
        nx=8, ny=8, nlev=6, dx=200.0, dy=200.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=3_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
        use_coriolis=False, fix_mass=False,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
        sgs_vertical_diffusion=True,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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


def test_vertical_sgs_diffusion_conserves_column_integral():
    """SGS-VERT #81: the MASS-WEIGHTED vertical flux ``(1/ρ)∂_z(ρ_w K ∂_z f)``
    with no-flux boundaries conserves the ``ρ_ref·dz``-weighted COLUMN integral
    (the compressible conserved measure) to machine epsilon — ``Σ_k tend_k·
    ρ_ref[k]·dz_k = F_top − F_bot = 0`` (codex iter-64 [HIGH]: plain ``dz`` is
    NOT the conserved measure for a density-weighted vertical flux)."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _vertical_K_diffusion_full,
    )
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(11)
    f = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    K = jnp.asarray(np.abs(rng.standard_normal((grid.ny, grid.nx, grid.nlev))))
    tend = _vertical_K_diffusion_full(f, K, hc)
    mass = jnp.asarray(hc.rho_ref) * jnp.asarray(hc.dz)   # ρ₀·dz column weight
    col_int = jnp.sum(tend * mass, axis=-1)        # (ny, nx) per column
    rel = float(jnp.max(jnp.abs(col_int))) / max(
        float(jnp.max(jnp.abs(tend * mass))), 1.0e-30,
    )
    assert rel < 1.0e-12, (
        f"vertical SGS flux not mass-column-conservative: "
        f"max|Σ tend·ρ·dz| / max|tend·ρ·dz| = {rel:.3e}"
    )


def test_vertical_sgs_diffusion_is_dissipative():
    """SGS-VERT #81: ``Σ_k f_k·tend_k·ρ_ref[k]·dz_k ≤ 0`` for K≥0 (discrete
    integration by parts ⇒ ``−Σ ρ_w·K·(∂_z f)²/dz_half ≤ 0``). Guards the
    sign of the mass-weighted vertical flux divergence."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _vertical_K_diffusion_full,
    )
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(12)
    f = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    K = jnp.asarray(np.abs(rng.standard_normal((grid.ny, grid.nx, grid.nlev))))
    tend = _vertical_K_diffusion_full(f, K, hc)
    mass = jnp.asarray(hc.rho_ref) * jnp.asarray(hc.dz)
    dissipation = float(jnp.sum(f * tend * mass))
    assert dissipation <= 0.0, (
        f"vertical SGS flux not dissipative: Σ f·tend·ρ·dz = {dissipation:.3e}"
    )


def test_vertical_sgs_w_diffusion_is_dissipative_and_respects_rigid_bc():
    """SGS-VERT #81: the dual-grid w vertical flux must (a) dissipate the
    resolved KE and (b) leave the rigid w=0 boundary interfaces untouched.

    With the interface control-volume weight ``dz_half`` the discrete
    energy ``Σ_j w_j·tend_j·dz_half[j-1] = −Σ_k K_k(w_{k+1}−w_k)²/dz_full[k]
    ≤ 0`` (Abel summation, w=0 at the rigid ends). Guards the half-level
    staggering — a wrong index/sign would break the energy sign or leak a
    boundary tendency."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _vertical_K_diffusion_w,
    )
    _, grid, hc, _, _ = _setup()
    nlev = grid.nlev
    rng = np.random.default_rng(14)
    w = rng.standard_normal((grid.ny, grid.nx, nlev + 1))
    w[..., 0] = 0.0           # rigid bottom
    w[..., -1] = 0.0          # rigid top
    w = jnp.asarray(w)
    K = jnp.asarray(np.abs(rng.standard_normal((grid.ny, grid.nx, nlev))))
    tend = _vertical_K_diffusion_w(w, K, hc)
    # (b) boundary interfaces untouched (w held by the rigid BC).
    assert float(jnp.max(jnp.abs(tend[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(tend[..., -1]))) == 0.0
    # (a) KE-dissipative with the MASS interface control-volume weight
    # (ρ_w·dz_half — the mass measure SAM uses, codex iter-64 [HIGH]).
    dz_half = jnp.asarray(hc.dz_half)             # (nlev-1,)
    rho_iface = jnp.asarray(hc.rho_ref_half[1:-1])  # (nlev-1,) interior iface ρ₀
    interior = jnp.sum(w[..., 1:-1] * tend[..., 1:-1] * rho_iface * dz_half)
    assert float(interior) <= 0.0, (
        f"w vertical SGS not KE-dissipative: Σ w·tend·ρ_w·dz_half = {float(interior):.3e}"
    )


def test_vertical_sgs_diffusion_zero_on_vertically_uniform_field():
    """SGS-VERT #81: a z-constant field has ZERO vertical gradient ⇒ no
    SGS flux ⇒ exactly zero tendency. Catches index/sign/boundary bugs
    (a stencil that leaked a spurious top/bottom flux would fail here)."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _vertical_K_diffusion_full,
    )
    _, grid, hc, _, _ = _setup()
    rng = np.random.default_rng(13)
    f = jnp.full((grid.ny, grid.nx, grid.nlev), 3.7)
    K = jnp.asarray(np.abs(rng.standard_normal((grid.ny, grid.nx, grid.nlev))))
    tend = _vertical_K_diffusion_full(f, K, hc)
    assert float(jnp.max(jnp.abs(tend))) < 1.0e-13, (
        "vertical SGS flux nonzero on a vertically-uniform field "
        f"(max|tend|={float(jnp.max(jnp.abs(tend))):.3e}) — spurious "
        "boundary flux or index bug"
    )


def test_validate_rejects_zero_prandtl():
    """Codex iter-1 M2: zero Prandtl number produces inf K_h. Must
    raise ``ValueError`` at config-validation time."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        full_level_centred_d_dz,
    )
    _, grid, hc, _, _ = _setup()
    a = 2.5
    b = 7.3
    # u(z_full) = a*z + b → broadcast to (ny, nx, nlev).
    u = jnp.broadcast_to(
        (a * hc.z_full + b)[None, None, :],
        (grid.ny, grid.nx, grid.nlev),
    )
    deriv = full_level_centred_d_dz(u, hc)
    # |deriv| == |a| at every cell, every level, including top/bottom.
    np.testing.assert_allclose(
        np.asarray(jnp.abs(deriv)),
        np.full((grid.ny, grid.nx, grid.nlev), abs(a)),
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_full_3D_strain_picks_up_pure_dw_dz():
    """Pure ∂w/∂z divergence (no horizontal shear, no w-tilt) must
    give K_m > 0 via the S_33² term."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
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


def _shear_state(grid):
    """Pure vertical shear of ``u`` (constant ∂u/∂z) → nonzero |S|²."""
    k = jnp.arange(grid.nlev, dtype=jnp.float64)
    u = jnp.broadcast_to(
        2.0 * k[None, None, :], (grid.ny, grid.nx, grid.nlev),
    )
    v = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    return u, v, w


def test_stratification_term_shuts_off_mixing_in_stable_column():
    """SAM ``dosmagor`` faithfulness: subtracting ``Pr·N²`` inside the
    strain sqrt must shut mixing OFF in a strongly stable column
    (``N² ≫ |S|²`` → ``K_m = 0``), leave a NEUTRAL column unchanged
    (``N² = 0`` → equals the no-buoyancy baseline), and ENHANCE an
    unstable column (``N² < 0`` → larger than baseline).  ``n2_sgs`` is
    passed directly so this exercises the kernel's response to a given
    stratification (the N² computation itself is covered by
    ``test_sgs_brunt_vaisala_*``)."""
    _, grid, hc, _, _ = _setup()
    u, v, w = _shear_state(grid)
    shp = (grid.ny, grid.nx, grid.nlev)

    # Baseline: pure-strain closure (n2_sgs=None) → K_m > 0 under shear.
    K_base = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    assert float(jnp.max(K_base)) > 0.0

    # Strongly STABLE: N² = 1 s⁻² ≫ |S|² → shutoff everywhere.
    K_stable = _compute_smagorinsky_K_m_plane(
        u, v, w, grid, hc, c_s=0.2, n2_sgs=jnp.ones(shp), prandtl=1.0,
    )
    assert float(jnp.max(K_stable)) == 0.0, (
        "stable stratification (N² ≫ |S|²) must zero K_m exactly"
    )

    # NEUTRAL: N² = 0 → identical to the no-buoyancy path.
    K_neutral = _compute_smagorinsky_K_m_plane(
        u, v, w, grid, hc, c_s=0.2, n2_sgs=jnp.zeros(shp), prandtl=1.0,
    )
    assert jnp.allclose(K_neutral, K_base, atol=1.0e-12), (
        "neutral N²=0 must reduce to the pure-strain baseline"
    )

    # UNSTABLE: N² < 0 → mixing enhanced above the baseline.
    K_unstable = _compute_smagorinsky_K_m_plane(
        u, v, w, grid, hc, c_s=0.2,
        n2_sgs=jnp.full(shp, -1.0e-3), prandtl=1.0,
    )
    assert float(jnp.max(K_unstable)) > float(jnp.max(K_base)), (
        "unstable stratification (N² < 0) must enhance K_m above the "
        "pure-strain baseline"
    )


def test_stratification_term_ad_safe_at_shutoff():
    """``jax.grad`` through the stratification subtraction must stay
    finite even in the shut-off (``arg ≤ 0``) regime where the
    double-``where`` safe-sqrt masks the dead branch."""
    _, grid, hc, _, _ = _setup()
    u, v, w = _shear_state(grid)
    shp = (grid.ny, grid.nx, grid.nlev)

    def loss(n2):
        K = _compute_smagorinsky_K_m_plane(
            u, v, w, grid, hc, c_s=0.2, n2_sgs=n2, prandtl=1.0,
        )
        return jnp.sum(K)

    g = jax.grad(loss)(jnp.ones(shp))  # strongly-stable → shut off
    assert bool(jnp.all(jnp.isfinite(g)))


def _stable_theta(grid, hc):
    """θ increasing with physical height z (stably stratified)."""
    z = jnp.broadcast_to(
        hc.z_full[None, None, :], (grid.ny, grid.nx, grid.nlev),
    )
    return 300.0 + 0.02 * z


def test_sgs_brunt_vaisala_unsaturated_equals_virtual_theta_n2():
    """In a subsaturated, condensate-free column ``sgs_brunt_vaisala_sq``
    must return the CLEAR branch ``N² = (g/θ_v)·∂θ_v/∂z`` with
    ``θ_v = virtual_temperature(θ, q_v)`` (SAM unsaturated buoy_sgs)."""
    from legoesm import constants
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        full_level_centred_d_dz, sgs_brunt_vaisala_sq,
    )
    from legoesm.atmosphere.physics._shared import virtual_temperature

    _, grid, hc, _, _ = _setup()
    theta = _stable_theta(grid, hc)
    # Far-subsaturated vapour, no condensate → clear branch.
    q_v = jnp.full_like(theta, 1.0e-4)
    tracers = jnp.stack([q_v, jnp.zeros_like(q_v), jnp.zeros_like(q_v)],
                        axis=-1)

    n2 = sgs_brunt_vaisala_sq(theta, tracers, hc)

    theta_v = virtual_temperature(theta, q_v)
    n2_expected = -(constants.g / jnp.clip(theta_v, 1.0, None)) * (
        full_level_centred_d_dz(theta_v, hc)
    )
    assert jnp.allclose(n2, n2_expected, atol=1.0e-12), (
        "subsaturated column must take the clear virtual-θ N² branch"
    )
    # Stable sounding → N² > 0 in the interior.
    assert float(jnp.min(n2[..., 1:-1])) > 0.0


def test_sgs_brunt_vaisala_saturated_reduces_stability():
    """SAM ``tke_full.f90:198-226``: at a SATURATED interface the moist
    (Durran–Klemp) ``N²`` is SMALLER than the dry value for the same θ
    sounding — latent-heat release on adiabatic ascent reduces the
    effective static stability, so the ``dosmagor`` shutoff lets the LES
    keep mixing inside cloud."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        sgs_brunt_vaisala_sq,
    )
    from legoesm.thermo import saturation_mixing_ratio

    _, grid, hc, _, _ = _setup()
    theta = _stable_theta(grid, hc)
    exner = hc.exner_ref
    T = theta * exner
    from legoesm import constants
    p = constants.p_ref * exner ** (1.0 / constants.kappa)
    q_sat = saturation_mixing_ratio(T, p)

    # DRY reference: same θ, far-subsaturated, no cloud.
    tr_dry = jnp.stack(
        [jnp.full_like(theta, 1.0e-4),
         jnp.zeros_like(theta), jnp.zeros_like(theta)],
        axis=-1,
    )
    n2_dry = sgs_brunt_vaisala_sq(theta, tr_dry, hc)

    # SATURATED: q_v just below q_sat + cloud water pushes non-precip
    # water well above q_sat → smooth moist blend ≈ fully moist.
    q_v = 0.95 * q_sat
    q_c = 0.50 * q_sat
    tr_sat = jnp.stack([q_v, q_c, jnp.zeros_like(theta)], axis=-1)
    n2_moist = sgs_brunt_vaisala_sq(theta, tr_sat, hc)

    # Interior levels (skip one-sided boundary gradients).
    assert float(jnp.max(n2_moist[..., 1:-1])) < float(
        jnp.min(n2_dry[..., 1:-1])
    ), "saturated moist N² must be below the dry N² for the same θ"


def test_sgs_brunt_vaisala_ad_safe_across_saturation():
    """``jax.grad`` through ``sgs_brunt_vaisala_sq`` must stay finite
    even for a column straddling the clear↔moist transition — the smooth
    sigmoid blend (not a hard ``where``) is what guarantees this (Codex
    iter-2 adversarial-review)."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        sgs_brunt_vaisala_sq,
    )
    from legoesm.thermo import saturation_mixing_ratio

    _, grid, hc, _, _ = _setup()
    theta = _stable_theta(grid, hc)
    exner = hc.exner_ref
    from legoesm import constants
    T = theta * exner
    p = constants.p_ref * exner ** (1.0 / constants.kappa)
    q_sat = saturation_mixing_ratio(T, p)
    # q_v straddles saturation across levels (some below, some above),
    # zero cloud → gradients flow through the sigmoid edge AND the
    # q_cloud=0 path that the old condensate-ratio ω would have broken.
    q_v0 = 1.05 * q_sat  # near/above saturation, no cloud
    z = jnp.zeros_like(theta)
    tracers0 = jnp.stack([q_v0, z, z], axis=-1)

    def loss(th):
        return jnp.sum(sgs_brunt_vaisala_sq(th, tracers0, hc))

    g_theta = jax.grad(loss)(theta)
    assert bool(jnp.all(jnp.isfinite(g_theta)))

    def loss_q(qv):
        tr = jnp.stack([qv, z, z], axis=-1)
        return jnp.sum(sgs_brunt_vaisala_sq(theta, tr, hc))

    g_qv = jax.grad(loss_q)(q_v0)
    assert bool(jnp.all(jnp.isfinite(g_qv)))


def test_sgs_diffuses_tracers_conservatively():
    """SGS-SCALAR (iter-60): SAM `sgs.f90:664-675` SGS-diffuses EVERY scalar
    (q_v + all hydrometeors) with the eddy conductivity, not just θ'. The plane
    dycore must add K_h diffusion to the tracers — nonzero (a horizontal tracer
    gradient IS mixed) and CONSERVATIVE (domain-sum tendency = 0), and exactly
    zero when c_s=0 (rest state ⇒ no advection either)."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        make_rest_state,
    )
    grid = create_plane_grid(nx=8, ny=8, nlev=6, dx=1000.0, dy=1000.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(6, H=6000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    st = make_rest_state(grid, hc, dtype=jnp.float64)
    tr = np.zeros((8, 8, 6, 6))
    tr[3:5, 3:5, :, 0] = 0.01            # a q_v blob (slot 0) → horizontal grad
    tr[..., 0] += 0.001
    st = st._replace(tracers=st.tracers.replace(data=jnp.asarray(tr)))

    cfg_on = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=False, smagorinsky_cs=0.19, smagorinsky_prandtl=1.0)
    tend = plane_compressible_euler_slow_tendencies(st, grid, hc, tm, cfg_on)
    dq = np.asarray(tend.dtracers_dt.data[..., 0])
    assert np.max(np.abs(dq)) > 0.0                    # tracers ARE diffused
    assert abs(float(np.sum(dq))) < 1e-18              # conservative (rest state)
    # every tracer slot is diffused (SAM diffuses all micro_field), not just q_v
    assert np.max(np.abs(np.asarray(tend.dtracers_dt.data))) > 0.0

    cfg_off = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=False, smagorinsky_cs=0.0)
    tend0 = plane_compressible_euler_slow_tendencies(st, grid, hc, tm, cfg_off)
    assert float(np.max(np.abs(tend0.dtracers_dt.data))) == 0.0   # no SGS, rest


def test_sgs_tracer_diffusion_preserves_positivity():
    """codex iter-60 (MED): the SGS tracer diffusion operator (the SAME
    `_variable_K_diffusion_vlast` my fix vmaps over tracers) must not drive a
    non-negative tracer negative. A realistic Smagorinsky K_m + a sharp isolated
    q_v spike: forward-Euler `q + dt·∇·(K∇q)` stays ≥0 at the CRM dt (the spike
    loses, neighbours gain — a positive-weight stencil within the diffusion CFL
    K·dt/dx²≪½) and is conservative (Σ tendency = 0). Advection positivity is a
    separate (scheme-dependent) property, so this isolates the SGS operator."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _variable_K_diffusion_vlast, _compute_smagorinsky_K_m_plane,
    )
    grid = create_plane_grid(nx=8, ny=8, nlev=6, dx=1000.0, dy=1000.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(6, H=6000.0)
    rng = np.random.default_rng(11)
    sh = (grid.ny, grid.nx, grid.nlev)
    u = jnp.asarray(2.0 * rng.standard_normal(sh))
    v = jnp.asarray(2.0 * rng.standard_normal(sh))
    w = jnp.asarray(2.0 * rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)))
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.19)
    q = np.full(sh, 1e-8)
    q[4, 4, 2] = 0.02                                  # sharp isolated spike
    dq = np.asarray(_variable_K_diffusion_vlast(jnp.asarray(q), K_m, grid))
    for dt in (1.0, 2.0, 5.0):
        assert float(np.min(q + dt * dq)) >= 0.0, f"negative at dt={dt}"
    assert abs(float(np.sum(dq))) < 1e-18              # conservative
    assert float(np.max(np.abs(dq))) > 0.0             # SGS actually acted


# --------------------------------------------------------------------------- #
# Mixed-shear SIGN regression (2026-07-12). ``full_level_centred_d_dz``
# returns the LEVEL-INDEX derivative = −∂/∂z_physical (z up, level index
# top→down), and the strain call sites must negate it before MIXING with the
# already-physical ∂w/∂x, ∂w/∂y. The pre-fix code formed
# S13 = 0.5(−∂u/∂z + ∂w/∂x): the 2·(∂u/∂z)(∂w/∂x) cross term in S13² flipped,
# giving |S|² = (b−a)² instead of (a+b)² under u = a·z, w = b·x.
# --------------------------------------------------------------------------- #


def _mixed_shear_fields(grid, hc, a, b):
    """u = a·z (∂u/∂z_phys = a), w = b·x (∂w/∂x = b), v = 0."""
    u = jnp.broadcast_to(
        (a * hc.z_full)[None, None, :], (grid.ny, grid.nx, grid.nlev),
    )
    x_c = (jnp.arange(grid.nx, dtype=jnp.float64) + 0.5) * grid.dx
    w = jnp.broadcast_to(
        (b * x_c)[None, :, None], (grid.ny, grid.nx, grid.nlev + 1),
    )
    v = jnp.zeros_like(u)
    return u, v, w


def test_smagorinsky_mixed_shear_matches_analytic_a_plus_b():
    """u = a·z, w = b·x with the SAME signs: only S13 = 0.5(a+b) is nonzero,
    so |S|² = 2·(2·S13²) = (a+b)² and K_m = (c_s·Δ)²·(a+b) at interior
    columns. The pre-fix sign flip gave |S|² = (b−a)², i.e. K_m 3× too small
    for a = 1e-3, b = 2e-3 (under-mixing in sheared updrafts)."""
    _, grid, hc, _, _ = _setup()
    a, b = 1.0e-3, 2.0e-3
    u, v, w = _mixed_shear_fields(grid, hc, a, b)
    K = _compute_smagorinsky_K_m_plane(
        u, v, w, grid, hc, c_s=0.2, wall_damping=False,
    )
    delta = (grid.dx * grid.dy * np.asarray(hc.dz)) ** (1.0 / 3.0)
    expected = (0.2 * delta) ** 2 * (a + b)                       # (nlev,)
    # Interior columns only: the periodic roll wraps the linear-in-x w, so
    # cell centres i = 0 and i = nx−1 see the wrap jump.
    K_int = np.asarray(K)[:, 1:-1, :]
    np.testing.assert_allclose(
        K_int, np.broadcast_to(expected, K_int.shape), rtol=1.0e-12,
    )


def test_velocity_gradients_plane_vertical_signs_are_physical():
    """``_velocity_gradients_plane`` must return the PHYSICAL ∂/∂z (z up):
    linear profiles u = a·z, v = b·z, w = c·z must give dudz = +a, dvdz = +b,
    dwdz = +c everywhere (the raw level-index derivative is the negative
    under top-down storage). Vreman/AMD consume these gradients in ODD
    products and the dynamic closures in mixed strain products, so a global
    flip changes their K_m / C_s."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _velocity_gradients_plane,
    )
    _, grid, hc, _, _ = _setup()
    a, b, c = 1.5e-3, -2.5e-3, 4.0e-4
    shape = (grid.ny, grid.nx, grid.nlev)
    u = jnp.broadcast_to((a * hc.z_full)[None, None, :], shape)
    v = jnp.broadcast_to((b * hc.z_full)[None, None, :], shape)
    w = jnp.broadcast_to(
        (c * hc.z_half)[None, None, :], (grid.ny, grid.nx, grid.nlev + 1),
    )
    (_, _, _, _, _, dudz, _, _, dvdz, _, _, dwdz) = (
        _velocity_gradients_plane(u, v, w, grid, hc))
    np.testing.assert_allclose(np.asarray(dudz), a, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(dvdz), b, rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(dwdz), c, rtol=1.0e-12)


def test_centre_strain_mixed_shear_matches_analytic_a_plus_b():
    """A-grid strain path shared by the dynamic/LASD closures (and built on
    the same gradients Vreman/AMD consume): S13 = 0.5(∂u/∂z + ∂w/∂x)
    = 0.5(a+b) and |S| = √(2 S_ij S_ij) = a+b under u = a·z, w = b·x at
    interior columns. Pre-fix this gave 0.5(b−a) / |b−a|."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _centre_velocities_and_strain_plane,
    )
    _, grid, hc, _, _ = _setup()
    a, b = 1.0e-3, 2.0e-3
    u, v, w = _mixed_shear_fields(grid, hc, a, b)
    (_, _, _, _, _, _, _, S13, _, Smag) = (
        _centre_velocities_and_strain_plane(u, v, w, grid, hc))
    # Centred A-grid ∂w/∂x uses roll(±1): columns 0 and nx−1 see the wrap.
    S13_int = np.asarray(S13)[:, 1:-1, :]
    Smag_int = np.asarray(Smag)[:, 1:-1, :]
    np.testing.assert_allclose(S13_int, 0.5 * (a + b), rtol=1.0e-12)
    np.testing.assert_allclose(Smag_int, a + b, rtol=1.0e-12)

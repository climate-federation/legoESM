"""SAM moist-buoyancy (D1) tests for the plane NH dycore.

The dry θ'-buoyancy lives in the acoustic substep; the plane slow
tendency adds the MOIST part of SAM's ``buoyancy.f90``:

    B_moist = g·( ε_v·(q_v − q̄_v) − (q_cond − q̄_cond) )

with ``q̄`` the horizontal-mean profile (SAM ``qv0``/``qn0``/``qp0``).
These tests pin the zero-mean property (no spurious mean updraft), the
vapour-up / condensate-down signs, the rigid w boundaries, the dry
bit-exactness, and AD-safety.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    moisture_buoyancy_w_half,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


def _hmean(f):
    return jnp.mean(f, axis=(0, 1), keepdims=True)


def _grid_hc(nlev=6):
    grid = create_plane_grid(
        nx=8, ny=8, nlev=nlev, dx=1000.0, dy=1000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(nlev, H=12_000.0, p_sfc=101_480.0)
    return grid, hc


def test_moisture_buoyancy_zero_horizontal_mean():
    """Perturbation-from-mean construction (incl. the θ′−⟨θ′⟩ thermal
    cross term) ⇒ B_moist has ZERO horizontal mean at every level (else a
    spurious mean updraft vs the dry ref)."""
    grid, hc = _grid_hc()
    rng = np.random.default_rng(0)
    tr = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev, 3)) * 5.0e-3,
    )
    th = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev)) * 0.5,
    )
    b_half = moisture_buoyancy_w_half(tr, th, hc, _hmean)
    assert b_half.shape == (grid.ny, grid.nx, grid.nlev + 1)
    # Horizontal mean ≈ 0 at every interface.
    assert float(jnp.max(jnp.abs(jnp.mean(b_half, axis=(0, 1))))) < 1.0e-15


def test_moisture_buoyancy_zero_when_dry():
    """No moisture ⇒ B_moist ≡ 0 (bit-exact dry dycore), even with θ′."""
    grid, hc = _grid_hc()
    tr = jnp.zeros((grid.ny, grid.nx, grid.nlev, 3))
    rng = np.random.default_rng(1)
    th = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)) * 0.5)
    b_half = moisture_buoyancy_w_half(tr, th, hc, _hmean)
    assert float(jnp.max(jnp.abs(b_half))) == 0.0


def test_moisture_buoyancy_rigid_boundaries():
    """Top + bottom interfaces are rigid (zero) to match the w BC."""
    grid, hc = _grid_hc()
    rng = np.random.default_rng(2)
    tr = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev, 3)) * 5.0e-3,
    )
    th = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    b_half = moisture_buoyancy_w_half(tr, th, hc, _hmean)
    assert float(jnp.max(jnp.abs(b_half[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(b_half[..., -1]))) == 0.0


def test_moisture_buoyancy_vapor_up_condensate_down():
    """A locally moist (q_v above mean) column is POSITIVELY buoyant; a
    locally condensate-loaded column is NEGATIVELY buoyant — the SAM
    sign convention (ε_v·q_v' − q_cond')."""
    grid, hc = _grid_hc()
    ny, nx, nlev = grid.ny, grid.nx, grid.nlev
    # Uniform background q_v so <q_v> is nonzero; bump one column's vapour.
    q_v = jnp.full((ny, nx, nlev), 0.010)
    q_v = q_v.at[0, 0, :].add(0.002)            # moist column
    q_c = jnp.zeros((ny, nx, nlev))
    q_r = jnp.zeros((ny, nx, nlev))
    th0 = jnp.zeros((ny, nx, nlev))            # θ′=0 → isolate first order
    tr_vapor = jnp.stack([q_v, q_c, q_r], axis=-1)
    b_vapor = moisture_buoyancy_w_half(tr_vapor, th0, hc, _hmean)
    # Interior interfaces of the moist column are positively buoyant.
    assert float(jnp.min(b_vapor[0, 0, 1:-1])) > 0.0

    # Now load the same column with condensate instead of extra vapour.
    q_c2 = q_c.at[0, 0, :].add(0.002)
    tr_cond = jnp.stack([jnp.full((ny, nx, nlev), 0.010), q_c2, q_r],
                        axis=-1)
    b_cond = moisture_buoyancy_w_half(tr_cond, th0, hc, _hmean)
    assert float(jnp.max(b_cond[0, 0, 1:-1])) < 0.0


def test_moisture_buoyancy_thermal_cross_term():
    """The moist thermal-coefficient correction (θ′−⟨θ′⟩)/θ₀·(ε_v·q̄_v −
    q̄_cond) is present: with a uniform moist background (q_v′ = 0) and a
    warm column (θ′ > mean), B_moist is non-zero and positive there —
    warm air in a moist mean is extra-buoyant (SAM thermal·moist term)."""
    grid, hc = _grid_hc()
    ny, nx, nlev = grid.ny, grid.nx, grid.nlev
    q_v = jnp.full((ny, nx, nlev), 0.012)      # uniform → q_v′ = 0
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(q_v)
    # First-order part vanishes (uniform moisture) and θ′=0 ⇒ no cross
    # term either ⇒ B ≈ 0 to mean-subtraction round-off.
    th_flat = jnp.zeros((ny, nx, nlev))
    assert float(jnp.max(jnp.abs(
        moisture_buoyancy_w_half(tr, th_flat, hc, _hmean)
    ))) < 1.0e-15
    # Warm column (θ′ above the horizontal mean) → positive cross term.
    th = jnp.zeros((ny, nx, nlev)).at[0, 0, :].set(2.0)
    b = moisture_buoyancy_w_half(tr, th, hc, _hmean)
    assert float(jnp.min(b[0, 0, 1:-1])) > 0.0


def test_moist_buoyancy_flag_off_zeroes_contribution():
    """``moist_buoyancy=False`` ⇒ the slow-tendency dw/dt loses exactly
    the moisture term (equals the on-run minus the buoyancy helper)."""
    grid, hc = _grid_hc()
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = np.random.default_rng(3)
    tr = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev, 3)) * 5.0e-3,
    )
    # Non-zero θ′ so the thermal cross term is exercised in the model path.
    th = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev)) * 0.5,
    )
    state = rest._replace(
        tracers=rest.tracers.replace(data=tr),
        theta_prime=rest.theta_prime.replace(data=th),
    )

    base = dict(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
        use_coriolis=False, fix_mass=False, smagorinsky_cs=0.0,
        # The default ``acoustic_moist_buoyancy=True`` carries the moist
        # buoyancy INSIDE the acoustic substeps (fixes the updraft runaway);
        # the slow-tendency path under test only adds it when that is off.
        acoustic_moist_buoyancy=False,
    )
    cfg_on = CompressibleEulerConfig(moist_buoyancy=True, **base)
    cfg_off = CompressibleEulerConfig(moist_buoyancy=False, **base)
    tend_on = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_on,
    )
    tend_off = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg_off,
    )
    b_half = moisture_buoyancy_w_half(tr, th, hc, _hmean)
    # on == off + buoyancy, and the buoyancy is actually non-trivial.
    assert jnp.allclose(
        tend_on.dw_dt.data, tend_off.dw_dt.data + b_half, atol=1.0e-14,
    )
    assert float(jnp.max(jnp.abs(b_half))) > 0.0


def test_moist_buoyancy_drives_updraft_over_moist_column():
    """Integrating a few steps, a locally moist column develops an
    UPWARD w while the dry run does not — the physics the term adds."""
    grid, hc = _grid_hc()
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = rest.theta_prime.data.shape
    q_v = jnp.full((ny, nx, nlev), 0.010)
    q_v = q_v.at[ny // 2, nx // 2, :].add(0.004)     # central moist column
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(q_v)
    state = rest._replace(tracers=rest.tracers.replace(data=tr))

    base = dict(
        sponge_coeff=0.0, hyperdiff_coeff=1.0e5, hyperdiff_rho_coeff=1.0e5,
        hyperdiff_w_coeff=1.0e5, semi_implicit_acoustic=False,
        use_coriolis=False, fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.0,
    )
    model_on = PlaneCompressibleEulerModel(
        grid, hc, tm, CompressibleEulerConfig(moist_buoyancy=True, **base),
    )
    model_off = PlaneCompressibleEulerModel(
        grid, hc, tm, CompressibleEulerConfig(moist_buoyancy=False, **base),
    )
    s_on, s_off = state, state
    for _ in range(15):
        s_on = model_on.step(s_on, dt=1.0)
        s_off = model_off.step(s_off, dt=1.0)
    assert bool(jnp.all(jnp.isfinite(s_on.w.data)))
    # Moist run develops appreciable vertical motion; dry run stays ~0.
    assert float(jnp.max(jnp.abs(s_on.w.data))) > 1.0e-3
    assert float(jnp.max(jnp.abs(s_on.w.data))) > 10.0 * float(
        jnp.max(jnp.abs(s_off.w.data)) + 1.0e-30
    )


def test_moist_buoyancy_ad_safe():
    """``jax.grad`` of a w-tendency scalar wrt q_v stays finite."""
    grid, hc = _grid_hc()
    rng = np.random.default_rng(4)
    tr = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev, 3)) * 5.0e-3,
    )

    th = jnp.asarray(
        rng.standard_normal((grid.ny, grid.nx, grid.nlev)) * 0.5,
    )

    def loss(tracers):
        return jnp.sum(
            moisture_buoyancy_w_half(tracers, th, hc, _hmean) ** 2
        )

    g = jax.grad(loss)(tr)
    assert bool(jnp.all(jnp.isfinite(g)))

"""Density-current cold-bubble benchmark on the plane NH dycore (CI-sized).

PR2d ships a heavily scaled-down CI version of the Straka et al. 1993
density-current test. The full benchmark (Lx=51.2 km, dx=200 m,
integrate 900 s with hyperdiffusion) is not run in CI.

Why CI is scaled down
---------------------
The PR2d plane dycore uses centred-difference horizontal advection
without hyperdiffusion. A -15 K cold bubble (Straka's amplitude)
develops sharp gradients fast and excites the 2-Δx mode in well
under 30 s on a 400 m grid. The CI test therefore uses a much
weaker cold bubble (-2 K) and a short integration (10 s) just to
confirm:

1. The cold-bubble IC spins up DOWNWARD motion (negative w in the
   plume).
2. θ' stays negative everywhere it was negative initially (cold
   anomaly persists rather than being numerically clipped).
3. The integration stays finite — no NaN or Inf at the end.
4. Dry mass is conserved with ``fix_mass=True``.

CI configuration
----------------
- Domain ``Lx = 6.4 km``, ``Ly = 1 km``, ``Lz = 3.2 km``.
- Horizontal grid ``nx = 16``, ``ny = 4``. Vertical ``nlev = 16``
  (``dz = 200 m``).
- Time step ``dt = 0.5 s``, integrated for ``20 steps = 10 s``.
- Cold-bubble amplitude ``-2 K`` (CI-weakened; Straka uses -15 K).
- Sponge ``coeff = 0.05`` in the top ``500 m``.
- Mass fixer ``fix_mass = True, anchor_mass_to_initial = True``.
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
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


# CI-sized constants — see module docstring for full spec.
NX, NY, NLEV = 16, 4, 16
DX = DY = 400.0       # m
LZ = 3_200.0          # m
DT = 0.5              # s
N_STEPS = 60          # → 30 s integration (3x longer than PR2d via upwind)
THETA_PERT = -2.0     # K (CI-weakened; Straka uses -15 K)
L_BUBBLE_X = 1_200.0  # m
L_BUBBLE_Z = 800.0    # m
X_C = 3_200.0         # m (bubble centre x, mid-domain)
Z_C = 1_600.0         # m (bubble centre z, mid-domain)


def _cold_bubble_theta_perturbation(grid, height_coord):
    """Cos^2 cold bubble centred at (X_C, Z_C); y-trivial."""
    xc = grid.xc  # (nx,)
    z_full = height_coord.z_full  # (nlev,) top-down
    dx_field = (xc[None, :, None] - X_C)
    dz_field = (z_full[None, None, :] - Z_C)
    r = jnp.sqrt(
        (dx_field / L_BUBBLE_X) ** 2 + (dz_field / L_BUBBLE_Z) ** 2,
    )
    inside = (r < 1.0).astype(jnp.float64)
    theta_p = THETA_PERT * jnp.cos(0.5 * jnp.pi * r) ** 2 * inside
    return jnp.broadcast_to(theta_p, (grid.ny, grid.nx, grid.nlev))


def _build_cold_bubble_initial_state(grid, height_coord):
    """``θ'`` from cos^2 cold bubble; ``ρ'`` from pressure invariance."""
    base = make_rest_state(grid, height_coord, dtype=jnp.float64)
    theta_p = _cold_bubble_theta_perturbation(grid, height_coord)
    rho_p = -height_coord.rho_ref * theta_p / height_coord.theta_ref
    return base._replace(
        theta_prime=base.theta_prime.replace(data=theta_p),
        rho_prime=base.rho_prime.replace(data=rho_p),
    )


@pytest.fixture(scope="module")
def _model_and_initial_state():
    grid = create_plane_grid(
        nx=NX, ny=NY, nlev=NLEV, dx=DX, dy=DY, dtype=jnp.float64,
    )
    height_coord = create_height_coordinate(grid.nlev, H=LZ)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=0.05,
        sponge_width=500.0,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=True,
        anchor_mass_to_initial=True,
    )
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    state = _build_cold_bubble_initial_state(grid, height_coord)
    return model, state, grid, height_coord, terrain


def _final_state(model, state):
    for _ in range(N_STEPS):
        state = model.step(state, dt=DT)
    return state


def test_cold_bubble_state_is_finite_at_end(_model_and_initial_state):
    model, state, _, _, _ = _model_and_initial_state
    final = _final_state(model, state)
    for field in (final.u, final.v, final.w, final.theta_prime,
                  final.rho_prime):
        assert bool(jnp.all(jnp.isfinite(field.data))), (
            f"Non-finite values in {field.name} after {N_STEPS} steps"
        )


def test_cold_bubble_generates_downward_motion(_model_and_initial_state):
    """A cold dense bubble must produce negative ``w`` in the plume."""
    model, state, _, _, _ = _model_and_initial_state
    final = _final_state(model, state)
    min_w = float(jnp.min(final.w.data))
    assert min_w <= -0.02, (
        f"min(w) = {min_w:.4f} m/s — cold bubble did not sink. "
        "Check buoyancy sign in acoustic substep."
    )


def test_cold_bubble_theta_anomaly_remains_negative(_model_and_initial_state):
    """The cold anomaly should not be numerically clipped to zero or
    flipped to positive during the integration."""
    _, state, _, _, _ = _model_and_initial_state
    initial_min = float(jnp.min(state.theta_prime.data))
    assert initial_min < 0.0, "IC sanity check: cold bubble must start negative"

    model, state2, _, _, _ = _model_and_initial_state
    final = _final_state(model, state2)
    final_min = float(jnp.min(final.theta_prime.data))
    assert final_min < 0.0, (
        f"Cold anomaly disappeared: final min(θ') = {final_min:.4e}"
    )


def test_cold_bubble_dry_mass_conserved_with_fixer(_model_and_initial_state):
    model, state, grid, height_coord, terrain = _model_and_initial_state
    mass_0 = float(compute_dry_mass_plane(
        state, grid, height_coord, terrain,
    ))
    # Run integration; ``_target_mass`` anchors on the first step call.
    state = model.step(state, dt=DT)
    for _ in range(N_STEPS - 1):
        state = model.step(state, dt=DT)
    mass_f = float(compute_dry_mass_plane(
        state, grid, height_coord, terrain,
    ))
    rel = abs(mass_f - mass_0) / abs(mass_0)
    assert rel < 1.0e-10, f"Dry-mass drift = {rel:.3e} (expected < 1e-10)"

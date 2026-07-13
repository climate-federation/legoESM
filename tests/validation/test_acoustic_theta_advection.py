"""Monotone (van-Leer) vertical theta' advection in the acoustic substep (iter-200).

The split-explicit acoustic substep updates theta' by the vertical w-advection of
``theta_total``. The legacy scheme is a CENTRED difference (non-monotone) which
overshoots at the SHARP tropopause theta-gradient -> a dispersive COLD DRIFT of
the cold-point over long RCE runs. ``acoustic_theta_advection="van_leer"`` swaps
in a monotone van-Leer TVD scheme (SAM advects scalars monotonically). These
tests pin:

* the new layout-agnostic kernel helper reproduces the already-tested plane
  van-Leer vertical advection bit-for-bit (correctness by equivalence);
* the van-Leer update introduces NO new extrema at a sharp theta jump (the TVD
  property = the cold-drift fix), while the centred update overshoots;
* the config default stays "centered" (byte-for-byte back-compat) and the
  "van_leer" path steps the full plane SI dycore finitely.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
    _theta_vert_advection_van_leer_kernel,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    vertical_advection_van_leer_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _setup(nx=4, ny=4, nlev=30, dx=2000.0, LZ=30000.0):
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             coriolis_mode="none")
    hc = create_height_coordinate(nlev, H=LZ)
    tm = make_flat_plane_terrain_metric(grid, hc)
    return grid, hc, tm


def test_kernel_helper_matches_plane_van_leer():
    """The layout-agnostic kernel helper == the tested plane van-Leer vertical
    advection (same flux-limiter logic, only ``J[..., None]`` vs ``J[:, :,
    None]`` broadcast) -> bit-for-bit identical for a (ny, nx) Jacobian."""
    _, hc, tm = _setup()
    nlev = hc.theta_ref.shape[0]
    ny, nx = 4, 4
    rng = np.random.default_rng(0)
    theta = jnp.asarray(rng.normal(size=(ny, nx, nlev)) * 5.0 + 300.0)
    w = jnp.asarray(rng.normal(size=(ny, nx, nlev + 1)) * 2.0)
    # rigid lids
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    J = tm.jacobian
    ref = vertical_advection_van_leer_plane(theta, w, hc, J)
    got = _theta_vert_advection_van_leer_kernel(theta, w, hc, J)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                               rtol=0.0, atol=0.0)


def test_van_leer_no_new_extrema_at_sharp_jump():
    """A monotone theta with a SHARP interior jump, advected one forward step by
    a uniform interior w: the van-Leer update creates NO new extrema (TVD), the
    centred update OVERSHOOTS outside the input range."""
    _, hc, _ = _setup(nlev=30, LZ=30000.0)
    nlev = hc.theta_ref.shape[0]
    # Sharp step in the interior (cold above index 14, warm below) -- mimics a
    # tropopause theta kink. Storage is top->surface (index 0 = top).
    theta_col = np.full(nlev, 360.0)
    theta_col[:15] = 200.0
    theta = jnp.asarray(theta_col).reshape(1, 1, nlev)
    w = jnp.zeros((1, 1, nlev + 1)).at[..., 1:-1].set(1.0)  # uniform interior
    J = jnp.ones((1, 1))
    dt = 5.0

    tend_vl = _theta_vert_advection_van_leer_kernel(theta, w, hc, J)
    new_vl = np.asarray(theta + dt * tend_vl).ravel()

    # Centred reference (the legacy kernel formula).
    dz_half = hc.dz_half
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    dz_c = dz_half[:-1] + dz_half[1:]
    inner = (theta[..., :-2] - theta[..., 2:]) / dz_c
    top = (theta[..., 0:1] - theta[..., 1:2]) / dz_half[0]
    bot = (theta[..., -2:-1] - theta[..., -1:]) / dz_half[-1]
    dtheta_dz = jnp.concatenate([top, inner, bot], axis=-1)
    tend_c = -w_full / J[..., None] * dtheta_dz
    new_c = np.asarray(theta + dt * tend_c).ravel()

    lo, hi = theta_col.min(), theta_col.max()
    tol = 1e-6
    # van-Leer: monotone -> stays within the input range.
    assert new_vl.max() <= hi + tol, f"van-Leer overshoot above: {new_vl.max()}"
    assert new_vl.min() >= lo - tol, f"van-Leer undershoot below: {new_vl.min()}"
    # Centred: dispersive -> overshoots outside the range at the jump.
    centred_overshoot = max(new_c.max() - hi, lo - new_c.min())
    assert centred_overshoot > 1e-3, (
        "expected the centred scheme to overshoot at the sharp jump; "
        f"got max overshoot {centred_overshoot}"
    )


def test_config_default_is_centered():
    """Byte-compat guard: the class default MUST stay 'centered' so every
    existing run/test is unchanged until a driver opts in to van_leer."""
    assert CompressibleEulerConfig().acoustic_theta_advection == "centered"


def test_acoustic_theta_van_leer_steps_finite():
    """The full plane SI dycore steps finitely with the van-Leer theta'
    acoustic advection (dispatch + compilation + a perturbed theta' that drives
    real vertical w through the substep)."""
    grid, hc, tm = _setup(nx=8, ny=8, nlev=30, dx=2000.0, LZ=30000.0)
    cfg = CompressibleEulerConfig(
        use_coriolis=False, smagorinsky_cs=0.0, sponge_coeff=0.0,
        hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
        acoustic_theta_advection="van_leer",
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Seed a theta' perturbation in the lowest cells so w develops.
    tp = state.theta_prime.data.at[..., -4:].add(
        jnp.asarray(np.random.default_rng(1).normal(
            size=state.theta_prime.data[..., -4:].shape) * 0.5))
    state = state._replace(theta_prime=state.theta_prime.replace(data=tp))
    for _ in range(5):
        state = model.step(state, dt=2.0)
    assert bool(jnp.all(jnp.isfinite(state.theta_prime.data)))
    assert bool(jnp.all(jnp.isfinite(state.w.data)))

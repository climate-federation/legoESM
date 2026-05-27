"""RCEMIP1 plane smoke test (10 steps).

Validates the PR4 RCE harness skeleton — not the full RCEMIP
equilibrium (which requires ~100 days of integration on a 100x100
km / 1 km grid; see ``scripts/run_rcemip_plane.py``). The CI test
just confirms the harness composes cleanly: dycore + Smag +
hyperdiff + sponge + tracer transport + bulk-flux + Newtonian
radiation physics_fn → stable, mass-conserving, no NaN over 10
steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add scripts/ to import path so the test can import make_rcemip_physics.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_rcemip_plane import (  # type: ignore
    _build_rcemip_initial_state,
    make_rcemip_physics,
    _rcemip_qv_profile,
    _rcemip_theta_profile,
)

from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
)
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, MicrophysicsConfig,
)
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, RadiationConfig,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


# iter-247 (Codex iter-243 round-1 HIGH#2): the original
# scripts/run_rcemip_plane.py:_make_rcemip_physics composed bulk
# surface fluxes + 5-day Newtonian relaxation toward 300 K + Kessler
# microphysics. The 0ec1da4b rename to make_rcemip_physics swapped
# Newtonian for the canonical gray-radiation factory. The iter-243
# minimal-rename fix passed radiation_config=None, silently dropping
# radiation coverage — restored here with the canonical replacement.
_RCEMIP_RADIATION_CFG = RadiationConfig(
    scheme="gray", gray=GrayRadiationConfig(),
)
_RCEMIP_MICROPHYSICS_CFG = MicrophysicsConfig(
    scheme="kessler", kessler=KesslerConfig(),
)


jax.config.update("jax_enable_x64", True)


def _setup_rcemip():
    grid = create_plane_grid(
        nx=8, ny=8, nlev=12, dx=4_000.0, dy=4_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=12_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=2_000.0,
        hyperdiff_coeff=1.0e6, hyperdiff_rho_coeff=1.0e6,
        hyperdiff_w_coeff=1.0e6,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
        fix_mass=True, anchor_mass_to_initial=True,
        use_coriolis=False,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = _build_rcemip_initial_state(grid, hc)
    return model, state, grid, hc, tm


def test_rcemip_profiles_match_wing_2018_values():
    """Wing 2018 Tab A1 expected values at specific altitudes.
    Codex iter-1 finding M1: pin numeric values, not just
    monotonicity (the original ``T_sfc * (1 + Γz)`` formula passed
    monotonicity tests but produced θ = 30450 K at 15 km)."""
    z = jnp.array([0.0, 4_000.0, 15_000.0, 20_000.0])
    theta = _rcemip_theta_profile(z, T_sfc=300.0)
    q_v = _rcemip_qv_profile(z, q_sfc=0.018)
    # θ_sfc = 300 K.
    assert float(theta[0]) == pytest.approx(300.0, rel=1.0e-12)
    # θ(4 km) = 300 + 6.7e-3 * 4000 = 326.8 K.
    assert float(theta[1]) == pytest.approx(326.8, rel=1.0e-6)
    # θ(15 km) = 300 + 6.7e-3 * 15000 = 400.5 K. Physical.
    assert float(theta[2]) == pytest.approx(400.5, rel=1.0e-6)
    # θ(20 km) = θ(15 km) constant tropopause cap.
    assert float(theta[3]) == pytest.approx(float(theta[2]))
    # q_v surface = 0.018; q_v(4 km) = 0.018 / e ≈ 6.62e-3.
    assert float(q_v[0]) == pytest.approx(0.018, rel=1.0e-12)
    import math
    assert float(q_v[1]) == pytest.approx(0.018 / math.e, rel=1.0e-6)
    # Stratosphere: q_v floored to 1e-9.
    assert float(q_v[3]) == pytest.approx(1.0e-9, rel=1.0e-12)


def test_rcemip_smoke_10_step_integration():
    """10-step integration must stay finite + mass-conservative."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))
    for _ in range(10):
        state = model.step(state, dt=2.0, physics_fn=physics_fn)
    # No NaN / Inf.
    for name, field in (
        ("u", state.u.data), ("w", state.w.data),
        ("theta_p", state.theta_prime.data),
        ("rho_p", state.rho_prime.data),
        ("tracers", state.tracers.data),
    ):
        assert bool(jnp.all(jnp.isfinite(field))), name
    # Mass conserved with anchor fixer.
    mass_f = float(compute_dry_mass_plane(state, grid, hc, tm))
    rel = abs(mass_f - mass0) / abs(mass0)
    assert rel < 1.0e-10, f"Dry mass drift = {rel:.3e}"


def test_rcemip_water_budget_positive_q_v_source():
    """Codex iter-1 minor: latent flux must increase domain-mean
    surface-layer q_v over one physics call; documents that water
    mass is NOT conserved (surface acts as a source) while dry mass
    is."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    tend = physics_fn(state, grid, hc, tm)
    # surface q_v tendency = lhflx / (rho L_v dz_sfc); for q_lo < q_sfc
    # this is positive.
    k_sfc = state.tracers.data.shape[-2] - 1
    surface_dq = tend.dtracers_dt.data[..., k_sfc, 0]
    mean_dq = float(jnp.mean(surface_dq))
    assert mean_dq > 0.0, (
        f"Surface latent flux tendency not positive: mean = {mean_dq:.3e}; "
        "should be > 0 because q_lo (initial) < q_sfc."
    )


def test_rcemip_physics_fn_is_differentiable():
    """Codex iter-1 finding M4: jax.grad through the bulk-flux +
    Newtonian physics must produce finite gradients. The sqrt in
    wind_speed has a 1 m/s floor so the kink at zero wind is avoided
    by construction; the Exner conversion is a constant scalar."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    # Scalar loss = sum(dtheta_prime_dt² + dtracers_dt²)
    # differentiated w.r.t. surface theta + q_v perturbations.
    k_sfc = state.theta_prime.data.shape[-1] - 1
    theta_sfc0 = state.theta_prime.data[..., k_sfc]
    qv_sfc0 = state.tracers.data[..., k_sfc, 0]

    def loss(theta_sfc, qv_sfc):
        s = state._replace(
            theta_prime=state.theta_prime.replace(
                data=state.theta_prime.data.at[..., k_sfc].set(theta_sfc),
            ),
            tracers=state.tracers.replace(
                data=state.tracers.data.at[..., k_sfc, 0].set(qv_sfc),
            ),
        )
        tend = physics_fn(s, grid, hc, tm)
        return (
            jnp.sum(tend.dtheta_prime_dt.data ** 2)
            + jnp.sum(tend.dtracers_dt.data ** 2)
        )

    g_theta, g_qv = jax.grad(loss, argnums=(0, 1))(theta_sfc0, qv_sfc0)
    assert bool(jnp.all(jnp.isfinite(g_theta)))
    assert bool(jnp.all(jnp.isfinite(g_qv)))
    # Non-trivial (perturbing theta_sfc / qv_sfc changes flux ->
    # tendencies -> loss).
    assert float(jnp.max(jnp.abs(g_theta))) > 0.0
    assert float(jnp.max(jnp.abs(g_qv))) > 0.0


def test_rcemip_physics_fn_returns_correct_tendency_shape():
    """``make_rcemip_physics`` must return a PlaneNonHydrostaticTendencies
    pytree with shapes matching the state."""
    model, state, grid, hc, tm = _setup_rcemip()
    physics_fn = make_rcemip_physics(grid, hc, tm, radiation_config=_RCEMIP_RADIATION_CFG, microphysics_config=_RCEMIP_MICROPHYSICS_CFG, dt=2.0)
    tend = physics_fn(state, grid, hc, tm)
    assert tend.du_dt.data.shape == state.u.data.shape
    assert tend.dv_dt.data.shape == state.v.data.shape
    assert tend.dw_dt.data.shape == state.w.data.shape
    assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
    assert tend.drho_prime_dt.data.shape == state.rho_prime.data.shape
    assert tend.dphis_dt.data.shape == state.phis.data.shape
    assert tend.dtracers_dt.data.shape == state.tracers.data.shape

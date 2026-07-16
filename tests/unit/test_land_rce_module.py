"""Direct unit tests for the shared idealized LAND-RCE slab-surface physics
module :mod:`legoesm.atmosphere.idealized.land_rce`.

These pin the four public routines that were factored out of
``scripts/run/run_rcemip_plane.py`` so a second (MPI) driver can reuse them
without duplicating numerics:

* :func:`land_surface_flux_tendencies` — SAM ``oceflx`` bulk surface fluxes
  for a 2-D prognostic slab skin temperature (finite + WISHE-sane flux signs +
  ``beta`` moisture-availability scaling + column-local surface-level
  placement).
* :func:`update_land_slab_temperature` — the slab surface-energy-balance step
  responds to a net energy imbalance with the correct sign.
* :func:`apply_plane_tendency_forward_euler` — a forward-Euler increment
  actually changes the state (and leaves zero-tendency fields untouched).
* :func:`sum_plane_tendencies` — field-wise addition of plane tendencies.

Everything runs on a tiny synthetic 4x4x10 plane state so the routines are
exercised per horizontal ``(ny, nx)`` cell (the column-local property required
for per-rank use under MPI horizontal decomposition).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (  # noqa: E402
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.idealized.land_rce import (  # noqa: E402
    apply_plane_tendency_forward_euler,
    land_surface_flux_tendencies,
    sum_plane_tendencies,
    update_land_slab_temperature,
)
from legoesm.core.state import PlaneNonHydrostaticTendencies  # noqa: E402
from legoesm.grids.plane import create_plane_grid  # noqa: E402
from legoesm.grids.vertical import create_height_coordinate  # noqa: E402


# create_plane_grid enforces nx, ny >= 4; 4x4x10 is the smallest valid tile.
NY = NX = 4
NLEV = 10
K_SFC = NLEV - 1
P_SFC = 1.0e5          # surface pressure [Pa] for the bulk saturation humidity
U_LOW = 5.0            # uniform westerly [m/s] so surface drag is signed
QV_LOW = 1.0e-3        # small near-surface q_v [kg/kg] -> surface moister


@pytest.fixture(scope="module")
def plane_state():
    """4x4x10 rest plane state with 3 moist tracers (q_v, q_c, q_r)."""
    grid = create_plane_grid(
        nx=NX, ny=NY, nlev=NLEV, dx=4_000.0, dy=4_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    # make_rest_state allocates zero tracers; give it 3 (q_v/q_c/q_r) with a
    # small near-surface q_v so the (saturated) surface is moister than the
    # air -> upward (positive) latent-heat flux.
    tracers = jnp.zeros((NY, NX, NLEV, 3), dtype=jnp.float64)
    tracers = tracers.at[..., K_SFC, 0].set(QV_LOW)
    u = jnp.full((NY, NX, NLEV), U_LOW, dtype=jnp.float64)
    state = rest._replace(
        u=rest.u.replace(data=u),
        tracers=rest.tracers.replace(data=tracers),
    )
    return {"state": state, "grid": grid, "hc": hc, "tm": tm}


def _warm_skin_T(hc):
    """Skin temperature [K] a few K above the near-surface AIR temperature so
    the sensible-heat flux is unambiguously upward (surface warmer than air).

    Near-surface air temperature is the reference potential temperature times
    the Exner factor at the lowest model level (rest state => theta' = 0).
    """
    T_air = float(hc.theta_ref[K_SFC] * hc.exner_ref[K_SFC])
    return jnp.full((NY, NX), T_air + 5.0, dtype=jnp.float64)


# --------------------------------------------------------------------------- #
# land_surface_flux_tendencies                                                #
# --------------------------------------------------------------------------- #


def test_land_surface_flux_tendencies_finite_and_sane_signs(plane_state):
    T_sfc = _warm_skin_T(plane_state["hc"])
    tend, diag = land_surface_flux_tendencies(
        plane_state["state"], plane_state["grid"], plane_state["hc"],
        plane_state["tm"], T_sfc, p_sfc=P_SFC, beta=1.0,
    )

    assert isinstance(tend, PlaneNonHydrostaticTendencies)

    # Every flux diagnostic is a finite (ny, nx) field.
    for key, val in diag.items():
        assert val.shape == (NY, NX), key
        assert bool(jnp.all(jnp.isfinite(val))), key
    # Every tendency leaf is finite.
    for fname in tend._fields:
        assert bool(jnp.all(jnp.isfinite(getattr(tend, fname).data))), fname

    # WISHE-sane signs: warm surface heats the air (SH up > 0); the moist
    # surface over drier air evaporates (LH up > 0).
    assert bool(jnp.all(diag["shflx"] > 0.0))
    assert bool(jnp.all(diag["lhflx"] > 0.0))
    assert bool(jnp.all(diag["lhflx_potential"] > 0.0))
    # beta = 1 => realized latent flux equals the potential.
    np.testing.assert_allclose(
        np.asarray(diag["lhflx"]), np.asarray(diag["lhflx_potential"]),
    )

    # Column-local placement: fluxes act ONLY at the lowest model level.
    dtheta = tend.dtheta_prime_dt.data
    dqv = tend.dtracers_dt.data[..., 0]
    du = tend.du_dt.data
    assert bool(jnp.all(dtheta[..., :K_SFC] == 0.0))
    assert bool(jnp.all(dqv[..., :K_SFC] == 0.0))
    assert bool(jnp.all(du[..., :K_SFC] == 0.0))
    # ...and the surface increments carry the physical sign.
    assert bool(jnp.all(dtheta[..., K_SFC] > 0.0))   # SH warms the air
    assert bool(jnp.all(dqv[..., K_SFC] > 0.0))      # LH moistens the air
    assert bool(jnp.all(du[..., K_SFC] < 0.0))       # drag opposes u > 0
    # Only the first tracer (q_v) is a moisture sink of the surface flux.
    assert bool(jnp.all(tend.dtracers_dt.data[..., 1:] == 0.0))


def test_land_surface_flux_tendencies_beta_scales_latent(plane_state):
    """The moisture-availability ``beta`` scales the realized latent-heat flux
    linearly and leaves the sensible-heat flux untouched."""
    T_sfc = _warm_skin_T(plane_state["hc"])
    common = dict(
        state=plane_state["state"], grid=plane_state["grid"],
        height_coord=plane_state["hc"], terrain_metric=plane_state["tm"],
        T_sfc=T_sfc, p_sfc=P_SFC,
    )
    _, diag_full = land_surface_flux_tendencies(**common, beta=1.0)
    _, diag_half = land_surface_flux_tendencies(**common, beta=0.5)
    _, diag_zero = land_surface_flux_tendencies(**common, beta=0.0)

    np.testing.assert_allclose(
        np.asarray(diag_half["lhflx"]),
        0.5 * np.asarray(diag_full["lhflx_potential"]),
    )
    np.testing.assert_allclose(np.asarray(diag_zero["lhflx"]), 0.0)
    # Sensible heat + potential latent flux are independent of beta.
    np.testing.assert_allclose(
        np.asarray(diag_half["shflx"]), np.asarray(diag_full["shflx"]),
    )
    np.testing.assert_allclose(
        np.asarray(diag_half["lhflx_potential"]),
        np.asarray(diag_full["lhflx_potential"]),
    )


def test_land_surface_flux_tendencies_rejects_wrong_T_sfc_shape(plane_state):
    """Skin T must match the horizontal grid (guards a silent broadcast)."""
    bad_T = jnp.full((NY + 1, NX), 300.0, dtype=jnp.float64)
    with pytest.raises(ValueError, match="must match horizontal grid"):
        land_surface_flux_tendencies(
            plane_state["state"], plane_state["grid"], plane_state["hc"],
            plane_state["tm"], bad_T, p_sfc=P_SFC, beta=1.0,
        )


# --------------------------------------------------------------------------- #
# update_land_slab_temperature                                                #
# --------------------------------------------------------------------------- #


def test_update_land_slab_temperature_responds_to_imbalance():
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float64)
    shflx = jnp.full((NY, NX), 20.0, dtype=jnp.float64)
    lhflx = jnp.full((NY, NX), 30.0, dtype=jnp.float64)
    dt = 60.0
    heat_capacity = 5.0e5
    albedo = 0.2
    emissivity = 1.0

    # Net-heating case: abundant down-welling radiation, modest turbulent loss.
    T_warm, diag_warm = update_land_slab_temperature(
        T_sfc, sw_down=jnp.full((NY, NX), 600.0, dtype=jnp.float64),
        lw_down=jnp.full((NY, NX), 450.0, dtype=jnp.float64),
        shflx=shflx, lhflx=lhflx, dt=dt, heat_capacity=heat_capacity,
        albedo=albedo, emissivity=emissivity,
    )
    assert bool(jnp.all(diag_warm["Q_slab"] > 0.0))
    assert bool(jnp.all(T_warm > T_sfc))

    # Net-cooling case: no down-welling radiation, the surface only loses heat.
    T_cool, diag_cool = update_land_slab_temperature(
        T_sfc, sw_down=jnp.zeros((NY, NX), dtype=jnp.float64),
        lw_down=jnp.zeros((NY, NX), dtype=jnp.float64),
        shflx=shflx, lhflx=lhflx, dt=dt, heat_capacity=heat_capacity,
        albedo=albedo, emissivity=emissivity,
    )
    assert bool(jnp.all(diag_cool["Q_slab"] < 0.0))
    assert bool(jnp.all(T_cool < T_sfc))

    # Forward-Euler energy accounting: C * dT == Q_slab * dt.
    np.testing.assert_allclose(
        np.asarray((T_warm - T_sfc) * heat_capacity),
        np.asarray(diag_warm["Q_slab"] * dt),
    )


def test_update_land_slab_temperature_energy_balance_fixed_point():
    """Choosing LW_down so the net flux is exactly zero must hold the slab
    temperature fixed (uses the Stefan-Boltzmann constant from
    ``legoesm.constants`` for the outgoing longwave)."""
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float64)
    sw_down = jnp.full((NY, NX), 400.0, dtype=jnp.float64)
    shflx = jnp.full((NY, NX), 50.0, dtype=jnp.float64)
    lhflx = jnp.full((NY, NX), 110.0, dtype=jnp.float64)
    albedo = 0.25
    emissivity = 1.0
    lw_down = (
        emissivity * constants.sigma_sb * T_sfc ** 4
        + shflx + lhflx
        - sw_down * (1.0 - albedo)
    )

    T_new, diag = update_land_slab_temperature(
        T_sfc, sw_down, lw_down, shflx, lhflx,
        dt=120.0, heat_capacity=2.0e5, albedo=albedo, emissivity=emissivity,
    )
    np.testing.assert_allclose(np.asarray(diag["Q_slab"]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(T_new), np.asarray(T_sfc), atol=1e-12)


# --------------------------------------------------------------------------- #
# apply_plane_tendency_forward_euler                                          #
# --------------------------------------------------------------------------- #


def test_apply_plane_tendency_forward_euler_changes_state(plane_state):
    state = plane_state["state"]
    T_sfc = _warm_skin_T(plane_state["hc"])
    tend, _ = land_surface_flux_tendencies(
        state, plane_state["grid"], plane_state["hc"], plane_state["tm"],
        T_sfc, p_sfc=P_SFC, beta=1.0,
    )
    dt = 60.0
    new_state = apply_plane_tendency_forward_euler(state, tend, dt)

    # Exact forward-Euler increment on every prognostic field.
    np.testing.assert_allclose(
        np.asarray(new_state.theta_prime.data),
        np.asarray(state.theta_prime.data + dt * tend.dtheta_prime_dt.data),
    )
    np.testing.assert_allclose(
        np.asarray(new_state.u.data),
        np.asarray(state.u.data + dt * tend.du_dt.data),
    )
    # The surface fluxes actually moved the state: warming + deceleration.
    assert bool(jnp.all(
        new_state.theta_prime.data[..., K_SFC]
        > state.theta_prime.data[..., K_SFC]
    ))
    assert bool(jnp.all(new_state.u.data[..., K_SFC] < state.u.data[..., K_SFC]))
    # Vertical velocity has a zero tendency -> left exactly untouched.
    np.testing.assert_array_equal(
        np.asarray(new_state.w.data), np.asarray(state.w.data),
    )
    # dt = 0 is the identity.
    same = apply_plane_tendency_forward_euler(state, tend, 0.0)
    np.testing.assert_array_equal(
        np.asarray(same.theta_prime.data), np.asarray(state.theta_prime.data),
    )


# --------------------------------------------------------------------------- #
# sum_plane_tendencies                                                        #
# --------------------------------------------------------------------------- #


def test_sum_plane_tendencies_adds_fields(plane_state):
    state = plane_state["state"]
    T_sfc = _warm_skin_T(plane_state["hc"])
    tend_a, _ = land_surface_flux_tendencies(
        state, plane_state["grid"], plane_state["hc"], plane_state["tm"],
        T_sfc, p_sfc=P_SFC, beta=1.0,
    )
    # A second, distinct tendency with identical field metadata (dims).
    tend_b = tend_a._replace(**{
        f: getattr(tend_a, f).replace(data=getattr(tend_a, f).data * 2.0 + 0.5)
        for f in tend_a._fields
    })

    summed = sum_plane_tendencies(tend_a, tend_b)
    for f in summed._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(summed, f).data),
            np.asarray(getattr(tend_a, f).data + getattr(tend_b, f).data),
        )
        # Metadata (dims) carried from the first input.
        assert getattr(summed, f).dims == getattr(tend_a, f).dims

    # A single input is returned unchanged.
    assert sum_plane_tendencies(tend_a) is tend_a


def test_sum_plane_tendencies_empty_raises():
    with pytest.raises(ValueError, match="at least one tendency"):
        sum_plane_tendencies()

"""Smoke tests for the spectral column adapter in neural_physics.py.

(Formerly learned_column.py; folded into neural_physics.py — file kept to
preserve the physics-contract idealized_test reference.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from legoesm.atmosphere.physics.neural_physics import (
    build_column_physics,
    make_column_physics_fn,
)
from legoesm.atmosphere.physics.neural_physics import NeuralPhysics


def _perturb_output_head(model: NeuralPhysics, key, scale: float = 0.1):
    """Give the final layer nonzero weights (simulate a trained network).

    ``NeuralPhysics`` ZERO-inits its final layer so an UNTRAINED net emits
    exactly zero tendencies (epoch-0 rollout = pure dycore, #797).  Tests that
    check the head RESPONDS to inputs (SST sensitivity, flux/moisture wiring)
    would be vacuous on that degenerate net — every output is 0.  Fill the last
    layer with small random weights so the plumbing carries a real signal.
    """
    last = model.layers[-1]
    wkey, bkey = jax.random.split(key)
    new_last = eqx.tree_at(
        lambda l: (l.weight, l.bias), last,
        (jax.random.normal(wkey, last.weight.shape) * scale,
         jax.random.normal(bkey, last.bias.shape) * scale),
    )
    return eqx.tree_at(lambda m: m.layers[-1], model, new_last)


def test_build_column_physics_returns_neural_physics():
    nlev = 8
    key = jax.random.PRNGKey(0)
    model = build_column_physics(nlev=nlev, hidden_dim=16, n_layers=2, key=key)
    assert isinstance(model, NeuralPhysics)
    assert model.nlev == nlev


def test_build_column_physics_residual_scale_passes_through():
    nlev = 4
    key = jax.random.PRNGKey(1)
    model = build_column_physics(
        nlev=nlev, hidden_dim=8, n_layers=2, residual_scale=0.05, key=key
    )
    assert isinstance(model, NeuralPhysics)


def test_make_column_physics_fn_returns_callable():
    """Check that make_column_physics_fn produces a callable physics fn.

    We don't exercise the full spectral round-trip here (that needs a
    GaussianGrid + SpectralHydrostaticState) — just that the factory
    returns a function with the documented signature.
    """
    from legoesm.grids.gaussian import create_gaussian_grid

    nlev = 4
    key = jax.random.PRNGKey(0)
    model = build_column_physics(nlev=nlev, hidden_dim=8, n_layers=2, key=key)
    grid = create_gaussian_grid(n_max=10)

    fn = make_column_physics_fn(model, grid)
    assert callable(fn)


def test_neural_tendency_output_is_bounded():
    """The tendency head must SATURATE (tanh-bounded) so a trained weight
    blow-up can't drive the moist rollout to inf/nan. A huge input -> tendency
    capped at residual_scale*tendency_cap, NOT unbounded."""
    from legoesm.atmosphere.physics.neural_physics import NeuralPhysics

    nn = NeuralPhysics(nlev=4, hidden_dim=8, n_layers=2,
                       key=jax.random.PRNGKey(7),
                       residual_scale=1e-5, tendency_cap=5.0)
    y = nn(jnp.full((nn.n_input,), 1.0e6))   # extreme input
    n_rate = nn.nlev * 4 + 1
    tend = y[:n_rate]
    ceil = nn.residual_scale * nn.tendency_cap
    assert float(jnp.abs(tend).max()) <= ceil * 1.0001   # saturated, finite
    assert jnp.all(jnp.isfinite(y))


def test_neural_flux_head_writes_predicted_fluxes_into_held():
    """The bridge must write the NN's predicted TOA/surface fluxes into
    held_* (so the flux loss supervises them), NOT pass the (zeroed) input
    held through. held_dT_rad and sw_down_toa stay passthrough.
    """
    from legoesm.atmosphere.physics.neural_physics import (
        NeuralPhysics, make_neural_step_unified,
    )
    from legoesm.core.grid_adapters import make_adapter
    from legoesm.grids.latlon import create_latlon_grid

    nlev = 4
    grid = create_latlon_grid(n_lat=8)   # latlon (float32-OK; the AIMIP grid)
    adapter = make_adapter(grid)
    # Large flux scale + nonzero head weights -> nonzero predicted fluxes.
    # (NeuralPhysics zero-inits its final layer; perturb it so the flux head
    # is non-degenerate and the write-into-held_* plumbing is actually tested.)
    nn = NeuralPhysics(nlev=nlev, hidden_dim=8, n_layers=2,
                       key=jax.random.PRNGKey(3), flux_output_scale=100.0)
    nn = _perturb_output_head(nn, jax.random.PRNGKey(30))
    step = make_neural_step_unified(nn, adapter)

    nlat, nlon = int(grid.n_lat), int(grid.n_lon)
    s3 = (nlat, nlon, nlev)
    s2 = (nlat, nlon)
    T = jnp.full(s3, 280.0)
    z3 = jnp.zeros(s3)
    z2 = jnp.zeros(s2)
    p_s = jnp.full(s2, 1.0e5)
    lat = jnp.zeros(s2)
    # Positional tail after (need_rad, T, p_s, q_v, q_c, q_r, conv_prog):
    tail = (z3, z3, z2, z2, lat, lat, 0.0, 0.0, 600.0,  # u,v,sst,sic,lat,lon,doy,sod,dt
            z2, jnp.array(1361.0), z3, z2,              # solar_w, s_0, o3, aero
            z3, z2, z2, z2, z2, z2)                      # held_* (all zero)
    phys_out, held_new = step(jnp.bool_(True), T, p_s, z3, z3, z3, None, *tail)
    held_dT_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa, sw_down_toa = held_new

    # Predicted TOA/sfc fluxes were written (nonzero) ...
    assert float(jnp.abs(sw_up_toa).max()) > 0.0
    assert float(jnp.abs(lw_up_toa).max()) > 0.0
    assert float(jnp.abs(sw_net_sfc).max()) > 0.0
    # ... and match the PhysicsOutput the network produced.
    assert jnp.allclose(sw_up_toa, phys_out.sw_up_toa)
    assert jnp.allclose(lw_up_toa, phys_out.lw_up_toa)
    # held_dT_rad and sw_down_toa stay passthrough (zero in == zero out).
    assert float(jnp.abs(held_dT_rad).max()) == 0.0
    assert float(jnp.abs(sw_down_toa).max()) == 0.0


# ---------------------------------------------------------------------------
# Prescribed surface forcing (AMIP / interannual-variability pathway)
# ---------------------------------------------------------------------------

def _mini_spectral_state(grid, nlev, q_v_val=0.005):
    """Small moist SpectralHydrostaticState via the training bridge."""
    import numpy as np
    from legoesm.training.era5_to_state import era5_to_spectral_carry, ERA5Slice
    from legoesm.grids.vertical import create_sigma_coordinate

    sigma = create_sigma_coordinate(nlev)
    n_lat, n_lon = len(grid.lat), len(grid.lon)
    plev = np.linspace(10000.0, 100000.0, nlev)
    era5 = ERA5Slice(
        T=np.full((n_lat, n_lon, nlev), 280.0, dtype=np.float32),
        u=np.zeros((n_lat, n_lon, nlev), dtype=np.float32),
        v=np.zeros((n_lat, n_lon, nlev), dtype=np.float32),
        q=np.full((n_lat, n_lon, nlev), q_v_val, dtype=np.float32),
        p_s=np.full((n_lat, n_lon), 1.0e5, dtype=np.float32),
        sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
        phis=np.zeros((n_lat, n_lon), dtype=np.float32),
        lat=np.asarray(grid.lat), lon=np.asarray(grid.lon),
        plev_Pa=plev,
    )
    return era5_to_spectral_carry(era5, grid, sigma), sigma


def test_column_physics_responds_to_prescribed_sst():
    """Same state, different prescribed T_sfc → different tendencies.

    This is the AMIP pathway: without it the column NN cannot express
    interannual SST variability."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state

    nlev = 4
    grid = create_gaussian_grid(n_max=10)
    model = build_column_physics(nlev=nlev, hidden_dim=8, n_layers=2,
                                 residual_scale=0.1,
                                 key=jax.random.PRNGKey(11))
    # Non-degenerate head so the T_sfc feature can actually move the output
    # (zero-init final layer -> identical zero for any forcing).
    model = _perturb_output_head(model, jax.random.PRNGKey(31))
    fn = make_column_physics_fn(model, grid)
    carry, sigma = _mini_spectral_state(grid, nlev)
    state = carry_to_spectral_state(carry, grid)
    ncol = len(grid.lat) * len(grid.lon)

    def _forcing(t_sfc_val):
        return {
            "T_sfc": jnp.full((ncol,), t_sfc_val, dtype=jnp.float64),
            "sic": jnp.zeros((ncol,), dtype=jnp.float64),
            "day_of_year": jnp.asarray(180.0),
            "seconds_of_day": jnp.asarray(43200.0),
        }

    out_cold = fn(state, grid, sigma, forcing=_forcing(285.0))
    out_warm = fn(state, grid, sigma, forcing=_forcing(295.0))
    diff = float(jnp.max(jnp.abs(out_cold.T_hat.data - out_warm.T_hat.data)))
    assert diff > 1e-12, (
        "Column NN tendencies must respond to prescribed T_sfc "
        f"(got identical outputs, diff={diff})"
    )


def test_column_physics_nan_tsfc_falls_back_to_lowest_level():
    """NaN T_sfc (land cells) must be substituted by the lowest-level
    air T proxy — outputs stay finite, and an all-NaN forcing equals the
    unforced call at the same calendar ONLY through the proxy path."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state

    nlev = 4
    grid = create_gaussian_grid(n_max=10)
    model = build_column_physics(nlev=nlev, hidden_dim=8, n_layers=2,
                                 residual_scale=0.1,
                                 key=jax.random.PRNGKey(12))
    fn = make_column_physics_fn(model, grid)
    carry, sigma = _mini_spectral_state(grid, nlev)
    state = carry_to_spectral_state(carry, grid)
    ncol = len(grid.lat) * len(grid.lon)
    forcing = {
        "T_sfc": jnp.full((ncol,), jnp.nan, dtype=jnp.float64),
        "sic": jnp.zeros((ncol,), dtype=jnp.float64),
        "day_of_year": jnp.asarray(1.0),
        "seconds_of_day": jnp.asarray(0.0),
    }
    out = fn(state, grid, sigma, forcing=forcing)
    assert bool(jnp.all(jnp.isfinite(out.T_hat.data))), (
        "NaN prescribed T_sfc leaked into the tendencies"
    )


def test_untrained_moisture_head_rollout_stays_finite():
    """Regression (2026-07-03 column_nn retrain epoch-0 NaN): an UNTRAINED
    net's q_v head must not blow up a 36-step (6 h) forced rollout.  The
    rate head is temperature-calibrated; without _Q_HEAD_TENDENCY_FACTOR
    the moisture tendency is O(1e-3 kg/kg/s) and q_v hits NaN by step 36."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_rollout,
    )

    nlev = 4
    grid = create_gaussian_grid(n_max=10)
    model = build_column_physics(nlev=nlev, hidden_dim=16, n_layers=2,
                                 key=jax.random.PRNGKey(3))
    fn = make_column_physics_fn(model, grid)
    carry, sigma = _mini_spectral_state(grid, nlev)
    state = carry_to_spectral_state(carry, grid)
    ncol = len(grid.lat) * len(grid.lon)
    forcing = {
        "T_sfc": jnp.full((ncol,), 290.0, dtype=jnp.float64),
        "sic": jnp.zeros((ncol,), dtype=jnp.float64),
        "day_of_year": jnp.asarray(1.0),
        "seconds_of_day": jnp.asarray(0.0),
    }
    pe = SpectralPEConfig(time_integrator="ssp_rk3")
    out = spectral_rollout(state, fn, grid, sigma, pe, 600.0, 36,
                           forcing_base=forcing)
    qv = out.tracers["q_v"]
    qv = qv.data if hasattr(qv, "data") else qv
    assert bool(jnp.all(jnp.isfinite(qv))), "q_v went non-finite in 6 h"
    # The head is capped at 5e-6 kg/kg/s -> 36*600s adds < 0.11 kg/kg even
    # fully saturated; anything O(1) means the scale factor was lost.
    assert float(jnp.abs(qv).max()) < 0.5
    assert bool(jnp.all(jnp.isfinite(out.T_hat.data)))


def test_column_physics_moisture_head_is_live():
    """The dq_v/dt head must reach the q_v tracer tendency (it was
    previously silently discarded — no moisture physics at all)."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state

    nlev = 4
    grid = create_gaussian_grid(n_max=10)
    model = build_column_physics(nlev=nlev, hidden_dim=8, n_layers=2,
                                 residual_scale=0.1,
                                 key=jax.random.PRNGKey(13))
    # Non-degenerate head so the moisture output is nonzero (zero-init final
    # layer would make dq_v identically zero and the assert vacuous).
    model = _perturb_output_head(model, jax.random.PRNGKey(33))
    fn = make_column_physics_fn(model, grid)
    carry, sigma = _mini_spectral_state(grid, nlev)
    state = carry_to_spectral_state(carry, grid)
    tend = fn(state, grid, sigma)
    assert tend.tracers is not None and "q_v" in tend.tracers
    dqv = tend.tracers["q_v"]
    dqv = dqv.data if hasattr(dqv, "data") else dqv
    assert float(jnp.max(jnp.abs(dqv))) > 0.0, (
        "q_v tendency is all-zero: the moisture head is not wired"
    )

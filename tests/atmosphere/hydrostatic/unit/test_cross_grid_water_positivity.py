"""Cross-grid tracer-positivity wiring (#1354/#1515).

The spectral and lat-lon C-grid lanes previously had NO tracer positivity
floor, so non-monotone transport left negative water.  Both now route through
the shared ``apply_water_positivity`` (default: column-conserving borrow).
These tests prove the stage actually FIRES on each lane — negatives removed and
the borrow conserves the dp-weighted column integral — so the wiring is not
silently inert.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.vertical import create_sigma_coordinate


# ---------------------------------------------------------------------------
# Spectral lane
# ---------------------------------------------------------------------------

def test_spectral_positivity_fires_and_conserves():
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel,
        isothermal_rest_state_spectral,
    )
    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(8)
    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
    state = isothermal_rest_state_spectral(
        grid, sigma, perturbation_amplitude=0.0)

    shp = (grid.n_lat, grid.n_lon, sigma.n_levels)
    qv = jnp.full(shp, 5.0e-3).at[:, :, 0].set(-2.0e-3)  # negative surface lobe
    state = state._replace(tracers={
        "q_v": Field(data=qv, name="q_v",
                     dims=("lat", "lon", "level"), units="kg/kg")})

    out = model._apply_tracer_positivity(state)
    qv_out = out.tracers["q_v"].data
    assert float(jnp.min(qv_out)) >= -1e-30            # negatives removed

    # Borrow conserves the dp-weighted global integral (T is spectral, untouched)
    p_s = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))
    dp = p_s[..., None] * sigma.dsigma
    pre = float(jnp.sum(qv * dp))
    post = float(jnp.sum(qv_out * dp))
    assert abs(post - pre) <= 1e-9 * abs(pre)

    # Non-vacuity: with the borrow OFF a plain floor would ADD water.
    model_off = SpectralPrimitiveEquationModel(
        grid, sigma, SpectralPEConfig(conservative_tracer_clamp=False))
    qv_plain = model_off._apply_tracer_positivity(state).tracers["q_v"].data
    assert float(jnp.min(qv_plain)) >= -1e-30
    assert float(jnp.sum(qv_plain * dp)) > pre + 1e-6 * abs(pre)  # created mass


# ---------------------------------------------------------------------------
# Lat-lon C-grid lane
# ---------------------------------------------------------------------------

def test_latlon_positivity_fires_and_conserves():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        CGridLatLonHydrostaticState,
    )
    grid = create_latlon_grid(
        n_lat=32, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(8)
    # fix_mass off so the positivity stage is isolated from the mass-fix rescale.
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(fix_mass=False))

    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
    qv = jnp.full((n_lat, n_lon, nlev), 5.0e-3).at[:, :, 0].set(-2.0e-3)
    state = CGridLatLonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        T=jnp.full((n_lat, n_lon, nlev), 300.0),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={"q_v": qv},
    )

    out = model._apply_safety_rails(state)
    qv_out = out.tracers["q_v"]
    assert float(jnp.min(qv_out)) >= -1e-30            # negatives removed

    dp = state.p_s[..., None] * sigma.dsigma
    pre = float(jnp.sum(qv * dp))
    post = float(jnp.sum(qv_out * dp))
    # rtol 1e-6: this lane's tracer storage is fp32, so the borrow's residual
    # redistribution conserves to fp32-class roundoff (measured 1.7e-8 here),
    # not the fp64 1e-9 the spectral lane hits.
    assert abs(post - pre) <= 1e-6 * abs(pre)          # borrow conserved
    assert jnp.allclose(out.T, state.T)                # borrow leaves T alone

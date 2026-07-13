"""Smoke test for the AMIP experiment setup.

Verifies that the full AMIP pipeline (forcing + dynamics + physics)
runs for a few steps without blowup, using synthetic forcing data.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.forcing.amip import AMIPForcingConfig, load_amip_forcing, get_forcing_at_time
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel as PrimitiveEquationModel,
    CDGridPrimitiveEquationConfig as PrimitiveEquationConfig,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.forcing.time_utils import day_to_calendar
from legoesm import constants


def _make_synthetic_forcing(path, n_lat=18, n_lon=36):
    """Create minimal synthetic AMIP forcing."""
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = np.array([np.datetime64("2000-01-01") + np.timedelta64(i * 30, "D")
                       for i in range(4)])
    sst = np.full((4, n_lat, n_lon), 293.0, dtype=np.float32)  # already in K
    sst += lat[None, :, None] * 0.2
    sic = np.zeros((4, n_lat, n_lon), dtype=np.float32)
    sic[:, :2, :] = 0.8
    sic[:, -2:, :] = 0.6

    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst),
         "sic": (("time", "lat", "lon"), sic)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def amip_setup(tmp_path_factory):
    """Set up a minimal AMIP experiment."""
    N, NLEV = 4, 5
    DT = 600.0

    forcing_path = str(tmp_path_factory.mktemp("amip_smoke") / "forcing.nc")
    _make_synthetic_forcing(forcing_path)

    grid = create_cubed_sphere(N)
    sigma = create_sigma_coordinate(NLEV)

    config = AMIPForcingConfig(
        path=forcing_path, sst_var="sst", sic_var="sic",
        sst_offset=0.0, sic_scale=1.0,
    )
    forcing = load_amip_forcing(config, grid)

    dycore_config = PrimitiveEquationConfig(
        hyperdiff_coeff=1e15,
        hyperdiff_ps_coeff=1e15,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = PrimitiveEquationModel(grid, sigma, dycore_config)
    state = held_suarez_init(grid, sigma, T_init=280.0)

    # Initialize moisture
    p_full = state.p_s.data[..., None] * sigma.sigma_full
    q_sat = saturation_mixing_ratio(state.T.data, p_full)
    q_v = 0.6 * q_sat * sigma.sigma_full ** 2
    q_v = jnp.minimum(q_v, q_sat)

    return {
        "grid": grid, "sigma": sigma, "model": model, "state": state,
        "q_v": q_v, "forcing": forcing, "config": config, "dt": DT, "N": N,
    }


def test_amip_5_steps(amip_setup):
    """Run 5 AMIP steps (dynamics + physics) without blowup."""
    grid = amip_setup["grid"]
    sigma = amip_setup["sigma"]
    model = amip_setup["model"]
    state = amip_setup["state"]
    q_v = amip_setup["q_v"]
    forcing = amip_setup["forcing"]
    DT = amip_setup["dt"]

    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=0.06, perpetual_equinox=False,
    )
    sbm_config = SBMConfig(tau_c=7200.0, rh_ref=0.7)

    sigma_full = sigma.sigma_full
    sigma_half = sigma.sigma_half
    dsigma = sigma.dsigma
    T_ice = 271.35

    for step in range(5):
        day = step * DT / 86400.0
        day_of_year, _ = day_to_calendar(day)
        sst, sic = get_forcing_at_time(forcing, day)

        # Dynamics
        state = model.step_with_physics(state, DT)

        # Physics (inline, not JIT for test simplicity)
        nlev = sigma_full.shape[0]
        ncol = 6 * grid.n * grid.n
        T_sfc = sic * T_ice + (1.0 - sic) * sst
        p_full = state.p_s.data[..., None] * sigma_full
        p_half = state.p_s.data[..., None] * sigma_half

        insol = daily_mean_insolation(
            grid.lat.reshape(ncol), day_of_year, 1360.0,
        )
        rad_out = gray_radiation(
            T=state.T.data.reshape(ncol, nlev),
            p_full=p_full.reshape(ncol, nlev),
            p_half=p_half.reshape(ncol, nlev + 1),
            sfc_temperature=T_sfc.reshape(ncol),
            lat=grid.lat.reshape(ncol),
            q_v=q_v.reshape(ncol, nlev),
            insolation=insol,
            config=gray_config,
        )
        dT_rad = rad_out.heating_rate.reshape(state.T.data.shape)
        new_T = state.T.data + DT * dT_rad
        state = state._replace(T=state.T.replace(data=new_T))

        # Large-scale condensation
        q_sat = saturation_mixing_ratio(new_T, p_full)
        excess = jnp.maximum(q_v - q_sat, 0.0)
        q_v = q_v - excess
        new_T = new_T + constants.L_v * excess / constants.c_pd
        state = state._replace(T=state.T.replace(data=new_T))
        q_v = jnp.maximum(q_v, 0.0)

    # Verify everything is finite
    assert jnp.all(jnp.isfinite(state.T.data)), "Temperature not finite"
    assert jnp.all(jnp.isfinite(state.u.data)), "Winds not finite"
    assert jnp.all(jnp.isfinite(state.p_s.data)), "Surface pressure not finite"
    assert jnp.all(q_v >= 0), "Negative humidity"

    # Basic physical bounds
    T_min = float(jnp.min(state.T.data))
    T_max = float(jnp.max(state.T.data))
    assert T_min > 100, f"T_min={T_min} too cold"
    assert T_max < 500, f"T_max={T_max} too hot"

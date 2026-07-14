"""AMIP + RRTMG(P) smoke test on the lat-lon finite-volume (C-grid) dycore.

Companion to ``test_amip_rrtmg.py`` (which exercises the cubed-sphere
C-D-grid dycore).  This module pins the *lat-lon finite-volume* AMIP path:
the ``CGridLatLonPrimitiveEquationModel`` — the solver that
``grid_type="latlon"`` + ``discretization="finite_volume"`` resolves to in
``driver.component_factory`` — coupled to the full correlated-k radiation
(RRTMGP) plus SBM convection and large-scale condensation, driven by
prescribed AMIP SST/sea-ice.

Kept small (16×32 lat-lon, L5, a few steps) — a regression guard that the
lat-lon FV grid can run the CMIP6 AMIP physics stack without blowup, not a
science-quality integration.

CI cost split:
- ``test_amip_rrtmg_latlon_fv_steps`` invokes the correlated-k RRTMGP graph
  (a non-trivial JIT compile, like the cubed-sphere ``test_amip_rrtmg.py``),
  so it is marked ``slow`` and excluded from the default ``-m 'not slow'``
  per-PR sweep, alongside the deck-level RRTMG case gated by
  ``LEGOESM_RUN_AMIP_INTEGRATION``.
- ``test_amip_rrtmg_latlon_fv_sst_varies_across_grid`` only exercises the
  grid-agnostic AMIP SST regridding onto the lat-lon grid (no radiation
  compile), so it stays in the default sweep as the cheap coverage.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.forcing.amip import (
    AMIPForcingConfig,
    load_amip_forcing,
    get_forcing_at_time,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationModel,
    CGridLatLonPrimitiveEquationConfig,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm import constants
from legoesm.forcing.time_utils import day_to_calendar


def _make_synthetic_forcing(path, n_lat=18, n_lon=36):
    """Minimal synthetic AMIP SST/SIC forcing on a regular lat-lon source."""
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = np.array([np.datetime64("2000-01-01") + np.timedelta64(i * 30, "D")
                      for i in range(4)])
    # Warm tropics, cool poles — a realistic meridional SST gradient so the
    # prescribed-SST forcing actually varies across the grid.
    sst = np.full((4, n_lat, n_lon), 300.0, dtype=np.float32)
    sst -= (np.abs(lat)[None, :, None] ** 2) * (28.0 / 85.0 ** 2)
    sic = np.zeros((4, n_lat, n_lon), dtype=np.float32)
    sic[:, :2, :] = 0.8
    sic[:, -2:, :] = 0.6
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst),
         "sic": (("time", "lat", "lon"), sic)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


def _setup(tmp_path):
    """Build a tiny lat-lon FV AMIP configuration with RRTMG radiation."""
    N_LAT, NLEV, DT = 16, 5, 600.0

    forcing_path = str(tmp_path / "forcing.nc")
    _make_synthetic_forcing(forcing_path)

    grid = create_latlon_grid(
        n_lat=N_LAT, radius=constants.R_earth, omega=constants.Omega,
    )
    sigma = create_sigma_coordinate(n_levels=NLEV)

    cfg = AMIPForcingConfig(
        path=forcing_path, sst_var="sst", sic_var="sic",
        sst_offset=0.0, sic_scale=1.0,
    )
    forcing = load_amip_forcing(cfg, grid)

    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(fix_mass=True),
    )
    state = held_suarez_init_latlon(grid, sigma, T_init=280.0)

    p_full = state.p_s.data[..., None] * sigma.sigma_full
    q_sat = saturation_mixing_ratio(state.T.data, p_full)
    q_v = jnp.minimum(0.6 * q_sat * sigma.sigma_full ** 2, q_sat)

    return dict(grid=grid, sigma=sigma, model=model, state=state,
                q_v=q_v, forcing=forcing, dt=DT)


@pytest.mark.slow
def test_amip_rrtmg_latlon_fv_steps(tmp_path):
    """Run a few AMIP steps on the lat-lon FV dycore with the full CMIP6
    radiation+convection physics stack and a time-varying prescribed SST;
    require finite, physically-bounded state throughout.

    Marked ``slow`` (excluded from the default ``-m 'not slow'`` sweep)
    because it compiles the RRTMGP correlated-k graph."""
    s = _setup(tmp_path)
    grid, sigma, model = s["grid"], s["sigma"], s["model"]
    state, q_v, forcing, DT = s["state"], s["q_v"], s["forcing"], s["dt"]

    sigma_full, sigma_half = sigma.sigma_full, sigma.sigma_half
    nlev = sigma.n_levels
    ncol = grid.n_lat * grid.n_lon
    lat_col = grid.lat2d.reshape(ncol)

    rrtmg_cfg = RRTMGPConfig(
        co2_ppmv=336.8, ch4_ppbv=1550.0, n2o_ppbv=301.0,
        sfc_emissivity=0.98, sfc_albedo=0.06, S_0=1360.0,
    )
    sbm_cfg = SBMConfig(tau_c=7200.0, rh_ref=0.7)
    T_ice, albedo_ice, albedo_ocean = constants.T_freeze_ocean, 0.65, 0.06

    sst_means = []
    for step in range(4):
        day = step * DT / 86400.0
        day_of_year, _ = day_to_calendar(day)
        sst, sic = get_forcing_at_time(forcing, day)
        sst_means.append(float(jnp.mean(sst)))

        # Dynamics on the lat-lon C-grid (finite-volume) dycore.
        state = model.step_with_physics(state, DT)

        T_sfc = sic * T_ice + (1.0 - sic) * sst
        albedo = sic * albedo_ice + (1.0 - sic) * albedo_ocean
        p_full = state.p_s.data[..., None] * sigma_full
        p_half = state.p_s.data[..., None] * sigma_half

        insol = daily_mean_insolation(lat_col, day_of_year, rrtmg_cfg.S_0)
        cos_zenith = jnp.clip(insol / rrtmg_cfg.S_0, 0.0, 1.0)

        rad_out = rrtmgp_radiation(
            T=state.T.data.reshape(ncol, nlev),
            p_full=p_full.reshape(ncol, nlev),
            p_half=p_half.reshape(ncol, nlev + 1),
            sfc_temperature=T_sfc.reshape(ncol),
            q_v=q_v.reshape(ncol, nlev),
            cos_zenith=cos_zenith,
            config=rrtmg_cfg,
            sfc_albedo_override=float(jnp.mean(albedo)),
        )
        assert jnp.all(jnp.isfinite(rad_out.heating_rate)), \
            "RRTMG heating rate not finite on lat-lon FV grid"
        new_T = state.T.data + DT * rad_out.heating_rate.reshape(state.T.data.shape)

        conv_out = sbm_convection(
            T=state.T.data.reshape(ncol, nlev),
            q_v=q_v.reshape(ncol, nlev),
            p_full=p_full.reshape(ncol, nlev),
            p_half=p_half.reshape(ncol, nlev + 1),
            dt=DT, config=sbm_cfg,
        )
        new_T = new_T + DT * conv_out.dT_dt.reshape(state.T.data.shape)
        q_v = q_v + DT * conv_out.dq_v_dt.reshape(q_v.shape)

        # Large-scale condensation.
        q_sat = saturation_mixing_ratio(new_T, p_full)
        excess = jnp.maximum(q_v - q_sat, 0.0)
        q_v = jnp.maximum(q_v - excess, 0.0)
        new_T = new_T + constants.L_v * excess / constants.c_pd
        state = state._replace(T=state.T.replace(data=new_T))

    # Finite + physically bounded after the full physics stack.
    assert jnp.all(jnp.isfinite(state.T.data)), "T not finite after RRTMG (latlon FV)"
    assert jnp.all(jnp.isfinite(state.u.data)), "u not finite after RRTMG (latlon FV)"
    assert jnp.all(jnp.isfinite(state.p_s.data)), "p_s not finite (latlon FV)"
    assert jnp.all(q_v >= 0.0), "negative humidity after RRTMG (latlon FV)"

    T_min, T_max = float(jnp.min(state.T.data)), float(jnp.max(state.T.data))
    assert T_min > 100.0, f"T_min={T_min} too cold (latlon FV)"
    assert T_max < 500.0, f"T_max={T_max} too hot (latlon FV)"

    # The prescribed SST regridded onto the lat-lon grid must be physical and
    # actually consumed (warm tropics → global-mean SST clearly above freezing).
    assert all(np.isfinite(m) for m in sst_means)
    assert sst_means[0] > constants.T_freeze_ocean, \
        f"regridded AMIP SST mean {sst_means[0]:.1f} K below ocean freezing"


def test_amip_rrtmg_latlon_fv_sst_varies_across_grid(tmp_path):
    """The regridded prescribed SST must carry the imposed meridional
    gradient on the lat-lon grid (tropics warmer than poles) — guards the
    grid-agnostic AMIP forcing regridding for the lat-lon target."""
    s = _setup(tmp_path)
    forcing, grid = s["forcing"], s["grid"]
    sst, _ = get_forcing_at_time(forcing, 0.0)
    assert sst.shape == (grid.n_lat, grid.n_lon)

    # Equatorial band warmer than polar band.
    eq = grid.n_lat // 2
    sst_eq = float(jnp.mean(sst[eq - 1:eq + 1, :]))
    sst_pole = float(jnp.mean(sst[:2, :]))
    assert sst_eq > sst_pole + 5.0, (
        f"regridded SST shows no meridional gradient: "
        f"eq={sst_eq:.1f} K, pole={sst_pole:.1f} K"
    )

"""CAM6's prescribed ERA5 surface fluxes carry no information from after the
forecast's initial time.

Two links make the claim, each pinned here:
1. the flux planes of a sample come from the flux store's snapshot at EXACTLY
   the initial time (no nearest match, no later snapshot), and
2. every lead of a multi-lead sample uses that one initial-time forcing.
The third link is ERA5's own convention: an hourly ``mean_*`` flux at time t
is the mean over the hour ENDING at t, so the t0 snapshot averages
(t0 - 1 h, t0] -- before the forecast starts.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import legoesm.training.era5_to_state as e2s
from tests.unit.test_era5_load_slice import _synthetic_era5_multitime
from tests.unit.test_era5_surface_flux_loader import _flux_config

_FLUX = ("mean_surface_sensible_heat_flux", "mean_surface_latent_heat_flux",
         "mean_eastward_turbulent_surface_stress",
         "mean_northward_turbulent_surface_stress",
         "mean_surface_downward_short_wave_radiation_flux",
         "mean_surface_net_short_wave_radiation_flux",
         "mean_surface_downward_long_wave_radiation_flux",
         "mean_surface_net_long_wave_radiation_flux")


def _time_coded_fluxes(state):
    """Flux store on the state store's grid and times; every value = 100*(k+1)
    at time index k (net radiation 10*(k+1), so the upwelling differences are
    time coded too), so a plane names the snapshot it came from."""
    t = state.time.values
    shape = (len(t), state.sizes["lat"], state.sizes["lon"])
    code = (100.0 * (np.arange(len(t)) + 1.0))[:, None, None]
    data = {v: (("time", "lat", "lon"),
                np.broadcast_to(code / 10.0 if "_net_" in v else code,
                                shape).astype(np.float32))
            for v in _FLUX}
    data["land_sea_mask"] = (("lat", "lon"), np.zeros(shape[1:], np.float32))
    return xr.Dataset(
        data,
        coords={"time": t, "lat": state.lat.values, "lon": state.lon.values})


@pytest.mark.parametrize("k", [0, 1])
def test_flux_planes_are_the_initial_time_snapshot(k):
    state = _synthetic_era5_multitime()
    sl = e2s.load_era5_slice(_flux_config(load_surface_fluxes=True), k,
                             ds=state, flux_ds=_time_coded_fluxes(state))
    c = 100.0 * (k + 1)
    # Each plane is a sign flip / difference of the coded means at ONE time.
    for name, want in (("sfc_shf", -c), ("sfc_lhf", -c), ("sfc_tau_x", -c),
                       ("sfc_tau_y", -c), ("sfc_sw_down", c), ("sfc_sw_up", 0.9 * c),
                       ("sfc_lw_up", 0.9 * c)):
        np.testing.assert_array_equal(getattr(sl, name), want, err_msg=name)


def test_uncached_sample_reads_its_forcing_slice_at_the_initial_time():
    """The uncached loader's forcing slice is the third ``_sample_inputs``
    slice, read at i_ic for every lead layout."""
    from legoesm.training import scale_build as sb

    state = _synthetic_era5_multitime()
    flux = _time_coded_fluxes(state)
    _, tg, f = sb._sample_inputs(_flux_config(), _flux_config(load_surface_fluxes=True),
                                 0, 1, state, None, flux, int, lead_offsets=(1,))
    np.testing.assert_array_equal(f.sfc_shf, -100.0)      # time 0, not 1
    assert len(tg) == 1


def test_a_missing_initial_time_flux_is_an_error_not_a_neighbour():
    state = _synthetic_era5_multitime()
    flux = _time_coded_fluxes(state).isel(time=[0])     # only t0 exists
    with pytest.raises(ValueError, match="no snapshot at the state time"):
        e2s.load_era5_slice(_flux_config(load_surface_fluxes=True), 1,
                            ds=state, flux_ds=flux)

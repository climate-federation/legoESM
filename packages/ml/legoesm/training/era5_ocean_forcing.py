"""ERA5 daily surface forcing for the lat-lon ocean (ocean-WB lane).

User decisions 2026-11-11: DKRZ on-disk daily ERA5 (store built by
``scripts/data/build_era5_ocean_forcing.py``), daily means held constant through each model day,
JRA55-do runoff climatology, air-sea fluxes scaled by the open-water fraction of a 1993-2017
ORAS5 sea-ice climatology, no restoring.  Two modes:

* ``bulk``: ERA5 state -> :class:`JRA55Slice` -> ``jra55_to_atm_surface`` -> coupler
  ``ocean_tile_response`` (the OMIP JRA55 scan-block arithmetic: q_net = SW (1 - albedo) + LW_down
  - LW_up - SH - LH, E = LH / L_v).  Reference heights: ERA5 wind 10 m, T/q 2 m (``coupler_cfg``).
* ``prescribed``: ERA5's own fluxes.  ERA5 is positive DOWNWARD: q_net = ssr + str + sshf + slhf;
  evaporation (positive up) = -e * rho_water; the coupler's stress is the stress ON THE AIR
  (``tau_x = -rho Cd |U| u``, opposite to the wind), so tau = -(ewss, nsss).

Both go through ``blend_ice_ocean_forcing`` with a ZERO ice response, i.e. stress, evaporation
and heat/SW scale with ``1 - A``; precipitation and runoff stay full-cell (vote 3-0).
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity

CHANNELS = ("u10", "v10", "t2m", "d2m", "msl", "sp", "ssrd", "strd", "tp", "sf", "sshf", "slhf",
            "ssr", "str", "ewss", "nsss", "e")
_I = {n: i for i, n in enumerate(CHANNELS)}


def load_era5_day(store, date) -> np.ndarray:
    a = np.load(Path(store) / f"{np.datetime64(date, 'D')}.npy", mmap_mode="r")
    if a.shape[0] != len(CHANNELS):
        raise ValueError(f"{date}: ERA5 day has {a.shape[0]} channels, expected {len(CHANNELS)}")
    return a


def era5_to_slice(a, runoff):
    """ERA5 day (17, lat, lon) -> JRA55Slice (SI rates; huss over water at (d2m, sp))."""
    from legoesm.forcing.jra55_do import JRA55Slice
    a = jnp.asarray(a, jnp.float64)
    g = lambda n: a[_I[n]]
    rho = constants.rho_water
    snow = jnp.maximum(g("sf"), 0.0) * rho
    rain = jnp.maximum(g("tp") * rho - snow, 0.0)
    return JRA55Slice(uas=g("u10"), vas=g("v10"), tas=g("t2m"),
                      huss=saturation_specific_humidity(g("d2m"), g("sp")),
                      psl=g("msl"), rsds=g("ssrd"), rlds=g("strd"), prra=rain, prsn=snow,
                      friver=jnp.asarray(runoff))


def _zero_ice_response(like):
    from legoesm.core.coupling_fields import TileResponse
    z = jnp.zeros_like(like)
    return TileResponse(*([z] * len(TileResponse._fields)))


def surface_forcing(mode, a, sst_C, runoff, ice_frac, ocean_mask, *, lat_rad=None,
                    lon_rad=None, day=0.0, ref_year=1993, coupler_cfg=None):
    """One model day's ``(FreshwaterForcing, OceanSurfaceForcing)`` from an ERA5 day."""
    from legoesm.coupler.ocean_forcing import blend_ice_ocean_forcing, open_water_bulk_forcing
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    a = jnp.asarray(a, jnp.float64)
    g = lambda n: a[_I[n]]
    slc = era5_to_slice(a, runoff)
    if mode == "bulk":
        from legoesm.forcing.jra55_do import jra55_to_atm_surface
        atm = jra55_to_atm_surface(slc, lat_rad, lon_rad, day, ref_year=ref_year)
        _, fw, sf = open_water_bulk_forcing(atm, sst_C + constants.T_freeze, slc.friver,
                                            coupler_cfg)
    elif mode == "prescribed":
        precip = slc.prra + slc.prsn
        fw = FreshwaterForcing(precip=precip, evap=-g("e") * constants.rho_water,
                               runoff=slc.friver, ice_fw=jnp.zeros_like(precip))
        sf = OceanSurfaceForcing(sw_down=g("ssrd"), q_net=g("ssr") + g("str") + g("sshf") + g("slhf"),
                                 tau_x=-g("ewss"), tau_y=-g("nsss"), freshwater=None)
    else:
        raise ValueError(f"unknown forcing mode {mode!r}; expected 'bulk' or 'prescribed'")
    return blend_ice_ocean_forcing(open_sf=sf, open_fw=fw, ice_resp=_zero_ice_response(sf.q_net),
                                   ice_concentration=jnp.asarray(ice_frac),
                                   ocean_mask=ocean_mask, sw_partition="prescaled")

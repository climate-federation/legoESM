"""Forced-ocean (OMIP) prognostic sea-ice surface-forcing partitioning.

``omip_sea_ice_surface_forcing`` advances a slab sea-ice tile and partitions the
surface forcing between the open-ocean fraction ``f_ocean = 1 - A`` (the
caller's full-cell bulk-flux forcing) and the ice tile (basal heat, melt/freeze
freshwater, brine salt, ice-ocean stress).  These tests pin the partition:

* with NO ice (concentration 0, warm SST so none forms) the blended forcing
  reduces EXACTLY to the caller's open-ocean forcing (ice contribution zero,
  f_ocean = 1) — i.e. it is a no-op on an ice-free ocean, so wiring it can never
  change an ice-free run;
* with (near-)full ice the open-ocean atmospheric fluxes are suppressed
  (scaled by f_ocean -> ~0) and the ocean sees the ice exchange instead;
* the result is finite and shape-preserving (jit-able).

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field
from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.ocean.freshwater import FreshwaterForcing
from legoesm.ocean.state import OceanSurfaceForcing

SHAPE = (4, 4)
DT = 3600.0


def _fld(val):
    return Field(jnp.full(SHAPE, float(val)), name="x", dims=("y", "x"), units="")


def _ice_state(conc, h_ice, T_ice):
    return SeaIceState(h_ice=_fld(h_ice), T_ice=_fld(T_ice),
                       concentration=_fld(conc))


def _atm(T_air):
    o = jnp.ones(SHAPE)
    return AtmToSurface(
        sw_down=200.0 * o, lw_down=300.0 * o,
        precip_total=1e-5 * o, precip_snow=jnp.zeros(SHAPE),
        T_lowest=float(T_air) * o, q_lowest=3e-3 * o,
        u_lowest=5.0 * o, v_lowest=jnp.zeros(SHAPE),
        p_lowest=1.0e5 * o, p_surface=1.0e5 * o,
        rho_lowest=1.25 * o, cos_zenith=0.5 * o,
        co2_ppmv=400.0 * o,
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )


def _open_ocean_forcing():
    o = jnp.ones(SHAPE)
    sf = OceanSurfaceForcing(
        sw_down=200.0 * o, q_net=120.0 * o,
        tau_x=0.1 * o, tau_y=0.02 * o,
    )
    fw = FreshwaterForcing(
        precip=1e-5 * o, evap=4e-6 * o, runoff=jnp.zeros(SHAPE),
        ice_fw=jnp.zeros(SHAPE), restoring=jnp.zeros(SHAPE),
    )
    return sf, fw


def test_no_ice_reduces_to_open_ocean_forcing():
    """Ice-free ocean (concentration 0, warm SST -> none forms): the blended
    forcing equals the caller's open-ocean forcing exactly (f_ocean=1, no ice
    exchange).  Guards that wiring the helper is a no-op on ice-free cells."""
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, 285.0)   # well above the -1.8 degC freezing point
    ice0 = _ice_state(conc=0.0, h_ice=0.0, T_ice=271.0)
    new_ice, fw, sf = omip_sea_ice_surface_forcing(
        ice_state=ice0, ice_config=SeaIceConfig(), atm=_atm(290.0),
        ocean_sst_K=sst_K, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
    )
    # No ice formed.
    assert float(jnp.max(new_ice.concentration.data)) == pytest.approx(0.0, abs=1e-12)
    # Blended forcing == open-ocean forcing (f_ocean == 1, ice exchange == 0).
    assert np.allclose(np.asarray(sf.q_net), np.asarray(sf0.q_net))
    assert np.allclose(np.asarray(sf.tau_x), np.asarray(sf0.tau_x))
    assert np.allclose(np.asarray(sf.tau_y), np.asarray(sf0.tau_y))
    assert np.allclose(np.asarray(fw.evap), np.asarray(fw0.evap))
    assert np.allclose(np.asarray(fw.precip), np.asarray(fw0.precip))
    assert np.allclose(np.asarray(fw.ice_fw), 0.0)


def test_full_ice_suppresses_open_ocean_atmospheric_fluxes():
    """Near-full ice cover: the open-ocean atmospheric exchange is scaled by
    f_ocean -> ~0, so the ocean's evaporation (open-ocean only) collapses and
    its net heat is no longer the bare open-ocean value."""
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, constants.T_freeze_ocean)  # at freezing
    ice0 = _ice_state(conc=1.0, h_ice=3.0, T_ice=250.0)
    new_ice, fw, sf = omip_sea_ice_surface_forcing(
        ice_state=ice0, ice_config=SeaIceConfig(), atm=_atm(240.0),
        ocean_sst_K=sst_K, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
    )
    A = float(jnp.mean(new_ice.concentration.data))
    assert A > 0.9, f"thick cold ice should persist, got A={A}"
    # Evaporation acts only on the open-ocean fraction -> suppressed ~ (1-A).
    assert float(jnp.max(fw.evap)) < 0.2 * float(jnp.max(fw0.evap))
    # The blended q_net is NOT the bare open-ocean value (open part suppressed,
    # ice basal exchange added).
    assert not np.allclose(np.asarray(sf.q_net), np.asarray(sf0.q_net))


def test_partition_is_finite_and_shape_preserving():
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, 274.0)
    ice0 = _ice_state(conc=0.4, h_ice=0.8, T_ice=268.0)
    new_ice, fw, sf = omip_sea_ice_surface_forcing(
        ice_state=ice0, ice_config=SeaIceConfig(), atm=_atm(263.0),
        ocean_sst_K=sst_K, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
    )
    for arr in (sf.q_net, sf.tau_x, sf.tau_y, sf.salt_flux, fw.precip,
                fw.evap, fw.ice_fw, new_ice.concentration.data):
        a = np.asarray(arr)
        assert a.shape == SHAPE
        assert np.all(np.isfinite(a))


def test_kpp_channel_carries_net_physical_freshwater():
    """sf.freshwater (the KPP surface-buoyancy channel) must equal the BLENDED
    physical net freshwater P - E + R + ice_fw — never the ice term alone and
    never a second mass application (the mass lives on the returned fw struct,
    consumed once via model.step(freshwater=fw)).  See blend_ice_ocean_forcing."""
    from legoesm.ocean.freshwater import net_freshwater_flux
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, 274.0)
    ice0 = _ice_state(conc=0.4, h_ice=0.8, T_ice=268.0)
    _, fw, sf = omip_sea_ice_surface_forcing(
        ice_state=ice0, ice_config=SeaIceConfig(), atm=_atm(263.0),
        ocean_sst_K=sst_K, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
    )
    np.testing.assert_allclose(np.asarray(sf.freshwater),
                               np.asarray(net_freshwater_flux(fw)))


def test_helper_is_jittable():
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, 274.0)
    ice0 = _ice_state(conc=0.4, h_ice=0.8, T_ice=268.0)
    cfg = SeaIceConfig()

    @jax.jit
    def run(ice_state, sst):
        return omip_sea_ice_surface_forcing(
            ice_state=ice_state, ice_config=cfg, atm=_atm(263.0),
            ocean_sst_K=sst, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
        )

    new_ice, fw, sf = run(ice0, sst_K)
    assert np.all(np.isfinite(np.asarray(sf.q_net)))

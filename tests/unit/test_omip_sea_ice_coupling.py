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
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm.coupler.ocean_forcing import (
    blend_ice_ocean_forcing,
    omip_sea_ice_surface_forcing,
)
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


def test_wrapper_partitions_with_post_step_concentration():
    """The wrapper must partition with the POST-step concentration (the ONE
    documented time level for this legacy entry).  Grow ice from OPEN WATER in
    one step (freezing SST, very cold air, NO shortwave, weak LW down -> the
    open-water surface energy balance Q_sfc < 0 -> lead freeze raises A from
    0), then check the pure f_open-scaled channels (sw_down, evap — no
    ice-term addition) against exactly (1 - A_new).  A regression to the
    PRE-step concentration would leave them unscaled (f_open = 1 - 0 = 1) and
    fail the exact match."""
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, constants.T_freeze_ocean)   # at freezing
    ice0 = _ice_state(conc=0.0, h_ice=0.0, T_ice=250.0)
    # Polar-night freezing: no SW (the default _atm carries 200 W/m2, which
    # cancels the cold-air heat loss and blocks open-water freeze) + weak
    # downwelling LW so the net surface balance is strongly negative.
    atm = _atm(240.0)._replace(sw_down=jnp.zeros(SHAPE),
                               lw_down=150.0 * jnp.ones(SHAPE))
    new_ice, fw, sf = omip_sea_ice_surface_forcing(
        ice_state=ice0, ice_config=SeaIceConfig(), atm=atm,
        ocean_sst_K=sst_K, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
    )
    A_new = np.clip(np.asarray(new_ice.concentration.data), 0.0, 1.0)
    # The step actually changed the concentration (else the time level is
    # untestable here) — lead freeze must have formed ice from open water.
    assert float(A_new.max()) > 1.0e-6, "no ice formed; freezing-case broken"
    f_open = 1.0 - A_new
    np.testing.assert_allclose(np.asarray(sf.sw_down),
                               f_open * np.asarray(sf0.sw_down), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fw.evap),
                               f_open * np.asarray(fw0.evap), rtol=1e-12)


def test_wrapper_masks_land_cells():
    """ocean_mask threaded through the wrapper: land cells (mask=0) keep the
    FULL open-ocean forcing (f_open=1) and receive zero ice->ocean exchange,
    regardless of what the (unmasked) ice step produced there.  Guards the
    --jra55-sea-ice land-mask routing (run_omip.py passes
    state_in.land_mask.data)."""
    sf0, fw0 = _open_ocean_forcing()
    sst_K = jnp.full(SHAPE, constants.T_freeze_ocean)
    ice0 = _ice_state(conc=0.8, h_ice=2.0, T_ice=255.0)
    mask = jnp.ones(SHAPE).at[:2, :].set(0.0)   # top half land
    land = np.asarray(mask) == 0.0
    new_ice, fw, sf = omip_sea_ice_surface_forcing(
        ice_state=ice0, ice_config=SeaIceConfig(), atm=_atm(250.0),
        ocean_sst_K=sst_K, open_ocean_sf=sf0, open_ocean_fw=fw0, dt=DT,
        ocean_mask=mask,
    )
    # Land: no ice->ocean exchange, full-cell open forcing untouched.
    np.testing.assert_allclose(np.asarray(fw.ice_fw)[land], 0.0)
    np.testing.assert_allclose(np.asarray(sf.salt_flux)[land], 0.0)
    np.testing.assert_allclose(np.asarray(sf.q_net)[land],
                               np.asarray(sf0.q_net)[land], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(sf.tau_x)[land],
                               np.asarray(sf0.tau_x)[land], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fw.evap)[land],
                               np.asarray(fw0.evap)[land], rtol=1e-12)
    # Ocean rows still partitioned (ice persists under the cold forcing).
    A_sea = np.asarray(new_ice.concentration.data)[~land]
    assert float(A_sea.max()) > 0.5
    assert not np.allclose(np.asarray(sf.q_net)[~land],
                           np.asarray(sf0.q_net)[~land])


# ===========================================================================
# blend_ice_ocean_forcing — direct-blend contracts (controlled TileResponse,
# no ice step => exact assertions).
# ===========================================================================
def _tile_response(shape, *, salt=0.0, heat=0.0, ice_fw=0.0, sx=0.0, sy=0.0):
    """TileResponse with only the ice->ocean channels set (per the verified
    conventions: heat/freshwater/salt PER-GRID-CELL, stress PER-ICE-TILE)."""
    z = jnp.zeros(shape)
    return TileResponse(
        T_sfc=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
        lhflx=z, tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z,
        co2_flux=z,
        freshwater_flux=jnp.full(shape, float(ice_fw)),
        ocean_heat_extraction=jnp.full(shape, float(heat)),
        ocean_stress_x=jnp.full(shape, float(sx)),
        ocean_stress_y=jnp.full(shape, float(sy)),
        surface_mass_flux=z,
        salt_flux=jnp.full(shape, float(salt)))


def test_blend_rejects_preowned_channels():
    """The blend OWNS sf.freshwater / sf.salt_flux: a caller pre-setting
    either raises loudly (silent overwrite could hide a double application
    or discard another supplied salt/freshwater source)."""
    resp = _tile_response(SHAPE)
    sf0, fw0 = _open_ocean_forcing()
    with pytest.raises(ValueError, match="freshwater"):
        blend_ice_ocean_forcing(
            ice_resp=resp, ice_concentration=jnp.zeros(SHAPE),
            open_sf=sf0._replace(freshwater=jnp.zeros(SHAPE)),
            open_fw=fw0)
    with pytest.raises(ValueError, match="salt_flux"):
        blend_ice_ocean_forcing(
            ice_resp=resp, ice_concentration=jnp.zeros(SHAPE),
            open_sf=sf0._replace(salt_flux=jnp.zeros(SHAPE)),
            open_fw=fw0)


def test_blend_preserves_untouched_channels():
    """Channels the blend does not own (e.g. ``chl`` for the RGB shortwave
    penetration) pass through untouched — the blend must never silently drop
    a caller-set field of the surface-forcing struct."""
    resp = _tile_response(SHAPE)
    sf0, fw0 = _open_ocean_forcing()
    chl = jnp.full(SHAPE, 0.2)
    fw, sf = blend_ice_ocean_forcing(
        ice_resp=resp, ice_concentration=jnp.full(SHAPE, 0.5),
        open_sf=sf0._replace(chl=chl), open_fw=fw0,
        ocean_mask=jnp.ones(SHAPE))
    assert sf.chl is chl


def test_blend_no_mask_equals_all_ocean_mask():
    """ocean_mask=None (legacy flat-bottom callers) == an all-ones mask."""
    resp = _tile_response(SHAPE, salt=1e-6, heat=5.0, ice_fw=2e-6, sx=0.01)
    sf0, fw0 = _open_ocean_forcing()
    conc = jnp.full(SHAPE, 0.4)
    fw_a, sf_a = blend_ice_ocean_forcing(
        ice_resp=resp, ice_concentration=conc,
        open_sf=sf0, open_fw=fw0, ocean_mask=None)
    fw_b, sf_b = blend_ice_ocean_forcing(
        ice_resp=resp, ice_concentration=conc,
        open_sf=sf0, open_fw=fw0, ocean_mask=jnp.ones(SHAPE))
    for a, b in ((fw_a.evap, fw_b.evap), (fw_a.ice_fw, fw_b.ice_fw),
                 (sf_a.q_net, sf_b.q_net), (sf_a.tau_x, sf_b.tau_x),
                 (sf_a.salt_flux, sf_b.salt_flux),
                 (sf_a.freshwater, sf_b.freshwater)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_jra55_builders_thread_land_mask_to_wrapper():
    """Source-wiring guard (AST-based): BOTH run_omip.py JRA55 block builders
    pass ocean_mask=state_in.land_mask.data to omip_sea_ice_surface_forcing —
    the land-bearing lat-lon-bathy/tripole lanes must never regress to the
    all-ocean default (spurious land-cell ice budgets at the blend)."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "scripts" / "run" / "run_omip.py").read_text()
    tree = ast.parse(src)
    builders = {"_build_jra55_block_fn", "_build_jra55_block_fn_interp"}
    seen = {}
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in builders):
            continue
        calls = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "omip_sea_ice_surface_forcing"
        ]
        assert len(calls) == 1, (
            f"{fn.name}: expected exactly 1 omip_sea_ice_surface_forcing "
            f"call, found {len(calls)}")
        kw = {k.arg: k.value for k in calls[0].keywords}
        assert "ocean_mask" in kw, (
            f"{fn.name}: sea-ice call lost its ocean_mask= land-mask routing")

        def _attr_chain(node):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))

        chain = _attr_chain(kw["ocean_mask"])
        assert chain == "state_in.land_mask.data", (
            f"{fn.name}: ocean_mask must be state_in.land_mask.data, "
            f"got {chain!r}")
        seen[fn.name] = True
    assert set(seen) == builders, (
        f"missing JRA55 builder(s) in run_omip.py: {builders - set(seen)}")

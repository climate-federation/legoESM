"""Forced-ocean (OMIP) prognostic sea-ice surface-forcing partitioning.

``blend_omip_ice_ocean_forcing`` is the ONE shared, mask-aware partitioning:
open-ocean heat / SW / stress / evaporation scale by ``f_open = 1 - A`` (a
single documented ice-concentration time level), each ice->ocean exchange
(basal heat, melt/freeze freshwater, brine salt, ice-ocean stress) enters
exactly once, precip + runoff stay full-cell, and land cells receive no
ice->ocean forcing.  ``omip_sea_ice_surface_forcing`` is the step-then-blend
wrapper.  These tests pin the partition contract (spec tests 1-6):

1. ice-free: every atmospheric channel bit-identical to the open input;
2. fully ice-covered: NO direct open-water stress/evap/heat/SW reaches ocean;
3. partial ice partitions all four channels with the SAME f_open;
4. land cells receive no ice->ocean forcing (open forcing passes through —
   the core's land mask owns those cells, as on the ice-free path);
5. the KPP/vmix buoyancy channel (``sf.freshwater``) carries the PHYSICAL
   net freshwater (P - E + R + ice), EXCLUDING the numerical SSS-restoring
   flux, while the mass channels stay on ``fw`` for the single in-core
   application;
6. brine real salt travels ONLY on ``sf.salt_flux`` (never the freshwater /
   buoyancy-freshwater channels).

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
    blend_omip_ice_ocean_forcing,
    omip_sea_ice_surface_forcing,
)
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import SeaIceState
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    net_freshwater_flux,
    physical_net_freshwater_flux,
)
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
    # KPP/vmix buoyancy channel: the PHYSICAL net freshwater (P - E + R + 0)
    # — populated even ice-free (the defect-3 fix: the boundary-layer closure
    # must see the freshwater buoyancy signal the mass path applies).
    assert np.allclose(np.asarray(sf.freshwater),
                       np.asarray(physical_net_freshwater_flux(fw)))


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
    documented time level).  Grow ice from OPEN WATER in one step (freezing
    SST, very cold air, NO shortwave, weak LW down -> the open-water surface
    energy balance Q_sfc < 0 -> lead freeze raises A from 0), then check the
    pure f_open-scaled channels (sw_down, evap — no ice-term addition)
    against exactly (1 - A_new).  A regression to the PRE-step concentration
    would leave them unscaled (f_open = 1 - 0 = 1) and fail the exact match."""
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


# ===========================================================================
# blend_omip_ice_ocean_forcing — the ONE shared, mask-aware partitioning.
# Controlled TileResponse (no ice step) => EXACT partition assertions.
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


def _blend(conc, resp, mask=None, restoring=0.0):
    sf0, fw0 = _open_ocean_forcing()
    if restoring != 0.0:
        fw0 = fw0._replace(restoring=jnp.full(SHAPE, float(restoring)))
    fw, sf = blend_omip_ice_ocean_forcing(
        ice_resp=resp, ice_concentration=jnp.full(SHAPE, float(conc)),
        open_ocean_sf=sf0, open_ocean_fw=fw0,
        ocean_mask=mask)
    return sf0, fw0, fw, sf


def test_blend_ice_free_bit_identical_open_channels():
    """(1) A = 0: every atmospheric channel is BIT-identical to the open
    input (heat, SW, stress, evap, precip, runoff); the ice channels are zero;
    the KPP buoyancy channel carries exactly the physical P - E + R."""
    resp = _tile_response(SHAPE)
    mask = jnp.ones(SHAPE).at[0, 0].set(0.0)     # include a land cell
    sf0, fw0, fw, sf = _blend(0.0, resp, mask=mask)
    for got, want in ((sf.q_net, sf0.q_net), (sf.sw_down, sf0.sw_down),
                      (sf.tau_x, sf0.tau_x), (sf.tau_y, sf0.tau_y),
                      (fw.evap, fw0.evap), (fw.precip, fw0.precip),
                      (fw.runoff, fw0.runoff)):
        assert np.array_equal(np.asarray(got), np.asarray(want))
    assert np.all(np.asarray(fw.ice_fw) == 0.0)
    assert np.all(np.asarray(sf.salt_flux) == 0.0)
    np.testing.assert_allclose(
        np.asarray(sf.freshwater),
        np.asarray(fw0.precip - fw0.evap + fw0.runoff))


def test_blend_full_ice_blocks_all_direct_atmospheric_forcing():
    """(2) A = 1 on wet cells: NO direct open-water stress / evaporation /
    heat / SW reaches the ocean; only the ice->ocean exchange does."""
    heat, salt, icefw, sx, sy = 15.0, -2.0e-6, 4.0e-6, 0.02, -0.01
    resp = _tile_response(SHAPE, salt=salt, heat=heat, ice_fw=icefw,
                          sx=sx, sy=sy)
    sf0, fw0, fw, sf = _blend(1.0, resp, mask=jnp.ones(SHAPE))
    # Open-water parts fully suppressed; ice exchange is all that remains.
    np.testing.assert_allclose(np.asarray(sf.q_net), -heat)      # basal draw
    np.testing.assert_allclose(np.asarray(sf.sw_down), 0.0)      # no SW
    np.testing.assert_allclose(np.asarray(sf.tau_x), -sx)        # ice stress
    np.testing.assert_allclose(np.asarray(sf.tau_y), -sy)        # (atm conv.)
    np.testing.assert_allclose(np.asarray(fw.evap), 0.0)         # no evap
    # Full-cell (no-snow-reservoir) channels are untouched.
    assert np.array_equal(np.asarray(fw.precip), np.asarray(fw0.precip))
    assert np.array_equal(np.asarray(fw.runoff), np.asarray(fw0.runoff))
    np.testing.assert_allclose(np.asarray(fw.ice_fw), icefw)
    np.testing.assert_allclose(np.asarray(sf.salt_flux), salt)


def test_blend_partial_ice_partitions_all_four_channels():
    """(3) A = 0.6: heat, SW, stress AND evaporation all scale by the SAME
    f_open = 0.4 (one concentration time level), each ice term added once with
    its verified area convention (exchange per-grid-cell; stress x f_ice)."""
    A = 0.6
    heat, salt, icefw, sx, sy = 12.0, 3.0e-6, 2.0e-6, 0.05, -0.02
    resp = _tile_response(SHAPE, salt=salt, heat=heat, ice_fw=icefw,
                          sx=sx, sy=sy)
    sf0, fw0, fw, sf = _blend(A, resp, mask=jnp.ones(SHAPE))
    f_open = 1.0 - A
    np.testing.assert_allclose(np.asarray(sf.q_net),
                               f_open * np.asarray(sf0.q_net) - heat)
    np.testing.assert_allclose(np.asarray(sf.sw_down),
                               f_open * np.asarray(sf0.sw_down))
    # Stress: open atmospheric stress x f_open PLUS the per-ice-tile stress
    # weighted by f_ice = A, sign-flipped to the atmospheric convention the
    # core's -tau consumer expects (force ON ocean = +A*stress).
    np.testing.assert_allclose(np.asarray(sf.tau_x),
                               f_open * np.asarray(sf0.tau_x) - A * sx)
    np.testing.assert_allclose(np.asarray(sf.tau_y),
                               f_open * np.asarray(sf0.tau_y) - A * sy)
    np.testing.assert_allclose(np.asarray(fw.evap),
                               f_open * np.asarray(fw0.evap))
    # Exchange channels are per-grid-cell (f_water = 1): no extra A factor.
    np.testing.assert_allclose(np.asarray(fw.ice_fw), icefw)
    np.testing.assert_allclose(np.asarray(sf.salt_flux), salt)
    assert np.array_equal(np.asarray(fw.precip), np.asarray(fw0.precip))
    assert np.array_equal(np.asarray(fw.runoff), np.asarray(fw0.runoff))


def test_blend_land_cells_receive_no_ice_forcing():
    """(4) land (mask = 0) cells: every ice->ocean term is ZERO and the open
    forcing passes through UNCHANGED (the ocean core's land mask owns those
    cells, exactly as on the ice-free path)."""
    heat, salt, icefw, sx = 20.0, 5.0e-6, 1.0e-5, 0.03
    resp = _tile_response(SHAPE, salt=salt, heat=heat, ice_fw=icefw, sx=sx)
    mask = jnp.ones(SHAPE).at[0, :].set(0.0)      # first row land
    sf0, fw0, fw, sf = _blend(0.5, resp, mask=mask)
    land = np.asarray(mask) < 0.5
    ocean = ~land
    # Land: ice terms zero, open forcing untouched.
    np.testing.assert_allclose(np.asarray(sf.q_net)[land],
                               np.asarray(sf0.q_net)[land])
    np.testing.assert_allclose(np.asarray(sf.tau_x)[land],
                               np.asarray(sf0.tau_x)[land])
    np.testing.assert_allclose(np.asarray(sf.sw_down)[land],
                               np.asarray(sf0.sw_down)[land])
    np.testing.assert_allclose(np.asarray(fw.evap)[land],
                               np.asarray(fw0.evap)[land])
    assert np.all(np.asarray(fw.ice_fw)[land] == 0.0)
    assert np.all(np.asarray(sf.salt_flux)[land] == 0.0)
    # Ocean: partitioned + ice terms present.
    np.testing.assert_allclose(np.asarray(sf.q_net)[ocean],
                               0.5 * np.asarray(sf0.q_net)[ocean] - heat)
    assert np.all(np.asarray(fw.ice_fw)[ocean] != 0.0)
    assert np.all(np.asarray(sf.salt_flux)[ocean] != 0.0)


def test_blend_kpp_channel_is_physical_net_excluding_restoring():
    """(5) the KPP/vmix buoyancy channel == P - E_open + R + ice_fw of the
    BLENDED freshwater, EXCLUDES the numerical SSS-restoring flux, and the
    mass channels remain on ``fw`` (single in-core application by contract:
    net_freshwater_flux(fw) - restoring == sf.freshwater exactly)."""
    A = 0.3
    resp = _tile_response(SHAPE, ice_fw=6.0e-6)
    sf0, fw0, fw, sf = _blend(A, resp, mask=jnp.ones(SHAPE),
                              restoring=7.5e-6)
    expect = np.asarray(physical_net_freshwater_flux(fw))
    np.testing.assert_allclose(np.asarray(sf.freshwater), expect)
    # Restoring travels on fw (mass path) but NOT in the buoyancy signal.
    np.testing.assert_allclose(
        np.asarray(net_freshwater_flux(fw)) - np.asarray(fw.restoring),
        np.asarray(sf.freshwater))
    assert not np.allclose(np.asarray(net_freshwater_flux(fw)),
                           np.asarray(sf.freshwater))
    # The buoyancy channel saw the partitioned evap + the ice freshwater.
    np.testing.assert_allclose(
        expect,
        np.asarray(fw0.precip) - (1.0 - A) * np.asarray(fw0.evap)
        + np.asarray(fw0.runoff) + 6.0e-6)


def test_blend_brine_salt_only_on_salt_flux_channel():
    """(6) brine real salt: travels ONLY on sf.salt_flux (applied once by the
    core's real-salt path); it never leaks into the freshwater mass channels
    or the KPP freshwater-buoyancy channel."""
    resp = _tile_response(SHAPE, salt=-4.0e-6)     # freeze: brine into ice
    sf0, fw0, fw, sf = _blend(0.5, resp, mask=jnp.ones(SHAPE))
    np.testing.assert_allclose(np.asarray(sf.salt_flux), -4.0e-6)
    # Freshwater channels see NO salt: identical to the salt-free blend.
    _, _, fw_ns, sf_ns = _blend(0.5, _tile_response(SHAPE),
                                mask=jnp.ones(SHAPE))
    assert np.array_equal(np.asarray(fw.ice_fw), np.asarray(fw_ns.ice_fw))
    assert np.array_equal(np.asarray(sf.freshwater),
                          np.asarray(sf_ns.freshwater))
    assert np.array_equal(np.asarray(fw.precip), np.asarray(fw_ns.precip))
    assert np.array_equal(np.asarray(fw.evap), np.asarray(fw_ns.evap))


def test_blend_rejects_preowned_channels():
    """The blend OWNS sf.freshwater / sf.salt_flux: a caller pre-setting
    either raises loudly (silent overwrite could hide a double application)."""
    resp = _tile_response(SHAPE)
    sf0, fw0 = _open_ocean_forcing()
    with pytest.raises(ValueError, match="freshwater"):
        blend_omip_ice_ocean_forcing(
            ice_resp=resp, ice_concentration=jnp.zeros(SHAPE),
            open_ocean_sf=sf0._replace(freshwater=jnp.zeros(SHAPE)),
            open_ocean_fw=fw0)
    with pytest.raises(ValueError, match="salt_flux"):
        blend_omip_ice_ocean_forcing(
            ice_resp=resp, ice_concentration=jnp.zeros(SHAPE),
            open_ocean_sf=sf0._replace(salt_flux=jnp.zeros(SHAPE)),
            open_ocean_fw=fw0)


def test_blend_preserves_untouched_channels():
    """Channels the blend does not own (e.g. ``chl`` for the RGB shortwave
    penetration) pass through untouched — the blend must never silently drop
    a caller-set field of the surface-forcing struct."""
    resp = _tile_response(SHAPE)
    sf0, fw0 = _open_ocean_forcing()
    chl = jnp.full(SHAPE, 0.2)
    fw, sf = blend_omip_ice_ocean_forcing(
        ice_resp=resp, ice_concentration=jnp.full(SHAPE, 0.5),
        open_ocean_sf=sf0._replace(chl=chl), open_ocean_fw=fw0,
        ocean_mask=jnp.ones(SHAPE))
    assert sf.chl is chl


def test_blend_no_mask_equals_all_ocean_mask():
    """ocean_mask=None (legacy flat-bottom callers) == an all-ones mask."""
    resp = _tile_response(SHAPE, salt=1e-6, heat=5.0, ice_fw=2e-6, sx=0.01)
    _, _, fw_a, sf_a = _blend(0.4, resp, mask=None)
    _, _, fw_b, sf_b = _blend(0.4, resp, mask=jnp.ones(SHAPE))
    for a, b in ((fw_a.evap, fw_b.evap), (fw_a.ice_fw, fw_b.ice_fw),
                 (sf_a.q_net, sf_b.q_net), (sf_a.tau_x, sf_b.tau_x),
                 (sf_a.salt_flux, sf_b.salt_flux),
                 (sf_a.freshwater, sf_b.freshwater)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

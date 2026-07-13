"""Prognostic sea-ice wiring in the OMIP CORE-II runner (--prognostic-sea-ice).

The OMIP runner (scripts/run/run_omip_core2.py) wires legoESM's REAL prognostic
sea-ice model (``legoesm.ice.step_sea_ice``) in place of the freeze-floor /
prescribed-siconc / ice-thermo-relaxation surrogates, so the ice tile's
brine-rejection salt flux + melt/freeze freshwater + ocean-heat extraction
balance the Arctic river runoff (the missing reservoir behind the Arctic SSS
crash).  This is pure INTEGRATION GLUE: no new sea-ice physics, no new ocean
salt/freshwater applicator.  These tests pin the glue contract:

(a) the AtmToSurface builder maps the sampled CORE-II fields correctly
    (shapes / units / air T in Kelvin);
(b) one ``step_sea_ice`` call on a tiny synthetic MPAS (Voronoi) grid returns a
    TileResponse with finite ``salt_flux`` / ``freshwater_flux`` /
    ``ocean_heat_extraction``;
(c) the ONE shared, mask-aware partition (``legoesm.coupler.ocean_forcing.
    blend_ice_ocean_forcing``, raw_core2 mode) scales open-water stress /
    evaporation / heat / SW by ``f_open = 1 - A`` and adds the ice basal heat,
    brine salt, melt/freeze freshwater, and ice stress exactly once; the ocean
    core's EXISTING salt application (``salt_flux_salinity_tendency`` — the
    literal function ``ocean_pe_mpas`` calls) moves SSS the expected direction;
(d) the --prognostic-sea-ice mutual-exclusion / prerequisite validation.

Fast + synthetic; run under JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.coupler.ocean_forcing import blend_ice_ocean_forcing
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    net_freshwater_flux,
    salt_flux_salinity_tendency,
)

from scripts.run import run_omip_core2 as R


# ---------------------------------------------------------------------------
# Synthetic CORE-II forcing sample (what sample_omip2_forcing returns: a dict
# of channels ALREADY on the model grid).  Air T is Kelvin per the OceanForcing
# contract.
# ---------------------------------------------------------------------------
def _synthetic_sampled_forcing(shape):
    rng = np.random.default_rng(0)
    return {
        "u10": np.full(shape, 6.0),
        "v10": np.full(shape, -2.0),
        "T_air": np.full(shape, 270.0),          # K (cold polar air)
        "q_air": np.full(shape, 2.0e-3),
        "sw_down": np.full(shape, 120.0),
        "lw_down": np.full(shape, 250.0),
        "precip": np.full(shape, 1.0e-5),
        "snow": np.full(shape, 4.0e-6),
        "slp": np.full(shape, 1.01e5),
    }


# ===========================================================================
# (a) AtmToSurface builder: field mapping, shapes, units, T in Kelvin
# ===========================================================================
def test_atm_to_surface_builder_maps_core2_fields():
    shape = (37,)
    forc = _synthetic_sampled_forcing(shape)
    atm = R._build_atm_to_surface_core2(forc, ramp=1.0)

    assert isinstance(atm, AtmToSurface)
    # Every field carries the grid shape.
    for name in ("sw_down", "lw_down", "precip_total", "precip_snow",
                 "T_lowest", "q_lowest", "u_lowest", "v_lowest",
                 "p_lowest", "p_surface", "rho_lowest"):
        assert getattr(atm, name).shape == shape, name

    # Direct mappings (units preserved).
    np.testing.assert_allclose(np.asarray(atm.u_lowest), forc["u10"])
    np.testing.assert_allclose(np.asarray(atm.v_lowest), forc["v10"])
    np.testing.assert_allclose(np.asarray(atm.q_lowest), forc["q_air"])
    np.testing.assert_allclose(np.asarray(atm.sw_down), forc["sw_down"])
    np.testing.assert_allclose(np.asarray(atm.lw_down), forc["lw_down"])
    np.testing.assert_allclose(np.asarray(atm.precip_total), forc["precip"])
    np.testing.assert_allclose(np.asarray(atm.precip_snow), forc["snow"])
    # slp -> p_surface == p_lowest.
    np.testing.assert_allclose(np.asarray(atm.p_surface), forc["slp"])
    np.testing.assert_allclose(np.asarray(atm.p_lowest), forc["slp"])

    # CRITICAL: air T must be Kelvin (CORE-II is already K), NOT degC -> never
    # T_freeze-shifted by the builder.
    np.testing.assert_allclose(np.asarray(atm.T_lowest), forc["T_air"])
    assert float(atm.T_lowest[0]) > 200.0   # plainly Kelvin, not ~ -3 degC

    # rho = p / (R_d * T_v), T_v = T*(1 + (1/eps - 1) q) — physical air density.
    T_v = forc["T_air"][0] * (1.0 + (1.0 / constants.epsilon - 1.0)
                              * forc["q_air"][0])
    rho_expect = forc["slp"][0] / (constants.R_d * T_v)
    np.testing.assert_allclose(float(atm.rho_lowest[0]), rho_expect, rtol=1e-12)
    assert 1.0 < float(atm.rho_lowest[0]) < 1.6  # sane sea-level air density

    assert float(atm.has_radiation) == 1.0
    assert float(atm.has_precipitation) == 1.0


def test_atm_to_surface_builder_ramp_scales_fluxes_not_state():
    shape = (12,)
    forc = _synthetic_sampled_forcing(shape)
    full = R._build_atm_to_surface_core2(forc, ramp=1.0)
    half = R._build_atm_to_surface_core2(forc, ramp=0.5)
    # Flux fields scale with the cold-start ramp.
    np.testing.assert_allclose(np.asarray(half.sw_down),
                               0.5 * np.asarray(full.sw_down))
    np.testing.assert_allclose(np.asarray(half.precip_total),
                               0.5 * np.asarray(full.precip_total))
    np.testing.assert_allclose(np.asarray(half.precip_snow),
                               0.5 * np.asarray(full.precip_snow))
    # State fields (winds, T, p) are NOT ramped.
    np.testing.assert_allclose(np.asarray(half.u_lowest),
                               np.asarray(full.u_lowest))
    np.testing.assert_allclose(np.asarray(half.T_lowest),
                               np.asarray(full.T_lowest))


def test_atm_to_surface_builder_handles_missing_snow_slp():
    """Cache without snow/slp -> precip_snow=0, p_surface=standard atmosphere
    (the same fallbacks the heat/momentum path uses)."""
    shape = (8,)
    forc = _synthetic_sampled_forcing(shape)
    del forc["snow"]
    del forc["slp"]
    atm = R._build_atm_to_surface_core2(forc, ramp=1.0)
    np.testing.assert_allclose(np.asarray(atm.precip_snow), 0.0)
    np.testing.assert_allclose(np.asarray(atm.p_surface),
                               float(constants.p_atm_std))


# ===========================================================================
# (b) one step_sea_ice on a tiny MPAS (Voronoi) grid -> finite ice->ocean fluxes
# ===========================================================================
@pytest.fixture(scope="module")
def _mpas_mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    # Level-2 icosahedral mesh (162 cells) — smallest global mesh.
    return create_voronoi_mesh(subdivision_level=2)


def test_step_sea_ice_on_mpas_returns_finite_ice_ocean_fluxes(_mpas_mesh):
    from legoesm.ice import (
        SeaIceConfig, init_dynamic_ice_state, step_sea_ice,
        grid_supports_ice_dynamics,
    )
    from legoesm.ice.config import BrineConfig

    mesh = _mpas_mesh
    assert grid_supports_ice_dynamics(mesh)          # Voronoi is supported
    nCells = int(mesh.nCells)
    shape = R._ice_state_spatial_shape(mesh, "mpas")
    assert shape == (nCells,)

    cfg = SeaIceConfig(dynamics="mevp", transport="advect",
                       brine=BrineConfig(enabled=True))
    ice_state = init_dynamic_ice_state(shape, S_ice_init=0.0)

    # Seed a thin existing ice cover in the (cold) northern cap so the brine /
    # basal-melt channels are exercised on step 1 (a pure zero-ice cold start
    # would return all-zero ice fluxes on the very first step).
    latC = np.asarray(mesh.latCell)
    seed = (latC > np.deg2rad(60.0)).astype(np.float64)
    ice_state = ice_state._replace(
        h_ice=ice_state.h_ice.replace(
            data=jnp.asarray(0.5 * seed)),                # 0.5 m where icy
        concentration=ice_state.concentration.replace(
            data=jnp.asarray(0.8 * seed)),
        S_ice=ice_state.S_ice.replace(
            data=jnp.asarray(4.0 * seed)),                # 4 PSU bulk salinity
    )

    forc = _synthetic_sampled_forcing(shape)
    atm = R._build_atm_to_surface_core2(forc, ramp=1.0)
    # Ocean: warm-ish sub-ice water (drives basal melt -> ocean heat extraction)
    # and ocean SST in KELVIN (the K-conversion the runner applies).
    sst_K = jnp.full(shape, constants.T_freeze_ocean + 0.5)   # ~ -0.85 degC
    ocn_u = jnp.zeros(shape)
    ocn_v = jnp.zeros(shape)

    new_state, resp = step_sea_ice(
        ice_state, atm, sst_K, ocn_u, ocn_v, cfg, U_min=0.0, dt=3600.0,
        grid=mesh)

    assert isinstance(resp, TileResponse)
    for name in ("salt_flux", "freshwater_flux", "ocean_heat_extraction"):
        arr = np.asarray(getattr(resp, name))
        assert arr.shape == shape, name
        assert np.all(np.isfinite(arr)), f"{name} not finite"
    # The ice tile actually exchanged with the ocean somewhere (non-trivial).
    assert np.any(np.asarray(resp.salt_flux) != 0.0) \
        or np.any(np.asarray(resp.freshwater_flux) != 0.0) \
        or np.any(np.asarray(resp.ocean_heat_extraction) != 0.0)
    # State carries forward finite.
    assert np.all(np.isfinite(np.asarray(new_state.h_ice.data)))
    assert np.all(np.isfinite(np.asarray(new_state.concentration.data)))


def test_surface_currents_mpas_reconstructs_cell_centres(_mpas_mesh):
    """ocean_u / ocean_v for the ice step come from the canonical Perot
    reconstruction of the edge-normal MPAS surface velocity -> (nCells,)."""
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.vertical import create_ocean_z_star
    mesh = _mpas_mesh
    zc = create_ocean_z_star(n_levels=4, H_max=400.0, dz_surface=20.0,
                             dz_deep=200.0)
    state = rest_state_mpas_ocean(mesh, zc, H_max=400.0)
    u_sfc, v_sfc = R._surface_currents(state, mesh, "mpas")
    assert u_sfc.shape == (mesh.nCells,)
    assert v_sfc.shape == (mesh.nCells,)
    # Rest state -> zero currents.
    np.testing.assert_allclose(np.asarray(u_sfc), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(v_sfc), 0.0, atol=1e-12)


# ===========================================================================
# (c) the ONE shared, mask-aware open-water/ice partition
#     (legoesm.coupler.ocean_forcing.blend_ice_ocean_forcing, raw_core2 mode —
#     what the runner's prognostic-ice path calls after step_sea_ice), and the
#     ocean core's EXISTING salt application moving SSS the expected direction.
# ===========================================================================
_ALPHA_OC = float(constants.alpha_ocean_broadband)
_TAU_ICE_SW = 0.03


def _zero_tile(shape):
    z = jnp.zeros(shape)
    return TileResponse(
        T_sfc=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
        lhflx=z, tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z,
        co2_flux=z, freshwater_flux=z, ocean_heat_extraction=z,
        ocean_stress_x=z, ocean_stress_y=z, surface_mass_flux=z, salt_flux=z)


def _open_forcing(shape):
    """UNMASKED full-cell open-ocean forcing exactly as the runner builds it for
    the prognostic-ice path: raw sw_down (no albedo), q_net = q_non_sw + sw_down."""
    swd = jnp.full(shape, 80.0)
    q_non_sw = jnp.full(shape, -110.0)
    sf = OceanSurfaceForcing(
        tau_x=jnp.full(shape, 0.08), tau_y=jnp.full(shape, -0.03),
        q_net=q_non_sw + swd, sw_down=swd)
    fw = FreshwaterForcing(
        precip=jnp.full(shape, 1e-5), evap=jnp.full(shape, 3e-6),
        runoff=jnp.full(shape, 2e-6), ice_fw=jnp.zeros(shape),
        restoring=jnp.zeros(shape))
    return sf, fw, swd, q_non_sw


def _ice_response(shape):
    return _zero_tile(shape)._replace(
        salt_flux=jnp.full(shape, -2.0e-6),   # brine reject: salt LEAVES ocean
        ocean_heat_extraction=jnp.full(shape, 15.0),  # ocean LOSES 15 W/m2
        freshwater_flux=jnp.full(shape, 4.0e-6),      # melt fw INTO ocean
        ocean_stress_x=jnp.full(shape, 0.02),  # ice->ocean stress (ON ocean)
        ocean_stress_y=jnp.full(shape, -0.01))


def test_blend_partial_ice_partitions_stress_evap_heat_sw():
    """Partial cover A: open-water stress/evap/heat/SW scale with f_open=1-A;
    ice basal heat, brine salt, melt freshwater, and ice stress added exactly
    once (the spec's intended partition — the OLD wiring left FULL open-water
    stress + evaporation acting under ice and only attenuated heat/SW)."""
    shape = (10,)
    sf, fw, swd, q_non_sw = _open_forcing(shape)
    resp = _ice_response(shape)
    A = 0.7
    conc = jnp.full(shape, A)

    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp, ice_concentration=conc,
        ocean_mask=jnp.ones(shape), sw_partition="raw_core2",
        alpha_ocean=_ALPHA_OC, sw_transmittance_ice=_TAU_ICE_SW)

    f_open = 1.0 - A
    # STRESS: open-water tau x f_open + ice back-reaction -A*stress (F11: the
    # core's -tau consumer then applies +A*stress force on the ocean).
    np.testing.assert_allclose(
        np.asarray(sf2.tau_x),
        f_open * np.asarray(sf.tau_x) - A * np.asarray(resp.ocean_stress_x))
    np.testing.assert_allclose(
        np.asarray(sf2.tau_y),
        f_open * np.asarray(sf.tau_y) - A * np.asarray(resp.ocean_stress_y))
    # EVAPORATION: open-water only.
    np.testing.assert_allclose(np.asarray(fw2.evap), f_open * np.asarray(fw.evap))
    # Precip + runoff stay FULL-CELL (no-snow-reservoir policy).
    np.testing.assert_allclose(np.asarray(fw2.precip), np.asarray(fw.precip))
    np.testing.assert_allclose(np.asarray(fw2.runoff), np.asarray(fw.runoff))
    # SW: open fraction gets the ocean-albedoed SW, ice fraction the small
    # transmittance; heat: sw_ocean + f_open*q_non_sw - basal extraction.
    sw_expect = np.asarray(swd) * (f_open * (1.0 - _ALPHA_OC) + A * _TAU_ICE_SW)
    np.testing.assert_allclose(np.asarray(sf2.sw_down), sw_expect)
    np.testing.assert_allclose(
        np.asarray(sf2.q_net),
        sw_expect + f_open * np.asarray(q_non_sw)
        - np.asarray(resp.ocean_heat_extraction))
    # Brine salt + melt freshwater each exactly once.
    np.testing.assert_allclose(np.asarray(sf2.salt_flux), np.asarray(resp.salt_flux))
    np.testing.assert_allclose(np.asarray(fw2.ice_fw),
                               np.asarray(resp.freshwater_flux))
    # KPP buoyancy channel = the blended physical net freshwater.
    np.testing.assert_allclose(np.asarray(sf2.freshwater),
                               np.asarray(net_freshwater_flux(fw2)))


def test_blend_full_ice_no_open_water_atmosphere():
    """A=1: NO direct open-water atmospheric stress, evaporation, or bulk
    heat/SW reaches the ocean — only the transmitted SW + the ice exchange."""
    shape = (7,)
    sf, fw, swd, q_non_sw = _open_forcing(shape)
    resp = _ice_response(shape)
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp,
        ice_concentration=jnp.ones(shape), ocean_mask=jnp.ones(shape),
        sw_partition="raw_core2", alpha_ocean=_ALPHA_OC,
        sw_transmittance_ice=_TAU_ICE_SW)
    # Stress is the ice stress ALONE (no trace of the 0.08 Pa wind stress).
    np.testing.assert_allclose(np.asarray(sf2.tau_x),
                               -np.asarray(resp.ocean_stress_x))
    np.testing.assert_allclose(np.asarray(sf2.tau_y),
                               -np.asarray(resp.ocean_stress_y))
    # Evaporation fully suppressed.
    np.testing.assert_allclose(np.asarray(fw2.evap), 0.0)
    # Only the small transmitted SW + the basal draw: no open q_non_sw at all.
    np.testing.assert_allclose(np.asarray(sf2.sw_down),
                               _TAU_ICE_SW * np.asarray(swd))
    np.testing.assert_allclose(
        np.asarray(sf2.q_net),
        _TAU_ICE_SW * np.asarray(swd)
        - np.asarray(resp.ocean_heat_extraction))


def test_blend_ice_free_bit_identical_dynamics():
    """A=0: stress, evaporation, precip/runoff, and the heat PARTITION reduce to
    the pre-existing open-ocean values (sw carries the standard open-water
    albedo exactly as the old under-ice attenuation gave at sic=0).  The KPP
    freshwater channel is the ONE deliberate change (fix B) — asserted last."""
    shape = (5,)
    sf, fw, swd, q_non_sw = _open_forcing(shape)
    resp = _zero_tile(shape)   # no ice -> zero ice response
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp,
        ice_concentration=jnp.zeros(shape), ocean_mask=jnp.ones(shape),
        sw_partition="raw_core2", alpha_ocean=_ALPHA_OC,
        sw_transmittance_ice=_TAU_ICE_SW)
    np.testing.assert_allclose(np.asarray(sf2.tau_x), np.asarray(sf.tau_x))
    np.testing.assert_allclose(np.asarray(sf2.tau_y), np.asarray(sf.tau_y))
    np.testing.assert_allclose(np.asarray(fw2.evap), np.asarray(fw.evap))
    np.testing.assert_allclose(np.asarray(fw2.precip), np.asarray(fw.precip))
    np.testing.assert_allclose(np.asarray(fw2.runoff), np.asarray(fw.runoff))
    np.testing.assert_allclose(np.asarray(fw2.ice_fw), 0.0)
    np.testing.assert_allclose(np.asarray(sf2.salt_flux), 0.0)
    # Heat partition at sic=0 == the legacy under-ice attenuation at sic=0.
    from legoesm.ocean.coupler.omip2_applicator import _ice_surface_heat
    sw_ref, q_ref = _ice_surface_heat(
        np.asarray(swd), np.asarray(q_non_sw), np.zeros(shape),
        under_ice=True, tau_ice_sw=_TAU_ICE_SW)
    np.testing.assert_allclose(np.asarray(sf2.sw_down), sw_ref)
    np.testing.assert_allclose(np.asarray(sf2.q_net), q_ref)
    # Fix B: the KPP buoyancy channel now carries P-E+R (was None).
    np.testing.assert_allclose(
        np.asarray(sf2.freshwater),
        np.asarray(fw.precip - fw.evap + fw.runoff))


def test_blend_heat_partition_matches_legacy_attenuation_any_conc():
    """raw_core2 heat/SW == the pre-existing _ice_surface_heat(under_ice=True)
    attenuation at the SAME concentration, for arbitrary A in [0,1] (bit-parity
    of the heat path; the blend only ADDS the stress/evap partition + ice terms
    on top)."""
    from legoesm.ocean.coupler.omip2_applicator import _ice_surface_heat
    shape = (9,)
    sf, fw, swd, q_non_sw = _open_forcing(shape)
    rng = np.random.default_rng(3)
    conc = jnp.asarray(rng.uniform(0.0, 1.0, shape))
    resp = _zero_tile(shape)   # isolate the open-water heat partition
    _, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp, ice_concentration=conc,
        ocean_mask=jnp.ones(shape), sw_partition="raw_core2",
        alpha_ocean=_ALPHA_OC, sw_transmittance_ice=_TAU_ICE_SW)
    sw_ref, q_ref = _ice_surface_heat(
        np.asarray(swd), np.asarray(q_non_sw), np.asarray(conc),
        under_ice=True, tau_ice_sw=_TAU_ICE_SW)
    np.testing.assert_allclose(np.asarray(sf2.sw_down), sw_ref, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(sf2.q_net), q_ref, rtol=1e-12)


def test_blend_land_cells_receive_no_ice_forcing():
    """Land/dry cells (ocean_mask=0) get ZERO ice->ocean flux on every channel
    AND keep the full-cell open forcing (A masked to 0 there) — a spurious
    land-ice budget never reaches the ocean salt/heat/FW/tau."""
    shape = (6,)
    sf, fw, swd, q_non_sw = _open_forcing(shape)
    resp = _ice_response(shape)
    mask = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    conc = jnp.full(shape, 0.5)
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp, ice_concentration=conc,
        ocean_mask=mask, sw_partition="raw_core2",
        alpha_ocean=_ALPHA_OC, sw_transmittance_ice=_TAU_ICE_SW)
    land = np.asarray(mask) < 0.5
    ocean = ~land
    # All ice->ocean channels zero on land; open forcing NOT partitioned there.
    assert np.all(np.asarray(sf2.salt_flux)[land] == 0.0)
    assert np.all(np.asarray(fw2.ice_fw)[land] == 0.0)
    np.testing.assert_allclose(np.asarray(sf2.tau_x)[land],
                               np.asarray(sf.tau_x)[land])
    np.testing.assert_allclose(np.asarray(fw2.evap)[land],
                               np.asarray(fw.evap)[land])
    # Ocean cells DO receive the partition + ice flux.
    assert np.all(np.asarray(sf2.salt_flux)[ocean] != 0.0)
    assert np.all(np.asarray(fw2.ice_fw)[ocean] != 0.0)
    np.testing.assert_allclose(
        np.asarray(sf2.tau_x)[ocean],
        0.5 * np.asarray(sf.tau_x)[ocean]
        - 0.5 * np.asarray(resp.ocean_stress_x)[ocean])


def test_blend_kpp_freshwater_buoyancy_and_single_mass_application():
    """Fix B: the physical freshwater P-E+R+ice alters the KPP surface-buoyancy
    inputs via sf.freshwater, while the freshwater MASS stays on the fw struct
    exactly once (nothing about the buoyancy channel changes P/E/R/ice_fw).
    Positive net freshwater (fresh into ocean) STABILISES: B_f decreases."""
    from legoesm.ocean.physics.vertical_mixing._shared import (
        surface_buoyancy_flux,
    )
    shape = (4,)
    sf, fw, _, _ = _open_forcing(shape)
    resp = _ice_response(shape)
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp,
        ice_concentration=jnp.full(shape, 0.4), ocean_mask=jnp.ones(shape),
        sw_partition="raw_core2", alpha_ocean=_ALPHA_OC,
        sw_transmittance_ice=_TAU_ICE_SW)
    # The channel carries EXACTLY the blended net physical freshwater.
    np.testing.assert_allclose(np.asarray(sf2.freshwater),
                               np.asarray(net_freshwater_flux(fw2)))
    T0 = jnp.full(shape, 5.0)
    S0 = jnp.full(shape, 34.0)
    kw = dict(g=float(constants.g), rho_0=float(constants.rho_ocean),
              c_sw=float(constants.c_p_seawater), real_salt_in_qs=True)
    B_without, _, _ = surface_buoyancy_flux(sf2.q_net, None, None, T0, S0, **kw)
    B_with, _, QS = surface_buoyancy_flux(
        sf2.q_net, sf2.freshwater, None, T0, S0, **kw)
    assert QS is not None
    # Net freshwater here is positive (P+R+melt > partitioned E) -> stabilising.
    assert float(jnp.min(jnp.asarray(net_freshwater_flux(fw2)))) > 0.0
    assert np.all(np.asarray(B_with) < np.asarray(B_without))
    # Single mass application: the fw struct is the ONLY mass carrier and its
    # components are untouched by the buoyancy channel (P/R full-cell, E
    # partitioned once, ice_fw masked once) — no second freshwater source.
    np.testing.assert_allclose(np.asarray(fw2.precip), np.asarray(fw.precip))
    np.testing.assert_allclose(np.asarray(fw2.runoff), np.asarray(fw.runoff))
    np.testing.assert_allclose(np.asarray(fw2.evap), 0.6 * np.asarray(fw.evap))
    np.testing.assert_allclose(np.asarray(fw2.ice_fw),
                               np.asarray(resp.freshwater_flux))


def test_blend_brine_salt_applied_once_and_not_in_freshwater():
    """Fix: the brine REAL-salt flux lands on sf.salt_flux exactly once and
    never leaks into the freshwater (virtual-salt) mass channel."""
    shape = (5,)
    sf, fw, _, _ = _open_forcing(shape)
    resp = _zero_tile(shape)._replace(salt_flux=jnp.full(shape, -3.0e-6))
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp,
        ice_concentration=jnp.full(shape, 0.6), ocean_mask=jnp.ones(shape),
        sw_partition="raw_core2", alpha_ocean=_ALPHA_OC,
        sw_transmittance_ice=_TAU_ICE_SW)
    np.testing.assert_allclose(np.asarray(sf2.salt_flux),
                               np.asarray(resp.salt_flux))
    # No salt in any freshwater component (ice_fw here is zero: no melt).
    np.testing.assert_allclose(np.asarray(fw2.ice_fw), 0.0)
    np.testing.assert_allclose(np.asarray(net_freshwater_flux(fw2)),
                               np.asarray(fw.precip - 0.4 * fw.evap + fw.runoff))


def test_blend_nan_in_dry_cells_does_not_leak():
    """codex r2 #2: a NaN concentration / ice flux in a LAND cell must not
    poison the blended forcing (NaN * 0 == NaN, so masking must be jnp.where,
    not multiplication)."""
    shape = (4,)
    sf, fw, _, _ = _open_forcing(shape)
    nan = float("nan")
    conc = jnp.asarray([0.5, nan, 0.5, nan])
    mask = jnp.asarray([1.0, 0.0, 1.0, 0.0])   # NaNs only in land cells
    resp = _ice_response(shape)._replace(
        freshwater_flux=jnp.asarray([1e-6, nan, 1e-6, nan]),
        salt_flux=jnp.asarray([-1e-6, nan, -1e-6, nan]),
        ocean_heat_extraction=jnp.asarray([5.0, nan, 5.0, nan]),
        ocean_stress_x=jnp.asarray([0.01, nan, 0.01, nan]),
        ocean_stress_y=jnp.asarray([0.0, nan, 0.0, nan]))
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp, ice_concentration=conc,
        ocean_mask=mask, sw_partition="raw_core2",
        alpha_ocean=_ALPHA_OC, sw_transmittance_ice=_TAU_ICE_SW)
    for arr in (sf2.tau_x, sf2.tau_y, sf2.q_net, sf2.sw_down, sf2.salt_flux,
                sf2.freshwater, fw2.evap, fw2.ice_fw):
        assert np.all(np.isfinite(np.asarray(arr))), "NaN leaked through mask"
    # Ocean cells still get the real partition.
    assert np.asarray(sf2.salt_flux)[0] != 0.0
    # Land cells: no ice forcing, full-cell open values.
    np.testing.assert_allclose(np.asarray(sf2.tau_x)[1], np.asarray(sf.tau_x)[1])
    assert np.asarray(fw2.ice_fw)[1] == 0.0


def test_blend_rejects_unknown_sw_partition_and_missing_albedo():
    """Dispatch hardening: unknown sw_partition raises; raw_core2 without
    alpha_ocean raises (no silent default)."""
    shape = (3,)
    sf, fw, _, _ = _open_forcing(shape)
    resp = _zero_tile(shape)
    with pytest.raises(ValueError, match="unknown sw_partition"):
        blend_ice_ocean_forcing(
            open_sf=sf, open_fw=fw, ice_resp=resp,
            ice_concentration=jnp.zeros(shape), sw_partition="not_a_mode")
    with pytest.raises(ValueError, match="requires\\s+alpha_ocean"):
        blend_ice_ocean_forcing(
            open_sf=sf, open_fw=fw, ice_resp=resp,
            ice_concentration=jnp.zeros(shape), sw_partition="raw_core2")


def test_core_salt_application_raises_sss_for_positive_salt_flux():
    """The EXACT function the MPAS PE core applies (salt_flux_salinity_tendency,
    ocean_pe_mpas.py:914) turns a POSITIVE salt flux into a POSITIVE dS/dt — so
    SSS rises.  A NEGATIVE flux (brine rejection: salt locked into new ice)
    lowers it.  This is the channel the routing helper feeds via sf.salt_flux."""
    shape = (6,)
    dz_0 = jnp.full(shape, 10.0)              # 10 m top cell
    rho_0 = float(constants.rho_ocean)
    S0 = jnp.full(shape, 34.0)
    dt = 3600.0

    pos = jnp.full(shape, 5.0e-6)             # salt INTO ocean
    dS_pos = salt_flux_salinity_tendency(pos, dz_0, rho_0)
    assert np.all(np.asarray(dS_pos) > 0.0)
    S_pos = np.asarray(S0 + dt * dS_pos)
    assert np.all(S_pos > np.asarray(S0))     # SSS rose

    neg = jnp.full(shape, -5.0e-6)            # salt OUT of ocean (into ice)
    dS_neg = salt_flux_salinity_tendency(neg, dz_0, rho_0)
    assert np.all(np.asarray(dS_neg) < 0.0)
    S_neg = np.asarray(S0 + dt * dS_neg)
    assert np.all(S_neg < np.asarray(S0))     # SSS fell

    # Dry / thin top cell (dz < 1 mm) -> no salinity change (the core's guard).
    dS_dry = salt_flux_salinity_tendency(pos, jnp.zeros(shape), rho_0)
    np.testing.assert_allclose(np.asarray(dS_dry), 0.0)


# ===========================================================================
# (d) --prognostic-sea-ice validation (mutual exclusion + prerequisites)
# ===========================================================================
def _args(**over):
    """Minimal argparse-like namespace exercising the validation block."""
    import types
    base = dict(
        prognostic_sea_ice=True, prognostic_ice_dynamics="mevp",
        prognostic_ice_salinity=None,
        freeze_floor=False, ice_thermo=False, ice_albedo=False,
        ice_albedo_seasonal=False,
        woa_init=True, grid="mpas", scan_block=0,
        ice_thermo_sw_trans=0.03, ice_thermo_tau_days=20.0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _run_validation(args):
    """Replicate the runner's --prognostic-sea-ice validation block exactly."""
    if args.prognostic_sea_ice:
        _ice_surrogates = [
            ("--freeze-floor", args.freeze_floor),
            ("--ice-thermo", args.ice_thermo),
            ("--ice-albedo", args.ice_albedo),
            ("--ice-albedo-seasonal", args.ice_albedo_seasonal),
        ]
        _on = [name for name, val in _ice_surrogates if val]
        if _on:
            raise ValueError(f"mutually exclusive with {_on}")
        if not args.woa_init:
            raise ValueError("requires --woa-init")
        if args.grid == "cubed_sphere":
            raise ValueError("not wired for cubed_sphere")
        if int(args.scan_block) > 0:
            raise ValueError("cannot run under --scan-block")


@pytest.mark.parametrize("surrogate", [
    "freeze_floor", "ice_thermo", "ice_albedo", "ice_albedo_seasonal",
])
def test_prognostic_ice_rejects_each_surrogate(surrogate):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run_validation(_args(**{surrogate: True}))


def test_prognostic_ice_requires_woa_init():
    with pytest.raises(ValueError, match="woa-init"):
        _run_validation(_args(woa_init=False))


def test_prognostic_ice_rejects_cube():
    with pytest.raises(ValueError, match="cubed_sphere"):
        _run_validation(_args(grid="cubed_sphere"))


def test_prognostic_ice_rejects_scan_block():
    with pytest.raises(ValueError, match="scan-block"):
        _run_validation(_args(scan_block=4))


def test_prognostic_ice_valid_config_passes():
    # mpas + woa_init + no surrogates + no scan -> no raise.
    _run_validation(_args())


def test_validation_block_matches_runner_source():
    """Guard against drift: the runner's real validation must enforce the same
    four rules this test mirrors (mutual exclusion, woa-init, cube, scan)."""
    import inspect
    src = inspect.getsource(R.main)
    assert "prognostic-sea-ice is mutually exclusive" in src
    assert "--prognostic-sea-ice requires --woa-init" in src
    assert "not wired for --grid cubed_sphere" in src
    assert "cannot run under --scan-block" in src


def test_runner_partitions_after_ice_step_via_shared_blend():
    """Under --prognostic-sea-ice the runner must (a) build the UNMASKED
    full-cell open-ocean forcing (NO under-ice attenuation inside
    compute_omip2_surface_forcing — the partition may not be applied twice) and
    (b) hand the ENTIRE open-water/ice partition to the ONE shared helper
    (blend_ice_ocean_forcing, raw_core2 mode) AFTER step_sea_ice, at the
    PRE-step concentration (the state the ice integrated its atmospheric
    fluxes over — flux-conserving A + (1-A) = 1; codex r4 #1).
    Source-introspection drift guard."""
    import inspect
    src = inspect.getsource(R.main)
    # The shared partition is called in raw_core2 mode...
    assert "blend_ice_ocean_forcing(" in src
    assert 'sw_partition="raw_core2"' in src
    # ...at the PRE-step concentration captured before step_sea_ice.
    assert "_ice_conc_pre = ice_state.concentration.data" in src
    assert "ice_concentration=_ice_conc_pre" in src
    # ...and the OLD in-runner routing (full atm stress + added ice stress,
    # beginning-of-step heat attenuation) is GONE.
    assert "_route_ice_response_to_ocean" not in src
    assert not hasattr(R, "_route_ice_response_to_ocean")
    # The prescribed-ice surrogate path still drives the legacy attenuation...
    assert "ice_albedo=_ice_alb" in src
    assert "under_ice=_under_ice" in src
    # ...but the prognostic path no longer forces it at forcing-build time
    # (the old `_under_ice = True` override inside `if ice_config is not None`).
    assert "_under_ice = True" not in src


def test_runner_routes_net_freshwater_to_kpp_channel():
    """Fix B drift guard: the non-cube host loop must set
    sf.freshwater = net_freshwater_flux(fw) for the plain (no-ice) freshwater
    path, and guard the contract with the setup-time surface-forcing-scheme
    check (an 'external' scheme would double-apply the freshwater)."""
    import inspect
    src = inspect.getsource(R.main)
    assert "freshwater=net_freshwater_flux(fw._replace(restoring=None))" in src
    assert "_validate_kpp_freshwater_contract(" in src


def test_kpp_freshwater_contract_guard_per_grid():
    """The contract guard blocks exactly the sf.freshwater-as-mass consumers
    (codex r1 #1 + r3 #2): LatLonCGrid 'external' applies it as virtual salt
    (double application); MPAS 'external' deposits only tau/q_net so it is
    KPP-safe; restoring/prescribed/combined never read sf.freshwater and must
    NOT be rejected."""
    # Safe combinations do not raise.
    R._validate_kpp_freshwater_contract("mpas", "none")
    R._validate_kpp_freshwater_contract("mpas", "external")
    R._validate_kpp_freshwater_contract("mpas", "combined")
    R._validate_kpp_freshwater_contract("tripole", "none")
    R._validate_kpp_freshwater_contract("tripole", "restoring")
    R._validate_kpp_freshwater_contract("latlon", "none")
    R._validate_kpp_freshwater_contract("latlon", "combined")
    # Double-application combinations fail fast.
    for grid_type in ("tripole", "latlon"):
        with pytest.raises(RuntimeError, match="KPP-freshwater contract"):
            R._validate_kpp_freshwater_contract(grid_type, "external")


def test_blend_kpp_channel_excludes_sss_restoring():
    """codex r1 #3: a nonzero fw.restoring (numerical SSS-restoring virtual
    flux) must stay on the MASS channel but NOT enter the KPP buoyancy signal."""
    shape = (4,)
    sf, fw, _, _ = _open_forcing(shape)
    fw = fw._replace(restoring=jnp.full(shape, 7.0e-6))
    resp = _zero_tile(shape)
    fw2, sf2 = blend_ice_ocean_forcing(
        open_sf=sf, open_fw=fw, ice_resp=resp,
        ice_concentration=jnp.zeros(shape), ocean_mask=jnp.ones(shape),
        sw_partition="raw_core2", alpha_ocean=_ALPHA_OC,
        sw_transmittance_ice=_TAU_ICE_SW)
    # Mass channel keeps the restoring component...
    np.testing.assert_allclose(np.asarray(fw2.restoring), 7.0e-6)
    # ...but the KPP buoyancy channel is the PHYSICAL P-E+R+ice only.
    np.testing.assert_allclose(
        np.asarray(sf2.freshwater),
        np.asarray(fw.precip - fw.evap + fw.runoff))
    assert not np.allclose(np.asarray(sf2.freshwater),
                           np.asarray(net_freshwater_flux(fw2)))


def test_runoff_component_vars_exclude_isf():
    """Fix D: --runoff + --isf must not double-count the ice-shelf melt.  The
    loader's component list drops sornfisf when ISF is separately applied, and
    the runner passes exclude_isf=args.isf."""
    assert R._runoff_component_vars(False) == ("sorunoff", "sornfisf",
                                               "Icb_flux")
    assert R._runoff_component_vars(True) == ("sorunoff", "Icb_flux")
    import inspect
    src = inspect.getsource(R.main)
    assert "exclude_isf=args.isf" in src


def test_under_ice_heat_attenuation_reduces_qnet():
    """The EXISTING under-ice mechanism the wiring leverages
    (_ice_surface_heat, under_ice=True) cuts under-ice SW + suppresses open-ocean
    turbulent+LW by (1-conc) -> ice-covered cells get LESS ocean heating than
    full open water.  This is the function the runner now drives with the live
    prognostic concentration."""
    from legoesm.ocean.coupler.omip2_applicator import _ice_surface_heat
    shape = (5,)
    sw_down = np.full(shape, 150.0)
    q_non_sw = np.full(shape, 60.0)            # +into ocean open-water non-SW
    conc = np.full(shape, 0.9)                 # 90% ice cover

    # Open water (no ice): full SW + full non-SW.
    _, q_open = _ice_surface_heat(sw_down, q_non_sw, None)
    # Under live ice: attenuated.
    _, q_ice = _ice_surface_heat(sw_down, q_non_sw, conc, under_ice=True,
                                 tau_ice_sw=0.03)
    assert np.all(q_ice < q_open)              # ice insulates -> less heating
    # At full ice cover the open-water non-SW is fully removed.
    _, q_full = _ice_surface_heat(sw_down, q_non_sw, np.ones(shape),
                                  under_ice=True, tau_ice_sw=0.03)
    # only the small transmitted SW remains; far below open water.
    assert np.all(q_full < 0.2 * q_open)

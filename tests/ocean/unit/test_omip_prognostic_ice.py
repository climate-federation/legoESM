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
(c) the runner partitions via the ONE shared mask-aware blend
    (``legoesm.coupler.ocean_forcing.blend_omip_ice_ocean_forcing``: open
    heat/SW/stress/evap x f_open = 1 - A + each ice term once — full partition
    suite in ``tests/unit/test_omip_sea_ice_coupling.py``), and the ocean
    core's EXISTING salt application (``salt_flux_salinity_tendency`` — the
    literal function ``ocean_pe_mpas`` calls) moves SSS in the expected
    direction;
(d) the --prognostic-sea-ice mutual-exclusion / prerequisite validation;
(e) --runoff/--isf never double-count the ice-shelf melt channel (sornfisf).

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
# (c) the runner partitions via the ONE shared blend; the corrected stress /
#     evap partition (f_open-scaled open forcing + ice terms once) replaces
#     the old full-atmospheric-stress-plus-ice-stress routing.  The FULL
#     partition suite (spec tests 1-6) lives in
#     tests/unit/test_omip_sea_ice_coupling.py — here we pin only the runner-
#     facing defect: open stress/evap must scale by f_open = 1 - A.
# ===========================================================================
def _zero_tile(shape):
    z = jnp.zeros(shape)
    return TileResponse(
        T_sfc=z, albedo=z, emissivity=z, z0=z, q_surface=z, shflx=z,
        lhflx=z, tau_x=z, tau_y=z, lw_up=z, u_ocean_sfc=z, v_ocean_sfc=z,
        co2_flux=z, freshwater_flux=z, ocean_heat_extraction=z,
        ocean_stress_x=z, ocean_stress_y=z, surface_mass_flux=z, salt_flux=z)


def test_blend_scales_open_stress_and_evap_by_open_fraction():
    """The shared blend the runner calls must NOT deliver the full open-water
    atmospheric stress/evaporation under ice (the reviewed defect): open
    channels scale by f_open = 1 - A, the ice stress enters once weighted by
    A (atmospheric sign convention), and each exchange channel enters once."""
    from legoesm.coupler.ocean_forcing import blend_omip_ice_ocean_forcing

    shape = (10,)
    q0 = jnp.full(shape, -30.0)               # open-water net heat [W/m2]
    tau0_x = jnp.full(shape, 0.08)            # full-cell CORE-II wind stress
    tau0_y = jnp.full(shape, -0.03)
    sf = OceanSurfaceForcing(
        tau_x=tau0_x, tau_y=tau0_y, q_net=q0, sw_down=jnp.full(shape, 80.0))
    fw = FreshwaterForcing(
        precip=jnp.full(shape, 1e-5), evap=jnp.full(shape, 3e-6),
        runoff=jnp.zeros(shape), ice_fw=jnp.zeros(shape),
        restoring=jnp.zeros(shape))

    salt = jnp.full(shape, -2.0e-6)           # brine reject: salt LEAVES ocean
    heat = jnp.full(shape, 15.0)              # ocean LOSES 15 W/m2 to the ice
    icefw = jnp.full(shape, 4.0e-6)           # ice melt freshwater INTO ocean
    sx = jnp.full(shape, 0.02)                # ice->ocean stress (force ON ocean)
    sy = jnp.full(shape, -0.01)
    resp = _zero_tile(shape)._replace(
        salt_flux=salt, ocean_heat_extraction=heat, freshwater_flux=icefw,
        ocean_stress_x=sx, ocean_stress_y=sy)

    A = 0.7
    fw2, sf2 = blend_omip_ice_ocean_forcing(
        ice_resp=resp, ice_concentration=jnp.full(shape, A),
        open_ocean_sf=sf, open_ocean_fw=fw, ocean_mask=jnp.ones(shape))

    f_open = 1.0 - A
    # salt_flux on the surface forcing (the in-core real-salt channel), once.
    np.testing.assert_allclose(np.asarray(sf2.salt_flux), np.asarray(salt))
    # q_net: open part x f_open MINUS the basal heat extraction.
    np.testing.assert_allclose(np.asarray(sf2.q_net),
                               f_open * np.asarray(q0) - np.asarray(heat))
    # Stress: NOT full tau0 + ice stress — f_open*tau0 - A*ocean_stress.
    np.testing.assert_allclose(np.asarray(sf2.tau_x),
                               f_open * np.asarray(tau0_x) - A * np.asarray(sx))
    np.testing.assert_allclose(np.asarray(sf2.tau_y),
                               f_open * np.asarray(tau0_y) - A * np.asarray(sy))
    # Evaporation acts on the open fraction only; P stays full-cell.
    np.testing.assert_allclose(np.asarray(fw2.evap),
                               f_open * np.asarray(fw.evap))
    np.testing.assert_allclose(np.asarray(fw2.precip), np.asarray(fw.precip))
    # Ice melt freshwater enters the EXISTING ice_fw channel (not P-E).
    np.testing.assert_allclose(np.asarray(fw2.ice_fw), np.asarray(icefw))
    # net freshwater includes the ice melt (P - E_open + R + ice_fw).
    np.testing.assert_allclose(
        np.asarray(net_freshwater_flux(fw2)),
        np.asarray(fw.precip - f_open * fw.evap + fw.runoff + icefw))
    # KPP buoyancy channel mirrors the same physical net (restoring is zero).
    np.testing.assert_allclose(np.asarray(sf2.freshwater),
                               np.asarray(net_freshwater_flux(fw2)))


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


def test_runner_partitions_via_shared_blend_not_attenuation():
    """Under --prognostic-sea-ice the runner builds the UNMASKED full-cell
    open-ocean bulk forcing (NO under-ice attenuation inside
    compute_omip2_surface_forcing — that would double-suppress once the blend
    scales by f_open) and partitions with the ONE shared mask-aware blend
    using the POST-step concentration + the ocean/land mask.  Source-
    introspection guard (drift): the prognostic path must call the blend and
    must NOT re-enable the prescribed-ice attenuation override."""
    import inspect
    src = inspect.getsource(R.main)
    # The shared partitioning is wired with mask + post-step concentration.
    assert "blend_omip_ice_ocean_forcing(" in src
    assert "ocean_mask=state.land_mask.data" in src
    # The old prognostic-path attenuation override is gone: under_ice is only
    # ever the --ice-thermo prescribed surrogate, never forced True for the
    # live ice.
    assert "_under_ice = True" not in src
    assert "_under_ice = args.ice_thermo" in src
    # KPP/vmix freshwater-buoyancy contract: the physical net freshwater is
    # placed on the surface forcing for the boundary-layer closures.
    assert "physical_net_freshwater_flux" in src


def test_under_ice_heat_attenuation_reduces_qnet():
    """The under-ice mechanism of the PRESCRIBED-siconc surrogate
    (--ice-thermo; _ice_surface_heat, under_ice=True) cuts under-ice SW +
    suppresses open-ocean turbulent+LW by (1-conc) -> ice-covered cells get
    LESS ocean heating than full open water.  The prognostic-ice path does
    NOT use it (it partitions via the shared blend instead); this pins the
    legacy surrogate only."""
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


# ===========================================================================
# (e) --runoff/--isf ice-shelf-melt double count (sornfisf).
# ===========================================================================
def test_runoff_channels_exclude_sornfisf_when_isf_active():
    """--runoff sums rivers + ice-shelf melt + icebergs; --isf applies the
    SAME sornfisf channel separately as a depth-banded melt.  With both flags
    the loader must DROP sornfisf from the surface-runoff sum — the meltwater
    enters exactly once."""
    full = R.runoff_source_channels(exclude_isf=False)
    assert full == ("sorunoff", "sornfisf", "Icb_flux")
    excl = R.runoff_source_channels(exclude_isf=True)
    assert "sornfisf" not in excl
    # Rivers + icebergs are NEVER dropped (only the ISF channel moves).
    assert excl == ("sorunoff", "Icb_flux")


def test_runner_wires_isf_exclusion_into_runoff_loader():
    """Source-introspection guard (drift): the --runoff call site must pass
    exclude_isf=args.isf so the exclusion can never silently detach from the
    --isf flag, and the loader must select channels via the tested helper.
    The provenance guard must also be present: excluding _RUNOFF_NC's
    sornfisf while --isf applies a DIFFERENT file is a silent dataset
    substitution and must warn loudly."""
    import inspect
    src = inspect.getsource(R.main)
    assert "exclude_isf=args.isf" in src
    loader_src = inspect.getsource(R.load_runoff_monthly)
    assert "runoff_source_channels(exclude_isf)" in loader_src
    # Provenance guard (codex r2): resolve-compare the two forcing paths.
    assert "Path(args.isf_forcing_file).resolve()" in src
    assert "Path(_RUNOFF_NC).resolve()" in src

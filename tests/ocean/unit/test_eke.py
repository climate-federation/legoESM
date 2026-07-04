"""Direct unit tests for the prognostic EKE closure (build-spec gate E1).

Tests the pure closure properties: kappa_GM monotone + nonnegative in E, the
production form (kappa_GM * sigma^2), dissipation sign + E^{3/2} scaling, the
mixing-length floor, and finiteness — independent of the state/step coupling.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    eke_3d_local_tendency,
    eke_deformation_radius,
    eke_kappa_gm,
    eke_len_composite,
    eke_local_tendency,
    eke_mixing_length,
    eke_rhines_length,
)


def test_eke_config_defaults_match_veros_acc():
    cfg = EKEConfig()
    assert cfg.c_k == 0.4
    assert cfg.c_eps == 0.5
    assert cfg.l_min == 100.0


def test_mixing_length_floor():
    cfg = EKEConfig(l_min=100.0)
    L_rossby = jnp.array([10.0, 100.0, 5.0e4])
    L = eke_mixing_length(L_rossby, cfg)
    assert float(L[0]) == 100.0      # floored
    assert float(L[1]) == 100.0      # at the floor
    assert float(L[2]) == 5.0e4      # above the floor, unchanged


def test_kappa_gm_nonnegative_and_monotone_in_E():
    cfg = EKEConfig()
    L = jnp.full((20,), 3.0e4)
    E = jnp.linspace(0.0, 1.0, 20)
    kappa = eke_kappa_gm(E, L, cfg)
    assert jnp.all(kappa >= 0.0), "kappa_GM must be non-negative"
    # monotone non-decreasing in E (sqrt).
    assert jnp.all(jnp.diff(kappa) >= -1e-12), "kappa_GM must be monotone in E"
    # exact form away from the regulariser.
    E1 = jnp.array([0.25]); L1 = jnp.array([3.0e4])
    np.testing.assert_allclose(
        np.asarray(eke_kappa_gm(E1, L1, cfg)),
        np.asarray(cfg.c_k * L1 * jnp.sqrt(E1)), rtol=1e-6,
    )


def test_kappa_gm_capped():
    cfg = EKEConfig(kappa_gm_max=2.0e3)
    kappa = eke_kappa_gm(jnp.array([1.0e6]), jnp.array([1.0e5]), cfg)
    assert float(kappa[0]) == 2.0e3


def test_tendency_zero_at_zero_E():
    """At E=0: production (kappa~0) and dissipation (E^{3/2}=0) both vanish."""
    cfg = EKEConfig()
    E = jnp.zeros((5,)); sigma = jnp.full((5,), 1.0e-5); L = jnp.full((5,), 3.0e4)
    t = eke_local_tendency(E, sigma, L, cfg)
    assert float(jnp.max(jnp.abs(t))) < 1e-15


def test_production_form_and_sign():
    """Production = kappa_GM * sigma^2 >= 0; equals the closed form."""
    cfg = EKEConfig()
    E = jnp.array([0.04]); sigma = jnp.array([2.0e-5]); L = jnp.array([3.0e4])
    # With dissipation subtracted; isolate by checking production-only via a
    # tiny E where dissipation (E^{3/2}) is sub-dominant, plus the closed form.
    kappa = eke_kappa_gm(E, L, cfg)
    prod = kappa * sigma ** 2
    diss = cfg.c_eps * E ** 1.5 / L
    np.testing.assert_allclose(
        np.asarray(eke_local_tendency(E, sigma, L, cfg)),
        np.asarray(prod - diss), rtol=1e-10,
    )
    assert float(prod[0]) >= 0.0


def test_dissipation_dominates_at_large_E():
    """With no production (sigma=0) the tendency is pure dissipation: <= 0 and
    scales as E^{3/2}."""
    cfg = EKEConfig()
    L = jnp.array([3.0e4])
    sigma0 = jnp.array([0.0])
    for E in (jnp.array([0.01]), jnp.array([0.1]), jnp.array([1.0])):
        t = eke_local_tendency(E, sigma0, L, cfg)
        assert float(t[0]) <= 0.0, "dissipation-only tendency must be <= 0"
    # E^{3/2} scaling: doubling... 8x E -> 8^{1.5}=~22.6x dissipation magnitude.
    t1 = -float(eke_local_tendency(jnp.array([0.1]), sigma0, L, cfg)[0])
    t8 = -float(eke_local_tendency(jnp.array([0.8]), sigma0, L, cfg)[0])
    np.testing.assert_allclose(t8 / t1, 8.0 ** 1.5, rtol=1e-6)


def test_tendency_finite_on_field():
    cfg = EKEConfig()
    rng = np.random.default_rng(0)
    E = jnp.asarray(np.abs(rng.standard_normal((8, 16))) * 0.05)
    sigma = jnp.asarray(np.abs(rng.standard_normal((8, 16))) * 1e-5)
    L = jnp.asarray(rng.uniform(1e4, 5e4, (8, 16)))
    t = eke_local_tendency(E, sigma, L, cfg)
    assert t.shape == (8, 16)
    assert jnp.all(jnp.isfinite(t))


# ---------------------------------------------------------------------------
# E2 — GM/Redi coupling (prognostic kappa_GM) + config + validation
# ---------------------------------------------------------------------------


def test_gmredi_config_accepts_eke():
    """GMRediConfig has an optional eke field (presence-based selection)."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    assert GMRediConfig().eke is None
    cfg = GMRediConfig(eke=EKEConfig())
    assert isinstance(cfg.eke, EKEConfig)


def test_validate_eke_config_raises_on_bad_params():
    from legoesm.ocean.physics.lateral_mixing.eke import validate_eke_config
    validate_eke_config(EKEConfig())  # default must pass
    for bad in (EKEConfig(c_k=0.0), EKEConfig(c_eps=-1.0), EKEConfig(l_min=0.0),
                EKEConfig(kappa_gm_max=0.0)):
        try:
            validate_eke_config(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def _eke_coupling_inputs(E_val):
    import numpy as np
    from legoesm.ocean.vertical import create_ocean_z_star
    nlat, nlon, nlev = 4, 6, 5
    z = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    # Stable stratification: rho increases with depth (k index).
    rho = jnp.asarray(
        1025.0 + np.linspace(0.0, 2.0, nlev)[None, None, :]
        * np.ones((nlat, nlon, 1))
    )
    S_x = jnp.full((nlat, nlon, nlev - 1), 1.0e-3)
    S_y = jnp.full((nlat, nlon, nlev - 1), 5.0e-4)
    jac = jnp.ones((nlat, nlon))
    f = jnp.full((nlat, nlon), 1.0e-4)
    E = jnp.full((nlat, nlon), float(E_val))
    return E, rho, S_x, S_y, z, jac, f


def test_compute_eke_kappa_gm_prognostic_and_monotone():
    """The prognostic kappa_GM is >= 0, finite, increases with E, and returns the
    Eady rate + mixing length for the EKE source/sink. Reuses the shared
    _eady_growth_and_length (no duplicate numerics)."""
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    vcfg, ecfg = VisbeckConfig(), EKEConfig()
    E_lo, *rest = _eke_coupling_inputs(0.01)
    k_lo, sig, L = compute_eke_kappa_gm(E_lo, *rest, vcfg, ecfg)
    E_hi, *rest_hi = _eke_coupling_inputs(0.25)
    k_hi, _, _ = compute_eke_kappa_gm(E_hi, *rest_hi, vcfg, ecfg)
    assert k_lo.shape == (4, 6) and sig.shape == (4, 6) and L.shape == (4, 6)
    assert jnp.all(jnp.isfinite(k_lo)) and jnp.all(jnp.isfinite(sig))
    assert jnp.all(k_lo >= 0.0)
    assert jnp.all(L >= ecfg.l_min)              # mixing length floored
    assert float(jnp.mean(k_hi)) > float(jnp.mean(k_lo))  # kappa grows with E
    # E=0 -> kappa_GM = 0 (no prognostic mixing without eddy energy).
    E0, *rest0 = _eke_coupling_inputs(0.0)
    k0, _, _ = compute_eke_kappa_gm(E0, *rest0, vcfg, ecfg)
    assert float(jnp.max(k0)) < 1e-6


# ---------------------------------------------------------------------------
# E3 — positivity (semi-implicit dissipation, no clipping)
# ---------------------------------------------------------------------------


def test_E3_positivity_preserved_over_steps():
    """The local EKE update keeps E >= 0 over many steps for a wide range of
    E0/sigma/dt — by construction (semi-implicit), no floor/clip needed."""
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    cfg = EKEConfig()
    rng = np.random.default_rng(7)
    L = jnp.asarray(rng.uniform(1e4, 5e4, (6, 8)))
    for dt in (300.0, 3600.0, 86400.0, 10.0 * 86400.0):  # incl. huge dt
        E = jnp.asarray(np.abs(rng.standard_normal((6, 8))) * 0.05)
        sigma = jnp.asarray(np.abs(rng.standard_normal((6, 8))) * 1e-5)
        for _ in range(50):
            E = eke_apply_local_source(E, sigma, L, cfg, dt)
            assert jnp.all(E >= 0.0), f"E went negative at dt={dt}"
            assert jnp.all(jnp.isfinite(E))


def test_E3_grows_from_small_E_when_forced():
    """With production (sigma>0), E grows away from ~0 toward a bounded steady
    state (production dominates near 0; dissipation ~ E^{3/2} caps it)."""
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    cfg = EKEConfig()
    L = jnp.full((1,), 3.0e4)
    sigma = jnp.full((1,), 3.0e-5)
    E = jnp.full((1,), 1.0e-6)
    traj = [float(E[0])]
    for _ in range(400):
        E = eke_apply_local_source(E, sigma, L, cfg, 3600.0)
        traj.append(float(E[0]))
    assert traj[-1] > traj[0], "E should grow under forcing"
    assert jnp.isfinite(E[0]) and float(E[0]) < 1e3, "E should stay bounded"
    # near steady state: last step changes little.
    assert abs(traj[-1] - traj[-2]) < 0.05 * traj[-1] + 1e-9


# ---------------------------------------------------------------------------
# E5 — differentiability
# ---------------------------------------------------------------------------


def test_E5_differentiable_through_closure_and_coupling():
    """jax.grad through the EKE local update + the prognostic kappa_GM coupling is
    finite and nonzero."""
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    cfg, vcfg = EKEConfig(), VisbeckConfig()
    E0, rho, S_x, S_y, z, jac, f = _eke_coupling_inputs(0.04)

    def loss(E):
        kappa, sigma, L = compute_eke_kappa_gm(
            E, rho, S_x, S_y, z, jac, f, vcfg, cfg)
        E1 = eke_apply_local_source(E, sigma, L, cfg, 3600.0)
        return jnp.sum(kappa ** 2) + jnp.sum(E1 ** 2)

    g = jax.grad(loss)(E0)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad through EKE closure/coupling"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "zero grad — path not differentiated"


# ---------------------------------------------------------------------------
# E4 — budget closure: E-transport conserves the area integral of E
# ---------------------------------------------------------------------------


def _int_residual(tend, area):
    a = np.asarray(area)
    t = np.asarray(tend)
    return abs(float(np.sum(t * a))) / (float(np.sum(np.abs(t) * a)) + 1e-300)


def test_E4_transport_conserves_integral_E():
    """Flux-form advection + lateral diffusion of E each conserve the area-integral
    of E to machine-eps on a periodic domain (telescoping; no-flux walls)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_horizontal_transport,
    )
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)  # float64: measure scheme, not f32
    nlat, nlon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(2)
    E = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon))) * 0.05)
    U_bar = jnp.asarray(0.1 * rng.standard_normal((nlat, nlon + 1)))
    U_bar = U_bar.at[:, -1].set(U_bar[:, 0])               # periodic wrap (u-pt n_lon == 0)
    V_bar = jnp.asarray(0.1 * rng.standard_normal((nlat + 1, nlon)))
    V_bar = V_bar.at[0].set(0.0).at[-1].set(0.0)            # N/S walls
    mask = jnp.ones((nlat, nlon))
    u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    area = grid.area
    Z = jnp.zeros_like
    # advection only
    t_adv = eke_horizontal_transport(
        E, U_bar, V_bar, grid, EKEConfig(k_iso=0.0), mask, u_mask, v_mask)
    assert _int_residual(t_adv, area) < 1e-12, "advection not conservative"
    # diffusion only (no flow)
    t_diff = eke_horizontal_transport(
        E, Z(U_bar), Z(V_bar), grid, EKEConfig(k_iso=1000.0), mask, u_mask, v_mask)
    assert _int_residual(t_diff, area) < 1e-12, "lateral diffusion not conservative"
    # combined
    t = eke_horizontal_transport(
        E, U_bar, V_bar, grid, EKEConfig(k_iso=500.0), mask, u_mask, v_mask)
    assert _int_residual(t, area) < 1e-12, "combined transport not conservative"
    assert jnp.all(jnp.isfinite(t))


# ---------------------------------------------------------------------------
# E6 — state threading + step integration (EKE-on integrates an eke field)
# ---------------------------------------------------------------------------


def test_E6_step_integrates_eke_field():
    """With gm_redi.eke set, the model step integrates a prognostic eke field
    that stays >= 0 + finite and evolves; EKE-off leaves eke None (zero-behaviour
    is covered by the existing step suite)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig, Field
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=85.0,
    )
    nlat, nlon = grid.n_lat, grid.n_lon
    # Horizontal T perturbation -> baroclinic slopes -> nonzero Eady rate (so EKE
    # production is active).
    T = np.asarray(state.T.data)
    lat = np.degrees(np.asarray(grid.lat))
    T = T + 2.0 * np.tanh(lat / 20.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    # Initialise eke as a Field (stable pytree structure across steps).
    eke0 = EKEConfig().e_min
    state = state._replace(
        eke=Field(data=jnp.full((nlat, nlon), eke0), name="eke",
                  dims=("lat", "lon"), units="m^2/s^2"))

    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=8, enable_runtime_checks=False,
        gm_redi=GMRediConfig(kappa_GM=0.0, kappa_Redi=1.0e3, eke=EKEConfig()),
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)

    for _ in range(15):
        state = model.step(state, dt=1800.0)
        assert state.eke is not None, "eke field dropped from state"
        E = np.asarray(state.eke.data)
        assert np.all(E >= 0.0), "eke went negative"
        assert np.all(np.isfinite(E)), "eke non-finite"
        assert np.all(np.isfinite(np.asarray(state.T.data)))
    # eke evolved away from the uniform initial value (production/transport active).
    assert float(np.max(np.asarray(state.eke.data))) > eke0


def test_E6_restart_round_trip_eke_field():
    """EKE-on: the eke Field round-trips bit-identically through the field-generic
    ocean restart I/O (save_restart/load_restart serialise every Field). EKE-off
    (eke=None) round-trip is already covered by test_restart_round_trip_bit_identical."""
    import tempfile, os as _os
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import Field
    from legoesm.ocean.restart import save_restart, load_restart

    grid = create_latlon_grid(8, 16)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=85.0)
    rng = np.random.default_rng(3)
    E = np.abs(rng.standard_normal((grid.n_lat, grid.n_lon))) * 0.05
    state = state._replace(
        eke=Field(data=jnp.asarray(E), name="eke", dims=("lat", "lon"),
                  units="m^2/s^2"))
    with tempfile.TemporaryDirectory() as d:
        out = _os.path.join(d, "restart_eke.npz")
        save_restart(state, out, time_s=0.0, step=0, sha="eke")
        state2 = load_restart(out, state)
    assert state2.eke is not None, "eke dropped on restart round-trip"
    np.testing.assert_array_equal(
        np.asarray(state2.eke.data), np.asarray(state.eke.data))


# ---------------------------------------------------------------------------
# E7 — idealized baroclinic channel (tier 2): EKE spins up bounded, kappa_GM
# responds to E, and the run stays stable vs EKE-off.
# ---------------------------------------------------------------------------


def _baroclinic_channel(eke_on, n_steps=30, dt=1800.0):
    """Coarse re-entrant channel (periodic-lon, polar walls) with a meridional
    T front -> baroclinic slopes -> Eady-rate forcing for EKE. Returns the final
    state, the model, and the GM/Redi config. EKE-off uses a constant GM kappa
    so the two runs are a fair stability comparison."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig, Field
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(16, 32)
    z_coord = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    # Strong meridional T front -> baroclinic slopes (EKE production source).
    T = np.asarray(state.T.data)
    lat = np.degrees(np.asarray(grid.lat))
    T = T + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    nlat, nlon = grid.n_lat, grid.n_lon
    eke_cfg = EKEConfig()
    state = state._replace(
        eke=Field(data=jnp.full((nlat, nlon), eke_cfg.e_min), name="eke",
                  dims=("lat", "lon"), units="m^2/s^2"))

    gm = GMRediConfig(kappa_GM=1.0e3, kappa_Redi=1.0e3,
                      eke=eke_cfg if eke_on else None)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        n_barotropic_substeps=8, enable_runtime_checks=False, gm_redi=gm)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)

    E_max_trace = []
    for _ in range(n_steps):
        state = model.step(state, dt=dt)
        if eke_on:
            E = np.asarray(state.eke.data)
            assert np.all(E >= 0.0) and np.all(np.isfinite(E))
            E_max_trace.append(float(np.max(E)))
        assert np.all(np.isfinite(np.asarray(state.T.data)))
        assert np.all(np.isfinite(np.asarray(state.u.data)))
    return state, model, cfg, np.array(E_max_trace)


def test_E7_channel_eke_spins_up_bounded_and_kappa_responds():
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_eke_step_kappa,
    )
    state, model, cfg, E_max_trace = _baroclinic_channel(eke_on=True)
    gm = cfg.gm_redi
    e_min = gm.eke.e_min

    # (a) E spun up above the uniform initial floor (production active). The
    # ABSOLUTE level stays small because EKE equilibrates on a multi-year
    # dissipation timescale (L/(c_eps·√E)) while this gate runs ~hours — the gate
    # verifies the SIGN of the evolution (production dominant -> E grows), the
    # boundedness, and the kappa response, NOT the equilibrium. Observed ~30x
    # growth from the floor; assert a clear margin so this is not floor noise.
    E_final = np.asarray(state.eke.data)
    assert float(np.max(E_final)) > 5.0 * e_min, "EKE did not spin up"
    # (b) ...and stays BOUNDED (no blow-up: orders of magnitude below any
    # numerical explosion) and finite + non-negative.
    assert float(np.max(E_final)) < 1.0e3, "EKE blew up"
    assert np.all(np.isfinite(E_final)) and np.all(E_final >= 0.0)
    # bounded trajectory: the running max never exploded.
    assert np.all(np.isfinite(E_max_trace)) and float(np.max(E_max_trace)) < 1.0e3

    # (c) kappa_GM RESPONDS to E: read the prognostic coefficient from the final
    # state, assert >= 0, finite, varies in space (std > 0), and differs from the
    # baseline computed at the uniform initial E (it evolved with E).
    lm = state.land_mask.data
    kappa_final, _sig, _L = compute_eke_step_kappa(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        state.eke.data, model.grid, model.z_coord, gm,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=lm,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    kf = np.asarray(kappa_final)
    wet = np.asarray(lm) > 0
    assert np.all(kf >= 0.0) and np.all(np.isfinite(kf))
    assert float(np.std(kf[wet])) > 0.0, "kappa_GM is spatially uniform"
    kappa_base, _, _ = compute_eke_step_kappa(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        jnp.full_like(state.eke.data, e_min), model.grid, model.z_coord, gm,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=lm,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    assert float(np.max(np.abs(kf - np.asarray(kappa_base)))) > 0.0, \
        "kappa_GM did not respond to the evolved E"


def test_E7_channel_stable_with_eke_off():
    """The same channel with EKE off (constant GM kappa) also runs stable +
    finite — EKE adds the prognostic closure without destabilising the run."""
    state, _model, _cfg, _ = _baroclinic_channel(eke_on=False)
    assert np.all(np.isfinite(np.asarray(state.T.data)))
    assert np.all(np.isfinite(np.asarray(state.u.data)))
    assert np.all(np.isfinite(np.asarray(state.v.data)))


# ---------------------------------------------------------------------------
# L1 — Rhines-limited mixing length (eke_len variant), pure functions.
# Reproduces Veros eke_len = max(lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines))
# (veros/core/eke.py:54-67). Tested as pure formulas here; wired at L2.
# ---------------------------------------------------------------------------

# Representative developed-ACC column (~ -45° latitude), used across the L1 tests:
_ACC_BETA = 1.62e-11      # 2Ω cos45°/R [1/(m·s)]
_ACC_FMID = 1.03e-4       # |f| at -45° [1/s]
_ACC_INT_N_DZ = 8.0       # ∫N dz [m/s] (N~2e-3 over ~4 km) -> c1 = 8/π ≈ 2.55 m/s
_ACC_EKE = 1.0e-6         # specific eddy energy [m²/s²] (Veros ACC eke ~1e-6)


def test_eke_len_config_defaults_match_veros():
    cfg = EKEConfig()
    assert cfg.mixing_length_scheme == "rossby"   # default = legoESM pre-eke_len path
    assert cfg.eke_cross == 1.0                    # Veros settings.py defaults
    assert cfg.eke_crhin == 1.0


def test_validate_eke_config_raises_on_bad_mixing_length_params():
    from legoesm.ocean.physics.lateral_mixing.eke import validate_eke_config
    validate_eke_config(EKEConfig(mixing_length_scheme="rhines"))   # valid scheme
    for bad in (EKEConfig(mixing_length_scheme="bogus"),
                EKEConfig(eke_cross=0.0), EKEConfig(eke_crhin=-1.0)):
        try:
            validate_eke_config(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_rhines_length_matches_formula_and_monotone_in_E():
    cfg = EKEConfig()
    E = jnp.array([1.0e-6, 1.0e-4, 1.0e-2])
    L = eke_rhines_length(E, jnp.array(_ACC_BETA), cfg)
    # Exact: sqrt(sqrt(E)/beta) (the +1e-30 regulariser is negligible at these E).
    expect = np.sqrt(np.sqrt(np.asarray(E, float)) / _ACC_BETA)
    np.testing.assert_allclose(np.asarray(L), expect, rtol=1e-10)
    assert float(L[0]) < float(L[1]) < float(L[2])     # monotone increasing in E
    assert 1.0e3 < float(L[0]) < 5.0e4                  # ACC regime -> O(10 km)


def test_rhines_length_beta_floor_finite_at_zero_beta():
    cfg = EKEConfig()
    L = eke_rhines_length(jnp.array(1.0e-4), jnp.array(0.0), cfg)   # beta=0 -> floor
    assert np.isfinite(float(L)) and float(L) > 0.0


def test_deformation_radius_matches_formula_both_branches():
    cfg = EKEConfig()
    c1 = _ACC_INT_N_DZ / np.pi
    L_mid = c1 / _ACC_FMID
    L_eq = np.sqrt(c1 / (2.0 * _ACC_BETA))
    # Midlatitude: |f| large -> c1/|f| is the smaller branch.
    L = eke_deformation_radius(
        jnp.array(_ACC_INT_N_DZ), jnp.array(_ACC_FMID), jnp.array(_ACC_BETA), cfg)
    assert L_mid < L_eq                                # midlat branch wins here
    np.testing.assert_allclose(float(L), L_mid, rtol=1e-9)
    # Equatorial: |f| -> 0 -> the sqrt branch caps the (otherwise huge) radius.
    L_eqr = eke_deformation_radius(
        jnp.array(_ACC_INT_N_DZ), jnp.array(1.0e-8), jnp.array(_ACC_BETA), cfg)
    np.testing.assert_allclose(float(L_eqr), L_eq, rtol=1e-6)
    assert float(L_eqr) < c1 / 1.0e-8                  # capped below the midlat value


def test_deformation_radius_zero_at_zero_stratification():
    cfg = EKEConfig()
    L = eke_deformation_radius(
        jnp.array(0.0), jnp.array(_ACC_FMID), jnp.array(_ACC_BETA), cfg)
    assert float(L) == 0.0   # c1=0 -> midlat branch is 0 and wins the min


def test_eke_len_composite_picks_min_and_floors():
    cfg = EKEConfig(l_min=100.0, eke_cross=2.0, eke_crhin=1.0)   # ACC weights
    # min(2·200km, 1·8km) = 8km, above the 100 m floor.
    eke_len = eke_len_composite(jnp.array(200.0e3), jnp.array(8.0e3), cfg)
    np.testing.assert_allclose(float(eke_len), 8.0e3, rtol=1e-12)
    # Floor dominates when both candidate lengths are tiny.
    assert float(eke_len_composite(jnp.array(10.0), jnp.array(5.0), cfg)) == 100.0


def test_eke_len_full_acc_regime_is_order_km_not_hundreds_km():
    """The headline gap: in the developed ACC the Rhines scale limits eke_len to
    O(km) — far below the ~200 km the legoESM Visbeck length saturates at (the
    "rossby" scheme), the ~25x reduction that blocked EKE adoption (E9). Shown on
    the pure composite. (For this column the Veros-form deformation radius itself
    is ~25 km; the ~200 km comparator below is a representative saturated Visbeck
    L_rossby, not computed from _ACC_INT_N_DZ.)"""
    cfg = EKEConfig(eke_cross=2.0, eke_crhin=1.0)
    L_def = eke_deformation_radius(
        jnp.array(_ACC_INT_N_DZ), jnp.array(_ACC_FMID), jnp.array(_ACC_BETA), cfg)
    L_rhines = eke_rhines_length(jnp.array(_ACC_EKE), jnp.array(_ACC_BETA), cfg)
    eke_len = eke_len_composite(L_def, L_rhines, cfg)
    assert float(L_rhines) < float(L_def)              # Rhines is the limiter
    np.testing.assert_allclose(float(eke_len), float(L_rhines), rtol=1e-12)
    assert float(eke_len) < 3.0e4                      # < 30 km
    # vs the rossby-only scheme at a representative saturated L_rossby (~200 km):
    rossby_only = eke_mixing_length(jnp.array(2.0e5), cfg)
    assert float(eke_len) < 0.1 * float(rossby_only)   # >10x smaller


# ---------------------------------------------------------------------------
# L2 — scheme dispatch + β + ∫N dz wired into compute_eke_kappa_gm.
# Default "rossby" stays bit-identical (the unchanged tests above cover it);
# here we exercise the new "rhines" branch + the dispatch guards.
# ---------------------------------------------------------------------------

def _coupling_with_beta(E_val, beta_val=1.62e-11):
    """`_eke_coupling_inputs` plus a Visbeck cfg and a β field (df/dy at ~ -45°)."""
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    E, rho, S_x, S_y, z, jac, f = _eke_coupling_inputs(E_val)
    beta = jnp.full(E.shape, float(beta_val))
    return (E, rho, S_x, S_y, z, jac, f), VisbeckConfig(), beta


def test_L2_eady_growth_returns_int_N_dz():
    """_eady_growth_and_length returns ∫N dz AND the local Eady growth sigma(z)
    (5-tuple): ∫N dz positive+finite (the rhines deformation radius reuses the
    shared N, no duplicate numerics); sigma(z) is the depth-resolved <N|S|> at
    interior interfaces whose depth-average is sigma_bar (drives the 3-D EKE
    source)."""
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        _eady_growth_and_length,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, _beta = _coupling_with_beta(0.01)
    out = _eady_growth_and_length(rho, S_x, S_y, z, jac, f, vcfg)
    assert len(out) == 6              # dz_half appended (Treguier reuse)
    sigma_bar, _L, _wet, int_N_dz, sigma, _dzh = out
    assert int_N_dz.shape == (4, 6)
    assert jnp.all(int_N_dz >= 0.0) and jnp.all(jnp.isfinite(int_N_dz))
    assert float(jnp.mean(int_N_dz)) > 0.0   # stratified -> ∫N dz > 0
    # local Eady growth sigma(z): interior interfaces (nlev-1), >= 0, finite;
    # sigma_bar is its depth-average so it must lie within the column extremes.
    nlev = int(z.dz_ref.shape[0])
    assert sigma.shape == (4, 6, nlev - 1)
    assert jnp.all(sigma >= 0.0) and jnp.all(jnp.isfinite(sigma))
    assert jnp.all(sigma_bar <= jnp.max(sigma, axis=-1) + 1e-12)
    assert jnp.all(sigma_bar >= jnp.min(sigma, axis=-1) - 1e-12)


def test_L2_rhines_scheme_gives_smaller_L_than_rossby():
    """The headline wiring effect: the rhines eke_len (deformation/Rhines-limited)
    is strictly smaller than the Visbeck "rossby" length on the same column — the
    ~π (and Rhines) reduction that the recipe needs (L5)."""
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, beta = _coupling_with_beta(0.01)
    ecfg_ros = EKEConfig(mixing_length_scheme="rossby")
    ecfg_rhi = EKEConfig(mixing_length_scheme="rhines")
    _k0, _s0, L_ros = compute_eke_kappa_gm(
        E, rho, S_x, S_y, z, jac, f, vcfg, ecfg_ros)
    k_rhi, _s1, L_rhi = compute_eke_kappa_gm(
        E, rho, S_x, S_y, z, jac, f, vcfg, ecfg_rhi, beta=beta)
    assert jnp.all(jnp.isfinite(L_rhi)) and jnp.all(k_rhi >= 0.0)
    assert float(jnp.mean(L_rhi)) < float(jnp.mean(L_ros))


def test_L2_rhines_smaller_E_shrinks_L():
    """Lower eddy energy -> smaller Rhines scale -> smaller eke_len once Rhines limits."""
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    ecfg = EKEConfig(mixing_length_scheme="rhines")
    (E_hi, rho, S_x, S_y, z, jac, f), vcfg, beta = _coupling_with_beta(1.0e-2)
    (E_lo, *_rest), _v, _b = _coupling_with_beta(1.0e-6)
    _k0, _s0, L_hi = compute_eke_kappa_gm(
        E_hi, rho, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
    _k1, _s1, L_lo = compute_eke_kappa_gm(
        E_lo, rho, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
    assert float(jnp.mean(L_lo)) < float(jnp.mean(L_hi))


def test_L2_rhines_requires_beta():
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, _beta = _coupling_with_beta(0.01)
    ecfg = EKEConfig(mixing_length_scheme="rhines")
    try:
        compute_eke_kappa_gm(E, rho, S_x, S_y, z, jac, f, vcfg, ecfg)  # no beta
    except ValueError:
        return
    raise AssertionError("expected ValueError when rhines scheme has no beta")


def test_L2_unknown_scheme_raises_in_dispatch():
    """Dispatch discipline: an unknown scheme raises in compute_eke_kappa_gm
    (defense-in-depth beyond validate_eke_config) — no silent fallback."""
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, beta = _coupling_with_beta(0.01)
    ecfg = EKEConfig(mixing_length_scheme="bogus")
    try:
        compute_eke_kappa_gm(E, rho, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mixing_length_scheme")


def test_L2_rhines_dry_column_finite_no_nan():
    """A fully-dry column (jacobian=0 -> N²=0/0=NaN) must NOT produce NaN L, kappa,
    or grads in the rhines path: int_N_dz is wet-masked like sigma_bar/N_bar, so the
    column collapses to L=l_min, kappa=0. Locks the L2-review robustness fix (the
    rossby path was already NaN-free here; rhines must match it on land columns)."""
    import jax
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, beta = _coupling_with_beta(0.01)
    jac = jac.at[0, 0].set(0.0)                      # column (0,0) fully dry
    ecfg = EKEConfig(mixing_length_scheme="rhines")
    kappa, sigma, L = compute_eke_kappa_gm(
        E, rho, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
    assert jnp.all(jnp.isfinite(L)), "rhines L has NaN on a dry column"
    assert jnp.all(jnp.isfinite(kappa)) and jnp.all(jnp.isfinite(sigma))
    assert float(kappa[0, 0]) == 0.0                 # dry column masked to 0
    # And no NaN-grad trap: d(Σκ)/dE is finite everywhere.
    def _loss(E_in):
        k, _s, _l = compute_eke_kappa_gm(
            E_in, rho, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
        return jnp.sum(k)
    g = jax.grad(_loss)(E)
    assert jnp.all(jnp.isfinite(g)), "rhines kappa grad has NaN on a dry column"


# ---------------------------------------------------------------------------
# L3 — differentiability of the rhines mixing length at TYPICAL values
# (finite AND nonzero). Edge cases (E=0, dry column) are locked by L1's grad
# probe + the L2 dry-column test; here we confirm the chain is smooth and
# carries gradient where it should, for adjoint/DA use.
# ---------------------------------------------------------------------------

def test_L3_rhines_length_chain_differentiable_finite_and_nonzero():
    """grad through eke_deformation_radius + eke_rhines_length + eke_len_composite
    is finite AND nonzero in both limiting regimes (Rhines-limited vs deformation-
    limited)."""
    import jax
    cfg = EKEConfig(eke_cross=2.0, eke_crhin=1.0)

    # (a) Rhines-limited (small E): eke_len follows L_rhines -> d/dE > 0.
    def eke_len_of_E(E):
        L_def = eke_deformation_radius(
            jnp.array(_ACC_INT_N_DZ), jnp.array(_ACC_FMID), jnp.array(_ACC_BETA), cfg)
        L_rh = eke_rhines_length(E, jnp.array(_ACC_BETA), cfg)
        return eke_len_composite(L_def, L_rh, cfg)
    g_E = jax.grad(eke_len_of_E)(jnp.array(_ACC_EKE))   # 1e-6 -> Rhines limits
    assert jnp.isfinite(g_E) and float(g_E) > 0.0

    # (b) deformation-limited (large E): eke_len follows L_def -> d/d(∫N dz) > 0.
    def eke_len_of_intN(intN):
        L_def = eke_deformation_radius(
            intN, jnp.array(_ACC_FMID), jnp.array(_ACC_BETA), cfg)
        L_rh = eke_rhines_length(jnp.array(1.0e-2), jnp.array(_ACC_BETA), cfg)
        return eke_len_composite(L_def, L_rh, cfg)
    g_N = jax.grad(eke_len_of_intN)(jnp.array(_ACC_INT_N_DZ))
    assert jnp.isfinite(g_N) and float(g_N) > 0.0


def test_L3_rhines_coupling_differentiable_through_kappa():
    """grad of the prognostic kappa_GM (rhines scheme) w.r.t. E and rho is finite +
    nonzero — the full coupling stays differentiable (mirrors E5 for the rossby
    path). E=0.05 puts the deformation radius as the limiter so kappa carries
    gradient through BOTH √E and ∫N dz(rho)."""
    import jax
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, beta = _coupling_with_beta(0.05)
    ecfg = EKEConfig(mixing_length_scheme="rhines")

    def loss_E(E_in):
        k, _s, _l = compute_eke_kappa_gm(
            E_in, rho, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
        return jnp.sum(k)

    def loss_rho(rho_in):
        k, _s, _l = compute_eke_kappa_gm(
            E, rho_in, S_x, S_y, z, jac, f, vcfg, ecfg, beta=beta)
        return jnp.sum(k)

    gE = jax.grad(loss_E)(E)
    gR = jax.grad(loss_rho)(rho)
    assert jnp.all(jnp.isfinite(gE)) and float(jnp.sum(jnp.abs(gE))) > 0.0
    assert jnp.all(jnp.isfinite(gR)) and float(jnp.sum(jnp.abs(gR))) > 0.0


# ---------------------------------------------------------------------------
# Stage 2 — 3-D prognostic EKE equation (depth-resolved, on the W-grid /
# interior interfaces), matching Veros's 3-D vs.eke. The closure functions:
#   eke_3d_local_tendency      P(z) - eps(z)
#   eke_3d_vertical_diffusion  implicit K = alpha_eke·A_v (reuse implicit solver)
#   eke_3d_horizontal_transport per-level flux-form advection + lateral diffusion
# 2-D path must stay bit-identical (regression below). See gm_redi_latlon_cgrid.
# ---------------------------------------------------------------------------


def test_stage2_config_alpha_eke_and_eke_3d_defaults():
    """The new EKEConfig fields exist with the Veros-matching / off-by-default
    values: alpha_eke = 1.0 (Veros settings.alpha_eke; ACC leaves the default),
    eke_3d = False (the 2-D path is the default)."""
    cfg = EKEConfig()
    assert cfg.alpha_eke == 1.0      # Veros settings.py "factor vertical friction"
    assert cfg.eke_3d is False       # default = 2-D depth-integrated closure


def test_stage2_validate_eke_config_rejects_negative_alpha_eke():
    from legoesm.ocean.physics.lateral_mixing.eke import validate_eke_config
    validate_eke_config(EKEConfig(alpha_eke=0.0))     # >= 0 ok (0 => no vdiff)
    validate_eke_config(EKEConfig(alpha_eke=2.5))
    try:
        validate_eke_config(EKEConfig(alpha_eke=-1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("alpha_eke < 0 must raise ValueError")


def _eke_3d_depth_resolved_inputs(E_val):
    """Stage-1 depth-resolved (kappa_GM(z), sigma(z), L(z)) at interior interfaces
    from a stably stratified, baroclinic column — the genuine 3-D source/sink
    inputs the 3-D path receives. Returns (E3, sigma3, L3, cfg) with E3 a 3-D field
    at the (nlev-1) interfaces."""
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_eke_kappa_gm,
    )
    from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    E2, rho, S_x, S_y, z, jac, f = _eke_coupling_inputs(E_val)  # 2-D E (n_lat,n_lon)
    nlat, nlon = E2.shape
    M = rho.shape[-1] - 1
    E3 = jnp.broadcast_to(E2[:, :, None], (nlat, nlon, M)) + 0.0
    vcfg, ecfg = VisbeckConfig(), EKEConfig(eke_3d=True)
    kappa3, sigma3, L3 = compute_eke_kappa_gm(
        E3, rho, S_x, S_y, z, jac, f, vcfg, ecfg, depth_resolved=True)
    return E3, kappa3, sigma3, L3, ecfg


# (a) 3-D source/sink shapes + signs ----------------------------------------

def test_stage2_local_tendency_shape_and_signs():
    """eke_3d_local_tendency returns (n_lat, n_lon, nlev-1) with P >= 0 and eps >= 0
    (so the production-only and dissipation-only limits have the right sign), and is
    bit-identical to the shape-agnostic 2-D eke_local_tendency on the same 3-D
    inputs (no duplicate numerics)."""
    E3, kappa3, sigma3, L3, cfg = _eke_3d_depth_resolved_inputs(0.04)
    # E3 + the LOCAL Eady growth sigma3 carry the full vertical structure
    # (n_lat, n_lon, nlev-1); the "rossby" mixing length L3 has no genuine depth
    # dependence, so Stage 1 returns it as (n_lat, n_lon, 1), broadcasting over
    # levels. kappa3 = c_k·L3·√E3 and the local-tendency output broadcast to the
    # full 3-D (n_lat, n_lon, nlev-1).
    assert E3.shape == (4, 6, 4) and sigma3.shape == (4, 6, 4)
    assert L3.shape == (4, 6, 1) and kappa3.shape == (4, 6, 4)
    tend = eke_3d_local_tendency(E3, sigma3, L3, cfg)
    assert tend.shape == (4, 6, 4)
    assert jnp.all(jnp.isfinite(tend))
    # Delegates to the validated 2-D closure ⇒ bit-identical.
    np.testing.assert_array_equal(
        np.asarray(tend), np.asarray(eke_local_tendency(E3, sigma3, L3, cfg)))
    # At E=0: eps=0 (E^{3/2}=0) and kappa=c_k·L·√(0+1e-30) ≈ c_k·L·1e-15, so the
    # production P=kappa·sigma^2 is bounded by the documented sqrt-regulariser in
    # eke_kappa_gm, NOT bit-zero. With L~3e4, sigma~5e-5 this is O(1e-22): assert it
    # is negligibly small (well below e_min·any-rate), not exactly zero.
    t0 = eke_3d_local_tendency(jnp.zeros_like(E3), sigma3, L3, cfg)
    assert float(jnp.max(jnp.abs(t0))) < 1e-18, "E=0 tendency should be negligible"
    # P >= 0 always: with sigma>0 and E>0, the production term kappa·sigma^2 >= 0.
    P = eke_kappa_gm(E3, L3, cfg) * sigma3 ** 2
    eps = cfg.c_eps * jnp.maximum(E3, 0.0) ** 1.5 / jnp.maximum(L3, cfg.l_min)
    assert jnp.all(P >= 0.0), "production must be >= 0"
    assert jnp.all(eps >= 0.0), "dissipation must be >= 0"
    # tendency == P - eps exactly (the documented decomposition).
    np.testing.assert_allclose(np.asarray(tend), np.asarray(P - eps), rtol=1e-12)


def test_stage2_local_tendency_dissipation_dominates_at_large_E():
    """At large E the depth-resolved sink (eps ~ E^{3/2}) overwhelms the source
    (P ~ kappa·sigma^2 ~ sqrt(E)) ⇒ tendency < 0, per level."""
    _E3, _k, sigma3, L3, cfg = _eke_3d_depth_resolved_inputs(0.04)
    E_big = jnp.full_like(sigma3, 100.0)
    tend = eke_3d_local_tendency(E_big, sigma3, L3, cfg)
    assert jnp.all(tend < 0.0), "dissipation must dominate at large E"


# (b) implicit vertical EKE diffusion: variance down + column integral conserved

def _w_grid_metrics(M, H=4000.0):
    """Uniform W-grid metrics for the M-interface EKE field: M layer thicknesses
    and M-1 spacings (build_dz_half), plus a constant A_v at the M-1 interfaces."""
    from legoesm.ocean.physics.vertical_mixing import build_dz_half
    dz_w = jnp.full((M,), H / M)
    return dz_w, build_dz_half(dz_w)


def test_stage2_vertical_diffusion_reduces_variance_and_conserves_column():
    """The implicit vertical EKE diffusion (K = alpha_eke·A_v, reusing
    implicit_vertical_diffusion_ocean) reduces the vertical variance of E AND
    conserves the column integral sum(E·dz_w) under zero-flux BCs (backward Euler
    is conservative). Also: alpha_eke = 0 is the identity (no diffusion)."""
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_3d_vertical_diffusion,
    )
    nlat, nlon, M = 3, 4, 6
    dz_w, dz_half_w = _w_grid_metrics(M)
    A_v = jnp.full((M - 1,), 5.0e-2)
    cfg = EKEConfig(eke_3d=True)
    # A vertically structured E (spike near the top W-level) so there is variance
    # to remove.  Float64 to measure the scheme, not f32 round-off.
    rng = np.random.default_rng(11)
    base = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon, M))) * 1.0e-3)
    E0 = base.at[:, :, 0].add(0.5)
    col0 = jnp.sum(E0 * dz_w, axis=-1)               # (n_lat, n_lon) column integral
    var0 = jnp.var(E0, axis=-1)                       # per-column vertical variance
    E1 = eke_3d_vertical_diffusion(E0, A_v, dz_w, dz_half_w, 86400.0, cfg)
    assert E1.shape == E0.shape and jnp.all(jnp.isfinite(E1))
    var1 = jnp.var(E1, axis=-1)
    assert jnp.all(var1 <= var0 + 1e-15), "vertical diffusion must not increase variance"
    assert float(jnp.mean(var1)) < float(jnp.mean(var0)), "variance should decrease"
    # Column integral conserved to ~machine precision (zero-flux BCs).
    col1 = jnp.sum(E1 * dz_w, axis=-1)
    rel = jnp.abs(col1 - col0) / (jnp.abs(col0) + 1e-300)
    assert float(jnp.max(rel)) < 1e-12, "column integral not conserved"
    # alpha_eke = 0 ⇒ K = 0 ⇒ identity.
    E_id = eke_3d_vertical_diffusion(E0, A_v, dz_w, dz_half_w, 86400.0,
                                     EKEConfig(eke_3d=True, alpha_eke=0.0))
    np.testing.assert_allclose(np.asarray(E_id), np.asarray(E0), rtol=0, atol=0)


# (c) 3-D advection conserves the volume integral of E -----------------------

def test_stage2_3d_advection_conserves_volume_integral():
    """Per-level flux-form advection + lateral diffusion of the 3-D E conserve the
    VOLUME integral sum_k(sum_xy E·area·dz_w) to ~machine precision (each level's
    area integral telescopes: periodic lon, no-flux N/S walls; lateral diffusion is
    div of grad).  Tested with a per-level-varying flow (u(z), v(z))."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_3d_horizontal_transport,
    )
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)  # f64: measure scheme
    nlat, nlon = grid.n_lat, grid.n_lon
    M = 5
    dz_w, _ = _w_grid_metrics(M)
    rng = np.random.default_rng(4)
    E = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon, M))) * 0.05)
    # Per-level flow, with a depth-dependent shear so it is NOT the depth mean.
    U = jnp.asarray(0.1 * rng.standard_normal((nlat, nlon + 1, M)))
    U = U.at[:, -1, :].set(U[:, 0, :])                   # periodic wrap (u-pt n_lon == 0)
    V = jnp.asarray(0.1 * rng.standard_normal((nlat + 1, nlon, M)))
    V = V.at[0].set(0.0).at[-1].set(0.0)                 # N/S walls
    mask = jnp.ones((nlat, nlon))
    u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    area = np.asarray(grid.area)

    def vol_residual(tend):
        t = np.asarray(tend)
        num = 0.0
        den = 0.0
        for k in range(M):
            w = area * float(dz_w[k])
            num += float(np.sum(t[:, :, k] * w))
            den += float(np.sum(np.abs(t[:, :, k]) * w))
        return abs(num) / (den + 1e-300)

    # advection only
    t_adv = eke_3d_horizontal_transport(
        E, U, V, grid, EKEConfig(k_iso=0.0, eke_3d=True), mask, u_mask, v_mask)
    assert t_adv.shape == (nlat, nlon, M)
    assert vol_residual(t_adv) < 1e-12, "3-D advection not volume-conservative"
    # lateral diffusion only (no flow)
    Z = jnp.zeros_like
    t_diff = eke_3d_horizontal_transport(
        E, Z(U), Z(V), grid, EKEConfig(k_iso=1000.0, eke_3d=True),
        mask, u_mask, v_mask)
    assert vol_residual(t_diff) < 1e-12, "3-D lateral diffusion not conservative"
    # combined
    t = eke_3d_horizontal_transport(
        E, U, V, grid, EKEConfig(k_iso=500.0, eke_3d=True), mask, u_mask, v_mask)
    assert vol_residual(t) < 1e-12, "combined 3-D transport not conservative"
    assert jnp.all(jnp.isfinite(t))


def test_stage2_3d_advection_differs_from_depthmean_under_shear():
    """The 3-D transport advects each interface by ITS OWN flow, not the depth-mean
    — so under vertical shear the per-level tendency differs from advecting every
    level by the depth-averaged velocity (the key 3-D vs 2-D distinction)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_3d_horizontal_transport,
    )
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    nlat, nlon, M = grid.n_lat, grid.n_lon, 4
    rng = np.random.default_rng(8)
    E = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon, M))) * 0.05)
    U = jnp.asarray(0.2 * rng.standard_normal((nlat, nlon + 1, M)))
    U = U.at[:, -1, :].set(U[:, 0, :])
    V = jnp.asarray(0.2 * rng.standard_normal((nlat + 1, nlon, M)))
    V = V.at[0].set(0.0).at[-1].set(0.0)
    mask = jnp.ones((nlat, nlon)); u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    cfg = EKEConfig(k_iso=0.0, eke_3d=True)
    t_per_level = eke_3d_horizontal_transport(E, U, V, grid, cfg, mask, u_mask, v_mask)
    # Replace per-level flow with the depth mean (broadcast back over levels).
    U_bar = jnp.broadcast_to(jnp.mean(U, axis=-1, keepdims=True), U.shape)
    V_bar = jnp.broadcast_to(jnp.mean(V, axis=-1, keepdims=True), V.shape)
    t_depthmean = eke_3d_horizontal_transport(
        E, U_bar, V_bar, grid, cfg, mask, u_mask, v_mask)
    assert float(jnp.max(jnp.abs(t_per_level - t_depthmean))) > 0.0, \
        "3-D transport collapsed to the depth-mean advection"


# (d) finiteness + positivity (E stays >= e_min over a stepped budget) -------

def test_stage2_3d_budget_positive_and_finite_over_steps():
    """A full operator-split 3-D EKE step (semi-implicit local source/sink +
    per-level horizontal transport + implicit vertical diffusion) keeps E finite
    and >= e_min over many steps, for a structured initial field + sheared flow."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_3d_horizontal_transport, eke_3d_vertical_diffusion,
    )
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    nlat, nlon, M = grid.n_lat, grid.n_lon, 5
    dz_w, dz_half_w = _w_grid_metrics(M)
    A_v = jnp.full((M - 1,), 1.0e-2)
    cfg = EKEConfig(eke_3d=True, k_iso=500.0)
    rng = np.random.default_rng(21)
    sigma3 = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon, M))) * 2.0e-5)
    L3 = jnp.asarray(rng.uniform(1.0e4, 5.0e4, (nlat, nlon, M)))
    U = jnp.asarray(0.1 * rng.standard_normal((nlat, nlon + 1, M)))
    U = U.at[:, -1, :].set(U[:, 0, :])
    V = jnp.asarray(0.1 * rng.standard_normal((nlat + 1, nlon, M)))
    V = V.at[0].set(0.0).at[-1].set(0.0)
    mask = jnp.ones((nlat, nlon)); u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    E = jnp.full((nlat, nlon, M), cfg.e_min)
    dt = 1800.0
    for _ in range(40):
        # local source/sink (semi-implicit ⇒ E >= 0 by construction).
        E = eke_apply_local_source(E, sigma3, L3, cfg, dt)
        # explicit per-level horizontal transport.
        E = E + dt * eke_3d_horizontal_transport(E, U, V, grid, cfg, mask, u_mask, v_mask)
        # implicit vertical diffusion (conserves the column integral, stays >= 0).
        E = eke_3d_vertical_diffusion(E, A_v, dz_w, dz_half_w, dt, cfg)
        # floor at e_min (matches the 2-D step's positivity treatment).
        E = jnp.maximum(E, cfg.e_min)
        assert jnp.all(jnp.isfinite(E)), "E went non-finite"
        assert jnp.all(E >= cfg.e_min - 1e-15), "E dropped below e_min"
    assert float(jnp.max(E)) < 1.0e3, "E blew up"


def test_stage2_3d_budget_differentiable():
    """jax.grad through the whole 3-D EKE budget (source/sink + per-level transport
    + implicit vertical diffusion) is finite and nonzero — the implicit solve and
    the advection are AD-safe."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_3d_horizontal_transport, eke_3d_vertical_diffusion,
    )
    grid = create_latlon_grid(6, 8, dtype=jnp.float64)
    nlat, nlon, M = grid.n_lat, grid.n_lon, 4
    dz_w, dz_half_w = _w_grid_metrics(M)
    A_v = jnp.full((M - 1,), 1.0e-2)
    cfg = EKEConfig(eke_3d=True, k_iso=500.0)
    rng = np.random.default_rng(33)
    sigma3 = jnp.full((nlat, nlon, M), 3.0e-5)
    L3 = jnp.full((nlat, nlon, M), 3.0e4)
    U = jnp.asarray(0.1 * rng.standard_normal((nlat, nlon + 1, M)))
    U = U.at[:, -1, :].set(U[:, 0, :])
    V = jnp.asarray(0.1 * rng.standard_normal((nlat + 1, nlon, M)))
    V = V.at[0].set(0.0).at[-1].set(0.0)
    mask = jnp.ones((nlat, nlon)); u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    dt = 1800.0

    def loss(E):
        src = eke_3d_local_tendency(E, sigma3, L3, cfg)
        tr = eke_3d_horizontal_transport(E, U, V, grid, cfg, mask, u_mask, v_mask)
        E1 = eke_3d_vertical_diffusion(E + dt * (src + tr), A_v, dz_w, dz_half_w, dt, cfg)
        return jnp.sum(E1 ** 2)

    E0 = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon, M))) * 0.04)
    g = jax.grad(loss)(E0)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad through 3-D EKE budget"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "zero grad — 3-D budget not differentiated"


# (e) regression: the 2-D path / functions are bit-identical (unchanged) -----

def test_stage2_2d_path_bit_identical_regression():
    """Stage 2 adds the 3-D path WITHOUT changing the 2-D one. Verify the 2-D
    closure + 2-D transport produce bit-identical results to direct re-computation
    (the eke_3d flag defaults False and is inert for the 2-D functions), so the
    existing validated path is untouched."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        eke_horizontal_transport,
    )
    # 2-D local tendency unchanged: a default cfg and one with eke_3d=True give the
    # SAME 2-D result (the flag does not touch the 2-D function).
    rng = np.random.default_rng(99)
    E = jnp.asarray(np.abs(rng.standard_normal((5, 7))) * 0.05)
    sigma = jnp.asarray(np.abs(rng.standard_normal((5, 7))) * 1e-5)
    L = jnp.asarray(rng.uniform(1e4, 5e4, (5, 7)))
    t_default = eke_local_tendency(E, sigma, L, EKEConfig())
    t_flag = eke_local_tendency(E, sigma, L, EKEConfig(eke_3d=True, alpha_eke=2.0))
    np.testing.assert_array_equal(np.asarray(t_default), np.asarray(t_flag))
    # 2-D horizontal transport unchanged by the new fields.
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    nlat, nlon = grid.n_lat, grid.n_lon
    E2 = jnp.asarray(np.abs(rng.standard_normal((nlat, nlon))) * 0.05)
    U_bar = jnp.asarray(0.1 * rng.standard_normal((nlat, nlon + 1)))
    U_bar = U_bar.at[:, -1].set(U_bar[:, 0])
    V_bar = jnp.asarray(0.1 * rng.standard_normal((nlat + 1, nlon)))
    V_bar = V_bar.at[0].set(0.0).at[-1].set(0.0)
    mask = jnp.ones((nlat, nlon)); u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    t2_default = eke_horizontal_transport(
        E2, U_bar, V_bar, grid, EKEConfig(k_iso=500.0), mask, u_mask, v_mask)
    t2_flag = eke_horizontal_transport(
        E2, U_bar, V_bar, grid, EKEConfig(k_iso=500.0, eke_3d=True, alpha_eke=2.0),
        mask, u_mask, v_mask)
    np.testing.assert_array_equal(np.asarray(t2_default), np.asarray(t2_flag))

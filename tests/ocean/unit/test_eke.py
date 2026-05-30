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

    cfg = LatLonCGridOceanConfig(
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
    cfg = LatLonCGridOceanConfig(
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
    """_eady_growth_and_length now returns ∫N dz (4-tuple) — positive + finite on a
    stratified column — so the rhines deformation radius reuses the shared N (no
    duplicate numerics)."""
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        _eady_growth_and_length,
    )
    (E, rho, S_x, S_y, z, jac, f), vcfg, _beta = _coupling_with_beta(0.01)
    out = _eady_growth_and_length(rho, S_x, S_y, z, jac, f, vcfg)
    assert len(out) == 4
    _sigma_bar, _L, _wet, int_N_dz = out
    assert int_N_dz.shape == (4, 6)
    assert jnp.all(int_N_dz >= 0.0) and jnp.all(jnp.isfinite(int_N_dz))
    assert float(jnp.mean(int_N_dz)) > 0.0   # stratified -> ∫N dz > 0


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

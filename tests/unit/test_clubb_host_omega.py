"""clubb_step: CLUBB's mean vertical velocity from the host omega, and the
l_implemented = .true. call, as CAM's clubb_intr does.

CESM2.1 (clubb_intr.F90:1676-1679, 1778, cam_cesm2_1_rel_60):
``wm_zt = -omega / (rho_ds_zt * gravit)`` (no ``omega_bot``), ``wm_zm =
zt2zm(wm_zt)``; CLUBB is called with ``l_implemented = .true.`` (:113).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig, clubb_step, init_clubb_moments, zt2zm)

from tests.unit.test_clubb_liquid_partition import _NCOL, _NLEV, _column  # noqa: E402

_CFG = CLUBBConfig(trop_cloud_top_press=0.0, prognostic=True)


def _omega():
    """Top-down (ncol, nlev) [Pa/s]: sinking aloft, rising near the ground."""
    s = (np.arange(_NLEV) + 0.5) / _NLEV
    return jnp.asarray(np.broadcast_to(0.3 * np.sin(np.pi * s) - 0.1 * s,
                                       (_NCOL, _NLEV)).copy())


def _run(col, omega, monkeypatch=None, got=None, cam_density=False):
    if monkeypatch is not None:
        real_core = clubb_mod.advance_clubb_core

        def core(*a, **k):
            got.update(k)
            return real_core(*a, **k)

        monkeypatch.setattr(clubb_mod, "advance_clubb_core", core)
    return clubb_step(col["u"], col["v"], col["T"], col["q_v"],
                      init_clubb_moments(_NCOL, _NLEV, _CFG, dtype=jnp.float64),
                      col["p_full"], col["p_half"], col["z_full"], col["z_half"],
                      col["T_sfc"], col["q_sfc"], col["rho"], 300.0, _CFG,
                      omega=omega, omega_cam_layer_density=cam_density)


def test_wm_is_minus_omega_over_rho_g_and_core_is_implemented(monkeypatch):
    col, got = _column(), {}
    omega = _omega()
    _run(col, omega, monkeypatch, got)
    rho_ds_zt = jnp.flip(col["rho"], axis=1)
    wm_zt = -jnp.flip(omega, axis=1) / (rho_ds_zt * constants.g)
    np.testing.assert_array_equal(np.asarray(got["wm_zt"]), np.asarray(wm_zt))
    np.testing.assert_array_equal(np.asarray(got["wm_zm"]),
                                  np.asarray(zt2zm(wm_zt, got["gr"])))
    assert got["l_implemented"] is True
    # Sign: omega > 0 (sinking) -> wm < 0, and vice versa.
    w = np.asarray(got["wm_zt"])
    o = np.asarray(jnp.flip(omega, axis=1))
    assert np.all(np.sign(w[o != 0]) == -np.sign(o[o != 0]))


def test_no_omega_keeps_wm_zero(monkeypatch):
    got = {}
    _run(_column(), None, monkeypatch, got)
    assert not np.any(np.asarray(got["wm_zt"])) and not np.any(np.asarray(got["wm_zm"]))
    assert got["l_implemented"] is True


def test_zero_omega_is_bit_identical_to_none():
    col = _column()
    a = _run(col, None)
    b = _run(col, jnp.zeros_like(col["T"]))
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_omega_changes_the_step_but_not_the_column_water():
    """wm reaches the step (flux/variance equations) while the host keeps the
    mean advection: the flux-form rtm advance telescopes, so the column
    (rho dz) total-water change is the same with and without omega.  The
    standalone path's advective-form mean advection would change it.  Bound
    as test_advance_clubb_core_conserves_thlm_rtm (1e-9 relative)."""
    col = _column()
    a = _run(col, None)
    b = _run(col, _omega())
    assert float(jnp.max(jnp.abs(a[3] - b[3]))) > 0.0
    dz = -jnp.diff(col["z_half"], axis=1)          # top-down, positive
    w = col["rho"] * dz
    ca = jnp.sum(a[3] * w, axis=1)
    cb = jnp.sum(b[3] * w, axis=1)
    rel = float(jnp.max(jnp.abs(cb - ca) / jnp.abs(ca)))
    assert rel < 1e-9, rel


def test_grad_through_omega_is_finite():
    col = _column()

    def loss(omega):
        out = _run(col, omega)
        return jnp.sum(out[2] ** 2) + jnp.sum(out[3] ** 2)

    g = jax.grad(loss)(_omega())
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_mpas_column_lane_calls_the_core_implemented(monkeypatch):
    """The MPAS turbulence factory (the MPAS _run_column lane and the FV3-duo
    column lane both build their physics with make_physics(model_type="mpas"))
    reaches the core with l_implemented = .true.  No host omega reaches it yet
    on this lane, so wm is zero there (pending the dycore-omega decision)."""
    import types

    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas)
    from legoesm.atmosphere.physics.turbulence.clubb import pack_clubb_moments
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _make_mpas_turbulence)
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    state = state._replace(tracers={"q_v": jnp.full((ncell, nlev), 0.008,
                                                    dtype=state.T.data.dtype)})
    cfg = CLUBBConfig(prognostic=True)
    seen = []
    real_core = clubb_mod.advance_clubb_core

    def core(*a, **k):
        seen.append((k["l_implemented"], np.asarray(k["wm_zt"])))
        return real_core(*a, **k)

    monkeypatch.setattr(clubb_mod, "advance_clubb_core", core)
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="clubb", clubb=cfg), 300.0)
    phys_state = types.SimpleNamespace(
        clubb_moments=pack_clubb_moments(init_clubb_moments(ncell, nlev, cfg)))
    fn(state, mesh, sigma, phys_state=phys_state,
       forcing={"T_sfc": jnp.full((ncell,), 295.0)})
    assert seen, "the CLUBB core was not called"
    assert all(impl is True and not np.any(wm) for impl, wm in seen)


def test_cam_layer_density_flag_gives_cams_wm(monkeypatch):
    """omega_cam_layer_density (the GCM host-omega lane): CAM's rho_in =
    pdel/(g dz) (clubb_intr.F90:1654-1656) and half the lowest wm_zt at the
    surface momentum level (CAM's ghost level holds 0)."""
    col, got = _column(), {}
    omega = _omega()
    _run(col, omega, monkeypatch, got, cam_density=True)
    dz = col["z_half"][:, :-1] - col["z_half"][:, 1:]
    pdel = col["p_half"][:, 1:] - col["p_half"][:, :-1]
    rho_in = jnp.flip((1.0 / constants.g) * (pdel / dz), axis=1)
    wm_zt = -1.0 * jnp.flip(omega, axis=1) / (rho_in * constants.g)
    np.testing.assert_array_equal(np.asarray(got["wm_zt"]), np.asarray(wm_zt))
    wm_zm = zt2zm(wm_zt, got["gr"])
    np.testing.assert_array_equal(np.asarray(got["wm_zm"][:, 1:]),
                                  np.asarray(wm_zm[:, 1:]))
    np.testing.assert_array_equal(np.asarray(got["wm_zm"][:, 0]),
                                  0.5 * np.asarray(wm_zt[:, 0]))
    assert not np.allclose(np.asarray(rho_in), np.asarray(jnp.flip(col["rho"], axis=1)),
                           rtol=1e-12, atol=0.0), "fixture cannot tell the densities apart"


def _mpas_setup(host_omega, n_sub=None):
    """Small MPAS state with divergent edge winds, prognostic CLUBB."""
    import types

    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas)
    from legoesm.atmosphere.physics.turbulence.clubb import pack_clubb_moments
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.normal(0.0, 5.0, state.u.data.shape), dtype=jnp.float64)
    f64 = lambda f: f.replace(data=jnp.asarray(f.data, dtype=jnp.float64))  # noqa: E731
    state = state._replace(
        u=state.u.replace(data=u), T=f64(state.T), p_s=f64(state.p_s),
        phis=f64(state.phis),
        tracers={"q_v": state.T.replace(
            data=jnp.full((ncell, nlev), 0.008, dtype=jnp.float64),
            name="q_v", units="kg/kg")})
    cfg = CLUBBConfig(prognostic=True)
    tc = TurbulenceConfig(scheme="clubb", clubb=cfg, host_omega=host_omega)
    phys_state = types.SimpleNamespace(
        clubb_moments=pack_clubb_moments(init_clubb_moments(ncell, nlev, cfg)))
    return mesh, sigma, state, tc, phys_state


def _expected_wm_zt(omega, state, sigma):
    from legoesm.atmosphere.physics._shared import compute_heights_from_sigma
    ncell, nlev = state.T.data.shape
    p_half = sigma.pressure_at_half(state.p_s.data).reshape(ncell, nlev + 1)
    _, z_half = compute_heights_from_sigma(
        state.T.data, p_half, q_v=state.tracers["q_v"].data)
    rho_in = jnp.flip((1.0 / constants.g) * (
        (p_half[:, 1:] - p_half[:, :-1]) / (z_half[:, :-1] - z_half[:, 1:])), axis=1)
    return -1.0 * jnp.flip(omega, axis=1) / (rho_in * constants.g)


def _record_core(monkeypatch):
    seen = []
    real_core = clubb_mod.advance_clubb_core

    def core(*a, **k):
        seen.append(np.asarray(k["wm_zt"]))
        return real_core(*a, **k)

    monkeypatch.setattr(clubb_mod, "advance_clubb_core", core)
    return seen


def test_mpas_lane_flag_hands_clubb_the_continuity_omega(monkeypatch):
    """Flag on: the MPAS bridge passes the closed-continuity omega of the edge
    winds (the driver's wap diagnostic), and CLUBB's wm is CAM's map of it."""
    from legoesm.atmosphere.physics.turbulence.integration import (
        _make_mpas_turbulence, mpas_host_omega)
    mesh, sigma, state, tc, ps = _mpas_setup(True)
    seen = _record_core(monkeypatch)
    fn = _make_mpas_turbulence(tc, 300.0)
    fn(state, mesh, sigma, phys_state=ps,
       forcing={"T_sfc": jnp.full((state.T.data.shape[0],), 295.0)})
    omega = mpas_host_omega(state, mesh, sigma)
    assert float(jnp.max(jnp.abs(omega))) > 0.0
    want = np.asarray(_expected_wm_zt(omega, state, sigma))
    assert seen and all(np.array_equal(w, want) for w in seen)


def test_column_lane_flag_needs_the_host_omega(monkeypatch):
    """A column model (cell winds) has no edge mesh: without a host-supplied
    omega the bridge refuses; with one, CLUBB gets exactly that omega."""
    from legoesm.atmosphere.physics.turbulence.integration import (
        HOST_OMEGA_FORCING_KEY, _make_mpas_turbulence)
    mesh, sigma, state, tc, ps = _mpas_setup(True)
    ncell, nlev = state.T.data.shape
    col = state._replace(u=state.T.replace(data=jnp.full((ncell, nlev), 3.0)),
                         v=state.T.replace(data=jnp.full((ncell, nlev), -2.0)))
    fn = _make_mpas_turbulence(tc, 300.0)
    frc = {"T_sfc": jnp.full((ncell,), 295.0)}
    with pytest.raises(ValueError, match="host must pass"):
        fn(col, mesh, sigma, phys_state=ps, forcing=frc)
    seen = _record_core(monkeypatch)
    omega = jnp.asarray(np.broadcast_to(np.linspace(0.05, 0.3, nlev), (ncell, nlev)).copy())
    fn(col, mesh, sigma, phys_state=ps, forcing={**frc, HOST_OMEGA_FORCING_KEY: omega})
    want = np.asarray(_expected_wm_zt(omega, col, sigma))
    assert seen and all(np.array_equal(w, want) for w in seen)


def test_macmic_subcycle_holds_the_pre_physics_omega(monkeypatch):
    """CAM6 keeps state%omega through the physics: every macmic sub-step's
    CLUBB call sees the omega of the PRE-physics winds, although CLUBB moves
    the winds between sub-steps."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig as GWDConfig)
    from legoesm.atmosphere.physics.turbulence.integration import mpas_host_omega
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    mesh, sigma, state, tc, _ = _mpas_setup(True)
    pc = PhysicsConfig(radiation=RadiationConfig(scheme="none"),
                       convection=ConvectionConfig(scheme="none"),
                       turbulence=tc,
                       microphysics=MicrophysicsConfig(scheme="none"),
                       gravity_wave_drag=GWDConfig(scheme="none"),
                       # the spies read concrete values: no module remat (it traces its body)
                       module_checkpoint=False)
    ncell, nlev = state.T.data.shape
    ps = init_physics_state(ncol=ncell, nlev=nlev, physics_config=pc,
                            dtype=jnp.float64)
    seen = []
    real_step = clubb_mod.clubb_step

    def step(*a, **k):
        seen.append(np.asarray(k["omega"]))
        return real_step(*a, **k)

    monkeypatch.setattr(clubb_mod, "clubb_step", step)
    fn = make_physics(pc, model_type="mpas", dt=1800.0, cld_macmic_num_steps=3)
    tend, _ = fn(state, mesh, sigma, phys_state=ps,
                 forcing={"T_sfc": jnp.full((ncell,), 295.0)})
    want = np.asarray(mpas_host_omega(state, mesh, sigma))
    assert len(seen) >= 3
    assert all(np.array_equal(w, want) for w in seen)
    # non-vacuous: the winds the later sub-steps see have a different omega
    u1 = state.u.data + 600.0 * tend.du_dt.data
    moved = np.asarray(mpas_host_omega(state._replace(u=state.u.replace(data=u1)),
                                       mesh, sigma))
    assert np.max(np.abs(moved - want)) > 0.0


def test_host_omega_refused_where_it_would_be_inert():
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics)
    with pytest.raises(ValueError, match="host_omega"):
        make_turbulence_physics(
            TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=False),
                             host_omega=True), "mpas", 300.0)
    with pytest.raises(ValueError, match="host_omega"):
        make_turbulence_physics(TurbulenceConfig(scheme="louis", host_omega=True),
                                "mpas", 300.0)
    with pytest.raises(NotImplementedError, match="host_omega"):
        make_turbulence_physics(
            TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True),
                             host_omega=True), "hydrostatic", 300.0)


@pytest.mark.parametrize("hybrid", [False, True])
def test_host_omega_is_the_dycores_flux_form_continuity(hybrid):
    """Over a non-uniform surface pressure the advective div(u) dp and the
    dycore's flux-form div(u dp_edge) differ; the host omega's implied surface
    pressure tendency equals the MPAS dycore's own."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, mpas_model_tendencies)
    from legoesm.atmosphere.physics.turbulence.integration import (
        mpas_flux_form_continuity)
    from legoesm.grids.vertical import create_hybrid_coordinate
    mesh, sigma, state, _, _ = _mpas_setup(True)
    if hybrid:
        nlev = state.T.data.shape[1]
        s_h = np.linspace(0.05, 1.0, nlev + 1)
        a = 0.4 * s_h * (1.0 - s_h)
        sigma = create_hybrid_coordinate(nlev, jnp.asarray(a), jnp.asarray(s_h - a))
    ncell = state.T.data.shape[0]
    ps = 1.0e5 - 8.0e3 * jnp.asarray(np.random.default_rng(5).random(ncell))
    state = state._replace(p_s=state.p_s.replace(data=ps))
    omega, dps = mpas_flux_form_continuity(state, mesh, sigma)
    cfg = MPASPrimitiveEquationConfig()
    assert cfg.nu_del4_ps == 0.0
    tend = mpas_model_tendencies(state, mesh, sigma, cfg)
    # relative to the field's maximum: measured 0 (sigma) and 2.0e-8 (hybrid,
    # float32 mesh metrics); the advective div(u) dp form is off by 2.1e-2
    ref = np.asarray(tend.dp_s_dt.data)
    assert np.max(np.abs(np.asarray(dps) - ref)) <= 1e-6 * np.max(np.abs(ref))
    assert np.all(np.isfinite(np.asarray(omega)))

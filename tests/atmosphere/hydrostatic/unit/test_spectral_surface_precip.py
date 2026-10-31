"""Surface precipitation on the spectral lane (#2277).

The spectral convection and microphysics bridges publish the surface precip
[kg/m^2/s, +down] with ``return_sfc_precip=True``; the combined physics sums
it; the model step integrates it with the integrator's own stage weights and
publishes it on ``model._sfc_precip``.

Water budget: the published precip is the scheme's own surface flux exactly
(identity tests, 1e-12), and the column water the physics removes,
``-sum_k dp_k * sum_species dq_k / g``, equals it to the scheme's measured
closure residual, stored per case (closure tests).  Kessler is not used: its
faithful-port sedimentation does not close the column water budget (finding
reported separately).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.physics.convection.config import (
    BechtoldConfig, ConvectionConfig, ZhangMcFarlaneConfig)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics)
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics)
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.core.conservation import WATER_MASS_SPECIES
from legoesm.core.field import Field
from legoesm.diagnostics.column_integrals import column_mass_integral
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.thermo import saturation_mixing_ratio

jax.config.update("jax_enable_x64", True)

_ZM = ZhangMcFarlaneConfig(land_fraction="none")   # aquaplanet columns
DT = 300.0


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(10)


def _moist_state(grid, sigma_coord, rh_low, rh_high, q_c=0.0, q_r=0.0):
    """Warm moist column (CAPE-positive at rh_low ~ 0.95); ``q_c``/``q_r``
    seeded in the lower troposphere so the microphysics rains at once."""
    rest = isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0)
    nlev = sigma_coord.n_levels
    sf = sigma_coord.sigma_full
    T_col = jnp.maximum(300.0 * jnp.power(jnp.clip(sf, 0.05, None), 0.19), 200.0)
    T = jnp.broadcast_to(T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev))
    p = jnp.broadcast_to((sf * 1e5)[None, None, :], T.shape)
    low = (sf > 0.7)[None, None, :]
    qv = jnp.where(low, rh_low, rh_high) * saturation_mixing_ratio(T, p)
    tr = {"q_v": qv, "q_c": jnp.where(low, q_c, 0.0) * jnp.ones_like(qv),
          "q_r": jnp.where(low, q_r, 0.0) * jnp.ones_like(qv)}
    tracers = {k: Field(data=v, name=k, dims=("lat", "lon", "level"),
                        units="kg/kg") for k, v in tr.items()}
    return rest._replace(T_hat=rest.T_hat.replace(data=sh_analysis_3d(grid, T)),
                         tracers=tracers)


def _column_water_loss(tend, state, grid, sigma_coord):
    """-(1/g) sum_k dp_k sum_species dq_k [kg/m^2/s], (n_lat, n_lon)."""
    p_s = spectral_pe_to_grid(state, grid, sigma_coord)["p_s"]
    p_half = sigma_coord.pressure_at_half(p_s)
    dq = sum(tend.tracers[n].data for n in WATER_MASS_SPECIES
             if n in tend.tracers)
    return -column_mass_integral(dq, p_s, sigma_coord.dsigma,
                                 dp=p_half[..., 1:] - p_half[..., :-1])


def _phys_state(grid, sigma_coord, physics_config):
    """Cold-start PhysicsState carry (ZM reads its PBL height, zero)."""
    return init_physics_state(grid.n_lat * grid.n_lon, sigma_coord.n_levels,
                              physics_config)


# Column closure residual max|W - P| [kg/m^2/s] measured per case (x64, the
# fixtures below; SLURM 2026-10-31).  Scheme-intrinsic: Bechtold shows the same
# residual with rain_to_surface off; the published precip is the scheme's own
# flux exactly (the identity tests).  Each case is checked at 10x its value.
MEASURED_CLOSURE = {
    "zm_net_flux": 4.07e-11,
    "bechtold_rain_to_surface": 4.64e-12,
    "sundqvist": 1.23e-9,
    "combined_zm_sundqvist": 3.94e-8,
}


def _assert_closure(case, tend, sfc, state, grid, sigma_coord):
    P = np.asarray(sfc[0])
    assert float(P.max()) > 0.0          # it rained (non-vacuity)
    resid = float(np.max(np.abs(
        np.asarray(_column_water_loss(tend, state, grid, sigma_coord)) - P)))
    print(f"closure {case}: max|W-P| = {resid:.3e}, max P = {P.max():.3e}")
    assert resid <= 10.0 * MEASURED_CLOSURE[case], resid


_CONV_CASES = {
    "zm_net_flux": ("zhang_mcfarlane_convection",
                    ConvectionConfig(scheme="zhang_mcfarlane",
                                     zhang_mcfarlane=_ZM)),
    "bechtold_rain_to_surface": ("bechtold_convection", ConvectionConfig(
        scheme="bechtold", rain_to_surface=True,
        bechtold=BechtoldConfig(enable_stochastic=False))),
}


def _run_convection(grid, sigma_coord, case, monkeypatch):
    """Bridge call with the kernel's rain source captured (spy)."""
    from legoesm.atmosphere.physics.convection import integration as ci
    name, cfg = _CONV_CASES[case]
    real, seen = getattr(ci, name), {}

    def spy(*a, **k):
        out = real(*a, **k)
        seen["dq_r"] = out[0].dq_r_conv_dt
        return out
    monkeypatch.setattr(ci, name, spy)
    state = _moist_state(grid, sigma_coord, 0.95, 0.5)
    fn = make_convection_physics(cfg, model_type="spectral_pe", dt=DT)
    tend, _, sfc = fn(state, grid, sigma_coord, phys_state=_phys_state(
        grid, sigma_coord, PhysicsConfig(convection=cfg)),
        return_sfc_precip=True)
    return state, tend, sfc, seen["dq_r"]


@pytest.mark.parametrize("case", sorted(_CONV_CASES))
def test_convection_publishes_the_rain_that_left(grid, sigma_coord, case,
                                                 monkeypatch):
    """Identity: published P == sum_k dq_r dp / g of the rain the bridge
    routed out (column sum clipped at 0 for ZM's net flux)."""
    state, _, sfc, dq_r = _run_convection(grid, sigma_coord, case, monkeypatch)
    p_s = spectral_pe_to_grid(state, grid, sigma_coord)["p_s"]
    p_half = sigma_coord.pressure_at_half(p_s)
    rain = column_mass_integral(
        dq_r.reshape(p_half.shape[:-1] + (-1,)), p_s, sigma_coord.dsigma,
        dp=p_half[..., 1:] - p_half[..., :-1])
    P = np.asarray(sfc[0])
    assert float(P.max()) > 0.0
    np.testing.assert_allclose(P, np.maximum(np.asarray(rain), 0.0),
                               rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("case", sorted(_CONV_CASES))
def test_convection_column_closure(grid, sigma_coord, case, monkeypatch):
    state, tend, sfc, _ = _run_convection(grid, sigma_coord, case, monkeypatch)
    _assert_closure(case, tend, sfc, state, grid, sigma_coord)


def test_convection_rain_to_q_r_is_not_surface_precip(grid, sigma_coord):
    state = _moist_state(grid, sigma_coord, 0.95, 0.5)
    fn = make_convection_physics(
        ConvectionConfig(scheme="bechtold", rain_to_surface=False,
                         bechtold=BechtoldConfig(enable_stochastic=False)),
        model_type="spectral_pe", dt=DT)
    tend, _, sfc = fn(state, grid, sigma_coord, return_sfc_precip=True)
    assert float(jnp.max(tend.tracers["q_r"].data)) > 0.0   # rain booked
    assert float(jnp.max(jnp.abs(sfc[0]))) == 0.0


def _run_sundqvist(grid, sigma_coord, monkeypatch):
    from legoesm.atmosphere.physics.microphysics import integration as mi
    real, seen = mi.sundqvist_microphysics, {}

    def spy(*a, **k):
        out = real(*a, **k)
        seen["out"] = out
        return out
    monkeypatch.setattr(mi, "sundqvist_microphysics", spy)
    state = _moist_state(grid, sigma_coord, 1.10, 0.5, q_c=1e-3, q_r=1e-3)
    fn = make_microphysics_physics(
        MicrophysicsConfig(scheme="sundqvist"), "spectral_pe", DT)
    tend, sfc = fn(state, grid, sigma_coord, return_sfc_precip=True)
    return state, tend, sfc, seen["out"]


def test_microphysics_publishes_the_scheme_precip(grid, sigma_coord,
                                                  monkeypatch):
    _, _, sfc, out = _run_sundqvist(grid, sigma_coord, monkeypatch)
    assert float(jnp.max(out.precipitation)) > 0.0
    np.testing.assert_array_equal(np.asarray(sfc[0]).reshape(-1),
                                  np.asarray(out.precipitation))
    if out.precip_frozen is None:
        assert sfc[1] is None
    else:
        np.testing.assert_array_equal(np.asarray(sfc[1]).reshape(-1),
                                      np.asarray(out.precip_frozen))


def test_microphysics_column_closure(grid, sigma_coord, monkeypatch):
    state, tend, sfc, _ = _run_sundqvist(grid, sigma_coord, monkeypatch)
    _assert_closure("sundqvist", tend, sfc, state, grid, sigma_coord)


def test_microphysics_refuses_a_dropped_species(grid, sigma_coord):
    state = _moist_state(grid, sigma_coord, 1.10, 0.5, q_c=1e-3, q_r=1e-3)
    state = state._replace(tracers={k: v for k, v in state.tracers.items()
                                    if k != "q_r"})
    fn = make_microphysics_physics(
        MicrophysicsConfig(scheme="kessler"), "spectral_pe", DT)
    with pytest.raises(ValueError, match="q_r"):
        fn(state, grid, sigma_coord, return_sfc_precip=True)
    fn(state, grid, sigma_coord)          # default call unchanged


def test_combined_budget(grid, sigma_coord):
    state = _moist_state(grid, sigma_coord, 1.10, 0.5, q_c=1e-3, q_r=1e-3)
    pc = PhysicsConfig(radiation=RadiationConfig(scheme="none"),
                       convection=ConvectionConfig(scheme="zhang_mcfarlane",
                                                   zhang_mcfarlane=_ZM),
                       microphysics=MicrophysicsConfig(scheme="sundqvist"))
    fn = make_physics(pc, model_type="spectral_pe", dt=DT,
                      surface_bulk_coeffs=None)
    tend, _, sfc = fn(state, grid, sigma_coord,
                      phys_state=_phys_state(grid, sigma_coord, pc),
                      return_sfc_precip=True)
    _assert_closure("combined_zm_sundqvist", tend, sfc, state, grid,
                    sigma_coord)


def _constant_precip_physics(P, frozen):
    """Zero tendency, constant surface precip ``(P, frozen)``."""
    def physics_fn(state, grid, sigma_coord, forcing_data=None,
                   return_sfc_precip=False):
        z = jax.tree.map(jnp.zeros_like, state)
        return (z, None, (P, frozen)) if return_sfc_precip else (z, None)
    physics_fn._emits_sfc_precip = True
    return physics_fn


@pytest.mark.parametrize("integrator", ["ssp_rk54", "leapfrog_si"])
@pytest.mark.parametrize("with_forcing", [False, True])
def test_step_publishes_the_integrated_rate(grid, sigma_coord, integrator,
                                            with_forcing):
    """A constant rate integrates to itself: stage weights sum to one (RK);
    leapfrog publishes the applied tendency's rate, Euler startup included."""
    lf = integrator == "leapfrog_si"
    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord,
        SpectralPEConfig(time_integrator=integrator, semi_implicit=lf))
    state = isothermal_rest_state_spectral(grid, sigma_coord,
                                           perturbation_amplitude=0.0)
    P = jnp.linspace(1e-5, 3e-5, grid.n_lat * grid.n_lon).reshape(
        grid.n_lat, grid.n_lon)
    fn = _constant_precip_physics(P, 0.25 * P)
    fd = {"x": jnp.asarray(0.0)} if with_forcing else None
    for _ in range(3 if lf else 1):
        state = model.step(state, DT, physics_fn=fn, forcing_data=fd)
        np.testing.assert_allclose(np.asarray(model._sfc_precip[0]),
                                   np.asarray(P), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(np.asarray(model._sfc_precip[1]),
                                   0.25 * np.asarray(P), rtol=1e-12, atol=0.0)


def test_step_unknown_frozen_stays_unknown(grid, sigma_coord):
    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord, SpectralPEConfig(time_integrator="ssp_rk54"))
    state = isothermal_rest_state_spectral(grid, sigma_coord,
                                           perturbation_amplitude=0.0)
    P = jnp.full((grid.n_lat, grid.n_lon), 1e-5)
    model.step(state, DT, physics_fn=_constant_precip_physics(P, None))
    assert model._sfc_precip[1] is None


def test_semi_implicit_rk_publishes_none(grid, sigma_coord):
    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord,
        SpectralPEConfig(time_integrator="ssp_rk3", semi_implicit=True))
    state = isothermal_rest_state_spectral(grid, sigma_coord,
                                           perturbation_amplitude=0.0)
    P = jnp.full((grid.n_lat, grid.n_lon), 1e-5)
    model.step(state, DT, physics_fn=_constant_precip_physics(P, P))
    assert model._sfc_precip is None


def _linear_precip_physics(c, a):
    """Physics heating the global-mean T mode at ``c`` [K/s] with precip
    ``a * T_00`` (T_00 = the global-mean temperature coefficient), so the
    precip varies within the step in a known way."""
    def physics_fn(state, grid, sigma_coord, forcing_data=None,
                   return_sfc_precip=False):
        z = jax.tree.map(jnp.zeros_like, state)
        dT = z.T_hat.data.at[0, :].set(c)
        tend = z._replace(T_hat=z.T_hat.replace(data=dT))
        t00 = jnp.real(state.T_hat.data[0, 0])
        P = a * t00 * jnp.ones((grid.n_lat, grid.n_lon))
        return (tend, None, (P, None)) if return_sfc_precip else (tend, None)
    physics_fn._emits_sfc_precip = True
    return physics_fn


def test_rk_rate_is_the_stage_weighted_mean(grid, sigma_coord):
    """Precip linear in a T mode that the physics heats at a constant rate:
    the step-mean rate is the mid-step value a*(T00 + c*dt/2), not the
    start-of-step a*T00 (a stage-1-only or unweighted record fails)."""
    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord, SpectralPEConfig(time_integrator="ssp_rk54"))
    state = isothermal_rest_state_spectral(grid, sigma_coord,
                                           perturbation_amplitude=0.0)
    c, a = 1e-3, 1e-7
    t00 = float(jnp.real(state.T_hat.data[0, 0]))
    model.step(state, DT, physics_fn=_linear_precip_physics(c, a))
    P = np.asarray(model._sfc_precip[0])
    np.testing.assert_allclose(P, a * (t00 + c * DT / 2.0), rtol=1e-10, atol=0.0)
    assert abs(float(P.max()) - a * t00) > 1e3 * 1e-10 * a * t00


def test_leapfrog_publishes_the_lagged_physics_precip(grid, sigma_coord):
    """Leapfrog applies the physics evaluated at the lagged level: the
    published precip is the one of the stored n-1 state (Euler startup: n)."""
    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord,
        SpectralPEConfig(time_integrator="leapfrog_si", semi_implicit=True))
    state = isothermal_rest_state_spectral(grid, sigma_coord,
                                           perturbation_amplitude=0.0)
    a = 1e-7
    fn = _linear_precip_physics(1e-3, a)
    for k in range(3):
        lag = state if model._state_prev is None else model._state_prev
        expect = a * float(jnp.real(lag.T_hat.data[0, 0]))
        cur = a * float(jnp.real(state.T_hat.data[0, 0]))
        state = model.step(state, DT, physics_fn=fn)
        np.testing.assert_allclose(np.asarray(model._sfc_precip[0]), expect,
                                   rtol=1e-13, atol=0.0)
        if k > 0:
            assert abs(expect - cur) > 1e-6 * cur    # lag is observable


def test_combined_default_call_keeps_module_defaults(grid, sigma_coord):
    """Without ``return_sfc_precip`` the combined physics does not ask its
    modules for precip, so the microphysics species check stays off: a
    q_v-only state runs as before (codex review 2026-10-31)."""
    state = _moist_state(grid, sigma_coord, 1.10, 0.5)
    state = state._replace(tracers={"q_v": state.tracers["q_v"]})
    pc = PhysicsConfig(radiation=RadiationConfig(scheme="none"),
                       microphysics=MicrophysicsConfig(scheme="kessler"))
    fn = make_physics(pc, model_type="spectral_pe", dt=DT,
                      surface_bulk_coeffs=None)
    out = fn(state, grid, sigma_coord)
    assert len(out) == 2
    with pytest.raises(ValueError, match="q_c"):
        fn(state, grid, sigma_coord, return_sfc_precip=True)


def test_cpu_batch_integrate_publishes_none(grid, sigma_coord):
    """The CPU batch loop does not integrate the precip: no stale value from
    an earlier step() survives it (codex review 2026-10-31)."""
    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord, SpectralPEConfig(time_integrator="ssp_rk54"))
    state = isothermal_rest_state_spectral(grid, sigma_coord,
                                           perturbation_amplitude=0.0)
    P = jnp.full((grid.n_lat, grid.n_lon), 1e-5)
    fn = _constant_precip_physics(P, P)
    model.step(state, DT, physics_fn=fn)
    assert model._sfc_precip is not None
    cpu = jax.devices("cpu")[0]
    model._use_cpu_for_spectral = True
    model._cpu_device = model._default_device = cpu
    model.integrate(state, DT, DT, physics_fn=fn)
    assert model._sfc_precip is None

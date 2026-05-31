"""Unit tests for the ocean single-column model (:mod:`legoesm.ocean.scm`).

Covers construction / shapes, conservation (adiabatic heat + salt), the
surface-flux energy budget, unconditional stability of the implicit
vertical-mixing path, Crank-Nicolson Coriolis energy conservation, config
validation, and autodiff / jit compatibility.

Run with:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_ocean_scm.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.scm import (
    OceanColumnModel,
    OCEAN_TIME_INTEGRATORS,
    make_ocean_column_state,
    register_ocean_time_integrator,
    _apply_ocean_tendencies,
)
from legoesm.ocean.scm_forcing import (
    OceanSCMForcing,
    default_ocean_forcing,
    validate_ocean_forcing,
)
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


_NLEV = 24


def _no_mixing_cfg():
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
    )


def _constant_mixing_cfg():
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        convection=OceanConvectionConfig(scheme="none"),
    )


def _heat_content(state, dz):
    T = np.asarray(state.T.data[0, 0, 0])
    return float(np.sum(T * dz)) * constants.rho_ocean * constants.c_sw


def _salt_content(state, dz):
    S = np.asarray(state.S.data[0, 0, 0])
    return float(np.sum(S * dz))


# ---------------------------------------------------------------------------
# Construction & state builder
# ---------------------------------------------------------------------------


def test_create_state_shapes():
    scm = OceanColumnModel.create(
        nlev=_NLEV, dt=1800.0,
        T_profile=jnp.linspace(18.0, 4.0, _NLEV), H_max=2000.0,
    )
    assert scm.state.T.data.shape == (1, 1, 1, _NLEV)
    assert scm.state.S.data.shape == (1, 1, 1, _NLEV)
    assert scm.state.u.data.shape == (1, 1, 1, _NLEV)
    assert scm.state.v.data.shape == (1, 1, 1, _NLEV)
    assert scm.state.eta.data.shape == (1, 1, 1)
    assert float(scm.state.H_bathy.data[0, 0, 0]) == pytest.approx(2000.0)
    assert float(scm.state.land_mask.data[0, 0, 0]) == 1.0


def test_make_column_state_validation():
    from legoesm.ocean.vertical import create_ocean_z_star
    z = create_ocean_z_star(n_levels=_NLEV, H_max=1000.0)
    with pytest.raises(ValueError):
        make_ocean_column_state(_NLEV, T_profile=jnp.zeros(_NLEV + 1), z_coord=z)
    # scalar salinity broadcasts
    st = make_ocean_column_state(
        _NLEV, T_profile=jnp.zeros(_NLEV), S_profile=34.0, z_coord=z,
    )
    assert np.allclose(np.asarray(st.S.data), 34.0)


def test_z_coord_nlev_mismatch_raises():
    from legoesm.ocean.vertical import create_ocean_z_star
    z = create_ocean_z_star(n_levels=_NLEV + 3, H_max=1000.0)
    with pytest.raises(ValueError, match="n_levels"):
        OceanColumnModel.create(
            nlev=_NLEV, dt=900.0, T_profile=jnp.zeros(_NLEV), z_coord=z,
        )


# ---------------------------------------------------------------------------
# Conservation & identity
# ---------------------------------------------------------------------------


def test_zero_forcing_zero_physics_is_identity():
    """No physics, no forcing, zero background diffusivity -> state frozen."""
    scm = OceanColumnModel.create(
        physics_config=_no_mixing_cfg(), nlev=_NLEV, dt=3600.0,
        T_profile=jnp.linspace(20.0, 2.0, _NLEV),
        S_profile=jnp.linspace(34.0, 35.0, _NLEV),
        u=0.3, v=-0.2, H_max=1500.0,
        K_v_background=0.0, A_v_background=0.0,
    )
    T0 = np.array(scm.state.T.data)
    S0 = np.array(scm.state.S.data)
    u0 = np.array(scm.state.u.data)
    for _ in range(5):
        scm.step()
    assert np.allclose(np.asarray(scm.state.T.data), T0)
    assert np.allclose(np.asarray(scm.state.S.data), S0)
    assert np.allclose(np.asarray(scm.state.u.data), u0)


def test_adiabatic_heat_and_salt_conserved():
    """Constant interior mixing redistributes but conserves column T and S."""
    scm = OceanColumnModel.create(
        physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=3600.0,
        T_profile=jnp.linspace(22.0, 3.0, _NLEV),
        S_profile=jnp.linspace(33.0, 36.0, _NLEV), H_max=1500.0,
    )
    dz = np.asarray(scm.z_coord.dz_ref)
    hc0, sc0 = _heat_content(scm.state, dz), _salt_content(scm.state, dz)
    scm.run(nsteps=50, save_every=10**9)
    assert abs(_heat_content(scm.state, dz) - hc0) / abs(hc0) < 1e-10
    assert abs(_salt_content(scm.state, dz) - sc0) / abs(sc0) < 1e-10


def test_surface_heat_flux_energy_budget():
    """Column heat-content change equals the integrated surface heat flux."""
    q = 75.0  # W/m^2 into ocean
    scm = OceanColumnModel.create(
        physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=1800.0,
        T_profile=jnp.full((_NLEV,), 12.0), H_max=1500.0,
        forcing=OceanSCMForcing(q_net=lambda t: q),
    )
    dz = np.asarray(scm.z_coord.dz_ref)
    hc0 = _heat_content(scm.state, dz)
    nsteps = 80
    scm.run(nsteps=nsteps, save_every=10**9)
    dHC = _heat_content(scm.state, dz) - hc0
    expected = q * nsteps * scm.dt
    assert abs(dHC - expected) / abs(expected) < 1e-6


def test_surface_evaporation_increases_mean_salinity():
    """Net evaporation (E-P>0) raises the column-mean salinity; the column
    salt-content change matches the integrated virtual salt flux."""
    emp = 1.0e-7  # m/s evaporation
    scm = OceanColumnModel.create(
        physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=1800.0,
        T_profile=jnp.full((_NLEV,), 12.0), S_profile=35.0, H_max=1500.0,
        forcing=OceanSCMForcing(e_minus_p=lambda t: emp),
    )
    dz = np.asarray(scm.z_coord.dz_ref)
    sc0 = _salt_content(scm.state, dz)
    S_top0 = float(scm.state.S.data[0, 0, 0, 0])
    nsteps = 60
    scm.run(nsteps=nsteps, save_every=10**9)
    dSC = _salt_content(scm.state, dz) - sc0
    assert dSC > 0.0  # saltier
    # d/dt (column salt) = S_top * E_minus_P (only top forced; mixing is
    # zero column-integral).  S_top drifts a little, so allow 5%.
    expected = S_top0 * emp * nsteps * scm.dt
    assert abs(dSC - expected) / abs(expected) < 0.05


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_implicit_unconditionally_stable_large_K_and_dt():
    """Convective K_conv≈1 with a thin top cell + large dt: the implicit
    solve stays finite where explicit Euler (K·dt/dz² ≫ 0.5) would blow up."""
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )
    scm = OceanColumnModel.create(
        physics_config=cfg, nlev=_NLEV, dt=7200.0,           # 2-hour step
        T_profile=jnp.linspace(8.0, 16.0, _NLEV),            # unstable
        H_max=300.0, dz_surface=4.0, dz_deep=20.0,
        implicit_vertical_mixing=True,
    )
    # explicit CFL number for the convective K on the top cell is enormous:
    K, dz0, dt = 1.0, 4.0, 7200.0
    assert K * dt / dz0**2 > 100.0
    scm.run(nsteps=30, save_every=10**9)
    assert np.all(np.isfinite(np.asarray(scm.state.T.data)))


def test_coriolis_rotation_conserves_kinetic_energy():
    """The CN Coriolis sub-step is an exact rotation: |u|²+|v|² is preserved
    and the velocity stays bounded over many inertial periods (forward Euler
    would amplify it without bound)."""
    f = 1.0e-4
    scm = OceanColumnModel.create(
        physics_config=_no_mixing_cfg(), nlev=_NLEV, dt=1800.0,
        T_profile=jnp.full((_NLEV,), 10.0), u=0.5, v=0.0, H_max=1000.0,
        K_v_background=0.0, A_v_background=0.0,
        forcing=OceanSCMForcing(f_c=f),
    )
    speed0 = float(np.hypot(scm.state.u.data[0, 0, 0, 0],
                           scm.state.v.data[0, 0, 0, 0]))
    Tin = 2 * np.pi / f
    scm.run(nsteps=int(5 * Tin / scm.dt), save_every=10**9)
    u = np.asarray(scm.state.u.data)
    v = np.asarray(scm.state.v.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    speed = float(np.hypot(u[0, 0, 0, 0], v[0, 0, 0, 0]))
    assert speed == pytest.approx(speed0, rel=1e-6)  # exact rotation


# ---------------------------------------------------------------------------
# Integrators & validation
# ---------------------------------------------------------------------------


def test_explicit_mode_integrator_swap_runs():
    for integ in ("forward_euler", "rk2", "rk4"):
        scm = OceanColumnModel.create(
            physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=600.0,
            T_profile=jnp.linspace(18.0, 6.0, _NLEV), H_max=2000.0,
            implicit_vertical_mixing=False, time_integrator=integ,
            K_v_background=1e-4, A_v_background=1e-3,
        )
        scm.run(nsteps=10, save_every=10**9)
        assert np.all(np.isfinite(np.asarray(scm.state.T.data)))


def test_implicit_mode_rejects_non_euler():
    with pytest.raises(ValueError, match="forward_euler"):
        OceanColumnModel.create(
            nlev=_NLEV, dt=900.0, T_profile=jnp.zeros(_NLEV),
            implicit_vertical_mixing=True, time_integrator="rk4",
        )


def test_unknown_integrator_raises():
    with pytest.raises(ValueError, match="Unknown time_integrator"):
        OceanColumnModel.create(
            nlev=_NLEV, dt=900.0, T_profile=jnp.zeros(_NLEV),
            implicit_vertical_mixing=False, time_integrator="bogus",
        )


def test_conflicting_surface_scheme_rejected():
    cfg = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="prescribed"),
    )
    with pytest.raises(ValueError, match="double-count"):
        OceanColumnModel.create(
            physics_config=cfg, nlev=_NLEV, dt=900.0,
            T_profile=jnp.zeros(_NLEV),
        )


def test_lateral_mixing_forced_off_no_crash():
    """A config carrying lateral mixing (meaningless on a column) runs: the
    SCM forces ∇²≡0 instead of crashing on the missing horizontal grid."""
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="constant"),
        lateral_mixing=LateralMixingConfig(scheme="harmonic"),
    )
    scm = OceanColumnModel.create(
        physics_config=cfg, nlev=_NLEV, dt=900.0,
        T_profile=jnp.linspace(15.0, 5.0, _NLEV), H_max=1500.0,
    )
    scm.run(nsteps=5, save_every=10**9)
    assert np.all(np.isfinite(np.asarray(scm.state.T.data)))


def test_register_custom_integrator():
    def euler_clone(state, aux, f, dt, t):
        tend, aux_out = f(state, aux, t)
        return _apply_ocean_tendencies(state, tend, dt), aux_out

    register_ocean_time_integrator("euler_clone_test", euler_clone)
    assert "euler_clone_test" in OCEAN_TIME_INTEGRATORS
    scm = OceanColumnModel.create(
        physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=600.0,
        T_profile=jnp.linspace(15.0, 5.0, _NLEV), H_max=1500.0,
        implicit_vertical_mixing=False, time_integrator="euler_clone_test",
        K_v_background=1e-4, A_v_background=1e-3,
    )
    scm.run(nsteps=5, save_every=10**9)
    assert np.all(np.isfinite(np.asarray(scm.state.T.data)))


def test_nlev_too_small_rejected():
    with pytest.raises(ValueError, match="nlev >= 2"):
        OceanColumnModel.create(
            nlev=1, dt=900.0, T_profile=jnp.zeros(1),
            implicit_vertical_mixing=False, time_integrator="forward_euler",
        )


def test_legacy_4arg_integrator_rejected_with_forcing():
    """A 4-arg integrator only samples forcing at the outer-step time; the
    constructor must reject it when forcing is supplied (parity with the
    atmosphere SCM)."""
    import warnings

    def legacy(state, aux, f, dt):  # 4-arg legacy signature
        tend, aux_out = f(state, aux)
        return _apply_ocean_tendencies(state, tend, dt), aux_out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        register_ocean_time_integrator("legacy_4arg_test", legacy)
    # without forcing it is allowed
    OceanColumnModel.create(
        physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=600.0,
        T_profile=jnp.full((_NLEV,), 10.0), H_max=1500.0,
        implicit_vertical_mixing=False, time_integrator="legacy_4arg_test",
    )
    # with forcing it is rejected
    with pytest.raises(ValueError, match="legacy 4-arg"):
        OceanColumnModel.create(
            physics_config=_constant_mixing_cfg(), nlev=_NLEV, dt=600.0,
            T_profile=jnp.full((_NLEV,), 10.0), H_max=1500.0,
            implicit_vertical_mixing=False, time_integrator="legacy_4arg_test",
            forcing=OceanSCMForcing(tau_x=lambda t: 0.1),
        )


def test_forcing_validation():
    with pytest.raises(ValueError):
        validate_ocean_forcing(OceanSCMForcing(tau_x=0.1))  # not callable
    with pytest.raises(ValueError):
        validate_ocean_forcing(OceanSCMForcing(f_c=float("nan")))
    validate_ocean_forcing(default_ocean_forcing())  # ok


# ---------------------------------------------------------------------------
# Differentiability / JIT (end-to-end autodiff is a project invariant)
# ---------------------------------------------------------------------------


def _pure_step(scm, state):
    """One full model step as a pure function of an explicit state."""
    tend, _ = scm._tend_fn(state, None, 0.0)
    star = _apply_ocean_tendencies(state, tend, scm.dt)
    mixed = scm._apply_implicit_vertical_mixing(star, tend.K_v, tend.A_v, scm.dt)
    return scm._apply_coriolis_rotation(mixed)


def test_step_is_differentiable_and_jittable():
    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
    )
    scm = OceanColumnModel.create(
        physics_config=cfg, nlev=_NLEV, dt=1800.0,
        T_profile=jnp.linspace(20.0, 4.0, _NLEV), H_max=2000.0,
        forcing=OceanSCMForcing(f_c=1e-4, tau_x=lambda t: 0.05, q_net=lambda t: -50.0),
    )
    T0 = jnp.linspace(20.0, 4.0, _NLEV)
    z = scm.z_coord

    def loss(T_profile):
        st = make_ocean_column_state(
            _NLEV, T_profile=T_profile, z_coord=z,
        )
        st2 = _pure_step(scm, st)
        return jnp.mean(st2.T.data)

    g = jax.grad(loss)(T0)
    assert g.shape == T0.shape
    assert np.all(np.isfinite(np.asarray(g)))
    # jit the same pure step
    out = jax.jit(loss)(T0)
    assert np.isfinite(float(out))

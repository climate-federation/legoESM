"""Tests for the single-column model driver (issue #277)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics import (
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
    MicrophysicsConfig,
    ConvectionConfig,
    GravityWaveDragConfig,
)
from legoesm.atmosphere.forcing.scm.scm import (
    SCMGrid,
    SCMHistory,
    SingleColumnModel,
    TIME_INTEGRATORS,
    make_column_state,
    make_scm_grid,
    register_time_integrator,
)


NLEV = 16


def _default_T_profile():
    # Roughly tropospheric: 220 K at top -> 295 K at surface.
    return jnp.linspace(220.0, 295.0, NLEV)


def _default_qv_profile():
    return jnp.linspace(1e-6, 1.5e-2, NLEV)


def test_make_column_state_shapes():
    u_profile = jnp.linspace(-8.0, 2.0, NLEV)
    v_profile = jnp.linspace(1.0, 4.0, NLEV)
    state = make_column_state(
        NLEV,
        T_profile=_default_T_profile(),
        q_v_profile=_default_qv_profile(),
        u=u_profile,
        v=v_profile,
    )
    assert state.T.data.shape == (1, 1, 1, NLEV)
    assert jnp.allclose(state.u.data[0, 0, 0], u_profile)
    assert state.v is not None and jnp.allclose(state.v.data[0, 0, 0], v_profile)
    assert state.p_s.data.shape == (1, 1, 1)
    assert state.v is not None and state.v.data.shape == (1, 1, 1, NLEV)
    assert state.tracers is not None and "q_v" in state.tracers
    assert state.tracers["q_v"].data.shape == (1, 1, 1, NLEV)


def test_make_column_state_rejects_wrong_profile_shape():
    with pytest.raises(ValueError):
        make_column_state(NLEV, T_profile=jnp.zeros(NLEV + 1))
    with pytest.raises(ValueError):
        make_column_state(NLEV, T_profile=_default_T_profile(), u=jnp.zeros(NLEV + 1))


def test_scm_grid_attrs():
    grid = make_scm_grid(latitude_deg=45.0, longitude_deg=10.0)
    assert isinstance(grid, SCMGrid)
    assert grid.grid_lat.shape == (1, 1, 1)
    assert grid.grid_lon.shape == (1, 1, 1)
    assert grid.grid_n_columns == 1
    assert grid.grid_shape_2d == (1, 1, 1)
    assert float(grid.grid_lat[0, 0, 0]) == pytest.approx(jnp.deg2rad(45.0))


def test_scm_zero_physics_step_is_identity():
    """With every physics module disabled, the column must not change."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    T0 = _default_T_profile()
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0, T_profile=T0,
    )
    final, hist = scm.run(nsteps=5, save_every=1)
    assert jnp.allclose(final.T.data[0, 0, 0], T0)
    assert hist.T.shape == (5, NLEV)


def test_scm_gray_radiation_cools_stratosphere():
    """Gray radiation should produce non-zero tendencies and physically
    plausible cooling at upper levels under no other physics."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    T0 = _default_T_profile()
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=600.0,
        T_profile=T0, latitude_deg=0.0,
    )
    final, _ = scm.run(nsteps=20)
    # State must remain finite and have changed.
    assert jnp.all(jnp.isfinite(final.T.data))
    assert not jnp.allclose(final.T.data[0, 0, 0], T0)


def test_scm_run_produces_history_with_qv():
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=_default_T_profile(),
        q_v_profile=_default_qv_profile(),
        latitude_deg=15.0,
    )
    final, hist = scm.run(nsteps=10, save_every=2)
    assert isinstance(hist, SCMHistory)
    assert hist.q_v is not None
    assert hist.q_v.shape[1] == NLEV
    # Positivity of q_v preserved.
    assert jnp.all(hist.q_v >= 0.0)
    assert jnp.all(jnp.isfinite(final.T.data))


@pytest.mark.parametrize("integrator", ["forward_euler", "rk2", "rk4"])
def test_scm_swap_time_integrator(integrator):
    """Every registered time integrator must drive the same physics
    pipeline and produce a finite, physically plausible column."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=600.0,
        T_profile=_default_T_profile(), latitude_deg=0.0,
        time_integrator=integrator,
    )
    assert scm.time_integrator == integrator
    final, _ = scm.run(nsteps=4)
    assert jnp.all(jnp.isfinite(final.T.data))


def test_scm_unknown_integrator_raises():
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="Unknown time_integrator"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=300.0,
            T_profile=_default_T_profile(),
            time_integrator="not_a_real_scheme",
        )


def test_register_time_integrator_extends_registry():
    """Custom integrators register and become selectable by name."""
    def double_euler(state, phys, f, dt, t):
        from legoesm.atmosphere.forcing.scm.scm import _apply_tendencies
        tend, phys_out = f(state, phys, t)
        mid = _apply_tendencies(state, tend, 0.5 * dt)
        tend2, phys_out2 = f(mid, phys_out, t + 0.5 * dt)
        return _apply_tendencies(mid, tend2, 0.5 * dt), phys_out2

    register_time_integrator("double_euler_test", double_euler)
    assert "double_euler_test" in TIME_INTEGRATORS

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=_default_T_profile(),
        time_integrator="double_euler_test",
    )
    final, _ = scm.run(nsteps=3)
    assert jnp.all(jnp.isfinite(final.T.data))


def test_scm_swap_physics_scheme():
    """legoESM philosophy: any scheme can be swapped via PhysicsConfig."""
    base = dict(
        nlev=NLEV, dt=300.0, T_profile=_default_T_profile(),
        q_v_profile=_default_qv_profile(), latitude_deg=0.0,
    )
    for turb_scheme in ("none", "louis"):
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme=turb_scheme),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        scm = SingleColumnModel.create(physics_config=cfg, **base)
        final, _ = scm.run(nsteps=3)
        assert jnp.all(jnp.isfinite(final.T.data))


@pytest.mark.parametrize(
    "conv_scheme",
    ["mass_flux", "edmf", "zhang_mcfarlane", "tiedtke", "bechtold", "emanuel"],
)
def test_rk_rejects_stateful_convection(conv_scheme):
    """Profile/scalar-carrying convection schemes must be rejected
    under multi-stage integrators (they read prior conv_prog_profile
    or conv_stoch_state).

    ``emanuel`` is prognostic: its faithful Emanuel-1991 cloud-base
    mass-flux closure reads ``conv_prog_profile[:, -1]`` and relaxes it
    between calls, so it must be rejected under RK like the other
    carry-reading schemes."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme=conv_scheme),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=300.0,
            T_profile=_default_T_profile(),
            q_v_profile=_default_qv_profile(),
            time_integrator="rk4",
        )


@pytest.mark.parametrize("conv_scheme", ["sbm", "kain_fritsch"])
def test_rk_allows_diagnostic_convection(conv_scheme):
    """Diagnostic convection (``del conv_prog_profile``) is safe under
    multi-stage integrators and must NOT be rejected."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme=conv_scheme),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=_default_T_profile(),
        q_v_profile=_default_qv_profile(),
        time_integrator="rk2",
    )


def test_rk_rejects_throttled_convection():
    """update_interval_steps != 1 makes convection non-autonomous in
    step index — unsafe under multi-stage integrators."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="sbm", update_interval_steps=4),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=300.0,
            T_profile=_default_T_profile(),
            time_integrator="rk2",
        )


def test_rk_rejects_stateful_physics():
    """rk2/rk4 must refuse stateful or diurnal physics — they would
    silently reuse the stage-1 carry/time across all stages."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=True),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SingleColumnModel.create(
            physics_config=cfg, nlev=NLEV, dt=300.0,
            T_profile=_default_T_profile(),
            time_integrator="rk2",
        )


def test_scm_materialises_new_tracers_from_physics():
    """When microphysics is enabled, q_c/q_r/q_i are pre-allocated and
    carried, not silently dropped on tendency return."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=_default_T_profile(),
        q_v_profile=_default_qv_profile(),
        latitude_deg=0.0,
    )
    # Pre-allocation contract: condensate species are present from step 0.
    assert "q_c" in scm.state.tracers
    final, _ = scm.run(nsteps=2)
    assert jnp.all(jnp.isfinite(final.T.data))
    for k in ("q_v", "q_c"):
        assert k in final.tracers


# ---------------------------------------------------------------------------
# Swap matrix: every parameterization × every safe time integrator
# ---------------------------------------------------------------------------
#
# Each parametrized case enables ONE scheme in the named category (the
# rest disabled), runs ``_RCE_NSTEPS`` physics steps from a tropical
# initial sounding, and asserts the column is finite and stays in a
# plausible RCE range. The integrator parametrization keeps the
# physics deliberately stateless+autonomous so rk2/rk4 are well
# defined (matching the SCM's gating rules).

_RCE_NLEV = 20
_RCE_DT = 600.0
_RCE_NSTEPS = 24  # ~4 model hours


def _tropical_T_profile(nlev=_RCE_NLEV):
    sigma = jnp.linspace(0.01, 1.0, nlev)
    z = -8.0e3 * jnp.log(jnp.maximum(sigma, 1e-3))
    return jnp.maximum(300.0 - 6.5e-3 * z, 200.0)


def _tropical_qv_profile(nlev=_RCE_NLEV):
    sigma = jnp.linspace(0.01, 1.0, nlev)
    z = -8.0e3 * jnp.log(jnp.maximum(sigma, 1e-3))
    return 1.8e-2 * jnp.exp(-z / 3.0e3)


def _run_and_assert_stable(cfg, integrator="forward_euler"):
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=_RCE_NLEV, dt=_RCE_DT,
        T_profile=_tropical_T_profile(),
        q_v_profile=_tropical_qv_profile(),
        latitude_deg=0.0,
        time_integrator=integrator,
    )
    final, _ = scm.run(nsteps=_RCE_NSTEPS)
    assert jnp.all(jnp.isfinite(final.T.data)), "non-finite T"
    if final.tracers is not None and "q_v" in final.tracers:
        assert jnp.all(jnp.isfinite(final.tracers["q_v"].data)), "non-finite q_v"
    T_col = final.T.data[0, 0, 0]
    assert float(T_col[-1]) > 200.0 and float(T_col[-1]) < 330.0, (
        f"T_sfc out of RCE band: {float(T_col[-1])}"
    )
    assert float(jnp.min(T_col)) > 140.0, (
        f"T floor unphysical: min={float(jnp.min(T_col))}"
    )
    assert float(jnp.max(T_col)) < 350.0, (
        f"T ceiling unphysical: max={float(jnp.max(T_col))}"
    )


def _baseline_cfg(**overrides) -> PhysicsConfig:
    fields = dict(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    fields.update(overrides)
    return PhysicsConfig(**fields)


@pytest.mark.parametrize("rad_scheme", ["gray"])  # rrtmgp needs solver
def test_swap_radiation_schemes(rad_scheme):
    cfg = _baseline_cfg(
        radiation=RadiationConfig(scheme=rad_scheme, diurnal_cycle=False),
    )
    _run_and_assert_stable(cfg)


@pytest.mark.parametrize(
    "conv_scheme",
    ["sbm", "dca", "kuo", "mass_flux", "edmf",
     "zhang_mcfarlane", "kain_fritsch", "emanuel",
     "tiedtke", "bechtold"],
)
def test_swap_convection_schemes(conv_scheme):
    cfg = _baseline_cfg(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme=conv_scheme),
    )
    _run_and_assert_stable(cfg)


@pytest.mark.parametrize(
    "turb_scheme",
    ["smagorinsky", "louis", "tke", "mynn25", "clubb_lite",
     "holtslag_boville", "ysu", "edmf"],
)
def test_swap_turbulence_schemes(turb_scheme):
    cfg = _baseline_cfg(
        turbulence=TurbulenceConfig(scheme=turb_scheme),
    )
    _run_and_assert_stable(cfg)


@pytest.mark.parametrize(
    "micro_scheme",
    ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"],
)
def test_swap_microphysics_schemes(micro_scheme):
    cfg = _baseline_cfg(
        microphysics=MicrophysicsConfig(scheme=micro_scheme),
    )
    _run_and_assert_stable(cfg)


@pytest.mark.parametrize(
    "gwd_scheme",
    ["rayleigh", "lindzen", "mcfarlane", "hines"],
)
def test_swap_gwd_schemes(gwd_scheme):
    cfg = _baseline_cfg(
        gravity_wave_drag=GravityWaveDragConfig(scheme=gwd_scheme),
    )
    _run_and_assert_stable(cfg)


@pytest.mark.parametrize("integrator", ["forward_euler", "rk2", "rk4"])
def test_swap_integrator_on_full_diagnostic_rce(integrator):
    """Every integrator must drive a realistic diagnostic-only RCE
    config (gray rad + SBM conv + Smagorinsky + Kessler + Rayleigh)
    to a stable, plausible column."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="sbm"),
        turbulence=TurbulenceConfig(scheme="smagorinsky"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
    )
    _run_and_assert_stable(cfg, integrator=integrator)


def test_scm_state_remains_pytree_compatible():
    """SCMHistory and HydrostaticState must round-trip through jax.tree."""
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=NLEV, dt=300.0,
        T_profile=_default_T_profile(),
    )
    final, hist = scm.run(nsteps=2)
    leaves = jax.tree_util.tree_leaves(final)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    leaves_h = jax.tree_util.tree_leaves(hist)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves_h)

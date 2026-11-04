"""turbulence=mynn25 lane wiring: SCM cold-start peel, packed carry, refusals
(votes M3, M7, N1-N4, N6, P2-P6)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

jax.config.update("jax_enable_x64", True)

NLEV = 20


def _cfg(scheme="mynn25"):
    return PhysicsConfig(turbulence=TurbulenceConfig(scheme=scheme))


def _scm(**kw):
    T0 = jnp.linspace(220.0, 300.0, NLEV)
    q0 = 1.0e-2 * jnp.exp(-jnp.linspace(6.0, 0.0, NLEV))
    args = dict(physics_config=_cfg(), nlev=NLEV, dt=60.0, T_profile=T0,
                q_v_profile=q0, q_c_profile=jnp.zeros(NLEV), u=5.0, v=-2.0,
                cell_area_m2=(25.0e3) ** 2, land_frac=0.0)
    args.update(kw)
    return SingleColumnModel.create(**args)


def test_scm_first_step_runs_the_cold_start_build_then_never_again():
    scm = _scm()
    assert not scm._mynn_initialized
    assert scm.phys_state.qke.shape == (1, 3, NLEV)
    s0, p0, t0 = scm.state, scm.phys_state, scm.t_seconds
    _, ps_init = scm._tend_fn_init(s0, p0, t0)
    _, ps_norm = scm._tend_fn(s0, p0, t0)
    # the two builds differ on this state (else the peel test proves nothing)
    assert not np.allclose(ps_init.qke, ps_norm.qke)
    scm.step()
    assert scm._mynn_initialized
    np.testing.assert_array_equal(scm.phys_state.qke, ps_init.qke)
    s1, p1, t1 = scm.state, scm.phys_state, scm.t_seconds
    _, ps2_norm = scm._tend_fn(s1, p1, t1)
    scm.step()
    np.testing.assert_array_equal(scm.phys_state.qke, ps2_norm.qke)
    assert np.all(np.isfinite(np.asarray(scm.state.T.data)))


def test_scm_pure_step_refuses_before_the_cold_start():
    scm = _scm()
    with pytest.raises(ValueError, match="cold-start"):
        scm.pure_step(scm.state, scm.phys_state, scm.t_seconds)


def test_scm_without_cell_area_refuses():
    scm = _scm(cell_area_m2=None)
    with pytest.raises(ValueError, match="grid_area"):
        scm.step()


def test_scm_without_cloud_water_refuses():
    scm = _scm(q_c_profile=None)
    if "q_c" in (scm.state.tracers or {}):
        pytest.skip("SCM always allocates q_c")
    with pytest.raises(ValueError, match="q_c"):
        scm.step()


def test_scm_direct_construction_needs_both_builds():
    scm = _scm()
    with pytest.raises(ValueError, match="physics_fn_init"):
        SingleColumnModel(
            physics_fn=scm.physics_fn, state=scm.state,
            phys_state=scm.phys_state, grid=scm.grid,
            sigma_coord=scm.sigma_coord, dt=60.0)


def test_make_physics_refuses_mynn25_without_or_with_wrong_flag():
    with pytest.raises(ValueError, match="mynn_initflag"):
        make_physics(_cfg(), model_type="hydrostatic", dt=60.0)
    with pytest.raises(ValueError, match="mynn_initflag"):
        make_physics(_cfg(), model_type="hydrostatic", dt=60.0,
                     mynn_initflag=1)
    with pytest.raises(ValueError, match="mynn_initflag"):
        make_physics(_cfg("louis"), model_type="hydrostatic", dt=60.0,
                     mynn_initflag=False)


def test_make_physics_refuses_mynn25_with_macmic_subcycling():
    with pytest.raises(ValueError, match="cld_macmic_num_steps"):
        make_physics(_cfg(), model_type="mpas", dt=60.0, mynn_initflag=False,
                     cld_macmic_num_steps=2)


def test_compiled_pipeline_refuses_mynn25():
    from types import SimpleNamespace
    from legoesm.driver.physics_pipeline import _resolve_turbulence
    with pytest.raises(NotImplementedError, match="mynn25"):
        _resolve_turbulence(SimpleNamespace(turbulence="mynn25"))


def test_bridge_routes_cloud_water_and_drops_absent_ice():
    """N6: the hydrostatic bridge hands the port's dqc to q_c; with no q_i
    tracer no ice tendency is emitted."""
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics)
    scm = _scm()
    assert "q_i" not in scm.state.tracers
    fn = make_turbulence_physics(_cfg().turbulence, "hydrostatic", 60.0,
                                 mynn_initflag=True)
    ps = scm.phys_state._replace(dyn_w_ls=jnp.zeros_like(scm.state.T.data))
    tend, upd = fn(scm.state, scm.grid, scm.sigma_coord, phys_state=ps)
    assert set(tend.tracer_tendencies) == {"q_v", "q_c"}
    assert tend.tracer_tendencies["q_c"].data.shape == scm.state.T.data.shape
    assert upd["qke"].shape == (1, 3, NLEV)


def test_scm_set_time_reaches_the_cold_start_build():
    """codex R3: step 1 runs the cold-start physics, so it must get the clock."""
    seen = []
    scm = _scm()
    scm.physics_fn_init.set_time = lambda d, s: seen.append((d, s))
    scm.set_time(10.0, 3600.0)
    assert seen == [(10.0, 3600.0)]


def test_lane_without_vapour_tracer_refuses():
    """codex R3 round 3: dqv must have a q_v tracer to land on."""
    from legoesm.atmosphere.physics.turbulence.integration import (
        mynn_lane_kwargs)
    scm = _scm()
    tr = {k: v for k, v in scm.state.tracers.items() if k != "q_v"}
    st = scm.state._replace(tracers=tr)
    T = scm.state.T.data.reshape(1, NLEV)
    with pytest.raises(ValueError, match="q_v"):
        mynn_lane_kwargs(st, scm.grid, scm.sigma_coord, scm.phys_state,
                         T, T, T, 1, False)


def test_scm_direct_construction_refuses_multistage_integrator():
    """codex R3 round 4: RK stages would repeat the cold start."""
    scm = _scm()
    with pytest.raises(ValueError, match="forward_euler"):
        SingleColumnModel(
            physics_fn=scm.physics_fn, physics_fn_init=scm.physics_fn_init,
            state=scm.state, phys_state=scm.phys_state, grid=scm.grid,
            sigma_coord=scm.sigma_coord, dt=60.0, time_integrator="rk2")

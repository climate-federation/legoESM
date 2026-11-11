"""MPAS column-lane wiring of the OpenIFS 49r1 WMS drag and METHOX.

The kernels are oracle-checked in test_ifs_wms_ozone_oracle.py; these tests
execute the MPAS factories the IFS deck runs (gravity-wave drag factory and
make_physics(model_type="mpas")) and show each wired input is consumed.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.atmosphere.physics.gravity_wave_drag import integration as gwd_int  # noqa: E402
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig  # noqa: E402
from legoesm.atmosphere.physics.gravity_wave_drag.ifs_wms import wms_call_interval  # noqa: E402
from legoesm.atmosphere.physics.methane_oxidation import (  # noqa: E402
    MethoxConfig, make_methox_physics, methox_q_lim, methox_tendency,
)
from legoesm.grids.vertical import make_cam6_l32_levels  # noqa: E402
from legoesm.grids.voronoi import reconstruct_cell_velocity  # noqa: E402

DT = 300.0


@pytest.fixture(scope="module")
def setup():
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    coord = make_cam6_l32_levels()
    state = held_suarez_init_mpas(mesh, coord, T_init=290.0, perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    rng = np.random.default_rng(0)
    u_edge = jnp.asarray(rng.standard_normal(state.u.data.shape) * 8.0)
    T = jnp.asarray(290.0 - 80.0 * np.linspace(1.0, 0.0, nlev)[None, :]
                    + rng.standard_normal((ncell, nlev)))
    q = jnp.asarray(np.full((ncell, nlev), 2.0e-6))     # below RQLIM: oxidation active
    state = state._replace(
        u=state.u.replace(data=u_edge), T=state.T.replace(data=T),
        tracers={nm: state.T.replace(data=val, name=nm, units="kg/kg")
                 for nm, val in (("q_v", q), ("q_c", jnp.zeros_like(q)),
                                 ("q_r", jnp.zeros_like(q)))})
    u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)
    col_state = state._replace(u=state.u.replace(data=u_cell, name="u"),
                               v=state.u.replace(data=v_cell, name="v"))
    mesh = mesh._replace(land_frac=jnp.ones(ncell))
    return mesh, coord, col_state


def _seed(cfg, state, coord):
    return SimpleNamespace(gwd_spectrum=gwd_int.gwd_carry_seed(
        cfg, state.T.data.shape[0], coord.n_levels, state.T.data.dtype))


@pytest.mark.parametrize("scheme", ["ifs_wms", "mcfarlane+ifs_wms"])
def test_mpas_wms_factory_retains_recomputes_and_reads_area(setup, scheme):
    mesh, coord, state = setup
    cfg = GravityWaveDragConfig(scheme=scheme)
    fn = gwd_int._make_mpas_gwd(cfg, DT)
    t1, carry = fn(state, mesh, coord, phys_state=_seed(cfg, state, coord))
    du1 = np.asarray(t1.du_dt.data)
    assert np.abs(du1).max() > 0.0
    assert carry.shape == gwd_int.gwd_carry_shape(cfg, *state.T.data.shape)
    sheared = state._replace(u=state.u.replace(data=state.u.data * 1.2),
                             v=state.v.replace(data=state.v.data * 1.2))
    n_keep = int(wms_call_interval(DT) / DT) - 1
    if scheme == "ifs_wms":                       # retained between recomputes
        for _ in range(n_keep):
            t, carry = fn(sheared, mesh, coord, phys_state=SimpleNamespace(gwd_spectrum=carry))
            np.testing.assert_array_equal(np.asarray(t.du_dt.data), du1)
        t, carry = fn(sheared, mesh, coord, phys_state=SimpleNamespace(gwd_spectrum=carry))
        assert np.abs(np.asarray(t.du_dt.data) - du1).max() > 1e-9
    # PGAW only matters on fine grids (49r1 resolution factor, dx < 1/ZDXA)
    fine = mesh._replace(areaCell=jnp.asarray(mesh.areaCell) * 1.0e-6)
    t4, _ = fn(state, fine, coord, phys_state=_seed(cfg, state, coord))
    assert np.abs(np.asarray(t4.du_dt.data) - du1).max() > 1e-9
    with pytest.raises(ValueError, match="carry"):
        fn(state, mesh, coord, phys_state=None)


def test_mpas_methox_module_ch4_source(setup):
    """R5a: external GHG -> the forcing's CH4, absent = error; constant GHG
    -> the run's constant CH4 (what radiation reads in that mode)."""
    mesh, coord, state = setup
    q = state.tracers["q_v"].data
    p = coord.pressure_at_full(state.p_s.data)

    def expect(ch4):
        return methox_tendency(q, p, MethoxConfig()._replace(
            q_lim_kg_kg=methox_q_lim(jnp.asarray(ch4))))

    ext = make_methox_physics(MethoxConfig(enabled=True))
    got = ext(state, mesh, coord,
              forcing={"ghg_vmr": {"ch4": 1.0e-6}}).tracer_tendencies["q_v"].data
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect(1.0e-6)), rtol=1e-14, atol=0)
    with pytest.raises(ValueError, match="carries no CH4"):
        ext(state, mesh, coord, forcing={})
    const = make_methox_physics(MethoxConfig(enabled=True, ch4_vmr_constant=1.9e-6))
    got2 = const(state, mesh, coord, forcing={}).tracer_tendencies["q_v"].data
    np.testing.assert_allclose(np.asarray(got2), np.asarray(expect(1.9e-6)), rtol=1e-14, atol=0)
    assert np.abs(np.asarray(got2) - np.asarray(got)).max() > 0.0


def test_make_physics_mpas_adds_methox_only_when_enabled(setup):
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    mesh, coord, state = setup
    forcing = {"T_sfc": jnp.full((state.T.data.shape[0],), 292.0)}

    def qv(enabled):
        cfg = PhysicsConfig(radiation=RadiationConfig(scheme="none"),
                            methox=MethoxConfig(enabled=enabled, ch4_vmr_constant=1.9e-6))
        out = make_physics(cfg, model_type="mpas", dt=DT)(state, mesh, coord, forcing=forcing)
        comb = out[0] if isinstance(out, tuple) else out
        tt = comb.tracer_tendencies or {}
        return np.asarray(getattr(tt.get("q_v"), "data", 0.0))

    want = make_methox_physics(MethoxConfig(enabled=True, ch4_vmr_constant=1.9e-6))(
        state, mesh, coord, forcing=forcing).tracer_tendencies["q_v"].data
    np.testing.assert_allclose(qv(True) - qv(False), np.asarray(want), rtol=1e-12, atol=1e-30)
    with pytest.raises(ValueError, match="MPAS column lane"):
        make_physics(PhysicsConfig(methox=MethoxConfig(enabled=True, ch4_vmr_constant=1.9e-6)),
                     model_type="hydrostatic", dt=DT)


def test_validate_strict_refuses_ifs_schemes_off_column_lane():
    from legoesm.driver.config import ExperimentConfig
    with pytest.raises(ValueError, match="ifs_methox runs only"):
        ExperimentConfig(ifs_methox=True).validate_strict()
    with pytest.raises(ValueError, match="ifs_wms drag runs only"):
        ExperimentConfig(gravity_wave_drag="ifs_wms").validate_strict()

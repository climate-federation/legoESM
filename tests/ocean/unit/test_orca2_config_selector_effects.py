"""Record-free execution tests for the ORCA2 config-tail selectors.

These tests deliberately target model guards and small written-order helpers.
They need neither the ORCA2 deck nor a NEMO record, and each assertion stops
failing if the named selector is ignored at its consumer.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _nemo_ws_stage1_ldf_source_order,
    _nemo_ws_stage_pair,
    _nemo_ws_stage_r3uv,
)
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_horizontal_viscosity,
)
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ocean.vertical import create_ocean_z_star


class _SelectorTouched(RuntimeError):
    pass


class _BoolProbe:
    """A non-JAX config sentinel that raises when a live branch consumes it."""

    def __init__(self, field):
        self.field = field

    def __bool__(self):
        raise _SelectorTouched(self.field)

    def __eq__(self, _other):
        raise _SelectorTouched(self.field)

    def __ne__(self, _other):
        raise _SelectorTouched(self.field)

    def __repr__(self):
        return f"_BoolProbe({self.field!r})"


def _vortex_config():
    return recipe.build_nemo_testcase_card("VORTEX-zco").recipe.model_config


def _record_free_l8_config():
    cfg = _vortex_config()
    selected = recipe.orca2_l8_krhs_selection()
    selected.update(
        rk3_stage3_heat_nemo_written=True,
        rk3_stage3_salt_nemo_written=True,
        rk3_stage3_tracer_nemo_stored_thickness=True,
        tracer_time_integrator="rk3_ws",
        tracer_advection="fct2",
        gm_redi=recipe.orca2_l2_isoneutral_gm_redi(),
    )
    return cfg._replace(**selected)


@pytest.fixture(scope="module")
def gyre_card():
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64())
    return recipe.build_nemo_testcase_card("GYRE-zco")


@pytest.mark.parametrize("field", [
    "nemo_hpg_sco_fold_geometry",
    "rk3_vector_raw_vmask",
])
def test_vortex_step_consumes_each_early_stage_selector(field):
    card = recipe.build_nemo_testcase_card("VORTEX-zco")
    cfg = card.recipe.model_config._replace(**{field: _BoolProbe(field)})
    with pytest.raises(_SelectorTouched, match=field):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
        ).step(card.recipe.initial_state, dt=card.dt_s)


@pytest.mark.parametrize("field", [
    "nemo_wzv_stored_r3_ratios",
    "nemo_zad_written_association",
    "nemo_stage2_momentum_wzv_half_step",
    "nemo_stage_momentum_wzv_stored_kmm_ratios",
    "nemo_ldf_stored_f_area",
    "nemo_ldf_fold_rows",
    "nemo_ldf_stage_r3f",
    "rk3_stage_transport_raw_vmask",
    "rk3_stage_transport_fold_exchanges",
    "fct_fold_exchange",
    "rk3_stage_wzv_nemo_clock_pair",
    "rk3_stage_tracer_wzv_stored_kmm_ratios",
    "rk3_stage_transport_stored_r3uv",
    "vmix_eos_use_config_rho0",
])
def test_gyre_step_consumes_each_staged_selector(gyre_card, field):
    cfg = gyre_card.recipe.model_config._replace(**{field: _BoolProbe(field)})
    with pytest.raises(_SelectorTouched, match=field):
        LatLonCGridOceanModel(
            gyre_card.recipe.grid, gyre_card.recipe.z_coord, cfg,
        ).step(gyre_card.recipe.initial_state, dt=gyre_card.dt_s)


@pytest.mark.parametrize("field", [
    "redi_flux_geometry_stage",
    "nemo_slope_north_fold",
    "redi_a33_source_ordering",
    "redi_flux_stored_face_ratios",
])
def test_gyre_step_consumes_each_new_gm_redi_selector(gyre_card, field):
    gm = gyre_card.recipe.model_config.gm_redi._replace(
        **{field: _BoolProbe(field)})
    cfg = gyre_card.recipe.model_config._replace(gm_redi=gm)
    with pytest.raises(_SelectorTouched, match=field):
        LatLonCGridOceanModel(
            gyre_card.recipe.grid, gyre_card.recipe.z_coord, cfg,
        ).step(gyre_card.recipe.initial_state, dt=gyre_card.dt_s)


@pytest.mark.parametrize("field", [
    "bbl_nemo_stage3_krhs",
    "bbl_nemo_pivot_ghost",
    "rk3_stage3_salt_nemo_written",
    "rk3_stage3_tracer_krhs_nemo_order",
    "rk3_stage3_geothermal_nemo_written",
    "rk3_stage3_geothermal_after_ldf_nemo_order",
    "rk3_stage3_damping_after_geothermal_nemo_order",
])
def test_each_new_l8_prerequisite_is_consumed_by_the_model_guard(field):
    cfg = _record_free_l8_config()
    LatLonCGridOceanModel._validate_config(cfg)
    with pytest.raises(ValueError):
        LatLonCGridOceanModel._validate_config(cfg._replace(**{field: False}))


def test_l8_and_l9_selectors_each_reach_their_own_composition_guard():
    cfg = _vortex_config()
    with pytest.raises(ValueError, match="rk3_stage3_l8_nemo_order"):
        LatLonCGridOceanModel._validate_config(
            cfg._replace(rk3_stage3_l8_nemo_order=True))
    with pytest.raises(ValueError, match="rk3_stage3_l9_eiv_nemo_order"):
        LatLonCGridOceanModel._validate_config(
            cfg._replace(rk3_stage3_l9_eiv_nemo_order=True))


def test_vector_rhs_and_stage1_ldf_selectors_reach_distinct_guards():
    cfg = _vortex_config()
    with pytest.raises(ValueError, match="requires vector-form"):
        LatLonCGridOceanModel._validate_config(
            cfg._replace(rk3_vector_rhs_nemo_order=True))

    vector = cfg._replace(momentum_advection="vector_invariant")
    with pytest.raises(ValueError, match="requires rk3_vector_rhs"):
        LatLonCGridOceanModel._validate_config(
            vector._replace(rk3_stage1_ldf_source_order=True))
    assert _nemo_ws_stage1_ldf_source_order(
        vector._replace(rk3_stage1_ldf_source_order=True,
                        rk3_vector_rhs_nemo_order=True), False) is True
    with pytest.raises(ValueError, match="private stage-RHS"):
        _nemo_ws_stage1_ldf_source_order(
            vector._replace(rk3_stage1_ldf_source_order=True,
                            rk3_vector_rhs_nemo_order=True), True)


def test_stage_wzv_clock_pair_selector_changes_the_carried_pair():
    cfg = _vortex_config()
    before = jnp.asarray([[0.25]])
    after = jnp.asarray([[0.75]])
    assert _nemo_ws_stage_pair(
        cfg._replace(rk3_stage_r3uv_nemo_literal=False), 1,
        (before, after)) is None
    got = _nemo_ws_stage_pair(
        cfg._replace(rk3_stage_r3uv_nemo_literal=True), 1,
        (before, after))
    assert got[0] == 1
    np.testing.assert_array_equal(got[1], before)
    np.testing.assert_array_equal(got[2], after)


def test_stage_wzv_clock_pair_changes_the_executed_wzv_clock(
        gyre_card, monkeypatch):
    """Pin the dt/after-level pair handed to the actual continuity solve."""
    state = gyre_card.recipe.initial_state
    grid, z_coord = gyre_card.recipe.grid, gyre_card.recipe.z_coord
    h_ref = jnp.asarray(z_coord.nemo_e3t_0, dtype=jnp.float64)
    u_mask = jnp.broadcast_to(
        state.u_mask.data[..., None], state.u.data.shape)
    v_mask = jnp.broadcast_to(
        state.v_mask.data[..., None], state.v.data.shape)
    seen = []

    def fake_wzv(*args, **kwargs):
        clock = float(args[9])
        after_level = int(kwargs["stage_after_level"])
        seen.append((clock, after_level))
        value = clock + after_level
        return jnp.full_like(h_ref, value), None, None

    monkeypatch.setattr(omlc, "nemo_qco_wzv_operands", fake_wzv)

    def selected(flag):
        cfg = gyre_card.recipe.model_config._replace(
            rk3_stage_wzv_nemo_clock_pair=flag)
        return omlc._nemo_ws_stage_transport(
            (state.u.data, state.v.data), h_ref, 1,
            eta_stage=state.eta.data, stored_h_stage=True,
            h_ref=h_ref, Hu_avg=jnp.zeros_like(state.u.data[..., 0]),
            Hv_avg=jnp.zeros_like(state.v.data[..., 0]),
            u_mask_3d=u_mask, v_mask_3d=v_mask, grid=grid,
            z_coord=z_coord, H_bathy=state.H_bathy.data,
            config=cfg, dt=10800.0, eta_before=state.eta.data,
            eta_after=state.eta.data, literal_wzv=True)[2]

    on = np.asarray(selected(True))
    off = np.asarray(selected(False))
    assert seen == [(5400.0, 2), (10800.0, 0)]
    assert float(on[0, 0, 0]) == 5402.0
    assert float(off[0, 0, 0]) == 10800.0
    assert not np.array_equal(on, off)


def test_nemo_ldf_fold_rows_changes_the_executed_fold_operator():
    """The config bit must change the literal folded viscosity result."""
    grid = create_synthetic_tripole(8, 12, dtype=jnp.float64)
    rng = np.random.default_rng(12)
    u = jnp.asarray(rng.normal(size=(8, 13, 3)))
    v = jnp.asarray(rng.normal(size=(9, 12, 3)))
    h = jnp.asarray(rng.uniform(1.0, 3.0, size=(8, 12, 3)))
    e3u = jnp.asarray(rng.uniform(1.0, 3.0, size=u.shape))
    e3v = jnp.asarray(rng.uniform(1.0, 3.0, size=v.shape))
    e3f = jnp.asarray(rng.uniform(1.0, 3.0, size=(9, 13, 3)))
    thickness = (h, e3u, e3v, e3f, e3u + 0.25, e3v + 0.5)
    z_coord = create_ocean_z_star(3, 300.0, 100.0, 100.0)
    base = recipe.build_nemo_testcase_card(
        "GYRE-zco").recipe.model_config._replace(
        lateral_viscosity_coefficient_source="nemo_constant",
        nemo_ldf_stored_f_area=False,
    )
    mask = jnp.ones((8, 12), dtype=jnp.float64)

    def selected(flag):
        return _bc_horizontal_viscosity(
            jnp.zeros_like(u), jnp.zeros_like(v), u, v, grid, mask,
            jnp.ones_like(u), jnp.ones_like(v),
            base._replace(nemo_ldf_fold_rows=flag), z_coord,
            300.0 * mask, 10.0, h_k=h,
            ldf_thickness_operands=thickness,
        )[:2]

    on_u, on_v = (np.asarray(x) for x in selected(True))
    off_u, off_v = (np.asarray(x) for x in selected(False))
    assert np.count_nonzero(on_u.view(np.uint64) != off_u.view(np.uint64)) == 39
    assert np.count_nonzero(on_v.view(np.uint64) != off_v.view(np.uint64)) == 36
    assert on_u.view(np.uint64)[7, 0, 0] == np.uint64(4498141291464064851)
    assert off_u.view(np.uint64)[7, 0, 0] == np.uint64(4503684672064473110)


def _ws_rk3_tripole_pivot_fixture():
    from legoesm.grids.tripole import create_synthetic_tripole_pivot
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    grid = create_synthetic_tripole_pivot(8, 12, dtype=jnp.float64)
    base_z = create_ocean_z_star(3, 600.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, base_z, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=600.0)
    z_coord = create_partial_cell_coordinate(base_z, state.H_bathy.data)
    rng = np.random.default_rng(420)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            0.02 * rng.standard_normal(state.u.data.shape))
            * state.u_mask.data[..., None]),
        v=state.v.replace(data=jnp.asarray(
            0.02 * rng.standard_normal(state.v.data.shape))
            * state.v_mask.data[..., None]),
        T=state.T.replace(data=state.T.data + jnp.asarray(
            rng.standard_normal(state.T.data.shape))),
        S=state.S.replace(data=state.S.data + jnp.asarray(
            0.1 * rng.standard_normal(state.S.data.shape))),
    )
    base = LatLonCGridOceanConfig.from_flat(
        barotropic_solver="explicit_substep",
        tracer_time_integrator="rk3_ws",
        momentum_time_integrator="rk3_ws",
        tracer_advection="fct2",
        momentum_advection="flux_form",
        momentum_flux_scheme="none",
        vertical_momentum_scheme="none",
        barotropic_slow_forcing_depth_evaluation="min_rule_live",
        barotropic_face_depth="nemo_ssh_avg",
        n_barotropic_substeps=3,
        enable_runtime_checks=False,
    )
    return grid, z_coord, state, base


def test_fct_fold_exchange_changes_a_full_tripole_ws_rk3_step():
    """Exercise the model call-site on a record-free stored-pivot fold."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, base = _ws_rk3_tripole_pivot_fixture()

        def selected(flag):
            return LatLonCGridOceanModel(
                grid, z_coord, base._replace(fct_fold_exchange=flag),
            ).step(state, 600.0)

        on, off = selected(True), selected(False)
        on_t, off_t = np.asarray(on.T.data), np.asarray(off.T.data)
        on_s, off_s = np.asarray(on.S.data), np.asarray(off.S.data)
        assert np.count_nonzero(
            on_t.view(np.uint64) != off_t.view(np.uint64)) == 29
        assert np.count_nonzero(
            on_s.view(np.uint64) != off_s.view(np.uint64)) == 48
        assert on_t.view(np.uint64)[6, 6, 1] == np.uint64(4625908986858152799)
        assert off_t.view(np.uint64)[6, 6, 1] == np.uint64(4625908986858152689)
    finally:
        set_policy(prev_policy)


def test_rk3_end_of_stage_pivot_fold_changes_a_full_tripole_ws_rk3_step():
    """Exercise rk3_end_of_stage_pivot_fold on a record-free tripole step."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, base = _ws_rk3_tripole_pivot_fixture()

        def selected(flag):
            return LatLonCGridOceanModel(
                grid, z_coord, base._replace(rk3_end_of_stage_pivot_fold=flag),
            ).step(state, 600.0)

        on, off = selected(True), selected(False)
        on_t, off_t = np.asarray(on.T.data), np.asarray(off.T.data)
        on_u, off_u = np.asarray(on.u.data), np.asarray(off.u.data)
        assert np.count_nonzero(
            on_t.view(np.uint64) != off_t.view(np.uint64)) == 62
        assert np.count_nonzero(
            on_u.view(np.uint64) != off_u.view(np.uint64)) == 48
        assert on_t.view(np.uint64)[5, 7, 0] == np.uint64(4626600210828169908)
        assert off_t.view(np.uint64)[5, 7, 0] == np.uint64(4626600210828170265)
        assert on_u.view(np.uint64)[6, 0, 0] == np.uint64(4580213822658509297)
        assert off_u.view(np.uint64)[6, 0, 0] == np.uint64(4580213822683598223)
    finally:
        set_policy(prev_policy)


def test_rk3_stage_transport_fold_exchanges_changes_a_full_tripole_ws_rk3_step():
    """Exercise rk3_stage_transport_fold_exchanges on a record-free tripole step."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, base = _ws_rk3_tripole_pivot_fixture()

        def selected(flag):
            return LatLonCGridOceanModel(
                grid, z_coord,
                base._replace(rk3_stage_transport_fold_exchanges=flag),
            ).step(state, 600.0)

        on, off = selected(True), selected(False)
        on_t, off_t = np.asarray(on.T.data), np.asarray(off.T.data)
        on_s, off_s = np.asarray(on.S.data), np.asarray(off.S.data)
        on_u, off_u = np.asarray(on.u.data), np.asarray(off.u.data)
        on_v, off_v = np.asarray(on.v.data), np.asarray(off.v.data)
        assert np.count_nonzero(
            on_t.view(np.uint64) != off_t.view(np.uint64)) == 22
        assert np.count_nonzero(
            on_s.view(np.uint64) != off_s.view(np.uint64)) == 20
        assert np.count_nonzero(
            on_u.view(np.uint64) != off_u.view(np.uint64)) == 27
        assert np.count_nonzero(
            on_v.view(np.uint64) != off_v.view(np.uint64)) == 21
        assert on_t.view(np.uint64)[6, 6, 2] == np.uint64(4624101206129409913)
        assert off_t.view(np.uint64)[6, 6, 2] == np.uint64(4624101206129407921)
        assert on_u.view(np.uint64)[7, 0, 0] == np.uint64(4575974430849054534)
        assert off_u.view(np.uint64)[7, 0, 0] == np.uint64(4575974430849064564)
    finally:
        set_policy(prev_policy)


def test_rk3_vector_raw_vmask_changes_a_full_tripole_ws_rk3_step():
    """Exercise rk3_vector_raw_vmask on a record-free tripole step."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, base = _ws_rk3_tripole_pivot_fixture()

        def selected(flag):
            return LatLonCGridOceanModel(
                grid, z_coord, base._replace(rk3_vector_raw_vmask=flag),
            ).step(state, 600.0)

        on, off = selected(True), selected(False)
        on_t, off_t = np.asarray(on.T.data), np.asarray(off.T.data)
        on_s, off_s = np.asarray(on.S.data), np.asarray(off.S.data)
        on_u, off_u = np.asarray(on.u.data), np.asarray(off.u.data)
        on_v, off_v = np.asarray(on.v.data), np.asarray(off.v.data)
        assert np.count_nonzero(
            on_t.view(np.uint64) != off_t.view(np.uint64)) == 22
        assert np.count_nonzero(
            on_s.view(np.uint64) != off_s.view(np.uint64)) == 20
        assert np.count_nonzero(
            on_u.view(np.uint64) != off_u.view(np.uint64)) == 27
        assert np.count_nonzero(
            on_v.view(np.uint64) != off_v.view(np.uint64)) == 21
        assert on_t.view(np.uint64)[6, 6, 2] == np.uint64(4624101206129409913)
        assert off_t.view(np.uint64)[6, 6, 2] == np.uint64(4624101206129407921)
        assert on_v.view(np.uint64)[7, 0, 0] == np.uint64(13796485353636929082)
        assert off_v.view(np.uint64)[7, 0, 0] == np.uint64(13796485353636929150)
    finally:
        set_policy(prev_policy)


def _stage3_tracer_step_fixture():
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig,
    )
    from legoesm.ocean.state import (
        LatLonCGridOceanConfig,
        OceanSurfaceForcing,
    )
    from legoesm.ocean.vertical import create_partial_cell_coordinate

    grid = create_synthetic_tripole(4, 6, dtype=jnp.float64)
    base_z = create_ocean_z_star(3, 300.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, base_z, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=300.0)
    z_coord = create_partial_cell_coordinate(base_z, state.H_bathy.data)
    rng = np.random.default_rng(92)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            0.05 * rng.standard_normal(state.u.data.shape))),
        v=state.v.replace(data=jnp.asarray(
            0.05 * rng.standard_normal(state.v.data.shape))),
        T=state.T.replace(data=state.T.data + jnp.asarray(
            0.4 * rng.standard_normal(state.T.data.shape))),
        S=state.S.replace(data=state.S.data + jnp.asarray(
            0.1 * rng.standard_normal(state.S.data.shape))),
        eta=state.eta.replace(data=jnp.asarray(
            0.03 * rng.standard_normal(state.eta.data.shape))),
    )
    base_cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver="explicit_substep",
        momentum_time_integrator="rk3_ws",
        tracer_time_integrator="rk3_ws",
        tracer_advection="fct2",
        momentum_advection="flux_form",
        momentum_flux_scheme="none",
        vertical_momentum_scheme="none",
        barotropic_slow_forcing_depth_evaluation="min_rule_live",
        barotropic_face_depth="nemo_ssh_avg",
        surface_stress_implicit=True,
        rk3_stage3_heat_nemo_written=True,
        rk3_stage3_salt_nemo_written=True,
        rk3_stage3_geothermal_nemo_written=False,
        rk3_stage3_tracer_krhs_nemo_order=True,
        rk3_stage3_tracer_nemo_stored_thickness=True,
        n_barotropic_substeps=2,
        enable_runtime_checks=False,
        physics=OceanPhysicsConfig(
            lateral_mixing=LateralMixingConfig(scheme="none"),
            shortwave_penetration=ShortwavePenetrationConfig(
                scheme="nemo_qsr_rgb", nemo_time_step_s=300.0),
        ),
    )
    forcing = OceanSurfaceForcing(
        q_nonsolar=jnp.asarray(
            -120.0 + 15.0 * rng.standard_normal((4, 6)), dtype=jnp.float64),
        sw_down=jnp.asarray(
            180.0 + 20.0 * rng.standard_normal((4, 6)), dtype=jnp.float64),
        chl=jnp.full((4, 6), 0.1, dtype=jnp.float64),
        salt_flux_sfx=jnp.asarray(
            2.5e-5 + 1.0e-5 * rng.standard_normal((4, 6)), dtype=jnp.float64),
        geothermal_heat_flux_mwm2=jnp.asarray(
            65.0 + 10.0 * rng.standard_normal((4, 6)), dtype=jnp.float64),
    )
    return grid, z_coord, state, forcing, base_cfg


def test_surface_salt_flux_sfx_changes_written_stage3_salt_source():
    """Pin the recorded sfx operand's stage-3 written source effect via model step."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, forcing, base_cfg = _stage3_tracer_step_fixture()
        model = LatLonCGridOceanModel(grid, z_coord, base_cfg)
        on = model.step(state, 300.0, surface_forcing=forcing)
        off = model.step(
            state, 300.0,
            surface_forcing=forcing._replace(
                salt_flux_sfx=jnp.zeros((4, 6), dtype=jnp.float64)),
        )
        on_s = np.asarray(on.S.data).view(np.uint64)
        off_s = np.asarray(off.S.data).view(np.uint64)
        assert np.count_nonzero(on_s != off_s) == 48
        assert on_s[0, 0, 0] == np.uint64(4630119873119761462)
        assert off_s[0, 0, 0] == np.uint64(4630119872943128958)
    finally:
        set_policy(prev_policy)


def test_rk3_stage3_salt_geothermal_and_krhs_order_change_model_step():
    """Exercise rk3_stage3_salt_nemo_written, rk3_stage3_geothermal_nemo_written, and rk3_stage3_tracer_krhs_nemo_order on a record-free step."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, forcing, base_cfg = _stage3_tracer_step_fixture()

        def step_with(**kw):
            return LatLonCGridOceanModel(
                grid, z_coord, base_cfg._replace(**kw),
            ).step(state, 300.0, surface_forcing=forcing)

        salt_on = step_with(rk3_stage3_tracer_krhs_nemo_order=False)
        salt_off = step_with(
            rk3_stage3_tracer_krhs_nemo_order=False,
            rk3_stage3_salt_nemo_written=False,
        )
        s_on = np.asarray(salt_on.S.data).view(np.uint64)
        s_off = np.asarray(salt_off.S.data).view(np.uint64)
        assert np.count_nonzero(s_on != s_off) == 48
        assert s_on[0, 0, 0] == np.uint64(4630119873119761461)
        assert s_off[0, 0, 0] == np.uint64(4630119872943128958)

        geo_on = step_with(
            rk3_stage3_tracer_krhs_nemo_order=False,
            rk3_stage3_geothermal_nemo_written=True,
        )
        t_geo_on = np.asarray(geo_on.T.data).view(np.uint64)
        t_geo_off = np.asarray(salt_on.T.data).view(np.uint64)
        assert np.count_nonzero(t_geo_on != t_geo_off) == 48
        assert t_geo_on[0, 0, 1] == np.uint64(4626009601048700807)
        assert t_geo_off[0, 0, 1] == np.uint64(4626009601048700780)

        krhs_on = LatLonCGridOceanModel(grid, z_coord, base_cfg).step(
            state, 300.0, surface_forcing=forcing)
        t_krhs_on = np.asarray(krhs_on.T.data).view(np.uint64)
        assert np.count_nonzero(t_krhs_on != t_geo_off) == 11
        assert t_krhs_on[0, 1, 0] == np.uint64(4626263631292573473)
        assert t_geo_off[0, 1, 0] == np.uint64(4626263631292573474)
    finally:
        set_policy(prev_policy)


def test_stage_transport_stored_r3uv_pair_runs_the_written_ratio_path(
        monkeypatch):
    """The carried pair reaches native U/V ratio interpolation."""
    import legoesm.ocean.vertical as vertical

    calls = []

    def fake_geometry(eta, *_args, **_kwargs):
        calls.append(np.asarray(eta))
        return SimpleNamespace(
            r3u=jnp.asarray([[float(eta[0, 0])]]),
            r3v=jnp.asarray([[2.0 * float(eta[0, 0])]]))

    monkeypatch.setattr(
        vertical, "nemo_qco_live_face_geometry_from_operands", fake_geometry)
    ops = SimpleNamespace(
        e3u_0=None, e3v_0=None, umask3=None, vmask3=None,
        hu_0=None, hv_0=None, area_t=None, area_u=None, area_v=None)
    pair = (1, jnp.asarray([[0.25]]), jnp.asarray([[0.75]]))
    r3u, r3v = _nemo_ws_stage_r3uv(pair, ops, jnp.float64)
    assert len(calls) == 2
    expected_u = ((np.float64(2.0) / np.float64(3.0)) * np.float64(0.25)
                  + (np.float64(1.0) / np.float64(3.0)) * np.float64(0.75))
    expected_v = ((np.float64(2.0) / np.float64(3.0)) * np.float64(0.5)
                  + (np.float64(1.0) / np.float64(3.0)) * np.float64(1.5))
    np.testing.assert_array_equal(np.asarray(r3u), np.asarray([[expected_u]]))
    np.testing.assert_array_equal(np.asarray(r3v), np.asarray([[expected_v]]))


def test_end_of_stage_pivot_selector_cannot_run_on_a_nonfold_grid():
    """The selector reaches the stage program, not only the card tail."""
    card = recipe.build_nemo_testcase_card("VORTEX-zco")
    cfg = card.recipe.model_config._replace(rk3_end_of_stage_pivot_fold=True)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    with pytest.raises(ValueError, match="active T-pivot fold"):
        model.step(card.recipe.initial_state, dt=card.dt_s)

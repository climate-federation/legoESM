"""Record-free execution tests for the ORCA2 config-tail selectors.

These tests deliberately target model guards and small written-order helpers.
They need neither the ORCA2 deck nor a NEMO record, and each assertion stops
failing if the named selector is ignored at its consumer.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax, jax.numpy as jnp
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
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield recipe.build_nemo_testcase_card("GYRE-zco")
    finally:
        set_policy(prev_policy)


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
        m = LatLonCGridOceanModel(gyre_card.recipe.grid, gyre_card.recipe.z_coord, cfg)
        jax.make_jaxpr(lambda s: m.step(s, dt=gyre_card.dt_s))(
            gyre_card.recipe.initial_state)


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
        m = LatLonCGridOceanModel(gyre_card.recipe.grid, gyre_card.recipe.z_coord, cfg)
        jax.make_jaxpr(lambda s: m.step(s, dt=gyre_card.dt_s))(
            gyre_card.recipe.initial_state)


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
        on_s, off_s = np.asarray(on.S.data), np.asarray(off.S.data)
        on_u, off_u = np.asarray(on.u.data), np.asarray(off.u.data)
        assert np.count_nonzero(
            on_t.view(np.uint64) != off_t.view(np.uint64)) == 62
        assert np.count_nonzero(
            on_u.view(np.uint64) != off_u.view(np.uint64)) == 48
        assert on_t.view(np.uint64)[5, 7, 0] == np.uint64(4626600210828169908)
        assert off_t.view(np.uint64)[5, 7, 0] == np.uint64(4626600210828170265)
        assert on_u.view(np.uint64)[6, 0, 0] == np.uint64(4580213822658509297)
        assert off_u.view(np.uint64)[6, 0, 0] == np.uint64(4580213822683598223)
        assert on_u.view(np.uint64)[7, 0, 0] == np.uint64(13811414117467377035)
        assert off_u.view(np.uint64)[7, 0, 0] == np.uint64(4575974430849064564)
        assert on_t.view(np.uint64)[7, 7, 0] == np.uint64(4626479143473887009)
        assert off_t.view(np.uint64)[7, 7, 0] == np.uint64(4625960570868119510)
        assert on_s.view(np.uint64)[7, 7, 0] == np.uint64(4630126004937552226)
        assert off_s.view(np.uint64)[7, 7, 0] == np.uint64(4630094117933671990)
    finally:
        set_policy(prev_policy)


def test_rk3_end_of_stage_pivot_fold_stage3_site_changes_top_fold_row():
    """Pin the stage-3 end-of-stage pivot fold consumer at ocean_model_latlon_cgrid.py:11991."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, base = _ws_rk3_tripole_pivot_fixture()
        on = LatLonCGridOceanModel(
            grid, z_coord, base._replace(rk3_end_of_stage_pivot_fold=True),
        ).step(state, 600.0)
        off = LatLonCGridOceanModel(
            grid, z_coord, base._replace(rk3_end_of_stage_pivot_fold=False),
        ).step(state, 600.0)
        on_u, off_u = np.asarray(on.u.data).view(np.uint64), np.asarray(off.u.data).view(np.uint64)
        on_t, off_t = np.asarray(on.T.data).view(np.uint64), np.asarray(off.T.data).view(np.uint64)
        on_s, off_s = np.asarray(on.S.data).view(np.uint64), np.asarray(off.S.data).view(np.uint64)
        assert np.count_nonzero(on_u[7] != off_u[7]) == 27
        assert np.count_nonzero(on_t[7] != off_t[7]) == 27
        assert np.count_nonzero(on_s[7] != off_s[7]) == 27
        assert on_u[7, 0, 0] == np.uint64(13811414117467377035)
        assert on_t[7, 7, 0] == np.uint64(4626479143473887009)
        assert on_s[7, 7, 0] == np.uint64(4630126004937552226)
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


def test_rk3_vector_raw_vmask_shared_ratio_fold_or_changes_tripole_ws_rk3_step():
    """Exercise rk3_vector_raw_vmask through the shared ratio-fold OR at ocean_model_latlon_cgrid.py:2339."""
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


def test_rk3_vector_raw_vmask_changes_step_and_tendencies_with_fold_exchanges_off():
    """Exercise rk3_vector_raw_vmask with rk3_stage_transport_fold_exchanges=False across non-2339 consumers."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, base = _ws_rk3_tripole_pivot_fixture()

        # 1. Full step with fold_exchanges=False and raw vmask=1 on top fold row
        # exercises ocean_model_latlon_cgrid.py:6848 and :7699.
        grid_step = grid._replace(
            ff_f=grid.f_v[1:],
            dy_v=jnp.where(grid.dy_v == 0.0, grid.dy_v[-2:-1], grid.dy_v),
            dx_v=jnp.where(grid.dx_v == 0.0, grid.dx_v[-2:-1], grid.dx_v),
        )
        ny, nx, nz = state.T.data.shape
        e3 = jnp.broadcast_to(z_coord.dz_ref[None, None, :], (ny, nx, nz))
        e1u = grid_step.dy_u[:, 1:]
        e2u = grid_step.dx_u[:, 1:]
        e1v = grid_step.dy_v[1:, :]
        e2v = grid_step.dx_v[1:, :]
        e1f = grid_step.dy_v[1:, :]
        e2f = grid_step.dx_u[:, 1:]
        een_bt = SimpleNamespace(
            e1e2t=grid_step.area_T,
            r1_e1e2t=1.0 / grid_step.area_T,
            e1e2f=e1f * e2f,
            r1_e1e2f=1.0 / (e1f * e2f),
            e1u=e1u, r1_e1u=1.0 / e1u, e2u=e2u, r1_e2u=1.0 / e2u,
            e1v=e1v, r1_e1v=1.0 / e1v, e2v=e2v, r1_e2v=1.0 / e2v,
            e1f=e1f, e2f=e2f,
            umask=jnp.ones((ny, nx, nz), dtype=jnp.float64),
            vmask=jnp.ones((ny, nx, nz), dtype=jnp.float64),
            fmask=jnp.ones((ny, nx, nz), dtype=jnp.float64),
            ssmask=jnp.ones((ny, nx), dtype=jnp.float64),
            ssumask=jnp.ones((ny, nx), dtype=jnp.float64),
            ssvmask=jnp.ones((ny, nx), dtype=jnp.float64),
            ff_f=grid_step.f_v[1:],
            ff_t=grid_step.f_T,
            e3t_0=e3, e3u_0=e3, e3v_0=e3, e3f_0=e3,
        )
        zc_raw = z_coord._replace(
            nemo_e3t_0=e3,
            nemo_hu_0=jnp.sum(e3, axis=-1),
            nemo_hv_0=jnp.sum(e3, axis=-1),
            nemo_e1e2t=grid_step.area_T,
            nemo_e1e2u=e1u * e2u,
            nemo_e1e2v=e1v * e2v,
            nemo_e2u=e2u,
            nemo_e1v=e1v,
            nemo_een_barotropic=een_bt,
        )
        state_step = state._replace(
            v=state.v.replace(data=state.v.data.at[-1, :, :].set(0.05)),
            eta=state.eta.replace(
                data=jnp.linspace(
                    -0.05, 0.05, ny * nx, dtype=jnp.float64
                ).reshape(ny, nx)
            ),
        )
        cfg_step = base._replace(
            momentum_advection="vector_invariant",
            momentum_flux_scheme="centered",
            vertical_momentum_scheme="nemo_advective",
            zad_qco_evaluation="nemo_literal",
            wzv_call2_evaluation="nemo_literal",
            nemo_stage_momentum_wzv_split=True,
            nemo_first_wzv_after_ssh="rk3_extrapolated",
            vorticity_scheme="een_total",
            ke_gradient_scheme="c2",
            coriolis_scheme="explicit_ab2",
            een_fold_image_v_row=True,
            surface_stress_implicit=True,
            rk3_stage_transport_fold_exchanges=False,
        )
        step_on = LatLonCGridOceanModel(
            grid_step, zc_raw, cfg_step._replace(rk3_vector_raw_vmask=True),
        ).step(state_step, 600.0)
        step_off = LatLonCGridOceanModel(
            grid_step, zc_raw, cfg_step._replace(rk3_vector_raw_vmask=False),
        ).step(state_step, 600.0)
        step_u_on = np.asarray(step_on.u.data).view(np.uint64)
        step_u_off = np.asarray(step_off.u.data).view(np.uint64)
        step_v_on = np.asarray(step_on.v.data).view(np.uint64)
        step_v_off = np.asarray(step_off.v.data).view(np.uint64)
        assert np.count_nonzero(step_u_on != step_u_off) == 129
        assert np.count_nonzero(step_v_on != step_v_off) == 144
        assert step_v_on[-1, 0, 0] == np.uint64(4587354100208601891)
        assert step_v_off[-1, 0, 0] == np.uint64(0)
        assert step_u_on[-1, 0, 0] == np.uint64(4576421447979356078)
        assert step_u_off[-1, 0, 0] == np.uint64(4576410480201352036)

        # 2. ocean_model_latlon_cgrid.py:2440 (_nemo_ws_stage_transport velocity-form w_momentum).
        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean

        u_mask_3d = jnp.broadcast_to(
            state_step.u_mask.data[..., None], state_step.u.data.shape)
        v_mask_3d = jnp.broadcast_to(
            state_step.v_mask.data[..., None], state_step.v.data.shape)

        def run_w_mom(flag):
            cfg_w = base._replace(
                rk3_stage_transport_fold_exchanges=False,
                rk3_vector_raw_vmask=flag,
            )
            return np.asarray(omlc._nemo_ws_stage_transport(
                (state_step.u.data, state_step.v.data), e3, 1,
                eta_stage=state_step.eta.data, stored_h_stage=True, h_ref=e3,
                Hu_avg=jnp.full((ny, nx + 1), 300.0, dtype=jnp.float64),
                Hv_avg=jnp.full((ny + 1, nx), 300.0, dtype=jnp.float64),
                u_mask_3d=u_mask_3d, v_mask_3d=v_mask_3d,
                grid=grid_step, z_coord=zc_raw, H_bathy=state_step.H_bathy.data,
                config=cfg_w, dt=600.0,
                eta_before=state_step.eta.data,
                eta_after=state_step.eta.data * 1.2,
                literal_wzv=True, momentum_velocity_form_w=True,
            )[-1]).view(np.uint64)

        wm_on, wm_off = run_w_mom(True), run_w_mom(False)
        assert np.count_nonzero(wm_on != wm_off) == 36
        assert wm_on[7, 0, 0] == np.uint64(13767131978609372008)
        assert wm_off[7, 0, 0] == np.uint64(13767131618970328617)

        # 3. ocean_pe_latlon_cgrid.py:5595 (_nemo_qco_zad_operands fold_exchange_ratios).
        def run_zad(flag):
            cfg_z = base._replace(
                rk3_stage_transport_fold_exchanges=False,
                rk3_vector_raw_vmask=flag,
                zad_qco_evaluation="nemo_literal",
                vertical_momentum_scheme="nemo_advective",
                wzv_call2_evaluation="nemo_literal",
                nemo_first_wzv_after_ssh="rk3_extrapolated",
                momentum_advection="vector_invariant",
                momentum_flux_scheme="centered",
            )
            res = latlon_cgrid_ocean_baroclinic_tendencies(
                state_step, grid_step, zc_raw, cfg_z, dt=600.0, momentum_only=True,
            )
            return (
                np.asarray(res.du_dt.data).view(np.uint64),
                np.asarray(res.dv_dt.data).view(np.uint64),
            )

        zad_du_on, zad_dv_on = run_zad(True)
        zad_du_off, zad_dv_off = run_zad(False)
        assert np.count_nonzero(zad_du_on != zad_du_off) == 39
        assert np.count_nonzero(zad_dv_on != zad_dv_off) == 36
        assert zad_du_on[7, 0, 0] == np.uint64(4501496900116524386)
        assert zad_du_off[7, 0, 0] == np.uint64(4501496758677846748)

        # 4. ocean_pe_latlon_cgrid.py:5817 (flux_form + een_planetary), :5868 (vector_invariant + een_total),
        # :5973 (_bc_vertical_momentum_advection), :6418 (dv_dt mask), and :6557 (_mv diagnostic mask).
        grid_pe = create_synthetic_tripole(6, 8, dtype=jnp.float64)
        grid_pe = grid_pe._replace(ff_f=grid_pe.f_v[1:])
        z_pe = create_ocean_z_star(3, 300.0)
        st_pe = rest_state_latlon_cgrid_ocean(
            grid_pe, z_pe, T_water_init_C=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=300.0)
        rng_pe = np.random.default_rng(19)
        st_pe = st_pe._replace(
            u=st_pe.u.replace(data=jnp.asarray(
                0.1 * rng_pe.standard_normal(st_pe.u.data.shape))),
            v=st_pe.v.replace(data=jnp.asarray(
                0.1 * rng_pe.standard_normal(st_pe.v.data.shape))),
            T=st_pe.T.replace(data=st_pe.T.data + jnp.asarray(
                0.3 * rng_pe.standard_normal(st_pe.T.data.shape))),
            S=st_pe.S.replace(data=st_pe.S.data + jnp.asarray(
                0.05 * rng_pe.standard_normal(st_pe.S.data.shape))),
            eta=st_pe.eta.replace(data=jnp.asarray(
                0.02 * rng_pe.standard_normal(st_pe.eta.data.shape))),
        )
        cfg_5817 = base._replace(
            rk3_stage_transport_fold_exchanges=False,
            momentum_advection="flux_form",
            momentum_flux_scheme="none",
            vorticity_scheme="een_planetary",
            een_fold_image_v_row=True,
        )
        res_5817_on = latlon_cgrid_ocean_baroclinic_tendencies(
            st_pe, grid_pe, z_pe, cfg_5817._replace(rk3_vector_raw_vmask=True),
            dt=300.0, momentum_only=True,
        )
        res_5817_off = latlon_cgrid_ocean_baroclinic_tendencies(
            st_pe, grid_pe, z_pe, cfg_5817._replace(rk3_vector_raw_vmask=False),
            dt=300.0, momentum_only=True,
        )
        dv_5817_on = np.asarray(res_5817_on.dv_dt.data).view(np.uint64)
        dv_5817_off = np.asarray(res_5817_off.dv_dt.data).view(np.uint64)
        assert np.count_nonzero(dv_5817_on != dv_5817_off) == 4
        assert dv_5817_on[6, 3, 0] == np.uint64(0)
        assert dv_5817_off[6, 3, 0] == np.uint64(9223372036854775808)

        cfg_5868 = base._replace(
            rk3_stage_transport_fold_exchanges=False,
            momentum_advection="vector_invariant",
            momentum_flux_scheme="centered",
            vorticity_scheme="een_total",
            ke_gradient_scheme="c2",
            coriolis_scheme="explicit_ab2",
            een_fold_image_v_row=True,
        )
        res_5868_on = latlon_cgrid_ocean_baroclinic_tendencies(
            st_pe, grid_pe, z_pe, cfg_5868._replace(rk3_vector_raw_vmask=True),
            dt=300.0, momentum_only=True,
        )
        res_5868_off = latlon_cgrid_ocean_baroclinic_tendencies(
            st_pe, grid_pe, z_pe, cfg_5868._replace(rk3_vector_raw_vmask=False),
            dt=300.0, momentum_only=True,
        )
        dv_5868_on = np.asarray(res_5868_on.dv_dt.data).view(np.uint64)
        dv_5868_off = np.asarray(res_5868_off.dv_dt.data).view(np.uint64)
        assert np.count_nonzero(dv_5868_on != dv_5868_off) == 4
        assert dv_5868_on[6, 3, 0] == np.uint64(0)
        assert dv_5868_off[6, 3, 0] == np.uint64(9223372036854775808)

        cfg_pe_raw = cfg_5868._replace(
            vertical_momentum_scheme="upwind_perturbation",
            een_fold_image_v_row=False,
        )
        raw_vmask = jnp.ones_like(st_pe.v.data)
        res_raw_on, diag_raw_on = latlon_cgrid_ocean_baroclinic_tendencies(
            st_pe, grid_pe, z_pe, cfg_pe_raw._replace(rk3_vector_raw_vmask=True),
            dt=300.0, momentum_only=True, diagnose_momentum=True,
            nemo_literal_hpg_vmask=raw_vmask,
        )
        res_raw_off, diag_raw_off = latlon_cgrid_ocean_baroclinic_tendencies(
            st_pe, grid_pe, z_pe, cfg_pe_raw._replace(rk3_vector_raw_vmask=False),
            dt=300.0, momentum_only=True, diagnose_momentum=True,
            nemo_literal_hpg_vmask=raw_vmask,
        )
        assert np.count_nonzero(
            np.asarray(res_raw_on.dv_dt.data).view(np.uint64)
            != np.asarray(res_raw_off.dv_dt.data).view(np.uint64)
        ) == 48
        assert np.count_nonzero(
            np.asarray(diag_raw_on.vertadv_v.data).view(np.uint64)
            != np.asarray(diag_raw_off.vertadv_v.data).view(np.uint64)
        ) == 24
        assert np.count_nonzero(
            np.asarray(diag_raw_on.vortcor_v.data).view(np.uint64)
            != np.asarray(diag_raw_off.vortcor_v.data).view(np.uint64)
        ) == 47
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


def test_l4b_and_l6_geothermal_and_damping_validator_coupling_and_valid_pair_step():
    """Pin validator coupling when geothermal_after_ldf or damping_after_geothermal is toggled alone, and measure valid pairs."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.physics.vertical_mixing import VerticalMixingConfig
    from legoesm.ocean.sponge import SpongeForcing

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, forcing, base_cfg = _stage3_tracer_step_fixture()
        raw3 = jnp.broadcast_to(z_coord.dz_ref[None, None, :], state.T.data.shape)
        raw2 = jnp.ones(state.T.data.shape[:2], dtype=jnp.float64) * 300.0
        z_coord = z_coord._replace(
            nemo_e3t_0=raw3,
            nemo_e3w_0=raw3,
            nemo_hu_0=raw2,
            nemo_hv_0=raw2,
            nemo_e1e2t=grid.area_T,
            nemo_e1e2u=(grid.dx_u * grid.dy_u)[:, 1:],
            nemo_e1e2v=(grid.dx_v * grid.dy_v)[1:, :],
            nemo_e2u=grid.dy_u[:, 1:],
            nemo_e1v=grid.dx_v[1:, :],
        )
        gm = recipe.orca2_l2_isoneutral_gm_redi()._replace(
            nemo_slope_north_fold=False,
            slope_n2_evaluation="recompute",
            slope_prd_evaluation="density_roundtrip",
            slope_depth_evaluation="legacy_jacobian_t_surface",
            redi_w_slope_stage_evaluation="redi_tuple",
            redi_flux_reference_thickness_evaluation="tpoint_alias",
            redi_flux_stored_face_ratios=False,
            redi_flux_geometry_stage="step_entry",
        )
        l2_cfg = base_cfg._replace(
            implicit_vertical_mixing=False,
            gm_redi=gm,
            physics=base_cfg.physics._replace(
                vertical_mixing=VerticalMixingConfig(scheme="constant"),
            ),
        )

        # Validator coupling: toggling either L4b or L6 selector alone raises ValueError.
        with pytest.raises(
            ValueError,
            match="rk3_stage3_geothermal_after_ldf_nemo_order requires rk3_stage3_geothermal_nemo_written",
        ):
            LatLonCGridOceanModel._validate_config(
                l2_cfg._replace(rk3_stage3_geothermal_after_ldf_nemo_order=True)
            )
        with pytest.raises(
            ValueError,
            match="rk3_stage3_geothermal_nemo_written is not supported with rk3_stage3_tracer_krhs_nemo_order: the written stage-3 Krhs term list does not yet carry tra_bbc",
        ):
            LatLonCGridOceanModel._validate_config(
                l2_cfg._replace(
                    rk3_stage3_geothermal_nemo_written=True,
                    rk3_stage3_geothermal_after_ldf_nemo_order=False,
                )
            )
        with pytest.raises(
            ValueError,
            match="rk3_stage3_geothermal_nemo_written is not supported with tracer lateral diffusion: NEMO applies tra_ldf before tra_bbc; select the L4b ordered composition",
        ):
            LatLonCGridOceanModel._validate_config(
                l2_cfg._replace(
                    rk3_stage3_tracer_krhs_nemo_order=False,
                    rk3_stage3_geothermal_nemo_written=True,
                    rk3_stage3_geothermal_after_ldf_nemo_order=False,
                )
            )
        with pytest.raises(
            ValueError,
            match="rk3_stage3_damping_after_geothermal_nemo_order requires the certified L4b ordered composition",
        ):
            LatLonCGridOceanModel._validate_config(
                l2_cfg._replace(rk3_stage3_damping_after_geothermal_nemo_order=True)
            )

        sponge = SpongeForcing(
            gamma=jnp.full(state.T.data.shape, 1e-4, dtype=jnp.float64),
            T_ref=state.T.data + 1.5,
            S_ref=state.S.data - 0.3,
            nemo_kt=1,
        )
        cfg_l4b_on = l2_cfg._replace(
            rk3_stage3_tracer_krhs_nemo_order=False,
            rk3_stage3_geothermal_nemo_written=True,
            rk3_stage3_geothermal_after_ldf_nemo_order=True,
        )
        cfg_l4b_off = l2_cfg._replace(
            rk3_stage3_tracer_krhs_nemo_order=False,
            rk3_stage3_geothermal_nemo_written=False,
            rk3_stage3_geothermal_after_ldf_nemo_order=False,
        )
        out_l4b_on = LatLonCGridOceanModel(grid, z_coord, cfg_l4b_on).step(
            state, 300.0, surface_forcing=forcing, sponge=sponge, nemo_kt=1,
        )
        out_l4b_off = LatLonCGridOceanModel(grid, z_coord, cfg_l4b_off).step(
            state, 300.0, surface_forcing=forcing, sponge=sponge, nemo_kt=1,
        )
        t_l4b_on = np.asarray(out_l4b_on.T.data).view(np.uint64)
        t_l4b_off = np.asarray(out_l4b_off.T.data).view(np.uint64)
        assert np.count_nonzero(t_l4b_on != t_l4b_off) == 6
        assert t_l4b_on[1, 0, 2] == np.uint64(4625465393761942146)
        assert t_l4b_off[1, 0, 2] == np.uint64(4625465393755890474)

        cfg_l6_on = cfg_l4b_on._replace(
            rk3_stage3_damping_after_geothermal_nemo_order=True,
        )
        out_l6_on = LatLonCGridOceanModel(grid, z_coord, cfg_l6_on).step(
            state, 300.0, surface_forcing=forcing, sponge=sponge, nemo_kt=1,
        )
        t_l6_on = np.asarray(out_l6_on.T.data).view(np.uint64)
        s_l6_on = np.asarray(out_l6_on.S.data).view(np.uint64)
        s_l4b_on = np.asarray(out_l4b_on.S.data).view(np.uint64)
        assert np.count_nonzero(t_l6_on != t_l4b_on) == 18
        assert np.count_nonzero(s_l6_on != s_l4b_on) == 18
        assert t_l6_on[1, 0, 0] == np.uint64(4626479647477587073)
        assert t_l4b_on[1, 0, 0] == np.uint64(4626466981215184375)
        assert s_l6_on[1, 0, 0] == np.uint64(4630131660021248911)
        assert s_l4b_on[1, 0, 0] == np.uint64(4630132926647489180)
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


def test_stage_wzv_clock_pair_changes_the_executed_wzv_clock_all_sites(
        gyre_card, monkeypatch):
    """Pin both tracer (w_stage) and momentum (w_momentum) continuity call sites."""
    state = gyre_card.recipe.initial_state
    grid, z_coord = gyre_card.recipe.grid, gyre_card.recipe.z_coord
    h_ref = jnp.asarray(z_coord.nemo_e3t_0, dtype=jnp.float64)
    u_mask = jnp.broadcast_to(
        state.u_mask.data[..., None], state.u.data.shape)
    v_mask = jnp.broadcast_to(
        state.v_mask.data[..., None], state.v.data.shape)
    seen_kw = []

    def fake_wzv(*args, **kwargs):
        clock = float(args[9])
        after_level = int(kwargs.get("stage_after_level", 0))
        seen_kw.append((
            clock,
            after_level,
            bool(kwargs.get("stored_r3_ratios", False)),
            int(kwargs.get("stored_kmm_level", 0)),
            bool(kwargs.get("half_step_after_ratio", False)),
        ))
        return jnp.full_like(h_ref, clock + after_level), None, None

    monkeypatch.setattr(omlc, "nemo_qco_wzv_operands", fake_wzv)

    def call_both(
        *,
        clock_pair=False,
        r3=False,
        tracer_kmm=False,
        mom_half=False,
        mom_kmm=False,
    ):
        seen_kw.clear()
        cfg = gyre_card.recipe.model_config._replace(
            rk3_stage_wzv_nemo_clock_pair=clock_pair,
            nemo_wzv_stored_r3_ratios=r3,
            rk3_stage_tracer_wzv_stored_kmm_ratios=tracer_kmm,
            nemo_stage2_momentum_wzv_half_step=mom_half,
            nemo_stage_momentum_wzv_stored_kmm_ratios=mom_kmm,
        )
        omlc._nemo_ws_stage_transport(
            (state.u.data, state.v.data), h_ref, 1,
            eta_stage=state.eta.data, stored_h_stage=True,
            h_ref=h_ref, Hu_avg=jnp.zeros_like(state.u.data[..., 0]),
            Hv_avg=jnp.zeros_like(state.v.data[..., 0]),
            u_mask_3d=u_mask, v_mask_3d=v_mask, grid=grid,
            z_coord=z_coord, H_bathy=state.H_bathy.data,
            config=cfg, dt=10800.0, eta_before=state.eta.data,
            eta_after=state.eta.data, literal_wzv=True,
            momentum_velocity_form_w=True)
        return list(seen_kw)

    assert call_both(clock_pair=True) == [
        (5400.0, 2, False, 0, False),
        (5400.0, 2, False, 0, False),
    ]
    assert call_both(r3=True) == [
        (10800.0, 0, True, 0, False),
        (10800.0, 0, True, 0, False),
    ]
    assert call_both(tracer_kmm=True) == [
        (10800.0, 0, True, 1, False),
        (10800.0, 0, False, 0, False),
    ]
    assert call_both(mom_half=True, mom_kmm=True) == [
        (10800.0, 0, False, 0, False),
        (10800.0, 0, False, 1, True),
    ]


def test_rk3_stage3_salt_geothermal_and_krhs_order_change_model_step_qsr_sites():
    """Exercise nemo_qsr_2bd consumer sites (:9163 and :9171) on the small step fixture."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, forcing, base_cfg = _stage3_tracer_step_fixture()
        sw_2bd = base_cfg.physics.shortwave_penetration._replace(
            scheme="nemo_qsr_2bd",
        )
        cfg_2bd = base_cfg._replace(
            rk3_stage3_heat_nemo_written=False, rk3_stage3_salt_nemo_written=False, rk3_stage3_geothermal_nemo_written=False, rk3_stage3_tracer_krhs_nemo_order=False,
            rk3_stage3_tracer_nemo_stored_thickness=False, physics=base_cfg.physics._replace(shortwave_penetration=sw_2bd))
        with pytest.raises(
            _SelectorTouched, match="nemo_qsr_stage3_kmm_source_association",
        ):
            m_qsr = LatLonCGridOceanModel(
                grid,
                z_coord,
                cfg_2bd._replace(
                    nemo_qsr_stage3_kmm_source_association=_BoolProbe(
                        "nemo_qsr_stage3_kmm_source_association"
                    )
                ),
            )
            jax.make_jaxpr(
                lambda s: m_qsr.step(s, 300.0, surface_forcing=forcing)
            )(state)
        with pytest.raises(
            _SelectorTouched, match="nemo_qsr_kbb_live_stretch",
        ):
            sw_probe = sw_2bd._replace(
                nemo_qsr_kbb_live_stretch=type("_P", (), {"c": 0, "__bool__": lambda s: (setattr(s, "c", s.c + 1), (_ for _ in ()).throw(_SelectorTouched("nemo_qsr_kbb_live_stretch")) if s.c >= 2 else True)[1]})()
            )
            m_kbb = LatLonCGridOceanModel(
                grid,
                z_coord,
                cfg_2bd._replace(
                    physics=cfg_2bd.physics._replace(
                        shortwave_penetration=sw_probe
                    )
                ),
            )
            jax.make_jaxpr(
                lambda s: m_kbb.step(s, 300.0, surface_forcing=forcing)
            )(state)
    finally:
        set_policy(prev_policy)


def test_l4b_and_l6_geothermal_and_damping_validator_coupling_and_valid_pair_step_extra_sites():
    """Exercise PE damping suppression (:6382), EIV Krhs guards (:4202, :10827), and zdf a33 (:14425)."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        latlon_cgrid_ocean_baroclinic_tendencies,
    )
    from legoesm.ocean.physics.vertical_mixing import VerticalMixingConfig
    from legoesm.ocean.sponge import SpongeForcing

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, forcing, base_cfg = _stage3_tracer_step_fixture()
        raw3 = jnp.broadcast_to(z_coord.dz_ref[None, None, :], state.T.data.shape)
        raw2 = jnp.ones(state.T.data.shape[:2], dtype=jnp.float64) * 300.0
        z_coord = z_coord._replace(
            nemo_e3t_0=raw3,
            nemo_e3w_0=raw3,
            nemo_hu_0=raw2,
            nemo_hv_0=raw2,
            nemo_e1e2t=grid.area_T,
            nemo_e1e2u=(grid.dx_u * grid.dy_u)[:, 1:],
            nemo_e1e2v=(grid.dx_v * grid.dy_v)[1:, :],
            nemo_e2u=grid.dy_u[:, 1:],
            nemo_e1v=grid.dx_v[1:, :],
        )
        gm = recipe.orca2_l2_isoneutral_gm_redi()._replace(
            nemo_slope_north_fold=False,
            slope_n2_evaluation="recompute",
            slope_prd_evaluation="density_roundtrip",
            slope_depth_evaluation="legacy_jacobian_t_surface",
            redi_w_slope_stage_evaluation="redi_tuple",
            redi_flux_reference_thickness_evaluation="tpoint_alias",
            redi_flux_stored_face_ratios=False,
            redi_flux_geometry_stage="step_entry",
        )
        l2_cfg = base_cfg._replace(
            implicit_vertical_mixing=False,
            gm_redi=gm,
            physics=base_cfg.physics._replace(
                vertical_mixing=VerticalMixingConfig(scheme="constant"),
            ),
        )
        sponge = SpongeForcing(
            gamma=jnp.full(state.T.data.shape, 1e-4, dtype=jnp.float64),
            T_ref=state.T.data + 1.5,
            S_ref=state.S.data - 0.3,
            nemo_kt=1,
        )
        cfg_l4b_on = l2_cfg._replace(
            rk3_stage3_tracer_krhs_nemo_order=False,
            rk3_stage3_geothermal_nemo_written=True,
            rk3_stage3_geothermal_after_ldf_nemo_order=True,
        )
        cfg_l6_on = cfg_l4b_on._replace(
            rk3_stage3_damping_after_geothermal_nemo_order=True,
        )
        pe_l6 = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg_l6_on, sponge=sponge,
        )
        pe_l4b = latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord, cfg_l4b_on, sponge=sponge,
        )
        assert not np.array_equal(
            np.asarray(pe_l6.dT_dt.data), np.asarray(pe_l4b.dT_dt.data)
        )

        eiv_cfg = l2_cfg._replace(
            rk3_stage3_tracer_krhs_nemo_order=False,
            gm_redi=gm._replace(
                gm_bolus_advection="through_fct",
                kappa_GM=1000.0,
            ),
        )
        with pytest.raises(
            ValueError,
            match="NEMO rk3_ws eddy-induced velocity \\(GM bolus through FCT\\) requires rk3_stage3_tracer_krhs_nemo_order",
        ):
            LatLonCGridOceanModel._check_l3_krhs_route(eiv_cfg)
        model_eiv = LatLonCGridOceanModel(grid, z_coord, l2_cfg)
        model_eiv.prime_step_caches(state)
        with pytest.raises(
            ValueError,
            match="NEMO rk3_ws eddy-induced velocity requires rk3_stage3_tracer_krhs_nemo_order",
        ):
            jax.make_jaxpr(
                lambda s: model_eiv._step_impl(
                    s, 300.0, surface_forcing=forcing, config=eiv_cfg
                )
            )(state)

        m_zdf = LatLonCGridOceanModel(
            grid,
            z_coord,
            l2_cfg._replace(
                implicit_vertical_mixing=True,
                gm_redi=gm._replace(
                    redi_a33_source_ordering=_BoolProbe("redi_a33_source_ordering")
                ),
            ),
        )
        with pytest.raises(_SelectorTouched, match="redi_a33_source_ordering"):
            m_zdf._apply_implicit_vertical_mixing(
                state,
                dt=300.0,
                surface_forcing=forcing,
                K33_iso=jnp.ones((4, 6, 2), dtype=jnp.float64),
            )
    finally:
        set_policy(prev_policy)


def test_stage_transport_stored_r3uv_pair_reaches_both_stage2_and_stage3_transport(
        monkeypatch):
    """Exercise both stage-2 (:9724) and stage-3 (:9795) stored-r3uv transport handoffs."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid, z_coord, state, forcing, base_cfg = _stage3_tracer_step_fixture()
        real_transport = omlc._nemo_ws_stage_transport
        seen_pairs = []

        def spy_transport(*args, **kwargs):
            seen_pairs.append((int(args[2]), kwargs.get("nemo_stage_pair") is not None))
            return real_transport(*args, **kwargs)

        monkeypatch.setattr(omlc, "_nemo_ws_stage_transport", spy_transport)
        cfg = base_cfg._replace(
            rk3_stage_transport_stored_r3uv=True,
            rk3_stage_r3uv_nemo_literal=True,
        )
        m = LatLonCGridOceanModel(grid, z_coord, cfg)
        jax.make_jaxpr(lambda s: m.step(s, 300.0, surface_forcing=forcing))(state)
        assert seen_pairs == [(0, False), (1, True), (2, True)]
    finally:
        set_policy(prev_policy)

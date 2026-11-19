"""Record-free statement tests recovered from the ORCA2 lane stays list."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.state import LatLonCGridOceanConfig


class _Fold:
    is_active = True
    vector_sign_v = -1.0

    def __init__(self, size):
        self.perm_v = (size - np.arange(size)) % size


class _Grid:
    def __init__(self, fold):
        self.fold = fold


def test_een_fold_image_row_and_disabled_plant():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        grid = create_synthetic_tripole(6, 8, dtype=jnp.float64)
        grid = grid._replace(ff_f=grid.f_v[1:])
        zc = create_ocean_z_star(3, 300.0)
        st = rest_state_latlon_cgrid_ocean(
            grid, zc, T_water_init_C=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=300.0)
        rng = np.random.default_rng(19)
        st = st._replace(
            u=st.u.replace(data=jnp.asarray(
                0.1 * rng.standard_normal(st.u.data.shape))),
            v=st.v.replace(data=jnp.asarray(
                0.1 * rng.standard_normal(st.v.data.shape))),
            T=st.T.replace(data=st.T.data + jnp.asarray(
                0.3 * rng.standard_normal(st.T.data.shape))),
            S=st.S.replace(data=st.S.data + jnp.asarray(
                0.05 * rng.standard_normal(st.S.data.shape))),
            eta=st.eta.replace(data=jnp.asarray(
                0.02 * rng.standard_normal(st.eta.data.shape))),
        )
        for adv, flux_scheme, vort, want_on, want_off in [
            ("flux_form", "none", "een_planetary",
             np.uint64(4506226019843816725), np.uint64(13749299632200641326)),
            ("vector_invariant", "centered", "een_total",
             np.uint64(4506002909676158083), np.uint64(13749304629572458198)),
        ]:
            cfg_on = LatLonCGridOceanConfig.from_flat(
                momentum_advection=adv,
                momentum_flux_scheme=flux_scheme,
                vorticity_scheme=vort,
                een_fold_image_v_row=True,
            )
            cfg_off = cfg_on._replace(een_fold_image_v_row=False)
            res_on = pe.latlon_cgrid_ocean_baroclinic_tendencies(
                st, grid, zc, cfg_on, momentum_only=True)
            res_off = pe.latlon_cgrid_ocean_baroclinic_tendencies(
                st, grid, zc, cfg_off, momentum_only=True)
            du_on = np.asarray(res_on.du_dt.data).view(np.uint64)
            du_off = np.asarray(res_off.du_dt.data).view(np.uint64)
            assert np.count_nonzero(du_on != du_off) == 27
            assert du_on[5, 0, 0] == want_on
            assert du_off[5, 0, 0] == want_off

        with pytest.raises(ValueError, match="active north fold"):
            pe.latlon_cgrid_ocean_baroclinic_tendencies(
                st, grid._replace(fold=None), zc,
                LatLonCGridOceanConfig.from_flat(
                    momentum_advection="vector_invariant",
                    vorticity_scheme="een_total",
                    een_fold_image_v_row=True,
                ),
                momentum_only=True,
            )
    finally:
        set_policy(prev_policy)


def test_fold_top_v_operands_and_disabled_plant():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_fold_top_v_wet,
        _nemo_fold_top_v_operands,
    )

    mask = jnp.arange(12.0).reshape(4, 3)
    base = LatLonCGridOceanConfig()
    off_cfg = base._replace(barotropic=base.barotropic._replace(
        barotropic_fold_top_v_face_ssh=False))
    on_cfg = base._replace(barotropic=base.barotropic._replace(
        barotropic_fold_top_v_face_ssh=True))
    raw = type("Raw", (), {
        "vmask": jnp.asarray([
            [[0.0], [0.0], [0.0]],
            [[1.0], [1.0], [1.0]],
            [[1.0], [0.0], [1.0]],
        ])
    })()
    z_coord = type("Coordinate", (), {"nemo_een_barotropic": raw})()
    off_wet = _nemo_fold_top_v_wet(off_cfg, z_coord, mask.dtype)
    on_wet = _nemo_fold_top_v_wet(on_cfg, z_coord, mask.dtype)
    same, flag = _nemo_fold_top_v_operands(mask, False, off_wet)
    assert same is mask and flag is False
    moved, flag = _nemo_fold_top_v_operands(
        mask, False, on_wet)
    assert flag is True
    np.testing.assert_array_equal(np.asarray(moved)[-1], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(moved)[:-1], np.asarray(mask)[:-1])


def test_literal_flux_update_matches_written_order_and_legacy_plant_moves():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_flux_form_external_velocity_update,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig

    base = LatLonCGridOceanConfig()
    literal = base._replace(barotropic=base.barotropic._replace(
        barotropic_flux_update_nemo_literal=True,
        barotropic_entry_depth_nemo_literal=True))
    args = tuple(jnp.asarray(value, dtype=jnp.float64) for value in (
        0.125, 7.0, 7.25, 7.5, 8.0, 8.5, 0.03125, -0.0625, 0.015625,
        830.7692307692307, 1.0, 1.0e-6))
    actual = _nemo_flux_form_external_velocity_update(
        *args, config=literal, substep_index=0)
    legacy = _nemo_flux_form_external_velocity_update(*args)
    r = np.float64
    forced = r(r(r(args[2] * args[6]) + r(args[3] * args[7]))
               + r(args[4] * args[8]))
    expected = r(r(r(args[4] * args[0]) + r(args[9] * forced))
                 * r(args[10] / r(r(args[5] + 1.0) - args[10])))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not np.array_equal(np.asarray(actual), np.asarray(legacy))


def test_ascending_level_sum_is_written_order_and_control_can_fail():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.dynamics.barotropic_common import ascending_level_sum
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import (
        create_ocean_z_star,
        create_partial_cell_coordinate,
    )

    prev_policy = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
        # Thirty-three levels force XLA's generic reduction tree to differ from
        # NEMO's explicitly ascending scalar accumulation for this fixed input.
        rng = np.random.default_rng(1)
        values = rng.standard_normal((20, 33))
        values *= np.exp(rng.uniform(-50.0, 50.0, values.shape))
        expected = np.zeros(values.shape[:-1], dtype=np.float64)
        for level in range(values.shape[-1]):
            expected = expected + values[..., level]
        ordered = np.asarray(jax.jit(ascending_level_sum)(jnp.asarray(values)))
        np.testing.assert_array_equal(ordered, expected)

        grid_bt = create_synthetic_tripole(4, 6, dtype=jnp.float64)
        base_z_bt = create_ocean_z_star(33, 3300.0)
        st_bt = rest_state_latlon_cgrid_ocean(
            grid_bt, base_z_bt, T_water_init_C=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=3300.0)
        zc_bt = create_partial_cell_coordinate(base_z_bt, st_bt.H_bathy.data)
        rng_bt = np.random.default_rng(1)
        v_vals = rng_bt.standard_normal(st_bt.v.data.shape) * np.exp(
            rng_bt.uniform(-50.0, 0.0, st_bt.v.data.shape))
        st_bt = st_bt._replace(
            u=st_bt.u.replace(data=jnp.zeros_like(st_bt.u.data)),
            v=st_bt.v.replace(data=jnp.asarray(v_vals)),
        )

        def run_bt(asc):
            cfg = LatLonCGridOceanConfig.from_flat(
                barotropic_solver="explicit_substep",
                outer_integrator="ab2",
                momentum_time_integrator="euler",
                tracer_time_integrator="euler",
                tracer_advection="none",
                momentum_advection="flux_form",
                momentum_flux_scheme="none",
                vertical_momentum_scheme="none",
                coriolis_scheme="explicit_ab2",
                barotropic_coriolis_split="frozen",
                barotropic_slow_forcing_depth_evaluation="nemo_literal",
                barotropic_rhs_ascending_level_sum=asc,
                surface_stress_implicit=True,
                implicit_vertical_mixing=False,
                n_barotropic_substeps=1,
                enable_runtime_checks=False,
            )
            return LatLonCGridOceanModel(grid_bt, zc_bt, cfg).step(
                st_bt, 1.0e-6)

        on_u = np.asarray(run_bt(True).u.data).view(np.uint64)
        off_u = np.asarray(run_bt(False).u.data).view(np.uint64)
        assert np.count_nonzero(on_u != off_u) == 127
        assert on_u[0, 2, 0] == np.uint64(4421695770371195880)
        assert off_u[0, 2, 0] == np.uint64(4421695770371195878)
    finally:
        set_policy(prev_policy)


def test_npcrv_area_selector_changes_live_relative_vorticity_by_area_ratio(
        monkeypatch):
    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_vor_ene_vorticity_cgrid,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = ensure_geometry(create_latlon_grid(8, 12))
    z_coord = create_ocean_z_star(n_levels=2, H_max=1000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=10.0, T_deep=4.0,
        S_uniform=35.0, H_max=1000.0)
    jj_u = jnp.arange(grid.n_lat, dtype=jnp.float64)[:, None, None]
    jj_v = jnp.arange(grid.n_lat + 1, dtype=jnp.float64)[:, None, None]
    ii_u = jnp.arange(grid.n_lon + 1, dtype=jnp.float64)[None, :, None]
    ii_v = jnp.arange(grid.n_lon, dtype=jnp.float64)[None, :, None]
    u = jnp.broadcast_to(
        jnp.sin((jj_u + 1.0) * (ii_u + 1.0) / 17.0) * 0.03,
        state.u.data.shape)
    v = jnp.broadcast_to(
        jnp.cos((jj_v + 1.0) * (ii_v + 2.0) / 19.0) * 0.02,
        state.v.data.shape).at[0].set(0.0).at[-1].set(0.0)
    state = state._replace(
        u=state.u.replace(data=u), v=state.v.replace(data=v))

    generic_area = jnp.asarray(grid.area_q)
    monkeypatch.setattr(pe, "_nemo_f_vertex_area", lambda *_: 2.0 * generic_area)
    monkeypatch.setattr(
        pe, "curl_vertex_cgrid",
        lambda u_arg, v_arg, grid_arg: nemo_vor_ene_vorticity_cgrid(
            u_arg, v_arg, grid_arg))
    captured = []

    def capture_zeta(zeta, _h_vtx, _h_v, v_arg, _h_u, u_arg,
                     *_args, **_kwargs):
        captured.append(np.asarray(zeta))
        return jnp.zeros_like(u_arg), jnp.zeros_like(v_arg)

    monkeypatch.setattr(pe, "pv_flux_al81_partial_cell", capture_zeta)
    base = LatLonCGridOceanConfig.from_flat(
        A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0, bottom_drag_r=0.0,
        vorticity_scheme="een_total", een_metric_weighting="nemo",
        een_nemo_relative_vorticity=False)
    for enabled in (False, True):
        pe.latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord,
            base._replace(een_nemo_relative_vorticity=enabled),
            momentum_only=True)
    assert len(captured) == 2
    off, on = captured
    assert np.count_nonzero(off) > 0
    np.testing.assert_array_equal(on, off * np.float64(0.5))


def test_npcrv_area_refuses_non_nemo_metric_weighting():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux

    with pytest.raises(ValueError, match="requires een_metric_weighting='nemo'"):
        _bc_pv_flux(*([None] * 12), een_metric_weighting="off",
                    een_nemo_zeta=True)

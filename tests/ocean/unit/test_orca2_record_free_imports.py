"""Record-free statement tests recovered from the ORCA2 lane stays list."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class _Fold:
    is_active = True
    vector_sign_v = -1.0

    def __init__(self, size):
        self.perm_v = (size - np.arange(size)) % size


class _Grid:
    def __init__(self, fold):
        self.fold = fold


def test_een_fold_image_row_and_disabled_plant():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _een_fold_image_v_row,
    )

    size = 6
    values = jnp.arange(30.0).reshape(5, size, 1)
    thickness = values + 100.0
    mask = jnp.ones_like(values).at[-1].set(0.0).at[-2, 2].set(0.0)
    sentinel = object()
    off = _een_fold_image_v_row(
        False, _Grid(_Fold(size)), thickness, values,
        sentinel, sentinel, sentinel, mask)
    assert off[0] is thickness and off[1] is values and off[5] is mask

    moved = _een_fold_image_v_row(
        True, _Grid(_Fold(size)), thickness, values,
        sentinel, sentinel, sentinel, mask)
    perm = (size - np.arange(size)) % size
    wet = np.asarray(mask)[-2][perm]
    np.testing.assert_array_equal(np.asarray(moved[5])[-1], wet)
    np.testing.assert_array_equal(
        np.asarray(moved[1])[-1], -np.asarray(values)[-2][perm] * wet)
    np.testing.assert_array_equal(
        np.asarray(moved[0])[-1], np.asarray(thickness)[-2][perm])
    with pytest.raises(ValueError, match="active north fold"):
        _een_fold_image_v_row(
            True, _Grid(None), thickness, values,
            sentinel, sentinel, sentinel, mask)


def test_fold_top_v_operands_and_disabled_plant():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_fold_top_v_operands,
    )

    mask = jnp.arange(12.0).reshape(4, 3)
    same, flag = _nemo_fold_top_v_operands(mask, False, None)
    assert same is mask and flag is False
    moved, flag = _nemo_fold_top_v_operands(
        mask, False, jnp.asarray([1.0, 0.0, 1.0]))
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
    from legoesm.ocean.dynamics.barotropic_common import ascending_level_sum

    rng = np.random.default_rng(276)
    values = rng.standard_normal((400, 30))
    values *= np.exp(rng.uniform(-20.0, 0.0, values.shape))
    expected = np.zeros(values.shape[:-1], dtype=np.float64)
    for level in range(values.shape[-1]):
        expected = expected + values[..., level]
    ordered = np.asarray(jax.jit(ascending_level_sum)(jnp.asarray(values)))
    np.testing.assert_array_equal(ordered, expected)
    lanes = [values[:, level::4].sum(axis=1) for level in range(4)]
    four_lane = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    assert np.count_nonzero(four_lane != expected) > 0


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

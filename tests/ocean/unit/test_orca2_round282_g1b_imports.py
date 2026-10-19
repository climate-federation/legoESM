"""Red-capable controls for the two G1b statements measured in round 282."""

from pathlib import Path

import numpy as np
import pytest


OMT4 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round222/acquisition/orca2_omt4_frames_10step_a_np2")


def _card():
    if not (OMT4 / "namelist_cfg").is_file():
        pytest.skip("admitted OMT-4 deck is absent")
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
    )
    card = prior._selected_card("omt4", OMT4)
    # Round 318 ships the fold unit; round 282's arms are relative to it off.
    cfg = card.recipe.model_config
    cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_literal_fold_unit_vector_form=False))
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def test_g1b_orca2_selectors_default_off():
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig

    assert LatLonCGridOceanConfig._field_defaults[
        "een_nemo_relative_vorticity"] is False
    assert LatLonCGridOceanConfig._field_defaults[
        "een_nemo_top_f_u_halo"] is False
    assert BarotropicConfig._field_defaults[
        "barotropic_literal_fold_unit_vector_form"] is False


def test_round282_arms_change_exactly_the_registered_fields():
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round282_g1b_orca2_gate as gate,
    )

    base = _card()
    assert base.recipe.model_config.een_nemo_relative_vorticity is True
    assert (base.recipe.model_config.barotropic
            .barotropic_literal_fold_unit_vector_form) is False
    for arm, expected in gate.ARMS.items():
        selected = gate.arm_card(base, arm)
        cfg = selected.recipe.model_config
        actual = {
            name for name, value in (
                (gate.A, cfg.een_nemo_relative_vorticity),
                (gate.C, cfg.een_nemo_top_f_u_halo),
                (gate.B,
                 cfg.barotropic.barotropic_literal_fold_unit_vector_form),
            ) if value
        }
        assert actual == set(expected)
    with pytest.raises(gate.GateError, match="outside A/B/C"):
        gate.arm_card(base, "npcrv_f_area", plant="extra-selector")


def test_npcrv_area_uses_raw_nemo_f_metrics_and_control_can_fail():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _nemo_f_vertex_area,
    )

    card = _card()
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    actual = np.asarray(_nemo_f_vertex_area(grid, z_coord))
    raw = z_coord.nemo_een_barotropic
    expected = (np.asarray(raw.e1f, dtype=actual.dtype)
                * np.asarray(raw.e2f, dtype=actual.dtype))
    np.testing.assert_array_equal(actual[1:, 1:], expected)
    np.testing.assert_array_equal(actual[1:, 0], expected[:, -1])
    np.testing.assert_array_equal(actual[0], np.asarray(grid.area_q)[0])
    assert np.count_nonzero(np.asarray(grid.area_q)[1:, 1:] != expected) > 0
    with pytest.raises(ValueError, match="e1f and e2f"):
        _nemo_f_vertex_area(
            grid, z_coord._replace(nemo_een_barotropic=None))


def test_npcrv_selector_is_bound_at_the_executing_call_site():
    """Plant from the round-282 review: hard-wiring the kwarg false fails."""
    import inspect
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe

    source = inspect.getsource(pe.latlon_cgrid_ocean_baroclinic_tendencies)
    assert "een_nemo_zeta=config.een_nemo_relative_vorticity" in source
    assert "een_nemo_top_f_u_halo=config.een_nemo_top_f_u_halo" in source


def test_npcrv_area_selector_changes_live_relative_vorticity_by_area_ratio(
    monkeypatch,
):
    """Exercise statement A through its live caller on a small grid.

    The two-times NEMO F area makes the selected relative vorticity exactly
    one half of the unselected value.  This goes red if either the
    ``een_nemo_zeta`` dispatch clause or the call-site ``nemo_f_area`` operand
    is removed.
    """
    import jax.numpy as jnp

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
        S_uniform=35.0, H_max=1000.0,
    )
    jj_u = jnp.arange(grid.n_lat, dtype=jnp.float64)[:, None, None]
    jj_v = jnp.arange(grid.n_lat + 1, dtype=jnp.float64)[:, None, None]
    ii_u = jnp.arange(grid.n_lon + 1, dtype=jnp.float64)[None, :, None]
    ii_v = jnp.arange(grid.n_lon, dtype=jnp.float64)[None, :, None]
    u = jnp.sin((jj_u + 1.0) * (ii_u + 1.0) / 17.0) * 0.03
    v = jnp.cos((jj_v + 1.0) * (ii_v + 2.0) / 19.0) * 0.02
    u = jnp.broadcast_to(u, state.u.data.shape)
    v = jnp.broadcast_to(v, state.v.data.shape).at[0].set(0.0).at[-1].set(0.0)
    state = state._replace(
        u=state.u.replace(data=u),
        v=state.v.replace(data=v),
    )

    generic_area = jnp.asarray(grid.area_q)
    nemo_area = generic_area * jnp.float64(2.0)
    monkeypatch.setattr(pe, "_nemo_f_vertex_area", lambda *_args: nemo_area)
    monkeypatch.setattr(
        pe, "curl_vertex_cgrid",
        lambda u_arg, v_arg, grid_arg: nemo_vor_ene_vorticity_cgrid(
            u_arg, v_arg, grid_arg),
    )

    captured = []

    def capture_zeta(zeta, _h_vtx, _h_v, v_arg, _h_u, u_arg,
                     *_args, **_kwargs):
        captured.append(np.asarray(zeta))
        return jnp.zeros_like(u_arg), jnp.zeros_like(v_arg)

    monkeypatch.setattr(pe, "pv_flux_al81_partial_cell", capture_zeta)
    base = LatLonCGridOceanConfig.from_flat(
        A_h=0.0, A_v=0.0, K_h=0.0, K_v=0.0, bottom_drag_r=0.0,
        vorticity_scheme="een_total", een_metric_weighting="nemo",
        een_nemo_relative_vorticity=False,
    )
    for enabled in (False, True):
        pe.latlon_cgrid_ocean_baroclinic_tendencies(
            state, grid, z_coord,
            base._replace(een_nemo_relative_vorticity=enabled),
            momentum_only=True,
        )
    assert len(captured) == 2
    off, on = captured
    assert np.count_nonzero(off) > 0
    np.testing.assert_array_equal(on, off * np.float64(0.5))


def test_npcrv_area_refuses_non_nemo_metric_weighting():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_pv_flux

    with pytest.raises(ValueError, match="requires een_metric_weighting='nemo'"):
        _bc_pv_flux(*([None] * 12), een_metric_weighting="off",
                    een_nemo_zeta=True)


def test_vector_fold_unit_predicate_is_selected_only_by_its_arm():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_external_mode_active,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round282_g1b_orca2_gate as gate,
    )

    base = _card()
    assert not nemo_literal_external_mode_active(
        gate.arm_card(base, "control").recipe.model_config)
    assert nemo_literal_external_mode_active(
        gate.arm_card(base, "vector_fold_unit").recipe.model_config)

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
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe


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

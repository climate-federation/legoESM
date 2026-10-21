"""L4b: certified L2 plus geothermal in NEMO's stage-3 call order."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES,
    CASES,
    G5,
    INPUTS,
    _differs,
    _flat,
    _root,
    _values,
)


L2 = CASES / "nemo_testcase_l2_orca2iso"
L4B = CASES / "nemo_testcase_l4b_orca2iso_bbc"
RUN = L4B / "run.sh"
L4B_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/"
    "oracle_l4b_r54/twin_a")
needs_l4b_record = pytest.mark.skipif(
    not (L4B_RECORD / "oracle_geothermal_kt00000001.bin").is_file(),
    reason="no admitted L4b record")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


L2_DECK = _module("l4b_l2_deck_test", L2 / "l2_deck.py")
L4B_DECK = _module("l4b_deck_test", L4B / "l4b_deck.py")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _rendered_root(base: Path, name: str, renderer) -> Path:
    root = _root(base, G5, name)
    (root / "namelist_cfg").write_text(renderer.render_text(
        (root / "namelist_cfg").read_text()))
    return root


def test_l4b_deck_is_l2_plus_exact_geothermal_block_only(tmp_path):
    l2_root = _rendered_root(tmp_path, "l2", L2_DECK)
    l4b_root = _rendered_root(tmp_path, "l4b", L4B_DECK)
    l2 = _values(l2_root / "namelist_cfg")
    l4b = _values(l4b_root / "namelist_cfg")
    changed = {key: (l2.get(key), l4b.get(key)) for key in set(l2) | set(l4b)
               if l2.get(key) != l4b.get(key)}
    assert changed == {
        "nambbc.ln_trabbc": (False, True),
        "nambbc.nn_geoflx": (None, 2),
        "nambbc.cn_dir": (None, "'./'"),
        "nambbc.sn_qgh": (
            None,
            "'geothermal_heating.nc', -12., 'heatflow', .false., .true., "
            "'yearly', '', '', ''",
        ),
    }
    assert {key: value for key, value in l4b.items()
            if key.startswith("namtra_ldf.")} == recipe._ORCA2L2_LDF_FIELDS
    assert l4b["namtra_eiv.ln_ldfeiv"] is False
    assert l4b["nambbl.ln_trabbl"] is False
    assert l4b["namtra_dmp.ln_tradmp"] is False
    with pytest.raises(ValueError, match="certified G5 namtra_ldf"):
        L4B_DECK.render_text(L4B_DECK.render_text(
            (G5 / "namelist_cfg").read_text()))


def test_acquisition_entrypoint_has_all_modes_and_no_pipefail_grep_trap():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    for token in ("--preflight-only", "--run", "--admit-existing",
                  "L4B_PREFLIGHT_7_OF_7_PASS", "smoke_2step",
                  "ref_100step", "twin_a", "twin_b", "twin_c",
                  "check_numeric_log", "check_acting", "G5_RECORDS"):
        assert token in text
    assert "| grep -q" not in text
    admit_start = text.index("admit() {")
    admit = text[admit_start:text.index("\npreflight\n", admit_start)]
    assert 'check_acting "$SMOKE"\n' in admit
    assert 'check_acting_scope "$SMOKE" "$REF" ||' not in admit
    preflight = text[
        text.index("seven_item_preflight() {"):text.index("\npreflight() {")
    ]
    assert "trap 'rm -rf -- \"$scratch\"' EXIT" in preflight


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l4b")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        l2 = recipe.build_orca2l2_iso_card(
            _rendered_root(base, "l2_card", L2_DECK))
        l4b = recipe.build_orca2l4b_iso_bbc_card(
            _rendered_root(base, "l4b_card", L4B_DECK))
        return l2, l4b
    finally:
        set_policy(before)


def test_l4b_card_is_l2_plus_only_the_two_geothermal_order_selectors(cards):
    l2, l4b = cards
    a, b = _flat(l2.recipe.model_config), _flat(l4b.recipe.model_config)
    changed = {key for key in set(a) | set(b) if _differs(a.get(key), b.get(key))}
    assert changed == {
        "rk3_stage3_geothermal_nemo_written",
        "rk3_stage3_geothermal_after_ldf_nemo_order",
    }
    cfg = l4b.recipe.model_config
    assert cfg.rk3_stage3_geothermal_nemo_written is True
    assert cfg.rk3_stage3_geothermal_after_ldf_nemo_order is True
    assert recipe._orca2l2_isoneutral_selected(cfg.gm_redi)
    assert cfg.tracer_advection == "none"
    assert cfg.bbl_adv_option == cfg.bbl_diffusive_option == 0
    assert recipe._orca2g1_selector_failures(l4b) == []
    assert l4b.unmeasured_features == recipe.ORCA2L4B_UNMEASURED
    for got, want in zip(jax.tree_util.tree_leaves(l4b.recipe.initial_state),
                         jax.tree_util.tree_leaves(l2.recipe.initial_state)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_l4b_guard_accepts_only_geothermal_after_certified_l2(cards):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    l2, l4b = cards
    l2_cfg, cfg = l2.recipe.model_config, l4b.recipe.model_config
    LatLonCGridOceanModel._validate_config(cfg)
    with pytest.raises(ValueError, match="select the L4b ordered composition"):
        LatLonCGridOceanModel._validate_config(cfg._replace(
            rk3_stage3_geothermal_after_ldf_nemo_order=False))
    with pytest.raises(ValueError, match="requires rk3_stage3_geothermal"):
        LatLonCGridOceanModel._validate_config(l2_cfg._replace(
            rk3_stage3_geothermal_after_ldf_nemo_order=True))
    with pytest.raises(ValueError, match="certified L2"):
        LatLonCGridOceanModel._validate_config(cfg._replace(gm_redi=None))
    with pytest.raises(ValueError, match="certified L2"):
        LatLonCGridOceanModel._validate_config(cfg._replace(
            gm_redi=cfg.gm_redi._replace(kappa_GM=1.0)))
    with pytest.raises(ValueError, match="certified L2"):
        LatLonCGridOceanModel._validate_config(cfg._replace(K_h=1.0))
    with pytest.raises(ValueError, match="written stage-3 Krhs"):
        LatLonCGridOceanModel._validate_config(cfg._replace(
            rk3_stage3_tracer_krhs_nemo_order=True))
    with pytest.raises(ValueError, match="does not support tra_bbl"):
        LatLonCGridOceanModel._validate_config(cfg._replace(
            bbl_diffusive_option=1))


def test_one_step_l4b_is_l2_plus_one_bottom_geothermal_source(cards):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    l2, l4b = cards
    state = l2.recipe.initial_state
    zero = jnp.zeros_like(state.eta.data)
    one = jnp.ones_like(state.eta.data)
    freshwater = FreshwaterForcing(
        precip=zero, evap=zero, runoff=zero, ice_fw=zero, restoring=zero)
    forcing = OceanSurfaceForcing(
        sw_down=one, q_net=one, tau_x=zero, tau_y=zero, salt_flux=zero,
        chl=0.1 * one, taum=zero, ice_concentration=zero,
        tau_i_native=zero, tau_j_native=zero,
        runoff_tracer_content=(zero, zero), q_nonsolar=one,
        ice_ocean_top_drag_rCdu=zero, salt_flux_sfx=zero,
        geothermal_heat_flux_mwm2=one)
    hooks = _NEMOWSRK3TestHooks(expose_live_stage_operands=True)

    def observed_step(card):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=forcing))

    l2_trace, l4b_trace = observed_step(l2), observed_step(l4b)
    l2_sources, l2_thickness = l2_trace.stage_tracer_sources
    l4b_sources, l4b_thickness = l4b_trace.stage_tracer_sources

    def assert_bits_equal(got, want):
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64))

    for got, want in zip(jax.tree_util.tree_leaves(l4b_thickness),
                         jax.tree_util.tree_leaves(l2_thickness)):
        assert_bits_equal(got, want)
    for stage in (0, 1):
        for tracer in (0, 1):
            assert_bits_equal(l4b_sources[stage][tracer], l2_sources[stage][tracer])
    assert_bits_equal(
        l4b_trace.stage3_composed_source[1],
        l2_trace.stage3_composed_source[1])
    assert l4b_trace.stage3_before_geothermal is not None
    assert_bits_equal(
        l4b_trace.stage3_before_geothermal[0],
        l2_trace.stage3_composed_source[0])
    assert_bits_equal(
        l4b_trace.stage3_before_geothermal[1],
        l2_trace.stage3_composed_source[1])

    assert l4b_trace.stage3_geothermal_rate is not None
    expected_composed = nemo_source_round(
        l4b_trace.stage3_before_geothermal[0]
        + l4b_trace.stage3_geothermal_rate)
    assert_bits_equal(
        l4b_trace.stage3_composed_source[0], expected_composed)

    active = jnp.asarray(l4b.recipe.z_coord.is_active)
    source_unequal = (
        np.asarray(l4b_trace.stage3_composed_source[0]).view(np.uint64)
        != np.asarray(l4b_trace.stage3_before_geothermal[0]).view(np.uint64))
    wet = np.asarray(active) > 0.5
    bottom = np.zeros_like(wet)
    jj, ii = np.nonzero(wet.any(axis=-1))
    kb = wet.sum(axis=-1) - 1
    bottom[jj, ii, kb[jj, ii]] = True
    np.testing.assert_array_equal(source_unequal, bottom)

    explicit_unequal = (
        np.asarray(l4b_trace.stage3_explicit_temperature).view(np.uint64)
        != np.asarray(l2_trace.stage3_explicit_temperature).view(np.uint64))
    np.testing.assert_array_equal(explicit_unequal, bottom)

    for name in ("S", "u", "v", "eta"):
        assert_bits_equal(
            getattr(l4b_trace.state_after, name).data,
            getattr(l2_trace.state_after, name).data)
    l2_t = np.asarray(l2_trace.state_after.T.data)
    l4b_t = np.asarray(l4b_trace.state_after.T.data)
    unequal = l4b_t.view(np.uint64) != l2_t.view(np.uint64)
    # The producing stage-3 source is bottom-only (asserted exactly above).
    # The subsequent implicit vertical solve legitimately carries that signal
    # upward within each heated wet column; it must not reach a dry cell, and
    # every wet bottom cell must remain observably changed.
    assert not np.any(unequal[~wet])
    assert np.all(unequal[bottom])


def test_ordered_stage3_add_is_exact_jit_traced_x64_and_gradient_finite():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_stage3_ldf_then_geothermal,
    )

    # Bind the exact kernel result below to the production stage-3 wiring.
    # Removing the L4b call while leaving the helper behind must still fail.
    production = inspect.getsource(LatLonCGridOceanModel)
    assert "_ordered_stage3 = _nemo_stage3_ldf_then_geothermal(" in production

    active = jnp.ones((1,), dtype=jnp.float64)
    stage_t = jnp.array([1.0e16], dtype=jnp.float64)
    stage_s = jnp.array([3.0], dtype=jnp.float64)
    dT = jnp.array([-1.0e16], dtype=jnp.float64)
    dS = jnp.array([4.0], dtype=jnp.float64)
    qgh = jnp.array([1.0], dtype=jnp.float64)
    kernel = jax.jit(_nemo_stage3_ldf_then_geothermal)
    got_t, got_s = kernel(stage_t, stage_s, dT, dS, qgh, active)
    np.testing.assert_array_equal(np.asarray(got_t).view(np.uint64),
                                  np.array([1.0], np.float64).view(np.uint64))
    np.testing.assert_array_equal(np.asarray(got_s).view(np.uint64),
                                  np.array([7.0], np.float64).view(np.uint64))
    # The forbidden tra_bbc -> tra_ldf association is observably different.
    reversed_t = (np.float64(1.0e16) + np.float64(1.0)) + np.float64(-1.0e16)
    assert reversed_t == 0.0 and np.asarray(got_t)[0] != reversed_t

    def objective(t, ldf, bbc):
        value, _ = _nemo_stage3_ldf_then_geothermal(
            t, stage_s, ldf, dS, bbc, active)
        return jnp.sum(value)

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(
        stage_t, dT, qgh)
    assert all(np.isfinite(np.asarray(value)).all() for value in gradients)
    for value in gradients:
        np.testing.assert_array_equal(np.asarray(value), np.ones((1,), np.float64))


@needs_l4b_record
def test_l4b_record_is_exact_under_jit_traced_inputs_and_x64(
    monkeypatch,
):
    scorer = _module(
        "l4b_ladder_score",
        CASES / "nemo_testcase_g2_orca2bathy_ladder.py",
    )
    real_jit = jax.jit
    traced_calls = []

    def recording_jit(fun, *jit_args, **jit_kwargs):
        compiled = real_jit(fun, *jit_args, **jit_kwargs)

        def recorded(*args, **kwargs):
            traced_calls.append(tuple(
                (np.shape(arg), np.asarray(arg).dtype) for arg in args))
            return compiled(*args, **kwargs)

        return recorded

    monkeypatch.setattr(jax, "jit", recording_jit)
    exact = scorer.run(
        L4B_RECORD, arm="geothermal", kt_from=1, kt_to=1,
        allow_dirty=True, case="ORCA2L4b-iso-bbc",
    )
    assert exact["jit"] is True
    assert exact["traced_inputs"] == ["e3t_Kmm", "heatflow_mwm2"]
    assert any(
        len(call) == 2
        and len(call[0][0]) == 3 and len(call[1][0]) == 2
        and call[0][1] == call[1][1] == np.dtype(np.float64)
        for call in traced_calls
    )
    assert exact["per_kt"][0]["trend_row"]["n_unequal"] == 0

    removed = scorer.run(
        L4B_RECORD, arm="geothermal", kt_from=1, kt_to=1,
        allow_dirty=True, without_geothermal=True,
        case="ORCA2L4b-iso-bbc",
    )
    assert removed["per_kt"][0]["trend_row"]["n_unequal"] == 16_428

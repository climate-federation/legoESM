"""L6: certified L4b plus ORCA2 interior T/S damping."""

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
from legoesm.ocean.sponge import SpongeForcing
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


L4B = CASES / "nemo_testcase_l4b_orca2iso_bbc"
L6 = CASES / "nemo_testcase_l6_orca2dmp"
RUN = L6 / "run.sh"
RESTO = INPUTS / "resto.nc"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


L4B_DECK = _module("l6_l4b_deck_test", L4B / "l4b_deck.py")
L6_DECK = _module("l6_deck_test", L6 / "l6_deck.py")


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
    if renderer is L6_DECK:
        (root / "resto.nc").symlink_to(RESTO)
    return root


def _assert_bits_equal(got, want):
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64))


def test_l6_deck_is_l4b_plus_exact_damping_block_only(tmp_path):
    l4b_root = _rendered_root(tmp_path, "l4b", L4B_DECK)
    l6_root = _rendered_root(tmp_path, "l6", L6_DECK)
    l4b = _values(l4b_root / "namelist_cfg")
    l6 = _values(l6_root / "namelist_cfg")
    changed = {key: (l4b.get(key), l6.get(key))
               for key in set(l4b) | set(l6) if l4b.get(key) != l6.get(key)}
    assert changed == {
        "namtra_dmp.ln_tradmp": (False, True),
        "namtra_dmp.nn_zdmp": (None, 0),
    }
    assert l6["nambbl.ln_trabbl"] is False
    assert l6["nambbc.ln_trabbc"] is True
    assert l6["namtra_eiv.ln_ldfeiv"] is False
    with pytest.raises(ValueError):
        L6_DECK.render_text(L6_DECK.render_text(
            (G5 / "namelist_cfg").read_text()))


def test_acquisition_entrypoint_has_ladder_boundary():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    for token in ("--preflight-only", "--run", "--admit-existing",
                  "L6_PREFLIGHT_7_OF_7_PASS", "smoke_2step",
                  "ref_100step", "twin_a", "twin_b", "twin_c",
                  "check_numeric_log", "check_acting", "resto.nc",
                  "STAMPED_RENDERED_DECK"):
        assert token in text
    assert "| grep -q" not in text
    assert "trap 'rm -rf -- \"$scratch\"' EXIT" in text
    admit_start = text.index("admit() {")
    admit = text[admit_start:text.index("\npreflight\n", admit_start)]
    assert 'check_acting "$SMOKE"\n' in admit


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES) or not RESTO.is_file():
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l6")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        l4b = recipe.build_orca2l4b_iso_bbc_card(
            _rendered_root(base, "l4b_card", L4B_DECK))
        l6_root = _rendered_root(base, "l6_card", L6_DECK)
        l6 = recipe.build_orca2l6_dmp_card(l6_root)
        return l4b, l6, l6_root
    finally:
        set_policy(before)


def test_l6_card_is_l4b_plus_only_damping_selector_and_forcing(cards):
    l4b, l6, _ = cards
    a, b = _flat(l4b.recipe.model_config), _flat(l6.recipe.model_config)
    changed = {key for key in set(a) | set(b) if _differs(a.get(key), b.get(key))}
    assert changed == {"rk3_stage3_damping_after_geothermal_nemo_order"}
    cfg = l6.recipe.model_config
    assert cfg.rk3_stage3_damping_after_geothermal_nemo_order is True
    assert cfg.rk3_stage3_geothermal_after_ldf_nemo_order is True
    assert recipe._orca2l2_isoneutral_selected(cfg.gm_redi)
    assert cfg.bbl_adv_option == cfg.bbl_diffusive_option == 0
    assert l4b.recipe.tracer_damping is None
    assert l6.recipe.tracer_damping is not None
    assert l6.recipe.tracer_damping.gamma.shape == l6.recipe.initial_state.T.data.shape
    assert recipe._orca2g1_selector_failures(l6) == []
    assert l6.unmeasured_features == recipe.ORCA2L6_UNMEASURED
    for got, want in zip(jax.tree_util.tree_leaves(l6.recipe.initial_state),
                         jax.tree_util.tree_leaves(l4b.recipe.initial_state)):
        _assert_bits_equal(got, want)


def test_l6_guard_requires_complete_l4b_and_no_bbl(cards):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    l4b, l6, _ = cards
    cfg = l6.recipe.model_config
    LatLonCGridOceanModel._validate_config(cfg)
    with pytest.raises(ValueError, match="L4b ordered composition"):
        LatLonCGridOceanModel._validate_config(cfg._replace(
            rk3_stage3_geothermal_after_ldf_nemo_order=False))
    with pytest.raises(ValueError, match="does not support tra_bbl"):
        LatLonCGridOceanModel._validate_config(cfg._replace(
            bbl_diffusive_option=1))
    with pytest.raises(ValueError, match="certified L2"):
        LatLonCGridOceanModel._validate_config(cfg._replace(gm_redi=None))
    assert not l4b.recipe.model_config.rk3_stage3_damping_after_geothermal_nemo_order


def test_climatology_and_resto_inputs_are_exact_and_step_varying(cards):
    _, l6, root = cards
    tmask = np.asarray(l6.recipe.z_coord.is_active) > 0.5
    kt1 = recipe.build_orca2_damping_for_step(root, tmask, kt=1, dt_s=l6.dt_s)
    kt2 = recipe.build_orca2_damping_for_step(root, tmask, kt=2, dt_s=l6.dt_s)
    _assert_bits_equal(kt1.gamma, l6.recipe.tracer_damping.gamma)
    _assert_bits_equal(kt1.T_ref, l6.recipe.initial_state.T.data)
    _assert_bits_equal(kt1.S_ref, l6.recipe.initial_state.S.data)
    assert np.any(np.asarray(kt2.T_ref).view(np.uint64)
                  != np.asarray(kt1.T_ref).view(np.uint64))
    rate = np.asarray(kt1.gamma)
    assert np.isfinite(rate).all()
    assert rate.min() == 0.0
    assert rate.max() == np.float64(1.1574074074074073e-05)


def test_one_step_l6_is_l4b_plus_final_damping_statement(cards):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    l4b, l6, _ = cards
    base = l6.recipe.initial_state
    rate = l6.recipe.tracer_damping.gamma
    wet_damped = (rate > 0.0).astype(jnp.float64)
    state = base._replace(
        T=base.T.replace(data=base.T.data + wet_damped),
        S=base.S.replace(data=base.S.data - 2.0 * wet_damped),
    )
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

    def observed(card, sponge):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=forcing, sponge=sponge))

    l4b_trace = observed(l4b, None)
    l6_trace = observed(l6, l6.recipe.tracer_damping)
    assert l6_trace.stage3_before_damping is not None
    assert l6_trace.stage3_damping_rate is not None
    for got, want in zip(l6_trace.stage3_before_damping,
                         l4b_trace.stage3_composed_source):
        _assert_bits_equal(got, want)
    expected_t = nemo_source_round(
        l6_trace.stage3_before_damping[0]
        + l6.recipe.tracer_damping.gamma
        * (l6.recipe.tracer_damping.T_ref - state.T.data))
    expected_s = nemo_source_round(
        l6_trace.stage3_before_damping[1]
        + l6.recipe.tracer_damping.gamma
        * (l6.recipe.tracer_damping.S_ref - state.S.data))
    _assert_bits_equal(l6_trace.stage3_composed_source[0], expected_t)
    _assert_bits_equal(l6_trace.stage3_composed_source[1], expected_s)
    assert np.any(np.asarray(l6_trace.stage3_composed_source[0]).view(np.uint64)
                  != np.asarray(l6_trace.stage3_before_damping[0]).view(np.uint64))


def test_ordered_damping_add_is_jit_traced_x64_and_gradient_finite():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_stage3_l4b_then_damping,
    )

    production = inspect.getsource(LatLonCGridOceanModel)
    assert "_ordered_stage3 = _nemo_stage3_l4b_then_damping(" in production
    assert production.index("_nemo_stage3_ldf_then_geothermal(") < production.index(
        "_nemo_stage3_l4b_then_damping(")

    def kernel(l4b_t, l4b_s, T, S, gamma, T_ref, S_ref):
        sponge = SpongeForcing(gamma=gamma, T_ref=T_ref, S_ref=S_ref)
        return _nemo_stage3_l4b_then_damping(l4b_t, l4b_s, T, S, sponge)

    args = tuple(jnp.asarray([x], dtype=jnp.float64) for x in (
        0.0, 7.0, 2.0, 3.0, 1.0, 3.0, 5.0))
    got_t, got_s = jax.jit(kernel)(*args)
    _assert_bits_equal(got_t, np.asarray([1.0], dtype=np.float64))
    _assert_bits_equal(got_s, np.asarray([9.0], dtype=np.float64))

    # Position discriminator: LDF(1e16), BBC(-1e16), then DMP(+1) is one;
    # moving DMP before BBC loses the unit in binary64 and produces zero.
    correct = np.float64(
        (np.float64(1.0e16) + np.float64(-1.0e16)) + np.float64(1.0))
    moved = np.float64(
        (np.float64(1.0e16) + np.float64(1.0)) + np.float64(-1.0e16))
    assert correct == 1.0 and moved == 0.0

    def scalar_sum(x):
        values = list(args)
        values[2] = jnp.asarray([x], dtype=jnp.float64)
        out = kernel(*values)
        return out[0][0] + out[1][0]

    assert np.isfinite(np.asarray(jax.grad(scalar_sum)(2.0)))

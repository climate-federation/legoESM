"""L8: G9 FCT plus BBL/isoneutral/geothermal/damping Krhs order."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES, CASES, INPUTS, _root, _values,
)


G9 = CASES / "nemo_testcase_g9_orca2tadv"
L8 = CASES / "nemo_testcase_l8_orca2krhs"
RUN = L8 / "run.sh"
RESTO = INPUTS / "resto.nc"
L8_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
    "round56/oracle_l8_r56/twin_a")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


L8_DECK = _module("l8_deck_test", L8 / "l8_deck.py")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_l8_deck_is_g9_plus_exact_orca2_active_blocks(tmp_path):
    rendered = tmp_path / "namelist_cfg"
    rendered.write_text(L8_DECK.render_text(
        (G9 / "namelist_cfg").read_text()))
    values = _values(rendered)
    assert values["nambbl.ln_trabbl"] is True
    assert values["nambbl.nn_bbl_ldf"] == 1
    assert values["nambbl.nn_bbl_adv"] == 0
    assert values["nambbl.rn_ahtbbl"] == 1000.0
    assert values["nambbl.rn_gambbl"] == 10.0
    assert values["namtra_ldf.ln_traldf_iso"] is True
    assert values["nambbc.ln_trabbc"] is True
    assert values["namtra_dmp.ln_tradmp"] is True
    assert values["namtra_adv.ln_traadv_fct"] is True
    with pytest.raises(ValueError):
        L8_DECK.render_text(L8_DECK.render_text(
            (G9 / "namelist_cfg").read_text()))


def test_l8_acquisition_has_fail_closed_ladder_boundary():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    for token in ("--preflight-only", "--run", "--admit-existing",
                  "NEMO_OPERATOR_ACK", "smoke_2step", "ref_100step",
                  "twin_a", "twin_b", "twin_c", "check_acting \"$SMOKE\"",
                  "gfortran -std=f2008", "REAL_LOG", "stamped.output",
                  "L8_PREFLIGHT_7_OF_7_PASS"):
        assert token in text
    assert "trap 'rm -rf -- \"$scratch\"' EXIT" in text
    operational = "\n".join(line for line in text.splitlines()
                            if "grep -F" not in line)
    assert 'check_acting "$REF"' not in operational


def test_one_step_l8_pins_every_inexact_statement_boundary_bit_for_bit():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_stage3_l8_krhs_walk,
    )

    words = [float.fromhex(value) for value in (
        "0x1.8484bbe636a7cp+25", "0x1.80c4d885778a5p+25",
        "-0x1.e4576fad0145bp+34", "-0x1.fbbb660e98d04p+8",
        "0x1.99a22105eba7ep+34", "0x1.792a95b49c117p+27",
        "-0x1.0452f7986b12bp+33",
    )]
    expected = [float.fromhex(value) for value in (
        "0x1.8484bbe636a7cp+25", "0x1.82a4ca35d7190p+26",
        "-0x1.e2d4cae2cb6e9p+34", "-0x1.e2d4cb61ba481p+34",
        "-0x1.24caa96f3a80cp+32", "-0x1.190154c195a03p+32",
        "-0x1.90d3a1f935e2cp+33",
    )]
    adv, *terms = [jnp.asarray([value], dtype=jnp.float64) for value in words]
    groups = tuple((((term,), (term,))) for term in terms)
    _, _, boundaries = jax.jit(_nemo_stage3_l8_krhs_walk)(
        adv, adv, *groups)
    for (got_t, got_s), want in zip(boundaries, expected):
        want_bits = np.asarray([want], dtype=np.float64).view(np.uint64)
        np.testing.assert_array_equal(np.asarray(got_t).view(np.uint64), want_bits)
        np.testing.assert_array_equal(np.asarray(got_s).view(np.uint64), want_bits)

    # Planted call moves: LDF/BBC and BBL/DMP swaps must trip position pins,
    # even where a final-only sum would hide an earlier commutative move.
    for planted in (
            groups[:2] + (groups[3], groups[2]) + groups[4:],
            groups[:4] + (groups[5], groups[4])):
        _, _, moved = _nemo_stage3_l8_krhs_walk(adv, adv, *planted)
        assert any(np.any(np.asarray(got[0]).view(np.uint64)
                          != np.asarray(want[0]).view(np.uint64))
                   for got, want in zip(moved, boundaries))


def test_l8_order_is_jitted_and_has_finite_gradient():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_stage3_l8_krhs_walk,
    )

    def scalar(x):
        terms = tuple(((jnp.asarray([v]),), (jnp.asarray([v]),))
                      for v in (0.2, -3.0, 7.0, 11.0, -13.0, x))
        t, s, _ = _nemo_stage3_l8_krhs_walk(
            jnp.asarray([17.0]), jnp.asarray([19.0]), *terms)
        return t[0] + s[0]

    value = jax.jit(scalar)(jnp.asarray(0.1, dtype=jnp.float64))
    grad = jax.grad(scalar)(jnp.asarray(0.1, dtype=jnp.float64))
    assert np.isfinite(np.asarray(value)) and np.isfinite(np.asarray(grad))


@pytest.mark.skipif(not (L8_RECORD / "oracle_rktracer_stage3_kt00000001.bin").is_file(),
                    reason="no admitted round-56 L8 record")
def test_l8_record_requires_jit_traced_inputs_and_x64():
    ladder = _module(
        "l8_ladder_score",
        Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_g2_orca2bathy_ladder.py",
    )

    def score(**plants):
        return ladder.run(
            L8_RECORD, arm="l8_krhs", kt_from=1, kt_to=1,
            allow_dirty=True, case="ORCA2L8-krhs", **plants)

    selected = score()
    assert selected["gate_pass"] is True
    assert all(row["n_unequal"] == row["dry_card_nonzero"] == 0
               for point in selected["per_kt"] for row in point["rows"])
    for plant in (
        {"without_l8_jit": True},
        {"without_l8_traced_inputs": True},
        {"without_l8_x64": True},
    ):
        assert score(**plant)["gate_pass"] is False


@pytest.fixture(scope="module")
def l8_card(tmp_path_factory):
    if (not all((INPUTS / name).is_file() for name in CARD_FILES)
            or not RESTO.is_file()):
        pytest.skip("no ORCA2 inputs")
    root = _root(tmp_path_factory.mktemp("l8"), G9, "l8")
    (root / "namelist_cfg").write_text(L8_DECK.render_text(
        (root / "namelist_cfg").read_text()))
    (root / "resto.nc").symlink_to(RESTO)
    return recipe.build_orca2l8_krhs_card(root)


def test_l8_card_selects_only_the_exact_supported_composition(l8_card):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    cfg = l8_card.recipe.model_config
    assert cfg.rk3_stage3_l8_nemo_order is True
    assert cfg.tracer_advection == "fct2"
    assert (cfg.bbl_adv_option, cfg.bbl_diffusive_option,
            cfg.bbl_aht_m2_s, cfg.bbl_gamma_s) == (0, 1, 1000.0, 10.0)
    assert recipe._orca2l2_isoneutral_selected(cfg.gm_redi)
    assert cfg.rk3_stage3_geothermal_after_ldf_nemo_order is True
    assert cfg.rk3_stage3_damping_after_geothermal_nemo_order is True
    assert recipe._orca2g1_selector_failures(l8_card) == []
    LatLonCGridOceanModel._validate_config(cfg)
    for field in ("rk3_stage3_tracer_krhs_nemo_order",
                  "rk3_stage3_geothermal_nemo_written",
                  "rk3_stage3_damping_after_geothermal_nemo_order",
                  "bbl_nemo_stage3_krhs"):
        with pytest.raises(ValueError):
            LatLonCGridOceanModel._validate_config(cfg._replace(**{field: False}))


def test_one_step_l8_route_executes_with_finite_tracers(l8_card):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    state = l8_card.recipe.initial_state
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

    def step():
        model = LatLonCGridOceanModel(
            l8_card.recipe.grid, l8_card.recipe.z_coord,
            l8_card.recipe.model_config)
        return jax.device_get(model.step(
            state, dt=l8_card.dt_s, freshwater=freshwater,
            surface_forcing=forcing, sponge=l8_card.recipe.tracer_damping,
            nemo_kt=1))

    correct = step()
    assert all(np.isfinite(np.asarray(getattr(correct, name).data)).all()
               for name in ("T", "S"))


def test_one_step_l8_production_route_passes_bbc_bbl_dmp_by_identity(
    l8_card, monkeypatch,
):
    """The live finalizer, not only the walker helper, owns NEMO's order."""
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as cgrid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.physics import bbl_adv
    from legoesm.ocean.state import OceanSurfaceForcing

    state = l8_card.recipe.initial_state
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
    sponge = l8_card.recipe.tracer_damping
    captured = {}

    original_bbc = cgrid._nemo_stage3_ldf_then_geothermal
    original_bbl = bbl_adv.apply_bbl_diffusive_tendency
    original_walk = cgrid._nemo_stage3_l8_krhs_walk

    def spy_bbc(*args, **kwargs):
        captured.setdefault("geothermal_rates", []).append(args[4])
        return original_bbc(*args, **kwargs)

    def spy_bbl(*args, **kwargs):
        result = original_bbl(*args, **kwargs)
        captured["bbl"] = result
        return result

    def spy_walk(adv_t, adv_s, *groups):
        captured["groups"] = groups
        return original_walk(adv_t, adv_s, *groups)

    monkeypatch.setattr(cgrid, "_nemo_stage3_ldf_then_geothermal", spy_bbc)
    monkeypatch.setattr(bbl_adv, "apply_bbl_diffusive_tendency", spy_bbl)
    monkeypatch.setattr(cgrid, "_nemo_stage3_l8_krhs_walk", spy_walk)
    model = LatLonCGridOceanModel(
        l8_card.recipe.grid, l8_card.recipe.z_coord,
        l8_card.recipe.model_config)
    model.prime_step_caches(state)
    model._step_impl(
        state, dt=l8_card.dt_s, freshwater=freshwater,
        surface_forcing=forcing, sponge=sponge)

    groups = captured["groups"]
    # zero-based group 3/4/5 are BBC, BBL and DMP at the production call.
    # The first composition call receives BBC; the second uses zero BBC to
    # isolate LDF.  The walker must receive the first call's exact object.
    assert groups[3][0][0] is captured["geothermal_rates"][0]
    assert groups[3][1] == ()
    assert groups[4][0][0] is captured["bbl"][0]
    assert groups[4][1][0] is captured["bbl"][1]
    bbl_t, bbl_s = (np.asarray(groups[4][i][0]) for i in (0, 1))
    wet = np.asarray(l8_card.recipe.z_coord.is_active) > 0.5
    bottom = wet & ~np.concatenate(
        (wet[..., 1:], np.zeros_like(wet[..., :1])), axis=-1)
    assert np.any(bbl_t) and np.any(bbl_s)
    assert not np.any((bbl_t != 0.0) & ~bottom)
    assert not np.any((bbl_s != 0.0) & ~bottom)
    gamma = jnp.asarray(sponge.gamma, dtype=state.T.data.dtype)
    expected_t = cgrid.nemo_source_round(
        gamma * (jnp.asarray(sponge.T_ref) - state.T.data))
    expected_s = cgrid.nemo_source_round(
        gamma * (jnp.asarray(sponge.S_ref) - state.S.data))
    np.testing.assert_array_equal(np.asarray(groups[5][0][0]), expected_t)
    np.testing.assert_array_equal(np.asarray(groups[5][1][0]), expected_s)

"""L9: certified L8 stage-3 order plus certified L3 EIV transport."""

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
L9 = CASES / "nemo_testcase_l9_orca2eiv_krhs"
RUN = L9 / "run.sh"
RESTO = INPUTS / "resto.nc"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


L9_DECK = _module("l9_deck_test", L9 / "l9_deck.py")
L9_RECORDS = _module("l9_records_test", L9 / "l9_records.py")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


@pytest.fixture(scope="module")
def l9_card(tmp_path_factory):
    if (not all((INPUTS / name).is_file() for name in CARD_FILES)
            or not RESTO.is_file()):
        pytest.skip("no ORCA2 inputs")
    root = _root(tmp_path_factory.mktemp("l9"), G9, "l9")
    (root / "namelist_cfg").write_text(L9_DECK.render_text(
        (root / "namelist_cfg").read_text()))
    (root / "resto.nc").symlink_to(RESTO)
    return recipe.build_orca2l9_eiv_krhs_card(root)


def test_l9_deck_is_l8_plus_only_complete_l3_eiv(tmp_path):
    rendered = tmp_path / "namelist_cfg"
    rendered.write_text(L9_DECK.render_text((G9 / "namelist_cfg").read_text()))
    values = _values(rendered)
    assert {key: values[key] for key in recipe._ORCA2L3_EIV_FIELDS} == (
        recipe._ORCA2L3_EIV_FIELDS)
    for key, value in recipe._ORCA2L1_BBL_FIELDS.items():
        assert values[key] == value
    assert values["namtra_ldf.ln_traldf_iso"] is True
    assert values["nambbc.ln_trabbc"] is True
    assert values["namtra_dmp.ln_tradmp"] is True
    with pytest.raises(ValueError):
        L9_DECK.render_text(rendered.read_text())


def test_l9_combined_record_reader_selftest(capsys):
    L9_RECORDS.selftest()
    assert "L9_RECORD_SELFTEST_OK" in capsys.readouterr().out


def test_l9_acquisition_is_operator_only_and_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    for token in ("--preflight-only", "--run", "--admit-existing",
                  "NEMO_OPERATOR_ACK", "L9_R57_OPERATOR", "smoke_2step",
                  "ref_100step", "twin_a", "twin_b", "twin_c",
                  'check_acting "$SMOKE"', "gfortran",
                  "L9_PREFLIGHT_7_OF_7_PASS"):
        assert token in text
    assert "trap 'rm -rf -- \"$scratch\"' EXIT" in text
    operational = "\n".join(line for line in text.splitlines()
                            if "grep -F" not in line)
    assert 'check_acting "$REF"' not in operational


def test_l9_pins_eiv_then_every_krhs_position_with_inexact_words():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_stage3_l8_krhs_walk,
        _nemo_stage3_l9_eiv_transport,
    )

    u_words = [float.fromhex(value) for value in (
        "0x1.64d6a9b31f28fp+31", "-0x1.921fb54442d18p+29")]
    v_words = [float.fromhex(value) for value in (
        "-0x1.3bd3cc9be45dep+28", "0x1.0c6f7a0b5ed8dp+30")]
    u, v = jax.jit(_nemo_stage3_l9_eiv_transport)(
        *(jnp.asarray([x], dtype=jnp.float64)
          for x in (u_words[0], v_words[0], u_words[1], v_words[1])))
    assert np.asarray(u).item().hex() == "0x1.004ebc620e749p+31"
    assert np.asarray(v).item().hex() == "0x1.7af50dc8cb82bp+29"

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
    _, _, boundaries = jax.jit(_nemo_stage3_l8_krhs_walk)(adv, adv, *groups)
    for (got_t, got_s), want in zip(boundaries, expected):
        want_bits = np.asarray([want], dtype=np.float64).view(np.uint64)
        np.testing.assert_array_equal(np.asarray(got_t).view(np.uint64), want_bits)
        np.testing.assert_array_equal(np.asarray(got_s).view(np.uint64), want_bits)

    # Each adjacent call move must alter its first displaced boundary.
    for index in range(len(groups) - 1):
        planted = list(groups)
        planted[index], planted[index + 1] = planted[index + 1], planted[index]
        _, _, moved = _nemo_stage3_l8_krhs_walk(adv, adv, *planted)
        assert any(np.any(np.asarray(got[0]).view(np.uint64)
                          != np.asarray(want[0]).view(np.uint64))
                   for got, want in zip(moved, boundaries))


def test_l9_selector_relaxes_l8_only_for_exact_l3(l9_card):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    cfg = l9_card.recipe.model_config
    assert cfg.rk3_stage3_l8_nemo_order is True
    assert cfg.rk3_stage3_l9_eiv_nemo_order is True
    assert recipe._orca2l3_eiv_selected(cfg.gm_redi)
    assert recipe._orca2g1_selector_failures(l9_card) == []
    LatLonCGridOceanModel._validate_config(cfg)
    LatLonCGridOceanModel._check_l3_krhs_route(cfg)

    with pytest.raises(ValueError, match="requires.*Krhs tracer route"):
        LatLonCGridOceanModel._check_l3_krhs_route(
            cfg._replace(rk3_stage3_tracer_krhs_nemo_order=False))

    plants = (
        cfg._replace(rk3_stage3_l9_eiv_nemo_order=False),
        cfg._replace(rk3_stage3_l8_nemo_order=False),
        cfg._replace(rk3_stage3_tracer_krhs_nemo_order=False),
        cfg._replace(gm_redi=cfg.gm_redi._replace(
            gm_bolus_kappa_face_average=False)),
    )
    for planted in plants:
        with pytest.raises(ValueError):
            LatLonCGridOceanModel._validate_config(planted)


def test_l9_order_helpers_are_jitted_and_gradient_finite():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_stage3_l8_krhs_walk,
        _nemo_stage3_l9_eiv_transport,
    )

    def scalar(x):
        u, v = _nemo_stage3_l9_eiv_transport(x, 3.1, -0.7, x * 0.2)
        groups = tuple(((jnp.asarray([value]),), (jnp.asarray([value]),))
                       for value in (0.2, -3.0, 7.0, 11.0, -13.0, x))
        t, s, _ = _nemo_stage3_l8_krhs_walk(
            jnp.asarray([u]), jnp.asarray([v]), *groups)
        return t[0] + s[0]

    x = jnp.asarray(0.1, dtype=jnp.float64)
    ordinary = scalar(x)
    value, grad = jax.jit(jax.value_and_grad(scalar))(x)
    np.testing.assert_array_equal(
        np.asarray([value], dtype=np.float64).view(np.uint64),
        np.asarray([ordinary], dtype=np.float64).view(np.uint64))
    assert np.isfinite(np.asarray(grad))


def test_l9_live_eiv_executes_before_l8_finalizer(l9_card, monkeypatch):
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as cgrid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    state = l9_card.recipe.initial_state
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
    calls = []
    original_eiv = cgrid._nemo_stage3_l9_eiv_transport
    original_walk = cgrid._nemo_stage3_l8_krhs_walk

    def spy_eiv(*args):
        calls.append("eiv")
        return original_eiv(*args)

    def spy_walk(*args):
        calls.append("krhs")
        return original_walk(*args)

    monkeypatch.setattr(cgrid, "_nemo_stage3_l9_eiv_transport", spy_eiv)
    monkeypatch.setattr(cgrid, "_nemo_stage3_l8_krhs_walk", spy_walk)
    model = LatLonCGridOceanModel(
        l9_card.recipe.grid, l9_card.recipe.z_coord,
        l9_card.recipe.model_config)
    model.prime_step_caches(state)
    result = model._step_impl(
        state, dt=l9_card.dt_s, freshwater=freshwater,
        surface_forcing=forcing, sponge=l9_card.recipe.tracer_damping)

    assert calls == ["eiv", "krhs"]
    assert all(np.isfinite(np.asarray(getattr(result, name).data)).all()
               for name in ("T", "S"))

    # DECISION 147 remains an eager per-call refusal even after the model has
    # been constructed (and could already have a cached executable).
    model.config = model.config._replace(
        rk3_stage3_tracer_krhs_nemo_order=False)
    with pytest.raises(ValueError, match="requires.*Krhs tracer route"):
        model.step(
            state, dt=l9_card.dt_s, freshwater=freshwater,
            surface_forcing=forcing, sponge=l9_card.recipe.tracer_damping)

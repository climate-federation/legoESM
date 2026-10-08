"""The TSUNAMI card (round 2, RK3 build): transcription, explicit switches,
the key_RK3 deviation, j-periodicity as card data, the one-level rest step.

The oracle comparison waits for the acquisition record
(scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_tsunami/run_rk3.sh).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as CONSTANTS
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    TSUNAMI_DEVIATIONS,
    TSUNAMI_NAMELIST,
    TsunamiResolvedNamelist,
    TSUNAMI_UNMEASURED,
    build_tsunami_zco_card,
    tsunami_horizontal_coordinates,
    validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
)
from legoesm.ocean.vertical import create_z_star_from_thicknesses


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(previous)


@pytest.fixture(scope="module")
def card():
    return build_tsunami_zco_card()


def test_construction_and_shapes(card):
    s = card.recipe.initial_state
    assert card.case == "TSUNAMI-zco"
    assert (card.dt_s, card.n_steps, card.dummy_bottom_records) == (1000.0, 100, 1)
    # usrdef_nam.F90:113-117: NINT(2000/10)+1 = 201 each way, kpk = 2.
    assert s.eta.data.shape == (201, 201)
    assert s.T.data.shape == (201, 201, 1)
    assert s.u.data.shape == (201, 202, 1) and s.v.data.shape == (202, 201, 1)
    assert s.uu_b.data.shape == (201, 202) and s.vv_b.data.shape == (202, 201)
    assert np.all(np.asarray(s.T.data) == 20.0)
    assert np.all(np.asarray(s.S.data) == 30.0)
    for f in (s.u, s.v, s.uu_b, s.vv_b):
        assert not np.any(np.asarray(f.data))
    assert np.all(np.asarray(card.recipe.land_mask) == 1.0)


_NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
_CASE = _NEMO / "tests/TSUNAMI"
_STATS = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
          "testcases/nemo_testcase_full_statistics.py")
_UNIT_SUFFIXES = ("_m2_s", "_deg", "_km", "_m", "_s")
_ALIASES = {"ln_usr_sbc": "namsbc.ln_usr"}   # card label -> NEMO key
_NOT_IN_NAMELIST = {"key_RK3", "key_qco", "key_vco_1d", "nn_e_resolved",
                    "n_baro_upd"}


def _nemo_value(text):
    t = text.strip().lower()
    if t in (".true.", ".false."):
        return t == ".true."
    return float(t.replace("d", "e"))


def _resolved_nemo_namelist():
    spec = importlib.util.spec_from_file_location("tsunami_stats", _STATS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    resolved = mod.parse_namelist_values(_CASE / "EXPREF/namelist_ref")
    resolved.update(mod.parse_namelist_values(_CASE / "EXPREF/namelist_cfg"))
    return resolved


def _lookup(resolved, field):
    if field in _ALIASES:
        return _nemo_value(resolved[_ALIASES[field]])
    names = [field.lower()] + [field[: -len(x)].lower()
                               for x in _UNIT_SUFFIXES if field.endswith(x)]
    for name in names:
        vals = {v for k, v in resolved.items() if k.split(".", 1)[1] == name}
        if vals:
            assert len(vals) == 1, (field, vals)
            return _nemo_value(vals.pop())
    raise AssertionError(f"{field} not in NEMO's TSUNAMI namelists")


@pytest.mark.skipif(not _CASE.is_dir(), reason="NEMO 5.0.2 tree not present")
def test_explicit_switch_values_match_nemo_namelists():
    resolved = _resolved_nemo_namelist()
    for field in TsunamiResolvedNamelist._fields:
        if field in _NOT_IN_NAMELIST:
            continue
        assert getattr(TSUNAMI_NAMELIST, field) == _lookup(resolved, field), field
    keys = (_CASE / "cpp_TSUNAMI.fcm").read_text().split()
    # every built key equals the shipped one EXCEPT the recorded deviations
    differ = {k for k in ("key_RK3", "key_qco", "key_vco_1d")
              if getattr(TSUNAMI_NAMELIST, k) != (k in keys)}
    assert differ == {d[0] for d in TSUNAMI_DEVIATIONS}
    for field, shipped, built, _ in TSUNAMI_DEVIATIONS:
        assert (field in keys, getattr(TSUNAMI_NAMELIST, field)) == (shipped, built)


@pytest.mark.skipif(not _CASE.is_dir(), reason="NEMO 5.0.2 tree not present")
def test_rk3_stage_selector_is_the_module_constant():
    src = (_NEMO / "src/OCE/stprk3_stg.F90").read_text().splitlines()
    assert "n_baro_upd =  np_HYB" in src[43]
    assert TSUNAMI_NAMELIST.n_baro_upd == "np_HYB"


def test_switch_lookup_refuses_a_planted_drift():
    resolved = {"namdyn_spg.nn_e": "6", "namdom.rn_dt": "1000."}
    assert _lookup(resolved, "rn_Dt_s") == 1000.0
    with pytest.raises(AssertionError):
        _lookup(resolved, "rn_atfp")


def test_validator_refuses_an_spg_term_switched_on(card, monkeypatch):
    monkeypatch.setattr(recipe, "TSUNAMI_NAMELIST",
                        TSUNAMI_NAMELIST._replace(ln_tide=True))
    with pytest.raises(ValueError, match="ln_tide"):
        validate_nemo_testcase_card(card)


def test_explicit_switch_values(card):
    cfg = card.recipe.model_config
    assert cfg.barotropic.barotropic_time_filter == "nemo_boxcar1_ab3"
    assert cfg.barotropic.n_barotropic_substeps == 6
    assert cfg.barotropic.barotropic_coriolis == "een_metric"
    assert (cfg.A_v, cfg.K_v, cfg.K_h) == (1.2e-4, 1.2e-5, 0.0)
    assert cfg.lateral_viscosity.A_h == 0.0
    assert cfg.eos == "nemo_seos" and cfg.eos_nemo_seos.a0 == 1.6550e-1
    # dynspg_ts.F90:1223-1240 by hand: zcmax = sqrt(grav*H*(2/dx^2)).
    zcmax = math.sqrt(CONSTANTS.g * 100.0 * 2.0 / 1.0e8)
    assert math.ceil(1000.0 / 0.8 * zcmax) == 6


def test_f_plane(card):
    f0 = 2.0 * CONSTANTS.Omega * math.sin(math.pi / 180.0 * 38.5)
    src = tsunami_horizontal_coordinates()
    assert src["f0"] == f0
    assert np.all(np.asarray(card.recipe.grid.f_T) == src["f0"])
    assert (src["ii0"], src["ij0"]) == (40, 80)


def test_initial_ssh_against_formula_at_three_cells(card):
    eta = np.asarray(card.recipe.initial_state.eta.data)
    # Interior extremes of zti, ztj are 161 and 121 cells (i = j = 201).
    zmax = math.sqrt((10.0 * 161) ** 2 + (10.0 * 121) ** 2) / 20.0
    cells = {(79, 39): 0.0, (79, 45): 60.0, (85, 43): math.hypot(40.0, 60.0)}
    for (j, i), dist in cells.items():
        want = 0.1 * math.cos(dist / zmax * math.pi * 0.5)
        assert eta[j, i] == pytest.approx(want, rel=1e-14, abs=0.0)
    assert eta[79, 39] == 0.1
    assert eta[0, 0] == 0.0
    assert np.count_nonzero(eta) == 325


def test_execution_gate_refuses_with_named_blockers(card):
    assert [b.split(":")[0] for b in TSUNAMI_UNMEASURED] == [
        "B4", "B4j", "B6", "B7"]
    with pytest.raises(ValueError, match="B4:periodic_seam"):
        validate_nemo_testcase_card_for_execution(card)


def test_j_periodicity_is_card_data(card):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        NEMOTestcaseCard, build_nemo_testcase_card)
    assert card.j_periodic is True and TSUNAMI_NAMELIST.ln_Jperio is True
    assert NEMOTestcaseCard._field_defaults["j_periodic"] is False
    assert build_nemo_testcase_card("VORTEX-zco").j_periodic is False
    with pytest.raises(ValueError, match="ln_Jperio"):
        validate_nemo_testcase_card(card._replace(j_periodic=False))


def test_meridional_periodicity_scope_feeds_the_existing_y_wrap(card):
    from legoesm.grids import halo_latlon as hl
    import jax.numpy as jnp
    v = jnp.ones((5, 4))
    assert hl.get_meridionally_periodic() is False
    assert np.asarray(hl.zero_polar_lat_ends(v))[[0, -1]].sum() == 0.0
    with hl.meridional_periodicity(card.j_periodic):
        assert hl.get_meridionally_periodic() is True
        assert np.all(np.asarray(hl.zero_polar_lat_ends(v)) == 1.0)
    assert hl.get_meridionally_periodic() is False
    with pytest.raises(RuntimeError):
        with hl.meridional_periodicity(True):
            raise RuntimeError("restore on error")
    assert hl.get_meridionally_periodic() is False


def _step_fields(card, eta):
    """One model step of the card from (eta, the card's rest u/v/T/S)."""
    import jax.numpy as jnp
    from legoesm.grids.halo_latlon import meridional_periodicity
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    r = card.recipe
    s0 = r.initial_state._replace(
        eta=r.initial_state.eta.replace(data=jnp.asarray(eta)))
    with meridional_periodicity(card.j_periodic):
        s1 = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config).step(
            s0, card.dt_s)
    return {f: (np.asarray(getattr(s0, f).data), np.asarray(getattr(s1, f).data))
            for f in ("u", "v", "T", "S", "eta", "w", "uu_b", "vv_b")}


def test_one_wet_level_rest_state_does_not_move_bit_for_bit(card):
    """B5: one wet level, zero forcing, flat ssh -> nothing moves, bitwise.

    Non-vacuity: the SAME step from the card's own ssh bump moves eta, so the
    comparison can fail.
    """
    assert card.recipe.z_coord.n_levels == 1
    eta0 = np.asarray(card.recipe.initial_state.eta.data)
    rest = _step_fields(card, np.zeros_like(eta0))
    for name, (a, b) in rest.items():
        assert a.shape == b.shape and a.tobytes() == b.tobytes(), name
    moved = _step_fields(card, eta0)
    assert moved["eta"][0].tobytes() != moved["eta"][1].tobytes()


def test_validator_refuses_planted_drift(card):
    cfg = card.recipe.model_config
    bad_cfg = cfg._replace(barotropic=cfg.barotropic._replace(
        n_barotropic_substeps=30))
    bad = card._replace(recipe=card.recipe._replace(model_config=bad_cfg))
    with pytest.raises(ValueError, match="filter/substeps"):
        validate_nemo_testcase_card(bad)
    with pytest.raises(ValueError, match="TSUNAMI_UNMEASURED"):
        validate_nemo_testcase_card(card._replace(unmeasured_features=()))


def test_single_level_is_opt_in_only():
    with pytest.raises(ValueError, match=">= 2 thicknesses"):
        create_z_star_from_thicknesses(np.array([100.0]))
    z = create_z_star_from_thicknesses(np.array([100.0]), allow_single_level=True)
    assert z.n_levels == 1


def _roll_state(fields, sj, si):
    out = {}
    for name, f in fields.items():
        if name in ("uu_b", "u"):      # ni+1 faces: roll the distinct ni, re-close
            g = np.roll(f[:, :-1], (sj, si), axis=(0, 1))
            out[name] = np.concatenate([g, g[:, :1]], axis=1)
        elif name in ("vv_b", "v"):
            g = np.roll(f[:-1], (sj, si), axis=(0, 1))
            out[name] = np.concatenate([g, g[:1]], axis=0)
        else:
            out[name] = np.roll(f, (sj, si), axis=(0, 1))
    return out


def _seam_equivariance_unequal(card, seam_shift):
    """Cells where step(roll(bump on seam)) != roll(step(bump on seam))."""
    eta0 = np.asarray(card.recipe.initial_state.eta.data)  # centre (79, 39)
    sj, si = seam_shift
    on_seam = {k: b for k, (_, b) in _step_fields(
        card, np.roll(eta0, (sj, si), axis=(0, 1))).items()}
    moved = {k: b for k, (_, b) in _step_fields(
        card, np.roll(eta0, (sj + 37, si + 61), axis=(0, 1))).items()}
    want = _roll_state(on_seam, 37, 61)
    return {k: int(np.count_nonzero(moved[k] != want[k]))
            for k in ("eta", "uu_b", "vv_b")}


def test_i_seam_is_translation_equivariant_bit_for_bit(card):
    """B4, i-seam: the bump straddling the periodic i-seam steps exactly as
    the same bump moved 61 cells into the interior."""
    assert _seam_equivariance_unequal(card, (0, -39)) == {
        "eta": 0, "uu_b": 0, "vv_b": 0}


@pytest.mark.xfail(strict=True, reason=(
    "B4j: the card's step walls the j-seam; the existing y-wrap scope does "
    "not reach its barotropic path (round-2 probe: 402 ssh cells unequal, "
    "same count with the scope off)"))
def test_j_seam_is_translation_equivariant_bit_for_bit(card):
    assert _seam_equivariance_unequal(card, (-79, 0)) == {
        "eta": 0, "uu_b": 0, "vv_b": 0}

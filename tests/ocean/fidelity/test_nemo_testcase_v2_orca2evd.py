"""Vertical-mixing lane V2: certified V1 plus ORCA2's enhanced vertical diffusion."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES, CASES, INPUTS, ORCA2_DECK, V1, _differs, _flat, _root, _values,
)

V2 = CASES / "nemo_testcase_v2_orca2evd"
RUN = V2 / "run.sh"
EVD_KEYS = {"namzdf.ln_zdfevd": True, "namzdf.nn_evdm": 0, "namzdf.rn_evd": 100.0}
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_v2_deck_is_v1_plus_orca2s_evd_lines_only():
    v1, v2 = _values(V1 / "namelist_cfg"), _values(V2 / "namelist_cfg")
    changed = {k: (v1.get(k), v2.get(k)) for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}
    assert changed == {"namzdf.ln_zdfevd": (False, True),
                       "namzdf.nn_evdm": (None, 0), "namzdf.rn_evd": (None, 100.0)}
    assert (V1 / "namelist_ice_cfg").read_bytes() == (V2 / "namelist_ice_cfg").read_bytes()
    assert recipe._ORCA2V2_EVD_FIELDS == EVD_KEYS


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_v2_evd_lines_are_orca2s_own():
    orca2 = _values(ORCA2_DECK)
    assert {k: orca2[k] for k in EVD_KEYS} == EVD_KEYS


def test_reader_refuses_evd_on_the_wrong_rung_and_a_planted_line(tmp_path):
    with pytest.raises(ValueError, match="ln_zdfevd"):
        recipe.read_orca2g1_deck(V2, "ORCA2V1-tke")
    with pytest.raises(ValueError, match="ln_zdfevd"):
        recipe.read_orca2g1_deck(V1, "ORCA2V2-evd")
    planted = tmp_path / "planted"
    planted.mkdir()
    text = (V2 / "namelist_cfg").read_text()
    assert text.count("nn_evdm  =    0") == 1
    (planted / "namelist_cfg").write_text(text.replace("nn_evdm  =    0", "nn_evdm  =    1"))
    with pytest.raises(ValueError, match="nn_evdm"):
        recipe.read_orca2g1_deck(planted, "ORCA2V2-evd")
    assert recipe.read_orca2g1_deck(V2, "ORCA2V2-evd") == recipe.read_orca2g1_deck(
        V1, "ORCA2V1-tke")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("v2")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        v1 = recipe.build_orca2v1_tke_card(_root(base, V1))
        v2 = recipe.build_orca2v2_evd_card(_root(base, V2))
    finally:
        set_policy(before)
    return v1, v2


@needs_inputs
def test_v2_card_is_v1_plus_the_evd_selection_only(cards):
    v1, v2 = cards
    a, b = _flat(v1.recipe.model_config), _flat(v2.recipe.model_config)
    moved = {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    assert moved == {"physics.convection.scheme",
                     "physics.convection.enhanced_diffusion.nu_conv",
                     "physics.convection.enhanced_diffusion.trigger_step_entry_bn2"}
    conv = v2.recipe.model_config.physics.convection
    ed = conv.enhanced_diffusion
    # zdfevd.f90:107-110 avt <- rn_evd where MIN(rn2,rn2b) <= -1e-12; :121 nn_evdm=0
    assert (conv.scheme, ed.K_conv, ed.nu_conv, ed.K_bg, ed.nu_bg, ed.n2_threshold,
            ed.two_level_trigger, ed.evd_composition, ed.n2_mode, ed.n2_eos_form,
            ed.evd_n2_time_level, ed.trigger_step_entry_bn2) == (
        "enhanced_diffusion", 100.0, 0.0, 0.0, 0.0, -1e-12, True, "nemo_replace",
        "nemo_bn2", "eos80", "nemo_now_before", True)
    assert v1.recipe.model_config.physics.convection.scheme == "none"
    s1, s2 = v1.recipe.initial_state, v2.recipe.initial_state
    assert not [n for n in s1._fields if _differs(
        getattr(getattr(s1, n), "data", getattr(s1, n)),
        getattr(getattr(s2, n), "data", getattr(s2, n)))]
    assert (v2.dt_s, v2.n_steps, v2.unmeasured_features) == (
        v1.dt_s, v1.n_steps, v1.unmeasured_features)


@needs_inputs
def test_v2_selector_fires_on_each_planted_evd_field(cards):
    v1, v2 = cards
    assert recipe._orca2g1_selector_failures(v2) == []
    assert recipe._orca2g1_selector_failures(v1) == []
    cfg = v2.recipe.model_config
    conv = cfg.physics.convection
    for plant in ({"nu_conv": 100.0}, {"K_conv": 1.0}, {"two_level_trigger": False},
                  {"n2_threshold": 0.0}, {"evd_composition": "additive"},
                  {"evd_n2_time_level": "solver_state"},
                  {"trigger_step_entry_bn2": False}):
        bad = conv._replace(enhanced_diffusion=conv.enhanced_diffusion._replace(**plant))
        card = v2._replace(recipe=v2.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(convection=bad))))
        assert recipe._orca2g1_selector_failures(card) == ["ln_zdfevd"], plant
    card = v2._replace(recipe=v2.recipe._replace(model_config=cfg._replace(
        physics=cfg.physics._replace(convection=conv._replace(scheme="none")))))
    assert recipe._orca2g1_selector_failures(card) == ["ln_zdfevd"]


def test_run_script_reuses_v1_binaries_unbuilt_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    assert "ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/bin/nemo.exe" in text
    assert "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe" in text
    assert text.index('run_one "$REC_BIN" "$SMOKE"') < text.index('run_one "$REF_BIN" "$REF" 100')
    assert "ln_zdfevd = +T" in text and "nn_evdm += +0" in text
    out = subprocess.run(["bash", str(RUN), "--bogus"], capture_output=True)
    assert out.returncode == 64


V2_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_vmix_rounds/round12/oracle_v2_r12")
V1_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_vmix_rounds/round1/oracle_v1_r1")
V2_TWINS = [V2_RECORD / t for t in ("twin_a", "twin_b", "twin_c")]


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
@pytest.mark.skipif(not all(t.is_dir() for t in V2_TWINS)
                    or not (V1_RECORD / "twin_b").is_dir(),
                    reason="the operator's V1/V2 records are not on this machine")
def test_v2_independent_kt1_2_every_field_bit_for_bit_and_evd_is_not_inert():
    import netCDF4
    import numpy as np

    # control: EVD moves NEMO's kt-2 T, so matching V2's record is not matching V1's
    kt2 = "ORCA2G1_00000002_restart.nc"
    with netCDF4.Dataset(V1_RECORD / "twin_b" / kt2) as a, \
            netCDF4.Dataset(V2_RECORD / "twin_b" / kt2) as b:
        assert np.count_nonzero(np.asarray(a["tn"][:]) != np.asarray(b["tn"][:]))
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "v2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    out = ladder.run(V2_TWINS[0], arm="independent", kt_from=1, kt_to=2,
                     restart_roots=V2_TWINS, allow_dirty=True, case="ORCA2V2-evd")
    assert [k["kt"] for k in out["per_kt"]] == [1, 2]
    levels = out["per_kt"][1]["all_levels"]
    assert {"en", "avm_k", "avt_k", "dissl", "avm_k_surface"} <= set(levels or {})
    for k in out["per_kt"]:
        bad = [r["name"] for r in k["rows"] if r["n_unequal"]]
        for name, lv in (k["all_levels"] or {}).items():
            s = lv.get("summary")
            if (lv["n_unequal"] if s is None else
                    s["full"]["n_unequal"] or s["partial"]["n_unequal"]
                    or s["dry_n_unequal"]):
                bad.append(name)
        assert not bad, f"kt {k['kt']} unequal: {bad}"

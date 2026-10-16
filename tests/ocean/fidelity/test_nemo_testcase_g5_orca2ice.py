"""Grid-ladder G5: certified G4 plus ORCA2's exact SI3 selection."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G4 = CASES / "nemo_testcase_g4_orca2sbc"
G5 = CASES / "nemo_testcase_g5_orca2ice"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
ORCA2 = NEMO / "cfgs/ORCA2_OMIP_L4/EXP00"
R5 = NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY"
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")
needs_nemo = pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _values(path):
    return recipe._orca2g1_namelist_values(path)


def _root(tmp_path, rung):
    root = tmp_path / rung.name
    root.mkdir()
    shutil.copy(rung / "namelist_cfg", root / "namelist_cfg")
    if (rung / "namelist_ice_cfg").is_file():
        shutil.copy(rung / "namelist_ice_cfg", root / "namelist_ice_cfg")
    (root / "domain_cfg.nc").symlink_to(INPUTS / CARD_FILES[0])
    for name in CARD_FILES[1:]:
        (root / name).symlink_to(INPUTS / name)
    return root


def test_g5_deck_is_g4_plus_nn_ice_only():
    g4, g5 = _values(G4 / "namelist_cfg"), _values(G5 / "namelist_cfg")
    changed = {key: (g4.get(key), g5.get(key)) for key in set(g4) | set(g5)
               if g4.get(key) != g5.get(key)}
    assert changed == {"namsbc.nn_ice": (0, 2)}
    assert hashlib.sha256((G5 / "namelist_ice_cfg").read_bytes()).hexdigest() == (
        recipe._ORCA2G5_ICE_NAMELIST_SHA256)


@needs_nemo
def test_g5_resolves_as_round_defined_omt0_with_exact_orca2_si3():
    """The brief's OMT-0 target is G4's five-OFF program + forcing + SI3."""
    g4, g5 = _values(G4 / "namelist_cfg"), _values(G5 / "namelist_cfg")
    target = dict(g4)
    target["namsbc.nn_ice"] = _values(ORCA2 / "namelist_cfg")["namsbc.nn_ice"]
    assert target == g5                         # resolved-deck diff is empty
    assert (G5 / "namelist_ice_cfg").read_bytes() == (ORCA2 / "namelist_ice_cfg").read_bytes()
    # ORCA2 states empty SI3 overlay groups, so every SI3 value resolves from
    # the shared reference without a G5-only assignment.
    assert _values(G5 / "namelist_ice_cfg") == {}
    assert len(_values(NEMO / "cfgs/SHARED/namelist_ice_ref")) == 185


def test_ice_contract_refuses_planted_selection_and_overlay(tmp_path):
    root = _root(tmp_path, G5)
    assert recipe.orca2g_ice_failures(root, "ORCA2G5-ice") == []
    text = (root / "namelist_cfg").read_text()
    (root / "namelist_cfg").write_text(text.replace("nn_ice      = 2", "nn_ice      = 0"))
    with pytest.raises(ValueError, match="nn_ice"):
        recipe.build_orca2g5_ice_card(root)
    (root / "namelist_cfg").write_text(text)
    (root / "namelist_ice_cfg").write_text(
        (root / "namelist_ice_cfg").read_text().replace("&nampar", "&nampar\n   jpl = 4", 1))
    assert recipe.orca2g_ice_failures(root, "ORCA2G5-ice") == ["namelist_ice_cfg"]
    with pytest.raises(ValueError, match="namelist_ice_cfg"):
        recipe.build_orca2g5_ice_card(root)


@needs_inputs
def test_g5_card_is_g4_plus_si3_entry_and_record_route(tmp_path):
    g4_root, g5_root = _root(tmp_path, G4), _root(tmp_path, G5)
    g4 = recipe.build_orca2g4_sbc_card(g4_root)
    g5 = recipe.build_nemo_testcase_card("ORCA2G5-ice", deck_root=g5_root)
    assert g5.recipe.model_config == g4.recipe.model_config
    for field in ("T", "S", "u", "v", "uu_b", "vv_b"):
        assert np.array_equal(np.asarray(getattr(g5.recipe.initial_state, field).data),
                              np.asarray(getattr(g4.recipe.initial_state, field).data))
    wet = np.asarray(g5.recipe.land_mask) > 0.5
    delta = (np.asarray(g4.recipe.initial_state.eta.data)
             - np.asarray(g5.recipe.initial_state.eta.data))
    np.testing.assert_allclose(delta[wet], delta[wet][0], rtol=0.0,
                               atol=np.finfo(np.float64).eps)
    assert delta[wet][0] == pytest.approx(0.09343505872684969, abs=1.0e-16)
    assert not np.any(delta[~wet])
    assert (g5.surface_boundary_condition, g5.surface_input_operator) == (
        "nemo_post_si3_records", "nemo_ocean_surface_input_v1")
    assert g5.unmeasured_features == recipe.ORCA2G5_UNMEASURED
    assert "implicit_ice_ocean_top_drag_rCdU_ice" in g5.unmeasured_features
    assert g5.icebergs_enabled is False and g5.iceberg_inputs == ()
    assert recipe._orca2g1_selector_failures(g5) == []
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g5, g5_root)


@needs_nemo
def test_reused_writer_is_post_sbc_and_records_every_ice_exchange():
    source = (R5 / "MY_SRC/stprk3.F90").read_text()
    call = source.index("CALL l4_dump_ocean_surface_input( kstp, Nbb )")
    assert source.rfind("CALL sbc", 0, call) >= 0
    body = source[source.index("SUBROUTINE l4_dump_ocean_surface_input"):]
    for field in ("utau", "vtau", "qsr", "qns", "emp", "sfx", "fwfice",
                  "rnf", "fr_i", "snwice_mass", "rCdU_ice", "rnf_tsc"):
        assert re.search(rf"\b{field}\b", body), field
    assert "STATUS='NEW'" in body
    cpp = (R5 / "cpp_ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY.fcm").read_text()
    assert "key_si3" in cpp


def test_run_script_is_fail_closed_smoke_first_and_never_uses_step1_restart():
    run = G5 / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    driver = text[text.index("preflight\nif"):]
    assert driver.index('if [[ "$mode" == --preflight-only ]]') < driver.index("build_target")
    executions = re.findall(r'^run_one "\$(\w+)"', text, re.M)
    assert executions == ["SMOKE", "REF", "TWIN_A", "TWIN_B", "TWIN_C"]
    assert "steps=$((steps - steps % FSBC))" in text
    assert 'for step in $(seq 2 "$steps")' in text
    assert "seq 1 \"$steps\"" not in text[text.index("for step in"):]
    assert "key_si3" in text and "rCdU_ice" in text and "fwfice" in text
    assert "expected_boundary=kt11_after_10_completed" in text
    # Build is isolated under cfgs/MY_SRC; the shared NEMO src tree is never a target.
    assert not re.search(r'cp .*\$NEMO_ROOT/src|patch .*\$NEMO_ROOT/src', text)


def test_source_controls_are_real():
    # Omitting the selected SI3 case must still make the old reader refuse it.
    with pytest.raises(ValueError, match="nn_ice"):
        recipe.read_orca2g1_deck(G5)
    # A non-G5 card must refuse an otherwise valid SI3 overlay rather than ignore it.
    assert "namsbc.nn_ice" in recipe.orca2g_ice_failures(G5, "ORCA2G4-sbc")

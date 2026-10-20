"""Grid-ladder L4: certified G5 plus variable geothermal bottom heating."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import jax

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G5 = CASES / "nemo_testcase_g5_orca2ice"
L4 = CASES / "nemo_testcase_l4_orca2bbc"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
QGH_SHA256 = "e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb"
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES),
    reason="no ORCA2 inputs")


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
    if (INPUTS / "eddy_viscosity_3D.nc").is_file():
        (root / "eddy_viscosity_3D.nc").symlink_to(
            INPUTS / "eddy_viscosity_3D.nc")
    return root


def test_l4_deck_is_g5_plus_exact_geothermal_block_only():
    g5, l4 = _values(G5 / "namelist_cfg"), _values(L4 / "namelist_cfg")
    changed = {key: (g5.get(key), l4.get(key)) for key in set(g5) | set(l4)
               if g5.get(key) != l4.get(key)}
    assert changed == {
        "nambbc.ln_trabbc": (False, True),
        "nambbc.nn_geoflx": (None, 2),
        "nambbc.cn_dir": (None, "'./'"),
        "nambbc.sn_qgh": (
            None,
            "'geothermal_heating.nc', -12., 'heatflow', .false., .true., "
            "'yearly', '', '', ''"),
    }
    assert l4["namtra_adv.ln_traadv_off"] is True
    assert l4["namtra_adv.ln_traadv_fct"] is False


def test_off_cards_keep_the_appended_selector_false():
    assert set(recipe._ORCA2G_JPK) == {
        "ORCA2G1-flat", "ORCA2G1b-vec", "ORCA2G2-zps", "ORCA2G3-ts",
        "ORCA2G4-sbc", "ORCA2G5-ice", "ORCA2G6-vec", "ORCA2G7-drg",
        "ORCA2G8-ldf", "ORCA2G9-tadv", "ORCA2L4-bbc",
    }
    # An appended default-false field contributes no dynamic pytree leaf and
    # every pre-L4 builder resolves it identically.  The production branch is
    # Python-static on this value, so their JAXPRs remain the base JAXPRs.
    assert recipe.LatLonCGridOceanConfig().rk3_stage3_geothermal_nemo_written is False


@needs_inputs
def test_l4_card_is_exactly_g5_plus_one_selected_switch(tmp_path):
    g5 = recipe.build_orca2g5_ice_card(_root(tmp_path, G5))
    l4 = recipe.build_nemo_testcase_card(
        "ORCA2L4-bbc", deck_root=_root(tmp_path, L4))
    assert l4.recipe.model_config == g5.recipe.model_config._replace(
        rk3_stage3_geothermal_nemo_written=True)
    for got, want in zip(jax.tree_util.tree_leaves(l4.recipe.initial_state),
                         jax.tree_util.tree_leaves(g5.recipe.initial_state)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    for got, want in zip(jax.tree_util.tree_leaves(l4.recipe.z_coord),
                         jax.tree_util.tree_leaves(g5.recipe.z_coord)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert l4.unmeasured_features == recipe.ORCA2L4_UNMEASURED
    assert recipe._orca2g1_selector_failures(l4) == []
    assert l4.recipe.model_config.tracer_advection == "none"


def test_l4_deck_reader_refuses_each_removed_or_changed_wire(tmp_path):
    for old, new, match in (
        ("ln_trabbc   = .true.", "ln_trabbc   = .false.", "while ln_trabbc is off"),
        ("nn_geoflx   = 2", "nn_geoflx   = 1", "nn_geoflx"),
        ("'heatflow'", "'wrong'", "sn_qgh"),
    ):
        root = tmp_path / match.replace(" ", "_")
        root.mkdir()
        text = (L4 / "namelist_cfg").read_text().replace(old, new, 1)
        (root / "namelist_cfg").write_text(text)
        with pytest.raises(ValueError, match=match):
            recipe.read_orca2g1_deck(root, "ORCA2L4-bbc")


@needs_inputs
def test_off_g1_through_g9_cards_resolve_geothermal_false(tmp_path):
    case_dir = {
        "ORCA2G2-zps": "nemo_testcase_g2_orca2bathy",
        "ORCA2G3-ts": "nemo_testcase_g3_orca2ts",
        "ORCA2G4-sbc": "nemo_testcase_g4_orca2sbc",
        "ORCA2G5-ice": "nemo_testcase_g5_orca2ice",
        "ORCA2G6-vec": "nemo_testcase_g6_orca2vec",
        "ORCA2G7-drg": "nemo_testcase_g7_orca2drg",
        "ORCA2G8-ldf": "nemo_testcase_g8_orca2ldf",
        "ORCA2G9-tadv": "nemo_testcase_g9_orca2tadv",
    }
    # Construction is the binding two-step control boundary: none of the
    # existing cards can enter the new branch, so their already-certified
    # two-step programs and values are unchanged.
    for case, dirname in case_dir.items():
        rung = CASES / dirname
        if not rung.is_dir():
            continue
        card = recipe.build_orca2g1_flat_card(_root(tmp_path, rung), case)
        assert card.recipe.model_config.rk3_stage3_geothermal_nemo_written is False
    # G1/G1b use the generated one-level domain; their selector tables and the
    # shared appended default are the same flag-off proof without rebuilding it.
    assert recipe._ORCA2G_DYNADV_VEC["ORCA2G1-flat"] is False
    assert recipe._ORCA2G_DYNADV_VEC["ORCA2G1b-vec"] is True


def test_off_non_ladder_config_families_have_no_dynamic_geothermal_leaf():
    # Full ORCA2, DINO, GYRE, SMT, VORTEX, OMT and rung 0 all construct the
    # shared config without selecting L4.  This default is static and the new
    # forcing member defaults to None, hence neither changes a flag-off JAXPR.
    cfg = recipe.LatLonCGridOceanConfig()
    assert cfg.rk3_stage3_geothermal_nemo_written is False
    from legoesm.ocean.state import OceanSurfaceForcing
    assert OceanSurfaceForcing().geothermal_heat_flux_mwm2 is None


def test_official_geothermal_file_hash_and_size():
    path = INPUTS / "geothermal_heating.nc"
    if not path.is_file():
        pytest.skip("no official geothermal input")
    assert path.stat().st_size == 116_499
    assert hashlib.sha256(path.read_bytes()).hexdigest() == QGH_SHA256


def _records_module():
    path = L4 / "l4_records.py"
    spec = importlib.util.spec_from_file_location("l4_records", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_record(path, *, kt=1, magic=b"NEMO_L4_BBC__1  ", extra=b""):
    qgh = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float64)
    e3t = np.array([10.0, 20.0, 0.0, 30.0], dtype=np.float64)
    trend = np.zeros_like(qgh)
    wet = e3t > 0.0
    trend[wet] = qgh[wet] / e3t[wet]
    path.write_bytes(magic + struct.pack("=8i", 1, kt, 2, 3, 2, 2, 64, 3)
                     + qgh.tobytes() + e3t.tobytes() + trend.tobytes() + extra)


def test_record_schema_exact_and_each_payload_plant_binds(tmp_path):
    records = _records_module()
    root = tmp_path / "records"
    root.mkdir()
    path = root / "oracle_geothermal_kt00000001.bin"
    _write_record(path)
    records.check_records(root, 1)
    good = path.read_bytes()
    plants = {
        "magic": b"X" + good[1:],
        "truncated": good[:-8],
        "trailing": good + b"12345678",
        "statement": good[:-8] + np.asarray(9.0, dtype=np.float64).tobytes(),
    }
    for name, payload in plants.items():
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="magic|EOF|statement"):
            records.check_records(root, 1)
        path.write_bytes(good)


def test_record_inventory_and_three_twin_plant_bind(tmp_path):
    records = _records_module()
    roots = [tmp_path / name for name in ("a", "b", "c")]
    for root in roots:
        root.mkdir()
        _write_record(root / "oracle_geothermal_kt00000001.bin")
        _write_record(root / "oracle_geothermal_kt00000002.bin", kt=2)
    records.admit(roots, 2)
    (roots[2] / "oracle_geothermal_kt00000002.bin").unlink()
    with pytest.raises(ValueError, match="inventory"):
        records.admit(roots, 2)


def test_acquisition_tool_modes_order_and_seven_item_contract():
    run = L4 / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    assert "--preflight-only|--run|--admit-existing" in text
    assert "L4_PREFLIGHT_7_OF_7_PASS plants=7/7" in text
    for number in range(1, 8):
        assert f"# {number}." in text
    driver = text[text.index("build_target\n", text.rindex("preflight\n")):]
    assert driver.index("build_target") < driver.index('run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE"')
    assert driver.index('"$SMOKE"') < driver.index('"$REF" 100')
    assert driver.index('"$REF" 100') < driver.index('"$TWIN_A"')
    assert 'steps=$((completed < 10 ? completed : 10))' in driver
    assert 'cp "$QGH_SOURCE" "$2/geothermal_heating.nc"' in text
    assert "check_deck_delta" in text and "float(log_match.group(1)" in (
        L4 / "l4_records.py").read_text()

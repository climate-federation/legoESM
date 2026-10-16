"""Grid-ladder G5: certified G4 plus ORCA2's exact SI3 selection."""

from __future__ import annotations

import hashlib
import re
import shutil
import struct
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
P2 = NEMO / "cfgs/ORCA2_OMIP_L4_P2VBND_R2"
ICE_REF = NEMO / "cfgs/SHARED/namelist_ice_ref"
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
def test_g5_is_g4_plus_orca2_ice_and_is_not_decision103_omt0():
    g4, g5 = _values(G4 / "namelist_cfg"), _values(G5 / "namelist_cfg")
    changed = {key: (g4.get(key), g5.get(key)) for key in set(g4) | set(g5)
               if g4.get(key) != g5.get(key)}
    assert changed == {"namsbc.nn_ice": (0, 2)}
    # Decision 103's OMT-0 explicitly has no bulk forcing, sea ice, penetrating
    # shortwave, SSS restoring, runoff, or freshwater-budget correction.
    omt0 = {"namsbc.ln_blk": False, "namsbc.nn_ice": 0,
            "namsbc.ln_traqsr": False, "namsbc.ln_ssr": False,
            "namsbc.ln_rnf": False, "namsbc.nn_fwb": 0}
    assert {key: g5[key] for key in omt0} != omt0
    assert {key: g5[key] for key in omt0} == {
        "namsbc.ln_blk": True, "namsbc.nn_ice": 2,
        "namsbc.ln_traqsr": True, "namsbc.ln_ssr": True,
        "namsbc.ln_rnf": True, "namsbc.nn_fwb": 2,
    }
    assert (G5 / "namelist_ice_cfg").read_bytes() == (ORCA2 / "namelist_ice_cfg").read_bytes()
    # ORCA2 states empty SI3 overlay groups, so every SI3 value resolves from
    # the shared reference without a G5-only assignment.
    assert _values(G5 / "namelist_ice_cfg") == {}
    assert len(_values(ICE_REF)) == 185


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


def _writer_is_immediately_after_exact_sbc(source):
    statements = [line.split("!", 1)[0].strip() for line in source.splitlines()]
    statements = [line for line in statements if line]
    sbc = [i for i, line in enumerate(statements) if re.fullmatch(
        r"CALL\s+sbc\s*\(\s*kstp\s*,\s*Nbb\s*,\s*Nbb\s*\)", line, re.I)]
    writer = [i for i, line in enumerate(statements) if re.fullmatch(
        r"CALL\s+l4_dump_ocean_surface_input\s*\(\s*kstp\s*,\s*Nbb\s*\)",
        line, re.I)]
    return len(sbc) == len(writer) == 1 and writer[0] == sbc[0] + 1


@needs_nemo
def test_reused_writer_is_post_exact_sbc_and_records_every_ice_exchange():
    source = (R5 / "MY_SRC/stprk3.F90").read_text()
    assert _writer_is_immediately_after_exact_sbc(source)
    sbc_line = "                         CALL sbc        ( kstp, Nbb, Nbb )                ! Sea Boundary Condition (including sea-ice)"
    writer_line = "                         CALL l4_dump_ocean_surface_input( kstp, Nbb )     ! round-1 WRITE-only per-step acquisition"
    planted = source.replace(f"{sbc_line}\n{writer_line}", f"{writer_line}\n{sbc_line}", 1)
    assert planted != source
    assert not _writer_is_immediately_after_exact_sbc(planted)
    body = source[source.index("SUBROUTINE l4_dump_ocean_surface_input"):]
    for field in ("utau", "vtau", "qsr", "qns", "emp", "sfx", "fwfice",
                  "rnf", "fr_i", "snwice_mass", "rCdU_ice", "rnf_tsc"):
        assert re.search(rf"\b{field}\b", body), field
    assert "STATUS='NEW'" in body
    cpp = (R5 / "cpp_ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY.fcm").read_text()
    assert "key_si3" in cpp


@needs_nemo
def test_acquisition_preflight_rejects_writer_selector_mismatch():
    run = G5 / "run.sh"
    good = subprocess.run(
        ["bash", str(run), "--check-writer-contract",
         str(P2 / "MY_SRC/icedyn_adv_pra.F90"), str(ICE_REF),
         str(G5 / "namelist_ice_cfg")], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    bad = subprocess.run(
        ["bash", str(run), "--check-writer-contract",
         str(R5 / "MY_SRC/icedyn_adv_pra.F90"), str(ICE_REF),
         str(G5 / "namelist_ice_cfg")], capture_output=True, text=True)
    assert bad.returncode != 0
    assert "requires nn_icesal=2 and ponds off" in bad.stderr


def test_surface_payload_size_gate_refuses_truncation(tmp_path):
    run = G5 / "run.sh"
    path = tmp_path / "oracle_ocean_surface_input_kt00000001.bin"
    jpi, jpj, jpts, halo = 6, 6, 2, 1
    header = (1, 1, 1, jpi, jpj, jpts, 10, halo, 20, 12, 1, 2, 64)
    count = (20 * jpi * jpj + 12 * (jpi - 2 * halo) * (jpj - 2 * halo)
             + (jpi - 2) * (jpj - 2)
             + 2 * (jpi - 2 * halo) * (jpj - 2 * halo) * jpts)
    frame = b"NEMO_L4_SBCIN_1 " + struct.pack("=13i", *header) + np.zeros(
        count, dtype=np.float64).tobytes()
    path.write_bytes(frame)
    good = subprocess.run(
        ["bash", str(run), "--check-surface-payload", str(tmp_path), "1"],
        capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    path.write_bytes(frame[:-8])
    bad = subprocess.run(
        ["bash", str(run), "--check-surface-payload", str(tmp_path), "1"],
        capture_output=True, text=True)
    assert bad.returncode != 0
    assert "bad G5 surface payload size" in bad.stderr


def test_run_script_is_fail_closed_smoke_first_and_never_uses_step1_restart():
    run = G5 / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    driver = text[text.index("preflight\nif"):]
    assert driver.index('if [[ "$mode" == --preflight-only ]]') < driver.index("build_targets")
    executions = re.findall(r'^run_one "([^\"]+)" "\$(\w+)"', text, re.M)
    assert executions == [
        ("$TARGET/BLD/bin/nemo.exe", "SMOKE"),
        ("$REF_TARGET/BLD/bin/nemo.exe", "REF"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_A"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_B"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_C"),
    ]
    assert "steps=$((steps - steps % FSBC))" in text
    assert 'for step in $(seq 2 "$steps")' in text
    assert "seq 1 \"$steps\"" not in text[text.index("for step in"):]
    assert "key_si3" in text and "rCdU_ice" in text and "fwfice" in text
    assert "expected_boundary=kt11_after_10_completed" in text
    assert "path.stat().st_size != expected_bytes" in text
    assert "reference binary carries the surface writer" in text
    assert "! -name 'icedyn_adv_pra.F90'" in text and 'cp -a "$P2_PRA"' in text
    assert 'makenemo -n "$name" -m conda-scalarmath -y --clean' in text
    # Build is isolated under cfgs/MY_SRC; the shared NEMO src tree is never a target.
    assert not re.search(r'cp .*\$NEMO_ROOT/src|patch .*\$NEMO_ROOT/src', text)


def test_source_controls_are_real():
    # Omitting the selected SI3 case must still make the old reader refuse it.
    with pytest.raises(ValueError, match="nn_ice"):
        recipe.read_orca2g1_deck(G5)
    # A non-G5 card must refuse an otherwise valid SI3 overlay rather than ignore it.
    assert "namsbc.nn_ice" in recipe.orca2g_ice_failures(G5, "ORCA2G4-sbc")


# ---- round 32: the G5 record reader, admission and top-drag replay ---------
RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
              "round31/oracle_g5_r31")
needs_record = pytest.mark.skipif(not (RECORD / "twin_a").is_dir(), reason="no G5 record")


def _g5_records():
    import sys
    sys.path.insert(0, str(G5))
    import g5_records
    return g5_records


def _write_surface(path: Path, kt: int, nx: int, ny: int, fill) -> None:
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_phase2b_exchange_gate as schema)
    header = (1, kt, 1 if kt % 2 else 3, nx, ny, 2, 10, 2, 20, 12, 1, 2, 64)
    chunks = []
    shapes = {"full": (nx, ny), "reduced": (nx - 4, ny - 4), "halo1": (nx - 2, ny - 2),
              "reduced3d": (nx - 4, ny - 4, 2)}
    for name, kind in schema.FIELDS:
        chunks.append(np.asfortranarray(fill(name, shapes[kind])).ravel(order="F"))
    with path.open("wb") as fh:
        fh.write(schema.MAGIC.ljust(16).encode())
        fh.write(struct.pack("=13i", *header))
        np.concatenate(chunks).astype(np.float64).tofile(fh)


def test_g5_surface_reader_cuts_halos_and_transposes(tmp_path):
    g5 = _g5_records()
    nx, ny = 9, 8

    def fill(name, shape):        # value encodes (field, x, y) of the WRITTEN array
        idx = np.indices(shape[:2])
        base = 1000.0 * idx[0] + idx[1]
        return base[..., None].repeat(shape[2], -1) if len(shape) == 3 else base

    _write_surface(tmp_path / "oracle_ocean_surface_input_kt00000001.bin", 1, nx, ny, fill)
    rec = g5.surface(tmp_path, 1)
    # full fields lose two halo cells, halo1 one, reduced none; (y, x) order
    assert rec["utau"].shape == rec["taum"].shape == rec["rCdU_ice"].shape == (ny - 4, nx - 4)
    assert rec["utau"][0, 0] == 1000.0 * 2 + 2 and rec["utau"][1, 0] == 2 + 1 + 2000.0
    assert rec["taum"][0, 0] == 0.0 and rec["rCdU_ice"][0, 0] == 1000.0 + 1
    assert rec["rnf_tsc"].shape == (ny - 4, nx - 4, 2)
    full = g5.surface(tmp_path, 1, keep_halo=True)
    assert full["utau"].shape == (ny, nx) and full["rCdU_ice"].shape == (ny - 2, nx - 2)
    with pytest.raises(g5.Refuse):          # a frame for another step is refused
        (tmp_path / "oracle_ocean_surface_input_kt00000001.bin").rename(
            tmp_path / "oracle_ocean_surface_input_kt00000002.bin")
        g5.surface(tmp_path, 2)


@needs_record
def test_g5_admission_passes_and_its_passivity_plant_fires():
    g5 = _g5_records()
    res = g5.admit(RECORD)
    assert res["passivity"]["step"] == 10 and res["passivity"]["restart_fields"] == {
        "ORCA2G1_00000010_restart.nc": 38}
    assert res["twin_identical_files"] == 60
    with pytest.raises(g5.Refuse, match="PASSIVITY FAILED"):
        g5.admit(RECORD, "passivity")


@needs_record
def test_g5_top_drag_replay_is_exact_and_its_plant_fires():
    g5 = _g5_records()
    res = g5.top_drag(RECORD)
    for face, n in (("zCdU_u", 3519), ("zCdU_v", 3593)):
        assert res[face]["nemo_nonzero_faces"] == n
        assert res[face]["replay_vs_nemo_on_nemo_nonzero"]["unequal"] == 0
    with pytest.raises(g5.Refuse, match="plant fired"):
        g5.top_drag(RECORD, plant=True)

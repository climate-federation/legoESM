from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4.nemo_acquisition_extents import (
    DeckDimensions, ExtentError, fixture_include, validate_writer,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import (
    load_deck, read_record, score_deck,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_readset import derive_and_gate

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition"
PPSRC = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo")


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("o4_records")
    subprocess.run(["python3", str(HERE/"write_stub_records.py"), str(out), "--source-real",
                    str(ROOT/"scripts/validate/ocean_fidelity/testcases/nemo_si3_o7_orca2_sas_acquisition/ocean.output.excerpt"),
                    "--nontrivial-score-fixture"],
                   check=True, cwd=ROOT)
    return out


@pytest.fixture(scope="module")
def independent_records(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("o4_independent_records")
    subprocess.run(["python3", str(HERE/"write_stub_records.py"), str(out), "--source-real",
                    str(ROOT/"scripts/validate/ocean_fidelity/testcases/nemo_si3_o7_orca2_sas_acquisition/ocean.output.excerpt")],
                   check=True, cwd=ROOT)
    return out


def test_o4_record_schema_and_reader_plants(records: Path) -> None:
    deck = load_deck(records)
    assert len(deck) == 18
    path = records / "o4_kind01_step0001_rank0000.bin"
    for plant in ("header", "truncation", "step"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            read_record(path, plant)


def test_o4_production_scorer_and_plants(records: Path) -> None:
    assert score_deck(records)["status"] == "PASS"
    for plant in ("one-bit", "one-bit-zapsmall", "one-bit-lbc", "one-bit-step",
                  "invert-sweep", "wrong-fold-sign", "omit-subcycle-lbc"):
        with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
            score_deck(records, plant)


def test_o4_stub_records_are_independent_and_score(independent_records: Path) -> None:
    assert score_deck(independent_records)["status"] == "PASS"
    stub = (HERE/"stub_nemo.sh").read_text()
    assert "--nontrivial-score-fixture" not in stub
    assert "write_stub_records.py" not in stub  # path is supplied explicitly by run.sh


def test_o4_shared_extent_gate_and_plants() -> None:
    deck = DeckDimensions(184, 152, 31, 2)
    assert validate_writer(PPSRC, HERE/"o4_prather_writer.F90", deck)["arrays_checked"] == 104
    for plant in ("missing-allocation", "writer-mode", "dummy-actual"):
        with pytest.raises(ExtentError, match="STATUS PLANT-FIRED"):
            validate_writer(PPSRC, HERE/"o4_prather_writer.F90", deck, plant)
    with pytest.raises(ExtentError, match="STATUS PLANT-FIRED"):
        fixture_include(PPSRC, HERE/"o4_prather_writer.F90", deck, "fixture-allocation")


def test_o4_shared_readset_registration_and_plant() -> None:
    assert derive_and_gate(PPSRC)["status"] == "PASS_SOURCE_DERIVED_READ_SETS"
    with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
        derive_and_gate(PPSRC, "drop-sxice")


def test_o4_patch_and_driver_are_read_only_preflight_tools() -> None:
    source_root = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/ICE")
    groups = {
        "icedyn_adv_pra": ("icedyn_adv_pra_o4.patch", "icedyn_adv_pra_o4_use.patch", "icedyn_adv_pra_o4_capture.patch"),
        "icedyn_adv": ("icedyn_adv_o4.patch",), "icedyn": ("icedyn_o4.patch",),
        "icestp": ("icestp_o4.patch", "icestp_o4_capture.patch"),
    }
    for unit, patches in groups.items():
        for patch in patches:
            subprocess.run(["patch", "--dry-run", "--batch", "--fuzz=0",
                            str(source_root/f"{unit}.F90"), str(HERE/patch)], check=True)
    subprocess.run(["bash", "-n", str(HERE/"run.sh")], check=True)
    text = (HERE/"run.sh").read_text()
    assert "--preflight-only" in text and "write_manifest" in text
    assert "run.user.stdout.log" in text and "run.user.stderr.log" in text


def test_o4_receipt_digests_rehash_exact_files() -> None:
    receipt = (ROOT/"docs/ice/fidelity/si3_r25_o4_acquisition_receipt.md").read_text()
    rows = re.findall(r"^SHA256 ([^ ]+) ([0-9a-f]{64})$", receipt, re.MULTILINE)
    assert len(rows) >= 6
    for relative, expected in rows:
        path = ROOT/relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative

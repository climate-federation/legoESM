"""Controls for the operator-only SI3 O7 NEMO acquisition tool."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "scripts/validate/ocean_fidelity/testcases"
ACQ = TOOLS / "nemo_si3_o7_orca2_sas_acquisition"
SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_o7_gate", TOOLS / "nemo_si3_o7_orca2_sas_gate.py"
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_o7_schema_covers_every_preregistered_output_and_intermediate() -> None:
    assert set(gate.KINDS) == set(range(1, 10))
    assert [name for name, _ in gate.KINDS[2]] == [
        "wndm_ice",
        "zsipt",
        "Cd_ice",
        "Ch_ice",
        "Ce_ice",
        "theta_zu_i",
        "q_zu_i",
        "utau_ice",
        "vtau_ice",
    ]
    assert [name for name, _ in gate.KINDS[7]] == [
        "qsr_ice",
        "qns_ice",
        "dqns_ice",
        "qla_ice",
        "dqla_ice",
        "evap_ice",
        "devap_ice",
        "qevap_ice",
        "qtr_ice_top",
        "zsq",
        "zsipt",
        "z_qlw",
        "z_dqlw",
        "z_qsb",
        "z_dqsb",
        "tprecip",
        "sprecip",
        "emp_oce",
        "emp_ice",
        "emp_tot",
        "qemp_oce",
        "qemp_ice",
        "qns_tot",
        "qsr_tot",
        "qprec_ice",
        "zcptrain",
        "zcptsnw",
        "zcptn",
        "zevap",
        "zsnw",
    ]
    assert dict(gate.KINDS[3]) == {"utau_ice": "F2", "vtau_ice": "F2"}
    assert dict(gate.KINDS[8])["u_ice"] == "F2"
    assert dict(gate.KINDS[8])["smask0"] == "I2"


@pytest.mark.parametrize(
    "plant",
    [
        "header",
        "truncation",
        "extent",
        "checksum",
        "time-level",
        "missing-frame",
        "wet-1ulp",
        "absolute-path",
    ],
)
def test_o7_reader_controls_fire(plant: str) -> None:
    with pytest.raises(ValueError, match="STATUS PLANT-FIRED"):
        gate.run_self_test(plant)


def test_o7_reader_retains_signed_zero_and_declared_fortran_extents(tmp_path: Path) -> None:
    path = tmp_path / "o7_kind01_step0001_rank0000.bin"
    gate._write_synthetic_record(path, 1)
    raw = bytearray(path.read_bytes())
    struct.pack_into("<Q", raw, gate.HEADER_BYTES, 1 << 63)
    path.write_bytes(raw)
    record = gate.read_o7_record(path)
    assert record["fields"]["wndi"].shape == (8, 6)
    assert record["fields"]["wndi"][0, 0] == 0.0
    assert np.signbit(record["fields"]["wndi"][0, 0])


def test_o7_patches_mark_every_insertion_for_real_ppsrc_stripping() -> None:
    for unit in gate.OBSERVER_UNITS:
        lines = (ACQ / f"{unit}_o7.patch").read_text().splitlines()
        inserted = [
            line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")
        ]
        inside = False
        blocks = 0
        for line in inserted:
            if line.strip() == "! O7_OBSERVER_BEGIN":
                assert not inside
                inside = True
                blocks += 1
            elif line.strip() == "! O7_OBSERVER_END":
                assert inside
                inside = False
            else:
                assert inside, f"{unit}: insertion outside observer markers: {line!r}"
        assert not inside
        assert blocks >= 2


def test_o7_writer_uses_declared_full_or_nemo_filled_ranges() -> None:
    text = (ACQ / "o7_si3_writer.F90").read_text()
    assert "a(nn_hls+1:jpi-nn_hls,nn_hls+1:jpj-nn_hls)" in text
    assert "SIZE(a,1) == ni" in text
    assert "put_f2 requires full halo" in text
    assert "ACCESS='stream'" in text and "WRITE(unit) a" in text
    assert "ABS(" not in text and "MERGE(" not in text


def test_o7_script_default_is_read_only_and_operator_run_is_explicit() -> None:
    text = (ACQ / "run.sh").read_text()
    assert "mode=${1:---preflight-only}" in text
    assert "# ---- Operator-only --run mode" in text
    assert text.index('if [[ "$mode" == --preflight-only ]]') < text.index("./makenemo")
    assert 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' in text
    assert "--admit-existing" in text and "producer_commit.txt" in text
    assert "manifest/records.sha256" in text and "manifest/inputs.sha256" in text


def test_o7_read_only_preflight_exercises_real_ppsrc_and_all_controls(tmp_path: Path) -> None:
    base = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O2_REF/BLD")
    compiler = Path("/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran")
    if not base.is_dir() or not compiler.is_file():
        pytest.skip("NEMO oracle build/compiler not mounted")
    report = tmp_path / "preflight.json"
    proc = subprocess.run(
        [str(ACQ / "run.sh"), "--preflight-only", str(report)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    result = json.loads(report.read_text())
    assert result == {"status": "PREFLIGHT_OK", "nemo_run": False, "plants": 24}
    assert "PASS_SAME_PROGRAM_REAL_PPSRC_MARKERS" in proc.stdout
    assert "PASS_11_LANDINGS_AND_11_MISPLACEMENT_PLANTS" in proc.stdout
    assert "PREFLIGHT_OK: no NEMO build or integration was run" in proc.stdout

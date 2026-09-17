"""Fail-closed tests for the zero-cost ORCA2 inventory probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
from pathlib import Path

import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"

# The seven reference streams whose writer instrumentation is gone from the
# current ORCA2_OMIP_L4 source card. Kept in step with EXPECTED_ABSENT_STREAMS
# in the acquisition script, which is asserted below.
REGISTERED_ABSENT_STREAMS = (
    "oracle_bbl_diffusive_kt00000001.bin",
    "oracle_een_e3f0vor_kt00000001.bin",
    "oracle_een_e3fvor_kt00000001.bin",
    "oracle_een_q_kt00000001.bin",
    "oracle_een_zpvo_kt00000001.bin",
    "oracle_zdf_sh2_operands_kt00000001.bin",
    "oracle_zdf_sh2_operands_kt00000002.bin",
)
# The one inherited stream compared only in part: three 94x152x31 double
# fields (zFu, zFv, zFw) after a 48-byte header, of which the never-assigned
# third field is excluded. These mirror PARTIAL_STREAM, PARTIAL_STREAM_BYTES
# and PARTIAL_COMPARED_BYTES in the acquisition script, asserted below.
PARTIAL_STREAM = "oracle_transport_kt00000001_s1.bin"
TRANSPORT_HEADER_BYTES = 16 + 8 * 4
TRANSPORT_FIELD_BYTES = 94 * 152 * 31 * 8
TRANSPORT_STREAM_BYTES = TRANSPORT_HEADER_BYTES + 3 * TRANSPORT_FIELD_BYTES
TRANSPORT_COMPARED_BYTES = TRANSPORT_HEADER_BYTES + 2 * TRANSPORT_FIELD_BYTES

SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_orca2_parallel_inventory",
    TESTCASES / "nemo_testcase_l2_orca2_parallel_inventory.py",
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _write_tke(path: Path, *, trailing: bool = False) -> None:
    jpi, jpj, jpk = 1, 1, 1
    payload = probe.TKE_3D_FIELDS + probe.TKE_2D_FIELDS
    header = (
        1,
        2,
        3,
        3,
        jpi,
        jpj,
        jpk,
        64,
        probe.TKE_3D_FIELDS,
        probe.TKE_2D_FIELDS,
        payload,
        1,
        1,
        3,
        1,
    )
    extents = (jpi, jpj, jpk) * probe.TKE_3D_FIELDS + (jpi, jpj, 1) * probe.TKE_2D_FIELDS
    raw = (
        probe.TKE_MAGIC.encode("ascii").ljust(16, b" ")
        + struct.pack(f"={probe.TKE_HEADER_INTS}i", *header)
        + struct.pack(f"={3 * probe.TKE_FIELDS}i", *extents)
        + b"\0" * (8 * payload)
    )
    path.write_bytes(raw + (b"X" if trailing else b""))


def _write_transport(path: Path, *, third_field: bytes) -> None:
    """Write a stage-1 transport stream: header, zFu, zFv, then zFw.

    zFu and zFv carry the same deterministic pattern in both runs, as two real
    transports must; only the third field differs, exactly as the reference and
    the candidate differ in the uninitialised buffer the model dumps there.
    """
    header = b"NEMO_L2_TRANSP_1" + struct.pack(
        "=8i", 1, 1, 3, 3, 94, 152, 31, 1
    )
    assert len(header) == TRANSPORT_HEADER_BYTES
    signal = (b"\x11\x22\x33\x44\x55\x66\x77\x88" * (TRANSPORT_FIELD_BYTES // 8))
    assert len(third_field) == TRANSPORT_FIELD_BYTES
    path.write_bytes(header + signal + signal + third_field)
    assert path.stat().st_size == TRANSPORT_STREAM_BYTES


def _flip_byte(data: bytes, offset: int) -> bytes:
    return data[:offset] + bytes([data[offset] ^ 0xFF]) + data[offset + 1:]


def _write_orca2_boundary(path: Path) -> None:
    header = (1, 2, 3, 3, 94, 152, 31, 64, 1, 0, 94 * 152 * 31, 1, 1, 3, 1)
    size = 16 + 15 * 4 + 3 * 4 + 94 * 152 * 31 * 8
    with path.open("wb") as handle:
        handle.write(b"NEMO_L4_TKEB_1".ljust(16, b" "))
        handle.write(struct.pack("=15i", *header))
        handle.write(struct.pack("=3i", 94, 152, 31))
        handle.seek(size - 1)
        handle.write(b"\0")


def _write_entry_stage(path: Path, *, kt: int, stage: int | None) -> None:
    if stage is None:
        magic = probe.ENTRY_MAGIC
        level = 1 if kt % 2 else 3
        header = (1, kt, level, probe.NX, probe.NY, probe.NZ, probe.NTR, 64)
    else:
        magic = probe.STAGE_MAGIC
        level = 2 if stage == 2 else (3 if kt % 2 else 1)
        header = (
            1,
            kt,
            stage,
            level,
            probe.NX,
            probe.NY,
            probe.NZ,
            probe.NTR,
            64,
        )
    payload = 4 * probe.NX * probe.NY * probe.NZ + probe.NX * probe.NY
    expected_bytes = 16 + 4 * len(header) + 8 * payload
    with path.open("wb") as handle:
        handle.write(magic.encode("ascii").ljust(16, b" "))
        handle.write(struct.pack(f"={len(header)}i", *header))
        handle.seek(expected_bytes - 1)
        handle.write(b"\0")


def test_current_dispatch_keys_are_read_without_importing_model():
    source = """
def build_nemo_testcase_card(case):
    builders = {"GYRE-zco": build_gyre, "LOCK_EXCHANGE-zco": build_lock}
    return builders[case]()
"""
    assert probe._dispatch_keys(source) == ("GYRE-zco", "LOCK_EXCHANGE-zco")


def test_historical_execution_guard_is_read_from_ast():
    source = """
def build_orca2_zps_card():
    return NEMOTestcaseCard("ORCA2-zps", unmeasured_features=("one", "two"))
"""
    assert probe._historical_unmeasured_features(source) == ("one", "two")


def test_tke_reader_reaches_exact_eof_and_rejects_trailing_byte(tmp_path):
    exact = tmp_path / "exact.bin"
    _write_tke(exact)
    row = probe._read_tke_schema(exact)
    assert row["exact_eof"] is True
    trailing = tmp_path / "trailing.bin"
    _write_tke(trailing, trailing=True)
    with pytest.raises(probe.InventoryError, match="trailing bytes"):
        probe._read_tke_schema(trailing)


@pytest.mark.parametrize(("kt", "stage"), [(1, None), (2, 1), (2, 2), (1, 3)])
def test_entry_and_stage_readers_parse_schema_and_exact_eof(tmp_path, kt, stage):
    record = tmp_path / f"entry-stage-{kt}-{stage}.bin"
    _write_entry_stage(record, kt=kt, stage=stage)
    row = probe._read_entry_stage_schema(record, kt, stage)
    assert row["exact_eof"] is True
    assert row["fields"] == ["T", "S", "u", "v", "ssh"]
    with record.open("ab") as handle:
        handle.write(b"X")
    with pytest.raises(probe.InventoryError, match="exact EOF"):
        probe._read_entry_stage_schema(record, kt, stage)


def test_phase2v_field_map_names_every_field_and_only_boundary_is_missing():
    tke = {
        "fields_3d": list(probe.TKE_3D_FIELD_NAMES),
        "fields_2d": list(probe.TKE_2D_FIELD_NAMES),
    }
    zdf = {
        "fields_3d": list(probe.ZDF_3D_FIELDS),
        "fields_2d": list(probe.ZDF_2D_FIELDS),
    }
    contract = probe._tke_contract_map(tke, zdf)
    assert [row["phase2v_field"] for row in contract["phase2v_field_map"]] == list(
        probe.TKE_FIELD_NAMES
    )
    assert contract["missing"] == ["en_after_boundaries"]


def test_missing_boundary_and_provenance_plants_flip_and_refuse():
    predicates = {
        "stream_counts": (101, 101),
        "schemas_valid": True,
        "entry_stage_twins_exact": True,
        "producer_hashes_match": True,
    }
    for plant in ("missing-boundary", "provenance-mismatch"):
        with pytest.raises(probe.InventoryError, match=f"{plant} plant flipped decision"):
            probe._run_decision_plant(plant, **predicates)


def test_phase2v_decision_emits_binding_missing_boundary_verdict():
    decision = probe._reuse_decision(
        stream_counts=(101, 101),
        schemas_valid=True,
        entry_stage_twins_exact=True,
        producer_hashes_match=True,
        available_tke_fields=set(probe.ROUND101_TKE_CONTRACT)
        - {"en_after_boundaries"},
    )
    assert decision["entry_stage_reusable"] is True
    assert decision["acquisition_needed"] is True
    assert decision["verdict"] == (
        "REUSABLE FOR entry/stage; TKE boundary frame MISSING"
    )


def test_boundary_acquisition_is_new_np2_write_only_passivity_gate():
    acquisition = (
        TESTCASES
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "run.sh"
    ).read_text()
    assert "REFERENCE_CFG=ORCA2_ICE_PISCES" in acquisition
    assert "FAILED_TARGET_CFG=ORCA2_OMIP_L4_P2VBND" in acquisition
    assert "TARGET_CFG=ORCA2_OMIP_L4_P2VBND_R2" in acquisition
    assert 'verify_exp00_copy "$SOURCE_ROOT/EXP00" "$TARGET_ROOT/EXP00"' in acquisition
    assert "mpirun -np 2" in acquisition
    assert "gfortran -fsyntax-only" in acquisition
    assert "EXPECTED_BASELINE_STREAMS=101" in acquisition
    assert 'cmp -s "$BASELINE_RUN/$name" "$TARGET_RUN/$name"' in acquisition
    assert "write-only passivity failed" in acquisition


def test_boundary_finalize_is_post_run_only_and_reads_native_nemo_output():
    acquisition = (
        TESTCASES
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "run.sh"
    ).read_text()
    function_start = acquisition.index("finalize_existing() {")
    function_end = acquisition.index(
        '\n}\n\nif [[ "$MODE" == --finalize ]]', function_start
    )
    finalize = acquisition[function_start:function_end]
    dispatch = acquisition.index('if [[ "$MODE" == --finalize ]]')
    clean_tree_gate = acquisition.index('cd "$REPO"', dispatch)

    assert "--run|--preflight-only|--finalize" in acquisition
    assert dispatch < clean_tree_gate
    assert "makenemo" not in finalize
    assert "mpirun" not in finalize
    assert "-name 'ocean.output'" in finalize
    assert "-name 'ocean.output.*'" in finalize
    assert "-name 'ocean.output_*'" in finalize
    assert "ORCA2_TKE_BOUNDARY_DUMP" in finalize
    assert 'marker_kt" -eq "$EXPECTED_KT' in finalize
    assert 'marker_count" -eq 1' in finalize
    assert (
        "grep -Fq 'ORCA2_TKE_BOUNDARY_DUMP' "
        '"$TARGET_RUN/run.user.stdout.log"'
    ) not in acquisition
    assert "existing target binary digest does not match binary manifest" in finalize
    assert "does not match registered manifest size" in finalize
    assert "orca2_tke_boundary_admission.json" in acquisition
    assert "orca2_tke_boundary_outputs.sha256" in acquisition
    assert "orca2_tke_boundary_references.sha256" in acquisition
    assert "EXPECTED_BASELINE_ORACLE_MANIFEST_SHA256" in finalize

    # The registered-absent list and the narrowed transport comparison are the
    # two deliberate narrowings of the passivity gate; both stay pinned here.
    for name in REGISTERED_ABSENT_STREAMS:
        assert name in acquisition
    assert "readonly EXPECTED_COMPARED_STREAMS=94" in acquisition
    assert "readonly EXPECTED_TARGET_STREAMS=95" in acquisition
    assert "readonly EXPECTED_ABSENT_STREAM_COUNT=7" in acquisition
    assert "ln_dynadv_vec" in acquisition
    assert "stream_is_expected_absent" in finalize
    assert "compare_inherited_stream" in finalize


def test_boundary_finalize_admits_synthetic_twin_and_named_plants_refuse(tmp_path):
    source = (
        TESTCASES
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "run.sh"
    ).read_text()
    nemo_root = tmp_path / "nemo"
    baseline = tmp_path / "baseline"
    target = tmp_path / "target"
    binary = (
        nemo_root
        / "cfgs/ORCA2_OMIP_L4_P2VBND_R2/BLD/bin/nemo.exe"
    )
    binary.parent.mkdir(parents=True)
    baseline.mkdir()
    target.mkdir()
    binary_bytes = b"synthetic executable\n"
    binary.write_bytes(binary_bytes)
    (target / "nemo").write_bytes(binary_bytes)
    binary_digest = hashlib.sha256(binary_bytes).hexdigest()
    binary_manifest = target / "binary.sha256"
    binary_manifest.write_text(f"{binary_digest}  {binary}\n")
    (target / "producer_commit.txt").write_text("1" * 40 + "\n")
    (target / "run.user.stdout.log").write_text("STOP 0\n")
    (target / "run.user.time.log").write_text("MPIRUN_RC=0\nRUN DONE\n")
    (target / "time.step").write_text("10\n")
    marker = target / "ocean.output_0000"
    marker.write_text(" ORCA2_TKE_BOUNDARY_DUMP            2 3 3\n")

    # The reference run holds seven streams whose writer no longer exists in
    # the current source card, so a passive candidate cannot produce them: they
    # exist only in the baseline here, exactly as in the real pair.
    for name in REGISTERED_ABSENT_STREAMS:
        (baseline / name).write_bytes(f"absent-{name}\n".encode())
    for index in range(93):
        name = f"oracle_inherited_{index:03d}.bin"
        payload = f"inherited-{index}\n".encode()
        (baseline / name).write_bytes(payload)
        (target / name).write_bytes(payload)
    # The 94th comparable stream is the stage-1 transport stream, the one the
    # gate compares only in part. The reference dumps zeros in the excluded
    # third field and this candidate dumps the sea-ice initialisation value, so
    # the twin reproduces the real disagreement the exclusion exists for.
    _write_transport(baseline / PARTIAL_STREAM, third_field=bytes(
        TRANSPORT_FIELD_BYTES
    ))
    _write_transport(target / PARTIAL_STREAM, third_field=struct.pack(
        "=d", 270.0
    ) * (TRANSPORT_FIELD_BYTES // 8))
    baseline_manifest = "".join(
        f"{hashlib.sha256((baseline / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(path.name for path in baseline.glob("oracle_*.bin"))
    )
    baseline_manifest_digest = hashlib.sha256(
        baseline_manifest.encode()
    ).hexdigest()
    record = target / "oracle_tke_boundary_kt00000002.bin"
    _write_orca2_boundary(record)

    replacements = {
        "readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2": (
            f"readonly NEMO_ROOT={nemo_root}"
        ),
        (
            "readonly BASELINE_RUN=/data/abyssal/dbalwada/"
            "nemo-testcases-l4/runs/"
            "variant_icebergs_off_phase2v_tke_a_10step_np2"
        ): (
            f"readonly BASELINE_RUN={baseline}"
        ),
        (
            "readonly TARGET_RUN=/data/abyssal/dbalwada/"
            "nemo-testcases-l2/phase3/parallel/orca2/"
            "oracle_phase2v_tke_boundary_np2"
        ): (
            f"readonly TARGET_RUN={target}"
        ),
        (
            "readonly EXPECTED_BASELINE_ORACLE_MANIFEST_SHA256="
            "4fbffaed98202a052059c503a637bc2f8d5e57c28e7345f5bfce5b62320708a9"
        ): (
            "readonly EXPECTED_BASELINE_ORACLE_MANIFEST_SHA256="
            f"{baseline_manifest_digest}"
        ),
    }
    script_text = source
    for registered, synthetic in replacements.items():
        assert registered in script_text
        script_text = script_text.replace(registered, synthetic, 1)
    script = tmp_path / "run.sh"
    script.write_text(script_text)

    def finalize() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), "--finalize"],
            check=False,
            capture_output=True,
            text=True,
        )

    finalization_artifacts = [
        target / "oracle_tke_boundary_kt00000002.bin.stamp",
        target / "orca2_tke_boundary_admission.json",
        target / "orca2_tke_boundary_outputs.sha256",
        target / "orca2_tke_boundary_references.sha256",
    ]

    def assert_no_finalization_artifacts() -> None:
        assert not [path for path in finalization_artifacts if path.exists()]

    admitted = finalize()
    assert admitted.returncode == 0, admitted.stdout + admitted.stderr
    assert "ORCA2_TKE_BOUNDARY_MARKER PASS kt=2" in admitted.stdout
    assert (
        "expected=94 compared=94 identical=94 missing=0 absent=7/7"
        in admitted.stdout
    )
    assert "excluded_field=zFw" in admitted.stdout
    for name in REGISTERED_ABSENT_STREAMS:
        assert f"ORCA2_TKE_BOUNDARY_ABSENT {name}" in admitted.stdout
    report = json.loads((target / "orca2_tke_boundary_admission.json").read_text())
    assert report["verdict"] == "PASS"
    assert report["exit_code"] == 0
    assert report["binary_sha256"] == binary_digest
    assert report["passivity"]["byte_identical"] == 94
    assert report["passivity"]["expected_compared_streams"] == 94
    assert report["passivity"]["expected_absent"]["streams"] == sorted(
        REGISTERED_ABSENT_STREAMS
    )
    assert (
        report["passivity"]["partial_comparison"]["excluded_field"] == "zFw"
    )
    assert (
        report["references"]["baseline_oracle_streams_sha256"]
        == baseline_manifest_digest
    )
    digests = subprocess.run(
        ["sha256sum", "-c", "orca2_tke_boundary_outputs.sha256"],
        cwd=target,
        check=False,
        capture_output=True,
        text=True,
    )
    assert digests.returncode == 0, digests.stdout + digests.stderr

    binary.write_bytes(binary_bytes + b"post-finalize plant")
    changed_reference = subprocess.run(
        ["sha256sum", "-c", "orca2_tke_boundary_outputs.sha256"],
        cwd=target,
        check=False,
        capture_output=True,
        text=True,
    )
    assert changed_reference.returncode != 0
    assert f"{binary}: FAILED" in changed_reference.stdout
    binary.write_bytes(binary_bytes)

    marker.write_text(
        " ORCA2_TKE_BOUNDARY_DUMP            2 3 3\n"
        " ORCA2_TKE_BOUNDARY_DUMP            2 3 3\n"
    )
    duplicate_marker = finalize()
    assert duplicate_marker.returncode == 66
    assert "REFUSE: ocean.output boundary marker mismatch" in duplicate_marker.stderr
    assert_no_finalization_artifacts()
    marker.write_text(" ORCA2_TKE_BOUNDARY_DUMP            2 3 3\n")
    readmitted = finalize()
    assert readmitted.returncode == 0, readmitted.stdout + readmitted.stderr

    inherited = target / "oracle_inherited_000.bin"
    inherited_bytes = inherited.read_bytes()
    inherited.write_bytes(inherited_bytes + b"plant")
    changed_stream = finalize()
    assert changed_stream.returncode == 69
    assert (
        "compared=94 identical=93 missing=0 absent=7/7 differing=1"
        in changed_stream.stdout
    )
    assert "REFUSE: write-only passivity failed" in changed_stream.stderr
    refused_report = json.loads(
        (target / "orca2_tke_boundary_admission.json").read_text()
    )
    assert refused_report["exit_code"] == 69
    inherited.write_bytes(inherited_bytes)

    # The narrowed comparison must still refuse when a stream that is NOT on
    # the registered-absent list goes missing.
    inherited.unlink()
    absent_unregistered = finalize()
    assert absent_unregistered.returncode == 69
    assert "missing=1" in absent_unregistered.stdout
    assert "REFUSE: write-only passivity failed" in absent_unregistered.stderr
    inherited.write_bytes(inherited_bytes)

    # A registered-absent stream reappearing changes the compared count, which
    # must refuse rather than be silently tolerated.
    reappeared = target / REGISTERED_ABSENT_STREAMS[0]
    reappeared.write_bytes((baseline / reappeared.name).read_bytes())
    absent_returned = finalize()
    assert absent_returned.returncode == 69
    assert "compared=95" in absent_returned.stdout
    assert "absent=6/7" in absent_returned.stdout
    assert "REFUSE: write-only passivity failed" in absent_returned.stderr
    reappeared.unlink()
    readmitted = finalize()
    assert readmitted.returncode == 0, readmitted.stdout + readmitted.stderr

    # The partially compared stage-1 transport stream. The excluded third
    # field is the ONLY part these two runs disagree on, and every admission
    # above happened with that disagreement in place: the exclusion is being
    # exercised, not merely described in a line that prints unconditionally.
    transport = target / PARTIAL_STREAM
    transport_bytes = transport.read_bytes()
    baseline_transport = (baseline / PARTIAL_STREAM).read_bytes()
    assert transport_bytes != baseline_transport
    assert (
        transport_bytes[:TRANSPORT_COMPARED_BYTES]
        == baseline_transport[:TRANSPORT_COMPARED_BYTES]
    )
    partial = report["passivity"]["partial_comparison"]
    assert partial["stream"] == PARTIAL_STREAM
    assert partial["compared_bytes"] == TRANSPORT_COMPARED_BYTES
    assert partial["required_bytes"] == TRANSPORT_STREAM_BYTES

    def plant_transport(mutated: bytes) -> subprocess.CompletedProcess[str]:
        transport.write_bytes(mutated)
        result = finalize()
        transport.write_bytes(transport_bytes)
        return result

    # A corruption in the header, in the first transport (zFu) or in the second
    # (zFv) lies inside the compared span, so it must refuse and name the
    # stream it caught.
    for label, offset in (
        ("header", 20),
        ("zFu", TRANSPORT_HEADER_BYTES + 1000),
        ("zFv", TRANSPORT_HEADER_BYTES + TRANSPORT_FIELD_BYTES + 1000),
        ("zFv-last-byte", TRANSPORT_COMPARED_BYTES - 1),
    ):
        corrupted = plant_transport(_flip_byte(transport_bytes, offset))
        assert corrupted.returncode == 69, label
        assert (
            f"ORCA2_TKE_BOUNDARY_DIFFERS {PARTIAL_STREAM}" in corrupted.stdout
        ), label
        assert "compared=94 identical=93" in corrupted.stdout, label
        assert "differing=1" in corrupted.stdout, label
        assert "REFUSE: write-only passivity failed" in corrupted.stderr, label

    # A corruption confined to the excluded third field is admitted, at the
    # first excluded byte and at the last byte of the stream alike.
    for label, offset in (
        ("zFw", TRANSPORT_COMPARED_BYTES + 1000),
        ("zFw-last-byte", TRANSPORT_STREAM_BYTES - 1),
    ):
        excluded = plant_transport(_flip_byte(transport_bytes, offset))
        assert excluded.returncode == 0, label + excluded.stdout + excluded.stderr
        assert "compared=94 identical=94" in excluded.stdout, label

    # A wrong total length refuses under its own named condition, short or
    # padded, so no byte can hide in the excluded tail.
    for label, mutated in (
        ("short", transport_bytes[:-8]),
        ("padded", transport_bytes + bytes(8)),
    ):
        wrong_length = plant_transport(mutated)
        assert wrong_length.returncode == 69, label
        assert (
            f"ORCA2_TKE_BOUNDARY_WRONG_LENGTH {PARTIAL_STREAM}"
            in wrong_length.stdout
        ), label
        assert "wrong_length=1" in wrong_length.stdout, label
        assert (
            f"REFUSE: inherited stream {PARTIAL_STREAM} is not the registered "
            f"{TRANSPORT_STREAM_BYTES} bytes" in wrong_length.stderr
        ), label

    readmitted = finalize()
    assert readmitted.returncode == 0, readmitted.stdout + readmitted.stderr

    marker.write_text(" ORCA2_TKE_BOUNDARY_DUMP            3 3 3\n")
    wrong_kt = finalize()
    assert wrong_kt.returncode == 66
    assert "REFUSE: ocean.output boundary marker mismatch" in wrong_kt.stderr
    assert_no_finalization_artifacts()
    marker.write_text(" ORCA2_TKE_BOUNDARY_DUMP            2 3 3\n")

    baseline_inherited = baseline / inherited.name
    baseline_inherited.write_bytes(inherited_bytes + b"plant")
    inherited.write_bytes(inherited_bytes + b"plant")
    changed_baseline = finalize()
    assert changed_baseline.returncode == 65
    assert "Phase-2v A oracle-stream manifest digest changed" in changed_baseline.stderr
    assert_no_finalization_artifacts()
    baseline_inherited.write_bytes(inherited_bytes)
    inherited.write_bytes(inherited_bytes)

    binary_manifest.write_text(
        f"{binary_digest}  {binary}\n{binary_digest}  {binary}\n"
    )
    multiline_manifest = finalize()
    assert multiline_manifest.returncode == 66
    assert "REFUSE: existing binary manifest" in multiline_manifest.stderr
    assert_no_finalization_artifacts()

    binary_manifest.write_text("")
    empty_manifest = finalize()
    assert empty_manifest.returncode == 66
    assert "REFUSE: existing binary manifest" in empty_manifest.stderr
    assert_no_finalization_artifacts()

    binary_manifest.write_text(f"{binary_digest}  {binary}")
    unterminated_manifest = finalize()
    assert unterminated_manifest.returncode == 66
    assert "REFUSE: existing binary manifest" in unterminated_manifest.stderr
    assert_no_finalization_artifacts()
    binary_manifest.write_text(f"{binary_digest}  {binary}\n")

    (target / "nemo").write_bytes(binary_bytes + b"plant")
    wrong_binary = finalize()
    assert wrong_binary.returncode == 66
    assert "REFUSE: existing target binary digest" in wrong_binary.stderr
    assert_no_finalization_artifacts()
    (target / "nemo").write_bytes(binary_bytes)

    real_target = tmp_path / "target-real"
    target.rename(real_target)
    target.symlink_to(real_target, target_is_directory=True)
    wrong_run_dir = finalize()
    assert wrong_run_dir.returncode == 64
    assert "REFUSE: finalization run directory" in wrong_run_dir.stderr
    target.unlink()
    real_target.rename(target)

    readmitted = finalize()
    assert readmitted.returncode == 0, readmitted.stdout + readmitted.stderr

    with record.open("r+b") as handle:
        handle.truncate(record.stat().st_size - 1)
    wrong_size = finalize()
    assert wrong_size.returncode == 66
    assert "REFUSE: existing record size" in wrong_size.stderr
    assert_no_finalization_artifacts()


def test_exp00_copy_check_compares_link_text_and_regular_bytes(tmp_path):
    acquisition = (
        TESTCASES
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "run.sh"
    ).read_text()
    start = acquisition.index("verify_exp00_copy() {")
    end = acquisition.index("\n\n# USER-EXECUTED", start)
    helper = acquisition[start:end]

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "namelist_cfg").write_bytes(b"same bytes")
    (target / "namelist_cfg").write_bytes(b"same bytes")
    (source / "nemo").symlink_to("../BLD/bin/nemo.exe")
    (target / "nemo").symlink_to("../BLD/bin/nemo.exe")

    def run_check() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f'{helper}\nverify_exp00_copy "$1" "$2"',
                "copy-check",
                str(source),
                str(target),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert run_check().returncode == 0

    (target / "nemo").unlink()
    (target / "nemo").symlink_to("../different/nemo.exe")
    link_plant = run_check()
    assert link_plant.returncode == 68
    assert "file-by-file EXP00 copy differs for nemo" in link_plant.stderr

    (target / "nemo").unlink()
    (target / "nemo").symlink_to("../BLD/bin/nemo.exe")
    (target / "namelist_cfg").write_bytes(b"different bytes")
    byte_plant = run_check()
    assert byte_plant.returncode == 68
    assert "file-by-file EXP00 copy differs for namelist_cfg" in byte_plant.stderr


def test_require_fails_closed():
    with pytest.raises(probe.InventoryError, match="named predicate"):
        probe.require(False, "named predicate")

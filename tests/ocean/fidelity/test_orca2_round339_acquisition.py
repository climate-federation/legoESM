from __future__ import annotations

import json
import struct
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_common as common,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_ice_cover_gate as ice,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_k0_smoke_gate as k0,
)


def _write_record(path: Path, *, kt: int, rank: int, value: float) -> None:
    field = np.full((6, 6), -0.0, dtype=np.float64, order="F")
    field[2:4, 2:4] = value
    header = (1, kt, rank, 1 + 2 * rank, 1, 6, 6, 3, 4, 3, 4, 64, 2)
    checksum = ice.xor_checksum(field)
    signed = checksum if checksum < (1 << 63) else checksum - (1 << 64)
    path.write_bytes(
        ice.MAGIC.ljust(16).encode("ascii")
        + struct.pack(f"={len(header)}i", *header)
        + struct.pack("=q", signed)
        + field.tobytes(order="F")
    )


def test_ice_record_parser_and_plants(tmp_path: Path) -> None:
    path = tmp_path / ice.record_name(1, 0)
    _write_record(path, kt=1, rank=0, value=0.25)
    record = ice.read_record(path)
    assert record["header"]["real_bits"] == 64
    assert np.signbit(record["values"][0, 0])
    for plant in ("header", "truncation", "checksum", "nonfinite"):
        with pytest.raises(common.GateError):
            ice.read_record(path, plant=plant)


def test_ice_mean_native_cover_and_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ice, "NX", 4)
    monkeypatch.setattr(ice, "NY", 2)
    monkeypatch.setattr(ice, "EXPECTED_KT", (1, 3))
    monkeypatch.setattr(ice, "N_SAMPLES", 2)
    for kt, offset in ((1, 0.0), (3, 0.25)):
        for rank in (0, 1):
            _write_record(tmp_path / ice.record_name(kt, rank), kt=kt, rank=rank,
                          value=offset + 0.25 * (rank + 1))
    mean, hashes = ice.build_mean(tmp_path)
    assert len(hashes) == 4
    np.testing.assert_array_equal(mean[:, :2], 0.375)
    np.testing.assert_array_equal(mean[:, 2:], 0.625)
    output = tmp_path / "ice_cover_clim.nc"
    ice.write_climatology(output, mean, {"producer_commit": "a" * 40})
    result = ice.verify_climatology(output, mean)
    assert result["minimum"] == 0.375
    changed, _ = ice.build_mean(tmp_path, plant="one-ulp")
    assert not np.array_equal(changed.view(np.uint64), mean.view(np.uint64))


def test_ice_cover_schema_plant_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ice, "NX", 4)
    monkeypatch.setattr(ice, "NY", 2)
    expected = np.zeros((2, 4), dtype=np.float64)
    path = tmp_path / "bad.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time_counter", 1)
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 4)
        dataset.createVariable("ice_cover", "f4", ("time_counter", "y", "x"))[:] = 0
    with pytest.raises(common.GateError):
        ice.verify_climatology(path, expected)


def _write_namelist_ref(path: Path) -> None:
    path.write_text(
        "&namsbc\n nn_ice=0\n/\n"
        "&namsbc_iif\n sn_ice='x'\n/\n"
        "&namberg\n ln_icebergs=.false.\n/\n"
        "&namdrg\n ln_drgice_imp=.true.\n/\n"
        "&namzdf_tke\n ln_mxl0=.true.\n nn_mxlice=0\n/\n",
        encoding="utf-8",
    )


def _write_restart(path: Path, value: float) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("x", 2)
        dataset.createVariable("state", "f8", ("x",))[:] = value


def _k0_fixture(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "namelist_ref"
    _write_namelist_ref(reference)
    (tmp_path / "ocean.output").write_text(
        "nn_ice = 1\nln_icebergs = F\nln_drgice_imp = F\nln_mxl0 = T\n"
        "nn_mxlice = 0\nnn_itend = 10\nsn_ice = ice_cover_clim.nc\n",
        encoding="utf-8",
    )
    (tmp_path / "run.user.stdout.log").write_text("STOP 0\n", encoding="utf-8")
    (tmp_path / "compiled_keys.txt").write_text("key_si3 key_qco key_RK3\n", encoding="utf-8")
    (tmp_path / "nemo").write_bytes(b"binary")
    (tmp_path / "ice_cover_clim.nc").write_bytes(b"cover")
    for rank in (0, 1):
        _write_restart(tmp_path / f"ORCA2_00000010_restart_{rank:04d}.nc", rank + 1.0)
    files = [p for p in tmp_path.iterdir() if p.is_file() and p.name not in
             {"namelist_ref", "producer_manifest.json"}]
    common.write_manifest(tmp_path / "producer_manifest.json", producer_commit="b" * 40,
                          binary=tmp_path / "nemo", files=files, root=tmp_path)
    return tmp_path, reference


def test_k0_gate_and_every_plant(tmp_path: Path) -> None:
    root, reference = _k0_fixture(tmp_path)
    result = k0.run_gate(root, reference, expected_commit="b" * 40)
    assert result["status"] == "PASS_R339_K0_SMOKE"
    assert result["scores_physics"] is False
    for plant in k0.PLANTS:
        with pytest.raises(common.GateError):
            k0.run_gate(root, reference, expected_commit="b" * 40, plant=plant)


def test_manifest_rejects_digest_and_escape(tmp_path: Path) -> None:
    root, _ = _k0_fixture(tmp_path)
    (root / "nemo").write_bytes(b"changed")
    with pytest.raises(common.GateError):
        common.verify_manifest(root / "producer_manifest.json")
    payload = json.loads((root / "producer_manifest.json").read_text())
    payload["files"] = {"../escape": "0" * 64}
    (root / "producer_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(common.GateError):
        common.verify_manifest(root / "producer_manifest.json")

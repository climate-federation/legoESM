#!/usr/bin/env python3
"""Walk the admitted GYRE kt=2 ``stp2d`` slow-forcing producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round16_slow_forcing as round16  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round78_uamid_walk as round78  # noqa: E402
import nemo_testcase_l2_gyre_round81_btstep_gate as round81  # noqa: E402
import nemo_testcase_l2_gyre_round82_btstep_walk as round82  # noqa: E402
import nemo_testcase_l2_gyre_round117_preloop_gate as round117_record  # noqa: E402
import nemo_testcase_l2_gyre_round146_rhs_family_gate as round146_family  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.dynamics import (  # noqa: E402
    ocean_model_latlon_cgrid as model_module,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
ROUND64_PRODUCER = "3b3b045bd9e03b60330204e7590e4c4470b7a0ca"
ROUND117_PRODUCER = "c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd"
ROUND117_BOUNDARIES = (
    "after_hpg", "after_ldf", "after_vor", "after_keg",
    "after_zad", "after_adv",
)
ROUND117_FINAL_MAX = {
    "u": np.float64(1.0529650291768787e-11),
    "v": np.float64(1.0765559917925099e-11),
}
ROUND139_RECORD = "oracle_slow_forcing_split_kt00001081.bin"
ROUND139_MAGIC = "NEMO_L2_R139SLOW"
ROUND139_FIELDS = (
    "incoming_u", "incoming_v", "coriolis_u", "coriolis_v",
    "final_u", "final_v",
)
ROUND139_KT = 1081
ROUND139_KMM = 1
ROUND139_HEADER_INTS = 11
ROUND139_COUNT = round81.NX * round81.NY
ROUND139_EXPECTED_SIZE = (
    16 + ROUND139_HEADER_INTS * 4
    + len(ROUND139_FIELDS) * ROUND139_COUNT * 8
)
ROUND139_PARENT_EXTERNAL_SHA256 = (
    "6bc0f990ccba183a48ad09549b72b549603694e917389eb4f8b9c03c88ceaf09")
ROUND139_PARENT_RESTART_SHA256 = (
    "6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976")
ROUND139_PARENT_PROCESS_SHA256 = (
    "526d1fc73faeda990c661f2363a5bb328168bae17d4aea315daf05c35b4cd7b0")
ROUND139_PARENT_QCO_SHA256 = (
    "626d21e229f7ced8f606f6385e04224fb2e086cef92d81d95dff7e2ef3e90878")
ROUND139_RECORD_SHA256 = (
    "0fee139d96d9a3731ad1d40a6dc28ead3b95b4a1bbb68f51d7cc99baf1215e95")
ROUND139_MANIFEST = "round139_outputs.sha256"
ROUND139_MANIFEST_MEMBERS = (
    ROUND139_RECORD,
    ROUND139_RECORD + ".stamp",
    round81.DEVELOPED_RECORD,
    round81.DEVELOPED_RECORD + ".stamp",
    round81.DEVELOPED_QCO_RECORD,
    round81.DEVELOPED_QCO_RECORD + ".stamp",
    "GYRE_OMIP_L2_P3_00001080_restart.nc",
    "oracle_process_budget_kt00001081.bin",
    "round139_parent_record_validation.json",
    "round139_record_validation.json",
    "round139_passive_admission_plant.log",
    "round139_record-header_plant.log",
    "round139_record-replay-ulp_plant.log",
    "round139_record-stamp_plant.log",
    "round139_record-truncation_plant.log",
    "round139_record-header_plant.json",
    "round139_record-replay-ulp_plant.json",
    "round139_record-stamp_plant.json",
    "round139_record-truncation_plant.json",
)
ROUND140_OPERAND_REGISTRY = (
    "incoming_u", "incoming_v", "coriolis_u", "coriolis_v",
    "mask_u", "mask_v", "final_u", "final_v",
)
ROUND140_CALLBACK_FIELDS = ROUND140_OPERAND_REGISTRY
ROUND140_RHS_RECORD = "oracle_developed_rhs_kt00001081.bin"
ROUND140_RHS_MAGIC = "NEMO_L2_R140RHS"
ROUND140_RHS_VERSION = 3
ROUND140_RHS_FIELDS = (
    "e3u", "rhs_u", "umask", "e3v", "rhs_v", "vmask",
    "depth_mean_u", "depth_mean_v", "r1_hu0", "r1_hv0",
    "post_drag_u", "post_drag_v", "cd_u", "cd_v", "r1_rho0",
    "wind_tau_u", "wind_tau_v", "wind_r1_hu", "wind_r1_hv",
    "post_wind_u", "post_wind_v",
)
ROUND140_RHS_EXPECTED_SIZE = 1_486_548
ROUND140_RHS_MANIFEST = "round140_outputs.sha256"
ROUND140_RHS_MANIFEST_MEMBERS = (
    ROUND140_RHS_RECORD, ROUND140_RHS_RECORD + ".stamp",
    ROUND139_RECORD, ROUND139_RECORD + ".stamp",
    round81.DEVELOPED_RECORD, round81.DEVELOPED_RECORD + ".stamp",
    round81.DEVELOPED_QCO_RECORD, round81.DEVELOPED_QCO_RECORD + ".stamp",
    "GYRE_OMIP_L2_P3_00001080_restart.nc",
    "oracle_process_budget_kt00001081.bin",
    "round140_parent_record_validation.json",
    "round140_passive_admission_plant.log",
    "round140_rhs_header_plant.log", "round140_rhs_header_plant.json",
    "round140_rhs_replay_ulp_plant.log", "round140_rhs_replay_ulp_plant.json",
    "round140_rhs_stamp_plant.log", "round140_rhs_stamp_plant.json",
    "round140_rhs_truncation_plant.log", "round140_rhs_truncation_plant.json",
)
ROUND140_RHS_REGISTRY = (
    "thickness_u", "thickness_v", "rhs_u", "rhs_v",
    "mask3_u", "mask3_v", "reciprocal_u", "reciprocal_v",
    "depth_mean_u", "depth_mean_v",
)
ROUND143_DIRECTED_REGISTRY = tuple(
    f"{arm}_{boundary}_{face}"
    for arm in ("ordinary", "rhs", "depth", "drag", "wind")
    for boundary in ("incoming", "final")
    for face in ("u", "v")
)
ROUND147_FAMILY_REGISTRY = tuple(
    f"{arm}_{boundary}_{face}"
    for arm in ("ordinary",) + round146_family.FAMILIES
    for boundary in ("incoming", "final")
    for face in ("u", "v")
)
ROUND144_WIND_REGISTRY = tuple(
    f"{arm}_{boundary}_{face}"
    for arm in ("live", "density", "stress", "inverse_depth", "all", "terminal")
    for boundary in ("incoming", "final")
    for face in ("u", "v")
)
ROUND141_RHS_REGISTRY = ROUND140_RHS_REGISTRY
ROUND142_DIRECTED_REGISTRY = (
    "ordinary_incoming_u", "ordinary_incoming_v",
    "ordinary_final_u", "ordinary_final_v",
    "directed_incoming_u", "directed_incoming_v",
    "directed_final_u", "directed_final_v",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def owned3(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2, :30]


def owned3_with_bottom(value) -> np.ndarray:
    """Keep NEMO's non-contributing ``jpk`` slot for literal ``SUM`` replay."""
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2, :]


def owned2(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2]


def native_u(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, 1:, ...]


def native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :, ...]


def bottom_value(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select NEMO's deepest wet level from an already face-staggered field."""
    require(values.shape == mask.shape and values.ndim == 3, "bad bottom-field shape")
    count = np.sum(mask, axis=-1, dtype=np.int64)
    require(np.all(count[mask.any(axis=-1)] > 0), "wet face has no wet level")
    index = np.maximum(count - 1, 0)[..., None]
    return np.take_along_axis(values, index, axis=-1)[..., 0]


def source_chain(
    rhs: np.ndarray,
    e3: np.ndarray,
    mask3: np.ndarray,
    reciprocal_ref: np.ndarray,
    inverse_depth: np.ndarray,
    drag_coefficient: np.ndarray,
    bottom_velocity: np.ndarray,
    barotropic_velocity: np.ndarray,
    rho_reciprocal: np.float64,
    stress: np.ndarray,
    coriolis: np.ndarray,
    mask2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay compiled stp2d/dynspg statements without reassociation."""
    depth = round16._source_sum(e3, rhs, mask3, reciprocal_ref)
    residual = bottom_velocity - barotropic_velocity
    drag_increment = (inverse_depth * drag_coefficient) * residual
    post_drag = depth + drag_increment
    wind_increment = (rho_reciprocal * stress) * inverse_depth
    post_wind = post_drag + wind_increment
    final = post_wind - coriolis * mask2
    return {
        "depth_mean": depth,
        "drag_increment": drag_increment,
        "post_drag": post_drag,
        "wind_increment": wind_increment,
        "post_wind": post_wind,
        "final": final,
    }


def comparison(candidate, oracle, active) -> dict[str, object]:
    return round78.comparison(
        np.asarray(candidate, dtype=np.float64),
        np.asarray(oracle, dtype=np.float64),
        np.asarray(active, dtype=bool),
    )


def _round139_take(payload: bytes, offset: int, count: int,
                   label: str) -> tuple[bytes, int]:
    """Take one exact binary segment and fail closed on truncation."""
    stop = offset + count
    require(stop <= len(payload), f"truncated Round-139 {label}")
    return payload[offset:stop], stop


def read_round139_record_bytes(payload: bytes, *,
                               expected_kt: int = ROUND139_KT) -> dict:
    """Read the native step-entry slow-forcing split without defaults."""
    magic_bytes, offset = _round139_take(payload, 0, 16, "magic")
    try:
        magic = magic_bytes.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise RuntimeError("Round-139 magic is not ASCII") from error
    raw_header, offset = _round139_take(
        payload, offset, ROUND139_HEADER_INTS * 4, "header")
    header = struct.unpack(f"={ROUND139_HEADER_INTS}i", raw_header)
    (version, kt, kmm, jpi, jpj, bits, ntsi, ntei, ntsj, ntej,
     nfields) = header
    expected = (
        1, expected_kt, ROUND139_KMM, round81.JPI, round81.JPJ, 64,
        round81.NTSI, round81.NTEI, round81.NTSJ, round81.NTEJ,
        len(ROUND139_FIELDS),
    )
    require(magic == ROUND139_MAGIC,
            f"bad Round-139 magic {magic!r}")
    require(header == expected,
            f"bad Round-139 header {(magic, *header)}")
    arrays = {}
    array_bytes = ROUND139_COUNT * 8
    for name in ROUND139_FIELDS:
        raw, offset = _round139_take(payload, offset, array_bytes, name)
        values = np.frombuffer(raw, dtype=np.float64).copy()
        require(values.size == ROUND139_COUNT,
                f"bad Round-139 {name} element count")
        values = values.reshape((round81.NX, round81.NY), order="F").T
        require(np.all(np.isfinite(values)),
                f"non-finite Round-139 {name}")
        arrays[name] = values
    require(offset == len(payload), "trailing Round-139 payload")
    return {
        "header": {
            "version": version,
            "kt": kt,
            "Kmm": kmm,
            "jpi": jpi,
            "jpj": jpj,
            "bits": bits,
            "ntsi": ntsi,
            "ntei": ntei,
            "ntsj": ntsj,
            "ntej": ntej,
            "nfields": nfields,
            "registry_level": "before",
        },
        **arrays,
    }


def read_round139_record(path: Path) -> dict:
    require(time_level_for_dump(path.name) == "before",
            "Round-139 record level is not BEFORE")
    require(path.stat().st_size == ROUND139_EXPECTED_SIZE,
            "Round-139 record size changed")
    return read_round139_record_bytes(path.read_bytes())


def read_round140_rhs_bytes(payload: bytes) -> dict[str, object]:
    """Read the developed ``NEMO_L2_R140RHS`` stream fail closed."""
    require(len(payload) == ROUND140_RHS_EXPECTED_SIZE,
            "Round-140 developed-RHS record size changed")
    magic_bytes, offset = _round139_take(payload, 0, 16, "RHS magic")
    try:
        magic = magic_bytes.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise RuntimeError("Round-140 RHS magic is not ASCII") from error
    raw_header, offset = _round139_take(payload, offset, 8 * 4, "RHS header")
    version, kt, kbb, krhs, nx, ny, nz, bits = struct.unpack(
        "=8i", raw_header)
    raw_sizes, offset = _round139_take(payload, offset, 7 * 4, "RHS sizes")
    sizes = struct.unpack("=7i", raw_sizes)
    expected_sizes = (nx * ny * nz,) * 6 + ((nx - 4) * (ny - 4),)
    require(
        (magic, version, kt, kbb, krhs, nx, ny, nz, bits, sizes)
        == (ROUND140_RHS_MAGIC, ROUND140_RHS_VERSION, ROUND139_KT,
            1, 3, 36, 26, 31, 64, expected_sizes),
        "bad Round-140 RHS header "
        + repr((magic, version, kt, kbb, krhs, nx, ny, nz, bits, sizes)),
    )

    def take_values(count: int, label: str) -> np.ndarray:
        nonlocal offset
        raw, offset = _round139_take(payload, offset, count * 8, label)
        values = np.frombuffer(raw, dtype=np.float64).copy()
        require(values.size == count, f"bad Round-140 RHS {label} count")
        require(np.all(np.isfinite(values)),
                f"non-finite Round-140 RHS {label}")
        return values

    def field3(label: str) -> np.ndarray:
        return take_values(nx * ny * nz, label).reshape(
            (nx, ny, nz), order="F").transpose(1, 0, 2)[2:-2, 2:-2, :nz - 1]

    def full2(label: str) -> np.ndarray:
        return take_values(nx * ny, label).reshape(
            (nx, ny), order="F").T[2:-2, 2:-2]

    def interior2(label: str) -> np.ndarray:
        return take_values((nx - 4) * (ny - 4), label).reshape(
            (nx - 4, ny - 4), order="F").T

    fields = {
        "e3u": field3("e3u"),
        "rhs_u": field3("rhs_u"),
        "umask": field3("umask"),
        "e3v": field3("e3v"),
        "rhs_v": field3("rhs_v"),
        "vmask": field3("vmask"),
        "depth_mean_u": interior2("depth_mean_u"),
        "depth_mean_v": interior2("depth_mean_v"),
        "r1_hu0": full2("r1_hu0"),
        "r1_hv0": full2("r1_hv0"),
        "post_drag_u": interior2("post_drag_u"),
        "post_drag_v": interior2("post_drag_v"),
        "cd_u": full2("cd_u"),
        "cd_v": full2("cd_v"),
    }
    fields["r1_rho0"] = float(take_values(1, "r1_rho0")[0])
    fields.update({
        "wind_tau_u": full2("wind_tau_u"),
        "wind_tau_v": full2("wind_tau_v"),
        "wind_r1_hu": full2("wind_r1_hu"),
        "wind_r1_hv": full2("wind_r1_hv"),
        "post_wind_u": interior2("post_wind_u"),
        "post_wind_v": interior2("post_wind_v"),
    })
    require(offset == len(payload), "trailing Round-140 RHS payload")
    require(tuple(fields) == ROUND140_RHS_FIELDS,
            "Round-140 RHS field census changed")
    return {
        "header": {
            "magic": magic, "version": version, "kt": kt,
            "Kbb": kbb, "Krhs": krhs, "nx": nx, "ny": ny,
            "nz": nz, "bits": bits, "sizes": sizes,
        },
        "fields": fields,
    }


def read_round140_rhs(path: Path) -> dict[str, object]:
    require(path.name == ROUND140_RHS_RECORD,
            "Round-140 developed-RHS record name changed")
    return read_round140_rhs_bytes(path.read_bytes())


def _round140_source_sum(e3, rhs, mask, reciprocal) -> np.ndarray:
    """Replay NEMO's SUM after the parser has removed only the jpk slot."""
    product = (e3 * rhs) * mask
    require(product.shape[-1] == 30,
            "Round-140 RHS replay physical-level count changed")
    total = np.array(product[..., 0], copy=True)
    for level in range(1, product.shape[-1]):
        total = total + product[..., level]
    return total * reciprocal


def validate_round140_rhs_replay(record: dict[str, object], *,
                                 post_wind_u=None) -> dict[str, object]:
    """Replay the two directly closed statements in the acquired stream."""
    fields = record["fields"]
    rows = {}
    for face in ("u", "v"):
        active3 = fields[f"{face}mask"] != 0.0
        active2 = active3[..., 0]
        depth = _round140_source_sum(
            fields[f"e3{face}"], fields[f"rhs_{face}"],
            fields[f"{face}mask"], fields[f"r1_h{face}0"])
        rows[f"depth_{face}"] = comparison(
            depth, fields[f"depth_mean_{face}"], active2)
        wind = ((np.float64(fields["r1_rho0"])
                 * fields[f"wind_tau_{face}"])
                * fields[f"wind_r1_h{face}"])
        final = fields[f"post_drag_{face}"] + wind
        observed = (post_wind_u if face == "u" and post_wind_u is not None
                    else fields[f"post_wind_{face}"])
        rows[f"wind_{face}"] = comparison(final, observed, active2)
    require(all(row["bit_exact"] for row in rows.values()),
            "Round-140 RHS source replay is not bit exact")
    return rows


def _round139_bits_equal(left, right) -> bool:
    left = np.ascontiguousarray(np.asarray(left, dtype=np.float64))
    right = np.ascontiguousarray(np.asarray(right, dtype=np.float64))
    require(left.shape == right.shape, "Round-139 replay extents differ")
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def validate_round139_record(fields: dict, u_mask, v_mask, *,
                             final_u=None, final_v=None) -> dict:
    """Replay compiled lines 323/324 with their recorded association."""
    u_mask = np.asarray(u_mask, dtype=np.float64)
    v_mask = np.asarray(v_mask, dtype=np.float64)
    require(
        fields["incoming_u"].shape == fields["incoming_v"].shape
        == fields["coriolis_u"].shape == fields["coriolis_v"].shape
        == fields["final_u"].shape == fields["final_v"].shape
        == u_mask.shape == v_mask.shape == (round81.NY, round81.NX),
        "Round-139 operand extents changed",
    )
    expected_u = fields["incoming_u"] - fields["coriolis_u"] * u_mask
    expected_v = fields["incoming_v"] - fields["coriolis_v"] * v_mask
    observed_u = fields["final_u"] if final_u is None else final_u
    observed_v = fields["final_v"] if final_v is None else final_v
    replay_u = _round139_bits_equal(observed_u, expected_u)
    replay_v = _round139_bits_equal(observed_v, expected_v)
    require(replay_u and replay_v,
            "Round-139 final forcing does not replay bit for bit")
    return {
        "u_bit_exact": replay_u,
        "v_bit_exact": replay_v,
        "u_wet_faces": int(np.count_nonzero(u_mask)),
        "v_wet_faces": int(np.count_nonzero(v_mask)),
    }


def _validate_round139_stamp(path: Path, producer: str, words=None) -> str:
    stamp = path.with_name(path.name + ".stamp")
    require(stamp.is_file(), "Round-139 record stamp is missing")
    actual = stamp.read_text().split() if words is None else list(words)
    digest = sha256(path)
    require(actual == [digest, producer, path.name],
            "Round-139 record stamp mismatch")
    return digest


def _verify_round139_closed_run(root: Path) -> dict[str, object]:
    """Re-admit the acquisition's closed outputs and planted violations."""
    manifest_path = root / ROUND139_MANIFEST
    require(manifest_path.is_file(), "Round-139 closed manifest is missing")
    entries = {}
    for line in manifest_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, "malformed Round-139 manifest row")
        digest, name = parts
        require(name not in entries, f"duplicate Round-139 manifest row {name}")
        path = root / name
        require(path.is_file(), f"missing Round-139 manifest member {name}")
        require(sha256(path) == digest,
                f"Round-139 manifest member changed: {name}")
        entries[name] = digest
    require(tuple(entries) == ROUND139_MANIFEST_MEMBERS,
            "Round-139 closed manifest census or order changed")

    inherited = {
        "restart": (
            "GYRE_OMIP_L2_P3_00001080_restart.nc",
            ROUND139_PARENT_RESTART_SHA256),
        "process": (
            "oracle_process_budget_kt00001081.bin",
            ROUND139_PARENT_PROCESS_SHA256),
        "external": (
            round81.DEVELOPED_RECORD, ROUND139_PARENT_EXTERNAL_SHA256),
        "qco": (
            round81.DEVELOPED_QCO_RECORD, ROUND139_PARENT_QCO_SHA256),
    }
    inherited_hashes = {}
    for label, (name, expected) in inherited.items():
        actual = sha256(root / name)
        require(actual == expected,
                f"Round-139 inherited {label} record changed")
        inherited_hashes[label] = actual

    markers = {
        "passive-admission": "STATUS PLANT-FIRED: passive-admission",
        "record-header": "STATUS PLANT-FIRED",
        "record-replay-ulp": "STATUS PLANT-FIRED",
        "record-stamp": "STATUS PLANT-FIRED",
        "record-truncation": "STATUS PLANT-FIRED",
    }
    plant_markers = {}
    for name, marker in markers.items():
        lines = (root / f"round139_{name.replace('-', '_')}_plant.log")
        if not lines.is_file():
            lines = root / f"round139_{name}_plant.log"
        require(lines.is_file(), f"Round-139 {name} plant log is missing")
        final = lines.read_text().splitlines()
        require(final and marker in final[-1],
                f"Round-139 {name} plant marker changed")
        plant_markers[name] = final[-1]

    timing = (root / "run.user.time.log").read_text().splitlines()
    stdout = (root / "run.user.stdout.log").read_text().splitlines()
    require(timing and timing[-1] == "RUN_DONE",
            "Round-139 acquisition lacks RUN_DONE")
    require(stdout and stdout[-1] == "STOP 0",
            "Round-139 acquisition lacks STOP 0")
    parent = json.loads(
        (root / "round139_parent_record_validation.json").read_text())
    child = json.loads((root / "round139_record_validation.json").read_text())
    require(parent.get("status") == "AT-BAR",
            "Round-139 parent-record validation moved")
    require(child.get("status") == "PASS",
            "Round-139 split-record validation moved")
    return {
        "manifest": str(manifest_path),
        "manifest_members": len(entries),
        "inherited_sha256": inherited_hashes,
        "plants": plant_markers,
        "completion": {"nemo": stdout[-1], "run": timing[-1]},
        "parent_record_status": parent["status"],
        "split_record_status": child["status"],
    }


def _validate_round140_registry(registry=ROUND140_OPERAND_REGISTRY) -> None:
    require(tuple(registry) == ROUND140_OPERAND_REGISTRY,
            "Round-140 developed operand registry changed")
    require(len(registry) == len(set(registry)),
            "Round-140 developed operand registry has duplicates")


def _validate_round140_rhs_registry(registry=ROUND140_RHS_REGISTRY) -> None:
    require(tuple(registry) == ROUND140_RHS_REGISTRY,
            "Round-140 developed-RHS registry changed")
    require(len(registry) == len(set(registry)),
            "Round-140 developed-RHS registry has duplicates")


def _validate_round140_rhs_stamp(path: Path, producer: str, words=None) -> str:
    stamp = path.with_name(path.name + ".stamp")
    require(stamp.is_file(), "Round-140 RHS record stamp is missing")
    actual = stamp.read_text().split() if words is None else list(words)
    digest = sha256(path)
    require(actual == [digest, producer, path.name],
            "Round-140 RHS record stamp mismatch")
    return digest


def _verify_round140_rhs_closed_run(root: Path) -> dict[str, object]:
    """Verify the closed acquisition and every inherited scientific byte."""
    manifest_path = root / ROUND140_RHS_MANIFEST
    require(manifest_path.is_file(), "Round-140 RHS manifest is missing")
    entries = {}
    for line in manifest_path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, "malformed Round-140 RHS manifest row")
        digest, name = parts
        require(name not in entries,
                f"duplicate Round-140 RHS manifest row {name}")
        path = root / name
        require(path.is_file(), f"missing Round-140 RHS manifest member {name}")
        require(sha256(path) == digest,
                f"Round-140 RHS manifest member changed: {name}")
        entries[name] = digest
    require(tuple(entries) == ROUND140_RHS_MANIFEST_MEMBERS,
            "Round-140 RHS manifest census or order changed")
    inherited = {
        ROUND139_RECORD: ROUND139_RECORD_SHA256,
        round81.DEVELOPED_RECORD: ROUND139_PARENT_EXTERNAL_SHA256,
        round81.DEVELOPED_QCO_RECORD: ROUND139_PARENT_QCO_SHA256,
        "GYRE_OMIP_L2_P3_00001080_restart.nc": ROUND139_PARENT_RESTART_SHA256,
        "oracle_process_budget_kt00001081.bin": ROUND139_PARENT_PROCESS_SHA256,
    }
    for name, expected in inherited.items():
        require(sha256(root / name) == expected,
                f"Round-140 instrument perturbed inherited {name}")
    producer = (root / "producer_commit.txt").read_text().strip()
    for name in (ROUND139_RECORD, round81.DEVELOPED_RECORD,
                 round81.DEVELOPED_QCO_RECORD):
        words = (root / f"{name}.stamp").read_text().split()
        require(words == [sha256(root / name), producer, name],
                f"Round-140 inherited stamp changed: {name}")
    parent = json.loads(
        (root / "round140_parent_record_validation.json").read_text())
    require(parent.get("status") == "AT-BAR",
            "Round-140 parent-record validation moved")
    markers = {
        "passive": ("round140_passive_admission_plant.log",
                    "STATUS PLANT-FIRED: passive-admission"),
        "header": ("round140_rhs_header_plant.log", "STATUS PLANT-FIRED"),
        "replay": ("round140_rhs_replay_ulp_plant.log", "STATUS PLANT-FIRED"),
        "stamp": ("round140_rhs_stamp_plant.log", "STATUS PLANT-FIRED"),
        "truncation": ("round140_rhs_truncation_plant.log", "STATUS PLANT-FIRED"),
    }
    seen = {}
    for name, (filename, marker) in markers.items():
        lines = (root / filename).read_text().splitlines()
        require(lines and marker in lines[-1],
                f"Round-140 RHS {name} plant marker changed")
        seen[name] = lines[-1]
    timing = (root / "run.user.time.log").read_text().splitlines()
    stdout = (root / "run.user.stdout.log").read_text().splitlines()
    require(timing and timing[-1] == "RUN_DONE",
            "Round-140 RHS acquisition lacks RUN_DONE")
    require(stdout and stdout[-1] == "STOP 0",
            "Round-140 RHS acquisition lacks STOP 0")
    return {
        "manifest": str(manifest_path),
        "manifest_members": len(entries),
        "plants": seen,
        "completion": {"nemo": stdout[-1], "run": timing[-1]},
        "parent_record_status": parent["status"],
    }


def measure_round139_record(args) -> dict[str, object]:
    """Admit the passive split record before any production comparison."""
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-139 record admission worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-139 record admission commit mismatch")
    root = args.round139_root
    record_path = root / ROUND139_RECORD
    require(record_path.is_file(), "Round-139 split record is missing")
    producer_path = root / "producer_commit.txt"
    require(producer_path.is_file(), "Round-139 producer stamp is missing")
    producer = producer_path.read_text().strip()
    require(producer.lower() == args.expect_record_commit.lower(),
            "Round-139 record producer commit changed")

    if args.plant == "record-stamp":
        planted_words = "0" * 64, producer, record_path.name
        fired = False
        try:
            _validate_round139_stamp(record_path, producer, planted_words)
        except RuntimeError:
            fired = True
        require(fired, "Round-139 stamp plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round139-record-v1",
            "status": "PLANT-FIRED",
            "plant": args.plant,
            "plant_fires": True,
            "worktree": stamp,
        }

    payload = record_path.read_bytes()
    if args.plant == "record-header":
        planted = bytearray(payload)
        planted[16:20] = struct.pack("=i", 2)
        fired = False
        try:
            read_round139_record_bytes(bytes(planted))
        except RuntimeError:
            fired = True
        require(fired, "Round-139 header plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round139-record-v1",
            "status": "PLANT-FIRED",
            "plant": args.plant,
            "plant_fires": True,
            "worktree": stamp,
        }
    if args.plant == "record-truncation":
        fired = False
        try:
            read_round139_record_bytes(payload[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-139 truncation plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round139-record-v1",
            "status": "PLANT-FIRED",
            "plant": args.plant,
            "plant_fires": True,
            "worktree": stamp,
        }

    closed_run = _verify_round139_closed_run(root)
    digest = _validate_round139_stamp(record_path, producer)
    fields = read_round139_record(record_path)
    external_path = root / round81.DEVELOPED_RECORD
    require(external_path.is_file(),
            "Round-139 inherited external record is missing")
    require(round81.sha256(external_path) == ROUND139_PARENT_EXTERNAL_SHA256,
            "Round-139 inherited external record changed")
    external = round81.read_record(
        external_path, expected_kt=ROUND139_KT, has_final_pssh=True)
    if args.plant == "record-replay-ulp":
        planted = np.array(fields["final_u"], copy=True)
        active = external["u_mask"] != 0.0
        location = tuple(int(value) for value in np.argwhere(active)[0])
        planted[location] = np.nextafter(
            planted[location], np.float64(np.inf))
        fired = False
        try:
            validate_round139_record(
                fields, external["u_mask"], external["v_mask"],
                final_u=planted)
        except RuntimeError:
            fired = True
        require(fired, "Round-139 replay-ULP plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round139-record-v1",
            "status": "PLANT-FIRED",
            "plant": args.plant,
            "plant_fires": True,
            "location": list(location),
            "worktree": stamp,
        }

    replay = validate_round139_record(
        fields, external["u_mask"], external["v_mask"])
    return {
        "format": "nemo-testcase-l2-gyre-round139-record-v1",
        "status": "PASS",
        "worktree": stamp,
        "producer_commit": producer,
        "record": str(record_path),
        "record_sha256": digest,
        "record_size": record_path.stat().st_size,
        "header": fields["header"],
        "replay": replay,
        "closed_run": closed_run,
        "plant": args.plant,
        "plant_fires": False,
    }


def measure_round140_rhs_record(args) -> dict[str, object]:
    """Admit the passive developed three-dimensional RHS record."""
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-140 RHS admission worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-140 RHS admission commit mismatch")
    root = args.round140_rhs_root
    record_path = root / ROUND140_RHS_RECORD
    require(record_path.is_file(), "Round-140 RHS record is missing")
    producer = (root / "producer_commit.txt").read_text().strip()
    require(producer.lower() == args.expect_rhs_record_commit.lower(),
            "Round-140 RHS producer commit changed")
    payload = record_path.read_bytes()
    if args.plant == "rhs-record-stamp":
        fired = False
        try:
            _validate_round140_rhs_stamp(
                record_path, producer,
                ("0" * 64, producer, record_path.name))
        except RuntimeError:
            fired = True
        require(fired, "Round-140 RHS stamp plant stayed green")
        return {"format": "nemo-testcase-l2-gyre-round140-rhs-record-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_fires": True, "worktree": stamp}
    if args.plant == "rhs-record-header":
        planted = bytearray(payload)
        planted[16:20] = struct.pack("=i", ROUND140_RHS_VERSION + 1)
        fired = False
        try:
            read_round140_rhs_bytes(bytes(planted))
        except RuntimeError:
            fired = True
        require(fired, "Round-140 RHS header plant stayed green")
        return {"format": "nemo-testcase-l2-gyre-round140-rhs-record-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_fires": True, "worktree": stamp}
    if args.plant == "rhs-record-truncation":
        fired = False
        try:
            read_round140_rhs_bytes(payload[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-140 RHS truncation plant stayed green")
        return {"format": "nemo-testcase-l2-gyre-round140-rhs-record-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_fires": True, "worktree": stamp}
    record = read_round140_rhs_bytes(payload)
    if args.plant == "rhs-record-replay-ulp":
        planted = np.array(record["fields"]["post_wind_u"], copy=True)
        active = record["fields"]["umask"][..., 0] != 0.0
        location = tuple(int(value) for value in np.argwhere(active)[0])
        planted[location] = np.nextafter(
            planted[location], np.float64(np.inf))
        fired = False
        try:
            validate_round140_rhs_replay(record, post_wind_u=planted)
        except RuntimeError:
            fired = True
        require(fired, "Round-140 RHS replay-ULP plant stayed green")
        return {"format": "nemo-testcase-l2-gyre-round140-rhs-record-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_location": list(location), "plant_fires": True,
                "worktree": stamp}
    replay = validate_round140_rhs_replay(record)
    closed = _verify_round140_rhs_closed_run(root)
    digest = _validate_round140_rhs_stamp(record_path, producer)
    return {
        "format": "nemo-testcase-l2-gyre-round140-rhs-record-v1",
        "status": "PASS", "worktree": stamp,
        "producer_commit": producer, "record": str(record_path),
        "record_sha256": digest, "record_size": len(payload),
        "header": record["header"], "replay": replay,
        "closed_run": closed, "plant": args.plant,
        "plant_fires": False,
    }


def measure_round140_developed(args) -> dict[str, object]:
    """Split the first developed non-bit forcing boundary in production."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-140 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-140 measurement commit mismatch")
    if args.plant == "developed-missing-row":
        fired = False
        try:
            _validate_round140_registry(ROUND140_OPERAND_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-140 missing-row plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round140-developed-split-v1",
            "status": "PLANT-FIRED",
            "worktree": stamp,
            "plant": args.plant,
            "plant_fires": True,
        }
    _validate_round140_registry()
    record_admission = measure_round139_record(args)
    fields = read_round139_record(args.round139_root / ROUND139_RECORD)
    external = round81.read_record(
        args.round139_root / round81.DEVELOPED_RECORD,
        expected_kt=ROUND139_KT, has_final_pssh=True)
    record_replay = validate_round139_record(
        fields, external["u_mask"], external["v_mask"])

    card, state, freshwater, surface, payload, entry = (
        round82._developed_inputs(args))
    all_cells = np.ones((round81.NY, round81.NX), dtype=bool)
    active = {
        "u": np.asarray(external["u_mask"] != 0.0, dtype=bool),
        "v": np.asarray(external["v_mask"] != 0.0, dtype=bool),
    }
    live_masks = {
        "u": native_u(state.u_mask.data),
        "v": native_v(state.v_mask.data),
    }
    mask_rows = {
        face: round82._developed_comparison(
            live_masks[face], external[f"{face}_mask"], all_cells)
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in mask_rows.values()),
            "developed model masks differ from the admitted record")

    captured, producer = _round140_callback_trace(
        args, card, state, freshwater, surface,
        jnp.asarray(payload["ssha"]))
    observer_identity = captured.trace_state_identity
    require(observer_identity["bit_exact"],
            "developed operand callback moved the returned production state")
    live = {
        "u": {
            "incoming": native_u(producer["incoming_u"]),
            "coriolis": native_u(producer["coriolis_u"]),
            "final": native_u(producer["final_u"]),
        },
        "v": {
            "incoming": native_v(producer["incoming_v"]),
            "coriolis": native_v(producer["coriolis_v"]),
            "final": native_v(producer["final_v"]),
        },
    }
    actual_external = {
        "u": gate._trace_native(
            captured.substeps["slow_u"], "slow_u")[0],
        "v": gate._trace_native(
            captured.substeps["slow_v"], "slow_v")[0],
    }
    trace_final_identity = {
        face: round82._developed_comparison(
            live[face]["final"], actual_external[face], active[face])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in trace_final_identity.values()),
            "developed producer trace differs from the external call: "
            + repr(trace_final_identity))

    rows = {
        "incoming_u": round82._developed_comparison(
            live["u"]["incoming"], fields["incoming_u"], active["u"]),
        "incoming_v": round82._developed_comparison(
            live["v"]["incoming"], fields["incoming_v"], active["v"]),
        "coriolis_u": round82._developed_comparison(
            live["u"]["coriolis"], fields["coriolis_u"], active["u"]),
        "coriolis_v": round82._developed_comparison(
            live["v"]["coriolis"], fields["coriolis_v"], active["v"]),
        "mask_u": mask_rows["u"],
        "mask_v": mask_rows["v"],
        "final_u": round82._developed_comparison(
            live["u"]["final"], fields["final_u"], active["u"]),
        "final_v": round82._developed_comparison(
            live["v"]["final"], fields["final_v"], active["v"]),
    }
    require(tuple(rows) == ROUND140_OPERAND_REGISTRY,
            "Round-140 result omitted a registered operand")
    first = next(({"boundary": name, **rows[name]}
                  for name in ROUND140_OPERAND_REGISTRY
                  if not rows[name]["bit_exact"]), None)

    if args.plant == "incoming-ulp":
        planted_native, location, direction = _round117_propagating_ulp(
            live["u"]["incoming"], live["u"]["coriolis"], active["u"])
        incoming_override = (
            jnp.asarray(_full_from_native(
                producer["incoming_u"], planted_native, "u")),
            jnp.asarray(producer["incoming_v"]),
        )
        planted_trace, planted_producer = _round140_callback_trace(
            args, card, state, freshwater, surface,
            jnp.asarray(payload["ssha"]),
            incoming_override=incoming_override)
        plant_rows = {
            "incoming_u": round82._developed_comparison(
                native_u(planted_producer["incoming_u"]),
                live["u"]["incoming"], active["u"]),
            "coriolis_u": round82._developed_comparison(
                native_u(planted_producer["coriolis_u"]),
                live["u"]["coriolis"], active["u"]),
            "final_u": round82._developed_comparison(
                native_u(planted_producer["final_u"]),
                live["u"]["final"], active["u"]),
        }
        plant_fires = bool(
            planted_trace.trace_state_identity["bit_exact"]
            and
            plant_rows["incoming_u"]["differing_cells"] == 1
            and plant_rows["coriolis_u"]["bit_exact"]
            and plant_rows["final_u"]["differing_cells"] > 0)
        require(plant_fires,
                "Round-140 incoming-U ULP did not cross production subtraction")
        return {
            "format": "nemo-testcase-l2-gyre-round140-developed-split-v1",
            "status": "PLANT-FIRED",
            "worktree": stamp,
            "plant": args.plant,
            "plant_location": list(location),
            "plant_direction": float(direction),
            "plant_rows": plant_rows,
            "plant_fires": True,
        }

    nemo_incoming_override = (
        jnp.asarray(_full_from_native(
            producer["incoming_u"], fields["incoming_u"], "u")),
        jnp.asarray(_full_from_native(
            producer["incoming_v"], fields["incoming_v"], "v")),
    )
    directed_trace, directed_producer = _round140_callback_trace(
        args, card, state, freshwater, surface,
        jnp.asarray(payload["ssha"]),
        incoming_override=nemo_incoming_override)
    directed_rows = {
        "incoming_u_vs_record": round82._developed_comparison(
            native_u(directed_producer["incoming_u"]),
            fields["incoming_u"], active["u"]),
        "incoming_v_vs_record": round82._developed_comparison(
            native_v(directed_producer["incoming_v"]),
            fields["incoming_v"], active["v"]),
        "coriolis_u_vs_baseline": round82._developed_comparison(
            native_u(directed_producer["coriolis_u"]),
            live["u"]["coriolis"], active["u"]),
        "coriolis_v_vs_baseline": round82._developed_comparison(
            native_v(directed_producer["coriolis_v"]),
            live["v"]["coriolis"], active["v"]),
        "final_u_vs_record": round82._developed_comparison(
            native_u(directed_producer["final_u"]),
            fields["final_u"], active["u"]),
        "final_v_vs_record": round82._developed_comparison(
            native_v(directed_producer["final_v"]),
            fields["final_v"], active["v"]),
    }
    predictions = {
        "incoming_u_first": bool(
            first is not None and first["boundary"] == "incoming_u"),
        "incoming_u_580_of_580": (
            rows["incoming_u"]["differing_cells"]
            == rows["incoming_u"]["cells_scored"] == 580),
        "incoming_v_570_of_570": (
            rows["incoming_v"]["differing_cells"]
            == rows["incoming_v"]["cells_scored"] == 570),
        "coriolis_pair_bit": bool(
            rows["coriolis_u"]["bit_exact"]
            and rows["coriolis_v"]["bit_exact"]),
        "mask_pair_bit": bool(
            rows["mask_u"]["bit_exact"] and rows["mask_v"]["bit_exact"]),
        "NEMO_incoming_pair_makes_final_pair_bit": bool(
            directed_trace.trace_state_identity["bit_exact"]
            and
            directed_rows["final_u_vs_record"]["bit_exact"]
            and directed_rows["final_v_vs_record"]["bit_exact"]),
        "ordinary_trace_reproduces_round138_final_pair": bool(
            rows["final_u"]["differing_cells"] == 580
            and rows["final_v"]["differing_cells"] == 570
            and rows["final_u"]["absolute_max"]
            == 4.2854247978022983e-13
            and rows["final_v"]["absolute_max"]
            == 4.4333086294645174e-13),
        "live_hook_returned_state_bit": observer_identity["bit_exact"],
    }
    return {
        "format": "nemo-testcase-l2-gyre-round140-developed-split-v1",
        "status": "MEASURED",
        "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry,
        "record_admission": record_admission,
        "record_replay": record_replay,
        "observer_state_identity": observer_identity,
        "trace_final_vs_actual_external_call": trace_final_identity,
        "first_non_bit_operand": first,
        "rows": rows,
        "NEMO_incoming_directed_arm": directed_rows,
        "predictions": predictions,
        "all_frozen_predictions_confirmed": all(predictions.values()),
        "plant": args.plant,
        "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
        },
    }


def measure_round140_rhs_developed(args) -> dict[str, object]:
    """Walk the developed slow forcing upstream through its 3-D RHS."""
    raise RuntimeError(
        "RETRACTED: materializing the upstream RHS operands moved the "
        "returned production state; no developed-RHS comparison is valid")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-140 RHS measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-140 RHS measurement commit mismatch")
    if args.plant == "rhs-missing-row":
        fired = False
        try:
            _validate_round140_rhs_registry(ROUND140_RHS_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-140 RHS missing-row plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round140-rhs-walk-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
        }
    _validate_round140_rhs_registry()
    admission = measure_round140_rhs_record(args)
    record = read_round140_rhs(args.round140_rhs_root / ROUND140_RHS_RECORD)
    fields = record["fields"]
    split = read_round139_record(args.round140_rhs_root / ROUND139_RECORD)
    boundary_calibration = {
        face: comparison(
            fields[f"post_wind_{face}"], split[f"incoming_{face}"],
            fields[f"{face}mask"][..., 0] != 0.0)
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in boundary_calibration.values()),
            "Round-140 RHS and Round-139 incoming boundaries differ")

    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    captured, producer = _round140_callback_trace(
        args, card, state, freshwater, surface, jnp.asarray(payload["ssha"]))
    require(captured.trace_state_identity["bit_exact"],
            "Round-140 RHS callback moved the returned production state")
    active = {
        "u3": fields["umask"] != 0.0,
        "v3": fields["vmask"] != 0.0,
    }
    active["u2"] = active["u3"][..., 0]
    active["v2"] = active["v3"][..., 0]
    live = {
        "thickness_u": native_u(producer["thickness_u"]),
        "thickness_v": native_v(producer["thickness_v"]),
        "rhs_u": native_u(producer["rhs_u"]),
        "rhs_v": native_v(producer["rhs_v"]),
        "mask3_u": native_u(producer["mask3_u"]),
        "mask3_v": native_v(producer["mask3_v"]),
        "reciprocal_u": np.float64(1.0) / native_u(producer["depth_u"]),
        "reciprocal_v": np.float64(1.0) / native_v(producer["depth_v"]),
        "depth_mean_u": native_u(producer["depth_mean_u"]),
        "depth_mean_v": native_v(producer["depth_mean_v"]),
    }
    oracle = {
        "thickness_u": fields["e3u"],
        "thickness_v": fields["e3v"],
        "rhs_u": fields["rhs_u"],
        "rhs_v": fields["rhs_v"],
        "mask3_u": fields["umask"],
        "mask3_v": fields["vmask"],
        "reciprocal_u": fields["r1_hu0"],
        "reciprocal_v": fields["r1_hv0"],
        "depth_mean_u": fields["depth_mean_u"],
        "depth_mean_v": fields["depth_mean_v"],
    }
    rows = {}
    for name in ROUND140_RHS_REGISTRY:
        face = "u" if name.endswith("_u") else "v"
        mask = active[f"{face}{'3' if live[name].ndim == 3 else '2'}"]
        rows[name] = comparison(live[name], oracle[name], mask)
    require(tuple(rows) == ROUND140_RHS_REGISTRY,
            "Round-140 RHS result omitted a registered operand")
    first = next(({"boundary": name, **rows[name]}
                  for name in ROUND140_RHS_REGISTRY
                  if not rows[name]["bit_exact"]), None)
    final_identity = {
        face: comparison(
            native_u(producer["final_u"]) if face == "u"
            else native_v(producer["final_v"]),
            gate._trace_native(
                captured.substeps[f"slow_{face}"], f"slow_{face}")[0],
            active[f"{face}2"])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in final_identity.values()),
            "Round-140 RHS callback differs from the ordinary external call")
    predictions = {
        "geometry_pair_bit": bool(
            rows["thickness_u"]["bit_exact"]
            and rows["thickness_v"]["bit_exact"]),
        "mask_pair_bit": bool(
            rows["mask3_u"]["bit_exact"] and rows["mask3_v"]["bit_exact"]),
        "reference_reciprocal_pair_bit": bool(
            rows["reciprocal_u"]["bit_exact"]
            and rows["reciprocal_v"]["bit_exact"]),
        "rhs_u_first": bool(first is not None and first["boundary"] == "rhs_u"),
        "callback_returned_state_bit": captured.trace_state_identity["bit_exact"],
        "same_run_post_wind_boundary_bit": bool(
            boundary_calibration["u"]["bit_exact"]
            and boundary_calibration["v"]["bit_exact"]),
    }
    return {
        "format": "nemo-testcase-l2-gyre-round140-rhs-walk-v1",
        "status": "MEASURED", "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry, "record_admission": admission,
        "record_replay": validate_round140_rhs_replay(record),
        "observer_state_identity": captured.trace_state_identity,
        "observer_final_vs_actual_external_call": final_identity,
        "same_run_boundary_calibration": boundary_calibration,
        "first_non_bit_operand": first, "rows": rows,
        "predictions": predictions,
        "all_frozen_predictions_confirmed": all(predictions.values()),
        "plant": args.plant, "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
        },
    }


@jax.jit
def _round141_depth_reduction(h_u, h_v, rhs_u, rhs_v, mask_u, mask_v):
    """Replay the production stacked depth reductions from captured RHS."""
    pair_u = jnp.sum(
        jnp.stack([h_u, rhs_u * h_u], axis=-1), axis=-2)
    pair_v = jnp.sum(
        jnp.stack([h_v, rhs_v * h_v], axis=-1), axis=-2)
    depth_u = pair_u[..., 1] / jnp.maximum(pair_u[..., 0], 1.0e-10)
    depth_v = pair_v[..., 1] / jnp.maximum(pair_v[..., 0], 1.0e-10)
    return depth_u * mask_u, depth_v * mask_v


def _round141_rhs_callback_trace(
        args, card, state, freshwater, surface, eta_after_override):
    """Capture one RHS face per compiled step and prove each callback passive."""
    rhs = {}
    finals = {}
    traces = {}
    for face in ("u", "v"):
        rhs_captures = []
        final_captures = []

        def rhs_sink(value):
            rhs_captures.append(np.asarray(value))

        def final_sink(*values):
            require(len(values) == len(ROUND140_CALLBACK_FIELDS),
                    "Round-141 final callback field census changed")
            final_captures.append({
                name: np.asarray(value)
                for name, value in zip(
                    ROUND140_CALLBACK_FIELDS, values, strict=True)
            })

        hooks = model_module._NEMOWSRK3TestHooks(
            slow_forcing_rhs_observer=rhs_sink,
            slow_forcing_rhs_observer_face=face,
            barotropic_slow_forcing_override=final_sink,
        )
        (_, _, _, _, _, trace) = round82._capture_external_context(
            args, card, state, freshwater, surface,
            eta_after_override=eta_after_override,
            traced_hooks=hooks,
            plain_hooks=model_module._NEMOWSRK3TestHooks(),
            require_state_identity=False,
        )
        require(rhs_captures, f"Round-141 RHS {face} callback did not fire")
        require(final_captures,
                f"Round-141 final {face} callback did not fire")
        for duplicate in rhs_captures[1:]:
            require(round82._pytree_identity(
                duplicate, rhs_captures[0])["bit_exact"],
                f"Round-141 RHS {face} callbacks differ")
        for duplicate in final_captures[1:]:
            require(round82._pytree_identity(
                duplicate, final_captures[0])["bit_exact"],
                f"Round-141 final {face} callbacks differ")
        rhs[face] = rhs_captures[0]
        finals[face] = final_captures[0]
        traces[face] = trace
    return traces, (rhs["u"], rhs["v"]), finals


def measure_round141_rhs_developed(args) -> dict[str, object]:
    """Score the developed completed 3-D momentum RHS in production JIT."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-141 RHS measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-141 RHS measurement commit mismatch")
    if args.plant == "rhs-missing-row":
        fired = False
        try:
            _validate_round140_rhs_registry(ROUND141_RHS_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-141 RHS missing-row plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round141-rhs-walk-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
        }
    _validate_round140_rhs_registry()
    admission = measure_round140_rhs_record(args)
    record = read_round140_rhs(args.round140_rhs_root / ROUND140_RHS_RECORD)
    fields = record["fields"]
    split = read_round139_record(args.round140_rhs_root / ROUND139_RECORD)
    active3 = {
        "u": fields["umask"] != 0.0,
        "v": fields["vmask"] != 0.0,
    }
    active2 = {face: active3[face][..., 0] for face in ("u", "v")}
    boundary_calibration = {
        face: comparison(
            fields[f"post_wind_{face}"], split[f"incoming_{face}"],
            active2[face])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in boundary_calibration.values()),
            "Round-140 RHS and Round-139 incoming boundaries differ")

    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    minimal_traces, captured_rhs, captured_final = _round141_rhs_callback_trace(
        args, card, state, freshwater, surface, jnp.asarray(payload["ssha"]))
    control_trace, control_final = _round140_callback_trace(
        args, card, state, freshwater, surface, jnp.asarray(payload["ssha"]))
    require(control_trace.trace_state_identity["bit_exact"],
            "Round-140 control callback moved the returned production state")
    callback_identity = {
        face: {
            name: comparison(
                captured_final[face][name], control_final[name],
                np.ones_like(captured_final[face][name], dtype=bool))
            for name in ROUND140_CALLBACK_FIELDS
        }
        for face in ("u", "v")
    }
    observer_state_bit = all(
        trace.trace_state_identity["bit_exact"]
        for trace in minimal_traces.values())
    observer_external_bit = all(
        row["bit_exact"] for rows in callback_identity.values()
        for row in rows.values())
    if not (observer_state_bit and observer_external_bit):
        return {
            "format": "nemo-testcase-l2-gyre-round141-rhs-walk-v1",
            "status": "REFUTED-NONPASSIVE-OBSERVER", "worktree": stamp,
            "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
            "entry": entry, "record_admission": admission,
            "record_replay": validate_round140_rhs_replay(record),
            "observer_state_identity": {
                face: trace.trace_state_identity
                for face, trace in minimal_traces.items()},
            "control_state_identity": control_trace.trace_state_identity,
            "observer_external_boundary_identity": callback_identity,
            "same_run_boundary_calibration": boundary_calibration,
            "first_non_bit_operand": None,
            "rows": {},
            "predictions": {
                "observer_state_bit": observer_state_bit,
                "observer_external_boundaries_bit": observer_external_bit,
            },
            "all_frozen_predictions_confirmed": False,
            "scientific_rows_withheld": True,
            "plant": args.plant, "plant_fires": False,
            "scope": {
                "production_physics_changed": False,
                "day_240_carry": "UNMEASURED",
                "DINO": "NO-PRODUCTION-CHANGE",
                "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
                "OVERFLOW": "NO-PRODUCTION-CHANGE",
                "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
            },
        }

    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, card.recipe.grid)
    mask_u = state.u_mask.data
    mask_v = state.v_mask.data
    rhs_u, rhs_v = captured_rhs
    depth_u, depth_v = jax.device_get(_round141_depth_reduction(
        h_u, h_v, jnp.asarray(rhs_u), jnp.asarray(rhs_v), mask_u, mask_v))
    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), 1.0e-10)
    H_v = jnp.maximum(jnp.sum(h_v, axis=-1), 1.0e-10)
    expected_masks = gate.expected_masks(card)
    live = {
        "thickness_u": native_u(h_u),
        "thickness_v": native_v(h_v),
        "rhs_u": native_u(rhs_u),
        "rhs_v": native_v(rhs_v),
        "mask3_u": np.asarray(expected_masks["u"], dtype=np.float64),
        "mask3_v": np.asarray(expected_masks["v"], dtype=np.float64),
        "reciprocal_u": native_u(np.float64(1.0) / np.asarray(H_u)),
        "reciprocal_v": native_v(np.float64(1.0) / np.asarray(H_v)),
        "depth_mean_u": native_u(depth_u),
        "depth_mean_v": native_v(depth_v),
    }
    oracle = {
        "thickness_u": fields["e3u"],
        "thickness_v": fields["e3v"],
        "rhs_u": fields["rhs_u"],
        "rhs_v": fields["rhs_v"],
        "mask3_u": fields["umask"],
        "mask3_v": fields["vmask"],
        "reciprocal_u": fields["r1_hu0"],
        "reciprocal_v": fields["r1_hv0"],
        "depth_mean_u": fields["depth_mean_u"],
        "depth_mean_v": fields["depth_mean_v"],
    }
    if args.plant == "rhs-observer-ulp":
        planted = np.array(oracle["rhs_u"], copy=True)
        location = tuple(int(value) for value in np.argwhere(active3["u"])[0])
        planted[location] = np.nextafter(
            planted[location], np.float64(np.inf))
        row = comparison(live["rhs_u"], planted, active3["u"])
        baseline = comparison(live["rhs_u"], oracle["rhs_u"], active3["u"])
        require(row != baseline,
                "Round-141 RHS observer ULP plant did not move its row")
        return {
            "format": "nemo-testcase-l2-gyre-round141-rhs-walk-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
            "plant_location": list(location),
            "ordinary_row": baseline, "planted_row": row,
        }

    rows = {}
    for name in ROUND141_RHS_REGISTRY:
        face = "u" if name.endswith("_u") else "v"
        mask = active3[face] if live[name].ndim == 3 else active2[face]
        rows[name] = comparison(live[name], oracle[name], mask)
    require(tuple(rows) == ROUND141_RHS_REGISTRY,
            "Round-141 RHS result omitted a registered operand")
    first = next(({"boundary": name, **rows[name]}
                  for name in ROUND141_RHS_REGISTRY
                  if not rows[name]["bit_exact"]), None)
    predictions = {
        "geometry_pair_bit": bool(
            rows["thickness_u"]["bit_exact"]
            and rows["thickness_v"]["bit_exact"]),
        "mask_pair_bit": bool(
            rows["mask3_u"]["bit_exact"]
            and rows["mask3_v"]["bit_exact"]),
        "reference_reciprocal_pair_bit": bool(
            rows["reciprocal_u"]["bit_exact"]
            and rows["reciprocal_v"]["bit_exact"]),
        "rhs_u_first": bool(first is not None and first["boundary"] == "rhs_u"),
        "rhs_v_non_bit": not rows["rhs_v"]["bit_exact"],
        "callback_returned_state_bit": observer_state_bit,
        "callback_external_boundaries_bit": observer_external_bit,
        "same_run_post_wind_boundary_bit": all(
            row["bit_exact"] for row in boundary_calibration.values()),
    }
    return {
        "format": "nemo-testcase-l2-gyre-round141-rhs-walk-v1",
        "status": "MEASURED", "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry, "record_admission": admission,
        "record_replay": validate_round140_rhs_replay(record),
        "observer_state_identity": {
            face: trace.trace_state_identity
            for face, trace in minimal_traces.items()},
        "control_state_identity": control_trace.trace_state_identity,
        "observer_external_boundary_identity": callback_identity,
        "same_run_boundary_calibration": boundary_calibration,
        "first_non_bit_operand": first, "rows": rows,
        "predictions": predictions,
        "all_frozen_predictions_confirmed": all(predictions.values()),
        "plant": args.plant, "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "day_240_carry": "UNMEASURED",
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
        },
    }


def measure_round142_rhs_directed(args) -> dict[str, object]:
    """Substitute NEMO's completed 3-D RHS without observing model RHS."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-142 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-142 measurement commit mismatch")
    if args.plant == "rhs-directed-missing-row":
        fired = False
        try:
            _validate_round142_registry(ROUND142_DIRECTED_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-142 missing-row plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round142-rhs-directed-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
        }
    _validate_round142_registry()
    admission = measure_round140_rhs_record(args)
    record = read_round140_rhs(args.round140_rhs_root / ROUND140_RHS_RECORD)
    fields = record["fields"]
    split = read_round139_record(args.round140_rhs_root / ROUND139_RECORD)
    active3 = {
        "u": fields["umask"] != 0.0,
        "v": fields["vmask"] != 0.0,
    }


def _validate_round143_registry(registry=ROUND143_DIRECTED_REGISTRY) -> None:
    require(tuple(registry) == ROUND143_DIRECTED_REGISTRY,
            "Round-143 directed-row registry changed")


def _round143_depth_ulp(value, active) -> tuple[np.ndarray, tuple[int, ...]]:
    """Plant one ULP in a finite active post-depth U word."""
    planted = np.asarray(value, dtype=np.float64).copy()
    active = np.asarray(active, dtype=bool)
    require(planted.shape == active.shape,
            "Round-143 depth plant extents differ")
    candidates = np.argwhere(active & np.isfinite(planted) & (planted != 0.0))
    require(candidates.size > 0, "Round-143 depth plant has no active word")
    location = tuple(int(i) for i in candidates[
        np.argmax(np.abs(planted[tuple(candidates.T)]))])
    planted[location] = np.nextafter(planted[location], np.inf)
    require(np.count_nonzero(
        planted.view(np.uint64) != np.asarray(value).view(np.uint64)) == 1,
        "Round-143 depth plant did not change exactly one word")
    return planted, location


def measure_round143_downstream_directed(args) -> dict[str, object]:
    """Substitute recorded depth, drag, and wind boundaries in source order."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-143 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-143 measurement commit mismatch")
    if args.plant == "downstream-missing-row":
        fired = False
        try:
            _validate_round143_registry(ROUND143_DIRECTED_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-143 missing-row plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round143-downstream-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
        }
    _validate_round143_registry()
    admission = measure_round140_rhs_record(args)
    record = read_round140_rhs(args.round140_rhs_root / ROUND140_RHS_RECORD)
    fields = record["fields"]
    split = read_round139_record(args.round140_rhs_root / ROUND139_RECORD)
    active3 = {
        "u": fields["umask"] != 0.0,
        "v": fields["vmask"] != 0.0,
    }
    active2 = {face: active3[face][..., 0] for face in ("u", "v")}
    calibration = {
        face: comparison(
            fields[f"post_wind_{face}"], split[f"incoming_{face}"],
            active2[face])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in calibration.values()),
            "Round-143 terminal record calibration is not BIT")

    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    eta_after = jnp.asarray(payload["ssha"])
    arms = {}
    arms["ordinary"] = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after)
    arms["rhs"] = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after,
        rhs_override=(jnp.asarray(fields["rhs_u"]),
                      jnp.asarray(fields["rhs_v"])))
    arms["depth"] = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after,
        depth_override=(jnp.asarray(fields["depth_mean_u"]),
                        jnp.asarray(fields["depth_mean_v"])))
    arms["drag"] = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after,
        drag_override=(jnp.asarray(fields["post_drag_u"]),
                       jnp.asarray(fields["post_drag_v"])))
    ordinary_producer = arms["ordinary"][1]
    wind_override = tuple(
        jnp.asarray(_full_from_native(
            ordinary_producer[f"incoming_{face}"],
            fields[f"post_wind_{face}"], face))
        for face in ("u", "v")
    )
    arms["wind"] = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after,
        incoming_override=wind_override)

    def row(producer, boundary: str, face: str) -> dict:
        native = (native_u(producer[f"{boundary}_{face}"])
                  if face == "u" else native_v(producer[f"{boundary}_{face}"]))
        return comparison(native, split[f"{boundary}_{face}"], active2[face])

    rows = {
        f"{arm}_{boundary}_{face}": row(producer, boundary, face)
        for arm, (_, producer) in arms.items()
        for boundary in ("incoming", "final")
        for face in ("u", "v")
    }
    require(tuple(rows) == ROUND143_DIRECTED_REGISTRY,
            "Round-143 result omitted a registered row")
    expected = {
        "ordinary_incoming_u": 4.2854247978022983e-13,
        "ordinary_incoming_v": 4.433308633699682e-13,
        "ordinary_final_u": 4.2854247978022983e-13,
        "ordinary_final_v": 4.4333086294645174e-13,
        "rhs_incoming_u": 1.3307884389737387e-13,
        "rhs_incoming_v": 1.089946783815101e-13,
        "rhs_final_u": 1.3307884389737387e-13,
        "rhs_final_v": 1.089946783815101e-13,
    }
    require(all(rows[name]["absolute_max"] == value
                for name, value in expected.items()),
            "Round-143 controls do not reproduce Round 142")

    callback_identity = {}
    external_identity = {}
    for arm, (trace, producer) in arms.items():
        callback_identity[arm] = trace.trace_state_identity
        external_identity[arm] = {
            face: comparison(
                native_u(producer[f"final_{face}"])
                if face == "u" else native_v(producer[f"final_{face}"]),
                gate._trace_native(
                    trace.substeps[f"slow_{face}"], f"slow_{face}")[0],
                active2[face])
            for face in ("u", "v")
        }
    require(all(identity["bit_exact"]
                for identity in callback_identity.values()),
            "Round-143 callback moved its plain arm")
    require(all(row_["bit_exact"]
                for arm_rows in external_identity.values()
                for row_ in arm_rows.values()),
            "Round-143 callback differs from an actual external-call operand")

    rhs_max = {
        face: rows[f"rhs_incoming_{face}"]["absolute_max"]
        for face in ("u", "v")
    }
    depth_magnitude = bool(all(
        rows[f"depth_incoming_{face}"]["absolute_max"] <= 1.0e-18
        and rhs_max[face]
        >= 1.0e3 * rows[f"depth_incoming_{face}"]["absolute_max"]
        for face in ("u", "v")))
    drag_exact = bool(all(
        rows[f"drag_incoming_{face}"]["bit_exact"]
        for face in ("u", "v")))
    wind_exact = bool(all(
        rows[f"wind_incoming_{face}"]["bit_exact"]
        for face in ("u", "v")))
    first_closing_family = (
        "depth_average" if depth_magnitude else
        ("drag" if drag_exact else ("wind" if wind_exact else None)))

    if args.plant == "downstream-depth-ulp":
        planted_u, location = _round143_depth_ulp(
            fields["depth_mean_u"], active2["u"])
        planted_trace, planted = _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            depth_override=(jnp.asarray(planted_u),
                            jnp.asarray(fields["depth_mean_v"])))
        require(planted_trace.trace_state_identity["bit_exact"],
                "Round-143 ULP callback moved its plain arm")
        planted_row = comparison(
            native_u(planted["incoming_u"]),
            native_u(arms["depth"][1]["incoming_u"]), active2["u"])
        require(planted_row["differing_cells"] > 0,
                "Round-143 one-ULP depth plant did not reach incoming forcing")
        return {
            "format": "nemo-testcase-l2-gyre-round143-downstream-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
            "plant_location": list(location),
            "plant_downstream_row": planted_row,
        }

    return {
        "format": "nemo-testcase-l2-gyre-round143-downstream-v1",
        "status": "MEASURED", "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry, "record_admission": admission,
        "record_replay": validate_round140_rhs_replay(record),
        "terminal_record_calibration": calibration,
        "callback_state_identity": callback_identity,
        "callback_final_vs_external_call": external_identity,
        "rows": rows,
        "predictions": {
            "depth_removes_magnitude": depth_magnitude,
            "drag_then_live_wind_is_bit": drag_exact,
            "post_wind_terminal_is_bit": wind_exact,
        },
        "first_closing_family": first_closing_family,
        "plant": args.plant, "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "day_240_carry": "UNMEASURED",
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
        },
    }
    active2 = {face: active3[face][..., 0] for face in ("u", "v")}
    calibration = {
        face: comparison(
            fields[f"post_wind_{face}"], split[f"incoming_{face}"],
            active2[face])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in calibration.values()),
            "Round-140 RHS and Round-139 incoming boundaries differ")

    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    eta_after = jnp.asarray(payload["ssha"])
    ordinary_trace, ordinary = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after)
    require(ordinary_trace.trace_state_identity["bit_exact"],
            "Round-142 ordinary callback moved the production state")
    override = (jnp.asarray(fields["rhs_u"]), jnp.asarray(fields["rhs_v"]))
    directed_trace, directed = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after,
        rhs_override=override)
    require(directed_trace.trace_state_identity["bit_exact"],
            "Round-142 directed callback moved its production arm")

    def row(producer, boundary: str, face: str) -> dict:
        native = (native_u(producer[f"{boundary}_{face}"])
                  if face == "u" else native_v(producer[f"{boundary}_{face}"]))
        return comparison(native, split[f"{boundary}_{face}"], active2[face])

    rows = {
        "ordinary_incoming_u": row(ordinary, "incoming", "u"),
        "ordinary_incoming_v": row(ordinary, "incoming", "v"),
        "ordinary_final_u": row(ordinary, "final", "u"),
        "ordinary_final_v": row(ordinary, "final", "v"),
        "directed_incoming_u": row(directed, "incoming", "u"),
        "directed_incoming_v": row(directed, "incoming", "v"),
        "directed_final_u": row(directed, "final", "u"),
        "directed_final_v": row(directed, "final", "v"),
    }
    require(tuple(rows) == ROUND142_DIRECTED_REGISTRY,
            "Round-142 result omitted a registered row")
    ordinary_reproduced = bool(
        rows["ordinary_incoming_u"]["absolute_max"]
        == 4.2854247978022983e-13
        and rows["ordinary_incoming_v"]["absolute_max"]
        == 4.433308633699682e-13
        and rows["ordinary_final_u"]["absolute_max"]
        == 4.2854247978022983e-13
        and rows["ordinary_final_v"]["absolute_max"]
        == 4.4333086294645174e-13)
    require(ordinary_reproduced,
            "Round-142 default path does not reproduce Round 140")

    external_identity = {}
    for arm, trace, producer in (
            ("ordinary", ordinary_trace, ordinary),
            ("directed", directed_trace, directed)):
        external_identity[arm] = {
            face: comparison(
                native_u(producer[f"final_{face}"])
                if face == "u" else native_v(producer[f"final_{face}"]),
                gate._trace_native(
                    trace.substeps[f"slow_{face}"], f"slow_{face}")[0],
                active2[face])
            for face in ("u", "v")
        }
    require(all(row_["bit_exact"]
                for arm_rows in external_identity.values()
                for row_ in arm_rows.values()),
            "Round-142 callback differs from an actual external-call operand")

    state_impact = round82._pytree_identity(
        directed_trace.state_after, ordinary_trace.state_after)
    require(not state_impact["bit_exact"],
            "Round-142 RHS override did not execute through production")
    exact_closure = bool(
        rows["directed_incoming_u"]["bit_exact"]
        and rows["directed_incoming_v"]["bit_exact"])
    family_confirmed = bool(all(
        rows[f"directed_incoming_{face}"]["absolute_max"] <= 1.0e-18
        and rows[f"ordinary_incoming_{face}"]["absolute_max"]
        >= 1.0e3 * rows[f"directed_incoming_{face}"]["absolute_max"]
        for face in ("u", "v")))

    if args.plant == "rhs-directed-ulp":
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, card.recipe.z_coord,
            min_water_column_m=(
                card.recipe.model_config.min_water_column_m))
        h_u = native_u(min_cell_to_uface(h_k))
        planted_u, location = _round142_propagating_rhs_ulp(
            fields, active3["u"], h_u)
        planted_trace, planted = _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            rhs_override=(jnp.asarray(planted_u), jnp.asarray(fields["rhs_v"])))
        require(planted_trace.trace_state_identity["bit_exact"],
                "Round-142 ULP callback moved its production arm")
        planted_row = comparison(
            native_u(planted["incoming_u"]),
            native_u(directed["incoming_u"]), active2["u"])
        plant_fires = planted_row["differing_cells"] > 0
        require(plant_fires,
                "Round-142 one-ULP RHS plant did not reach incoming forcing")
        return {
            "format": "nemo-testcase-l2-gyre-round142-rhs-directed-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
            "plant_location": list(location),
            "plant_downstream_row": planted_row,
        }

    return {
        "format": "nemo-testcase-l2-gyre-round142-rhs-directed-v1",
        "status": "MEASURED", "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry, "record_admission": admission,
        "record_replay": validate_round140_rhs_replay(record),
        "same_run_boundary_calibration": calibration,
        "ordinary_callback_state_identity": ordinary_trace.trace_state_identity,
        "directed_callback_state_identity": directed_trace.trace_state_identity,
        "callback_final_vs_external_call": external_identity,
        "directed_state_vs_ordinary": state_impact,
        "rows": rows,
        "predictions": {
            "ordinary_reproduces_round140": ordinary_reproduced,
            "directed_incoming_pair_bit": exact_closure,
            "completed_rhs_family_magnitude_confirmed": family_confirmed,
        },
        "exact_closure_prediction_confirmed": exact_closure,
        "completed_rhs_family_magnitude_confirmed": family_confirmed,
        "conditional_cumulative_acquisition_required": family_confirmed,
        "plant": args.plant, "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "day_240_carry": "UNMEASURED",
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
        },
    }


def _validate_round144_registry(registry=ROUND144_WIND_REGISTRY) -> None:
    require(tuple(registry) == ROUND144_WIND_REGISTRY,
            "Round-144 wind-row registry changed")


def _round144_stress_ulp(fields, active) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose one recorded U-stress ULP that changes NEMO's written result."""
    stress = np.asarray(fields["wind_tau_u"], dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(stress.shape == active.shape,
            "Round-144 stress plant extents differ")
    rho = np.float64(fields["r1_rho0"])
    depth = np.asarray(fields["wind_r1_hu"], dtype=np.float64)
    post_drag = np.asarray(fields["post_drag_u"], dtype=np.float64)
    baseline = post_drag + (rho * stress) * depth
    for location_array in np.argwhere(active & np.isfinite(stress)):
        location = tuple(int(index) for index in location_array)
        for direction in (np.inf, -np.inf):
            candidate = np.nextafter(stress[location], direction)
            changed = post_drag[location] + (
                rho * candidate) * depth[location]
            if changed.view(np.uint64) != baseline[location].view(np.uint64):
                planted = stress.copy()
                planted[location] = candidate
                require(np.count_nonzero(
                    planted.view(np.uint64) != stress.view(np.uint64)) == 1,
                    "Round-144 stress plant did not change exactly one word")
                return planted, location
    raise RuntimeError("no one-ULP U-stress plant changes the written result")


def _round145_inverse_depth_ulp(
    fields, active,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose one recorded U reciprocal ULP that changes the wind result."""
    depth = np.asarray(fields["wind_r1_hu"], dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(depth.shape == active.shape,
            "Round-145 inverse-depth plant extents differ")
    rho = np.float64(fields["r1_rho0"])
    stress = np.asarray(fields["wind_tau_u"], dtype=np.float64)
    post_drag = np.asarray(fields["post_drag_u"], dtype=np.float64)
    baseline = post_drag + (rho * stress) * depth
    for location_array in np.argwhere(active & np.isfinite(depth)):
        location = tuple(int(index) for index in location_array)
        for direction in (np.inf, -np.inf):
            candidate = np.nextafter(depth[location], direction)
            changed = post_drag[location] + (
                rho * stress[location]) * candidate
            if changed.view(np.uint64) != baseline[location].view(np.uint64):
                planted = depth.copy()
                planted[location] = candidate
                require(np.count_nonzero(
                    planted.view(np.uint64) != depth.view(np.uint64)) == 1,
                    "Round-145 reciprocal plant did not change one word")
                return planted, location
    raise RuntimeError("no one-ULP U reciprocal changes the written result")


def measure_round144_wind_operands(args) -> dict[str, object]:
    """Walk the developed wind operands through the production-jitted step."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-144 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-144 measurement commit mismatch")
    if args.plant == "wind-missing-row":
        fired = False
        try:
            _validate_round144_registry(ROUND144_WIND_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-144 missing-row plant stayed green")
        return {
            "format": "nemo-testcase-l2-gyre-round144-wind-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
        }
    _validate_round144_registry()
    if args.round145_wind_routing:
        inherited = measure_round140_rhs_record(args)
        require(inherited["status"] == "PASS",
                "Round-145 RHS record admission did not pass")
    else:
        inherited = measure_round143_downstream_directed(args)
        require(inherited["status"] == "MEASURED",
                "Round-143 inherited controls did not measure")
    record = read_round140_rhs(args.round140_rhs_root / ROUND140_RHS_RECORD)
    fields = record["fields"]
    split = read_round139_record(args.round140_rhs_root / ROUND139_RECORD)
    active3 = {
        "u": fields["umask"] != 0.0,
        "v": fields["vmask"] != 0.0,
    }
    active2 = {face: active3[face][..., 0] for face in ("u", "v")}
    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    eta_after = jnp.asarray(payload["ssha"])
    drag_override = (
        jnp.asarray(fields["post_drag_u"]),
        jnp.asarray(fields["post_drag_v"]),
    )
    density = np.float64(fields["r1_rho0"])
    stresses = (
        jnp.asarray(fields["wind_tau_u"]),
        jnp.asarray(fields["wind_tau_v"]),
    )
    inverse_depths = (
        jnp.asarray(fields["wind_r1_hu"]),
        jnp.asarray(fields["wind_r1_hv"]),
    )
    arms = {
        "live": _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override),
        "density": _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override,
            wind_operand_override=(density, None, None)),
        "stress": _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override,
            wind_operand_override=(None, stresses, None)),
        "inverse_depth": _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override,
            wind_operand_override=(None, None, inverse_depths)),
        "all": _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override,
            wind_operand_override=(density, stresses, inverse_depths)),
    }
    live_producer = arms["live"][1]
    terminal_override = tuple(
        jnp.asarray(_full_from_native(
            live_producer[f"incoming_{face}"],
            fields[f"post_wind_{face}"], face))
        for face in ("u", "v")
    )
    arms["terminal"] = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after,
        incoming_override=terminal_override)

    def row(producer, boundary: str, face: str) -> dict:
        native = (native_u(producer[f"{boundary}_{face}"])
                  if face == "u" else native_v(producer[f"{boundary}_{face}"]))
        reference = np.asarray(split[f"{boundary}_{face}"])
        active_face = np.asarray(active2[face], dtype=bool)
        result = comparison(native, reference, active_face)
        absolute = np.abs(np.asarray(native) - reference)
        selected = absolute[active_face]
        result["absolute_rms"] = float(np.sqrt(np.mean(selected * selected)))
        different = (
            np.asarray(native).view(np.uint64) != reference.view(np.uint64))
        locations = np.argwhere(active_face & different)
        result["first_unequal_index"] = (
            [int(index) for index in locations[0]]
            if locations.size else None)
        return result

    rows = {
        f"{arm}_{boundary}_{face}": row(producer, boundary, face)
        for arm, (_, producer) in arms.items()
        for boundary in ("incoming", "final")
        for face in ("u", "v")
    }
    require(tuple(rows) == ROUND144_WIND_REGISTRY,
            "Round-144 result omitted a registered row")
    callback_identity = {
        arm: trace.trace_state_identity for arm, (trace, _) in arms.items()
    }
    require(all(value["bit_exact"] for value in callback_identity.values()),
            "Round-144 callback moved its plain arm")
    external_identity = {
        arm: {
            face: comparison(
                native_u(producer[f"final_{face}"])
                if face == "u" else native_v(producer[f"final_{face}"]),
                gate._trace_native(
                    trace.substeps[f"slow_{face}"], f"slow_{face}")[0],
                active2[face])
            for face in ("u", "v")
        }
        for arm, (trace, producer) in arms.items()
    }
    require(all(row_["bit_exact"]
                for arm_rows in external_identity.values()
                for row_ in arm_rows.values()),
            "Round-144 callback differs from an external-call operand")

    substitution_effect = {
        family: {
            face: comparison(
                native_u(arms[family][1][f"incoming_{face}"])
                if face == "u" else native_v(
                    arms[family][1][f"incoming_{face}"]),
                native_u(live_producer[f"incoming_{face}"])
                if face == "u" else native_v(
                    live_producer[f"incoming_{face}"]),
                active2[face])
            for face in ("u", "v")
        }
        for family in ("density", "stress", "inverse_depth", "all")
    }
    first_moving_family = next((
        family for family in ("density", "stress", "inverse_depth")
        if any(not row_["bit_exact"]
               for row_ in substitution_effect[family].values())), None)
    first_closing_family = next((
        family for family in ("density", "stress", "inverse_depth", "all")
        if all(rows[f"{family}_incoming_{face}"]["bit_exact"]
               for face in ("u", "v"))), None)
    if (args.round145_wind_routing
            and all(rows[f"live_incoming_{face}"]["bit_exact"]
                    for face in ("u", "v"))):
        first_closing_family = "live"

    if args.plant == "wind-stress-ulp":
        planted_u, location = _round144_stress_ulp(fields, active2["u"])
        planted_trace, planted = _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override,
            wind_operand_override=(
                density, (jnp.asarray(planted_u), stresses[1]),
                inverse_depths))
        require(planted_trace.trace_state_identity["bit_exact"],
                "Round-144 stress-ULP callback moved its plain arm")
        planted_row = comparison(
            native_u(planted["incoming_u"]),
            native_u(arms["all"][1]["incoming_u"]), active2["u"])
        require(planted_row["differing_cells"] > 0,
                "Round-144 stress ULP did not reach incoming forcing")
        return {
            "format": "nemo-testcase-l2-gyre-round144-wind-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
            "plant_location": list(location),
            "plant_downstream_row": planted_row,
        }
    if args.plant == "wind-inverse-depth-ulp":
        planted_u, location = _round145_inverse_depth_ulp(fields, active2["u"])
        planted_trace, planted = _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            drag_override=drag_override,
            wind_operand_override=(
                None, None, (jnp.asarray(planted_u), inverse_depths[1])))
        require(planted_trace.trace_state_identity["bit_exact"],
                "Round-145 reciprocal-ULP callback moved its plain arm")
        planted_row = comparison(
            native_u(planted["incoming_u"]),
            native_u(live_producer["incoming_u"]), active2["u"])
        require(planted_row["differing_cells"] > 0,
                "Round-145 reciprocal ULP did not reach incoming forcing")
        return {
            "format": "nemo-testcase-l2-gyre-round145-wind-routing-v1",
            "status": "PLANT-FIRED", "worktree": stamp,
            "plant": args.plant, "plant_fires": True,
            "plant_location": list(location),
            "plant_downstream_row": planted_row,
        }

    terminal_exact = all(
        rows[f"terminal_incoming_{face}"]["bit_exact"]
        for face in ("u", "v"))
    require(terminal_exact, "Round-144 terminal calibration is not BIT")
    return {
        "format": "nemo-testcase-l2-gyre-round144-wind-v1",
        "status": "MEASURED", "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry, "inherited_round143": inherited,
        "rows": rows, "substitution_effect": substitution_effect,
        "callback_state_identity": callback_identity,
        "callback_final_vs_external_call": external_identity,
        "first_moving_family": first_moving_family,
        "first_closing_family": first_closing_family,
        "predictions": {
            "density_inert": all(
                row_["bit_exact"]
                for row_ in substitution_effect["density"].values()),
            "stress_first": first_moving_family == "stress",
            "stress_closes": all(
                rows[f"stress_incoming_{face}"]["bit_exact"]
                for face in ("u", "v")),
            "inverse_depth_closes": all(
                rows[f"inverse_depth_incoming_{face}"]["bit_exact"]
                for face in ("u", "v")),
            "all_inputs_close": all(
                rows[f"all_incoming_{face}"]["bit_exact"]
                for face in ("u", "v")),
            "terminal_is_bit": terminal_exact,
        },
        "plant": args.plant, "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "day_240_carry": "UNMEASURED",
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only",
        },
    }


def round117_source_order_accumulators(
    hpg_u, hpg_v, ldf_u, ldf_v, vor_u, vor_v,
    keg_u, keg_v, zad_u, zad_v,
):
    """Materialize the compiled stage-1 accumulator order under JIT."""
    hpg_u = jax.lax.optimization_barrier(hpg_u)
    hpg_v = jax.lax.optimization_barrier(hpg_v)
    ldf_u = jax.lax.optimization_barrier(hpg_u + ldf_u)
    ldf_v = jax.lax.optimization_barrier(hpg_v + ldf_v)
    vor_u = jax.lax.optimization_barrier(ldf_u + vor_u)
    vor_v = jax.lax.optimization_barrier(ldf_v + vor_v)
    keg_u = jax.lax.optimization_barrier(vor_u + keg_u)
    keg_v = jax.lax.optimization_barrier(vor_v + keg_v)
    zad_u = jax.lax.optimization_barrier(keg_u + zad_u)
    zad_v = jax.lax.optimization_barrier(keg_v + zad_v)
    return {
        "after_hpg_u": hpg_u,
        "after_hpg_v": hpg_v,
        "after_ldf_u": ldf_u,
        "after_ldf_v": ldf_v,
        "after_vor_u": vor_u,
        "after_vor_v": vor_v,
        "after_keg_u": keg_u,
        "after_keg_v": keg_v,
        "after_zad_u": zad_u,
        "after_zad_v": zad_v,
        "after_adv_u": zad_u,
        "after_adv_v": zad_v,
    }


def round117_first_nonbit(rows: dict[str, dict[str, dict]]) -> dict | None:
    """Return the first cumulative boundary, with U ordered before V."""
    for boundary in ROUND117_BOUNDARIES:
        for face in ("u", "v"):
            row = rows[face][boundary]
            if not row["bit_exact"]:
                return {"boundary": boundary, "face": face, **row}
    return None


def _full_from_native(full, native, face: str) -> np.ndarray:
    """Replace only the owned native face extent, retaining excluded halos."""
    result = np.array(full, dtype=np.float64, copy=True)
    native = np.asarray(native, dtype=np.float64)
    if face == "u":
        require(result[:, 1:].shape == native.shape,
                "U native/full extents disagree")
        result[:, 1:] = native
    elif face == "v":
        require(result[1:, :].shape == native.shape,
                "V native/full extents disagree")
        result[1:, :] = native
    else:
        raise ValueError(f"unknown face {face!r}")
    return result


def _round117_live_trace(
    card, seeded, freshwater, surface, execution_mode: str, *,
    incoming_override=None, final_override=None, association_arm=False,
    zad_w_override=None, eta_after_override=None,
):
    hooks = model_module._NEMOWSRK3TestHooks(
        expose_live_stage_operands=True,
        slow_forcing_incoming_override=incoming_override,
        barotropic_slow_forcing_override=final_override,
        stage1_zad_w_override=zad_w_override,
        nemo_stage_rhs_accumulation_order_arm=association_arm,
    )
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    model.prime_step_caches(seeded)
    return jax.device_get(round82._execute_step(
        model, seeded, card.dt_s, freshwater, surface, execution_mode,
        eta_after_override=eta_after_override))


def _round140_callback_trace(
    args, card, state, freshwater, surface, eta_after_override, *,
    incoming_override=None, rhs_override=None, depth_override=None,
    drag_override=None, wind_operand_override=None,
):
    """Capture slow operands without changing the production return value."""
    captures = []

    def sink(*values):
        require(len(values) == len(ROUND140_CALLBACK_FIELDS),
                "Round-140 callback field census changed")
        captures.append({
            name: np.asarray(value)
            for name, value in zip(
                ROUND140_CALLBACK_FIELDS, values, strict=True)
        })

    traced_hooks = model_module._NEMOWSRK3TestHooks(
        slow_forcing_incoming_override=incoming_override,
        slow_forcing_rhs_override=rhs_override,
        slow_forcing_depth_override=depth_override,
        slow_forcing_drag_override=drag_override,
        slow_forcing_wind_operand_override=wind_operand_override,
        barotropic_slow_forcing_override=sink,
    )
    plain_hooks = model_module._NEMOWSRK3TestHooks(
        slow_forcing_incoming_override=incoming_override,
        slow_forcing_rhs_override=rhs_override,
        slow_forcing_depth_override=depth_override,
        slow_forcing_drag_override=drag_override,
        slow_forcing_wind_operand_override=wind_operand_override,
    )
    (_, _, _, _, _, trace) = round82._capture_external_context(
        args, card, state, freshwater, surface,
        eta_after_override=eta_after_override,
        traced_hooks=traced_hooks, plain_hooks=plain_hooks)
    require(captures, "Round-140 slow-forcing callback did not fire")
    for duplicate in captures[1:]:
        require(round82._pytree_identity(
            duplicate, captures[0])["bit_exact"],
            "Round-140 slow-forcing callbacks differ")
    return trace, captures[0]


def _validate_round142_registry(registry=ROUND142_DIRECTED_REGISTRY) -> None:
    require(tuple(registry) == ROUND142_DIRECTED_REGISTRY,
            "Round-142 directed-row registry changed")


@jax.jit
def _round142_u_depth_reduction(h_u, rhs_u, mask_u):
    """Use the production U stacked-sum expression for plant selection."""
    pair = jnp.sum(jnp.stack([h_u, rhs_u * h_u], axis=-1), axis=-2)
    return pair[..., 1] / jnp.maximum(pair[..., 0], 1.0e-10) * mask_u


def _round142_propagating_rhs_ulp(
        fields, active, h_u) -> tuple[np.ndarray, tuple[int, ...]]:
    """Plant one wet U-RHS word that survives the model depth reduction."""
    rhs = np.asarray(fields["rhs_u"], dtype=np.float64)
    h_u = np.asarray(h_u, dtype=np.float64)
    require(rhs.shape == h_u.shape == active.shape,
            "Round-142 plant extents differ")
    mask2 = np.asarray(active[..., 0], dtype=np.float64)
    baseline = np.asarray(_round142_u_depth_reduction(
        jnp.asarray(h_u), jnp.asarray(rhs), jnp.asarray(mask2)))
    for k in range(rhs.shape[-1]):
        planted_level = np.array(rhs, copy=True)
        direction = np.where(
            planted_level[..., k] >= 0.0,
            np.float64(np.inf), np.float64(-np.inf))
        planted_level[..., k] = np.nextafter(
            planted_level[..., k], direction)
        changed = np.asarray(_round142_u_depth_reduction(
            jnp.asarray(h_u), jnp.asarray(planted_level), jnp.asarray(mask2)))
        changed_bits = changed.view(np.uint64) != baseline.view(np.uint64)
        locations = np.argwhere(changed_bits & active[..., k])
        if locations.size:
            j, i = (int(value) for value in locations[0])
            planted = np.array(rhs, copy=True)
            planted[j, i, k] = planted_level[j, i, k]
            single = np.asarray(_round142_u_depth_reduction(
                jnp.asarray(h_u), jnp.asarray(planted), jnp.asarray(mask2)))
            require(single[j, i].view(np.uint64)
                    != baseline[j, i].view(np.uint64),
                    "Round-142 selected ULP does not survive alone")
            return planted, (j, i, k)
    raise RuntimeError("no one-ULP U-RHS plant survives the model depth sum")


@jax.jit
def _round117_subtract(incoming_u, incoming_v, coriolis_u, coriolis_v,
                       mask_u, mask_v):
    """Isolated JIT replay; deliberately not labelled production."""
    return (
        (incoming_u - coriolis_u) * mask_u,
        (incoming_v - coriolis_v) * mask_v,
    )


def _admit_round64(args) -> tuple[dict, dict]:
    admission = json.loads(args.round64_admission.read_text())
    require(admission.get("verdict") == "PASS", "Round-64 admission failed")
    require(
        (
            admission.get("byte_identical_records"),
            len(admission.get("classified_changed_records", [])),
            admission.get("admitted_difference_count"),
        ) == (43, 20, 132),
        "Round-64 inherited-record census changed",
    )
    producer = (args.round64_root / "producer_commit.txt").read_text().strip()
    require(producer == ROUND64_PRODUCER, "Round-64 producer commit changed")
    stage_path = args.round64_root / "oracle_momstage_kt00000002_s1.bin"
    original_stage = args.round46_root / stage_path.name
    require(stage_path.is_file() and original_stage.is_file(), "missing kt=2 stage record")
    classified = [
        row for row in admission["classified_changed_records"]
        if row["record"] == stage_path.name
    ]
    require(len(classified) == 1, "kt=2 stage record lacks one admission classification")
    stage_admission = classified[0]
    require(stage_admission["consumed_equal"], "kt=2 stage consumed projection differs")
    require(stage_admission["owned_field_differences"] == [],
            "kt=2 stage has an owned-field difference")
    require(
        stage_admission["reason_counts"] == {
            "halo": 7,
            "owned_defined_violation": 0,
            "owned_undefined_region": 0,
            "owned_undefined_slot": 0,
        },
        "kt=2 stage admission reasons changed",
    )
    used_fields = {
        "after_adv_u", "after_adv_v", "e3u_0", "e3v_0", "umask", "vmask",
        "u_Kmm", "v_Kmm", "uu_b_Kmm", "vv_b_Kmm", "has_ldf",
    }
    require(used_fields <= set(stage_admission["compared_fields"]),
            "a consumed kt=2 stage field is outside admission coverage")
    slow_path = args.round64_root / "oracle_slow_forcing_kt00000001.bin"
    stage = round46.read_stage(stage_path)
    slow = round16.read_slow_forcing(slow_path)
    require(stage["header"]["kt"] == 2 and stage["header"]["stage"] == 1,
            "wrong kt/stage record")
    require(stage["arrays"]["has_ldf"] == 1.0, "kt=2 RHS lacks final LDF row")
    return stage, slow


def _admit_round117_direct(args, external: dict) -> dict[str, object]:
    """Admit and parse the same-run kt=2 slow/pre-loop record pair."""
    admission = json.loads(args.preloop_admission.read_text())
    require(admission.get("verdict") == "PASS",
            "Round-117 direct-record admission failed")
    require(
        (
            admission.get("byte_identical_records"),
            len(admission.get("classified_changed_records", [])),
            admission.get("admitted_difference_count"),
        ) == (46, 20, 132),
        "Round-117 inherited-record census changed",
    )
    producer = (args.preloop_root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_preloop_record_commit == ROUND117_PRODUCER,
            "Round-117 direct-record producer changed")
    validation = json.loads(
        (args.preloop_root / "round117_preloop_validation.json").read_text())
    require(validation.get("status") == "PASS",
            "Round-117 direct-record gate did not pass")
    require(validation.get("producer_commit") == producer,
            "Round-117 validation producer changed")

    slow_path = args.preloop_root / round117_record.SLOW_RECORD
    preloop_path = args.preloop_root / round117_record.PRELOOP_RECORD
    for path in (slow_path, preloop_path):
        parts = path.with_name(path.name + ".stamp").read_text().split()
        require(len(parts) == 3, f"malformed direct-record stamp for {path.name}")
        require(parts == [sha256(path), producer, path.name],
                f"direct-record stamp moved for {path.name}")
        registered = validation["records"][path.name]
        require(registered["sha256"] == sha256(path),
                f"direct-record validation hash moved for {path.name}")

    slow = round117_record.read_slow_record(slow_path)
    preloop = round117_record.read_preloop_record(preloop_path)
    fields = preloop["fields"]
    for face in ("u", "v"):
        incoming = fields[f"incoming_{face}"]
        coriolis = fields[f"cor_{face}"][2:-2, 2:-2]
        mask = fields[f"{face}_mask"][2:-2, 2:-2]
        final = fields[f"final_{face}"]
        require(np.array_equal(slow["fields"][f"post_wind_{face}"], incoming),
                f"same-run slow/pre-loop {face.upper()} boundary moved")
        require(np.array_equal(incoming - coriolis * mask, final),
                f"compiled pre-loop {face.upper()} subtract did not replay")
        require(np.array_equal(final, external[f"slow_{face}"][0]),
                f"pre-loop/Round-81 {face.upper()} final boundary moved")
    return {
        "admission": admission,
        "validation": validation,
        "slow": slow,
        "preloop": preloop,
        "producer_commit": producer,
    }


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-83 measurement worktree is dirty")
    require(stamp["commit"] == args.expect_commit, "Round-83 commit stamp mismatch")

    stage, static = _admit_round64(args)
    bt = round82._admit(args)
    base, card, seeded, captured = round78._context(args)
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    trace = captured.slow_forcing_operands
    substeps = captured.substeps
    masks = gate.expected_masks(card)
    active = {
        "u3": np.asarray(masks["u"], dtype=bool),
        "v3": np.asarray(masks["v"], dtype=bool),
    }
    active["u2"] = active["u3"][..., 0]
    active["v2"] = active["v3"][..., 0]

    stage_arrays = stage["arrays"]
    oracle = {
        "u": {
            "rhs": owned3(stage_arrays["after_adv_u"]),
            "e3": owned3(stage_arrays["e3u_0"]),
            "mask3": owned3(stage_arrays["umask"]),
            "reciprocal_ref": np.asarray(static["r1_hu0"], dtype=np.float64),
            "inverse_depth": np.asarray(bt["inverse_depth_u"][0], dtype=np.float64),
            "drag_coefficient": np.asarray(bt["drag_coefficient_u"][0], dtype=np.float64),
            "bottom_velocity": bottom_value(
                owned3(stage_arrays["u_Kmm"]), owned3(stage_arrays["umask"]) != 0.0),
            "barotropic_velocity": owned2(stage_arrays["uu_b_Kmm"]),
            "stress": None,
            "coriolis": np.asarray(bt["cor_u"][0], dtype=np.float64),
            "final": np.asarray(bt["slow_u"][0], dtype=np.float64),
        },
        "v": {
            "rhs": owned3(stage_arrays["after_adv_v"]),
            "e3": owned3(stage_arrays["e3v_0"]),
            "mask3": owned3(stage_arrays["vmask"]),
            "reciprocal_ref": np.asarray(static["r1_hv0"], dtype=np.float64),
            "inverse_depth": np.asarray(bt["inverse_depth_v"][0], dtype=np.float64),
            "drag_coefficient": np.asarray(bt["drag_coefficient_v"][0], dtype=np.float64),
            "bottom_velocity": bottom_value(
                owned3(stage_arrays["v_Kmm"]), owned3(stage_arrays["vmask"]) != 0.0),
            "barotropic_velocity": owned2(stage_arrays["vv_b_Kmm"]),
            "stress": None,
            "coriolis": np.asarray(bt["cor_v"][0], dtype=np.float64),
            "final": np.asarray(bt["slow_v"][0], dtype=np.float64),
        },
    }
    live = {
        "u": {
            "rhs": native_u(trace["du_dt"]),
            "e3": native_u(trace["h_u"]),
            "mask3": active["u3"].astype(np.float64),
            "reciprocal_ref": np.float64(1.0) / native_u(trace["H_u"]),
            "inverse_depth": gate._trace_native(substeps["inverse_depth_u"], "inverse_depth_u")[0],
            "drag_coefficient": gate._trace_native(
                substeps["drag_coefficient_u"], "drag_coefficient_u")[0],
            "bottom_velocity": bottom_value(native_u(seeded.u.data), active["u3"]),
            "barotropic_velocity": gate._trace_native(substeps["u_entry"], "u_entry")[0],
            "stress": native_u(trace["wind_tau_u"]),
            "coriolis": gate._trace_native(substeps["cor_u"], "cor_u")[0],
        },
        "v": {
            "rhs": native_v(trace["dv_dt"]),
            "e3": native_v(trace["h_v"]),
            "mask3": active["v3"].astype(np.float64),
            "reciprocal_ref": np.float64(1.0) / native_v(trace["H_v"]),
            "inverse_depth": gate._trace_native(substeps["inverse_depth_v"], "inverse_depth_v")[0],
            "drag_coefficient": gate._trace_native(
                substeps["drag_coefficient_v"], "drag_coefficient_v")[0],
            "bottom_velocity": bottom_value(native_v(seeded.v.data), active["v3"]),
            "barotropic_velocity": gate._trace_native(substeps["v_entry"], "v_entry")[0],
            "stress": native_v(trace["wind_tau_v"]),
            "coriolis": gate._trace_native(substeps["cor_v"], "cor_v")[0],
        },
    }
    # The inherited slow-forcing stream is kt=1; its stress is not silently
    # reused at kt=2.  No direct kt=2 stress record exists.  Use the live kt=2
    # value only as a cross-record calibration operand, and withhold a direct
    # wind-stress identity claim.  The replay must still recover NEMO's final
    # recorded forcing exactly or the joined record is refused.
    oracle["u"]["stress"] = live["u"]["stress"]
    oracle["v"]["stress"] = live["v"]["stress"]
    rho_reciprocal = np.float64(static["r1_rho0"])
    require(float(trace["wind_r1_rho0"]) == float(rho_reciprocal),
            "live/oracle density reciprocal differs")

    ordinary_plant_target = {
        "e3-ulp": comparison(live["u"]["e3"], oracle["u"]["e3"], active["u3"]),
        "rhs-ulp": comparison(live["u"]["rhs"], oracle["u"]["rhs"], active["u3"]),
        "final-ulp": comparison(
            source_chain(
                **{key: value for key, value in oracle["u"].items() if key != "final"},
                rho_reciprocal=rho_reciprocal,
                mask2=active["u2"].astype(np.float64),
            )["final"],
            oracle["u"]["final"],
            active["u2"],
        ),
    }

    plant_detail = None
    if args.plant == "e3-ulp":
        delta = np.where(
            active["u3"], np.abs(live["u"]["e3"] - oracle["u"]["e3"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["e3"] = np.array(oracle["u"]["e3"], copy=True)
        direction = np.inf if oracle["u"]["e3"][at] >= live["u"]["e3"][at] else -np.inf
        oracle["u"]["e3"][at] = np.nextafter(oracle["u"]["e3"][at], direction)
        plant_detail = {"field": "e3_u", "location": list(map(int, at))}
    elif args.plant == "rhs-ulp":
        delta = np.where(
            active["u3"], np.abs(live["u"]["rhs"] - oracle["u"]["rhs"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["rhs"] = np.array(oracle["u"]["rhs"], copy=True)
        direction = np.inf if oracle["u"]["rhs"][at] >= live["u"]["rhs"][at] else -np.inf
        oracle["u"]["rhs"][at] = np.nextafter(oracle["u"]["rhs"][at], direction)
        plant_detail = {"field": "rhs_u", "location": list(map(int, at))}
    elif args.plant == "final-ulp":
        replay = source_chain(
            **{key: value for key, value in oracle["u"].items() if key != "final"},
            rho_reciprocal=rho_reciprocal,
            mask2=active["u2"].astype(np.float64),
        )["final"]
        delta = np.where(
            active["u2"], np.abs(replay - oracle["u"]["final"]), -np.inf)
        at = np.unravel_index(np.argmax(delta), delta.shape)
        oracle["u"]["final"] = np.array(oracle["u"]["final"], copy=True)
        direction = np.inf if oracle["u"]["final"][at] >= replay[at] else -np.inf
        oracle["u"]["final"][at] = np.nextafter(oracle["u"]["final"][at], direction)
        plant_detail = {"field": "slow_u", "location": list(map(int, at))}

    rows = {}
    cross_record = {}
    reference_geometry_arms = {}
    for face in ("u", "v"):
        active3 = active[f"{face}3"]
        active2 = active[f"{face}2"]
        for name in oracle[face]:
            if name == "final":
                continue
            require(np.asarray(oracle[face][name]).shape == np.asarray(live[face][name]).shape,
                    f"{face}.{name} live/oracle shape mismatch")
        oracle_chain = source_chain(
            **{key: value for key, value in oracle[face].items() if key != "final"},
            rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        live_chain = source_chain(
            **live[face], rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        reference_geometry_inputs = dict(live[face])
        for name in ("e3", "mask3", "reciprocal_ref"):
            reference_geometry_inputs[name] = oracle[face][name]
        reference_geometry_chain = source_chain(
            **reference_geometry_inputs,
            rho_reciprocal=rho_reciprocal,
            mask2=active2.astype(np.float64),
        )
        cross_record[face] = comparison(oracle_chain["final"], oracle[face]["final"], active2)
        reference_geometry_arms[face] = {
            "depth_mean": comparison(
                reference_geometry_chain["depth_mean"], oracle_chain["depth_mean"], active2),
            "final_slow_forcing": comparison(
                reference_geometry_chain["final"], oracle[face]["final"], active2),
            "movement_from_current_depth_mean": comparison(
                reference_geometry_chain["depth_mean"], live_chain["depth_mean"], active2),
        }
        rows[face] = {
            "e3": comparison(live[face]["e3"], oracle[face]["e3"], active3),
            "mask": comparison(live[face]["mask3"], oracle[face]["mask3"], active3),
            "three_dimensional_rhs": comparison(live[face]["rhs"], oracle[face]["rhs"], active3),
            "reference_depth_reciprocal": comparison(
                live[face]["reciprocal_ref"], oracle[face]["reciprocal_ref"], active2),
            "depth_mean": comparison(live_chain["depth_mean"], oracle_chain["depth_mean"], active2),
            "drag_coefficient": comparison(
                live[face]["drag_coefficient"], oracle[face]["drag_coefficient"], active2),
            "inverse_depth": comparison(
                live[face]["inverse_depth"], oracle[face]["inverse_depth"], active2),
            "bottom_velocity": comparison(
                live[face]["bottom_velocity"], oracle[face]["bottom_velocity"], active2),
            "barotropic_velocity": comparison(
                live[face]["barotropic_velocity"], oracle[face]["barotropic_velocity"], active2),
            "post_drag": comparison(live_chain["post_drag"], oracle_chain["post_drag"], active2),
            "post_wind": comparison(live_chain["post_wind"], oracle_chain["post_wind"], active2),
            "coriolis": comparison(live[face]["coriolis"], oracle[face]["coriolis"], active2),
            "final_slow_forcing": comparison(live_chain["final"], oracle[face]["final"], active2),
            "oracle_rhs_substitution_final": comparison(
                oracle_chain["final"], oracle[face]["final"], active2),
        }

    cross_exact = all(row["bit_exact"] for row in cross_record.values())
    first = None
    source_order = (
        "e3", "mask", "three_dimensional_rhs", "reference_depth_reciprocal",
        "depth_mean", "drag_coefficient", "inverse_depth", "bottom_velocity",
        "barotropic_velocity", "post_drag", "post_wind",
        "coriolis", "final_slow_forcing",
    )
    for boundary in source_order:
        for face in ("u", "v"):
            if first is None and not rows[face][boundary]["bit_exact"]:
                first = {"boundary": boundary, "face": face, **rows[face][boundary]}

    confirmed = bool(
        args.plant == "none"
        and cross_exact
        and first is not None
        and first["boundary"] == "three_dimensional_rhs"
        and all(rows[face]["oracle_rhs_substitution_final"]["bit_exact"] for face in ("u", "v"))
    )
    planted_row = {
        "e3-ulp": rows["u"]["e3"],
        "rhs-ulp": rows["u"]["three_dimensional_rhs"],
        "final-ulp": cross_record["u"],
    }.get(args.plant)
    plant_fires = bool(
        args.plant != "none"
        and planted_row != ordinary_plant_target[args.plant]
    )
    status = "CONFIRMED" if confirmed else ("PLANT_FIRED" if plant_fires else "REFUTED")
    return {
        "format": "nemo-testcase-l2-gyre-round83-slow-forcing-walk-v1",
        "status": status,
        "worktree": stamp,
        "execution_regime": "production-jit-cpu-fp64-x64-libm",
        "round64_stage_sha256": sha256(args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
        "round81_btstep_sha256": round81.sha256(args.record_root / round81.RECORD),
        "cross_record_replay": cross_record,
        "kt2_wind_stress_identity": "WITHHELD_NO_DIRECT_RECORD",
        "reference_geometry_arm": reference_geometry_arms,
        "first_non_bit_statement": first,
        "rows": rows,
        "plant": args.plant,
        "plant_detail": plant_detail,
        "plant_fires": plant_fires,
    }


def _round117_propagating_ulp(incoming, coriolis, mask):
    """Find one incoming ULP that survives the written subtract-and-mask."""
    incoming = np.asarray(incoming, dtype=np.float64)
    coriolis = np.asarray(coriolis, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    ordinary = (incoming - coriolis) * mask.astype(np.float64)
    for location in map(tuple, np.argwhere(mask)):
        for direction in (np.float64(np.inf), np.float64(-np.inf)):
            planted = np.array(incoming, copy=True)
            planted[location] = np.nextafter(planted[location], direction)
            changed = (planted - coriolis) * mask.astype(np.float64)
            if changed[location].view(np.uint64) != ordinary[location].view(np.uint64):
                return planted, tuple(int(index) for index in location), direction
    raise RuntimeError("no active one-ULP incoming plant survives subtraction")


def _round119_propagating_hpg_ulp(hpg, ldf, vor, keg, zad, mask):
    """Find one HPG ULP that survives every compiled accumulator boundary."""
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (hpg, ldf, vor, keg, zad)
    ]
    mask = np.asarray(mask, dtype=bool)
    require(all(value.shape == arrays[0].shape for value in arrays),
            "Round-119 term extents differ")
    require(mask.shape == arrays[0].shape, "Round-119 plant mask differs")
    for location in map(tuple, np.argwhere(mask)):
        ordinary = []
        value = arrays[0][location]
        ordinary.append(value)
        for term in arrays[1:]:
            value = value + term[location]
            ordinary.append(value)
        for direction in (np.float64(np.inf), np.float64(-np.inf)):
            planted = np.array(arrays[0], copy=True)
            value = np.nextafter(planted[location], direction)
            planted[location] = value
            changed = [value.view(np.uint64) != ordinary[0].view(np.uint64)]
            for index, term in enumerate(arrays[1:], start=1):
                value = value + term[location]
                changed.append(
                    value.view(np.uint64) != ordinary[index].view(np.uint64))
            if all(changed):
                return planted, tuple(int(index) for index in location), direction
    raise RuntimeError(
        "no active one-ULP HPG plant survives all source-order boundaries")


def _float64_word(value) -> dict[str, object]:
    """Serialize one fp64 value without losing its exact machine word."""
    scalar = np.asarray(value, dtype=np.float64).reshape(())
    bits = int(scalar.view(np.uint64))
    return {
        "value": float(scalar),
        "uint64": bits,
        "hex": f"0x{bits:016x}",
    }


def _ordered_float64_word(value) -> int:
    """Map an fp64 word to a monotone integer for signed ULP differences."""
    bits = int(np.asarray(value, dtype=np.float64).reshape(()).view(np.uint64))
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def _signed_ulp_difference(left, right) -> int:
    """Return the signed representable-word distance ``left - right``."""
    return _ordered_float64_word(left) - _ordered_float64_word(right)


def _round120_keg_ulp_addend(addend, after_vor, zad, location):
    """Plant one KEG ULP that survives the local KEG and ZAD additions."""
    addend = np.asarray(addend, dtype=np.float64)
    after_vor = np.asarray(after_vor, dtype=np.float64)
    zad = np.asarray(zad, dtype=np.float64)
    require(addend.shape == after_vor.shape == zad.shape,
            "Round-120 KEG plant extents differ")
    ordinary_keg = after_vor[location] + addend[location]
    ordinary_zad = ordinary_keg + zad[location]
    for direction in (np.float64(np.inf), np.float64(-np.inf)):
        planted = np.array(addend, copy=True)
        planted[location] = np.nextafter(planted[location], direction)
        changed_keg = after_vor[location] + planted[location]
        changed_zad = changed_keg + zad[location]
        if (changed_keg.view(np.uint64) != ordinary_keg.view(np.uint64)
                and changed_zad.view(np.uint64) != ordinary_zad.view(np.uint64)):
            return planted, direction
    raise RuntimeError(
        "no one-ULP KEG addend plant survives the KEG and ZAD additions")


def measure_round117(args) -> dict[str, object]:
    """Split the current-tip producer in the existing admitted Round-83 gate."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-117 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-117 commit stamp mismatch")

    stage, static = _admit_round64(args)
    external = round82._admit(args)
    direct = _admit_round117_direct(
        args, external) if (
            args.round118 or args.round119 or args.round120 or args.round121
        ) else None
    (base, card, seeded, freshwater, surface,
     captured) = round82._context(args)
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    ordinary = _round117_live_trace(
        card, seeded, freshwater, surface, args.execution_mode)
    source_order = (
        _round117_live_trace(
            card, seeded, freshwater, surface, args.execution_mode,
            association_arm=("stage1-source-order", None))
        if args.round119 or args.round120 or args.round121 else None)
    trace_identity = round82._pytree_identity(
        ordinary.state_after, captured.plain_state)
    require(trace_identity["bit_exact"],
            "producer trace moved the returned production state")

    producer = ordinary.slow_forcing_producer
    masks = gate.expected_masks(card)
    active = {
        "u3": np.asarray(masks["u"], dtype=bool),
        "v3": np.asarray(masks["v"], dtype=bool),
        "t3": np.asarray(masks["T"], dtype=bool),
    }
    active["u2"] = active["u3"][..., 0]
    active["v2"] = active["v3"][..., 0]
    active["t2"] = active["t3"][..., 0]
    require(np.array_equal(external["u_mask"] != 0.0, active["u2"]),
            "Round-81 U mask differs from the live native mask")
    require(np.array_equal(external["v_mask"] != 0.0, active["v2"]),
            "Round-81 V mask differs from the live native mask")

    live_split = {
        "u": {
            "incoming": native_u(producer["incoming_u"]),
            "coriolis": native_u(producer["coriolis_u"]),
            "final": native_u(producer["final_u"]),
        },
        "v": {
            "incoming": native_v(producer["incoming_v"]),
            "coriolis": native_v(producer["coriolis_v"]),
            "final": native_v(producer["final_v"]),
        },
    }
    if direct is not None:
        direct_fields = direct["preloop"]["fields"]
        oracle_split = {
            face: {
                "direct_incoming": np.asarray(
                    direct_fields[f"incoming_{face}"], dtype=np.float64),
                "direct_coriolis": np.asarray(
                    direct_fields[f"cor_{face}"][2:-2, 2:-2],
                    dtype=np.float64),
                "final": np.asarray(
                    direct_fields[f"final_{face}"], dtype=np.float64),
            }
            for face in ("u", "v")
        }
    else:
        oracle_split = {
            "u": {
                "substep1_coriolis_proxy": np.asarray(
                    external["cor_u"][0], dtype=np.float64),
                "final": np.asarray(external["slow_u"][0], dtype=np.float64),
            },
            "v": {
                "substep1_coriolis_proxy": np.asarray(
                    external["cor_v"][0], dtype=np.float64),
                "final": np.asarray(external["slow_v"][0], dtype=np.float64),
            },
        }
        for face in ("u", "v"):
            mask = active[f"{face}2"].astype(np.float64)
            oracle_split[face]["proxy_incoming_preimage"] = (
                oracle_split[face]["final"]
                + oracle_split[face]["substep1_coriolis_proxy"] * mask)

    actual_external_final = {
        "u": gate._trace_native(
            captured.substeps["slow_u"], "slow_u")[0],
        "v": gate._trace_native(
            captured.substeps["slow_v"], "slow_v")[0],
    }
    trace_final_identity = {
        face: comparison(
            live_split[face]["final"], actual_external_final[face],
            active[f"{face}2"])
        for face in ("u", "v")
    }
    require(all(row["bit_exact"] for row in trace_final_identity.values()),
            "WRITE-only producer trace changed the actual external forcing")

    isolated_final_u, isolated_final_v = jax.device_get(_round117_subtract(
        jnp.asarray(live_split["u"]["incoming"]),
        jnp.asarray(live_split["v"]["incoming"]),
        jnp.asarray(live_split["u"]["coriolis"]),
        jnp.asarray(live_split["v"]["coriolis"]),
        jnp.asarray(active["u2"], dtype=np.float64),
        jnp.asarray(active["v2"], dtype=np.float64),
    ))
    if direct is not None:
        split_rows = {
            face: {
                "incoming_vs_direct_record": comparison(
                    live_split[face]["incoming"],
                    oracle_split[face]["direct_incoming"],
                    active[f"{face}2"]),
                "preloop_coriolis_vs_direct_record": comparison(
                    live_split[face]["coriolis"],
                    oracle_split[face]["direct_coriolis"],
                    active[f"{face}2"]),
                "final_vs_direct_record": comparison(
                    live_split[face]["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
                "isolated_subtract_vs_production_final": comparison(
                    isolated_final_u if face == "u" else isolated_final_v,
                    live_split[face]["final"], active[f"{face}2"]),
            }
            for face in ("u", "v")
        }
        split_order = (
            "incoming_vs_direct_record",
            "preloop_coriolis_vs_direct_record",
            "final_vs_direct_record",
        )
    else:
        split_rows = {
            face: {
                "incoming_vs_substep_proxy_preimage": comparison(
                    live_split[face]["incoming"],
                    oracle_split[face]["proxy_incoming_preimage"],
                    active[f"{face}2"]),
                "preloop_coriolis_vs_substep1_proxy": comparison(
                    live_split[face]["coriolis"],
                    oracle_split[face]["substep1_coriolis_proxy"],
                    active[f"{face}2"]),
                "final_vs_direct_record": comparison(
                    live_split[face]["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
                "isolated_subtract_vs_production_final": comparison(
                    isolated_final_u if face == "u" else isolated_final_v,
                    live_split[face]["final"], active[f"{face}2"]),
            }
            for face in ("u", "v")
        }
        split_order = (
            "incoming_vs_substep_proxy_preimage",
            "preloop_coriolis_vs_substep1_proxy",
            "final_vs_direct_record",
        )
    proxy_first = None
    for boundary in split_order:
        for face in ("u", "v"):
            row = split_rows[face][boundary]
            if proxy_first is None and not row["bit_exact"]:
                proxy_first = {"boundary": boundary, "face": face, **row}
    direct_first = next((
        {"boundary": boundary, "face": face, **split_rows[face][boundary]}
        for boundary in split_order
        for face in ("u", "v")
        if not split_rows[face][boundary]["bit_exact"]
    ), None) if direct is not None else next((
        {"boundary": "final_vs_direct_record", "face": face,
         **split_rows[face]["final_vs_direct_record"]}
        for face in ("u", "v")
        if not split_rows[face]["final_vs_direct_record"]["bit_exact"]
    ), None)
    cancellation = None
    if direct is not None:
        cancellation = {}
        for face in ("u", "v"):
            mask = active[f"{face}2"]
            incoming_error = (
                live_split[face]["incoming"]
                - oracle_split[face]["direct_incoming"])[mask]
            coriolis_error = (
                live_split[face]["coriolis"]
                - oracle_split[face]["direct_coriolis"])[mask]
            final_error = (
                live_split[face]["final"]
                - oracle_split[face]["final"])[mask]
            component_l1 = float(
                np.sum(np.abs(incoming_error), dtype=np.float64)
                + np.sum(np.abs(coriolis_error), dtype=np.float64))
            final_l1 = float(np.sum(np.abs(final_error), dtype=np.float64))
            cancellation[face] = {
                "incoming_error_max": float(
                    np.max(np.abs(incoming_error), initial=0.0)),
                "coriolis_error_max": float(
                    np.max(np.abs(coriolis_error), initial=0.0)),
                "final_error_max": float(
                    np.max(np.abs(final_error), initial=0.0)),
                "component_l1": component_l1,
                "final_l1": final_l1,
                "cancellation_fraction": (
                    0.0 if component_l1 == 0.0 else
                    float(1.0 - final_l1 / component_l1)),
                "subtract_cancelling_cells": int(np.count_nonzero(
                    (incoming_error != 0.0) & (coriolis_error != 0.0)
                    & (np.signbit(incoming_error)
                       == np.signbit(coriolis_error)))),
            }

    if args.plant == "incoming-ulp":
        planted_native, location, direction = _round117_propagating_ulp(
            live_split["u"]["incoming"], live_split["u"]["coriolis"],
            active["u2"])
        incoming_override = (
            jnp.asarray(_full_from_native(
                producer["incoming_u"], planted_native, "u")),
            jnp.asarray(producer["incoming_v"]),
        )
        planted = _round117_live_trace(
            card, seeded, freshwater, surface, args.execution_mode,
            incoming_override=incoming_override)
        planted_producer = planted.slow_forcing_producer
        plant_rows = {
            "incoming_u": comparison(
                native_u(planted_producer["incoming_u"]),
                live_split["u"]["incoming"], active["u2"]),
            "coriolis_u": comparison(
                native_u(planted_producer["coriolis_u"]),
                live_split["u"]["coriolis"], active["u2"]),
            "final_u": comparison(
                native_u(planted_producer["final_u"]),
                live_split["u"]["final"], active["u2"]),
        }
        plant_fires = bool(
            plant_rows["incoming_u"]["differing_cells"] == 1
            and plant_rows["coriolis_u"]["bit_exact"]
            and plant_rows["final_u"]["differing_cells"] > 0)
        require(plant_fires, "incoming ULP plant did not cross production subtract")
        return {
            "format": (
                "nemo-testcase-l2-gyre-round119-association-walk-v1"
                if args.round119 else
                ("nemo-testcase-l2-gyre-round118-producer-walk-v1"
                 if args.round118 else
                 "nemo-testcase-l2-gyre-round117-producer-walk-v1")),
            "status": "PLANT_FIRED",
            "worktree": stamp,
            "execution_regime": args.execution_mode,
            "trace_noninterference": trace_identity,
            "plant": args.plant,
            "plant_location": list(location),
            "plant_direction": float(direction),
            "plant_rows": plant_rows,
            "plant_fires": plant_fires,
        }

    stage_arrays = stage["arrays"]
    parts = ordinary.operator_operands[0]
    cumulative = jax.device_get(jax.jit(round117_source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))
    cumulative_rows = {face: {} for face in ("u", "v")}
    for face in ("u", "v"):
        for boundary in ROUND117_BOUNDARIES:
            candidate = (
                native_u(cumulative[f"{boundary}_{face}"])
                if face == "u" else
                native_v(cumulative[f"{boundary}_{face}"]))
            reference = owned3(stage_arrays[f"{boundary}_{face}"])
            cumulative_rows[face][boundary] = comparison(
                candidate, reference, active[f"{face}3"])
    cumulative_first = round117_first_nonbit(cumulative_rows)
    cumulative_closure = {
        "u": comparison(
            native_u(cumulative["after_adv_u"]),
            native_u(producer["rhs_u"]), active["u3"]),
        "v": comparison(
            native_v(cumulative["after_adv_v"]),
            native_v(producer["rhs_v"]), active["v3"]),
    }
    oracle_after_adv_identity = {
        face: comparison(
            owned3(stage_arrays[f"after_adv_{face}"]),
            owned3(stage_arrays[f"after_zad_{face}"]), active[f"{face}3"])
        for face in ("u", "v")
    }
    incremental_residual = {face: {} for face in ("u", "v")}
    for face in ("u", "v"):
        previous = np.zeros_like(
            native_u(cumulative["after_hpg_u"])
            if face == "u" else native_v(cumulative["after_hpg_v"]))
        mask = active[f"{face}3"]
        for boundary in ROUND117_BOUNDARIES:
            candidate = (
                native_u(cumulative[f"{boundary}_{face}"])
                if face == "u" else
                native_v(cumulative[f"{boundary}_{face}"]))
            residual = candidate - owned3(stage_arrays[f"{boundary}_{face}"])
            delta = residual - previous
            incremental_residual[face][boundary] = float(
                np.max(np.abs(delta[mask]), initial=0.0))
            previous = residual

    association_walk = None
    round120_walk = None
    round121_walk = None
    zad_operand_walk = None
    if args.round119 or args.round120 or args.round121:
        source_parts = source_order.operator_operands[0]

        def _native_field(value, face):
            data = value.data if hasattr(value, "data") else value
            return native_u(data) if face == "u" else native_v(data)

        raw_names = ("hpg", "ldf", "vorticity", "keg", "zad")
        raw_identity = {
            face: {
                name: comparison(
                    _native_field(source_parts[f"{name}_{face}"], face),
                    _native_field(parts[f"{name}_{face}"], face),
                    active[f"{face}3"],
                )
                for name in raw_names
            }
            for face in ("u", "v")
        }
        production_vs_isolated = {face: {} for face in ("u", "v")}
        production_vs_oracle = {face: {} for face in ("u", "v")}
        for face in ("u", "v"):
            for boundary in ROUND117_BOUNDARIES:
                production_key = (
                    "after_adv" if boundary in ("after_zad", "after_adv")
                    else boundary)
                production_value = _native_field(
                    source_parts[f"{production_key}_{face}"], face)
                isolated_value = (
                    native_u(cumulative[f"{boundary}_{face}"])
                    if face == "u" else
                    native_v(cumulative[f"{boundary}_{face}"]))
                production_vs_isolated[face][boundary] = comparison(
                    production_value, isolated_value, active[f"{face}3"])
                production_vs_oracle[face][boundary] = comparison(
                    production_value,
                    owned3(stage_arrays[f"{boundary}_{face}"]),
                    active[f"{face}3"])

        ordinary_final = {
            face: _native_field(parts[f"after_ldf_{face}"], face)
            for face in ("u", "v")
        }
        source_final = {
            face: _native_field(source_parts[f"after_adv_{face}"], face)
            for face in ("u", "v")
        }
        ordinary_live_closure = {
            "u": comparison(
                ordinary_final["u"], native_u(producer["rhs_u"]), active["u3"]),
            "v": comparison(
                ordinary_final["v"], native_v(producer["rhs_v"]), active["v3"]),
        }
        source_producer = source_order.slow_forcing_producer
        source_live_closure = {
            "u": comparison(
                source_final["u"], native_u(source_producer["rhs_u"]),
                active["u3"]),
            "v": comparison(
                source_final["v"], native_v(source_producer["rhs_v"]),
                active["v3"]),
        }
        source_to_ordinary = {
            face: comparison(
                source_final[face], ordinary_final[face], active[f"{face}3"])
            for face in ("u", "v")
        }
        association_confirmed = bool(
            all(row["bit_exact"] for rows in raw_identity.values()
                for row in rows.values())
            and all(row["bit_exact"] for rows in production_vs_isolated.values()
                    for row in rows.values())
            and all(row["bit_exact"] for row in ordinary_live_closure.values())
            and all(row["bit_exact"] for row in source_live_closure.values())
            and source_to_ordinary["u"]["differing_cells"] == 6882
            and source_to_ordinary["v"]["differing_cells"] == 6566
            and all(row["absolute_max"]
                    == np.float64(8.470329472543003e-22)
                    for row in source_to_ordinary.values()))
        association_walk = {
            "ordinary_label": (
                "full production step; actual combined (HPG+KEG) -> VOR -> "
                "ZAD -> LDF accumulator"),
            "source_order_label": (
                "full production step; private HPG -> LDF -> VOR -> KEG -> "
                "ZAD association arm"),
            "isolated_label": "isolated-closure JIT; not production",
            "raw_operand_identity": raw_identity,
            "source_order_production_vs_isolated": production_vs_isolated,
            "source_order_production_vs_oracle": production_vs_oracle,
            "ordinary_final_vs_ordinary_live_total": ordinary_live_closure,
            "source_final_vs_source_live_total": source_live_closure,
            "source_order_vs_ordinary_final": source_to_ordinary,
            "state_after_arm_vs_ordinary": round82._pytree_identity(
                source_order.state_after, ordinary.state_after),
            "prediction_confirmed": association_confirmed,
        }

        if args.round120:
            arm_names = (
                "unmasked-materialized",
                "masked-materialized",
                "literal-subtract",
            )
            arm_parts = {}
            arms = {}
            for arm_name in arm_names:
                arm_trace = _round117_live_trace(
                    card, seeded, freshwater, surface, args.execution_mode,
                    association_arm=("stage1-source-order", None, arm_name))
                arm_operator_parts = arm_trace.operator_operands[0]
                arm_parts[arm_name] = arm_operator_parts
                arm_vs_isolated = {face: {} for face in ("u", "v")}
                arm_vs_baseline = {face: {} for face in ("u", "v")}
                for face in ("u", "v"):
                    for boundary in ROUND117_BOUNDARIES:
                        production_key = (
                            "after_adv"
                            if boundary in ("after_zad", "after_adv")
                            else boundary)
                        arm_value = _native_field(
                            arm_operator_parts[f"{production_key}_{face}"],
                            face)
                        isolated_value = (
                            native_u(cumulative[f"{boundary}_{face}"])
                            if face == "u" else
                            native_v(cumulative[f"{boundary}_{face}"]))
                        baseline_value = _native_field(
                            source_parts[f"{production_key}_{face}"], face)
                        arm_vs_isolated[face][boundary] = comparison(
                            arm_value, isolated_value, active[f"{face}3"])
                        arm_vs_baseline[face][boundary] = comparison(
                            arm_value, baseline_value, active[f"{face}3"])
                arm_raw_identity = {
                    face: {
                        name: comparison(
                            _native_field(
                                arm_operator_parts[f"{name}_{face}"], face),
                            _native_field(source_parts[f"{name}_{face}"], face),
                            active[f"{face}3"],
                        )
                        for name in raw_names
                    }
                    for face in ("u", "v")
                }
                arm_final = {
                    face: _native_field(
                        arm_operator_parts[f"after_adv_{face}"], face)
                    for face in ("u", "v")
                }
                arm_live = arm_trace.slow_forcing_producer
                arm_live_closure = {
                    "u": comparison(
                        arm_final["u"], native_u(arm_live["rhs_u"]),
                        active["u3"]),
                    "v": comparison(
                        arm_final["v"], native_v(arm_live["rhs_v"]),
                        active["v3"]),
                }
                arms[arm_name] = {
                    "label": "full production step; private one-variable arm",
                    "production_vs_isolated": arm_vs_isolated,
                    "production_vs_baseline": arm_vs_baseline,
                    "raw_operand_identity_vs_baseline": arm_raw_identity,
                    "live_total_closure": arm_live_closure,
                    "state_after_vs_baseline": round82._pytree_identity(
                        arm_trace.state_after, source_order.state_after),
                    "closes_production_isolated_split": all(
                        row["bit_exact"]
                        for face_rows in arm_vs_isolated.values()
                        for row in face_rows.values()),
                    "one_variable_operands_and_closure": bool(
                        all(row["bit_exact"]
                            for face_rows in arm_raw_identity.values()
                            for row in face_rows.values())
                        and all(row["bit_exact"]
                                for row in arm_live_closure.values())),
                }

            baseline_production_u = _native_field(
                source_parts["after_keg_u"], "u")
            baseline_isolated_u = native_u(cumulative["after_keg_u"])
            mismatch_mask = (
                baseline_production_u.view(np.uint64)
                != baseline_isolated_u.view(np.uint64)) & active["u3"]
            mismatch_locations = [
                tuple(int(index) for index in location)
                for location in np.argwhere(mismatch_mask)
            ]
            baseline_rows_exact_before_keg = all(
                production_vs_isolated[face][boundary]["bit_exact"]
                for face in ("u", "v")
                for boundary in ("after_hpg", "after_ldf", "after_vor"))
            baseline_v_exact = all(
                production_vs_isolated["v"][boundary]["bit_exact"]
                for boundary in ROUND117_BOUNDARIES)
            baseline_common = bool(
                all(row["bit_exact"] for face_rows in raw_identity.values()
                    for row in face_rows.values())
                and all(row["bit_exact"]
                        for row in ordinary_live_closure.values())
                and all(row["bit_exact"]
                        for row in source_live_closure.values()))
            target = None
            if args.execution_mode == "production-jit":
                require(len(mismatch_locations) == 1,
                        "Round-120 did not reproduce exactly one U KEG word")
                location = mismatch_locations[0]
                require(bool(active["u3"][location]),
                        "Round-120 KEG target is not active")
                full_location = (location[0], location[1] + 1, location[2])
                unmasked_addend = _native_field(
                    arm_parts["unmasked-materialized"][
                        "applied_keg_addend_u"], "u")
                masked_addend = _native_field(
                    arm_parts["masked-materialized"][
                        "applied_keg_addend_u"], "u")
                write_only_addend = _native_field(
                    source_parts["keg_u"], "u")
                oracle_after_keg = owned3(stage_arrays["after_keg_u"])
                target = {
                    "native_index": list(location),
                    "full_face_index": list(full_location),
                    "active_mask": float(active["u3"][location]),
                    "after_vor_input": _float64_word(
                        _native_field(
                            source_parts["after_vor_u"], "u")[location]),
                    "unmasked_materialized_addend": _float64_word(
                        unmasked_addend[location]),
                    "masked_materialized_addend": _float64_word(
                        masked_addend[location]),
                    "write_only_masked_keg": _float64_word(
                        write_only_addend[location]),
                    "production_after_keg": _float64_word(
                        baseline_production_u[location]),
                    "isolated_after_keg": _float64_word(
                        baseline_isolated_u[location]),
                    "oracle_after_keg": _float64_word(
                        oracle_after_keg[location]),
                    "production_minus_isolated_signed_ulp": (
                        _signed_ulp_difference(
                            baseline_production_u[location],
                            baseline_isolated_u[location])),
                    "unmasked_equals_masked_addend_bits": bool(
                        unmasked_addend[location].view(np.uint64)
                        == masked_addend[location].view(np.uint64)),
                    "unmasked_equals_write_only_bits": bool(
                        unmasked_addend[location].view(np.uint64)
                        == write_only_addend[location].view(np.uint64)),
                }
                baseline_reproduced = bool(
                    baseline_common
                    and baseline_rows_exact_before_keg
                    and baseline_v_exact
                    and production_vs_isolated["u"]["after_keg"][
                        "differing_cells"] == 1
                    and production_vs_isolated["u"]["after_keg"][
                        "absolute_max"]
                    == np.float64(4.1359030627651384e-25)
                    and all(production_vs_isolated["u"][boundary][
                        "differing_cells"] == 1
                        for boundary in ("after_zad", "after_adv"))
                    and source_to_ordinary["u"]["differing_cells"] == 6882
                    and source_to_ordinary["v"]["differing_cells"] == 6566
                    and all(row["absolute_max"]
                            == np.float64(8.470329472543003e-22)
                            for row in source_to_ordinary.values()))
            else:
                require(not mismatch_locations,
                        "Round-120 eager control has a U KEG mismatch")
                baseline_reproduced = bool(
                    baseline_common
                    and baseline_rows_exact_before_keg
                    and baseline_v_exact
                    and all(
                        row["bit_exact"]
                        for face_rows in production_vs_isolated.values()
                        for row in face_rows.values()))

            closed_arms = [
                name for name in arm_names
                if arms[name]["closes_production_isolated_split"]
            ]
            if args.execution_mode == "production-eager":
                classification = "eager-control-has-no-split"
            elif ("unmasked-materialized" in closed_arms
                  and "masked-materialized" in closed_arms
                  and "literal-subtract" not in closed_arms):
                classification = "materialization-fusion; mask is inert"
            elif closed_arms == ["unmasked-materialized"]:
                classification = "unmasked materialization-fusion"
            elif closed_arms == ["masked-materialized"]:
                classification = "mask materialization"
            elif closed_arms == ["literal-subtract"]:
                classification = "add-versus-subtract association"
            elif not closed_arms:
                classification = "owner withheld; no arm closes"
            else:
                classification = "owner withheld; multiple arms close"
            round120_confirmed = bool(
                baseline_reproduced
                and all(arms[name]["one_variable_operands_and_closure"]
                        for name in arm_names)
                and (args.execution_mode == "production-eager" or (
                    target is not None
                    and target["active_mask"] == 1.0
                    and target["unmasked_equals_masked_addend_bits"]
                    and target["unmasked_equals_write_only_bits"]
                    and arms["unmasked-materialized"][
                        "closes_production_isolated_split"])))
            round120_walk = {
                "baseline_reproduced": baseline_reproduced,
                "baseline_mismatch_locations_u": [
                    list(location) for location in mismatch_locations],
                "target": target,
                "arms": arms,
                "closed_arms": closed_arms,
                "classification": classification,
                "prediction_confirmed": round120_confirmed,
            }

        _, live_r3u, live_r3v = ordinary.stage_qco[0]
        zad_live = {
            "velocity_u": native_u(parts["operand_velocity_u"]),
            "velocity_v": native_v(parts["operand_velocity_v"]),
            "ww": np.asarray(parts["operand_zad_w"], dtype=np.float64)[..., :30],
            "r3u": native_u(live_r3u)[..., 0],
            "r3v": native_v(live_r3v)[..., 0],
            "thickness_u": native_u(parts["operand_zad_h_u"]),
            "thickness_v": native_v(parts["operand_zad_h_v"]),
            "area_t": np.asarray(card.recipe.grid.area_T, dtype=np.float64),
            "reciprocal_area_u": native_u(
                np.float64(1.0) / (
                    np.asarray(card.recipe.grid.dx_u, dtype=np.float64)
                    * np.asarray(card.recipe.grid.dy_u, dtype=np.float64))),
            "reciprocal_area_v": native_v(
                np.float64(1.0) / (
                    np.asarray(card.recipe.grid.dx_v, dtype=np.float64)
                    * np.asarray(card.recipe.grid.dy_v, dtype=np.float64))),
        }
        zad_oracle = {
            "velocity_u": owned3(stage_arrays["u_Kmm"]),
            "velocity_v": owned3(stage_arrays["v_Kmm"]),
            "ww": owned3(stage_arrays["ww"]),
            "r3u": owned2(stage_arrays["r3u_Kmm"]),
            "r3v": owned2(stage_arrays["r3v_Kmm"]),
            "thickness_u": owned3(stage_arrays["e3u_Kmm"]),
            "thickness_v": owned3(stage_arrays["e3v_Kmm"]),
            "area_t": owned2(stage_arrays["e1e2t"]),
            "reciprocal_area_u": owned2(stage_arrays["r1_e1e2u"]),
            "reciprocal_area_v": owned2(stage_arrays["r1_e1e2v"]),
        }
        zad_active = {
            "velocity_u": active["u3"], "velocity_v": active["v3"],
            "ww": active["t3"], "r3u": active["u2"],
            "r3v": active["v2"], "thickness_u": active["u3"],
            "thickness_v": active["v3"], "area_t": active["t2"],
            "reciprocal_area_u": active["u2"],
            "reciprocal_area_v": active["v2"],
        }
        zad_order = tuple(zad_live)
        zad_rows = {
            name: comparison(zad_live[name], zad_oracle[name], zad_active[name])
            for name in zad_order
        }
        zad_first = next(
            (name for name in zad_order if not zad_rows[name]["bit_exact"]), None)
        zad_operand_walk = {
            "order": list(zad_order),
            "rows": zad_rows,
            "first_non_bit": zad_first,
            "largest_absolute_max": max(
                zad_order, key=lambda name: zad_rows[name]["absolute_max"]),
            "prediction_confirmed": bool(
                zad_first == "ww"
                and max(zad_order,
                        key=lambda name: zad_rows[name]["absolute_max"]) == "ww"
                and all(zad_rows[name]["bit_exact"] for name in (
                    "velocity_u", "velocity_v", "thickness_u", "thickness_v",
                    "area_t", "reciprocal_area_u", "reciprocal_area_v"))),
        }

        if args.round121:
            oracle_w = owned3_with_bottom(stage_arrays["ww"])
            live_w = np.asarray(parts["operand_zad_w"], dtype=np.float64)
            require(oracle_w.shape == live_w.shape,
                    "Round-121 oracle/live W extents differ")
            directed = _round117_live_trace(
                card, seeded, freshwater, surface, args.execution_mode,
                zad_w_override=jnp.asarray(oracle_w))
            directed_parts = directed.operator_operands[0]
            directed_w = np.asarray(
                directed_parts["operand_zad_w"], dtype=np.float64)
            directed_w_row = comparison(
                directed_w[..., :30], owned3(stage_arrays["ww"]),
                active["t3"])
            non_w_raw_rows = {
                face: {
                    name: comparison(
                        _native_field(
                            directed_parts[f"{name}_{face}"], face),
                        _native_field(parts[f"{name}_{face}"], face),
                        active[f"{face}3"],
                    )
                    for name in ("hpg", "ldf", "vorticity", "keg")
                }
                for face in ("u", "v")
            }
            non_w_zad_inputs = {
                "velocity_u": comparison(
                    native_u(directed_parts["operand_velocity_u"]),
                    native_u(parts["operand_velocity_u"]), active["u3"]),
                "velocity_v": comparison(
                    native_v(directed_parts["operand_velocity_v"]),
                    native_v(parts["operand_velocity_v"]), active["v3"]),
                "thickness_u": comparison(
                    native_u(directed_parts["operand_zad_h_u"]),
                    native_u(parts["operand_zad_h_u"]), active["u3"]),
                "thickness_v": comparison(
                    native_v(directed_parts["operand_zad_h_v"]),
                    native_v(parts["operand_zad_h_v"]), active["v3"]),
            }
            pre_zad_rows = {
                face: {
                    boundary: comparison(
                        _native_field(
                            directed_parts[f"{boundary}_{face}"], face),
                        _native_field(parts[f"{boundary}_{face}"], face),
                        active[f"{face}3"],
                    )
                    for boundary in (
                        "after_hpg", "after_vor", "after_keg")
                }
                for face in ("u", "v")
            }
            raw_zad_effect = {
                face: comparison(
                    _native_field(directed_parts[f"zad_{face}"], face),
                    _native_field(parts[f"zad_{face}"], face),
                    active[f"{face}3"],
                )
                for face in ("u", "v")
            }
            directed_cumulative = jax.device_get(jax.jit(
                round117_source_order_accumulators)(
                    directed_parts["hpg_u"].data,
                    directed_parts["hpg_v"].data,
                    directed_parts["ldf_u"].data,
                    directed_parts["ldf_v"].data,
                    directed_parts["vorticity_u"].data,
                    directed_parts["vorticity_v"].data,
                    directed_parts["keg_u"].data,
                    directed_parts["keg_v"].data,
                    directed_parts["zad_u"].data,
                    directed_parts["zad_v"].data,
                ))
            directed_incremental_residual = {}
            for face in ("u", "v"):
                native = native_u if face == "u" else native_v
                before_residual = (
                    native(directed_cumulative[f"after_keg_{face}"])
                    - owned3(stage_arrays[f"after_keg_{face}"]))
                after_residual = (
                    native(directed_cumulative[f"after_zad_{face}"])
                    - owned3(stage_arrays[f"after_zad_{face}"]))
                delta = after_residual - before_residual
                mask = active[f"{face}3"]
                directed_incremental_residual[face] = {
                    "differing_cells": int(np.count_nonzero(delta[mask])),
                    "absolute_max": float(
                        np.max(np.abs(delta[mask]), initial=0.0)),
                }
            state_effect = round82._pytree_identity(
                directed.state_after, ordinary.state_after)
            ordinary_w_reproduced = bool(
                zad_rows["ww"]["differing_cells"] == 18000
                and zad_rows["ww"]["absolute_max"]
                == np.float64(7.946658315637966e-7)
                and incremental_residual["u"]["after_zad"]
                == np.float64(1.9220297482797664e-9)
                and incremental_residual["v"]["after_zad"]
                == np.float64(1.966061294804274e-9))
            one_variable = bool(
                all(row["bit_exact"]
                    for rows in non_w_raw_rows.values()
                    for row in rows.values())
                and all(row["bit_exact"]
                        for row in non_w_zad_inputs.values())
                and all(row["bit_exact"]
                        for rows in pre_zad_rows.values()
                        for row in rows.values()))
            residual_at_floor = all(
                row["absolute_max"]
                <= np.float64(8.470329472543003e-22)
                for row in directed_incremental_residual.values())
            round121_confirmed = bool(
                ordinary_w_reproduced and directed_w_row["bit_exact"]
                and one_variable and residual_at_floor
                and all(row["differing_cells"] > 0
                        for row in raw_zad_effect.values())
                and not state_effect["bit_exact"])
            round121_walk = {
                "ordinary_w_row": zad_rows["ww"],
                "ordinary_incremental_zad_residual_max": {
                    face: incremental_residual[face]["after_zad"]
                    for face in ("u", "v")},
                "directed_consumed_w_vs_oracle": directed_w_row,
                "non_w_raw_operand_identity": non_w_raw_rows,
                "non_w_zad_input_identity": non_w_zad_inputs,
                "pre_zad_boundary_identity": pre_zad_rows,
                "directed_vs_ordinary_raw_zad": raw_zad_effect,
                "directed_incremental_zad_residual": (
                    directed_incremental_residual),
                "returned_state_vs_ordinary": state_effect,
                "ordinary_reproduction_confirmed": ordinary_w_reproduced,
                "one_variable_confirmed": one_variable,
                "association_floor_confirmed": residual_at_floor,
                "prediction_confirmed": round121_confirmed,
            }

            if args.plant == "zad-w-ulp":
                require(args.execution_mode == "production-jit",
                        "Round-121 W plant requires production JIT")
                eligible = (
                    active["t3"] & np.isfinite(oracle_w[..., :30])
                    & (oracle_w[..., :30] != 0.0))
                locations = np.argwhere(eligible)
                require(locations.size > 0,
                        "Round-121 W plant found no eligible word")
                location = tuple(int(value) for value in locations[0])
                planted_w = np.array(oracle_w, copy=True)
                planted_w[location] = np.nextafter(
                    planted_w[location], np.float64(np.inf))
                planted = _round117_live_trace(
                    card, seeded, freshwater, surface, args.execution_mode,
                    zad_w_override=jnp.asarray(planted_w))
                planted_parts = planted.operator_operands[0]
                planted_w_row = comparison(
                    np.asarray(planted_parts["operand_zad_w"])[..., :30],
                    directed_w[..., :30], active["t3"])
                planted_non_w = {
                    face: {
                        name: comparison(
                            _native_field(
                                planted_parts[f"{name}_{face}"], face),
                            _native_field(
                                directed_parts[f"{name}_{face}"], face),
                            active[f"{face}3"],
                        )
                        for name in ("hpg", "ldf", "vorticity", "keg")
                    }
                    for face in ("u", "v")
                }
                planted_zad = {
                    face: comparison(
                        _native_field(planted_parts[f"zad_{face}"], face),
                        _native_field(directed_parts[f"zad_{face}"], face),
                        active[f"{face}3"],
                    )
                    for face in ("u", "v")
                }
                planted_state = round82._pytree_identity(
                    planted.state_after, directed.state_after)
                plant_fires = bool(
                    planted_w_row["differing_cells"] == 1
                    and all(row["bit_exact"]
                            for rows in planted_non_w.values()
                            for row in rows.values())
                    and any(row["differing_cells"] > 0
                            for row in planted_zad.values())
                    and not planted_state["bit_exact"])
                return {
                    "format": "nemo-testcase-l2-gyre-round121-w-walk-v1",
                    "status": (
                        "PLANT_FIRED" if plant_fires else "PLANT_INERT"),
                    "worktree": stamp,
                    "execution_regime": args.execution_mode,
                    "plant": args.plant,
                    "plant_location": list(location),
                    "plant_direction": "+infinity",
                    "consumed_w_row": planted_w_row,
                    "non_w_raw_operand_rows": planted_non_w,
                    "raw_zad_rows": planted_zad,
                    "returned_state_row": planted_state,
                    "plant_fires": plant_fires,
                }

        if args.plant == "association-hpg-ulp":
            planted_native, location, direction = _round119_propagating_hpg_ulp(
                _native_field(parts["hpg_u"], "u"),
                _native_field(parts["ldf_u"], "u"),
                _native_field(parts["vorticity_u"], "u"),
                _native_field(parts["keg_u"], "u"),
                _native_field(parts["zad_u"], "u"), active["u3"])
            hpg_override = (
                jnp.asarray(_full_from_native(
                    parts["hpg_u"].data, planted_native, "u")),
                jnp.asarray(parts["hpg_v"].data),
            )
            planted = _round117_live_trace(
                card, seeded, freshwater, surface, args.execution_mode,
                association_arm=("stage1-source-order", hpg_override))
            planted_parts = planted.operator_operands[0]
            planted_boundaries = {}
            for boundary in ROUND117_BOUNDARIES:
                production_key = (
                    "after_adv" if boundary in ("after_zad", "after_adv")
                    else boundary)
                planted_boundaries[boundary] = comparison(
                    _native_field(
                        planted_parts[f"{production_key}_u"], "u"),
                    _native_field(source_parts[f"{production_key}_u"], "u"),
                    active["u3"])
            planted_raw_identity = {
                name: comparison(
                    _native_field(planted_parts[f"{name}_u"], "u"),
                    _native_field(source_parts[f"{name}_u"], "u"),
                    active["u3"])
                for name in raw_names
            }
            plant_fires = bool(
                planted_raw_identity["hpg"]["differing_cells"] == 1
                and all(planted_raw_identity[name]["bit_exact"]
                        for name in raw_names if name != "hpg")
                and all(row["differing_cells"] > 0
                        for row in planted_boundaries.values()))
            require(plant_fires,
                    "Round-119 HPG ULP did not propagate through every boundary")
            return {
                "format": "nemo-testcase-l2-gyre-round119-association-walk-v1",
                "status": "PLANT_FIRED",
                "worktree": stamp,
                "execution_regime": args.execution_mode,
                "trace_noninterference": trace_identity,
                "plant": args.plant,
                "plant_location": list(location),
                "plant_direction": float(direction),
                "raw_operand_rows": planted_raw_identity,
                "boundary_rows": planted_boundaries,
                "plant_fires": plant_fires,
            }

        if args.plant == "association-keg-ulp":
            require(args.round120,
                    "the production KEG plant is a Round-120 control")
            require(args.execution_mode == "production-jit",
                    "the production KEG plant requires production JIT")
            require(round120_walk is not None
                    and round120_walk["target"] is not None,
                    "Round-120 KEG plant has no reproduced target")
            unmasked_parts = arm_parts["unmasked-materialized"]
            base_addend_u = np.asarray(
                unmasked_parts["applied_keg_addend_u"].data,
                dtype=np.float64)
            base_addend_v = np.asarray(
                unmasked_parts["applied_keg_addend_v"].data,
                dtype=np.float64)
            reference = _round117_live_trace(
                card, seeded, freshwater, surface, args.execution_mode,
                association_arm=(
                    "stage1-source-order", None, "override-materialized",
                    (jnp.asarray(base_addend_u),
                     jnp.asarray(base_addend_v))))
            reference_parts = reference.operator_operands[0]
            full_location = tuple(
                round120_walk["target"]["full_face_index"])
            planted_u, direction = _round120_keg_ulp_addend(
                base_addend_u,
                np.asarray(reference_parts["after_vor_u"].data,
                           dtype=np.float64),
                np.asarray(reference_parts["zad_u"].data, dtype=np.float64),
                full_location,
            )
            planted = _round117_live_trace(
                card, seeded, freshwater, surface, args.execution_mode,
                association_arm=(
                    "stage1-source-order", None, "override-materialized",
                    (jnp.asarray(planted_u), jnp.asarray(base_addend_v))))
            planted_parts = planted.operator_operands[0]
            applied_row = comparison(
                _native_field(
                    planted_parts["applied_keg_addend_u"], "u"),
                _native_field(
                    reference_parts["applied_keg_addend_u"], "u"),
                active["u3"],
            )
            planted_boundaries = {}
            for boundary in ROUND117_BOUNDARIES:
                production_key = (
                    "after_adv"
                    if boundary in ("after_zad", "after_adv")
                    else boundary)
                planted_boundaries[boundary] = comparison(
                    _native_field(
                        planted_parts[f"{production_key}_u"], "u"),
                    _native_field(
                        reference_parts[f"{production_key}_u"], "u"),
                    active["u3"],
                )
            planted_raw_identity = {
                name: comparison(
                    _native_field(planted_parts[f"{name}_u"], "u"),
                    _native_field(reference_parts[f"{name}_u"], "u"),
                    active["u3"],
                )
                for name in raw_names
            }
            reference_identity = {
                "applied_addend": comparison(
                    _native_field(
                        reference_parts["applied_keg_addend_u"], "u"),
                    _native_field(
                        unmasked_parts["applied_keg_addend_u"], "u"),
                    active["u3"],
                ),
                "raw_operands": {
                    name: comparison(
                        _native_field(reference_parts[f"{name}_u"], "u"),
                        _native_field(unmasked_parts[f"{name}_u"], "u"),
                        active["u3"],
                    )
                    for name in raw_names
                },
            }
            plant_fires = bool(
                applied_row["differing_cells"] == 1
                and reference_identity["applied_addend"]["bit_exact"]
                and all(row["bit_exact"] for row in
                        reference_identity["raw_operands"].values())
                and all(planted_raw_identity[name]["bit_exact"]
                        for name in raw_names)
                and all(planted_boundaries[boundary]["bit_exact"]
                        for boundary in (
                            "after_hpg", "after_ldf", "after_vor"))
                and all(planted_boundaries[boundary]["differing_cells"] == 1
                        for boundary in (
                            "after_keg", "after_zad", "after_adv")))
            return {
                "format": "nemo-testcase-l2-gyre-round120-keg-walk-v1",
                "status": (
                    "PLANT_FIRED" if plant_fires else "PLANT_INERT"),
                "worktree": stamp,
                "execution_regime": args.execution_mode,
                "plant": args.plant,
                "plant_location_native": round120_walk[
                    "target"]["native_index"],
                "plant_location_full_face": list(full_location),
                "plant_direction": float(direction),
                "unplanted_override_identity": reference_identity,
                "applied_addend_row": applied_row,
                "raw_operand_rows": planted_raw_identity,
                "boundary_rows": planted_boundaries,
                "plant_fires": plant_fires,
            }

    substeps = captured.substeps
    source_replay = {}
    if direct is not None:
        slow_fields = direct["slow"]["fields"]
        rho_reciprocal = np.float64(slow_fields["r1_rho0"])
        require(float(producer["wind_r1_rho0"]) == float(rho_reciprocal),
                "live/direct density reciprocal differs")
        for face in ("u", "v"):
            # round16._source_sum implements NEMO's 1:jpkm1 reduction by
            # deliberately omitting the final, non-contributing jpk slot.
            # Preserve that slot here: trimming first would omit physical
            # level jpkm1 as well (the Round-28 instrument defect).
            e3 = owned3_with_bottom(slow_fields[f"e3{face}"])
            mask3 = owned3_with_bottom(slow_fields[f"{face}mask"])
            direct_rhs = owned3_with_bottom(slow_fields[f"krhs_{face}"])
            inherited_rhs = owned3(stage_arrays[f"after_adv_{face}"])
            reciprocal_ref = owned2(slow_fields[f"r1_h{face}0"])
            inverse_depth = owned2(slow_fields[f"r1_h{face}"])
            drag = owned2(slow_fields[f"cd_{face}"])
            bottom = bottom_value(
                owned3(stage_arrays[f"{face}_Kmm"]), mask3[..., :30] != 0.0)
            barotropic = owned2(
                direct["preloop"]["fields"][f"{face}_kmm"])
            stress = owned2(slow_fields[f"{face}tau"])
            coriolis = oracle_split[face]["direct_coriolis"]
            chain = source_chain(
                rhs=direct_rhs, e3=e3, mask3=mask3,
                reciprocal_ref=reciprocal_ref,
                inverse_depth=inverse_depth, drag_coefficient=drag,
                bottom_velocity=bottom, barotropic_velocity=barotropic,
                rho_reciprocal=rho_reciprocal, stress=stress,
                coriolis=coriolis,
                mask2=active[f"{face}2"].astype(np.float64),
            )
            source_replay[face] = {
                "inherited_after_adv_to_same_run_krhs": comparison(
                    inherited_rhs, direct_rhs[..., :30], active[f"{face}3"]),
                "depth_replay_to_direct_record": comparison(
                    chain["depth_mean"], slow_fields[f"depth_{face}"],
                    active[f"{face}2"]),
                "post_drag_replay_to_direct_record": comparison(
                    chain["post_drag"], slow_fields[f"post_drag_{face}"],
                    active[f"{face}2"]),
                "post_wind_replay_to_direct_record": comparison(
                    chain["post_wind"], slow_fields[f"post_wind_{face}"],
                    active[f"{face}2"]),
                "forward_final_vs_direct_record": comparison(
                    chain["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
            }
    else:
        # The Round-64 stream has no direct kt=2 stress snapshot.  Keep the
        # proxy join explicit and withhold input certification.
        rho_reciprocal = np.float64(static["r1_rho0"])
        require(float(producer["wind_r1_rho0"]) == float(rho_reciprocal),
                "live/oracle density reciprocal differs")
        for face in ("u", "v"):
            if face == "u":
                e3 = owned3(stage_arrays["e3u_0"])
                mask3 = owned3(stage_arrays["umask"])
                reciprocal_ref = np.asarray(static["r1_hu0"], dtype=np.float64)
                inverse_depth = gate._trace_native(
                    substeps["inverse_depth_u"], "inverse_depth_u")[0]
                drag = gate._trace_native(
                    substeps["drag_coefficient_u"], "drag_coefficient_u")[0]
                bottom = bottom_value(
                    owned3(stage_arrays["u_Kmm"]), mask3 != 0.0)
                barotropic = owned2(stage_arrays["uu_b_Kmm"])
                stress = native_u(producer["wind_tau_u"])
                coriolis = oracle_split[face]["substep1_coriolis_proxy"]
            else:
                e3 = owned3(stage_arrays["e3v_0"])
                mask3 = owned3(stage_arrays["vmask"])
                reciprocal_ref = np.asarray(static["r1_hv0"], dtype=np.float64)
                inverse_depth = gate._trace_native(
                    substeps["inverse_depth_v"], "inverse_depth_v")[0]
                drag = gate._trace_native(
                    substeps["drag_coefficient_v"], "drag_coefficient_v")[0]
                bottom = bottom_value(
                    owned3(stage_arrays["v_Kmm"]), mask3 != 0.0)
                barotropic = owned2(stage_arrays["vv_b_Kmm"])
                stress = native_v(producer["wind_tau_v"])
                coriolis = oracle_split[face]["substep1_coriolis_proxy"]
            chain = source_chain(
                rhs=owned3(stage_arrays[f"after_adv_{face}"]),
                e3=e3, mask3=mask3, reciprocal_ref=reciprocal_ref,
                inverse_depth=inverse_depth, drag_coefficient=drag,
                bottom_velocity=bottom, barotropic_velocity=barotropic,
                rho_reciprocal=rho_reciprocal, stress=stress,
                coriolis=coriolis,
                mask2=active[f"{face}2"].astype(np.float64),
            )
            source_replay[face] = {
                "post_wind_vs_substep_proxy_preimage": comparison(
                    chain["post_wind"],
                    oracle_split[face]["proxy_incoming_preimage"],
                    active[f"{face}2"]),
                "forward_final_vs_direct_record": comparison(
                    chain["final"], oracle_split[face]["final"],
                    active[f"{face}2"]),
            }

    if direct is not None:
        p1_confirmed = bool(
            direct_first is not None
            and direct_first["boundary"] == "incoming_vs_direct_record"
            and direct_first["face"] == "u"
            and all(
                split_rows[face]["incoming_vs_direct_record"]
                ["differing_cells"] == (580 if face == "u" else 570)
                and abs(
                    split_rows[face]["incoming_vs_direct_record"]
                    ["absolute_max"] - ROUND117_FINAL_MAX[face])
                <= np.float64(1.0e-22)
                and split_rows[face]["preloop_coriolis_vs_direct_record"]
                ["bit_exact"]
                and split_rows[face]["final_vs_direct_record"]
                ["differing_cells"] == (580 if face == "u" else 570)
                and abs(
                    split_rows[face]["final_vs_direct_record"]
                    ["absolute_max"] - ROUND117_FINAL_MAX[face])
                <= np.float64(1.0e-22)
                for face in ("u", "v"))
            and all(split_rows[face]["isolated_subtract_vs_production_final"]
                    ["bit_exact"] for face in ("u", "v")))
    else:
        p1_confirmed = False
    ldf_ranges = {
        "u": (np.float64(6.3e-15), np.float64(1.02e-13)),
        "v": (np.float64(8.7e-15), np.float64(1.41e-13)),
    }
    common_p2 = bool(
        cumulative_first is not None
        and cumulative_first["boundary"] == "after_ldf"
        and all(cumulative_rows[face]["after_hpg"]["bit_exact"]
                for face in ("u", "v"))
        and all(
            ldf_ranges[face][0]
            <= cumulative_rows[face]["after_ldf"]["absolute_max"]
            <= ldf_ranges[face][1]
            for face in ("u", "v"))
        and all(max(incremental_residual[face],
                    key=incremental_residual[face].get) == "after_zad"
                for face in ("u", "v"))
        and all(row["bit_exact"] for row in oracle_after_adv_identity.values())
    )
    if direct is not None:
        p2_confirmed = bool(
            common_p2
            and all(
                abs(cumulative_rows[face]["after_ldf"]["absolute_max"]
                    - value) <= np.float64(1.0e-26)
                for face, value in {
                    "u": np.float64(2.5292467120726215e-14),
                    "v": np.float64(3.502735092670824e-14),
                }.items())
            and cumulative_closure["u"]["differing_cells"] == 6882
            and cumulative_closure["v"]["differing_cells"] == 6566
            and all(
                row["absolute_max"] == np.float64(8.470329472543003e-22)
                for row in cumulative_closure.values())
            and all(
                row["bit_exact"]
                for face_rows in source_replay.values()
                for row in face_rows.values()))
    else:
        p2_confirmed = bool(
            common_p2
            and all(row["bit_exact"] for row in cumulative_closure.values()))

    magnitude = None
    if args.execution_mode == "production-jit" and direct is None:
        directed_final = (
            jnp.asarray(_full_from_native(
                producer["final_u"], oracle_split["u"]["final"], "u")),
            jnp.asarray(_full_from_native(
                producer["final_v"], oracle_split["v"]["final"], "v")),
        )
        directed = _round117_live_trace(
            card, seeded, freshwater, surface, args.execution_mode,
            final_override=directed_final)
        directed_producer = directed.slow_forcing_producer
        override_identity = {
            "incoming_u": comparison(
                native_u(directed_producer["incoming_u"]),
                live_split["u"]["incoming"], active["u2"]),
            "incoming_v": comparison(
                native_v(directed_producer["incoming_v"]),
                live_split["v"]["incoming"], active["v2"]),
            "coriolis_u": comparison(
                native_u(directed_producer["coriolis_u"]),
                live_split["u"]["coriolis"], active["u2"]),
            "coriolis_v": comparison(
                native_v(directed_producer["coriolis_v"]),
                live_split["v"]["coriolis"], active["v2"]),
            "final_u": comparison(
                native_u(directed_producer["final_u"]),
                oracle_split["u"]["final"], active["u2"]),
            "final_v": comparison(
                native_v(directed_producer["final_v"]),
                oracle_split["v"]["final"], active["v2"]),
        }
        require(all(row["bit_exact"] for row in override_identity.values()),
                "directed arm changed or missed a registered producer input")
        next_entry = gate.read_entry(
            args.entry_root / "oracle_step_entry_kt00000003.bin")
        references = {
            "ssh": np.asarray(next_entry["ssh"], dtype=np.float64),
            "T": np.asarray(next_entry["T"], dtype=np.float64)[..., :30],
            "S": np.asarray(next_entry["S"], dtype=np.float64)[..., :30],
        }
        ordinary_values = {
            "ssh": np.asarray(ordinary.barotropic_targets[4]),
            "T": np.asarray(ordinary.stage_outputs[2][2]),
            "S": np.asarray(ordinary.stage_outputs[2][3]),
        }
        directed_values = {
            "ssh": np.asarray(directed.barotropic_targets[4]),
            "T": np.asarray(directed.stage_outputs[2][2]),
            "S": np.asarray(directed.stage_outputs[2][3]),
        }
        field_masks = {"ssh": active["t2"], "T": active["t3"],
                       "S": active["t3"]}
        rows = {
            name: {
                "ordinary": comparison(
                    ordinary_values[name], references[name], field_masks[name]),
                "record_directed": comparison(
                    directed_values[name], references[name], field_masks[name]),
                "directed_minus_ordinary": comparison(
                    directed_values[name], ordinary_values[name],
                    field_masks[name]),
            }
            for name in ("ssh", "T", "S")
        }
        bands = {"ssh": 0.10, "T": 0.10, "S": 0.50}
        band_ok = all(
            abs(rows[name]["record_directed"]["absolute_max"]
                / rows[name]["ordinary"]["absolute_max"] - 1.0)
            < bands[name]
            for name in rows
        )
        predicted_direction = all(
            rows[name]["record_directed"]["absolute_max"]
            < rows[name]["ordinary"]["absolute_max"]
            for name in ("ssh", "T")
        )
        magnitude_confirmed = bool(
            all(rows[name]["directed_minus_ordinary"]["differing_cells"] > 0
                for name in rows)
            and all(not rows[name]["record_directed"]["bit_exact"]
                    for name in rows)
            and band_ok and predicted_direction)
        magnitude = {
            "intervention": (
                "replace only the final frozen slow-U/slow-V pair at the "
                "external-solver call"),
            "changed_operand_registry": ["final_slow_forcing_pair"],
            "producer_override_identity": override_identity,
            "rows": rows,
            "band_ok": band_ok,
            "predicted_ssh_and_T_improve": predicted_direction,
            "prediction_confirmed": magnitude_confirmed,
        }

    input_certified = bool(
        direct is not None
        and all(
            row["bit_exact"]
            for face_rows in source_replay.values()
            for row in face_rows.values()))
    if args.round120:
        prediction_confirmed = bool(
            round120_walk is not None
            and round120_walk["prediction_confirmed"]
            and zad_operand_walk is not None
            and zad_operand_walk["prediction_confirmed"])
    elif args.round121:
        prediction_confirmed = bool(
            p2_confirmed
            and zad_operand_walk is not None
            and zad_operand_walk["prediction_confirmed"]
            and round121_walk is not None
            and round121_walk["prediction_confirmed"])
    else:
        prediction_confirmed = bool(
            p1_confirmed and p2_confirmed
            and (magnitude is None or magnitude["prediction_confirmed"])
            and (not args.round119 or (
                association_walk is not None
                and association_walk["prediction_confirmed"]
                and zad_operand_walk is not None
                and zad_operand_walk["prediction_confirmed"])))
    status = (
        "MEASURED" if args.execution_mode == "production-eager" else
        ("CONFIRMED" if prediction_confirmed else "REFUTED"))
    return {
        "format": (
            "nemo-testcase-l2-gyre-round121-w-walk-v1"
            if args.round121 else
            ("nemo-testcase-l2-gyre-round120-keg-walk-v1"
            if args.round120 else
            ("nemo-testcase-l2-gyre-round119-association-walk-v1"
             if args.round119 else
            ("nemo-testcase-l2-gyre-round118-producer-walk-v1"
             if direct is not None else
             "nemo-testcase-l2-gyre-round117-producer-walk-v1")))),
        "status": status,
        "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "round64_stage_sha256": sha256(
            args.round64_root / "oracle_momstage_kt00000002_s1.bin"),
        "round81_btstep_sha256": round81.sha256(
            args.record_root / round81.RECORD),
        "trace_noninterference": trace_identity,
        "trace_final_vs_actual_external_call": trace_final_identity,
        "producer_split": {
            "record_limitation": (
                None if direct is not None else
                "Round-81 cor_u/cor_v is the substep-1 dyn_cor_2D result at "
                "compiled lines 683-686, not the pre-loop Kmm result at line "
                "322; direct pre-loop incoming and Coriolis rows are absent"),
            "incoming_reference": (
                "admitted direct same-run NEMO pre-loop field"
                if direct is not None else
                "PROXY algebraic preimage from final plus substep-1 Coriolis; "
                "not a direct NEMO pre-loop field"),
            "source_order": ["incoming", "coriolis", "final"],
            "rows": split_rows,
            "first_proxy_non_bit": proxy_first,
            "first_direct_non_bit": direct_first,
            "incoming_coriolis_cancellation": cancellation,
            "prediction_confirmed": p1_confirmed,
        },
        "isolated_subtract": {
            "label": "isolated-closure JIT; not production",
            "rows": {
                face: split_rows[face][
                    "isolated_subtract_vs_production_final"]
                for face in ("u", "v")},
        },
        ("round64_round117_join" if direct is not None
         else "round64_round81_join"): {
            "kt2_wind_stress_identity": (
                "DIRECT_SAME_RUN" if direct is not None
                else "WITHHELD_NO_DIRECT_RECORD"),
            "rows": source_replay,
            "direct_input_certified": input_certified,
        },
        "current_tip_cumulative_rhs": {
            "label": (
                "production-step operands; isolated source-order JIT "
                "accumulator"),
            "source_order": list(ROUND117_BOUNDARIES),
            "rows": cumulative_rows,
            "incremental_residual_maxima": incremental_residual,
            "first_non_bit": cumulative_first,
            "live_total_closure": cumulative_closure,
            "oracle_after_adv_equals_after_zad": oracle_after_adv_identity,
            "prediction_confirmed": p2_confirmed,
        },
        "record_directed_magnitude": magnitude,
        "production_association_walk": association_walk,
        "round120_keg_discriminator": round120_walk,
        "round121_w_intervention": round121_walk,
        "current_tip_zad_operand_walk": zad_operand_walk,
        "candidate_eligible": False,
        "candidate_reason": (
            "the KEG split is classified but is a one-word last-bit effect; "
            "W's day-240 ownership remains unmeasured and no magnitude "
            "candidate is preregistered"
            if args.round120 and round120_walk is not None else
            ("the production association is attributed, but it is a last-bit "
             "effect below the measured W-owned ZAD magnitude; the next "
             "candidate must start at W's current-tip producer"
            if args.round119 and association_walk is not None
            and association_walk["prediction_confirmed"] else
            ("direct producer inputs are certified, but the isolated cumulative "
             "association does not close bit-for-bit to the live production RHS"
            if direct is not None else
            "diagnostic only; incoming kt2 Ue_rhs/Ve_rhs is not directly "
            "recorded and the cumulative LDF output lacks a complete direct-"
            "input source-exactness proof"))),
        "prediction_confirmed": prediction_confirmed,
        "plant": args.plant,
        "plant_fires": False,
    }


def _round121_proxy_class(original, oracle_w, audit):
    """Return a harness-only model that changes only kt=2 stage-1 ZAD W."""
    oracle_w = np.asarray(oracle_w, dtype=np.float64)

    class _Round121OneStepWModel:
        def __init__(self, *model_args, **model_kwargs):
            require(
                model_kwargs.get("_nemo_ws_test_hooks") is None,
                "Round-121 trajectory proxy refuses a pre-existing hook",
            )
            model_kwargs.pop("_nemo_ws_test_hooks", None)
            self._ordinary = original(*model_args, **model_kwargs)
            hooks = model_module._NEMOWSRK3TestHooks(
                stage1_zad_w_override=jnp.asarray(oracle_w))
            self._directed = original(
                *model_args, **model_kwargs, _nemo_ws_test_hooks=hooks)
            self._round121_steps = 0
            self._round121_interventions = []
            audit.append(self)

        def __getattr__(self, name):
            return getattr(self._ordinary, name)

        def prime_step_caches(self, state):
            self._ordinary.prime_step_caches(state)
            self._directed.prime_step_caches(state)

        def step(self, *step_args, **step_kwargs):
            self._round121_steps += 1
            if self._round121_steps == 2:
                self._round121_interventions.append(2)
                return self._directed.step(*step_args, **step_kwargs)
            return self._ordinary.step(*step_args, **step_kwargs)

    return _Round121OneStepWModel


def measure_round121_trajectory(args) -> dict[str, object]:
    """Delegate the one-step W intervention to the certified harnesses."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-121 trajectory worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-121 trajectory commit stamp mismatch")
    stage, _ = _admit_round64(args)
    stage_path = args.round64_root / "oracle_momstage_kt00000002_s1.bin"
    oracle_w = owned3_with_bottom(stage["arrays"]["ww"])
    original = model_module.LatLonCGridOceanModel
    audit = []
    model_module.LatLonCGridOceanModel = _round121_proxy_class(
        original, oracle_w, audit)
    try:
        if args.trajectory_kind == "ladder":
            result = gate.run(
                gate.ROOT, stage2_root=gate.STAGE2_ROOT,
                stage3_root=gate.STAGE3_ROOT, max_step=10,
                trajectory_only=True)
            expected_steps = 10
            payload = {"ladder": result}
        else:
            import nemo_testcase_l2_gyre_year_fromrest as year_fromrest

            member = args.member_root / f"lego_seed0_{args.trajectory_tag}"
            require(not member.exists(),
                    f"Round-121 member output already exists: {member}")
            exit_code = year_fromrest.run_member(
                0, args.member_root, days=360, tag=args.trajectory_tag,
                snap_steps=6)
            require(exit_code == 0, "Round-121 year member failed")
            expected_steps = 2160
            payload = {"member": str(member)}
    finally:
        model_module.LatLonCGridOceanModel = original
    require(len(audit) == 1,
            f"Round-121 expected one harness model, got {len(audit)}")
    proxy = audit[0]
    require(proxy._round121_steps == expected_steps,
            "Round-121 trajectory step count changed")
    require(proxy._round121_interventions == [2],
            "Round-121 intervention did not execute exactly at kt=2")
    return {
        "format": "nemo-testcase-l2-gyre-round121-w-trajectory-v1",
        "status": "MEASURED",
        "worktree": stamp,
        "execution_regime": "production-jit-cpu-fp64-x64-libm",
        "trajectory_kind": args.trajectory_kind,
        "intervention": {
            "step": 2,
            "stage": 1,
            "operand": "ZAD W only",
            "record": str(stage_path),
            "record_sha256": sha256(stage_path),
            "record_producer": ROUND64_PRODUCER,
            "intervention_count": 1,
        },
        **payload,
    }


def _validate_round147_registry(registry=ROUND147_FAMILY_REGISTRY) -> None:
    require(tuple(registry) == ROUND147_FAMILY_REGISTRY,
            "Round-147 family registry changed")


def _round147_owned(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    require(value.shape == (round146_family.NY, round146_family.NX,
                            round146_family.NZ),
            "Round-147 NEMO family extent changed")
    return value[2:-2, 2:-2, :round146_family.NZ - 1]


def _round147_oracle_addends(fields: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Recover each recorded NEMO addend from adjacent cumulative writes."""
    addends = {}
    previous = {
        "u": np.zeros_like(_round147_owned(fields["after_hpg_u"])),
        "v": np.zeros_like(_round147_owned(fields["after_hpg_v"])),
    }
    for family in round146_family.FAMILIES:
        current = {
            face: _round147_owned(fields[f"after_{family}_{face}"])
            for face in ("u", "v")
        }
        addends[family] = tuple(
            current[face] - previous[face] for face in ("u", "v"))
        previous = current
    require(tuple(addends) == round146_family.FAMILIES,
            "Round-147 oracle family census changed")
    return addends


def _round147_directed_rhs_ulp(base_rhs, thickness, active):
    """Plant one consumed directed-RHS word above the fused rounding floor."""
    planted, location = _round142_propagating_rhs_ulp(
        {"rhs_u": np.asarray(base_rhs, dtype=np.float64)},
        np.asarray(active, dtype=bool), np.asarray(thickness, dtype=np.float64))
    base_rhs = np.asarray(base_rhs, dtype=np.float64)
    planted[location] = (
        base_rhs[location] + np.float64(65536.0) * np.spacing(base_rhs[location]))
    require(planted[location] != base_rhs[location],
            "Round-147 directed-RHS plant rounded away at its input")
    return planted, location


def measure_round147_rhs_families(args) -> dict[str, object]:
    """Rank NEMO's five developed RHS families through the production step."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-147 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-147 measurement commit mismatch")
    if args.plant == "family-missing-row":
        fired = False
        try:
            _validate_round147_registry(ROUND147_FAMILY_REGISTRY[:-1])
        except RuntimeError:
            fired = True
        require(fired, "Round-147 missing-row plant stayed green")
        return {"format": "nemo-testcase-l2-gyre-round147-rhs-family-v1",
                "status": "PLANT-FIRED", "worktree": stamp,
                "plant": args.plant, "plant_fires": True}
    _validate_round147_registry()

    family_root = args.round146_rhs_family_root
    validation = json.loads(
        (family_root / "round146_rhs_family_validation.json").read_text())
    require(validation.get("status") == "PASS",
            "Round-146 family record is not admitted")
    require(validation["worktree"]["commit"].lower()
            == args.expect_family_record_commit.lower(),
            "Round-146 admission commit changed")
    family_path = family_root / round146_family.RECORD
    stamp_words = family_path.with_name(family_path.name + ".stamp").read_text().split()
    require(stamp_words == [sha256(family_path),
                            args.expect_family_record_commit,
                            family_path.name],
            "Round-146 family record stamp changed")
    family_record = round146_family.read_record_bytes(family_path.read_bytes())
    require(validation["record_sha256"] == sha256(family_path),
            "Round-146 validation hash changed")

    rhs_record = read_round140_rhs(args.round140_rhs_root / ROUND140_RHS_RECORD)
    rhs_fields = rhs_record["fields"]
    split = read_round139_record(args.round140_rhs_root / ROUND139_RECORD)
    active = {"u": rhs_fields["umask"] != 0.0,
              "v": rhs_fields["vmask"] != 0.0}
    active2 = {face: active[face][..., 0] for face in ("u", "v")}
    oracle_addends = _round147_oracle_addends(family_record["fields"])

    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    eta_after = jnp.asarray(payload["ssha"])
    exposed = _round117_live_trace(
        card, state, freshwater, surface, args.execution_mode,
        eta_after_override=eta_after)
    ordinary_trace, ordinary = _round140_callback_trace(
        args, card, state, freshwater, surface, eta_after)
    trace_identity = round82._pytree_identity(
        exposed.state_after, ordinary_trace.plain_state)
    require(trace_identity["bit_exact"],
            "Round-147 live-family exposure moved production state")

    parts = exposed.operator_operands[0]
    live_total = (native_u(exposed.stage1_full_rhs[0]),
                  native_v(exposed.stage1_full_rhs[1]))
    live_names = {"hpg": "hpg", "ldf": "ldf", "vor": "vorticity",
                  "keg": "keg", "zad": "zad"}
    live_addends = {
        family: (native_u(parts[f"{name}_u"].data),
                 native_v(parts[f"{name}_v"].data))
        for family, name in live_names.items()
    }
    rhs_calibration = {
        "u": comparison(live_total[0], rhs_fields["rhs_u"], active["u"]),
        "v": comparison(live_total[1], rhs_fields["rhs_v"], active["v"]),
    }
    require(all(not row["bit_exact"] for row in rhs_calibration.values()),
            "Round-147 completed-RHS discriminator became vacuous")

    arms = {"ordinary": (ordinary_trace, ordinary)}
    overrides = {}
    for family in round146_family.FAMILIES:
        overrides[family] = tuple(
            live_total[index]
            + (oracle_addends[family][index] - live_addends[family][index])
            for index in range(2))
        arms[family] = _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            rhs_override=tuple(jnp.asarray(value)
                               for value in overrides[family]))

    def row(producer, boundary: str, face: str) -> dict:
        value = (native_u(producer[f"{boundary}_{face}"])
                 if face == "u" else native_v(producer[f"{boundary}_{face}"]))
        return comparison(value, split[f"{boundary}_{face}"], active2[face])

    rows = {
        f"{arm}_{boundary}_{face}": row(producer, boundary, face)
        for arm, (_, producer) in arms.items()
        for boundary in ("incoming", "final")
        for face in ("u", "v")
    }
    require(tuple(rows) == ROUND147_FAMILY_REGISTRY,
            "Round-147 result omitted a registered row")
    require(rows["ordinary_incoming_u"]["absolute_max"]
            == 4.2854247978022983e-13
            and rows["ordinary_incoming_v"]["absolute_max"]
            == 4.433308633699682e-13,
            "Round-147 ordinary arm does not reproduce Round 143")

    local_rows = {
        family: {
            face: comparison(live_addends[family][index],
                             oracle_addends[family][index], active[face])
            for index, face in enumerate(("u", "v"))
        }
        for family in round146_family.FAMILIES
    }
    reductions = {
        family: {
            face: 1.0 - (rows[f"{family}_incoming_{face}"]["absolute_max"]
                        / rows[f"ordinary_incoming_{face}"]["absolute_max"])
            for face in ("u", "v")
        }
        for family in round146_family.FAMILIES
    }
    largest = {
        face: max(round146_family.FAMILIES,
                  key=lambda family: reductions[family][face])
        for face in ("u", "v")
    }
    owner = largest["u"] if largest["u"] == largest["v"] else None

    if args.plant == "family-input-ulp":
        family = owner or largest["u"]
        planted_u, location = _round147_directed_rhs_ulp(
            overrides[family][0], rhs_fields["e3u"], active["u"])
        planted_override = (
            planted_u,
            overrides[family][1],
        )
        planted_trace, planted = _round140_callback_trace(
            args, card, state, freshwater, surface, eta_after,
            rhs_override=tuple(jnp.asarray(value) for value in planted_override))
        require(planted_trace.trace_state_identity["bit_exact"],
                "Round-147 family plant callback moved its plain arm")
        planted_row = comparison(
            native_u(planted["incoming_u"]),
            native_u(arms[family][1]["incoming_u"]), active2["u"])
        require(planted_row["differing_cells"] > 0,
                "Round-147 family-input ULP did not reach incoming forcing")
        return {"format": "nemo-testcase-l2-gyre-round147-rhs-family-v1",
                "status": "PLANT-FIRED", "worktree": stamp,
                "plant": args.plant, "plant_fires": True,
                "family": family, "plant_location": list(location),
                "plant_downstream_row": planted_row}

    return {
        "format": "nemo-testcase-l2-gyre-round147-rhs-family-v1",
        "status": "MEASURED", "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry, "family_record_admission": validation,
        "trace_noninterference": trace_identity,
        "rhs_calibration": rhs_calibration,
        "local_family_rows": local_rows, "rows": rows,
        "reduction_fraction": reductions, "largest_family": largest,
        "unambiguous_owner": owner,
        "hpg_prediction_confirmed": bool(
            owner == "hpg" and reductions["hpg"]["u"] > 0.5
            and reductions["hpg"]["v"] > 0.5),
        "plant": args.plant, "plant_fires": False,
        "scope": {"production_physics_changed": False,
                  "day_240_carry": "UNMEASURED",
                  "DINO": "NO-PRODUCTION-CHANGE",
                  "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
                  "OVERFLOW": "NO-PRODUCTION-CHANGE",
                  "ORCA2": "UNMEASURED-WITH-SPEC; GYRE diagnostic only"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    round_group = parser.add_mutually_exclusive_group()
    round_group.add_argument(
        "--round117", action="store_true",
        help="run the proxy-era current-tip production producer split")
    round_group.add_argument(
        "--round118", action="store_true",
        help="run the admitted direct pre-loop producer split")
    round_group.add_argument(
        "--round119", action="store_true",
        help="run the full-step stage-1 accumulator-association discriminator")
    round_group.add_argument(
        "--round120", action="store_true",
        help="run the full-step stage-1 KEG materialization discriminator")
    round_group.add_argument(
        "--round121", action="store_true",
        help="run the full-step stage-1 NEMO-W magnitude discriminator")
    round_group.add_argument(
        "--round121-trajectory", action="store_true",
        help="run the one-step NEMO-W arm through a certified trajectory")
    round_group.add_argument(
        "--round139-record-only", action="store_true",
        help="admit the developed step-1081 slow-forcing split record")
    round_group.add_argument(
        "--round140-developed", action="store_true",
        help="run the production step-1081 developed operand split")
    round_group.add_argument(
        "--round140-rhs-record-only", action="store_true",
        help="admit the developed step-1081 three-dimensional RHS record")
    round_group.add_argument(
        "--round141-rhs-developed", action="store_true",
        help="run the passive production step-1081 completed-RHS split")
    round_group.add_argument(
        "--round142-rhs-directed", action="store_true",
        help="substitute the recorded completed RHS without observing it")
    round_group.add_argument(
        "--round143-downstream-directed", action="store_true",
        help="substitute recorded depth, drag, and wind boundaries")
    round_group.add_argument(
        "--round144-wind-operands", action="store_true",
        help="substitute recorded wind operands one family at a time")
    round_group.add_argument(
        "--round145-wind-routing", action="store_true",
        help="score the production QCO-to-wind candidate without old pins")
    round_group.add_argument(
        "--round147-rhs-family-directed", action="store_true",
        help="substitute developed RHS families one at a time")
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--expect-rhs-record-commit", default="")
    parser.add_argument("--round64-root", type=Path, default=ROOT / "round64/oracle_krhs_split")
    parser.add_argument("--round46-root", type=Path, default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--round64-admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    parser.add_argument("--record-root", type=Path, default=ROOT / "round81/oracle_btstep_kt2")
    parser.add_argument("--uamid-root", type=Path, default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument("--admission", type=Path,
                        default=ROOT / "round81/oracle_btstep_kt2/round81_admission.json")
    parser.add_argument("--entry-root", type=Path,
                        default=ROOT / "round75/oracle_advmean_kt2")
    parser.add_argument(
        "--preloop-root", type=Path,
        default=ROOT / "round117/oracle_preloop_forcing")
    parser.add_argument(
        "--preloop-admission", type=Path,
        default=(ROOT / "round117/oracle_preloop_forcing"
                 / "round117_admission.json"))
    parser.add_argument(
        "--expect-preloop-record-commit", default=ROUND117_PRODUCER)
    parser.add_argument(
        "--execution-mode",
        choices=("production-jit", "production-eager"),
        default="production-jit")
    parser.add_argument(
        "--trajectory-kind", choices=("ladder", "year"), default="ladder")
    parser.add_argument(
        "--member-root", type=Path, default=ROOT / "round121")
    parser.add_argument(
        "--round139-root", type=Path,
        default=ROOT / "round139/oracle_developed_slow_forcing")
    parser.add_argument(
        "--round140-rhs-root", type=Path,
        default=ROOT / "round140/oracle_developed_rhs")
    parser.add_argument(
        "--round146-rhs-family-root", type=Path,
        default=ROOT / "round146/oracle_developed_rhs_families")
    parser.add_argument("--expect-family-record-commit", default="")
    parser.add_argument(
        "--daily-root", type=Path,
        default=ROOT / "round132/oracle_daily_restarts")
    parser.add_argument(
        "--daily-audit", type=Path,
        default=ROOT / "round136/daily_record_audit.json")
    parser.add_argument("--trajectory-tag", default="round121_w")
    parser.add_argument("--plant", choices=(
                            "none", "e3-ulp", "rhs-ulp", "final-ulp",
                            "incoming-ulp", "association-hpg-ulp",
                            "association-keg-ulp", "zad-w-ulp",
                            "record-stamp", "record-header",
                            "record-truncation", "record-replay-ulp",
                            "developed-missing-row", "rhs-record-stamp",
                            "rhs-record-header", "rhs-record-truncation",
                            "rhs-record-replay-ulp", "rhs-missing-row",
                            "rhs-observer-ulp", "rhs-directed-missing-row",
                            "rhs-directed-ulp", "downstream-missing-row",
                            "downstream-depth-ulp", "wind-missing-row",
                            "wind-stress-ulp", "wind-inverse-depth-ulp",
                            "family-missing-row", "family-input-ulp"),
                        default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = (
            measure_round147_rhs_families(args)
            if args.round147_rhs_family_directed else
            measure_round144_wind_operands(args)
            if (args.round144_wind_operands or args.round145_wind_routing) else
            (measure_round143_downstream_directed(args)
            if args.round143_downstream_directed else
            (measure_round142_rhs_directed(args)
            if args.round142_rhs_directed else
            (measure_round141_rhs_developed(args)
            if args.round141_rhs_developed else
            (measure_round140_rhs_record(args)
            if args.round140_rhs_record_only else
            (measure_round140_developed(args)
            if args.round140_developed else
            (measure_round139_record(args)
            if args.round139_record_only else
            (measure_round121_trajectory(args)
            if args.round121_trajectory else
            (measure_round117(args)
            if (args.round117 or args.round118 or args.round119
                or args.round120 or args.round121) else measure(args))))))))))
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError, KeyError, ValueError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.plant != "none":
        prefix = ("ROUND147 RHS FAMILY" if args.round147_rhs_family_directed else (
            "ROUND145 WIND" if args.round145_wind_routing else
            ("ROUND144 WIND" if args.round144_wind_operands else
            ("ROUND143 DOWNSTREAM" if args.round143_downstream_directed else
            ("ROUND142 RHS" if args.round142_rhs_directed else
            ("ROUND141 RHS" if args.round141_rhs_developed else
            ("ROUND140 RHS" if args.round140_rhs_record_only else
            ("ROUND140 DEVELOPED" if args.round140_developed else
            ("ROUND139 RECORD" if args.round139_record_only else
            ("ROUND121" if args.round121 else
            ("ROUND120" if args.round120 else
            ("ROUND119" if args.round119 else
            ("ROUND118" if args.round118 else
             ("ROUND117" if args.round117 else "ROUND83"))))))))))))))
        state = "STATUS PLANT-FIRED" if report["plant_fires"] else "STATUS PLANT-INERT"
        print(f"{prefix} {args.plant.upper()} {state}")
        return 1
    if args.round139_record_only:
        print(
            "ROUND139 SLOW-FORCING RECORD " + report["status"] + ": "
            + report["record_sha256"]
        )
        return 0 if report["status"] == "PASS" else 1
    if args.round140_developed:
        print(
            "ROUND140 DEVELOPED SLOW-FORCING MEASURED: first="
            + repr(report["first_non_bit_operand"])
        )
        return 0
    if args.round140_rhs_record_only:
        print(
            "ROUND140 DEVELOPED RHS RECORD " + report["status"] + ": "
            + report.get("record_sha256", "plant")
        )
        return 0 if report["status"] == "PASS" else 1
    if args.round141_rhs_developed:
        print(
            "ROUND141 DEVELOPED RHS MEASURED: first="
            + repr(report["first_non_bit_operand"])
        )
        return 0
    if args.round142_rhs_directed:
        print(
            "ROUND142 DEVELOPED RHS DIRECTED: family_confirmed="
            + repr(report["completed_rhs_family_magnitude_confirmed"])
            + " exact=" + repr(report["exact_closure_prediction_confirmed"])
        )
        return 0
    if args.round143_downstream_directed:
        print(
            "ROUND143 DOWNSTREAM DIRECTED: first="
            + repr(report["first_closing_family"])
        )
        return 0
    if args.round147_rhs_family_directed:
        print("ROUND147 RHS FAMILY DIRECTED: owner="
              + repr(report["unambiguous_owner"]))
        return 0
    if args.round144_wind_operands or args.round145_wind_routing:
        print(
            ("ROUND145 WIND ROUTING: first=" if args.round145_wind_routing
             else "ROUND144 WIND OPERANDS: first=")
            + repr(report["first_moving_family"])
            + " closes=" + repr(report["first_closing_family"])
        )
        return 0
    if args.round121_trajectory:
        print("ROUND121 W TRAJECTORY " + report["status"] + ": "
              + report["trajectory_kind"])
        return 0
    if (args.round117 or args.round118 or args.round119 or args.round120
            or args.round121):
        prefix = (
            "ROUND121" if args.round121 else
            ("ROUND120" if args.round120 else
            ("ROUND119" if args.round119 else
             ("ROUND118" if args.round118 else "ROUND117"))))
        print(
            prefix + " PRODUCER " + report["status"] + ": first="
            + repr(report["producer_split"]["first_direct_non_bit"])
        )
        return 0 if report["status"] in ("CONFIRMED", "MEASURED") else 1
    print(f"ROUND83 SLOW FORCING {report['status']}: first={report['first_non_bit_statement']}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

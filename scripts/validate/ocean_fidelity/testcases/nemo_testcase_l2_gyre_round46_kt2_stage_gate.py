#!/usr/bin/env python3
"""Round-46 GYRE kt=2, stage-by-stage momentum fidelity gate.

The acquisition phase uses ``--mode validate``.  It parses all six named,
ranked streams through physical EOF, replays compiled WZV/KEG/ZAD arithmetic,
checks the widened round-40/41 kt=1 twins, and verifies the producer SHA.  The
operator resumes with ``--mode all`` for NEMO-given-input and kt=2 trajectory
scores.  A non-executed operator is represented by a header presence flag,
never by an all-zero pseudo-tendency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    SCALAR_MATH_ROOT,
    STAGE_WW_ROOT,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_entry,
    read_stage as read_phase3_stage,
    read_stage_ww,
    read_transport,
    require,
    score,
)
from nemo_testcase_l2_gyre_round40_stage3_operators import read_stage3_terms
from nemo_testcase_l2_gyre_round41_dynadv_split import (
    _keg_replay,
    _model_terms,
    _zad_replay,
    read_split,
)
from nemo_testcase_l2_gyre_round21_admission import _compare_self_describing

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage")
ROUND41 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split")
ADVMEAN_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round75/oracle_advmean_kt2")
MEMORY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48/oracle_bt_memory")
BTSTEP_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round81/oracle_btstep_kt2")
STAGE_CLOSURE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round94/oracle_stage_closure")
STAGE1_W_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round98/oracle_stage1_w_walk")
STAGE1_R3_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round99/"
    "oracle_stage1_r3_operands")
TKE_STATEMENT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round101/"
    "oracle_tke_statement_walk")
TKE_OPERAND_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round59/"
    "oracle_tke_operands/oracle_tke_operands_kt00000002.bin")
YEAR_ENTRY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/nemo_seed0")
MAGIC = "NEMO_L2_R46STG1"
DIMS = (36, 26, 31)
OWNED_DIMS = (32, 22, 31)
TKE_STATEMENT_MAGIC = b"NEMO_L2_R101TKE "
TKE_STATEMENT_FIELDS = (
    "en_entry", "en_after_boundaries", "en_after_langmuir",
    "rhs_pre_sweep", "en_post_sweep",
)
TKE_RHS_INTERMEDIATES = (
    "p_avt_rn2",
    "zfact3_dissl",
    "dissipation_product",
    "after_stratification",
    "parenthesized_sum",
    "dt_product",
    "masked_increment",
    "final_accumulation",
)
TKE_RHS_POSTHOC_OPERANDS = ("p_avt_operand", "rn2_operand")
TKE_RHS_SELECTORS = TKE_RHS_INTERMEDIATES + TKE_RHS_POSTHOC_OPERANDS
BN2_INTERMEDIATES = (
    "zrw",
    "zaw",
    "zbw",
    "temperature_contribution",
    "salinity_contribution",
    "contribution_difference",
    "gravity_product",
    "thickness_division",
    "masked_rn2",
)
STAGE1_HANDOFF_BOUNDARIES = (
    "full_accumulator",
    "projected_accumulator",
    "post_transport",
    "post_w_zad",
    "rk_input",
    "raw_rk",
    "corrected_output",
)
OWNED_3D_FIELDS = {"tke_en", "tke_avt_k", "tke_dissl"}
HEADER_FIELDS = (
    "version",
    "kt",
    "stage",
    "Kbb",
    "Kmm",
    "Krhs",
    "Kaa",
    "jpi",
    "jpj",
    "jpk",
    "jpkm1",
    "ntsi",
    "ntei",
    "ntsj",
    "ntej",
    "bits",
)
STAGES = tuple((kt, stage) for kt in (1, 2) for stage in (1, 2, 3))
STAGE_SLOTS = {
    (1, 1): (1, 1),
    (1, 2): (1, 3),
    (1, 3): (1, 2),
    (2, 1): (3, 3),
    (2, 2): (3, 1),
    (2, 3): (3, 2),
}
PRESENCE = {
    1: {"hpg": 1, "vor": 1, "keg": 1, "zad": 1, "ldf": 1, "zdf": 0},
    2: {"hpg": 1, "vor": 1, "keg": 1, "zad": 1, "ldf": 0, "zdf": 0},
    3: {"hpg": 1, "vor": 1, "keg": 1, "zad": 1, "ldf": 1, "zdf": 1},
}
REQUIRED = {
    "u_Kbb",
    "v_Kbb",
    "u_Kmm",
    "v_Kmm",
    "u_Kaa_in",
    "v_Kaa_in",
    "T_Kbb",
    "S_Kbb",
    "ssh_Kbb",
    "T_Kmm",
    "S_Kmm",
    "ssh_Kmm",
    "T_Kaa_in",
    "S_Kaa_in",
    "ssh_Kaa",
    "rhd_in",
    "r1_Dt",
    "tke_en",
    "tke_avm_k",
    "tke_avt_k",
    "tke_dissl",
    "r3t_Kbb",
    "r3u_Kbb",
    "r3v_Kbb",
    "r3t_Kmm",
    "r3u_Kmm",
    "r3v_Kmm",
    "r3t_Kaa",
    "r3u_Kaa",
    "r3v_Kaa",
    "e3t_Kbb",
    "e3u_Kbb",
    "e3v_Kbb",
    "e3w_Kbb",
    "e3t_Kmm",
    "e3u_Kmm",
    "e3v_Kmm",
    "e3w_Kmm",
    "e3t_Kaa",
    "e3u_Kaa",
    "e3v_Kaa",
    "e3w_Kaa",
    "e3t_0",
    "e3u_0",
    "e3v_0",
    "e3w_0",
    "umask",
    "vmask",
    "tmask",
    "wmask",
    "e1e2t",
    "e1e2u",
    "e1e2v",
    "r1_e1e2u",
    "r1_e1e2v",
    "r1_e1u",
    "r1_e2v",
    "r1_e1e2t",
    "e2u",
    "e1v",
    "uu_b_Kbb",
    "vv_b_Kbb",
    "uu_b_Kmm",
    "vv_b_Kmm",
    "uu_b_Kaa",
    "vv_b_Kaa",
    "rhs_entry_u",
    "rhs_entry_v",
    "ww",
    "wsd_effective",
    "after_hpg_u",
    "after_hpg_v",
    "after_vor_u",
    "after_vor_v",
    "after_keg_u",
    "after_keg_v",
    "after_zad_u",
    "after_zad_v",
    "after_adv_u",
    "after_adv_v",
    "post_baro_u",
    "post_baro_v",
    "has_hpg",
    "has_vor",
    "has_keg",
    "has_zad",
    "has_ldf",
    "has_zdf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _xy(raw: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return raw.reshape((nx, ny), order="F").T.copy()


def _xyz(raw: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return raw.reshape((nx, ny, nz), order="F").transpose(1, 0, 2).copy()


def _owned3(value, nlev: int = 30) -> np.ndarray:
    return np.asarray(value)[2:-2, 2:-2, :nlev]


def _owned2(value) -> np.ndarray:
    return np.asarray(value)[2:-2, 2:-2]


def read_stage1_w_walk_record(path: Path) -> dict:
    """Read the direct kt=1 stage-1 transport-W operand acquisition."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kbb, kmm, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_R98W_1", f"{path}: bad magic {magic!r}")
    require((version, kt, kbb, kmm, kaa) == (1, 1, 1, 1, 3),
            f"{path}: wrong clock/slots {header}")
    require((nx, ny, nz, bits) == (*DIMS, 64),
            f"{path}: wrong dimensions/dtype {header}")
    n2, n3 = nx * ny, nx * ny * nz
    # The compiled record writes hdiv and ze3div at their declared local
    # one-halo extent (Nis0-1:Nie0+1,Njs0-1:Nje0+1,jpk).  GYRE has nn_hls=2,
    # hence 34x24x31 rather than the full 36x26x31 used by e3t_3d and pww.
    local_nx, local_ny = nx - 2, ny - 2
    local_n3 = local_nx * local_ny * nz
    require(values.size == 2 * local_n3 + 2 * n3 + 2 * n2 + 1,
            f"{path}: wrong payload size {values.size}")
    offset = 0
    result = {}
    for name in ("hdiv", "e3div"):
        result[name] = _xyz(
            values[offset:offset + local_n3], local_nx, local_ny, nz)
        offset += local_n3
    for name in ("r3_kaa", "r3_kbb"):
        result[name] = _xy(values[offset:offset + n2], nx, ny)
        offset += n2
    result["e3t_0"] = _xyz(values[offset:offset + n3], nx, ny, nz)
    offset += n3
    result["r1_dt"] = float(values[offset])
    offset += 1
    result["ww"] = _xyz(values[offset:offset + n3], nx, ny, nz)
    offset += n3
    require(offset == values.size, f"{path}: reader did not consume physical EOF")
    result["header"] = {
        "version": version, "kt": kt, "Kbb": kbb, "Kmm": kmm,
        "Kaa": kaa, "jpi": nx, "jpj": ny, "jpk": nz, "bits": bits,
        "local_jpi": local_nx, "local_jpj": local_ny,
    }
    return result


def read_stage1_r3_operand_record(path: Path) -> dict:
    """Read the same-call inputs and output of ``dom_qco_r3c_RK3``."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kbb, kmm, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_R99R3_1", f"{path}: bad magic {magic!r}")
    require((version, kt, kbb, kmm, kaa) == (1, 1, 1, 1, 3),
            f"{path}: wrong clock/slots {header}")
    require((nx, ny, nz, bits) == (*DIMS, 64),
            f"{path}: wrong dimensions/dtype {header}")
    n2 = nx * ny
    require(values.size == 6 * n2, f"{path}: wrong payload size {values.size}")
    result = {}
    offset = 0
    for name in (
        "ssh_kaa", "r1_ht_0", "r3_kaa", "ssh_kbb", "r3_kbb", "ht_0",
    ):
        result[name] = _xy(values[offset:offset + n2], nx, ny)
        offset += n2
    require(offset == values.size, f"{path}: reader did not consume physical EOF")
    result["header"] = {
        "version": version, "kt": kt, "Kbb": kbb, "Kmm": kmm,
        "Kaa": kaa, "jpi": nx, "jpj": ny, "jpk": nz, "bits": bits,
    }
    return result


def read_tke_statement_walk_record(
    path: Path, *, plant: str | None = None, expected_kt: int = 2,
    expected_slots: tuple[int, int] = (3, 3),
) -> dict:
    """Read the fixed kt=2 TKE statement-boundary stream through EOF."""
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    offset = 0

    def take(count: int) -> bytes:
        nonlocal offset
        require(offset + count <= len(raw), f"{path}: record is truncated")
        value = raw[offset:offset + count]
        offset += count
        return value

    magic = take(16)
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == TKE_STATEMENT_MAGIC, f"{path}: bad magic {magic!r}")
    values = list(struct.unpack("=13i", take(13 * 4)))
    keys = (
        "version", "kt", "Kbb", "Kmm", "jpi", "jpj", "jpk", "jpkm1",
        "ntsi", "ntei", "ntsj", "ntej", "bits",
    )
    header = dict(zip(keys, values, strict=True))
    require((header["version"], header["kt"], header["Kbb"],
             header["Kmm"]) == (1, expected_kt, *expected_slots),
            f"{path}: wrong version/clock/slots {header}")
    require((header["jpi"], header["jpj"], header["jpk"],
             header["jpkm1"]) == (*DIMS, 30),
            f"{path}: wrong global dimensions {header}")
    require((header["ntsi"], header["ntei"], header["ntsj"],
             header["ntej"], header["bits"]) == (3, 34, 3, 24, 64),
            f"{path}: wrong owned bounds/dtype {header}")
    nx = header["ntei"] - header["ntsi"] + 1
    ny = header["ntej"] - header["ntsj"] + 1
    nz = header["jpk"]
    count = nx * ny * nz
    arrays = {}
    for name in TKE_STATEMENT_FIELDS:
        values = np.frombuffer(take(count * 8), dtype="=f8").copy()
        arrays[name] = values.reshape((nx, ny, nz), order="F")
        require(np.all(np.isfinite(arrays[name])),
                f"{path}: {name} contains NaN/Inf")
    require(offset == len(raw),
            f"{path}: record has {len(raw) - offset} trailing bytes")
    require(len(raw) == 873028,
            f"{path}: physical EOF {len(raw)} != registered 873028")
    return {"header": header, "arrays": arrays, "sha256": sha256(path)}


def read_admitted_tke_statement_walk(
    root: Path, *, plant: str | None = None,
) -> dict:
    """Admit the Round-101 stream only after digest and producer checks."""
    path = root / "oracle_tke_statement_walk_kt00000002.bin"
    producer_path = root / "producer_commit.txt"
    stamp_path = path.with_name(path.name + ".stamp")
    require(path.is_file(), f"missing {path}")
    require(producer_path.is_file(), f"missing {producer_path}")
    require(stamp_path.is_file(), f"missing {stamp_path}")
    producer = producer_path.read_text(encoding="utf-8").strip()
    require(len(producer) == 40, f"{producer_path}: malformed producer commit")
    parts = stamp_path.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, f"{stamp_path}: malformed stamp")
    expected = "0" * 40 if plant == "stamp" else producer
    require(parts[0] == sha256(path), f"{stamp_path}: digest mismatch")
    require(parts[1] == expected, f"{stamp_path}: producer commit mismatch")
    require(parts[2] == path.name, f"{stamp_path}: record name mismatch")
    record = read_tke_statement_walk_record(
        path, plant=plant if plant in {"header", "truncation"} else None)
    record["producer_commit"] = producer
    return record


def _tke_statement_duplicate_rows(record: dict, legacy: dict, *,
                                  plant_ulp: bool = False) -> list[dict]:
    """Bit-test Round-59 duplicates on their compiled-consumed domains.

    The Round-59 matrix writer deliberately zero-fills ``jpk`` because NEMO's
    TKE solve consumes only ``1:jpkm1``.  The Round-101 statement writer
    captures the live ``en`` image at every level.  Preserve the complete
    stored-array comparison as a diagnostic, but do not mistake the two
    different sentinel policies for a moved statement.
    """
    pairs = {
        "en_entry": "en_entry",
        "rhs_pre_sweep": "rhs_pre_sweep",
        "en_post_sweep": "en_post_sweep",
    }
    rows = []
    for name, legacy_name in pairs.items():
        candidate = np.asarray(record["arrays"][name], dtype=np.float64)
        reference = np.asarray(legacy["arrays"][legacy_name], dtype=np.float64)
        require(candidate.shape == reference.shape,
                f"TKE duplicate {name} shape mismatch: "
                f"{candidate.shape} != {reference.shape}")
        if plant_ulp and name == "rhs_pre_sweep":
            candidate = candidate.copy()
            candidate[0, 0, 1] = np.nextafter(
                candidate[0, 0, 1], np.float64(np.inf))
        domains = {
            "complete_stored_array": (...,),
            "compiled_consumed_1_jpkm1": (..., slice(0, 30)),
            "unconsumed_jpk_sentinel": (..., slice(30, 31)),
        }
        for domain, selector in domains.items():
            left = np.ascontiguousarray(candidate[selector])
            right = np.ascontiguousarray(reference[selector])
            unequal = int(np.count_nonzero(
                left.view(np.uint64) != right.view(np.uint64)))
            binding = (
                domain == "complete_stored_array"
                if name in {"en_entry", "en_post_sweep"}
                else domain == "compiled_consumed_1_jpkm1"
            )
            rows.append({
                "name": (
                    f"GYRE-zco.kt2.tke_statement_record.{name}.{domain}"
                ),
                "field": name,
                "domain": domain,
                "admission_binding": binding,
                "compared_cells": int(left.size),
                "n_unequal": unequal,
                "absolute_max": float(np.max(np.abs(left - right))),
                "classification": "BIT" if unequal == 0 else "DEBT",
            })
    if plant_ulp:
        target = next(
            row for row in rows
            if row["field"] == "rhs_pre_sweep"
            and row["domain"] == "compiled_consumed_1_jpkm1"
        )
        require(target["n_unequal"] == 1,
                "TKE statement one-ULP plant did not flip exactly one cell")
    return rows


def read_admitted_stage1_w_walk(root: Path, *, plant_stamp: bool = False) -> dict:
    """Read the direct W record only after its digest/producer stamp closes."""
    path = root / "oracle_stage1_w_walk_kt00000001.bin"
    producer_path = root / "producer_commit.txt"
    stamp_path = path.with_name(path.name + ".stamp")
    require(producer_path.is_file(), f"missing {producer_path}")
    require(stamp_path.is_file(), f"missing {stamp_path}")
    producer = producer_path.read_text(encoding="utf-8").strip()
    require(len(producer) == 40, f"{producer_path}: malformed producer commit")
    parts = stamp_path.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, f"{stamp_path}: malformed stamp")
    expected = "0" * 40 if plant_stamp else producer
    require(parts[0] == sha256(path), f"{stamp_path}: digest mismatch")
    require(parts[1] == expected, f"{stamp_path}: producer commit mismatch")
    require(parts[2] == path.name, f"{stamp_path}: record name mismatch")
    record = read_stage1_w_walk_record(path)
    record["sha256"] = parts[0]
    record["producer_commit"] = producer
    return record


def read_admitted_stage1_r3_operands(
    root: Path, *, plant_stamp: bool = False,
) -> dict:
    """Read the same-call ratio record only after its producer stamp closes."""
    path = root / "oracle_stage1_r3_operands_kt00000001.bin"
    producer_path = root / "producer_commit.txt"
    stamp_path = path.with_name(path.name + ".stamp")
    require(producer_path.is_file(), f"missing {producer_path}")
    require(stamp_path.is_file(), f"missing {stamp_path}")
    producer = producer_path.read_text(encoding="utf-8").strip()
    require(len(producer) == 40, f"{producer_path}: malformed producer commit")
    parts = stamp_path.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, f"{stamp_path}: malformed stamp")
    expected = "0" * 40 if plant_stamp else producer
    require(parts[0] == sha256(path), f"{stamp_path}: digest mismatch")
    require(parts[1] == expected, f"{stamp_path}: producer commit mismatch")
    require(parts[2] == path.name, f"{stamp_path}: record name mismatch")
    record = read_stage1_r3_operand_record(path)
    record["sha256"] = parts[0]
    record["producer_commit"] = producer
    return record


def read_stage(
    path: Path,
    *,
    expected_kt: int | None = None,
    expected_stage: int | None = None,
    plant: str | None = None,
) -> dict:
    """Fail-closed named/ranked reader; duplicate, extra and short fields fail."""
    with path.open("rb") as f:
        raw_magic = f.read(16)
        require(len(raw_magic) == 16, f"{path}: short magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = f.read(64)
        require(len(raw_header) == 64, f"{path}: short header")
        values = list(struct.unpack("=16i", raw_header))
        if plant == "header":
            values[2] = 9
        elif plant == "slot":
            values[3] = 1 if values[3] == 3 else 3
        header = dict(zip(HEADER_FIELDS, values, strict=True))
        require(magic == MAGIC, f"{path}: bad magic {magic!r}")
        require(
            header["version"] == 1 and header["bits"] == 64, f"{path}: wrong version/dtype {header}"
        )
        require(
            (header["jpi"], header["jpj"], header["jpk"], header["jpkm1"]) == (*DIMS, 30),
            f"{path}: wrong domain {header}",
        )
        require(
            (header["ntsi"], header["ntei"], header["ntsj"], header["ntej"]) == (3, 34, 3, 24),
            f"{path}: wrong owned bounds {header}",
        )
        observed_step = (header["kt"], header["stage"])
        require(observed_step in STAGES, f"{path}: wrong kt/stage {header}")
        require((expected_kt is None) == (expected_stage is None),
                "read_stage requires expected_kt and expected_stage together")
        if expected_kt is not None:
            require(
                observed_step == (expected_kt, expected_stage),
                f"{path}: record identity mismatch: expected "
                f"kt={expected_kt}/stage={expected_stage}, got "
                f"kt={header['kt']}/stage={header['stage']}",
            )
        expected_kbb, expected_kmm = STAGE_SLOTS[observed_step]
        require(
            (header["Kbb"], header["Kmm"]) == (expected_kbb, expected_kmm),
            f"{path}: stage-slot mismatch: kt={header['kt']}/"
            f"stage={header['stage']} requires Kbb={expected_kbb}/"
            f"Kmm={expected_kmm}, got Kbb={header['Kbb']}/"
            f"Kmm={header['Kmm']}",
        )
        arrays: dict[str, np.ndarray | float] = {}
        while True:
            raw_name = f.read(16)
            if not raw_name:
                break
            require(len(raw_name) == 16, f"{path}: truncated field name")
            name = raw_name.decode("ascii").rstrip()
            require(name not in arrays, f"{path}: duplicate field {name!r}")
            shape = f.read(16)
            require(len(shape) == 16, f"{path}: short shape for {name}")
            rank, n1, n2, n3 = struct.unpack("=4i", shape)
            require(rank in (0, 2, 3), f"{path}: invalid rank for {name}")
            count = 1 if rank == 0 else n1 * n2 * (n3 if rank == 3 else 1)
            raw = f.read(8 * count)
            if plant == "truncation" and name == "u_Kbb":
                raw = raw[:-8]
            require(len(raw) == 8 * count, f"{path}: short payload for {name}")
            a = np.frombuffer(raw, dtype=np.float64)
            require(np.isfinite(a).all(), f"{path}: non-finite {name}")
            if rank == 0:
                arrays[name] = float(a[0])
            elif rank == 2:
                require((n1, n2, n3) == (*DIMS[:2], 1), f"{path}: bad 2-D extents for {name}")
                arrays[name] = _xy(a, n1, n2)
            else:
                expected = OWNED_DIMS if name in OWNED_3D_FIELDS else DIMS
                require((n1, n2, n3) == expected,
                        f"{path}: bad 3-D extents for {name}: "
                        f"{(n1, n2, n3)} != {expected}")
                arrays[name] = _xyz(a, n1, n2, n3)
    missing = REQUIRED - arrays.keys()
    # Stage-specific fields are additive to the common contract.
    if header["stage"] == 3:
        missing |= {
            "after_ldf_u",
            "after_ldf_v",
            "pre_zdf_rhs_u",
            "pre_zdf_rhs_v",
            "post_zdf_u",
            "post_zdf_v",
        } - arrays.keys()
    else:
        missing |= {"post_update_u", "post_update_v"} - arrays.keys()
    require(not missing, f"{path}: missing fields {sorted(missing)}")
    allowed = set(REQUIRED) | {
        "after_ldf_u",
        "after_ldf_v",
        "pre_zdf_rhs_u",
        "pre_zdf_rhs_v",
        "post_zdf_u",
        "post_zdf_v",
        "post_update_u",
        "post_update_v",
        "pre_baro_u",
        "pre_baro_v",
    }
    require(
        not (arrays.keys() - allowed), f"{path}: unknown fields {sorted(arrays.keys() - allowed)}"
    )
    for op, expected in PRESENCE[header["stage"]].items():
        require(
            arrays[f"has_{op}"] == float(expected), f"{path}: false operator-presence flag for {op}"
        )
    require(
        np.count_nonzero(arrays["wsd_effective"]) == 0, f"{path}: GYRE wsd arm is not resolved zero"
    )
    return {"header": header, "arrays": arrays}


def _wzv_replay(a: dict) -> np.ndarray:
    """Compiled divhor.f90:123-154 + sshwzv.f90:293-299 replay."""
    u, v = a["u_Kmm"], a["v_Kmm"]
    out = np.zeros(DIMS[::-1], dtype=np.float64).transpose(1, 0, 2)
    # out is (ny,nx,nz); pww bottom was initialized to zero by NEMO.
    out = np.zeros((DIMS[1], DIMS[0], DIMS[2]), dtype=np.float64)
    ze3div = np.zeros_like(out)
    for k in range(DIMS[2] - 1):
        for j in range(1, 25):
            for i in range(1, 35):
                e2u, e2uw = a["e2u"][j, i], a["e2u"][j, i - 1]
                e1v, e1vs = a["e1v"][j, i], a["e1v"][j - 1, i]
                zu = np.float64(e2u * a["e3u_Kmm"][j, i, k])
                zu = np.float64(zu * u[j, i, k])
                zuw = np.float64(e2uw * a["e3u_Kmm"][j, i - 1, k])
                zuw = np.float64(zuw * u[j, i - 1, k])
                zv = np.float64(e1v * a["e3v_Kmm"][j, i, k])
                zv = np.float64(zv * v[j, i, k])
                zvs = np.float64(e1vs * a["e3v_Kmm"][j - 1, i, k])
                zvs = np.float64(zvs * v[j - 1, i, k])
                hdiv = np.float64(np.float64(zu - zuw) + np.float64(zv - zvs))
                hdiv = np.float64(hdiv * a["r1_e1e2t"][j, i])
                hdiv = np.float64(hdiv / a["e3t_Kmm"][j, i, k])
                ze3div[j, i, k] = np.float64(hdiv * a["e3t_Kmm"][j, i, k])
    for k in range(DIMS[2] - 2, -1, -1):
        for j in range(1, 25):
            for i in range(1, 35):
                stretch = np.float64(a["r3t_Kaa"][j, i] - a["r3t_Kbb"][j, i])
                stretch = np.float64(a["r1_Dt"] * a["e3t_0"][j, i, k] * stretch)
                total = np.float64(ze3div[j, i, k] + stretch)
                out[j, i, k] = np.float64(out[j, i, k + 1] - total) * a["tmask"][j, i, k]
    return out


def _split_view(a: dict) -> dict:
    return {
        "uu_Kmm": a["u_Kmm"],
        "vv_Kmm": a["v_Kmm"],
        "before_keg_u": a["after_vor_u"],
        "before_keg_v": a["after_vor_v"],
        "after_keg_u": a["after_keg_u"],
        "after_keg_v": a["after_keg_v"],
        "after_zad_u": a["after_zad_u"],
        "after_zad_v": a["after_zad_v"],
        **{
            k: a[k]
            for k in (
                "ww",
                "e3t_Kmm",
                "e3u_Kmm",
                "e3v_Kmm",
                "e3w_Kmm",
                "e3t_0",
                "e3u_0",
                "e3v_0",
                "e3w_0",
                "e1e2t",
                "e1e2u",
                "e1e2v",
                "r1_e1u",
                "r1_e2v",
                "r1_e1e2u",
                "r1_e1e2v",
                "tmask",
                "umask",
                "vmask",
                "wmask",
            )
        },
    }


def _calibrate(records: dict, plant: str | None) -> dict:
    rows = {}
    for key, record in records.items():
        a = record["arrays"]
        wzv_inputs = a
        if key == (2, 1):
            # stp2d.f90:157-162 updates r3t(Kaa)=ssh(Kaa)*r1_ht_0 after
            # r46_begin recorded the named bundle and immediately before WZV.
            # The same bundle carries the developed Kbb identity
            # r3t(Kbb)=ssh(Kbb)*r1_ht_0, so recover NEMO's stored reciprocal
            # from those two operands and replay the missing statement.  Every
            # wet WZV cell has nonzero developed ssh; dry cells are masked out
            # by sshwzv.f90:297-298.
            wet = a["tmask"][..., 0] > 0.5
            ext2 = np.zeros_like(wet)
            ext2[1:25, 1:35] = True
            require(np.all(a["ssh_Kbb"][wet & ext2] != 0.0),
                    "kt2 stage1 cannot recover stored r1_ht_0 from zero ssh_Kbb")
            r1_ht_0 = np.divide(
                a["r3t_Kbb"], a["ssh_Kbb"],
                out=np.zeros_like(a["r3t_Kbb"]),
                where=a["ssh_Kbb"] != 0.0,
            )
            r3t_kaa = a["ssh_Kaa"] * r1_ht_0
            rows["kt2.s1.r3t_Kaa_replay"] = int(np.count_nonzero(
                (a["ssh_Kbb"] * r1_ht_0)[wet & ext2]
                != a["r3t_Kbb"][wet & ext2]
            ))
            wzv_inputs = {**a, "r3t_Kaa": r3t_kaa}
        ww = _wzv_replay(wzv_inputs)
        keg_u, keg_v = _keg_replay(_split_view(a))
        zad_u, zad_v = _zad_replay(_split_view(a))
        if plant == "calibration" and key == (2, 1):
            keg_u[3, 3, 0] = np.nextafter(keg_u[3, 3, 0], np.inf)
        ext = np.s_[1:25, 1:35, :]
        rows[f"kt{key[0]}.s{key[1]}.ww"] = int(np.count_nonzero(ww[ext] != a["ww"][ext]))
        rows[f"kt{key[0]}.s{key[1]}.keg_u"] = int(np.count_nonzero(keg_u != a["after_keg_u"]))
        rows[f"kt{key[0]}.s{key[1]}.keg_v"] = int(np.count_nonzero(keg_v != a["after_keg_v"]))
        rows[f"kt{key[0]}.s{key[1]}.zad_u"] = int(np.count_nonzero(zad_u != a["after_zad_u"]))
        rows[f"kt{key[0]}.s{key[1]}.zad_v"] = int(np.count_nonzero(zad_v != a["after_zad_v"]))
    require(all(v == 0 for v in rows.values()), f"source calibration failed: {rows}")
    return rows


def _given_inputs(
    records: dict,
    plant: str | None,
    *,
    kt: int = 2,
    stages: tuple[int, ...] = (1, 2, 3),
) -> tuple[list[dict], dict]:
    """Run legoESM operator routes on complete NEMO stage bundles."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    ocean = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    rows = []
    for stage in stages:
        a = records[(kt, stage)]["arrays"]
        un, vn = _owned3(a["u_Kmm"]), _owned3(a["v_Kmm"])
        ju = np.concatenate([un[:, -1:, :], un], axis=1)
        jv = np.concatenate([np.zeros_like(vn[:1]), vn], axis=0)
        ub, vb = _owned3(a["u_Kbb"]), _owned3(a["v_Kbb"])
        ju_b = np.concatenate([ub[:, -1:, :], ub], axis=1)
        jv_b = np.concatenate([np.zeros_like(vb[:1]), vb], axis=0)
        st = card.recipe.initial_state._replace(
            u=card.recipe.initial_state.u.replace(data=jnp.asarray(ju)),
            v=card.recipe.initial_state.v.replace(data=jnp.asarray(jv)),
            T=card.recipe.initial_state.T.replace(data=jnp.asarray(_owned3(a["T_Kmm"]))),
            S=card.recipe.initial_state.S.replace(data=jnp.asarray(_owned3(a["S_Kmm"]))),
            eta=card.recipe.initial_state.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kmm"]))),
        )
        _, surface = _surface_forcings(card, st, kt)
        hu0, hv0 = _owned3(a["e3u_Kmm"]), _owned3(a["e3v_Kmm"])
        hu = np.concatenate([hu0[:, -1:, :], hu0], axis=1)
        hv = np.concatenate([np.ones_like(hv0[:1]), hv0], axis=0)

        def operators(state):
            return ocean.tendencies(
                state,
                surface,
                dt=card.dt_s,
                momentum_only=True,
                skip_lateral_viscosity=(stage == 2),
                # Compiled dyn_ldf reads Kbb at every stage, whereas the rest
                # of the RHS reads this stage's Kmm state.
                ldf_state=(state.T.data, state.S.data,
                           jnp.asarray(ju_b), jnp.asarray(jv_b)),
                momentum_flux_face_thickness=(jnp.asarray(hu), jnp.asarray(hv)),
                zad_continuity_dt=np.float64(1.0 / a["r1_Dt"]),
                nemo_operator_association=True,
                return_nemo_operator_components=True,
            )

        _, diagnostics, components = jax.jit(operators)(st)
        vor_before_u = a["after_ldf_u"] if stage == 1 else a["after_hpg_u"]
        vor_before_v = a["after_ldf_v"] if stage == 1 else a["after_hpg_v"]
        references = {
            "hpg": (a["after_hpg_u"], a["after_hpg_v"]),
            "vorticity": (a["after_vor_u"] - vor_before_u, a["after_vor_v"] - vor_before_v),
            "advection": (a["after_adv_u"] - a["after_vor_u"], a["after_adv_v"] - a["after_vor_v"]),
        }
        for op, pair in references.items():
            for face, ref_full in zip(("u", "v"), pair, strict=True):
                value = np.asarray(components[f"{op}_{face}"].data)
                got = value[:, 1:, :] if face == "u" else value[1:, :, :]
                ref = _owned3(ref_full)
                mask = _owned3(a[f"{face}mask"]) > 0.5
                rows.append(
                    {
                        "name": f"GYRE-zco.kt{kt}.s{stage}.{op}.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                        "reference_max_abs": float(np.max(np.abs(ref[mask]))),
                        "model_max_abs": float(np.max(np.abs(got[mask]))),
                        "execution": "production-jit model component from NEMO stage inputs",
                    }
                )
        # LDF is exposed by the ordinary momentum diagnostics and exists only
        # in stages 1 and 3.  ZDF is an update/solve, scored in trajectory.
        if stage in (1, 3):
            prior = a["after_hpg_u"] if stage == 1 else a["after_adv_u"]
            prior_v = a["after_hpg_v"] if stage == 1 else a["after_adv_v"]
            for face, prior_full in (("u", prior), ("v", prior_v)):
                value = sum(
                    np.asarray(getattr(diagnostics, f"{name}_{face}").data)
                    for name in ("Ah_lap", "Bh_bilap", "Cs_smag", "Cl_leith")
                )
                got = value[:, 1:, :] if face == "u" else value[1:, :, :]
                ref = _owned3(a[f"after_ldf_{face}"] - prior_full)
                mask = _owned3(a[f"{face}mask"]) > 0.5
                rows.append(
                    {
                        "name": f"GYRE-zco.kt{kt}.s{stage}.ldf.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                        "reference_max_abs": float(np.max(np.abs(ref[mask]))),
                        "model_max_abs": float(np.max(np.abs(got[mask]))),
                        "execution": "production-jit model diagnostic from NEMO stage inputs",
                    }
                )
        # Full-accumulator score through the production JIT path. Stages 2/3
        # have the same source order as the shared model route. Stage 1 is the
        # compiled stp2d exception (HPG -> LDF -> VOR -> KEG -> ZAD), so rebuild
        # only its accumulator association from the already-computed production
        # arrays inside a second JIT with an explicit barrier at every routine
        # boundary. This is neither an oracle replay nor a tendency subtraction.
        if stage == 1:
            def stage1_accumulators(parts):
                hpg_u = parts["after_hpg_u"].data
                hpg_v = parts["after_hpg_v"].data
                ldf_u = jax.lax.optimization_barrier(hpg_u + parts["ldf_u"].data)
                ldf_v = jax.lax.optimization_barrier(hpg_v + parts["ldf_v"].data)
                vor_u = jax.lax.optimization_barrier(ldf_u + parts["vorticity_u"].data)
                vor_v = jax.lax.optimization_barrier(ldf_v + parts["vorticity_v"].data)
                keg_u = jax.lax.optimization_barrier(vor_u + parts["keg_u"].data)
                keg_v = jax.lax.optimization_barrier(vor_v + parts["keg_v"].data)
                adv_u = jax.lax.optimization_barrier(keg_u + parts["zad_u"].data)
                adv_v = jax.lax.optimization_barrier(keg_v + parts["zad_v"].data)
                return {"hpg_u": hpg_u, "hpg_v": hpg_v,
                        "ldf_u": ldf_u, "ldf_v": ldf_v,
                        "vor_u": vor_u, "vor_v": vor_v,
                        "adv_u": adv_u, "adv_v": adv_v}
            accumulators = jax.jit(stage1_accumulators)(components)
            boundaries = ("hpg", "ldf", "vor", "adv")
            execution = "production-jit components in compiled stage-1 accumulator order"
        else:
            accumulators = {
                "hpg_u": components["after_hpg_u"].data,
                "hpg_v": components["after_hpg_v"].data,
                "vor_u": components["after_vor_u"].data,
                "vor_v": components["after_vor_v"].data,
                "adv_u": components["after_adv_u"].data,
                "adv_v": components["after_adv_v"].data,
                "ldf_u": components["after_ldf_u"].data,
                "ldf_v": components["after_ldf_v"].data,
            }
            boundaries = ("hpg", "vor", "adv", "ldf") if stage == 3 else (
                "hpg", "vor", "adv")
            execution = "production-jit accumulator at model routine barrier"
        for op in boundaries:
            for face in ("u", "v"):
                value = np.asarray(accumulators[f"{op}_{face}"])
                got = value[:, 1:, :] if face == "u" else value[1:, :, :]
                ref = _owned3(a[f"after_{op}_{face}"])
                mask = _owned3(a[f"{face}mask"]) > 0.5
                if plant == "given" and (stage, op, face) == (1, "hpg", "u"):
                    got = got.copy()
                    got[tuple(np.argwhere(mask)[0])] += 1.0
                if (plant == "stage-rhs-ulp"
                        and (stage, op, face) == (1, "hpg", "u")):
                    got = got.copy()
                    index = tuple(np.argwhere(mask)[0])
                    got[index] = np.nextafter(got[index], np.float64(np.inf))
                rows.append({
                    "name": f"GYRE-zco.kt{kt}.s{stage}.post_{op}_accumulator.{face}",
                    "n": int(mask.sum()),
                    "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                    "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                    "execution": execution,
                })
        # Separate accumulator-level KEG and explicit ZAD scores retain the
        # exact NEMO ww injection; the combined advection component above uses
        # the model's own WZV route and therefore diagnoses that seam too.
        model = _model_terms(_split_view(a))
        for op in ("keg", "zad"):
            for face in ("u", "v"):
                ref = a[f"after_{op}_{face}"][2:-2, 2:-2, :30]
                got = np.asarray(model[f"after_{op}_{face}"])
                mask = a[f"{face}mask"][2:-2, 2:-2, :30] > 0.5
                if plant == "given" and (stage, op, face) == (1, "zad", "u"):
                    got = got.copy()
                    got[tuple(np.argwhere(mask)[0])] += 1.0
                rows.append(
                    {
                        "name": f"GYRE-zco.kt{kt}.s{stage}.{op}_accumulator.{face}",
                        "n": int(mask.sum()),
                        "n_unequal": int(np.count_nonzero(got[mask] != ref[mask])),
                        "max_abs": float(np.max(np.abs(got[mask] - ref[mask]))),
                    }
                )
    if plant == "given":
        require(any(r["max_abs"] > 0.5 for r in rows), "given-input plant did not land")
    if plant == "stage-rhs-ulp":
        planted = next(
            row for row in rows
            if row["name"] == "GYRE-zco.kt1.s1.post_hpg_accumulator.u")
        require(planted["n_unequal"] == 1,
                "one-ULP exact-accumulator plant did not flip exactly one cell")
    first = {}
    # Historical audit only: rounds 49/50 changed VOR/LDF after this round-48
    # prediction was registered.  Retain the discriminating measurement and
    # label the old prediction; never make a current gate fail on a retracted
    # owner expectation.
    expected = {1: "ldf", 2: "vor", 3: "vor"} if kt == 2 else {}
    for stage in stages:
        order = ("hpg", "ldf", "vor", "adv") if stage == 1 else (
            ("hpg", "vor", "adv", "ldf") if stage == 3 else ("hpg", "vor", "adv")
        )
        measured = None
        counts = None
        for op in order:
            pair = [
                row for row in rows
                if row["name"] in {
                    f"GYRE-zco.kt{kt}.s{stage}.post_{op}_accumulator.u",
                    f"GYRE-zco.kt{kt}.s{stage}.post_{op}_accumulator.v",
                }
            ]
            require(len(pair) == 2, f"missing model-path boundary kt{kt} stage {stage} {op}")
            by_face = {row["name"].rsplit(".", 1)[-1]: row for row in pair}
            if any(row["n_unequal"] for row in pair):
                measured = op
                counts = {face: by_face[face]["n_unequal"] for face in ("u", "v")}
                break
        first[f"stage{stage}"] = {
            "operator": measured,
            "n_unequal": counts,
            "preregistered_operator": expected.get(stage),
            "prediction": (
                "CONFIRMED" if measured == expected.get(stage) else
                "REFUTED" if stage in expected else
                "ROUND96_MEASUREMENT"
            ),
            "interpretation": (
                "POSTHOC_AFTER_ROUND49_ROUND50" if kt == 2 else
                "ROUND96_PREREGISTERED_COMPILED_ORDER"
            ),
        }
    return rows, first


def _trajectory(records: dict, plant: str | None) -> list[dict]:
    """Advance the exact kt=1 card state, then score kt=2 stage states."""
    # The widened record closes every named 3-D/TKE field, but not NEMO's
    # persistent dynspg_ts AB3/AM4 substep history.  Seeding that history from
    # an independent legoESM kt=1 step moved every stage globally even after
    # all recorded entry fields were injected bit-exactly.  Refuse to print
    # those contaminated numbers as a stage verdict; a future acquisition must
    # add the six ubb_e/ub_e/vbb_e/vb_e/sshbb_e/sshb_e endpoint arrays.
    if plant == "trajectory":
        require(False, "trajectory plant: missing-history refusal fired")
    return [{
        "name": "GYRE-zco.kt1_end_to_kt2_stage_trajectory",
        "status": "UNMEASURED",
        "reason": "NEMO kt=1-end cross-window barotropic history was not recorded",
        "required_fields": [
            "ubb_e", "ub_e", "vbb_e", "vb_e", "sshbb_e", "sshb_e"
        ],
    }]

    # Retained below as the preregistered experimental arm; unreachable until
    # the required record exists and the refusal above is replaced by a parser.
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "trajectory requires x64")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    a = records[(2, 1)]["arrays"]
    # Independently advancing kt=1 preserves every hidden prognostic and TKE
    # history slot.  Reconstructing only named fields from the initial state
    # silently seeded different viscosity and produced a false all-cell miss.
    st = card.recipe.initial_state
    freshwater1, surface1 = _surface_forcings(card, st, 1)
    st = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg
    ).step(st, dt=card.dt_s, freshwater=freshwater1, surface_forcing=surface1)
    # Replace every recorded kt=1 endpoint operand while retaining the
    # otherwise-unrecorded history slots from the independently advanced
    # legoESM step.  This is the consumed-field bridge: no NEMO routine is
    # called, and the entry rows below prove the fields used by kt=2 are exact.
    u0, v0 = _owned3(a["u_Kbb"]), _owned3(a["v_Kbb"])
    ub0, vb0 = _owned2(a["uu_b_Kbb"]), _owned2(a["vv_b_Kbb"])

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    st = st._replace(
        u=st.u.replace(data=jnp.asarray(np.concatenate([u0[:, -1:, :], u0], axis=1))),
        v=st.v.replace(data=jnp.asarray(np.concatenate([np.zeros_like(v0[:1]), v0], axis=0))),
        T=st.T.replace(data=jnp.asarray(_owned3(a["T_Kbb"]))),
        S=st.S.replace(data=jnp.asarray(_owned3(a["S_Kbb"]))),
        eta=st.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kbb"]))),
        uu_b=st.uu_b.replace(data=jnp.asarray(np.concatenate([ub0[:, -1:], ub0], axis=1))),
        vv_b=st.vv_b.replace(data=jnp.asarray(np.concatenate([np.zeros_like(vb0[:1]), vb0], axis=0))),
        tke=field(a["tke_en"][..., 1:30], "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(a["tke_avm_k"], 31)[..., 1:30], "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(a["tke_avt_k"][..., 1:30], "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(a["tke_dissl"][..., 1:30], "tke_dissl", ("lat", "lon", "level")),
        tke_avm_surface=field(_owned3(a["tke_avm_k"], 31)[..., 0], "tke_avm_surface", ("lat", "lon")),
    )
    freshwater, surface = _surface_forcings(card, st, 2)
    masks = expected_masks(card)
    rows = []
    entry = lego_fields(st)
    for field, ref in (
        ("u", _owned3(a["u_Kbb"])),
        ("v", _owned3(a["v_Kbb"])),
        ("T", _owned3(a["T_Kbb"])),
        ("S", _owned3(a["S_Kbb"])),
        ("ssh", _owned2(a["ssh_Kbb"])),
    ):
        mask = masks[field]
        rows.append({
            "name": f"GYRE-zco.kt1_end_bridge.{field}",
            "n": int(mask.sum()),
            "n_unequal": int(np.count_nonzero(entry[field][mask] != ref[mask])),
            "max_abs": float(np.max(np.abs(entry[field][mask] - ref[mask]))),
        })
    for stage in (1, 2, 3):
        hooks = _NEMOWSRK3TestHooks(expose_momentum_stage=stage)
        got = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg, _nemo_ws_test_hooks=hooks
        ).step(st, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
        fields = lego_fields(got)
        refa = records[(2, stage)]["arrays"]
        for face in ("u", "v"):
            ref = _owned3(refa[f"post_baro_{face}"])
            candidate = fields[face]
            mask = masks[face]
            if plant == "trajectory" and (stage, face) == (1, "u"):
                candidate = candidate.copy()
                candidate[tuple(np.argwhere(mask)[0])] += 1.0
            rows.append(
                {
                    "name": f"GYRE-zco.kt2.s{stage}.post_baro.{face}",
                    "n": int(mask.sum()),
                    "n_unequal": int(np.count_nonzero(candidate[mask] != ref[mask])),
                    "max_abs": float(np.max(np.abs(candidate[mask] - ref[mask]))),
                }
            )
    if plant == "trajectory":
        require(any(r["max_abs"] > 0.5 for r in rows), "trajectory plant did not land")
    return rows


def _u_full(value):
    value = np.asarray(value)
    return np.concatenate([value[:, -1:, ...], value], axis=1)


def _v_full(value):
    value = np.asarray(value)
    return np.concatenate([np.zeros_like(value[:1]), value], axis=0)


def _classification(row: dict) -> dict:
    row["classification"] = "BIT" if row["exact"] else row["status"]
    return row


def _bitwise_classification(row: dict) -> dict:
    """Require identical IEEE-754 words, including the sign of zero."""
    row["exact"] = row["n_unequal"] == 0
    row["classification"] = "BIT" if row["exact"] else row["status"]
    return row


def _exact_identity(name: str, candidate, reference, mask) -> dict:
    """Bit census for one installed production-entry operand."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(candidate.shape == reference.shape == mask.shape,
            f"{name}: entry identity shapes disagree: {candidate.shape}, "
            f"{reference.shape}, {mask.shape}")
    require(bool(np.any(mask)), f"{name}: entry identity mask is empty")
    require(np.all(np.isfinite(candidate[mask]))
            and np.all(np.isfinite(reference[mask])),
            f"{name}: entry identity contains NaN/Inf")
    unequal = int(np.count_nonzero(
        candidate[mask].view(np.uint64) != reference[mask].view(np.uint64)))
    return {
        "name": name,
        "compared_cells": int(np.count_nonzero(mask)),
        "unequal": unequal,
        "max_abs": float(np.max(np.abs(candidate[mask] - reference[mask]))),
        "exact": unequal == 0,
    }


def _unmeasured(kt: int, stage: int | str, field: str, reason: str) -> dict:
    return {
        "name": f"GYRE-zco.kt{kt}.s{stage}.{field}",
        "kt": kt,
        "stage": stage,
        "field": field,
        "classification": "UNMEASURED_WITH_SPEC",
        "reason": reason,
    }


def _raw_history_override(memory_root: Path):
    """Read the admitted kt=1 endpoint histories without importing round 51."""
    from nemo_testcase_l2_gyre_round48_bt_memory_gate import read_record

    arrays = read_record(memory_root / "oracle_bt_memory_kt00000001_end.bin")["arrays"]
    return (
        _u_full(_owned2(arrays["ub_e"])),
        _u_full(_owned2(arrays["ubb_e"])),
        _v_full(_owned2(arrays["vb_e"])),
        _v_full(_owned2(arrays["vbb_e"])),
        _owned2(arrays["sshb_e"]),
        _owned2(arrays["sshbb_e"]),
    )


def _bridge_kt2_state(card, cfg, records):
    """Build the kt=2 Kbb state, retaining non-recorded pytrees from kt=1."""
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    state = card.recipe.initial_state
    freshwater, surface = _surface_forcings(card, state, 1)
    state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg
    ).step(state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
    a = records[(2, 1)]["arrays"]

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    state = state._replace(
        u=state.u.replace(data=jnp.asarray(_u_full(_owned3(a["u_Kbb"])))),
        v=state.v.replace(data=jnp.asarray(_v_full(_owned3(a["v_Kbb"])))),
        T=state.T.replace(data=jnp.asarray(_owned3(a["T_Kbb"]))),
        S=state.S.replace(data=jnp.asarray(_owned3(a["S_Kbb"]))),
        eta=state.eta.replace(data=jnp.asarray(_owned2(a["ssh_Kbb"]))),
        uu_b=state.uu_b.replace(data=jnp.asarray(_u_full(_owned2(a["uu_b_Kbb"])) )),
        vv_b=state.vv_b.replace(data=jnp.asarray(_v_full(_owned2(a["vv_b_Kbb"])) )),
        tke=field(a["tke_en"][..., 1:30], "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(a["tke_avm_k"], 31)[..., 1:30],
                      "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(a["tke_avt_k"][..., 1:30],
                      "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(a["tke_dissl"][..., 1:30],
                        "tke_dissl", ("lat", "lon", "level")),
        tke_avm_surface=field(_owned3(a["tke_avm_k"], 31)[..., 0],
                              "tke_avm_surface", ("lat", "lon")),
    )
    return state


def _bridge_stage_context(state, arrays, *, plant: bool = False):
    """Install the pre-stage closure bundle recorded after ``zdf_phy``."""
    import jax.numpy as jnp
    from legoesm.core.field import Field

    tke = np.array(arrays["tke_en"][..., 1:30], copy=True)
    if plant:
        mask = _owned3(arrays["wmask"], 31)[..., 1:30] > 0.5
        index = tuple(np.argwhere(mask & (tke != 0.0))[0])
        tke[index] = np.nextafter(tke[index], np.float64(np.inf))

    def field(data, name, dims):
        return Field(data=jnp.asarray(data), name=name, dims=dims)

    return state._replace(
        tke=field(tke, "tke", ("lat", "lon", "level")),
        tke_avm=field(_owned3(arrays["tke_avm_k"], 31)[..., 1:30],
                      "tke_avm", ("lat", "lon", "level")),
        tke_avt=field(arrays["tke_avt_k"][..., 1:30],
                      "tke_avt", ("lat", "lon", "level")),
        tke_dissl=field(arrays["tke_dissl"][..., 1:30],
                        "tke_dissl", ("lat", "lon", "level")),
        tke_avm_surface=field(_owned3(arrays["tke_avm_k"], 31)[..., 0],
                              "tke_avm_surface", ("lat", "lon")),
    )


def _bridge_kt2_production_entry(
    card, cfg, records, stage_root: Path, memory_root: Path,
    year_entry_root: Path,
):
    """Install and prove the complete NEMO kt=2 production-step entry.

    This deliberately starts with the same ``_bridge_kt2_state`` used by the
    established round-93+ stage twins.  It then replaces that lane's
    post-``zdf_phy`` closure context with the raw closure carries consumed at
    kt=2 and installs the six recorded AB3/AM4 histories as real state, rather
    than combining kt=1 state/forcing with a kt=2 closure record.
    """
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    kt = 2
    stage_entry_path = stage_root / "oracle_step_entry_kt00000002.bin"
    year_entry_path = year_entry_root / "oracle_step_entry_kt00000002.bin"
    require(stage_entry_path.is_file() and year_entry_path.is_file(),
            "kt=2 production bridge is missing a step-entry dump")
    stage_sha = sha256(stage_entry_path)
    year_sha = sha256(year_entry_path)
    require(stage_sha == year_sha,
            "round-46 and year-owner kt=2 step entries are not bit-identical")
    step_entry = read_entry(year_entry_path)
    require(
        (step_entry["kt"], step_entry["Nbb"]) == (kt, 3),
        f"{year_entry_path}: production-entry slot mismatch: expected "
        f"kt=2/Nbb=3, got kt={step_entry['kt']}/Nbb={step_entry['Nbb']}",
    )

    # This is the exact bridge used by the established stage twins.  Its
    # kt=2 Kbb physical state and prognostic barotropic pair are recorded
    # values; the one preliminary card step only supplies static/inactive
    # pytree structure.
    state = _bridge_kt2_state(card, cfg, records)
    # Round 59 records kt=2 closure ENTRY memory.  NEMO produced that memory at
    # kt=1, so it is the kt=1 stage record bit for bit; records[(2, 1)] instead
    # holds the closure OUTPUT produced during kt=2.
    state = _bridge_stage_context(state, records[(1, 1)]["arrays"])
    raw_history = tuple(
        jnp.asarray(value) for value in _raw_history_override(memory_root))
    state = state._replace(bt_hist=raw_history)

    present = {name for name, value in zip(state._fields, state)
               if value is not None}
    active = {
        "u", "v", "T", "S", "eta", "uu_b", "vv_b", "bt_hist",
        "tke", "tke_avm", "tke_avt", "tke_dissl", "tke_avm_surface",
    }
    static = {"H_bathy", "land_mask", "u_mask", "v_mask"}
    diagnostic = {"w"}
    require(present == active | static | diagnostic,
            "kt=2 bridge has an unaccounted active/inactive state carry: "
            f"present={sorted(present)}, expected={sorted(active | static | diagnostic)}")
    require(cfg.outer_integrator == "forward_euler"
            and cfg.tracer_time_integrator == "rk3_ws"
            and not cfg.barotropic_forcing_centred,
            "kt=2 active-carry inventory no longer matches the resolved card")
    tke_cfg = cfg.physics.vertical_mixing.tke
    require(tke_cfg.advection_scheme == "none"
            and not tke_cfg.source_eke_diss,
            "kt=2 bridge needs an additional TKE/EKE carry")

    masks = expected_masks(card)
    fields = lego_fields(state)
    physical = {}
    for name in ("T", "S", "u", "v", "ssh"):
        reference = np.asarray(step_entry[name])
        if reference.ndim == 3:
            reference = reference[..., :fields[name].shape[-1]]
        physical[name] = _exact_identity(
            f"kt2.entry.{name}", fields[name], reference, masks[name])

    stage_record = records[(2, 1)]
    stage_header = stage_record["header"]
    require(
        (stage_header["kt"], stage_header["stage"],
         stage_header["Kbb"], stage_header["Kmm"]) == (2, 1, 3, 3),
        "kt=2 production-stage slot mismatch after record admission",
    )
    a = stage_record["arrays"]
    barotropic = {
        "uu_b": _exact_identity(
            "kt2.entry.uu_b", np.asarray(state.uu_b.data)[:, 1:],
            _owned2(a["uu_b_Kbb"]), masks["u"][..., 0]),
        "vv_b": _exact_identity(
            "kt2.entry.vv_b", np.asarray(state.vv_b.data)[1:, :],
            _owned2(a["vv_b_Kbb"]), masks["v"][..., 0]),
    }
    history_reference = _raw_history_override(memory_root)
    history_names = ("ub_e", "ubb_e", "vb_e", "vbb_e", "sshb_e", "sshbb_e")
    histories = {
        name: _exact_identity(
            f"kt2.entry.{name}", candidate, reference,
            np.ones(np.shape(reference), dtype=bool))
        for name, candidate, reference in zip(
            history_names, state.bt_hist, history_reference, strict=True)
    }

    closure_reference = records[(1, 1)]["arrays"]
    wet_w = _owned3(closure_reference["wmask"], 31)[..., 1:30] > 0.5
    closure = {
        "tke": _exact_identity(
            "kt2.entry.tke", state.tke.data,
            closure_reference["tke_en"][..., 1:30], wet_w),
        "tke_avm": _exact_identity(
            "kt2.entry.tke_avm", state.tke_avm.data,
            _owned3(closure_reference["tke_avm_k"], 31)[..., 1:30], wet_w),
        "tke_avt": _exact_identity(
            "kt2.entry.tke_avt", state.tke_avt.data,
            closure_reference["tke_avt_k"][..., 1:30], wet_w),
        "tke_dissl": _exact_identity(
            "kt2.entry.tke_dissl", state.tke_dissl.data,
            closure_reference["tke_dissl"][..., 1:30], wet_w),
        "tke_avm_surface": _exact_identity(
            "kt2.entry.tke_avm_surface", state.tke_avm_surface.data,
            _owned3(closure_reference["tke_avm_k"], 31)[..., 0],
            masks["ssh"]),
    }

    # Thickness is derived state, not a NamedTuple carry.  Rebuild it through
    # the exact production helpers and score all four consumed staggerings.
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    bundle = model._tke_step_entry_n2_bundle(state)
    require(bundle is not None, "kt=2 entry did not materialize TKE thickness")
    u_mask, v_mask = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord, min_water_column_m=cfg.min_water_column_m)
    h_u, h_v, _, _ = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, u_mask.astype(state.eta.data.dtype),
        v_mask.astype(state.eta.data.dtype), card.recipe.grid)
    thickness = {
        "e3t_Kbb": _exact_identity(
            "kt2.entry.e3t_Kbb", bundle.e3t_Kmm,
            _owned3(a["e3t_Kbb"])[..., :30],
            _owned3(a["tmask"])[..., :30] > 0.5),
        "e3u_Kbb": _exact_identity(
            "kt2.entry.e3u_Kbb", np.asarray(h_u)[:, 1:, :],
            _owned3(a["e3u_Kbb"])[..., :30],
            _owned3(a["umask"])[..., :30] > 0.5),
        "e3v_Kbb": _exact_identity(
            "kt2.entry.e3v_Kbb", np.asarray(h_v)[1:, :, :],
            _owned3(a["e3v_Kbb"])[..., :30],
            _owned3(a["vmask"])[..., :30] > 0.5),
        "e3w_Kbb": _exact_identity(
            "kt2.entry.e3w_Kbb", bundle.e3w_Kmm,
            _owned3(a["e3w_Kbb"])[..., 1:30],
            _owned3(a["wmask"])[..., 1:30] > 0.5),
    }
    groups = {
        "physical": physical,
        "thickness": thickness,
        "barotropic": barotropic,
        "barotropic_history": histories,
        "tke_carry": closure,
    }
    rows = [row for group in groups.values() for row in group.values()]
    require(all(row["exact"] for row in rows),
            "kt=2 production entry is not bit-identical to its NEMO records: "
            + ", ".join(row["name"] for row in rows if not row["exact"]))
    audit = {
        "kt": kt,
        "stage": "before zdf_phy; Kbb=Kmm=3; before WS-RK3 stage 1",
        "slots": {
            "Nbb": step_entry["Nbb"],
            "Kbb": stage_header["Kbb"],
            "Kmm": stage_header["Kmm"],
        },
        "round46_entry": str(stage_entry_path),
        "year_owner_entry": str(year_entry_path),
        "entry_sha256": stage_sha,
        "round46_year_owner_bit_identical": True,
        "groups": groups,
        "all_exact": True,
        "active_state_carries": sorted(active),
        "inactive_or_diagnostic_state": sorted(static | diagnostic),
    }
    return state, audit


def _stage_reference(records, next_entries, kt: int, stage: int) -> dict:
    a = records[(kt, stage)]["arrays"]
    if stage < 3:
        nxt = records[(kt, stage + 1)]["arrays"]
        return {
            "u": _owned3(a["post_baro_u"]),
            "v": _owned3(a["post_baro_v"]),
            "T": _owned3(nxt["T_Kmm"]),
            "S": _owned3(nxt["S_Kmm"]),
            "ssh": _owned2(nxt["ssh_Kmm"]),
        }
    entry = next_entries[kt + 1]
    return {
        "u": _owned3(a["post_baro_u"]),
        "v": _owned3(a["post_baro_v"]),
        "T": entry["T"][..., :30],
        "S": entry["S"][..., :30],
        "ssh": entry["ssh"],
    }


def _entry_override(records, kt: int, stage: int, *, plant=False):
    import jax.numpy as jnp

    a = records[(kt, stage)]["arrays"]
    T = _owned3(a["T_Kmm"]).copy()
    if plant:
        index = tuple(np.argwhere(_owned3(a["tmask"]) > 0.5)[0])
        T[index] = np.nextafter(T[index], np.float64(np.inf))
    return (
        stage,
        jnp.asarray(_u_full(_owned3(a["u_Kmm"]))),
        jnp.asarray(_v_full(_owned3(a["v_Kmm"]))),
        jnp.asarray(T),
        jnp.asarray(_owned3(a["S_Kmm"])),
        jnp.asarray(_owned2(a["ssh_Kmm"])),
    )


def _barotropic_override(records, advmean_root: Path, kt: int):
    import jax.numpy as jnp
    from nemo_testcase_l2_gyre_round14_advmean import read_advmean
    from nemo_testcase_l2_gyre_phase3_gate import read_bt

    avg = read_advmean(
        advmean_root / f"oracle_bt_advmean_operands_kt{kt:08d}.bin",
        expected_kt=kt,
    )
    bt = read_bt(
        advmean_root / f"oracle_bt_frames_kt{kt:08d}.bin", kt)
    next_entry = read_entry(
        advmean_root / f"oracle_step_entry_kt{kt + 1:08d}.bin")
    return (
        jnp.asarray(next_entry["ssh"]),
        jnp.asarray(_u_full(bt["uu_b"])),
        jnp.asarray(_v_full(bt["vv_b"])),
        jnp.asarray(_u_full(avg["post_lbc_u"])),
        jnp.asarray(_v_full(avg["post_lbc_v"])),
    )


def _final_history_from_btstep(arrays: dict) -> tuple[np.ndarray, ...]:
    """Apply NEMO's final ``bb<-b, b<-n`` history rotation to a record."""
    last = -1
    return (
        arrays["u_entry"][last], arrays["u_b"][last],
        arrays["v_entry"][last], arrays["v_b"][last],
        arrays["eta_entry"][last], arrays["eta_b"][last],
    )


def _history_reference(memory_root: Path, btstep_root: Path, kt: int):
    """Return the six absolute histories at the completed external boundary."""
    if kt == 1:
        from nemo_testcase_l2_gyre_round48_bt_memory_gate import read_record

        arrays = read_record(
            memory_root / "oracle_bt_memory_kt00000001_end.bin")["arrays"]
        return tuple(arrays[name] for name in (
            "ub_e", "ubb_e", "vb_e", "vbb_e", "sshb_e", "sshbb_e"))
    from nemo_testcase_l2_gyre_round81_btstep_gate import read_record

    arrays = read_record(
        btstep_root / "oracle_bt_step_operands_kt00000002.bin", expected_kt=2)
    # dynspg_ts rotates bb<-b, b<-n, n<-a at the end of each substep.
    return _final_history_from_btstep(arrays)


def _closure_rows(context, tke_entry, arrays, masks, kt: int, stage: int,
                  mode: str, boundary: str) -> list[dict]:
    """Score the closure fields computed once and consumed by every stage."""
    candidate_fields = {
        "tke_en": tke_entry,
        "tke_avm_k": context.tke_avm,
        "tke_avt_k": context.tke_avt,
        "tke_dissl": context.tke_dissl,
        "tke_avm_surface": context.tke_avm_surface,
    }
    references = {
        "tke_en": np.asarray(arrays["tke_en"])[..., 1:30],
        "tke_avm_k": _owned3(arrays["tke_avm_k"], 31)[..., 1:30],
        "tke_avt_k": np.asarray(arrays["tke_avt_k"])[..., 1:30],
        "tke_dissl": np.asarray(arrays["tke_dissl"])[..., 1:30],
        "tke_avm_surface": _owned3(arrays["tke_avm_k"], 31)[..., 0],
    }
    wet_w = _owned3(arrays["wmask"], 31)[..., 1:30] > 0.5
    field_masks = {name: wet_w for name in candidate_fields}
    field_masks["tke_avm_surface"] = masks["ssh"]
    rows = []
    for field, candidate_field in candidate_fields.items():
        if candidate_field is None:
            row = _unmeasured(
                kt, stage, field,
                "model stage-entry context has no explicit carried field")
            row.update({"entry_mode": mode, "boundary": boundary})
            rows.append(row)
            continue
        candidate = np.asarray(
            candidate_field if field == "tke_en" else candidate_field.data)
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.{boundary}.{field}",
            references[field], candidate, field_masks[field]))
        row.update({"kt": kt, "stage": stage, "field": field,
                    "entry_mode": mode, "boundary": boundary})
        rows.append(row)
    return rows


def _output_rows(trace, records, next_entries, transports, masks, area_t,
                 context, kt: int, mode: str,
                 direct_stage_ww: dict | None = None) -> list[dict]:
    rows = []
    for stage, output in enumerate(trace.stage_outputs, start=1):
        u, v, T, S, eta = (np.asarray(value) for value in output)
        candidates = {"u": u[:, 1:, :], "v": v[1:, :, :],
                      "T": T, "S": S, "ssh": eta}
        refs = _stage_reference(records, next_entries, kt, stage)
        for field in ("T", "S", "u", "v", "ssh"):
            row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.output.{field}",
                refs[field], candidates[field], masks[field]))
            row.update({"kt": kt, "stage": stage, "field": field,
                        "entry_mode": mode})
            rows.append(row)

        a = records[(kt, stage)]["arrays"]
        geom = trace.stage_geometry[stage - 1]
        ww = np.asarray(geom[2])
        ww_nlev = ww.shape[-1]
        ww_reference = _owned3(a["ww"], ww_nlev)
        ww_boundary = "pre_external_velocity_form"
        if direct_stage_ww is not None and (kt, stage) in direct_stage_ww:
            ww_reference = np.asarray(
                direct_stage_ww[(kt, stage)]["ww"])[..., :ww_nlev]
            ww_boundary = "post_tra_adv_trp_transport_form"
        geom_rows = (
            ("e3t_Kmm", _owned3(a["e3t_Kmm"]), np.asarray(geom[3]),
             _owned3(a["tmask"]) > 0.5),
            ("e3u_Kmm", _owned3(a["e3u_Kmm"]), np.asarray(geom[4])[:, 1:, :],
             _owned3(a["umask"]) > 0.5),
            ("e3v_Kmm", _owned3(a["e3v_Kmm"]), np.asarray(geom[5])[1:, :, :],
             _owned3(a["vmask"]) > 0.5),
            ("ww", ww_reference, ww,
             _owned3(a["wmask"], ww_nlev) > 0.5),
        )
        for field, ref, candidate, mask in geom_rows:
            row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.handoff.{field}",
                ref, candidate, mask))
            row.update({"kt": kt, "stage": stage, "field": field,
                        "entry_mode": mode})
            if field == "ww":
                row["reference_boundary"] = ww_boundary
            rows.append(row)

        rows.extend(_closure_rows(
            context, trace.tke_entry, a, masks, kt, stage, mode, "output"))

        transport = transports.get((kt, stage))
        if transport is None:
            for field in ("zFu", "zFv", "zFw"):
                rows.append(_unmeasured(
                    kt, stage, field,
                    "no admitted direct transport payload for this kt/stage"))
            continue
        for field, candidate, mask in (
            ("zFu", np.asarray(geom[7])[:, 1:, :], _owned3(a["umask"]) > 0.5),
            ("zFv", np.asarray(geom[8])[1:, :, :], _owned3(a["vmask"]) > 0.5),
            ("zFw", np.asarray(geom[2]) * area_t[..., None],
             _owned3(a["wmask"], np.asarray(geom[2]).shape[-1]) > 0.5),
        ):
            nlev = candidate.shape[-1]
            row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.handoff.{field}",
                np.asarray(transport[field])[..., :nlev], candidate,
                mask[..., :nlev]))
            row.update({"kt": kt, "stage": stage, "field": field,
                        "entry_mode": mode})
            rows.append(row)
    return rows


def _entry_rows(trace, records, masks, context, kt: int, stage: int,
                mode: str, *, plant_rhs: bool = False) -> list[dict]:
    """Prove the exact Kmm state that actually entered one compiled stage."""
    a = records[(kt, stage)]["arrays"]
    u, v, temperature, salinity, eta = (
        np.asarray(value) for value in trace.stage_states[stage - 1])
    candidates = {
        "u": u[:, 1:, :], "v": v[1:, :, :],
        "T": temperature, "S": salinity, "ssh": eta,
    }
    references = {
        "u": _owned3(a["u_Kmm"]), "v": _owned3(a["v_Kmm"]),
        "T": _owned3(a["T_Kmm"]), "S": _owned3(a["S_Kmm"]),
        "ssh": _owned2(a["ssh_Kmm"]),
    }
    rows = []
    for field in ("T", "S", "u", "v", "ssh"):
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.entry.{field}",
            references[field], candidates[field], masks[field]))
        row.update({"kt": kt, "stage": stage, "field": field,
                    "entry_mode": mode})
        rows.append(row)
    rhs_boundary = "pre_zdf_rhs" if stage == 3 else "after_adv"
    rhs_u, rhs_v = (np.asarray(value)
                    for value in trace.stage_rhs[stage - 1])
    rhs_candidates = {"u": rhs_u[:, 1:, :], "v": rhs_v[1:, :, :]}
    for face in ("u", "v"):
        reference = _owned3(a[f"{rhs_boundary}_{face}"]).copy()
        candidate = rhs_candidates[face]
        mask = masks[face]
        planted_at = None
        if plant_rhs and face == "u":
            equal = (
                np.ascontiguousarray(candidate).view(np.uint64)
                == np.ascontiguousarray(reference).view(np.uint64)
            ) & mask
            indices = np.argwhere(equal & np.isfinite(reference))
            require(indices.size > 0,
                    "stage-RHS plant found no exact finite wet cell")
            planted_at = tuple(int(value) for value in indices[0])
            reference[planted_at] = np.nextafter(
                reference[planted_at], np.float64(np.inf))
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.entry.momentum_rhs_{face}",
            reference, candidate, mask))
        row.update({
            "kt": kt,
            "stage": stage,
            "field": f"momentum_rhs_{face}",
            "entry_mode": mode,
            "boundary": "entry",
            "nemo_boundary": rhs_boundary,
            "plant_index": planted_at,
        })
        rows.append(row)
    if kt == 1 and stage == 1:
        rhs_walk_names = (
            ("operator_accumulator", "compiled stp2d HPG/LDF/VOR/KEG/ZAD"),
            ("transport_reconcile", "model-only stage transport reassociation"),
            ("zad_reassociation", "model-only stage ZAD reassociation"),
            ("final", "RK assignment input"),
        )
        reference_by_face = {
            face: _owned3(a[f"after_adv_{face}"])
            for face in ("u", "v")
        }
        for boundary_index, (boundary, statement) in enumerate(rhs_walk_names):
            for face, candidate_full in zip(
                    ("u", "v"), trace.stage1_rhs_walk[boundary_index],
                    strict=True):
                candidate_full = np.asarray(candidate_full)
                candidate = (candidate_full[:, 1:, :] if face == "u"
                             else candidate_full[1:, :, :])
                row = _classification(score(
                    f"GYRE-zco.kt1.s1.rhs_walk.{boundary}.{face}",
                    reference_by_face[face], candidate, masks[face]))
                row.update({
                    "kt": 1,
                    "stage": 1,
                    "field": f"stage1_rhs_{boundary}_{face}",
                    "entry_mode": mode,
                    "boundary": boundary,
                    "statement": statement,
                    "reference_boundary": "after_adv",
                })
                rows.append(row)
    qco = tuple(np.asarray(value) for value in trace.stage_qco[stage - 1])
    qco_candidates = {
        "r3t_Kmm": qco[0],
        "r3u_Kmm": qco[1][:, 1:, 0],
        "r3v_Kmm": qco[2][1:, :, 0],
    }
    for field, candidate in qco_candidates.items():
        face = field[2]
        mask = masks["ssh" if face == "t" else face][..., 0] if face != "t" else masks["ssh"]
        row = _classification(score(
            f"GYRE-zco.kt{kt}.s{stage}.entry.{field}",
            _owned2(a[field]), candidate, mask))
        row.update({"kt": kt, "stage": stage, "field": field,
                    "entry_mode": mode, "boundary": "entry"})
        rows.append(row)
    rows.extend(_closure_rows(
        context, trace.tke_entry, a, masks, kt, stage, mode, "entry"))
    return rows


_STAGE1_W_ORDER = (
    "transport_u", "transport_v", "zonal_difference",
    "meridional_difference", "numerator", "scaled", "hdiv", "e3div",
    "r3_delta", "stretch", "bracket", "incoming_carry",
    "outgoing_carry", "ww",
)


def _stage1_w_recurrence_trace(
    e3div, e3t_0, r3_kbb, r3_kaa, r1_dt, tmask, *, source_round: bool,
    plant_carry_at: tuple[int, int, int] | None = None,
):
    """Trace the compiled QCO stretch and bottom-up W recurrence."""
    import jax
    import jax.numpy as jnp

    materialize = (
        (lambda value: jax.lax.reduce_precision(value, 11, 52))
        if source_round else jax.lax.optimization_barrier)
    e3div = jnp.asarray(e3div)
    e3t_0 = jnp.asarray(e3t_0)
    r3_kbb = jnp.asarray(r3_kbb)
    r3_kaa = jnp.asarray(r3_kaa)
    tmask = jnp.asarray(tmask)
    r3_delta = materialize(r3_kaa - r3_kbb)
    stretch = materialize(
        materialize(r1_dt * e3t_0) * r3_delta[..., None])
    incoming = []
    outgoing = [None] * e3div.shape[-1]
    brackets = [None] * e3div.shape[-1]
    carry = jnp.zeros_like(r3_kbb)
    for level in range(e3div.shape[-1] - 1, -1, -1):
        if plant_carry_at is not None and level == plant_carry_at[2]:
            j, i, _ = plant_carry_at
            carry = carry.at[j, i].set(jnp.nextafter(
                carry[j, i], jnp.asarray(jnp.inf, dtype=carry.dtype)))
        incoming.append(carry)
        bracket = materialize(e3div[..., level] + stretch[..., level])
        carry = materialize(
            carry - materialize(bracket * tmask[..., level]))
        brackets[level] = bracket
        outgoing[level] = carry
    incoming = list(reversed(incoming))
    return {
        "r3_delta": r3_delta,
        "stretch": stretch,
        "bracket": jnp.stack(brackets, axis=-1),
        "incoming_carry": jnp.stack(incoming, axis=-1),
        "outgoing_carry": jnp.stack(outgoing, axis=-1),
        "ww": jnp.stack(outgoing + [jnp.zeros_like(carry)], axis=-1),
    }


def _stage1_transport_w_trace(
    transport_u, transport_v, reciprocal_area, thickness_t, e3t_0,
    r3_kbb, r3_kaa, r1_dt, tmask, *, source_round: bool,
):
    """Trace the existing shared transport-form W arithmetic."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round

    materialize = (
        (lambda value: jax.lax.reduce_precision(value, 11, 52))
        if source_round else jax.lax.optimization_barrier)
    transport_u = jnp.asarray(transport_u)[..., :30]
    transport_v = jnp.asarray(transport_v)[..., :30]
    tmask = jnp.asarray(tmask)[..., :30]
    west = jnp.roll(transport_u, 1, axis=1)
    south = jnp.concatenate(
        [jnp.zeros_like(transport_v[:1]), transport_v[:-1]], axis=0)
    zonal = materialize(transport_u - west)
    meridional = materialize(transport_v - south)
    numerator = materialize(zonal + meridional)
    scaled = materialize(numerator * reciprocal_area[..., None]) * tmask
    safe_thickness = jnp.where(tmask > 0.5, thickness_t[..., :30], 1.0)
    hdiv = materialize(scaled / safe_thickness) * tmask
    e3div = materialize(hdiv * thickness_t[..., :30]) * tmask
    recurrence = _stage1_w_recurrence_trace(
        e3div, e3t_0[..., :30], r3_kbb, r3_kaa, r1_dt, tmask,
        source_round=source_round)
    return {
        "transport_u": transport_u,
        "transport_v": transport_v,
        "zonal_difference": zonal,
        "meridional_difference": meridional,
        "numerator": numerator,
        "scaled": scaled,
        "hdiv": hdiv,
        "e3div": e3div,
    } | recurrence


def _stage1_w_scalar_recurrence(
    e3div, e3t_0, r3_kbb, r3_kaa, r1_dt, tmask,
):
    """Scalar binary64 replay of the compiled QCO recurrence statements."""
    e3div = np.asarray(e3div, dtype=np.float64)
    e3t_0 = np.asarray(e3t_0, dtype=np.float64)
    r3_kbb = np.asarray(r3_kbb, dtype=np.float64)
    r3_kaa = np.asarray(r3_kaa, dtype=np.float64)
    tmask = np.asarray(tmask, dtype=np.float64)
    shape = e3div.shape
    r3_delta = np.zeros(shape[:2], dtype=np.float64)
    stretch = np.zeros(shape, dtype=np.float64)
    bracket = np.zeros(shape, dtype=np.float64)
    incoming = np.zeros(shape, dtype=np.float64)
    outgoing = np.zeros(shape, dtype=np.float64)
    carry = np.zeros(shape[:2], dtype=np.float64)
    for j in range(shape[0]):
        for i in range(shape[1]):
            r3_delta[j, i] = np.float64(r3_kaa[j, i] - r3_kbb[j, i])
    for level in range(shape[-1] - 1, -1, -1):
        for j in range(shape[0]):
            for i in range(shape[1]):
                incoming[j, i, level] = carry[j, i]
                first = np.float64(np.float64(r1_dt) * e3t_0[j, i, level])
                stretch[j, i, level] = np.float64(
                    first * r3_delta[j, i])
                bracket[j, i, level] = np.float64(
                    e3div[j, i, level] + stretch[j, i, level])
                masked = np.float64(
                    bracket[j, i, level] * tmask[j, i, level])
                carry[j, i] = np.float64(carry[j, i] - masked)
                outgoing[j, i, level] = carry[j, i]
    return {
        "r3_delta": r3_delta,
        "stretch": stretch,
        "bracket": bracket,
        "incoming_carry": incoming,
        "outgoing_carry": outgoing,
        "ww": np.concatenate(
            [outgoing, np.zeros((*shape[:2], 1), dtype=np.float64)], axis=-1),
    }


def _stage1_transport_w_scalar_reference(
    arrays, transport_u, transport_v, *, r3_kaa=None, r1_dt=None,
):
    """Scalar replay of compiled divhor + WZV, calibrated to direct W."""
    shape = (22, 32, 30)
    tmask = _owned3(arrays["tmask"]) > 0.5
    thickness = _owned3(arrays["e3t_Kmm"])
    e3t_0 = _owned3(arrays["e3t_0"])
    reciprocal_area = _owned2(arrays["r1_e1e2t"])
    r3_kbb = _owned2(arrays["r3t_Kbb"])
    r3_kaa = (_owned2(arrays["r3t_Kaa"])
              if r3_kaa is None else np.asarray(r3_kaa))
    r1_dt = np.float64(arrays["r1_Dt"] if r1_dt is None else r1_dt)
    hdiv = np.zeros(shape, dtype=np.float64)
    e3div = np.zeros(shape, dtype=np.float64)
    zonal = np.zeros(shape, dtype=np.float64)
    meridional = np.zeros(shape, dtype=np.float64)
    numerator = np.zeros(shape, dtype=np.float64)
    scaled = np.zeros(shape, dtype=np.float64)
    stretch = np.zeros(shape, dtype=np.float64)
    bracket = np.zeros(shape, dtype=np.float64)
    incoming = np.zeros(shape, dtype=np.float64)
    outgoing = np.zeros(shape, dtype=np.float64)
    for level in range(30):
        for j in range(22):
            for i in range(32):
                west = transport_u[j, i - 1, level]
                south = (np.float64(0.0) if j == 0
                         else transport_v[j - 1, i, level])
                zu = np.float64(transport_u[j, i, level] - west)
                zv = np.float64(transport_v[j, i, level] - south)
                total = np.float64(zu + zv)
                area_scaled = np.float64(total * reciprocal_area[j, i])
                value = np.float64(area_scaled / thickness[j, i, level])
                zonal[j, i, level] = zu
                meridional[j, i, level] = zv
                numerator[j, i, level] = total
                scaled[j, i, level] = area_scaled * tmask[j, i, level]
                hdiv[j, i, level] = value * tmask[j, i, level]
                e3div[j, i, level] = np.float64(
                    value * thickness[j, i, level]) * tmask[j, i, level]
    carry = np.zeros((22, 32), dtype=np.float64)
    r3_delta = np.asarray(r3_kaa - r3_kbb, dtype=np.float64)
    for level in range(29, -1, -1):
        for j in range(22):
            for i in range(32):
                incoming[j, i, level] = carry[j, i]
                term = np.float64(
                    np.float64(r1_dt * e3t_0[j, i, level])
                    * r3_delta[j, i])
                stretch[j, i, level] = term
                total = np.float64(e3div[j, i, level] + term)
                bracket[j, i, level] = total
                carry[j, i] = np.float64(
                    carry[j, i] - np.float64(total * tmask[j, i, level]))
                outgoing[j, i, level] = carry[j, i]
    ww = np.concatenate(
        [outgoing, np.zeros((22, 32, 1), dtype=np.float64)], axis=-1)
    return {
        "transport_u": np.asarray(transport_u)[..., :30],
        "transport_v": np.asarray(transport_v)[..., :30],
        "zonal_difference": zonal,
        "meridional_difference": meridional,
        "numerator": numerator,
        "scaled": scaled,
        "hdiv": hdiv,
        "e3div": e3div,
        "r3_delta": r3_delta,
        "stretch": stretch,
        "bracket": bracket,
        "incoming_carry": incoming,
        "outgoing_carry": outgoing,
        "ww": ww,
    }


def _stage1_w_walk(
    records, transports, direct_stage_ww, direct_stage_state, direct_w_record,
    direct_r3_record, advmean_root: Path, plant: str | None,
):
    """Walk kt=1 stage-1 transport W from admitted NEMO operands."""
    import jax
    import jax.numpy as jnp

    arrays = records[(1, 1)]["arrays"]
    transport = transports[(1, 1)]
    transport_u = np.asarray(transport["zFu"])
    transport_v = np.asarray(transport["zFv"])
    direct_ww = np.asarray(direct_stage_ww[(1, 1)]["ww"])
    wet_surface = _owned3(arrays["tmask"])[..., 0] > 0.5
    reference_depth = np.sum(
        _owned3(arrays["e3t_0"]) * _owned3(arrays["tmask"]), axis=-1)
    require(np.all(reference_depth[wet_surface] > 0.0),
            "kt1 stage1 cannot recover positive reference depth")
    r1_h0 = np.zeros_like(reference_depth)
    r1_h0[wet_surface] = 1.0 / reference_depth[wet_surface]
    # Compiled stprk3_stg does not form the stage-1 ratio from the already
    # interpolated ssh(Kaa).  Under np_HYB it saves the full external-mode
    # ssha, forms r3ta from that full-step field, and independently applies
    # the RK3 interpolation to r3t (stprk3_stg.f90:150-190).  Keep the old
    # algebraic reconstruction as a scored retraction, but drive the W walk
    # with the association that NEMO actually executes.
    legacy_r3_kaa = (
        np.asarray(direct_stage_state[(1, 1)]["ssh"]) * r1_h0)
    full_external_ssh = np.asarray(
        _barotropic_override(records, advmean_root, 1)[0])
    r3ta = np.multiply(full_external_ssh, r1_h0, dtype=np.float64)
    r2_3 = np.float64(2.0) / np.float64(3.0)
    r1_3 = np.float64(1.0) / np.float64(3.0)
    r3_kbb = _owned2(arrays["r3t_Kbb"])
    r3_kaa = np.add(
        np.multiply(r2_3, r3_kbb, dtype=np.float64),
        np.multiply(r1_3, r3ta, dtype=np.float64),
        dtype=np.float64,
    )
    same_call = {
        name: _owned2(direct_r3_record[name])
        for name in (
            "ssh_kaa", "r1_ht_0", "r3_kaa", "ssh_kbb", "r3_kbb", "ht_0"
        )
    }
    same_call_kbb_product = np.multiply(
        same_call["ssh_kbb"], same_call["r1_ht_0"], dtype=np.float64)
    same_call_kaa_product = np.multiply(
        same_call["ssh_kaa"], same_call["r1_ht_0"], dtype=np.float64)
    same_call_full_ratio = np.multiply(
        full_external_ssh, same_call["r1_ht_0"], dtype=np.float64)
    same_call_hyb_ratio = np.add(
        np.multiply(r2_3, same_call["r3_kbb"], dtype=np.float64),
        np.multiply(r1_3, same_call_full_ratio, dtype=np.float64),
        dtype=np.float64,
    )
    same_call_ratio_rows = []
    for name, candidate, interpretation in (
        (
            "recorded_kbb_vs_stage_record",
            _owned2(arrays["r3t_Kbb"]),
            "independent_record_identity",
        ),
        (
            "ssh_kbb_times_stored_reciprocal",
            same_call_kbb_product,
            "compiled_dom_qco_r3c_RK3_product",
        ),
        (
            "ssh_kaa_times_stored_reciprocal",
            same_call_kaa_product,
            "retracted_algebraic_stage_ratio",
        ),
        (
            "independently_interpolated_hyb_ratio",
            same_call_hyb_ratio,
            "compiled_stprk3_stg_HYB_association",
        ),
    ):
        reference_name = "r3_kbb" if "kbb" in name else "r3_kaa"
        row = _classification(score(
            f"GYRE-zco.kt1.s1.r3_same_call.{name}",
            same_call[reference_name], candidate, wet_surface))
        row.update({"boundary": name, "interpretation": interpretation})
        same_call_ratio_rows.append(row)
    r1_dt = np.float64(1.0 / direct_stage_ww[(1, 1)]["rDt_s"])
    reference = _stage1_transport_w_scalar_reference(
        arrays, transport_u, transport_v, r3_kaa=r3_kaa, r1_dt=r1_dt)
    wet_w = _owned3(arrays["wmask"], 31) > 0.5
    scalar_direct = _classification(score(
        "GYRE-zco.kt1.s1.w_walk.scalar_replay_vs_direct_w",
        direct_ww, reference["ww"], wet_w))
    require(scalar_direct["absolute_max"] <= 3.0e-23,
            "scalar transport-W replay is outside its registered rounding bound")
    operands = (
        jnp.asarray(transport_u), jnp.asarray(transport_v),
        jnp.asarray(_owned2(arrays["r1_e1e2t"])),
        jnp.asarray(_owned3(arrays["e3t_Kmm"])),
        jnp.asarray(_owned3(arrays["e3t_0"])),
        jnp.asarray(_owned2(arrays["r3t_Kbb"])),
        jnp.asarray(r3_kaa),
        jnp.asarray(r1_dt),
        jnp.asarray(_owned3(arrays["tmask"])),
    )
    ordinary = jax.device_get(jax.jit(
        lambda: _stage1_transport_w_trace(*operands, source_round=False))())
    source_rounded = jax.device_get(jax.jit(
        lambda: _stage1_transport_w_trace(*operands, source_round=True))())
    wet_t = _owned3(arrays["tmask"]) > 0.5
    wet_2d = wet_t[..., 0]

    direct_fields = {
        "hdiv": np.asarray(direct_w_record["hdiv"])[1:-1, 1:-1, :30],
        "e3div": np.asarray(direct_w_record["e3div"])[1:-1, 1:-1, :30],
        "r3_kaa": _owned2(direct_w_record["r3_kaa"]),
        "r3_kbb": _owned2(direct_w_record["r3_kbb"]),
        "e3t_0": _owned3(direct_w_record["e3t_0"]),
        "r1_dt": np.asarray([direct_w_record["r1_dt"]], dtype=np.float64),
        "ww": _owned3(direct_w_record["ww"], 31),
    }
    reconstructed_fields = {
        "hdiv": np.asarray(source_rounded["hdiv"]),
        "e3div": np.asarray(source_rounded["e3div"]),
        "r3_kaa": np.asarray(r3_kaa),
        "r3_kbb": _owned2(arrays["r3t_Kbb"]),
        "e3t_0": _owned3(arrays["e3t_0"]),
        "r1_dt": np.asarray([r1_dt], dtype=np.float64),
        "ww": np.asarray(source_rounded["ww"]),
    }
    direct_masks = {
        "hdiv": wet_t, "e3div": wet_t,
        "r3_kaa": wet_2d, "r3_kbb": wet_2d,
        "e3t_0": wet_t, "r1_dt": np.asarray([True]), "ww": wet_w,
    }
    direct_input_order = (
        "hdiv", "e3div", "r3_kaa", "r3_kbb", "e3t_0", "r1_dt")
    direct_input_rows = []
    for name in direct_input_order:
        row = _classification(score(
            f"GYRE-zco.kt1.s1.w_walk.direct_input.{name}",
            direct_fields[name], reconstructed_fields[name], direct_masks[name]))
        row["boundary"] = name
        direct_input_rows.append(row)
    legacy_r3_row = _classification(score(
        "GYRE-zco.kt1.s1.w_walk.retracted_algebraic_r3_kaa",
        direct_fields["r3_kaa"], legacy_r3_kaa, wet_2d))
    legacy_r3_row.update({
        "boundary": "r3_kaa",
        "association": "interpolated_ssh_times_static_reciprocal",
        "interpretation": "RETRACTED_NOT_COMPILED_ORDER",
    })
    hyb_r3_row = _classification(score(
        "GYRE-zco.kt1.s1.w_walk.compiled_hyb_r3_kaa",
        direct_fields["r3_kaa"], r3_kaa, wet_2d))
    hyb_r3_row.update({
        "boundary": "r3_kaa",
        "association": "r2_3_times_r3_kbb_plus_r1_3_times_full_step_r3ta",
        "interpretation": "COMPILED_ORDER",
    })

    direct_scalar = _stage1_w_scalar_recurrence(
        direct_fields["e3div"], direct_fields["e3t_0"],
        direct_fields["r3_kbb"], direct_fields["r3_kaa"],
        direct_fields["r1_dt"][0], wet_t.astype(np.float64))
    carry_plant_at = None
    if plant == "stage-w-carry-ulp":
        j, i = (int(value) for value in np.argwhere(wet_2d)[0])
        carry_plant_at = (j, i, direct_fields["e3div"].shape[-1] - 1)
    direct_source = jax.device_get(jax.jit(
        lambda: _stage1_w_recurrence_trace(
            jnp.asarray(direct_fields["e3div"]),
            jnp.asarray(direct_fields["e3t_0"]),
            jnp.asarray(direct_fields["r3_kbb"]),
            jnp.asarray(direct_fields["r3_kaa"]),
            jnp.asarray(direct_fields["r1_dt"][0]),
            jnp.asarray(wet_t.astype(np.float64)), source_round=True,
            plant_carry_at=carry_plant_at))())
    recurrence_rows = []
    for name in (
        "r3_delta", "stretch", "bracket", "incoming_carry",
        "outgoing_carry", "ww",
    ):
        oracle = (direct_fields["ww"] if name == "ww"
                  else np.asarray(direct_scalar[name]))
        mask = wet_2d if name == "r3_delta" else wet_w if name == "ww" else wet_t
        row = _classification(score(
            f"GYRE-zco.kt1.s1.w_walk.direct_recurrence.{name}",
            oracle, np.asarray(direct_source[name]), mask))
        row["boundary"] = name
        recurrence_rows.append(row)
    if plant == "stage-w-carry-ulp":
        incoming_row = next(
            row for row in recurrence_rows
            if row["boundary"] == "incoming_carry")
        require(incoming_row["n_unequal"] == 1,
                "stage-w-carry-ulp did not flip exactly one direct incoming "
                "carry cell")

    direct_scalar_vs_w = _classification(score(
        "GYRE-zco.kt1.s1.w_walk.direct_scalar_replay_vs_recorded_w",
        direct_fields["ww"], direct_scalar["ww"], wet_w))
    round21_vs_round98 = _classification(score(
        "GYRE-zco.kt1.s1.w_walk.round21_w_vs_round98_direct_w",
        direct_fields["ww"], direct_ww, wet_w))
    r3_ulp_control = None
    if plant == "stage-w-direct-r3-ulp":
        planted_r3 = direct_fields["r3_kaa"].copy()
        at = tuple(np.argwhere(wet_2d)[0])
        planted_r3[at] = np.nextafter(planted_r3[at], np.float64(np.inf))
        r3_ulp_control = _classification(score(
            "GYRE-zco.kt1.s1.w_walk.control.direct_r3_kaa_ulp",
            direct_fields["r3_kaa"], planted_r3, wet_2d))
        require(r3_ulp_control["n_unequal"] == 1,
                "direct r3_kaa one-ULP plant did not flip exactly one cell")

    def active(name):
        if name == "ww":
            return wet_w
        if name == "r3_delta":
            return wet_2d
        return wet_t

    ordinary_rows = []
    candidate_rows = []
    for name in _STAGE1_W_ORDER:
        oracle = np.asarray(
            direct_ww if name == "ww" else reference[name]).copy()
        planted_at = None
        if plant == "stage-w-transport-ulp" and name == "transport_u":
            planted_at = tuple(int(value) for value in np.argwhere(wet_t)[0])
            oracle[planted_at] = np.nextafter(
                oracle[planted_at], np.float64(np.inf))
        for destination, values, label in (
            (ordinary_rows, ordinary, "production_association"),
            (candidate_rows, source_rounded, "source_rounded_candidate"),
        ):
            row = _classification(score(
                f"GYRE-zco.kt1.s1.w_walk.{label}.{name}",
                oracle, np.asarray(values[name]), active(name)))
            row.update({"boundary": name, "association": label,
                        "plant_index": planted_at})
            destination.append(row)
    first = next((row for row in ordinary_rows
                  if row["classification"] != "BIT"), None)
    candidate_first = next((row for row in candidate_rows
                            if row["classification"] != "BIT"), None)
    if plant:
        if plant == "stage-w-transport-ulp":
            target = "transport_u"
            row = next(
                value for value in ordinary_rows if value["boundary"] == target)
            require(row["n_unequal"] == 1,
                    f"{plant} did not flip exactly one named row cell")
    direct_first = next(
        (row for row in direct_input_rows if row["classification"] != "BIT"),
        None)
    recurrence_first = next(
        (row for row in recurrence_rows if row["classification"] != "BIT"),
        None)
    return {
        "compiled_order": list(_STAGE1_W_ORDER),
        "direct_record_clock_seconds": direct_stage_ww[(1, 1)]["rDt_s"],
        "pre_external_zad_operand_w": _classification(score(
            "GYRE-zco.kt1.s1.w_walk.pre_external_zad_operand_w",
            _owned3(arrays["ww"], 31), direct_ww, wet_w)),
        "production_rows": ordinary_rows,
        "first_nonbit": (None if first is None else {
            key: first[key] for key in (
                "boundary", "n_unequal", "absolute_max", "classification")}),
        "source_rounded_candidate_rows": candidate_rows,
        "source_rounded_first_nonbit": (
            None if candidate_first is None else {
                key: candidate_first[key] for key in (
                    "boundary", "n_unequal", "absolute_max", "classification")}),
        "scalar_replay_vs_direct_w": scalar_direct,
        "direct_record": {
            "sha256": direct_w_record["sha256"],
            "producer_commit": direct_w_record["producer_commit"],
            "header": direct_w_record["header"],
        },
        "same_call_ratio_record": {
            "sha256": direct_r3_record["sha256"],
            "producer_commit": direct_r3_record["producer_commit"],
            "header": direct_r3_record["header"],
        },
        "same_call_ratio_rows": same_call_ratio_rows,
        "direct_input_rows": direct_input_rows,
        "r3_kaa_association_rows": [legacy_r3_row, hyb_r3_row],
        "direct_input_first_nonbit": (
            None if direct_first is None else {
                key: direct_first[key] for key in (
                    "boundary", "n_unequal", "absolute_max", "classification")
            }),
        "direct_recurrence_rows": recurrence_rows,
        "direct_recurrence_first_nonbit": (
            None if recurrence_first is None else {
                key: recurrence_first[key] for key in (
                    "boundary", "n_unequal", "absolute_max", "classification")
            }),
        "direct_scalar_replay_vs_recorded_w": direct_scalar_vs_w,
        "round21_w_vs_round98_direct_w": round21_vs_round98,
        "direct_r3_ulp_control": r3_ulp_control,
        "first_direct_nonbit_statement": (
            None if direct_first is None and recurrence_first is None else
            {key: (direct_first or recurrence_first)[key] for key in (
                "boundary", "n_unequal", "absolute_max", "classification")}
        ),
        "plant": plant,
    }


def _external_rows(outputs, histories, records, advmean_root: Path,
                   memory_root: Path, btstep_root: Path, masks, kt: int,
                   mode: str) -> list[dict]:
    """Score the split-explicit handoff consumed by the RK3 stage ladder."""
    eta, uu_b, vv_b, hu_avg, hv_avg = (
        np.asarray(value) for value in outputs)
    reference = _barotropic_override(records, advmean_root, kt)
    ref_eta, ref_u, ref_v, ref_hu, ref_hv = (
        np.asarray(value) for value in reference)
    rows = []
    fields = (
        ("ssh", ref_eta, eta, masks["ssh"]),
        ("uu_b", ref_u[:, 1:], uu_b[:, 1:], masks["u"][..., 0]),
        ("vv_b", ref_v[1:, :], vv_b[1:, :], masks["v"][..., 0]),
        ("Hu_avg", ref_hu[:, 1:], hu_avg[:, 1:], masks["u"][..., 0]),
        ("Hv_avg", ref_hv[1:, :], hv_avg[1:, :], masks["v"][..., 0]),
    )
    for field, oracle, candidate, mask in fields:
        row = _classification(score(
            f"GYRE-zco.kt{kt}.external.output.{field}",
            oracle, candidate, mask))
        row.update({"kt": kt, "stage": "external", "field": field,
                    "entry_mode": mode})
        rows.append(row)
    history_ref = _history_reference(memory_root, btstep_root, kt)
    history_fields = (
        ("ub_e", history_ref[0], histories[0][:, 1:], masks["u"][..., 0]),
        ("ubb_e", history_ref[1], histories[1][:, 1:], masks["u"][..., 0]),
        ("vb_e", history_ref[2], histories[2][1:, :], masks["v"][..., 0]),
        ("vbb_e", history_ref[3], histories[3][1:, :], masks["v"][..., 0]),
        ("sshb_e", history_ref[4], histories[4], masks["ssh"]),
        ("sshbb_e", history_ref[5], histories[5], masks["ssh"]),
    )
    for field, oracle, candidate, mask in history_fields:
        if kt == 1:
            oracle = _owned2(oracle)
        row = _classification(score(
            f"GYRE-zco.kt{kt}.external.output.{field}",
            oracle, candidate, mask))
        row.update({"kt": kt, "stage": "external", "field": field,
                    "entry_mode": mode})
        rows.append(row)
    return rows


def _rk3_vector_assignment_scalar(before, rhs, dt_stage, face_mask):
    """Replay the three binary64 operations in the compiled vector write."""
    before = np.asarray(before, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    face_mask = np.asarray(face_mask, dtype=np.float64)
    scaled = np.multiply(np.float64(dt_stage), rhs, dtype=np.float64)
    summed = np.add(before, scaled, dtype=np.float64)
    return np.multiply(summed, face_mask, dtype=np.float64)


def _assignment_execution_discriminator(
    traces: dict, records: dict, plant: str | None,
) -> dict:
    """Separate isolated eager/JIT proof from the full production step."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        rk3_stage_velocity_update,
    )

    def isolated(before, rhs, dt_stage, mask):
        return rk3_stage_velocity_update(
            before, rhs, dt_stage, mask, vector_form=True)

    isolated_jit = jax.jit(isolated)
    rows = []
    plant_target = None
    for kt, stage in sorted(traces):
        trace = traces[(kt, stage)]
        arrays = records[(kt, stage)]["arrays"]
        rhs_boundary = "pre_zdf_rhs" if stage == 3 else "after_adv"
        dt_stage = np.float64(trace.stage_coefficients[stage - 1][0])
        for face, face_index in (("u", 0), ("v", 1)):
            before = _owned3(arrays[f"{face}_Kbb"])
            rhs = _owned3(arrays[f"{rhs_boundary}_{face}"])
            active = _owned3(arrays[f"{face}mask"]) > 0.5
            mask = active.astype(np.float64)
            reference_kind = "direct_nemo_output"
            if stage < 3:
                oracle = _owned3(arrays[f"post_update_{face}"])
            else:
                oracle = _rk3_vector_assignment_scalar(
                    before, rhs, dt_stage, mask)
                reference_kind = "compiled_statement_transcription"

            eager = np.asarray(isolated(
                jnp.asarray(before), jnp.asarray(rhs), jnp.asarray(dt_stage),
                jnp.asarray(mask)))
            compiled = np.asarray(jax.device_get(isolated_jit(
                jnp.asarray(before), jnp.asarray(rhs), jnp.asarray(dt_stage),
                jnp.asarray(mask))))
            for execution, candidate in (
                ("isolated-closure eager", eager),
                ("isolated-closure JIT", compiled),
            ):
                row = _classification(score(
                    f"GYRE-zco.kt{kt}.s{stage}.rk3_assignment.{face}."
                    f"{execution.replace(' ', '_')}",
                    oracle, candidate, active))
                row.update({
                    "kt": kt, "stage": stage, "face": face,
                    "execution": execution,
                    "reference_kind": reference_kind,
                    "nemo_statement": (
                        "stprk3_stg.f90:671-674" if stage < 3
                        else "dynzdf.f90:166-170"),
                })
                rows.append(row)

            raw_full = np.asarray(trace.stage_raw_velocities[stage - 1][face_index])
            rhs_full = np.asarray(trace.stage_rhs[stage - 1][face_index])
            before_full = np.asarray(trace.stage_states[0][face_index])
            if face == "u":
                production_raw = raw_full[:, 1:, :]
                production_rhs = rhs_full[:, 1:, :]
                production_before = before_full[:, 1:, :]
            else:
                production_raw = raw_full[1:, :, :]
                production_rhs = rhs_full[1:, :, :]
                production_before = before_full[1:, :, :]
            production_replay = _rk3_vector_assignment_scalar(
                production_before, production_rhs, dt_stage, mask)
            clean = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.rk3_assignment.{face}."
                "production_step_transcription",
                production_replay, production_raw, active))
            scored_reference = production_replay
            planted_at = None
            if (plant == "stage-assignment-output-ulp"
                    and kt == 1 and stage == 1 and face == "u"):
                equal = (
                    np.ascontiguousarray(production_replay).view(np.uint64)
                    == np.ascontiguousarray(production_raw).view(np.uint64)
                ) & active & np.isfinite(production_replay)
                indices = np.argwhere(equal)
                require(indices.size > 0,
                        "production assignment plant found no exact wet cell")
                planted_at = tuple(int(value) for value in indices[0])
                scored_reference = production_replay.copy()
                scored_reference[planted_at] = np.nextafter(
                    scored_reference[planted_at], np.float64(np.inf))
            production_row = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.rk3_assignment.{face}.production_step",
                scored_reference, production_raw, active))
            production_row.update({
                "kt": kt, "stage": stage, "face": face,
                "execution": "production step",
                "reference_kind": "scalar_replay_from_captured_live_operands",
                "clean_n_unequal": clean["n_unequal"],
                "clean_absolute_max": clean["absolute_max"],
                "plant_index": planted_at,
                "nemo_statement": (
                    "stprk3_stg.f90:671-674" if stage < 3
                    else "dynzdf.f90:166-170"),
            })
            rows.append(production_row)
            if planted_at is not None:
                require(
                    production_row["n_unequal"] == clean["n_unequal"] + 1,
                    "production assignment one-ULP plant did not add exactly "
                    "one unequal cell")
                plant_target = production_row["name"]

            production_nemo = _classification(score(
                f"GYRE-zco.kt{kt}.s{stage}.rk3_assignment.{face}."
                "production_step_vs_nemo",
                oracle, production_raw, active))
            production_nemo.update({
                "kt": kt, "stage": stage, "face": face,
                "execution": "production step",
                "reference_kind": reference_kind,
                "interpretation": "includes_live_operand_error",
                "nemo_statement": (
                    "stprk3_stg.f90:671-674" if stage < 3
                    else "dynzdf.f90:166-170"),
            })
            rows.append(production_nemo)

    isolated_rows = [
        row for row in rows if row["execution"].startswith("isolated-")]
    production_rows = [
        row for row in rows
        if row["execution"] == "production step"
        and row["reference_kind"]
        == "scalar_replay_from_captured_live_operands"]
    discriminator_fired = (
        all(row["classification"] == "BIT" for row in isolated_rows)
        and any(row["classification"] != "BIT" for row in production_rows)
    )
    if plant == "stage-assignment-output-ulp":
        require(plant_target is not None,
                "production assignment plant did not reach its target")
    return {
        "rows": rows,
        "isolated_rows_all_bit": all(
            row["classification"] == "BIT" for row in isolated_rows),
        "production_transcription_rows_all_bit": all(
            row["classification"] == "BIT" for row in production_rows),
        "production_fusion_discriminator_fired": discriminator_fired,
        "plant_target": plant_target,
        "plant": plant == "stage-assignment-output-ulp",
    }


def _tke_production_statement_rows(
    trace, statement_record: dict, entry_mode: str,
    plant: str | None = None,
) -> dict:
    """Score compiled TKE statement boundaries from the full jitted step."""
    production = trace.tke_statement_trace
    fields = (
        "en_entry", "en_after_boundaries", "en_after_langmuir",
        "rhs_pre_sweep", "en_post_sweep",
    )
    statements = {
        "en_entry": "zdftke.f90:268",
        "en_after_boundaries": "zdftke.f90:284-324",
        "en_after_langmuir": "zdftke.f90:326-395",
        "rhs_pre_sweep": "zdftke.f90:399-473",
        "en_post_sweep": "zdftke.f90:475-495",
    }
    rows = []
    plant_target = None
    for field in fields:
        candidate = np.asarray(getattr(production, field))
        # The raw Fortran stream is (i,j,k); the model production trace and
        # every stage-table field use (lat=j, lon=i, k).
        reference_full = np.asarray(
            statement_record["arrays"][field]).swapaxes(0, 1)
        # legoESM carries NEMO levels 2:jpkm1 at entry.  Once the compiled
        # boundary assignment has executed, the private trace prepends the
        # separately represented z=0 row and spans levels 1:jpkm1.
        reference = (reference_full[..., 1:30]
                     if field == "en_entry" else reference_full[..., :30])
        require(candidate.shape == reference.shape,
                f"production TKE {field} shape {candidate.shape} != "
                f"oracle {reference.shape}")
        clean = _classification(score(
            f"GYRE-zco.kt2.tke_statement.production_step.{field}.clean",
            reference, candidate, np.ones(reference.shape, dtype=bool)))
        scored_reference = reference
        planted_at = None
        if (plant == "stage-tke-production-ulp"
                and entry_mode == "NEMO_TKE_RECORDED"
                and field == "en_after_boundaries"):
            equal = (
                np.ascontiguousarray(reference).view(np.uint64)
                == np.ascontiguousarray(candidate).view(np.uint64)
            ) & np.isfinite(reference)
            indices = np.argwhere(equal)
            require(indices.size > 0,
                    "TKE production plant found no exact boundary cell")
            planted_at = tuple(int(value) for value in indices[0])
            scored_reference = reference.copy()
            scored_reference[planted_at] = np.nextafter(
                scored_reference[planted_at], np.float64(np.inf))
        row = _classification(score(
            f"GYRE-zco.kt2.tke_statement.production_step.{field}",
            scored_reference, candidate,
            np.ones(reference.shape, dtype=bool)))
        row.update({
            "kt": 2,
            "stage": 1,
            "field": field,
            "entry_mode": entry_mode,
            "execution": "production step",
            "domain": ("NEMO levels 2:jpkm1" if field == "en_entry"
                       else "NEMO levels 1:jpkm1"),
            "nemo_statement": statements[field],
            "clean_n_unequal": clean["n_unequal"],
            "clean_absolute_max": clean["absolute_max"],
            "plant_index": planted_at,
        })
        rows.append(row)
        if planted_at is not None:
            require(row["n_unequal"] == clean["n_unequal"] + 1,
                    "TKE production one-ULP plant did not add exactly one "
                    "unequal cell")
            plant_target = row["name"]
    if (plant == "stage-tke-production-ulp"
            and entry_mode == "NEMO_TKE_RECORDED"):
        require(plant_target is not None,
                "TKE production plant did not reach its target")
    # Decompose the first moved block in compiled order.  The NEMO-changed
    # interior mask is defined by the two independently written oracle
    # boundaries, never by legoESM's output, so a candidate cannot choose the
    # cells on which it is judged.
    oracle_entry = np.asarray(
        statement_record["arrays"]["en_entry"]).swapaxes(0, 1)[..., 1:30]
    oracle_boundary = np.asarray(
        statement_record["arrays"]["en_after_boundaries"]
    ).swapaxes(0, 1)[..., :30]
    candidate_boundary = np.asarray(production.en_after_boundaries)
    changed_interior = (
        np.ascontiguousarray(oracle_boundary[..., 1:30]).view(np.uint64)
        != np.ascontiguousarray(oracle_entry).view(np.uint64)
    )
    unchanged_interior = ~changed_interior
    require(
        oracle_boundary[..., :1].size
        + int(np.count_nonzero(changed_interior))
        + int(np.count_nonzero(unchanged_interior))
        == oracle_boundary.size,
        "TKE boundary decomposition does not partition the binding domain")
    boundary_block_rows = []
    for name, reference, candidate, mask, statement in (
        ("surface_assignment", oracle_boundary[..., :1],
         candidate_boundary[..., :1],
         np.ones(oracle_boundary[..., :1].shape, dtype=bool),
         "zdftke.f90:284-289"),
        ("bottom_assignment_changed_cells", oracle_boundary[..., 1:30],
         candidate_boundary[..., 1:30], changed_interior,
         "zdftke.f90:299-308"),
        ("unchanged_interior", oracle_boundary[..., 1:30],
         candidate_boundary[..., 1:30], unchanged_interior,
         "no compiled boundary write"),
    ):
        selected = int(np.count_nonzero(mask))
        if selected:
            row = _classification(score(
                f"GYRE-zco.kt2.tke_statement.production_step.boundary.{name}",
                reference, candidate, mask))
        else:
            row = {
                "name": (
                    "GYRE-zco.kt2.tke_statement.production_step.boundary."
                    f"{name}"),
                "n_unequal": 0,
                "absolute_max": 0.0,
                "classification": "BIT",
            }
        row.update({
            "field": name,
            "entry_mode": entry_mode,
            "execution": "production step",
            "selected_cells": selected,
            "nemo_statement": statement,
        })
        boundary_block_rows.append(row)
    surface_row, bottom_row, unchanged_row = boundary_block_rows
    if surface_row["classification"] != "BIT":
        first_boundary_statement = {
            "field": surface_row["field"],
            "nemo_statement": surface_row["nemo_statement"],
            "n_unequal": surface_row["n_unequal"],
            "absolute_max": surface_row["absolute_max"],
            "classification": surface_row["classification"],
        }
    elif unchanged_row["classification"] != "BIT":
        first_boundary_statement = None
    elif bottom_row["classification"] != "BIT":
        first_boundary_statement = {
            "field": bottom_row["field"],
            "nemo_statement": bottom_row["nemo_statement"],
            "n_unequal": bottom_row["n_unequal"],
            "absolute_max": bottom_row["absolute_max"],
            "classification": bottom_row["classification"],
        }
    else:
        first_boundary_statement = None

    first = next(
        (row for row in rows if row["classification"] != "BIT"), None)
    return {
        "execution": "production step",
        "entry_mode": entry_mode,
        "rows": rows,
        "boundary_block_rows": boundary_block_rows,
        "first_nonbit_boundary_statement": first_boundary_statement,
        "entry_surface_level1": {
            "classification": "UNMEASURED_WITH_SPEC",
            "reason": (
                "legoesm does not carry NEMO en(jk=1) at entry because "
                "zdftke.f90:284-289 overwrites it before any later consumer"
            ),
        },
        "first_nonbit": None if first is None else {
            key: first[key] for key in (
                "field", "n_unequal", "absolute_max", "classification",
                "nemo_statement")
        },
        "plant_target": plant_target,
    }


def _tke_matrix_statement_rows(
    trace, operand_record: dict, statement_record: dict, entry_mode: str,
    plant: str | None = None,
) -> dict:
    """Subdivide the compiled matrix/RHS block into its recorded outputs.

    The Round-101 walk reports ``zdftke.f90:399-473`` as one row.  The
    independently admitted Round-59 record stores the four arrays that block
    writes, captured at ``zdftke.f90:472`` -- the statement immediately before
    the Round-101 RHS callback at ``:473`` -- so the same instant can be split
    into one row per compiled assignment without a new acquisition.

    ``p_pdlr`` (``zdftke.f90:421``) executes first but is read at exactly one
    place, ``zdftke.f90:712`` inside ``tke_avn``, and never enters ``en``; it
    is reported UNMEASURED-WITH-SPEC rather than silently dropped.
    """
    production = trace.tke_statement_trace
    arrays = operand_record["arrays"]

    def solved(name: str) -> np.ndarray:
        # Raw stream is (i,j,k); every model-side field is (lat=j, lon=i, k).
        # NEMO jk = 2..jpkm1 is record index 1..29.
        return np.asarray(arrays[name]).swapaxes(0, 1)[..., 1:30]

    # Compiled order inside `DO jk = 2, jpkm1` (zdftke.f90:425-443).
    walk = (
        ("zd_up", "matrix_upper", "zdftke.f90:434",
         "zd_up(ji,jk) = zzd_up"),
        ("zd_lw", "matrix_lower", "zdftke.f90:435",
         "zd_lw(ji,jk) = zzd_lw"),
        ("zdiag", "matrix_diag", "zdftke.f90:436",
         "zdiag(ji,jk) = 1 - zzd_lw - zzd_up + zfact2*dissl*wmask"),
        ("en_rhs", "rhs_pre_sweep", "zdftke.f90:439-442",
         "en(ji,jj,jk) = en + rn_Dt*(p_sh2 - p_avt*rn2 + zfact3*dissl*en)"
         "*wmask"),
    )
    record_field = {
        "zd_up": "matrix_upper",
        "zd_lw": "matrix_lower",
        "zdiag": "matrix_diag",
        "en_rhs": "rhs_pre_sweep",
    }
    rows = []
    plant_target = None
    for name, trace_field, statement, text in walk:
        candidate = np.asarray(getattr(production, trace_field))
        if trace_field == "rhs_pre_sweep":
            # The traced RHS prepends the separately held z=0 row.
            candidate = candidate[..., 1:]
        reference = solved(record_field[name])
        require(candidate.shape == reference.shape,
                f"production TKE {name} shape {candidate.shape} != "
                f"oracle {reference.shape}")
        mask = np.ones(reference.shape, dtype=bool)
        clean = _classification(score(
            f"GYRE-zco.kt2.tke_matrix.production_step.{name}.clean",
            reference, candidate, mask))
        scored_reference = reference
        planted_at = None
        if ((plant == "stage-tke-matrix-ulp" and name == "zdiag")
                or (plant == "stage-tke-rhs-ulp" and name == "en_rhs")):
            equal = (
                np.ascontiguousarray(reference).view(np.uint64)
                == np.ascontiguousarray(candidate).view(np.uint64)
            ) & np.isfinite(reference)
            indices = np.argwhere(equal)
            require(indices.size > 0,
                    "TKE matrix/RHS plant found no exact target cell")
            planted_at = tuple(int(value) for value in indices[0])
            scored_reference = reference.copy()
            scored_reference[planted_at] = np.nextafter(
                scored_reference[planted_at], np.float64(np.inf))
        row = _classification(score(
            f"GYRE-zco.kt2.tke_matrix.production_step.{name}",
            scored_reference, candidate, mask))
        row.update({
            "kt": 2,
            "stage": 1,
            "field": name,
            "entry_mode": entry_mode,
            "execution": "production step",
            "domain": "NEMO levels 2:jpkm1",
            "nemo_statement": statement,
            "nemo_text": text,
            "clean_n_unequal": clean["n_unequal"],
            "clean_absolute_max": clean["absolute_max"],
            "plant_index": planted_at,
        })
        rows.append(row)
        if planted_at is not None:
            require(row["n_unequal"] == clean["n_unequal"] + 1,
                    "TKE matrix/RHS one-ULP plant did not add exactly one "
                    "unequal cell")
            plant_target = row["name"]
    if plant in ("stage-tke-matrix-ulp", "stage-tke-rhs-ulp"):
        require(plant_target is not None,
                "TKE matrix/RHS plant did not reach its target")

    # Operand attribution for the RHS assignment: is the miss owned by the
    # arithmetic or inherited through p_sh2 (zdftke.f90:439)?
    shear_reference = solved("sh2")
    shear_candidate = np.asarray(production.rhs_shear)
    require(shear_candidate.shape == shear_reference.shape,
            f"production p_sh2 shape {shear_candidate.shape} != "
            f"oracle {shear_reference.shape}")
    shear_row = _classification(score(
        "GYRE-zco.kt2.tke_matrix.production_step.p_sh2_operand",
        shear_reference, shear_candidate,
        np.ones(shear_reference.shape, dtype=bool)))
    shear_row.update({
        "field": "p_sh2_operand",
        "entry_mode": entry_mode,
        "execution": "production step",
        "domain": "NEMO levels 2:jpkm1",
        "nemo_statement": "zdftke.f90:439",
        "note": "consumed operand, not an output of this block",
    })
    rhs_row = next(row for row in rows if row["field"] == "en_rhs")

    # ONE-VARIABLE SWAP.  A set-inclusion test ("every RHS-unequal cell is a
    # p_sh2-unequal cell") is NOT causal: NEMO's RHS also consumes p_avt,
    # rn2, dissl, the post-Langmuir en and wmask, and coincident errors in
    # those would pass it.  So rebuild the statement from NEMO's own recorded
    # operands with EXACTLY ONE substitution -- the production p_sh2 -- using
    # the single committed transcription, and ask whether that alone
    # reproduces the production output.
    from nemo_testcase_l2_gyre_round103_tke_block_replay import rebuild_block

    langmuir_reference = np.asarray(
        statement_record["arrays"]["en_after_langmuir"]).swapaxes(0, 1)
    production_rhs = np.asarray(production.rhs_pre_sweep)[..., 1:]
    recorded_rhs = solved("rhs_pre_sweep")
    # The raw record stream is (i,j,k); every production field is (j,i,k).
    # Hand the replay a consistently oriented view so its output can be
    # compared against the production arrays without a second transpose.
    oriented = {
        key: (np.asarray(value).swapaxes(0, 1)
              if np.ndim(value) == 3 else value)
        for key, value in arrays.items()
    }
    held = rebuild_block(oriented, langmuir_reference)["en_rhs"]
    swapped = rebuild_block(
        oriented, langmuir_reference,
        p_sh2_override=shear_candidate)["en_rhs"]

    def _row(name, reference, candidate, note):
        out = _classification(score(
            f"GYRE-zco.kt2.tke_matrix.production_step.swap.{name}",
            reference, candidate, np.ones(reference.shape, dtype=bool)))
        out.update({
            "field": name, "entry_mode": entry_mode,
            "execution": "recorded-operand replay of the compiled statement",
            "note": note,
        })
        return out

    all_held = _row(
        "all_operands_recorded", recorded_rhs, held,
        "every operand at NEMO's recorded value; must be BIT or the "
        "reference side of the comparison is itself wrong")
    swap_vs_production = _row(
        "p_sh2_swapped_vs_production", production_rhs, swapped,
        "only p_sh2 replaced by the production value; BIT means every other "
        "operand the production step fed this statement was already NEMO's")
    swap_vs_nemo = _row(
        "p_sh2_swapped_vs_nemo", recorded_rhs, swapped,
        "only p_sh2 replaced; must reproduce the production row exactly")

    sole = (
        all_held["classification"] == "BIT"
        and swap_vs_production["classification"] == "BIT"
        and swap_vs_nemo["n_unequal"] == rhs_row["n_unequal"]
        and swap_vs_nemo["absolute_max"] == rhs_row["absolute_max"]
        and shear_row["classification"] != "BIT"
        and rhs_row["classification"] != "BIT")

    rhs_unequal = (
        np.ascontiguousarray(recorded_rhs).view(np.uint64)
        != np.ascontiguousarray(production_rhs).view(np.uint64))
    shear_unequal = (
        np.ascontiguousarray(shear_reference).view(np.uint64)
        != np.ascontiguousarray(shear_candidate).view(np.uint64))
    rhs_not_explained = int(np.count_nonzero(rhs_unequal & ~shear_unequal))
    attribution = {
        "prediction": (
            "swapping ONLY p_sh2 into NEMO's recorded operands reproduces "
            "the production right-hand side exactly"),
        "p_sh2_row": shear_row,
        "one_variable_swap": [all_held, swap_vs_production, swap_vs_nemo],
        "rhs_unequal_cells": int(np.count_nonzero(rhs_unequal)),
        "p_sh2_unequal_cells": int(np.count_nonzero(shear_unequal)),
        "rhs_unequal_cells_with_bit_equal_p_sh2": rhs_not_explained,
        "subset_test_note": (
            "reported for continuity only; it is NOT the evidence -- "
            "coincident operand errors would pass it"),
        "verdict": (
            "CONFIRMED_SOLE_OPERAND_P_SH2" if sole else "REFUTED"),
    }

    first = next(
        (row for row in rows if row["classification"] != "BIT"), None)
    return {
        "execution": "production step",
        "entry_mode": entry_mode,
        "record": "round59/oracle_tke_operands_kt00000002.bin",
        "record_write_site": (
            "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:172-190"
            " called from "
            "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:463"),
        "reference_build_statements": (
            "the matrix/RHS writes in the Round-59 build are "
            "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:425-427 and "
            ":430-433; the two builds' compiled zdftke.f90 differ in zero "
            "lines once the recorder calls are removed, so the citations "
            "below to the Round-101 build name the same statements"),
        "rows": rows,
        "p_sh2_attribution": attribution,
        "excluded_outputs": [
            {
                "field": "p_pdlr",
                "nemo_statement": "zdftke.f90:421",
                "classification": "UNMEASURED_WITH_SPEC",
                "reason": (
                    "first in compiled order but read only at "
                    "zdftke.f90:712 inside tke_avn; it never enters en, so "
                    "it is a separate consumer chain and needs its own walk"
                ),
            },
            {
                "field": "wave_coupled_surface_block",
                "nemo_statement": "zdftke.f90:451-468",
                "classification": "NOT_EXECUTED",
                "reason": (
                    "cpl_phioc is set .TRUE. only in sbccpl.f90:629 (coupled "
                    "runs) and ln_phioc = .false. in EXP00/namelist_ref:593"
                ),
            },
        ],
        "first_nonbit": None if first is None else {
            key: first[key] for key in (
                "field", "n_unequal", "absolute_max", "classification",
                "nemo_statement", "nemo_text")
        },
        "plant_target": plant_target,
    }


def _shear_statement_rows(
    trace, records: dict, card, state, operand_record: dict,
    entry_mode: str, *, legacy_trace=None, plant: str | None = None,
    execution: str = "production step JIT",
) -> dict:
    """Walk NEMO's shear-production routine at its OWN stage (decision 41).

    Round 103 named ``p_sh2`` as the sole magnitude owner of the TKE
    right-hand-side miss and stopped, because ``p_sh2`` is written in a
    different routine.  This is that routine, driven from NEMO's recorded
    stage entry through the real production step.

    The compiled statements, Round-59 build
    ``GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90``, executed ELSE
    branch (``cpl_sdrftx .AND. ln_stshear`` is false on this card):
    ``zsh2u`` at ``:99-103``, ``zsh2v`` at ``:104-108``, ``p_sh2`` at
    ``:112-113``.  Only the last has a recorded output, so the two face
    statements are attributed by one-variable operand swaps rather than by a
    row of their own; the groups are named in ``OPERAND_GROUPS``.

    ``legacy_trace``, when given, is a SECOND production step that differs
    from the first in exactly one thing: the free-surface field routed into
    the shear's face metric.  It is the discriminator for the time level,
    and it is a whole-step ablation, so ONLY its ``p_sh2`` row is read here.
    """
    from nemo_testcase_l2_gyre_round104_shear_replay import (
        OPERAND_GROUPS, model_operands, recorded_operands, rebuild_sh2,
        _lat_lon as shear_lat_lon, K as SHEAR_K,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )

    arrays = operand_record["arrays"]
    recorded = shear_lat_lon(np.asarray(arrays["sh2"]))[..., SHEAR_K]
    op, recorded_face_metrics = recorded_operands(
        records[(2, 1)]["arrays"], arrays["avm_entry"],
        return_face_metrics=True)
    z_coord = card.recipe.z_coord
    u_mask, v_mask = compute_face_masks_3d(z_coord.is_active)
    model = model_operands(
        z_coord, state.u.data, state.v.data, state.eta.data,
        state.tke_avm.data, u_mask, v_mask)
    production = np.asarray(trace.tke_statement_trace.rhs_shear)
    require(production.shape == recorded.shape,
            f"production p_sh2 shape {production.shape} != oracle "
            f"{recorded.shape}")

    def row(name, reference, candidate, statement, note):
        out = _classification(score(
            f"GYRE-zco.kt2.tke_shear.{name}", reference, candidate,
            np.ones(np.shape(reference), dtype=bool)))
        out.update({
            "field": name, "entry_mode": entry_mode,
            "execution": execution,
            "nemo_statement": statement, "note": note,
        })
        return out

    captured_face_metrics = trace.tke_statement_trace.shear_face_metrics
    require(captured_face_metrics is not None,
            "production trace did not capture the shear face metrics")
    require(len(captured_face_metrics) == 4,
            "production shear trace must contain four face metrics")
    metric_names = ("e3u_now", "e3u_before", "e3v_now", "e3v_before")
    captured_metric_rows = [
        row(
            f"captured_face_metric.{name}", reference,
            np.asarray(candidate), "R59TKE zdfsh2.f90:102,107",
            "direct WRITE-only capture of the face metric consumed by the "
            "same production step; reference is reconstructed from NEMO's "
            "recorded r3 and raw e3w_1d operands",
        )
        for name, reference, candidate in zip(
            metric_names, recorded_face_metrics, captured_face_metrics)
    ]

    # The plant lands on the TIME-LEVEL row, not on the baseline production
    # row.  The baseline row differs in every wet cell, so a one-ULP plant on
    # its reference could only ever pick a DRY cell, where it proves the row
    # plumbing moves and nothing about the arithmetic.  The time-level row is
    # bit-exact over all 20,416 cells, so the same plant can be required to
    # land on a WET cell -- one the shear arithmetic actually produced.
    plant_target = None
    planted_at = None

    rows = [
        row("all_operands_recorded", recorded, rebuild_sh2(op),
            "R59TKE zdfsh2.f90:99-113",
            "every operand at NEMO's recorded value; must be BIT or the "
            "reference side of this walk is itself wrong"),
        row("production_step_vs_recorded", recorded, production,
            "R59TKE zdfsh2.f90:99-113",
            "the model's own p_sh2 through the real production step, "
            "driven from NEMO's recorded stage entry"),
        row("model_operand_replay_vs_production", production,
            rebuild_sh2(op, **model), "R59TKE zdfsh2.f90:99-113",
            "the compiled statement on NEMO'S OWN time level for the face "
            "metric and the model's values for everything else, against what "
            "the model actually produced; NOT-BIT here is the finding, not a "
            "defect in the mirror -- the production step builds that metric "
            "from a DIFFERENT free-surface field, which is what the "
            "time-level rows below isolate"),
    ]

    operand_rows = [
        row(f"operand.{name}", op[name], model[name],
            "R59TKE zdfsh2.f90:99-113",
            "one model-constructed zdf_sh2 operand against NEMO's recorded "
            "one, on the same window")
        for name in sorted(op)
    ]
    swap_rows = [
        row(f"swap.{group}", production,
            rebuild_sh2(op, **{k: model[k] for k in keys}),
            "R59TKE zdfsh2.f90:99-113",
            "NEMO's recorded operands with ONLY this group replaced by the "
            "model's, against the production output")
        for group, keys in OPERAND_GROUPS.items()
    ]

    time_level = None
    if legacy_trace is not None:
        legacy = np.asarray(legacy_trace.tke_statement_trace.rhs_shear)
        scored_recorded = recorded
        if plant == "stage-shear-operand-ulp":
            clean_legacy = _classification(score(
                "GYRE-zco.kt2.tke_shear.time_level.clean",
                recorded, legacy, np.ones(recorded.shape, dtype=bool)))
            require(clean_legacy["n_unequal"] == 0,
                    "shear plant needs a bit-exact time-level row to corrupt")
            # WET only: a dry cell is zero on both sides by construction and
            # corrupting it would prove nothing about the arithmetic.
            wet = (recorded != 0.0) & np.isfinite(recorded)
            indices = np.argwhere(wet)
            require(indices.size > 0,
                    "shear plant found no WET p_sh2 cell to corrupt")
            planted_at = tuple(int(v) for v in indices[0])
            scored_recorded = recorded.copy()
            scored_recorded[planted_at] = np.nextafter(
                scored_recorded[planted_at], np.float64(np.inf))
        legacy_rows = [
            row("time_level.step_entry_ssh_production_vs_recorded",
                scored_recorded, legacy, "R59TKE zdfsh2.f90:102,107",
                "the SAME production step with exactly one thing changed: "
                "the free-surface field routed into the shear's face "
                "metric is the step-entry ssh, which is what "
                "stprk3.f90:168's zdf_phy(kstp,Nbb,Nbb,Nrhs) and "
                "zdfphy.f90:319-320's zdf_sh2(Kbb,Kmm,...) make r3u(Kmm) "
                "and r3u(Kbb) on this card"),
            row("time_level.step_entry_ssh_vs_model_operand_replay",
                rebuild_sh2(op, **model), legacy,
                "R59TKE zdfsh2.f90:99-113",
                "BIT here proves the mirror above IS the model's shear "
                "expression, so the one non-bit row is attributable"),
        ]
        if planted_at is not None:
            require(legacy_rows[0]["n_unequal"] == 1,
                    "shear one-ULP plant did not add exactly one unequal "
                    "cell to the time-level row")
            require(float(recorded[planted_at]) != 0.0,
                    "shear plant landed on a cell the shear never produced")
            plant_target = legacy_rows[0]["name"]
        time_level = {
            "plant_index": planted_at,
            "plant_baseline_p_sh2": (
                None if planted_at is None
                else float(recorded[planted_at])),
            "prediction": (
                "routing the step-entry ssh into the shear face metric "
                "makes the production p_sh2 bit-identical to NEMO's"),
            "rows": legacy_rows,
            "legoesm_routing": (
                "ocean_model_latlon_cgrid.py:8255-8258 passes "
                "eta_now=_nemo_ws_zdf_eta_kmm, built at :6112-6115 as "
                "0.5*(state.eta + state_new.eta) = NEMO's stage-3 "
                "Kmm=N+1/2 ssh (correct for tra_zdf's e3w(Kmm)), and "
                ":9059/:9119-9190 feed that same field to the shear's "
                "r3u/r3v"),
            "verdict": (
                "CONFIRMED_SHEAR_FACE_METRIC_TIME_LEVEL"
                if (planted_at is None
                    and all(r["classification"] == "BIT" for r in legacy_rows))
                else "PLANTED" if planted_at is not None else "REFUTED"),
        }
    if plant == "stage-shear-operand-ulp":
        require(plant_target is not None,
                "shear plant did not reach its target")

    first = next((r for r in rows[1:2] if r["classification"] != "BIT"), None)
    return {
        "execution": execution,
        "entry_mode": entry_mode,
        "record_sh2": "round59/oracle_tke_operands_kt00000002.bin",
        "record_operands": "round46/oracle_momstage_kt00000002_s1.bin",
        "compiled_branch": (
            "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:97-109 "
            "(the no-Stokes ELSE branch) and :112-113"),
        "rows": rows,
        "captured_face_metric_rows": captured_metric_rows,
        "operand_identity_rows": operand_rows,
        "one_variable_swap_rows": swap_rows,
        "time_level_discriminator": time_level,
        "first_nonbit": None if first is None else {
            key: first[key] for key in (
                "field", "n_unequal", "absolute_max", "classification",
                "nemo_statement")
        },
        "plant_target": plant_target,
    }


def _tke_surface_operand_rows(
    trace, statement_record: dict, operand_record: dict,
    entry_mode: str, rho0: float,
) -> dict:
    """Attribute the compiled surface write using live production operands."""
    production = trace.tke_statement_trace
    reference_taum = np.asarray(
        operand_record["arrays"]["taum_entry"]).swapaxes(0, 1)
    reference_surface = np.asarray(
        statement_record["arrays"]["en_after_boundaries"]
    ).swapaxes(0, 1)[..., 0]
    candidate_taum = np.asarray(production.taum_surface)
    candidate_surface = np.asarray(production.surface_dirichlet)
    require(
        candidate_taum.shape == candidate_surface.shape
        == reference_taum.shape == reference_surface.shape,
        "TKE surface-operand shapes do not match the 32 x 22 owned domain")

    arrays = operand_record["arrays"]
    zbbrau = np.float64(arrays["rn_ebb"]) / np.float64(rho0)
    replay = np.maximum(
        np.float64(arrays["rn_emin0"]), zbbrau * reference_taum)

    rows = []
    for name, reference, candidate, statement in (
        ("production_taum_vs_recorded", reference_taum, candidate_taum,
         "sbcmod taum input consumed by zdftke.f90:285"),
        ("recorded_operand_scalar_replay", reference_surface, replay,
         "zdftke.f90:257,285"),
        ("production_surface_dirichlet", reference_surface,
         candidate_surface, "zdftke.f90:257,285"),
    ):
        row = _classification(score(
            f"GYRE-zco.kt2.tke_statement.production_step.surface.{name}",
            reference, candidate,
            np.ones(reference.shape, dtype=bool)))
        row.update({
            "field": name,
            "entry_mode": entry_mode,
            "execution": "production step",
            "domain": "all 32 x 22 owned surface cells",
            "nemo_statement": statement,
        })
        rows.append(row)
    return {
        "entry_mode": entry_mode,
        "rho0": float(np.float64(rho0)),
        "rn_ebb": float(np.float64(arrays["rn_ebb"])),
        "rn_emin0": float(np.float64(arrays["rn_emin0"])),
        "rows": rows,
    }


def _bn2_source_replay(state, card, operand_record: dict) -> dict:
    """Rebuild compiled TEOS-10/``bn2`` from the admitted kt=2 entry.

    The round-59 record contains the consumed final ``rn2`` and masks but not
    the local ``pab`` work array.  Reconstruct alpha/beta in the compiled
    TEOS-10 association, then require the complete masked replay to reproduce
    that independently recorded output BIT before exposing any local value.
    """
    from legoesm.ocean.eos import _NEMO_RHO0, _ROQUET_TEOS10

    arrays = operand_record["arrays"]
    z_coord = card.recipe.z_coord
    tke_cfg = card.recipe.model_config.physics.vertical_mixing.tke
    require(getattr(tke_cfg, "n2_eos_form", "seos") == "teos10",
            "round-108 bn2 replay requires the compiled GYRE TEOS-10 arm")

    def yx(name: str) -> np.ndarray:
        value = np.asarray(arrays[name], dtype=np.float64)
        return value.swapaxes(0, 1) if value.ndim >= 2 else value

    # The state bridge installed NEMO's raw step-entry T/S.  Do not apply the
    # model's below-seafloor extrapolation here: ``bn2`` consumes the recorded
    # fields verbatim, and dry signed-zero words are part of the bit contract.
    T = np.asarray(state.T.data, dtype=np.float64)
    S = np.asarray(state.S.data, dtype=np.float64)
    require(T.shape == S.shape and T.ndim == 3,
            f"round-108 T/S entry shape mismatch: {T.shape} vs {S.shape}")
    nlev = T.shape[-1]
    eta = np.asarray(state.eta.data, dtype=np.float64)
    depth = np.asarray(state.H_bathy.data, dtype=np.float64)
    wet_depth = depth > np.float64(0.0)
    reciprocal_depth = np.float64(1.0) / np.where(
        wet_depth, depth, np.float64(1.0))
    r3t = np.where(
        wet_depth, eta * reciprocal_depth, np.float64(0.0))
    stretch = np.float64(1.0) + r3t
    require(np.all(np.isfinite(stretch) & (stretch > 0.0)),
            "recorded-entry bn2 stretch is not finite and positive")

    gdept_0 = np.asarray(z_coord.nemo_gdept_0, dtype=np.float64)[..., :nlev]
    gdepw_0 = np.asarray(z_coord.nemo_gdepw_0, dtype=np.float64)[..., 1:nlev]
    e3w_0 = np.asarray(z_coord.nemo_e3w_0, dtype=np.float64)[..., 1:nlev]
    stretch3 = stretch[..., None]
    gdept = gdept_0 * stretch3
    gdepw = gdepw_0 * stretch3
    e3w = e3w_0 * stretch3

    tmask = yx("tmask")[..., :nlev]
    wmask = yx("wmask")[..., 1:nlev]
    require(tmask.shape == T.shape and wmask.shape == T.shape[:-1] + (nlev - 1,),
            "round-108 NEMO masks do not match the admitted tracer entry")

    require(np.float64(_NEMO_RHO0) == np.float64(
        card.recipe.model_config.constants.rho_0),
        "round-108 TEOS-10 rho0 differs from the instantiated GYRE card")
    coefficients = _ROQUET_TEOS10
    zh = gdept * np.float64(coefficients["r1_Z0"])
    zt = T * np.float64(coefficients["r1_T0"])
    zs = np.sqrt(
        np.abs(S + np.float64(coefficients["rdeltaS"]))
        * np.float64(coefficients["r1_S0"]))

    def horner(prefix: str) -> np.ndarray:
        c = coefficients
        zn3 = np.float64(c[prefix + "003"])
        zn2 = (
            np.float64(c[prefix + "012"]) * zt
            + np.float64(c[prefix + "102"]) * zs
            + np.float64(c[prefix + "002"]))
        zn1 = (
            ((np.float64(c[prefix + "031"]) * zt
              + np.float64(c[prefix + "121"]) * zs
              + np.float64(c[prefix + "021"])) * zt
             + (np.float64(c[prefix + "211"]) * zs
                + np.float64(c[prefix + "111"])) * zs
             + np.float64(c[prefix + "011"])) * zt
            + ((np.float64(c[prefix + "301"]) * zs
                + np.float64(c[prefix + "201"])) * zs
               + np.float64(c[prefix + "101"])) * zs
            + np.float64(c[prefix + "001"]))
        zn0 = (
            ((((np.float64(c[prefix + "050"]) * zt
                + np.float64(c[prefix + "140"]) * zs
                + np.float64(c[prefix + "040"])) * zt
               + (np.float64(c[prefix + "230"]) * zs
                  + np.float64(c[prefix + "130"])) * zs
               + np.float64(c[prefix + "030"])) * zt
              + ((np.float64(c[prefix + "320"]) * zs
                  + np.float64(c[prefix + "220"])) * zs
                 + np.float64(c[prefix + "120"])) * zs
              + np.float64(c[prefix + "020"])) * zt
             + (((np.float64(c[prefix + "410"]) * zs
                  + np.float64(c[prefix + "310"])) * zs
                 + np.float64(c[prefix + "210"])) * zs
                + np.float64(c[prefix + "110"])) * zs
             + np.float64(c[prefix + "010"])) * zt
            + ((((np.float64(c[prefix + "500"]) * zs
                  + np.float64(c[prefix + "400"])) * zs
                 + np.float64(c[prefix + "300"])) * zs
                + np.float64(c[prefix + "200"])) * zs
               + np.float64(c[prefix + "100"])) * zs
            + np.float64(c[prefix + "000"]))
        return ((zn3 * zh + zn2) * zh + zn1) * zh + zn0

    reciprocal_rho0 = np.float64(1.0) / np.float64(_NEMO_RHO0)
    alpha = (horner("ALP") * reciprocal_rho0) * tmask
    beta = ((horner("BET") / zs) * reciprocal_rho0) * tmask

    gd_upper = gdept_0[..., :-1] * stretch3
    gd_lower = gdept_0[..., 1:] * stretch3
    gw = gdepw_0 * stretch3
    zrw = (gw - gd_lower) / (gd_upper - gd_lower)
    one_minus_zrw = np.float64(1.0) - zrw
    zaw = alpha[..., 1:] * one_minus_zrw + alpha[..., :-1] * zrw
    zbw = beta[..., 1:] * one_minus_zrw + beta[..., :-1] * zrw
    temperature_delta = T[..., :-1] - T[..., 1:]
    salinity_delta = S[..., :-1] - S[..., 1:]
    temperature_contribution = zaw * temperature_delta
    salinity_contribution = zbw * salinity_delta
    contribution_difference = (
        temperature_contribution - salinity_contribution)
    gravity_product = (
        np.float64(card.recipe.model_config.constants.g)
        * contribution_difference)
    thickness_division = gravity_product / e3w
    masked_rn2 = thickness_division * wmask
    reference_rn2 = yx("rn2")[..., 1:nlev]

    replay_row = _bitwise_classification(score(
        "GYRE-zco.kt2.bn2.reference_replay.masked_rn2",
        reference_rn2, masked_rn2,
        np.ones(reference_rn2.shape, dtype=bool)))
    require(replay_row["classification"] == "BIT",
            "compiled-order NEMO-from-NEMO bn2 replay moved before scoring")
    recorded_e3w_row = _bitwise_classification(score(
        "GYRE-zco.kt2.bn2.input.e3w",
        yx("e3w_Kmm")[..., 1:nlev], e3w,
        np.ones(e3w.shape, dtype=bool)))
    require(recorded_e3w_row["classification"] == "BIT",
            "source-reconstructed live e3w moved from the round-59 record")

    return {
        "inputs": {
            "T": T,
            "S": S,
            "gdept": gdept,
            "gdepw": gdepw,
            "gdept_0": gdept_0,
            "gdepw_0": gdepw_0,
            "stretch": stretch,
            "e3w": e3w,
            "alpha": alpha,
            "beta": beta,
            "wmask": wmask,
            "gravity": np.asarray(
                card.recipe.model_config.constants.g, dtype=np.float64),
        },
        "references": {
            "zrw": zrw,
            "zaw": zaw,
            "zbw": zbw,
            "temperature_contribution": temperature_contribution,
            "salinity_contribution": salinity_contribution,
            "contribution_difference": contribution_difference,
            "gravity_product": gravity_product,
            "thickness_division": thickness_division,
            "masked_rn2": masked_rn2,
        },
        "recorded_rn2": reference_rn2,
        "reference_replay": replay_row,
        "input_rows": [recorded_e3w_row],
        "source_statements": {
            "alpha_beta": "eosbn2.f90:1257-1310",
            "bn2": "eosbn2.f90:1609-1618",
        },
    }


def _bn2_zrw_propagation_rows(
    replay: dict,
    candidate_zrw: np.ndarray,
    candidate_output: np.ndarray,
) -> dict:
    """Carry one measured ``zrw`` through the remaining compiled statements."""
    inputs = replay["inputs"]
    zrw = np.asarray(candidate_zrw, dtype=np.float64)
    alpha = np.asarray(inputs["alpha"], dtype=np.float64)
    beta = np.asarray(inputs["beta"], dtype=np.float64)
    temperature = np.asarray(inputs["T"], dtype=np.float64)
    salinity = np.asarray(inputs["S"], dtype=np.float64)
    e3w = np.asarray(inputs["e3w"], dtype=np.float64)
    wmask = np.asarray(inputs["wmask"], dtype=np.float64)
    gravity = np.asarray(inputs["gravity"], dtype=np.float64)

    one_minus_zrw = np.float64(1.0) - zrw
    zaw = alpha[..., 1:] * one_minus_zrw + alpha[..., :-1] * zrw
    zbw = beta[..., 1:] * one_minus_zrw + beta[..., :-1] * zrw
    temperature_contribution = (
        zaw * (temperature[..., :-1] - temperature[..., 1:]))
    salinity_contribution = (
        zbw * (salinity[..., :-1] - salinity[..., 1:]))
    propagated = (
        gravity * (temperature_contribution - salinity_contribution)
        / e3w * wmask)
    reference = np.asarray(replay["recorded_rn2"], dtype=np.float64)
    output = np.asarray(candidate_output, dtype=np.float64)
    owned = np.ones(reference.shape, dtype=bool)
    propagated_row = _bitwise_classification(score(
        "GYRE-zco.kt2.bn2.zrw_source_propagation.masked_rn2",
        reference, propagated, owned))
    residual_row = _bitwise_classification(score(
        "GYRE-zco.kt2.bn2.zrw_source_propagation.production_residual",
        propagated, output, owned))
    return {
        "method": (
            "measured zrw plus recorded alpha/beta/T/S/e3w/wmask, replayed "
            "through eosbn2.f90:1613-1618 in NumPy source order"),
        "propagated_output_row": propagated_row,
        "production_residual_row": residual_row,
    }


def _bn2_intermediate_rows(
    trace,
    replay: dict,
    entry_mode: str,
    intermediate: str,
    plant: str | None = None,
    execution: str = "production step JIT",
) -> dict:
    """Score one returned compiled-``bn2`` value and its final output."""
    require(intermediate in BN2_INTERMEDIATES,
            f"unknown bn2 intermediate {intermediate!r}")
    production = trace.tke_statement_trace
    candidate_value = production.bn2_intermediate
    candidate_output = production.bn2_output
    require(candidate_value is not None and candidate_output is not None,
            "production trace did not return the requested bn2 value/output")
    reference = np.asarray(replay["references"][intermediate])
    candidate = np.asarray(candidate_value)
    require(candidate.shape == reference.shape,
            f"production bn2 {intermediate} shape {candidate.shape} != "
            f"source replay {reference.shape}")

    clean = _bitwise_classification(score(
        f"GYRE-zco.kt2.bn2.production_step.{intermediate}.clean",
        reference, candidate, np.ones(reference.shape, dtype=bool)))
    scored_reference = reference
    planted_at = None
    planted_baseline = None
    if plant == "stage-bn2-intermediate-ulp":
        equal = (
            np.ascontiguousarray(reference).view(np.uint64)
            == np.ascontiguousarray(candidate).view(np.uint64)
        ) & np.isfinite(reference) & (reference != 0.0)
        indices = np.argwhere(equal)
        require(indices.size > 0,
                "bn2 intermediate plant found no exact finite nonzero cell")
        planted_at = tuple(int(value) for value in indices[0])
        planted_baseline = float(reference[planted_at])
        scored_reference = reference.copy()
        scored_reference[planted_at] = np.nextafter(
            scored_reference[planted_at], np.float64(np.inf))

    statements = {
        "zrw": "eosbn2.f90:1610-1611",
        "zaw": "eosbn2.f90:1613",
        "zbw": "eosbn2.f90:1614",
        "temperature_contribution": "eosbn2.f90:1616",
        "salinity_contribution": "eosbn2.f90:1617",
        "contribution_difference": "eosbn2.f90:1616-1617",
        "gravity_product": "eosbn2.f90:1616-1617",
        "thickness_division": "eosbn2.f90:1616-1618",
        "masked_rn2": "eosbn2.f90:1616-1618",
    }
    row = _bitwise_classification(score(
        f"GYRE-zco.kt2.bn2.production_step.{intermediate}",
        scored_reference, candidate, np.ones(reference.shape, dtype=bool)))
    row.update({
        "field": intermediate,
        "entry_mode": entry_mode,
        "execution": execution,
        "domain": "all 32 x 22 owned columns, NEMO levels 2:jpkm1",
        "nemo_statement": statements[intermediate],
        "clean_n_unequal": clean["n_unequal"],
        "clean_absolute_max": clean["absolute_max"],
        "plant_index": planted_at,
        "plant_baseline": planted_baseline,
    })
    if planted_at is not None:
        require(row["n_unequal"] == clean["n_unequal"] + 1,
                "bn2 ULP plant did not add exactly one unequal cell")

    recorded_output = np.asarray(replay["recorded_rn2"])
    output = np.asarray(candidate_output)
    require(output.shape == recorded_output.shape,
            "production bn2 output shape differs from the round-59 record")
    output_row = _bitwise_classification(score(
        "GYRE-zco.kt2.bn2.production_step.masked_rn2_output",
        recorded_output, output, np.ones(recorded_output.shape, dtype=bool)))
    output_row.update({
        "field": "masked_rn2_output",
        "entry_mode": entry_mode,
        "execution": execution,
        "domain": "all 32 x 22 owned columns, NEMO levels 2:jpkm1",
        "nemo_statement": "eosbn2.f90:1616-1618",
    })
    propagation = (
        _bn2_zrw_propagation_rows(replay, candidate, output)
        if intermediate == "zrw" else None)
    return {
        "intermediate": intermediate,
        "one_returned_intermediate": True,
        "reference_replay": replay["reference_replay"],
        "input_rows": replay["input_rows"],
        "row": row,
        "output_row": output_row,
        "zrw_downstream_propagation": propagation,
        "plant_target": None if planted_at is None else row["name"],
    }


def _bn2_isolated_rows(
    replay: dict,
    intermediate: str,
    production_value=None,
) -> dict:
    """Score isolated eager/JIT labels without calling either production."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

    require(intermediate in BN2_INTERMEDIATES,
            f"unknown isolated bn2 intermediate {intermediate!r}")
    inputs = replay["inputs"]

    def expression(T, S, gdept, gdepw, e3w, gdept_0, gdepw_0,
                   stretch, alpha, beta, gravity):
        return compute_buoyancy_frequency_nemo_bn2(
            T, S, gdept, gdepw,
            g=gravity, eos_form="seos", e3w_int=e3w,
            e3w_source="mesh_reference", zrw_evaluation="nemo_literal",
            zrw_gdept_0=gdept_0, zrw_gdepw_0=gdepw_0,
            zrw_stretch=stretch, _alpha_beta_override=(alpha, beta),
            _return_intermediate=intermediate)

    operands = tuple(jnp.asarray(inputs[name]) for name in (
        "T", "S", "gdept", "gdepw", "e3w", "gdept_0", "gdepw_0",
        "stretch", "alpha", "beta", "gravity"))
    eager_output, eager_value = jax.device_get(expression(*operands))
    jit_output, jit_value = jax.device_get(jax.jit(expression)(*operands))
    reference = np.asarray(replay["references"][intermediate])
    output_reference = np.asarray(replay["recorded_rn2"])
    rows = []
    output_rows = []
    for label, value, output in (
        ("isolated-closure eager", eager_value, eager_output),
        ("isolated-closure JIT", jit_value, jit_output),
    ):
        row = _bitwise_classification(score(
            "GYRE-zco.kt2.bn2." + label.replace(" ", "_")
            + f".{intermediate}",
            reference, np.asarray(value),
            np.ones(reference.shape, dtype=bool)))
        row.update({
            "field": intermediate,
            "execution": label,
            "one_returned_intermediate": True,
            "nemo_statement": "R101TKEW eosbn2.f90:1609-1618",
        })
        rows.append(row)
        output_row = _bitwise_classification(score(
            "GYRE-zco.kt2.bn2." + label.replace(" ", "_")
            + ".masked_rn2_output",
            output_reference, np.asarray(output),
            np.ones(output_reference.shape, dtype=bool)))
        output_row.update({
            "field": "masked_rn2_output",
            "execution": label,
            "nemo_statement": "R101TKEW eosbn2.f90:1616-1618",
        })
        output_rows.append(output_row)
    production_vs_isolated = None
    if production_value is not None:
        production = np.asarray(production_value)
        isolated = np.asarray(jit_value)
        require(production.shape == isolated.shape,
                "production and isolated bn2 values have different shapes")
        production_vs_isolated = _bitwise_classification(score(
            "GYRE-zco.kt2.bn2.production_vs_isolated_JIT." + intermediate,
            isolated, production, np.ones(isolated.shape, dtype=bool)))
    return {
        "intermediate": intermediate,
        "note": (
            "isolated-closure JIT is not production; only the sibling "
            "production-step row certifies full-step fusion"),
        "rows": rows,
        "output_rows": output_rows,
        "production_vs_isolated_jit": production_vs_isolated,
    }


def _tke_rhs_intermediate_rows(
    trace,
    operand_record: dict,
    statement_record: dict,
    entry_mode: str,
    intermediate: str,
    plant: str | None = None,
    execution: str = "production step JIT",
) -> dict:
    """Score one returned production RHS intermediate against source order."""
    from nemo_testcase_l2_gyre_round103_tke_block_replay import rebuild_block

    require(intermediate in TKE_RHS_SELECTORS,
            f"unknown TKE RHS intermediate {intermediate!r}")
    production = trace.tke_statement_trace
    candidate_value = production.rhs_intermediate
    require(candidate_value is not None,
            "production trace did not return the requested RHS intermediate")

    arrays = operand_record["arrays"]
    langmuir = np.asarray(statement_record["arrays"]["en_after_langmuir"])
    replay = rebuild_block(arrays, langmuir)
    if intermediate == "p_avt_operand":
        reference = np.asarray(arrays["avt_entry"])[..., 1:30].swapaxes(0, 1)
    elif intermediate == "rn2_operand":
        reference = np.asarray(arrays["rn2"])[..., 1:30].swapaxes(0, 1)
    else:
        reference = np.asarray(replay[intermediate]).swapaxes(0, 1)
    candidate = np.asarray(candidate_value)
    require(candidate.shape == reference.shape,
            f"production TKE {intermediate} shape {candidate.shape} != "
            f"oracle replay {reference.shape}")

    recorded_rhs = np.asarray(arrays["rhs_pre_sweep"])[..., 1:30]
    replay_rhs = np.asarray(replay["en_rhs"])
    replay_row = _bitwise_classification(score(
        "GYRE-zco.kt2.tke_rhs_intermediate.reference_replay",
        recorded_rhs, replay_rhs, np.ones(recorded_rhs.shape, dtype=bool)))
    require(replay_row["classification"] == "BIT",
            "NEMO-from-NEMO RHS replay moved before intermediate scoring")

    clean = _bitwise_classification(score(
        f"GYRE-zco.kt2.tke_rhs_intermediate.production_step.{intermediate}.clean",
        reference, candidate, np.ones(reference.shape, dtype=bool)))
    scored_reference = reference
    planted_at = None
    planted_baseline = None
    if plant == "stage-tke-rhs-intermediate-ulp":
        equal = (
            np.ascontiguousarray(reference).view(np.uint64)
            == np.ascontiguousarray(candidate).view(np.uint64)
        ) & np.isfinite(reference) & (reference != 0.0)
        indices = np.argwhere(equal)
        require(indices.size > 0,
                "TKE RHS intermediate plant found no exact nonzero cell")
        planted_at = tuple(int(value) for value in indices[0])
        planted_baseline = float(reference[planted_at])
        scored_reference = reference.copy()
        scored_reference[planted_at] = np.nextafter(
            scored_reference[planted_at], np.float64(np.inf))

    statement = {
        "p_avt_rn2": "zdftke.f90:440",
        "zfact3_dissl": "zdftke.f90:261,441",
        "dissipation_product": "zdftke.f90:441",
        "after_stratification": "zdftke.f90:439-440",
        "parenthesized_sum": "zdftke.f90:439-442",
        "dt_product": "zdftke.f90:439-442",
        "masked_increment": "zdftke.f90:439-442",
        "final_accumulation": "zdftke.f90:439-442",
        "p_avt_operand": "zdftke.f90:440 p_avt input",
        "rn2_operand": "zdftke.f90:440 rn2 input",
    }[intermediate]
    row = _bitwise_classification(score(
        f"GYRE-zco.kt2.tke_rhs_intermediate.production_step.{intermediate}",
        scored_reference, candidate, np.ones(reference.shape, dtype=bool)))
    row.update({
        "field": intermediate,
        "entry_mode": entry_mode,
        "execution": execution,
        "domain": "NEMO levels 2:jpkm1",
        "nemo_statement": statement,
        "clean_n_unequal": clean["n_unequal"],
        "clean_absolute_max": clean["absolute_max"],
        "plant_index": planted_at,
        "plant_baseline": planted_baseline,
    })
    if planted_at is not None:
        require(row["n_unequal"] == clean["n_unequal"] + 1,
                "TKE RHS intermediate ULP plant did not add exactly one "
                "unequal cell")
    downstream_product_replay = None
    if intermediate in TKE_RHS_POSTHOC_OPERANDS:
        recorded_avt = np.asarray(arrays["avt_entry"])[..., 1:30].swapaxes(0, 1)
        recorded_rn2 = np.asarray(arrays["rn2"])[..., 1:30].swapaxes(0, 1)
        candidate_product = (
            candidate * recorded_rn2
            if intermediate == "p_avt_operand"
            else recorded_avt * candidate
        )
        product_reference = np.asarray(replay["p_avt_rn2"]).swapaxes(0, 1)
        downstream_product_replay = _bitwise_classification(score(
            "GYRE-zco.kt2.tke_rhs_intermediate.posthoc_operand_swap."
            f"{intermediate}",
            product_reference,
            candidate_product,
            np.ones(product_reference.shape, dtype=bool),
        ))
        downstream_product_replay.update({
            "field": "p_avt_rn2",
            "execution": "recorded-association one-variable replay",
            "swapped_operand": intermediate,
            "nemo_statement": "zdftke.f90:440",
            "note": (
                "the selected production operand is multiplied by the other "
                "recorded NEMO operand in source order; this replay does not "
                "claim to be the production-JIT multiplication"),
        })
    return {
        "intermediate": intermediate,
        "one_returned_intermediate": True,
        "reference_replay": replay_row,
        "row": row,
        "downstream_product_replay": downstream_product_replay,
        "plant_target": None if planted_at is None else row["name"],
    }


def _tke_rhs_isolated_rows(
    operand_record: dict,
    statement_record: dict,
    intermediate: str = "",
) -> dict:
    """Label isolated eager/JIT separately from the production-step row."""
    import jax
    import jax.numpy as jnp

    arrays = operand_record["arrays"]

    def yx(name):
        return np.asarray(arrays[name]).swapaxes(0, 1)

    levels = slice(1, 30)
    en = jnp.asarray(np.asarray(
        statement_record["arrays"]["en_after_langmuir"]
    ).swapaxes(0, 1)[..., levels])
    sh2 = jnp.asarray(yx("sh2")[..., levels])
    avt = jnp.asarray(yx("avt_entry")[..., levels])
    rn2 = jnp.asarray(yx("rn2")[..., levels])
    dissl = jnp.asarray(yx("dissl_entry")[..., levels])
    wmask = jnp.asarray(yx("wmask")[..., levels])
    reference = yx("rhs_pre_sweep")[..., levels]
    dt = jnp.asarray(arrays["rn_Dt"], dtype=en.dtype)
    ediss = jnp.asarray(arrays["rn_ediss"], dtype=en.dtype)

    def model_expression(en_, sh2_, avt_, rn2_, dissl_, wmask_):
        buoy_source = -avt_ * rn2_
        diss_rate = ediss * dissl_
        return en_ + dt * (
            sh2_ + buoy_source + 0.5 * diss_rate * en_
        ) * wmask_

    def nemo_expression(en_, sh2_, avt_, rn2_, dissl_, wmask_):
        zfact3 = 0.5 * ediss
        return en_ + dt * (
            sh2_ - avt_ * rn2_ + zfact3 * dissl_ * en_
        ) * wmask_

    def nemo_expression_with_intermediate(
        en_, sh2_, avt_, rn2_, dissl_, wmask_,
    ):
        zfact3 = 0.5 * ediss
        p_avt_rn2 = avt_ * rn2_
        zfact3_dissl = zfact3 * dissl_
        dissipation = zfact3_dissl * en_
        stratified = sh2_ - p_avt_rn2
        parenthesized = stratified + dissipation
        scaled = dt * parenthesized
        increment = scaled * wmask_
        final = en_ + increment
        values = {
            "p_avt_operand": avt_,
            "rn2_operand": rn2_,
            "p_avt_rn2": p_avt_rn2,
            "zfact3_dissl": zfact3_dissl,
            "dissipation_product": dissipation,
            "after_stratification": stratified,
            "parenthesized_sum": parenthesized,
            "dt_product": scaled,
            "masked_increment": increment,
            "final_accumulation": final,
        }
        return final, values[intermediate]

    operands = (en, sh2, avt, rn2, dissl, wmask)
    if intermediate:
        require(intermediate in TKE_RHS_SELECTORS,
                f"unknown isolated TKE RHS intermediate {intermediate!r}")
        eager_pair = jax.device_get(
            nemo_expression_with_intermediate(*operands))
        jit_pair = jax.device_get(
            jax.jit(nemo_expression_with_intermediate)(*operands))
        candidates = (
            ("isolated-closure eager NEMO expression", eager_pair[0]),
            ("isolated-closure JIT NEMO expression", jit_pair[0]),
        )
    else:
        candidates = (
            ("isolated-closure eager model expression",
             jax.device_get(model_expression(*operands))),
            ("isolated-closure JIT model expression",
             jax.device_get(jax.jit(model_expression)(*operands))),
            ("isolated-closure eager NEMO expression",
             jax.device_get(nemo_expression(*operands))),
            ("isolated-closure JIT NEMO expression",
             jax.device_get(jax.jit(nemo_expression)(*operands))),
        )
    rows = []
    for label, candidate in candidates:
        row = _bitwise_classification(score(
            "GYRE-zco.kt2.tke_matrix." + label.replace(" ", "_"),
            reference, np.asarray(candidate), np.ones(reference.shape, bool)))
        row.update({
            "field": "en_rhs", "execution": label,
            "nemo_statement": "R101TKEW zdftke.f90:439-442",
        })
        rows.append(row)
    report = {
        "note": (
            "isolated-closure JIT is not production; only the sibling "
            "production-step row certifies full-step fusion"),
        "rows": rows,
    }
    if intermediate:
        from nemo_testcase_l2_gyre_round103_tke_block_replay import rebuild_block

        raw_langmuir = np.asarray(
            statement_record["arrays"]["en_after_langmuir"])
        if intermediate == "p_avt_operand":
            intermediate_reference = yx("avt_entry")[..., levels]
        elif intermediate == "rn2_operand":
            intermediate_reference = yx("rn2")[..., levels]
        else:
            intermediate_reference = np.asarray(
                rebuild_block(arrays, raw_langmuir)[intermediate]
            ).swapaxes(0, 1)
        intermediate_rows = []
        for label, candidate in (
            ("isolated-closure eager NEMO expression", eager_pair[1]),
            ("isolated-closure JIT NEMO expression", jit_pair[1]),
        ):
            row = _bitwise_classification(score(
                "GYRE-zco.kt2.tke_rhs_intermediate."
                + label.replace(" ", "_") + f".{intermediate}",
                intermediate_reference, np.asarray(candidate),
                np.ones(intermediate_reference.shape, dtype=bool)))
            row.update({
                "field": intermediate,
                "execution": label,
                "one_returned_intermediate": True,
                "nemo_statement": "R101TKEW zdftke.f90:261,439-442",
            })
            intermediate_rows.append(row)
        report.update({
            "intermediate": intermediate,
            "intermediate_rows": intermediate_rows,
        })
    return report


def _execute_production_step(
    model, step_state, dt, freshwater, surface_forcing, *, execution_mode: str,
):
    """Run the ordinary production closure, compiled or with JIT disabled."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    require(execution_mode in ("production-jit", "production-eager"),
            f"unknown production execution mode {execution_mode!r}")
    if execution_mode == "production-jit":
        return model.step(
            step_state, dt=dt, freshwater=freshwater,
            surface_forcing=surface_forcing)

    def device_array(value):
        return (jnp.asarray(value)
                if isinstance(value, (np.ndarray, np.generic)) else value)

    step_state = jax.tree_util.tree_map(device_array, step_state)
    freshwater = jax.tree_util.tree_map(device_array, freshwater)
    surface_forcing = jax.tree_util.tree_map(device_array, surface_forcing)
    step_state = model._seed_tke_preclosure_carry(step_state)
    model.prime_step_caches(step_state)
    with jax.disable_jit():
        return LatLonCGridOceanModel._step_jitted.__wrapped__(
            model, step_state, dt, freshwater, surface_forcing)


def _tke_program_twin(
    records: dict, advmean_root: Path, memory_root: Path,
    tke_statement_record: dict, tke_operand_record: dict,
    plant: str | None,
    execution_mode: str = "production-jit",
    rhs_materialization: str = "",
    rhs_intermediate: str = "",
    bn2_intermediate: str = "",
) -> dict:
    """Bounded production-step TKE subwalk split from the compiler-heavy table."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "TKE program twin requires x64")
    require(execution_mode in ("production-jit", "production-eager"),
            f"unknown TKE production execution mode {execution_mode!r}")
    execution_label = (
        "production step JIT" if execution_mode == "production-jit"
        else "production closure eager")
    requested_rhs_materialization = rhs_materialization
    hook_rhs_materialization = (
        "" if rhs_materialization == "baseline" else rhs_materialization)
    require(sum(bool(value) for value in (
                rhs_materialization, rhs_intermediate, bn2_intermediate)) <= 1,
            "TKE RHS and bn2 measurement selectors are mutually exclusive")
    if bn2_intermediate:
        require(bn2_intermediate in BN2_INTERMEDIATES,
                f"unknown bn2 intermediate {bn2_intermediate!r}")

    def production_step(model, step_state, dt, freshwater, surface_forcing):
        return _execute_production_step(
            model, step_state, dt, freshwater, surface_forcing,
            execution_mode=execution_mode)

    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    state = _bridge_kt2_state(card, cfg, records)
    raw_history = tuple(
        jnp.asarray(value) for value in _raw_history_override(memory_root))

    # The stage record is post-zdf_phy.  Replace only closure memory with the
    # independently admitted pre-tke_tke entry; every other kt2 state field
    # stays on the consolidated bridge.
    tke_arrays = tke_operand_record["arrays"]

    def tke_field(name: str, levels=slice(1, 30)):
        return jnp.asarray(
            np.asarray(tke_arrays[name]).swapaxes(0, 1)[..., levels])

    state = state._replace(
        tke=state.tke.replace(data=tke_field("en_entry")),
        tke_avm=state.tke_avm.replace(data=tke_field("avm_entry")),
        tke_avt=state.tke_avt.replace(data=tke_field("avt_entry")),
        tke_dissl=state.tke_dissl.replace(data=tke_field("dissl_entry")),
        tke_avm_surface=state.tke_avm_surface.replace(
            data=tke_field("avm_entry", 0)),
    )
    input_bridge = []
    for field, candidate, reference in (
        ("en_entry", state.tke.data, tke_field("en_entry")),
        ("avm_entry", state.tke_avm.data, tke_field("avm_entry")),
        ("avt_entry", state.tke_avt.data, tke_field("avt_entry")),
        ("dissl_entry", state.tke_dissl.data, tke_field("dissl_entry")),
        ("avm_surface_entry", state.tke_avm_surface.data,
         tke_field("avm_entry", 0)),
    ):
        reference = np.asarray(reference)
        row = _classification(score(
            f"GYRE-zco.kt2.tke_program_input.{field}",
            reference, np.asarray(candidate),
            np.ones(reference.shape, dtype=bool)))
        row.update({
            "field": field,
            "entry_mode": "NEMO_TKE_CLOSURE_MEMORY_RECORDED",
        })
        input_bridge.append(row)
    require(all(row["classification"] == "BIT" for row in input_bridge),
            "TKE pre-closure input bridge is not bit-identical")

    bn2_replay = (
        _bn2_source_replay(state, card, tke_operand_record)
        if bn2_intermediate else None)

    freshwater, surface = _surface_forcings(card, state, 2)
    hooks = _NEMOWSRK3TestHooks(
        expose_live_stage_operands=True,
        tke_rhs_materialization=hook_rhs_materialization,
        tke_rhs_intermediate=rhs_intermediate,
        bn2_intermediate=bn2_intermediate,
        bn2_alpha_beta_override=(
            None if bn2_replay is None else (
                jnp.asarray(bn2_replay["inputs"]["alpha"]),
                jnp.asarray(bn2_replay["inputs"]["beta"]))),
        bn2_tracer_override=(
            None if bn2_replay is None else (
                jnp.asarray(bn2_replay["inputs"]["T"]),
                jnp.asarray(bn2_replay["inputs"]["S"]))),
        barotropic_raw_history_override=raw_history,
        stage_barotropic_output_override=_barotropic_override(
            records, advmean_root, 2),
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks)
    recorded_taum = jnp.asarray(
        np.asarray(tke_arrays["taum_entry"]).swapaxes(0, 1))
    recorded_surface = surface._replace(taum=recorded_taum)
    if bn2_intermediate:
        given_trace = jax.device_get(production_step(
            model, state, card.dt_s, freshwater, recorded_surface))
        chained_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                bn2_intermediate=bn2_intermediate,
                barotropic_raw_history_override=raw_history,
                stage_barotropic_output_override=_barotropic_override(
                    records, advmean_root, 2)))
        chained_trace = jax.device_get(production_step(
            chained_model, state, card.dt_s, freshwater, recorded_surface))
        return {
            "format": "nemo-testcase-l2-gyre-bn2-production-walk-v1",
            "worktree": worktree_stamp(),
            "execution": execution_label,
            "bn2_intermediate": bn2_intermediate,
            "input_bridge": input_bridge,
            "isolated_bn2_discriminator": _bn2_isolated_rows(
                bn2_replay, bn2_intermediate,
                given_trace.tke_statement_trace.bn2_intermediate),
            "given_nemo_entry_bn2_intermediate": _bn2_intermediate_rows(
                given_trace, bn2_replay, "NEMO_TKE_RECORDED",
                bn2_intermediate, plant, execution=execution_label),
            "chained": _bn2_intermediate_rows(
                chained_trace, bn2_replay, "MODEL_BN2_UPSTREAM",
                bn2_intermediate, None, execution=execution_label),
        }
    if requested_rhs_materialization or rhs_intermediate:
        given_trace = jax.device_get(production_step(
            model, state, card.dt_s, freshwater, recorded_surface))
        given = _tke_production_statement_rows(
            given_trace, tke_statement_record, "NEMO_TKE_RECORDED", plant)
        given_matrix = _tke_matrix_statement_rows(
            given_trace, tke_operand_record, tke_statement_record,
            "NEMO_TKE_RECORDED", plant)
        report = {
            "format": "nemo-testcase-l2-gyre-tke-rhs-production-walk-v2",
            "worktree": worktree_stamp(),
            "execution": execution_label,
            "rhs_materialization": requested_rhs_materialization,
            "rhs_intermediate": rhs_intermediate,
            "input_bridge": input_bridge,
            "isolated_rhs_discriminator": _tke_rhs_isolated_rows(
                tke_operand_record, tke_statement_record, rhs_intermediate),
            "given_nemo_entry": given,
            "given_nemo_entry_matrix_walk": given_matrix,
            "chained": None,
        }
        if rhs_intermediate:
            report["given_nemo_entry_rhs_intermediate"] = (
                _tke_rhs_intermediate_rows(
                    given_trace, tke_operand_record, tke_statement_record,
                    "NEMO_TKE_RECORDED", rhs_intermediate, plant,
                    execution=execution_label))
        return report
    model_forcing_trace = jax.device_get(production_step(
        model, state, card.dt_s, freshwater, surface))
    model_forcing = _tke_production_statement_rows(
        model_forcing_trace, tke_statement_record,
        "NEMO_CLOSURE_MEMORY_MODEL_TAUM")
    rho0 = float(card.recipe.model_config.constants.rho_0)
    model_forcing_operands = _tke_surface_operand_rows(
        model_forcing_trace, tke_statement_record, tke_operand_record,
        "NEMO_CLOSURE_MEMORY_MODEL_TAUM", rho0)

    given_trace = jax.device_get(production_step(
        model, state, card.dt_s, freshwater, recorded_surface))
    given = _tke_production_statement_rows(
        given_trace, tke_statement_record, "NEMO_TKE_RECORDED", plant)
    given_operands = _tke_surface_operand_rows(
        given_trace, tke_statement_record, tke_operand_record,
        "NEMO_TKE_RECORDED", rho0)
    given_matrix = _tke_matrix_statement_rows(
        given_trace, tke_operand_record, tke_statement_record,
        "NEMO_TKE_RECORDED", plant)
    # Round 104.  The shear walk's time-level discriminator is a SECOND
    # production step differing in exactly one thing: the free-surface field
    # routed into the shear's face metric (the existing
    # ``legacy_zdf_entry_kmm_eta`` hook makes it the step-entry ssh).  It is
    # a whole-step ablation, so only its p_sh2 row is read.
    legacy_eta_trace = None
    if plant not in ("stage-tke-production-ulp", "stage-tke-matrix-ulp"):
        legacy_eta_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                barotropic_raw_history_override=raw_history,
                stage_barotropic_output_override=_barotropic_override(
                    records, advmean_root, 2),
                legacy_zdf_entry_kmm_eta=True))
        legacy_eta_trace = jax.device_get(production_step(
            legacy_eta_model, state, card.dt_s, freshwater,
            recorded_surface))
    given_shear = _shear_statement_rows(
        given_trace, records, card, state, tke_operand_record,
        "NEMO_TKE_RECORDED", legacy_trace=legacy_eta_trace, plant=plant,
        execution=execution_label)
    literal_counterfactual = None
    if plant not in ("stage-tke-production-ulp", "stage-tke-matrix-ulp",
                     "stage-shear-operand-ulp"):
        vmix = cfg.physics.vertical_mixing
        literal_vmix = vmix._replace(tke=vmix.tke._replace(
            tke_langmuir_evaluation="nemo_literal"))
        literal_cfg = cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=literal_vmix))
        literal_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, literal_cfg,
            _nemo_ws_test_hooks=hooks)
        literal_trace = jax.device_get(production_step(
            literal_model, state, card.dt_s, freshwater,
            recorded_surface))
        literal_counterfactual = _tke_production_statement_rows(
            literal_trace, tke_statement_record,
            "NEMO_TKE_RECORDED_LITERAL_LANGMUIR_COUNTERFACTUAL")
    model_operand_rows = {
        row["field"]: row for row in model_forcing_operands["rows"]}
    given_operand_rows = {
        row["field"]: row for row in given_operands["rows"]}
    surface_attribution = {
        "prediction": (
            "model taum non-BIT; recorded-operand replay BIT; recorded-taum "
            "production surface BIT"),
        "model_forcing": model_forcing_operands,
        "recorded_taum_forcing": given_operands,
        "verdict": (
            "CONFIRMED_INHERITED_MODEL_TAUM"
            if (model_operand_rows["production_taum_vs_recorded"]
                ["classification"] != "BIT"
                and model_operand_rows["recorded_operand_scalar_replay"]
                ["classification"] == "BIT"
                and given_operand_rows["production_taum_vs_recorded"]
                ["classification"] == "BIT"
                and given_operand_rows["production_surface_dirichlet"]
                ["classification"] == "BIT")
            else "REFUTED")
    }
    if plant in ("stage-tke-production-ulp", "stage-tke-matrix-ulp",
                 "stage-shear-operand-ulp", "stage-tke-rhs-ulp"):
        return {
            "format": "nemo-testcase-l2-gyre-tke-production-walk-v7",
            "worktree": worktree_stamp(),
            "rhs_materialization": rhs_materialization,
            "execution": execution_label,
            "input_bridge": input_bridge,
            "model_forcing_diagnostic": model_forcing,
            "surface_operand_attribution": surface_attribution,
            "literal_langmuir_counterfactual": literal_counterfactual,
            "given_nemo_entry": given,
            "given_nemo_entry_matrix_walk": given_matrix,
            "given_nemo_entry_shear_walk": given_shear,
            "chained": None,
        }

    chained_trace = None
    state = card.recipe.initial_state
    for kt in (1, 2):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True))
        state = model._seed_tke_preclosure_carry(state)
        freshwater, surface = _surface_forcings(card, state, kt)
        trace = jax.device_get(production_step(
            model, state, card.dt_s, freshwater, surface))
        if kt == 2:
            chained_trace = trace
        state = trace.state_after
    require(chained_trace is not None,
            "chained TKE production run did not expose kt2")
    return {
        "format": "nemo-testcase-l2-gyre-tke-production-walk-v7",
        "worktree": worktree_stamp(),
        "rhs_materialization": rhs_materialization,
        "execution": execution_label,
        "input_bridge": input_bridge,
        "model_forcing_diagnostic": model_forcing,
        "surface_operand_attribution": surface_attribution,
        "literal_langmuir_counterfactual": literal_counterfactual,
        "given_nemo_entry": given,
        "given_nemo_entry_matrix_walk": given_matrix,
        "given_nemo_entry_shear_walk": given_shear,
        "chained": _tke_production_statement_rows(
            chained_trace, tke_statement_record, "LEGO_CHAINED"),
    }


def _stage1_handoff_pair(trace, boundary: str):
    """Select exactly one stage-1 momentum pair from the live step trace."""
    require(boundary in STAGE1_HANDOFF_BOUNDARIES,
            f"unknown stage-1 handoff boundary {boundary!r}")
    pairs = {
        "full_accumulator": trace.stage1_full_rhs,
        "projected_accumulator": trace.stage1_rhs_walk[0],
        "post_transport": trace.stage1_rhs_walk[1],
        "post_w_zad": trace.stage1_rhs_walk[2],
        "rk_input": trace.stage1_rhs_walk[3],
        "raw_rk": trace.stage_raw_velocities[0],
        "corrected_output": trace.stage_outputs[0][:2],
    }
    return pairs[boundary]


def _score_stage1_handoff_pair(
    pair, arrays: dict, masks: dict, boundary: str, execution: str,
    *, plant: bool = False,
) -> dict:
    """Bit-score one selected U/V handoff; a plant moves one U reference."""
    reference_boundary = (
        "post_update" if boundary == "raw_rk"
        else "post_baro" if boundary == "corrected_output"
        else "after_adv")
    rows = []
    plant_target = None
    for face, full_candidate in zip(("u", "v"), pair, strict=True):
        candidate = np.asarray(full_candidate)
        candidate = (candidate[:, 1:, :] if face == "u"
                     else candidate[1:, :, :])
        reference = _owned3(arrays[f"{reference_boundary}_{face}"]).copy()
        active = masks[face]
        clean = _bitwise_classification(score(
            f"GYRE-zco.kt1.s1.handoff.{boundary}.{face}.{execution}",
            reference, candidate, active))
        scored_reference = reference
        planted_at = None
        if plant and face == "u":
            equal = (
                np.ascontiguousarray(candidate).view(np.uint64)
                == np.ascontiguousarray(reference).view(np.uint64)
            ) & active & np.isfinite(reference) & (reference != 0.0)
            indices = np.argwhere(equal)
            require(indices.size > 0,
                    "stage-1 handoff plant found no exact finite nonzero U cell")
            planted_at = tuple(int(value) for value in indices[0])
            scored_reference = reference.copy()
            scored_reference[planted_at] = np.nextafter(
                scored_reference[planted_at], np.float64(np.inf))
        row = _bitwise_classification(score(
            f"GYRE-zco.kt1.s1.handoff.{boundary}.{face}.{execution}",
            scored_reference, candidate, active))
        row.update({
            "kt": 1,
            "stage": 1,
            "face": face,
            "boundary": boundary,
            "execution": execution,
            "reference_boundary": reference_boundary,
            "clean_n_unequal": clean["n_unequal"],
            "clean_absolute_max": clean["absolute_max"],
            "plant_index": planted_at,
        })
        rows.append(row)
        if planted_at is not None:
            require(row["n_unequal"] == clean["n_unequal"] + 1,
                    "stage-1 handoff one-ULP plant did not add exactly one "
                    "unequal cell")
            plant_target = row["name"]
    require(not plant or plant_target is not None,
            "stage-1 handoff plant did not reach its production target")
    return {"rows": rows, "plant_target": plant_target, "plant": plant}


def _stage1_isolated_handoff(records: dict, card, masks: dict,
                             advmean_root: Path, boundary: str) -> dict:
    """JIT the shared RK write/correction with NEMO's recorded operands."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_common import (
        nemo_reference_depth_reciprocal,
        rk3_stage_barotropic_correction,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        rk3_stage_velocity_update,
    )

    require(boundary in ("raw_rk", "corrected_output"),
            "isolated JIT is defined only for the RK write and correction")
    a = records[(1, 1)]["arrays"]
    external = _barotropic_override(records, advmean_root, 1)
    external_targets = {"u": external[1], "v": external[2]}

    def operands(face: str):
        widen = _u_full if face == "u" else _v_full
        before = widen(_owned3(a[f"{face}_Kbb"]))
        rhs = widen(_owned3(a[f"after_adv_{face}"]))
        mask = widen(_owned3(a[f"{face}mask"]))
        thickness = widen(_owned3(a[f"e3{face}_0"]))
        # The round-46 stage writer enters before the external solve, so its
        # Kaa barotropic slot is still zero.  The compiled stage consumes the
        # admitted post-external frame used by the production override.
        target = external_targets[face]
        mask2d = np.max(mask, axis=-1)
        return tuple(jnp.asarray(value) for value in (
            before, rhs, mask, thickness, target, mask2d))

    def isolated(before, rhs, mask, thickness, target, mask2d):
        raw = rk3_stage_velocity_update(
            before, rhs, np.float64(card.dt_s / 3.0), mask,
            vector_form=True)
        reciprocal = nemo_reference_depth_reciprocal(
            jnp.sum(thickness, axis=-1), mask2d)
        corrected = rk3_stage_barotropic_correction(
            raw, target, thickness, reciprocal, mask)
        return raw, corrected

    compiled = jax.jit(isolated)
    selected = []
    for face in ("u", "v"):
        raw, corrected = jax.device_get(compiled(*operands(face)))
        selected.append(raw if boundary == "raw_rk" else corrected)
    return _score_stage1_handoff_pair(
        tuple(selected), a, masks, boundary, "isolated-closure_JIT")


def _stage1_handoff_walk(
    records: dict, advmean_root: Path, boundary: str,
    execution_mode: str, plant: str | None,
) -> dict:
    """Focused kt=1 stage-1 handoff walk through the production closure."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64),
            "stage-1 handoff walk requires x64")
    require(boundary in STAGE1_HANDOFF_BOUNDARIES,
            f"unknown stage-1 handoff boundary {boundary!r}")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    source_disposition = {
        "momentum_advection": cfg.momentum_advection,
        "momentum_time_integrator": cfg.momentum_time_integrator,
        "adaptive_implicit_vertadv": cfg.adaptive_implicit_vertadv,
        "compiled_branch": "ln_dynadv_vec stage-1 Krhs carried from stp_2D",
        "transport_w_momentum_disposition": (
            "tracer inputs; no stage-1 vector momentum call"),
        "citations": [
            "R98WWALK stp2d.f90:141-176,202-213",
            "R98WWALK stprk3.f90:188-202",
            "R98WWALK stprk3_stg.f90:326-347,363-374,661-675,731-760",
        ],
    }
    if execution_mode == "isolated-jit":
        isolated = _stage1_isolated_handoff(
            records, card, masks, advmean_root, boundary)
        return {
            "format": "nemo-testcase-l2-gyre-stage1-handoff-v1",
            "worktree": worktree_stamp(),
            "selected_boundary": boundary,
            "execution": "isolated-closure JIT",
            "source_disposition": source_disposition,
            **isolated,
        }

    require(execution_mode in ("production-jit", "production-eager"),
            f"unknown handoff execution mode {execution_mode!r}")
    require(not plant or (
        plant == "stage1-handoff-output-ulp"
        and execution_mode == "production-jit"),
        "stage-1 handoff plant requires a production-jit boundary")
    state = _bridge_stage_context(
        card.recipe.initial_state, records[(1, 1)]["arrays"])
    freshwater, surface = _surface_forcings(card, state, 1)
    hooks = _NEMOWSRK3TestHooks(
        expose_live_stage_operands=True,
        stage_barotropic_output_override=_barotropic_override(
            records, advmean_root, 1),
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks)
    trace = jax.device_get(_execute_production_step(
        model, state, card.dt_s, freshwater, surface,
        execution_mode=execution_mode))
    execution = (
        "production_step_JIT" if execution_mode == "production-jit"
        else "production_closure_eager")
    scored = _score_stage1_handoff_pair(
        _stage1_handoff_pair(trace, boundary),
        records[(1, 1)]["arrays"], masks, boundary, execution,
        plant=plant == "stage1-handoff-output-ulp")
    return {
        "format": "nemo-testcase-l2-gyre-stage1-handoff-v1",
        "worktree": worktree_stamp(),
        "selected_boundary": boundary,
        "execution": execution.replace("_", " "),
        "source_disposition": source_disposition,
        **scored,
    }


def _stage_twin(records: dict, stage_root: Path, advmean_root: Path,
                memory_root: Path, btstep_root: Path,
                stage_closure_root: Path, stage1_w_root: Path,
                stage1_r3_root: Path,
                plant: str | None, *, walk_only: bool = False,
                production_tke_only: bool = False,
                production_tke_entry_ulp: tuple[str, tuple[int, ...]] | None = None,
                production_tke_taum=None,
                production_tke_post_sweep=None,
                production_entry_root: Path = YEAR_ENTRY_ROOT,
                skip_stage1_operator_replay: bool = False,
                ) -> dict:
    """Decision-41 stage tables from recorded entries and the shared stage."""
    direct_w_record = read_admitted_stage1_w_walk(
        stage1_w_root, plant_stamp=plant == "stage-w-record-stamp")
    direct_r3_record = read_admitted_stage1_r3_operands(
        stage1_r3_root, plant_stamp=plant == "stage-r3-record-stamp")
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "stage twin requires x64")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)

    if production_tke_only:
        # Round 93+ already established this recorded-entry stage twin.  Keep
        # the focused K_H measurement on that same driver, but leave the live
        # operand observer disabled: ``step`` then returns _step_jitted's
        # ordinary production state, rather than an observer-shaped trace.
        require(plant is None, "production TKE-only mode does not accept a legacy plant")
        state, entry_audit = _bridge_kt2_production_entry(
            card, cfg, records, stage_root, memory_root,
            Path(production_entry_root))
        entry_plant = None
        if production_tke_entry_ulp is not None:
            field_name, index = production_tke_entry_ulp
            require(field_name in {"tke", "tke_avm", "tke_avt", "tke_dissl"},
                    f"unsupported production TKE entry plant {field_name!r}")
            field = getattr(state, field_name)
            data = np.array(field.data, dtype=np.float64, copy=True)
            require(len(index) == data.ndim
                    and all(0 <= value < size
                            for value, size in zip(index, data.shape)),
                    f"production TKE entry plant index {index} is out of bounds")
            before = np.float64(data[index])
            after = np.nextafter(before, np.float64(np.inf))
            require(before.view(np.uint64) != after.view(np.uint64),
                    "production TKE entry plant did not move one binary64 ULP")
            data[index] = after
            state = state._replace(**{
                field_name: field.replace(data=jnp.asarray(data)),
            })
            entry_plant = {
                "field": field_name,
                "index": list(index),
                "before": float(before),
                "after": float(after),
                "changed_entry_cells": 1,
            }
        # This is kt=2 by construction: the full before-level state above and
        # this forcing call have the same step index as the round-59 record.
        freshwater, surface = _surface_forcings(card, state, 2)
        require(production_tke_taum is not None,
                "recorded-entry production TKE mode requires kt=2 taum")
        recorded_taum = jnp.asarray(production_tke_taum)
        require(recorded_taum.shape == state.eta.data.shape,
                "recorded kt=2 taum shape does not match the entry state")
        require(bool(jnp.all(jnp.isfinite(recorded_taum))),
                "recorded kt=2 taum contains NaN/Inf")
        surface = surface._replace(taum=recorded_taum)
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=_barotropic_override(
                records, advmean_root, 2),
        )
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=hooks)
        entry_n2 = model._tke_step_entry_n2_bundle(state)
        entry_sh2 = model._tke_step_entry_p_sh2(state)
        require(entry_n2 is not None and entry_sh2 is not None,
                "kt=2 production step did not materialize TKE entry operands")
        injection = None
        solve_calls = []
        real_tke_solve = None
        if production_tke_post_sweep is not None:
            import legoesm.ocean.physics.vertical_mixing.tke as tke_module

            injected_array = np.asarray(
                production_tke_post_sweep, dtype=np.float64)
            require(injected_array.shape == state.tke.data.shape,
                    "NEMO en_post_sweep injection shape does not match the "
                    "production TKE solver output slot")
            require(np.all(np.isfinite(injected_array)),
                    "NEMO en_post_sweep injection contains NaN/Inf")
            injected_energy = jnp.asarray(injected_array)
            real_tke_solve = tke_module._solve_tke_backward_euler

            def inject_recorded_post_sweep(*args, **kwargs):
                solved = real_tke_solve(*args, **kwargs)
                require(solved.shape == injected_energy.shape,
                        "production TKE solver output shape moved before "
                        "the NEMO en_post_sweep injection")
                solve_calls.append(1)
                return jnp.asarray(injected_energy, dtype=solved.dtype)

            tke_module._solve_tke_backward_euler = inject_recorded_post_sweep
            injection = {
                "source": "NEMO recorded en_post_sweep",
                "target": (
                    "_solve_tke_backward_euler return consumed by final "
                    "compute_mixing_lengths/compute_K_from_tke"),
            }
        # The intervention changes a Python global read while JAX traces the
        # production step.  Clear staging/executable caches so the baseline,
        # injection, and one-ULP control each trace the function installed for
        # that arm instead of reusing another arm's executable.
        jax.clear_caches()
        try:
            state_after = jax.device_get(model.step(
                    state, dt=card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
        finally:
            if real_tke_solve is not None:
                tke_module._solve_tke_backward_euler = real_tke_solve
        if injection is not None:
            require(len(solve_calls) == 1,
                    "NEMO en_post_sweep injection did not intercept exactly "
                    f"one production TKE solve: {len(solve_calls)}")
            injection["solve_call_count"] = len(solve_calls)
            injection["injection_citation"] = (
                "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:3175-3196")
        require(getattr(state_after, "tke_avt", None) is not None,
                "production _step_jitted result has no tke_avt K_H carry")
        return {
            "format": "nemo-testcase-l2-gyre-stage-twin-production-tke-v3",
            "worktree": worktree_stamp(),
            "label": "recorded-entry production step (_step_jitted)",
            "kt": 2,
            "stage": "pre-zdf_phy closure feeding WS-RK3 stage 1",
            "entry": (
                "NEMO kt=2 before-level entry; Kbb=Kmm=3; forcing kt=2"),
            "entry_record": entry_audit["year_owner_entry"],
            "entry_slots": entry_audit["slots"],
            "entry_state_identity": entry_audit,
            "forcing_kt": 2,
            "forcing_entry": {"taum": np.asarray(recorded_taum)},
            "derived_entry_operands": {
                "rn2": np.asarray(entry_n2.rn2),
                "rn2b": np.asarray(entry_n2.rn2b),
                "sh2": np.asarray(entry_sh2),
                "e3t_Kmm": np.asarray(entry_n2.e3t_Kmm),
                "e3w_Kmm": np.asarray(entry_n2.e3w_Kmm),
            },
            "stage_barotropic_handoff_kt": 2,
            "bridge_citation": (
                "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1198-1385"),
            "existing_stage_twin_citation": (
                "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:3329-3359"),
            "output_carry": "LatLonCGridOceanState.tke_avt",
            "extraction_citation": (
                "ocean_model_latlon_cgrid.py:10822-10827"),
            "step_citation": "ocean_model_latlon_cgrid.py:11012-11025",
            "carry_declaration_citation": "state.py:577-581",
            "entry_plant": entry_plant,
            "post_sweep_injection": injection,
            # Private in-process payload consumed by round 54.  This focused
            # mode is not serialized by the round-46 CLI.
            "entry_carry": {
                name: np.asarray(getattr(state, name).data)
                for name in (
                    "tke", "tke_avm", "tke_avt", "tke_dissl",
                    "tke_avm_surface")
            },
            "candidate_k_h": np.asarray(state_after.tke_avt.data),
            "candidate_tke_post_sweep": np.asarray(state_after.tke.data),
            "candidate_k_m": np.asarray(state_after.tke_avm.data),
            "candidate_dissl": np.asarray(state_after.tke_dissl.data),
        }

    next_entries = {
        kt: read_entry(stage_root / f"oracle_step_entry_kt{kt:08d}.bin")
        for kt in (2, 3)
    }
    from nemo_testcase_l2_gyre_round71_fct_stage2_gate import (
        read_record as read_tracer_stage)

    transports = {}
    for kt in (1, 2):
        for stage in (1, 2):
            path = advmean_root / (
                f"oracle_rktracer_operands_kt{kt:08d}_s{stage}.bin")
            record = read_tracer_stage(path, stage, expected_kt=kt)["fields"]
            transports[(kt, stage)] = {
                field: np.ascontiguousarray(record[field].swapaxes(0, 1))
                for field in ("zFu", "zFv", "zFw")
            }
    for kt, transport_root in ((1, advmean_root), (2, stage_closure_root)):
        path = transport_root / f"oracle_tracer_transport_kt{kt:08d}_s3.bin"
        header = records[(kt, 3)]["header"]
        transports[(kt, 3)] = read_transport(
            path, 3, expected_kt=kt,
            expected_slots=tuple(
                header[name] for name in ("Kbb", "Kmm", "Kaa", "Krhs")))

    direct_stage_ww = {(1, 1): read_stage_ww(
        STAGE_WW_ROOT / "oracle_rkstage_ww_kt00000001_s1.bin", 1)}
    direct_stage_state = {(1, 1): read_phase3_stage(
        SCALAR_MATH_ROOT / "oracle_stage_kt00000001_s1.bin", 1)}
    stage1_w_walk = _stage1_w_walk(
        records, transports, direct_stage_ww, direct_stage_state,
        direct_w_record, direct_r3_record, advmean_root, plant)
    if walk_only or plant in {
        "stage-w-transport-ulp", "stage-w-carry-ulp",
        "stage-w-direct-r3-ulp",
    }:
        return {
            "format": "nemo-testcase-l2-gyre-stage-twin-v4",
            "worktree": worktree_stamp(),
            "given_nemo_entry": [], "chained": [],
            "stage_entry_identity": [], "first_owned_nonbit": None,
            "stage1_w_walk": stage1_w_walk,
        }

    if plant == "stage-rhs-ulp":
        planted_rows, _ = _given_inputs(
            records, "stage-rhs-ulp", kt=1, stages=(1,))
        planted = next(
            row for row in planted_rows
            if row["name"] == "GYRE-zco.kt1.s1.post_hpg_accumulator.u")
        return {
            "format": "nemo-testcase-l2-gyre-stage-twin-v3",
            "worktree": worktree_stamp(),
            "given_nemo_entry": [], "chained": [],
            "stage_entry_identity": [], "first_owned_nonbit": None,
            "stage_rhs_ulp_plant_flipped_row": planted["n_unequal"] == 1,
            "stage_rhs_ulp_plant_target": planted["name"],
        }

    given = []
    given_entries = []
    given_traces = {}
    states = {1: _bridge_stage_context(
                  card.recipe.initial_state, records[(1, 1)]["arrays"],
                  plant=plant == "stage-context-ulp"),
              2: _bridge_kt2_state(card, cfg, records)}
    raw_history = tuple(jnp.asarray(value) for value in _raw_history_override(memory_root))
    kt_values = ((1,) if plant in {
        "stage-entry-ulp", "stage-context-ulp", "stage-assignment-output-ulp"
    } else (1, 2))
    stage_values = ((2,) if plant == "stage-entry-ulp" else
                    (1,) if plant in {
                        "stage-context-ulp", "stage-assignment-output-ulp"
                    } else (1, 2, 3))
    for kt in kt_values:
        state = states[kt]
        freshwater, surface = _surface_forcings(card, state, kt)
        for stage in stage_values:
            entry_override = None if stage == 1 else _entry_override(
                records, kt, stage,
                plant=(plant == "stage-entry-ulp" and kt == 1 and stage == 2))
            hooks = _NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                barotropic_raw_history_override=(raw_history if kt == 2 else None),
                stage_barotropic_output_override=_barotropic_override(
                    records, advmean_root, kt),
                stage_entry_override=entry_override,
            )
            trace = jax.device_get(LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, cfg,
                _nemo_ws_test_hooks=hooks).step(
                    state, dt=card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
            given_traces[(kt, stage)] = trace
            entry_rows = _entry_rows(
                trace, records, masks, state, kt, stage, "NEMO_RECORDED",
                plant_rhs=False)
            given_entries.extend(entry_rows)
            if plant == "stage-entry-ulp" and (kt, stage) == (1, 2):
                planted = next(row for row in entry_rows
                               if kt == 1 and stage == 2
                               and row["field"] == "T")
                require(planted["n_unequal"] == 1,
                        "one-ULP stage-entry plant did not flip exactly one entry row cell")
                return {
                    "format": "nemo-testcase-l2-gyre-stage-twin-v1",
                    "worktree": worktree_stamp(),
                    "given_nemo_entry": [], "chained": [],
                    "stage_entry_identity": entry_rows,
                    "first_owned_nonbit": None,
                    "stage_entry_ulp_plant_flipped_row": True,
                    "stage_entry_ulp_plant_changed_output": False,
                }
            if plant == "stage-context-ulp" and (kt, stage) == (1, 1):
                planted = next(row for row in entry_rows
                               if row["field"] == "tke_en")
                require(planted["n_unequal"] == 1,
                        "one-ULP stage-context plant did not flip exactly one row cell")
                return {
                    "format": "nemo-testcase-l2-gyre-stage-twin-v2",
                    "worktree": worktree_stamp(),
                    "given_nemo_entry": [], "chained": [],
                    "stage_entry_identity": entry_rows,
                    "first_owned_nonbit": None,
                    "stage_context_ulp_plant_flipped_row": True,
                }
            rows = _output_rows(
                trace, records, next_entries, transports, masks,
                np.asarray(card.recipe.grid.area_T), state, kt,
                "NEMO_RECORDED", direct_stage_ww)
            given.extend(row for row in rows if row.get("stage") == stage)

    assignment_discriminator = _assignment_execution_discriminator(
        given_traces, records, plant)
    if plant == "stage-assignment-output-ulp":
        return {
            "format": "nemo-testcase-l2-gyre-stage-twin-v5",
            "worktree": worktree_stamp(),
            "given_nemo_entry": [], "chained": [],
            "stage_entry_identity": [], "first_owned_nonbit": None,
            "stage1_w_walk": stage1_w_walk,
            "assignment_execution_discriminator": assignment_discriminator,
        }

    for kt in (1, 2):
        state = states[kt]
        freshwater, surface = _surface_forcings(card, state, kt)
        hooks = _NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_raw_history_override=(raw_history if kt == 2 else None),
        )
        external = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=hooks).step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
        external_outputs = (
            external.state_after_barotropic.eta.data,
            external.state_after_barotropic.uu_b.data,
            external.state_after_barotropic.vv_b.data,
            external.transport_average[0], external.transport_average[1],
        )
        given.extend(_external_rows(
            external_outputs, external.state_after_barotropic.bt_hist,
            records, advmean_root, memory_root, btstep_root, masks, kt,
            "NEMO_RECORDED"))

    chained = []
    state = card.recipe.initial_state
    for kt in (1, 2):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True))
        # Mirror the public step shim before observing the stage-entry carry.
        # A cold start seeds avm/avt/surface-avm/dissl; scoring the caller's
        # pre-shim object would falsely report those consumed fields absent.
        state = model._seed_tke_preclosure_carry(state)
        freshwater, surface = _surface_forcings(card, state, kt)
        trace = jax.device_get(model.step(
                    state, dt=card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
        chained.extend(_external_rows(
            (trace.barotropic_targets[4], trace.barotropic_targets[0],
             trace.barotropic_targets[1], trace.barotropic_targets[2],
             trace.barotropic_targets[3]),
            trace.state_after.bt_hist, records, advmean_root, memory_root,
            btstep_root, masks, kt, "LEGO_CHAINED"))
        chained.extend(_output_rows(
            trace, records, next_entries, transports, masks,
            np.asarray(card.recipe.grid.area_T), state, kt,
            "LEGO_CHAINED", direct_stage_ww))
        state = trace.state_after

    measured = [row for row in given
                if row.get("classification") != "UNMEASURED_WITH_SPEC"]
    first = next((row for row in measured
                  if row.get("classification") != "BIT"), None)
    if skip_stage1_operator_replay:
        stage1_operator_rows, stage1_operator_first = [], {"stage1": None}
    else:
        stage1_operator_rows, stage1_operator_first = _given_inputs(
            records, None, kt=1, stages=(1,))
    stage1_rhs_rows = [
        row for row in given_entries
        if row.get("field", "").startswith("stage1_rhs_")
    ]
    first_rhs = next(
        (row for row in stage1_rhs_rows
         if row.get("classification") != "BIT"), None)
    return {
        "format": "nemo-testcase-l2-gyre-stage-twin-v5",
        "worktree": worktree_stamp(),
        "given_nemo_entry": given,
        "chained": chained,
        "stage_entry_identity": given_entries,
        "stage1_operator_walk": {
            "replay_omitted": skip_stage1_operator_replay,
            "omission_reason": (
                "post-table replay exceeds the per-process LLVM compiler "
                "limit; stage outputs and the separately committed operator "
                "walk remain measured"
                if skip_stage1_operator_replay else None),
            "compiled_order": ("hpg", "ldf", "vor", "adv"),
            "rows": stage1_operator_rows,
            "first_nonbit": stage1_operator_first["stage1"],
            "post_operator_rhs_walk": stage1_rhs_rows,
            "first_nonbit_model_statement": (
                None if first_rhs is None else {
                    key: first_rhs[key] for key in (
                        "boundary", "statement", "field", "n_unequal",
                        "absolute_max", "classification")
                }
            ),
        },
        "stage1_w_walk": stage1_w_walk,
        "assignment_execution_discriminator": assignment_discriminator,
        "first_owned_nonbit": None if first is None else {
            key: first[key] for key in (
                "kt", "stage", "field", "n_unequal", "absolute_max",
                "classification")},
        "stage_entry_ulp_plant_flipped_row": None,
        "stage_entry_ulp_plant_changed_output": None,
    }


def _exact_payload_pairs(left: dict, right: dict, pairs: dict[str, str], *,
                         right_is_full: bool) -> dict:
    """Bit-test the duplicate kt=2 payloads written by independent streams."""
    rows = {}
    for left_name, right_name in pairs.items():
        a = np.asarray(left[left_name])
        b = np.asarray(right[right_name])
        if right_is_full:
            b = _owned2(b) if a.ndim == 2 else _owned3(b, a.shape[-1])
        rows[f"{left_name}={right_name}"] = bool(
            a.shape == b.shape
            and np.array_equal(
                np.ascontiguousarray(a).view(np.uint64),
                np.ascontiguousarray(b).view(np.uint64),
            )
        )
    return {"pair_count": len(rows), "all_bit_identical": all(rows.values()),
            "pairs": rows}


def run(
    root: Path,
    *,
    expect_commit: str,
    mode: str,
    plant: str | None,
    round40_kt1: Path,
    round41_kt1: Path,
    advmean_root: Path = ADVMEAN_ROOT,
    memory_root: Path = MEMORY_ROOT,
    btstep_root: Path = BTSTEP_ROOT,
    stage_closure_root: Path = STAGE_CLOSURE_ROOT,
    stage1_w_root: Path = STAGE1_W_ROOT,
    stage1_r3_root: Path = STAGE1_R3_ROOT,
    tke_statement_root: Path = TKE_STATEMENT_ROOT,
    tke_operand_record: Path = TKE_OPERAND_RECORD,
    execution_mode: str = "production-jit",
    stage1_handoff_boundary: str = "",
    tke_rhs_materialization: str = "",
    tke_rhs_intermediate: str = "",
    bn2_intermediate: str = "",
    skip_stage1_operator_replay: bool = False,
) -> dict:
    stamp = worktree_stamp()
    require(not skip_stage1_operator_replay or mode == "stage-twin",
            "--skip-stage1-operator-replay is valid only in stage-twin mode")
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(
        len(expected) == 40 and stamp["commit"].lower() == expected,
        f"commit stamp mismatch: {stamp['commit']} != {expected}",
    )
    records = {}
    hashes = {}
    for kt, stage in STAGES:
        path = root / f"oracle_momstage_kt{kt:08d}_s{stage}.bin"
        p = (plant if (kt, stage) == (2, 1)
             and plant in {"header", "slot", "truncation"} else None)
        records[(kt, stage)] = read_stage(
            path, expected_kt=kt, expected_stage=stage, plant=p)
        hashes[path.name] = sha256(path)
    calibration = _calibrate(records, plant)
    new40 = root / "oracle_rkstage3_terms_kt00000001.bin"
    new41 = root / "oracle_dynadv_split_kt00000001_s3.bin"
    legacy_records = {
        "round40_kt1": read_stage3_terms(new40, expect_kt=1)["header"],
        "round40_kt2": read_stage3_terms(
            root / "oracle_rkstage3_terms_kt00000002.bin", expect_kt=2
        )["header"],
        "round41_kt1": read_split(new41, expect_kt=1)["header"],
        "round41_kt2": read_split(root / "oracle_dynadv_split_kt00000002_s3.bin", expect_kt=2)[
            "header"
        ],
    }
    twin_reports = {
        "round40": _compare_self_describing(
            round40_kt1, new40, [True], max_listed=64),
        "round41": _compare_self_describing(
            round41_kt1, new41, [True], max_listed=64),
    }
    twin = {key: bool(value["consumed_equal"])
            for key, value in twin_reports.items()}
    raw_twin = {
        "round40": new40.read_bytes() == round40_kt1.read_bytes(),
        "round41": new41.read_bytes() == round41_kt1.read_bytes(),
    }
    if plant == "twin":
        twin["round40"] = False
    require(all(twin.values()), f"consumed kt1 legacy twin moved: {twin}")

    stage3 = records[(2, 3)]["arrays"]
    kt2_payload_identity = {
        "rkstage3_terms": _exact_payload_pairs(
            read_stage3_terms(root / "oracle_rkstage3_terms_kt00000002.bin",
                              expect_kt=2)["arrays"],
            stage3,
            {
                "before_u": "rhs_entry_u", "before_v": "rhs_entry_v",
                "after_hpg_u": "after_hpg_u", "after_hpg_v": "after_hpg_v",
                "after_vor_u": "after_vor_u", "after_vor_v": "after_vor_v",
                "after_adv_u": "after_adv_u", "after_adv_v": "after_adv_v",
                "uu_Kmm": "u_Kmm", "vv_Kmm": "v_Kmm", "ww": "ww",
                "r3t_Kmm": "r3t_Kmm", "r3u_Kmm": "r3u_Kmm",
                "r3v_Kmm": "r3v_Kmm", "e3u_Kmm": "e3u_Kmm",
                "e3v_Kmm": "e3v_Kmm", "e3t_Kmm": "e3t_Kmm",
                "e3w_Kmm": "e3w_Kmm", "e3u_0": "e3u_0",
                "e3v_0": "e3v_0", "e3t_0": "e3t_0",
                "umask": "umask", "vmask": "vmask", "tmask": "tmask",
                "wmask": "wmask", "e1e2t": "e1e2t",
                "r1_e1u": "r1_e1u", "r1_e2v": "r1_e2v",
            },
            right_is_full=True,
        ),
        "dynadv_split": _exact_payload_pairs(
            read_split(root / "oracle_dynadv_split_kt00000002_s3.bin",
                       expect_kt=2)["arrays"],
            stage3,
            {
                "before_keg_u": "after_vor_u", "before_keg_v": "after_vor_v",
                "after_keg_u": "after_keg_u", "after_keg_v": "after_keg_v",
                "after_zad_u": "after_zad_u", "after_zad_v": "after_zad_v",
                "uu_Kmm": "u_Kmm", "vv_Kmm": "v_Kmm", "ww": "ww",
                "wsd_effective": "wsd_effective", "e3t_Kmm": "e3t_Kmm",
                "e3u_Kmm": "e3u_Kmm", "e3v_Kmm": "e3v_Kmm",
                "e3w_Kmm": "e3w_Kmm", "e3t_0": "e3t_0",
                "e3u_0": "e3u_0", "e3v_0": "e3v_0", "e3w_0": "e3w_0",
                "e1e2t": "e1e2t", "e1e2u": "e1e2u", "e1e2v": "e1e2v",
                "r1_e1u": "r1_e1u", "r1_e2v": "r1_e2v",
                "r1_e1e2u": "r1_e1e2u", "r1_e1e2v": "r1_e1e2v",
                "tmask": "tmask", "umask": "umask", "vmask": "vmask",
                "wmask": "wmask",
            },
            right_is_full=False,
        ),
    }
    require(all(row["all_bit_identical"] for row in kt2_payload_identity.values()),
            f"kt2 duplicate payload moved: {kt2_payload_identity}")
    report = {
        "format": "nemo-testcase-l2-gyre-round46-kt2-stage-v1",
        "worktree": stamp,
        "mode": mode,
        "record_sha256": hashes,
        "calibration_cells_unequal": calibration,
        "legacy_records_parsed": {key: True for key in legacy_records},
        "legacy_kt1_raw_byte_identity": raw_twin,
        "legacy_kt1_consumed_identity": twin,
        "legacy_kt1_admission": twin_reports,
        "kt2_duplicate_payload_identity": kt2_payload_identity,
        "plant": plant,
        "given_inputs": [],
        "model_path_first_nonbit": {},
        "trajectory": [],
        "stage_twin": None,
        "stage1_handoff_walk": None,
        "tke_statement_walk": None,
        "status": "PASS",
    }
    if mode in {"given-inputs", "all"}:
        report["given_inputs"], report["model_path_first_nonbit"] = _given_inputs(
            records, plant
        )
    if mode in {"trajectory", "all"}:
        report["trajectory"] = _trajectory(records, plant)
        if any(row.get("status") != "PASS" for row in report["trajectory"]):
            report["status"] = "UNMEASURED"
    if mode in {"stage-twin", "stage-w-walk"}:
        report["stage_twin"] = _stage_twin(
            records, root, advmean_root, memory_root, btstep_root,
            stage_closure_root, stage1_w_root, stage1_r3_root,
            plant,
            walk_only=mode == "stage-w-walk",
            skip_stage1_operator_replay=skip_stage1_operator_replay)
    if mode == "stage1-handoff-walk":
        require(bool(stage1_handoff_boundary),
                "stage1-handoff-walk requires --stage1-handoff-boundary")
        report["stage1_handoff_walk"] = _stage1_handoff_walk(
            records, advmean_root, stage1_handoff_boundary,
            execution_mode, plant)
    if mode == "stage-tke-walk":
        from nemo_testcase_l2_gyre_round54_tke_operands import (
            read_record as read_tke_operand_record,
        )

        statement_record = read_admitted_tke_statement_walk(
            tke_statement_root)
        legacy_record = read_tke_operand_record(tke_operand_record)
        duplicate_rows = _tke_statement_duplicate_rows(
            statement_record, legacy_record)
        require(all(row["classification"] == "BIT"
                    for row in duplicate_rows
                    if row["admission_binding"]),
                "Round-101 consumed duplicate boundary moved from Round 59 "
                "before production scoring")
        report["tke_statement_walk"] = _tke_program_twin(
            records, advmean_root, memory_root, statement_record,
            legacy_record, plant, execution_mode, tke_rhs_materialization,
            tke_rhs_intermediate, bn2_intermediate)
    if mode == "stage-tke-record":
        from nemo_testcase_l2_gyre_round54_tke_operands import (
            read_record as read_tke_operand_record,
        )

        record_plant = {
            "stage-tke-record-header": "header",
            "stage-tke-record-truncation": "truncation",
            "stage-tke-record-stamp": "stamp",
        }.get(plant)
        statement_record = read_admitted_tke_statement_walk(
            tke_statement_root, plant=record_plant)
        legacy_record = read_tke_operand_record(tke_operand_record)
        duplicate_rows = _tke_statement_duplicate_rows(
            statement_record, legacy_record,
            plant_ulp=plant == "stage-tke-record-ulp")
        if plant != "stage-tke-record-ulp":
            require(all(row["classification"] == "BIT"
                        for row in duplicate_rows
                        if row["admission_binding"]),
                    "Round-101 consumed duplicate boundary moved from Round 59")
        report["tke_statement_walk"] = {
            "format": "nemo-testcase-l2-gyre-tke-statement-record-v1",
            "worktree": worktree_stamp(),
            "record": str(
                tke_statement_root
                / "oracle_tke_statement_walk_kt00000002.bin"),
            "record_sha256": statement_record["sha256"],
            "producer_commit": statement_record["producer_commit"],
            "header": statement_record["header"],
            "fields": list(TKE_STATEMENT_FIELDS),
            "physical_eof_bytes": 873028,
            "round59_record": str(tke_operand_record),
            "round59_sha256": sha256(tke_operand_record),
            "duplicate_rows": duplicate_rows,
            "new_boundaries": [
                "en_after_boundaries", "en_after_langmuir"],
        }
    if mode == "stage-twin":
        missing = [
            row for table in ("given_nemo_entry", "chained")
            for row in report["stage_twin"][table]
            if row.get("classification") == "UNMEASURED_WITH_SPEC"
        ]
        if missing:
            report["status"] = "UNMEASURED"
            report["stage_twin"]["missing_required_rows"] = [
                row["name"] for row in missing]
            # Decision 41 permits an ownership label only after the complete
            # stage contract closes.  Retain the ordering diagnostic without
            # allowing a partial table to print an owned-stage claim.
            report["stage_twin"]["first_measured_nonbit"] = (
                report["stage_twin"]["first_owned_nonbit"])
            report["stage_twin"]["first_owned_nonbit"] = None
    return report


def plant_aware_status(status: str, plant: str | None) -> str:
    """Never print a success label while a plant is firing.

    This gate's exit code is 1 for EVERY plant unconditionally, so it
    discriminates nothing: it cannot tell a plant that fired from one that
    did not.  What discriminates is the in-gate ``require`` on the targeted
    row -- the row must move from its clean count to clean + 1 -- together
    with the named ``plant_target``; a plant that failed to land aborts
    there and never reaches this label.  The label's only job is therefore
    to stop a scraped log reading ``STATUS PASS`` on a run whose whole
    purpose was to corrupt a reference.  Same contract as the round-103
    block replay and the receipt citation gate, whose statuses are derived
    rather than assumed.
    """
    return "PLANT-FIRED" if plant else status


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--expect-commit", required=True)
    p.add_argument(
        "--mode",
        choices=("validate", "given-inputs", "trajectory", "all", "stage-twin",
                 "stage-w-walk", "stage1-handoff-walk",
                 "stage-tke-record", "stage-tke-walk"),
        default="validate",
    )
    p.add_argument("--round40-kt1", type=Path, required=True)
    p.add_argument(
        "--round41-kt1", type=Path, default=ROUND41 / "oracle_dynadv_split_kt00000001_s3.bin"
    )
    p.add_argument("--advmean-root", type=Path, default=ADVMEAN_ROOT)
    p.add_argument("--memory-root", type=Path, default=MEMORY_ROOT)
    p.add_argument("--btstep-root", type=Path, default=BTSTEP_ROOT)
    p.add_argument(
        "--stage-closure-root", type=Path, default=STAGE_CLOSURE_ROOT)
    p.add_argument("--stage1-w-root", type=Path, default=STAGE1_W_ROOT)
    p.add_argument("--stage1-r3-root", type=Path, default=STAGE1_R3_ROOT)
    p.add_argument(
        "--tke-statement-root", type=Path, default=TKE_STATEMENT_ROOT)
    p.add_argument(
        "--tke-operand-record", type=Path, default=TKE_OPERAND_RECORD)
    p.add_argument(
        "--execution-mode",
        choices=("production-jit", "production-eager", "isolated-jit"),
        default="production-jit",
        help=("execution boundary for focused production walks; eager invokes "
              "the complete production closure with JIT disabled"),
    )
    p.add_argument(
        "--stage1-handoff-boundary",
        choices=STAGE1_HANDOFF_BOUNDARIES,
        default="",
        help="return and score one kt=1 stage-1 momentum handoff pair",
    )
    p.add_argument(
        "--skip-stage1-operator-replay", action="store_true",
        help=("stage-twin only: retain complete stage tables but omit the "
              "redundant post-table operator replay when the process-local "
              "LLVM compiler ceiling prevents materialization"),
    )
    p.add_argument("--tke-rhs-materialization", default="")
    p.add_argument(
        "--tke-rhs-intermediate",
        choices=TKE_RHS_SELECTORS,
        default="",
        help=("return and score exactly one compiled-order RHS intermediate; "
              "the final accumulation is returned alongside it"),
    )
    p.add_argument(
        "--bn2-intermediate",
        choices=BN2_INTERMEDIATES,
        default="",
        help=("return and score exactly one compiled-order bn2 intermediate; "
              "the final rn2 output is returned alongside it"),
    )
    p.add_argument(
        "--plant",
        choices=("header", "slot", "truncation", "calibration", "given", "trajectory",
                 "twin", "stage-entry-ulp", "stage-context-ulp",
                 "stage-rhs-ulp", "stage-w-transport-ulp",
                 "stage-w-carry-ulp", "stage-w-direct-r3-ulp",
                 "stage-assignment-output-ulp", "stage-w-record-stamp",
                 "stage-r3-record-stamp", "stage-tke-record-header",
                 "stage-tke-record-truncation", "stage-tke-record-stamp",
                 "stage-tke-record-ulp", "stage-tke-production-ulp",
                 "stage-tke-matrix-ulp",
                 "stage-shear-operand-ulp", "stage-tke-rhs-ulp",
                 "stage-tke-rhs-intermediate-ulp",
                 "stage-bn2-intermediate-ulp",
                 "stage1-handoff-output-ulp",
                 "stamp"),
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = run(
        args.root,
        expect_commit=args.expect_commit,
        mode=args.mode,
        plant=args.plant,
        round40_kt1=args.round40_kt1,
        round41_kt1=args.round41_kt1,
        advmean_root=args.advmean_root,
        memory_root=args.memory_root,
        btstep_root=args.btstep_root,
        stage_closure_root=args.stage_closure_root,
        stage1_w_root=args.stage1_w_root,
        stage1_r3_root=args.stage1_r3_root,
        tke_statement_root=args.tke_statement_root,
        tke_operand_record=args.tke_operand_record,
        execution_mode=args.execution_mode,
        stage1_handoff_boundary=args.stage1_handoff_boundary,
        tke_rhs_materialization=args.tke_rhs_materialization,
        tke_rhs_intermediate=args.tke_rhs_intermediate,
        bn2_intermediate=args.bn2_intermediate,
        skip_stage1_operator_replay=args.skip_stage1_operator_replay,
    )
    report["status"] = plant_aware_status(report["status"], args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS", report["status"])
    return 1 if args.plant or report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())

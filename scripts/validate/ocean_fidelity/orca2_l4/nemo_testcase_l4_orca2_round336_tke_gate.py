#!/usr/bin/env python3
"""Admit the passive, production-arithmetic ORCA2 round-336 kt=2 TKE walk."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round326_hmi_gate as hmi,
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
    nemo_testcase_l4_orca2_round333_tracer_gate as tracer,
    nemo_testcase_l4_orca2_round333_traldf_gate as ldf,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    BUILD_CONFIG_FILES,
    _observer_only,
    _regular_files,
)

MAGIC = "NEMO_L4_R336TKE"
HEADER_NAMES = (
    "version", "kt", "kbb", "kmm", "rank", "jpi", "jpj", "jpk",
    "ntsi", "ntei", "ntsj", "ntej", "real_bits", "n_fields",
)
FIELDS = (
    "entry_en", "entry_avt", "entry_avm_k", "entry_sh2", "entry_dissl",
    "entry_rn2", "entry_rn2b", "entry_drag_bottom", "entry_drag_top",
    "entry_nmln", "entry_hmlp", "entry_fr_i", "entry_hm_i", "entry_taum",
    "entry_htau", "entry_uu_kbb", "entry_vv_kbb", "entry_tmask",
    "entry_umask", "entry_vmask", "entry_ssmask", "entry_mbkt", "entry_mikt",
    "entry_gdepw_kmm", "entry_e3t_kmm", "entry_e3w_kmm", "entry_avtb_2d",
    "entry_avtb", "entry_avmb",
    "after_bc_en", "post_langmuir_en", "matrix_rhs_en", "matrix_diag", "matrix_upper",
    "matrix_lower", "solve_diag", "solve_rhs", "solve_bottom", "solve_back",
    "after_solve_en", "after_tke_en", "tke_pdlr", "pre_sweep_mxl",
    "after_lup_mxld", "after_ldown_mxl", "final_mxl", "final_mxld",
    "pre_pdlr_avt", "final_avm", "final_avt", "tke_wmask",
)
FULL_3D = {
    "entry_avm_k", "entry_rn2", "entry_rn2b", "entry_uu_kbb", "entry_vv_kbb",
    "entry_tmask", "entry_umask", "entry_vmask", "final_avm", "tke_wmask",
}
FULL_2D = {
    "entry_nmln", "entry_hmlp", "entry_fr_i", "entry_ssmask", "entry_mbkt",
    "entry_mikt",
}
FILLED_2D = {"entry_drag_bottom", "entry_drag_top"}
INTERIOR_2D = {
    "entry_hm_i", "entry_taum", "entry_htau", "entry_avtb_2d",
}
ONE_D = {"entry_avtb", "entry_avmb"}
NEGATIVE_ZERO_EDGE = {
    "matrix_diag", "matrix_upper",
    "matrix_lower", "solve_diag", "solve_rhs", "solve_bottom", "solve_back",
    "after_solve_en", "tke_pdlr", "pre_pdlr_avt",
}
REAL_LAYOUT = (94, 152, 31, 3, 92, 3, 150)
PLANTS = (
    "header", "field-order", "truncation", "checksum", "stamp", "extent",
    "entry", "rn2", "langmuir", "solve-steps",
    "pre-sweep", "sweep-steps", "north-halo-avm-k", "pdlr", "final",
    "signed-zero", "source-values", "time-level", "drag-range",
)
OBSERVER_UNITS = ("zdftke",)
WRITER_UNITS = ("tke_round336_writer",)
STANDARD_OUTPUTS = (
    "ORCA2_00000010_restart_0000.nc",
    "ORCA2_00000010_restart_0001.nc",
    "ORCA2_00000010_restart_ice_0000.nc",
    "ORCA2_00000010_restart_ice_0001.nc",
    "ocean.output",
)
REFERENCE_KEYS = tuple(
    f"{kind}_r{rank:04d}"
    for kind in ("tracer", "ldf", "ice")
    for rank in (0, 1)
)

# External array reads in the active ORCA2 tke_tke/tke_avn blocks.  Scalars
# and selectors are pinned by the run's namelist/ocean.output identity; local
# work arrays are derived by the blocks themselves.  Keeping this as an
# executable census makes a newly introduced array read fail preflight until
# it is deliberately classified and recorded.
READ_SET_FIELDS = {
    "en": ("entry_en", "after_tke_en"),
    "p_avt": ("entry_avt",), "p_avm": ("entry_avm_k",),
    "p_sh2": ("entry_sh2",), "dissl": ("entry_dissl",),
    "rn2": ("entry_rn2",), "rn2b": ("entry_rn2b",),
    "rCdU_bot": ("entry_drag_bottom",), "rCdU_top": ("entry_drag_top",),
    "nmln": ("entry_nmln",), "fr_i": ("entry_fr_i",),
    "hm_i": ("entry_hm_i",), "taum": ("entry_taum",),
    "htau": ("entry_htau",), "uu": ("entry_uu_kbb",),
    "vv": ("entry_vv_kbb",), "tmask": ("entry_tmask",),
    "umask": ("entry_umask",), "vmask": ("entry_vmask",),
    "wmask": ("tke_wmask",), "ssmask": ("entry_ssmask",),
    "mbkt": ("entry_mbkt",), "mikt": ("entry_mikt",),
    "gdepw": ("entry_gdepw_kmm",), "e3t": ("entry_e3t_kmm",),
    "e3w": ("entry_e3w_kmm",), "avtb_2d": ("entry_avtb_2d",),
    "avtb": ("entry_avtb",), "avmb": ("entry_avmb",),
    "p_pdlr": ("tke_pdlr",),
}
PINNED_READ_CONSTANTS = (
    "Kbb", "Kmm", "rn_Dt", "rho0", "grav", "vkarmn", "rpi", "rsmall",
    "rn_ebb", "rn_emin", "rn_emin0", "rn_bshear", "rn_ediff",
    "rn_ediss", "rn_efr", "rn_lc", "rn_mxl0", "rn_mxlice", "ri_cri",
    "rmxl_min", "nn_eice", "nn_etau", "nn_mxl", "nn_mxlice", "nn_pdl",
    "ln_lc", "ln_mxl0", "ln_drg_OFF", "ln_isfcav", "cpl_sdrftx",
    "cpl_phioc", "ln_phioc", "ln_sdw", "ln_mxhsw",
)
INACTIVE_BRANCH_READS = (
    "h_i", "hsw", "phioc", "ut0sd", "vt0sd", "utau", "vtau",
)

GateError = r331.GateError
require = r331.require


def record_name(rank: int) -> str:
    return f"oracle_r336_tke_kt00000002_r{rank:04d}.bin"


def audit_read_set(source_path: Path, writer_path: Path) -> dict[str, object]:
    """Mechanically prove the source-derived external-read census is covered."""
    source = source_path.read_text()
    writer = writer_path.read_text()
    start = source.find("SUBROUTINE tke_tke(")
    stop = source.find("END SUBROUTINE tke_avn", start)
    require(start >= 0 and stop > start, "cannot isolate TKE arithmetic blocks")
    blocks = source[start:stop]
    missing_source = [name for name in READ_SET_FIELDS
                      if re.search(rf"\b{re.escape(name)}\b", blocks, re.I) is None]
    require(not missing_source,
            f"read-set census tokens absent from arithmetic: {missing_source}")
    missing_fields = [field for fields in READ_SET_FIELDS.values() for field in fields
                      if field not in FIELDS or
                      re.search(rf"['\"]{re.escape(field)}['\"]", writer) is None]
    require(not missing_fields, f"read-set fields absent from record: {missing_fields}")
    missing_constants = [name for name in PINNED_READ_CONSTANTS
                         if re.search(rf"\b{re.escape(name)}\b", blocks, re.I) is None]
    require(not missing_constants,
            f"pinned read constants absent from arithmetic: {missing_constants}")
    return {
        "status": "PASS_R336_TKE_READ_SET",
        "recorded_array_reads": len(READ_SET_FIELDS),
        "recorded_fields": sum(len(value) for value in READ_SET_FIELDS.values()),
        "pinned_scalar_or_selector_reads": len(PINNED_READ_CONSTANTS),
        "inactive_branch_reads": list(INACTIVE_BRANCH_READS),
    }


def _layout(header: dict[str, int]) -> tuple[int, ...]:
    return tuple(header[name] for name in (
        "jpi", "jpj", "jpk", "ntsi", "ntei", "ntsj", "ntej"))


def _extent(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    if name in FULL_3D:
        return header["jpi"], header["jpj"], header["jpk"]
    if name in FULL_2D:
        return header["jpi"], header["jpj"], 1
    if name in FILLED_2D:
        return (header["ntei"] - header["ntsi"] + 3,
                header["ntej"] - header["ntsj"] + 3, 1)
    if name in ONE_D:
        return header["jpk"], 1, 1
    nx = header["ntei"] - header["ntsi"] + 1
    ny = header["ntej"] - header["ntsj"] + 1
    return nx, ny, 1 if name in INTERIOR_2D else header["jpk"]


def _origin(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    if name in FULL_3D or name in FULL_2D or name in ONE_D:
        return 1, 1, 1
    if name in FILLED_2D:
        return header["ntsi"] - 1, header["ntsj"] - 1, 1
    return header["ntsi"], header["ntsj"], 1


def read_record(path: Path, plant: str | None = None,
                expected_layout: tuple[int, ...] = REAL_LAYOUT) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = r331._take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i",
        r331._take(stream, 4 * len(HEADER_NAMES), "header"),
    )
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant in {"header", "time-level"}:
        header["kmm"] = -1
    if plant == "extent":
        header["ntei"] -= 1
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require((header["version"], header["kt"], header["kbb"], header["kmm"],
             header["real_bits"], header["n_fields"]) == (2, 2, 3, 3, 64, len(FIELDS)),
            "wrong version/kt/RK/field-count header")
    require(_layout(header) == expected_layout,
            f"wrong declared TKE layout {_layout(header)} != {expected_layout}")
    require(1 <= header["ntsi"] <= header["ntei"] <= header["jpi"],
            "invalid i bounds")
    require(1 <= header["ntsj"] <= header["ntej"] <= header["jpj"],
            "invalid j bounds")
    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = r331._take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", r331._take(stream, 28, f"{name} metadata"))
        if plant == "drag-range" and name == "entry_drag_bottom":
            i0 -= 1
        expected_rank = (1 if name in ONE_D else
                         2 if name in INTERIOR_2D | FILLED_2D | FULL_2D else 3)
        require(rank == expected_rank,
                f"{name}: bad array rank {rank}")
        require((n1, n2, n3) == _extent(name, header),
                f"{name}: wrong extent {(n1, n2, n3)}")
        require((i0, j0, k0) == _origin(name, header),
                f"{name}: wrong origin {(i0, j0, k0)}")
        checksum, = struct.unpack("=Q", r331._take(stream, 8, f"{name} checksum"))
        payload = r331._take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        shape = (n1,) if rank == 1 else (n1, n2) if rank == 2 else (n1, n2, n3)
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(shape, order="F")
        require(bool(np.all(np.isfinite(array))), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == FIELDS, "field order differs from TKE source order")
    record = {"header": header, "fields": fields, "checksums": checksums,
              "sha256": hashlib.sha256(raw).hexdigest()}
    if plant == "checksum":
        record["fields"][FIELDS[0]].flat[0] = np.nextafter(
            record["fields"][FIELDS[0]].flat[0], np.inf)
    r331._verify_checksums(record)
    return record


def _bits_equal(left: np.ndarray, right: np.ndarray, message: str) -> None:
    require(left.shape == right.shape, f"{message}: shape differs")
    require(np.array_equal(left.view(np.uint64), right.view(np.uint64)), message)


def _wet_index(wet: np.ndarray) -> tuple[int, ...]:
    indices = np.argwhere(wet)
    require(bool(indices.size), "record has no wet interface")
    return tuple(int(v) for v in indices[0])


def verify_record(record: dict, source: dict | None = None,
                  ice: dict | None = None, plant: str | None = None) -> dict[str, int]:
    require(plant is None or plant in {
        "entry", "rn2", "langmuir", "solve-steps",
        "pre-sweep", "sweep-steps", "north-halo-avm-k", "pdlr", "final",
        "signed-zero", "source-values"}, f"unsupported semantic plant {plant}")
    fields = record["fields"]
    header = record["header"]
    i0, i1 = header["ntsi"] - 1, header["ntei"]
    j0, j1 = header["ntsj"] - 1, header["ntej"]
    wet_full = fields["tke_wmask"] != 0.0
    wet = wet_full[i0:i1, j0:j1, :]
    index = _wet_index(wet)
    field_for_plant = {
        "entry": "entry_en", "rn2": "entry_rn2", "langmuir": "post_langmuir_en",
        "solve-steps": "solve_diag",
        "pre-sweep": "pre_sweep_mxl", "sweep-steps": "after_lup_mxld",
        "pdlr": "tke_pdlr", "final": "final_avt",
    }
    if plant in field_for_plant:
        name = field_for_plant[plant]
        target = fields[name]
        plant_index = index
        if plant == "final":
            process_index = _wet_index(wet[..., 1:-1])
            plant_index = (process_index[0], process_index[1], process_index[2] + 1)
        if name in FULL_3D:
            global_index = (plant_index[0] + i0, plant_index[1] + j0,
                            plant_index[2])
            if plant == "final":
                target[global_index] = np.nextafter(target[global_index], np.inf)
            else:
                target[global_index] = np.nan
        else:
            if plant == "final":
                target[plant_index] = np.nextafter(target[plant_index], np.inf)
            else:
                target[plant_index] = np.nan
    if plant == "north-halo-avm-k":
        fields["entry_avm_k"][i0, header["ntej"], index[2]] = np.nan
    if plant == "source-values":
        fields["entry_drag_bottom"].flat[0] = np.nextafter(
            fields["entry_drag_bottom"].flat[0], np.float64(np.inf))
    for name, values in fields.items():
        require(bool(np.all(np.isfinite(values))),
                f"{name}: non-finite value after semantic checks")
    for name in NEGATIVE_ZERO_EDGE:
        edge = fields[name][..., -1]
        if plant == "signed-zero" and name == "matrix_diag":
            edge.flat[0] = 0.0
        require(bool(np.all(edge.view(np.uint64) == np.uint64(1 << 63))),
                f"{name}: unfilled bottom edge lost -0 sentinel")
    require(bool(np.all(fields["entry_nmln"] == np.floor(fields["entry_nmln"]))),
            "mixed-layer level is not integer-valued")
    require(bool(np.all(fields["entry_mbkt"] == np.floor(fields["entry_mbkt"]))),
            "bottom level is not integer-valued")
    require(bool(np.all(fields["entry_mikt"] == np.floor(fields["entry_mikt"]))),
            "top level is not integer-valued")
    drag_i0, drag_i1 = header["ntsi"] - 2, header["ntei"] + 1
    drag_j0, drag_j1 = header["ntsj"] - 2, header["ntej"] + 1
    expected_bottom_drag = np.float64(-0.0004) * fields["entry_ssmask"][
        drag_i0:drag_i1, drag_j0:drag_j1]
    _bits_equal(fields["entry_drag_bottom"], expected_bottom_drag,
                "bottom drag differs from -rn_Cd0*rn_Uc0*ssmask")
    interior = wet[..., 1:-1]
    require(bool(np.any(interior)), "record has no wet process-range interface")
    for name in ("pre_sweep_mxl", "after_lup_mxld", "after_ldown_mxl",
                 "final_mxl", "final_mxld", "pre_pdlr_avt", "final_avt"):
        require(bool(np.all(fields[name][..., 1:-1][interior] >= 0.0)),
                f"{name}: negative wet process-range value")
    require(bool(np.all(fields["final_mxld"][..., 1:-1][interior] >=
                        fields["final_mxl"][..., 1:-1][interior])),
            "final wet dissipative length is below mixing length")
    pdlr = fields["tke_pdlr"][..., 1:-1][interior]
    require(bool(np.all((pdlr >= 0.1) & (pdlr <= 1.0))),
            "wet inverse Prandtl ratio is outside [0.1,1]")
    expected_final = np.maximum(
        fields["tke_pdlr"] * fields["pre_pdlr_avt"],
        fields["entry_avtb_2d"][..., None] * fields["entry_avtb"][None, None, :],
    ) * fields["tke_wmask"][i0:i1, j0:j1]
    _bits_equal(fields["final_avt"][..., 1:-1][interior],
                expected_final[..., 1:-1][interior],
                "final avt differs from the recorded NEMO pdlr statement")
    if source is not None:
        tracer_fields = source["tracer"]["fields"]
        ldf_fields = source["ldf"]["fields"]
        tracer_header = source["tracer"]["header"]
        require(header["kt"] == tracer_header["kt"],
                "TKE and round-333 tracer records have different steps")
        require(header["kmm"] == tracer_header["Kbb"],
                "TKE Kmm does not name the round-333 Kbb slot")
        _bits_equal(fields["entry_tmask"], ldf_fields["tmask"],
                    "TKE tmask differs from admitted round-333 LDF source")
        expected_e3t = ldf_fields["e3t_3d"] * (
            np.float64(1.0) + tracer_fields["r3t_Kbb"][..., None] * ldf_fields["tmask"])
        expected_e3w = ldf_fields["e3w_1d"][None, None, :] * (
            np.float64(1.0) + tracer_fields["r3t_Kbb"][..., None])
        _bits_equal(fields["entry_e3t_kmm"], expected_e3t[i0:i1, j0:j1],
                    "TKE live e3t(Kmm) differs from admitted round-333 replay")
        _bits_equal(fields["entry_e3w_kmm"], expected_e3w[i0:i1, j0:j1],
                    "TKE live e3w(Kmm) differs from admitted round-333 replay")
    if ice is not None:
        _bits_equal(fields["entry_fr_i"], ice["fields"]["fr_i"],
                    "fr_i differs from admitted ice-cover source")
        _bits_equal(fields["entry_hm_i"], ice["fields"]["hm_i"][i0:i1, j0:j1],
                    "hm_i differs from admitted ice source")
        _bits_equal(fields["entry_taum"], ice["fields"]["taum"][i0:i1, j0:j1],
                    "taum differs from admitted surface source")
    return {"wet_interfaces": int(np.count_nonzero(wet)),
            "wet_process_interfaces": int(np.count_nonzero(interior))}


def run_gate(root: Path, tracer_root: Path, ldf_root: Path, ice_root: Path,
             reference_digests: dict[str, str], expect_commit: str,
             plant: str | None = None) -> dict[str, object]:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit,
            f"producer commit {produced} != {expect_commit}")
    require(set(reference_digests) == set(REFERENCE_KEYS),
            "reference digest keys are incomplete")
    counts: dict[str, object] = {}
    hashes: dict[str, object] = {}
    for rank in (0, 1):
        path = root / record_name(rank)
        tracer_path = tracer_root / tracer.record_name(rank)
        ldf_path = ldf_root / ldf.record_name(rank)
        ice_path = ice_root / hmi.record_name(2, rank)
        require(path.is_file(), f"missing {path.name}")
        if plant is None:
            for kind, reference in (("tracer", tracer_path), ("ldf", ldf_path),
                                    ("ice", ice_path)):
                require(reference.is_file(), f"missing {kind} reference {reference.name}")
                require(hashlib.sha256(reference.read_bytes()).hexdigest() ==
                        reference_digests[f"{kind}_r{rank:04d}"],
                        f"{kind} reference digest changed: {reference.name}")
        parser_plant = plant if rank == 0 and plant in {
            "header", "field-order", "truncation", "checksum", "extent", "time-level",
            "drag-range",
        } else None
        record = read_record(path, plant=parser_plant)
        require(record["header"]["rank"] == rank, f"{path.name}: wrong rank")
        source = None
        ice = None
        if plant is None:
            source = {
                "tracer": tracer.read_stage3(tracer_path),
                "ldf": ldf.read_record(ldf_path),
            }
            ice = hmi.read_record(ice_path)
        semantic_plant = plant if rank == 0 and plant in {
            "entry", "rn2", "langmuir", "solve-steps",
            "pre-sweep", "sweep-steps", "north-halo-avm-k", "pdlr", "final",
            "signed-zero", "source-values"} else None
        counts[str(rank)] = verify_record(record, source, ice, semantic_plant)
        hashes[str(rank)] = {"tke": record["sha256"]}
        if plant is None:
            hashes[str(rank)].update(
                tracer=reference_digests[f"tracer_r{rank:04d}"],
                ldf=reference_digests[f"ldf_r{rank:04d}"],
                ice=reference_digests[f"ice_r{rank:04d}"])
    return {
        "format": "nemo-testcase-l4-orca2-round336-tke-v2",
        "status": "PASS_R336_TKE_KT2_ADMISSION",
        "producer_commit": produced,
        "counts": counts,
        "sha256": hashes,
    }


def output_contract(same_program_path: Path, writer_source: Path) -> dict[str, object]:
    """Derive admissible run outputs from the compiled observer/writer census."""
    program = json.loads(same_program_path.read_text())
    require(program.get("observer_units") == list(OBSERVER_UNITS),
            "same-program observer list moved")
    require(program.get("writer_units") == list(WRITER_UNITS),
            "same-program writer list moved")
    source = writer_source.read_text()
    require(len(re.findall(r"\bOPEN\s*\(", source, flags=re.IGNORECASE)) == 1,
            "compiled writer OPEN census moved")
    require('\"oracle_r336_tke_kt\",I8.8,\"_r\",I4.4,\".bin\"' in source,
            "compiled writer output format moved")
    return {
        "format": "nemo-testcase-l4-orca2-round336-output-contract-v1",
        "status": "PASS_R336_OUTPUT_CONTRACT",
        "observer_units": program["observer_units"],
        "writer_units": program["writer_units"],
        "observer_outputs": [record_name(rank) for rank in (0, 1)],
        "standard_outputs": list(STANDARD_OUTPUTS),
    }


def verify_output_set(root: Path, contract_path: Path, role: str) -> dict[str, object]:
    require(role in ("observed", "twin"), f"unknown output role {role}")
    contract = json.loads(contract_path.read_text())
    require(contract.get("format") ==
            "nemo-testcase-l4-orca2-round336-output-contract-v1",
            "output contract format moved")
    expected = set(contract["standard_outputs"])
    if role == "observed":
        expected.update(contract["observer_outputs"])
    actual = {
        path.name for path in root.iterdir()
        if path.is_file() and (
            path.name.startswith("oracle_") or
            (path.name.startswith("ORCA2_00000010_restart") and
             path.suffix == ".nc") or
            path.name == "ocean.output")
    }
    require(actual == expected,
            f"{role} output set differs: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}")
    return {"status": "PASS_R336_OUTPUT_SET", "role": role,
            "files": sorted(actual)}


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    require(tuple(patches) == OBSERVER_UNITS, "wrong round-336 TKE observer unit order")
    for name in BUILD_CONFIG_FILES:
        want = (ref_bld / name).read_text().replace(ref_cfg, new_cfg)
        require((new_bld / name).read_text() == want, f"build config differs: {name}")
    inserted = [line[1:] for line in patches["zdftke"].read_text().splitlines()
                if line.startswith("+") and not line.startswith("+++")]
    ref_ppsrc = _regular_files(ref_bld / "ppsrc")
    new_ppsrc = _regular_files(new_bld / "ppsrc")
    writers = {f"nemo/{unit}.f90" for unit in WRITER_UNITS}
    require(new_ppsrc - ref_ppsrc == writers,
            f"ppsrc: unexpected additions {sorted(new_ppsrc - ref_ppsrc)}")
    require(not ref_ppsrc - new_ppsrc,
            f"ppsrc: missing pristine files {sorted(ref_ppsrc - new_ppsrc)}")
    observer = "nemo/zdftke.f90"
    bad = [rel for rel in sorted(ref_ppsrc - {observer})
           if (ref_bld / "ppsrc" / rel).read_bytes() !=
           (new_bld / "ppsrc" / rel).read_bytes()]
    require(not bad, "ppsrc differs from pristine beyond zdftke: " + " ".join(bad))
    _observer_only(
        (ref_bld / "ppsrc" / observer).read_text().splitlines(),
        (new_bld / "ppsrc" / observer).read_text().splitlines(),
        inserted, "zdftke")
    ref_obj = _regular_files(ref_bld / "obj")
    new_obj = _regular_files(new_bld / "obj")
    writer_obj = {f"{unit}.o" for unit in WRITER_UNITS}
    require(new_obj - ref_obj == writer_obj,
            f"obj: unexpected additions {sorted(new_obj - ref_obj)}")
    require(not ref_obj - new_obj,
            f"obj: missing pristine files {sorted(ref_obj - new_obj)}")
    return {"status": "PASS_SAME_PROGRAM_PRISTINE_PLUS_R336_OBSERVER",
            "identical_ppsrc": len(ref_ppsrc) - 1,
            "fp_census_obj": len(ref_obj),
            "observer_units": list(patches), "writer_units": list(WRITER_UNITS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--tracer-reference-dir", type=Path)
    parser.add_argument("--ldf-reference-dir", type=Path)
    parser.add_argument("--ice-reference-dir", type=Path)
    parser.add_argument("--reference-sha256", action="append", default=[])
    parser.add_argument("--expect-commit")
    parser.add_argument("--same-program", type=Path, metavar="NEW_BLD")
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", action="append", default=[])
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--read-set-source", type=Path)
    parser.add_argument("--writer-source", type=Path)
    parser.add_argument("--build-output-contract", type=Path,
                        metavar="SAME_PROGRAM_JSON")
    parser.add_argument("--compiled-writer", type=Path)
    parser.add_argument("--check-output-set", type=Path, metavar="RUN_DIR")
    parser.add_argument("--output-contract", type=Path)
    parser.add_argument("--output-role", choices=("observed", "twin"))
    args = parser.parse_args()
    try:
        if args.read_set_source is not None:
            require(args.writer_source is not None,
                    "--read-set-source needs --writer-source")
            result = audit_read_set(args.read_set_source, args.writer_source)
        elif args.build_output_contract is not None:
            require(args.compiled_writer is not None,
                    "--build-output-contract needs --compiled-writer")
            result = output_contract(args.build_output_contract, args.compiled_writer)
        elif args.check_output_set is not None:
            require(args.output_contract is not None and args.output_role is not None,
                    "--check-output-set needs --output-contract and --output-role")
            result = verify_output_set(
                args.check_output_set, args.output_contract, args.output_role)
        elif args.same_program is not None:
            require(None not in (args.reference_bld, args.new_cfg, args.reference_cfg),
                    "--same-program needs reference BLD and config names")
            entries = [item.split("=", 1) for item in args.observer_patch]
            require(all(len(item) == 2 for item in entries),
                    "observer patches must be UNIT=PATH")
            result = same_program(
                args.same_program, args.reference_bld,
                {unit: Path(path) for unit, path in entries},
                new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(None not in (args.record_dir, args.tracer_reference_dir,
                                 args.ldf_reference_dir, args.ice_reference_dir,
                                 args.expect_commit),
                    "admission needs record/reference dirs and expected commit")
            entries = [item.split("=", 1) for item in args.reference_sha256]
            require(all(len(item) == 2 for item in entries),
                    "reference digests must be NAME=SHA256")
            digests = {name: digest for name, digest in entries}
            require(all(re.fullmatch(r"[0-9a-f]{64}", value)
                        for value in digests.values()),
                    "reference digests must be lowercase SHA-256")
            result = run_gate(args.record_dir, args.tracer_reference_dir,
                              args.ldf_reference_dir, args.ice_reference_dir,
                              digests,
                              args.expect_commit, args.plant)
            require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, UnicodeError, ValueError, KeyError, struct.error) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

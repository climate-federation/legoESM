#!/usr/bin/env python3
"""Exact ten-step CPU gate for the quarter-degree GYRE stage ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    _trace_native,
    bt_frame,
    expected_masks,
    lego_fields,
    read_entry,
    read_stage,
    read_bt_substeps,
    read_tracer_stage12,
    read_zdf_entry,
)
from nemo_testcase_state_ulp_probe import ulp_distance  # noqa: E402

NEMO_DIMS = (126, 86, 31)
DEFAULT_ORACLES = {
    "r4": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
        "round1/oracle_gyre_r4_stage/run"
    ),
    "r4_smag": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
        "round2/oracle_gyre_r4_smag_stage/run"
    ),
}
DEFAULT_AHM_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round4/oracle_gyre_r4_smag_ahm/run"
)
ADMISSIONS_PATH = HERE / "gyre_hires_oracle_admissions.json"
ORACLE_FLAVOURS = ("scalar", "vector")
FIELDS = ("T", "S", "u", "v", "ssh")
# Compiled stprk3.f90:200-222: step 1 leaves Nbb=3, so step 2 enters
# (Nbb,Nnn,Naa,Nrhs)=(3,2,1,1); the swaps after stages 1/2 give these Naa
# registry slots.  Keep this independent of the record headers being checked.
STEP2_STAGE_KAA = {1: 1, 2: 2, 3: 1}
WRITER_STEP1_RECORDS = (
    "oracle_step_entry_kt00000002.bin",
    "oracle_rhs_kt00000001.bin",
    "oracle_stage_kt00000001_s1.bin",
    "oracle_stage_kt00000001_s2.bin",
    "oracle_stage_kt00000001_s3.bin",
    "oracle_zdf_entry_kt00000001.bin",
    "oracle_rktracer_operands_kt00000001_s1.bin",
    "oracle_rktracer_operands_kt00000001_s2.bin",
    "oracle_tracer_transport_kt00000001_s3.bin",
    "oracle_rktracer_stage3_kt00000001.bin",
    "oracle_qsr_stage3_kt00000001.bin",
)
WRITER_SCOPED_RECORDS = frozenset(
    name for name in WRITER_STEP1_RECORDS
    if "rktracer_operands" in name or "tracer_transport" in name
)
TRACER_STAGE12_ARRAYS = (
    "zero_T", "zero_S", "zFu", "zFv", "zFw",
    "after_advection_T", "after_advection_S",
    "after_sbc_T", "after_sbc_S", "Kbb_T", "Kbb_S",
    "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _writer_loop_bounds(path: Path) -> dict:
    """Read the P3 writer's own interior loop bounds from ``tra_zdf``."""
    raw = path.read_bytes()[:80]
    if len(raw) != 80 or raw[:16].decode("ascii").rstrip() != "NEMO_L2_TRAZD_1":
        raise ValueError(f"{path}: not a complete tra_zdf header")
    fields = (
        "version", "kt", "stage", "Kbb", "Kmm", "Krhs", "Kaa",
        "nx", "ny", "nz", "nzm1", "ntsi", "ntei", "ntsj", "ntej",
        "bits",
    )
    header = dict(zip(fields, struct.unpack("=16i", raw[16:])))
    if (header["version"], header["bits"], header["nzm1"]) != (
            1, 64, header["nz"] - 1):
        raise ValueError(f"{path}: malformed tra_zdf header")
    if not (1 <= header["ntsi"] <= header["ntei"] <= header["nx"]
            and 1 <= header["ntsj"] <= header["ntej"] <= header["ny"]):
        raise ValueError(f"{path}: loop bounds outside the recorded domain")
    return header


def _read_full_tracer_stage12(path: Path) -> tuple[bytes, dict, np.ndarray]:
    """Read the complete fixed-layout record, including its ignored trailer."""
    raw = path.read_bytes()
    if len(raw) < 60:
        raise ValueError(f"{path}: short tracer-stage record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=11i", raw[16:60])
    fields = (
        "version", "kt", "stage", "Kbb", "Kmm", "Krhs", "Kaa",
        "nx", "ny", "nz", "bits",
    )
    header = dict(zip(fields, header_values))
    if magic != "NEMO_L2_RKTRA_1" or header["version"] != 1 \
            or header["bits"] != 64 or header["stage"] not in (1, 2):
        raise ValueError(f"{path}: malformed tracer-stage header")
    values = np.frombuffer(raw, dtype=np.float64, offset=60)
    n3 = header["nx"] * header["ny"] * header["nz"]
    n2 = header["nx"] * header["ny"]
    if values.size != len(TRACER_STAGE12_ARRAYS) * n3 + 3 * n2:
        raise ValueError(f"{path}: malformed tracer-stage payload")
    return raw[:60], header, values


def _read_full_tracer_transport(path: Path) -> tuple[bytes, dict, np.ndarray]:
    """Read the complete fixed-layout active-tracer transport record."""
    raw = path.read_bytes()
    if len(raw) < 60:
        raise ValueError(f"{path}: short tracer-transport record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=11i", raw[16:60])
    fields = (
        "version", "kt", "stage", "Kbb", "Kmm", "Kaa", "Krhs",
        "nx", "ny", "nz", "bits",
    )
    header = dict(zip(fields, header_values))
    if magic != "NEMO_L2_TRTRP_1" or header["version"] != 1 \
            or header["bits"] != 64 or header["stage"] != 3:
        raise ValueError(f"{path}: malformed tracer-transport header")
    values = np.frombuffer(raw, dtype=np.float64, offset=60)
    n3 = header["nx"] * header["ny"] * header["nz"]
    if values.size != 3 * n3:
        raise ValueError(f"{path}: malformed tracer-transport payload")
    return raw[:60], header, values


def compare_scalar_writer_identity(
    twin_root: Path,
    candidate_root: Path,
    zdf_record: Path,
    *,
    plant_active_word: bool = False,
) -> dict:
    """Prove a widened writer leaves every word consumed by the gates fixed.

    NEMO allocates the tracer transport work arrays at full-halo extent but
    initializes them only over ``ntsi:ntei``/``ntsj:ntej``.  The P3 readers
    crop to that same box before any gate consumes a word.  Compare that box
    exactly, retain raw byte identity for every other record and for the
    ignored r3 trailer, and report (rather than silently discard) moving
    uninitialized halo words.
    """
    bounds = _writer_loop_bounds(zdf_record)
    rows = []
    active_mismatches = 0
    excluded = []
    for name in WRITER_STEP1_RECORDS:
        twin = twin_root / name
        candidate = candidate_root / name
        if not twin.is_file() or not candidate.is_file():
            raise FileNotFoundError(
                f"writer identity input missing: {twin if not twin.is_file() else candidate}")
        if name not in WRITER_SCOPED_RECORDS:
            equal = twin.read_bytes() == candidate.read_bytes()
            active_mismatches += int(not equal)
            rows.append({"record": name, "scope": "all_bytes", "exact": equal})
            continue

        stage12 = "rktracer_operands" in name
        reader = (
            _read_full_tracer_stage12 if stage12
            else _read_full_tracer_transport
        )
        array_names = (
            TRACER_STAGE12_ARRAYS if stage12 else ("zFu", "zFv", "zFw")
        )
        twin_header, twin_meta, twin_values = reader(twin)
        cand_header, cand_meta, cand_values = reader(candidate)
        if twin_header != cand_header or twin_meta != cand_meta:
            raise ValueError(f"{name}: tracer-stage headers differ")
        if (twin_meta["nx"], twin_meta["ny"], twin_meta["nz"]) != (
                bounds["nx"], bounds["ny"], bounds["nz"]):
            raise ValueError(f"{name}: domain differs from tra_zdf loop record")
        nx, ny, nz = twin_meta["nx"], twin_meta["ny"], twin_meta["nz"]
        n3, n2 = nx * ny * nz, nx * ny
        candidate_values = np.array(cand_values, copy=True)
        if plant_active_word and stage12 and name.endswith("_s1.bin"):
            # First word in NEMO's own interior loop box of zFu.
            offset = 2 * n3
            index = ((bounds["ntsi"] - 1)
                     + nx * (bounds["ntsj"] - 1))
            candidate_values[offset + index] = np.nextafter(
                candidate_values[offset + index], np.float64(np.inf))

        record_active = 0
        record_excluded = []
        for array_index, array_name in enumerate(array_names):
            begin = array_index * n3
            left = twin_values[begin:begin + n3].reshape((nx, ny, nz), order="F")
            right = candidate_values[begin:begin + n3].reshape(
                (nx, ny, nz), order="F")
            unequal = left.view(np.uint64) != right.view(np.uint64)
            consumed = np.zeros((nx, ny, nz), dtype=bool)
            consumed[
                bounds["ntsi"] - 1:bounds["ntei"],
                bounds["ntsj"] - 1:bounds["ntej"],
                :,
            ] = True
            record_active += int(np.count_nonzero(unequal & consumed))
            for index in np.argwhere(unequal & ~consumed):
                i, j, k = (int(value) for value in index)
                record_excluded.append({
                    "array": array_name,
                    "fortran_index_1based": [i + 1, j + 1, k + 1],
                    "reason": "outside recorded ntsi:ntei/ntsj:ntej loop bounds",
                })

        trailer_left = twin_values[len(array_names) * n3:]
        trailer_right = candidate_values[len(array_names) * n3:]
        trailer_mismatches = int(np.count_nonzero(
            trailer_left.view(np.uint64) != trailer_right.view(np.uint64)))
        record_active += trailer_mismatches
        active_mismatches += record_active
        excluded.extend({"record": name, **item} for item in record_excluded)
        rows.append({
            "record": name,
            "scope": "NEMO loop bounds plus complete r3 trailer",
            "active_or_consumed_mismatches": record_active,
            "excluded_uninitialized_halo_words": len(record_excluded),
            "exact": record_active == 0,
        })

    report = {
        "format": "gyre-scalar-step2-writer-identity-v1",
        "loop_bounds": bounds,
        "reader_scope": (
            "all words returned by the P3 [2:-2,2:-2] tracer reader; "
            "all bytes of every other consumed record"
        ),
        "rows": rows,
        "excluded_words": excluded,
        "plant_active_word": plant_active_word,
        "active_or_consumed_mismatches": active_mismatches,
        "verdict": "PASS" if active_mismatches == 0 else "DEBT",
    }
    return report


def admit_oracle(
    deck: str,
    math_flavour: str,
    oracle_root: Path | None = None,
    *,
    admissions_path: Path = ADMISSIONS_PATH,
) -> dict:
    """Fail closed unless the requested oracle matches its registered bytes."""
    if math_flavour not in ORACLE_FLAVOURS:
        raise ValueError(f"unknown oracle math flavour {math_flavour!r}")
    try:
        registry = json.loads(admissions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read oracle admission registry: {exc}") from exc
    if registry.get("format") != "gyre-hires-oracle-admissions-v1":
        raise ValueError("unsupported oracle admission registry format")
    key = f"{deck}:{math_flavour}"
    try:
        registered = registry["admissions"][key]
    except KeyError as exc:
        raise ValueError(f"unregistered oracle selection {key!r}") from exc
    if (registered.get("deck"), registered.get("math_flavour")) != (
        deck, math_flavour,
    ):
        raise ValueError(f"{key}: admission deck/flavour mismatch")
    root = Path(registered["root"])
    if oracle_root is not None and oracle_root.resolve() != root.resolve():
        raise ValueError(
            f"{key}: unregistered oracle root {oracle_root}; expected {root}"
        )
    manifest = Path(registered["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"ACQUISITION_NEEDED: {manifest}")
    if _sha256(manifest) != registered["manifest_sha256"]:
        raise ValueError(f"{key}: manifest digest mismatch")
    checked_proofs = {}
    for name, proof in registered.get("proofs", {}).items():
        path = Path(proof["path"])
        if not path.is_file():
            raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
        actual = _sha256(path)
        if actual != proof["sha256"]:
            raise ValueError(f"{key}: proof digest mismatch for {name}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{key}: malformed proof {name}: {exc}") from exc
        if payload.get("verdict") != proof["verdict"]:
            raise ValueError(f"{key}: proof verdict mismatch for {name}")
        checked_proofs[name] = {
            "path": str(path),
            "sha256": actual,
            "verdict": payload["verdict"],
        }
    checked = {}
    for relative, expected in registered["files"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{key}: digest mismatch for {relative}: {actual} != {expected}"
            )
        checked[relative] = actual
    return {
        **registered,
        "key": key,
        "root": str(root),
        "manifest": str(manifest),
        "checked_files": checked,
        "checked_proofs": checked_proofs,
    }


def compiled_usrdef_math_program(
    executable: Path, *, expected_flavour: str = "vector"
) -> dict:
    """Inventory transcendental calls in the admitted usrdef_sbc routine."""
    if not executable.is_file():
        raise FileNotFoundError(f"missing admitted NEMO executable: {executable}")
    completed = subprocess.run(
        ["objdump", "-d", "--demangle",
         "--disassemble=__usrdef_sbc_MOD_usrdef_sbc_oce", str(executable)],
        check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise ValueError(f"objdump failed for {executable}: {completed.stderr.strip()}")
    calls = sorted(set(re.findall(
        r"callq?\s+[^<]*<([^>@]+)(?:@[^>]*)?>", completed.stdout)))
    transcendental_names = ("sin", "cos", "exp")
    vector_transcendentals = [
        name for name in calls
        if "_ZGV" in name and name.endswith(transcendental_names)
    ]
    scalar_transcendentals = [
        name for name in calls if name in transcendental_names
    ]
    vector_sin = [name for name in vector_transcendentals if name.endswith("sin")]
    scalar_sin = [name for name in scalar_transcendentals if name == "sin"]
    if expected_flavour == "vector":
        if ({name.rsplit("_", 1)[-1] for name in vector_transcendentals}
                != {"sin", "cos"}
                or set(scalar_transcendentals) != {"sin", "cos"}):
            raise ValueError(
                "vector oracle usrdef_sbc did not expose vector and scalar "
                "sin/cos with no exp"
            )
    elif expected_flavour == "scalar":
        if vector_transcendentals or set(scalar_transcendentals) != {"sin", "cos"}:
            raise ValueError(
                "scalar oracle usrdef_sbc did not expose scalar-only sin/cos "
                "with no exp"
            )
    else:
        raise ValueError(f"unknown expected math flavour {expected_flavour!r}")
    return {
        "executable": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "routine": "__usrdef_sbc_MOD_usrdef_sbc_oce",
        "transcendental_calls": calls,
        "vector_transcendental_calls": vector_transcendentals,
        "scalar_transcendental_calls": scalar_transcendentals,
        "vector_sin_calls": vector_sin,
        "scalar_sin_calls": scalar_sin,
        "math_flavour": expected_flavour,
    }


def wind_expression_arms(r1_rho0, stress, current_reciprocal, depth_ref, mask):
    """Replay the compiled ``stp2d`` wind expression and two discriminators."""
    rho = np.float64(r1_rho0)
    tau = np.asarray(stress, dtype=np.float64)
    live_reciprocal = np.asarray(current_reciprocal, dtype=np.float64)
    depth = np.asarray(depth_ref, dtype=np.float64)
    wet = np.asarray(mask, dtype=np.float64)
    if not (tau.shape == live_reciprocal.shape == depth.shape == wet.shape):
        raise ValueError("wind-expression operand shape mismatch")
    one = np.float64(1.0)
    stored_reciprocal = wet / ((depth + one) - wet)
    return {
        # GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stp2d.f90:198-199.
        "nemo_written": (rho * tau) * stored_reciprocal,
        "right_associated": rho * (tau * stored_reciprocal),
        "reciprocal_first": (rho * stored_reciprocal) * tau,
        "current_reciprocal": (rho * tau) * live_reciprocal,
        "stored_reciprocal": stored_reciprocal,
    }


def coefficient_branch_census(diagnostics, t_active, f_active):
    """Classify the final max/min arm on active T and F coefficient points."""
    result = {}
    for point, active in (("T", t_active), ("F", f_active)):
        wet = np.asarray(active, dtype=bool)
        lower = np.asarray(diagnostics[f"{point.lower()}_lower_clamped"], dtype=bool)
        upper = np.asarray(diagnostics[f"{point.lower()}_upper_clamped"], dtype=bool)
        if wet.shape != lower.shape or lower.shape != upper.shape or not wet.any():
            raise ValueError(f"{point}: invalid Smagorinsky branch mask")
        # The source evaluates MIN(MAX(deformation, lower), upper), so an
        # upper clamp owns a cell even if the intermediate MAX chose lower.
        arms = {
            "lower_limit": wet & lower & ~upper,
            "deformation": wet & ~lower & ~upper,
            "upper_limit": wet & upper,
        }
        count = int(np.count_nonzero(wet))
        counts = {name: int(np.count_nonzero(mask)) for name, mask in arms.items()}
        if sum(counts.values()) != count:
            raise ValueError(f"{point}: Smagorinsky branch census is not exhaustive")
        result[point] = {
            "active_count": count,
            "counts": counts,
            "fractions": {name: value / count for name, value in counts.items()},
        }
    return result


def exact_score(name: str, oracle, candidate, mask, *, plant_one_ulp=False) -> dict:
    """Score the preregistered bit bar and optionally plant exactly one ULP."""
    reference = np.asarray(oracle, dtype=np.float64)
    value = np.asarray(candidate, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if reference.shape != value.shape or value.shape != active.shape:
        raise ValueError(f"{name}: shape mismatch")
    if not active.any():
        raise ValueError(f"{name}: empty active mask")
    if plant_one_ulp:
        value = value.copy()
        index = tuple(np.argwhere(active)[0])
        toward = np.inf if value[index] >= 0.0 else -np.inf
        value[index] = np.nextafter(value[index], toward)
    unequal = int(
        np.count_nonzero(reference[active].view(np.uint64) != value[active].view(np.uint64))
    )
    ulps = ulp_distance(reference[active], value[active])
    absolute = np.abs(reference[active] - value[active])
    worst = int(np.argmax(absolute))
    active_indices = np.argwhere(active)
    unequal_bits = (
        reference[active].view(np.uint64) != value[active].view(np.uint64)
    )
    first_unequal = (
        active_indices[int(np.flatnonzero(unequal_bits)[0])]
        if unequal else None
    )
    return {
        "name": name,
        "status": "AT-BAR" if unequal == 0 else "DEBT",
        "n": int(active.sum()),
        "n_unequal": unequal,
        "max_abs": float(absolute[worst]),
        "ulp_at_max_abs": int(ulps[worst]),
        "max_ulp": int(np.max(ulps)),
        "max_abs_index": [int(index) for index in active_indices[worst]],
        "first_unequal_index": (
            [int(index) for index in first_unequal]
            if first_unequal is not None else None
        ),
        "first_unequal_ulp": (
            int(ulps[int(np.flatnonzero(unequal_bits)[0])]) if unequal else 0
        ),
        "bit_bar": 0,
        "plant_one_ulp": plant_one_ulp,
    }


def first_rk_difference(
    stage_walk: list[dict], *, minimum_max_abs: float = 0.0,
) -> dict | None:
    """Return the first field that differs in source stage/field order."""
    for stage in stage_walk:
        for field, row in zip(FIELDS, stage["rows"], strict=True):
            if row["n_unequal"] and row["max_abs"] >= minimum_max_abs:
                return {
                    "stage": stage["stage"],
                    "field": field,
                    "minimum_max_abs": minimum_max_abs,
                    "score": row,
                }
    return None


def accumulated_component_score(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
    component: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Score a component at the accumulator boundary that consumed it.

    NEMO records the accumulator before and after an operator, not the
    operator addend. Floating-point endpoint subtraction is not the inverse
    of the executed addition, so start from NEMO's exact preceding
    accumulator and add the isolated legoESM component instead.
    """
    return exact_score(name, after, before + component, mask)


def _full_xyz(values: np.ndarray, dims=NEMO_DIMS) -> np.ndarray:
    nx, ny, nz = dims
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_tracer_stage3_hires(
    path: Path, *, dims=NEMO_DIMS, expected_kt: int = 1,
) -> dict:
    """Read the P3 stage-3 tracer accumulator at the selected resolution."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKTR3_1":
        raise ValueError(f"{path}: bad stage-3 tracer magic")
    expected_slots = (
        (1, 2, 3, 3) if expected_kt % 2 else (3, 2, 1, 1)
    )
    if ((version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
            != (1, expected_kt, 3, *expected_slots, *dims, 64)):
        raise ValueError(f"{path}: bad stage-3 tracer header")
    n3, n2 = nx * ny * nz, nx * ny
    if values.size != 16 * n3 + 3 * n2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad stage-3 tracer payload")
    names = (
        "zero_T", "zero_S", "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
        "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
        "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result = {
        name: _full_xyz(values[index * n3:(index + 1) * n3], dims)
        for index, name in enumerate(names)
    }
    offset = 16 * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        block = values[offset + index * n2:offset + (index + 1) * n2]
        result[name] = block.reshape((nx, ny), order="F")[2:-2, 2:-2].T
    result.update({"Kbb": kbb, "Kmm": kmm, "Krhs": krhs, "Kaa": kaa})
    return result


def read_qsr_stage3_hires(
    path: Path, *, dims=NEMO_DIMS,
    expected_kt: int = 1,
) -> dict:
    """Read the isolated executed ``tra_qsr`` increment."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, krhs, nx, ny, nz, bits = header
    expected_krhs = 3 if expected_kt % 2 else 1
    if magic != "NEMO_L2_QSR___1":
        raise ValueError(f"{path}: bad QSR magic")
    if ((version, kt, stage, kmm, krhs, nx, ny, nz, bits)
            != (1, expected_kt, 3, 2, expected_krhs, *dims, 64)):
        raise ValueError(f"{path}: bad QSR header")
    n2 = (nx - 4) * (ny - 4)
    n3 = nx * ny * nz
    if values.size != n2 + n3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad QSR payload")
    return {
        "qsr": values[:n2].reshape((nx - 4, ny - 4), order="F").T,
        "dT_dt": _full_xyz(values[n2:], dims),
        "Kmm": kmm,
        "Krhs": krhs,
    }


def _native_f_to_vertex(native: np.ndarray) -> np.ndarray:
    """Map NEMO F(i,j) to legoESM vertex[j+1,(i+1) mod nx]."""
    ny, nx, nz = native.shape
    result = np.zeros((ny + 1, nx + 1, nz), dtype=np.float64)
    columns = (np.arange(nx + 1) - 1) % nx
    result[1:] = native[:, columns]
    return result


def read_ahm(path: Path, *, expected_kt: int, dims=NEMO_DIMS) -> dict:
    """Read the acquired post-ldf_dyn mode-32 coefficient pair."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii")
        version, kt, kbb, nx, ny, nz, bits = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    if magic != "NEMO_L4_AHMTF__1":
        raise ValueError(f"{path}: bad AHM magic")
    if (version, kt, nx, ny, nz, bits) != (1, expected_kt, *dims, 64):
        raise ValueError(f"{path}: bad AHM header")
    count = nx * ny * nz
    if values.size != 2 * count or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad AHM payload")
    return {
        "kt": kt,
        "Kbb": kbb,
        "ahmt": _full_xyz(values[:count], dims)[..., :nz - 1],
        "ahmf": _native_f_to_vertex(
            _full_xyz(values[count:], dims))[..., :nz - 1],
    }


def read_dynldf(path: Path, *, expected_kt: int, dims=NEMO_DIMS) -> dict:
    """Read NEMO's stage-3 RHS immediately before and after dyn_ldf."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii")
        header = struct.unpack("=10i", handle.read(40))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, nx, ny, nz, bits = header
    if magic != "NEMO_L4_DYNLDF_1":
        raise ValueError(f"{path}: bad dyn_ldf magic")
    if ((version, kt, stage, nx, ny, nz, bits)
            != (1, expected_kt, 3, *dims, 64)):
        raise ValueError(f"{path}: bad dyn_ldf header")
    count = nx * ny * nz
    if values.size != 4 * count or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad dyn_ldf payload")
    blocks = [
        _full_xyz(block, dims)[..., :nz - 1]
        for block in np.split(values, 4)
    ]
    return {
        "kt": kt, "stage": stage, "Kbb": kbb, "Kmm": kmm, "Krhs": krhs,
        "u_before": blocks[0], "v_before": blocks[1],
        "u_after": blocks[2], "v_after": blocks[3],
    }


def read_stage2_terms(path: Path, *, dims=NEMO_DIMS) -> dict:
    """Read the scalar writer's source-ordered stage-2 momentum RHS."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, krhs, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKTRM_1":
        raise ValueError(f"{path}: bad stage-2 term magic")
    if ((version, kt, stage, kmm, krhs, nx, ny, nz, bits)
            != (1, 1, 2, 3, 2, *dims, 64)):
        raise ValueError(f"{path}: bad stage-2 term header")
    count = nx * ny * nz
    if values.size != 8 * count or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: bad stage-2 term payload")
    names = (
        "before_u", "before_v", "after_hpg_u", "after_hpg_v",
        "after_vorticity_u", "after_vorticity_v",
        "after_advection_u", "after_advection_v",
    )
    result = {
        name: _full_xyz(values[index * count:(index + 1) * count], dims)
        for index, name in enumerate(names)
    }
    result.update({"stage": stage, "Kmm": kmm, "Krhs": krhs})
    return result


def read_stage2_composition(path: Path, *, dims=NEMO_DIMS) -> dict:
    """Read raw update and post-barotropic-replacement stage-2 states."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKSTG_1":
        raise ValueError(f"{path}: bad stage-2 composition magic")
    if ((version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
            != (1, 1, 2, 1, 3, 2, 2, *dims, 64)):
        raise ValueError(f"{path}: bad stage-2 composition header")
    n3, n2 = nx * ny * nz, nx * ny
    ni = (nx - 4) * (ny - 4)
    if values.size != 10 * n3 + 2 * n2 + 2 * ni:
        raise ValueError(f"{path}: bad stage-2 composition payload")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite stage-2 composition payload")
    result = {}
    offset = 0
    for name in (
        "Kbb_u", "Kbb_v", "Kmm_u", "Kmm_v", "Krhs_u", "Krhs_v",
        "raw_Kaa_u", "raw_Kaa_v",
    ):
        result[name] = _full_xyz(values[offset:offset + n3], dims)
        offset += n3
    offset += 2 * n2  # barotropic Kaa u/v, not scored by this discriminator
    offset += 2 * ni  # zub/zvb, already represented by the post state
    result["post_mean_Kaa_u"] = _full_xyz(values[offset:offset + n3], dims)
    offset += n3
    result["post_mean_Kaa_v"] = _full_xyz(values[offset:offset + n3], dims)
    offset += n3
    if offset != values.size:
        raise ValueError(f"{path}: unread stage-2 composition payload")
    return result


def _redundant_faces(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map native NEMO east/north faces to legoESM redundancy."""
    u = np.asarray(u)[..., :30]
    v = np.asarray(v)[..., :30]
    return (
        np.concatenate([u[:, -1:, :], u], axis=1),
        np.concatenate([v[-1:, :, :], v], axis=0),
    )


def mismatch_map(reference, candidate, regions: dict[str, np.ndarray], clamps) -> dict:
    """Count bit differences by preregistered spatial/clamp class."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    unequal = reference.view(np.uint64) != candidate.view(np.uint64)
    mapped = {name: int(np.count_nonzero(unequal & mask))
              for name, mask in regions.items()}
    mapped.update({
        "lower_clamped": int(np.count_nonzero(unequal & np.asarray(clamps[0]))),
        "upper_clamped": int(np.count_nonzero(unequal & np.asarray(clamps[1]))),
    })
    mapped["all"] = int(np.count_nonzero(unequal))
    return mapped


def run(
    oracle_root: Path, output: Path, *, deck: str = "r4",
    ahm_root: Path = DEFAULT_AHM_ROOT,
    math_flavour: str = "vector",
    trajectory_only: bool = False,
) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask,
        nemo_smagorinsky_bilaplacian_coefficients_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        GYRE_R4_SMAG_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree ladder certification requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if deck not in DEFAULT_ORACLES:
        raise ValueError(f"unknown ladder deck {deck!r}")
    admission = admit_oracle(deck, math_flavour, oracle_root)
    base = GYRE_R4_NAMELIST if deck == "r4" else GYRE_R4_SMAG_NAMELIST
    case = f"GYRE-zco-{deck}"
    namelist = base._replace(nn_itend=10, nn_stock=10, nn_write=10)
    card = build_nemo_testcase_card("GYRE-zco", namelist=namelist)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    def host_and_clear(value):
        host = jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf)
            if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
            value,
        )
        jax.clear_caches()
        return host

    masks = expected_masks(card)
    state = card.recipe.initial_state
    steps = []
    controls = []
    rk_stages = []
    tracer_stage1_transport = []
    tracer_stage1_walk = []
    tracer_stage1_transport_discriminator = None
    stage2_momentum_term_walk = None
    vertical_candidate = None
    barotropic_resolution_walk = None
    smagorinsky_coefficients = []
    smagorinsky_branch_census = []
    smagorinsky_operator = []
    ahm_records = {}
    if deck == "r4_smag":
        cell3 = (
            np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
            * np.asarray(state.land_mask.data)[..., None]
        )
        coefficient_vertex_mask = jax.vmap(
            lambda level: compute_vertex_mask(level, grid=card.recipe.grid),
            in_axes=-1, out_axes=-1,
        )(jnp.asarray(cell3))
    for kt in range(1, 11):
        path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"ACQUISITION_NEEDED: {path}")
        oracle = read_entry(path, dims=NEMO_DIMS)
        expected_nbb = 1 if kt % 2 else 3
        if (oracle["kt"], oracle["Nbb"]) != (kt, expected_nbb):
            raise ValueError(f"{path}: wrong kt/Nbb")
        candidate = lego_fields(state)
        rows = []
        for field in FIELDS:
            reference = oracle[field] if field == "ssh" else oracle[field][..., :30]
            row = exact_score(
                f"{case}.kt{kt}.before.{field}", reference, candidate[field], masks[field]
            )
            rows.append(row)
            if kt == 1:
                planted = exact_score(
                    f"control.one_ulp.{field}",
                    reference,
                    candidate[field],
                    masks[field],
                    plant_one_ulp=True,
                )
                controls.append(
                    {
                        **planted,
                        "status": "FIRED" if planted["n_unequal"] > 0 else "MISSED",
                    }
                )
        steps.append({"kt": kt, "Nbb": oracle["Nbb"], "rows": rows})
        if (not trajectory_only and deck == "r4_smag" and math_flavour == "vector"
                and kt in (2, 3, 10)):
            # Rule 12: score the changed coefficient on NEMO's own Kbb
            # velocity input, not legoESM's already-diverged trajectory.
            u_input = np.concatenate(
                [np.zeros((oracle["u"].shape[0], 1, 30)), oracle["u"][..., :30]],
                axis=1,
            )
            v_input = np.concatenate(
                [np.zeros((1, oracle["v"].shape[1], 30)), oracle["v"][..., :30]],
                axis=0,
            )
            got_t, got_f, diagnostics = (
                nemo_smagorinsky_bilaplacian_coefficients_cgrid(
                    jnp.asarray(u_input), jnp.asarray(v_input), card.recipe.grid,
                    3.5, card.dt_s, mask=state.land_mask.data,
                    cell_mask=jnp.asarray(cell3), u_mask=state.u_mask.data,
                    v_mask=state.v_mask.data,
                    vertex_mask=coefficient_vertex_mask,
                    return_diagnostics=True,
                )
            )
            got_t, got_f, diagnostics = host_and_clear(
                (got_t, got_f, diagnostics))
            vertex_mask_host = np.asarray(coefficient_vertex_mask, dtype=bool)
            smagorinsky_branch_census.append({
                "kt": kt,
                "input_time_level": "Nbb before velocity",
                "active_definition": (
                    "wet T cells use cell_mask; F cells use the mode-32 "
                    "nonzero coefficient support, including free-slip walls"
                ),
                "branches": coefficient_branch_census(
                    diagnostics,
                    np.asarray(cell3, dtype=bool),
                    np.asarray(got_f != 0.0, dtype=bool),
                ),
            })
            if kt == 10:
                continue
            ahm_path = ahm_root / f"oracle_ahm_kt{kt:08d}.bin"
            if not ahm_path.is_file():
                raise FileNotFoundError(f"ACQUISITION_NEEDED: {ahm_path}")
            ahm = read_ahm(ahm_path, expected_kt=kt)
            ahm_records[kt] = ahm
            full_t = np.ones_like(ahm["ahmt"], dtype=bool)
            full_f = np.ones_like(ahm["ahmf"], dtype=bool)
            spatial = np.any(ahm["ahmf"] != 0.0, axis=-1)
            coords = np.argwhere(spatial)
            if not coords.size:
                raise ValueError(f"{ahm_path}: empty F coefficient support")
            j0, i0 = coords.min(axis=0)
            j1, i1 = coords.max(axis=0)
            edge2 = spatial & (
                (np.indices(spatial.shape)[0] == j0)
                | (np.indices(spatial.shape)[0] == j1)
                | (np.indices(spatial.shape)[1] == i0)
                | (np.indices(spatial.shape)[1] == i1)
            )
            corner2 = spatial & (
                ((np.indices(spatial.shape)[0] == j0)
                 | (np.indices(spatial.shape)[0] == j1))
                & ((np.indices(spatial.shape)[1] == i0)
                   | (np.indices(spatial.shape)[1] == i1))
            )
            regions = {
                "interior": np.broadcast_to((spatial & ~edge2)[..., None], full_f.shape),
                "walls": np.broadcast_to((edge2 & ~corner2)[..., None], full_f.shape),
                "corners": np.broadcast_to(corner2[..., None], full_f.shape),
                "outside": np.broadcast_to((~spatial)[..., None], full_f.shape),
            }
            vertex_mask_float = np.asarray(coefficient_vertex_mask, dtype=np.float64)
            legacy_f = got_f * vertex_mask_float
            smagorinsky_coefficients.append({
                "kt": kt,
                "Kbb": ahm["Kbb"],
                "NEMO_source": str(ahm_path),
                "production": {
                    "ahmt": exact_score(
                        f"{case}.kt{kt}.ahmt", ahm["ahmt"], got_t, full_t),
                    "ahmf": exact_score(
                        f"{case}.kt{kt}.ahmf", ahm["ahmf"], got_f, full_f),
                    "ahmf_mismatch_map": mismatch_map(
                        ahm["ahmf"], got_f, regions,
                        (diagnostics["f_lower_clamped"],
                         diagnostics["f_upper_clamped"]),
                    ),
                },
                "planted_old_fmask_defect": {
                    "ahmf": exact_score(
                        f"control.kt{kt}.legacy_fmask.ahmf",
                        ahm["ahmf"], legacy_f, full_f),
                    "ahmf_mismatch_map": mismatch_map(
                        ahm["ahmf"], legacy_f, regions,
                        (diagnostics["f_lower_clamped"],
                         diagnostics["f_upper_clamped"]),
                    ),
                },
                "support": {
                    "NEMO_nonzero_T": int(np.count_nonzero(ahm["ahmt"])),
                    "NEMO_nonzero_F": int(np.count_nonzero(ahm["ahmf"])),
                    "binary_vertex_mask_nonzero": int(
                        np.count_nonzero(vertex_mask_host)),
                },
            })
        freshwater, surface = gyre_surface_forcings(card, state, kt)
        state_before = state
        state = model.step(state_before, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        if (deck == "r4_smag" and math_flavour == "vector"
                and kt in (2, 3)):
            dyn_path = ahm_root / f"oracle_dynldf_kt{kt:08d}.bin"
            if not dyn_path.is_file():
                raise FileNotFoundError(f"ACQUISITION_NEEDED: {dyn_path}")
            dyn = read_dynldf(dyn_path, expected_kt=kt)

            def exposed_stage3_rhs(kind, coefficient_override=None):
                diagnostic_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_stage3_momentum_rhs=kind,
                        smagorinsky_coefficient_override=coefficient_override,
                    ),
                )
                diagnostic = diagnostic_model.step(
                    state_before, card.dt_s,
                    freshwater=freshwater, surface_forcing=surface,
                )
                return host_and_clear(diagnostic)

            ahm = ahm_records[kt]
            nemo_pair = (jnp.asarray(ahm["ahmt"]), jnp.asarray(ahm["ahmf"]))
            legacy_pair = (
                nemo_pair[0],
                nemo_pair[1] * coefficient_vertex_mask,
            )
            pre = exposed_stage3_rhs("pre_ldf")
            post_nemo = exposed_stage3_rhs("post_ldf", nemo_pair)
            post_legacy = exposed_stage3_rhs("post_ldf", legacy_pair)
            oracle_delta = {
                "u": dyn["u_after"] - dyn["u_before"],
                "v": dyn["v_after"] - dyn["v_before"],
            }
            nemo_delta = {
                "u": lego_fields(post_nemo)["u"] - lego_fields(pre)["u"],
                "v": lego_fields(post_nemo)["v"] - lego_fields(pre)["v"],
            }
            legacy_delta = {
                "u": lego_fields(post_legacy)["u"] - lego_fields(pre)["u"],
                "v": lego_fields(post_legacy)["v"] - lego_fields(pre)["v"],
            }
            substituted_rows = [
                exact_score(
                    f"{case}.kt{kt}.stage3.dynldf.NEMO_coefficient.{field}",
                    oracle_delta[field], nemo_delta[field], masks[field],
                )
                for field in ("u", "v")
            ]
            legacy_rows = [
                exact_score(
                    f"control.kt{kt}.stage3.dynldf.legacy_fmask.{field}",
                    oracle_delta[field], legacy_delta[field], masks[field],
                )
                for field in ("u", "v")
            ]
            substituted_max = max(row["max_abs"] for row in substituted_rows)
            legacy_max = max(row["max_abs"] for row in legacy_rows)
            smagorinsky_operator.append({
                "kt": kt,
                "NEMO_source": str(dyn_path),
                "stage": dyn["stage"],
                "Kbb": dyn["Kbb"],
                "Kmm": dyn["Kmm"],
                "Krhs": dyn["Krhs"],
                "changed_operand": (
                    "stage-3 ahmt/ahmf only; the stage-1 production call is unchanged"
                ),
                "NEMO_coefficient_override": substituted_rows,
                "planted_old_fmask_override": legacy_rows,
                "owner": (
                    "COEFFICIENT: erroneous final F-mask multiplication on mode 32"
                    if (kt == 2 and substituted_max <= 1.0e-20
                        and legacy_max >= 1.0e-10)
                    else (
                        "INCONCLUSIVE: prior trajectory inputs differ before kt=3"
                        if kt == 3 else "UNRESOLVED"
                    )
                ),
                "residual_disposition": (
                    "source-composition residue below 1e-20 m/s2"
                    if kt == 2 and substituted_max <= 1.0e-20 else
                    "the record lacks stage-3 input operands needed to separate "
                    "operator arithmetic from inherited trajectory debt"
                ),
            })
        if kt == 1 and not trajectory_only:
            state = host_and_clear(state)
            for stage in (1, 2):
                stage_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_momentum_stage=stage,
                        expose_tracer_stage=stage,
                    ),
                )
                stage_state = stage_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                stage_state = host_and_clear(stage_state)
                stage_oracle = read_stage(
                    oracle_root / f"oracle_stage_kt{kt:08d}_s{stage}.bin",
                    stage,
                    dims=NEMO_DIMS,
                )
                stage_rows = []
                for field in FIELDS:
                    reference = (
                        stage_oracle[field]
                        if field == "ssh"
                        else stage_oracle[field][..., :30]
                    )
                    stage_rows.append(
                        exact_score(
                            f"{case}.kt1.rk_stage{stage}.{field}",
                            reference,
                            lego_fields(stage_state)[field],
                            masks[field],
                        )
                    )
                rk_stages.append({"stage": stage, "rows": stage_rows})
            stage_oracle = read_stage(
                oracle_root / f"oracle_stage_kt{kt:08d}_s3.bin",
                3,
                dims=NEMO_DIMS,
            )
            stage_rows = []
            for field in FIELDS:
                reference = (
                    stage_oracle[field]
                    if field == "ssh"
                    else stage_oracle[field][..., :30]
                )
                stage_rows.append(
                    exact_score(
                        f"{case}.kt1.rk_stage3.{field}",
                        reference,
                        lego_fields(state)[field],
                        masks[field],
                    )
                )
            rk_stages.append({"stage": 3, "rows": stage_rows})

            if deck == "r4":
                tracer_oracle = read_tracer_stage12(
                    oracle_root / "oracle_rktracer_operands_kt00000001_s1.bin",
                    1,
                    dims=NEMO_DIMS,
                )
                transport_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_tracer_transport_stage=1,
                    ),
                )
                transport_state = transport_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                transport_state = host_and_clear(transport_state)
                transport_values = {
                    "zFu": np.asarray(transport_state.u.data)[:, 1:, :],
                    "zFv": np.asarray(transport_state.v.data)[1:, :, :],
                    "zFw": np.asarray(transport_state.T.data),
                }
                tracer_stage1_transport = [
                    exact_score(
                        f"{case}.kt1.stage1.transport.{name}",
                        tracer_oracle[name][..., :30],
                        transport_values[name],
                        masks[{"zFu": "u", "zFv": "v", "zFw": "T"}[name]],
                    )
                    for name in ("zFu", "zFv", "zFw")
                ]
                boundary_candidates = {}
                for boundary in ("after_advection", "after_sbc"):
                    boundary_model = LatLonCGridOceanModel(
                        card.recipe.grid,
                        card.recipe.z_coord,
                        card.recipe.model_config,
                        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                            expose_tracer_stage1_boundary=boundary,
                        ),
                    )
                    boundary_state = boundary_model.step(
                        state_before,
                        card.dt_s,
                        freshwater=freshwater,
                        surface_forcing=surface,
                    )
                    boundary_state = host_and_clear(boundary_state)
                    boundary_fields = lego_fields(boundary_state)
                    boundary_candidates[boundary] = boundary_fields
                    tracer_stage1_walk.append({
                        "boundary": boundary,
                        "rows": [
                            exact_score(
                                f"{case}.kt1.stage1.{boundary}.{field}",
                                tracer_oracle[f"{boundary}_{field}"][..., :30],
                                boundary_fields[field],
                                masks[field],
                            )
                            for field in ("T", "S")
                        ],
                    })

                # Compare the increment itself, not just the two endpoint
                # maxima.  Equal endpoint maxima do not prove that SBC added
                # no error because the unequal cells and signs may differ.
                sbc_increment_rows = []
                for field in ("T", "S"):
                    oracle_increment = (
                        tracer_oracle[f"after_sbc_{field}"][..., :30]
                        - tracer_oracle[f"after_advection_{field}"][..., :30]
                    )
                    candidate_increment = (
                        boundary_candidates["after_sbc"][field]
                        - boundary_candidates["after_advection"][field]
                    )
                    sbc_increment_rows.append(exact_score(
                        f"{case}.kt1.stage1.sbc_increment.{field}",
                        oracle_increment, candidate_increment, masks[field],
                    ))

                # One-variable experiment required by the scalar kt=2 stop
                # rule: inject NEMO's recorded stage-1 zF triplet into the
                # tracer consumer while leaving momentum and later stages on
                # production operands.  zFu/zFv are already exact, so this
                # changes only the observed zFw debt.
                oracle_transport = (
                    jnp.asarray(tracer_oracle["zFu"][..., :30]),
                    jnp.asarray(tracer_oracle["zFv"][..., :30]),
                    jnp.asarray(tracer_oracle["zFw"]),
                )
                override_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        stage1_tracer_transport_override=oracle_transport,
                    ),
                )
                override_state = override_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                override_state = host_and_clear(override_state)
                override_fields = lego_fields(override_state)
                oracle2 = read_entry(
                    oracle_root / "oracle_step_entry_kt00000002.bin",
                    dims=NEMO_DIMS,
                )
                override_rows = []
                for field in FIELDS:
                    reference = (
                        oracle2[field]
                        if field == "ssh" else oracle2[field][..., :30]
                    )
                    override_rows.append(exact_score(
                        f"{case}.kt2.NEMO_stage1_transport_override.{field}",
                        reference, override_fields[field], masks[field],
                    ))
                tracer_stage1_transport_discriminator = {
                    "changed_operand": (
                        "stage-1 tracer zFu/zFv/zFw triplet only; zFu/zFv "
                        "are independently exact, so the effective changed "
                        "operand is zFw"
                    ),
                    "sbc_increment_rows": sbc_increment_rows,
                    "NEMO_transport_override_rows": override_rows,
                }

                # The first prognostic split against scalar NEMO is stage-2
                # U/V, not the stage-1 tracer accumulator.  Compare NEMO's
                # source-ordered operator increments with legoESM's own
                # already-computed components, then substitute the complete
                # NEMO RHS at exactly the stage update boundary.
                term_oracle = read_stage2_terms(
                    oracle_root / "oracle_rkstage2_terms_kt00000001.bin",
                    dims=NEMO_DIMS,
                )

                def diagnostic_fields(**hook_args):
                    diagnostic_model = LatLonCGridOceanModel(
                        card.recipe.grid,
                        card.recipe.z_coord,
                        card.recipe.model_config,
                        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_args),
                    )
                    diagnostic = diagnostic_model.step(
                        state_before,
                        card.dt_s,
                        freshwater=freshwater,
                        surface_forcing=surface,
                    )
                    return lego_fields(host_and_clear(diagnostic))

                operator_rows = []
                boundaries = {
                    "hpg": ("before", "after_hpg"),
                    "vorticity": ("after_hpg", "after_vorticity"),
                    "advection": ("after_vorticity", "after_advection"),
                }
                for operator, (left, right) in boundaries.items():
                    exposed = diagnostic_fields(
                        expose_momentum_operator=operator,
                        expose_momentum_operator_stage=2,
                    )
                    rows = []
                    for field in ("u", "v"):
                        rows.append(accumulated_component_score(
                            f"{case}.kt1.stage2."
                            f"{operator}_accumulation.{field}",
                            term_oracle[f"{left}_{field}"][..., :30],
                            term_oracle[f"{right}_{field}"][..., :30],
                            exposed[field], masks[field],
                        ))
                    operator_rows.append({"operator": operator, "rows": rows})

                total_fields = diagnostic_fields(expose_stage2_momentum_rhs=True)
                total_rows = [
                    exact_score(
                        f"{case}.kt1.stage2.completed_rhs.{field}",
                        term_oracle[f"after_advection_{field}"][..., :30],
                        total_fields[field], masks[field],
                    )
                    for field in ("u", "v")
                ]
                oracle_rhs = _redundant_faces(
                    term_oracle["after_advection_u"],
                    term_oracle["after_advection_v"],
                )
                overridden_stage2 = diagnostic_fields(
                    stage2_momentum_rhs_override=(
                        jnp.asarray(oracle_rhs[0]), jnp.asarray(oracle_rhs[1])),
                    expose_momentum_stage=2,
                )
                stage2_oracle = read_stage(
                    oracle_root / "oracle_stage_kt00000001_s2.bin",
                    2,
                    dims=NEMO_DIMS,
                )
                override_stage_rows = [
                    exact_score(
                        f"{case}.kt1.stage2.NEMO_rhs_override.{field}",
                        stage2_oracle[field][..., :30],
                        overridden_stage2[field], masks[field],
                    )
                    for field in ("u", "v")
                ]
                composition_oracle = read_stage2_composition(
                    oracle_root / "oracle_rkstage2_operands_kt00000001.bin",
                    dims=NEMO_DIMS,
                )
                raw_fields = diagnostic_fields(expose_stage2_raw_momentum=True)
                overridden_raw = diagnostic_fields(
                    stage2_momentum_rhs_override=(
                        jnp.asarray(oracle_rhs[0]), jnp.asarray(oracle_rhs[1])),
                    expose_stage2_raw_momentum=True,
                )
                raw_rows = []
                override_raw_rows = []
                for field in ("u", "v"):
                    reference = composition_oracle[f"raw_Kaa_{field}"][..., :30]
                    raw_rows.append(exact_score(
                        f"{case}.kt1.stage2.raw_update.{field}",
                        reference, raw_fields[field], masks[field],
                    ))
                    override_raw_rows.append(exact_score(
                        f"{case}.kt1.stage2.NEMO_rhs_raw_update.{field}",
                        reference, overridden_raw[field], masks[field],
                    ))
                stage2_momentum_term_walk = {
                    "operator_increment_rows": operator_rows,
                    "completed_rhs_rows": total_rows,
                    "raw_update_rows": raw_rows,
                    "NEMO_completed_rhs_raw_update_rows": override_raw_rows,
                    "NEMO_completed_rhs_override_rows": override_stage_rows,
                }

            if deck == "r4":
                import jax.numpy as jnp

                zdf = read_zdf_entry(
                    oracle_root / "oracle_zdf_entry_kt00000001.bin",
                    dims=NEMO_DIMS,
                )
                seeded = model._seed_tke_preclosure_carry(state_before)
                model.prime_step_caches(seeded)
                lego_avt, lego_avm = model.diagnose_vertical_K(
                    seeded, card.dt_s, surface
                )
                lego_avt, lego_avm = host_and_clear((lego_avt, lego_avm))
                interface_mask = (
                    np.asarray(card.recipe.z_coord.is_active)[..., 1:]
                    & (np.asarray(seeded.land_mask.data) > 0.5)[..., None]
                )
                coefficient_rows = [
                    exact_score(
                        f"{case}.kt1.zdf_entry.avt",
                        zdf["avt"][..., 1:30],
                        lego_avt,
                        interface_mask,
                    ),
                    exact_score(
                        f"{case}.kt1.zdf_entry.avm",
                        zdf["avm"][..., 1:30],
                        lego_avm,
                        interface_mask,
                    ),
                ]
                vertical_control_state = model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                    _vertical_K_test_override=(
                        jnp.asarray(zdf["avt"][..., 1:30]),
                        jnp.asarray(zdf["avm"][..., 1:30]),
                    ),
                )
                vertical_control_state = host_and_clear(vertical_control_state)
                oracle2 = read_entry(
                    oracle_root / "oracle_step_entry_kt00000002.bin",
                    dims=NEMO_DIMS,
                )
                faithful = lego_fields(state)
                controlled = lego_fields(vertical_control_state)
                arm_rows = []
                for field in FIELDS:
                    reference = oracle2[field] if field == "ssh" else oracle2[field][..., :30]
                    arm_rows.append(
                        {
                            "field": field,
                            "faithful": exact_score(
                                f"{case}.kt2.faithful.{field}",
                                reference,
                                faithful[field],
                                masks[field],
                            ),
                            "oracle_zdf_override": exact_score(
                                f"{case}.kt2.oracle_zdf_override.{field}",
                                reference,
                                controlled[field],
                                masks[field],
                            ),
                        }
                    )
                vertical_candidate = {
                    "candidate": "stage-entry TKE/EVD vertical coefficients",
                    "changed_operand": "avt/avm profile bundle only",
                    "coefficient_rows": coefficient_rows,
                    "one_variable_arm": arm_rows,
                }

            if deck == "r4":
                bt_oracle = read_bt_substeps(
                    oracle_root / "oracle_bt_substeps_kt00000001.bin",
                    expected_dims=NEMO_DIMS[:2],
                )
                trace_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_barotropic_substeps=True,
                    ),
                )
                trace = trace_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                trace = host_and_clear(trace)
                ordered = (
                    "eta_entry", "u_entry", "v_entry",
                    "eta_mid", "u_mid", "v_mid", "eta_exit", "eta_pgf",
                    "pgf_u", "pgf_v", "cor_u", "cor_v", "trd_u", "trd_v",
                    "slow_u", "slow_v", "u_exit", "v_exit",
                    "transport_metric_u", "transport_metric_v",
                )

                def bt_mask(name):
                    if name.startswith("eta"):
                        return masks["ssh"]
                    if name.startswith("u_") or name.endswith("_u"):
                        return masks["u"][..., 0]
                    return masks["v"][..., 0]

                first = None
                for substep in range(bt_oracle["ncycle"]):
                    for name in ordered:
                        reference = bt_oracle[name][substep]
                        candidate = _trace_native(bt_frame(trace.substeps, name), name)[substep]
                        row = exact_score(
                            f"{case}.kt1.bt.jn{substep + 1:02d}.{name}",
                            reference, candidate, bt_mask(name),
                        )
                        if row["status"] == "DEBT":
                            active = bt_mask(name)
                            unequal = reference[active].view(np.uint64) != candidate[active].view(np.uint64)
                            first_index = tuple(np.argwhere(active)[np.flatnonzero(unequal)[0]])
                            first = {
                                "substep": substep + 1,
                                "field": name,
                                "row": row,
                                "first_active_index": [int(index) for index in first_index],
                                "first_ulp": int(ulp_distance(
                                    np.asarray([reference[first_index]]),
                                    np.asarray([candidate[first_index]]),
                                )[0]),
                            }
                            break
                    if first is not None:
                        break

                # One-variable discriminator: replace only the frozen slow
                # forcing at the external-solver boundary with NEMO's recorded
                # value.  This is a private test hook, not a card selector.
                oracle_slow_u = np.asarray(trace.slow_forcing[1]).copy()
                oracle_slow_v = np.asarray(trace.slow_forcing[2]).copy()
                oracle_slow_u[:, 1:] = bt_oracle["slow_u"][0]
                oracle_slow_v[1:, :] = bt_oracle["slow_v"][0]
                control_model = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                        expose_barotropic_substeps=True,
                        barotropic_slow_forcing_override=(
                            jnp.asarray(oracle_slow_u), jnp.asarray(oracle_slow_v),
                        ),
                    ),
                )
                controlled = control_model.step(
                    state_before,
                    card.dt_s,
                    freshwater=freshwater,
                    surface_forcing=surface,
                )
                controlled = host_and_clear(controlled)
                control_rows = []
                for name in (
                    "slow_u", "slow_v", "u_exit", "v_exit",
                    "transport_metric_u", "transport_metric_v",
                ):
                    control_rows.append(exact_score(
                        f"{case}.kt1.bt.jn01.oracle_slow_override.{name}",
                        bt_oracle[name][0],
                        _trace_native(bt_frame(controlled.substeps, name), name)[0],
                        bt_mask(name),
                    ))
                depth_only_u = np.asarray(trace.slow_forcing_operands["depth_u"])[:, 1:]
                depth_only_v = np.asarray(trace.slow_forcing_operands["depth_v"])[1:, :]
                operand_u = trace.slow_forcing_operands
                wind_arms = {}
                for face, native, active in (
                    ("u", lambda value: np.asarray(value)[:, 1:], masks["u"][..., 0]),
                    ("v", lambda value: np.asarray(value)[1:, :], masks["v"][..., 0]),
                ):
                    arms = wind_expression_arms(
                        operand_u["wind_r1_rho0"],
                        native(operand_u[f"wind_tau_{face}"]),
                        native(operand_u[f"wind_r1_h{face}"]),
                        native(operand_u[f"H_{face}"]),
                        active,
                    )
                    wind_arms[face] = {
                        name: exact_score(
                            f"{case}.kt1.wind.{face}.{name}",
                            bt_oracle[f"slow_{face}"][0], value, active,
                        )
                        for name, value in arms.items()
                        if name != "stored_reciprocal"
                    }
                    production_slow = native(
                        trace.slow_forcing[1 if face == "u" else 2])
                    wind_arms[face]["production_slow"] = exact_score(
                        f"{case}.kt1.wind.{face}.production_slow",
                        bt_oracle[f"slow_{face}"][0], production_slow, active)
                    wind_arms[face]["reciprocal"] = exact_score(
                        f"{case}.kt1.wind.{face}.stored_vs_current_reciprocal",
                        arms["stored_reciprocal"],
                        native(operand_u[f"wind_r1_h{face}"]),
                        active,
                    )
                math_program = compiled_usrdef_math_program(
                    oracle_root / "nemo", expected_flavour=math_flavour)
                if math_flavour == "scalar":
                    wind_owner = (
                        "CLOSED_FOR_SCALAR_ORACLE: production slow and NEMO's "
                        "written left-associated expression are bit-exact; the "
                        "right-associated and reciprocal-first controls differ. "
                        "The admitted usrdef_sbc routine calls scalar libm sin."
                    )
                else:
                    wind_owner = (
                        "UPSTREAM_TAU_POLICY: production slow equals the "
                        "reconstructed left-associated scalar-libm composite, "
                        "but remains in debt to NEMO because the admitted "
                        "usrdef_sbc routine calls GNU vector sin"
                    )
                barotropic_resolution_walk = {
                    "first_difference": first,
                    "term": (
                        "stp2d wind term (r1_rho0 * utau) * r1_hu_0, already "
                        "inside frozen Ue_rhs/Ve_rhs before dynspg_ts substep 1"
                    ),
                    "raw_operand_ownership": wind_owner,
                    "rest_state_discriminator": {
                        "depth_only_u_max_abs": float(np.max(np.abs(depth_only_u))),
                        "depth_only_v_max_abs": float(np.max(np.abs(depth_only_v))),
                        "oracle_slow_override_rows": control_rows,
                    },
                    "wind_expression_discriminator": wind_arms,
                    "compiled_usrdef_math_program": math_program,
                    "arithmetic": {
                        "rn_Dt_s": card.dt_s,
                        "nn_e": card.recipe.model_config.barotropic.n_barotropic_substeps,
                        "substep_dt_s": card.dt_s / card.recipe.model_config.barotropic.n_barotropic_substeps,
                        "substep_reciprocal": 1.0 / (
                            card.dt_s / card.recipe.model_config.barotropic.n_barotropic_substeps
                        ),
                        "shipped_substep_dt_s": 14400.0 / 50.0,
                        "shipped_substep_reciprocal": 1.0 / (14400.0 / 50.0),
                        "grid_spacing_expression_m": 106000.0 / 4.0,
                        "grid_spacing_reciprocal": 1.0 / (106000.0 / 4.0),
                        "transport_weight_first_three": np.asarray(
                            trace.substeps["transport_weight"]
                        )[:3].tolist(),
                        "transport_weight_normalization": "r1_wgt2s from sum(wgtbtp2)",
                        "bottom_drag_at_substep1": "zero: entry velocity is rest",
                        "pgf_at_substep1": "bit-equal through pgf_u/pgf_v before slow_u/slow_v",
                    },
                }
    passed = all(row["status"] == "AT-BAR" for step in steps for row in step["rows"]) and all(
        control["status"] == "FIRED" for control in controls
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-exact-ladder-v1",
        "worktree": worktree_stamp(),
        "case": case,
        "deck": deck,
        "oracle_admission": admission,
        "oracle_math_flavour": math_flavour,
        "oracle_root": str(oracle_root),
        "ahm_oracle_root": str(ahm_root) if deck == "r4_smag" else None,
        "policy": "fp64/x64/cpu/libm certified",
        "acceptance": "zero unequal active float64 values in every field/step",
        "namelist": namelist._asdict(),
        "steps": steps,
        "rk_stage_walk": rk_stages,
        "tracer_stage1_transport": tracer_stage1_transport,
        "tracer_stage1_walk": tracer_stage1_walk,
        "tracer_stage1_transport_discriminator": (
            tracer_stage1_transport_discriminator),
        "stage2_momentum_term_walk": stage2_momentum_term_walk,
        "vertical_candidate": vertical_candidate,
        "barotropic_resolution_walk": barotropic_resolution_walk,
        "smagorinsky_coefficients": smagorinsky_coefficients,
        "smagorinsky_branch_census": smagorinsky_branch_census,
        "smagorinsky_operator": smagorinsky_operator,
        "controls": controls,
        "trajectory_only": trajectory_only,
        "verdict": "PASS" if passed else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_stage2_composition_only(
    oracle_root: Path, output: Path, *, math_flavour: str = "scalar",
) -> dict:
    """Run the narrow source/update controls without the ten-step ladder."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree composition gate requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_oracle("r4", math_flavour, oracle_root)
    card = build_nemo_testcase_card(
        "GYRE-zco",
        namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=10, nn_stock=10, nn_write=10),
    )
    state = card.recipe.initial_state
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    terms = read_stage2_terms(
        oracle_root / "oracle_rkstage2_terms_kt00000001.bin",
        dims=NEMO_DIMS,
    )
    composition = read_stage2_composition(
        oracle_root / "oracle_rkstage2_operands_kt00000001.bin",
        dims=NEMO_DIMS,
    )
    oracle_rhs = _redundant_faces(
        terms["after_advection_u"], terms["after_advection_v"])

    def run_model(**hook_args):
        model = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_args),
        )
        diagnostic = model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        return jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf)
            if isinstance(leaf, jax.Array) else leaf,
            diagnostic,
        )

    masks = expected_masks(card)
    production_raw = lego_fields(run_model(expose_stage2_raw_momentum=True))
    overridden_raw = lego_fields(run_model(
        stage2_momentum_rhs_override=(
            jnp.asarray(oracle_rhs[0]), jnp.asarray(oracle_rhs[1])),
        expose_stage2_raw_momentum=True,
    ))
    production_rows = []
    override_rows = []
    controls = []
    for field in ("u", "v"):
        reference = composition[f"raw_Kaa_{field}"][..., :30]
        production_rows.append(exact_score(
            f"GYRE-zco-r4.kt1.stage2.production_raw_update.{field}",
            reference, production_raw[field], masks[field],
        ))
        override_rows.append(exact_score(
            f"GYRE-zco-r4.kt1.stage2.NEMO_rhs_raw_update.{field}",
            reference, overridden_raw[field], masks[field],
        ))
        planted = exact_score(
            f"control.one_ulp.raw_update.{field}",
            reference, reference, masks[field], plant_one_ulp=True,
        )
        controls.append({
            **planted,
            "status": "FIRED" if planted["n_unequal"] == 1 else "MISSED",
        })

    # Prove the private RHS injection is consumed, rather than merely
    # returning a coincidentally exact diagnostic. Perturb one wet native-U
    # RHS value by a material fixed amount and require the raw update to move.
    planted_rhs = [np.array(value, copy=True) for value in oracle_rhs]
    wet_u_index = tuple(int(value) for value in np.argwhere(masks["u"])[0])
    internal_u_index = (
        wet_u_index[0], wet_u_index[1] + 1, wet_u_index[2])
    rhs_delta = np.float64(1.0e-6)
    planted_rhs[0][internal_u_index] += rhs_delta
    planted_raw = lego_fields(run_model(
        stage2_momentum_rhs_override=(
            jnp.asarray(planted_rhs[0]), jnp.asarray(planted_rhs[1])),
        expose_stage2_raw_momentum=True,
    ))
    rhs_effect_rows = [
        exact_score(
            f"control.material_rhs_override.{field}",
            overridden_raw[field], planted_raw[field], masks[field],
        )
        for field in ("u", "v")
    ]
    rhs_effect_fired = any(row["n_unequal"] > 0 for row in rhs_effect_rows)

    # Score each isolated legoESM term through the actual accumulator
    # boundary. This avoids attempting to reconstruct NEMO's addend by
    # subtracting two rounded accumulator snapshots.
    boundaries = {
        "hpg": ("before", "after_hpg"),
        "vorticity": ("after_hpg", "after_vorticity"),
        "advection": ("after_vorticity", "after_advection"),
    }
    accumulation_rows = []
    for operator, (left, right) in boundaries.items():
        exposed = lego_fields(run_model(
            expose_momentum_operator=operator,
            expose_momentum_operator_stage=2,
        ))
        rows = [
            accumulated_component_score(
                f"GYRE-zco-r4.kt1.stage2."
                f"{operator}_accumulation.{field}",
                terms[f"{left}_{field}"][..., :30],
                terms[f"{right}_{field}"][..., :30],
                exposed[field], masks[field],
            )
            for field in ("u", "v")
        ]
        accumulation_rows.append({"boundary": operator, "rows": rows})
    first_accumulation_debt = next(
        (
            {
                "boundary": group["boundary"],
                "fields": [
                    row["name"].rsplit(".", 1)[-1]
                    for row in group["rows"] if row["status"] == "DEBT"
                ],
            }
            for group in accumulation_rows
            if any(row["status"] == "DEBT" for row in group["rows"])
        ),
        None,
    )

    # Prove the stage-1 zF substitution reaches the tracer consumer. The
    # scientific arm substitutes NEMO's zFw and does not move the kt=2
    # maxima; this control changes one active zFw by 1 m3/s and requires a
    # downstream prognostic tracer to move.
    tracer_stage1 = read_tracer_stage12(
        oracle_root / "oracle_rktracer_operands_kt00000001_s1.bin",
        1, dims=NEMO_DIMS,
    )
    transport = [
        np.array(tracer_stage1[name][..., :30]
                 if name != "zFw" else tracer_stage1[name], copy=True)
        for name in ("zFu", "zFv", "zFw")
    ]
    baseline_transport_state = lego_fields(run_model(
        stage1_tracer_transport_override=tuple(jnp.asarray(x) for x in transport),
    ))
    zfw_index = next(
        tuple(int(value) for value in index)
        for index in np.argwhere(masks["T"])
        if index[2] > 0
    )
    transport[2][zfw_index] += np.float64(1.0)
    planted_transport_state = lego_fields(run_model(
        stage1_tracer_transport_override=tuple(jnp.asarray(x) for x in transport),
    ))
    zfw_effect_rows = [
        exact_score(
            f"control.material_zFw_override.{field}",
            baseline_transport_state[field], planted_transport_state[field],
            masks[field],
        )
        for field in ("T", "S")
    ]
    zfw_effect_fired = any(row["n_unequal"] > 0 for row in zfw_effect_rows)

    controls.extend([
        {
            "name": "control.material_rhs_override",
            "status": "FIRED" if rhs_effect_fired else "MISSED",
            "delta": float(rhs_delta),
            "index": list(internal_u_index),
            "rows": rhs_effect_rows,
        },
        {
            "name": "control.material_zFw_override",
            "status": "FIRED" if zfw_effect_fired else "MISSED",
            "delta_m3_s": 1.0,
            "index": list(zfw_index),
            "rows": zfw_effect_rows,
        },
    ])
    report = {
        "format": "nemo-testcase-l2-gyre-r4-narrow-discriminator-v2",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "policy": "fp64/x64/cpu/libm certified",
        "changed_operand": "complete NEMO stage-2 momentum RHS only",
        "comparison_boundary": (
            "raw Kaa immediately after the RK update and before barotropic "
            "mean replacement"
        ),
        "production_raw_rows": production_rows,
        "NEMO_rhs_override_raw_rows": override_rows,
        "source_order_accumulation_rows": accumulation_rows,
        "first_nonexact_accumulation_boundary": first_accumulation_debt,
        "mean_replacement_boundary": {
            "status": "OPEN_INPUT_VERSUS_ARITHMETIC",
            "reason": (
                "the NEMO-RHS raw update is exact, while the recorded post-"
                "replacement state is not; recorded barotropic targets and "
                "zub/zvb have not yet been substituted"
            ),
        },
        "controls": controls,
        "verdict": (
            "RAW_UPDATE_AT_BAR"
            if all(row["status"] == "AT-BAR" for row in override_rows)
            and all(control["status"] == "FIRED" for control in controls)
            else "DEBT"
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_stage3_geometry_only(
    oracle_root: Path, output: Path, *, math_flavour: str = "scalar",
) -> dict:
    """Bound a post-hoc ratio replay; do not call it executed NEMO arithmetic."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    if jax.default_backend() != "cpu":
        raise RuntimeError("quarter-degree geometry gate requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_oracle("r4", math_flavour, oracle_root)
    card = build_nemo_testcase_card(
        "GYRE-zco",
        namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=10, nn_stock=10, nn_write=10),
    )
    state = card.recipe.initial_state
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(tracer_process_trace=()),
    )
    trace = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    trace = jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf)
        if isinstance(leaf, jax.Array) else leaf,
        trace,
    )
    oracle = read_tracer_stage3_hires(
        oracle_root / "oracle_rktracer_stage3_kt00000001.bin",
        dims=NEMO_DIMS,
    )
    masks = expected_masks(card)
    mask3 = masks["T"]
    mask2 = np.any(mask3, axis=-1)
    tbb = np.asarray(trace.Tbb)[..., :30]
    qbb = np.asarray(trace.q_Kbb)
    qmm = np.asarray(trace.q_Kmm)
    qaa = np.asarray(trace.q_Kaa)
    oracle_tbb = oracle["Kbb_T"][..., :30]
    oracle_qbb = np.float64(1.0) + oracle["r3t_Kbb"]
    oracle_qmm = np.float64(1.0) + oracle["r3t_Kmm"]
    oracle_qaa = np.float64(1.0) + oracle["r3t_Kaa"]
    operand_rows = [
        exact_score("stage3.geometry.Tbb", oracle_tbb, tbb, mask3),
        exact_score("stage3.geometry.q_Kbb", oracle_qbb, qbb, mask2),
        exact_score("stage3.geometry.q_Kmm", oracle_qmm, qmm, mask2),
        exact_score("stage3.geometry.q_Kaa", oracle_qaa, qaa, mask2),
    ]
    oracle_geometry = (
        oracle_qbb[..., None] * oracle_tbb / oracle_qaa[..., None])
    literal_ratio = qbb[..., None] * tbb / qaa[..., None]
    e3t_0 = np.asarray(card.recipe.z_coord.nemo_e3t_0)[..., :30]
    thickness_association = (
        (e3t_0 * qbb[..., None]) * tbb
        / (e3t_0 * qaa[..., None]))
    production_geometry = np.asarray(trace.boundaries[0])[..., :30]
    composition_rows = [
        exact_score(
            "stage3.geometry.literal_ratio_replay",
            oracle_geometry, literal_ratio, mask3),
        exact_score(
            "stage3.geometry.production_reference_thickness",
            oracle_geometry, production_geometry, mask3),
        exact_score(
            "stage3.geometry.reference_thickness_reconstruction",
            production_geometry, thickness_association, mask3),
    ]
    planted_qbb = np.array(qbb, copy=True)
    plant_index = tuple(int(value) for value in np.argwhere(mask2)[0])
    planted_qbb[plant_index] = np.nextafter(
        planted_qbb[plant_index], np.inf)
    planted_literal = planted_qbb[..., None] * tbb / qaa[..., None]
    plant = exact_score(
        "control.one_ulp.q_Kbb_replay",
        literal_ratio, planted_literal, mask3)
    replay_bounded = (
        all(row["status"] == "AT-BAR" for row in operand_rows)
        and composition_rows[0]["status"] == "AT-BAR"
        and composition_rows[2]["status"] == "AT-BAR"
        and plant["status"] == "DEBT")
    report = {
        "format": "nemo-testcase-l2-gyre-r4-stage3-geometry-v1",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "policy": "fp64/x64/cpu/libm certified",
        "operand_rows": operand_rows,
        "composition_rows": composition_rows,
        "one_variable_replay": {
            "changed_expression": (
                "remove the algebraically cancelling e3t_0 factor from the "
                "stage-3 Kbb-to-Kaa geometry association"
            ),
            "control": {
                **plant,
                "status": "FIRED" if plant["status"] == "DEBT" else "MISSED",
                "index": list(plant_index),
            },
        },
        "interpretation": (
            "POSTHOC_RATIO_REPLAY_ONLY: compiled trazdf.f90:284-289 "
            "retains e3t_0 in the executed RHS; the current record does not "
            "contain its matrix/RHS operands"
        ),
        "owner": "UNMEASURED_EXECUTED_TRAZDF_OPERANDS",
        "verdict": "REPORT_ONLY" if replay_bounded else "DEBT",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_step2_tracer_owner(oracle_root: Path, output: Path) -> dict:
    """Replay the first material quarter-degree tracer split at stage 3."""
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.physics.shortwave_penetration import (
        shortwave_penetration_tendency,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian
    from nemo_testcase_l2_gyre_round35_trazdf_matrix import (
        read_trazdf_matrix,
    )

    if jax.default_backend() != "cpu":
        raise RuntimeError("step-2 tracer discriminator requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_oracle("r4_step2", "scalar", oracle_root)
    namelist = GYRE_R4_NAMELIST._replace(
        nn_itend=10, nn_stock=10, nn_write=10)
    card = build_nemo_testcase_card("GYRE-zco", namelist=namelist)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    state = ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)

    # Preserve legoESM's hidden closure/RK memories, but bridge every primary
    # NEMO before-level field.  The following source-order comparison is thus
    # a step-2 operator walk, not a compounding step-1 trajectory comparison.
    entry = read_entry(
        oracle_root / "oracle_step_entry_kt00000002.bin", dims=NEMO_DIMS)
    u, v = _redundant_faces(entry["u"], entry["v"])
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"][..., :30])),
        S=state.S.replace(data=jnp.asarray(entry["S"][..., :30])),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"])),
    )
    freshwater, surface = gyre_surface_forcings(card, state, 2)
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True),
    ).step(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    traced = jax.device_get(traced)

    # Measure every RK boundary instead of inferring stage 3 from the first
    # detailed tracer record.  The hooks expose the stage T/S and momentum
    # only after the ordinary production step has completed, so these runs do
    # not perturb the stage being measured.
    wet_masks = expected_masks(card)
    rk_stage_walk = []
    for stage in (1, 2):
        stage_state = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_stage=stage,
                expose_tracer_stage=stage,
            ),
        ).step(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
        stage_state = jax.device_get(stage_state)
        stage_oracle = read_stage(
            oracle_root / f"oracle_stage_kt00000002_s{stage}.bin",
            stage,
            dims=NEMO_DIMS,
            expected_kt=2,
            expected_kaa=STEP2_STAGE_KAA[stage],
        )
        stage_fields = lego_fields(stage_state)
        rows = [
            exact_score(
                f"GYRE-zco-r4.kt2.rk_stage{stage}.{field}",
                (stage_oracle[field] if field == "ssh"
                 else stage_oracle[field][..., :30]),
                stage_fields[field],
                wet_masks[field],
            )
            for field in FIELDS
        ]
        rk_stage_walk.append({"stage": stage, "rows": rows})
        jax.clear_caches()

    stage_oracle = read_stage(
        oracle_root / "oracle_stage_kt00000002_s3.bin",
        3,
        dims=NEMO_DIMS,
        expected_kt=2,
        expected_kaa=STEP2_STAGE_KAA[3],
    )
    stage_fields = lego_fields(traced.state_after)
    rk_stage_walk.append({
        "stage": 3,
        "rows": [
            exact_score(
                f"GYRE-zco-r4.kt2.rk_stage3.{field}",
                (stage_oracle[field] if field == "ssh"
                 else stage_oracle[field][..., :30]),
                stage_fields[field],
                wet_masks[field],
            )
            for field in FIELDS
        ],
    })
    first_any_rk = first_rk_difference(rk_stage_walk)
    first_owner_scale_rk = first_rk_difference(
        rk_stage_walk, minimum_max_abs=1.0e-10)

    oracle = read_tracer_stage3_hires(
        oracle_root / "oracle_rktracer_stage3_kt00000002.bin",
        expected_kt=2,
    )
    qsr_record = read_qsr_stage3_hires(
        oracle_root / "oracle_qsr_stage3_kt00000002.bin", expected_kt=2)
    zdf_record = read_trazdf_matrix(
        oracle_root / "oracle_trazdf_matrix_kt00000002.bin",
        expect_kt=2, expected_domain=NEMO_DIMS)
    wet = expected_masks(card)["T"]
    e3 = np.asarray(card.recipe.z_coord.dz_ref, dtype=np.float64)
    qbb = np.asarray(1.0 + oracle["r3t_Kbb"], dtype=np.float64)
    qmm = np.asarray(1.0 + oracle["r3t_Kmm"], dtype=np.float64)
    qaa = np.asarray(1.0 + oracle["r3t_Kaa"], dtype=np.float64)
    base = qbb[..., None] * oracle["Kbb_T"][..., :30]
    nemo_boundaries = {
        "geometry": base / qaa[..., None],
        "advection": (
            base + card.dt_s * qmm[..., None]
            * oracle["after_advection_T"][..., :30]) / qaa[..., None],
        "surface_boundary": (
            base + card.dt_s * qmm[..., None]
            * oracle["after_sbc_T"][..., :30]) / qaa[..., None],
        "shortwave": (
            base + card.dt_s * qmm[..., None]
            * oracle["after_qsr_T"][..., :30]) / qaa[..., None],
        "lateral_diffusion": (
            base + card.dt_s * qmm[..., None]
            * oracle["after_ldf_T"][..., :30]) / qaa[..., None],
    }
    b0, badv, bsbc, bqsr, bldf, _ = (
        np.asarray(value, dtype=np.float64) for value in traced.boundaries)
    lego_boundaries = {
        "geometry": b0,
        "advection": badv,
        "surface_boundary": bsbc,
        "shortwave": bqsr,
        "lateral_diffusion": bldf,
    }
    boundary_rows = [
        exact_score(
            f"GYRE-zco-r4.kt2.stage3.{name}.T",
            nemo_boundaries[name], lego_boundaries[name], wet)
        for name in nemo_boundaries
    ]

    # Replay the removed arm with exactly one changed variable: the Kbb
    # attenuation/thickness stretch is static one instead of live 1+r3t(Kbb).
    live_stretch = np.asarray(compute_ocean_jacobian(
        state.eta.data, state.H_bathy.data, card.recipe.z_coord))
    live_qsr = np.asarray(shortwave_penetration_tendency(
        surface.sw_down, card.recipe.z_coord.dz_ref,
        card.recipe.z_coord.z_half_ref, jnp.asarray(live_stretch),
        card.recipe.model_config.physics.shortwave_penetration,
        rho_0=card.recipe.model_config.rho_0,
        c_sw=card.recipe.model_config.physics.constants.c_sw,
        z_half_stretch=jnp.asarray(live_stretch),
    ))
    static_qsr = np.asarray(shortwave_penetration_tendency(
        surface.sw_down, card.recipe.z_coord.dz_ref,
        card.recipe.z_coord.z_half_ref, jnp.asarray(live_stretch),
        card.recipe.model_config.physics.shortwave_penetration,
        rho_0=card.recipe.model_config.rho_0,
        c_sw=card.recipe.model_config.physics.constants.c_sw,
    ))
    reverted_shortwave = (
        bqsr + card.dt_s * np.asarray(traced.qsr_association.thickness_kbb)
        * (static_qsr - live_qsr)
        / np.maximum(qaa[..., None] * e3, 1.0e-10)
    )
    live_row = exact_score(
        "one_variable.live_Kbb_stretch", nemo_boundaries["shortwave"],
        bqsr, wet)
    static_row = exact_score(
        "one_variable.static_Kbb_stretch", nemo_boundaries["shortwave"],
        reverted_shortwave, wet)
    qsr_increment_row = exact_score(
        "NEMO.executed_qsr_increment_vs_recorded_boundary",
        qsr_record["dT_dt"][..., :30],
        oracle["after_qsr_T"][..., :30]
        - oracle["after_sbc_T"][..., :30], wet)
    qsr_term_row = exact_score(
        "GYRE-zco-r4.kt2.stage3.qsr_kmm.executed_term",
        qsr_record["dT_dt"][..., :30],
        np.asarray(traced.qsr_association.qsr_kmm, dtype=np.float64),
        wet,
    )

    def zdf_native(name: str, stop: int | None = None) -> np.ndarray:
        value = np.asarray(zdf_record["arrays"][name], dtype=np.float64)
        value = value[2:-2, 2:-2]
        if value.ndim == 3 and stop is not None:
            value = value[..., :stop]
        return value.transpose(1, 0, 2) if value.ndim == 3 else value.T

    vertical = traced.vertical_solve
    vertical_expected = {
        "heat_K": zdf_native("avt")[:, :, 1:30],
        "isoneutral_K": zdf_native("ah_wslp2")[:, :, 1:30],
        "effective_K": (
            zdf_native("avt")[:, :, 1:30]
            + zdf_native("ah_wslp2")[:, :, 1:30]),
        "e3t_after": zdf_native("e3t_Kaa", 30),
        "e3w_now": zdf_native("e3w_Kmm")[:, :, 1:30],
        "content_T": zdf_native("rhs_T", 30),
        "lower": zdf_native("zwi"),
        "diagonal": zdf_native("zwd"),
        "upper": zdf_native("zws"),
        "solved_T": zdf_native("sol_T_pre_clamp", 30),
    }
    vertical_masks = {
        name: wet[:, :, 1:] if value.shape[-1] == 29 else wet
        for name, value in vertical_expected.items()
    }
    vertical_rows = [
        exact_score(
            f"GYRE-zco-r4.kt2.tra_zdf.{name}", expected,
            np.asarray(getattr(vertical, name), dtype=np.float64),
            vertical_masks[name],
        )
        for name, expected in vertical_expected.items()
    ]
    first_vertical = next(
        (row["name"].rsplit(".", 1)[-1] for row in vertical_rows
         if row["n_unequal"]),
        "NONE",
    )
    report = {
        "format": "nemo-testcase-l2-gyre-r4-step2-tracer-owner-v1",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "policy": "fp64/x64/cpu/libm certified",
        "bridge": (
            "NEMO kt=2 T/S/U/V/SSH primary state; legoESM hidden closure "
            "and RK memories retained from its certified step 1"),
        "step": 2,
        "rk_stage_walk": rk_stage_walk,
        "first_rk_difference": first_any_rk,
        "first_rk_difference_above_1e-10": first_owner_scale_rk,
        "boundary_rows": boundary_rows,
        "executed_qsr_record_identity": qsr_increment_row,
        "executed_qsr_term_score": qsr_term_row,
        "vertical_solve_rows": vertical_rows,
        "first_remaining_vertical_operand_or_boundary": first_vertical,
        "one_variable_experiment": {
            "changed_variable": (
                "Kbb qsr_2BD z_half/e3t stretch only: live 1+r3t(Kbb) "
                "versus the removed static-one default"),
            "production_live": live_row,
            "reverted_static": static_row,
        },
        "owner": (
            "shared-physics Kbb qsr_2BD call omitted z_half_stretch, while "
            "the stage-3 replacement subtracted the live-Kbb field"),
        "compiled_nemo_sources": {
            "stage3_call": (
                "GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:691-704"),
            "live_stretch_kernel": (
                "GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/"
                "traqsr.f90:615-645"),
        },
        "verdict": (
            "CONFIRMED" if static_row["max_abs"] > live_row["max_abs"]
            else "REFUTED"),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def prandtl_split(nemo_avt, nemo_avm, lego_avt, lego_avm, mask, *,
                  threshold: float = 1.0e-12) -> dict:
    """Split an avt residual into the common TKE factor and the Prandtl factor.

    Above its floors NEMO writes ``avm = zav`` and ``avt = pdlr*zav``
    (zdftke.f90:684-692), so ``avt/avm`` isolates the inverse Prandtl number
    and a residual common to both coefficients cancels in it.
    """
    def ratio(avt, avm):
        avm = np.asarray(avm, dtype=np.float64)
        return np.asarray(avt, dtype=np.float64) / np.where(avm > 0.0, avm, 1.0)

    def relative(candidate, reference):
        reference = np.asarray(reference, dtype=np.float64)
        return (np.abs(np.asarray(candidate, dtype=np.float64) - reference)
                / np.maximum(np.abs(reference), np.finfo(np.float64).tiny))

    active = np.asarray(mask, dtype=bool)
    rel_ratio = relative(ratio(lego_avt, lego_avm), ratio(nemo_avt, nemo_avm))
    rel_avm = relative(lego_avm, nemo_avm)
    prandtl = active & (rel_ratio > threshold)
    common = active & (rel_avm > threshold)
    return {
        "prandtl_cells": prandtl,
        "common_cells": common,
        "summary": {
            "threshold_relative": threshold,
            "prandtl_cells": int(prandtl.sum()),
            "prandtl_levels": sorted({int(k) for k in np.nonzero(prandtl)[2]}),
            "prandtl_max_relative": float(np.max(rel_ratio[active])),
            "common_factor_cells": int(common.sum()),
            "common_factor_levels": sorted(
                {int(k) for k in np.nonzero(common)[2]}),
            "common_factor_max_relative": float(np.max(rel_avm[active])),
        },
    }


def run_step2_vertical_owner(oracle_root: Path, output: Path) -> dict:
    """Round 9 (F5): measure, do not infer, the step-2 tra_zdf T owner.

    Same bridge as :func:`run_step2_tracer_owner`: NEMO's kt=2 T/S/U/V/SSH,
    legoESM's carried closure memories.  Each arm changes ONE input of the
    production-jitted step: NEMO's executed ``avt`` at the solve seam
    (everywhere, or only where the inverse Prandtl number differs, or only
    elsewhere), NEMO's kt=1 ``avm``/``avt`` as the carried closure memory,
    and the same traced step evaluated eagerly instead of under JIT.
    """
    import time

    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOVerticalSolveTestInput,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
        gyre_surface_forcings,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from nemo_testcase_l2_gyre_round35_trazdf_matrix import (
        read_trazdf_matrix,
    )

    if jax.default_backend() != "cpu":
        raise RuntimeError("step-2 vertical owner discriminator requires CPU")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    clock = time.perf_counter()
    timings = {}
    admission = admit_oracle("r4_step2", "scalar", oracle_root)
    card = build_nemo_testcase_card(
        "GYRE-zco", namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=10, nn_stock=10, nn_write=10))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    freshwater, surface = gyre_surface_forcings(card, state, 1)
    state = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    entry = read_entry(
        oracle_root / "oracle_step_entry_kt00000002.bin", dims=NEMO_DIMS)
    u, v = _redundant_faces(entry["u"], entry["v"])
    state = state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"][..., :30])),
        S=state.S.replace(data=jnp.asarray(entry["S"][..., :30])),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"])),
    )
    freshwater, surface = gyre_surface_forcings(card, state, 2)
    timings["step1_and_bridge_s"] = time.perf_counter() - clock

    wet = expected_masks(card)["T"]
    iwet = wet[:, :, 1:]
    record = read_trazdf_matrix(
        oracle_root / "oracle_trazdf_matrix_kt00000002.bin",
        expect_kt=2, expected_domain=NEMO_DIMS)

    def native(name: str, stop: int | None = None) -> np.ndarray:
        value = np.asarray(record["arrays"][name], dtype=np.float64)
        value = value[2:-2, 2:-2]
        if stop is not None:
            value = value[..., :stop]
        return value.transpose(1, 0, 2)

    zdf1 = read_zdf_entry(
        oracle_root / "oracle_zdf_entry_kt00000001.bin", dims=NEMO_DIMS)
    zdf2 = read_zdf_entry(
        oracle_root / "oracle_zdf_entry_kt00000002.bin", dims=NEMO_DIMS,
        expected_kt=2)
    nemo_avt = native("avt")[:, :, 1:30]
    nemo_avm = zdf2["avm"][:, :, 1:30]
    nemo_solved = native("sol_T_pre_clamp", 30)
    stage3 = read_stage(
        oracle_root / "oracle_stage_kt00000002_s3.bin", 3, dims=NEMO_DIMS,
        expected_kt=2, expected_kaa=STEP2_STAGE_KAA[3])
    record_rows = [
        exact_score("NEMO.kt2.tra_zdf_avt_vs_zdf_entry_avt",
                    zdf2["avt"][:, :, 1:30], nemo_avt, iwet),
        exact_score("kt1.carried_avm_k_vs_NEMO_avm",
                    zdf1["avm"][:, :, 1:30],
                    np.asarray(state.tke_avm.data), iwet),
        exact_score("kt1.carried_avt_k_vs_NEMO_avt",
                    zdf1["avt"][:, :, 1:30],
                    np.asarray(state.tke_avt.data), iwet),
    ]

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True))

    def jit_step(step_state, override=None):
        return jax.device_get(model.step(
            step_state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface, _vertical_K_test_override=override))

    def arm_rows(name: str, result, *, reference=None) -> list[dict]:
        vertical = result.vertical_solve
        rows = [
            exact_score(f"{name}.heat_K_vs_NEMO_avt", nemo_avt,
                        np.asarray(vertical.heat_K), iwet),
            exact_score(f"{name}.viscosity_K_vs_NEMO_avm", nemo_avm,
                        np.asarray(vertical.viscosity_K), iwet),
            exact_score(f"{name}.solved_T_vs_NEMO", nemo_solved,
                        np.asarray(vertical.solved_T), wet),
            exact_score(f"{name}.stage3_T_vs_NEMO", stage3["T"][..., :30],
                        np.asarray(result.state_after.T.data), wet),
        ]
        if reference is not None:
            rows.append(exact_score(
                f"{name}.solved_T_vs_reference_arm", reference,
                np.asarray(vertical.solved_T), wet))
        return rows

    clock = time.perf_counter()
    free = jit_step(state)
    timings["jit_free_s"] = time.perf_counter() - clock
    own = {name: np.asarray(getattr(free.vertical_solve, name),
                            dtype=np.float64)
           for name in ("heat_K", "viscosity_K", "effective_K", "e3w_now",
                        "e3t_after", "content_T", "solved_T")}
    split = prandtl_split(nemo_avt, nemo_avm, own["heat_K"],
                          own["viscosity_K"], iwet)
    prandtl = split["prandtl_cells"]
    walls = np.zeros_like(iwet)
    wet_cols = wet[..., 0]
    js, is_ = np.nonzero(wet_cols)
    walls[js.min(), :, :] = walls[js.max(), :, :] = True
    walls[:, is_.min(), :] = walls[:, is_.max(), :] = True
    split["summary"]["prandtl_cells_on_wet_boundary_rows"] = int(
        (prandtl & walls).sum())

    def fixed(heat, replace_heat: bool):
        selectors = np.zeros(6, dtype=np.bool_)
        selectors[0] = replace_heat
        return _NEMOVerticalSolveTestInput(
            heat_K=jnp.asarray(heat),
            viscosity_K=jnp.asarray(own["viscosity_K"]),
            formed_K=jnp.asarray(own["effective_K"]),
            tracer_e3w=jnp.asarray(own["e3w_now"]),
            tracer_e3t=jnp.asarray(own["e3t_after"]),
            temperature_content=jnp.asarray(own["content_T"]),
            replace=jnp.asarray(selectors))

    plant_scale = np.float64(1.0 + 2.0 ** -40)
    seam_inputs = {
        "seam_disabled": fixed(own["heat_K"], False),
        "nemo_avt_everywhere": fixed(nemo_avt, True),
        "nemo_avt_prandtl_cells_only": fixed(
            np.where(prandtl, nemo_avt, own["heat_K"]), True),
        "nemo_avt_outside_prandtl_cells": fixed(
            np.where(prandtl, own["heat_K"], nemo_avt), True),
        "control_plant_nemo_avt_scaled": fixed(
            np.where(iwet, nemo_avt * plant_scale, nemo_avt), True),
    }
    clock = time.perf_counter()
    seam = {name: jit_step(state, override)
            for name, override in seam_inputs.items()}
    timings["jit_seam_arms_s"] = time.perf_counter() - clock
    disabled_solved = np.asarray(seam["seam_disabled"].vertical_solve.solved_T)
    nemo_solved_arm = np.asarray(
        seam["nemo_avt_everywhere"].vertical_solve.solved_T)
    arms = {"free_jit": arm_rows("free_jit", free)}
    # References: the disabled seam against the free step (is the seam
    # passive?), the plant against the NEMO-avt arm (does one input scale
    # move the solve?), every other arm against the disabled seam.
    references = {"seam_disabled": own["solved_T"],
                  "control_plant_nemo_avt_scaled": nemo_solved_arm}
    for name, result in seam.items():
        arms[name] = arm_rows(
            name, result, reference=references.get(name, disabled_solved))

    clock = time.perf_counter()
    memory_avm = jnp.asarray(np.where(iwet, zdf1["avm"][:, :, 1:30], 0.0))
    memory_avt = jnp.asarray(np.where(iwet, zdf1["avt"][:, :, 1:30], 0.0))
    arms["NEMO_kt1_avm_as_carried_memory"] = arm_rows(
        "NEMO_kt1_avm_as_carried_memory",
        jit_step(state._replace(
            tke_avm=state.tke_avm.replace(data=memory_avm))))
    arms["NEMO_kt1_avt_as_carried_memory"] = arm_rows(
        "NEMO_kt1_avt_as_carried_memory",
        jit_step(state._replace(
            tke_avt=state.tke_avt.replace(data=memory_avt))))
    timings["jit_memory_arms_s"] = time.perf_counter() - clock

    def row(name: str) -> dict:
        return next(item for item in arms[name] if "solved_T_vs_NEMO" in
                    item["name"])

    free_max = row("free_jit")["max_abs"]
    nemo_max = row("nemo_avt_everywhere")["max_abs"]
    plant_fired = arms["control_plant_nemo_avt_scaled"][-1]["n_unequal"] > 0
    seam_magnitude_kept = row("seam_disabled")["max_abs"] == free_max
    owner_confirmed = (plant_fired and seam_magnitude_kept
                       and nemo_max <= 1.0e-2 * free_max)
    # A tree whose free residual already sits at the NEMO-avt floor has no
    # avt owner left to name; say so instead of calling it refuted.
    no_avt_residual = plant_fired and free_max <= 2.0 * nemo_max
    report = {
        "format": "nemo-testcase-l2-gyre-r4-step2-vertical-owner-v1",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "policy": "fp64/x64/cpu/libm certified",
        "bridge": (
            "NEMO kt=2 T/S/U/V/SSH primary state; legoESM hidden closure "
            "and RK memories retained from its certified step 1"),
        "record_rows": record_rows,
        "avt_decomposition": split["summary"],
        "arms": arms,
        "plant_scale": float(plant_scale),
        "control_plant_fired": plant_fired,
        "seam_disabled_keeps_free_magnitude": seam_magnitude_kept,
        "owner": (
            "the executed tra_zdf avt (TKE closure output) owns the step-2 "
            "solved-T residual" if owner_confirmed
            else "no avt-owned residual: free solve already at the NEMO-avt "
            "floor" if no_avt_residual else "UNMEASURED"),
        "compiled_nemo_sources": {
            "closure_coefficients": (
                "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90:681-692"),
            "prandtl_number": (
                "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90:389-404"),
            "shear_divisor": (
                "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfsh2.f90:99-103"),
        },
        "timings_s": timings,
        "verdict": ("CONFIRMED" if owner_confirmed
                    else "NO_AVT_RESIDUAL" if no_avt_residual
                    else "REFUTED"),
    }
    arrays = {
        "nemo_avt": nemo_avt, "nemo_avm": nemo_avm,
        "nemo_solved_T": nemo_solved,
        "jit_heat_K": own["heat_K"], "jit_viscosity_K": own["viscosity_K"],
        "jit_solved_T": own["solved_T"], "prandtl_cells": prandtl,
        "interface_wet": iwet,
    }
    for name, result in seam.items():
        arrays[f"{name}_solved_T"] = np.asarray(
            result.vertical_solve.solved_T)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), **arrays)

    # Eager last: a JIT-only composition owner shows up as an eager arm that
    # moves the residual while the source program is unchanged.
    clock = time.perf_counter()
    with jax.disable_jit():
        eager = jax.device_get(model._step_impl(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
            _return_tracer_process_trace=True))
    timings["eager_s"] = time.perf_counter() - clock
    arms["production_eager"] = arm_rows(
        "production_eager", eager, reference=own["solved_T"])
    eager_heat = np.asarray(eager.vertical_solve.heat_K, dtype=np.float64)
    eager_visc = np.asarray(eager.vertical_solve.viscosity_K,
                            dtype=np.float64)
    report["eager_vs_jit"] = [
        exact_score("eager_vs_jit.heat_K", own["heat_K"], eager_heat, iwet),
        exact_score("eager_vs_jit.viscosity_K", own["viscosity_K"],
                    eager_visc, iwet),
    ]
    report["eager_avt_decomposition"] = prandtl_split(
        nemo_avt, nemo_avm, eager_heat, eager_visc, iwet)["summary"]
    arrays.update({"eager_heat_K": eager_heat,
                   "eager_viscosity_K": eager_visc,
                   "eager_solved_T": np.asarray(eager.vertical_solve.solved_T)})
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, indent=2))
    return report


def _kmm_stretch_scores(card, tracer, qsr, eta_b, eta_a, prefix: str):
    """Score three ``1+r3t(Kmm)`` compositions against NEMO's stage-3 words.

    NEMO's statement is ``r3t(Kmm) = r1_2*(r3t(Kbb)+r3ta)`` with
    ``r3t = ssh*r1_ht_0`` (stage 2, np_HYB), consumed once by ``tra_qsr`` at
    stage 3.  Inputs are NEMO's own sea surfaces, so every unequal word is
    composition, not trajectory.
    """
    import jax.numpy as jnp

    from legoesm.ocean.eos import nemo_r3t_stretch
    from legoesm.ocean.physics.shortwave_penetration import (
        shortwave_penetration_tendency,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    zc, cfg = card.recipe.z_coord, card.recipe.model_config
    wet = expected_masks(card)["T"]
    wet2, nz = wet[..., 0], wet.shape[-1]
    eta_b = jnp.asarray(eta_b, dtype=jnp.float64)
    eta_a = jnp.asarray(eta_a, dtype=jnp.float64)
    depth = card.recipe.initial_state.H_bathy.data

    def stretch(eta):
        return np.asarray(nemo_r3t_stretch(
            zc, eta, depth, evaluation="nemo_reciprocal"))

    def thickness(eta):
        return np.asarray(compute_layer_thickness(
            eta, depth, zc, min_water_column_m=cfg.min_water_column_m))

    one = np.float64(1.0)
    column = np.asarray(depth, dtype=np.float64)
    # domain.f90 r1_ht_0 = 1/ht_0 on wet columns; the Kbb/Kaa rows below
    # check this reciprocal against NEMO's own r3t words.
    r1_ht = np.where(column > 0.0,
                     one / np.where(column > 0.0, column, one), 0.0)
    r3t_bb = np.asarray(eta_b) * r1_ht
    r3t_aa = np.asarray(eta_a) * r1_ht
    record = one + tracer["r3t_Kmm"]
    candidates = {
        # NEMO's statement: average the two ratios, then add one.
        "nemo_statement_mean_of_ratios": one + np.float64(0.5) * (
            r3t_bb + r3t_aa),
        # legoESM's stage weights (_qco_t_ratio on the averaged sea
        # surface); the stage-3 qsr_2BD stretch since round 9.
        "legoesm_ratio_of_mean_eta": stretch(np.float64(0.5) * (eta_b + eta_a)),
        # The round-8 stage-3 qsr_2BD stretch (mean live / reference
        # thickness), kept as the measured "before" arm.
        "round8_qsr_mean_thickness": (
            (np.float64(0.5) * (thickness(eta_b) + thickness(eta_a)))[..., 0]
            / np.maximum(thickness(jnp.zeros_like(eta_b))[..., 0], 1.0e-10)),
    }
    rows = [
        exact_score(f"{prefix}r3t.Kbb.ssh_times_r1_ht_0_vs_record",
                    tracer["r3t_Kbb"], r3t_bb, wet2),
        exact_score(f"{prefix}r3t.Kaa.ssh_times_r1_ht_0_vs_record",
                    tracer["r3t_Kaa"], r3t_aa, wet2),
        exact_score(f"{prefix}stretch.Kbb.legoesm_vs_record",
                    one + tracer["r3t_Kbb"], stretch(eta_b), wet2),
    ] + [exact_score(f"{prefix}stretch.Kmm.{name}_vs_record", record, value,
                     wet2)
         for name, value in candidates.items()]
    before = tracer["after_sbc_T"][..., :nz]
    accumulated = tracer["after_qsr_T"][..., :nz]
    qsr_rows = []
    for name, value in (("record", record), *candidates.items()):
        increment = np.asarray(shortwave_penetration_tendency(
            jnp.asarray(qsr["qsr"]), zc.dz_ref, zc.z_half_ref,
            jnp.asarray(value), cfg.physics.shortwave_penetration,
            rho_0=cfg.rho_0, c_sw=cfg.physics.constants.c_sw,
            z_half_stretch=jnp.asarray(value)), dtype=np.float64)
        qsr_rows.append(accumulated_component_score(
            f"{prefix}tra_qsr.accumulator.{name}_stretch", before,
            accumulated, increment, wet))
    return rows, qsr_rows


def run_stage3_kmm_stretch(oracle_root: Path, output: Path) -> dict:
    """Round 9 (F3): legoESM's stage-3 ``1+r3t(Kmm)`` compositions.

    Quarter degree at kt=2 (the scalar step-2 record) and the shipped
    one-degree card at kt=1 (its scalar stage-3 record, the only step that
    record holds), so the changed operator is scored on every card it runs.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_NAMELIST,
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    import nemo_testcase_l2_gyre_phase3_gate as one_degree

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    admission = admit_oracle("r4_step2", "scalar", oracle_root)
    card = build_nemo_testcase_card(
        "GYRE-zco", namelist=GYRE_R4_NAMELIST._replace(
            nn_itend=10, nn_stock=10, nn_write=10))
    entry = read_entry(
        oracle_root / "oracle_step_entry_kt00000002.bin", dims=NEMO_DIMS)
    after = read_stage(
        oracle_root / "oracle_stage_kt00000002_s3.bin", 3, dims=NEMO_DIMS,
        expected_kt=2, expected_kaa=STEP2_STAGE_KAA[3])
    rows, qsr_rows = _kmm_stretch_scores(
        card,
        read_tracer_stage3_hires(
            oracle_root / "oracle_rktracer_stage3_kt00000002.bin",
            expected_kt=2),
        read_qsr_stage3_hires(
            oracle_root / "oracle_qsr_stage3_kt00000002.bin", expected_kt=2),
        entry["ssh"], after["ssh"], "r4.kt2.")
    root_1deg = Path(one_degree.SCALAR_MATH_ROOT)
    rows_1deg, qsr_rows_1deg = _kmm_stretch_scores(
        build_nemo_testcase_card("GYRE-zco"),
        one_degree.read_tracer_stage3(
            root_1deg / "oracle_rktracer_stage3_kt00000001.bin"),
        one_degree.read_qsr_stage3(
            root_1deg / "oracle_qsr_stage3_kt00000001.bin"),
        one_degree.read_entry(
            root_1deg / "oracle_step_entry_kt00000001.bin")["ssh"],
        one_degree.read_stage(
            root_1deg / "oracle_stage_kt00000001_s3.bin", 3)["ssh"],
        "1deg.kt1.")
    report = {
        "format": "nemo-testcase-l2-gyre-stage3-kmm-stretch-v2",
        "worktree": worktree_stamp(),
        "oracle_admission": admission,
        "one_degree_root": str(root_1deg),
        "policy": "fp64/x64/cpu/libm certified, eager kernel replay",
        "inputs": ("NEMO's own entry (Kbb) and stage-3 (N+1) sea surfaces: "
                   "quarter degree kt=2, one degree kt=1"),
        "stretch_rows": rows + rows_1deg,
        "qsr_accumulator_rows": qsr_rows + qsr_rows_1deg,
        "verdict": "REPORT_ONLY",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", choices=tuple(DEFAULT_ORACLES), default="r4")
    parser.add_argument(
        "--math-flavour", choices=ORACLE_FLAVOURS, default="vector")
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--ahm-root", type=Path, default=DEFAULT_AHM_ROOT)
    parser.add_argument("--stage2-composition-only", action="store_true")
    parser.add_argument("--stage3-geometry-only", action="store_true")
    parser.add_argument("--step2-tracer-owner-only", action="store_true")
    parser.add_argument("--step2-vertical-owner-only", action="store_true")
    parser.add_argument("--stage3-kmm-stretch-only", action="store_true")
    parser.add_argument(
        "--trajectory-only", action="store_true",
        help="score the ten entry records and one-ULP controls only",
    )
    parser.add_argument("--writer-identity-only", action="store_true")
    parser.add_argument("--writer-twin-root", type=Path)
    parser.add_argument("--writer-candidate-root", type=Path)
    parser.add_argument("--writer-zdf-record", type=Path)
    parser.add_argument("--plant-writer-active-word", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.writer_identity_only:
            if not all((args.writer_twin_root, args.writer_candidate_root,
                        args.writer_zdf_record)):
                raise ValueError(
                    "writer identity requires twin root, candidate root, "
                    "and tra_zdf record")
            report = compare_scalar_writer_identity(
                args.writer_twin_root,
                args.writer_candidate_root,
                args.writer_zdf_record,
                plant_active_word=args.plant_writer_active_word,
            )
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 0 if report["verdict"] == "PASS" else 1
        if (args.writer_twin_root or args.writer_candidate_root
                or args.writer_zdf_record or args.plant_writer_active_word):
            raise ValueError(
                "writer identity arguments require --writer-identity-only")
        admission = admit_oracle(args.deck, args.math_flavour, args.oracle_root)
        narrow = sum((args.stage2_composition_only,
                      args.stage3_geometry_only,
                      args.step2_tracer_owner_only,
                      args.step2_vertical_owner_only,
                      args.stage3_kmm_stretch_only))
        if narrow > 1:
            raise ValueError("choose only one narrow discriminator")
        if narrow and args.trajectory_only:
            raise ValueError(
                "--trajectory-only cannot be combined with a discriminator")
        if args.stage2_composition_only:
            if args.deck != "r4":
                raise ValueError("stage-2 composition-only gate requires deck r4")
            report = run_stage2_composition_only(
                Path(admission["root"]), args.output,
                math_flavour=args.math_flavour,
            )
        elif args.stage3_geometry_only:
            if args.deck != "r4":
                raise ValueError("stage-3 geometry discriminator requires deck r4")
            report = run_stage3_geometry_only(
                Path(admission["root"]), args.output,
                math_flavour=args.math_flavour,
            )
        elif args.step2_tracer_owner_only:
            if (args.deck, args.math_flavour) != ("r4", "scalar"):
                raise ValueError(
                    "step-2 tracer discriminator requires r4/scalar")
            step2_admission = admit_oracle("r4_step2", "scalar")
            report = run_step2_tracer_owner(
                Path(step2_admission["root"]), args.output)
        elif args.step2_vertical_owner_only or args.stage3_kmm_stretch_only:
            if (args.deck, args.math_flavour) != ("r4", "scalar"):
                raise ValueError(
                    "step-2 record discriminators require r4/scalar")
            step2_admission = admit_oracle("r4_step2", "scalar")
            discriminator = (run_step2_vertical_owner
                             if args.step2_vertical_owner_only
                             else run_stage3_kmm_stretch)
            report = discriminator(
                Path(step2_admission["root"]), args.output)
        else:
            report = run(
                Path(admission["root"]), args.output,
                deck=args.deck, ahm_root=args.ahm_root,
                math_flavour=args.math_flavour,
                trajectory_only=args.trajectory_only,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["verdict"] in (
        "PASS", "RAW_UPDATE_AT_BAR", "REPORT_ONLY", "CONFIRMED",
        "NO_AVT_RESIDUAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())

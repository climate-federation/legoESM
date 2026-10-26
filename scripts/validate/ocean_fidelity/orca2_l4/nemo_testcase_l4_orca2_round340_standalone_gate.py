#!/usr/bin/env python3
"""Exact gate and file inputs for the round-340 standalone ORCA2 card."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Mapping

import netCDF4
import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_o1_numerical_gate as o1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round326_hmi_gate as round326,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round333_tracer_gate as round333,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_common as round339,
)

K0_SELECTORS = {
    "nn_ice": 1,
    "ln_drgice_imp": False,
    "ln_mxl0": True,
    "nn_mxlice": 0,
    "ln_icebergs": False,
}
ICE_COVER_SHA256 = "2b2d50bbda68bbc934c95f812256834e68631c47f25b5d2d70bf0fade0ba73dd"
SURFACE_ORDER = ("S0_F", "S0_M", "S1_ENTRY", "S1_EXIT", "S3_ENTRY",
                 "S3_COVER", "S3_FREEZE", "S3_CLAMP", "S3_EXIT",
                 "S2_ENTRY", "S2_EXIT")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integers(data: bytes, offset: int, count: int) -> tuple[tuple[int, ...], int]:
    end = offset + 4 * count
    require(end <= len(data), "truncated record header")
    return struct.unpack(f"={count}i", data[offset:end]), end


def _finite_payload(data: bytes, offset: int, count: int) -> int:
    end = offset + 8 * count
    require(end <= len(data), "truncated binary64 payload")
    values = np.frombuffer(data, dtype=np.dtype("=f8"), count=count, offset=offset)
    require(bool(np.isfinite(values).all()), "record contains non-finite payload")
    return end


def inspect_exact_record(path: Path) -> dict[str, int | str]:
    """Parse one native-stream record using the writers' declared extents."""

    data = path.read_bytes()
    require(len(data) >= 16, f"truncated record: {path.name}")
    magic = data[:16].decode("ascii")
    offset = 16
    if magic == "NEMO_R340_S4_1  ":
        header, offset = _integers(data, offset, 7)
        version, kt, kid, rank, jpi, jpj, bits = header
        bounds, offset = _integers(data, offset, 4)
        ilo, ihi, jlo, jhi = bounds
        ni, nj = ihi - ilo + 1, jhi - jlo + 1
        require((version, bits) == (1, 64), "surface schema moved")
        require(1 <= kt <= 10 and 0 <= kid <= 7 and rank in (0, 1),
                "surface time/boundary/rank moved")
        offset = _finite_payload(data, offset, 12 * jpi * jpj + 5 * ni * nj)
        kind = "surface"
    elif magic == "NEMO_R340_STATE1":
        header, offset = _integers(data, offset, 10)
        version, kt, stage, level, rank, jpi, jpj, jpk, jpts, bits = header
        require((version, bits) == (1, 64), "state schema moved")
        require(1 <= kt <= 10 and 0 <= stage <= 3 and rank in (0, 1),
                "state time/stage/rank moved")
        require(level in (1, 2, 3), "state time level moved")
        count = jpi * jpj * (jpk * (jpts + 2) + 1)
        offset = _finite_payload(data, offset, count)
        kind, kid = "state", stage
    elif magic == "NEMO_R340_IIF_1 ":
        header, offset = _integers(data, offset, 9)
        version, kt, kid, kbb, kmm, rank, jpi, jpj, bits = header
        bounds, offset = _integers(data, offset, 5)
        ilo, jlo, ihi, jhi, size = bounds
        ni, nj = ihi - ilo + 1, jhi - jlo + 1
        require(size == ni * nj, "ice-if declared extent moved")
        require((version, bits) == (1, 64), "ice-if schema moved")
        require(1 <= kt <= 10 and 0 <= kid <= 3 and rank in (0, 1),
                "ice-if time/boundary/rank moved")
        offset = _finite_payload(data, offset, 8 * jpi * jpj + 6 * ni * nj)
        kind = "iif"
    elif magic == "NEMO_R340_FLDRD1":
        header, offset = _integers(data, offset, 9)
        version, kt, call, field, rank, _isecsbc, _nbb, _naa, bits = header
        require((version, bits) == (1, 64), "fld_read schema moved")
        require(1 <= kt <= 10 and call > 0 and field > 0 and rank in (0, 1),
                "fld_read time/call/field/rank moved")
        offset += 34  # FLD%clvar has the declared CHARACTER(len=34).
        require(offset <= len(data), "truncated fld_read variable name")
        _nrec, offset = _integers(data, offset, 4)
        offset = _finite_payload(data, offset, 1)  # rec_shft
        fnow_bounds, offset = _integers(data, offset, 7)
        *_, fnow_size = fnow_bounds
        offset = _finite_payload(data, offset, fnow_size)
        fdta_bounds, offset = _integers(data, offset, 10)
        associated, *_, fdta_size = fdta_bounds
        require(associated in (0, 1), "fld_read association flag moved")
        if associated:
            offset = _finite_payload(data, offset, fdta_size)
        else:
            require(fdta_size == 0, "absent fld_read array has a nonzero size")
        kind, kid = "fld", call
    else:
        raise GateError(f"unknown record magic in {path.name}: {magic!r}")
    require(offset == len(data), f"record has trailing or missing bytes: {path.name}")
    result: dict[str, int | str] = {
        "kind": kind, "kt": kt, "kid": kid, "rank": rank,
    }
    if kind == "state":
        result["level"] = level
    elif kind == "iif":
        result.update(kbb=kbb, kmm=kmm)
    return result


def inspect_acquisition_records(root: Path) -> dict[str, int]:
    """Fail closed on the exact ten-step/two-rank record inventory."""

    records = [inspect_exact_record(path)
               for path in sorted(root.glob("oracle_r340_*.bin"))]
    by_kind = {kind: [row for row in records if row["kind"] == kind]
               for kind in ("surface", "state", "iif", "fld")}
    require(len(by_kind["surface"]) == 160, "surface frame count moved")
    require(len(by_kind["state"]) == 80, "state frame count moved")
    require(len(by_kind["iif"]) == 60, "ice-if frame count moved")
    require(bool(by_kind["fld"]), "S0-F record inventory is empty")
    surface = {(r["kt"], r["kid"], r["rank"]) for r in by_kind["surface"]}
    states = {(r["kt"], r["kid"], r["rank"]) for r in by_kind["state"]}
    iif = {(r["kt"], r["kid"], r["rank"]) for r in by_kind["iif"]}
    require(surface == {(kt, kid, rank) for kt in range(1, 11)
                        for kid in range(8) for rank in range(2)},
            "surface frame inventory duplicated or incomplete")
    require(states == {(kt, stage, rank) for kt in range(1, 11)
                       for stage in range(4) for rank in range(2)},
            "state frame inventory duplicated or incomplete")
    state_levels = {(r["kt"], r["kid"], r["rank"], r["level"])
                    for r in by_kind["state"]}
    expected_levels = set()
    for kt in range(1, 11):
        before = 1 if kt % 2 else 3
        after = 3 if kt % 2 else 1
        for rank in range(2):
            expected_levels |= {
                (kt, 0, rank, before), (kt, 1, rank, after),
                (kt, 2, rank, 2), (kt, 3, rank, after),
            }
    require(state_levels == expected_levels, "derived RK3 time levels moved")
    require(iif == {(kt, kid, rank) for kt in range(1, 11)
                    for kid in ((0, 1, 2, 3) if kt % 2 else (0, 3))
                    for rank in range(2)},
            "ice-if frame inventory duplicated or incomplete")
    require(all(row["kbb"] == row["kmm"] == (1 if row["kt"] % 2 else 3)
                for row in by_kind["iif"]), "ice-if time levels moved")
    return {kind: len(rows) for kind, rows in by_kind.items()}


def validate_k0_selectors(selectors: Mapping[str, object]) -> None:
    require(dict(selectors) == K0_SELECTORS,
            f"resolved standalone selectors moved: {dict(selectors)!r}")


def read_decision171_cover(path: Path) -> np.ndarray:
    """Read only the pinned native-grid Decision-171 climatology."""

    require(path.is_file(), f"missing Decision-171 ice cover: {path}")
    require(sha256(path) == ICE_COVER_SHA256, "Decision-171 ice-cover digest moved")
    with netCDF4.Dataset(path) as dataset:
        require(set(dataset.dimensions) == {"time_counter", "y", "x"},
                "ice-cover dimensions moved")
        variable = dataset.variables.get("ice_cover")
        require(variable is not None, "ice_cover variable is absent")
        require(variable.dimensions == ("time_counter", "y", "x"),
                "ice_cover axis order moved")
        require(variable.dtype == np.dtype("float64"), "ice_cover is not binary64")
        value = np.asarray(variable[:], dtype=np.float64)
    require(value.shape == (1, 148, 180), f"ice-cover shape moved: {value.shape}")
    require(bool(np.isfinite(value).all()), "ice-cover contains non-finite values")
    return value[0]


def read_core_field(path: Path, variable: str) -> np.ndarray:
    """Read a named CORE normal-year field without a fallback or regrid."""

    require(path.is_file(), f"missing CORE input: {path}")
    with netCDF4.Dataset(path) as dataset:
        require(variable in dataset.variables,
                f"CORE variable {variable!r} is absent from {path.name}")
        value = np.asarray(dataset.variables[variable][:], dtype=np.float64)
    require(bool(np.isfinite(value).all()), f"CORE field {variable} is non-finite")
    return value


def load_core_normal_year(root: Path, manifest: Path) -> dict[str, np.ndarray]:
    """Load and map the digest-pinned round-5 CORE forcing and weights."""

    declared = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        declared[name] = digest
    required = {filename for _, filename, _, _ in o1.FIELD_SPECS}
    required |= {
        "weights_core2_orca2_bicub.nc",
        "weights_core2_orca2_bilin.nc",
        "ORCA_R2_zps_domcfg.nc",
    }
    for name in required:
        require(name in declared, f"round-5 input manifest omits {name}")
        require(sha256(root / name) == declared[name],
                f"round-5 CORE input digest moved: {name}")

    mapped: dict[str, np.ndarray] = {}
    weights = {bicubic: o1.load_weights(root, bicubic)
               for bicubic in (False, True)}
    for field, filename, variable, bicubic in o1.FIELD_SPECS:
        with netCDF4.Dataset(root / filename) as dataset:
            source = np.asarray(dataset[variable][0].data, dtype=np.float64)
        indices, coefficients = weights[bicubic]
        mapped[field] = np.asarray(o1.nemo_fld_interp(
            source, indices, coefficients, bicubic=bicubic))
    with netCDF4.Dataset(root / "ORCA_R2_zps_domcfg.nc") as dataset:
        cosine, sine = o1.nemo_t_rotation_from_domain(*(
            np.asarray(dataset[name][:].data, dtype=np.float64)
            for name in ("glamt", "gphit", "glamv", "gphiv")
        ))
    mapped["wndi"], mapped["wndj"] = map(np.asarray, o1.rotate_en_to_ij(
        mapped["wndi"], mapped["wndj"], cosine, sine))
    require(all(np.isfinite(value).all() for value in mapped.values()),
            "mapped CORE normal-year input contains non-finite values")
    return mapped


def verify_round339_prerequisites(ice_root: Path, smoke_root: Path) -> dict:
    ice_manifest = round339.verify_manifest(ice_root / "producer_manifest.json")
    smoke_manifest = round339.verify_manifest(smoke_root / "producer_manifest.json")
    ice_report = json.loads((ice_root / "round339_ice_cover_admission.json").read_text())
    smoke_report = json.loads((smoke_root / "round339_k0_smoke_admission.json").read_text())
    require(ice_report["climatology"]["sha256"] == ICE_COVER_SHA256,
            "round-339 admission names another climatology")
    require(smoke_report["status"] == "PASS_R339_K0_SMOKE",
            "K0 smoke is not admitted")
    validate_k0_selectors({
        "nn_ice": int(smoke_report["resolved"]["nn_ice"]),
        "ln_drgice_imp": smoke_report["resolved"]["ln_drgice_imp"] == "T",
        "ln_mxl0": smoke_report["resolved"]["ln_mxl0"] == "T",
        "nn_mxlice": int(smoke_report["resolved"]["nn_mxlice"]),
        "ln_icebergs": False,
    })
    return {"ice": ice_manifest, "smoke": smoke_manifest}


def exact_score(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Reuse the unchanged rung-10 word scorer for a surface boundary."""

    return rung10.score(np.asarray(actual, dtype=np.float64),
                        np.asarray(expected, dtype=np.float64))


def score_surface_boundaries(
    actual: Mapping[str, Mapping[str, np.ndarray]],
    expected: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    require(tuple(actual) == SURFACE_ORDER, "actual surface boundary order moved")
    require(tuple(expected) == SURFACE_ORDER, "oracle surface boundary order moved")
    rows: dict[str, dict[str, object]] = {}
    for boundary in SURFACE_ORDER:
        require(tuple(actual[boundary]) == tuple(expected[boundary]),
                f"{boundary} field inventory moved")
        for name in actual[boundary]:
            key = f"{boundary}/{name}"
            rows[key] = exact_score(actual[boundary][name], expected[boundary][name])
    unequal = sum(int(row["unequal"]) for row in rows.values())
    total = sum(int(row["count"]) for row in rows.values())
    require(unequal == 0, f"surface exact bar failed: {unequal} / {total}")
    return {"status": "PASS_SURFACE_EXACT", "unequal": 0, "count": total,
            "rows": rows}


def score_composed_card(
    surface_actual: Mapping[str, Mapping[str, np.ndarray]],
    surface_expected: Mapping[str, Mapping[str, np.ndarray]],
    rung10_actual: Mapping[str, np.ndarray],
    rung10_expected: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Apply surface 0/n first, then the existing rung-10 row scorer."""

    surface = score_surface_boundaries(surface_actual, surface_expected)
    require(tuple(rung10_actual) == tuple(rung10_expected),
            "rung-10 row inventory moved")
    rows = {name: exact_score(rung10_actual[name], rung10_expected[name])
            for name in rung10_actual}
    # Keep the rung-10 predicate unchanged: every existing row is exact.
    require(all(row["bit_identical"] for row in rows.values()),
            "existing per-row rung-10 bar failed")
    return {"status": "PASS_S4_EXACT", "surface": surface, "rung10": rows,
            "helper_provenance": {
                "entry_and_bar": rung10.__name__,
                "core_mapping": o1.__name__,
                "mxlice": round326.__name__,
                "source_order": round333.__name__,
                "manifest": round339.__name__,
            }}

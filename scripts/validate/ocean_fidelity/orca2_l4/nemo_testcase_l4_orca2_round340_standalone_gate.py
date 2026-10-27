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

from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_damping_for_step,
)

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
    nemo_testcase_l4_orca2_round332_rescore_gate as round332,
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
SURFACE_ORDER = (
    "S0_ENTRY", "S0_EXIT", "S1_ENTRY", "S1_TURB", "S1_FLUX", "S1_EXIT",
    "S3_ENTRY", "S3_COVER", "S3_FREEZE", "S3_CLAMP", "S3_EXIT",
    "S2_ENTRY", "S2_RNF", "S2_SSR", "S2_FWB", "S2_LBC", "S2_EXIT",
)
OUTER_BOUNDARIES = {
    0: "S0_ENTRY", 1: "S0_EXIT", 2: "S1_ENTRY", 3: "S1_EXIT",
    4: "S3_ENTRY", 5: "S3_EXIT", 6: "S2_ENTRY", 7: "S2_RNF",
    8: "S2_SSR", 9: "S2_FWB", 10: "S2_LBC", 11: "S2_EXIT",
}
IIF_BOUNDARIES = {0: "S3_COVER", 1: "S3_FREEZE", 2: "S3_CLAMP", 3: "S3_EXIT"}
OUTER_FULL_FIELDS = (
    "sst_m", "sss_m", "ssu_m", "ssv_m", "ssh_m", "e3t_m", "frq_m",
    "utau", "vtau", "utauU", "vtauV", "emp", "rnf", "fr_i",
)
OUTER_REDUCED_FIELDS = (
    "qsr", "qns", "sfx", "taum", "wndm", "rnf_tsc_T", "rnf_tsc_S",
    "erp", "qrp", "coefice",
)
OUTER_FIELDS_BY_KID = {
    0: (),
    1: OUTER_FULL_FIELDS[:7],
    2: OUTER_FULL_FIELDS[:7],
    3: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5],
    4: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5],
    5: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i",),
    6: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i",),
    7: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i", "rnf") + OUTER_REDUCED_FIELDS[5:7],
    8: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i", "rnf") + OUTER_REDUCED_FIELDS[5:],
    9: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i", "rnf") + OUTER_REDUCED_FIELDS[5:],
    10: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i", "rnf") + OUTER_REDUCED_FIELDS[5:],
    11: OUTER_FULL_FIELDS[:7] + ("utau", "vtau", "emp") + OUTER_REDUCED_FIELDS[:5] + ("fr_i", "rnf") + OUTER_REDUCED_FIELDS[5:] + ("utauU", "vtauV"),
}
S1_TURB_FIELDS = (
    "theta_air", "q_air", "sst", "ssu", "ssv", "tsk", "ssq", "Cd", "Ch",
    "Ce", "rhoa", "sensible", "latent", "evap", "qsr", "utau", "vtau",
    "taum", "wndm",
)
S1_FLUX_FIELDS = ("precip", "snow", "qlw_down", "qlwn", "qns", "emp")
IIF_FULL_FIELDS = (
    "sss_m", "sst_m", "ts_kbb", "ts_kmm", "fr_i", "ff_f", "tmask", "emp",
    "utau", "vtau",
)
IIF_REDUCED_FIELDS = ("ice_cover", "qsr", "qns", "sfx", "taum", "wndm", "smask0")
IIF_FIELDS_BY_KID = {
    0: tuple(name for name in IIF_FULL_FIELDS if name != "fr_i") + IIF_REDUCED_FIELDS,
    1: IIF_FULL_FIELDS + IIF_REDUCED_FIELDS,
    2: IIF_FULL_FIELDS + IIF_REDUCED_FIELDS,
    3: IIF_FULL_FIELDS + IIF_REDUCED_FIELDS,
}
A2D0_EXTENT = "Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)"
FULL_2D_EXTENT = "jpi,jpj"
R340_ALLOCATED_EXTENTS: dict[str, dict[str, str]] = {
    "sbc_oce": {
        "utau": FULL_2D_EXTENT,
        "vtau": FULL_2D_EXTENT,
        "utauU": FULL_2D_EXTENT,
        "vtauV": FULL_2D_EXTENT,
        "taum": A2D0_EXTENT,
        "wndm": A2D0_EXTENT,
        "qns": A2D0_EXTENT,
        "qsr": A2D0_EXTENT,
        "emp": FULL_2D_EXTENT,
        "sfx": A2D0_EXTENT,
        "rnf": FULL_2D_EXTENT,
        "fr_i": FULL_2D_EXTENT,
        "sst_m": FULL_2D_EXTENT,
        "sss_m": FULL_2D_EXTENT,
        "ssu_m": FULL_2D_EXTENT,
        "ssv_m": FULL_2D_EXTENT,
        "ssh_m": FULL_2D_EXTENT,
        "e3t_m": FULL_2D_EXTENT,
        "frq_m": FULL_2D_EXTENT,
    },
    "sbcrnf": {
        "rnf_tsc": f"{A2D0_EXTENT},jpts",
    },
    "sbcssr": {
        "erp": A2D0_EXTENT,
        "qrp": A2D0_EXTENT,
        "coefice": A2D0_EXTENT,
    },
    "dom_oce": {
        "ff_f": FULL_2D_EXTENT,
        "tmask": f"{FULL_2D_EXTENT},jpk",
        "umask": f"{FULL_2D_EXTENT},jpk",
        "vmask": f"{FULL_2D_EXTENT},jpk",
        "smask0": A2D0_EXTENT,
    },
    "oce": {
        "ts": f"{FULL_2D_EXTENT},jpk,jpts,jpt",
        "uu": f"{FULL_2D_EXTENT},jpk,jpt",
        "vv": f"{FULL_2D_EXTENT},jpk,jpt",
        "ssh": f"{FULL_2D_EXTENT},jpt",
    },
}


class GateError(RuntimeError):
    def __init__(
        self, message: str, *, report: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


def require(
    condition: bool,
    message: str,
    *,
    report: Mapping[str, object] | None = None,
) -> None:
    if not condition:
        raise GateError(message, report=report)


def verify_r340_allocated_extents(
    ppsrc_root: Path, ocean_output: Path,
) -> dict[str, object]:
    """Validate NEMO's preprocessed ALLOCATE extents and runtime halo width."""
    try:
        extents = round339.validate_nemo_allocated_extents(
            Path(ppsrc_root), R340_ALLOCATED_EXTENTS)
        round339.validate_ocean_output(Path(ocean_output), {"nn_hls": "2"})
    except round339.GateError as exc:
        raise GateError(str(exc)) from exc
    return {"nn_hls": 2, "units": extents}


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


def _take_f64(data: bytes, offset: int, count: int) -> tuple[np.ndarray, int]:
    end = _finite_payload(data, offset, count)
    return np.frombuffer(data, dtype=np.dtype("=f8"), count=count, offset=offset), end


def _full_owned(values: np.ndarray, jpi: int, jpj: int,
                bounds: tuple[int, int, int, int]) -> np.ndarray:
    ilo, ihi, jlo, jhi = bounds
    full = values.reshape((jpi, jpj), order="F")
    return np.ascontiguousarray(full[ilo - 1:ihi, jlo - 1:jhi].T)


def _reduced(values: np.ndarray, ni: int, nj: int) -> np.ndarray:
    return np.ascontiguousarray(values.reshape((ni, nj), order="F").T)


def read_exact_record(path: Path) -> dict[str, object]:
    """Decode every registered field and crop full arrays to rank-owned cells."""

    data = path.read_bytes()
    require(len(data) >= 16, f"truncated record: {path.name}")
    magic, offset = data[:16].decode("ascii"), 16
    fields: dict[str, np.ndarray] = {}
    result: dict[str, object]
    if magic == "NEMO_R340_S4_3  ":
        header, offset = _integers(data, offset, 7)
        version, kt, kid, rank, ni, nj, bits = header
        require((version, bits) == (3, 64), "surface schema moved")
        require(1 <= kt <= 10 and kid in OUTER_BOUNDARIES and rank in (0, 1),
                "surface time/boundary/rank moved")
        for name in OUTER_FIELDS_BY_KID[kid]:
            value, offset = _take_f64(data, offset, ni * nj)
            fields[name] = _reduced(value, ni, nj)
        result = {"kind": "surface", "kt": kt, "kid": kid, "rank": rank}
    elif magic in ("NEMO_R340_S1T_1 ", "NEMO_R340_S1F_1 "):
        header, offset = _integers(data, offset, 7)
        version, kt, kid, rank, ni, nj, bits = header
        kind = "s1t" if kid == 0 else "s1f"
        names = S1_TURB_FIELDS if kind == "s1t" else S1_FLUX_FIELDS
        require((version, bits) == (1, 64) and 1 <= kt <= 10 and rank in (0, 1),
                "S1 schema/time/rank moved")
        for name in names:
            value, offset = _take_f64(data, offset, ni * nj)
            fields[name] = _reduced(value, ni, nj)
        result = {"kind": kind, "kt": kt, "kid": kid, "rank": rank}
    elif magic == "NEMO_R340_STATE1":
        header, offset = _integers(data, offset, 10)
        version, kt, stage, level, rank, jpi, jpj, jpk, jpts, bits = header
        require((version, bits) == (1, 64), "state schema moved")
        require(1 <= kt <= 10 and 0 <= stage <= 3 and rank in (0, 1),
                "state time/stage/rank moved")
        require(level in (1, 2, 3), "state time level moved")
        h = 2
        n3, n2 = jpi * jpj * jpk, jpi * jpj
        pts, offset = _take_f64(data, offset, n3 * jpts)
        pts = pts.reshape((jpi, jpj, jpk, jpts), order="F")
        for tracer, name in enumerate(("T", "S")):
            fields[name] = np.ascontiguousarray(
                pts[h:-h, h:-h, :jpk - 1, tracer].transpose(1, 0, 2))
        for name in ("u", "v"):
            value, offset = _take_f64(data, offset, n3)
            value = value.reshape((jpi, jpj, jpk), order="F")
            fields[name] = np.ascontiguousarray(
                value[h:-h, h:-h, :jpk - 1].transpose(1, 0, 2))
        value, offset = _take_f64(data, offset, n2)
        fields["ssh"] = np.ascontiguousarray(
            value.reshape((jpi, jpj), order="F")[h:-h, h:-h].T)
        result = {"kind": "state", "kt": kt, "kid": stage, "rank": rank,
                  "level": level}
    elif magic == "NEMO_R340_IIF_2 ":
        header, offset = _integers(data, offset, 9)
        version, kt, kid, kbb, kmm, rank, jpi, jpj, bits = header
        ni, nj = jpi, jpj
        require((version, bits) == (2, 64), "ice-if declared extent/schema moved")
        require(1 <= kt <= 10 and kid in IIF_BOUNDARIES and rank in (0, 1),
                "ice-if time/boundary/rank moved")
        for name in IIF_FIELDS_BY_KID[kid]:
            value, offset = _take_f64(data, offset, ni * nj)
            fields[name] = _reduced(value, ni, nj)
        result = {"kind": "iif", "kt": kt, "kid": kid, "rank": rank,
                  "kbb": kbb, "kmm": kmm}
    elif magic == "NEMO_R340_FLDRD1":
        header, offset = _integers(data, offset, 9)
        version, kt, call, field, rank, isecsbc, nbb, naa, bits = header
        require(version in (1, 2, 3) and bits == 64, "fld_read schema moved")
        require(1 <= kt <= 10 and call > 0 and field > 0 and rank in (0, 1),
                "fld_read time/call/field/rank moved")
        require(offset + 34 <= len(data), "truncated fld_read variable name")
        clvar = data[offset:offset + 34].decode("ascii").rstrip(" \x00")
        offset += 34
        nrec, offset = _integers(data, offset, 4)
        shift, offset = _take_f64(data, offset, 1)
        now_decl, offset = _integers(data, offset, 11 if version >= 2 else 7)
        now_lo, now_hi = now_decl[:3], now_decl[3:6]
        if version >= 2:
            owned = now_decl[6:10]
            now_size = now_decl[10]
        else:
            owned = (now_lo[0], now_hi[0], now_lo[1], now_hi[1])
            now_size = now_decl[6]
        now, offset = _take_f64(data, offset, now_size)
        now_shape = ((owned[1] - owned[0] + 1,
                      owned[3] - owned[2] + 1,
                      now_hi[2] - now_lo[2] + 1)
                     if version == 3 else
                     tuple(hi - lo + 1 for lo, hi in zip(now_lo, now_hi, strict=True)))
        now = now.reshape(now_shape, order="F")
        if version < 3:
            ilo, ihi, jlo, jhi = owned
            now = now[ilo - now_lo[0]:ihi - now_lo[0] + 1,
                      jlo - now_lo[1]:jhi - now_lo[1] + 1, :]
        fields["fnow"] = np.ascontiguousarray(now.transpose(1, 0, 2))
        data_decl, offset = _integers(data, offset, 14 if version >= 2 else 10)
        associated, data_lo, data_hi, data_size = (
            data_decl[0], data_decl[1:5], data_decl[5:9],
            data_decl[13] if version >= 2 else data_decl[9])
        require(associated in (0, 1), "fld_read association flag moved")
        if associated:
            raw, offset = _take_f64(data, offset, data_size)
            shape = ((data_decl[10] - data_decl[9] + 1,
                      data_decl[12] - data_decl[11] + 1,
                      data_hi[2] - data_lo[2] + 1,
                      data_hi[3] - data_lo[3] + 1)
                     if version == 3 else
                     tuple(hi - lo + 1 for lo, hi in zip(data_lo, data_hi, strict=True)))
            raw = raw.reshape(shape, order="F")
            if version == 2:
                ilo, ihi, jlo, jhi = data_decl[9:13]
                raw = raw[ilo - data_lo[0]:ihi - data_lo[0] + 1,
                          jlo - data_lo[1]:jhi - data_lo[1] + 1, :, :]
            fields["fdta"] = np.ascontiguousarray(
                raw.transpose(1, 0, 2, 3))
        else:
            require(data_size == 0, "absent fld_read array has a nonzero size")
        fields.update(
            isecsbc=np.asarray([isecsbc], np.float64),
            nrec=np.asarray(nrec, np.float64),
            rec_shft=shift.copy(), nbb=np.asarray([nbb], np.float64),
            naa=np.asarray([naa], np.float64),
        )
        result = {"kind": "fld", "kt": kt, "kid": call, "field": field,
                  "rank": rank, "clvar": clvar}
    else:
        raise GateError(f"unknown record magic in {path.name}: {magic!r}")
    require(offset == len(data), f"record has trailing or missing bytes: {path.name}")
    require(all(np.isfinite(value).all() for value in fields.values()),
            f"record contains non-finite payload: {path.name}")
    return {**result, "fields": fields}


def inspect_exact_record(path: Path) -> dict[str, int | str]:
    """Parse one native-stream record using the writers' declared extents."""
    decoded = read_exact_record(path)
    return {name: value for name, value in decoded.items() if name != "fields"}


def inspect_acquisition_records(root: Path) -> dict[str, int]:
    """Fail closed on the exact ten-step/two-rank record inventory."""

    records = [inspect_exact_record(path)
               for path in sorted(root.glob("oracle_r340_*.bin"))]
    by_kind = {kind: [row for row in records if row["kind"] == kind]
               for kind in ("surface", "s1t", "s1f", "state", "iif", "fld")}
    require(len(by_kind["surface"]) == 240, "surface frame count moved")
    require(len(by_kind["s1t"]) == 10, "S1 turbulent frame count moved")
    require(len(by_kind["s1f"]) == 10, "S1 flux frame count moved")
    require(len(by_kind["state"]) == 80, "state frame count moved")
    require(len(by_kind["iif"]) == 60, "ice-if frame count moved")
    require(bool(by_kind["fld"]), "S0-F record inventory is empty")
    surface = {(r["kt"], r["kid"], r["rank"]) for r in by_kind["surface"]}
    states = {(r["kt"], r["kid"], r["rank"]) for r in by_kind["state"]}
    iif = {(r["kt"], r["kid"], r["rank"]) for r in by_kind["iif"]}
    require(surface == {(kt, kid, rank) for kt in range(1, 11)
                        for kid in range(12) for rank in range(2)},
            "surface frame inventory duplicated or incomplete")
    for kind in ("s1t", "s1f"):
        require({(r["kt"], r["rank"]) for r in by_kind[kind]} ==
                {(kt, rank) for kt in range(1, 11, 2) for rank in range(2)},
                f"{kind} frame inventory duplicated or incomplete")
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
    """Map every CORE record reachable during the ten-step K0 replay.

    The files have independent 6-hourly, daily, and monthly cadences.  NEMO
    uses the containing record when ``ln_tint=.false.``; consequently the
    first ten steps need records 1--5, 1--2, and 1 respectively.  Keeping the
    leading record axis here prevents the caller from silently freezing
    record 0 across the replay.
    """

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
            source_variable = dataset[variable]
            count = {1460: 5, 365: 2, 12: 1}.get(source_variable.shape[0])
            require(count is not None,
                    f"CORE cadence moved for {variable}: {source_variable.shape}")
            sources = [np.asarray(source_variable[index].data, dtype=np.float64)
                       for index in range(count)]
        indices, coefficients = weights[bicubic]
        mapped[field] = np.stack([
            np.asarray(o1.nemo_fld_interp(
                source, indices, coefficients, bicubic=bicubic))
            for source in sources
        ])
    with netCDF4.Dataset(root / "ORCA_R2_zps_domcfg.nc") as dataset:
        cosine, sine = o1.nemo_t_rotation_from_domain(*(
            np.asarray(dataset[name][:].data, dtype=np.float64)
            for name in ("glamt", "gphit", "glamv", "gphiv")
        ))
    wind_shape = mapped["wndi"].shape
    mapped["wndi"], mapped["wndj"] = map(np.asarray, o1.rotate_en_to_ij(
        mapped["wndi"], mapped["wndj"],
        np.broadcast_to(cosine, wind_shape), np.broadcast_to(sine, wind_shape)))
    require(all(np.isfinite(value).all() for value in mapped.values()),
            "mapped CORE normal-year input contains non-finite values")
    return mapped


def _monthly_slots(path: Path, variable: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the December/January slots used by K0's first ten steps."""

    require(path.is_file(), f"missing K0 field file: {path}")
    with netCDF4.Dataset(path) as dataset:
        require(variable in dataset.variables,
                f"K0 variable {variable!r} is absent from {path.name}")
        source = dataset.variables[variable]
        source.set_auto_maskandscale(False)
        value = np.asarray(source[:], dtype=np.float64)
    require(value.shape[0] == 12 and value.shape[-2:] == (148, 180),
            f"K0 monthly field shape moved: {path.name} {value.shape}")
    if value.ndim == 4:
        value = value[:, 0]
    require(value.shape == (12, 148, 180),
            f"K0 surface field shape moved: {path.name} {value.shape}")
    require(bool(np.isfinite(value).all()), f"K0 field is non-finite: {path}")
    return value[11], value[0]


def _build_stress_lbc_and_masks(
    mask: np.ndarray,
    u_mask: np.ndarray,
    v_mask: np.ndarray,
    fold,
):
    """Build T-pivot stress LBC plan and folded masks (lbcnfd.f90:588-628,688-706)."""
    import jax.numpy as jnp
    from legoesm.ocean import bulk_flux_omip as bulk

    ny, nx = mask.shape
    require(ny >= 2 and nx >= 2 and nx % 2 == 0,
            f"invalid T-pivot grid shape: {(ny, nx)}")
    center = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    center_index = center.copy()
    center_sign = np.ones((ny, nx), dtype=np.float64)
    perm_t = np.asarray(fold.perm_T, dtype=np.int64)
    half = nx // 2
    # lbcnfd.f90:619-628 folds and negates only ji = jpiglo/2+2 .. jpiglo
    # (0-based 91..179) on the T-pivot row jpj-1, leaving ji = 0 and ji = 90
    # unnegated so +0.0 is not flipped to -0.0.
    center_index[-1, half + 1:] = center[-1, perm_t[half + 1:]]
    center_sign[-1, half + 1:] = float(fold.vector_sign_u)
    east_index = np.roll(center_index, -1, axis=1)
    east_sign = np.roll(center_sign, -1, axis=1)
    # lbcnfd.f90:588-606 folds row jpj (north neighbor of row -1) from row -2.
    north_index = np.empty_like(center_index)
    north_index[:-1] = center_index[1:]
    north_index[-1] = center[-2, perm_t]
    north_sign = np.ones((ny, nx), dtype=np.float64)
    north_sign[:-1] = center_sign[1:]
    north_sign[-1] = float(fold.vector_sign_u)
    stress_lbc = bulk.NemoTStressLbcPlan(
        *(jnp.asarray(value) for value in (
            center_index, east_index, north_index,
            center_sign, east_sign, north_sign)))
    flat_mask = mask.reshape(-1)
    tmask_east = flat_mask[east_index]
    tmask_north = flat_mask[north_index]
    umask = np.asarray(u_mask, dtype=np.float64)[:, 1:]
    vmask = np.asarray(v_mask, dtype=np.float64)[1:, :].copy()
    # dommsk.f90:232 and lbcnfd.f90:688-706 fold V-grid row jpj-1 from row -2.
    vmask[-1] = vmask[-2, np.asarray(fold.perm_v, dtype=np.int64)]
    return stress_lbc, tmask_east, tmask_north, umask, vmask


def build_real_composed_inputs(
    card, deck_root: Path, core_manifest: Path, ice_cover_path: Path,
) -> tuple[dict[str, object], ...]:
    """Build ten K0 controls from the exact files named by its namelists.

    This is the standalone smoke/input boundary, not an oracle-record bridge:
    CORE is mapped through the deck weights, while ice cover, runoff, restoring
    SSS, chlorophyll, geothermal heat flow, masks, and metrics come directly
    from the same K0 deck products NEMO opens.
    """

    import jax.numpy as jnp
    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.ocean.forcing.nemo_surface import NemoSurfaceMeans
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    root = Path(deck_root)
    mapped = load_core_normal_year(root, Path(core_manifest))
    cover = read_decision171_cover(Path(ice_cover_path))
    runoff_before, runoff_after = _monthly_slots(
        root / "runoff_core_monthly.nc", "sorunoff")
    sss_before, sss_after = _monthly_slots(root / "sss_data.nc", "sss")
    with netCDF4.Dataset(root / "chlorophyll.nc") as dataset:
        chlorophyll_records = np.asarray(
            dataset.variables["CHLA"][[11, 0]], dtype=np.float64)
    damping_records = {}
    for filename, variable in (
        ("data_1m_potential_temperature_nomask.nc", "votemper"),
        ("data_1m_salinity_nomask.nc", "vosaline"),
    ):
        with netCDF4.Dataset(root / filename) as dataset:
            source = dataset.variables[variable]
            source.set_auto_maskandscale(False)
            raw = np.asarray(source[[11, 0]], dtype=np.float64)
        damping_records[variable] = np.moveaxis(raw, 1, -1)
    static_fields = {}
    with netCDF4.Dataset(root / "zdfiwm_forcing_orca2.nc") as dataset:
        dataset.set_auto_maskandscale(False)
        for variable in (
            "power_bot", "power_cri", "power_nsq", "power_sho",
            "scale_bot", "scale_cri",
        ):
            static_fields[variable] = np.asarray(
                dataset.variables[variable][0], dtype=np.float64)
    with netCDF4.Dataset(root / "geothermal_heating.nc") as dataset:
        source = dataset.variables["heatflow"]
        source.set_auto_maskandscale(False)
        geothermal_raw = np.asarray(source[0], dtype=np.float64)
    geothermal = ladder.read_geothermal_heatflow(root, (148, 180))
    with netCDF4.Dataset(root / "ORCA_R2_zps_domcfg.nc") as dataset:
        ff_f = np.asarray(dataset.variables["ff_f"][:], dtype=np.float64)
    with netCDF4.Dataset(root / "runoff_core_monthly.nc") as dataset:
        mouth = np.asarray(dataset.variables["socoefr0"][:], dtype=np.float64)

    state = card.ocean.recipe.initial_state
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    ny, nx = mask.shape
    require((ny, nx) == (148, 180), "real K0 card grid moved")
    stress_lbc, tmask_east, tmask_north, umask, vmask = (
        _build_stress_lbc_and_masks(
            mask,
            np.asarray(state.u_mask.data, dtype=np.float64),
            np.asarray(state.v_mask.data, dtype=np.float64),
            card.ocean.recipe.grid.fold,
        )
    )
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    one = jnp.ones((ny, nx), dtype=jnp.float64)
    means = NemoSurfaceMeans(*(zero for _ in range(7)))
    s0 = bulk.NemoS0Inputs(
        prior=means, kt=1, nit000=1, forcing_stride=2,
        restart_means_present=False, use_conservative_temperature=False,
        remove_inverse_barometer=False, u_before=zero, v_before=zero,
        temperature_now=zero, salinity_now=zero, ssh_now=zero,
        ssh_inverse_barometer=zero, e3t_reference=one, r3t_now=zero,
        # traqsr.F90 initializes fraqsr_1lev=1 before the first stage-3 call.
        surface_mask=jnp.asarray(mask), shortwave_fraction_top=one,
    )
    s2 = bulk.NemoS2Inputs(
        stress_lbc=stress_lbc, post_ice_sfx=zero,
        runoff_now=zero, prior_runoff_mass=zero,
        prior_runoff_temperature_content=zero,
        prior_runoff_salinity_content=zero,
        prior_restoring_volume_flux=zero, prior_restoring_heat_flux=zero,
        prior_restoring_coefficient=one, sss_target=zero,
        runoff_mouth_mask=jnp.asarray(mouth), surface_mask=jnp.asarray(mask),
        tmask_east=jnp.asarray(tmask_east), tmask_north=jnp.asarray(tmask_north),
        umask=jnp.asarray(umask), vmask=jnp.asarray(vmask),
        # K0 ocean.output reports rn_fwb0=0.0 for nn_fwb=2 at cold start.
        freshwater_budget_correction=jnp.asarray(0.0, dtype=jnp.float64),
        runoff_factor=jnp.asarray(1.0, dtype=jnp.float64),
        reference_density=jnp.asarray(
            card.ocean.recipe.model_config.physics.constants.rho_0,
            dtype=jnp.float64),
        restoring_mm_per_day=jnp.asarray(-166.67, dtype=jnp.float64),
        restoring_bound_mm_per_day=jnp.asarray(4.0, dtype=jnp.float64),
        seconds_per_day=jnp.asarray(86400.0, dtype=jnp.float64),
        salinity_floor=jnp.asarray(1.0e-20, dtype=jnp.float64),
        heat_capacity=jnp.asarray(
            card.ocean.recipe.model_config.physics.constants.c_sw,
            dtype=jnp.float64),
    )
    geothermal = jnp.asarray(geothermal, dtype=jnp.float64)

    def fld_fields(fnow, *, clock, nrec, nbb, naa, fdta=None):
        value = np.asarray(fnow, dtype=np.float64)
        if value.ndim == 2:
            value = value[..., None]
        result = {"fnow": value}
        if fdta is not None:
            before, after = (np.asarray(item, dtype=np.float64) for item in fdta)
            if before.ndim == 2:
                before, after = before[..., None], after[..., None]
            # fld_update swaps the logical before record into slot 2 and
            # writes the logical after record into slot 1 at the year seam.
            result["fdta"] = np.stack((after, before), axis=-1)
        result.update(
            # The oracle joins two MPI slabs.  Scalar metadata is therefore
            # present once per rank even though the gridded arrays join into
            # one global ORCA2 field.
            isecsbc=np.asarray([clock, clock], dtype=np.float64),
            nrec=np.tile(np.asarray(nrec, dtype=np.float64), 2),
            rec_shft=np.asarray([0.0, 0.0], dtype=np.float64),
            nbb=np.asarray([nbb, nbb], dtype=np.float64),
            naa=np.asarray([naa, naa], dtype=np.float64),
        )
        return result

    controls = []
    for kt in range(1, 11):
        dt_s = int(card.ocean.dt_s)
        # fldread.F90:195-206 adds half of nn_fsbc-1 to NEMO's step-entry
        # clock: (2*kt-1)*dt/2 + dt/2 == kt*dt for nn_fsbc=2.
        isecsbc = kt * dt_s
        retained_kt = kt if kt % 2 else kt - 1
        retained_clock = retained_kt * dt_s

        def selected(name: str, cadence_seconds: int):
            index = retained_clock // cadence_seconds
            records = mapped[name]
            require(0 <= index < records.shape[0],
                    f"CORE record selection escaped loaded window: {name} {index}")
            return jnp.asarray(records[index], dtype=jnp.float64)

        atmosphere = bulk.NemoStandaloneAtmosphereFields(
            *(selected(name, cadence) for name, cadence in (
                ("wndi", 21600), ("wndj", 21600), ("tair", 21600),
                ("humi", 21600), ("slp", 21600),
                ("qsr_down", 86400), ("qlw_down", 86400),
                ("precip_raw", 2678400), ("snow_raw", 2678400))))
        pair_runoff = bulk.NemoStandaloneFieldPair(
            jnp.asarray(runoff_before), jnp.asarray(runoff_after), isecsbc,
            -1339200, 1339200)
        pair_sss = bulk.NemoStandaloneFieldPair(
            jnp.asarray(sss_before), jnp.asarray(sss_after), isecsbc,
            -1339200, 1339200)
        s0f = bulk.NemoStandaloneS0FInputs(
            atmosphere, atmosphere, isecsbc, isecsbc - 1, isecsbc + 1,
            pair_runoff, pair_sss)
        ice_if = bulk.NemoSbcIceIfInputs(
            jnp.asarray(cover), jnp.asarray(cover), isecsbc,
            -15768000, 15768000, zero, zero, zero, jnp.asarray(ff_f),
            jnp.asarray(mask), jnp.asarray(mask))
        fld_actual = {}
        midpoint = (2 * kt - 1) * dt_s // 2
        surface_call = 4 if kt == 1 else 1
        if kt % 2:
            if kt == 1:
                monthly_nrec = (1, 1339200, 12, -1339200)
                for field_index, variable in enumerate(
                        ("votemper", "vosaline"), start=1):
                    before, after = damping_records[variable]
                    now = bulk.nemo_fld_time_interp(
                        jnp.asarray(before), jnp.asarray(after), midpoint,
                        -1339200, 1339200)
                    fld_actual[
                        f"kt01/S0_F/c001/f{field_index:03d}/{variable}"
                    ] = fld_fields(
                        now, clock=midpoint, nrec=monthly_nrec,
                        nbb=2, naa=1, fdta=(before, after))
                annual_nrec = (0, 0, 1, 31536000)
                for field_index, variable in enumerate(static_fields, start=1):
                    fld_actual[
                        f"kt01/S0_F/c002/f{field_index:03d}/{variable}"
                    ] = fld_fields(
                        static_fields[variable], clock=midpoint,
                        nrec=annual_nrec, nbb=1, naa=2)
                fld_actual["kt01/S0_F/c003/f001/heatflow"] = fld_fields(
                    geothermal_raw, clock=midpoint, nrec=annual_nrec,
                    nbb=1, naa=2)
            for field_index, (field, _filename, clvar, _bicubic) in enumerate(
                    o1.FIELD_SPECS, start=1):
                cadence = (21600 if mapped[field].shape[0] == 5 else
                           86400 if mapped[field].shape[0] == 2 else 2678400)
                record = retained_clock // cadence
                key = (f"kt{kt:02d}/S0_F/c{surface_call:03d}/"
                       f"f{field_index:03d}/{clvar}")
                fld_actual[key] = fld_fields(
                    selected(field, cadence), clock=isecsbc,
                    nrec=(record, record * cadence,
                          record + 1, (record + 1) * cadence),
                    nbb=1, naa=2)

            monthly_nrec = (1, 1339200, 12, -1339200)
            ice_now = bulk.nemo_fld_time_interp(
                jnp.asarray(cover), jnp.asarray(cover), isecsbc,
                -15768000, 15768000)
            fld_actual[
                f"kt{kt:02d}/S0_F/c{surface_call + 1:03d}/f001/ice_cover"
            ] = fld_fields(
                ice_now, clock=isecsbc,
                nrec=(1, 15768000, 1, -15768000), nbb=2, naa=1,
                fdta=(cover, cover))
            runoff_now = bulk.nemo_fld_time_interp(
                jnp.asarray(runoff_before), jnp.asarray(runoff_after),
                isecsbc, -1339200, 1339200)
            fld_actual[
                f"kt{kt:02d}/S0_F/c{surface_call + 2:03d}/f001/sorunoff"
            ] = fld_fields(
                runoff_now, clock=isecsbc, nrec=monthly_nrec, nbb=2, naa=1,
                fdta=(runoff_before, runoff_after))
            sss_now = bulk.nemo_fld_time_interp(
                jnp.asarray(sss_before), jnp.asarray(sss_after),
                isecsbc, -1339200, 1339200)
            fld_actual[
                f"kt{kt:02d}/S0_F/c{surface_call + 3:03d}/f001/sss"
            ] = fld_fields(
                sss_now, clock=isecsbc, nrec=monthly_nrec, nbb=2, naa=1,
                fdta=(sss_before, sss_after))

        ocean_call = surface_call + 4 if kt % 2 else 1
        chl_now = bulk.nemo_fld_time_interp(
            jnp.asarray(chlorophyll_records[0]),
            jnp.asarray(chlorophyll_records[1]), midpoint,
            -1339200, 1339200)
        fld_actual[
            f"kt{kt:02d}/S0_F/c{ocean_call:03d}/f001/CHLA"
        ] = fld_fields(
            chl_now, clock=midpoint, nrec=(1, 1339200, 12, -1339200),
            nbb=2, naa=1,
            fdta=(chlorophyll_records[0], chlorophyll_records[1]))
        damping_call = ocean_call + 1
        for field_index, variable in enumerate(("votemper", "vosaline"), start=1):
            before, after = damping_records[variable]
            now = bulk.nemo_fld_time_interp(
                jnp.asarray(before), jnp.asarray(after), midpoint,
                -1339200, 1339200)
            fld_actual[
                f"kt{kt:02d}/S0_F/c{damping_call:03d}/f{field_index:03d}/{variable}"
            ] = fld_fields(
                now, clock=midpoint, nrec=(1, 1339200, 12, -1339200),
                nbb=2, naa=1, fdta=(before, after))
        controls.append({
            "s0f": s0f, "ice_if": ice_if, "s0": s0, "s2": s2,
            "chl": jnp.asarray(ladder.chlorophyll_at_step(
                root, kt, mask > 0.5, dt_s=card.ocean.dt_s)),
            "geothermal_heat_flux_mwm2": geothermal,
            "sponge": build_orca2_damping_for_step(
                root, np.asarray(card.ocean.recipe.z_coord.is_active) > 0.5,
                kt=kt, dt_s=card.ocean.dt_s),
            "_s0f_actual": fld_actual,
        })
    return tuple(controls)


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


def _join_rank_fields(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    require(len(rows) == 2 and {int(row["rank"]) for row in rows} == {0, 1},
            "record boundary does not contain exactly ranks 0 and 1")
    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    left, right = (row["fields"] for row in ordered)
    require(tuple(left) == tuple(right), "rank field inventories differ")
    joined = {}
    for name in left:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        require(a.dtype == b.dtype == np.dtype(np.float64),
                f"{name}: rank dtype moved")
        joined[name] = (np.concatenate((a, b), axis=1)
                        if a.ndim >= 2 else np.concatenate((a, b)))
    return joined


def load_oracle_boundaries(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Unpack all S0-F/S0-M/S1/S3/S2 records on global owned cells."""

    decoded = [read_exact_record(path)
               for path in sorted(root.glob("oracle_r340_*.bin"))]
    result: dict[str, dict[str, np.ndarray]] = {}
    for kt in range(1, 11):
        for kid, boundary in OUTER_BOUNDARIES.items():
            rows = [row for row in decoded if row["kind"] == "surface"
                    and row["kt"] == kt and row["kid"] == kid]
            result[f"kt{kt:02d}/{boundary}"] = _join_rank_fields(rows)
        if kt % 2:
            for kind, boundary in (("s1t", "S1_TURB"), ("s1f", "S1_FLUX")):
                rows = [row for row in decoded if row["kind"] == kind
                        and row["kt"] == kt]
                result[f"kt{kt:02d}/{boundary}"] = _join_rank_fields(rows)
        for kid in ((0, 1, 2, 3) if kt % 2 else (0, 3)):
            rows = [row for row in decoded if row["kind"] == "iif"
                    and row["kt"] == kt and row["kid"] == kid]
            boundary = f"{IIF_BOUNDARIES[kid]}_IIF"
            result[f"kt{kt:02d}/{boundary}"] = _join_rank_fields(rows)
    fld_keys = sorted({
        (int(row["kt"]), int(row["kid"]), int(row["field"]), str(row["clvar"]))
        for row in decoded if row["kind"] == "fld"
    })
    for kt, call, field, clvar in fld_keys:
        rows = [row for row in decoded if row["kind"] == "fld"
                and (row["kt"], row["kid"], row["field"], row["clvar"]) ==
                (kt, call, field, clvar)]
        result[f"kt{kt:02d}/S0_F/c{call:03d}/f{field:03d}/{clvar}"] = \
            _join_rank_fields(rows)
    return result


def load_oracle_state_ladder(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Unpack the 200 registered rung-10 rows from both owned MPI slabs."""

    decoded = [read_exact_record(path)
               for path in sorted(root.glob("oracle_r340_state_*.bin"))]
    result = {}
    for kt in range(1, 11):
        for stage in range(4):
            rows = [row for row in decoded if row["kind"] == "state"
                    and row["kt"] == kt and row["kid"] == stage]
            result[f"kt{kt:02d}/{'entry' if stage == 0 else f'stage{stage}'}"] = \
                _join_rank_fields(rows)
    require(sum(len(fields) for fields in result.values()) == 200,
            "rung-10 ladder is not the registered 200 rows")
    return result


def _ordered_fields(names, values):
    """Materialize one registered boundary in writer declaration order."""

    return {name: np.asarray(values[name]) for name in names}


def _surface_boundaries_for_step(kt, state_before, controls, surface):
    """Rebuild the WRITE-only S0/S1/S3/S2 frames from production results."""

    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.ocean.freshwater import NemoRunoffSurface

    means = surface.means
    mean_fields = {
        "sst_m": means.sst, "sss_m": means.sss, "ssu_m": means.ssu,
        "ssv_m": means.ssv, "ssh_m": means.ssh, "e3t_m": means.e3t,
        "frq_m": means.frq,
    }
    pre = surface.bulk_before_ice
    post = surface.bulk
    tail = surface.s2
    prior_s2 = controls["s2"]
    zero = np.zeros_like(np.asarray(post.qsr))

    def outer(flux, *, fr_i=None, runoff=None, erp=None, qrp=None,
              coefficient=None, use_lbc=False, include_uv=False):
        values = dict(mean_fields)
        values.update(
            utau=tail.utau if use_lbc else flux.utau,
            vtau=tail.vtau if use_lbc else flux.vtau,
            emp=flux.emp, qsr=flux.qsr, qns=flux.qns,
            sfx=tail.sfx, taum=flux.taum, wndm=flux.wndm,
            fr_i=(surface.ice_if.fr_i if fr_i is None else fr_i),
            rnf=(tail.runoff.mass_flux if runoff is None else runoff.mass_flux),
            rnf_tsc_T=(tail.runoff.temperature_content if runoff is None
                       else runoff.temperature_content),
            rnf_tsc_S=(tail.runoff.salinity_content if runoff is None
                       else runoff.salinity_content),
            erp=tail.erp if erp is None else erp,
            qrp=tail.qrp if qrp is None else qrp,
            coefice=(tail.restoring_coefficient if coefficient is None
                     else coefficient),
            utauU=tail.utau_u if include_uv else zero,
            vtauV=tail.vtau_v if include_uv else zero,
        )
        return values

    prior_runoff = NemoRunoffSurface(
        prior_s2.prior_runoff_mass,
        prior_s2.prior_runoff_temperature_content,
        prior_s2.prior_runoff_salinity_content,
    )
    pre_values = outer(pre, fr_i=zero, runoff=prior_runoff,
                       erp=prior_s2.prior_restoring_volume_flux,
                       qrp=prior_s2.prior_restoring_heat_flux,
                       coefficient=prior_s2.prior_restoring_coefficient)
    post_values = outer(post, runoff=prior_runoff,
                        erp=prior_s2.prior_restoring_volume_flux,
                        qrp=prior_s2.prior_restoring_heat_flux,
                        coefficient=prior_s2.prior_restoring_coefficient)
    rnf_values = outer(post, erp=prior_s2.prior_restoring_volume_flux,
                       qrp=prior_s2.prior_restoring_heat_flux,
                       coefficient=prior_s2.prior_restoring_coefficient)
    ssr_flux = post._replace(qns=tail.qns_after_ssr, emp=tail.emp_after_ssr)
    ssr_values = outer(ssr_flux)
    final_flux = post._replace(qns=tail.qns, emp=tail.emp)
    fwb_values = outer(final_flux)
    lbc_values = outer(final_flux, use_lbc=True)
    exit_values = outer(final_flux, use_lbc=True, include_uv=True)

    result = {}
    if kt % 2:
        outer_values = {
            0: {}, 1: mean_fields, 2: mean_fields, 3: pre_values, 4: pre_values,
            5: post_values, 6: post_values, 7: rnf_values, 8: ssr_values,
            9: fwb_values, 10: lbc_values, 11: exit_values,
        }
    else:
        # sbcmod.f90:444-521 skips MOD(kt-1, nn_fsbc) == 0 updates on even kt,
        # retaining kt-1 S2_EXIT (tail) values across kid=3..10 before
        # recomputing utauU/vtauV at kid=11 (sbcmod.f90:568-569).
        outer_values = {
            0: {}, 1: mean_fields, 2: mean_fields, 3: lbc_values, 4: lbc_values,
            5: lbc_values, 6: lbc_values, 7: lbc_values, 8: lbc_values,
            9: lbc_values, 10: lbc_values, 11: exit_values,
        }
    for kid, boundary in OUTER_BOUNDARIES.items():
        result[f"kt{kt:02d}/{boundary}"] = _ordered_fields(
            OUTER_FIELDS_BY_KID[kid], outer_values[kid])

    atmosphere, _, _ = bulk.nemo_orca2_k0_s0f(controls["s0f"])
    turb_values = {
        "theta_air": pre.theta_air, "q_air": pre.q_air,
        "sst": means.sst, "ssu": means.ssu, "ssv": means.ssv,
        "tsk": pre.tsk, "ssq": pre.ssq, "Cd": pre.Cd, "Ch": pre.Ch,
        "Ce": pre.Ce, "rhoa": pre.rhoa, "sensible": pre.sensible,
        "latent": pre.latent, "evap": pre.evap, "qsr": pre.qsr,
        "utau": pre.utau, "vtau": pre.vtau, "taum": pre.taum,
        "wndm": pre.wndm,
    }
    flux_values = {
        "precip": atmosphere.precip, "snow": atmosphere.snow,
        "qlw_down": atmosphere.qlw_down, "qlwn": pre.qlwn,
        "qns": pre.qns, "emp": pre.emp,
    }
    if kt % 2:
        result[f"kt{kt:02d}/S1_TURB"] = _ordered_fields(
            S1_TURB_FIELDS, turb_values)
        result[f"kt{kt:02d}/S1_FLUX"] = _ordered_fields(
            S1_FLUX_FIELDS, flux_values)

    iif = surface.ice_if
    ice_inputs = controls["ice_if"]
    ts_before = np.asarray(state_before.T.data)[..., 0]
    ts_after = np.asarray(iif.ts_surface) if kt % 2 else ts_before
    common = {
        "sss_m": means.sss, "sst_m": means.sst,
        "ff_f": ice_inputs.ff_f, "tmask": ice_inputs.tmask_surface,
        "emp": post.emp if kt % 2 else tail.emp,
        "utau": post.utau if kt % 2 else tail.utau,
        "vtau": post.vtau if kt % 2 else tail.vtau,
        "ice_cover": iif.ice_cover, "sfx": tail.sfx,
        "taum": post.taum, "wndm": post.wndm, "smask0": ice_inputs.smask0,
    }
    kids = (0, 1, 2, 3) if kt % 2 else (0, 3)
    for kid in kids:
        values = dict(common)
        values.update(
            ts_kbb=ts_before if kid < 2 else ts_after,
            ts_kmm=ts_before if kid < 2 else ts_after,
            fr_i=(iif.freezing_temperature if kid in (1, 2) else iif.fr_i),
            qsr=(pre.qsr if kid < 3 else post.qsr) if kt % 2 else post.qsr,
            qns=(pre.qns if kid < 3 else post.qns) if kt % 2 else tail.qns,
        )
        boundary = f"{IIF_BOUNDARIES[kid]}_IIF"
        result[f"kt{kt:02d}/{boundary}"] = _ordered_fields(
            IIF_FIELDS_BY_KID[kid], values)
    return result


def run_composed_card(card, initial_state, step_inputs, *, model=None):
    """Drive the production surface/ocean binding for exactly ten steps.

    ``model`` is the rung-10 ``LatLonCGridOceanModel`` configured with the
    existing WRITE-only live-stage observer.  Its carried ``state_after`` is
    still sourced from the separately compiled ordinary production step; the
    three detached stage outputs are used only to materialize the registered
    200-row ladder.  Each element of ``step_inputs`` supplies the keyword
    operands accepted by :func:`nemo_orca2_k0_ocean_step` and therefore cannot
    carry a NEMO surface exchange record.
    """

    from legoesm.ocean import bulk_flux_omip as bulk

    if model is None:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
            _NEMOWSRK3TestHooks,
        )

        recipe = card.ocean.recipe
        model = LatLonCGridOceanModel(
            recipe.grid, recipe.z_coord, recipe.model_config,
            iwm_forcing=recipe.iwm_forcing,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True),
        )

    inputs = tuple(step_inputs)
    require(len(inputs) == 10, "composed card requires steps 1--10 exactly")
    state = initial_state
    ladder: dict[str, dict[str, np.ndarray]] = {}
    surfaces = []
    surface_actual: dict[str, dict[str, np.ndarray]] = {}
    s0f_actual_all: dict[str, dict[str, np.ndarray]] = {}
    carry_s0 = carry_s2 = prior_surface = None
    for kt, controls in enumerate(inputs, start=1):
        controls = dict(controls)
        require("model" not in controls and "nemo_kt" not in controls,
                "step inputs may not replace the production driver/model")
        if carry_s0 is not None:
            controls["s0"] = controls["s0"]._replace(
                prior=carry_s0.prior,
                shortwave_fraction_top=carry_s0.shortwave_fraction_top,
            )
            controls["s2"] = controls["s2"]._replace(
                prior_runoff_mass=carry_s2.prior_runoff_mass,
                prior_runoff_temperature_content=(
                    carry_s2.prior_runoff_temperature_content),
                prior_runoff_salinity_content=(
                    carry_s2.prior_runoff_salinity_content),
                prior_restoring_volume_flux=(
                    carry_s2.prior_restoring_volume_flux),
                prior_restoring_heat_flux=carry_s2.prior_restoring_heat_flux,
                prior_restoring_coefficient=carry_s2.prior_restoring_coefficient,
            )
        s0f_actual = controls.pop("_s0f_actual", {})
        controls["prior_surface"] = prior_surface
        ladder[f"kt{kt:02d}/entry"] = rung10._candidate_fields(state)
        result = bulk.nemo_orca2_k0_ocean_step(
            card, state, model=model, nemo_kt=kt, **controls)
        surface_actual.update(
            _surface_boundaries_for_step(kt, state, controls, result.surface))
        s0f_actual_all.update(s0f_actual)
        trace = result.ocean
        require(hasattr(trace, "state_after") and hasattr(trace, "stage_outputs"),
                "rung-10 driver lacks the registered live-stage observer")
        require(len(trace.stage_outputs) == 3,
                "rung-10 driver did not expose all three stage exits")
        for stage, values in enumerate(trace.stage_outputs, start=1):
            ladder[f"kt{kt:02d}/stage{stage}"] = \
                rung10._stage_candidate_fields(values)
        surfaces.append(result.surface)
        carry_s0, carry_s2 = result.next_s0, result.next_s2
        prior_surface = result.surface
        state = trace.state_after
    require(sum(len(fields) for fields in ladder.values()) == 200,
            "composed card did not produce the registered 200 ladder rows")
    # load_oracle_boundaries registers all source-ordered surface frames first,
    # followed by the sorted generic fld_read inventory.
    surface_actual.update({key: s0f_actual_all[key]
                           for key in sorted(s0f_actual_all)})
    return {"state_after": state, "surface_steps": tuple(surfaces),
            "surface_actual": surface_actual, "rung10": ladder}


def exact_score(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Reuse the unchanged rung-10 word scorer for a surface boundary."""

    return rung10.score(np.asarray(actual, dtype=np.float64),
                        np.asarray(expected, dtype=np.float64))


def score_surface_boundaries(
    actual: Mapping[str, Mapping[str, np.ndarray]],
    expected: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    require(tuple(actual) == tuple(expected), "surface boundary inventory/order moved")
    rows: dict[str, dict[str, object]] = {}
    by_boundary: dict[str, dict[str, object]] = {}
    by_step: dict[str, dict[str, object]] = {}
    for boundary in expected:
        parts = boundary.split("/")
        require(len(parts) >= 2 and parts[0].startswith("kt"),
                f"malformed surface boundary key: {boundary}")
        step = parts[0]
        label = parts[1]
        require(
            label == "S0_F"
            or label in SURFACE_ORDER
            or (label.endswith("_IIF") and label[:-4] in SURFACE_ORDER),
            f"unregistered surface boundary: {boundary}",
        )
        require(tuple(actual[boundary]) == tuple(expected[boundary]),
                f"{boundary} field inventory moved")
        b_unequal = 0
        b_total = 0
        b_max_abs = 0.0
        b_first_non_bit: str | None = None
        for name in actual[boundary]:
            key = f"{boundary}/{name}"
            row = exact_score(actual[boundary][name], expected[boundary][name])
            rows[key] = row
            row_unequal = int(row["unequal"])
            b_unequal += row_unequal
            b_total += int(row["count"])
            b_max_abs = max(b_max_abs, float(row["max_abs"]))
            if row_unequal > 0 and b_first_non_bit is None:
                b_first_non_bit = key
        by_boundary[boundary] = {
            "unequal": b_unequal,
            "count": b_total,
            "max_abs": b_max_abs,
            "first_non_bit_row": b_first_non_bit,
        }
        step_entry = by_step.setdefault(step, {
            "unequal": 0,
            "count": 0,
            "max_abs": 0.0,
            "first_non_bit_row": None,
        })
        step_entry["unequal"] = int(step_entry["unequal"]) + b_unequal
        step_entry["count"] = int(step_entry["count"]) + b_total
        step_entry["max_abs"] = max(float(step_entry["max_abs"]), b_max_abs)
        if b_first_non_bit is not None and step_entry["first_non_bit_row"] is None:
            step_entry["first_non_bit_row"] = b_first_non_bit
    unequal = sum(int(row["unequal"]) for row in rows.values())
    total = sum(int(row["count"]) for row in rows.values())
    max_abs = max((float(row["max_abs"]) for row in rows.values()), default=0.0)
    first_non_bit_row = next(
        (key for key, row in rows.items() if int(row["unequal"]) > 0),
        None,
    )
    summary = {
        "status": "PASS_SURFACE_EXACT" if unequal == 0 else "FAIL_SURFACE_EXACT",
        "unequal": unequal,
        "count": total,
        "max_abs": max_abs,
        "first_non_bit_row": first_non_bit_row,
        "by_boundary": by_boundary,
        "by_step": by_step,
        "rows": rows,
    }
    require(
        unequal == 0,
        f"surface exact bar failed: {unequal} / {total}",
        report=summary,
    )
    return summary


def score_composed_card(
    surface_actual: Mapping[str, Mapping[str, np.ndarray]],
    oracle_root: Path,
    rung10_actual: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    """Replay decoded owned cells: surface exact first, then the 200-row bar."""

    surface_expected = load_oracle_boundaries(oracle_root)
    rung10_expected = load_oracle_state_ladder(oracle_root)
    surface_error: GateError | None = None
    try:
        surface = score_surface_boundaries(surface_actual, surface_expected)
    except GateError as exc:
        if exc.report is None:
            raise
        surface = exc.report
        surface_error = exc
    require(tuple(rung10_actual) == tuple(rung10_expected), "rung-10 checkpoints moved")
    rows = {}
    for checkpoint in rung10_expected:
        require(tuple(rung10_actual[checkpoint]) == tuple(rung10_expected[checkpoint]),
                f"{checkpoint}: rung-10 field inventory moved")
        for field in rung10_expected[checkpoint]:
            name = f"{checkpoint}/{field}"
            rows[name] = round332.score_with_bar(
                np.asarray(rung10_actual[checkpoint][field], dtype=np.float64),
                np.asarray(rung10_expected[checkpoint][field], dtype=np.float64),
            )
    require(len(rows) == 200, "rung-10 scorer did not replay all 200 rows")
    rung10_passed = all(row["classification"] != "OVER_BAR" for row in rows.values())
    report = {
        "status": (
            "PASS_S4_EXACT"
            if surface_error is None and rung10_passed
            else "FAIL_S4_SURFACE"
            if surface_error is not None
            else "FAIL_S4_RUNG10"
        ),
        "surface": surface,
        "rung10": rows,
        "helper_provenance": {
            "entry_and_bar": rung10.__name__,
            "core_mapping": o1.__name__,
            "mxlice": round326.__name__,
            "source_order": round333.__name__,
            "manifest": round339.__name__,
        },
    }
    if surface_error is not None:
        raise GateError(str(surface_error), report=report) from surface_error
    require(
        rung10_passed,
        "rung-10 bar failed (existing normalized per-row predicate)",
        report=report,
    )
    return report

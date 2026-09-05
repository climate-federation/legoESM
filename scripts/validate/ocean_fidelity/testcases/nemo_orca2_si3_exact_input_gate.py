#!/usr/bin/env python3
"""Fail-closed ORCA1-ice/ORCA2 thermodynamic exact-input admission gate.

The Lane-4 files are decoded from their on-disk extent tables.  Common fields
are independently bridged to the older call-order writer.  Physics scoring is
withheld until the post-frazil/pre-ZDF state exists; in particular, this gate
never invents newly activated ice columns from the pre-frazil ENTRY frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from legoesm.ice.config import SI3ThermoConfig

BAR = 1.0e-15
THD_MAGIC = b"NEMO_L4_O1ITHD1 "
THD_FIELDS = (
    "a_i",
    "v_i",
    "v_s",
    "sv_i",
    "t_su",
    "e_i",
    "e_s",
    "szv_i",
    "qns_ice",
    "qsr_ice",
    "dqns_ice",
    "evap_ice",
    "qprec_ice",
    "qml_ice",
    "qcn_ice",
    "qtr_ice_top",
)
COMMON_STATE_FIELDS = THD_FIELDS[:8]
EXPECTED_CALLS = (1, 3, 5)
EXPECTED_FRAMES = tuple((kt, frame) for kt in EXPECTED_CALLS for frame in (0, 1))
LEGACY_GLOBAL_FIELDS = (
    "a_i",
    "v_i",
    "v_s",
    "sv_i",
    "oa_i",
    "t_su",
    "a_ip",
    "v_ip",
    "v_il",
    "e_i",
    "e_s",
    "szv_i",
)

# Rule-1d registry.  The f1 flux arrays are deliberately still labelled ENTRY:
# ice_thd mutates their selected 1-D copies, not these global source arrays.
FIELD_REGISTRY = {
    "a_i": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "v_i": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "v_s": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "sv_i": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "t_su": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "e_i": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "e_s": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "szv_i": ("state", "f0 PRE_FRAZIL / f1 POST_CORRECTION", "icethd.F90:112-190"),
    "qns_ice": ("forcing", "ice_sbc_flx output at ice_thd ENTRY", "icethd.F90:359-363"),
    "qsr_ice": ("forcing", "ice_sbc_flx output at ice_thd ENTRY", "icethd.F90:359-363"),
    "dqns_ice": ("forcing", "ice_sbc_flx output at ice_thd ENTRY", "icethd.F90:359-363"),
    "evap_ice": ("forcing", "ice_sbc_flx output at ice_thd ENTRY", "icethd.F90:359-363"),
    "qprec_ice": ("forcing", "ice_sbc_flx output at ice_thd ENTRY", "icethd.F90:359"),
    "qml_ice": ("forcing", "ice_flx_other/ZDF input at ice_thd ENTRY", "icethd.F90:369-371"),
    "qcn_ice": ("forcing", "ice_flx_other/ZDF input at ice_thd ENTRY", "icethd.F90:369-371"),
    "qtr_ice_top": ("forcing", "blk_ice_2 input at ice_thd ENTRY", "icethd.F90:369-371"),
}

MISSING_OPERANDS = (
    {
        "boundary": "POST_FRAZIL_PRE_ZDF_1D",
        "source": "icethd.F90:115-148",
        "time_level": (
            "after ice_thd_frazil and ice_thd_1d2d(jl,1), before tendency "
            "initialization and ice_thd_zdf"
        ),
        "fields": (
            "nptidx",
            "a_i_1d",
            "h_i_1d",
            "h_s_1d",
            "t_su_1d",
            "e_i_1d",
            "e_s_1d",
            "s_i_1d",
            "sz_i_1d",
            "oa_i_1d",
            "t_s_1d",
        ),
        "reason": (
            "pre-frazil global ENTRY cannot define post-frazil state; equal active "
            "counts do not establish unchanged indices, geometry, or enthalpy"
        ),
    },
    {
        "boundary": "FULL_RANK0_BLK_ICE_1_2",
        "source": "sbcblk.F90:1218-1346",
        "time_level": "before ice_thd at each executed SI3 call",
        "fields": ("all executed scalar operands, intermediates, and outputs", "ssmask"),
        "reason": "retained bulk probe is one point, not every rank-zero wet cell",
    },
    {
        "boundary": "FULL_RANK0_ICE_FLX_OTHER",
        "source": "icesbc.F90:307-437",
        "time_level": "after ice_sbc_flx and before ice_thd",
        "fields": ("all executed operands", "qsb_ice_bot", "fhld", "qlead", "ssmask"),
        "reason": "thermodynamic frame contains outputs but not their ocean/velocity operands",
    },
    {
        "boundary": "FULL_RANK0_ICE_UPDATE_FLX",
        "source": "icestp.F90:201-213; iceupdate.F90:105-194",
        "time_level": "after ice_thd and before/after ice_update_flx",
        "fields": ("heat/water/salt accumulators", "qsr", "qns", "emp", "sfx", "ssmask"),
        "reason": "exchange output cannot be recomputed from the 16-field thd frame",
    },
)

EXECUTED_CALL_ORDER = (
    {"order": 1, "call": "ice_thd_frazil", "status": "EXECUTED", "source": "icethd.F90:115"},
    {
        "order": 2,
        "call": "active selection + ice_thd_1d2d(1)",
        "status": "EXECUTED",
        "source": "icethd.F90:117-140",
    },
    {
        "order": 3,
        "call": "ice_thd_zdf (BL99/P07)",
        "status": "EXECUTED",
        "source": "icethd.F90:148",
    },
    {
        "order": 4,
        "call": "ice_thd_dh",
        "status": "EXECUTED_ln_icedH_TRUE",
        "source": "icethd.F90:150",
    },
    {"order": 5, "call": "ice_thd_temp", "status": "EXECUTED", "source": "icethd.F90:152"},
    {
        "order": 6,
        "call": "ice_thd_sal (option 2)",
        "status": "EXECUTED",
        "source": "icethd.F90:154",
    },
    {"order": 7, "call": "ice_thd_temp", "status": "EXECUTED", "source": "icethd.F90:156"},
    {
        "order": 8,
        "call": "ice_thd_mono",
        "status": "SKIPPED_ln_virtual_itd_FALSE",
        "source": "icethd.F90:158-159",
    },
    {
        "order": 9,
        "call": "ice_thd_da",
        "status": "SKIPPED_ln_icedA_FALSE",
        "source": "icethd.F90:161",
    },
    {"order": 10, "call": "ice_thd_1d2d(2)", "status": "EXECUTED", "source": "icethd.F90:163"},
    {
        "order": 11,
        "call": "ice_thd_pnd",
        "status": "SKIPPED_ln_pnd_FALSE",
        "source": "icethd.F90:176-177",
    },
    {"order": 12, "call": "ice_itd_rem", "status": "SKIPPED_jpl_1", "source": "icethd.F90:179"},
    {
        "order": 13,
        "call": "ice_thd_do",
        "status": "EXECUTED_ln_icedO_TRUE",
        "source": "icethd.F90:181",
    },
    {
        "order": 14,
        "call": "ice_cor + aging + LBC",
        "status": "EXECUTED",
        "source": "icethd.F90:183-190",
    },
)


class GateError(RuntimeError):
    """A schema, selector, coverage, bridge, or exact-input failure."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_value(text: str, label: str, cast):
    match = re.search(rf"\b{re.escape(label)}\s*=\s*([^\s,]+)", text, flags=re.IGNORECASE)
    if not match:
        raise GateError(f"missing resolved selector line: {label}")
    token = match.group(1)
    if cast is bool:
        token = token.upper().strip(".")
        if token not in {"T", "F", "TRUE", "FALSE"}:
            raise GateError(f"invalid logical for {label}: {match.group(1)}")
        return token in {"T", "TRUE"}
    return cast(token)


def _resolved_numeric_tail(text: str, label: str) -> float:
    match = re.search(
        rf"^.*\b{re.escape(label)}\b.*=\s*"
        rf"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        raise GateError(f"missing resolved numeric tail: {label}")
    return float(match.group(1).replace("D", "E").replace("d", "e"))


def resolve_selectors(text: str) -> dict[str, object]:
    """Read the complete Decision-11 identity from NEMO's resolved output."""
    return {
        "n_categories": _resolved_value(text, "jpl", int),
        "n_ice_layers": _resolved_value(text, "nlay_i", int),
        "n_snow_layers": _resolved_value(text, "nlay_s", int),
        "conductivity": ("p07" if _resolved_value(text, "ln_cndi_P07", bool) else "not_p07"),
        "salinity_scheme": _resolved_value(text, "nn_icesal", int),
        "new_ice_salinity_fraction": _resolved_value(text, "rn_sinew", float),
        "drainage": _resolved_value(text, "ln_drainage", bool),
        "flushing": _resolved_value(text, "ln_flushing", bool),
        "ponds": _resolved_value(text, "ln_pnd", bool),
        "pond_level": _resolved_value(text, "ln_pnd_LEV", bool),
        "lateral_melt": _resolved_value(text, "ln_icedA", bool),
        "virtual_itd": _resolved_value(text, "ln_virtual_itd", bool),
        "thermodynamic_growth": _resolved_value(text, "ln_icedH", bool),
        "open_water_growth": _resolved_value(text, "ln_icedO", bool),
        "landfast_l16": _resolved_value(text, "ln_landfast_L16", bool),
        "evp": _resolved_value(text, "ln_rhg_EVP", bool),
        "aevp": _resolved_value(text, "ln_aEVP", bool),
        "strength_h79": _resolved_value(text, "ln_str_H79", bool),
        "ridging": _resolved_value(text, "ln_ridging", bool),
        "rafting": _resolved_value(text, "ln_rafting", bool),
        "prather": _resolved_value(text, "ln_adv_Pra", bool),
    }


def supported_selectors() -> dict[str, object]:
    identity = SI3ThermoConfig()
    return {
        "n_categories": 1,
        **identity._asdict(),
        "pond_level": False,
        "virtual_itd": False,
        "thermodynamic_growth": True,
        "open_water_growth": True,
        "landfast_l16": True,
        "evp": True,
        "aevp": True,
        "strength_h79": True,
        "ridging": True,
        "rafting": True,
        "prather": True,
    }


def selector_rows(resolved: dict[str, object]) -> list[dict[str, object]]:
    supported = supported_selectors()
    return [
        {
            "selector": name,
            "oracle": resolved[name],
            "implemented_identity": supported[name],
            "status": "VERIFIED" if resolved[name] == supported[name] else "UNSUPPORTED",
        }
        for name in supported
    ]


def _validate_cadence(nn_fsbc: int, ice_dt: float) -> None:
    require(nn_fsbc == 2, f"cadence mismatch nn_fsbc={nn_fsbc}")
    require(ice_dt == 21600.0, f"ice timestep mismatch rDt_ice={ice_dt}")


@dataclass(frozen=True)
class ThdFrame:
    path: Path
    kt: int
    frame: int
    jpi: int
    jpj: int
    jpl: int
    nlay_i: int
    nlay_s: int
    extents: dict[str, tuple[int, int, int, int]]
    fields: dict[str, np.ndarray]


def _parse_l4_thd(path: Path) -> ThdFrame:
    raw = path.read_bytes()
    require(len(raw) >= 60, f"{path}: truncated base header")
    require(raw[:16] == THD_MAGIC, f"{path}: bad magic")
    header = struct.unpack("<11i", raw[16:60])
    version, kt, frame, jpi, jpj, jpl, ni, ns, bits, nf, payload = header
    require(version == 1, f"{path}: unsupported version {version}")
    require(frame in (0, 1), f"{path}: unregistered frame {frame}")
    require(bits == 64 and nf == len(THD_FIELDS), f"{path}: width/field count")
    require(min(jpi, jpj, jpl, ni, ns) > 0, f"{path}: non-positive dimensions")
    extent_end = 60 + 4 * 4 * nf
    require(len(raw) >= extent_end, f"{path}: truncated extent table")
    ext = np.frombuffer(raw, "<i4", count=4 * nf, offset=60).copy()
    ext = ext.reshape((4, nf), order="F")
    require(np.all(ext > 0), f"{path}: non-positive field extent")

    expected_state = {
        "a_i": (jpi, jpj, jpl, 1),
        "v_i": (jpi, jpj, jpl, 1),
        "v_s": (jpi, jpj, jpl, 1),
        "sv_i": (jpi, jpj, jpl, 1),
        "t_su": (jpi, jpj, jpl, 1),
        "e_i": (jpi, jpj, ni, jpl),
        "e_s": (jpi, jpj, ns, jpl),
        "szv_i": (jpi, jpj, ni, jpl),
    }
    extents = {name: tuple(int(v) for v in ext[:, index]) for index, name in enumerate(THD_FIELDS)}
    for name, wanted in expected_state.items():
        require(extents[name] == wanted, f"{path}: {name} extent {extents[name]} != {wanted}")

    flux_xy = extents["qns_ice"][:2]
    require(all(value > 0 for value in flux_xy), f"{path}: bad flux spatial extent")
    for name in THD_FIELDS[8:]:
        wanted_tail = (1, 1) if name == "qprec_ice" else (jpl, 1)
        require(
            extents[name][:2] == flux_xy and extents[name][2:] == wanted_tail,
            f"{path}: inconsistent {name} extent {extents[name]}",
        )
    dx, dy = jpi - flux_xy[0], jpj - flux_xy[1]
    require(
        dx >= 0 and dy >= 0 and dx % 2 == 0 and dy % 2 == 0,
        f"{path}: state/flux halo relation {jpi, jpj}/{flux_xy}",
    )

    derived = int(sum(np.prod(ext[:, index], dtype=np.int64) for index in range(nf)))
    require(payload == derived, f"{path}: payload {payload} != derived {derived}")
    expected_eof = extent_end + 8 * payload
    require(len(raw) == expected_eof, f"{path}: exact-EOF size {len(raw)} != {expected_eof}")

    fields: dict[str, np.ndarray] = {}
    offset = extent_end
    for name in THD_FIELDS:
        shape = extents[name]
        count = int(np.prod(shape, dtype=np.int64))
        values = np.frombuffer(raw, "<f8", count=count, offset=offset).copy()
        require(np.all(np.isfinite(values)), f"{path}: non-finite {name}")
        fields[name] = values.reshape(shape, order="F")
        offset += 8 * count
    require(offset == len(raw), f"{path}: decoder did not reach EOF")

    match = re.fullmatch(r"oracle_orca1ice_thd_kt(\d{8})_f([01])\.bin", path.name)
    require(match is not None, f"{path}: unregistered filename")
    require(
        (int(match.group(1)), int(match.group(2))) == (kt, frame),
        f"{path}: filename/header step-frame mismatch",
    )
    return ThdFrame(path, kt, frame, jpi, jpj, jpl, ni, ns, extents, fields)


def _read_l4_frames(root: Path) -> list[ThdFrame]:
    paths = sorted(root.glob("oracle_orca1ice_thd_kt*_f*.bin"))
    require(paths, f"{root}: no ORCA1-ice thermodynamic records")
    frames = [_parse_l4_thd(path) for path in paths]
    observed = tuple((item.kt, item.frame) for item in frames)
    require(observed == EXPECTED_FRAMES, f"{root}: frame sequence {observed} != {EXPECTED_FRAMES}")
    dimensions = {(item.jpi, item.jpj, item.jpl, item.nlay_i, item.nlay_s) for item in frames}
    require(len(dimensions) == 1, f"{root}: frame dimensions changed")
    return frames


def _read_legacy_global(
    path: Path,
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], list[dict[str, int]]]:
    frames: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    headers: list[dict[str, int]] = []
    with path.open("rb") as handle:
        record = 0
        while magic := handle.read(16):
            require(magic == b"NEMO_L3THD_001  ", f"{path}: legacy magic record {record}")
            header_raw = handle.read(44)
            require(len(header_raw) == 44, f"{path}: truncated legacy header")
            version, kt, stage, payload, nx, ny, nc, ni, ns, npti, bits = struct.unpack(
                "=11i", header_raw
            )
            require(
                version == 1 and bits == 64 and stage in range(8), f"{path}: legacy header {record}"
            )
            nbase = nx * ny * nc if payload == 0 else npti
            names = (
                LEGACY_GLOBAL_FIELDS
                if payload == 0
                else ("a_i", "h_i", "h_s", "t_su", "e_i", "e_s", "sz_i")
            )
            nvalue = nbase * (9 + 2 * ni + ns) if payload == 0 else nbase * (4 + 2 * ni + ns)
            raw = handle.read(8 * nvalue)
            require(len(raw) == 8 * nvalue, f"{path}: truncated legacy payload {record}")
            values = np.frombuffer(raw, "=f8")
            require(np.all(np.isfinite(values)), f"{path}: non-finite legacy payload {record}")
            headers.append({"kt": kt, "stage": stage, "payload": payload, "npti": npti})
            if payload == 0:
                decoded: dict[str, np.ndarray] = {}
                offset = 0
                for name in names:
                    layers = ni if name in ("e_i", "szv_i") else ns if name == "e_s" else 1
                    count = nbase * layers
                    shape = (nx, ny, layers, nc) if layers > 1 else (nx, ny, nc, 1)
                    decoded[name] = values[offset : offset + count].reshape(shape, order="F")
                    offset += count
                require(offset == nvalue, f"{path}: legacy field walk")
                frames[(kt, stage)] = decoded
            record += 1
        require(record > 0, f"{path}: empty legacy thermodynamics stream")
    observed = [(row["kt"], row["stage"]) for row in headers]
    expected = [(kt, stage) for kt in (1, 3, 5, 7, 9) for stage in range(8)]
    require(observed == expected, f"{path}: legacy call-order frames")
    return frames, headers


def _read_zdf_input_headers(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    with path.open("rb") as handle:
        while magic := handle.read(16):
            require(magic == b"NEMO_L3ZIN_002  ", f"{path}: ZDF-input magic")
            header_raw = handle.read(24)
            require(len(header_raw) == 24, f"{path}: truncated ZDF-input header")
            version, kt, category, npti, bits, nvalue = struct.unpack("=6i", header_raw)
            require(
                version == 2 and category == 1 and bits == 64 and nvalue == (13 + 3) * npti,
                f"{path}: ZDF-input header {(version, kt, category, npti, bits, nvalue)}",
            )
            payload = handle.read(8 * nvalue)
            require(len(payload) == 8 * nvalue, f"{path}: truncated ZDF-input payload")
            require(
                np.all(np.isfinite(np.frombuffer(payload, "=f8"))),
                f"{path}: non-finite ZDF-input payload",
            )
            rows.append({"kt": kt, "npti": npti, "values": nvalue})
    require([row["kt"] for row in rows] == [1, 3, 5, 7, 9], f"{path}: ZDF-input cadence")
    return rows


def _score_row(
    oracle: np.ndarray,
    candidate: np.ndarray,
    *,
    name: str,
    plant: bool = False,
) -> dict[str, object]:
    oracle = np.asarray(oracle, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    require(oracle.shape == candidate.shape, f"{name}: shape mismatch")
    require(
        np.all(np.isfinite(oracle)) and np.all(np.isfinite(candidate)),
        f"{name}: non-finite score operand",
    )
    if plant:
        candidate = candidate.copy()
        candidate[0] = np.nextafter(candidate[0], np.inf)
    same = oracle.view(np.uint64) == candidate.view(np.uint64)
    non_bit = int(np.count_nonzero(~same))
    difference = np.abs(candidate - oracle)
    maximum = float(np.max(difference, initial=0.0))
    scale = max(float(np.max(np.abs(oracle), initial=0.0)), 1.0)
    normalized = maximum / scale
    row_ulp = float(np.spacing(np.float64(scale)))
    return {
        "name": name,
        "count": int(oracle.size),
        "bit_identical": int(np.count_nonzero(same)),
        "non_bit": non_bit,
        "max_absolute": maximum,
        "max_normalized": normalized,
        "row_scale_ulp": row_ulp,
        "max_row_scale_ulp": maximum / row_ulp,
        "status": "BIT_IDENTICAL" if non_bit == 0 else "DEBT",
    }


def _bridge_rows(
    l4_frames: list[ThdFrame],
    legacy: dict[tuple[int, int], dict[str, np.ndarray]],
    *,
    plant_field: str | None = None,
) -> list[dict[str, object]]:
    rows = []
    for frame in l4_frames:
        stage = 0 if frame.frame == 0 else 7
        require((frame.kt, stage) in legacy, f"missing legacy bridge kt{frame.kt} stage{stage}")
        for field in COMMON_STATE_FIELDS:
            rows.append(
                _score_row(
                    frame.fields[field],
                    legacy[(frame.kt, stage)][field],
                    name=f"kt{frame.kt}.f{frame.frame}.{field}",
                    plant=(plant_field == field and frame.kt == 1 and frame.frame == 0),
                )
            )
    return rows


def _coverage_rows(bridge_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_field = {
        field: all(
            row["status"] == "BIT_IDENTICAL"
            for row in bridge_rows
            if row["name"].endswith(f".{field}")
        )
        for field in COMMON_STATE_FIELDS
    }
    rows = []
    for field in THD_FIELDS:
        kind, level, source = FIELD_REGISTRY[field]
        rows.append(
            {
                "field": field,
                "kind": kind,
                "registered_time_level": level,
                "source": source,
                "load_status": "VERIFIED",
                "bridge_status": (
                    "VERIFIED_BIT_IDENTICAL"
                    if field in by_field and by_field[field]
                    else "DEBT"
                    if field in by_field
                    else "WAIVED_NO_SECOND_WRITER"
                ),
                "physics_status": "UNMEASURED_MISSING_OPERANDS",
            }
        )
    require({row["field"] for row in rows} == set(THD_FIELDS), "field coverage registry incomplete")
    return rows


def _synthetic_raw(*, kt: int, frame: int) -> bytes:
    jpi, jpj, jpl, ni, ns = 7, 8, 1, 3, 3
    flux_x, flux_y = 3, 4
    ext = np.empty((4, len(THD_FIELDS)), dtype="<i4")
    ext[:, :5] = np.asarray([jpi, jpj, jpl, 1])[:, None]
    ext[:, 5] = [jpi, jpj, ni, jpl]
    ext[:, 6] = [jpi, jpj, ns, jpl]
    ext[:, 7] = [jpi, jpj, ni, jpl]
    ext[:, 8:12] = np.asarray([flux_x, flux_y, jpl, 1])[:, None]
    ext[:, 12] = [flux_x, flux_y, 1, 1]
    ext[:, 13:] = np.asarray([flux_x, flux_y, jpl, 1])[:, None]
    payload = int(sum(np.prod(ext[:, index], dtype=np.int64) for index in range(len(THD_FIELDS))))
    header = struct.pack(
        "<11i",
        1,
        kt,
        frame,
        jpi,
        jpj,
        jpl,
        ni,
        ns,
        64,
        len(THD_FIELDS),
        payload,
    )
    values = np.linspace(1.0, 2.0, payload, dtype="<f8")
    return THD_MAGIC + header + ext.tobytes(order="F") + values.tobytes()


def _self_test() -> dict[str, object]:
    schema_plants: dict[str, bool] = {}
    row_plants: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        valid_paths = []
        for kt, frame in EXPECTED_FRAMES:
            path = root / f"oracle_orca1ice_thd_kt{kt:08d}_f{frame}.bin"
            path.write_bytes(_synthetic_raw(kt=kt, frame=frame))
            valid_paths.append(path)
        parsed = _read_l4_frames(root)

        source = valid_paths[0]
        valid = source.read_bytes()
        variants = {
            "magic": b"X" + valid[1:],
            "version": valid[:16] + struct.pack("<i", 2) + valid[20:],
            "field_count": valid[:52] + struct.pack("<i", 15) + valid[56:],
            "payload_count": (
                valid[:56]
                + struct.pack("<i", struct.unpack("<i", valid[56:60])[0] + 1)
                + valid[60:]
            ),
            "extent": valid[:60] + struct.pack("<i", 0) + valid[64:],
            "step": valid[:20] + struct.pack("<i", 3) + valid[24:],
            "truncated": valid[:-1],
            "trailing": valid + b"X",
        }
        for label, raw in variants.items():
            path = root / "oracle_orca1ice_thd_kt00000001_f0.bin"
            path.write_bytes(raw)
            try:
                _parse_l4_thd(path)
            except GateError:
                schema_plants[label] = True
            else:
                raise GateError(f"synthetic schema plant did not bind: {label}")
            path.write_bytes(valid)

        missing = valid_paths[-1]
        retained = missing.read_bytes()
        missing.unlink()
        try:
            _read_l4_frames(root)
        except GateError:
            schema_plants["frame_sequence"] = True
        else:
            raise GateError("synthetic frame-sequence plant did not bind")
        missing.write_bytes(retained)

        equal = np.asarray([1.0, 2.0], dtype=np.float64)
        for field in COMMON_STATE_FIELDS:
            row = _score_row(equal, equal, name=field, plant=True)
            require(
                row["non_bit"] == 1 and row["status"] == "DEBT",
                f"synthetic row plant did not bind: {field}",
            )
            row_plants[field] = True

        supported = supported_selectors()
        planted = supported | {"pond_level": True}
        require(
            any(row["status"] == "UNSUPPORTED" for row in selector_rows(planted)),
            "synthetic selector plant did not bind",
        )
        schema_plants["selector"] = True

        try:
            _validate_cadence(4, 21600.0)
        except GateError:
            schema_plants["cadence"] = True
        else:
            raise GateError("synthetic cadence plant did not bind")

    return {
        "status": "PASS",
        "valid_synthetic_frames": len(parsed),
        "schema_plants": schema_plants,
        "row_plants": row_plants,
    }


def run(root: Path, *, plant_field: str | None = None) -> dict[str, object]:
    l4_frames = _read_l4_frames(root)
    output = root / "ocean.output"
    require(output.is_file(), f"{root}: missing resolved ocean.output")
    text = output.read_text(encoding="utf-8", errors="replace")
    resolved = resolve_selectors(text)
    selectors = selector_rows(resolved)
    unsupported = [row for row in selectors if row["status"] == "UNSUPPORTED"]
    require(
        not unsupported, "selector mismatch: " + ", ".join(row["selector"] for row in unsupported)
    )
    nn_fsbc = _resolved_value(text, "nn_fsbc", int)
    ice_dt = _resolved_numeric_tail(text, "rDt_ice")
    _validate_cadence(nn_fsbc, ice_dt)

    dimensions = l4_frames[0]
    require(
        (dimensions.jpl, dimensions.nlay_i, dimensions.nlay_s)
        == (resolved["n_categories"], resolved["n_ice_layers"], resolved["n_snow_layers"]),
        "header dimensions disagree with resolved selectors",
    )
    legacy_path = root / "oracle_si3_thd_frames.bin"
    zin_path = root / "oracle_si3_zdf_inputs.bin"
    require(legacy_path.is_file(), f"{root}: missing legacy call-order stream")
    require(zin_path.is_file(), f"{root}: missing ZDF-input stream")
    legacy, legacy_headers = _read_legacy_global(legacy_path)
    zin_headers = _read_zdf_input_headers(zin_path)
    bridge = _bridge_rows(l4_frames, legacy, plant_field=plant_field)
    first_non_bit = next((row for row in bridge if row["non_bit"]), None)

    active_counts = []
    for kt in EXPECTED_CALLS:
        entry = next(item for item in l4_frames if (item.kt, item.frame) == (kt, 0))
        active_entry = int(np.count_nonzero(entry.fields["a_i"] > 1.0e-10))
        post_zdf = next(row for row in legacy_headers if row["kt"] == kt and row["stage"] == 1)
        zin = next(row for row in zin_headers if row["kt"] == kt)
        require(post_zdf["npti"] == zin["npti"], f"kt{kt}: ZDF npti mismatch")
        active_counts.append(
            {
                "kt": kt,
                "pre_frazil_active_from_f0": active_entry,
                "post_frazil_pre_zdf_npti": zin["npti"],
                "active_count_delta_not_state_equivalence": zin["npti"] - active_entry,
            }
        )

    coverage = _coverage_rows(bridge)
    result = {
        "format": "nemo-orca1ice-real-geometry-exact-input-v1",
        "root": str(root),
        "provisional_oracle": True,
        "reason_provisional": "phase2x twin B not yet admitted at raw identity",
        "resolved_output": {"path": str(output), "sha256": _sha256(output)},
        "selectors": selectors,
        "cadence": {
            "nn_fsbc": nn_fsbc,
            "ice_timestep_seconds": ice_dt,
            "executed_ocean_steps": list(EXPECTED_CALLS),
            "source": "icestp.F90:126,377-380; sbcmod.F90:474-477,604",
        },
        "streams": {
            "self_describing_thd": {
                "files": len(l4_frames),
                "sha256": {item.path.name: _sha256(item.path) for item in l4_frames},
            },
            "legacy_call_order": {"path": str(legacy_path), "sha256": _sha256(legacy_path)},
            "zdf_inputs": {"path": str(zin_path), "sha256": _sha256(zin_path)},
        },
        "frame_registry": [
            {
                "kt": item.kt,
                "frame": item.frame,
                "fields": list(THD_FIELDS),
                "payload_values": sum(item.fields[name].size for name in THD_FIELDS),
                "status": "VERIFIED_LOADED_TO_EXACT_EOF",
            }
            for item in l4_frames
        ],
        "executed_call_order": EXECUTED_CALL_ORDER,
        "field_coverage": coverage,
        "common_writer_bridge": {
            "rows": bridge,
            "comparisons": sum(row["count"] for row in bridge),
            "non_bit": sum(row["non_bit"] for row in bridge),
            "first_non_bit": first_non_bit,
        },
        "active_point_census": active_counts,
        "missing_operands": MISSING_OPERANDS,
        "scored_physics_rows": 0,
        "bit_identity": "NOT_MEASURED_FOR_PHYSICS",
        "plant_field": plant_field,
    }
    if first_non_bit is not None:
        result["status"] = "STOP_COMMON_WRITER_NONBIT"
    else:
        result["status"] = "STOP_MISSING_OPERANDS"
        result["first_unmeasured_boundary"] = "POST_FRAZIL_PRE_ZDF_1D"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant-field", choices=COMMON_STATE_FIELDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            require(args.root is None, "self-test does not accept a run root")
            result = _self_test()
            code = 0
        else:
            require(args.root is not None, "run root is required")
            result = run(args.root, plant_field=args.plant_field)
            code = 1
    except (GateError, OSError, ValueError, struct.error) as exc:
        result = {
            "root": None if args.root is None else str(args.root),
            "status": "INVALID",
            "error_type": "GateError",
            "error": str(exc),
        }
        code = 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

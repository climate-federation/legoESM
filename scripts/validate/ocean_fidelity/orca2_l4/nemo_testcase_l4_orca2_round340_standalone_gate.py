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
    if magic == "NEMO_R340_S4_2  ":
        header, offset = _integers(data, offset, 7)
        version, kt, kid, rank, jpi, jpj, bits = header
        bounds, offset = _integers(data, offset, 4)
        ilo, ihi, jlo, jhi = bounds
        ni, nj = ihi - ilo + 1, jhi - jlo + 1
        require((version, bits) == (2, 64), "surface schema moved")
        require(1 <= kt <= 10 and kid in OUTER_BOUNDARIES and rank in (0, 1),
                "surface time/boundary/rank moved")
        nfull, nred = jpi * jpj, ni * nj
        for name in OUTER_FULL_FIELDS:
            value, offset = _take_f64(data, offset, nfull)
            fields[name] = _full_owned(value, jpi, jpj, bounds)
        for name in OUTER_REDUCED_FIELDS:
            value, offset = _take_f64(data, offset, nred)
            fields[name] = _reduced(value, ni, nj)
        scalars, offset = _take_f64(data, offset, 2)
        fields["a_fwb"], fields["a_fwb_b"] = scalars[:1], scalars[1:]
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
    elif magic == "NEMO_R340_IIF_1 ":
        header, offset = _integers(data, offset, 9)
        version, kt, kid, kbb, kmm, rank, jpi, jpj, bits = header
        declared, offset = _integers(data, offset, 5)
        ilo, jlo, ihi, jhi, size = declared
        bounds = (ilo, ihi, jlo, jhi)
        ni, nj = ihi - ilo + 1, jhi - jlo + 1
        require(size == ni * nj and (version, bits) == (1, 64),
                "ice-if declared extent/schema moved")
        require(1 <= kt <= 10 and kid in IIF_BOUNDARIES and rank in (0, 1),
                "ice-if time/boundary/rank moved")
        for name in IIF_FULL_FIELDS:
            value, offset = _take_f64(data, offset, jpi * jpj)
            fields[name] = _full_owned(value, jpi, jpj, bounds)
        for name in IIF_REDUCED_FIELDS:
            value, offset = _take_f64(data, offset, ni * nj)
            fields[name] = _reduced(value, ni, nj)
        result = {"kind": "iif", "kt": kt, "kid": kid, "rank": rank,
                  "kbb": kbb, "kmm": kmm}
    elif magic == "NEMO_R340_FLDRD1":
        header, offset = _integers(data, offset, 9)
        version, kt, call, field, rank, isecsbc, nbb, naa, bits = header
        require((version, bits) == (1, 64), "fld_read schema moved")
        require(1 <= kt <= 10 and call > 0 and field > 0 and rank in (0, 1),
                "fld_read time/call/field/rank moved")
        require(offset + 34 <= len(data), "truncated fld_read variable name")
        clvar = data[offset:offset + 34].decode("ascii").rstrip(" \x00")
        offset += 34
        nrec, offset = _integers(data, offset, 4)
        shift, offset = _take_f64(data, offset, 1)
        now_decl, offset = _integers(data, offset, 7)
        now_lo, now_hi, now_size = now_decl[:3], now_decl[3:6], now_decl[6]
        now, offset = _take_f64(data, offset, now_size)
        now_shape = tuple(hi - lo + 1 for lo, hi in zip(now_lo, now_hi, strict=True))
        fields["fnow"] = np.ascontiguousarray(
            now.reshape(now_shape, order="F").transpose(1, 0, 2))
        data_decl, offset = _integers(data, offset, 10)
        associated, data_lo, data_hi, data_size = (
            data_decl[0], data_decl[1:5], data_decl[5:9], data_decl[9])
        require(associated in (0, 1), "fld_read association flag moved")
        if associated:
            raw, offset = _take_f64(data, offset, data_size)
            shape = tuple(hi - lo + 1 for lo, hi in zip(data_lo, data_hi, strict=True))
            fields["fdta"] = np.ascontiguousarray(
                raw.reshape(shape, order="F").transpose(1, 0, 2, 3))
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
    for kt, controls in enumerate(inputs, start=1):
        require("model" not in controls and "nemo_kt" not in controls,
                "step inputs may not replace the production driver/model")
        ladder[f"kt{kt:02d}/entry"] = rung10._candidate_fields(state)
        result = bulk.nemo_orca2_k0_ocean_step(
            card, state, model=model, nemo_kt=kt, **controls)
        trace = result.ocean
        require(hasattr(trace, "state_after") and hasattr(trace, "stage_outputs"),
                "rung-10 driver lacks the registered live-stage observer")
        require(len(trace.stage_outputs) == 3,
                "rung-10 driver did not expose all three stage exits")
        for stage, values in enumerate(trace.stage_outputs, start=1):
            ladder[f"kt{kt:02d}/stage{stage}"] = \
                rung10._stage_candidate_fields(values)
        surfaces.append(result.surface)
        state = trace.state_after
    require(sum(len(fields) for fields in ladder.values()) == 200,
            "composed card did not produce the registered 200 ladder rows")
    return {"state_after": state, "surface_steps": tuple(surfaces),
            "rung10": ladder}


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
    oracle_root: Path,
    rung10_actual: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    """Replay decoded owned cells: surface exact first, then the 200-row bar."""

    surface_expected = load_oracle_boundaries(oracle_root)
    rung10_expected = load_oracle_state_ladder(oracle_root)
    surface = score_surface_boundaries(surface_actual, surface_expected)
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
    require(all(row["classification"] != "OVER_BAR" for row in rows.values()),
            "rung-10 bar failed (existing normalized per-row predicate)")
    return {"status": "PASS_S4_EXACT", "surface": surface, "rung10": rows,
            "helper_provenance": {
                "entry_and_bar": rung10.__name__,
                "core_mapping": o1.__name__,
                "mxlice": round326.__name__,
                "source_order": round333.__name__,
                "manifest": round339.__name__,
            }}

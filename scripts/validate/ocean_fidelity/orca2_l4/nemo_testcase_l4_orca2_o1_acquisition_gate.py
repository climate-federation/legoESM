#!/usr/bin/env python3
"""Accept the WRITE-only ORCA2 O1 mapped-input/open-ocean-bulk record."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_phase2b_exchange_gate as exchange,
)

MAGIC = "NEMO_L4_BLKIO_1"
RECORD = "oracle_sbcblk_o1_kt00000001.bin"
NX, NY, BITS = 90, 148, 64
HEADER_FMT = "=8i"
INPUT_FIELDS = (
    "wndi", "wndj", "tair", "humi", "qsr_down", "qlw_down",
    "precip_raw", "snow_raw", "slp",
)
OUTPUT_FIELDS = (
    "theta_air", "q_air", "precip", "sst", "ssu", "ssv", "tsk",
    "ssq", "cd_du", "sensible", "latent", "evap", "qlwn", "qsr",
    "qns", "emp", "utau", "vtau", "taum", "wndm",
)
O1_UNDEFINED_FIELDS = frozenset({"cd_du", "qlwn"})

HYGIENE_STREAMS = {
    "oracle_bt_advmean_operands_kt00000001.bin",
    "oracle_bt_drag_operands_kt00000001.bin",
    "oracle_bt_ordered_operands_kt00000001.bin",
    "oracle_bt_substeps_kt00000001.bin",
    "oracle_ocean_surface_input_kt00000001.bin",
    "oracle_rkstage3_wzv_kt00000001.bin",
    "oracle_slow_forcing_kt00000001.bin",
}


class _PairReader:
    """Byte-exact paired reader which scores only an explicit defined mask."""

    def __init__(self, left: Path, right: Path):
        self.left_path, self.right_path = left, right
        self.left, self.right = left.open("rb"), right.open("rb")
        self.rows: list[dict] = []

    def close(self) -> None:
        require(self.left.read(1) == b"" and self.right.read(1) == b"",
                f"{self.left_path.name}: parser did not consume both records")
        self.left.close()
        self.right.close()

    def exact_bytes(self, count: int, label: str) -> None:
        a, b = self.left.read(count), self.right.read(count)
        require(len(a) == count and len(b) == count, f"{label}: truncated")
        require(a == b, f"{self.left_path.name}:{label}: metadata differs")

    def field(self, count: int, label: str, defined: np.ndarray) -> None:
        byte_offset = self.left.tell()
        a = np.fromfile(self.left, dtype=np.float64, count=count)
        b = np.fromfile(self.right, dtype=np.float64, count=count)
        require(a.size == count and b.size == count, f"{label}: truncated")
        mask = np.asarray(defined, dtype=bool).reshape(-1)
        require(mask.size == count, f"{label}: mask has {mask.size}/{count} slots")
        exact = np.frombuffer(np.ascontiguousarray(a[mask]).tobytes(), dtype=np.uint8)
        other = np.frombuffer(np.ascontiguousarray(b[mask]).tobytes(), dtype=np.uint8)
        require(np.array_equal(exact, other),
                f"{self.left_path.name}:{label}: defined cells differ")
        self.rows.append({
            "field": label,
            "defined_f64": int(mask.sum()),
            "canonical_zero_f64": int((~mask).sum()),
            "byte_offset": byte_offset,
            "first_defined_index": int(np.flatnonzero(mask)[0]) if mask.any() else None,
            "status": "EXACT_DEFINED_BYTES",
        })


def _defined_masks(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return masks in NEMO Fortran storage order for rank 0."""
    from netCDF4 import Dataset

    with Dataset(root / "mesh_mask_0000.nc") as dataset:
        reduced3 = {
            grid: np.asarray(dataset[f"{grid.lower()}mask"][0], dtype=bool).transpose(2, 1, 0)
            for grid in "TUV"
        }
        # fmask is stored as a real array in this diagnostic file.
        f3 = np.asarray(dataset["fmask"][0], dtype=bool).transpose(2, 1, 0)
    reduced3["F"] = f3
    reduced2 = {grid: values[:, :, 0] for grid, values in reduced3.items()}
    full2: dict[str, np.ndarray] = {}
    full3: dict[str, np.ndarray] = {}
    for grid in "TUVF":
        full2[grid] = np.zeros((94, 152), dtype=bool)
        full2[grid][2:92, 2:150] = reduced2[grid]
        full3[grid] = np.zeros((94, 152, reduced3[grid].shape[2]), dtype=bool)
        full3[grid][2:92, 2:150, :] = reduced3[grid]
    return {"reduced2": reduced2, "reduced3": reduced3,
            "full2": full2, "full3": full3}


def _flat(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool).ravel(order="F")


def _compare_hygiene_record(
    left: Path, right: Path, masks: dict[str, dict[str, np.ndarray]]
) -> dict:
    """Compare source-defined slots; enumerate every canonical-zero class."""
    name = left.name
    pair = _PairReader(left, right)
    one = np.ones(1, dtype=bool)
    try:
        if name == "oracle_slow_forcing_kt00000001.bin":
            pair.exact_bytes(16 + 15 * 4, "magic_and_header")
            for field, grid in (("e3u_3d", "U"), ("uu_Krhs", "U"), ("umask", "U"),
                                ("e3v_3d", "V"), ("vv_Krhs", "V"), ("vmask", "V")):
                pair.field(94 * 152 * 31, field, _flat(masks["full3"][grid]))
            for field, grid in (("Ue_rhs_after_average", "U"), ("Ve_rhs_after_average", "V")):
                pair.field(90 * 148, field, _flat(masks["reduced2"][grid]))
            for field, grid in (("r1_hu_0", "U"), ("r1_hv_0", "V")):
                pair.field(94 * 152, field, _flat(masks["full2"][grid]))
            for field, grid in (("Ue_rhs_after_drag", "U"), ("Ve_rhs_after_drag", "V")):
                pair.field(90 * 148, field, _flat(masks["reduced2"][grid]))
            for field, grid in (("CdU_u", "U"), ("CdU_v", "V")):
                pair.field(94 * 152, field, _flat(masks["full2"][grid]))
            pair.field(1, "r1_rho0", one)
            for field, grid in (("utauU", "U"), ("vtauV", "V"),
                                ("r1_hu_Kbb", "U"), ("r1_hv_Kbb", "V")):
                pair.field(94 * 152, field, _flat(masks["full2"][grid]))
            for field, grid in (("Ue_rhs_after_wind", "U"), ("Ve_rhs_after_wind", "V")):
                pair.field(90 * 148, field, _flat(masks["reduced2"][grid]))
        elif name == "oracle_rkstage3_wzv_kt00000001.bin":
            pair.exact_bytes(16 + 8 * 4, "magic_and_header")
            t3 = _flat(masks["full3"]["T"])
            pair.field(94 * 152 * 31, "ww_after_wzv", t3)
            pair.field(94 * 152 * 31, "ww_after_inactive_wAimp", t3)
            pair.field(94 * 152 * 31, "pFw", t3)
        elif name == "oracle_ocean_surface_input_kt00000001.bin":
            pair.exact_bytes(16 + 13 * 4, "magic_and_header")
            kinds = dict(exchange.FIELDS)
            grids = {field: ("U" if field == "utauU" else "V" if field == "vtauV" else "T")
                     for field, _ in exchange.FIELDS}
            for field, allocation in exchange.FIELDS:
                grid = grids[field]
                if allocation == "full":
                    count, mask = 94 * 152, _flat(masks["full2"][grid])
                elif allocation == "reduced":
                    count, mask = 90 * 148, _flat(masks["reduced2"][grid])
                elif allocation == "halo1":
                    count = 92 * 150
                    halo1 = np.zeros((92, 150), dtype=bool)
                    halo1[1:91, 1:149] = masks["reduced2"][grid]
                    mask = _flat(halo1)
                else:
                    count = 2 * 90 * 148
                    mask = _flat(np.broadcast_to(
                        masks["reduced2"][grid][:, :, None], (90, 148, 2)))
                if field in exchange.INACTIVE_ICEBERG_FIELDS:
                    mask = np.zeros(count, dtype=bool)
                pair.field(count, field, mask)
        elif name.startswith("oracle_bt_"):
            pair.exact_bytes(16 + 6 * 4, "magic_and_header")
            full = {grid: _flat(masks["full2"][grid]) for grid in "TUVF"}
            reduced = {grid: _flat(masks["reduced2"][grid]) for grid in "TUVF"}
            if "substeps" in name:
                fields = (("sshn_e", "T", "full"), ("un_e", "U", "full"),
                          ("vn_e", "V", "full"), ("eta_mid", "T", "full"),
                          ("u_mid", "U", "full"), ("v_mid", "V", "full"),
                          ("ssha_e", "T", "full"), ("zsshp2_e", "T", "full"),
                          ("zu_spg", "U", "full"), ("zv_spg", "V", "full"),
                          ("u_cor", "U", "full"), ("v_cor", "V", "full"),
                          ("zu_trd", "U", "full"), ("zv_trd", "V", "full"),
                          ("zu_frc", "U", "reduced"), ("zv_frc", "V", "reduced"),
                          ("ua_e", "U", "full"), ("va_e", "V", "full"),
                          ("zhU", "U", "full"), ("zhV", "V", "full"))
                for jn in range(1, 66):
                    pair.exact_bytes(4, f"jn_{jn}")
                    for field, grid, allocation in fields:
                        mask = full[grid] if allocation == "full" else reduced[grid]
                        pair.field(mask.size, f"jn{jn}.{field}", mask)
            elif "drag" in name:
                pair.field(full["U"].size, "zCdU_u", full["U"])
                pair.field(full["V"].size, "zCdU_v", full["V"])
                fields = (("un_e", "U"), ("vn_e", "V"), ("hur_e", "U"), ("hvr_e", "V"),
                          ("product_u", "U"), ("product_v", "V"), ("drag_u", "U"),
                          ("drag_v", "V"), ("u_cor", "U"), ("v_cor", "V"),
                          ("zu_trd", "U"), ("zv_trd", "V"))
                for jn in range(1, 66):
                    pair.exact_bytes(4, f"jn_{jn}")
                    for field, grid in fields:
                        pair.field(full[grid].size, f"jn{jn}.{field}", full[grid])
            elif "advmean" in name:
                pair.field(1, "r1_wgt2s", one)
                pair.field(65, "wgtbtp2", np.ones(65, dtype=bool))
                pair.field(full["U"].size, "r1_e2u", full["U"])
                pair.field(full["V"].size, "r1_e1v", full["V"])
                fields = (("before_u", "U"), ("before_v", "V"), ("zhU", "U"),
                          ("zhV", "V"), ("u_mid", "U"), ("v_mid", "V"),
                          ("zhup2", "U"), ("zhvp2", "V"), ("un_adv", "U"), ("vn_adv", "V"))
                for jn in range(1, 66):
                    pair.exact_bytes(4, f"jn_{jn}")
                    pair.field(1, f"jn{jn}.za2", one)
                    for field, grid in fields:
                        pair.field(full[grid].size, f"jn{jn}.{field}", full[grid])
                for suffix in ("pre_lbc_u", "pre_lbc_v", "post_lbc_u", "post_lbc_v"):
                    grid = "U" if suffix.endswith("u") else "V"
                    pair.field(full[grid].size, suffix, full[grid])
            else:
                pair.field(1, "rDt_e", one)
                fields = (
                    [(f, g, "full") for f, g in (
                        ("un_e","U"),("vn_e","V"),("ub_e","U"),("vb_e","V"),
                        ("ubb_e","U"),("vbb_e","V"),("sshn_e","T"),("sshb_e","T"),
                        ("sshbb_e","T"),("u_mid","U"),("v_mid","V"),("eta_mid","T"),
                        ("zhup2","U"),("zhvp2","V"),("zhU","U"),("zhV","V"),
                        ("e2u","U"),("e1v","V"),("r1_e1e2t","T"),("du","T"),
                        ("dv","T"),("zhdiv","T"),("ssh_frc","T"),("ssha_e","T"),
                        ("zsshu_a","U"),("zsshv_a","V"),("zsshp2_e","T"),
                        ("r1_e1u","U"),("r1_e2v","V"),("zu_spg","U"),("zv_spg","V"),
                        ("u_cor","U"),("v_cor","V"),("zu_trd","U"),("zv_trd","V"),
                        ("ua_e","U"),("va_e","V"),("hu_e","U"),("hv_e","V"),
                        ("hur_e","U"),("hvr_e","V"))] +
                    [(f, g, "reduced") for f, g in (
                        ("zu_frc","U"),("zv_frc","V"),("ffu_nw","U"),("ffu_ne","U"),
                        ("ffu_sw","U"),("ffu_se","U"),("ffv_sw","V"),("ffv_se","V"),
                        ("ffv_nw","V"),("ffv_ne","V"))]
                )
                for jn in range(1, 3):
                    pair.exact_bytes(4, f"jn_{jn}")
                    pair.field(7, f"jn{jn}.coefficients", np.ones(7, dtype=bool))
                    for field, grid, allocation in fields:
                        mask = full[grid] if allocation == "full" else reduced[grid]
                        pair.field(mask.size, f"jn{jn}.{field}", mask)
        else:
            raise GateError(f"no hygiene schema for {name}")
        pair.close()
    except Exception:
        pair.left.close()
        pair.right.close()
        raise
    return {
        "status": "EXACT_DEFINED_BYTES",
        "defined_f64": sum(row["defined_f64"] for row in pair.rows),
        "canonical_zero_f64": sum(row["canonical_zero_f64"] for row in pair.rows),
        "undefined_classes": [
            "rank0_halo_bands", "masked_land", "inactive_or_unallocated_components"
        ],
        "fields": pair.rows,
    }


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def read_o1(
    path: Path,
    expected_sha256: str | None = None,
    *,
    require_canonical_undefined: bool = False,
) -> dict:
    """Walk both frames; payload sizes derive only from their write lists."""
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kind, fields in enumerate((INPUT_FIELDS, OUTPUT_FIELDS)):
            raw_magic = handle.read(16)
            require(len(raw_magic) == 16, f"frame {kind}: truncated magic")
            require(raw_magic.decode("ascii").rstrip() == MAGIC,
                    f"frame {kind}: bad magic")
            raw_header = handle.read(struct.calcsize(HEADER_FMT))
            require(len(raw_header) == struct.calcsize(HEADER_FMT),
                    f"frame {kind}: truncated header")
            header = struct.unpack(HEADER_FMT, raw_header)
            expected = (1, 1, kind, NX, NY, len(fields), 0, BITS)
            require(header == expected, f"frame {kind}: header {header} != {expected}")
            count = NX * NY * len(fields)
            payload = np.fromfile(handle, dtype=np.float64, count=count)
            require(payload.size == count, f"frame {kind}: truncated payload")
            require(bool(np.isfinite(payload).all()),
                    f"frame {kind}: non-finite payload")
            row = {
                "kind": kind,
                "fields": list(fields),
                "header": list(header),
                "payload_f64": int(payload.size),
            }
            if kind == 1:
                undefined_nonzero = {}
                for field in O1_UNDEFINED_FIELDS:
                    index = fields.index(field)
                    values = payload[index * NX * NY:(index + 1) * NX * NY]
                    undefined_nonzero[field] = int(np.count_nonzero(values))
                row["undefined_nonzero"] = undefined_nonzero
                if require_canonical_undefined:
                    require(
                        not any(undefined_nonzero.values()),
                        f"nonzero canonical O1 undefined field: {undefined_nonzero}",
                    )
            rows.append(row)
        require(handle.read(1) == b"", "trailing payload")
    digest = exchange.sha256(path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, "O1 record SHA-256 mismatch")
    return {"frames": rows, "bytes": path.stat().st_size, "sha256": digest}


def planted_controls(
    path: Path, *, require_canonical_undefined: bool = False
) -> dict[str, str]:
    expected_digest = exchange.sha256(path)
    results = {}

    def expect(name: str, mutate, *, bind_digest: bool = False) -> None:
        with tempfile.TemporaryDirectory(prefix=f"orca2-o1-{name}-") as td:
            altered = Path(td) / RECORD
            shutil.copyfile(path, altered)
            mutate(altered)
            try:
                read_o1(
                    altered,
                    expected_digest if bind_digest else None,
                    require_canonical_undefined=require_canonical_undefined,
                )
            except (GateError, OSError, UnicodeError, struct.error, ValueError):
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"plant {name} did not fail")

    def patch(target: Path, offset: int, value: bytes) -> None:
        with target.open("r+b") as handle:
            handle.seek(offset)
            handle.write(value)

    expect("header_field_count", lambda p: patch(p, 16 + 5 * 4, struct.pack("=i", 8)))

    def one_ulp(target: Path) -> None:
        offset = 16 + struct.calcsize(HEADER_FMT)
        with target.open("r+b") as handle:
            handle.seek(offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))

    expect("one_ulp_payload", one_ulp, bind_digest=True)
    if require_canonical_undefined:
        frame_bytes = 16 + struct.calcsize(HEADER_FMT) + len(INPUT_FIELDS) * NX * NY * 8
        cd_du_offset = (
            frame_bytes + 16 + struct.calcsize(HEADER_FMT)
            + OUTPUT_FIELDS.index("cd_du") * NX * NY * 8
        )
        expect(
            "undefined_cd_du_nonzero",
            lambda p: patch(p, cd_du_offset, struct.pack("=d", 1.0)),
            bind_digest=False,
        )
    return results


def compare_o1_defined(left: Path, right: Path) -> dict:
    """Byte-compare every source-owned O1 field and enumerate the two exclusions."""
    pair = _PairReader(left, right)
    all_defined = np.ones(NX * NY, dtype=bool)
    none_defined = np.zeros(NX * NY, dtype=bool)
    try:
        for kind, fields in enumerate((INPUT_FIELDS, OUTPUT_FIELDS)):
            pair.exact_bytes(16 + struct.calcsize(HEADER_FMT), f"frame_{kind}_header")
            for field in fields:
                pair.field(
                    NX * NY,
                    f"frame{kind}.{field}",
                    none_defined if field in O1_UNDEFINED_FIELDS else all_defined,
                )
        pair.close()
    except Exception:
        pair.left.close()
        pair.right.close()
        raise
    return {
        "status": "EXACT_DEFINED_BYTES",
        "defined_f64": sum(row["defined_f64"] for row in pair.rows),
        "undefined_f64": sum(row["canonical_zero_f64"] for row in pair.rows),
        "undefined_fields": sorted(O1_UNDEFINED_FIELDS),
        "fields": pair.rows,
    }


def o1_defined_identity_planted_controls(left: Path, right: Path) -> dict[str, str]:
    """One binding one-ULP mutation through the real comparator per owned field."""
    results: dict[str, str] = {}
    header_bytes = 16 + struct.calcsize(HEADER_FMT)
    frame_bytes = header_bytes + len(INPUT_FIELDS) * NX * NY * 8
    field_offsets: list[tuple[str, int]] = []
    for kind, fields, base in (
        (0, INPUT_FIELDS, header_bytes),
        (1, OUTPUT_FIELDS, frame_bytes + header_bytes),
    ):
        for index, field in enumerate(fields):
            if field not in O1_UNDEFINED_FIELDS:
                field_offsets.append((f"frame{kind}.{field}", base + index * NX * NY * 8))
    for label, offset in field_offsets:
        with tempfile.TemporaryDirectory(prefix="orca2-o1-defined-plant-") as td:
            altered = Path(td) / RECORD
            shutil.copyfile(right, altered)
            with altered.open("r+b") as handle:
                handle.seek(offset)
                value = struct.unpack("=d", handle.read(8))[0]
                handle.seek(offset)
                handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
            try:
                compare_o1_defined(left, altered)
            except GateError:
                results[label] = "PASS_NONZERO"
            else:
                raise GateError(f"O1 defined identity plant did not fail: {label}")
    return results


def compare_inherited_records(run_dir: Path, accepted: Path) -> dict:
    """Require exact bytes except the preregistered source-undefined slots."""
    expected = exchange.phase1.expected_inventory() | {exchange.RECORD}
    masks = _defined_masks(accepted)
    rows = []
    for name in sorted(expected):
        left = accepted / name
        right = run_dir / name
        require(left.is_file(), f"accepted inherited record missing: {name}")
        require(right.is_file(), f"candidate inherited record missing: {name}")
        left_digest = exchange.sha256(left)
        right_digest = exchange.sha256(right)
        raw_exact = left_digest == right_digest
        row = {
            "file": name,
            "raw_status": "EXACT_BYTES" if raw_exact else "DIFF_BYTES",
            "accepted_sha256": left_digest,
            "candidate_sha256": right_digest,
        }
        if raw_exact:
            row["status"] = "EXACT_BYTES"
        elif name in HYGIENE_STREAMS:
            row["defined_cell_identity"] = _compare_hygiene_record(left, right, masks)
            row["status"] = "EXACT_DEFINED_BYTES_UNDEFINED_SLOTS_EXCLUDED"
        else:
            row["status"] = "DIFF_DEFINED_BYTES"
        rows.append(row)
    raw_exact = sum(row["raw_status"] == "EXACT_BYTES" for row in rows)
    defined_exact = sum(row["status"] in {
        "EXACT_BYTES", "EXACT_DEFINED_BYTES_UNDEFINED_SLOTS_EXCLUDED"
    } for row in rows)
    return {
        "raw_exact": raw_exact,
        "defined_exact": defined_exact,
        "total": len(rows),
        "raw_differing": [row["file"] for row in rows if row["raw_status"] != "EXACT_BYTES"],
        "defined_differing": [row["file"] for row in rows if row["status"] == "DIFF_DEFINED_BYTES"],
        "rows": rows,
    }


def defined_identity_planted_controls(run_dir: Path, accepted: Path) -> dict[str, str]:
    """One binding one-ULP plant in a source-defined slot per stream family."""
    masks = _defined_masks(accepted)
    results: dict[str, str] = {}
    for name in sorted(HYGIENE_STREAMS):
        baseline = _compare_hygiene_record(accepted / name, run_dir / name, masks)
        target = next(row for row in baseline["fields"] if row["first_defined_index"] is not None)
        with tempfile.TemporaryDirectory(prefix="orca2-defined-identity-") as td:
            altered = Path(td) / name
            shutil.copyfile(run_dir / name, altered)
            offset = target["byte_offset"] + 8 * target["first_defined_index"]
            with altered.open("r+b") as handle:
                handle.seek(offset)
                value = struct.unpack("=d", handle.read(8))[0]
                handle.seek(offset)
                handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
            try:
                _compare_hygiene_record(accepted / name, altered, masks)
            except GateError:
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"defined-cell identity plant did not fail: {name}")
    return results


def validate(
    run_dir: Path,
    control: Path,
    legacy_manifest: Path,
    surface_sha256: str,
    o1_sha256: str | None,
    plants: bool,
    accepted_instrumented: Path | None = None,
    defined_o1_against: Path | None = None,
    canonical_o1_undefined: bool = False,
) -> dict:
    expected = exchange.phase1.expected_inventory() | {exchange.RECORD, RECORD}
    observed = {p.name for p in run_dir.glob("oracle_*.bin")}
    require(
        observed == expected,
        f"record inventory missing={sorted(expected-observed)} "
        f"extra={sorted(observed-expected)}",
    )
    legacy_manifest_for_candidate = (
        None if accepted_instrumented is not None else legacy_manifest
    )
    result = {
        "status": "PASS",
        "oracle_label": "VARIANT",
        "legacy_records": exchange.validate_legacy_records(
            run_dir, legacy_manifest_for_candidate),
        "surface_input": exchange.validate_surface(
            run_dir / exchange.RECORD,
            None if accepted_instrumented is not None else surface_sha256,
            icebergs_off=True,
        ),
        "o1": read_o1(
            run_dir / RECORD,
            o1_sha256,
            require_canonical_undefined=canonical_o1_undefined,
        ),
        "identity": exchange.phase1.validate_identity(control, run_dir),
    }
    if accepted_instrumented is not None:
        result["accepted_legacy_records"] = exchange.validate_legacy_records(
            accepted_instrumented, legacy_manifest)
        result["accepted_surface_input"] = exchange.validate_surface(
            accepted_instrumented / exchange.RECORD,
            surface_sha256,
            icebergs_off=True,
        )
        inherited = compare_inherited_records(run_dir, accepted_instrumented)
        result["inherited_record_identity"] = inherited
        if inherited["defined_exact"] != inherited["total"]:
            result["status"] = "FAIL"
            result["error"] = (
                "defined inherited-record identity failed: "
                f"{inherited['defined_exact']} / {inherited['total']} exact"
            )
    if defined_o1_against is not None:
        result["o1_defined_identity"] = compare_o1_defined(
            defined_o1_against / RECORD, run_dir / RECORD
        )
    if plants:
        result["o1_planted_controls"] = planted_controls(
            run_dir / RECORD,
            require_canonical_undefined=canonical_o1_undefined,
        )
        if accepted_instrumented is not None:
            result["defined_identity_planted_controls"] = (
                defined_identity_planted_controls(run_dir, accepted_instrumented)
            )
        if defined_o1_against is not None:
            result["o1_defined_identity_planted_controls"] = (
                o1_defined_identity_planted_controls(
                    defined_o1_against / RECORD, run_dir / RECORD
                )
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--legacy-record-manifest", type=Path, required=True)
    parser.add_argument("--surface-sha256", required=True)
    parser.add_argument("--o1-sha256")
    parser.add_argument("--accepted-instrumented", type=Path)
    parser.add_argument("--defined-o1-against", type=Path)
    parser.add_argument("--canonical-o1-undefined", action="store_true")
    parser.add_argument("--plant-controls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(
            args.run_dir, args.control, args.legacy_record_manifest,
            args.surface_sha256, args.o1_sha256, args.plant_controls,
            args.accepted_instrumented, args.defined_o1_against,
            args.canonical_o1_undefined,
        )
    except (GateError, exchange.GateError, exchange.phase1.GateError,
            OSError, UnicodeError,
            struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
        code = 1
    else:
        code = 0 if result["status"] == "PASS" else 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

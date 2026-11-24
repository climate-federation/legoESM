#!/usr/bin/env python3
"""Admit the passive full-array kt=2 stage-3 traldf_iso_lap records."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as dyn_gate,
    nemo_testcase_l4_orca2_round332_source_order_gate as r332,
    nemo_testcase_l4_orca2_round333_tracer_gate as r333,
)

MAGIC = "NEMO_L4_R333LI"
HEADER_NAMES = (
    "version", "kt", "stage", "Kbb", "Kmm", "Krhs", "rank", "nx", "ny",
    "nz", "ntracer", "real_bits", "n_fields",
)
FOUR_D_FIELDS = (
    "pt_kbb", "rhs_before", "rhs_after", "rhs_increment", "dit", "djt", "dkt",
    "A11", "A22", "A13", "A23", "hmsku", "hmskv", "zfu", "zfv", "vmsku",
    "vmskv", "zahu_w", "zahv_w", "A31", "A32", "zfw_lower", "zfw_upper",
    "horizontal_seen", "vertical_seen", "rhs_seen", "divergence", "divisor",
    "rhs_preadd", "rhs_postadd",
)
THREE_D_FIELDS = (
    "e3t_3d", "e3u_3d", "e3v_3d", "tmask", "umask", "vmask", "wmask",
    "ahtu", "ahtv", "uslp", "vslp", "wslpi", "wslpj", "ah_wslp2", "akz",
    "a33_msku", "a33_mskv", "a33_ahu_w", "a33_ahv_w", "a33_pah",
    "a33_pakz_raw", "a33_ze3w2", "a33_zcoef0", "a33_pakz_final",
    "a33_seen", "rDt", "r1_Dt",
)
TWO_D_FIELDS = (
    "r3t_kmm", "r3u_kmm", "r3v_kmm", "e2_e1u", "e1_e2v", "e2u", "e1v",
    "e1u", "e2v", "e1t", "e2t", "e1e2t", "r1_e1e2t",
)
ONE_D_FIELDS = ("e3w_1d",)
FIELDS = FOUR_D_FIELDS + THREE_D_FIELDS + TWO_D_FIELDS + ONE_D_FIELDS
PLANTS = (
    "header", "field-order", "truncation", "checksum", "stamp", "time-level",
    "tracer-pass", "extent", "signed-zero", "wet", "increment", "nonfinite",
    "scheme-replay", "rhs-replay", "a33-replay",
)
OBSERVER_UNITS = ("dynzdf", "stprk3_stg", "dynvor", "zdfphy", "zdftke", "traldf_iso")
WRITER_UNITS = (
    "dynzdf_round328_writer", "tke_round331_writer", "round332_writer",
    "round333_traldf_writer",
)

GateError = r332.GateError
require = r332.require


def record_name(rank: int) -> str:
    return f"oracle_r333_traldf_kt00000002_r{rank:04d}.bin"


def _shape_for(name: str, header: dict[str, int]) -> tuple[int, ...]:
    base = (header["nx"], header["ny"], header["nz"])
    if name in FOUR_D_FIELDS:
        return (*base, header["ntracer"])
    if name in THREE_D_FIELDS:
        return base
    if name in TWO_D_FIELDS:
        return base[:2]
    if name in ONE_D_FIELDS:
        return (base[2],)
    raise GateError(f"unknown traldf field {name}")


def read_record(path: Path, *, plant: str | None = None,
                expected_shape: tuple[int, int, int] = (94, 152, 31)) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = r333._take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(f"={len(HEADER_NAMES)}i", r333._take(
        stream, 4 * len(HEADER_NAMES), "header"))
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant == "header":
        header["stage"] = 2
    elif plant == "time-level":
        header["Kmm"] = 1
    elif plant == "tracer-pass":
        header["ntracer"] = 1
    require(magic == MAGIC, f"bad magic {magic!r}")
    require((header["version"], header["kt"], header["stage"], header["real_bits"])
            == (1, 2, 3, 64), "wrong version/step/stage/precision")
    kbb, kmm, _kaa, krhs = r333.rk3_levels_for_step_stage(
        header["kt"], header["stage"])
    require((header["Kbb"], header["Kmm"], header["Krhs"])
            == (kbb, kmm, krhs), "time levels disagree with RK3 routing")
    require(header["rank"] in (0, 1), "wrong rank")
    require((header["nx"], header["ny"], header["nz"]) == expected_shape,
            "record lacks exact jpi,jpj,jpk extents")
    require(header["ntracer"] == 2, "record does not contain both T and S passes")
    require(header["n_fields"] == len(FIELDS), "wrong field count")
    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = r333._take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        metadata = list(struct.unpack("=9i", r333._take(stream, 36, f"{name} metadata")))
        if plant == "extent" and index == 0:
            metadata[1] -= 1
        array_rank, *dims, i0, j0, k0, n0 = metadata
        shape = _shape_for(name, header)
        padded = (*shape, *(1 for _ in range(4 - len(shape))))
        require((array_rank, *dims, i0, j0, k0, n0)
                == (len(shape), *padded, 1, 1, 1, 1),
                f"{name}: wrong rank, extent, or origin")
        checksum, = struct.unpack("=Q", r333._take(stream, 8, f"{name} checksum"))
        payload = r333._take(stream, int(np.prod(shape)) * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(shape, order="F")
        require(name not in fields, f"duplicate field {name}")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    if plant == "field-order":
        order[0], order[1] = order[1], order[0]
    require(tuple(order) == FIELDS, "field order differs from executing order")
    return {"header": header, "fields": fields, "checksums": checksums,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_checksums(record: dict) -> None:
    for name, values in record["fields"].items():
        observed = int(np.bitwise_xor.reduce(values.view(np.uint64).reshape(-1), initial=0))
        require(observed == record["checksums"][name], f"{name}: checksum mismatch")


def _bits_equal_on(actual: np.ndarray, expected: np.ndarray,
                   mask: np.ndarray, label: str) -> None:
    if expected.shape != actual.shape:
        expected = np.broadcast_to(expected, actual.shape)
    require(bool(np.array_equal(actual[mask].view(np.uint64),
                                expected[mask].view(np.uint64))),
            f"{label}: recorded statement does not replay bitwise")


def _replay_self_consistency(fields: dict[str, np.ndarray]) -> None:
    """Replay every recorded NEMO statement from its raw operands."""
    hseen = fields["horizontal_seen"] == 1.0
    vseen = fields["vertical_seen"] == 1.0

    pt = fields["pt_kbb"]
    dit = (np.roll(pt, -1, axis=0) - pt) * fields["umask"][..., None]
    djt = (np.roll(pt, -1, axis=1) - pt) * fields["vmask"][..., None]
    _bits_equal_on(fields["dit"], dit, hseen, "zdit")
    _bits_equal_on(fields["djt"], djt, hseen, "zdjt")
    dkt = np.zeros_like(pt)
    dkt[:, :, 1:-1, :] = ((pt[:, :, :-2, :] - pt[:, :, 1:-1, :])
                           * fields["wmask"][:, :, 1:-1, None])
    dkt_seen = hseen.copy()
    dkt_seen[1:, ...] |= hseen[:-1, ...]
    dkt_seen[:, 1:, ...] |= dkt_seen[:, :-1, ...]
    dkt_seen[:, :, 1:, :] |= dkt_seen[:, :, :-1, :]
    _bits_equal_on(fields["dkt"], dkt, dkt_seen, "zdkt")

    e3u = fields["e3u_3d"][..., None]
    e3v = fields["e3v_3d"][..., None]
    r3u = fields["r3u_kmm"][..., None, None]
    r3v = fields["r3v_kmm"][..., None, None]
    a11 = fields["e2_e1u"][..., None, None] * (
        e3u * (np.float64(1.0) + r3u * fields["umask"][..., None]))
    a22 = fields["e1_e2v"][..., None, None] * (
        e3v * (np.float64(1.0) + r3v * fields["vmask"][..., None]))
    _bits_equal_on(fields["A11"], a11, hseen, "A11")
    _bits_equal_on(fields["A22"], a22, hseen, "A22")
    hmsku = np.float64(1.0) / np.maximum(
        (np.roll(fields["wmask"], -1, axis=0) + np.roll(fields["wmask"], -1, axis=2))
        + (np.roll(np.roll(fields["wmask"], -1, axis=0), -1, axis=2)
           + fields["wmask"]), np.float64(1.0))
    hmskv = np.float64(1.0) / np.maximum(
        (np.roll(fields["wmask"], -1, axis=1) + np.roll(fields["wmask"], -1, axis=2))
        + (np.roll(np.roll(fields["wmask"], -1, axis=1), -1, axis=2)
           + fields["wmask"]), np.float64(1.0))
    hmsku = hmsku[..., None]
    hmskv = hmskv[..., None]
    a13 = (-fields["e2u"][..., None, None] * fields["uslp"][..., None]) * hmsku
    a23 = (-fields["e1v"][..., None, None] * fields["vslp"][..., None]) * hmskv
    _bits_equal_on(fields["hmsku"], hmsku, hseen, "horizontal u-mask quotient")
    _bits_equal_on(fields["hmskv"], hmskv, hseen, "horizontal v-mask quotient")
    _bits_equal_on(fields["A13"], a13, hseen, "A13")
    _bits_equal_on(fields["A23"], a23, hseen, "A23")
    dkt_ip1 = np.roll(dkt, -1, axis=0)
    dkt_jp1 = np.roll(dkt, -1, axis=1)
    dkt_kp1 = np.roll(dkt, -1, axis=2)
    zfu = fields["ahtu"][..., None] * (
        a11 * dit + a13 * ((dkt_ip1 + dkt_kp1)
                           + (np.roll(dkt_ip1, -1, axis=2) + dkt)))
    zfv = fields["ahtv"][..., None] * (
        a22 * djt + a23 * ((dkt_jp1 + dkt_kp1)
                           + (np.roll(dkt_jp1, -1, axis=2) + dkt)))
    _bits_equal_on(fields["zfu"], zfu, hseen, "zfu")
    _bits_equal_on(fields["zfv"], zfv, hseen, "zfv")

    ordinary = vseen.copy()
    ordinary[:, :, fields["pt_kbb"].shape[2] - 2:, :] = False
    vmsku = fields["wmask"] / np.maximum(
        (fields["umask"] + np.roll(np.roll(fields["umask"], 1, axis=0), -1, axis=2))
        + (np.roll(fields["umask"], 1, axis=0) + np.roll(fields["umask"], -1, axis=2)),
        np.float64(1.0))
    vmskv = fields["wmask"] / np.maximum(
        (fields["vmask"] + np.roll(np.roll(fields["vmask"], 1, axis=1), -1, axis=2))
        + (np.roll(fields["vmask"], 1, axis=1) + np.roll(fields["vmask"], -1, axis=2)),
        np.float64(1.0))
    zahu = ((fields["ahtu"] + np.roll(np.roll(fields["ahtu"], 1, axis=0), -1, axis=2))
            + (np.roll(fields["ahtu"], 1, axis=0)
               + np.roll(fields["ahtu"], -1, axis=2))) * vmsku
    zahv = ((fields["ahtv"] + np.roll(np.roll(fields["ahtv"], 1, axis=1), -1, axis=2))
            + (np.roll(fields["ahtv"], 1, axis=1)
               + np.roll(fields["ahtv"], -1, axis=2))) * vmskv
    a31 = (((-zahu * fields["e2t"][..., None]) * vmsku)
           * np.roll(fields["wslpi"], -1, axis=2))[..., None]
    a32 = (((-zahv * fields["e1t"][..., None]) * vmskv)
           * np.roll(fields["wslpj"], -1, axis=2))[..., None]
    _bits_equal_on(fields["vmsku"], vmsku[..., None], ordinary,
                   "vertical u-mask quotient")
    _bits_equal_on(fields["vmskv"], vmskv[..., None], ordinary,
                   "vertical v-mask quotient")
    _bits_equal_on(fields["zahu_w"], zahu[..., None], ordinary,
                   "vertical averaged u coefficient")
    _bits_equal_on(fields["zahv_w"], zahv[..., None], ordinary,
                   "vertical averaged v coefficient")
    _bits_equal_on(fields["A31"], a31, ordinary, "A31")
    _bits_equal_on(fields["A32"], a32, ordinary, "A32")
    dit_im1 = np.roll(dit, 1, axis=0)
    djt_jm1 = np.roll(djt, 1, axis=1)
    dit_kp1 = np.roll(dit, -1, axis=2)
    djt_kp1 = np.roll(djt, -1, axis=2)
    e3w_kp1 = np.roll(fields["e3w_1d"], -1)[None, None, :, None]
    r3t = fields["r3t_kmm"][..., None, None]
    vertical_a33 = ((fields["e1e2t"][..., None, None]
                     / (e3w_kp1 * (np.float64(1.0) + r3t)))
                    * np.roll(fields["wmask"], -1, axis=2)[..., None])
    vertical_a33 = ((vertical_a33
                     * (np.roll(fields["ah_wslp2"], -1, axis=2)[..., None]
                        - np.roll(fields["akz"], -1, axis=2)[..., None]))
                    * (pt - np.roll(pt, -1, axis=2)))
    zfw_upper = (a31 * ((dit + np.roll(dit_im1, -1, axis=2))
                        + (dit_im1 + dit_kp1))
                 + a32 * ((djt + np.roll(djt_jm1, -1, axis=2))
                          + (djt_jm1 + djt_kp1))) + vertical_a33
    _bits_equal_on(fields["zfw_upper"], zfw_upper, ordinary, "zfw upper")
    deepest = vseen & ~ordinary
    _bits_equal_on(fields["zfw_upper"], np.zeros_like(zfw_upper), deepest,
                   "deepest zfw upper")
    zfw_lower = np.zeros_like(zfw_upper)
    zfw_lower[:, :, 1:, :] = fields["zfw_upper"][:, :, :-1, :]
    _bits_equal_on(fields["zfw_lower"], zfw_lower, vseen, "zfw lower")

    divergence = ((fields["zfu"] - np.roll(fields["zfu"], 1, axis=0))
                  + (fields["zfv"] - np.roll(fields["zfv"], 1, axis=1)))
    divergence = divergence + (fields["zfw_lower"] - fields["zfw_upper"])
    _bits_equal_on(fields["divergence"], divergence, vseen, "divergence")
    e3t = fields["e3t_3d"][..., None]
    stretched_e3t = e3t * (np.float64(1.0) + r3t * fields["tmask"][..., None])
    divisor = fields["r1_e1e2t"][..., None, None] / stretched_e3t
    divisor = np.broadcast_to(divisor, fields["divisor"].shape)
    _bits_equal_on(fields["divisor"], divisor, vseen, "RHS divisor")
    increment = (divergence * fields["r1_e1e2t"][..., None, None]) / stretched_e3t
    rhs_post = fields["rhs_preadd"] + increment
    _bits_equal_on(fields["rhs_postadd"], rhs_post, vseen, "RHS addition")
    _bits_equal_on(fields["rhs_after"], fields["rhs_postadd"], vseen,
                   "final RHS versus recorded update")

    seen = fields["a33_seen"] == 1.0
    umask_i0 = np.roll(fields["umask"], 1, axis=2)
    umask_i1 = np.roll(fields["umask"], 1, axis=0)
    a33_msku = fields["wmask"] / np.maximum(
        (umask_i0 + umask_i1) + (np.roll(umask_i1, 1, axis=2) + fields["umask"]),
        np.float64(1.0))
    vmask_k0 = np.roll(fields["vmask"], 1, axis=2)
    vmask_j1 = np.roll(fields["vmask"], 1, axis=1)
    a33_mskv = fields["wmask"] / np.maximum(
        (vmask_k0 + vmask_j1) + (np.roll(vmask_j1, 1, axis=2) + fields["vmask"]),
        np.float64(1.0))
    _bits_equal_on(fields["a33_msku"], a33_msku, seen, "A33 u-mask quotient")
    _bits_equal_on(fields["a33_mskv"], a33_mskv, seen, "A33 v-mask quotient")
    ahtu_k0 = np.roll(fields["ahtu"], 1, axis=2)
    ahtu_i1 = np.roll(fields["ahtu"], 1, axis=0)
    a33_ahu = ((ahtu_k0 + ahtu_i1) +
               (np.roll(ahtu_i1, 1, axis=2) + fields["ahtu"])) * a33_msku
    ahtv_k0 = np.roll(fields["ahtv"], 1, axis=2)
    ahtv_j1 = np.roll(fields["ahtv"], 1, axis=1)
    a33_ahv = ((ahtv_k0 + ahtv_j1) +
               (np.roll(ahtv_j1, 1, axis=2) + fields["ahtv"])) * a33_mskv
    _bits_equal_on(fields["a33_ahu_w"], a33_ahu, seen, "A33 averaged u coefficient")
    _bits_equal_on(fields["a33_ahv_w"], a33_ahv, seen, "A33 averaged v coefficient")
    pah = ((a33_ahu * fields["wslpi"]) * fields["wslpi"]
           + (a33_ahv * fields["wslpj"]) * fields["wslpj"])
    _bits_equal_on(fields["a33_pah"], pah, seen, "A33 slope-square coefficient")
    u0 = ((fields["ahtu"] + np.roll(fields["ahtu"], 1, axis=2))
          / (fields["e1u"][..., None] * fields["e1u"][..., None]))
    u1_ahtu = np.roll(fields["ahtu"], 1, axis=0)
    u1_e1u = np.roll(fields["e1u"], 1, axis=0)[..., None]
    u1 = (u1_ahtu + np.roll(u1_ahtu, 1, axis=2)) / (u1_e1u * u1_e1u)
    v0 = ((fields["ahtv"] + np.roll(fields["ahtv"], 1, axis=2))
          / (fields["e2v"][..., None] * fields["e2v"][..., None]))
    v1_ahtv = np.roll(fields["ahtv"], 1, axis=1)
    v1_e2v = np.roll(fields["e2v"], 1, axis=1)[..., None]
    v1 = (v1_ahtv + np.roll(v1_ahtv, 1, axis=2)) / (v1_e2v * v1_e2v)
    pakz_raw = ((u0 + u1) + (v0 + v1)) * np.float64(0.25)
    _bits_equal_on(fields["a33_pakz_raw"], pakz_raw, seen, "A33 raw coefficient")
    e3w = fields["e3w_1d"][None, None, :]
    stretch = e3w * (np.float64(1.0) + fields["r3t_kmm"][..., None])
    ze3w2 = stretch * stretch
    _bits_equal_on(fields["a33_ze3w2"], ze3w2, seen, "A33 ze3w_2")
    zcoef0 = fields["rDt"] * (pakz_raw + fields["a33_pah"] / ze3w2)
    _bits_equal_on(fields["a33_zcoef0"], zcoef0, seen, "A33 zcoef0")
    pakz_final = (np.maximum(zcoef0 - np.float64(0.5), np.float64(0.0))
                  * ze3w2) * fields["r1_Dt"]
    _bits_equal_on(fields["a33_pakz_final"], pakz_final, seen,
                   "A33 limited coefficient")
    _bits_equal_on(fields["akz"], fields["a33_pakz_final"], seen,
                   "A33 output coefficient")


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    require(tuple(patches) == OBSERVER_UNITS, "wrong observer unit order")
    ref = (ref_bld / "ppsrc/nemo/traldf_iso.f90").read_text().splitlines()
    new = (new_bld / "ppsrc/nemo/traldf_iso.f90").read_text().splitlines()
    expected = [line[1:] for line in patches["traldf_iso"].read_text().splitlines()
                if line.startswith("+") and not line.startswith("+++")]
    observed = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
            None, ref, new, autojunk=False).get_opcodes():
        if tag == "insert":
            observed.extend(new[j1:j2])
    mismatch = next((index for index, pair in enumerate(zip(expected, observed))
                     if pair[0] != pair[1]), min(len(expected), len(observed)))
    require(expected == observed,
            f"traldf patch ledger mismatch expected={len(expected)} observed={len(observed)} "
            f"first={mismatch}")
    return dyn_gate.same_program_observers(
        new_bld, ref_bld, patches, WRITER_UNITS, new_cfg=new_cfg, ref_cfg=ref_cfg)


def run_gate(root: Path, *, expect_commit: str, plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    produced = (root / "producer_commit.txt").read_text(encoding="ascii").strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit, "producer commit moved")
    inherited = r333.run_gate(root, expect_commit=expect_commit)
    require(inherited["status"] == "PASS_R333_TRACER_ACQUISITION",
            "inherited round-333 admission moved")
    records = {}
    signed_zeros = {}
    wet_counts = {}
    for rank in (0, 1):
        parser_plant = plant if rank == 0 and plant in (
            "header", "field-order", "truncation", "time-level", "tracer-pass", "extent"
        ) else None
        record = read_record(root / record_name(rank), plant=parser_plant)
        require(record["header"]["rank"] == rank, "rank tag moved")
        if rank == 0 and plant == "checksum":
            record["fields"]["zfu"][2, 2, 1, 0] = np.nextafter(
                record["fields"]["zfu"][2, 2, 1, 0], np.inf)
        _verify_checksums(record)
        fields = record["fields"]
        if rank == 0 and plant == "nonfinite":
            fields["dit"][2, 2, 1, 0] = np.nan
        require(all(bool(np.isfinite(values).all()) for values in fields.values()),
                f"rank {rank}: non-finite record payload")
        if rank == 0 and plant == "rhs-replay":
            fields["divergence"][2, 2, 1, 0] = np.nextafter(
                fields["divergence"][2, 2, 1, 0], np.inf)
        if rank == 0 and plant == "scheme-replay":
            fields["zfu"][2, 2, 1, 0] = np.nextafter(
                fields["zfu"][2, 2, 1, 0], np.inf)
        if rank == 0 and plant == "a33-replay":
            fields["a33_zcoef0"][2, 2, 1] = np.nextafter(
                fields["a33_zcoef0"][2, 2, 1], np.inf)
        _replay_self_consistency(fields)
        wet = r333._full_wet_mask(root, rank)
        if rank == 0 and plant == "wet":
            index = tuple(np.argwhere(wet)[0])
            fields["rhs_seen"][(*index, 0)] = 0.0
        for tracer in range(2):
            require(bool(np.all(fields["rhs_seen"][..., tracer][wet] == 1.0)),
                    f"rank {rank} tracer {tracer}: wet RHS coverage incomplete")
            require(bool(np.any(fields["rhs_increment"][..., tracer][wet] != 0.0)),
                    f"rank {rank} tracer {tracer}: passive increment")
        if rank == 0 and plant == "increment":
            index = (*tuple(np.argwhere(wet)[0]), 0)
            fields["rhs_increment"][index] = np.nextafter(fields["rhs_increment"][index], np.inf)
        expected_increment = fields["rhs_after"] - fields["rhs_before"]
        require(np.array_equal(fields["rhs_increment"].view(np.uint64),
                               expected_increment.view(np.uint64)),
                f"rank {rank}: per-pass increment differs from recorded endpoints")
        if rank == 0 and plant == "signed-zero":
            for values in fields.values():
                values[values == 0.0] = 0.0
        sign_count = sum(int(np.count_nonzero((values == 0.0) & np.signbit(values)))
                         for values in fields.values())
        require(sign_count > 0, f"rank {rank}: signed zeros were normalized")
        records[rank] = record
        signed_zeros[str(rank)] = sign_count
        wet_counts[str(rank)] = int(np.count_nonzero(wet))
    return {
        "format": "nemo-testcase-l4-orca2-round333-traldf-internal-v1",
        "status": "PASS_R333_TRALDF_INTERNAL_ACQUISITION",
        "producer_commit": produced,
        "inherited_status": inherited["status"],
        "tracer_passes": ["T", "S"],
        "kpass_loop": False,
        "recorded_fields": list(FIELDS),
        "records": {record_name(rank): records[rank]["sha256"] for rank in records},
        "signed_zero_census": signed_zeros,
        "wet_cell_census": wet_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--same-program", type=Path)
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", action="append", default=[])
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    args = parser.parse_args()
    try:
        if args.same_program:
            entries = [item.split("=", 1) for item in args.observer_patch]
            result = same_program(args.same_program, args.reference_bld,
                                  {unit: Path(path) for unit, path in entries},
                                  new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(args.record_dir is not None and args.expect_commit is not None,
                    "admission arguments incomplete")
            result = run_gate(args.record_dir, expect_commit=args.expect_commit,
                              plant=args.plant)
            require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, UnicodeError, ValueError, struct.error) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

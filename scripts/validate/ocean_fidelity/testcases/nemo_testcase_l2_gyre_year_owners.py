#!/usr/bin/env python3
"""WHO OWNS THE GYRE DAY-30 GAP -- the day-by-day walk, the decomposition,
the surface-forcing STATEMENT gate, and the threshold-switch trace.

Preregistration: ``docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_owners.md``.
Read it first.  Every constant, expectation and falsifier below is fixed there.

WHY THIS EXISTS.  The year receipt scored GYRE DISTINGUISHABLE at every day and
left the OWNER unnamed: the gap is `1.4241e-02` K at day 30 against a floor of
`6.8065e-10` K, and `2.768e-03` K of it is already present after TEN steps.
Nothing in the campaign resolves the interval between.

MODES
  --step-gap N       legoESM stepped N steps, scored against NEMO's per-step
                     ENTRY dumps.  The certified card's own writer covers
                     kstp = nit000..nit000+59, so days 0..10 are already on
                     disk at STEP resolution -- no new NEMO run is needed for
                     them.  (cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90)
  --decompose DAY    the gap at DAY by FIELD, by DEPTH and by REGION, plus the
                     fraction-of-NEMO's-own-from-rest-signal table that is the
                     only one of the three that can name a LEADING FIELD.
  --process-record   validates the Round-123 compiled-order NEMO process
                     frames for steps 1081..1440 before they may be scored.
  --produce-process-trace
                     runs legoESM independently from rest through step 1440
                     and records the matching production-JIT boundaries.
  --process-budget   admits both traces, closes day 180 -> 240, and ranks the
                     signed day-240 temperature carry by process.
  --vertical-record  validates the Round-125 ``tra_zdf`` internal records
                     against the already admitted process trajectory.
  --forcing-gate     legoESM's CURRENT surface forcing against the LITERAL
                     usrdef_sbc transcription, BIT-EXACT, evaluated on NEMO's
                     OWN state at every day boundary the record holds.  This is
                     the statement test the kt=1 dump cannot perform, because
                     kt=1 samples the seasonal clock at ONE of its 2160 values.
  --switch-trace     the enhanced-vertical-diffusion trigger mask, step by
                     step, on two members; reports the FIRST step at which the
                     two masks differ.
  --self-check       the arithmetic and every plant.

PLANTS (each exits NON-ZERO; each is exercised by the committed unit test)
  forcing-phase            evaluates legoESM's forcing one step late
  forcing-qsr-pi           swaps usrdef_sbc's literal 3.1415 for rpi
  forcing-nyear            restores the (nyear-1) subtraction, as if year 2
  forcing-stress-transpose swaps the geographic stress pair the model consumes
  switch-blind             freezes the EVD trigger mask, so the trace must
                           REFUSE rather than report "no crossing"
  process-stamp            supplies a wrong producer commit
  process-truncation       removes one binary64 word from the first frame
  process-sbc-ulp          moves one post-SBC RHS value by one ULP
  process-sbc-effect       moves post-SBC RHS by the minimum searched amount
                           that survives into a decoded temperature row
  process-trajectory-ulp   breaks the first Taa-to-next-Tbb chain by one ULP
  lego-process-stamp       supplies a wrong legoESM producer commit
  lego-process-ulp         moves one decoded legoESM process boundary by 1 ULP
  lego-process-effect      exercises the stored full-production effect plant
  vertical-stamp           supplies a wrong vertical-record producer commit
  vertical-truncation      removes one binary64 word from an internal record
  vertical-matrix-ulp      moves one consumed matrix coefficient by one ULP
  vertical-trajectory-ulp  breaks one internal/process Tbb comparison by 1 ULP

``--plant day-offset`` is NOT a gate plant and never exits non-zero: the
day-by-day walk and the per-step walk report numbers, they do not carry a bar.
It reads NEMO one day late so the reader can see how much a day-misalignment
would move the table, and it is labelled a DIAGNOSTIC everywhere it appears.
An earlier version of this docstring claimed every plant exits non-zero; that
was false for this one, and an independent review said so.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

CASE = "GYRE-zco"
DT_S = 14400.0
STEPS_PER_DAY = 6
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners")
YEAR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest")
YEAR_HEAD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest_head")
DEFAULT_MESH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/mesh_mask.nc")
# The per-step ENTRY dumps.  The year run's own pristine member wrote 60 of
# them; the ten-step ladder's run wrote 10.  Both are the same instrument.
DEFAULT_ENTRY_ROOT = YEAR_ROOT / "nemo_pristine"
# usrdef_sbc.f90:140 -- the emp branch latitude, and :192-193 the wind's half
# period.  These are READ OFF THE SOURCE, not chosen.
EMP_SPLIT_LAT_DEG = 37.2
WIND_BAND_LAT_DEG = (15.0, 29.0)
FIELDS = ("T", "S", "u", "v", "ssh")
# The MODEL-FACING rows.  NEMO carries qns and qsr apart and the shared
# forcing object carries their materialized SUM, so "q_net" is the row the
# model actually consumes; sw_down pins qsr beside it, and the two together
# pin qns without inventing a difference of rounded sums.
SBC_FIELDS = ("qsr", "q_net", "emp", "utau", "vtau")
# Round 123 extends this owner instrument with the raw, compiled-order tracer
# boundaries needed for a day-180-to-240 process budget.  These are record
# FORMAT constants, frozen in the round-123 preregistration, not physics
# tolerances.
PROCESS_MAGIC = "NEMO_L2_R123PROC"
PROCESS_START_STEP = 1081
PROCESS_END_STEP = 1440
PROCESS_JPI = 36
PROCESS_JPJ = 26
PROCESS_JPK = 31
PROCESS_STORAGE_BITS = 64
PROCESS_HEADER_INTS = 11
PROCESS_RECORD_BYTES = (
    16 + PROCESS_HEADER_INTS * 4 + 8
    + (6 * PROCESS_JPI * PROCESS_JPJ * PROCESS_JPK
       + 3 * PROCESS_JPI * PROCESS_JPJ) * 8
)
PROCESS_RESTART_HASHES = {
    "GYRE_OMIP_L2_P3_00001080_restart.nc":
        "6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976",
    "GYRE_OMIP_L2_P3_00001440_restart.nc":
        "96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a",
}
PROCESS_ROWS = (
    "geometry", "advection", "surface_boundary", "shortwave",
    "lateral_diffusion", "vertical_diffusion",
)
LEGO_PROCESS_FIELDS = (
    "Tbb", "q_Kbb", "q_Kmm", "q_Kaa", "B0", "Badv", "Bsbc", "Bqsr",
    "Bldf", "Bpre", "Taa",
)
LEGO_PROCESS_TRACE_STEPS = PROCESS_END_STEP - PROCESS_START_STEP + 1
DEFAULT_IMMUTABLE_GYRE_YEAR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/gyre")
# Round 125 reuses the existing self-describing ``tra_zdf`` record.  The
# source writer remains armed for steps 1--2; one additive source line widens
# it to the 360-step day-180-to-240 interval.  These are byte-layout constants,
# not scientific tolerances.
VERTICAL_RECORD_EARLY_STEPS = (1, 2)
VERTICAL_RECORD_STEPS = (
    *VERTICAL_RECORD_EARLY_STEPS,
    *range(PROCESS_START_STEP, PROCESS_END_STEP + 1),
)
VERTICAL_RECORD_BYTES = (
    80 + 44 * 32 + 11 * 8
    + 27 * PROCESS_JPI * PROCESS_JPJ * PROCESS_JPK * 8
    + 3 * PROCESS_JPI * PROCESS_JPJ * (PROCESS_JPK - 1) * 8
    + 3 * PROCESS_JPI * PROCESS_JPJ * 8
)
VERTICAL_REQUIRED_FIELDS = (
    "T_Kbb_in", "T_Krhs_in",
    "e3t_Kbb", "e3t_Kmm", "e3t_Kaa", "e3w_Kmm",
    "r3t_Kbb", "r3t_Kmm", "r3t_Kaa",
    "avt", "ah_wslp2", "akz", "zwt_mix",
    "zwi", "zwd", "zws", "zwt_lu",
    "rhs_T", "fwd_T", "sol_T_pre_clamp", "sol_T_post_clamp",
)
DEFAULT_PROCESS_RECORD_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123/"
    "oracle_process_budget")
PROCESS_RECORD_COMMIT = "af3f7215060fc17c71adc6794817c710df8ee471"
DEFAULT_ADMITTED_YEAR_RUN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/"
    "nemo_seed0")
# The geographic stress rotation cannot be a BIT-EXACT row: recovering the
# native pair from tau_x/tau_y in binary64 costs about 1e-17 Pa because
# cos^2 + sin^2 is not exactly 1.  So it is scored as a DISCRIMINATION instead
# -- how much further from the literal a TRANSPOSED pair lands than the
# identity does.  Self-calibrating, no absolute tolerance, and it cannot pass
# vacuously on a domain where the two are indistinguishable.
ROTATION_ROUNDTRIP_MIN_RATIO = 1.0e6


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _year():
    return _load("_gyre_year", "nemo_testcase_l2_gyre_year_fromrest.py")


def _gate():
    return _load("nemo_testcase_l2_gyre_phase3_gate",
                 "nemo_testcase_l2_gyre_phase3_gate.py")


def _trazdf():
    return _load("_gyre_round35_trazdf",
                 "nemo_testcase_l2_gyre_round35_trazdf_matrix.py")


def _round16():
    _load("nemo_testcase_l2_gyre_round15_eligibility",
          "nemo_testcase_l2_gyre_round15_eligibility.py")
    return _load("nemo_testcase_l2_gyre_round16_discriminator",
                 "nemo_testcase_l2_gyre_round16_discriminator.py")


def _policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    import jax
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")


def _rms(values, mask) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values[mask] ** 2)))


# --------------------------------------- day-180-to-240 process boundaries ---
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _take(stream: io.BytesIO, count: int, label: str) -> bytes:
    payload = stream.read(count)
    require(len(payload) == count, f"process record truncated in {label}")
    return payload


def _process_xyz(stream: io.BytesIO, label: str) -> np.ndarray:
    count = PROCESS_JPI * PROCESS_JPJ * PROCESS_JPK
    values = np.frombuffer(_take(stream, count * 8, label), dtype="=f8")
    full = values.reshape(
        (PROCESS_JPI, PROCESS_JPJ, PROCESS_JPK), order="F")
    return full[2:-2, 2:-2].transpose(1, 0, 2).copy()


def _process_xy(stream: io.BytesIO, label: str) -> np.ndarray:
    count = PROCESS_JPI * PROCESS_JPJ
    values = np.frombuffer(_take(stream, count * 8, label), dtype="=f8")
    full = values.reshape((PROCESS_JPI, PROCESS_JPJ), order="F")
    return full[2:-2, 2:-2].T.copy()


def read_process_record(path: Path, *, truncate: bool = False) -> dict:
    """Read one frozen Round-123 NEMO process-boundary frame.

    The crop and transpose are the same NEMO-(i,j,k) to model-(j,i,k)
    operation used by ``read_entry`` in the certified phase-3 gate.  The
    unused NEMO bottom level remains in the returned arrays; scoring crops it
    to the card's 30-level wet mask, never pads the candidate.
    """
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if truncate:
        raw = raw[:-8]
    require(len(raw) == PROCESS_RECORD_BYTES,
            f"{path.name}: {len(raw)} bytes, expected "
            f"{PROCESS_RECORD_BYTES}")
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii")
    header = struct.unpack(
        "=11i", _take(stream, PROCESS_HEADER_INTS * 4, "header"))
    (version, kstp, kstg, kbb, kmm, krhs, kaa, nx, ny, nz,
     storage_bits) = header
    require(magic == PROCESS_MAGIC, f"{path.name}: bad magic {magic!r}")
    require(version == 1, f"{path.name}: version {version}, expected 1")
    require(kstg == 3, f"{path.name}: stage {kstg}, expected 3")
    require((nx, ny, nz, storage_bits) == (
        PROCESS_JPI, PROCESS_JPJ, PROCESS_JPK, PROCESS_STORAGE_BITS),
        f"{path.name}: dimensions/storage {(nx, ny, nz, storage_bits)}")
    match = re.fullmatch(r"oracle_process_budget_kt(\d{8})\.bin", path.name)
    require(match is not None, f"{path.name}: wrong process-record filename")
    require(int(match.group(1)) == kstp,
            f"{path.name}: filename/header step mismatch {kstp}")
    require(all(1 <= level <= 3 for level in (kbb, kmm, krhs, kaa)),
            f"{path.name}: time-level slot outside 1..3")
    rdt = struct.unpack("=d", _take(stream, 8, "rDt"))[0]
    require(rdt == DT_S, f"{path.name}: rDt {rdt}, expected {DT_S}")
    fields = {
        "Tbb": _process_xyz(stream, "Tbb"),
        "r3t_Kbb": _process_xy(stream, "r3t Kbb"),
        "r3t_Kmm": _process_xy(stream, "r3t Kmm"),
        "r3t_Kaa": _process_xy(stream, "r3t Kaa"),
        "rhs_after_advection": _process_xyz(stream, "RHS after advection"),
        "rhs_after_surface_boundary": _process_xyz(
            stream, "RHS after surface boundary"),
        "rhs_after_shortwave": _process_xyz(stream, "RHS after shortwave"),
        "rhs_after_lateral_diffusion": _process_xyz(
            stream, "RHS after lateral diffusion"),
        "Taa": _process_xyz(stream, "Taa after vertical diffusion"),
    }
    require(stream.read(1) == b"", f"{path.name}: trailing bytes")
    for name, values in fields.items():
        require(np.all(np.isfinite(values)),
                f"{path.name}: {name} contains non-finite values")
    return {
        "path": str(path), "sha256": digest, "header": header, "kstp": kstp,
        "rDt": rdt, **fields,
    }


def process_temperature_rows(record: dict) -> dict[str, np.ndarray]:
    """Decode the six preregistered temperature-budget rows for one step."""
    nlev = record["Tbb"].shape[-1] - 1
    tbb = np.asarray(record["Tbb"][..., :nlev], dtype=np.float64)
    taa = np.asarray(record["Taa"][..., :nlev], dtype=np.float64)
    qbb = (1.0 + np.asarray(record["r3t_Kbb"], dtype=np.float64))[..., None]
    qmm = (1.0 + np.asarray(record["r3t_Kmm"], dtype=np.float64))[..., None]
    qaa = (1.0 + np.asarray(record["r3t_Kaa"], dtype=np.float64))[..., None]
    base = qbb * tbb

    def accumulated(name: str) -> np.ndarray:
        rhs = np.asarray(record[name][..., :nlev], dtype=np.float64)
        return (base + record["rDt"] * qmm * rhs) / qaa

    b0 = base / qaa
    badv = accumulated("rhs_after_advection")
    bsbc = accumulated("rhs_after_surface_boundary")
    bqsr = accumulated("rhs_after_shortwave")
    bldf = accumulated("rhs_after_lateral_diffusion")
    rows = {
        "geometry": b0 - tbb,
        "advection": badv - b0,
        "surface_boundary": bsbc - badv,
        "shortwave": bqsr - bsbc,
        "lateral_diffusion": bldf - bqsr,
        "vertical_diffusion": taa - bldf,
    }
    combined = np.zeros_like(tbb)
    for name in PROCESS_ROWS:
        combined = combined + rows[name]
    rows["rounding_closure"] = (taa - tbb) - combined
    return rows


def _different_cells(left: np.ndarray, right: np.ndarray,
                     mask: np.ndarray) -> int:
    return int(np.count_nonzero(
        np.asarray(left)[mask].view(np.uint64)
        != np.asarray(right)[mask].view(np.uint64)))


def _record_manifest(root: Path, records: list[Path], manifest_name: str,
                     expected_set: str) -> dict[str, str]:
    manifest = root / manifest_name
    require(manifest.is_file(), f"missing {manifest.name}")
    rows: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        words = line.split()
        require(len(words) == 2,
                f"{manifest.name}:{number}: malformed sha256 row")
        name = Path(words[1].lstrip("*")).name
        require(re.fullmatch(r"[0-9a-f]{64}", words[0]) is not None,
                f"{manifest.name}:{number}: malformed digest")
        require(name not in rows, f"{manifest.name}: duplicate {name}")
        rows[name] = words[0]
    require(set(rows) == {path.name for path in records},
            f"{manifest.name} file set differs from {expected_set}")
    return rows


def _check_record_stamp(root: Path, expected_commit: str, manifest_name: str,
                        stamp_name: str) -> None:
    manifest = root / manifest_name
    stamp = root / stamp_name
    require(stamp.is_file(), f"missing {stamp.name}")
    words = stamp.read_text(encoding="utf-8").split()
    require(len(words) == 3, f"{stamp.name}: malformed stamp")
    require(words[0] == _sha256(manifest),
            f"{stamp.name}: manifest digest mismatch")
    require(words[1] == expected_commit,
            f"{stamp.name}: producer commit mismatch")
    require(words[2] == manifest.name,
            f"{stamp.name}: stamped filename mismatch")


def _process_manifest(root: Path, records: list[Path]) -> dict[str, str]:
    return _record_manifest(
        root, records, "process_records.sha256", "steps 1081..1440")


def _check_process_stamp(root: Path, expected_commit: str) -> None:
    _check_record_stamp(root, expected_commit, "process_records.sha256",
                        "process_records.stamp")


def _resolved_process_card(path: Path) -> dict[str, bool]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    patterns = {
        "itend_1440": r"number of the last time step\s+nn_itend\s*=\s*1440\b",
        "dt_14400": r"ocean time step\s+rn_Dt\s*=\s*14400(?:\.0+)?\b",
        "tiling_off": r"Tiling \(T\) or not \(F\)\s+ln_tile\s*=\s*F\b",
        "qsr_on": r"Light penetration in temperature Eq\.\s+ln_traqsr\s*=\s*T\b",
        "bdy_off": r"open boundaries not used \(ln_bdy = F\)",
        "bbc_off": r"geothermal heating at ocean bottom\s+ln_trabbc\s*=\s*F\b",
        "bbl_off": r"bottom boundary layer flag\s+ln_trabbl\s*=\s*F\b",
        "dmp_off": r"Apply relaxation\s+or not\s+ln_tradmp\s*=\s*F\b",
        "mfc_off": r"convection mass flux \(mfc\)\s+ln_zdfmfc\s*=\s*F\b",
        "osm_off": r"OSMOSIS-OBL closure \(OSM\)\s+ln_zdfosm\s*=\s*F\b",
        "npc_off": r"non-penetrative convection \(npc\)\s+ln_zdfnpc\s*=\s*F\b",
    }
    rows = {name: re.search(pattern, text) is not None
            for name, pattern in patterns.items()}
    require(all(rows.values()),
            "resolved process card differs: "
            + ", ".join(name for name, ok in rows.items() if not ok))
    return rows


def _process_sbc_ulp_control(record: dict, mask: np.ndarray) -> dict:
    """Move one RHS bit and prove the decoded boundary is not self-blind."""
    before = np.asarray(record["rhs_after_surface_boundary"],
                        dtype=np.float64)
    adv = np.asarray(record["rhs_after_advection"], dtype=np.float64)
    for j, i, k in np.argwhere(mask):
        old = before[j, i, k]
        for direction in (np.inf, -np.inf):
            new = np.nextafter(old, direction)
            if not np.isfinite(new) or new == old:
                continue
            old_delta = old - adv[j, i, k]
            new_delta = new - adv[j, i, k]
            if (np.asarray(old_delta).view(np.uint64)
                    == np.asarray(new_delta).view(np.uint64)):
                continue
            planted = dict(record)
            planted_sbc = np.array(before, copy=True)
            planted_sbc[j, i, k] = new
            planted["rhs_after_surface_boundary"] = planted_sbc
            original_rows = process_temperature_rows(record)
            planted_rows = process_temperature_rows(planted)
            moved = {name: _different_cells(
                original_rows[name], planted_rows[name], mask)
                for name in (*PROCESS_ROWS, "rounding_closure")}
            return {
                "index_jik": [int(j), int(i), int(k)],
                "old_uint64": int(np.asarray(old).view(np.uint64)),
                "new_uint64": int(np.asarray(new).view(np.uint64)),
                "raw_surface_rhs_increment_moved": True,
                "decoded_temperature_rows_moved": moved,
            }
    raise GateError("one-ULP surface-boundary plant is inert in every wet cell")


def _process_sbc_effect_control(record: dict, mask: np.ndarray) -> dict:
    """Prove an RHS boundary perturbation propagates into the budget rows.

    One RHS ULP is a parser control, but it is normally far below one ULP of
    the roughly 20-K accumulated temperature expression.  This control starts
    at one temperature ULP mapped back into RHS units and increases only until
    the decoded surface-boundary row moves.  The chosen perturbation and every
    moved row are reported; no fixed scientific tolerance is introduced.
    """
    baseline = process_temperature_rows(record)
    before = np.asarray(record["rhs_after_surface_boundary"],
                        dtype=np.float64)
    qmm = 1.0 + np.asarray(record["r3t_Kmm"], dtype=np.float64)
    qaa = 1.0 + np.asarray(record["r3t_Kaa"], dtype=np.float64)
    tbb = np.asarray(record["Tbb"], dtype=np.float64)
    for j, i, k in np.argwhere(mask):
        scale = max(abs(float(tbb[j, i, k])), 1.0)
        temperature_ulp = float(np.spacing(scale))
        denominator = record["rDt"] * abs(float(qmm[j, i]))
        if denominator == 0.0:
            continue
        mapped = temperature_ulp * abs(float(qaa[j, i])) / denominator
        rhs_ulp = abs(float(np.spacing(abs(before[j, i, k]))))
        base_delta = max(mapped, rhs_ulp)
        for multiplier in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            new = before[j, i, k] + multiplier * base_delta
            if not np.isfinite(new) or new == before[j, i, k]:
                continue
            planted = dict(record)
            planted_sbc = np.array(before, copy=True)
            planted_sbc[j, i, k] = new
            planted["rhs_after_surface_boundary"] = planted_sbc
            planted_rows = process_temperature_rows(planted)
            moved = {name: _different_cells(
                baseline[name], planted_rows[name], mask)
                for name in (*PROCESS_ROWS, "rounding_closure")}
            if moved["surface_boundary"]:
                return {
                    "index_jik": [int(j), int(i), int(k)],
                    "rhs_delta": float(new - before[j, i, k]),
                    "mapped_temperature_ulp_K": temperature_ulp,
                    "starting_rhs_delta": base_delta,
                    "multiplier": multiplier,
                    "decoded_temperature_rows_moved": moved,
                }
    raise GateError("effect-scale surface-boundary plant is inert")


def validate_process_record(root: Path, expected_commit: str,
                            *, plant: str | None = None) -> dict:
    """Admit all 360 passive NEMO frames, or exercise one named plant."""
    root = Path(root)
    expected_names = [
        f"oracle_process_budget_kt{step:08d}.bin"
        for step in range(PROCESS_START_STEP, PROCESS_END_STEP + 1)
    ]
    records = [root / name for name in expected_names]
    observed = sorted(root.glob("oracle_process_budget_kt*.bin"))
    require([path.name for path in observed] == expected_names,
            "process-record set is not exactly steps 1081..1440")
    producer = (root / "producer_commit.txt").read_text(
        encoding="utf-8").strip()
    require(producer == expected_commit,
            "producer_commit.txt differs from --expect-commit")
    manifest = _process_manifest(root, records)

    if plant == "process-stamp":
        try:
            _check_process_stamp(root, "0" * 40)
        except GateError as error:
            return {"status": "PLANT-FIRED", "plant": plant,
                    "reason": str(error)}
        raise GateError("process-stamp plant stayed green")
    _check_process_stamp(root, expected_commit)
    if plant == "process-truncation":
        try:
            read_process_record(records[0], truncate=True)
        except GateError as error:
            return {"status": "PLANT-FIRED", "plant": plant,
                    "reason": str(error)}
        raise GateError("process-truncation plant stayed green")

    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    mask = gate.expected_masks(build_nemo_testcase_card(CASE))["T"]
    require(mask.shape == (PROCESS_JPJ - 4, PROCESS_JPI - 4,
                           PROCESS_JPK - 1),
            f"process mask shape {mask.shape} differs from frozen layout")

    first = read_process_record(records[0])
    if plant == "process-sbc-ulp":
        control = _process_sbc_ulp_control(first, mask)
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": control}
    if plant == "process-sbc-effect":
        control = _process_sbc_effect_control(first, mask)
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": control}
    if plant == "process-trajectory-ulp":
        second = read_process_record(records[1])
        require(_different_cells(first["Taa"][..., :mask.shape[-1]],
                                 second["Tbb"][..., :mask.shape[-1]],
                                 mask) == 0,
                "unplanted first/second process frames do not chain")
        planted = np.array(first["Taa"], copy=True)
        j, i, k = np.argwhere(mask)[0]
        planted[j, i, k] = np.nextafter(planted[j, i, k], np.inf)
        moved = _different_cells(planted[..., :mask.shape[-1]],
                                 second["Tbb"][..., :mask.shape[-1]], mask)
        require(moved == 1,
                f"trajectory ULP plant moved {moved} cells, expected 1")
        return {"status": "PLANT-FIRED", "plant": plant,
                "chained_cells_unequal": moved,
                "index_jik": [int(j), int(i), int(k)]}

    require(plant in (None, "none"), f"unknown process plant {plant!r}")
    activity = {name: 0 for name in PROCESS_ROWS}
    max_rounding_closure = 0.0
    chained_cells_unequal = 0
    previous_taa = None
    headers = []
    digests = {}
    for expected_step, path in enumerate(records, PROCESS_START_STEP):
        record = first if expected_step == PROCESS_START_STEP \
            else read_process_record(path)
        require(record["kstp"] == expected_step,
                f"{path.name}: step sequence mismatch")
        require(record["sha256"] == manifest[path.name],
                f"{path.name}: sha256 manifest mismatch")
        digests[path.name] = record["sha256"]
        if expected_step in (PROCESS_START_STEP, PROCESS_END_STEP):
            headers.append(list(record["header"]))
        if previous_taa is not None:
            chained_cells_unequal += _different_cells(
                previous_taa[..., :mask.shape[-1]],
                record["Tbb"][..., :mask.shape[-1]], mask)
        previous_taa = record["Taa"]
        rows = process_temperature_rows(record)
        for name in PROCESS_ROWS:
            activity[name] += int(np.count_nonzero(rows[name][mask]))
        max_rounding_closure = max(
            max_rounding_closure,
            float(np.max(np.abs(rows["rounding_closure"][mask]))),
        )
    require(chained_cells_unequal == 0,
            f"process frames fail Taa-to-next-Tbb chain in "
            f"{chained_cells_unequal} wet cells")
    for name in PROCESS_ROWS[1:]:
        require(activity[name] > 0,
                f"active process row {name} never moves a wet cell")

    resolved = _resolved_process_card(root / "ocean.output")
    restart_rows = {}
    for name, expected in PROCESS_RESTART_HASHES.items():
        observed_digest = _sha256(root / name)
        require(observed_digest == expected,
                f"passive restart {name} is {observed_digest}, expected "
                f"{expected}")
        restart_rows[name] = observed_digest
    binary_words = (root / "binary.sha256").read_text(
        encoding="utf-8").split()
    require(binary_words, "binary.sha256 is empty")
    binary_digest = _sha256(root / "nemo")
    require(binary_words[0] == binary_digest,
            "run binary differs from binary.sha256")

    from legoesm.ocean.fidelity.provenance import worktree_stamp
    return {
        "format": "gyre-year-owners-process-record-v1",
        "status": "PASS", "case": CASE, "plant": plant,
        "root": str(root), "producer_commit": producer,
        "worktree": worktree_stamp(), "binary_sha256": binary_digest,
        "layout": {
            "magic": PROCESS_MAGIC, "record_count": len(records),
            "bytes_per_record": PROCESS_RECORD_BYTES,
            "total_record_bytes": sum(path.stat().st_size for path in records),
            "first_and_last_headers": headers,
            "records_manifest_sha256": _sha256(
                root / "process_records.sha256"),
        },
        "controls": {
            "resolved_card": resolved,
            "restart_sha256": restart_rows,
            "chained_cells_unequal": chained_cells_unequal,
            "active_row_nonzero_cell_visits": activity,
            "max_abs_step_rounding_closure_K": max_rounding_closure,
        },
        "first_record_sha256": digests[expected_names[0]],
        "last_record_sha256": digests[expected_names[-1]],
    }


# ----------------------- day-180-to-240 tra_zdf internal record admission ---
def _vertical_field(record: dict, name: str, nlev: int | None = None,
                    *, transpose: bool = True) -> np.ndarray:
    """Return one checked ``tra_zdf`` field on the NEMO interior box."""
    trazdf = _trazdf()
    values = np.asarray(trazdf._box(record, name, nlev), dtype=np.float64)
    if not transpose:
        return values
    if values.ndim == 2:
        return values.T.copy()
    return np.transpose(values, (1, 0, 2)).copy()


def _vertical_calibration(record: dict) -> dict[str, int]:
    """Rebuild every consumed temperature boundary from recorded operands."""
    trazdf = _trazdf()
    rebuilt = trazdf.nemo_rebuild(record)
    nlev = record["header"]["jpkm1"]
    pairs = {
        "zwt_mix": (_vertical_field(record, "zwt_mix", transpose=False),
                    rebuilt["zwt_mix"]),
        "zwi": (_vertical_field(record, "zwi", transpose=False),
                rebuilt["zwi"]),
        "zwd": (_vertical_field(record, "zwd", transpose=False),
                rebuilt["zwd"]),
        "zws": (_vertical_field(record, "zws", transpose=False),
                rebuilt["zws"]),
        "zwt_lu": (_vertical_field(record, "zwt_lu", transpose=False),
                   rebuilt["zwt_lu"]),
        "rhs_T": (_vertical_field(record, "rhs_T", nlev, transpose=False),
                  rebuilt["rhs_T"][..., :nlev]),
        "fwd_T": (_vertical_field(record, "fwd_T", nlev, transpose=False),
                  rebuilt["fwd_T"][..., :nlev]),
        "sol_T": (_vertical_field(
            record, "sol_T_pre_clamp", nlev, transpose=False),
            rebuilt["sol_T"][..., :nlev]),
    }
    return {
        name: int(np.count_nonzero(
            np.asarray(observed).view(np.uint64)
            != np.asarray(expected).view(np.uint64)))
        for name, (observed, expected) in pairs.items()
    }


def _read_vertical_record(path: Path, expect_step: int) -> dict:
    trazdf = _trazdf()
    try:
        return trazdf.read_trazdf_matrix(path, expect_kt=expect_step)
    except trazdf.RecordError as error:
        raise GateError(str(error)) from error


def _vertical_matrix_ulp_control(record: dict) -> dict:
    """Move one consumed diagonal bit and prove calibration catches it."""
    trazdf = _trazdf()
    baseline = trazdf.nemo_rebuild(record)["zwd"]
    planted = dict(record)
    arrays = dict(record["arrays"])
    diagonal = np.array(arrays["zwd"], copy=True)
    planted["arrays"] = arrays
    arrays["zwd"] = diagonal
    isl, jsl = trazdf._interior(record["header"])
    local = diagonal[isl, jsl, :]
    for index in np.ndindex(local.shape):
        old = local[index]
        new = np.nextafter(old, np.inf)
        if np.isfinite(new) and new != old:
            local[index] = new
            observed = _vertical_field(
                planted, "zwd", transpose=False)
            moved = int(np.count_nonzero(
                observed.view(np.uint64) != baseline.view(np.uint64)))
            require(moved == 1,
                    f"vertical matrix ULP plant moved {moved} registered "
                    "coefficients, expected 1")
            return {
                "field": "zwd", "interior_index_ijk": list(index),
                "old_uint64": int(np.asarray(old).view(np.uint64)),
                "new_uint64": int(np.asarray(new).view(np.uint64)),
                "registered_coefficients_moved": moved,
                "consumed_by": "LU diagonal recurrence",
            }
    raise GateError("vertical matrix ULP plant found no movable coefficient")


def _vertical_trajectory_ulp_control(vertical_tbb: np.ndarray,
                                     process_tbb: np.ndarray,
                                     mask: np.ndarray) -> dict:
    """Break one real alignment bit after proving the baseline is exact."""
    require(_different_cells(vertical_tbb, process_tbb, mask) == 0,
            "unplanted vertical/process Tbb fields differ")
    planted = np.array(vertical_tbb, copy=True)
    j, i, k = np.argwhere(mask)[0]
    planted[j, i, k] = np.nextafter(planted[j, i, k], np.inf)
    moved = _different_cells(planted, process_tbb, mask)
    require(moved == 1,
            f"vertical trajectory ULP plant moved {moved} cells, expected 1")
    return {"field": "T_Kbb_in", "cells_unequal": moved,
            "index_jik": [int(j), int(i), int(k)]}


def validate_vertical_record(root: Path, expected_commit: str, *,
                             process_root: Path = DEFAULT_PROCESS_RECORD_ROOT,
                             plant: str | None = None) -> dict:
    """Admit the reused ``tra_zdf`` interval record or fire one plant."""
    root = Path(root)
    process_root = Path(process_root)
    expected_names = [
        f"oracle_trazdf_matrix_kt{step:08d}.bin"
        for step in VERTICAL_RECORD_STEPS
    ]
    records = [root / name for name in expected_names]
    observed = sorted(root.glob("oracle_trazdf_matrix_kt*.bin"))
    require([path.name for path in observed] == expected_names,
            "vertical-record set is not exactly steps 1, 2 and 1081..1440")
    require(VERTICAL_RECORD_BYTES == 6965416,
            f"vertical record layout is {VERTICAL_RECORD_BYTES}, expected "
            "6965416 bytes")
    for path in records:
        require(path.stat().st_size == VERTICAL_RECORD_BYTES,
                f"{path.name}: {path.stat().st_size} bytes, expected "
                f"{VERTICAL_RECORD_BYTES}")

    producer_path = root / "producer_commit.txt"
    require(producer_path.is_file(), f"missing {producer_path.name}")
    producer = producer_path.read_text(encoding="utf-8").strip()
    require(producer == expected_commit,
            "producer_commit.txt differs from --expect-commit")
    manifest = _record_manifest(
        root, records, "vertical_records.sha256",
        "steps 1, 2 and 1081..1440")
    process_names = [
        f"oracle_process_budget_kt{step:08d}.bin"
        for step in range(PROCESS_START_STEP, PROCESS_END_STEP + 1)
    ]
    process_records = [process_root / name for name in process_names]
    observed_process = sorted(process_root.glob(
        "oracle_process_budget_kt*.bin"))
    require([path.name for path in observed_process] == process_names,
            "admitted process-record set is not exactly steps 1081..1440")
    process_producer = (process_root / "producer_commit.txt").read_text(
        encoding="utf-8").strip()
    require(process_producer == PROCESS_RECORD_COMMIT,
            "admitted process record has the wrong producer commit")
    process_manifest = _process_manifest(process_root, process_records)
    _check_process_stamp(process_root, PROCESS_RECORD_COMMIT)
    if plant == "vertical-stamp":
        try:
            _check_record_stamp(root, "0" * 40, "vertical_records.sha256",
                                "vertical_records.stamp")
        except GateError as error:
            return {"status": "PLANT-FIRED", "plant": plant,
                    "reason": str(error)}
        raise GateError("vertical-stamp plant stayed green")
    _check_record_stamp(root, expected_commit, "vertical_records.sha256",
                        "vertical_records.stamp")

    first_interval = records[len(VERTICAL_RECORD_EARLY_STEPS)]
    trazdf = _trazdf()
    if plant == "vertical-truncation":
        try:
            trazdf.read_trazdf_matrix(
                first_interval, expect_kt=PROCESS_START_STEP, truncate=True)
        except trazdf.RecordError as error:
            return {"status": "PLANT-FIRED", "plant": plant,
                    "reason": str(error)}
        raise GateError("vertical-truncation plant stayed green")

    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    mask = gate.expected_masks(build_nemo_testcase_card(CASE))["T"]
    mask2 = np.any(mask, axis=-1)
    require(mask.shape == (PROCESS_JPJ - 4, PROCESS_JPI - 4,
                           PROCESS_JPK - 1),
            f"vertical mask shape {mask.shape} differs from frozen layout")
    first = _read_vertical_record(first_interval, PROCESS_START_STEP)
    if plant == "vertical-matrix-ulp":
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": _vertical_matrix_ulp_control(first)}
    if plant == "vertical-trajectory-ulp":
        process = read_process_record(
            process_root
            / f"oracle_process_budget_kt{PROCESS_START_STEP:08d}.bin")
        tbb = _vertical_field(first, "T_Kbb_in", mask.shape[-1])
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": _vertical_trajectory_ulp_control(
                    tbb, process["Tbb"][..., :mask.shape[-1]], mask)}
    require(plant in (None, "none"),
            f"unknown vertical-record plant {plant!r}")

    alignment = {
        "T_Kbb_in_vs_process_Tbb": 0,
        "T_Krhs_in_vs_process_post_ldf": 0,
        "sol_T_post_clamp_vs_process_Taa": 0,
        "r3t_Kbb_vs_process": 0,
        "r3t_Kmm_vs_process": 0,
        "r3t_Kaa_vs_process": 0,
        "sol_T_post_clamp_vs_next_T_Kbb_in": 0,
    }
    calibration = {
        name: 0 for name in (
            "zwt_mix", "zwi", "zwd", "zws", "zwt_lu",
            "rhs_T", "fwd_T", "sol_T")
    }
    activity = {name: 0 for name in VERTICAL_REQUIRED_FIELDS}
    first_header = None
    last_header = None
    first_digest = None
    last_digest = None
    previous_solution = None
    for position, (step, path) in enumerate(
            zip(VERTICAL_RECORD_STEPS, records, strict=True)):
        record = first if step == PROCESS_START_STEP else \
            _read_vertical_record(path, step)
        require(record["sha256"] == manifest[path.name],
                f"{path.name}: sha256 manifest mismatch")
        require(tuple(record["order"]) == tuple(trazdf.EXPECTED_ARRAYS),
                f"{path.name}: vertical field order differs from the "
                "compiled writer")
        require(all(name in record["arrays"]
                    for name in VERTICAL_REQUIRED_FIELDS),
                f"{path.name}: required vertical field is missing")
        if position == 0:
            first_header = dict(record["header"])
            first_digest = record["sha256"]
        if position == len(records) - 1:
            last_header = dict(record["header"])
            last_digest = record["sha256"]
        if step < PROCESS_START_STEP:
            continue

        per_record = _vertical_calibration(record)
        for name, unequal in per_record.items():
            calibration[name] += unequal
        process_path = process_root / f"oracle_process_budget_kt{step:08d}.bin"
        process = read_process_record(process_path)
        require(process["sha256"] == process_manifest[process_path.name],
                f"{process_path.name}: sha256 manifest mismatch")
        nlev = mask.shape[-1]
        current_tbb = _vertical_field(record, "T_Kbb_in", nlev)
        if previous_solution is not None:
            alignment["sol_T_post_clamp_vs_next_T_Kbb_in"] += \
                _different_cells(previous_solution, current_tbb, mask)
        pairs3 = {
            "T_Kbb_in_vs_process_Tbb": (
                current_tbb,
                process["Tbb"][..., :nlev]),
            "T_Krhs_in_vs_process_post_ldf": (
                _vertical_field(record, "T_Krhs_in", nlev),
                process["rhs_after_lateral_diffusion"][..., :nlev]),
            "sol_T_post_clamp_vs_process_Taa": (
                _vertical_field(record, "sol_T_post_clamp", nlev),
                process["Taa"][..., :nlev]),
        }
        for name, (left, right) in pairs3.items():
            alignment[name] += _different_cells(left, right, mask)
        for name in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa"):
            left = _vertical_field(record, name)
            right = process[name]
            alignment[f"{name}_vs_process"] += int(np.count_nonzero(
                left[mask2].view(np.uint64) != right[mask2].view(np.uint64)))
        for name in VERTICAL_REQUIRED_FIELDS:
            values = _vertical_field(record, name, transpose=False)
            activity[name] += int(np.count_nonzero(values))
        previous_solution = _vertical_field(
            record, "sol_T_post_clamp", nlev)

    require(all(value == 0 for value in alignment.values()),
            "vertical record differs from admitted process trajectory: "
            + ", ".join(f"{name}={value}" for name, value in alignment.items()
                        if value))
    require(all(value == 0 for value in calibration.values()),
            "vertical record fails compiled-arithmetic calibration: "
            + ", ".join(f"{name}={value}" for name, value in calibration.items()
                        if value))
    for name in ("T_Kbb_in", "T_Krhs_in", "avt", "ah_wslp2", "zwt_mix",
                 "zwi", "zwd", "zws", "zwt_lu", "rhs_T", "fwd_T",
                 "sol_T_pre_clamp"):
        require(activity[name] > 0,
                f"vertical field {name} never contains a nonzero value")

    resolved = _resolved_process_card(root / "ocean.output")
    restart_rows = {}
    for name, expected in PROCESS_RESTART_HASHES.items():
        observed_digest = _sha256(root / name)
        require(observed_digest == expected,
                f"passive restart {name} is {observed_digest}, expected "
                f"{expected}")
        require(_sha256(DEFAULT_ADMITTED_YEAR_RUN / name) == expected,
                f"admitted source restart {name} no longer has frozen hash")
        restart_rows[name] = observed_digest
    binary_words = (root / "binary.sha256").read_text(
        encoding="utf-8").split()
    require(binary_words, "binary.sha256 is empty")
    binary_digest = _sha256(root / "nemo")
    require(binary_words[0] == binary_digest,
            "run binary differs from binary.sha256")

    from legoesm.ocean.fidelity.provenance import worktree_stamp
    return {
        "format": "gyre-year-owners-vertical-record-v1",
        "status": "PASS", "case": CASE, "plant": plant,
        "root": str(root), "process_root": str(process_root),
        "producer_commit": producer, "worktree": worktree_stamp(),
        "binary_sha256": binary_digest,
        "layout": {
            "record_steps": list(VERTICAL_RECORD_STEPS),
            "record_count": len(records),
            "bytes_per_record": VERTICAL_RECORD_BYTES,
            "interval_record_count": LEGO_PROCESS_TRACE_STEPS,
            "interval_record_bytes": (
                LEGO_PROCESS_TRACE_STEPS * VERTICAL_RECORD_BYTES),
            "total_record_bytes": sum(path.stat().st_size for path in records),
            "first_header": first_header, "last_header": last_header,
            "records_manifest_sha256": _sha256(
                root / "vertical_records.sha256"),
        },
        "controls": {
            "resolved_card": resolved,
            "process_producer_commit": process_producer,
            "process_manifest_sha256": _sha256(
                process_root / "process_records.sha256"),
            "restart_sha256": restart_rows,
            "process_alignment_cells_unequal": alignment,
            "compiled_calibration_cells_unequal": calibration,
            "field_nonzero_value_visits": activity,
        },
        "first_record_sha256": first_digest,
        "last_record_sha256": last_digest,
    }


# ----------------------------------- independent legoESM production trace ---
def lego_process_temperature_rows(frame: dict) -> dict[str, np.ndarray]:
    """Decode one production-JIT legoESM stage-3 process frame.

    ``Bpre`` is the actual combined content handed to the implicit solve;
    ``Bldf`` is the same components accumulated one-by-one in NEMO's compiled
    order.  Their association difference is retained in ``rounding_closure``
    rather than attributed to vertical diffusion.
    """
    tbb = np.asarray(frame["Tbb"], dtype=np.float64)
    taa = np.asarray(frame["Taa"], dtype=np.float64)
    rows = {
        "geometry": np.asarray(frame["B0"], dtype=np.float64) - tbb,
        "advection": (np.asarray(frame["Badv"], dtype=np.float64)
                      - np.asarray(frame["B0"], dtype=np.float64)),
        "surface_boundary": (np.asarray(frame["Bsbc"], dtype=np.float64)
                             - np.asarray(frame["Badv"], dtype=np.float64)),
        "shortwave": (np.asarray(frame["Bqsr"], dtype=np.float64)
                      - np.asarray(frame["Bsbc"], dtype=np.float64)),
        "lateral_diffusion": (np.asarray(frame["Bldf"], dtype=np.float64)
                              - np.asarray(frame["Bqsr"], dtype=np.float64)),
        "vertical_diffusion": (taa
                               - np.asarray(frame["Bpre"], dtype=np.float64)),
    }
    combined = np.zeros_like(tbb)
    for name in PROCESS_ROWS:
        combined = combined + rows[name]
    rows["rounding_closure"] = (taa - tbb) - combined
    return rows


def _state_bit_mismatches(left, right) -> int:
    """Count unequal bytes across two state pytrees, including signed zero."""
    import jax

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    require(left_tree == right_tree, "production state pytree structures differ")
    mismatches = 0
    for number, (left_leaf, right_leaf) in enumerate(
            zip(left_leaves, right_leaves, strict=True)):
        a = np.asarray(left_leaf)
        b = np.asarray(right_leaf)
        require(a.shape == b.shape and a.dtype == b.dtype,
                f"production state leaf {number} shape/dtype differs")
        mismatches += int(np.count_nonzero(
            a.view(np.uint8).reshape(-1) != b.view(np.uint8).reshape(-1)))
    return mismatches


def _trace_frame(trace) -> dict[str, np.ndarray]:
    b0, badv, bsbc, bqsr, bldf, bpre = trace.boundaries
    return {
        "Tbb": np.asarray(trace.Tbb, dtype=np.float64),
        "q_Kbb": np.asarray(trace.q_Kbb, dtype=np.float64),
        "q_Kmm": np.asarray(trace.q_Kmm, dtype=np.float64),
        "q_Kaa": np.asarray(trace.q_Kaa, dtype=np.float64),
        "B0": np.asarray(b0, dtype=np.float64),
        "Badv": np.asarray(badv, dtype=np.float64),
        "Bsbc": np.asarray(bsbc, dtype=np.float64),
        "Bqsr": np.asarray(bqsr, dtype=np.float64),
        "Bldf": np.asarray(bldf, dtype=np.float64),
        "Bpre": np.asarray(bpre, dtype=np.float64),
        "Taa": np.asarray(trace.Taa, dtype=np.float64),
    }


def _trace_effect_control(control_trace, planted_trace, mask: np.ndarray,
                          index: tuple[int, int, int], delta: float) -> dict:
    control = _trace_frame(control_trace)
    planted = _trace_frame(planted_trace)
    moved = {}
    for name in LEGO_PROCESS_FIELDS:
        active_mask = mask if control[name].ndim == 3 else np.any(mask, axis=-1)
        moved[name] = _different_cells(
            control[name], planted[name], active_mask)
    state_mismatches = _state_bit_mismatches(
        control_trace.state_after, planted_trace.state_after)
    j, i, k = index
    expected = DT_S * delta * control["q_Kmm"][j, i] / control["q_Kaa"][j, i]
    observed = planted["Bsbc"][j, i, k] - control["Bsbc"][j, i, k]
    scale_ulp = abs(float(np.spacing(control["Bsbc"][j, i, k])))
    require(all(moved[name] == 0 for name in
                ("Tbb", "q_Kbb", "q_Kmm", "q_Kaa", "B0", "Badv")),
            "production effect plant moved an upstream process boundary")
    require(all(moved[name] > 0 for name in
                ("Bsbc", "Bqsr", "Bldf", "Bpre", "Taa")),
            "production effect plant failed to reach a downstream boundary")
    require(state_mismatches == 0,
            "write-only effect plant changed the separately compiled state")
    require(abs(observed - expected) <= 8.0 * max(scale_ulp, np.finfo(float).tiny),
            "surface effect plant does not match dt*qmm/qaa*delta")
    return {
        "index_jik": list(index), "rate_delta_K_per_s": delta,
        "expected_Bsbc_delta_K": float(expected),
        "observed_Bsbc_delta_K": float(observed),
        "Bsbc_delta_abs_error_K": float(abs(observed - expected)),
        "moved_cells": moved,
        "carried_state_unequal_bytes": state_mismatches,
        "status": "PLANT-FIRED",
    }


def _write_trace_manifest(root: Path, paths: list[Path], commit: str) -> dict:
    rows = {path.name: _sha256(path) for path in paths}
    manifest = root / "trace_files.sha256"
    manifest.write_text("".join(
        f"{digest}  {name}\n" for name, digest in sorted(rows.items())))
    stamp = root / "trace_files.stamp"
    stamp.write_text(f"{_sha256(manifest)} {commit} {manifest.name}\n")
    return rows


def produce_lego_process_trace(root: Path, expected_commit: str,
                               *, mesh_path: Path = DEFAULT_MESH) -> dict:
    """Run seed zero independently from rest and write steps 1081--1440."""
    _policy()
    from numpy.lib.format import open_memmap

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"process trace requires a clean tree, dirty={stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            f"trace commit {stamp['commit']} != --expect-commit "
            f"{expected_commit}")
    root = Path(root)
    require(not root.exists(),
            f"trace root already exists; refusing overwrite: {root}")
    root.mkdir(parents=True)

    year = _year()
    gate = _gate()
    card = build_nemo_testcase_card(CASE)
    require(card.dt_s == DT_S, f"card dt {card.dt_s} != {DT_S}")
    mesh = year.nemo_operands(mesh_path)
    operands = year.reconcile_operands(card, mesh)
    state = card.recipe.initial_state
    require(float(np.max(np.abs(np.asarray(state.T.data)
                                - np.asarray(card.recipe.initial_state.T.data))))
            == 0.0, "seed-zero initial state was modified")
    masks = gate.expected_masks(card)
    wet = masks["T"]

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(tracer_process_trace=()))
    plant_index = tuple(int(x) for x in np.argwhere(wet)[0])
    plant_delta = float(np.ldexp(1.0, -40))
    plant_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(*plant_index, plant_delta)))

    shape3 = tuple(int(x) for x in np.asarray(state.T.data).shape)
    shape2 = shape3[:2]
    maps = {}
    paths = []
    for name in LEGO_PROCESS_FIELDS:
        shape = ((LEGO_PROCESS_TRACE_STEPS,) + shape2
                 if name.startswith("q_")
                 else (LEGO_PROCESS_TRACE_STEPS,) + shape3)
        path = root / f"{name}.npy"
        paths.append(path)
        maps[name] = open_memmap(path, mode="w+", dtype="<f8", shape=shape)

    snapshot_root = root / "lego_seed0"
    snapshot_root.mkdir()
    started = time.time()
    total_unequal_bytes = 0
    max_step_unequal_bytes = 0
    effect_control = None
    for completed in range(PROCESS_END_STEP):
        kt = completed + 1
        freshwater, surface = gate._surface_forcings(card, state, kt)
        if kt < PROCESS_START_STEP:
            state = ordinary_model.step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
            if kt == PROCESS_START_STEP - 1:
                np.savez(snapshot_root / "day180.npz",
                         **year._snapshot(state, gate))
            continue

        trace = trace_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        reference = ordinary_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        unequal_bytes = _state_bit_mismatches(trace.state_after, reference)
        total_unequal_bytes += unequal_bytes
        max_step_unequal_bytes = max(max_step_unequal_bytes, unequal_bytes)
        require(unequal_bytes == 0,
                f"step {kt}: diagnostic carried state differs from ordinary "
                f"production by {unequal_bytes} bytes")
        if kt == PROCESS_START_STEP:
            planted = plant_model.step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
            effect_control = _trace_effect_control(
                trace, planted, wet, plant_index, plant_delta)
        frame = _trace_frame(trace)
        index = kt - PROCESS_START_STEP
        for name in LEGO_PROCESS_FIELDS:
            require(frame[name].shape == maps[name].shape[1:],
                    f"step {kt} {name}: shape {frame[name].shape}, expected "
                    f"{maps[name].shape[1:]}")
            require(np.all(np.isfinite(frame[name])),
                    f"step {kt} {name}: non-finite trace value")
            maps[name][index] = frame[name]
        state = reference
        if kt % 60 == 0:
            print(f"  lego process trace step {kt:4d}  "
                  f"{time.time() - started:7.1f} s", flush=True)

    np.savez(snapshot_root / "day240.npz", **year._snapshot(state, gate))
    for array in maps.values():
        array.flush()
    del maps
    require(effect_control is not None, "production effect plant never ran")
    metadata = {
        "format": "gyre-legoesm-process-trace-v1", "case": CASE,
        "producer_commit": expected_commit,
        "steps": [PROCESS_START_STEP, PROCESS_END_STEP],
        "record_count": LEGO_PROCESS_TRACE_STEPS,
        "dt_s": DT_S, "seed": 0, "platform": "cpu",
        "precision": "fp64/libm", "production_entry": "model.step/_step_jitted",
        "shape_3d": list(shape3), "shape_2d": list(shape2),
        "fields": list(LEGO_PROCESS_FIELDS),
        "carried_state_unequal_bytes_total": total_unequal_bytes,
        "carried_state_unequal_bytes_max_step": max_step_unequal_bytes,
        "effect_control": effect_control, "operands": operands,
        "wall_seconds": time.time() - started, "worktree": stamp,
    }
    metadata_path = root / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    paths.extend([
        snapshot_root / "day180.npz", snapshot_root / "day240.npz",
        metadata_path,
    ])
    file_hashes = _write_trace_manifest(root, paths, expected_commit)
    report = dict(metadata)
    report["root"] = str(root)
    report["trace_files_sha256"] = file_hashes
    print(f"STATUS PASS: legoESM process trace {LEGO_PROCESS_TRACE_STEPS} "
          f"frames; carried state unequal bytes {total_unequal_bytes}")
    return report


def _trace_file_manifest(root: Path) -> dict[str, str]:
    path = root / "trace_files.sha256"
    require(path.is_file(), f"missing {path.name}")
    rows = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        words = line.split()
        require(len(words) == 2 and re.fullmatch(
            r"[0-9a-f]{64}", words[0]) is not None,
            f"{path.name}:{number}: malformed row")
        name = Path(words[1]).name
        require(name not in rows, f"{path.name}: duplicate {name}")
        rows[name] = words[0]
    return rows


def _check_trace_stamp(root: Path, expected_commit: str) -> None:
    manifest = root / "trace_files.sha256"
    stamp = root / "trace_files.stamp"
    require(stamp.is_file(), f"missing {stamp.name}")
    words = stamp.read_text().split()
    require(len(words) == 3, f"{stamp.name}: malformed stamp")
    require(words[0] == _sha256(manifest),
            f"{stamp.name}: manifest digest mismatch")
    require(words[1] == expected_commit,
            f"{stamp.name}: producer commit mismatch")
    require(words[2] == manifest.name,
            f"{stamp.name}: stamped filename mismatch")


def _load_lego_trace_arrays(root: Path) -> dict[str, np.ndarray]:
    return {name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in LEGO_PROCESS_FIELDS}


def validate_lego_process_trace(root: Path, expected_commit: str,
                                *, plant: str | None = None,
                                mesh_path: Path = DEFAULT_MESH) -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    require(metadata["format"] == "gyre-legoesm-process-trace-v1",
            "wrong legoESM trace format")
    require(metadata["producer_commit"] == expected_commit,
            "legoESM trace manifest producer commit mismatch")
    if plant == "lego-process-stamp":
        try:
            _check_trace_stamp(root, "0" * 40)
        except GateError as error:
            return {"status": "PLANT-FIRED", "plant": plant,
                    "reason": str(error)}
        raise GateError("lego-process-stamp plant stayed green")
    _check_trace_stamp(root, expected_commit)
    manifest = _trace_file_manifest(root)
    expected_files = {f"{name}.npy" for name in LEGO_PROCESS_FIELDS} | {
        "day180.npz", "day240.npz", "manifest.json"}
    require(set(manifest) == expected_files,
            "legoESM trace manifest file set differs from the frozen layout")
    file_paths = {name: (root / name if name not in ("day180.npz", "day240.npz")
                         else root / "lego_seed0" / name)
                  for name in expected_files}
    for name, path in file_paths.items():
        require(path.is_file(), f"missing legoESM trace file {path}")
        require(_sha256(path) == manifest[name],
                f"legoESM trace sha256 mismatch for {name}")

    arrays = _load_lego_trace_arrays(root)
    shape3 = tuple(metadata["shape_3d"])
    shape2 = tuple(metadata["shape_2d"])
    for name, array in arrays.items():
        expected_shape = ((LEGO_PROCESS_TRACE_STEPS,) + shape2
                          if name.startswith("q_") else
                          (LEGO_PROCESS_TRACE_STEPS,) + shape3)
        require(array.shape == expected_shape and array.dtype == np.float64,
                f"{name}: shape/dtype {array.shape}/{array.dtype}, expected "
                f"{expected_shape}/float64")

    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    mask = gate.expected_masks(build_nemo_testcase_card(CASE))["T"]
    require(mask.shape == shape3, "legoESM trace/mask shape mismatch")
    first = {name: np.asarray(array[0]) for name, array in arrays.items()}
    if plant == "lego-process-ulp":
        original = lego_process_temperature_rows(first)
        planted = {name: np.array(value, copy=True)
                   for name, value in first.items()}
        j, i, k = (int(x) for x in np.argwhere(mask)[0])
        planted["Bsbc"][j, i, k] = np.nextafter(
            planted["Bsbc"][j, i, k], np.inf)
        changed = lego_process_temperature_rows(planted)
        moved = {name: _different_cells(original[name], changed[name], mask)
                 for name in (*PROCESS_ROWS, "rounding_closure")}
        require(moved["surface_boundary"] > 0,
                "lego-process-ulp plant did not move its consumed row")
        return {"status": "PLANT-FIRED", "plant": plant,
                "index_jik": [j, i, k], "moved_rows": moved}
    if plant == "lego-process-effect":
        control = metadata["effect_control"]
        require(control["status"] == "PLANT-FIRED"
                and control["moved_cells"]["Bsbc"] > 0
                and control["moved_cells"]["Taa"] > 0
                and control["carried_state_unequal_bytes"] == 0,
                "stored production effect control is vacuous")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": control}
    require(plant in (None, "none"), f"unknown lego process plant {plant!r}")

    activity = {name: 0 for name in PROCESS_ROWS}
    max_step_closure = 0.0
    chained = 0
    surface_subsurface_visits = 0
    previous_taa = None
    for index in range(LEGO_PROCESS_TRACE_STEPS):
        frame = {name: np.asarray(array[index])
                 for name, array in arrays.items()}
        for name, value in frame.items():
            require(np.all(np.isfinite(value)),
                    f"legoESM trace frame {index} {name} is non-finite")
        if previous_taa is not None:
            chained += _different_cells(previous_taa, frame["Tbb"], mask)
        previous_taa = frame["Taa"]
        rows = lego_process_temperature_rows(frame)
        surface_subsurface_visits += int(np.count_nonzero(
            rows["surface_boundary"][..., 1:][mask[..., 1:]]))
        for name in PROCESS_ROWS:
            activity[name] += int(np.count_nonzero(rows[name][mask]))
        max_step_closure = max(
            max_step_closure,
            float(np.max(np.abs(rows["rounding_closure"][mask]))))
    require(chained == 0,
            f"legoESM trace fails Taa-to-next-Tbb chain in {chained} cells")
    expected_surface_visits = int(mask[..., 0].sum()) * LEGO_PROCESS_TRACE_STEPS
    require(surface_subsurface_visits == 0,
            "legoESM surface-boundary bucket moved "
            f"{surface_subsurface_visits} subsurface wet cells; a non-surface "
            "process was misclassified")
    require(activity["surface_boundary"] == expected_surface_visits,
            "legoESM surface-boundary activity is "
            f"{activity['surface_boundary']}, expected exactly the "
            f"{expected_surface_visits} wet top-cell visits")
    for name in PROCESS_ROWS[1:]:
        require(activity[name] > 0,
                f"legoESM active process row {name} never moves a wet cell")
    require(metadata["carried_state_unequal_bytes_total"] == 0
            and metadata["carried_state_unequal_bytes_max_step"] == 0,
            "legoESM diagnostic/reference carried-state comparison failed")
    return {
        "format": "gyre-legoesm-process-trace-validation-v1",
        "status": "PASS", "case": CASE, "root": str(root),
        "producer_commit": expected_commit, "worktree": worktree_stamp(),
        "layout": {"record_count": LEGO_PROCESS_TRACE_STEPS,
                   "shape_3d": list(shape3), "shape_2d": list(shape2),
                   "fields": list(LEGO_PROCESS_FIELDS)},
        "controls": {
            "chained_cells_unequal": chained,
            "active_row_nonzero_cell_visits": activity,
            "surface_subsurface_nonzero_cell_visits": (
                surface_subsurface_visits),
            "expected_surface_top_cell_visits": expected_surface_visits,
            "max_abs_step_rounding_closure_K": max_step_closure,
            "carried_state_unequal_bytes_total": 0,
            "effect_control": metadata["effect_control"],
        },
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing snapshot {path}")
    with np.load(path) as handle:
        return {name: np.asarray(handle[name], dtype=np.float64)
                for name in handle.files}


def _projection(component: np.ndarray, endpoint: np.ndarray,
                mask: np.ndarray, endpoint_rms: float) -> tuple[float, float]:
    denominator = float(np.sum(endpoint[mask] * endpoint[mask]))
    require(denominator > 0.0, "zero day-240 endpoint error")
    fraction = float(np.sum(component[mask] * endpoint[mask])) / denominator
    return fraction * endpoint_rms, fraction


def score_process_budget(nemo_process_root: Path, lego_trace_root: Path,
                         expected_commit: str, *,
                         immutable_lego_root: Path = DEFAULT_IMMUTABLE_GYRE_YEAR,
                         nemo_root: Path = YEAR_ROOT,
                         mesh_path: Path = DEFAULT_MESH) -> dict:
    """Close and rank the independent day-180-to-240 process budget."""
    _policy()
    year = _year()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    nemo_validation = validate_process_record(
        nemo_process_root, "af3f7215060fc17c71adc6794817c710df8ee471")
    lego_validation = validate_lego_process_trace(
        lego_trace_root, expected_commit, mesh_path=mesh_path)
    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _dz, _dy, _area, bands = year._geometry(card, mesh_path)
    nlev = wet3.shape[-1]
    lat = np.asarray(mesh["gphit"], dtype=np.float64)

    generated180 = _load_npz(Path(lego_trace_root) / "lego_seed0/day180.npz")
    generated240 = _load_npz(Path(lego_trace_root) / "lego_seed0/day240.npz")
    immutable180 = _load_npz(
        Path(immutable_lego_root) / "lego_seed0_year/day180.npz")
    immutable240 = _load_npz(
        Path(immutable_lego_root) / "lego_seed0_year/day240.npz")
    immutable_mismatches = {}
    for day, generated, immutable in (
            (180, generated180, immutable180),
            (240, generated240, immutable240)):
        immutable_mismatches[str(day)] = {}
        for name in FIELDS:
            a = np.asarray(generated[name], dtype=np.float64)
            b = np.asarray(immutable[name], dtype=np.float64)
            require(a.shape == b.shape, f"day {day} {name}: shape mismatch")
            unequal = int(np.count_nonzero(
                a.view(np.uint64) != b.view(np.uint64)))
            immutable_mismatches[str(day)][name] = unequal
            require(unequal == 0,
                    f"day {day} {name}: generated trace differs from immutable "
                    f"year member in {unequal} cells")

    nemo180 = year._load_nemo(nemo_root, 0, 180, nlev)
    nemo240 = year._load_nemo(nemo_root, 0, 240, nlev)
    lego_start = np.asarray(generated180["T"], dtype=np.float64)
    lego_end = np.asarray(generated240["T"], dtype=np.float64)
    nemo_start = np.asarray(nemo180["T"], dtype=np.float64)
    nemo_end = np.asarray(nemo240["T"], dtype=np.float64)
    endpoint = lego_end - nemo_end
    endpoint_rms = _rms(endpoint, wet3)
    require(endpoint_rms == 1.6446741930292448e-2,
            f"day-240 headline {endpoint_rms:.17e} differs from frozen "
            "1.6446741930292448e-2 K")
    immutable030 = _load_npz(
        Path(immutable_lego_root) / "lego_seed0_year/day030.npz")
    nemo030 = year._load_nemo(nemo_root, 0, 30, nlev)
    day30_rms = _rms(np.asarray(immutable030["T"]) - nemo030["T"], wet3)

    lego_arrays = _load_lego_trace_arrays(Path(lego_trace_root))
    components_lego = {name: np.zeros_like(lego_start) for name in PROCESS_ROWS}
    components_nemo = {name: np.zeros_like(nemo_start) for name in PROCESS_ROWS}
    block_lego = [{name: np.zeros_like(lego_start) for name in PROCESS_ROWS}
                  for _ in range(6)]
    block_nemo = [{name: np.zeros_like(nemo_start) for name in PROCESS_ROWS}
                  for _ in range(6)]
    block_start_lego = []
    block_end_lego = []
    block_start_nemo = []
    block_end_nemo = []
    current_block = -1
    for index, step in enumerate(range(PROCESS_START_STEP,
                                       PROCESS_END_STEP + 1)):
        block = index // 60
        if block != current_block:
            current_block = block
            block_start_lego.append(np.asarray(lego_arrays["Tbb"][index]).copy())
        lego_frame = {name: np.asarray(array[index])
                      for name, array in lego_arrays.items()}
        nemo_frame = read_process_record(
            Path(nemo_process_root)
            / f"oracle_process_budget_kt{step:08d}.bin")
        lego_rows = lego_process_temperature_rows(lego_frame)
        nemo_rows = process_temperature_rows(nemo_frame)
        if index % 60 == 0:
            block_start_nemo.append(
                np.asarray(nemo_frame["Tbb"][..., :nlev]).copy())
        for name in PROCESS_ROWS:
            components_lego[name] += lego_rows[name]
            components_nemo[name] += nemo_rows[name]
            block_lego[block][name] += lego_rows[name]
            block_nemo[block][name] += nemo_rows[name]
        if index % 60 == 59:
            block_end_lego.append(np.asarray(lego_frame["Taa"]).copy())
            block_end_nemo.append(
                np.asarray(nemo_frame["Taa"][..., :nlev]).copy())

    require(_different_cells(np.asarray(lego_arrays["Tbb"][0]),
                             lego_start, wet3) == 0,
            "legoESM first trace Tbb differs from day-180 snapshot")
    require(_different_cells(np.asarray(lego_arrays["Taa"][-1]),
                             lego_end, wet3) == 0,
            "legoESM final trace Taa differs from day-240 snapshot")
    first_nemo = read_process_record(
        Path(nemo_process_root)
        / f"oracle_process_budget_kt{PROCESS_START_STEP:08d}.bin")
    last_nemo = read_process_record(
        Path(nemo_process_root)
        / f"oracle_process_budget_kt{PROCESS_END_STEP:08d}.bin")
    require(_different_cells(first_nemo["Tbb"][..., :nlev], nemo_start,
                             wet3) == 0,
            "NEMO first trace Tbb differs from day-180 restart")
    require(_different_cells(last_nemo["Taa"][..., :nlev], nemo_end,
                             wet3) == 0,
            "NEMO final trace Taa differs from day-240 restart")

    def sum_rows(rows):
        total = np.zeros_like(lego_start)
        for name in PROCESS_ROWS:
            total = total + rows[name]
        return total

    closure_lego = (lego_end - lego_start) - sum_rows(components_lego)
    closure_nemo = (nemo_end - nemo_start) - sum_rows(components_nemo)
    component = {"incoming": lego_start - nemo_start}
    for name in PROCESS_ROWS:
        component[name] = components_lego[name] - components_nemo[name]
    component["rounding_closure"] = closure_lego - closure_nemo
    reconstruction = np.zeros_like(endpoint)
    for values in component.values():
        reconstruction = reconstruction + values
    reconstruction_residual = endpoint - reconstruction
    max_reconstruction = float(np.max(np.abs(reconstruction_residual[wet3])))

    block_components = []
    for block in range(6):
        item = {name: block_lego[block][name] - block_nemo[block][name]
                for name in PROCESS_ROWS}
        local_lego_closure = ((block_end_lego[block] - block_start_lego[block])
                              - sum_rows(block_lego[block]))
        local_nemo_closure = ((block_end_nemo[block] - block_start_nemo[block])
                              - sum_rows(block_nemo[block]))
        item["rounding_closure"] = local_lego_closure - local_nemo_closure
        block_components.append(item)

    regions = _regions(lat, wet2)
    horizontal_masks = {name: wet3 & regions[name][..., None]
                        for name in ("west_third", "interior_third",
                                     "east_third")}
    latitude_masks = {
        f"south_le_{EMP_SPLIT_LAT_DEG}N": wet3 & (
            lat <= EMP_SPLIT_LAT_DEG)[..., None],
        f"north_gt_{EMP_SPLIT_LAT_DEG}N": wet3 & (
            lat > EMP_SPLIT_LAT_DEG)[..., None],
    }

    def strongest_partition(values, partitions):
        scored = []
        for name, local_mask in partitions.items():
            carry, fraction = _projection(values, endpoint, local_mask,
                                          _rms(endpoint, local_mask))
            # The local carry above has a local normalization.  Ranking the
            # location instead uses its additive contribution to the GLOBAL
            # endpoint dot product, reported beside the native local metric.
            additive = float(np.sum(values[local_mask] * endpoint[local_mask]))
            scored.append((abs(additive), name, carry, fraction,
                           int(local_mask.sum())))
        _, name, carry, fraction, cells = max(scored)
        return {"name": name, "local_signed_carry_K": carry,
                "local_projection_fraction": fraction, "cells": cells}

    ranking = []
    for name, values in component.items():
        carry, fraction = _projection(values, endpoint, wet3, endpoint_rms)
        if name == "incoming":
            birth = {"interval_days": "before day 180", "signed_carry_K": carry}
        else:
            block_rows = []
            for block, block_values in enumerate(block_components):
                block_value = block_values[name]
                block_carry, _ = _projection(
                    block_value, endpoint, wet3, endpoint_rms)
                block_rows.append((abs(block_carry), block, block_carry))
            _, block, block_carry = max(block_rows)
            birth = {"interval_days": [180 + 10 * block,
                                        180 + 10 * (block + 1)],
                     "signed_carry_K": block_carry}
        depth_location = strongest_partition(values, bands)
        longitude_location = strongest_partition(values, horizontal_masks)
        latitude_location = strongest_partition(values, latitude_masks)
        wind_mask = wet3 & regions["wind_band_15_29N"][..., None]
        wind_dot = float(np.sum(values[wind_mask] * endpoint[wind_mask]))
        ranking.append({
            "owner": name, "signed_carry_K": carry,
            "abs_signed_carry_K": abs(carry),
            "projection_fraction": fraction,
            "component_rms_K": _rms(values, wet3),
            "day30_carry_K": None,
            "day30_carry_status": "UNAVAILABLE-BY-RECORD",
            "birth": birth, "largest_depth_partition": depth_location,
            "largest_longitude_partition": longitude_location,
            "largest_latitude_partition": latitude_location,
            "wind_band_additive_dot_K2": wind_dot,
        })
    ranking.sort(key=lambda row: row["abs_signed_carry_K"], reverse=True)
    signed_sum = float(sum(row["signed_carry_K"] for row in ranking))
    sum_abs = float(sum(row["abs_signed_carry_K"] for row in ranking))
    report = {
        "format": "gyre-day240-process-budget-v1", "case": CASE,
        "status": "PASS", "interval_days": [180, 240],
        "nemo_process_root": str(nemo_process_root),
        "lego_trace_root": str(lego_trace_root),
        "immutable_lego_root": str(immutable_lego_root),
        "nemo_root": str(nemo_root),
        "headline": {"day240_T3D_rms_K": endpoint_rms,
                     "day30_T3D_rms_K": day30_rms},
        "endpoint_controls": {
            "immutable_cells_unequal": immutable_mismatches,
            "max_abs_reconstruction_residual_K": max_reconstruction,
            "lego_interval_closure_rms_K": _rms(closure_lego, wet3),
            "nemo_interval_closure_rms_K": _rms(closure_nemo, wet3),
            "signed_carry_sum_K": signed_sum,
            "signed_carry_minus_endpoint_rms_K": signed_sum - endpoint_rms,
            "sum_abs_signed_carry_K": sum_abs,
            "cancellation_ratio": sum_abs / endpoint_rms,
        },
        "ranking": ranking, "largest_owner": ranking[0]["owner"],
        "nemo_validation": nemo_validation,
        "lego_validation": lego_validation,
        "worktree": worktree_stamp(),
    }
    print("\nDAY-240 PROCESS MAGNITUDE RANKING -- independent trajectories")
    print(f"  {'rank':>4s} {'owner':>22s} {'signed carry K':>16s} "
          f"{'component rms K':>16s} {'birth days':>14s}")
    for rank, row in enumerate(ranking, 1):
        print(f"  {rank:4d} {row['owner']:>22s} "
              f"{row['signed_carry_K']:16.8e} "
              f"{row['component_rms_K']:16.8e} "
              f"{str(row['birth']['interval_days']):>14s}")
    print(f"  day-240 T rms {endpoint_rms:.17e} K; signed carry sum "
          f"{signed_sum:.17e} K; max reconstruction residual "
          f"{max_reconstruction:.3e} K")
    return report


def _process_math_self_check(failures: list[str]) -> None:
    shape = (2, 2, 4)
    record = {
        "Tbb": np.full(shape, 2.0), "Taa": np.full(shape, 2.75),
        "r3t_Kbb": np.zeros(shape[:2]),
        "r3t_Kmm": np.zeros(shape[:2]),
        "r3t_Kaa": np.zeros(shape[:2]), "rDt": 1.0,
        "rhs_after_advection": np.full(shape, 0.10),
        "rhs_after_surface_boundary": np.full(shape, 0.20),
        "rhs_after_shortwave": np.full(shape, 0.30),
        "rhs_after_lateral_diffusion": np.full(shape, 0.40),
    }
    rows = process_temperature_rows(record)
    reconstructed = np.zeros(shape)
    reconstructed = reconstructed[..., :-1]
    for name in (*PROCESS_ROWS, "rounding_closure"):
        reconstructed += rows[name]
    if not np.array_equal(
            reconstructed,
            (record["Taa"] - record["Tbb"])[..., :-1]):
        failures.append("process rows do not reconstruct a synthetic endpoint")
    control = _process_sbc_ulp_control(
        record, np.ones(shape[:-1] + (shape[-1] - 1,), dtype=bool))
    if not control["raw_surface_rhs_increment_moved"]:
        failures.append("process surface-boundary ULP control is inert")
    decoded_ulp_moves = sum(
        control["decoded_temperature_rows_moved"].values())
    if decoded_ulp_moves != 0:
        failures.append("synthetic raw RHS ULP unexpectedly reached a decoded "
                        "temperature row")
    propagated = _process_sbc_effect_control(
        record, np.ones(shape[:-1] + (shape[-1] - 1,), dtype=bool))
    if not propagated["decoded_temperature_rows_moved"]["surface_boundary"]:
        failures.append("process surface-boundary effect control is inert")
    print(f"  process layout {PROCESS_RECORD_BYTES} bytes and synthetic "
          "endpoint closure -- OK")
    print("  raw RHS ULP moves its decoded RHS increment but 0 temperature "
          "rows; effect-scale plant moves the temperature budget -- OK")
    lego_frame = {
        "Tbb": np.full(shape[:-1] + (shape[-1] - 1,), 2.0),
        "B0": np.full(shape[:-1] + (shape[-1] - 1,), 2.1),
        "Badv": np.full(shape[:-1] + (shape[-1] - 1,), 2.2),
        "Bsbc": np.full(shape[:-1] + (shape[-1] - 1,), 2.3),
        "Bqsr": np.full(shape[:-1] + (shape[-1] - 1,), 2.4),
        "Bldf": np.full(shape[:-1] + (shape[-1] - 1,), 2.5),
        "Bpre": np.full(shape[:-1] + (shape[-1] - 1,), 2.55),
        "Taa": np.full(shape[:-1] + (shape[-1] - 1,), 2.75),
    }
    lego_rows = lego_process_temperature_rows(lego_frame)
    lego_sum = np.zeros_like(lego_frame["Tbb"])
    for name in (*PROCESS_ROWS, "rounding_closure"):
        lego_sum += lego_rows[name]
    if not np.array_equal(lego_sum,
                          lego_frame["Taa"] - lego_frame["Tbb"]):
        failures.append("legoESM process rows do not close a synthetic endpoint")
    planted_lego = {name: np.array(value, copy=True)
                    for name, value in lego_frame.items()}
    planted_lego["Bsbc"][0, 0, 0] = np.nextafter(
        planted_lego["Bsbc"][0, 0, 0], np.inf)
    planted_rows = lego_process_temperature_rows(planted_lego)
    synthetic_mask = np.ones_like(lego_frame["Tbb"], dtype=bool)
    if _different_cells(lego_rows["surface_boundary"],
                        planted_rows["surface_boundary"], synthetic_mask) == 0:
        failures.append("legoESM process-boundary ULP plant is inert")
    else:
        print("  legoESM boundary rows close and their ULP plant fires -- OK")


# ------------------------------------------------------- the per-step walk ---
def step_gap(steps: int, out: Path, *, entry_root: Path = DEFAULT_ENTRY_ROOT,
             plant: str | None = None) -> dict:
    """legoESM step by step against NEMO's per-step ENTRY dumps.

    NEMO's ``oracle_step_entry_kt{n}.bin`` holds ``ts/uu/vv/ssh(...,Nbb)`` at
    the ENTRY of step ``n``, i.e. the state after ``n-1`` completed steps.  So
    the dump at ``kt = n`` is compared against legoESM after ``n-1`` calls to
    ``model.step``.  That is the same convention the certified ladder uses.

    ponytail: this walks the card with the certified gate's OWN
    ``_surface_forcings`` + ``LatLonCGridOceanModel.step`` -- the same two
    calls ``run_member`` and ``--census`` make -- rather than adding a third
    copy of the physics.  What is new here is only the comparison cadence.
    """
    _policy()
    import jax.numpy as jnp  # noqa: F401  (policy must be set before jax use)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    gate = _gate()
    card = build_nemo_testcase_card(CASE)
    masks = gate.expected_masks(card)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    state = card.recipe.initial_state
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rest = gate.read_entry(
        Path(entry_root) / "oracle_step_entry_kt00000001.bin")
    rows = []
    started = time.time()
    for completed in range(steps + 1):
        kt = completed + 1                      # the dump this state matches
        path = Path(entry_root) / f"oracle_step_entry_kt{kt:08d}.bin"
        if path.is_file():
            read_kt = kt + 1 if plant == "day-offset" else kt
            read = Path(entry_root) / f"oracle_step_entry_kt{read_kt:08d}.bin"
            require(read.is_file(),
                    f"plant day-offset: no dump at kt={read_kt}")
            oracle = gate.read_entry(read)
            fields = gate.lego_fields(state)
            row = {"kt": kt, "steps_completed": completed,
                   "day": completed / STEPS_PER_DAY,
                   "oracle": str(read), "oracle_sha256": gate.sha256(read)}
            for name in FIELDS:
                mask = masks[name]
                left = np.asarray(fields[name], dtype=np.float64)
                # The oracle's binary record carries NEMO's jpk = 31 levels;
                # the card and its masks carry the 30 the model integrates.
                # Crop the ORACLE, never pad the candidate.
                right = np.asarray(oracle[name], dtype=np.float64)
                if right.ndim == 3:
                    right = right[..., :left.shape[-1]]
                require(left.shape == right.shape == mask.shape,
                        f"kt={kt} {name}: shapes {left.shape} {right.shape} "
                        f"{mask.shape}")
                row[f"rms_{name}"] = _rms(left - right, mask)
                row[f"max_{name}"] = float(np.max(np.abs((left - right)[mask])))
                row[f"unequal_{name}"] = int(np.count_nonzero(
                    left[mask].view(np.uint64) != right[mask].view(np.uint64)))
                # The dimensionless version: the gap over the signal NEMO
                # ITSELF has developed from rest by this step.  Five fields in
                # five units cannot be ranked any other way.
                base = np.asarray(rest[name], dtype=np.float64)
                if base.ndim == 3:
                    base = base[..., :left.shape[-1]]
                signal = _rms(right - base, mask)
                row[f"frac_{name}"] = (row[f"rms_{name}"] / signal
                                       if signal > 0.0 else float("nan"))
            rows.append(row)
        if completed == steps:
            break
        freshwater, surface = gate._surface_forcings(card, state, completed + 1)
        state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
        require(all(bool(np.all(np.isfinite(np.asarray(value))))
                    for value in gate.lego_fields(state).values()),
                f"non-finite legoESM state after step {completed + 1}")

    report = {"format": "gyre-year-owners-step-gap-v1", "case": CASE,
              "steps": steps, "entry_root": str(entry_root), "plant": plant,
              "rows": rows, "wall_seconds": time.time() - started,
              "worktree": worktree_stamp()}
    (out / "step_gap.json").write_text(json.dumps(report, indent=2))
    print(f"{'kt':>5s}{'day':>7s}" + "".join(f"{f'rms {n}':>14s}" for n in FIELDS)
          + "".join(f"{f'frac {n}':>12s}" for n in FIELDS))
    for row in rows:
        print(f"{row['kt']:>5d}{row['day']:>7.3f}"
              + "".join(f"{row[f'rms_{n}']:>14.4e}" for n in FIELDS)
              + "".join(f"{row[f'frac_{n}']:>12.4e}" for n in FIELDS))
    return report


# ------------------------------------------------------- the decomposition ---
def _regions(lat, wet2):
    """The REGION cut.  Two of the three boundaries are read off usrdef_sbc."""
    columns = np.flatnonzero(np.any(wet2, axis=0))
    require(columns.size > 0, "no wet column")
    lo, hi = int(columns.min()), int(columns.max())
    third = (hi - lo + 1) / 3.0
    index = np.arange(wet2.shape[1])[None, :] * np.ones_like(wet2, dtype=int)
    west = wet2 & (index < lo + third)
    east = wet2 & (index >= lo + 2 * third)
    middle = wet2 & ~west & ~east
    return {
        "west_third": west, "interior_third": middle, "east_third": east,
        # usrdef_sbc.f90:140 -- the emp branch splits at 37.2 N.
        f"emp_south_le_{EMP_SPLIT_LAT_DEG}N": wet2 & (lat <= EMP_SPLIT_LAT_DEG),
        f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N": wet2 & (lat > EMP_SPLIT_LAT_DEG),
        # usrdef_sbc.f90:192-193 -- the wind's half period runs 15 N to 29 N.
        "wind_band_15_29N": wet2 & (lat >= WIND_BAND_LAT_DEG[0])
                                 & (lat <= WIND_BAND_LAT_DEG[1]),
    }


def decompose(day: int, *, lego_root: Path = YEAR_HEAD,
              nemo_root: Path = YEAR_ROOT, seed: int = 0,
              mesh_path: Path = DEFAULT_MESH) -> dict:
    """The gap at ``day`` by FIELD, by DEPTH and by REGION.

    Each cut's blind spot is in the preregistration's section 3 and is
    reported next to its table rather than left to the reader.
    """
    _policy()
    year = _year()
    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, dz, dy, area, bands = year._geometry(card, mesh_path)
    gate_masks = gate.expected_masks(card)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0,
                       dtype=np.float64)[..., :wet3.shape[-1]]
    nlev = wet3.shape[-1]

    lego = year._load_lego(lego_root, seed, day)
    nemo = year._load_nemo(nemo_root, seed, day, nlev)
    # The REST state, so every field's gap can be expressed as a fraction of
    # the signal NEMO itself has developed.  Read from the card, which the
    # year harness's A1 leg proves is bit-identical to NEMO's step-1 BEFORE
    # level on every field.
    rest = gate.lego_fields(card.recipe.initial_state)

    masks = {"T": wet3, "S": wet3, "ssh": wet2,
             "u": gate_masks["u"], "v": gate_masks["v"]}
    field_rows = {}
    for name in FIELDS:
        mask = masks[name]
        left = np.asarray(lego[name], dtype=np.float64)
        right = np.asarray(nemo[name], dtype=np.float64)
        base = np.asarray(rest[name], dtype=np.float64)
        require(left.shape == right.shape == mask.shape == base.shape,
                f"{name}: shapes {left.shape} {right.shape} {mask.shape}")
        signal = _rms(right - base, mask)
        field_rows[name] = {
            "gap_rms": _rms(left - right, mask),
            "nemo_from_rest_rms": signal,
            "fraction_of_nemo_signal": (_rms(left - right, mask) / signal
                                        if signal > 0.0 else float("nan")),
            "nemo_std": float(np.std(right[mask])),
            "wet_cells": int(mask.sum()),
        }
    leading = max(
        (name for name in FIELDS
         if np.isfinite(field_rows[name]["fraction_of_nemo_signal"])),
        key=lambda name: field_rows[name]["fraction_of_nemo_signal"])

    difference = np.asarray(lego["T"], dtype=np.float64) - np.asarray(
        nemo["T"], dtype=np.float64)
    total = float(np.sum(difference[wet3] ** 2))
    depth_rows = {}
    for name, mask in list(bands.items()) + [("top_cell", wet3 & (
            np.arange(nlev)[None, None, :] == 0))]:
        depth_rows[name] = {
            "gap_rms_K": _rms(difference, mask),
            "share_of_sum_dT2": float(np.sum(difference[mask] ** 2)) / total,
            "cells": int(mask.sum()),
            "depth_range_m": [float(depth[mask].min()),
                              float(depth[mask].max())],
        }
    region_rows = {}
    for name, mask2 in _regions(lat, wet2).items():
        mask = wet3 & mask2[..., None]
        region_rows[name] = {
            "gap_rms_K": _rms(difference, mask),
            "share_of_sum_dT2": float(np.sum(difference[mask] ** 2)) / total,
            "cells": int(mask.sum()),
            "share_of_cells": float(mask.sum()) / float(wet3.sum()),
        }
    peak = np.unravel_index(int(np.argmax(np.abs(np.where(wet3, difference,
                                                          0.0)))),
                            difference.shape)
    report = {
        "format": "gyre-year-owners-decompose-v1", "case": CASE, "day": day,
        "seed": seed, "lego_root": str(lego_root), "nemo_root": str(nemo_root),
        "nemo_restart": nemo["path"], "nemo_sha256": nemo["sha256"],
        "mesh_sha256": mesh["mesh_sha256"],
        "fields": field_rows, "leading_field": leading,
        "depth": depth_rows, "region": region_rows,
        "peak": {"j": int(peak[0]), "i": int(peak[1]), "k": int(peak[2]),
                 "dT_K": float(difference[peak]),
                 "depth_m": float(depth[peak]),
                 "lat_deg": float(lat[peak[0], peak[1]])},
        "worktree": worktree_stamp(),
    }
    print(f"\nDAY {day}: FIELD -- gap, NEMO's own from-rest signal, and the ratio")
    print(f"  {'field':>6s}{'gap rms':>14s}{'NEMO signal':>14s}"
          f"{'gap/signal':>14s}{'gap/std':>12s}{'cells':>8s}")
    for name in FIELDS:
        row = field_rows[name]
        ratio = (row["gap_rms"] / row["nemo_std"] if row["nemo_std"] > 0
                 else float("nan"))
        print(f"  {name:>6s}{row['gap_rms']:>14.4e}"
              f"{row['nemo_from_rest_rms']:>14.4e}"
              f"{row['fraction_of_nemo_signal']:>14.4e}{ratio:>12.4e}"
              f"{row['wet_cells']:>8d}")
    print(f"  LEADING FIELD (dimensionless): {leading}")
    print(f"\nDAY {day}: DEPTH -- temperature only")
    print(f"  {'band':>12s}{'gap rms [K]':>14s}{'share dT^2':>12s}{'cells':>8s}")
    for name, row in depth_rows.items():
        print(f"  {name:>12s}{row['gap_rms_K']:>14.4e}"
              f"{row['share_of_sum_dT2']:>12.4f}{row['cells']:>8d}")
    print(f"\nDAY {day}: REGION -- temperature only")
    print(f"  {'region':>24s}{'gap rms [K]':>14s}{'share dT^2':>12s}"
          f"{'share cells':>13s}")
    for name, row in region_rows.items():
        print(f"  {name:>24s}{row['gap_rms_K']:>14.4e}"
              f"{row['share_of_sum_dT2']:>12.4f}{row['share_of_cells']:>13.4f}")
    print(f"\n  peak |dT| {report['peak']['dT_K']:+.4e} K at "
          f"j={report['peak']['j']} i={report['peak']['i']} "
          f"k={report['peak']['k']} ({report['peak']['depth_m']:.1f} m, "
          f"{report['peak']['lat_deg']:.2f} N)")
    return report


# ------------------------------------------- the surface-forcing STATEMENT ---
def _nyear_from_restart(path: Path) -> tuple[int, int, float, int]:
    """``ndastp``/``kt``/``adatrj`` read off NEMO's OWN restart.

    ``nyear`` is an OPERAND of the forcing (``usrdef_sbc.f90:108``) whose value
    cannot be read off the source.  Rule 0 says read it; this reads it from the
    run rather than assuming the run is in year 1.
    """
    import netCDF4

    with netCDF4.Dataset(path) as handle:
        ndastp = int(np.asarray(handle.variables["ndastp"][...]))
        kt = int(np.asarray(handle.variables["kt"][...]))
        adatrj = float(np.asarray(handle.variables["adatrj"][...]))
    return ndastp // 10000, kt, adatrj, ndastp


def forcing_gate(*, nemo_root: Path = YEAR_ROOT, seed: int = 0,
                 mesh_path: Path = DEFAULT_MESH,
                 entry_root: Path = DEFAULT_ENTRY_ROOT,
                 days: tuple[int, ...] = tuple(range(30, 361, 30)),
                 plant: str | None = None) -> dict:
    """legoESM's CURRENT forcing vs the LITERAL usrdef_sbc, on NEMO'S OWN state.

    WHAT THIS ANSWERS.  The kt=1 oracle dump certifies all five forcing fields
    BIT-EXACT -- at ONE value of a clock with 2160 distinct values in the year.
    Every seasonal statement in ``usrdef_sbc`` is a function of that scalar, so
    a wrong phase, denominator or calendar operand is invisible there and
    finite on day 30.  Evaluating BOTH transcriptions on the SAME state
    separates the STATEMENT from the state: any difference here is a
    transcription defect, because the inputs are identical by construction.

    WHAT IT CANNOT SEE, stated rather than discovered later: a statement
    misread the SAME way by both arms.  The mitigation is that the literal arm
    is certified against NEMO's own kt=1 dump (round 8 / round 16) and that the
    receipt re-reads the compiled routine line by line.
    """
    _policy()
    import jax.numpy as jnp
    year = _year()
    gate = _gate()
    r16 = _round16()
    from legoesm.ocean.eos import (
        nemo_potential_temperature_from_conservative)
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_qns
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card, gyre_surface_boundary_condition)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _dz, _dy, _area, _bands = year._geometry(card, mesh_path)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    nlev = wet3.shape[-1]

    template = card.recipe.initial_state
    cos_u = np.asarray(card.recipe.grid.cos_alpha_u, dtype=np.float64)[:, 1:]
    sin_u = np.asarray(card.recipe.grid.sin_alpha_u, dtype=np.float64)[:, 1:]

    def current(surface_ct, surface_sa, kt):
        """THE PRODUCTION PATH, called -- not a second copy of its body.

        A first version of this gate re-assembled the certified gate's own
        composition here instead of CALLING it.  An independent review proved
        the consequence by construction: poisoning ``_surface_forcings`` to
        raise left the gate reporting ALL BIT-EXACT with zero calls, and
        TRANSPOSING ``tau_x``/``tau_y`` inside the real ``_surface_forcings``
        left all five fields bit-exact at all thirteen clock samples while the
        per-step walk's kt=2 velocity residual went from `2.2e-13` to `2.2e-02`
        m/s.  The gate certified the usrdef_sbc STATEMENTS and said nothing
        about the forcing the MODEL CONSUMES.

        So this calls ``_surface_forcings`` on a state carrying the supplied
        surface tracers, and returns the MODEL-FACING quantities:

          qsr   <- surface.sw_down          (what radiation receives)
          q_net <- surface.q_net            (NEMO carries qns and qsr apart;
                                             the shared object carries a sum,
                                             so the SUM is the model-facing row)
          emp   <- freshwater.evap          (what the salt budget receives)
          utau  <- surface.tau_i_native     (what the momentum stage receives)
          vtau  <- surface.tau_j_native

        plus ``utau_roundtrip``/``vtau_roundtrip``, recovered by INVERTING the
        geographic rotation the gate applies to ``tau_x``/``tau_y``.  A
        transposed or mis-signed rotation moves those two and nothing else,
        which is exactly the defect the review planted.
        """
        ct = jnp.asarray(surface_ct, dtype=jnp.float64)
        sa = jnp.asarray(surface_sa, dtype=jnp.float64)
        nlev = int(np.asarray(template.T.data).shape[-1])
        broadcast = lambda v: jnp.broadcast_to(  # noqa: E731
            jnp.asarray(v, dtype=jnp.float64)[..., None],
            np.asarray(template.T.data).shape)
        state = template._replace(
            T=template.T.replace(data=broadcast(ct)),
            S=template.S.replace(data=broadcast(sa)))
        require(nlev >= 1, "the card has no vertical levels")
        freshwater, surface = gate._surface_forcings(card, state, kt)
        sw = np.asarray(surface.sw_down, dtype=np.float64)
        q_total = np.asarray(surface.q_net, dtype=np.float64)
        tau_x = np.asarray(surface.tau_x, dtype=np.float64)
        tau_y = np.asarray(surface.tau_y, dtype=np.float64)
        if plant == "forcing-stress-transpose":
            tau_x, tau_y = tau_y, tau_x
        # phase3_gate.py:682-689 builds tau_x/tau_y from the native pair as
        #   tau_x = -(cos*utau - sin*vtau);  tau_y = -(sin*utau + cos*vtau)
        # so the inverse is utau = -(cos*tau_x + sin*tau_y) and
        # vtau = sin*tau_x - cos*tau_y.  Derived from those two lines, not
        # assumed.
        return {"qsr": sw,
                "q_net": q_total,
                "emp": np.asarray(freshwater.evap, dtype=np.float64),
                "utau": np.asarray(surface.tau_i_native, dtype=np.float64),
                "vtau": np.asarray(surface.tau_j_native, dtype=np.float64),
                "utau_roundtrip": -(cos_u * tau_x + sin_u * tau_y),
                "vtau_roundtrip": sin_u * tau_x - cos_u * tau_y,
                "utau_roundtrip_transposed": -(cos_u * tau_y + sin_u * tau_x),
                "vtau_roundtrip_transposed": sin_u * tau_y - cos_u * tau_x}

    def literal(surface_ct, surface_sa, kt, nyear):
        ct = np.asarray(surface_ct, dtype=np.float64)
        pt = np.asarray(nemo_potential_temperature_from_conservative(
            jnp.asarray(ct), jnp.asarray(surface_sa)), dtype=np.float64)
        fields, _sites = r16._literal_sbc(
            lat, wet2, ct, pt, kt=kt, nyear=nyear,
            qsr_pi=(3.141592653589793 if plant == "forcing-qsr-pi" else None))
        # phase3_gate.py:673 materializes q_net = nemo_source_round(qns + qsr).
        # In fp64 that round is the identity, so the literal arm forms the same
        # binary64 sum rather than a difference of rounded sums, which would
        # manufacture a residual that neither model has.
        fields["q_net"] = np.asarray(fields["qns"], dtype=np.float64) + \
            np.asarray(fields["qsr"], dtype=np.float64)
        return fields

    rows = []
    # kt = 1, the ONE point the oracle dump certifies, carried here as the
    # gate's own anchor so a regression in either arm shows up next to the
    # day rows rather than in another artifact.
    anchor = gate.read_entry(
        Path(entry_root) / "oracle_step_entry_kt00000001.bin")
    cases = [("kt1", 1, np.asarray(anchor["T"])[..., 0],
              np.asarray(anchor["S"])[..., 0], 1, str(entry_root))]
    for day in days:
        step = day * STEPS_PER_DAY
        restart = year._load_nemo(nemo_root, seed, day, nlev)
        nyear, recorded_kt, adatrj, ndastp = _nyear_from_restart(
            Path(restart["path"]))
        require(recorded_kt == step,
                f"day {day}: restart kt={recorded_kt}, expected {step}")
        # The forcing that CONSUMES this state is step kt = n+1's, and its
        # nyear is the one `day(n+1)` sets.  Within a year the two agree; the
        # gate asserts that rather than assuming it, and records ndastp.
        # The forcing that CONSUMES the day-d restart is step d*6+1's.  At
        # d = 360 that step lies PAST nn_itend and the run never takes it;
        # the row is kept as a THIRTEENTH SAMPLE OF THE SEASONAL CLOCK (and
        # is labelled so), not as a step of the scored trajectory.  Its nyear
        # is the day-360 restart's, which is why the label says "clock".
        rows_kt = step + 1
        label = "clock" if step >= 2160 else "entering"
        cases.append((f"day{day:03d}_{label}_kt{rows_kt}", rows_kt,
                      np.asarray(restart["T"])[..., 0],
                      np.asarray(restart["S"])[..., 0], nyear,
                      restart["path"]))

    for name, kt, ct, sa, nyear, source in cases:
        # The plants perturb ONE arm.  legoESM's path has no nyear operand at
        # all -- that is the point of the nyear plant -- so the nyear plant
        # has to enter through the LITERAL arm, and the phase plant through
        # the current one.  A first version computed use_nyear and then passed
        # nyear, so the nyear plant was a dead variable and proved nothing.
        use_kt = kt + 1 if plant == "forcing-phase" else kt
        use_nyear = 2 if plant == "forcing-nyear" else nyear
        left = current(ct, sa, use_kt)
        right = literal(ct, sa, kt, use_nyear)
        row = {"case": name, "kt": kt, "nyear": nyear, "source": source,
               "ztime_hours": 4.0 * kt - (nyear - 1) * 24.0 * 360.0}
        for field in SBC_FIELDS:
            a = np.asarray(left[field], dtype=np.float64)
            b = np.asarray(right[field], dtype=np.float64)
            require(a.shape == b.shape == wet2.shape,
                    f"{name} {field}: shapes {a.shape} {b.shape}")
            unequal = int(np.count_nonzero(
                a[wet2].view(np.uint64) != b[wet2].view(np.uint64)))
            row[field] = {
                "wet_cells_unequal": unequal,
                "max_abs": float(np.max(np.abs((a - b)[wet2]))),
                "exact": unequal == 0,
                "field_max_abs": float(np.max(np.abs(b[wet2]))),
            }
        # THE ROTATION ROUND TRIP.  The five rows above pin the native pair
        # the model receives; they do NOT pin the geographic pair the shared
        # forcing object carries, which is where a transposition or a sign
        # error would live -- and an independent review proved that by
        # planting one and watching the five rows stay bit-exact.
        rotation = {}
        for native, recovered in (("utau", "utau_roundtrip"),
                                  ("vtau", "vtau_roundtrip")):
            reference = np.asarray(right[native], dtype=np.float64)
            identity = float(np.max(np.abs(
                (np.asarray(left[recovered]) - reference)[wet2])))
            transposed = float(np.max(np.abs(
                (np.asarray(left[f"{recovered}_transposed"])
                 - reference)[wet2])))
            ratio = (transposed / identity if identity > 0.0
                     else float("inf"))
            rotation[native] = {
                "identity_max_abs": identity,
                "transposed_max_abs": transposed,
                "ratio": ratio,
                "discriminates": ratio >= ROTATION_ROUNDTRIP_MIN_RATIO}
        # ANY component discriminating is enough, and the blind ones are
        # REPORTED rather than hidden.  GYRE is exactly 45 degrees
        # (sin_alpha_u == cos_alpha_u to 0.0, MEASURED) and its wind has
        # vtau == -utau to 0.0, so the transposed recovery of utau is
        # ALGEBRAICALLY the identity recovery: -(cos*tau_y + sin*tau_x)
        # = utau*(2 sin cos - cos^2 + sin^2) = utau at 45 degrees.  The utau
        # component is therefore structurally blind to a transposition on THIS
        # card, and vtau is what sees it.  A gate that demanded both would go
        # red on a correct model for a reason that is the card's geometry.
        rotation["blind_components"] = sorted(
            name for name, entry in rotation.items()
            if not entry["discriminates"])
        row["rotation_roundtrip"] = rotation
        row["exact"] = (all(row[f]["exact"] for f in SBC_FIELDS)
                        and any(entry["discriminates"]
                                for name, entry in rotation.items()
                                if name != "blind_components"))
        rows.append(row)

    exact = all(row["exact"] for row in rows)
    blind = sorted({name for row in rows
                    for name in row["rotation_roundtrip"]["blind_components"]})
    report = {"format": "gyre-year-owners-forcing-gate-v1", "case": CASE,
              "plant": plant, "seed": seed, "nemo_root": str(nemo_root),
              "mesh_sha256": mesh["mesh_sha256"], "rows": rows,
              "rotation_blind_components": blind,
              "all_bit_exact": exact, "worktree": worktree_stamp()}
    print(f"\nSURFACE-FORCING STATEMENT GATE -- legoESM's CURRENT path vs the "
          f"LITERAL usrdef_sbc, on NEMO'S OWN state.  BIT-EXACT bar.")
    print(f"  {'case':>26s}{'kt':>6s}{'ztime h':>10s}"
          + "".join(f"{f:>12s}" for f in SBC_FIELDS) + "   verdict")
    for row in rows:
        print(f"  {row['case']:>26s}{row['kt']:>6d}{row['ztime_hours']:>10.1f}"
              + "".join(f"{row[f]['wet_cells_unequal']:>12d}"
                        for f in SBC_FIELDS)
              + ("   BIT-EXACT" if row["exact"] else "   DEBT"))
    print(f"  ALL BIT-EXACT: {exact}")
    print(f"\n  THE GEOGRAPHIC STRESS ROTATION -- the five rows above are the "
          f"NATIVE pair the\n  model receives; this is the ROTATED pair it "
          f"also carries.  Recovering the\n  native pair from it costs ~1e-17 "
          f"Pa in binary64, so the row is scored by\n  how much further a "
          f"TRANSPOSED pair lands, not by bit equality.")
    print(f"  {'case':>26s}{'component':>11s}{'identity':>13s}"
          f"{'transposed':>13s}{'ratio':>13s}   verdict")
    for row in rows:
        for native, entry in row["rotation_roundtrip"].items():
            if native == "blind_components":
                continue
            print(f"  {row['case']:>26s}{native:>11s}"
                  f"{entry['identity_max_abs']:>13.4e}"
                  f"{entry['transposed_max_abs']:>13.4e}"
                  f"{entry['ratio']:>13.4e}"
                  + ("   DISCRIMINATES" if entry["discriminates"]
                     else "   BLIND (structural, see the code comment)"))
    if not exact:
        worst = max((row[f]["max_abs"], row["case"], f)
                    for row in rows for f in SBC_FIELDS)
        print(f"  worst |difference| {worst[0]:.6e} in {worst[2]} at "
              f"{worst[1]}")
    return report


# ------------------------------------------------------- the day-by-day gap --
def day_gap(*, lego_root: Path = DEFAULT_ROOT, lego_tag: str = "daily",
            nemo_root: Path | None = None, seed: int = 0,
            mesh_path: Path = DEFAULT_MESH, days: tuple[int, ...] = (),
            plant: str | None = None) -> dict:
    """The DAY-BY-DAY gap: one row per day, one metric, both sides identical.

    ``nemo_root`` must hold a restart at every requested day.  Until the
    early-days acquisition runs, the only days NEMO has are the year record's
    multiples of 30, so this refuses loudly rather than silently scoring a
    shorter list than it was asked for (the year round's own lesson).
    """
    _policy()
    year = _year()
    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, _dz, _dy, _area, bands = year._geometry(card, mesh_path)
    gate_masks = gate.expected_masks(card)
    masks = {"T": wet3, "S": wet3, "ssh": wet2,
             "u": gate_masks["u"], "v": gate_masks["v"]}
    rest = gate.lego_fields(card.recipe.initial_state)
    nlev = wet3.shape[-1]
    rows = []
    for day in days:
        read_day = day + 1 if plant == "day-offset" else day
        lego = year._load_lego(Path(lego_root), seed, day) if not lego_tag else (
            year._load_lego(Path(lego_root), f"{seed}_{lego_tag}", day))
        nemo = year._load_nemo(Path(nemo_root), seed, read_day, nlev)
        row = {"day": day, "nemo_day_read": read_day,
               "nemo_restart": nemo["path"], "nemo_sha256": nemo["sha256"]}
        for name in FIELDS:
            mask = masks[name]
            difference = (np.asarray(lego[name], dtype=np.float64)
                          - np.asarray(nemo[name], dtype=np.float64))
            signal = _rms(np.asarray(nemo[name], dtype=np.float64)
                          - np.asarray(rest[name], dtype=np.float64), mask)
            row[f"rms_{name}"] = _rms(difference, mask)
            row[f"frac_{name}"] = (row[f"rms_{name}"] / signal
                                   if signal > 0.0 else float("nan"))
        for name, mask in bands.items():
            row[f"rms_T3D_{name}"] = _rms(
                np.asarray(lego["T"], dtype=np.float64)
                - np.asarray(nemo["T"], dtype=np.float64), mask)
        rows.append(row)
    ratios = [rows[i]["rms_T"] / rows[i - 1]["rms_T"]
              for i in range(1, len(rows)) if rows[i - 1]["rms_T"] > 0]
    report = {"format": "gyre-year-owners-day-gap-v1", "case": CASE,
              "seed": seed, "plant": plant, "rows": rows,
              "max_day_to_day_ratio": max(ratios) if ratios else None,
              "worktree": worktree_stamp()}
    print(f"\nDAY-BY-DAY GAP -- legoESM vs NEMO, wet rms, fp64")
    print(f"  {'day':>5s}" + "".join(f"{f'rms {n}':>14s}" for n in FIELDS)
          + f"{'ratio':>9s}   leads")
    previous = None
    for row in rows:
        ratio = (row["rms_T"] / previous if previous else float("nan"))
        previous = row["rms_T"]
        finite = [n for n in FIELDS if np.isfinite(row[f"frac_{n}"])]
        leads = max(finite, key=lambda n: row[f"frac_{n}"]) if finite else "-"
        print(f"  {row['day']:>5d}"
              + "".join(f"{row[f'rms_{n}']:>14.4e}" for n in FIELDS)
              + f"{ratio:>9.3f}   {leads}")
    if report["max_day_to_day_ratio"] is not None:
        print(f"  max day-to-day T ratio {report['max_day_to_day_ratio']:.3f}")
    return report


# ------------------------------------------------------- the switch trace ---
def switch_trace(steps: int, out: Path, *, seeds: tuple[int, int] = (0, 1),
                 mesh_path: Path = DEFAULT_MESH, every: int = 1,
                 start: int = 1, plant: str | None = None) -> dict:
    """The ENHANCED-VERTICAL-DIFFUSION trigger mask, step by step, two members.

    The year receipt measures one member moving five orders of magnitude
    between day 180 and day 210 while three others did not, and names NEMO's
    ``ln_zdfevd`` hard branch (``IF( MIN(rn2,rn2b) <= -1.e-12 ) p_avt =
    rn_evd``) as the PLAUSIBLE mechanism with no cell-level trace run.  This
    is that trace.

    The mask is taken from the MODEL'S OWN convective coefficient field, not
    from a re-derivation of its trigger: ``_enhanced_diffusion_K`` is the
    function the step calls, and a cell has fired iff its tracer diffusivity
    reaches ``K_conv``.  ponytail: importing the model's own private helper in
    a probe beats re-deriving the trigger and then arguing they agree.
    """
    _policy()
    import jax.numpy as jnp
    year = _year()
    gate = _gate()
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        _enhanced_diffusion_K)

    card = build_nemo_testcase_card(CASE)
    conv = card.recipe.model_config.physics.convection
    k_conv = float(conv.enhanced_diffusion.K_conv)
    require(k_conv > 0.0, "the card's EVD K_conv is zero; nothing can fire")
    mesh = year.nemo_operands(mesh_path)
    active = (np.asarray(card.recipe.z_coord.is_active)
              & (np.asarray(card.recipe.land_mask) > 0.5)[..., None])
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)

    def fired(state):
        # NEMO's trigger is MIN(rn2, rn2b) (zdfevd.f90:108).  On THIS card
        # stprk3.F90:173-174 sets `rn2 = rn2b` and then calls
        # `zdf_phy(kstp, Nbb, Nbb, Nrhs)`, so both arms ARE the whole-step
        # entry tracer and the MIN is over two identical fields.  The before
        # arm is passed explicitly anyway, so the two-level branch the model
        # runs is the branch measured here rather than an argued equivalent.
        K, _A = _enhanced_diffusion_K(
            state, card.recipe.z_coord, conv,
            before_tracers=(state.T.data, state.S.data),
            cc=card.recipe.model_config.physics.constants)
        mask = np.asarray(K, dtype=np.float64) >= k_conv
        if plant == "switch-blind":
            mask = np.zeros_like(mask)
        return mask

    models, states = {}, {}
    for seed in seeds:
        pert = year.nemo_istate_perturbation(
            np.broadcast_to(depth, depth.shape), np.broadcast_to(
                np.asarray(card.recipe.grid.native_lat_T_deg,
                           dtype=np.float64)[..., None], depth.shape),
            active.astype(np.float64), seed)
        state = card.recipe.initial_state
        state = state._replace(T=state.T.replace(
            data=jnp.asarray(np.asarray(state.T.data) + pert,
                             dtype=jnp.float64)))
        states[seed] = state
        models[seed] = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config)

    left, right = seeds
    rows = []
    first_difference = None
    started = time.time()
    for completed in range(steps):
        kt = completed + 1
        if kt >= start and (kt - start) % every == 0:
            masks = {seed: fired(states[seed]) for seed in seeds}
            differing = np.asarray(masks[left] != masks[right])
            count = int(np.count_nonzero(differing))
            row = {"kt": kt, "day": (kt - 1) / STEPS_PER_DAY,
                   "fired_left": int(masks[left].sum()),
                   "fired_right": int(masks[right].sum()),
                   "cells_differing": count,
                   "T_rms": _rms(np.asarray(states[left].T.data)
                                 - np.asarray(states[right].T.data), active)}
            if count and first_difference is None:
                j, i, k = (int(x) for x in np.argwhere(differing)[0])
                first_difference = {
                    "kt": kt, "day": (kt - 1) / STEPS_PER_DAY,
                    "j": j, "i": i, "k": k,
                    "depth_m": float(depth[j, i, k]),
                    "lat_deg": float(lat[j, i]),
                    "cells_differing": count,
                    "T_rms_at_crossing": row["T_rms"],
                    "fired_left": row["fired_left"],
                    "fired_right": row["fired_right"]}
                row["first_difference"] = first_difference
            rows.append(row)
            print(f"  kt {kt:>5d} day {row['day']:>7.2f}  fired "
                  f"{row['fired_left']:>5d}/{row['fired_right']:>5d}  "
                  f"differing {count:>4d}  T rms {row['T_rms']:.4e}",
                  flush=True)
        for seed in seeds:
            freshwater, surface = gate._surface_forcings(
                card, states[seed], kt)
            states[seed] = models[seed].step(
                states[seed], dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
    require(any(row["fired_left"] or row["fired_right"] for row in rows),
            "the EVD trigger NEVER fired on either member over the traced "
            "window; the instrument sees nothing and no 'no crossing' "
            "statement may be made from it")
    report = {"format": "gyre-year-owners-switch-trace-v1", "case": CASE,
              "seeds": list(seeds), "steps": steps, "every": every,
              "start": start, "plant": plant, "K_conv": k_conv,
              "n2_threshold": float(conv.enhanced_diffusion.n2_threshold),
              "evd_n2_time_level": conv.enhanced_diffusion.evd_n2_time_level,
              "rows": rows, "first_difference": first_difference,
              "wall_seconds": time.time() - started,
              "worktree": worktree_stamp()}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"switch_trace_seeds{left}{right}.json").write_text(
        json.dumps(report, indent=2))
    print(f"  FIRST DIFFERING TRIGGER MASK: {first_difference}")
    return report


# ------------------------------------------------------------- self-check ---
def self_check() -> int:
    """The arithmetic, and every plant SHOWN to fail."""
    failures = []

    def expect_raises(label, fn):
        try:
            fn()
        except Exception as error:                   # noqa: BLE001
            print(f"  PLANT {label}: raised {type(error).__name__} -- OK")
            return
        failures.append(label)
        print(f"  PLANT {label}: DID NOT FAIL")

    # 1. the region cut partitions the wet surface exactly once
    lat = np.tile(np.linspace(10.0, 50.0, 6)[:, None], (1, 6))
    wet2 = np.zeros((6, 6), dtype=bool)
    wet2[1:-1, 1:-1] = True
    regions = _regions(lat, wet2)
    thirds = (regions["west_third"].astype(int)
              + regions["interior_third"].astype(int)
              + regions["east_third"].astype(int))
    if not np.array_equal(thirds, wet2.astype(int)):
        failures.append("thirds do not partition the wet surface")
    else:
        print("  thirds partition the wet surface exactly once -- OK")
    emp = (regions[f"emp_south_le_{EMP_SPLIT_LAT_DEG}N"].astype(int)
           + regions[f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N"].astype(int))
    if not np.array_equal(emp, wet2.astype(int)):
        failures.append("the emp split does not partition the wet surface")
    else:
        print("  the emp split partitions the wet surface exactly once -- OK")
    for name in ("west_third", "east_third", "wind_band_15_29N"):
        if not regions[name].any():
            failures.append(f"region {name} is empty -- the cut is vacuous")
    # 2. the literal SBC's kt default is byte-unchanged by the new argument
    r16 = _round16()
    small_lat = np.array([[20.0, 40.0]])
    small_wet = np.array([[True, True]])
    ct = np.array([[20.0, 10.0]])
    pt = np.array([[19.9, 9.9]])
    base, _ = r16._literal_sbc(small_lat, small_wet, ct, pt)
    again, _ = r16._literal_sbc(small_lat, small_wet, ct, pt, kt=1, nyear=1)
    # NOTE: a first version wrote this as a for/else with no break, so the
    # "OK" line printed even when the loop had appended failures.  An
    # independent review found it.  The five fields checked are the literal
    # transcription's own, not the gate's model-facing rows.
    literal_fields = ("qsr", "qns", "emp", "utau", "vtau")
    drifted = [field for field in literal_fields
               if not np.array_equal(np.asarray(base[field]).view(np.uint64),
                                     np.asarray(again[field]).view(np.uint64))]
    if drifted:
        failures.append(f"_literal_sbc default changed {drifted}")
    else:
        print("  _literal_sbc's kt=1/nyear=1 default is byte-unchanged -- OK")
    # 3. the literal SBC MOVES with kt, so a phase plant CAN be detected
    later, _ = r16._literal_sbc(small_lat, small_wet, ct, pt, kt=181)
    moved = [f for f in literal_fields
             if not np.array_equal(np.asarray(base[f]).view(np.uint64),
                                   np.asarray(later[f]).view(np.uint64))]
    if set(moved) != set(literal_fields):
        failures.append(f"only {moved} move between kt=1 and kt=181; a phase "
                        "plant could not be seen on the others")
    else:
        print("  every SBC field moves between kt=1 and kt=181 -- the phase "
              "plant is non-vacuous -- OK")
    # 4. the nyear term is a NO-OP inside year 1 and NOT a no-op at year 2
    y2, _ = r16._literal_sbc(small_lat, small_wet, ct, pt, kt=181, nyear=2)
    if np.array_equal(np.asarray(later["qsr"]).view(np.uint64),
                      np.asarray(y2["qsr"]).view(np.uint64)):
        failures.append("the nyear plant changes nothing; it is vacuous")
    else:
        print("  the nyear=2 plant moves qsr -- non-vacuous -- OK")
    # 5. the qsr_pi plant moves qsr (usrdef_sbc's literal is 3.1415, not rpi)
    swapped, _ = r16._literal_sbc(small_lat, small_wet, ct, pt,
                                  qsr_pi=3.141592653589793)
    if np.array_equal(np.asarray(base["qsr"]).view(np.uint64),
                      np.asarray(swapped["qsr"]).view(np.uint64)):
        failures.append("3.1415 -> rpi changes nothing; the plant is vacuous")
    else:
        print(f"  3.1415 -> rpi moves qsr by "
              f"{float(np.max(np.abs(base['qsr'] - swapped['qsr']))):.4e} "
              "W/m2 -- non-vacuous -- OK")
    # 6. the registry now reads the sixty dumps the card writes, and only those
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for kt in (1, 10, 11, 60):
        if time_level_for_dump(f"oracle_step_entry_kt{kt:08d}.bin") != "before":
            failures.append(f"entry dump kt={kt} is not registered 'before'")
    expect_raises("registry-beyond-60",
                  lambda: time_level_for_dump(
                      "oracle_step_entry_kt00000061.bin"))
    # 7. the long-horizon process budget is additive, and its RHS ULP control
    # reaches a decoded raw process boundary rather than perturbing a zero.
    _process_math_self_check(failures)
    # 8. the Round-125 record geometry is frozen independently of a future
    # acquisition, and its trajectory plant detects exactly one changed bit.
    if (VERTICAL_RECORD_BYTES != 6_965_416
            or len(VERTICAL_RECORD_STEPS) != 362
            or VERTICAL_RECORD_STEPS[:2] != (1, 2)
            or VERTICAL_RECORD_STEPS[2:] != tuple(range(1081, 1441))):
        failures.append("Round-125 vertical-record geometry changed")
    else:
        print("  Round-125 vertical-record geometry is frozen -- OK")
    vertical_baseline = np.ones((2, 2, 2), dtype=np.float64)
    vertical_mask = np.ones_like(vertical_baseline, dtype=bool)
    control = _vertical_trajectory_ulp_control(
        vertical_baseline, vertical_baseline.copy(), vertical_mask)
    if control["cells_unequal"] != 1:
        failures.append("vertical trajectory plant did not move one cell")
    else:
        print("  vertical trajectory ULP plant moves one cell -- OK")
    broken_vertical = vertical_baseline.copy()
    broken_vertical[0, 0, 0] = np.nextafter(
        broken_vertical[0, 0, 0], np.inf)
    expect_raises(
        "vertical-trajectory-blind",
        lambda: _vertical_trajectory_ulp_control(
            broken_vertical, vertical_baseline, vertical_mask))
    if failures:
        for item in failures:
            print(f"  FAILED: {item}")
        return 1
    print("  self-check: all checks passed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-gap", type=int, default=None,
                        help="walk N legoESM steps against NEMO's per-step "
                             "entry dumps")
    parser.add_argument("--decompose", type=int, default=None,
                        help="decompose the gap at this day")
    parser.add_argument("--process-record", type=Path, default=None,
                        help="validate a Round-123 process-record root")
    parser.add_argument("--produce-process-trace", action="store_true",
                        help="run the independent legoESM Round-124 process "
                             "trace into --root")
    parser.add_argument("--lego-process-record", type=Path, default=None,
                        help="validate a Round-124 legoESM process trace")
    parser.add_argument("--process-budget", type=Path, default=None,
                        help="score this NEMO process root against "
                             "--lego-process-record")
    parser.add_argument("--vertical-record", type=Path, default=None,
                        help="validate a Round-125 tra_zdf internal-record "
                             "root")
    parser.add_argument("--vertical-process-root", type=Path,
                        default=DEFAULT_PROCESS_RECORD_ROOT,
                        help="admitted Round-123 process root used to align "
                             "the vertical record step by step")
    parser.add_argument("--expect-commit", default=None,
                        help="required clean producer commit for records")
    parser.add_argument("--forcing-gate", action="store_true")
    parser.add_argument("--day-gap", action="store_true")
    parser.add_argument("--switch-trace", type=int, default=None,
                        help="trace the EVD trigger mask over N steps")
    parser.add_argument("--equal-input", type=int, default=None,
                        help="run step KT from NEMO's OWN entry state and "
                             "compare against NEMO's next entry state")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lego-root", type=Path, default=None)
    parser.add_argument("--lego-tag", default="daily")
    parser.add_argument("--nemo-root", type=Path, default=YEAR_ROOT)
    parser.add_argument("--immutable-lego-root", type=Path,
                        default=DEFAULT_IMMUTABLE_GYRE_YEAR)
    parser.add_argument("--entry-root", type=Path, default=DEFAULT_ENTRY_ROOT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--days", default=None,
                        help="comma-separated day list")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plant", default=None)
    args = parser.parse_args(argv)

    report = None
    if args.self_check:
        return self_check()
    if args.produce_process_trace:
        require(args.expect_commit is not None,
                "--produce-process-trace needs --expect-commit")
        report = produce_lego_process_trace(
            args.root, args.expect_commit, mesh_path=args.mesh)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        return 0
    if args.vertical_record is not None:
        require(args.expect_commit is not None,
                "--vertical-record needs --expect-commit")
        report = validate_vertical_record(
            args.vertical_record, args.expect_commit,
            process_root=args.vertical_process_root, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report.get('reason', report.get('control', 'moved'))}")
            return 1
        print("STATUS PASS: vertical tra_zdf record "
              f"{report['layout']['record_count']} frames, "
              f"{report['layout']['total_record_bytes']} bytes")
        return 0
    if args.process_budget is not None:
        require(args.lego_process_record is not None,
                "--process-budget needs --lego-process-record")
        require(args.expect_commit is not None,
                "--process-budget needs --expect-commit")
        report = score_process_budget(
            args.process_budget, args.lego_process_record,
            args.expect_commit, immutable_lego_root=args.immutable_lego_root,
            nemo_root=args.nemo_root, mesh_path=args.mesh)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        return 0
    if args.lego_process_record is not None:
        require(args.expect_commit is not None,
                "--lego-process-record needs --expect-commit")
        report = validate_lego_process_trace(
            args.lego_process_record, args.expect_commit,
            plant=args.plant, mesh_path=args.mesh)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report.get('reason', report.get('control', 'moved'))}")
            return 1
        print("STATUS PASS: legoESM process trace "
              f"{report['layout']['record_count']} frames")
        return 0
    if args.process_record is not None:
        require(args.expect_commit is not None,
                "--process-record needs --expect-commit")
        report = validate_process_record(
            args.process_record, args.expect_commit, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report.get('reason', report.get('control', 'moved'))}")
            return 1
        print("STATUS PASS: round123 process record "
              f"{report['layout']['record_count']} frames, "
              f"{report['layout']['total_record_bytes']} bytes")
        return 0
    if args.step_gap is not None:
        report = step_gap(args.step_gap, args.root,
                          entry_root=args.entry_root, plant=args.plant)
    elif args.decompose is not None:
        report = decompose(args.decompose,
                           lego_root=(args.lego_root or YEAR_HEAD),
                           nemo_root=args.nemo_root, seed=args.seed,
                           mesh_path=args.mesh)
    elif args.forcing_gate:
        days = (tuple(int(x) for x in args.days.split(","))
                if args.days else tuple(range(30, 361, 30)))
        report = forcing_gate(nemo_root=args.nemo_root, seed=args.seed,
                              mesh_path=args.mesh,
                              entry_root=args.entry_root, days=days,
                              plant=args.plant)
        if not report["all_bit_exact"]:
            print("FORCING STATEMENT GATE: DEBT")
            if args.json:
                Path(args.json).write_text(json.dumps(report, indent=2))
            return 1
    elif args.day_gap:
        require(args.days is not None, "--day-gap needs --days")
        report = day_gap(lego_root=(args.lego_root or args.root),
                         lego_tag=args.lego_tag, nemo_root=args.nemo_root,
                         seed=args.seed, mesh_path=args.mesh,
                         days=tuple(int(x) for x in args.days.split(",")),
                         plant=args.plant)
    elif args.equal_input is not None:
        report = equal_input_step(args.equal_input,
                                  entry_root=args.entry_root, plant=args.plant)
    elif args.switch_trace is not None:
        left, right = (int(x) for x in args.seeds.split(","))
        report = switch_trace(args.switch_trace, args.root,
                              seeds=(left, right), mesh_path=args.mesh,
                              every=args.every, start=args.start,
                              plant=args.plant)
    else:
        parser.error("choose a mode")
    if args.json and report is not None:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json}")
    return 0




# --------------------------------------------- the EQUAL-INPUT single step ---
def equal_input_step(kt: int, *, entry_root: Path = DEFAULT_ENTRY_ROOT,
                     plant: str | None = None) -> dict:
    """Does step ``kt`` CREATE the difference, or merely AMPLIFY the one it is
    handed?

    The per-step walk shows the temperature difference going from `2.0e-15` K
    entering step 2 to `4.2e-04` K entering step 3.  A trajectory comparison
    cannot separate "step 2's operators disagree" from "step 2 amplified what
    step 1 left".  This separates them by running step ``kt`` from NEMO'S OWN
    entry state and comparing the output against NEMO's own next entry state:

      FREE-RUN  legoESM's own state entering kt   -> its own state entering kt+1
      EQUAL-IN  NEMO's state entering kt          -> legoESM's state entering kt+1

    If EQUAL-IN collapses to rounding, the step's operators agree and the
    difference was carried in.  If it does not, the operators disagree and the
    step is an owner.

    TWO ARMS, because one prognostic pair has no entry record.  The step-entry
    dump carries ``ts/uu/vv/ssh`` and NOT the prognostic barotropic pair
    ``uu_b``/``vv_b``, which this card requires.  Arm A reseeds the 3-D state
    only and leaves legoESM's own barotropic pair; arm B additionally reseeds
    that pair from ``oracle_bt_frames_kt{kt-1}.bin``, which the card's writer
    records at ``Naa`` after ``stp_2D`` and which ``stprk3`` then swaps into
    ``Nbb`` -- i.e. exactly the seed step ``kt`` consumes.  A and B differing is
    itself the measurement of how much the barotropic pair carries.
    """
    require(kt >= 2, "equal-input needs a step whose entry NEMO recorded")
    _policy()
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    gate = _gate()
    baro = _load("nemo_testcase_overflow_barotropic_gate",
                 "nemo_testcase_overflow_barotropic_gate.py")
    card = build_nemo_testcase_card(CASE)
    masks = gate.expected_masks(card)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)

    entry = {n: gate.read_entry(
        Path(entry_root) / f"oracle_step_entry_kt{n:08d}.bin")
        for n in (kt, kt + 1)}
    frames = gate.read_bt(
        Path(entry_root) / f"oracle_bt_frames_kt{kt - 1:08d}.bin", kt - 1)

    # The free-run arm: step the card forward to the entry of kt, as the
    # certified ladder does, and take one more step.
    state = card.recipe.initial_state
    for completed in range(kt - 1):
        freshwater, surface = gate._surface_forcings(card, state, completed + 1)
        state = model.step(state, dt=card.dt_s, freshwater=freshwater,
                           surface_forcing=surface)
    free_entry = state

    def one_step(start):
        freshwater, surface = gate._surface_forcings(card, start, kt)
        return model.step(start, dt=card.dt_s, freshwater=freshwater,
                          surface_forcing=surface)

    seeded = baro.state_from_oracle_entry(free_entry, entry[kt], masks)
    if plant == "equal-input-noop":
        # The plant this arm needs: if the reseed silently did nothing, the
        # two arms would be identical and the measurement would be vacuous.
        seeded = free_entry
    seeded_b = seeded
    moved_barotropic = {}
    for name, values in (("uu_b", frames["uu_b"]), ("vv_b", frames["vv_b"])):
        field = getattr(seeded_b, name)
        require(field is not None, f"the card carries no {name}")
        current = np.array(field.data, dtype=np.float64, copy=True)
        reference = np.asarray(values, dtype=np.float64)
        before = np.array(current, copy=True)
        if name == "uu_b":
            current[:, 1:] = np.where(masks["u"][..., 0], reference,
                                      current[:, 1:])
        else:
            current[1:, :] = np.where(masks["v"][..., 0], reference,
                                      current[1:, :])
        # The barotropic arm exists to see whether the prognostic pair carries
        # the difference.  If the reseed writes nothing, "the arm changed
        # nothing" is a statement about the harness, not about the physics --
        # so the size of the write is MEASURED and reported next to the arm.
        moved_barotropic[name] = float(np.max(np.abs(current - before)))
        seeded_b = seeded_b._replace(
            **{name: field.replace(data=jnp.asarray(current))})

    rows = {}
    for arm, start in (("free_run", free_entry), ("equal_input_3d", seeded),
                       ("equal_input_3d_plus_barotropic", seeded_b)):
        out = gate.lego_fields(one_step(start))
        row = {}
        for name in FIELDS:
            mask = masks[name]
            left = np.asarray(out[name], dtype=np.float64)
            right = np.asarray(entry[kt + 1][name], dtype=np.float64)
            if right.ndim == 3:
                right = right[..., :left.shape[-1]]
            row[name] = {
                "rms": _rms(left - right, mask),
                "max_abs": float(np.max(np.abs((left - right)[mask]))),
                "cells_unequal": int(np.count_nonzero(
                    left[mask].view(np.uint64) != right[mask].view(np.uint64))),
            }
        rows[arm] = row
    # The reseed must actually have changed the input, or every arm is the
    # free run wearing a different name.
    moved = {name: float(np.max(np.abs(
        np.asarray(gate.lego_fields(seeded)[name], dtype=np.float64)
        - np.asarray(gate.lego_fields(free_entry)[name], dtype=np.float64))))
        for name in FIELDS}
    require(max(moved.values()) > 0.0,
            "the reseed changed NOTHING, so the equal-input arm is the free "
            "run under another name and measures nothing")
    report = {"format": "gyre-year-owners-equal-input-v2", "case": CASE,
              "kt": kt, "plant": plant, "entry_root": str(entry_root),
              "reseed_moved_input_by": moved,
              "reseed_moved_barotropic_by": moved_barotropic, "arms": rows,
              "worktree": worktree_stamp()}
    print(f"\nEQUAL-INPUT STEP {kt}: does the step CREATE the difference or "
          f"AMPLIFY the one it is handed?")
    print(f"  the reseed moved the 3-D INPUT by: "
          + "  ".join(f"{n} {moved[n]:.4e}" for n in FIELDS))
    print(f"  the reseed moved the BAROTROPIC pair by: "
          + "  ".join(f"{n} {v:.4e}" for n, v in moved_barotropic.items()))
    print(f"  {'arm':>32s}" + "".join(f"{f'rms {n}':>14s}" for n in FIELDS))
    for arm, row in rows.items():
        print(f"  {arm:>32s}"
              + "".join(f"{row[n]['rms']:>14.4e}" for n in FIELDS))
    return report


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

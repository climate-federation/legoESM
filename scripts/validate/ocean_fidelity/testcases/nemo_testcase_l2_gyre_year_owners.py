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
  --produce-vertical-trace
                     extends the existing production process trace with the
                     exact tracer-ZDF operands consumed during steps
                     1081--1440; ordinary carried state remains separate.
  --vertical-budget  admits both vertical traces and decomposes the inherited
                     day-240 vertical-diffusion carry by source boundary.
  --trigger-temperature-process-budget
                     admits the same process/vertical traces and propagates
                     their cumulative temperature rows through the production
                     EVD-trigger closure in the Round-127 Shapley contexts.
  --developed-step-walk
                     drives one production-JIT step from NEMO's admitted
                     day-180 restart and compares the recorded stage-3
                     temperature boundaries and active-branch maps.
  --developed-transport-walk
                     extends that same production step across the Round-154
                     stage-3 U-transport operands and written values.
  --developed-vertical-sensitivity
                     drives independent day-180-to-240 production-JIT arms
                     with NEMO's recorded heat or complete effective tracer
                     diffusivity at the implicit-solve boundary.
  --forcing-gate     legoESM's CURRENT surface forcing against the LITERAL
                     usrdef_sbc transcription, BIT-EXACT, evaluated on NEMO's
                     OWN state at every day boundary the record holds.  This is
                     the statement test the kt=1 dump cannot perform, because
                     kt=1 samples the seasonal clock at ONE of its 2160 values.
  --switch-trace     the enhanced-vertical-diffusion trigger mask, step by
                     step, on two members; reports the FIRST step at which the
                     two masks differ.
  --self-check       the arithmetic and every plant.

PLANTS (each exits NON-ZERO; controls have committed synthetic tests and the
record-backed plants are persisted in their round evidence)
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
  lego-vertical-stamp      supplies a wrong legoESM vertical-trace commit
  lego-vertical-matrix-ulp changes one stored, consumed matrix coefficient
  trigger-process-registry removes one real cumulative temperature row and
                           proves the recorded Kbb endpoint no longer closes
  trigger-process-level    perturbs a consumed temperature level until the
                           production N2 threshold changes
  missing-day              removes a developed-state requested-day row
  missing-process-row      removes a developed-state process boundary
  missing-branch           removes a developed-state branch family
  entry-temperature-ulp    moves one consumed entry T value by one ULP
  transport-un-adv-ulp     moves one observed stage-3 un_adv value by one ULP
                           and requires a written transport row to change
  developed-vertical-avt-ulp
                           moves one consumed NEMO avt interface by one ULP;
                           the matrix and day-240 temperature must both move

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
DAILY_ATTRIBUTION_FAMILIES = ("tracer", "vector", "ssh", "tke")
DAILY_ATTRIBUTION_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
DAILY_TRACER_SUBFAMILIES = ("temperature", "salinity")
DAILY_TRACER_CADENCES_DAYS = (1, 2, 4, 8)
DAILY_ATTRIBUTION_BASELINE_T = {
    30: 6.89043148782590898e-05,
    240: 1.64467402331753935e-02,
    360: 1.12235712478436639e-02,
}
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
LEGO_VERTICAL_FIELDS = (
    "heat_K", "isoneutral_K", "effective_K", "e3t_after", "e3w_now",
    "content_T", "lower", "diagonal", "upper", "solved_T",
)
DEFAULT_REFERENCE_PROCESS_TRACE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round124/"
    "lego_process_trace_v3")
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


def _vertical_trace_frame(trace) -> dict[str, np.ndarray]:
    """Copy the exact tracer-ZDF arrays returned by the production closure."""
    vertical = trace.vertical_solve
    return {
        "heat_K": np.asarray(vertical.heat_K, dtype=np.float64),
        "isoneutral_K": np.asarray(
            vertical.isoneutral_K, dtype=np.float64),
        "effective_K": np.asarray(vertical.effective_K, dtype=np.float64),
        "e3t_after": np.asarray(vertical.e3t_after, dtype=np.float64),
        "e3w_now": np.asarray(vertical.e3w_now, dtype=np.float64),
        "content_T": np.asarray(vertical.content_T, dtype=np.float64),
        "lower": np.asarray(vertical.lower, dtype=np.float64),
        "diagonal": np.asarray(vertical.diagonal, dtype=np.float64),
        "upper": np.asarray(vertical.upper, dtype=np.float64),
        "solved_T": np.asarray(vertical.solved_T, dtype=np.float64),
    }


def _vertical_effect_control(model, state, freshwater, surface, trace,
                             wet: np.ndarray) -> dict:
    """Perturb one consumed heat-K interface in the full production step."""
    import jax.numpy as jnp

    baseline_vertical = _vertical_trace_frame(trace)
    interface_wet = wet[..., :-1] & wet[..., 1:]
    candidates = np.argwhere(
        interface_wet & np.isfinite(baseline_vertical["heat_K"])
        & (baseline_vertical["heat_K"] > 0.0))
    require(candidates.size > 0,
            "vertical effect plant found no positive wet heat-K interface")
    index = tuple(int(x) for x in candidates[0])
    heat = np.array(baseline_vertical["heat_K"], copy=True)
    old = float(heat[index])
    # A single stored-matrix ULP is tested separately.  This production-effect
    # plant uses a small, fixed physical coefficient delta so it cannot pass
    # merely by moving a diagnostic input while rounding out of the solve.
    delta = float(np.ldexp(1.0, -40))
    heat[index] = old + delta
    require(heat[index] != old, "vertical effect plant rounded to zero")
    planted = model.step(
        state, dt=DT_S, freshwater=freshwater, surface_forcing=surface,
        _vertical_K_test_override=(
            jnp.asarray(heat), trace.vertical_solve.viscosity_K))
    planted_vertical = _vertical_trace_frame(planted)
    baseline_process = _trace_frame(trace)
    planted_process = _trace_frame(planted)
    upstream_names = tuple(name for name in LEGO_PROCESS_FIELDS if name != "Taa")
    upstream_moved = {
        name: _different_cells(
            baseline_process[name], planted_process[name],
            wet if baseline_process[name].ndim == 3 else np.any(wet, axis=-1))
        for name in upstream_names
    }
    moved = {
        name: _different_cells(
            baseline_vertical[name], planted_vertical[name],
            interface_wet if baseline_vertical[name].shape[-1] == wet.shape[-1] - 1
            else wet)
        for name in LEGO_VERTICAL_FIELDS
    }
    solved_moved = _different_cells(
        np.asarray(trace.vertical_solve.solved_T),
        np.asarray(planted.vertical_solve.solved_T), wet)
    require(all(value == 0 for value in upstream_moved.values()),
            "vertical effect plant moved an upstream process boundary")
    require(moved["heat_K"] == 1 and moved["effective_K"] == 1,
            "vertical effect plant did not move exactly one K interface")
    require(sum(moved[name] for name in ("lower", "diagonal", "upper")) > 0,
            "vertical effect plant did not reach a consumed matrix")
    require(solved_moved > 0,
            "vertical effect plant did not reach the solved temperature")
    return {
        "status": "PLANT-FIRED", "index_jik": list(index),
        "heat_K_before_m2_s": old,
        "heat_K_delta_m2_s": delta,
        "upstream_cells_moved": upstream_moved,
        "vertical_cells_moved": moved,
        "solved_temperature_cells_moved": solved_moved,
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
                               *, mesh_path: Path = DEFAULT_MESH,
                               include_vertical: bool = False,
                               reference_process_trace: Path =
                               DEFAULT_REFERENCE_PROCESS_TRACE) -> dict:
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
    vertical_model = (
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                tracer_process_trace=(), vertical_solve_trace=True))
        if include_vertical else trace_model)
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
    trace_fields = list(LEGO_PROCESS_FIELDS)
    if include_vertical:
        trace_fields.extend(LEGO_VERTICAL_FIELDS)
    for name in trace_fields:
        if name.startswith("q_"):
            shape = (LEGO_PROCESS_TRACE_STEPS,) + shape2
        elif name in ("heat_K", "isoneutral_K", "effective_K", "e3w_now"):
            shape = (LEGO_PROCESS_TRACE_STEPS,) + shape3[:-1] + (
                shape3[-1] - 1,)
        else:
            shape = (LEGO_PROCESS_TRACE_STEPS,) + shape3
        path = root / f"{name}.npy"
        paths.append(path)
        maps[name] = open_memmap(path, mode="w+", dtype="<f8", shape=shape)

    snapshot_root = root / "lego_seed0"
    snapshot_root.mkdir()
    started = time.time()
    total_unequal_bytes = 0
    max_step_unequal_bytes = 0
    effect_control = None
    vertical_effect_control = None
    reference_arrays = (
        _load_lego_trace_arrays(Path(reference_process_trace))
        if include_vertical else None)
    reference_moved = {name: 0 for name in LEGO_PROCESS_FIELDS}
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
            if kt == 1260:
                np.savez(snapshot_root / "day210.npz",
                         **year._snapshot(state, gate))
            continue

        trace = trace_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        vertical_trace = (
            vertical_model.step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
            if include_vertical else trace)
        # ``step`` replaces each write-only result's state_after with its own
        # independently compiled ordinary production call.  Use the unchanged
        # Round-124 observer's copy as the trajectory; comparing the vertical
        # observer against it proves both reference calls agree byte-for-byte.
        reference = trace.state_after
        unequal_bytes = _state_bit_mismatches(trace.state_after, reference)
        if include_vertical:
            unequal_bytes += _state_bit_mismatches(
                vertical_trace.state_after, reference)
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
            if include_vertical:
                vertical_effect_control = _vertical_effect_control(
                    vertical_model, state, freshwater, surface,
                    vertical_trace, wet)
        frame = _trace_frame(trace)
        if include_vertical:
            frame.update(_vertical_trace_frame(vertical_trace))
        index = kt - PROCESS_START_STEP
        for name in trace_fields:
            require(frame[name].shape == maps[name].shape[1:],
                    f"step {kt} {name}: shape {frame[name].shape}, expected "
                    f"{maps[name].shape[1:]}")
            require(np.all(np.isfinite(frame[name])),
                    f"step {kt} {name}: non-finite trace value")
            maps[name][index] = frame[name]
        if reference_arrays is not None:
            for name in LEGO_PROCESS_FIELDS:
                active_mask = wet if frame[name].ndim == 3 else np.any(
                    wet, axis=-1)
                reference_moved[name] += _different_cells(
                    frame[name], np.asarray(reference_arrays[name][index]),
                    active_mask)
        state = reference
        if kt == 1260:
            np.savez(snapshot_root / "day210.npz",
                     **year._snapshot(state, gate))
        if kt % 60 == 0:
            print(f"  lego process trace step {kt:4d}  "
                  f"{time.time() - started:7.1f} s", flush=True)

    np.savez(snapshot_root / "day240.npz", **year._snapshot(state, gate))
    for array in maps.values():
        array.flush()
    del maps
    require(effect_control is not None, "production effect plant never ran")
    if include_vertical:
        require(vertical_effect_control is not None,
                "vertical production effect plant never ran")
        require(all(value == 0 for value in reference_moved.values()),
                "fresh vertical trace differs from the admitted Round-124 "
                "process trajectory: " + ", ".join(
                    f"{name}={value}" for name, value in reference_moved.items()
                    if value))
    metadata = {
        "format": ("gyre-legoesm-process-vertical-trace-v2"
                   if include_vertical else "gyre-legoesm-process-trace-v1"),
        "case": CASE,
        "producer_commit": expected_commit,
        "steps": [PROCESS_START_STEP, PROCESS_END_STEP],
        "record_count": LEGO_PROCESS_TRACE_STEPS,
        "dt_s": DT_S, "seed": 0, "platform": "cpu",
        "precision": "fp64/libm", "production_entry": "model.step/_step_jitted",
        "shape_3d": list(shape3), "shape_2d": list(shape2),
        "fields": trace_fields,
        "carried_state_unequal_bytes_total": total_unequal_bytes,
        "carried_state_unequal_bytes_max_step": max_step_unequal_bytes,
        "effect_control": effect_control, "operands": operands,
        "wall_seconds": time.time() - started, "worktree": stamp,
    }
    if include_vertical:
        metadata.update({
            "vertical_effect_control": vertical_effect_control,
            "reference_process_trace": str(reference_process_trace),
            "reference_process_cells_unequal": reference_moved,
        })
    metadata_path = root / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    paths.extend([snapshot_root / "day180.npz",
                  snapshot_root / "day240.npz", metadata_path])
    if include_vertical:
        paths.append(snapshot_root / "day210.npz")
    file_hashes = _write_trace_manifest(root, paths, expected_commit)
    report = dict(metadata)
    report["root"] = str(root)
    report["trace_files_sha256"] = file_hashes
    kind = "process+vertical" if include_vertical else "process"
    print(f"STATUS PASS: legoESM {kind} trace {LEGO_PROCESS_TRACE_STEPS} "
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
    require(metadata["format"] in (
        "gyre-legoesm-process-trace-v1",
        "gyre-legoesm-process-vertical-trace-v2"),
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
    has_vertical = (
        metadata["format"] == "gyre-legoesm-process-vertical-trace-v2")
    expected_files = {f"{name}.npy" for name in LEGO_PROCESS_FIELDS} | {
        "day180.npz", "day240.npz", "manifest.json"}
    if has_vertical:
        expected_files |= {f"{name}.npy" for name in LEGO_VERTICAL_FIELDS}
        expected_files.add("day210.npz")
    require(set(manifest) == expected_files,
            "legoESM trace manifest file set differs from the frozen layout")
    file_paths = {name: (root / name if name not in (
                             "day180.npz", "day210.npz", "day240.npz")
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


def _load_vertical_trace_arrays(root: Path) -> dict[str, np.ndarray]:
    return {name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in LEGO_VERTICAL_FIELDS}


def _literal_vertical_matrix(K: np.ndarray, e3t: np.ndarray,
                             e3w: np.ndarray, wet: np.ndarray):
    """NEMO's active non-Aimp/non-MFC matrix association in numpy."""
    negative_zero = np.full(e3t.shape[:-1] + (1,), -0.0, dtype=np.float64)
    interface_wet = wet[..., 1:] & wet[..., :-1]
    product = -np.float64(DT_S) * K
    coeff = product / np.where(interface_wet, e3w, 1.0)
    coeff = coeff * interface_wet
    lower = np.concatenate((negative_zero, coeff), axis=-1)
    upper = np.concatenate((coeff, negative_zero), axis=-1)
    diagonal = e3t - (lower + upper)
    diagonal = np.where(wet, diagonal, 1.0)
    return lower, diagonal, upper


def validate_lego_vertical_trace(root: Path, expected_commit: str, *,
                                 plant: str | None = None,
                                 mesh_path: Path = DEFAULT_MESH) -> dict:
    """Admit the Round-126 extension of the existing production trace."""
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    require(metadata["format"] == "gyre-legoesm-process-vertical-trace-v2",
            "wrong legoESM vertical trace format")
    if plant == "lego-vertical-stamp":
        try:
            _check_trace_stamp(root, "0" * 40)
        except GateError as error:
            return {"status": "PLANT-FIRED", "plant": plant,
                    "reason": str(error)}
        raise GateError("lego-vertical-stamp plant stayed green")
    process_validation = validate_lego_process_trace(
        root, expected_commit, mesh_path=mesh_path)
    arrays = _load_vertical_trace_arrays(root)
    shape3 = tuple(metadata["shape_3d"])
    shape_if = shape3[:-1] + (shape3[-1] - 1,)
    for name, array in arrays.items():
        expected = (LEGO_PROCESS_TRACE_STEPS,) + (
            shape_if if name in (
                "heat_K", "isoneutral_K", "effective_K", "e3w_now")
            else shape3)
        require(array.shape == expected and array.dtype == np.float64,
                f"{name}: shape/dtype {array.shape}/{array.dtype}, expected "
                f"{expected}/float64")

    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    wet = gate.expected_masks(build_nemo_testcase_card(CASE))["T"]
    interface_wet = wet[..., 1:] & wet[..., :-1]
    matrix_unequal = {name: 0 for name in ("lower", "diagonal", "upper")}
    effective_unequal = 0
    solved_unequal = 0
    process_arrays = _load_lego_trace_arrays(root)
    first_expected = None
    for index in range(LEGO_PROCESS_TRACE_STEPS):
        frame = {name: np.asarray(array[index])
                 for name, array in arrays.items()}
        require(all(np.all(np.isfinite(value)) for value in frame.values()),
                f"vertical trace frame {index} contains a non-finite value")
        expected_effective = frame["heat_K"] + frame["isoneutral_K"]
        effective_unequal += _different_cells(
            expected_effective, frame["effective_K"], interface_wet)
        expected_matrix = _literal_vertical_matrix(
            frame["effective_K"], frame["e3t_after"], frame["e3w_now"], wet)
        if index == 0:
            first_expected = expected_matrix
        for name, expected in zip(
                ("lower", "diagonal", "upper"), expected_matrix,
                strict=True):
            matrix_unequal[name] += _different_cells(
                expected, frame[name], wet)
        solved_unequal += _different_cells(
            np.asarray(process_arrays["Taa"][index]), frame["solved_T"], wet)
    require(effective_unequal == 0,
            f"effective K differs from heat+isoneutral in {effective_unequal} "
            "wet interfaces")
    # This comparison is deliberately REPORTING, not admission.  It is an
    # isolated NumPy reconstruction of a matrix returned by the full
    # production JIT.  XLA may reassociate the diagonal while leaving the
    # captured lower/upper exact; Decision 41/L-amend forbid treating an
    # isolated closure as the production boundary.  Admission instead pins
    # the direct return by hash, solved_T == process Taa, and the full-step
    # one-interface K plant that reaches the returned matrix and solution.
    require(solved_unequal == 0,
            f"vertical trace solved temperature differs from process Taa in "
            f"{solved_unequal} cells")

    if plant == "lego-vertical-matrix-ulp":
        require(first_expected is not None, "matrix plant has no first frame")
        baseline = np.asarray(arrays["diagonal"][0])
        planted = np.array(baseline, copy=True)
        j, i, k = (int(x) for x in np.argwhere(wet)[0])
        planted[j, i, k] = np.nextafter(planted[j, i, k], np.inf)
        moved = _different_cells(baseline, planted, wet)
        require(moved == 1,
                f"vertical matrix ULP plant moved {moved} cells, expected 1")
        return {"status": "PLANT-FIRED", "plant": plant,
                "field": "diagonal", "index_jik": [j, i, k],
                "registered_cells_moved": moved}
    require(plant in (None, "none"),
            f"unknown lego vertical plant {plant!r}")
    effect = metadata["vertical_effect_control"]
    require(effect["status"] == "PLANT-FIRED"
            and effect["solved_temperature_cells_moved"] > 0
            and all(value == 0 for value in
                    effect["upstream_cells_moved"].values()),
            "stored vertical production effect control is vacuous")
    return {
        "format": "gyre-legoesm-vertical-trace-validation-v1",
        "status": "PASS", "case": CASE, "root": str(root),
        "producer_commit": expected_commit, "worktree": worktree_stamp(),
        "layout": {"record_count": LEGO_PROCESS_TRACE_STEPS,
                   "shape_3d": list(shape3),
                   "fields": list(LEGO_VERTICAL_FIELDS)},
        "controls": {
            "effective_K_cells_unequal": effective_unequal,
            "matrix_rebuild_cells_unequal": matrix_unequal,
            "reference_process_cells_unequal": metadata[
                "reference_process_cells_unequal"],
            "vertical_effect_control": effect,
            "process_validation": process_validation["controls"],
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


# --------------------- day-180-to-240 vertical-diffusion sub-owner budget ---
VERTICAL_OWNER_ROWS = (
    "live_column_gradient", "free_surface_weighting", "tke_heat_diffusivity",
    "isoneutral_diffusivity", "matrix_content_association",
    "implicit_solve_association", "interaction_order_residual",
)


def _literal_vertical_solve(content: np.ndarray, lower: np.ndarray,
                            diagonal: np.ndarray, upper: np.ndarray,
                            wet: np.ndarray) -> np.ndarray:
    """The three compiled ``tra_zdf`` recurrences, in written order."""
    nlev = content.shape[-1]
    lu = np.array(diagonal, copy=True)
    for k in range(1, nlev):
        lu[..., k] = (diagonal[..., k]
                      - lower[..., k] * upper[..., k - 1]
                      / lu[..., k - 1])
    work = np.array(content, copy=True)
    for k in range(1, nlev):
        work[..., k] = (content[..., k]
                        - lower[..., k] / lu[..., k - 1]
                        * work[..., k - 1])
    solved = np.array(work, copy=True)
    solved[..., -1] = work[..., -1] / lu[..., -1] * wet[..., -1]
    for k in range(nlev - 2, -1, -1):
        solved[..., k] = ((work[..., k]
                           - upper[..., k] * solved[..., k + 1])
                          / lu[..., k] * wet[..., k])
    return solved


def _vertical_increment(pre_solve: np.ndarray, K: np.ndarray,
                        e3t: np.ndarray, e3w: np.ndarray,
                        wet: np.ndarray) -> np.ndarray:
    lower, diagonal, upper = _literal_vertical_matrix(K, e3t, e3w, wet)
    content = e3t * pre_solve
    return (_literal_vertical_solve(
        content, lower, diagonal, upper, wet) - pre_solve) * wet


def score_vertical_budget(nemo_vertical_root: Path, lego_trace_root: Path,
                          expected_commit: str, *,
                          process_root: Path = DEFAULT_PROCESS_RECORD_ROOT,
                          immutable_lego_root: Path =
                          DEFAULT_IMMUTABLE_GYRE_YEAR,
                          nemo_root: Path = YEAR_ROOT,
                          mesh_path: Path = DEFAULT_MESH) -> dict:
    """Telescope the inherited day-240 vertical carry into source boundaries."""
    _policy()
    year = _year()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    nemo_validation = validate_vertical_record(
        nemo_vertical_root, "4cac617cd928007506f2de7ccb098f87e04204d1",
        process_root=process_root)
    lego_validation = validate_lego_vertical_trace(
        lego_trace_root, expected_commit, mesh_path=mesh_path)
    card = build_nemo_testcase_card(CASE)
    mesh, wet, wet2, _dz, _dy, _area, bands = year._geometry(card, mesh_path)
    nlev = wet.shape[-1]
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    lego_end = _load_npz(
        Path(lego_trace_root) / "lego_seed0/day240.npz")["T"]
    immutable_end = _load_npz(
        Path(immutable_lego_root) / "lego_seed0_year/day240.npz")["T"]
    require(_different_cells(lego_end, immutable_end, wet) == 0,
            "vertical trace day-240 temperature differs from immutable member")
    nemo_end = year._load_nemo(nemo_root, 0, 240, nlev)["T"]
    endpoint = lego_end - nemo_end
    endpoint_rms = _rms(endpoint, wet)
    require(endpoint_rms == 1.6446741930292448e-2,
            f"day-240 headline {endpoint_rms:.17e} differs from frozen value")

    arrays = _load_vertical_trace_arrays(Path(lego_trace_root))
    process_arrays = _load_lego_trace_arrays(Path(lego_trace_root))
    components = {name: np.zeros_like(endpoint) for name in VERTICAL_OWNER_ROWS}
    blocks = [{name: np.zeros_like(endpoint) for name in VERTICAL_OWNER_ROWS}
              for _ in range(6)]
    max_step_residual = 0.0
    oracle_rebuild_unequal = 0
    lego_rebuild_unequal = 0
    boundary = {
        "nemo_surface_nonzero": 0, "nemo_bottom_nonzero": 0,
        "lego_surface_nonzero": 0, "lego_bottom_nonzero": 0,
        "nemo_surface_max_abs": 0.0, "nemo_bottom_max_abs": 0.0,
        "lego_surface_max_abs": 0.0, "lego_bottom_max_abs": 0.0,
    }
    for index, step in enumerate(range(PROCESS_START_STEP,
                                       PROCESS_END_STEP + 1)):
        record = _read_vertical_record(
            Path(nemo_vertical_root)
            / f"oracle_trazdf_matrix_kt{step:08d}.bin", step)
        e3t_n = _vertical_field(record, "e3t_Kaa", nlev)
        e3w_n = _vertical_field(record, "e3w_Kmm")[:, :, 1:nlev]
        heat_n = _vertical_field(record, "avt")[:, :, 1:nlev]
        iso_n = _vertical_field(record, "ah_wslp2")[:, :, 1:nlev]
        rhs_n = _vertical_field(record, "rhs_T", nlev)
        pre_n = rhs_n / e3t_n
        sol_n = _vertical_field(record, "sol_T_pre_clamp", nlev)
        vertical_n = (sol_n - pre_n) * wet
        matrix_n = tuple(_vertical_field(record, name) for name in
                         ("zwi", "zwd", "zws"))
        rebuilt_n = _literal_vertical_solve(
            rhs_n, *matrix_n, wet) * wet
        oracle_rebuild_unequal += _different_cells(sol_n, rebuilt_n, wet)

        frame = {name: np.asarray(array[index])
                 for name, array in arrays.items()}
        pre_l = np.divide(
            frame["content_T"], frame["e3t_after"],
            out=np.zeros_like(frame["content_T"]),
            where=frame["e3t_after"] != 0.0)
        vertical_l = (frame["solved_T"] - pre_l) * wet

        h0 = _vertical_increment(
            pre_n, heat_n + iso_n, e3t_n, e3w_n, wet)
        h1 = _vertical_increment(
            pre_l, heat_n + iso_n, e3t_n, e3w_n, wet)
        h2 = _vertical_increment(
            pre_l, heat_n + iso_n,
            frame["e3t_after"], frame["e3w_now"], wet)
        h3 = _vertical_increment(
            pre_l, frame["heat_K"] + iso_n,
            frame["e3t_after"], frame["e3w_now"], wet)
        h4 = _vertical_increment(
            pre_l, frame["effective_K"],
            frame["e3t_after"], frame["e3w_now"], wet)
        h5 = (_literal_vertical_solve(
            frame["content_T"], frame["lower"], frame["diagonal"],
            frame["upper"], wet) - pre_l) * wet
        lego_rebuild_unequal += _different_cells(
            h5 + pre_l, frame["solved_T"], wet)
        h6 = vertical_l
        rows = {
            "live_column_gradient": h1 - h0,
            "free_surface_weighting": h2 - h1,
            "tke_heat_diffusivity": h3 - h2,
            "isoneutral_diffusivity": h4 - h3,
            "matrix_content_association": h5 - h4,
            "implicit_solve_association": h6 - h5,
        }
        subtotal = np.zeros_like(endpoint)
        for value in rows.values():
            subtotal += value
        rows["interaction_order_residual"] = (
            (vertical_l - vertical_n) - subtotal)
        reconstruction = np.zeros_like(endpoint)
        for name in VERTICAL_OWNER_ROWS:
            components[name] += rows[name]
            blocks[index // 60][name] += rows[name]
            reconstruction += rows[name]
        residual = (vertical_l - vertical_n) - reconstruction
        max_step_residual = max(
            max_step_residual, float(np.max(np.abs(residual[wet]))))

        zwt_n = _vertical_field(record, "zwt_mix")
        boundary["nemo_surface_nonzero"] += int(np.count_nonzero(
            zwt_n[..., 0][wet2]))
        boundary["nemo_bottom_nonzero"] += int(np.count_nonzero(
            zwt_n[..., -1][wet2]))
        boundary["lego_surface_nonzero"] += int(np.count_nonzero(
            frame["lower"][..., 0][wet2]))
        boundary["lego_bottom_nonzero"] += int(np.count_nonzero(
            frame["upper"][..., -1][wet2]))
        for key, values in (
                ("nemo_surface_max_abs", zwt_n[..., 0]),
                ("nemo_bottom_max_abs", zwt_n[..., -1]),
                ("lego_surface_max_abs", frame["lower"][..., 0]),
                ("lego_bottom_max_abs", frame["upper"][..., -1])):
            boundary[key] = max(
                boundary[key], float(np.max(np.abs(values[wet2]))))

    require(oracle_rebuild_unequal == 0,
            f"offline source-order solve differs from NEMO in "
            f"{oracle_rebuild_unequal} wet cells")
    total_component = np.zeros_like(endpoint)
    for value in components.values():
        total_component += value
    vertical_carry, _ = _projection(total_component, endpoint, wet, endpoint_rms)
    inherited_carry = 2.4168271578053416e-2
    require(abs(vertical_carry - inherited_carry) <= 2.0e-15,
            f"sub-owner carry {vertical_carry:.17e} does not close inherited "
            f"{inherited_carry:.17e}")

    regions = _regions(lat, wet2)
    horizontal_masks = {name: wet & regions[name][..., None]
                        for name in ("west_third", "interior_third",
                                     "east_third")}
    latitude_masks = {
        f"south_le_{EMP_SPLIT_LAT_DEG}N": wet & (
            lat <= EMP_SPLIT_LAT_DEG)[..., None],
        f"north_gt_{EMP_SPLIT_LAT_DEG}N": wet & (
            lat > EMP_SPLIT_LAT_DEG)[..., None],
    }

    def strongest(values, partitions):
        scored = []
        for name, local_mask in partitions.items():
            additive = float(np.sum(values[local_mask] * endpoint[local_mask]))
            carry, fraction = _projection(
                values, endpoint, local_mask, _rms(endpoint, local_mask))
            scored.append((abs(additive), name, carry, fraction,
                           int(local_mask.sum())))
        _, name, carry, fraction, cells = max(scored)
        return {"name": name, "local_signed_carry_K": carry,
                "local_projection_fraction": fraction, "cells": cells}

    ranking = []
    for name, values in components.items():
        carry, fraction = _projection(values, endpoint, wet, endpoint_rms)
        block_rows = []
        for block, block_values in enumerate(blocks):
            block_carry, _ = _projection(
                block_values[name], endpoint, wet, endpoint_rms)
            block_rows.append((abs(block_carry), block, block_carry))
        _, block, block_carry = max(block_rows)
        ranking.append({
            "owner": name, "signed_carry_K": carry,
            "abs_signed_carry_K": abs(carry),
            "projection_fraction": fraction,
            "component_rms_K": _rms(values, wet),
            "birth": {"interval_days": [180 + 10 * block,
                                          190 + 10 * block],
                      "signed_carry_K": block_carry},
            "largest_depth_partition": strongest(values, bands),
            "largest_longitude_partition": strongest(
                values, horizontal_masks),
            "largest_latitude_partition": strongest(values, latitude_masks),
        })
    ranking.sort(key=lambda row: row["abs_signed_carry_K"], reverse=True)
    signed_sum = float(sum(row["signed_carry_K"] for row in ranking))
    discriminator = conditional_vertical_closure(
        nemo_vertical_root, lego_trace_root, mesh_path=mesh_path)
    report = {
        "format": "gyre-day240-vertical-subowner-budget-v1",
        "status": "PASS", "case": CASE,
        "interval_days": [180, 240],
        "nemo_vertical_root": str(nemo_vertical_root),
        "lego_trace_root": str(lego_trace_root),
        "headline": {"day240_T3D_rms_K": endpoint_rms,
                     "inherited_vertical_carry_K": inherited_carry},
        "controls": {
            "nemo_offline_solve_cells_unequal": oracle_rebuild_unequal,
            "lego_offline_vs_production_solve_cells_unequal": (
                lego_rebuild_unequal),
            "max_abs_step_telescope_residual_K": max_step_residual,
            "signed_carry_sum_K": signed_sum,
            "signed_carry_minus_inherited_K": signed_sum - inherited_carry,
            "surface_bottom": boundary,
        },
        "ranking": ranking, "largest_owner": ranking[0]["owner"],
        "closure_vs_upstream": discriminator,
        "nemo_validation": nemo_validation,
        "lego_validation": lego_validation,
        "worktree": worktree_stamp(),
    }
    print("\nDAY-240 VERTICAL SUB-OWNER RANKING -- independent trajectories")
    print(f"  {'rank':>4s} {'owner':>30s} {'signed carry K':>16s} "
          f"{'component rms K':>16s} {'birth days':>14s}")
    for rank, row in enumerate(ranking, 1):
        print(f"  {rank:4d} {row['owner']:>30s} "
              f"{row['signed_carry_K']:16.8e} "
              f"{row['component_rms_K']:16.8e} "
              f"{str(row['birth']['interval_days']):>14s}")
    print(f"  inherited carry {inherited_carry:.17e} K; signed sub-owner "
          f"sum {signed_sum:.17e} K; max step residual "
          f"{max_step_residual:.3e} K")
    return report


def _load_kamm_twin_module():
    """Load the existing TKE restart bridge; do not duplicate its mapping."""
    name = "_round126_kamm_twin"
    if name in sys.modules:
        return sys.modules[name]
    path = _HERE.parent / "dino_1226" / "kamm_twin_90d.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _closed_barotropic_histories_to_faces(
        uu_b: np.ndarray, vv_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map NEMO's 2-D closed-basin ``uu_n/vv_n`` onto model faces."""
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)

    uu_b = np.asarray(uu_b)
    vv_b = np.asarray(vv_b)
    require(uu_b.ndim == 2 and vv_b.ndim == 2,
            "barotropic restart histories must both be two-dimensional")
    return (_u_east_to_face(uu_b[..., None])[..., 0],
            _v_north_to_face(vv_b[..., None])[..., 0])


def _gyre_checkpoint_state(card, restart_path: Path):
    """Map a GYRE restart onto the certified card without rebuilding geometry."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        neumann_fill_cgrid)
    from legoesm.ocean.fidelity.nemo_io import (
        read_nemo_restart, read_nemo_restart_en,
        read_nemo_restart_tke_coefficients)
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)

    restart = read_nemo_restart(str(restart_path), nn_hls=0)
    nlev = card.recipe.initial_state.T.data.shape[-1]
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    T = neumann_fill_cgrid(
        jnp.asarray(restart.T[..., :nlev]), jnp.asarray(active),
        card.recipe.grid)
    S = neumann_fill_cgrid(
        jnp.asarray(restart.S[..., :nlev]), jnp.asarray(active),
        card.recipe.grid)
    u = _u_east_to_face(restart.u[..., :nlev])
    v = _v_north_to_face(restart.v[..., :nlev])
    state = card.recipe.initial_state
    updates = {
        "T": state.T.replace(data=T), "S": state.S.replace(data=S),
        "u": state.u.replace(data=jnp.asarray(u)),
        "v": state.v.replace(data=jnp.asarray(v)),
        "eta": state.eta.replace(data=jnp.asarray(restart.ssh)),
    }
    if state.uu_b is not None and restart.uu_b is not None:
        uu_b, vv_b = _closed_barotropic_histories_to_faces(
            restart.uu_b, restart.vv_b)
        updates["uu_b"] = state.uu_b.replace(
            data=jnp.asarray(uu_b))
        updates["vv_b"] = state.vv_b.replace(
            data=jnp.asarray(vv_b))
    state = state._replace(**updates)
    en = read_nemo_restart_en(str(restart_path), nn_hls=0)[..., :nlev]
    avm, avt, dissl = read_nemo_restart_tke_coefficients(
        str(restart_path), nn_hls=0)
    state = _load_kamm_twin_module().bridge_tke_from_restart(
        state, en, np.asarray(state.land_mask.data),
        restart_avm=avm[..., :nlev], restart_avt=avt[..., :nlev],
        restart_dissl=dissl[..., :nlev])
    return state, restart


def _upper_100m_jet_centroid(u_east: np.ndarray, v_north: np.ndarray,
                             depths: np.ndarray, thickness: np.ndarray,
                             west: np.ndarray, lat: np.ndarray,
                             lon: np.ndarray) -> dict:
    """Fastest-decile western upper-ocean speed centroid on T points."""
    u_east = np.asarray(u_east, dtype=np.float64)
    v_north = np.asarray(v_north, dtype=np.float64)
    west_u = np.concatenate(
        [np.zeros_like(u_east[:, :1]), u_east[:, :-1]], axis=1)
    south_v = np.concatenate(
        [np.zeros_like(v_north[:1, :]), v_north[:-1, :]], axis=0)
    u_t = 0.5 * (west_u + u_east)
    v_t = 0.5 * (south_v + v_north)
    level_depth = np.mean(depths, axis=(0, 1)) if depths.ndim == 3 else depths
    upper = level_depth <= 100.0
    require(bool(np.any(upper)), "upper-100-m jet proxy selected no levels")
    weights = np.asarray(thickness, dtype=np.float64)[upper]
    speed3 = np.sqrt(u_t[..., upper] ** 2 + v_t[..., upper] ** 2)
    speed = np.sum(speed3 * weights, axis=-1) / np.sum(weights)
    values = speed[west]
    threshold = float(np.quantile(values, 0.9))
    selected = west & (speed >= threshold) & (speed > 0.0)
    require(bool(np.any(selected)), "jet centroid selected no cells")
    jj, ii = np.indices(speed.shape)
    w = speed[selected]
    denom = float(np.sum(w))
    return {
        "centroid_j": float(np.sum(jj[selected] * w) / denom),
        "centroid_i": float(np.sum(ii[selected] * w) / denom),
        "centroid_lat_deg": float(np.sum(lat[selected] * w) / denom),
        "centroid_lon_deg": float(np.sum(lon[selected] * w) / denom),
        "fastest_decile_threshold_m_s": threshold,
        "selected_cells": int(np.count_nonzero(selected)),
    }


def conditional_vertical_closure(nemo_vertical_root: Path,
                                 lego_trace_root: Path, *,
                                 mesh_path: Path = DEFAULT_MESH) -> dict:
    """Run the production closure on NEMO entries at days 180 and 210."""
    _policy()
    import jax
    import netCDF4
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    card = build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    gate = _gate()
    wet = gate.expected_masks(card)["T"]
    interface_wet = wet[..., :-1] & wet[..., 1:]
    year = _year()
    mesh = year.nemo_operands(mesh_path)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    with netCDF4.Dataset(mesh_path) as handle:
        lon = np.asarray(handle.variables["glamt"][0], dtype=np.float64)
    west = _regions(lat, np.any(wet, axis=-1))["west_third"]
    depths = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    thickness = np.asarray(card.recipe.z_coord.dz_ref, dtype=np.float64)
    lego_arrays = _load_vertical_trace_arrays(Path(lego_trace_root))

    @jax.jit
    def production_closure(entry, forcing):
        return model.diagnose_vertical_K(entry, card.dt_s, forcing)

    evd = card.recipe.model_config.physics.convection.enhanced_diffusion
    require(evd is not None, "GYRE closure discriminator requires EVD")
    evd_replacement = float(evd.K_conv)

    checkpoints = []
    for day, restart_step, step in ((180, 1080, 1081),
                                    (210, 1260, 1261)):
        restart_path = (Path(nemo_vertical_root)
                        / f"GYRE_OMIP_L2_P3_{restart_step:08d}_restart.nc")
        entry, restart = _gyre_checkpoint_state(card, restart_path)
        freshwater, surface = gate._surface_forcings(card, entry, step)
        conditional_heat, _ = production_closure(entry, surface)
        conditional_heat = np.asarray(conditional_heat, dtype=np.float64)
        record = _read_vertical_record(
            Path(nemo_vertical_root)
            / f"oracle_trazdf_matrix_kt{step:08d}.bin", step)
        nemo_heat = _vertical_field(record, "avt")[:, :, 1:wet.shape[-1]]
        own_heat = np.asarray(
            lego_arrays["heat_K"][step - PROCESS_START_STEP])

        def error(candidate):
            difference = candidate[interface_wet] - nemo_heat[interface_wet]
            return {"rms_m2_s": float(np.sqrt(np.mean(difference ** 2))),
                    "max_abs_m2_s": float(np.max(np.abs(difference))),
                    "cells_unequal": int(np.count_nonzero(
                        candidate[interface_wet].view(np.uint64)
                        != nemo_heat[interface_wet].view(np.uint64)))}

        own_error = error(own_heat)
        conditional_error = error(conditional_heat)
        reduction = (1.0 - conditional_error["rms_m2_s"]
                     / own_error["rms_m2_s"])

        def evd_mask(candidate):
            return candidate[interface_wet] == evd_replacement

        nemo_evd = evd_mask(nemo_heat)
        own_evd = evd_mask(own_heat)
        conditional_evd = evd_mask(conditional_heat)
        record_t = _vertical_field(record, "T_Kbb_in", wet.shape[-1])
        entry_t = np.asarray(entry.T.data)
        require(_different_cells(record_t, entry_t, wet) == 0,
                f"day {day}: bridged checkpoint T differs from record entry")

        lego_snapshot = _load_npz(
            Path(lego_trace_root) / f"lego_seed0/day{day}.npz")
        jet_lego = _upper_100m_jet_centroid(
            lego_snapshot["u"], lego_snapshot["v"], depths, thickness,
            west, lat, lon)
        jet_nemo = _upper_100m_jet_centroid(
            restart.u[..., :wet.shape[-1]],
            restart.v[..., :wet.shape[-1]], depths, thickness,
            west, lat, lon)
        displacement = {
            "delta_j_cells": jet_lego["centroid_j"] - jet_nemo["centroid_j"],
            "delta_i_cells": jet_lego["centroid_i"] - jet_nemo["centroid_i"],
        }
        displacement["magnitude_cells"] = float(np.hypot(
            displacement["delta_j_cells"], displacement["delta_i_cells"]))
        checkpoints.append({
            "day": day, "entry_step": step,
            "model_own_trajectory_avt_error": own_error,
            "model_on_nemo_entry_avt_error": conditional_error,
            "rms_error_reduction_fraction": reduction,
            "evd_replacement_value_m2_s": evd_replacement,
            "evd_active_cells": {
                "nemo": int(np.count_nonzero(nemo_evd)),
                "model_own_trajectory": int(np.count_nonzero(own_evd)),
                "model_on_nemo_entry": int(np.count_nonzero(
                    conditional_evd)),
            },
            "evd_trigger_disagreement_cells": {
                "model_own_trajectory": int(np.count_nonzero(
                    own_evd != nemo_evd)),
                "model_on_nemo_entry": int(np.count_nonzero(
                    conditional_evd != nemo_evd)),
            },
            "jet_centroid_model": jet_lego,
            "jet_centroid_nemo": jet_nemo,
            "jet_centroid_displacement": displacement,
        })

    reductions = [row["rms_error_reduction_fraction"] for row in checkpoints]
    vectors = [(
        row["jet_centroid_displacement"]["delta_j_cells"],
        row["jet_centroid_displacement"]["delta_i_cells"])
        for row in checkpoints]
    consistent_position = (
        all(float(np.hypot(*vector)) > 1.0e-6 for vector in vectors)
        and float(np.dot(vectors[0], vectors[1])) > 0.0)
    if all(value >= 0.90 for value in reductions):
        verdict = ("UPSTREAM-JET-POSITION-RESPONSE" if consistent_position
                   else "UPSTREAM-STATE-RESPONSE")
    elif any(value < 0.50 for value in reductions):
        verdict = "TKE-CLOSURE-OWNER"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "status": "PASS", "execution": "production-step jax.jit",
        "checkpoints": checkpoints,
        "jet_displacement_consistent": consistent_position,
        "verdict": verdict,
    }


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


# -------------------------------------- Round-134 daily reset attribution ---
def _direct_snapshot(member: Path, day: int) -> dict[str, np.ndarray]:
    path = Path(member) / f"day{day:03d}.npz"
    require(path.is_file(), f"missing reset-arm snapshot {path}")
    with np.load(path) as handle:
        arrays = {name: np.asarray(handle[name], dtype=np.float64)
                  for name in FIELDS}
    require(all(np.all(np.isfinite(values)) for values in arrays.values()),
            f"{path}: non-finite scored field")
    return arrays


def _bit_unequal(left: np.ndarray, right: np.ndarray) -> int:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape,
            f"bit comparison shapes differ {left.shape} != {right.shape}")
    return int(np.count_nonzero(left.view(np.uint64) != right.view(np.uint64)))


def _daily_arm_manifest(member: Path, family: str, expect_commit: str, *,
                        interval_days: int = 1,
                        expected_variables: tuple[str, ...] | None = None) -> dict:
    path = Path(member) / "manifest.json"
    require(path.is_file(), f"missing reset-arm manifest {path}")
    manifest = json.loads(path.read_text())
    stamp = manifest.get("worktree", {})
    require(stamp.get("commit") == expect_commit and stamp.get("clean") is True,
            f"{path}: arm is not a clean {expect_commit} product")
    require(manifest.get("steps") == 2160
            and manifest.get("snapshot_step_interval") == STEPS_PER_DAY,
            f"{path}: arm is not the registered 2160-step daily run")
    reset = manifest.get("daily_reset", {})
    require(reset.get("family") == family,
            f"{path}: reset family {reset.get('family')!r} != {family!r}")
    observed_interval = int(reset.get("interval_days", 1))
    require(observed_interval == interval_days,
            f"{path}: interval {observed_interval} != {interval_days} days")
    if expected_variables is not None:
        require(tuple(reset.get("registered_variables", ()))
                == tuple(expected_variables),
                f"{path}: registered variables are not exactly "
                f"{tuple(expected_variables)}")
    expected_days = (() if family == "control" else
                     tuple(range(interval_days, 360, interval_days)))
    expected_count = len(expected_days)
    require(reset.get("applied_boundary_count") == expected_count,
            f"{path}: applied {reset.get('applied_boundary_count')} boundaries, "
            f"expected {expected_count}")
    observed_days = tuple(int(row["day"])
                          for row in reset.get("applied_boundaries", ()))
    require(observed_days == expected_days,
            f"{path}: reset boundary days do not equal {expected_days}")
    return manifest


def _daily_metric_rows(year, members: dict[str, Path], specs: dict[str, dict],
                       free_member: Path, nemo_daily_root: Path,
                       masks: dict[str, np.ndarray], nlev: int) -> tuple[list, dict]:
    """Score registered reset arms with one shared RMS implementation."""
    rows = []
    baseline = {}
    first_label = next(iter(members))
    for day in DAILY_ATTRIBUTION_DAYS:
        free = _direct_snapshot(free_member, day)
        nemo = year._load_nemo(
            Path("."), 0, day, nlev, directory=nemo_daily_root)
        for label, member in members.items():
            reset = _direct_snapshot(member, day)
            for name in FIELDS:
                mask = masks[name]
                free_gap = _rms(free[name] - nemo[name], mask)
                reset_gap = _rms(reset[name] - nemo[name], mask)
                rows.append({
                    "family": label,
                    "reset_family": specs[label]["family"],
                    "interval_days": specs[label]["interval_days"],
                    "day": day,
                    "field": name,
                    "free_vs_nemo_rms": free_gap,
                    "reset_vs_nemo_rms": reset_gap,
                    "reset_vs_free_rms": _rms(
                        reset[name] - free[name], mask),
                    "removed_gap_rms": free_gap - reset_gap,
                })
                if label == first_label:
                    baseline[(day, name)] = free_gap
    return rows, baseline


def _validate_daily_metric_registry(rows: list[dict], labels) -> None:
    expected = {
        (label, day, field)
        for label in labels
        for day in DAILY_ATTRIBUTION_DAYS
        for field in FIELDS
    }
    observed = [(row["family"], row["day"], row["field"]) for row in rows]
    require(len(observed) == len(expected),
            f"daily metric registry has {len(observed)} rows, "
            f"expected {len(expected)}")
    require(len(set(observed)) == len(observed),
            "daily metric registry contains duplicate keys")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    require(not missing and not extra,
            f"daily metric registry mismatch missing={missing[:1]} "
            f"extra={extra[:1]}")


def _daily_birth_report(year, reset_member: Path, free_member: Path,
                        nemo_daily_root: Path, card, wet3: np.ndarray,
                        wet2: np.ndarray, bands: dict[str, np.ndarray],
                        nlev: int) -> dict:
    """Locate the first reset response and first west/upper gap reduction."""
    first_difference = None
    for day in range(1, 361):
        free = _direct_snapshot(free_member, day)
        reset = _direct_snapshot(reset_member, day)
        unequal = _bit_unequal(reset["T"], free["T"])
        if unequal:
            first_difference = (day, reset["T"] - free["T"], unequal)
            break
    require(first_difference is not None,
            "winning reset arm never moves temperature")
    birth_day, birth_delta, birth_unequal = first_difference
    total = float(np.sum(birth_delta[wet3] ** 2))
    require(total > 0.0, "winning arm's first differing T row has zero energy")
    depth_birth = {
        name: {
            "rms_K": _rms(birth_delta, mask),
            "share_of_sum_dT2": float(np.sum(birth_delta[mask] ** 2)) / total,
            "cells": int(mask.sum()),
        } for name, mask in bands.items()
    }
    lat = np.asarray(card.recipe.grid.native_lat_T_deg, dtype=np.float64)
    regions = _regions(lat, wet2)
    region_birth = {}
    for name, mask2 in regions.items():
        mask = wet3 & mask2[..., None]
        region_birth[name] = {
            "rms_K": _rms(birth_delta, mask),
            "share_of_sum_dT2": float(np.sum(birth_delta[mask] ** 2)) / total,
            "cells": int(mask.sum()),
        }
    peak = np.unravel_index(int(np.argmax(np.abs(np.where(
        wet3, birth_delta, 0.0)))), birth_delta.shape)
    strongest_depth = max(
        depth_birth, key=lambda name: depth_birth[name]["share_of_sum_dT2"])
    third_names = ("west_third", "interior_third", "east_third")
    strongest_third = max(
        third_names,
        key=lambda name: region_birth[name]["share_of_sum_dT2"])

    west_upper = bands["0_100"] & regions["west_third"][..., None]
    first_prevented_growth = None
    for day in range(1, 361):
        free = _direct_snapshot(free_member, day)
        reset = _direct_snapshot(reset_member, day)
        nemo = year._load_nemo(
            Path("."), 0, day, nlev, directory=nemo_daily_root)
        free_gap = _rms(free["T"] - nemo["T"], west_upper)
        reset_gap = _rms(reset["T"] - nemo["T"], west_upper)
        if reset_gap < free_gap:
            first_prevented_growth = {
                "day": day,
                "free_vs_nemo_T_rms_K": free_gap,
                "reset_vs_nemo_T_rms_K": reset_gap,
                "removed_T_rms_K": free_gap - reset_gap,
                "region": "west_third",
                "depth": "0_100",
            }
            break
    require(first_prevented_growth is not None,
            "winning reset never reduces western-upper-100-m T RMS")
    return {
        "day": birth_day,
        "T_cells_unequal": birth_unequal,
        "T_rms_K": _rms(birth_delta, wet3),
        "depth": depth_birth,
        "region": region_birth,
        "strongest_depth": strongest_depth,
        "strongest_third": strongest_third,
        "peak": {"j": int(peak[0]), "i": int(peak[1]),
                 "k": int(peak[2]), "dT_K": float(birth_delta[peak])},
        "first_west_upper_gap_reduction": first_prevented_growth,
    }


def daily_reset_attribution(
        arms_root: Path, free_member: Path, nemo_daily_root: Path,
        expect_commit: str, *, mesh_path: Path = DEFAULT_MESH,
        plant: str | None = None) -> dict:
    """Rank daily oracle reset leverage at the day-240 T3D endpoint."""
    _policy()
    year = _year()
    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["commit"] == expect_commit and stamp["clean"] is True,
            f"scorer is not a clean {expect_commit} product")
    card = build_nemo_testcase_card(CASE)
    _mesh, wet3, wet2, _dz, _dy, _area, bands = year._geometry(
        card, mesh_path)
    masks = gate.expected_masks(card)
    masks = {"T": wet3, "S": wet3, "ssh": wet2,
             "u": masks["u"], "v": masks["v"]}
    nlev = wet3.shape[-1]

    members = {
        family: Path(arms_root) / family / "lego_seed0_year"
        for family in ("control", *DAILY_ATTRIBUTION_FAMILIES)
    }
    manifests = {
        family: _daily_arm_manifest(
            member, family, expect_commit,
            expected_variables=tuple(year.DAILY_RESET_VARIABLES[family]))
        for family, member in members.items()
    }

    control_unequal = {name: 0 for name in FIELDS}
    first_control_mismatch = None
    for day in range(1, 361):
        control = _direct_snapshot(members["control"], day)
        free = _direct_snapshot(free_member, day)
        for name in FIELDS:
            unequal = _bit_unequal(control[name], free[name])
            control_unequal[name] += unequal
            if unequal and first_control_mismatch is None:
                first_control_mismatch = {
                    "day": day, "field": name, "cells_unequal": unequal}
    require(first_control_mismatch is None,
            f"daily control does not reproduce immutable free arm: "
            f"{first_control_mismatch}")

    day1 = _direct_snapshot(members["control"], 1)
    day1_unequal = {}
    for family in DAILY_ATTRIBUTION_FAMILIES:
        candidate = _direct_snapshot(members[family], 1)
        day1_unequal[family] = {
            name: _bit_unequal(candidate[name], day1[name]) for name in FIELDS}
        require(all(value == 0 for value in day1_unequal[family].values()),
                f"{family}: reset moved the pre-reset day-1 snapshot")

    specs = {
        family: {"family": family, "interval_days": 1}
        for family in DAILY_ATTRIBUTION_FAMILIES
    }
    rows, baseline = _daily_metric_rows(
        year,
        {family: members[family] for family in DAILY_ATTRIBUTION_FAMILIES},
        specs, free_member, nemo_daily_root, masks, nlev)

    expected_rows = (len(DAILY_ATTRIBUTION_FAMILIES)
                     * len(DAILY_ATTRIBUTION_DAYS) * len(FIELDS))
    if plant == "daily-attribution-registry":
        rows.pop()
    if len(rows) != expected_rows:
        if plant == "daily-attribution-registry":
            return {
                "format": "gyre-daily-reset-attribution-v1",
                "status": "PLANT-FIRED", "plant": plant,
                "control": {"registered_rows": len(rows),
                            "required_rows": expected_rows},
                "worktree": stamp,
            }
        raise GateError(
            f"attribution registry has {len(rows)} rows, expected {expected_rows}")
    require(plant in (None, "none"), f"plant {plant!r} did not fire")

    for day, expected in DAILY_ATTRIBUTION_BASELINE_T.items():
        require(baseline[(day, "T")] == expected,
                f"immutable day-{day} T baseline {baseline[(day, 'T')]:.17e} "
                f"!= preregistered {expected:.17e}")

    day240 = [row for row in rows
              if row["day"] == 240 and row["field"] == "T"]
    ranking = sorted(({
        "family": row["family"],
        "day240_free_vs_nemo_T_rms_K": row["free_vs_nemo_rms"],
        "day240_reset_vs_nemo_T_rms_K": row["reset_vs_nemo_rms"],
        "day240_removed_T_rms_K": row["removed_gap_rms"],
    } for row in day240), key=lambda row: row["day240_removed_T_rms_K"],
        reverse=True)
    winner = ranking[0]["family"]

    birth = _daily_birth_report(
        year, members[winner], members["control"], nemo_daily_root,
        card, wet3, wet2, bands, nlev)
    birth_day = birth["day"]
    strongest_depth = birth["strongest_depth"]
    strongest_third = birth["strongest_third"]

    report = {
        "format": "gyre-daily-reset-attribution-v1", "status": "PASS",
        "case": CASE, "execution": "production step through self._step_jitted",
        "days": list(DAILY_ATTRIBUTION_DAYS), "fields": list(FIELDS),
        "arms_root": str(arms_root), "free_member": str(free_member),
        "nemo_daily_root": str(nemo_daily_root),
        "control_reproduction": {
            "days_checked": 360, "fields": list(FIELDS),
            "cells_unequal": control_unequal, "all_bit_identical": True,
        },
        "pre_reset_day1_cells_unequal": day1_unequal,
        "manifests": {family: {
            "path": str(members[family] / "manifest.json"),
            "sha256": _sha256(members[family] / "manifest.json"),
            "reset": manifests[family]["daily_reset"],
        } for family in members},
        "rows": rows, "registered_row_count": len(rows),
        "all_moved_rows_registered": len(rows) == expected_rows,
        "ranking": ranking, "winner": winner,
        "birth": birth,
        "frozen_prediction": {
            "predicted_winner": "vector", "actual_winner": winner,
            "winner_confirmed": winner == "vector",
            "predicted_first_day": 2, "actual_first_day": birth_day,
            "first_day_confirmed": birth_day == 2,
            "predicted_depth": "0_100", "actual_depth": strongest_depth,
            "depth_confirmed": strongest_depth == "0_100",
            "predicted_third": "west_third", "actual_third": strongest_third,
            "third_confirmed": strongest_third == "west_third",
        },
        "three_step_cadence": "UNMEASURED: the oracle record is daily only",
        "worktree": stamp,
    }
    print("\nDAILY RESET DAY-240 TEMPERATURE RANKING")
    print(f"  {'rank':>4s} {'family':>10s} {'free gap':>14s} "
          f"{'reset gap':>14s} {'removed':>14s}")
    for rank, row in enumerate(ranking, 1):
        print(f"  {rank:4d} {row['family']:>10s} "
              f"{row['day240_free_vs_nemo_T_rms_K']:14.6e} "
              f"{row['day240_reset_vs_nemo_T_rms_K']:14.6e} "
              f"{row['day240_removed_T_rms_K']:14.6e}")
    print(f"  winner {winner}; first T difference day {birth_day}; "
          f"{strongest_third}/{strongest_depth}")
    return report


# ------------------------------- Round-135 tracer-family discriminator ---
def daily_tracer_subfamily_attribution(
        arms_root: Path, free_member: Path, nemo_daily_root: Path,
        expect_commit: str, *, mesh_path: Path = DEFAULT_MESH,
        include_cadence: bool = False, plant: str | None = None) -> dict:
    """Split daily tracer leverage into T/S and score the winner's cadence."""
    _policy()
    year = _year()
    gate = _gate()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["commit"] == expect_commit and stamp["clean"] is True,
            f"scorer is not a clean {expect_commit} product")
    card = build_nemo_testcase_card(CASE)
    _mesh, wet3, wet2, _dz, _dy, _area, bands = year._geometry(
        card, mesh_path)
    gate_masks = gate.expected_masks(card)
    masks = {"T": wet3, "S": wet3, "ssh": wet2,
             "u": gate_masks["u"], "v": gate_masks["v"]}
    nlev = wet3.shape[-1]

    control_member = Path(arms_root) / "control" / "lego_seed0_year"
    control_manifest = _daily_arm_manifest(
        control_member, "control", expect_commit,
        expected_variables=tuple(year.DAILY_RESET_VARIABLES["control"]))
    control_unequal = {name: 0 for name in FIELDS}
    first_control_mismatch = None
    for day in range(1, 361):
        control = _direct_snapshot(control_member, day)
        free = _direct_snapshot(free_member, day)
        for name in FIELDS:
            unequal = _bit_unequal(control[name], free[name])
            control_unequal[name] += unequal
            if unequal and first_control_mismatch is None:
                first_control_mismatch = {
                    "day": day, "field": name, "cells_unequal": unequal}
    require(first_control_mismatch is None,
            f"Round-135 control does not reproduce immutable free arm: "
            f"{first_control_mismatch}")

    members = {
        family: Path(arms_root) / family / "lego_seed0_year"
        for family in DAILY_TRACER_SUBFAMILIES
    }
    specs = {
        family: {"family": family, "interval_days": 1}
        for family in DAILY_TRACER_SUBFAMILIES
    }
    manifests = {
        family: _daily_arm_manifest(
            member, family, expect_commit,
            expected_variables=tuple(year.DAILY_RESET_VARIABLES[family]))
        for family, member in members.items()
    }

    day1 = _direct_snapshot(control_member, 1)
    day1_unequal = {}
    for family, member in members.items():
        candidate = _direct_snapshot(member, 1)
        day1_unequal[family] = {
            name: _bit_unequal(candidate[name], day1[name]) for name in FIELDS}
        require(all(value == 0 for value in day1_unequal[family].values()),
                f"{family}: reset moved the pre-reset day-1 snapshot")

    rows, baseline = _daily_metric_rows(
        year, members, specs, free_member, nemo_daily_root, masks, nlev)
    if plant == "daily-subfamily-registry":
        rows.pop()
    try:
        _validate_daily_metric_registry(rows, DAILY_TRACER_SUBFAMILIES)
    except GateError as error:
        if plant == "daily-subfamily-registry":
            return {
                "format": "gyre-daily-tracer-subfamily-v1",
                "status": "PLANT-FIRED", "plant": plant,
                "control": {"error": str(error),
                            "registered_rows": len(rows),
                            "required_rows": 80},
                "worktree": stamp,
            }
        raise
    if plant == "daily-subfamily-registry":
        raise GateError("daily-subfamily-registry plant did not fire")

    for day, expected in DAILY_ATTRIBUTION_BASELINE_T.items():
        require(baseline[(day, "T")] == expected,
                f"immutable day-{day} T baseline {baseline[(day, 'T')]:.17e} "
                f"!= preregistered {expected:.17e}")

    day240 = [row for row in rows
              if row["day"] == 240 and row["field"] == "T"]
    ranking = sorted(({
        "family": row["family"],
        "day240_free_vs_nemo_T_rms_K": row["free_vs_nemo_rms"],
        "day240_reset_vs_nemo_T_rms_K": row["reset_vs_nemo_rms"],
        "day240_removed_T_rms_K": row["removed_gap_rms"],
    } for row in day240), key=lambda row: row["day240_removed_T_rms_K"],
        reverse=True)
    winner = ranking[0]["family"]
    removed = {row["family"]: row["day240_removed_T_rms_K"]
               for row in ranking}
    birth = _daily_birth_report(
        year, members[winner], control_member, nemo_daily_root,
        card, wet3, wet2, bands, nlev)
    prediction = {
        "predicted_winner": "temperature",
        "actual_winner": winner,
        "winner_confirmed": winner == "temperature",
        "temperature_removed_min_K": 1.55e-2,
        "temperature_removed_K": removed["temperature"],
        "temperature_bound_confirmed": removed["temperature"] >= 1.55e-2,
        "salinity_removed_max_K": 1.0e-3,
        "salinity_removed_K": removed["salinity"],
        "salinity_bound_confirmed": removed["salinity"] < 1.0e-3,
        "predicted_first_west_upper_reduction_day": 2,
        "actual_first_west_upper_reduction_day": (
            birth["first_west_upper_gap_reduction"]["day"]),
        "first_west_upper_reduction_confirmed": (
            birth["first_west_upper_gap_reduction"]["day"] == 2),
        "predicted_depth": "0_100",
        "actual_depth": birth["strongest_depth"],
        "depth_confirmed": birth["strongest_depth"] == "0_100",
        "predicted_third": "west_third",
        "actual_third": birth["strongest_third"],
        "third_confirmed": birth["strongest_third"] == "west_third",
    }

    report = {
        "format": "gyre-daily-tracer-subfamily-v1", "status": "PASS",
        "case": CASE, "execution": "production step through self._step_jitted",
        "days": list(DAILY_ATTRIBUTION_DAYS), "fields": list(FIELDS),
        "arms_root": str(arms_root), "free_member": str(free_member),
        "nemo_daily_root": str(nemo_daily_root),
        "control_reproduction": {
            "days_checked": 360, "fields": list(FIELDS),
            "cells_unequal": control_unequal, "all_bit_identical": True,
            "manifest": {
                "path": str(control_member / "manifest.json"),
                "sha256": _sha256(control_member / "manifest.json"),
                "reset": control_manifest["daily_reset"],
            },
        },
        "pre_reset_day1_cells_unequal": day1_unequal,
        "subfamily_manifests": {family: {
            "path": str(members[family] / "manifest.json"),
            "sha256": _sha256(members[family] / "manifest.json"),
            "reset": manifests[family]["daily_reset"],
        } for family in DAILY_TRACER_SUBFAMILIES},
        "subfamily_rows": rows,
        "subfamily_registered_row_count": len(rows),
        "subfamily_registry_exact": True,
        "subfamily_ranking": ranking,
        "winner": winner,
        "birth": birth,
        "frozen_prediction": prediction,
        "combined_tracer_parent_day240_removed_T_rms_K": (
            1.6394643374813826e-2),
        "ownership_caveat": (
            "reset leverage identifies a state subfamily, not a first wrong "
            "producer statement"),
        "worktree": stamp,
    }

    if include_cadence:
        cadence_members = {}
        cadence_specs = {}
        cadence_manifests = {}
        for interval_days in DAILY_TRACER_CADENCES_DAYS:
            label = f"{interval_days}d"
            member = (members[winner] if interval_days == 1 else
                      Path(arms_root) / f"{winner}_{interval_days}d"
                      / "lego_seed0_year")
            cadence_members[label] = member
            cadence_specs[label] = {
                "family": winner, "interval_days": interval_days}
            cadence_manifests[label] = _daily_arm_manifest(
                member, winner, expect_commit, interval_days=interval_days,
                expected_variables=tuple(
                    year.DAILY_RESET_VARIABLES[winner]))
            candidate = _direct_snapshot(member, 1)
            unequal = {
                name: _bit_unequal(candidate[name], day1[name])
                for name in FIELDS}
            require(all(value == 0 for value in unequal.values()),
                    f"{winner} {label}: moved pre-reset day-1 snapshot")

        cadence_rows, cadence_baseline = _daily_metric_rows(
            year, cadence_members, cadence_specs, free_member,
            nemo_daily_root, masks, nlev)
        if plant == "daily-cadence-score-registry":
            cadence_rows.pop()
        try:
            _validate_daily_metric_registry(
                cadence_rows,
                tuple(f"{days}d" for days in DAILY_TRACER_CADENCES_DAYS))
        except GateError as error:
            if plant == "daily-cadence-score-registry":
                return {
                    "format": "gyre-daily-tracer-subfamily-v1",
                    "status": "PLANT-FIRED", "plant": plant,
                    "control": {"error": str(error),
                                "registered_rows": len(cadence_rows),
                                "required_rows": 160},
                    "worktree": stamp,
                }
            raise
        if plant == "daily-cadence-score-registry":
            raise GateError("daily-cadence-score-registry plant did not fire")
        require(cadence_baseline == baseline,
                "cadence scoring changed the immutable free baseline")

        cadence_ranking = []
        for interval_days in DAILY_TRACER_CADENCES_DAYS:
            label = f"{interval_days}d"
            row = next(
                item for item in cadence_rows
                if item["family"] == label and item["day"] == 240
                and item["field"] == "T")
            cadence_ranking.append({
                "label": label,
                "interval_days": interval_days,
                "day240_free_vs_nemo_T_rms_K": row["free_vs_nemo_rms"],
                "day240_reset_vs_nemo_T_rms_K": row["reset_vs_nemo_rms"],
                "day240_removed_T_rms_K": row["removed_gap_rms"],
            })
        cadence_removed = [
            row["day240_removed_T_rms_K"] for row in cadence_ranking]
        positive = all(value > 0.0 for value in cadence_removed)
        nonincreasing = all(
            left >= right
            for left, right in zip(cadence_removed, cadence_removed[1:]))
        report.update({
            "cadence_rows": cadence_rows,
            "cadence_registered_row_count": len(cadence_rows),
            "cadence_registry_exact": True,
            "cadence_ranking": cadence_ranking,
            "cadence_manifests": {label: {
                "path": str(cadence_members[label] / "manifest.json"),
                "sha256": _sha256(
                    cadence_members[label] / "manifest.json"),
                "reset": cadence_manifests[label]["daily_reset"],
            } for label in cadence_members},
            "cadence_prediction": {
                "all_day240_removals_positive": positive,
                "removal_nonincreasing_with_interval": nonincreasing,
                "prediction_confirmed": positive and nonincreasing,
            },
            "all_moved_rows_registered": (
                len(rows) == 80 and len(cadence_rows) == 160),
        })
    else:
        require(plant not in ("daily-cadence-score-registry",),
                f"plant {plant!r} requires --with-cadence")
        report["cadence_status"] = "PENDING_MEASURED_WINNER"
        report["all_moved_rows_registered"] = len(rows) == 80

    require(plant in (None, "none"), f"plant {plant!r} did not fire")
    print("\nDAILY TRACER SUBFAMILY DAY-240 TEMPERATURE RANKING")
    for rank, row in enumerate(ranking, 1):
        print(f"  {rank:2d} {row['family']:>11s} "
              f"reset={row['day240_reset_vs_nemo_T_rms_K']:.12e} "
              f"removed={row['day240_removed_T_rms_K']:.12e}")
    print(f"  winner {winner}; west/upper reduction day "
          f"{birth['first_west_upper_gap_reduction']['day']}")
    if include_cadence:
        print("\nMEASURED-WINNER CADENCE")
        for row in report["cadence_ranking"]:
            print(f"  {row['label']:>3s} "
                  f"reset={row['day240_reset_vs_nemo_T_rms_K']:.12e} "
                  f"removed={row['day240_removed_T_rms_K']:.12e}")
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
    try:
        control = _trigger_shapley_control()
        print(f"  trigger Shapley owner/interaction/cancellation {control} "
              "-- OK")
    except Exception as error:  # noqa: BLE001
        failures.append(f"trigger Shapley control failed: {error}")
    try:
        context = _temperature_context_marginal_control()
        telescope = _temperature_process_telescope_control()
        print(f"  trigger temperature-process context {context} and "
              f"telescope {telescope} -- OK")
    except Exception as error:  # noqa: BLE001
        failures.append(
            f"trigger temperature-process controls failed: {error}")
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
    parser.add_argument("--produce-vertical-trace", action="store_true",
                        help="extend the existing production process trace "
                             "with consumed tracer-ZDF operands")
    parser.add_argument("--lego-process-record", type=Path, default=None,
                        help="validate a Round-124 legoESM process trace")
    parser.add_argument("--process-budget", type=Path, default=None,
                        help="score this NEMO process root against "
                             "--lego-process-record")
    parser.add_argument("--vertical-record", type=Path, default=None,
                        help="validate a Round-125 tra_zdf internal-record "
                             "root")
    parser.add_argument("--lego-vertical-record", type=Path, default=None,
                        help="validate a Round-126 legoESM vertical trace")
    parser.add_argument("--vertical-budget", type=Path, default=None,
                        help="score this NEMO vertical record against "
                             "--lego-vertical-record")
    parser.add_argument("--trigger-state-budget", type=Path, default=None,
                        help="attribute this NEMO EVD record's trigger "
                             "crossings to T, S, and live depth")
    parser.add_argument(
        "--trigger-temperature-process-budget", type=Path, default=None,
        help="decompose this NEMO EVD record's temperature-trigger share "
             "among the admitted compiled-order process rows")
    parser.add_argument("--trigger-lego-daily-root", type=Path,
                        default=DEFAULT_TRIGGER_LEGO_DAILY,
                        help="independent daily legoESM core-state root")
    parser.add_argument("--developed-step-walk", action="store_true",
                        help="run the Round-136 step-1081 production walk "
                             "from NEMO's admitted day-180 restart")
    parser.add_argument("--developed-fct-walk", action="store_true",
                        help="extend the developed step walk across the "
                             "admitted Round-153 61-field FCT record")
    parser.add_argument("--developed-transport-walk", action="store_true",
                        help="extend the developed step walk across the "
                             "admitted Round-154 stage-3 transport record")
    parser.add_argument("--developed-process-root", type=Path,
                        default=DEFAULT_PROCESS_RECORD_ROOT)
    parser.add_argument("--developed-vertical-root", type=Path,
                        default=DEFAULT_DEVELOPED_VERTICAL_ROOT)
    parser.add_argument("--daily-record-root", type=Path,
                        default=DEFAULT_DEVELOPED_DAILY_ROOT)
    parser.add_argument("--daily-record-audit", type=Path, default=None)
    parser.add_argument("--developed-fct-record-root", type=Path,
                        default=None)
    parser.add_argument("--developed-transport-record-root", type=Path,
                        default=None)
    parser.add_argument("--developed-stage2-walk", action="store_true",
                        help="walk NEMO's compiled stage-2 momentum program "
                             "across the admitted Round-156 record")
    parser.add_argument("--developed-stage2-record-root", type=Path,
                        default=None)
    parser.add_argument("--developed-stage1-output-walk", action="store_true",
                        help="walk the developed stage-1 momentum output "
                             "through the admitted RHS and stage records")
    parser.add_argument("--developed-rhs-root", type=Path, default=None)
    parser.add_argument("--developed-rhs-family-root", type=Path,
                        default=None)
    parser.add_argument("--developed-tke-walk", action="store_true",
                        help="walk the developed step-1081 TKE closure across "
                             "the admitted Round-164 records")
    parser.add_argument("--developed-tke-record-root", type=Path,
                        default=None)
    parser.add_argument(
        "--developed-vertical-sensitivity", action="store_true",
        help="rank the day-240 sensitivity to NEMO heat diffusivity and "
             "the complete effective vertical tracer coefficient")
    parser.add_argument("--developed-stage2-adv-split", action="store_true",
                        help="split NEMO's vector-invariant dyn_adv into its "
                             "kinetic-energy gradient and vertical advection "
                             "halves at the developed day-180 state")
    parser.add_argument("--developed-stage2-wzv-walk", action="store_true",
                        help="walk the PRODUCER of NEMO's stage-2 vertical "
                             "velocity at the developed day-180 state")
    parser.add_argument("--developed-stage2-wzv-split", action="store_true",
                        help="score NEMO's SECOND per-stage continuity solve "
                             "at the developed day-180 state")
    parser.add_argument("--developed-stage2-face-r3", action="store_true",
                        help="substitute the oracle's own r3u/r3v into the "
                             "velocity-form continuity producer and walk the "
                             "face ratio to its own producer statement")
    parser.add_argument("--developed-stage2-t-r3", action="store_true",
                        help="substitute the oracle's own T-point ratio into "
                             "the velocity-form continuity producer and score "
                             "legoESM's stage sea surface height")
    parser.add_argument("--developed-process-rank-split", action="store_true",
                        help="run round 152's one-step magnitude ranking in "
                             "both stage programs and report which row grew")
    parser.add_argument("--reference-process-trace", type=Path,
                        default=DEFAULT_REFERENCE_PROCESS_TRACE,
                        help="admitted Round-124 process trace that a new "
                             "vertical trace must reproduce")
    parser.add_argument("--vertical-process-root", type=Path,
                        default=DEFAULT_PROCESS_RECORD_ROOT,
                        help="admitted Round-123 process root used to align "
                             "the vertical record step by step")
    parser.add_argument("--expect-commit", default=None,
                        help="required clean producer commit for records")
    parser.add_argument("--forcing-gate", action="store_true")
    parser.add_argument("--day-gap", action="store_true")
    parser.add_argument("--daily-reset-attribution", action="store_true",
                        help="score the four Round-134 daily reset arms")
    parser.add_argument("--daily-tracer-subfamily-attribution",
                        action="store_true",
                        help="score Round-135 T-only/S-only reset arms")
    parser.add_argument("--with-cadence", action="store_true",
                        help="also score the measured winner at 2/4/8 days")
    parser.add_argument("--arms-root", type=Path,
                        help="root containing control/tracer/vector/ssh/tke")
    parser.add_argument("--free-member", type=Path,
                        help="immutable free-run member directory")
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
    if args.developed_stage2_face_r3:
        require(args.expect_commit is not None,
                "--developed-stage2-face-r3 needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage2-face-r3 needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage2-face-r3 needs "
                "--developed-stage2-record-root")
        try:
            report = developed_stage2_face_r3_walk(
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.developed_stage2_record_root,
                args.root, plant=args.plant)
        except GateError as error:
            if args.plant in (None, "none"):
                raise
            if str(error).startswith("PLANT-BLIND"):
                # A control that did NOT catch its plant must never print the
                # marker a caught one prints (round 103's defect).
                print(f"STATUS PLANT-BLIND: {args.plant}: {error}")
                return 2
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        require(args.plant in (None, "none"),
                f"plant {args.plant!r} did not fire")
        print(json.dumps(report["vertical_velocity_scored_against_nemo"],
                         indent=2))
        print(json.dumps(report["producer_decomposition"], indent=2))
        print("STATUS PASS")
        return 0
    if args.developed_stage2_t_r3:
        require(args.expect_commit is not None,
                "--developed-stage2-t-r3 needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage2-t-r3 needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage2-t-r3 needs "
                "--developed-stage2-record-root")
        try:
            report = developed_stage2_t_r3_walk(
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.developed_stage2_record_root,
                args.root, plant=args.plant)
        except GateError as error:
            if args.plant in (None, "none"):
                raise
            if str(error).startswith("PLANT-BLIND"):
                # A control that did NOT catch its plant must never print the
                # marker a caught one prints (round 103's defect).
                print(f"STATUS PLANT-BLIND: {args.plant}: {error}")
                return 2
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        require(args.plant in (None, "none"),
                f"plant {args.plant!r} did not fire")
        print(json.dumps(report["vertical_velocity_scored_against_nemo"],
                         indent=2))
        print(json.dumps(report["stage_sea_surface_height"], indent=2))
        print("STATUS PASS")
        return 0
    if args.developed_stage1_output_walk:
        require(args.expect_commit is not None,
                "--developed-stage1-output-walk needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage1-output-walk needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage1-output-walk needs "
                "--developed-stage2-record-root")
        require(args.developed_rhs_root is not None,
                "--developed-stage1-output-walk needs --developed-rhs-root")
        require(args.developed_rhs_family_root is not None,
                "--developed-stage1-output-walk needs "
                "--developed-rhs-family-root")
        report = developed_stage1_output_walk(
            args.daily_record_root, args.daily_record_audit,
            args.expect_commit, args.developed_stage2_record_root,
            args.developed_rhs_root, args.developed_rhs_family_root,
            args.root, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        require(args.plant in (None, "none"),
                f"plant {args.plant!r} did not fire")
        print("DEVELOPED STAGE-1 OUTPUT FIRST NON-BIT: "
              f"{report['first_non_bit']}")
        print("STATUS PASS")
        return 0
    if args.developed_tke_walk:
        require(args.expect_commit is not None,
                "--developed-tke-walk needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-tke-walk needs --daily-record-audit")
        require(args.developed_tke_record_root is not None,
                "--developed-tke-walk needs --developed-tke-record-root")
        report = developed_tke_statement_walk(
            args.daily_record_root, args.daily_record_audit,
            args.expect_commit, args.developed_tke_record_root,
            args.root, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        require(args.plant in (None, "none"),
                f"plant {args.plant!r} did not fire")
        print("DEVELOPED TKE FIRST NON-BIT: "
              f"{report['first_non_bit_statement']}")
        print("STATUS PASS")
        return 0
    if args.developed_vertical_sensitivity:
        require(args.expect_commit is not None,
                "--developed-vertical-sensitivity needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-vertical-sensitivity needs "
                "--daily-record-audit")
        try:
            report = developed_vertical_day240_sensitivity(
                args.developed_process_root, args.developed_vertical_root,
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.root, plant=args.plant)
        except GateError as error:
            if args.plant in (None, "none"):
                raise
            if str(error).startswith("PLANT-BLIND"):
                print(f"STATUS PLANT-BLIND: {args.plant}: {error}")
                return 2
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "REFUTED":
            print("STATUS REFUTED: " + report["conclusion"])
            return 1
        require(args.plant in (None, "none"),
                f"plant {args.plant!r} did not fire")
        print(json.dumps(report["ranking"], indent=2))
        print("STATUS PASS")
        return 0
    if args.developed_process_rank_split:
        require(args.expect_commit is not None,
                "--developed-process-rank-split needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-process-rank-split needs --daily-record-audit")
        try:
            report = developed_process_rank_split(
                args.developed_process_root, args.daily_record_root,
                args.daily_record_audit, args.expect_commit, args.root,
                plant=args.plant)
        except GateError as error:
            if args.plant in (None, "none"):
                raise
            if str(error).startswith("PLANT-BLIND"):
                # A control that did NOT catch its plant must never print the
                # marker a caught one prints (round 103's defect).
                print(f"STATUS PLANT-BLIND: {args.plant}: {error}")
                return 2
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        require(args.plant in (None, "none"),
                f"plant {args.plant!r} did not fire")
        print(json.dumps(report["comparison"], indent=2))
        print(json.dumps(report["cancellation"], indent=2))
        print("STATUS PASS")
        return 0
    if args.developed_stage2_wzv_split:
        require(args.expect_commit is not None,
                "--developed-stage2-wzv-split needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage2-wzv-split needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage2-wzv-split needs "
                "--developed-stage2-record-root")
        try:
            report = developed_stage2_wzv_split_walk(
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.developed_stage2_record_root,
                args.root, plant=args.plant)
        except GateError as error:
            if args.plant in (None, "none"):
                raise
            if str(error).startswith("PLANT-BLIND"):
                # A control that did NOT catch its plant must never print the
                # marker a caught one prints (round 103's defect).  Exit 2,
                # distinct from a fired plant's 1.
                print(f"STATUS PLANT-BLIND: {args.plant}: {error}")
                return 2
            # Any OTHER fail-closed check tripping under a plant is the plant
            # being caught by a control earlier in the walk.
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        for name, row in report[
                "vertical_velocity_scored_against_nemo"].items():
            print(f"  WW {name}: rms {row['active_rms']:.6e} "
                  f"max {row['active_max_abs']:.6e} "
                  f"removed {row['rms_removed_fraction'] * 100.0:.3f}% "
                  f"relative {row['relative_to_own_rms']:.6e}")
        for name, row in report["stage2_rhs_scored_against_nemo"].items():
            print(f"  RHS {name}: u rms {row['u']['active_rms']:.6e} "
                  f"v rms {row['v']['active_rms']:.6e}")
        out = report["stage2_output_velocity_scored_against_nemo"]
        for name in ("before_shared", "after_momentum"):
            print(f"  STAGE2-OUT {name}: u rms {out[name]['u']['active_rms']:.6e} "
                  f"v rms {out[name]['v']['active_rms']:.6e}")
        identity = report["controls"]["tracer_identity"]
        print("  TRACER IDENTITY cells moved: "
              + ", ".join(f"{key} {row['cells_unequal']}"
                          for key, row in identity.items()))
        step_move = report["one_step_tracer_move_through_the_velocity"]
        print(f"  ONE-STEP T rms {step_move['T']['one_step_rms']:.6e} K "
              f"({step_move['T']['as_fraction_of_the_shared_field_move']:.4f} "
              "of round 159's shared-field move)")
        clock = report["residual_discriminator"]["oracle_stage_clock_pair"]
        print(f"  RESIDUAL clock/level pair: rms {clock['active_rms']:.6e} "
              f"removed {clock['residual_removed_fraction'] * 100.0:.3f}% "
              f"relative {clock['relative_to_own_rms']:.6e}")
        for tag, row in report["residual_discriminator"][
                "live_thickness_ratio_operand"].items():
            print(f"  RESIDUAL r3{tag}(Kmm) vs oracle: cells "
                  f"{row['active_cells_unequal']}/{row['active_cells_scored']} "
                  f"rms {row['active_rms']:.6e} "
                  f"relative {row['relative_to_own_rms']:.6e}")
        print("STATUS PASS: round-160 developed stage-2 wzv split walk")
        return 0
    if args.developed_stage2_wzv_walk:
        require(args.expect_commit is not None,
                "--developed-stage2-wzv-walk needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage2-wzv-walk needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage2-wzv-walk needs "
                "--developed-stage2-record-root")
        try:
            report = developed_stage2_wzv_walk(
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.developed_stage2_record_root,
                args.root, plant=args.plant)
        except GateError as error:
            if args.plant in (None, "none"):
                raise
            if str(error).startswith("PLANT-BLIND"):
                # A control that did NOT catch its plant must never print the
                # marker a caught one prints; round 103 shipped a plant that
                # announced success while firing and this campaign scrapes
                # logs.  Exit 2, distinct from a fired plant's 1.
                print(f"STATUS PLANT-BLIND: {args.plant}: {error}")
                return 2
            # Any OTHER fail-closed check tripping under a plant is the plant
            # being caught by a control earlier in the walk.
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        for name, row in report[
                "vertical_velocity_scored_against_nemo"].items():
            print(f"  WW {name}: rms {row['active_rms']:.6e} "
                  f"max {row['active_max_abs']:.6e} "
                  f"removed {row['rms_removed_fraction'] * 100.0:.3f}% "
                  f"cells {row['active_cells_unequal']}/"
                  f"{row['active_cells_scored']}")
        for name, row in report["stage2_rhs_scored_against_nemo"].items():
            print(f"  RHS {name}: u rms {row['u']['active_rms']:.6e} "
                  f"v rms {row['v']['active_rms']:.6e}")
        trow = report["tracer_consequence_of_changing_the_shared_field"]
        print(f"  TRACER one-step T rms {trow['T']['one_step_rms']:.6e} K "
              f"({trow['T']['cells_unequal']}/{trow['T']['cells_scored']} "
              f"cells), S rms {trow['S']['one_step_rms']:.6e}")
        print("STATUS PASS: round-159 developed stage-2 wzv producer walk")
        return 0
    if args.developed_stage2_adv_split:
        require(args.expect_commit is not None,
                "--developed-stage2-adv-split needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage2-adv-split needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage2-adv-split needs "
                "--developed-stage2-record-root")
        try:
            report = developed_stage2_advection_split(
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.developed_stage2_record_root,
                args.root, plant=args.plant)
        except GateError as error:
            # A plant that trips a fail-closed check raises rather than
            # returning a report; it still announces itself, because this
            # campaign scrapes logs for the marker.
            if args.plant in (None, "none"):
                raise
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        for tag in ("u", "v"):
            row = report["split"][tag]
            print(f"  SPLIT {tag}: keg {row['keg_difference_rms']:.6e} "
                  f"zad {row['zad_difference_rms']:.6e} "
                  f"owner {row['owner']} ratio {row['owner_ratio']:.4g}")
        for name, row in report["arms_scored_against_nemo_after_adv"].items():
            print(f"  ARM {name}: u rms {row['u']['active_rms']:.6e} "
                  f"v rms {row['v']['active_rms']:.6e}")
        print("STATUS PASS: round-158 developed stage-2 advection split")
        return 0
    if args.developed_stage2_walk:
        require(args.expect_commit is not None,
                "--developed-stage2-walk needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-stage2-walk needs --daily-record-audit")
        require(args.developed_stage2_record_root is not None,
                "--developed-stage2-walk needs "
                "--developed-stage2-record-root")
        try:
            report = developed_stage2_rhs_walk(
                args.daily_record_root, args.daily_record_audit,
                args.expect_commit, args.developed_stage2_record_root,
                args.root, plant=args.plant)
        except GateError as error:
            # A plant that trips a fail-closed check raises rather than
            # returning a report; it still has to announce itself, because
            # this campaign scrapes logs for the marker.
            if args.plant in (None, "none"):
                raise
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        print("STATUS PASS: developed stage-2 first non-bit row "
              f"{report['first_non_bit_row']}")
        return 0
    if (args.developed_step_walk or args.developed_fct_walk
            or args.developed_transport_walk):
        require(args.expect_commit is not None,
                "--developed-step-walk needs --expect-commit")
        require(args.daily_record_audit is not None,
                "--developed-step-walk needs --daily-record-audit")
        if args.developed_fct_walk:
            require(args.developed_fct_record_root is not None,
                    "--developed-fct-walk needs "
                    "--developed-fct-record-root")
        if args.developed_transport_walk:
            require(args.developed_transport_record_root is not None,
                    "--developed-transport-walk needs "
                    "--developed-transport-record-root")
        report = developed_state_process_walk(
            args.developed_process_root, args.developed_vertical_root,
            args.daily_record_root, args.daily_record_audit,
            args.expect_commit, args.root, mesh_path=args.mesh,
            plant=args.plant,
            fct_record_root=(args.developed_fct_record_root
                             if args.developed_fct_walk else None),
            transport_record_root=(args.developed_transport_record_root
                                   if args.developed_transport_walk else None))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report['control']}")
            return 1
        if args.developed_transport_walk:
            print("STATUS PASS: developed stage-3 transport first owner "
                  f"{report['internal_statement_owner']}")
        elif args.developed_fct_walk:
            print("STATUS PASS: developed-state FCT first owner "
                  f"{report['internal_statement_owner']}")
        else:
            print("STATUS PASS: developed-state step 1081 first non-bit "
                  f"{report['first_non_bit_boundary']}")
        return 0
    if args.daily_tracer_subfamily_attribution:
        require(not args.daily_reset_attribution,
                "choose one daily-reset attribution mode")
        require(args.expect_commit is not None,
                "--daily-tracer-subfamily-attribution needs --expect-commit")
        require(args.arms_root is not None and args.free_member is not None,
                "--daily-tracer-subfamily-attribution needs --arms-root and "
                "--free-member")
        report = daily_tracer_subfamily_attribution(
            args.arms_root, args.free_member, args.nemo_root,
            args.expect_commit, mesh_path=args.mesh,
            include_cadence=args.with_cadence, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        print(f"STATUS PASS: daily tracer subfamily owner={report['winner']}")
        return 0
    if args.daily_reset_attribution:
        require(not args.with_cadence,
                "--with-cadence belongs to tracer-subfamily attribution")
        require(args.expect_commit is not None,
                "--daily-reset-attribution needs --expect-commit")
        require(args.arms_root is not None and args.free_member is not None,
                "--daily-reset-attribution needs --arms-root and --free-member")
        report = daily_reset_attribution(
            args.arms_root, args.free_member, args.nemo_root,
            args.expect_commit, mesh_path=args.mesh, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: {report['control']}")
            return 1
        print(f"STATUS PASS: daily reset owner={report['winner']}")
        return 0
    if args.produce_process_trace or args.produce_vertical_trace:
        require(args.expect_commit is not None,
                "trace production needs --expect-commit")
        report = produce_lego_process_trace(
            args.root, args.expect_commit, mesh_path=args.mesh,
            include_vertical=args.produce_vertical_trace,
            reference_process_trace=args.reference_process_trace)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        return 0
    if args.trigger_state_budget is not None:
        require(args.lego_vertical_record is not None,
                "--trigger-state-budget needs --lego-vertical-record")
        require(args.expect_commit is not None,
                "--trigger-state-budget needs --expect-commit")
        report = score_trigger_state_budget(
            args.trigger_state_budget, args.lego_vertical_record,
            args.trigger_lego_daily_root, args.expect_commit,
            mesh_path=args.mesh, process_root=args.vertical_process_root,
            plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report['control']}")
            return 1
        print("STATUS PASS: EVD trigger-state budget "
              f"owner={report['ranking'][0]['owner']}")
        return 0
    if args.trigger_temperature_process_budget is not None:
        require(args.lego_vertical_record is not None,
                "--trigger-temperature-process-budget needs "
                "--lego-vertical-record")
        require(args.expect_commit is not None,
                "--trigger-temperature-process-budget needs --expect-commit")
        report = score_trigger_temperature_process_budget(
            args.trigger_temperature_process_budget,
            args.lego_vertical_record, args.trigger_lego_daily_root,
            args.expect_commit, mesh_path=args.mesh,
            process_root=args.vertical_process_root, plant=args.plant)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report['control']}")
            return 1
        print("STATUS PASS: EVD temperature-process budget "
              f"owner={report['ranking'][0]['owner']}")
        return 0
    if args.vertical_budget is not None:
        require(args.lego_vertical_record is not None,
                "--vertical-budget needs --lego-vertical-record")
        require(args.expect_commit is not None,
                "--vertical-budget needs --expect-commit")
        report = score_vertical_budget(
            args.vertical_budget, args.lego_vertical_record,
            args.expect_commit, process_root=args.vertical_process_root,
            immutable_lego_root=args.immutable_lego_root,
            nemo_root=args.nemo_root, mesh_path=args.mesh)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        return 0
    if args.lego_vertical_record is not None:
        require(args.expect_commit is not None,
                "--lego-vertical-record needs --expect-commit")
        report = validate_lego_vertical_trace(
            args.lego_vertical_record, args.expect_commit,
            plant=args.plant, mesh_path=args.mesh)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
            print(f"  wrote {args.json}")
        if report["status"] == "PLANT-FIRED":
            print(f"STATUS PLANT-FIRED: {args.plant}: "
                  f"{report.get('reason', report.get('control', 'moved'))}")
            return 1
        print("STATUS PASS: legoESM vertical trace "
              f"{report['layout']['record_count']} frames")
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


# ------------------------- Round-127 EVD trigger-state magnitude budget ---
TRIGGER_PLAYERS = ("temperature", "salinity", "live_depth")
TRIGGER_DAY_START = 180
TRIGGER_DAY_STOP = 240
TRIGGER_VERTICAL_RECORD_COMMIT = (
    "4cac617cd928007506f2de7ccb098f87e04204d1")
TRIGGER_LEGO_TRACE_COMMIT = (
    "34070a99e43227412a6118ecdcf31180b0e1c6d3")
DEFAULT_TRIGGER_VERTICAL_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round125/"
    "oracle_vertical_decomposition")
DEFAULT_TRIGGER_LEGO_TRACE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round126/"
    "lego_vertical_trace_v2")
DEFAULT_TRIGGER_LEGO_DAILY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/"
    "gyre/lego_seed0_year")


def _trigger_subset_label(mask: int) -> str:
    names = [name for bit, name in enumerate(TRIGGER_PLAYERS)
             if mask & (1 << bit)]
    return "+".join(names) if names else "nemo"


def _trigger_shapley(values: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Exact three-player Shapley allocation for scalar or array values."""
    require(set(values) == set(range(8)),
            "trigger Shapley cube must contain all eight subsets")
    shape = np.asarray(values[0]).shape
    require(all(np.asarray(value).shape == shape for value in values.values()),
            "trigger Shapley cube values have unequal shapes")
    result = {}
    # Six times each weight is 2, 1, 1, or 2 for subset sizes 0, 1, 1, 2.
    for player in range(3):
        bit = 1 << player
        numerator = np.zeros(shape, dtype=np.float64)
        for subset in range(8):
            if subset & bit:
                continue
            size = int(subset.bit_count())
            weight6 = 2 if size in (0, 2) else 1
            numerator = numerator + weight6 * (
                np.asarray(values[subset | bit], dtype=np.float64)
                - np.asarray(values[subset], dtype=np.float64))
        result[TRIGGER_PLAYERS[player]] = numerator / np.float64(6.0)
    return result


def _trigger_shapley_control() -> dict:
    """Non-vacuous synthetic ownership, interaction, and cancellation arms."""
    values = {}
    for subset in range(8):
        temperature = bool(subset & 1)
        salinity = bool(subset & 2)
        depth = bool(subset & 4)
        values[subset] = np.asarray([
            float(temperature),
            float(temperature and salinity),
            float(temperature and not depth),
        ])
    shares = _trigger_shapley(values)
    expected = {
        "temperature": np.asarray([1.0, 0.5, 0.5]),
        "salinity": np.asarray([0.0, 0.5, 0.0]),
        "live_depth": np.asarray([0.0, 0.0, -0.5]),
    }
    require(all(np.array_equal(shares[name], expected[name])
                for name in TRIGGER_PLAYERS),
            "synthetic trigger Shapley owner/interaction/cancellation failed")
    closure = sum(shares.values())
    require(np.array_equal(closure, values[7] - values[0]),
            "synthetic trigger Shapley cube does not close")
    return {name: shares[name].tolist() for name in TRIGGER_PLAYERS}


def _eta_for_recorded_r3t(r3t: np.ndarray, H_bathy: np.ndarray,
                           wet2: np.ndarray) -> tuple[np.ndarray, dict]:
    """Invert the recorded NEMO multiply, proving the forward bits recover."""
    target = np.asarray(r3t, dtype=np.float64)
    H = np.asarray(H_bathy, dtype=np.float64)
    wet2 = np.asarray(wet2, dtype=bool)
    require(target.shape == H.shape == wet2.shape,
            "r3t inverse operands have unequal shapes")
    require(bool(np.all(np.isfinite(target[wet2])))
            and bool(np.all(np.isfinite(H[wet2]) & (H[wet2] > 0.0))),
            "could not invert non-finite r3t or non-positive wet depth")
    safe_H = np.where(wet2, H, 1.0)
    reciprocal = np.float64(1.0) / safe_H
    eta = np.where(wet2, target / reciprocal, 0.0)
    iterations = 0
    for iterations in range(65):
        forward = np.where(wet2, eta * reciprocal, 0.0)
        equal = forward.view(np.uint64) == target.view(np.uint64)
        if bool(np.all(equal[wet2])):
            break
        pending = wet2 & ~equal
        upward = np.nextafter(eta, np.inf)
        downward = np.nextafter(eta, -np.inf)
        up_forward = upward * reciprocal
        down_forward = downward * reciprocal
        up_exact = (up_forward.view(np.uint64) == target.view(np.uint64))
        down_exact = (down_forward.view(np.uint64) == target.view(np.uint64))
        eta = np.where(pending & up_exact, upward, eta)
        eta = np.where(pending & ~up_exact & down_exact, downward, eta)
        unresolved = pending & ~up_exact & ~down_exact
        eta = np.where(unresolved & (forward < target), upward, eta)
        eta = np.where(unresolved & (forward > target), downward, eta)
        # A signed-zero mismatch is numerically equal but still a different
        # recorded bit.  Copying that sign into eta makes the positive
        # reciprocal multiplication preserve it.
        signed_zero = unresolved & (forward == target) & (target == 0.0)
        eta = np.where(signed_zero, target, eta)
    forward = np.where(wet2, eta * reciprocal, 0.0)
    unequal = int(np.count_nonzero(
        forward[wet2].view(np.uint64) != target[wet2].view(np.uint64)))
    require(unequal == 0,
            f"could not invert recorded r3t multiply in {unequal} wet cells")
    return eta, {"wet_cells_unequal": unequal,
                 "search_iterations": int(iterations)}


def _trigger_masks_from_K(heat_K: np.ndarray, replacement: float,
                          interface_wet: np.ndarray) -> np.ndarray:
    values = np.asarray(heat_K, dtype=np.float64)
    require(values.shape == interface_wet.shape,
            "trigger coefficient/mask shapes differ")
    return (values == replacement) & interface_wet


def _trigger_spatial_census(disagreement: np.ndarray,
                            model_mask: np.ndarray,
                            nemo_mask: np.ndarray,
                            depth: np.ndarray,
                            regions: dict[str, np.ndarray]) -> dict:
    disagreement = np.asarray(disagreement, dtype=bool)
    require(disagreement.shape == depth.shape,
            "trigger census depth shape differs from mask")
    depth_masks = {
        "0_100m": depth <= 100.0,
        "100_1000m": (depth > 100.0) & (depth <= 1000.0),
        "below_1000m": depth > 1000.0,
    }
    region_names = (
        "west_third", "interior_third", "east_third",
        f"emp_south_le_{EMP_SPLIT_LAT_DEG}N",
        f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N",
    )
    return {
        "total": int(np.count_nonzero(disagreement)),
        "model_only": int(np.count_nonzero(model_mask & ~nemo_mask)),
        "nemo_only": int(np.count_nonzero(nemo_mask & ~model_mask)),
        "depth": {name: int(np.count_nonzero(disagreement & selected))
                  for name, selected in depth_masks.items()},
        "region": {name: int(np.count_nonzero(
            disagreement & regions[name][..., None]))
            for name in region_names},
    }


def _add_trigger_census(total: dict, row: dict) -> None:
    total["total"] += row["total"]
    total["model_only"] += row["model_only"]
    total["nemo_only"] += row["nemo_only"]
    for group in ("depth", "region"):
        for name, value in row[group].items():
            total[group][name] += value


def score_trigger_state_budget(
        nemo_vertical_root: Path, lego_trace_root: Path,
        lego_daily_root: Path, expected_commit: str, *,
        mesh_path: Path = DEFAULT_MESH,
        process_root: Path = DEFAULT_PROCESS_RECORD_ROOT,
        plant: str | None = None) -> dict:
    """Attribute EVD trigger crossings to T, S, and live free-surface depth."""
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.eos import nemo_r3t_stretch
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            "trigger-state budget refuses a dirty worktree")
    require(stamp["commit"] == expected_commit,
            "trigger-state budget commit differs from --expect-commit")
    require(plant in (None, "none", "trigger-stored-mask-bit",
                      "trigger-temperature-bit"),
            f"unknown trigger-state plant {plant!r}")

    nemo_vertical_root = Path(nemo_vertical_root)
    lego_trace_root = Path(lego_trace_root)
    lego_daily_root = Path(lego_daily_root)
    admission_nemo = validate_vertical_record(
        nemo_vertical_root, TRIGGER_VERTICAL_RECORD_COMMIT,
        process_root=process_root)
    admission_lego = validate_lego_vertical_trace(
        lego_trace_root, TRIGGER_LEGO_TRACE_COMMIT, mesh_path=mesh_path)
    _trigger_shapley_control()

    card = build_nemo_testcase_card(CASE)
    gate = _gate()
    wet = gate.expected_masks(card)["T"]
    wet2 = np.any(wet, axis=-1)
    interface_wet = wet[..., :-1] & wet[..., 1:]
    nlev = wet.shape[-1]
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            bn2_intermediate="masked_rn2"))
    carrier_path = (nemo_vertical_root
                    / "GYRE_OMIP_L2_P3_00001080_restart.nc")
    carrier, _ = _gyre_checkpoint_state(card, carrier_path)
    H_bathy = np.asarray(carrier.H_bathy.data, dtype=np.float64)
    evd = card.recipe.model_config.physics.convection.enhanced_diffusion
    require(evd is not None, "trigger-state budget requires active EVD")
    threshold = float(evd.n2_threshold)
    replacement = float(evd.K_conv)
    lego_arrays = _load_vertical_trace_arrays(lego_trace_root)

    @jax.jit
    def production_closure(entry, forcing):
        return model.diagnose_vertical_K(entry, card.dt_s, forcing)

    @jax.jit
    def production_stretch(eta):
        return nemo_r3t_stretch(
            card.recipe.z_coord, eta, carrier.H_bathy.data,
            evaluation="nemo_reciprocal")

    year = _year()
    mesh = year.nemo_operands(mesh_path)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    regions = _regions(lat, wet2)
    raw_gdepw = np.asarray(
        card.recipe.z_coord.nemo_gdepw_0, dtype=np.float64)
    raw_gdepw = raw_gdepw[..., 1:nlev]

    daily_rows = []
    aggregate = {
        name: {"bit_signed_cell_equivalents": 0.0,
               "bit_absolute_cell_equivalents": 0.0,
               "margin_signed_sum_s-2": 0.0,
               "margin_abs_sum_s-2": 0.0,
               "margin_sum_squares_s-4": 0.0,
               "margin_samples": 0}
        for name in TRIGGER_PLAYERS
    }
    endpoint_controls = {
        "nemo_cells_unequal": 0, "lego_cells_unequal": 0,
        "rn2_rn2b_cells_unequal": 0,
        "r3t_stretch_cells_unequal": 0,
    }
    daily_records: dict[int, dict] = {}

    for day in range(TRIGGER_DAY_START, TRIGGER_DAY_STOP):
        step = STEPS_PER_DAY * day + 1
        record = _read_vertical_record(
            nemo_vertical_root
            / f"oracle_trazdf_matrix_kt{step:08d}.bin", step)
        daily_records[step] = record
        nemo_T = _vertical_field(record, "T_Kbb_in", nlev)
        nemo_S = _vertical_field(record, "S_Kbb_in", nlev)
        nemo_r3t = _vertical_field(record, "r3t_Kbb")
        nemo_eta, inverse = _eta_for_recorded_r3t(
            nemo_r3t, H_bathy, wet2)
        recovered_stretch = np.asarray(
            production_stretch(jnp.asarray(nemo_eta)), dtype=np.float64)
        expected_stretch = np.maximum(
            np.float64(1.0) + nemo_r3t, np.float64(1.0e-6))
        stretch_unequal = int(np.count_nonzero(
            recovered_stretch[wet2].view(np.uint64)
            != expected_stretch[wet2].view(np.uint64)))
        endpoint_controls["r3t_stretch_cells_unequal"] += stretch_unequal
        require(stretch_unequal == 0,
                f"day {day}: exact r3t inverse changed {stretch_unequal} "
                "production stretch bits")

        lego = _load_npz(lego_daily_root / f"day{day:03d}.npz")
        nemo_values = {
            "temperature": nemo_T, "salinity": nemo_S,
            "live_depth": nemo_eta,
        }
        lego_values = {
            "temperature": lego["T"], "salinity": lego["S"],
            "live_depth": lego["ssh"],
        }
        base = carrier._replace(
            T=carrier.T.replace(data=jnp.asarray(nemo_T)),
            S=carrier.S.replace(data=jnp.asarray(nemo_S)),
            eta=carrier.eta.replace(data=jnp.asarray(nemo_eta)),
        )
        _, surface = gate._surface_forcings(card, base, step)
        rn2_by_subset = {}
        rn2b_by_subset = {}
        masks = {}
        margins = {}
        for subset in range(8):
            values = {
                name: (lego_values[name] if subset & (1 << bit)
                       else nemo_values[name])
                for bit, name in enumerate(TRIGGER_PLAYERS)
            }
            state = carrier._replace(
                T=carrier.T.replace(
                    data=jnp.asarray(values["temperature"])),
                S=carrier.S.replace(data=jnp.asarray(values["salinity"])),
                eta=carrier.eta.replace(
                    data=jnp.asarray(values["live_depth"])),
            )
            _, _, rn2, rn2b = production_closure(state, surface)
            rn2 = np.asarray(rn2, dtype=np.float64)
            rn2b = np.asarray(rn2b, dtype=np.float64)
            unequal = _different_cells(rn2, rn2b, interface_wet)
            endpoint_controls["rn2_rn2b_cells_unequal"] += unequal
            require(unequal == 0,
                    f"day {day} subset {_trigger_subset_label(subset)}: "
                    f"rn2/rn2b differ in {unequal} wet interfaces")
            rn2_by_subset[subset] = rn2
            rn2b_by_subset[subset] = rn2b
            masks[subset] = ((np.minimum(rn2, rn2b) <= threshold)
                             & interface_wet)
            margins[subset] = np.minimum(rn2, rn2b) - threshold

        nemo_heat = _vertical_field(record, "avt")[:, :, 1:nlev]
        lego_heat = np.asarray(
            lego_arrays["heat_K"][step - PROCESS_START_STEP])
        nemo_record_mask = _trigger_masks_from_K(
            nemo_heat, replacement, interface_wet)
        lego_record_mask = _trigger_masks_from_K(
            lego_heat, replacement, interface_wet)
        nemo_endpoint = int(np.count_nonzero(masks[0] != nemo_record_mask))
        lego_endpoint = int(np.count_nonzero(masks[7] != lego_record_mask))
        endpoint_controls["nemo_cells_unequal"] += nemo_endpoint
        endpoint_controls["lego_cells_unequal"] += lego_endpoint
        require(nemo_endpoint == 0,
                f"day {day}: production NEMO-input mask differs from record "
                f"in {nemo_endpoint} cells")
        require(lego_endpoint == 0,
                f"day {day}: production model-input mask differs from trace "
                f"in {lego_endpoint} cells")

        if day == TRIGGER_DAY_START and plant == "trigger-stored-mask-bit":
            planted = np.array(lego_record_mask, copy=True)
            index = tuple(int(x) for x in np.argwhere(interface_wet)[0])
            planted[index] = ~planted[index]
            moved = int(np.count_nonzero(masks[7] != planted))
            require(moved == 1,
                    f"stored-mask plant moved {moved} registered bits")
            return {
                "format": "gyre-trigger-state-budget-v1",
                "status": "PLANT-FIRED", "plant": plant,
                "control": {"index_jik": list(index),
                            "registered_bits_moved": moved,
                            "clean_endpoint_cells_unequal": lego_endpoint},
                "worktree": stamp,
            }

        if day == TRIGGER_DAY_START and plant == "trigger-temperature-bit":
            active = masks[0] & interface_wet
            candidates = np.argwhere(active)
            if candidates.size == 0:
                candidates = np.argwhere(interface_wet)
            candidate_margins = np.abs(margins[0][tuple(candidates.T)])
            j, i, k = (int(x) for x in candidates[
                int(np.argmin(candidate_margins))])
            baseline_T = np.asarray(nemo_T, dtype=np.float64)
            selected = None
            for exponent in range(-48, 9):
                for sign in (1.0, -1.0):
                    planted_T = np.array(baseline_T, copy=True)
                    delta = np.float64(sign * (2.0 ** exponent))
                    planted_T[j, i, k] = baseline_T[j, i, k] + delta
                    if (planted_T[j, i, k].view(np.uint64)
                            == baseline_T[j, i, k].view(np.uint64)):
                        continue
                    planted_state = base._replace(
                        T=base.T.replace(data=jnp.asarray(planted_T)))
                    _, _, planted_rn2, planted_rn2b = production_closure(
                        planted_state, surface)
                    planted_mask = (
                        np.minimum(np.asarray(planted_rn2),
                                   np.asarray(planted_rn2b)) <= threshold
                    ) & interface_wet
                    moved = int(np.count_nonzero(planted_mask != masks[0]))
                    if moved:
                        selected = {
                            "temperature_index_jik": [j, i, k],
                            "temperature_before_C": float(baseline_T[j, i, k]),
                            "temperature_delta_C": float(delta),
                            "trigger_bits_moved": moved,
                        }
                        break
                if selected is not None:
                    break
            require(selected is not None,
                    "temperature plant could not flip a production trigger")
            _, _, repeat_rn2, repeat_rn2b = production_closure(base, surface)
            repeat_mask = (
                np.minimum(np.asarray(repeat_rn2),
                           np.asarray(repeat_rn2b)) <= threshold
            ) & interface_wet
            selected["untouched_nemo_bits_moved"] = int(np.count_nonzero(
                repeat_mask != masks[0]))
            require(selected["untouched_nemo_bits_moved"] == 0,
                    "temperature plant changed the untouched NEMO arm")
            return {
                "format": "gyre-trigger-state-budget-v1",
                "status": "PLANT-FIRED", "plant": plant,
                "control": selected, "worktree": stamp,
            }

        f_values = {subset: (masks[subset] != nemo_record_mask).astype(
            np.float64) for subset in range(8)}
        bit_shares = _trigger_shapley(f_values)
        margin_shares = _trigger_shapley(margins)
        bit_closure = sum(bit_shares.values()) - (f_values[7] - f_values[0])
        margin_closure = (sum(margin_shares.values())
                          - (margins[7] - margins[0]))
        require(float(np.max(np.abs(bit_closure[interface_wet])))
                <= 4.0 * np.finfo(np.float64).eps,
                f"day {day}: bit Shapley cube does not close")
        require(float(np.max(np.abs(margin_closure[interface_wet])))
                <= 16.0 * np.finfo(np.float64).eps * max(
                    1.0, float(np.max(np.abs(
                        (margins[7] - margins[0])[interface_wet])))),
                f"day {day}: margin Shapley cube does not close")
        full_mismatch = masks[7] != nemo_record_mask
        player_rows = {}
        for name in TRIGGER_PLAYERS:
            bit_values = bit_shares[name][interface_wet]
            margin_values = margin_shares[name][full_mismatch]
            row = {
                "bit_signed_cell_equivalents": float(np.sum(bit_values)),
                "bit_absolute_cell_equivalents": float(
                    np.sum(np.abs(bit_values))),
                "margin_signed_sum_s-2": float(np.sum(margin_values)),
                "margin_abs_sum_s-2": float(np.sum(np.abs(margin_values))),
                "margin_rms_s-2": (float(np.sqrt(np.mean(
                    margin_values ** 2))) if margin_values.size else 0.0),
            }
            player_rows[name] = row
            aggregate[name]["bit_signed_cell_equivalents"] += row[
                "bit_signed_cell_equivalents"]
            aggregate[name]["bit_absolute_cell_equivalents"] += row[
                "bit_absolute_cell_equivalents"]
            aggregate[name]["margin_signed_sum_s-2"] += row[
                "margin_signed_sum_s-2"]
            aggregate[name]["margin_abs_sum_s-2"] += row[
                "margin_abs_sum_s-2"]
            aggregate[name]["margin_sum_squares_s-4"] += float(np.sum(
                margin_values ** 2))
            aggregate[name]["margin_samples"] += int(margin_values.size)
        disagreement_count = int(np.count_nonzero(full_mismatch))
        abs_total = sum(row["bit_absolute_cell_equivalents"]
                        for row in player_rows.values())
        daily_rows.append({
            "day": day, "entry_step": step,
            "full_trigger_disagreement_cells": disagreement_count,
            "model_only_cells": int(np.count_nonzero(
                masks[7] & ~nemo_record_mask)),
            "nemo_only_cells": int(np.count_nonzero(
                nemo_record_mask & ~masks[7])),
            "players": player_rows,
            "absolute_shapley_total": abs_total,
            "cancellation_ratio": (
                abs_total / disagreement_count if disagreement_count else 0.0),
            "bit_closure_max_abs": float(np.max(
                np.abs(bit_closure[interface_wet]))),
            "margin_closure_max_abs_s-2": float(np.max(
                np.abs(margin_closure[interface_wet]))),
            "r3t_inverse": inverse,
        })

    require(plant in (None, "none"),
            f"plant {plant!r} did not fire at day {TRIGGER_DAY_START}")
    require(all(value == 0 for value in endpoint_controls.values()),
            f"trigger endpoint controls moved: {endpoint_controls}")
    inherited = {row["day"]: row["full_trigger_disagreement_cells"]
                 for row in daily_rows}
    require(inherited[180] == 15 and inherited[210] == 11,
            "inherited trigger disagreement headlines are not 15/11")

    ranking = []
    for name in TRIGGER_PLAYERS:
        row = dict(aggregate[name])
        samples = row.pop("margin_samples")
        squares = row.pop("margin_sum_squares_s-4")
        row["owner"] = name
        row["continuous_margin_rms_s-2"] = (
            float(np.sqrt(squares / samples)) if samples else 0.0)
        ranking.append(row)
    ranking.sort(key=lambda row: row["bit_absolute_cell_equivalents"],
                 reverse=True)
    ranking_abs_total = sum(
        row["bit_absolute_cell_equivalents"] for row in ranking)
    for row in ranking:
        row["fraction_of_absolute_shapley"] = (
            row["bit_absolute_cell_equivalents"] / ranking_abs_total
            if ranking_abs_total else 0.0)

    # Direct per-step birth/census.  This is deliberately the admitted stored
    # trajectory, independent of the daily hybrid attribution above.
    timeline = []
    interval_census = None
    first_difference = None
    for step in range(PROCESS_START_STEP, PROCESS_END_STEP + 1):
        record = daily_records.get(step)
        if record is None:
            record = _read_vertical_record(
                nemo_vertical_root
                / f"oracle_trazdf_matrix_kt{step:08d}.bin", step)
        nemo_heat = _vertical_field(record, "avt")[:, :, 1:nlev]
        lego_heat = np.asarray(
            lego_arrays["heat_K"][step - PROCESS_START_STEP])
        nemo_mask = _trigger_masks_from_K(
            nemo_heat, replacement, interface_wet)
        lego_mask = _trigger_masks_from_K(
            lego_heat, replacement, interface_wet)
        disagreement = nemo_mask != lego_mask
        r3t = _vertical_field(record, "r3t_Kbb")
        depth = raw_gdepw * (np.float64(1.0) + r3t[..., None])
        census = _trigger_spatial_census(
            disagreement, lego_mask, nemo_mask, depth, regions)
        timeline.append({"step": step,
                         "day": (step - 1) / STEPS_PER_DAY,
                         **census})
        if interval_census is None:
            interval_census = {
                "total": 0, "model_only": 0, "nemo_only": 0,
                "depth": {name: 0 for name in census["depth"]},
                "region": {name: 0 for name in census["region"]},
            }
        _add_trigger_census(interval_census, census)
        if first_difference is None and census["total"]:
            first_difference = timeline[-1]
    require(first_difference is not None,
            "trigger timeline has no differing cell")

    def day_prediction(day: int) -> dict:
        row = next(item for item in daily_rows if item["day"] == day)
        ordered = sorted(
            TRIGGER_PLAYERS,
            key=lambda name: row["players"][name][
                "bit_absolute_cell_equivalents"], reverse=True)
        total = row["absolute_shapley_total"]
        return {
            "order": ordered,
            "temperature_largest": ordered[0] == "temperature",
            "salinity_second": ordered[1] == "salinity",
            "temperature_at_least_half": (
                row["players"]["temperature"][
                    "bit_absolute_cell_equivalents"] >= 0.5 * total),
            "eta_at_most_one": (
                row["players"]["live_depth"][
                    "bit_absolute_cell_equivalents"] <= 1.0),
        }

    interval_order = [row["owner"] for row in ranking]
    first_depth = max(first_difference["depth"],
                      key=first_difference["depth"].get)
    first_region_third = max(
        ("west_third", "interior_third", "east_third"),
        key=first_difference["region"].get)
    first_latitude = max(
        (f"emp_south_le_{EMP_SPLIT_LAT_DEG}N",
         f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N"),
        key=first_difference["region"].get)
    predictions = {
        "day180": day_prediction(180),
        "day210": day_prediction(210),
        "interval_order": interval_order,
        "interval_temperature_largest": interval_order[0] == "temperature",
        "interval_salinity_second": interval_order[1] == "salinity",
        "interval_temperature_at_least_half": (
            ranking[0]["owner"] == "temperature"
            and ranking[0]["fraction_of_absolute_shapley"] >= 0.5),
        "first_difference_step_is_1081": (
            first_difference["step"] == PROCESS_START_STEP),
        "first_depth_is_0_100m": first_depth == "0_100m",
        "first_region_is_west_third": first_region_third == "west_third",
        "first_latitude_is_south": first_latitude == (
            f"emp_south_le_{EMP_SPLIT_LAT_DEG}N"),
    }
    predictions["all_frozen_magnitude_predictions_confirmed"] = all((
        predictions["day180"]["temperature_largest"],
        predictions["day180"]["salinity_second"],
        predictions["day180"]["temperature_at_least_half"],
        predictions["day180"]["eta_at_most_one"],
        predictions["day210"]["temperature_largest"],
        predictions["day210"]["salinity_second"],
        predictions["day210"]["temperature_at_least_half"],
        predictions["day210"]["eta_at_most_one"],
        predictions["interval_temperature_largest"],
        predictions["interval_salinity_second"],
        predictions["interval_temperature_at_least_half"],
        predictions["first_difference_step_is_1081"],
        predictions["first_depth_is_0_100m"],
        predictions["first_region_is_west_third"],
        predictions["first_latitude_is_south"],
    ))
    return {
        "format": "gyre-trigger-state-budget-v1", "status": "PASS",
        "case": CASE, "execution": "production-step jax.jit",
        "days": [TRIGGER_DAY_START, TRIGGER_DAY_STOP - 1],
        "steps": [PROCESS_START_STEP, PROCESS_END_STEP],
        "n2_threshold_s-2": threshold,
        "evd_replacement_m2_s": replacement,
        "admission": {
            "nemo_record": admission_nemo["status"],
            "lego_trace": admission_lego["status"],
            "nemo_producer_commit": TRIGGER_VERTICAL_RECORD_COMMIT,
            "lego_producer_commit": TRIGGER_LEGO_TRACE_COMMIT,
        },
        "endpoint_controls": endpoint_controls,
        "inherited_disagreements": {"day180": inherited[180],
                                     "day210": inherited[210]},
        "daily_rows": daily_rows, "ranking": ranking,
        "full_interval_trigger_disagreement_visits": int(sum(
            row["full_trigger_disagreement_cells"] for row in daily_rows)),
        "full_interval_cancellation_ratio": (
            ranking_abs_total / sum(
                row["full_trigger_disagreement_cells"]
                for row in daily_rows)
            if any(row["full_trigger_disagreement_cells"]
                   for row in daily_rows) else 0.0),
        "timeline": timeline,
        "first_difference": first_difference,
        "interval_spatial_census": interval_census,
        "predictions": predictions,
        "scope": {
            "true_birth_before_day180": "UNMEASURED",
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC",
        },
        "worktree": stamp,
    }


# ---------------- Round-128 temperature-process trigger decomposition ----
TRIGGER_TEMPERATURE_PROCESS_ROWS = (
    "incoming_before_day180",
    *PROCESS_ROWS,
    "floating_point_closure",
)
TRIGGER_TEMPERATURE_CONTEXT_WEIGHTS6 = (2, 1, 1, 2)


def _temperature_context_marginal(
        candidates: dict[int, np.ndarray],
        references: dict[int, np.ndarray]) -> np.ndarray:
    """Temperature's Shapley marginal across the four S/depth contexts."""
    require(set(candidates) == set(range(4))
            and set(references) == set(range(4)),
            "temperature marginal needs all four salinity/depth contexts")
    shape = np.asarray(candidates[0]).shape
    require(all(np.asarray(values).shape == shape
                for values in (*candidates.values(), *references.values())),
            "temperature marginal context shapes differ")
    numerator = np.zeros(shape, dtype=np.float64)
    for context, weight6 in enumerate(
            TRIGGER_TEMPERATURE_CONTEXT_WEIGHTS6):
        numerator = numerator + np.float64(weight6) * (
            np.asarray(candidates[context], dtype=np.float64)
            - np.asarray(references[context], dtype=np.float64))
    return numerator / np.float64(6.0)


def _temperature_context_marginal_control() -> dict:
    """Synthetic owner/interaction control for the four-context reduction."""
    references = {
        index: np.asarray([float(index), -float(index)])
        for index in range(4)
    }
    increments = {
        0: np.asarray([1.0, -1.0]),
        1: np.asarray([2.0, 3.0]),
        2: np.asarray([-1.0, 5.0]),
        3: np.asarray([4.0, -2.0]),
    }
    candidates = {
        index: references[index] + increments[index]
        for index in range(4)
    }
    observed = _temperature_context_marginal(candidates, references)
    expected = (
        2.0 * increments[0] + increments[1] + increments[2]
        + 2.0 * increments[3]) / 6.0
    require(np.array_equal(observed, expected),
            "temperature four-context Shapley control failed")
    return {"observed": observed.tolist(), "expected": expected.tolist()}


def _temperature_process_boundaries(
        nemo_temperature: np.ndarray, lego_temperature: np.ndarray,
        incoming: np.ndarray,
        cumulative: dict[str, np.ndarray]) -> tuple[
            list[tuple[str, np.ndarray]], dict[str, np.ndarray], float]:
    """Build the frozen source-order temperature telescope for one entry."""
    nemo_temperature = np.asarray(nemo_temperature, dtype=np.float64)
    lego_temperature = np.asarray(lego_temperature, dtype=np.float64)
    incoming = np.asarray(incoming, dtype=np.float64)
    require(nemo_temperature.shape == lego_temperature.shape == incoming.shape,
            "temperature telescope endpoint shapes differ")
    require(set(cumulative) == set(PROCESS_ROWS),
            "temperature telescope process registry is incomplete")

    boundaries = [("nemo_temperature", np.array(
        nemo_temperature, copy=True))]
    deltas: dict[str, np.ndarray] = {}
    previous = boundaries[0][1]
    components = {
        "incoming_before_day180": incoming,
        **{name: np.asarray(cumulative[name], dtype=np.float64)
           for name in PROCESS_ROWS},
    }
    for name in TRIGGER_TEMPERATURE_PROCESS_ROWS[:-1]:
        after = previous + components[name]
        deltas[name] = after - previous
        boundaries.append((name, after))
        previous = after
    raw_residual = lego_temperature - previous
    deltas["floating_point_closure"] = lego_temperature - previous
    boundaries.append(("floating_point_closure", np.array(
        lego_temperature, copy=True)))
    return boundaries, deltas, float(np.max(np.abs(raw_residual)))


def _temperature_process_telescope_control() -> dict:
    """Synthetic registry control with a nonzero closure-association row."""
    shape = (2, 2, 3)
    nemo = np.full(shape, 2.0, dtype=np.float64)
    incoming = np.full(shape, 0.125, dtype=np.float64)
    cumulative = {
        name: np.full(shape, (index + 1) * 0.03125, dtype=np.float64)
        for index, name in enumerate(PROCESS_ROWS)
    }
    expected = np.array(nemo, copy=True)
    expected += incoming
    for name in PROCESS_ROWS:
        expected += cumulative[name]
    lego = np.nextafter(expected, np.inf)
    boundaries, deltas, raw_residual = _temperature_process_boundaries(
        nemo, lego, incoming, cumulative)
    combined = np.zeros(shape, dtype=np.float64)
    for name in TRIGGER_TEMPERATURE_PROCESS_ROWS:
        combined += deltas[name]
    residual = (lego - nemo) - combined
    require(np.max(np.abs(residual)) <= np.finfo(np.float64).eps,
            "synthetic temperature telescope does not close")
    require(np.any(deltas["floating_point_closure"] != 0.0),
            "synthetic temperature closure row is vacuous")
    require(np.array_equal(boundaries[-1][1].view(np.uint64),
                           lego.view(np.uint64)),
            "synthetic temperature telescope loses its exact endpoint")
    return {"raw_residual_max_abs_K": raw_residual,
            "closed_residual_max_abs_K": float(np.max(np.abs(residual)))}


def _weighted_trigger_spatial_census(
        weights: np.ndarray, depth: np.ndarray,
        regions: dict[str, np.ndarray]) -> dict:
    """Sum absolute trigger cell-equivalents in the canonical spatial bins."""
    weights = np.abs(np.asarray(weights, dtype=np.float64))
    depth = np.asarray(depth, dtype=np.float64)
    require(weights.shape == depth.shape,
            "weighted trigger census depth shape differs")
    depth_masks = {
        "0_100m": depth <= 100.0,
        "100_1000m": (depth > 100.0) & (depth <= 1000.0),
        "below_1000m": depth > 1000.0,
    }
    region_names = (
        "west_third", "interior_third", "east_third",
        f"emp_south_le_{EMP_SPLIT_LAT_DEG}N",
        f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N",
    )
    return {
        "absolute_cell_equivalents": float(np.sum(weights)),
        "depth": {name: float(np.sum(weights[selected]))
                  for name, selected in depth_masks.items()},
        "region": {name: float(np.sum(weights[np.broadcast_to(
            regions[name][..., None], weights.shape)]))
            for name in region_names},
    }


def _add_weighted_trigger_census(total: dict, row: dict) -> None:
    total["absolute_cell_equivalents"] += row[
        "absolute_cell_equivalents"]
    for group in ("depth", "region"):
        for name, value in row[group].items():
            total[group][name] += value


def _temperature_process_registry_plant(
        nemo_temperature: np.ndarray, lego_temperature: np.ndarray,
        deltas: dict[str, np.ndarray], wet: np.ndarray) -> dict:
    """Drop one real physics row while retaining the frozen closure row."""
    candidates = []
    for name in PROCESS_ROWS:
        values = np.asarray(deltas[name], dtype=np.float64)
        candidates.append((float(np.max(np.abs(values[wet]))), name))
    magnitude, removed = max(candidates)
    require(magnitude > 0.0,
            "process-registry plant found no nonzero physics row")
    combined = np.zeros_like(nemo_temperature, dtype=np.float64)
    for name in TRIGGER_TEMPERATURE_PROCESS_ROWS:
        if name != removed:
            combined += deltas[name]
    residual = (np.asarray(lego_temperature) - np.asarray(nemo_temperature)
                - combined)
    moved = int(np.count_nonzero(residual[wet] != 0.0))
    require(moved > 0,
            "process-registry plant did not break endpoint closure")
    return {"removed_row": removed,
            "removed_row_max_abs_K": magnitude,
            "endpoint_cells_moved": moved,
            "endpoint_residual_max_abs_K": float(np.max(
                np.abs(residual[wet])))}


def score_trigger_temperature_process_budget(
        nemo_vertical_root: Path, lego_trace_root: Path,
        lego_daily_root: Path, expected_commit: str, *,
        mesh_path: Path = DEFAULT_MESH,
        process_root: Path = DEFAULT_PROCESS_RECORD_ROOT,
        plant: str | None = None) -> dict:
    """Propagate cumulative temperature-process rows through production N2."""
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(plant in (None, "none", "trigger-process-registry",
                      "trigger-process-level"),
            f"unknown trigger-temperature-process plant {plant!r}")
    stamp = worktree_stamp()
    require(stamp["clean"],
            "trigger-temperature-process budget refuses a dirty worktree")
    require(stamp["commit"] == expected_commit,
            "trigger-temperature-process budget commit differs from "
            "--expect-commit")
    _temperature_context_marginal_control()
    _temperature_process_telescope_control()

    # Re-run the admitted Round-127 production cube first.  This re-admits
    # both immutable records and preserves its exact state-owner boundary.
    inherited = score_trigger_state_budget(
        nemo_vertical_root, lego_trace_root, lego_daily_root,
        expected_commit, mesh_path=mesh_path, process_root=process_root)
    inherited_temperature = next(
        row for row in inherited["ranking"] if row["owner"] == "temperature")
    require(inherited["full_interval_trigger_disagreement_visits"] == 782,
            "inherited full-trigger visit count is not 782")
    require(inherited_temperature["bit_signed_cell_equivalents"] == 608.0
            and inherited_temperature["bit_absolute_cell_equivalents"]
            == 1097.0,
            "inherited temperature trigger headline is not +608/1097")

    nemo_vertical_root = Path(nemo_vertical_root)
    lego_trace_root = Path(lego_trace_root)
    lego_daily_root = Path(lego_daily_root)
    process_root = Path(process_root)
    card = build_nemo_testcase_card(CASE)
    gate = _gate()
    wet = gate.expected_masks(card)["T"]
    wet2 = np.any(wet, axis=-1)
    interface_wet = wet[..., :-1] & wet[..., 1:]
    nlev = wet.shape[-1]
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            bn2_intermediate="masked_rn2"))
    carrier, _ = _gyre_checkpoint_state(
        card, nemo_vertical_root / "GYRE_OMIP_L2_P3_00001080_restart.nc")
    H_bathy = np.asarray(carrier.H_bathy.data, dtype=np.float64)
    evd = card.recipe.model_config.physics.convection.enhanced_diffusion
    require(evd is not None,
            "temperature-process budget requires active enhanced diffusion")
    threshold = float(evd.n2_threshold)
    replacement = float(evd.K_conv)

    @jax.jit
    def production_closure(entry, forcing):
        return model.diagnose_vertical_K(entry, card.dt_s, forcing)

    year = _year()
    mesh = year.nemo_operands(mesh_path)
    lat = np.asarray(mesh["gphit"], dtype=np.float64)
    regions = _regions(lat, wet2)
    raw_gdepw = np.asarray(
        card.recipe.z_coord.nemo_gdepw_0, dtype=np.float64)[..., 1:nlev]
    lego_vertical_arrays = _load_vertical_trace_arrays(lego_trace_root)
    lego_process_arrays = _load_lego_trace_arrays(lego_trace_root)

    first_nemo_process = read_process_record(
        process_root
        / f"oracle_process_budget_kt{PROCESS_START_STEP:08d}.bin")
    first_nemo_temperature = np.asarray(
        first_nemo_process["Tbb"][..., :nlev], dtype=np.float64)
    first_lego_temperature = np.asarray(
        lego_process_arrays["Tbb"][0], dtype=np.float64)
    incoming = first_lego_temperature - first_nemo_temperature
    cumulative = {
        name: np.zeros_like(first_nemo_temperature) for name in PROCESS_ROWS
    }
    next_process_step = PROCESS_START_STEP

    aggregate = {}
    for name in TRIGGER_TEMPERATURE_PROCESS_ROWS:
        aggregate[name] = {
            "bit_signed_cell_equivalents": 0.0,
            "bit_absolute_cell_equivalents": 0.0,
            "margin_signed_sum_s-2": 0.0,
            "margin_abs_sum_s-2": 0.0,
            "margin_sum_squares_s-4": 0.0,
            "margin_samples": 0,
            "all_wet_temperature": {
                "upper_sum_squares_K2": 0.0,
                "lower_sum_squares_K2": 0.0,
                "gradient_sum_squares_K2": 0.0,
                "samples": 0,
                "upper_max_abs_K": 0.0,
                "lower_max_abs_K": 0.0,
                "gradient_max_abs_K": 0.0,
            },
            "sensitive_temperature": {
                "upper_sum_squares_K2": 0.0,
                "lower_sum_squares_K2": 0.0,
                "gradient_sum_squares_K2": 0.0,
                "samples": 0,
                "upper_max_abs_K": 0.0,
                "lower_max_abs_K": 0.0,
                "gradient_max_abs_K": 0.0,
            },
            "spatial_absolute": None,
            "first_observed_effect": None,
            "daily": [],
        }

    endpoint = {
        "full_trigger_disagreement_visits": 0,
        "temperature_bit_signed_cell_equivalents": 0.0,
        "temperature_bit_absolute_cell_equivalents": 0.0,
        "temperature_margin_signed_sum_s-2": 0.0,
        "temperature_margin_abs_sum_s-2": 0.0,
    }
    controls = {
        "nemo_endpoint_cells_unequal": 0,
        "lego_endpoint_cells_unequal": 0,
        "rn2_rn2b_cells_unequal": 0,
        "temperature_endpoint_cells_unequal": 0,
        "max_raw_temperature_reconstruction_residual_K": 0.0,
        "max_bit_telescope_residual": 0.0,
        "max_margin_telescope_residual_s-2": 0.0,
        "production_closure_evaluations": 0,
    }
    daily_rows = []

    for day in range(TRIGGER_DAY_START, TRIGGER_DAY_STOP):
        step = STEPS_PER_DAY * day + 1
        while next_process_step < step:
            index = next_process_step - PROCESS_START_STEP
            lego_frame = {
                field: np.asarray(values[index])
                for field, values in lego_process_arrays.items()
            }
            nemo_frame = read_process_record(
                process_root
                / f"oracle_process_budget_kt{next_process_step:08d}.bin")
            lego_rows = lego_process_temperature_rows(lego_frame)
            nemo_rows = process_temperature_rows(nemo_frame)
            for name in PROCESS_ROWS:
                cumulative[name] += lego_rows[name] - nemo_rows[name]
            next_process_step += 1

        record = _read_vertical_record(
            nemo_vertical_root
            / f"oracle_trazdf_matrix_kt{step:08d}.bin", step)
        nemo_T = _vertical_field(record, "T_Kbb_in", nlev)
        nemo_S = _vertical_field(record, "S_Kbb_in", nlev)
        nemo_r3t = _vertical_field(record, "r3t_Kbb")
        nemo_eta, inverse = _eta_for_recorded_r3t(
            nemo_r3t, H_bathy, wet2)
        lego = _load_npz(lego_daily_root / f"day{day:03d}.npz")
        lego_T = np.asarray(
            lego_process_arrays["Tbb"][step - PROCESS_START_STEP],
            dtype=np.float64)
        require(_different_cells(lego_T, lego["T"], wet) == 0,
                f"day {day}: process trace Tbb differs from daily snapshot")
        boundaries, temperature_deltas, raw_residual = (
            _temperature_process_boundaries(
                nemo_T, lego_T, incoming, cumulative))
        raw_residual_wet = float(np.max(np.abs(
            (lego_T - boundaries[-2][1])[wet])))
        controls["max_raw_temperature_reconstruction_residual_K"] = max(
            controls["max_raw_temperature_reconstruction_residual_K"],
            raw_residual_wet)
        controls["temperature_endpoint_cells_unequal"] += _different_cells(
            boundaries[-1][1], lego_T, wet)
        require(raw_residual_wet <= 1.0e-12,
                f"day {day}: raw temperature telescope residual "
                f"{raw_residual_wet:.17e} K exceeds 1e-12 K")

        if day == TRIGGER_DAY_START + 1 \
                and plant == "trigger-process-registry":
            return {
                "format": "gyre-trigger-temperature-process-budget-v1",
                "status": "PLANT-FIRED", "plant": plant,
                "control": _temperature_process_registry_plant(
                    nemo_T, lego_T, temperature_deltas, wet),
                "worktree": stamp,
            }

        base = carrier._replace(
            T=carrier.T.replace(data=jnp.asarray(nemo_T)),
            S=carrier.S.replace(data=jnp.asarray(nemo_S)),
            eta=carrier.eta.replace(data=jnp.asarray(nemo_eta)),
        )
        _, surface = gate._surface_forcings(card, base, step)
        context_values = {
            context: {
                "salinity": (lego["S"] if context & 1 else nemo_S),
                "live_depth": (lego["ssh"] if context & 2 else nemo_eta),
            }
            for context in range(4)
        }

        boundary_outputs = []
        for boundary_name, temperature in boundaries:
            masks = {}
            margins = {}
            for context in range(4):
                values = context_values[context]
                state = carrier._replace(
                    T=carrier.T.replace(data=jnp.asarray(temperature)),
                    S=carrier.S.replace(
                        data=jnp.asarray(values["salinity"])),
                    eta=carrier.eta.replace(
                        data=jnp.asarray(values["live_depth"])),
                )
                _, _, rn2, rn2b = production_closure(state, surface)
                controls["production_closure_evaluations"] += 1
                rn2 = np.asarray(rn2, dtype=np.float64)
                rn2b = np.asarray(rn2b, dtype=np.float64)
                unequal = _different_cells(rn2, rn2b, interface_wet)
                controls["rn2_rn2b_cells_unequal"] += unequal
                require(unequal == 0,
                        f"day {day} boundary {boundary_name} context "
                        f"{context}: rn2/rn2b differ in {unequal} cells")
                minimum = np.minimum(rn2, rn2b)
                masks[context] = (minimum <= threshold) & interface_wet
                margins[context] = minimum - threshold
            boundary_outputs.append({"name": boundary_name,
                                     "masks": masks, "margins": margins})

        nemo_heat = _vertical_field(record, "avt")[:, :, 1:nlev]
        lego_heat = np.asarray(
            lego_vertical_arrays["heat_K"][step - PROCESS_START_STEP])
        nemo_record_mask = _trigger_masks_from_K(
            nemo_heat, replacement, interface_wet)
        lego_record_mask = _trigger_masks_from_K(
            lego_heat, replacement, interface_wet)
        reference_masks = boundary_outputs[0]["masks"]
        reference_margins = boundary_outputs[0]["margins"]
        marginal_outputs = []
        for output in boundary_outputs:
            marginal_outputs.append({
                "name": output["name"],
                "bit": _temperature_context_marginal(
                    {context: (mask != nemo_record_mask).astype(np.float64)
                     for context, mask in output["masks"].items()},
                    {context: (mask != nemo_record_mask).astype(np.float64)
                     for context, mask in reference_masks.items()}),
                "margin": _temperature_context_marginal(
                    output["margins"], reference_margins),
            })

        nemo_endpoint = int(np.count_nonzero(
            reference_masks[0] != nemo_record_mask))
        lego_endpoint = int(np.count_nonzero(
            boundary_outputs[-1]["masks"][3] != lego_record_mask))
        controls["nemo_endpoint_cells_unequal"] += nemo_endpoint
        controls["lego_endpoint_cells_unequal"] += lego_endpoint
        require(nemo_endpoint == 0 and lego_endpoint == 0,
                f"day {day}: process endpoint masks differ from records "
                f"(NEMO={nemo_endpoint}, legoESM={lego_endpoint})")

        if day == TRIGGER_DAY_START and plant == "trigger-process-level":
            candidates = np.argwhere(interface_wet)
            candidate_margins = np.abs(
                reference_margins[0][tuple(candidates.T)])
            j, i, k = (int(value) for value in candidates[
                int(np.argmin(candidate_margins))])
            selected = None
            for level_name, level in (("upper", k), ("lower", k + 1)):
                for exponent in range(-48, 9):
                    for sign in (1.0, -1.0):
                        planted_T = np.array(nemo_T, copy=True)
                        delta = np.float64(sign * (2.0 ** exponent))
                        planted_T[j, i, level] = (
                            nemo_T[j, i, level] + delta)
                        if (planted_T[j, i, level].view(np.uint64)
                                == nemo_T[j, i, level].view(np.uint64)):
                            continue
                        planted_state = base._replace(
                            T=base.T.replace(data=jnp.asarray(planted_T)))
                        _, _, planted_rn2, planted_rn2b = production_closure(
                            planted_state, surface)
                        planted_mask = (
                            np.minimum(np.asarray(planted_rn2),
                                       np.asarray(planted_rn2b)) <= threshold
                        ) & interface_wet
                        moved = int(np.count_nonzero(
                            planted_mask != reference_masks[0]))
                        if moved:
                            selected = {
                                "interface_index_jik": [j, i, k],
                                "consumed_level": level_name,
                                "temperature_level_index": level,
                                "temperature_before_C": float(
                                    nemo_T[j, i, level]),
                                "temperature_delta_C": float(delta),
                                "trigger_bits_moved": moved,
                            }
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    break
            require(selected is not None,
                    "consumed-level plant could not move a production "
                    "threshold bit")
            _, _, repeat_rn2, repeat_rn2b = production_closure(base, surface)
            repeat_mask = (
                np.minimum(np.asarray(repeat_rn2),
                           np.asarray(repeat_rn2b)) <= threshold
            ) & interface_wet
            selected["untouched_endpoint_bits_moved"] = int(
                np.count_nonzero(repeat_mask != reference_masks[0]))
            require(selected["untouched_endpoint_bits_moved"] == 0,
                    "consumed-level plant changed the untouched endpoint")
            return {
                "format": "gyre-trigger-temperature-process-budget-v1",
                "status": "PLANT-FIRED", "plant": plant,
                "control": selected, "worktree": stamp,
            }

        full_mismatch = (
            boundary_outputs[-1]["masks"][3] != nemo_record_mask)
        endpoint_bit = marginal_outputs[-1]["bit"]
        endpoint_margin = marginal_outputs[-1]["margin"]
        inherited_day = inherited["daily_rows"][
            day - TRIGGER_DAY_START]["players"]["temperature"]
        observed_day = {
            "bit_signed_cell_equivalents": float(np.sum(
                endpoint_bit[interface_wet])),
            "bit_absolute_cell_equivalents": float(np.sum(
                np.abs(endpoint_bit[interface_wet]))),
            "margin_signed_sum_s-2": float(np.sum(
                endpoint_margin[full_mismatch])),
            "margin_abs_sum_s-2": float(np.sum(
                np.abs(endpoint_margin[full_mismatch]))),
        }
        for key, value in observed_day.items():
            require(abs(value - inherited_day[key]) <= 1.0e-18,
                    f"day {day}: temperature endpoint {key} {value!r} "
                    f"differs from inherited {inherited_day[key]!r}")
        endpoint["full_trigger_disagreement_visits"] += int(
            np.count_nonzero(full_mismatch))
        endpoint["temperature_bit_signed_cell_equivalents"] += observed_day[
            "bit_signed_cell_equivalents"]
        endpoint["temperature_bit_absolute_cell_equivalents"] += observed_day[
            "bit_absolute_cell_equivalents"]
        endpoint["temperature_margin_signed_sum_s-2"] += observed_day[
            "margin_signed_sum_s-2"]
        endpoint["temperature_margin_abs_sum_s-2"] += observed_day[
            "margin_abs_sum_s-2"]

        bit_rows = {}
        margin_rows = {}
        for index, name in enumerate(TRIGGER_TEMPERATURE_PROCESS_ROWS, 1):
            bit_rows[name] = (marginal_outputs[index]["bit"]
                              - marginal_outputs[index - 1]["bit"])
            margin_rows[name] = (marginal_outputs[index]["margin"]
                                 - marginal_outputs[index - 1]["margin"])
        bit_residual = sum(bit_rows.values()) - endpoint_bit
        margin_residual = sum(margin_rows.values()) - endpoint_margin
        bit_max = float(np.max(np.abs(bit_residual[interface_wet])))
        margin_max = float(np.max(np.abs(
            margin_residual[interface_wet])))
        controls["max_bit_telescope_residual"] = max(
            controls["max_bit_telescope_residual"], bit_max)
        controls["max_margin_telescope_residual_s-2"] = max(
            controls["max_margin_telescope_residual_s-2"], margin_max)
        require(bit_max <= 8.0 * np.finfo(np.float64).eps,
                f"day {day}: propagated bit telescope does not close")
        require(margin_max <= 32.0 * np.finfo(np.float64).eps,
                f"day {day}: propagated margin telescope does not close")

        sensitive = np.zeros_like(interface_wet)
        for values in bit_rows.values():
            sensitive |= values != 0.0
        depth = raw_gdepw * (np.float64(1.0) + nemo_r3t[..., None])
        day_process_rows = {}
        for name in TRIGGER_TEMPERATURE_PROCESS_ROWS:
            bit_values = bit_rows[name]
            margin_values = margin_rows[name]
            delta_T = temperature_deltas[name]
            upper = delta_T[..., :-1]
            lower = delta_T[..., 1:]
            gradient = upper - lower
            row = aggregate[name]
            signed = float(np.sum(bit_values[interface_wet]))
            absolute = float(np.sum(np.abs(bit_values[interface_wet])))
            margin_signed = float(np.sum(margin_values[full_mismatch]))
            margin_absolute = float(np.sum(
                np.abs(margin_values[full_mismatch])))
            row["bit_signed_cell_equivalents"] += signed
            row["bit_absolute_cell_equivalents"] += absolute
            row["margin_signed_sum_s-2"] += margin_signed
            row["margin_abs_sum_s-2"] += margin_absolute
            row["margin_sum_squares_s-4"] += float(np.sum(
                margin_values[full_mismatch] ** 2))
            row["margin_samples"] += int(np.count_nonzero(full_mismatch))

            for label, selected_mask in (("all_wet_temperature",
                                           interface_wet),
                                          ("sensitive_temperature",
                                           sensitive & interface_wet)):
                metrics = row[label]
                samples = int(np.count_nonzero(selected_mask))
                metrics["samples"] += samples
                if samples:
                    metrics["upper_sum_squares_K2"] += float(np.sum(
                        upper[selected_mask] ** 2))
                    metrics["lower_sum_squares_K2"] += float(np.sum(
                        lower[selected_mask] ** 2))
                    metrics["gradient_sum_squares_K2"] += float(np.sum(
                        gradient[selected_mask] ** 2))
                    metrics["upper_max_abs_K"] = max(
                        metrics["upper_max_abs_K"], float(np.max(
                            np.abs(upper[selected_mask]))))
                    metrics["lower_max_abs_K"] = max(
                        metrics["lower_max_abs_K"], float(np.max(
                            np.abs(lower[selected_mask]))))
                    metrics["gradient_max_abs_K"] = max(
                        metrics["gradient_max_abs_K"], float(np.max(
                            np.abs(gradient[selected_mask]))))

            spatial = _weighted_trigger_spatial_census(
                bit_values, depth, regions)
            if row["spatial_absolute"] is None:
                row["spatial_absolute"] = {
                    "absolute_cell_equivalents": 0.0,
                    "depth": {key: 0.0 for key in spatial["depth"]},
                    "region": {key: 0.0 for key in spatial["region"]},
                }
            _add_weighted_trigger_census(row["spatial_absolute"], spatial)
            effect = (bit_values != 0.0) & interface_wet
            if row["first_observed_effect"] is None and bool(np.any(effect)):
                census = _trigger_spatial_census(
                    effect, bit_values > 0.0, bit_values < 0.0,
                    depth, regions)
                census["absolute_cell_equivalents"] = absolute
                row["first_observed_effect"] = {
                    "day": day, "entry_step": step, **census,
                }
            day_row = {
                "bit_signed_cell_equivalents": signed,
                "bit_absolute_cell_equivalents": absolute,
                "margin_signed_sum_s-2": margin_signed,
                "margin_abs_sum_s-2": margin_absolute,
                "threshold_effect_cells": int(np.count_nonzero(effect)),
                "upper_rms_K": (float(np.sqrt(np.mean(
                    upper[sensitive & interface_wet] ** 2)))
                    if bool(np.any(sensitive & interface_wet)) else 0.0),
                "lower_rms_K": (float(np.sqrt(np.mean(
                    lower[sensitive & interface_wet] ** 2)))
                    if bool(np.any(sensitive & interface_wet)) else 0.0),
                "gradient_rms_K": (float(np.sqrt(np.mean(
                    gradient[sensitive & interface_wet] ** 2)))
                    if bool(np.any(sensitive & interface_wet)) else 0.0),
            }
            row["daily"].append({"day": day, **day_row})
            day_process_rows[name] = day_row

        daily_rows.append({
            "day": day, "entry_step": step,
            "full_trigger_disagreement_cells": int(np.count_nonzero(
                full_mismatch)),
            "temperature_endpoint": observed_day,
            "temperature_sensitive_interfaces": int(np.count_nonzero(
                sensitive & interface_wet)),
            "process_rows": day_process_rows,
            "raw_temperature_reconstruction_residual_K": raw_residual_wet,
            "r3t_inverse": inverse,
        })

    require(plant in (None, "none"),
            f"plant {plant!r} did not fire")
    require(controls["max_raw_temperature_reconstruction_residual_K"]
            <= 1.0e-12,
            "temperature telescope exceeded its preregistered residual")
    require(controls["temperature_endpoint_cells_unequal"] == 0,
            "exact temperature endpoints are not bit-identical")
    require(controls["nemo_endpoint_cells_unequal"] == 0
            and controls["lego_endpoint_cells_unequal"] == 0
            and controls["rn2_rn2b_cells_unequal"] == 0,
            f"production endpoint controls moved: {controls}")
    frozen_endpoint = {
        "full_trigger_disagreement_visits": 782,
        "temperature_bit_signed_cell_equivalents": 608.0,
        "temperature_bit_absolute_cell_equivalents": 1097.0,
        "temperature_margin_signed_sum_s-2": -2.1584924289917324e-05,
        "temperature_margin_abs_sum_s-2": 4.2348751127228244e-04,
    }
    for key, expected in frozen_endpoint.items():
        require(abs(endpoint[key] - expected) <= 1.0e-18,
                f"temperature-process endpoint {key} {endpoint[key]!r} "
                f"differs from frozen {expected!r}")

    ranking = []
    for name in TRIGGER_TEMPERATURE_PROCESS_ROWS:
        values = aggregate[name]
        row = {
            "owner": name,
            "bit_signed_cell_equivalents": values[
                "bit_signed_cell_equivalents"],
            "bit_absolute_cell_equivalents": values[
                "bit_absolute_cell_equivalents"],
            "margin_signed_sum_s-2": values["margin_signed_sum_s-2"],
            "margin_abs_sum_s-2": values["margin_abs_sum_s-2"],
            "continuous_margin_rms_s-2": (
                float(np.sqrt(values["margin_sum_squares_s-4"]
                              / values["margin_samples"]))
                if values["margin_samples"] else 0.0),
            "first_observed_effect": values["first_observed_effect"],
            "spatial_absolute": values["spatial_absolute"],
            "daily": values["daily"],
        }
        for label in ("all_wet_temperature", "sensitive_temperature"):
            metrics = values[label]
            samples = metrics["samples"]
            row[label] = {
                "samples": samples,
                "upper_rms_K": (float(np.sqrt(
                    metrics["upper_sum_squares_K2"] / samples))
                    if samples else 0.0),
                "lower_rms_K": (float(np.sqrt(
                    metrics["lower_sum_squares_K2"] / samples))
                    if samples else 0.0),
                "gradient_rms_K": (float(np.sqrt(
                    metrics["gradient_sum_squares_K2"] / samples))
                    if samples else 0.0),
                "upper_max_abs_K": metrics["upper_max_abs_K"],
                "lower_max_abs_K": metrics["lower_max_abs_K"],
                "gradient_max_abs_K": metrics["gradient_max_abs_K"],
            }
        ranking.append(row)
    ranking.sort(key=lambda row: row["bit_absolute_cell_equivalents"],
                 reverse=True)
    signed_sum = float(sum(
        row["bit_signed_cell_equivalents"] for row in ranking))
    absolute_process_total = float(sum(
        row["bit_absolute_cell_equivalents"] for row in ranking))

    by_name = {row["owner"]: row for row in ranking}
    vertical_spatial = by_name["vertical_diffusion"]["spatial_absolute"]
    vertical_depth = max(
        vertical_spatial["depth"], key=vertical_spatial["depth"].get)
    vertical_longitude = max(
        ("west_third", "interior_third", "east_third"),
        key=vertical_spatial["region"].get)
    vertical_latitude = max(
        (f"emp_south_le_{EMP_SPLIT_LAT_DEG}N",
         f"emp_north_gt_{EMP_SPLIT_LAT_DEG}N"),
        key=vertical_spatial["region"].get)
    first_nonincoming_process_day = min(
        (by_name[name]["first_observed_effect"]["day"]
         for name in PROCESS_ROWS
         if by_name[name]["first_observed_effect"] is not None),
        default=None)
    ordered_names = [row["owner"] for row in ranking]
    predictions = {
        "vertical_diffusion_largest": ordered_names[0]
        == "vertical_diffusion",
        "lateral_diffusion_second": len(ordered_names) > 1
        and ordered_names[1] == "lateral_diffusion",
        "vertical_depth_is_0_100m": vertical_depth == "0_100m",
        "vertical_longitude_is_west_third": vertical_longitude
        == "west_third",
        "vertical_latitude_is_south": vertical_latitude
        == f"emp_south_le_{EMP_SPLIT_LAT_DEG}N",
        "incoming_first_observed_day_is_180": (
            by_name["incoming_before_day180"]["first_observed_effect"]
            is not None
            and by_name["incoming_before_day180"]
            ["first_observed_effect"]["day"] == 180),
        "first_nonincoming_process_day_is_181":
            first_nonincoming_process_day == 181,
    }
    predictions["all_frozen_magnitude_predictions_confirmed"] = all(
        predictions.values())

    report = {
        "format": "gyre-trigger-temperature-process-budget-v1",
        "status": "PASS", "case": CASE,
        "execution": "production-step jax.jit",
        "days": [TRIGGER_DAY_START, TRIGGER_DAY_STOP - 1],
        "steps": [PROCESS_START_STEP, PROCESS_END_STEP],
        "process_order": list(TRIGGER_TEMPERATURE_PROCESS_ROWS),
        "n2_threshold_s-2": threshold,
        "evd_replacement_m2_s": replacement,
        "inherited_round127": {
            "full_trigger_disagreement_visits": inherited[
                "full_interval_trigger_disagreement_visits"],
            "temperature": inherited_temperature,
            "endpoint_controls": inherited["endpoint_controls"],
        },
        "endpoint": endpoint,
        "controls": controls,
        "daily_rows": daily_rows,
        "ranking": ranking,
        "signed_process_sum_cell_equivalents": signed_sum,
        "signed_process_minus_temperature_endpoint": (
            signed_sum
            - endpoint["temperature_bit_signed_cell_equivalents"]),
        "absolute_process_cell_equivalents": absolute_process_total,
        "process_cancellation_ratio_to_temperature_endpoint": (
            absolute_process_total
            / endpoint["temperature_bit_absolute_cell_equivalents"]),
        "first_nonincoming_process_day": first_nonincoming_process_day,
        "predictions": predictions,
        "scope": {
            "true_incoming_birth_before_day180": "UNMEASURED",
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC",
        },
        "worktree": stamp,
    }
    print("\nEVD TEMPERATURE-PROCESS RANKING -- production JIT")
    print(f"  {'rank':>4s} {'owner':>30s} {'signed cells':>14s} "
          f"{'absolute cells':>16s} {'first day':>10s}")
    for rank, row in enumerate(ranking, 1):
        first = row["first_observed_effect"]
        first_day = "none" if first is None else str(first["day"])
        print(f"  {rank:4d} {row['owner']:>30s} "
              f"{row['bit_signed_cell_equivalents']:14.6f} "
              f"{row['bit_absolute_cell_equivalents']:16.6f} "
              f"{first_day:>10s}")
    print(f"  endpoint signed/absolute +{endpoint['temperature_bit_signed_cell_equivalents']:.6f}/"
          f"{endpoint['temperature_bit_absolute_cell_equivalents']:.6f}; "
          f"process absolute {absolute_process_total:.6f}")
    return report


# ---------------- Round-136 developed-state, equal-input process walk -----
DEVELOPED_REQUESTED_DAYS = (30, 90, 180, 240)
DEVELOPED_ENTRY_STEP = 1080
DEVELOPED_PROCESS_STEP = 1081
DEVELOPED_VERTICAL_RECORD_COMMIT = (
    "4cac617cd928007506f2de7ccb098f87e04204d1")
DEFAULT_DEVELOPED_DAILY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round132/"
    "oracle_daily_restarts")
DEFAULT_DEVELOPED_VERTICAL_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round125/"
    "oracle_vertical_decomposition")
DEVELOPED_BOUNDARY_FIELDS = {
    "geometry": "B0",
    "advection": "Badv",
    "surface_boundary": "Bsbc",
    "shortwave": "Bqsr",
    "lateral_diffusion": "Bldf",
    "vertical_diffusion": "Taa",
}
DEVELOPED_GEOMETRY_OPERANDS = ("q_Kbb", "q_Kmm", "q_Kaa")
DEVELOPED_BRANCHES = ("fct_nonosc", "evd_replacement", "tke_floors")
DEVELOPED_FCT_COMMON_FIELDS = (
    "p2dt", "transport_u", "transport_v", "transport_w", "e3t_3d",
    "r3t_Kbb", "r3t_Kmm", "r3t_Kaa", "tmask", "wmask", "r1_e1e2t",
)
DEVELOPED_FCT_TRACER_FIELDS = (
    "base", "now", "rhs_entry", "first_u", "first_v", "first_w",
    "first_div", "midpoint", "average_u", "average_v", "average_w",
    "upstream_div", "rhs_after_up", "anti_pre_u", "anti_pre_v",
    "anti_pre_w", "coef_u", "coef_v", "coef_w", "anti_post_u",
    "anti_post_v", "anti_post_w", "final_div", "divisor", "rhs_final",
)
DEVELOPED_TRANSPORT_U_ROWS = (
    "un_adv", "r1_hu_0", "one_plus_r3u_Kmm", "live_inverse_depth", "uu_b_Kmm",
    "zub", "e2u", "e3u_0", "umask", "live_e3u_Kmm", "uu_Kmm",
    "corrected_u", "zFu",
)
DEVELOPED_STAGE2_SPLIT_ROWS = (
    "calibration_nemo_reprojection",
    "production_baseline",
    "nemo_depth_mean_substituted",
    "nemo_depth_mean_substituted_unmasked_weight",
)


def developed_record_availability() -> list[dict]:
    """Freeze what the admitted record can and cannot measure exactly."""
    return [
        {"day": 30, "entry_step": 180, "process_step": None,
         "status": "UNAVAILABLE_BY_RECORD"},
        {"day": 90, "entry_step": 540, "process_step": None,
         "status": "UNAVAILABLE_BY_RECORD"},
        {"day": 180, "entry_step": DEVELOPED_ENTRY_STEP,
         "process_step": DEVELOPED_PROCESS_STEP, "status": "MEASURED"},
        {"day": 240, "entry_step": 1440, "process_step": None,
         "status": "UNAVAILABLE_BY_RECORD",
         "reason": "frame 1440 begins from unrecorded full step-1439 state"},
    ]


def _process_cumulative_boundaries(record: dict) -> dict[str, np.ndarray]:
    """Materialize the six cumulative values bracketed by the R123 writes."""
    nlev = record["Tbb"].shape[-1] - 1
    tbb = np.asarray(record["Tbb"][..., :nlev], dtype=np.float64)
    qbb = (1.0 + np.asarray(record["r3t_Kbb"],
                            dtype=np.float64))[..., None]
    qmm = (1.0 + np.asarray(record["r3t_Kmm"],
                            dtype=np.float64))[..., None]
    qaa = (1.0 + np.asarray(record["r3t_Kaa"],
                            dtype=np.float64))[..., None]
    base = qbb * tbb

    def accumulated(field: str) -> np.ndarray:
        rhs = np.asarray(record[field][..., :nlev], dtype=np.float64)
        return (base + record["rDt"] * qmm * rhs) / qaa

    return {
        "geometry": base / qaa,
        "advection": accumulated("rhs_after_advection"),
        "surface_boundary": accumulated("rhs_after_surface_boundary"),
        "shortwave": accumulated("rhs_after_shortwave"),
        "lateral_diffusion": accumulated("rhs_after_lateral_diffusion"),
        "vertical_diffusion": np.asarray(
            record["Taa"][..., :nlev], dtype=np.float64),
    }


def _bit_mismatch_count(actual, expected, mask=None) -> int:
    # Restart variables can be transposed NetCDF views.  Compare their value
    # bytes in logical C order rather than requiring their source strides to
    # make the final axis contiguous.
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    require(actual.shape == expected.shape,
            f"bit comparison shapes differ: {actual.shape} vs "
            f"{expected.shape}")
    different = actual.view(np.uint8).reshape(actual.shape + (-1,)) != (
        expected.view(np.uint8).reshape(expected.shape + (-1,)))
    different = np.any(different, axis=-1)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        require(selected.shape == actual.shape,
                "bit comparison mask shape differs")
        different &= selected
    return int(np.count_nonzero(different))


def _score_developed_row(actual, expected, mask: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == mask.shape,
            "developed-row arrays/mask have unequal shapes")
    bits = actual.view(np.uint64) != expected.view(np.uint64)
    differing = bits & mask
    delta = actual[mask] - expected[mask]
    first = None
    if np.any(differing):
        first = [int(value) for value in np.argwhere(differing)[0]]
    return {
        "cells_scored": int(np.count_nonzero(mask)),
        "cells_unequal": int(np.count_nonzero(differing)),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "first_unequal_jik": first,
        "bit_exact": not bool(np.any(differing)),
    }


def _rank_developed_process_rows(increment_rows: dict[str, dict]) -> list[dict]:
    """Rank the complete compiled-order process registry by one-step RMS."""
    require(set(increment_rows) == set(PROCESS_ROWS),
            "developed-state process ranking input is incomplete")
    ranked = []
    for rank, name in enumerate(
            sorted(PROCESS_ROWS,
                   key=lambda row: increment_rows[row]["rms"],
                   reverse=True), 1):
        row = increment_rows[name]
        ranked.append({
            "rank": rank,
            "name": name,
            "cells_unequal": row["cells_unequal"],
            "max_abs_temperature_contribution_K": row["max_abs"],
            "rms_temperature_contribution_K": row["rms"],
            "rms_effective_tendency_K_s": row["rms"] / DT_S,
        })
    return ranked


def _interface_cells(mask: np.ndarray, nlev: int) -> np.ndarray:
    """Project an active W-interface map onto both adjacent T cells."""
    mask = np.asarray(mask, dtype=bool)
    require(mask.shape[-1] == nlev - 1,
            "interface activity does not have nlev-1 levels")
    cells = np.zeros(mask.shape[:-1] + (nlev,), dtype=bool)
    cells[..., :-1] |= mask
    cells[..., 1:] |= mask
    return cells


def _branch_census(mask: np.ndarray, unequal: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    unequal = np.asarray(unequal, dtype=bool)
    require(mask.shape == unequal.shape,
            "branch/unequal activity shapes differ")
    active = int(np.count_nonzero(mask))
    overlap = int(np.count_nonzero(mask & unequal))
    return {
        "active_cells": active,
        "overlap_with_first_non_bit_cells": overlap,
        "overlap_fraction_of_first_non_bit": (
            float(overlap / np.count_nonzero(unequal))
            if np.count_nonzero(unequal) else 0.0),
    }


def _validate_developed_registry(report: dict, plant: str | None = None) -> None:
    """Require every requested day, process boundary, and branch family."""
    candidate = json.loads(json.dumps(report))
    if plant == "missing-day":
        candidate["availability"] = candidate["availability"][:-1]
    elif plant == "missing-process-row":
        candidate["cumulative_boundaries"].pop(PROCESS_ROWS[-1], None)
    elif plant == "missing-branch":
        candidate["branches"].pop(DEVELOPED_BRANCHES[-1], None)
    elif plant == "missing-ranking-row":
        candidate["process_ranking"] = candidate["process_ranking"][:-1]
    elif plant not in (None, "none"):
        raise GateError(f"unknown developed registry plant {plant!r}")
    require(
        [row["day"] for row in candidate["availability"]]
        == list(DEVELOPED_REQUESTED_DAYS),
        "developed-state requested-day registry is incomplete")
    require(set(candidate["cumulative_boundaries"]) == set(PROCESS_ROWS),
            "developed-state process-row registry is incomplete")
    require(set(candidate["geometry_operands"])
            == set(DEVELOPED_GEOMETRY_OPERANDS),
            "developed-state geometry-operand registry is incomplete")
    require(set(candidate["increment_rows"]) == set(PROCESS_ROWS),
            "developed-state increment-row registry is incomplete")
    require(set(candidate["branches"]) == set(DEVELOPED_BRANCHES),
            "developed-state branch registry is incomplete")
    require(candidate["first_non_bit_process_call"] in PROCESS_ROWS[1:],
            "developed-state first process call is unregistered")
    ranking = candidate["process_ranking"]
    require({row["name"] for row in ranking} == set(PROCESS_ROWS),
            "developed-state process ranking is incomplete")
    require([row["rank"] for row in ranking]
            == list(range(1, len(PROCESS_ROWS) + 1)),
            "developed-state process ranking is not contiguous")


def _developed_registry_plant(plant: str) -> dict:
    synthetic = {
        "availability": developed_record_availability(),
        "cumulative_boundaries": {name: {} for name in PROCESS_ROWS},
        "geometry_operands": {
            name: {} for name in DEVELOPED_GEOMETRY_OPERANDS},
        "increment_rows": {name: {} for name in PROCESS_ROWS},
        "branches": {name: {} for name in DEVELOPED_BRANCHES},
        "first_non_bit_process_call": "advection",
        "process_ranking": [
            {"rank": rank, "name": name}
            for rank, name in enumerate(PROCESS_ROWS, 1)],
    }
    try:
        _validate_developed_registry(synthetic, plant=plant)
    except GateError as error:
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": str(error)}
    raise GateError(f"developed registry plant {plant!r} stayed green")


def _developed_state_inventory(state) -> list[dict]:
    mapped = {
        "u", "v", "T", "S", "eta", "uu_b", "vv_b", "tke",
        "tke_avm", "tke_avt", "tke_dissl", "tke_avm_surface", "bt_hist",
    }
    static = {"H_bathy", "land_mask", "u_mask", "v_mask"}
    recomputed = {"w"}
    rows = []
    for name, value in zip(state._fields, state, strict=True):
        if name in mapped:
            classification = "mapped_from_restart"
            require(value is not None,
                    f"restart-mapped state leaf {name} is None")
        elif name in static:
            classification = "static_card_geometry"
            require(value is not None, f"static state leaf {name} is None")
        elif name in recomputed:
            classification = "diagnostic_recomputed_in_step"
            require(value is not None,
                    f"recomputed diagnostic state leaf {name} is None")
        else:
            require(value is None,
                    f"unclassified live developed-state leaf {name}")
            classification = "inert_none_for_resolved_card"
        if isinstance(value, tuple):
            shape = [list(np.asarray(item).shape) for item in value]
            dtype = [str(np.asarray(item).dtype) for item in value]
        elif value is None:
            shape = None
            dtype = None
        else:
            array = np.asarray(getattr(value, "data", value))
            shape = list(array.shape)
            dtype = str(array.dtype)
        rows.append({"name": name, "classification": classification,
                     "shape": shape, "dtype": dtype})
    require({row["name"] for row in rows} == set(state._fields),
            "developed-state inventory omitted a state leaf")
    return rows


def _developed_fct_record(root: Path) -> dict:
    """Read and map the admitted R153 stream onto legoESM's owned domain."""
    gate153 = _load(
        "nemo_testcase_l2_gyre_round153_developed_fct_gate",
        "nemo_testcase_l2_gyre_round153_developed_fct_gate.py")
    admission_path = root / "round153_developed_fct_admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    require(admission["status"] == "PASS", "Round-153 FCT record is not admitted")
    record = gate153.self_describing.read_self_describing_record(
        root / gate153.RECORD, magic_expected=gate153.MAGIC,
        header_expected=gate153.HEADER, rows_expected=gate153.EXPECTED_ROWS)
    require(record["sha256"] == admission["record"]["sha256"],
            "Round-153 FCT record differs from its admission")
    fields = record["fields"]

    def value(name):
        return np.asarray(fields[name]["values"], dtype=np.float64)

    def owned3(name, islice, jslice):
        return np.ascontiguousarray(
            value(name)[islice, jslice].transpose(1, 0, 2))

    def owned2(name, islice, jslice):
        return np.ascontiguousarray(value(name)[islice, jslice, 0].T)

    common = {
        "p2dt": np.asarray([value("p2dt").item()], dtype=np.float64),
        "transport_u": owned3("transport_u", slice(1, 34), slice(2, 24)),
        "transport_v": owned3("transport_v", slice(2, 34), slice(1, 24)),
        "transport_w": np.pad(
            owned3("transport_w", slice(1, 33), slice(1, 23)),
            ((0, 0), (0, 0), (0, 1))),
        "e3t_3d": owned3("e3t_3d", slice(1, 33), slice(1, 23)),
        "r3t_Kbb": owned2("r3t_Kbb", slice(1, 33), slice(1, 23)),
        "r3t_Kmm": owned2("r3t_Kmm", slice(1, 33), slice(1, 23)),
        "r3t_Kaa": owned2("r3t_Kaa", slice(1, 33), slice(1, 23)),
        "tmask": owned3("tmask", slice(1, 33), slice(1, 23)),
        "wmask": owned3("wmask", slice(1, 33), slice(1, 23)),
        "r1_e1e2t": owned2("r1_e1e2t", slice(1, 33), slice(1, 23)),
    }
    mappings = {
        "base": (slice(2, 34), slice(2, 24)),
        "now": (slice(2, 34), slice(2, 24)),
        "rhs_entry": (slice(None), slice(None)),
        "first_u": (slice(1, 34), slice(2, 24)),
        "first_v": (slice(2, 34), slice(1, 24)),
        "first_w": (slice(1, 33), slice(1, 23)),
        "first_div": (slice(1, 33), slice(1, 23)),
        "midpoint": (slice(1, 33), slice(1, 23)),
        "average_u": (slice(None), slice(1, 23)),
        "average_v": (slice(1, 33), slice(None)),
        "average_w": (slice(1, 33), slice(1, 23)),
        "upstream_div": (slice(None), slice(None)),
        "rhs_after_up": (slice(None), slice(None)),
        "anti_pre_u": (slice(None), slice(1, 23)),
        "anti_pre_v": (slice(1, 33), slice(None)),
        "anti_pre_w": (slice(None), slice(None)),
        "coef_u": (slice(None), slice(1, 23)),
        "coef_v": (slice(1, 33), slice(None)),
        "coef_w": (slice(None), slice(None)),
        "anti_post_u": (slice(None), slice(1, 23)),
        "anti_post_v": (slice(1, 33), slice(None)),
        "anti_post_w": (slice(None), slice(None)),
        "final_div": (slice(None), slice(None)),
        "divisor": (slice(None), slice(None)),
        "rhs_final": (slice(None), slice(None)),
    }
    tracers = {
        tracer: {
            name: owned3(f"{name}_{tracer}", *slices)
            for name, slices in mappings.items()
        }
        for tracer in ("T", "S")
    }
    classified = set(DEVELOPED_FCT_COMMON_FIELDS)
    classified.update(
        f"{name}_{tracer}" for tracer in ("T", "S")
        for name in DEVELOPED_FCT_TRACER_FIELDS)
    require(classified == set(fields), "Round-153 FCT field registry is incomplete")
    return {
        "sha256": record["sha256"], "admission": admission,
        "common": common, "tracers": tracers,
        "field_count": len(fields),
    }


def _developed_transport_record(root: Path) -> dict:
    """Read the admitted R154 stage-3 transport record on legoESM's U grid."""
    gate154 = _load(
        "nemo_testcase_l2_gyre_round154_developed_transport_gate",
        "nemo_testcase_l2_gyre_round154_developed_transport_gate.py")
    admission_path = root / "round154_developed_transport_admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    require(admission["status"] == "PASS",
            "Round-154 transport record is not admitted")
    record = gate154.read_record(root / gate154.RECORD)
    require(record["sha256"] == admission["record"]["sha256"],
            "Round-154 transport record differs from its admission")
    fields = record["fields"]

    def owned2(name):
        return np.ascontiguousarray(fields[name][1:34, 2:24].T)

    def owned3(name):
        return np.ascontiguousarray(
            fields[name][1:34, 2:24, :30].transpose(1, 0, 2))

    recorded_r3u = owned2("r3u_Kmm")
    oracle = {
        "un_adv": owned2("un_adv"),
        "r1_hu_0": owned2("r1_hu_0"),
        "one_plus_r3u_Kmm": np.float64(1.0) + recorded_r3u,
        "uu_b_Kmm": owned2("uu_b_Kmm"),
        "zub": owned2("zub"),
        "e2u": owned2("e2u"),
        "e3u_0": owned3("e3u_0"),
        "umask": owned3("umask"),
        "uu_Kmm": owned3("uu_Kmm"),
        "zFu": owned3("zFu"),
    }
    b = np.float64
    oracle["live_inverse_depth"] = (
        oracle["r1_hu_0"] / oracle["one_plus_r3u_Kmm"])
    oracle["live_e3u_Kmm"] = (
        oracle["e3u_0"]
        * (b(1.0) + recorded_r3u[..., None] * oracle["umask"]))
    oracle["corrected_u"] = (
        oracle["uu_Kmm"] + oracle["zub"][..., None] * oracle["umask"])
    require(set(oracle) == set(DEVELOPED_TRANSPORT_U_ROWS),
            "Round-154 U-transport row registry is incomplete")
    return {
        "sha256": record["sha256"], "admission": admission,
        "rows": oracle, "field_count": len(fields),
    }


def _score_developed_fct(actual, expected) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape,
            f"developed FCT shapes differ: {actual.shape} vs {expected.shape}")
    different = actual.view(np.uint64) != expected.view(np.uint64)
    delta = actual - expected
    first = ([int(value) for value in np.argwhere(different)[0]]
             if np.any(different) else None)
    return {
        "cells_scored": int(actual.size),
        "cells_unequal": int(np.count_nonzero(different)),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "first_unequal_index": first,
        "bit_exact": not bool(np.any(different)),
    }


def _developed_transport_mode_rows(actual: dict, expected: dict) -> dict:
    """Score one stage-3 U observation and retain compiled operand order."""
    required = set(DEVELOPED_TRANSPORT_U_ROWS)
    require(set(actual) == required,
            "observed developed transport registry is incomplete")
    require(set(expected) == required,
            "oracle developed transport registry is incomplete")
    rows = {
        name: _score_developed_fct(actual[name], expected[name])
        for name in DEVELOPED_TRANSPORT_U_ROWS
    }
    active3 = np.asarray(expected["umask"]) != 0.0
    active2 = np.any(active3, axis=-1)
    for name, row in rows.items():
        values = np.asarray(actual[name])
        reference = np.asarray(expected[name])
        active = active3 if values.ndim == 3 else active2
        require(values.shape == reference.shape == active.shape,
                f"developed transport active mask differs for {name}")
        bits = values.view(np.uint64) != reference.view(np.uint64)
        delta = values[active] - reference[active]
        row["active_cells_scored"] = int(np.count_nonzero(active))
        row["active_cells_unequal"] = int(np.count_nonzero(bits & active))
        row["active_max_abs"] = float(np.max(np.abs(delta), initial=0.0))
    first = next(
        (name for name in DEVELOPED_TRANSPORT_U_ROWS
         if not rows[name]["bit_exact"]), "NONE")
    return {
        "rows": rows,
        "first_non_bit_row": first,
        "rows_scored": len(rows),
        "bit_exact_rows": int(sum(row["bit_exact"] for row in rows.values())),
    }


def _score_developed_active(actual, expected, active) -> dict:
    """Score one 3-D U field against NEMO's, whole-field and on wet faces."""
    actual = np.ascontiguousarray(actual, dtype=np.float64)
    expected = np.ascontiguousarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape == active.shape,
            f"stage-2 split shapes differ: {actual.shape} vs "
            f"{expected.shape} vs {active.shape}")
    bits = actual.view(np.uint64) != expected.view(np.uint64)
    delta = (actual - expected)[active]
    return {
        "cells_scored": int(actual.size),
        "cells_unequal": int(np.count_nonzero(bits)),
        "active_cells_scored": int(np.count_nonzero(active)),
        "active_cells_unequal": int(np.count_nonzero(bits & active)),
        "active_max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "active_rms": float(
            np.sqrt(np.mean(delta * delta)) if delta.size else 0.0),
        "bit_exact": not bool(np.any(bits)),
    }


def _developed_stage2_velocity_split(production: dict, oracle: dict, *,
                                     plant_index=None) -> dict:
    """Split the developed stage-2 U velocity difference in two.

    Stage 2 binds Kaa = N+1/2 and stage 3 reads that field as its Kmm
    (``GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:217-259``
    and ``:261-277``).  Its program has an EXTERNAL half, which writes the
    barotropic correction ``zub = uu_b(Kaa) - SUM(e3u_3d*uu(Kaa))*r1_hu_0``
    at ``:753`` and applies it at ``:772``, and an INTERNAL 3-D half, the
    stage RHS at ``:400-497`` and the thickness-weighted assignment at
    ``:687-703``.  The admitted day-180 record carries every operand of the
    external half and none of the internal one, so this is the split it can
    make.

    ``rk3_stage_barotropic_correction`` is legoESM's own transcription of
    ``:753,772`` -- the SAME function the production stage calls -- and it is
    driven here with NEMO's recorded ``uu_b(Kmm)``, ``e3u_0``, ``r1_hu_0``
    and ``umask``.  Installing NEMO's depth mean leaves the deviation, so the
    difference that is REMOVED belongs to the external half.  The arms are an
    isolated JIT of that shared statement and are never relabelled as
    production.

    SCOPE, because the external half writes THREE fields at ``:217-256`` and
    this arm installs ONE of them.  ``ssh(Kaa)`` reaches ``uu(Kaa)`` only
    through ``r3u(Kaa)``, and ``r3u(Kaa)`` is the divisor of the INTERNAL
    assignment at ``:699``, not of the correction, so neither is varied here.
    ``r3u(Kaa)`` is stage 3's ``r3u(Kmm)`` and is measured non-bit at a
    RELATIVE 6.522560269672795e-13; to first order its carry is that relative
    difference times the velocity, five orders below the difference being
    split.  That is a bound read off a measured operand, not a substitution
    arm: what survives this arm is the internal half PLUS that bounded term,
    and round 157's record settles it by substitution.

    The removed part is an e3u-weighted column mean while the score is an
    unweighted rms over faces, so the two are not orthogonal and the SIGN of
    a near-zero removed fraction carries no meaning; only its magnitude does.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_common import (
        rk3_stage_barotropic_correction)

    b = np.float64
    umask = np.ascontiguousarray(oracle["umask"], dtype=b)
    active3 = umask != 0.0
    active2 = np.any(active3, axis=-1)
    uu_nemo = np.ascontiguousarray(oracle["uu_Kmm"], dtype=b)
    uu_lego = np.ascontiguousarray(production["uu_Kmm"], dtype=b)
    e3u_0 = np.ascontiguousarray(oracle["e3u_0"], dtype=b)
    r1_depth = np.ascontiguousarray(oracle["r1_hu_0"], dtype=b)
    target = np.ascontiguousarray(oracle["uu_b_Kmm"], dtype=b)
    planted = None
    if plant_index is not None:
        target = target.copy()
        planted = tuple(int(value) for value in plant_index)
        before = target[planted]
        target[planted] = np.nextafter(before, np.inf)
        require(target[planted] != before,
                "stage-2 uu_b ULP plant rounded away")

    @jax.jit
    def reproject(field, mean, weight, reciprocal, mask):
        return rk3_stage_barotropic_correction(
            field, mean, weight, reciprocal, mask)

    def run(field, weight):
        return np.ascontiguousarray(jax.device_get(reproject(
            jnp.asarray(field), jnp.asarray(target), jnp.asarray(weight),
            jnp.asarray(r1_depth), jnp.asarray(umask))), dtype=b)

    # NEMO's SUM at :753 carries no mask; production hands the helper a
    # face-masked reference thickness.  Both arms are measured rather than
    # argued equal, because legoESM's dry-face velocity is not asserted zero.
    masked_weight = e3u_0 * umask
    rows = {
        "calibration_nemo_reprojection": _score_developed_active(
            run(uu_nemo, masked_weight), uu_nemo, active3),
        "production_baseline": _score_developed_active(
            uu_lego, uu_nemo, active3),
        "nemo_depth_mean_substituted": _score_developed_active(
            run(uu_lego, masked_weight), uu_nemo, active3),
        "nemo_depth_mean_substituted_unmasked_weight":
            _score_developed_active(run(uu_lego, e3u_0), uu_nemo, active3),
    }
    require(set(rows) == set(DEVELOPED_STAGE2_SPLIT_ROWS),
            "developed stage-2 split registry is incomplete")
    baseline_max = rows["production_baseline"]["active_max_abs"]
    baseline_rms = rows["production_baseline"]["active_rms"]
    for name, row in rows.items():
        row["active_max_abs_removed_fraction"] = (
            float((baseline_max - row["active_max_abs"]) / baseline_max)
            if baseline_max > 0.0 else 0.0)
        row["active_rms_removed_fraction"] = (
            float((baseline_rms - row["active_rms"]) / baseline_rms)
            if baseline_rms > 0.0 else 0.0)

    # KNOWN ANSWER.  What the substitution removes at each level is the
    # depth-mean error the stage installed, so it must reproduce the
    # separately recorded uu_b(Kmm) row of the same walk.
    removed = uu_lego - run(uu_lego, masked_weight)
    recorded = (np.ascontiguousarray(production["uu_b_Kmm"], dtype=b)
                - np.ascontiguousarray(oracle["uu_b_Kmm"], dtype=b))
    discrepancy = (removed - recorded[..., None])[active3]
    external = {
        "recorded_uu_b_active_max_abs": float(np.max(
            np.abs(recorded[active2]), initial=0.0)),
        "recorded_uu_b_active_columns_unequal": int(np.count_nonzero(
            (np.ascontiguousarray(production["uu_b_Kmm"], dtype=b).view(
                np.uint64)
             != np.ascontiguousarray(oracle["uu_b_Kmm"], dtype=b).view(
                np.uint64)) & active2)),
        "removed_active_max_abs": float(np.max(
            np.abs(removed[active3]), initial=0.0)),
        "removed_minus_recorded_active_max_abs": float(np.max(
            np.abs(discrepancy), initial=0.0)),
    }

    reprojected = run(uu_lego, masked_weight)
    residual = reprojected - uu_nemo
    base_delta = uu_lego - uu_nemo

    def level_rms(field):
        out = []
        for level in range(field.shape[-1]):
            selected = field[..., level][active3[..., level]]
            out.append(float(
                np.sqrt(np.mean(selected * selected))
                if selected.size else 0.0))
        return out

    return {
        "statement": "stprk3_stg.f90:753,772 via "
                     "rk3_stage_barotropic_correction",
        "mode": "isolated_closure_jit",
        "scored_against": "NEMO recorded uu(Kmm) at stage 3",
        "rows": rows,
        "external_half_known_answer": external,
        "baseline_level_active_rms": level_rms(base_delta),
        "residual_level_active_rms": level_rms(residual),
        "reprojection_sha256": hashlib.sha256(
            np.ascontiguousarray(reprojected).tobytes()).hexdigest(),
        "planted_uu_b_index": list(planted) if planted is not None else None,
    }


def _developed_fct_mode_rows(
        observed: dict[str, tuple[np.ndarray, ...]], bundle: dict,
        common_actual: dict[str, np.ndarray]) -> dict:
    """Score one complete FCT observation mode in compiled-write order."""
    from legoesm.ocean.advection import NEMO_FCT_TRACE_FIELDS

    require(set(observed) == {"T", "S"},
            "developed FCT observer omitted a tracer")
    require(set(common_actual) == set(DEVELOPED_FCT_COMMON_FIELDS),
            "developed FCT common-input registry is incomplete")
    common = {}
    for name in DEVELOPED_FCT_COMMON_FIELDS:
        expected = bundle["common"][name]
        if name.startswith("r3t_"):
            # legoESM consumes live thickness, not a stored r3.  Score that
            # exact operand against NEMO's recorded e3t_0*(1+r3), rather than
            # manufacturing an r3 value with a lossy thickness/e3t_0 - 1.
            expected = (bundle["common"]["e3t_3d"]
                        * (np.float64(1.0) + expected[..., None]))
        common[name] = _score_developed_fct(common_actual[name], expected)
    tracers = {}
    trace_offset = 13
    for tracer in ("T", "S"):
        values = observed[tracer]
        require(len(values) == trace_offset + len(NEMO_FCT_TRACE_FIELDS),
                f"{tracer} FCT observer returned {len(values)} values")
        actual = {
            "base": values[1], "now": values[0],
            "rhs_entry": np.zeros_like(values[0]),
        }
        actual.update({
            name: values[trace_offset + index]
            for index, name in enumerate(NEMO_FCT_TRACE_FIELDS)
        })
        require(set(actual) == set(DEVELOPED_FCT_TRACER_FIELDS),
                f"{tracer} FCT statement registry is incomplete")
        rows = {
            name: _score_developed_fct(
                actual[name], bundle["tracers"][tracer][name])
            for name in DEVELOPED_FCT_TRACER_FIELDS
        }
        limiter = {}
        for face in ("u", "v", "w"):
            got = np.asarray(actual[f"coef_{face}"])
            want = np.asarray(bundle["tracers"][tracer][f"coef_{face}"])
            got_active = got.view(np.uint64) != np.float64(1.0).view(np.uint64)
            want_active = (
                want.view(np.uint64) != np.float64(1.0).view(np.uint64))
            limiter[face] = {
                "lego_active": int(np.count_nonzero(got_active)),
                "NEMO_active": int(np.count_nonzero(want_active)),
                "both_active": int(np.count_nonzero(got_active & want_active)),
                "selection_disagrees": int(np.count_nonzero(
                    got_active != want_active)),
                "coefficient_cells_unequal": rows[f"coef_{face}"][
                    "cells_unequal"],
            }
        tracers[tracer] = {"rows": rows, "limiter_activity": limiter}

    context_order = list(DEVELOPED_FCT_COMMON_FIELDS) + [
        f"{tracer}.{name}" for tracer in ("T", "S")
        for name in ("base", "now", "rhs_entry")]
    statement_order = [
        f"{tracer}.{name}" for tracer in ("T", "S")
        for name in DEVELOPED_FCT_TRACER_FIELDS[3:]]
    flat = dict(common)
    flat.update({
        f"{tracer}.{name}": row
        for tracer in ("T", "S")
        for name, row in tracers[tracer]["rows"].items()
    })
    first_context = next(
        (name for name in context_order if not flat[name]["bit_exact"]),
        "NONE")
    first_statement = next(
        (name for name in statement_order if not flat[name]["bit_exact"]),
        "NONE")
    return {
        "common_inputs": common, "tracers": tracers,
        "first_non_bit_context": first_context,
        "first_non_bit_statement": first_statement,
        "context_order": context_order, "statement_order": statement_order,
        "rows_scored": len(flat),
        "bit_exact_rows": int(sum(row["bit_exact"] for row in flat.values())),
    }


DEVELOPED_STAGE2_ROWS = (
    "stage1_output", "after_hpg", "after_vor", "after_adv",
    "stage2_rhs_total", "uu_Kaa_raw", "uu_Kaa_final",
)


def _developed_stage2_record(root: Path) -> dict:
    """Read the admitted Round-156 stage-2 record on legoESM's face grid.

    The window and the transpose are the SAME ones
    ``_developed_transport_record`` uses, so the two records land on one grid
    and no second index convention enters the campaign.
    """
    gate156 = _load(
        "nemo_testcase_l2_gyre_round156_developed_stage2_gate",
        "nemo_testcase_l2_gyre_round156_developed_stage2_gate.py")
    admission_path = root / "round156_developed_stage2_admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    require(admission["status"] == "PASS",
            "Round-156 stage-2 record is not admitted")
    record = gate156.read_record(root / gate156.RECORD)
    require(record["sha256"] == admission["record"]["sha256"],
            "Round-156 stage-2 record differs from its admission")
    meta = record["meta"]
    require(meta["kt"] == 1081 and meta["kstg"] == 2,
            "Round-156 record is not step 1081 stage 2")
    fields = record["fields"]

    # legoESM's U face grid is (n_lat, n_lon+1) with column 0 the west wall,
    # and its V face grid is (n_lat+1, n_lon) with row 0 the south wall, so
    # the two components take DIFFERENT windows out of NEMO's global array.
    # The U window is the one ``_developed_transport_record`` already uses.
    WINDOWS = {"u": (slice(1, 34), slice(2, 24)),
               "v": (slice(2, 34), slice(1, 24))}

    def owned3(name, tag):
        index = 0 if tag == "u" else 1
        i_window, j_window = WINDOWS[tag]
        return np.ascontiguousarray(
            fields[name][index][i_window, j_window, :30].transpose(1, 0, 2))

    rows = {}
    for name in ("rhs_entry", "uu_vv_Kbb", "uu_vv_Kmm", "umask_vmask",
                 "after_hpg", "after_vor", "after_adv", "uu_vv_Kaa_raw",
                 "uu_vv_Kaa_final"):
        rows[f"{name}_u"] = owned3(name, "u")
        rows[f"{name}_v"] = owned3(name, "v")
    # NEMO's ``ww`` is a T-point INTERFACE field over the full jpk, and it is
    # the operand dyn_zad reads (stprk3_stg.f90:495 records it after wzv and
    # before dyn_hpg, and nothing between there and dyn_zad writes it).  Its
    # window is the T window the Round-153/154 readers already use, and all
    # jpk levels are kept because legoESM's own vertical velocity carries one
    # interface per level plus the bottom.
    # The cell window is the one the two FACE windows imply on a C grid:
    # legoESM's u face 0 is the WEST face of its cell 0, so the cell that owns
    # u window index 1 starts one further east, and the same argument in the
    # other direction fixes the row.  The cell control below refuses any other
    # choice, which is how the first attempt at this window was caught.
    T_WINDOW = (slice(2, 34), slice(2, 24))
    rows["ww_t"] = np.ascontiguousarray(
        fields["rhd_ww"][1][T_WINDOW].transpose(1, 0, 2))
    rows["rhd_t"] = np.ascontiguousarray(
        fields["rhd_ww"][0][T_WINDOW][..., :30].transpose(1, 0, 2))
    for name, index, tag in (("r3u_r3v_Kmm", 0, "u"), ("r3u_r3v_Kmm", 1, "v")):
        i_window, j_window = WINDOWS[tag]
        rows[f"r3_Kmm_{tag}"] = np.ascontiguousarray(
            fields[name][index][i_window, j_window].T)
    # The stage's own sea surface heights, on the same T window, so a walk
    # can rebuild legoESM's face-ratio statement from the ORACLE's operand
    # instead of from legoESM's (round 161).
    rows["ssh_Kmm_t"] = np.ascontiguousarray(
        fields["ssh_Kmm_ssh_Kaa"][0][T_WINDOW].T)
    rows["ssh_Kaa_t"] = np.ascontiguousarray(
        fields["ssh_Kmm_ssh_Kaa"][1][T_WINDOW].T)
    rows["rDt"] = np.float64(fields["rDt_r1_Dt"][0])
    rows["ln_dynadv_vec"] = float(fields["flags_vec_linssh"][0])
    rows["lk_linssh"] = float(fields["flags_vec_linssh"][1])
    # The compiled selector at stprk3_stg.f90:721 takes the VECTOR arm when
    # either flag is set, and this deck sets ln_dynadv_vec.  A walk that
    # assumed the thickness-weighted arm would calibrate the wrong statement,
    # so the arm is read from the record and refused if it ever changes.
    require(rows["ln_dynadv_vec"] == 1.0 and rows["lk_linssh"] == 0.0,
            "Round-156 record does not carry the vector stage-update arm: "
            f"ln_dynadv_vec={rows['ln_dynadv_vec']} "
            f"lk_linssh={rows['lk_linssh']}")
    return {"sha256": record["sha256"], "admission": admission,
            "rows": rows, "meta": meta, "field_count": len(fields)}


def _score_stage2_face(actual, expected, mask) -> dict:
    """Score one face field over NEMO's own active faces, bitwise first."""
    actual = np.ascontiguousarray(np.asarray(actual, dtype=np.float64))
    expected = np.ascontiguousarray(np.asarray(expected, dtype=np.float64))
    require(actual.shape == expected.shape,
            f"stage-2 shapes differ: {actual.shape} vs {expected.shape}")
    active = np.asarray(mask) != 0.0
    while active.ndim < actual.ndim:
        active = active[..., None]
    active = np.broadcast_to(active, actual.shape)
    different = actual.view(np.uint64) != expected.view(np.uint64)
    delta = (actual - expected)[active]
    return {
        "cells_scored": int(actual.size),
        "cells_unequal": int(np.count_nonzero(different)),
        "active_cells_scored": int(np.count_nonzero(active)),
        "active_cells_unequal": int(np.count_nonzero(different & active)),
        "active_max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "active_rms": float(
            np.sqrt(np.mean(delta * delta)) if delta.size else 0.0),
    }


def _developed_tke_records(root: Path) -> dict:
    """Admit the two Round-164 streams and their same-build duplicates."""
    root = Path(root)
    operands_path = root / "oracle_tke_operands_kt00001081.bin"
    statements_path = root / "oracle_tke_statement_walk_kt00001081.bin"
    producer_path = root / "producer_commit.txt"
    require(producer_path.is_file(), f"missing {producer_path}")
    producer = producer_path.read_text(encoding="utf-8").strip()
    require(re.fullmatch(r"[0-9a-f]{40}", producer) is not None,
            f"malformed developed-TKE producer commit {producer!r}")
    for path in (operands_path, statements_path):
        stamp = Path(str(path) + ".stamp")
        require(path.is_file() and stamp.is_file(),
                f"developed-TKE stream or stamp is missing: {path}")
        parts = stamp.read_text(encoding="utf-8").strip().split()
        require(parts == [_sha256(path), producer, path.name],
                f"developed-TKE stamp disagrees for {path.name}: {parts}")

    round54 = _load(
        "nemo_testcase_l2_gyre_round54_tke_operands",
        "nemo_testcase_l2_gyre_round54_tke_operands.py")
    stage_gate = _load(
        "nemo_testcase_l2_gyre_round46_kt2_stage_gate",
        "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py")
    operands = round54.read_record(
        operands_path, expected_kt=DEVELOPED_PROCESS_STEP,
        expected_slots=(1, 1))
    statements = stage_gate.read_tke_statement_walk_record(
        statements_path, expected_kt=DEVELOPED_PROCESS_STEP,
        expected_slots=(1, 1))

    duplicate_rows = {}
    for name in ("en_entry", "rhs_pre_sweep", "en_post_sweep"):
        left = np.asarray(statements["arrays"][name])[..., :PROCESS_JPK - 1]
        right = np.asarray(operands["arrays"][name])[..., :PROCESS_JPK - 1]
        require(left.shape == right.shape,
                f"developed-TKE duplicate {name} shapes differ")
        different = left.view(np.uint64) != right.view(np.uint64)
        duplicate_rows[name] = {
            "cells": int(left.size),
            "cells_unequal": int(np.count_nonzero(different)),
            "max_abs": float(np.max(np.abs(left - right), initial=0.0)),
            "bit_exact": not bool(np.any(different)),
        }
    require(all(row["bit_exact"] for row in duplicate_rows.values()),
            f"same-build developed-TKE duplicates moved: {duplicate_rows}")
    return {
        "producer_commit": producer,
        "operand_path": str(operands_path),
        "statement_path": str(statements_path),
        "operand_sha256": _sha256(operands_path),
        "statement_sha256": _sha256(statements_path),
        "operands": operands,
        "statements": statements,
        "duplicate_rows": duplicate_rows,
    }


def developed_tke_statement_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Walk step-1081 TKE through the existing production-jitted program."""
    require(plant in (None, "none", "developed-tke-entry-ulp",
                      "developed-shear-velocity-ulp"),
            f"unknown developed-TKE plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.physics.vertical_mixing import tke as tke_module

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"developed-TKE walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "developed-TKE walk commit differs from --expect-commit")
    records = _developed_tke_records(record_root)
    bundle = _developed_entry_bundle(daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(bundle["payload"]["ssha"])
    # Drive the TKE program from NEMO's recorded surface-stress modulus.  The
    # shared forcing builder reconstructs this value from stress components;
    # that is the chained-model lane, not a given-NEMO-entry statement walk.
    # Existing stage twins use the same explicit operand substitution.
    recorded_taum = jnp.asarray(
        np.asarray(records["operands"]["arrays"]["taum_entry"])
        .swapaxes(0, 1))
    surface = surface._replace(taum=recorded_taum)

    mixing_calls = []
    real_mixing = tke_module.compute_mixing_lengths

    def capture_mixing(*args, **kwargs):
        momentum, dissipation = real_mixing(*args, **kwargs)
        energy = args[0]

        def sink(source, left, right):
            mixing_calls.append((np.asarray(source), np.asarray(left),
                                 np.asarray(right)))

        jax.debug.callback(
            sink, energy, momentum, dissipation, ordered=True)
        return momentum, dissipation

    hooks = _NEMOWSRK3TestHooks(expose_live_stage_operands=True)
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    tke_module.compute_mixing_lengths = capture_mixing
    try:
        trace = trace_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(trace)
        jax.effects_barrier()
    finally:
        tke_module.compute_mixing_lengths = real_mixing
    require(mixing_calls,
            "production step observed no mixing-length calls")
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config).step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
    observer_bytes = _state_bit_mismatches(trace.state_after, ordinary)
    require(observer_bytes == 0,
            f"developed-TKE observer moved {observer_bytes} state bytes")

    if plant == "developed-tke-entry-ulp":
        planted = np.asarray(state.tke.data).copy()
        candidates = np.argwhere(np.isfinite(planted) & (planted != 0.0))
        require(candidates.size > 0,
                "developed-TKE plant found no nonzero entry energy")
        index = tuple(int(value) for value in candidates[
            int(np.argmax(np.abs(planted[tuple(candidates.T)])))])
        before = planted[index]
        planted[index] = np.nextafter(before, np.inf)
        planted_state = state._replace(
            tke=state.tke.replace(data=jnp.asarray(planted)))
        planted_trace = trace_model.step(
            planted_state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(planted_trace)
        clean_fields = trace.tke_statement_trace
        moved_fields = planted_trace.tke_statement_trace
        moved = {}
        for name in (
                "en_entry", "en_after_boundaries", "en_after_langmuir",
                "rhs_pre_sweep", "en_post_sweep"):
            left = np.asarray(getattr(clean_fields, name))
            right = np.asarray(getattr(moved_fields, name))
            moved[name] = int(np.count_nonzero(
                left.view(np.uint64) != right.view(np.uint64)))
        require(any(value > 0 for value in moved.values()),
                f"developed-TKE entry ULP moved no registered row: {moved}")
        return {
            "status": "PLANT-FIRED", "plant": plant,
            "control": {"index": list(index), "baseline": float(before),
                        "moved_rows": moved},
        }

    oracle = records["operands"]["arrays"]
    statement_oracle = records["statements"]["arrays"]
    production = trace.tke_statement_trace

    # The full RK3 production step legitimately enters the TKE closure more
    # than once.  Select the final closure belonging to the statement trace
    # by its consumed post-sweep energy, never by callback order.  Duplicate
    # matches are permitted only when their returned lengths are bitwise the
    # same (for example, a repeated compiled invocation).
    post_sweep_energy = np.asarray(production.en_post_sweep)[..., 1:]
    entry_energy = np.asarray(production.en_entry)

    def same_bits(left, right):
        return (left.shape == right.shape
                and np.array_equal(left.view(np.uint64),
                                   right.view(np.uint64)))

    final_calls = [call for call in mixing_calls
                   if same_bits(call[0], post_sweep_energy)]
    entry_calls = [call for call in mixing_calls
                   if same_bits(call[0], entry_energy)]
    require(final_calls,
            "no production mixing-length call consumed the traced "
            "post-sweep TKE")
    require(all(same_bits(call[1], final_calls[0][1])
                and same_bits(call[2], final_calls[0][2])
                for call in final_calls[1:]),
            "duplicate post-sweep mixing-length calls disagree")
    mixing_momentum, mixing_dissipation = final_calls[0][1:]
    mixing_call_classification = {
        "observed": len(mixing_calls),
        "entry_energy": len(entry_calls),
        "post_sweep_energy": len(final_calls),
        "other_energy": len(mixing_calls) - len(entry_calls) - len(final_calls),
        "selection": "bitwise input match to traced en_post_sweep[...,1:]",
    }

    def yx(name, *, statement=False):
        source = statement_oracle if statement else oracle
        return np.asarray(source[name], dtype=np.float64).swapaxes(0, 1)

    wet30 = yx("wmask")[..., :PROCESS_JPK - 1] != 0.0
    wet29 = yx("wmask")[..., 1:PROCESS_JPK - 1] != 0.0

    def scored(actual, expected, mask, citation):
        row = _score_developed_row(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64), np.asarray(mask))
        row["nemo_statement"] = citation
        return row

    # Extend the existing Round-104/105 shear walk at the same developed
    # entry.  The restart bridge supplies NEMO's U/V/SSH exactly; the
    # Round-164 record supplies the viscosity and final p_sh2 that the TKE
    # RHS above consumes.  The source-literal replay must reproduce that
    # recorded endpoint before any production operand row is interpreted.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d)
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)
    from nemo_testcase_l2_gyre_round104_shear_replay import (
        model_operands, rebuild_sh2)

    u_mask, v_mask = compute_face_masks_3d(card.recipe.z_coord.is_active)
    reference_u = _u_east_to_face(bundle["payload"]["un"])
    reference_v = _v_north_to_face(bundle["payload"]["vn"])
    reference_eta = np.asarray(bundle["payload"]["sshn"], dtype=np.float64)
    reference_avm = yx("avm_entry")[..., 1:30]
    reference_operands, reference_shear = model_operands(
        card.recipe.z_coord, reference_u, reference_v, reference_eta,
        reference_avm, u_mask, v_mask, return_intermediates=True)
    model_shear_operands, model_shear = model_operands(
        card.recipe.z_coord, state.u.data, state.v.data, state.eta.data,
        state.tke_avm.data, u_mask, v_mask, return_intermediates=True)
    recorded_sh2 = yx("sh2")[..., 1:30]
    replayed_sh2 = rebuild_sh2(reference_operands)
    captured_metrics = production.shear_face_metrics
    require(captured_metrics is not None and len(captured_metrics) == 4,
            "developed shear walk did not capture four production metrics")
    e3u_now, e3u_before, e3v_now, e3v_before = (
        np.asarray(value, dtype=np.float64) for value in captured_metrics)

    def all_cells(value):
        return np.ones(np.shape(value), dtype=bool)

    shear_rows = {
        "reference_replay_p_sh2": scored(
            replayed_sh2, recorded_sh2, all_cells(recorded_sh2),
            "zdfsh2.f90:97-114"),
        "u_face_entry": scored(
            state.u.data, reference_u, all_cells(reference_u),
            "zdfsh2.f90:100-101"),
        "v_face_entry": scored(
            state.v.data, reference_v, all_cells(reference_v),
            "zdfsh2.f90:105-106"),
        "u_vertical_difference": scored(
            model_shear["du"], reference_shear["du"],
            all_cells(reference_shear["du"]), "zdfsh2.f90:100-101"),
        "v_vertical_difference": scored(
            model_shear["dv"], reference_shear["dv"],
            all_cells(reference_shear["dv"]), "zdfsh2.f90:105-106"),
        "avm_entry": scored(
            state.tke_avm.data, reference_avm, all_cells(reference_avm),
            "zdfsh2.f90:99,104"),
        "avm_face_u": scored(
            model_shear_operands["avm_face_u"],
            reference_operands["avm_face_u"],
            all_cells(reference_operands["avm_face_u"]),
            "zdfsh2.f90:99"),
        "avm_face_v": scored(
            model_shear_operands["avm_face_v"],
            reference_operands["avm_face_v"],
            all_cells(reference_operands["avm_face_v"]),
            "zdfsh2.f90:104"),
        "r3u_step_entry": scored(
            model_shear["r3u"], reference_shear["r3u"],
            all_cells(reference_shear["r3u"]), "domqco.f90:213-215"),
        "r3v_step_entry": scored(
            model_shear["r3v"], reference_shear["r3v"],
            all_cells(reference_shear["r3v"]), "domqco.f90:213,216-217"),
        "e3u_now": scored(
            e3u_now, reference_shear["e3u"],
            all_cells(reference_shear["e3u"]), "zdfsh2.f90:102"),
        "e3u_before": scored(
            e3u_before, reference_shear["e3u"],
            all_cells(reference_shear["e3u"]), "zdfsh2.f90:102"),
        "e3v_now": scored(
            e3v_now, reference_shear["e3v"],
            all_cells(reference_shear["e3v"]), "zdfsh2.f90:107"),
        "e3v_before": scored(
            e3v_before, reference_shear["e3v"],
            all_cells(reference_shear["e3v"]), "zdfsh2.f90:107"),
        "divisor_u_from_captured_metrics": scored(
            e3u_now * e3u_before, reference_operands["divisor_u"],
            all_cells(reference_operands["divisor_u"]),
            "zdfsh2.f90:102"),
        "divisor_v_from_captured_metrics": scored(
            e3v_now * e3v_before, reference_operands["divisor_v"],
            all_cells(reference_operands["divisor_v"]),
            "zdfsh2.f90:107"),
        "wumask": scored(
            model_shear_operands["wumask"], reference_operands["wumask"],
            all_cells(reference_operands["wumask"]),
            "dommsk.f90:237-242"),
        "wvmask": scored(
            model_shear_operands["wvmask"], reference_operands["wvmask"],
            all_cells(reference_operands["wvmask"]),
            "dommsk.f90:237-242"),
        "coast_u": scored(
            model_shear_operands["coast_u"], reference_operands["coast_u"],
            all_cells(reference_operands["coast_u"]),
            "zdfsh2.f90:112"),
        "coast_v": scored(
            model_shear_operands["coast_v"], reference_operands["coast_v"],
            all_cells(reference_operands["coast_v"]),
            "zdfsh2.f90:113"),
        "zsh2u_isolated_replay": scored(
            model_shear["zsh2u"], reference_shear["zsh2u"],
            all_cells(reference_shear["zsh2u"]), "zdfsh2.f90:99-103"),
        "zsh2v_isolated_replay": scored(
            model_shear["zsh2v"], reference_shear["zsh2v"],
            all_cells(reference_shear["zsh2v"]), "zdfsh2.f90:104-108"),
        "model_isolated_replay_vs_production": scored(
            rebuild_sh2(model_shear_operands), production.rhs_shear,
            all_cells(recorded_sh2), "zdfsh2.f90:97-114"),
        "production_p_sh2": scored(
            production.rhs_shear, recorded_sh2, wet29,
            "zdfsh2.f90:111-114"),
    }
    require(shear_rows["reference_replay_p_sh2"]["bit_exact"],
            "developed NEMO shear replay does not reproduce recorded p_sh2")

    if plant == "developed-shear-velocity-ulp":
        # Start from the face carrying the largest nonzero source-literal U
        # contribution.  Try its two consumed levels and both one-ULP
        # directions; the first production run that moves p_sh2 is the
        # discriminating control.  Every trial changes exactly one bit-level
        # neighbour of one nonzero consumed velocity value.
        order = np.argsort(np.abs(reference_shear["zsh2u"]), axis=None)[::-1]
        fired = None
        clean_p_sh2 = np.asarray(production.rhs_shear)
        clean_du = np.asarray(model_shear["du"])
        for flat in order:
            face_index = np.unravel_index(
                int(flat), reference_shear["zsh2u"].shape)
            if reference_shear["zsh2u"][face_index] == 0.0:
                break
            j, i, k = (int(value) for value in face_index)
            for level in (k, k + 1):
                for direction in (np.inf, -np.inf):
                    planted_u = np.asarray(state.u.data).copy()
                    index = (j, i, level)
                    baseline = planted_u[index]
                    if baseline == 0.0 or not np.isfinite(baseline):
                        continue
                    planted_u[index] = np.nextafter(baseline, direction)
                    planted_state = state._replace(
                        u=state.u.replace(data=jnp.asarray(planted_u)))
                    planted_trace = trace_model.step(
                        planted_state, dt=card.dt_s,
                        freshwater=freshwater, surface_forcing=surface,
                        _nemo_stage1_zad_eta_after_override=ssha)
                    jax.device_get(planted_trace)
                    moved_p = int(np.count_nonzero(
                        clean_p_sh2.view(np.uint64) != np.asarray(
                            planted_trace.tke_statement_trace.rhs_shear
                        ).view(np.uint64)))
                    planted_du = (planted_u[..., :-1]
                                  - planted_u[..., 1:])
                    moved_du = int(np.count_nonzero(
                        clean_du.view(np.uint64)
                        != planted_du.view(np.uint64)))
                    if moved_p and moved_du:
                        fired = {
                            "u_index_jik": list(index),
                            "baseline_velocity_m_s": float(baseline),
                            "planted_velocity_m_s": float(planted_u[index]),
                            "vertical_difference_cells_moved": moved_du,
                            "production_p_sh2_cells_moved": moved_p,
                        }
                        break
                if fired is not None:
                    break
            if fired is not None:
                break
        require(fired is not None,
                "PLANT-BLIND: no one-ULP consumed velocity perturbation "
                "moved both the vertical difference and production p_sh2")
        return {
            "status": "PLANT-FIRED", "plant": plant,
            "control": fired,
        }

    statement_rows = {
        "en_entry": scored(
            production.en_entry, yx("en_entry", statement=True)[..., 1:30],
            wet29, "zdftke.f90:267"),
        "en_after_boundaries": scored(
            production.en_after_boundaries,
            yx("en_after_boundaries", statement=True)[..., :30], wet30,
            "zdftke.f90:283-323"),
        "en_after_langmuir": scored(
            production.en_after_langmuir,
            yx("en_after_langmuir", statement=True)[..., :30], wet30,
            "zdftke.f90:325-394"),
        "matrix_upper": scored(
            production.matrix_upper, yx("matrix_upper")[..., 1:30], wet29,
            "zdftke.f90:424-433"),
        "matrix_lower": scored(
            production.matrix_lower, yx("matrix_lower")[..., 1:30], wet29,
            "zdftke.f90:424-434"),
        "matrix_diagonal": scored(
            production.matrix_diag, yx("matrix_diag")[..., 1:30], wet29,
            "zdftke.f90:424-435"),
        "rhs_pre_sweep": scored(
            production.rhs_pre_sweep,
            yx("rhs_pre_sweep", statement=True)[..., :30], wet30,
            "zdftke.f90:438-441"),
        "en_post_sweep": scored(
            production.en_post_sweep,
            yx("en_post_sweep", statement=True)[..., :30], wet30,
            "zdftke.f90:475-494"),
    }
    operand_rows = {
        "entry_en": scored(
            trace.tke_entry, yx("en_entry")[..., 1:30], wet29,
            "zdftke.f90:267"),
        "p_sh2": scored(
            production.rhs_shear, yx("sh2")[..., 1:30], wet29,
            "zdftke.f90:438"),
        "entry_avm": scored(
            state.tke_avm.data, yx("avm_entry")[..., 1:30], wet29,
            "zdftke.f90:428-431,435"),
        "entry_avt": scored(
            state.tke_avt.data, yx("avt_entry")[..., 1:30], wet29,
            "zdftke.f90:439"),
        "entry_dissl": scored(
            state.tke_dissl.data, yx("dissl_entry")[..., 1:30], wet29,
            "zdftke.f90:435,440"),
        "rn2": scored(
            production.bn2_output, yx("rn2")[..., 1:30], wet29,
            "zdftke.f90:439"),
    }
    closure_rows = {
        "mxl_momentum": scored(
            mixing_momentum, yx("mxl_momentum")[..., 1:30], wet29,
            "zdftke.f90:611-693"),
        "mxl_dissipation": scored(
            mixing_dissipation, yx("mxl_dissipation")[..., 1:30], wet29,
            "zdftke.f90:611-693"),
        "avm_closure": scored(
            trace.state_after.tke_avm.data,
            yx("avm_closure")[..., 1:30], wet29,
            "zdftke.f90:700-704"),
        "avt_closure": scored(
            trace.state_after.tke_avt.data,
            yx("avt_closure")[..., 1:30], wet29,
            "zdftke.f90:700-712"),
        "dissl_output": scored(
            trace.state_after.tke_dissl.data,
            yx("dissl_output")[..., 1:30], wet29,
            "zdftke.f90:700-705"),
        "zdfphy_avt_copy": scored(
            trace.state_after.tke_avt.data,
            yx("avt_pre_evd")[..., 1:30], wet29,
            "zdfphy.f90:348-354"),
    }
    ordered = (
        "en_entry", "en_after_boundaries", "en_after_langmuir",
        "matrix_upper", "matrix_lower", "matrix_diagonal",
        "rhs_pre_sweep", "en_post_sweep")
    first_statement = next(
        (name for name in ordered if not statement_rows[name]["bit_exact"]),
        "NONE")
    operand_order = (
        "entry_en", "entry_avm", "entry_dissl", "p_sh2", "entry_avt",
        "rn2")
    first_operand = next(
        (name for name in operand_order if not operand_rows[name]["bit_exact"]),
        "NONE")
    report = {
        "format": "gyre-round165-developed-tke-statement-walk-v1",
        "status": "PASS", "worktree": stamp,
        "entry_step": DEVELOPED_ENTRY_STEP,
        "process_step": DEVELOPED_PROCESS_STEP,
        "execution": "LatLonCGridOceanModel.step -> self._step_jitted",
        "entry_overrides": {
            "surface_taum": "recorded operand taum_entry",
            "stage1_zad_eta_after": "recorded restart ssha",
        },
        "admission": {
            key: value for key, value in records.items()
            if key not in ("operands", "statements")},
        "observer_state_unequal_bytes": observer_bytes,
        "mixing_call_classification": mixing_call_classification,
        "statement_order": list(ordered),
        "statement_rows": statement_rows,
        "operand_order": list(operand_order),
        "operand_rows": operand_rows,
        "closure_rows": closure_rows,
        "shear_walk": {
            "compiled_order": [
                "u/v vertical differences", "avm face sums",
                "live e3 face factors and divisors", "W-face masks",
                "U/V face shear", "T-point p_sh2"],
            "rows": shear_rows,
            "production_rows": [
                "u_face_entry", "v_face_entry", "e3u_now",
                "e3u_before", "e3v_now", "e3v_before",
                "production_p_sh2"],
            "isolated_diagnostics": [
                "r3u_step_entry", "r3v_step_entry",
                "divisor_u_from_captured_metrics",
                "divisor_v_from_captured_metrics",
                "zsh2u_isolated_replay", "zsh2v_isolated_replay",
                "model_isolated_replay_vs_production"],
        },
        "first_non_bit_statement": first_statement,
        "first_non_bit_operand": first_operand,
        "predictions": {
            "entry_boundary_langmuir_bit": all(
                statement_rows[name]["bit_exact"] for name in
                ("en_entry", "en_after_boundaries", "en_after_langmuir")),
            "first_non_bit_statement_is_rhs": (
                first_statement == "rhs_pre_sweep"),
            "first_non_bit_rhs_operand_is_sh2": first_operand == "p_sh2",
        },
    }
    report["all_frozen_predictions_confirmed"] = all(
        report["predictions"].values())
    Path(evidence_root).mkdir(parents=True, exist_ok=True)
    return report


def developed_stage1_output_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, rhs_root: Path, family_root: Path,
        evidence_root: Path, *, plant: str | None = None) -> dict:
    """Walk the developed stage-1 momentum output in compiled order.

    This extends the shared developed-state bridge and live-operand trace.
    NEMO's completed stage-1 RHS comes from the admitted round-140 record,
    its five cumulative operator boundaries from round 146, and its corrected
    stage output from round 156.  No new index convention or step driver is
    introduced here.
    """
    require(plant in (None, "none", "stage1-entry-u-ulp"),
            f"unknown developed stage-1 plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        rk3_stage_velocity_update)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"stage-1 output walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "stage-1 output walk commit differs from --expect-commit")
    oracle_stage = _developed_stage2_record(Path(stage2_record_root))
    stage_rows = oracle_stage["rows"]
    bundle = _developed_entry_bundle(daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(bundle["payload"]["ssha"])

    hooks = _NEMOWSRK3TestHooks(expose_live_stage_operands=True)
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    trace = trace_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)
    jax.device_get(trace)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config).step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
    passivity = _state_bit_mismatches(trace.state_after, ordinary)
    require(passivity == 0,
            f"stage-1 live-operand trace moved {passivity} state bytes")

    if plant == "stage1-entry-u-ulp":
        planted_u = np.array(state.u.data, copy=True)
        active = np.asarray(state.u_mask.data) != 0.0
        candidate_mask = (active[..., None] & np.isfinite(planted_u)
                          & (planted_u != 0.0))
        candidates = np.argwhere(candidate_mask)
        require(candidates.size > 0, "stage-1 entry plant found no live U")
        candidate_values = planted_u[tuple(candidates.T)]
        index = tuple(int(value) for value in candidates[
            int(np.argmax(np.abs(candidate_values)))])
        planted_u[index] = np.nextafter(planted_u[index], np.inf)
        planted_state = state._replace(
            u=state.u.replace(data=jnp.asarray(planted_u)))
        planted = trace_model.step(
            planted_state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        moved = _score_stage2_face(
            np.asarray(planted.stage_outputs[0][0]),
            np.asarray(trace.stage_outputs[0][0]),
            np.asarray(state.u_mask.data))
        require(moved["active_cells_unequal"] > 0,
                f"stage-1 entry ULP moved no stage output: {moved}")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": {"index": list(index), "stage1_output_u": moved}}

    round83 = _load(
        "nemo_testcase_l2_gyre_round83_slow_forcing_walk",
        "nemo_testcase_l2_gyre_round83_slow_forcing_walk.py")
    rhs_path = Path(rhs_root) / round83.ROUND140_RHS_RECORD
    rhs_record = round83.read_round140_rhs(rhs_path)
    rhs_fields = rhs_record["fields"]
    round146 = _load(
        "nemo_testcase_l2_gyre_round146_rhs_family_gate",
        "nemo_testcase_l2_gyre_round146_rhs_family_gate.py")
    family_validation_path = (
        Path(family_root) / "round146_rhs_family_validation.json")
    family_validation = json.loads(
        family_validation_path.read_text(encoding="utf-8"))
    require(family_validation["status"] == "PASS",
            "round-146 family record is not admitted")
    family_path = Path(family_root) / round146.RECORD
    require(family_validation["record_sha256"] == _sha256(family_path),
            "round-146 family record differs from its admission")
    family_record = round146.read_record_bytes(family_path.read_bytes())

    def native(value, face):
        data = np.asarray(value.data if hasattr(value, "dims") else value,
                          dtype=np.float64)
        return np.ascontiguousarray(
            data[:, 1:, :] if face == "u" else data[1:, :, :])

    def owned(value):
        data = np.asarray(value, dtype=np.float64)
        require(data.shape == (round146.NY, round146.NX, round146.NZ),
                f"round-146 family extent changed: {data.shape}")
        return np.ascontiguousarray(data[2:-2, 2:-2, :round146.NZ - 1])

    previous = {
        face: np.zeros_like(owned(
            family_record["fields"][f"after_hpg_{face}"]))
        for face in ("u", "v")
    }
    oracle_addends = {}
    for family in round146.FAMILIES:
        current = {
            face: owned(family_record["fields"][f"after_{family}_{face}"])
            for face in ("u", "v")
        }
        oracle_addends[family] = tuple(
            current[face] - previous[face] for face in ("u", "v"))
        previous = current

    live_names = {"hpg": "hpg", "ldf": "ldf", "vor": "vorticity",
                  "keg": "keg", "zad": "zad"}
    operands = trace.operator_operands[0]
    masks = {"u": rhs_fields["umask"], "v": rhs_fields["vmask"]}
    family_rows = {}
    for family in round146.FAMILIES:
        family_rows[family] = {}
        name = live_names[family]
        for index, face in enumerate(("u", "v")):
            family_rows[family][face] = _score_stage2_face(
                native(operands[f"{name}_{face}"], face),
                oracle_addends[family][index], masks[face])

    full_rhs = {
        "u": native(trace.stage1_full_rhs[0], "u"),
        "v": native(trace.stage1_full_rhs[1], "v"),
    }
    rhs_rows = {
        face: _score_stage2_face(
            full_rhs[face], rhs_fields[f"rhs_{face}"], masks[face])
        for face in ("u", "v")
    }

    @jax.jit
    def raw_update(before, rhs, mask, coefficient):
        return rk3_stage_velocity_update(
            before, rhs, coefficient, mask, vector_form=True)

    raw_rows = {}
    correction_rows = {}
    output_rows = {}
    for index, face in enumerate(("u", "v")):
        before = np.asarray(getattr(state, face).data)
        mask = np.broadcast_to(
            np.asarray(getattr(state, f"{face}_mask").data)[..., None],
            before.shape)
        rhs_full = np.array(trace.stage1_full_rhs[index], copy=True)
        if face == "u":
            rhs_full[:, 1:, :] = rhs_fields["rhs_u"]
        else:
            rhs_full[1:, :, :] = rhs_fields["rhs_v"]
        oracle_raw = np.asarray(jax.device_get(raw_update(
            jnp.asarray(before), jnp.asarray(rhs_full), jnp.asarray(mask),
            jnp.asarray(trace.stage_coefficients[0][0]))))
        model_raw = np.asarray(trace.stage_raw_velocities[0][index])
        oracle_final = np.asarray(stage_rows[f"uu_vv_Kmm_{face}"])
        model_final = np.asarray(trace.stage_outputs[0][index])
        raw_rows[face] = _score_stage2_face(
            model_raw, oracle_raw, mask)
        correction_rows[face] = _score_stage2_face(
            model_final - model_raw, oracle_final - oracle_raw,
            stage_rows[f"umask_vmask_{face}"])
        output_rows[face] = _score_stage2_face(
            model_final, oracle_final, stage_rows[f"umask_vmask_{face}"])

    ordered = [
        *(f"{family}_{face}" for family in round146.FAMILIES
          for face in ("u", "v")),
        "completed_rhs_u", "completed_rhs_v", "raw_update_u", "raw_update_v",
        "barotropic_correction_u", "barotropic_correction_v",
        "stage1_output_u", "stage1_output_v",
    ]
    flat = {
        **{f"{family}_{face}": family_rows[family][face]
           for family in round146.FAMILIES for face in ("u", "v")},
        **{f"completed_rhs_{face}": rhs_rows[face]
           for face in ("u", "v")},
        **{f"raw_update_{face}": raw_rows[face] for face in ("u", "v")},
        **{f"barotropic_correction_{face}": correction_rows[face]
           for face in ("u", "v")},
        **{f"stage1_output_{face}": output_rows[face]
           for face in ("u", "v")},
    }
    first_non_bit = next(
        (name for name in ordered
         if flat[name]["active_cells_unequal"] > 0), "NONE")
    evidence_root.mkdir(parents=True, exist_ok=True)
    return {
        "format": "gyre-round164-developed-stage1-output-walk-v1",
        "status": "PASS", "worktree": stamp,
        "entry": {"step": DEVELOPED_ENTRY_STEP,
                  "process_step": DEVELOPED_PROCESS_STEP},
        "records": {
            "rhs_sha256": _sha256(rhs_path),
            "family_sha256": _sha256(family_path),
            "stage_sha256": oracle_stage["sha256"],
        },
        "observer_state_unequal_bytes": passivity,
        "compiled_order": ordered, "rows": flat,
        "family_rows": family_rows, "completed_rhs": rhs_rows,
        "raw_update": raw_rows, "barotropic_correction": correction_rows,
        "stage1_output": output_rows, "first_non_bit": first_non_bit,
        "compiled_citations": {
            "family_order": "stp2d.f90:141-176",
            "stage_call": "stprk3.f90:188-194",
            "raw_update": "stprk3_stg.f90:668-674",
            "barotropic_correction": "stprk3_stg.f90:734-759",
        },
    }


def developed_stage2_rhs_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Walk NEMO's compiled stage-2 momentum program at the developed state.

    Round 155 named the stage-2 velocity the magnitude owner of the stage-3
    transport difference and round 156 measured that the external barotropic
    half of the stage-2 program carries none of it.  This walk scores the
    INTERNAL half -- the right-hand side at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:497,510,525``
    and the vector-arm assignment at ``:721-724`` -- against NEMO's own
    recorded snapshots, from NEMO's admitted day-180 entry, under production
    JIT.
    """
    require(plant in (None, "none", "entry-kbb-ulp", "hpg-rank-scale"),
            f"unknown developed stage-2 plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        rk3_stage_velocity_update)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"stage-2 walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "stage-2 walk commit differs from --expect-commit")
    evidence_root = Path(evidence_root)
    oracle = _developed_stage2_record(Path(stage2_record_root))
    rows = dict(oracle["rows"])
    active_u = np.asarray(rows["umask_vmask_u"]) != 0.0
    if plant == "entry-kbb-ulp":
        # The window control must refuse a before-level velocity that is not
        # the entry state, which is what a wrong window or a moved record
        # looks like.
        moved = np.array(rows["uu_vv_Kbb_u"], copy=True)
        index = tuple(int(value) for value in np.argwhere(active_u)[0])
        moved[index] = np.nextafter(moved[index], np.inf)
        rows["uu_vv_Kbb_u"] = moved
    if plant == "hpg-rank-scale":
        # The magnitude ranking must not be an artefact of the row order: a
        # pressure-gradient snapshot scaled by one part in a million has to
        # take the top of the ranking away from the advection.
        rows["after_hpg_u"] = np.asarray(
            rows["after_hpg_u"]) * np.float64(1.0 + 1.0e-6)

    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    payload = bundle["payload"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])

    def run(*, exposed=("u", "v"), passive=True, **hook_kwargs):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_kwargs))
        result = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(result)
        # A WRITE-only hook substitutes its own slots AFTER the ordinary step
        # completes, so passivity is judged on every OTHER prognostic field.
        # An override arm changes the step itself and is not passive; it says
        # so rather than being excused.
        if passive:
            neutral = result._replace(
                **{name: getattr(ordinary, name) for name in exposed})
            observer_unequal.append(_state_bit_mismatches(neutral, ordinary))
        return result

    def run_uv(**kwargs):
        result = run(**kwargs)
        return (np.asarray(result.u.data), np.asarray(result.v.data))

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary = ordinary_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)

    # Window control.  NEMO's recorded Kbb velocity is the step-entry level,
    # which this walk loads from the same restart, so it must be BIT against
    # legoESM's entry state.  A wrong window or a wrong transpose cannot
    # survive this, and it is the only check that proves the two grids are
    # the same grid before any difference is attributed.
    entry_control = {
        tag: _score_stage2_face(
            np.asarray(getattr(state, tag).data),
            rows[f"uu_vv_Kbb_{tag}"], rows[f"umask_vmask_{tag}"])
        for tag in ("u", "v")
    }
    require(all(row["cells_unequal"] == 0 for row in entry_control.values()),
            "the Round-156 record window does not land on legoESM's face "
            f"grid: {entry_control}")

    observer_unequal: list[int] = []
    production = {}
    production["stage1_output"] = run_uv(expose_momentum_stage=1)
    production["uu_Kaa_final"] = run_uv(expose_momentum_stage=2)
    production["uu_Kaa_raw"] = run_uv(expose_stage2_raw_momentum=True)
    production["stage2_rhs_total"] = run_uv(expose_stage2_momentum_rhs=True)
    components = {
        name: run_uv(expose_momentum_operator=name,
                     expose_momentum_operator_stage=2)
        for name in ("hpg", "vorticity", "advection")
    }
    # NEMO's dyn_vor and dyn_adv ACCUMULATE into the slot dyn_hpg overwrote
    # (stprk3_stg.f90:497,510,525), so the compiled-order snapshots are
    # cumulative.  legoESM publishes the three components separately, so the
    # two cumulative rows below carry ONE instrument-side addition each and
    # are labelled as such; the ``stage2_rhs_total`` row is the production
    # accumulation itself and carries none.
    production["after_hpg"] = components["hpg"]
    production["after_vor"] = tuple(
        components["hpg"][i] + components["vorticity"][i] for i in (0, 1))
    production["after_adv"] = tuple(
        production["after_vor"][i] + components["advection"][i]
        for i in (0, 1))

    oracle_for_row = {
        "stage1_output": "uu_vv_Kmm",
        "after_hpg": "after_hpg",
        "after_vor": "after_vor",
        "after_adv": "after_adv",
        "stage2_rhs_total": "after_adv",
        "uu_Kaa_raw": "uu_vv_Kaa_raw",
        "uu_Kaa_final": "uu_vv_Kaa_final",
    }
    walk = {}
    for name in DEVELOPED_STAGE2_ROWS:
        source = oracle_for_row[name]
        for tag, index in (("u", 0), ("v", 1)):
            walk[f"{name}_{tag}"] = _score_stage2_face(
                production[name][index], rows[f"{source}_{tag}"],
                rows[f"umask_vmask_{tag}"])
            walk[f"{name}_{tag}"]["oracle_group"] = source
            walk[f"{name}_{tag}"]["instrument_additions"] = (
                {"after_vor": 1, "after_adv": 2}.get(name, 0))

    # Calibration, the discipline rounds 155 and 156 established: legoESM's
    # own transcription of the compiled assignment must rebuild NEMO's output
    # bit for bit from NEMO's OWN operands before any statement is named.
    @jax.jit
    def isolated_assignment(before, rhs, mask):
        return rk3_stage_velocity_update(
            before, rhs, np.float64(rows["rDt"]), mask, vector_form=True)

    calibration = {}
    for tag, index in (("u", 0), ("v", 1)):
        rebuilt = np.asarray(jax.device_get(isolated_assignment(
            jnp.asarray(rows[f"uu_vv_Kbb_{tag}"]),
            jnp.asarray(rows[f"after_adv_{tag}"]),
            jnp.asarray(rows[f"umask_vmask_{tag}"]))))
        calibration[tag] = _score_stage2_face(
            rebuilt, rows[f"uu_vv_Kaa_raw_{tag}"],
            rows[f"umask_vmask_{tag}"])

    # Budget: the vector arm makes the raw stage-2 velocity difference
    # exactly rDt times the right-hand-side difference, to one rounding.
    budget = {}
    for tag in ("u", "v"):
        predicted = (float(rows["rDt"])
                     * walk[f"stage2_rhs_total_{tag}"]["active_max_abs"])
        observed = walk[f"uu_Kaa_raw_{tag}"]["active_max_abs"]
        budget[tag] = {
            "rDt": float(rows["rDt"]),
            "predicted_max_abs": predicted,
            "observed_max_abs": observed,
            "relative_disagreement": (
                abs(predicted - observed) / observed if observed else 0.0),
        }

    # OWNED or INHERITED.  The stage-2 entry velocity carries a difference of
    # its own, so the right-hand-side rows above cannot say by themselves
    # whether stage 2 makes the magnitude or merely passes it on.  One
    # directed substitution settles it: replace the stage-2 entry velocity
    # with NEMO's own recorded uu(Kmm)/vv(Kmm) and leave every other entry
    # field at legoESM's value.  The null arm is the calibration of the
    # substitution mechanism itself -- it hands the step legoESM's OWN entry
    # bundle and must reproduce the production rows byte for byte.
    entry_bundle = run(expose_tracer_stage=1, exposed=("T", "S", "eta"))
    lego_entry = (
        jnp.asarray(production["stage1_output"][0]),
        jnp.asarray(production["stage1_output"][1]),
        entry_bundle.T.data, entry_bundle.S.data, entry_bundle.eta.data)
    substitution = {}
    armed_rows = {}
    for arm, velocity in (
            ("null", (lego_entry[0], lego_entry[1])),
            ("nemo_entry_velocity",
             (jnp.asarray(rows["uu_vv_Kmm_u"]),
              jnp.asarray(rows["uu_vv_Kmm_v"])))):
        armed = run_uv(
            passive=False,
            stage_entry_override=(
                2, velocity[0], velocity[1],
                lego_entry[2], lego_entry[3], lego_entry[4]),
            expose_stage2_momentum_rhs=True)
        armed_rows[arm] = armed
        substitution[arm] = {
            tag: _score_stage2_face(
                armed[index], rows[f"after_adv_{tag}"],
                rows[f"umask_vmask_{tag}"])
            for tag, index in (("u", 0), ("v", 1))
        }
    # The mechanism is NOT byte-neutral, and that is registered rather than
    # excused: the production step builds the stage thickness by averaging
    # the step's own live thicknesses, while the override rebuilds it from
    # the stage free surface, which is the same number algebraically and a
    # different one bitwise.  The offset below measures it.  Because BOTH
    # arms carry that same offset, the comparison between them is still one
    # variable -- the entry velocity -- and the removed fractions are taken
    # against the null arm, never against the production row.
    for tag, index in (("u", 0), ("v", 1)):
        substitution["null"][tag]["offset_from_production_max_abs"] = float(
            np.max(np.abs(
                np.asarray(armed_rows["null"][index])
                - np.asarray(production["stage2_rhs_total"][index])),
                initial=0.0))
        base_max = substitution["null"][tag]["active_max_abs"]
        base_rms = substitution["null"][tag]["active_rms"]
        for arm in substitution:
            row = substitution[arm][tag]
            row["max_abs_removed_fraction"] = (
                float((base_max - row["active_max_abs"]) / base_max)
                if base_max > 0.0 else 0.0)
            row["rms_removed_fraction"] = (
                float((base_rms - row["active_rms"]) / base_rms)
                if base_rms > 0.0 else 0.0)

    require(all(value == 0 for value in observer_unequal),
            "a stage-2 exposure hook moved the production state outside "
            f"u/v: {observer_unequal}")
    # Decision 43 ranks by magnitude, so each family's DIFFERENCE is reported
    # next to the size of the term it sits in.  NEMO's own increments are
    # differences of its recorded cumulative snapshots and carry one rounding
    # on the oracle side; they are context for the ranking, never a scored
    # row.
    oracle_terms = {}
    for tag in ("u", "v"):
        active = np.asarray(rows[f"umask_vmask_{tag}"]) != 0.0
        hpg = np.asarray(rows[f"after_hpg_{tag}"])
        vor = np.asarray(rows[f"after_vor_{tag}"]) - hpg
        adv = np.asarray(rows[f"after_adv_{tag}"]) - np.asarray(
            rows[f"after_vor_{tag}"])
        for name, term, row in (("hpg", hpg, f"after_hpg_{tag}"),
                                ("vorticity", vor, f"after_vor_{tag}"),
                                ("advection", adv, f"after_adv_{tag}")):
            values = term[active]
            term_rms = float(np.sqrt(np.mean(values * values)))
            oracle_terms[f"{name}_{tag}"] = {
                "nemo_term_rms": term_rms,
                "nemo_term_max_abs": float(np.max(np.abs(values), initial=0.0)),
                "cumulative_row": row,
                "cumulative_difference_rms": walk[row]["active_rms"],
                "relative_difference_rms": (
                    walk[row]["active_rms"] / term_rms if term_rms else 0.0),
            }

    first_non_bit = next(
        (f"{name}_{tag}" for name in DEVELOPED_STAGE2_ROWS
         for tag in ("u", "v")
         if walk[f"{name}_{tag}"]["active_cells_unequal"] > 0), None)
    ranking = sorted(
        oracle_terms, key=lambda name: -oracle_terms[name][
            "relative_difference_rms"])

    if plant == "hpg-rank-scale":
        # The scaled row must RISE, above both vorticity rows and above the
        # unscaled pressure-gradient row.  It cannot reach the top: scaling by
        # one part in a million puts it at about 1e-6 relative, which is four
        # orders below the advection and two above the vorticity, and that is
        # the whole point -- the ranking is a measurement, so the control has
        # to predict where the planted row LANDS rather than assume the top.
        require(ranking.index("hpg_u")
                < min(ranking.index("vorticity_u"),
                      ranking.index("vorticity_v"),
                      ranking.index("hpg_v")),
                "the magnitude ranking did not lift the scaled "
                f"pressure-gradient row above the floor rows: {ranking}")
        return {
            "status": "PLANT-FIRED", "plant": plant,
            "control": {"ranking": ranking,
                        "row": oracle_terms["hpg_u"]},
        }

    report = {
        "status": "PASS",
        "case": CASE,
        "step": DEVELOPED_PROCESS_STEP,
        "commit": expected_commit,
        "record": {
            "root": str(stage2_record_root),
            "sha256": oracle["sha256"],
            "field_count": oracle["field_count"],
            "meta": oracle["meta"],
            "ln_dynadv_vec": rows["ln_dynadv_vec"],
            "lk_linssh": rows["lk_linssh"],
        },
        "compiled_citations": {
            "rhs_entry": "stprk3_stg.f90:462",
            "dyn_hpg": "stprk3_stg.f90:497",
            "after_hpg": "stprk3_stg.f90:499",
            "dyn_vor": "stprk3_stg.f90:510",
            "after_vor": "stprk3_stg.f90:512",
            "dyn_adv": "stprk3_stg.f90:525",
            "after_adv": "stprk3_stg.f90:531",
            "stage_update_selector": "stprk3_stg.f90:721",
            "stage_update_vector_arm": "stprk3_stg.f90:723",
            "uu_Kaa_raw": "stprk3_stg.f90:738",
            "barotropic_correction": "stprk3_stg.f90:787",
            "barotropic_correction_applied": "stprk3_stg.f90:810",
        },
        "walk": walk,
        "assignment_calibration": calibration,
        "entry_velocity_substitution": substitution,
        "oracle_term_magnitudes": oracle_terms,
        "entry_window_control": entry_control,
        "budget": budget,
        "first_non_bit_row": first_non_bit,
        "magnitude_ranked_families": ranking,
        "observer_state_unequal_bytes": observer_unequal,
        "predictions": {
            "stage1_entry_non_bit_above_1e_7": (
                walk["stage1_output_u"]["active_cells_unequal"] > 0
                and walk["stage1_output_u"]["active_max_abs"] >= 1.0e-7),
            "first_non_bit_family_is_hpg": (
                walk["after_hpg_u"]["active_cells_unequal"] > 0),
            "budget_closes_within_one_percent": all(
                row["relative_disagreement"] < 0.01
                for row in budget.values()),
            "assignment_calibration_bit": all(
                row["active_cells_unequal"] == 0
                for row in calibration.values()),
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report


DEVELOPED_ADV_SPLIT_PLANTS = ("keg-zad-sum", "nemo-ww-ulp")


def developed_stage2_advection_split(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Split NEMO's vector-invariant ``dyn_adv`` into its two compiled halves.

    Round 157 measured that the developed stage-2 velocity difference is
    owned by ``dyn_adv`` at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525``, entered
    through the vector-invariant arm at ``:523``.  Under that arm
    ``dynadv.f90:171`` calls ``dyn_keg`` (the kinetic-energy gradient,
    ``dynkeg.f90:121-130`` with ``nn_dynkeg = 0``) and ``dynadv.f90:176``
    calls ``dyn_zad`` (the explicit vertical advection,
    ``dynzad.f90:112-126`` with the bottom statement at ``:134-137``), in that
    order, into one shared ``Krhs`` accumulator.

    The record has no boundary between the two halves, so NEMO's own split is
    RECONSTRUCTED rather than read: the kinetic-energy gradient is a pure
    function of ``puu/pvv(Kmm)`` and the static horizontal metrics, all of
    which the record or the card carries, so driving legoESM's PRODUCTION
    stage with NEMO's own entry velocity evaluates that half on NEMO's own
    operands.  The vertical half is then NEMO's total advection increment
    minus that, and the receipt states the assumption instead of hiding it.

    Every arm runs through ``LatLonCGridOceanModel.step`` under production
    just-in-time compilation from NEMO's admitted day-180 entry, never an
    isolated closure (operator note L-amend).
    """
    require(plant in (None, "none") + DEVELOPED_ADV_SPLIT_PLANTS,
            f"unknown developed stage-2 advection plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"advection split requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "advection split commit differs from --expect-commit")
    evidence_root = Path(evidence_root)
    oracle = _developed_stage2_record(Path(stage2_record_root))
    rows = dict(oracle["rows"])

    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    payload = bundle["payload"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])

    observer_unequal: list[int] = []

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary = ordinary_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)

    def run_uv(*, passive=True, **hook_kwargs):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_kwargs))
        result = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(result)
        if passive:
            # A WRITE-only exposure substitutes its own slots AFTER the
            # ordinary step completes, so passivity is judged on every OTHER
            # prognostic field.  An operand-substitution arm changes the step
            # itself and is not passive; it says so rather than being excused.
            neutral = result._replace(u=ordinary.u, v=ordinary.v)
            observer_unequal.append(_state_bit_mismatches(neutral, ordinary))
        return (np.asarray(result.u.data), np.asarray(result.v.data))

    # Window control.  NEMO's recorded before-level velocity is the step-entry
    # level, which this walk loads from the same restart, so it must be BIT
    # against legoESM's entry state.  A wrong window or transpose cannot
    # survive it, and it is the only check that proves the two grids are one
    # grid before any difference is attributed.
    entry_control = {
        tag: _score_stage2_face(
            np.asarray(getattr(state, tag).data),
            rows[f"uu_vv_Kbb_{tag}"], rows[f"umask_vmask_{tag}"])
        for tag in ("u", "v")
    }
    require(all(row["cells_unequal"] == 0 for row in entry_control.values()),
            "the Round-156 record window does not land on legoESM's face "
            f"grid: {entry_control}")

    # T-WINDOW CONTROL.  The vertical velocity this walk substitutes is a
    # T-point field, and the entry control above proves only the two FACE
    # windows.  Two properties of NEMO's own arrays pin the cell window:
    # ``eos`` leaves the density anomaly identically zero on every dry column
    # (104 of them on this card), and ``sshwzv`` leaves ``ww`` identically
    # zero at the bottom interface.  A window off by one cell breaks the land
    # pattern, which is the only check available here because legoESM's own
    # vertical velocity is not bit-equal to NEMO's.
    nemo_dry_column = np.all(np.asarray(rows["rhd_t"]) == 0.0, axis=-1)
    lego_dry_column = np.asarray(state.land_mask.data) <= 0.5
    t_window_control = {
        "nemo_dry_columns": int(np.count_nonzero(nemo_dry_column)),
        "lego_dry_columns": int(np.count_nonzero(lego_dry_column)),
        "columns_disagreeing": int(
            np.count_nonzero(nemo_dry_column != lego_dry_column)),
        "ww_bottom_interface_max_abs": float(
            np.max(np.abs(np.asarray(rows["ww_t"])[..., -1]), initial=0.0)),
        "ww_dry_column_max_abs": float(np.max(
            np.abs(np.asarray(rows["ww_t"])[nemo_dry_column]), initial=0.0)),
    }
    # Only the dry-column pattern DISCRIMINATES: the bottom interface is
    # architecturally zero in NEMO, so it reads 0.0 for every candidate
    # window and pins nothing.  It stays as a sanity row, labelled, and the
    # refusal rests on the land pattern alone.
    t_window_control["bottom_interface_row_is_not_discriminating"] = True
    require(t_window_control["columns_disagreeing"] == 0,
            "the Round-156 record cell window does not land on legoESM's "
            f"T grid: {t_window_control}")

    def expose(name):
        return run_uv(expose_momentum_operator=name,
                      expose_momentum_operator_stage=2)

    production = {
        "keg": expose("keg"),
        "zad": expose("zad"),
        "advection": expose("advection"),
        "after_adv": run_uv(expose_stage2_momentum_rhs=True),
    }

    # NEMO's own advection increment.  It is the difference of two recorded
    # cumulative snapshots, so it carries one rounding on the oracle side and
    # is context for the split, never a scored production row.
    nemo_increment = {
        tag: (np.asarray(rows[f"after_adv_{tag}"])
              - np.asarray(rows[f"after_vor_{tag}"]))
        for tag in ("u", "v")
    }

    # Authority control: this walk's production after-advection row must
    # reproduce the round-157 row it is splitting, or its split is a split of
    # a different number.
    authority = {
        tag: _score_stage2_face(
            production["after_adv"][index], rows[f"after_adv_{tag}"],
            rows[f"umask_vmask_{tag}"])
        for tag, index in (("u", 0), ("v", 1))
    }
    ROUND157_AFTER_ADV_RMS = {"u": 3.844166e-12, "v": 6.428546e-12}
    for tag, row in authority.items():
        # Round 157 printed seven significant digits, so the comparison
        # carries that rounding; the tolerance is still 1e-6 relative, which
        # binds far below any move this round could make.
        require(abs(row["active_rms"] - ROUND157_AFTER_ADV_RMS[tag])
                <= 5.0e-18,
                f"the production after-advection row moved from round 157 "
                f"on {tag}: {row['active_rms']} vs "
                f"{ROUND157_AFTER_ADV_RMS[tag]}")

    # Split control: the two halves must reproduce the bucket they came out
    # of, so the split adds no term of its own.
    halves_sum = {
        tag: production["keg"][index] + production["zad"][index]
        for tag, index in (("u", 0), ("v", 1))
    }
    halves_control = {
        tag: _score_stage2_face(
            halves_sum[tag], production["advection"][index],
            rows[f"umask_vmask_{tag}"])
        for tag, index in (("u", 0), ("v", 1))
    }
    require(all(row["active_cells_unequal"] == 0
                for row in halves_control.values()),
            "legoESM's two advection halves do not sum to the bucket they "
            f"came out of: {halves_control}")
    if plant == "keg-zad-sum":
        # The control has to catch a half that is not the half it is named.
        planted = {
            tag: halves_sum[tag] * np.float64(1.0 + 1.0e-12)
            for tag in ("u", "v")
        }
        planted_control = {
            tag: _score_stage2_face(
                planted[tag], production["advection"][index],
                rows[f"umask_vmask_{tag}"])
            for tag, index in (("u", 0), ("v", 1))
        }
        require(any(row["active_cells_unequal"] > 0
                    for row in planted_control.values()),
                "the halves-sum control did NOT catch a sum scaled by one "
                f"part in a million million: {planted_control}")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": planted_control}

    # ARM W.  NEMO's own recorded vertical velocity at the stage-2 dyn_zad
    # call, every other stage input left at legoESM's value.  dyn_keg reads no
    # vertical velocity, so its row in this arm is a CONTROL: it has to be
    # byte-identical to the production row or the hook reaches further than
    # dyn_zad.
    ww_arm = {}
    for component in ("keg", "zad", "advection"):
        ww_arm[component] = run_uv(
            passive=False,
            stage2_zad_operand_override=(jnp.asarray(rows["ww_t"]), None, None),
            expose_momentum_operator=component,
            expose_momentum_operator_stage=2)
    ww_arm["after_adv"] = run_uv(
        passive=False,
        stage2_zad_operand_override=(jnp.asarray(rows["ww_t"]), None, None),
        expose_stage2_momentum_rhs=True)

    if plant == "nemo-ww-ulp":
        # The substituted operand must be LIVE inside dyn_zad, not merely
        # accepted by the hook.  Install NEMO's recorded vertical velocity a
        # second time with one cell moved by a single unit in the last place;
        # the arm's vertical-advection field has to move with it.
        # The cell has to be one the routine READS.  dynzad.f90:112-114 takes
        # ww(jk+1) over jk = 1..jpk-2, so it reads the INTERIOR interfaces
        # only: the surface interface and the bottom one are never touched.
        # The first version of this plant moved the surface interface, which
        # NEMO leaves non-zero, and it fired because nothing consumed the
        # moved value -- a control that perturbs what the consumer does not
        # read proves nothing.  This one moves the largest interior
        # interface, which every downstream statement depends on.
        moved = np.array(rows["ww_t"], copy=True)
        interior = np.abs(moved[..., 1:moved.shape[-1] - 1])
        row, column, level = np.unravel_index(
            int(np.argmax(interior)), interior.shape)
        index = (int(row), int(column), int(level) + 1)
        # A single unit in the last place moved exactly one face last time,
        # one rounding from announcing nothing.  One part in a million
        # million is still far below every difference this walk attributes
        # and leaves the control a margin.
        moved[index] = moved[index] * np.float64(1.0 + 1.0e-12)
        planted_zad = run_uv(
            passive=False,
            stage2_zad_operand_override=(jnp.asarray(moved), None, None),
            expose_momentum_operator="zad",
            expose_momentum_operator_stage=2)
        moved_rows = {
            tag: _score_stage2_face(
                planted_zad[component], ww_arm["zad"][component],
                rows[f"umask_vmask_{tag}"])
            for tag, component in (("u", 0), ("v", 1))
        }
        require(any(row["active_cells_unequal"] > 0
                    for row in moved_rows.values()),
                "a one-unit-in-the-last-place move of NEMO's recorded "
                "vertical velocity did not reach the stage-2 vertical "
                f"advection: {moved_rows}")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": {"moved_cell": list(index), "rows": moved_rows}}

    # LIVENESS.  NEMO's vertical velocity is not legoESM's, so a substitution
    # that reached dyn_zad must have moved its output.  An inert hook would
    # otherwise report "the vertical velocity removes nothing" while never
    # having been read.
    ww_liveness = {
        tag: _score_stage2_face(
            ww_arm["zad"][index], production["zad"][index],
            rows[f"umask_vmask_{tag}"])
        for tag, index in (("u", 0), ("v", 1))
    }
    require(all(row["active_cells_unequal"] > 0
                for row in ww_liveness.values()),
            "the stage-2 vertical-velocity substitution left the vertical "
            f"advection unchanged, so it was never read: {ww_liveness}")

    # ARM E.  NEMO's own stage-2 entry velocity, which is the ONLY dynamic
    # operand the kinetic-energy gradient reads (dynkeg.f90:121-124 takes
    # puu/pvv(Kmm) and nothing else), with every other entry field left at
    # legoESM's value.  This is round 157's substitution, re-run with the two
    # halves exposed instead of the total.
    stage1_out = run_uv(expose_momentum_stage=1)

    def stage_entry_arm(velocity, *, name):
        armed = {}
        # Only the kinetic-energy half is needed from these arms: it is the
        # one quantity the split measures on NEMO's own operands, and the
        # vertical half is then NEMO's total increment minus it.  The
        # right-hand-side total is the null the removed fractions use.
        for component in ("keg",):
            armed[component] = run_uv(
                passive=False,
                stage_entry_override=(2, velocity[0], velocity[1],
                                      *_entry_tracers),
                expose_momentum_operator=component,
                expose_momentum_operator_stage=2)
        armed["after_adv"] = run_uv(
            passive=False,
            stage_entry_override=(2, velocity[0], velocity[1],
                                  *_entry_tracers),
            expose_stage2_momentum_rhs=True)
        armed["_name"] = name
        return armed

    _tracer_state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_tracer_stage=1)).step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)
    _entry_tracers = (_tracer_state.T.data, _tracer_state.S.data,
                      _tracer_state.eta.data)

    arms = {
        "null": stage_entry_arm(
            (jnp.asarray(stage1_out[0]), jnp.asarray(stage1_out[1])),
            name="legoESM's own stage-1 output velocity"),
        "nemo_entry_velocity": stage_entry_arm(
            (jnp.asarray(rows["uu_vv_Kmm_u"]), jnp.asarray(rows["uu_vv_Kmm_v"])),
            name="NEMO's recorded uu/vv(Kmm)"),
    }

    def score_against_nemo_total(frame):
        return {
            tag: _score_stage2_face(
                frame[index], rows[f"after_adv_{tag}"],
                rows[f"umask_vmask_{tag}"])
            for tag, index in (("u", 0), ("v", 1))
        }

    scored = {
        "production": score_against_nemo_total(production["after_adv"]),
        "null_entry": score_against_nemo_total(arms["null"]["after_adv"]),
        "nemo_entry_velocity": score_against_nemo_total(
            arms["nemo_entry_velocity"]["after_adv"]),
        "nemo_stage2_ww": score_against_nemo_total(ww_arm["after_adv"]),
    }
    base = scored["null_entry"]
    for name, row in scored.items():
        for tag in ("u", "v"):
            base_rms = base[tag]["active_rms"]
            base_max = base[tag]["active_max_abs"]
            row[tag]["rms_removed_fraction"] = (
                float((base_rms - row[tag]["active_rms"]) / base_rms)
                if base_rms > 0.0 else 0.0)
            row[tag]["max_abs_removed_fraction"] = (
                float((base_max - row[tag]["active_max_abs"]) / base_max)
                if base_max > 0.0 else 0.0)
    # The ww arm has its OWN null: it changes nothing but the vertical
    # velocity, so its removed fractions are taken against the production
    # row, which is the same program with the live operand.
    for tag in ("u", "v"):
        prod_rms = scored["production"][tag]["active_rms"]
        scored["nemo_stage2_ww"][tag]["rms_removed_vs_production"] = (
            float((prod_rms - scored["nemo_stage2_ww"][tag]["active_rms"])
                  / prod_rms) if prod_rms > 0.0 else 0.0)

    # dyn_keg reads no vertical velocity: the ww arm's kinetic-energy row is
    # a hook-reach control, not a measurement.
    keg_reach_control = {
        tag: _score_stage2_face(
            ww_arm["keg"][index], production["keg"][index],
            rows[f"umask_vmask_{tag}"])
        for tag, index in (("u", 0), ("v", 1))
    }

    # THE SPLIT.  NEMO's kinetic-energy half is legoESM's production stage
    # driven by NEMO's own entry velocity; NEMO's vertical half is its total
    # advection increment minus that.  The assumption -- that legoESM's
    # transcription of dynkeg.f90:121-130 is NEMO's statement -- is stated in
    # the receipt and is the only step in this split that is read off the
    # source rather than measured.
    split = {}
    for tag, index in (("u", 0), ("v", 1)):
        mask = rows[f"umask_vmask_{tag}"]
        keg_nemo = arms["nemo_entry_velocity"]["keg"][index]
        keg_null = arms["null"]["keg"][index]
        zad_nemo_implied = nemo_increment[tag] - np.asarray(keg_nemo)
        d_keg = np.asarray(production["keg"][index]) - np.asarray(keg_nemo)
        d_zad = np.asarray(production["zad"][index]) - zad_nemo_implied
        active = np.asarray(mask) != 0.0
        def rms(values):
            picked = values[active]
            return float(np.sqrt(np.mean(picked * picked))
                         if picked.size else 0.0)
        split[tag] = {
            "nemo_increment_rms": rms(nemo_increment[tag]),
            "production_keg_rms": rms(np.asarray(production["keg"][index])),
            "production_zad_rms": rms(np.asarray(production["zad"][index])),
            "keg_difference_rms": rms(d_keg),
            "zad_difference_rms": rms(d_zad),
            "keg_difference_max_abs": float(
                np.max(np.abs(d_keg[active]), initial=0.0)),
            "zad_difference_max_abs": float(
                np.max(np.abs(d_zad[active]), initial=0.0)),
            # The substitution mechanism is not byte-neutral (round 157
            # measured its offset at 1.7e-21), so the mechanism's own cost on
            # the kinetic-energy half is registered next to the number it
            # could contaminate rather than assumed away.
            "substitution_mechanism_offset_max_abs": float(np.max(np.abs(
                (np.asarray(keg_null)
                 - np.asarray(production["keg"][index]))[active]),
                initial=0.0)),
            "sum_of_halves_rms": rms(d_keg + d_zad),
            "total_difference_rms": scored["production"][tag]["active_rms"],
            "owner": ("zad" if rms(d_zad) > rms(d_keg) else "keg"),
            "owner_ratio": (rms(d_zad) / rms(d_keg)) if rms(d_keg) else
                           float("inf"),
        }
        zad_on_nemo_ww = np.asarray(ww_arm["zad"][index])
        closure = (zad_on_nemo_ww - zad_nemo_implied)[active]
        split[tag]["zad_on_nemo_ww_vs_implied_rms"] = float(
            np.sqrt(np.mean(closure * closure)) if closure.size else 0.0)
        split[tag]["zad_on_nemo_ww_vs_implied_max_abs"] = float(
            np.max(np.abs(closure), initial=0.0))
        split[tag]["sum_reproduces_total"] = (
            abs(split[tag]["sum_of_halves_rms"]
                - split[tag]["total_difference_rms"])
            / split[tag]["total_difference_rms"]
            if split[tag]["total_difference_rms"] else 0.0)

    require(all(row["cells_unequal"] == 0
                for row in keg_reach_control.values()),
            "the stage-2 vertical-velocity substitution reached the "
            f"kinetic-energy gradient, which reads no vertical velocity: "
            f"{keg_reach_control}")
    require(all(value == 0 for value in observer_unequal),
            "an advection-split exposure hook moved the production state "
            f"outside u/v: {observer_unequal}")

    report = {
        "status": "PASS",
        "case": CASE,
        "step": DEVELOPED_PROCESS_STEP,
        "commit": expected_commit,
        "record": {
            "root": str(stage2_record_root),
            "sha256": oracle["sha256"],
            "field_count": oracle["field_count"],
            "meta": oracle["meta"],
        },
        "compiled_citations": {
            "dyn_adv_call": "stprk3_stg.f90:525",
            "vector_arm_selector": "stprk3_stg.f90:523",
            "dynadv_vector_case": "dynadv.f90:153",
            "dyn_keg_call": "dynadv.f90:171",
            "dyn_zad_call": "dynadv.f90:176",
            "keg_c2_energy": "dynkeg.f90:121-125",
            "keg_c2_gradient": "dynkeg.f90:129-130",
            "zad_transport": "dynzad.f90:112-114",
            "zad_shear": "dynzad.f90:119-120",
            "zad_update": "dynzad.f90:123-126",
            "zad_bottom": "dynzad.f90:134-137",
        },
        "entry_window_control": entry_control,
        "t_window_control": t_window_control,
        "authority_control": authority,
        "halves_control": halves_control,
        "keg_hook_reach_control": keg_reach_control,
        "ww_substitution_liveness": ww_liveness,
        "arms_scored_against_nemo_after_adv": scored,
        "split": split,
        "observer_state_unequal_bytes": observer_unequal,
        "predictions": {
            "halves_sum_is_the_bucket": all(
                row["active_cells_unequal"] == 0
                for row in halves_control.values()),
            "zad_owns_both_components": all(
                split[tag]["owner"] == "zad" and split[tag]["owner_ratio"] >= 2.0
                for tag in ("u", "v")),
            "split_accounts_for_the_total": all(
                split[tag]["sum_reproduces_total"] < 0.01
                for tag in ("u", "v")),
            "keg_hook_reach_is_zero": all(
                row["cells_unequal"] == 0
                for row in keg_reach_control.values()),
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report


DEVELOPED_WZV_WALK_PLANTS = ("wzv-cell-window", "wzv-form-inert")


def developed_stage2_wzv_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Walk the PRODUCER of the stage-2 vertical velocity, not its consumer.

    Round 158 measured that installing NEMO's own recorded stage-2 vertical
    velocity at the vertical-advection call removes 99.7% of the stage
    right-hand-side difference, so the advection arithmetic is exonerated and
    the owner is whatever builds that field.

    NEMO solves continuity TWICE per stage on this deck.  The stage program
    takes the vector-invariant arm at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356`` and calls
    ``wzv`` on the RAW stage velocity at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360``; the
    flux-form call on the already-corrected transports at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:367`` sits in
    the ``ELSE`` and does not execute.  The tracer transport then re-solves
    continuity in the transport form at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`` and
    overwrites the same array before forming its own vertical transport at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:279-281``.  The two
    indicator branches are
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:123-130`` and
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:132-138``.

    legoESM builds ONE stage vertical velocity, in the transport form, and
    hands it to both consumers.  This walk measures what that costs.

    Every arm is one production step through ``LatLonCGridOceanModel.step``
    under production just-in-time compilation from NEMO's admitted day-180
    entry, never an isolated closure (operator note L-amend).
    """
    require(plant in (None, "none") + DEVELOPED_WZV_WALK_PLANTS,
            f"unknown developed stage-2 wzv plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"wzv walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "wzv walk commit differs from --expect-commit")
    evidence_root = Path(evidence_root)
    oracle = _developed_stage2_record(Path(stage2_record_root))
    rows = dict(oracle["rows"])

    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    payload = bundle["payload"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])

    nlev = int(np.asarray(state.T.data).shape[-1])
    nemo_ww = np.ascontiguousarray(np.asarray(rows["ww_t"])[..., :nlev])
    cell_mask = np.asarray(gate.expected_masks(card)["T"])[..., :nlev]

    observer_unequal: list[int] = []
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary = ordinary_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)

    def run(*, passive=True, **hook_kwargs):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_kwargs))
        result = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(result)
        if passive:
            # A WRITE-only exposure substitutes its slots AFTER the ordinary
            # step completes, so passivity is judged on every OTHER field.
            neutral = result._replace(
                u=ordinary.u, v=ordinary.v, T=ordinary.T)
            observer_unequal.append(_state_bit_mismatches(neutral, ordinary))
        return result

    def stage2_w(*, form, entry=None, passive=True):
        hooks = dict(expose_tracer_transport_stage=2,
                     expose_tracer_transport_as_ww=True,
                     stage2_wzv_velocity_form=form)
        if entry is not None:
            hooks["stage_entry_override"] = entry
        return np.asarray(run(passive=passive, **hooks).T.data)

    def stage2_rhs(*, form, entry=None, passive=True):
        hooks = dict(expose_stage2_momentum_rhs=True,
                     stage2_wzv_velocity_form=form)
        if entry is not None:
            hooks["stage_entry_override"] = entry
        result = run(passive=passive, **hooks)
        return (np.asarray(result.u.data), np.asarray(result.v.data))

    # FACE-WINDOW CONTROL.  NEMO's recorded step-entry velocity is the level
    # this walk loads from the same restart, so it must be BIT on legoESM's
    # face grid before any difference is attributed.
    entry_control = {
        tag: _score_stage2_face(
            np.asarray(getattr(state, tag).data),
            rows[f"uu_vv_Kbb_{tag}"], rows[f"umask_vmask_{tag}"])
        for tag in ("u", "v")
    }
    require(all(row["cells_unequal"] == 0 for row in entry_control.values()),
            "the Round-156 record window does not land on legoESM's face "
            f"grid: {entry_control}")

    # CELL-WINDOW CONTROL.  The vertical velocity is a T-point field, so the
    # face control does not cover it.  What pins the cell window is the
    # dry-column pattern: NEMO's density anomaly is identically zero on every
    # dry column and legoESM has the same 104 of them (round 158 enumerated
    # all 25 candidate windows and exactly one satisfies it).
    nemo_dry_column = np.all(np.asarray(rows["rhd_t"]) == 0.0, axis=-1)
    lego_dry_column = np.asarray(state.land_mask.data) <= 0.5
    if plant == "wzv-cell-window":
        # A window one cell off has to be refused, in BOTH directions: round
        # 158's first attempt at this window WAS one cell off and this control
        # is what caught it, so the plant reproduces that failure deliberately.
        # A blind control must NOT be able to print the fired marker, so the
        # arm that was caught RETURNS and the arm that was not RAISES.
        caught = {}
        for name, axis in (("shifted_one_column_east", 1),
                           ("shifted_one_row_north", 0)):
            rolled = np.roll(nemo_dry_column, 1, axis=axis)
            caught[name] = int(
                np.count_nonzero(rolled != lego_dry_column))
        blind = [name for name, count in caught.items() if count == 0]
        if blind:
            raise GateError(
                "PLANT-BLIND: the cell-window control did not refuse "
                f"{blind}: {caught}")
        return {"status": "PLANT-FIRED", "plant": plant, "control": caught}
    cell_window = {
        "nemo_dry_columns": int(np.count_nonzero(nemo_dry_column)),
        "lego_dry_columns": int(np.count_nonzero(lego_dry_column)),
        "columns_disagreeing": int(
            np.count_nonzero(nemo_dry_column != lego_dry_column)),
    }
    require(cell_window["columns_disagreeing"] == 0,
            "the Round-156 record cell window does not land on legoESM's T "
            f"grid: {cell_window}")
    # MASK CONTROL.  The scored mask is the card's own wet-cell mask, so it
    # has to agree with NEMO's: the oracle's vertical velocity must be
    # identically zero everywhere the card calls dry.
    off_mask_max = float(np.max(np.abs(nemo_ww[~(cell_mask != 0.0)]),
                                initial=0.0))
    require(off_mask_max == 0.0,
            "NEMO's stage-2 vertical velocity is non-zero where the card's "
            f"mask is dry, so the two masks are not one mask: {off_mask_max}")

    production_w = stage2_w(form=False)
    form_w = stage2_w(form=True, passive=False)

    # LIVENESS.  The arm must actually change the solve; an inert flag would
    # otherwise report "the call form removes nothing" while never firing.
    form_liveness = _score_stage2_face(form_w, production_w, cell_mask)
    if plant == "wzv-form-inert":
        # Claim the arm is on while leaving it off.  The liveness refusal has
        # to catch a run that never selected the other call form.  As above,
        # a blind control cannot print the fired marker: being caught is
        # ``active_cells_unequal == 0``, and anything else is PLANT-BLIND.
        inert = stage2_w(form=False, passive=False)
        inert_liveness = _score_stage2_face(inert, production_w, cell_mask)
        if inert_liveness["active_cells_unequal"] > 0:
            raise GateError(
                "PLANT-BLIND: the liveness control did not refuse an arm "
                f"that never selected the other call form: {inert_liveness}")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": inert_liveness}
    require(form_liveness["active_cells_unequal"] > 0,
            "selecting NEMO's velocity-indicator call form left the stage-2 "
            f"vertical velocity unchanged, so it never fired: {form_liveness}")

    # The stage-2 entry velocity arm: NEMO's own uu/vv(Kmm) with legoESM's
    # production call form, which separates "the form is wrong" from "the
    # velocity handed to it is inherited wrong from stage 1".
    tracer_entry = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_tracer_stage=1)).step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)
    entry_override = (
        2, jnp.asarray(rows["uu_vv_Kmm_u"]), jnp.asarray(rows["uu_vv_Kmm_v"]),
        tracer_entry.T.data, tracer_entry.S.data, tracer_entry.eta.data)
    stage1_out = run(expose_momentum_stage=1)
    lego_entry_override = (
        2, jnp.asarray(np.asarray(stage1_out.u.data)),
        jnp.asarray(np.asarray(stage1_out.v.data)),
        tracer_entry.T.data, tracer_entry.S.data, tracer_entry.eta.data)
    null_w = stage2_w(form=False, entry=lego_entry_override, passive=False)
    entry_w = stage2_w(form=False, entry=entry_override, passive=False)
    both_w = stage2_w(form=True, entry=entry_override, passive=False)

    scored = {
        "production": _score_stage2_face(production_w, nemo_ww, cell_mask),
        "null_entry": _score_stage2_face(null_w, nemo_ww, cell_mask),
        "velocity_form": _score_stage2_face(form_w, nemo_ww, cell_mask),
        "nemo_entry_velocity": _score_stage2_face(
            entry_w, nemo_ww, cell_mask),
        "velocity_form_and_nemo_entry": _score_stage2_face(
            both_w, nemo_ww, cell_mask),
    }
    active = cell_mask != 0.0
    own_rms = float(np.sqrt(np.mean(nemo_ww[active] ** 2)))
    base = scored["production"]["active_rms"]
    base_max = scored["production"]["active_max_abs"]
    for row in scored.values():
        row["relative_to_own_rms"] = (
            float(row["active_rms"] / own_rms) if own_rms > 0.0 else 0.0)
        row["rms_removed_fraction"] = (
            float((base - row["active_rms"]) / base) if base > 0.0 else 0.0)
        row["max_abs_removed_fraction"] = (
            float((base_max - row["active_max_abs"]) / base_max)
            if base_max > 0.0 else 0.0)

    # What the call form does to the consumer round 158 named, against the
    # same NEMO row and the same ceiling.
    production_rhs = stage2_rhs(form=False)
    form_rhs = stage2_rhs(form=True, passive=False)
    rhs = {}
    for name, frame in (("production", production_rhs),
                        ("velocity_form", form_rhs)):
        rhs[name] = {
            tag: _score_stage2_face(
                frame[index], rows[f"after_adv_{tag}"],
                rows[f"umask_vmask_{tag}"])
            for tag, index in (("u", 0), ("v", 1))
        }
    ROUND158 = {
        "production": {"u": 3.844166e-12, "v": 6.428546e-12},
        "nemo_ww_ceiling": {"u": 1.121563e-14, "v": 1.308047e-14},
    }
    for tag in ("u", "v"):
        # AUTHORITY after round 163: production now selects NEMO's second
        # continuity solve.  It must sit on the independently measured
        # round-158 oracle-ww ceiling, rather than reproduce round 158's old
        # single-solve production row.  The historical value stays in the
        # report as the before arm.
        require(rhs["production"][tag]["active_rms"]
                <= 1.01 * ROUND158["nemo_ww_ceiling"][tag],
                "the landed production after-advection row exceeds the "
                f"round-158 oracle-ww ceiling on {tag}: "
                f"{rhs['production'][tag]['active_rms']}")
        rhs["velocity_form"][tag]["rms_removed_fraction"] = float(
            (rhs["production"][tag]["active_rms"]
             - rhs["velocity_form"][tag]["active_rms"])
            / rhs["production"][tag]["active_rms"])
        rhs["velocity_form"][tag]["times_the_nemo_ww_ceiling"] = float(
            rhs["velocity_form"][tag]["active_rms"]
            / ROUND158["nemo_ww_ceiling"][tag])

    # THE TRACER CONSEQUENCE.  NEMO's tracer transport re-solves continuity in
    # the transport form, so correcting the MOMENTUM vertical velocity must
    # not touch the tracer terms -- but legoESM shares one field between the
    # two consumers, so changing it moves the one-step tracer state.  That
    # measured move is the reason a landing has to produce two vertical
    # velocities instead of changing the shared one.
    form_plain = run(passive=False, stage2_wzv_velocity_form=True)
    tracer_cells = cell_mask != 0.0
    tracer = {}
    for name, field in (("T", "T"), ("S", "S")):
        before = np.asarray(getattr(ordinary, field).data)[tracer_cells]
        after = np.asarray(getattr(form_plain, field).data)[tracer_cells]
        delta = after - before
        tracer[name] = {
            "cells_scored": int(before.size),
            "cells_unequal": int(np.count_nonzero(delta != 0.0)),
            "one_step_rms": float(np.sqrt(np.mean(delta * delta))),
            "one_step_max_abs": float(np.max(np.abs(delta), initial=0.0)),
        }
    ROUND152_TERM_RMS = {"zdf": 2.1834e-5, "tracer_ldf": 1.0655e-5}
    tracer["round152_term_rms_for_scale"] = ROUND152_TERM_RMS
    tracer["T"]["as_fraction_of_the_zdf_term"] = float(
        tracer["T"]["one_step_rms"] / ROUND152_TERM_RMS["zdf"])

    require(all(count == 0 for count in observer_unequal),
            f"a passive exposure moved production state: {observer_unequal}")

    report = {
        "format": "gyre-round159-developed-stage2-wzv-walk-v1",
        "status": "PASS",
        "worktree": stamp,
        "record": {"sha256": oracle["sha256"], "meta": oracle["meta"]},
        "entry": {"step": DEVELOPED_ENTRY_STEP,
                  "process_step": DEVELOPED_PROCESS_STEP},
        "controls": {
            "entry_face_window": entry_control,
            "cell_window": cell_window,
            "nemo_ww_off_mask_max_abs": off_mask_max,
            "velocity_form_liveness": form_liveness,
            "passive_observer_unequal": observer_unequal,
        },
        "nemo_ww_own_rms": own_rms,
        "vertical_velocity_scored_against_nemo": scored,
        "stage2_rhs_scored_against_nemo": rhs,
        "round158_reference": ROUND158,
        "tracer_consequence_of_changing_the_shared_field": tracer,
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report


def _state_leaf_move(left, right) -> dict:
    """Largest move between two state pytrees, in row-scale last places.

    ``_state_bit_mismatches`` counts unequal bytes, which cannot tell a
    scheduling difference in the last bits from a real perturbation.  This
    reports both: how many bytes differ, and the largest difference expressed
    in units of the last place of the field's own largest value -- the same
    row-scale unit the ladder's move gate uses.
    """
    import jax

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    require(left_tree == right_tree, "production state pytree structures differ")
    worst = {"unequal_bytes": 0, "max_abs": 0.0, "max_row_scale_ulps": 0.0,
             "leaf": None}
    for number, (left_leaf, right_leaf) in enumerate(
            zip(left_leaves, right_leaves, strict=True)):
        a = np.asarray(left_leaf)
        b = np.asarray(right_leaf)
        require(a.shape == b.shape and a.dtype == b.dtype,
                f"production state leaf {number} shape/dtype differs")
        unequal = int(np.count_nonzero(
            a.view(np.uint8).reshape(-1) != b.view(np.uint8).reshape(-1)))
        if unequal == 0:
            continue
        worst["unequal_bytes"] += unequal
        if not np.issubdtype(a.dtype, np.floating):
            # A non-float leaf that moves at all is a real perturbation.
            worst.update(leaf=number, max_abs=float("inf"),
                         max_row_scale_ulps=float("inf"))
            continue
        delta = float(np.max(np.abs(
            a.astype(np.float64) - b.astype(np.float64)), initial=0.0))
        scale = float(np.spacing(max(
            float(np.max(np.abs(b.astype(np.float64)), initial=0.0)), 1.0)))
        ulps = delta / scale
        if ulps > worst["max_row_scale_ulps"]:
            worst.update(leaf=number, max_abs=delta,
                         max_row_scale_ulps=ulps)
    return worst


DEVELOPED_WZV_SPLIT_PLANTS = ("wzv-split-shared", "wzv-split-inert")


def developed_stage2_wzv_split_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Score NEMO's SECOND per-stage continuity solve against the oracle.

    Round 159 measured that the oracle solves continuity twice per stage on
    this deck: once on the RAW stage velocity for the momentum vertical
    advection (``stprk3_stg.f90:360``, the velocity indicator at
    ``divhor.f90:126-130``) and once on the barotropically corrected
    transports for the tracer transport (``traadv.f90:274``, the transport
    indicator at ``divhor.f90:134-138``), the second overwriting the same
    array.  legoESM solved it once.  Round 160 builds the second solve at
    stages 2 and 3; this walk scores it.

    Three things have to hold at once and each has its own row: the momentum
    field moves onto the oracle's, the TRACER field does not move at all, and
    the stage-2 right-hand side lands on round 158's oracle-``ww`` ceiling.
    Every arm is one production step through ``LatLonCGridOceanModel.step``
    under production just-in-time compilation from the oracle's admitted
    day-180 entry, never an isolated closure (operator note L-amend).
    """
    require(plant in (None, "none") + DEVELOPED_WZV_SPLIT_PLANTS,
            f"unknown developed stage-2 wzv split plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        nemo_stage_momentum_wzv_resolved)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"wzv split walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "wzv split walk commit differs from --expect-commit")
    evidence_root = Path(evidence_root)
    oracle = _developed_stage2_record(Path(stage2_record_root))
    rows = dict(oracle["rows"])

    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    payload = bundle["payload"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])

    require(nemo_stage_momentum_wzv_resolved(card.recipe.model_config),
            "the GYRE card does not resolve NEMO's two-solve stage program, "
            "so this walk would score a program the card cannot reach")

    nlev = int(np.asarray(state.T.data).shape[-1])
    nemo_ww = np.ascontiguousarray(np.asarray(rows["ww_t"])[..., :nlev])
    cell_mask = np.asarray(gate.expected_masks(card)["T"])[..., :nlev]

    observer_unequal: list[int] = []

    def step(**hook_kwargs):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_kwargs)
            if hook_kwargs else None)
        result = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(result)
        return result

    # The two ordinary steps: production (the single shared solve, which is
    # what the tip runs because round 160 is HELD) and the candidate arm (the
    # second solve, behind its private one-variable selector).
    before_plain = step()
    after_plain = step(nemo_stage_momentum_wzv_split=True)

    def run(base, *, passive=True, wrote=("u", "v", "T"), label="",
            **hook_kwargs):
        result = step(**hook_kwargs)
        if passive:
            # A WRITE-only exposure substitutes its OWN slots AFTER the
            # ordinary step completes, so passivity is judged on every other
            # field.  ``wrote`` names exactly the slots this exposure writes,
            # so an exposure that touches a field it did not declare is
            # caught rather than excused.
            #
            # Asking the compiler for an extra output changes how it schedules
            # the rest, and the campaign has measured that floor many times
            # (operator notes L and AL).  So the control is a MAGNITUDE, in
            # the campaign's own row-scale units: no untouched field may move
            # by more than MAX_ULP_MOVE units in the last place of its own
            # largest value.  Every non-zero move is reported, never excused.
            neutral = result._replace(
                **{name: getattr(base, name) for name in wrote})
            row = _state_leaf_move(neutral, base)
            row["arm"] = label or ",".join(sorted(hook_kwargs))
            observer_unequal.append(row)
        return result

    def stage_w(stage, *, momentum, legacy=False, base=None, **extra):
        hooks = dict(expose_tracer_transport_stage=stage,
                     expose_tracer_transport_as_ww=True,
                     expose_stage_momentum_w=momentum)
        if not legacy:
            hooks["nemo_stage_momentum_wzv_split"] = True
        hooks.update(extra)
        arm_base = base if base is not None else (
            before_plain if legacy else after_plain)
        tag = "before" if legacy else "after"
        kind = "momentum" if momentum else "tracer"
        return np.asarray(
            run(arm_base, passive=not extra,
                label=f"stage{stage}-{kind}-w-{tag}", **hooks).T.data)

    def stage2_rhs(*, legacy=False):
        hooks = dict(expose_stage2_momentum_rhs=True)
        if not legacy:
            hooks["nemo_stage_momentum_wzv_split"] = True
        base = before_plain if legacy else after_plain
        result = run(base, wrote=("u", "v"),
                     label="stage2-rhs-" + ("before" if legacy else "after"),
                     **hooks)
        return (np.asarray(result.u.data), np.asarray(result.v.data))

    def stage2_out(*, legacy=False):
        hooks = dict(expose_momentum_stage=2)
        if not legacy:
            hooks["nemo_stage_momentum_wzv_split"] = True
        base = before_plain if legacy else after_plain
        result = run(base, wrote=("u", "v"),
                     label="stage2-out-" + ("before" if legacy else "after"),
                     **hooks)
        return (np.asarray(result.u.data), np.asarray(result.v.data))

    # FACE-WINDOW CONTROL, the same one round 159 used.
    entry_control = {
        tag: _score_stage2_face(
            np.asarray(getattr(state, tag).data),
            rows[f"uu_vv_Kbb_{tag}"], rows[f"umask_vmask_{tag}"])
        for tag in ("u", "v")
    }
    require(all(row["cells_unequal"] == 0 for row in entry_control.values()),
            "the Round-156 record window does not land on legoESM's face "
            f"grid: {entry_control}")

    # CELL-WINDOW CONTROL: the dry-column pattern, both directions.
    nemo_dry_column = np.all(np.asarray(rows["rhd_t"]) == 0.0, axis=-1)
    lego_dry_column = np.asarray(state.land_mask.data) <= 0.5
    cell_window = {
        "nemo_dry_columns": int(np.count_nonzero(nemo_dry_column)),
        "lego_dry_columns": int(np.count_nonzero(lego_dry_column)),
        "columns_disagreeing": int(
            np.count_nonzero(nemo_dry_column != lego_dry_column)),
    }
    require(cell_window["columns_disagreeing"] == 0,
            "the Round-156 record cell window does not land on legoESM's T "
            f"grid: {cell_window}")
    off_mask_max = float(np.max(np.abs(nemo_ww[~(cell_mask != 0.0)]),
                                initial=0.0))
    require(off_mask_max == 0.0,
            "NEMO's stage-2 vertical velocity is non-zero where the card's "
            f"mask is dry, so the two masks are not one mask: {off_mask_max}")

    # The four stage-2 fields the split touches or must not touch.
    before_mom_w = stage_w(2, momentum=True, legacy=True)
    before_trc_w = stage_w(2, momentum=False, legacy=True)
    after_mom_w = stage_w(2, momentum=True)
    after_trc_w = stage_w(2, momentum=False)

    # LIVENESS.  With the split OFF the two consumers must read ONE field;
    # with it ON they must read two.  An inert split would otherwise report
    # "the tracer field did not move" while proving nothing.
    liveness = {
        "before_momentum_equals_tracer": _score_stage2_face(
            before_mom_w, before_trc_w, cell_mask),
        "after_momentum_versus_tracer": _score_stage2_face(
            after_mom_w, after_trc_w, cell_mask),
    }
    if plant == "wzv-split-inert":
        # Claim the split is on while leaving it off.  Being CAUGHT is the two
        # consumers reading one field; anything else is PLANT-BLIND, and a
        # blind control must not be able to print the fired marker.
        if liveness["before_momentum_equals_tracer"][
                "active_cells_unequal"] > 0:
            raise GateError(
                "PLANT-BLIND: the liveness control did not refuse an arm "
                "whose two consumers still read one field: "
                f"{liveness['before_momentum_equals_tracer']}")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": liveness["before_momentum_equals_tracer"]}
    require(liveness["before_momentum_equals_tracer"][
                "active_cells_unequal"] == 0,
            "the before arm's two consumers do not read one field, so it is "
            f"not the pre-split program: {liveness}")
    require(liveness["after_momentum_versus_tracer"][
                "active_cells_unequal"] > 0,
            "the split left the momentum and tracer fields equal, so it "
            f"never fired: {liveness}")

    # THE TRACER IDENTITY.  The oracle re-solves continuity for the tracers,
    # so the split must leave their field and their stage state untouched.
    tracer_identity = {
        "stage2_tracer_w": _score_stage2_face(
            after_trc_w, before_trc_w, cell_mask),
    }
    for field in ("T", "S"):
        # expose_tracer_stage substitutes T, S AND the stage eta, so those
        # three are its declared slots and every other field is the control.
        before_stage = np.asarray(getattr(
            run(before_plain, wrote=("T", "S", "eta"),
                label=f"stage2-tracer-{field}-before",
                expose_tracer_stage=2), field).data)
        after_stage = np.asarray(getattr(
            run(after_plain, wrote=("T", "S", "eta"),
                label=f"stage2-tracer-{field}-after",
                nemo_stage_momentum_wzv_split=True,
                expose_tracer_stage=2), field).data)
        tracer_identity[f"stage2_tracer_{field}"] = _score_stage2_face(
            after_stage, before_stage, cell_mask)
    if plant == "wzv-split-shared":
        # Claim the split is on while the SHARED field is the one that moved
        # (round 159's arm).  The tracer-identity control must refuse it.
        shared = stage_w(2, momentum=False, legacy=True,
                         base=before_plain,
                         stage2_wzv_velocity_form=True)
        caught = _score_stage2_face(shared, before_trc_w, cell_mask)
        if caught["active_cells_unequal"] == 0:
            raise GateError(
                "PLANT-BLIND: the tracer-identity control did not refuse an "
                f"arm that moved the shared field: {caught}")
        return {"status": "PLANT-FIRED", "plant": plant, "control": caught}
    require(all(row["cells_unequal"] == 0
                for row in tracer_identity.values()),
            "the split moved the tracer path, so it changed two things: "
            f"{tracer_identity}")

    active = cell_mask != 0.0
    own_rms = float(np.sqrt(np.mean(nemo_ww[active] ** 2)))
    scored = {
        "before_shared": _score_stage2_face(before_mom_w, nemo_ww, cell_mask),
        "after_momentum": _score_stage2_face(after_mom_w, nemo_ww, cell_mask),
        "after_tracer": _score_stage2_face(after_trc_w, nemo_ww, cell_mask),
    }
    base_rms = scored["before_shared"]["active_rms"]
    base_max = scored["before_shared"]["active_max_abs"]
    for row in scored.values():
        row["relative_to_own_rms"] = (
            float(row["active_rms"] / own_rms) if own_rms > 0.0 else 0.0)
        row["rms_removed_fraction"] = (
            float((base_rms - row["active_rms"]) / base_rms)
            if base_rms > 0.0 else 0.0)
        row["max_abs_removed_fraction"] = (
            float((base_max - row["active_max_abs"]) / base_max)
            if base_max > 0.0 else 0.0)

    # AUTHORITY.  The before arm must reproduce rounds 158 and 159 exactly, or
    # this walk is splitting a different number.
    ROUND159_PRODUCTION_WW = 1.2326857042024439e-08
    ROUND159_VELOCITY_FORM_WW = 2.334682468902387e-13
    ROUND158 = {
        "production": {"u": 3.844166e-12, "v": 6.428546e-12},
        "nemo_ww_ceiling": {"u": 1.121563e-14, "v": 1.308047e-14},
    }
    require(abs(scored["before_shared"]["active_rms"]
                - ROUND159_PRODUCTION_WW) <= 5.0e-22,
            "the before arm does not reproduce round 159's production "
            f"vertical velocity: {scored['before_shared']['active_rms']}")

    rhs = {}
    for name, frame in (("before_shared", stage2_rhs(legacy=True)),
                        ("after_momentum", stage2_rhs())):
        rhs[name] = {
            tag: _score_stage2_face(
                frame[index], rows[f"after_adv_{tag}"],
                rows[f"umask_vmask_{tag}"])
            for tag, index in (("u", 0), ("v", 1))
        }
    for tag in ("u", "v"):
        require(abs(rhs["before_shared"][tag]["active_rms"]
                    - ROUND158["production"][tag]) <= 5.0e-18,
                "the before arm's after-advection row moved from round 158 "
                f"on {tag}: {rhs['before_shared'][tag]['active_rms']}")
        rhs["after_momentum"][tag]["rms_removed_fraction"] = float(
            (ROUND158["production"][tag]
             - rhs["after_momentum"][tag]["active_rms"])
            / ROUND158["production"][tag])
        rhs["after_momentum"][tag]["times_the_nemo_ww_ceiling"] = float(
            rhs["after_momentum"][tag]["active_rms"]
            / ROUND158["nemo_ww_ceiling"][tag])

    # STAGE 3's ENTRY: the stage-2 output velocity, which is the field round
    # 155 attributed 95.2% of the developed stage-3 transport difference to
    # and round 157 measured at 1.30926020461275e-06 m/s.
    ROUND157_STAGE2_OUTPUT = {"u": 1.30926020461275e-06}
    stage2_output = {}
    for name, frame in (("before_shared", stage2_out(legacy=True)),
                        ("after_momentum", stage2_out())):
        stage2_output[name] = {
            tag: _score_stage2_face(
                frame[index], rows[f"uu_vv_Kaa_final_{tag}"],
                rows[f"umask_vmask_{tag}"])
            for tag, index in (("u", 0), ("v", 1))
        }
    for tag in ("u", "v"):
        base_out = stage2_output["before_shared"][tag]["active_rms"]
        stage2_output["after_momentum"][tag]["rms_removed_fraction"] = (
            float((base_out - stage2_output["after_momentum"][tag]
                   ["active_rms"]) / base_out) if base_out > 0.0 else 0.0)
    stage2_output["round157_reference"] = ROUND157_STAGE2_OUTPUT

    # STAGE 3: the split must fire there too (stprk3_stg.f90:358 skips only
    # stage 1).  There is no developed stage-3 vertical-velocity record, so
    # this row is the two fields against EACH OTHER, not against the oracle.
    stage3_split = _score_stage2_face(
        stage_w(3, momentum=True), stage_w(3, momentum=False), cell_mask)

    # THE TRACER CONSEQUENCE, for comparison with round 159's shared-field
    # move of 1.203492e-07 K.  The split changes the final one-step tracer
    # state only THROUGH the corrected stage-2 velocity.
    ROUND159_SHARED_TRACER_RMS = 1.2034924e-07
    tracer_step = {}
    for field in ("T", "S"):
        delta = (np.asarray(getattr(after_plain, field).data)[active]
                 - np.asarray(getattr(before_plain, field).data)[active])
        tracer_step[field] = {
            "cells_scored": int(delta.size),
            "cells_unequal": int(np.count_nonzero(delta != 0.0)),
            "one_step_rms": float(np.sqrt(np.mean(delta * delta))),
            "one_step_max_abs": float(np.max(np.abs(delta), initial=0.0)),
        }
    tracer_step["round159_shared_field_T_rms"] = ROUND159_SHARED_TRACER_RMS
    tracer_step["T"]["as_fraction_of_the_shared_field_move"] = float(
        tracer_step["T"]["one_step_rms"] / ROUND159_SHARED_TRACER_RMS)

    # THE DISCRIMINATOR for the residual round 159 registered.
    clock_w = stage_w(2, momentum=True, base=after_plain,
                      nemo_stage_momentum_wzv_split=True,
                      stage2_momentum_wzv_clock_pair=True)
    residual = {
        "after_momentum": scored["after_momentum"],
        "oracle_stage_clock_pair": _score_stage2_face(
            clock_w, nemo_ww, cell_mask),
    }
    residual["oracle_stage_clock_pair"]["relative_to_own_rms"] = float(
        residual["oracle_stage_clock_pair"]["active_rms"] / own_rms)
    residual["oracle_stage_clock_pair"]["residual_removed_fraction"] = float(
        (scored["after_momentum"]["active_rms"]
         - residual["oracle_stage_clock_pair"]["active_rms"])
        / scored["after_momentum"]["active_rms"])
    r3_result = run(after_plain, wrote=("u", "v"), label="stage2-face-r3",
                    nemo_stage_momentum_wzv_split=True,
                    expose_stage_face_r3=2)
    r3_rows = {}
    for tag, field in (("u", "u"), ("v", "v")):
        lego_r3 = np.asarray(getattr(r3_result, field).data)[..., 0]
        r3_rows[tag] = _score_stage2_face(
            lego_r3, np.asarray(rows[f"r3_Kmm_{tag}"]),
            np.asarray(rows[f"umask_vmask_{tag}"])[..., 0])
        own = float(np.sqrt(np.mean(
            np.asarray(rows[f"r3_Kmm_{tag}"]) ** 2)))
        r3_rows[tag]["relative_to_own_rms"] = (
            float(r3_rows[tag]["active_rms"] / own) if own > 0.0 else 0.0)
    residual["live_thickness_ratio_operand"] = r3_rows

    from legoesm.ocean.fidelity.ulp_move_gate import MAX_ULP_MOVE

    over = [row for row in observer_unequal
            if row["max_row_scale_ulps"] > MAX_ULP_MOVE]
    require(not over,
            "a passive exposure moved production state by more than "
            f"{MAX_ULP_MOVE} row-scale units in the last place: {over}")

    report = {
        "format": "gyre-round160-developed-stage2-wzv-split-walk-v1",
        "status": "PASS",
        "worktree": stamp,
        "record": {"sha256": oracle["sha256"], "meta": oracle["meta"]},
        "entry": {"step": DEVELOPED_ENTRY_STEP,
                  "process_step": DEVELOPED_PROCESS_STEP},
        "controls": {
            "entry_face_window": entry_control,
            "cell_window": cell_window,
            "nemo_ww_off_mask_max_abs": off_mask_max,
            "split_liveness": liveness,
            "tracer_identity": tracer_identity,
            "passive_observer_unequal": observer_unequal,
        },
        "nemo_ww_own_rms": own_rms,
        "vertical_velocity_scored_against_nemo": scored,
        "stage2_rhs_scored_against_nemo": rhs,
        "stage2_output_velocity_scored_against_nemo": stage2_output,
        "stage3_momentum_versus_tracer_field": stage3_split,
        "one_step_tracer_move_through_the_velocity": tracer_step,
        "residual_discriminator": residual,
        "round158_reference": ROUND158,
        "round159_reference": {
            "production_ww_rms": ROUND159_PRODUCTION_WW,
            "velocity_form_ww_rms": ROUND159_VELOCITY_FORM_WW,
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report


DEVELOPED_FACE_R3_PLANTS = ("face-r3-inert", "face-r3-tracer")


def developed_stage2_face_r3_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Does the live face free-surface ratio own the stage-2 residual?

    Round 160 built NEMO's second per-stage continuity solve
    (``stprk3_stg.f90:360``, velocity indicator at ``divhor.f90:126-130``) and
    took the developed stage-2 vertical velocity to ``2.334682468902387e-13``
    m/s, which is still ``9.74e-08`` of that field's own size.  It also
    measured that the face ratio the velocity indicator rebuilds each face
    transport from differs from the oracle's OWN recorded ``r3u(Kmm)`` on
    every active face, and registered that it had NOT shown the operand to own
    the residual.  This walk shows it, one variable.

    Two arms take the velocity indicator and both get the substitution: round
    160's CORRECTED arm, where the momentum vertical advection reads its own
    solve, and round 159's SHARED arm, where the single field is built in the
    velocity form.  The substitution is made by an observer that wraps the
    producer for exactly ONE call, the technique the tracer and transport
    observers in this instrument already use, so no production line changes
    and no card can reach it.

    The walk also decomposes the ratio difference itself with no model step at
    all: the oracle never rebuilds the stage ratio from the stage sea surface
    height (``stprk3_stg.f90:185`` calls ``dom_qco_r3c_RK3`` once per step and
    ``:211``/``:255`` COMBINE two ratios per stage), while legoESM builds one
    ratio from the already-combined height at ``vertical.py:247``.  The two are
    the same number algebraically, so the walk asks which side of that
    boundary carries the difference: the STATEMENT row is legoESM's form on the
    oracle's own recorded ``ssh(Kmm)``, the OPERAND row is legoESM's live
    ratio against that same reconstruction.
    """
    require(plant in (None, "none") + DEVELOPED_FACE_R3_PLANTS,
            f"unknown developed face-r3 plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe_module
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        nemo_stage_momentum_wzv_resolved)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import (
        nemo_qco_live_face_geometry_cgrid,
        nemo_qco_live_face_geometry_from_operands,
        nemo_qco_resolved_mesh_operands)
    from legoesm.ocean.eos import nemo_source_round

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"face-r3 walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "face-r3 walk commit differs from --expect-commit")
    evidence_root = Path(evidence_root)
    oracle = _developed_stage2_record(Path(stage2_record_root))
    rows = dict(oracle["rows"])

    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    payload = bundle["payload"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])
    require(nemo_stage_momentum_wzv_resolved(card.recipe.model_config),
            "the GYRE card does not resolve NEMO's two-solve stage program")

    nlev = int(np.asarray(state.T.data).shape[-1])
    nemo_ww = np.ascontiguousarray(np.asarray(rows["ww_t"])[..., :nlev])
    cell_mask = np.asarray(gate.expected_masks(card)["T"])[..., :nlev]
    face_mask = {tag: np.asarray(rows[f"umask_vmask_{tag}"])[..., 0]
                 for tag in ("u", "v")}
    oracle_r3 = {tag: np.asarray(rows[f"r3_Kmm_{tag}"])
                 for tag in ("u", "v")}

    # --- the operand the substitution installs, on NEMO's native extent ----
    # legoESM's redundant U face 0 is the periodic wrap of the last native
    # face and its V row 0 is the closed south wall, which is the map
    # ``nemo_qco_live_face_geometry_cgrid`` writes once; invert it here rather
    # than index the producer's own arrays a second way.
    native_r3 = {"u": oracle_r3["u"][:, 1:], "v": oracle_r3["v"][1:, :]}

    real_geometry = pe_module.nemo_qco_live_face_geometry_from_operands
    real_wzv = model_module.nemo_qco_wzv_operands
    geometry_calls = {"count": 0}

    def substituted_geometry(eta, e3u_0, e3v_0, umask3, vmask3, hu_0, hv_0,
                             area_t, area_u, area_v):
        # domzgr_substitute.h90:127 / vertical.py:251 -- NEMO's OWN thickness
        # statement with the oracle's ratio in place of legoESM's, and nothing
        # else touched.  The reciprocals this helper also returns are not
        # consumed by the velocity indicator (the caller takes ``[:2]``), so
        # they keep legoESM's values and are reported as such.
        geometry_calls["count"] += 1
        geom = real_geometry(eta, e3u_0, e3v_0, umask3, vmask3, hu_0, hv_0,
                             area_t, area_u, area_v)
        b = nemo_source_round
        one = jnp.asarray(1.0, dtype=geom.e3u.dtype)
        r3u = jnp.asarray(substitute_operand["u"], dtype=geom.e3u.dtype)
        r3v = jnp.asarray(substitute_operand["v"], dtype=geom.e3v.dtype)
        e3u = b(jnp.asarray(e3u_0, dtype=geom.e3u.dtype)
                * b(one + r3u[..., None] * jnp.asarray(umask3)))
        e3v = b(jnp.asarray(e3v_0, dtype=geom.e3v.dtype)
                * b(one + r3v[..., None] * jnp.asarray(vmask3)))
        return geom._replace(e3u=e3u, e3v=e3v, r3u=r3u, r3v=r3v)

    substitute_operand = native_r3
    ledgers: dict[str, list] = {}

    def run(label, *, substitute_ordinal=None, observe=True, **hook_kwargs):
        """One production step, optionally with the producer observer on."""
        ledger: list[dict] = []

        def capture(*args, **kwargs):
            velocity_indicator = (
                kwargs.get("volume_transport_override") is None
                and kwargs.get("transport_after_override") is None)
            index = sum(row["velocity_indicator"] for row in ledger)
            row = {"call": len(ledger),
                   "velocity_indicator": bool(velocity_indicator),
                   "velocity_ordinal": index if velocity_indicator else None,
                   "substituted": False}
            ledger.append(row)
            if (substitute_ordinal is not None and velocity_indicator
                    and index == substitute_ordinal):
                row["substituted"] = True
                before = geometry_calls["count"]
                pe_module.nemo_qco_live_face_geometry_from_operands = (
                    substituted_geometry)
                try:
                    result = real_wzv(*args, **kwargs)
                finally:
                    pe_module.nemo_qco_live_face_geometry_from_operands = (
                        real_geometry)
                require(geometry_calls["count"] == before + 1,
                        "the substituted producer ran "
                        f"{geometry_calls['count'] - before} times in one "
                        "velocity-indicator call, not once")
                return result
            return real_wzv(*args, **kwargs)

        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_kwargs)
            if hook_kwargs else None)
        if observe:
            model_module.nemo_qco_wzv_operands = capture
        try:
            result = model.step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface,
                _nemo_stage1_zad_eta_after_override=ssha)
            jax.device_get(result)
        finally:
            model_module.nemo_qco_wzv_operands = real_wzv
        if observe:
            ledgers[label] = ledger
        return result

    def ledger_summary(label):
        ledger = ledgers[label]
        return {
            "calls": len(ledger),
            "velocity_indicator_calls": sum(
                row["velocity_indicator"] for row in ledger),
            "substituted_calls": sum(row["substituted"] for row in ledger),
            "pattern": [row["velocity_indicator"] for row in ledger],
        }

    def stage2_momentum_w(result_hooks, *, substitute_ordinal=None,
                          label=""):
        result = run(label, substitute_ordinal=substitute_ordinal,
                     expose_tracer_transport_stage=2,
                     expose_tracer_transport_as_ww=True,
                     **result_hooks)
        return np.asarray(result.T.data)

    CORRECTED = {"nemo_stage_momentum_wzv_split": True,
                 "expose_stage_momentum_w": True}
    SHARED = {"stage2_wzv_velocity_form": True}

    # ---- OBSERVER PASSIVITY: the wrapper alone must move ZERO bytes -------
    plain = run("plain-unobserved", observe=False,
                nemo_stage_momentum_wzv_split=True)
    observed = run("plain-observed", nemo_stage_momentum_wzv_split=True)
    observer_passivity = _state_leaf_move(observed, plain)
    require(observer_passivity["unequal_bytes"] == 0,
            "the producer observer is not passive: "
            f"{observer_passivity}")

    # ---- THE FOUR SCORED ARMS --------------------------------------------
    arms = {}
    arms["corrected_baseline"] = stage2_momentum_w(
        CORRECTED, label="corrected-baseline")
    arms["corrected_oracle_r3"] = stage2_momentum_w(
        CORRECTED, substitute_ordinal=0, label="corrected-oracle-r3")
    arms["shared_velocity_form_baseline"] = stage2_momentum_w(
        SHARED, label="shared-baseline")
    arms["shared_velocity_form_oracle_r3"] = stage2_momentum_w(
        SHARED, substitute_ordinal=0, label="shared-oracle-r3")

    for label in ("corrected-baseline", "shared-baseline"):
        require(ledger_summary(label)["substituted_calls"] == 0,
                f"{label} substituted a call it should not have")
    for label in ("corrected-oracle-r3", "shared-oracle-r3"):
        require(ledger_summary(label)["substituted_calls"] == 1,
                f"{label} did not substitute exactly one call: "
                f"{ledger_summary(label)}")

    own_rms = float(np.sqrt(np.mean(nemo_ww[cell_mask != 0.0] ** 2)))
    scored = {name: _score_stage2_face(values, nemo_ww, cell_mask)
              for name, values in arms.items()}
    for row in scored.values():
        row["relative_to_own_rms"] = float(row["active_rms"] / own_rms)

    # AUTHORITY: both baselines must reproduce round 159/160's own number, or
    # this walk is dividing a different residual.
    ROUND160_CORRECTED_WW = 2.334682468902387e-13
    for name in ("corrected_baseline", "shared_velocity_form_baseline"):
        require(abs(scored[name]["active_rms"] - ROUND160_CORRECTED_WW)
                <= 5.0e-22,
                f"{name} does not reproduce the round-160 residual: "
                f"{scored[name]['active_rms']}")

    base = scored["corrected_baseline"]["active_rms"]
    for arm, name in (("corrected_oracle_r3", "corrected_baseline"),
                      ("shared_velocity_form_oracle_r3",
                       "shared_velocity_form_baseline")):
        reference = scored[name]["active_rms"]
        scored[arm]["residual_removed_fraction"] = float(
            (reference - scored[arm]["active_rms"]) / reference)

    # LIVENESS: a substitution that moves nothing is a control that perturbs
    # a zero and proves nothing.
    liveness = {
        "corrected": _score_stage2_face(
            arms["corrected_oracle_r3"], arms["corrected_baseline"],
            cell_mask),
        "shared": _score_stage2_face(
            arms["shared_velocity_form_oracle_r3"],
            arms["shared_velocity_form_baseline"], cell_mask),
    }
    if plant == "face-r3-inert":
        # Claim the oracle's ratio is installed while handing the producer
        # legoESM's OWN live ratio: the liveness control must refuse it.
        live = run("plant-live-r3", expose_stage_face_r3=2,
                   nemo_stage_momentum_wzv_split=True)
        substitute_operand = {
            "u": np.asarray(live.u.data)[..., 0][:, 1:],
            "v": np.asarray(live.v.data)[..., 0][1:, :],
        }
        planted = stage2_momentum_w(
            CORRECTED, substitute_ordinal=0, label="plant-inert")
        caught = _score_stage2_face(
            planted, arms["corrected_baseline"], cell_mask)
        if caught["active_cells_unequal"] != 0:
            raise GateError(
                "PLANT-BLIND: the liveness control did not refuse a "
                f"substitution that installs legoESM's own ratio: {caught}")
        return {"status": "PLANT-FIRED", "plant": plant, "control": caught}
    require(all(row["active_cells_unequal"] > 0 for row in liveness.values()),
            f"the substitution moved nothing, so it proves nothing: {liveness}")

    # TRACER IDENTITY: the momentum substitution must not reach the tracers.
    tracer_identity = {
        "stage2_tracer_w": _score_stage2_face(
            np.asarray(run("tracer-w-substituted", substitute_ordinal=0,
                           nemo_stage_momentum_wzv_split=True,
                           expose_tracer_transport_stage=2,
                           expose_tracer_transport_as_ww=True).T.data),
            np.asarray(run("tracer-w-baseline",
                           nemo_stage_momentum_wzv_split=True,
                           expose_tracer_transport_stage=2,
                           expose_tracer_transport_as_ww=True).T.data),
            cell_mask),
    }
    if plant == "face-r3-tracer":
        # Claim the substitution is momentum-only while wiring it into the
        # SHARED producer that the tracers also read (the split off): the
        # tracer-identity control must refuse it.
        planted = np.asarray(run(
            "plant-tracer", substitute_ordinal=0,
            stage2_wzv_velocity_form=True,
            expose_tracer_transport_stage=2,
            expose_tracer_transport_as_ww=True).T.data)
        against = np.asarray(run(
            "plant-tracer-base", stage2_wzv_velocity_form=True,
            expose_tracer_transport_stage=2,
            expose_tracer_transport_as_ww=True).T.data)
        caught = _score_stage2_face(planted, against, cell_mask)
        if caught["active_cells_unequal"] == 0:
            raise GateError(
                "PLANT-BLIND: the tracer-identity control did not refuse a "
                f"substitution that reached the tracer field: {caught}")
        return {"status": "PLANT-FIRED", "plant": plant, "control": caught}
    require(tracer_identity["stage2_tracer_w"]["cells_unequal"] == 0,
            "the momentum substitution moved the tracer path: "
            f"{tracer_identity}")

    # ---- THE PRODUCER WALK, offline from the record alone ----------------
    live = run("live-face-r3", expose_stage_face_r3=2,
               nemo_stage_momentum_wzv_split=True)
    live_r3 = {"u": np.asarray(live.u.data)[..., 0],
               "v": np.asarray(live.v.data)[..., 0]}
    def offline_ratio(mesh_ops):
        # The RATIO itself, not ``1 + ratio`` minus one.  The redundant-face
        # helper returns the stretching factor, and recovering the ratio from
        # it costs half a unit in the last place of 1.0 -- about 1.1e-16,
        # which is three orders ABOVE the ratio's own last place and would be
        # read as a statement difference.  So the native helper is called and
        # the same wrap/wall map the redundant one writes is applied here.
        native = nemo_qco_live_face_geometry_from_operands(
            jnp.asarray(rows["ssh_Kmm_t"]), mesh_ops.e3u_0, mesh_ops.e3v_0,
            mesh_ops.umask3, mesh_ops.vmask3, mesh_ops.hu_0, mesh_ops.hv_0,
            mesh_ops.area_t, mesh_ops.area_u, mesh_ops.area_v)
        ratio_u = np.asarray(native.r3u)
        ratio_v = np.asarray(native.r3v)
        return {
            "u": np.concatenate([ratio_u[:, -1:], ratio_u], axis=1),
            "v": np.concatenate([np.zeros_like(ratio_v[:1]), ratio_v], axis=0),
        }

    ops = nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid,
        jnp.asarray(rows["umask_vmask_u"]), jnp.asarray(rows["umask_vmask_v"]),
        jnp.float64, nlev)
    offline_r3 = offline_ratio(ops)
    # MESH-SOURCE CONTROL.  legoESM's LIVE stage ratio is built by the stage
    # helper from the CARD's rebuilt mesh operands, while the continuity
    # producer resolves NEMO's own raw ones.  If those two operand sets
    # disagreed, the live-versus-statement row below would be measuring the
    # mesh rather than the sea surface height, so both are evaluated on the
    # SAME oracle ssh and scored against each other.
    from legoesm.ocean.vertical import nemo_qco_card_mesh_operands

    h_ref = jnp.asarray(card.recipe.z_coord.h_partial, dtype=jnp.float64)
    if h_ref.ndim == 1:
        h_ref = jnp.broadcast_to(
            h_ref, (*np.asarray(state.T.data).shape[:2], h_ref.shape[-1]))
    card_ops = nemo_qco_card_mesh_operands(
        h_ref[..., :nlev], jnp.asarray(rows["umask_vmask_u"]),
        jnp.asarray(rows["umask_vmask_v"]), card.recipe.grid, jnp.float64)
    card_r3 = offline_ratio(card_ops)
    mesh_source = {
        "card_carries_raw_nemo_mesh": bool(
            getattr(card.recipe.z_coord, "nemo_e1e2t", None) is not None),
        "u": _score_stage2_face(card_r3["u"], offline_r3["u"], face_mask["u"]),
        "v": _score_stage2_face(card_r3["v"], offline_r3["v"], face_mask["v"]),
    }
    # THE INSTRUMENT'S OWN FLOOR.  legoESM's LIVE stage ratio can only be read
    # out of the model as the stretching factor ``1 + ratio``, so the exposure
    # recovers it by subtracting one and carries that quantisation.  The
    # offline rows above do NOT, and the rows that use the live ratio are
    # reported next to this bound rather than below it.
    _, _, round_trip_u, round_trip_v = nemo_qco_live_face_geometry_cgrid(
        jnp.asarray(rows["ssh_Kmm_t"]), ops.e3u_0, ops.e3v_0, ops.umask3,
        ops.vmask3, ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v)
    live_readout_floor = {
        "u": _score_stage2_face(
            np.asarray(round_trip_u) - 1.0, offline_r3["u"], face_mask["u"]),
        "v": _score_stage2_face(
            np.asarray(round_trip_v) - 1.0, offline_r3["v"], face_mask["v"]),
        "half_ulp_of_one": float(np.spacing(1.0) / 2.0),
    }
    producer = {}
    for tag in ("u", "v"):
        own = float(np.sqrt(np.mean(oracle_r3[tag] ** 2)))
        rows_for_tag = {
            "live_versus_oracle": _score_stage2_face(
                live_r3[tag], oracle_r3[tag], face_mask[tag]),
            "statement_given_oracle_ssh": _score_stage2_face(
                offline_r3[tag], oracle_r3[tag], face_mask[tag]),
            "live_versus_statement": _score_stage2_face(
                live_r3[tag], offline_r3[tag], face_mask[tag]),
        }
        for row in rows_for_tag.values():
            row["relative_to_own_rms"] = float(row["active_rms"] / own)
        producer[tag] = rows_for_tag
        producer[tag]["operand_own_rms"] = own
        producer[tag]["statement_share_of_live"] = float(
            rows_for_tag["statement_given_oracle_ssh"]["active_rms"]
            / rows_for_tag["live_versus_oracle"]["active_rms"])
        producer[tag]["operand_share_of_live"] = float(
            rows_for_tag["live_versus_statement"]["active_rms"]
            / rows_for_tag["live_versus_oracle"]["active_rms"])

    # ---- THE UNRELATED-ARM PASSIVITY CONTROL round 160's review asked for -
    # Round 160 loosened a zero-byte passivity requirement to two units in the
    # last place at row scale.  These two exposure arms are on code this round
    # did not write, and they answer whether that floor is general.
    # The baseline is the ORDINARY production step -- no split, no observer --
    # because these arms carry no split either; scoring them against the
    # split arm would measure round 160's physics, not an exposure.
    ordinary = run("ordinary-production", observe=False)
    unrelated = {}
    for label, wrote, hooks in (
            ("expose_momentum_stage=1", ("u", "v"),
             {"expose_momentum_stage": 1}),
            ("expose_tracer_stage=3", ("T", "S", "eta"),
             {"expose_tracer_stage": 3})):
        exposed = run(f"unrelated-{label}", observe=False, **hooks)
        neutral = exposed._replace(
            **{name: getattr(ordinary, name) for name in wrote})
        unrelated[label] = _state_leaf_move(neutral, ordinary)

    report = {
        "format": "gyre-round161-developed-stage2-face-r3-walk-v1",
        "status": "PASS",
        "worktree": stamp,
        "record": {"sha256": oracle["sha256"], "meta": oracle["meta"]},
        "entry": {"step": DEVELOPED_ENTRY_STEP,
                  "process_step": DEVELOPED_PROCESS_STEP},
        "controls": {
            "observer_passivity": observer_passivity,
            "call_ledgers": {label: ledger_summary(label)
                             for label in sorted(ledgers)},
            "substitution_liveness": liveness,
            "tracer_identity": tracer_identity,
            "unrelated_exposure_passivity": unrelated,
        },
        "nemo_ww_own_rms": own_rms,
        "vertical_velocity_scored_against_nemo": scored,
        "producer_decomposition": producer,
        "producer_mesh_source_control": mesh_source,
        "live_ratio_readout_floor": live_readout_floor,
        "round160_reference": {
            "corrected_ww_rms": ROUND160_CORRECTED_WW,
            "shared_ww_rms": 1.2326857042024439e-08,
            "r3u_operand_rms": 1.4516405645036015e-13,
            "r3v_operand_rms": 1.3880628732222712e-13,
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report


DEVELOPED_T_R3_PLANTS = ("t-r3-inert", "t-r3-tracer")


def developed_stage2_t_r3_walk(
        daily_root: Path, daily_audit: Path, expected_commit: str,
        stage2_record_root: Path, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Does the T-point free-surface ratio own the stage-2 residual?

    Rounds 159, 160 and 161 substituted three operands of the velocity-form
    continuity solve from the oracle's own record -- the stage entry velocity,
    the stage clock and after-level pair, and the face free-surface ratio --
    and every one of them is inert, leaving the residual UNOWNED at
    ``2.334682468902387e-13`` m/s.  The T-point ratio is the one operand of
    that solve nothing has installed.

    It enters the compiled program TWICE, both inside one statement pair: the
    whole horizontal divergence is divided by the live thickness at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130`` and the
    same live thickness multiplies the divergence back at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153``.  legoESM
    forms the ratio at ``ocean_pe_latlon_cgrid.py:1674``, the thickness at
    ``ocean_pe_latlon_cgrid.py:1675`` and hands it to the divergence block at
    ``ocean_pe_latlon_cgrid.py:1699``, so replacing that ONE operand replaces
    both compiled occurrences and nothing else.

    The stretching term at
    ``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298`` reads the
    ratio at the AFTER and BEFORE levels, not the now level -- legoESM's
    answering pair is ``ocean_pe_latlon_cgrid.py:1717-1718`` -- so it is not
    part of this substitution, and the walk says so rather than leaving it
    implied.

    The record carries the stage sea surface height, not the T-point ratio, so
    the operand installed is NEMO's own ratio statement
    (``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:257``) evaluated on
    the oracle's recorded ``ssh(Kmm)``.  The walk calibrates that
    reconstruction first: fed legoESM's OWN height it must reproduce the
    production step bit for bit, which is what makes it the same statement
    rather than a second implementation of it.

    The same observer answers round 161's other open item by sinking the
    height legoESM hands the producer and scoring it against the oracle's
    recorded one.
    """
    require(plant in (None, "none") + DEVELOPED_T_R3_PLANTS,
            f"unknown developed t-r3 plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe_module
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        nemo_stage_momentum_wzv_resolved)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"t-r3 walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "t-r3 walk commit differs from --expect-commit")
    evidence_root = Path(evidence_root)
    oracle = _developed_stage2_record(Path(stage2_record_root))
    rows = dict(oracle["rows"])

    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    payload = bundle["payload"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])
    require(nemo_stage_momentum_wzv_resolved(card.recipe.model_config),
            "the GYRE card does not resolve NEMO's two-solve stage program")

    nlev = int(np.asarray(state.T.data).shape[-1])
    nemo_ww = np.ascontiguousarray(np.asarray(rows["ww_t"])[..., :nlev])
    cell_mask = np.asarray(gate.expected_masks(card)["T"])[..., :nlev]
    cell_mask2 = np.any(cell_mask, axis=-1)
    oracle_ssh = np.ascontiguousarray(np.asarray(rows["ssh_Kmm_t"]))

    real_level = pe_module.nemo_transport_wzv_divergence_level
    real_wzv = model_module.nemo_qco_wzv_operands
    level_calls = {"count": 0}
    # "oracle" installs NEMO's recorded height in the ratio statement;
    # "identity" installs legoESM's OWN height, which must reproduce the
    # production step bit for bit and is what calibrates the transcription.
    height_source = {"mode": "oracle"}

    def substituted_thickness(args):
        """NEMO's live e3t from ONE substituted operand, nothing else moved.

        ``ocean_pe_latlon_cgrid.py:1668-1675`` in the producer's own operands:
        the reference thickness column sum, its reciprocal, the ratio, and the
        thickness.  Only the HEIGHT fed to the ratio changes.
        """
        eta_now, _eta_before, u, _v, grid, z_coord = args[:6]
        u_mask_3d, v_mask_3d, mask_3d = args[6], args[7], args[8]
        levels = u.shape[-1]
        ops = nemo_qco_resolved_mesh_operands(
            z_coord, grid, u_mask_3d, v_mask_3d, eta_now.dtype, levels)
        e3t0 = ops.e3t_0
        tmask = jnp.asarray(mask_3d, dtype=eta_now.dtype)
        h0 = jnp.zeros_like(eta_now)
        for jk in range(levels):
            h0 = jax.lax.optimization_barrier(
                h0 + e3t0[..., jk] * tmask[..., jk])
        h0_safe = jnp.where(h0 > 0.0, h0, 1.0)
        r1_h0 = jax.lax.optimization_barrier(1.0 / h0_safe)
        if height_source["mode"] == "identity":
            height = eta_now
        else:
            height = jnp.asarray(oracle_ssh, dtype=eta_now.dtype)
        r3 = jax.lax.optimization_barrier(height * r1_h0)
        return e3t0 * (1.0 + r3[..., None] * tmask) * tmask

    ledgers: dict[str, list] = {}
    sunk: dict[str, list] = {}

    def run(label, *, substitute_ordinal=None, sink_ordinal=None,
            observe=True, **hook_kwargs):
        """One production step, optionally with the producer observer on."""
        ledger: list[dict] = []
        heights: list[np.ndarray] = []

        def capture(*args, **kwargs):
            velocity_indicator = (
                kwargs.get("volume_transport_override") is None
                and kwargs.get("transport_after_override") is None)
            index = sum(row["velocity_indicator"] for row in ledger)
            row = {"call": len(ledger),
                   "velocity_indicator": bool(velocity_indicator),
                   "velocity_ordinal": index if velocity_indicator else None,
                   "substituted": False, "sunk": False}
            ledger.append(row)
            if (sink_ordinal is not None and velocity_indicator
                    and index == sink_ordinal):
                row["sunk"] = True
                jax.debug.callback(
                    lambda value: heights.append(np.asarray(value)),
                    args[0], ordered=True)
            if (substitute_ordinal is not None and velocity_indicator
                    and index == substitute_ordinal):
                row["substituted"] = True
                thickness = substituted_thickness(args)
                before = level_calls["count"]
                level_index = {"jk": 0}

                def substituted_level(flux_u, flux_u_west, flux_v,
                                      flux_v_south, r1_area_t, live_e3t,
                                      tmask, **level_kwargs):
                    jk = level_index["jk"]
                    level_index["jk"] += 1
                    level_calls["count"] += 1
                    # The level index is PER CALL, and it is checked.  JAX
                    # CLAMPS an out-of-range integer index instead of raising,
                    # so a counter carried across calls silently hands every
                    # level the deepest thickness and returns a plausible
                    # number.  That is what the first version of this walk did
                    # from its second substituted arm onward, and the walk's
                    # own inert plant is what caught it.
                    require(0 <= jk < thickness.shape[-1],
                            f"substituted level index {jk} is outside the "
                            f"{thickness.shape[-1]} levels of the thickness")
                    return real_level(
                        flux_u, flux_u_west, flux_v, flux_v_south, r1_area_t,
                        thickness[..., jk], tmask, **level_kwargs)

                pe_module.nemo_transport_wzv_divergence_level = (
                    substituted_level)
                try:
                    result = real_wzv(*args, **kwargs)
                finally:
                    pe_module.nemo_transport_wzv_divergence_level = real_level
                require(level_calls["count"] == before + nlev,
                        "the substituted divergence block ran "
                        f"{level_calls['count'] - before} levels in one "
                        f"velocity-indicator call, not {nlev}")
                return result
            return real_wzv(*args, **kwargs)

        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_kwargs)
            if hook_kwargs else None)
        if observe:
            model_module.nemo_qco_wzv_operands = capture
        try:
            result = model.step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface,
                _nemo_stage1_zad_eta_after_override=ssha)
            jax.device_get(result)
            jax.effects_barrier()
        finally:
            model_module.nemo_qco_wzv_operands = real_wzv
        if observe:
            ledgers[label] = ledger
            if sink_ordinal is not None:
                sunk[label] = heights
        return result

    def ledger_summary(label):
        ledger = ledgers[label]
        return {
            "calls": len(ledger),
            "velocity_indicator_calls": sum(
                row["velocity_indicator"] for row in ledger),
            "substituted_calls": sum(row["substituted"] for row in ledger),
            "sunk_calls": sum(row["sunk"] for row in ledger),
            "pattern": [row["velocity_indicator"] for row in ledger],
        }

    def stage2_momentum_w(result_hooks, *, substitute_ordinal=None, label=""):
        result = run(label, substitute_ordinal=substitute_ordinal,
                     expose_tracer_transport_stage=2,
                     expose_tracer_transport_as_ww=True,
                     **result_hooks)
        return np.asarray(result.T.data)

    CORRECTED = {"nemo_stage_momentum_wzv_split": True,
                 "expose_stage_momentum_w": True}
    SHARED = {"stage2_wzv_velocity_form": True}

    # ---- OBSERVER PASSIVITY: the wrapper alone must move ZERO bytes -------
    plain = run("plain-unobserved", observe=False,
                nemo_stage_momentum_wzv_split=True)
    observed = run("plain-observed", nemo_stage_momentum_wzv_split=True)
    observer_passivity = _state_leaf_move(observed, plain)
    require(observer_passivity["unequal_bytes"] == 0,
            f"the producer observer is not passive: {observer_passivity}")

    # ---- ORDER C: the stage height legoESM hands the producer -------------
    sunk_state = run("eta-sink", sink_ordinal=0,
                     nemo_stage_momentum_wzv_split=True)
    sink_passivity = _state_leaf_move(sunk_state, plain)
    require(sink_passivity["unequal_bytes"] == 0,
            f"the height sink is not passive: {sink_passivity}")
    heights = sunk[ "eta-sink"]
    require(len(heights) == 1,
            f"the height sink fired {len(heights)} times, not once")
    lego_ssh = np.ascontiguousarray(
        np.asarray(heights[0], dtype=np.float64))
    stage_height = {
        "scored": _score_stage2_face(lego_ssh, oracle_ssh, cell_mask2),
        "oracle_own_rms": float(
            np.sqrt(np.mean(oracle_ssh[cell_mask2 != 0.0] ** 2))),
    }
    stage_height["relative_to_own_rms"] = float(
        stage_height["scored"]["active_rms"]
        / stage_height["oracle_own_rms"])
    # Round 161 predicted this height difference from the face ratio it
    # measured, dividing by the reciprocal reference depth.  Close the loop in
    # the other direction: the height difference divided by the reference
    # depth must reproduce that face-ratio difference.
    ops_check = nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid,
        jnp.asarray(rows["umask_vmask_u"]), jnp.asarray(rows["umask_vmask_v"]),
        jnp.float64, nlev)
    tmask_check = np.asarray(
        np.any(np.asarray(ops_check.e3t_0) != 0.0, axis=-1), dtype=np.float64)
    depth = np.sum(
        np.asarray(ops_check.e3t_0)
        * np.asarray(cell_mask, dtype=np.float64), axis=-1)
    safe_depth = np.where(depth > 0.0, depth, 1.0)
    implied = (lego_ssh - oracle_ssh) / safe_depth
    stage_height["implied_t_ratio_difference_rms"] = float(
        np.sqrt(np.mean(implied[cell_mask2 != 0.0] ** 2)))
    stage_height["round161_face_ratio_difference_rms"] = {
        "u": 1.4516405645036015e-13, "v": 1.3880628732222712e-13}
    stage_height["implied_over_measured_face_u"] = float(
        stage_height["implied_t_ratio_difference_rms"]
        / 1.4516405645036015e-13)
    stage_height["tmask_columns"] = int(np.count_nonzero(tmask_check))

    # ---- THE CALIBRATION: the same statement, not a second one -----------
    arms = {}
    arms["corrected_baseline"] = stage2_momentum_w(
        CORRECTED, label="corrected-baseline")
    height_source["mode"] = "identity"
    arms["corrected_identity"] = stage2_momentum_w(
        CORRECTED, substitute_ordinal=0, label="corrected-identity")
    height_source["mode"] = "oracle"
    calibration = _score_stage2_face(
        arms["corrected_identity"], arms["corrected_baseline"], cell_mask)
    require(calibration["cells_unequal"] == 0,
            "the substituted ratio statement fed legoESM's own height does "
            f"not reproduce the production step: {calibration}")

    # ---- THE FOUR SCORED ARMS --------------------------------------------
    arms["corrected_oracle_r3t"] = stage2_momentum_w(
        CORRECTED, substitute_ordinal=0, label="corrected-oracle-r3t")
    arms["shared_velocity_form_baseline"] = stage2_momentum_w(
        SHARED, label="shared-baseline")
    arms["shared_velocity_form_oracle_r3t"] = stage2_momentum_w(
        SHARED, substitute_ordinal=0, label="shared-oracle-r3t")

    # THE CALIBRATION AGAIN, AFTER EVERY OTHER SUBSTITUTED ARM.  The first
    # version of this walk was bit-exact on its FIRST substituted arm and
    # wrong on every one after it, because the per-level index was carried
    # across calls.  A calibration that runs only first cannot see that, so
    # the identity arm is repeated last and must still be bit-exact.
    height_source["mode"] = "identity"
    arms["corrected_identity_repeat"] = stage2_momentum_w(
        CORRECTED, substitute_ordinal=0, label="corrected-identity-repeat")
    height_source["mode"] = "oracle"
    calibration_repeat = _score_stage2_face(
        arms["corrected_identity_repeat"], arms["corrected_baseline"],
        cell_mask)
    require(calibration_repeat["cells_unequal"] == 0,
            "the substituted ratio statement stopped reproducing the "
            f"production step after other arms ran: {calibration_repeat}")

    for label in ("corrected-baseline", "shared-baseline"):
        require(ledger_summary(label)["substituted_calls"] == 0,
                f"{label} substituted a call it should not have")
    for label in ("corrected-identity", "corrected-identity-repeat",
                  "corrected-oracle-r3t", "shared-oracle-r3t"):
        require(ledger_summary(label)["substituted_calls"] == 1,
                f"{label} did not substitute exactly one call: "
                f"{ledger_summary(label)}")

    own_rms = float(np.sqrt(np.mean(nemo_ww[cell_mask != 0.0] ** 2)))
    scored = {name: _score_stage2_face(values, nemo_ww, cell_mask)
              for name, values in arms.items()}
    for row in scored.values():
        row["relative_to_own_rms"] = float(row["active_rms"] / own_rms)

    # AUTHORITY: both baselines must reproduce round 159/160's own number, or
    # this walk is dividing a different residual.
    ROUND160_CORRECTED_WW = 2.334682468902387e-13
    for name in ("corrected_baseline", "shared_velocity_form_baseline"):
        require(abs(scored[name]["active_rms"] - ROUND160_CORRECTED_WW)
                <= 5.0e-22,
                f"{name} does not reproduce the round-160 residual: "
                f"{scored[name]['active_rms']}")

    for arm, name in (("corrected_oracle_r3t", "corrected_baseline"),
                      ("shared_velocity_form_oracle_r3t",
                       "shared_velocity_form_baseline")):
        reference = scored[name]["active_rms"]
        scored[arm]["residual_removed_fraction"] = float(
            (reference - scored[arm]["active_rms"]) / reference)

    # LIVENESS: a substitution that moves nothing is a control that perturbs
    # a zero and proves nothing.
    liveness = {
        "corrected": _score_stage2_face(
            arms["corrected_oracle_r3t"], arms["corrected_baseline"],
            cell_mask),
        "shared": _score_stage2_face(
            arms["shared_velocity_form_oracle_r3t"],
            arms["shared_velocity_form_baseline"], cell_mask),
    }
    if plant == "t-r3-inert":
        # Claim the oracle's height is installed while handing the ratio
        # statement legoESM's OWN height: the liveness control must refuse it.
        height_source["mode"] = "identity"
        planted = stage2_momentum_w(
            CORRECTED, substitute_ordinal=0, label="plant-inert")
        height_source["mode"] = "oracle"
        caught = _score_stage2_face(
            planted, arms["corrected_baseline"], cell_mask)
        if caught["active_cells_unequal"] != 0:
            raise GateError(
                "PLANT-BLIND: the liveness control did not refuse a "
                f"substitution that installs legoESM's own height: {caught}")
        return {"status": "PLANT-FIRED", "plant": plant, "control": caught}
    require(all(row["active_cells_unequal"] > 0 for row in liveness.values()),
            f"the substitution moved nothing, so it proves nothing: {liveness}")

    # TRACER IDENTITY: the momentum substitution must not reach the tracers.
    tracer_identity = {
        "stage2_tracer_w": _score_stage2_face(
            np.asarray(run("tracer-w-substituted", substitute_ordinal=0,
                           nemo_stage_momentum_wzv_split=True,
                           expose_tracer_transport_stage=2,
                           expose_tracer_transport_as_ww=True).T.data),
            np.asarray(run("tracer-w-baseline",
                           nemo_stage_momentum_wzv_split=True,
                           expose_tracer_transport_stage=2,
                           expose_tracer_transport_as_ww=True).T.data),
            cell_mask),
    }
    if plant == "t-r3-tracer":
        # Claim the substitution is momentum-only while wiring it into the
        # SHARED producer that the tracers also read (the split off): the
        # tracer-identity control must refuse it.
        planted = np.asarray(run(
            "plant-tracer", substitute_ordinal=0,
            stage2_wzv_velocity_form=True,
            expose_tracer_transport_stage=2,
            expose_tracer_transport_as_ww=True).T.data)
        against = np.asarray(run(
            "plant-tracer-base", stage2_wzv_velocity_form=True,
            expose_tracer_transport_stage=2,
            expose_tracer_transport_as_ww=True).T.data)
        caught = _score_stage2_face(planted, against, cell_mask)
        if caught["active_cells_unequal"] == 0:
            raise GateError(
                "PLANT-BLIND: the tracer-identity control did not refuse a "
                f"substitution that reached the tracer field: {caught}")
        return {"status": "PLANT-FIRED", "plant": plant, "control": caught}
    require(tracer_identity["stage2_tracer_w"]["cells_unequal"] == 0,
            "the momentum substitution moved the tracer path: "
            f"{tracer_identity}")

    report = {
        "format": "gyre-round162-developed-stage2-t-r3-walk-v1",
        "status": "PASS",
        "worktree": stamp,
        "record": {"sha256": oracle["sha256"], "meta": oracle["meta"]},
        "entry": {"step": DEVELOPED_ENTRY_STEP,
                  "process_step": DEVELOPED_PROCESS_STEP},
        "controls": {
            "observer_passivity": observer_passivity,
            "height_sink_passivity": sink_passivity,
            "statement_calibration": calibration,
            "statement_calibration_repeated_last": calibration_repeat,
            "call_ledgers": {label: ledger_summary(label)
                             for label in sorted(ledgers)},
            "substitution_liveness": liveness,
            "tracer_identity": tracer_identity,
        },
        "nemo_ww_own_rms": own_rms,
        "vertical_velocity_scored_against_nemo": scored,
        "stage_sea_surface_height": stage_height,
        "round160_reference": {
            "corrected_ww_rms": ROUND160_CORRECTED_WW,
            "shared_ww_rms": 1.2326857042024439e-08,
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report

COMBINED_CANCELLATION_PAIR = ("advection", "vertical_diffusion")
DEVELOPED_RANK_SPLIT_PLANTS = ("rank-combined-mispair",)
# Round 129 measured the year harness's run-to-run temperature floor.
RUN_TO_RUN_FLOOR_K = 2.0e-10


def developed_process_rank_split(
        process_root: Path, daily_root: Path, daily_audit: Path,
        expected_commit: str, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Round 152's one-step magnitude ranking, in BOTH stage programs.

    Round 160's second continuity solve makes the developed stage-2 momentum
    vertical velocity five orders more exact and makes the YEAR worse.  The
    corrected velocity reaches the tracers only through the stage-3 transport,
    and round 134 measured that the tracer pair carries 99.7% of the day-240
    gap, so the question "which day-240 owner does the corrected velocity
    feed" is answered by running the SAME ranking in both programs from the
    same developed entry and reading which row grew.

    Round 162 adds the measurement that turns round 161's PLAUSIBLE
    cancellation into a verdict.  Round 161 measured that the corrected
    velocity collapses the tracer ADVECTION row by a factor of 1,776 and grows
    the VERTICAL DIFFUSION row, i.e. the two move in opposite directions, and
    named the discriminating measurement rather than running it: the COMBINED
    one-step temperature increment of the two rows, scored against NEMO's
    combined trend, in both arms, plus the cell-by-cell correlation of the two
    error fields in production.  If the two errors cancel, the combined row is
    below the vertical-diffusion row alone in production and GROWS in the
    corrected arm even though the advection row alone collapses.
    """
    require(plant in (None, "none") + DEVELOPED_RANK_SPLIT_PLANTS,
            f"unknown developed rank-split plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"process-rank split requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "process-rank split commit differs from --expect-commit")
    process_root = Path(process_root)
    evidence_root = Path(evidence_root)
    process_admission = validate_process_record(
        process_root, PROCESS_RECORD_COMMIT)
    record = read_process_record(
        process_root
        / f"oracle_process_budget_kt{DEVELOPED_PROCESS_STEP:08d}.bin")
    bundle = _developed_entry_bundle(daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    state = bundle["state"]
    wet = bundle["wet"]
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(bundle["payload"]["ssha"])
    nemo_increments = process_temperature_rows(record)

    def ranked(**extra):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                tracer_process_trace=(), vertical_solve_trace=True, **extra))
        trace = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        jax.device_get(trace)
        frame = _trace_frame(trace)
        lego = lego_process_temperature_rows(frame)
        increments = {
            name: _score_developed_row(lego[name], nemo_increments[name], wet)
            for name in PROCESS_ROWS
        }
        return increments, _rank_developed_process_rows(increments), lego

    # Preserve round 152's historical one-solve authority explicitly now that
    # round 163 made the two-solve arm the GYRE production default.  The
    # no-hook call below is the current production row and must reproduce the
    # explicit corrected arm bit for bit.
    production_rows, production_rank, production_lego = ranked(
        nemo_stage_momentum_wzv_split=False)
    corrected_rows, corrected_rank, corrected_lego = ranked(
        nemo_stage_momentum_wzv_split=True)
    landed_rows, landed_rank, landed_lego = ranked()

    # AUTHORITY: the production arm must reproduce round 152's own table, or
    # this is ranking a different step.
    ROUND152 = {
        "vertical_diffusion": 2.1834089000437955e-5,
        "lateral_diffusion": 1.0655217366755294e-5,
        "shortwave": 2.513690760498398e-7,
        "advection": 1.0974591404090626e-8,
        "geometry": 6.281725340556424e-12,
        "surface_boundary": 4.6577410196032515e-15,
    }
    for name, value in ROUND152.items():
        require(abs(production_rows[name]["rms"] - value)
                <= 1.0e-6 * abs(value),
                f"the production arm does not reproduce round 152's {name} "
                f"row: {production_rows[name]['rms']} against {value}")

    landed_identity = {
        name: _score_developed_row(
            landed_lego[name], corrected_lego[name], wet)
        for name in PROCESS_ROWS
    }
    require(all(row["cells_unequal"] == 0
                for row in landed_identity.values()),
            "the landed no-hook production arm differs from the explicit "
            f"two-solve arm: {landed_identity}")

    comparison = []
    for name in PROCESS_ROWS:
        before = production_rows[name]["rms"]
        after = corrected_rows[name]["rms"]
        comparison.append({
            "name": name,
            "production_rms_K": before,
            "corrected_rms_K": after,
            "change_K": float(after - before),
            "relative_change": (float((after - before) / before)
                                if before > 0.0 else None),
            "grew": bool(after > before),
            "production_cells_unequal": production_rows[name][
                "cells_unequal"],
            "corrected_cells_unequal": corrected_rows[name]["cells_unequal"],
        })
    grew = [row for row in comparison if row["grew"]]
    grew.sort(key=lambda row: row["change_K"], reverse=True)

    # ---- ROUND 162: the discriminating measurement for the cancellation ---
    first, second = COMBINED_CANCELLATION_PAIR
    mask = np.asarray(wet, dtype=bool)
    nemo_combined = nemo_increments[first] + nemo_increments[second]
    if plant == "rank-combined-mispair":
        # Wire the combined row to the WRONG NEMO trend while the two single
        # rows keep the right one.  The decomposition control below must
        # refuse it: a combined row that does not decompose into the two rows
        # the ranking already scored is measuring a different pairing.
        nemo_combined = (nemo_increments[first]
                         + nemo_increments["lateral_diffusion"])

    def combined_score(lego_rows):
        return _score_developed_row(
            lego_rows[first] + lego_rows[second], nemo_combined, mask)

    combined = {"production": combined_score(production_lego),
                "corrected": combined_score(corrected_lego)}

    # The two error fields, on the production arm, over the wet cells.
    e_first = (production_lego[first] - nemo_increments[first])[mask]
    e_second = (production_lego[second] - nemo_increments[second])[mask]
    combined_error = ((production_lego[first] + production_lego[second])
                      - nemo_combined)[mask]
    scale = float(np.max(np.abs(combined_error)))
    gap = combined_error - (e_first + e_second)
    residual = float(np.max(np.abs(gap)))
    decomposition = {
        "max_abs_residual_K": residual,
        "cells_with_a_residual": int(np.count_nonzero(gap)),
        "combined_error_max_abs_K": scale,
        "allowed_K": 8.0 * float(np.spacing(scale)),
        # The magnitudes the identity is exact or inexact AT, so a hard zero
        # can be read against the size of the numbers that produced it.
        "increment_max_abs_K": {
            f"lego_{first}": float(np.max(np.abs(production_lego[first][mask]))),
            f"lego_{second}": float(np.max(np.abs(production_lego[second][mask]))),
            f"nemo_{first}": float(np.max(np.abs(nemo_increments[first][mask]))),
            f"nemo_{second}": float(np.max(np.abs(nemo_increments[second][mask]))),
        },
    }
    decomposition["holds"] = bool(
        residual <= decomposition["allowed_K"])
    if plant in DEVELOPED_RANK_SPLIT_PLANTS:
        if decomposition["holds"]:
            raise GateError(
                "PLANT-BLIND: the decomposition control did not refuse a "
                f"combined row wired to the wrong trend: {decomposition}")
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": decomposition}
    require(decomposition["holds"],
            "the combined row does not decompose into the two scored rows: "
            f"{decomposition}")

    rms_first = float(np.sqrt(np.mean(e_first * e_first)))
    rms_second = float(np.sqrt(np.mean(e_second * e_second)))
    cross = float(2.0 * np.mean(e_first * e_second))
    quadrature = float(np.sqrt(rms_first ** 2 + rms_second ** 2))
    # The sum rule the verdict rests on, checked rather than assumed.
    quadrature_identity = float(
        combined["production"]["rms"] ** 2
        - (rms_first ** 2 + cross + rms_second ** 2))
    centred_first = e_first - float(np.mean(e_first))
    centred_second = e_second - float(np.mean(e_second))
    pearson = float(
        np.sum(centred_first * centred_second)
        / np.sqrt(np.sum(centred_first ** 2) * np.sum(centred_second ** 2)))
    cosine = float(
        np.sum(e_first * e_second)
        / np.sqrt(np.sum(e_first ** 2) * np.sum(e_second ** 2)))
    signed = e_first * e_second
    opposite = float(np.count_nonzero(signed < 0.0) / signed.size)
    growth = float(combined["corrected"]["rms"]
                   - combined["production"]["rms"])
    tests = {
        "correlation_below_minus_point_one": bool(pearson < -0.1),
        "production_combined_below_vertical_alone": bool(
            combined["production"]["rms"] < production_rows[second]["rms"]),
        "combined_grows_in_corrected": bool(
            growth > 10.0 * RUN_TO_RUN_FLOOR_K),
    }
    if all(tests.values()):
        verdict = "CONFIRMED"
    elif pearson >= -0.1 or growth <= 0.0:
        verdict = "REFUTED"
    else:
        verdict = "NEITHER"
    cancellation = {
        "pair": list(COMBINED_CANCELLATION_PAIR),
        "cells_scored": int(np.count_nonzero(mask)),
        "production_row_rms_K": {first: rms_first, second: rms_second},
        "production_combined_rms_K": combined["production"]["rms"],
        "corrected_combined_rms_K": combined["corrected"]["rms"],
        "production_vertical_diffusion_rms_K": production_rows[second]["rms"],
        "quadrature_reference_K": quadrature,
        "quadrature_identity_residual_K2": quadrature_identity,
        "cross_term_K2": cross,
        "combined_growth_K": growth,
        "run_to_run_floor_K": RUN_TO_RUN_FLOOR_K,
        "pearson_correlation": pearson,
        "uncentred_cosine": cosine,
        "opposite_sign_fraction": opposite,
        "decomposition_control": decomposition,
        "tests": tests,
        "verdict": verdict,
    }
    report = {
        "format": "gyre-round162-developed-process-rank-split-v2",
        "status": "PASS",
        "worktree": stamp,
        "process_record_admission": process_admission["status"],
        "entry": {"step": DEVELOPED_ENTRY_STEP,
                  "process_step": DEVELOPED_PROCESS_STEP},
        "round152_reference": ROUND152,
        "production_ranking": production_rank,
        "corrected_ranking": corrected_rank,
        "landed_production_ranking": landed_rank,
        "landed_production_rows": landed_rows,
        "landed_equals_explicit_corrected": landed_identity,
        "comparison": comparison,
        "rows_that_grew": grew,
        "largest_growth": grew[0] if grew else None,
        "largest_owner_production": production_rank[0]["name"],
        "largest_owner_corrected": corrected_rank[0]["name"],
        "combined_scored_rows": combined,
        "cancellation": cancellation,
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    return report

def _developed_entry_bundle(daily_root, daily_audit, expected_commit):
    """Build NEMO's admitted day-180 entry state for a developed walk.

    Extracted from ``developed_state_process_walk`` so that a second
    developed-state walk drives the SAME bridge, the same audit and the
    same source checks instead of a second copy of them.
    """
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    year = _year()
    audit, daily_hashes = year._daily_record_contract(
        daily_audit, daily_root, expected_commit)
    restart_path = year._daily_restart_path(daily_root, DEVELOPED_ENTRY_STEP)
    payload = year._load_daily_reset_payload(restart_path, PROCESS_JPK - 1)
    expected_payload_hash = daily_hashes.get(restart_path.name)
    require(expected_payload_hash is not None,
            "step-1080 restart is absent from the admitted daily manifest")
    year._require_payload_digest(payload, expected_payload_hash)

    card = build_nemo_testcase_card(CASE)
    gate = _gate()
    wet = gate.expected_masks(card)["T"]
    wet2 = np.any(wet, axis=-1)
    interface_wet = wet[..., :-1] & wet[..., 1:]
    state, _ = _gyre_checkpoint_state(card, restart_path)
    history = (
        jnp.asarray(year._face_history(payload["ub_e"], "u")),
        jnp.asarray(year._face_history(payload["ubb_e"], "u")),
        jnp.asarray(year._face_history(payload["vb_e"], "v")),
        jnp.asarray(year._face_history(payload["vbb_e"], "v")),
        jnp.asarray(payload["sshb_e"]),
        jnp.asarray(payload["sshbb_e"]),
    )
    state = state._replace(bt_hist=history)
    inventory = _developed_state_inventory(state)
    for row in inventory:
        print(f"  STATE {row['name']}: {row['classification']} "
              f"shape={row['shape']} dtype={row['dtype']}", flush=True)

    land = np.asarray(state.land_mask.data) > 0.5
    expected_tke = np.where(land[..., None], payload["en"][..., 1:], 0.0)
    expected_avm = np.where(
        land[..., None], payload["avm_k"][..., 1:], 0.0)
    expected_avt = np.where(
        land[..., None], payload["avt_k"][..., 1:], 0.0)
    expected_dissl = np.where(
        land[..., None], payload["dissl"][..., 1:], 0.0)
    expected_avm_surface = np.where(land, payload["avm_k"][..., 0], 0.0)
    source_checks = {
        "T_wet": _bit_mismatch_count(state.T.data, payload["tn"], wet),
        "S_wet": _bit_mismatch_count(state.S.data, payload["sn"], wet),
        "u_faces": _bit_mismatch_count(
            state.u.data, _u_east_to_face(payload["un"])),
        "v_faces": _bit_mismatch_count(
            state.v.data, _v_north_to_face(payload["vn"])),
        "eta": _bit_mismatch_count(state.eta.data, payload["sshn"]),
        "uu_b": _bit_mismatch_count(
            state.uu_b.data, year._face_history(payload["uu_n"], "u")),
        "vv_b": _bit_mismatch_count(
            state.vv_b.data, year._face_history(payload["vv_n"], "v")),
        "tke": _bit_mismatch_count(state.tke.data, expected_tke),
        "tke_avm": _bit_mismatch_count(state.tke_avm.data, expected_avm),
        "tke_avt": _bit_mismatch_count(state.tke_avt.data, expected_avt),
        "tke_dissl": _bit_mismatch_count(
            state.tke_dissl.data, expected_dissl),
        "tke_avm_surface": _bit_mismatch_count(
            state.tke_avm_surface.data, expected_avm_surface),
    }
    for index, (actual, expected) in enumerate(zip(
            state.bt_hist, history, strict=True)):
        source_checks[f"bt_hist_{index}"] = _bit_mismatch_count(
            actual, expected)
    require(all(value == 0 for value in source_checks.values()),
            "restart bridge changed a mapped source: " + ", ".join(
                f"{name}={value}" for name, value in source_checks.items()
                if value))
    return {
        "audit": audit, "daily_hashes": daily_hashes,
        "restart_path": restart_path, "payload": payload,
        "card": card, "gate": gate, "wet": wet, "wet2": wet2,
        "interface_wet": interface_wet, "state": state,
        "inventory": inventory, "land": land,
        "source_checks": source_checks,
    }


def developed_state_process_walk(
        process_root: Path, vertical_root: Path, daily_root: Path,
        daily_audit: Path, expected_commit: str, evidence_root: Path, *,
        mesh_path: Path = DEFAULT_MESH, plant: str | None = None,
        fct_record_root: Path | None = None,
        transport_record_root: Path | None = None) -> dict:
    """Run step 1081 through production JIT from NEMO's exact day-180 state."""
    if plant in ("missing-day", "missing-process-row", "missing-branch",
                 "missing-ranking-row"):
        return _developed_registry_plant(plant)
    require(plant in (None, "none", "entry-temperature-ulp",
                      "fct-transport-ulp", "transport-un-adv-ulp",
                      "stage2-uu-b-ulp", "vertical-heat-ulp"),
            f"unknown developed-state plant {plant!r}")
    _policy()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean import advection as advection_module
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.advection import NEMO_FCT_TRACE_FIELDS
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"developed-state walk requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "developed-state walk commit differs from --expect-commit")
    process_root = Path(process_root)
    vertical_root = Path(vertical_root)
    daily_root = Path(daily_root)
    daily_audit = Path(daily_audit)
    evidence_root = Path(evidence_root)
    fct_bundle = (
        _developed_fct_record(Path(fct_record_root))
        if fct_record_root is not None else None)
    transport_bundle = (
        _developed_transport_record(Path(transport_record_root))
        if transport_record_root is not None else None)

    process_admission = validate_process_record(
        process_root, PROCESS_RECORD_COMMIT)
    vertical_admission = validate_vertical_record(
        vertical_root, DEVELOPED_VERTICAL_RECORD_COMMIT,
        process_root=process_root)
    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    audit = bundle["audit"]
    payload = bundle["payload"]
    card = bundle["card"]
    gate = bundle["gate"]
    wet = bundle["wet"]
    wet2 = bundle["wet2"]
    interface_wet = bundle["interface_wet"]
    state = bundle["state"]
    inventory = bundle["inventory"]
    restart_path = bundle["restart_path"]
    source_checks = bundle["source_checks"]

    record_path = (process_root
                   / f"oracle_process_budget_kt{DEVELOPED_PROCESS_STEP:08d}.bin")
    record = read_process_record(record_path)
    entry_t_unequal = _different_cells(
        np.asarray(state.T.data),
        np.asarray(record["Tbb"][..., :wet.shape[-1]]), wet)

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    control_trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True,
            tracer_process_branch_activity=True))
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    ssha = jnp.asarray(payload["ssha"])
    control_trace = control_trace_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)
    trace = trace_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)

    fct_modes = {}
    fct_observer_unequal_bytes = 0
    if fct_bundle is not None:
        real_fct = advection_module.fct_tracer_advection

        def collapse(calls, label):
            require(len(calls) >= 2 and len(calls) % 2 == 0,
                    f"{label} observed {len(calls)} FCT calls")
            pair = {"T": calls[0], "S": calls[1]}
            for index, duplicate in enumerate(calls[2:], 2):
                original = calls[index % 2]
                require(len(duplicate) == len(original)
                        and all(np.array_equal(left, right)
                                for left, right in zip(
                                    duplicate, original, strict=True)),
                        f"{label} observed distinct duplicate FCT calls")
            return pair

        def run_observed(*, eager=False, plant_index=None):
            calls = []

            def sink(*values):
                calls.append(tuple(np.asarray(value) for value in values))

            def capture(*values, **kwargs):
                call_values = list(values)
                base = kwargs.get("tracer_before")
                active = kwargs.get("active_mask")
                h_base = kwargs.get("base_thickness")
                h_after = kwargs.get("after_thickness")
                implicit_w = kwargs.get("implicit_w")
                require(base is not None and active is not None
                        and h_base is not None and h_after is not None
                        and implicit_w is not None,
                        "developed FCT call lacks a recorded NEMO operand")
                if plant_index is not None:
                    old = call_values[1][plant_index]
                    call_values[1] = call_values[1].at[plant_index].set(
                        jnp.nextafter(old, jnp.asarray(jnp.inf, old.dtype)))
                div_h, div_w, fct_trace = real_fct(
                    *call_values, **kwargs, return_nemo_trace=True)
                grid = call_values[5]
                observed = (
                    call_values[0], base,
                    call_values[1] * jnp.asarray(grid.dy_u)[..., None],
                    call_values[2] * jnp.asarray(grid.dx_v)[..., None],
                    call_values[3] * jnp.asarray(grid.area_T)[..., None],
                    h_base, call_values[4], h_after, active, implicit_w,
                    call_values[1], call_values[2], call_values[3],
                    *fct_trace,
                )
                jax.debug.callback(sink, *observed, ordered=True)
                return div_h, div_w

            observed_model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config)
            advection_module.fct_tracer_advection = capture
            try:
                if eager:
                    with jax.disable_jit():
                        result = observed_model._step_impl(
                            state, card.dt_s, freshwater=freshwater,
                            surface_forcing=surface,
                            _nemo_stage1_zad_eta_after_override=ssha)
                        jax.device_get(result)
                else:
                    result = observed_model.step(
                        state, dt=card.dt_s, freshwater=freshwater,
                        surface_forcing=surface,
                        _nemo_stage1_zad_eta_after_override=ssha)
                    jax.device_get(result)
                jax.effects_barrier()
            finally:
                advection_module.fct_tracer_advection = real_fct
            return result, collapse(
                calls, "production eager" if eager else "production-step JIT")

        observed_state, production_jit = run_observed()
        fct_modes["production_step_jit"] = production_jit
        if plant == "fct-transport-ulp":
            raw_u = production_jit["T"][10]
            candidates = np.argwhere(np.isfinite(raw_u) & (raw_u != 0.0))
            require(candidates.size > 0, "no nonzero production U transport")
            plant_index = tuple(int(value) for value in candidates[0])
            _planted_state, planted = run_observed(plant_index=plant_index)
            trace_offset = 13
            first_u = NEMO_FCT_TRACE_FIELDS.index("first_u")
            moved = _score_developed_fct(
                planted["T"][trace_offset + first_u],
                production_jit["T"][trace_offset + first_u])
            require(moved["cells_unequal"] > 0,
                    "production FCT transport ULP plant moved no first face")
            return {
                "status": "PLANT-FIRED", "plant": plant,
                "control": {"raw_u_index": list(plant_index),
                            "first_u_T": moved},
            }
        _eager_state, production_eager = run_observed(eager=True)
        fct_modes["production_eager"] = production_eager

        isolated = {}
        for tracer in ("T", "S"):
            observed = production_jit[tracer]
            (now, base, _p_u, _p_v, _p_w, h_base, h_now, h_after,
             active, implicit_w, raw_u, raw_v, raw_w) = observed[:13]

            @jax.jit
            def isolated_call(
                    now_value, base_value, u_value, v_value, w_value,
                    h_base_value, h_now_value, h_after_value, active_value,
                    implicit_value):
                return real_fct(
                    now_value, u_value, v_value, w_value, h_now_value,
                    card.recipe.grid, card.dt_s, high_order="centred2",
                    tracer_before=base_value, active_mask=active_value,
                    low_order_predictor="nemo_rk3_two_step",
                    base_thickness=h_base_value,
                    after_thickness=h_after_value,
                    implicit_w=implicit_value, return_nemo_trace=True)

            _dh, _dw, isolated_trace = isolated_call(
                jnp.asarray(now), jnp.asarray(base), jnp.asarray(raw_u),
                jnp.asarray(raw_v), jnp.asarray(raw_w), jnp.asarray(h_base),
                jnp.asarray(h_now), jnp.asarray(h_after), jnp.asarray(active),
                jnp.asarray(implicit_w))
            isolated[tracer] = tuple(observed[:13]) + tuple(
                np.asarray(value) for value in jax.device_get(isolated_trace))
        fct_modes["isolated_closure_jit"] = isolated

    transport_modes = {}
    transport_observer_unequal_bytes = 0
    transport_plant_index = None
    if transport_bundle is not None:
        real_corrected = model_module._nemo_stage_corrected_velocity
        real_metric = model_module._nemo_metric_stage_transport
        real_qco_faces = model_module._nemo_ws_qco_stage_faces

        def run_transport_observed(*, eager=False, plant_index=None):
            corrected_calls = []
            metric_calls = []
            qco_calls = []
            corrected_trace_count = 0
            metric_trace_count = 0

            def corrected_capture(
                    velocity, transport_average, inverse_depth,
                    barotropic_velocity, face_mask):
                nonlocal corrected_trace_count
                call_index = corrected_trace_count
                corrected_trace_count += 1
                average = transport_average
                if plant_index is not None and call_index == 4:
                    old = average[plant_index]
                    average = average.at[plant_index].set(
                        jnp.nextafter(old, jnp.asarray(jnp.inf, old.dtype)))
                result, correction = real_corrected(
                    velocity, average, inverse_depth, barotropic_velocity,
                    face_mask, return_correction=True)

                def sink(*values):
                    corrected_calls.append(tuple(
                        np.asarray(value) for value in values))

                jax.debug.callback(
                    sink, velocity, average, inverse_depth,
                    barotropic_velocity, face_mask, correction, result,
                    ordered=True)
                return result

            def metric_capture(metric, face_thickness, corrected_velocity):
                nonlocal metric_trace_count
                metric_trace_count += 1
                result = real_metric(
                    metric, face_thickness, corrected_velocity)

                def sink(*values):
                    metric_calls.append(tuple(
                        np.asarray(value) for value in values))

                jax.debug.callback(
                    sink, metric, face_thickness, corrected_velocity, result,
                    ordered=True)
                return result

            def qco_capture(*values, **kwargs):
                result = real_qco_faces(*values, **kwargs)
                if kwargs.get("include_reciprocals", False):
                    def sink(*arrays):
                        qco_calls.append(tuple(
                            np.asarray(value) for value in arrays))

                    jax.debug.callback(
                        sink, result[0], result[2], result[4],
                        values[1], values[2], values[3],
                        ordered=True)
                return result

            observed_model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config)
            model_module._nemo_stage_corrected_velocity = corrected_capture
            model_module._nemo_metric_stage_transport = metric_capture
            model_module._nemo_ws_qco_stage_faces = qco_capture
            try:
                if eager:
                    with jax.disable_jit():
                        result = observed_model._step_impl(
                            state, card.dt_s, freshwater=freshwater,
                            surface_forcing=surface,
                            _nemo_stage1_zad_eta_after_override=ssha)
                        jax.device_get(result)
                else:
                    result = observed_model.step(
                        state, dt=card.dt_s, freshwater=freshwater,
                        surface_forcing=surface,
                        _nemo_stage1_zad_eta_after_override=ssha)
                    jax.device_get(result)
                jax.effects_barrier()
            finally:
                model_module._nemo_stage_corrected_velocity = real_corrected
                model_module._nemo_metric_stage_transport = real_metric
                model_module._nemo_ws_qco_stage_faces = real_qco_faces

            require(len(corrected_calls) == 6,
                    f"transport observer saw {len(corrected_calls)} "
                    "corrected-velocity calls, expected six")
            require(len(metric_calls) == 6,
                    f"transport observer saw {len(metric_calls)} metric "
                    "transport calls, expected six")
            stage3_u = corrected_calls[4]
            stage3_metric_u = metric_calls[4]
            require(np.array_equal(np.asarray(stage3_metric_u[0]),
                                   np.asarray(card.recipe.grid.dy_u)),
                    "the fifth metric-transport call is not the U call: its "
                    "metric is not the card's e2u")
            matches = [row for row in qco_calls
                       if np.array_equal(row[0], stage3_metric_u[1])
                       and np.array_equal(row[2], stage3_u[2])]
            require(bool(matches),
                    "no reciprocal-QCO call matches the stage-3 U operands")
            one_plus_r3 = matches[-1][1]
            actual = {
                "un_adv": stage3_u[1],
                "r1_hu_0": None,
                "one_plus_r3u_Kmm": one_plus_r3,
                "live_inverse_depth": stage3_u[2],
                "uu_b_Kmm": stage3_u[3],
                "zub": stage3_u[5],
                "e2u": stage3_metric_u[0],
                "e3u_0": None,
                "umask": stage3_u[4],
                "live_e3u_Kmm": stage3_metric_u[1],
                "uu_Kmm": stage3_u[0],
                "corrected_u": stage3_u[6],
                "zFu": stage3_metric_u[3],
            }
            # Zero-ssh through the SAME shared QCO builder exposes the static
            # reference e3u_0 and r1_hu_0 operands without inverting a live
            # product.  This is a context row, not a second transport formula.
            reference = real_qco_faces(
                jnp.zeros_like(state.eta.data),
                jnp.asarray(matches[-1][3]),
                jnp.asarray(matches[-1][4]),
                jnp.asarray(matches[-1][5]),
                card.recipe.grid, include_reciprocals=True)
            actual["e3u_0"] = np.asarray(reference[0])
            actual["r1_hu_0"] = np.asarray(reference[4])
            require(set(actual) == set(DEVELOPED_TRANSPORT_U_ROWS),
                    "production U-transport observation registry is incomplete")
            replay_live = actual["e3u_0"] * (
                np.float64(1.0)
                + (actual["one_plus_r3u_Kmm"] - np.float64(1.0))[..., None]
                * actual["umask"])
            closure = _score_developed_fct(
                replay_live, actual["live_e3u_Kmm"])
            return result, actual, closure

        (transport_state, production_transport,
         production_transport_closure) = run_transport_observed()
        transport_modes["production_step_jit"] = production_transport
        if plant == "transport-un-adv-ulp":
            un_adv = production_transport["un_adv"]
            inverse = production_transport["live_inverse_depth"]
            baro = production_transport["uu_b_Kmm"]
            baseline = (un_adv * inverse) - baro
            candidates = np.argwhere(np.isfinite(un_adv) & (un_adv != 0.0))
            for candidate in candidates:
                index = tuple(int(value) for value in candidate)
                moved = np.nextafter(un_adv[index], np.inf)
                if ((moved * inverse[index]) - baro[index]) != baseline[index]:
                    transport_plant_index = index
                    break
            require(transport_plant_index is not None,
                    "no nonzero un_adv ULP reaches the stage-3 correction")
            _planted_state, planted, _planted_closure = run_transport_observed(
                plant_index=transport_plant_index)
            moved = _score_developed_fct(
                planted["zFu"], production_transport["zFu"])
            require(moved["cells_unequal"] > 0,
                    "production un_adv ULP plant moved no completed zFu")
            return {
                "status": "PLANT-FIRED", "plant": plant,
                "control": {"un_adv_index": list(transport_plant_index),
                            "zFu": moved},
            }
        (_eager_transport_state, eager_transport,
         eager_transport_closure) = run_transport_observed(eager=True)
        transport_modes["production_eager"] = eager_transport

        @jax.jit
        def isolated_transport(
                velocity, average, inverse, barotropic, mask, metric,
                thickness):
            # The SAME two shared helpers the production stage calls, so this
            # closure measures legoESM's transcription and not a second copy
            # of the compiled statements written inside the instrument.
            corrected, correction = real_corrected(
                velocity, average, inverse, barotropic, mask,
                return_correction=True)
            completed = real_metric(metric, thickness, corrected)
            return correction, corrected, completed

        isolated_values = isolated_transport(
            jnp.asarray(production_transport["uu_Kmm"]),
            jnp.asarray(production_transport["un_adv"]),
            jnp.asarray(production_transport["live_inverse_depth"]),
            jnp.asarray(production_transport["uu_b_Kmm"]),
            jnp.asarray(production_transport["umask"]),
            jnp.asarray(production_transport["e2u"]),
            jnp.asarray(production_transport["live_e3u_Kmm"]))
        isolated_rows = dict(production_transport)
        (isolated_rows["zub"], isolated_rows["corrected_u"],
         isolated_rows["zFu"]) = (
            np.asarray(value) for value in jax.device_get(isolated_values))
        transport_modes["isolated_closure_jit"] = isolated_rows

        # Decision 43 ranks owners by MAGNITUDE, so the completed transport
        # is also attributed operand by operand: the SAME isolated closure
        # re-evaluates stprk3_stg.f90:303-304 and :314 with ONE operand
        # replaced by NEMO's recorded value and scores the product against
        # NEMO's own zFu.  The ``all`` arm is the calibration: NEMO's own
        # operands must rebuild NEMO's transport bit for bit, or the
        # statement association itself is the candidate.  These arms are
        # isolated-closure JIT and are never relabelled as production.
        oracle_rows = transport_bundle["rows"]
        substitution_operands = (
            "e2u", "live_e3u_Kmm", "uu_Kmm", "un_adv",
            "live_inverse_depth", "uu_b_Kmm", "umask")
        oracle_active = np.asarray(oracle_rows["umask"]) != 0.0

        def transport_arm(substituted):
            picked = tuple(
                jnp.asarray(
                    oracle_rows[name] if name in substituted
                    else production_transport[name])
                for name in ("uu_Kmm", "un_adv", "live_inverse_depth",
                             "uu_b_Kmm", "umask", "e2u", "live_e3u_Kmm"))
            completed = np.asarray(
                jax.device_get(isolated_transport(*picked)[2]))
            row = _score_developed_fct(completed, oracle_rows["zFu"])
            delta = completed[oracle_active] - np.asarray(
                oracle_rows["zFu"])[oracle_active]
            bits = (completed.view(np.uint64)
                    != np.asarray(oracle_rows["zFu"]).view(np.uint64))
            row["substituted"] = list(substituted)
            row["active_cells_scored"] = int(np.count_nonzero(oracle_active))
            row["active_cells_unequal"] = int(
                np.count_nonzero(bits & oracle_active))
            row["active_max_abs"] = float(
                np.max(np.abs(delta), initial=0.0))
            row["active_rms"] = float(
                np.sqrt(np.mean(delta * delta)) if delta.size else 0.0)
            return row, completed

        none_row, none_completed = transport_arm(())
        # Calibration of the attribution instrument itself: with nothing
        # substituted the isolated closure must reproduce the production
        # transport row it stands in for, BYTE for byte, not merely with the
        # same count of unequal cells.
        require(np.array_equal(
                    none_completed.view(np.uint64),
                    np.asarray(production_transport["zFu"]).view(np.uint64)),
                "transport attribution baseline is not byte-identical to the "
                "production transport row")
        transport_attribution = {"none": none_row}
        baseline_max = none_row["active_max_abs"]
        baseline_rms = none_row["active_rms"]
        for name in substitution_operands:
            transport_attribution[name] = transport_arm((name,))[0]
        transport_attribution["uu_Kmm+un_adv"] = transport_arm(
            ("uu_Kmm", "un_adv"))[0]
        transport_attribution["all"] = transport_arm(substitution_operands)[0]
        for name, row in transport_attribution.items():
            row["active_max_abs_removed_fraction"] = (
                float((baseline_max - row["active_max_abs"]) / baseline_max)
                if baseline_max > 0.0 else 0.0)
            # The argmax is free to move between arms, so the ranking uses the
            # rms, which is a whole-field quantity, and reports both.
            row["active_rms_removed_fraction"] = (
                float((baseline_rms - row["active_rms"]) / baseline_rms)
                if baseline_rms > 0.0 else 0.0)
        ranked = sorted(
            (name for name in substitution_operands),
            key=lambda name: -transport_attribution[name][
                "active_rms_removed_fraction"])

        # Round 156: the magnitude owner of the transport difference is the
        # stage-2 velocity, so the SAME production observation is split
        # between the two halves of the compiled stage-2 program.
        stage2_split = _developed_stage2_velocity_split(
            production_transport, oracle_rows)
        if plant == "stage2-uu-b-ulp":
            oracle_active2 = np.any(oracle_active, axis=-1)
            candidates = np.argwhere(
                oracle_active2
                & (np.asarray(oracle_rows["uu_b_Kmm"]) != 0.0))
            require(candidates.size > 0,
                    "no active nonzero uu_b column to plant")
            index = tuple(int(value) for value in candidates[0])
            planted_split = _developed_stage2_velocity_split(
                production_transport, oracle_rows, plant_index=index)
            before = stage2_split["reprojection_sha256"]
            after = planted_split["reprojection_sha256"]
            require(after != before,
                    "stage-2 uu_b ULP plant moved no reprojected velocity")
            return {
                "status": "PLANT-FIRED", "plant": plant,
                "control": {
                    "uu_b_index": list(index),
                    "before_reprojection_sha256": before,
                    "after_reprojection_sha256": after,
                    "before_row": stage2_split["rows"][
                        "nemo_depth_mean_substituted"],
                    "after_row": planted_split["rows"][
                        "nemo_depth_mean_substituted"],
                },
            }

    ordinary = ordinary_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=ssha)
    observer_unequal_bytes = _state_bit_mismatches(
        trace.state_after, ordinary)
    require(observer_unequal_bytes == 0,
            f"process observer changed {observer_unequal_bytes} state bytes")
    if fct_bundle is not None:
        fct_observer_unequal_bytes = _state_bit_mismatches(
            observed_state, ordinary)
        require(fct_observer_unequal_bytes == 0,
                "FCT statement observer changed production state")
    if transport_bundle is not None:
        transport_observer_unequal_bytes = _state_bit_mismatches(
            transport_state, ordinary)
        require(transport_observer_unequal_bytes == 0,
                "transport observer changed production state")

    lego_frame = _trace_frame(trace)
    control_frame = _trace_frame(control_trace)
    branch_observer_unequal = {
        name: _different_cells(
            lego_frame[name], control_frame[name],
            wet if lego_frame[name].ndim == 3 else wet2)
        for name in LEGO_PROCESS_FIELDS
    }
    lego_vertical_frame = _vertical_trace_frame(trace)
    control_vertical_frame = _vertical_trace_frame(control_trace)
    for name in LEGO_VERTICAL_FIELDS:
        values = lego_vertical_frame[name]
        selected = (interface_wet
                    if values.shape[-1] == wet.shape[-1] - 1 else wet)
        branch_observer_unequal[f"vertical_{name}"] = _different_cells(
            values, control_vertical_frame[name], selected)
    require(all(value == 0 for value in branch_observer_unequal.values()),
            "FCT branch observer moved an existing process/vertical row: "
            + ", ".join(
                f"{name}={value}" for name, value
                in branch_observer_unequal.items() if value))
    fct_walk = None
    if fct_bundle is not None:
        for mode_name, observed in fct_modes.items():
            for tracer in ("T", "S"):
                require(len(observed[tracer]) == 35,
                        f"{mode_name} {tracer} has incomplete observation")
            for index in range(2, 10):
                require(np.array_equal(
                    observed["T"][index], observed["S"][index]),
                    f"{mode_name} T/S FCT context index {index} differs")
        zcoord = card.recipe.z_coord
        def common_actual(observed):
            reference = observed["T"]
            return {
                "p2dt": np.asarray([card.dt_s], dtype=np.float64),
                "transport_u": reference[2],
                "transport_v": reference[3],
                "transport_w": reference[4],
                "e3t_3d": np.asarray(zcoord.nemo_e3t_0, dtype=np.float64),
                "r3t_Kbb": reference[5],
                "r3t_Kmm": reference[6],
                "r3t_Kaa": reference[7],
                "tmask": reference[8],
                # GYRE-zco has full-depth wet columns, so the consumed
                # interior W mask and T mask have identical owned values.
                "wmask": reference[8],
                "r1_e1e2t": np.float64(1.0) / np.asarray(
                    card.recipe.grid.area_T, dtype=np.float64),
            }
        mode_rows = {
            name: _developed_fct_mode_rows(
                values, fct_bundle, common_actual(values))
            for name, values in fct_modes.items()}
        production = mode_rows["production_step_jit"]
        all_record_fields = set(DEVELOPED_FCT_COMMON_FIELDS)
        all_record_fields.update(
            f"{name}_{tracer}" for tracer in ("T", "S")
            for name in DEVELOPED_FCT_TRACER_FIELDS)
        require(len(all_record_fields) == fct_bundle["field_count"] == 61,
                "developed FCT scored-field registry is not 61 rows")
        fct_walk = {
            "record_root": str(fct_record_root),
            "record_sha256": fct_bundle["sha256"],
            "admission_sha256": _sha256(
                Path(fct_record_root)
                / "round153_developed_fct_admission.json"),
            "record_fields": sorted(all_record_fields),
            "field_count": fct_bundle["field_count"],
            "modes": mode_rows,
            "authoritative_mode": "production_step_jit",
            "first_non_bit_context": production["first_non_bit_context"],
            "first_non_bit_statement": production[
                "first_non_bit_statement"],
            "observer_state_unequal_bytes": fct_observer_unequal_bytes,
        }
    transport_walk = None
    if transport_bundle is not None:
        mode_rows = {
            mode_name: _developed_transport_mode_rows(
                values, transport_bundle["rows"])
            for mode_name, values in transport_modes.items()
        }
        production = mode_rows["production_step_jit"]
        transport_walk = {
            "record_root": str(transport_record_root),
            "record_sha256": transport_bundle["sha256"],
            "admission_sha256": _sha256(
                Path(transport_record_root)
                / "round154_developed_transport_admission.json"),
            "record_fields": transport_bundle["admission"]["record"][
                "fields"],
            "record_field_count": transport_bundle["field_count"],
            "scored_rows": list(DEVELOPED_TRANSPORT_U_ROWS),
            "modes": mode_rows,
            "authoritative_mode": "production_step_jit",
            "first_non_bit_row": production["first_non_bit_row"],
            "observer_state_unequal_bytes": transport_observer_unequal_bytes,
            "model_live_e3_reconstruction": {
                "production_step_jit": production_transport_closure,
                "production_eager": eager_transport_closure,
            },
            "row_provenance": {
                "e3u_0": "static card geometry, re-evaluated EAGERLY through "
                         "the shared QCO builder at zero ssh from the "
                         "production call's own h_ref and masks",
                "r1_hu_0": "static card geometry, same eager re-evaluation",
                "live_inverse_depth": "production value, the r1_hu the stage "
                                      "consumed",
                "zub": "production value returned by the shared corrected-"
                       "velocity helper",
                "default": "production value sunk from the executed stage",
            },
            "operand_attribution": {
                "mode": "isolated_closure_jit",
                "scored_against": "NEMO recorded zFu",
                "baseline_active_max_abs": baseline_max,
                "baseline_active_rms": baseline_rms,
                "arms": transport_attribution,
                "ranked_by_removed_fraction": ranked,
                "ranking_metric": "active_rms_removed_fraction",
                "magnitude_owner": ranked[0],
            },
            "stage2_velocity_split": stage2_split,
        }
    if plant == "entry-temperature-ulp":
        j, i, k = (int(value) for value in np.argwhere(wet)[0])
        planted_t = np.array(state.T.data, copy=True)
        before = planted_t[j, i, k]
        planted_t[j, i, k] = np.nextafter(before, np.inf)
        require(planted_t[j, i, k] != before,
                "entry-temperature ULP plant rounded away")
        planted_state = state._replace(
            T=state.T.replace(data=jnp.asarray(planted_t)))
        planted_trace = trace_model.step(
            planted_state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=ssha)
        planted_frame = _trace_frame(planted_trace)
        moved = {
            name: _different_cells(
                lego_frame[field], planted_frame[field], wet)
            for name, field in DEVELOPED_BOUNDARY_FIELDS.items()
        }
        require(any(value > 0 for value in moved.values()),
                "entry-temperature ULP plant moved no registered boundary")
        return {
            "status": "PLANT-FIRED", "plant": plant,
            "control": {"index_jik": [j, i, k],
                        "old_uint64": int(np.asarray(before).view(np.uint64)),
                        "new_uint64": int(np.asarray(
                            planted_t[j, i, k]).view(np.uint64)),
                        "moved_boundaries": moved},
        }

    nemo_boundaries = _process_cumulative_boundaries(record)
    lego_boundaries = {
        name: np.asarray(lego_frame[field], dtype=np.float64)
        for name, field in DEVELOPED_BOUNDARY_FIELDS.items()
    }
    nemo_qco = {
        "q_Kbb": 1.0 + np.asarray(record["r3t_Kbb"], dtype=np.float64),
        "q_Kmm": 1.0 + np.asarray(record["r3t_Kmm"], dtype=np.float64),
        "q_Kaa": 1.0 + np.asarray(record["r3t_Kaa"], dtype=np.float64),
    }
    geometry_operands = {
        name: _score_developed_row(lego_frame[name], values, wet2)
        for name, values in nemo_qco.items()
    }
    boundary_rows = {
        name: _score_developed_row(
            lego_boundaries[name], nemo_boundaries[name], wet)
        for name in PROCESS_ROWS
    }
    nemo_increments = process_temperature_rows(record)
    lego_increments = lego_process_temperature_rows(lego_frame)
    increment_rows = {
        name: _score_developed_row(
            lego_increments[name], nemo_increments[name], wet)
        for name in PROCESS_ROWS
    }
    process_ranking = _rank_developed_process_rows(increment_rows)
    first_non_bit = next(
        (name for name in PROCESS_ROWS
         if boundary_rows[name]["cells_unequal"]), "NONE")
    first_non_bit_process_call = next(
        (name for name in PROCESS_ROWS[1:]
         if increment_rows[name]["cells_unequal"]), "NONE")
    first_diff = (
        np.zeros_like(wet, dtype=bool) if first_non_bit == "NONE" else
        ((lego_boundaries[first_non_bit].view(np.uint64)
          != nemo_boundaries[first_non_bit].view(np.uint64)) & wet))

    vertical_record = _read_vertical_record(
        vertical_root
        / f"oracle_trazdf_matrix_kt{DEVELOPED_PROCESS_STEP:08d}.bin",
        DEVELOPED_PROCESS_STEP)
    nlev = wet.shape[-1]
    vertical_expected = {
        "heat_K": _vertical_field(vertical_record, "avt")[:, :, 1:nlev],
        "isoneutral_K": _vertical_field(
            vertical_record, "ah_wslp2")[:, :, 1:nlev],
        "effective_K": (
            _vertical_field(vertical_record, "avt")[:, :, 1:nlev]
            + _vertical_field(vertical_record, "ah_wslp2")[:, :, 1:nlev]),
        "e3t_after": _vertical_field(vertical_record, "e3t_Kaa", nlev),
        "e3w_now": _vertical_field(vertical_record, "e3w_Kmm")[:, :, 1:nlev],
        "content_T": _vertical_field(vertical_record, "rhs_T", nlev),
        "lower": _vertical_field(vertical_record, "zwi"),
        "diagonal": _vertical_field(vertical_record, "zwd"),
        "upper": _vertical_field(vertical_record, "zws"),
        "solved_T": _vertical_field(
            vertical_record, "sol_T_pre_clamp", nlev),
    }
    vertical_rows = {}
    for name in LEGO_VERTICAL_FIELDS:
        selected = interface_wet if name in (
            "heat_K", "isoneutral_K", "effective_K", "e3w_now") else wet
        vertical_rows[name] = _score_developed_row(
            lego_vertical_frame[name], vertical_expected[name], selected)
    vertical_order = list(LEGO_VERTICAL_FIELDS)
    first_vertical_non_bit = next(
        (name for name in vertical_order
         if not vertical_rows[name]["bit_exact"]), "NONE")
    if plant == "vertical-heat-ulp":
        control = _vertical_effect_control(
            trace_model, state, freshwater, surface, trace, wet)
        return {"status": "PLANT-FIRED", "plant": plant,
                "control": control}
    nemo_heat = _vertical_field(vertical_record, "avt")[
        :, :, 1:wet.shape[-1]]
    lego_heat = np.asarray(trace.vertical_solve.heat_K, dtype=np.float64)
    lego_viscosity = np.asarray(
        trace.vertical_solve.viscosity_K, dtype=np.float64)
    require(nemo_heat.shape == lego_heat.shape == interface_wet.shape,
            "developed EVD coefficient shapes differ")
    config = card.recipe.model_config.physics
    evd_value = float(config.convection.enhanced_diffusion.K_conv)
    tke_config = config.vertical_mixing.tke
    kh_floor = float(tke_config.kappaH_min)
    km_floor = float(tke_config.kappaM_min)
    energy_floor = float(tke_config.tke_background)
    nemo_evd = (nemo_heat == evd_value) & interface_wet
    lego_evd = (lego_heat == evd_value) & interface_wet
    nemo_kh_floor = (nemo_heat == kh_floor) & interface_wet & ~nemo_evd
    lego_kh_floor = (lego_heat == kh_floor) & interface_wet & ~lego_evd
    lego_km_floor = (lego_viscosity == km_floor) & interface_wet & ~lego_evd
    lego_energy_floor = (
        np.asarray(trace.state_after.tke.data) == energy_floor) & interface_wet
    fct_activity = np.asarray(trace.fct_activity, dtype=bool) & wet
    nemo_evd_cells = _interface_cells(nemo_evd, wet.shape[-1]) & wet
    lego_evd_cells = _interface_cells(lego_evd, wet.shape[-1]) & wet
    nemo_kh_cells = _interface_cells(nemo_kh_floor, wet.shape[-1]) & wet
    lego_floor_cells = (
        _interface_cells(
            lego_kh_floor | lego_km_floor | lego_energy_floor,
            wet.shape[-1]) & wet)
    branches = {
        "fct_nonosc": {
            "NEMO": {"status": "UNMEASURED",
                     "reason": "R123 stores no nonosc coefficients"},
            "legoESM_on_NEMO_entry": {
                "status": "MEASURED_PRODUCTION_STEP",
                **_branch_census(fct_activity, first_diff),
            },
        },
        "evd_replacement": {
            "NEMO": {"status": "MEASURED_ADMITTED_AVT",
                     **_branch_census(nemo_evd_cells, first_diff)},
            "legoESM_on_NEMO_entry": {
                "status": "MEASURED_PRODUCTION_STEP",
                **_branch_census(lego_evd_cells, first_diff)},
            "interface_cells_disagree": int(np.count_nonzero(
                nemo_evd != lego_evd)),
        },
        "tke_floors": {
            "NEMO": {
                "status": "PARTIAL_MEASURED_ADMITTED_AVT",
                "diffusivity_floor_interfaces": int(np.count_nonzero(
                    nemo_kh_floor)),
                **_branch_census(nemo_kh_cells, first_diff),
                "unmeasured": ["viscosity_floor", "post_solve_energy_floor"],
            },
            "legoESM_on_NEMO_entry": {
                "status": "MEASURED_PRODUCTION_STEP",
                "diffusivity_floor_interfaces": int(np.count_nonzero(
                    lego_kh_floor)),
                "viscosity_floor_interfaces": int(np.count_nonzero(
                    lego_km_floor)),
                "post_solve_energy_floor_interfaces": int(np.count_nonzero(
                    lego_energy_floor)),
                **_branch_census(lego_floor_cells, first_diff),
            },
        },
    }
    if fct_walk is not None:
        authoritative = fct_walk["modes"]["production_step_jit"]
        nemo_active = {}
        lego_active = {}
        selection_disagrees = {}
        for tracer in ("T", "S"):
            activity_rows = authoritative["tracers"][tracer][
                "limiter_activity"]
            nemo_active[tracer] = sum(
                row["NEMO_active"] for row in activity_rows.values())
            lego_active[tracer] = sum(
                row["lego_active"] for row in activity_rows.values())
            selection_disagrees[tracer] = sum(
                row["selection_disagrees"]
                for row in activity_rows.values())
        branches["fct_nonosc"]["NEMO"] = {
            "status": "MEASURED_DIRECT_COEFFICIENT_RECORD",
            "active_face_coefficients": nemo_active,
        }
        branches["fct_nonosc"]["legoESM_on_NEMO_entry"].update({
            "active_face_coefficients": lego_active,
            "selection_disagrees": selection_disagrees,
            "direct_record_status": "MEASURED_PRODUCTION_STEP",
        })

    evidence_root.mkdir(parents=True, exist_ok=True)
    maps_path = evidence_root / (
        f"developed_branch_maps_{expected_commit[:12]}.npz")
    require(not maps_path.exists(),
            f"refusing to overwrite developed branch maps {maps_path}")
    branch_map_values = {
        "first_non_bit_cells": first_diff,
        "lego_fct_nonosc": fct_activity, "nemo_evd": nemo_evd,
        "lego_evd": lego_evd,
        "nemo_tke_diffusivity_floor": nemo_kh_floor,
        "lego_tke_diffusivity_floor": lego_kh_floor,
        "lego_tke_viscosity_floor": lego_km_floor,
        "lego_tke_energy_floor": lego_energy_floor,
    }
    if fct_bundle is not None:
        coef_start = 13
        from legoesm.ocean.advection import NEMO_FCT_TRACE_FIELDS
        one_bits = np.float64(1.0).view(np.uint64)
        for tracer in ("T", "S"):
            observed = fct_modes["production_step_jit"][tracer]
            for face in ("u", "v", "w"):
                index = coef_start + NEMO_FCT_TRACE_FIELDS.index(
                    f"coef_{face}")
                got = np.asarray(observed[index])
                want = np.asarray(
                    fct_bundle["tracers"][tracer][f"coef_{face}"])
                branch_map_values[f"lego_{tracer}_coef_{face}_active"] = (
                    got.view(np.uint64) != one_bits)
                branch_map_values[f"nemo_{tracer}_coef_{face}_active"] = (
                    want.view(np.uint64) != one_bits)
    np.savez_compressed(maps_path, **branch_map_values)

    predictions = {
        "entry_T_bit_exact": entry_t_unequal == 0,
        "geometry_bit_exact": boundary_rows["geometry"]["bit_exact"],
        "first_non_bit_is_advection": first_non_bit == "advection",
        "fct_active": bool(np.any(fct_activity)),
        "fct_overlaps_first_non_bit": bool(np.any(
            fct_activity & first_diff)),
        "evd_active": bool(np.any(nemo_evd) or np.any(lego_evd)),
        "tke_floor_active": bool(
            np.any(nemo_kh_floor) or np.any(lego_floor_cells)),
        "observer_bit_exact": observer_unequal_bytes == 0,
    }
    if fct_walk is not None:
        production_fct = fct_walk["modes"]["production_step_jit"]
        coef_rows = [
            production_fct["tracers"][tracer]["rows"][f"coef_{face}"]
            for tracer in ("T", "S") for face in ("u", "v", "w")]
        first_context = fct_walk["first_non_bit_context"]
        first_statement = fct_walk["first_non_bit_statement"]
        predictions.update({
            "recorded_limiters_active_T_and_S": all(
                fct_bundle["admission"]["limiter_active_coefficients"][tracer]
                > 0 for tracer in ("T", "S")),
            "first_FCT_non_bit_no_later_than_first_horizontal_faces": (
                first_context != "NONE"
                or first_statement in ("T.first_u", "T.first_v",
                                       "S.first_u", "S.first_v")),
            "developed_active_limiter_row_differs": any(
                not row["bit_exact"] for row in coef_rows),
            "FCT_observer_bit_exact": fct_observer_unequal_bytes == 0,
        })
    if transport_walk is not None:
        production_transport_rows = transport_walk["modes"][
            "production_step_jit"]["rows"]
        predictions.update({
            "transport_reference_geometry_bit_exact": all(
                production_transport_rows[name]["bit_exact"]
                for name in ("e2u", "e3u_0", "umask", "r1_hu_0")),
            "transport_first_non_bit_is_un_adv": (
                transport_walk["first_non_bit_row"] == "un_adv"),
            "transport_later_Kmm_state_non_bit": any(
                not production_transport_rows[name]["bit_exact"]
                for name in ("one_plus_r3u_Kmm", "uu_b_Kmm", "uu_Kmm")),
            "transport_observer_bit_exact": (
                transport_observer_unequal_bytes == 0),
            "transport_oracle_operands_rebuild_zFu_bit": (
                transport_walk["operand_attribution"]["arms"]["all"][
                    "bit_exact"]),
            "transport_uu_Kmm_removes_over_90_percent": (
                transport_walk["operand_attribution"]["arms"]["uu_Kmm"][
                    "active_rms_removed_fraction"] > 0.90),
            "transport_un_adv_removes_under_1_percent": (
                abs(transport_walk["operand_attribution"]["arms"]["un_adv"][
                    "active_rms_removed_fraction"]) < 0.01),
            "transport_magnitude_owner_is_uu_Kmm": (
                transport_walk["operand_attribution"]["magnitude_owner"]
                == "uu_Kmm"),
        })
        split_rows = transport_walk["stage2_velocity_split"]["rows"]
        split_known = transport_walk["stage2_velocity_split"][
            "external_half_known_answer"]
        predictions.update({
            "stage2_split_calibration_below_1e_15": (
                split_rows["calibration_nemo_reprojection"]["active_max_abs"]
                < 1.0e-15),
            "stage2_split_baseline_reproduces_round155": (
                split_rows["production_baseline"]["active_max_abs"]
                == 1.30926020461275e-6
                and split_rows["production_baseline"]["active_cells_unequal"]
                == 17400),
            "stage2_external_half_removes_under_1_percent": (
                abs(split_rows["nemo_depth_mean_substituted"][
                    "active_rms_removed_fraction"]) < 0.01),
            "stage2_removed_matches_recorded_uu_b": (
                split_known["removed_minus_recorded_active_max_abs"]
                < split_known["recorded_uu_b_active_max_abs"] * 1.0e-3),
        })
    report = {
        "format": "gyre-developed-state-process-walk-v1",
        "status": "PASS", "case": CASE,
        "execution": "LatLonCGridOceanModel.step -> self._step_jitted",
        "entry_step": DEVELOPED_ENTRY_STEP,
        "process_step": DEVELOPED_PROCESS_STEP,
        "availability": developed_record_availability(),
        "record_contract": {
            "process_fields": [
                "Tbb", "r3t_Kbb", "r3t_Kmm", "r3t_Kaa",
                "rhs_after_advection", "rhs_after_surface_boundary",
                "rhs_after_shortwave", "rhs_after_lateral_diffusion", "Taa"],
            "not_recorded": [
                "salinity", "ttrd_xad", "ttrd_yad", "ttrd_zad",
                "nonosc coefficients", "day30", "day90", "step1441"],
        },
        "first_non_bit_boundary": (
            first_non_bit if entry_t_unequal == 0 else "ENTRY_T_MISMATCH"),
        "first_non_bit_process_call": first_non_bit_process_call,
        "first_directly_scored_active_statement": (
            "CALL tra_adv -> active CALL tra_adv_fct"
            if first_non_bit_process_call == "advection" else
            first_non_bit_process_call),
        "internal_statement_owner": (
            "UNMEASURED: the developed record stores no FCT faces, "
            "coefficients, or NEMO limiter activity"),
        "claim": (
            "first recorded non-bit compiled call is CALL tra_adv"
            if entry_t_unequal == 0 and first_non_bit == "advection" else
            "no internal statement named"),
        "entry": {
            "restart": str(restart_path),
            "restart_sha256": payload["sha256"],
            "daily_audit": str(daily_audit),
            "daily_audit_sha256": _sha256(daily_audit),
            "T_vs_process_Tbb_cells_unequal": entry_t_unequal,
            "source_mapping_cells_unequal": source_checks,
            "state_inventory": inventory,
            "ssha_scratch_source": "restart ssha via private exact-entry override",
        },
        "observer_state_unequal_bytes": observer_unequal_bytes,
        "branch_observer_cells_unequal": branch_observer_unequal,
        "cumulative_boundaries": boundary_rows,
        "geometry_operands": geometry_operands,
        "increment_rows": increment_rows,
        "process_ranking": process_ranking,
        "vertical_chain": {
            "compiled_order": vertical_order,
            "rows": vertical_rows,
            "first_non_bit": first_vertical_non_bit,
            "first_observed_statement": (
                "zdfphy.f90:352-354 avt=avt_k"
                if first_vertical_non_bit == "heat_K" else
                first_vertical_non_bit),
            "upstream_TKE_operands_recorded": False,
            "unrecorded_upstream_operands": [
                "en", "sh2", "rn2", "rn2b", "avm_k", "avt_k",
                "zdf_tke sweep intermediates"],
        },
        "branches": branches,
        "branch_maps": {"path": str(maps_path),
                        "sha256": _sha256(maps_path)},
        "predictions": predictions,
        "all_frozen_predictions_confirmed": all(predictions.values()),
        "admissions": {
            "process_record_count": process_admission["layout"]["record_count"],
            "vertical_record_count": vertical_admission["layout"]["record_count"],
            "daily_record_count": audit["inventory"]["observed_count"],
        },
        "scope": {
            "production_physics_changed": False,
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC",
        },
        "worktree": stamp,
    }
    if fct_walk is not None:
        report["format"] = "gyre-developed-state-fct-walk-v1"
        report["fct_walk"] = fct_walk
        report["record_contract"]["FCT_record_fields"] = (
            fct_walk["record_fields"])
        report["record_contract"]["not_recorded"].remove(
            "nonosc coefficients")
        first_context = fct_walk["first_non_bit_context"]
        first_statement = fct_walk["first_non_bit_statement"]
        if first_context != "NONE":
            owner = f"INHERITED_CONTEXT:{first_context}"
            claim = (
                "the first developed FCT mismatch is inherited at caller "
                f"context row {first_context}; no downstream FCT statement "
                "is an admissible owner")
        elif first_statement != "NONE":
            owner = first_statement
            claim = (
                "the first developed production-JIT FCT non-bit statement is "
                f"{first_statement}")
        else:
            owner = "NONE"
            claim = "all directly recorded developed FCT rows are bit exact"
        report["internal_statement_owner"] = owner
        report["claim"] = claim
        report["first_directly_scored_active_statement"] = first_statement
        report["admissions"]["FCT_record_fields"] = fct_walk["field_count"]
    if transport_walk is not None:
        report["format"] = "gyre-developed-stage3-transport-walk-v1"
        report["transport_walk"] = transport_walk
        report["record_contract"]["transport_record_fields"] = (
            transport_walk["record_fields"])
        first_transport = transport_walk["first_non_bit_row"]
        if first_transport in (
                "un_adv", "r1_hu_0", "one_plus_r3u_Kmm",
                "live_inverse_depth", "uu_b_Kmm", "e2u", "e3u_0",
                "umask", "live_e3u_Kmm", "uu_Kmm"):
            owner = f"INHERITED_CONTEXT:{first_transport}"
            claim = (
                "the first developed stage-3 U-transport mismatch is "
                f"inherited at {first_transport}; no downstream transport "
                "statement is an admissible owner")
        elif first_transport == "zub":
            owner = "stprk3_stg.f90:303-304/308-309 barotropic correction"
            claim = "the written stage-3 barotropic correction is first non-bit"
        elif first_transport == "corrected_u":
            owner = "stprk3_stg.f90:314 corrected velocity add"
            claim = "the stage-3 corrected U velocity is first non-bit"
        elif first_transport == "zFu":
            owner = "stprk3_stg.f90:314 metric transport product"
            claim = "the completed stage-3 U transport is first non-bit"
        else:
            owner = "NONE"
            claim = "all registered developed stage-3 U transport rows are bit exact"
        report["internal_statement_owner"] = owner
        report["claim"] = claim
        report["first_directly_scored_active_statement"] = first_transport
        report["admissions"]["transport_record_fields"] = (
            transport_walk["record_field_count"])
    _validate_developed_registry(report)
    print("\nDEVELOPED-STATE STEP 1081 -- cumulative temperature boundaries")
    print(f"  {'boundary':>22s} {'unequal':>10s} {'max abs K':>16s} "
          f"{'rms K':>16s}")
    for name in PROCESS_ROWS:
        row = boundary_rows[name]
        print(f"  {name:>22s} {row['cells_unequal']:10d} "
              f"{row['max_abs']:16.8e} {row['rms']:16.8e}")
    print("  geometry operands (2-D wet columns):")
    for name, row in geometry_operands.items():
        print(f"  {name:>22s} {row['cells_unequal']:10d} "
              f"{row['max_abs']:16.8e} {row['rms']:16.8e}")
    print(f"  FIRST NON-BIT: {report['first_non_bit_boundary']}")
    print("  FIRST NON-BIT PROCESS CALL: "
          f"{report['first_non_bit_process_call']}")
    print("  isolated one-step contribution ranking (largest RMS first):")
    for row in process_ranking:
        print(f"  {row['rank']:2d} {row['name']:>20s} "
              f"{row['rms_temperature_contribution_K']:16.8e} K  "
              f"{row['rms_effective_tendency_K_s']:16.8e} K/s")
    print("\nDEVELOPED TRACER ZDF -- compiled dataflow walk")
    print(f"  {'row':>20s} {'unequal':>10s} {'max abs':>16s} {'bit':>6s}")
    for name in vertical_order:
        row = vertical_rows[name]
        print(f"  {name:>20s} {row['cells_unequal']:10d} "
              f"{row['max_abs']:16.8e} "
              f"{'BIT' if row['bit_exact'] else 'DEBT':>6s}")
    print(f"  FIRST NON-BIT VERTICAL ROW: {first_vertical_non_bit}")
    if fct_walk is not None:
        print("\nDEVELOPED FCT -- 61-field compiled-order walk")
        print(f"  {'mode':>24s} {'exact rows':>12s} "
              f"{'first context':>28s} {'first statement':>24s}")
        for mode_name, mode in fct_walk["modes"].items():
            print(f"  {mode_name:>24s} "
                  f"{mode['bit_exact_rows']:5d}/{mode['rows_scored']:<5d} "
                  f"{mode['first_non_bit_context']:>28s} "
                  f"{mode['first_non_bit_statement']:>24s}")
        print(f"  AUTHORITATIVE OWNER: {report['internal_statement_owner']}")
    if transport_walk is not None:
        print("\nDEVELOPED STAGE-3 U TRANSPORT -- compiled-order walk")
        print(f"  {'row':>24s} {'unequal':>10s} {'max abs':>16s} {'bit':>6s}")
        rows = transport_walk["modes"]["production_step_jit"]["rows"]
        for name in DEVELOPED_TRANSPORT_U_ROWS:
            row = rows[name]
            print(f"  {name:>24s} {row['cells_unequal']:10d} "
                  f"{row['max_abs']:16.8e} "
                  f"{'BIT' if row['bit_exact'] else 'DEBT':>6s}")
        print(f"  FIRST NON-BIT: {transport_walk['first_non_bit_row']}")
        print(f"  AUTHORITATIVE OWNER: {report['internal_statement_owner']}")
        attribution = transport_walk["operand_attribution"]
        print("\n  directed operand substitution (isolated closure JIT, "
              "scored against NEMO's recorded zFu):")
        print(f"  {'arm':>24s} {'active unequal':>15s} "
              f"{'active max abs':>16s} {'max rm':>8s} "
              f"{'active rms':>16s} {'rms rm':>8s}")
        for name, row in attribution["arms"].items():
            print(f"  {name:>24s} {row['active_cells_unequal']:15d} "
                  f"{row['active_max_abs']:16.8e} "
                  f"{row['active_max_abs_removed_fraction']:8.4f} "
                  f"{row['active_rms']:16.8e} "
                  f"{row['active_rms_removed_fraction']:8.4f}")
        print(f"  MAGNITUDE OWNER: {attribution['magnitude_owner']}")
        split = transport_walk["stage2_velocity_split"]
        print("\n  stage-2 velocity split (isolated JIT of the model's own "
              "stprk3_stg.f90:753,772 statement, NEMO's operands):")
        print(f"{'arm':>46} {'active unequal':>15} {'active max abs':>16} "
              f"{'rms rm':>8}")
        for name in DEVELOPED_STAGE2_SPLIT_ROWS:
            row = split["rows"][name]
            print(f"{name:>46} {row['active_cells_unequal']:>15} "
                  f"{row['active_max_abs']:>16.8e} "
                  f"{row['active_rms_removed_fraction']:>8.4f}")
        known = split["external_half_known_answer"]
        print(f"  EXTERNAL HALF removed max "
              f"{known['removed_active_max_abs']:.8e}; recorded uu_b row "
              f"{known['recorded_uu_b_active_max_abs']:.8e}; "
              f"disagreement {known['removed_minus_recorded_active_max_abs']:.8e}")
    return report


# ---------------- Round-167 developed vertical-coefficient sensitivity ----
ROUND167_HISTORICAL_ARTIFACTS = {
    "round134_daily_tke_reset": {
        "path": Path(
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round134/"
            "daily_reset_attribution.json"),
        "sha256": (
            "0702a2bbc114a27c66eeb3b1555e731ce06d1989247d7afc27318f909e7c57cd"),
    },
    "round126_vertical_subowners": {
        "path": Path(
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round126/"
            "day240_vertical_subowners.json"),
        "sha256": (
            "b359b8274cd1178990385e10fff0e050958ed123a7e190d82dd46461307cdab8"),
    },
}


def _rank_vertical_sensitivity(final_rows: dict[str, dict]) -> list[dict]:
    """Rank registered directed arms by removed day-240 T3D RMS."""
    require("free" in final_rows,
            "vertical sensitivity ranking has no free arm")
    free = float(final_rows["free"]["T"]["rms"])
    ranking = []
    for name in ("heat_K", "effective_K"):
        require(name in final_rows,
                f"vertical sensitivity ranking has no {name} arm")
        value = float(final_rows[name]["T"]["rms"])
        ranking.append({
            "arm": name,
            "day240_T3D_rms_K": value,
            "day240_T3D_rms_removed_K": free - value,
            "removed_fraction": (free - value) / free if free else 0.0,
        })
    return sorted(ranking,
                  key=lambda row: row["day240_T3D_rms_removed_K"],
                  reverse=True)


def developed_vertical_day240_sensitivity(
        process_root: Path, vertical_root: Path, daily_root: Path,
        daily_audit: Path, expected_commit: str, evidence_root: Path, *,
        plant: str | None = None) -> dict:
    """Rank TKE heat-K versus complete-K at the developed solve boundary.

    Every arm starts from NEMO's admitted step-1080 state and runs the real
    production-jitted step through step 1440.  The only directed input is the
    private coefficient seam immediately before the implicit tracer solve.
    """
    require(plant in (None, "none", "developed-vertical-avt-ulp"),
            f"unknown developed vertical-sensitivity plant {plant!r}")
    _policy()
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            "developed vertical sensitivity requires clean tree: "
            f"{stamp['dirty_paths']}")
    require(stamp["commit"] == expected_commit,
            "developed vertical sensitivity commit differs from "
            "--expect-commit")
    process_root = Path(process_root)
    vertical_root = Path(vertical_root)
    daily_root = Path(daily_root)
    daily_audit = Path(daily_audit)
    evidence_root = Path(evidence_root)

    process_admission = validate_process_record(
        process_root, PROCESS_RECORD_COMMIT)
    vertical_admission = validate_vertical_record(
        vertical_root, DEVELOPED_VERTICAL_RECORD_COMMIT,
        process_root=process_root)
    bundle = _developed_entry_bundle(
        daily_root, daily_audit, expected_commit)
    card = bundle["card"]
    gate = bundle["gate"]
    wet = bundle["wet"]
    interface_wet = bundle["interface_wet"]
    initial_state = bundle["state"]
    payload = bundle["payload"]
    nlev = wet.shape[-1]

    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            tracer_process_trace=(), vertical_solve_trace=True))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)

    def recorded_vertical(step: int) -> tuple[dict, np.ndarray, np.ndarray]:
        path = vertical_root / (
            f"oracle_trazdf_matrix_kt{step:08d}.bin")
        record = _read_vertical_record(path, step)
        calibration = _vertical_calibration(record)
        require(all(value == 0 for value in calibration.values()),
                f"step {step}: NEMO vertical calibration is not bit exact")
        heat = _vertical_field(record, "avt")[:, :, 1:nlev]
        effective = _vertical_field(record, "zwt_mix")[:, :, 1:nlev]
        require(heat.shape == effective.shape == interface_wet.shape,
                f"step {step}: recorded vertical coefficient shape differs")
        return record, heat, effective

    def forcing(state, step: int):
        freshwater, surface = gate._surface_forcings(card, state, step)
        ssha = (jnp.asarray(payload["ssha"])
                if step == DEVELOPED_PROCESS_STEP else None)
        return freshwater, surface, ssha

    def traced_step(state, step: int, override=None):
        freshwater, surface, ssha = forcing(state, step)
        return trace_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _vertical_K_test_override=override,
            _nemo_stage1_zad_eta_after_override=ssha)

    def production_step(state, step: int, override=None):
        freshwater, surface, ssha = forcing(state, step)
        return ordinary_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface,
            _vertical_K_test_override=override,
            _nemo_stage1_zad_eta_after_override=ssha)

    first_step: dict[str, dict] = {}
    effective_failure = None

    def run_arm(name: str, *, plant_first_heat: bool = False):
        nonlocal effective_failure
        state = initial_state
        identity_mismatched_bytes = 0
        for step in range(PROCESS_START_STEP, PROCESS_END_STEP + 1):
            if name == "free":
                # Only the first free step needs internal rows.  All later
                # free steps use the ordinary production kernel directly.
                selected = (traced_step(state, step)
                            if step == PROCESS_START_STEP
                            else production_step(state, step))
                target = None
            else:
                probe = traced_step(state, step)
                record, nemo_heat, nemo_effective = recorded_vertical(step)
                vertical = _vertical_trace_frame(probe)
                viscosity = np.asarray(
                    probe.vertical_solve.viscosity_K, dtype=np.float64)
                if name == "identity":
                    target = vertical["heat_K"]
                elif name in ("heat_K", "heat_K_ulp"):
                    target = np.array(nemo_heat, copy=True)
                    if plant_first_heat and step == PROCESS_START_STEP:
                        candidates = np.argwhere(
                            interface_wet & np.isfinite(target)
                            & (target > 0.0))
                        require(candidates.size > 0,
                                "PLANT-BLIND: no positive recorded avt")
                        values = target[tuple(candidates.T)]
                        index = tuple(int(value) for value in
                                      candidates[int(np.argmax(values))])
                        old = float(target[index])
                        target[index] = np.nextafter(old, np.inf)
                        require(target[index] != old,
                                "PLANT-BLIND: recorded avt ULP was inert")
                        first_step["plant"] = {
                            "index_jik": list(index),
                            "old_uint64": int(
                                np.asarray(old).view(np.uint64)),
                            "new_uint64": int(
                                np.asarray(target[index]).view(np.uint64)),
                        }
                elif name == "effective_K":
                    # The production closure adds isoneutral K after the
                    # private heat-K seam.  This is the one-variable preimage
                    # of NEMO's complete zwt coefficient.
                    target = nemo_effective - vertical["isoneutral_K"]
                else:  # pragma: no cover - private caller registry
                    raise AssertionError(name)
                override = (jnp.asarray(target), jnp.asarray(viscosity))
                selected = (traced_step(state, step, override)
                            if step == PROCESS_START_STEP
                            else production_step(state, step, override))
                selected_state = (selected.state_after
                                  if step == PROCESS_START_STEP else selected)
                if name == "identity":
                    mismatch = _state_bit_mismatches(
                        probe.state_after, selected_state)
                    identity_mismatched_bytes += mismatch
                    require(mismatch == 0,
                            f"identity seam moved step {step} state by "
                            f"{mismatch} bytes")
                if step == PROCESS_START_STEP:
                    selected_vertical = _vertical_trace_frame(selected)
                    probe_process = _trace_frame(probe)
                    selected_process = _trace_frame(selected)
                    upstream = {
                        row: _different_cells(
                            probe_process[row], selected_process[row],
                            wet if probe_process[row].ndim == 3
                            else np.any(wet, axis=-1))
                        for row in LEGO_PROCESS_FIELDS if row != "Taa"
                    }
                    require(all(value == 0 for value in upstream.values()),
                            f"{name}: coefficient intervention moved an "
                            "upstream tracer boundary")
                    if name == "heat_K":
                        require(_different_cells(
                            selected_vertical["heat_K"], nemo_heat,
                            interface_wet) == 0,
                            "heat-K seam did not consume NEMO avt")
                    if name == "effective_K":
                        effective_row = _score_developed_row(
                            selected_vertical["effective_K"], nemo_effective,
                            interface_wet)
                    first_step[name] = {
                        "upstream_cells_moved": upstream,
                        "vertical": selected_vertical,
                        "target_heat_K": target,
                    }
                    if name == "effective_K" and not effective_row["bit_exact"]:
                        effective_failure = {
                            "prediction": "effective_K first-step complete "
                                          "coefficient is BIT",
                            "status": "REFUTED",
                            "row": effective_row,
                        }
                        return None, identity_mismatched_bytes
            if step == PROCESS_START_STEP and name == "free":
                first_step[name] = {
                    "vertical": _vertical_trace_frame(selected),
                    "process": _trace_frame(selected),
                }
            state = (selected.state_after
                     if step == PROCESS_START_STEP else selected)
            if step % 60 == 0:
                print(f"  round167 {name}: completed step {step}/1440",
                      flush=True)
        return state, identity_mismatched_bytes

    final_states = {}
    identity_bytes = 0
    arms = (["heat_K"]
            if plant == "developed-vertical-avt-ulp" else
            ["free", "identity", "heat_K", "effective_K"])
    for arm in arms:
        arm_state, arm_identity = run_arm(arm)
        identity_bytes += arm_identity
        if arm_state is None:
            break
        final_states[arm] = arm_state

    if plant == "developed-vertical-avt-ulp":
        planted_state, _ = run_arm("heat_K_ulp", plant_first_heat=True)
        baseline_matrix = first_step["heat_K"]["vertical"]
        planted_matrix = first_step["heat_K_ulp"]["vertical"]
        matrix_moved = sum(
            _different_cells(baseline_matrix[name], planted_matrix[name], wet)
            for name in ("lower", "diagonal", "upper"))
        final_t_moved = _different_cells(
            gate.lego_fields(final_states["heat_K"])["T"],
            gate.lego_fields(planted_state)["T"], wet)
        if matrix_moved == 0 or final_t_moved == 0:
            raise GateError(
                "PLANT-BLIND: one-ULP recorded avt moved "
                f"matrix={matrix_moved}, day240_T={final_t_moved}")
        raise GateError(
            "one-ULP recorded avt was caught: "
            f"index={first_step['plant']['index_jik']}, "
            f"matrix_cells={matrix_moved}, day240_T_cells={final_t_moved}")

    year = _year()
    final_restart = year._daily_restart_path(daily_root, PROCESS_END_STEP)
    final_payload = year._load_daily_reset_payload(final_restart, nlev)
    final_digest = bundle["daily_hashes"].get(final_restart.name)
    require(final_digest is not None,
            "step-1440 restart absent from admitted daily manifest")
    year._require_payload_digest(final_payload, final_digest)
    oracle_fields = {
        "T": final_payload["tn"], "S": final_payload["sn"],
        "u": final_payload["un"], "v": final_payload["vn"],
        "ssh": final_payload["sshn"],
    }
    masks = gate.expected_masks(card)
    final_rows = {}
    for arm, state in final_states.items():
        fields = gate.lego_fields(state)
        final_rows[arm] = {
            name: _score_developed_row(fields[name], oracle_fields[name],
                                       masks[name])
            for name in FIELDS
        }

    first_record = read_process_record(
        process_root
        / f"oracle_process_budget_kt{PROCESS_START_STEP:08d}.bin")
    first_vertical_record, nemo_heat, nemo_effective = recorded_vertical(
        PROCESS_START_STEP)
    nemo_process_rows = process_temperature_rows(first_record)
    lego_process_rows = lego_process_temperature_rows(
        first_step["free"]["process"])
    first_rows = {
        "vertical_diffusion": _score_developed_row(
            lego_process_rows["vertical_diffusion"],
            nemo_process_rows["vertical_diffusion"], wet),
        "free_heat_K": _score_developed_row(
            first_step["free"]["vertical"]["heat_K"], nemo_heat,
            interface_wet),
        "free_effective_K": _score_developed_row(
            first_step["free"]["vertical"]["effective_K"],
            nemo_effective, interface_wet),
        "heat_arm_effective_K": _score_developed_row(
            first_step["heat_K"]["vertical"]["effective_K"],
            nemo_effective, interface_wet),
    }
    if "effective_K" in first_step:
        first_rows["effective_arm_effective_K"] = _score_developed_row(
            first_step["effective_K"]["vertical"]["effective_K"],
            nemo_effective, interface_wet)
    require(identity_bytes == 0,
            "identity coefficient arm moved the production trajectory")

    historical = {}
    for name, item in ROUND167_HISTORICAL_ARTIFACTS.items():
        require(_sha256(item["path"]) == item["sha256"],
                f"historical artifact changed: {item['path']}")
        historical[name] = {
            "path": str(item["path"]), "sha256": item["sha256"]}
    round134 = json.loads(
        ROUND167_HISTORICAL_ARTIFACTS[
            "round134_daily_tke_reset"]["path"].read_text())
    tke_reset = next(row for row in round134["ranking"]
                     if row["family"] == "tke")
    round126 = json.loads(
        ROUND167_HISTORICAL_ARTIFACTS[
            "round126_vertical_subowners"]["path"].read_text())
    historical["round134_daily_tke_reset"].update(tke_reset)
    historical["round126_vertical_subowners"].update(round126["headline"])

    if "effective_K" in final_rows:
        ranking = _rank_vertical_sensitivity(final_rows)
    else:
        free_value = final_rows["free"]["T"]["rms"]
        heat_value = final_rows["heat_K"]["T"]["rms"]
        ranking = [{
            "arm": "heat_K",
            "day240_T3D_rms_K": heat_value,
            "day240_T3D_rms_removed_K": free_value - heat_value,
            "removed_fraction": ((free_value - heat_value) / free_value
                                 if free_value else 0.0),
            "ranking_status": "BOUNDED_ONLY_EFFECTIVE_ARM_REFUTED",
        }]
    free_rms = final_rows["free"]["T"]["rms"]
    heat_removed = next(
        row["day240_T3D_rms_removed_K"] for row in ranking
        if row["arm"] == "heat_K")
    if effective_failure is not None:
        conclusion = (
            "frozen complete-coefficient preimage prediction is REFUTED; "
            "heat-only day-240 sensitivity is bounded but the requested "
            "TKE-versus-solve ranking is incomplete")
    else:
        effective_removed = next(
            row["day240_T3D_rms_removed_K"] for row in ranking
            if row["arm"] == "effective_K")
        conclusion = (
            "TKE/closure heat diffusivity carries the ranked developed-state "
            "day-240 sensitivity"
            if heat_removed > 0.0 and heat_removed >= effective_removed else
            "TKE/closure heat diffusivity is not the ranked developed-state "
            "day-240 owner; continue at the complete implicit tracer solve")
    evidence_root.mkdir(parents=True, exist_ok=True)
    return {
        "format": "gyre-round167-developed-vertical-sensitivity-v1",
        "status": "REFUTED" if effective_failure is not None else "PASS",
        "case": CASE,
        "execution": "LatLonCGridOceanModel.step -> self._step_jitted",
        "worktree": stamp,
        "interval": {"entry_step": DEVELOPED_ENTRY_STEP,
                     "first_step": PROCESS_START_STEP,
                     "final_step": PROCESS_END_STEP,
                     "entry_day": 180, "final_day": 240},
        "admission": {
            "process": process_admission["status"],
            "vertical": vertical_admission["status"],
            "daily_audit": str(daily_audit),
            "entry_restart": str(bundle["restart_path"]),
            "entry_restart_sha256": payload["sha256"],
            "final_restart": str(final_restart),
            "final_restart_sha256": final_payload["sha256"],
        },
        "controls": {
            "identity_trajectory_mismatched_bytes": identity_bytes,
            "first_step_compiled_calibration_cells_unequal":
                _vertical_calibration(first_vertical_record),
            "first_step_rows": first_rows,
            "effective_preimage_prediction": effective_failure,
            "historical": historical,
        },
        "interventions": {
            "heat_K": "replace only pre-isoneutral heat diffusivity with "
                      "NEMO avt at every developed step",
            "effective_K": "choose pre-isoneutral heat diffusivity so the "
                           "production sum is NEMO zwt_mix bit for bit",
            "viscosity": "model value in every arm",
            "all_other_inputs": "free-running arm state and forcing",
        },
        "day240_rows": final_rows,
        "registered_row_count": sum(
            len(rows) for rows in final_rows.values()),
        "all_moved_rows_registered": set(final_rows)
            == ({"free", "identity", "heat_K"}
                if effective_failure is not None else
                {"free", "identity", "heat_K", "effective_K"}),
        "free_day240_T3D_rms_K": free_rms,
        "ranking": ranking,
        "winner": ranking[0],
        "conclusion": conclusion,
        "compiled_source": {
            "closure_to_avt": "zdftke.f90:681-712",
            "effective_coefficient": "trazdf.f90:418-444",
            "implicit_matrix": "trazdf.f90:445-480",
            "implicit_solve": "trazdf.f90:527-582",
        },
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

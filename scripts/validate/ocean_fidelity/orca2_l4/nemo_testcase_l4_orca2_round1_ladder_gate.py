#!/usr/bin/env python3
"""Round-1 ORCA2 ocean entry gate and admitted-root ladder inventory.

This gate deliberately distinguishes three claims:

* the current legoESM card versus the pinned ORCA1-ice root at kt=1;
* the admitted V2 versus ORCA1-ice NEMO-root differential through kt=10;
* whether the records contain the per-step surface inputs and full-domain
  step-entry state needed for an actual legoESM kt=1..10 trajectory.

The second claim is never presented as a legoESM trajectory.  The third claim
stops fail-closed until both MPI slabs exist for the surface and entry records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (
    REPO_ROOT,
    REPO_ROOT / "packages/core",
    REPO_ROOT / "packages/ocean",
):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp

NX, NY, NZ, NTR = 94, 152, 31, 2
HALO = 2
FIELD_ORDER = ("T", "S", "u", "v", "ssh")
EXPECTED_UNMEASURED = (
    "staged_gm_eiv",
    "linear_implicit_bottom_drag",
    "internal_wave_mixing",
    "spatial_lateral_viscosity",
    "freshwater_budget_carry",
    "si3_jpl5_layered_prather_state",
)
DEFAULT_COMPILED_SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
    "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90"
)
SOURCE_CITATION = "ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:442-459"
DEFAULT_TRAJECTORY_SOURCE_ROOT = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
    "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
)
TRAJECTORY_CITATIONS = {
    "initial_ts": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:217-254",
    "initial_ts_mask": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:307-310"),
    "initial_ssh": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465"),
    "global_index_map": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/mppini.f90:1586-1594"),
    "eos80_init": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:2284-2293"),
    "een_e3f": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:912-937"),
    "rab_polynomial": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1281-1330"),
    "stage_dump": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:215-231,329-348",
    "runoff_tracer": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:314-328",
    "salt_flux": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:290-311",
    "ldf_dyn_coefficient": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/ldfdyn.f90:348-353"),
    "qsr_rgb_chlorophyll": (
        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traqsr.f90:213,258-468"),
}


class GateError(RuntimeError):
    """A mechanically binding round-1 condition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_stamp() -> dict[str, object]:
    """Refuse a report whose code cannot be tied to a clean commit."""

    try:
        return worktree_stamp()
    except RuntimeError as exc:
        raise GateError(str(exc)) from exc


def _owned_xyz(block: np.ndarray) -> np.ndarray:
    local = block.reshape((NX, NY, NZ), order="F")
    return local[HALO:-HALO, HALO:-HALO].transpose(1, 0, 2)[..., :30]


def _owned_xy(block: np.ndarray) -> np.ndarray:
    local = block.reshape((NX, NY), order="F")
    return local[HALO:-HALO, HALO:-HALO].T


def read_state_frame(path: Path, *, kt: int, stage: int | None) -> dict[str, np.ndarray]:
    """Read a step-entry or RK-stage frame on rank-zero-owned cells."""

    require(path.is_file(), f"missing state frame: {path}")
    with path.open("rb", buffering=0) as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, f"{path.name}: truncated magic")
        magic = raw_magic.decode("ascii").rstrip()
        if stage is None:
            raw_header = handle.read(struct.calcsize("=8i"))
            require(len(raw_header) == struct.calcsize("=8i"),
                    f"{path.name}: truncated header")
            header = struct.unpack("=8i", raw_header)
            level = 1 if kt % 2 else 3
            require(magic == "NEMO_L1_ENTRY_1", f"{path.name}: bad magic {magic!r}")
            require(header == (1, kt, level, NX, NY, NZ, NTR, 64),
                    f"{path.name}: bad header {header}")
        else:
            raw_header = handle.read(struct.calcsize("=9i"))
            require(len(raw_header) == struct.calcsize("=9i"),
                    f"{path.name}: truncated header")
            header = struct.unpack("=9i", raw_header)
            level = {1: 3 if kt % 2 else 1, 2: 2, 3: 3 if kt % 2 else 1}[stage]
            require(magic == "NEMO_L1_STAGE_1", f"{path.name}: bad magic {magic!r}")
            require(header == (1, kt, stage, level, NX, NY, NZ, NTR, 64),
                    f"{path.name}: bad header {header}")
        values = np.fromfile(handle, dtype=np.float64)

    n3 = NX * NY * NZ
    n2 = NX * NY
    require(values.size == 4 * n3 + n2,
            f"{path.name}: payload {values.size} != {4 * n3 + n2}")
    require(bool(np.isfinite(values).all()), f"{path.name}: non-finite payload")
    return {
        "T": _owned_xyz(values[0:n3]),
        "S": _owned_xyz(values[n3:2 * n3]),
        "u": _owned_xyz(values[2 * n3:3 * n3]),
        "v": _owned_xyz(values[3 * n3:4 * n3]),
        "ssh": _owned_xy(values[4 * n3:]),
    }


def score(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    require(actual.shape == expected.shape,
            f"score shape mismatch: {actual.shape} != {expected.shape}")
    require(actual.dtype == expected.dtype == np.dtype(np.float64),
            f"score dtype mismatch: {actual.dtype} != {expected.dtype} != float64")
    actual_bits = np.ascontiguousarray(actual).view(np.uint64)
    expected_bits = np.ascontiguousarray(expected).view(np.uint64)
    unequal_mask = actual_bits != expected_bits
    unequal = int(np.count_nonzero(unequal_mask))
    if unequal:
        delta = np.abs(actual[unequal_mask] - expected[unequal_mask])
        first = [int(value) for value in np.argwhere(unequal_mask)[0]]
        maximum = float(np.max(delta))
        mean = float(np.mean(delta))
    else:
        first = None
        maximum = 0.0
        mean = 0.0
    residual = actual - expected
    return {
        "bit_identical": unequal == 0,
        "unequal": unequal,
        "count": int(actual.size),
        "max_abs": maximum,
        "mean_abs_over_unequal": mean,
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "first_unequal_index": first,
    }


def compare_fields(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> dict[str, object]:
    rows = {name: score(actual[name], expected[name]) for name in FIELD_ORDER}
    ranking = sorted(
        (
            {"field": name, **row}
            for name, row in rows.items()
            if not row["bit_identical"]
        ),
        key=lambda row: (-float(row["max_abs"]), FIELD_ORDER.index(str(row["field"]))),
    )
    first_non_bit = next(
        (name for name in FIELD_ORDER if not rows[name]["bit_identical"]), None
    )
    return {"rows": rows, "ranked_non_bit_by_max_abs": ranking,
            "first_non_bit_field": first_non_bit}


def card_fields(deck_root: Path) -> tuple[dict[str, np.ndarray], object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")
    require(jnp.ones(1).dtype == jnp.float64, "JAX x64 is not active")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    card = build_orca2_zps_card(deck_root)
    config = card.recipe.model_config
    require(config.tracer_time_integrator == "rk3_ws",
            "ORCA2 card no longer selects WS-RK3")
    require(config.vorticity_scheme == "een_total",
            "ORCA2 card no longer selects total-EEN vorticity")
    require(config.physics.vertical_mixing.tke.tke_langmuir_evaluation == "vectorized",
            "ORCA2 card no longer selects the vectorized Langmuir arm")
    require(tuple(card.unmeasured_features) == EXPECTED_UNMEASURED,
            "ORCA2 unmeasured_features tuple changed")
    state = card.recipe.initial_state
    fields = {
        "T": np.asarray(state.T.data)[:, :90],
        "S": np.asarray(state.S.data)[:, :90],
        "u": np.asarray(state.u.data)[:, 1:91],
        "v": np.asarray(state.v.data)[1:, :90],
        "ssh": np.asarray(state.eta.data)[:, :90],
    }
    return fields, card


_FORTRAN_REAL = re.compile(r"^[-0-9.eE+*/ ]+$")


def _fortran_real(expression: str) -> float:
    """Evaluate one Fortran real literal or literal quotient as float64."""

    cleaned = expression.replace("_wp", "").strip()
    require(bool(_FORTRAN_REAL.match(cleaned)),
            f"unexpected Fortran real expression: {expression!r}")
    return float(eval(cleaned, {"__builtins__": {}}, {}))  # noqa: S307


def extract_compiled_eos80_coefficients(path: Path) -> dict[str, float]:
    """Parse the compiled ``eos_init`` EOS-80 case into a coefficient map.

    Mechanical, not by eye: the block is delimited by its own two ``CASE``
    lines, and every simple ``NAME = <real>`` assignment inside it is taken.
    122 numbers copied by hand is exactly where a silent wrong-number defect
    enters, so the gate re-derives them from the compiled file every run.
    """

    require(path.is_file(), f"compiled EOS source missing: {path}")
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.strip().startswith("CASE( np_eos80 )")
              and "polynomial EOS-80" in line]
    require(len(starts) == 1,
            f"expected one EOS-80 eos_init case, found {len(starts)}")
    start = starts[0]
    ends = [i for i in range(start + 1, len(lines))
            if lines[i].strip().startswith("CASE( np_seos )")]
    require(bool(ends), "EOS-80 eos_init case has no following CASE")
    assignment = re.compile(r"^\s*([A-Za-z]\w*)\s*=\s*([^!]+?)\s*$")
    coefficients: dict[str, float] = {}
    for line in lines[start + 1:ends[0]]:
        match = assignment.match(line)
        if match is None:
            continue
        name, expression = match.group(1), match.group(2)
        if name[:3] not in ("EOS", "ALP", "BET") and name not in (
            "rdeltaS", "r1_S0", "r1_T0", "r1_Z0"
        ):
            continue
        require(name not in coefficients, f"duplicate EOS-80 coefficient {name}")
        coefficients[name] = _fortran_real(expression)
    return coefficients


def validate_eos80_coefficients(
    path: Path, *, plant: bool = False
) -> dict[str, object]:
    """The committed EOS-80 set must equal the compiled one, bit for bit."""

    from legoesm.ocean.eos import _ROQUET_EOS80  # noqa: PLC2701

    compiled = extract_compiled_eos80_coefficients(path)
    committed = dict(_ROQUET_EOS80)
    if plant:
        compiled["EOS000"] = np.nextafter(compiled["EOS000"], np.inf)
    require(set(compiled) == set(committed),
            "EOS-80 coefficient NAME set differs: "
            f"compiled-only={sorted(set(compiled) - set(committed))}, "
            f"committed-only={sorted(set(committed) - set(compiled))}")
    differing = sorted(
        name for name in compiled
        if np.float64(compiled[name]).tobytes()
        != np.float64(committed[name]).tobytes()
    )
    require(not differing,
            f"EOS-80 coefficients differ from the compiled source: {differing}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "citation": TRAJECTORY_CITATIONS["eos80_init"],
        "coefficients_compared": len(compiled),
        "density_terms": sum(1 for name in compiled if name.startswith("EOS")),
        "alpha_terms": sum(1 for name in compiled if name.startswith("ALP")),
        "beta_terms": sum(1 for name in compiled if name.startswith("BET")),
        "normalization": {name: compiled[name] for name in
                          ("rdeltaS", "r1_S0", "r1_T0", "r1_Z0")},
        "bit_identical": True,
    }


def validate_shared_eos_branch(path: Path) -> dict[str, object]:
    """The expansion coefficients branch once; the N-squared assembly never.

    NEMO runs ONE polynomial for TEOS-10 and EOS-80 and selects between them
    only through the coefficients eos_init loads, and its buoyancy-frequency
    routine carries no equation-of-state selection at all.  That is why the
    ORCA2 statement is a coefficient-set selection on the evaluator this repo
    already ships rather than a second formula.
    """

    lines = path.read_text().splitlines()
    rab = [i for i, line in enumerate(lines)
           if line.strip().startswith("SUBROUTINE rab_3d_t")]
    require(len(rab) == 1, "expected one rab_3d_t definition")
    end = next(i for i in range(rab[0], len(lines))
               if lines[i].strip() == "END SUBROUTINE rab_3d_t")
    body = "\n".join(lines[rab[0]:end])
    require("CASE( np_teos10, np_eos80 )" in body,
            "rab_3d_t no longer shares one case between TEOS-10 and EOS-80")
    require(body.count("CASE( np_eos80 )") == 0,
            "rab_3d_t grew a separate EOS-80 case")
    bn2 = [i for i, line in enumerate(lines)
           if line.strip().startswith("SUBROUTINE bn2_t")]
    require(len(bn2) == 1, "expected one bn2_t definition")
    bn2_end = next(i for i in range(bn2[0], len(lines))
                   if lines[i].strip() == "END SUBROUTINE bn2_t")
    bn2_body = "\n".join(lines[bn2[0]:bn2_end])
    require("SELECT CASE" not in bn2_body and "neos" not in bn2_body,
            "bn2_t grew an equation-of-state branch")
    return {
        "rab_case_shared": True,
        "bn2_has_no_eos_branch": True,
        "rab_citation": TRAJECTORY_CITATIONS["rab_polynomial"],
        "bn2_citation": (
            "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
            "/eosbn2.f90:1587-1647"),
    }


def validate_compiled_source(path: Path) -> dict[str, object]:
    require(path.is_file(), f"compiled source missing: {path}")
    lines = path.read_text().splitlines()
    require(len(lines) >= 459, f"compiled source too short: {len(lines)}")
    excerpt = lines[441:459]
    joined = "\n".join(excerpt)
    for token in (
        "snwice_mass  (:,:) = tmask(:,:,1) * SUM",
        "zsshadj = glob_2Dsum",
        "ssh(:,:,Kmm) = ssh(:,:,Kmm) - zsshadj",
        "ssh(:,:,Kbb) = ssh(:,:,Kbb) - zsshadj",
    ):
        require(token in joined, f"compiled source anchor missing: {token}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "citation": SOURCE_CITATION,
        "line_start": 442,
        "line_end": 459,
    }


def validate_surface_frame(path: Path, kt: int, rank: int) -> dict[str, object]:
    """Validate the dynamic-kt form of the frozen Phase-2b schema."""

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_phase2b_exchange_gate as surface,
    )

    with path.open("rb", buffering=0) as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack(surface.HEADER_FMT, handle.read(struct.calcsize(surface.HEADER_FMT)))
    wanted = list(surface.HEADER)
    wanted[1] = kt
    wanted[2] = 1 if kt % 2 else 3
    require(magic == surface.MAGIC, f"{path.name}: bad surface magic")
    require(header == tuple(wanted), f"{path.name}: bad surface header {header}")
    count = surface.payload_count(header)
    offset = 16 + struct.calcsize(surface.HEADER_FMT)
    require(path.stat().st_size == offset + 8 * count,
            f"{path.name}: surface schema/EOF mismatch")
    values = np.memmap(path, dtype=np.float64, mode="r", offset=offset, shape=(count,))
    require(bool(np.isfinite(values).all()), f"{path.name}: non-finite surface payload")
    return {"file": path.name, "kt": kt, "rank": rank,
            "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def read_surface_fields(path: Path, kt: int, rank: int) -> dict[str, np.ndarray]:
    """Decode one rank-local post-``sbc`` operand frame to owned arrays."""

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_phase2b_exchange_gate as surface,
    )

    validate_surface_frame(path, kt, rank)
    offset = 16 + struct.calcsize(surface.HEADER_FMT)
    payload = np.memmap(path, dtype=np.float64, mode="r", offset=offset)
    sizes = {
        "full": NX * NY,
        "reduced": (NX - 2 * HALO) * (NY - 2 * HALO),
        "halo1": (NX - 2) * (NY - 2),
        "reduced3d": NTR * (NX - 2 * HALO) * (NY - 2 * HALO),
    }
    decoded: dict[str, np.ndarray] = {}
    cursor = 0
    for name, allocation in surface.FIELDS:
        size = sizes[allocation]
        values = np.asarray(payload[cursor:cursor + size])
        cursor += size
        if allocation == "full":
            field = values.reshape((NX, NY), order="F")
            field = field[HALO:-HALO, HALO:-HALO].T
        elif allocation == "reduced":
            field = values.reshape((NX - 2 * HALO, NY - 2 * HALO), order="F").T
        elif allocation == "halo1":
            field = values.reshape((NX - 2, NY - 2), order="F")
            field = field[1:-1, 1:-1].T
        else:
            field = values.reshape(
                (NX - 2 * HALO, NY - 2 * HALO, NTR), order="F"
            ).transpose(1, 0, 2)
        require(field.shape[:2] == (NY - 2 * HALO, NX - 2 * HALO),
                f"{path.name}: bad owned shape for {name}: {field.shape}")
        decoded[name] = field
    require(cursor == payload.size,
            f"{path.name}: schema walk consumed {cursor}/{payload.size}")
    return decoded


def assemble_state_fields(root: Path, kt: int, *, stage: int | None) -> dict[str, np.ndarray]:
    """Assemble the two owned longitude slabs for an entry state."""

    require(stage is None, "full-domain stage frames were not acquired")
    slabs = []
    for rank in range(2):
        name = (
            f"oracle_step_entry_kt{kt:08d}.bin" if rank == 0 else
            f"oracle_step_entry_rank{rank:04d}_kt{kt:08d}.bin"
        )
        slabs.append(read_state_frame(root / name, kt=kt, stage=None))
    return {
        field: np.concatenate([slabs[0][field], slabs[1][field]], axis=1)
        for field in FIELD_ORDER
    }


def assemble_surface_fields(root: Path, kt: int) -> dict[str, np.ndarray]:
    slabs = []
    for rank in range(2):
        name = (
            f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
            f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin"
        )
        slabs.append(read_surface_fields(root / name, kt, rank))
    require(slabs[0].keys() == slabs[1].keys(), "surface slab field mismatch")
    return {
        field: np.concatenate([slabs[0][field], slabs[1][field]], axis=1)
        for field in slabs[0]
    }


def chlorophyll_at_step(
    deck_root: Path, kt: int, wet: np.ndarray, *, dt_s: float
) -> np.ndarray:
    """Reproduce the first-month monthly ``fld_read`` interpolation."""

    from netCDF4 import Dataset

    require(1 <= kt <= 240, "chlorophyll step must be in 1..240")
    path = deck_root / "chlorophyll.nc"
    require(path.is_file(), f"missing ORCA2 chlorophyll input: {path}")
    with Dataset(path) as dataset:
        source = np.asarray(dataset.variables["CHLA"][:], dtype=np.float64)
    require(source.shape == (12, 148, 180),
            f"unexpected chlorophyll shape {source.shape}")
    # The admitted run resolves record centres at -15.5, +15.5 and +45.0 days.
    # NEMO forms the midpoint clock and weights at fldread.f90:243-246.  The
    # January centre is crossed between kt=124 and kt=125.
    dt_i = int(dt_s)
    require(float(dt_i) == float(dt_s) and dt_i % 2 == 0,
            f"chlorophyll clock needs an even integer dt, got {dt_s}")
    midpoint_s = (2 * kt - 1) * dt_i // 2
    centres_s = (-1339200, 1339200, 3888000)
    if midpoint_s < centres_s[1]:
        before_index, after_index = 11, 0
        before_s, after_s = centres_s[:2]
    else:
        before_index, after_index = 0, 1
        before_s, after_s = centres_s[1:]
    after = np.float64(midpoint_s - before_s) / np.float64(after_s - before_s)
    before = np.float64(1.0) - after
    result = before * source[before_index] + after * source[after_index]
    return np.where(wet, result, np.float64(0.0))


def validate_round5_admission(root: Path, *, plant: bool = False) -> dict[str, object]:
    path = root / "round1_surface_admission.json"
    require(path.is_file(), f"missing acquisition admission: {path}")
    result = json.loads(path.read_text())
    require(result.get("status") == "PASS", "acquisition admission is not PASS")
    require(result.get("surface_frames_total") == 20, "admission surface total")
    require(result.get("entry_frames_total") == 20, "admission entry total")
    require(result.get("inherited_streams_passive") == 107,
            "admission inherited passivity")
    require(result.get("twin_surface_frames_raw_exact") is True,
            "admission surface twins are not exact")
    require(result.get("twin_rank1_entry_frames_raw_exact") is True,
            "admission rank-1 entry twins are not exact")
    rows = list(result.get("surface_frames", [])) + list(
        result.get("rank1_entry_frames", []))
    require(len(rows) == 30, f"admission manifest has {len(rows)} rows")
    for index, row in enumerate(rows):
        record = root / str(row["file"])
        observed = sha256(record)
        if plant and index == 0:
            observed = "0" * 64
        require(observed == row["sha256"],
                f"admission digest mismatch: {record.name}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "status": result["status"],
        "surface_frames_total": result["surface_frames_total"],
        "entry_frames_total": result["entry_frames_total"],
        "inherited_streams_passive": result["inherited_streams_passive"],
        "manifest_rows_verified": len(rows),
    }


def validate_trajectory_sources(root: Path) -> dict[str, object]:
    anchors = {
        "stprk3.f90": (
            "CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )",
            "CALL l1_dump_stage( kstp, 1, Naa )",
            "SUBROUTINE l1_dump_stage",
        ),
        "trasbc.f90": (
            "IF( ln_rnf ) THEN",
            "rnf_tsc(ji,jj,jp_tem) * zdep",
            "rnf_tsc(ji,jj,jp_sal) * zdep",
        ),
    }
    rows = {}
    for name, tokens in anchors.items():
        path = root / name
        require(path.is_file(), f"missing compiled trajectory source: {path}")
        source = path.read_text()
        for token in tokens:
            require(token in source, f"compiled {name} anchor missing: {token}")
        rows[name] = {"path": str(path), "sha256": sha256(path)}
    return {"files": rows, "citations": TRAJECTORY_CITATIONS}


def surface_support(root: Path) -> dict[str, object]:
    present: list[dict[str, object]] = []
    missing: list[str] = []
    for rank in range(2):
        for kt in range(1, 11):
            name = (
                f"oracle_ocean_surface_input_kt{kt:08d}.bin"
                if rank == 0 else
                f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin"
            )
            path = root / name
            if path.is_file():
                present.append(validate_surface_frame(path, kt, rank))
            else:
                missing.append(name)
    return {
        "required": 20,
        "present": present,
        "missing": missing,
        "trajectory_supported": not missing,
    }


def entry_support(root: Path) -> dict[str, object]:
    """Require both owned MPI slabs for every step-entry state."""

    present: list[dict[str, object]] = []
    missing: list[str] = []
    for rank in range(2):
        for kt in range(1, 11):
            name = (
                f"oracle_step_entry_kt{kt:08d}.bin"
                if rank == 0 else
                f"oracle_step_entry_rank{rank:04d}_kt{kt:08d}.bin"
            )
            path = root / name
            if path.is_file():
                fields = read_state_frame(path, kt=kt, stage=None)
                present.append({
                    "file": name,
                    "kt": kt,
                    "rank": rank,
                    "owned_shapes": {
                        field: list(values.shape) for field, values in fields.items()
                    },
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
            else:
                missing.append(name)
    return {
        "required": 20,
        "present": present,
        "missing": missing,
        "trajectory_supported": not missing,
    }


def initial_output_substitution_check(
    root: Path, entry: dict[str, np.ndarray]
) -> dict[str, object]:
    """Prove the rank-0 output.init shard is not the step-entry operand."""

    from netCDF4 import Dataset

    path = root / "output.init_0000.nc"
    require(path.is_file(), f"missing initial-output shard: {path}")
    variable = {
        "T": "votemper",
        "S": "vosaline",
        "u": "vozocrtx",
        "v": "vomecrty",
        "ssh": "sossheig",
    }
    candidate: dict[str, np.ndarray] = {}
    with Dataset(path) as dataset:
        for field, name in variable.items():
            values = np.asarray(dataset.variables[name][0], dtype=np.float64)
            if field != "ssh":
                values = np.moveaxis(values[:30], 0, -1)
            candidate[field] = values
    compared = compare_fields(candidate, entry)
    require(compared["first_non_bit_field"] is not None,
            "output.init unexpectedly became the step-entry state")
    return {
        "file": path.name,
        "sha256": sha256(path),
        "disposition": "NOT_A_STEP_ENTRY_OPERAND",
        **compared,
    }


def root_differential(v2_root: Path, orca1ice_root: Path) -> dict[str, object]:
    checkpoints: list[dict[str, object]] = []
    for kt in range(1, 11):
        for stage in (None, 1, 2, 3):
            if stage is None:
                name = f"oracle_step_entry_kt{kt:08d}.bin"
                label = "entry"
            else:
                name = f"oracle_stage_kt{kt:08d}_s{stage}.bin"
                label = f"stage{stage}"
            left_path = v2_root / name
            right_path = orca1ice_root / name
            left = read_state_frame(left_path, kt=kt, stage=stage)
            right = read_state_frame(right_path, kt=kt, stage=stage)
            compared = compare_fields(left, right)
            checkpoints.append({"kt": kt, "checkpoint": label,
                                "record": name,
                                "v2_sha256": sha256(left_path),
                                "orca1ice_sha256": sha256(right_path),
                                **compared})
    kt10 = next(
        row for row in checkpoints
        if row["kt"] == 10 and row["checkpoint"] == "entry"
    )
    ssh_max = float(kt10["rows"]["ssh"]["max_abs"])
    prediction_status = "CONFIRMED" if 0.014 <= ssh_max <= 0.017 else "REFUTED"
    return {
        "claim_label": "NEMO_ROOT_DIFFERENTIAL_NOT_LEGOESM_TRAJECTORY",
        "v2_root": str(v2_root),
        "orca1ice_root": str(orca1ice_root),
        "checkpoints": checkpoints,
        "kt10_entry_ssh_max_abs_m": ssh_max,
        "r1_p4": {
            "status": prediction_status,
            "predicted_interval_m": [0.014, 0.017],
            "observed_m": ssh_max,
        },
    }


def _candidate_fields(state) -> dict[str, np.ndarray]:
    return {
        "T": np.asarray(state.T.data)[..., :30],
        "S": np.asarray(state.S.data)[..., :30],
        "u": np.asarray(state.u.data)[:, 1:, :30],
        "v": np.asarray(state.v.data)[1:, :, :30],
        "ssh": np.asarray(state.eta.data),
    }


def _stage_candidate_fields(stage_output) -> dict[str, np.ndarray]:
    u, v, temperature, salinity, ssh = stage_output
    return {
        "T": np.asarray(temperature)[..., :30],
        "S": np.asarray(salinity)[..., :30],
        "u": np.asarray(u)[:, 1:, :30],
        "v": np.asarray(v)[1:, :, :30],
        "ssh": np.asarray(ssh),
    }


def _rank0_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values[:, :90] for name, values in fields.items()}


def _surface_forcings(
    card, deck_root: Path, fields: dict[str, np.ndarray], kt: int
):
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    shape = fields["emp"].shape
    require(shape == (148, 180), f"assembled surface shape is {shape}")
    zeros = np.zeros(shape, dtype=np.float64)
    native_i = fields["utau"]
    native_j = fields["vtau"]
    cos_alpha = np.asarray(card.recipe.grid.cos_alpha_u)[:, 1:]
    sin_alpha = np.asarray(card.recipe.grid.sin_alpha_u)[:, 1:]
    require(cos_alpha.shape == sin_alpha.shape == shape,
            "ORCA2 stress-rotation shape mismatch")
    # Inverse of the production atmospheric east/north -> NEMO-native rotation.
    # The native pair is also supplied directly, so production never performs
    # this lossy round trip; geographic fields keep the forcing API complete.
    tau_x = -(cos_alpha * native_i - sin_alpha * native_j)
    tau_y = -(sin_alpha * native_i + cos_alpha * native_j)
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    chl = chlorophyll_at_step(deck_root, kt, wet, dt_s=card.dt_s)
    q_total = np.asarray(nemo_source_round(
        jnp.asarray(fields["qns"]) + jnp.asarray(fields["qsr"])
    ))
    freshwater = FreshwaterForcing(
        precip=jnp.asarray(zeros),
        evap=jnp.asarray(fields["emp"]),
        # The runoff's WATER, exactly as recorded.  NEMO applies it in two
        # places and legoESM reaches both from this one field: the barotropic
        # sea-surface forcing ``r1_rho0*(emp - rnf)`` (stp2d.f90:278-281),
        # which is legoESM's ``precip - evap + runoff + ice_fw``, and the
        # horizontal divergence (sbcrnf.f90:279-283 at divhor.f90:142), which
        # production reaches through ``runoff_mass_flux``.  It is deliberately
        # NOT in the tracer dilution term: NEMO's emp excludes it
        # (trasbc.f90:282-288).
        runoff=jnp.asarray(fields["rnf"]),
        ice_fw=jnp.asarray(zeros),
        restoring=jnp.asarray(zeros),
    )
    surface = OceanSurfaceForcing(
        sw_down=jnp.asarray(fields["qsr"]),
        q_net=jnp.asarray(q_total),
        tau_x=jnp.asarray(tau_x),
        tau_y=jnp.asarray(tau_y),
        salt_flux=jnp.asarray(np.float64(1.0e-3) * fields["sfx"]),
        chl=jnp.asarray(chl),
        taum=jnp.asarray(fields["taum"]),
        ice_concentration=jnp.asarray(fields["fr_i"]),
        tau_i_native=jnp.asarray(native_i),
        tau_j_native=jnp.asarray(native_j),
        # The runoff's TRACER content, recorded exactly by the acquisition
        # (surface-frame fields ``rnf_tsc``): NEMO's river-runoff tracer
        # source.  Its MASS channel is NOT switched on here -- that is a
        # separate statement and would move the sea surface.
        runoff_tracer_content=(
            jnp.asarray(fields["rnf_tsc"][..., 0]),
            jnp.asarray(fields["rnf_tsc"][..., 1]),
        ),
    )
    return freshwater, surface


# The unbuilt statements this ladder is allowed to stop on, each keyed by the
# EXACT refusal text the production code prints.  Matching that text is the
# point: without it a refusal from a different routine would be labelled as one
# of these and cited to the wrong compiled line.  Round 7's entry was the EEN
# vertex thickness on a tripolar fold row; round 8 transcribed it, and the walk
# then reached the lateral-viscosity coefficient.
UNBUILT_STATEMENTS = {
    "STOP_PRODUCTION_QSR_RGB_PIPELINE_GAP": {
        "refusal": (
            "shortwave_penetration_tendency is the two-band Jerlov kernel but "
            "got scheme='nemo_qsr_rgb'"),
        "legoesm_source": (
            "packages/ocean/legoesm/ocean/physics/combined.py:336-355"),
        "citation": "qsr_rgb_chlorophyll",
        "resolved_setting": (
            "the record resolves ln_qsr_rgb = .true. with nn_chldta = 1 "
            "(run ocean.output:1207,1211), so NEMO dispatches qsr_RGBc, the "
            "three-band chlorophyll attenuation with the Morel-Berthon "
            "vertical profile; the shared ocean physics pipeline has only the "
            "two-band Jerlov kernel and carries neither the chlorophyll field "
            "nor the live thickness the RGB kernel reads"),
    },
}


def _attribute_stage_row(stage, field, candidate, oracle, surface_fields):
    """Name the statement that owns a stage row -- by MEASUREMENT, not a map.

    Round 12 REFUTED the hard-coded rule this replaced.  It bound every
    stage-1 temperature or salinity row to NEMO's river-runoff tracer source,
    and that statement can only reach a column where the runoff is non-zero
    and only the top cell of it: with the runoff channel transcribed and
    deposited bit-exactly (round-12 runoff gate, 0 of 799,200 unequal at every
    stage), 191,282 of the 233,341 disagreeing cells still lie off EVERY
    runoff column and the count does not move.  A citation that survives its
    own statement being satisfied is not an attribution.

    So the runoff is named only when the disagreement is confined to where it
    can act; otherwise the row is UNATTRIBUTED and says what was ruled out.
    """
    if stage != 1 or field not in ("T", "S"):
        return {"source_citation": TRAJECTORY_CITATIONS["stage_dump"]}
    unequal = np.asarray(candidate[field]) != np.asarray(oracle[field])
    runoff = np.asarray(surface_fields["rnf"])[:unequal.shape[0],
                                               :unequal.shape[1]] != 0.0
    reach = np.zeros_like(unequal)
    reach[..., 0] = runoff                      # nk_rnf = 1, the top cell
    off_reach = int((unequal & ~reach).sum())
    if off_reach == 0:
        return {"source_citation": TRAJECTORY_CITATIONS["runoff_tracer"]}
    return {
        "source_citation": "UNATTRIBUTED",
        "ruled_out": {
            "statement": TRAJECTORY_CITATIONS["runoff_tracer"],
            "why": ("the river-runoff tracer source reaches only the top cell "
                    "of a column carrying a non-zero runoff, and the "
                    "disagreement is not confined there"),
            "unequal_cells": int(unequal.sum()),
            "unequal_cells_the_runoff_cannot_reach": off_reach,
        },
    }


def unbuilt_statement_blocker(exc: Exception, kt: int) -> dict[str, object]:
    """Classify a deliberate "not built" refusal, or refuse to label it."""

    message = str(exc)
    matches = [name for name, entry in UNBUILT_STATEMENTS.items()
               if entry["refusal"] in message]
    require(len(matches) == 1,
            "the production step raised an unregistered refusal; it is not a "
            "known unbuilt ORCA2 statement and must not be labelled as one: "
            f"{message}")
    entry = UNBUILT_STATEMENTS[matches[0]]
    return {
        "status": matches[0],
        "kt": kt,
        "message": message,
        "legoesm_source": entry["legoesm_source"],
        "nemo_source_citation": TRAJECTORY_CITATIONS[entry["citation"]],
        "resolved_setting": entry["resolved_setting"],
    }


def entry_eligibility(bridge: dict[str, object]) -> tuple[bool, str]:
    """Whether the card's OWN entry state is exact, and what that certifies.

    Only sea-surface height may come from NEMO (Decision 52).  If any other
    field still needs the record, the twin is NOT eligible and says so.
    """

    rows = bridge["rows"]
    independent = all(
        rows[name]["bit_identical"] for name in ("T", "S", "u", "v"))
    return independent, (
        "DECISION52_SSH_ONLY" if independent
        else "STOP_INITIAL_T_S_TRANSCRIPTION")


def unaltered_initial_ts(deck_root: Path, card) -> dict[str, np.ndarray]:
    """ABLATION: the initial state WITHOUT the ORCA_R2 hand alterations.

    This is the pre-round-7 card, rebuilt through the same helper the card
    uses, so the gate can show the alterations own the whole entry residual
    rather than merely shrinking it.  Rule 4: it ablates the transcription,
    not a proxy for it.
    """

    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_initial_ts

    tmask = np.asarray(card.recipe.z_coord.is_active)
    temperature, salinity = build_orca2_initial_ts(
        deck_root / "data_1m_potential_temperature_nomask.nc",
        deck_root / "data_1m_salinity_nomask.nc",
        tmask,
        apply_hand_alterations=False,
    )
    return {"T": temperature, "S": salinity}


def candidate_trajectory(
    deck_root: Path,
    root: Path,
    card,
    *,
    max_step: int = 10,
    bridge_ssh: bool = True,
) -> dict[str, object]:
    """Run production ORCA2 with either its own or Decision-52 entry SSH."""

    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    require(1 <= max_step <= 10, "candidate max_step must be in 1..10")
    entry1 = assemble_state_fields(root, 1, stage=None)
    state = card.recipe.initial_state
    independent = compare_fields(_candidate_fields(state), entry1)
    unaltered = unaltered_initial_ts(deck_root, card)
    ablation = {
        name: score(unaltered[name], entry1[name]) for name in ("T", "S")
    }
    require(not ablation["T"]["bit_identical"] and
            not ablation["S"]["bit_identical"],
            "the hand-alteration ablation is vacuous: removing the compiled "
            "ORCA_R2 alterations changed nothing")
    # Decision 52 authorizes exactly this operand replacement and nothing else.
    # Independent mode deliberately skips it and executes the card's own SSH.
    if bridge_ssh:
        state = state._replace(
            eta=state.eta.replace(data=jnp.asarray(entry1["ssh"], dtype=jnp.float64))
        )
    executed_entry = compare_fields(_candidate_fields(state), entry1)
    claim_label = (
        "INDEPENDENT_WITH_DECISION52_SSH" if bridge_ssh else "INDEPENDENT"
    )

    # The kt=1 chlorophyll frame is an independent emitted witness for the
    # input-file interpolation used at all ten steps.
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_rgb_gate as rgb_gate,
    )

    emitted_chl = np.asarray(rgb_gate.read_rgb(
        root / "oracle_rgb_chl_kt00000001.bin"
    )["chl"])
    wet = np.asarray(card.recipe.initial_state.land_mask.data)[:, :90] > 0.5
    reconstructed_chl = chlorophyll_at_step(
        deck_root, 1,
        np.asarray(card.recipe.initial_state.land_mask.data) > 0.5,
        dt_s=card.dt_s,
    )
    chl_row = score(reconstructed_chl[:, :90], emitted_chl)
    require(chl_row["bit_identical"], "kt1 chlorophyll interpolation is non-bit")

    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True),
    )
    checkpoints: list[dict[str, object]] = []
    first_non_bit: dict[str, object] | None = None
    first_field: str | None = None
    for kt in range(1, max_step + 1):
        oracle_entry = assemble_state_fields(root, kt, stage=None)
        entry_comparison = compare_fields(_candidate_fields(state), oracle_entry)
        entry_row = {"kt": kt, "checkpoint": "entry", **entry_comparison}
        checkpoints.append(entry_row)
        if first_non_bit is None and entry_comparison["first_non_bit_field"] is not None:
            first_field = str(entry_comparison["first_non_bit_field"])
            first_non_bit = {
                "kt": kt,
                "checkpoint": "entry",
                "field": first_field,
                "source_citation": TRAJECTORY_CITATIONS[
                    "initial_ssh" if first_field == "ssh" else "initial_ts"
                ],
            }

        surface_fields = assemble_surface_fields(root, kt)
        freshwater, surface = _surface_forcings(card, deck_root, surface_fields, kt)
        try:
            trace = model.step(
                state,
                dt=card.dt_s,
                freshwater=freshwater,
                surface_forcing=surface,
            )
        except (NotImplementedError, ValueError) as exc:
            # An explicitly UNBUILT statement, named by the production code.
            # It is recorded, not swallowed: the gate still exits non-zero and
            # registers no magnitude.  A refusal from any OTHER routine is not
            # labelled as this one, and anything that is not a deliberate
            # refusal (a ValueError, a shape error) is a defect and propagates.
            blocker = unbuilt_statement_blocker(exc, kt)
            entry_exact, eligibility = entry_eligibility(executed_entry)
            return {
                "claim_label": claim_label,
                "initial_mode": (
                    "decision52_ssh_bridge" if bridge_ssh else "card_own_state"
                ),
                "execution": "production-jit-cpu-fp64-x64-libm",
                "executed_initial_state_vs_nemo": executed_entry,
                "decision52_bridge": executed_entry if bridge_ssh else None,
                "independent_entry_before_bridge": independent,
                "hand_alteration_ablation": ablation,
                "independent_entry_ts_bit_identical": entry_exact,
                "given_nemo_entry_eligibility": eligibility,
                "chlorophyll_kt1_input_reconstruction": chl_row,
                "first_non_bit_statement": first_non_bit,
                "kt10_same_field_magnitude":
                    f"UNMEASURED_{blocker['status']}",
                "checkpoints": checkpoints,
                "execution_blocker": blocker,
            }
        for stage in (1, 2, 3):
            oracle_stage = read_state_frame(
                root / f"oracle_stage_kt{kt:08d}_s{stage}.bin",
                kt=kt,
                stage=stage,
            )
            candidate_stage = _rank0_fields(
                _stage_candidate_fields(trace.stage_outputs[stage - 1])
            )
            comparison = compare_fields(candidate_stage, oracle_stage)
            checkpoint = {"kt": kt, "checkpoint": f"stage{stage}", **comparison}
            checkpoints.append(checkpoint)
            if first_non_bit is None and comparison["first_non_bit_field"] is not None:
                first_field = str(comparison["first_non_bit_field"])
                first_non_bit = {
                    "kt": kt,
                    "checkpoint": f"stage{stage}",
                    "field": first_field,
                    **_attribute_stage_row(
                        stage, first_field, candidate_stage, oracle_stage,
                        surface_fields),
                }
        state = trace.state_after

    require(first_non_bit is not None, "candidate trajectory unexpectedly stayed bit-exact")
    require(first_field is not None, "first non-bit field was not recorded")
    kt10_rows = {
        row["checkpoint"]: row["rows"][first_field]
        for row in checkpoints if row["kt"] == max_step
    }
    entry_ts_independent, eligibility = entry_eligibility(executed_entry)
    return {
        "claim_label": claim_label,
        "initial_mode": (
            "decision52_ssh_bridge" if bridge_ssh else "card_own_state"
        ),
        "execution": "production-jit-cpu-fp64-x64-libm",
        "executed_initial_state_vs_nemo": executed_entry,
        "decision52_bridge": executed_entry if bridge_ssh else None,
        "independent_entry_before_bridge": independent,
        "hand_alteration_ablation": ablation,
        "independent_entry_ts_bit_identical": entry_ts_independent,
        "given_nemo_entry_eligibility": eligibility,
        "chlorophyll_kt1_input_reconstruction": chl_row,
        "first_non_bit_statement": first_non_bit,
        "kt10_same_field_magnitude": kt10_rows,
        "checkpoints": checkpoints,
        "unsupported_recorded_channels": {
            "rnf_tsc": (
                "TRANSCRIBED in round 12: the recorded runoff tracer content "
                "is now the production step's runoff source, deposited at all "
                "three Runge-Kutta stages"
            ),
            "rnf": (
                "the runoff MASS channel is recorded but NOT switched on; it "
                "forces the sea surface and the horizontal divergence and is "
                "a separate statement"
            ),
            "rnf_tsc_b": "recorded carry; no production state field",
            "freshwater_budget_carry": "card registry remains UNMEASURED",
        },
        "resolved_surface_mapping": {
            "stress": "utau/vtau native T-grid pair",
            "heat": "q_net=nemo_source_round(qns+qsr); sw_down=qsr",
            "freshwater": "net=-emp through FreshwaterForcing.evap",
            "salt": "salt_flux=1e-3*sfx (NEMO record is g m-2 s-1)",
            "tke": "taum and fr_i supplied",
            "chlorophyll": "yearly December/January fld_read interpolation",
        },
    }


def run_gate(
    deck_root: Path,
    v2_root: Path,
    orca1ice_root: Path,
    compiled_source: Path,
    trajectory_source_root: Path,
    *,
    max_step: int = 10,
    initial_mode: str = "decision52-bridge",
    plant: str | None = None,
) -> dict[str, object]:
    stamp = provenance_stamp()
    source = validate_compiled_source(compiled_source)
    candidate, card = card_fields(deck_root)
    if plant == "kt1_T":
        candidate["T"] = candidate["T"].copy()
        candidate["T"].flat[0] = np.nextafter(candidate["T"].flat[0], np.inf)

    v2_entry = read_state_frame(
        v2_root / "oracle_step_entry_kt00000001.bin", kt=1, stage=None
    )
    card_vs_v2 = compare_fields(candidate, v2_entry)
    require(card_vs_v2["first_non_bit_field"] is None,
            "planted kt1 T identity control fired" if plant else
            f"current card no longer matches V2 at kt=1: {card_vs_v2['first_non_bit_field']}")

    orca1ice_entry = read_state_frame(
        orca1ice_root / "oracle_step_entry_kt00000001.bin", kt=1, stage=None
    )
    card_vs_orca1ice = compare_fields(candidate, orca1ice_entry)
    require(card_vs_orca1ice["first_non_bit_field"] == "ssh",
            "REFUTED R1-P2: first non-bit kt=1 field is not ssh")
    for name in ("T", "S", "u", "v"):
        require(card_vs_orca1ice["rows"][name]["bit_identical"],
                f"REFUTED R1-P2: kt=1 {name} is non-bit")
    ssh_max = float(card_vs_orca1ice["rows"]["ssh"]["max_abs"])
    require(0.015 <= ssh_max <= 0.016,
            f"REFUTED R1-P2: kt=1 ssh max_abs={ssh_max:.17g}")

    differential = root_differential(v2_root, orca1ice_root)
    support = surface_support(orca1ice_root)
    full_entry = entry_support(orca1ice_root)
    output_init = initial_output_substitution_check(orca1ice_root, orca1ice_entry)
    admission = None
    trajectory_sources = None
    eos_branch = None
    eos_coefficients = None
    trajectory = None
    if not support["trajectory_supported"]:
        status = "STOP_RECORD_GAP"
        trajectory_claim = "UNMEASURED_RECORD_GAP"
    elif not full_entry["trajectory_supported"]:
        status = "STOP_ENTRY_RECORD_GAP"
        trajectory_claim = "UNMEASURED_ENTRY_RECORD_GAP"
    else:
        admission = validate_round5_admission(
            orca1ice_root, plant=plant == "surface_hash"
        )
        trajectory_sources = validate_trajectory_sources(trajectory_source_root)
        eos_branch = validate_shared_eos_branch(
            trajectory_source_root / "eosbn2.f90")
        eos_coefficients = validate_eos80_coefficients(
            trajectory_source_root / "eosbn2.f90",
            plant=plant == "eos80_coeff",
        )
        require(initial_mode in ("decision52-bridge", "independent"),
                f"unknown initial mode: {initial_mode}")
        trajectory = candidate_trajectory(
            deck_root,
            orca1ice_root,
            card,
            max_step=max_step,
            bridge_ssh=initial_mode == "decision52-bridge",
        )
        status = trajectory.get("execution_blocker", {}).get(
            "status",
            "LADDER_MEASURED"
            if trajectory["independent_entry_ts_bit_identical"] else
            "STOP_INITIAL_TS_TRANSCRIPTION_GAP",
        )
        trajectory_claim = (
            "MEASURED_INDEPENDENT_WITH_DECISION52_SSH"
            if initial_mode == "decision52-bridge"
            else "MEASURED_INDEPENDENT_CARD_OWN_STATE"
        )
    return {
        "worktree": stamp,
        "status": status,
        "card": card.case,
        "whole_step_identity": "orca2_vector_een_c2",
        "unmeasured_features": list(card.unmeasured_features),
        "compiled_source": source,
        "kt1_card_vs_v2": card_vs_v2,
        "kt1_card_vs_pinned_orca1ice": card_vs_orca1ice,
        "independent_first_non_bit_statement": {
            "claim_label": "INDEPENDENT",
            "field": "ssh",
            "source_citation": SOURCE_CITATION,
            "ownership": "INITIAL_SI3_CATEGORY_LOAD_CONFIGURATION",
            "landing_eligibility": "DECISION_NEEDED",
        },
        "admitted_root_differential": differential,
        "surface_input_support": support,
        "full_entry_support": full_entry,
        "output_init_substitution_check": output_init,
        "acquisition_admission": admission,
        "trajectory_sources": trajectory_sources,
        "eos80_shared_branch": eos_branch,
        "eos80_coefficient_audit": eos_coefficients,
        "candidate_trajectory": trajectory,
        "trajectory_claim": trajectory_claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--orca1ice-root", type=Path, required=True)
    parser.add_argument("--compiled-source", type=Path, default=DEFAULT_COMPILED_SOURCE)
    parser.add_argument(
        "--trajectory-source-root", type=Path,
        default=DEFAULT_TRAJECTORY_SOURCE_ROOT,
    )
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument(
        "--initial-mode",
        choices=("decision52-bridge", "independent"),
        default="decision52-bridge",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--plant", choices=("kt1_T", "surface_hash", "eos80_coeff"))
    args = parser.parse_args()
    try:
        result = run_gate(
            args.deck_root,
            args.v2_root,
            args.orca1ice_root,
            args.compiled_source,
            args.trajectory_source_root,
            max_step=args.max_step,
            initial_mode=args.initial_mode,
            plant=args.plant,
        )
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    if result["status"] == "STOP_RECORD_GAP":
        print("REFUSE: STOP_RECORD_GAP: exact per-step ocean surface inputs are missing",
              file=sys.stderr)
        return 2
    if result["status"] == "STOP_ENTRY_RECORD_GAP":
        print("REFUSE: STOP_ENTRY_RECORD_GAP: full-domain step-entry states are missing",
              file=sys.stderr)
        return 3
    if result["status"] in (
        "STOP_INITIAL_TS_TRANSCRIPTION_GAP",
        *UNBUILT_STATEMENTS,
    ):
        print(
            f"REFUSE: {result['status']}: the admitted ORCA2 ladder cannot "
            "reach kt=10",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

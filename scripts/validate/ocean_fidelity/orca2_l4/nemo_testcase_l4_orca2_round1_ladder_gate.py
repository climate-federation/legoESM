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
    "stage_dump": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:215-231,329-348",
    "runoff_tracer": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:314-328",
    "salt_flux": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:290-311",
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
    return {
        "bit_identical": unequal == 0,
        "unequal": unequal,
        "count": int(actual.size),
        "max_abs": maximum,
        "mean_abs_over_unequal": mean,
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


def chlorophyll_at_step(deck_root: Path, kt: int, wet: np.ndarray) -> np.ndarray:
    """Reproduce the yearly December/January ``fld_read`` interpolation."""

    from netCDF4 import Dataset

    require(1 <= kt <= 10, "chlorophyll step must be in 1..10")
    path = deck_root / "chlorophyll.nc"
    require(path.is_file(), f"missing ORCA2 chlorophyll input: {path}")
    with Dataset(path) as dataset:
        source = np.asarray(dataset.variables["CHLA"][:], dtype=np.float64)
    require(source.shape == (12, 148, 180),
            f"unexpected chlorophyll shape {source.shape}")
    # fldread.F90's yearly interpolation spans the two 31-day centred records.
    # With the 3-hour ORCA2 clock there are 496 steps between the December and
    # January centres; kt=1 is 249/496 of that interval.  Keep NEMO's two-term
    # multiply/add association, independently pinned against its kt=1 dump.
    after = np.float64(248 + kt) / np.float64(496)
    before = np.float64(1.0) - after
    result = before * source[11] + after * source[0]
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
    chl = chlorophyll_at_step(deck_root, kt, wet)
    q_total = np.asarray(nemo_source_round(
        jnp.asarray(fields["qns"]) + jnp.asarray(fields["qsr"])
    ))
    freshwater = FreshwaterForcing(
        precip=jnp.asarray(zeros),
        evap=jnp.asarray(fields["emp"]),
        runoff=jnp.asarray(zeros),
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
    )
    return freshwater, surface


def candidate_trajectory(
    deck_root: Path,
    root: Path,
    card,
    *,
    max_step: int = 10,
) -> dict[str, object]:
    """Run the production ORCA2 step with the acquired exact surface operands."""

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
    # Decision 52 authorizes exactly this operand replacement and nothing else.
    state = state._replace(
        eta=state.eta.replace(data=jnp.asarray(entry1["ssh"], dtype=jnp.float64))
    )
    bridge = compare_fields(_candidate_fields(state), entry1)

    # The kt=1 chlorophyll frame is an independent emitted witness for the
    # input-file interpolation used at all ten steps.
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_rgb_gate as rgb_gate,
    )

    emitted_chl = np.asarray(rgb_gate.read_rgb(
        root / "oracle_rgb_chl_kt00000001.bin"
    )["chl"])
    wet = np.asarray(card.recipe.initial_state.land_mask.data)[:, :90] > 0.5
    reconstructed_chl = chlorophyll_at_step(deck_root, 1,
                                             np.asarray(card.recipe.initial_state.land_mask.data) > 0.5)
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
                "source_citation": TRAJECTORY_CITATIONS["initial_ts"],
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
        except ValueError as exc:
            if "compute_buoyancy_frequency_nemo_bn2 eos_form='eos80'" not in str(exc):
                raise
            require(first_non_bit is not None,
                    "EOS80 BN2 stop preceded a registered entry mismatch")
            require(first_field is not None,
                    "EOS80 BN2 stop has no registered first field")
            return {
                "claim_label": "INDEPENDENT_WITH_DECISION52_SSH",
                "execution": "production-jit-cpu-fp64-x64-libm",
                "decision52_bridge": bridge,
                "independent_entry_before_bridge": independent,
                "given_nemo_entry_eligibility": "STOP_INITIAL_T_S_TRANSCRIPTION",
                "chlorophyll_kt1_input_reconstruction": chl_row,
                "first_non_bit_statement": first_non_bit,
                "kt10_same_field_magnitude": "UNMEASURED_STOP_PRODUCTION_EOS80_BN2_GAP",
                "checkpoints": checkpoints,
                "execution_blocker": {
                    "status": "STOP_PRODUCTION_EOS80_BN2_GAP",
                    "message": str(exc),
                    "nemo_source_citation": (
                        "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/"
                        "nemo/eosbn2.f90:1587-1647"
                    ),
                    "legoesm_source": "packages/ocean/legoesm/ocean/eos.py:748-753",
                },
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
                    "source_citation": (
                        TRAJECTORY_CITATIONS["runoff_tracer"]
                        if stage == 1 and first_field in ("T", "S") else
                        TRAJECTORY_CITATIONS["stage_dump"]
                    ),
                }
        state = trace.state_after

    require(first_non_bit is not None, "candidate trajectory unexpectedly stayed bit-exact")
    require(first_field is not None, "first non-bit field was not recorded")
    kt10_rows = {
        row["checkpoint"]: row["rows"][first_field]
        for row in checkpoints if row["kt"] == max_step
    }
    return {
        "claim_label": "INDEPENDENT_WITH_DECISION52_SSH",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "decision52_bridge": bridge,
        "independent_entry_before_bridge": independent,
        "given_nemo_entry_eligibility": "STOP_INITIAL_T_S_TRANSCRIPTION",
        "chlorophyll_kt1_input_reconstruction": chl_row,
        "first_non_bit_statement": first_non_bit,
        "kt10_same_field_magnitude": kt10_rows,
        "checkpoints": checkpoints,
        "unsupported_recorded_channels": {
            "rnf_tsc": (
                "recorded exactly in every surface frame; the production step "
                "has no channel for NEMO's runoff T/S source"
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
        trajectory = candidate_trajectory(
            deck_root, orca1ice_root, card, max_step=max_step
        )
        status = trajectory.get("execution_blocker", {}).get(
            "status", "STOP_INITIAL_TS_TRANSCRIPTION_GAP"
        )
        trajectory_claim = "MEASURED_INDEPENDENT_WITH_DECISION52_SSH"
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
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=("kt1_T", "surface_hash"))
    args = parser.parse_args()
    try:
        result = run_gate(
            args.deck_root,
            args.v2_root,
            args.orca1ice_root,
            args.compiled_source,
            args.trajectory_source_root,
            max_step=args.max_step,
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
        "STOP_PRODUCTION_EOS80_BN2_GAP",
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

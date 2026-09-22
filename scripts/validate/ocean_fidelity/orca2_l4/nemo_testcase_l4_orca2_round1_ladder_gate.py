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
    unequal_mask = actual != expected
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


def run_gate(
    deck_root: Path,
    v2_root: Path,
    orca1ice_root: Path,
    compiled_source: Path,
    *,
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
    if not support["trajectory_supported"]:
        status = "STOP_RECORD_GAP"
        trajectory_claim = "UNMEASURED_RECORD_GAP"
    elif not full_entry["trajectory_supported"]:
        status = "STOP_ENTRY_RECORD_GAP"
        trajectory_claim = "UNMEASURED_ENTRY_RECORD_GAP"
    else:
        status = "READY_FOR_CANDIDATE_TRAJECTORY"
        trajectory_claim = "RECORDS_AVAILABLE_BUT_CANDIDATE_NOT_RUN"
    return {
        "worktree": stamp,
        "status": status,
        "card": card.case,
        "whole_step_identity": "orca2_vector_een_c2",
        "unmeasured_features": list(card.unmeasured_features),
        "compiled_source": source,
        "kt1_card_vs_v2": card_vs_v2,
        "kt1_card_vs_pinned_orca1ice": card_vs_orca1ice,
        "first_non_bit_statement": {
            "field": "ssh",
            "source_citation": SOURCE_CITATION,
            "ownership": "INITIAL_SI3_CATEGORY_LOAD_CONFIGURATION",
            "landing_eligibility": "DECISION_NEEDED",
        },
        "admitted_root_differential": differential,
        "surface_input_support": support,
        "full_entry_support": full_entry,
        "output_init_substitution_check": output_init,
        "trajectory_claim": trajectory_claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--orca1ice-root", type=Path, required=True)
    parser.add_argument("--compiled-source", type=Path, default=DEFAULT_COMPILED_SOURCE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=("kt1_T",))
    args = parser.parse_args()
    try:
        result = run_gate(
            args.deck_root,
            args.v2_root,
            args.orca1ice_root,
            args.compiled_source,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

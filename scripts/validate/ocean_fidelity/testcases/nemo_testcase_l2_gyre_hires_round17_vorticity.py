#!/usr/bin/env python3
"""Common-state stage-3 ENE-q discriminator for quarter-degree GYRE.

Both NEMO arithmetic flavours and legoESM start from the same vector-math
day-180 restart.  The primary row is the round-16 western-boundary native
surface-V signed mean, narrowed to the production stage-3 vorticity/Coriolis
call.  Only the already formed NEMO q operand is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_testcase_l2_gyre_hires_divergence import wet_span_bins  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

VECTOR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_vector"
)
SCALAR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_scalar"
)
COMMON_RESTART = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round15/oracle_gyre_r4_smag_budget_180d_corrected/control/"
    "GYRE_R4_SMAG_R14_BUDGET_00004320_restart.nc"
)
COMMON_RESTART_SHA256 = (
    "a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12"
)
STAGE_NAME = "oracle_r17_stage3_vor_kt00004321.bin"
OUTPUT_RESTART = "GYRE_R17_VORTEX_00004321_restart.nc"
NATIVE_DIMS = (126, 86, 31)
ACTIVE_LEVELS = 30
TWIN_CLEAR_FACTOR = 4.0
ROUND18_TRANSPORT_WALK = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round18/stage3_vorticity_transport_walk.json"
)
ROUND18_TRANSPORT_WALK_SHA256 = (
    "4bd8ea66ccbb076c243f7c0671575eb4f514f432bc4d0321bb3ccc0244045e5a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)


def read_stage_vorticity(path: Path) -> dict[str, object]:
    """Read the RHS immediately before and after stage-3 ``dyn_vor``."""
    raw = path.read_bytes()
    require(len(raw) >= 60, f"{path}: short stage-3 record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=11i", raw[16:60])
    keys = (
        "version", "kt", "stage", "Kbb", "Kmm", "Krhs", "Kaa",
        "nx", "ny", "nz", "bits",
    )
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L2_R17VOR1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["stage"],
         header["nx"], header["ny"], header["nz"], header["bits"])
        == (1, 4321, 3, *NATIVE_DIMS, 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=60)
    n3 = header["nx"] * header["ny"] * header["nz"]
    require(values.size == 4 * n3, f"{path}: malformed payload")
    names = ("before_u", "before_v", "after_u", "after_v")
    arrays = {
        name: _xyz(values[index * n3:(index + 1) * n3],
                   header["nx"], header["ny"], header["nz"])
        for index, name in enumerate(names)
    }
    for name, array in arrays.items():
        require(np.all(np.isfinite(array[2:-2, 2:-2, :ACTIVE_LEVELS])),
                f"{path}: non-finite active {name} word")
    return {"header": header, **arrays}


def read_ene_operands(path: Path) -> dict[str, object]:
    """Read q and metric transports from one executed ``vor_ene`` call."""
    raw = path.read_bytes()
    require(len(raw) >= 48, f"{path}: short ENE operand record")
    magic = raw[:16].decode("ascii").rstrip()
    values_header = struct.unpack("=8i", raw[16:48])
    keys = ("version", "kt", "Kmm", "kvor", "nx", "ny", "nz", "bits")
    header = dict(zip(keys, values_header, strict=True))
    require(magic == "NEMO_L2_R17Q__1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["nz"], header["bits"])
        == (1, 4321, NATIVE_DIMS[0], NATIVE_DIMS[1], ACTIVE_LEVELS, 64),
        f"{path}: bad header {header}",
    )
    require(header["kvor"] == 4,
            f"{path}: expected executed np_CRV=4, got {header['kvor']}")
    values = np.frombuffer(raw, dtype=np.float64, offset=48)
    n2 = header["nx"] * header["ny"]
    require(values.size == 3 * n2 * header["nz"],
            f"{path}: malformed payload")
    slabs = values.reshape((header["nz"], 3, n2))
    arrays = {}
    for index, name in enumerate(("q", "Fu", "Fv")):
        arrays[name] = np.stack([
            slab.reshape((header["nx"], header["ny"]), order="F").T
            for slab in slabs[:, index, :]
        ], axis=-1)
    # dynvor.f90:537-573 does not define the outer work-array halo.  Admit only
    # words consumed by the executed q/transport stencil; refusing an
    # uninitialised, unread halo word would make the guard depend on stack
    # history, while accepting a non-finite consumed word would hide a failure.
    consumed = {
        "q": arrays["q"][1:-2, 1:-2],
        "Fu": arrays["Fu"][2:-1, 1:-2],
        "Fv": arrays["Fv"][1:-2, 2:-1],
    }
    for name, array in consumed.items():
        require(np.all(np.isfinite(array)),
                f"{path}: non-finite consumed {name} word")
    return {"header": header, **arrays}


def read_stage2_state(path: Path) -> dict[str, object]:
    """Read the stage-2 output that becomes stage 3's Kmm input.

    The inherited P3 writer calls ``l1_dump_stage(..., 2, Naa)`` immediately
    after the stage-2 call and before swapping ``Naa`` into ``Nnn``
    (compiled ``stprk3.f90:204-208``).  The round-17 header independently
    records Kmm=2 for stage 3, so this is the named velocity/SSH provider.
    """
    raw = path.read_bytes()
    require(len(raw) >= 52, f"{path}: short stage-2 state record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=9i", raw[16:52])
    keys = ("version", "kt", "stage", "Kaa", "nx", "ny", "nz",
            "ntr", "bits")
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["stage"], header["Kaa"],
         header["nx"], header["ny"], header["nz"], header["ntr"],
         header["bits"])
        == (1, 4321, 2, 2, *NATIVE_DIMS, 2, 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=52)
    n3 = header["nx"] * header["ny"] * header["nz"]
    n2 = header["nx"] * header["ny"]
    require(values.size == 4 * n3 + n2, f"{path}: malformed payload")
    arrays = {
        name: _xyz(values[index * n3:(index + 1) * n3],
                   header["nx"], header["ny"], header["nz"])[
                       2:-2, 2:-2, :ACTIVE_LEVELS]
        for index, name in enumerate(("T", "S", "u", "v"))
    }
    arrays["ssh"] = values[4 * n3:].reshape(
        (header["nx"], header["ny"]), order="F")[2:-2, 2:-2].T
    for name, array in arrays.items():
        require(np.all(np.isfinite(array)),
                f"{path}: non-finite active {name} word")
    return {"header": header, **arrays}


def wordwise_result(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, object]:
    """Report exact binary64 identity and ULP distance over named words."""
    from nemo_testcase_state_ulp_probe import ulp_distance

    candidate = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64))
    oracle = np.ascontiguousarray(np.asarray(oracle, dtype=np.float64))
    require(candidate.shape == oracle.shape,
            f"wordwise provider shapes differ: {candidate.shape} != {oracle.shape}")
    require(np.all(np.isfinite(candidate)) and np.all(np.isfinite(oracle)),
            "wordwise provider contains a non-finite value")
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    ulps = ulp_distance(candidate, oracle)
    return {
        "words": int(candidate.size),
        "unequal_words": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(np.abs(candidate - oracle))),
        "max_ulp": int(np.max(ulps)),
        "bit_exact": bool(not np.any(unequal)),
    }


def classify_factor_walk(arms: dict[str, dict[str, object]]) -> tuple[list[str], str]:
    """Apply the preregistered two-boundary exactness rule."""
    sufficient = sorted(
        name for name, row in arms.items()
        if row["formed_operand"]["bit_exact"]
        and row["source_accumulator"]["bit_exact"]
        and row["untouched_rows_bit_exact"]
    )
    return sufficient, (
        "SINGLE_FACTOR_SUFFICIENT" if sufficient
        else "COUPLED_FORMATION_COMPOSITION"
    )


def manifest_contract(root: Path, flavour: str) -> dict[str, object]:
    """Verify the closed acquisition and same-binary passivity control."""
    manifest = root / "manifest.sha256"
    require(manifest.is_file(), f"{root}: missing manifest.sha256")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        member = Path(relative)
        require(not member.is_absolute() and ".." not in member.parts,
                f"{root}: manifest member escapes root: {relative}")
        path = root / relative
        require(path.is_file(), f"{root}: manifest member missing: {relative}")
        require(sha256(path) == digest,
                f"{root}: manifest member moved: {relative}")
        checked += 1
    require(checked > 0, f"{root}: empty manifest")
    recorded_flavour = (root / "manifest/precision_flavour.txt").read_text(
        encoding="utf-8").strip()
    require(recorded_flavour == flavour,
            f"{root}: precision flavour {recorded_flavour!r} != {flavour!r}")
    restart_line = (root / "manifest/common_restart.sha256").read_text(
        encoding="utf-8").strip()
    require(restart_line.startswith(COMMON_RESTART_SHA256 + "  "),
            f"{root}: common restart pin differs")
    control = root / "control" / OUTPUT_RESTART
    instrumented = root / "instrumented" / OUTPUT_RESTART
    require(sha256(control) == sha256(instrumented),
            f"{root}: writer moved the terminal restart")
    return {
        "root": str(root.resolve()),
        "precision_flavour": flavour,
        "producer_commit": (root / "manifest/producer_commit.txt").read_text(
            encoding="utf-8").strip(),
        "manifest_sha256": sha256(manifest),
        "verified_members": checked,
        "terminal_restart_sha256": sha256(control),
    }


def _native_v_term(record: dict[str, object]) -> np.ndarray:
    """Map NEMO's accumulated stage-3 V increment onto legoESM V faces."""
    return _native_v_field(record, "after_v") - _native_v_field(
        record, "before_v")


def _native_v_field(record: dict[str, object], name: str) -> np.ndarray:
    """Map one native NEMO V field onto legoESM's owned north faces."""
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        nemo_closed_v_north_to_face,
    )

    native = np.asarray(record[name])
    owned = native[2:-2, 2:-2, :ACTIVE_LEVELS]
    return np.asarray(nemo_closed_v_north_to_face(owned))


def _q_to_legoesm(operands: dict[str, object]) -> np.ndarray:
    """Map NEMO F points, including the live south/west halo, to vertices."""
    q = np.asarray(operands["q"])
    mapped = q[1:-2, 1:-2, :]
    require(mapped.shape == (83, 123, ACTIVE_LEVELS),
            f"mapped q has wrong shape {mapped.shape}")
    return mapped


def _transports_to_legoesm(
    operands: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Map NEMO's consumed ``zwx``/``zwy`` ranges to legoESM faces."""
    fu = np.asarray(operands["Fu"])[2:-2, 1:-2, :]
    fv = np.asarray(operands["Fv"])[1:-2, 2:-2, :]
    nx, ny, _ = NATIVE_DIMS
    require(fu.shape == (ny - 4, nx - 3, ACTIVE_LEVELS),
            f"mapped F_u has wrong shape {fu.shape}")
    require(fv.shape == (ny - 3, nx - 4, ACTIVE_LEVELS),
            f"mapped F_v has wrong shape {fv.shape}")
    return fu, fv


def source_accumulator_result(
    candidate_term: np.ndarray,
    oracle_record: dict[str, object],
    selection: np.ndarray,
) -> dict[str, object]:
    """Score NEMO's executed ``rhs = rhs + term`` statement directly."""
    before = _native_v_field(oracle_record, "before_v")
    after = _native_v_field(oracle_record, "after_v")
    candidate_after = np.asarray(before + candidate_term, dtype=np.float64)
    require(candidate_after.shape == after.shape == selection.shape,
            "source-accumulator provider shapes differ")
    require(np.all(np.isfinite(candidate_after[selection]))
            and np.all(np.isfinite(after[selection])),
            "source-accumulator row has non-finite active values")
    left = np.ascontiguousarray(candidate_after[selection])
    right = np.ascontiguousarray(after[selection])
    unequal = left.view(np.uint64) != right.view(np.uint64)
    residual = left - right
    return {
        "registered_unequal_words": int(np.count_nonzero(unequal)),
        "registered_max_abs_m_s2": float(np.max(np.abs(residual))),
        "registered_signed_mean_m_s2": float(np.mean(
            residual, dtype=np.float64)),
        "bit_exact": bool(not np.any(unequal)),
    }


def classify_operand_boundary(
    arm_results: dict[str, dict[str, object]],
) -> tuple[list[str], str]:
    """Classify the preregistered q/F_u subsets at the exact source bar."""
    names = ("q_only", "Fu_only", "q_plus_Fu")
    exact = [
        name for name in names
        if arm_results[name]["source_accumulator"]["bit_exact"]
    ]
    boundary = (
        "TRANSPORT_FORMATION" if "q_plus_Fu" in exact
        else "COMPLETED_STENCIL"
    )
    return exact, boundary


def active_plant_locations(
    western: np.ndarray,
    q: np.ndarray,
    fu: np.ndarray,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Pick consumed q/F_u words with maximal surface-V stencil leverage."""
    require(q.shape[0] == western.shape[0]
            and q.shape[1] == western.shape[1] + 1,
            "q/registered-V plant shapes differ")
    require(fu.shape[0] + 1 == western.shape[0]
            and fu.shape[1] == western.shape[1] + 1,
            "F_u/registered-V plant shapes differ")
    fu_pad = np.pad(fu[..., 0], ((1, 1), (0, 0)), mode="constant")
    rows, cols = np.nonzero(western)
    require(rows.size > 0, "registered V plant row is empty")
    west_sensitivity = fu_pad[rows, cols] + fu_pad[rows + 1, cols]
    east_sensitivity = fu_pad[rows, cols + 1] + fu_pad[rows + 1, cols + 1]
    all_sensitivity = np.stack(
        [np.abs(west_sensitivity), np.abs(east_sensitivity)], axis=1)
    flat = int(np.argmax(all_sensitivity))
    face_index, side = np.unravel_index(flat, all_sensitivity.shape)
    require(all_sensitivity[face_index, side] > 0.0,
            "every consumed q plant coefficient is zero")
    j = int(rows[face_index])
    i = int(cols[face_index] + side)
    q_index = (j, i, 0)

    q_w = np.abs(q[rows, cols, 0])
    q_e = np.abs(q[rows, cols + 1, 0])
    all_q = np.stack([q_w, q_e], axis=1)
    flat_fu = int(np.argmax(all_q))
    face_fu, side_fu = np.unravel_index(flat_fu, all_q.shape)
    require(all_q[face_fu, side_fu] > 0.0,
            "every consumed F_u plant coefficient is zero")
    j_fu = int(rows[face_fu])
    i_fu = int(cols[face_fu] + side_fu)
    # In the zero-padded latitude stencil, V row j consumes F_u rows j-1 and
    # j.  Prefer j where it exists; only the final V row needs j-1.
    row_fu = j_fu if j_fu < fu.shape[0] else j_fu - 1
    fu_index = (row_fu, i_fu, 0)
    return q_index, fu_index


def signed_row(candidate: np.ndarray, reference: np.ndarray,
               selection: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    require(candidate.shape == reference.shape,
            "stage-3 row provider shapes differ")
    require(selection.shape == candidate.shape,
            "stage-3 row selection shape differs")
    require(np.any(selection), "western-boundary surface-V row is empty")
    require(np.all(np.isfinite(candidate[selection]))
            and np.all(np.isfinite(reference[selection])),
            "stage-3 row has non-finite active values")
    return float(np.mean(candidate[selection] - reference[selection],
                         dtype=np.float64))


def substitution_verdict(
    baseline: np.ndarray,
    substituted: np.ndarray,
    vector: np.ndarray,
    scalar: np.ndarray,
    selection: np.ndarray,
    untouched_exact: bool,
) -> dict[str, object]:
    """Apply the preregistered common-state NEMO-envelope verdict."""
    baseline_minus_vector = signed_row(baseline, vector, selection)
    substituted_minus_scalar = signed_row(substituted, scalar, selection)
    scalar_minus_vector = signed_row(scalar, vector, selection)
    envelope = abs(scalar_minus_vector)
    residual = abs(substituted_minus_scalar)
    candidate = abs(baseline_minus_vector)
    ratio = None if envelope == 0.0 else candidate / envelope
    confirms = residual <= envelope and untouched_exact
    return {
        "baseline_legoesm_minus_vector_signed_mean_m_s2": baseline_minus_vector,
        "substituted_legoesm_minus_scalar_signed_mean_m_s2":
            substituted_minus_scalar,
        "scalar_minus_vector_signed_mean_m_s2": scalar_minus_vector,
        "nemo_twin_envelope_abs_m_s2": envelope,
        "substituted_residual_abs_m_s2": residual,
        "baseline_candidate_to_twin_ratio": ratio,
        "round16_four_times_context_clears": (
            candidate > 0.0 if envelope == 0.0
            else ratio is not None and ratio >= TWIN_CLEAR_FACTOR
        ),
        "untouched_rows_bit_exact": untouched_exact,
        "verdict": "CONFIRM_Q_FORMATION" if confirms else "REFUTE_Q_FORMATION",
    }


def _load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None,
            f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def common_state(card):
    """Bridge the pinned restart, including the six external-mode histories."""
    import jax.numpy as jnp

    owners = _load_module("_r17_gyre_year_owners",
                          "nemo_testcase_l2_gyre_year_owners.py")
    year = _load_module("_r17_gyre_year_fromrest",
                        "nemo_testcase_l2_gyre_year_fromrest.py")
    state, _ = owners._gyre_checkpoint_state(card, COMMON_RESTART)
    payload = year._load_daily_reset_payload(COMMON_RESTART, ACTIVE_LEVELS)
    require(payload["sha256"] == COMMON_RESTART_SHA256,
            "restart bridge read a different common state")
    history = (
        jnp.asarray(year._face_history(payload["ub_e"], "u")),
        jnp.asarray(year._face_history(payload["ubb_e"], "u")),
        jnp.asarray(year._face_history(payload["vb_e"], "v")),
        jnp.asarray(year._face_history(payload["vbb_e"], "v")),
        jnp.asarray(payload["sshb_e"]),
        jnp.asarray(payload["sshbb_e"]),
    )
    return state._replace(bt_hist=history)


def common_card():
    """Build the registered physical deck; step 4321 belongs to the forcing."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_nemo_testcase_card,
    )

    return build_nemo_testcase_card(
        "GYRE-zco", namelist=GYRE_R4_SMAG_NAMELIST)


# Compatibility for the already-committed round-17 direct tests.  New tools
# use the public names above; the aliases do not create a second implementation.
_manifest_contract = manifest_contract
_common_state = common_state
_common_card = common_card


def run(
    vector_root: Path,
    scalar_root: Path,
    *,
    plant_q: bool = False,
    plant_fu: bool = False,
) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "round-18 gate requires CPU")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "production JIT is required")
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "round-18 policy is not certified fp64/libm")
    require(sha256(COMMON_RESTART) == COMMON_RESTART_SHA256,
            "common developed restart moved")
    require(not (plant_q and plant_fu),
            "q and F_u plants must be run separately")

    vector_contract = manifest_contract(vector_root, "vector-math-conda")
    scalar_contract = manifest_contract(scalar_root, "scalar-math-conda")
    vector_stage = read_stage_vorticity(vector_root / "instrumented" / STAGE_NAME)
    scalar_stage = read_stage_vorticity(scalar_root / "instrumented" / STAGE_NAME)
    require(vector_stage["header"] == scalar_stage["header"],
            "scalar/vector stage headers differ")
    kmm = int(vector_stage["header"]["Kmm"])
    q_name = f"oracle_r17_ene_q_kmm{kmm}_kt00004321.bin"
    vector_operands = read_ene_operands(vector_root / "instrumented" / q_name)
    scalar_operands = read_ene_operands(scalar_root / "instrumented" / q_name)
    require(vector_operands["header"] == scalar_operands["header"],
            "scalar/vector ENE headers differ")

    card = common_card()
    state = common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, 4321)
    scalar_q = _q_to_legoesm(scalar_operands)
    vector_q = _q_to_legoesm(vector_operands)
    scalar_fu, scalar_fv = _transports_to_legoesm(scalar_operands)
    vector_fu, vector_fv = _transports_to_legoesm(vector_operands)

    mask = np.asarray(state.v_mask.data, dtype=bool)
    western = wet_span_bins(mask) == 0
    selection = western[..., None] & np.zeros(
        (*mask.shape, ACTIVE_LEVELS), dtype=bool)
    selection[..., 0] = western

    def expose(
        name: str,
        q_override=None,
        transport_override=None,
    ) -> np.ndarray:
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_operator=name,
                expose_momentum_operator_stage=3,
                stage3_vorticity_q_override=q_override,
                stage3_vorticity_transport_override=transport_override,
            ),
        )
        exposed = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        return np.asarray(exposed.v.data)

    if plant_q or plant_fu:
        q_index, fu_index = active_plant_locations(
            western, scalar_q, scalar_fu)
        if plant_q:
            planted = np.array(scalar_q, copy=True)
            planted[q_index] += 1.0
            reference_term = expose(
                "vorticity", jnp.asarray(scalar_q), None)
            planted_term = expose(
                "vorticity", jnp.asarray(planted), None)
            name = "q"
            index = q_index
        else:
            planted = np.array(scalar_fu, copy=True)
            planted[fu_index] += 1.0
            reference_term = expose(
                "vorticity", None, (jnp.asarray(scalar_fu), None))
            planted_term = expose(
                "vorticity", None, (jnp.asarray(planted), None))
            name = "F_u"
            index = fu_index
        changed = np.ascontiguousarray(
            planted_term[selection]).view(np.uint64) != np.ascontiguousarray(
                reference_term[selection]).view(np.uint64)
        plant_result = {
            "operand": name,
            "native_index": list(index),
            "registered_changed_words": int(np.count_nonzero(changed)),
            "registered_signed_mean_move_m_s2": signed_row(
                planted_term, reference_term, selection),
        }
        require(plant_result["registered_changed_words"] > 0,
                f"planted active {name} word did not move the scored V row")
        return {
            "format": "gyre-r4-smag-round18-stage3-vorticity-plant-v2",
            "worktree": worktree_stamp(),
            "common_restart": {
                "path": str(COMMON_RESTART), "sha256": COMMON_RESTART_SHA256,
                "step": 4320, "next_step": 4321,
            },
            "policy": "fp64-libm-certified-cpu",
            "jax_backend": jax.default_backend(),
            "plant": name,
            "providers": {
                "vector": vector_contract, "scalar": scalar_contract},
            "stage_header": vector_stage["header"],
            "operand_record": q_name,
            "owner_boundary": "PLANT_FIRED",
            "exact_operand_subsets": [],
            "plant_result": plant_result,
        }

    overrides = {
        "baseline": (None, None),
        "q_only": (jnp.asarray(scalar_q), None),
        "Fu_only": (None, (jnp.asarray(scalar_fu), None)),
        "q_plus_Fu": (
            jnp.asarray(scalar_q), (jnp.asarray(scalar_fu), None)),
        "Fv_control": (None, (None, jnp.asarray(scalar_fv))),
    }
    exposed = {}
    for arm, (q_override, transport_override) in overrides.items():
        exposed[arm] = {
            name: expose(name, q_override, transport_override)
            for name in ("hpg", "vorticity", "advection")
        }
    baseline_hpg = exposed["baseline"]["hpg"]
    baseline_term = exposed["baseline"]["vorticity"]
    baseline_advection = exposed["baseline"]["advection"]
    vector_term = _native_v_term(vector_stage)
    scalar_term = _native_v_term(scalar_stage)
    arm_results = {}
    for arm, fields in exposed.items():
        untouched = {
            "stage3_after_hpg_v": bool(np.array_equal(
                baseline_hpg, fields["hpg"])),
            "stage3_advection_v": bool(np.array_equal(
                baseline_advection, fields["advection"])),
        }
        arm_results[arm] = {
            "untouched_rows": untouched,
            "posthoc_term_subtraction_context": substitution_verdict(
                baseline_term, fields["vorticity"], vector_term, scalar_term,
                selection, all(untouched.values())),
            "source_accumulator": source_accumulator_result(
                fields["vorticity"], scalar_stage, selection),
        }

    fv_control_exact = bool(np.array_equal(
        baseline_term, exposed["Fv_control"]["vorticity"]))
    require(fv_control_exact,
            "F_v negative control moved the V-stencil result")
    require(all(all(row["untouched_rows"].values())
                for row in arm_results.values()),
            "an operand substitution moved HPG or advection")
    exact_subsets, boundary = classify_operand_boundary(arm_results)

    q_selection = np.zeros_like(scalar_q, dtype=bool)
    q_selection[:, :-1, 0] |= western
    q_selection[:, 1:, 0] |= western
    q_diff = np.asarray(scalar_q) - vector_q
    report = {
        "format": "gyre-r4-smag-round18-stage3-vorticity-v2",
        "worktree": worktree_stamp(),
        "common_restart": {
            "path": str(COMMON_RESTART), "sha256": COMMON_RESTART_SHA256,
            "step": 4320, "next_step": 4321,
        },
        "policy": "fp64-libm-certified-cpu",
        "jax_backend": jax.default_backend(),
        "plant": "q" if plant_q else "F_u" if plant_fu else None,
        "providers": {"vector": vector_contract, "scalar": scalar_contract},
        "stage_header": vector_stage["header"],
        "operand_record": q_name,
        "reduction": (
            "stage-3 native surface V; western-boundary wet-span bin; "
            "signed mean and source-accumulator word identity"
        ),
        "nemo_scalar_minus_vector_q": {
            "active_surface_max_abs_s-1_m-1": float(np.max(
                np.abs(q_diff[q_selection]))),
            "active_surface_unequal_words": int(np.count_nonzero(
                q_diff[q_selection] != 0.0)),
        },
        "nemo_scalar_minus_vector_metric_transports": {
            "F_u_surface_max_abs_m3_s": float(np.max(np.abs(
                scalar_fu[..., 0] - vector_fu[..., 0]))),
            "F_u_surface_unequal_words": int(np.count_nonzero(
                scalar_fu[..., 0] != vector_fu[..., 0])),
            "F_v_surface_max_abs_m3_s": float(np.max(np.abs(
                scalar_fv[..., 0] - vector_fv[..., 0]))),
            "F_v_surface_unequal_words": int(np.count_nonzero(
                scalar_fv[..., 0] != vector_fv[..., 0])),
        },
        "source_accumulator_comparison": (
            "float64(NEMO_before + legoESM_term) versus NEMO_after"),
        "arms": arm_results,
        "Fv_negative_control_full_v_bit_exact": fv_control_exact,
        "exact_operand_subsets": exact_subsets,
        "owner_boundary": boundary,
        "plant_result": None,
        "result": arm_results["q_only"][
            "posthoc_term_subtraction_context"],
        "source_citations": [
            "compiled stprk3_stg.f90:399-410",
            "compiled dynvor.f90:234-239",
            "compiled dynvor.f90:536-573",
        ],
    }
    return report


def run_factor_walk(
    vector_root: Path,
    scalar_root: Path,
    *,
    plant_q: bool = False,
    plant_fu: bool = False,
) -> dict:
    """Walk the six preregistered source factors of live stage-3 q/F_u."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
        compute_vertex_mask,
        nemo_een_ene_vertex_coriolis,
        nemo_vor_ene_vorticity_cgrid,
        pv_flux_ene,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        nemo_closed_u_east_to_face,
        nemo_closed_v_north_to_face,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_card_mesh_operands,
        nemo_qco_live_face_geometry_cgrid,
        nemo_qco_live_vorticity_e3f_cgrid,
        nemo_qco_vorticity_f_cgrid,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "round-19 gate requires CPU")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "production JIT is required")
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "round-19 policy is not certified fp64/libm")
    require(sha256(COMMON_RESTART) == COMMON_RESTART_SHA256,
            "common developed restart moved")
    require(not (plant_q and plant_fu),
            "q and F_u plants must be run separately")
    stamp = worktree_stamp()
    require(stamp["clean"],
            f"round-19 factor walk requires clean tree: {stamp['dirty_paths']}")

    vector_contract = manifest_contract(vector_root, "vector-math-conda")
    scalar_contract = manifest_contract(scalar_root, "scalar-math-conda")
    scalar_stage = read_stage_vorticity(
        scalar_root / "instrumented" / STAGE_NAME)
    kmm = int(scalar_stage["header"]["Kmm"])
    require(kmm == 2, f"stage-3 factor walk requires Kmm=2, got {kmm}")
    q_name = f"oracle_r17_ene_q_kmm{kmm}_kt00004321.bin"
    scalar_operands = read_ene_operands(
        scalar_root / "instrumented" / q_name)
    scalar_stage2 = read_stage2_state(
        scalar_root / "instrumented" / "oracle_stage_kt00004321_s2.bin")
    scalar_q = _q_to_legoesm(scalar_operands)
    scalar_fu, _ = _transports_to_legoesm(scalar_operands)

    card = common_card()
    state = common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, 4321)
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    config = card.recipe.model_config
    trace_model = LatLonCGridOceanModel(
        grid, z_coord, config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    trace = trace_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface)
    trace = jax.device_get(trace)
    ordinary = LatLonCGridOceanModel(grid, z_coord, config).step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface)
    left_leaves, left_tree = jax.tree_util.tree_flatten(trace.state_after)
    right_leaves, right_tree = jax.tree_util.tree_flatten(
        jax.device_get(ordinary))
    require(left_tree == right_tree,
            "trace and ordinary state pytree structures differ")
    passive_mismatches = 0
    for left, right in zip(left_leaves, right_leaves, strict=True):
        left = np.ascontiguousarray(np.asarray(left))
        right = np.ascontiguousarray(np.asarray(right))
        require(left.shape == right.shape and left.dtype == right.dtype,
                "trace and ordinary state leaf shape/dtype differs")
        passive_mismatches += int(np.count_nonzero(
            left.view(np.uint8) != right.view(np.uint8)))
    require(passive_mismatches == 0,
            f"live q/F_u trace moved {passive_mismatches} state bytes")
    parts = trace.operator_operands[2]
    live_q = np.asarray(parts["operand_vorticity_q"], dtype=np.float64)
    live_fu = np.asarray(parts["operand_vorticity_Fu"], dtype=np.float64)
    live_u = np.asarray(parts["operand_velocity_u"], dtype=np.float64)
    live_v = np.asarray(parts["operand_velocity_v"], dtype=np.float64)
    live_hu = np.asarray(parts["operand_momentum_h_u"], dtype=np.float64)
    live_hf = np.asarray(parts["operand_vorticity_h_vtx"], dtype=np.float64)
    live_f = np.asarray(nemo_een_ene_vertex_coriolis(grid),
                        dtype=np.float64)
    live_e2u = np.asarray(grid.dy_u, dtype=np.float64)
    require(live_q.shape == scalar_q.shape
            and live_fu.shape == scalar_fu.shape,
            "live and scalar formed-operand shapes differ")

    scalar_u = nemo_closed_u_east_to_face(scalar_stage2["u"])
    scalar_v = nemo_closed_v_north_to_face(scalar_stage2["v"])
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m)
    umask3, vmask3 = compute_face_masks_3d(z_coord.is_active, grid)
    qco_mesh = nemo_qco_card_mesh_operands(
        h_ref, umask3, vmask3, grid, jnp.float64)
    scalar_hu = np.asarray(nemo_qco_live_face_geometry_cgrid(
        jnp.asarray(scalar_stage2["ssh"]), qco_mesh.e3u_0,
        qco_mesh.e3v_0, qco_mesh.umask3, qco_mesh.vmask3,
        qco_mesh.hu_0, qco_mesh.hv_0, qco_mesh.area_t,
        qco_mesh.area_u, qco_mesh.area_v)[0], dtype=np.float64)
    scalar_hf = np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
        jnp.asarray(scalar_stage2["ssh"]), z_coord, jnp.float64,
        grid=grid), dtype=np.float64)
    scalar_f = np.asarray(nemo_qco_vorticity_f_cgrid(
        z_coord, jnp.float64), dtype=np.float64)
    raw_e2u = np.asarray(z_coord.nemo_een_barotropic.e2u,
                        dtype=np.float64)
    scalar_e2u = np.concatenate([raw_e2u[:, :1], raw_e2u], axis=1)
    require(scalar_u.shape == live_u.shape and scalar_v.shape == live_v.shape,
            "stage-2 velocity mapping differs from live stage-3 shapes")
    require(scalar_hu.shape == live_hu.shape
            and scalar_hf.shape == live_hf.shape
            and scalar_e2u.shape == live_e2u.shape,
            "scalar mesh provider differs from live factor shapes")

    @jax.jit
    def form_q(u_value, v_value, f_value, hf_value):
        zeta = nemo_vor_ene_vorticity_cgrid(u_value, v_value, grid)
        total = zeta + f_value[..., jnp.newaxis]
        denominator = jnp.maximum(hf_value, jnp.asarray(1.0e-10))
        return nemo_source_round(
            nemo_source_round(total) / denominator)

    @jax.jit
    def form_fu(e2u_value, hu_value, u_value):
        formed = nemo_source_round(
            nemo_source_round(e2u_value[..., jnp.newaxis] * hu_value)
            * u_value)
        return nemo_source_round(formed * umask3)

    calibrated_q = np.asarray(form_q(
        jnp.asarray(live_u), jnp.asarray(live_v), jnp.asarray(live_f),
        jnp.asarray(live_hf)))
    calibrated_fu = np.asarray(form_fu(
        jnp.asarray(live_e2u), jnp.asarray(live_hu), jnp.asarray(live_u)))
    calibration = {
        "q": wordwise_result(calibrated_q, live_q),
        "F_u": wordwise_result(calibrated_fu, live_fu),
    }
    require(calibration["q"]["bit_exact"],
            "round-19 q re-formation does not reproduce the captured live q")
    require(calibration["F_u"]["bit_exact"],
            "round-19 F_u re-formation does not reproduce captured live F_u")

    vertex_mask = compute_vertex_mask(state.land_mask.data, grid=grid)

    @jax.jit
    def replay_v_term(q_value, fu_value):
        zeta = nemo_vor_ene_vorticity_cgrid(
            jnp.asarray(live_u), jnp.asarray(live_v), grid)
        _, v_term = pv_flux_ene(
            zeta, jnp.asarray(live_hf),
            jnp.asarray(parts["operand_momentum_h_v"]), jnp.asarray(live_v),
            jnp.asarray(live_hu), jnp.asarray(live_u),
            umask3, vmask3, vertex_mask,
            f_vtx=jnp.asarray(live_f), q_boundary="nemo_live",
            metric_widths=(grid.dx_u, grid.dx_v, grid.dy_u, grid.dy_v),
            potential_vorticity_override=q_value,
            metric_transport_override=(fu_value, None),
        )
        # ``return_nemo_operator_components`` publishes ``_mv(term)`` after
        # the common diagnostic mask, not the unmasked kernel return
        # (ocean_pe_latlon_cgrid.py:6234-6238,6296-6301).
        return v_term * vmask3

    replayed_live_v = np.asarray(replay_v_term(
        jnp.asarray(live_q), jnp.asarray(live_fu)))
    kernel_calibration = wordwise_result(
        replayed_live_v, np.asarray(parts["vorticity_v"].data))
    require(kernel_calibration["bit_exact"],
            "dynamic ENE replay does not reproduce captured production V")

    mask = np.asarray(state.v_mask.data, dtype=bool)
    western = wet_span_bins(mask) == 0
    selection = western[..., None] & np.zeros(
        (*mask.shape, ACTIVE_LEVELS), dtype=bool)
    selection[..., 0] = western

    if plant_q or plant_fu:
        q_index, fu_index = active_plant_locations(western, live_q, live_fu)
        if plant_q:
            reference = live_q
            planted = np.array(live_q, copy=True)
            index = q_index
            planted[index] += 1.0
            before = np.asarray(replay_v_term(
                jnp.asarray(reference), jnp.asarray(scalar_fu)))
            after = np.asarray(replay_v_term(
                jnp.asarray(planted), jnp.asarray(scalar_fu)))
            operand = "q"
        else:
            reference = live_fu
            planted = np.array(live_fu, copy=True)
            index = fu_index
            planted[index] += 1.0
            before = np.asarray(replay_v_term(
                jnp.asarray(scalar_q), jnp.asarray(reference)))
            after = np.asarray(replay_v_term(
                jnp.asarray(scalar_q), jnp.asarray(planted)))
            operand = "F_u"
        changed = (np.ascontiguousarray(after[selection]).view(np.uint64)
                   != np.ascontiguousarray(before[selection]).view(np.uint64))
        plant_result = {
            "operand": operand,
            "index": list(index),
            "registered_changed_words": int(np.count_nonzero(changed)),
            "registered_signed_mean_move_m_s2": signed_row(
                after, before, selection),
        }
        require(plant_result["registered_changed_words"] > 0,
                f"planted live {operand} word did not move the scored V row")
        return {
            "format": "gyre-r4-smag-round19-formed-factor-plant-v1",
            "worktree": stamp,
            "policy": "fp64-libm-certified-cpu",
            "trace_passivity_mismatched_bytes": passive_mismatches,
            "providers": {"vector": vector_contract,
                          "scalar": scalar_contract},
            "calibration": calibration,
            "kernel_calibration": kernel_calibration,
            "plant": operand,
            "plant_result": plant_result,
            "owner_boundary": "PLANT_FIRED",
        }

    candidates = {
        "q__kmm_velocity": (
            "q",
            np.asarray(form_q(
                jnp.asarray(scalar_u), jnp.asarray(scalar_v),
                jnp.asarray(live_f), jnp.asarray(live_hf))),
        ),
        "q__coriolis": (
            "q",
            np.asarray(form_q(
                jnp.asarray(live_u), jnp.asarray(live_v),
                jnp.asarray(scalar_f), jnp.asarray(live_hf))),
        ),
        "q__live_f_thickness": (
            "q",
            np.asarray(form_q(
                jnp.asarray(live_u), jnp.asarray(live_v),
                jnp.asarray(live_f), jnp.asarray(scalar_hf))),
        ),
        "Fu__e2u_metric": (
            "F_u",
            np.asarray(form_fu(
                jnp.asarray(scalar_e2u), jnp.asarray(live_hu),
                jnp.asarray(live_u))),
        ),
        "Fu__live_u_thickness": (
            "F_u",
            np.asarray(form_fu(
                jnp.asarray(live_e2u), jnp.asarray(scalar_hu),
                jnp.asarray(live_u))),
        ),
        "Fu__kmm_u_velocity": (
            "F_u",
            np.asarray(form_fu(
                jnp.asarray(live_e2u), jnp.asarray(live_hu),
                jnp.asarray(scalar_u))),
        ),
    }
    require(sha256(ROUND18_TRANSPORT_WALK)
            == ROUND18_TRANSPORT_WALK_SHA256,
            "round-18 seam-isolation control moved")
    round18 = json.loads(ROUND18_TRANSPORT_WALK.read_text(encoding="utf-8"))
    require(round18["common_restart"]["sha256"] == COMMON_RESTART_SHA256,
            "round-18 control used a different common restart")
    require(round18["arms"]["q_plus_Fu"]["source_accumulator"]
            ["registered_unequal_words"] == 0,
            "round-18 q+F_u control is no longer exact")
    isolation_control = all(
        all(row["untouched_rows"].values())
        for row in round18["arms"].values())
    require(isolation_control,
            "round-18 substitution moved HPG or advection")
    arms = {}
    for name, (formed_name, candidate) in candidates.items():
        q_value = candidate if formed_name == "q" else scalar_q
        fu_value = candidate if formed_name == "F_u" else scalar_fu
        vorticity = np.asarray(replay_v_term(
            jnp.asarray(q_value), jnp.asarray(fu_value)))
        oracle_formed = scalar_q if formed_name == "q" else scalar_fu
        arms[name] = {
            "formed_quantity": formed_name,
            "formed_operand": wordwise_result(candidate, oracle_formed),
            "source_accumulator": source_accumulator_result(
                vorticity, scalar_stage, selection),
            "untouched_rows_bit_exact": isolation_control,
        }
    require(all(row["untouched_rows_bit_exact"] for row in arms.values()),
            "a factor substitution moved HPG or advection")
    sufficient, boundary = classify_factor_walk(arms)

    report = {
        "format": "gyre-r4-smag-round19-formed-factor-walk-v1",
        "worktree": stamp,
        "common_restart": {
            "path": str(COMMON_RESTART), "sha256": COMMON_RESTART_SHA256,
            "step": 4320, "next_step": 4321,
        },
        "policy": "fp64-libm-certified-cpu",
        "jax_backend": jax.default_backend(),
        "trace_passivity_mismatched_bytes": passive_mismatches,
        "providers": {"vector": vector_contract, "scalar": scalar_contract},
        "stage_header": scalar_stage["header"],
        "stage2_header": scalar_stage2["header"],
        "operand_record": q_name,
        "calibration": calibration,
        "kernel_calibration": kernel_calibration,
        "seam_isolation_control": {
            "path": str(ROUND18_TRANSPORT_WALK),
            "sha256": ROUND18_TRANSPORT_WALK_SHA256,
            "commit": round18["worktree"]["commit"],
            "all_hpg_and_advection_rows_bit_exact": isolation_control,
            "q_plus_Fu_source_accumulator_unequal_words": 0,
        },
        "live_formed_vs_scalar_nemo": {
            "q": wordwise_result(live_q, scalar_q),
            "F_u": wordwise_result(live_fu, scalar_fu),
        },
        "factor_providers_live_vs_scalar_nemo": {
            "curl_Kmm_u": wordwise_result(live_u, scalar_u),
            "curl_Kmm_v": wordwise_result(live_v, scalar_v),
            "coriolis_ff_f": wordwise_result(live_f, scalar_f),
            "live_F_thickness": wordwise_result(live_hf, scalar_hf),
            "e2u_metric": wordwise_result(live_e2u, scalar_e2u),
            "live_U_thickness": wordwise_result(live_hu, scalar_hu),
        },
        "arms": arms,
        "sufficient_single_factors": sufficient,
        "owner_boundary": boundary,
        "plant": None,
        "source_citations": [
            "compiled stprk3.f90:204-208",
            "compiled stprk3_stg.f90:404-430",
            "compiled dynvor.f90:552-591",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, default=VECTOR_ROOT)
    parser.add_argument("--scalar-root", type=Path, default=SCALAR_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor-walk", action="store_true")
    plants = parser.add_mutually_exclusive_group()
    plants.add_argument("--plant-q", action="store_true")
    plants.add_argument("--plant-fu", action="store_true")
    args = parser.parse_args()
    runner = run_factor_walk if args.factor_walk else run
    report = runner(
        args.vector_root, args.scalar_root,
        plant_q=args.plant_q, plant_fu=args.plant_fu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if report["plant"] is not None:
        print(
            ("GYRE_R19_FORMED_FACTOR" if args.factor_walk
             else "GYRE_R18_STAGE3_VORTICITY")
            + " PLANT_FIRED operand="
            + report["plant"] + " changed="
            + str(report["plant_result"]["registered_changed_words"])
        )
    elif args.factor_walk:
        print(
            "GYRE_R19_FORMED_FACTOR " + report["owner_boundary"]
            + " sufficient=" + ",".join(
                report["sufficient_single_factors"])
            + " live_q_unequal=" + str(
                report["live_formed_vs_scalar_nemo"]["q"]["unequal_words"])
            + " live_Fu_unequal=" + str(
                report["live_formed_vs_scalar_nemo"]["F_u"]["unequal_words"])
        )
    else:
        print(
            "GYRE_R18_STAGE3_VORTICITY " + report["owner_boundary"]
            + " exact_subsets=" + ",".join(report["exact_operand_subsets"])
            + " q_Fu_unequal=" + str(
                report["arms"]["q_plus_Fu"]["source_accumulator"]
                ["registered_unequal_words"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

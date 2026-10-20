#!/usr/bin/env python3
"""Locate the earliest common-state GYRE momentum stage boundary.

The admitted scalar/vector NEMO records and legoESM production-JIT trace all
start from the same day-180 restart.  This gate compares the completed stage-1
RHS, external-mode handoff, and corrected stage outputs before asking for a
new source-ordered contributor record.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from nemo_testcase_l2_gyre_hires_round17_vorticity import (
    COMMON_RESTART,
    COMMON_RESTART_SHA256,
    NATIVE_DIMS,
    common_card,
    common_state,
    manifest_contract,
    sha256,
    wordwise_result,
)
from nemo_testcase_l2_gyre_phase3_gate import (
    expected_masks,
    read_stage,
    require,
)


VECTOR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_vector"
)
SCALAR_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round17/oracle_gyre_r4_smag_stage3_vorticity_scalar"
)
RHS_RECORD = "oracle_rhs_kt00000001.bin"
BT_RECORD = "oracle_bt_frames_kt00004321.bin"
RHS_FAMILY_RECORD = "oracle_r20_stage1_rhs_kt00004321.bin"
AHM_RECORD = "oracle_ahm_kt00004321.bin"
STEP = 4321
ACTIVE_LEVELS = 30
TWIN_CLEAR_FACTOR = 4.0
FACE_ORDER = ("u", "v")
SOURCE_BOUNDARIES = (
    ("after_hpg", "hydrostatic_pressure_gradient"),
    ("after_ldf", "lateral_diffusion"),
    ("after_vor", "total_vorticity_coriolis"),
    ("after_keg", "kinetic_energy_gradient"),
    ("after_adv", "vertical_advection"),
)


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[
        2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_completed_rhs(path: Path) -> dict[str, object]:
    """Read the stage-1 3-D RHS emitted immediately after ``stp_2D``."""
    raw = path.read_bytes()
    require(len(raw) >= 44, f"{path}: short RHS record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=7i", raw[16:44])
    keys = ("version", "kt", "Krhs", "nx", "ny", "nz", "bits")
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["nz"], header["bits"])
        == (1, STEP, *NATIVE_DIMS, 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=44)
    count = header["nx"] * header["ny"] * header["nz"]
    require(values.size == 2 * count, f"{path}: malformed RHS payload")
    fields = {
        "u": _xyz(values[:count], header["nx"], header["ny"], header["nz"]),
        "v": _xyz(values[count:], header["nx"], header["ny"], header["nz"]),
    }
    for name, field in fields.items():
        require(np.all(np.isfinite(field[..., :ACTIVE_LEVELS])),
                f"{path}: non-finite active {name} RHS word")
    return {"header": header, **fields}


def read_bt_handoff(path: Path) -> dict[str, object]:
    """Read the split-explicit output consumed by every RK stage."""
    raw = path.read_bytes()
    require(len(raw) >= 40, f"{path}: short barotropic record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=6i", raw[16:40])
    keys = ("version", "kt", "Kaa", "nx", "ny", "bits")
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L1_BTFRM_1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["bits"])
        == (1, STEP, NATIVE_DIMS[0], NATIVE_DIMS[1], 64),
        f"{path}: bad header {header}",
    )
    values = np.frombuffer(raw, dtype=np.float64, offset=40)
    count = header["nx"] * header["ny"]
    require(values.size == 4 * count, f"{path}: malformed barotropic payload")
    names = ("uu_b", "vv_b", "un_adv", "vn_adv")
    fields = {
        name: _xy(values[index * count:(index + 1) * count],
                  header["nx"], header["ny"])
        for index, name in enumerate(names)
    }
    for name, field in fields.items():
        require(np.all(np.isfinite(field)),
                f"{path}: non-finite active {name} word")
    return {"header": header, **fields}


def read_stage1_rhs_family(path: Path) -> dict[str, object]:
    """Read cumulative stage-1 RHS boundaries in executed NEMO order."""
    raw = path.read_bytes()
    require(len(raw) >= 88, f"{path}: short RHS-family record")
    magic = raw[:16].decode("ascii").rstrip()
    header_values = struct.unpack("=8i", raw[16:48])
    keys = ("version", "kt", "Kbb", "Krhs", "nx", "ny", "nz", "bits")
    header = dict(zip(keys, header_values, strict=True))
    require(magic == "NEMO_L2_R20RHS_1", f"{path}: bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["nx"], header["ny"],
         header["nz"], header["bits"])
        == (1, STEP, *NATIVE_DIMS, 64),
        f"{path}: bad header {header}",
    )
    sizes = struct.unpack("=10i", raw[48:88])
    count = header["nx"] * header["ny"] * header["nz"]
    require(sizes == (count,) * 10,
            f"{path}: bad array-size contract {sizes}")
    values = np.frombuffer(raw, dtype=np.float64, offset=88)
    require(values.size == 10 * count, f"{path}: malformed RHS-family payload")
    fields: dict[str, object] = {"header": header}
    cursor = 0
    for boundary, _term in SOURCE_BOUNDARIES:
        for face in FACE_ORDER:
            fields[f"{boundary}_{face}"] = _xyz(
                values[cursor:cursor + count], header["nx"],
                header["ny"], header["nz"])
            cursor += count
    for name, field in fields.items():
        if name == "header":
            continue
        require(np.all(np.isfinite(field[..., :ACTIVE_LEVELS])),
                f"{path}: non-finite active {name} RHS word")
    return fields


def envelope_score(
    candidate: np.ndarray,
    scalar: np.ndarray,
    vector: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    """Score exact words and the preregistered four-times twin maximum."""
    candidate = np.asarray(candidate, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    require(candidate.shape == scalar.shape == vector.shape == active.shape,
            "provider shapes differ")
    require(np.any(active), "comparison mask is empty")
    candidate_row = wordwise_result(candidate[active], scalar[active])
    twin_row = wordwise_result(scalar[active], vector[active])
    candidate_max = float(candidate_row["max_abs"])
    twin_max = float(twin_row["max_abs"])
    if twin_max == 0.0:
        ratio = None if candidate_max == 0.0 else float("inf")
        clears = candidate_max > 0.0
    else:
        ratio = candidate_max / twin_max
        clears = ratio >= TWIN_CLEAR_FACTOR
    return {
        "legoesm_minus_scalar": candidate_row,
        "scalar_minus_vector": twin_row,
        "candidate_to_twin_max_ratio": ratio,
        "twin_clear_factor": TWIN_CLEAR_FACTOR,
        "clears_four_times_twin": bool(clears),
    }


def first_clearing_stage(stages: list[dict[str, object]]) -> dict | None:
    """Return the first stage/face row clearing the registered envelope."""
    for stage in stages:
        rows = stage["rows"]
        for face in FACE_ORDER:
            if rows[face]["clears_four_times_twin"]:
                return {"stage": stage["stage"], "field": face,
                        "score": rows[face]}
    return None


def first_clearing_source_boundary(
    boundaries: list[dict[str, object]],
) -> dict | None:
    """Return the first cumulative source boundary outside the twin floor."""
    for boundary in boundaries:
        for face in FACE_ORDER:
            if boundary["rows"][face]["clears_four_times_twin"]:
                return {
                    "boundary": boundary["boundary"],
                    "term": boundary["term"],
                    "field": face,
                    "score": boundary["rows"][face],
                }
    return None


def coefficient_factor_owner(arms: dict[str, dict[str, object]]) -> str:
    """Classify only an exact, upstream-preserving coefficient substitution."""
    single_exact = [
        name for name in ("ahmt", "ahmf")
        if arms[name]["untouched_hpg_bit_exact"]
        and arms[name]["after_ldf_bit_exact"]
    ]
    pair_exact = (
        arms["ahmt_ahmf"]["untouched_hpg_bit_exact"]
        and arms["ahmt_ahmf"]["after_ldf_bit_exact"])
    if len(single_exact) == 1:
        return f"COEFFICIENT_{single_exact[0].upper()}"
    if single_exact:
        return "COEFFICIENT_NON_UNIQUE"
    if pair_exact:
        return "COUPLED_COEFFICIENT_FIELDS"
    return "COEFFICIENT_EXONERATED_OPERATOR_WALK_REQUIRED"


def _face_fields(values: tuple) -> dict[str, np.ndarray]:
    return {
        "u": np.asarray(values[0], dtype=np.float64)[:, 1:, :],
        "v": np.asarray(values[1], dtype=np.float64)[1:, :, :],
    }


def _active_levels(field: np.ndarray) -> np.ndarray:
    """Drop NEMO's closed jpk slot; legoESM stores the 30 active levels."""
    field = np.asarray(field, dtype=np.float64)
    require(field.ndim == 3 and field.shape[-1] == NATIVE_DIMS[-1],
            f"oracle field has unexpected shape {field.shape}")
    return field[..., :ACTIVE_LEVELS]


def _trace_source_boundary(operands: dict, boundary: str) -> dict[str, np.ndarray]:
    """Return native legoESM face arrays from production accumulator barriers."""
    return {
        "u": np.asarray(operands[f"{boundary}_u"].data,
                        dtype=np.float64)[:, 1:, :],
        "v": np.asarray(operands[f"{boundary}_v"].data,
                        dtype=np.float64)[1:, :, :],
    }


def _plant_one_ulp(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    planted = np.array(field, dtype=np.float64, copy=True)
    index = tuple(np.argwhere(mask)[0])
    direction = np.inf if planted[index] >= 0.0 else -np.inf
    planted[index] = np.nextafter(planted[index], direction)
    return planted


def _state_bytes_equal(left, right) -> tuple[bool, int]:
    import jax

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    require(left_tree == right_tree, "trace and ordinary pytrees differ")
    mismatches = 0
    for candidate, reference in zip(left_leaves, right_leaves, strict=True):
        candidate = np.ascontiguousarray(np.asarray(candidate))
        reference = np.ascontiguousarray(np.asarray(reference))
        require(candidate.shape == reference.shape
                and candidate.dtype == reference.dtype,
                "trace and ordinary state leaf shape/dtype differs")
        mismatches += int(np.count_nonzero(
            candidate.view(np.uint8) != reference.view(np.uint8)))
    return mismatches == 0, mismatches


def run(
    vector_root: Path,
    scalar_root: Path,
    *,
    rhs_family_vector_root: Path | None = None,
    rhs_family_scalar_root: Path | None = None,
    ahm_vector_root: Path | None = None,
    ahm_scalar_root: Path | None = None,
    plant: str = "none",
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "round-20 gate requires CPU")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "round-20 policy is not certified fp64/libm")
    require(sha256(COMMON_RESTART) == COMMON_RESTART_SHA256,
            "common developed restart moved")
    stamp = worktree_stamp()
    require(stamp["clean"],
            f"round-20 stage walk requires clean tree: {stamp['dirty_paths']}")

    providers = {
        "vector": manifest_contract(vector_root, "vector-math-conda"),
        "scalar": manifest_contract(scalar_root, "scalar-math-conda"),
    }
    require((rhs_family_vector_root is None)
            == (rhs_family_scalar_root is None),
            "both scalar and vector RHS-family providers are required")
    if rhs_family_vector_root is not None:
        providers["rhs_family_vector"] = manifest_contract(
            rhs_family_vector_root, "vector-math-conda")
        providers["rhs_family_scalar"] = manifest_contract(
            rhs_family_scalar_root, "scalar-math-conda")
    card = common_card()
    state = common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, STEP)
    trace = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    ).step(state, dt=card.dt_s, freshwater=freshwater,
           surface_forcing=surface)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
    ).step(state, dt=card.dt_s, freshwater=freshwater,
           surface_forcing=surface)
    trace = jax.device_get(trace)
    ordinary = jax.device_get(ordinary)
    passive, passive_bytes = _state_bytes_equal(trace.state_after, ordinary)
    require(passive, f"live-stage trace moved {passive_bytes} state bytes")

    # NEMO's stage-1 accumulator order is HPG -> LDF -> VOR -> KEG -> ZAD
    # (compiled stp2d.f90:159-190).  legoESM's ordinary tendency evaluator
    # forms the same isolated terms in a different order.  Use the existing
    # private Round-119 association arm for the cumulative comparison; retain
    # the ordinary trace above as the independent passivity control.
    def source_order_step(coefficient_override=None):
        return LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                nemo_stage_rhs_accumulation_order_arm=(
                    "stage1-source-order", None),
                smagorinsky_coefficient_override=coefficient_override,
            ),
        ).step(state, dt=card.dt_s, freshwater=freshwater,
               surface_forcing=surface)

    require((ahm_vector_root is None) == (ahm_scalar_root is None),
            "both scalar and vector AHM providers are required")
    require(ahm_vector_root is None or rhs_family_vector_root is not None,
            "AHM factor walk requires the cumulative RHS providers")
    source_order_trace = None
    source_order_state_identity = None
    if rhs_family_vector_root is not None:
        source_order_trace = source_order_step()
        source_order_trace = jax.device_get(source_order_trace)
        arm_exact, arm_bytes = _state_bytes_equal(
            source_order_trace.state_after, ordinary)
        source_order_state_identity = {
            "bit_exact": arm_exact,
            "mismatched_bytes": arm_bytes,
        }

    masks = expected_masks(card)
    planted_control = None
    stages = []
    for stage_index in (1, 2):
        scalar = read_stage(
            scalar_root / "instrumented"
            / f"oracle_stage_kt{STEP:08d}_s{stage_index}.bin",
            stage_index, NATIVE_DIMS, expected_kt=STEP)
        vector = read_stage(
            vector_root / "instrumented"
            / f"oracle_stage_kt{STEP:08d}_s{stage_index}.bin",
            stage_index, NATIVE_DIMS, expected_kt=STEP)
        live = _face_fields(trace.stage_outputs[stage_index - 1])
        rows = {}
        for face in FACE_ORDER:
            candidate = live[face]
            scalar_active = _active_levels(scalar[face])
            if plant == f"stage{stage_index}-{face}":
                candidate = _plant_one_ulp(scalar_active, masks[face])
            rows[face] = envelope_score(
                candidate, scalar_active,
                _active_levels(vector[face]), masks[face])
            if plant == f"stage{stage_index}-{face}":
                planted_control = rows[face]
        stages.append({"stage": stage_index, "rows": rows})

    scalar_rhs = read_completed_rhs(
        scalar_root / "instrumented" / RHS_RECORD)
    vector_rhs = read_completed_rhs(
        vector_root / "instrumented" / RHS_RECORD)
    live_rhs = _face_fields(trace.stage1_full_rhs)
    rhs_rows = {}
    for face in FACE_ORDER:
        candidate = live_rhs[face]
        scalar_active = _active_levels(scalar_rhs[face])
        if plant == f"rhs-{face}":
            candidate = _plant_one_ulp(scalar_active, masks[face])
        rhs_rows[face] = envelope_score(
            candidate, scalar_active,
            _active_levels(vector_rhs[face]), masks[face])
        if plant == f"rhs-{face}":
            planted_control = rhs_rows[face]

    scalar_bt = read_bt_handoff(
        scalar_root / "instrumented" / BT_RECORD)
    vector_bt = read_bt_handoff(
        vector_root / "instrumented" / BT_RECORD)
    live_bt = {
        "u": np.asarray(trace.barotropic_targets[0], dtype=np.float64)[:, 1:],
        "v": np.asarray(trace.barotropic_targets[1], dtype=np.float64)[1:, :],
    }
    bt_rows = {}
    for face, name in (("u", "uu_b"), ("v", "vv_b")):
        active = masks[face][..., 0]
        candidate = live_bt[face]
        if plant == f"bt-{face}":
            candidate = _plant_one_ulp(scalar_bt[name], active)
        bt_rows[face] = envelope_score(
            candidate, scalar_bt[name], vector_bt[name], active)
        if plant == f"bt-{face}":
            planted_control = bt_rows[face]

    source_boundaries = None
    first_source_boundary = None
    source_order_calibration = None
    coefficient_factor_walk = None
    if rhs_family_vector_root is not None:
        require(source_order_trace is not None,
                "source-order trace was not materialized")
        scalar_family = read_stage1_rhs_family(
            rhs_family_scalar_root / "instrumented" / RHS_FAMILY_RECORD)
        vector_family = read_stage1_rhs_family(
            rhs_family_vector_root / "instrumented" / RHS_FAMILY_RECORD)
        live_operands = source_order_trace.operator_operands[0]
        live_final = _trace_source_boundary(live_operands, "after_adv")
        source_order_rhs = _face_fields(source_order_trace.stage1_full_rhs)
        source_order_calibration = {}
        for face in FACE_ORDER:
            active = masks[face]
            source_order_calibration[face] = {
                "scalar_final_equals_inherited_completed_rhs": wordwise_result(
                    _active_levels(scalar_family[f"after_adv_{face}"])[active],
                    _active_levels(scalar_rhs[face])[active]),
                "vector_final_equals_inherited_completed_rhs": wordwise_result(
                    _active_levels(vector_family[f"after_adv_{face}"])[active],
                    _active_levels(vector_rhs[face])[active]),
                "legoesm_source_order_final_equals_source_order_rhs": (
                    wordwise_result(
                        live_final[face][active],
                        source_order_rhs[face][active])),
            }
        require(all(
            row[check]["bit_exact"]
            for row in source_order_calibration.values()
            for check in row
        ), "source-ordered final frame does not close to completed RHS")
        source_boundaries = []
        for boundary, term in SOURCE_BOUNDARIES:
            live = _trace_source_boundary(live_operands, boundary)
            rows = {}
            for face in FACE_ORDER:
                candidate = live[face]
                scalar_active = _active_levels(
                    scalar_family[f"{boundary}_{face}"])
                if plant == f"{boundary}-{face}":
                    candidate = _plant_one_ulp(scalar_active, masks[face])
                rows[face] = envelope_score(
                    candidate,
                    scalar_active,
                    _active_levels(vector_family[f"{boundary}_{face}"]),
                    masks[face],
                )
                if plant == f"{boundary}-{face}":
                    planted_control = rows[face]
            source_boundaries.append({
                "boundary": boundary,
                "term": term,
                "rows": rows,
            })
        first_source_boundary = first_clearing_source_boundary(
            source_boundaries)

        if ahm_vector_root is not None:
            import jax.numpy as jnp
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                compute_vertex_mask,
                nemo_smagorinsky_bilaplacian_coefficients_cgrid,
            )
            from nemo_testcase_l2_gyre_hires_ladder_gate import read_ahm

            providers["ahm_vector"] = manifest_contract(
                ahm_vector_root, "vector-math-conda")
            providers["ahm_scalar"] = manifest_contract(
                ahm_scalar_root, "scalar-math-conda")
            scalar_ahm = read_ahm(
                ahm_scalar_root / "instrumented" / AHM_RECORD,
                expected_kt=STEP,
            )
            vector_ahm = read_ahm(
                ahm_vector_root / "instrumented" / AHM_RECORD,
                expected_kt=STEP,
            )
            cell3 = (
                np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
                * np.asarray(state.land_mask.data)[..., None]
            )
            coefficient_vertex_mask = jax.vmap(
                lambda level: compute_vertex_mask(
                    level, grid=card.recipe.grid),
                in_axes=-1, out_axes=-1,
            )(jnp.asarray(cell3))
            live_ahmt, live_ahmf = (
                nemo_smagorinsky_bilaplacian_coefficients_cgrid(
                    state.u.data, state.v.data, card.recipe.grid,
                    card.recipe.model_config.lateral_viscosity.C_smag,
                    card.dt_s, mask=state.land_mask.data,
                    cell_mask=jnp.asarray(cell3),
                    u_mask=state.u_mask.data, v_mask=state.v_mask.data,
                    vertex_mask=coefficient_vertex_mask,
                )
            )
            live_ahmt, live_ahmf = jax.device_get((live_ahmt, live_ahmf))
            scalar_pair = (
                np.asarray(scalar_ahm["ahmt"]),
                np.asarray(scalar_ahm["ahmf"]),
            )
            vector_pair = (
                np.asarray(vector_ahm["ahmt"]),
                np.asarray(vector_ahm["ahmf"]),
            )
            live_pair = (np.asarray(live_ahmt), np.asarray(live_ahmf))
            coefficient_rows = {
                name: envelope_score(
                    live_pair[index], scalar_pair[index], vector_pair[index],
                    np.ones_like(scalar_pair[index], dtype=bool),
                )
                for index, name in enumerate(("ahmt", "ahmf"))
            }
            arms = {}
            for name, override in (
                ("ahmt", (scalar_pair[0], live_pair[1])),
                ("ahmf", (live_pair[0], scalar_pair[1])),
                ("ahmt_ahmf", scalar_pair),
            ):
                factor_trace = jax.device_get(source_order_step(
                    tuple(jnp.asarray(value) for value in override)))
                factor_operands = factor_trace.operator_operands[0]
                rows = {}
                for boundary in ("after_hpg", "after_ldf"):
                    live_boundary = _trace_source_boundary(
                        factor_operands, boundary)
                    rows[boundary] = {
                        face: envelope_score(
                            live_boundary[face],
                            _active_levels(
                                scalar_family[f"{boundary}_{face}"]),
                            _active_levels(
                                vector_family[f"{boundary}_{face}"]),
                            masks[face],
                        )
                        for face in FACE_ORDER
                    }
                arms[name] = {
                    "rows": rows,
                    "untouched_hpg_bit_exact": all(
                        rows["after_hpg"][face]["legoesm_minus_scalar"]
                        ["bit_exact"] for face in FACE_ORDER),
                    "after_ldf_bit_exact": all(
                        rows["after_ldf"][face]["legoesm_minus_scalar"]
                        ["bit_exact"] for face in FACE_ORDER),
                }
            coefficient_factor_walk = {
                "input_time_level": "Nbb before velocity",
                "coefficient_rows": coefficient_rows,
                "arms": arms,
                "owner": coefficient_factor_owner(arms),
            }

    first_stage = first_clearing_stage(stages)
    rhs_clears = any(
        rhs_rows[face]["clears_four_times_twin"] for face in FACE_ORDER)
    bt_clears = any(
        bt_rows[face]["clears_four_times_twin"] for face in FACE_ORDER)
    if first_stage is None:
        next_boundary = "NO_STAGE_BOUNDARY_CLEARS_TWIN"
    elif first_stage["stage"] == 1 and rhs_clears and source_boundaries is None:
        next_boundary = "STAGE1_SOURCE_ORDERED_RHS_ACQUISITION_REQUIRED"
    elif first_stage["stage"] == 1 and rhs_clears:
        require(first_source_boundary is not None,
                "completed RHS clears but no cumulative source boundary does")
        next_boundary = "STAGE1_FACTOR_WALK_REQUIRED"
    elif first_stage["stage"] == 1 and bt_clears:
        next_boundary = "STAGE1_EXTERNAL_MODE_WALK_REQUIRED"
    elif first_stage["stage"] == 1:
        next_boundary = "STAGE1_RAW_UPDATE_AND_CORRECTION_RECORD_REQUIRED"
    else:
        next_boundary = "STAGE2_SOURCE_ORDERED_RHS_ACQUISITION_REQUIRED"

    if plant != "none":
        require(planted_control is not None,
                "selected one-ULP plant was not evaluated")
        require(planted_control["legoesm_minus_scalar"]["unequal_words"] == 1,
                "active one-ULP plant did not isolate exactly one word")

    return {
        "format": "gyre-r4-smag-round20-stage-boundary-v1",
        "worktree": stamp,
        "common_restart": {
            "path": str(COMMON_RESTART),
            "sha256": COMMON_RESTART_SHA256,
            "step": STEP - 1,
            "next_step": STEP,
        },
        "policy": "fp64-libm-certified-cpu",
        "jax_backend": jax.default_backend(),
        "providers": providers,
        "trace_passivity": {
            "bit_exact": passive,
            "mismatched_bytes": passive_bytes,
        },
        "source_order_arm_vs_ordinary_state": source_order_state_identity,
        "plant": plant,
        "plant_control": planted_control,
        "stages": stages,
        "completed_stage1_rhs": rhs_rows,
        "source_ordered_stage1_rhs": source_boundaries,
        "source_order_calibration": source_order_calibration,
        "first_source_boundary_beyond_twin": first_source_boundary,
        "coefficient_factor_walk": coefficient_factor_walk,
        "external_mode_handoff": bt_rows,
        "first_stage_beyond_twin": first_stage,
        "next_boundary": next_boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, default=VECTOR_ROOT)
    parser.add_argument("--scalar-root", type=Path, default=SCALAR_ROOT)
    parser.add_argument("--rhs-family-vector-root", type=Path)
    parser.add_argument("--rhs-family-scalar-root", type=Path)
    parser.add_argument("--ahm-vector-root", type=Path)
    parser.add_argument("--ahm-scalar-root", type=Path)
    parser.add_argument("--plant", default="none", choices=(
        "none", "stage1-u", "stage1-v", "stage2-u", "stage2-v",
        "rhs-u", "rhs-v", "bt-u", "bt-v",
        "after_hpg-u", "after_hpg-v", "after_ldf-u", "after_ldf-v",
        "after_vor-u", "after_vor-v", "after_keg-u", "after_keg-v",
        "after_adv-u", "after_adv-v",
    ))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.vector_root,
        args.scalar_root,
        rhs_family_vector_root=args.rhs_family_vector_root,
        rhs_family_scalar_root=args.rhs_family_scalar_root,
        ahm_vector_root=args.ahm_vector_root,
        ahm_scalar_root=args.ahm_scalar_root,
        plant=args.plant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    print(json.dumps({
        "first_stage_beyond_twin": report["first_stage_beyond_twin"],
        "first_source_boundary_beyond_twin": (
            report["first_source_boundary_beyond_twin"]),
        "next_boundary": report["next_boundary"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

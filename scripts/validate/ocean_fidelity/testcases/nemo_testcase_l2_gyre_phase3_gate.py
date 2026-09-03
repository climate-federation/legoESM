#!/usr/bin/env python3
"""Fail-closed GYRE whole-step WS-RK3 trajectory certification gate.

This is lane 2's composition of lane 1 machinery: the binary record format,
the central time-level registry, fp64 pointwise scoring, scaling-first private
one-variable arms, first-over-bar retention, and the shared log-log growth
instrument.  A scientific DEBT is an expected exit status, never a harness
failure and never permission to assign an unsupported owner.
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

BAR = 1.0e-15
CASE = "GYRE-zco"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10")
DIMS = (36, 26, 31)
LEVELS = {1: {"Kaa": 3, "Kmm": 1}, 2: {"Kaa": 2, "Kmm": 3}, 3: {"Kaa": 3, "Kmm": 2}}
BIT_IDENTITY_EXPECTED = {
    "oracle_step_entry_kt00000001.bin": (
        "9def5f4986853f497a5e0507bea185fe1ec4348715e2aeca0b14507df8e24fa0"
    ),
    "oracle_step_entry_kt00000002.bin": (
        "887f3bbb51e047ee7c072ae74b77a8e5b461525341b1ab228ae7d800fa757777"
    ),
    "oracle_stage_kt00000001_s1.bin": (
        "35e6892b799aeaf8d06d4affcd71b5ba0c71dc41bc0e8970c033459c46cd1402"
    ),
    "oracle_stage_kt00000001_s2.bin": (
        "55e780b8d56eb249e5387123b20aeae8f735325200714c02a816ef719d6b3351"
    ),
    "oracle_stage_kt00000001_s3.bin": (
        "3703a9f2e369f8f05cc439564729e36801227b54eb8afb7df2552a5120a45f2e"
    ),
    "oracle_rhs_kt00000001.bin": "a09426f638de0ce384b739621d08cf7ae27d41339cadcc2f16f8bbd3de215c45",
    "oracle_bt_frames_kt00000001.bin": (
        "b8f474af46b665773152bd2f152d20f29f658b51a052422dcb35dc406fdf2fa9"
    ),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_namelist_blocks(path: Path) -> set[str]:
    """Enumerate the live NAMDYN/NAMZDF/NAMTRA groups from NEMO output.

    This is deliberately driven by the resolved runtime namelist rather than
    by this gate's disposition dictionary: a newly emitted fourteenth group
    must become an unaccounted DEBT row instead of remaining invisible.
    """
    blocks = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            token = line.lstrip()
            if not token.startswith("&"):
                continue
            name = token[1:].split(None, 1)[0].upper()
            if name.startswith(("NAMDYN", "NAMZDF", "NAMTRA")):
                blocks.add(name.lower())
    require(bool(blocks), f"{path}: no resolved NAMDYN/NAMZDF/NAMTRA blocks")
    return blocks


def resolved_program_coverage_rows(path: Path, checks: dict[str, bool]) -> list[dict]:
    """Return fail-closed rows for the union of runtime and disposition sets."""
    oracle_blocks = resolved_namelist_blocks(path)
    disposed_blocks = set(checks)
    rows = []
    for block in sorted(oracle_blocks | disposed_blocks):
        present = block in oracle_blocks
        disposed = block in disposed_blocks
        verified = present and disposed and bool(checks.get(block, False))
        row = {
            "name": f"resolved_program.{block}",
            "status": "VERIFIED" if verified else "DEBT",
            "runtime_namelist_present": present,
            "card_disposition_present": disposed,
        }
        if not present:
            row["reason"] = "static disposition has no resolved runtime block"
        elif not disposed:
            row["reason"] = "resolved runtime block has no card disposition"
        elif not checks[block]:
            row["reason"] = "resolved block's card selection failed verification"
        rows.append(row)
    return rows


def _registered(path: Path, expected: str) -> str:
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    observed = time_level_for_dump(path.name)
    require(observed == expected, f"{path}: registry {observed!r}, expected {expected!r}")
    return observed


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_entry(path: Path) -> dict:
    level = _registered(path, "before")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nbb, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require((version, nx, ny, nz, ntr, bits) == (1, *DIMS, 2, 64), f"{path}: bad header")
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")

    def xy(block):
        return block.reshape((nx, ny), order="F")[2:-2, 2:-2].T

    return {
        "kt": kt,
        "Nbb": nbb,
        "registry_level": level,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count : 2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count : 4 * count], nx, ny, nz),
        "ssh": xy(values[4 * count :]),
    }


def read_stage(path: Path, expected_stage: int) -> dict:
    level = _registered(path, "after")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kaa, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic")
    require((version, nx, ny, nz, ntr, bits) == (1, *DIMS, 2, 64), f"{path}: bad header")
    require(
        (kt, stage, kaa) == (1, expected_stage, LEVELS[expected_stage]["Kaa"]),
        f"{path}: wrong Kaa",
    )
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    return {
        "stage": stage,
        "Kaa": kaa,
        "registry_level": level,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count : 2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count : 4 * count], nx, ny, nz),
    }


def read_transport(path: Path, expected_stage: int) -> dict:
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_TRANSP_1", f"{path}: bad magic")
    require((version, nx, ny, nz, bits) == (1, *DIMS, 64), f"{path}: bad header")
    require(
        (kt, stage, kmm) == (1, expected_stage, LEVELS[expected_stage]["Kmm"]),
        f"{path}: wrong Kmm",
    )
    require(values.size == 3 * nx * ny * nz, f"{path}: bad payload")
    # NEMO does not own or initialize the four-cell transport halo.  Validate
    # every owned A2D cell while recording (not laundering) any halo sentinel.
    owned_finite = True
    for block in np.split(values, 3):
        full = block.reshape((nx, ny, nz), order="F")
        owned_finite &= bool(np.all(np.isfinite(full[2:-2, 2:-2])))
    require(owned_finite, f"{path}: non-finite owned payload")
    return {
        "stage": stage,
        "Kmm": kmm,
        "registry_level": level,
        "nonowned_nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
    }


def read_rhs(path: Path) -> dict:
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nrhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic")
    require((version, kt, nrhs, nx, ny, nz, bits) == (1, 1, 3, *DIMS, 64), f"{path}: bad header")
    require(values.size == 2 * nx * ny * nz, f"{path}: bad payload")
    return {"Nrhs": nrhs, "registry_level": level}


def read_bt(path: Path, expected_kt: int) -> dict:
    level = _registered(path, "after")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kaa, nx, ny, bits = header
    require(magic == "NEMO_L1_BTFRM_1", f"{path}: bad magic")
    expected_kaa = 3 if expected_kt % 2 else 1
    require(
        (version, kt, kaa, nx, ny, bits) == (1, expected_kt, expected_kaa, DIMS[0], DIMS[1], 64),
        f"{path}: bad header",
    )
    require(values.size == 4 * nx * ny, f"{path}: bad payload")
    return {"kt": kt, "Kaa": kaa, "registry_level": level}


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_zdf_entry(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, level, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_ZDF___1", f"{path}: bad magic")
    require((version, kt, nx, ny, nz, bits) == (1, 1, *DIMS, 64), f"{path}: bad header")
    avm_count = nx * ny * nz
    avt_count = (nx - 4) * (ny - 4) * nz
    require(values.size == avm_count + avt_count, f"{path}: bad payload")
    return {
        "level": level,
        # avm is a full-halo array; avt is NEMO's A2D interior allocation.
        "avm": _xyz(values[:avm_count], nx, ny, nz),
        "avt": values[avm_count:].reshape((nx - 4, ny - 4, nz), order="F").transpose(1, 0, 2),
    }


def read_qsr_stage3(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, krhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_QSR___1", f"{path}: bad magic")
    require(
        (version, kt, stage, nx, ny, nz, bits) == (1, 1, 3, *DIMS, 64),
        f"{path}: bad header",
    )
    count2 = (nx - 4) * (ny - 4)
    full_count3 = nx * ny * nz
    require(values.size == count2 + full_count3, f"{path}: bad payload")
    return {
        "Kmm": kmm,
        "Krhs": krhs,
        # qsr is A2D interior; the Krhs tracer increment is full-halo.
        "qsr": values[:count2].reshape((nx - 4, ny - 4), order="F").T,
        "dT_dt": _xyz(values[count2:], nx, ny, nz),
    }


# The solver returns ONE keyed barotropic substep frame (see
# barotropic_latlon_cgrid._run_substep_loop); this map is the mimicry-only glue
# from THIS oracle dump's field names to that frame's keys, and it lives in the
# harness, never in the model.  Only two names differ; every other name is
# shared with the L1-overflow registry verbatim.
BT_TRACE_KEY = {
    "transport_metric_u": "transport_u",
    "transport_metric_v": "transport_v",
}

ORACLE_BT_SUBSTEP_NAMES = (
    "eta_entry", "u_entry", "v_entry",
    "eta_mid", "u_mid", "v_mid",
    "eta_exit", "eta_pgf",
    "pgf_u", "pgf_v", "cor_u", "cor_v", "trd_u", "trd_v",
    "slow_u", "slow_v", "u_exit", "v_exit",
    "transport_metric_u", "transport_metric_v",
)


def read_bt_substeps(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        version, kt, ncycle, nx, ny, bits = header
        require(magic == "NEMO_L2_BTSUB_2", f"{path}: bad magic")
        require(
            (version, kt, ncycle, nx, ny, bits) == (2, 1, 50, DIMS[0], DIMS[1], 64),
            f"{path}: bad header",
        )
        records = {name: [] for name in ORACLE_BT_SUBSTEP_NAMES}
        count = nx * ny
        for expected in range(1, ncycle + 1):
            raw_jn = handle.read(4)
            require(len(raw_jn) == 4, f"{path}: truncated at substep {expected}")
            (jn,) = struct.unpack("=i", raw_jn)
            require(jn == expected, f"{path}: substep sequence {jn} != {expected}")
            for name in ORACLE_BT_SUBSTEP_NAMES:
                # zu_frc/zv_frc are A2D interior arrays; the other recurrence
                # operands retain their full two-halo allocation.
                if name in {"slow_u", "slow_v"}:
                    nvalue = (nx - 4) * (ny - 4)
                    values = np.fromfile(handle, dtype=np.float64, count=nvalue)
                    row = values.reshape((nx - 4, ny - 4), order="F").T
                else:
                    nvalue = count
                    values = np.fromfile(handle, dtype=np.float64, count=nvalue)
                    row = _xy(values, nx, ny)
                require(values.size == nvalue, f"{path}: truncated {name} payload")
                records[name].append(row)
        require(handle.read(1) == b"", f"{path}: trailing payload")
    return {"kt": kt, "ncycle": ncycle, **{name: np.stack(rows) for name, rows in records.items()}}


ENE_COEFFICIENT_NAMES = (
    "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
    "ffv_nw", "ffv_ne", "ffv_sw", "ffv_se",
)


def read_ene_coefficients(path: Path) -> dict:
    """Read the WRITE-only frozen-coefficient record from dyn_cor_2D_init."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kmm, nvor_scheme, nx, ny, bits = header
    require(magic == "NEMO_L2_ENECO_1", f"{path}: bad magic")
    require(
        (version, kt, kmm, nx, ny, bits) == (1, 1, 1, DIMS[0], DIMS[1], 64),
        f"{path}: bad header",
    )
    # ffu/ffv are A2D(0) interior allocations, unlike the full-halo recurrence
    # arrays.  The header retains global jpi/jpj for format consistency.
    interior_nx, interior_ny = nx - 4, ny - 4
    count = interior_nx * interior_ny
    require(values.size == len(ENE_COEFFICIENT_NAMES) * count, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "Kmm": kmm,
        "nvor_scheme": nvor_scheme,
        **{
            name: values[index * count : (index + 1) * count].reshape(
                (interior_nx, interior_ny), order="F"
            ).T
            for index, name in enumerate(ENE_COEFFICIENT_NAMES)
        },
    }


def expected_masks(card) -> dict:
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    v = active & np.roll(active, -1, axis=0)
    u[:, -1] = False
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}


def lego_fields(state) -> dict:
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data)[:, 1:, :],
        "v": np.asarray(state.v.data)[1:, :, :],
        "ssh": np.asarray(state.eta.data),
    }


def score(name: str, oracle, candidate, mask, *, plant=False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is {candidate.dtype}")
    require(bool(active.any()), f"{name}: empty mask")
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: non-finite candidate")
    absolute = float(np.max(np.abs(candidate[active] - oracle[active])))
    reference = float(np.max(np.abs(oracle[active])))
    normalized = absolute / max(reference, 1.0)
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "absolute_max": absolute,
        "reference_max_abs": reference,
        "normalized_max_abs": normalized,
        "bar": BAR,
        "n": int(active.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
        "relative_max_abs": absolute / reference if reference else None,
    }


def _mark_kt1_uninformative(row: dict, field: str, kt: int) -> dict:
    if kt == 1 and field in {"u", "v", "ssh"} and row["status"] == "AT-BAR":
        row["status"] = "UNINFORMATIVE"
        row["reason"] = {
            "u": (
                "at-rest initial U is identically zero; exactness cannot "
                "exercise momentum evolution"
            ),
            "v": (
                "at-rest initial V is identically zero; exactness cannot "
                "exercise momentum evolution"
            ),
            "ssh": (
                "at-rest initial SSH is identically zero; exactness cannot "
                "exercise barotropic evolution"
            ),
        }[field]
    return row


def _surface_forcings(card, state, kt: int):
    import jax.numpy as jnp
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_boundary_condition
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    sbc = gyre_surface_boundary_condition(card, kt * card.dt_s)
    sst = state.T.data[..., 0]
    # usrdef_sbc.F90:109-120,138-145: qns+qsr is the Haney term plus EMP
    # heat content, evaluated once from the entering Kbb/Nbb SST.
    q_total = -40.0 * (sst - sbc.t_star_c) - sbc.emp_kg_m2_s * sst * NEMO_CONSTANTS_CONFIG.c_sw
    surface = OceanSurfaceForcing(
        sw_down=sbc.qsr_w_m2,
        q_net=q_total,
        # usrdef_sbc's utau/vtau are already NEMO native-i/native-j ocean
        # components on the 45-degree grid.  The public forcing object carries
        # geographic atmospheric stress, so inverse-rotate to east/north and
        # reverse the reaction sign; surface_stress_faces then recovers exactly
        # the source-native pair instead of rotating it a second time.
        tau_x=-(
            card.recipe.grid.cos_alpha_u[:, 1:] * sbc.utau_pa
            - card.recipe.grid.sin_alpha_u[:, 1:] * sbc.vtau_pa
        ),
        tau_y=-(
            card.recipe.grid.sin_alpha_u[:, 1:] * sbc.utau_pa
            + card.recipe.grid.cos_alpha_u[:, 1:] * sbc.vtau_pa
        ),
    )
    zeros = jnp.zeros_like(sbc.emp_kg_m2_s)
    freshwater = FreshwaterForcing(
        precip=zeros,
        evap=sbc.emp_kg_m2_s,
        runoff=zeros,
        ice_fw=zeros,
        restoring=zeros,
    )
    return freshwater, surface


def _trajectory_growth(steps: list[dict]) -> dict:
    path = Path(__file__).with_name("nemo_testcase_phase3_trajectory_gate.py")
    spec = importlib.util.spec_from_file_location("lane1_phase3_trajectory", path)
    require(spec is not None and spec.loader is not None, "cannot load lane-1 growth instrument")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.characterize_growth(steps)


def validate_one_variable_arms(arms: dict, *, plant=False) -> list[dict]:
    rows = []
    for name, manifest in arms.items():
        changed = list(manifest["changed_operands"])
        if plant and name == "omit_stage_barotropic_correction":
            changed.append("second_illegal_operand")
        ok = len(changed) == 1
        rows.append(
            {
                "name": f"arm_manifest.{name}",
                "status": "VERIFIED" if ok else "DEBT",
                "changed_operands": changed,
            }
        )
    return rows


def _trace_native(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    if name.startswith("u_") or name.endswith("_u"):
        return values[:, :, 1:]
    if name.startswith("v_") or name.endswith("_v"):
        return values[:, 1:, :]
    return values


def score_causal_arm(name, control, faithful, oracle, masks, nlev) -> dict:
    """Scaling table first, then the honest causal label."""
    faithful_fields = lego_fields(faithful)
    control_fields = lego_fields(control)
    rows = {}
    movement_max = 0.0
    residual_max = 0.0
    improvement = False
    for field in ("T", "S", "u", "v", "ssh"):
        reference = oracle[field] if field == "ssh" else oracle[field][..., :nlev]
        active = masks[field]
        scale = max(float(np.max(np.abs(reference[active]))), 1.0)
        faithful_abs = float(
            np.max(np.abs(faithful_fields[field][active] - reference[active]))
        ) / scale
        movement = float(
            np.max(
                np.abs(
                    control_fields[field][active]
                    - faithful_fields[field][active]
                )
            )
        ) / scale
        arm_abs = float(
            np.max(np.abs(control_fields[field][active] - reference[active]))
        ) / scale
        rows[field] = {
            "faithful_residual": faithful_abs,
            "causal_movement": movement,
            "arm_residual": arm_abs,
            "movement_over_faithful_residual": movement / faithful_abs if faithful_abs else None,
        }
        movement_max = max(movement_max, movement)
        residual_max = max(residual_max, faithful_abs)
        improvement = improvement or arm_abs < faithful_abs
    if movement_max <= BAR:
        label = "CAUSAL_NEAR_NULL_AFTER_DIRECT_MATCH"
    elif improvement and movement_max >= 0.1 * residual_max:
        label = "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
    else:
        label = "CAUSAL_NONPRIMARY_AT_KT2"
    return {
        "name": name,
        "classification": "DIAGNOSTIC_ONE_VARIABLE_CAUSAL_ARM",
        "scaling_check_before_owner_label": True,
        "worst_faithful_residual": residual_max,
        "worst_causal_movement": movement_max,
        "fields": rows,
        "owner_label": label,
    }


def run(
    root: Path,
    *,
    max_step=10,
    plant_state=False,
    plant_registry=False,
    plant_arm=False,
    plant_coverage=False,
    plant_barotropic=False,
    plant_ene_coefficient=False,
) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_barotropic_coriolis,
        _nemo_literal_een_coefficients,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(1 <= max_step <= 10, "max_step must be in 1..10")

    card = build_nemo_testcase_card(CASE)
    # NEMO QCO represents E-P through changing volume, with sfx=0
    # (usrdef_sbc.F90:138-145).  The fix_eta_drift projection is legoESM's
    # required source-inclusive realization of that real-freshwater contract.
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    tke_cfg = cfg.physics.vertical_mixing.tke
    evd_cfg = cfg.physics.convection.enhanced_diffusion
    coverage_checks = {
        "namdyn_adv": (
            cfg.momentum_advection == "vector_invariant"
            and cfg.ke_gradient_scheme == "c2"
            and cfg.vertical_momentum_scheme == "nemo_advective"
        ),
        "namdyn_vor": cfg.vorticity_scheme == "ene_total",
        "namdyn_hpg": cfg.pgf_scheme == "nemo_sco",
        "namdyn_spg": (
            cfg.barotropic.barotropic_time_filter == "nemo_ab3am4"
            and cfg.barotropic.n_barotropic_substeps == 50
        ),
        "namdyn_ldf": (
            cfg.lateral_viscosity_operator == "nemo_div_curl"
            and cfg.lateral_viscosity_e3_weighting == "nemo_e3"
            and cfg.lateral_viscosity.A_h == 1.0e5
        ),
        "namtra_adv": cfg.tracer_advection == "fct2",
        "namtra_ldf": (
            cfg.gm_redi is not None
            and cfg.gm_redi.slope_scheme == "nemo_iso_lap"
            and cfg.gm_redi.kappa_Redi == 1000.0
        ),
        "namtra_eiv": cfg.gm_redi is not None and cfg.gm_redi.kappa_GM == 0.0,
        "namtra_qsr": (
            cfg.physics.shortwave_penetration.scheme == "jerlov_2band"
            and cfg.physics.shortwave_penetration.water_type == "I"
        ),
        "namtra_dmp": getattr(cfg, "tracer_damping", None) is None,
        "namtra_mle": cfg.physics.mle is None,
        "namzdf": (
            cfg.adaptive_implicit_vertadv is False
            and cfg.A_v == 0.0
            and cfg.K_v == 0.0
            and cfg.physics.vertical_mixing.scheme == "tke"
            and cfg.physics.convection.scheme == "enhanced_diffusion"
            and evd_cfg.K_conv == 100.0
            and evd_cfg.nu_conv == 100.0
        ),
        "namzdf_tke": (
            tke_cfg.prognostic
            and tke_cfg.c_k == 0.1
            and tke_cfg.c_eps == 0.7
            and tke_cfg.tke_mxl_choice == 3
            and tke_cfg.lc
            and tke_cfg.kappaM_min == 1.2e-4
            and tke_cfg.kappaH_min == 1.2e-5
            and tke_cfg.n2_eos_form == "teos10"
        ),
    }
    if plant_coverage:
        coverage_checks["namdyn_vor"] = False
    coverage_rows = resolved_program_coverage_rows(
        root / "output.namelist.dyn", coverage_checks
    )
    require(
        all(row["status"] == "VERIFIED" for row in coverage_rows),
        f"resolved-program coverage failure: {coverage_rows}",
    )
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels

    artifacts = {}
    zdf_path = root / "oracle_zdf_entry_kt00000001.bin"
    qsr_path = root / "oracle_qsr_stage3_kt00000001.bin"
    bt_substep_path = root / "oracle_bt_substeps_kt00000001.bin"
    ene_coefficient_path = root / "oracle_bt_ene_coeff_kt00000001.bin"
    for path in (zdf_path, qsr_path, bt_substep_path, ene_coefficient_path):
        require(path.is_file(), f"missing causal artifact {path}")
        artifacts[path.name] = sha256(path)
    zdf_entry = read_zdf_entry(zdf_path)
    qsr_stage3 = read_qsr_stage3(qsr_path)
    bt_substeps = read_bt_substeps(bt_substep_path)
    oracle_ene_coefficients = read_ene_coefficients(ene_coefficient_path)
    stages = {}
    transports = {}
    for stage in (1, 2, 3):
        stage_path = root / f"oracle_stage_kt00000001_s{stage}.bin"
        transport_path = root / f"oracle_transport_kt00000001_s{stage}.bin"
        require(stage_path.is_file() and transport_path.is_file(), f"missing stage {stage} records")
        stages[stage] = read_stage(stage_path, stage)
        transports[stage] = read_transport(transport_path, stage)
        artifacts[stage_path.name] = sha256(stage_path)
        artifacts[transport_path.name] = sha256(transport_path)
    rhs_path = root / "oracle_rhs_kt00000001.bin"
    rhs = read_rhs(rhs_path)
    artifacts[rhs_path.name] = sha256(rhs_path)
    bt = {}
    for kt in range(1, max_step + 1):
        path = root / f"oracle_bt_frames_kt{kt:08d}.bin"
        bt[kt] = read_bt(path, kt)
        artifacts[path.name] = sha256(path)

    registry_rows = []
    for stage in (1, 2, 3):
        observed_kmm = transports[stage]["Kmm"]
        if plant_registry and stage == 1:
            observed_kmm = 2
        ok = stages[stage]["Kaa"] == LEVELS[stage]["Kaa"] and observed_kmm == LEVELS[stage]["Kmm"]
        registry_rows.append(
            {
                "name": f"GYRE.kt1.stage{stage}.Kaa_Kmm_registry",
                "status": "VERIFIED" if ok else "DEBT",
                "observed": {"Kaa": stages[stage]["Kaa"], "Kmm": observed_kmm},
                "expected": LEVELS[stage],
            }
        )

    state = card.recipe.initial_state
    steps = []
    first_over_bar = None
    exact_prefix = True
    faithful_kt2 = None
    for kt in range(1, max_step + 1):
        path = root / f"oracle_step_entry_kt{kt:08d}.bin"
        oracle = read_entry(path)
        expected_nbb = 1 if kt % 2 else 3
        require(
            (oracle["kt"], oracle["Nbb"]) == (kt, expected_nbb),
            f"{path}: wrong Nbb",
        )
        artifacts[path.name] = sha256(path)
        candidate = lego_fields(state)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = oracle[field] if field == "ssh" else oracle[field][..., :nlev]
            row = score(
                f"{CASE}.kt{kt}.before.{field}",
                reference,
                candidate[field],
                masks[field],
                plant=plant_state and kt == 1 and field == "T",
            )
            rows.append(_mark_kt1_uninformative(row, field, kt))
        over = [row["name"].rsplit(".", 1)[-1] for row in rows if row["status"] == "DEBT"]
        exact_here = all(row["exact"] for row in rows)
        steps.append(
            {
                "kt": kt,
                "Nbb": oracle["Nbb"],
                "exact_prefix_entering": exact_prefix,
                "exact_at_step": exact_here,
                "rows": rows,
            }
        )
        exact_prefix = exact_prefix and exact_here
        if over and first_over_bar is None:
            first_over_bar = {"kt": kt, "fields": over}
        if kt < max_step:
            freshwater, surface = _surface_forcings(card, state, kt)
            state = model.step(state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
            if kt == 1:
                faithful_kt2 = state

    require(faithful_kt2 is not None or max_step == 1, "kt2 candidate was not produced")

    freshwater0, surface0 = _surface_forcings(card, card.recipe.initial_state, 1)
    seeded_entry = model._seed_tke_preclosure_carry(card.recipe.initial_state)
    model.prime_step_caches(seeded_entry)

    # C1: direct stage-entry TKE/EVD coefficients.  NEMO W levels 2..30
    # correspond to legoESM's 29 interior interfaces.
    lego_avt, lego_avm = model.diagnose_vertical_K(
        seeded_entry, card.dt_s, surface0
    )
    interface_mask = (
        np.asarray(card.recipe.z_coord.is_active)[..., 1:]
        & (np.asarray(seeded_entry.land_mask.data) > 0.5)[..., None]
    )
    direct_operand_rows = [
        score(
            f"{CASE}.kt1.zdf_entry.avt",
            zdf_entry["avt"][..., 1:30],
            np.asarray(lego_avt),
            interface_mask,
        ),
        score(
            f"{CASE}.kt1.zdf_entry.avm",
            zdf_entry["avm"][..., 1:30],
            np.asarray(lego_avm),
            interface_mask,
        ),
    ]

    # C2: isolate the card's penetrative-qsr tendency by one selector only.
    cfg_without_qsr = cfg._replace(
        physics=cfg.physics._replace(shortwave_penetration=None)
    )
    no_qsr_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_without_qsr
    )
    full_tendency = model.tendencies(seeded_entry, surface0, dt=card.dt_s)
    no_qsr_tendency = no_qsr_model.tendencies(
        seeded_entry, surface0, dt=card.dt_s
    )
    lego_qsr_tendency = np.asarray(
        full_tendency.dT_dt.data - no_qsr_tendency.dT_dt.data
    )
    oracle_qsr_tendency = qsr_stage3["dT_dt"][..., :nlev]
    direct_operand_rows.extend(
        [
            score(
                f"{CASE}.kt1.qsr.surface_flux",
                qsr_stage3["qsr"],
                np.asarray(surface0.sw_down),
                masks["ssh"],
            ),
            score(
                f"{CASE}.kt1.qsr.penetration_tendency",
                oracle_qsr_tendency,
                lego_qsr_tendency,
                masks["T"],
            ),
        ]
    )

    # B1: stop at the external-mode boundary and score every recurrence frame.
    trace = model._step_impl(
        seeded_entry,
        card.dt_s,
        freshwater=freshwater0,
        surface_forcing=surface0,
        _return_barotropic_substeps=True,
    )
    generic_cfg = cfg._replace(
        barotropic=cfg.barotropic._replace(
            barotropic_een_coefficient_evaluation="generic"
        )
    )
    generic_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, generic_cfg
    )
    generic_model.prime_step_caches(seeded_entry)
    generic_trace = generic_model._step_impl(
        seeded_entry,
        card.dt_s,
        freshwater=freshwater0,
        surface_forcing=surface0,
        _return_barotropic_substeps=True,
    )
    literal_trace = trace

    # E1: coefficient/state/product walk at the first live-ENE divergence.
    # At cold start jn=2 still has (za1,za2,za3)=(1,0,0), so its u_mid/v_mid
    # are the jn=1 exits.  Score that fact explicitly before coefficients.
    ene_state_rows = [
        {
            "name": f"{CASE}.kt1.bt.jn02.predictor_weights",
            "status": "AT-BAR",
            "oracle": [1.0, 0.0, 0.0],
            "candidate": [1.0, 0.0, 0.0],
            "source": "dynspg_ts.F90:552-572",
        }
    ]
    for component in ("u", "v"):
        ene_state_rows.append(
            score(
                f"{CASE}.kt1.bt.jn02.{component}_mid_from_jn01_exit",
                bt_substeps[f"{component}_exit"][0],
                _trace_native(
                    trace.substeps[f"{component}_mid"], f"{component}_mid",
                )[1],
                masks[component][..., 0],
            )
        )

    literal_coefficients = {
        name: np.asarray(value)
        for name, value in _nemo_literal_een_coefficients(
            card.recipe.initial_state.eta.data,
            card.recipe.z_coord,
            jnp.float64,
            scheme="ene",
        ).items()
    }
    coefficient_rows = []
    oracle_coeff_dict = {
        name: oracle_ene_coefficients[name] for name in ENE_COEFFICIENT_NAMES
    }
    for name in ENE_COEFFICIENT_NAMES:
        component = "u" if name.startswith("ffu") else "v"
        candidate = literal_coefficients[name]
        if plant_ene_coefficient and name == "ffu_nw":
            candidate = candidate.copy()
            first = tuple(np.argwhere(masks["u"][..., 0])[0])
            candidate[first] += 1.0
        coefficient_rows.append(
            score(
                f"{CASE}.kt1.bt.ene_coefficient.{name}",
                oracle_coeff_dict[name],
                candidate,
                masks[component][..., 0],
            )
        )
    ene_coefficient_control = {
        "name": "control.planted_ene_coefficient",
        "status": "NOT_REQUESTED",
    }
    if plant_ene_coefficient:
        planted_row = coefficient_rows[0]
        require(
            planted_row["status"] == "DEBT" and planted_row["absolute_max"] >= 1.0,
            "planted ENE coefficient did not fire at its registered magnitude",
        )
        ene_coefficient_control = {
            "name": "control.planted_ene_coefficient",
            "status": "VERIFIED",
            "observed_absolute_max": planted_row["absolute_max"],
            "expected_minimum": 1.0,
        }

    def full_faces(u_native, v_native):
        return (
            jnp.asarray(np.concatenate([u_native[:, -1:], u_native], axis=1)),
            jnp.asarray(np.concatenate([np.zeros_like(v_native[:1]), v_native], axis=0)),
        )

    oracle_u_mid, oracle_v_mid = full_faces(
        bt_substeps["u_mid"][1], bt_substeps["v_mid"][1]
    )
    candidate_u_mid = trace.substeps["u_mid"][1]
    candidate_v_mid = trace.substeps["v_mid"][1]
    oracle_cor_u, oracle_cor_v, oracle_terms = _nemo_literal_barotropic_coriolis(
        oracle_u_mid,
        oracle_v_mid,
        {name: jnp.asarray(value) for name, value in oracle_coeff_dict.items()},
        return_terms=True,
    )
    literal_cor_u, literal_cor_v, literal_terms = _nemo_literal_barotropic_coriolis(
        candidate_u_mid,
        candidate_v_mid,
        {name: jnp.asarray(value) for name, value in literal_coefficients.items()},
        return_terms=True,
    )
    product_rows = []
    for name in oracle_terms:
        component = name[0]
        product_rows.append(
            score(
                f"{CASE}.kt1.bt.jn02.ene_product.{name}",
                np.asarray(oracle_terms[name]),
                np.asarray(literal_terms[name]),
                masks[component][..., 0],
            )
        )
    reconstructed_term_rows = [
        score(
            f"{CASE}.kt1.bt.jn02.oracle_coefficient_reconstruction.trd_u",
            bt_substeps["cor_u"][1],
            np.asarray(oracle_cor_u)[:, 1:],
            masks["u"][..., 0],
        ),
        score(
            f"{CASE}.kt1.bt.jn02.oracle_coefficient_reconstruction.trd_v",
            bt_substeps["cor_v"][1],
            np.asarray(oracle_cor_v)[1:, :],
            masks["v"][..., 0],
        ),
        score(
            f"{CASE}.kt1.bt.jn02.literal_coefficient_arm.trd_u",
            bt_substeps["cor_u"][1],
            np.asarray(literal_cor_u)[:, 1:],
            masks["u"][..., 0],
        ),
        score(
            f"{CASE}.kt1.bt.jn02.literal_coefficient_arm.trd_v",
            bt_substeps["cor_v"][1],
            np.asarray(literal_cor_v)[1:, :],
            masks["v"][..., 0],
        ),
    ]
    causal_rows = []
    for component in ("u", "v"):
        native_literal = _trace_native(
            literal_trace.substeps[f"trd_{component}"],
            f"trd_{component}",
        )[1]
        causal_rows.append(
            score(
                f"{CASE}.kt1.bt.jn02.literal_selector_causal_arm.trd_{component}",
                bt_substeps[f"trd_{component}"][1],
                native_literal,
                masks[component][..., 0],
            )
        )
    generic_native_u = _trace_native(
        generic_trace.substeps["trd_u"], "trd_u"
    )[1]
    literal_native_u = _trace_native(
        literal_trace.substeps["trd_u"], "trd_u"
    )[1]
    active_u = masks["u"][..., 0]
    generic_residual = float(
        np.max(np.abs(generic_native_u[active_u] - bt_substeps["trd_u"][1][active_u]))
    )
    causal_movement = float(
        np.max(np.abs(literal_native_u[active_u] - generic_native_u[active_u]))
    )
    ene_causal_scaling = {
        "historical_generic_residual": generic_residual,
        "causal_movement": causal_movement,
        "movement_over_historical_generic_residual": (
            causal_movement / generic_residual if generic_residual else None
        ),
        "arm_residual": max(row["absolute_max"] for row in causal_rows),
        "ene_component_owner_label": (
            "CONFIRMED_CAUSAL_OWNER_OF_ENE_COMPONENT"
            if all(row["status"] == "AT-BAR" for row in coefficient_rows)
            and all(row["status"] == "AT-BAR" for row in product_rows)
            and all(row["status"] == "AT-BAR" for row in reconstructed_term_rows)
            else "UNMEASURED_ENE_COMPONENT"
        ),
        "combined_trd_owner_label": (
            "CONFIRMED_OWNER"
            if all(row["status"] == "AT-BAR" for row in causal_rows)
            else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
        ),
        "scaling_check_before_owner_label": True,
        "rows": causal_rows,
    }
    trace_order = (
        "eta_entry", "u_entry", "v_entry",
        "eta_mid", "u_mid", "v_mid",
        "slow_u", "slow_v", "eta_exit", "eta_pgf",
        "pgf_u", "pgf_v", "trd_u", "trd_v", "u_exit", "v_exit",
    )
    trace_masks = {
        "eta": masks["ssh"],
        "u": masks["u"][..., 0],
        "v": masks["v"][..., 0],
    }
    barotropic_rows = []
    barotropic_first_over_bar = None
    for substep in range(50):
        for name in trace_order:
            candidate = _trace_native(
                trace.substeps[BT_TRACE_KEY.get(name, name)], name)[substep]
            stagger = "u" if name.startswith("u_") or name.endswith("_u") else (
                "v" if name.startswith("v_") or name.endswith("_v") else "eta"
            )
            row = score(
                f"{CASE}.kt1.bt.jn{substep + 1:02d}.{name}",
                bt_substeps[name][substep],
                candidate,
                trace_masks[stagger],
                plant=(plant_barotropic and substep == 0 and name == "eta_entry"),
            )
            row["substep"] = substep + 1
            row["boundary"] = name
            barotropic_rows.append(row)
            if row["status"] == "DEBT" and barotropic_first_over_bar is None:
                barotropic_first_over_bar = {
                    "substep": substep + 1,
                    "boundary": name,
                    "absolute_max": row["absolute_max"],
                    "reference_max_abs": row["reference_max_abs"],
                }

    # The two dumped metric transports are registered but not silently divided
    # by a re-derived metric inside this gate.
    barotropic_unmeasured = [
        {
            "name": f"{CASE}.kt1.bt.{name}",
            "status": "UNMEASURED",
            "reason": (
                "oracle stores e2u/e1v metric transport; no independent "
                "metric operand was dumped"
            ),
        }
        for name in ("transport_metric_u", "transport_metric_v")
    ]

    stage_rows = []
    if max_step >= 2:
        for stage in (1, 2, 3):
            if stage == 3:
                stage_state = faithful_kt2
            else:
                stage_state = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    cfg,
                    _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_momentum_stage=stage),
                ).step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    freshwater=freshwater0,
                    surface_forcing=surface0,
                )
            fields = lego_fields(stage_state)
            for velocity in ("u", "v"):
                row = score(
                    f"{CASE}.kt1.stage{stage}.{velocity}",
                    stages[stage][velocity][..., :nlev],
                    fields[velocity],
                    masks[velocity],
                )
                row["frame"] = "instantaneous_prognostic_Kaa"
                stage_rows.append(row)
            for tracer in ("T", "S"):
                stage_rows.append(
                    {
                        "name": f"{CASE}.kt1.stage{stage}.{tracer}",
                        "status": "UNMEASURED",
                        "reason": (
                            "NEMO Kaa tracer artifact is present and registered, "
                            "but production legoESM exposes only momentum stage "
                            "states; no private tracer-stage surrogate is scored"
                        ),
                    }
                )

    arm_manifest = {
        "omit_stage_barotropic_correction": {
            "changed_operands": ["stage_barotropic_correction"],
            "hooks": _NEMOWSRK3TestHooks(stage_barotropic_correction=False),
        },
        "omit_momentum_transport_reconcile": {
            "changed_operands": ["momentum_transport_reconcile"],
            "hooks": _NEMOWSRK3TestHooks(momentum_transport_reconcile=False),
        },
        "omit_surface_boundary_forcing": {
            "changed_operands": ["surface_forcing"],
            "hooks": _NEMOWSRK3TestHooks(),
        },
        "omit_freshwater_forcing": {
            "changed_operands": ["freshwater"],
            "hooks": _NEMOWSRK3TestHooks(),
        },
    }
    arm_rows = validate_one_variable_arms(arm_manifest, plant=plant_arm)
    causal_manifest = {
        "oracle_stage_entry_avm_avt": {
            "changed_operands": ["vertical_diffusivity_profile_bundle"],
        },
        "oracle_stage3_qsr_penetration": {
            "changed_operands": ["penetrative_shortwave_tendency"],
        },
    }
    arm_rows.extend(validate_one_variable_arms(causal_manifest, plant=False))
    arms = {}
    causal_arms = {}
    operator_scaling = []
    if max_step >= 2:
        oracle2 = read_entry(root / "oracle_step_entry_kt00000002.bin")
        faithful_fields = lego_fields(faithful_kt2)

        oracle_avt = jnp.asarray(zdf_entry["avt"][..., 1:30])
        oracle_avm = jnp.asarray(zdf_entry["avm"][..., 1:30])
        vertical_control = model.step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
            _vertical_K_test_override=(oracle_avt, oracle_avm),
        )
        qsr_delta = jnp.asarray(oracle_qsr_tendency - lego_qsr_tendency)
        qsr_control = model.step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
            _shortwave_tendency_test_delta=qsr_delta,
        )
        causal_arms["oracle_stage_entry_avm_avt"] = score_causal_arm(
            "oracle_stage_entry_avm_avt",
            vertical_control,
            faithful_kt2,
            oracle2,
            masks,
            nlev,
        )
        causal_arms["oracle_stage3_qsr_penetration"] = score_causal_arm(
            "oracle_stage3_qsr_penetration",
            qsr_control,
            faithful_kt2,
            oracle2,
            masks,
            nlev,
        )

        for name, manifest in arm_manifest.items():
            arm_model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, cfg, _nemo_ws_test_hooks=manifest["hooks"]
            )
            if name == "omit_surface_boundary_forcing":
                control = arm_model.step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    freshwater=freshwater0,
                )
            elif name == "omit_freshwater_forcing":
                control = arm_model.step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    surface_forcing=surface0,
                )
            else:
                control = arm_model.step(
                    card.recipe.initial_state,
                    dt=card.dt_s,
                    freshwater=freshwater0,
                    surface_forcing=surface0,
                )
            control_fields = lego_fields(control)
            field_rows = {}
            movements = []
            residuals = []
            improves = []
            for field in ("T", "S", "u", "v", "ssh"):
                reference = oracle2[field] if field == "ssh" else oracle2[field][..., :nlev]
                row = score(
                    f"{CASE}.kt2.arm.{name}.{field}",
                    reference,
                    control_fields[field],
                    masks[field],
                )
                field_rows[field] = row
                active = masks[field]
                movement_abs = float(
                    np.max(np.abs(control_fields[field][active] - faithful_fields[field][active]))
                )
                scale = max(float(np.max(np.abs(reference[active]))), 1.0)
                movements.append(movement_abs / scale)
                faithful_abs = (
                    float(np.max(np.abs(faithful_fields[field][active] - reference[active])))
                    / scale
                )
                residuals.append(faithful_abs)
                improves.append(row["normalized_max_abs"] < faithful_abs)
            residual = max(residuals)
            movement = max(movements)
            clears = all(row["status"] == "AT-BAR" for row in field_rows.values())
            if clears:
                label = "CONFIRMED_OWNER"
            elif movement < 0.1 * residual or not any(improves):
                label = "REFUTED_AS_PRIMARY_OWNER"
            else:
                label = "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
            arms[name] = {
                "classification": "DIAGNOSTIC_ONE_VARIABLE_ARM",
                "changed_operand": manifest["changed_operands"][0],
                "scaling_check_before_owner_label": True,
                "faithful_worst_normalized_residual": residual,
                "arm_worst_normalized_movement": movement,
                "movement_over_faithful_residual": (
                    movement / residual if residual else float("inf")
                ),
                "owner_label": label,
                "field_rows": field_rows,
            }

        operator_arm_configs = {
            "momentum_scheme_identity": cfg._replace(
                momentum_advection="flux_form",
                # Renamed by the isomorphism branch's UP3 selector split
                # (9bbf9f7bd): the public "upwind3" resolved NEMO's
                # advected-velocity-pair rule (dynadv_up3.F90:166,169-172)
                # whenever momentum_time_integrator == "rk3_ws", which this
                # arm's base card is, and Oceananigans' transport rule
                # otherwise.  "nemo_up3" names that same NEMO rule, so this
                # control arm is unchanged in behaviour.
                momentum_flux_scheme="nemo_up3",
                vertical_momentum_scheme="nemo_up3",
                vorticity_scheme="al81",
                ke_gradient_scheme="centered",
                adaptive_implicit_vertadv=True,
                zad_bottom_face_mask="min_rule",
                zad_qco_evaluation="generic",
                wzv_call2_evaluation="generic",
            ),
            "momentum_level_laplacian": cfg._replace(
                lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0)
            ),
            "tracer_isoneutral_laplacian": cfg._replace(gm_redi=None),
            "tke_evd_background_identity": cfg._replace(
                physics=None, A_v=1.0e-4, K_v=0.0
            ),
            "two_band_shortwave": cfg._replace(
                physics=cfg.physics._replace(shortwave_penetration=None)
            ),
        }
        for name, arm_cfg in operator_arm_configs.items():
            control = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, arm_cfg
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater0,
                surface_forcing=surface0,
            )
            control_fields = lego_fields(control)
            for field in ("T", "S", "u", "v", "ssh"):
                reference = (
                    oracle2[field]
                    if field == "ssh"
                    else oracle2[field][..., :nlev]
                )
                active = masks[field]
                scale = max(float(np.max(np.abs(reference[active]))), 1.0)
                residual = float(
                    np.max(
                        np.abs(
                            faithful_fields[field][active] - reference[active]
                        )
                    )
                    / scale
                )
                term = float(
                    np.max(
                        np.abs(
                            control_fields[field][active]
                            - faithful_fields[field][active]
                        )
                    )
                    / scale
                )
                operator_scaling.append(
                    {
                        "operator_arm": name,
                        "field": field,
                        "faithful_residual": residual,
                        "term_magnitude": term,
                        "residual_over_term": (
                            residual / term if term else None
                        ),
                        "diagnostic_power": (
                            "NEAR-NULL_AT_KT2"
                            if name in {
                                "momentum_scheme_identity",
                                "momentum_level_laplacian",
                            }
                            else "SCALING_ONLY"
                        ),
                        "owner_label": "UNMEASURED_SCALING_ONLY",
                    }
                )

    instrumentation_identity_rows = []
    for name, expected in BIT_IDENTITY_EXPECTED.items():
        path = root / name
        require(path.is_file(), f"missing bit-identity control {path}")
        observed = sha256(path)
        instrumentation_identity_rows.append(
            {
                "name": f"instrumentation_bit_identity.{name}",
                "status": "VERIFIED" if observed == expected else "DEBT",
                "expected_sha256": expected,
                "observed_sha256": observed,
            }
        )
    if plant_barotropic:
        require(
            barotropic_first_over_bar == {
                "substep": 1,
                "boundary": "eta_entry",
                "absolute_max": 1.0,
                "reference_max_abs": 0.0,
            },
            "barotropic planted violation did not become the first boundary debt",
        )
    failed_controls = [
        row["name"]
        for row in registry_rows + arm_rows + instrumentation_identity_rows
        if row["status"] == "DEBT"
    ]
    require(not failed_controls, f"planted/control failure: {failed_controls}")
    status = "AT-BAR" if first_over_bar is None else "DEBT"
    return {
        "format": "nemo-testcase-l2-gyre-phase3-v2",
        "case": CASE,
        "status": status,
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "oracle_root": str(root),
        "max_step": max_step,
        "continue_after_first": True,
        "first_over_bar": first_over_bar,
        "selectors": {
            "eos": cfg.eos,
            "eos_depth": cfg.eos_depth,
            "momentum_time_integrator": cfg.momentum_time_integrator,
            "tracer_time_integrator": cfg.tracer_time_integrator,
            "rk3_ws_scheme_identity": "nemo_kmm+two_step_fct+stage_correction+transport_reconcile",
            "pgf": cfg.pgf_scheme,
            "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
            "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
            "barotropic_coriolis_split": cfg.barotropic_coriolis_split,
            "barotropic_coriolis": cfg.barotropic.barotropic_coriolis,
            "barotropic_ene_coefficient_evaluation": (
                cfg.barotropic.barotropic_een_coefficient_evaluation
            ),
            "freshwater_closure": cfg.freshwater_closure,
        },
        "full_stage_program": {
            "oracle_stage_headers": {
                str(stage): {
                    "Kaa": stages[stage]["Kaa"],
                    "Kmm": transports[stage]["Kmm"],
                }
                for stage in (1, 2, 3)
            },
            "rhs": rhs,
            "barotropic_frames": bt,
            "momentum_rows": stage_rows,
            "tracer_stage_numerics": "UNMEASURED",
            "barotropic_frame_numerics": "MEASURED_WITHIN_SUBSTEP_LOOP",
        },
        "direct_oracle_operands": direct_operand_rows,
        "barotropic_substep_boundary": {
            "resolved": {
                "nn_bt_flt": 3,
                "rn_bt_alpha": 0.07,
                "nn_e": 50,
                "rn_Dt_e_seconds": 288.0,
                "ln_bt_fw": True,
            },
            "first_over_bar": barotropic_first_over_bar,
            "rows": barotropic_rows,
            "unmeasured": barotropic_unmeasured,
        },
        "ene_operand_walk": {
            "state_and_weights": ene_state_rows,
            "coefficient_rows": coefficient_rows,
            "product_and_sum_rows": product_rows,
            "recorded_term_reconstruction": reconstructed_term_rows,
            "causal_scaling": ene_causal_scaling,
            "planted_control": ene_coefficient_control,
        },
        "instrumentation_bit_identity": instrumentation_identity_rows,
        "registry_rows": registry_rows,
        "resolved_program_coverage": coverage_rows,
        "arm_manifest_rows": arm_rows,
        "one_variable_arms": arms,
        "causal_oracle_injection_arms": causal_arms,
        "operator_scaling_before_owner": operator_scaling,
        "owner_verdict": (
            "ENE_COMPONENT_CONFIRMED; COMBINED_TRD_REMAINDER_UNMEASURED"
        ),
        "growth_characterization": _trajectory_growth(steps),
        "steps": steps,
        "artifacts": artifacts,
        "unmeasured": [
            "numerical T/S agreement at internal Kaa stages",
            "metric-weighted zhU/zhV transport comparison (metrics not dumped in this round)",
            "a two-sided source-isolated owner for the first over-bar whole-step row",
            "the bottom-drag operand owner of the post-ENE combined trd remainder",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=ROOT)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-state", action="store_true")
    parser.add_argument("--plant-registry", action="store_true")
    parser.add_argument("--plant-arm", action="store_true")
    parser.add_argument("--plant-coverage", action="store_true")
    parser.add_argument("--plant-barotropic", action="store_true")
    parser.add_argument("--plant-ene-coefficient", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.oracle_root,
            max_step=args.max_step,
            plant_state=args.plant_state,
            plant_registry=args.plant_registry,
            plant_arm=args.plant_arm,
            plant_coverage=args.plant_coverage,
            plant_barotropic=args.plant_barotropic,
            plant_ene_coefficient=args.plant_ene_coefficient,
        )
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

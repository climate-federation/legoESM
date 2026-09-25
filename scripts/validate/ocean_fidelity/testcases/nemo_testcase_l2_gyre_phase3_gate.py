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
from legoesm.ocean.fidelity.provenance import worktree_stamp

BAR = 1.0e-15
CASE = "GYRE-zco"
SCALAR_MATH_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round19_oracle_v2_external")
ROOT = SCALAR_MATH_ROOT
STAGE2_ROOT = SCALAR_MATH_ROOT
STAGE3_ROOT = SCALAR_MATH_ROOT
STAGE_WW_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round21_oracle_v2_stage_ww")
DIMS = (36, 26, 31)
LEVELS = {1: {"Kaa": 3, "Kmm": 1}, 2: {"Kaa": 2, "Kmm": 3}, 3: {"Kaa": 3, "Kmm": 2}}
BIT_IDENTITY_EXPECTED = {
    "oracle_step_entry_kt00000001.bin": (
        "9def5f4986853f497a5e0507bea185fe1ec4348715e2aeca0b14507df8e24fa0"
    ),
    "oracle_step_entry_kt00000002.bin": (
        "ba96e02e6f06f50604bc5920e0ec023f9d07662a536f51a7319e8515673f86d1"
    ),
    "oracle_stage_kt00000001_s1.bin": (
        "ce25b004e7e8289b6e803263f895576981ce22516ccddfbd85d7be5ce5bcaedc"
    ),
    "oracle_stage_kt00000001_s2.bin": (
        "e29972359b9fe9929d38dfc58ca0d5f0f84c9a0a7ce65905481349a8a52ef875"
    ),
    "oracle_stage_kt00000001_s3.bin": (
        "810d4ae83d5827d67cb7a09fe436a5d3f2d4201e89727fb86c891c349bb1c714"
    ),
    "oracle_rhs_kt00000001.bin": "a09426f638de0ce384b739621d08cf7ae27d41339cadcc2f16f8bbd3de215c45",
    "oracle_bt_frames_kt00000001.bin": (
        "7489d1edebacb71298ca5e65d3c9b54777f745642c01a4c4a696e9d6b5e913d8"
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


def require_scalar_math_roots(*roots: Path) -> list[dict]:
    """Fail closed unless every selected root has the certified v2 kt=2 entry."""
    name = "oracle_step_entry_kt00000002.bin"
    expected = BIT_IDENTITY_EXPECTED[name]
    canonical = SCALAR_MATH_ROOT / name
    require(canonical.is_file(), f"missing scalar-math identity control {canonical}")
    canonical_sha = sha256(canonical)
    require(
        canonical_sha == expected,
        f"scalar-math identity control changed: {canonical_sha} != {expected}",
    )
    rows = []
    for label, root in zip(("oracle", "stage2", "stage3"), roots, strict=True):
        path = root / name
        require(path.is_file(), f"missing {label} scalar-math identity control {path}")
        observed = sha256(path)
        require(
            observed == canonical_sha,
            f"{label} root is not certified scalar-math v2 at kt=2: "
            f"{observed} != {canonical_sha}",
        )
        rows.append(
            {
                "name": f"scalar_math_root_identity.{label}",
                "path": str(path),
                "status": "VERIFIED",
                "sha256": observed,
            }
        )
    return rows


def resolved_namelist_blocks(path: Path) -> set[str]:
    """Enumerate the live dynamics/tracer/drag groups from NEMO output.

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
            if name.startswith(("NAMDYN", "NAMZDF", "NAMTRA", "NAMDRG")):
                blocks.add(name.lower())
    require(bool(blocks), f"{path}: no resolved dynamics/tracer/drag blocks")
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
        "ssh": _xy(values[4 * count :], nx, ny),
    }


def read_transport(
    path: Path,
    expected_stage: int,
    *,
    expected_kt: int = 1,
    expected_slots: tuple[int, int, int, int] | None = None,
) -> dict:
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        if magic == "NEMO_L1_TRANSP_1":
            header = struct.unpack("=8i", handle.read(32))
        elif magic == "NEMO_L2_TRTRP_1":
            header = struct.unpack("=11i", handle.read(44))
        else:
            raise AssertionError(f"{path}: bad magic {magic!r}")
        values = np.fromfile(handle, dtype=np.float64)
    if magic == "NEMO_L1_TRANSP_1":
        version, kt, stage, kmm, nx, ny, nz, bits = header
    else:
        version, kt, stage, kbb, kmm, kaa, krhs, nx, ny, nz, bits = header
        if expected_slots is None:
            expected_slots = (
                1, LEVELS[expected_stage]["Kmm"],
                LEVELS[expected_stage]["Kaa"], 3)
        require(
            (kbb, kmm, kaa, krhs) == expected_slots,
            f"{path}: wrong stage indices",
        )
    require((version, nx, ny, nz, bits) == (1, *DIMS, 64), f"{path}: bad header")
    require((kt, stage) == (expected_kt, expected_stage),
            f"{path}: wrong kt/stage")
    require(values.size == 3 * nx * ny * nz, f"{path}: bad payload")
    # NEMO does not own or initialize the four-cell transport halo.  In the
    # vector-invariant arm this legacy momentum-side record also precedes the
    # tra_adv_trp call that initializes zFw.  Validate every owned horizontal
    # transport cell; retain the raw zFw sentinel count for the explicitly
    # UNINFORMATIVE row below instead of laundering it into a comparison.
    owned_finite = True
    for block in np.split(values, 3)[:2]:
        full = block.reshape((nx, ny, nz), order="F")
        owned_finite &= bool(np.all(np.isfinite(full[2:-2, 2:-2])))
    require(owned_finite, f"{path}: non-finite owned horizontal payload")
    count = nx * ny * nz
    return {
        "stage": stage,
        "Kmm": kmm,
        "registry_level": level,
        "nonowned_nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
        "zFw_nonfinite_count": int(np.count_nonzero(~np.isfinite(values[2 * count :]))),
        "zFu": _xyz(values[:count], nx, ny, nz),
        "zFv": _xyz(values[count : 2 * count], nx, ny, nz),
        "zFw": _xyz(values[2 * count :], nx, ny, nz),
    }


def read_stage_ww(path: Path, expected_stage: int) -> dict:
    """Read post-``tra_adv_trp`` ww/pFw at one RK3 stage.

    Unlike ``oracle_transport_*``, this record is emitted after the executed
    vector-invariant tracer path calls ``wzv(...,np_transport)``
    (``traadv.F90:220-235``), so ``ww`` is defined at all three stages.
    """
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=10i", handle.read(40))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_STGWW_1", f"{path}: bad magic {magic!r}")
    require(
        (version, kt, stage, kmm, kaa, nx, ny, nz, bits)
        == (1, 1, expected_stage, LEVELS[expected_stage]["Kmm"],
            LEVELS[expected_stage]["Kaa"], *DIMS, 64),
        f"{path}: bad header {header}",
    )
    require(kbb == 1, f"{path}: expected Kbb=1, got {kbb}")
    count = nx * ny * nz
    require(values.size == 1 + 2 * count, f"{path}: bad payload")
    require(np.isfinite(values[0]), f"{path}: non-finite stage clock")
    ww = _xyz(values[1 : 1 + count], nx, ny, nz)
    p_fw = _xyz(values[1 + count :], nx, ny, nz)
    require(np.all(np.isfinite(ww)) and np.all(np.isfinite(p_fw)),
            f"{path}: non-finite parser-visible payload")
    return {
        "stage": stage,
        "Kmm": kmm,
        "registry_level": level,
        "rDt_s": float(values[0]),
        "ww": ww,
        "pFw": p_fw,
    }


def read_stage2_terms(path: Path) -> dict:
    """Read source-ordered stage-2 Krhs snapshots from WRITE-only MY_SRC."""
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, krhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKTRM_1", f"{path}: bad magic")
    require(
        (version, kt, stage, kmm, krhs, nx, ny, nz, bits)
        == (1, 1, 2, 3, 2, *DIMS, 64),
        f"{path}: bad header",
    )
    count = nx * ny * nz
    require(values.size == 8 * count, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    names = (
        "before_u", "before_v", "after_hpg_u", "after_hpg_v",
        "after_vorticity_u", "after_vorticity_v",
        "after_advection_u", "after_advection_v",
    )
    result = {
        name: _xyz(values[index * count : (index + 1) * count], nx, ny, nz)
        for index, name in enumerate(names)
    }
    result.update({"stage": stage, "Kmm": kmm, "Krhs": krhs,
                   "registry_level": level})
    return result


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
    count = nx * ny
    return {
        "kt": kt,
        "Kaa": kaa,
        "registry_level": level,
        "uu_b": _xy(values[:count], nx, ny),
        "vv_b": _xy(values[count : 2 * count], nx, ny),
        "un_adv": _xy(values[2 * count : 3 * count], nx, ny),
        "vn_adv": _xy(values[3 * count :], nx, ny),
    }


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


def read_tracer_stage3(path: Path) -> dict:
    """Read the source-ordered stage-3 tracer completion record."""
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKTR3_1", f"{path}: bad magic")
    require(
        (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
        == (1, 1, 3, 1, 2, 3, 3, *DIMS, 64),
        f"{path}: bad header",
    )
    count = nx * ny * nz
    require(values.size == 16 * count + 3 * nx * ny, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    names = (
        "zero_T", "zero_S", "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
        "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
        "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result = {
        name: _xyz(values[index * count:(index + 1) * count], nx, ny, nz)
        for index, name in enumerate(names)
    }
    offset = 16 * count
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        begin = offset + index * nx * ny
        result[name] = _xy(values[begin:begin + nx * ny], nx, ny)
    result.update({"registry_level": level, "Kbb": kbb, "Kmm": kmm,
                   "Krhs": krhs, "Kaa": kaa})
    return result


# The solver returns ONE keyed barotropic substep frame (see
# barotropic_latlon_cgrid._run_substep_loop); this map is the mimicry-only glue
# from THIS oracle dump's field names to that frame's keys, and it lives in the
# harness, never in the model.  Only two names differ; every other name is
# shared with the L1-overflow registry verbatim.
BT_TRACE_KEY = {
    "transport_metric_u": "transport_u",
    "transport_metric_v": "transport_v",
}

# The solver's barotropic substep frame is NAME-KEYED at and after the
# lane-1/lane-2 merge (c9526e585, which unified the two branches' rival
# positional layouts) and POSITIONAL before it.  This is the exact reach of
# that merge hunk, so the gate must read both: a before/after comparison runs
# ONE instrument against TWO trees, and a positional-only reader raises
# KeyError on the merged tree while a keyed-only reader raises TypeError on
# the pre-merge tree.  Order below is b2f7c298's BT_SUBSTEP_NAMES verbatim.
BT_PRE_MERGE_ORDER = (
    "eta_entry", "u_entry", "v_entry",
    "eta_mid", "u_mid", "v_mid",
    "eta_exit", "eta_pgf",
    "pgf_u", "pgf_v", "trd_u", "trd_v", "slow_u", "slow_v",
    "u_exit", "v_exit", "transport_metric_u", "transport_metric_v",
)


# The isomorphism branch's UP3 selector split (9bbf9f7bd) renamed the public
# "upwind3" to "nemo_up3"/"oceananigans_up3".  Pre-split, "upwind3" resolved
# NEMO's advected-velocity-pair rule (dynadv_up3.F90:166,169-172) whenever
# momentum_time_integrator == "rk3_ws" -- which this arm's base card is -- and
# Oceananigans' transport rule otherwise.  So the two names select the SAME
# NEMO arm on their respective trees, and reading the tree's own accepted set
# lets one committed gate score both.
def _nemo_up3_name() -> str:
    try:
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            VALID_MOMENTUM_FLUX_SCHEME,
        )
    except ImportError:          # pre-split tree: no exported valid set
        return "upwind3"
    return ("nemo_up3" if "nemo_up3" in VALID_MOMENTUM_FLUX_SCHEME
            else "upwind3")


_NEMO_UP3_NAME = _nemo_up3_name()


def bt_frame(substeps, name):
    """One barotropic substep frame by name, keyed or positional."""
    if isinstance(substeps, dict):
        return substeps[BT_TRACE_KEY.get(name, name)]
    return substeps[BT_PRE_MERGE_ORDER.index(name)]

ORACLE_BT_SUBSTEP_NAMES = (
    "eta_entry", "u_entry", "v_entry",
    "eta_mid", "u_mid", "v_mid",
    "eta_exit", "eta_pgf",
    "pgf_u", "pgf_v", "cor_u", "cor_v", "trd_u", "trd_v",
    "slow_u", "slow_v", "u_exit", "v_exit",
    "transport_metric_u", "transport_metric_v",
)


def read_bt_substeps(
    path: Path, *, expected_dims: tuple[int, int] = DIMS[:2],
    expected_ncycle: int = 50,
) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        version, kt, ncycle, nx, ny, bits = header
        # The instrumented GYRE run on disk emits format 1 (18 fields); the
        # re-instrumented run that would emit format 2 (adds cor_u/cor_v, the
        # separated barotropic Coriolis) was never made -- the lane's credits
        # ran out first.  Read BOTH, so one committed instrument can score the
        # oracle set that exists as well as the one that was planned.  Under
        # format 1 the cor_u/cor_v rows are simply absent, which is why
        # --without-oracle-ene-coefficients (the arm that consumes them) is
        # required with it.
        require(magic in ("NEMO_L2_BTSUB_1", "NEMO_L2_BTSUB_2"),
                f"{path}: bad magic {magic!r}")
        _fields = (BT_PRE_MERGE_ORDER if version == 1
                   else ORACLE_BT_SUBSTEP_NAMES)
        require(
            (version, kt, ncycle, nx, ny, bits)
            == (version, 1, expected_ncycle, *expected_dims, 64)
            and version in (1, 2),
            f"{path}: bad header",
        )
        records = {name: [] for name in _fields}
        count = nx * ny
        for expected in range(1, ncycle + 1):
            raw_jn = handle.read(4)
            require(len(raw_jn) == 4, f"{path}: truncated at substep {expected}")
            (jn,) = struct.unpack("=i", raw_jn)
            require(jn == expected, f"{path}: substep sequence {jn} != {expected}")
            for name in _fields:
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
    from legoesm.ocean.fidelity.ulp_move_gate import record_residual_field

    record_residual_field(name, oracle, candidate, active)
    n_unequal = int(np.count_nonzero(
        candidate[active].view(np.uint64) != oracle[active].view(np.uint64)))
    absolute = float(np.max(np.abs(candidate[active] - oracle[active])))
    reference = float(np.max(np.abs(oracle[active])))
    normalized = absolute / max(reference, 1.0)
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "n_unequal": n_unequal,
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
    from legoesm.ocean.eos import nemo_potential_temperature_from_conservative
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_qns
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_boundary_condition
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    sbc = gyre_surface_boundary_condition(card, kt * card.dt_s)
    sst = state.T.data[..., 0]
    sst_m = nemo_potential_temperature_from_conservative(
        sst, state.S.data[..., 0])
    # usrdef_sbc.F90:109-120,138-145: qns+qsr is the Haney term plus EMP
    # heat content, evaluated once from the entering Kbb/Nbb SST.
    qns = nemo_gyre_qns(
        sst,
        sst_m,
        sbc.t_star_c,
        sbc.qsr_w_m2,
        sbc.emp_kg_m2_s,
    )
    # NEMO carries qns/qsr separately; the shared forcing object carries their
    # materialized sum.
    from legoesm.core.source_rounding import nemo_source_round

    q_total = nemo_source_round(qns + sbc.qsr_w_m2)
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
        # NEMO's usrdef_sbc fields are already in the native ocean
        # referential.  Preserve those source operands through the shared
        # forcing interface; reconstructing them after the inverse geographic
        # rotation loses 1--3 ulp before sbcmod.F90's face interpolation.
        tau_i_native=sbc.utau_pa,
        tau_j_native=sbc.vtau_pa,
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
    stage2_root: Path = STAGE2_ROOT,
    stage3_root: Path = STAGE3_ROOT,
    max_step=10,
    plant_state=False,
    plant_registry=False,
    plant_arm=False,
    plant_coverage=False,
    plant_barotropic=False,
    plant_ene_coefficient=False,
    plant_drag=False,
    plant_stage2_thermodynamics=False,
    plant_stage2_term=False,
    plant_barotropic_state=False,
    stage2_term_limit="hpg",
    stage2_term_only=None,
    measure_stage2_update_arm=False,
    measure_stage2_tracers=False,
    measure_stage3_completion=False,
    without_oracle_ene_coefficients=False,
    trajectory_only=False,
) -> dict:
    scalar_math_root_identity = require_scalar_math_roots(
        root, stage2_root, stage3_root)
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_barotropic_coriolis,
        _nemo_literal_een_coefficients,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "certification requires production JIT; JAX_DISABLE_JIT is forbidden")
    require(1 <= max_step <= 10, "max_step must be in 1..10")
    require(
        stage2_term_limit in ("hpg", "vorticity", "advection"),
        "stage2_term_limit must be hpg, vorticity, or advection",
    )
    require(
        stage2_term_only in (None, "none", "hpg", "vorticity", "advection"),
        "stage2_term_only must be none, hpg, vorticity, advection, or None",
    )

    def host_result_and_release_compilation(value):
        """Materialize one diagnostic executable, then release its JIT cache.

        Each private hook changes static ``self`` and therefore creates a full
        GYRE executable.  Keeping a dozen such executables resident was the
        round-8 host OOM; scoring needs host values, not compiled programs.
        """
        host = jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf)
            if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
            value,
        )
        jax.clear_caches()
        return host

    card = build_nemo_testcase_card(CASE)
    # ONE CARD, ONE PROGRAM.  The card resolves its own freshwater pair --
    # real_freshwater (NEMO QCO carries E-P as volume with sfx=0,
    # usrdef_sbc.F90:138-145) with fix_eta_drift=False (decision 35: NEMO has
    # no global eta projection).  The gate no longer overrides it, so what it
    # certifies is what the card runs.
    cfg = card.recipe.model_config
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
            cfg.physics.shortwave_penetration.scheme == "nemo_qsr_2bd"
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
            and evd_cfg.evd_n2_time_level == "nemo_now_before"
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
        "namdrg": (
            cfg.bottom_drag.bottom_drag_scheme == "nemo_quadratic"
            and cfg.zdf_drag_in_matrix
            and cfg.zdf_baroclinic_only
            and cfg.barotropic_drag_substep
        ),
        "namdrg_bot": (
            cfg.bottom_drag.bottom_drag_cd0 == 1.0e-3
            and cfg.bottom_drag.bottom_drag_cdmax == 0.1
            and cfg.bottom_drag.bottom_drag_ke0 == 2.5e-3
            and cfg.bottom_drag.bottom_drag_z0 == 3.0e-3
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
    tracer_stage3_path = stage3_root / "oracle_rktracer_stage3_kt00000001.bin"
    ene_coefficient_path = root / "oracle_bt_ene_coeff_kt00000001.bin"
    for path in (zdf_path, qsr_path, bt_substep_path, tracer_stage3_path):
        require(path.is_file(), f"missing causal artifact {path}")
        artifacts[path.name] = sha256(path)
    # Rule 1 disposition, not a relaxed bar: the ENE-coefficient arm scores
    # against an oracle dump the instrumented GYRE run never emitted, so with
    # --without-oracle-ene-coefficients the arm is WAIVED and listed loudly in
    # "unmeasured" rather than aborting the whole gate.  Without the flag the
    # dump is still required.  Every other row, and the DEBT verdict, is
    # unaffected; this is what lets ONE instrument score two trees when the
    # oracle side is missing on both.
    if without_oracle_ene_coefficients:
        require(not plant_ene_coefficient,
                "--plant-ene-coefficient needs the oracle ENE dump")
    else:
        require(ene_coefficient_path.is_file(),
                f"missing causal artifact {ene_coefficient_path}")
    ene_available = ene_coefficient_path.is_file() and not (
        without_oracle_ene_coefficients)
    if ene_available:
        artifacts[ene_coefficient_path.name] = sha256(ene_coefficient_path)
    zdf_entry = read_zdf_entry(zdf_path)
    qsr_stage3 = read_qsr_stage3(qsr_path)
    tracer_stage3 = read_tracer_stage3(tracer_stage3_path)
    bt_substeps = read_bt_substeps(bt_substep_path)
    oracle_ene_coefficients = (
        read_ene_coefficients(ene_coefficient_path) if ene_available else None)
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

    # The source-order dump came from a WRITE-only extension of this same
    # executable.  Admit it only if its ordinary stage-1 state is byte-for-byte
    # the already pinned oracle record; the source-order file then gets its own
    # fixed digest and strict header/payload checks below.
    stage2_identity_path = stage2_root / "oracle_stage_kt00000001_s1.bin"
    stage2_term_path = stage2_root / "oracle_rkstage2_terms_kt00000001.bin"
    require(stage2_identity_path.is_file(), f"missing {stage2_identity_path}")
    require(stage2_term_path.is_file(), f"missing {stage2_term_path}")
    stage2_identity_sha = sha256(stage2_identity_path)
    require(
        stage2_identity_sha == BIT_IDENTITY_EXPECTED[
            "oracle_stage_kt00000001_s1.bin"],
        f"{stage2_identity_path}: instrumentation changed ordinary state",
    )
    stage2_term_sha = sha256(stage2_term_path)
    require(
        stage2_term_sha
        == "99c335f9b0ea5a3aebe807943768bee66d0a1acb1b6c88ddddfb00915c8b68b6",
        f"{stage2_term_path}: unregistered content hash",
    )
    stage2_terms = read_stage2_terms(stage2_term_path)
    artifacts[f"stage2_terms/{stage2_identity_path.name}"] = stage2_identity_sha
    artifacts[f"stage2_terms/{stage2_term_path.name}"] = stage2_term_sha
    stage3_identity_rows = []
    for name in ("oracle_stage_kt00000001_s3.bin",
                 "oracle_step_entry_kt00000002.bin"):
        identity_path = stage3_root / name
        require(identity_path.is_file(), f"missing {identity_path}")
        observed = sha256(identity_path)
        expected = BIT_IDENTITY_EXPECTED[name]
        require(observed == expected,
                f"{identity_path}: instrumentation changed ordinary state")
        stage3_identity_rows.append({
            "name": f"stage3_instrumentation_bit_identity.{name}",
            "status": "VERIFIED",
            "expected_sha256": expected,
            "observed_sha256": observed,
        })
        artifacts[f"stage3/{name}"] = observed
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
    barotropic_state_steps = []
    barotropic_state_first_over_bar = None
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
        # Score the independently prognostic Kaa pair written by stp_2D.  The
        # existing oracle_bt_frames writer records exactly this pair; after
        # stprk3.F90:213 swaps Naa into Nbb it seeds the next step.  Advance
        # once even at max_step so every available Kaa frame is covered.
        freshwater, surface = _surface_forcings(card, state, kt)
        next_state = model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        require(next_state.uu_b is not None and next_state.vv_b is not None,
                "NEMO identity step returned no prognostic uu_b/vv_b pair")
        baro_candidate = {
            "uu_b": np.asarray(next_state.uu_b.data)[:, 1:],
            "vv_b": np.asarray(next_state.vv_b.data)[1:, :],
        }
        baro_rows = []
        for component, mask_name in (("uu_b", "u"), ("vv_b", "v")):
            candidate_b = baro_candidate[component]
            if plant_barotropic_state and kt == 1 and component == "uu_b":
                candidate_b = candidate_b.copy()
                active_b = masks[mask_name][..., 0]
                first = tuple(np.argwhere(active_b)[0])
                row_scale = max(float(np.max(np.abs(bt[kt][component][active_b]))), 1.0)
                candidate_b[first] += 3.0 * np.spacing(row_scale)
            baro_row = score(
                f"{CASE}.kt{kt}.after.{component}",
                bt[kt][component], candidate_b, masks[mask_name][..., 0])
            # At kt=1 both models enter from the same rest state, so Decision
            # 8 makes Kaa an equal-input identity boundary.  At later steps
            # the already-diverged full prognostic state feeds the external
            # mode; those rows are downstream trajectory consistency, not an
            # equal-input operator-identity claim.  The kt=1 three-ULP plant
            # remains non-vacuous even though it is below the 1e-15 bar.
            if kt == 1 and not baro_row["exact"]:
                baro_row["status"] = "DEBT"
                baro_row["reason"] = (
                    "prognostic uu_b/vv_b identity boundary requires bit equality")
            elif kt > 1:
                baro_row["reason"] = (
                    "downstream full-state consistency; inputs are not bit-identical")
            baro_rows.append(baro_row)
        barotropic_state_steps.append({"kt": kt, "rows": baro_rows})
        debt_components = [
            row["name"].rsplit(".", 1)[-1]
            for row in baro_rows if row["status"] == "DEBT"
        ]
        if debt_components and barotropic_state_first_over_bar is None:
            barotropic_state_first_over_bar = {
                "kt": kt, "fields": debt_components,
            }
        if plant_barotropic_state and kt == 1:
            require(baro_rows[0]["status"] == "DEBT",
                    "planted three-ulp uu_b violation did not fire")
        if kt == 1:
            faithful_kt2 = next_state
        if kt < max_step:
            state = next_state

    require(faithful_kt2 is not None or max_step == 1, "kt2 candidate was not produced")

    # Low-memory certification route: the trajectory loop above is the same
    # production-jitted model and the same score/registry machinery as the
    # full gate.  Returning here prevents the private stage/term hooks below
    # from compiling several additional whole GYRE programs in one process;
    # those hooks are intentionally run as separate boundary probes.
    if trajectory_only:
        return {
            "worktree": worktree_stamp(),
            "format": "nemo-testcase-l2-gyre-phase3-trajectory-only-v2",
            "case": CASE,
            "status": (
                "AT-BAR" if first_over_bar is None
                and barotropic_state_first_over_bar is None else "DEBT"),
            "execution_regime": "production-jit-cpu-fp64-x64-libm",
            "oracle_root": str(root),
            "scalar_math_root_identity": scalar_math_root_identity,
            "max_step": max_step,
            "first_over_bar": first_over_bar,
            "barotropic_state_first_over_bar": barotropic_state_first_over_bar,
            "steps": steps,
            "barotropic_state_steps": barotropic_state_steps,
            "artifacts": artifacts,
            "low_memory_route": {
                "same_production_step": True,
                "private_stage_hooks": "SEPARATE_PROBES",
            },
        }

    freshwater0, surface0 = _surface_forcings(card, card.recipe.initial_state, 1)
    seeded_entry = model._seed_tke_preclosure_carry(card.recipe.initial_state)
    model.prime_step_caches(seeded_entry)

    # C1: direct stage-entry TKE/EVD coefficients.  NEMO W levels 2..30
    # correspond to legoESM's 29 interior interfaces.
    lego_avt, lego_avm = jax.jit(
        lambda entry, forcing: model.diagnose_vertical_K(
            entry, card.dt_s, forcing)
    )(seeded_entry, surface0)
    lego_avt, lego_avm = host_result_and_release_compilation(
        (lego_avt, lego_avm))
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
    full_tendency = jax.jit(
        lambda entry, forcing: model.tendencies(
            entry, forcing, dt=card.dt_s)
    )(seeded_entry, surface0)
    full_tendency = host_result_and_release_compilation(full_tendency)
    no_qsr_tendency = jax.jit(
        lambda entry, forcing: no_qsr_model.tendencies(
            entry, forcing, dt=card.dt_s)
    )(seeded_entry, surface0)
    no_qsr_tendency = host_result_and_release_compilation(no_qsr_tendency)
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
    # The trajectory compile is no longer needed.  Drop it before compiling
    # the three private trace variants: retaining all four static ``self``
    # executables was the round-8 host-memory exhaustion root cause.
    jax.clear_caches()
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True),
    )
    trace = trace_model.step(
        seeded_entry, card.dt_s, freshwater=freshwater0,
        surface_forcing=surface0)
    trace = host_result_and_release_compilation(trace)
    del trace_model
    generic_cfg = cfg._replace(
        barotropic=cfg.barotropic._replace(
            barotropic_een_coefficient_evaluation="generic"
        )
    )
    generic_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, generic_cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True),
    )
    generic_trace = generic_model.step(
        seeded_entry, card.dt_s, freshwater=freshwater0,
        surface_forcing=surface0)
    generic_trace = host_result_and_release_compilation(generic_trace)
    del generic_model
    literal_trace = trace
    drag_omit_model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            omit_barotropic_substep_drag=True),
    )
    drag_omit_trace = drag_omit_model.step(
        seeded_entry, card.dt_s, freshwater=freshwater0,
        surface_forcing=surface0)
    drag_omit_trace = host_result_and_release_compilation(drag_omit_trace)
    del drag_omit_model

    # D1: reconstruct NEMO's nonlinear bottom-drag operand directly from the
    # resolved kt=1 program.  The entering 3-D velocity is at rest, hence
    # zdfdrg.F90:184-190 gives one frozen wet-T rate Cd0*sqrt(ke0)=5e-5.
    # dyn_drg_init.F90:1614-1618 averages that rate to native U/V faces;
    # dynspg_ts.F90:699-705 multiplies the SUBSTEP-ENTRY velocity by the rate
    # and current inverse depth.  No absent ENE-coefficient dump participates.
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    drag_rate_t = np.where(
        wet,
        cfg.bottom_drag.bottom_drag_cd0
        * np.sqrt(cfg.bottom_drag.bottom_drag_ke0),
        0.0,
    )
    drag_rate_u = 0.5 * (drag_rate_t + np.roll(drag_rate_t, -1, axis=1))
    drag_rate_v = 0.5 * (drag_rate_t + np.roll(drag_rate_t, -1, axis=0))
    area_t = np.asarray(card.recipe.z_coord.nemo_e1e2t)
    area_u = np.asarray(card.recipe.z_coord.nemo_e1e2u)
    area_v = np.asarray(card.recipe.z_coord.nemo_e1e2v)
    hu0 = np.asarray(card.recipe.z_coord.nemo_hu_0)
    hv0 = np.asarray(card.recipe.z_coord.nemo_hv_0)
    drag_operand_rows = []
    for substep in (0, 1):
        eta_entry = bt_substeps["eta_entry"][substep]
        weighted_eta = area_t * eta_entry
        hu = hu0 + 0.5 * (
            weighted_eta + np.roll(weighted_eta, -1, axis=1)
        ) / area_u
        hv = hv0 + 0.5 * (
            weighted_eta + np.roll(weighted_eta, -1, axis=0)
        ) / area_v
        oracle_drag_u = np.where(
            masks["u"][..., 0],
            -drag_rate_u * bt_substeps["u_entry"][substep]
            / np.maximum(hu, cfg.min_water_column_m),
            0.0,
        )
        oracle_drag_v = np.where(
            masks["v"][..., 0],
            -drag_rate_v * bt_substeps["v_entry"][substep]
            / np.maximum(hv, cfg.min_water_column_m),
            0.0,
        )
        for component, reference in (
            ("u", oracle_drag_u), ("v", oracle_drag_v)
        ):
            candidate = _trace_native(
                bt_frame(trace.substeps, f"drag_{component}"),
                f"drag_{component}",
            )[substep]
            drag_operand_rows.append(score(
                f"{CASE}.kt1.bt.jn{substep + 1:02d}.drag_{component}",
                reference,
                candidate,
                masks[component][..., 0],
                plant=(plant_drag and substep == 1 and component == "u"),
            ))

    if plant_drag:
        planted_drag = next(
            row for row in drag_operand_rows
            if row["name"].endswith("jn02.drag_u"))
        require(
            planted_drag["status"] == "DEBT"
            and planted_drag["absolute_max"] >= 1.0,
            "planted drag violation did not fire at its registered magnitude",
        )
    drag_control = {
        "name": "control.planted_drag_operand",
        "status": "VERIFIED" if plant_drag else "NOT_REQUESTED",
    }

    drag_causal_rows = []
    drag_omission_rows = []
    drag_movements = []
    for component in ("u", "v"):
        reference = bt_substeps[f"trd_{component}"][1]
        active = masks[component][..., 0]
        production = _trace_native(
            bt_frame(trace.substeps, f"trd_{component}"),
            f"trd_{component}",
        )[1]
        omitted = _trace_native(
            bt_frame(drag_omit_trace.substeps, f"trd_{component}"),
            f"trd_{component}",
        )[1]
        drag_causal_rows.append(score(
            f"{CASE}.kt1.bt.jn02.drag_composition.trd_{component}",
            reference, production, active))
        drag_omission_rows.append(score(
            f"{CASE}.kt1.bt.jn02.omit_drag.trd_{component}",
            reference, omitted, active))
        drag_movements.append(float(np.max(
            np.abs(production[active] - omitted[active]))))
    drag_residual = max(row["absolute_max"] for row in drag_causal_rows)
    drag_omitted_residual = max(
        row["absolute_max"] for row in drag_omission_rows)
    drag_movement = max(drag_movements)
    drag_owner_label = (
        "CONFIRMED_CAUSAL_OWNER_AT_SUBSTEP2"
        if all(row["status"] == "AT-BAR" for row in drag_causal_rows)
        and all(row["status"] == "DEBT" for row in drag_omission_rows)
        and all(row["status"] == "AT-BAR" for row in drag_operand_rows)
        else (
            "NEAR_NULL_NO_DISCRIMINATING_POWER"
            if drag_movement < 0.1 * drag_omitted_residual
            else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
        )
    )
    drag_operand_walk = {
        "resolved": {
            "nn_drg": "np_non_lin",
            "ln_drgimp": True,
            "rn_Cd0": cfg.bottom_drag.bottom_drag_cd0,
            "rn_Cdmax": cfg.bottom_drag.bottom_drag_cdmax,
            "rn_ke0": cfg.bottom_drag.bottom_drag_ke0,
            "rn_z0": cfg.bottom_drag.bottom_drag_z0,
            "frozen_wet_t_rate_m_s": float(
                cfg.bottom_drag.bottom_drag_cd0
                * np.sqrt(cfg.bottom_drag.bottom_drag_ke0)),
            "coefficient_time_level": "Kmm_once_per_whole_step",
            "substep_velocity_time_level": "entry_un_e_vn_e",
        },
        "operand_rows": drag_operand_rows,
        "production_combined_tendency_rows": drag_causal_rows,
        "private_omission_rows": drag_omission_rows,
        "scaling_check_before_owner_label": True,
        "production_residual": drag_residual,
        "omission_residual": drag_omitted_residual,
        "production_over_omission_residual": (
            drag_residual / drag_omitted_residual
            if drag_omitted_residual else None),
        "arm_movement": drag_movement,
        "owner_label": drag_owner_label,
        "planted_control": drag_control,
    }

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
                    bt_frame(trace.substeps, f"{component}_mid"), f"{component}_mid",
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
    if ene_available:
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
        candidate_u_mid = bt_frame(trace.substeps, "u_mid")[1]
        candidate_v_mid = bt_frame(trace.substeps, "v_mid")[1]
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
                bt_frame(literal_trace.substeps, f"trd_{component}"),
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
            bt_frame(generic_trace.substeps, "trd_u"), "trd_u"
        )[1]
        literal_native_u = _trace_native(
            bt_frame(literal_trace.substeps, "trd_u"), "trd_u"
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
    else:
        # WAIVED, not passed: the oracle ENE-coefficient dump is absent, so
        # every row in this arm is UNMEASURED.  Named in "unmeasured" below.
        coefficient_rows = []
        product_rows = []
        reconstructed_term_rows = []
        ene_coefficient_control = {
            "name": "control.planted_ene_coefficient",
            "status": "WAIVED_NO_ORACLE_ENE_COEFFICIENT_DUMP",
        }
        ene_causal_scaling = {
            "ene_component_owner_label":
                "WAIVED_NO_ORACLE_ENE_COEFFICIENT_DUMP",
            "combined_trd_owner_label":
                "WAIVED_NO_ORACLE_ENE_COEFFICIENT_DUMP",
            "rows": [],
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
                bt_frame(trace.substeps, name), name)[substep]
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
    stage1_thermodynamic_rows = []
    stage1_transport_rows = []
    stage2_thermodynamic_arm_rows = []
    stage2_thermodynamic_scaling = None
    stage2_thermodynamic_control = {
        "name": "control.planted_stage1_thermodynamic_operand",
        "status": "NOT_REQUESTED",
    }
    stage2_term_rows = []
    stage2_term_first_over_bar = None
    stage2_term_control = {
        "name": "control.planted_stage2_source_term",
        "status": "NOT_REQUESTED",
    }
    stage2_update_arm = {
        "status": "UNMEASURED",
        "reason": "run with --measure-stage2-update-arm",
    }
    stage2_tracer_rows = []
    stage3_completion = {
        "status": "UNMEASURED",
        "reason": "run with --measure-stage3-completion",
    }
    if max_step >= 2:
        candidate_stage_states = {}
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
                stage_state = host_result_and_release_compilation(stage_state)
            candidate_stage_states[stage] = stage_state
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
            if stage != 1:
                for tracer in ("T", "S"):
                    stage_rows.append(
                        {
                            "name": f"{CASE}.kt1.stage{stage}.{tracer}",
                            "status": "UNMEASURED",
                            "reason": "ordered walk has not yet reached this tracer Kaa stage",
                        }
                    )

        # stprk3_stg.F90:452-565 advances stage-1 tracers before stage-2
        # eos+dyn_hpg.  Expose that exact legoESM operand only after the normal
        # step has completed, so this is a WRITE-only diagnostic seam.
        tracer_stage1_state = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_tracer_stage=1),
        ).step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
        )
        tracer_stage1_state = host_result_and_release_compilation(
            tracer_stage1_state)
        tracer_stage1_fields = lego_fields(tracer_stage1_state)
        for name in ("T", "S", "ssh"):
            candidate = tracer_stage1_fields[name]
            if plant_stage2_thermodynamics and name == "T":
                candidate = np.asarray(candidate).copy()
                first = tuple(np.argwhere(masks[name])[0])
                candidate[first] += 1.0
            stage1_thermodynamic_rows.append(score(
                f"{CASE}.kt1.stage1.thermodynamic_operand.{name}",
                stages[1][name][..., :nlev] if name != "ssh" else stages[1][name],
                candidate,
                masks[name],
            ))

        if measure_stage2_tracers:
            # Keep this optional: each static diagnostic hook compiles a full
            # GYRE executable.  It exposes exactly the T/S Kaa written after
            # the stage-2 QCO combine (stprk3_stg.F90:537-565).
            tracer_stage2_state = LatLonCGridOceanModel(
                card.recipe.grid,
                card.recipe.z_coord,
                cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    expose_tracer_stage=2),
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater0,
                surface_forcing=surface0,
            )
            tracer_stage2_state = host_result_and_release_compilation(
                tracer_stage2_state)
            tracer_stage2_fields = lego_fields(tracer_stage2_state)
            for name in ("T", "S", "ssh"):
                stage2_tracer_rows.append(score(
                    f"{CASE}.kt1.stage2.thermodynamic_operand.{name}",
                    (stages[2][name][..., :nlev]
                     if name != "ssh" else stages[2][name]),
                    tracer_stage2_fields[name],
                    masks[name],
                ))

        if measure_stage3_completion:
            pre_zdf_state = LatLonCGridOceanModel(
                card.recipe.grid,
                card.recipe.z_coord,
                cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    expose_pre_implicit_state=True),
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater0,
                surface_forcing=surface0,
            )
            pre_zdf_state = host_result_and_release_compilation(pre_zdf_state)
            pre_fields = lego_fields(pre_zdf_state)
            surf = full_tendency.surface_tracer_forcing
            surf_T = (np.zeros_like(pre_fields["T"]) if surf is None
                      else np.asarray(surf.dT_dt.data))
            surf_S = (np.zeros_like(pre_fields["S"]) if surf is None
                      else np.asarray(surf.dS_dt.data))
            candidate_pre = {
                "T": pre_fields["T"] + card.dt_s * surf_T,
                "S": pre_fields["S"] + card.dt_s * surf_S,
            }
            r3bb = tracer_stage3["r3t_Kbb"][..., None]
            r3mm = tracer_stage3["r3t_Kmm"][..., None]
            r3aa = tracer_stage3["r3t_Kaa"][..., None]
            oracle_pre = {}
            pre_rows = []
            for name in ("T", "S"):
                oracle_pre[name] = (
                    (1.0 + r3bb) * tracer_stage3[f"Kbb_{name}"]
                    + card.dt_s * (1.0 + r3mm)
                    * tracer_stage3[f"after_ldf_{name}"]
                ) / (1.0 + r3aa)
                pre_rows.append(score(
                    f"{CASE}.kt1.stage3.pre_zdf.{name}",
                    oracle_pre[name][..., :nlev],
                    candidate_pre[name],
                    masks[name],
                ))

            # Supply the oracle's effective solve input while retaining the
            # normal surface-source addition inside _apply_implicit... .
            oracle_override = (
                jnp.asarray(oracle_pre["T"][..., :nlev])
                - card.dt_s * jnp.asarray(surf_T),
                jnp.asarray(oracle_pre["S"][..., :nlev])
                - card.dt_s * jnp.asarray(surf_S),
            )
            injected = LatLonCGridOceanModel(
                card.recipe.grid,
                card.recipe.z_coord,
                cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    pre_implicit_tracer_override=oracle_override),
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater0,
                surface_forcing=surface0,
            )
            injected = host_result_and_release_compilation(injected)
            injected_fields = lego_fields(injected)
            final_rows = []
            movements = []
            residuals = []
            for name in ("T", "S"):
                final_rows.append(score(
                    f"{CASE}.kt1.stage3.oracle_pre_zdf.{name}",
                    stages[3][name][..., :nlev],
                    injected_fields[name],
                    masks[name],
                ))
                active = masks[name]
                reference = stages[3][name][..., :nlev]
                scale = max(float(np.max(np.abs(reference[active]))), 1.0)
                movements.append(float(np.max(np.abs(
                    injected_fields[name][active]
                    - lego_fields(faithful_kt2)[name][active]))) / scale)
                residuals.append(float(np.max(np.abs(
                    lego_fields(faithful_kt2)[name][active]
                    - reference[active]))) / scale)
            movement = max(movements)
            residual = max(residuals)
            pre_clears = all(row["status"] == "AT-BAR" for row in pre_rows)
            final_clears = all(row["status"] == "AT-BAR" for row in final_rows)
            stage3_completion = {
                "status": "MEASURED",
                "source_order": ["advection", "sbc", "qsr", "ldf", "zdf"],
                "oracle_accumulator_term_max_abs": {
                    "advection_T": float(np.max(np.abs(
                        tracer_stage3["after_advection_T"]))),
                    "advection_S": float(np.max(np.abs(
                        tracer_stage3["after_advection_S"]))),
                    "sbc_T": float(np.max(np.abs(
                        tracer_stage3["after_sbc_T"]
                        - tracer_stage3["after_advection_T"]))),
                    "sbc_S": float(np.max(np.abs(
                        tracer_stage3["after_sbc_S"]
                        - tracer_stage3["after_advection_S"]))),
                    "qsr_T": float(np.max(np.abs(
                        tracer_stage3["after_qsr_T"]
                        - tracer_stage3["after_sbc_T"]))),
                    "qsr_S": float(np.max(np.abs(
                        tracer_stage3["after_qsr_S"]
                        - tracer_stage3["after_sbc_S"]))),
                    "ldf_T": float(np.max(np.abs(
                        tracer_stage3["after_ldf_T"]
                        - tracer_stage3["after_qsr_T"]))),
                    "ldf_S": float(np.max(np.abs(
                        tracer_stage3["after_ldf_S"]
                        - tracer_stage3["after_qsr_S"]))),
                },
                "pre_zdf_rows": pre_rows,
                "oracle_pre_zdf_causal_rows": final_rows,
                "scaling_check_before_owner_label": True,
                "causal_movement": movement,
                "faithful_residual": residual,
                "movement_over_faithful_residual": (
                    movement / residual if residual else None),
                "owner_label": (
                    "PRE_ZDF_INPUT_CONFIRMED_CAUSAL_OWNER" if final_clears else
                    "ZDF_SOLVER_FIRST_OWNER_CAPABLE_BOUNDARY"
                    if pre_clears else
                    "PRE_ZDF_ACCUMULATOR_DEBT_NO_ZDF_OWNER"),
            }

        transport_stage1_state = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_tracer_transport_stage=1),
        ).step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
        )
        transport_stage1_state = host_result_and_release_compilation(
            transport_stage1_state)
        transport_stage1_fields = lego_fields(transport_stage1_state)
        for name, candidate_name, mask_name in (
            ("zFu", "u", "u"), ("zFv", "v", "v"), ("zFw", "T", "T")
        ):
            if name == "zFw":
                # stprk3_stg.F90:286-304 leaves zFw for the vector-invariant
                # branch to tra_adv_trp, but the legacy transport instrument is
                # positioned earlier at :309-320.  Its structural zero is not
                # the tracer-consumed vertical transport and cannot adjudicate
                # the WZV arm.
                transport_row = {
                    "name": f"{CASE}.kt1.stage1.transport.{name}",
                    "status": "UNINFORMATIVE",
                    "oracle_nonfinite_count": transports[1]["zFw_nonfinite_count"],
                    "reason": (
                    "oracle zFw was dumped before tra_adv_trp fills it in the "
                    "vector-invariant branch"),
                }
            else:
                transport_row = score(
                    f"{CASE}.kt1.stage1.transport.{name}",
                    transports[1][name][..., :nlev],
                    transport_stage1_fields[candidate_name],
                    masks[mask_name],
                )
            stage1_transport_rows.append(transport_row)

        if plant_stage2_thermodynamics:
            planted = stage1_thermodynamic_rows[0]
            require(
                planted["status"] == "DEBT" and planted["absolute_max"] >= 1.0,
                "planted stage-1 thermodynamic operand did not fire",
            )
            stage2_thermodynamic_control = {
                "name": "control.planted_stage1_thermodynamic_operand",
                "status": "VERIFIED",
                "observed_absolute_max": planted["absolute_max"],
                "expected_minimum": 1.0,
            }

        oracle_bundle = (
            jnp.asarray(stages[1]["T"][..., :nlev]),
            jnp.asarray(stages[1]["S"][..., :nlev]),
            jnp.asarray(stages[1]["ssh"]),
        )
        injected_stage2 = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_stage=2,
                stage2_thermodynamic_override=oracle_bundle,
            ),
        ).step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
        )
        injected_stage2 = host_result_and_release_compilation(injected_stage2)
        injected_fields = lego_fields(injected_stage2)
        faithful_stage2 = {
            row["name"].rsplit(".", 1)[-1]: row
            for row in stage_rows
            if row["name"] in {
                f"{CASE}.kt1.stage2.u", f"{CASE}.kt1.stage2.v"
            }
        }
        faithful_stage2_fields = lego_fields(candidate_stage_states[2])
        movements = []
        faithful_residuals = []
        for component in ("u", "v"):
            reference = stages[2][component][..., :nlev]
            row = score(
                f"{CASE}.kt1.stage2.oracle_stage1_thermodynamic_bundle.{component}",
                reference,
                injected_fields[component],
                masks[component],
            )
            stage2_thermodynamic_arm_rows.append(row)
            faithful_residuals.append(
                faithful_stage2[component]["absolute_max"])
            faithful_candidate = faithful_stage2_fields[component]
            movements.append(float(np.max(np.abs(
                np.asarray(injected_fields[component])[masks[component]]
                - np.asarray(faithful_candidate)[masks[component]]))))
        residual = max(faithful_residuals)
        movement = max(movements)
        clears = all(
            row["status"] == "AT-BAR"
            for row in stage2_thermodynamic_arm_rows)
        stage2_thermodynamic_scaling = {
            "faithful_stage2_residual": residual,
            "causal_movement": movement,
            "movement_over_faithful_residual": movement / residual if residual else None,
            "arm_residual": max(
                row["absolute_max"] for row in stage2_thermodynamic_arm_rows),
            "scaling_check_before_owner_label": True,
            "owner_label": (
                "CONFIRMED_CAUSAL_OWNER" if clears else
                "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
                if movement >= 0.9 * residual else
                "NEAR_NULL_NO_DISCRIMINATING_POWER"
                if movement < 0.1 * residual else
                "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
            ),
        }

        oracle_stage2_components = {
            "hpg_u": stage2_terms["after_hpg_u"] - stage2_terms["before_u"],
            "hpg_v": stage2_terms["after_hpg_v"] - stage2_terms["before_v"],
            "vorticity_u": (
                stage2_terms["after_vorticity_u"]
                - stage2_terms["after_hpg_u"]),
            "vorticity_v": (
                stage2_terms["after_vorticity_v"]
                - stage2_terms["after_hpg_v"]),
            "advection_u": (
                stage2_terms["after_advection_u"]
                - stage2_terms["after_vorticity_u"]),
            "advection_v": (
                stage2_terms["after_advection_v"]
                - stage2_terms["after_vorticity_v"]),
        }
        _bt_ops = card.recipe.z_coord.nemo_een_barotropic
        reference_face_thickness = {
            "u": np.asarray(_bt_ops.e3u_0)[..., :nlev],
            "v": np.asarray(_bt_ops.e3v_0)[..., :nlev],
        }

        def _reference_baroclinic(array, component):
            active = np.asarray(masks[component], dtype=np.float64)
            weights = reference_face_thickness[component] * active
            depth = np.sum(weights, axis=-1)
            mean = np.sum(np.asarray(array) * weights, axis=-1) / np.maximum(
                depth, 1.0e-30)
            return (np.asarray(array) - mean[..., None]) * active

        ordered_operators = ("hpg", "vorticity", "advection")
        limit_index = ordered_operators.index(stage2_term_limit)
        operators_to_measure = (
            () if stage2_term_only == "none"
            else (stage2_term_only,) if stage2_term_only is not None
            else ordered_operators[:limit_index + 1])
        for operator in operators_to_measure:
            exposed = LatLonCGridOceanModel(
                card.recipe.grid,
                card.recipe.z_coord,
                cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    expose_momentum_operator=operator),
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater0,
                surface_forcing=surface0,
            )
            exposed = host_result_and_release_compilation(exposed)
            exposed_fields = lego_fields(exposed)
            for component in ("u", "v"):
                name = f"{operator}_{component}"
                candidate = exposed_fields[component]
                if plant_stage2_term and name == "hpg_u":
                    candidate = np.asarray(candidate).copy()
                    first = tuple(np.argwhere(masks[component])[0])
                    candidate[first] += 2.0
                oracle_component = oracle_stage2_components[name][..., :nlev]
                raw_row = score(
                    f"{CASE}.kt1.stage2.source_term.raw.{name}",
                    oracle_component,
                    candidate,
                    masks[component],
                )
                raw_row["status"] = "UNINFORMATIVE"
                raw_row["reason"] = (
                    "raw source term contains a depth-mean gauge that the "
                    "stage Kaa barotropic replacement removes")
                raw_row["source_order"] = (
                    "dyn_hpg -> dyn_vor -> dyn_adv; stprk3_stg.F90:321-367")
                stage2_term_rows.append(raw_row)
                row = score(
                    f"{CASE}.kt1.stage2.source_term.baroclinic.{name}",
                    _reference_baroclinic(oracle_component, component),
                    _reference_baroclinic(candidate, component),
                    masks[component],
                )
                row["source_order"] = raw_row["source_order"]
                row["projection"] = (
                    "reference e3u_0/hu_0 or e3v_0/hv_0 depth mean; "
                    "stprk3_stg.F90:433-446")
                stage2_term_rows.append(row)
                if row["status"] == "DEBT" and stage2_term_first_over_bar is None:
                    stage2_term_first_over_bar = {
                        "operator": operator,
                        "component": component,
                        "absolute_max": row["absolute_max"],
                        "reference_max_abs": row["reference_max_abs"],
                    }
            if stage2_term_first_over_bar is not None:
                break
        measured_operators = {
            row["name"].rsplit(".", 1)[-1].rsplit("_", 1)[0]
            for row in stage2_term_rows
            if ".baroclinic." in row["name"]
        }
        for operator in ordered_operators:
            if operator in measured_operators:
                continue
            for component in ("u", "v"):
                stage2_term_rows.append({
                    "name": (
                        f"{CASE}.kt1.stage2.source_term.baroclinic."
                        f"{operator}_{component}"),
                    "status": "UNMEASURED",
                    "reason": (
                        "ordered walk stopped at the preceding first-over-bar "
                        "boundary" if stage2_term_first_over_bar is not None
                        else (
                            f"isolated probe selected {stage2_term_only}"
                            if stage2_term_only is not None
                            else f"probe limit is {stage2_term_limit}")),
                })
        if plant_stage2_term:
            planted_term = next(
                row for row in stage2_term_rows
                if row["name"].endswith("baroclinic.hpg_u"))
            require(
                planted_term["status"] == "DEBT"
                and planted_term["absolute_max"] >= 1.0,
                "planted stage-2 source-term violation did not fire",
            )
            stage2_term_control = {
                "name": "control.planted_stage2_source_term",
                "status": "VERIFIED",
                "observed_absolute_max": planted_term["absolute_max"],
                "expected_minimum": 1.0,
            }

        if measure_stage2_update_arm:
            legacy_stage2 = LatLonCGridOceanModel(
                card.recipe.grid,
                card.recipe.z_coord,
                cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    expose_momentum_stage=2,
                    legacy_preproject_stage_rhs=True,
                ),
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater0,
                surface_forcing=surface0,
            )
            legacy_stage2 = host_result_and_release_compilation(legacy_stage2)
            legacy_fields = lego_fields(legacy_stage2)
            legacy_rows = []
            movements = []
            for component in ("u", "v"):
                legacy_rows.append(score(
                    f"{CASE}.kt1.stage2.legacy_preproject_rhs.{component}",
                    stages[2][component][..., :nlev],
                    legacy_fields[component],
                    masks[component],
                ))
                movements.append(float(np.max(np.abs(
                    np.asarray(legacy_fields[component])[masks[component]]
                    - np.asarray(faithful_stage2_fields[component])[
                        masks[component]]))))
            production_rows = [
                faithful_stage2[component] for component in ("u", "v")]
            production_residual = max(
                row["absolute_max"] for row in production_rows)
            legacy_residual = max(row["absolute_max"] for row in legacy_rows)
            movement = max(movements)
            stage2_update_arm = {
                "status": "MEASURED",
                "classification": "DIAGNOSTIC_ONE_VARIABLE_CAUSAL_ARM",
                "production_rows": production_rows,
                "legacy_preproject_rows": legacy_rows,
                "production_residual": production_residual,
                "legacy_residual": legacy_residual,
                "causal_movement": movement,
                "movement_over_legacy_residual": (
                    movement / legacy_residual if legacy_residual else None),
                "scaling_check_before_owner_label": True,
                "owner_label": (
                    "CONFIRMED_CAUSAL_OWNER_OF_STAGE2_ASSOCIATION"
                    if all(row["status"] == "AT-BAR" for row in production_rows)
                    and all(row["status"] == "DEBT" for row in legacy_rows)
                    else "NEAR_NULL_NO_DISCRIMINATING_POWER"
                    if movement < 0.1 * legacy_residual
                    else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"),
                "source": "stprk3_stg.F90:396-446",
            }

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
        "omit_barotropic_substep_drag": {
            "changed_operands": ["barotropic_substep_drag"],
            "hooks": _NEMOWSRK3TestHooks(
                omit_barotropic_substep_drag=True),
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
        vertical_control = host_result_and_release_compilation(vertical_control)
        qsr_delta = jnp.asarray(oracle_qsr_tendency - lego_qsr_tendency)
        qsr_control = model.step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
            _shortwave_tendency_test_delta=qsr_delta,
        )
        qsr_control = host_result_and_release_compilation(qsr_control)
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
            control = host_result_and_release_compilation(control)
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
                momentum_flux_scheme=_NEMO_UP3_NAME,
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
            control = host_result_and_release_compilation(control)
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

    instrumentation_identity_rows = list(stage3_identity_rows)
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
    status = (
        "AT-BAR" if first_over_bar is None
        and barotropic_state_first_over_bar is None else "DEBT")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-phase3-v2",
        "case": CASE,
        "scalar_math_root_identity": scalar_math_root_identity,
        "status": status,
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "execution_regime": "production_jit",
        "oracle_root": str(root),
        "max_step": max_step,
        "continue_after_first": True,
        "first_over_bar": first_over_bar,
        "barotropic_state_first_over_bar": barotropic_state_first_over_bar,
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
            "bottom_drag_scheme": cfg.bottom_drag.bottom_drag_scheme,
            "zdf_drag_in_matrix": cfg.zdf_drag_in_matrix,
            "zdf_baroclinic_only": cfg.zdf_baroclinic_only,
            "barotropic_drag_substep": cfg.barotropic_drag_substep,
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
            "barotropic_frames": {
                str(kt): {
                    "kt": frame["kt"], "Kaa": frame["Kaa"],
                    "registry_level": frame["registry_level"],
                }
                for kt, frame in bt.items()
            },
            "momentum_rows": stage_rows,
            "tracer_stage_numerics": (
                "STAGES1_2_MEASURED; STAGE3_IS_WHOLE_STEP"
                if measure_stage2_tracers else
                "STAGE1_MEASURED; LATER_STAGES_UNMEASURED"),
            "barotropic_frame_numerics": "MEASURED_WITHIN_SUBSTEP_LOOP",
        },
        "rk_stage2_thermodynamic_boundary": {
            "source_order": [
                "stage1_tracer_Kaa", "stage2_eos_hpg", "stage2_vorticity",
                "stage2_advection", "stage2_corrected_Kaa",
            ],
            "stage1_operand_rows": stage1_thermodynamic_rows,
            "stage1_transport_rows": stage1_transport_rows,
            "stage2_operand_rows": stage2_tracer_rows,
            "oracle_bundle_causal_rows": stage2_thermodynamic_arm_rows,
            "causal_scaling": stage2_thermodynamic_scaling,
            "planted_control": stage2_thermodynamic_control,
        },
        "rk_stage2_source_order_walk": {
            "oracle_artifact": str(stage2_term_path),
            "instrumentation_state_bit_identity": stage2_identity_sha,
            "source_order": ["hpg", "vorticity", "advection"],
            "rows": stage2_term_rows,
            "first_over_bar": stage2_term_first_over_bar,
            "scaling_check_before_owner_label": True,
            "owner_label": (
                "OPERAND_FIRST_DIVERGENCE_ONLY_NO_CAUSAL_OWNER"
                if stage2_term_first_over_bar is not None
                else "ALL_DIRECT_SOURCE_TERMS_AT_BAR"),
            "planted_control": stage2_term_control,
        },
        "rk_stage2_update_association_arm": stage2_update_arm,
        "rk_stage3_completion_walk": stage3_completion,
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
        "bottom_drag_operand_walk": drag_operand_walk,
        "instrumentation_bit_identity": instrumentation_identity_rows,
        "registry_rows": registry_rows,
        "resolved_program_coverage": coverage_rows,
        "arm_manifest_rows": arm_rows,
        "one_variable_arms": arms,
        "causal_oracle_injection_arms": causal_arms,
        "operator_scaling_before_owner": operator_scaling,
        "owner_verdict": (
            "BOTTOM_DRAG_CONFIRMED_AT_SUBSTEP2; NEXT_BOUNDARY_FROM_REGISTER"
            if drag_owner_label == "CONFIRMED_CAUSAL_OWNER_AT_SUBSTEP2"
            else "BOTTOM_DRAG_NOT_SOLE_OWNER_AT_SUBSTEP2"
        ),
        "growth_characterization": _trajectory_growth(steps),
        "steps": steps,
        "barotropic_state_steps": barotropic_state_steps,
        "artifacts": artifacts,
        "unmeasured": ([] if ene_available else [
            "ENE-coefficient arm (coefficients, products, term "
            "reconstruction and the ENE causal-owner labels): WAIVED -- the "
            "instrumented GYRE run never emitted "
            "oracle_bt_ene_coeff_kt00000001.bin",
        ]) + ([] if measure_stage2_tracers else [
            "numerical T/S agreement at internal stage-2 Kaa",
        ]) + ([] if measure_stage3_completion else [
            "source-ordered stage-3 pre-ZDF tracer completion",
        ]) + [
            "metric-weighted zhU/zhV transport comparison (metrics not dumped in this round)",
            "a two-sided source-isolated owner for the first over-bar whole-step row",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=ROOT)
    parser.add_argument("--stage2-oracle-root", type=Path, default=STAGE2_ROOT)
    parser.add_argument("--stage3-oracle-root", type=Path, default=STAGE3_ROOT)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-state", action="store_true")
    parser.add_argument("--plant-registry", action="store_true")
    parser.add_argument("--plant-arm", action="store_true")
    parser.add_argument("--plant-coverage", action="store_true")
    parser.add_argument("--plant-barotropic", action="store_true")
    parser.add_argument("--plant-ene-coefficient", action="store_true")
    parser.add_argument("--plant-drag", action="store_true")
    parser.add_argument("--plant-stage2-thermodynamics", action="store_true")
    parser.add_argument("--plant-stage2-term", action="store_true")
    parser.add_argument("--plant-barotropic-state", action="store_true")
    parser.add_argument(
        "--stage2-term-limit",
        choices=("hpg", "vorticity", "advection"),
        default="hpg",
        help="last source-order operator eligible for measurement",
    )
    parser.add_argument(
        "--stage2-term-only",
        choices=("none", "hpg", "vorticity", "advection"),
        help="measure one operator in an isolated low-memory process",
    )
    parser.add_argument("--measure-stage2-update-arm", action="store_true")
    parser.add_argument("--measure-stage2-tracers", action="store_true")
    parser.add_argument("--measure-stage3-completion", action="store_true")
    parser.add_argument(
        "--trajectory-only", action="store_true",
        help=("score only the production-JIT kt trajectory and return before "
              "compiling private stage/operator hooks"),
    )
    parser.add_argument(
        "--without-oracle-ene-coefficients", action="store_true",
        help=("WAIVE the ENE-coefficient arm because the instrumented GYRE "
              "run never emitted oracle_bt_ene_coeff_kt00000001.bin. The arm "
              "is recorded as UNMEASURED; every other row and the DEBT "
              "verdict are unchanged. Required to score a tree against the "
              "oracle set that exists."))
    from legoesm.ocean.fidelity.ulp_move_gate import (
        add_ulp_compare_arguments,
        capture_residual_fields,
        comparison_exit_code,
        persist_ulp_comparison,
        run_ulp_comparison,
        write_residual_artifact,
    )

    add_ulp_compare_arguments(parser)
    args = parser.parse_args(argv)
    try:
        with capture_residual_fields() as residuals:
            report = run(
                args.oracle_root,
                stage2_root=args.stage2_oracle_root,
                stage3_root=args.stage3_oracle_root,
                max_step=args.max_step,
                plant_state=args.plant_state,
                plant_registry=args.plant_registry,
                plant_arm=args.plant_arm,
                plant_coverage=args.plant_coverage,
                plant_barotropic=args.plant_barotropic,
                plant_ene_coefficient=args.plant_ene_coefficient,
                plant_drag=args.plant_drag,
                plant_stage2_thermodynamics=args.plant_stage2_thermodynamics,
                plant_stage2_term=args.plant_stage2_term,
                plant_barotropic_state=args.plant_barotropic_state,
                stage2_term_limit=args.stage2_term_limit,
                stage2_term_only=args.stage2_term_only,
                measure_stage2_update_arm=args.measure_stage2_update_arm,
                measure_stage2_tracers=args.measure_stage2_tracers,
                measure_stage3_completion=args.measure_stage3_completion,
                without_oracle_ene_coefficients=(
                    args.without_oracle_ene_coefficients),
                trajectory_only=args.trajectory_only,
            )
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    if args.output:
        write_residual_artifact(report, args.output, residuals)
    elif args.compare_to:
        print(
            "FAIL: --compare-to requires --output for the residual sidecar",
            file=sys.stderr,
        )
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.compare_to:
        comparison = run_ulp_comparison(args, report)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        print(persist_ulp_comparison(args, comparison))
        return comparison_exit_code(comparison)
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

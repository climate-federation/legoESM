#!/usr/bin/env python3
"""Non-vacuous kt=2 ORCA2 SH2 score on rank-zero owned cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from legoesm.ocean.physics.vertical_mixing._shared import (
    avm_weighted_shear_production,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2p_zdf_acquisition_gate as schema,
    nemo_testcase_l4_orca2_phase2s_zdf_acquisition_gate as acquisition,
)

NX, NY, NZ, HALO = schema.NX, schema.NY, schema.NZ, schema.HALO
OWN_X, OWN_Y, ACTIVE_Z = NX - 2 * HALO, NY - 2 * HALO, NZ - 1
RECORD = "oracle_zdf_sh2_operands_kt00000002.bin"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_arrays(path: Path, mesh: Path, *, require_shear: bool = True
                ) -> tuple[dict[str, np.ndarray], dict]:
    metadata = acquisition.read_zdf_v2(
        path, mesh, require_shear=require_shear)
    extents = [tuple(row) for row in metadata["extent_table"]]
    fields = list(zip(
        schema.FIELDS_3D + schema.FIELDS_2D,
        schema.ALLOCATION_3D + schema.ALLOCATION_2D,
        extents,
        strict=True,
    ))
    arrays: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        handle.seek(acquisition.PAYLOAD_OFFSET)
        for name, _allocation, extent in fields:
            count = int(np.prod(extent))
            values = np.fromfile(handle, dtype=np.float64, count=count)
            require(values.size == count, f"{name}: truncated")
            arrays[name] = values.reshape(extent, order="F")
        require(handle.read(1) == b"", "array walk did not stop at exact EOF")
    return arrays, metadata


def full3(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    return arrays[name].transpose(1, 0, 2)


def owned3(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = arrays[name].transpose(1, 0, 2)
    index = (schema.FIELDS_3D + schema.FIELDS_2D).index(name)
    allocation = (schema.ALLOCATION_3D + schema.ALLOCATION_2D)[index]
    return value[HALO:-HALO, HALO:-HALO] if allocation == "full" else value


def _ordered_bits(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where(bits & sign, ~bits, bits | sign)


def _cell_class(index: tuple[int, int, int], wet: np.ndarray) -> str:
    j, i, k = index
    if j == wet.shape[0] - 1:
        return "north-fold-row"
    if not wet[max(0, j - 1):j + 2, max(0, i - 1):i + 2, k].all():
        return "coast-or-bottom-adjacent"
    return "interior"


def score(candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    candidate = np.asarray(candidate, np.float64)
    target = np.asarray(target, np.float64)
    mask = np.asarray(mask, bool)
    require(candidate.shape == target.shape == mask.shape, "score shape mismatch")
    unequal = mask & (candidate.view(np.uint64) != target.view(np.uint64))
    ulp = np.abs(
        _ordered_bits(candidate[mask]).astype(object)
        - _ordered_bits(target[mask]).astype(object)
    )
    row = {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(mask.sum()),
        "max_abs": float(np.max(np.abs(candidate[mask] - target[mask]), initial=0.0)),
        "max_row_scale_ulp": int(np.max(ulp, initial=0)),
    }
    if unequal.any():
        first = tuple(int(v) for v in np.argwhere(unequal)[0])
        row.update({
            "first_zero_based_j_i_k": list(first),
            "first_candidate": float(candidate[first]),
            "first_target": float(target[first]),
            "first_cell_class": _cell_class(first, mask),
        })
    return row


def localization(candidate: np.ndarray, target: np.ndarray,
                 mask: np.ndarray) -> dict:
    unequal = (np.asarray(candidate).view(np.uint64)
               != np.asarray(target).view(np.uint64)) & mask
    return {
        "per_j": np.count_nonzero(unequal, axis=(1, 2)).tolist(),
        "per_i": np.count_nonzero(unequal, axis=(0, 2)).tolist(),
        "per_k": np.count_nonzero(unequal, axis=(0, 1)).tolist(),
        "west_owned_edge": int(np.count_nonzero(unequal[:, 0, :])),
        "east_owned_edge": int(np.count_nonzero(unequal[:, -1, :])),
        "south_owned_edge": int(np.count_nonzero(unequal[0, :, :])),
        "north_fold_owned_edge": int(np.count_nonzero(unequal[-1, :, :])),
        "strict_owned_interior": int(np.count_nonzero(unequal[1:-1, 1:-1, :])),
    }


def validate(deck: Path, root: Path, mesh: Path, plant: bool) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")

    path = root / RECORD
    arrays, metadata = read_arrays(path, mesh)
    card = build_orca2_zps_card(deck)
    cfg = card.recipe.model_config.physics.vertical_mixing.tke
    tuple_got = (
        cfg.tke_shear_production, cfg.tke_shear_avm_weighting,
        cfg.tke_shear_evaluation_stage, cfg.tke_shear_metric_source,
    )
    tuple_expected = (
        "nemo_face_native_nbb2", "nemo_face", "step_entry",
        "nemo_qco_live_face",
    )
    require(tuple_got == tuple_expected,
            f"ORCA2 production tuple {tuple_got} != {tuple_expected}")
    header = metadata["header"]
    require(header[1:5] == [2, 3, 3, 1],
            f"kt=2 executed levels are not Kbb=Kmm=3: {header[1:5]}")

    # NEMO east/north faces -> legoESM west/south faces.  Evaluate on a
    # one-halo T tile and crop the result: passing only the 90-column owned
    # tile would make the shared kernel's global periodic avm sum wrap at the
    # MPI partition edge, which is a harness artifact rather than production.
    u_now = full3(arrays, "u_Kmm")[1:151, 0:93, :ACTIVE_Z]
    u_before_record = full3(arrays, "u_Kbb")[1:151, 0:93, :ACTIVE_Z]
    v_now = full3(arrays, "v_Kmm")[0:151, 1:93, :ACTIVE_Z]
    v_before_record = full3(arrays, "v_Kbb")[0:151, 1:93, :ACTIVE_Z]
    require(np.array_equal(u_now, u_before_record)
            and np.array_equal(v_now, v_before_record),
            "executed Kbb=Kmm aliases do not carry identical velocities")
    u_mask = full3(arrays, "umask")[1:151, 0:93, :ACTIVE_Z]
    v_mask = full3(arrays, "vmask")[0:151, 1:93, :ACTIVE_Z]
    metrics = (
        full3(arrays, "e3uw_Kmm")[1:151, 0:93, 1:ACTIVE_Z],
        full3(arrays, "e3uw_Kbb")[1:151, 0:93, 1:ACTIVE_Z],
        full3(arrays, "e3vw_Kmm")[0:151, 1:93, 1:ACTIVE_Z],
        full3(arrays, "e3vw_Kbb")[0:151, 1:93, 1:ACTIVE_Z],
    )
    avm = full3(arrays, "avm_k_pre")[1:151, 1:93, 1:ACTIVE_Z]
    target = owned3(arrays, "sh2")[:, :, 1:ACTIVE_Z]
    wet = full3(arrays, "wumask")[2:150, 2:92, 1:ACTIVE_Z].astype(bool)
    # zdfsh2 writes at T/W points; the union of adjacent live U/V W faces is
    # the informative set, further constrained by the recorded T-column mask.
    wet = wet | full3(arrays, "wvmask")[2:150, 2:92, 1:ACTIVE_Z].astype(bool)
    wet &= full3(arrays, "e3w_Kmm")[2:150, 2:92, 1:ACTIVE_Z] > 0.0

    direct = jax.jit(avm_weighted_shear_production)
    args = tuple(jnp.asarray(x) for x in (
        u_now, v_now, u_now, v_now, np.ones_like(avm), u_mask, v_mask, avm
    ))
    metric_args = tuple(jnp.asarray(x) for x in metrics)
    production_tile = np.asarray(direct(*args, face_metrics=metric_args))
    production = production_tile[1:-1, 1:-1]
    # The independent source-literal call retains the formal Kmm/Kbb operands;
    # the real header proves they alias for this executed RK3 arm.
    literal_args = tuple(jnp.asarray(x) for x in (
        u_now, v_now, u_before_record, v_before_record,
        np.ones_like(avm), u_mask, v_mask, avm
    ))
    source_literal_tile = np.asarray(direct(*literal_args, face_metrics=metric_args))
    source_literal = source_literal_tile[1:-1, 1:-1]
    complete = wet.copy()
    complete[[0, -1], :, :] = False
    complete[:, [0, -1], :] = False
    rows = {
        "production_restored_tuple_all_owned": score(production, target, wet),
        "production_restored_tuple_record_complete_stencil": score(
            production, target, complete),
        "source_literal_formal_Kmm_Kbb_all_owned": score(
            source_literal, target, wet),
        "source_literal_formal_Kmm_Kbb_record_complete_stencil": score(
            source_literal, target, complete),
    }
    loc = localization(production, target, wet)

    plants = {}
    if plant:
        planted = target.copy()
        index = tuple(int(v) for v in np.argwhere(wet)[0])
        planted[index] = np.nextafter(planted[index], np.inf)
        row = score(production, planted, wet)
        require(row["unequal"] >= 1, "one-bit SH2 target plant did not fire")
        plants["sh2_target_one_bit"] = "PASS_NONZERO"
        wrong_tuple = tuple_expected[:2] + ("implicit_solve_state",) + tuple_expected[3:]
        try:
            require(wrong_tuple == tuple_expected, "selector tuple plant")
        except GateError:
            plants["selector_tuple"] = "PASS_NONZERO"
        else:
            raise GateError("selector tuple plant did not fire")
        raise GateError("SH2 one-bit/selector plants rejected through production gate")

    first_debt = rows["production_restored_tuple_record_complete_stencil"]
    if not first_debt["unequal"]:
        first_debt = None
    return {
        "status": "AT_BAR" if first_debt is None else "STOP_SHARED_SH2_DEBT",
        "execution": {"backend": jax.default_backend(), "jit": "production",
                      "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
        "record": {"path": str(path), "sha256": sha256(path),
                   "header": header, "exact_eof": metadata["exact_eof"]},
        "resolved_tuple": list(tuple_got),
        "time_level_resolution": {
            "formal_expression": "Kmm*Kbb",
            "executed_call": "zdf_phy(kstp,Nbb,Nbb,Nrhs) => Nbb*Nbb",
            "source": "zdfsh2.F90:80-89; stprk3.F90:163-165; zdfphy.F90:264-269",
        },
        "rows": rows,
        "all_owned_localization": loc,
        "record_complete_definition": (
            "exclude the outer owned i/j rows: their face/avm operands live "
            "in writer-canonicalized MPI/fold halos and cannot be reconstructed "
            "from this rank-zero record alone"),
        "first_departure": None if first_debt is None else {
            "statement": "zdfsh2.F90:80-94 face product/divisor or T-point coast sum",
            "owner": "GYRE_OWNER_SHARED_SH2",
            "cell": first_debt.get("first_zero_based_j_i_k"),
            "cell_class": first_debt.get("first_cell_class"),
            "scope": "the frame has no face-intermediate target; a WRITE-only zsh2u/zsh2v dump is required to discriminate line 80/85 from line 93",
        },
        "plants": plants,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, args.mesh, args.plant)
    except (GateError, acquisition.GateError, OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

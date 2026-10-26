#!/usr/bin/env python3
"""Cellwise ORCA2 kt=1 ``zdf_sh2`` entry discriminator.

The accepted cold-start frame has zero velocity in both Kbb and Kmm.  The
script therefore reports the exact SH2 result but refuses to treat that
zero-gradient row as arithmetic certification.  Its non-vacuous rows are the
three carried inputs that exist before ``zdf_sh2``: ``avm_k``, ``avt_k`` and
``en``.
"""

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

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy, get_policy, set_policy,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)
from legoesm.ocean.physics.vertical_mixing._shared import (  # noqa: E402
    avm_weighted_shear_production,
)

NX, NY, NZ, HALO = 94, 152, 31, 2
OWN_X, OWN_Y, ACTIVE_Z = NX - 4, NY - 4, NZ - 1
FIELDS_3D = (
    "sh2", "avm_k_pre", "avt_k_pre", "en_pre", "rn2", "rn2b",
    "u_Kbb", "u_Kmm", "v_Kbb", "v_Kmm",
    "e3uw_Kbb", "e3uw_Kmm", "e3vw_Kbb", "e3vw_Kmm",
    "umask", "vmask", "wumask", "wvmask", "gdepw_Kmm",
    "e3t_Kmm", "e3w_Kmm",
)
ALLOC_3D = ("reduced", "full", "reduced", "reduced", *("full",) * 17)
FIELDS_2D = ("taum", "fr_i", "rCdU_bot", "mbkt_real")
ALLOC_2D = ("reduced", "full", "full", "full")


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


def read_frame(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    full3, reduced3 = NX * NY * NZ, OWN_X * OWN_Y * NZ
    full2, reduced2 = NX * NY, OWN_X * OWN_Y
    expected_count = 3 * reduced3 + 18 * full3 + reduced2 + 3 * full2
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=13i", handle.read(52))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L4_ZSH2_1", f"bad magic {magic!r}")
    require(header == (1, 1, 1, 1, 3, NX, NY, NZ, 64, 21, 4,
                       expected_count, 1), f"bad header {header}")
    require(values.size == expected_count, "derived payload count mismatch")
    require(path.stat().st_size == 68 + 8 * expected_count,
            "schema did not stop at exact EOF")
    arrays: dict[str, np.ndarray] = {}
    allocations: dict[str, str] = {}
    cursor = 0
    for name, allocation in zip(FIELDS_3D, ALLOC_3D, strict=True):
        shape = ((NX, NY, NZ) if allocation == "full"
                 else (OWN_X, OWN_Y, NZ))
        count = int(np.prod(shape))
        arrays[name] = values[cursor:cursor + count].reshape(shape, order="F")
        allocations[name] = allocation
        cursor += count
    for name, allocation in zip(FIELDS_2D, ALLOC_2D, strict=True):
        shape = ((NX, NY) if allocation == "full" else (OWN_X, OWN_Y))
        count = int(np.prod(shape))
        arrays[name] = values[cursor:cursor + count].reshape(shape, order="F")
        allocations[name] = allocation
        cursor += count
    require(cursor == values.size, "field walk did not reach exact EOF")
    require(all(np.isfinite(value).all() for value in arrays.values()),
            "non-finite frame value")
    return arrays, allocations


def owned3(arrays: dict[str, np.ndarray], allocations: dict[str, str],
           name: str) -> np.ndarray:
    value = arrays[name].transpose(1, 0, 2)
    return value[HALO:-HALO, HALO:-HALO] if allocations[name] == "full" else value


def _ordered_bits(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where(bits & sign, ~bits, bits | sign)


def _cell_class(index: tuple[int, int, int], active: np.ndarray,
                e3t0: np.ndarray, e3t1d: np.ndarray) -> str:
    j, i, k = index
    if j == active.shape[0] - 1:
        return "north-fold-row"
    wet_levels = int(active[j, i].sum())
    if k == wet_levels - 1 and wet_levels and not np.array_equal(
            np.asarray(e3t0[j, i, k]), np.asarray(e3t1d[k])):
        return "partial-cell-bottom"
    neighbours = active[max(0, j-1):j+2, max(0, i-1):i+2, k]
    if not neighbours.all():
        return "coast"
    return "interior"


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray,
          *, active: np.ndarray, e3t0: np.ndarray,
          e3t1d: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate, np.float64)
    oracle = np.asarray(oracle, np.float64)
    mask = np.asarray(mask, bool)
    require(candidate.shape == oracle.shape == mask.shape, "score shape")
    require(np.isfinite(candidate[mask]).all() and np.isfinite(oracle[mask]).all(),
            "non-finite scored value")
    unequal_mask = mask & (candidate.view(np.uint64) != oracle.view(np.uint64))
    points = np.argwhere(unequal_mask)
    result: dict[str, object] = {
        "status": "AT_BAR" if not len(points) else "DEBT",
        "unequal": int(unequal_mask.sum()),
        "count": int(mask.sum()),
        "max_abs": float(np.max(np.abs(candidate[mask] - oracle[mask]),
                                initial=0.0)),
        "max_row_scale_ulp": int(np.max(np.abs(
            _ordered_bits(candidate[mask]).astype(object)
            - _ordered_bits(oracle[mask]).astype(object)), initial=0)),
    }
    if len(points):
        index = tuple(int(x) for x in points[0])
        result.update({
            "first_zero_based_j_i_k": list(index),
            "first_candidate": float(candidate[index]),
            "first_oracle": float(oracle[index]),
            "first_cell_class": _cell_class(index, active, e3t0, e3t1d),
        })
    return result


def validate(deck: Path, root: Path, *, plants: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy not active")

    record = root / "oracle_zdf_sh2_operands_kt00000001.bin"
    arrays, allocations = read_frame(record)
    card = build_orca2_zps_card(deck)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config, iwm_forcing=card.recipe.iwm_forcing)
    state = model._seed_tke_preclosure_carry(card.recipe.initial_state)
    tke_cfg = model.config.physics.vertical_mixing.tke
    active = np.asarray(card.recipe.z_coord.is_active)[:, :OWN_X, :ACTIVE_Z]
    wet_w = np.asarray(card.recipe.z_coord.is_active)[:, :OWN_X, 1:ACTIVE_Z]
    e3t0 = np.asarray(card.recipe.z_coord.nemo_e3t_0)[:, :OWN_X, :ACTIVE_Z]
    e3t1d = np.asarray(card.recipe.z_coord.dz_ref)[:ACTIVE_Z]

    target_sh2 = owned3(arrays, allocations, "sh2")[:, :, 1:ACTIVE_Z]
    production_sh2 = np.asarray(jax.jit(model._tke_step_entry_p_sh2)(state))[
        :, :OWN_X]
    source_rows = {
        "production_card_sh2": score(
            production_sh2, target_sh2, wet_w, active=active,
            e3t0=e3t0, e3t1d=e3t1d),
    }

    # Isolate the canonical face-native shared operator using the exact NEMO
    # frame operands.  NEMO stores east/north faces; legoESM stores redundant
    # west/south faces, hence the one-cell offset in these local-halo slices.
    def full3(name: str) -> np.ndarray:
        require(allocations[name] == "full", f"{name} is not full-domain")
        return arrays[name].transpose(1, 0, 2)

    u_now = full3("u_Kmm")[2:150, 1:92, :ACTIVE_Z]
    u_before = full3("u_Kbb")[2:150, 1:92, :ACTIVE_Z]
    v_now = full3("v_Kmm")[1:150, 2:92, :ACTIVE_Z]
    v_before = full3("v_Kbb")[1:150, 2:92, :ACTIVE_Z]
    u_mask = full3("umask")[2:150, 1:92, :ACTIVE_Z]
    v_mask = full3("vmask")[1:150, 2:92, :ACTIVE_Z]
    e3un = full3("e3uw_Kmm")[2:150, 1:92, 1:ACTIVE_Z]
    e3ub = full3("e3uw_Kbb")[2:150, 1:92, 1:ACTIVE_Z]
    e3vn = full3("e3vw_Kmm")[1:150, 2:92, 1:ACTIVE_Z]
    e3vb = full3("e3vw_Kbb")[1:150, 2:92, 1:ACTIVE_Z]
    avm = owned3(arrays, allocations, "avm_k_pre")[:, :, 1:ACTIVE_Z]
    literal = np.asarray(jax.jit(avm_weighted_shear_production)(
        jnp.asarray(u_now), jnp.asarray(v_now),
        jnp.asarray(u_before), jnp.asarray(v_before),
        jnp.ones_like(jnp.asarray(avm)), jnp.asarray(u_mask),
        jnp.asarray(v_mask), jnp.asarray(avm),
        face_metrics=(jnp.asarray(e3un), jnp.asarray(e3ub),
                      jnp.asarray(e3vn), jnp.asarray(e3vb))))
    source_rows["source_literal_face_native_sh2"] = score(
        literal, target_sh2, wet_w, active=active,
        e3t0=e3t0, e3t1d=e3t1d)

    target_avm = avm
    target_avt = owned3(arrays, allocations, "avt_k_pre")[:, :, 1:ACTIVE_Z]
    target_en = owned3(arrays, allocations, "en_pre")[:, :, 1:ACTIVE_Z]
    candidate_avm = np.asarray(state.tke_avm.data)[:, :OWN_X]
    candidate_avt = np.asarray(state.tke_avt.data)[:, :OWN_X]
    require(state.tke is not None, "ORCA2 card did not seed cold-start en")
    candidate_en = np.asarray(state.tke.data)[:, :OWN_X]
    entry_rows = {
        "avm_k_pre": score(candidate_avm, target_avm, wet_w,
                           active=active, e3t0=e3t0, e3t1d=e3t1d),
        "avt_k_pre": score(candidate_avt, target_avt, wet_w,
                           active=active, e3t0=e3t0, e3t1d=e3t1d),
        "en_pre": score(candidate_en, target_en, wet_w,
                         active=active, e3t0=e3t0, e3t1d=e3t1d),
    }

    controls: dict[str, str] = {}
    if plants:
        planted = target_sh2.copy()
        index = tuple(np.argwhere(wet_w)[0])
        planted[index] = np.nextafter(planted[index], np.inf)
        row = score(production_sh2, planted, wet_w, active=active,
                    e3t0=e3t0, e3t1d=e3t1d)
        require(row["unequal"] == 1, "SH2 one-bit target plant did not fire")
        controls["sh2_target_one_bit"] = "PASS_NONZERO"
        planted_avm = target_avm.copy()
        index = tuple(np.argwhere(wet_w)[0])
        planted_avm[index] = np.nextafter(planted_avm[index], np.inf)
        row = score(candidate_avm, planted_avm, wet_w, active=active,
                    e3t0=e3t0, e3t1d=e3t1d)
        require(row["unequal"] == 1, "entry-state one-bit plant did not fire")
        controls["preclosure_target_one_bit"] = "PASS_NONZERO"

    zero_operands = {
        name: int(np.count_nonzero(full3(name)))
        for name in ("u_Kbb", "u_Kmm", "v_Kbb", "v_Kmm")
    }
    require(all(value == 0 for value in zero_operands.values()),
            f"cold-start velocity operand unexpectedly nonzero: {zero_operands}")
    return {
        "status": "STOP_AT_ORCA2_SH2_SELECTOR_DEBT",
        "execution": {"backend": jax.default_backend(), "jit": "production",
                      "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
        "record": {"path": str(record), "sha256": sha256(record),
                   "bytes": record.stat().st_size,
                   "derived_payload_f64": sum(value.size for value in arrays.values())},
        "selectors": {
            "production_tke_shear_production": tke_cfg.tke_shear_production,
            "production_tke_shear_avm_weighting": tke_cfg.tke_shear_avm_weighting,
            "production_tke_shear_metric_source": tke_cfg.tke_shear_metric_source,
            "nemo_resolved": "face-native Nbb*Nbb at step entry, avm face sum, live QCO faces",
            "status": "ORCA2_CARD_SELECTOR_DEBT",
        },
        "source_rows": source_rows,
        "source_row_disposition": (
            "AT_BAR_VACUOUS_ZERO_GRADIENT: all four velocity operands are "
            "identically zero, so neither exact row certifies the selected "
            "shared arithmetic"),
        "zero_velocity_nonzero_counts": zero_operands,
        "preclosure_entry_rows": entry_rows,
        "first_departure": {
            "boundary": "ORCA2 card zdf_sh2 selector dispatch",
            "owner": "LANE4_ORCA2_CARD_SELECTOR",
            "source": (
                "zdfphy.F90:268; zdfsh2.F90:80-100; "
                "stprk3_stg.F90:146,290,297"),
            "detail": (
                "the three preclosure inputs are exact after the Lane-4 card "
                "initialization repair, but the card still selects "
                "squared_centered/tpoint/tpoint_jacobian instead of NEMO's "
                "face-native Nbb*Nbb-at-step-entry/avm-face/live-QCO combination"),
        },
        "shared_operator_handoff": {
            "owner": "GYRE_OWNER_SHARED_TKE",
            "status": "UNMEASURED_ZERO_GRADIENT",
            "reason": "no nonzero face shear exists in the kt=1 cold-start frame",
        },
        "plants": controls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--plants", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, plants=args.plants)
    except (GateError, OSError, ValueError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

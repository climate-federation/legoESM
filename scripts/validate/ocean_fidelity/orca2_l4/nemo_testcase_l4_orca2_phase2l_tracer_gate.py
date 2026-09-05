#!/usr/bin/env python3
"""ORCA2 stage-1 tracer boundary with oracle-supplied transports."""

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

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
    validate_nemo_testcase_card,
)
from legoesm.ocean.freshwater import FreshwaterForcing  # noqa: E402

NX, NY, NZ = 94, 152, 31
OWNED_NX, OWNED_NY, NLEV = NX - 4, NY - 4, NZ - 1
TRACER_RECORD = "oracle_rktracer_operands_kt00000001_s1.bin"
STAGE3_RECORD = "oracle_stage_kt00000001_s3.bin"
BT_RECORD = "oracle_bt_frames_kt00000001.bin"


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[2:-2, 2:-2].T


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_tracer(path: Path) -> dict[str, np.ndarray | dict[str, int]]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    expected = (1, 1, 1, 1, 1, 3, 3, NX, NY, NZ, 64)
    require(magic == "NEMO_L2_RKTRA_1", f"bad tracer magic {magic!r}")
    require(header == expected, f"bad tracer header {header}")
    n2, n3 = NX * NY, NX * NY * NZ
    require(values.size == 15 * n3 + 3 * n2, "bad tracer payload")
    require(np.isfinite(values).all(), "non-finite tracer payload")
    names = (
        "zero_T", "zero_S", "zFu", "zFv", "zFw",
        "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "Kbb_T", "Kbb_S",
        "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result: dict[str, np.ndarray | dict[str, int]] = {
        name: _xyz(values[index * n3:(index + 1) * n3])
        for index, name in enumerate(names)
    }
    result["header"] = {
        "version": 1, "kt": 1, "stage": 1, "Kbb": 1, "Kmm": 1,
        "Krhs": 3, "Kaa": 3, "jpi": NX, "jpj": NY, "jpk": NZ,
        "bits": 64, "payload_f64": int(values.size),
    }
    return result


def read_final_ssh(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_STAGE_1", f"bad stage magic {magic!r}")
    require(header == (1, 1, 3, 3, NX, NY, NZ, 2, 64), "bad stage header")
    n3, n2 = NX * NY * NZ, NX * NY
    require(values.size == 4 * n3 + n2, "bad stage payload")
    return _xy(values[4 * n3:])


def read_external_transports(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_BTFRM_1", f"bad BT magic {magic!r}")
    require(header == (1, 1, 3, NX, NY, 64), "bad BT header")
    n2 = NX * NY
    require(values.size == 4 * n2, "bad BT payload")
    return _xy(values[2 * n2:3 * n2]), _xy(values[3 * n2:])


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    candidate_grid = np.asarray(candidate, np.float64)
    oracle_grid = np.asarray(oracle, np.float64)
    unequal_grid = candidate_grid.view(np.uint64) != oracle_grid.view(np.uint64)
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.size and actual.shape == expected.shape, "empty tracer score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite defined tracer cell")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    ia, ie = actual.view(np.int64), expected.view(np.int64)
    oa = ia ^ ((ia >> 63) & 0x7fffffffffffffff)
    oe = ie ^ ((ie >> 63) & 0x7fffffffffffffff)
    bottom = mask & ~np.concatenate(
        [mask[..., 1:], np.zeros_like(mask[..., :1])], axis=-1)
    per_row_n = mask.sum(axis=(1, 2))
    per_row_bad = (unequal_grid & mask).sum(axis=(1, 2))
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()), "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
        "max_ulp": int(np.abs(oa - oe).max(initial=0)),
        "localization": {
            "per_level_unequal": (unequal_grid & mask).sum(axis=(0, 1)).tolist(),
            "bottom_unequal": int(np.count_nonzero(unequal_grid & bottom)),
            "interior_unequal": int(np.count_nonzero(
                unequal_grid & mask & ~bottom)),
            "per_row": [
                {"j_zero_based": int(j), "unequal": int(per_row_bad[j]),
                 "count": int(per_row_n[j]),
                 "fraction": float(per_row_bad[j] / per_row_n[j])}
                for j in range(mask.shape[0]) if per_row_n[j]
            ],
        },
    }


def validate(deck_root: Path, oracle_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "tracer gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    paths = {name: oracle_root / name
             for name in (TRACER_RECORD, STAGE3_RECORD, BT_RECORD)}
    for path in paths.values():
        require(path.is_file(), f"missing {path}")
    tracer = read_tracer(paths[TRACER_RECORD])
    final_ssh = read_final_ssh(paths[STAGE3_RECORD])
    un_adv, vn_adv = read_external_transports(paths[BT_RECORD])

    card = build_orca2_zps_card(deck_root)
    validate_nemo_testcase_card(card)
    cfg = card.recipe.model_config
    require(cfg.tracer_advection == "fct2",
            "ORCA2 card does not select FCT2/FCT2")
    require(cfg.momentum_advection == "vector_invariant" and
            cfg.ke_gradient_scheme == "c2", "ORCA2 card is not vector/C2")

    # Registered upstream shared constructibility gaps are run-and-discarded;
    # the exact external endpoint and stage-1 tracer transports are substituted
    # below, so none of these proxy values reaches the scored advection RHS.
    vmix = cfg.physics.vertical_mixing
    diagnostic_cfg = cfg._replace(
        een_e3f_scheme="min",
        lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0),
        physics=cfg.physics._replace(
            vertical_mixing=vmix._replace(
                tke=vmix.tke._replace(n2_eos_form="teos10")),
            convection=cfg.physics.convection._replace(
                enhanced_diffusion=(
                    cfg.physics.convection.enhanced_diffusion._replace(
                        n2_eos_form="teos10"))),
        ),
    )

    eta_after = np.asarray(card.recipe.initial_state.eta.data).copy()
    hu_avg = np.zeros(card.recipe.initial_state.u.data.shape[:2], np.float64)
    hv_avg = np.zeros(card.recipe.initial_state.v.data.shape[:2], np.float64)
    eta_after[:, :OWNED_NX] = final_ssh
    hu_avg[:, 1:OWNED_NX + 1] = un_adv
    hv_avg[1:OWNED_NY + 1, :OWNED_NX] = vn_adv

    shape = eta_after.shape + (NLEV,)
    zfu, zfv = (np.zeros(shape, np.float64) for _ in range(2))
    zfw = np.zeros(eta_after.shape + (NZ,), np.float64)
    zfu[:, :OWNED_NX] = np.asarray(tracer["zFu"])[..., :NLEV]
    zfv[:, :OWNED_NX] = np.asarray(tracer["zFv"])[..., :NLEV]
    zfw[:, :OWNED_NX] = np.asarray(tracer["zFw"])
    hooks = _NEMOWSRK3TestHooks(
        external_mode_result_override=(
            jnp.asarray(eta_after), jnp.asarray(hu_avg), jnp.asarray(hv_avg)),
        stage1_tracer_transport_override=(
            jnp.asarray(zfu), jnp.asarray(zfv), jnp.asarray(zfw)),
        expose_tracer_stage1_boundary="after_advection",
    )
    zeros = np.zeros(eta_after.shape, np.float64)
    freshwater = FreshwaterForcing(
        precip=jnp.asarray(zeros), evap=jnp.asarray(zeros),
        runoff=jnp.asarray(zeros), ice_fw=jnp.asarray(zeros),
        restoring=jnp.asarray(zeros),
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, diagnostic_cfg,
        _nemo_ws_test_hooks=hooks)
    result = model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater)

    candidate = {
        "T": np.asarray(result.T.data)[:, :OWNED_NX, :NLEV],
        "S": np.asarray(result.S.data)[:, :OWNED_NX, :NLEV],
    }
    expected = {
        name: np.asarray(tracer[f"after_advection_{name}"])[..., :NLEV]
        for name in ("T", "S")
    }
    # FCT2's stage-1 RK arm resolves to centered advection
    # (traadv.F90:280-283,355-365).  Exclude only cells whose one-cell stencil
    # reaches the absent rank-1 or tripolar-fold support values.
    wet = np.asarray(card.recipe.z_coord.is_active)[:, :OWNED_NX, :NLEV]
    support = np.zeros((OWNED_NY, OWNED_NX), bool)
    support[:-1, 1:-1] = True
    mask = wet & support[..., None]
    if plant:
        candidate = {name: expected[name].copy() for name in ("T", "S")}
        target = tuple(np.argwhere(mask)[0])
        candidate["T"][target] = np.nextafter(candidate["T"][target], np.inf)
    rows = {name: score(candidate[name], expected[name], mask)
            for name in ("T", "S")}
    if plant:
        require(rows["T"]["unequal"] == 1 and rows["S"]["unequal"] == 0,
                "tracer plant did not fire exactly once")
        raise GateError(
            "planted stage-1 tracer cell rejected through scorer "
            f"({rows['T']['unequal']}/{rows['T']['count']})")

    first = next((name for name, row in rows.items()
                  if row["status"] != "AT_BAR"), None)
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O6-A/stage1-centered-precursor-to-stage3-FCT",
        "result": "AT_BAR" if first is None else "DEBT",
        "first_over_bar_field": first,
        "owner": ("CONFIRMED_SHARED_STAGE1_CENTERED_TRACER_AT_BAR"
                  if first is None else "GYRE_OWNER_SHARED_TRACER_ADVECTION"),
        "rows": rows,
        "record": {"path": str(paths[TRACER_RECORD]),
                   "sha256": sha256(paths[TRACER_RECORD]),
                   "schema": tracer["header"]},
        "operand_substitution": {
            "external_mode": "ORACLE_SUPPLIED",
            "zFu_zFv_zFw": "ORACLE_SUPPLIED",
            "certifies_transport_or_external_mode": False,
        },
        "resolved": {
            "ln_traadv_fct": True, "nn_fct_h": 2, "nn_fct_v": 2,
            "nn_fct_imp": 1, "stage1_executed_arm": "centered_FCT2_precursor",
            "stage3_executed_arm": "two_step_FCT2",
        },
        "comparison_domain": (
            "rank0 wet T cells, 30 levels, excluding west/east MPI support "
            "columns and north-fold support row"
        ),
        "execution": {"backend": jax.default_backend(),
                      "production_jit": True, "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, plant=args.plant)
    except (GateError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

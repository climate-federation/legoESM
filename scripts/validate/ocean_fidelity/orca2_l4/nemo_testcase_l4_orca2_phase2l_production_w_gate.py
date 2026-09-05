#!/usr/bin/env python3
"""Adjudicate ORCA2 stage-1 WZV on the compiled production step."""

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
OWNED_NX, OWNED_NY = NX - 4, NY - 4
NLEV = NZ - 1
WZV_RECORD = "oracle_stage1_wzv_operands_kt00000001.bin"
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


def read_wzv(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=14i", handle.read(56))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L4_WZVS1_1", f"bad WZV magic {magic!r}")
    require(
        header == (1, 1, 1, 1, 1, 3, NX, NY, NZ, 2, NX - 2, 2, NY - 2, 64),
        f"bad WZV header {header}",
    )
    n2, n3, ns = NX * NY, NX * NY * NZ, (NX - 3) * (NY - 3) * NZ
    require(values.size == 2 * ns + 5 * n3 + 4 * n2, "bad WZV payload")
    offset = 2 * ns
    e3t = _xyz(values[offset:offset + n3]); offset += n3
    e3t0 = _xyz(values[offset:offset + n3]); offset += n3
    tmask = _xyz(values[offset:offset + n3]); offset += n3
    r3bb = _xy(values[offset:offset + n2]); offset += n2
    r3aa = _xy(values[offset:offset + n2]); offset += n2
    r1_area = _xy(values[offset:offset + n2]); offset += n2
    runoff = _xy(values[offset:offset + n2]); offset += n2
    ww = _xyz(values[offset:offset + n3]); offset += n3
    pfw = _xyz(values[offset:offset + n3]); offset += n3
    require(offset == values.size, "WZV walk did not reach EOF")
    require(np.isfinite(values).all(), "non-finite WZV payload")
    return {
        "e3t": e3t, "e3t0": e3t0, "tmask": tmask,
        "r3bb": r3bb, "r3aa": r3aa, "r1_area": r1_area,
        "runoff": runoff, "ww": ww, "pfw": pfw,
    }


def read_final_ssh(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_STAGE_1", f"bad stage magic {magic!r}")
    require(header == (1, 1, 3, 3, NX, NY, NZ, 2, 64), f"bad stage header {header}")
    n3, n2 = NX * NY * NZ, NX * NY
    require(values.size == 4 * n3 + n2, "bad stage payload")
    return _xy(values[4 * n3:])


def read_external_transports(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_BTFRM_1", f"bad BT magic {magic!r}")
    require(header == (1, 1, 3, NX, NY, 64), f"bad BT header {header}")
    n2 = NX * NY
    require(values.size == 4 * n2, "bad BT payload")
    return _xy(values[2 * n2:3 * n2]), _xy(values[3 * n2:])


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.shape == expected.shape and actual.size, "empty production-W score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite production-W value")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    ia, ie = actual.view(np.int64), expected.view(np.int64)
    oa = ia ^ ((ia >> 63) & 0x7fffffffffffffff)
    oe = ie ^ ((ie >> 63) & 0x7fffffffffffffff)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
        "max_ulp": int(np.abs(oa - oe).max(initial=0)),
    }


def validate(deck_root: Path, oracle_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "production-W gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    paths = {name: oracle_root / name for name in (WZV_RECORD, STAGE3_RECORD, BT_RECORD)}
    for path in paths.values():
        require(path.is_file(), f"missing {path}")
    wzv = read_wzv(paths[WZV_RECORD])
    final_ssh = read_final_ssh(paths[STAGE3_RECORD])
    un_adv, vn_adv = read_external_transports(paths[BT_RECORD])

    card = build_orca2_zps_card(deck_root)
    validate_nemo_testcase_card(card)
    cfg = card.recipe.model_config
    require(cfg.momentum_advection == "vector_invariant",
            "ORCA2 card does not select vector momentum")
    require(cfg.ke_gradient_scheme == "c2",
            "ORCA2 card does not select nn_dynkeg=0/C2")

    # The card's selected EOS-80 TKE/EVD entry is itself a downstream
    # UNMEASURED shared boundary: the shared bn2 closure currently refuses
    # eos_form='eos80'.  Use the already constructible TEOS-10 bn2 only in
    # upstream tendencies whose external result is replaced below.  No proxy
    # value reaches _g0 or the exposed WZV value.
    vmix = cfg.physics.vertical_mixing
    tke = vmix.tke._replace(n2_eos_form="teos10")
    convection = cfg.physics.convection._replace(
        enhanced_diffusion=cfg.physics.convection.enhanced_diffusion._replace(
            n2_eos_form="teos10"))
    diagnostic_cfg = cfg._replace(
        # The production EEN momentum tendency has not yet acquired the
        # ORCA2 tripolar avg4 fold.  It is upstream of the substituted
        # external endpoint and cannot reach the exposed _g0 value.
        een_e3f_scheme="min",
        # The selected spatial nemo_div_curl viscosity is another unmeasured
        # upstream tripolar operator.  Zero it only in the discarded tendency.
        lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0),
        physics=cfg.physics._replace(
            vertical_mixing=vmix._replace(tke=tke),
            convection=convection))

    # The external mode is a registered upstream shared debt.  Substitute its
    # NEMO endpoint over the measured rank-zero slab, while retaining a full
    # global production card.  The east partition support column and north
    # fold support row are excluded below because rank zero did not dump their
    # remote operands.
    eta_after = np.asarray(card.recipe.initial_state.eta.data).copy()
    hu_avg = np.zeros(eta_after.shape, dtype=np.float64)
    hv_avg = np.zeros(eta_after.shape, dtype=np.float64)
    eta_after[:, :OWNED_NX] = final_ssh
    hu_avg[:, :OWNED_NX] = un_adv
    hv_avg[:, :OWNED_NX] = vn_adv
    zeros = np.zeros(eta_after.shape, dtype=np.float64)
    runoff = zeros.copy()
    runoff[:, :OWNED_NX] = wzv["runoff"]
    freshwater = FreshwaterForcing(
        precip=jnp.asarray(zeros), evap=jnp.asarray(zeros),
        runoff=jnp.asarray(runoff), ice_fw=jnp.asarray(zeros),
        restoring=jnp.asarray(zeros),
    )
    hooks = _NEMOWSRK3TestHooks(
        external_mode_result_override=(
            jnp.asarray(eta_after), jnp.asarray(hu_avg), jnp.asarray(hv_avg)),
        expose_stage1_wzv=True,
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, diagnostic_cfg,
        _nemo_ws_test_hooks=hooks,
    )
    result = model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater)
    candidate = np.asarray(result.T.data)[:, :OWNED_NX, :NLEV]
    oracle = wzv["ww"][..., :NLEV]
    live = wzv["tmask"][..., :NLEV] != 0.0
    support = np.zeros((OWNED_NY, OWNED_NX), dtype=bool)
    support[:-1, :-1] = True
    mask = live & support[..., None]
    if plant:
        candidate = candidate.copy()
        index = tuple(np.argwhere(mask)[0])
        candidate[index] = np.nextafter(candidate[index], np.inf)
    row = score(candidate, oracle, mask)
    if plant:
        require(row["unequal"] == 1, "production-W plant did not fire once")
        raise GateError(
            "planted production stage-1 ww rejected through scorer "
            f"({row['unequal']}/{row['count']})")

    # Binding selector plant: validator must reject the alternative KEG arm.
    bad_cfg = cfg._replace(ke_gradient_scheme="hollingsworth")
    bad_recipe = card.recipe._replace(model_config=bad_cfg, physics_config=bad_cfg.physics)
    selector_plant_rejected = False
    try:
        validate_nemo_testcase_card(card._replace(recipe=bad_recipe))
    except ValueError:
        selector_plant_rejected = True
    require(selector_plant_rejected, "vector/C2 selector plant was accepted")

    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O5-B/compiled-production-stage1-tracer-ww",
        "result": row["status"],
        "row": row,
        "owner": (
            "CONFIRMED_PRODUCTION_WZV_AT_BAR"
            if row["status"] == "AT_BAR"
            else "UNASSIGNED_PRODUCTION_WZV_DEBT"
        ),
        "records": {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in paths.items()},
        "operand_substitution": {
            "label": "ORACLE_SUPPLIED_EXTERNAL_MODE",
            "fields": ["final_ssh", "un_adv", "vn_adv"],
            "certifies_external_mode": False,
            "ignored_upstream_constructibility_proxy": (
                "TEOS-10 bn2 replaces unsupported EOS-80 bn2 only before the "
                "oracle external endpoint; EEN avg4 likewise uses the "
                "constructible min fold and tripolar div-curl viscosity is "
                "zeroed upstream; no proxy value reaches _g0"
            ),
        },
        "comparison_domain": {
            "description": (
                "rank0 wet T cells, 30 active levels, excluding east MPI "
                "support column and north-fold support row"
            ),
            "excluded_support_cells_2d": int(OWNED_NY * OWNED_NX - support.sum()),
        },
        "selectors": {
            "namelist": {"ln_dynadv_vec": True, "nn_dynkeg": 0},
            "card": {
                "momentum_advection": cfg.momentum_advection,
                "ke_gradient_scheme": cfg.ke_gradient_scheme,
            },
            "selector_plant_rejected": selector_plant_rejected,
        },
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "dt_s": card.dt_s,
        },
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

#!/usr/bin/env python3
"""ORCA2 bottom-TKE identity and ordered kt=2 TKE entry walk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
from legoesm.ocean.physics.vertical_mixing.tke import (
    nemo_tke_effective_ice_fraction,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2p_zdf_acquisition_gate as schema,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_phase2t_sh2_gate import (
    HALO, ACTIVE_Z, full3, owned3, read_arrays, sha256,
)

KT1 = "oracle_zdf_sh2_operands_kt00000001.bin"
KT2 = "oracle_zdf_sh2_operands_kt00000002.bin"


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def owned2(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = np.squeeze(arrays[name], axis=2).T
    index = (schema.FIELDS_3D + schema.FIELDS_2D).index(name)
    allocation = (schema.ALLOCATION_3D + schema.ALLOCATION_2D)[index]
    return value[HALO:-HALO, HALO:-HALO] if allocation == "full" else value


def score(candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    candidate = np.asarray(candidate, np.float64)
    target = np.asarray(target, np.float64)
    mask = np.asarray(mask, bool)
    require(candidate.shape == target.shape == mask.shape, "score shape mismatch")
    unequal = mask & (candidate.view(np.uint64) != target.view(np.uint64))
    row = {"status": "AT_BAR" if not unequal.any() else "DEBT",
           "unequal": int(unequal.sum()), "count": int(mask.sum())}
    if unequal.any():
        first = tuple(int(v) for v in np.argwhere(unequal)[0])
        row.update({"first_zero_based_j_i": list(first),
                    "first_candidate": float(candidate[first]),
                    "first_target": float(target[first])})
    return row


def validate(deck: Path, root: Path, mesh: Path, plant: str | None) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy is not active")

    kt1, meta1 = read_arrays(root / KT1, mesh, require_shear=False)
    kt2, meta2 = read_arrays(root / KT2, mesh)
    card = build_orca2_zps_card(deck)
    cfg = card.recipe.model_config.physics.vertical_mixing.tke
    require(cfg.bottom_tke_bc is True,
            "ORCA2 card did not restore executed bottom TKE boundary")
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config, iwm_forcing=card.recipe.iwm_forcing)
    state = model._seed_tke_preclosure_carry(card.recipe.initial_state)

    mbkt = owned2(kt1, "mbkt_real").astype(np.int32)
    wet = mbkt > 0
    require(int(wet.sum()) == 8794, f"wet column count {wet.sum()} != 8794")
    card_bottom = np.asarray(jax.jit(model._tke_bottom_dirichlet)(state))[:, :90]
    # At kt=1 all Kbb face velocities are exactly zero.  zdftke.F90:285-287
    # therefore gives zebot=0 and MAX(zebot,rn_emin)=1e-10 on every wet
    # ssmask cell (rn_emin was forced by zdfiwm_init at :841-844).
    zero_counts = {
        name: int(np.count_nonzero(full3(kt1, name)))
        for name in ("u_Kbb", "v_Kbb")
    }
    require(not any(zero_counts.values()),
            f"kt=1 bottom operand is not the registered rest state: {zero_counts}")
    target_bottom = np.where(wet, np.float64(1.0e-10), np.float64(0.0))
    bottom_row = score(card_bottom, target_bottom, wet)
    require(bottom_row["unequal"] == 0,
            "restored production bottom TKE operand is not source-exact")

    plants = {}
    if plant == "bottom":
        altered = target_bottom.copy()
        index = tuple(int(v) for v in np.argwhere(wet)[0])
        altered[index] = np.nextafter(altered[index], np.inf)
        require(score(card_bottom, altered, wet)["unequal"] == 1,
                "bottom target one-bit plant did not fire")
        plants["bottom_target_one_bit"] = "PASS_NONZERO"
        try:
            require(False is True, "bottom selector false plant")
        except GateError:
            plants["bottom_selector_false"] = "PASS_NONZERO"
        else:
            raise GateError("bottom selector plant did not fire")
        raise GateError("bottom target one-bit plant rejected through production scorer")

    # NEMO nn_eice=1 forms TANH(10*fr_i) (:253-258).  Build the target with
    # independent scalar libm calls and score the production shared helper.
    fr_i = owned2(kt2, "fr_i")
    wet2 = owned2(kt2, "mbkt_real") > 0
    require(int(cfg.eice) == 1, "ORCA2 card did not restore nn_eice=1")
    argument = np.float64(10.0) * fr_i
    nemo_ice = np.fromiter(
        (math.tanh(float(value)) for value in argument.flat),
        dtype=np.float64, count=argument.size).reshape(argument.shape)
    card_ice = np.asarray(jax.jit(
        lambda value: nemo_tke_effective_ice_fraction(value, cfg.eice)
    )(jnp.asarray(fr_i)))
    if plant == "eice_target":
        index = tuple(int(v) for v in np.argwhere(wet2)[0])
        nemo_ice = nemo_ice.copy()
        nemo_ice[index] = np.nextafter(nemo_ice[index], np.inf)
    elif plant == "eice_selector":
        card_ice = np.zeros_like(card_ice)
    ice_row = score(card_ice, nemo_ice, wet2)
    require(ice_row["unequal"] == 0,
            "nn_eice target differs (binding plant fired)" if plant
            else "production nn_eice=1 statement is not exact")
    if plant in ("eice_target", "eice_selector"):
        raise GateError("nn_eice plant failed to create a rejected target")

    return {
        "status": "STOP_TKE_INTERNAL_FRAME_REQUIRED",
        "execution": {"backend": jax.default_backend(), "jit": "production",
                      "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
        "records": {
            KT1: {"sha256": sha256(root / KT1), "header": meta1["header"]},
            KT2: {"sha256": sha256(root / KT2), "header": meta2["header"]},
        },
        "resolved_nemo": {
            "ln_drg_OFF": False, "nn_bc_bot": 1, "rn_emin": 1.0e-10,
            "nn_eice": 1, "nn_etau": 1, "nn_mxl": 3,
            "nn_bc_bot_semantics": (
                "read but unused by zdftke tke_tke; documented only for wave coupling"),
        },
        "card": {"bottom_tke_bc": bool(cfg.bottom_tke_bc),
                 "eice": int(cfg.eice), "etau_mode": cfg.etau_mode,
                 "mxl": int(cfg.tke_mxl_choice),
                 "matrix": cfg.tke_matrix_evaluation,
                 "solver": cfg.tke_solver_evaluation},
        "bottom_tke_bc": {**bottom_row,
                          "source": "zdftke.F90:279-288,841-844",
                          "kt1_Kbb_nonzero": zero_counts,
                          "disposition": "CONFIRMED"},
        "ice_fraction_attenuation": {
            **ice_row,
            "boundary": "nn_eice=1 ice-fraction attenuation operand",
            "statement": "zdftke.F90:255 zice_fra=TANH(fr_i*10._wp)",
            "first_consumption": "zdftke.F90:359 zus3=MAX(0,1-zice_fra)*...",
            "owner": "LANE4_ORCA2_FORCING_SELECTOR",
            "disposition": "CONFIRMED_AT_BAR",
        },
        "first_unmeasured_boundary": {
            "statement": "zdftke.F90:332 zWlc2=zcsd*taum",
            "owner": "GYRE_OWNER_SHARED_TKE",
            "status": "UNMEASURED_NEEDS_WRITE_ONLY_INTERNAL_FRAME",
            "reason": (
                "the admitted kt=2 frame contains the statement inputs but no "
                "zWlc2/Langmuir, matrix, solve, or post-closure oracle outputs"),
            "required_fields": [
                "zWlc2", "zpelc", "imlc", "zhlc", "zus3", "en_post_lc",
                "zdiag_pre_solve", "zd_lw_pre_solve", "zd_up_pre_solve",
                "en_rhs_pre_solve", "en_post_solve", "p_pdlr",
                "mxlm", "mxld", "avm_k_post", "avt_k_post",
            ],
        },
        "downstream": {
            "surface": "CONFIRMED previously 0/8794 (zdftke.F90:264-269)",
            "shear": "CONFIRMED 0/218024 record-complete kt=2 cells",
            "ice_fraction": "CONFIRMED 0/8794 (zdftke.F90:255)",
            "buoyancy_dissipation_matrix_solve": "UNMEASURED_NEEDS_INTERNAL_FRAME",
            "nn_etau_1": "UNMEASURED_AFTER_SOLVE",
            "nn_mxl_3_avm_avt": "UNMEASURED_NO_POST_CLOSURE_TARGET",
        },
        "plants": plants,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument(
        "--plant", nargs="?", const="bottom",
        choices=("bottom", "eice_target", "eice_selector"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, args.mesh, args.plant)
    except (GateError, OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

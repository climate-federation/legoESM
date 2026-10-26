#!/usr/bin/env python3
"""Non-shear ORCA2 ``tke_tke`` entry discriminator for Phase 2s.

This gate deliberately stops before shared TKE arithmetic.  It proves the
surface Dirichlet statement from the admitted kt=1 operand frame, then records
the first executed selector/statement departures that do not require shear.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy, get_policy, set_policy,
)
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG  # noqa: E402
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    _surface_tke_dirichlet,
)
from nemo_testcase_l4_orca2_phase2r_zdf_sh2_gate import (  # noqa: E402
    HALO, owned3, read_frame, sha256,
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def owned2(arrays: dict[str, np.ndarray], allocations: dict[str, str],
           name: str) -> np.ndarray:
    value = arrays[name].T
    return (value[HALO:-HALO, HALO:-HALO]
            if allocations[name] == "full" else value)


def bits_equal(candidate: np.ndarray, target: np.ndarray,
               mask: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate, np.float64)
    target = np.asarray(target, np.float64)
    mask = np.asarray(mask, bool)
    require(candidate.shape == target.shape == mask.shape, "score shape")
    unequal = mask & (candidate.view(np.uint64) != target.view(np.uint64))
    result: dict[str, object] = {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(mask.sum()),
    }
    if unequal.any():
        first = tuple(int(v) for v in np.argwhere(unequal)[0])
        result.update({
            "first_zero_based_j_i": list(first),
            "first_candidate": float(candidate[first]),
            "first_target": float(target[first]),
        })
    return result


def validate(deck: Path, oracle_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy not active")

    record = oracle_root / "oracle_zdf_sh2_operands_kt00000001.bin"
    arrays, allocations = read_frame(record)
    card = build_orca2_zps_card(deck)
    cfg = card.recipe.model_config.physics.vertical_mixing.tke

    taum = owned2(arrays, allocations, "taum")
    fr_i = owned2(arrays, allocations, "fr_i")
    mbkt = owned2(arrays, allocations, "mbkt_real").astype(np.int32)
    wet = mbkt > 0
    require(int(wet.sum()) > 0, "no wet columns")

    # zdftke.F90:238,264-269.  Materialize the two source statements before
    # the MAX, independently of the production helper under test.
    rho0 = float(NEMO_CONSTANTS_CONFIG.rho_0)
    source_surface = np.asarray(jax.jit(lambda stress: jnp.maximum(
        jnp.asarray(1.0e-4, dtype=stress.dtype),
        nemo_source_round(
            nemo_source_round(jnp.asarray(67.83, dtype=stress.dtype)
                              / jnp.asarray(rho0, dtype=stress.dtype))
            * stress),
    ))(jnp.asarray(taum)))
    production_surface = np.asarray(jax.jit(
        lambda stress: _surface_tke_dirichlet(cfg, stress, rho0)
    )(jnp.asarray(taum)))
    if plant:
        index = tuple(int(v) for v in np.argwhere(wet)[0])
        source_surface = source_surface.copy()
        source_surface[index] = np.nextafter(source_surface[index], np.inf)
    surface_row = bits_equal(production_surface, source_surface, wet)
    require(surface_row["unequal"] == 0,
            "surface-Dirichlet target differs (binding target plant fired)" if plant
            else "surface-Dirichlet source statement is not exact")

    # Before tke_tke the canonical cold start has zero Kbb velocity.  NEMO's
    # active bottom branch nevertheless assigns rn_emin to every wet bottom
    # boundary (zdftke.F90:279-288).  The card selector is false, so its
    # corresponding boundary value remains the zero pre-closure value.
    en_pre = owned3(arrays, allocations, "en_pre")
    bottom_before = np.asarray([
        en_pre[j, i, mbkt[j, i]] for j, i in np.argwhere(wet)
    ], dtype=np.float64)
    # zdfiwm_init lowers rn_emin to 1e-10 for this resolved deck before the
    # closure runs (zdftke.F90:841-844; confirmed by the accepted cold-start
    # en record).  Do not substitute the namzdf_tke reference default.
    bottom_target = np.full_like(bottom_before, 1.0e-10)
    bottom_unequal = bottom_before.view(np.uint64) != bottom_target.view(np.uint64)
    require(int(bottom_unequal.sum()) > 0,
            "bottom selector discriminator is vacuous")

    # NEMO nn_eice=1 evaluates TANH(10*fr_i) at :253 and consumes it in
    # zus3 at :363.  The card's eice=0 supplies zero effective ice.  Counts
    # are over owned wet surface cells; no bit-exact transcendental claim is
    # made before Decision 9 extends scalar-libm coverage.
    nemo_effective_ice = np.tanh(10.0 * fr_i)
    card_effective_ice = np.zeros_like(fr_i)
    ice_unequal = wet & (nemo_effective_ice != card_effective_ice)
    require(int(ice_unequal.sum()) > 0, "nn_eice discriminator is vacuous")

    nonzero_shear_operands = {
        name: int(np.count_nonzero(owned3(arrays, allocations, name)))
        for name in ("u_Kbb", "u_Kmm", "v_Kbb", "v_Kmm")
    }
    require(all(value == 0 for value in nonzero_shear_operands.values()),
            "kt=1 shear unexpectedly non-vacuous")

    return {
        "status": "STOP_AT_TKE_BOTTOM_CARD_SELECTOR",
        "execution": {
            "backend": jax.default_backend(), "jit": "production",
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
        },
        "record": {"path": str(record), "sha256": sha256(record)},
        "surface_dirichlet": {
            **surface_row,
            "source": "zdftke.F90:238,264-269",
            "non_floor_wet_columns": int((wet & (source_surface > 1.0e-4)).sum()),
            "disposition": "CONFIRMED",
        },
        "first_departure": {
            "boundary": "tke_tke bottom-boundary card selector",
            "owner": "LANE4_ORCA2_CARD_SELECTOR",
            "source": "zdftke.F90:279-288 (ln_drg_OFF=.false.)",
            "card_bottom_tke_bc": bool(cfg.bottom_tke_bc),
            "nemo_bottom_branch": True,
            "unequal": int(bottom_unequal.sum()),
            "count": int(bottom_unequal.size),
            "first_statement_effect": "zero pre-closure boundary -> rn_emin=1e-10",
            "disposition": "CONFIRMED",
        },
        "next_non_shear_departure": {
            "boundary": "under-ice Langmuir attenuation selector",
            "owner": "LANE4_ORCA2_CARD_SELECTOR_THEN_GYRE_OWNER_SHARED_TKE",
            "source": "zdftke.F90:246,253-258,305-370 (first consumption :363)",
            "card_eice": int(cfg.eice),
            "nemo_nn_eice": 1,
            "unequal": int(ice_unequal.sum()),
            "count": int(wet.sum()),
            "note": "source operand differs; source-literal shared arithmetic is not scored",
            "disposition": "CONFIRMED_SELECTOR_DEBT",
        },
        "other_registered_selector_debts": {
            "mixing_length": {
                "nemo": "zdf_mxl before zdf_tke; tke_avn after solve",
                "source": "zdfphy.F90:280,286; zdftke.F90 tke_avn",
                "status": "UNMEASURED_NO_NMLD_HMLP_OR_POST_TKE_LENGTH_FRAME",
            },
            "nn_etau": {"nemo": 1, "card": cfg.etau_mode,
                         "source": "zdftke.F90:490-496",
                         "status": "UNMEASURED_AFTER_SOLVE"},
            "langmuir_evaluation": {"nemo": "source-order loops",
                                     "card": cfg.tke_langmuir_evaluation,
                                     "status": "UNMEASURED_AFTER_SELECTOR"},
        },
        "kt1_shear_operand_nonzero_counts": nonzero_shear_operands,
        "kt2_discriminator": {
            "fields": ["u_Kbb", "u_Kmm", "v_Kbb", "v_Kmm",
                       "e3uw_Kbb", "e3uw_Kmm", "e3vw_Kbb", "e3vw_Kmm",
                       "avm_k_pre", "sh2"],
            "statements": "zdfsh2.F90:80-100 then zdftke.F90:389-420",
            "status": "PENDING_USER_MPI",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, plant=args.plant)
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

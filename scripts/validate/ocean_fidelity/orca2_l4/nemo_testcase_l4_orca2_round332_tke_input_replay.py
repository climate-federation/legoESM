#!/usr/bin/env python3
"""Offline replay of the two round-332 TKE source-order statements."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    _NEMO_MXL0_LENGTH_SCALE,
    _NEMO_MXL0_VKARMN,
    _mxl0_surface_anchor,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
    nemo_testcase_l4_orca2_round332_source_order_gate as r332,
)

CELLS = (
    (24, 7, 0), (24, 73, 0), (109, 115, 0), (113, 38, 0),
    (116, 112, 0), (116, 112, 1), (121, 112, 0),
)


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def bits(values) -> np.ndarray:
    return np.ascontiguousarray(values).view(np.uint64)


def assemble(records: dict[int, dict], name: str) -> np.ndarray:
    """Assemble rank-local ``(i,j,k)`` slabs as global ``(j,i,k)``."""
    return np.concatenate(
        [records[rank]["fields"][name] for rank in (0, 1)], axis=0
    ).transpose(1, 0, 2)


def measure(record_dir: Path, deck_root: Path) -> dict[str, object]:
    source_records = {
        rank: r332.read_record(record_dir / r332.record_name(rank))
        for rank in (0, 1)
    }
    tke_records = {
        rank: r331.read_record(record_dir / r331.record_name(rank))
        for rank in (0, 1)
    }
    for record in (*source_records.values(), *tke_records.values()):
        r331._verify_checksums(record)

    _, shipped = rung10.card_fields(deck_root)
    card = rung10.comparison_card(shipped)
    fields = rung10.assemble_surface_fields(record_dir, 1)
    _, surface = rung10._surface_forcings(card, deck_root, fields, 1)
    cfg = card.recipe.model_config.physics.vertical_mixing.tke
    mask = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    taum = jnp.asarray(surface.taum)
    fr_i = jnp.asarray(surface.ice_concentration)
    hm_i = jnp.asarray(surface.ice_mean_thickness_hm_i)
    tmask = jnp.asarray(mask, dtype=taum.dtype)

    def anchor(stress, fraction, thickness, wet):
        return _mxl0_surface_anchor(
            cfg, stress, card.recipe.model_config.constants.rho_0,
            card.recipe.model_config.constants.g, wet, fraction, thickness)

    anchor_eager = np.asarray(anchor(taum, fr_i, hm_i, tmask))
    anchor_jit = np.asarray(jax.jit(anchor)(taum, fr_i, hm_i, tmask))

    rho_0 = np.float64(card.recipe.model_config.constants.rho_0)
    gravity = np.float64(card.recipe.model_config.constants.g)
    zraug = (np.float64(_NEMO_MXL0_VKARMN)
             * np.float64(_NEMO_MXL0_LENGTH_SCALE) / (rho_0 * gravity))
    fr_np, hm_np, taum_np = map(np.asarray, (fr_i, hm_i, taum))
    open_water = (np.float64(1.0) - fr_np) * zraug
    open_water = open_water * taum_np
    ice = fr_np * hm_np
    ice = ice * np.float64(2.0)
    anchor_nemo = np.maximum(
        np.float64(cfg.mxl0_min_m), (open_water + ice) * np.asarray(tmask))

    def contracted_old(stress, fraction, thickness, wet):
        return jnp.maximum(
            jnp.asarray(cfg.mxl0_min_m, dtype=stress.dtype),
            (((1.0 - fraction) * zraug * stress
              + fraction * thickness * 2.0) * wet))

    anchor_old_jit = np.asarray(jax.jit(contracted_old)(taum, fr_i, hm_i, tmask))

    eta = jnp.asarray(card.recipe.initial_state.eta.data)
    depth = jnp.asarray(card.recipe.initial_state.H_bathy.data)
    e3t_0 = jnp.asarray(card.recipe.z_coord.nemo_e3t_0)
    r3t = nemo_source_round(eta / depth)
    stretch = nemo_source_round(1.0 + r3t)
    live_e3t = np.asarray(nemo_source_round(e3t_0 * stretch[..., None]))
    zmxlm = assemble(source_records, "tke_zmxlm")
    zmxld = assemble(source_records, "tke_zmxld")

    rows = []
    for j, i, k in CELLS:
        lup = jnp.asarray(anchor_jit[j, i])
        for level in range(k + 1):
            lup = nemo_source_round(lup + live_e3t[j, i, level])
        expected = zmxlm[j, i, k + 1]
        row = {
            "cell_jik": [j, i, k],
            "anchor_hex": np.float64(anchor_jit[j, i]).hex(),
            "live_e3t_hex": [
                np.float64(live_e3t[j, i, level]).hex()
                for level in range(k + 1)
            ],
            "lup_hex": np.float64(lup).hex(),
            "recorded_zmxlm_hex": np.float64(expected).hex(),
            "lup_matches_zmxlm": bool(bits(lup) == bits(expected)),
            "zmxld_not_below_lup": bool(zmxld[j, i, k + 1] >= expected),
            "old_anchor_ulp_delta": (
                int(bits(anchor_old_jit[j, i]).item())
                - int(bits(anchor_nemo[j, i]).item())),
        }
        rows.append(row)

    require(all(row["lup_matches_zmxlm"] for row in rows),
            "lup recurrence does not reproduce all seven recorded zmxlm values")
    require(all(row["zmxld_not_below_lup"] for row in rows),
            "a seven-cell dissipation length is below the binding lup arm")
    selected = tuple((j, i) for j, i, _ in CELLS)
    require(all(bits(anchor_eager[j, i]) == bits(anchor_nemo[j, i])
                for j, i in selected),
            "eager surface anchor differs at a required seven-cell input")
    require(all(bits(anchor_jit[j, i]) == bits(anchor_nemo[j, i])
                for j, i in selected),
            "JIT surface anchor differs at a required seven-cell input")

    en = assemble(tke_records, "after_tke_en")[..., 1:]
    wet = assemble(tke_records, "tke_wmask")[..., 1:] != 0.0
    avt = assemble(tke_records, "after_tke_avt")
    pdlr = assemble(source_records, "tke_pdlr")[..., 1:]
    interior_mxl = zmxlm[..., 1:]
    raw_k = (np.float64(cfg.c_k) * interior_mxl) * np.sqrt(en)
    literal = pdlr * raw_k
    eager_old = raw_k / (np.float64(1.0) / pdlr)
    jit_old = np.asarray(jax.jit(lambda k_m, ratio: k_m / (1.0 / ratio))(
        jnp.asarray(raw_k), jnp.asarray(pdlr)))
    literal_miss = int(np.count_nonzero((bits(literal) != bits(avt)) & wet))
    eager_miss = int(np.count_nonzero((bits(eager_old) != bits(avt)) & wet))
    jit_miss = int(np.count_nonzero((bits(jit_old) != bits(avt)) & wet))
    require(literal_miss == 0, "literal pdlr*K_M replay differs from NEMO")
    require(jit_miss == 0, "JIT reciprocal division no longer contracts to NEMO")

    return {
        "format": "nemo-testcase-l4-orca2-round332-tke-input-replay-v1",
        "status": "PASS_R332_TKE_INPUT_REPLAY",
        "backend": jax.default_backend(),
        "owning_statement": {
            "nemo": "compiled zdftke.f90:620-623; recurrence :687-691",
            "legoesm": "tke.py:_mxl0_surface_anchor; compute_mixing_lengths lup",
            "binding_arm": "lup",
        },
        "seven_cells": rows,
        "anchor": {
            "eager_matches_nemo_required_cells": True,
            "jit_matches_nemo_required_cells": True,
            "old_jit_mismatch_count_required_cells": int(sum(
                bool(bits(anchor_old_jit[j, i]) != bits(anchor_nemo[j, i]))
                for j, i in selected)),
        },
        "prandtl": {
            "wet_cells": int(np.count_nonzero(wet)),
            "literal_multiply_mismatches": literal_miss,
            "old_eager_division_mismatches": eager_miss,
            "old_eager_match_fraction": float(1.0 - eager_miss / np.count_nonzero(wet)),
            "old_jit_division_mismatches": jit_miss,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--deck-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.record_dir, args.deck_root)
    except (ReplayError, r331.GateError, OSError, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R332_TKE_INPUT_REPLAY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

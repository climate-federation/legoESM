#!/usr/bin/env python3
"""Round 35: gate NEMO's stored-reciprocal arithmetic in live ``r3f``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round32_readout_gate as r32,
)

EXPECTED_PARENT_MAX = {
    "u_momentum": 3.181628207426175e-09,
    "v_momentum": 2.9702048395431957e-09,
}


def capture_r3f_boundary(deck_root: Path, record_root: Path) -> dict:
    """Compare the two arithmetic spellings at the live ``r3f`` boundary."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.vertical import nemo_t_fold_f_owned
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round11_dynldf_operator_gate as r11,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    r32._policy()
    _, card = ladder.card_fields(deck_root)
    entry = r11.read_entry_frame(record_root, 2)
    eta = jnp.asarray(entry["ssh"], dtype=jnp.float64)
    raw = card.recipe.z_coord.nemo_een_barotropic
    geom = ensure_geometry(card.recipe.grid)
    b = lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=jnp.float64)
    quarter = jnp.asarray(0.25, dtype=jnp.float64)

    def build():
        area_t = b(jnp.asarray(geom.area_T, dtype=jnp.float64))
        area_eta = b(area_t * eta)
        east = jnp.roll(area_eta, -1, axis=1)
        north = jnp.concatenate(
            [area_eta[1:], jnp.zeros_like(area_eta[:1])], axis=0)
        northeast = jnp.roll(north, -1, axis=1)
        quad = b(b(area_eta + east) + b(north + northeast))
        hf0 = jnp.asarray(raw.hf_0, dtype=jnp.float64)
        wet_f = (hf0 > 0.0).astype(jnp.float64)
        r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
        area_f = b(jnp.asarray(geom.area_q[1:, 1:], dtype=jnp.float64))
        numerator = b(b(quarter * quad) * r1_hf0)
        divided = nemo_t_fold_f_owned(b(numerator / area_f), card.recipe.grid)
        reciprocal = b(one / area_f)
        multiplied = nemo_t_fold_f_owned(
            b(numerator * reciprocal), card.recipe.grid)
        return divided, multiplied, area_t, area_f

    divided, multiplied, area_t, area_f = (
        np.asarray(value, np.float64) for value in jax.jit(build)())
    raw_area_t = np.asarray(raw.e1t, np.float64) * np.asarray(raw.e2t, np.float64)
    raw_area_f = np.asarray(raw.e1f, np.float64) * np.asarray(raw.e2f, np.float64)
    return {
        "area_t_grid_vs_raw_product": r32.score(area_t, raw_area_t),
        "area_f_grid_vs_raw_product": r32.score(area_f, raw_area_f),
        "r3f_division_vs_stored_reciprocal": r32.score(divided, multiplied),
    }


def capture(deck_root: Path, record_root: Path, *, plant: bool = False) -> dict:
    boundary = capture_r3f_boundary(deck_root, record_root)
    parent = r32.capture_ldf_replay(deck_root, record_root)
    arm = r32.capture_ldf_replay(
        deck_root,
        record_root,
        plant=plant,
        r3f_reciprocal_order=True,
    )
    parent_reproduced = all(
        parent[name]["max_abs"] == expected
        for name, expected in EXPECTED_PARENT_MAX.items()
    )
    arm_exact = all(
        arm[name]["bit_identical"]
        for name in ("u_momentum", "v_momentum")
    )
    source_boundary_moves = (
        boundary["area_t_grid_vs_raw_product"]["bit_identical"]
        and boundary["area_f_grid_vs_raw_product"]["bit_identical"]
        and not boundary["r3f_division_vs_stored_reciprocal"]["bit_identical"]
    )
    at_bar = (
        source_boundary_moves
        and parent_reproduced
        and parent["status"] == "HELD"
        and arm_exact
    )
    return {
        "status": "AT_BAR" if at_bar else "DEBT",
        "claim_label": "given NEMO's entry (kt=2 recorded state)",
        "r3f_boundary": boundary,
        "parent": {
            "status": parent["status"],
            "u_momentum": parent["u_momentum"],
            "v_momentum": parent["v_momentum"],
        },
        "stored_reciprocal_arm": {
            "status": arm["status"],
            "u_momentum": arm["u_momentum"],
            "v_momentum": arm["v_momentum"],
        },
        "plant": plant,
        "citations": {
            "stored_reciprocal": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "domhgr.f90:155-157"
            ),
            "r3f": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "domqco.f90:273-286"
            ),
            "consumer": (
                "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
                "dynldf_lev.f90:123"
            ),
        },
        "worktree": arm["worktree"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = capture(args.deck_root, args.record_root, plant=args.plant)
    except (r32.GateError, OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.plant:
        return 1 if result["status"] != "AT_BAR" else 3
    return 0 if result["status"] == "AT_BAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())

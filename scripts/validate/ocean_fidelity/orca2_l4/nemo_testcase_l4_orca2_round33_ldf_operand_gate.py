#!/usr/bin/env python3
"""Round 33: walk the live ORCA2 LDF coefficient into the compiled operator."""

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
    nemo_testcase_l4_orca2_round11_dynldf_operator_gate as r11,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round32_readout_gate as r32,
)


def _mapped_record_ahmf(native: np.ndarray) -> np.ndarray:
    ny, nx, nz = native.shape
    out = np.zeros((ny + 1, nx + 1, nz), dtype=np.float64)
    out[1:] = native[:, (np.arange(nx + 1) - 1) % nx]
    return out


def capture(deck_root: Path, record_root: Path, base_json: Path,
            *, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask,
    )

    r32._policy()
    base = r32._read(base_json)
    r32.require(base["status"] == "HELD", "the registered parent replay is not HELD")
    _, card = ladder.card_fields(deck_root)
    zc = card.recipe.z_coord
    record = r11._stitch(record_root, "output.init_{rank:04d}.nc", ("ahmf",))
    carried = np.asarray(zc.nemo_ldf_ahmf, np.float64)
    oracle = _mapped_record_ahmf(record["ahmf"])
    if plant:
        oracle = np.array(oracle, copy=True)
        index = np.unravel_index(int(np.argmax(np.abs(oracle))), oracle.shape)
        oracle[index] = np.nextafter(oracle[index], np.inf)

    cell_mask = jnp.asarray(card.recipe.initial_state.land_mask.data)
    surface_vtx = compute_vertex_mask(cell_mask, grid=card.recipe.grid)
    active = jnp.asarray(zc.is_active).astype(jnp.float64)
    cell3 = active * cell_mask[..., None]
    binary = jax.vmap(
        lambda m2: compute_vertex_mask(m2, grid=card.recipe.grid),
        in_axes=-1, out_axes=-1,
    )(cell3) * surface_vtx[..., None]
    after_second_mask = carried * np.asarray(binary, np.float64)

    carried_score = r32.score(carried, oracle)
    second_mask_score = r32.score(after_second_mask, carried)
    substitutions = {}
    for name in ("none", "kbb", "kmm", "both"):
        substitutions[name] = r32.capture_ldf_replay(
            deck_root, record_root, plant=plant,
            face_thickness_substitution=name)
    replay = substitutions["none"]
    first_face_owner = next(
        (name for name in ("kbb", "kmm", "both")
         if substitutions[name]["status"] == "PASS"),
        None,
    )
    at_bar = (
        carried_score["bit_identical"]
        and second_mask_score["unequal"] > 0
        and replay["status"] == "PASS"
    )
    return {
        "status": "AT_BAR" if at_bar else "DEBT",
        "claim_label": "given NEMO's entry (kt=2 recorded state)",
        "parent_replay": {
            "status": base["status"],
            "u_momentum": base["u_momentum"],
            "v_momentum": base["v_momentum"],
        },
        "operand_walk": {
            "carried_ahmf_vs_admitted_record": carried_score,
            "production_second_binary_mask_vs_carried_ahmf": second_mask_score,
        },
        "corrected_replay": {
            "status": replay["status"],
            "u_momentum": replay["u_momentum"],
            "v_momentum": replay["v_momentum"],
        },
        "face_thickness_walk": {
            name: {
                "status": row["status"],
                "u_momentum": row["u_momentum"],
                "v_momentum": row["v_momentum"],
            }
            for name, row in substitutions.items()
        },
        "first_face_substitution_at_bar": first_face_owner,
        "plant": plant,
        "citations": {
            "file_read": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353",
            "single_mask": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393",
            "consumer": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123",
        },
        "worktree": replay["worktree"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--base-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = capture(
            args.deck_root, args.record_root, args.base_json, plant=args.plant)
    except (r32.GateError, OSError, ValueError, RuntimeError) as exc:
        print(f"GATE REFUSED: {exc}", file=sys.stderr)
        return 1
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))
    if args.plant:
        return 1 if result["status"] != "AT_BAR" else 3
    return 0 if result["status"] == "AT_BAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())

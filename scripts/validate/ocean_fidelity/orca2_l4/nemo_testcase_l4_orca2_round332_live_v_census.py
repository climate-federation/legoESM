#!/usr/bin/env python3
"""Compare admitted live post-fold V with legoESM's production stage-1 V."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
    nemo_testcase_l4_orca2_round332_source_order_gate as raw_gate,
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def signs(array: np.ndarray, select: np.ndarray) -> dict[str, int]:
    values = np.asarray(array)[select]
    zero = values == 0.0
    return {
        "positive_zero": int(np.count_nonzero(zero & ~np.signbit(values))),
        "negative_zero": int(np.count_nonzero(zero & np.signbit(values))),
        "nonzero": int(np.count_nonzero(~zero)),
    }


def joint(nemo: np.ndarray, lego: np.ndarray, select: np.ndarray) -> dict[str, int]:
    n, l = np.asarray(nemo)[select], np.asarray(lego)[select]
    require(bool(np.all(n == l)), "selected live values differ numerically")
    ns, ls = np.signbit(n), np.signbit(l)
    return {
        "nemo_positive_lego_positive": int(np.count_nonzero(~ns & ~ls)),
        "nemo_positive_lego_negative": int(np.count_nonzero(~ns & ls)),
        "nemo_negative_lego_positive": int(np.count_nonzero(ns & ~ls)),
        "nemo_negative_lego_negative": int(np.count_nonzero(ns & ls)),
        "bit_identical": int(np.count_nonzero(n.view(np.uint64) == l.view(np.uint64))),
        "value_identical": int(n.size),
    }


def assemble_raw(root: Path) -> np.ndarray:
    slabs = []
    for rank in (0, 1):
        record = raw_gate.read_raw_record(root / raw_gate.raw_record_name(rank))
        raw_gate.r331._verify_checksums(record)
        local = record["fields"][raw_gate.RAW_FIELDS[0]]
        require(local.shape == (94, 152, 31), f"rank {rank}: wrong raw shape")
        slabs.append(local[2:92, 2:150, :30].transpose(1, 0, 2))
    return np.concatenate(slabs, axis=1)


def measure(deck_root: Path, record_root: Path, expect_commit: str) -> dict:
    import jax
    import jax.numpy as jnp

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"] == expect_commit,
            "live-V census requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu" and bool(jax.config.jax_enable_x64)
            and not jax.config.jax_disable_jit,
            "live-V census requires CPU fp64 production JIT")

    _, base = rung10.card_fields(deck_root)
    card = r329.selected_card(base, arm="recordice", rung="rung10")
    entry = rung10.assemble_state_fields(record_root, 1, stage=None)
    state = card.recipe.initial_state._replace(
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    surface_fields = rung10.assemble_surface_fields(record_root, 1)
    freshwater, surface = rung10._surface_forcings(card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True),
    )
    trace = model.step(
        state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface,
        **rung10.damping_step_kwargs(card, deck_root, 1))
    lego = rung10._stage_candidate_fields(trace.stage_outputs[0])["v"]
    nemo = assemble_raw(record_root)
    require(nemo.shape == lego.shape == (148, 180, 30), "global V shape moved")

    mask = np.broadcast_to(
        np.asarray(state.v_mask.data)[1:, :, None] != 0.0, lego.shape)
    wet = mask
    dry = ~mask
    perm = np.asarray(card.recipe.grid.fold.perm_v)
    dump = rung10.read_state_frame(
        record_root / "oracle_stage_kt00000001_s1.bin", kt=1, stage=1)["v"]
    registered_2d = ((~mask[146, perm[:90], :])
                     & (dump[147, :90, :] == 0.0)
                     & ~np.signbit(dump[147, :90, :]))
    registered = np.zeros_like(dry)
    registered[147, :90, :] = registered_2d
    require(int(np.count_nonzero(registered)) == 2032,
            "registered signed-zero family moved")
    other_dry = dry & ~registered

    result = {
        "format": "nemo-testcase-l4-orca2-round332-live-v-census-v1",
        "status": "PASS_R332_LIVE_V_CENSUS",
        "worktree": stamp,
        "record_root": str(record_root),
        "execution": "production-jit-cpu-fp64-x64-libm",
        "populations": {},
    }
    for name, select in (("registered_2032", registered),
                         ("other_dry_halo", other_dry),
                         ("all_dry_halo", dry), ("wet_control", wet)):
        result["populations"][name] = {
            "count": int(np.count_nonzero(select)),
            "nemo": signs(nemo, select),
            "legoesm": signs(lego, select),
            "joint": joint(nemo, lego, select),
        }

    require(result["populations"]["wet_control"]["joint"]["bit_identical"]
            == result["populations"]["wet_control"]["count"],
            "wet control is not bit-identical")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.record_root, args.expect_commit)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

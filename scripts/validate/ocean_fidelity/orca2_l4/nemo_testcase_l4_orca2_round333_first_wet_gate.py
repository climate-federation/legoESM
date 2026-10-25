#!/usr/bin/env python3
"""Round 333: locate rung 10's first real wet-cell departure.

This is a thin extension of round 332's reviewed production-JIT scorer.  It
runs only through kt=2, retains numerical residuals out of band from the JSON
rows, and adds native-mask and admitted-ice metadata to the first wet-value
and first over-bar rows.  It changes neither the card nor score ordering.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round326_hmi_gate as hmi,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round332_rescore_gate as r332,
)


class GateError(RuntimeError):
    """The wet-value measurement is incomplete or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


_DETAILS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_NEXT_DETAIL = 0


def located_score(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Round-332 score with residual arrays retained outside serialized rows."""

    global _NEXT_DETAIL
    row = r332.score_with_bar(actual, expected)
    valued = np.asarray(actual) != np.asarray(expected)
    detail = _NEXT_DETAIL
    _NEXT_DETAIL += 1
    _DETAILS[detail] = (np.asarray(actual), np.asarray(expected), valued)
    return {**row, "_round333_detail": detail, "value_unequal": int(np.count_nonzero(valued))}


def _native_masks(card) -> dict[str, np.ndarray]:
    state = card.recipe.initial_state
    tmask = np.asarray(state.land_mask.data) > 0.5
    if tmask.ndim == 2:
        tmask = np.broadcast_to(tmask[..., None], state.T.data.shape)
    umask = np.asarray(state.u_mask.data) > 0.5
    vmask = np.asarray(state.v_mask.data) > 0.5
    if umask.ndim == 2:
        umask = np.broadcast_to(umask[..., None], state.u.data.shape)
    if vmask.ndim == 2:
        vmask = np.broadcast_to(vmask[..., None], state.v.data.shape)
    return {
        "T": tmask[..., :30],
        "S": tmask[..., :30],
        "u": umask[:, 1:, :30],
        "v": vmask[1:, :, :30],
        "ssh": tmask[..., 0],
    }


def checkpoint_mask(mask: np.ndarray, checkpoint: str) -> np.ndarray:
    """Match full-domain entry rows or rank-0 stage rows without ambiguity."""

    require(
        checkpoint == "entry" or checkpoint.startswith("stage"),
        f"unknown checkpoint {checkpoint}",
    )
    return mask if checkpoint == "entry" else mask[:, :90]


def _full_native_mask(card, field: str) -> np.ndarray:
    state = card.recipe.initial_state
    if field in ("T", "S", "ssh"):
        mask = np.asarray(state.land_mask.data) > 0.5
        if mask.ndim == 3:
            return mask[..., 0] if field == "ssh" else mask[..., :30]
        return mask
    if field == "u":
        mask = np.asarray(state.u_mask.data) > 0.5
        return mask[:, 1:, :30] if mask.ndim == 3 else mask[:, 1:]
    mask = np.asarray(state.v_mask.data) > 0.5
    return mask[1:, :, :30] if mask.ndim == 3 else mask[1:, :]


def classify_location(
    *, field: str, index: tuple[int, ...], card, ice_fraction: np.ndarray
) -> dict[str, object]:
    """Return the preregistered seam/fold/coast/ice/open flags."""

    j, i = index[:2]
    k = index[2] if len(index) == 3 else 0
    full = _full_native_mask(card, field)
    require(
        bool(full[j, i, k] if full.ndim == 3 else full[j, i]),
        "maximum is outside the native wet mask",
    )
    ny, nx = full.shape[:2]
    neighbours = ((j, (i - 1) % nx), (j, (i + 1) % nx))
    neighbour_cells = list(neighbours)
    if j > 0:
        neighbour_cells.append((j - 1, i))
    if j + 1 < ny:
        neighbour_cells.append((j + 1, i))
    coast = any(
        not bool(full[jj, ii, k] if full.ndim == 3 else full[jj, ii]) for jj, ii in neighbour_cells
    )
    seam = i in (0, nx - 1)
    if field == "v":
        fold = j >= ny - 2
    else:
        fold = j == ny - 1
    under_ice = bool(ice_fraction[j, i] > 0.0)
    return {
        "seam": seam,
        "fold": fold,
        "coast": coast,
        "under_ice": under_ice,
        "open_ocean": not any((seam, fold, coast, under_ice)),
        "ice_fraction": float(ice_fraction[j, i]),
    }


def _locate(
    row: dict[str, object], masks: dict[str, np.ndarray], card, ice_fraction: np.ndarray
) -> dict[str, object]:
    detail = int(row.pop("_round333_detail"))
    actual, expected, valued = _DETAILS.pop(detail)
    wet = checkpoint_mask(masks[str(row["field"])], str(row["checkpoint"]))
    require(
        wet.shape == actual.shape == expected.shape,
        f"native support {wet.shape} != row {actual.shape}",
    )
    valued_wet = valued & wet
    n_unequal = int(np.count_nonzero(valued_wet))
    residual = actual - expected
    wet_values = residual[wet]
    wet_rms = float(np.sqrt(np.mean(wet_values * wet_values)))
    require(n_unequal > 0, "selected row has no unequal wet values")
    absolute = np.abs(residual)
    maximum = float(np.max(absolute[valued_wet]))
    max_mask = valued_wet & (absolute == maximum)
    maxima = np.argwhere(max_mask)
    require(maxima.size > 0, "maximum wet-value index is absent")
    index = tuple(int(value) for value in maxima[0])
    global_index = {
        "i": index[1],
        "j": index[0],
        "k": index[2] if len(index) == 3 else 0,
        "rank": index[1] // 90,
    }
    return {
        **{key: value for key, value in row.items() if not key.startswith("_round333")},
        "wet_max_abs": maximum,
        "wet_rms": wet_rms,
        "unequal_wet": n_unequal,
        "max_location": global_index,
        "max_ties": int(maxima.shape[0]),
        "location_flags": classify_location(
            field=str(row["field"]), index=index, card=card, ice_fraction=ice_fraction
        ),
    }


def measure(
    label: str, deck_root: Path, twin_a: Path, hmi_root: Path, expect_commit: str
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(
        stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
        "round-333 measurement requires its clean committed tree",
    )
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
        and bool(jax.config.jax_enable_x64),
        "fp64/libm is not active",
    )
    require(
        jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
        "round-333 measurement requires production JIT on CPU",
    )
    require(label in ("given_nemo_entry", "independent"), "unknown label")

    _, shipped = rung10.card_fields(deck_root)
    card = r329.selected_card(shipped, arm="recordice", rung="rung10")
    masks = _native_masks(card)
    ice_records = [hmi.read_record(hmi_root / hmi.record_name(2, rank)) for rank in range(2)]
    ice_fraction = np.concatenate(
        [hmi.owned(record["fields"]["fr_i"]) for record in ice_records], axis=1
    )
    require(
        ice_fraction.shape == masks["ssh"].shape,
        "SI3 ice fraction does not cover the full T domain",
    )

    original_score, rung10.score = rung10.score, located_score
    try:
        trajectory = rung10.candidate_trajectory(
            deck_root, twin_a, card, max_step=2, bridge_ssh=(label == "given_nemo_entry")
        )
    finally:
        rung10.score = original_score

    rows: list[dict[str, object]] = []
    for checkpoint in trajectory["checkpoints"]:
        for field in rung10.FIELD_ORDER:
            row = checkpoint["rows"][field]
            rows.append(
                {
                    "kt": checkpoint["kt"],
                    "checkpoint": checkpoint["checkpoint"],
                    "field": field,
                    **row,
                }
            )

    first_wet = None
    first_over = None
    for source in rows:
        detail = int(source["_round333_detail"])
        _, _, valued = _DETAILS[detail]
        wet = checkpoint_mask(masks[str(source["field"])], str(source["checkpoint"]))
        require(
            valued.shape == wet.shape,
            f"{source['checkpoint']} {source['field']} shape mismatch: "
            f"{valued.shape} != {wet.shape}",
        )
        if first_wet is None and bool(np.any(valued & wet)):
            first_wet = dict(source)
        if first_over is None and source["classification"] == "OVER_BAR":
            first_over = dict(source)
    require(
        first_wet is not None and first_over is not None,
        "kt=1..2 does not contain both registered boundaries",
    )
    located_wet = _locate(first_wet, masks, card, ice_fraction)
    if (first_over["kt"], first_over["checkpoint"], first_over["field"]) == (
        located_wet["kt"],
        located_wet["checkpoint"],
        located_wet["field"],
    ):
        located_over = dict(located_wet)
        _DETAILS.pop(int(first_over["_round333_detail"]), None)
    else:
        located_over = _locate(first_over, masks, card, ice_fraction)
    _DETAILS.clear()
    return {
        "format": "nemo-testcase-l4-orca2-round333-first-wet-v1",
        "status": "PASS_R333_FIRST_WET_ROW",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "row_order": "kt, entry/stage1/stage2/stage3, T/S/u/v/ssh",
        "first_wet_value_row": located_wet,
        "first_over_bar_row": located_over,
        "boundaries_coincide": (located_wet["kt"], located_wet["checkpoint"], located_wet["field"])
        == (located_over["kt"], located_over["checkpoint"], located_over["field"]),
    }


def self_test(*, plant: bool = False) -> dict[str, object]:
    actual = np.array([[[-0.0], [2.0 + 8.0e-15]]], dtype=np.float64)
    expected = np.array([[[0.0], [2.0]]], dtype=np.float64)
    row = located_score(actual, expected)
    require(
        row["unequal"] == 2 and row["value_unequal"] == 1, "signed-zero/value split did not bind"
    )
    if plant:
        row["value_unequal"] = 0
    require(row["value_unequal"] == 1, "value-difference plant stayed green")
    _DETAILS.clear()
    return {"status": "PASS_R333_SCORE_CONTROL"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--label", choices=("given_nemo_entry", "independent"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--hmi-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(
                all((args.label, args.deck_root, args.twin_a, args.hmi_root, args.expect_commit)),
                "measurement arguments incomplete",
            )
            result = measure(
                args.label, args.deck_root, args.twin_a, args.hmi_root, args.expect_commit
            )
        elif args.self_test:
            result = self_test(plant=args.plant)
        else:
            raise GateError("no operation selected")
        require(not args.plant, "value-difference plant stayed green")
    except (GateError, r332.GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

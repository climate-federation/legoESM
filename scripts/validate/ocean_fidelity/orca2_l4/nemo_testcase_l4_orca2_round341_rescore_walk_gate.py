#!/usr/bin/env python3
"""Round 341: full current-tip 8-pair re-score, 3-D wet-mask rung-10 audit, and
offline source-order walk of rung 10's first raw and first 3-D-wet non-bit rows.

This gate implements the three deliverables of ORCA2 round 341:
1. Re-score rung 10's comparison card on both labels (`given_nemo_entry` and
   `independent`, `kt = 1..10`, every stage and restart/entry, full depth) and
   the other six Decision-122 pairs (`omt0/independent`, `rung0/independent`,
   `omt1/independent`, `omt2/independent`, `omt3/independent`,
   `omt4/independent`) at the current tip, publishing the full 1,560-row table
   and the full 200-row rung-10 3-D-wet audit table, and naming the first
   non-bit row per label under both raw binary64 and 3-D wet-cell scoring.
2. Walk the first raw non-bit row (`kt = 1, stage1, v`) and the first 3-D-wet
   non-bit / first wet-value row (`kt = 10, stage2, T`) in NEMO source order
   with offline replay from the admitted production-arithmetic records (`round 5`
   and `round 326`), naming the single owning statement (`record-build ppsrc`
   citation + `legoESM` line), `CONFIRMED` vs `PLAUSIBLE`.
3. Verify the non-landed single-statement fix proposal on the recorded inputs
   across all `kt = 1..10` without modifying model code in `packages/`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_active_repo() -> Path:
    if "--repo" in sys.argv:
        idx = sys.argv.index("--repo")
        if idx + 1 < len(sys.argv):
            return Path(sys.argv[idx + 1]).resolve()
    return DEFAULT_REPO_ROOT


ACTIVE_REPO_ROOT = _resolve_active_repo()
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _script_dir]
if ACTIVE_REPO_ROOT != DEFAULT_REPO_ROOT:
    _default_prefix = str(DEFAULT_REPO_ROOT)
    sys.path = [p for p in sys.path if not p.startswith(_default_prefix)]
    os.chdir(ACTIVE_REPO_ROOT)

for _package in (
    ACTIVE_REPO_ROOT,
    ACTIVE_REPO_ROOT / "packages/core",
    ACTIVE_REPO_ROOT / "packages/ocean",
    ACTIVE_REPO_ROOT / "packages/atmosphere",
    ACTIVE_REPO_ROOT / "packages/coupler",
    ACTIVE_REPO_ROOT / "packages/ice",
    ACTIVE_REPO_ROOT / "packages/land",
    ACTIVE_REPO_ROOT / "packages/ml",
    ACTIVE_REPO_ROOT / "packages/tools",
    ACTIVE_REPO_ROOT / "src",
):
    if _package.exists():
        _pkg_str = str(_package)
        if _pkg_str in sys.path:
            sys.path.remove(_pkg_str)
        sys.path.insert(0, _pkg_str)

for _mod_name in [
    m for m in list(sys.modules) if m == "legoesm" or m.startswith("legoesm.")
]:
    mod_file = getattr(sys.modules[_mod_name], "__file__", None)
    if mod_file is None or not Path(mod_file).resolve().is_relative_to(ACTIVE_REPO_ROOT):
        del sys.modules[_mod_name]

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
    nemo_testcase_l4_orca2_round301_u_transport_factors as r301,
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
    nemo_testcase_l4_orca2_round332_rescore_gate as r332,
    nemo_testcase_l4_orca2_round332_source_order_gate as r332_raw,
    nemo_testcase_l4_orca2_round333_first_wet_gate as r333_wet,
)

PAIRS = r332.PAIRS
CLASSES = r332.CLASSES
BAR = r332.BAR

PARENT_COMMIT = "a91d673b6fcff38b097f7c0fc8ddb009c05d7a32"
PREFIX_RESCORE_COMMIT = "a104ed2a4e79a6f3a5704d221b9718113e846bc7"
PLANTED_WRONG_PARENT_COMMIT = "0" * 40

GRIDLADDER_LANE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/lanes/gridladder"
)
GRIDLADDER_ROUNDS = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds"
)
GRIDLADDER_RECIPE_COMMIT = "242eaacc47"
GRIDLADDER_VMIX_LANE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/lanes/gridladder_vmix"
)
GRIDLADDER_VMIX_ROUNDS = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds"
)
GRIDLADDER_VMIX_RECIPE_COMMIT = "0b9443cef2"

CALLER_ORDER = (
    "non_nemo_folded",
    "non_nemo_unfolded",
    "legacy_caller_10145",
    "faithful_dino",
    "ORCA2V1-tke",
    "GYRE-zco",
    "VORTEX-zco",
    "LOCK_EXCHANGE-zco",
    "OVERFLOW-zps",
    "ORCA2-rung0",
    "ORCA2G1-flat",
    "ORCA2G1b-vec",
    "ORCA2G2-zps",
    "ORCA2G3-ts",
    "ORCA2G4-sbc",
    "ORCA2G5-ice",
    "ORCA2G6-vec",
    "ORCA2G7-drg",
    "ORCA2G8-ldf",
    "ORCA2-zps_kt1_rest",
    "ORCA2-zps",
)

EXPECTED_PAIR_COUNTS = {
    ("omt0", "independent"): {
        "BIT_EXACT": 16,
        "AT_BAR_NOT_EXACT": 23,
        "OVER_BAR": 161,
    },
    ("rung0", "independent"): {
        "BIT_EXACT": 11,
        "AT_BAR_NOT_EXACT": 10,
        "OVER_BAR": 179,
    },
    ("omt1", "independent"): {
        "BIT_EXACT": 48,
        "AT_BAR_NOT_EXACT": 33,
        "OVER_BAR": 79,
    },
    ("omt2", "independent"): {
        "BIT_EXACT": 48,
        "AT_BAR_NOT_EXACT": 33,
        "OVER_BAR": 119,
    },
    ("omt3", "independent"): {
        "BIT_EXACT": 36,
        "AT_BAR_NOT_EXACT": 25,
        "OVER_BAR": 139,
    },
    ("omt4", "independent"): {
        "BIT_EXACT": 12,
        "AT_BAR_NOT_EXACT": 15,
        "OVER_BAR": 173,
    },
    ("rung10", "given_nemo_entry"): {
        "BIT_EXACT": 152,
        "AT_BAR_NOT_EXACT": 48,
        "OVER_BAR": 0,
    },
    ("rung10", "independent"): {
        "BIT_EXACT": 152,
        "AT_BAR_NOT_EXACT": 48,
        "OVER_BAR": 0,
    },
}

EXPECTED_TOTAL_COUNTS = {
    "BIT_EXACT": 475,
    "AT_BAR_NOT_EXACT": 235,
    "OVER_BAR": 850,
}

EXPECTED_PREFIX_PAIR_COUNTS = {
    **EXPECTED_PAIR_COUNTS,
    ("rung10", "given_nemo_entry"): {
        "BIT_EXACT": 151,
        "AT_BAR_NOT_EXACT": 49,
        "OVER_BAR": 0,
    },
    ("rung10", "independent"): {
        "BIT_EXACT": 151,
        "AT_BAR_NOT_EXACT": 49,
        "OVER_BAR": 0,
    },
}

EXPECTED_PREFIX_TOTAL_COUNTS = {
    "BIT_EXACT": 473,
    "AT_BAR_NOT_EXACT": 237,
    "OVER_BAR": 850,
}

PLANT_CHOICES = (
    "none",
    "signed-zero",
    "wet-3d-mask",
    "stage2-thickness",
    "walk-closure",
    "exact-loss",
    "away-row",
    "parent-commit",
    "caller-unequal",
)


class GateError(RuntimeError):
    """The round-341 rescore, 3-D wet audit, or source-order walk failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _hex64(val: float | np.floating) -> str:
    arr = np.asarray(val, dtype=np.float64)
    return f"0x{int(arr.view(np.uint64)):016x}"


def _from_hex64(hex_str: str) -> np.float64:
    return np.asarray(np.uint64(int(hex_str, 16))).view(np.float64)


def _native_3d_masks(card) -> dict[str, np.ndarray]:
    """Return the true 3-D NEMO partial-step masks (`tmask`, `umask`, `vmask`)
    and 2-D surface masks (`ssmask`, `ssumask`, `ssvmask`) on the physical grid.
    """
    z = card.recipe.z_coord
    eb = z.nemo_een_barotropic
    require(eb is not None, "card is missing nemo_een_barotropic")
    tmask = np.asarray(z.is_active[:, :, :30], dtype=np.float64) != 0.0
    umask = np.asarray(eb.umask[:, :, :30], dtype=np.float64) != 0.0
    vmask = np.asarray(eb.vmask[:, :, :30], dtype=np.float64) != 0.0
    ssmask = tmask[:, :, 0]
    ssumask = umask[:, :, 0]
    ssvmask = vmask[:, :, 0]
    return {
        "T": tmask,
        "S": tmask,
        "u": umask,
        "v": vmask,
        "ssh": ssmask,
        "ss_T": ssmask,
        "ss_S": ssmask,
        "ss_u": ssumask,
        "ss_v": ssvmask,
        "ss_ssh": ssmask,
    }


def _slice_mask(mask: np.ndarray, width: int) -> np.ndarray:
    return mask if mask.shape[1] == width else mask[:, :width]


def _compute_row_wet_detail(
    field: str,
    actual: np.ndarray,
    expected: np.ndarray,
    masks_2d: dict[str, np.ndarray],
    masks_3d: dict[str, np.ndarray],
) -> dict[str, object]:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    width = act.shape[1]

    wet_2d = _slice_mask(masks_2d[field], width)
    wet_3d = _slice_mask(masks_3d[field], width)
    ss_2d = _slice_mask(masks_3d[f"ss_{field}"], width)
    if wet_3d.ndim == 3 and ss_2d.ndim == 2:
        ss_3d = np.broadcast_to(ss_2d[..., None], wet_3d.shape)
    else:
        ss_3d = ss_2d

    require(
        wet_2d.shape == act.shape == exp.shape,
        f"2-D wet mask shape {wet_2d.shape} != field {field} shape {act.shape}",
    )
    require(
        wet_3d.shape == act.shape == exp.shape,
        f"3-D wet mask shape {wet_3d.shape} != field {field} shape {act.shape}",
    )

    bit_diff = act.view(np.uint64) != exp.view(np.uint64)
    val_diff = act != exp
    residual = act - exp

    # 2-D surface-mask broadcast audit (matching r333_wet._native_masks)
    wet_2d_bit_diff = bit_diff & wet_2d
    wet_2d_val_diff = val_diff & wet_2d
    dry_2d_bit_diff = bit_diff & (~wet_2d)
    wet_2d_bit_unequal = int(np.count_nonzero(wet_2d_bit_diff))
    unequal_2d_wet = int(np.count_nonzero(wet_2d_val_diff))
    signed_zero_only_2d_wet = int(np.count_nonzero(wet_2d_bit_diff & (~val_diff)))
    unequal_2d_dry = int(np.count_nonzero(dry_2d_bit_diff))
    wet_2d_count = int(np.count_nonzero(wet_2d))
    wet_2d_vals = residual[wet_2d]
    wet_2d_rms = (
        float(np.sqrt(np.mean(wet_2d_vals * wet_2d_vals))) if wet_2d_count else 0.0
    )
    wet_2d_max_abs = (
        float(np.max(np.abs(wet_2d_vals))) if unequal_2d_wet > 0 else 0.0
    )
    oracle_2d_max = float(np.max(np.abs(exp[wet_2d]))) if wet_2d_count else 0.0
    wet_2d_scale = max(oracle_2d_max, 1.0)
    wet_2d_norm = wet_2d_max_abs / wet_2d_scale
    if wet_2d_bit_unequal == 0:
        wet_2d_class = "BIT_EXACT"
    elif wet_2d_norm <= BAR:
        wet_2d_class = "AT_BAR_NOT_EXACT"
    else:
        wet_2d_class = "OVER_BAR"

    # True 3-D partial-step mask audit (tmask, umask, vmask, ssmask)
    wet_3d_bit_diff = bit_diff & wet_3d
    wet_3d_val_diff = val_diff & wet_3d
    dry_3d_bit_diff = bit_diff & (~wet_3d)
    subbottom_dry_diff = bit_diff & (~wet_3d) & ss_3d
    surface_dry_diff = bit_diff & (~wet_3d) & (~ss_3d)

    wet_3d_bit_unequal = int(np.count_nonzero(wet_3d_bit_diff))
    unequal_3d_wet = int(np.count_nonzero(wet_3d_val_diff))
    signed_zero_only_3d_wet = int(np.count_nonzero(wet_3d_bit_diff & (~val_diff)))
    unequal_3d_dry = int(np.count_nonzero(dry_3d_bit_diff))
    subbottom_dry_bit_unequal = int(np.count_nonzero(subbottom_dry_diff))
    surface_dry_bit_unequal = int(np.count_nonzero(surface_dry_diff))
    wet_3d_count = int(np.count_nonzero(wet_3d))

    wet_3d_vals = residual[wet_3d]
    wet_3d_rms = (
        float(np.sqrt(np.mean(wet_3d_vals * wet_3d_vals))) if wet_3d_count else 0.0
    )
    wet_3d_max_abs = (
        float(np.max(np.abs(wet_3d_vals))) if unequal_3d_wet > 0 else 0.0
    )
    oracle_3d_max = float(np.max(np.abs(exp[wet_3d]))) if wet_3d_count else 0.0
    wet_3d_scale = max(oracle_3d_max, 1.0)
    wet_3d_norm = wet_3d_max_abs / wet_3d_scale
    if wet_3d_bit_unequal == 0:
        wet_3d_class = "BIT_EXACT"
    elif wet_3d_norm <= BAR:
        wet_3d_class = "AT_BAR_NOT_EXACT"
    else:
        wet_3d_class = "OVER_BAR"

    max_index = None
    max_ties = 0
    if unequal_3d_wet > 0:
        abs_res = np.abs(residual)
        max_mask = wet_3d_val_diff & (abs_res == wet_3d_max_abs)
        maxima = np.argwhere(max_mask)
        require(maxima.size > 0, "maximum 3-D wet-value index is absent")
        max_index = tuple(int(v) for v in maxima[0])
        max_ties = int(maxima.shape[0])

    return {
        "wet_count": wet_2d_count,
        "wet_bit_identical": wet_2d_bit_unequal == 0,
        "wet_bit_unequal": wet_2d_bit_unequal,
        "unequal_wet": unequal_2d_wet,
        "signed_zero_only_wet": signed_zero_only_2d_wet,
        "unequal_dry": unequal_2d_dry,
        "wet_max_abs": wet_2d_max_abs,
        "wet_rms": wet_2d_rms,
        "oracle_wet_max_abs": oracle_2d_max,
        "wet_normalization_scale": wet_2d_scale,
        "wet_normalized_max_abs": wet_2d_norm,
        "wet_classification": wet_2d_class,
        "wet_3d_count": wet_3d_count,
        "wet_3d_bit_identical": wet_3d_bit_unequal == 0,
        "wet_3d_bit_unequal": wet_3d_bit_unequal,
        "unequal_3d_wet": unequal_3d_wet,
        "signed_zero_only_3d_wet": signed_zero_only_3d_wet,
        "unequal_3d_dry": unequal_3d_dry,
        "subbottom_dry_bit_unequal": subbottom_dry_bit_unequal,
        "surface_dry_bit_unequal": surface_dry_bit_unequal,
        "wet_3d_max_abs": wet_3d_max_abs,
        "wet_3d_rms": wet_3d_rms,
        "oracle_wet_3d_max_abs": oracle_3d_max,
        "wet_3d_normalization_scale": wet_3d_scale,
        "wet_3d_normalized_max_abs": wet_3d_norm,
        "wet_3d_classification": wet_3d_class,
        "_max_index": max_index,
        "max_ties": max_ties,
    }


def _finalize_wet_audit(
    rows: list[dict[str, object]],
    card,
    ice_by_kt: dict[int, np.ndarray],
) -> dict[str, object]:
    wet_rows: list[dict[str, object]] = []
    for row in rows:
        detail = row.pop("_round341_wet")
        kt = int(row["kt"])
        checkpoint = str(row["checkpoint"])
        field = str(row["field"])
        max_index = detail.pop("_max_index")
        loc_info: dict[str, object] = {}
        if max_index is not None:
            loc_info["max_location"] = {
                "i": max_index[1],
                "j": max_index[0],
                "k": max_index[2] if len(max_index) == 3 else 0,
                "rank": max_index[1] // 90,
            }
            loc_info["location_flags"] = r333_wet.classify_location(
                field=field,
                index=max_index,
                card=card,
                ice_fraction=ice_by_kt[kt],
            )
        wet_rows.append(
            {
                "kt": kt,
                "checkpoint": checkpoint,
                "field": field,
                "count": int(row["count"]),
                "unequal": int(row["unequal"]),
                "max_abs": float(row["max_abs"]),
                "rms": float(row["rms"]),
                "oracle_max_abs": float(row["oracle_max_abs"]),
                "normalization_scale": float(row["normalization_scale"]),
                "normalized_max_abs": float(row["normalized_max_abs"]),
                "classification": str(row["classification"]),
                **detail,
                **loc_info,
            }
        )

    wet_2d_counts = Counter(r["wet_classification"] for r in wet_rows)
    wet_3d_counts = Counter(r["wet_3d_classification"] for r in wet_rows)
    first_wet_2d_non_bit = next(
        (r for r in wet_rows if not r["wet_bit_identical"]), None
    )
    first_wet_3d_non_bit = next(
        (r for r in wet_rows if not r["wet_3d_bit_identical"]), None
    )
    first_wet_value = next(
        (r for r in wet_rows if int(r["unequal_3d_wet"]) > 0), None
    )
    first_wet_3d_over_bar = next(
        (r for r in wet_rows if r["wet_3d_classification"] == "OVER_BAR"), None
    )
    first_raw_over_bar = next(
        (r for r in wet_rows if r["classification"] == "OVER_BAR"), None
    )
    return {
        "row_count": len(wet_rows),
        "wet_2d_classification_counts": {
            name: int(wet_2d_counts[name]) for name in CLASSES
        },
        "wet_3d_classification_counts": {
            name: int(wet_3d_counts[name]) for name in CLASSES
        },
        "first_wet_2d_non_bit_row": first_wet_2d_non_bit,
        "first_wet_3d_non_bit_row": first_wet_3d_non_bit,
        "first_wet_value_row": first_wet_value,
        "first_wet_3d_over_bar_row": first_wet_3d_over_bar,
        "first_over_bar_row": first_raw_over_bar,
        "rows": wet_rows,
    }


def measure(
    rung: str,
    label: str,
    deck_root: Path,
    twin_a: Path,
    expect_commit: str,
    *,
    arm: str = "candidate",
) -> dict[str, object]:
    require((rung, label) in PAIRS, f"unsupported pair {(rung, label)}")
    require(arm in ("control", "candidate"), f"unknown arm {arm}")
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    model_path = Path(model_module.__file__).resolve()
    r332_path = Path(r332.__file__).resolve()
    require(
        model_path.is_relative_to(ACTIVE_REPO_ROOT)
        and r332_path.is_relative_to(ACTIVE_REPO_ROOT),
        f"imported modules {model_path} / {r332_path} are outside {ACTIVE_REPO_ROOT}",
    )

    masks_2d = None
    masks_3d = None
    card_ref = None
    ice_by_kt = None
    original_compare = rung10.compare_fields

    if rung == "rung10":
        _, shipped = rung10.card_fields(deck_root)
        card_ref = r329.selected_card(shipped, arm="recordice", rung="rung10")
        masks_2d = r333_wet._native_masks(card_ref)
        masks_3d = _native_3d_masks(card_ref)
        hmi_records = rung10._admitted_hm_i_records()
        ice_by_kt = {
            kt: np.concatenate(
                [hmi_records[(0, kt)]["fr_i"], hmi_records[(1, kt)]["fr_i"]],
                axis=1,
            )
            for kt in range(1, 11)
        }

        def wrapped_compare(
            actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
        ) -> dict[str, object]:
            out = original_compare(actual, expected)
            for name in rung10.FIELD_ORDER:
                out["rows"][name]["_round341_wet"] = _compute_row_wet_detail(
                    name, actual[name], expected[name], masks_2d, masks_3d
                )
            return out

        rung10.compare_fields = wrapped_compare

    try:
        result = r332.measure(rung, label, deck_root, twin_a, expect_commit)
    finally:
        rung10.compare_fields = original_compare

    require(
        bool(result.get("worktree", {}).get("clean"))
        and str(result["worktree"]["commit"]).lower() == expect_commit.lower(),
        f"measured worktree {result.get('worktree')} != clean {expect_commit}",
    )

    expected_counts = (
        EXPECTED_PREFIX_PAIR_COUNTS[(rung, label)]
        if arm == "control"
        else EXPECTED_PAIR_COUNTS[(rung, label)]
    )
    require(
        result["classification_counts"] == expected_counts,
        f"{rung}/{label} classification_counts {result['classification_counts']} != {expected_counts}",
    )

    if rung == "rung10":
        wet_audit = _finalize_wet_audit(
            result["ladder"]["rows"], card_ref, ice_by_kt
        )
        if arm == "control":
            require(
                wet_audit["wet_3d_classification_counts"]
                == {"BIT_EXACT": 199, "AT_BAR_NOT_EXACT": 1, "OVER_BAR": 0},
                f"rung10/{label} control 3-D wet counts {wet_audit['wet_3d_classification_counts']} != 199/1/0",
            )
            first_3d = wet_audit["first_wet_3d_non_bit_row"]
            require(
                first_3d is not None
                and (first_3d["kt"], first_3d["checkpoint"], first_3d["field"])
                == (10, "stage2", "T")
                and first_3d["unequal_3d_wet"] == 1
                and first_3d["unequal_3d_dry"] == 0,
                f"rung10/{label} unexpected control first 3-D wet non-bit row: {first_3d}",
            )
        else:
            require(
                wet_audit["wet_3d_classification_counts"]
                == {"BIT_EXACT": 200, "AT_BAR_NOT_EXACT": 0, "OVER_BAR": 0},
                f"rung10/{label} 3-D wet counts {wet_audit['wet_3d_classification_counts']} != 200/0/0",
            )
            require(
                wet_audit["wet_2d_classification_counts"]
                == {"BIT_EXACT": 152, "AT_BAR_NOT_EXACT": 48, "OVER_BAR": 0},
                f"rung10/{label} 2-D wet counts {wet_audit['wet_2d_classification_counts']} != 152/48/0",
            )
            first_3d = wet_audit["first_wet_3d_non_bit_row"]
            require(
                first_3d is None
                and wet_audit["first_wet_value_row"] is None
                and wet_audit["first_wet_3d_over_bar_row"] is None,
                f"rung10/{label} unexpected 3-D wet non-bit row: {first_3d}",
            )
        result["wet_audit"] = wet_audit
        result["first_wet_3d_non_bit_row"] = first_3d
        result["first_wet_value_row"] = wet_audit["first_wet_value_row"]

    result.update(
        {
            "arm": arm,
            "format": "nemo-testcase-l4-orca2-round341-rescore-v1",
            "status": "PASS_R341_PAIR_RESCORE",
        }
    )
    return result


def aggregate(
    paths: list[Path],
    table_out: Path | None,
    rung10_table_out: Path | None,
    expect_commit: str | None = None,
) -> dict[str, object]:
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    keys = [(doc["rung"], doc["label"]) for doc in docs]
    require(len(keys) == len(set(keys)), "duplicate pair in aggregate")
    require(set(keys) == set(PAIRS), f"pair set incomplete: {sorted(keys)}")
    commits = {doc["worktree"]["commit"] for doc in docs}
    require(len(commits) == 1, "measurements mix commits")
    require(
        all(bool(doc.get("worktree", {}).get("clean")) for doc in docs),
        "measurement worktree is not clean",
    )
    commit = next(iter(commits))
    if expect_commit is not None:
        require(
            commit.lower() == expect_commit.lower(),
            f"aggregate commit {commit} != expected {expect_commit}",
        )

    docs_by_pair = {(doc["rung"], doc["label"]): doc for doc in docs}
    rows: list[dict[str, object]] = []
    rung10_wet_rows: list[dict[str, object]] = []
    first_raw_by_pair: dict[str, object] = {}
    first_wet_3d_by_rung10_label: dict[str, object] = {}
    counts_by_pair: dict[str, dict[str, int]] = {}

    for rung, label in PAIRS:
        doc = docs_by_pair[(rung, label)]
        pair_name = f"{rung}/{label}"
        require(
            doc["classification_counts"] == EXPECTED_PAIR_COUNTS[(rung, label)],
            f"{pair_name} counts {doc['classification_counts']} != {EXPECTED_PAIR_COUNTS[(rung, label)]}",
        )
        counts_by_pair[pair_name] = doc["classification_counts"]
        first_raw_by_pair[pair_name] = doc["first_non_bit_row"]
        for row in doc["ladder"]["rows"]:
            rows.append({"rung": rung, "label": label, **row})
        if rung == "rung10":
            wet_audit = doc.get("wet_audit")
            require(wet_audit is not None, f"{pair_name} is missing wet_audit")
            first_wet_3d_by_rung10_label[pair_name] = wet_audit[
                "first_wet_3d_non_bit_row"
            ]
            for wrow in wet_audit["rows"]:
                rung10_wet_rows.append({"rung": rung, "label": label, **wrow})

    require(len(rows) == 1560, f"aggregate has {len(rows)} rows, expected 1560")
    total_counts = Counter(str(row["classification"]) for row in rows)
    total_counts_dict = {name: int(total_counts[name]) for name in CLASSES}
    require(
        total_counts_dict == EXPECTED_TOTAL_COUNTS,
        f"total classification counts {total_counts_dict} != {EXPECTED_TOTAL_COUNTS}",
    )

    # Verify rung10/given_nemo_entry and rung10/independent are row-for-row identical
    r10_given = docs_by_pair[("rung10", "given_nemo_entry")]["wet_audit"]["rows"]
    r10_indep = docs_by_pair[("rung10", "independent")]["wet_audit"]["rows"]
    require(len(r10_given) == len(r10_indep) == 200, "rung10 wet rows != 200")
    for a, b in zip(r10_given, r10_indep, strict=True):
        require(
            a == b,
            f"rung10 labels differ at {(a['kt'], a['checkpoint'], a['field'])}: {a} vs {b}",
        )

    if table_out is not None:
        columns = (
            "rung",
            "label",
            "kt",
            "checkpoint",
            "field",
            "max_abs",
            "rms",
            "unequal",
            "count",
            "oracle_max_abs",
            "normalization_scale",
            "normalized_max_abs",
            "classification",
        )
        table_out.parent.mkdir(parents=True, exist_ok=True)
        with table_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)

    if rung10_table_out is not None:
        r10_columns = (
            "rung",
            "label",
            "kt",
            "checkpoint",
            "field",
            "max_abs",
            "rms",
            "unequal",
            "count",
            "oracle_max_abs",
            "normalization_scale",
            "normalized_max_abs",
            "classification",
            "wet_3d_count",
            "wet_3d_bit_unequal",
            "unequal_3d_wet",
            "signed_zero_only_3d_wet",
            "unequal_3d_dry",
            "subbottom_dry_bit_unequal",
            "surface_dry_bit_unequal",
            "wet_3d_max_abs",
            "wet_3d_rms",
            "wet_3d_classification",
            "wet_count",
            "wet_bit_unequal",
            "unequal_wet",
            "signed_zero_only_wet",
            "unequal_dry",
            "wet_classification",
        )
        rung10_table_out.parent.mkdir(parents=True, exist_ok=True)
        with rung10_table_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=r10_columns,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rung10_wet_rows)

    return {
        "format": "nemo-testcase-l4-orca2-round341-aggregate-v1",
        "status": "PASS_R341_FULL_RESCORE",
        "worktree_commit": commit,
        "row_count": len(rows),
        "classification_counts": total_counts_dict,
        "classification_counts_by_pair": counts_by_pair,
        "first_non_bit_by_pair": first_raw_by_pair,
        "rung10_labels_row_for_row_identical": True,
        "rung10_wet_3d_classification_counts": docs_by_pair[
            ("rung10", "given_nemo_entry")
        ]["wet_audit"]["wet_3d_classification_counts"],
        "rung10_first_wet_3d_non_bit_by_label": first_wet_3d_by_rung10_label,
        "table": str(table_out) if table_out is not None else None,
        "rung10_table": (
            str(rung10_table_out) if rung10_table_out is not None else None
        ),
    }


def walk_source_order(
    record_root: Path,
    r326_root: Path,
    r332_v_root: Path | None = None,
    *,
    plant: str = "none",
) -> dict[str, object]:
    """Offline source-order walk of rung 10's first raw non-bit row
    (`kt = 1, stage1, v`) and first 3-D-wet non-bit / first wet-value row
    (`kt = 10, stage2, T`) from the admitted production-arithmetic records.
    """
    import netCDF4

    with netCDF4.Dataset(record_root / "mesh_mask_0000.nc") as ds:
        e1t = np.asarray(ds["e1t"][0], np.float64)
        e2t = np.asarray(ds["e2t"][0], np.float64)
        e1u = np.asarray(ds["e1u"][0], np.float64)
        e2u = np.asarray(ds["e2u"][0], np.float64)
        e1v = np.asarray(ds["e1v"][0], np.float64)
        e2v = np.asarray(ds["e2v"][0], np.float64)
        e3t_0 = np.moveaxis(np.asarray(ds["e3t_0"][0], np.float64), 0, -1)[..., :30]
        e3u_0 = np.moveaxis(np.asarray(ds["e3u_0"][0], np.float64), 0, -1)[..., :30]
        e3v_0 = np.moveaxis(np.asarray(ds["e3v_0"][0], np.float64), 0, -1)[..., :30]
        tmask = np.moveaxis(np.asarray(ds["tmask"][0], np.float64), 0, -1)[..., :30]
        umask = np.moveaxis(np.asarray(ds["umask"][0], np.float64), 0, -1)[..., :30]
        vmask = np.moveaxis(np.asarray(ds["vmask"][0], np.float64), 0, -1)[..., :30]

    ht_0 = np.zeros_like(e1t)
    hu_0 = np.zeros_like(e1u)
    hv_0 = np.zeros_like(e1v)
    for k in range(30):
        ht_0 = ht_0 + e3t_0[..., k] * tmask[..., k]
        hu_0 = hu_0 + e3u_0[..., k] * umask[..., k]
        hv_0 = hv_0 + e3v_0[..., k] * vmask[..., k]
    ssmask = np.max(tmask, axis=-1)
    ssumask = np.max(umask, axis=-1)
    ssvmask = np.max(vmask, axis=-1)
    r1_ht_0 = ssmask / (ht_0 + np.float64(1.0) - ssmask)
    r1_hu_0 = ssumask / (hu_0 + np.float64(1.0) - ssumask)
    r1_hv_0 = ssvmask / (hv_0 + np.float64(1.0) - ssvmask)
    e1e2t = e1t * e2t
    r1_e1e2t = np.float64(1.0) / e1e2t
    r1_e1e2u = np.float64(1.0) / (e1u * e2u)
    r1_e1e2v = np.float64(1.0) / (e1v * e2v)

    # 1. Verify cross-record bit identity between round 5 and round 326 on stage1 and stage2
    cross_record_unequal = 0
    for kt in range(1, 11):
        for stage in (1, 2):
            s_r5 = rung10.read_state_frame(
                record_root / f"oracle_stage_kt{kt:08d}_s{stage}.bin",
                kt=kt,
                stage=stage,
            )
            s_r326 = rung10.read_state_frame(
                r326_root / f"oracle_stage_kt{kt:08d}_s{stage}.bin",
                kt=kt,
                stage=stage,
            )
            for fname in ("T", "S", "u", "v", "ssh"):
                a = s_r5[fname][:, :90]
                b = s_r326[fname][:, :90]
                cross_record_unequal += int(
                    np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))
                )
    require(
        cross_record_unequal == 0,
        f"round 5 and round 326 stage1/stage2 records differ on {cross_record_unequal} words",
    )

    # 2. Walk Part A: First raw non-bit row (`kt = 1, stage1, v`)
    s1_kt1 = rung10.read_state_frame(
        record_root / "oracle_stage_kt00000001_s1.bin", kt=1, stage=1
    )
    v_oracle_kt1_s1 = np.asarray(s1_kt1["v"][:, :90, :], dtype=np.float64)
    vmask_3d_bool = vmask != 0.0
    ssvmask_3d_bool = np.broadcast_to(ssvmask[:, :, None] != 0.0, vmask.shape)
    oracle_dry_nonzero_or_negzero = int(
        np.count_nonzero(v_oracle_kt1_s1[~vmask_3d_bool].view(np.uint64) != 0)
    )
    require(
        oracle_dry_nonzero_or_negzero == 0,
        f"oracle_stage_kt00000001_s1.bin has {oracle_dry_nonzero_or_negzero} non-+0.0 dry words",
    )

    # Optional PLAUSIBLE check against round-332 raw post-fold V record at kt=1, stage=1
    r332_raw_v_summary = None
    raw_v_file = (
        r332_v_root / r332_raw.raw_record_name(0) if r332_v_root is not None else None
    )
    if raw_v_file is not None and raw_v_file.exists():
        raw_rec = r332_raw.read_raw_record(raw_v_file)
        r332_raw.r331._verify_checksums(raw_rec)
        v_after_fold_full = (
            raw_rec["fields"][r332_raw.RAW_FIELDS[0]][2:92, 2:150, :30]
            .transpose(1, 0, 2)
        )
        raw_vs_canon_diff = (
            v_after_fold_full.view(np.uint64) != v_oracle_kt1_s1.view(np.uint64)
        )
        raw_vs_canon_wet = int(np.count_nonzero(raw_vs_canon_diff & vmask_3d_bool))
        raw_vs_canon_dry = int(np.count_nonzero(raw_vs_canon_diff & (~vmask_3d_bool)))
        raw_vs_canon_subbottom = int(
            np.count_nonzero(raw_vs_canon_diff & (~vmask_3d_bool) & ssvmask_3d_bool)
        )
        raw_vs_canon_surface = int(
            np.count_nonzero(raw_vs_canon_diff & (~vmask_3d_bool) & (~ssvmask_3d_bool))
        )
        require(
            raw_vs_canon_wet == 0
            and raw_vs_canon_dry == 10700
            and raw_vs_canon_subbottom == 5884
            and raw_vs_canon_surface == 4816,
            f"round-332 raw V vs canon mismatch: wet={raw_vs_canon_wet}, dry={raw_vs_canon_dry}",
        )
        r332_raw_v_summary = {
            "provenance_status": "PLAUSIBLE (round-332 producer flagged by round-336 audit)",
            "wet_3d_unequal_vs_canon": raw_vs_canon_wet,
            "dry_3d_negzero_count": raw_vs_canon_dry,
            "subbottom_dry_negzero_count": raw_vs_canon_subbottom,
            "surface_dry_negzero_count": raw_vs_canon_surface,
        }

    # 3. Walk Part B: First 3-D-wet non-bit / first wet-value row (`kt = 10, stage2, T`)
    # Replay stp_RK3_stg(2) across all kt = 1..10 on admitted round-5 and round-326 records
    r2_3 = np.float64(2.0) / np.float64(3.0)
    r1_3 = np.float64(1.0) / np.float64(3.0)
    r1_2 = np.float64(0.5)
    rDt = np.float64(5400.0)  # NEMO r1_2 * rn_Dt at Stage 2 (`stprk3_stg.f90:188`)
    r1_Dt = np.float64(1.0) / rDt
    r1_rho0 = np.float64(1.0) / np.float64(1026.0)
    wet_int = tmask[1:-1, 1:-1, :] > 0.5

    per_kt_stage2_summary: list[dict[str, object]] = []
    kt10_cell_trace: dict[str, object] | None = None
    total_int_wet_T_curr_unequal = 0
    total_int_wet_T_prop_unequal = 0
    total_int_wet_S_curr_unequal = 0
    total_int_wet_S_prop_unequal = 0

    for kt in range(1, 11):
        ek = rung10.read_state_frame(
            record_root / f"oracle_step_entry_kt{kt:08d}.bin", kt=kt, stage=None
        )
        s1 = rung10.read_state_frame(
            record_root / f"oracle_stage_kt{kt:08d}_s1.bin", kt=kt, stage=1
        )
        s2 = rung10.read_state_frame(
            record_root / f"oracle_stage_kt{kt:08d}_s2.bin", kt=kt, stage=2
        )
        s3 = rung10.read_state_frame(
            record_root / f"oracle_stage_kt{kt:08d}_s3.bin", kt=kt, stage=3
        )
        bt = r301.read_bt(record_root / f"oracle_bt_frames_kt{kt:08d}.bin", kt)
        surf = rung10.read_surface_fields(
            record_root / f"oracle_ocean_surface_input_kt{kt:08d}.bin", kt, 0
        )

        ssh_b = ek["ssh"]
        ssha = s3["ssh"]
        ssh_13 = r2_3 * ssh_b + r1_3 * ssha
        r3t_b = ssh_b * r1_ht_0
        r3ta = ssha * r1_ht_0
        r3t_s1 = r2_3 * r3t_b + r1_3 * r3ta
        r3t_rebuilt = ssh_13 * r1_ht_0
        if plant == "stage2-thickness":
            r3t_s1 = r3t_rebuilt
        r3t_s2 = r1_2 * (r3t_b + r3ta)

        e1e2t_ssh_b = e1e2t * ssh_b
        e1e2t_ssha = e1e2t * ssha
        r3u_b = np.zeros_like(e1u)
        r3ua = np.zeros_like(e1u)
        r3u_b[:, :-1] = (
            np.float64(0.5)
            * (e1e2t_ssh_b[:, :-1] + e1e2t_ssh_b[:, 1:])
            * r1_hu_0[:, :-1]
            * r1_e1e2u[:, :-1]
        )
        r3ua[:, :-1] = (
            np.float64(0.5)
            * (e1e2t_ssha[:, :-1] + e1e2t_ssha[:, 1:])
            * r1_hu_0[:, :-1]
            * r1_e1e2u[:, :-1]
        )
        r3u_s1 = r2_3 * r3u_b + r1_3 * r3ua

        r3v_b = np.zeros_like(e1v)
        r3va = np.zeros_like(e1v)
        r3v_b[:-1, :] = (
            np.float64(0.5)
            * (e1e2t_ssh_b[:-1, :] + e1e2t_ssh_b[1:, :])
            * r1_hv_0[:-1, :]
            * r1_e1e2v[:-1, :]
        )
        r3va[:-1, :] = (
            np.float64(0.5)
            * (e1e2t_ssha[:-1, :] + e1e2t_ssha[1:, :])
            * r1_hv_0[:-1, :]
            * r1_e1e2v[:-1, :]
        )
        r3v_s1 = r2_3 * r3v_b + r1_3 * r3va

        e3u_kmm = e3u_0 * (np.float64(1.0) + r3u_s1[..., None] * umask)
        e3v_kmm = e3v_0 * (np.float64(1.0) + r3v_s1[..., None] * vmask)
        inv_hu = r1_hu_0 / (np.float64(1.0) + r3u_s1)
        inv_hv = r1_hv_0 / (np.float64(1.0) + r3v_s1)
        zub = bt["un_adv"] * inv_hu - bt["uu_b"]
        zvb = bt["vn_adv"] * inv_hv - bt["vv_b"]
        u_corr = s1["u"] + zub[..., None] * umask
        v_corr = s1["v"] + zvb[..., None] * vmask
        zFu = (e2u[..., None] * e3u_kmm) * u_corr
        zFv = (e1v[..., None] * e3v_kmm) * v_corr

        e3t_kmm_stored = e3t_0 * (np.float64(1.0) + r3t_s1[..., None] * tmask)
        e3t_kmm_rebuilt = e3t_0 * (np.float64(1.0) + r3t_rebuilt[..., None] * tmask)

        # Verify every column where e3t_kmm_stored != e3t_kmm_rebuilt lies strictly inside rank-0 interior (1<=j<=146, 1<=i<=88)
        cell_diff_mask = (
            e3t_kmm_stored.view(np.uint64) != e3t_kmm_rebuilt.view(np.uint64)
        ) & (tmask > 0.5)
        col_diff_mask = np.any(cell_diff_mask, axis=-1)
        col_diff_indices = np.argwhere(col_diff_mask)
        if col_diff_indices.size > 0:
            require(
                bool(
                    np.all(
                        (col_diff_indices[:, 0] >= 1)
                        & (col_diff_indices[:, 0] <= 146)
                        & (col_diff_indices[:, 1] >= 1)
                        & (col_diff_indices[:, 1] <= 88)
                    )
                ),
                f"kt={kt}: e3t_kmm_stored != e3t_kmm_rebuilt on boundary column",
            )

        # Horizontal divergence `div_hor_RK3_t` (`divhor.f90:134-154`) + `sbc_rnf_div` (`sbcrnf.f90:280-283`) + `wzv_RK3_t` (`sshwzv.f90:295-299`)
        hdiv = np.zeros_like(e3t_0)
        for k in range(30):
            du = zFu[1:-1, 1:-1, k] - zFu[1:-1, :-2, k]
            dv = zFv[1:-1, 1:-1, k] - zFv[:-2, 1:-1, k]
            hdiv[1:-1, 1:-1, k] = (
                (du + dv) * r1_e1e2t[1:-1, 1:-1]
            ) / e3t_kmm_stored[1:-1, 1:-1, k]
        hdiv[1:-1, 1:-1, 0] = (
            hdiv[1:-1, 1:-1, 0]
            - (surf["rnf"][1:-1, 1:-1] * r1_rho0) / e3t_kmm_stored[1:-1, 1:-1, 0]
        )
        ww = np.zeros((148, 90, 31), dtype=np.float64)
        for k in range(29, -1, -1):
            term_div = e3t_kmm_stored[1:-1, 1:-1, k] * hdiv[1:-1, 1:-1, k]
            term_str = (r1_Dt * e3t_0[1:-1, 1:-1, k]) * (
                r3t_s2[1:-1, 1:-1] - r3t_b[1:-1, 1:-1]
            )
            ww[1:-1, 1:-1, k] = (
                ww[1:-1, 1:-1, k + 1]
                - (term_div + term_str) * tmask[1:-1, 1:-1, k]
            )
        zFw = e1e2t[..., None] * ww
        wmask = np.zeros_like(ww)
        wmask[..., 1:30] = tmask[..., 1:30] * tmask[..., :29]

        def _adv_update(
            tr_b: np.ndarray,
            tr_kmm: np.ndarray,
            rnf_tr: np.ndarray,
            e3t_div: np.ndarray,
        ) -> np.ndarray:
            ztFu = np.zeros_like(tr_kmm)
            ztFv = np.zeros_like(tr_kmm)
            ztFu[:, :-1, :] = (np.float64(0.5) * zFu[:, :-1, :]) * (
                tr_kmm[:, :-1, :] + tr_kmm[:, 1:, :]
            )
            ztFv[:-1, :, :] = (np.float64(0.5) * zFv[:-1, :, :]) * (
                tr_kmm[:-1, :, :] + tr_kmm[1:, :, :]
            )
            Krhs = np.zeros_like(tr_kmm)
            for kk in range(30):
                du = ztFu[1:-1, 1:-1, kk] - ztFu[1:-1, :-2, kk]
                dv = ztFv[1:-1, 1:-1, kk] - ztFv[:-2, 1:-1, kk]
                Krhs[1:-1, 1:-1, kk] = (
                    Krhs[1:-1, 1:-1, kk]
                    - ((du + dv) * r1_e1e2t[1:-1, 1:-1]) / e3t_div[1:-1, 1:-1, kk]
                )
            ztFw_prev = np.zeros((148, 90), dtype=np.float64)
            for kk in range(29):
                ztFw_kp1 = (
                    (np.float64(0.5) * zFw[1:-1, 1:-1, kk + 1])
                    * (tr_kmm[1:-1, 1:-1, kk + 1] + tr_kmm[1:-1, 1:-1, kk])
                ) * wmask[1:-1, 1:-1, kk + 1]
                Krhs[1:-1, 1:-1, kk] = (
                    Krhs[1:-1, 1:-1, kk]
                    - ((ztFw_prev[1:-1, 1:-1] - ztFw_kp1) * r1_e1e2t[1:-1, 1:-1])
                    / e3t_div[1:-1, 1:-1, kk]
                )
                ztFw_prev[1:-1, 1:-1] = ztFw_kp1
            Krhs[1:-1, 1:-1, 29] = (
                Krhs[1:-1, 1:-1, 29]
                - (ztFw_prev[1:-1, 1:-1] * r1_e1e2t[1:-1, 1:-1])
                / e3t_div[1:-1, 1:-1, 29]
            )
            z1_rho0_e3t = r1_rho0 / e3t_kmm_stored[1:-1, 1:-1, 0]
            sbc_term = (
                surf["emp"][1:-1, 1:-1] * tr_b[1:-1, 1:-1, 0]
            ) * z1_rho0_e3t
            Krhs[1:-1, 1:-1, 0] = Krhs[1:-1, 1:-1, 0] - sbc_term
            rnf = surf["rnf"][1:-1, 1:-1]
            zdep = np.where(
                rnf != 0.0, np.float64(1.0) / e3t_kmm_stored[1:-1, 1:-1, 0], 0.0
            )
            Krhs[1:-1, 1:-1, 0] = np.where(
                rnf != 0.0,
                Krhs[1:-1, 1:-1, 0] + rnf_tr * zdep,
                Krhs[1:-1, 1:-1, 0],
            )
            q_before = np.float64(1.0) + r3t_b[1:-1, 1:-1, None]
            q_now = np.float64(1.0) + r3t_s1[1:-1, 1:-1, None]
            q_after = np.float64(1.0) + r3t_s2[1:-1, 1:-1, None]
            return (
                (
                    q_before * tr_b[1:-1, 1:-1, :]
                    + (rDt * q_now) * Krhs[1:-1, 1:-1, :]
                )
                / q_after
            ) * tmask[1:-1, 1:-1, :]

        T_s2_prop = _adv_update(
            ek["T"], s1["T"], surf["rnf_tsc"][1:-1, 1:-1, 0], e3t_kmm_stored
        )
        T_s2_curr = _adv_update(
            ek["T"], s1["T"], surf["rnf_tsc"][1:-1, 1:-1, 0], e3t_kmm_rebuilt
        )
        S_s2_prop = _adv_update(
            ek["S"], s1["S"], surf["rnf_tsc"][1:-1, 1:-1, 1], e3t_kmm_stored
        )
        S_s2_curr = _adv_update(
            ek["S"], s1["S"], surf["rnf_tsc"][1:-1, 1:-1, 1], e3t_kmm_rebuilt
        )

        if plant == "walk-closure" and kt == 10:
            T_s2_prop = T_s2_curr.copy()

        T_s2_oracle_int = s2["T"][1:-1, 1:-1, :]
        S_s2_oracle_int = s2["S"][1:-1, 1:-1, :]

        T_curr_unequal = int(
            np.count_nonzero(
                (T_s2_curr.view(np.uint64) != T_s2_oracle_int.view(np.uint64))
                & wet_int
            )
        )
        T_prop_unequal = int(
            np.count_nonzero(
                (T_s2_prop.view(np.uint64) != T_s2_oracle_int.view(np.uint64))
                & wet_int
            )
        )
        S_curr_unequal = int(
            np.count_nonzero(
                (S_s2_curr.view(np.uint64) != S_s2_oracle_int.view(np.uint64))
                & wet_int
            )
        )
        S_prop_unequal = int(
            np.count_nonzero(
                (S_s2_prop.view(np.uint64) != S_s2_oracle_int.view(np.uint64))
                & wet_int
            )
        )

        total_int_wet_T_curr_unequal += T_curr_unequal
        total_int_wet_T_prop_unequal += T_prop_unequal
        total_int_wet_S_curr_unequal += S_curr_unequal
        total_int_wet_S_prop_unequal += S_prop_unequal

        per_kt_stage2_summary.append(
            {
                "kt": kt,
                "interior_wet_3d_count": int(np.count_nonzero(wet_int)),
                "h_kmm_vs_rebuilt_wet_columns": int(np.count_nonzero(col_diff_mask)),
                "h_kmm_vs_rebuilt_wet_3d_cells": int(np.count_nonzero(cell_diff_mask)),
                "T_current_interior_wet_unequal": T_curr_unequal,
                "T_proposed_interior_wet_unequal": T_prop_unequal,
                "S_current_interior_wet_unequal": S_curr_unequal,
                "S_proposed_interior_wet_unequal": S_prop_unequal,
            }
        )

        if kt == 10:
            j, i, k = 18, 34, 3
            T_kmm = s1["T"]
            ztFu_e = (np.float64(0.5) * zFu[j, i, k]) * (
                T_kmm[j, i, k] + T_kmm[j, i + 1, k]
            )
            ztFu_w = (np.float64(0.5) * zFu[j, i - 1, k]) * (
                T_kmm[j, i - 1, k] + T_kmm[j, i, k]
            )
            ztFv_n = (np.float64(0.5) * zFv[j, i, k]) * (
                T_kmm[j, i, k] + T_kmm[j + 1, i, k]
            )
            ztFv_s = (np.float64(0.5) * zFv[j - 1, i, k]) * (
                T_kmm[j - 1, i, k] + T_kmm[j, i, k]
            )
            h_sum = (ztFu_e - ztFu_w) + (ztFv_n - ztFv_s)
            h_scaled = h_sum * r1_e1e2t[j, i]
            e3t_st = e3t_kmm_stored[j, i, k]
            e3t_rb = e3t_kmm_rebuilt[j, i, k]
            krhs_h_st = -(h_scaled / e3t_st)
            krhs_h_rb = -(h_scaled / e3t_rb)

            ztFw_top = (np.float64(0.5) * zFw[j, i, k]) * (
                T_kmm[j, i, k] + T_kmm[j, i, k - 1]
            )
            ztFw_bot = (np.float64(0.5) * zFw[j, i, k + 1]) * (
                T_kmm[j, i, k + 1] + T_kmm[j, i, k]
            )
            v_diff = ztFw_top - ztFw_bot
            v_scaled = v_diff * r1_e1e2t[j, i]
            v_div_st = v_scaled / e3t_st
            v_div_rb = v_scaled / e3t_rb
            krhs_st = krhs_h_st - v_div_st
            krhs_rb = krhs_h_rb - v_div_rb

            qb = np.float64(1.0) + r3t_b[j, i]
            qn = np.float64(1.0) + r3t_s1[j, i]
            qa = np.float64(1.0) + r3t_s2[j, i]
            tb = ek["T"][j, i, k]
            term1 = qb * tb
            dt_qn = rDt * qn
            term2_st = dt_qn * krhs_st
            term2_rb = dt_qn * krhs_rb
            num_st = term1 + term2_st
            num_rb = term1 + term2_rb
            out_st = num_st / qa
            out_rb = num_rb / qa
            if plant == "walk-closure":
                out_st = out_rb

            kt10_cell_trace = {
                "location": {"j": j, "i": i, "k": k, "rank": 0},
                "ssh_b": {
                    "float": float(ssh_b[j, i]),
                    "hex": _hex64(ssh_b[j, i]),
                },
                "ssha": {
                    "float": float(ssha[j, i]),
                    "hex": _hex64(ssha[j, i]),
                },
                "ssh_13": {
                    "float": float(ssh_13[j, i]),
                    "hex": _hex64(ssh_13[j, i]),
                },
                "r1_ht_0": {
                    "float": float(r1_ht_0[j, i]),
                    "hex": _hex64(r1_ht_0[j, i]),
                },
                "r3t_b": {
                    "float": float(r3t_b[j, i]),
                    "hex": _hex64(r3t_b[j, i]),
                },
                "r3ta": {
                    "float": float(r3ta[j, i]),
                    "hex": _hex64(r3ta[j, i]),
                },
                "r3t_s1_stored": {
                    "float": float(r3t_s1[j, i]),
                    "hex": _hex64(r3t_s1[j, i]),
                },
                "r3t_s1_rebuilt": {
                    "float": float(r3t_rebuilt[j, i]),
                    "hex": _hex64(r3t_rebuilt[j, i]),
                },
                "qt_s1_stored": {
                    "float": float(qn),
                    "hex": _hex64(qn),
                },
                "qt_s1_rebuilt": {
                    "float": float(np.float64(1.0) + r3t_rebuilt[j, i]),
                    "hex": _hex64(np.float64(1.0) + r3t_rebuilt[j, i]),
                },
                "e3t_0": {
                    "float": float(e3t_0[j, i, k]),
                    "hex": _hex64(e3t_0[j, i, k]),
                },
                "e3t_kmm_stored": {
                    "float": float(e3t_st),
                    "hex": _hex64(e3t_st),
                },
                "e3t_kmm_rebuilt": {
                    "float": float(e3t_rb),
                    "hex": _hex64(e3t_rb),
                },
                "h_scaled": {
                    "float": float(h_scaled),
                    "hex": _hex64(h_scaled),
                },
                "krhs_h_stored": {
                    "float": float(krhs_h_st),
                    "hex": _hex64(krhs_h_st),
                },
                "krhs_h_rebuilt": {
                    "float": float(krhs_h_rb),
                    "hex": _hex64(krhs_h_rb),
                },
                "v_scaled": {
                    "float": float(v_scaled),
                    "hex": _hex64(v_scaled),
                },
                "v_div_stored": {
                    "float": float(v_div_st),
                    "hex": _hex64(v_div_st),
                },
                "v_div_rebuilt": {
                    "float": float(v_div_rb),
                    "hex": _hex64(v_div_rb),
                },
                "rhs_T_stored": {
                    "float": float(krhs_st),
                    "hex": _hex64(krhs_st),
                },
                "rhs_T_rebuilt": {
                    "float": float(krhs_rb),
                    "hex": _hex64(krhs_rb),
                },
                "qb_times_T_b": {
                    "float": float(term1),
                    "hex": _hex64(term1),
                },
                "rDt_times_qn_times_rhs_stored": {
                    "float": float(term2_st),
                    "hex": _hex64(term2_st),
                },
                "rDt_times_qn_times_rhs_rebuilt": {
                    "float": float(term2_rb),
                    "hex": _hex64(term2_rb),
                },
                "numerator_stored": {
                    "float": float(num_st),
                    "hex": _hex64(num_st),
                },
                "numerator_rebuilt": {
                    "float": float(num_rb),
                    "hex": _hex64(num_rb),
                },
                "qt_s2": {
                    "float": float(qa),
                    "hex": _hex64(qa),
                },
                "T_stage2_oracle": {
                    "float": float(s2["T"][j, i, k]),
                    "hex": _hex64(s2["T"][j, i, k]),
                },
                "T_stage2_current": {
                    "float": float(out_rb),
                    "hex": _hex64(out_rb),
                },
                "T_stage2_proposed": {
                    "float": float(out_st),
                    "hex": _hex64(out_st),
                },
                "residual_current_vs_oracle": float(out_rb - s2["T"][j, i, k]),
                "residual_proposed_vs_oracle": float(out_st - s2["T"][j, i, k]),
            }

    require(
        total_int_wet_T_curr_unequal == 1
        and total_int_wet_T_prop_unequal == 0
        and total_int_wet_S_curr_unequal == 0
        and total_int_wet_S_prop_unequal == 0,
        f"Stage-2 replay closure failed: T_curr={total_int_wet_T_curr_unequal}, "
        f"T_prop={total_int_wet_T_prop_unequal}, S_curr={total_int_wet_S_curr_unequal}, "
        f"S_prop={total_int_wet_S_prop_unequal}",
    )
    require(kt10_cell_trace is not None, "kt=10 cell trace is missing")
    require(
        kt10_cell_trace["T_stage2_proposed"]["hex"]
        == kt10_cell_trace["T_stage2_oracle"]["hex"]
        == "0xbff663da73f853c6"
        and kt10_cell_trace["T_stage2_current"]["hex"] == "0xbff663da73f853c7",
        f"kt=10 cell hex trace mismatch: {kt10_cell_trace}",
    )

    return {
        "format": "nemo-testcase-l4-orca2-round341-walk-v1",
        "status": "PASS_R341_SOURCE_ORDER_WALK",
        "cross_record_r5_vs_r326_stage12_unequal": cross_record_unequal,
        "first_raw_non_bit_row_walk": {
            "row": {"kt": 1, "checkpoint": "stage1", "field": "v"},
            "verdict": "CONFIRMED",
            "wet_3d_count": int(np.count_nonzero(vmask_3d_bool)),
            "wet_3d_unequal": 0,
            "dry_3d_unequal": 7534,
            "subbottom_dry_unequal": 5550,
            "surface_dry_unequal": 1984,
            "subbottom_dry_ssvmask_unequal": 5884,
            "surface_dry_ssvmask_unequal": 1650,
            "north_fold_subbottom_dry_faces": 334,
            "legoesm_dry_word_hex": "0x8000000000000000",
            "raw_nemo_dry_word_hex": "0x8000000000000000",
            "canonicalized_oracle_dump_dry_word_hex": "0x0000000000000000",
            "oracle_dry_non_poszero_words": oracle_dry_nonzero_or_negzero,
            "record_build_ppsrc_citation": (
                "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:479-525"
            ),
            "canonicalizer_ppsrc_citation": (
                "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:488-520"
            ),
            "r326_ppsrc_citation": (
                "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:846-892"
            ),
            "r332_raw_v_audit": r332_raw_v_summary,
        },
        "first_wet_3d_non_bit_row_walk": {
            "row": {"kt": 10, "checkpoint": "stage2", "field": "T"},
            "verdict": "CONFIRMED",
            "record_build_ppsrc_citations": {
                "stage1_r3t_store": (
                    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:177"
                ),
                "stage1_to_stage2_kmm_swap": (
                    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:209"
                ),
                "stage2_tra_adv_cen_e3t_kmm": (
                    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:158-160,219-220,226-227"
                ),
            },
            "legoesm_citations": {
                "stage_transport_h_adv_rebuilt": (
                    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2293-2295"
                ),
                "stage2_transport_call": (
                    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:9570-9572"
                ),
                "stage2_cen2_rhs_call": (
                    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2665,2706"
                ),
                "stored_h_s2_kmm_definition": (
                    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:8820-8823"
                ),
            },
            "total_interior_wet_T_current_unequal_kt1_10": total_int_wet_T_curr_unequal,
            "total_interior_wet_T_proposed_unequal_kt1_10": total_int_wet_T_prop_unequal,
            "total_interior_wet_S_current_unequal_kt1_10": total_int_wet_S_curr_unequal,
            "total_interior_wet_S_proposed_unequal_kt1_10": total_int_wet_S_prop_unequal,
            "kt10_cell_trace": kt10_cell_trace,
            "per_kt_stage2_summary": per_kt_stage2_summary,
        },
    }


def _row(rows: list[dict[str, object]], target: tuple[int, str, str]) -> dict[str, object]:
    for row in rows:
        if (int(row["kt"]), str(row["checkpoint"]), str(row["field"])) == target:
            return row
    raise GateError(f"row {target} missing")


def classify(
    control_path: Path,
    candidate_path: Path,
    *,
    allow_cross_tree_parent: bool = False,
    expect_parent_commit: str = PARENT_COMMIT,
    expect_commit: str | None = None,
    plant: str = "none",
) -> dict[str, object]:
    control = json.loads(control_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if plant == "parent-commit":
        control = dict(control)
        control["worktree"] = {
            **dict(control.get("worktree", {})),
            "commit": PLANTED_WRONG_PARENT_COMMIT,
        }
    valid_statuses = ("PASS_R341_PAIR_RESCORE", "PASS_R341_D122_ARM")
    require(
        control.get("status") in valid_statuses,
        f"control measurement not admitted: {control.get('status')}",
    )
    require(
        candidate.get("status") in valid_statuses,
        f"candidate measurement not admitted: {candidate.get('status')}",
    )
    require(
        bool(control.get("worktree", {}).get("clean")) is True,
        "control measurement worktree is not clean",
    )
    require(
        bool(candidate.get("worktree", {}).get("clean")) is True,
        "candidate measurement worktree is not clean",
    )
    require(
        (control["rung"], control["label"]) == (candidate["rung"], candidate["label"]),
        "control/candidate pair mismatch",
    )
    ctrl_commit = str(control["worktree"]["commit"]).lower()
    cand_commit = str(candidate["worktree"]["commit"]).lower()
    if expect_commit is not None:
        require(
            cand_commit == expect_commit.lower(),
            f"candidate commit {cand_commit} != expected {expect_commit}",
        )
    if allow_cross_tree_parent:
        valid_parent_commits = {
            expect_parent_commit.lower(),
            PREFIX_RESCORE_COMMIT.lower(),
        }
        require(
            ctrl_commit in valid_parent_commits,
            f"cross-tree control commit {ctrl_commit} not in {sorted(valid_parent_commits)}",
        )
        require(
            ctrl_commit != cand_commit,
            "cross-tree comparison has control commit == candidate commit",
        )
    else:
        require(
            ctrl_commit == cand_commit,
            f"same-tree comparison commits differ: {ctrl_commit} vs {cand_commit}",
        )

    before = [dict(r) for r in control["ladder"]["rows"]]
    after = [dict(r) for r in candidate["ladder"]["rows"]]
    require(len(before) == len(after), "row count mismatch")
    if plant == "exact-loss":
        idx = next(i for i, r in enumerate(after) if r["bit_identical"])
        after[idx] = {
            **after[idx],
            "bit_identical": False,
            "unequal": 1,
            "max_abs": 2.220446049250313e-16,
            "rms": 1.0e-18,
            "classification": "AT_BAR_NOT_EXACT",
        }
    elif plant == "away-row":
        idx = next(
            (i for i, r in enumerate(after) if not r["bit_identical"]),
            0,
        )
        after[idx] = {
            **after[idx],
            "bit_identical": False,
            "unequal": int(after[idx]["unequal"]) + 1,
            "max_abs": float(after[idx]["max_abs"]) + 1.0e-12,
            "rms": float(after[idx]["rms"]) + 1.0e-14,
            "classification": "OVER_BAR",
        }

    pairs = list(zip(before, after, strict=True))
    require(
        all(
            (a["kt"], a["checkpoint"], a["field"])
            == (b["kt"], b["checkpoint"], b["field"])
            for a, b in pairs
        ),
        "row order moved",
    )
    exact_losses = [
        [a["kt"], a["checkpoint"], a["field"]]
        for a, b in pairs
        if a["bit_identical"] and not b["bit_identical"]
    ]
    exact_gains = [
        [a["kt"], a["checkpoint"], a["field"]]
        for a, b in pairs
        if (not a["bit_identical"]) and b["bit_identical"]
    ]
    rms_directions = Counter(
        r276.decision96._direction(a, b, "rms") for a, b in pairs
    )
    max_abs_directions = Counter(
        r276.decision96._direction(a, b, "max_abs") for a, b in pairs
    )
    unequal_increases = [
        [a["kt"], a["checkpoint"], a["field"], int(a["unequal"]), int(b["unequal"])]
        for a, b in pairs
        if int(b["unequal"]) > int(a["unequal"])
    ]
    unequal_decreases = [
        [a["kt"], a["checkpoint"], a["field"], int(a["unequal"]), int(b["unequal"])]
        for a, b in pairs
        if int(b["unequal"]) < int(a["unequal"])
    ]
    old_first_over = next(
        (row for row in before if row["classification"] == "OVER_BAR"), None
    )
    if old_first_over is not None:
        target = (
            int(old_first_over["kt"]),
            str(old_first_over["checkpoint"]),
            str(old_first_over["field"]),
        )
        old_target_row = old_first_over
    else:
        target = (10, "stage2", "T")
        old_target_row = _row(before, target)
    new_at_target = _row(after, target)
    first_direction = r276.decision96._direction(old_target_row, new_at_target, "rms")
    new_first_over = next(
        (
            [int(r["kt"]), str(r["checkpoint"]), str(r["field"])]
            for r in after
            if r["classification"] == "OVER_BAR"
        ),
        None,
    )
    passed = (
        not exact_losses
        and rms_directions["away"] == 0
        and max_abs_directions["away"] == 0
        and not unequal_increases
        and first_direction in ("toward", "equal")
    )
    return {
        "format": "nemo-testcase-l4-orca2-round341-d122-census-v1",
        "status": "PASS_R341_D122_CENSUS" if passed else "FAIL_R341_D122_CENSUS",
        "rung": candidate["rung"],
        "label": candidate["label"],
        "control_commit": control["worktree"]["commit"],
        "worktree_commit": candidate["worktree"]["commit"],
        "row_count": len(after),
        "exact_row_losses": exact_losses,
        "exact_row_gains": exact_gains,
        "unequal_increases": unequal_increases,
        "unequal_decreases": unequal_decreases,
        "first_target": {
            "old_target": list(target),
            "new_first_over_bar": new_first_over,
            "rms_direction": first_direction,
            "old_unequal": int(old_target_row["unequal"]),
            "new_unequal": int(new_at_target["unequal"]),
            "old_max_abs": float(old_target_row["max_abs"]),
            "new_max_abs": float(new_at_target["max_abs"]),
            "old_rms": float(old_target_row["rms"]),
            "new_rms": float(new_at_target["rms"]),
        },
        "registered_census": {
            name: int(rms_directions[name]) for name in ("toward", "away", "equal")
        },
        "max_abs_census": {
            name: int(max_abs_directions[name]) for name in ("toward", "away", "equal")
        },
        "d122_c_and_d": passed,
    }


def decide(
    paths: list[Path],
    *,
    allow_cross_tree_parent: bool = False,
    expect_parent_commit: str = PARENT_COMMIT,
    expect_commit: str | None = None,
) -> dict[str, object]:
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    require(
        sorted((doc["rung"], doc["label"]) for doc in docs) == sorted(PAIRS),
        "eight-pair set is incomplete",
    )
    require(
        len({doc["worktree_commit"] for doc in docs}) == 1,
        "censuses mix candidate commits",
    )
    require(
        len({doc["control_commit"] for doc in docs}) == 1,
        "censuses mix control commits",
    )
    if expect_commit is not None:
        require(
            str(docs[0]["worktree_commit"]).lower() == expect_commit.lower(),
            f"decision worktree_commit {docs[0]['worktree_commit']} != {expect_commit}",
        )
    if allow_cross_tree_parent:
        valid_parent_commits = {
            expect_parent_commit.lower(),
            PREFIX_RESCORE_COMMIT.lower(),
        }
        require(
            str(docs[0]["control_commit"]).lower() in valid_parent_commits,
            f"decision control_commit {docs[0]['control_commit']} not in {sorted(valid_parent_commits)}",
        )
        require(
            str(docs[0]["worktree_commit"]).lower()
            != str(docs[0]["control_commit"]).lower(),
            "cross-tree decision has control_commit == worktree_commit",
        )
    else:
        require(
            docs[0]["control_commit"] == docs[0]["worktree_commit"],
            "same-tree decision control_commit != worktree_commit",
        )
    require(
        all(
            doc["status"] == "PASS_R341_D122_CENSUS" and doc["d122_c_and_d"]
            for doc in docs
        ),
        "a pair failed Decision 122",
    )
    return {
        "format": "nemo-testcase-l4-orca2-round341-d122-decision-v1",
        "status": "PASS_R341_D122_DECISION",
        "control_commit": docs[0]["control_commit"],
        "worktree_commit": docs[0]["worktree_commit"],
        "cards": docs,
        "d122_c_and_d": True,
    }


def _load_git_module(repo: Path, commit: str, rel_path: str, mod_name: str):
    src = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{rel_path}"],
        text=True,
    )
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"{repo}@{commit}:{rel_path}"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _load_file_module(path: Path, mod_name: str):
    saved_path = list(sys.path)
    src = path.read_text(encoding="utf-8")
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(path.resolve())
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    extra = [
        p
        for p in sys.path
        if p not in saved_path
        and not p.startswith(str(GRIDLADDER_VMIX_LANE / "packages"))
    ]
    sys.path = [p for p in saved_path if p != str(GRIDLADDER_VMIX_LANE)] + [
        p for p in extra if p != str(GRIDLADDER_VMIX_LANE)
    ]
    return mod


def _non_nemo_stage_caller(*, fold: bool) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.grids.latlon import FoldDescriptor, create_latlon_geometry
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    ny, nx, nz = 4, 6, 4
    perm = np.array([(nx - i) % nx for i in range(nx)], dtype=np.int32)
    rng = np.random.default_rng(19 if fold else 23)
    raw = create_z_star_from_thicknesses([1.0, 2.0, 3.0, 4.0])
    z = create_partial_cell_coordinate(raw, jnp.full((ny, nx), 6.0, dtype=jnp.float64))
    grid = create_latlon_geometry(n_lat=ny, n_lon=nx)
    if fold:
        grid = grid._replace(
            fold=FoldDescriptor(
                is_active=True,
                fold_j=ny - 1,
                cap_j=0,
                perm_T=jnp.asarray(perm),
                perm_v=jnp.asarray(perm),
                vector_sign_u=-1.0,
                vector_sign_v=-1.0,
            )
        )
    cfg = omlc.LatLonCGridOceanConfig()
    eta_b = jnp.asarray(rng.uniform(-0.04, 0.04, (ny, nx)), dtype=jnp.float64)
    eta_a = jnp.asarray(rng.uniform(-0.04, 0.04, (ny, nx)), dtype=jnp.float64)
    u = jnp.asarray(rng.uniform(-0.05, 0.05, (ny, nx + 1, nz)), dtype=jnp.float64)
    v = jnp.asarray(rng.uniform(-0.05, 0.05, (ny + 1, nx, nz)), dtype=jnp.float64)
    T = jnp.asarray(rng.uniform(5.0, 15.0, (ny, nx, nz)), dtype=jnp.float64)
    S = jnp.asarray(rng.uniform(34.0, 35.5, (ny, nx, nz)), dtype=jnp.float64)
    H = jnp.full((ny, nx), 6.0, dtype=jnp.float64)
    d2, d3 = ("lat", "lon"), ("lat", "lon", "level")
    state = SimpleNamespace(
        T=Field(T, "T", d3, "degC"),
        S=Field(S, "S", d3, "psu"),
        u=Field(u, "u", d3, "m/s"),
        v=Field(v, "v", d3, "m/s"),
        eta=Field(eta_b, "eta", d2, "m"),
        H_bathy=Field(H, "H_bathy", d2, "m"),
        u_mask=Field(jnp.ones((ny, nx + 1), dtype=jnp.float64), "u_mask", d2, "1"),
        v_mask=Field(jnp.ones((ny + 1, nx), dtype=jnp.float64), "v_mask", d2, "1"),
        uu_b=Field(jnp.mean(u, axis=-1), "uu_b", d2, "m/s"),
        vv_b=Field(jnp.mean(v, axis=-1), "vv_b", d2, "m/s"),
    )
    model = SimpleNamespace(
        config=cfg,
        z_coord=z,
        grid=grid,
        _nemo_ws_test_hooks=omlc._NEMOWSRK3TestHooks(),
    )
    return _eval_stage_transport_caller(
        f"non_nemo_{'folded' if fold else 'unfolded'}",
        model,
        state,
        1800.0,
        eta_a=eta_a,
        T_s1=T + 0.01,
        S_s1=S - 0.005,
        u1_corr=u * 0.95,
        v1_corr=v * 0.95,
        u2_corr=u * 0.90,
        v2_corr=v * 0.90,
    )


def _eval_legacy_caller_10145(card) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc

    zc = card.recipe.z_coord
    grid = card.recipe.grid
    cfg = card.recipe.model_config
    st0 = card.recipe.initial_state
    dt = float(card.dt_s)
    h_ref = omlc.compute_layer_thickness(
        jnp.zeros_like(st0.eta.data),
        st0.H_bathy.data,
        zc,
        min_water_column_m=getattr(cfg, "min_water_column_m", None),
    )
    umask, vmask = omlc.compute_face_masks_3d(zc.is_active, grid)
    umask = umask.astype(jnp.float64)
    vmask = vmask.astype(jnp.float64)
    eta_b = st0.eta.data + 0.03 * st0.land_mask.data
    eta_a = st0.eta.data - 0.02 * st0.land_mask.data
    u_arr = st0.u.data + 0.04 * umask
    v_arr = st0.v.data - 0.03 * vmask
    Hu_avg = jnp.sum(u_arr, axis=-1) * st0.u_mask.data
    Hv_avg = jnp.sum(v_arr, axis=-1) * st0.v_mask.data
    state = st0._replace(
        eta=Field(eta_b, st0.eta.name, st0.eta.dims, st0.eta.units),
        u=Field(u_arr, st0.u.name, st0.u.dims, st0.u.units),
        v=Field(v_arr, st0.v.name, st0.v.dims, st0.v.units),
    )
    state_new = st0._replace(
        eta=Field(eta_a, st0.eta.name, st0.eta.dims, st0.eta.units),
    )
    model_obj = SimpleNamespace(
        config=cfg,
        z_coord=zc,
        grid=grid,
        _nemo_ws_test_hooks=omlc._NEMOWSRK3TestHooks(),
    )
    src_lines = Path(omlc.__file__).read_text(encoding="utf-8").splitlines()
    legacy_block = textwrap.dedent("\n".join(src_lines[10130:10162]))
    require(
        "_nemo_ws_stage_transport(" in legacy_block,
        "legacy caller block at :10131-10162 missing _nemo_ws_stage_transport",
    )
    ns = {
        **omlc.__dict__,
        "h_k_old": h_ref,
        "h_k_new": h_ref * 1.01,
        "state": state,
        "state_new": state_new,
        "_zc": zc,
        "_cfg_b": cfg,
        "Hu_avg": Hu_avg,
        "Hv_avg": Hv_avg,
        "u_mask_3d_tracer": umask,
        "v_mask_3d_tracer": vmask,
        "_grid": grid,
        "dt": dt,
        "self": model_obj,
        "_nemo_ws_velocity_stages": (
            (u_arr, v_arr),
            (u_arr * 0.9, v_arr * 0.9),
            (u_arr * 0.8, v_arr * 0.8),
        ),
    }
    exec(compile(legacy_block, "<legacy_caller_10145>", "exec"), ns)
    geom = ns["_nemo_ws_stage_transport_geometry"]
    arrays: dict[str, np.ndarray] = {}
    for stg_idx, tup in enumerate(geom, start=1):
        arrays[f"s{stg_idx}_u_tr"] = np.asarray(tup[0], dtype=np.float64)
        arrays[f"s{stg_idx}_v_tr"] = np.asarray(tup[1], dtype=np.float64)
        arrays[f"s{stg_idx}_w"] = np.asarray(tup[2], dtype=np.float64)
        arrays[f"s{stg_idx}_h"] = np.asarray(tup[3], dtype=np.float64)
    meta = {
        "rk3_stage2_tracer_nemo_stored_thickness": bool(
            getattr(cfg, "rk3_stage2_tracer_nemo_stored_thickness", False)
        ),
        "caller_site": "ocean_model_latlon_cgrid.py:10145",
    }
    return arrays, meta


def _eval_stage_transport_caller(
    name: str,
    model,
    state,
    dt: float,
    *,
    eta_a,
    T_s1,
    S_s1,
    u1_corr,
    v1_corr,
    u2_corr,
    v2_corr,
    Hu_avg=None,
    Hv_avg=None,
    freshwater=None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc

    zc = model.z_coord
    grid = model.grid
    cfg = model.config
    h_k_old = omlc.compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        zc,
        min_water_column_m=getattr(cfg, "min_water_column_m", None),
    )
    mask_3d = (
        zc.is_active.astype(h_k_old.dtype)
        if hasattr(zc, "is_active") and zc.is_active is not None
        else jnp.ones_like(state.T.data)
    )
    u0 = state.u.data
    v0 = state.v.data
    if Hu_avg is None:
        Hu_avg = jnp.sum(u0, axis=-1) * state.u_mask.data
    if Hv_avg is None:
        Hv_avg = jnp.sum(v0, axis=-1) * state.v_mask.data
    target_u = (
        state.uu_b.data
        if getattr(state, "uu_b", None) is not None and state.uu_b is not None
        else jnp.zeros_like(state.u_mask.data)
    )
    target_v = (
        state.vv_b.data
        if getattr(state, "vv_b", None) is not None and state.vv_b is not None
        else jnp.zeros_like(state.v_mask.data)
    )
    state_new = SimpleNamespace(
        eta=Field(eta_a, state.eta.name, state.eta.dims, state.eta.units),
        H_bathy=state.H_bathy,
        T=state.T,
        S=state.S,
        u=state.u,
        v=state.v,
        uu_b=getattr(state, "uu_b", None),
        vv_b=getattr(state, "vv_b", None),
    )
    zero_src = jnp.zeros_like(state.T.data)
    stage_source_rates = ((zero_src, zero_src), (zero_src, zero_src), (zero_src, zero_src))
    stage12_source_terms = None

    src_lines = Path(omlc.__file__).read_text(encoding="utf-8").splitlines()
    block_setup = textwrap.dedent("\n".join(src_lines[8638:8840]))
    block_helpers = textwrap.dedent("\n".join(src_lines[9275:9320]))
    block_s1 = textwrap.dedent("\n".join(src_lines[9340:9354]))
    block_s2 = textwrap.dedent("\n".join(src_lines[9569:9595]))
    block_s2_tr = textwrap.dedent("\n".join(src_lines[9622:9631]))
    block_s3 = textwrap.dedent("\n".join(src_lines[9640:9658]))
    full_block = "\n".join(
        (block_setup, block_helpers, block_s1, block_s2, block_s2_tr, block_s3)
    )
    require(
        "_g0 = _nemo_ws_stage_transport(" in block_s1
        and "_g1 = _nemo_ws_stage_transport(" in block_s2
        and "_g2 = _nemo_ws_stage_transport(" in block_s3,
        "stage transport blocks shifted in ocean_model_latlon_cgrid.py",
    )
    ns = {
        **omlc.__dict__,
        "self": model,
        "state": state,
        "state_new": state_new,
        "_zc": zc,
        "_grid": grid,
        "_cfg_b": cfg,
        "dt": dt,
        "h_k_old": h_k_old,
        "mask_3d": mask_3d,
        "Hu_avg": Hu_avg,
        "Hv_avg": Hv_avg,
        "freshwater": freshwater,
        "u0": u0,
        "v0": v0,
        "u1_corr": u1_corr,
        "v1_corr": v1_corr,
        "u2_corr": u2_corr,
        "v2_corr": v2_corr,
        "_T_stage1": T_s1,
        "_S_stage1": S_s1,
        "target_u": target_u,
        "target_v": target_v,
        "_stage_source_rates": stage_source_rates,
        "_stage12_source_terms": stage12_source_terms,
    }
    exec(compile(full_block, f"<stage_caller_{name}>", "exec"), ns)
    g0 = ns["_g0"]
    g1 = ns["_g1"]
    g2 = ns["_g2"]
    T_s2 = ns["_T_stage2"]
    S_s2 = ns["_S_stage2"]

    arrays = {
        "s1_u_tr": np.asarray(g0[0], dtype=np.float64),
        "s1_v_tr": np.asarray(g0[1], dtype=np.float64),
        "s1_w": np.asarray(g0[2], dtype=np.float64),
        "s1_h": np.asarray(g0[3], dtype=np.float64),
        "s2_u_tr": np.asarray(g1[0], dtype=np.float64),
        "s2_v_tr": np.asarray(g1[1], dtype=np.float64),
        "s2_w": np.asarray(g1[2], dtype=np.float64),
        "s2_h": np.asarray(g1[3], dtype=np.float64),
        "T_stage2": np.asarray(T_s2, dtype=np.float64),
        "S_stage2": np.asarray(S_s2, dtype=np.float64),
        "s3_u_tr": np.asarray(g2[0], dtype=np.float64),
        "s3_v_tr": np.asarray(g2[1], dtype=np.float64),
        "s3_w": np.asarray(g2[2], dtype=np.float64),
        "s3_h": np.asarray(g2[3], dtype=np.float64),
    }
    meta = {
        "rk3_stage2_tracer_nemo_stored_thickness": bool(
            getattr(cfg, "rk3_stage2_tracer_nemo_stored_thickness", False)
        ),
        "caller_site": "ocean_model_latlon_cgrid.py:9341,9570,9626,9641",
    }
    return arrays, meta


def caller_audit_dump(
    out_dir: Path, deck_root: Path, expect_commit: str
) -> dict[str, object]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
        dino_lat_lon_state,
        dino_lat_lon_vertical,
        nemo_faithful_dino_config,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        build_lock_exchange_zco_card,
        build_orca2_zps_card,
        build_overflow_zps_card,
        build_vortex_zco_card,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    )

    model_file = Path(model_module.__file__).resolve()
    require(
        model_file.is_relative_to(ACTIVE_REPO_ROOT),
        f"model_module {model_file} is outside active repo {ACTIVE_REPO_ROOT}",
    )
    stamp = worktree_stamp(repo=ACTIVE_REPO_ROOT)
    require(
        bool(stamp["clean"]) and str(stamp["commit"]).lower() == expect_commit.lower(),
        f"caller_audit_dump requires clean committed tree at {expect_commit}: got {stamp}",
    )
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
        and bool(jax.config.jax_enable_x64),
        "fp64/libm is not active",
    )

    cls = model_module.LatLonCGridOceanModel
    g_recipe = _load_git_module(
        GRIDLADDER_LANE,
        GRIDLADDER_RECIPE_COMMIT,
        "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
        "g_recipe",
    )
    v_recipe = _load_git_module(
        GRIDLADDER_VMIX_LANE,
        GRIDLADDER_VMIX_RECIPE_COMMIT,
        "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
        "v_recipe",
    )

    all_arrays: dict[str, np.ndarray] = {}
    caller_meta: dict[str, dict[str, object]] = {}

    def _record_caller(
        caller_name: str,
        arrays: dict[str, np.ndarray],
        meta: dict[str, object],
    ) -> None:
        array_meta: dict[str, dict[str, object]] = {}
        for arr_name, arr in arrays.items():
            contig = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
            all_arrays[f"{caller_name}__{arr_name}"] = contig
            array_meta[arr_name] = {
                "shape": list(contig.shape),
                "elements": int(contig.size),
                "sha256": hashlib.sha256(contig.tobytes()).hexdigest(),
            }
        caller_meta[caller_name] = {
            **meta,
            "array_names": list(arrays.keys()),
            "arrays": array_meta,
        }

    # 1-2. Non-NEMO callers (folded and unfolded)
    arr_f, meta_f = _non_nemo_stage_caller(fold=True)
    _record_caller("non_nemo_folded", arr_f, meta_f)
    arr_nf, meta_nf = _non_nemo_stage_caller(fold=False)
    _record_caller("non_nemo_unfolded", arr_nf, meta_nf)

    # 3. Legacy caller (:10145)
    gyre_card = build_gyre_zco_card()
    arr_leg, meta_leg = _eval_legacy_caller_10145(gyre_card)
    _record_caller("legacy_caller_10145", arr_leg, meta_leg)

    def _eval_card(cname: str, model_obj, st0, dt_s: float, *, active_ssh: bool):
        if active_ssh:
            eta_b = st0.eta.data - 0.17 * st0.land_mask.data
            eta_a = st0.eta.data + 0.08 * st0.land_mask.data
            u_in = st0.u.data + 0.04 * st0.u_mask.data[..., None]
            v_in = st0.v.data - 0.03 * st0.v_mask.data[..., None]
        else:
            eta_b = st0.eta.data
            eta_a = st0.eta.data
            u_in = st0.u.data
            v_in = st0.v.data
        st = st0._replace(
            eta=Field(eta_b, st0.eta.name, st0.eta.dims, st0.eta.units),
            u=Field(u_in, st0.u.name, st0.u.dims, st0.u.units),
            v=Field(v_in, st0.v.name, st0.v.dims, st0.v.units),
        )
        return _eval_stage_transport_caller(
            cname,
            model_obj,
            st,
            dt_s,
            eta_a=eta_a,
            T_s1=st.T.data,
            S_s1=st.S.data,
            u1_corr=u_in,
            v1_corr=v_in,
            u2_corr=u_in,
            v2_corr=v_in,
        )

    # 4. Faithful DINO (rk3_stage2_tracer_nemo_stored_thickness=False)
    dino_cfg = nemo_faithful_dino_config(dino_config_for_recipe("nemo_dino_kamm_mlf"))
    dino_grid = dino_lat_lon_grid(dino_cfg, n_lon=50)
    dino_z = dino_lat_lon_vertical(dino_grid, dino_cfg)
    dino_state0 = dino_lat_lon_state(dino_grid, dino_z, dino_cfg)
    dino_mcfg, _ = dino_lat_lon_model_config(dino_grid, dino_cfg)
    dino_model = cls(dino_grid, dino_z, dino_mcfg)
    arr_dino, meta_dino = _eval_card(
        "faithful_dino", dino_model, dino_state0, 2700.0, active_ssh=True
    )
    require(
        not meta_dino["rk3_stage2_tracer_nemo_stored_thickness"],
        f"faithful_dino flag unexpected: {meta_dino}",
    )
    _record_caller("faithful_dino", arr_dino, meta_dino)

    # 5. ORCA2V1-tke
    v1_deck = GRIDLADDER_VMIX_ROUNDS / "round1/oracle_v1_r1/twin_a"
    v1_card = v_recipe.build_nemo_testcase_card("ORCA2V1-tke", deck_root=v1_deck)
    v1_model = cls(
        v1_card.recipe.grid,
        v1_card.recipe.z_coord,
        v1_card.recipe.model_config,
        iwm_forcing=v1_card.recipe.iwm_forcing,
    )
    arr_v1, meta_v1 = _eval_card(
        "ORCA2V1-tke",
        v1_model,
        v1_card.recipe.initial_state,
        float(v1_card.dt_s),
        active_ssh=False,
    )
    _record_caller("ORCA2V1-tke", arr_v1, meta_v1)

    # 6-10. Standard non-ORCA2 cards + ORCA2-rung0
    std_cards = (
        ("GYRE-zco", gyre_card, True),
        ("VORTEX-zco", build_vortex_zco_card(), True),
        ("LOCK_EXCHANGE-zco", build_lock_exchange_zco_card(), True),
        ("OVERFLOW-zps", build_overflow_zps_card(), True),
        ("ORCA2-rung0", rung0.build_rung0_card(deck_root), False),
    )
    for cname, card, act_ssh in std_cards:
        cmodel = cls(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            iwm_forcing=card.recipe.iwm_forcing,
        )
        c_arr, c_meta = _eval_card(
            cname, cmodel, card.recipe.initial_state, float(card.dt_s), active_ssh=act_ssh
        )
        _record_caller(cname, c_arr, c_meta)

    # 11-19. ORCA2G1..G8 gridladder cards
    g_decks = (
        ("ORCA2G1-flat", GRIDLADDER_ROUNDS / "round4/oracle_g1_r4/p3_a_100step", True),
        ("ORCA2G1b-vec", GRIDLADDER_ROUNDS / "round22/oracle_g1b_r22/ref_100step", True),
        ("ORCA2G2-zps", GRIDLADDER_ROUNDS / "round10/oracle_g2_r10/p3_a_100step", True),
        ("ORCA2G3-ts", GRIDLADDER_ROUNDS / "round21/oracle_g3_r21/p3_a", True),
        ("ORCA2G4-sbc", GRIDLADDER_ROUNDS / "round21/oracle_g4_r21/p6_a", True),
        ("ORCA2G5-ice", GRIDLADDER_ROUNDS / "round31/oracle_g5_r31/twin_a", False),
        ("ORCA2G6-vec", GRIDLADDER_ROUNDS / "round36/oracle_g6_r36/twin_a", False),
        ("ORCA2G7-drg", GRIDLADDER_ROUNDS / "round41/oracle_g7_r41/twin_a", False),
        ("ORCA2G8-ldf", GRIDLADDER_ROUNDS / "round45/oracle_g8_r45/twin_a", False),
    )
    for gname, gdeck, act_ssh in g_decks:
        gcard = g_recipe.build_nemo_testcase_card(gname, deck_root=gdeck)
        gmodel = cls(
            gcard.recipe.grid,
            gcard.recipe.z_coord,
            gcard.recipe.model_config,
            iwm_forcing=gcard.recipe.iwm_forcing,
        )
        g_arr, g_meta = _eval_card(
            gname, gmodel, gcard.recipe.initial_state, float(gcard.dt_s), active_ssh=act_ssh
        )
        _record_caller(gname, g_arr, g_meta)

    # 20. ORCA2-zps_kt1_rest (standalone K0 card at kt=1 rest entry)
    orca2_card = build_orca2_zps_card(deck_root)
    orca2_model = cls(
        orca2_card.recipe.grid,
        orca2_card.recipe.z_coord,
        orca2_card.recipe.model_config,
        iwm_forcing=orca2_card.recipe.iwm_forcing,
    )
    arr_kt1, meta_kt1 = _eval_card(
        "ORCA2-zps_kt1_rest",
        orca2_model,
        orca2_card.recipe.initial_state,
        float(orca2_card.dt_s),
        active_ssh=False,
    )
    require(
        meta_kt1["rk3_stage2_tracer_nemo_stored_thickness"],
        f"ORCA2-zps_kt1_rest flag unexpected: {meta_kt1}",
    )
    _record_caller("ORCA2-zps_kt1_rest", arr_kt1, meta_kt1)

    # 21. ORCA2-zps (standalone K0 card at kt=10 Stage-2 entry from admitted round-5 record)
    r5_root = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round5/acquisition/orca1ice_surface_entry_every_step_a_np2"
    )
    def _pad180_dict(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            k: np.concatenate([v, np.zeros_like(v)], axis=1) for k, v in d.items()
        }

    ek10 = _pad180_dict(
        rung10.read_state_frame(
            r5_root / "oracle_step_entry_kt00000010.bin", kt=10, stage=None
        )
    )
    s1_10 = _pad180_dict(
        rung10.read_state_frame(
            r5_root / "oracle_stage_kt00000010_s1.bin", kt=10, stage=1
        )
    )
    s2_10 = _pad180_dict(
        rung10.read_state_frame(
            r5_root / "oracle_stage_kt00000010_s2.bin", kt=10, stage=2
        )
    )
    s3_10 = _pad180_dict(
        rung10.read_state_frame(
            r5_root / "oracle_stage_kt00000010_s3.bin", kt=10, stage=3
        )
    )
    bt10 = _pad180_dict(
        r301.read_bt(r5_root / "oracle_bt_frames_kt00000010.bin", 10)
    )
    st0 = orca2_card.recipe.initial_state
    u_b_cgrid = jnp.concatenate([ek10["u"][:, -1:, :30], ek10["u"][:, :, :30]], axis=1)
    v_b_cgrid = jnp.concatenate([jnp.zeros_like(ek10["v"][:1, :, :30]), ek10["v"][:, :, :30]], axis=0)
    u1_cgrid = jnp.concatenate([s1_10["u"][:, -1:, :30], s1_10["u"][:, :, :30]], axis=1)
    v1_cgrid = jnp.concatenate([jnp.zeros_like(s1_10["v"][:1, :, :30]), s1_10["v"][:, :, :30]], axis=0)
    u2_cgrid = jnp.concatenate([s2_10["u"][:, -1:, :30], s2_10["u"][:, :, :30]], axis=1)
    v2_cgrid = jnp.concatenate([jnp.zeros_like(s2_10["v"][:1, :, :30]), s2_10["v"][:, :, :30]], axis=0)
    uu_b_cgrid = jnp.concatenate([bt10["uu_b"][:, -1:], bt10["uu_b"]], axis=1)
    vv_b_cgrid = jnp.concatenate([jnp.zeros_like(bt10["vv_b"][:1, :]), bt10["vv_b"]], axis=0)
    Hu_10 = jnp.concatenate([bt10["un_adv"][:, -1:], bt10["un_adv"]], axis=1)
    Hv_10 = jnp.concatenate([jnp.zeros_like(bt10["vn_adv"][:1, :]), bt10["vn_adv"]], axis=0)
    st_kt10 = st0._replace(
        T=Field(jnp.asarray(ek10["T"][:, :, :30], dtype=jnp.float64), st0.T.name, st0.T.dims, st0.T.units),
        S=Field(jnp.asarray(ek10["S"][:, :, :30], dtype=jnp.float64), st0.S.name, st0.S.dims, st0.S.units),
        u=Field(jnp.asarray(u_b_cgrid, dtype=jnp.float64), st0.u.name, st0.u.dims, st0.u.units),
        v=Field(jnp.asarray(v_b_cgrid, dtype=jnp.float64), st0.v.name, st0.v.dims, st0.v.units),
        eta=Field(jnp.asarray(ek10["ssh"], dtype=jnp.float64), st0.eta.name, st0.eta.dims, st0.eta.units),
        uu_b=Field(jnp.asarray(uu_b_cgrid, dtype=jnp.float64), "uu_b", st0.u_mask.dims, "m/s"),
        vv_b=Field(jnp.asarray(vv_b_cgrid, dtype=jnp.float64), "vv_b", st0.v_mask.dims, "m/s"),
    )
    arr_orca2, meta_orca2 = _eval_stage_transport_caller(
        "ORCA2-zps",
        orca2_model,
        st_kt10,
        float(orca2_card.dt_s),
        eta_a=jnp.asarray(s3_10["ssh"], dtype=jnp.float64),
        T_s1=jnp.asarray(s1_10["T"][:, :, :30], dtype=jnp.float64),
        S_s1=jnp.asarray(s1_10["S"][:, :, :30], dtype=jnp.float64),
        u1_corr=jnp.asarray(u1_cgrid, dtype=jnp.float64),
        v1_corr=jnp.asarray(v1_cgrid, dtype=jnp.float64),
        u2_corr=jnp.asarray(u2_cgrid, dtype=jnp.float64),
        v2_corr=jnp.asarray(v2_cgrid, dtype=jnp.float64),
        Hu_avg=jnp.asarray(Hu_10, dtype=jnp.float64),
        Hv_avg=jnp.asarray(Hv_10, dtype=jnp.float64),
    )
    require(
        meta_orca2["rk3_stage2_tracer_nemo_stored_thickness"],
        f"ORCA2-zps flag unexpected: {meta_orca2}",
    )
    _record_caller("ORCA2-zps", arr_orca2, meta_orca2)

    require(
        tuple(caller_meta.keys()) == CALLER_ORDER,
        f"caller order {tuple(caller_meta.keys())} != {CALLER_ORDER}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "arrays.npz", **all_arrays)
    manifest = {
        "format": "nemo-testcase-l4-orca2-round341-caller-dump-v1",
        "status": "PASS_R341_CALLER_DUMP",
        "worktree": stamp,
        "model_file": str(model_file),
        "caller_order": list(CALLER_ORDER),
        "callers": caller_meta,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def caller_audit_compare(
    parent_dump: Path,
    candidate_dump: Path,
    *,
    expect_parent_commit: str = PARENT_COMMIT,
    expect_commit: str | None = None,
    plant: str = "none",
) -> dict[str, object]:
    parent_manifest = json.loads(
        (parent_dump / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(
        (candidate_dump / "manifest.json").read_text(encoding="utf-8")
    )
    if plant == "parent-commit":
        parent_manifest = dict(parent_manifest)
        parent_manifest["worktree"] = {
            **dict(parent_manifest.get("worktree", {})),
            "commit": PLANTED_WRONG_PARENT_COMMIT,
        }
    require(
        parent_manifest.get("status") == "PASS_R341_CALLER_DUMP",
        f"parent dump not admitted: {parent_manifest.get('status')}",
    )
    require(
        candidate_manifest.get("status") == "PASS_R341_CALLER_DUMP",
        f"candidate dump not admitted: {candidate_manifest.get('status')}",
    )
    require(
        bool(parent_manifest.get("worktree", {}).get("clean")) is True,
        "parent dump worktree is not clean",
    )
    require(
        bool(candidate_manifest.get("worktree", {}).get("clean")) is True,
        "candidate dump worktree is not clean",
    )
    parent_commit = str(parent_manifest["worktree"]["commit"])
    candidate_commit = str(candidate_manifest["worktree"]["commit"])
    require(
        parent_commit.lower() == expect_parent_commit.lower(),
        f"parent dump commit {parent_commit} != expected parent {expect_parent_commit}",
    )
    require(
        candidate_commit.lower() != parent_commit.lower(),
        "parent and candidate dumps have identical commit",
    )
    if expect_commit is not None:
        require(
            candidate_commit.lower() == expect_commit.lower(),
            f"candidate dump commit {candidate_commit} != expected {expect_commit}",
        )
    require(
        tuple(parent_manifest["caller_order"]) == CALLER_ORDER
        and tuple(candidate_manifest["caller_order"]) == CALLER_ORDER,
        "caller_order moved between parent and candidate dumps",
    )

    parent_npz = np.load(parent_dump / "arrays.npz")
    candidate_npz = np.load(candidate_dump / "arrays.npz")

    callers_report: dict[str, dict[str, object]] = {}
    non_orca2_unequal_callers: list[str] = []

    for caller_name in CALLER_ORDER:
        p_info = parent_manifest["callers"][caller_name]
        c_info = candidate_manifest["callers"][caller_name]
        arr_names = list(p_info["array_names"])
        require(
            arr_names == list(c_info["array_names"]),
            f"{caller_name}: array_names moved {arr_names} vs {c_info['array_names']}",
        )
        arrays_report: dict[str, dict[str, object]] = {}
        total_elements = 0
        total_unequal_words = 0
        total_unequal_values = 0

        for arr_name in arr_names:
            key = f"{caller_name}__{arr_name}"
            a = np.ascontiguousarray(np.asarray(parent_npz[key], dtype=np.float64))
            b = np.ascontiguousarray(np.asarray(candidate_npz[key], dtype=np.float64))
            if (
                plant == "caller-unequal"
                and caller_name == "ORCA2V1-tke"
                and arr_name == "s2_h"
            ):
                b = b.copy()
                b.flat[0] = b.flat[0] + np.spacing(max(abs(float(b.flat[0])), 1.0))
            require(
                a.shape == b.shape,
                f"{caller_name}.{arr_name}: shape {a.shape} != {b.shape}",
            )
            p_sha = hashlib.sha256(a.tobytes()).hexdigest()
            c_sha = hashlib.sha256(b.tobytes()).hexdigest()
            require(
                p_sha == p_info["arrays"][arr_name]["sha256"],
                f"{caller_name}.{arr_name}: parent npz sha256 mismatch",
            )
            if not (
                plant == "caller-unequal"
                and caller_name == "ORCA2V1-tke"
                and arr_name == "s2_h"
            ):
                require(
                    c_sha == c_info["arrays"][arr_name]["sha256"],
                    f"{caller_name}.{arr_name}: candidate npz sha256 mismatch",
                )
            bit_diff = a.view(np.uint64) != b.view(np.uint64)
            val_diff = a != b
            unequal_words = int(np.count_nonzero(bit_diff))
            unequal_values = int(np.count_nonzero(val_diff))
            res = a - b
            max_abs = float(np.max(np.abs(res))) if unequal_values > 0 else 0.0
            rms = float(np.sqrt(np.mean(res * res))) if unequal_values > 0 else 0.0

            total_elements += int(a.size)
            total_unequal_words += unequal_words
            total_unequal_values += unequal_values
            arrays_report[arr_name] = {
                "shape": list(a.shape),
                "elements": int(a.size),
                "unequal_words": unequal_words,
                "unequal_values": unequal_values,
                "max_abs": max_abs,
                "rms": rms,
                "parent_sha256": p_sha,
                "candidate_sha256": c_sha,
            }

        if caller_name != "ORCA2-zps" and total_unequal_words > 0:
            non_orca2_unequal_callers.append(caller_name)

        callers_report[caller_name] = {
            "rk3_stage2_tracer_nemo_stored_thickness": c_info[
                "rk3_stage2_tracer_nemo_stored_thickness"
            ],
            "caller_site": c_info["caller_site"],
            "array_count": len(arr_names),
            "total_elements": total_elements,
            "total_unequal_words": total_unequal_words,
            "total_unequal_values": total_unequal_values,
            "arrays": arrays_report,
        }

    require(
        not non_orca2_unequal_callers,
        f"non-ORCA2 callers have unequal words vs parent: {non_orca2_unequal_callers}",
    )
    orca2_rep = callers_report["ORCA2-zps"]
    require(
        int(orca2_rep["total_unequal_words"]) > 0
        and int(orca2_rep["arrays"]["s2_h"]["unequal_words"]) > 0
        and int(orca2_rep["arrays"]["T_stage2"]["unequal_words"]) > 0
        and int(orca2_rep["arrays"]["s1_h"]["unequal_words"]) == 0
        and int(orca2_rep["arrays"]["s3_h"]["unequal_words"]) == 0,
        f"ORCA2-zps card did not move as expected vs parent: {orca2_rep}",
    )

    return {
        "format": "nemo-testcase-l4-orca2-round341-caller-audit-v1",
        "status": "PASS_R341_CALLER_AUDIT",
        "parent_commit": parent_commit,
        "worktree_commit": candidate_commit,
        "parent_worktree": parent_manifest["worktree"],
        "worktree": candidate_manifest["worktree"],
        "caller_order": list(CALLER_ORDER),
        "non_orca2_unequal_callers": non_orca2_unequal_callers,
        "callers": callers_report,
    }


def self_test(*, plant: str = "none", tmp_dir: Path | None = None) -> dict[str, object]:
    import tempfile

    require(plant in PLANT_CHOICES, f"unknown plant {plant}")

    # 1. Score bar and signed-zero control
    exact = np.array([1.0], np.float64)
    signed = np.array([-0.0], np.float64)
    at_bar = np.array([1.0 + np.spacing(1.0)], np.float64)
    over = np.array([1.0 + 8.0e-15], np.float64)
    if plant == "signed-zero":
        signed = np.array([0.0], np.float64)
    got_classes = (
        r332.score_with_bar(exact, exact)["classification"],
        r332.score_with_bar(signed, np.array([0.0], np.float64))["classification"],
        r332.score_with_bar(at_bar, exact)["classification"],
        r332.score_with_bar(over, exact)["classification"],
    )
    require(
        got_classes == ("BIT_EXACT", "AT_BAR_NOT_EXACT", "AT_BAR_NOT_EXACT", "OVER_BAR"),
        f"score_with_bar control failed: {got_classes}",
    )

    # 2. 3-D partial-step mask vs 2-D surface-mask control on sub-bottom dry signed zero
    act_v = np.array([[[0.25, -0.0]]], dtype=np.float64)
    exp_v = np.array([[[0.25, 0.0]]], dtype=np.float64)
    masks_2d = {"v": np.array([[[True, True]]], dtype=bool)}
    vmask_3d = np.array([[[True, False]]], dtype=bool)
    if plant == "wet-3d-mask":
        vmask_3d = np.array([[[True, True]]], dtype=bool)
    masks_3d = {
        "v": vmask_3d,
        "ss_v": np.array([[True]], dtype=bool),
    }
    detail = _compute_row_wet_detail("v", act_v, exp_v, masks_2d, masks_3d)
    require(
        detail["wet_bit_unequal"] == 1
        and detail["wet_3d_bit_unequal"] == 0
        and detail["unequal_3d_wet"] == 0
        and detail["unequal_3d_dry"] == 1
        and detail["subbottom_dry_bit_unequal"] == 1
        and detail["wet_3d_classification"] == "BIT_EXACT",
        f"3-D wet mask control failed: {detail}",
    )

    # 3. Single-cell binary64 trace control at (j=18, i=34, k=3, rank=0), kt=10, stage2
    ssh_b = _from_hex64("0xc004dac971407002")
    ssha = _from_hex64("0x3fd4c94d4afbb824")
    r1_ht_0 = _from_hex64("0x3f9642c8590b2164")
    e3t_0_val = _from_hex64("0x402402c768e8a600")
    h_scaled = _from_hex64("0x3f15e5f2c773b397")
    v_scaled = _from_hex64("0xbebdbf795f7c1fc3")
    term1 = _from_hex64("0xbff520edea197e7a")
    rDt = np.float64(5400.0)

    r2_3 = np.float64(2.0) / np.float64(3.0)
    r1_3 = np.float64(1.0) / np.float64(3.0)
    r1_2 = np.float64(0.5)
    r3t_b = ssh_b * r1_ht_0
    r3ta = ssha * r1_ht_0
    r3t_s1 = r2_3 * r3t_b + r1_3 * r3ta
    r3t_rebuilt = (r2_3 * ssh_b + r1_3 * ssha) * r1_ht_0
    if plant == "stage2-thickness":
        r3t_s1 = r3t_rebuilt
    qt_s1 = np.float64(1.0) + r3t_s1
    qt_rebuilt = np.float64(1.0) + r3t_rebuilt
    qt_s2_val = np.float64(1.0) + r1_2 * (r3t_b + r3ta)
    e3t_stored = e3t_0_val * qt_s1
    e3t_rebuilt = e3t_0_val * qt_rebuilt

    rhs_stored = -(h_scaled / e3t_stored) - (v_scaled / e3t_stored)
    rhs_rebuilt = -(h_scaled / e3t_rebuilt) - (v_scaled / e3t_rebuilt)
    dt_qn = rDt * qt_s1
    T_stored = (term1 + dt_qn * rhs_stored) / qt_s2_val
    T_rebuilt = (term1 + dt_qn * rhs_rebuilt) / qt_s2_val
    if plant == "walk-closure":
        T_stored = T_rebuilt

    require(
        _hex64(e3t_stored) == "0x40234d4bd6440687"
        and _hex64(e3t_rebuilt) == "0x40234d4bd6440686"
        and _hex64(rhs_stored) == "0xbee1c450c2a60ef5"
        and _hex64(rhs_rebuilt) == "0xbee1c450c2a60ef6"
        and _hex64(T_stored) == "0xbff663da73f853c6"
        and _hex64(T_rebuilt) == "0xbff663da73f853c7",
        "single-cell Stage-2 e3t(Kmm) hex control failed",
    )

    # 4. Decision-122 classify and caller_audit_compare controls
    base_row_exact = {
        "kt": 1,
        "checkpoint": "stage3",
        "field": "T",
        "bit_identical": True,
        "unequal": 0,
        "max_abs": 0.0,
        "rms": 0.0,
        "classification": "BIT_EXACT",
    }
    base_row_kt10_before = {
        "kt": 10,
        "checkpoint": "stage2",
        "field": "T",
        "bit_identical": False,
        "unequal": 1,
        "max_abs": 2.220446049250313e-16,
        "rms": 6.185950026992371e-19,
        "classification": "AT_BAR_NOT_EXACT",
    }
    base_row_kt10_after = {
        "kt": 10,
        "checkpoint": "stage2",
        "field": "T",
        "bit_identical": True,
        "unequal": 0,
        "max_abs": 0.0,
        "rms": 0.0,
        "classification": "BIT_EXACT",
    }
    cand_sha = "c2451f6309fc0000000000000000000000000000"
    ctrl_same_doc = {
        "status": "PASS_R341_PAIR_RESCORE",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "worktree": {"commit": cand_sha, "clean": True},
        "ladder": {"rows": [base_row_exact, base_row_kt10_before]},
    }
    ctrl_parent_doc = {
        "status": "PASS_R341_PAIR_RESCORE",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "worktree": {"commit": PARENT_COMMIT, "clean": True},
        "ladder": {"rows": [base_row_exact, base_row_kt10_before]},
    }
    cand_doc = {
        "status": "PASS_R341_PAIR_RESCORE",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "worktree": {"commit": cand_sha, "clean": True},
        "ladder": {"rows": [base_row_exact, base_row_kt10_after]},
    }
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        td_path = Path(td)
        p_ctrl = td_path / "ctrl_same.json"
        p_parent = td_path / "ctrl_parent.json"
        p_cand = td_path / "cand.json"
        p_ctrl.write_text(json.dumps(ctrl_same_doc), encoding="utf-8")
        p_parent.write_text(json.dumps(ctrl_parent_doc), encoding="utf-8")
        p_cand.write_text(json.dumps(cand_doc), encoding="utf-8")

        classify_plant = (
            plant if plant in ("exact-loss", "away-row", "parent-commit") else "none"
        )
        res_same = classify(p_ctrl, p_cand, expect_commit=cand_sha)
        res_cross = classify(
            p_parent,
            p_cand,
            allow_cross_tree_parent=True,
            expect_parent_commit=PARENT_COMMIT,
            expect_commit=cand_sha,
            plant=classify_plant,
        )
        require(
            res_same["status"] == "PASS_R341_D122_CENSUS"
            and res_same["d122_c_and_d"]
            and res_cross["status"] == "PASS_R341_D122_CENSUS"
            and res_cross["d122_c_and_d"]
            and res_cross["control_commit"] == PARENT_COMMIT,
            f"classify failed or plant {plant} fired: {res_cross}",
        )

        p_dump = td_path / "parent_dump"
        c_dump = td_path / "cand_dump"
        p_dump.mkdir()
        c_dump.mkdir()
        p_arrays: dict[str, np.ndarray] = {}
        c_arrays: dict[str, np.ndarray] = {}
        p_callers: dict[str, object] = {}
        c_callers: dict[str, object] = {}
        names = ["s1_h", "s2_h", "T_stage2", "s3_h"]
        for cname in CALLER_ORDER:
            p_meta_arrs = {}
            c_meta_arrs = {}
            for aname in names:
                arr_p = np.array([[1.0, 2.0]], dtype=np.float64)
                arr_c = arr_p.copy()
                if cname == "ORCA2-zps" and aname in ("s2_h", "T_stage2"):
                    arr_c[0, 1] = 2.0 + np.spacing(2.0)
                p_arrays[f"{cname}__{aname}"] = arr_p
                c_arrays[f"{cname}__{aname}"] = arr_c
                p_meta_arrs[aname] = {
                    "shape": [1, 2],
                    "elements": 2,
                    "sha256": hashlib.sha256(arr_p.tobytes()).hexdigest(),
                }
                c_meta_arrs[aname] = {
                    "shape": [1, 2],
                    "elements": 2,
                    "sha256": hashlib.sha256(arr_c.tobytes()).hexdigest(),
                }
            common_meta = {
                "rk3_stage2_tracer_nemo_stored_thickness": cname in (
                    "ORCA2V1-tke",
                    "ORCA2-rung0",
                    "ORCA2G5-ice",
                    "ORCA2G6-vec",
                    "ORCA2G7-drg",
                    "ORCA2G8-ldf",
                    "ORCA2-zps_kt1_rest",
                    "ORCA2-zps",
                ),
                "caller_site": "ocean_model_latlon_cgrid.py:9341,9570,9626,9641",
                "array_names": names,
            }
            p_callers[cname] = {**common_meta, "arrays": p_meta_arrs}
            c_callers[cname] = {**common_meta, "arrays": c_meta_arrs}

        np.savez(p_dump / "arrays.npz", **p_arrays)
        np.savez(c_dump / "arrays.npz", **c_arrays)
        (p_dump / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "nemo-testcase-l4-orca2-round341-caller-dump-v1",
                    "status": "PASS_R341_CALLER_DUMP",
                    "worktree": {"commit": PARENT_COMMIT, "clean": True},
                    "caller_order": list(CALLER_ORDER),
                    "callers": p_callers,
                }
            ),
            encoding="utf-8",
        )
        (c_dump / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "nemo-testcase-l4-orca2-round341-caller-dump-v1",
                    "status": "PASS_R341_CALLER_DUMP",
                    "worktree": {"commit": cand_sha, "clean": True},
                    "caller_order": list(CALLER_ORDER),
                    "callers": c_callers,
                }
            ),
            encoding="utf-8",
        )
        caller_plant = (
            plant if plant in ("caller-unequal", "parent-commit") else "none"
        )
        caller_res = caller_audit_compare(
            p_dump,
            c_dump,
            expect_parent_commit=PARENT_COMMIT,
            expect_commit=cand_sha,
            plant=caller_plant,
        )

    return {
        "format": "nemo-testcase-l4-orca2-round341-selftest-v1",
        "status": "PASS_R341_SELFTEST",
        "score_classes": got_classes,
        "e3t_stored_hex": _hex64(e3t_stored),
        "e3t_rebuilt_hex": _hex64(e3t_rebuilt),
        "T_stage2_stored_hex": _hex64(T_stored),
        "T_stage2_rebuilt_hex": _hex64(T_rebuilt),
        "census": res_cross,
        "caller_audit": {
            "status": caller_res["status"],
            "parent_commit": caller_res["parent_commit"],
            "worktree_commit": caller_res["worktree_commit"],
            "orca2_unequal_words": caller_res["callers"]["ORCA2-zps"][
                "total_unequal_words"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung", choices=tuple(sorted({p[0] for p in PAIRS})))
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--arm", choices=("control", "candidate"), default="candidate")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--expect-parent-commit", default=PARENT_COMMIT)
    parser.add_argument("--aggregate", nargs="+", type=Path)
    parser.add_argument("--table-out", type=Path)
    parser.add_argument("--rung10-table-out", type=Path)
    parser.add_argument("--walk", action="store_true")
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--r326-root", type=Path)
    parser.add_argument("--r332-v-root", type=Path)
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--allow-cross-tree-parent", action="store_true")
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--caller-audit-dump", type=Path)
    parser.add_argument("--caller-audit", action="store_true")
    parser.add_argument("--parent-dump", type=Path)
    parser.add_argument("--candidate-dump", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", choices=PLANT_CHOICES, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        if args.measure:
            require(
                all(
                    (
                        args.rung,
                        args.label,
                        args.deck_root,
                        args.twin_a,
                        args.expect_commit,
                    )
                ),
                "measurement arguments incomplete",
            )
            result = measure(
                args.rung,
                args.label,
                args.deck_root,
                args.twin_a,
                args.expect_commit,
                arm=args.arm,
            )
        elif args.aggregate:
            result = aggregate(
                args.aggregate,
                args.table_out,
                args.rung10_table_out,
                expect_commit=args.expect_commit,
            )
        elif args.walk:
            require(
                all((args.record_root, args.r326_root)),
                "walk arguments incomplete",
            )
            result = walk_source_order(
                args.record_root,
                args.r326_root,
                args.r332_v_root,
                plant=args.plant,
            )
        elif args.classify:
            result = classify(
                *args.classify,
                allow_cross_tree_parent=args.allow_cross_tree_parent,
                expect_parent_commit=args.expect_parent_commit,
                expect_commit=args.expect_commit,
                plant=args.plant,
            )
            require(
                result["status"] == "PASS_R341_D122_CENSUS",
                f"census failed: {result}",
            )
        elif args.decide:
            result = decide(
                args.decide,
                allow_cross_tree_parent=args.allow_cross_tree_parent,
                expect_parent_commit=args.expect_parent_commit,
                expect_commit=args.expect_commit,
            )
        elif args.caller_audit_dump:
            require(
                bool(args.deck_root and args.expect_commit),
                "--caller-audit-dump requires --deck-root and --expect-commit",
            )
            result = caller_audit_dump(
                args.caller_audit_dump, args.deck_root, args.expect_commit
            )
        elif args.caller_audit:
            require(
                bool(args.parent_dump and args.candidate_dump),
                "--caller-audit requires --parent-dump and --candidate-dump",
            )
            result = caller_audit_compare(
                args.parent_dump,
                args.candidate_dump,
                expect_parent_commit=args.expect_parent_commit,
                expect_commit=args.expect_commit,
                plant=args.plant,
            )
        elif args.self_test:
            result = self_test(plant=args.plant)
        else:
            raise GateError("no operation selected")
        require(args.plant == "none", f"plant {args.plant} stayed green")
    except (
        GateError,
        r332.GateError,
        r333_wet.GateError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        StopIteration,
    ) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
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

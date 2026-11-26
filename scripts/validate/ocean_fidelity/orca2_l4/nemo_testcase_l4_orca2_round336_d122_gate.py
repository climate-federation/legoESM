#!/usr/bin/env python3
"""Round 336 Resume 19: Decision-122, rung-10 wet-value rescore, parent-SHA
provenance enforcement, and cross-tree caller output comparator for the
north-fold V-face mask restoration in ``_tke_bottom_dirichlet``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

PARENT_COMMIT = "254fec16a822b9548fdd47e196611027ce9ac37d"
PLANTED_WRONG_PARENT_COMMIT = "fb1a92720a71892d8549a61efa887ea926186bfa"
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
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
    nemo_testcase_l4_orca2_round332_rescore_gate as r332,
    nemo_testcase_l4_orca2_round333_first_wet_gate as r333_wet,
)

PAIRS = (
    ("rung0", "independent"),
    ("omt1", "independent"),
    ("omt2", "independent"),
    ("omt3", "independent"),
    ("omt4", "independent"),
    ("rung10", "given_nemo_entry"),
    ("rung10", "independent"),
)

CALLER_ORDER = (
    "non_nemo_folded",
    "non_nemo_unfolded",
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

GRIDLADDER_LANE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/lanes/gridladder")
GRIDLADDER_RECIPE_COMMIT = "242eaacc47"
GRIDLADDER_VMIX_LANE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/lanes/gridladder_vmix"
)
GRIDLADDER_VMIX_RECIPE_COMMIT = "0b9443cef2"
GRIDLADDER_ROUNDS = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds")
GRIDLADDER_VMIX_ROUNDS = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds"
)


class GateError(RuntimeError):
    """The same-tree arm, wet-value audit, caller audit, or D122 check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _strip_fold_top_v_in_bottom_dirichlet(cfg):
    vmix = getattr(getattr(cfg, "physics", None), "vertical_mixing", None)
    if vmix is None or vmix.scheme != "tke":
        return cfg, False
    if not bool(getattr(vmix.tke, "bottom_tke_bc", False)):
        return cfg, False
    if not bool(getattr(vmix.tke, "nemo_sh2_fold_top_v", False)):
        return cfg, True
    new_tke = vmix.tke._replace(nemo_sh2_fold_top_v=False)
    new_vmix = vmix._replace(tke=new_tke)
    new_phys = cfg.physics._replace(vertical_mixing=new_vmix)
    if hasattr(cfg, "_replace"):
        return cfg._replace(physics=new_phys), True
    return SimpleNamespace(**{**cfg.__dict__, "physics": new_phys}), True


def _wrap_bottom_dirichlet(original, counter: list[int], *, strip_flag: bool):
    def wrapped(
        self,
        state,
        z_coord=None,
        config=None,
        grid=None,
        *,
        u_now=None,
        v_now=None,
    ):
        cfg_b = self.config if config is None else config
        cfg_eff, active = _strip_fold_top_v_in_bottom_dirichlet(cfg_b)
        if active:
            counter[0] += 1
        use_cfg = cfg_eff if (strip_flag and active) else config
        return original(
            self,
            state,
            z_coord=z_coord,
            config=use_cfg,
            grid=grid,
            u_now=u_now,
            v_now=v_now,
        )

    return wrapped


def _compute_wet_row_detail(
    field: str,
    actual: np.ndarray,
    expected: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, object]:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    full_wet = masks[field]
    wet = full_wet if act.shape[1] == full_wet.shape[1] else full_wet[:, :90]
    require(
        wet.shape == act.shape == exp.shape,
        f"wet mask shape {wet.shape} != field {field} shape {act.shape}",
    )
    bit_diff = act.view(np.uint64) != exp.view(np.uint64)
    val_diff = act != exp
    wet_bit_diff = bit_diff & wet
    wet_val_diff = val_diff & wet
    dry_bit_diff = bit_diff & (~wet)

    wet_bit_unequal = int(np.count_nonzero(wet_bit_diff))
    unequal_wet = int(np.count_nonzero(wet_val_diff))
    signed_zero_only_wet = int(np.count_nonzero(wet_bit_diff & (~val_diff)))
    unequal_dry = int(np.count_nonzero(dry_bit_diff))
    wet_count = int(np.count_nonzero(wet))

    residual = act - exp
    wet_values = residual[wet]
    wet_rms = float(np.sqrt(np.mean(wet_values * wet_values))) if wet_count else 0.0
    wet_max_abs = float(np.max(np.abs(wet_values))) if unequal_wet > 0 else 0.0
    oracle_wet_max = float(np.max(np.abs(exp[wet]))) if wet_count else 0.0
    wet_scale = max(oracle_wet_max, 1.0)
    wet_norm = wet_max_abs / wet_scale
    if wet_bit_unequal == 0:
        wet_class = "BIT_EXACT"
    elif wet_norm <= r332.BAR:
        wet_class = "AT_BAR_NOT_EXACT"
    else:
        wet_class = "OVER_BAR"

    max_index = None
    max_ties = 0
    if unequal_wet > 0:
        abs_res = np.abs(residual)
        max_mask = wet_val_diff & (abs_res == wet_max_abs)
        maxima = np.argwhere(max_mask)
        require(maxima.size > 0, "maximum wet-value index is absent")
        max_index = tuple(int(v) for v in maxima[0])
        max_ties = int(maxima.shape[0])

    return {
        "wet_count": wet_count,
        "wet_bit_identical": wet_bit_unequal == 0,
        "wet_bit_unequal": wet_bit_unequal,
        "unequal_wet": unequal_wet,
        "signed_zero_only_wet": signed_zero_only_wet,
        "unequal_dry": unequal_dry,
        "wet_max_abs": wet_max_abs,
        "wet_rms": wet_rms,
        "oracle_wet_max_abs": oracle_wet_max,
        "wet_normalization_scale": wet_scale,
        "wet_normalized_max_abs": wet_norm,
        "wet_classification": wet_class,
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
        detail = row.pop("_round336_wet")
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
                "unequal": int(row["unequal"]),
                "max_abs": float(row["max_abs"]),
                "rms": float(row["rms"]),
                "classification": str(row["classification"]),
                **detail,
                **loc_info,
            }
        )

    wet_counts = Counter(r["wet_classification"] for r in wet_rows)
    first_wet_non_bit = next(
        (r for r in wet_rows if not r["wet_bit_identical"]), None
    )
    first_wet_value = next(
        (r for r in wet_rows if int(r["unequal_wet"]) > 0), None
    )
    first_wet_over_bar = next(
        (r for r in wet_rows if r["wet_classification"] == "OVER_BAR"), None
    )
    first_raw_over_bar = next(
        (r for r in wet_rows if r["classification"] == "OVER_BAR"), None
    )
    by_key = {(r["kt"], r["checkpoint"], r["field"]): r for r in wet_rows}
    return {
        "row_count": len(wet_rows),
        "wet_classification_counts": {
            name: int(wet_counts[name]) for name in r332.CLASSES
        },
        "first_wet_non_bit_row": first_wet_non_bit,
        "first_wet_value_row": first_wet_value,
        "first_wet_over_bar_row": first_wet_over_bar,
        "first_over_bar_row": first_raw_over_bar,
        "kt2_stage3_T": by_key.get((2, "stage3", "T")),
        "kt2_stage3_S": by_key.get((2, "stage3", "S")),
        "rows": wet_rows,
    }


def measure(
    rung: str,
    label: str,
    arm: str,
    deck_root: Path,
    twin_a: Path,
    expect_commit: str,
) -> dict[str, object]:
    require((rung, label) in PAIRS, f"pair {(rung, label)} is outside Decision 122")
    require(arm in ("control", "candidate"), f"unknown arm {arm}")
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    model_path = Path(model_module.__file__).resolve()
    r332_path = Path(r332.__file__).resolve()
    require(
        model_path.is_relative_to(ACTIVE_REPO_ROOT)
        and r332_path.is_relative_to(ACTIVE_REPO_ROOT),
        f"imported modules {model_path} / {r332_path} are outside {ACTIVE_REPO_ROOT}",
    )

    calls = [0]
    cls = model_module.LatLonCGridOceanModel
    original_bd = cls._tke_bottom_dirichlet
    cls._tke_bottom_dirichlet = _wrap_bottom_dirichlet(
        original_bd, calls, strip_flag=(arm == "control")
    )

    masks = None
    card_ref = None
    ice_by_kt = None
    original_compare = rung10.compare_fields

    if rung == "rung10":
        _, shipped = rung10.card_fields(deck_root)
        card_ref = r329.selected_card(shipped, arm="recordice", rung="rung10")
        masks = r333_wet._native_masks(card_ref)
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
                out["rows"][name]["_round336_wet"] = _compute_wet_row_detail(
                    name, actual[name], expected[name], masks
                )
            return out

        rung10.compare_fields = wrapped_compare

    try:
        result = r332.measure(rung, label, deck_root, twin_a, expect_commit)
    finally:
        cls._tke_bottom_dirichlet = original_bd
        rung10.compare_fields = original_compare

    require(
        (calls[0] > 0) == (rung == "rung10"),
        f"bottom_dirichlet call execution count moved for {rung}: {calls[0]}",
    )
    require(
        bool(result.get("worktree", {}).get("clean"))
        and str(result["worktree"]["commit"]).lower() == expect_commit.lower(),
        f"measured worktree {result.get('worktree')} != clean {expect_commit}",
    )
    if rung == "rung10":
        result["wet_audit"] = _finalize_wet_audit(
            result["ladder"]["rows"], card_ref, ice_by_kt
        )
    result.update(
        {
            "format": "nemo-testcase-l4-orca2-round336-d122-arm-v1",
            "status": "PASS_R336_D122_ARM",
            "arm": arm,
            "bottom_dirichlet_calls": calls[0],
        }
    )
    return result


def _row(rows: list[dict[str, object]], key: tuple[int, str, str]) -> dict[str, object]:
    return next(
        row
        for row in rows
        if (int(row["kt"]), str(row["checkpoint"]), str(row["field"])) == key
    )


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
    require(
        candidate.get("status") == "PASS_R336_D122_ARM",
        f"candidate arm is not admitted: {candidate.get('status')}",
    )
    require(
        bool(control.get("worktree", {}).get("clean")) is True,
        "control worktree is not clean",
    )
    require(
        bool(candidate.get("worktree", {}).get("clean")) is True,
        "candidate worktree is not clean",
    )
    if expect_commit is not None:
        require(
            str(candidate["worktree"]["commit"]).lower() == expect_commit.lower(),
            f"candidate commit {candidate['worktree']['commit']} != {expect_commit}",
        )
    if allow_cross_tree_parent:
        require(
            control.get("status") == "PASS_R336_D122_ARM",
            f"parent arm is not admitted: {control.get('status')}",
        )
        require(
            str(control["worktree"]["commit"]).lower()
            == expect_parent_commit.lower(),
            f"parent commit {control['worktree']['commit']} != expected parent {expect_parent_commit}",
        )
        require(
            str(candidate["worktree"]["commit"]).lower()
            != str(control["worktree"]["commit"]).lower(),
            "cross-tree parent and candidate commits are identical",
        )
    else:
        require(
            control.get("status") == "PASS_R336_D122_ARM",
            f"control arm is not admitted: {control.get('status')}",
        )
        same_nonexecuting = (
            control["arm"] == candidate["arm"] == "candidate"
            and control["rung"] != "rung10"
            and control["bottom_dirichlet_calls"]
            == candidate["bottom_dirichlet_calls"]
            == 0
        )
        require(
            (control["arm"] == "control" and candidate["arm"] == "candidate")
            or same_nonexecuting,
            "arm order moved",
        )
        require(
            control["worktree"]["commit"] == candidate["worktree"]["commit"],
            "arms are not from one committed tree",
        )
    require(
        (control["rung"], control["label"]) == (candidate["rung"], candidate["label"]),
        "pair moved",
    )
    before, after = control["ladder"]["rows"], candidate["ladder"]["rows"]
    require(len(before) == len(after), "row count moved")
    if plant == "exact-loss":
        after = [dict(r) for r in after]
        for r in after:
            if r["bit_identical"]:
                r["bit_identical"] = False
                r["unequal"] = 1
                r["max_abs"] = 1.0e-12
                r["rms"] = 1.0e-14
                break
    elif plant == "away-row":
        after = [dict(r) for r in after]
        for r in after:
            if not r["bit_identical"]:
                r["rms"] = float(r["rms"]) * 2.0 + 1.0e-12
                r["max_abs"] = float(r["max_abs"]) * 2.0 + 1.0e-12
                break
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
    old_first = next(row for row in before if row["classification"] == "OVER_BAR")
    target = (int(old_first["kt"]), str(old_first["checkpoint"]), str(old_first["field"]))
    new_at_old_first = _row(after, target)
    first_direction = r276.decision96._direction(old_first, new_at_old_first, "rms")
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
        "format": "nemo-testcase-l4-orca2-round336-d122-census-v1",
        "status": "PASS_R336_D122_CENSUS" if passed else "FAIL_R336_D122_CENSUS",
        "rung": candidate["rung"],
        "label": candidate["label"],
        "control_commit": control["worktree"]["commit"],
        "worktree_commit": candidate["worktree"]["commit"],
        "row_count": len(after),
        "exact_row_losses": exact_losses,
        "unequal_increases": unequal_increases,
        "first_over_bar": {
            "old_target": list(target),
            "new_first_over_bar": new_first_over,
            "rms_direction": first_direction,
            "old_unequal": int(old_first["unequal"]),
            "new_unequal": int(new_at_old_first["unequal"]),
            "old_max_abs": float(old_first["max_abs"]),
            "new_max_abs": float(new_at_old_first["max_abs"]),
            "old_rms": float(old_first["rms"]),
            "new_rms": float(new_at_old_first["rms"]),
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
        "seven-pair set is incomplete",
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
        require(
            str(docs[0]["control_commit"]).lower() == expect_parent_commit.lower(),
            f"decision control_commit {docs[0]['control_commit']} != expected parent {expect_parent_commit}",
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
            doc["status"] == "PASS_R336_D122_CENSUS" and doc["d122_c_and_d"]
            for doc in docs
        ),
        "a pair failed Decision 122",
    )
    return {
        "format": "nemo-testcase-l4-orca2-round336-d122-decision-v1",
        "status": "PASS_R336_D122_DECISION",
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
    extra = [p for p in sys.path if p not in saved_path and not p.startswith(str(GRIDLADDER_VMIX_LANE / "packages"))]
    sys.path = [p for p in saved_path if p != str(GRIDLADDER_VMIX_LANE)] + [
        p for p in extra if p != str(GRIDLADDER_VMIX_LANE)
    ]
    return mod


def _non_nemo_staircase_caller(*, fold: bool) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        TKEConfig,
        VerticalMixingConfig,
    )
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    ny, nx, nz_bot = 4, 6, 4
    perm = np.array([(nx - i) % nx for i in range(nx)], dtype=np.int32)
    rng = np.random.default_rng(19)

    def _sym(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[-1] = 0.5 * (out[-1] + out[-1, perm])
        out[-1, 0] = out[-1, nx // 2] = 0.0
        return out

    one = np.ones((ny, nx))
    raw = create_z_star_from_thicknesses(
        [1.0, 2.0, 3.0, 4.0],
        nemo_gdept_0_m=np.broadcast_to([0.5, 2.0, 4.5, 8.0], (ny, nx, nz_bot)).copy(),
        nemo_gdepw_0_m=np.broadcast_to([0.0, 1.0, 3.0, 6.0], (ny, nx, nz_bot)).copy(),
        nemo_e3t_0_m=np.broadcast_to([1.0, 2.0, 3.0, 4.0], (ny, nx, nz_bot)).copy(),
        nemo_e3w_0_m=np.broadcast_to([1.0, 1.5, 2.5, 3.5], (ny, nx, nz_bot)).copy(),
        nemo_hu_0_m=6.0 * one,
        nemo_hv_0_m=6.0 * one,
        nemo_e1e2t_m=one,
        nemo_e1e2u_m=one,
        nemo_e1e2v_m=one,
    )
    base_z = create_partial_cell_coordinate(raw, jnp.full((ny, nx), 6.0))
    raw_vmask = np.zeros((ny, nx, nz_bot), dtype=np.float64)
    raw_vmask[..., :3] = 1.0
    z = base_z._replace(
        nemo_een_barotropic=SimpleNamespace(vmask=jnp.asarray(raw_vmask))
    )

    eta = jnp.asarray(_sym(rng.uniform(-0.05, 0.05, (ny, nx))), dtype=jnp.float64)
    avm = jnp.asarray(
        _sym(rng.uniform(0.05, 0.2, (ny, nx, nz_bot - 1))), dtype=jnp.float64
    )
    tke0 = jnp.asarray(
        _sym(rng.uniform(1.0e-4, 5.0e-4, (ny, nx, nz_bot - 1))), dtype=jnp.float64
    )
    u = np.zeros((ny, nx + 1, nz_bot), dtype=np.float64)
    v = np.zeros((ny + 1, nx, nz_bot), dtype=np.float64)
    v_row = np.array([0.8, -0.3, 0.5, -0.1, 0.4, -0.6], dtype=np.float64)
    v[-2, :, :] = v_row[:, None]
    v[-1, :, :] = -v_row[perm, None]
    u_in = jnp.asarray(u, dtype=jnp.float64)
    v_in = jnp.asarray(v, dtype=jnp.float64)

    d3, d2 = ("lat", "lon", "level"), ("lat", "lon")
    tke_cfg = TKEConfig(
        prognostic=True,
        bottom_tke_bc=True,
        tke_background=1.0e-14,
        mxl_min=1.0e-3,
        kappaM_min=1.0e-12,
        kappaH_min=1.0e-12,
        tke_preclosure_coeff_source="carried_previous_step",
        tke_shear_evaluation_stage="step_entry",
        tke_shear_production="nemo_face_native_nbb2",
        tke_shear_avm_weighting="nemo_face",
        tke_shear_metric_source="nemo_qco_live_face",
        nemo_sh2_fold_top_v=False,
    )
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=tke_cfg),
        convection=OceanConvectionConfig(scheme="none"),
    )
    bottom_drag = SimpleNamespace(
        bottom_drag_scheme="nemo_linear",
        bottom_drag_cd0=1.0e-3,
        bottom_drag_cdmax=0.1,
        bottom_drag_z0=3.0e-3,
        bottom_drag_ke0=2.5e-3,
        bottom_drag_uc0=0.4,
    )
    cfg = SimpleNamespace(
        physics=physics,
        bottom_drag=bottom_drag,
        eos="linear",
        eos_linear=None,
        eos_nemo_seos=None,
        implicit_ice_ocean_top_drag=False,
        zdf_implicit_solver_evaluation="factored",
        zdf_baroclinic_only=False,
        zdf_drag_in_matrix=False,
        A_v=1.0e-4,
        K_v=1.0e-5,
    )
    grid = SimpleNamespace(
        lat_T=jnp.zeros((ny, nx), dtype=jnp.float64),
        dx_T=jnp.ones((ny, nx), dtype=jnp.float64),
        dy_T=jnp.ones((ny, nx), dtype=jnp.float64),
        area_T=jnp.ones((ny, nx), dtype=jnp.float64),
        dx_U=jnp.ones((ny, nx + 1), dtype=jnp.float64),
        dy_U=jnp.ones((ny, nx + 1), dtype=jnp.float64),
        area_U=jnp.ones((ny, nx + 1), dtype=jnp.float64),
        dx_V=jnp.ones((ny + 1, nx), dtype=jnp.float64),
        dy_V=jnp.ones((ny + 1, nx), dtype=jnp.float64),
        area_V=jnp.ones((ny + 1, nx), dtype=jnp.float64),
        fold=SimpleNamespace(is_active=True, perm_T=jnp.asarray(perm))
        if fold
        else None,
    )
    model = SimpleNamespace(
        config=cfg,
        z_coord=z,
        grid=grid,
        _iwm_forcing=None,
        _nemo_ws_test_hooks=SimpleNamespace(
            zdf_momentum_observer=None,
            tke_rhs_materialization="",
            tke_rhs_intermediate="",
        ),
        _tke_post_mixing_active=lambda: False,
        _tke_realized_kdiss_active=lambda: False,
        _lat_dependent_constant_vmix=lambda: False,
    )
    for name in (
        "_tke_bottom_dirichlet",
        "_tke_bottom_level",
        "_tke_step_entry_p_sh2",
    ):
        setattr(model, name, MethodType(getattr(LatLonCGridOceanModel, name), model))

    class _StateNS(SimpleNamespace):
        def _replace(self, **kwargs):
            return _StateNS(**{**self.__dict__, **kwargs})

    T_arr = jnp.broadcast_to(
        jnp.asarray([15.0, 10.0, 6.0, 2.0], dtype=jnp.float64), (ny, nx, nz_bot)
    )
    S_arr = jnp.full((ny, nx, nz_bot), 35.0, dtype=jnp.float64)
    state = _StateNS(
        T=Field(T_arr, "T", d3, "degC"),
        S=Field(S_arr, "S", d3, "psu"),
        H_bathy=Field(jnp.full((ny, nx), 6.0, dtype=jnp.float64), "H_bathy", d2, "m"),
        land_mask=Field(jnp.ones((ny, nx), dtype=jnp.float64), "land_mask", d2, "1"),
        eta=Field(eta, "eta", d2, "m"),
        u=Field(u_in, "u", d3, "m/s"),
        v=Field(v_in, "v", d3, "m/s"),
        tke=Field(tke0, "tke", d3, "m2/s2"),
        tke_avm=Field(avm, "tke_avm", d3, "m2/s"),
        tke_avt=Field(avm, "tke_avt", d3, "m2/s"),
        tke_avm_surface=None,
        tke_dissl=None,
    )
    bot = np.asarray(model._tke_bottom_dirichlet(state, u_now=u_in, v_now=v_in))
    kv, av = LatLonCGridOceanModel._apply_implicit_vertical_mixing(
        model,
        state,
        10800.0,
        None,
        tke_old=tke0,
        u_now=u_in,
        v_now=v_in,
        eta_now=eta,
        eta_step_entry=eta,
        return_K_profiles=True,
    )
    arrays = {
        "bot_dirichlet": np.asarray(bot, dtype=np.float64),
        "K_v": np.asarray(kv, dtype=np.float64),
        "A_v": np.asarray(av, dtype=np.float64),
    }
    meta = {
        "vmix_scheme": "tke",
        "bottom_tke_bc": True,
        "nemo_sh2_fold_top_v": False,
        "fold": fold,
        "bottom_dirichlet_is_none": False,
    }
    return arrays, meta


def _eval_model_caller(
    name: str,
    model,
    state,
    dt: float,
    forcing,
    *,
    u_val: float,
    v_val: float,
    avm_override: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import jax.numpy as jnp
    from legoesm.core.field import Field

    vmix = getattr(
        getattr(model.config, "physics", None), "vertical_mixing", None
    )
    scheme = getattr(vmix, "scheme", "none") if vmix is not None else "none"
    tke_cfg = getattr(vmix, "tke", None) if vmix is not None else None
    bot_flag = bool(getattr(tke_cfg, "bottom_tke_bc", False)) and scheme == "tke"
    fold_flag = bool(getattr(tke_cfg, "nemo_sh2_fold_top_v", False)) and scheme == "tke"

    st = model._seed_tke_preclosure_carry(state) if scheme == "tke" else state
    u_nz = jnp.full_like(st.u.data, u_val, dtype=jnp.float64)
    v_nz = jnp.full_like(st.v.data, v_val, dtype=jnp.float64)
    u_f = Field(u_nz, st.u.name, st.u.dims, st.u.units)
    v_f = Field(v_nz, st.v.name, st.v.dims, st.v.units)
    repl = {"u": u_f, "v": v_f}
    if hasattr(st, "u_before") and st.u_before is not None:
        repl["u_before"] = u_f
    if hasattr(st, "v_before") and st.v_before is not None:
        repl["v_before"] = v_f
    if avm_override is not None and getattr(st, "tke_avm", None) is not None:
        avm_arr = jnp.full_like(st.tke_avm.data, avm_override, dtype=jnp.float64)
        repl["tke_avm"] = Field(
            avm_arr, st.tke_avm.name, st.tke_avm.dims, st.tke_avm.units
        )
    st = st._replace(**repl)

    bot = model._tke_bottom_dirichlet(st, u_now=u_nz, v_now=v_nz)
    require(
        (bot is None) == (not bot_flag),
        f"{name}: _tke_bottom_dirichlet None={bot is None} vs bottom_tke_bc={bot_flag}",
    )

    n2_tr = model._n2_before_advection_tracers(st)
    n2_tr_b = model._n2_nemo_before_tracers(st)
    n2_bundle = model._tke_step_entry_n2_bundle(st) if scheme == "tke" else None
    tke_old = (
        st.tke.data
        if (scheme == "tke" and getattr(st, "tke", None) is not None)
        else None
    )

    kv, av = model._apply_implicit_vertical_mixing(
        st,
        dt,
        forcing,
        tke_old=tke_old,
        u_now=u_nz,
        v_now=v_nz,
        eta_now=st.eta.data,
        eta_step_entry=st.eta.data,
        n2_tracers=n2_tr,
        n2_tracers_before=n2_tr_b,
        tke_n2_bundle=n2_bundle,
        return_K_profiles=True,
    )
    vmix_out = model._apply_implicit_vertical_mixing(
        st,
        dt,
        forcing,
        tke_old=tke_old,
        u_now=u_nz,
        v_now=v_nz,
        eta_now=st.eta.data,
        eta_step_entry=st.eta.data,
        n2_tracers=n2_tr,
        n2_tracers_before=n2_tr_b,
        tke_n2_bundle=n2_bundle,
        return_tke=(scheme == "tke"),
    )
    if scheme == "tke":
        state_out, tke_new = vmix_out
    else:
        state_out, tke_new = vmix_out, None

    arrays: dict[str, np.ndarray] = {}
    if bot is not None:
        arrays["bot_dirichlet"] = np.asarray(bot, dtype=np.float64)
    arrays["K_v"] = np.asarray(kv, dtype=np.float64)
    arrays["A_v"] = np.asarray(av, dtype=np.float64)
    arrays["u"] = np.asarray(state_out.u.data, dtype=np.float64)
    arrays["v"] = np.asarray(state_out.v.data, dtype=np.float64)
    arrays["T"] = np.asarray(state_out.T.data, dtype=np.float64)
    arrays["S"] = np.asarray(state_out.S.data, dtype=np.float64)
    if tke_new is not None:
        arrays["tke"] = np.asarray(tke_new, dtype=np.float64)
    for attr in ("tke_avm", "tke_avt", "tke_avm_surface", "tke_dissl"):
        val = getattr(state_out, attr, None)
        if val is not None and getattr(val, "data", None) is not None:
            arrays[attr] = np.asarray(val.data, dtype=np.float64)

    meta = {
        "vmix_scheme": scheme,
        "bottom_tke_bc": bot_flag,
        "nemo_sh2_fold_top_v": fold_flag,
        "bottom_dirichlet_is_none": bot is None,
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
    g4_forcing_mod = _load_file_module(
        GRIDLADDER_VMIX_LANE
        / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_g4_orca2sbc/g4_forcing_steps.py",
        "g4_forcing_steps",
    )
    g5_records_mod = _load_file_module(
        GRIDLADDER_VMIX_LANE
        / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_g5_orca2ice/g5_records.py",
        "g5_records",
    )
    if "rCdU_ice" not in g5_records_mod.ROUTE_FIELDS:
        g5_records_mod.ROUTE_FIELDS = (*g5_records_mod.ROUTE_FIELDS, "rCdU_ice")

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

    # 1-2. Non-NEMO prognostic TKE callers (folded and unfolded)
    arr_f, meta_f = _non_nemo_staircase_caller(fold=True)
    _record_caller("non_nemo_folded", arr_f, meta_f)
    arr_nf, meta_nf = _non_nemo_staircase_caller(fold=False)
    _record_caller("non_nemo_unfolded", arr_nf, meta_nf)

    # 3. Faithful DINO (bottom_tke_bc=True, nemo_sh2_fold_top_v=False)
    dino_cfg = nemo_faithful_dino_config(dino_config_for_recipe("nemo_dino_kamm_mlf"))
    dino_grid = dino_lat_lon_grid(dino_cfg, n_lon=50)
    dino_z = dino_lat_lon_vertical(dino_grid, dino_cfg)
    dino_state0 = dino_lat_lon_state(dino_grid, dino_z, dino_cfg)
    dino_mcfg, _ = dino_lat_lon_model_config(dino_grid, dino_cfg)
    dino_model = cls(dino_grid, dino_z, dino_mcfg)
    ny_d, nx_d, nlev_d = dino_state0.T.data.shape
    tke_init_d = jnp.full((ny_d, nx_d, nlev_d - 1), 1.0e-4, dtype=jnp.float64)
    dino_state = dino_state0._replace(
        tke=Field(tke_init_d, "tke", ("lat", "lon", "level"), "m2/s2"),
        T_before=dino_state0.T,
        S_before=dino_state0.S,
        u_before=dino_state0.u,
        v_before=dino_state0.v,
        eta_before=dino_state0.eta,
    )
    arr_dino, meta_dino = _eval_model_caller(
        "faithful_dino",
        dino_model,
        dino_state,
        2700.0,
        None,
        u_val=0.08,
        v_val=-0.06,
    )
    require(
        meta_dino["bottom_tke_bc"] and not meta_dino["nemo_sh2_fold_top_v"],
        f"faithful_dino flags unexpected: {meta_dino}",
    )
    _record_caller("faithful_dino", arr_dino, meta_dino)

    # 4. ORCA2V1-tke (scheme="tke", bottom_tke_bc=False, nemo_sh2_fold_top_v=True)
    v1_deck = GRIDLADDER_VMIX_ROUNDS / "round1/oracle_v1_r1/twin_a"
    v1_card = v_recipe.build_nemo_testcase_card("ORCA2V1-tke", deck_root=v1_deck)
    _, v1_forcing = g5_records_mod.nemo_supplied_forcing(v1_card, v1_deck, 1)
    v1_forcing = v1_forcing._replace(
        ice_mean_thickness_hm_i=g5_records_mod.read_v1_hm_i(v1_deck, 1)
    )
    v1_model = cls(
        v1_card.recipe.grid,
        v1_card.recipe.z_coord,
        v1_card.recipe.model_config,
        iwm_forcing=v1_card.recipe.iwm_forcing,
    )
    arr_v1, meta_v1 = _eval_model_caller(
        "ORCA2V1-tke",
        v1_model,
        v1_card.recipe.initial_state,
        float(v1_card.dt_s),
        v1_forcing,
        u_val=0.08,
        v_val=-0.06,
    )
    require(
        meta_v1["vmix_scheme"] == "tke"
        and not meta_v1["bottom_tke_bc"]
        and meta_v1["nemo_sh2_fold_top_v"],
        f"ORCA2V1-tke flags unexpected: {meta_v1}",
    )
    _record_caller("ORCA2V1-tke", arr_v1, meta_v1)

    # 5-9. Standard non-ORCA2 cards
    std_cards = (
        ("GYRE-zco", build_gyre_zco_card()),
        ("VORTEX-zco", build_vortex_zco_card()),
        ("LOCK_EXCHANGE-zco", build_lock_exchange_zco_card()),
        ("OVERFLOW-zps", build_overflow_zps_card()),
        ("ORCA2-rung0", rung0.build_rung0_card(deck_root)),
    )
    for cname, card in std_cards:
        cmodel = cls(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            iwm_forcing=card.recipe.iwm_forcing,
        )
        c_arr, c_meta = _eval_model_caller(
            cname,
            cmodel,
            card.recipe.initial_state,
            float(card.dt_s),
            None,
            u_val=0.08,
            v_val=-0.06,
        )
        require(
            not c_meta["bottom_tke_bc"] and not c_meta["nemo_sh2_fold_top_v"],
            f"{cname} flags unexpected: {c_meta}",
        )
        _record_caller(cname, c_arr, c_meta)

    # 10-18. ORCA2G1..G8 gridladder cards
    g_decks = (
        ("ORCA2G1-flat", GRIDLADDER_ROUNDS / "round4/oracle_g1_r4/p3_a_100step"),
        ("ORCA2G1b-vec", GRIDLADDER_ROUNDS / "round22/oracle_g1b_r22/ref_100step"),
        ("ORCA2G2-zps", GRIDLADDER_ROUNDS / "round10/oracle_g2_r10/p3_a_100step"),
        ("ORCA2G3-ts", GRIDLADDER_ROUNDS / "round21/oracle_g3_r21/p3_a"),
        ("ORCA2G4-sbc", GRIDLADDER_ROUNDS / "round21/oracle_g4_r21/p6_a"),
        ("ORCA2G5-ice", GRIDLADDER_ROUNDS / "round31/oracle_g5_r31/twin_a"),
        ("ORCA2G6-vec", GRIDLADDER_ROUNDS / "round36/oracle_g6_r36/twin_a"),
        ("ORCA2G7-drg", GRIDLADDER_ROUNDS / "round41/oracle_g7_r41/twin_a"),
        ("ORCA2G8-ldf", GRIDLADDER_ROUNDS / "round45/oracle_g8_r45/twin_a"),
    )
    for gname, gdeck in g_decks:
        gcard = g_recipe.build_nemo_testcase_card(gname, deck_root=gdeck)
        if gname == "ORCA2G4-sbc":
            _, gforcing = g4_forcing_mod.nemo_supplied_forcing(gcard, gdeck, 1)
        elif gname in ("ORCA2G5-ice", "ORCA2G6-vec", "ORCA2G7-drg", "ORCA2G8-ldf"):
            _, gforcing = g5_records_mod.nemo_supplied_forcing(gcard, gdeck, 1)
        else:
            gforcing = None
        gmodel = cls(
            gcard.recipe.grid,
            gcard.recipe.z_coord,
            gcard.recipe.model_config,
            iwm_forcing=gcard.recipe.iwm_forcing,
        )
        g_arr, g_meta = _eval_model_caller(
            gname,
            gmodel,
            gcard.recipe.initial_state,
            float(gcard.dt_s),
            gforcing,
            u_val=0.08,
            v_val=-0.06,
        )
        require(
            not g_meta["bottom_tke_bc"] and not g_meta["nemo_sh2_fold_top_v"],
            f"{gname} flags unexpected: {g_meta}",
        )
        _record_caller(gname, g_arr, g_meta)

    # 19. ORCA2-zps_kt1_rest (kt=1 rest entry u=v=0, covering phase2t_tke_entry_gate.py:93)
    orca2_card = build_orca2_zps_card(deck_root)
    orca2_model = cls(
        orca2_card.recipe.grid,
        orca2_card.recipe.z_coord,
        orca2_card.recipe.model_config,
        iwm_forcing=orca2_card.recipe.iwm_forcing,
    )
    arr_kt1, meta_kt1 = _eval_model_caller(
        "ORCA2-zps_kt1_rest",
        orca2_model,
        orca2_card.recipe.initial_state,
        float(orca2_card.dt_s),
        None,
        u_val=0.0,
        v_val=0.0,
    )
    require(
        meta_kt1["bottom_tke_bc"] and meta_kt1["nemo_sh2_fold_top_v"],
        f"ORCA2-zps_kt1_rest flags unexpected: {meta_kt1}",
    )
    _record_caller("ORCA2-zps_kt1_rest", arr_kt1, meta_kt1)

    # 20. ORCA2-zps (active north-fold velocity and viscosity)
    arr_orca2, meta_orca2 = _eval_model_caller(
        "ORCA2-zps",
        orca2_model,
        orca2_card.recipe.initial_state,
        float(orca2_card.dt_s),
        None,
        u_val=0.4,
        v_val=-0.5,
        avm_override=0.02,
    )
    require(
        meta_orca2["bottom_tke_bc"] and meta_orca2["nemo_sh2_fold_top_v"],
        f"ORCA2-zps flags unexpected: {meta_orca2}",
    )
    _record_caller("ORCA2-zps", arr_orca2, meta_orca2)

    require(
        tuple(caller_meta.keys()) == CALLER_ORDER,
        f"caller order {tuple(caller_meta.keys())} != {CALLER_ORDER}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "arrays.npz", **all_arrays)
    manifest = {
        "format": "nemo-testcase-l4-orca2-round336-caller-dump-v1",
        "status": "PASS_R336_CALLER_DUMP",
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
        parent_manifest.get("status") == "PASS_R336_CALLER_DUMP",
        f"parent dump not admitted: {parent_manifest.get('status')}",
    )
    require(
        candidate_manifest.get("status") == "PASS_R336_CALLER_DUMP",
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
                and arr_name == "K_v"
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
                and arr_name == "K_v"
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
            "vmix_scheme": c_info["vmix_scheme"],
            "bottom_tke_bc": c_info["bottom_tke_bc"],
            "nemo_sh2_fold_top_v": c_info["nemo_sh2_fold_top_v"],
            "bottom_dirichlet_is_none": c_info["bottom_dirichlet_is_none"],
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
        and int(orca2_rep["arrays"]["bot_dirichlet"]["unequal_words"]) > 0
        and int(orca2_rep["arrays"]["K_v"]["unequal_words"]) > 0
        and int(orca2_rep["arrays"]["T"]["unequal_words"]) > 0,
        f"ORCA2-zps card did not move vs parent: {orca2_rep}",
    )

    return {
        "format": "nemo-testcase-l4-orca2-round336-caller-audit-v2",
        "status": "PASS_R336_CALLER_AUDIT",
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
    base_row_over_before = {
        "kt": 2,
        "checkpoint": "stage3",
        "field": "T",
        "bit_identical": False,
        "unequal": 18,
        "max_abs": 7.678080393702658e-12,
        "rms": 5.335750398605872e-14,
        "classification": "OVER_BAR",
    }
    base_row_over_after = {
        "kt": 2,
        "checkpoint": "stage3",
        "field": "T",
        "bit_identical": True,
        "unequal": 0,
        "max_abs": 0.0,
        "rms": 0.0,
        "classification": "BIT_EXACT",
    }
    base_row_kt3 = {
        "kt": 3,
        "checkpoint": "stage3",
        "field": "T",
        "bit_identical": False,
        "unequal": 40,
        "max_abs": 1.0e-10,
        "rms": 1.0e-12,
        "classification": "OVER_BAR",
    }
    cand_sha = "c2451f6309fc0000000000000000000000000000"
    ctrl_same_doc = {
        "status": "PASS_R336_D122_ARM",
        "arm": "control",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "bottom_dirichlet_calls": 30,
        "worktree": {"commit": cand_sha, "clean": True},
        "ladder": {"rows": [base_row_exact, base_row_over_before, base_row_kt3]},
    }
    ctrl_parent_doc = {
        "status": "PASS_R336_D122_ARM",
        "arm": "candidate",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "bottom_dirichlet_calls": 30,
        "worktree": {"commit": PARENT_COMMIT, "clean": True},
        "ladder": {"rows": [base_row_exact, base_row_over_before, base_row_kt3]},
    }
    cand_doc = {
        "status": "PASS_R336_D122_ARM",
        "arm": "candidate",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "bottom_dirichlet_calls": 30,
        "worktree": {"commit": cand_sha, "clean": True},
        "ladder": {"rows": [base_row_exact, base_row_over_after, base_row_kt3]},
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
            res_same["status"] == "PASS_R336_D122_CENSUS"
            and res_same["d122_c_and_d"]
            and res_cross["status"] == "PASS_R336_D122_CENSUS"
            and res_cross["d122_c_and_d"]
            and res_cross["control_commit"] == PARENT_COMMIT,
            f"classify failed or plant {plant} fired: {res_cross}",
        )

        # Synthetic caller dump pair for caller_audit_compare self-test
        p_dump = td_path / "parent_dump"
        c_dump = td_path / "cand_dump"
        p_dump.mkdir()
        c_dump.mkdir()
        p_arrays: dict[str, np.ndarray] = {}
        c_arrays: dict[str, np.ndarray] = {}
        p_callers: dict[str, object] = {}
        c_callers: dict[str, object] = {}
        for cname in CALLER_ORDER:
            names = (
                ["bot_dirichlet", "K_v", "T"]
                if cname in ("non_nemo_folded", "non_nemo_unfolded", "faithful_dino", "ORCA2-zps_kt1_rest", "ORCA2-zps")
                else ["K_v", "T"]
            )
            p_meta_arrs = {}
            c_meta_arrs = {}
            for aname in names:
                arr_p = np.array([[1.0, 2.0]], dtype=np.float64)
                arr_c = arr_p.copy()
                if cname == "ORCA2-zps":
                    arr_c[0, 1] = 2.5
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
                "vmix_scheme": "tke" if "bot_dirichlet" in names or cname in ("ORCA2V1-tke", "GYRE-zco") else "constant",
                "bottom_tke_bc": "bot_dirichlet" in names,
                "nemo_sh2_fold_top_v": cname in ("ORCA2V1-tke", "ORCA2-zps_kt1_rest", "ORCA2-zps"),
                "bottom_dirichlet_is_none": "bot_dirichlet" not in names,
                "array_names": names,
            }
            p_callers[cname] = {**common_meta, "arrays": p_meta_arrs}
            c_callers[cname] = {**common_meta, "arrays": c_meta_arrs}

        np.savez(p_dump / "arrays.npz", **p_arrays)
        np.savez(c_dump / "arrays.npz", **c_arrays)
        (p_dump / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "nemo-testcase-l4-orca2-round336-caller-dump-v1",
                    "status": "PASS_R336_CALLER_DUMP",
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
                    "format": "nemo-testcase-l4-orca2-round336-caller-dump-v1",
                    "status": "PASS_R336_CALLER_DUMP",
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
        "status": "PASS_R336_D122_SELF_TEST",
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
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung")
    parser.add_argument("--label")
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--expect-parent-commit", default=PARENT_COMMIT)
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--allow-cross-tree-parent", action="store_true")
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--caller-audit-dump", type=Path)
    parser.add_argument("--caller-audit", action="store_true")
    parser.add_argument("--parent-dump", type=Path)
    parser.add_argument("--candidate-dump", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--plant",
        choices=("none", "exact-loss", "away-row", "parent-commit", "caller-unequal"),
        default="none",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            result = measure(
                args.rung,
                args.label,
                args.arm,
                args.deck_root,
                args.twin_a,
                args.expect_commit,
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
                result["status"] == "PASS_R336_D122_CENSUS",
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

#!/usr/bin/env python3
"""Round 336 Resume 18: Decision-122 and rung-10 wet-value rescore gate for
the north-fold V-face mask restoration in ``_tke_bottom_dirichlet``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

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


class GateError(RuntimeError):
    """The same-tree arm, wet-value audit, or Decision-122 comparison failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _strip_fold_top_v_in_bottom_dirichlet(cfg):
    vmix = getattr(getattr(cfg, "physics", None), "vertical_mixing", None)
    if vmix is None or vmix.scheme != "tke":
        return cfg, False
    if not bool(getattr(vmix.tke, "bottom_tke_bc", False)):
        return cfg, False
    new_tke = vmix.tke._replace(nemo_sh2_fold_top_v=False)
    new_vmix = vmix._replace(tke=new_tke)
    new_phys = cfg.physics._replace(vertical_mixing=new_vmix)
    return cfg._replace(physics=new_phys), True


def _wrap_bottom_dirichlet(original, counter: list[int], *, strip_flag: bool):
    def wrapped(self, state, z_coord=None, config=None, grid=None, *,
                u_now=None, v_now=None):
        cfg_b = self.config if config is None else config
        cfg_eff, active = _strip_fold_top_v_in_bottom_dirichlet(cfg_b)
        if active:
            counter[0] += 1
        use_cfg = cfg_eff if (strip_flag and active) else config
        return original(self, state, z_coord=z_coord, config=use_cfg,
                        grid=grid, u_now=u_now, v_now=v_now)

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
    if rung == "rung10":
        require(
            calls[0] == 30,
            f"expected 30 bottom_dirichlet calls on rung10 (10 steps * 3 stages), got {calls[0]}",
        )
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
    plant: str = "none",
) -> dict[str, object]:
    control = json.loads(control_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    require(
        candidate.get("status") == "PASS_R336_D122_ARM",
        f"candidate arm is not admitted: {candidate.get('status')}",
    )
    if allow_cross_tree_parent:
        require(
            control.get("status") in ("PASS_R336_D122_ARM", "PASS_R333_D122_ARM"),
            f"parent arm is not admitted: {control.get('status')}",
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


def decide(paths: list[Path]) -> dict[str, object]:
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
        all(
            doc["status"] == "PASS_R336_D122_CENSUS" and doc["d122_c_and_d"]
            for doc in docs
        ),
        "a pair failed Decision 122",
    )
    return {
        "format": "nemo-testcase-l4-orca2-round336-d122-decision-v1",
        "status": "PASS_R336_D122_DECISION",
        "worktree_commit": docs[0]["worktree_commit"],
        "cards": docs,
        "d122_c_and_d": True,
    }


def caller_audit(deck_root: Path, expect_commit: str) -> dict[str, object]:
    import hashlib
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
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
    from tests.ocean.unit.test_nemo_sh2_fold_top_v import _bottom_bc_production_case

    stamp = worktree_stamp()
    require(
        stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
        "caller_audit requires clean committed tree",
    )
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
        and bool(jax.config.jax_enable_x64),
        "fp64/libm is not active",
    )

    cls = model_module.LatLonCGridOceanModel
    original_bd = cls._tke_bottom_dirichlet

    def _run_non_nemo_arm(strip_flag: bool):
        calls = [0]
        cls._tke_bottom_dirichlet = _wrap_bottom_dirichlet(
            original_bd, calls, strip_flag=strip_flag
        )
        try:
            bot_f, kv_f, _ = _bottom_bc_production_case(False, fold=True)
            bot_nf, kv_nf, _ = _bottom_bc_production_case(False, fold=False)
        finally:
            cls._tke_bottom_dirichlet = original_bd
        return calls[0], bot_f, kv_f, bot_nf, kv_nf

    c_ctrl, bf_ctrl, kvf_ctrl, bnf_ctrl, kvnf_ctrl = _run_non_nemo_arm(True)
    c_cand, bf_cand, kvf_cand, bnf_cand, kvnf_cand = _run_non_nemo_arm(False)
    require(c_ctrl == c_cand and c_cand > 0, "non-NEMO bottom_dirichlet call count moved")
    for a, b, label in (
        (bf_ctrl, bf_cand, "folded_bot"),
        (kvf_ctrl, kvf_cand, "folded_kv"),
        (bnf_ctrl, bnf_cand, "unfolded_bot"),
        (kvnf_ctrl, kvnf_cand, "unfolded_kv"),
    ):
        unequal = int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
        require(unequal == 0, f"non-NEMO {label} moved: {unequal} unequal words")

    non_orca2_cards = {
        "GYRE-zco": build_gyre_zco_card(),
        "LOCK_EXCHANGE-zco": build_lock_exchange_zco_card(),
        "OVERFLOW-zps": build_overflow_zps_card(),
        "VORTEX-zco": build_vortex_zco_card(),
        "ORCA2-rung0": rung0.rung0_card(deck_root),
    }
    card_rows = {}
    for name, card in non_orca2_cards.items():
        vmix = card.recipe.model_config.physics.vertical_mixing
        tke_cfg = getattr(vmix, "tke", None)
        fold_flag = bool(getattr(tke_cfg, "nemo_sh2_fold_top_v", False))
        bot_flag = bool(getattr(tke_cfg, "bottom_tke_bc", False)) and vmix.scheme == "tke"
        require(not fold_flag, f"{name} unexpectedly enables nemo_sh2_fold_top_v")
        require(not bot_flag, f"{name} unexpectedly enables bottom_tke_bc")
        model = cls(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            iwm_forcing=card.recipe.iwm_forcing,
        )
        out = model._tke_bottom_dirichlet(card.recipe.initial_state)
        require(out is None, f"{name} _tke_bottom_dirichlet did not return None")
        card_rows[name] = {
            "vmix_scheme": vmix.scheme,
            "bottom_tke_bc": bot_flag,
            "nemo_sh2_fold_top_v": fold_flag,
            "bottom_dirichlet_return": None,
            "bit_identical_to_parent": True,
        }

    orca2 = build_orca2_zps_card(deck_root)
    orca2_tke = orca2.recipe.model_config.physics.vertical_mixing.tke
    require(
        bool(orca2_tke.bottom_tke_bc) and bool(orca2_tke.nemo_sh2_fold_top_v),
        "ORCA2 card does not enable both bottom_tke_bc and nemo_sh2_fold_top_v",
    )
    orca2_model = cls(
        orca2.recipe.grid,
        orca2.recipe.z_coord,
        orca2.recipe.model_config,
        iwm_forcing=orca2.recipe.iwm_forcing,
    )
    state0 = orca2_model._seed_tke_preclosure_carry(orca2.recipe.initial_state)
    calls_kt1_ctrl, calls_kt1_cand = [0], [0]
    cls._tke_bottom_dirichlet = _wrap_bottom_dirichlet(
        original_bd, calls_kt1_ctrl, strip_flag=True
    )
    try:
        kt1_ctrl = np.asarray(jax.jit(orca2_model._tke_bottom_dirichlet)(state0))
    finally:
        cls._tke_bottom_dirichlet = original_bd
    cls._tke_bottom_dirichlet = _wrap_bottom_dirichlet(
        original_bd, calls_kt1_cand, strip_flag=False
    )
    try:
        kt1_cand = np.asarray(jax.jit(orca2_model._tke_bottom_dirichlet)(state0))
    finally:
        cls._tke_bottom_dirichlet = original_bd
    kt1_unequal = int(
        np.count_nonzero(kt1_ctrl.view(np.uint64) != kt1_cand.view(np.uint64))
    )
    require(kt1_unequal == 0, f"kt=1 rest-state bottom_dirichlet moved: {kt1_unequal}")

    return {
        "format": "nemo-testcase-l4-orca2-round336-caller-audit-v1",
        "status": "PASS_R336_CALLER_AUDIT",
        "worktree": stamp,
        "non_nemo_prognostic_tke": {
            "nemo_sh2_fold_top_v": False,
            "bottom_tke_bc": True,
            "folded_bot_sha256": hashlib.sha256(bf_cand.tobytes()).hexdigest(),
            "folded_kv_sha256": hashlib.sha256(kvf_cand.tobytes()).hexdigest(),
            "unfolded_bot_sha256": hashlib.sha256(bnf_cand.tobytes()).hexdigest(),
            "unfolded_kv_sha256": hashlib.sha256(kvnf_cand.tobytes()).hexdigest(),
            "unequal_words_vs_parent": 0,
            "bit_identical_to_parent": True,
        },
        "non_orca2_cards": card_rows,
        "orca2_kt1_rest_entry": {
            "bottom_tke_bc": True,
            "nemo_sh2_fold_top_v": True,
            "unequal_words_vs_parent": kt1_unequal,
            "bit_identical_at_rest": True,
            "sha256": hashlib.sha256(kt1_cand.tobytes()).hexdigest(),
        },
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
    ctrl_doc = {
        "status": "PASS_R336_D122_ARM",
        "arm": "control",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "bottom_dirichlet_calls": 30,
        "worktree": {"commit": "deadbeef"},
        "ladder": {"rows": [base_row_exact, base_row_over_before, base_row_kt3]},
    }
    cand_doc = {
        "status": "PASS_R336_D122_ARM",
        "arm": "candidate",
        "rung": "rung10",
        "label": "given_nemo_entry",
        "bottom_dirichlet_calls": 30,
        "worktree": {"commit": "deadbeef"},
        "ladder": {"rows": [base_row_exact, base_row_over_after, base_row_kt3]},
    }
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        p_ctrl = Path(td) / "ctrl.json"
        p_cand = Path(td) / "cand.json"
        p_ctrl.write_text(json.dumps(ctrl_doc), encoding="utf-8")
        p_cand.write_text(json.dumps(cand_doc), encoding="utf-8")
        res = classify(p_ctrl, p_cand, plant=plant)
        require(
            res["status"] == "PASS_R336_D122_CENSUS" and res["d122_c_and_d"],
            f"classify failed or plant {plant} fired: {res}",
        )
    return {"status": "PASS_R336_D122_SELF_TEST", "census": res}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung")
    parser.add_argument("--label")
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--allow-cross-tree-parent", action="store_true")
    parser.add_argument("--decide", nargs="+", type=Path)
    parser.add_argument("--caller-audit", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--plant",
        choices=("none", "exact-loss", "away-row"),
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
                plant=args.plant,
            )
            require(
                result["status"] == "PASS_R336_D122_CENSUS",
                f"census failed: {result}",
            )
        elif args.decide:
            result = decide(args.decide)
        elif args.caller_audit:
            require(
                bool(args.deck_root and args.expect_commit),
                "--caller-audit requires --deck-root and --expect-commit",
            )
            result = caller_audit(args.deck_root, args.expect_commit)
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

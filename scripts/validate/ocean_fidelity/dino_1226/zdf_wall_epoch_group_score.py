#!/usr/bin/env python3
"""Frozen dependency-lattice attribution for corrected-T wall epoch drift."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import eta_flicker_decay as F
import tcarry_basin_reverdict as T


NEMO_SHA256 = "52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a"
CURRENT_PRODUCER = "e013e95ca54957a4454878ed7118e623da0a19ba"
CURRENT_ARTIFACT_SHA256 = (
    "c8c7135a12a75332cb1662052dc7c346b6bae7523dabe75e0f2223c515ac616a")
HISTORICAL_RECEIPT_SHA256 = (
    "5fd033345548dd5b80390293b0ee786d3c23f2a68055ae86afe2b616ad4e5dd0")
CURRENT = {"ratio_first8": 1.6511846170859847,
           "wall_share_first8": 0.17020235431211447}
HISTORICAL = {"ratio_first8": 1.1607251697830108,
              "wall_share_first8": 0.11941867158491266}
HISTORICAL_BAND = {
    "ratio_first8": (1.0470043852687352, 1.2805669019825299),
    "wall_share_first8": (0.0934682579050795, 0.14512176885262343),
}

# The 16 ZDF-sweep changes that postdate the corrected-T basin-lane epoch.
# Wind is deliberately excluded: the current control and every new arm retain
# nemo_literal, whose changed stress hash is separately owned. Metric and
# continuity are excluded because their completed factorial is the input null.
FAITHFUL = {
    "tke_preclosure_coeff_source": "carried_previous_step",
    "tke_shear_evaluation_stage": "step_entry",
    "tke_shear_metric_source": "nemo_qco_live_face",
    "tke_n2_evaluation_stage": "step_entry",
    "tke_matrix_evaluation": "nemo_literal",
    "tke_solver_evaluation": "nemo_literal",
    "tke_langmuir_evaluation": "nemo_literal",
    "tke_etau_exponential_evaluation": "nemo_literal",
    "tke_htau_evaluation": "nemo_literal",
    "tke_mxl_raw_evaluation": "nemo_literal",
    "zdf_implicit_solver_evaluation": "nemo_literal",
    "gm_redi_slope_n2_evaluation": "carried_step_entry",
    "gm_redi_slope_prd_evaluation": "nemo_literal",
    "gm_redi_slope_metric_evaluation": "nemo_reciprocal",
    "gm_redi_slope_face_thickness_evaluation": "nemo_qco_live",
    "gm_redi_slope_depth_evaluation": "nemo_qco_live_literal",
}
GROUPS = {
    "slope_n2_only": {
        "gm_redi_slope_n2_evaluation": "recompute",
    },
    # Reverting the literal matrix is legal only with its literal solver and
    # literal Langmuir consumers reverted too.
    "tke_core": {
        "tke_matrix_evaluation": "factored",
        "tke_solver_evaluation": "shared_thomas",
        "tke_langmuir_evaluation": "vectorized",
    },
    # Fourth corner for the reachable TKE-core x slope-N2 interaction.
    "core_slope_n2": {
        "tke_matrix_evaluation": "factored",
        "tke_solver_evaluation": "shared_thomas",
        "tke_langmuir_evaluation": "vectorized",
        "gm_redi_slope_n2_evaluation": "recompute",
    },
    # Full dependency closure of the entry group.  Its effect is estimable
    # only against core_slope_n2, where every forced downstream reversion is
    # already held legacy.
    "entry_closed": {
        "tke_preclosure_coeff_source": "current_subiteration",
        "tke_shear_evaluation_stage": "implicit_solve_state",
        "tke_shear_metric_source": "tpoint_jacobian",
        "tke_n2_evaluation_stage": "implicit_solve_state",
        "tke_matrix_evaluation": "factored",
        "tke_solver_evaluation": "shared_thomas",
        "tke_langmuir_evaluation": "vectorized",
        "gm_redi_slope_n2_evaluation": "recompute",
    },
    "mxl_zdf": {
        "tke_etau_exponential_evaluation": "jax_expression",
        "tke_htau_evaluation": "jax_expression",
        "tke_mxl_raw_evaluation": "factored",
        "zdf_implicit_solver_evaluation": "shared_thomas",
    },
    "slope_rest": {
        "gm_redi_slope_prd_evaluation": "density_roundtrip",
        "gm_redi_slope_metric_evaluation": "division",
        "gm_redi_slope_face_thickness_evaluation": "static_face",
        "gm_redi_slope_depth_evaluation": "legacy_jacobian_t_surface",
    },
}

# Direct faithful-selector prerequisites transcribed from executable ValueError
# guards.  A GROUPS arm is legal iff every selector left faithful also leaves
# all of its prerequisites faithful.  The remaining ten selectors have no
# selector-to-selector requirement in the validators.
FAITHFUL_REQUIRES = {
    "tke_matrix_evaluation": {
        "tke_preclosure_coeff_source", "tke_n2_evaluation_stage"},
    "tke_solver_evaluation": {"tke_matrix_evaluation"},
    "tke_langmuir_evaluation": {
        "tke_matrix_evaluation", "tke_n2_evaluation_stage"},
    "gm_redi_slope_n2_evaluation": {"tke_n2_evaluation_stage"},
}


def _assert_legal_group_lattice() -> int:
    """Prove registered arms are legal and return all legal subset count."""
    for group, changes in GROUPS.items():
        reverted = set(changes)
        for selector, requirements in FAITHFUL_REQUIRES.items():
            if selector not in reverted:
                broken = requirements & reverted
                if broken:
                    raise SystemExit(
                        f"STOP illegal group {group}: faithful {selector} "
                        f"requires faithful {sorted(broken)}")
    coupled = set(FAITHFUL_REQUIRES)
    for requirements in FAITHFUL_REQUIRES.values():
        coupled.update(requirements)
    independent_count = len(FAITHFUL) - len(coupled)
    legal_coupled = 0
    coupled = sorted(coupled)
    for mask in range(1 << len(coupled)):
        reverted = {name for bit, name in enumerate(coupled)
                    if mask & (1 << bit)}
        legal = all(
            selector in reverted or not (requirements & reverted)
            for selector, requirements in FAITHFUL_REQUIRES.items())
        legal_coupled += int(legal)
    return legal_coupled * (1 << independent_count)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _scalar(z, key):
    if key not in z.files:
        raise SystemExit(f"STOP missing artifact stamp {key}")
    return np.asarray(z[key]).item()


def _bit_identical(left, right) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.ascontiguousarray(a).tobytes()
            == np.ascontiguousarray(b).tobytes())


def _common_receipts(z, config) -> dict[str, object]:
    if (not bool(_scalar(z, "stable"))
            or _scalar(z, "control_dtype") != "float64"
            or _scalar(z, "nemo_ladder_mode") != "both"
            or int(_scalar(z, "capture_every_steps")) != 1
            or float(_scalar(z, "dt_seconds")) != 2700.0
            or np.asarray(z["eta"]).shape[0] != 160
            or np.asarray(z["eta"]).dtype != np.float64):
        raise SystemExit("STOP runtime/per-step eta receipt")
    session = os.environ.get("CODEX_SESSION_ID")
    if not session or _scalar(z, "codex_session_id") != session:
        raise SystemExit("STOP session receipt")
    stress = {key: _scalar(z, key) for key in (
        "twin_start_mode", "bridge_before_stress_stagger",
        "bridge_before_stress_reconstruction_seconds",
        "bridge_before_stress_sha256")}
    errors = T.tpoint_stress_receipt_errors(stress, config)
    if errors:
        raise SystemExit("STOP start/stress receipt:\n  " + "\n  ".join(errors))
    return stress


def _load_current(path: str):
    if sha256(path) != CURRENT_ARTIFACT_SHA256:
        raise SystemExit("STOP current-control artifact SHA mismatch")
    with np.load(path) as z:
        if (_scalar(z, "producer_git_sha") != CURRENT_PRODUCER
                or int(_scalar(z, "producer_dirty_tracked_files")) != 0):
            raise SystemExit("STOP current-control producer receipt")
        config = json.loads(str(_scalar(z, "run_config")))
        stress = _common_receipts(z, config)
        for key, value in FAITHFUL.items():
            if config.get(key) != value:
                raise SystemExit(f"STOP current {key}={config.get(key)!r} != {value!r}")
        if (config.get("vface_zonal_metric_evaluation")
                != "legacy_tracer_midpoint"
                or config.get("barotropic_continuity_evaluation") != "generic"):
            raise SystemExit("STOP current metric/continuity selector receipt")
        return (config, str(_scalar(z, "initial_state_sha256")),
                np.asarray(z["land_mask"]), stress)


def _load_group(path: str, producer: str, group: str, current_config: dict,
                initial_hash: str, land_mask: np.ndarray):
    with np.load(path) as z:
        if (_scalar(z, "producer_git_sha") != producer
                or int(_scalar(z, "producer_dirty_tracked_files")) != 0):
            raise SystemExit(f"STOP {group} producer receipt")
        config = json.loads(str(_scalar(z, "run_config")))
        stress = _common_receipts(z, config)
        expected = dict(current_config)
        expected.update(GROUPS[group])
        if config != expected:
            differing = sorted(key for key in config.keys() | expected.keys()
                               if config.get(key) != expected.get(key))
            raise SystemExit(f"STOP {group} config differs at {differing}")
        if str(_scalar(z, "initial_state_sha256")) != initial_hash:
            raise SystemExit(f"STOP {group} initial-state hash")
        if not _bit_identical(z["land_mask"], land_mask):
            raise SystemExit(f"STOP {group} land mask")
        return stress


def _score(nemo: str, artifact: str, detail: Path) -> dict[str, float]:
    record = F.analyse(nemo, artifact, str(detail), F.ewt.MESH_MASK)
    if (not record["self_check"]["land_poison_identical"]
            or not (0.95 <= record["self_check"]["plant_large"]["recovery_ratio"]
                    <= 1.05)):
        raise SystemExit(f"STOP flicker controls for {artifact}")
    return {
        "ratio_first8": record["regions"]["all"]
        ["ratio_lego_over_nemo"]["first8"],
        "wall_share_first8": record["wall_share"]["legoESM"]["first8"]["wall"],
    }


def _closure(score: dict[str, float]) -> dict[str, float]:
    return {key: (CURRENT[key] - score[key]) / (CURRENT[key] - HISTORICAL[key])
            for key in CURRENT}


def classify(score: dict[str, float]) -> str:
    closure = _closure(score)
    in_historical_band = all(
        HISTORICAL_BAND[key][0] <= score[key] <= HISTORICAL_BAND[key][1]
        for key in CURRENT)
    if in_historical_band and all(closure[key] >= 0.5 for key in CURRENT):
        return "EPOCH_RESTORED"
    if all(closure[key] >= 0.5 for key in CURRENT):
        return "MAJORITY_OWNER"
    if all(abs(closure[key]) <= 0.10 for key in CURRENT):
        return "BOUNDED_SMALL"
    return "OPEN_MIXED_OR_PARTIAL"


def classify_closure(closure: dict[str, float]) -> str:
    """Classify a conditional contrast without inventing a missing endpoint."""
    if all(closure[key] >= 0.50 for key in CURRENT):
        return "CONDITIONAL_MAJORITY_OWNER"
    if all(abs(closure[key]) <= 0.10 for key in CURRENT):
        return "BOUNDED_SMALL"
    return "OPEN_MIXED_OR_PARTIAL"


def _classifier_plants() -> dict[str, object]:
    majority = {key: CURRENT[key] - 0.60 * (CURRENT[key] - HISTORICAL[key])
                for key in CURRENT}
    mixed = {"ratio_first8": majority["ratio_first8"],
             "wall_share_first8": CURRENT["wall_share_first8"]}
    plants = {
        "restored": classify(dict(HISTORICAL)),
        "majority": classify(majority),
        "small": classify(dict(CURRENT)),
        "mixed": classify(mixed),
    }
    expected = {"restored": "EPOCH_RESTORED", "majority": "MAJORITY_OWNER",
                "small": "BOUNDED_SMALL", "mixed": "OPEN_MIXED_OR_PARTIAL"}
    if plants != expected:
        raise SystemExit(f"STOP classifier plants failed: {plants}")
    conditional = {
        "majority": classify_closure({key: 0.60 for key in CURRENT}),
        "small": classify_closure({key: 0.05 for key in CURRENT}),
        "mixed": classify_closure(
            {"ratio_first8": 0.60, "wall_share_first8": 0.0}),
    }
    expected_conditional = {
        "majority": "CONDITIONAL_MAJORITY_OWNER",
        "small": "BOUNDED_SMALL",
        "mixed": "OPEN_MIXED_OR_PARTIAL",
    }
    if conditional != expected_conditional:
        raise SystemExit(f"STOP conditional classifier plants failed: {conditional}")
    plants["conditional"] = conditional
    return plants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    for name in ("nemo", "current", "slope_n2_only", "tke_core",
                 "core_slope_n2", "entry_closed", "mxl_zdf", "slope_rest",
                 "historical_receipt", "producer_commit", "out"):
        parser.add_argument("--" + name.replace("_", "-"))
    args = parser.parse_args()
    plants = _classifier_plants()
    legal_selector_subsets = _assert_legal_group_lattice()
    if legal_selector_subsets != 14336:
        raise SystemExit(
            f"STOP selector-lattice count {legal_selector_subsets} != 14336")
    if args.self_test:
        print(json.dumps({"classifier_plants": plants,
                          "legal_selector_subsets": legal_selector_subsets},
                         sort_keys=True))
        return 0
    missing = [name for name, value in vars(args).items()
               if name != "self_test" and value is None]
    if missing:
        parser.error(f"required for scoring: {', '.join(missing)}")

    root = Path(__file__).resolve().parents[4]
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True):
        raise SystemExit("STOP dirty scorer checkout")
    if sha256(args.nemo) != NEMO_SHA256:
        raise SystemExit("STOP certified NEMO comparator SHA mismatch")
    if sha256(args.historical_receipt) != HISTORICAL_RECEIPT_SHA256:
        raise SystemExit("STOP historical corrected-T receipt SHA mismatch")
    historical_record = json.loads(Path(args.historical_receipt).read_text())
    historical_from_receipt = {
        "ratio_first8": historical_record["regions"]["all"]
        ["ratio_lego_over_nemo"]["first8"],
        "wall_share_first8": historical_record["wall_share"]
        ["legoESM"]["first8"]["wall"],
    }
    if historical_from_receipt != HISTORICAL:
        raise SystemExit("STOP historical receipt values changed")
    if subprocess.run(
            ["git", "diff", "--quiet", f"{CURRENT_PRODUCER}..{args.producer_commit}",
             "--", "packages/core", "packages/ocean",
             "scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"],
            cwd=root, check=False).returncode != 0:
        raise SystemExit("STOP production model/harness changed since current control")

    current_config, initial_hash, land_mask, current_stress = _load_current(
        args.current)
    group_paths = {name: getattr(args, name) for name in GROUPS}
    group_stress = {
        name: _load_group(path, args.producer_commit, name, current_config,
                          initial_hash, land_mask)
        for name, path in group_paths.items()
    }
    T.check_tpoint_stress_receipt_plants(current_stress, current_config)

    out = Path(args.out)
    current_score = _score(
        args.nemo, args.current, out.with_name(out.stem + "_current.json"))
    if current_score != CURRENT:
        raise SystemExit(f"STOP current score {current_score} != frozen {CURRENT}")
    scores = {name: _score(
        args.nemo, path, out.with_name(out.stem + f"_{name}.json"))
        for name, path in group_paths.items()}
    arm_closures = {name: _closure(score) for name, score in scores.items()}
    arm_classes = {name: classify(score) for name, score in scores.items()}

    def contrast_closure(left: str, right: str) -> dict[str, float]:
        """Closure added by moving from left endpoint to right endpoint."""
        return {
            key: (scores[left][key] - scores[right][key])
            / (CURRENT[key] - HISTORICAL[key]) for key in CURRENT}

    conditional = {
        # The only reachable entry contrast: add the entry reversions after
        # every downstream selector forced by their dependency closure is
        # already legacy.
        "entry_given_legacy_core_and_slope_n2": contrast_closure(
            "core_slope_n2", "entry_closed"),
        "tke_core_given_legacy_slope_n2": contrast_closure(
            "slope_n2_only", "core_slope_n2"),
    }
    core_slope_interaction = {
        key: ((scores["slope_n2_only"][key]
               - scores["core_slope_n2"][key])
              - (CURRENT[key] - scores["tke_core"][key]))
        / (CURRENT[key] - HISTORICAL[key])
        for key in CURRENT
    }
    conditional_classes = {
        name: classify_closure(value) for name, value in conditional.items()}
    primitive = {
        "tke_core_at_faithful_slope_n2": arm_closures["tke_core"],
        "mxl_zdf": arm_closures["mxl_zdf"],
        "slope_n2_at_faithful_entry": arm_closures["slope_n2_only"],
        "slope_rest_at_faithful_slope_n2": arm_closures["slope_rest"],
        "entry_given_legacy_core_and_slope_n2": conditional[
            "entry_given_legacy_core_and_slope_n2"],
    }
    primitive_classes = {
        "tke_core_at_faithful_slope_n2": arm_classes["tke_core"],
        "mxl_zdf": arm_classes["mxl_zdf"],
        "slope_n2_at_faithful_entry": arm_classes["slope_n2_only"],
        "slope_rest_at_faithful_slope_n2": arm_classes["slope_rest"],
        "entry_given_legacy_core_and_slope_n2": conditional_classes[
            "entry_given_legacy_core_and_slope_n2"],
    }
    owners = [name for name, value in primitive_classes.items()
              if value in ("EPOCH_RESTORED", "MAJORITY_OWNER",
                           "CONDITIONAL_MAJORITY_OWNER")]
    if owners == ["entry_given_legacy_core_and_slope_n2"]:
        disposition = "LOCALIZED_TO_ENTRY_CONDITIONAL_ON_LEGACY_DEPENDENCIES"
        next_action = ("split entry only through legal dependency closures; "
                       "do not claim its unreachable independent main effect")
    elif len(owners) == 1:
        disposition = "LOCALIZED_TO_" + owners[0].upper()
        next_action = f"split {owners[0]} with dependency-coherent children"
    elif len(owners) > 1:
        disposition = "COMPOSITION_MULTIPLE_REACHABLE_CONTRASTS"
        next_action = "run complement interactions among the named contrasts"
    else:
        disposition = "OPEN_DISTRIBUTED_OR_INTERACTION"
        next_action = "run the all-16-legacy universe gate, then complement pairs"
    result = {
        "schema": "zdf-wall-epoch-lattice-score-v2",
        "session_id": os.environ["CODEX_SESSION_ID"],
        "producer_commit": args.producer_commit,
        "current_producer": CURRENT_PRODUCER,
        "nemo_sha256": NEMO_SHA256,
        "current_artifact_sha256": CURRENT_ARTIFACT_SHA256,
        "historical_receipt_sha256": HISTORICAL_RECEIPT_SHA256,
        "initial_state_sha256": initial_hash,
        "current": CURRENT,
        "historical": HISTORICAL,
        "historical_band": HISTORICAL_BAND,
        "groups": GROUPS,
        "faithful_selector_requires": {
            key: sorted(value) for key, value in FAITHFUL_REQUIRES.items()},
        "legal_selector_subsets": legal_selector_subsets,
        "structurally_unreachable_contrasts": [
            {
                "contrast": "legacy preclosure with faithful matrix",
                "reason": "literal matrix requires carried-previous-step coefficients",
            },
            {
                "contrast": "legacy N2 stage with faithful matrix/Langmuir/slope-N2",
                "reason": "all three faithful consumers require the step-entry N2 bundle",
            },
            {
                "contrast": "independent entry main effect at faithful downstream selectors",
                "reason": "entry reversion forces the core and slope-N2 dependency closure",
                "interaction_status": "UNIDENTIFIABLE_FROM_LEGAL_CORNERS",
            },
        ],
        "group_artifact_sha256": {
            name: sha256(path) for name, path in group_paths.items()},
        "scores": scores,
        "arm_closure_fraction": arm_closures,
        "arm_classification": arm_classes,
        "conditional_closure_fraction": conditional,
        "conditional_classification": conditional_classes,
        "tke_core_x_slope_n2_interaction_fraction": core_slope_interaction,
        "primitive_closure_fraction": primitive,
        "primitive_classification": primitive_classes,
        "disposition": disposition,
        "next_action": next_action,
        "classifier_plants": plants,
        "stress_receipts": {"current": current_stress, **group_stress},
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={out} sha256={sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

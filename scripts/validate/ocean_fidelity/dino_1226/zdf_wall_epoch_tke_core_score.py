#!/usr/bin/env python3
"""Frozen legal-lattice sub-peel of the wall-epoch TKE-core owner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import zdf_wall_epoch_group_score as G


CURRENT_SHA256 = G.CURRENT_ARTIFACT_SHA256
FULL_CORE_SHA256 = (
    "1be9010230834eca71f349aa672c20c5704c2f6b0887ed19d0b4eff01e38ee3a")
FULL_CORE_SCORE = {
    "ratio_first8": 1.163579045567568,
    "wall_share_first8": 0.11693494406953533,
}
CHANGES = {
    "solver_only": {"tke_solver_evaluation": "shared_thomas"},
    "langmuir_only": {"tke_langmuir_evaluation": "vectorized"},
    "solver_langmuir": {
        "tke_solver_evaluation": "shared_thomas",
        "tke_langmuir_evaluation": "vectorized",
    },
    "full_core": {
        "tke_matrix_evaluation": "factored",
        "tke_solver_evaluation": "shared_thomas",
        "tke_langmuir_evaluation": "vectorized",
    },
}
MODEL_PATHS = (
    "packages/core", "packages/ocean",
    "scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py",
)


def _model_diff_zero(root: Path, left: str, right: str) -> None:
    for commit in (left, right):
        if subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=root, check=False).returncode:
            raise SystemExit(f"STOP missing producer commit {commit}")
    if subprocess.run(
            ["git", "diff", "--quiet", f"{left}..{right}", "--", *MODEL_PATHS],
            cwd=root, check=False).returncode:
        raise SystemExit(
            f"STOP production model/harness differs: {left}..{right}")


def _load(
    path: str,
    *,
    name: str,
    root: Path,
    selected_producer: str,
    expected_config: dict[str, object] | None,
    expected_hash: str | None,
    initial_hash: str | None,
    land_mask: np.ndarray | None,
) -> dict[str, object]:
    if expected_hash is not None and G.sha256(path) != expected_hash:
        raise SystemExit(f"STOP {name} retained-artifact SHA mismatch")
    with np.load(path) as z:
        if (int(G._scalar(z, "producer_dirty_tracked_files")) != 0
                or not bool(G._scalar(z, "stable"))):
            raise SystemExit(f"STOP {name} clean/stable receipt")
        eta = np.asarray(z["eta"])
        if eta.shape != (160, 199, 52) or eta.dtype != np.float64:
            raise SystemExit(
                f"STOP {name} full-halo eta receipt {eta.shape}/{eta.dtype}")
        producer = str(G._scalar(z, "producer_git_sha"))
        if expected_hash is None and producer != selected_producer:
            raise SystemExit(
                f"STOP {name} producer {producer} != {selected_producer}")
        _model_diff_zero(root, producer, selected_producer)
        config = json.loads(str(G._scalar(z, "run_config")))
        stress = G._common_receipts(z, config)
        if expected_config is None:
            for key, value in G.FAITHFUL.items():
                if config.get(key) != value:
                    raise SystemExit(
                        f"STOP current {key}={config.get(key)!r} != {value!r}")
            if (config.get("vface_zonal_metric_evaluation")
                    != "legacy_tracer_midpoint"
                    or config.get("barotropic_continuity_evaluation")
                    != "generic"):
                raise SystemExit("STOP current metric/continuity selectors")
        elif config != expected_config:
            differing = sorted(
                key for key in config.keys() | expected_config.keys()
                if config.get(key) != expected_config.get(key))
            raise SystemExit(f"STOP {name} config differs at {differing}")
        artifact_initial = str(G._scalar(z, "initial_state_sha256"))
        if initial_hash is not None and artifact_initial != initial_hash:
            raise SystemExit(f"STOP {name} initial-state hash")
        artifact_mask = np.asarray(z["land_mask"])
        if land_mask is not None and not G._bit_identical(
                artifact_mask, land_mask):
            raise SystemExit(f"STOP {name} land mask")
        return {
            "config": config,
            "initial_hash": artifact_initial,
            "land_mask": artifact_mask,
            "producer": producer,
            "stress": stress,
            "sha256": G.sha256(path),
        }


def classify_component(values: dict[str, float]) -> str:
    if all(values[key] >= 0.50 for key in G.CURRENT):
        return "MAJORITY_OWNER"
    if all(abs(values[key]) <= 0.10 for key in G.CURRENT):
        return "BOUNDED_SMALL"
    return "OPEN_MIXED_OR_PARTIAL"


def resolve(classes: dict[str, str]) -> tuple[str, str]:
    owners = sorted(
        name for name, value in classes.items() if value == "MAJORITY_OWNER")
    open_terms = sorted(
        name for name, value in classes.items()
        if value == "OPEN_MIXED_OR_PARTIAL")
    if len(owners) == 1 and not open_terms:
        owner = owners[0]
        return ("LOCALIZED_TO_" + owner.upper(),
                f"walk the wall/interior operand ladder for {owner}")
    if len(owners) == 1:
        return ("MAJORITY_" + owners[0].upper() + "_WITH_OPEN_COMPONENTS",
                "split the named open components before claiming sole ownership")
    if len(owners) > 1:
        return ("COMPOSITION_MULTIPLE_TKE_CORE_COMPONENTS",
                "retain the measured component decomposition; do not pick one")
    return ("OPEN_DISTRIBUTED_OR_SUBBAR_COMPOSITION",
            "refine the component ladder without relaxing the frozen bars")


def _controls() -> dict[str, object]:
    for arm, changes in CHANGES.items():
        reverted = set(changes)
        for selector, requirements in G.FAITHFUL_REQUIRES.items():
            if selector not in reverted and requirements & reverted:
                raise SystemExit(
                    f"STOP illegal registered arm {arm}: faithful {selector} "
                    f"requires {sorted(requirements & reverted)}")
    classes = {
        "majority": classify_component({key: 0.60 for key in G.CURRENT}),
        "small": classify_component({key: 0.05 for key in G.CURRENT}),
        "mixed": classify_component(
            {"ratio_first8": 0.60, "wall_share_first8": 0.0}),
    }
    expected = {
        "majority": "MAJORITY_OWNER", "small": "BOUNDED_SMALL",
        "mixed": "OPEN_MIXED_OR_PARTIAL",
    }
    if classes != expected:
        raise SystemExit(f"STOP component classifier plants {classes}")
    disposition_plants = {
        "localized": resolve({
            "solver": "MAJORITY_OWNER", "langmuir": "BOUNDED_SMALL",
            "interaction": "BOUNDED_SMALL", "matrix": "BOUNDED_SMALL"})[0],
        "composition": resolve({
            "solver": "MAJORITY_OWNER", "langmuir": "MAJORITY_OWNER",
            "interaction": "BOUNDED_SMALL", "matrix": "BOUNDED_SMALL"})[0],
        "open": resolve({
            "solver": "BOUNDED_SMALL", "langmuir": "BOUNDED_SMALL",
            "interaction": "OPEN_MIXED_OR_PARTIAL",
            "matrix": "BOUNDED_SMALL"})[0],
    }
    if disposition_plants != {
            "localized": "LOCALIZED_TO_SOLVER",
            "composition": "COMPOSITION_MULTIPLE_TKE_CORE_COMPONENTS",
            "open": "OPEN_DISTRIBUTED_OR_SUBBAR_COMPOSITION"}:
        raise SystemExit(f"STOP disposition plants {disposition_plants}")
    terms = (0.17, 0.23, -0.04, 0.64)
    if abs(sum(terms) - 1.0) > 1e-15:
        raise SystemExit("STOP decomposition plant")
    return {"registered_arms_legal": True, "classifier": classes,
            "disposition": disposition_plants,
            "decomposition_sum": sum(terms)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    for name in (
            "nemo", "current", "full_core", "solver_only", "langmuir_only",
            "solver_langmuir", "historical_receipt", "producer_commit", "out"):
        parser.add_argument("--" + name.replace("_", "-"))
    args = parser.parse_args()
    controls = _controls()
    if args.self_test:
        print(json.dumps({"controls": controls}, sort_keys=True))
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
    if G.sha256(args.nemo) != G.NEMO_SHA256:
        raise SystemExit("STOP certified NEMO comparator SHA mismatch")
    if G.sha256(args.historical_receipt) != G.HISTORICAL_RECEIPT_SHA256:
        raise SystemExit("STOP historical corrected-T receipt SHA mismatch")
    historical = json.loads(Path(args.historical_receipt).read_text())
    receipt_scores = {
        "ratio_first8": historical["regions"]["all"]
        ["ratio_lego_over_nemo"]["first8"],
        "wall_share_first8": historical["wall_share"]
        ["legoESM"]["first8"]["wall"],
    }
    if receipt_scores != G.HISTORICAL:
        raise SystemExit("STOP historical receipt values changed")

    current_meta = _load(
        args.current, name="current", root=root,
        selected_producer=args.producer_commit, expected_config=None,
        expected_hash=CURRENT_SHA256, initial_hash=None, land_mask=None)
    current_config = current_meta["config"]
    initial_hash = current_meta["initial_hash"]
    land_mask = current_meta["land_mask"]
    G.T.check_tpoint_stress_receipt_plants(
        current_meta["stress"], current_config)

    paths = {
        "full_core": args.full_core,
        "solver_only": args.solver_only,
        "langmuir_only": args.langmuir_only,
        "solver_langmuir": args.solver_langmuir,
    }
    meta = {}
    for name, path in paths.items():
        expected = dict(current_config)
        expected.update(CHANGES[name])
        meta[name] = _load(
            path, name=name, root=root,
            selected_producer=args.producer_commit,
            expected_config=expected,
            expected_hash=(FULL_CORE_SHA256 if name == "full_core" else None),
            initial_hash=initial_hash, land_mask=land_mask)

    out = Path(args.out)
    current_score = G._score(
        args.nemo, args.current, out.with_name(out.stem + "_current.json"))
    if current_score != G.CURRENT:
        raise SystemExit(f"STOP current score {current_score} != {G.CURRENT}")
    scores = {
        name: G._score(
            args.nemo, path, out.with_name(out.stem + f"_{name}.json"))
        for name, path in paths.items()
    }
    if scores["full_core"] != FULL_CORE_SCORE:
        raise SystemExit(
            f"STOP retained full-core score {scores['full_core']} "
            f"!= {FULL_CORE_SCORE}")

    denominator = {
        key: G.CURRENT[key] - G.HISTORICAL[key] for key in G.CURRENT}

    def normalized(numerator: dict[str, float]) -> dict[str, float]:
        return {key: numerator[key] / denominator[key] for key in G.CURRENT}

    solver = normalized({
        key: G.CURRENT[key] - scores["solver_only"][key] for key in G.CURRENT})
    langmuir = normalized({
        key: G.CURRENT[key] - scores["langmuir_only"][key]
        for key in G.CURRENT})
    interaction = normalized({
        key: ((G.CURRENT[key] - scores["solver_langmuir"][key])
              - (G.CURRENT[key] - scores["solver_only"][key])
              - (G.CURRENT[key] - scores["langmuir_only"][key]))
        for key in G.CURRENT})
    matrix = normalized({
        key: scores["solver_langmuir"][key] - scores["full_core"][key]
        for key in G.CURRENT})
    components = {
        "solver_main_at_faithful_langmuir": solver,
        "langmuir_main_at_faithful_solver": langmuir,
        "solver_x_langmuir_at_faithful_matrix": interaction,
        "matrix_given_legacy_solver_and_langmuir": matrix,
    }
    component_classes = {
        name: classify_component(values) for name, values in components.items()}
    full_core_closure = G._closure(scores["full_core"])
    reconstructed = {
        key: sum(values[key] for values in components.values())
        for key in G.CURRENT}
    if any(abs(reconstructed[key] - full_core_closure[key]) > 5e-13
           for key in G.CURRENT):
        raise SystemExit(
            f"STOP component reconstruction {reconstructed} != "
            f"{full_core_closure}")
    disposition, next_action = resolve(component_classes)

    conditional = {
        "solver_given_legacy_langmuir": normalized({
            key: scores["langmuir_only"][key]
            - scores["solver_langmuir"][key] for key in G.CURRENT}),
        "langmuir_given_legacy_solver": normalized({
            key: scores["solver_only"][key]
            - scores["solver_langmuir"][key] for key in G.CURRENT}),
    }
    result = {
        "schema": "zdf-wall-epoch-tke-core-subpeel-v1",
        "session_id": os.environ["CODEX_SESSION_ID"],
        "producer_commit": args.producer_commit,
        "retained_producer_model_diff_zero": {
            "current": current_meta["producer"],
            "full_core": meta["full_core"]["producer"],
        },
        "nemo_sha256": G.NEMO_SHA256,
        "historical_receipt_sha256": G.HISTORICAL_RECEIPT_SHA256,
        "current_artifact_sha256": CURRENT_SHA256,
        "full_core_artifact_sha256": FULL_CORE_SHA256,
        "initial_state_sha256": initial_hash,
        "eta_shape": [160, 199, 52],
        "current": G.CURRENT,
        "historical": G.HISTORICAL,
        "historical_band": G.HISTORICAL_BAND,
        "changes": CHANGES,
        "artifact_sha256": {
            "current": current_meta["sha256"],
            **{name: values["sha256"] for name, values in meta.items()},
        },
        "scores": scores,
        "arm_closure_fraction": {
            name: G._closure(score) for name, score in scores.items()},
        "arm_classification": {
            name: G.classify(score) for name, score in scores.items()},
        "component_closure_fraction": components,
        "component_classification": component_classes,
        "conditional_closure_fraction": conditional,
        "conditional_classification": {
            name: classify_component(values)
            for name, values in conditional.items()},
        "full_core_closure_fraction": full_core_closure,
        "reconstructed_full_core_closure_fraction": reconstructed,
        "full_core_reconstruction_pass": True,
        "disposition": disposition,
        "next_action": next_action,
        "controls": controls,
        "stress_receipts": {
            "current": current_meta["stress"],
            **{name: values["stress"] for name, values in meta.items()},
        },
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={out} sha256={G.sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

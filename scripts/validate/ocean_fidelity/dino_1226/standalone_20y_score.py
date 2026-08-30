#!/usr/bin/env python3
"""Fail-closed ensemble classifier for the preregistered DINO T1 battery.

Each admitted member directory must contain the runner/converter manifest and
``statistics.json`` with one scalar mapping for every frozen family.  Snapshot
reduction remains owned by the recorded family instruments; this scorer never
retypes a physical reducer.  It owns only ensemble admission, the horizon-
matched floor, bootstrap bands, simultaneous family maximum, and capstone
classification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "dino_standalone_20y_score_v1"
MEMBERS = tuple(range(6))
FAMILIES = (
    "acc",
    "basin_row_transport",
    "mld_seasonal_cycle",
    "ts_water_mass_census",
    "density_contrasts",
    "variability",
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1455
R_BAR = 2.0


class AdmissionError(RuntimeError):
    """An input cannot participate in the T1 science verdict."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ensemble(root: Path, model: str) -> tuple[
        dict[str, dict[str, np.ndarray]], dict[str, float], list[dict[str, Any]]]:
    """Load six standalone members into family/scalar arrays."""
    accumulated: dict[str, dict[str, list[float]]] = {
        family: {} for family in FAMILIES}
    quantum: dict[str, float] = {}
    receipts = []
    reference_names: dict[str, set[str]] | None = None
    for member in MEMBERS:
        directory = root / f"m{member}"
        manifest_path = directory / "manifest.json"
        stats_path = directory / "statistics.json"
        manifest = _read_json(manifest_path)
        stats = _read_json(stats_path)
        reasons = []
        if manifest.get("claim_admissible") is not True:
            reasons.append("claim_admissible is not true")
        if manifest.get("twin_start_mode") != "standalone":
            reasons.append("start mode is not standalone")
        if manifest.get("bridge_paths") != [] or manifest.get("restart_paths") != []:
            reasons.append("bridge/restart path is present")
        if manifest.get("storage_dtype") != "float64":
            reasons.append("storage is not fp64")
        if manifest.get("compute_dtype") != "float64":
            reasons.append("compute is not fp64")
        if manifest.get("member") != member:
            reasons.append("member label mismatch")
        if stats.get("schema") != "dino_standalone_20y_statistics_v1":
            reasons.append("statistics schema mismatch")
        if stats.get("member") != member:
            reasons.append("statistics member mismatch")
        families = stats.get("families")
        if not isinstance(families, dict) or set(families) != set(FAMILIES):
            reasons.append("statistics do not contain exactly six families")
            families = {}
        current_names = {
            family: set(families.get(family, {})) for family in FAMILIES}
        if any(not names for names in current_names.values()):
            reasons.append("one or more families have no scalars")
        if reference_names is None:
            reference_names = current_names
        elif current_names != reference_names:
            reasons.append("scalar names differ across members")
        if reasons:
            raise AdmissionError(
                f"{model} member {member} inadmissible: " + "; ".join(reasons))

        for family in FAMILIES:
            for name, value in families[family].items():
                value = float(value)
                if not np.isfinite(value):
                    raise AdmissionError(
                        f"{model} member {member} {family}/{name} is nonfinite")
                accumulated[family].setdefault(name, []).append(value)
        member_quantum = stats.get("quantum", {})
        for key, value in member_quantum.items():
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise AdmissionError(
                    f"{model} member {member} invalid quantum {key}={value}")
            quantum[key] = max(quantum.get(key, 0.0), value)
        receipts.append({
            "member": member,
            "manifest_sha256": _sha256(manifest_path),
            "statistics_sha256": _sha256(stats_path),
        })

    arrays = {
        family: {name: np.asarray(values, dtype=np.float64)
                 for name, values in scalars.items()}
        for family, scalars in accumulated.items()
    }
    return arrays, quantum, receipts


def classify_family(lego: dict[str, np.ndarray],
                    nemo: dict[str, np.ndarray],
                    quantum: dict[str, float], *,
                    rng: np.random.Generator) -> dict[str, Any]:
    if set(lego) != set(nemo):
        raise AdmissionError("lego/NEMO scalar-name mismatch within family")
    names = sorted(lego)
    lmat = np.column_stack([lego[name] for name in names])
    nmat = np.column_stack([nemo[name] for name in names])
    if lmat.shape != (6, len(names)) or nmat.shape != lmat.shape:
        raise AdmissionError("every scalar must contain exactly six members")

    li = rng.integers(0, 6, size=(BOOTSTRAP_DRAWS, 6))
    ni = rng.integers(0, 6, size=(BOOTSTRAP_DRAWS, 6))
    lb = lmat[li]
    nb = nmat[ni]
    gap_b = lb.mean(axis=1) - nb.mean(axis=1)
    floor_b = np.sqrt(lb.std(axis=1, ddof=1) ** 2
                      + nb.std(axis=1, ddof=1) ** 2)
    valid = np.isfinite(floor_b) & (floor_b > 0.0)
    zero_fraction = 1.0 - valid.mean(axis=0)
    r_b = np.full_like(floor_b, np.nan)
    np.divide(np.abs(gap_b), floor_b, out=r_b, where=valid)

    scalars: dict[str, Any] = {}
    eligible_columns = []
    for column, name in enumerate(names):
        lv = lmat[:, column]
        nv = nmat[:, column]
        lsd = float(np.std(lv, ddof=1))
        nsd = float(np.std(nv, ddof=1))
        floor = float(np.hypot(lsd, nsd))
        gap = float(np.mean(lv) - np.mean(nv))
        q = float(quantum.get(name, 0.0))
        quantized_reasons = []
        if np.unique(lv).size < 4:
            quantized_reasons.append("lego_unique_lt_4")
        if np.unique(nv).size < 4:
            quantized_reasons.append("nemo_unique_lt_4")
        if floor < 10.0 * q:
            quantized_reasons.append("floor_lt_10x_quantum")
        if zero_fraction[column] > 0.01:
            quantized_reasons.append("bootstrap_bad_floor_gt_1pct")
        finite_r = r_b[:, column][np.isfinite(r_b[:, column])]
        if finite_r.size:
            r_lo, r_hi = np.percentile(finite_r, (2.5, 97.5))
        else:
            r_lo = r_hi = float("nan")
        if quantized_reasons:
            verdict = "UNRESOLVED_QUANTIZED"
        elif r_hi <= R_BAR:
            verdict = "CONFIRM"
            eligible_columns.append(column)
        elif r_lo > R_BAR:
            verdict = "REFUTE"
            eligible_columns.append(column)
        else:
            verdict = "UNRESOLVED"
            eligible_columns.append(column)
        imbalance = (
            max(lsd, nsd) / min(lsd, nsd)
            if min(lsd, nsd) > 0 else float("inf"))
        scalars[name] = {
            "lego_mean": float(np.mean(lv)), "nemo_mean": float(np.mean(nv)),
            "lego_spread": lsd, "nemo_spread": nsd, "gap": gap,
            "floor": floor, "R": abs(gap) / floor if floor > 0 else None,
            "R_lo": float(r_lo), "R_hi": float(r_hi),
            "bootstrap_bad_floor_fraction": float(zero_fraction[column]),
            "quantum": q, "quantized_reasons": quantized_reasons,
            "side_spread": "ONE_SIDED" if imbalance > 10.0 else "BALANCED",
            "verdict": verdict,
        }

    if any(row["verdict"] == "REFUTE" for row in scalars.values()):
        family_verdict = "REFUTE"
    elif any(row["verdict"].startswith("UNRESOLVED")
             for row in scalars.values()):
        family_verdict = "UNRESOLVED"
    else:
        maximum = np.nanmax(r_b[:, eligible_columns], axis=1)
        family_hi = float(np.percentile(maximum[np.isfinite(maximum)], 97.5))
        family_verdict = "CONFIRM" if family_hi <= R_BAR else "UNRESOLVED"
    maximum = (np.nanmax(r_b[:, eligible_columns], axis=1)
               if eligible_columns else np.asarray([], dtype=np.float64))
    family_hi = (float(np.percentile(maximum[np.isfinite(maximum)], 97.5))
                 if np.isfinite(maximum).any() else None)
    return {
        "verdict": family_verdict,
        "bootstrap_max_R_hi": family_hi,
        "scalars": scalars,
    }


def classifier_self_test() -> dict[str, Any]:
    """Planted controls for all scalar classifier branches and quantization."""
    base = np.asarray([-2.2, -1.1, -0.3, 0.4, 1.2, 2.5])
    cases = {
        "confirm": (base, base + 0.02, "CONFIRM"),
        "refute": (base + 12.0, base, "REFUTE"),
        "unresolved": (base + 2.0, base, "UNRESOLVED"),
        "quantized": (np.ones(6), np.ones(6), "UNRESOLVED_QUANTIZED"),
    }
    receipt = {}
    for index, (name, (lego, nemo, expected)) in enumerate(cases.items()):
        result = classify_family(
            {"plant": lego}, {"plant": nemo}, {"plant": 0.0},
            rng=np.random.default_rng(BOOTSTRAP_SEED + index))
        got = result["scalars"]["plant"]["verdict"]
        if got != expected:
            raise RuntimeError(
                f"classifier plant {name} expected {expected}, got {got}")
        receipt[name] = got
    # The simultaneous path is exercised with two individually confirming
    # columns; the recorded maximum must exist and the family must confirm.
    simultaneous = classify_family(
        {"a": base, "b": base[::-1]},
        {"a": base + 0.01, "b": base[::-1] - 0.01},
        {"a": 0.0, "b": 0.0},
        rng=np.random.default_rng(BOOTSTRAP_SEED + 99))
    if (simultaneous["verdict"] != "CONFIRM"
            or simultaneous["bootstrap_max_R_hi"] is None):
        raise RuntimeError("simultaneous-family maximum plant did not fire")
    receipt["family_maximum"] = "FIRED"
    return receipt


def run(args: argparse.Namespace) -> int:
    plants = classifier_self_test()
    lego, lq, lego_receipts = load_ensemble(args.lego_root, "legoESM")
    nemo, nq, nemo_receipts = load_ensemble(args.nemo_root, "NEMO")
    quantum = {key: max(lq.get(key, 0.0), nq.get(key, 0.0))
               for key in set(lq) | set(nq)}
    families = {}
    for index, family in enumerate(FAMILIES):
        families[family] = classify_family(
            lego[family], nemo[family], quantum,
            rng=np.random.default_rng(BOOTSTRAP_SEED + index))
    verdicts = [families[family]["verdict"] for family in FAMILIES]
    if all(verdict == "CONFIRM" for verdict in verdicts):
        capstone = "STANDALONE_STATISTICALLY_INDISTINGUISHABLE_AT_20Y"
    elif any(verdict == "REFUTE" for verdict in verdicts):
        capstone = "STANDALONE_DISTINGUISHABLE_AT_20Y"
    else:
        capstone = "STANDALONE_UNRESOLVED_AT_20Y"
    result = {
        "schema": SCHEMA,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "R_bar": R_BAR,
        "classifier_plants": plants,
        "lego_receipts": lego_receipts,
        "nemo_receipts": nemo_receipts,
        "families": families,
        "verdict": capstone,
        "scope": "20-year horizon-specific from-rest climate distribution",
        "equilibrium_claim": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"T1_VERDICT={capstone}")
    print(f"OUTPUT={args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lego-root", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(classifier_self_test(), sort_keys=True))
        return 0
    if args.output.exists():
        parser.error(f"refusing existing output {args.output}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

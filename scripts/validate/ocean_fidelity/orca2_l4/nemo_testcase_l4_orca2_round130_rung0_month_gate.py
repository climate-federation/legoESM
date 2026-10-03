#!/usr/bin/env python3
"""Score the independent ORCA2 hierarchy rung-0 240-step month."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round71_independent_month_gate as prior_month,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_record_gate as record_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)


STEPS = 240
FIELDS = ("T", "S", "u", "v", "ssh")
FIELD_UNITS = {"T": "degC", "S": "g/kg", "u": "m/s", "v": "m/s", "ssh": "m"}
EXPECTED_ADMISSION_SHA256 = "3df42be0f9b185706cc615a1fa0320bcd19968e8de35bdfe64073bb185303d85"
EXPECTED_RMS_ORDER = ("T", "ssh", "S", "u", "v")
EXPECTED_MAX_ORDER = ("T", "S", "ssh", "u", "v")
PLANTS = ("none", "terminal-ulp", "terminal-nonfinite")


class GateError(RuntimeError):
    """The rung-0 month record or measurement violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_admission(path: Path) -> dict[str, object]:
    """Bind the terminal record to round 83's admitted binary and decks."""

    require(path.is_file(), f"missing rung-0 admission: {path}")
    observed = sha256(path)
    require(observed == EXPECTED_ADMISSION_SHA256,
            "round-83 rung-0 admission digest changed")
    report = json.loads(path.read_text())
    require(report.get("status") == "PASS_RUNG0_RECORD",
            "round-83 rung-0 record is not admitted")
    require(report.get("claim_label") == "independent",
            "round-83 record is not labelled independent")
    require(report.get("binary_sha256") == record_gate.BINARY_SHA256,
            "round-83 NEMO binary identity changed")
    require(report.get("month_step") == STEPS,
            "round-83 terminal record is not kt=240")
    require(report.get("ten_step_restart_shards") == 40
            and report.get("ten_step_twin_field_comparisons") == 100,
            "round-83 calibration census changed")
    decks = report.get("decks", [])
    require(len(decks) == 3 and decks[-1].get("steps") == STEPS
            and decks[-1].get("stock") == STEPS
            and decks[-1].get("restart_mode") == "periodic",
            "round-83 month deck provenance changed")
    return {
        "path": str(path),
        "sha256": observed,
        "status": report["status"],
        "binary_sha256": report["binary_sha256"],
        "month_step": report["month_step"],
    }


def terminal_ledger_rows(path: Path, month_root: Path) -> dict[str, str]:
    """Select the two terminal shards by full path from the cumulative ledger."""

    require(path.is_file(), f"missing cumulative restart ledger: {path}")
    wanted = {
        f"ORCA2_{STEPS:08d}_restart_{rank:04d}.nc" for rank in (0, 1)
    }
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                f"malformed cumulative SHA-256 row: {line!r}")
        target = Path(fields[1].lstrip("* "))
        if target.parent == month_root and target.name in wanted:
            require(target.name not in rows,
                    f"duplicate month terminal row: {target.name}")
            rows[target.name] = fields[0]
    require(set(rows) == wanted,
            f"terminal ledger coverage changed: {sorted(rows)}")
    return rows


def first_nonfinite(fields: dict[str, np.ndarray]) -> dict[str, object] | None:
    """Return the first non-finite value in the frozen field/index order."""

    for name in FIELDS:
        values = np.asarray(fields[name])
        locations = np.argwhere(~np.isfinite(values))
        if locations.size:
            index = tuple(int(value) for value in locations[0])
            return {"field": name, "index": list(index),
                    "value": float(values[index])}
    return None


def score_field(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, object]:
    """Score one terminal field without hiding non-finite values."""

    candidate = np.asarray(candidate)
    oracle = np.asarray(oracle)
    require(candidate.shape == oracle.shape,
            f"terminal shape mismatch: {candidate.shape} != {oracle.shape}")
    require(candidate.dtype == oracle.dtype == np.dtype(np.float64),
            "terminal score is not fp64")
    require(bool(np.isfinite(candidate).all() and np.isfinite(oracle).all()),
            "terminal score contains non-finite values")
    candidate_bits = np.ascontiguousarray(candidate).view(np.uint64)
    oracle_bits = np.ascontiguousarray(oracle).view(np.uint64)
    unequal_mask = candidate_bits != oracle_bits
    delta = candidate - oracle
    absolute = np.abs(delta)
    argmax = tuple(int(value) for value in np.unravel_index(
        int(np.argmax(absolute)), absolute.shape))
    unequal = int(np.count_nonzero(unequal_mask))
    return {
        "bit_identical": unequal == 0,
        "unequal": unequal,
        "count": int(candidate.size),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(absolute[argmax]),
        "argmax_jik": list(argmax),
        "candidate_at_argmax": float(candidate[argmax]),
        "oracle_at_argmax": float(oracle[argmax]),
    }


def rank_rows(rows: dict[str, dict[str, object]], metric: str) -> list[dict[str, object]]:
    require(metric in {"rms", "max_abs"}, f"unknown ranking metric {metric}")
    return sorted(
        ({"field": name, "units": FIELD_UNITS[name], **rows[name]} for name in FIELDS),
        key=lambda row: (-float(row[metric]), FIELDS.index(str(row["field"]))),
    )


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    """Apply frozen protocol predicates and retain prediction refutations."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "terminal-ulp":
        report["terminal_restart"]["files"][0]["sha256"] = "0" * 64
    elif plant == "terminal-nonfinite":
        report["terminal"]["rows"]["T"]["max_abs"] = float("inf")

    require(report.get("claim_label") == "independent",
            "month claim is not independent")
    require(report.get("initial_mode") == "card_own_state"
            and report.get("decision52_bridge") is None,
            "rung-0 month used a recorded-entry bridge")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "rung-0 execution policy changed")
    require(report.get("steps_completed") == STEPS,
            "rung-0 month did not complete 240 steps")
    require(report.get("first_nonfinite") is None,
            "rung-0 month contains a non-finite state")
    require(report.get("unmeasured_features") == ["linear_implicit_bottom_drag"],
            "rung-0 unmeasured-feature registry changed")
    require(report["record_admission"].get("status") == "PASS_RUNG0_RECORD",
            "rung-0 record admission is not PASS")
    require(report["terminal_restart"].get("status") == "BIT_EXACT_ORIENTATION",
            "terminal restart orientation is not admitted")
    for row in report["terminal_restart"]["files"]:
        require(row.get("sha256") != "0" * 64,
                "terminal restart one-ULP/digest plant fired")

    rows = report["terminal"]["rows"]
    require(tuple(rows) == FIELDS, "terminal field registry changed")
    for name in FIELDS:
        row = rows[name]
        require(int(row["count"]) > 0
                and 0 <= int(row["unequal"]) <= int(row["count"]),
                f"{name}: terminal census is invalid")
        require(bool(row["bit_identical"]) == (int(row["unequal"]) == 0),
                f"{name}: bit flag/count disagreement")
        require(math.isfinite(float(row["rms"]))
                and math.isfinite(float(row["max_abs"])),
                f"{name}: terminal score is non-finite")
    rms_order = tuple(row["field"] for row in report["terminal"]["ranking_by_rms"])
    max_order = tuple(row["field"] for row in report["terminal"]["ranking_by_max_abs"])
    require(set(rms_order) == set(FIELDS) and set(max_order) == set(FIELDS),
            "terminal ranking is incomplete")
    predictions = {
        "all_five_non_bit": {
            "status": "CONFIRMED" if all(not rows[name]["bit_identical"] for name in FIELDS)
            else "REFUTED",
            "observed_bit_identical": [name for name in FIELDS if rows[name]["bit_identical"]],
        },
        "rms_order": {
            "status": "CONFIRMED" if rms_order == EXPECTED_RMS_ORDER else "REFUTED",
            "predicted": list(EXPECTED_RMS_ORDER),
            "observed": list(rms_order),
        },
        "max_order": {
            "status": "CONFIRMED" if max_order == EXPECTED_MAX_ORDER else "REFUTED",
            "predicted": list(EXPECTED_MAX_ORDER),
            "observed": list(max_order),
        },
    }
    return {**report, "status": "PASS_RUNG0_INDEPENDENT_MONTH",
            "prediction_ledger": predictions}


def run_month(
    deck_root: Path,
    month_root: Path,
    restart_ledger: Path,
    admission_path: Path,
    expect_commit: str,
    *,
    progress_interval: int = 20,
) -> dict[str, object]:
    """Advance the rung-0 card from its own state and score kt=240."""

    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    require(progress_interval >= 1, "progress interval must be positive")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-130 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-130 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "rung-0 month requires production JIT on CPU")

    admission = validate_admission(admission_path)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    require(card.n_steps == STEPS, "rung-0 card month length changed")
    ledger_rows = terminal_ledger_rows(restart_ledger, month_root)
    with tempfile.TemporaryDirectory(prefix="orca2-r130-ledger-") as temporary:
        filtered_ledger = Path(temporary) / "terminal.sha256"
        filtered_ledger.write_text("".join(
            f"{digest}  {name}\n" for name, digest in sorted(ledger_rows.items())))
        oracle, restart = prior_month.read_terminal_restart(
            month_root, filtered_ledger, card)
    restart["cumulative_ledger"] = {
        "path": str(restart_ledger),
        "sha256": sha256(restart_ledger),
        "selected_rows": len(ledger_rows),
    }

    state = card.recipe.initial_state
    initial_fields = rung0.candidate_fields(state)
    require(first_nonfinite(initial_fields) is None,
            "rung-0 initial state is non-finite")
    freshwater, surface = rung0_ladder._zero_forcing(
        tuple(np.asarray(state.eta.data).shape))
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    started = time.time()
    for step in range(1, STEPS + 1):
        state = jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        nonfinite = first_nonfinite(rung0.candidate_fields(state))
        require(nonfinite is None,
                f"first non-finite step={step} field={nonfinite}")
        if step % progress_interval == 0:
            print(f"MONTH_PROGRESS step={step}/{STEPS} "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    candidate = rung0.candidate_fields(state)
    rows = {name: score_field(candidate[name], oracle[name]) for name in FIELDS}
    return {
        "format": "nemo-testcase-l4-orca2-round130-rung0-month-v1",
        "status": "MEASURED_RUNG0_INDEPENDENT_MONTH",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed": STEPS,
        "dt_s": float(card.dt_s),
        "wall_seconds": time.time() - started,
        "first_nonfinite": None,
        "unmeasured_features": list(card.unmeasured_features),
        "record_admission": admission,
        "terminal_restart": restart,
        "terminal": {
            "rows": rows,
            "ranking_by_rms": rank_rows(rows, "rms"),
            "ranking_by_max_abs": rank_rows(rows, "max_abs"),
        },
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--month-root", type=Path)
    parser.add_argument("--restart-ledger", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-interval", type=int, default=20)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.plant != "none", "classification mode requires a plant")
            require(not any((args.deck_root, args.month_root, args.restart_ledger,
                             args.admission, args.expect_commit)),
                    "classification mode cannot take run inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.month_root, args.restart_ledger,
                         args.admission, args.expect_commit)),
                    "run mode requires every record input and commit stamp")
            raw = run_month(
                args.deck_root, args.month_root, args.restart_ledger,
                args.admission, args.expect_commit,
                progress_interval=args.progress_interval)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, prior_month.GateError, rung0.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_RUNG0_INDEPENDENT_MONTH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

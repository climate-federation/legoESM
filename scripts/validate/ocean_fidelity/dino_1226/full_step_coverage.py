#!/usr/bin/env python
"""Fail-closed full-step CALL coverage gate for NEMO DINO ``stp_MLF``.

The oracle CALL list is parsed on every invocation. Dispatch wrappers are
replaced by the concrete DINO-selected routine(s) one level down. A VERIFIED
row is accepted only after its committed receipt artifact checksum and receipt
fields pass validation; campaign prose is intentionally not evidence here.

Exit 1 while any active physical row is UNMEASURED, or on source/registry/
receipt drift. ``--self-test`` proves checksum and registration controls fire.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORACLE = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
REGISTRY_JSON = REPO / "docs/ocean/fidelity/dino_full_step_coverage_registry.json"

sys.path.insert(0, str(HERE))
import full_step_coverage_registry as reg  # noqa: E402


def _load_parser():
    path = HERE / "gen_step_wiring.py"
    spec = importlib.util.spec_from_file_location("dino_step_wiring_parser", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.ORACLE = str(ORACLE)
    module.STPMLF = str(ORACLE / "MY_SRC/stpmlf.F90")
    module.CPP_FCM = str(ORACLE / "cpp_DINO.fcm")
    module.OCEAN_OUTPUT = str(ORACLE / reg.RUN_REL / "ocean.output")
    return module


@dataclass(frozen=True)
class Row:
    order: str
    parent_call: str
    routine: str
    nemo_source: str
    legoesm_symbol: str
    disposition: str
    resolution: str
    receipt: dict | None = None
    leverage_rank: int | None = None
    leverage_reason: str | None = None


ALLOWED = {"VERIFIED", "WAIVED", "INACTIVE", "UNMEASURED"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> list[str]:
    errors = []
    for rel, expected in reg.SOURCE_SHA256.items():
        path = ORACLE / rel
        if not path.is_file():
            errors.append(f"missing oracle source: {path}")
        elif _sha(path) != expected:
            errors.append(f"source sha mismatch: {rel}: {_sha(path)} != {expected}")
    return errors


def _nested(data, pointer):
    for part in pointer.split("."):
        data = data[part]
    return data


def _receipt(receipt_id: str, expected_sha: str | None = None) -> tuple[dict | None, str | None]:
    path = REPO / reg.RECEIPT_ARTIFACT
    expected_sha = expected_sha or reg.RECEIPT_ARTIFACT_SHA256
    if not path.is_file():
        return None, f"receipt artifact missing: {reg.RECEIPT_ARTIFACT}"
    actual = _sha(path)
    if actual != expected_sha:
        return None, f"receipt sha mismatch: {actual} != {expected_sha}"
    data = json.loads(path.read_text())
    item = data.get("receipts", {}).get(receipt_id)
    if not item:
        return None, f"receipt id missing: {receipt_id}"
    required = ("measured_number", "bar", "pass")
    missing = [key for key in required if key not in item]
    if missing or item.get("pass") is not True:
        return None, f"invalid receipt {receipt_id}: missing={missing}, pass={item.get('pass')}"
    if data.get("precision") != "fp64" or not data.get("producer_sha"):
        return None, f"invalid receipt provenance for {receipt_id}"
    source = data["source_artifact"]
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{source['commit']}:{source['path']}"], cwd=REPO
        )
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{data['producer_sha']}:{data['probe_path']}"],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        return None, f"committed source receipt/probe unavailable: {exc}"
    source_actual = hashlib.sha256(raw).hexdigest()
    if source_actual != source["sha256"]:
        return None, f"source artifact sha mismatch: {source_actual} != {source['sha256']}"
    source_data = json.loads(raw)
    if source_data.get("probe_commit_sha") != data["producer_sha"]:
        return None, "source producer SHA does not match compact receipt"
    if source_data.get("checked_out_parent_sha") != data["producer_sha"]:
        return None, "source checked-out SHA does not match compact receipt"
    if source_data.get("fp64") is not True or data["precision"] != "fp64":
        return None, "source/compact precision is not fp64"
    if source_data.get("cpu_only") is not True or data.get("backend") != "cpu":
        return None, "source/compact backend is not CPU"
    if source_data.get("bars", {}).get("pointwise_column") != 1e-15:
        return None, "source pointwise bar is not 1e-15"
    for pointer, expected in reg.SOURCE_ASSERTIONS[receipt_id]:
        observed = _nested(source_data, pointer)
        if observed != expected:
            return None, f"source receipt mismatch at {pointer}: {observed!r} != {expected!r}"
    return {
        "probe_path": data["probe_path"],
        "artifact_path": reg.RECEIPT_ARTIFACT,
        "artifact_sha256": actual,
        "source_artifact_sha256": data["source_artifact"]["sha256"],
        "measured_number": item["measured_number"],
        "bar": item["bar"],
        "precision": data["precision"],
        "producer_sha": data["producer_sha"],
        "source_row": item.get("source_row", item.get("source_rows")),
    }, None


def _registered_parent_calls() -> set[str]:
    if not REGISTRY_JSON.is_file():
        return set()
    data = json.loads(REGISTRY_JSON.read_text())
    return {row["parent_call"] for row in data.get("rows", [])}


def _quote_resolution(reason: str) -> str:
    """Attach the exact resolved-output/cpp line to parser-derived reasons."""
    if "(ocean.output)" in reason:
        match = re.search(r"\b([A-Za-z_]\w*)=([TF])\b", reason)
        if match:
            name, value = match.groups()
            output = ORACLE / reg.RUN_REL / "ocean.output"
            for line_no, line in enumerate(output.read_text().splitlines(), 1):
                if re.search(rf"\b{re.escape(name)}\s*=\s*{value}\b", line):
                    quote = line.strip()
                    return f"{reason}; {reg.RUN_REL}/ocean.output:{line_no} `{quote}`"
    if "cpp_DINO.fcm" in reason and "cpp_DINO.fcm:" not in reason:
        quote = (ORACLE / "cpp_DINO.fcm").read_text().splitlines()[0].strip()
        return f"{reason}; cpp_DINO.fcm:1 `{quote}`"
    return reason


def build_rows(calls=None, *, validate_registration=True) -> tuple[list[Row], list[str]]:
    parser = _load_parser()
    calls = list(calls if calls is not None else parser.parse_stpmlf(parser.STPMLF))
    errors = []
    if len(calls) != reg.EXPECTED_DIRECT_CALLS:
        errors.append(
            f"unregistered call count: parsed {len(calls)}, expected {reg.EXPECTED_DIRECT_CALLS}"
        )
    if validate_registration:
        expected = _registered_parent_calls()
        actual = {f"stpmlf.F90:{call.line} CALL {call.name}" for call in calls}
        for item in sorted(actual - expected):
            errors.append(f"unregistered call: {item}")
        for item in sorted(expected - actual):
            errors.append(f"registered call missing from source: {item}")
    rows = []
    for ordinal, call in enumerate(calls, 1):
        key = (call.line, call.name)
        activity, reason = call.live, call.live_reason
        if activity == "UNRESOLVED":
            if key not in reg.RESOLVED:
                errors.append(
                    f"unregistered unresolved call: stpmlf.F90:{call.line} CALL {call.name}"
                )
                continue
            activity, reason = reg.RESOLVED[key]
        elif activity == "DEAD":
            activity = "INACTIVE"
        elif activity == "LIVE":
            activity = "ACTIVE"

        targets = reg.DISPATCH.get(key)
        if targets is None:
            targets = [(call.name, f"MY_SRC/stpmlf.F90:{call.line}", reason)]
        for subordinal, target in enumerate(targets, 1):
            routine, source, dispatch_reason = target[:3]
            target_activity = target[3] if len(target) == 4 else activity
            order = f"{ordinal:03d}" + (f".{subordinal}" if len(targets) > 1 else "")
            if target_activity == "INACTIVE":
                disposition = "INACTIVE"
            elif target_activity == "WAIVED" or call.name in reg.WAIVED_NAMES:
                disposition = "WAIVED"
            elif routine not in reg.LEGO:
                errors.append(f"unregistered active call: {source} CALL {routine}")
                continue
            else:
                disposition = "UNMEASURED"

            receipt = None
            receipt_id = reg.RECEIPTS.get((call.line, routine))
            if receipt_id:
                receipt, receipt_error = _receipt(receipt_id)
                if receipt_error:
                    errors.append(f"{source} {routine}: {receipt_error}")
                else:
                    disposition = "VERIFIED"
            rank_reason = reg.LEVERAGE.get(routine)
            rows.append(
                Row(
                    order=order,
                    parent_call=f"stpmlf.F90:{call.line} CALL {call.name}",
                    routine=routine,
                    nemo_source=source,
                    legoesm_symbol=reg.LEGO.get(routine, "MISSING"),
                    disposition=disposition,
                    resolution=_quote_resolution(
                        reason if targets[0][0] == call.name else dispatch_reason
                    ),
                    receipt=receipt,
                    leverage_rank=rank_reason[0]
                    if disposition == "UNMEASURED" and rank_reason
                    else None,
                    leverage_reason=rank_reason[1]
                    if disposition == "UNMEASURED" and rank_reason
                    else None,
                )
            )
    bad = [row for row in rows if row.disposition not in ALLOWED]
    if bad:
        errors.append(f"invalid dispositions: {bad}")
    return rows, errors


def _print(rows: list[Row], errors: list[str]) -> None:
    counts = {kind: sum(r.disposition == kind for r in rows) for kind in sorted(ALLOWED)}
    claim = counts["VERIFIED"] + counts["UNMEASURED"]
    fraction = counts["VERIFIED"] / claim if claim else 1.0
    print(f"FULL-STEP COVERAGE: {counts['VERIFIED']}/{claim} = {fraction:.1%} VERIFIED")
    print("denominator excludes WAIVED and INACTIVE rows")
    print("counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if errors:
        print("STRUCTURAL ERRORS:")
        for error in errors:
            print(f"  ERROR {error}")
    unmeasured = [r for r in rows if r.disposition == "UNMEASURED"]
    unmeasured.sort(
        key=lambda r: (
            r.leverage_rank if r.leverage_rank is not None else 10_000,
            r.order,
        )
    )
    print(f"RANKED UNMEASURED ({len(unmeasured)}; leverage is stated planning judgment):")
    for fallback, row in enumerate(unmeasured, 1):
        rank = row.leverage_rank if row.leverage_rank is not None else 100 + fallback
        reason = row.leverage_reason or "active physical/state call; leverage not yet isolated"
        print(f"  {rank:03d} {row.order} {row.routine} [{row.nemo_source}] -- {reason}")


def self_test() -> int:
    failures = []
    _, error = _receipt("zdf_sh2", expected_sha="0" * 64)
    if error is None or "sha mismatch" not in error:
        failures.append("planted fake receipt SHA was accepted")
    else:
        print("CONTROL PASS: planted fake receipt rejected (sha mismatch)")

    parser = _load_parser()
    calls = parser.parse_stpmlf(parser.STPMLF)
    fake = replace(calls[50], line=9999, name="eos", live="LIVE", live_reason="planted")
    calls[50] = fake
    _, errors = build_rows(calls)
    if not any("unregistered" in item and "9999 CALL eos" in item for item in errors):
        failures.append("planted unregistered CALL was accepted")
    else:
        print("CONTROL PASS: planted unregistered CALL rejected")
    if failures:
        for failure in failures:
            print(f"CONTROL FAIL: {failure}")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json", action="store_true", help="emit the complete machine-readable registry"
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--write-registry", action="store_true")
    ns = ap.parse_args(argv)
    if ns.self_test:
        return self_test()
    rows, errors = build_rows(validate_registration=not ns.write_registry)
    errors = verify_sources() + errors
    payload = {
        "schema": "dino-full-step-coverage-v1",
        "rows": [asdict(r) for r in rows],
        "errors": errors,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if ns.write_registry:
        if errors:
            _print(rows, errors)
            return 1
        REGISTRY_JSON.write_text(rendered)
        print(f"wrote {REGISTRY_JSON.relative_to(REPO)} ({len(rows)} rows)")
        return 0
    if not REGISTRY_JSON.is_file() or REGISTRY_JSON.read_text() != rendered:
        errors.append("committed registry JSON is stale; run --write-registry")
        payload["errors"] = errors
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if ns.json:
        print(rendered, end="")
    else:
        _print(rows, errors)
    return 1 if errors or any(r.disposition == "UNMEASURED" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

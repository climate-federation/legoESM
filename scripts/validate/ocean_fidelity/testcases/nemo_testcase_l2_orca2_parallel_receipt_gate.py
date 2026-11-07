#!/usr/bin/env python3
"""Pin every file:line citation in the ORCA2 parallel inventory receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legoesm.ocean.fidelity.provenance import worktree_stamp

REPO = Path(__file__).resolve().parents[4]
EVIDENCE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2")
PHASE3 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
NEMO_TARGET = NEMO / "cfgs/ORCA2_OMIP_L4"
RUNS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/runs")
RUN_A = RUNS / "variant_icebergs_off_phase2v_tke_a_10step_np2"
RUN_B = RUNS / "variant_icebergs_off_phase2v_tke_b_10step_np2"
HISTORICAL_REVISION = "b7ce08cc8afa5cf377922abf198cf1794fab8a73"
DEFAULT_RECEIPT = (
    REPO / "docs/ocean/fidelity/testcases" / "nemo_testcases_l2_orca2_parallel_inventory_receipt.md"
)
DEFAULT_MANIFEST = (
    REPO
    / "scripts/validate/ocean_fidelity/testcases/manifests"
    / "nemo_testcase_l2_orca2_parallel_citations.json"
)


@dataclass(frozen=True)
class GitBlob:
    revision: str
    path: str


SOURCES: dict[str, Path | GitBlob] = {
    "round101_prereg.md": (
        REPO / "docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round101.md"
    ),
    "round101_tke_statement_boundary_receipt.md": (
        REPO
        / "docs/ocean/fidelity/testcases"
        / "nemo_testcases_l2_gyre_round101_tke_statement_boundary_receipt.md"
    ),
    "nemo_testcase_recipe.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"
    ),
    "recipes.py": REPO / "packages/ocean/legoesm/ocean/recipes.py",
    "nemo_recipe.py": REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py",
    "run_omip.py": REPO / "scripts/run/run_omip.py",
    "build_core2_nyf_zarr.py": REPO / "scripts/data/build_core2_nyf_zarr.py",
    "core2.py": REPO / "packages/ocean/legoesm/ocean/forcing/core2.py",
    "prepare_omip_forcing.py": REPO / "scripts/data/prepare_omip_forcing.py",
    "nemo_native_fields.py": (
        REPO / "packages/ocean/legoesm/ocean/forcing/nemo_native_fields.py"
    ),
    "vertical.py": REPO / "packages/ocean/legoesm/ocean/vertical.py",
    "tripole.py": REPO / "packages/core/legoesm/grids/tripole.py",
    "nemo_testcase_l2_orca2_parallel_inventory.py": (
        REPO
        / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_orca2_parallel_inventory.py"
    ),
    "test_nemo_testcase_l2_orca2_parallel_inventory.py": (
        REPO
        / "tests/ocean/fidelity"
        / "test_nemo_testcase_l2_orca2_parallel_inventory.py"
    ),
    "test_nemo_testcase_l2_orca2_parallel_receipt_gate.py": (
        REPO
        / "tests/ocean/fidelity"
        / "test_nemo_testcase_l2_orca2_parallel_receipt_gate.py"
    ),
    "orca2_boundary_run.sh": (
        REPO
        / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition/run.sh"
    ),
    "orca2_boundary_writer.F90": (
        REPO
        / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "l2_orca2_tke_boundary.F90"
    ),
    "orca2_boundary.patch": (
        REPO
        / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition"
        / "zdftke_orca2_boundary.patch"
    ),
    "historical_nemo_testcase_recipe.py": GitBlob(
        HISTORICAL_REVISION,
        "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
    ),
    "historical_nemo_fld_read.py": GitBlob(
        HISTORICAL_REVISION,
        "packages/ocean/legoesm/ocean/forcing/nemo_fld_read.py",
    ),
    "phase2w_handoff_receipt.md": GitBlob(
        HISTORICAL_REVISION,
        "docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2w_handoff_receipt.md",
    ),
    "phase2v_tke_walk_writer.patch": GitBlob(
        HISTORICAL_REVISION,
        "scripts/validate/ocean_fidelity/orca2_l4/phase2v_tke_walk_writer.patch",
    ),
    "cpp_ORCA2_OMIP_L4.fcm": NEMO_TARGET / "cpp_ORCA2_OMIP_L4.fcm",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90": (NEMO_TARGET / "BLD/ppsrc/nemo/stprk3.f90"),
    "ORCA2_OMIP_L4/MY_SRC/stprk3_stg.F90": (NEMO_TARGET / "MY_SRC/stprk3_stg.F90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90": (NEMO_TARGET / "BLD/ppsrc/nemo/zdftke.f90"),
    "ORCA2_OMIP_L4/EXP00/namelist_cfg": NEMO_TARGET / "EXP00/namelist_cfg",
    "orca2_inventory.json": EVIDENCE / "orca2_inventory.json",
    "orca2_inventory_plant.txt": EVIDENCE / "orca2_inventory_plant.txt",
    "orca2_inventory_missing_boundary_plant.txt": (
        EVIDENCE / "orca2_inventory_missing_boundary_plant.txt"
    ),
    "orca2_inventory_provenance_mismatch_plant.txt": (
        EVIDENCE / "orca2_inventory_provenance_mismatch_plant.txt"
    ),
    "acquisition2.log": EVIDENCE / "acquisition2.log",
    "finalize_fix4.log": EVIDENCE / "finalize_fix4.log",
    "orca2_finalize_admitted.log": EVIDENCE / "orca2_finalize_admitted.log",
    "orca2_passivity_plant.txt": EVIDENCE / "orca2_passivity_plant.txt",
    "orca2_boundary_plant.sh": (
        REPO
        / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_orca2_tke_boundary_acquisition/passivity_plant.sh"
    ),
    "orca2_boundary_admission.json": (
        EVIDENCE
        / "oracle_phase2v_tke_boundary_np2"
        / "orca2_tke_boundary_admission.json"
    ),
    "orca2_citation_gate.json": EVIDENCE / "orca2_citation_gate.json",
    "orca2_citation_gate_plant.json": EVIDENCE / "orca2_citation_gate_plant.json",
    "orca2_citation_gate_summary.txt": EVIDENCE / "orca2_citation_gate_summary.txt",
    "orca2_final_tests.txt": EVIDENCE / "orca2_final_tests.txt",
    "orca2_review.txt": EVIDENCE / "orca2_review.txt",
    "phase2v_a_run.user.time.log": RUN_A / "run.user.time.log",
    "phase2v_b_run.user.time.log": RUN_B / "run.user.time.log",
    "phase2v_a_ocean.output": RUN_A / "ocean.output",
    "round24_orca2_entry_stage.json": (PHASE3 / "round24/crosscard/orca2_entry_stage.json"),
    "round24_orca2_production_w.json": (PHASE3 / "round24/crosscard/orca2_production_w.json"),
    "round27_orca2_hpg_model_arm.json": (PHASE3 / "round27/orca2/orca2_hpg_model_arm.json"),
}

SPAN = re.compile(r"`([^`\n]+)`")
FULL_CITATION = re.compile(r"^(?P<source>[^`]+):(?P<lines>\d[\d,\-]*)$")


class CitationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CitationError(message)


def _source_text(source: str) -> str:
    item = SOURCES.get(source)
    require(item is not None, f"unmapped source alias {source!r}")
    if isinstance(item, GitBlob):
        result = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{item.revision}:{item.path}"],
            check=False,
            capture_output=True,
            text=True,
        )
        require(result.returncode == 0, f"unreadable git source {source}: {result.stderr}")
        return result.stdout
    require(item.is_file(), f"unreadable citation source {source}: {item}")
    return item.read_text(errors="replace")


def _line_numbers(spec: str) -> tuple[int, ...]:
    numbers: list[int] = []
    for group in spec.split(","):
        if "-" in group:
            first_raw, last_raw = group.split("-", 1)
            first, last = int(first_raw), int(last_raw)
            require(last >= first, f"reversed citation range {group}")
            numbers.extend(range(first, last + 1))
        else:
            numbers.append(int(group))
    require(bool(numbers), f"empty citation range {spec!r}")
    return tuple(numbers)


def _shift_spec(spec: str, amount: int) -> str:
    groups = []
    for group in spec.split(","):
        if "-" in group:
            first_raw, last_raw = group.split("-", 1)
            groups.append(f"{int(first_raw) + amount}-{int(last_raw) + amount}")
        else:
            groups.append(str(int(group) + amount))
    return ",".join(groups)


def _line_payload(source: str, spec: str) -> bytes:
    body = _source_text(source).splitlines()
    numbers = _line_numbers(spec)
    require(
        all(1 <= number <= len(body) for number in numbers),
        f"{source}:{spec} outside 1..{len(body)}",
    )
    return ("\n".join(body[number - 1] for number in numbers) + "\n").encode()


def _digest(source: str, spec: str) -> str:
    return hashlib.sha256(_line_payload(source, spec)).hexdigest()


def extract(receipt: str) -> tuple[str, ...]:
    citations = []
    for match in SPAN.finditer(receipt):
        value = match.group(1).strip()
        if FULL_CITATION.fullmatch(value):
            citations.append(value)
    return tuple(dict.fromkeys(citations))


def run(receipt: Path, manifest: Path, plant: str | None = None) -> dict[str, Any]:
    expected = json.loads(manifest.read_text())
    citations = extract(receipt.read_text())
    citation_set = set(citations)
    expected_set = set(expected)
    rows = []
    for citation in citations:
        match = FULL_CITATION.fullmatch(citation)
        assert match is not None
        source, spec = match.group("source"), match.group("lines")
        actual_spec = _shift_spec(spec, 1) if citation == plant else spec
        try:
            actual = _digest(source, actual_spec)
            status = "PASS" if actual == expected.get(citation) else "FAIL"
            detail = "pinned line payload" if status == "PASS" else "line payload digest moved"
        except CitationError as exc:
            actual = None
            status = "FAIL"
            detail = str(exc)
        rows.append(
            {
                "citation": citation,
                "detail": detail,
                "expected_sha256": expected.get(citation),
                "observed_sha256": actual,
                "status": status,
            }
        )
    failures = [row for row in rows if row["status"] != "PASS"]
    unmapped = sorted(citation_set - expected_set)
    unused = sorted(expected_set - citation_set)
    status = "PASS" if not failures and not unmapped and not unused else "FAIL"
    return {
        "citations": rows,
        "citations_found": len(citations),
        "failures": failures,
        "format": "nemo-testcase-l2-orca2-parallel-citation-gate-v1",
        "worktree": worktree_stamp(),
        "manifest": str(manifest),
        "plant": plant,
        "receipt": str(receipt),
        "status": status,
        "unmapped": unmapped,
        "unused_manifest_entries": unused,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", metavar="CITATION")
    args = parser.parse_args()
    report = run(args.receipt, args.manifest, args.plant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if report["status"] == "PASS":
        if args.plant:
            print("REFUSE: shifted citation plant did not fire", file=sys.stderr)
            return 2
        return 0
    print("REFUSE: ORCA2 receipt citation gate failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared fail-closed checks for the round-339 operator-only NEMO tools."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round328_dynzdf_gate as r328,
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_program(
    new_bld: Path,
    reference_bld: Path,
    observer_patch: Path,
    *,
    observer_unit: str,
    writer_unit: str,
    new_cfg: str,
    reference_cfg: str,
) -> dict:
    """Reuse round 328's generic observer-only whole-program proof."""
    return r328.same_program_observers(
        new_bld,
        reference_bld,
        {observer_unit: observer_patch},
        (writer_unit,),
        new_cfg=new_cfg,
        ref_cfg=reference_cfg,
    )


def same_program_no_observer(new_bld: Path, reference_bld: Path, *,
                             new_cfg: str, reference_cfg: str) -> dict:
    """Reuse round 328's whole-program proof for an observer-free rebuild."""
    return r328.same_program_observers(
        new_bld, reference_bld, {}, (), new_cfg=new_cfg, ref_cfg=reference_cfg)


def namelist_groups(text: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            current = line[1:].split()[0].lower()
            groups.setdefault(current, set())
            continue
        if line.startswith("/"):
            current = None
            continue
        if current is None:
            continue
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*=", line)
        if match:
            groups[current].add(match.group(1).lower())
    return groups


def validate_group_membership(reference: Path, expected: dict[str, str]) -> None:
    groups = namelist_groups(reference.read_text(encoding="utf-8"))
    for variable, group in expected.items():
        variable = variable.lower()
        group = group.lower()
        require(group in groups, f"namelist group {group} is absent")
        owners = sorted(name for name, variables in groups.items() if variable in variables)
        require(owners == [group], f"{variable} belongs to {owners}, expected {group}")


def output_has(text: str, name: str, value: str) -> bool:
    pattern = rf"(^|\s){re.escape(name)}\s*=\s*{re.escape(value)}(\s|$)"
    return re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE) is not None


def validate_ocean_output(path: Path, expected: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    require("E R R O R" not in text, "ocean.output contains E R R O R")
    for name, value in expected.items():
        require(output_has(text, name, value), f"resolved {name} is not {value}")


def write_manifest(path: Path, *, producer_commit: str, binary: Path,
                   files: list[Path], root: Path) -> None:
    root = root.resolve()
    records: dict[str, str] = {}
    for item in files:
        resolved = item.resolve()
        require(resolved.is_relative_to(root), f"manifest path is outside run: {item}")
        records[resolved.relative_to(root).as_posix()] = sha256(resolved)
    payload = {
        "producer_commit": producer_commit,
        "binary_sha256": sha256(binary),
        "files": dict(sorted(records.items())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_manifest(path: Path, *, expected_commit: str | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    if expected_commit is not None:
        require(payload.get("producer_commit") == expected_commit,
                "manifest producer commit moved")
    files = payload.get("files")
    require(isinstance(files, dict) and files, "manifest file inventory is empty")
    for name, digest in files.items():
        candidate = (root / name).resolve()
        require(candidate.is_relative_to(root), f"manifest path escapes run: {name}")
        require(candidate.is_file(), f"manifest file is missing: {name}")
        require(sha256(candidate) == digest, f"manifest digest moved: {name}")
    return payload


_FP_MNEMONIC = re.compile(
    r"^(?:f(?:add|sub|mul|div|sqrt)|(?:add|sub|mul|div|sqrt|min|max)(?:s|p)[sd])"
)


def fp_instruction_census(path: Path) -> dict[str, int]:
    """Count scalar/vector floating arithmetic instructions per ELF function."""
    output = subprocess.run(
        ["objdump", "-drwC", "--no-show-raw-insn", str(path)],
        check=True, capture_output=True, text=True).stdout
    current: str | None = None
    counts: dict[str, int] = {}
    for line in output.splitlines():
        function = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if function:
            current = function.group(1)
            counts.setdefault(current, 0)
            continue
        instruction = re.match(r"^\s*[0-9a-f]+:\s+([A-Za-z0-9_.]+)", line)
        if current is not None and instruction and _FP_MNEMONIC.match(instruction.group(1)):
            counts[current] += 1
    return counts


def compare_fp_census(reference: Path, observed: Path, *, function_pattern: str) -> dict:
    left = fp_instruction_census(reference)
    right = fp_instruction_census(observed)
    names = sorted(name for name in left if re.search(function_pattern, name))
    require(names, f"no function matches {function_pattern!r}")
    for name in names:
        require(left[name] > 0, f"floating-point census is zero for {name}")
        require(name in right, f"observed object lacks function {name}")
        require(left[name] == right[name],
                f"floating-point census moved for {name}: {left[name]} -> {right[name]}")
    return {name: left[name] for name in names}

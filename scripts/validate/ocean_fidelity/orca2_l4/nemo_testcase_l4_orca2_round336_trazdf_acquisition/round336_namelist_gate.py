#!/usr/bin/env python3
"""Validate every staged NEMO namelist assignment against namelist_ref groups."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

GROUP = re.compile(r"^\s*&([A-Za-z][A-Za-z0-9_]*)", re.IGNORECASE)
ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)(?:\s*\([^=]*\))?\s*=", re.IGNORECASE
)


def assignments(path: Path) -> list[tuple[str, str, int]]:
    result: list[tuple[str, str, int]] = []
    group: str | None = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("!", 1)[0]
        match = GROUP.match(line)
        if match:
            group = match.group(1).lower()
            continue
        if line.strip().startswith("/"):
            group = None
            continue
        match = ASSIGNMENT.match(line)
        if match and group is not None:
            result.append((group, match.group(1).lower(), line_number))
    return result


def validate(staged: Path, reference: Path, plant: str | None = None) -> int:
    reference_rows = assignments(reference)
    staged_rows = assignments(staged)
    reference_groups: dict[str, set[str]] = {}
    for group, variable, _line in reference_rows:
        reference_groups.setdefault(group, set()).add(variable)
    if not staged_rows:
        raise ValueError("staged namelist contains no assignments")

    if plant == "misspelled":
        group, variable, line = staged_rows[0]
        staged_rows[0] = (group, f"{variable}_r336_typo", line)
    elif plant == "misplaced":
        group, variable, line = staged_rows[0]
        wrong = next((candidate for candidate, members in reference_groups.items()
                      if candidate != group and variable not in members), None)
        if wrong is None:
            raise ValueError("could not construct misplaced-variable plant")
        staged_rows[0] = (wrong, variable, line)

    errors = []
    for group, variable, line in staged_rows:
        if group not in reference_groups:
            errors.append(f"{staged}:{line}: group &{group} absent from {reference}")
        elif variable not in reference_groups[group]:
            owners = sorted(name for name, members in reference_groups.items()
                            if variable in members)
            owner_text = ",".join(f"&{name}" for name in owners) or "no reference group"
            errors.append(
                f"{staged}:{line}: {variable} is not a member of &{group}; "
                f"reference owner={owner_text}"
            )
    if errors:
        raise ValueError("\n".join(errors))
    print(f"R336_NAMELIST_GROUP_PASS variables={len(staged_rows)} groups={len(set(g for g, _, _ in staged_rows))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--plant", choices=("misspelled", "misplaced"))
    args = parser.parse_args()
    try:
        return validate(args.staged, args.reference, args.plant)
    except (OSError, ValueError) as error:
        print(f"REFUSE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

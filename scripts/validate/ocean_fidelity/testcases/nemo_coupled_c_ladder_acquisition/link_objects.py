#!/usr/bin/env python3
"""Resolve the active NEMO object closure from FCM ``*.done`` dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path


def active_objects(makefile: Path) -> list[str]:
    physical = makefile.read_text().splitlines()
    logical: list[str] = []
    current = ""
    for line in physical:
        current += line[:-1] + " " if line.endswith("\\") else line
        if not line.endswith("\\"):
            logical.append(current)
            current = ""
    dependencies: dict[str, tuple[str, ...]] = {}
    for line in logical:
        if line.startswith("\t") or ":" not in line or "=" in line.split(":", 1)[0]:
            continue
        targets, deps = line.split(":", 1)
        for target in targets.split():
            if target.endswith(".done"):
                dependencies[target] = tuple(deps.split())
    pending = ["nemogcm.done"]
    seen: set[str] = set()
    objects = {"nemo.o"}
    while pending:
        target = pending.pop()
        if target in seen:
            continue
        seen.add(target)
        if target not in dependencies:
            raise ValueError(f"STATUS PLANT-FIRED: unresolved FCM dependency {target}")
        for dep in dependencies[target]:
            if dep.endswith(".done"):
                pending.append(dep)
            elif dep.endswith(".o"):
                objects.add(dep)
    return sorted(objects)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("makefile", type=Path)
    parser.add_argument("--object-dir", type=Path)
    args = parser.parse_args()
    for name in active_objects(args.makefile):
        print(args.object_dir / name if args.object_dir else name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

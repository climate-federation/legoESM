#!/usr/bin/env python
"""Resolve the machine profile for the current host — the lesommer
``lego-detect-machine`` helper.

Each ``config/machines/<name>.yaml`` declares an ``machine.hostnames`` list of
fnmatch patterns; the first profile whose pattern matches ``socket.gethostname()``
wins, else ``default``.  Importable (``resolve_machine``) and a CLI::

    python scripts/experiment/lego_detect_machine.py            # print detected name
    python scripts/experiment/lego_detect_machine.py --dump     # print the resolved profile
    python scripts/experiment/lego_detect_machine.py --host levante1   # test a hostname
"""
from __future__ import annotations

import argparse
import fnmatch
import socket
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MACHINES_DIR = _REPO_ROOT / "config" / "machines"


def load_machine(name: str, machines_dir: Path | None = None) -> dict[str, Any]:
    """Load one machine profile by name; returns the inner ``machine:`` mapping."""
    machines_dir = machines_dir or _MACHINES_DIR
    path = machines_dir / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no machine profile {name!r} at {path}")
    doc = yaml.safe_load(path.read_text()) or {}
    return doc.get("machine", doc)


def resolve_machine(
    hostname: str | None = None,
    machines_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the machine profile matching ``hostname`` (default: this host).

    First profile whose ``hostnames`` fnmatch-patterns match wins; falls back to
    ``default``.  The ``default`` profile (empty ``hostnames``) never matches by
    pattern, so it is only ever the fallback — deterministic regardless of file
    iteration order.
    """
    machines_dir = machines_dir or _MACHINES_DIR
    host = hostname or socket.gethostname()
    candidates = sorted(p.stem for p in machines_dir.glob("*.yaml"))
    for name in candidates:
        if name == "default":
            continue
        prof = load_machine(name, machines_dir)
        for pat in prof.get("hostnames", []) or []:
            if fnmatch.fnmatch(host, pat):
                return prof
    return load_machine("default", machines_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=None, help="hostname to test (default: this host)")
    p.add_argument("--dump", action="store_true", help="print the full resolved profile")
    args = p.parse_args(argv)
    prof = resolve_machine(args.host)
    if args.dump:
        print(yaml.safe_dump({"machine": prof}, sort_keys=False), end="")
    else:
        print(prof.get("name", "default"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

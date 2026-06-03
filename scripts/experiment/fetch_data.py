#!/usr/bin/env python
"""Check / fetch the external datasets a template needs — the lesommer
``fetch_data.py`` (``check`` | ``fetch``).

A template's ``experiment.data`` lists dataset ids resolved against
``config/data_catalog.yaml``; each maps to a ``target`` path under the machine's
``data_root``.  Idealized templates list no data and report clean immediately.

    python scripts/experiment/fetch_data.py check 2d/williamson2_sw     # no data needed
    python scripts/experiment/fetch_data.py check coupled/amip          # reports present/missing
    python scripts/experiment/fetch_data.py fetch coupled/amip          # stage missing (where automatable)

``check`` exits non-zero if any required dataset is missing; ``fetch`` downloads
catalog entries that carry a ``url`` (gs:// via gsutil, http(s):// via curl) and
prints manual/credentialed instructions for the rest.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_DIR = _REPO_ROOT / "config" / "templates"
_CATALOG = _REPO_ROOT / "config" / "data_catalog.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lego_detect_machine import resolve_machine  # noqa: E402


def _load_catalog(catalog: Path | None = None) -> dict[str, Any]:
    catalog = catalog or _CATALOG
    return (yaml.safe_load(catalog.read_text()) or {}).get("datasets", {})


def _template_data_ids(template: str, templates_dir: Path | None = None) -> list[str]:
    templates_dir = templates_dir or _TEMPLATES_DIR
    rel = template[:-5] if template.endswith(".yaml") else template
    path = templates_dir / f"{rel}.yaml"
    if not path.is_file():
        raise SystemExit(f"ERROR: template {template!r} not found at {path}")
    doc = yaml.safe_load(path.read_text()) or {}
    return list((doc.get("experiment") or {}).get("data", []) or [])


def _data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(resolve_machine().get("data_root", "./data"))


def _status(ids: list[str], catalog: dict[str, Any], data_root: Path):
    """Yield (dataset_id, entry, target_path, present) for each required dataset."""
    for ds in ids:
        entry = catalog.get(ds)
        if entry is None:
            yield ds, None, None, False
            continue
        target = data_root / entry.get("target", ds)
        yield ds, entry, target, target.exists()


def cmd_check(ids, catalog, data_root) -> int:
    if not ids:
        print("  no external data required (idealized template).")
        return 0
    missing = 0
    for ds, entry, target, present in _status(ids, catalog, data_root):
        if entry is None:
            print(f"  ?? {ds}: NOT in data_catalog.yaml")
            missing += 1
        elif present:
            print(f"  OK {ds}: {target}")
        else:
            print(f"  -- {ds}: MISSING ({target})")
            missing += 1
    print(f"\n  {len(ids) - missing}/{len(ids)} present under {data_root}.")
    return 0 if missing == 0 else 1


def _fetch_one(entry: dict[str, Any], target: Path) -> bool:
    """Best-effort automated fetch; return True on success."""
    url = (entry or {}).get("url", "") or ""
    target.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("gs://") and shutil.which("gsutil"):
        return subprocess.run(["gsutil", "-m", "cp", "-r", url, str(target)]).returncode == 0
    if url.startswith(("http://", "https://")) and shutil.which("curl"):
        return subprocess.run(["curl", "-fSL", "-o", str(target), url]).returncode == 0
    return False


def cmd_fetch(ids, catalog, data_root) -> int:
    if not ids:
        print("  no external data required (idealized template).")
        return 0
    failed = 0
    for ds, entry, target, present in _status(ids, catalog, data_root):
        if present:
            print(f"  OK {ds}: already present ({target})")
            continue
        if entry is None:
            print(f"  ?? {ds}: NOT in data_catalog.yaml — cannot fetch")
            failed += 1
            continue
        print(f"  .. {ds}: fetching -> {target}")
        if _fetch_one(entry, target):
            print(f"  OK {ds}: staged")
        else:
            failed += 1
            cred = entry.get("credentials", "")
            print(f"  !! {ds}: no automatable url; obtain manually -> {target}"
                  + (f"\n       ({cred})" if cred else ""))
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["check", "fetch"])
    p.add_argument("template", help="template id, e.g. coupled/amip")
    p.add_argument("--data-root", default=None,
                   help="override the machine profile's data_root")
    args = p.parse_args(argv)

    ids = _template_data_ids(args.template)
    catalog = _load_catalog()
    data_root = _data_root(args.data_root)
    return (cmd_check if args.action == "check" else cmd_fetch)(ids, catalog, data_root)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Check / fetch the external datasets a template needs — the lesommer
``fetch_data.py`` (``check`` | ``fetch``).

A template's ``experiment.data`` lists dataset ids resolved against
``config/data_catalog.yaml``; each maps to a ``target`` path under the machine's
``data_root``.  Idealized templates list no data and report clean immediately.

    python scripts/experiment/fetch_data.py check 2d/williamson2_sw     # no data needed
    python scripts/experiment/fetch_data.py check 3d_idealized/hydrostatic_gray_1yr       # reports present/missing
    python scripts/experiment/fetch_data.py fetch 3d_idealized/hydrostatic_gray_1yr       # stage missing (where automatable)

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
        from legoesm.experiment_registry import retired_template_message
        retired = retired_template_message(template)
        if retired is not None:
            raise SystemExit(f"ERROR: {retired}")
        raise SystemExit(f"ERROR: template {template!r} not found at {path}")
    doc = yaml.safe_load(path.read_text()) or {}
    return list((doc.get("experiment") or {}).get("data", []) or [])


def _data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(resolve_machine().get("data_root", "./data"))


def _verify(entry: dict[str, Any], target: Path) -> tuple[bool, str]:
    """Is ``target`` present AND intact per the catalog's integrity metadata?

    codex review MEDIUM: bare ``target.exists()`` accepts empty/partial/corrupt
    downloads.  Honor optional ``sha256`` (file checksum) and ``min_bytes`` (file
    size floor); for directory targets (zarr stores / dirs), require a non-empty
    directory (and the ``marker`` file if the catalog names one).
    """
    if not target.exists():
        return False, "missing"
    if target.is_dir():
        # codex review (followup): a non-empty directory is NOT enough — an
        # interrupted zarr/forcing copy leaves a partial tree. Directory datasets
        # MUST declare a completeness `marker` in the catalog (e.g. a zarr's
        # `.zmetadata`, or a `.complete` sentinel written after a verified stage);
        # its presence is what attests the store is whole.
        marker = entry.get("marker")
        if not marker:
            return False, "directory dataset requires a 'marker' in data_catalog.yaml"
        if not (target / marker).exists():
            return False, f"incomplete: marker {marker!r} absent"
        if not any(target.iterdir()):
            return False, "dir present but empty"
        return True, "dir ok"
    sha = entry.get("sha256")
    if sha:
        import hashlib
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h != sha:
            return False, f"sha256 mismatch ({h[:12]}…)"
    min_bytes = entry.get("min_bytes")
    if min_bytes is not None and target.stat().st_size < int(min_bytes):
        return False, f"size {target.stat().st_size} < min_bytes {min_bytes}"
    return True, "ok"


def _status(ids: list[str], catalog: dict[str, Any], data_root: Path):
    """Yield (dataset_id, entry, target_path, present, reason) for each dataset."""
    for ds in ids:
        entry = catalog.get(ds)
        if entry is None:
            yield ds, None, None, False, "not in catalog"
            continue
        target = data_root / entry.get("target", ds)
        present, reason = _verify(entry, target)
        yield ds, entry, target, present, reason


def cmd_check(ids, catalog, data_root) -> int:
    if not ids:
        print("  no external data required (idealized template).")
        return 0
    missing = 0
    for ds, entry, target, present, reason in _status(ids, catalog, data_root):
        if entry is None:
            print(f"  ?? {ds}: NOT in data_catalog.yaml")
            missing += 1
        elif present:
            print(f"  OK {ds}: {target}")
        else:
            print(f"  -- {ds}: MISSING ({target}) [{reason}]")
            missing += 1
    print(f"\n  {len(ids) - missing}/{len(ids)} present under {data_root}.")
    return 0 if missing == 0 else 1


def _fetch_one(entry: dict[str, Any], target: Path) -> bool:
    """Download to a temp path, verify, then atomically move into place.

    codex review MEDIUM: never write directly to the final target — a
    failed/interrupted download must not leave a partial file that future
    ``check`` runs accept. Stage to ``<target>.partial``, verify with
    :func:`_verify`, then ``os.replace``; remove the partial on any failure.
    """
    url = (entry or {}).get("url", "") or ""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    _rm(tmp)
    try:
        if url.startswith("gs://") and shutil.which("gsutil"):
            ok = subprocess.run(["gsutil", "-m", "cp", "-r", url, str(tmp)]).returncode == 0
        elif url.startswith(("http://", "https://")) and shutil.which("curl"):
            ok = subprocess.run(["curl", "-fSL", "-o", str(tmp), url]).returncode == 0
        else:
            return False  # no automatable url
        if not ok or not _verify(entry, tmp)[0]:
            _rm(tmp)
            return False
        import os
        # Replace any stale/partial target (os.replace can't overwrite a
        # non-empty directory) — safe because the staged tmp is already verified.
        _rm(target)
        os.replace(tmp, target)
        return True
    except Exception:  # noqa: BLE001 — never leave a partial behind
        _rm(tmp)
        return False


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def cmd_fetch(ids, catalog, data_root) -> int:
    if not ids:
        print("  no external data required (idealized template).")
        return 0
    failed = 0
    for ds, entry, target, present, reason in _status(ids, catalog, data_root):
        if present:
            print(f"  OK {ds}: already present ({target})")
            continue
        if entry is None:
            print(f"  ?? {ds}: NOT in data_catalog.yaml — cannot fetch")
            failed += 1
            continue
        print(f"  .. {ds}: fetching -> {target} [{reason}]")
        if _fetch_one(entry, target):
            print(f"  OK {ds}: staged + verified")
        else:
            failed += 1
            cred = entry.get("credentials", "")
            print(f"  !! {ds}: no automatable+verifiable url; obtain manually -> {target}"
                  + (f"\n       ({cred})" if cred else ""))
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["check", "fetch"])
    p.add_argument("template", help="template id, e.g. 3d_idealized/hydrostatic_gray_1yr")
    p.add_argument("--data-root", default=None,
                   help="override the machine profile's data_root")
    args = p.parse_args(argv)

    ids = _template_data_ids(args.template)
    catalog = _load_catalog()
    data_root = _data_root(args.data_root)
    return (cmd_check if args.action == "check" else cmd_fetch)(ids, catalog, data_root)


if __name__ == "__main__":
    raise SystemExit(main())

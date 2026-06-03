"""Cache-directory resolution and content-keyed hashing.

Cache root resolution order:

1. ``$LEGOESM_OCEAN_FIDELITY_CACHE`` (explicit per-harness override)
2. ``$LEGOESM_CACHE_DIR`` (shared cache root, e.g. with ERA5 ingest)
3. ``~/.cache/legoesm/ocean_fidelity/``

Subdirectories (created on first use): ``veros/``, ``obs/``, ``regrid_weights/``.

The hashing helpers are content-addressed (sorted-key JSON + blake2s) so
cache keys are stable across dict insertion order and Python versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_PRIMARY_ENV = "LEGOESM_OCEAN_FIDELITY_CACHE"
_FALLBACK_ENV = "LEGOESM_CACHE_DIR"
_DEFAULT_REL = Path(".cache") / "legoesm" / "ocean_fidelity"

ALLOWED_SUBDIRS: tuple[str, ...] = (
    "veros", "obs", "regrid_weights", "forcing",
)


def get_cache_root() -> Path:
    """Return the ocean-fidelity cache root, creating it if absent."""
    explicit = os.environ.get(_PRIMARY_ENV)
    if explicit:
        root = Path(explicit).expanduser()
    else:
        shared = os.environ.get(_FALLBACK_ENV)
        if shared:
            root = Path(shared).expanduser() / "ocean_fidelity"
        else:
            root = Path.home() / _DEFAULT_REL
    root.mkdir(parents=True, exist_ok=True)
    return root


def sub(subdir: str) -> Path:
    """Return a known cache subdirectory, creating it if absent."""
    if subdir not in ALLOWED_SUBDIRS:
        raise ValueError(
            f"Unknown cache subdir {subdir!r}; expected one of {ALLOWED_SUBDIRS}"
        )
    path = get_cache_root() / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_key(payload: Any) -> str:
    """Return a 16-hex blake2s digest of a JSON-serializable payload.

    Sorted keys guarantee dict insertion order does not change the key.
    Non-JSON objects are coerced via ``str(...)`` (``default=str``); pass
    plain dicts / lists / scalars for stable keys.
    """
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=8).hexdigest()


def write_json(path: Path | str, payload: Any) -> None:
    """Atomic JSON write via temp-file + rename in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def read_json(path: Path | str) -> Any:
    """Read a JSON file written by :func:`write_json` (or any standard JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

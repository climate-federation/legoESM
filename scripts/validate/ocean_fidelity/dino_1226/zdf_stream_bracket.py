"""Exact stream gates shared by the day-180 ZDF instrumentation receipts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_byte_identical(left: Path, right: Path) -> bool:
    """Production exact-file predicate: size and SHA256 must both agree."""
    return left.stat().st_size == right.stat().st_size and sha256(left) == sha256(right)


def stream_manifest(directory: Path) -> dict[str, dict[str, int | str]]:
    """Return a canonical name/size/SHA manifest for every raw dump."""
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.glob("*.bin"))
    }


def manifest_sha256(manifest: dict[str, dict[str, int | str]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def one_bit_file_control(path: Path) -> bool:
    """Perturb a copied file by one bit and pass it through the production gate."""
    with tempfile.TemporaryDirectory(prefix="zdf-stream-control-", dir="/tmp") as tmp:
        planted = Path(tmp) / path.name
        shutil.copyfile(path, planted)
        offset = planted.stat().st_size // 2
        with planted.open("r+b") as stream:
            stream.seek(offset)
            original = stream.read(1)
            if len(original) != 1:
                raise RuntimeError(f"cannot plant one-bit stream control in {path}")
            stream.seek(offset)
            stream.write(bytes((original[0] ^ 1,)))
        return not files_byte_identical(path, planted)

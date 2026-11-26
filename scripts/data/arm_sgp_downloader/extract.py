"""Unpack ZIP archives next to the preserved original, refusing paths that leave the target."""
import zipfile
from pathlib import Path


def unpack(archive):
    archive = Path(archive)
    target = archive.with_suffix("")
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (target / name).resolve().is_relative_to(target.resolve()):
                raise ValueError(f"unsafe path in archive: {name}")
        bundle.extractall(target)
    return target

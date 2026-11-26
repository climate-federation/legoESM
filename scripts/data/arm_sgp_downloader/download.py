"""Resumable, checksum-verified, never-overwriting anonymous downloads."""
import concurrent.futures
import hashlib
import contextlib
import os
import re

import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **_):
        return items

from . import config, discovery, metadata

CHUNK = 1 << 20
NETCDF_MAGIC = (b"CDF", b"\x89HDF")


def digest(path, checksum):
    kind, expected = checksum.split(":", 1)
    if kind == "gitblob":
        hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    else:
        hasher = hashlib.new(kind)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest() == expected


def resume_total(reply, offset):
    """Total size from a 206 reply that really starts at offset; 0 if unstated, None if it starts elsewhere."""
    match = re.fullmatch(r"bytes (\d+)-\d+/(\d+|\*)", reply.headers.get("content-range", ""))
    if not match or int(match[1]) != offset:
        return None
    return 0 if match[2] == "*" else int(match[2])


def request_url(entry, credential):
    if entry.get("auth") != "armlive":
        return entry["url"]
    return f"{config.ARMLIVE_URL}saveData?user={credential}&file={entry['filename']}"


def fetch(entry, root, credential=None):
    """Return a catalogue record; an existing original file is never replaced."""
    dest = root / "raw" / entry["source"] / entry["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = {k: entry[k] for k in ("source", "url", "filename", "datastream")}
    if dest.exists():
        if entry["checksum"] and not digest(dest, entry["checksum"]):
            return dict(base, status="conflict", detail="local original differs from source; left untouched")
        return dict(base, status="present", path=str(dest), size=dest.stat().st_size)

    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": config.USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    etag = None
    total = 0
    with requests.get(request_url(entry, credential), headers=headers, stream=True, timeout=config.TIMEOUT_S) as reply:
        if reply.status_code == 416:
            part.unlink()
            return dict(base, status="failed", detail="resume refused; partial file discarded")
        if reply.status_code not in (200, 206):
            return dict(base, status="failed", http_status=reply.status_code)
        if reply.status_code == 206:
            total = resume_total(reply, offset)
            if total is None:
                part.unlink()
                return dict(base, status="failed", detail="resume reply starts elsewhere; partial file discarded")
        else:
            total = int(reply.headers.get("content-length", 0))
        with open(part, "ab" if reply.status_code == 206 else "wb") as handle:
            for block in reply.iter_content(CHUNK):
                handle.write(block)
        if total and part.stat().st_size != total:
            return dict(base, status="failed", detail="short read; partial file kept for resume")
        etag = reply.headers.get("etag")
    if entry.get("auth") == "armlive":
        with open(part, "rb") as handle:
            if not handle.read(4).startswith(NETCDF_MAGIC):
                part.unlink()
                return dict(base, status="failed", detail="reply is not NetCDF; discarded")
    if entry["size"] is not None and part.stat().st_size != entry["size"]:
        return dict(base, status="failed", detail="size mismatch; partial file kept for resume")
    if entry["checksum"] and not digest(part, entry["checksum"]):
        part.unlink()
        return dict(base, status="failed", detail="checksum mismatch; download discarded")
    try:
        os.link(part, dest)  # unlike rename, fails if an original appeared meanwhile
    except FileExistsError:
        part.unlink()
        return dict(base, status="conflict", detail="original appeared during download; left untouched")
    part.unlink()
    os.chmod(dest, 0o444)
    return dict(base, status="downloaded", path=str(dest), size=dest.stat().st_size,
                checksum=entry["checksum"], etag=etag)


def attempt(entry, root, credential):
    try:
        return fetch(entry, root, credential)
    except requests.RequestException as error:
        return {"filename": entry["filename"], "url": entry["url"], "status": "failed", "detail": type(error).__name__}


@contextlib.contextmanager
def only_run(root):
    """Two runs would append to the same partial files."""
    lock = root / ".download.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"another download holds {lock}; delete it only if none is running") from None
    try:
        yield
    finally:
        os.close(handle)
        lock.unlink()


def run(root, entries):
    credential = discovery.armlive_credential()
    records = []
    with only_run(root), concurrent.futures.ThreadPoolExecutor(config.ARMLIVE_WORKERS) as pool:
        jobs = pool.map(lambda entry: attempt(entry, root, credential), entries)
        for record in tqdm(jobs, desc="files", total=len(entries)):
            metadata.append(root, record)
            records.append(record)
    return records
